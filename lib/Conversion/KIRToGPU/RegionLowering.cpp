#include "Construction.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include <functional>
#include "Intent/Dialect/GPU/IR/Program.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::kir_to_gpu {

namespace {

bool isShapeMetadataOnly(Operation *operation) {
  auto scalarInteger = [](Type type) { return type.isIntOrIndex(); };
  auto shapeOperand = [](OpOperand &use) {
    Operation *owner = use.getOwner();
    if (!isa<intent::FullOp, intent::BroadcastOp, intent::ReshapeOp>(owner))
      return false;
    auto relation = owner->getAttrOfType<ShapeRelationAttr>("shape");
    return relation && llvm::any_of(relation.getAxes(), [&](Attribute attribute) {
      auto axis = cast<ShapeExprAttr>(attribute);
      return axis.getKind() == 1 &&
             axis.getPayload() == use.getOperandNumber();
    });
  };
  llvm::DenseMap<Value, bool> known;
  llvm::DenseSet<Value> active;
  std::function<bool(Value)> onlyShape = [&](Value value) {
    if (auto found = known.find(value); found != known.end())
      return found->second;
    if (!active.insert(value).second)
      return false;
    bool result = llvm::all_of(value.getUses(), [&](OpOperand &use) {
      if (shapeOperand(use))
        return true;
      Operation *owner = use.getOwner();
      return owner->getNumRegions() == 0 && owner->getNumResults() != 0 &&
             isMemoryEffectFree(owner) &&
             llvm::all_of(owner->getOperandTypes(), scalarInteger) &&
             llvm::all_of(owner->getResultTypes(), scalarInteger) &&
             llvm::all_of(owner->getResults(), onlyShape);
    });
    active.erase(value);
    known[value] = result;
    return result;
  };
  return operation->getNumRegions() == 0 && operation->getNumResults() != 0 &&
         isMemoryEffectFree(operation) &&
         llvm::all_of(operation->getResultTypes(), scalarInteger) &&
         (isa<intent::DimOp>(operation) ||
          llvm::all_of(operation->getOperandTypes(), scalarInteger)) &&
         llvm::all_of(operation->getResults(), onlyShape);
}

} // namespace

ScalarRegionLowering::ScalarRegionLowering(
    OpBuilder &builder,
    llvm::DenseMap<Value, Value> values,
    ArrayRef<Value> views,
    llvm::DenseMap<int64_t, Value> abiDimensions,
    llvm::DenseMap<StringAttr, Value> parameters,
    CanonicalKernelAnalysis &canonicalAnalysis,
    func::FuncOp physicalKernel)
    : builder(builder), values(std::move(values)), views(views),
      abiDimensions(std::move(abiDimensions)), parameters(std::move(parameters)),
      canonicalAnalysis(canonicalAnalysis), physicalKernel(physicalKernel) {}

FailureOr<SmallVector<Value>> ScalarRegionLowering::lowerBlock(Block &source) {
  for (Operation &operation : source) {
    if (auto yield = dyn_cast<intent::YieldOp>(operation)) {
      SmallVector<Value> results;
      for (Value value : yield.getInputs()) {
        FailureOr<Value> lowered = get(value);
        if (failed(lowered))
          return failure();
        results.push_back(*lowered);
      }
      return results;
    }
    if (isa<intent::ReturnOp>(operation))
      return SmallVector<Value>{};
    // Shape-only users consume the canonical relation directly. A numerical
    // use requests the logical runtime value through get(), in its own scope.
    if (isShapeMetadataOnly(&operation))
      continue;
    if (failed(lower(&operation)))
      return failure();
  }
  return SmallVector<Value>{};
}

FailureOr<SmallVector<Value>> ScalarRegionLowering::lowerWorksetBlock(
    Block &source) {
  SmallVector<Operation *> ancestors;
  for (Operation *parent = source.getParentOp();
       parent && !isa<func::FuncOp>(parent); parent = parent->getParentOp())
    ancestors.push_back(parent);
  // Workset extraction must retain the lexical constraints used by its accesses.
  for (Operation *ancestor : llvm::reverse(ancestors))
    for (Operation &operation : *ancestor->getBlock()) {
      if (&operation == ancestor)
        break;
      if (isa<intent::AssumeInBoundsOp>(operation) &&
          failed(lower(&operation)))
        return failure();
    }
  return lowerBlock(source);
}

llvm::DenseMap<Value, Value> &ScalarRegionLowering::mapping() { return values; }

FailureOr<Value> ScalarRegionLowering::lowerValue(Value source) { return get(source); }

FailureOr<Value> ScalarRegionLowering::lowerIndexValue(Value source) {
  FailureOr<Value> value = get(source);
  return succeeded(value) ? asIndex(source.getLoc(), *value)
                          : FailureOr<Value>(failure());
}

FailureOr<Value> ScalarRegionLowering::get(Value source) {
  auto found = values.find(source);
  if (found != values.end())
    return found->second;
  if (Operation *definition = source.getDefiningOp()) {
    if (failed(lower(definition)))
      return failure();
    found = values.find(source);
    if (found != values.end())
      return found->second;
  }
  return failure();
}

void ScalarRegionLowering::mapResults(Operation *source, Operation *target) {
  for (auto [from, to] : llvm::zip(source->getResults(), target->getResults()))
    values[from] = to;
  if (Attribute node = source->getAttr("intent.node"))
    target->setAttr(gpu::originAttr, node);
}

void ScalarRegionLowering::attachOrigin(Operation *source, Operation *target) {
  if (Attribute node = source->getAttr("intent.node"))
    target->setAttr(gpu::originAttr, node);
}

LogicalResult ScalarRegionLowering::lower(Operation *operation) {
  if (operation->getNumResults() > 0) {
    bool complete = llvm::all_of(operation->getResults(),
                                 [&](Value result) { return values.count(result); });
    if (complete)
      return success();
  }
  if (auto constant = dyn_cast<intent::ConstantOp>(operation))
    return lower(constant);
  if (auto dim = dyn_cast<intent::DimOp>(operation))
    return lower(dim);
  if (auto domain = dyn_cast<intent::DomainOp>(operation))
    return lower(domain);
  if (isa<intent::DomainProductOp>(operation))
    return success();
  if (auto subregion = dyn_cast<intent::SubregionOp>(operation))
    return lower(subregion);
  if (auto end = dyn_cast<intent::RegionEndOp>(operation))
    return lower(end);
  if (auto indices = dyn_cast<intent::IndicesOp>(operation))
    return lower(indices);
  if (auto full = dyn_cast<intent::FullOp>(operation))
    return lower(full);
  if (auto broadcastOp = dyn_cast<intent::BroadcastOp>(operation))
    return lower(broadcastOp);
  if (auto reshape = dyn_cast<intent::ReshapeOp>(operation))
    return lower(reshape);
  if (auto transpose = dyn_cast<intent::TransposeOp>(operation))
    return lower(transpose);
  if (auto join = dyn_cast<intent::JoinOp>(operation))
    return lower(join);
  if (auto unary = dyn_cast<intent::UnaryOp>(operation))
    return lower(unary);
  if (auto binary = dyn_cast<intent::BinaryOp>(operation))
    return lower(binary);
  if (auto compare = dyn_cast<intent::CompareOp>(operation))
    return lower(compare);
  if (auto select = dyn_cast<intent::SelectOp>(operation))
    return lower(select);
  if (auto cast = dyn_cast<intent::CastOp>(operation))
    return lower(cast);
  if (auto bitcast = dyn_cast<intent::BitcastOp>(operation))
    return lower(bitcast);
  if (auto mask = dyn_cast<intent::MaskOp>(operation))
    return lower(mask);
  if (auto schema = dyn_cast<StructuredOpInterface>(operation))
    return lowerStructured(schema);
  if (auto contract = dyn_cast<intent::ContractOp>(operation))
    return lower(contract);
  if (auto contract = dyn_cast<intent::ScaledContractOp>(operation))
    return lower(contract);
  if (auto contract = dyn_cast<intent::SparseContractOp>(operation))
    return lower(contract);
  if (auto histogram = dyn_cast<intent::HistogramOp>(operation))
    return lower(histogram);
  if (auto quantize = dyn_cast<intent::QuantizeOp>(operation))
    return lower(quantize);
  if (auto dot = dyn_cast<intent::QuantizedDotOp>(operation))
    return lower(dot);
  if (auto random = dyn_cast<intent::RandomBitsOp>(operation))
    return lower(random);
  if (auto buffer = dyn_cast<intent::BufferOp>(operation))
    return lower(buffer);
  if (auto gather = dyn_cast<intent::GatherOp>(operation))
    return lower(gather);
  if (auto load = dyn_cast<intent::ViewLoadOp>(operation))
    return lower(load);
  if (auto load = dyn_cast<intent::BufferLoadOp>(operation))
    return lower(load);
  if (auto store = dyn_cast<intent::ViewStoreOp>(operation))
    return lower(store);
  if (auto store = dyn_cast<intent::BufferStoreOp>(operation))
    return lower(store);
  if (auto store = dyn_cast<intent::ScatterUniqueOp>(operation))
    return lower(store);
  if (auto scatter = dyn_cast<intent::ScatterReduceOp>(operation))
    return lower(scatter);
  if (auto atomic = dyn_cast<intent::AtomicLoadOp>(operation))
    return lower(atomic);
  if (auto atomic = dyn_cast<intent::AtomicStoreOp>(operation))
    return lower(atomic);
  if (auto atomic = dyn_cast<intent::AtomicRMWOp>(operation))
    return lower(atomic);
  if (auto atomic = dyn_cast<intent::AtomicCompareExchangeOp>(operation))
    return lower(atomic);
  if (auto record = dyn_cast<intent::MakeRecordOp>(operation))
    return lower(record);
  if (auto tuple = dyn_cast<intent::MakeTupleOp>(operation))
    return lower(tuple);
  if (auto extract = dyn_cast<intent::ExtractOp>(operation))
    return lower(extract);
  if (auto ifOperation = dyn_cast<intent::IfOp>(operation))
    return lower(ifOperation);
  if (isa<intent::ForOp, intent::ParallelOp>(operation))
    return lowerLoop(operation);
  if (auto whileOperation = dyn_cast<intent::WhileOp>(operation))
    return lower(whileOperation);
  if (auto assumption = dyn_cast<intent::AssumeInBoundsOp>(operation))
    return lower(assumption);
  return operation->emitOpError(
      "has no scalar shared-GPU construction; tensor/structured families require physical co-realization");
}

} // namespace intent::kir_to_gpu
