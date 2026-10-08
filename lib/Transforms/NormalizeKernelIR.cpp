#include "Intent/Transforms/Passes.h"
#include "Intent/Analysis/ProductSchema.h"
#include "Intent/Interfaces/StructuredOpInterface.h"
#include "Intent/Transforms/PassManager.h"
#include "Intent/Dialect/Intent/IR/IntentAttrs.h"
#include "Intent/Dialect/Intent/IR/IntentDialect.h"
#include "Intent/Dialect/Intent/IR/IntentOps.h"
#include "Intent/Dialect/Intent/IR/IntentTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent {
namespace {

struct Reference {
  Value value;
  SmallVector<unsigned, 4> path;

  bool operator==(const Reference &other) const {
    return value == other.value && path == other.path;
  }
};

using ReferenceMapping = SmallVector<std::pair<Reference, Reference>>;

Reference resolve(Reference reference) {
  while (Operation *operation = reference.value.getDefiningOp()) {
    if (auto extract = dyn_cast<ExtractOp>(operation)) {
      reference.path.insert(reference.path.begin(), extract.getField());
      reference.value = extract.getProduct();
    } else if (!reference.path.empty() &&
               isa<MakeRecordOp, MakeTupleOp>(operation)) {
      reference.value = operation->getOperand(reference.path.front());
      reference.path.erase(reference.path.begin());
    } else {
      break;
    }
  }
  return reference;
}

SmallVector<std::pair<Reference, Type>> leaves(ValueRange values) {
  SmallVector<std::pair<Reference, Type>> result;
  for (Value value : values)
    walkProductLeaves(value.getType(), [&](Type type, ArrayRef<unsigned> path) {
      result.emplace_back(
          resolve({value, SmallVector<unsigned, 4>(path)}), type);
    });
  return result;
}

SmallVector<Reference> flatten(ValueRange values) {
  SmallVector<Reference> result;
  for (auto &leaf : leaves(values))
    result.push_back(std::move(leaf.first));
  return result;
}

Type elementType(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    return tensor.getElementType();
  return type;
}

Value stripScalarBroadcast(Value value) {
  while (Operation *operation = value.getDefiningOp()) {
    if (!isa<BroadcastOp, FullOp>(operation))
      break;
    Value source = operation->getOperand(0);
    if (auto tensor = dyn_cast<RankedTensorType>(source.getType());
        tensor && tensor.getRank() != 0)
      break;
    value = source;
  }
  return value;
}

int64_t integerConstant(IntegerAttr attribute) {
  // Match the integer carrier exposed by the former MLIR Python binding:
  // unsigned attributes zero-extend into it; index/signless/signed ones sign-extend.
  // In particular, signless i1 retains its signed bit interpretation.
  return attribute.getType().isUnsignedInteger()
             ? static_cast<int64_t>(attribute.getUInt())
             : attribute.getValue().getSExtValue();
}

Attribute constantAttribute(Value value) {
  value = stripScalarBroadcast(value);
  Operation *operation = value.getDefiningOp();
  if (!operation)
    return {};
  Attribute source;
  if (auto constant = dyn_cast<ConstantOp>(operation))
    source = constant.getValue();
  else if (isa<CastOp>(operation))
    source = constantAttribute(operation->getOperand(0));
  if (!source)
    return {};
  Type destination = elementType(value.getType());
  if (destination.isF16() || destination.isBF16() || destination.isF32() ||
      destination.isF64()) {
    double number;
    if (auto floating = dyn_cast<FloatAttr>(source))
      number = floating.getValueAsDouble();
    else if (auto integer = dyn_cast<IntegerAttr>(source))
      number = static_cast<double>(integerConstant(integer));
    else
      return {};
    return FloatAttr::get(destination, number);
  }
  if (auto integer = dyn_cast<IntegerType>(destination))
    if (auto attribute = dyn_cast<IntegerAttr>(source))
      return IntegerAttr::get(integer, integerConstant(attribute));
  return {};
}

bool isElementwise(Operation *operation) {
  return isa<ConstantOp, DimOp, BroadcastOp, FullOp, CastOp, BitcastOp,
             UnaryOp, BinaryOp, CompareOp, SelectOp, MaskOp, MakeTupleOp,
             MakeRecordOp, ExtractOp, YieldOp>(operation);
}

DictionaryAttr semanticAttributes(Operation *operation) {
  NamedAttrList attributes(operation->getAttrs());
  for (StringRef name : {"intent.node", "intent.result_nodes",
                         "intent.result_names", "intent.region_argument_nodes",
                         "intent.region_argument_names"})
    attributes.erase(name);
  return attributes.getDictionary(operation->getContext());
}

const Reference *lookup(const ReferenceMapping &mapping,
                        const Reference &reference) {
  for (const auto &[source, target] : mapping)
    if (source == reference)
      return &target;
  return nullptr;
}

bool sameReference(Reference lhs, Reference rhs,
                   const ReferenceMapping &mapping) {
  lhs = resolve(std::move(lhs));
  rhs = resolve(std::move(rhs));
  for (size_t count = lhs.path.size() + 1; count-- > 0;) {
    Reference prefix{lhs.value, {lhs.path.begin(), lhs.path.begin() + count}};
    if (const Reference *target = lookup(mapping, prefix)) {
      Reference mapped = *target;
      mapped.path.append(lhs.path.begin() + count, lhs.path.end());
      return resolve(std::move(mapped)) == rhs;
    }
  }
  if (lhs == rhs)
    return true;
  if (!lhs.path.empty() || !rhs.path.empty())
    return false;
  Attribute leftConstant = constantAttribute(lhs.value);
  Attribute rightConstant = constantAttribute(rhs.value);
  if (leftConstant && rightConstant)
    return leftConstant == rightConstant;
  Value left = stripScalarBroadcast(lhs.value);
  Value right = stripScalarBroadcast(rhs.value);
  if (const Reference *target = lookup(mapping, {left, {}}))
    return *target == Reference{right, {}};
  if (left == right)
    return true;
  Operation *leftOp = left.getDefiningOp();
  Operation *rightOp = right.getDefiningOp();
  if (elementType(left.getType()) != elementType(right.getType()) || !leftOp ||
      !rightOp || !isElementwise(leftOp) ||
      leftOp->getName() != rightOp->getName() ||
      semanticAttributes(leftOp) != semanticAttributes(rightOp) ||
      leftOp->getNumOperands() != rightOp->getNumOperands())
    return false;
  return llvm::all_of(llvm::zip(leftOp->getOperands(), rightOp->getOperands()),
                      [&](auto operands) {
    return sameReference({std::get<0>(operands), {}},
                         {std::get<1>(operands), {}}, mapping);
  });
}

bool sameCombine(Region &lhs, Region &rhs) {
  Block &left = lhs.front();
  Block &right = rhs.front();
  auto leftArguments = leaves(left.getArguments());
  auto rightArguments = leaves(right.getArguments());
  if (leftArguments.size() != rightArguments.size())
    return false;
  ReferenceMapping mapping;
  for (auto [a, b] : llvm::zip(leftArguments, rightArguments)) {
    if (elementType(a.second) != elementType(b.second))
      return false;
    mapping.emplace_back(a.first, b.first);
  }
  auto leftYield = flatten(left.getTerminator()->getOperands());
  auto rightYield = flatten(right.getTerminator()->getOperands());
  if (leftYield.size() != rightYield.size())
    return false;
  for (auto [a, b] : llvm::zip(leftYield, rightYield))
    if (!sameReference(a, b, mapping))
      return false;
  return true;
}

template <typename CollectiveOp>
bool isElementwiseBody(Block &body) {
  for (Operation &operation : body) {
    if (!isElementwise(&operation) && !isa<CollectiveOp>(operation))
      return false;
    if (auto dim = dyn_cast<DimOp>(operation))
      for (OpOperand &use : dim.getResult().getUses())
        if (!isa<BroadcastOp, FullOp>(use.getOwner()) ||
            use.getOperandNumber() == 0)
          return false;
  }
  return true;
}

ReduceOp summaryReduction(Operation *operation) {
  auto schema = cast<StructuredOpInterface>(operation);
  Block &body = schema.getSummarizeRegion()->front();
  auto reductions = body.getOps<ReduceOp>();
  if (!llvm::hasSingleElement(reductions) ||
      !isElementwiseBody<ReduceOp>(body))
    return {};
  ReduceOp reduction = *reductions.begin();
  if (flatten(body.getTerminator()->getOperands()) !=
      flatten(reduction.getResults()))
    return {};
  ArrayAttr axes = reduction.getAxes();
  if (axes.size() != 1 || cast<IntegerAttr>(axes[0]).getInt() != schema.getIterationAxes().front() ||
      !sameCombine(schema.getCombine(), reduction.getCombine()))
    return {};
  ReferenceMapping mapping;
  for (const auto &relation : schema.getValueRelations())
    if (auto argument = dyn_cast<BlockArgument>(relation.to); argument && argument.getOwner() == &body &&
        (relation.kind == StructuredRelationKind::SourceSlice ||
         relation.kind == StructuredRelationKind::Capture))
      mapping.push_back({{relation.to, {}}, {relation.from, {}}});
  auto lhs = flatten(reduction.getIdentities());
  auto rhs = flatten(schema.getIdentities());
  if (lhs.size() != rhs.size())
    return {};
  for (auto [a, b] : llvm::zip(lhs, rhs))
    if (!sameReference(a, b, mapping))
      return {};
  return reduction;
}

bool hasElementScan(RegionScanOp operation, ReduceOp reduction) {
  Block &body = operation.getEmit().front();
  auto scans = body.getOps<ScanOp>();
  if (!llvm::hasSingleElement(scans) || !isElementwiseBody<ScanOp>(body))
    return false;
  ScanOp scan = *scans.begin();
  if (scan.getAxis() != operation.getAxis() || !scan.getInclusive() ||
      scan.getReverse() || !sameCombine(operation.getCombine(), scan.getCombine()))
    return false;
  auto schema = cast<StructuredOpInterface>(operation.getOperation());
  ReferenceMapping mapping;
  for (auto [argument, value] : llvm::zip(schema.getSummarizeSources(), schema.getEmitSources()))
    mapping.push_back({{argument, {}}, {value, {}}});
  for (auto [argument, value] : llvm::zip(schema.getSummarizeCaptures(), schema.getEmitCaptures()))
    mapping.push_back({{argument, {}}, {value, {}}});
  if (reduction->getNumOperands() != scan->getNumOperands())
    return false;
  for (auto [a, b] : llvm::zip(reduction.getOperands(), scan.getOperands()))
    if (!sameReference({a, {}}, {b, {}}, mapping))
      return false;
  return llvm::all_of(operation.getApply().front(),
                      [](Operation &nested) { return isElementwise(&nested); });
}

struct DimensionSubstitution {
  int64_t source;
  int64_t target;
  int64_t extent;

  Type remap(Type type) const {
    MLIRContext *context = type.getContext();
    if (auto tensor = dyn_cast<RankedTensorType>(type)) {
      auto encoding = cast<TensorShapeAttr>(tensor.getEncoding());
      SmallVector<int64_t> dimensions(encoding.getDimensions().asArrayRef());
      SmallVector<int64_t> shape(tensor.getShape());
      for (auto [axis, dimension] : llvm::enumerate(dimensions))
        if (dimension == source) {
          dimensions[axis] = target;
          shape[axis] = extent;
        }
      return RankedTensorType::get(
          shape, tensor.getElementType(),
          TensorShapeAttr::get(context, DenseI64ArrayAttr::get(context, dimensions)));
    }
    if (auto fields = getProductComponents(type)) {
      SmallVector<Attribute> remapped;
      for (Attribute field : fields)
        remapped.push_back(TypeAttr::get(remap(cast<TypeAttr>(field).getValue())));
      if (auto record = dyn_cast<RecordType>(type))
        return RecordType::get(context, record.getFieldNames(),
                               ArrayAttr::get(context, remapped));
      return intent::TupleType::get(context, ArrayAttr::get(context, remapped));
    }
    return type;
  }

  void remap(Operation *operation) const {
    operation->walk([&](Operation *nested) {
      for (Value result : nested->getResults())
        result.setType(remap(result.getType()));
      for (Region &region : nested->getRegions())
        for (Block &block : region)
          for (BlockArgument argument : block.getArguments())
            argument.setType(remap(argument.getType()));
      if (auto dim = dyn_cast<DimOp>(nested);
          dim && dim.getDimension() == static_cast<uint64_t>(source))
        dim.setDimension(target);
      if (auto shape = nested->getAttrOfType<ShapeRelationAttr>("shape")) {
        SmallVector<Attribute> axes;
        for (Attribute attribute : shape.getAxes()) {
          auto axis = cast<ShapeExprAttr>(attribute);
          axes.push_back(ShapeExprAttr::get(
              nested->getContext(), axis.getKind(),
              axis.getDimension() == source ? target : axis.getDimension(),
              axis.getPayload()));
        }
        nested->setAttr("shape", ShapeRelationAttr::get(
                                    nested->getContext(),
                                    ArrayAttr::get(nested->getContext(), axes)));
      }
    });
  }
};

SmallVector<Value> cloneBody(Region &region, IRMapping &mapping,
                             Operation *owner,
                             const DimensionSubstitution &dimensions) {
  Block &body = region.front();
  OpBuilder builder(owner);
  for (Operation &original : body.without_terminator()) {
    Operation *cloned = builder.clone(original, mapping);
    dimensions.remap(cloned);
    if (auto extract = dyn_cast<ExtractOp>(cloned)) {
      Reference resolved = resolve({extract.getResult(), {}});
      if (resolved.path.empty() && resolved.value != extract.getResult()) {
        extract.getResult().replaceAllUsesWith(resolved.value);
        mapping.map(original.getResult(0), resolved.value);
        cloned->erase();
      }
    }
  }
  SmallVector<Value> results;
  for (Value value : body.getTerminator()->getOperands())
    results.push_back(mapping.lookupOrDefault(value));
  return results;
}

void normalizeRegions(ModuleOp module) {
  SmallVector<Operation *> regions;
  module.walk<WalkOrder::PostOrder>([&](Operation *operation) {
    if (isa<RegionFoldOp, RegionScanOp>(operation))
      regions.push_back(operation);
  });
  for (Operation *operation : regions) {
    ReduceOp reduction = summaryReduction(operation);
    if (!reduction)
      continue;
    auto scan = dyn_cast<RegionScanOp>(operation);
    if (scan && !hasElementScan(scan, reduction))
      continue;
    auto schema = cast<StructuredOpInterface>(operation);
    auto inputs = schema.getSources();
    auto initial = schema.getInitialStates();
    auto captures = schema.getCaptures();
    int64_t axis = schema.getIterationAxes().front();
    auto sourceType = cast<RankedTensorType>(inputs.front().getType());
    auto sliceType = cast<RankedTensorType>(schema.getSummarizeSources().front().getType());
    DimensionSubstitution dimensions{
        cast<TensorShapeAttr>(sliceType.getEncoding()).getDimensions()[axis],
        cast<TensorShapeAttr>(sourceType.getEncoding()).getDimensions()[axis],
        sourceType.getDimSize(axis)};
    IRMapping summarizeMapping;
    summarizeMapping.map(schema.getSummarizeSources(), inputs);
    summarizeMapping.map(schema.getSummarizeCaptures(), captures);
    SmallVector<Value> summary = cloneBody(*schema.getSummarizeRegion(),
                                           summarizeMapping, operation, dimensions);
    SmallVector<Value> replacement;
    if (scan) {
      IRMapping emitMapping;
      emitMapping.map(schema.getEmitSources(), inputs);
      emitMapping.map(schema.getEmitStates(), initial);
      emitMapping.map(schema.getEmitCaptures(), captures);
      replacement = cloneBody(scan.getEmit(), emitMapping, operation, dimensions);
      IRMapping applyMapping;
      applyMapping.map(schema.getApplySummaries(), summary);
      applyMapping.map(schema.getApplyStates(), initial);
      llvm::append_range(replacement,
                         cloneBody(scan.getApply(), applyMapping, operation, dimensions));
    } else {
      replacement = std::move(summary);
    }
    operation->replaceAllUsesWith(replacement);
    // Each region body is cloned once; removing its owner retires the original
    // provenance IDs. Clones keep the author's locations and their existing IDs.
    operation->erase();
  }
}

class NormalizeKernelIRPass
    : public PassWrapper<NormalizeKernelIRPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NormalizeKernelIRPass)

  StringRef getArgument() const final { return "intent-normalize-kernel"; }
  StringRef getDescription() const final {
    return "Normalize verified Intent source regions into canonical Kernel IR";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IntentDialect, func::FuncDialect>();
  }
  void runOnOperation() final {
    ModuleOp module = getOperation();
    if (failed(verifyKernelStructure(module))) {
      signalPassFailure();
      return;
    }
    normalizeRegions(module);
    if (failed(verifyKernelModule(module)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createNormalizeKernelIRPass() {
  return std::make_unique<NormalizeKernelIRPass>();
}

LogicalResult normalizeKernelModule(ModuleOp module) {
  PassManager manager(module.getContext(), ModuleOp::getOperationName());
  manager.addPass(createNormalizeKernelIRPass());
  if (failed(configurePassManager(manager)))
    return failure();
  return manager.run(module);
}

} // namespace intent
