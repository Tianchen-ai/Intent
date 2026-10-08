#include "Construction.h"
#include "Intent/Dialect/Intent/IR/Interface.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace intent::kir_to_cpu {

Construction::Construction(CanonicalKernelAnalysis &analysis, ModuleOp physical,
                           CPUEntryLayout entryLayout)
    : analysis(analysis), module(physical), builder(physical.getContext()),
      entryLayout(entryLayout) {}

LogicalResult Construction::lower(func::FuncOp source) {
  SmallVector<Type> types;
  auto interface = buildPublicInterface(source);
  if (failed(interface)) return failure();
  SmallVector<Value> runtimeArguments;
  for (BlockArgument argument : source.getArguments()) {
    if (isa<ConstexprType>(argument.getType())) {
      if (!argument.use_empty())
        return source.emitError("CPU physical ABI requires fully specialized constexpr parameters");
      continue;
    }
    runtimeArguments.push_back(argument);
    if (auto view = dyn_cast<ViewType>(argument.getType())) {
      auto tensor = cast<RankedTensorType>(view.getTensor());
      Type element = tensor.getElementType();
      if ((!element.isF16() && !element.isBF16() && !element.isF32() && !element.isF64() &&
           !isa<Float8E4M3FNType, Float8E5M2Type>(element) &&
           !element.isInteger(8) && !element.isInteger(16) && !element.isInteger(32) &&
           !element.isInteger(64) && !element.isInteger(1)))
        return source.emitError("CPU construction requires supported numeric views");
      auto memory = MemRefType::get(tensor.getShape(), tensor.getElementType());
      SmallVector<Attribute> constraints(tensor.getRank(), builder.getUnitAttr());
      if (view.getConstraints().getHasStrides()) {
        if (view.getConstraints().getStrides().size() != static_cast<size_t>(tensor.getRank()))
          return source.emitError("CPU declared stride constraints must cover the view rank");
        constraints.assign(view.getConstraints().getStrides().begin(), view.getConstraints().getStrides().end());
      }
      for (Attribute constraint : constraints)
        if (!isa<UnitAttr, IntegerAttr>(constraint))
          return source.emitError("CPU symbolic stride constraints are not implemented");
      if (entryLayout == CPUEntryLayout::StridedInputs && view.getAccess() == 0) {
        SmallVector<int64_t> strides;
        for (Attribute constraint : constraints) {
          auto fixed = dyn_cast<IntegerAttr>(constraint);
          strides.push_back(fixed ? fixed.getInt() : ShapedType::kDynamic);
        }
        memory = MemRefType::get(tensor.getShape(), tensor.getElementType(),
            StridedLayoutAttr::get(builder.getContext(), 0, strides));
      } else {
        SmallVector<int64_t> strides;
        int64_t offset;
        if (failed(memory.getStridesAndOffset(strides, offset)))
          return source.emitError("CPU contiguous view has no derived strides");
        for (auto [constraint, stride] : llvm::zip(constraints, strides)) {
          if (isa<UnitAttr>(constraint)) continue;
          auto fixed = dyn_cast<IntegerAttr>(constraint);
          if (!fixed || (!ShapedType::isDynamic(stride) && fixed.getInt() != stride))
            return source.emitError("CPU contiguous ABI cannot discharge this declared stride constraint");
        }
      }
      auto shape = dyn_cast_or_null<TensorShapeAttr>(tensor.getEncoding());
      if (!shape)
        return source.emitError("CPU view is missing canonical dimension identities");
      types.push_back(memory);
    } else if (argument.getType().isF32() || argument.getType().isF64() || argument.getType().isIndex() ||
               argument.getType().isInteger(8) || argument.getType().isInteger(16) ||
               argument.getType().isInteger(32) || argument.getType().isInteger(64) ||
               argument.getType().isInteger(1)) {
      types.push_back(argument.getType());
    } else {
      return source.emitError("CPU construction does not implement this parameter type");
    }
  }
  builder.setInsertionPointToEnd(module.getBody());
  function = builder.create<func::FuncOp>(source.getLoc(), source.getName(),
                                         builder.getFunctionType(types, {}));
  function->setAttr(interfaceAttr, *interface);
  function->setAttr(cpu::entryRequirementsAttr, cpu::EntryRequirementsAttr::get(
      builder.getContext(), entryLayout == CPUEntryLayout::Contiguous, true));
  function.addEntryBlock();
  builder.setInsertionPointToStart(&function.front());
  for (auto [oldValue, newValue] : llvm::zip(runtimeArguments, function.getArguments()))
    values.map(oldValue, newValue);
  if (failed(lowerBlock(source.front())))
    return failure();
  builder.create<func::ReturnOp>(source.getLoc());
  return success();
}

LogicalResult Construction::lowerBlock(Block &block) {
  for (Operation &operation : block.without_terminator())
    if (failed(lowerOperation(&operation))) return failure();
  // Only author-declared mutable buffers have lexical lifetime in this
  // construction. Storage for immutable values belongs to bufferization.
  for (auto buffer : block.getOps<BufferOp>())
    builder.create<memref::DeallocOp>(buffer.getLoc(), values.lookup(buffer.getResult()));
  return success();
}

LogicalResult Construction::lowerOperation(Operation *operation) {
  return llvm::TypeSwitch<Operation *, LogicalResult>(operation)
      .Case<ConstantOp, DimOp, DomainOp, SubregionOp, RegionEndOp, ParallelOp,
            BufferOp, FullOp, IndicesOp, JoinOp, BroadcastOp, MakeRecordOp,
            MakeTupleOp, ExtractOp, HistogramOp, QuantizeOp, QuantizedDotOp>(
          [&](auto op) { return lower(op); })
      .Case<IfOp, ForOp, WhileOp>(
          [&](auto op) { return orderedControl(op); })
      .Case<AssumeInBoundsOp>([](auto) { return success(); })
      .Case<AtomicLoadOp, AtomicStoreOp, AtomicRMWOp, AtomicCompareExchangeOp>(
          [&](auto op) { return atomicAccess(op); })
      .Case<ScatterReduceOp>([&](auto op) { return scatterReduce(op); })
      .Case<ViewLoadOp, BufferLoadOp, GatherOp>(
          [&](auto op) { return load(op); })
      .Case<ViewStoreOp, BufferStoreOp, ScatterUniqueOp>(
          [&](auto op) { return store(op); })
      .Case<BinaryOp, UnaryOp, CompareOp, SelectOp, CastOp, BitcastOp, MaskOp,
            RandomBitsOp>([&](auto op) { return pointwise(op); })
      .Case<TransposeOp, ReshapeOp>(
          [&](auto op) { return tensorShape(op); })
      .Case<RegionFoldOp, RegionScanOp>(
          [&](auto op) { return region(op); })
      .Case<ScanOp>([&](auto op) { return scan(op); })
      .Case<ReduceOp>([&](auto op) { return reduce(op); })
      .Case<ContractOp>([&](auto op) { return contract(op); })
      .Case<ScaledContractOp>([&](auto op) { return scaledContract(op); })
      .Case<SparseContractOp>([&](auto op) { return sparseContract(op); })
      .Default([&](Operation *op) -> LogicalResult {
        if (op->getName().getDialectNamespace() == "arith") {
          builder.clone(*op, values);
          return success();
        }
        return op->emitError(
            "CPU construction does not implement this canonical operation");
      });
}

} // namespace intent::kir_to_cpu
