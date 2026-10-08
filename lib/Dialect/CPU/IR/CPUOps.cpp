#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "Intent/Dialect/CPU/IR/CollectiveHelpers.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;
using namespace intent::cpu;

#define GET_OP_CLASSES
#include "Intent/Dialect/CPU/IR/CPUOps.cpp.inc"

LogicalResult TasksOp::verify() {
  Block &body = getBody().front();
  if (body.empty() || body.getNumArguments() != getCaptures().size() + 1 ||
      !body.getArgument(0).getType().isIndex() ||
      !isa<TaskYieldOp>(body.getTerminator()))
    return emitOpError("tasks require an index coordinate, explicit captures and task_yield");
  for (auto [argument, capture] : llvm::zip(body.getArguments().drop_front(), getCaptures()))
    if (argument.getType() != capture.getType())
      return emitOpError("task capture and body argument types must agree");
  llvm::APInt count;
  if (matchPattern(getCount(), m_ConstantInt(&count)) && count.isNegative())
    return emitOpError("task count cannot be negative");
  if (getOperation()->getParentOfType<TasksOp>() || getOperation()->getParentOfType<TaskDispatchOp>())
    return emitOpError("nested task scheduling has not been realized by CPU partitioning");
  return success();
}

void TasksOp::populateDependencies(
    bufferization::RegisterDependenciesFn registerDependencies) {
  for (auto [capture, argument] :
       llvm::zip_equal(getCaptures(), getBody().front().getArguments().drop_front()))
    if (isa<BaseMemRefType>(capture.getType()))
      registerDependencies(capture, argument);
}

LogicalResult TaskDispatchOp::verify() {
  Block &body = getBody().front();
  if (body.empty() || body.getNumArguments() != 1 ||
      !body.getArgument(0).getType().isIndex() || !isa<TaskYieldOp>(body.getTerminator()))
    return emitOpError("task dispatch requires one coordinate and task_yield");
  llvm::APInt count;
  if (matchPattern(getCount(), m_ConstantInt(&count)) && count.isNegative())
    return emitOpError("task dispatch count cannot be negative");
  llvm::APInt workers;
  if (matchPattern(getWorkerCount(), m_ConstantInt(&workers)) && !workers.isStrictlyPositive())
    return emitOpError("task dispatch worker budget must be positive");
  if (getOperation()->getParentOfType<TasksOp>() || getOperation()->getParentOfType<TaskDispatchOp>())
    return emitOpError("task dispatch cannot introduce nested scheduling");
  return success();
}

LogicalResult TaskYieldOp::verify() {
  if (!isa<TasksOp, TaskDispatchOp>(getOperation()->getParentOp()))
    return emitOpError("must terminate tasks or task_dispatch");
  return success();
}

LogicalResult InvokeOp::verify() {
  if (getAccesses().size() != getArguments().size())
    return emitOpError("requires one explicit storage access mask per argument");
  for (auto [argument, access] : llvm::zip_equal(getArguments(), getAccesses())) {
    if (access < 0 || access > 3)
      return emitOpError("argument access must be none, read, write or read-write");
    if (!isa<BaseMemRefType>(argument.getType()) && access != 0)
      return emitOpError("only storage arguments may carry memory effects");
  }
  return success();
}

LogicalResult InvokeOp::verifySymbolUses(SymbolTableCollection &symbols) {
  auto callee = symbols.lookupNearestSymbolFrom<func::FuncOp>(
      getOperation(), getCalleeAttr());
  if (!callee || !callee.isExternal())
    return emitOpError("requires a declaration for its lowered callable");
  if (!callee.getResultTypes().empty() ||
      !llvm::equal(callee.getArgumentTypes(), getArguments().getTypes()))
    return emitOpError("arguments must exactly match a callable without SSA results");
  return success();
}

void InvokeOp::populateDependencies(bufferization::RegisterDependenciesFn) {}

bool InvokeOp::mayBeTerminalBuffer(Value) { return false; }

void InvokeOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  for (auto [operand, access] :
       llvm::zip_equal(getOperation()->getOpOperands(), getAccesses())) {
    if (access & 1) effects.emplace_back(MemoryEffects::Read::get(), &operand);
    if (access & 2) effects.emplace_back(MemoryEffects::Write::get(), &operand);
  }
}

void ReduceOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  for (OpOperand &operand : getOperation()->getOpOperands())
    if (isa<MemRefType>(operand.get().getType()))
      effects.emplace_back(MemoryEffects::Read::get(), &operand);
}

LogicalResult ReduceOp::verify() {
  if (getInitial().getType() != getResult().getType() || getInputs().empty() ||
      getIndexingMaps().size() != getInputs().size() ||
      !llvm::hasSingleElement(getCombine()))
    return emitOpError("reduction requires one result/identity dtype, input maps and a single combine block");
  Block &block = getCombine().front();
  if (block.empty() || block.getNumArguments() != getInputs().size() + 1 ||
      block.getArgument(0).getType() != getResult().getType())
    return emitOpError("combine arguments must be accumulator followed by input elements");
  for (auto [number, input] : llvm::enumerate(getInputs())) {
    auto memory = dyn_cast<MemRefType>(input.getType());
    Type element = memory ? memory.getElementType() : input.getType();
    auto map = dyn_cast<AffineMapAttr>(getIndexingMaps()[number]);
    if (!isa<FloatType, IntegerType, IndexType>(element) || !map ||
        map.getValue().getNumDims() != 1 || map.getValue().getNumSymbols() ||
        map.getValue().getNumResults() != (memory ? memory.getRank() : 0) ||
        block.getArgument(number + 1).getType() != element)
      return emitOpError("reduction input map or scalar combine type is incomplete");
    for (AffineExpr index : map.getValue().getResults())
      if (!isa<AffineDimExpr>(index) &&
          (!isa<AffineConstantExpr>(index) || cast<AffineConstantExpr>(index).getValue() != 0))
        return emitOpError("reduction maps require the reduction coordinate or a broadcast zero");
  }
  auto yield = dyn_cast<YieldOp>(block.getTerminator());
  if (!yield || yield.getValue().getType() != getResult().getType())
    return emitOpError("combine must yield its accumulator dtype");
  for (Operation &operation : block.without_terminator())
    if (operation.getNumRegions() || !isMemoryEffectFree(&operation) ||
        !llvm::all_of(operation.getOperandTypes(), [](Type type) {
          return isa<FloatType, IntegerType, IndexType>(type);
        }) ||
        !llvm::all_of(operation.getResultTypes(), [](Type type) {
          return isa<FloatType, IntegerType, IndexType>(type);
        }))
      return emitOpError("combine must be a pure scalar expression");
  return success();
}

void ScanOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  unsigned outputBegin = getOperation()->getNumOperands() - getOutputs().size();
  for (OpOperand &operand : getOperation()->getOpOperands()) {
    if (!isa<MemRefType>(operand.get().getType())) continue;
    if (operand.getOperandNumber() >= outputBegin)
      effects.emplace_back(MemoryEffects::Write::get(), &operand);
    else effects.emplace_back(MemoryEffects::Read::get(), &operand);
  }
}

LogicalResult ScanOp::verify() {
  unsigned count = getSources().size();
  if (!count || getInitials().size() != count || getOutputs().size() != count ||
      !llvm::hasSingleElement(getCombine()))
    return emitOpError("scan requires matching sources, identities and destinations");
  if (failed(verifyCollectiveOutputs(getOperation(), getOutputs()))) return failure();
  auto first = cast<ShapedType>(getSources()[0].getType());
  if (getAxis() >= static_cast<uint64_t>(first.getRank()))
    return emitOpError("scan axis is outside its source rank");
  if (getNumResults()) {
    for (auto [source, initial, output] :
         llvm::zip(getSources(), getInitials(), getOutputs())) {
      auto input = dyn_cast<RankedTensorType>(source.getType());
      auto result = cast<RankedTensorType>(output.getType());
      if (!input || getAxis() >= static_cast<uint64_t>(input.getRank()) ||
          input.getDimSize(getAxis()) != first.getDimSize(getAxis()) ||
          input != result)
        return emitOpError("value scan must preserve source shapes and a common member extent");
      SmallVector<int64_t> shape(input.getShape());
      shape.erase(shape.begin() + getAxis());
      if (auto state = dyn_cast<RankedTensorType>(initial.getType())) {
        if (state.getShape() != ArrayRef<int64_t>(shape) ||
            state.getElementType() != input.getElementType())
          return emitOpError("scan identity must describe its complete source slice");
      } else if (initial.getType() != input.getElementType()) {
        return emitOpError("scalar scan identity must match its source element type");
      }
    }
    SmallVector<Type> arguments(getInitials().getTypes());
    llvm::append_range(arguments, getInitials().getTypes());
    llvm::append_range(arguments, getCaptures().getTypes());
    if (getCombine().front().empty() ||
        !isa<ScanYieldOp>(getCombine().front().getTerminator()))
      return emitOpError("scan helper requires scan_yield");
    return verifyValueCollectiveHelper(getOperation(), getCombine(), arguments,
                                      getInitials().getTypes());
  }
  if (isDestinationPassing()) {
    SmallVector<Type> states, members;
    for (auto [source, initial, output] : llvm::zip(getSources(), getInitials(), getOutputs())) {
      auto input = cast<MemRefType>(source.getType()), result = cast<MemRefType>(output.getType());
      auto state = dyn_cast<MemRefType>(initial.getType());
      if (!state) state = MemRefType::get({}, initial.getType());
      if (getAxis() >= static_cast<uint64_t>(input.getRank()) ||
          input.getDimSize(getAxis()) != first.getDimSize(getAxis()) ||
          input.getShape() != result.getShape() || input.getElementType() != result.getElementType() ||
          input.getElementType() != state.getElementType())
        return emitOpError("slice scan sources must agree on their member extent and preserve output types");
      SmallVector<int64_t> shape(input.getShape());
      shape.erase(shape.begin() + getAxis());
      if (ArrayRef<int64_t>(shape) != state.getShape())
        return emitOpError("slice scan identities must describe complete source slices");
      states.push_back(MemRefType::get(shape, state.getElementType()));
      members.push_back(MemRefType::get(shape, state.getElementType(),
          StridedLayoutAttr::get(getContext(), ShapedType::kDynamic,
              SmallVector<int64_t>(shape.size(), ShapedType::kDynamic))));
    }
    SmallVector<Type> arguments(states);
    llvm::append_range(arguments, members);
    llvm::append_range(arguments, getCaptures().getTypes());
    llvm::append_range(arguments, states);
    Block &body = getCombine().front();
    auto yield = body.empty() ? ScanYieldOp() : dyn_cast<ScanYieldOp>(body.getTerminator());
    if (!yield || yield.getNumOperands() || !llvm::equal(body.getArgumentTypes(), arguments))
      return emitOpError("slice scan combine requires incoming states, members, captures and destination slots");
    return verifyCollectiveHelper(getOperation(), getCombine(), count);
  }
  SmallVector<Type> elements;
  for (auto [source, initial, output] : llvm::zip(getSources(), getInitials(), getOutputs())) {
    auto inputType = cast<MemRefType>(source.getType()), outputType = cast<MemRefType>(output.getType());
    if (inputType.getShape() != first.getShape() || outputType.getShape() != inputType.getShape() ||
        inputType.getElementType() != initial.getType() || outputType.getElementType() != initial.getType())
      return emitOpError("scan components must preserve a common shape and their scalar accumulator dtypes");
    elements.push_back(initial.getType());
  }
  SmallVector<Type> arguments(elements);
  llvm::append_range(arguments, elements);
  llvm::append_range(arguments, getCaptures().getTypes());
  Block &body = getCombine().front();
  auto yield = body.empty() ? ScanYieldOp() : dyn_cast<ScanYieldOp>(body.getTerminator());
  if (!yield || !llvm::equal(body.getArgumentTypes(), arguments) || !llvm::equal(yield.getOperandTypes(), elements))
    return emitOpError("scan combine must accept two accumulator tuples and captures, then yield one tuple");
  for (Value capture : getCaptures())
    if (!isa<IntegerType, IndexType, FloatType>(capture.getType()))
      return emitOpError("scalar scan captures must be numeric scalars");
  for (Operation &operation : body.without_terminator())
    if (operation.getNumRegions() || !isMemoryEffectFree(&operation))
      return emitOpError("scalar scan combine must be a closed pure expression");
  return success();
}

LogicalResult HistogramOp::verify() {
  if (failed(verifyCollectiveOutputs(getOperation(), ValueRange{getOutput()})))
    return failure();
  auto values = cast<ShapedType>(getValues().getType());
  auto valid = cast<ShapedType>(getValid().getType());
  auto output = cast<ShapedType>(getOutput().getType());
  if (isa<RankedTensorType>(values) != bool(getResult()) ||
      isa<RankedTensorType>(valid) != bool(getResult()))
    return emitOpError("histogram operands must use one value/buffer form");
  if (!isa<IntegerType, IndexType>(values.getElementType()) || values.getElementType().isInteger(1) ||
      values.getShape() != valid.getShape() || !valid.getElementType().isInteger(1) ||
      output.getRank() != 1 || !isa<IntegerType>(output.getElementType()) || output.getElementType().isInteger(1) ||
      (!output.isDynamicDim(0) && output.getDimSize(0) <= 0))
    return emitOpError("histogram requires integer values, a matching bool predicate and a nonempty integer-bin vector");
  return success();
}

namespace {
void destinationEffects(Operation *operation,
                        SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  for (OpOperand &operand : operation->getOpOperands()) {
    if (!isa<MemRefType>(operand.get().getType())) continue;
    if (operand.getOperandNumber() + 1 == operation->getNumOperands())
      effects.emplace_back(MemoryEffects::Write::get(), &operand);
    else
      effects.emplace_back(MemoryEffects::Read::get(), &operand);
  }
}
} // namespace

void HistogramOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  destinationEffects(getOperation(), effects);
}

void QuantizeOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  destinationEffects(getOperation(), effects);
}

void QuantizedDotOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  destinationEffects(getOperation(), effects);
}

namespace {
LogicalResult verifyAtomicAddress(Operation *operation, Value target, ValueRange indices,
                                  TypeRange operands, Type result = {}) {
  auto memory = cast<MemRefType>(target.getType());
  Type element = memory.getElementType();
  if (indices.size() != static_cast<size_t>(memory.getRank()) ||
      !isa<IntegerType, FloatType>(element) || element.isInteger(1) ||
      llvm::any_of(operands, [&](Type type) { return type != element; }) ||
      (result && result != element))
    return operation->emitOpError("atomic operation requires complete coordinates and matching numeric scalar types");
  return success();
}
}

LogicalResult AtomicLoadOp::verify() {
  if (getOrdering() != intent::AtomicOrdering::Relaxed && getOrdering() != intent::AtomicOrdering::Acquire)
    return emitOpError("atomic load requires relaxed or acquire ordering");
  return verifyAtomicAddress(*this, getTarget(), getIndices(), {}, getValue().getType());
}

LogicalResult AtomicStoreOp::verify() {
  if (getOrdering() != intent::AtomicOrdering::Relaxed && getOrdering() != intent::AtomicOrdering::Release)
    return emitOpError("atomic store requires relaxed or release ordering");
  return verifyAtomicAddress(*this, getTarget(), getIndices(), TypeRange{getValue().getType()});
}

LogicalResult AtomicRMWOp::verify() {
  if (getUnsignedInteger() && !isa<IntegerType>(getValue().getType()))
    return emitOpError("unsigned atomic interpretation requires integer storage");
  if (getKind() == intent::AtomicRMWKind::BitwiseAnd || getKind() == intent::AtomicRMWKind::BitwiseOr ||
      getKind() == intent::AtomicRMWKind::BitwiseXor)
    if (!isa<IntegerType>(getValue().getType())) return emitOpError("bitwise atomic RMW requires an integer");
  return verifyAtomicAddress(*this, getTarget(), getIndices(), TypeRange{getValue().getType()}, getOldValue().getType());
}

LogicalResult AtomicCompareExchangeOp::verify() {
  return verifyAtomicAddress(*this, getTarget(), getIndices(),
      TypeRange{getExpected().getType(), getDesired().getType()}, getOldValue().getType());
}

namespace {
bool recordType(Type type, int64_t bytes) {
  auto memory = dyn_cast<ShapedType>(type);
  return memory && memory.getRank() == 2 &&
      (memory.isDynamicDim(1) || memory.getDimSize(1) == bytes) &&
      memory.getElementType().isSignlessInteger(8);
}
}

LogicalResult QuantizeOp::verify() {
  if (failed(verifyCollectiveOutputs(getOperation(), ValueRange{getOutput()})))
    return failure();
  auto input = cast<ShapedType>(getInput().getType());
  auto output = cast<ShapedType>(getOutput().getType());
  if (isa<RankedTensorType>(input) != bool(getResult()))
    return emitOpError("quantize operands must use one value/buffer form");
  if (getFormat() != intent::QuantFormat::Q8K || input.getRank() != 2 ||
      (!input.isDynamicDim(1) && input.getDimSize(1) != 256) ||
      !input.getElementType().isF32() || !recordType(output, 292) ||
      (!input.isDynamicDim(0) && !output.isDynamicDim(0) &&
       input.getDimSize(0) != output.getDimSize(0)))
    return emitOpError("Q8_K preparation requires f32[G,256] and u8[G,292] storage");
  return success();
}

LogicalResult QuantizedDotOp::verify() {
  if (failed(verifyCollectiveOutputs(getOperation(), ValueRange{getOutput()})))
    return failure();
  auto lhs = cast<ShapedType>(getLhs().getType());
  auto rhs = cast<ShapedType>(getRhs().getType());
  auto output = cast<ShapedType>(getOutput().getType());
  if (isa<RankedTensorType>(lhs) != bool(getResult()) ||
      isa<RankedTensorType>(rhs) != bool(getResult()))
    return emitOpError("quantized dot operands must use one value/buffer form");
  if (getLhsFormat() != intent::QuantFormat::Q4K || getRhsFormat() != intent::QuantFormat::Q8K ||
      !recordType(rhs, 292) || (lhs.getRank() != 2 && lhs.getRank() != 3) ||
      !lhs.getElementType().isSignlessInteger(8) ||
      (!lhs.isDynamicDim(lhs.getRank() - 1) && lhs.getShape().back() != 144) ||
      output.getRank() != lhs.getRank() - 2 || !output.getElementType().isF32())
    return emitOpError("quantized dots require Q4_K [G,144] or [R,G,144], shared Q8_K [G,292], and f32 [] or [R] output");
  unsigned recordAxis = lhs.getRank() - 2;
  if (!lhs.isDynamicDim(recordAxis) && !rhs.isDynamicDim(0) && lhs.getDimSize(recordAxis) != rhs.getDimSize(0))
    return emitOpError("quantized dot record extents must agree");
  if (output.getRank() && !lhs.isDynamicDim(0) && !output.isDynamicDim(0) &&
      lhs.getDimSize(0) != output.getDimSize(0))
    return emitOpError("quantized dot output must preserve the independent row extent");
  return success();
}
