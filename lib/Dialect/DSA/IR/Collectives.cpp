#include "Intent/Dialect/DSA/IR/DSAOps.h"
#include "Intent/Dialect/DSA/IR/Views.h"
#include "Intent/Analysis/BufferStorage.h"
#include "Intent/Dialect/DSA/IR/MemoryEffects.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::dsa {
namespace {

bool scalar(Type type) {
  return isa<IntegerType, IndexType, FloatType>(type);
}

bool scalarMatches(Type type, Type element) {
  return type == element || (type.isIndex() && element.isInteger(64));
}

bool sameExtent(Value lhs, Value rhs) {
  if (lhs == rhs) return true;
  APInt left, right;
  return matchPattern(lhs, m_ConstantInt(&left)) &&
         matchPattern(rhs, m_ConstantInt(&right)) && left == right;
}

bool stateView(Type type, ArrayRef<int64_t> shape, Type element) {
  auto memory = dyn_cast<MemRefType>(type);
  return memory && memory.hasStaticShape() && memory.getShape() == shape &&
         memory.getElementType() == element &&
         memory.getLayout().isIdentity() &&
         memory.getMemorySpaceAsInt() == nramSpace;
}

LogicalResult verifyHelper(Operation *owner, Region &region,
                           ValueRange captures, unsigned fields) {
  Block &body = region.front();
  auto yielded = dyn_cast<CollectiveYieldOp>(body.getTerminator());
  if (!yielded || yielded.getValues().size() != fields ||
      body.getNumArguments() != 2 * fields + captures.size())
    return owner->emitOpError(
        "combine requires two component tuples, explicit captures and one yielded tuple");
  for (unsigned field = 0; field < fields; ++field)
    if (body.getArgument(field).getType() !=
            body.getArgument(fields + field).getType() ||
        body.getArgument(field).getType() != yielded.getValues()[field].getType())
      return owner->emitOpError("combine component arguments and yields must agree");
  if (!llvm::equal(llvm::drop_begin(body.getArgumentTypes(), 2 * fields),
                   captures.getTypes()))
    return owner->emitOpError("combine capture types must match explicit operands");

  BufferStorageAnalysis storage(owner->getParentOfType<func::FuncOp>(),
                                storagePolicy());
  SmallVector<Value> borrowed;
  for (BlockArgument argument : body.getArguments())
    if (isa<BaseMemRefType>(argument.getType())) {
      auto origins = storage.origins(argument);
      if (!origins.complete)
        return owner->emitOpError("collective argument has unknown storage flow");
      llvm::append_range(borrowed, origins.values);
    }
  auto knownLocal = [&](Value memory) {
    auto origins = storage.origins(memory);
    return origins.complete && !origins.values.empty() &&
           llvm::all_of(origins.values, [&](Value origin) {
             if (llvm::is_contained(borrowed, origin))
               return true;
             auto allocation = origin.getDefiningOp<memref::AllocaOp>();
             return allocation &&
                    region.isAncestor(allocation->getParentRegion());
           });
  };
  WalkResult status = region.walk<WalkOrder::PreOrder>([&](Operation *operation) {
    for (Value operand : operation->getOperands())
      if (!region.isAncestor(operand.getParentRegion())) {
        operation->emitOpError("collective helper requires explicit captures");
        return WalkResult::interrupt();
      }
    if (isa<CollectiveYieldOp>(operation)) return WalkResult::advance();
    SmallVector<MemoryEffects::EffectInstance> effects;
    if (auto interface = dyn_cast<MemoryEffectOpInterface>(operation))
      interface.getEffects(effects);
    else if (operation->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
      return WalkResult::advance();
    else if (!isMemoryEffectFree(operation)) {
      operation->emitOpError("collective helper requires known local effects");
      return WalkResult::interrupt();
    }
    for (const auto &effect : effects) {
      Value memory = effect.getValue();
      // Some native tile operations accept either a memory operand or a uniform
      // scalar in the same operand slot. A scalar read is only an SSA dependency.
      if (memory && scalar(memory.getType()) &&
          isa<MemoryEffects::Read>(effect.getEffect()))
        continue;
      if (!memory || !isa<MemRefType>(memory.getType()) || !knownLocal(memory)) {
        operation->emitOpError(
            "collective helper effects must target borrowed inputs or local scratch");
        return WalkResult::interrupt();
      }
      if (isa<MemoryEffects::Read>(effect.getEffect())) continue;
      auto origins = storage.origins(memory);
      if (!isa<MemoryEffects::Allocate, MemoryEffects::Write>(effect.getEffect()) ||
          !llvm::all_of(origins.values, [&](Value origin) {
            auto allocation = origin.getDefiningOp<memref::AllocaOp>();
            return allocation && region.isAncestor(allocation->getParentRegion());
          })) {
        operation->emitOpError(
            "collective inputs are borrowed read-only; writes require local scratch");
        return WalkResult::interrupt();
      }
    }
    // Nested helpers validate their own formals. Their parent effects describe
    // precisely the actual values used by this invocation.
    return isa<SliceReduceOp, ScanOp>(operation) ? WalkResult::skip()
                                               : WalkResult::advance();
  });
  if (status.wasInterrupted()) return failure();
  for (Value value : yielded.getValues())
    if (isa<MemRefType>(value.getType()) && !knownLocal(value))
      return owner->emitOpError(
          "yielded state must borrow a helper input or use helper-local scratch");
  return success();
}

LogicalResult verifyCollective(Operation *operation, ValueRange sources,
                               ValueRange initials, ValueRange captures,
                               ValueRange outputs, ValueRange counts,
                               ValueRange finals, ArrayRef<int64_t> axes,
                               Region &combine, bool scan) {
  unsigned fields = sources.size();
  if (!fields || initials.size() != fields || outputs.size() != fields ||
      (scan && finals.size() != fields) || axes.empty() ||
      !llvm::hasSingleElement(combine) || combine.front().empty())
    return operation->emitOpError(
        "collective requires paired sources, initials, outputs and a nonempty combine block");
  SmallVector<int64_t> seen;
  for (int64_t axis : axes) {
    if (axis < 0 || llvm::is_contained(seen, axis))
      return operation->emitOpError("member axes must be distinct and nonnegative");
    seen.push_back(axis);
  }
  unsigned countArity = 0;
  for (Value source : sources) {
    if (!isCompleteLocalStorageView(source))
      return operation->emitOpError(
          "sources require complete contiguous local capacity views");
    countArity += cast<MemRefType>(source.getType()).getRank();
  }
  if (counts.size() != countArity)
    return operation->emitOpError("active counts must supply each axis of every source");
  if (failed(verifyHelper(operation, combine, captures, fields))) return failure();

  Block &body = combine.front();
  bool scalarHelper = llvm::all_of(body.getArguments().take_front(fields),
      [](BlockArgument argument) { return scalar(argument.getType()); });
  SmallVector<ValueRange> fieldCounts;
  unsigned offset = 0;
  SmallVector<SmallVector<int64_t>> stateShapes;
  SmallVector<SmallVector<Value>> stateCounts;
  for (unsigned field = 0; field < fields; ++field) {
    auto source = cast<MemRefType>(sources[field].getType());
    auto active = counts.slice(offset, source.getRank());
    offset += source.getRank();
    fieldCounts.push_back(active);
    for (int64_t axis : axes)
      if (axis >= source.getRank())
        return operation->emitOpError("member axis is outside a source rank");
    for (auto [axis, count] : llvm::enumerate(active)) {
      APInt constant;
      if (matchPattern(count, m_ConstantInt(&constant)) &&
          (constant.isNegative() ||
           constant.getSExtValue() > source.getDimSize(axis)))
        return operation->emitOpError("active extent exceeds its source capacity");
    }
    for (int64_t axis : axes)
      if (field && !sameExtent(active[axis], fieldCounts.front()[axis]))
        return operation->emitOpError("source fields must share actual member extents");
    SmallVector<int64_t> shape;
    SmallVector<Value> activeState;
    for (int64_t axis = 0; axis < source.getRank(); ++axis)
      if (!llvm::is_contained(axes, axis)) {
        shape.push_back(source.getDimSize(axis));
        activeState.push_back(active[axis]);
      }
    stateShapes.push_back(shape);
    stateCounts.push_back(activeState);
    Type element = source.getElementType();
    if (!isCompleteLocalStorageView(outputs[field]) ||
        !stateView(outputs[field].getType(), scan ? source.getShape() : ArrayRef<int64_t>(shape), element))
      return operation->emitOpError("output must preserve its component's free or prefix capacity axes");
    if (scan && (!isCompleteLocalStorageView(finals[field]) ||
                 !stateView(finals[field].getType(), shape, element)))
      return operation->emitOpError("scan final state must remove its component's member axis");
    Type initial = initials[field].getType();
    if (!scalarMatches(initial, element) &&
        !(isCompleteLocalStorageView(initials[field]) && stateView(initial, shape, element)))
      return operation->emitOpError("initial must be a scalar fill or a complete component state view");
    Type formal = body.getArgument(field).getType();
    if (scalar(formal)) {
      if (!scalarMatches(formal, element) || (!scalarHelper && !shape.empty()))
        return operation->emitOpError(
            "scalar combine fields require the common pointwise domain or a rank-zero state");
    } else if (!stateView(formal, shape, element)) {
      return operation->emitOpError("combine field must preserve its complete source member slice");
    }
  }
  if (scalarHelper)
    for (unsigned field = 1; field < fields; ++field) {
      if (stateShapes[field] != stateShapes.front() ||
          !llvm::equal(stateCounts[field], stateCounts.front(), sameExtent))
        return operation->emitOpError(
            "joint scalar combine fields require the same active free-coordinate domain");
    }
  for (Value capture : captures)
    if (!scalar(capture.getType()) && !isCompleteLocalStorageView(capture))
      return operation->emitOpError("captures must be scalar values or complete local snapshots");

  BufferStorageAnalysis storage(operation->getParentOfType<func::FuncOp>(),
                                storagePolicy());
  SmallVector<Value> readValues, writtenValues;
  for (ValueRange inputs : {sources, captures})
    for (Value input : inputs)
      if (isa<MemRefType>(input.getType())) readValues.push_back(input);
  auto verifyDestinations = [&](ValueRange destinations,
                                bool updatesInitial) -> LogicalResult {
    for (auto [field, destination] : llvm::enumerate(destinations)) {
      Value root = storage.uniqueOrigin(destination);
      auto overlaps = [&](Value input) { return !storage.disjoint(destination, input); };
      if (!root || !root.getDefiningOp<memref::AllocaOp>() ||
          llvm::any_of(readValues, overlaps) || llvm::any_of(writtenValues, overlaps))
        return operation->emitOpError(
            "collective outputs require independent local storage");
      for (auto [initialField, initial] : llvm::enumerate(initials))
        if (isa<MemRefType>(initial.getType()) && !storage.disjoint(initial, destination) &&
            !(updatesInitial && initialField == field &&
              initial.getType() == destination.getType() &&
              isCompleteStorageViewOf(initial, root) &&
              isCompleteStorageViewOf(destination, root)))
          return operation->emitOpError(
              "only a component's complete final state may replace its own initial storage");
      writtenValues.push_back(destination);
    }
    return success();
  };
  // Chunked reduction/scan may commit a completed component back to its seed.
  // Its combine still reads an independent snapshot before any final commit.
  if (failed(verifyDestinations(outputs, !scan)) ||
      failed(verifyDestinations(finals, true)))
    return failure();
  return success();
}

void collectiveEffects(Operation *operation, unsigned outputBegin,
                       unsigned outputCount, unsigned finalsBegin,
                       SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  for (OpOperand &operand : operation->getOpOperands()) {
    if (!isa<MemRefType>(operand.get().getType())) continue;
    unsigned index = operand.getOperandNumber();
    if ((index >= outputBegin && index < outputBegin + outputCount) ||
        index >= finalsBegin)
      effects.emplace_back(MemoryEffects::Write::get(), &operand);
    else
      effects.emplace_back(MemoryEffects::Read::get(), &operand);
  }
}

} // namespace

bool isCollectiveBorrowedArgument(BlockArgument argument) {
  return isa<SliceReduceOp, ScanOp>(argument.getOwner()->getParentOp()) &&
         isa<MemRefType>(argument.getType());
}

namespace {
template <typename Collective>
Value capturedBuffer(Collective operation, Value value) {
  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument || argument.getOwner() != &operation.getCombine().front() ||
      !isa<BaseMemRefType>(argument.getType()))
    return {};
  unsigned first = 2 * operation.getSources().size();
  unsigned index = argument.getArgNumber();
  if (index < first || index - first >= operation.getCaptures().size())
    return {};
  return operation.getCaptures()[index - first];
}

template <typename Collective>
void collectiveFlow(Collective operation,
                    bufferization::RegisterDependenciesFn registerDependencies) {
  if (operation.getCombine().empty())
    return;
  for (BlockArgument argument : operation.getCombine().front().getArguments())
    if (Value capture = capturedBuffer(operation, argument))
      registerDependencies(capture, argument);
}
} // namespace

void SliceReduceOp::populateDependencies(
    bufferization::RegisterDependenciesFn registerDependencies) {
  collectiveFlow(*this, registerDependencies);
}

bool SliceReduceOp::mayBeTerminalBuffer(Value value) {
  return !capturedBuffer(*this, value);
}

void ScanOp::populateDependencies(
    bufferization::RegisterDependenciesFn registerDependencies) {
  collectiveFlow(*this, registerDependencies);
}

bool ScanOp::mayBeTerminalBuffer(Value value) {
  return !capturedBuffer(*this, value);
}

LogicalResult SliceReduceOp::verify() {
  return verifyCollective(getOperation(), getSources(), getInitials(), getCaptures(),
                          getOutputs(), getCounts(), {}, getAxes(), getCombine(), false);
}

LogicalResult ScanOp::verify() {
  return verifyCollective(getOperation(), getSources(), getInitials(), getCaptures(),
                          getOutputs(), getCounts(), getFinals(),
                          {static_cast<int64_t>(getAxis())}, getCombine(), true);
}

LogicalResult CollectiveYieldOp::verify() {
  return isa<SliceReduceOp, ScanOp>((*this)->getParentOp())
             ? success()
             : emitOpError("requires an ordinary DSA collective parent");
}

void SliceReduceOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  collectiveEffects(getOperation(), getSources().size() + getInitials().size() + getCaptures().size(),
                    getOutputs().size(), getOperation()->getNumOperands(), effects);
}

void ScanOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  collectiveEffects(getOperation(), getSources().size() + getInitials().size() + getCaptures().size(),
                    getOutputs().size(), getOperation()->getNumOperands() - getFinals().size(), effects);
}

} // namespace intent::dsa
