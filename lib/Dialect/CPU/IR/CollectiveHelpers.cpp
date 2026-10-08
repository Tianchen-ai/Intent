#include "Intent/Dialect/CPU/IR/CollectiveHelpers.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/STLExtras.h"
#include <memory>

using namespace mlir;

namespace intent::cpu {

MemRefType collectiveStateType(Type type) {
  if (auto memory = dyn_cast<ShapedType>(type))
    return MemRefType::get(memory.getShape(), memory.getElementType());
  return MemRefType::get({}, type);
}

MemRefType collectiveMemberType(MemRefType source, ArrayRef<int64_t> axes) {
  SmallVector<int64_t> shape;
  for (int64_t axis = 0; axis < source.getRank(); ++axis)
    if (!llvm::is_contained(axes, axis))
      shape.push_back(source.getDimSize(axis));
  return MemRefType::get(shape, source.getElementType(),
      StridedLayoutAttr::get(source.getContext(), ShapedType::kDynamic,
          SmallVector<int64_t>(shape.size(), ShapedType::kDynamic)));
}

namespace {

template <typename Collective>
Value borrowedCollectiveInput(Collective operation, BlockArgument argument) {
  unsigned count = operation.getSources().size();
  unsigned index = argument.getArgNumber();
  if (operation.isDestinationPassing() && index >= count && index < 2 * count)
    return operation.getSources()[index - count];
  if (index >= 2 * count && index < 2 * count + operation.getCaptures().size())
    return operation.getCaptures()[index - 2 * count];
  return {};
}

template <typename Collective>
void populateCollectiveDependencies(
    Collective operation,
    bufferization::RegisterDependenciesFn registerDependencies) {
  for (BlockArgument argument : operation.getCombine().getArguments())
    if (isa<BaseMemRefType>(argument.getType()))
      if (Value source = borrowedCollectiveInput(operation, argument))
        registerDependencies(source, argument);
}

} // namespace

bool isCollectiveArgument(BlockArgument argument) {
  return isa<SliceReduceOp, ScanOp, RegionOpInterface>(
      argument.getOwner()->getParentOp());
}

bool isReadOnlyCollectiveArgument(BlockArgument argument) {
  Operation *owner = argument.getOwner()->getParentOp();
  if (auto program = dyn_cast<RegionOpInterface>(owner)) {
    auto schema = program.getRegionSchema(*argument.getOwner()->getParent());
    if (failed(schema)) return false;
    return (*schema)[argument.getArgNumber()].kind != RegionArgumentKind::Destinations;
  }
  auto isInput = [&](auto operation) {
    unsigned destinations = operation.isDestinationPassing() ? operation.getSources().size() : 0;
    return argument.getArgNumber() < argument.getOwner()->getNumArguments() - destinations;
  };
  if (auto reduce = dyn_cast<SliceReduceOp>(owner)) return isInput(reduce);
  if (auto scan = dyn_cast<ScanOp>(owner)) return isInput(scan);
  return false;
}

void SliceReduceOp::populateDependencies(
    bufferization::RegisterDependenciesFn registerDependencies) {
  populateCollectiveDependencies(*this, registerDependencies);
}

bool SliceReduceOp::mayBeTerminalBuffer(Value value) {
  return !borrowedCollectiveInput(*this, cast<BlockArgument>(value));
}

void ScanOp::populateDependencies(
    bufferization::RegisterDependenciesFn registerDependencies) {
  populateCollectiveDependencies(*this, registerDependencies);
}

bool ScanOp::mayBeTerminalBuffer(Value value) {
  return !borrowedCollectiveInput(*this, cast<BlockArgument>(value));
}

LogicalResult verifyCollectiveHelper(Operation *owner, Region &region,
                                     unsigned destinations) {
  Block &body = region.front();
  for (BlockArgument destination : body.getArguments().take_back(destinations)) {
    bool written = false;
    for (OpOperand &use : destination.getUses()) {
      Operation *user = use.getOwner();
      if (isa<memref::DimOp>(user)) continue;
      if (user->getBlock() != &body)
        return owner->emitOpError("collective destinations require unconditional complete writes");
      if (auto copy = dyn_cast<memref::CopyOp>(user)) {
        if (copy.getTarget() != destination || copy.getSource() == destination)
          return owner->emitOpError("collective destinations cannot be read before initialization");
      } else if (auto store = dyn_cast<memref::StoreOp>(user)) {
        if (store.getMemref() != destination ||
            cast<MemRefType>(destination.getType()).getRank() || !store.getIndices().empty())
          return owner->emitOpError("collective scalar destination stores require rank-zero slots");
      } else if (auto generic = dyn_cast<linalg::LinalgOp>(user)) {
        if (!llvm::is_contained(generic.getDpsInits(), destination) ||
            generic.getNumReductionLoops() ||
            !generic.getIndexingMapsArray()[use.getOperandNumber()].isPermutation() ||
            generic.payloadUsesValueFromOperand(&use))
          return owner->emitOpError("collective destination computation must define every element");
      } else {
        return owner->emitOpError("collective destination use does not prove a complete write");
      }
      written = true;
    }
    if (!written)
      return owner->emitOpError("collective helper must initialize every destination");
  }
  return verifyCollectiveHelperEffects(owner, region, destinations);
}

LogicalResult verifyCollectiveHelperEffects(Operation *owner, Region &region,
                                            unsigned destinations) {
  Block &body = region.front();
  std::unique_ptr<BufferViewFlowAnalysis> bufferFlow;
  WalkResult status = region.walk<WalkOrder::PreOrder>([&](Operation *nested) -> WalkResult {
    if (nested->getName().getDialectNamespace() == "intent") {
      nested->emitOpError("CPU collective helper cannot retain canonical operations");
      return WalkResult::interrupt();
    }
    for (Value input : nested->getOperands())
      if (!region.isAncestor(input.getParentRegion())) {
        nested->emitOpError("collective helper has an implicit capture: ") << input;
        return WalkResult::interrupt();
      }
    if (auto release = dyn_cast<bufferization::DeallocOp>(nested)) {
      // Native ownership conversion can merge a local allocation and a borrowed
      // input through control flow. Its condition selects the owning path; the
      // retained operands and resulting ownership flags keep their native ABI.
      for (auto [memory, condition] :
           llvm::zip(release.getMemrefs(), release.getConditions())) {
        if (matchPattern(condition, m_Zero())) continue;
        if (!bufferFlow) bufferFlow = std::make_unique<BufferViewFlowAnalysis>(owner);
        bool owned = false, borrowed = false, unknown = false;
        for (Value origin : bufferFlow->resolveReverse(memory)) {
          if (auto formal = dyn_cast<BlockArgument>(origin);
              formal && formal.getOwner() == &body) {
            borrowed = true;
            continue;
          }
          if (!region.isAncestor(origin.getParentRegion())) {
            borrowed = true;
            continue;
          }
          if (!bufferFlow->mayBeTerminalBuffer(origin)) continue;
          if (origin.getDefiningOp<memref::AllocOp>()) owned = true;
          else if (origin.getDefiningOp<memref::AllocaOp>()) borrowed = true;
          else unknown = true;
        }
        if (!owned || unknown || (borrowed && matchPattern(condition, m_One()))) {
          release.emitOpError("collective release requires local owned storage; "
                              "borrowed alternatives require native conditional ownership");
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    }
    bool collective = isa<SliceReduceOp, ScanOp, RegionOpInterface>(nested);
    SmallVector<MemoryEffects::EffectInstance> effects;
    if (auto interface = dyn_cast<MemoryEffectOpInterface>(nested)) {
      interface.getEffects(effects);
    } else if (isa<bufferization::AllocTensorOp>(nested)) {
      // The native bufferization anchor owns a fresh tensor (and optionally
      // copies another immutable tensor). It exposes BufferizableOpInterface,
      // not memref effects, until that allocation is materialized.
      return WalkResult::advance();
    } else if (nested->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
      return WalkResult::advance();
    } else if (!isMemoryEffectFree(nested)) {
      nested->emitOpError("collective helper requires known effects");
      return WalkResult::interrupt();
    }
    for (auto &effect : effects) {
      if (isa<MemoryEffects::Read, MemoryEffects::Allocate>(effect.getEffect())) continue;
      Value root = effect.getValue();
      while (root) {
        if (auto view = dyn_cast_or_null<ViewLikeOpInterface>(root.getDefiningOp()))
          root = view.getViewSource();
        else if (auto cast = root.getDefiningOp<memref::CastOp>())
          root = cast.getSource();
        else if (auto metadata = root.getDefiningOp<memref::ExtractStridedMetadataOp>())
          root = metadata.getSource();
        else break;
      }
      bool permitted;
      if (auto argument = dyn_cast_or_null<BlockArgument>(root)) {
        permitted = argument.getOwner() == &body &&
            argument.getArgNumber() >= body.getNumArguments() - destinations &&
            isa<MemoryEffects::Write>(effect.getEffect());
      } else {
        Operation *allocation = root ? root.getDefiningOp() : nullptr;
        permitted = isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(allocation) &&
            region.isAncestor(allocation->getParentRegion());
      }
      if (!permitted) {
        nested->emitOpError("collective helper write/free must target its destinations or local scratch");
        return WalkResult::interrupt();
      }
    }
    // The nested collective's verifier owns its formal arguments and scratch.
    // Its interface describes effects on the actual operands at this call site.
    return collective ? WalkResult::skip() : WalkResult::advance();
  });
  return failure(status.wasInterrupted());
}

void SliceReduceOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  unsigned outputBegin = getOperation()->getNumOperands() - getOutputs().size();
  for (OpOperand &operand : getOperation()->getOpOperands()) {
    if (!isa<MemRefType>(operand.get().getType())) continue;
    if (operand.getOperandNumber() >= outputBegin)
      effects.emplace_back(MemoryEffects::Write::get(), &operand);
    else effects.emplace_back(MemoryEffects::Read::get(), &operand);
  }
}

LogicalResult verifyCollectiveOutputs(Operation *operation, ValueRange outputs) {
  bool valueForm = operation->getNumResults() != 0;
  if (valueForm && operation->getNumResults() != outputs.size())
    return operation->emitOpError("each value result requires one destination shape");
  for (auto [index, output] : llvm::enumerate(outputs)) {
    if (!valueForm) {
      if (!isa<MemRefType>(output.getType()))
        return operation->emitOpError("buffer form requires memref destinations");
      continue;
    }
    auto shape = dyn_cast<RankedTensorType>(output.getType());
    Type result = operation->getResult(index).getType();
    if (!shape || (result != shape &&
        (shape.getRank() || result != shape.getElementType())))
      return operation->emitOpError("value result must match its tensor destination shape");
  }
  for (Value input : operation->getOperands()) {
    if (isa<RankedTensorType>(input.getType()) && !valueForm)
      return operation->emitOpError("buffer form cannot retain tensor operands");
  }
  return success();
}

LogicalResult verifyValueCollectiveHelper(Operation *owner, Region &region,
                                         TypeRange arguments,
                                         TypeRange results) {
  if (!llvm::hasSingleElement(region) || region.front().empty() ||
      !llvm::equal(region.front().getArgumentTypes(), arguments) ||
      !llvm::equal(region.front().getTerminator()->getOperandTypes(), results))
    return owner->emitOpError("value helper arguments and yielded components must match its schema");
  return verifyCollectiveHelperEffects(owner, region, 0);
}

LogicalResult SliceReduceOp::verify() {
  unsigned count = getSources().size();
  if (!count || getIdentities().size() != count || getOutputs().size() != count ||
      getAxes().empty() || !llvm::hasSingleElement(getCombine()))
    return emitOpError("slice reduction requires sources, identities, outputs, axes and one combine block");
  if (failed(verifyCollectiveOutputs(getOperation(), getOutputs()))) return failure();
  bool valueForm = getNumResults() != 0;
  auto first = cast<ShapedType>(getSources().front().getType());
  SmallVector<int64_t> seen;
  for (int64_t axis : getAxes()) {
    if (axis < 0 || axis >= first.getRank() || llvm::is_contained(seen, axis))
      return emitOpError("reduction member axes must be distinct source axes");
    seen.push_back(axis);
  }
  SmallVector<Type> states, members;
  for (auto [source, identity, output] : llvm::zip(getSources(), getIdentities(), getOutputs())) {
    auto input = cast<ShapedType>(source.getType());
    auto destination = cast<ShapedType>(output.getType());
    if (isa<RankedTensorType>(source.getType()) != valueForm)
      return emitOpError("sources and destinations must use the same value/buffer form");
    for (int64_t axis : getAxes())
      if (axis >= input.getRank() || input.getDimSize(axis) != first.getDimSize(axis))
        return emitOpError("reduction sources must share every member extent");
    auto member = collectiveMemberType(
        MemRefType::get(input.getShape(), input.getElementType()), getAxes());
    auto state = collectiveStateType(identity.getType());
    if (destination.getShape() != member.getShape() ||
        destination.getElementType() != input.getElementType() ||
        state.getElementType() != input.getElementType() ||
        (isa<ShapedType>(identity.getType()) && state.getShape() != member.getShape()))
      return emitOpError("reduction outputs and identities must preserve their complete source slices");
    states.push_back(collectiveStateType(destination));
    members.push_back(member);
  }
  Block &body = getCombine().front();
  auto yield = body.empty() ? SliceReduceYieldOp() : dyn_cast<SliceReduceYieldOp>(body.getTerminator());
  if (!yield) return emitOpError("slice reduction requires slice_reduce_yield");
  SmallVector<Type> arguments;
  if (valueForm) {
    llvm::append_range(arguments, getIdentities().getTypes());
    llvm::append_range(arguments, getIdentities().getTypes());
    llvm::append_range(arguments, getCaptures().getTypes());
    return verifyValueCollectiveHelper(getOperation(), getCombine(), arguments,
                                      getIdentities().getTypes());
  }
  if (isDestinationPassing()) {
    arguments = states;
    llvm::append_range(arguments, members);
    llvm::append_range(arguments, getCaptures().getTypes());
    llvm::append_range(arguments, states);
    if (yield.getNumOperands() || !llvm::equal(body.getArgumentTypes(), arguments))
      return emitOpError("slice reduction helper requires states, members, captures and destinations");
    return verifyCollectiveHelper(getOperation(), getCombine(), count);
  }
  for (Value identity : getIdentities()) {
    if (!isa<FloatType, IntegerType, IndexType>(identity.getType()))
      return emitOpError("scalar reduction helper identities must be numeric scalars");
    arguments.push_back(identity.getType());
  }
  llvm::append_range(arguments, getIdentities().getTypes());
  llvm::append_range(arguments, getCaptures().getTypes());
  if (!llvm::equal(body.getArgumentTypes(), arguments) ||
      !llvm::equal(yield.getOperandTypes(), getIdentities().getTypes()))
    return emitOpError("scalar reduction helper must combine two component tuples and captures");
  return verifyCollectiveHelper(getOperation(), getCombine(), 0);
}

} // namespace intent::cpu
