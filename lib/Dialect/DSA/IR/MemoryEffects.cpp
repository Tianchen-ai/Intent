#include "Intent/Dialect/DSA/IR/MemoryEffects.h"
#include "Intent/Analysis/BufferStorage.h"
#include "Intent/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;

namespace intent::dsa {
namespace {

void ordering(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), TransferOrderResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), TransferOrderResource::get());
}

void transferEffects(Operation *operation, unsigned output,
                     SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  for (OpOperand &operand : operation->getOpOperands()) {
    if (!isa<BaseMemRefType>(operand.get().getType()))
      continue;
    if (operand.getOperandNumber() == output)
      effects.emplace_back(MemoryEffects::Write::get(), &operand);
    else
      effects.emplace_back(MemoryEffects::Read::get(), &operand);
  }
}

void mixedStorageEffects(Operation *operation, ArrayRef<unsigned> written,
                         SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  for (OpOperand &operand : operation->getOpOperands()) {
    if (!isa<BaseMemRefType>(operand.get().getType()))
      continue;
    if (llvm::is_contained(written, operand.getOperandNumber()))
      effects.emplace_back(MemoryEffects::Write::get(), &operand);
    else
      effects.emplace_back(MemoryEffects::Read::get(), &operand);
  }
}

} // namespace

CompletionScope requiredCompletion(Operation *operation) {
  if (isa<StageTileOp>(operation))
    return CompletionScope::ExecutionGroup;
  if (auto load = dyn_cast<LoadTileOp>(operation); load && load.getAsynchronous())
    return CompletionScope::WorkUnit;
  if (auto gather = dyn_cast<GatherRowsOp>(operation);
      gather && gather.getAsynchronous())
    return CompletionScope::WorkUnit;
  return CompletionScope::Immediate;
}

bool completesTransfers(Operation *operation, CompletionScope scope) {
  if (scope == CompletionScope::Immediate)
    return true;
  if (isa<GroupSynchronizeOp, GroupGatherRowsOp>(operation))
    return true;
  auto synchronize = dyn_cast<SynchronizeOp>(operation);
  return scope == CompletionScope::WorkUnit && synchronize &&
         !synchronize.getLocalOnly();
}

bool hasStorageOrdering(Operation *operation) {
  auto interface = dyn_cast<MemoryEffectOpInterface>(operation);
  if (!interface)
    return false;
  SmallVector<MemoryEffects::EffectInstance> effects;
  interface.getEffects(effects);
  return llvm::any_of(effects, [](const auto &effect) {
    return effect.getResource() == TransferOrderResource::get();
  });
}

BufferStoragePolicy storagePolicy() {
  BufferStoragePolicy policy;
  policy.isBorrowedArgument = isCollectiveBorrowedArgument;
  policy.isOrderingBarrier = hasStorageOrdering;
  policy.isNonStorageResource = [](SideEffects::Resource *resource) {
    return resource == TransferOrderResource::get();
  };
  policy.provenDisjointOrigins = [](Value first, Value second) {
    auto lhs = dyn_cast<BlockArgument>(first);
    auto rhs = dyn_cast<BlockArgument>(second);
    if (!lhs || !rhs || lhs.getOwner() != rhs.getOwner())
      return false;
    auto function = dyn_cast<func::FuncOp>(lhs.getOwner()->getParentOp());
    if (!function || lhs.getOwner() != &function.front())
      return false;
    auto requirements = function->getAttrOfType<EntryRequirementsAttr>(
        entryRequirementsAttr);
    auto interface = getPublicInterface(function);
    if (!requirements || !requirements.getDisjointOutputs() || !interface)
      return false;
    auto source = getPublicView(interface, lhs.getArgNumber());
    auto target = getPublicView(interface, rhs.getArgNumber());
    return source && target &&
           (source.getAccess() != 0 || target.getAccess() != 0);
  };
  return policy;
}

void SynchronizeOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  ordering(effects);
}

void AtomicAddOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getOutputMutable());
  ordering(effects);
}

void GroupSynchronizeOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  ordering(effects);
}

void StageTileOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transferEffects(getOperation(), 1, effects);
  ordering(effects);
}

void LoadTileOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transferEffects(getOperation(), 1, effects);
  if (getAsynchronous())
    ordering(effects);
}

void GatherRowsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transferEffects(getOperation(), 2, effects);
  if (getAsynchronous())
    ordering(effects);
}

void GroupGatherRowsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  for (OpOperand &operand : getOperation()->getOpOperands()) {
    if (!isa<BaseMemRefType>(operand.get().getType()))
      continue;
    if (operand.getOperandNumber() != 3)
      effects.emplace_back(MemoryEffects::Read::get(), &operand);
    if (operand.getOperandNumber() >= 3)
      effects.emplace_back(MemoryEffects::Write::get(), &operand);
  }
  ordering(effects);
}

void IndexLayoutOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  mixedStorageEffects(getOperation(), {1}, effects);
}

void IndexBinaryOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  mixedStorageEffects(getOperation(), {2}, effects);
}

void SelectOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  mixedStorageEffects(getOperation(), {3, 4}, effects);
}

void BinaryOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  mixedStorageEffects(getOperation(), {2, 3}, effects);
}

void DivideRNOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  mixedStorageEffects(getOperation(), {2, 3}, effects);
}

void DivideCastOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  mixedStorageEffects(getOperation(), {2, 3, 4, 5, 6}, effects);
}

void MatrixTileOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  mixedStorageEffects(getOperation(), {2}, effects);
  if (getAccumulate())
    effects.emplace_back(MemoryEffects::Read::get(), &getAccumulatorMutable());
}

} // namespace intent::dsa
