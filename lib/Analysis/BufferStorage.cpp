#include "Intent/Analysis/BufferStorage.h"
#include "Intent/Dialect/Intent/IR/Interface.h"
#include "mlir/Dialect/Bufferization/IR/BufferViewFlowOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include <utility>

using namespace mlir;
namespace intent {
namespace {

struct CacheHintResource : SideEffects::Resource::Base<CacheHintResource> {
  StringRef getName() final { return "IntentCPUCacheHint"; }
};

struct PrefetchEffects
    : MemoryEffectOpInterface::ExternalModel<PrefetchEffects, memref::PrefetchOp> {
  void getEffects(Operation *operation,
                  SmallVectorImpl<MemoryEffects::EffectInstance> &effects) const {
    // MLIR 20 leaves prefetch effects unspecified. Both cache intentions observe
    // the descriptor's storage without modifying its contents; isWrite is a
    // cache hint, not a store. Keep the op's original speculation contract.
    effects.emplace_back(MemoryEffects::Read::get(),
                         &operation->getOpOperand(0));
    // Preserve the hint through ordinary MLIR dead-op elimination even though
    // it produces no SSA result. Cache state is separate from buffer contents.
    effects.emplace_back(MemoryEffects::Write::get(), CacheHintResource::get());
  }
};

struct DeallocationEffects
    : MemoryEffectOpInterface::ExternalModel<DeallocationEffects,
                                            bufferization::DeallocOp> {
  void getEffects(Operation *operation,
                  SmallVectorImpl<MemoryEffects::EffectInstance> &effects) const {
    auto release = cast<bufferization::DeallocOp>(operation);
    for (auto [memory, condition] :
         llvm::zip(release.getMemrefsMutable(), release.getConditions()))
      if (!matchPattern(condition, m_Zero()))
        effects.emplace_back(MemoryEffects::Free::get(), &memory);
  }
};

bool isBuffer(Value value) {
  return value && isa<BaseMemRefType>(value.getType());
}

bool isAllocation(Value value) {
  Operation *owner = value.getDefiningOp();
  return owner && hasEffect<MemoryEffects::Allocate>(owner, value);
}

bool isEntryArgument(Value value, func::FuncOp function) {
  auto argument = dyn_cast<BlockArgument>(value);
  return argument && argument.getOwner() == &function.front();
}

bool hasBufferResultOrArgument(Operation *operation) {
  if (llvm::any_of(operation->getResults(), isBuffer)) return true;
  for (Region &region : operation->getRegions())
    for (Block &block : region)
      if (llvm::any_of(block.getArguments(), isBuffer)) return true;
  return false;
}

bool describesBufferFlow(Operation *operation) {
  return isa<bufferization::BufferViewFlowOpInterface, ViewLikeOpInterface,
             BranchOpInterface, RegionBranchOpInterface,
             RegionBranchTerminatorOpInterface>(operation);
}

bool hasAtomicOrdering(Operation *operation) {
  return isa<memref::AtomicRMWOp, memref::GenericAtomicRMWOp>(operation);
}

func::FuncOp enclosingFunction(Operation *scope) {
  if (auto function = dyn_cast<func::FuncOp>(scope)) return function;
  return scope->getParentOfType<func::FuncOp>();
}

} // namespace

void registerBufferStorageInterfaces(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, memref::MemRefDialect *) {
    memref::PrefetchOp::attachInterface<PrefetchEffects>(*context);
  });
  registry.addExtension(+[](MLIRContext *context, bufferization::BufferizationDialect *) {
    bufferization::DeallocOp::attachInterface<DeallocationEffects>(*context);
  });
}

Value BufferStorageOriginFacts::uniqueOrigin() const {
  return complete && values.size() == 1 ? values.front() : Value{};
}

BufferStorageAnalysis::BufferStorageAnalysis(func::FuncOp function,
                                             BufferStoragePolicy policy)
    : function(function), policy(std::move(policy)), aliasAnalysis(function),
      dominance(function) {}

const BufferViewFlowAnalysis &BufferStorageAnalysis::getFlow() const {
  if (!flow) flow.emplace(function);
  return *flow;
}

BufferOriginAnalysis &BufferStorageAnalysis::getBufferOrigins() {
  if (!bufferOrigins) bufferOrigins.emplace(function);
  return *bufferOrigins;
}

BufferStorageOriginFacts BufferStorageAnalysis::origins(Value memory) const {
  BufferStorageOriginFacts result;
  if (!isBuffer(memory)) return result;
  result.complete = true;
  const auto &flow = getFlow();
  for (Value value : flow.resolveReverse(memory)) {
    if (!isBuffer(value) || !flow.mayBeTerminalBuffer(value)) continue;
    result.values.push_back(value);
    // A declared helper formal has identity within its invocation, but is not a
    // fresh allocation. Unknown operation terminals have no such contract.
    auto formal = dyn_cast<BlockArgument>(value);
    if (!isAllocation(value) && !isEntryArgument(value, function) &&
        !(formal && policy.isBorrowedArgument &&
          policy.isBorrowedArgument(formal)))
      result.complete = false;
  }
  if (result.values.empty()) result.complete = false;
  return result;
}

Value BufferStorageAnalysis::uniqueOrigin(Value memory) const {
  return origins(memory).uniqueOrigin();
}

bool isBufferStorageAliasOperation(Operation *operation) {
  return isa<ViewLikeOpInterface, memref::CastOp,
             memref::ExtractStridedMetadataOp>(operation);
}

BufferStorageAliasFacts BufferStorageAnalysis::aliases(Value root) const {
  BufferStorageAliasFacts result;
  result.root = root;
  if (!isBuffer(root)) {
    result.complete = false;
    return result;
  }
  llvm::DenseSet<Operation *> seenUsers;
  for (Value value : getFlow().resolve(root)) {
    if (!isBuffer(value)) continue;
    result.values.push_back(value);
    for (Operation *user : value.getUsers()) {
      if (!seenUsers.insert(user).second) continue;
      result.users.push_back(user);
      // Calls and returns may escape storage. Unknown buffer-producing users
      // cannot be silently treated as ordinary reads just because their effects
      // are known. Native forwarding interfaces own their complete flow.
      if (isa<CallOpInterface, func::ReturnOp,
              memref::ExtractAlignedPointerAsIndexOp>(user) ||
          (!describesBufferFlow(user) && hasBufferResultOrArgument(user)) ||
          !effects(user).complete)
        result.complete = false;
    }
  }
  return result;
}

bool BufferStorageLifetime::contains(Operation *operation) const {
  Operation *ancestor = allocation->getBlock()->findAncestorOpInBlock(*operation);
  return ancestor && allocation->isBeforeInBlock(ancestor) &&
         ancestor->isBeforeInBlock(end);
}

std::optional<BufferStorageLifetime>
BufferStorageAnalysis::lifetime(memref::AllocOp allocation) const {
  BufferStorageLifetime result{allocation, {}, aliases(allocation)};
  if (!result.aliases.complete) return std::nullopt;
  for (Operation *user : result.aliases.users) {
    auto release = dyn_cast<memref::DeallocOp>(user);
    if (!release) continue;
    if (result.end || release.getMemref() != allocation.getResult() ||
        release->getBlock() != allocation->getBlock() ||
        !allocation->isBeforeInBlock(release))
      return std::nullopt;
    result.end = release;
  }
  if (!result.end) return std::nullopt;
  for (Operation *user : result.aliases.users)
    if (user != result.end && !result.contains(user)) return std::nullopt;
  return result;
}

BufferStorageEffects BufferStorageAnalysis::effects(Operation *scope) const {
  BufferStorageEffects result;
  SmallVector<Operation *> pending;
  if (isa<func::FuncOp>(scope)) {
    for (Region &region : scope->getRegions())
      for (Block &block : region)
        for (Operation &operation : block) pending.push_back(&operation);
  } else pending.push_back(scope);
  while (!pending.empty()) {
    Operation *operation = pending.pop_back_val();
    bool recursive = operation->hasTrait<OpTrait::HasRecursiveMemoryEffects>();
    if (recursive)
      for (Region &region : operation->getRegions())
        for (Block &block : region)
          for (Operation &nested : block) pending.push_back(&nested);
    if (auto interface = dyn_cast<MemoryEffectOpInterface>(operation)) {
      SmallVector<MemoryEffects::EffectInstance> current;
      interface.getEffects(current);
      for (auto effect : current)
        if (effect.getResource() != CacheHintResource::get() &&
            !(policy.isNonStorageResource &&
              policy.isNonStorageResource(effect.getResource())))
          result.entries.push_back({operation, effect});
    } else if (!recursive) result.complete = false;
    result.ordered |= hasAtomicOrdering(operation) ||
                      (policy.isOrderingBarrier &&
                       policy.isOrderingBarrier(operation));
  }
  return result;
}

BufferStorageEffects BufferStorageAnalysis::accesses(Value memory) {
  BufferStorageEffects result;
  auto source = origins(memory);
  result.complete = source.complete;
  if (source.values.empty()) return result;

  // The function summary respects operation-owned execution boundaries. Alias
  // users also expose private storage inside those boundaries without treating
  // unrelated helper scratch as an effect on an outer actual operand.
  llvm::SetVector<Operation *> scopes;
  scopes.insert(function);
  for (Value origin : source.values) {
    auto closure = aliases(origin);
    result.complete &= closure.complete;
    for (Operation *user : closure.users) scopes.insert(user);
  }
  llvm::DenseSet<Operation *> seen;
  for (Operation *scope : scopes) {
    auto summary = effects(scope);
    result.complete &= summary.complete;
    result.ordered |= summary.ordered;
    llvm::DenseSet<Operation *> current;
    for (const auto &entry : summary.entries) {
      if (seen.contains(entry.operation)) continue;
      current.insert(entry.operation);
      if (!disjoint(memory, entry.effect.getValue())) result.entries.push_back(entry);
    }
    seen.insert(current.begin(), current.end());
  }
  return result;
}

bool BufferStorageAnalysis::disjoint(Value first, Value second) {
  if (!isBuffer(first) || !isBuffer(second) || first == second) return false;
  auto left = origins(first), right = origins(second);
  // Native local AA does not consume external buffer-flow models. In particular
  // an isolated task formal can carry an existing allocation. Resolve the
  // registered flow before asking AA about terminal storage identities.
  for (Value lhs : left.values)
    if (llvm::is_contained(right.values, lhs)) return false;
  auto allocatedAfter = [&](const BufferStorageOriginFacts &sources, Value existing) {
    return sources.complete && llvm::all_of(sources.values, [&](Value source) {
      return isAllocation(source) &&
          dominance.properlyDominates(existing, source.getDefiningOp());
    });
  };
  // A fresh allocation cannot alias a descriptor already available before it.
  // This also applies inside a helper with opaque state formals: it proves only
  // the local allocation's freshness, never disjointness between those formals.
  if (allocatedAfter(left, second) || allocatedAfter(right, first)) return true;
  if (!left.values.empty() && !right.values.empty())
    if (auto same = getBufferOrigins().isSameAllocation(first, second); same && !*same)
      return true;
  if (!left.complete || !right.complete) return false;
  auto interface = getPublicInterface(function);
  for (Value lhs : left.values)
    for (Value rhs : right.values) {
      if (lhs == rhs) return false;
      if (aliasAnalysis.alias(lhs, rhs).isNo()) continue;
      if (isEntryArgument(lhs, function) && isEntryArgument(rhs, function) && interface) {
        auto source = getPublicView(interface, cast<BlockArgument>(lhs).getArgNumber());
        auto target = getPublicView(interface, cast<BlockArgument>(rhs).getArgNumber());
        if (source && target &&
            (source.getConstraints().getNoalias() || target.getConstraints().getNoalias()))
          continue;
      }
      if (policy.provenDisjointOrigins && policy.provenDisjointOrigins(lhs, rhs))
        continue;
      return false;
    }
  return true;
}

bool BufferStorageAnalysis::disjointAt(Value first, Value second, Operation *scope) {
  if (disjoint(first, second)) return true;
  if (!isBuffer(first) || !isBuffer(second) || first == second ||
      !scope || enclosingFunction(scope) != function ||
      !dominance.dominates(first, scope) || !dominance.dominates(second, scope))
    return false;
  auto freshDuring = [&](Value memory, Value existing) {
    auto sources = origins(memory);
    return sources.complete && llvm::all_of(sources.values, [&](Value source) {
      auto allocation = source.getDefiningOp<memref::AllocOp>();
      if (!allocation) return false;
      Attribute space = allocation.getType().getMemorySpace();
      auto numericSpace = dyn_cast_or_null<IntegerAttr>(space);
      if (space && (!numericSpace || numericSpace.getInt() != 0)) return false;
      Operation *consumer = allocation->getBlock()->findAncestorOpInBlock(*scope);
      return consumer && allocation->isBeforeInBlock(consumer) &&
          dominance.properlyDominates(existing, allocation);
    });
  };
  // A backedge can carry an earlier instance of this static heap allocation.
  // Freshness distinguishes it only after the new allocation and within this
  // execution scope. Static alias/access closure must still retain that edge.
  return freshDuring(first, second) || freshDuring(second, first);
}

bool BufferStorageAnalysis::preserves(Operation *scope, Value memory) {
  auto summary = effects(scope);
  return !summary.ordered && preservesContents(summary, memory, scope);
}

bool BufferStorageAnalysis::preservesContents(Operation *scope, Value memory) {
  return preservesContents(effects(scope), memory, scope);
}

bool BufferStorageAnalysis::preservesContents(const BufferStorageEffects &summary,
                                             Value memory, Operation *scope) {
  if (!summary.complete) return false;
  for (const auto &entry : summary.entries) {
    const auto &effect = entry.effect;
    if (isa<MemoryEffects::Read, MemoryEffects::Allocate>(effect.getEffect())) continue;
    if (!disjointAt(memory, effect.getValue(), scope)) return false;
  }
  return true;
}

bool BufferStorageAnalysis::isLiveAt(Value memory, Operation *operation) const {
  auto origin = origins(memory);
  if (origin.values.empty()) return false;
  for (Value value : origin.values) {
    if (auto allocation = value.getDefiningOp<memref::AllocOp>()) {
      auto owner = lifetime(allocation);
      if (!owner || !owner->contains(operation)) return false;
    } else if (auto allocation = value.getDefiningOp<memref::AllocaOp>()) {
      Operation *owner = allocation->getParentWithTrait<OpTrait::AutomaticAllocationScope>();
      if (!owner || !owner->isAncestor(operation)) return false;
    } else if (!isEntryArgument(value, function)) {
      // A modeled region's opaque formal is valid for that region invocation.
      // This is a lifetime fact, not a fresh-allocation or no-alias assertion.
      auto formal = dyn_cast<BlockArgument>(value);
      if (!formal || !policy.isBorrowedArgument ||
          !policy.isBorrowedArgument(formal) ||
          !formal.getOwner()->getParent()->isAncestor(operation->getParentRegion()))
        return false;
    }
  }
  return true;
}

bool BufferStorageAnalysis::unchangedBetween(Value memory, Operation *from, Operation *to) {
  return unchangedBetween(memory, from, to, true);
}

bool BufferStorageAnalysis::contentsUnchangedBetween(Value memory, Operation *from,
                                                     Operation *to) {
  return unchangedBetween(memory, from, to, false);
}

bool BufferStorageAnalysis::unchangedBetween(Value memory, Operation *from,
                                            Operation *to, bool respectOrdering) {
  if (!isBuffer(memory) || enclosingFunction(from) != function ||
      enclosingFunction(to) != function || from->getBlock() != to->getBlock() ||
      !from->isBeforeInBlock(to) || !dominance.dominates(memory, to) ||
      !isLiveAt(memory, to)) return false;
  for (Operation *operation = from->getNextNode(); operation != to;
       operation = operation->getNextNode())
    if (!(respectOrdering ? preserves(operation, memory)
                          : preservesContents(operation, memory))) return false;
  return true;
}

bool BufferStorageAnalysis::readStable(Value memory, Operation *from, Operation *to) {
  if (!isBuffer(memory) || enclosingFunction(from) != function ||
      enclosingFunction(to) != function || !dominance.dominates(memory, to) ||
      !isLiveAt(memory, to)) return false;
  Operation *consumer = from->getBlock()->findAncestorOpInBlock(*to);
  if (!consumer || consumer == from || !from->isBeforeInBlock(consumer)) return false;
  for (Operation *operation = from; ; operation = operation->getNextNode()) {
    if (!preserves(operation, memory)) return false;
    if (operation == consumer) break;
  }
  return true;
}

} // namespace intent
