#include "Intent/Dialect/GPU/Analysis/MemoryEffects.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Analysis/ReductionConsumers.h"
#include "Intent/Dialect/GPU/Analysis/ResourceAlias.h"
#include "Intent/Dialect/GPU/Analysis/Resources.h"
#include "Intent/Dialect/GPU/IR/ProgramInterface.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "../Pointwise/Pointwise.h"
#include "ReductionChains.h"
#include "ReductionRealization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;

namespace intent::gpu {
namespace {

struct ConsumerSchedule {
  SmallVector<Operation *> operations;
  SmallVector<std::pair<Value, Value>> disjointViews;
  PhysicalExprAttr peak;
};

// This is a temporary proof over the original SSA. In particular, moving reads
// into either arm must already preserve their epoch; a prospective alias guard
// only authorizes the additional reads in the selected output traversal.
FailureOr<ConsumerSchedule> prepareSchedule(func::FuncOp kernel,
                                            ReductionConsumerGroup &group) {
  ConsumerSchedule schedule;
  llvm::DenseSet<Operation *> owned;
  owned.insert(group.reduce);
  for (Operation *operation : group.producerOperations) owned.insert(operation);
  for (Operation *operation : group.outputOperations) owned.insert(operation);
  for (StoreOp store : group.stores) owned.insert(store);
  for (LoadOp load : group.sourceLoads)
    if (!canReplayReadAt(load, group.reduce)) return failure();

  // Unrelated effects stay at their original anchors. We cannot move a grouped
  // store across one of them, or export a post-summary value before its join.
  Operation *last = group.stores.back();
  for (Operation *operation = group.reduce->getNextNode(); operation != last;
       operation = operation->getNextNode())
    if (!operation || (!owned.contains(operation) &&
                       !isMemoryEffectFree(operation))) return failure();
  for (Operation *operation : group.outputOperations)
    if (group.reduce->isBeforeInBlock(operation))
      for (Value result : operation->getResults())
        for (Operation *user : result.getUsers())
          if (!owned.contains(user)) return failure();

  // Coordinate carriers with other uses may remain outside. Close that decision
  // transitively so that none of their operands disappears into the branches.
  bool changed;
  do {
    changed = false;
    SmallVector<Operation *> retained;
    for (Operation *operation : owned) {
      if (operation == group.reduce || isa<StoreOp>(operation)) continue;
      if (llvm::any_of(operation->getResults(), [&](Value value) {
            return llvm::any_of(value.getUsers(), [&](Operation *user) {
              return !owned.contains(user);
            });
          })) retained.push_back(operation);
    }
    for (Operation *operation : retained) changed |= owned.erase(operation);
  } while (changed);
  for (LoadOp load : group.sourceLoads)
    if (!owned.contains(load)) return failure();
  for (Value value : group.wholeProducers)
    if (!owned.contains(value.getDefiningOp())) return failure();

  ResourceAliasAnalysis aliases;
  auto disjoint = [&](Value lhs, Value rhs) {
    if (aliases.disjointAt(lhs, rhs, group.reduce)) return true;
    auto left = dyn_cast<BlockArgument>(lhs), right = dyn_cast<BlockArgument>(rhs);
    if (lhs == rhs || !left || !right || left.getOwner() != &kernel.front() ||
        right.getOwner() != &kernel.front() || !getPublicView(lhs) ||
        !getPublicView(rhs)) return false;
    auto pair = std::pair<Value, Value>{lhs, rhs};
    auto reverse = std::pair<Value, Value>{rhs, lhs};
    if (!llvm::is_contained(schedule.disjointViews, pair) &&
        !llvm::is_contained(schedule.disjointViews, reverse))
      schedule.disjointViews.push_back(pair);
    return true;
  };
  for (LoadOp load : group.sourceLoads)
    for (StoreOp store : group.stores)
      if (!preservesMemoryReads(load, store, aliases, disjoint)) return failure();
  for (auto [index, store] : llvm::enumerate(group.stores)) {
    if (!getPublicView(store.getResource())) return failure();
    for (StoreOp preceding : ArrayRef<StoreOp>(group.stores).take_front(index))
      if (!disjoint(preceding.getResource(), store.getResource())) return failure();
  }

  for (Operation &operation : *group.reduce->getBlock())
    if (owned.contains(&operation)) schedule.operations.push_back(&operation);
  SmallVector<PhysicalExprAttr> stages;
  for (Operation *operation : schedule.operations) {
    if (isa<MakeRangeOp>(operation)) continue;
    auto words = nominalLivePayloadWords(operation);
    if (!words) return failure();
    stages.push_back(words);
  }
  schedule.peak = maximumNominalWorkingSet(kernel, stages);
  if (!schedule.peak) return failure();
  return schedule;
}

FailureOr<bool> realizeConsumers(func::FuncOp kernel,
                                 ReductionConsumerGroup &group,
                                 unsigned preferredResidentPrograms) {
  auto schedule = prepareSchedule(kernel, group);
  if (failed(schedule)) return false;
  auto capabilities = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  if (!capabilities || capabilities.getRegistersPerUnit() <= 0) return false;
  // Streaming repeats source reads. Do not perturb a fitting complete native
  // primitive/configuration family merely because some larger hypothetical
  // row exceeds the preferred budget. The adapter's nominal residency cost
  // preference is separate from the shared legality and construction proofs.
  int64_t budget = capabilities.getRegistersPerUnit() / preferredResidentPrograms;
  auto bounds = queryPositiveExtentBounds(schedule->peak, kernel);
  if (!bounds || bounds->first <= budget) return false;

  OpBuilder builder(group.reduce);
  Location location = group.reduce.getLoc();
  Value footprint = builder.create<PhysicalExprOp>(
      location, builder.getIndexType(), schedule->peak);
  Value limit = builder.create<arith::ConstantIndexOp>(location, budget);
  Value condition = builder.create<CompareOp>(location, builder.getI1Type(),
      footprint, limit, ComparePredicate::Gt);
  auto append = [&](Value predicate) {
    condition = builder.create<BinaryOp>(location, builder.getI1Type(), condition,
        predicate, BinaryOperator::LogicalAnd);
  };
  for (StoreOp store : group.stores) {
    auto injective = materializeNonOverlappingView(kernel, store.getResource());
    if (failed(injective)) return failure();
    append(*injective);
  }
  for (auto [lhs, rhs] : schedule->disjointViews) {
    OpBuilder entry(&kernel.front(), kernel.front().begin());
    Value overlap = entry.create<ViewOverlapOp>(location, entry.getI1Type(), lhs, rhs);
    Value zero = builder.create<arith::ConstantOp>(location, builder.getBoolAttr(false));
    append(builder.create<CompareOp>(location, builder.getI1Type(), overlap,
                                     zero, ComparePredicate::Eq));
  }
  auto branch = builder.create<scf::IfOp>(location, group.reduce.getResultTypes(),
                                         condition, true);
  for (Region &region : branch->getRegions())
    if (!region.front().empty()) region.front().back().erase();
  IRMapping selected;
  builder.setInsertionPointToStart(branch.thenBlock());
  for (Operation *operation : schedule->operations) builder.clone(*operation, selected);
  auto fastReduce = cast<ReduceOp>(selected.lookup(group.reduce.getOperation()));
  auto fastRange = cast<MakeRangeOp>(selected.lookupOrDefault(group.range.getOperation()));
  SmallVector<StoreOp> fastStores;
  for (StoreOp store : group.stores)
    fastStores.push_back(cast<StoreOp>(selected.lookup(store.getOperation())));
  builder.create<scf::YieldOp>(location, fastReduce.getResults());
  for (Operation *operation : schedule->operations)
    operation->moveBefore(branch.elseBlock(), branch.elseBlock()->end());
  builder.setInsertionPointToEnd(branch.elseBlock());
  builder.create<scf::YieldOp>(location, group.reduce.getResults());
  for (auto [result, joined] : llvm::zip(group.reduce.getResults(), branch.getResults()))
    result.replaceUsesWithIf(joined, [&](OpOperand &use) {
      return !branch->isProperAncestor(use.getOwner());
    });

  // Re-query in the real selected arm. Its geometry guard is public alias
  // authority; the original full-domain SSA does not serve as an epoch proof.
  for (LoadOp load : group.sourceLoads) {
    auto read = cast<LoadOp>(selected.lookup(load.getOperation()));
    if (!canReplayReadAt(read, fastStores.front()))
      return fastReduce.emitOpError("selected summary output cannot replay its source epoch");
  }
  if (failed(pointwise::realizeReusePointwiseTraversal(kernel, fastRange, fastStores,
          /*effectLocal=*/true, {}, [](StoreOp, StoreOp) {}))) return failure();
  ResourceAliasAnalysis aliases;
  WalkResult stable = branch.getThenRegion().walk([&](scf::ForOp loop) {
    auto effects = getEffectsRecursively(loop);
    if (!effects || llvm::any_of(*effects, [](const auto &effect) {
          return isa<MemoryEffects::Free>(effect.getEffect());
        })) return WalkResult::interrupt();
    return loop.walk([&](LoadOp load) {
      return preservesMemoryReads(load, loop, aliases, [&](Value lhs, Value rhs) {
        return aliases.disjointAt(lhs, rhs, loop);
      }) ? WalkResult::advance() : WalkResult::interrupt();
    });
  });
  if (stable.wasInterrupted())
    return fastReduce.emitOpError("selected output traversal does not preserve its source epoch");
  eraseDeadPhysicalValues(kernel);

  SmallVector<reduction::SourcePlan> sources;
  for (Value source : fastReduce.getSources()) {
    auto plan = reduction::analyzeSource(source, group.reductionAxis);
    if (failed(plan)) return fastReduce.emitOpError("selected summary has no bounded input supply");
    sources.push_back(*plan);
  }
  if (failed(reduction::realizeRuntimeReduce(fastReduce, sources, kernel,
          /*tileProducerFreeAxis=*/false,
          reduction::ReductionCarryForm::CompactSummary))) return failure();
  return true;
}

} // namespace

LogicalResult realizeReductionConsumerTraversals(
    ModuleOp module, unsigned preferredResidentPrograms) {
  if (!preferredResidentPrograms)
    return module.emitError("reduction residency preference must be positive");
  auto kernel = getPhysicalKernel(module);
  if (failed(kernel)) return failure();
  while (reduction::combineNestedReductions(*kernel)) {}
  if (failed(fuseIndependentReductions(module))) return failure();
  // Only completed original arms are excluded; their effectful stores retain
  // their summaries. Unmatched reductions are never cached across whole-kernel
  // DCE, and each successful rewrite restarts from the current operation tree.
  llvm::DenseSet<Operation *> originalArms;
  while (true) {
    SmallVector<ReduceOp> reductions;
    kernel->walk([&](ReduceOp reduce) {
      if (!originalArms.contains(reduce)) reductions.push_back(reduce);
    });
    bool changed = false;
    for (ReduceOp reduce : reductions) {
      for (int64_t axis : llvm::reverse(reduce.getAxes())) {
        auto group = queryReductionConsumerGroup(reduce, axis);
        if (failed(group)) continue;
        auto realized = realizeConsumers(*kernel, *group, preferredResidentPrograms);
        if (failed(realized)) return failure();
        if (*realized) {
          originalArms.insert(group->reduce);
          changed = true;
          break;
        }
      }
      if (changed) break;
    }
    if (!changed) break;
  }
  eraseDeadPhysicalValues(*kernel);
  return closeValueRelations(*kernel);
}

} // namespace intent::gpu
