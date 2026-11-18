#include "PassSupport.h"
#include "LocalTransfers.h"
#include "Intent/Target/DSA/BangC/Physical/IR/MemoryEffects.h"
#include "Intent/Target/DSA/BangC/Physical/Transforms/StoragePatterns.h"
#include "mlir/IR/Dominance.h"

using namespace mlir;
namespace intent::bangc::physical {
#define GEN_PASS_DEF_DSACOMPOSEPRIVATEVERSIONS
#include "Intent/Target/DSA/BangC/Physical/Transforms/Passes.h.inc"

namespace {
bool copyInitializedValue(memref::CopyOp operation, func::FuncOp function) {
  StorageAnalysis storage(function);
  LocalSupplyRelations relations(function);
  auto copy = detail::queryLocalTransfer(operation, storage, relations);
  if (!copy || !copy->identity) return false;
  auto fill = dyn_cast_or_null<FillOp>(storage.lastWriterBefore(copy->source, operation));
  if (!fill || fill->getBlock() != operation->getBlock() ||
      !isCompleteStorageViewOf(fill.getOutput(), copy->sourceOwner) ||
      !storage.unchangedBetween(copy->source, fill, operation)) return false;
  OpBuilder builder(operation);
  builder.create<FillOp>(operation.getLoc(), copy->destination, fill.getValue());
  operation.erase();
  return true;
}

bool composeVersion(memref::CopyOp initial, func::FuncOp function) {
  StorageAnalysis storage(function);
  LocalSupplyRelations relations(function);
  auto copy = detail::queryLocalTransfer(initial, storage, relations);
  if (!copy || !copy->identity || copy->source != copy->sourceOwner ||
      copy->source.getType() != copy->destination.getType()) return false;
  auto temporary = copy->destination.getDefiningOp<memref::AllocaOp>();
  auto original = copy->source.getDefiningOp<memref::AllocaOp>();
  if (!temporary || !original || temporary->getBlock() != initial->getBlock() ||
      temporary.getAlignment().value_or(0) > original.getAlignment().value_or(0))
    return false;
  DominanceInfo dominance(function);
  if (!dominance.properlyDominates(copy->source, temporary)) return false;
  memref::CopyOp publish;
  for (Operation *next = initial->getNextNode(); next; next = next->getNextNode()) {
    auto candidate = dyn_cast<memref::CopyOp>(next);
    if (candidate && candidate.getSource() == copy->destination &&
        candidate.getTarget() == copy->source) {
      publish = candidate;
      break;
    }
    auto effects = storage.effects(next);
    if (!effects.complete || effects.ordered || dsa::pendingCompletion(next)) return false;
    // The old private version must be unobserved until publication. Moving the
    // in-place updates into its storage must not alter any intervening input.
    for (const auto &effect : effects.entries)
      if (!isa<MemoryEffects::Allocate>(effect.effect.getEffect()) &&
          !storage.disjoint(copy->source, effect.effect.getValue())) return false;
  }
  if (!publish) return false;
  auto aliases = storage.aliases(copy->destination);
  if (!aliases.complete) return false;
  for (Operation *user : aliases.users) {
    if (user == initial || user == publish || isBufferStorageAliasOperation(user)) continue;
    Operation *local = initial->getBlock()->findAncestorOpInBlock(*user);
    if (!local || local == initial || local == publish ||
        !initial->isBeforeInBlock(local) || !local->isBeforeInBlock(publish)) return false;
  }
  copy->destination.replaceAllUsesWith(copy->source);
  initial.erase();
  publish.erase();
  temporary.erase();
  return true;
}

class ComposePrivateVersionsPass final
    : public impl::DSAComposePrivateVersionsBase<ComposePrivateVersionsPass> {
public:
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(verifyRealizedProgram(module))) return signalPassFailure();
    bool changed = false;
    for (auto function : module.getOps<func::FuncOp>()) {
      while (true) {
        auto rewritten = function.walk([&](memref::CopyOp copy) {
          return (copyInitializedValue(copy, function) || composeVersion(copy, function))
              ? WalkResult::interrupt() : WalkResult::advance();
        });
        if (!rewritten.wasInterrupted()) break;
        changed = true;
      }
      // Compose initialization and final publication as well as the loop's
      // private backedge. These existing storage analyses must run before
      // supply selection, while the matrix/reduction consumers remain typed.
      while (composeLocalTransfers(function) || eliminateUnreadLocalWrites(function))
        changed = true;
    }
    LogicalResult result = success();
    if (changed) {
      OpPassManager cleanup(ModuleOp::getOperationName());
      cleanup.addPass(createCanonicalizerPass());
      cleanup.addPass(createCSEPass());
      result = runPipeline(cleanup, module);
    }
    if (failed(detail::finishTransform(module, getArgument(), result)))
      signalPassFailure();
  }
};
} // namespace
} // namespace intent::bangc::physical
