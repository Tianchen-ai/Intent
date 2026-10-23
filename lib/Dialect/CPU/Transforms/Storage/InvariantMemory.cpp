#include "MemoryAccess.h"
#include "Intent/Dialect/CPU/Analysis/ExtentRelations.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include <memory>

using namespace mlir;

namespace intent::cpu::detail {
namespace {

bool foldInvariantDimensions(scf::ForOp loop) {
  using Variable = ValueBoundsConstraintSet::Variable;
  bool changed = false;
  for (auto [number, carried] : llvm::enumerate(loop.getRegionIterArgs())) {
    if (!isa<MemRefType>(carried.getType())) continue;
    SmallVector<memref::DimOp> dimensions;
    for (Operation *user : carried.getUsers())
      if (auto dimension = dyn_cast<memref::DimOp>(user))
        if (dimension.getConstantIndex()) dimensions.push_back(dimension);
    for (auto dimension : dimensions) {
      int64_t axis = *dimension.getConstantIndex();
      // SCF's value-bounds model proves invariant dimensions through the
      // backedge. Materialize that fact without changing the buffer identity.
      if (!haveEqualExtents(Variable(carried, axis),
                            Variable(loop.getInitArgs()[number], axis))) continue;
      OpBuilder builder(loop);
      Value initial = builder.create<memref::DimOp>(dimension.getLoc(),
                                                   loop.getInitArgs()[number], axis);
      dimension.getResult().replaceAllUsesWith(initial);
      dimension.erase();
      changed = true;
    }
  }
  return changed;
}

size_t movePureInvariants(scf::ForOp loop) {
  auto loopLike = cast<LoopLikeOpInterface>(loop.getOperation());
  return moveLoopInvariantCode(
      loopLike.getLoopRegions(),
      [&](Value value, Region *) {
        return loopLike.isDefinedOutsideOfLoop(value);
      },
      [](Operation *operation, Region *) {
        // Preserve existing conditional, task and parallel scopes. Native LICM
        // supplies the dependency closure for ordinary speculatable values.
        return operation->getNumRegions() == 0 &&
               isMemoryEffectFree(operation) && isSpeculatable(operation);
      },
      [&](Operation *operation, Region *) { loopLike.moveOutOfLoop(operation); });
}

enum class LoopExecution { Empty, NonEmpty, Unknown };

LoopExecution execution(scf::ForOp loop) {
  APInt lower, upper;
  if (matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) &&
      matchPattern(loop.getUpperBound(), m_ConstantInt(&upper)))
    return lower.slt(upper) ? LoopExecution::NonEmpty : LoopExecution::Empty;

  // The enclosing guard may have been created by an earlier fixed-point round.
  // Its condition and this loop's bounds must still be the current same SSA.
  auto branch = dyn_cast_or_null<scf::IfOp>(loop->getParentOp());
  if (!branch || loop->getBlock()->getParent() != &branch.getThenRegion())
    return LoopExecution::Unknown;
  auto compare = branch.getCondition().getDefiningOp<arith::CmpIOp>();
  return compare && compare.getPredicate() == arith::CmpIPredicate::slt &&
                 compare.getLhs() == loop.getLowerBound() &&
                 compare.getRhs() == loop.getUpperBound()
             ? LoopExecution::NonEmpty
             : LoopExecution::Unknown;
}

void guardLoop(scf::ForOp loop) {
  OpBuilder builder(loop);
  Location location = loop.getLoc();
  Value nonempty = builder.create<arith::CmpIOp>(
      location, arith::CmpIPredicate::slt, loop.getLowerBound(),
      loop.getUpperBound());
  SmallVector<Value> initial(loop.getInitArgs());
  auto guard = builder.create<scf::IfOp>(location, loop.getResultTypes(),
                                       nonempty, /*withElseRegion=*/true);
  Block &thenBody = guard.getThenRegion().front();
  loop->moveBefore(&thenBody, thenBody.begin());

  // Keep the original loop, including its body arguments, attributes and
  // automatic allocation scope. Only its external result uses are redirected.
  if (loop.getNumResults()) {
    builder.setInsertionPointToEnd(&thenBody);
    auto thenYield = builder.create<scf::YieldOp>(location, loop.getResults());
    builder.setInsertionPointToEnd(&guard.getElseRegion().front());
    builder.create<scf::YieldOp>(location, initial);
    for (auto [result, replacement] : llvm::zip(loop.getResults(), guard.getResults()))
      result.replaceUsesWithIf(replacement, [&](OpOperand &use) {
        return use.getOwner() != thenYield.getOperation();
      });
  }
}

} // namespace

bool placeInvariantMemory(func::FuncOp function) {
  if (function.isExternal())
    return false;
  SmallVector<scf::ForOp> loops;
  function.walk<WalkOrder::PostOrder>(
      [&](scf::ForOp loop) { loops.push_back(loop); });
  bool changed = false;
  for (scf::ForOp loop : loops) {
    changed |= foldInvariantDimensions(loop);
    changed |= movePureInvariants(loop) != 0;
    LoopExecution domain = execution(loop);
    if (domain == LoopExecution::Empty)
      continue;

    SmallVector<Operation *> reads;
    {
      // Pure motion above may have changed descriptor dominance. These facts
      // belong to this current loop and do not survive the guard/move below.
      std::unique_ptr<StorageAnalysis> storage;
      DominanceInfo dominance(function);
      for (Operation &operation : loop.getBody()->without_terminator()) {
        auto access = memoryAccess(&operation);
        if (!access || access->write ||
            !llvm::all_of(operation.getOperands(), [&](Value operand) {
              return dominance.properlyDominates(operand, loop);
            }))
          continue;
        if (!storage) storage = std::make_unique<StorageAnalysis>(function);
        if (!storage->preserves(loop, access->memory)) continue;
        reads.push_back(&operation);
      }
    }
    if (reads.empty())
      continue;
    if (domain == LoopExecution::Unknown)
      guardLoop(loop);
    // For an unknown trip count these operations execute in the then branch,
    // never on the zero-trip path. Existing enclosing guards remain outside.
    for (Operation *read : reads)
      read->moveBefore(loop);
    changed = true;
    // Reuse the newly available read results without a second value interpreter.
    movePureInvariants(loop);
  }
  return changed;
}

} // namespace intent::cpu::detail
