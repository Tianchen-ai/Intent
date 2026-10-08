#include "PassDetail.h"
#include "Intent/Dialect/DSA/IR/MemoryEffects.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"

using namespace mlir;

namespace intent::bangc {
namespace {

// One immutable query of the current loop. Transfers define the input frontier;
// its consumers remain the original local program, including scalar accesses,
// collective results and every implementation scratch buffer.
struct SupplyLoop {
  scf::ForOp loop;
  SmallVector<dsa::LoadTileOp> loads;
  SmallVector<memref::AllocaOp> inputs;
};

bool canPipelineIterations(scf::ForOp loop, func::FuncOp function) {
  if (loop.getNumResults() || !loop.getInductionVar().getType().isIndex())
    return false;
  auto lower = integerInterval(loop.getLowerBound(), function);
  auto upper = integerInterval(loop.getUpperBound(), function);
  auto step = integerInterval(loop.getStep(), function);
  if (!lower || !upper || !step || step->first <= 0) return false;
  // Pairing adds at most two original steps. Keep both the look-ahead index and
  // the new loop step representable, and require useful repeated computation.
  __int128 twice = __int128(step->second) * 2;
  return twice <= INT64_MAX && __int128(upper->second) + twice <= INT64_MAX &&
         __int128(lower->second) + step->second < upper->first;
}

std::optional<SupplyLoop> querySupplyLoop(scf::ForOp loop,
                                           func::FuncOp function) {
  if (!canPipelineIterations(loop, function)) return std::nullopt;
  dsa::StorageAnalysis storage(function);
  DominanceInfo dominance(function);
  SupplyLoop result{loop};
  bool computes = false, stores = false;
  for (Operation &operation : loop.getBody()->without_terminator()) {
    if (operation.getNumRegions() || isa<memref::AtomicRMWOp>(&operation))
      return std::nullopt;
    if (isa<dsa::SynchronizeOp>(&operation)) continue;
    if (dsa::requiredCompletion(&operation) != dsa::CompletionScope::Immediate)
      return std::nullopt;
    auto effects = storage.effects(&operation);
    if (!effects.complete || effects.ordered) return std::nullopt;
    auto load = dyn_cast<dsa::LoadTileOp>(&operation);
    bool supply = load && cast<MemRefType>(load.getSource().getType())
                                  .getMemorySpaceAsInt() == 0;
    if (supply) {
      auto input = load.getOutput().getDefiningOp<memref::AllocaOp>();
      if (!input || input->getBlock() != loop.getBody() ||
          !input.getType().hasStaticShape() ||
          !input.getType().getLayout().isIdentity() ||
          input.getType().getMemorySpaceAsInt() != dsa::nramSpace ||
          llvm::is_contained(result.inputs, input) ||
          !storage.preservesContents(loop, load.getSource()))
        return std::nullopt;
      auto uses = storage.accesses(input);
      // accesses() includes the function's ordering summary. The actual loop's
      // commands are checked individually above; unrelated waits do not make
      // this input's complete, dominated use set ineligible for double buffering.
      if (!storage.aliases(input).complete || !uses.complete ||
          llvm::any_of(uses.entries, [&](const auto &entry) {
            return entry.operation != load &&
                   !isa<MemoryEffects::Allocate>(entry.effect.getEffect()) &&
                   (!loop->isProperAncestor(entry.operation) ||
                    !dominance.properlyDominates(load, entry.operation));
          })) return std::nullopt;
      result.loads.push_back(load);
      result.inputs.push_back(input);
    }
    for (const auto &entry : effects.entries) {
      auto effect = entry.effect.getEffect();
      Value memory = entry.effect.getValue();
      auto type = memory ? dyn_cast<MemRefType>(memory.getType()) : MemRefType{};
      if (!type || isa<MemoryEffects::Free>(effect)) return std::nullopt;
      if (isa<MemoryEffects::Allocate>(effect)) {
        if (!isa<memref::AllocaOp>(&operation) ||
            type.getMemorySpaceAsInt() != dsa::nramSpace)
          return std::nullopt;
        continue;
      }
      if (!isa<MemoryEffects::Read, MemoryEffects::Write>(effect))
        return std::nullopt;
      if (type.getMemorySpaceAsInt() != dsa::nramSpace) {
        if (supply && memory == load.getSource() &&
            isa<MemoryEffects::Read>(effect)) continue;
        auto store = dyn_cast<dsa::StoreTileOp>(&operation);
        if (!store || memory != store.getDestination() ||
            type.getMemorySpaceAsInt() != 0 ||
            !isa<MemoryEffects::Write>(effect)) return std::nullopt;
        stores = true;
        continue;
      }
      Value origin = storage.uniqueOrigin(memory);
      if (!origin || !storage.aliases(origin).complete) return std::nullopt;
      if (isa<MemoryEffects::Write>(effect)) {
        auto allocation = origin.getDefiningOp<memref::AllocaOp>();
        if (!allocation || allocation->getBlock() != loop.getBody() ||
            llvm::any_of(storage.aliases(origin).users, [&](Operation *user) {
              return !loop->isProperAncestor(user);
            })) return std::nullopt;
      } else if ((!origin.getDefiningOp() ||
                  !loop->isProperAncestor(origin.getDefiningOp())) &&
                 (!dominance.properlyDominates(origin, loop) ||
                  !storage.preservesContents(loop, origin))) {
        return std::nullopt;
      }
      computes |= !supply && !isa<dsa::StoreTileOp>(&operation);
    }
  }
  if (result.loads.empty() || !stores || !computes) return std::nullopt;

  // Only address/extent definitions may execute with the next iteration's IV.
  // Scalar reads and any conditional computation remain at their old place.
  DenseSet<Value> available;
  std::function<bool(Value)> supplyValue = [&](Value value) {
    if (value == loop.getInductionVar() ||
        dominance.properlyDominates(value, loop)) return true;
    if (available.contains(value)) return true;
    Operation *definition = value.getDefiningOp();
    if (!definition || definition->getBlock() != loop.getBody() ||
        definition->getNumRegions() || !isMemoryEffectFree(definition) ||
        !isSpeculatable(definition) ||
        !llvm::all_of(definition->getOperands(), supplyValue)) return false;
    available.insert(value);
    return true;
  };
  for (dsa::LoadTileOp load : result.loads)
    for (Value operand : load->getOperands())
      if (operand != load.getOutput() && !supplyValue(operand))
        return std::nullopt;
  return result;
}

bool pipelineSupply(const SupplyLoop &supply, func::FuncOp function,
                    dsa::ConfigurationAttr config) {
  scf::ForOp loop = supply.loop;
  OpBuilder builder(loop);
  Location loc = loop.getLoc();
  Operation *previous = loop->getPrevNode();
  auto rollback = llvm::make_scope_exit([&] {
    while (loop->getPrevNode() != previous) loop->getPrevNode()->erase();
  });
  SmallVector<Value> first, second;
  for (memref::AllocaOp input : supply.inputs) {
    first.push_back(builder.clone(*input)->getResult(0));
    second.push_back(builder.clone(*input)->getResult(0));
  }
  Value active = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, loop.getLowerBound(), loop.getUpperBound());
  auto guard = builder.create<scf::IfOp>(loc, active, false);
  builder.setInsertionPointToStart(guard.thenBlock());

  auto bindings = [&](Value coordinate, ValueRange slots) {
    IRMapping mapping;
    mapping.map(loop.getInductionVar(), coordinate);
    for (auto [input, slot] : llvm::zip(supply.inputs, slots))
      mapping.map(input->getResult(0), slot);
    return mapping;
  };
  auto issue = [&](Value coordinate, ValueRange slots) {
    IRMapping mapping = bindings(coordinate, slots);
    std::function<Value(Value)> project = [&](Value value) -> Value {
      if (Value mapped = mapping.lookupOrNull(value)) return mapped;
      Operation *definition = value.getDefiningOp();
      if (!definition || definition->getBlock() != loop.getBody()) return value;
      for (Value operand : definition->getOperands())
        mapping.map(operand, project(operand));
      builder.clone(*definition, mapping);
      return mapping.lookup(value);
    };
    for (dsa::LoadTileOp load : supply.loads) {
      for (Value operand : load->getOperands())
        if (operand != load.getOutput()) mapping.map(operand, project(operand));
      cast<dsa::LoadTileOp>(builder.clone(*load, mapping)).setAsynchronous(true);
    }
  };
  auto compute = [&](Value coordinate, ValueRange slots) {
    IRMapping mapping = bindings(coordinate, slots);
    for (Operation &operation : loop.getBody()->without_terminator()) {
      if (llvm::any_of(supply.inputs, [&](memref::AllocaOp input) {
            return input.getOperation() == &operation;
          }) || llvm::any_of(supply.loads, [&](dsa::LoadTileOp load) {
            return load.getOperation() == &operation;
          })) continue;
      if (isa<dsa::SynchronizeOp>(&operation)) {
        // The alternate slot has no uses in this local program. Retain all
        // compute/move completion boundaries without draining that input IO.
        builder.create<dsa::SynchronizeOp>(operation.getLoc(), builder.getBoolAttr(true));
      } else {
        builder.clone(operation, mapping);
        // StoreTile has no asynchronous output form. Complete each original
        // write before later consumers or arena reuse, as in the old program.
        if (isa<dsa::StoreTileOp>(&operation))
          builder.create<dsa::SynchronizeOp>(operation.getLoc());
      }
    }
    // This wait also makes the alternate input available and completes all
    // current uses before the current slot can become a future destination.
    builder.create<dsa::SynchronizeOp>(loc);
  };

  // Preserve incoming transfer completion even if this loop follows an
  // explicitly asynchronous producer outside the selected local program.
  builder.create<dsa::SynchronizeOp>(loc);
  issue(loop.getLowerBound(), first);
  builder.create<dsa::SynchronizeOp>(loc);
  Value doubled = builder.create<arith::AddIOp>(loc, loop.getStep(), loop.getStep());
  auto pipeline = builder.create<scf::ForOp>(
      loc, loop.getLowerBound(), loop.getUpperBound(), doubled);
  pipeline->setAttrs(loop->getAttrs());
  builder.setInsertionPointToStart(pipeline.getBody());
  Value current = pipeline.getInductionVar();
  Value next = builder.create<arith::AddIOp>(loc, current, loop.getStep());
  Value hasNext = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, next, loop.getUpperBound());
  auto prefetch = builder.create<scf::IfOp>(loc, hasNext, false);
  {
    OpBuilder::InsertionGuard insertion(builder);
    builder.setInsertionPointToStart(prefetch.thenBlock());
    issue(next, second);
  }
  compute(current, first);
  auto remainder = builder.create<scf::IfOp>(loc, hasNext, false);
  {
    OpBuilder::InsertionGuard insertion(builder);
    builder.setInsertionPointToStart(remainder.thenBlock());
    Value following = builder.create<arith::AddIOp>(loc, next, loop.getStep());
    Value hasFollowing = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, following, loop.getUpperBound());
    auto future = builder.create<scf::IfOp>(loc, hasFollowing, false);
    {
      OpBuilder::InsertionGuard insertion(builder);
      builder.setInsertionPointToStart(future.thenBlock());
      issue(following, first);
    }
    compute(next, second);
  }
  if (!storageFitsBudget(function, config, measureStorage(function))) return false;
  rollback.release();
  loop.erase();
  return true;
}

} // namespace

void pipelineLocalSupply(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<scf::ForOp> loops;
  function.walk<WalkOrder::PostOrder>([&](scf::ForOp loop) { loops.push_back(loop); });
  for (scf::ForOp loop : loops)
    if (auto supply = querySupplyLoop(loop, function))
      pipelineSupply(*supply, function, config);
}

} // namespace intent::bangc
