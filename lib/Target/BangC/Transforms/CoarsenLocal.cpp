#include "PassDetail.h"
#include "Intent/Dialect/DSA/IR/ExecutionRelations.h"
#include "Intent/Dialect/DSA/IR/MemoryEffects.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace intent::bangc {
namespace {

std::optional<int64_t> integer(Value value) {
  APInt bits;
  if (!matchPattern(value, m_ConstantInt(&bits)) || !bits.isSignedIntN(64))
    return std::nullopt;
  return bits.getSExtValue();
}

// A local pointwise component has one independent member for every column.
// Workspace operands are separate: their bounded native window need not grow
// when the data tile covers more members.
struct LocalMembers {
  scf::ForOp loop;
  int64_t width, pieces;
  llvm::SetVector<Value> data, workspace;
  SmallVector<dsa::LoadTileOp> loads;
  SmallVector<dsa::StoreTileOp> stores;
  SmallVector<std::pair<Value, Value>> uniform;
};

std::optional<LocalMembers> queryMembers(scf::ForOp loop,
                                        func::FuncOp function) {
  auto lower = integer(loop.getLowerBound()), upper = integer(loop.getUpperBound());
  auto step = integer(loop.getStep());
  if (loop.getNumResults() || !loop.getInductionVar().getType().isIndex() ||
      !lower || !upper || !step || *step <= 0 ||
      *step % 64 || *lower < 0 || *upper <= *lower ||
      __int128(*upper) - *lower <= *step) return std::nullopt;
  int64_t pieces = llvm::divideCeil(*upper - *lower, *step);
  LocalMembers result{loop, *step, pieces};
  dsa::StorageAnalysis storage(function);
  dsa::UniformMemoryAnalysis uniforms(function, storage);
  dsa::ExecutionRelations execution(function);
  DominanceInfo dominance(function);
  auto independent = [&](Value value) {
    return execution.coordinateDependency(
               value, cast<BlockArgument>(loop.getInductionVar())) ==
           dsa::CoordinateDependency::Independent;
  };
  auto data = [&](Value value) {
    auto type = dyn_cast<MemRefType>(value.getType());
    if (!type) return independent(value);
    if (type.getShape() != ArrayRef<int64_t>({1, *step}) ||
        !type.getLayout().isIdentity() ||
        type.getMemorySpaceAsInt() != dsa::nramSpace ||
        !type.getElementType().isIntOrIndexOrFloat()) return false;
    result.data.insert(value);
    return true;
  };
  auto workspace = [&](Value value) {
    if (!value) return;
    result.workspace.insert(value);
  };
  auto activeCount = [&](Value value) {
    // A constant full tile is valid only when the original loop has no tail.
    if (integer(value) == step)
      return (*upper - *lower) % *step == 0;
    auto minimum = value.getDefiningOp<arith::MinSIOp>();
    if (!minimum) return false;
    Value remaining = integer(minimum.getLhs()) == step ? minimum.getRhs() :
                      integer(minimum.getRhs()) == step ? minimum.getLhs() : Value{};
    auto difference = remaining ? remaining.getDefiningOp<arith::SubIOp>() : arith::SubIOp{};
    return difference && difference.getLhs() == loop.getUpperBound() &&
           difference.getRhs() == loop.getInductionVar();
  };
  std::function<bool(Value, Value)> shifted = [&](Value offset, Value stride) {
    if (dsa::isSumOfIntegerProducts(offset, {{loop.getInductionVar(), stride}}))
      return true;
    if (auto add = offset.getDefiningOp<arith::AddIOp>()) {
      if (independent(add.getLhs())) return shifted(add.getRhs(), stride);
      if (independent(add.getRhs())) return shifted(add.getLhs(), stride);
    }
    if (auto sub = offset.getDefiningOp<arith::SubIOp>())
      return independent(sub.getRhs()) && shifted(sub.getLhs(), stride);
    return false;
  };
  bool computes = false;
  llvm::SmallPtrSet<Operation *, 16> memberOperations;
  for (Operation &operation : loop.getBody()->without_terminator()) {
    if (operation.getNumRegions()) return std::nullopt;
    auto effects = storage.effects(&operation);
    if (!effects.complete || effects.ordered ||
        dsa::requiredCompletion(&operation) != dsa::CompletionScope::Immediate)
      return std::nullopt;
    if (isa<memref::AllocaOp>(&operation)) continue;
    if (auto load = dyn_cast<dsa::LoadTileOp>(&operation)) {
      auto type = cast<MemRefType>(load.getSource().getType());
      if (type.getMemorySpaceAsInt() != 0 || integer(load.getRows()) != 1 ||
          !activeCount(load.getColumns()) || !independent(load.getSource()) ||
          !independent(load.getColumnStride()) ||
          !shifted(load.getOffset(), load.getColumnStride()) ||
          !storage.preservesContents(loop, load.getSource()) || !data(load.getOutput()))
        return std::nullopt;
      result.loads.push_back(load);
      memberOperations.insert(&operation);
      continue;
    }
    if (auto store = dyn_cast<dsa::StoreTileOp>(&operation)) {
      auto type = cast<MemRefType>(store.getDestination().getType());
      if (type.getMemorySpaceAsInt() != 0 || integer(store.getRows()) != 1 ||
          !activeCount(store.getColumns()) || !independent(store.getDestination()) ||
          !independent(store.getColumnStride()) ||
          !shifted(store.getOffset(), store.getColumnStride()) || !data(store.getInput()))
        return std::nullopt;
      for (auto previous : result.stores)
        if (!storage.disjoint(previous.getDestination(), store.getDestination()))
          return std::nullopt;
      result.stores.push_back(store);
      memberOperations.insert(&operation);
      continue;
    }
    bool memberOperation = true, valid = false;
    if (auto op = dyn_cast<dsa::FillOp>(&operation))
      valid = data(op.getOutput()) && !isa<MemRefType>(op.getValue().getType()) && independent(op.getValue());
    else if (auto op = dyn_cast<dsa::UnaryOp>(&operation)) {
      valid = data(op.getInput()) && data(op.getOutput()); workspace(op.getScratch());
    } else if (auto op = dyn_cast<dsa::BinaryOp>(&operation)) {
      valid = data(op.getLhs()) && data(op.getRhs()) && data(op.getOutput()); workspace(op.getScratch());
    } else if (auto op = dyn_cast<dsa::CastOp>(&operation))
      valid = data(op.getInput()) && data(op.getOutput());
    else if (auto op = dyn_cast<dsa::CompareOp>(&operation)) {
      valid = data(op.getLhs()) && data(op.getRhs()) && data(op.getOutput()); workspace(op.getScratch());
    } else if (auto op = dyn_cast<dsa::SelectOp>(&operation)) {
      valid = data(op.getCondition()) && data(op.getTrueValue()) &&
              data(op.getFalseValue()) && data(op.getOutput()); workspace(op.getScratch());
    } else if (auto op = dyn_cast<dsa::MaskedFillOp>(&operation))
      valid = data(op.getMask()) && data(op.getInput()) &&
              data(op.getOutput()) && independent(op.getValue());
    else if (auto op = dyn_cast<dsa::DivideRNOp>(&operation)) {
      valid = data(op.getLhs()) && data(op.getRhs()) && data(op.getOutput());
      workspace(op.getScratch()); workspace(op.getLaneIndices());
    } else if (auto op = dyn_cast<memref::CopyOp>(&operation))
      valid = data(op.getSource()) && data(op.getTarget());
    else {
      memberOperation = false;
      valid = isMemoryEffectFree(&operation) && isSpeculatable(&operation);
    }
    if (!valid) return std::nullopt;
    if (memberOperation) memberOperations.insert(&operation);
    computes |= memberOperation && !isa<dsa::FillOp, memref::CopyOp>(&operation);
  }
  if (!computes || result.loads.empty() || result.stores.empty()) return std::nullopt;

  // The only out-of-loop data that can acquire additional lanes is a proved
  // uniform snapshot. Other snapshots retain their original materialization.
  for (Value value : result.data) {
    if (result.workspace.contains(value)) return std::nullopt;
    auto allocation = value.getDefiningOp<memref::AllocaOp>();
    if (!allocation || !storage.aliases(value).complete) return std::nullopt;
    // Widen only actual member operands. Shape observations and view transforms
    // are not member computations, even when their memory effects are empty.
    for (Operation *user : storage.aliases(value).users)
      if (loop->isAncestor(user) &&
          (user->getBlock() != loop.getBody() || !memberOperations.contains(user)))
        return std::nullopt;
    if (allocation->getBlock() == loop.getBody()) {
      if (llvm::any_of(storage.aliases(value).users, [&](Operation *user) {
            return user->getBlock() != loop.getBody();
          })) return std::nullopt;
    } else {
      if (!dominance.properlyDominates(value, loop) ||
          !storage.preservesContents(loop, value)) return std::nullopt;
      Value scalar = uniforms.read(value, loop);
      if (!scalar || !independent(scalar)) return std::nullopt;
      result.uniform.emplace_back(value, scalar);
    }
  }
  for (Operation &operation : loop.getBody()->without_terminator()) {
    auto effects = storage.effects(&operation);
    for (const auto &entry : effects.entries) {
      if (isa<MemoryEffects::Allocate>(entry.effect.getEffect())) continue;
      Value memory = entry.effect.getValue();
      if (!memory || !isa<MemoryEffects::Read, MemoryEffects::Write>(entry.effect.getEffect()))
        return std::nullopt;
      if (auto load = dyn_cast<dsa::LoadTileOp>(&operation);
          load && memory == load.getSource()) continue;
      if (auto store = dyn_cast<dsa::StoreTileOp>(&operation);
          store && memory == store.getDestination()) continue;
      if (!result.data.contains(memory) && !result.workspace.contains(memory))
        return std::nullopt;
      Value origin = storage.uniqueOrigin(memory);
      if (!origin || !storage.aliases(origin).complete) return std::nullopt;
      if (isa<MemoryEffects::Write>(entry.effect.getEffect())) {
        auto allocation = origin.getDefiningOp<memref::AllocaOp>();
        if (!allocation || allocation->getBlock() != loop.getBody() ||
            llvm::any_of(storage.aliases(origin).users, [&](Operation *user) {
              return user->getBlock() != loop.getBody();
            })) return std::nullopt;
      } else if ((!origin.getDefiningOp() ||
                  origin.getDefiningOp()->getBlock() != loop.getBody()) &&
                 (!dominance.properlyDominates(origin, loop) ||
                  !storage.preservesContents(loop, origin))) return std::nullopt;
    }
  }
  return result;
}

bool coarsen(const LocalMembers &members, func::FuncOp function,
             dsa::ConfigurationAttr config) {
  // Bound the search by actual owned storage before constructing a candidate.
  // The final decision below measures the full native workspace and lifetimes.
  int64_t bytesPerColumn = 0;
  for (Value value : members.data) {
    Type element = cast<MemRefType>(value.getType()).getElementType();
    bytesPerColumn += llvm::divideCeil(element.isIndex() ? 64u : element.getIntOrFloatBitWidth(), 8u);
  }
  if (!bytesPerColumn) return false;
  int64_t factor = std::min(members.pieces,
      config.getLocalBytes() / members.width / bytesPerColumn);
  for (; factor > 1; factor /= 2) {
    int64_t width = members.width * factor;
    scf::ForOp loop = members.loop;
    // The final SCF induction update must remain representable as well as every
    // active transfer coordinate; growing a tile must not introduce wraparound.
    if (__int128(*integer(loop.getUpperBound())) + width > INT64_MAX) continue;
    Operation *previous = loop->getPrevNode();
    auto rollback = llvm::make_scope_exit([&] {
      while (loop->getPrevNode() != previous) loop->getPrevNode()->erase();
    });
    OpBuilder builder(loop);
    Location loc = loop.getLoc();
    auto widen = [&](Value value) {
      auto allocation = cast<memref::AllocaOp>(builder.clone(*value.getDefiningOp()));
      auto type = allocation.getType();
      allocation.getResult().setType(MemRefType::get({1, width}, type.getElementType(),
          type.getLayout(), type.getMemorySpace()));
      return allocation.getResult();
    };
    IRMapping mapping;
    for (auto [value, scalar] : members.uniform) {
      Value expanded = widen(value);
      builder.create<dsa::FillOp>(loc, expanded, scalar);
      mapping.map(value, expanded);
    }
    Value step = builder.create<arith::ConstantIndexOp>(loc, width);
    auto enlarged = builder.create<scf::ForOp>(loc, loop.getLowerBound(), loop.getUpperBound(), step);
    enlarged->setAttrs(loop->getAttrs());
    builder.setInsertionPointToStart(enlarged.getBody());
    mapping.map(loop.getInductionVar(), enlarged.getInductionVar());
    Value remaining = builder.create<arith::SubIOp>(loc, loop.getUpperBound(), enlarged.getInductionVar());
    Value active = builder.create<arith::MinSIOp>(loc, remaining, step);
    for (Operation &operation : loop.getBody()->without_terminator()) {
      if (auto allocation = dyn_cast<memref::AllocaOp>(&operation);
          allocation && members.data.contains(allocation.getResult())) {
        mapping.map(allocation.getResult(), widen(allocation.getResult()));
        continue;
      }
      Operation *copy = builder.clone(operation, mapping);
      if (auto load = dyn_cast<dsa::LoadTileOp>(copy))
        load.getColumnsMutable().assign(active);
      if (auto store = dyn_cast<dsa::StoreTileOp>(copy))
        store.getColumnsMutable().assign(active);
    }
    StorageUsage usage = measureStorage(function);
    // Input pipelining can retain two complete tiles across all native scratch
    // users. Reserve both, since the old short-lived input may share its slot
    // with later scratch. The pipeline still checks its actual rewritten arena.
    __int128 reserve = 0;
    for (auto load : members.loads) {
      auto type = cast<MemRefType>(load.getOutput().getType());
      Type element = type.getElementType();
      int64_t bytes = width * llvm::divideCeil(
          element.isIndex() ? 64u : element.getIntOrFloatBitWidth(), 8u);
      reserve += 2 * llvm::alignTo(bytes, int64_t(128));
    }
    if (reserve > INT64_MAX - usage.nram) continue;
    usage.nram += static_cast<int64_t>(reserve);
    if (!storageFitsBudget(function, config, usage)) continue;
    rollback.release();
    loop.erase();
    return true;
  }
  return false;
}

} // namespace

bool coarsenLocalPrograms(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<scf::ForOp> loops;
  function.walk<WalkOrder::PostOrder>([&](scf::ForOp loop) { loops.push_back(loop); });
  bool changed = false;
  for (scf::ForOp loop : loops)
    if (auto members = queryMembers(loop, function))
      changed |= coarsen(*members, function, config);
  return changed;
}

} // namespace intent::bangc
