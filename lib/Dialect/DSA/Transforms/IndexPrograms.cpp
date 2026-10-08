#include "PassSupport.h"
#include "Intent/Dialect/DSA/Transforms/StoragePatterns.h"
#include "Intent/Dialect/DSA/Transforms/LocalSupplyRelations.h"
#include "Intent/Dialect/DSA/Transforms/Passes.h"
#include "Intent/Dialect/DSA/Analysis/PhysicalProgram.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/MathExtras.h"
#include <functional>
#include <limits>

using namespace mlir;
namespace intent::dsa {
bool forwardIndexExpressions(func::FuncOp function) {
  SmallVector<memref::StoreOp> stores;
  function.walk([&](memref::StoreOp store) { stores.push_back(store); });
  auto integer = [](Type type) { return isa<IndexType, IntegerType>(type); };
  bool changed = false;
  for (auto store : stores) {
    StorageAnalysis storage(function);
    Value buffer = store.getMemref();
    auto allocation = buffer.getDefiningOp<memref::AllocaOp>();
    auto producer = dyn_cast<scf::ForOp>(store->getParentOp());
    auto type = store.getMemRefType();
    if (!allocation || !producer || allocation->getBlock() != producer->getBlock() ||
        !producer.getInitArgs().empty() || type.getRank() != 2 || type.getDimSize(0) != 1 ||
        !type.getLayout().isIdentity() || type.getMemorySpaceAsInt() != dsa::nramSpace ||
        (!type.getElementType().isInteger(32) && !type.getElementType().isInteger(64)) ||
        !matchPattern(producer.getLowerBound(), m_Zero()) || !matchPattern(producer.getStep(), m_One()) ||
        store.getIndices().size() != 2 || !matchPattern(store.getIndices()[0], m_Zero()) ||
        store.getIndices()[1] != producer.getInductionVar()) continue;
    // Replaying only integer arithmetic and immutable view strides avoids
    // moving data reads across intervening writes or duplicating FP work.
    bool pure = llvm::all_of(*producer.getBody(), [&](Operation &operation) {
      if (&operation == store.getOperation() || isa<scf::YieldOp>(operation)) return true;
      if (isa<dsa::StrideOp>(operation)) return true;
      return operation.getName().getDialectNamespace() == "arith" && !operation.getNumRegions() &&
          isMemoryEffectFree(&operation) && llvm::all_of(operation.getOperandTypes(), integer) &&
          llvm::all_of(operation.getResultTypes(), integer);
    });
    if (!pure) continue;
    auto writes = storage.writers(buffer);
    if (failed(writes))
      continue;
    bool uniqueWrite = llvm::all_of(*writes, [&](Operation *writer) {
      if (writer == store)
        return true;
      auto fill = dyn_cast<dsa::FillOp>(writer);
      return fill && isCompleteStorageViewOf(fill.getOutput(), buffer) &&
             fill->getBlock() == producer->getBlock() &&
             fill->isBeforeInBlock(producer);
    });
    SmallVector<memref::LoadOp> loads;
    for (Operation *user : storage.aliases(buffer).users)
      if (auto load = dyn_cast<memref::LoadOp>(user);
          load && detail::sameCompleteView(storage, load.getMemref(), buffer))
        loads.push_back(load);
    if (!uniqueWrite || loads.empty()) continue;
    for (auto load : loads) {
      StorageAnalysis currentStorage(function);
      auto consumer = dyn_cast<scf::ForOp>(load->getParentOp());
      if (!consumer || consumer->getBlock() != producer->getBlock() ||
          !producer->isBeforeInBlock(consumer) ||
          !consumer.getInitArgs().empty() ||
          consumer.getUpperBound() != producer.getUpperBound() ||
          !matchPattern(consumer.getLowerBound(), m_Zero()) ||
          !matchPattern(consumer.getStep(), m_One()) ||
          load.getIndices().size() != 2 ||
          !matchPattern(load.getIndices()[0], m_Zero()) ||
          load.getIndices()[1] != consumer.getInductionVar() ||
          !currentStorage.unchangedBetween(buffer, producer, consumer) ||
          !currentStorage.preserves(consumer, buffer))
        continue;
      IRMapping mapping;
      mapping.map(producer.getInductionVar(), consumer.getInductionVar());
      OpBuilder builder(load);
      for (Operation &operation : *producer.getBody())
        if (&operation != store.getOperation() && !isa<scf::YieldOp>(operation)) builder.clone(operation, mapping);
      load.getResult().replaceAllUsesWith(mapping.lookupOrDefault(store.getValue()));
      load.erase();
      changed = true;
    }
    if (llvm::all_of(buffer.getUsers(), [&](Operation *user) {
          return user == store.getOperation() || isa<dsa::FillOp>(user);
        })) {
      producer.erase();
      SmallVector<Operation *> fills(buffer.getUsers());
      for (Operation *fill : fills) fill->erase();
      allocation.erase();
    }
  }
  return changed;
}

bool realizeRangeComparisons(func::FuncOp function) {
  SmallVector<dsa::CompareOp> comparisons;
  function.walk([&](dsa::CompareOp compare) { comparisons.push_back(compare); });
  bool changed = false;
  for (auto compare : comparisons) {
    if (compare.getPredicate() != ComparePredicate::Ge) continue;
    StorageAnalysis storage(function);
    DominanceInfo dominance(function);
    LocalSupplyRelations relations(function);
    auto grid = queryLocalCoordinateGrid(compare, storage, relations);
    if (!grid) continue;
    if (!storage.readStable(grid->rows.getSource(), grid->rows, compare)) continue;
    auto column = queryLocalIndexSequence(grid->columns.getSource(), grid->columns,
                                         storage, dominance, relations);
    if (!column) continue;
    auto base = dyn_cast<AffineConstantExpr>(column->begin);
    int64_t columns = cast<MemRefType>(compare.getLhs().getType()).getDimSize(1);
    if (!base || columns <= 0 || base.getValue() > std::numeric_limits<int64_t>::max() - (columns - 1)) continue;
    OpBuilder b(compare);
    Location loc = compare.getLoc();
    // Both broadcasts pad with zero: GE is true in their inactive rows too.
    b.create<dsa::CompareRangeOp>(loc, grid->rows.getSource(), compare.getOutput(),
                                grid->rows.getRows(), b.getI64IntegerAttr(base.getValue()));
    compare.erase();
    for (auto copy : {grid->rows, grid->columns})
      if (llvm::all_of(copy.getOutput().getUsers(), [&](Operation *user) { return user == copy; })) copy.erase();
    changed = true;
  }
  return changed;
}

bool foldPresentRange(dsa::ReduceOp reduce, dsa::CompareRangeOp range, dsa::LoadTileOp copy,
                      func::FuncOp function, DominanceInfo &dominance) {
  auto exact = [&](Value value, int64_t expected) {
    auto interval = integerInterval(value, function);
    return interval && interval->first == expected && interval->second == expected;
  };
  int64_t rows = cast<MemRefType>(range.getOutput().getType()).getDimSize(0);
  if (!copy || !exact(range.getRows(), rows) || !exact(copy.getRows(), rows) ||
      !exact(copy.getColumns(), 1) || !exact(copy.getOffset(), 0) ||
      !exact(copy.getRowStride(), 1) || !exact(copy.getColumnStride(), 1)) return false;
  Value source = copy.getSource();
  StorageAnalysis storage(function);
  Value origin = storage.uniqueOrigin(source);
  if (!origin || !origin.getDefiningOp<memref::AllocaOp>() ||
      !isCompleteStorageViewOf(source, origin) ||
      cast<MemRefType>(source.getType()).getNumElements() != rows)
    return false;
  SmallVector<dsa::FillOp> fills;
  memref::StoreOp initialization;
  auto writers = storage.writers(source);
  if (failed(writers))
    return false;
  for (Operation *user : *writers) {
    if (auto store = dyn_cast<memref::StoreOp>(user);
        store && isCompleteStorageViewOf(store.getMemref(), origin)) {
      if (initialization) return false;
      initialization = store; continue;
    }
    if (auto fill = dyn_cast<dsa::FillOp>(user);
        fill && isCompleteStorageViewOf(fill.getOutput(), origin)) {
      fills.push_back(fill);
      continue;
    }
    return false;
  }
  auto ramp = initialization ? dyn_cast<scf::ForOp>(initialization->getParentOp()) : scf::ForOp();
  if (!ramp || !ramp.getInitArgs().empty() || ramp->isProperAncestor(copy) ||
      !dominance.dominates(ramp, copy) || !exact(ramp.getLowerBound(), 0) ||
      !exact(ramp.getUpperBound(), rows) || !exact(ramp.getStep(), 1) ||
      llvm::any_of(fills, [&](dsa::FillOp fill) { return !dominance.dominates(fill, ramp); })) return false;
  auto indices = initialization.getIndices();
  auto initialized = initialization.getMemRefType();
  bool complete = indices.size() == 2 &&
      ((initialized.getShape() == ArrayRef<int64_t>({1, rows}) && exact(indices[0], 0) && indices[1] == ramp.getInductionVar()) ||
       (initialized.getShape() == ArrayRef<int64_t>({rows, 1}) && indices[0] == ramp.getInductionVar() && exact(indices[1], 0)));
  auto coordinates = integerInterval(initialization.getValue(), function);
  if (!complete || !coordinates || coordinates->first < range.getBaseAttr().getInt()) return false;
  memref::LoadOp count;
  for (Operation *user : reduce.getOutput().getUsers()) {
    if (user == reduce) continue;
    auto load = dyn_cast<memref::LoadOp>(user);
    if (!load || load.getMemref() != reduce.getOutput() || count) return false;
    count = load;
  }
  if (!count || !count.getResult().hasOneUse()) return false;
  auto compare = dyn_cast<arith::CmpFOp>(*count.getResult().getUsers().begin());
  if (!compare || !compare.getResult().hasOneUse() ||
      (compare.getPredicate() != arith::CmpFPredicate::UNE && compare.getPredicate() != arith::CmpFPredicate::ONE)) return false;
  Value other = compare.getLhs() == count.getResult() ? compare.getRhs() : compare.getLhs();
  FloatAttr zero;
  if (!matchPattern(other, m_Constant(&zero)) || !zero.getValue().isZero()) return false;
  auto store = dyn_cast<memref::StoreOp>(*compare.getResult().getUsers().begin());
  auto loop = store ? dyn_cast<scf::ForOp>(store->getParentOp()) : scf::ForOp();
  if (!loop || !loop.getInitArgs().empty() || count->getBlock() != loop.getBody() || compare->getBlock() != loop.getBody() ||
      !exact(loop.getLowerBound(), 0) || !exact(loop.getUpperBound(), rows) || !exact(loop.getStep(), 1) ||
      store.getMemRefType().getShape() != ArrayRef<int64_t>({1, rows}) || !store.getMemRefType().getElementType().isInteger(1) ||
      store.getMemRefType().getMemorySpaceAsInt() != dsa::nramSpace || !store.getMemRefType().getLayout().isIdentity() ||
      store.getIndices().size() != 2 || !exact(store.getIndices()[0], 0) || store.getIndices()[1] != loop.getInductionVar() ||
      count.getIndices().size() != 2 || !exact(count.getIndices()[0], 0) || count.getIndices()[1] != loop.getInductionVar()) return false;
  for (Operation &operation : loop.getBody()->without_terminator())
    if (&operation != count && &operation != store && (operation.getNumRegions() || !isMemoryEffectFree(&operation))) return false;
  OpBuilder b(loop);
  b.create<dsa::FillOp>(loop.getLoc(), store.getMemref(), b.create<arith::ConstantIntOp>(loop.getLoc(), 1, 1));
  loop.erase();
  return true;
}

bool foldRangeCounts(func::FuncOp function) {
  DominanceInfo dominance(function);
  auto soleWriter = [&](Value buffer, Operation *consumer) -> Operation * {
    StorageAnalysis storage(function);
    Value origin = storage.uniqueOrigin(buffer);
    if (!origin || !origin.getDefiningOp<memref::AllocaOp>())
      return nullptr;
    Operation *writer = storage.uniqueWriter(buffer);
    if (!writer || !dominance.dominates(writer, consumer))
      return nullptr;
    auto completion = storage.completionOfUse(writer);
    if (failed(completion) || *completion != writer)
      return nullptr;
    for (const auto &entry : storage.effects(writer).entries)
      if (isa<MemoryEffects::Write>(entry.effect.getEffect()) &&
          !storage.disjoint(buffer, entry.effect.getValue()) &&
          !detail::sameCompleteView(storage, buffer, entry.effect.getValue()))
        return nullptr;
    return writer;
  };
  SmallVector<dsa::ReduceOp> reductions;
  function.walk([&](dsa::ReduceOp reduce) { reductions.push_back(reduce); });
  bool changed = false;
  for (auto reduce : reductions) {
    FloatAttr identity;
    if (reduce.getKind() != BinaryOperator::Add || !matchPattern(reduce.getIdentity(), m_Constant(&identity)) ||
        !identity.getValue().isZero() ||
        llvm::any_of(reduce.getScratch().getUsers(), [&](Operation *user) { return user != reduce; })) continue;
    Value converted = reduce.getInput();
    Operation *consumer = reduce;
    dsa::TransposeOp transpose;
    if (reduce.getAxis() == 0) {
      transpose = dyn_cast_or_null<dsa::TransposeOp>(soleWriter(converted, consumer));
      if (!transpose) continue;
      converted = transpose.getInput(); consumer = transpose;
    } else if (reduce.getAxis() != 1) continue;
    auto conversion = dyn_cast_or_null<dsa::CastOp>(soleWriter(converted, consumer));
    if (!conversion) continue;
    auto mask = cast<MemRefType>(conversion.getInput().getType());
    auto type = cast<MemRefType>(converted.getType());
    if (!mask.getElementType().isInteger(1) || !type.getElementType().isF32() || mask.getShape() != type.getShape()) continue;
    auto range = dyn_cast_or_null<dsa::CompareRangeOp>(soleWriter(conversion.getInput(), conversion));
    if (!range || !soleWriter(range.getRowCoordinates(), range)) continue;
    int64_t rows = mask.getDimSize(0), columns = mask.getDimSize(1);
    auto constant = [](Value value, int64_t expected) {
      auto op = value.getDefiningOp<arith::ConstantIndexOp>();
      return op && op.value() == expected;
    };
    if (columns > (1 << 24) || !constant(reduce.getCount(), columns) ||
        (transpose && (!constant(transpose.getRows(), rows) || !constant(transpose.getColumns(), columns)))) continue;
    auto output = cast<MemRefType>(reduce.getOutput().getType());
    if (!output.getLayout().isIdentity() || output.getNumElements() != rows ||
        (output.getDimSize(0) != 1 && output.getDimSize(1) != 1)) continue;
    if (foldPresentRange(reduce, range, dyn_cast_or_null<dsa::LoadTileOp>(soleWriter(range.getRowCoordinates(), range)), function, dominance)) {
      reduce.erase();
      if (transpose && llvm::all_of(transpose.getOutput().getUsers(), [&](Operation *user) { return user == transpose; })) transpose.erase();
      if (llvm::all_of(conversion.getOutput().getUsers(), [&](Operation *user) { return user == conversion; })) conversion.erase();
      changed = true;
      continue;
    }
    OpBuilder b(reduce);
    Location loc = reduce.getLoc();
    Value zero = b.create<arith::ConstantIndexOp>(loc, 0), one = b.create<arith::ConstantIndexOp>(loc, 1);
    Value extent = b.create<arith::ConstantIndexOp>(loc, rows);
    Value base = b.create<arith::ConstantIntOp>(loc, range.getBase(), 64);
    Value last = b.create<arith::ConstantIntOp>(loc, range.getBase() + columns - 1, 64);
    Value width = b.create<arith::ConstantIntOp>(loc, columns, 64);
    Value empty = b.create<arith::ConstantIntOp>(loc, 0, 64);
    Value unit = b.create<arith::ConstantIntOp>(loc, 1, 64);
    b.create<scf::ForOp>(loc, zero, extent, one, ValueRange{}, [&](OpBuilder &nested, Location loc, Value row, ValueRange) {
      Value query = nested.create<memref::LoadOp>(loc, range.getRowCoordinates(), ValueRange{row, zero});
      Value below = nested.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, query, base);
      Value full = nested.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, query, last);
      Value count = nested.create<arith::SubIOp>(loc, query, base);
      count = nested.create<arith::AddIOp>(loc, count, unit);
      count = nested.create<arith::SelectOp>(loc, full, width, count);
      count = nested.create<arith::SelectOp>(loc, below, empty, count);
      Value active = nested.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, row, range.getRows());
      count = nested.create<arith::SelectOp>(loc, active, count, width);
      Value result = nested.create<arith::SIToFPOp>(loc, nested.getF32Type(), count);
      SmallVector<Value> indices = output.getDimSize(0) == 1 ? SmallVector<Value>{zero, row} : SmallVector<Value>{row, zero};
      nested.create<memref::StoreOp>(loc, result, reduce.getOutput(), indices);
      nested.create<scf::YieldOp>(loc);
    });
    reduce.erase();
    auto eraseDeadWriter = [](Operation *op, Value buffer) {
      if (llvm::all_of(buffer.getUsers(), [&](Operation *user) { return user == op; })) op->erase();
    };
    if (transpose) eraseDeadWriter(transpose, transpose.getOutput());
    eraseDeadWriter(conversion, conversion.getOutput());
    changed = true;
  }
  return changed;
}

bool reuseGatherOffsets(func::FuncOp function) {
  DominanceInfo dominance(function);
  struct Candidate { scf::ForOp loop; memref::StoreOp store; };
  SmallVector<scf::ForOp> loops;
  SmallVector<Candidate> candidates;
  function.walk([&](scf::ForOp loop) { loops.push_back(loop); });
  bool changed = false;
  for (auto loop : loops) {
    if (!loop.getInitArgs().empty() || !matchPattern(loop.getLowerBound(), m_Zero()) ||
        !matchPattern(loop.getStep(), m_One())) continue;
    memref::StoreOp store;
    SmallVector<Value> sources;
    bool eligible = true;
    for (Operation &op : loop.getBody()->without_terminator()) {
      if (auto write = dyn_cast<memref::StoreOp>(op)) {
        if (store) { eligible = false; break; }
        store = write;
      } else if (auto read = dyn_cast<memref::LoadOp>(op)) sources.push_back(read.getMemref());
      else if (op.getNumRegions() || !isMemoryEffectFree(&op) ||
          (op.getName().getDialectNamespace() != "arith" && !isa<dsa::StrideOp>(op))) {
        eligible = false; break;
      }
    }
    if (!eligible || !store) continue;
    Value output = store.getMemref();
    auto type = store.getMemRefType();
    auto allocation = output.getDefiningOp<memref::AllocaOp>();
    if (!allocation || type.getRank() != 2 || type.getDimSize(0) != 1 || !type.getElementType().isInteger(64) ||
        type.getMemorySpaceAsInt() != dsa::nramSpace || store.getIndices().size() != 2 ||
        !matchPattern(store.getIndices()[0], m_Zero()) || store.getIndices()[1] != loop.getInductionVar()) continue;
    SmallVector<OpOperand *> consumers;
    SmallVector<dsa::FillOp> fills;
    for (Operation *user : output.getUsers()) {
      if (user == store.getOperation()) continue;
      if (auto fill = dyn_cast<dsa::FillOp>(user); fill && dominance.dominates(fill, loop)) {
        fills.push_back(fill); continue;
      }
      if (auto gather = dyn_cast<dsa::GatherRowsOp>(user); gather && gather.getRowOffsets() == output &&
          gather.getRows() == loop.getUpperBound() && dominance.dominates(loop, gather)) {
        consumers.push_back(&gather->getOpOperand(1)); continue;
      }
      if (auto gather = dyn_cast<dsa::GroupGatherRowsOp>(user); gather && gather.getRowOffsets() == output &&
          gather.getRows() == loop.getUpperBound() && dominance.dominates(loop, gather)) {
        consumers.push_back(&gather->getOpOperand(1)); continue;
      }
      if (auto plan = dyn_cast<dsa::GatherPlanOp>(user); plan && plan.getRowOffsets() == output &&
          plan.getRows() == loop.getUpperBound() && dominance.dominates(loop, plan)) {
        consumers.push_back(&plan->getOpOperand(0)); continue;
      }
      eligible = false; break;
    }
    if (!eligible || consumers.empty()) continue;
    bool reused = false;
    for (auto previous : candidates) {
      if (previous.loop->getBlock() != loop->getBlock() || previous.store.getMemref().getType() != type ||
          previous.loop.getUpperBound() != loop.getUpperBound()) continue;
      // A local coordinate tensor may be read by both expressions, but no
      // intervening mutation or escaping alias may change its contents.
      StorageAnalysis storage(function);
      bool stable = llvm::all_of(sources, [&](Value source) {
        Value origin = storage.uniqueOrigin(source);
        return origin && origin.getDefiningOp<memref::AllocaOp>() &&
               storage.aliases(origin).complete &&
               storage.readStable(source, previous.loop, loop);
      });
      if (!stable || previous.loop.getBody()->getOperations().size() != loop.getBody()->getOperations().size()) continue;
      IRMapping mapping;
      mapping.map(previous.loop.getInductionVar(), loop.getInductionVar());
      mapping.map(previous.store.getMemref(), output);
      bool equivalent = true;
      for (auto [a, c] : llvm::zip(*previous.loop.getBody(), *loop.getBody())) {
        if (a.getName() != c.getName() || a.getAttrs() != c.getAttrs() || a.getResultTypes() != c.getResultTypes() ||
            a.getNumOperands() != c.getNumOperands() || a.getNumRegions() || c.getNumRegions()) {
          equivalent = false; break;
        }
        for (auto [lhs, rhs] : llvm::zip(a.getOperands(), c.getOperands()))
          if (mapping.lookupOrDefault(lhs) != rhs) { equivalent = false; break; }
        if (!equivalent) break;
        mapping.map(a.getResults(), c.getResults());
      }
      if (!equivalent) continue;
      for (OpOperand *consumer : consumers) consumer->set(previous.store.getMemref());
      loop.erase();
      for (auto fill : fills) fill.erase();
      allocation.erase();
      reused = changed = true;
      break;
    }
    if (!reused) candidates.push_back({loop, store});
  }
  return changed;
}

bool normalizeLinearIndices(func::FuncOp function) {
  bool changed = false;
  function.walk([&](memref::StoreOp store) {
    Value root = store.getValue();
    if (auto cast = root.getDefiningOp<arith::IndexCastOp>()) root = cast.getIn();
    Type type = root.getType();
    if (!type.isIndex() && !type.isInteger(64)) return;
    auto definition = root.getDefiningOp();
    if (!definition || !isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(definition)) return;
    // Integer add/sub/mul are modular. Divisions and reads remain opaque SSA
    // leaves; cancellation never depends on overflow or runtime index bounds.
    SmallVector<std::pair<Value, APInt>> terms;
    APInt constant(64, 0);
    llvm::SmallPtrSet<Operation *, 16> arithmetic;
    std::function<void(Value, APInt)> collect = [&](Value value, APInt factor) {
      if (factor.isZero()) return;
      IntegerAttr literal;
      if (matchPattern(value, m_Constant(&literal))) {
        constant += literal.getValue().sextOrTrunc(64) * factor;
        return;
      }
      Operation *op = value.getDefiningOp();
      if (op && value.getType() == type) {
        if (auto add = dyn_cast<arith::AddIOp>(op)) {
          arithmetic.insert(op); collect(add.getLhs(), factor); collect(add.getRhs(), factor); return;
        }
        if (auto sub = dyn_cast<arith::SubIOp>(op)) {
          arithmetic.insert(op); collect(sub.getLhs(), factor); collect(sub.getRhs(), -factor); return;
        }
        if (auto mul = dyn_cast<arith::MulIOp>(op)) {
          if (matchPattern(mul.getRhs(), m_Constant(&literal))) {
            arithmetic.insert(op); collect(mul.getLhs(), factor * literal.getValue().sextOrTrunc(64)); return;
          }
          if (matchPattern(mul.getLhs(), m_Constant(&literal))) {
            arithmetic.insert(op); collect(mul.getRhs(), factor * literal.getValue().sextOrTrunc(64)); return;
          }
        }
      }
      for (auto &term : terms)
        if (term.first == value) { term.second += factor; return; }
      terms.emplace_back(value, factor);
    };
    collect(root, APInt(64, 1));
    int64_t count = constant.isZero() ? 0 : 1, multiplies = 0;
    for (const auto &term : terms) {
      if (term.second.isZero()) continue;
      ++count;
      multiplies += !term.second.isOne();
    }
    if (std::max<int64_t>(0, count - 1) + multiplies >= int64_t(arithmetic.size())) return;
    OpBuilder b(definition);
    auto literal = [&](const APInt &value) -> Value {
      return b.create<arith::ConstantOp>(store.getLoc(), type, b.getIntegerAttr(type, value));
    };
    Value result;
    for (const auto &term : terms) {
      if (term.second.isZero()) continue;
      Value value = term.first;
      if (!term.second.isOne()) value = b.create<arith::MulIOp>(store.getLoc(), value, literal(term.second));
      result = result ? b.create<arith::AddIOp>(store.getLoc(), result, value) : value;
    }
    if (!constant.isZero()) result = result ? b.create<arith::AddIOp>(store.getLoc(), result, literal(constant)) : literal(constant);
    if (!result) result = literal(APInt(64, 0));
    root.replaceAllUsesWith(result);
    changed = true;
  });
  return changed;
}
} // namespace intent::dsa

namespace intent::dsa {
#define GEN_PASS_DEF_DSANORMALIZEINDICES
#include "Intent/Dialect/DSA/Transforms/Passes.h.inc"

namespace {
struct NormalizeIndicesPass : impl::DSANormalizeIndicesBase<NormalizeIndicesPass> {
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(verifyRealizedProgram(module)))
      return signalPassFailure();
    auto function = *module.getOps<func::FuncOp>().begin();
    LogicalResult result = success();
    if (normalizeLinearIndices(function)) {
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
} // namespace intent::dsa
