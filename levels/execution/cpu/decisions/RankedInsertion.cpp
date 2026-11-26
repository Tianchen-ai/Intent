#include "intentdsl/levels/execution/cpu/ir/Owners.h"
#include "intentdsl/levels/execution/cpu/ir/ScoreCache.h"
#include "mlir/IR/IRMapping.h"

namespace intentdsl::cpu {
using namespace mlir;
namespace {
struct InsertionFold {
  memref::LoadOp input;
  arith::CmpFPredicate order;
  unsigned lastSlot;
};

// Each comparator retains the better key in its slot and passes the other
// key to the next comparator. For sorted initial keys, the last slot remains
// the weakest. Also prove that every other carry is unchanged on a miss.
std::optional<InsertionFold> insertionFold(scf::ForOp loop,
                                           const CallFacts &facts) {
  auto args = loop.getRegionIterArgs();
  if (args.size() < 2 || !matchPattern(loop.getLowerBound(), m_Zero()) ||
      !matchPattern(loop.getStep(), m_One()))
    return {};
  auto yielded = loop.getBody()->getTerminator()->getOperands();
  memref::LoadOp input;
  arith::CmpFOp prev;
  llvm::DenseSet<Value> guards;
  FloatAttr prevInit;
  unsigned lastSlot = 0;
  for (Operation &op : loop.getBody()->without_terminator()) {
    if (auto ld = dyn_cast<memref::LoadOp>(op)) {
      auto arg = dyn_cast<BlockArgument>(ld.getMemRef());
      if (input || !arg ||
          arg.getOwner() != &loop->getParentOfType<func::FuncOp>().front() ||
          !ld.getType().isF32() || ld.getIndices().empty() ||
          ld.getIndices().back() != loop.getInductionVar() ||
          !llvm::all_of(ld.getIndices().drop_back(), [&](Value x) {
            return loop.isDefinedOutsideOfLoop(x);
          }))
        return {};
      if (arg.getArgNumber() >= facts.arguments.size())
        return {};
      auto *rec = facts.arguments[arg.getArgNumber()].getAsObject();
      auto strides = rec ? rec->getArray("strides") : nullptr;
      if (!strides || strides->empty() || strides->back().getAsInteger() != 1)
        return {};
      input = ld;
      continue;
    }
    if (auto s = dyn_cast<arith::SelectOp>(op)) {
      if (!guards.contains(s.getCondition()))
        return {};
      continue;
    }
    if (auto cast = dyn_cast<arith::IndexCastOp>(op)) {
      if (cast.getIn() != loop.getInductionVar())
        return {};
      continue;
    }
    auto cmp = dyn_cast<arith::CmpFOp>(op);
    if (!cmp || !input ||
        (cmp.getPredicate() != arith::CmpFPredicate::OGT &&
         cmp.getPredicate() != arith::CmpFPredicate::OLT))
      return {};
    auto slot = dyn_cast<BlockArgument>(cmp.getRhs());
    if (!slot || slot.getOwner() != loop.getBody() || slot.getArgNumber() == 0)
      return {};
    lastSlot = slot.getArgNumber() - 1;
    auto init = loop.getInitArgs()[lastSlot].getDefiningOp<arith::ConstantOp>();
    auto f = init ? dyn_cast<FloatAttr>(init.getValue()) : FloatAttr{};
    if (!f || f.getValue().isNaN())
      return {};
    if (!prev) {
      if (cmp.getLhs() != input.getResult())
        return {};
    } else {
      auto loser = cmp.getLhs().getDefiningOp<arith::SelectOp>();
      if (!loser || loser.getCondition() != prev.getResult() ||
          loser.getTrueValue() != prev.getRhs() ||
          loser.getFalseValue() != prev.getLhs() ||
          cmp.getPredicate() != prev.getPredicate())
        return {};
      auto initialOrder = prevInit.getValue().compare(f.getValue());
      if (initialOrder == (cmp.getPredicate() == arith::CmpFPredicate::OGT
                               ? llvm::APFloat::cmpLessThan
                               : llvm::APFloat::cmpGreaterThan))
        return {};
    }
    auto winner = yielded[lastSlot].getDefiningOp<arith::SelectOp>();
    if (!winner || winner.getCondition() != cmp.getResult() ||
        winner.getTrueValue() != cmp.getLhs() || winner.getFalseValue() != slot)
      return {};
    guards.insert(cmp.getResult());
    prev = cmp;
    prevInit = f;
  }
  if (guards.size() < 2)
    return {};
  for (auto [next, old] : llvm::zip(yielded, args)) {
    Value v = next;
    while (auto s = v.getDefiningOp<arith::SelectOp>()) {
      if (!guards.contains(s.getCondition()))
        break;
      v = s.getFalseValue();
    }
    if (v != old)
      return {};
  }
  return InsertionFold{input, prev.getPredicate(), lastSlot};
}

void screen(scf::ForOp loop, InsertionFold fold, int64_t count, int64_t width,
            RewriterBase &rw) {
  OpBuilder::InsertionGuard guard(rw);
  rw.setInsertionPoint(loop);
  Location loc = loop.getLoc();
  Value lo = loop.getLowerBound(), hi = loop.getUpperBound(),
        one = loop.getStep();
  Value widthVal = rw.create<arith::ConstantIndexOp>(loc, width);
  Value stop = rw.create<arith::ConstantIndexOp>(loc, count - count % width);
  SmallVector<Value> coords(fold.input.getIndices());
  auto keyAt = [&](OpBuilder &b, Value iv) -> Value {
    coords.back() = iv;
    return b.create<memref::LoadOp>(loc, fold.input.getMemRef(), coords);
  };
  auto replay = [&](OpBuilder &b, Value beg, Value end, ValueRange initial) {
    return b.create<scf::ForOp>(
        loc, beg, end, one, initial,
        [&](OpBuilder &ib, Location, Value iv, ValueRange state) {
          auto x = keyAt(ib, iv);
          auto update = [&](OpBuilder &tb, Location) {
            IRMapping map;
            map.map(loop.getInductionVar(), iv);
            map.map(loop.getRegionIterArgs(), state);
            map.map(fold.input.getResult(), x);
            for (Operation &op : *loop.getBody())
              if (&op != fold.input.getOperation())
                tb.clone(op, map);
          };

          Value hit = ib.create<arith::CmpFOp>(loc, fold.order, x,
                                               state[fold.lastSlot]);
          auto step = ib.create<scf::IfOp>(
              loc, hit, update, [&](OpBuilder &eb, Location) {
                eb.create<scf::YieldOp>(loc, state);
              });
          ib.create<scf::YieldOp>(loc, step.getResults());
        });
  };
  // On monotonically improving input every key enters the chain. A probe
  // of one full vector selects the original recurrence for that row.
  Value dense, prev = keyAt(rw, lo);
  for (int64_t lane = 1; lane < width; ++lane) {
    Value x = keyAt(rw, rw.create<arith::ConstantIndexOp>(loc, lane));
    Value better = rw.create<arith::CmpFOp>(loc, fold.order, x, prev);
    dense = dense ? rw.create<arith::AndIOp>(loc, dense, better).getResult()
                  : better;
    prev = x;
  }
  auto row = rw.create<scf::IfOp>(loc, loop.getResultTypes(), dense, true);
  rw.setInsertionPointToStart(row.thenBlock());
  auto *ordered = rw.clone(*loop.getOperation());
  rw.create<scf::YieldOp>(loc, ordered->getResults());
  rw.setInsertionPointToStart(row.elseBlock());
  auto panels = rw.create<scf::ForOp>(
      loc, lo, stop, widthVal, loop.getInitArgs(),
      [&](OpBuilder &b, Location, Value start, ValueRange state) {
        Value hit;
        for (int64_t lane = 0; lane < width; ++lane) {
          Value iv = start;
          if (lane)
            iv = b.create<arith::AddIOp>(
                loc, start, b.create<arith::ConstantIndexOp>(loc, lane));
          Value qualifies = b.create<arith::CmpFOp>(
              loc, fold.order, keyAt(b, iv), state[fold.lastSlot]);
          hit = hit ? b.create<arith::OrIOp>(loc, hit, qualifies).getResult()
                    : qualifies;
        }
        auto filtered = b.create<scf::IfOp>(
            loc, hit,
            [&](OpBuilder &tb, Location) {
              auto end = tb.create<arith::AddIOp>(loc, start, widthVal);
              auto inner = replay(tb, start, end, state);
              tb.create<scf::YieldOp>(loc, inner.getResults());
            },
            [&](OpBuilder &eb, Location) {
              eb.create<scf::YieldOp>(loc, state);
            });
        b.create<scf::YieldOp>(loc, filtered.getResults());
      });
  auto result =
      count % width ? replay(rw, stop, hi, panels.getResults()) : panels;
  rw.create<scf::YieldOp>(loc, result.getResults());
  rw.replaceOp(loop, row.getResults());
}
} // namespace
void screenRankedInsertions(Block &body, RewriterBase &rw,
                            const compilerinfra::TargetProfile &p) {
  auto *machine = p.services.get<RowMachine>();
  auto *facts = p.services.get<CallFacts>();
  if (!machine || !facts)
    return;
  int64_t width = machine->vectorBits / 32;
  if (width < 2)
    return;
  for (auto loop : llvm::to_vector(body.getOps<scf::ForOp>())) {
    auto count = detail::staticIndexValue(loop.getUpperBound(), *facts);
    if (!count || *count < width)
      continue;
    auto plan = insertionFold(loop, *facts);
    if (plan)
      screen(loop, *plan, *count, width, rw);
  }
}
} // namespace intentdsl::cpu
