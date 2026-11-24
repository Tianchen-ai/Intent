#pragma once
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include <cmath>
namespace intentdsl::cpu {
// The 64 KiB L2-resident per-task budget includes all owners of this candidate
// (giving a 1024-token private score window at 16 packed owners).
inline constexpr int64_t batchScoreSlots = 64 * 1024 / sizeof(float);
struct RepeatedScore {
  mlir::scf::ForOp first, second;
  mlir::Operation *record, *replay;
};
inline std::optional<RepeatedScore> repeatedScore(mlir::Operation &space) {
  using namespace mlir;
  auto &bb = space.getRegion(0).front();
  auto loops = llvm::to_vector(bb.getOps<scf::ForOp>());
  if (loops.size() < 2)
    return {};

  auto nonMarkAttrs = [](Operation *op) {
    SmallVector<NamedAttribute, 4> res;
    for (auto attr : op->getAttrs())
      if (attr.getName() != "cpu.record_score" &&
          attr.getName() != "cpu.replay_score" &&
          attr.getName() != "cpu.owner_common" &&
          attr.getName() != "mojo_row.contiguous_simd" &&
          attr.getName() != "mojo_row.pack_scalars")
        res.push_back(attr);
    return res;
  };

  auto isPureOrRead = [](Operation *op) {
    return isMemoryEffectFree(op) || isa<memref::LoadOp, memref::DimOp>(op) ||
           op->getName().stripDialect() == "slice_read";
  };

  auto isNegInf = [](Value v) {
    auto cst = v ? v.getDefiningOp<arith::ConstantOp>() : arith::ConstantOp();
    auto fa = cst ? dyn_cast<FloatAttr>(cst.getValue()) : FloatAttr();
    return fa && fa.getValue().isNegInfinity();
  };

  auto sameExpr = [&](auto &self, Value x, Value y, Block *bx, Block *by,
                      Value ivX, Value ivY) -> bool {
    if (x == y || (ivX && x == ivX && y == ivY))
      return true;
    auto *p = x ? x.getDefiningOp() : nullptr;
    auto *q = y ? y.getDefiningOp() : nullptr;
    if (!p || !q || p->getName() != q->getName() || p->getNumRegions() != 0 ||
        q->getNumRegions() != 0 || p->getNumResults() != 1 ||
        q->getNumResults() != 1 || x.getType() != y.getType() ||
        nonMarkAttrs(p) != nonMarkAttrs(q) ||
        p->getPropertiesAsAttribute() != q->getPropertiesAsAttribute() ||
        p->getNumOperands() != q->getNumOperands())
      return false;
    if (bx) {
      bool inX = p->getBlock() == bx;
      bool inY = q->getBlock() == by;
      if (inX != inY)
        return false;
      if (!inX && (p->getBlock() != &bb || q->getBlock() != &bb))
        return false;
    } else if (p->getBlock() != &bb || q->getBlock() != &bb) {
      return false;
    }
    if (!isPureOrRead(p) || !isPureOrRead(q))
      return false;
    return llvm::all_of(
        llvm::zip(p->getOperands(), q->getOperands()), [&](auto pair) {
          return self(self, std::get<0>(pair), std::get<1>(pair), bx, by, ivX,
                      ivY);
        });
  };

  for (size_t i = 0; i < loops.size(); ++i) {
    auto a = loops[i];
    if (a.getNumResults() != 1 || !a.getResult(0).getType().isF32() ||
        !matchPattern(a.getStep(), m_One()))
      continue;
    auto peak = a.getBody()
                    ->getTerminator()
                    ->getOperand(0)
                    .getDefiningOp<arith::MaxNumFOp>();
    if (!peak)
      continue;
    Value iterArg = a.getRegionIterArgs()[0];
    Value candA = peak.getLhs() == iterArg
                      ? peak.getRhs()
                      : (peak.getRhs() == iterArg ? peak.getLhs() : Value{});
    if (!candA)
      continue;

    Value scoreA;
    Value condA;
    bool invertCond = false;
    if (auto ifA = candA.getDefiningOp<scf::IfOp>()) {
      if (ifA->getBlock() != a.getBody() || ifA.getNumResults() != 1 ||
          ifA.getElseRegion().empty() ||
          !ifA.getThenRegion().front().without_terminator().empty() ||
          (!ifA.getElseRegion().front().without_terminator().empty() &&
           !llvm::hasSingleElement(
               ifA.getElseRegion().front().without_terminator())))
        continue;
      Value tVal = ifA.thenYield().getOperand(0);
      Value eVal = ifA.elseYield().getOperand(0);
      if (isNegInf(eVal)) {
        scoreA = tVal;
        condA = ifA.getCondition();
      } else if (isNegInf(tVal)) {
        scoreA = eVal;
        condA = ifA.getCondition();
        invertCond = true;
      }
    } else if (auto selA = candA.getDefiningOp<arith::SelectOp>()) {
      if (selA->getBlock() != a.getBody())
        continue;
      if (isNegInf(selA.getFalseValue())) {
        scoreA = selA.getTrueValue();
        condA = selA.getCondition();
      } else if (isNegInf(selA.getTrueValue())) {
        scoreA = selA.getFalseValue();
        condA = selA.getCondition();
        invertCond = true;
      }
    } else {
      scoreA = candA;
    }

    auto *producer = scoreA ? scoreA.getDefiningOp() : nullptr;
    if (!producer || producer->getBlock() != a.getBody())
      continue;

    for (size_t j = i + 1; j < loops.size(); ++j) {
      auto b = loops[j];
      if (!matchPattern(b.getStep(), m_One()) ||
          !sameExpr(sameExpr, a.getLowerBound(), b.getLowerBound(), nullptr,
                    nullptr, Value{}, Value{}) ||
          !sameExpr(sameExpr, a.getUpperBound(), b.getUpperBound(), nullptr,
                    nullptr, Value{}, Value{}))
        continue;

      // Every load of the matched expression must see the same input in both folds.
      bool interveningWrite = false;
      for (Operation *op = a; op != b; op = op->getNextNode()) {
        if (op->walk([&](Operation *v) {
                if (v->getNumRegions() != 0 || isPureOrRead(v))
                  return WalkResult::advance();
                return WalkResult::interrupt();
              }).wasInterrupted()) {
          interveningWrite = true;
          break;
        }
      }
      if (interveningWrite)
        break;

      Operation *matchedReplay = nullptr;
      bool ambiguous = false;
      b.getBody()->walk([&](arith::SubFOp sub) {
        if (sub.getRhs() != a.getResult(0))
          return;
        auto *repOp = sub.getLhs().getDefiningOp();
        if (!repOp || repOp->getBlock() != b.getBody()) {
          ambiguous = true;
          return;
        }
        if (sub->getBlock() != b.getBody()) {
          auto ifB = dyn_cast<scf::IfOp>(sub->getParentOp());
          if (!ifB || ifB->getBlock() != b.getBody()) {
            ambiguous = true;
            return;
          }
          if (condA) {
            Region *expectedReg =
                invertCond ? &ifB.getElseRegion() : &ifB.getThenRegion();
            if (sub->getParentRegion() != expectedReg ||
                !sameExpr(sameExpr, condA, ifB.getCondition(), a.getBody(),
                          b.getBody(), a.getInductionVar(),
                          b.getInductionVar())) {
              ambiguous = true;
              return;
            }
          }
        }
        if (!sameExpr(sameExpr, scoreA, sub.getLhs(), a.getBody(), b.getBody(),
                      a.getInductionVar(), b.getInductionVar())) {
          ambiguous = true;
          return;
        }
        if (matchedReplay && matchedReplay != repOp)
          ambiguous = true;
        matchedReplay = repOp;
      });
      if (matchedReplay && !ambiguous)
        return RepeatedScore{a, b, producer, matchedReplay};
    }
  }
  return {};
}
} // namespace intentdsl::cpu
