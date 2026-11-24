#pragma once
#include "compilerinfra/Proofs.h"
#include "mlir/Analysis/AliasAnalysis/LocalAliasAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/JSON.h"
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

  auto fn = space.getParentOfType<func::FuncOp>();
  SmallVector<int64_t, 8> argPublicSlot;
  SmallVector<std::string, 8> argDirection;
  bool hasAbiFacts = false;
  if (fn) {
    bool projected = fn->hasAttr("kir.iteration");
    auto abiAttr = fn->getAttrOfType<StringAttr>(
        projected ? "kir.public_abi" : "kir.abi");
    if (abiAttr) {
      if (auto parsed = llvm::json::parse(abiAttr.getValue())) {
        if (auto *slots = parsed->getAsArray()) {
          bool ok = true;
          argPublicSlot.resize(fn.getNumArguments(), -1);
          argDirection.resize(fn.getNumArguments());
          for (unsigned k = 0; k < fn.getNumArguments(); ++k) {
            int64_t pubIdx = int64_t(k);
            if (projected) {
              auto src = fn.getArgAttrOfType<IntegerAttr>(k, "kir.source");
              if (!src) {
                ok = false;
                break;
              }
              pubIdx = src.getInt();
            }
            if (pubIdx < 0 || size_t(pubIdx) >= slots->size()) {
              ok = false;
              break;
            }
            auto *obj = (*slots)[pubIdx].getAsObject();
            auto dir = obj ? obj->getString("direction") : std::nullopt;
            if (!dir) {
              ok = false;
              break;
            }
            argPublicSlot[k] = pubIdx;
            argDirection[k] = dir->str();
          }
          hasAbiFacts = ok;
        }
      } else {
        llvm::consumeError(parsed.takeError());
      }
    }
  }

  auto resolveFuncArg = [&](auto &self, Value v) -> std::optional<unsigned> {
    if (!v || !fn)
      return std::nullopt;
    if (auto ba = dyn_cast<BlockArgument>(v)) {
      if (ba.getOwner() == &fn.front())
        return ba.getArgNumber();
      if (Operation *parent = ba.getOwner()->getParentOp()) {
        unsigned nArgs = ba.getOwner()->getNumArguments();
        unsigned nOps = parent->getNumOperands();
        if (nArgs == nOps && ba.getArgNumber() < nOps)
          return self(self, parent->getOperand(ba.getArgNumber()));
        if (nArgs == nOps + 1 && ba.getArgNumber() >= 1)
          return self(self, parent->getOperand(ba.getArgNumber() - 1));
      }
      return std::nullopt;
    }
    auto *def = v.getDefiningOp();
    if (isa_and_nonnull<memref::CastOp, memref::SubViewOp,
                        memref::ReinterpretCastOp>(def))
      return self(self, def->getOperand(0));
    return std::nullopt;
  };

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
    auto stripped = op->getName().stripDialect();
    return isMemoryEffectFree(op) || isa<memref::LoadOp, memref::DimOp>(op) ||
           stripped == "slice_read" || stripped == "lane_splat";
  };

  auto collectExprReadViews =
      [&](ArrayRef<Value> roots,
          llvm::SmallDenseSet<Value, 8> &rawReadViews,
          llvm::SmallDenseSet<unsigned, 8> &resolvedReadArgs,
          bool &allResolved) -> bool {
    llvm::SmallDenseSet<Value, 16> visited;
    SmallVector<Value, 16> work;
    for (Value r : roots)
      if (r)
        work.push_back(r);
    while (!work.empty()) {
      Value cur = work.pop_back_val();
      if (!visited.insert(cur).second)
        continue;
      if (isa<MemRefType>(cur.getType())) {
        rawReadViews.insert(cur);
        if (auto idx = resolveFuncArg(resolveFuncArg, cur))
          resolvedReadArgs.insert(*idx);
        else
          allResolved = false;
      }
      if (auto ba = dyn_cast<BlockArgument>(cur)) {
        if (auto ifParent = dyn_cast<scf::IfOp>(ba.getOwner()->getParentOp())) {
          (void)ifParent;
          return false;
        }
        continue;
      }
      Operation *def = cur.getDefiningOp();
      if (!def)
        continue;
      if (auto ifOp = dyn_cast<scf::IfOp>(def)) {
        if (ifOp.getElseRegion().empty())
          return false;
        work.push_back(ifOp.getCondition());
        for (Operation &sub : ifOp.getThenRegion().front())
          for (Value in : sub.getOperands())
            work.push_back(in);
        for (Operation &sub : ifOp.getElseRegion().front())
          for (Value in : sub.getOperands())
            work.push_back(in);
        continue;
      }
      if (def->getNumRegions() != 0 || !isPureOrRead(def))
        return false;
      for (Value in : def->getOperands())
        work.push_back(in);
    }
    return true;
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

      Operation *matchedReplay = nullptr;
      Value condB;
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
          condB = ifB.getCondition();
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
      if (!matchedReplay || ambiguous)
        continue;

      // Prove complete write-effect and alias safety across the entire [a, b]
      // interval (including inside a, between a and b, and inside b itself):
      // every memory read in the score expression, guard predicates, and loop
      // bounds must observe identical memory contents in both folds.
      llvm::SmallDenseSet<Value, 8> rawReadViews;
      llvm::SmallDenseSet<unsigned, 8> scoreReadArgs;
      bool allReadsResolved = true;
      if (!collectExprReadViews(
              {scoreA, condA, a.getLowerBound(), a.getUpperBound(),
               matchedReplay->getResult(0), condB, b.getLowerBound(),
               b.getUpperBound()},
              rawReadViews, scoreReadArgs, allReadsResolved))
        continue;
      // Reject reads from buffers defined inside [a, b] (e.g. local allocas).
      bool localReadBuf = llvm::any_of(rawReadViews, [&](Value v) {
        auto *def = v.getDefiningOp();
        return def && !def->isBeforeInBlock(a);
      });
      if (localReadBuf)
        continue;

      auto disjointPublicAbiArgs = [&](Value lhs, Value rhs) -> bool {
        if (!hasAbiFacts || !lhs || !rhs || lhs == rhs)
          return false;
        auto lIdx = resolveFuncArg(resolveFuncArg, lhs);
        auto rIdx = resolveFuncArg(resolveFuncArg, rhs);
        if (!lIdx || !rIdx || *lIdx == *rIdx || *lIdx >= argDirection.size() ||
            *rIdx >= argDirection.size() ||
            argPublicSlot[*lIdx] == argPublicSlot[*rIdx])
          return false;
        const auto &ld = argDirection[*lIdx], &rd = argDirection[*rIdx];
        return (ld == "In" && rd == "Out") || (ld == "Out" && rd == "In");
      };

      struct PublicAbiAliasModel {
        std::function<bool(Value, Value)> disjoint;
        AliasResult alias(Value lhs, Value rhs) {
          return disjoint(lhs, rhs) ? AliasResult::NoAlias
                                    : AliasResult::MayAlias;
        }
        ModRefResult getModRef(Operation *op, Value location) {
          auto mem = dyn_cast<MemoryEffectOpInterface>(op);
          if (!mem)
            return ModRefResult::getModAndRef();
          SmallVector<MemoryEffects::EffectInstance, 4> effects;
          mem.getEffects(effects);
          ModRefResult res = ModRefResult::getNoModRef();
          for (const auto &e : effects) {
            Value v = e.getValue();
            if (!v || !disjoint(v, location)) {
              if (isa<MemoryEffects::Write, MemoryEffects::Free,
                      MemoryEffects::Allocate>(e.getEffect()))
                res = res.merge(ModRefResult::getMod());
              if (isa<MemoryEffects::Read>(e.getEffect()))
                res = res.merge(ModRefResult::getRef());
            }
          }
          return res;
        }
      };

      AliasAnalysis aa(&space);
      aa.addAnalysisImplementation(LocalAliasAnalysis());
      aa.addAnalysisImplementation(PublicAbiAliasModel{disjointPublicAbiArgs});
      SmallVector<Value, 8> readLocations(rawReadViews.begin(),
                                          rawReadViews.end());
      if (!compilerinfra::proveValueReplay(producer, matchedReplay,
                                           readLocations, &aa)
               .proven)
        continue;

      return RepeatedScore{a, b, producer, matchedReplay};
    }
  }
  return {};
}
} // namespace intentdsl::cpu
