#pragma once
#include "intentdsl/levels/algorithm/kir/ir/FastMath.h"
#include "intentdsl/levels/execution/block/ir/Owners.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
namespace intentdsl::block {
inline bool rectangularTokenLoad(mlir::Operation &op, mlir::scf::ForOp loop) {
  if (op.getName().stripDialect() != "slice_read")
    return false;
  auto coords = op.getOperands().drop_front();
  auto axis = op.getAttrOfType<mlir::IntegerAttr>("axis");
  if (coords.empty() || !axis || axis.getInt() != int64_t(coords.size()) ||
      coords.back() != loop.getInductionVar())
    return false;
  return llvm::all_of(coords.drop_back(), [&](mlir::Value v) {
    return loop.isDefinedOutsideOfLoop(v);
  });
}
inline bool supportedPanelCmpI(mlir::arith::CmpIOp cmp) {
  using P = mlir::arith::CmpIPredicate;
  switch (cmp.getPredicate()) {
  case P::eq:
  case P::ne:
  case P::slt:
  case P::sle:
  case P::sgt:
  case P::sge:
    return true;
  default:
    return false;
  }
}
inline bool panelFoldBlock(mlir::Block &block, bool topLevel) {
  for (auto &v : block.without_terminator()) {
    if (auto br = mlir::dyn_cast<mlir::scf::IfOp>(v)) {
      if (br.getNumResults() == 0 || br.getElseRegion().empty() ||
          !panelFoldBlock(*br.thenBlock(), false) ||
          !panelFoldBlock(*br.elseBlock(), false))
        return false;
      continue;
    }
    if (v.getNumRegions() != 0)
      return false;
    auto kind = v.getName().stripDialect();
    if (topLevel && (kind == "slice_read" || kind == "lane_sum" ||
                     llvm::StringSwitch<bool>(v.getName().getStringRef())
                         .Cases("memref.load", "arith.floordivsi",
                                "arith.remsi", true)
                         .Default(false)))
      continue;
    if (auto cmp = mlir::dyn_cast<mlir::arith::CmpIOp>(v)) {
      if (!supportedPanelCmpI(cmp))
        return false;
      continue;
    }
    if (kind != "lane_splat" &&
        !llvm::StringSwitch<bool>(v.getName().getStringRef())
             .Cases("arith.index_cast", "arith.constant", true)
             .Cases("arith.sitofp", "arith.fptosi", "arith.cmpf", true)
             .Cases("arith.extf", "arith.truncf", "arith.maxnumf", true)
             .Cases("arith.minnumf", "arith.minimumf", "arith.minsi", true)
             .Cases("arith.addf", "arith.mulf", "arith.subf", "arith.divf", true)
             .Cases("arith.addi", "arith.subi", "arith.muli", true)
             .Cases("arith.andi", "arith.ori", "arith.xori", true)
             .Cases("arith.select", "math.exp", "math.exp2", "math.log",
                    "math.absf", true)
             .Default(false))
      return false;
  }
  return true;
}
inline int64_t foldBodyLanes(mlir::Operation *o, const CallFacts &facts) {
  int64_t lanes = 1;
  for (auto loop : o->getRegion(0).front().getOps<mlir::scf::ForOp>()) {
    if (!loop->hasAttr("kir.reassociate"))
      continue;
    loop.walk([&](mlir::Operation *inst) {
      for (auto v : inst->getResults())
        if (auto axis = intentdsl::kir::memberAxis(v))
          if (auto *sh = facts.shapeOf(axis->first))
            lanes = std::max<int64_t>(
                lanes,
                llvm::PowerOf2Ceil(std::max<int64_t>(
                    1, (*sh)[axis->second].getAsInteger().value_or(1))));
    });
  }
  return lanes;
}
inline int64_t maxFoldExtent(mlir::Operation *o, const CallFacts &facts) {
  int64_t maxExt = 0;
  for (auto loop : o->getRegion(0).front().getOps<mlir::scf::ForOp>()) {
    if (!loop->hasAttr("kir.reassociate"))
      continue;
    mlir::Value ub = loop.getUpperBound();
    if (auto c = ub.getDefiningOp<mlir::arith::ConstantOp>())
      if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue()))
        maxExt = std::max<int64_t>(maxExt, ia.getInt());
    if (auto d = ub.getDefiningOp<mlir::memref::DimOp>();
        d && d.getConstantIndex())
      if (auto *sh = facts.shapeOf(d.getSource()))
        maxExt = std::max<int64_t>(
            maxExt, (*sh)[*d.getConstantIndex()].getAsInteger().value_or(0));
  }
  return maxExt;
}
struct OrderedSelectionInfo {
  unsigned k = 0;
  mlir::Value candidateVal;
  mlir::Value candidateIdx;
  llvm::SmallPtrSet<mlir::Operation *, 32> chainOps;
};
inline bool matchOrderedSelectionChain(mlir::scf::ForOp loop,
                                       OrderedSelectionInfo *out = nullptr) {
  if (loop->hasAttr("kir.reassociate") ||
      !mlir::matchPattern(loop.getLowerBound(), mlir::m_Zero()) ||
      !mlir::matchPattern(loop.getStep(), mlir::m_One()) ||
      !loop.isDefinedOutsideOfLoop(loop.getUpperBound()))
    return false;
  unsigned total = loop.getNumRegionIterArgs();
  if (total < 2 || total % 2 != 0)
    return false;
  unsigned k = total / 2;
  if (k < 1 || k > 16)
    return false;
  auto initArgs = loop.getInitArgs();
  for (unsigned s = 0; s < k; ++s) {
    auto fc = initArgs[s].getDefiningOp<mlir::arith::ConstantOp>();
    auto fa = fc ? mlir::dyn_cast<mlir::FloatAttr>(fc.getValue())
                 : mlir::FloatAttr();
    if (!fa || !fa.getType().isF32() || !fa.getValue().isInfinity() ||
        !fa.getValue().isNegative())
      return false;
    auto ic = initArgs[k + s].getDefiningOp<mlir::arith::ConstantOp>();
    auto ia = ic ? mlir::dyn_cast<mlir::IntegerAttr>(ic.getValue())
                 : mlir::IntegerAttr();
    if (!ia || !ia.getType().isInteger(32) || ia.getInt() != -1)
      return false;
  }
  mlir::Block *body = loop.getBody();
  auto condOf = [&](mlir::Value v) -> mlir::Value {
    auto *def = v.getDefiningOp();
    if (!def || def->getBlock() != body)
      return {};
    if (auto sel = mlir::dyn_cast<mlir::arith::SelectOp>(def))
      return sel.getCondition();
    if (auto br = mlir::dyn_cast<mlir::scf::IfOp>(def);
        br && br.getNumResults() == 1 && !br.getElseRegion().empty() &&
        br.thenBlock()->without_terminator().empty() &&
        br.elseBlock()->without_terminator().empty())
      return br.getCondition();
    return {};
  };
  auto matchSel = [&](mlir::Value v, mlir::Value cond, mlir::Value &tv,
                      mlir::Value &fv, mlir::Operation *&opOut) -> bool {
    auto *def = v.getDefiningOp();
    if (!def || def->getBlock() != body)
      return false;
    if (auto sel = mlir::dyn_cast<mlir::arith::SelectOp>(def)) {
      if (sel.getCondition() != cond)
        return false;
      tv = sel.getTrueValue();
      fv = sel.getFalseValue();
      opOut = sel;
      return true;
    }
    if (auto br = mlir::dyn_cast<mlir::scf::IfOp>(def);
        br && br.getNumResults() == 1 && !br.getElseRegion().empty() &&
        br.getCondition() == cond &&
        br.thenBlock()->without_terminator().empty() &&
        br.elseBlock()->without_terminator().empty()) {
      tv = br.thenBlock()->getTerminator()->getOperand(0);
      fv = br.elseBlock()->getTerminator()->getOperand(0);
      opOut = br;
      return true;
    }
    return false;
  };
  auto *yield = body->getTerminator();
  auto iterArgs = loop.getRegionIterArgs();
  mlir::Value cond0 = condOf(yield->getOperand(0));
  auto cmp0 = cond0 ? cond0.getDefiningOp<mlir::arith::CmpFOp>()
                    : mlir::arith::CmpFOp();
  if (!cmp0 || cmp0->getBlock() != body ||
      cmp0.getPredicate() != mlir::arith::CmpFPredicate::OGT ||
      cmp0.getRhs() != iterArgs[0])
    return false;
  mlir::Value candVal0 = cmp0.getLhs();
  mlir::Value tv0, fv0, ti0, fi0;
  mlir::Operation *opNv0 = nullptr, *opNi0 = nullptr;
  if (!matchSel(yield->getOperand(0), cond0, tv0, fv0, opNv0) ||
      tv0 != candVal0 || fv0 != iterArgs[0] ||
      !matchSel(yield->getOperand(k), cond0, ti0, fi0, opNi0) ||
      fi0 != iterArgs[k])
    return false;
  auto idxCast = ti0.getDefiningOp<mlir::arith::IndexCastOp>();
  if (!idxCast || idxCast->getBlock() != body ||
      idxCast.getIn() != loop.getInductionVar() ||
      !idxCast.getType().isInteger(32))
    return false;
  llvm::SmallPtrSet<mlir::Operation *, 32> chainOps;
  chainOps.insert(idxCast);
  mlir::Value curVal = candVal0, curIdx = ti0;
  for (unsigned t = 0; t < k; ++t) {
    mlir::Value ct = condOf(yield->getOperand(t));
    auto cmpt = ct ? ct.getDefiningOp<mlir::arith::CmpFOp>()
                   : mlir::arith::CmpFOp();
    if (!cmpt || cmpt->getBlock() != body ||
        cmpt.getPredicate() != mlir::arith::CmpFPredicate::OGT ||
        cmpt.getLhs() != curVal || cmpt.getRhs() != iterArgs[t])
      return false;
    mlir::Value tv, fv, ti, fi;
    mlir::Operation *opNv = nullptr, *opNi = nullptr;
    if (!matchSel(yield->getOperand(t), ct, tv, fv, opNv) || tv != curVal ||
        fv != iterArgs[t] ||
        !matchSel(yield->getOperand(k + t), ct, ti, fi, opNi) || ti != curIdx ||
        fi != iterArgs[k + t])
      return false;
    chainOps.insert(cmpt);
    chainOps.insert(opNv);
    chainOps.insert(opNi);
    if (t + 1 < k) {
      mlir::Value nextCond = condOf(yield->getOperand(t + 1));
      auto nextCmp = nextCond ? nextCond.getDefiningOp<mlir::arith::CmpFOp>()
                              : mlir::arith::CmpFOp();
      if (!nextCmp || nextCmp->getBlock() != body)
        return false;
      mlir::Value nextVal = nextCmp.getLhs();
      mlir::Value ntv, nfv, yti, yfi, nti, nfi;
      mlir::Operation *opDispV = nullptr, *opNextNi = nullptr,
                      *opDispI = nullptr;
      if (!matchSel(nextVal, ct, ntv, nfv, opDispV) || ntv != iterArgs[t] ||
          nfv != curVal ||
          !matchSel(yield->getOperand(k + t + 1), nextCond, yti, yfi,
                    opNextNi) ||
          !matchSel(yti, ct, nti, nfi, opDispI) || nti != iterArgs[k + t] ||
          nfi != curIdx)
        return false;
      chainOps.insert(opDispV);
      chainOps.insert(opDispI);
      curVal = nextVal;
      curIdx = yti;
    }
  }
  auto fn = loop->getParentOfType<mlir::func::FuncOp>();
  bool hasLoad = false;
  for (mlir::Operation &inst : body->without_terminator()) {
    if (chainOps.contains(&inst))
      continue;
    if (inst.getNumRegions() != 0)
      return false;
    for (mlir::Value opnd : inst.getOperands())
      if (llvm::is_contained(iterArgs, opnd))
        return false;
    if (auto ld = mlir::dyn_cast<mlir::memref::LoadOp>(inst)) {
      auto arg = mlir::dyn_cast<mlir::BlockArgument>(ld.getMemRef());
      if (!fn || !arg || arg.getOwner() != &fn.front() ||
          ld.getIndices().empty() ||
          ld.getIndices().back() != loop.getInductionVar())
        return false;
      for (mlir::Value idx : ld.getIndices().drop_back())
        if (!loop.isDefinedOutsideOfLoop(idx))
          return false;
      hasLoad = true;
      continue;
    }
    if (auto cmpi = mlir::dyn_cast<mlir::arith::CmpIOp>(inst)) {
      if (!supportedPanelCmpI(cmpi))
        return false;
      continue;
    }
    if (!llvm::StringSwitch<bool>(inst.getName().getStringRef())
             .Cases("arith.index_cast", "arith.constant", true)
             .Cases("arith.sitofp", "arith.fptosi", "arith.cmpf", true)
             .Cases("arith.extf", "arith.truncf", "arith.maxnumf", true)
             .Cases("arith.minnumf", "arith.minimumf", "arith.minsi", true)
             .Cases("arith.addf", "arith.mulf", "arith.subf", "arith.divf", true)
             .Cases("arith.addi", "arith.subi", "arith.muli", true)
             .Cases("arith.andi", "arith.ori", "arith.xori", true)
             .Cases("arith.select", "math.exp", "math.exp2", "math.log",
                    "math.absf", true)
             .Default(false))
      return false;
  }
  if (!hasLoad)
    return false;
  if (out) {
    out->k = k;
    out->candidateVal = candVal0;
    out->candidateIdx = ti0;
    out->chainOps = std::move(chainOps);
  }
  return true;
}
inline bool panelFolds(mlir::Operation *o,
                       const compilerinfra::TargetProfile &p, int64_t panel) {
  auto machine = p.services.get<OwnerMachine>();
  if (!machine || machine->parallelMembers < 16 || panel < 16 ||
      panel > 4096 || !llvm::isPowerOf2_64(panel))
    return false;
  if (panel > 512) {
    auto facts = p.services.get<CallFacts>();
    if (!facts || foldBodyLanes(o, *facts) > 1)
      return false;
    int64_t maxExt = maxFoldExtent(o, *facts);
    if (maxExt > 0 && panel > int64_t(llvm::PowerOf2Ceil(maxExt)))
      return false;
  }
  bool matched = false;
  auto &body = o->getRegion(0).front();
  for (auto loop : body.getOps<mlir::scf::ForOp>()) {
    if (!loop->hasAttr("kir.reassociate"))
      continue;
    if (!mlir::matchPattern(loop.getStep(), mlir::m_One()))
      return false;
    for (auto [idx, v] :
         llvm::enumerate(loop.getBody()->getTerminator()->getOperands())) {
      auto *op = v.getDefiningOp();
      if (!op || !mlir::isa<mlir::arith::MaxNumFOp, mlir::arith::AddFOp>(op) ||
          op->getOperand(0) != loop.getRegionIterArgs()[idx])
        return false;
    }
    if (!panelFoldBlock(*loop.getBody(), true))
      return false;
    matched = true;
  }
  return matched;
}
inline bool hasReassociativeNormalization(mlir::Operation &space) {
  if (space.getNumRegions() == 0 || space.getRegion(0).empty())
    return false;
  using FMF = mlir::arith::FastMathFlags;
  unsigned reassocLoops = 0;
  bool hasReassocExpShift = false;
  for (auto loop : space.getRegion(0).front().getOps<mlir::scf::ForOp>()) {
    if (!loop->hasAttr("kir.reassociate"))
      continue;
    bool combinersReassoc = llvm::all_of(
        loop.getBody()->getTerminator()->getOperands(), [](mlir::Value y) {
          auto *comb = y.getDefiningOp();
          return comb && intentdsl::kir::hasFastMath(comb, FMF::reassoc);
        });
    if (!combinersReassoc)
      continue;
    ++reassocLoops;
    loop.walk([&](mlir::Operation *op) {
      if (!mlir::isa<mlir::math::ExpOp, mlir::math::Exp2Op>(op) ||
          !intentdsl::kir::hasFastMath(op, FMF::reassoc))
        return;
      mlir::Value arg = op->getOperand(0);
      if (auto mul = arg.getDefiningOp<mlir::arith::MulFOp>();
          mul && intentdsl::kir::hasFastMath(mul, FMF::reassoc)) {
        if (mul.getLhs().getDefiningOp<mlir::arith::SubFOp>())
          arg = mul.getLhs();
        else if (mul.getRhs().getDefiningOp<mlir::arith::SubFOp>())
          arg = mul.getRhs();
      }
      if (auto sub = arg.getDefiningOp<mlir::arith::SubFOp>();
          sub && intentdsl::kir::hasFastMath(sub, FMF::reassoc))
        hasReassocExpShift = true;
    });
  }
  return reassocLoops >= 2 && hasReassocExpShift;
}
} // namespace intentdsl::block
