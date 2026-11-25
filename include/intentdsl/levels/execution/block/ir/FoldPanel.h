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
    if (!mlir::matchPattern(loop.getStep(), mlir::m_One()) ||
        loop.getNumRegionIterArgs() == 0)
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
    if (!loop->hasAttr("kir.reassociate") || loop.getNumRegionIterArgs() == 0)
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

inline std::optional<int64_t>
orderedArgDim(mlir::func::FuncOp fn, const CallFacts *facts, mlir::Value view,
              int64_t axis) {
  auto arg = mlir::dyn_cast<mlir::BlockArgument>(view);
  if (!arg || arg.getOwner() != &fn.front() || axis < 0)
    return std::nullopt;
  if (auto sh = fn.getArgAttrOfType<mlir::DenseI64ArrayAttr>(
          arg.getArgNumber(), "block.shape")) {
    if (axis < int64_t(sh.size()))
      return sh[axis];
  }
  if (facts)
    if (const auto *sh = facts->shapeOf(view);
        sh && size_t(axis) < sh->size())
      return (*sh)[axis].getAsInteger();
  return std::nullopt;
}

inline std::optional<int64_t>
orderedStaticLoopExtent(mlir::scf::ForOp loop, mlir::func::FuncOp fn,
                        const CallFacts *facts) {
  mlir::Value ub = loop.getUpperBound();
  if (auto ci = ub.getDefiningOp<mlir::arith::ConstantIndexOp>())
    return ci.value();
  if (auto co = ub.getDefiningOp<mlir::arith::ConstantOp>())
    if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(co.getValue()))
      return ia.getInt();
  if (auto d = ub.getDefiningOp<mlir::memref::DimOp>();
      d && d.getConstantIndex())
    return orderedArgDim(fn, facts, d.getSource(), *d.getConstantIndex());
  return std::nullopt;
}

inline bool orderedViewsDisjoint(mlir::func::FuncOp fn, const CallFacts *facts,
                                 mlir::Value a, mlir::Value b) {
  auto argA = mlir::dyn_cast<mlir::BlockArgument>(a);
  auto argB = mlir::dyn_cast<mlir::BlockArgument>(b);
  if (!argA || !argB || argA == argB || argA.getOwner() != &fn.front() ||
      argB.getOwner() != &fn.front())
    return false;
  auto agA = fn.getArgAttrOfType<mlir::IntegerAttr>(argA.getArgNumber(),
                                                    "block.alias_group");
  auto agB = fn.getArgAttrOfType<mlir::IntegerAttr>(argB.getArgNumber(),
                                                    "block.alias_group");
  if (agA && agB)
    return agA.getInt() != agB.getInt();
  if (facts && argA.getArgNumber() < facts->arguments.size() &&
      argB.getArgNumber() < facts->arguments.size()) {
    auto *objA = facts->arguments[argA.getArgNumber()].getAsObject();
    auto *objB = facts->arguments[argB.getArgNumber()].getAsObject();
    auto gA = objA ? objA->getInteger("alias_group") : std::nullopt;
    auto gB = objB ? objB->getInteger("alias_group") : std::nullopt;
    if (gA && gB)
      return *gA != *gB;
  }
  return false;
}

inline bool supportedPurePanelArith(mlir::Operation &inst) {
  if (inst.getNumRegions() != 0)
    return false;
  if (auto cmp = mlir::dyn_cast<mlir::arith::CmpIOp>(inst))
    return supportedPanelCmpI(cmp);
  return llvm::StringSwitch<bool>(inst.getName().getStringRef())
      .Cases("arith.index_cast", "arith.constant", "arith.extsi", "arith.extui",
             "arith.trunci", true)
      .Cases("arith.sitofp", "arith.fptosi", "arith.cmpf", "arith.extf",
             "arith.truncf", true)
      .Cases("arith.maxnumf", "arith.minnumf", "arith.minimumf", "arith.minsi",
             true)
      .Cases("arith.addf", "arith.mulf", "arith.subf", "arith.divf", true)
      .Cases("arith.addi", "arith.subi", "arith.muli", "arith.divui",
             "arith.floordivsi", "arith.remui", "arith.remsi", true)
      .Cases("arith.andi", "arith.ori", "arith.xori", "arith.shli",
             "arith.shrui", true)
      .Cases("arith.select", "math.exp", "math.exp2", "math.log", "math.absf",
             true)
      .Default(false);
}

struct TriangularRecurrenceInfo {
  int64_t extent = 0;
  mlir::Value coeff;
  mlir::Value state;
  int coeffRowAxis = -1;
  int coeffColAxis = -1;
  int stateAxis = -1;
  llvm::SmallVector<mlir::Value, 4> coeffCoords;
  llvm::SmallVector<mlir::Value, 4> stateCoords;
  mlir::Operation *mulOp = nullptr;
  mlir::Operation *addOp = nullptr;
  mlir::Operation *subOp = nullptr;
  mlir::Operation *divOp = nullptr;
};

inline bool matchTriangularRecurrence(mlir::scf::ForOp outerLoop,
                                      mlir::func::FuncOp fn,
                                      const CallFacts *facts = nullptr,
                                      TriangularRecurrenceInfo *out = nullptr) {
  if (outerLoop.getNumResults() != 0 ||
      outerLoop->hasAttr("kir.reassociate") ||
      !mlir::matchPattern(outerLoop.getLowerBound(), mlir::m_Zero()) ||
      !mlir::matchPattern(outerLoop.getStep(), mlir::m_One()))
    return false;
  auto ext = orderedStaticLoopExtent(outerLoop, fn, facts);
  if (!ext || *ext < 2 || *ext > 256)
    return false;

  mlir::Value r = outerLoop.getInductionVar();
  mlir::scf::ForOp innerLoop;
  llvm::SmallVector<mlir::memref::LoadOp, 2> outerLoads;
  llvm::SmallVector<mlir::memref::StoreOp, 1> outerStores;
  for (mlir::Operation &inst : outerLoop.getBody()->without_terminator()) {
    if (auto fl = mlir::dyn_cast<mlir::scf::ForOp>(inst)) {
      if (innerLoop)
        return false;
      innerLoop = fl;
      continue;
    }
    if (inst.getNumRegions() != 0)
      return false;
    if (auto ld = mlir::dyn_cast<mlir::memref::LoadOp>(inst)) {
      outerLoads.push_back(ld);
      continue;
    }
    if (auto st = mlir::dyn_cast<mlir::memref::StoreOp>(inst)) {
      outerStores.push_back(st);
      continue;
    }
    if (!mlir::isa<mlir::arith::ConstantOp, mlir::arith::ConstantIndexOp,
                   mlir::memref::DimOp, mlir::arith::SubFOp,
                   mlir::arith::DivFOp>(inst))
      return false;
  }
  if (!innerLoop || outerStores.size() != 1 ||
      !mlir::matchPattern(innerLoop.getLowerBound(), mlir::m_Zero()) ||
      innerLoop.getUpperBound() != r ||
      !mlir::matchPattern(innerLoop.getStep(), mlir::m_One()) ||
      innerLoop.getNumRegionIterArgs() != 1 ||
      !innerLoop.getResult(0).getType().isF32() ||
      !mlir::matchPattern(innerLoop.getInitArgs()[0], mlir::m_PosZeroFloat()))
    return false;

  mlir::Value c = innerLoop.getInductionVar();
  mlir::Value carry = innerLoop.getRegionIterArgs()[0];
  llvm::SmallVector<mlir::memref::LoadOp, 2> innerLoads;
  mlir::arith::MulFOp mulOp;
  mlir::arith::AddFOp addOp;
  for (mlir::Operation &inst : innerLoop.getBody()->without_terminator()) {
    if (inst.getNumRegions() != 0)
      return false;
    if (auto ld = mlir::dyn_cast<mlir::memref::LoadOp>(inst)) {
      innerLoads.push_back(ld);
      continue;
    }
    if (auto m = mlir::dyn_cast<mlir::arith::MulFOp>(inst)) {
      if (mulOp)
        return false;
      mulOp = m;
      continue;
    }
    if (auto a = mlir::dyn_cast<mlir::arith::AddFOp>(inst)) {
      if (addOp)
        return false;
      addOp = a;
      continue;
    }
    if (!mlir::isa<mlir::arith::ConstantOp, mlir::arith::ConstantIndexOp,
                   mlir::memref::DimOp>(inst))
      return false;
  }
  if (innerLoads.size() != 2 || !mulOp || !addOp ||
      innerLoop.getBody()->getTerminator()->getOperand(0) != addOp.getResult())
    return false;
  bool addOrderOk =
      (addOp.getLhs() == carry && addOp.getRhs() == mulOp.getResult()) ||
      (addOp.getRhs() == carry && addOp.getLhs() == mulOp.getResult());
  if (!addOrderOk)
    return false;
  bool mulOrderOk =
      (mulOp.getLhs() == innerLoads[0].getResult() &&
       mulOp.getRhs() == innerLoads[1].getResult()) ||
      (mulOp.getRhs() == innerLoads[0].getResult() &&
       mulOp.getLhs() == innerLoads[1].getResult());
  if (!mulOrderOk)
    return false;

  mlir::memref::StoreOp stOp = outerStores[0];
  mlir::Value stateView = stOp.getMemRef();
  int stateIdx = (innerLoads[0].getMemRef() == stateView)
                     ? 0
                     : (innerLoads[1].getMemRef() == stateView ? 1 : -1);
  if (stateIdx < 0)
    return false;
  mlir::memref::LoadOp statePrevLd = innerLoads[stateIdx];
  mlir::memref::LoadOp coeffLd = innerLoads[1 - stateIdx];
  mlir::Value coeffView = coeffLd.getMemRef();
  if (!orderedViewsDisjoint(fn, facts, coeffView, stateView))
    return false;

  int stateAxis = -1;
  for (auto [i, idx] : llvm::enumerate(statePrevLd.getIndices())) {
    if (idx == c) {
      if (stateAxis >= 0)
        return false;
      stateAxis = int(i);
    } else if (!outerLoop.isDefinedOutsideOfLoop(idx)) {
      return false;
    }
  }
  if (stateAxis < 0 ||
      statePrevLd.getIndices().size() != stOp.getIndices().size())
    return false;
  for (auto [i, idx] : llvm::enumerate(stOp.getIndices())) {
    if (int(i) == stateAxis) {
      if (idx != r)
        return false;
    } else if (idx != statePrevLd.getIndices()[i]) {
      return false;
    }
  }

  int coeffRowAxis = -1, coeffColAxis = -1;
  for (auto [i, idx] : llvm::enumerate(coeffLd.getIndices())) {
    if (idx == r) {
      if (coeffRowAxis >= 0)
        return false;
      coeffRowAxis = int(i);
    } else if (idx == c) {
      if (coeffColAxis >= 0)
        return false;
      coeffColAxis = int(i);
    } else if (!outerLoop.isDefinedOutsideOfLoop(idx)) {
      return false;
    }
  }
  if (coeffRowAxis < 0 || coeffColAxis < 0)
    return false;

  auto stDim = orderedArgDim(fn, facts, stateView, stateAxis);
  auto cRowDim = orderedArgDim(fn, facts, coeffView, coeffRowAxis);
  auto cColDim = orderedArgDim(fn, facts, coeffView, coeffColAxis);
  if (!stDim || *stDim != *ext || !cRowDim || *cRowDim < *ext || !cColDim ||
      *cColDim < *ext)
    return false;

  mlir::Value storedVal = stOp.getValue();
  mlir::arith::DivFOp divOp = storedVal.getDefiningOp<mlir::arith::DivFOp>();
  mlir::Value diffVal = divOp ? divOp.getLhs() : storedVal;
  auto subOp = diffVal.getDefiningOp<mlir::arith::SubFOp>();
  if (!subOp || subOp.getRhs() != innerLoop.getResult(0))
    return false;
  auto rhsLd = subOp.getLhs().getDefiningOp<mlir::memref::LoadOp>();
  if (!rhsLd || rhsLd.getMemRef() != stateView ||
      rhsLd.getIndices() != stOp.getIndices() ||
      !innerLoop->isBeforeInBlock(rhsLd) || !rhsLd->isBeforeInBlock(stOp))
    return false;

  if (divOp) {
    if (outerLoads.size() != 2)
      return false;
    auto diagLd = divOp.getRhs().getDefiningOp<mlir::memref::LoadOp>();
    if (!diagLd || diagLd.getMemRef() != coeffView ||
        diagLd.getIndices().size() != coeffLd.getIndices().size())
      return false;
    for (auto [i, idx] : llvm::enumerate(diagLd.getIndices())) {
      if (int(i) == coeffRowAxis || int(i) == coeffColAxis) {
        if (idx != r)
          return false;
      } else if (idx != coeffLd.getIndices()[i]) {
        return false;
      }
    }
  } else if (outerLoads.size() != 1) {
    return false;
  }

  if (out) {
    out->extent = *ext;
    out->coeff = coeffView;
    out->state = stateView;
    out->coeffRowAxis = coeffRowAxis;
    out->coeffColAxis = coeffColAxis;
    out->stateAxis = stateAxis;
    out->coeffCoords.assign(coeffLd.getIndices().begin(),
                            coeffLd.getIndices().end());
    out->stateCoords.assign(stOp.getIndices().begin(), stOp.getIndices().end());
    out->mulOp = mulOp.getOperation();
    out->addOp = addOp.getOperation();
    out->subOp = subOp.getOperation();
    out->divOp = divOp ? divOp.getOperation() : nullptr;
  }
  return true;
}

struct ExclusiveScanLoopInfo {
  int64_t extent = 0;
  mlir::memref::LoadOp srcLoad;
  mlir::Operation *addOp = nullptr;
  llvm::SmallVector<std::pair<mlir::memref::StoreOp, bool>, 4> stores;
};

inline bool matchExclusiveScanLoop(mlir::scf::ForOp loop, mlir::func::FuncOp fn,
                                   const CallFacts *facts = nullptr,
                                   ExclusiveScanLoopInfo *out = nullptr) {
  if (loop.getNumRegionIterArgs() != 1 || loop->hasAttr("kir.reassociate") ||
      !mlir::matchPattern(loop.getLowerBound(), mlir::m_Zero()) ||
      !mlir::matchPattern(loop.getStep(), mlir::m_One()))
    return false;
  auto ext = orderedStaticLoopExtent(loop, fn, facts);
  if (!ext || *ext < 2 || *ext > 1024)
    return false;
  mlir::Type elemT = loop.getResult(0).getType();
  if (!elemT.isInteger(32) && !elemT.isF32())
    return false;

  mlir::Value iv = loop.getInductionVar();
  mlir::Value carry = loop.getRegionIterArgs()[0];
  llvm::SmallVector<mlir::memref::LoadOp, 2> loads;
  llvm::SmallVector<mlir::memref::StoreOp, 4> stores;
  mlir::Operation *addOp = nullptr;
  for (mlir::Operation &inst : loop.getBody()->without_terminator()) {
    if (inst.getNumRegions() != 0)
      return false;
    if (auto ld = mlir::dyn_cast<mlir::memref::LoadOp>(inst)) {
      loads.push_back(ld);
      continue;
    }
    if (auto st = mlir::dyn_cast<mlir::memref::StoreOp>(inst)) {
      stores.push_back(st);
      continue;
    }
    if (mlir::isa<mlir::arith::AddIOp, mlir::arith::AddFOp>(inst)) {
      if (addOp)
        return false;
      addOp = &inst;
      continue;
    }
    if (!mlir::isa<mlir::arith::ConstantOp, mlir::arith::ConstantIndexOp,
                   mlir::memref::DimOp>(inst))
      return false;
  }
  if (loads.size() != 1 || !addOp ||
      loop.getBody()->getTerminator()->getOperand(0) != addOp->getResult(0))
    return false;
  if (elemT.isF32() &&
      !intentdsl::kir::hasFastMath(addOp, mlir::arith::FastMathFlags::reassoc))
    return false;

  mlir::memref::LoadOp srcLd = loads[0];
  if (srcLd.getIndices().empty() || srcLd.getIndices().back() != iv ||
      !llvm::all_of(srcLd.getIndices().drop_back(), [&](mlir::Value v) {
        return loop.isDefinedOutsideOfLoop(v);
      }))
    return false;
  bool combOk = (addOp->getOperand(0) == carry &&
                 addOp->getOperand(1) == srcLd.getResult()) ||
                (addOp->getOperand(1) == carry &&
                 addOp->getOperand(0) == srcLd.getResult());
  if (!combOk)
    return false;

  llvm::SmallVector<std::pair<mlir::memref::StoreOp, bool>, 4> classified;
  for (size_t i = 0; i < stores.size(); ++i) {
    auto st = stores[i];
    if (st.getIndices().empty() || st.getIndices().back() != iv ||
        !llvm::all_of(st.getIndices().drop_back(), [&](mlir::Value v) {
          return loop.isDefinedOutsideOfLoop(v);
        }))
      return false;
    auto stDim = orderedArgDim(
        fn, facts, st.getMemRef(),
        mlir::cast<mlir::MemRefType>(st.getMemRef().getType()).getRank() - 1);
    if (!stDim || *stDim < *ext)
      return false;
    bool incl = false;
    if (st.getValue() == addOp->getResult(0))
      incl = true;
    else if (st.getValue() != carry)
      return false;
    if (!orderedViewsDisjoint(fn, facts, st.getMemRef(), srcLd.getMemRef())) {
      if (st.getMemRef() != srcLd.getMemRef() ||
          st.getIndices() != srcLd.getIndices() || !srcLd->isBeforeInBlock(st))
        return false;
    }
    for (size_t j = 0; j < i; ++j)
      if (!orderedViewsDisjoint(fn, facts, st.getMemRef(),
                                stores[j].getMemRef()))
        return false;
    classified.push_back({st, incl});
  }

  if (out) {
    out->extent = *ext;
    out->srcLoad = srcLd;
    out->addOp = addOp;
    out->stores = std::move(classified);
  }
  return true;
}

inline bool orderedValueDependsOn(mlir::Value val, mlir::Operation *target,
                                  mlir::Block *stopBlock) {
  llvm::SmallPtrSet<mlir::Value, 16> visited;
  llvm::SmallVector<mlir::Value, 8> worklist{val};
  while (!worklist.empty()) {
    mlir::Value cur = worklist.pop_back_val();
    if (!visited.insert(cur).second)
      continue;
    mlir::Operation *def = cur.getDefiningOp();
    if (!def || def->getBlock() != stopBlock)
      continue;
    if (def == target)
      return true;
    for (mlir::Value opnd : def->getOperands())
      worklist.push_back(opnd);
  }
  return false;
}

struct BoundedHistogramLoopInfo {
  int64_t bins = 0;
  mlir::memref::LoadOp histLoad;
  mlir::memref::StoreOp histStore;
  mlir::Operation *addOp = nullptr;
  mlir::Value binIdx;
  mlir::Value deltaVal;
};

inline bool matchBoundedHistogramLoop(mlir::scf::ForOp loop,
                                      mlir::func::FuncOp fn,
                                      const CallFacts *facts = nullptr,
                                      BoundedHistogramLoopInfo *out = nullptr) {
  if (loop.getNumResults() != 0 || loop->hasAttr("kir.reassociate") ||
      !mlir::matchPattern(loop.getLowerBound(), mlir::m_Zero()) ||
      !mlir::matchPattern(loop.getStep(), mlir::m_One()) ||
      !loop.isDefinedOutsideOfLoop(loop.getUpperBound()))
    return false;

  mlir::Value iv = loop.getInductionVar();
  llvm::SmallVector<mlir::memref::StoreOp, 2> stores;
  llvm::SmallVector<mlir::memref::LoadOp, 4> loads;
  for (mlir::Operation &inst : loop.getBody()->without_terminator()) {
    if (inst.getNumRegions() != 0)
      return false;
    if (auto st = mlir::dyn_cast<mlir::memref::StoreOp>(inst)) {
      stores.push_back(st);
      continue;
    }
    if (auto ld = mlir::dyn_cast<mlir::memref::LoadOp>(inst)) {
      loads.push_back(ld);
      continue;
    }
    if (!supportedPurePanelArith(inst))
      return false;
  }
  if (stores.size() != 1)
    return false;

  mlir::memref::StoreOp histSt = stores[0];
  mlir::Value histView = histSt.getMemRef();
  if (histSt.getIndices().empty())
    return false;
  mlir::Value binIdx = histSt.getIndices().back();
  if (binIdx == iv ||
      !llvm::all_of(histSt.getIndices().drop_back(), [&](mlir::Value v) {
        return loop.isDefinedOutsideOfLoop(v);
      }))
    return false;

  int rank = mlir::cast<mlir::MemRefType>(histView.getType()).getRank();
  auto binsOpt = orderedArgDim(fn, facts, histView, rank - 1);
  if (!binsOpt || *binsOpt < 2 || *binsOpt > 512)
    return false;

  mlir::Operation *addOp = histSt.getValue().getDefiningOp();
  if (!addOp || !mlir::isa<mlir::arith::AddIOp, mlir::arith::AddFOp>(addOp))
    return false;
  mlir::Type elemT = addOp->getResult(0).getType();
  if (!elemT.isInteger(32) &&
      !(elemT.isF32() && intentdsl::kir::hasFastMath(
                             addOp, mlir::arith::FastMathFlags::reassoc)))
    return false;

  mlir::memref::LoadOp histLd;
  mlir::Value deltaVal;
  for (int side : {0, 1}) {
    if (auto ld = addOp->getOperand(side).getDefiningOp<mlir::memref::LoadOp>();
        ld && ld.getMemRef() == histView &&
        ld.getIndices() == histSt.getIndices() && ld->isBeforeInBlock(histSt)) {
      histLd = ld;
      deltaVal = addOp->getOperand(1 - side);
      break;
    }
  }
  if (!histLd || orderedValueDependsOn(deltaVal, histLd, loop.getBody()) ||
      orderedValueDependsOn(binIdx, histLd, loop.getBody()))
    return false;

  for (auto ld : loads) {
    if (ld == histLd)
      continue;
    if (!orderedViewsDisjoint(fn, facts, ld.getMemRef(), histView) ||
        ld.getIndices().empty())
      return false;
    for (auto [k, idx] : llvm::enumerate(ld.getIndices())) {
      if (k + 1 == ld.getIndices().size() && idx == iv)
        continue;
      if (!loop.isDefinedOutsideOfLoop(idx))
        return false;
    }
  }

  if (out) {
    out->bins = *binsOpt;
    out->histLoad = histLd;
    out->histStore = histSt;
    out->addOp = addOp;
    out->binIdx = binIdx;
    out->deltaVal = deltaVal;
  }
  return true;
}

struct BoundedScatterLoopInfo {
  int64_t bins = 0;
  mlir::memref::LoadOp cursorLoad;
  mlir::memref::StoreOp cursorStore;
  mlir::Operation *cursorAdd = nullptr;
  mlir::Operation *posCast = nullptr;
  mlir::Value binIdx;
  llvm::SmallVector<mlir::memref::StoreOp, 4> scatterStores;
};

inline bool matchBoundedScatterLoop(mlir::scf::ForOp loop, mlir::func::FuncOp fn,
                                    const CallFacts *facts = nullptr,
                                    BoundedScatterLoopInfo *out = nullptr) {
  if (loop.getNumResults() != 0 || loop->hasAttr("kir.reassociate") ||
      !mlir::matchPattern(loop.getLowerBound(), mlir::m_Zero()) ||
      !mlir::matchPattern(loop.getStep(), mlir::m_One()) ||
      !loop.isDefinedOutsideOfLoop(loop.getUpperBound()))
    return false;

  mlir::Value iv = loop.getInductionVar();
  llvm::SmallVector<mlir::memref::StoreOp, 4> stores;
  llvm::SmallVector<mlir::memref::LoadOp, 4> loads;
  for (mlir::Operation &inst : loop.getBody()->without_terminator()) {
    if (inst.getNumRegions() != 0)
      return false;
    if (auto st = mlir::dyn_cast<mlir::memref::StoreOp>(inst)) {
      stores.push_back(st);
      continue;
    }
    if (auto ld = mlir::dyn_cast<mlir::memref::LoadOp>(inst)) {
      loads.push_back(ld);
      continue;
    }
    if (!supportedPurePanelArith(inst))
      return false;
  }
  if (stores.size() < 2)
    return false;

  mlir::memref::StoreOp cursorSt;
  mlir::memref::LoadOp cursorLd;
  mlir::Operation *cursorAdd = nullptr;
  int64_t bins = 0;
  for (auto st : stores) {
    auto add = st.getValue().getDefiningOp<mlir::arith::AddIOp>();
    if (!add || !add.getType().isInteger(32) || st.getIndices().empty())
      continue;
    mlir::memref::LoadOp candLd;
    for (int side : {0, 1}) {
      if (mlir::matchPattern(add->getOperand(1 - side), mlir::m_One()))
        if (auto ld =
                add->getOperand(side).getDefiningOp<mlir::memref::LoadOp>();
            ld && ld.getMemRef() == st.getMemRef() &&
            ld.getIndices() == st.getIndices() && ld->isBeforeInBlock(st))
          candLd = ld;
    }
    if (!candLd)
      continue;
    int rank = mlir::cast<mlir::MemRefType>(st.getMemRef().getType()).getRank();
    auto bOpt = orderedArgDim(fn, facts, st.getMemRef(), rank - 1);
    if (!bOpt || *bOpt < 2 || *bOpt > 512)
      continue;
    if (cursorSt)
      return false;
    cursorSt = st;
    cursorLd = candLd;
    cursorAdd = add.getOperation();
    bins = *bOpt;
  }
  if (!cursorSt)
    return false;

  mlir::Value binIdx = cursorSt.getIndices().back();
  if (binIdx == iv ||
      !llvm::all_of(cursorSt.getIndices().drop_back(), [&](mlir::Value v) {
        return loop.isDefinedOutsideOfLoop(v);
      }) ||
      orderedValueDependsOn(binIdx, cursorLd, loop.getBody()))
    return false;

  mlir::Operation *posCast = nullptr;
  for (mlir::Operation *user : cursorLd.getResult().getUsers()) {
    if (user == cursorAdd)
      continue;
    if (mlir::isa<mlir::arith::IndexCastOp>(user) &&
        user->getResult(0).getType().isIndex() && !posCast) {
      posCast = user;
      continue;
    }
    return false;
  }
  if (!posCast)
    return false;

  llvm::SmallVector<mlir::memref::StoreOp, 4> scatters;
  for (auto st : stores) {
    if (st == cursorSt)
      continue;
    if (st.getIndices().empty() ||
        st.getIndices().back() != posCast->getResult(0) ||
        !llvm::all_of(st.getIndices().drop_back(), [&](mlir::Value v) {
          return loop.isDefinedOutsideOfLoop(v);
        }) ||
        orderedValueDependsOn(st.getValue(), cursorLd, loop.getBody()) ||
        !orderedViewsDisjoint(fn, facts, st.getMemRef(), cursorSt.getMemRef()))
      return false;
    for (auto prev : scatters)
      if (!orderedViewsDisjoint(fn, facts, st.getMemRef(), prev.getMemRef()))
        return false;
    scatters.push_back(st);
  }
  if (scatters.empty())
    return false;

  for (auto ld : loads) {
    if (ld == cursorLd)
      continue;
    if (!orderedViewsDisjoint(fn, facts, ld.getMemRef(),
                              cursorSt.getMemRef()) ||
        ld.getIndices().empty())
      return false;
    for (auto sc : scatters)
      if (!orderedViewsDisjoint(fn, facts, ld.getMemRef(), sc.getMemRef()))
        return false;
    for (auto [k, idx] : llvm::enumerate(ld.getIndices())) {
      if (k + 1 == ld.getIndices().size() && idx == iv)
        continue;
      if (!loop.isDefinedOutsideOfLoop(idx))
        return false;
    }
  }

  if (out) {
    out->bins = bins;
    out->cursorLoad = cursorLd;
    out->cursorStore = cursorSt;
    out->cursorAdd = cursorAdd;
    out->posCast = posCast;
    out->binIdx = binIdx;
    out->scatterStores = std::move(scatters);
  }
  return true;
}

struct AssociativeScatterUpdate {
  mlir::BlockArgument targetView;
  mlir::memref::LoadOp pairedLoad;
  mlir::Operation *combiner = nullptr;
  mlir::memref::StoreOp store;
  mlir::Value contribution;
};

struct AssociativeScatterLoopInfo {
  int64_t extent = 0;
  llvm::SmallVector<AssociativeScatterUpdate, 2> updates;
};

inline bool matchAssociativeScatterLoop(mlir::scf::ForOp loop,
                                        mlir::func::FuncOp fn,
                                        const CallFacts *facts,
                                        AssociativeScatterLoopInfo *out = nullptr) {
  if (!loop || !fn || !loop->hasAttr("kir.reassociate") ||
      loop.getNumRegionIterArgs() != 0 ||
      !mlir::matchPattern(loop.getLowerBound(), mlir::m_Zero()) ||
      !mlir::matchPattern(loop.getStep(), mlir::m_One()) ||
      !loop.isDefinedOutsideOfLoop(loop.getUpperBound()))
    return false;
  auto ubDef = loop.getUpperBound().getDefiningOp();
  if (!ubDef ||
      !mlir::isa<mlir::arith::ConstantOp, mlir::arith::ConstantIndexOp,
                 mlir::memref::DimOp>(ubDef))
    return false;
  auto extOpt = orderedStaticLoopExtent(loop, fn, facts);
  if (!extOpt || *extOpt <= 0)
    return false;

  llvm::SmallVector<mlir::memref::StoreOp, 2> stores;
  llvm::SmallVector<mlir::memref::LoadOp, 4> loads;
  for (mlir::Operation &inst : loop.getBody()->without_terminator()) {
    if (inst.getNumRegions() != 0)
      return false;
    if (auto st = mlir::dyn_cast<mlir::memref::StoreOp>(inst)) {
      stores.push_back(st);
    } else if (auto ld = mlir::dyn_cast<mlir::memref::LoadOp>(inst)) {
      loads.push_back(ld);
    } else if (!mlir::isa<mlir::memref::DimOp>(inst) &&
               !mlir::isMemoryEffectFree(&inst)) {
      return false;
    }
  }
  if (stores.empty())
    return false;

  llvm::SmallVector<AssociativeScatterUpdate, 2> updates;
  llvm::SmallPtrSet<mlir::Operation *, 4> pairedLoads;
  llvm::SmallDenseSet<mlir::Value, 4> targets;
  for (auto st : stores) {
    auto target = mlir::dyn_cast<mlir::BlockArgument>(st.getMemRef());
    if (!target || target.getOwner() != &fn.front() ||
        !targets.insert(target).second || !st.getValue().hasOneUse())
      return false;
    mlir::Type elemTy =
        mlir::cast<mlir::MemRefType>(target.getType()).getElementType();
    if (!elemTy.isF32() && !elemTy.isInteger(32))
      return false;
    mlir::Operation *comb = st.getValue().getDefiningOp();
    if (!comb || comb->getBlock() != loop.getBody())
      return false;
    if (elemTy.isInteger(32)) {
      if (!mlir::isa<mlir::arith::AddIOp>(comb))
        return false;
    } else {
      if (!mlir::isa<mlir::arith::AddFOp>(comb) ||
          !intentdsl::kir::hasFastMath(comb,
                                       mlir::arith::FastMathFlags::reassoc))
        return false;
    }
    auto ld = comb->getOperand(0).getDefiningOp<mlir::memref::LoadOp>();
    if (!ld || ld->getBlock() != loop.getBody() || !ld.getResult().hasOneUse() ||
        ld.getMemRef() != target || ld.getIndices() != st.getIndices())
      return false;
    mlir::Value contrib = comb->getOperand(1);
    if (orderedValueDependsOn(contrib, ld, loop.getBody()))
      return false;
    pairedLoads.insert(ld.getOperation());
    updates.push_back({target, ld, comb, st, contrib});
  }

  for (auto ld : loads) {
    if (pairedLoads.contains(ld.getOperation()))
      continue;
    for (const auto &u : updates)
      if (!orderedViewsDisjoint(fn, facts, u.targetView, ld.getMemRef()))
        return false;
  }

  if (out) {
    out->extent = *extOpt;
    out->updates = std::move(updates);
  }
  return true;
}

inline bool canSplitAssociativeScatter(mlir::Operation *space,
                                       const CallFacts *facts, int64_t width,
                                       int64_t foldSplit,
                                       AssociativeScatterLoopInfo *out = nullptr) {
  if (!space || space->getNumRegions() != 1 ||
      !space->getRegion(0).hasOneBlock() || width < 1 || foldSplit < 2 ||
      !llvm::isPowerOf2_64(foldSplit))
    return false;
  auto fn = space->getParentOfType<mlir::func::FuncOp>();
  if (!fn)
    return false;
  mlir::scf::ForOp soleLoop;
  for (mlir::Operation &inst : space->getRegion(0).front().without_terminator()) {
    if (auto lp = mlir::dyn_cast<mlir::scf::ForOp>(inst)) {
      if (soleLoop)
        return false;
      soleLoop = lp;
      continue;
    }
    if (!mlir::isa<mlir::memref::DimOp>(inst) &&
        !mlir::isMemoryEffectFree(&inst))
      return false;
  }
  if (!soleLoop)
    return false;
  AssociativeScatterLoopInfo info;
  if (!matchAssociativeScatterLoop(soleLoop, fn, facts, &info))
    return false;
  if (foldSplit > int64_t(llvm::PowerOf2Ceil(std::max<int64_t>(2, info.extent))) ||
      llvm::divideCeil(info.extent, foldSplit) < 4)
    return false;
  if (out)
    *out = std::move(info);
  return true;
}
} // namespace intentdsl::block
