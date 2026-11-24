#pragma once
#include "intentdsl/levels/algorithm/kir/ir/FastMath.h"
#include "intentdsl/levels/execution/block/ir/Owners.h"
#include "mlir/Dialect/Math/IR/Math.h"
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
             .Cases("arith.extf", "arith.truncf", "arith.maxnumf", true)
             .Cases("arith.minnumf", "arith.minimumf", "arith.minsi", true)
             .Cases("arith.addf", "arith.mulf", "arith.subf", true)
             .Cases("arith.addi", "arith.subi", "arith.muli", true)
             .Cases("arith.andi", "arith.ori", "arith.xori", true)
             .Cases("arith.select", "math.exp", "math.exp2", "math.absf", true)
             .Default(false))
      return false;
  }
  return true;
}
inline bool panelFolds(mlir::Operation *o,
                       const compilerinfra::TargetProfile &p, int64_t panel) {
  auto machine = p.services.get<OwnerMachine>();
  if (!machine || machine->parallelMembers < 16 ||
      (panel != 16 && panel != 32 && panel != 64 && panel != 128))
    return false;
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
      if (auto sub = op->getOperand(0).getDefiningOp<mlir::arith::SubFOp>();
          sub && intentdsl::kir::hasFastMath(sub, FMF::reassoc))
        hasReassocExpShift = true;
    });
  }
  return reassocLoops >= 2 && hasReassocExpShift;
}
} // namespace intentdsl::block
