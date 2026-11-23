#pragma once
#include "intentdsl/levels/execution/block/ir/Owners.h"
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
    // Branches can guard per-token loads. Their values and effects require a
    // separate predicated tile lowering; leave such folds on the owner path.
    if (llvm::any_of(*loop.getBody(),
                     [](mlir::Operation &v) { return v.getNumRegions() != 0; }))
      return false;
    for (auto [idx, v] :
         llvm::enumerate(loop.getBody()->getTerminator()->getOperands())) {
      auto *op = v.getDefiningOp();
      if (!op || !mlir::isa<mlir::arith::MaxNumFOp, mlir::arith::AddFOp>(op) ||
          op->getOperand(0) != loop.getRegionIterArgs()[idx])
        return false;
    }
    for (auto &v : loop.getBody()->without_terminator()) {
      auto kind = v.getName().stripDialect();
      if (kind != "slice_read" && kind != "lane_splat" && kind != "lane_sum" &&
          !llvm::StringSwitch<bool>(v.getName().getStringRef())
               .Cases("memref.load", "arith.index_cast", "arith.constant", true)
               .Cases("arith.extf", "arith.truncf", "arith.maxnumf", true)
               .Cases("arith.addf", "arith.mulf", "arith.subf", true)
               .Cases("arith.addi", "arith.subi", "arith.muli", true)
               .Cases("arith.floordivsi", "arith.remsi", "math.exp",
                      "math.exp2", true)
               .Default(false))
        return false;
    }
    matched = true;
  }
  return matched;
}
} // namespace intentdsl::block
