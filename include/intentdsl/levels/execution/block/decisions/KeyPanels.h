#pragma once
#include "intentdsl/levels/execution/block/decisions/RowCapacity.h"
#include "intentdsl/levels/execution/block/ir/Owners.h"

namespace intentdsl::block {
// A band owns disjoint output bins. A collective across bins or an additional
// side effect in the row cannot be replicated across independent blocks.
inline int64_t privateKeyPanel(mlir::Operation *region,
                               const compilerinfra::TargetProfile &profile,
                               int64_t width) {
  auto caps = profile.services.get<RowCapacity>();
  auto machine = profile.services.get<OwnerMachine>();
  if (!caps || caps->nativePrivateHistogram || caps->independentWidths.empty() ||
      !machine || machine->residentRegisters <= 0 || width <= 0)
    return 0;
  int64_t lanes = machine->residentRegisters / width;
  bool counts = false;
  for (auto &op : region->getRegion(0).front().without_terminator()) {
    auto name = op.getName().stripDialect();
    if (name == "count_keys") {
      counts = true;
      lanes = std::min(lanes, op.getAttrOfType<mlir::IntegerAttr>("span").getInt());
    } else if (name != "write" || !op.getOperand(2).getDefiningOp() ||
               op.getOperand(2).getDefiningOp()->getName().stripDialect() != "count_keys") {
      return 0;
    }
  }
  return counts && lanes > 0 ? int64_t(1) << llvm::Log2_64(lanes) : 0;
}
} // namespace intentdsl::block
