#pragma once
#include "intentdsl/levels/algorithm/kir/ir/Ordered.h"
#include "intentdsl/levels/execution/cpu/ir/Task.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace intentdsl::cpu {
// Compare invocation-wide VNNI preparation with packing on every owner tile.
// Count transfers in elements; the element size cancels in the comparison.
inline int64_t invocationPanelElements(mlir::Operation &space,
                                       const CallFacts &facts,
                                       const RowMachine &machine,
                                       mlir::BlockArgument arg, int64_t tile,
                                       int64_t batch, bool transpose) {
  auto &record = *facts.arguments[arg.getArgNumber()].getAsObject();
  auto *shape = record.getArray("shape"), *strides = record.getArray("strides");
  auto group = record.getInteger("alias_group");
  if (!shape || !strides || shape->size() < 2 ||
      shape->size() != strides->size() || !group || *group < 0 || tile <= 0 ||
      machine.l2CacheBytes <= 0)
    return false;
  if (space
          .walk([&](mlir::Operation *op) {
            if (op->getName().stripDialect() != "slice_write" &&
                !mlir::isa<mlir::memref::StoreOp>(op))
              return mlir::WalkResult::advance();
            auto dst = mlir::dyn_cast<mlir::BlockArgument>(op->getOperand(1));
            auto *r = dst && dst.getOwner() == arg.getOwner()
                          ? facts.arguments[dst.getArgNumber()].getAsObject()
                          : nullptr;
            auto id = r ? r->getInteger("alias_group") : std::nullopt;
            return !id || *id < 0 || *id == *group
                       ? mlir::WalkResult::interrupt()
                       : mlir::WalkResult::advance();
          })
          .wasInterrupted())
    return false;
  int64_t elements = 1;
  for (int j = int(shape->size()) - 1; j >= 0; --j) {
    auto n = (*shape)[j].getAsInteger();
    if (!n || *n <= 0 || (*strides)[j].getAsInteger() != elements ||
        __builtin_mul_overflow(elements, *n, &elements))
      return false;
  }
  auto tokens = *(*shape)[shape->size() - 2].getAsInteger();
  auto width = *shape->back().getAsInteger();
  int64_t k = transpose ? width : tile;
  if (tokens % tile || k % 2 || elements > INT64_MAX / 2)
    return false;
  double panels = double(elements) / (tile * double(width));
  double calls = double(intentdsl::kir::ownerCount(&space, facts)) / batch *
                 (tokens / tile);
  double reuse = calls / panels;
  auto rowStride = *(*strides)[strides->size() - 2].getAsInteger();
  auto columnStride = *strides->back().getAsInteger();
  auto kStride = transpose ? columnStride : rowStride;
  auto nStride = transpose ? rowStride : columnStride;
  // Pairwise K interleaving reads either an adjacent pair or two rows.
  // Non-unit N strides additionally require a transpose of the source rows.
  double packTransfers = 1.0 + (kStride > 1) + (nStride > 1 ? 2. : 1.);
  double live = 2. * tile * width;
  double readTransfers = 1. + std::min(1., live / machine.l2CacheBytes);
  return packTransfers + reuse * readTransfers < reuse * (packTransfers + 1.)
             ? elements
             : 0;
}
} // namespace intentdsl::cpu
