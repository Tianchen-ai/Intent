#include "intentdsl/levels/execution/block/ir/FoldPanel.h"
#include "intentdsl/levels/storage/dsa/ir/Owners.h"

using namespace mlir;
namespace intentdsl::dsa {
std::optional<PrivateRows> privateRows(func::FuncOp fn,
                                       const compilerinfra::TargetProfile &p) {
  auto local = p.services.get<LocalSupply>();
  if (!local)
    return {};
  auto &space = fn.front().front();
  auto &bb = space.getRegion(0).front();
  if (bb.getNumArguments() != 1)
    return {};
  PrivateRows plan;
  for (auto loop : bb.getOps<scf::ForOp>()) {
    block::ExclusiveScanLoopInfo scan;
    if (block::matchExclusiveScanLoop(loop, fn, nullptr, &scan) &&
        loop.getResult(0).getType().isInteger(32))
      plan.scanWidth = std::max(plan.scanWidth, scan.extent);
    block::OrderedSelectionInfo sel;
    if (block::matchOrderedSelectionChain(loop, &sel)) {
      auto ld = sel.candidateVal.getDefiningOp<memref::LoadOp>();
      auto n = block::orderedStaticLoopExtent(loop, fn, nullptr);
      if (ld && ld.getIndices().size() == 2 &&
          ld.getIndices().back() == loop.getInductionVar() && n && *n > 0 &&
          *n <= INT32_MAX) {
        plan.k = std::max<int64_t>(plan.k, sel.k);
        plan.selectionWidth =
            std::min<int64_t>(*n, local->alignment / sizeof(float) * 8);
      }
    }
  }
  if (!plan.k && !plan.scanWidth)
    return {};
  auto owners = space.getAttrOfType<IntegerAttr>("block.owners");
  if (!owners)
    return {};
  Value row = bb.getArgument(0);
  llvm::DenseSet<int64_t> groups;
  for (BlockArgument arg : fn.getArguments()) {
    auto t = dyn_cast<MemRefType>(arg.getType());
    if (!t)
      continue;
    auto sh = fn.getArgAttrOfType<DenseI64ArrayAttr>(arg.getArgNumber(),
                                                     "block.shape");
    auto st = fn.getArgAttrOfType<DenseI64ArrayAttr>(arg.getArgNumber(),
                                                     "block.strides");
    auto group = fn.getArgAttrOfType<IntegerAttr>(arg.getArgNumber(),
                                                  "block.alias_group");
    if (t.getRank() != 2 || !st || !sh || !group || group.getInt() < 0 ||
        !groups.insert(group.getInt()).second || sh[0] < owners.getInt() ||
        st[1] != 1 || st[0] < sh[1] || sh[1] <= 0 ||
        t.getElementTypeBitWidth() != 32 || sh[1] > local->budget / 4)
      return {};
    bool access = false;
    for (auto *op : arg.getUsers()) {
      if (op == &space || isa<memref::DimOp>(op))
        continue;
      ValueRange cs;
      if (auto ld = dyn_cast<memref::LoadOp>(op))
        cs = ld.getIndices();
      else if (auto store = dyn_cast<memref::StoreOp>(op)) {
        cs = store.getIndices();
        plan.writes.insert(arg);
      } else
        return {};
      if (cs.size() != 2 || cs[0] != row)
        return {};
      access = true;
    }
    if (access) {
      plan.views.push_back(arg);
      plan.bytes += llvm::alignTo(sh[1] * 4, local->alignment);
    }
  }
  plan.bytes += 2 * llvm::alignTo(plan.scanWidth * 4, local->alignment);
  if (plan.k)
    plan.bytes += llvm::alignTo(3 * plan.selectionWidth * 4, local->alignment) +
                  2 * llvm::alignTo(plan.k * 4, local->alignment);
  if (plan.bytes > std::min(local->budget, local->capacity))
    return {};
  return plan;
}
} // namespace intentdsl::dsa
