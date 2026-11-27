#include "intentdsl/levels/execution/block/ir/FoldPanel.h"
#include "intentdsl/levels/storage/dsa/ir/Owners.h"

using namespace mlir;
namespace intentdsl::dsa {
// Neighboring positions share a contiguous input window. A task owns the
// entire output row, including any boundary predicates on the member loads.
static std::optional<PrivateRows> windowRows(func::FuncOp fn, Operation &space,
                                             const LocalSupply &local) {
  auto axes = space.getAttrOfType<DenseI64ArrayAttr>("axes");
  auto &bb = space.getRegion(0).front();
  if (!axes || axes.size() != bb.getNumArguments() || axes.size() < 2)
    return {};
  auto shape = [&](Value v) {
    return fn.getArgAttrOfType<DenseI64ArrayAttr>(
        cast<BlockArgument>(v).getArgNumber(), "block.shape");
  };
  auto domain = shape(space.getOperands().back());
  if (!domain)
    return {};
  PrivateRows plan;
  plan.lanes = domain[axes.asArrayRef().back()];
  auto owners = space.getAttrOfType<IntegerAttr>("block.owners");
  if (plan.lanes <= 1 || !owners ||
      owners.getInt() / plan.lanes < local.consumers)
    return {};
  Value lane = bb.getArguments().back();
  bool hasWindow = false;
  llvm::DenseSet<int64_t> groups;
  for (BlockArgument v : fn.getArguments()) {
    auto mt = dyn_cast<MemRefType>(v.getType());
    if (!mt)
      continue;
    auto group =
        fn.getArgAttrOfType<IntegerAttr>(v.getArgNumber(), "block.alias_group");
    auto st = fn.getArgAttrOfType<DenseI64ArrayAttr>(v.getArgNumber(),
                                                     "block.strides");
    auto sh = shape(v);
    if (!group || group.getInt() < 0 || !groups.insert(group.getInt()).second ||
        !sh || !st || !mt.getRank() || st.asArrayRef().back() != 1)
      return {};
    bool read = false, write = false;
    for (Operation *use : v.getUsers()) {
      if (use == &space || isa<memref::DimOp>(use))
        continue;
      ValueRange coords;
      if (auto ld = dyn_cast<memref::LoadOp>(use)) {
        read = true;
        coords = ld.getIndices();
        auto loop = ld->getParentOfType<scf::ForOp>();
        if (loop && ownerValueDependsOn(coords.back(), lane) &&
            ownerValueDependsOn(coords.back(), loop.getInductionVar()))
          hasWindow = true;
      } else if (auto stOp = dyn_cast<memref::StoreOp>(use)) {
        write = true;
        coords = stOp.getIndices();
        if (use->getBlock() != &bb || coords.size() != bb.getNumArguments() ||
            !llvm::equal(coords, bb.getArguments()))
          return {};
        int64_t span = 1;
        for (int i = coords.size() - 1; i >= 0; --i) {
          auto ds = shape(space.getOperand(i));
          if (!ds || ds[axes[i]] != sh[i] || st[i] < span)
            return {};
          int64_t delta;
          if (sh[i] <= 0 || __builtin_mul_overflow(sh[i] - 1, st[i], &delta) ||
              __builtin_add_overflow(span, delta, &span))
            return {};
        }
      } else
        return {};
      SmallVector<Value> prefix(coords.drop_back());
      for (auto [ax, c] : llvm::enumerate(prefix)) {
        auto ba = dyn_cast<BlockArgument>(c);
        if (!ba || ba.getOwner() != &bb || c == lane)
          return {};
        auto ds = shape(space.getOperand(ba.getArgNumber()));
        if (!ds || ds[axes[ba.getArgNumber()]] > sh[ax])
          return {};
      }
      if (plan.prefixes.count(v) && plan.prefixes.lookup(v) != prefix)
        return {};
      plan.prefixes[v] = prefix;
    }
    if (read && write)
      return {};
    if (write)
      plan.writes.insert(v);
    if (read || write) {
      auto len = sh.asArrayRef().back();
      if (len <= 0 || len > local.budget)
        return {};
      plan.views.push_back(v);
      plan.bytes +=
          llvm::alignTo(len * llvm::divideCeil(mt.getElementTypeBitWidth(), 8u),
                        local.alignment);
    }
  }
  if (!hasWindow || plan.writes.empty() ||
      plan.bytes > std::min(local.capacity, local.budget))
    return {};
  return plan;
}
std::optional<PrivateRows> privateRows(func::FuncOp fn,
                                       const compilerinfra::TargetProfile &p) {
  auto local = p.services.get<LocalSupply>();
  if (!local)
    return {};
  auto &space = fn.front().front();
  auto &bb = space.getRegion(0).front();
  if (bb.getNumArguments() != 1)
    return windowRows(fn, space, *local);
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
