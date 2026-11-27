#include "intentdsl/levels/execution/block/ir/FoldPanel.h"
#include "intentdsl/levels/storage/dsa/ir/Owners.h"

using namespace mlir;
namespace intentdsl::dsa {
// Split the pointwise suffix after a lane-private fold. All values crossing
// the split have explicit row storage and no intermediate write is observable.
static void planPointwise(Block &body, PrivateRows &rows,
                          const LocalSupply &local) {
  if (rows.lanes % 64)
    return;
  Operation *begin = nullptr;
  SmallVector<Value> inputs, temps;
  llvm::DenseSet<Value> values;
  int64_t work = 0;
  for (auto &op : body.without_terminator()) {
    auto k = op.getName().getStringRef();
    if (!begin && (k == "math.exp" || k == "arith.divf"))
      begin = &op;
    if (!begin)
      continue;
    if (auto store = dyn_cast<memref::StoreOp>(op)) {
      if (!rows.writes.contains(store.getMemref()) ||
          !values.contains(store.getValue()))
        return;
      continue;
    }
    if (!(k == "math.exp" || k == "arith.divf" || k == "arith.addf" ||
          k == "arith.subf" || k == "arith.mulf" || k == "arith.truncf" ||
          k == "arith.constant"))
      return;
    Type out = op.getResult(0).getType();
    if ((!out.isF32() && !(k == "arith.truncf" && out.isF16())) ||
        !llvm::all_of(op.getOperands(),
                      [](Value v) { return v.getType().isF32(); }))
      return;
    for (Value v : op.getOperands())
      if (values.insert(v).second)
        inputs.push_back(v);
    values.insert(op.getResult(0));
    temps.push_back(op.getResult(0));
    if (k == "math.exp")
      work = std::max<int64_t>(work, 1);
    if (k == "arith.divf")
      work = 6;
  }
  if (!begin || !work)
    return;
  int64_t extra = (inputs.size() + temps.size() + work) *
                  llvm::alignTo(rows.lanes * 4, local.alignment);
  if (extra > std::min(local.capacity, local.budget) - rows.bytes)
    return;
  rows.bytes += extra;
  rows.pointwise = begin;
  rows.inputs = std::move(inputs);
  rows.temporaries = std::move(temps);
  rows.workArrays = work;
}
// Recognize an affine tap displacement lane + tap + c. All three
// coefficients must be known and all index arithmetic representable.
static std::optional<std::array<int64_t, 3>>
windowIndex(Value v, Value lane, Value tap, func::FuncOp fn) {
  using Coeff = std::array<int64_t, 3>;
  if (v == lane)
    return Coeff{1, 0, 0};
  if (v == tap)
    return Coeff{0, 1, 0};
  if (auto c = v.getDefiningOp<arith::ConstantIndexOp>())
    return Coeff{0, 0, c.value()};
  if (auto dim = v.getDefiningOp<memref::DimOp>();
      dim && dim.getConstantIndex()) {
    if (auto n = block::orderedArgDim(fn, nullptr, dim.getSource(),
                                      *dim.getConstantIndex()))
      return Coeff{0, 0, *n};
  }
  auto *op = v.getDefiningOp();
  if (!op || !isa<arith::AddIOp, arith::SubIOp>(op))
    return {};
  auto a = windowIndex(op->getOperand(0), lane, tap, fn),
       b = windowIndex(op->getOperand(1), lane, tap, fn);
  if (!a || !b)
    return {};
  for (int i = 0; i < 3; ++i)
    if (isa<arith::AddIOp>(op)
            ? __builtin_add_overflow((*a)[i], (*b)[i], &(*a)[i])
            : __builtin_sub_overflow((*a)[i], (*b)[i], &(*a)[i]))
      return {};
  return a;
}
static void matchWindowFold(func::FuncOp fn, Block &bb, PrivateRows &r,
                            const LocalSupply &local) {
  if (r.lanes % 64)
    return;
  auto fzero = [](Value v) {
    auto c = v.getDefiningOp<arith::ConstantOp>();
    auto a = c ? dyn_cast<FloatAttr>(c.getValue()) : FloatAttr{};
    return a && a.getValue().isPosZero();
  };
  auto izero = [](Value v, int64_t n) {
    auto c = v.getDefiningOp<arith::ConstantIndexOp>();
    return c && c.value() == n;
  };
  auto widening = [](Value v) {
    auto ext = v.getDefiningOp<arith::ExtFOp>();
    return ext && ext.getIn().getType().isF16() && ext.getType().isF32()
               ? ext.getIn().getDefiningOp<memref::LoadOp>()
               : memref::LoadOp{};
  };
  for (auto loop : bb.getOps<scf::ForOp>()) {
    auto n = block::orderedStaticLoopExtent(loop, fn, nullptr);
    if (!n || *n < 1 || *n > r.lanes || !izero(loop.getLowerBound(), 0) ||
        !izero(loop.getStep(), 1) || loop.getNumResults() != 1 ||
        !fzero(loop.getInitArgs()[0]))
      continue;
    auto add = loop.getBody()
                   ->getTerminator()
                   ->getOperand(0)
                   .getDefiningOp<arith::AddFOp>();
    if (!add || add.getLhs() != loop.getRegionIterArgs()[0])
      continue;
    auto mul = add.getRhs().getDefiningOp<arith::MulFOp>();
    if (!mul)
      continue;
    auto cond = mul.getLhs().getDefiningOp<scf::IfOp>();
    auto weight = widening(mul.getRhs());
    if (!cond || !weight ||
        weight.getIndices().back() != loop.getInductionVar() ||
        !cond.elseBlock() || !fzero(cond.elseYield().getOperand(0)))
      continue;
    auto input = widening(cond.thenYield().getOperand(0));
    auto guard = cond.getCondition().getDefiningOp<arith::CmpIOp>();
    if (!input || !guard || guard.getPredicate() != arith::CmpIPredicate::sge ||
        guard.getLhs() != input.getIndices().back() ||
        !izero(guard.getRhs(), 0))
      continue;
    auto idx = windowIndex(guard.getLhs(), bb.getArguments().back(),
                           loop.getInductionVar(), fn);
    if (!idx || *idx != std::array<int64_t, 3>{1, 1, 1 - *n})
      continue;
    auto a = input.getMemRef(), w = weight.getMemRef();
    if (!r.prefixes.count(a) || !r.prefixes.count(w) || r.writes.contains(a) ||
        r.writes.contains(w))
      continue;
    auto length = [&](Value v) {
      return block::orderedArgDim(fn, nullptr, v,
                                  cast<MemRefType>(v.getType()).getRank() - 1);
    };
    if (length(a) != r.lanes || length(w) != *n)
      continue;
    if (loop.walk([&](Operation *op) {
              return op == loop || op == cond || op == input || op == weight ||
                             isMemoryEffectFree(op)
                         ? WalkResult::advance()
                         : WalkResult::interrupt();
            })
            .wasInterrupted())
      continue;
    // Padded half row, widened product, FP32 accumulator.
    int64_t bytes = llvm::alignTo((r.lanes + *n - 1) * 2, local.alignment) +
                    2 * llvm::alignTo(r.lanes * 4, local.alignment);
    if (bytes > std::min(local.capacity, local.budget) - r.bytes)
      return;
    r.bytes += bytes;
    r.windowFold = PrivateRows::WindowFold{loop, a, w, *n};
    return;
  }
}
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
  planPointwise(bb, plan, local);
  matchWindowFold(fn, bb, plan, local);
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
