#include "MemberAccess.h"
#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/KeyReduction.h"
#include "intentdsl/levels/algorithm/kir/ir/ProjectedLayout.h"
#include "intentdsl/levels/algorithm/kir/ir/Segment.h"
#include "intentdsl/levels/execution/cpu/ir/IndexedEffects.h"
#include "intentdsl/levels/execution/cpu/ir/MatrixTask.h"
#include "intentdsl/levels/execution/cpu/ir/Pack.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include <algorithm>
#include "Dialect.h.inc"
using namespace mlir;
#include "Lanes.h"
namespace CI_UNIT_NAMESPACE {
LogicalResult lowerMojoMatrix(func::FuncOp);
LogicalResult lowerMojoOrdered(func::FuncOp);
LogicalResult lowerMojoRow(func::FuncOp, int64_t);
void dialect(DialectRegistry &r) {
  r.insert<intentdsl::rowmojo::RowMojoDialect, mlir::arith::ArithDialect,
           mlir::math::MathDialect, mlir::scf::SCFDialect>();
}
namespace {
class Columns {
  OpBuilder b;
  Operation *task;
  Location loc;
  IRMapping reduced, tile;
  Value col;
  int64_t width;
  SmallVector<Operation *> stores, ownerStores;
  Operation *make(StringRef name, ValueRange args, TypeRange types,
                  ArrayRef<NamedAttribute> attrs = {}) {
    OperationState s(loc, ("mojo_row." + name).str());
    s.addOperands(args);
    s.addTypes(types);
    s.addAttributes(attrs);
    return b.create(s);
  }
  Value expression(Value v) {
    if (auto known = reduced.lookupOrNull(v))
      return known;
    if (auto known = tile.lookupOrNull(v))
      return known;
    auto *op = v.getDefiningOp();
    if (!op)
      return v;
    auto n = op->getName().stripDialect();
    Type t = v.getType();
    if (auto ranked = dyn_cast<RankedTensorType>(t))
      t = VectorType::get({width}, ranked.getElementType());
    Value result;
    if (n == "ordered_sum" || n == "segment_length") {
      bool fold = n == "ordered_sum";
      OperationState s(loc,
                       fold ? "mojo_row.segment_fold" : "mojo_row.segment_size");
      s.addOperands({op->getOperand(0), op->getOperand(1), col});
      s.addTypes(t);
      if (fold) {
        IRMapping mapping;
        op->getRegion(0).cloneInto(s.addRegion(), mapping);
      }
      auto *physical = b.create(s);
      if (fold && intentdsl::kir::contiguousFeatureRead(physical))
        physical->setAttr("feature_vector", b.getUnitAttr());
      result = physical->getResult(0);
    } else if (n == "count_keys" || n == "column_sum") {
      result = make(n == "count_keys" ? "key_counts" : "sum_columns",
                    {op->getOperand(0), op->getOperand(1), col}, t,
                    op->getAttrs())
                   ->getResult(0);
    } else if (n == "read_owner")
      result = make("owner_read", op->getOperands(), t)->getResult(0);
    else if (n == "gather_row")
      result = make("indexed_read",
                    {op->getOperand(0), op->getOperand(1), op->getOperand(2),
                     col},
                    t)
                   ->getResult(0);
    else if (n == "ordinal") {
      result = make("apply", {task->getRegion(0).front().getArgument(0), col},
                    t, b.getNamedAttr("kind", b.getStringAttr("ordinal")))
                   ->getResult(0);
    } else if (n == "read")
      result = make("read",
                    {op->getOperand(0),
                     task->getRegion(0).front().getArgument(0), col},
                    t)
                   ->getResult(0);
    else {
      SmallVector<Value> args;
      for (auto value : op->getOperands())
        args.push_back(expression(value));
      result =
          make("apply", args, t, b.getNamedAttr("kind", b.getStringAttr(n)))
              ->getResult(0);
      if (n == "scan_sum") {
        unsigned index = 0;
        for (auto &prior : *op->getBlock()) {
          if (&prior == op)
            break;
          index += prior.getName().stripDialect() == "scan_sum";
        }
        result.getDefiningOp()->setAttr("carry", b.getI64IntegerAttr(index));
      }
    }
    if (auto c = op->getAttr("value"))
      result.getDefiningOp()->setAttr("value", c);
    if (auto pos = op->getAttr("position"))
      result.getDefiningOp()->setAttr("position", pos);
    tile.map(v, result);
    return result;
  }

  Operation *sweepGroup(ArrayRef<Operation *> ops, StringRef range,
                        ValueRange prevs, int64_t lanes, StringRef cacheMode,
                        ArrayRef<Value> cacheWriteVals,
                        ArrayRef<Value> cacheReadVals) {
    OpBuilder::InsertionGuard restore(b);
    width = lanes;
    tile.clear();
    Operation *first = ops.front();
    auto code = first->getName().stripDialect();
    bool owned = code == "write_owner";
    bool store = code == "write" || owned;
    OperationState s(loc, "mojo_row.sweep");
    s.addOperands(prevs);
    if (!store) {
      for (Operation *ro : ops)
        s.addTypes(ro->getResult(0).getType());
    }
    s.addAttribute("width", b.getI64IntegerAttr(width));
    s.addAttribute("range", b.getStringAttr(range));
    s.addAttribute("algebra", b.getStringAttr(store ? "store" : code));
    if (!cacheMode.empty())
      s.addAttribute("cache_mode", b.getStringAttr(cacheMode));
    s.addRegion();
    auto *loop = b.create(s);
    auto &bb = loop->getRegion(0).emplaceBlock();
    col = bb.addArgument(b.getIndexType(), loc);
    b.setInsertionPointToEnd(&bb);
    for (auto [slotIdx, cv] : llvm::enumerate(cacheReadVals)) {
      Type cTy = cv.getType();
      if (auto rt = dyn_cast<RankedTensorType>(cTy))
        cTy = VectorType::get({width}, rt.getElementType());
      Value loaded =
          make("row_cache_load", {col}, {cTy},
               {b.getNamedAttr("slot", b.getI64IntegerAttr(slotIdx))})
              ->getResult(0);
      tile.map(cv, loaded);
    }
    if (store) {
      auto &writes = owned ? ownerStores : stores;
      for (auto *w : writes)
        expression(w->getOperand(2));
      for (auto *w : writes)
        if (owned)
          make("owner_write",
               {w->getOperand(0), w->getOperand(1),
                expression(w->getOperand(2))},
               {});
        else
          make("write",
               {w->getOperand(0), w->getOperand(1), col,
                expression(w->getOperand(2))},
               {}, w->getAttrs());
      make("yield", {}, {});
      return loop;
    }
    SmallVector<Value, 4> yVals;
    for (Operation *ro : ops)
      yVals.push_back(expression(ro->getOperand(0)));
    for (auto [slotIdx, cv] : llvm::enumerate(cacheWriteVals)) {
      Value cVal = expression(cv);
      make("row_cache_store", {col, cVal}, {},
           {b.getNamedAttr("slot", b.getI64IntegerAttr(slotIdx))});
    }
    make("yield", yVals, {});
    return loop;
  }

  Value sweep(Operation *op, StringRef range, Value prev, int64_t lanes) {
    SmallVector<Value, 1> prevs;
    if (prev)
      prevs.push_back(prev);
    auto *loop = sweepGroup({op}, range, prevs, lanes, "", {}, {});
    return loop->getNumResults() > 0 ? loop->getResult(0) : Value{};
  }

public:
  static bool storeInterval(ArrayRef<Operation *> body, Operation *write) {
    if (body.size() != 2 || body[1] != write ||
        body[0]->getName().stripDialect() != "read" ||
        write->getName().stripDialect() != "write" ||
        write->getOperand(2) != body[0]->getResult(0))
      return false;
    auto src = cast<MemRefType>(body[0]->getOperand(0).getType());
    auto dst = cast<MemRefType>(write->getOperand(0).getType());
    return src.getRank() == 2 && dst.getRank() == 2 &&
           src.getElementType() == dst.getElementType();
  }
  Columns(Operation *op) : b(op), task(op), loc(op->getLoc()) {}
  void run(int64_t lanes) {
    auto &bb = task->getRegion(0).front();
    SmallVector<Operation *> original;
    for (auto &op : bb.without_terminator())
      original.push_back(&op);
    bool hasUniform = false;
    for (auto *op : original) {
      auto n = op->getName().stripDialect();
      if (n == "write")
        stores.push_back(op);
      else if (n == "write_owner")
        ownerStores.push_back(op);
      else if (n == "uniform")
        hasUniform = true;
    }
    b.setInsertionPoint(&bb.back());
    auto ordered = original;
    std::stable_partition(ordered.begin(), ordered.end(), [](Operation *op) {
      auto n = op->getName().stripDialect();
      return n != "write" && n != "write_owner";
    });

    auto isReduction = [](Operation *op) {
      auto n = op->getName().stripDialect();
      return n == "sum" || n == "max_num" || n == "welford";
    };

    // Group reductions into dependency stages (fusing independent sums)
    SmallVector<SmallVector<Operation *, 4>, 4> redStages;
    llvm::DenseSet<Operation *> scheduledReds;
    for (auto *op : ordered) {
      if (!isReduction(op) || scheduledReds.contains(op))
        continue;
      if (op->getName().stripDialect() != "sum") {
        scheduledReds.insert(op);
        redStages.push_back({op});
        continue;
      }
      auto dependsOnUnscheduledRed = [&](Operation *root) {
        llvm::DenseSet<Operation *> seen;
        SmallVector<Operation *, 16> work{root};
        while (!work.empty()) {
          Operation *cur = work.pop_back_val();
          if (!seen.insert(cur).second)
            continue;
          if (cur != root && isReduction(cur) && !scheduledReds.contains(cur))
            return true;
          if (cur != root && isReduction(cur))
            continue;
          for (Value in : cur->getOperands())
            if (auto *def = in.getDefiningOp(); def && def->getBlock() == &bb)
              work.push_back(def);
        }
        return false;
      };
      SmallVector<Operation *, 4> stage{op};
      for (auto *cand : ordered) {
        if (stage.size() >= 4)
          break;
        if (cand == op || cand->getName().stripDialect() != "sum" ||
            scheduledReds.contains(cand))
          continue;
        if (!dependsOnUnscheduledRed(cand))
          stage.push_back(cand);
      }
      for (auto *sOp : stage)
        scheduledReds.insert(sOp);
      redStages.push_back(std::move(stage));
    }

    // Check if a 2-pass uniform+reduction+store row task can retain intermediate
    // tensor values in per-thread L1d scratch across sweeps (cols <= 8192)
    SmallVector<Value, 4> cachedCut;
    if (hasUniform && redStages.size() == 1 && !stores.empty() &&
        ownerStores.empty()) {
      llvm::DenseSet<Operation *> s1Ops, s2Ops;
      {
        SmallVector<Operation *, 16> work;
        for (Operation *ro : redStages[0])
          if (auto *def = ro->getOperand(0).getDefiningOp();
              def && def->getBlock() == &bb)
            work.push_back(def);
        while (!work.empty()) {
          Operation *cur = work.pop_back_val();
          if (!s1Ops.insert(cur).second)
            continue;
          for (Value in : cur->getOperands())
            if (auto *def = in.getDefiningOp(); def && def->getBlock() == &bb)
              work.push_back(def);
        }
      }
      {
        SmallVector<Operation *, 16> work;
        for (Operation *wo : stores)
          if (auto *def = wo->getOperand(2).getDefiningOp();
              def && def->getBlock() == &bb && !scheduledReds.contains(def))
            work.push_back(def);
        while (!work.empty()) {
          Operation *cur = work.pop_back_val();
          if (!s2Ops.insert(cur).second)
            continue;
          for (Value in : cur->getOperands())
            if (auto *def = in.getDefiningOp();
                def && def->getBlock() == &bb && !scheduledReds.contains(def))
              work.push_back(def);
        }
      }
      llvm::DenseSet<Operation *> storeDirect;
      for (Operation *wo : stores)
        if (auto *def = wo->getOperand(2).getDefiningOp())
          storeDirect.insert(def);

      // Unwrap widening casts (bf16/f16 -> f32) from S1 boundary into S2 so the
      // narrower bf16/f16 value is cached in L1d instead of f32
      bool unwrapped = true;
      llvm::DenseSet<Operation *> forcedRecompute;
      while (unwrapped) {
        unwrapped = false;
        for (Operation *op : original) {
          if (!s1Ops.contains(op) || forcedRecompute.contains(op) ||
              op->getName().stripDialect() != "cast")
            continue;
          auto inTy = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
          auto outTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
          auto *inDef = op->getOperand(0).getDefiningOp();
          if (!inTy || !outTy || !inDef || !s1Ops.contains(inDef) ||
              inTy.getElementTypeBitWidth() >= outTy.getElementTypeBitWidth() ||
              !(inTy.getElementType().isBF16() ||
                inTy.getElementType().isF16()))
            continue;
          bool usedInS2 = storeDirect.contains(op) ||
                          llvm::any_of(op->getResult(0).getUsers(), [&](Operation *u) {
                            return (s2Ops.contains(u) && !s1Ops.contains(u)) ||
                                   forcedRecompute.contains(u);
                          });
          if (usedInS2) {
            forcedRecompute.insert(op);
            unwrapped = true;
          }
        }
      }

      bool validCut = true;
      for (Operation *op : original) {
        if (!s1Ops.contains(op) || forcedRecompute.contains(op) ||
            op->getNumResults() != 1)
          continue;
        Value v = op->getResult(0);
        bool usedOutside =
            storeDirect.contains(op) ||
            llvm::any_of(v.getUsers(), [&](Operation *u) {
              return (s2Ops.contains(u) && !s1Ops.contains(u)) ||
                     forcedRecompute.contains(u);
            });
        if (!usedOutside)
          continue;
        auto rt = dyn_cast<RankedTensorType>(v.getType());
        if (!rt || !(rt.getElementType().isBF16() ||
                     rt.getElementType().isF16() ||
                     rt.getElementType().isF32())) {
          validCut = false;
          break;
        }
        cachedCut.push_back(v);
      }
      if (!validCut || cachedCut.empty() || cachedCut.size() > 4)
        cachedCut.clear();
    }

    if (!cachedCut.empty()) {
      SmallVector<Attribute, 4> cacheTypes;
      for (Value cv : cachedCut)
        cacheTypes.push_back(
            TypeAttr::get(cast<ShapedType>(cv.getType()).getElementType()));
      task->setAttr("mojo_row.row_cache", b.getArrayAttr(cacheTypes));

      for (StringRef mode : {StringRef("cached"), StringRef("uncached")}) {
        reduced.clear();
        bool useCache = mode == "cached";
        auto *rFull = sweepGroup(
            redStages[0], "full", {}, lanes, mode,
            useCache ? ArrayRef<Value>(cachedCut) : ArrayRef<Value>{}, {});
        auto *rTail = sweepGroup(
            redStages[0], "tail", rFull->getResults(), 1, mode,
            useCache ? ArrayRef<Value>(cachedCut) : ArrayRef<Value>{}, {});
        for (auto [idx, ro] : llvm::enumerate(redStages[0]))
          reduced.map(ro->getResult(0), rTail->getResult(idx));
        sweepGroup({stores.front()}, "full", {}, lanes, mode, {},
                   useCache ? ArrayRef<Value>(cachedCut) : ArrayRef<Value>{});
        sweepGroup({stores.front()}, "tail", {}, 1, mode, {},
                   useCache ? ArrayRef<Value>(cachedCut) : ArrayRef<Value>{});
      }
      for (auto *op : llvm::reverse(original))
        op->erase();
      task->setAttr("mojo_row.schedule", b.getStringAttr("static-row-groups"));
      task->setAttr("mojo_row.lanes", b.getI64IntegerAttr(lanes));
      return;
    }

    size_t stageCursor = 0;
    for (auto *op : ordered) {
      auto n = op->getName().stripDialect();
      if (n == "write" && op != stores.front())
        continue;
      if (n == "write_owner")
        continue;
      if (isReduction(op)) {
        if (stageCursor >= redStages.size() ||
            redStages[stageCursor].front() != op)
          continue;
        auto &stage = redStages[stageCursor++];
        auto *full = sweepGroup(stage, "full", {}, lanes, "", {}, {});
        auto *done =
            sweepGroup(stage, "tail", full->getResults(), 1, "", {}, {});
        for (auto [idx, ro] : llvm::enumerate(stage))
          reduced.map(ro->getResult(0), done->getResult(idx));
        continue;
      }
      if (n != "write")
        continue;
      if (original.size() == 2 &&
          original.front()->getName().stripDialect() == "count_keys" &&
          op->getOperand(2) == original.front()->getResult(0)) {
        sweep(op, "private_bins", {}, 1);
        continue;
      }
      if (original.size() == 2 &&
          original.front()->getName().stripDialect() == "gather_row" &&
          op->getOperand(2) == original.front()->getResult(0)) {
        sweep(op, "indexed_copy", {}, 1);
        continue;
      }
      if (storeInterval(original, op)) {
        sweep(op, "copy", {}, 1);
        continue;
      }
      auto full = sweep(op, "full", {}, lanes);
      sweep(op, "tail", full, 1);
    }
    if (!ownerStores.empty())
      sweep(ownerStores.front(), "owners", {}, 1);
    for (auto *op : llvm::reverse(original))
      op->erase();
    task->setAttr("mojo_row.schedule", b.getStringAttr("static-row-groups"));
    task->setAttr("mojo_row.lanes", b.getI64IntegerAttr(lanes));
  }
};
struct Realize : PassWrapper<Realize, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Realize)
  int64_t lanes;
  intentdsl::CallFacts facts;
  Realize(int64_t lanes, const intentdsl::CallFacts &f)
      : lanes(lanes), facts(f) {}
  void runOnOperation() final {
    auto &first =
        (*getOperation().getOps<func::FuncOp>().begin()).front().front();
    if (first.getName().getStringRef() == "cpu.owner_task") {
      auto fn = first.getParentOfType<func::FuncOp>();
      getContext().getOrLoadDialect<intentdsl::rowmojo::RowMojoDialect>();
      getContext().getOrLoadDialect<mlir::arith::ArithDialect>();
      getContext().getOrLoadDialect<mlir::math::MathDialect>();
      getContext().getOrLoadDialect<mlir::scf::SCFDialect>();
      if (failed(memberAccesses(fn, true)) || failed(lowerMojoOrdered(fn)))
        signalPassFailure();
      return;
    }
    if (intentdsl::cpu::findMatrixTask(first.getParentOfType<func::FuncOp>())) {
      getContext().getOrLoadDialect<intentdsl::rowmojo::RowMojoDialect>();
      getContext().getOrLoadDialect<mlir::arith::ArithDialect>();
      getContext().getOrLoadDialect<mlir::math::MathDialect>();
      if (failed(lowerMojoMatrix(first.getParentOfType<func::FuncOp>())))
        signalPassFailure();
      return;
    }
    for (auto fn : getOperation().getOps<func::FuncOp>())
      fn.walk([&](Operation *op) {
        auto n = op->getName().stripDialect();
        if (n != "count_keys" && n != "column_sum")
          return;
        auto extent = intentdsl::kir::keyExtent(op, facts);
        if (!extent) {
          op->emitError() << llvm::toString(extent.takeError());
          signalPassFailure();
          return;
        }
        op->setAttr("extent",
                    IntegerAttr::get(IntegerType::get(&getContext(), 64),
                                     *extent));
      });
    if (failed(intentdsl::specializeProjectedLayout(
            *getOperation().getOps<func::FuncOp>().begin(), facts, "mojo_row",
            true))) {
      signalPassFailure();
      return;
    }
    getContext().getOrLoadDialect<intentdsl::rowmojo::RowMojoDialect>();
    getContext().getOrLoadDialect<mlir::arith::ArithDialect>();
    getContext().getOrLoadDialect<mlir::math::MathDialect>();
    getContext().getOrLoadDialect<mlir::scf::SCFDialect>();
    auto fn = *getOperation().getOps<func::FuncOp>().begin();
    auto *op = &fn.front().front();
    if (op->getName().getStringRef() != "cpu.task") {
      op->emitError("expected cpu.task for Mojo row realization");
      signalPassFailure();
      return;
    }
    int64_t effLanes = laneCount(op, lanes);
    auto &body = op->getRegion(0).front();
    bool isTile = !body.without_terminator().empty() &&
                  body.front().getName().getStringRef() == "cpu.transpose_tile";
    if (!isTile) {
      if (failed(intentdsl::cpu::packMemberLoop(body))) {
        signalPassFailure();
        return;
      }
      if (!intentdsl::cpu::packed(body) &&
          !intentdsl::cpu::reservations(body) &&
          !intentdsl::cpu::formReservations(body) &&
          !intentdsl::cpu::atomicMembers(body) &&
          !intentdsl::cpu::formAtomicMembers(body)) {
        Columns(op).run(effLanes);
      }
    }
    if (failed(lowerMojoRow(fn, effLanes)))
      signalPassFailure();
  }
};
} // namespace
void realize(OpPassManager &pm, int64_t lanes, const intentdsl::CallFacts &f) {
  pm.addPass(std::make_unique<Realize>(lanes, f));
}
} // namespace CI_UNIT_NAMESPACE
