#include "intentdsl/levels/execution/cpu/ir/MatrixTask.h"
#include "intentdsl/levels/execution/cpu/ir/Owners.h"
#include "intentdsl/levels/execution/cpu/ir/ScoreCache.h"
#include "intentdsl/levels/execution/cpu/ir/RepeatedScore.h"
#include "intentdsl/levels/execution/cpu/ir/OwnerTraversal.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
using namespace mlir;
using namespace compilerinfra;
namespace intentdsl::cpu {
namespace {
void eraseDeadInBlock(Block &bb, RewriterBase &rw) {
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> candidates;
    for (Operation &op : bb.without_terminator())
      candidates.push_back(&op);
    for (Operation *op : llvm::reverse(candidates)) {
      if (!op->use_empty() || op->getNumRegions() != 0)
        continue;
      if (isMemoryEffectFree(op) || isa<memref::LoadOp, memref::DimOp>(op) ||
          op->getName().stripDialect() == "slice_read") {
        rw.eraseOp(op);
        changed = true;
      }
    }
  }
}
LogicalResult verifyOwnerLayouts(func::FuncOp fn, Operation &space,
                                 const TargetProfile &p) {
  auto facts = p.services.get<CallFacts>();
  auto grain = space.getAttrOfType<IntegerAttr>("cpu.grain");
  if (facts && failed(kir::verifyProjectedOwners(space, *facts)))
    return failure();
  if (!facts || !grain || grain.getInt() < 1 || grain.getInt() > 256 ||
      !llvm::isPowerOf2_64(grain.getInt()) ||
      space.getAttr("cpu.owners") !=
          Builder(fn.getContext())
              .getI64IntegerAttr(kir::ownerCount(&space, *facts)))
    return space.emitError(
        "owner range or task grain disagrees with call facts");
  for (auto arg : fn.getArguments())
    if (isa<MemRefType>(arg.getType()))
      for (auto key : {"shape", "strides"}) {
        auto *input =
            facts->arguments[arg.getArgNumber()].getAsObject()->getArray(key);
        auto attr = fn.getArgAttrOfType<DenseI64ArrayAttr>(
            arg.getArgNumber(), std::string("cpu.") + key);
        if (!input || !attr || attr.size() != input->size())
          return fn.emitError("missing task layout");
        for (auto [a, v] : llvm::zip(attr.asArrayRef(), *input))
          if (v.getAsInteger() != a)
            return fn.emitError("incorrect task layout");
      }
  return success();
}
} // namespace

bool ownerTask(func::FuncOp fn) {
  return fn && !fn.empty() &&
         fn.front().front().getName().getStringRef() == "cpu.owner_task";
}
LogicalResult verifyOwnerGrain(func::FuncOp fn, Operation &space,
                               const TargetProfile &p) {
  if (failed(kir::verifyOwnerBody(fn, space, "cpu")))
    return failure();
  auto facts = p.services.get<CallFacts>();
  auto grain = space.getAttrOfType<IntegerAttr>("cpu.grain");
  auto cached = space.getAttrOfType<IntegerAttr>("cpu.score_cache");
  if (auto axis = space.getAttrOfType<IntegerAttr>("cpu.innermost_owner")) {
    auto reuse = facts ? sharedInterval(&space, *facts) : std::nullopt;
    auto batch = space.getAttrOfType<IntegerAttr>("cpu.owner_batch");
    int64_t w = batch ? batch.getInt() : 0;
    if (!reuse || axis.getInt() != reuse->axis || cached ||
        !llvm::isPowerOf2_64(w) || w < 2 ||
        w > p.services.get<RowMachine>()->packedOwners || !grain ||
        grain.getInt() % w || reuse->group % w || reuse->extent % w)
      return space.emitError("shared interval must project identical input "
                             "coordinates across independent owners");
    auto bad = space.walk([&](Operation *op) {
      return op->hasAttr("cpu.owner_common") !=
                     (op->getNumResults() == 1 &&
                      invariantOwnerValue(op->getResult(0), space, *reuse))
                 ? WalkResult::interrupt()
                 : WalkResult::advance();
    });
    if (bad.wasInterrupted())
      return space.emitError("invalid common value for packed owners");
  } else if (space.hasAttr("cpu.owner_batch") ||
             space
                 .walk([](Operation *op) {
                   return op->hasAttr("cpu.owner_common")
                              ? WalkResult::interrupt()
                              : WalkResult::advance();
                 })
                 .wasInterrupted())
    return space.emitError(
        "packed owner annotations require a shared interval");
  auto panel = cached ? reusableScores(&space, p) : std::nullopt;
  if (cached && (!panel || cached.getInt() != panel->keys))
    return space.emitError(
        "score cache requires two equivalent contractions over the key axis");
  auto batch = space.getAttrOfType<IntegerAttr>("cpu.owner_batch");
  auto repeat = batch ? repeatedScore(space) : std::nullopt;
  auto window = space.getAttrOfType<IntegerAttr>("cpu.batch_score_window");
  if (window &&
      (!repeat || window.getInt() != batchScoreSlots / batch.getInt()))
    return space.emitError("invalid private score window");
  Operation *expectedRecord =
      panel ? panel->maxScore.getDefiningOp()
            : (window && repeat ? repeat->record : nullptr);
  Operation *expectedReplay =
      panel ? panel->weightScore.getDefiningOp()
            : (window && repeat ? repeat->replay : nullptr);
  auto marks = space.walk([&](Operation *v) {
    if (v->hasAttr("cpu.record_score") != (v == expectedRecord) ||
        v->hasAttr("cpu.replay_score") != (v == expectedReplay))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (marks.wasInterrupted())
    return space.emitError(
        "score recording and replay must refer to the same key and query");
  return verifyOwnerLayouts(fn, space, p);
}

llvm::Error materializeOwnerBuffers(func::FuncOp fn, RewriterBase &rw,
                                    StringRef realization) {
  auto &task = fn.front().front();
  task.setAttr("cpu.realization", rw.getStringAttr(realization));
  task.removeAttr("cpu.storage");
  auto &space = task.getRegion(0).front().front();
  auto &body = space.getRegion(0).front();
  Location loc = space.getLoc();
  OpBuilder::InsertionGuard guard(rw);

  auto cached = space.getAttrOfType<IntegerAttr>("cpu.score_cache");
  auto window = space.getAttrOfType<IntegerAttr>("cpu.batch_score_window");
  if (!cached && !window)
    return llvm::Error::success();

  Operation *recordOp = nullptr, *replayOp = nullptr;
  space.walk([&](Operation *op) {
    if (op->hasAttr("cpu.record_score"))
      recordOp = op;
    if (op->hasAttr("cpu.replay_score"))
      replayOp = op;
  });
  if (!recordOp || !replayOp)
    return llvm::createStringError(
        "missing score record/replay marks for local buffer materialization");

  int64_t slots = cached ? cached.getInt() : window.getInt();
  StringRef role = cached ? "score_cache" : "score_window";
  rw.setInsertionPointToStart(&body);
  OperationState bufState(loc, "cpu.local_buffer");
  bufState.addAttribute("elements", rw.getI64IntegerAttr(slots));
  bufState.addAttribute("element_type", TypeAttr::get(rw.getF32Type()));
  bufState.addAttribute("role", rw.getStringAttr(role));
  bufState.addTypes(MemRefType::get({slots}, rw.getF32Type()));
  Value buf = rw.create(bufState)->getResult(0);

  if (cached) {
    auto loop1 = recordOp->getParentOfType<scf::ForOp>();
    auto loop2 = replayOp->getParentOfType<scf::ForOp>();
    if (!loop1 || !loop2)
      return llvm::createStringError("score cache marks must reside in folds");
    recordOp->removeAttr("cpu.record_score");
    rw.setInsertionPointAfter(recordOp);
    OperationState st(loc, "cpu.buffer_store");
    st.addOperands({recordOp->getResult(0), buf, loop1.getInductionVar()});
    rw.create(st);

    rw.setInsertionPoint(replayOp);
    OperationState ld(loc, "cpu.buffer_load");
    ld.addOperands({buf, loop2.getInductionVar()});
    ld.addTypes(replayOp->getResult(0).getType());
    Value loaded = rw.create(ld)->getResult(0);
    replayOp->replaceAllUsesWith(ValueRange{loaded});
    rw.eraseOp(replayOp);
    eraseDeadInBlock(*loop2.getBody(), rw);
    return llvm::Error::success();
  }

  // Materialize windowed score recording, replay, and over-window recompute
  auto loop1 = recordOp->getParentOfType<scf::ForOp>();
  auto loop2 = replayOp->getParentOfType<scf::ForOp>();
  if (!loop1 || !loop2 || loop1->getBlock() != &body ||
      loop2->getBlock() != &body)
    return llvm::createStringError(
        "windowed score marks must reside in top-level owner folds");

  auto markCommon = [&](Operation *op) -> Value {
    op->setAttr("cpu.owner_common", rw.getUnitAttr());
    return op->getResult(0);
  };

  Value lo = loop1.getLowerBound();
  Value hi = loop1.getUpperBound();
  recordOp->removeAttr("cpu.record_score");

  rw.setInsertionPointAfter(recordOp);
  Value off1 =
      markCommon(rw.create<arith::SubIOp>(loc, loop1.getInductionVar(), lo));
  Value win1 = markCommon(rw.create<arith::ConstantIndexOp>(loc, slots));
  Value inWin = markCommon(
      rw.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, off1, win1));
  auto storeIf = rw.create<scf::IfOp>(loc, TypeRange{}, inWin, false);
  storeIf->setAttr("cpu.window_guard", rw.getUnitAttr());
  rw.setInsertionPointToStart(&storeIf.getThenRegion().front());
  OperationState st(loc, "cpu.buffer_store");
  st.addOperands({recordOp->getResult(0), buf, off1});
  rw.create(st);

  // Build two-way scf.if (hi - lo <= slots) around loop2
  rw.setInsertionPoint(loop2);
  Value span = markCommon(rw.create<arith::SubIOp>(loc, hi, lo));
  Value win2 = markCommon(rw.create<arith::ConstantIndexOp>(loc, slots));
  Value fits = markCommon(
      rw.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle, span, win2));
  auto replayIf =
      rw.create<scf::IfOp>(loc, loop2.getResultTypes(), fits, true);
  replayIf->setAttr("cpu.window_branch", rw.getUnitAttr());

  // Then branch: clone loop2, replace replayOp with buffer_load, erase dead score ops
  {
    auto &thenBb = replayIf.getThenRegion().front();
    rw.setInsertionPointToStart(&thenBb);
    IRMapping map;
    auto replayLoop = cast<scf::ForOp>(rw.clone(*loop2.getOperation(), map));
    Operation *clonedReplay = nullptr;
    for (Operation &op : *replayLoop.getBody()) {
      if (op.hasAttr("cpu.replay_score")) {
        clonedReplay = &op;
        break;
      }
    }
    if (!clonedReplay)
      return llvm::createStringError("missing cloned replay op in window");
    rw.setInsertionPoint(clonedReplay);
    Value off2 = markCommon(
        rw.create<arith::SubIOp>(loc, replayLoop.getInductionVar(), lo));
    OperationState ld(loc, "cpu.buffer_load");
    ld.addOperands({buf, off2});
    ld.addTypes(clonedReplay->getResult(0).getType());
    Value loaded = rw.create(ld)->getResult(0);
    clonedReplay->replaceAllUsesWith(ValueRange{loaded});
    rw.eraseOp(clonedReplay);
    eraseDeadInBlock(*replayLoop.getBody(), rw);

    rw.setInsertionPointToEnd(&thenBb);
    if (thenBb.mightHaveTerminator())
      thenBb.getTerminator()->erase();
    rw.create<scf::YieldOp>(loc, replayLoop.getResults());
  }

  // Else branch: move original loop2 for over-window recomputation
  replayOp->removeAttr("cpu.replay_score");
  loop2->replaceAllUsesWith(replayIf.getResults());
  {
    auto &elseBb = replayIf.getElseRegion().front();
    if (elseBb.mightHaveTerminator())
      elseBb.getTerminator()->erase();
    loop2->moveBefore(&elseBb, elseBb.end());
    rw.setInsertionPointToEnd(&elseBb);
    rw.create<scf::YieldOp>(loc, loop2.getResults());
  }
  return llvm::Error::success();
}

LogicalResult verifyOwnerBuffers(func::FuncOp fn, const TargetProfile &p,
                                 StringRef realization) {
  auto &task = fn.front().front();
  Builder b(fn.getContext());
  if (task.getOperands() != fn.getArguments() ||
      !llvm::hasSingleElement(task.getRegion(0)) ||
      task.getRegion(0).front().getOperations().size() != 2 ||
      fn.front().getOperations().size() != 3 ||
      task.getNextNode()->getName().getStringRef() != "cpu.join" ||
      task.getNextNode()->getOperand(0) != task.getResult(0) ||
      task.getRegion(0).front().back().getName().getStringRef() != "cpu.end")
    return task.emitError("invalid ordered task captures or completion");
  if (task.hasAttr("cpu.storage") ||
      task.getAttr("cpu.realization") != b.getStringAttr(realization))
    return task.emitError("incorrect ordered task carry storage");

  auto &space = task.getRegion(0).front().front();
  if (space.getName().getStringRef() != "cpu.owner_space" ||
      !llvm::hasSingleElement(space.getRegion(0)))
    return space.emitError("expected cpu.owner_space");
  if (failed(verifyOwnerLayouts(fn, space, p)))
    return failure();

  // No unmaterialized score marks may remain at cpu@buffers
  bool lingeringMarks = space
                            .walk([](Operation *op) {
                              return (op->hasAttr("cpu.record_score") ||
                                      op->hasAttr("cpu.replay_score"))
                                         ? WalkResult::interrupt()
                                         : WalkResult::advance();
                            })
                            .wasInterrupted();
  if (lingeringMarks)
    return space.emitError("unmaterialized score record/replay mark at cpu@buffers");

  auto cached = space.getAttrOfType<IntegerAttr>("cpu.score_cache");
  auto window = space.getAttrOfType<IntegerAttr>("cpu.batch_score_window");
  auto batch = space.getAttrOfType<IntegerAttr>("cpu.owner_batch");
  if (window && (!batch || window.getInt() != batchScoreSlots / batch.getInt()))
    return space.emitError("invalid materialized score window");

  Operation *localBuf = nullptr;
  unsigned stores = 0, loads = 0;
  bool badBuf =
      space
          .walk([&](Operation *op) {
            auto n = op->getName().getStringRef();
            if (n == "cpu.local_buffer") {
              if (localBuf || op->getBlock() != &space.getRegion(0).front())
                return WalkResult::interrupt();
              auto elems = op->getAttrOfType<IntegerAttr>("elements");
              auto role = op->getAttrOfType<StringAttr>("role");
              auto mt = dyn_cast<MemRefType>(op->getResult(0).getType());
              if (!elems || elems.getInt() <= 0 || !mt || mt.getRank() != 1 ||
                  !mt.getElementType().isF32() || !role)
                return WalkResult::interrupt();
              localBuf = op;
            } else if (n == "cpu.buffer_store") {
              ++stores;
              if (!localBuf || op->getNumOperands() != 3 ||
                  op->getOperand(1) != localBuf->getResult(0) ||
                  !op->getOperand(0).getType().isF32() ||
                  !op->getOperand(2).getType().isIndex())
                return WalkResult::interrupt();
            } else if (n == "cpu.buffer_load") {
              ++loads;
              if (!localBuf || op->getNumOperands() != 2 ||
                  op->getOperand(0) != localBuf->getResult(0) ||
                  !op->getResult(0).getType().isF32() ||
                  !op->getOperand(1).getType().isIndex())
                return WalkResult::interrupt();
            }
            return WalkResult::advance();
          })
          .wasInterrupted();
  if (badBuf)
    return space.emitError("invalid task-local buffer operation in cpu.owner_space");

  if (cached || window) {
    int64_t expected = cached ? cached.getInt() : window.getInt();
    if (!localBuf ||
        localBuf->getAttrOfType<IntegerAttr>("elements").getInt() != expected ||
        stores != 1 || loads != 1)
      return space.emitError("missing materialized local score buffer, store, or load");
    if (window) {
      bool hasGuard = false, hasBranch = false;
      for (auto ifOp : space.getRegion(0).front().getOps<scf::IfOp>()) {
        if (ifOp->hasAttr("cpu.window_branch")) {
          hasBranch = !ifOp.getElseRegion().empty();
          unsigned elseLoads = 0;
          ifOp.getElseRegion().walk([&](Operation *o) {
            if (o->getName().getStringRef() == "cpu.buffer_load")
              ++elseLoads;
          });
          if (elseLoads != 0)
            return ifOp.emitError(
                "over-window recomputation branch must not read score window");
        }
      }
      space.walk([&](scf::IfOp ifOp) {
        if (ifOp->hasAttr("cpu.window_guard"))
          hasGuard = true;
      });
      if (!hasGuard || !hasBranch)
        return space.emitError(
            "windowed owner task requires explicit store guard and recompute branch");
    }
  } else if (localBuf || stores || loads) {
    return space.emitError("unexpected local buffer in unbuffered owner task");
  }
  return success();
}
LogicalResult verifyOwnerTask(func::FuncOp fn, const TargetProfile &p,
                              StringRef storage) {
  auto &task = fn.front().front();
  Builder b(fn.getContext());
  if (task.getOperands() != fn.getArguments() ||
      !llvm::hasSingleElement(task.getRegion(0)) ||
      task.getRegion(0).front().getOperations().size() != 2 ||
      fn.front().getOperations().size() != 3 ||
      task.getNextNode()->getName().getStringRef() != "cpu.join" ||
      task.getNextNode()->getOperand(0) != task.getResult(0) ||
      task.getRegion(0).front().back().getName().getStringRef() != "cpu.end")
    return task.emitError("invalid ordered task captures or completion");
  if (storage == "pending"
          ? task.getAttr("cpu.storage") != b.getStringAttr("private-values")
          : task.hasAttr("cpu.storage") ||
                task.getAttr("cpu.realization") != b.getStringAttr(storage))
    return task.emitError("incorrect ordered task carry storage");
  return verifyOwnerGrain(fn, task.getRegion(0).front().front(), p);
}
} // namespace intentdsl::cpu
namespace CI_UNIT_NAMESPACE {
using namespace intentdsl;
void ownerTasks(Unit &u) {
  DecisionFamily f;
  f.name = "ordered-owners";
  f.applies = [](Operation *o) {
    return o->getName().getStringRef() == "cpu.owner_space" &&
           !o->hasAttr("cpu.grain");
  };
  f.read = [](Operation *o) { return o->getAttr("cpu.owner_choice"); };
  f.write = [](Operation *o, Attribute a) {
    o->setAttr("cpu.owner_choice", a);
    return llvm::Error::success();
  };
  f.verify = [](Operation *o, Attribute a,
                const TargetProfile &p) -> llvm::Error {
    if (auto d = dyn_cast<DictionaryAttr>(a)) {
      if (auto inner = d.getAs<IntegerAttr>("inner")) {
        auto reuse = cpu::sharedInterval(o, *p.services.get<CallFacts>());
        auto batch = d.getAs<IntegerAttr>("batch"),
             grain = d.getAs<IntegerAttr>("grain");
        auto w = batch ? batch.getInt() : 0;
        auto retained = d.getAs<IntegerAttr>("window");
        if (!reuse || reuse->axis != inner.getInt() ||
            d.size() != (retained ? 4u : 3u) || w < 2 ||
            !llvm::isPowerOf2_64(w) ||
            w > p.services.get<cpu::RowMachine>()->packedOwners || !grain ||
            grain.getInt() % w || reuse->group % w || reuse->extent % w ||
            (retained && (retained.getInt() != cpu::batchScoreSlots / w ||
                          !cpu::repeatedScore(*o))))
          return llvm::createStringError("invalid shared interval traversal");
      } else {
        auto s = cpu::reusableScores(o, p);
        auto extent = d.getAs<IntegerAttr>("scores");
        if (!s || !extent || extent.getInt() != s->keys)
          return llvm::createStringError("invalid retained score range");
      }
      a = d.get("grain");
    }
    auto w = dyn_cast<IntegerAttr>(a);
    if (!w || w.getInt() < 1 || w.getInt() > 256 ||
        !llvm::isPowerOf2_64(w.getInt()))
      return llvm::createStringError("invalid ordered task grain");
    return llvm::Error::success();
  };
  f.verifyCandidate = [](Operation *, const StructuralCandidate &c,
                         const TargetProfile &) -> llvm::Error {
    if (auto d = dyn_cast<DictionaryAttr>(c.value)) {
      auto w = d.getAs<IntegerAttr>("grain");
      bool shared = bool(d.getAs<IntegerAttr>("inner"));
      auto batch = d.getAs<IntegerAttr>("batch");
      if (!w || shared != bool(batch) ||
          c.id != (shared ? "shared-" : "retained-") +
                      std::to_string(w.getInt()) +
                      (shared ? "x" + std::to_string(batch.getInt()) : "") +
                      (d.get("window") ? "-scores" : "") ||
          c.topology !=
              (shared ? "shared-interval-owners" : "retained-scores") ||
          c.preconditions != std::vector<std::string>{"exclusive-output"})
        return llvm::createStringError("invalid retained row candidate");
      return llvm::Error::success();
    }
    if (c.id !=
            "grain-" + std::to_string(cast<IntegerAttr>(c.value).getInt()) ||
        c.topology != "independent-owners" ||
        c.preconditions != std::vector<std::string>{"exclusive-output"})
      return llvm::createStringError("incorrect owner task candidate");
    return llvm::Error::success();
  };
  u.families.push_back(
      {{std::move(f), "cpu@formed", "cpu@formed"}, COMPILERINFRA_SOURCE});
  Strategy s;
  s.name = "owner-grains";
  s.family = u.qualify("ordered-owners");
  s.enumerate = [](Operation *o, const TargetProfile &p, AnalysisManager &)
      -> llvm::Expected<std::vector<StructuralCandidate>> {
    std::vector<StructuralCandidate> cs;
    for (auto w : {1, 8, 32, 128, 256})
      cs.push_back({"grain-" + std::to_string(w),
                    "independent-owners",
                    Builder(o->getContext()).getI64IntegerAttr(w),
                    {"exclusive-output"}});
    if (auto panel = cpu::reusableScores(o, p)) {
      Builder b(o->getContext());
      for (int w : {8, 32, 128}) {
        auto d = b.getDictionaryAttr(
            {b.getNamedAttr("grain", b.getI64IntegerAttr(w)),
             b.getNamedAttr("scores", b.getI64IntegerAttr(panel->keys))});
        cs.push_back({"retained-" + std::to_string(w),
                      "retained-scores",
                      d,
                      {"exclusive-output"}});
      }
    }
    return cs;
  };
  u.strategies.push_back({std::move(s), COMPILERINFRA_SOURCE});
  Realizer r;
  r.name = "group-owners";
  r.family = u.qualify("ordered-owners");
  r.apply = [](Operation *o, Attribute a, const TargetProfile &p,
               RewriterBase &b) -> llvm::Error {
    auto f = p.services.get<CallFacts>();
    auto count = kir::ownerCount(o, *f);
    if (count < 0)
      return llvm::createStringError("unrepresentable ordered task range");
    auto fn = o->getParentOfType<func::FuncOp>();
    for (auto arg : fn.getArguments())
      if (isa<MemRefType>(arg.getType()))
        for (auto key : {"shape", "strides"}) {
          SmallVector<int64_t> values;
          for (auto &v :
               *f->arguments[arg.getArgNumber()].getAsObject()->getArray(key))
            values.push_back(*v.getAsInteger());
          fn.setArgAttr(arg.getArgNumber(), std::string("cpu.") + key,
                        b.getDenseI64ArrayAttr(values));
        }
    o->removeAttr("cpu.owner_choice");
    if (auto d = dyn_cast<DictionaryAttr>(a)) {
      if (auto axis = d.getAs<IntegerAttr>("inner")) {
        o->setAttr("cpu.innermost_owner", axis);
        o->setAttr("cpu.owner_batch", d.get("batch"));
        if (auto win = d.get("window")) {
          o->setAttr("cpu.batch_score_window", win);
          auto rep = *cpu::repeatedScore(*o);
          rep.record->setAttr("cpu.record_score", b.getUnitAttr());
          rep.replay->setAttr("cpu.replay_score", b.getUnitAttr());
        }
        auto shared = *cpu::sharedInterval(o, *f);
        o->walk([&](Operation *op) {
          if (op->getNumResults() == 1 &&
              cpu::invariantOwnerValue(op->getResult(0), *o, shared))
            op->setAttr("cpu.owner_common", b.getUnitAttr());
        });
      } else {
        auto s = cpu::reusableScores(o, p);
        o->setAttr("cpu.score_cache", d.get("scores"));
        s->maxScore.getDefiningOp()->setAttr("cpu.record_score",
                                             b.getUnitAttr());
        s->weightScore.getDefiningOp()->setAttr("cpu.replay_score",
                                                b.getUnitAttr());
      }
      a = d.get("grain");
    }
    o->setAttr("cpu.grain", a);
    o->setAttr("cpu.owners", b.getI64IntegerAttr(count));
    return llvm::Error::success();
  };
  u.realizers.push_back({std::move(r), COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
