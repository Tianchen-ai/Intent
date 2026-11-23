#include "intentdsl/levels/execution/cpu/ir/MatrixTask.h"
#include "intentdsl/levels/execution/cpu/ir/Owners.h"
#include "intentdsl/levels/execution/cpu/ir/ScoreCache.h"
#include "intentdsl/levels/execution/cpu/ir/RepeatedScore.h"
#include "intentdsl/levels/execution/cpu/ir/OwnerTraversal.h"
#include "mlir/IR/PatternMatch.h"
using namespace mlir;
using namespace compilerinfra;
namespace intentdsl::cpu {
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
  auto marks = space.walk([&](Operation *v) {
    if (v->hasAttr("cpu.record_score") !=
            (panel && v == panel->maxScore.getDefiningOp()) ||
        v->hasAttr("cpu.replay_score") !=
            (panel && v == panel->weightScore.getDefiningOp()))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (marks.wasInterrupted())
    return space.emitError(
        "score recording and replay must refer to the same key and query");
  auto batch = space.getAttrOfType<IntegerAttr>("cpu.owner_batch");
  auto repeat = batch ? repeatedScore(space) : std::nullopt;
  auto window = space.getAttrOfType<IntegerAttr>("cpu.batch_score_window");
  if (window &&
      (!repeat || window.getInt() != batchScoreSlots / batch.getInt()))
    return space.emitError("invalid private score window");
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
std::string ownerDispatcher(func::FuncOp fn, StringRef declare,
                            StringRef call) {
  auto &space = fn.front().front().getRegion(0).front().front();
  auto count = space.getAttrOfType<IntegerAttr>("cpu.owners").getInt();
  auto grain = space.getAttrOfType<IntegerAttr>("cpu.grain").getInt();
  std::string src;
  llvm::raw_string_ostream s(src);
  s << "#include <stdint.h>\n#include <stddef.h>\n"
    << declare << "\nint intent_entry(" << matrixHostParams(fn) << "){\n"
    << "const int64_t count=" << count << ", tile=" << grain
    << ";\n"
       "#pragma omp parallel for num_threads(workers) schedule(static) "
       "if(workers>1 && count>tile)\n"
       "for(int64_t lo=0;lo<count;lo+=tile){int64_t "
       "hi=count-lo<tile?count:lo+tile;"
    << call << ";}return 0;}\n";
  return src;
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
        if (auto win = d.get("window"))
          o->setAttr("cpu.batch_score_window", win);
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
