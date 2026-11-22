#include "intentdsl/levels/execution/cpu/ir/MatrixTask.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/StringExtras.h"
using namespace mlir;
using namespace compilerinfra;
namespace intentdsl::cpu {
static ArrayRef<MatrixImplementation> available(Operation &region,
                                                const MatrixMachine &caps) {
  return calculation(region).getName().stripDialect() == "q4k_dot"
             ? caps.records
             : caps.choices;
}
bool groupedI8(Operation &dot, const TargetProfile &p, int64_t width) {
  if (dot.getName().stripDialect() != "contract" ||
      !cast<MemRefType>(dot.getOperand(0).getType())
           .getElementType()
           .isInteger(8) ||
      p.services.get<MatrixMachine>()->integerPacking.empty())
    return false;
  auto &region = *dot.getParentOp();
  auto f = p.services.get<CallFacts>();
  auto g = kir::contractionShape(dot, region.getOperand(0), *f);
  if (!g) {
    llvm::consumeError(g.takeError());
    return false;
  }
  if (!cast<MemRefType>(region.getOperand(0).getType())
           .getElementType()
           .isInteger(32) ||
      llvm::any_of(region.getRegion(0).front(), [](Operation &op) {
        return op.getName().stripDialect() == "convert_matrix";
      }))
    return false;
  auto i = cast<BlockArgument>(dot.getOperand(0)).getArgNumber();
  auto k = dot.getAttrOfType<IntegerAttr>("lhs_k").getInt();
  return g->k > 0 && g->k % 4 == 0 && g->n % width == 0 &&
         (*f->arguments[i].getAsObject()->getArray("strides"))[k]
                 .getAsInteger() == 1;
}
bool matrixTask(func::FuncOp fn) {
  return fn && !fn.empty() &&
         fn.front().front().getName().getStringRef() == "cpu.matrix_task";
}
Operation &calculation(Operation &domain) {
  auto &bb = domain.getRegion(0).front();
  return *llvm::find_if(bb, [](Operation &o) {
    auto n = o.getName().stripDialect();
    return kir::isContraction(n);
  });
}
LogicalResult verifyMatrixTask(func::FuncOp fn, const TargetProfile &p,
                               StringRef storage) {
  auto &task = fn.front().front();
  if (!llvm::hasSingleElement(task.getRegion(0)) ||
      task.getRegion(0).front().getOperations().size() != 2)
    return task.emitError("expected a matrix region and task completion");
  auto &region = task.getRegion(0).front().front();
  if (fn.front().getOperations().size() != 3 ||
      task.getNextNode()->getName().getStringRef() != "cpu.join" ||
      task.getNextNode()->getOperand(0) != task.getResult(0) ||
      !llvm::hasSingleElement(task.getRegion(0)) ||
      task.getRegion(0).front().getOperations().size() != 2 ||
      region.getName().getStringRef() != "cpu.matrix_domain" ||
      task.getOperands() != fn.getArguments() ||
      failed(kir::verifyMatrixBody(fn, region, "cpu", true)))
    return task.emitError("invalid matrix task captures, computation or join");
  if (failed(verifyMatrixPartition(fn, region, p)))
    return failure();
  Builder b(fn.getContext());
  if (storage == "pending")
    return success(task.getAttr("cpu.storage") ==
                   b.getStringAttr("private-values"));
  return success(task.getAttr("cpu.realization") == b.getStringAttr(storage) &&
                 !task.hasAttr("cpu.storage"));
}
LogicalResult verifyMatrixPartition(func::FuncOp fn, Operation &region,
                                    const TargetProfile &p) {
  auto tile = region.getAttrOfType<DenseI64ArrayAttr>("cpu.output_tile");
  auto caps = p.services.get<MatrixMachine>();
  if (!tile || tile.size() != 4 || !caps ||
      !llvm::any_of(available(region, *caps), [&](auto &c) {
        return tile.asArrayRef() ==
               ArrayRef<int64_t>{c.rows, c.cols, c.depth, c.lanes};
      }))
    return region.emitError("invalid matrix implementation");
  auto facts = p.services.get<CallFacts>();
  auto s =
      kir::contractionShape(calculation(region), region.getOperand(0), *facts);
  auto packing = region.getAttrOfType<StringAttr>("cpu.packing");
  auto kind = calculation(region).getName().stripDialect();
  bool routed = kind == "contract" && calculation(region).getNumOperands() == 4;
  auto routing = region.getAttrOfType<StringAttr>("cpu.route_storage");
  if (routed ? (!routing || routing != "gather-once-private-output")
             : bool(routing))
    return region.emitError("routed matrix requires member-ordered input and "
                            "disjoint output scatter");
  if ((kind != "contract" && !packing) ||
      (packing &&
       !(kind == "window_contract" && packing == "mapped-f32-panels") &&
       (packing == caps->integerPacking
            ? !groupedI8(calculation(region), p, tile[1])
            : packing &&
                  packing != (kind == "contract" ? "rhs-panels-per-call"
                              : kind == "window_contract"
                                  ? "mapped-right-per-call"
                              : kind == "q4k_dot" ? "shared-q8k-per-call"
                                                  : "decoded-f32-per-call"))))
    return region.emitError("invalid shared activation packing contract");
  if (!s)
    return region.emitError() << llvm::toString(s.takeError());
  auto sh = region.getAttrOfType<DenseI64ArrayAttr>("cpu.extents");
  if (!sh || sh.asArrayRef() != ArrayRef<int64_t>{s->m, s->n, s->k, s->batches})
    return region.emitError("incorrect task output coverage");
  for (auto [i, v] : llvm::enumerate(fn.getArguments())) {
    auto strides = fn.getArgAttrOfType<DenseI64ArrayAttr>(i, "cpu.strides"),
         shape = fn.getArgAttrOfType<DenseI64ArrayAttr>(i, "cpu.shape");
    auto *s = facts->shapeOf(v);
    auto *t = facts->arguments[i].getAsObject()->getArray("strides");
    if (!s || !t || !strides || !shape || shape.size() != s->size() ||
        strides.size() != t->size())
      return region.emitError("missing CPU matrix layout");
    for (auto [a, b] : llvm::zip(shape.asArrayRef(), *s))
      if (a != b.getAsInteger())
        return region.emitError("incorrect matrix extent");
    for (auto [a, b] : llvm::zip(strides.asArrayRef(), *t))
      if (a != b.getAsInteger())
        return region.emitError("incorrect matrix stride");
  }
  return success();
}
std::string matrixHostParams(func::FuncOp fn) {
  SmallVector<std::string> params;
  for (auto [i, v] : llvm::enumerate(fn.getArguments())) {
    auto a = "a" + std::to_string(i);
    if (!isa<MemRefType>(v.getType())) {
      params.push_back((v.getType().isF32() ? "float " : "int32_t ") + a);
      continue;
    }
    int rank = cast<MemRefType>(v.getType()).getRank();
    params.push_back("void *" + a);
    if (rank > 2)
      for (int j = 0; j < rank - 1; ++j)
        params.push_back("int64_t " + a + "d" + std::to_string(j));
    for (int j = 0; j < rank; ++j)
      params.push_back("int64_t " + a + "s" + std::to_string(j));
  }
  return llvm::join(params, ",") + ",int64_t rows,int64_t columns,int64_t "
                                   "grain,int64_t panel,int64_t workers";
}
} // namespace intentdsl::cpu
namespace CI_UNIT_NAMESPACE {
using namespace intentdsl;
void matrixTasks(Unit &u) {
  DecisionFamily f;
  f.name = "matrix-partition";
  f.applies = [](Operation *op) {
    return op->getName().getStringRef() == "cpu.matrix_domain" &&
           !op->hasAttr("cpu.output_tile");
  };
  f.read = [](Operation *op) { return op->getAttr("cpu.matrix_choice"); };
  f.write = [](Operation *op, Attribute a) {
    op->setAttr("cpu.matrix_choice", a);
    return llvm::Error::success();
  };
  f.verify = [](Operation *op, Attribute a,
                const TargetProfile &p) -> llvm::Error {
    auto c = p.services.get<cpu::MatrixMachine>();
    auto t = dyn_cast<DenseI64ArrayAttr>(a);
    if (!c || !t || t.size() != 5 || t[4] < 0 || t[4] > 2 ||
        (t[4] != 0 &&
         cpu::calculation(*op).getName().stripDialect() != "contract" &&
         !(t[4] == 1 && cpu::calculation(*op).getName().stripDialect() ==
                            "window_contract")) ||
        (t[4] == 2 && !cpu::groupedI8(cpu::calculation(*op), p, t[1])) ||
        !llvm::any_of(cpu::available(*op, *c), [&](auto &r) {
          return t.asArrayRef().take_front(4) ==
                 ArrayRef<int64_t>{r.rows, r.cols, r.depth, r.lanes};
        }))
      return llvm::createStringError("unsupported CPU contraction block");
    return llvm::Error::success();
  };
  f.verifyCandidate = [](Operation *, const StructuralCandidate &c,
                         const TargetProfile &) -> llvm::Error {
    auto t = cast<DenseI64ArrayAttr>(c.value);
    if (c.topology != (t[4] ? "packed-output-tasks" : "output-tasks") ||
        c.id != std::to_string(t[0]) + "x" + std::to_string(t[1]) + "x" +
                    std::to_string(t[2]) +
                    (t[4] == 2 ? "-dot4"
                     : t[4]    ? "-pack"
                               : "") ||
        c.preconditions != std::vector<std::string>{"disjoint-output-blocks"})
      return llvm::createStringError("invalid matrix task candidate");
    return llvm::Error::success();
  };
  u.families.push_back(
      {{std::move(f), "cpu@formed", "cpu@formed"}, COMPILERINFRA_SOURCE});
  Strategy strategy;
  strategy.name = "contract-blocks";
  strategy.family = u.qualify("matrix-partition");
  strategy.enumerate = [](Operation *op, const TargetProfile &p,
                          AnalysisManager &)
      -> llvm::Expected<std::vector<StructuralCandidate>> {
    auto caps = p.services.get<cpu::MatrixMachine>();
    if (!caps)
      return llvm::createStringError("missing native matrix capability");
    Builder b(op->getContext());
    std::vector<StructuralCandidate> cs;
    for (auto &r : cpu::available(*op, *caps))
      for (int pack : {0, 1, 2}) {
        if (pack &&
            cpu::calculation(*op).getName().stripDialect() != "contract" &&
            !(pack == 1 && cpu::calculation(*op).getName().stripDialect() ==
                               "window_contract"))
          continue;
        if (pack == 2 && !cpu::groupedI8(cpu::calculation(*op), p, r.cols))
          continue;
        cs.push_back(
            {std::to_string(r.rows) + "x" + std::to_string(r.cols) + "x" +
                 std::to_string(r.depth) +
                 (pack == 2 ? "-dot4"
                  : pack    ? "-pack"
                            : ""),
             pack ? "packed-output-tasks" : "output-tasks",
             b.getDenseI64ArrayAttr({r.rows, r.cols, r.depth, r.lanes, pack}),
             {"disjoint-output-blocks"}});
      }
    return cs;
  };
  u.strategies.push_back({std::move(strategy), COMPILERINFRA_SOURCE});
  Realizer r;
  r.name = "own-output-blocks";
  r.family = u.qualify("matrix-partition");
  r.apply = [](Operation *op, Attribute t, const TargetProfile &p,
               RewriterBase &b) -> llvm::Error {
    auto facts = p.services.get<CallFacts>();
    auto g =
        kir::contractionShape(cpu::calculation(*op), op->getOperand(0), *facts);
    if (!g)
      return g.takeError();
    if (g->routed)
      op->setAttr("cpu.route_storage",
                  b.getStringAttr("gather-once-private-output"));
    auto row = cast<DenseI64ArrayAttr>(t);
    op->setAttr("cpu.output_tile",
                b.getDenseI64ArrayAttr(row.asArrayRef().take_front(4)));
    if (row[4])
      op->setAttr("cpu.packing",
                  b.getStringAttr(
                      row[4] == 2
                          ? p.services.get<cpu::MatrixMachine>()->integerPacking
                          : "rhs-panels-per-call"));
    op->setAttr("cpu.extents",
                b.getDenseI64ArrayAttr({g->m, g->n, g->k, g->batches}));
    if (cpu::calculation(*op).getName().stripDialect() == "q4k_dot")
      op->setAttr("cpu.packing", b.getStringAttr("shared-q8k-per-call"));
    if (cpu::calculation(*op).getName().stripDialect() == "scaled_contract")
      op->setAttr("cpu.packing", b.getStringAttr("decoded-f32-per-call"));
    if (cpu::calculation(*op).getName().stripDialect() == "window_contract")
      op->setAttr("cpu.packing",
                  b.getStringAttr(row[4] ? "mapped-f32-panels"
                                         : "mapped-right-per-call"));
    auto fn = op->getParentOfType<func::FuncOp>();
    for (auto [i, v] : llvm::enumerate(fn.getArguments())) {
      for (auto key : {"shape", "strides"}) {
        SmallVector<int64_t> axes;
        for (auto &dim : *facts->arguments[i].getAsObject()->getArray(key))
          axes.push_back(*dim.getAsInteger());
        fn.setArgAttr(i, std::string("cpu.") + key,
                      b.getDenseI64ArrayAttr(axes));
      }
    }
    op->removeAttr("cpu.matrix_choice");
    return llvm::Error::success();
  };
  u.realizers.push_back({std::move(r), COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
