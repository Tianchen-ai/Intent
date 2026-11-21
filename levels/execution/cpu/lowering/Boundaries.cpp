#include "intentdsl/levels/execution/cpu/ir/Dialect.h"
#include "intentdsl/levels/execution/cpu/ir/Task.h"
#include "intentdsl/levels/algorithm/kir/ir/RowSemantics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/IRMapping.h"
using namespace compilerinfra;
using namespace mlir;
namespace intentdsl::cpu {
LogicalResult verifyTask(ModuleOp m, const TargetProfile &profile,
                         StringRef storage, StringRef taskOp,
                         bool logicalBody) {
  auto funcs = m.getOps<func::FuncOp>();
  if (!llvm::hasSingleElement(funcs))
    return m.emitError("expected one host");
  auto fn = *funcs.begin();
  if (fn.empty() || fn.front().getOperations().size() != 3)
    return fn.emitError("expected task, join, return");
  auto &task = fn.front().front();
  auto *join = task.getNextNode();
  if (task.getName().getStringRef() != taskOp ||
      join->getName().getStringRef() != "cpu.join" ||
      join->getOperand(0) != task.getResult(0))
    return fn.emitError("join must complete the owned task range");
  auto rows = task.getAttrOfType<DenseI64ArrayAttr>("rows");
  auto *machine = profile.services.get<RowMachine>();
  if (!machine || !rows || rows.empty() ||
      !llvm::all_of(rows.asArrayRef(), [&](auto r) {
        return r > 0 && llvm::is_contained(machine->rows, r);
      }))
    return task.emitError("invalid row partition");
  if (!llvm::hasSingleElement(task.getRegion(0)))
    return task.emitError("expected one task body");
  auto &body = task.getRegion(0).front();
  auto domain = task.getAttrOfType<IntegerAttr>("domain_arg").getInt();
  if (domain < 0 || domain >= int64_t(fn.getNumArguments()) ||
      task.getOperands() != fn.getArguments() ||
      body.getNumArguments() != 1 + fn.getNumArguments() ||
      !body.getArgument(0).getType().isIndex() ||
      TypeRange(body.getArguments().drop_front()) != fn.getArgumentTypes())
    return task.emitError("invalid task capture ABI");
  auto attr = task.getAttrOfType<StringAttr>(
      storage == "pending" ? "cpu.storage" : "cpu.realization");
  if (!attr || attr != (storage == "pending" ? "private-values" : storage) ||
      (storage != "pending" && task.hasAttr("cpu.storage")))
    return task.emitError("invalid storage handoff");
  if (body.empty() || body.back().getName().getStringRef() != "cpu.end")
    return task.emitError("missing task completion");
  if (logicalBody)
    return intentdsl::verifyRowComputation(
        fn, body, body.getArguments().drop_front(), domain, "cpu", true);
  return success();
}
llvm::Error realizeLocalValues(Operation *op, Attribute value,
                               const TargetProfile &, RewriterBase &rw) {
  if (value != rw.getStringAttr("private-values"))
    return llvm::createStringError("unsupported storage contract");
  op->setAttr("cpu.realization", value);
  op->removeAttr("cpu.storage");
  return llvm::Error::success();
}
} // namespace intentdsl::cpu
namespace CI_UNIT_NAMESPACE {
void rowTasks(Unit &);
struct Construct : PassWrapper<Construct, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Construct)
  bool tasks;
  explicit Construct(bool tasks) : tasks(tasks) {}
  void runOnOperation() final {
    getContext().getOrLoadDialect<intentdsl::cpu::CpuDialect>();
    if (!tasks) {
      getOperation().walk<WalkOrder::PostOrder>([](Operation *op) {
        if (op->getName().getDialectNamespace() != "kir")
          return;
        OpBuilder b(op);
        OperationState s(op->getLoc(),
                         ("cpu." + op->getName().stripDialect()).str());
        s.addOperands(op->getOperands());
        s.addTypes(op->getResultTypes());
        for (auto &r : op->getRegions())
          s.addRegion()->takeBody(r);
        op->replaceAllUsesWith(b.create(s));
        op->erase();
      });
      return;
    }
    auto fn = *getOperation().getOps<func::FuncOp>().begin();
    auto *loop = &fn.front().front();
    OpBuilder b(loop);
    auto loc = loop->getLoc();
    OperationState s(loc, "cpu.task");
    s.addOperands(fn.getArguments());
    s.addTypes(b.getIndexType());
    s.addAttribute("rows", loop->getAttr("cpu.row_groups"));
    s.addAttribute(
        "domain_arg",
        b.getI64IntegerAttr(
            cast<BlockArgument>(loop->getOperand(0)).getArgNumber()));
    s.addRegion();
    auto *task = b.create(s);
    auto &body = task->getRegion(0).emplaceBlock();
    IRMapping map;
    map.map(loop->getRegion(0).front().getArgument(0),
            body.addArgument(b.getIndexType(), loc));
    for (auto a : fn.getArguments())
      map.map(a, body.addArgument(a.getType(), loc));
    b.setInsertionPointToEnd(&body);
    for (auto &op : loop->getRegion(0).front())
      b.clone(op, map);
    b.setInsertionPoint(loop);
    OperationState j(loc, "cpu.join");
    j.addOperands(task->getResult(0));
    b.create(j);
    loop->erase();
  }
};
void bind(Unit &u) {
  u.dialects = [](DialectRegistry &r) {
    r.insert<intentdsl::cpu::CpuDialect>();
  };
  rowTasks(u);
  DecisionFamily f;
  f.name = "storage";
  f.applies = [](Operation *op) {
    return op->getName().getStringRef() == "cpu.task";
  };
  f.read = [](Operation *op) { return op->getAttr("cpu.storage"); };
  f.write = [](Operation *op, Attribute a) {
    op->setAttr("cpu.storage", a);
    return llvm::Error::success();
  };
  f.verify = [](Operation *, Attribute a,
                const TargetProfile &) -> llvm::Error {
    if (!isa<StringAttr>(a) || cast<StringAttr>(a) != "private-values")
      return llvm::createStringError("invalid storage requirement");
    return llvm::Error::success();
  };
  u.families.push_back(
      {{f, "cpu@tasked", "cpu@buffers"}, COMPILERINFRA_SOURCE});
  u.strategies.push_back(
      {{"private-values", u.qualify("storage"),
        [](Operation *op, const TargetProfile &,
           AnalysisManager &) -> llvm::Expected<Attribute> {
          return Attribute(StringAttr::get(op->getContext(), "private-values"));
        }},
       COMPILERINFRA_SOURCE});
  u.realizers.push_back(
      {{"allocate", u.qualify("storage"), intentdsl::cpu::realizeLocalValues},
       COMPILERINFRA_SOURCE});
  for (StringRef phase : {"formed", "tasked", "buffers"})
    u.boundaries.push_back(
        {{phase.str(),
          [phase](OpPassManager &pm, const TargetProfile &) {
            if (phase != "buffers")
              pm.addPass(std::make_unique<Construct>(phase == "tasked"));
            return llvm::Error::success();
          },
          [phase](ModuleOp m, const TargetProfile &p) {
            if (phase != "formed")
              return intentdsl::cpu::verifyTask(
                  m, p, phase == "tasked" ? "pending" : "private-values");
            if (failed(intentdsl::verifyLogicalRows(m, "cpu", true)))
              return failure();
            auto &op = (*m.getOps<func::FuncOp>().begin()).front().front();
            auto rowGroups =
                op.getAttrOfType<DenseI64ArrayAttr>("cpu.row_groups");
            auto *caps = p.services.get<intentdsl::cpu::RowMachine>();
            if (!rowGroups || rowGroups.empty() || !caps ||
                !llvm::all_of(rowGroups.asArrayRef(), [&](int64_t r) {
                  return r > 0 && llvm::is_contained(caps->rows, r);
                }))
              return op.emitError(
                  "formed row partition is outside target capabilities");
            return success();
          }},
         COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
