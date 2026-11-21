#include "intentdsl/levels/execution/cpu/ir/Task.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Pass/Pass.h"
#include "Dialect.h.inc"
#include "Dialect.cpp.inc"
#define GET_OP_CLASSES
#include "Ops.h.inc"
#define GET_OP_CLASSES
#include "Ops.cpp.inc"

using namespace mlir;
void intentdsl::wefthost::WeftHostDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Ops.cpp.inc"
      >();
}
namespace CI_UNIT_NAMESPACE {
void buildPanels(Operation *);
LogicalResult verifyPanels(Operation *);
void dialect(DialectRegistry &r) {
  r.insert<intentdsl::wefthost::WeftHostDialect>();
}
namespace {
struct Lower : PassWrapper<Lower, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Lower)
  bool callable;
  explicit Lower(bool callable) : callable(callable) {}
  void runOnOperation() final {
    auto f = *getOperation().getOps<func::FuncOp>().begin();
    auto &task = f.front().front();
    OpBuilder b(&task);
    if (callable) {
      SmallVector<int64_t> ranks;
      for (Type t : task.getOperandTypes()) {
        auto view = dyn_cast<MemRefType>(t);
        ranks.push_back(view ? view.getRank() : 0);
      }
      task.setAttr("weft_host.slot_ranks", b.getDenseI64ArrayAttr(ranks));
      return;
    }
    getContext().getOrLoadDialect<intentdsl::wefthost::WeftHostDialect>();
    OperationState s(task.getLoc(), "weft_host.dispatch_rows");
    s.addOperands(task.getOperands());
    s.addTypes(task.getResultTypes());
    s.addAttributes(task.getAttrs());
    s.addAttribute("task_symbol", b.getStringAttr("row_task"));
    s.addRegion()->takeBody(task.getRegion(0));
    auto *dispatch = b.create(s);
    task.replaceAllUsesWith(dispatch);
    buildPanels(dispatch);
    task.erase();
  }
};
} // namespace
void lower(OpPassManager &pm, bool callable) {
  pm.addPass(std::make_unique<Lower>(callable));
}
LogicalResult verifyHost(ModuleOp m, const compilerinfra::TargetProfile &p,
                         bool callable) {
  if (failed(intentdsl::cpu::verifyTask(m, p, "weft-task-ssa",
                                        "weft_host.dispatch_rows", false)))
    return failure();
  auto &task = (*m.getOps<func::FuncOp>().begin()).front().front();
  if (failed(verifyPanels(&task)))
    return failure();
  if (task.getAttrOfType<StringAttr>("task_symbol") != "row_task")
    return task.emitError("unbound native row task");
  if (callable) {
    auto slots = task.getAttrOfType<DenseI64ArrayAttr>("weft_host.slot_ranks");
    if (!slots || slots.size() != int64_t(task.getNumOperands()))
      return task.emitError("incomplete bridge slots");
    for (auto [rank, t] :
         llvm::zip(slots.asArrayRef(), task.getOperandTypes())) {
      auto view = dyn_cast<MemRefType>(t);
      if (rank != (view ? view.getRank() : 0))
        return task.emitError("task bridge rank differs from capture");
    }
  }
  return success();
}
} // namespace CI_UNIT_NAMESPACE
