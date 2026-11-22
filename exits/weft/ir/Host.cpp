#include "intentdsl/levels/execution/cpu/ir/Task.h"
#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
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
  intentdsl::CallFacts facts;
  Lower(bool callable,const intentdsl::CallFacts &f) : callable(callable),facts(f) {}
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
      bool absolute = false;
      task.walk([&](Operation *o) {
        absolute |= o->getName().getStringRef() == "cpu.ordinal";
        if(o->getName().getStringRef()=="cpu.gather_row"){
          auto table=cast<BlockArgument>(o->getOperand(0)).getArgNumber()-1;
          auto ids=cast<BlockArgument>(o->getOperand(1)).getArgNumber()-1;
          auto *slot=facts.arguments[table].getAsObject();auto *shape=slot?slot->getArray("shape"):nullptr;
          if(!shape || shape->size()!=2 || !shape->front().getAsInteger()){o->emitError("missing indexed table height");signalPassFailure();return;}
          f.setArgAttr(table,"weft_host.table_rows",b.getI64IntegerAttr(*shape->front().getAsInteger()));
          f.setArgAttr(ids,"weft_host.task_indices",b.getUnitAttr());
        }
      });
      task.setAttr("weft_host.absolute_origin", b.getBoolAttr(absolute));
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
    if (cast<DictionaryAttr>(dispatch->getAttrOfType<ArrayAttr>("rows")[0])
            .getAs<IntegerAttr>("columns")
            .getInt() != 0)
      buildPanels(dispatch);
    task.erase();
  }
};
} // namespace
void lower(OpPassManager &pm, bool callable,const intentdsl::CallFacts &facts) {
  pm.addPass(std::make_unique<Lower>(callable,facts));
}
LogicalResult verifyHost(ModuleOp m, const compilerinfra::TargetProfile &p,
                         bool callable) {
  if (failed(intentdsl::cpu::verifyTask(m, p, "weft-task-ssa",
                                        "weft_host.dispatch_rows", false)))
    return failure();
  auto &task = (*m.getOps<func::FuncOp>().begin()).front().front();
  bool whole = cast<DictionaryAttr>(task.getAttrOfType<ArrayAttr>("rows")[0])
                   .getAs<IntegerAttr>("columns")
                   .getInt() == 0;
  if (whole && failed(intentdsl::cpu::verifyTask(
                   m, p, "weft-task-ssa", "weft_host.dispatch_rows", true)))
    return failure();
  if (!whole && failed(verifyPanels(&task)))
    return failure();
  if (task.getAttrOfType<StringAttr>("task_symbol") != "row_task")
    return task.emitError("unbound native row task");
  if (callable) {
    auto slots = task.getAttrOfType<DenseI64ArrayAttr>("weft_host.slot_ranks");
    bool absolute = false;
    task.walk([&](Operation *o) {
      absolute |= o->getName().getStringRef() == "cpu.ordinal";
    });
    if (task.getAttrOfType<BoolAttr>("weft_host.absolute_origin") !=
        BoolAttr::get(m.getContext(), absolute))
      return task.emitError(
          "task absolute index requires its logical interval origin");
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
