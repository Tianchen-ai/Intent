#include "intentdsl/levels/execution/cpu/ir/Task.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringSwitch.h"
using namespace compilerinfra;
using namespace mlir;
namespace CI_UNIT_NAMESPACE {
namespace {
struct Publish : PassWrapper<Publish, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Publish)
  void runOnOperation() final {
    auto fn = *getOperation().getOps<func::FuncOp>().begin();
    fn->setAttr("mojo_row.entry",
                StringAttr::get(&getContext(), "intent_entry"));
  }
};
} // namespace
void publishABI(OpPassManager &pm) { pm.addPass(std::make_unique<Publish>()); }
LogicalResult verifyLoops(ModuleOp m, const TargetProfile &p, bool callable) {
  if (failed(intentdsl::cpu::verifyTask(m, p, "private-values", "cpu.task",
                                        false)))
    return failure();
  auto fn = *m.getOps<func::FuncOp>().begin();
  auto &task = fn.front().front();
  if (task.getAttrOfType<StringAttr>("mojo_row.schedule") !=
      "static-row-groups")
    return task.emitError("unbound synchronous Mojo row schedule");
  auto lanes = task.getAttrOfType<IntegerAttr>("mojo_row.lanes");
  if (!lanes ||
      lanes.getInt() !=
          p.services.get<intentdsl::cpu::RowMachine>()->vectorBits / 32)
    return task.emitError(
        "sweep width differs from the effective vector target");
  if (callable &&
      fn->getAttrOfType<StringAttr>("mojo_row.entry") != "intent_entry")
    return fn.emitError("missing synchronous host entry");
  Operation *full = nullptr;
  for (auto &s : task.getRegion(0).front().without_terminator()) {
    if (s.getName().getStringRef() != "mojo_row.sweep")
      return s.emitError("expected a realized column sweep");
    auto alg = s.getAttrOfType<StringAttr>("algebra");
    bool store = alg == "store";
    auto w = s.getAttrOfType<IntegerAttr>("width").getInt();
    bool tail = full != nullptr;
    if ((!store && alg != "max_num" && alg != "sum") ||
        s.getAttrOfType<StringAttr>("range") != (tail ? "tail" : "full") ||
        w != (tail ? 1 : lanes.getInt()) ||
        s.getNumResults() != unsigned(!store) ||
        s.getNumOperands() != unsigned(tail && !store) ||
        (!store && !s.getResult(0).getType().isF32()) ||
        (tail && (alg != full->getAttr("algebra") ||
                  (!store && s.getOperand(0) != full->getResult(0)))))
      return s.emitError("invalid full sweep or scalar tail continuation");
    auto &bb = s.getRegion(0).front();
    if (bb.getNumArguments() != 1 || !bb.getArgument(0).getType().isIndex() ||
        bb.empty() || bb.back().getName().getStringRef() != "mojo_row.yield" ||
        bb.back().getNumOperands() != unsigned(!store))
      return s.emitError("invalid column induction or sweep yield");
    if (!store && bb.back().getOperand(0).getType() !=
                      VectorType::get({w}, Float32Type::get(m.getContext())))
      return s.emitError("expected f32 reduction lanes");
    unsigned writes = 0;
    for (auto &op : bb.without_terminator()) {
      auto code = op.getName().getStringRef();
      if (code == "mojo_row.read" || code == "mojo_row.write") {
        bool write = code == "mojo_row.write";
        writes += write;
        auto view = cast<MemRefType>(op.getOperand(0).getType());
        if ((view.getRank()<1 || view.getRank()>4) ||
            op.getOperand(1) != task.getRegion(0).front().getArgument(0) ||
            op.getOperand(2) != bb.getArgument(0) ||
            (write ? op.getOperand(3) : op.getResult(0)).getType() !=
                VectorType::get({w}, view.getElementType()))
          return op.emitError("invalid strided lane access");
      } else if (code == "mojo_row.apply") {
        auto k = op.getAttrOfType<StringAttr>("kind").getValue();
        auto nargs = llvm::StringSwitch<unsigned>(k)
                         .Cases("add", "sub", "mul", "div", 2)
                         .Cases("exp", "rsqrt", "cast", 1)
                         .Cases("and", "xor", "shl", "shr", "gt", 2)
                         .Cases("uniform", "ordinal", 2)
                         .Case("select", 3)
                         .Case("sqrt", 1)
                         .Case("maximum", 2)
                         .Case("constant", 0)
                         .Default(-1);
        if (k == "uniform" && op.getNumOperands() == 2) {
          auto position = dyn_cast<VectorType>(op.getOperand(1).getType());
          auto samples = dyn_cast<VectorType>(op.getResult(0).getType());
          if (!op.getOperand(0).getType().isUnsignedInteger(64) || !position ||
              !samples || !position.getElementType().isUnsignedInteger(64) ||
              !samples.getElementType().isF32() ||
              position.getShape() != samples.getShape())
            return op.emitError(
                "invalid Philox seed, member counters or sample lanes");
        }
        if (op.getNumOperands() != nargs ||
            (k == "constant" && !op.getAttrOfType<TypedAttr>("value")))
          return op.emitError("unknown or ill-formed lane expression");
      } else
        return op.emitError("unexpected sweep instruction");
    }
    if (store ? !writes : bool(writes))
      return s.emitError("sweep write count differs from its algebra");
    full = tail ? nullptr : &s;
  }
  if (full)
    return full->emitError("missing matching tail sweep");
  return success();
}
} // namespace CI_UNIT_NAMESPACE
