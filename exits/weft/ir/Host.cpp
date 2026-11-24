#include "../legalize/Matrix.h"
#include "../legalize/Ordered.h"
#include "../legalize/Rows.h"
#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
#include "intentdsl/levels/execution/cpu/ir/MatrixTask.h"
#include "intentdsl/levels/execution/cpu/ir/Owners.h"
#include "intentdsl/levels/execution/cpu/ir/Task.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
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
LogicalResult lowerWeftMatrix(func::FuncOp, bool);
bool hasWeftMatrixProgram(ModuleOp);
LogicalResult verifyWeftMatrix(ModuleOp, const compilerinfra::TargetProfile &,
                               bool);
void publishWeftMatrixABI(OpPassManager &);

void dialect(DialectRegistry &r) {
  r.insert<intentdsl::wefthost::WeftHostDialect, arith::ArithDialect,
           math::MathDialect>();
}

namespace {
struct Lower : PassWrapper<Lower, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Lower)
  std::string phase;
  intentdsl::CallFacts facts;
  bool grouped;
  int64_t tailVecWidth;
  int64_t privateStackBytes;
  Lower(StringRef p, const intentdsl::CallFacts &f, bool g, int64_t tv,
        int64_t ps)
      : phase(p.str()), facts(f), grouped(g), tailVecWidth(tv),
        privateStackBytes(ps) {}
  void runOnOperation() final {
    ModuleOp m = getOperation();
    if (hasWeftMatrixProgram(m) || hasWeftRowProgram(m) ||
        hasWeftOwnerProgram(m))
      return;
    auto f = *m.getOps<func::FuncOp>().begin();
    if (phase == "legal") {
      if (intentdsl::cpu::ownerTask(f) &&
          failed(bindWeftMemberStores(f.front().front(), grouped, true)))
        signalPassFailure();
      return;
    }
    if (phase != "realized")
      return;

    getContext().getOrLoadDialect<intentdsl::wefthost::WeftHostDialect>();
    getContext().getOrLoadDialect<arith::ArithDialect>();
    getContext().getOrLoadDialect<math::MathDialect>();
    if (intentdsl::cpu::matrixTask(f)) {
      if (failed(lowerWeftMatrix(f, grouped)))
        signalPassFailure();
    } else if (intentdsl::cpu::ownerTask(f)) {
      if (failed(lowerWeftOrdered(f, facts, grouped)))
        signalPassFailure();
    } else {
      if (failed(
              lowerWeftRow(f, facts, grouped, tailVecWidth, privateStackBytes)))
        signalPassFailure();
    }
  }
};

struct PublishWeftRowAndOrderedABI
    : PassWrapper<PublishWeftRowAndOrderedABI, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PublishWeftRowAndOrderedABI)
  void runOnOperation() final {
    ModuleOp m = getOperation();
    bool isRow = hasWeftRowProgram(m);
    bool isOwner = hasWeftOwnerProgram(m);
    if (!isRow && !isOwner)
      return;
    auto fn = *m.getOps<func::FuncOp>().begin();
    auto &prog = fn.front().front();
    OpBuilder b(m.getContext());
    fn->setAttr("weft_host.entry", b.getStringAttr("intent_entry"));
    llvm::SmallVector<char> buf;
    llvm::raw_svector_ostream os(buf);
    llvm::Error err = isRow ? describeWeftRowPackage(fn, prog, os)
                            : describeWeftOrderedPackage(fn, prog, os);
    if (err) {
      m.emitError() << llvm::toString(std::move(err));
      signalPassFailure();
      return;
    }
    m->setAttr("weft_host.package",
               b.getStringAttr(StringRef(buf.data(), buf.size())));
  }
};
} // namespace

void lower(OpPassManager &pm, StringRef phase,
           const intentdsl::CallFacts &facts, bool grouped,
           int64_t tailVecWidth, int64_t privateStackBytes) {
  pm.addPass(std::make_unique<Lower>(phase, facts, grouped, tailVecWidth,
                                     privateStackBytes));
}

void publishWeftABI(OpPassManager &pm) {
  publishWeftMatrixABI(pm);
  pm.addPass(std::make_unique<PublishWeftRowAndOrderedABI>());
}

LogicalResult verifyHost(ModuleOp m, const compilerinfra::TargetProfile &p,
                         StringRef phase) {
  bool callable = (phase == "callable");
  if (hasWeftMatrixProgram(m)) {
    if (phase == "legal")
      return m.emitError("unexpected weft_host.matrix_program at weft@legal");
    return verifyWeftMatrix(m, p, callable);
  }
  if (hasWeftRowProgram(m)) {
    if (phase == "legal")
      return m.emitError("unexpected weft_host.row_program at weft@legal");
    return verifyWeftRow(m, p, callable);
  }
  if (hasWeftOwnerProgram(m)) {
    if (phase == "legal")
      return m.emitError("unexpected weft_host.owner_program at weft@legal");
    return verifyWeftOrdered(m, p, callable);
  }
  if (phase != "legal")
    return m.emitError("unlowered CPU task past weft@legal");

  if (failed(intentdsl::cpu::verifyTask(m, p, "weft-task-ssa")))
    return failure();
  auto fn = *m.getOps<func::FuncOp>().begin();
  if (intentdsl::cpu::ownerTask(fn)) {
    if (failed(bindWeftMemberStores(fn.front().front(),
                                    p.services.get<RecordFields>()->grouped,
                                    false)))
      return failure();
  }
  return success();
}
} // namespace CI_UNIT_NAMESPACE
