#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/ProjectedLayout.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

using namespace compilerinfra;
using namespace mlir;

namespace CI_UNIT_NAMESPACE {
bool hasMojoMatrixProgram(ModuleOp);
bool hasMojoOrderedProgram(ModuleOp);
bool hasMojoRowProgram(ModuleOp);
llvm::Error describeMojoMatrixPackage(func::FuncOp, Operation &,
                                      llvm::raw_pwrite_stream &);
llvm::Error describeMojoOrderedPackage(func::FuncOp, Operation &,
                                       llvm::raw_pwrite_stream &);
llvm::Error describeMojoRowPackage(func::FuncOp, Operation &,
                                   llvm::raw_pwrite_stream &);
LogicalResult verifyMojoMatrix(ModuleOp, const TargetProfile &, bool callable);
LogicalResult verifyMojoOrdered(ModuleOp, const TargetProfile &, bool callable);
LogicalResult verifyMojoRow(ModuleOp, const TargetProfile &, bool callable);

namespace {
struct Publish : PassWrapper<Publish, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Publish)
  void runOnOperation() final {
    auto module = getOperation();
    auto fn = *module.getOps<func::FuncOp>().begin();
    fn->setAttr("mojo_row.entry",
                StringAttr::get(&getContext(), "intent_entry"));
    llvm::SmallVector<char> metadata;
    llvm::raw_svector_ostream os(metadata);
    llvm::Error err = llvm::Error::success();
    if (hasMojoMatrixProgram(module))
      err = describeMojoMatrixPackage(fn, fn.front().front(), os);
    else if (hasMojoOrderedProgram(module))
      err = describeMojoOrderedPackage(fn, fn.front().front(), os);
    else if (hasMojoRowProgram(module))
      err = describeMojoRowPackage(fn, fn.front().front(), os);
    else
      err = llvm::createStringError("missing realized Mojo terminal program");
    if (err) {
      fn.emitError() << llvm::toString(std::move(err));
      signalPassFailure();
      return;
    }
    module->setAttr("mojo_row.package",
                    StringAttr::get(&getContext(),
                                    StringRef(metadata.data(),
                                              metadata.size())));
  }
};
} // namespace

void publishABI(OpPassManager &pm) { pm.addPass(std::make_unique<Publish>()); }

LogicalResult verifyLoops(ModuleOp m, const TargetProfile &p, bool callable) {
  if (hasMojoMatrixProgram(m))
    return verifyMojoMatrix(m, p, callable);
  if (hasMojoOrderedProgram(m))
    return verifyMojoOrdered(m, p, callable);
  if (hasMojoRowProgram(m)) {
    auto fn = *m.getOps<func::FuncOp>().begin();
    if (failed(intentdsl::specializeProjectedLayout(
            fn, *p.services.get<intentdsl::CallFacts>(), "mojo_row", false)))
      return failure();
    return verifyMojoRow(m, p, callable);
  }
  return m.emitError("expected terminal Mojo matrix, ordered, or row program");
}
} // namespace CI_UNIT_NAMESPACE
