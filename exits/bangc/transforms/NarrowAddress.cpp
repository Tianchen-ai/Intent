#include "../legalize/Rows.h"
#include "compilerinfra/Proofs.h"
#include "mlir/Pass/Pass.h"

namespace CI_UNIT_NAMESPACE {
using namespace mlir;

bool canNarrowAddress(Operation &op, func::FuncOp,
                      const compilerinfra::IntegerRangeContext *existing) {
  if (op.getName().getStringRef() != "bangc_src.address")
    return false;
  compilerinfra::AddressProofOptions opts;
  opts.targetBitwidth = 32;
  opts.fallbackBitwidth = 64;
  opts.isSigned = true;
  return compilerinfra::proveNarrowAddress(&op, opts, existing).proven;
}

namespace {
struct NarrowProvenAddresses
    : PassWrapper<NarrowProvenAddresses, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NarrowProvenAddresses)
  void runOnOperation() final {
    auto m = getOperation();
    auto funcs = m.getOps<func::FuncOp>();
    if (!llvm::hasSingleElement(funcs))
      return;
    auto fn = *funcs.begin();
    compilerinfra::IntegerRangeContext rangeCtx(fn.getOperation());
    Builder b(&getContext());
    m.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "bangc_src.address" &&
          canNarrowAddress(*op, fn, &rangeCtx))
        op->setAttr("index_bits", b.getI32IntegerAttr(32));
    });
  }
};
} // namespace

void addressOptimization(compilerinfra::Unit &u) {
  u.contributions.push_back(
      {{"narrow-proven-addresses",
        "bangc@callable",
        [](OpPassManager &pm, const compilerinfra::TargetProfile &) {
          pm.addPass(std::make_unique<NarrowProvenAddresses>());
          return llvm::Error::success();
        },
        {},
        {},
        false},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
