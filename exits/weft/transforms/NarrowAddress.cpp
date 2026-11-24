#include "compilerinfra/Assembly.h"
#include "compilerinfra/Proofs.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace CI_UNIT_NAMESPACE {
using namespace mlir;

bool canNarrowWeftAddress(Operation &op) {
  if (op.getName().getStringRef() != "weft_host.address")
    return false;
  compilerinfra::AddressProofOptions opts;
  opts.targetBitwidth = 32;
  opts.fallbackBitwidth = 64;
  opts.isSigned = true;
  return compilerinfra::proveNarrowAddress(&op, opts).proven;
}

namespace {
struct NarrowWeft : PassWrapper<NarrowWeft, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NarrowWeft)
  void runOnOperation() final {
    compilerinfra::AddressProofOptions opts;
    opts.targetBitwidth = 32;
    opts.fallbackBitwidth = 64;
    opts.isSigned = true;
    Builder b(&getContext());
    getOperation().walk([&](Operation *hostFn) {
      StringRef n = hostFn->getName().getStringRef();
      if (n != "weft_host.host_fn" && n != "weft_host.host_row_fn" &&
          n != "weft_host.host_owner_fn")
        return;
      compilerinfra::IntegerRangeContext ranges(hostFn);
      if (!ranges.valid())
        return;
      hostFn->walk([&](Operation *op) {
        if (op->getName().getStringRef() == "weft_host.address" &&
            compilerinfra::proveNarrowAddress(op, opts, &ranges).proven)
          op->setAttr("index_bits", b.getI32IntegerAttr(32));
      });
    });
  }
};
} // namespace

void registerWeftAddressOptimization(compilerinfra::Unit &u) {
  u.contributions.push_back(
      {{"narrow-proven-addresses",
        "weft@realized",
        [](OpPassManager &pm, const compilerinfra::TargetProfile &) {
          pm.addPass(std::make_unique<NarrowWeft>());
          return llvm::Error::success();
        }},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
