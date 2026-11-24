#include "compilerinfra/Assembly.h"
#include "compilerinfra/Proofs.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include <memory>

namespace CI_UNIT_NAMESPACE {
using namespace mlir;

namespace {
Operation *findIsolatedScope(Operation *op) {
  Operation *cur = op;
  while (cur && cur->getParentOp() &&
         !cur->hasTrait<OpTrait::IsIsolatedFromAbove>())
    cur = cur->getParentOp();
  return cur;
}
} // namespace

bool canNarrowMojoAddress(Operation &op) {
  StringRef n = op.getName().getStringRef();
  if (n != "mojo_row.address" && n != "mojo_row.dyn_address")
    return false;
  compilerinfra::AddressProofOptions opts;
  opts.targetBitwidth = 32;
  opts.fallbackBitwidth = 64;
  opts.isSigned = true;
  return compilerinfra::proveNarrowAddress(&op, opts).proven;
}

namespace {
struct NarrowMojo : PassWrapper<NarrowMojo, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NarrowMojo)
  void runOnOperation() final {
    ModuleOp m = getOperation();
    llvm::DenseMap<Operation *, std::unique_ptr<compilerinfra::IntegerRangeContext>>
        scopeRanges;
    compilerinfra::AddressProofOptions opts;
    opts.targetBitwidth = 32;
    opts.fallbackBitwidth = 64;
    opts.isSigned = true;
    Builder b(&getContext());
    m.walk([&](Operation *op) {
      StringRef n = op->getName().getStringRef();
      if (n != "mojo_row.address" && n != "mojo_row.dyn_address")
        return;
      Operation *scope = findIsolatedScope(op);
      if (!scope)
        return;
      auto &ctxPtr = scopeRanges[scope];
      if (!ctxPtr)
        ctxPtr = std::make_unique<compilerinfra::IntegerRangeContext>(scope);
      if (ctxPtr->valid() &&
          compilerinfra::proveNarrowAddress(op, opts, ctxPtr.get()).proven)
        op->setAttr("index_bits", b.getI32IntegerAttr(32));
    });
  }
};
} // namespace

void registerMojoAddressOptimization(compilerinfra::Unit &u) {
  u.contributions.push_back(
      {{"narrow-proven-addresses",
        "mojo@realized",
        [](OpPassManager &pm, const compilerinfra::TargetProfile &) {
          pm.addPass(std::make_unique<NarrowMojo>());
          return llvm::Error::success();
        }},
       COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
