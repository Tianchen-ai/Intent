#include "../PassSupport.h"
#include "Intent/Dialect/CPU/Transforms/Control/Traversals.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

using namespace mlir;

namespace intent::cpu {

#define GEN_PASS_DEF_CPUFUSESHAREDTRAVERSALS
#include "Intent/Dialect/CPU/Transforms/Passes.h.inc"

namespace {

class FuseSharedTraversalsPass
    : public impl::CPUFuseSharedTraversalsBase<FuseSharedTraversalsPass> {
public:
  using CPUFuseSharedTraversalsBase::CPUFuseSharedTraversalsBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    CPUFuseSharedTraversalsBase::getDependentDialects(registry);
    registry.insert<affine::AffineDialect>();
  }

  void runOnOperation() final {
    if (failed(detail::transformFunctions(getOperation(), getArgument(),
                                          fuseSharedTraversals)))
      signalPassFailure();
  }
};

} // namespace

} // namespace intent::cpu
