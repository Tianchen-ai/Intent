#include "../PassSupport.h"
#include "Intent/Dialect/CPU/Transforms/Structure/Computations.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

using namespace mlir;

namespace intent::cpu {

#define GEN_PASS_DEF_CPUFOLDUNIFORMCOMPUTATIONS
#define GEN_PASS_DEF_CPUFUSESTRUCTUREDCOMPUTATIONS
#include "Intent/Dialect/CPU/Transforms/Passes.h.inc"

namespace {

class FoldUniformComputationsPass
    : public impl::CPUFoldUniformComputationsBase<FoldUniformComputationsPass> {
public:
  using CPUFoldUniformComputationsBase::CPUFoldUniformComputationsBase;
  void runOnOperation() final {
    if (failed(detail::transformFunctions(getOperation(), getArgument(),
                                          foldUniformComputations)))
      signalPassFailure();
  }
};

class FuseStructuredComputationsPass
    : public impl::CPUFuseStructuredComputationsBase<FuseStructuredComputationsPass> {
public:
  using CPUFuseStructuredComputationsBase::CPUFuseStructuredComputationsBase;
  void getDependentDialects(DialectRegistry &registry) const override {
    CPUFuseStructuredComputationsBase::getDependentDialects(registry);
    registry.insert<affine::AffineDialect>();
  }
  void runOnOperation() final {
    if (failed(detail::transformFunctions(getOperation(), getArgument(),
                                          fuseStructuredComputations)))
      signalPassFailure();
  }
};

} // namespace
} // namespace intent::cpu
