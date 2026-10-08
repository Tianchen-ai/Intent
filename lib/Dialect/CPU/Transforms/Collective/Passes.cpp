#include "../PassSupport.h"
#include "Intent/Dialect/CPU/Transforms/Collective/Collectives.h"

using namespace mlir;

namespace intent::cpu {

#define GEN_PASS_DEF_CPUREALIZESLICECOLLECTIVES
#define GEN_PASS_DEF_CPUNORMALIZEREDUCTIONS
#define GEN_PASS_DEF_CPUREALIZEHISTOGRAMS
#include "Intent/Dialect/CPU/Transforms/Passes.h.inc"

namespace {

class RealizeSliceCollectivesPass
    : public impl::CPURealizeSliceCollectivesBase<RealizeSliceCollectivesPass> {
public:
  using CPURealizeSliceCollectivesBase::CPURealizeSliceCollectivesBase;
  void runOnOperation() final {
    if (failed(detail::transformFunctions(getOperation(), getArgument(),
                                          realizeSliceCollectives)))
      signalPassFailure();
  }
};

class NormalizeReductionsPass
    : public impl::CPUNormalizeReductionsBase<NormalizeReductionsPass> {
public:
  using CPUNormalizeReductionsBase::CPUNormalizeReductionsBase;
  void runOnOperation() final {
    if (failed(detail::transformFunctions(getOperation(), getArgument(),
                                          normalizeScalarReductions)))
      signalPassFailure();
  }
};

class RealizeHistogramsPass
    : public impl::CPURealizeHistogramsBase<RealizeHistogramsPass> {
public:
  using CPURealizeHistogramsBase::CPURealizeHistogramsBase;
  void runOnOperation() final {
    if (failed(detail::transformFunctions(getOperation(), getArgument(),
                                          realizeHistograms)))
      signalPassFailure();
  }
};

} // namespace
} // namespace intent::cpu
