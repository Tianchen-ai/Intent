#include "../PassSupport.h"
#include "Intent/Dialect/CPU/Transforms/Contraction/Contraction.h"

using namespace mlir;

namespace intent::cpu {

#define GEN_PASS_DEF_CPUNORMALIZECONTRACTIONS
#define GEN_PASS_DEF_CPUFOLDCONTRACTIONINPUTS
#include "Intent/Dialect/CPU/Transforms/Passes.h.inc"

namespace {

class NormalizeContractionsPass
    : public impl::CPUNormalizeContractionsBase<NormalizeContractionsPass> {
public:
  using CPUNormalizeContractionsBase::CPUNormalizeContractionsBase;
  void runOnOperation() final {
    auto transform = [](func::FuncOp function) -> LogicalResult {
      if (failed(normalizeContractionSources(function)))
        return failure();
      return normalizeContractions(function);
    };
    if (failed(detail::transformFunctions(getOperation(), getArgument(),
                                          transform)))
      signalPassFailure();
  }
};

class FoldContractionInputsPass
    : public impl::CPUFoldContractionInputsBase<FoldContractionInputsPass> {
public:
  using CPUFoldContractionInputsBase::CPUFoldContractionInputsBase;
  void runOnOperation() final {
    if (failed(detail::transformWithImplementations(
            getOperation(), getArgument(), provider.getValue(),
            foldContractionInputs)))
      signalPassFailure();
  }
};

} // namespace
} // namespace intent::cpu
