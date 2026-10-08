#include "../PassSupport.h"
#include "Intent/Dialect/CPU/Transforms/Implementation/ImplementationInputs.h"

using namespace mlir;

namespace intent::cpu {

#define GEN_PASS_DEF_CPUPREPAREINPUTS
#include "Intent/Dialect/CPU/Transforms/Passes.h.inc"

namespace {

class PrepareInputsPass : public impl::CPUPrepareInputsBase<PrepareInputsPass> {
public:
  using CPUPrepareInputsBase::CPUPrepareInputsBase;
  void runOnOperation() final {
    auto transform = [](func::FuncOp function,
                        const ImplementationRegistry &implementations) -> LogicalResult {
      if (failed(reusePreparedInputs(function, implementations)))
        return failure();
      return groupQuantizedDots(function, implementations);
    };
    if (failed(detail::transformWithImplementations(
            getOperation(), getArgument(), provider.getValue(), transform)))
      signalPassFailure();
  }
};

} // namespace
} // namespace intent::cpu
