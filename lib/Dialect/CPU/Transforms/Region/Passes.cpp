#include "../PassSupport.h"
#include "Intent/Dialect/CPU/Transforms/Region/Regions.h"

using namespace mlir;

namespace intent::cpu {

#define GEN_PASS_DEF_CPUGROUPREGIONS
#define GEN_PASS_DEF_CPUREALIZEREGIONS
#include "Intent/Dialect/CPU/Transforms/Passes.h.inc"

namespace {

class GroupRegionsPass : public impl::CPUGroupRegionsBase<GroupRegionsPass> {
public:
  using CPUGroupRegionsBase::CPUGroupRegionsBase;
  void runOnOperation() final {
    auto transform = [](func::FuncOp function) -> LogicalResult {
      auto configuration = detail::currentConfiguration(function);
      return failed(configuration)
                 ? failure()
                 : groupRegionComputations(function, *configuration);
    };
    if (failed(detail::transformFunctions(getOperation(), getArgument(),
                                          transform)))
      signalPassFailure();
  }
};

class RealizeRegionsPass
    : public impl::CPURealizeRegionsBase<RealizeRegionsPass> {
public:
  using CPURealizeRegionsBase::CPURealizeRegionsBase;
  void runOnOperation() final {
    auto transform = [&](func::FuncOp function,
                        const ImplementationRegistry &implementations) {
      auto configuration = detail::currentConfiguration(function);
      return failed(configuration)
                 ? failure()
                 : realizeRegions(function, *configuration, implementations,
                                  simplifyFirstSummary);
    };
    if (failed(detail::transformWithImplementations(
            getOperation(), getArgument(), provider.getValue(), transform)))
      signalPassFailure();
  }
};

} // namespace
} // namespace intent::cpu
