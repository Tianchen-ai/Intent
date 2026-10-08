#include "../PassSupport.h"
#include "Intent/Dialect/CPU/Transforms/Configuration/Configuration.h"

using namespace mlir;

namespace intent::cpu {

#define GEN_PASS_DEF_CPUCONFIGURETARGET
#define GEN_PASS_DEF_CPUMATERIALIZECONFIGURATIONS
#include "Intent/Dialect/CPU/Transforms/Passes.h.inc"

namespace {

class ConfigureTargetPass
    : public impl::CPUConfigureTargetBase<ConfigureTargetPass> {
public:
  using CPUConfigureTargetBase::CPUConfigureTargetBase;
  void runOnOperation() final {
    auto module = getOperation();
    auto capabilities = CapabilitiesAttr::getChecked(
        [&]() { return module.emitError(); }, module.getContext(),
        vectorBits.getValue(), workers.getValue(), privateBytes.getValue(),
        matrixI8I32.getValue());
    if (!capabilities)
      return signalPassFailure();
    module->setAttr("intent_cpu.capabilities", capabilities);
    if (failed(detail::finishTransform(module, getArgument(), success(),
                                       CPUProgramStage::Values)))
      signalPassFailure();
  }
};

class MaterializeConfigurationsPass
    : public impl::CPUMaterializeConfigurationsBase<MaterializeConfigurationsPass> {
public:
  using CPUMaterializeConfigurationsBase::CPUMaterializeConfigurationsBase;
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(detail::verifyPassInput(module, getArgument(),
                                       CPUProgramStage::Buffers)))
      return signalPassFailure();
    auto implementations =
        lookupImplementationProvider(module, provider.getValue());
    if (failed(implementations))
      return signalPassFailure();
    if (failed(detail::finishTransform(
            module, getArgument(),
            materializeCPUConfigurations(module, **implementations,
                                         defaults.getValue(),
                                         overrides.getValue()))))
      signalPassFailure();
  }
};

} // namespace
} // namespace intent::cpu
