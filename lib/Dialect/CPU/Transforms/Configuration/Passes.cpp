#include "../PassSupport.h"
#include "Intent/Dialect/CPU/Transforms/Configuration/Configuration.h"
#include "mlir/AsmParser/AsmParser.h"

using namespace mlir;

namespace intent::cpu {

#define GEN_PASS_DEF_CPUCONFIGURETARGET
#define GEN_PASS_DEF_CPUBINDPROFILES
#define GEN_PASS_DEF_CPUMATERIALIZECONFIGURATIONS
#include "Intent/Dialect/CPU/Transforms/Passes.h.inc"

namespace {

class ConfigureTargetPass
    : public impl::CPUConfigureTargetBase<ConfigureTargetPass> {
public:
  using CPUConfigureTargetBase::CPUConfigureTargetBase;
  void runOnOperation() final {
    auto module = getOperation();
    auto capabilities = dyn_cast_or_null<CapabilitiesAttr>(parseAttribute(target, &getContext()));
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
    if (failed(implementations) || failed(verifyCPUProfiles(module, **implementations, provider)))
      return signalPassFailure();
    if (failed(detail::finishTransform(
            module, getArgument(),
            materializeCPUConfigurations(module, **implementations))))
      signalPassFailure();
  }
};

class BindProfilesPass : public impl::CPUBindProfilesBase<BindProfilesPass> {
public:
  using CPUBindProfilesBase::CPUBindProfilesBase;
  void runOnOperation() final {
    auto module = getOperation();
    auto implementations = lookupImplementationProvider(module, provider);
    if (failed(implementations) || failed(detail::verifyPassInput(module, getArgument(), CPUProgramStage::Values)) ||
        failed(detail::finishTransform(module, getArgument(),
            bindCPUProfiles(module, **implementations, provider, defaults, overrides), CPUProgramStage::Values)))
      signalPassFailure();
  }
};

} // namespace
} // namespace intent::cpu
