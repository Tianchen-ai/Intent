#include "Intent/Target/Triton/IR/Configuration.h"
#include "Intent/Target/Triton/IR/TritonOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Analysis/ProgramInterface.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/MathExtras.h"
#include <limits>

using namespace mlir;

namespace intent::triton {

bool isLegalDeviceOption(gpu::ParameterRole role, int64_t value,
                         gpu::CapabilitiesAttr capabilities,
                         bool requiresSingleCTA) {
  if (!capabilities || value <= 0) return false;
  switch (role) {
  case gpu::ParameterRole::ProviderWarps:
    return llvm::isPowerOf2_64(value) &&
           value <= capabilities.getMaxThreadsPerBlock() / 32;
  case gpu::ParameterRole::ProviderStages:
    return value <= std::numeric_limits<int32_t>::max();
  case gpu::ParameterRole::ProviderCTAs:
    return llvm::isPowerOf2_64(value) && value <= 16 &&
           (!requiresSingleCTA || value == 1) &&
           (value == 1 || (capabilities.getComputeCapabilityMajor() >= 9 &&
                          capabilities.getComputeCapabilityMajor() != 12));
  default: return false;
  }
}

FailureOr<ConfigurationSchema> ConfigurationSchema::read(func::FuncOp kernel) {
  ConfigurationSchema result;
  auto capabilities = kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
  if (!capabilities)
    return kernel.emitError("Triton configuration requires GPU capabilities"), failure();
  bool requiresSingleCTA = false;
  for (BlockArgument argument : kernel.getArguments()) {
    auto binding = gpu::getArgumentBinding(argument);
    requiresSingleCTA |= binding && binding.getKind() == gpu::ArgumentKind::Workspace;
  }
  kernel.walk([&](CtaBarrierOp) { requiresSingleCTA = true; });
  gpu::ParameterAttr stageDeclaration;
  if (failed(gpu::verifyParameterDeclarations(kernel))) return failure();
  auto parameters = kernel->getAttrOfType<ArrayAttr>(gpu::parametersAttr);
  for (Attribute attribute : parameters) {
    auto definition = cast<gpu::ParameterAttr>(attribute);
    if (!definition.isExtent()) continue;
    auto role = definition.getRole();
    StringAttr *option = nullptr;
    switch (role) {
    case gpu::ParameterRole::ProviderWarps: option = &result.warps; break;
    case gpu::ParameterRole::ProviderStages:
      option = &result.stages;
      stageDeclaration = definition;
      break;
    case gpu::ParameterRole::ProviderCTAs: option = &result.ctas; break;
    default: break;
    }
    bool provider = definition.getCategory() ==
                    gpu::ParameterCategory::Provider;
    if (provider != bool(option)) {
      return kernel.emitError("Triton program contains a foreign provider parameter"), failure();
    }
    if (option) {
      if (*option) {
        return kernel.emitError("duplicates a Triton provider-parameter role"), failure();
      }
      *option = definition.getName();
    }
    if (option)
      for (int64_t candidate : definition.getCandidates().asArrayRef())
        if (!isLegalDeviceOption(role, candidate, capabilities, requiresSingleCTA)) {
          kernel.emitError("declares a Triton option outside the current device/launch contract")
              << "; parameter=" << definition.getName() << "; value=" << candidate;
          return failure();
        }
  }
  if (!result.warps || !result.stages || !result.ctas)
    return kernel.emitError("Triton provider parameter domains are incomplete"), failure();

  bool stagesInKernel = false;
  auto walked = kernel.walk([&](Operation *operation) {
    Attribute stages = operation->getAttr(loopStagesAttr);
    if (!stages) return WalkResult::advance();
    if (!isa<scf::ForOp>(operation)) {
      operation->emitOpError("loop stages require an scf.for operation");
      return WalkResult::interrupt();
    }
    if (auto constant = dyn_cast<IntegerAttr>(stages)) {
      if (!constant.getType().isSignlessInteger(64) ||
          !isLegalDeviceOption(gpu::ParameterRole::ProviderStages,
                               constant.getInt(), capabilities,
                               requiresSingleCTA)) {
        operation->emitOpError("constant loop stages must be a positive i64 within the Triton stage limit");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }
    auto binding = dyn_cast<gpu::ParameterRefAttr>(stages);
    if (!binding || binding != stageDeclaration.getReference()) {
      operation->emitOpError("loop stages must be a positive i64 or bind the declared Triton stage parameter");
      return WalkResult::interrupt();
    }
    stagesInKernel = true;
    return WalkResult::advance();
  });
  if (walked.wasInterrupted()) return failure();

  llvm::SmallVector<gpu::ParameterRefAttr> formParameters;
  kernel.walk([&](TensorDescriptorChoiceOp choice) {
    formParameters.push_back(choice.getConfigParameter());
    result.kernelParameters.push_back(choice.getConfigParameter().getName());
  });
  for (Attribute attribute : parameters) {
    auto declaration = cast<gpu::ParameterAttr>(attribute);
    if (!declaration.isExtent() &&
        !llvm::is_contained(formParameters, declaration.getReference()))
      return kernel.emitError("boolean provider declaration has no tensor-descriptor choice"), failure();
  }
  for (Attribute attribute : parameters) {
    auto parameter = cast<gpu::ParameterAttr>(attribute);
    if (!parameter.isExtent()) continue;
    StringAttr name = parameter.getName();
    if (name == result.warps || name == result.ctas ||
        (name == result.stages && !stagesInKernel))
      continue;
    result.kernelParameters.push_back(name);
  }
  return result;
}

} // namespace intent::triton
