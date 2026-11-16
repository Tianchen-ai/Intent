#include "Intent/Target/Triton/IR/Configuration.h"
#include "Intent/Target/NVIDIA/IR/Target.h"
#include "Intent/Target/Triton/IR/TritonOps.h"
#include "Intent/Dialect/Block/IR/Program.h"
#include "Intent/Dialect/Block/Analysis/ProgramInterface.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/MathExtras.h"
#include <limits>

using namespace mlir;

namespace intent::triton {

bool isLegalDeviceOption(block::ParameterRole role, int64_t value,
                         block::CapabilitiesAttr capabilities,
                         bool requiresSingleCTA) {
  if (!capabilities || value <= 0) return false;
  auto device = dyn_cast<nvidia::DeviceAttr>(capabilities.getTarget());
  switch (role) {
  case block::ParameterRole::ProviderWarps:
    return llvm::isPowerOf2_64(value) &&
           value <= capabilities.getPreferredVectorElements() / 32;
  case block::ParameterRole::ProviderStages:
    return value <= std::numeric_limits<int32_t>::max();
  case block::ParameterRole::ProviderCTAs:
    return llvm::isPowerOf2_64(value) && value <= 16 &&
           (!requiresSingleCTA || value == 1) &&
           (value == 1 || (device && device.getComputeCapabilityMajor() >= 9 &&
                          device.getComputeCapabilityMajor() != 12));
  default: return false;
  }
}

FailureOr<ConfigurationSchema> ConfigurationSchema::read(func::FuncOp kernel) {
  ConfigurationSchema result;
  auto capabilities = kernel->getAttrOfType<block::CapabilitiesAttr>(block::capabilitiesAttr);
  if (!capabilities)
    return kernel.emitError("Triton configuration requires Block capabilities"), failure();
  bool requiresSingleCTA = false;
  for (BlockArgument argument : kernel.getArguments()) {
    auto binding = block::getArgumentBinding(argument);
    requiresSingleCTA |= binding && binding.getKind() == block::ArgumentKind::Workspace;
  }
  kernel.walk([&](CtaBarrierOp) { requiresSingleCTA = true; });
  block::ParameterAttr stageDeclaration;
  if (failed(block::verifyParameterDeclarations(kernel))) return failure();
  auto parameters = kernel->getAttrOfType<ArrayAttr>(block::parametersAttr);
  for (Attribute attribute : parameters) {
    auto definition = cast<block::ParameterAttr>(attribute);
    if (!definition.isExtent()) continue;
    auto role = definition.getRole();
    StringAttr *option = nullptr;
    switch (role) {
    case block::ParameterRole::ProviderWarps: option = &result.warps; break;
    case block::ParameterRole::ProviderStages:
      option = &result.stages;
      stageDeclaration = definition;
      break;
    case block::ParameterRole::ProviderCTAs: option = &result.ctas; break;
    default: break;
    }
    bool provider = definition.getCategory() ==
                    block::ParameterCategory::Provider;
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
          !isLegalDeviceOption(block::ParameterRole::ProviderStages,
                               constant.getInt(), capabilities,
                               requiresSingleCTA)) {
        operation->emitOpError("constant loop stages must be a positive i64 within the Triton stage limit");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }
    auto binding = dyn_cast<block::ParameterRefAttr>(stages);
    if (!binding || binding != stageDeclaration.getReference()) {
      operation->emitOpError("loop stages must be a positive i64 or bind the declared Triton stage parameter");
      return WalkResult::interrupt();
    }
    stagesInKernel = true;
    return WalkResult::advance();
  });
  if (walked.wasInterrupted()) return failure();

  llvm::SmallVector<block::ParameterRefAttr> formParameters;
  kernel.walk([&](TensorDescriptorChoiceOp choice) {
    formParameters.push_back(choice.getConfigParameter());
    result.kernelParameters.push_back(choice.getConfigParameter().getName());
  });
  for (Attribute attribute : parameters) {
    auto declaration = cast<block::ParameterAttr>(attribute);
    if (!declaration.isExtent() &&
        !llvm::is_contained(formParameters, declaration.getReference()))
      return kernel.emitError("boolean provider declaration has no tensor-descriptor choice"), failure();
  }
  for (Attribute attribute : parameters) {
    auto parameter = cast<block::ParameterAttr>(attribute);
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
