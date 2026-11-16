#include "Configurations.h"
#include "ConfigurationFacts.h"
#include "Intent/Dialect/Block/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/Block/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/Block/IR/PhysicalExpressions.h"
#include "Intent/Dialect/Block/IR/Program.h"
#include "Intent/Dialect/Block/Transforms/Configuration/PhysicalParameters.h"
#include "Intent/Dialect/Block/Transforms/Configuration/Resources.h"
#include "Intent/Target/Triton/IR/Configuration.h"
#include "Intent/Target/Triton/Transforms/Configuration/TuningProfiles.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
namespace intent::triton {
const block::TuningProfileSchema &tuningProfileSchema() {
  static const StringRef columns[] = {
      "ownership_m", "ownership_n", "reduction", "reduction_outer", "scan",
      "traversal_workers", "traversal_group", "NUM_WARPS", "NUM_STAGES",
      "NUM_CTAS", "USE_TENSOR_DESCRIPTOR"};
  static const block::ProviderOption options[] = {
      {"NUM_WARPS"}, {"NUM_STAGES"}, {"NUM_CTAS"}, {"USE_TENSOR_DESCRIPTOR", true}};
  static const block::TuningProfileSchema schema{"triton", columns, options};
  return schema;
}

namespace {
LogicalResult bindProviderDomains(func::FuncOp kernel,
                                  ArrayRef<DictionaryAttr> rows) {
  Builder builder(kernel.getContext());
  for (const block::ProviderOption &option : tuningProfileSchema().options) {
    auto declaration = block::lookupParameter(kernel, builder.getStringAttr(option.name));
    if (!declaration)
      return kernel.emitError("Triton configuration has no declared provider option: ")
             << option.name;
    SmallVector<int64_t> domain;
    for (DictionaryAttr row : rows) {
      auto value = row.getAs<IntegerAttr>(option.name);
      if (!value)
        return kernel.emitError("Triton configuration row has no provider option: ")
               << option.name;
      if (!llvm::is_contained(domain, value.getInt())) domain.push_back(value.getInt());
    }
    auto selected = block::ParameterAttr::get(kernel.getContext(), declaration.getName(),
        declaration.getValueType(), declaration.getRole(), declaration.getCategory(),
        declaration.getElementBitWidth(), builder.getDenseI64ArrayAttr(domain),
        declaration.getPhase(), declaration.getBinding());
    if (failed(block::updateParameter(kernel, selected))) return failure();
  }
  return success();
}

} // namespace

LogicalResult materializeLegalConfigs(func::FuncOp kernel,
                                      TensorDescriptorChoiceOp descriptorChoice) {
  auto space = block::ParameterSpace::read(kernel);
  if (failed(space))
    return failure();
  auto shared = space->configurations(block::ConfigurationStage::Shared);
  if (failed(shared))
    return failure();
  auto form = space->lookup("USE_TENSOR_DESCRIPTOR");
  if (!form)
    return kernel.emitError("Triton configuration has no declared access form");
  SmallVector<DictionaryAttr> configs;
  Builder builder(kernel.getContext());
  for (DictionaryAttr tuple : *shared) {
    NamedAttrList bindings(tuple);
    if (!descriptorChoice) bindings.erase(form.getName());
    DictionaryAttr config = bindings.getDictionary(kernel.getContext());
    if (!llvm::is_contained(configs, config)) configs.push_back(config);
  }
  if (!descriptorChoice) {
    SmallVector<Attribute> declarations;
    for (Attribute attribute : block::getParameterDeclarations(kernel))
      if (attribute != form) declarations.push_back(attribute);
    kernel->setAttr(block::parametersAttr, builder.getArrayAttr(declarations));
  }
  if (failed(ConfigurationSchema::read(kernel))) return failure();

  auto requirements = collectConfigurationRequirements(kernel);
  if (failed(requirements)) return failure();
  auto filtered = block::filterConfigurationRequirements(kernel, configs, *requirements);
  if (failed(filtered)) return failure();
  return block::writeConfigurations(kernel, *filtered, block::ConfigurationStage::Complete, *requirements);
}

FailureOr<SmallVector<TritonLocalOptions>> declareProviderOptions(
    func::FuncOp kernel,
    bool requiresCtaSynchronization, ArrayRef<scf::ForOp> loadPipelineLoops) {
  const ProgramConfigurationFacts facts(kernel);
  auto capabilities = kernel->getAttrOfType<block::CapabilitiesAttr>(block::capabilitiesAttr);
  auto space = block::ParameterSpace::read(kernel);
  if (failed(space)) return failure();
  auto rows = space->configurations(block::ConfigurationStage::Shared);
  if (failed(rows)) return failure();
  bool pipelineStagesAffectProgram = facts.pipelineStagesAffectProgram || !loadPipelineLoops.empty();
  SmallVector<TritonLocalOptions> localOptions;
  SmallVector<DictionaryAttr> selectedRows;
  DenseMap<DictionaryAttr, int64_t> equivalentStages;
  Builder builder(kernel.getContext());
  auto isDeviceOption = [&](int64_t warps, int64_t stages, int64_t ctas) {
    return isLegalDeviceOption(LaunchOption::Warps, warps,
                               capabilities, requiresCtaSynchronization) &&
           isLegalDeviceOption(LaunchOption::Stages, stages,
                               capabilities, requiresCtaSynchronization) &&
           isLegalDeviceOption(LaunchOption::CTAs, ctas,
                               capabilities, requiresCtaSynchronization);
  };
  for (DictionaryAttr row : *rows) {
    int64_t warps = row.getAs<IntegerAttr>("NUM_WARPS").getInt();
    int64_t stages = row.getAs<IntegerAttr>("NUM_STAGES").getInt();
    int64_t ctas = row.getAs<IntegerAttr>("NUM_CTAS").getInt();
    if (facts.straightLinePointwise)
      stages = 1;
    if (!isDeviceOption(warps, stages, ctas))
      continue;
    NamedAttrList bindings(row);
    if (!pipelineStagesAffectProgram) {
      bindings.erase("NUM_STAGES");
      auto inserted = equivalentStages.try_emplace(bindings.getDictionary(kernel.getContext()), stages);
      stages = inserted.first->second;
    }
    bindings.set("NUM_STAGES", builder.getI64IntegerAttr(stages));
    DictionaryAttr selected = bindings.getDictionary(kernel.getContext());
    if (!llvm::is_contained(selectedRows, selected)) selectedRows.push_back(selected);
    if (llvm::any_of(localOptions, [&](const TritonLocalOptions &option) {
          return option.warps == warps && option.stages == stages && option.ctas == ctas;
        }))
      continue;
    localOptions.push_back({warps, stages, ctas});
  }
  if (localOptions.empty())
    return kernel.emitError("Triton tuning profile has no legal provider options");
  if (failed(bindProviderDomains(kernel, selectedRows)) ||
      failed(block::writeConfigurations(kernel, selectedRows, block::ConfigurationStage::Shared)))
    return failure();
  if (!loadPipelineLoops.empty()) {
    auto stages =
        block::queryParameterBySymbol(kernel, builder.getStringAttr("NUM_STAGES"));
    if (failed(stages))
      return failure();
    for (scf::ForOp loop : loadPipelineLoops)
      loop->setAttr(loopStagesAttr, stages->getReference());
  }
  return localOptions;
}

LogicalResult finalizeConfigurationRequirements(func::FuncOp kernel) {
  auto space = block::ParameterSpace::read(kernel);
  if (failed(space))
    return failure();
  auto rows = space->configurations(block::ConfigurationStage::Complete);
  auto requirements = collectConfigurationRequirements(kernel);
  if (failed(rows) || failed(requirements))
    return failure();
  auto accepted = block::filterConfigurationRequirements(kernel, *rows, *requirements);
  if (failed(accepted))
    return failure();
  return block::writeConfigurations(kernel, *accepted,
                                 block::ConfigurationStage::Complete, *requirements);
}

} // namespace intent::triton
