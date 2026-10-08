#include "Intent/Dialect/GPU/IR/ProgramInterface.h"
#include "ConfigurationPolicy.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/PhysicalParameters.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::gpu {
namespace {

LogicalResult materializeCoverageBound(func::FuncOp kernel, ParameterAttr parameter) {
  auto schema = parameter;
  auto role = schema.getRole();
  auto category = schema.getCategory();
  const bool coverage = parameter.isDeferred();
  if ((role == ParameterRole::FullCoverage ||
       category == ParameterCategory::Coverage) &&
      (!coverage || category != ParameterCategory::Coverage)) {
    kernel.emitOpError(
        "full-coverage parameter lacks its typed coverage category or dimension");
    return failure();
  }
  if (coverage && !parameter.getBinding().getCoverageBound()) {
    PhysicalParameterBinding binding = queryParameterBinding(parameter);
    PhysicalExprAttr bound;
    if (binding.isExact() && binding.dimension)
      bound = queryArgumentExpression(resolveDimension(kernel, *binding.dimension));
    if (!bound) {
      kernel.emitOpError(
          "full-coverage parameter has no launch-visible bound expression");
      return failure();
    }
    auto previous = parameter.getBinding();
    auto updatedBinding = ParameterBindingAttr::get(
        kernel.getContext(), previous.getDimension(), previous.getSource(),
        bound, previous.getGroup(), previous.getPointwiseChunk(),
        previous.getPointwiseLocal());
    return updateParameter(kernel, ParameterAttr::get(
        kernel.getContext(), parameter.getName(), parameter.getValueType(),
        parameter.getRole(), parameter.getCategory(), parameter.getElementBitWidth(),
        parameter.getCandidates(), parameter.getPhase(), updatedBinding));
  }
  return success();
}

} // namespace

LogicalResult materializeSharedConfigTuples(func::FuncOp kernel) {
  auto tables = TuningProfiles::from(kernel->getParentOfType<ModuleOp>());
  if (failed(tables))
    return failure();
  auto declarations = ParameterSpace::read(kernel);
  if (failed(declarations))
    return failure();
  SmallVector<ParameterAttr> parameters;
  for (ParameterAttr parameter : declarations->extentDeclarations()) {
    if (failed(materializeCoverageBound(kernel, parameter)))
      return failure();
    if (parameter.getPhase() == ConfigurationBindingPhase::Shared)
      parameters.push_back(parameter);
  }

  Builder builder(kernel.getContext());
  const FragmentResourceAnalysis resources(kernel);
  const auto facts =
      configuration::analyzeConfigurationPolicy(kernel, parameters, resources);
  SmallVector<DictionaryAttr> tuples;
  auto append = [&](DictionaryAttr tuple) {
    if (!llvm::is_contained(tuples, tuple))
      tuples.push_back(tuple);
  };
  bool invalidFootprint = false;
  auto project = configuration::projectConfigurationProfiles(
      kernel, facts, *tables,
      [&](NamedAttrList &bindings, const configuration::TuningProfile &profile) {
        configuration::bindContractionFreeExtents(
            facts.freeExtents, bindings, profile, builder);
        if (failed(configuration::bindTraversalFragmentFootprints(
                kernel, parameters, resources, bindings, builder))) {
          invalidFootprint = true;
          return;
        }
        append(bindings.getDictionary(kernel.getContext()));
      });
  if (failed(project))
    return failure();
  if (invalidFootprint && tuples.empty())
    return kernel.emitOpError(
        "physical traversal has no profile within the fragment register budget");
  return writeConfigurations(kernel, tuples, ConfigurationStage::Shared);
}

LogicalResult verifySharedConfigTuples(func::FuncOp kernel) {
  auto space = ParameterSpace::read(kernel);
  if (failed(space))
    return failure();
  for (ParameterAttr parameter : space->extentDeclarations()) {
    if (parameter.isDeferred() && !parameter.getBinding().getCoverageBound()) {
      kernel.emitOpError(
          "full-coverage parameter requires a typed bound expression");
      return failure();
    }
  }
  auto tuples = space->configurations(ConfigurationStage::Shared);
  return failed(tuples) ? failure() : success();
}

} // namespace intent::gpu
