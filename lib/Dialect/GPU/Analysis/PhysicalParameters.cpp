#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/IR/ProgramInterface.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;

namespace intent::gpu {

ArrayAttr getParameterDeclarations(func::FuncOp kernel) {
  return kernel->getAttrOfType<ArrayAttr>(parametersAttr);
}

ParameterAttr lookupParameter(func::FuncOp kernel, StringAttr name) {
  return name ? lookupParameterDeclaration(kernel, ParameterRefAttr::get(kernel.getContext(), name))
              : ParameterAttr();
}

ParameterAttr lookupParameter(func::FuncOp kernel, ParameterRefAttr reference) {
  return lookupParameterDeclaration(kernel, reference);
}

FailureOr<ParameterSpace> ParameterSpace::read(func::FuncOp kernel) {
  ParameterSpace result;
  result.kernel = kernel;
  auto declarations = getParameterDeclarations(kernel);
  if (!declarations)
    return kernel.emitError("requires a kernel-owned parameter declaration table"), failure();
  llvm::StringSet<> names;
  for (Attribute attribute : declarations) {
    auto parameter = dyn_cast<ParameterAttr>(attribute);
    if (!parameter || !names.insert(parameter.getName().getValue()).second)
      return kernel.emitError("parameter declarations must be typed and have unique names"), failure();
    result.parameters.push_back(parameter);
    if (parameter.isExtent()) result.extents.push_back(parameter);
  }
  return result;
}

ParameterAttr ParameterSpace::lookup(StringAttr name) const {
  for (ParameterAttr parameter : parameters)
    if (parameter.getName() == name) return parameter;
  return {};
}

ParameterAttr ParameterSpace::lookup(ParameterRefAttr reference) const {
  return reference ? lookup(reference.getName()) : ParameterAttr();
}

ParameterAttr ParameterSpace::find(ParameterRole role) const {
  for (ParameterAttr parameter : parameters)
    if (parameter.getRole() == role) return parameter;
  return {};
}

LogicalResult ParameterSpace::verifyBindings(DictionaryAttr bindings,
                                             ConfigurationStage stage) const {
  if (!bindings)
    return kernel->emitError("configuration requires a binding dictionary");
  unsigned expected = 0;
  for (ParameterAttr parameter : parameters) {
    auto phase = parameter.getPhase();
    if (phase == ConfigurationBindingPhase::Deferred)
      continue;
    ++expected;
    auto value = bindings.getAs<IntegerAttr>(parameter.getName());
    if (!value || !value.getType().isSignlessInteger(64) ||
        !llvm::is_contained(parameter.getCandidates().asArrayRef(), value.getInt()))
      return kernel->emitError("configuration has no in-domain binding for ")
             << parameter.getName().getValue();
  }
  if (bindings.size() != expected)
    return kernel->emitError("configuration contains an undeclared or deferred binding");
  return success();
}

LogicalResult ParameterSpace::verifyRequirements(
    ArrayRef<ConfigurationRequirementAttr> requirements) const {
  bool valid = true;
  AttrTypeWalker walker;
  walker.addWalk([&](PhysicalExprAttr expression) {
    if (!valid) return;
    if (expression.getKind() == PhysicalExprKind::Parameter) {
      auto declaration = lookup(expression.getParameterReference());
      if (!declaration || !declaration.isExtent()) {
        kernel->emitError("configuration requirement references an undeclared or non-index parameter: ")
            << expression.getParameterReference();
        valid = false;
      } else if (declaration.isDeferred() &&
                 !declaration.getBinding().getCoverageBound()) {
        kernel->emitError("configuration requirement references an unresolved coverage parameter: ")
            << expression.getParameterReference();
        valid = false;
      }
    } else if (expression.getKind() == PhysicalExprKind::Dimension) {
      auto argument = resolveArgument(kernel, expression.getArgumentReference());
      if (!argument || queryArgumentExpression(argument) != expression) {
        kernel->emitError("configuration requirement does not match its current argument binding: ")
            << expression;
        valid = false;
      }
    }
  });
  for (ConfigurationRequirementAttr requirement : requirements) {
    if (!requirement)
      return kernel->emitError("configuration requirement cannot be null");
    if (failed(ConfigurationRequirementAttr::verify(
            [&] { return kernel->emitError("invalid configuration requirement: "); },
            requirement.getKind(), requirement.getMetric(),
            requirement.getPredicate(), requirement.getUsage(),
            requirement.getLimit(), requirement.getActivation(),
            requirement.getMessage())))
      return failure();
    if (ParameterRefAttr activation = requirement.getActivation()) {
      auto parameter = lookup(activation);
      if (!parameter || !parameter.getValueType().isSignlessInteger(1) ||
          parameter.getPhase() != ConfigurationBindingPhase::Provider)
        return kernel->emitError(
                   "configuration activation requires a declared provider boolean: ")
               << activation;
    }
    walker.walk(requirement);
    if (!valid) return failure();
  }
  return success();
}

FailureOr<SmallVector<ConfigurationRequirementAttr>>
ParameterSpace::requirements() const {
  SmallVector<ConfigurationRequirementAttr> result;
  Attribute attribute = kernel->getAttr(configurationsAttr);
  if (!attribute) return result;
  auto set = dyn_cast<ConfigurationSetAttr>(attribute);
  if (!set || !set.getRequirements())
    return kernel->emitError("configuration requirements require a complete typed configuration set");
  if (failed(ConfigurationSetAttr::verify(
          [&] { return kernel->emitError("invalid configuration set: "); },
          set.getStage(), set.getRows(), set.getRequirements())))
    return failure();
  for (Attribute attribute : set.getRequirements()) {
    auto requirement = dyn_cast<ConfigurationRequirementAttr>(attribute);
    if (!requirement)
      return kernel->emitError("configuration set contains an untyped requirement");
    result.push_back(requirement);
  }
  if (failed(verifyRequirements(result))) return failure();
  return result;
}

FailureOr<SmallVector<DictionaryAttr>>
ParameterSpace::configurations(ConfigurationStage stage) const {
  auto set = kernel->getAttrOfType<ConfigurationSetAttr>(configurationsAttr);
  if (!set || set.getStage() != stage)
    return kernel->emitError("requires a ") << stringifyConfigurationStage(stage)
           << " configuration set for the current program";
  SmallVector<DictionaryAttr> result;
  llvm::DenseSet<Attribute> unique;
  if (set.getRows().empty())
    return kernel->emitError("configuration set cannot be empty");
  if (failed(requirements())) return failure();
  for (Attribute attribute : set.getRows()) {
    auto row = dyn_cast<DictionaryAttr>(attribute);
    if (failed(verifyBindings(row, stage))) return failure();
    if (!unique.insert(row).second)
      return kernel->emitError("configuration set contains duplicate bindings");
    result.push_back(row);
  }
  return result;
}

} // namespace intent::gpu
