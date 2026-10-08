#include "Intent/Dialect/GPU/Analysis/Workspace.h"
#include "Intent/Dialect/GPU/Analysis/ConfigurationExpressions.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/IR/PhysicalExpressions.h"
#include "Intent/Dialect/GPU/IR/ProgramInterface.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::gpu {
namespace {

PhysicalExprAttr constant(MLIRContext *context, int64_t value) {
  return PhysicalExprAttr::get(context, PhysicalExprKind::Constant, value,
                               StringAttr::get(context, ""),
                               ArrayAttr::get(context, {}));
}

PhysicalExprAttr maximum(PhysicalExprAttr lhs, PhysicalExprAttr rhs) {
  if (lhs == rhs) return lhs;
  auto expression = PhysicalExprAttr::get(
      lhs.getContext(), PhysicalExprKind::Maximum, 0,
      StringAttr::get(lhs.getContext(), ""),
      ArrayAttr::get(lhs.getContext(), {lhs, rhs}));
  if (auto value = constantPhysicalExpression(expression))
    return constant(lhs.getContext(), *value);
  return expression;
}

} // namespace

FailureOr<ArrayAttr> workspaceAllocationShape(func::FuncOp kernel,
                                            ArrayAttr shape) {
  if (!shape)
    return kernel.emitError("workspace allocation requires a typed shape"),
           failure();
  auto space = ParameterSpace::read(kernel);
  if (failed(space)) return failure();
  auto rows = space->configurations(ConfigurationStage::Shared);
  if (failed(rows)) return failure();

  bool valid = true;
  AttrTypeWalker references;
  references.addWalk([&](PhysicalExprAttr expression) {
    if (!valid) return;
    if (expression.getKind() == PhysicalExprKind::Parameter) {
      auto declaration = space->lookup(expression.getParameterReference());
      if (!declaration || !declaration.isExtent()) {
        kernel.emitError(
            "workspace shape references an undeclared extent parameter: ")
            << expression.getParameterReference();
        valid = false;
      } else if (declaration.getPhase() == ConfigurationBindingPhase::Provider) {
        kernel.emitError(
            "workspace allocation cannot depend on an unselected provider "
            "parameter: ")
            << declaration.getName();
        valid = false;
      } else if (declaration.isDeferred() &&
                 !declaration.getBinding().getCoverageBound()) {
        kernel.emitError(
            "workspace allocation requires a host-bound deferred parameter: ")
            << declaration.getName();
        valid = false;
      }
    } else if (expression.getKind() == PhysicalExprKind::Dimension ||
               expression.getKind() == PhysicalExprKind::ScalarABI) {
      auto argument = resolveArgument(kernel, expression.getArgumentReference());
      if (!argument || queryArgumentExpression(argument) != expression) {
        kernel.emitError("workspace shape does not match its current ABI binding: ")
            << expression;
        valid = false;
      }
    }
  });
  for (Attribute axis : shape) {
    if (!isa<PhysicalExprAttr>(axis))
      return kernel.emitError(
                 "workspace shape requires physical extent expressions"),
             failure();
    references.walk(axis);
  }
  if (!valid) return failure();

  SmallVector<SmallVector<PhysicalExprAttr>> alternatives(shape.size());
  SmallVector<llvm::DenseSet<PhysicalExprAttr>> seen(shape.size());
  for (DictionaryAttr row : *rows) {
    for (auto [axis, attribute] : llvm::enumerate(shape)) {
      auto instantiated = instantiateConfigurationExpression(
          cast<PhysicalExprAttr>(attribute), row);
      if (failed(instantiated))
        return kernel.emitError(
                   "workspace extent has invalid concrete arithmetic in shared "
                   "configuration ")
                   << row << ": " << attribute,
               failure();
      PhysicalExprAttr bound = *instantiated;
      if (auto value = constantPhysicalExpression(bound); value && *value < 0)
        return kernel.emitError(
                   "workspace extent is negative in shared configuration ")
                   << row << ": " << bound,
               failure();
      if (seen[axis].insert(bound).second)
        alternatives[axis].push_back(bound);
    }
  }

  SmallVector<Attribute> result;
  for (auto &axis : alternatives) {
    // A balanced expression avoids a linear nesting depth for large finite
    // candidate tables. Each leaf still represents one complete tuple.
    while (axis.size() > 1) {
      SmallVector<PhysicalExprAttr> next;
      for (size_t index = 0; index < axis.size(); index += 2)
        next.push_back(index + 1 < axis.size()
                           ? maximum(axis[index], axis[index + 1])
                           : axis[index]);
      axis = std::move(next);
    }
    result.push_back(axis.front());
  }
  return ArrayAttr::get(kernel.getContext(), result);
}

} // namespace intent::gpu
