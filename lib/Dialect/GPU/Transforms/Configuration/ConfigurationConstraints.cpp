#include "ConfigurationPolicy.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <functional>
#include <limits>

using namespace mlir;

namespace intent::gpu::configuration {
namespace {

bool fitsFragmentFootprints(
    ArrayRef<FragmentType> fragments, CapabilitiesAttr capabilities,
    const NamedAttrList &bindings, StringAttr selected = {}, int64_t candidate = 0) {
  auto resolve = [&](PhysicalExprAttr expression) -> std::optional<int64_t> {
    if (expression.getKind() != PhysicalExprKind::Parameter)
      return std::nullopt;
    auto binding = dyn_cast_or_null<IntegerAttr>(
        bindings.get(expression.getParameterReference().getName().getValue()));
    if (!binding) return std::nullopt;
    return expression.getParameterReference().getName() == selected ? candidate : binding.getInt();
  };
  // This bounds individual payloads, not provider register allocation or
  // machine occupancy. ABI-dependent Unknown bounds remain for specialization.
  for (FragmentType fragment : fragments) {
    Type element = fragment.getElementType();
    unsigned bits = element.isIndex() ? 64 : element.getIntOrFloatBitWidth();
    auto bound = checkFragmentFootprint(
        fragment, capabilities.getRegistersPerUnit() - 1,
        std::max(1u, (bits + 31) / 32), resolve);
    if (bound == FootprintBound::Exceeds || bound == FootprintBound::Invalid)
      return false;
  }
  return true;
}

bool fitsCollectiveFootprints(const FragmentResourceAnalysis &resources,
                             func::FuncOp kernel, const NamedAttrList &bindings,
                             StringAttr selected = {}, int64_t candidate = 0) {
  NamedAttrList current(bindings);
  if (selected)
    current.set(selected, IntegerAttr::get(IntegerType::get(kernel.getContext(), 64), candidate));
  auto row = current.getDictionary(kernel.getContext());
  return llvm::all_of(resources.collectiveRequirements(), [&](auto requirement) {
    auto status = evaluateConfigurationRequirement(requirement, row, kernel).status;
    return status != RequirementStatus::Violated &&
           status != RequirementStatus::Invalid;
  });
}

} // namespace

void bindContractionFreeExtents(
    ArrayRef<ContractionFreeExtent> groups, NamedAttrList &bindings,
    const TuningProfile &profile, Builder &builder) {
  for (const ContractionFreeExtent &group : groups) {
    int64_t budget = requestedValue(profile, group.role);
    auto fits = [&](ParameterAttr selected, int64_t candidate) {
      std::function<std::optional<int64_t>(PhysicalExprAttr)> evaluate =
          [&](PhysicalExprAttr extent) -> std::optional<int64_t> {
        auto kind = extent.getKind();
        if (kind == PhysicalExprKind::Constant)
          return extent.getValue() <= budget
                     ? std::optional<int64_t>(extent.getValue()) : std::nullopt;
        if (kind == PhysicalExprKind::Parameter) {
          int64_t value = selected && extent.getParameterReference().getName() == selected.getName()
                              ? candidate
                              : cast<IntegerAttr>(bindings.get(extent.getParameterReference().getName().getValue())).getInt();
          return value <= budget ? std::optional<int64_t>(value) : std::nullopt;
        }
        int64_t product = 1;
        for (Attribute operand : extent.getOperands()) {
          auto factor = evaluate(cast<PhysicalExprAttr>(operand));
          if (!factor || product > budget / *factor) return std::nullopt;
          product *= *factor;
        }
        return product;
      };
      int64_t product = 1;
      for (PhysicalExprAttr extent : group.extents) {
        auto factor = evaluate(extent);
        if (!factor || product > budget / *factor) return false;
        product *= *factor;
      }
      return true;
    };
    // Keep the innermost coordinates wide; the profile targets their combined
    // M/N extent, not each independently tiled factor of that extent.
    for (ParameterAttr parameter : group.parameters) {
      if (fits({}, 0)) break;
      auto schema = parameter;
      int64_t current = cast<IntegerAttr>(bindings.get(schema.getName().getValue())).getInt();
      auto candidates = schema.getCandidates().asArrayRef();
      int64_t selected = *std::min_element(candidates.begin(), candidates.end());
      for (int64_t candidate : candidates)
        if (candidate <= current && candidate > selected && fits(parameter, candidate))
          selected = candidate;
      bindings.set(schema.getName(), builder.getI64IntegerAttr(selected));
    }
  }
}

LogicalResult bindTraversalFragmentFootprints(
    func::FuncOp kernel, ArrayRef<ParameterAttr> parameters,
    const FragmentResourceAnalysis &resources, NamedAttrList &bindings,
    Builder &builder) {
  auto capabilities = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  if (!capabilities || capabilities.getRegistersPerUnit() <= 0)
    return success();
  SmallVector<ArrayRef<FragmentType>> budgeted;
  for (ParameterAttr parameter : parameters) {
    auto schema = parameter;
    auto category = schema.getCategory();
    auto role = schema.getRole();
    bool physicalTraversal =
        (category == ParameterCategory::Reduction &&
         (role == ParameterRole::Reduction ||
          role == ParameterRole::ReductionInner ||
          role == ParameterRole::ReductionOuter)) ||
        (role == ParameterRole::ScanChunk &&
         (category == ParameterCategory::Scan ||
          category == ParameterCategory::RegionReduction ||
          category == ParameterCategory::RegionContraction)) ||
        isPointwiseTraversalParameter(parameter);
    if (!physicalTraversal)
      continue;
    auto fragments = resources.materializedTypesUsing(schema.getName());
    auto fits = [&](int64_t candidate) {
      return fitsFragmentFootprints(
          fragments, capabilities, bindings, schema.getName(), candidate) &&
          fitsCollectiveFootprints(resources, kernel, bindings, schema.getName(), candidate);
    };
    int64_t requested = cast<IntegerAttr>(
        bindings.get(schema.getName().getValue())).getInt();
    std::optional<int64_t> selected;
    for (int64_t candidate : schema.getCandidates().asArrayRef())
      if (candidate <= requested && (!selected || candidate > *selected) &&
          fits(candidate))
        selected = candidate;
    // Other traversal axes may still exceed their budget. Reach this axis's
    // legal minimum before binding those axes, then validate the complete tuple.
    if (!selected)
      selected = *llvm::min_element(schema.getCandidates().asArrayRef());
    bindings.set(schema.getName(), builder.getI64IntegerAttr(*selected));
    budgeted.push_back(fragments);
  }
  // Every axis has now been projected. Recheck those same payload sets against
  // the complete tuple, since another traversal axis may have changed it.
  return success(fitsCollectiveFootprints(resources, kernel, bindings) &&
                 llvm::all_of(budgeted, [&](ArrayRef<FragmentType> fragments) {
    return fitsFragmentFootprints(fragments, capabilities, bindings);
  }));
}

} // namespace intent::gpu::configuration
