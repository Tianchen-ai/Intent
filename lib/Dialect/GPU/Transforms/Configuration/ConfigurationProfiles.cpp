#include "ConfigurationPolicy.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "llvm/ADT/DenseSet.h"
#include <algorithm>
#include <limits>

using namespace mlir;

namespace intent::gpu::configuration {
namespace {

unsigned priority(TuningClass kind) {
  switch (kind) {
  case TuningClass::PersistentContraction: return 13;
  case TuningClass::RegionContraction: return 12;
  case TuningClass::RegionReduction: return 11;
  case TuningClass::Contraction: return 10;
  case TuningClass::Histogram: return 9;
  case TuningClass::MultiAxisReduction: return 8;
  case TuningClass::StatefulReduction: return 7;
  case TuningClass::Scan: return 6;
  case TuningClass::Reduction: return 5;
  case TuningClass::OnlineMoment: return 4;
  case TuningClass::PointwiseReduction: return 3;
  case TuningClass::Pointwise: return 2;
  case TuningClass::Execution: return 1;
  }
  llvm_unreachable("unknown tuning class");
}

StringRef familyFor(func::FuncOp kernel, const ConfigurationFacts &facts) {
  TuningClass kind = TuningClass::Execution;
  unsigned width = 0;
  for (ParameterAttr parameter : facts.parameters) {
    const auto &classification = facts.classifications.find(parameter)->second;
    if (priority(classification.kind) > priority(kind)) {
      kind = classification.kind;
      width = classification.width;
    } else if (classification.kind == kind) {
      width = std::max(width, classification.width);
    }
  }
  if (facts.parameters.empty())
    kind = TuningClass::Pointwise;
  auto capabilities = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  bool matrix = capabilities && capabilities.getMatrixUnits();
  bool narrow = width <= 16;
  if (facts.multipleRegionMatrixAccumulators &&
      (kind == TuningClass::Contraction || kind == TuningClass::RegionContraction))
    return "region_contraction_multi_accumulator";
  switch (kind) {
  case TuningClass::PersistentContraction:
    return matrix && narrow ? "persistent_contraction_narrow" : "persistent_contraction";
  case TuningClass::Contraction:
    return matrix && narrow ? "contraction_narrow" : "contraction";
  case TuningClass::RegionContraction:
    {
      bool found = false, allF32 = true;
      kernel.walk([&](ContractOp contract) {
        found = true;
        allF32 &= cast<FragmentType>(contract.getLhs().getType()).getElementType().isF32() &&
                  cast<FragmentType>(contract.getRhs().getType()).getElementType().isF32();
      });
      if (found && allF32) return "region_contraction_f32";
    }
    return facts.smallRegionRows ? "region_contraction_small_rows" : "region_contraction";
  case TuningClass::RegionReduction: return "region_reduction";
  case TuningClass::Scan: return "scan";
  case TuningClass::MultiAxisReduction: return "multi_axis_reduction";
  case TuningClass::Histogram: return "histogram";
  case TuningClass::Reduction: return "reduction";
  case TuningClass::StatefulReduction: return "stateful_reduction";
  case TuningClass::Execution: return "execution";
  case TuningClass::OnlineMoment: return "online_moment";
  case TuningClass::PointwiseReduction:
    return facts.hasTwoAxisPointwiseOwnership ? "pointwise_reduction_two_axis" : "pointwise_reduction";
  case TuningClass::Pointwise:
    if (facts.hasTwoAxisPointwiseOwnership && facts.hasFixedPointwiseLocal)
      return facts.pointwiseOnlyProgram ? "pointwise_only_fixed_local" : "pointwise_fixed_local";
    if (facts.hasTwoAxisPointwiseOwnership)
      return narrow ? "pointwise_two_axis_narrow" : "pointwise_two_axis";
    return narrow ? "pointwise_narrow" : "pointwise";
  }
  llvm_unreachable("unknown tuning class");
}

} // namespace

int64_t requestedValue(const TuningProfile &profile, ParameterRole role) {
  switch (role) {
  case ParameterRole::OwnershipM: return profile.ownershipM;
  case ParameterRole::OwnershipN: return profile.ownershipN;
  case ParameterRole::Reduction:
  case ParameterRole::ReductionInner: return profile.reduction;
  case ParameterRole::ReductionOuter: return profile.reductionOuter;
  case ParameterRole::ScanChunk: return profile.scan;
  case ParameterRole::TraversalWorkers: return profile.traversalWorkers;
  case ParameterRole::TraversalGroup: return profile.traversalGroup;
  case ParameterRole::ResidentWorkers: return std::numeric_limits<int64_t>::max();
  case ParameterRole::FullCoverage:
  case ParameterRole::ProviderWarps:
  case ParameterRole::ProviderStages:
  case ParameterRole::ProviderCTAs:
  case ParameterRole::ProviderAccessForm:
  case ParameterRole::ProviderOccupancy:
  case ParameterRole::ProviderLoadPolicy:
    llvm_unreachable("non-static parameter entered shared tuning table");
  }
  llvm_unreachable("unknown physical parameter role");
}

int64_t selectCandidate(ArrayRef<int64_t> candidates, int64_t requested) {
  int64_t selected = *std::min_element(candidates.begin(), candidates.end());
  for (int64_t candidate : candidates) {
    if (candidate == requested) return candidate;
    if (candidate <= requested && candidate > selected) selected = candidate;
  }
  return selected;
}

LogicalResult projectConfigurationProfiles(
    func::FuncOp kernel, const ConfigurationFacts &facts,
    const TuningProfiles &tables, ProfileBindingConsumer consume) {
  auto table = tables.table();
  if (failed(table)) return failure();
  StringRef family = familyFor(kernel, facts);
  auto rows = table->getFamilies().getAs<ArrayAttr>(family);
  if (!rows)
    return kernel.emitError("missing declared complete tuning family '") << family << "'";
  Builder builder(kernel.getContext());
  llvm::SmallDenseSet<ParameterAttr> selectedRows;
  for (const auto &group : facts.rowGroups)
    selectedRows.insert(group.second.front());
  for (Attribute attribute : rows) {
    auto row = cast<DenseI64ArrayAttr>(attribute).asArrayRef();
    TuningProfile profile{row[0], row[1], row[2], row[3], row[4], row[5], row[6]};
    NamedAttrList bindings;
    for (ParameterAttr parameter : facts.parameters) {
      int64_t requested = requestedValue(profile, parameter.getRole());
      const auto &classification = facts.classifications.find(parameter)->second;
      // Independent rows retained by a reduction use the row budget. Keep
      // contraction M/N roles and a separate pointwise traversal unchanged.
      if (parameter.getCategory() == ParameterCategory::Pointwise &&
          classification.reductionRow)
        requested = profile.ownershipM;
      for (const CorrelatedProfileParameters &relation : facts.correlatedProfiles) {
        if (parameter == relation.pointwise) requested = profile.ownershipM;
        if (parameter == relation.reduction) requested = profile.ownershipN;
      }
      if (auto local = facts.pointwiseLocalMultiplicity.find(parameter);
          local != facts.pointwiseLocalMultiplicity.end())
        requested = std::max<int64_t>(1, requested / local->second);
      if (!selectedRows.contains(parameter) &&
          llvm::is_contained(facts.pointwiseRowAxes, parameter))
        requested = 1;
      bindings.set(parameter.getName(), builder.getI64IntegerAttr(
          selectCandidate(parameter.getCandidates().asArrayRef(), requested)));
    }
    for (unsigned column = 7; column < table->getColumns().size(); ++column) {
      auto name = cast<StringAttr>(table->getColumns()[column]);
      auto parameter = queryParameterBySymbol(kernel, name);
      if (failed(parameter) || parameter->getPhase() != ConfigurationBindingPhase::Provider ||
          !llvm::is_contained(parameter->getCandidates().asArrayRef(), row[column]))
        return kernel.emitError("complete tuning row has no declared provider binding for ") << name;
      bindings.set(name, builder.getI64IntegerAttr(row[column]));
    }
    consume(bindings, profile);
  }
  return success();
}

} // namespace intent::gpu::configuration
