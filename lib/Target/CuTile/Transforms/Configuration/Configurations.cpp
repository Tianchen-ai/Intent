#include "Intent/Dialect/GPU/Transforms/Configuration/TuningProfiles.h"

#include "Configurations.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/Resources.h"
#include "Intent/Target/CuTile/Analysis/Program.h"
#include "Intent/Target/CuTile/IR/CuTileOps.h"
#include "Intent/Target/CuTile/Transforms/Configuration/TuningProfiles.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/MathExtras.h"
using namespace mlir;
namespace intent::cutile {
const gpu::TuningProfileSchema &tuningProfileSchema() {
  static const StringRef columns[] = {
      "ownership_m", "ownership_n", "reduction", "reduction_outer", "scan",
      "traversal_workers", "traversal_group", ctasParameter,
      workerWarpsParameter, occupancyParameter, accessFormParameter,
      loadPolicyParameter};
  static const gpu::ProviderOption options[] = {
      {ctasParameter, gpu::ParameterRole::ProviderCTAs},
      {workerWarpsParameter, gpu::ParameterRole::ProviderWarps},
      {occupancyParameter, gpu::ParameterRole::ProviderOccupancy},
      {accessFormParameter, gpu::ParameterRole::ProviderAccessForm},
      {loadPolicyParameter, gpu::ParameterRole::ProviderLoadPolicy}};
  static const gpu::TuningProfileSchema schema{"cutile", columns, options};
  return schema;
}

FailureOr<gpu::ParameterRefAttr> declareProviderParameter(
    func::FuncOp kernel, StringRef name, gpu::ParameterRole role,
    bool (*isLegal)(int64_t)) {
  auto declaration = gpu::lookupParameter(kernel, StringAttr::get(kernel.getContext(), name));
  if (!declaration || !declaration.isExtent() || declaration.getRole() != role ||
      declaration.getPhase() != gpu::ConfigurationBindingPhase::Provider ||
      declaration.getCategory() != gpu::ParameterCategory::Provider ||
      declaration.getCandidates().asArrayRef().empty() ||
      !llvm::all_of(declaration.getCandidates().asArrayRef(), isLegal))
    return kernel.emitError("cuTile complete profiles require a legal provider declaration for ") << name;
  return declaration.getReference();
}

namespace {

bool legalProviderValue(gpu::ParameterRole role, int64_t value) {
  switch (role) {
  case gpu::ParameterRole::ProviderCTAs: return isLegalCTAs(value);
  case gpu::ParameterRole::ProviderWarps: return isLegalWorkerWarps(value);
  case gpu::ParameterRole::ProviderOccupancy: return isLegalOccupancy(value);
  case gpu::ParameterRole::ProviderAccessForm: return isLegalAccessForm(value);
  case gpu::ParameterRole::ProviderLoadPolicy: return isLegalLoadPolicy(value);
  default: return false;
  }
}

LogicalResult filterProviderConfigurations(func::FuncOp kernel) {
  auto space = gpu::ParameterSpace::read(kernel);
  if (failed(space)) return failure();
  auto rows = space->configurations(gpu::ConfigurationStage::Shared);
  if (failed(rows)) return failure();
  SmallVector<DictionaryAttr> accepted;
  for (DictionaryAttr row : *rows) {
    bool legal = true;
    for (gpu::ParameterAttr declaration : space->declarations()) {
      if (declaration.getPhase() != gpu::ConfigurationBindingPhase::Provider) continue;
      auto value = row.getAs<IntegerAttr>(declaration.getName());
      legal &= value && legalProviderValue(declaration.getRole(), value.getInt());
    }
    if (legal) accepted.push_back(row);
  }
  if (accepted.empty())
    return kernel.emitError("cuTile complete profiles contain no legal provider configuration");
  for (gpu::ParameterAttr declaration : space->declarations()) {
    if (declaration.getPhase() != gpu::ConfigurationBindingPhase::Provider) continue;
    SmallVector<int64_t> candidates;
    for (DictionaryAttr row : accepted) {
      int64_t value = row.getAs<IntegerAttr>(declaration.getName()).getInt();
      if (!llvm::is_contained(candidates, value)) candidates.push_back(value);
    }
    if (failed(gpu::updateParameter(kernel, declaration.withCandidates(
            DenseI64ArrayAttr::get(kernel.getContext(), candidates)))))
      return failure();
  }
  return gpu::writeConfigurations(kernel, accepted, gpu::ConfigurationStage::Shared);
}

llvm::DenseSet<gpu::ParameterRefAttr> programParameterReferences(func::FuncOp kernel) {
  llvm::DenseSet<gpu::ParameterRefAttr> references;
  AttrTypeWalker walker;
  walker.addWalk([&](gpu::ParameterRefAttr reference) { references.insert(reference); });
  kernel.walk([&](Operation *operation) {
    if (auto parameter = dyn_cast<gpu::ParameterOp>(operation);
        parameter && parameter.getResult().use_empty()) return;
    for (NamedAttribute attribute : operation->getAttrs()) {
      if (operation == kernel.getOperation() &&
          (attribute.getName() == gpu::parametersAttr ||
           attribute.getName() == gpu::configurationsAttr)) continue;
      walker.walk(attribute.getValue());
    }
    for (Type type : operation->getResultTypes()) walker.walk(type);
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments()) walker.walk(argument.getType());
  });
  return references;
}

} // namespace

LogicalResult prepareLaunchConfigurations(func::FuncOp kernel) {
  if (failed(filterProviderConfigurations(kernel))) return failure();
  NativeProgramFeatures features = queryNativeProgramFeatures(kernel);
  auto capabilities =
      kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
  if (features.occupancySensitive &&
      failed(declareProviderParameter(
          kernel, occupancyParameter, gpu::ParameterRole::ProviderOccupancy,
          isLegalOccupancy)))
    return failure();
  if (features.matrixCompute &&
      failed(declareProviderParameter(
          kernel, ctasParameter,
          gpu::ParameterRole::ProviderCTAs, isLegalCTAs)))
    return failure();
  if (features.occupancySensitive &&
      failed(declareProviderParameter(
          kernel, workerWarpsParameter,
          gpu::ParameterRole::ProviderWarps, isLegalWorkerWarps)))
    return failure();

  auto space = gpu::ParameterSpace::read(kernel);
  if (failed(space)) return failure();
  auto shared = space->configurations(gpu::ConfigurationStage::Shared);
  if (failed(shared)) return failure();
  auto resident = space->find(gpu::ParameterRole::ResidentWorkers);
  auto ctas = space->find(gpu::ParameterRole::ProviderCTAs);
  auto occupancy = space->find(gpu::ParameterRole::ProviderOccupancy);
  if (!resident) return success();
  if (!ctas || !occupancy)
    return kernel.emitError("cuTile resident binding requires CTA and occupancy columns");
  if (!capabilities || capabilities.getComputeUnits() <= 0)
    return kernel.emitError("cuTile resident binding requires a positive compute-unit count");

  Builder builder(kernel.getContext());
  SmallVector<int64_t> counts;
  SmallVector<DictionaryAttr> projected;
  for (DictionaryAttr row : *shared) {
    NamedAttrList bindings(row);
    int64_t cluster = cast<IntegerAttr>(bindings.get(ctas.getName())).getInt();
    int64_t capacity = cast<IntegerAttr>(bindings.get(occupancy.getName())).getInt();
    if (!isLegalCTAs(cluster) || !isLegalOccupancy(capacity))
      return kernel.emitError("cuTile resident binding requires legal CTA and occupancy options");
    int64_t count;
    if (llvm::MulOverflow(capabilities.getComputeUnits() / cluster, capacity, count) || count <= 0)
      return kernel.emitError("cuTile resident capacity is not a positive representable count");
    if (!llvm::is_contained(counts, count)) counts.push_back(count);
    bindings.set(resident.getName(), builder.getI64IntegerAttr(count));
    auto bound = bindings.getDictionary(kernel.getContext());
    if (!llvm::is_contained(projected, bound)) projected.push_back(bound);
  }
  llvm::sort(counts);
  // Allocation envelopes consume the same correlated resident binding as the
  // eventual launch, before native access formation can create workspaces.
  if (failed(gpu::updateParameter(kernel, resident.withCandidates(
          DenseI64ArrayAttr::get(kernel.getContext(), counts)))))
    return failure();
  return gpu::writeConfigurations(kernel, projected,
                                   gpu::ConfigurationStage::Shared);
}

LogicalResult materializeClosedConfigs(func::FuncOp kernel) {
  auto space = gpu::ParameterSpace::read(kernel);
  if (failed(space))
    return failure();
  for (gpu::ParameterAttr domain : space->declarations())
    if (!domain.isExtent() ||
        (domain.getPhase() == gpu::ConfigurationBindingPhase::Provider) !=
            isCuTileProviderRole(domain.getRole()))
      return kernel.emitError("cuTile program contains a foreign provider parameter");
  auto shared = space->configurations(gpu::ConfigurationStage::Shared);
  if (failed(shared))
    return failure();
  auto features = queryNativeProgramFeatures(kernel);
  auto references = programParameterReferences(kernel);
  SmallVector<Attribute> declarations;
  SmallVector<StringAttr> removed;
  for (gpu::ParameterAttr declaration : space->declarations()) {
    bool used = true;
    switch (declaration.getRole()) {
    case gpu::ParameterRole::ProviderCTAs: used = features.matrixCompute; break;
    case gpu::ParameterRole::ProviderWarps:
    case gpu::ParameterRole::ProviderOccupancy: used = features.occupancySensitive; break;
    case gpu::ParameterRole::ProviderAccessForm:
    case gpu::ParameterRole::ProviderLoadPolicy:
      used = references.contains(declaration.getReference()); break;
    default: break;
    }
    if (used || references.contains(declaration.getReference())) declarations.push_back(declaration);
    else removed.push_back(declaration.getName());
  }
  SmallVector<gpu::ParameterOp> deadReads;
  kernel.walk([&](gpu::ParameterOp parameter) {
    if (parameter.getResult().use_empty() &&
        llvm::is_contained(removed, parameter.getDeclaration().getName()))
      deadReads.push_back(parameter);
  });
  for (auto read : deadReads) read.erase();
  kernel->setAttr(gpu::parametersAttr, ArrayAttr::get(kernel.getContext(), declarations));
  SmallVector<DictionaryAttr> encoded;
  for (DictionaryAttr row : *shared) {
    NamedAttrList bindings(row);
    for (StringAttr name : removed) bindings.erase(name);
    auto candidate = bindings.getDictionary(kernel.getContext());
    if (!llvm::is_contained(encoded, candidate)) encoded.push_back(candidate);
  }
  if (encoded.empty())
    return kernel.emitError("cuTile legalization produced no provider config");
  auto requirements = collectConfigurationRequirements(kernel);
  if (failed(requirements))
    return failure();
  auto accepted = gpu::filterConfigurationRequirements(kernel, encoded, *requirements);
  if (failed(accepted))
    return failure();
  return gpu::writeConfigurations(kernel, *accepted,
                                 gpu::ConfigurationStage::Complete, *requirements);
}

LogicalResult finalizeConfigurationRequirements(func::FuncOp kernel) {
  auto parameters = gpu::ParameterSpace::read(kernel);
  if (failed(parameters))
    return failure();
  auto rows = parameters->configurations(gpu::ConfigurationStage::Complete);
  auto requirements = collectConfigurationRequirements(kernel);
  if (failed(rows) || failed(requirements))
    return failure();
  auto accepted = gpu::filterConfigurationRequirements(kernel, *rows, *requirements);
  if (failed(accepted))
    return failure();
  return gpu::writeConfigurations(kernel, *accepted,
                                 gpu::ConfigurationStage::Complete, *requirements);
}

} // namespace intent::cutile
