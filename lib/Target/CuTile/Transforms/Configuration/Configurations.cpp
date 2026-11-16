#include "Intent/Dialect/Block/Transforms/Configuration/TuningProfiles.h"

#include "Configurations.h"
#include "Intent/Dialect/Block/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/Block/Transforms/Configuration/PhysicalParameters.h"
#include "Intent/Dialect/Block/Transforms/Configuration/Resources.h"
#include "Intent/Target/CuTile/Analysis/Program.h"
#include "Intent/Target/CuTile/IR/CuTileOps.h"
#include "Intent/Target/CuTile/Transforms/Configuration/TuningProfiles.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/MathExtras.h"
using namespace mlir;
namespace intent::cutile {
const block::TuningProfileSchema &tuningProfileSchema() {
  static const StringRef columns[] = {
      "ownership_m", "ownership_n", "reduction", "reduction_outer", "scan",
      "traversal_workers", "traversal_group", ctasParameter,
      workerWarpsParameter, occupancyParameter, accessFormParameter,
      loadPolicyParameter};
  static const block::ProviderOption options[] = {
      {ctasParameter, block::ParameterRole::ProviderCTAs},
      {workerWarpsParameter, block::ParameterRole::ProviderWarps},
      {occupancyParameter, block::ParameterRole::ProviderOccupancy},
      {accessFormParameter, block::ParameterRole::ProviderAccessForm},
      {loadPolicyParameter, block::ParameterRole::ProviderLoadPolicy}};
  static const block::TuningProfileSchema schema{"cutile", columns, options};
  return schema;
}

FailureOr<block::ParameterRefAttr> declareProviderParameter(
    func::FuncOp kernel, StringRef name, block::ParameterRole role,
    bool (*isLegal)(int64_t)) {
  auto declaration = block::lookupParameter(kernel, StringAttr::get(kernel.getContext(), name));
  if (!declaration || !declaration.isExtent() || declaration.getRole() != role ||
      declaration.getPhase() != block::ConfigurationBindingPhase::Provider ||
      declaration.getCategory() != block::ParameterCategory::Provider ||
      declaration.getCandidates().asArrayRef().empty() ||
      !llvm::all_of(declaration.getCandidates().asArrayRef(), isLegal))
    return kernel.emitError("cuTile complete profiles require a legal provider declaration for ") << name;
  return declaration.getReference();
}

namespace {

bool legalProviderValue(block::ParameterRole role, int64_t value) {
  switch (role) {
  case block::ParameterRole::ProviderCTAs: return isLegalCTAs(value);
  case block::ParameterRole::ProviderWarps: return isLegalWorkerWarps(value);
  case block::ParameterRole::ProviderOccupancy: return isLegalOccupancy(value);
  case block::ParameterRole::ProviderAccessForm: return isLegalAccessForm(value);
  case block::ParameterRole::ProviderLoadPolicy: return isLegalLoadPolicy(value);
  default: return false;
  }
}

LogicalResult filterProviderConfigurations(func::FuncOp kernel) {
  auto space = block::ParameterSpace::read(kernel);
  if (failed(space)) return failure();
  auto rows = space->configurations(block::ConfigurationStage::Shared);
  if (failed(rows)) return failure();
  SmallVector<DictionaryAttr> accepted;
  for (DictionaryAttr row : *rows) {
    bool legal = true;
    for (block::ParameterAttr declaration : space->declarations()) {
      if (declaration.getPhase() != block::ConfigurationBindingPhase::Provider) continue;
      auto value = row.getAs<IntegerAttr>(declaration.getName());
      legal &= value && legalProviderValue(declaration.getRole(), value.getInt());
    }
    if (legal) accepted.push_back(row);
  }
  if (accepted.empty())
    return kernel.emitError("cuTile complete profiles contain no legal provider configuration");
  for (block::ParameterAttr declaration : space->declarations()) {
    if (declaration.getPhase() != block::ConfigurationBindingPhase::Provider) continue;
    SmallVector<int64_t> candidates;
    for (DictionaryAttr row : accepted) {
      int64_t value = row.getAs<IntegerAttr>(declaration.getName()).getInt();
      if (!llvm::is_contained(candidates, value)) candidates.push_back(value);
    }
    if (failed(block::updateParameter(kernel, declaration.withCandidates(
            DenseI64ArrayAttr::get(kernel.getContext(), candidates)))))
      return failure();
  }
  return block::writeConfigurations(kernel, accepted, block::ConfigurationStage::Shared);
}

llvm::DenseSet<block::ParameterRefAttr> programParameterReferences(func::FuncOp kernel) {
  llvm::DenseSet<block::ParameterRefAttr> references;
  AttrTypeWalker walker;
  walker.addWalk([&](block::ParameterRefAttr reference) { references.insert(reference); });
  kernel.walk([&](Operation *operation) {
    if (auto parameter = dyn_cast<block::ParameterOp>(operation);
        parameter && parameter.getResult().use_empty()) return;
    for (NamedAttribute attribute : operation->getAttrs()) {
      if (operation == kernel.getOperation() &&
          (attribute.getName() == block::parametersAttr ||
           attribute.getName() == block::configurationsAttr)) continue;
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
      kernel->getAttrOfType<block::CapabilitiesAttr>(block::capabilitiesAttr);
  if (features.occupancySensitive &&
      failed(declareProviderParameter(
          kernel, occupancyParameter, block::ParameterRole::ProviderOccupancy,
          isLegalOccupancy)))
    return failure();
  if (features.matrixCompute &&
      failed(declareProviderParameter(
          kernel, ctasParameter,
          block::ParameterRole::ProviderCTAs, isLegalCTAs)))
    return failure();
  if (features.occupancySensitive &&
      failed(declareProviderParameter(
          kernel, workerWarpsParameter,
          block::ParameterRole::ProviderWarps, isLegalWorkerWarps)))
    return failure();

  auto space = block::ParameterSpace::read(kernel);
  if (failed(space)) return failure();
  auto shared = space->configurations(block::ConfigurationStage::Shared);
  if (failed(shared)) return failure();
  auto resident = space->find(block::ParameterRole::ResidentWorkers);
  auto ctas = space->find(block::ParameterRole::ProviderCTAs);
  auto occupancy = space->find(block::ParameterRole::ProviderOccupancy);
  if (!resident) return success();
  if (!ctas || !occupancy)
    return kernel.emitError("cuTile resident binding requires CTA and occupancy columns");
  if (!capabilities || capabilities.getParallelUnits() <= 0)
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
    if (llvm::MulOverflow(capabilities.getParallelUnits() / cluster, capacity, count) || count <= 0)
      return kernel.emitError("cuTile resident capacity is not a positive representable count");
    if (!llvm::is_contained(counts, count)) counts.push_back(count);
    bindings.set(resident.getName(), builder.getI64IntegerAttr(count));
    auto bound = bindings.getDictionary(kernel.getContext());
    if (!llvm::is_contained(projected, bound)) projected.push_back(bound);
  }
  llvm::sort(counts);
  // Allocation envelopes consume the same correlated resident binding as the
  // eventual launch, before native access formation can create workspaces.
  if (failed(block::updateParameter(kernel, resident.withCandidates(
          DenseI64ArrayAttr::get(kernel.getContext(), counts)))))
    return failure();
  return block::writeConfigurations(kernel, projected,
                                   block::ConfigurationStage::Shared);
}

LogicalResult materializeClosedConfigs(func::FuncOp kernel) {
  auto space = block::ParameterSpace::read(kernel);
  if (failed(space))
    return failure();
  for (block::ParameterAttr domain : space->declarations())
    if (!domain.isExtent() ||
        (domain.getPhase() == block::ConfigurationBindingPhase::Provider) !=
            isCuTileProviderRole(domain.getRole()))
      return kernel.emitError("cuTile program contains a foreign provider parameter");
  auto shared = space->configurations(block::ConfigurationStage::Shared);
  if (failed(shared))
    return failure();
  auto features = queryNativeProgramFeatures(kernel);
  auto references = programParameterReferences(kernel);
  SmallVector<Attribute> declarations;
  SmallVector<StringAttr> removed;
  for (block::ParameterAttr declaration : space->declarations()) {
    bool used = true;
    switch (declaration.getRole()) {
    case block::ParameterRole::ProviderCTAs: used = features.matrixCompute; break;
    case block::ParameterRole::ProviderWarps:
    case block::ParameterRole::ProviderOccupancy: used = features.occupancySensitive; break;
    case block::ParameterRole::ProviderAccessForm:
    case block::ParameterRole::ProviderLoadPolicy:
      used = references.contains(declaration.getReference()); break;
    default: break;
    }
    if (used || references.contains(declaration.getReference())) declarations.push_back(declaration);
    else removed.push_back(declaration.getName());
  }
  SmallVector<block::ParameterOp> deadReads;
  kernel.walk([&](block::ParameterOp parameter) {
    if (parameter.getResult().use_empty() &&
        llvm::is_contained(removed, parameter.getDeclaration().getName()))
      deadReads.push_back(parameter);
  });
  for (auto read : deadReads) read.erase();
  kernel->setAttr(block::parametersAttr, ArrayAttr::get(kernel.getContext(), declarations));
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
  auto accepted = block::filterConfigurationRequirements(kernel, encoded, *requirements);
  if (failed(accepted))
    return failure();
  return block::writeConfigurations(kernel, *accepted,
                                 block::ConfigurationStage::Complete, *requirements);
}

LogicalResult finalizeConfigurationRequirements(func::FuncOp kernel) {
  auto parameters = block::ParameterSpace::read(kernel);
  if (failed(parameters))
    return failure();
  auto rows = parameters->configurations(block::ConfigurationStage::Complete);
  auto requirements = collectConfigurationRequirements(kernel);
  if (failed(rows) || failed(requirements))
    return failure();
  auto accepted = block::filterConfigurationRequirements(kernel, *rows, *requirements);
  if (failed(accepted))
    return failure();
  return block::writeConfigurations(kernel, *accepted,
                                 block::ConfigurationStage::Complete, *requirements);
}

} // namespace intent::cutile
