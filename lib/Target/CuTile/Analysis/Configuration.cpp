#include "Intent/Target/CuTile/Analysis/Configuration.h"
#include "Intent/Target/CuTile/IR/CuTileOps.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Analysis/Resources.h"
#include "Intent/Dialect/GPU/IR/Program.h"

using namespace mlir;
namespace intent::cutile {

FailureOr<SmallVector<gpu::ConfigurationRequirementAttr>>
collectConfigurationRequirements(func::FuncOp kernel) {
  auto parameters = gpu::ParameterSpace::read(kernel);
  if (failed(parameters))
    return failure();
  SmallVector<Operation *> reductions;
  kernel.walk([&](ReduceOp reduce) {
    reductions.push_back(reduce);
  });
  auto requirements = gpu::collectReductionRequirements(
      kernel, reductions, gpu::ReductionRequirementScope::AllCandidates);
  const gpu::FragmentResourceAnalysis resources(kernel);
  llvm::append_range(requirements, gpu::collectPointwiseRequirements(kernel, resources));
  auto resident = parameters->find(gpu::ParameterRole::ResidentWorkers);
  auto ctas = parameters->find(gpu::ParameterRole::ProviderCTAs);
  auto occupancy = parameters->find(gpu::ParameterRole::ProviderOccupancy);
  if (resident && ctas && occupancy) {
    auto capabilities =
        kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
    if (!capabilities || capabilities.getComputeUnits() <= 0)
      return kernel.emitError("cuTile resident binding requires a positive compute-unit count"),
             failure();
    Builder builder(kernel.getContext());
    auto expression = [&](gpu::PhysicalExprKind kind, int64_t value,
                          Attribute symbol, ArrayRef<Attribute> operands) {
      return gpu::PhysicalExprAttr::get(kernel.getContext(), kind, value,
                                       symbol, builder.getArrayAttr(operands));
    };
    auto parameter = [&](gpu::ParameterAttr declaration) {
      return expression(gpu::PhysicalExprKind::Parameter, 0,
                        declaration.getReference(), {});
    };
    auto computeUnits = expression(gpu::PhysicalExprKind::Constant,
        capabilities.getComputeUnits(), builder.getStringAttr(""), {});
    auto clusters = expression(gpu::PhysicalExprKind::FloorDiv, 0,
        builder.getStringAttr(""), {computeUnits, parameter(ctas)});
    auto capacity = expression(gpu::PhysicalExprKind::Multiply, 0,
        builder.getStringAttr(""), {clusters, parameter(occupancy)});
    requirements.push_back(gpu::ConfigurationRequirementAttr::get(
        kernel.getContext(), gpu::ConfigurationRequirementKind::Legality,
        gpu::ConfigurationRequirementMetric::ResidentWorkers,
        gpu::ConfigurationRequirementPredicate::Equal, parameter(resident),
        capacity, gpu::ParameterRefAttr(),
        builder.getStringAttr("cuTile resident workers must match the CTA and occupancy binding")));
  }
  return requirements;
}

LogicalResult verifyClosedConfigs(func::FuncOp kernel) {
  auto requirements = collectConfigurationRequirements(kernel);
  if (failed(requirements))
    return failure();
  return gpu::verifyConfigurationRequirements(kernel, *requirements);
}

} // namespace intent::cutile
