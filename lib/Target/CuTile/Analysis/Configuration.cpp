#include "Intent/Target/CuTile/Analysis/Configuration.h"
#include "Intent/Target/CuTile/IR/CuTileOps.h"
#include "Intent/Target/CuTile/IR/Configuration.h"
#include "Intent/Dialect/Block/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/Block/Analysis/Resources.h"
#include "Intent/Dialect/Block/IR/Program.h"

using namespace mlir;
namespace intent::cutile {

FailureOr<SmallVector<block::ConfigurationRequirementAttr>>
collectConfigurationRequirements(func::FuncOp kernel) {
  auto parameters = block::ParameterSpace::read(kernel);
  if (failed(parameters))
    return failure();
  SmallVector<Operation *> reductions;
  kernel.walk([&](ReduceOp reduce) {
    reductions.push_back(reduce);
  });
  SmallVector<Value> matrixInputs;
  kernel.walk([&](MMAOp matrix) {
    matrixInputs.push_back(matrix.getLhs());
    matrixInputs.push_back(matrix.getRhs());
  });
  auto requirements = block::collectReductionRequirements(
      kernel, reductions, block::ReductionRequirementScope::AllCandidates, matrixInputs);
  const block::FragmentResourceAnalysis resources(kernel);
  llvm::append_range(requirements, block::collectPointwiseRequirements(kernel, resources, matrixInputs));
  auto resident = parameters->find(block::ParameterRole::ResidentWorkers);
  auto ctas = parameters->lookup(ctasParameter);
  auto occupancy = parameters->lookup(occupancyParameter);
  if (resident && ctas && occupancy) {
    auto capabilities =
        kernel->getAttrOfType<block::CapabilitiesAttr>(block::capabilitiesAttr);
    if (!capabilities || capabilities.getParallelUnits() <= 0)
      return kernel.emitError("cuTile resident binding requires a positive compute-unit count"),
             failure();
    Builder builder(kernel.getContext());
    auto expression = [&](block::PhysicalExprKind kind, int64_t value,
                          Attribute symbol, ArrayRef<Attribute> operands) {
      return block::PhysicalExprAttr::get(kernel.getContext(), kind, value,
                                       symbol, builder.getArrayAttr(operands));
    };
    auto parameter = [&](block::ParameterAttr declaration) {
      return expression(block::PhysicalExprKind::Parameter, 0,
                        declaration.getReference(), {});
    };
    auto computeUnits = expression(block::PhysicalExprKind::Constant,
        capabilities.getParallelUnits(), builder.getStringAttr(""), {});
    auto clusters = expression(block::PhysicalExprKind::FloorDiv, 0,
        builder.getStringAttr(""), {computeUnits, parameter(ctas)});
    auto capacity = expression(block::PhysicalExprKind::Multiply, 0,
        builder.getStringAttr(""), {clusters, parameter(occupancy)});
    requirements.push_back(block::ConfigurationRequirementAttr::get(
        kernel.getContext(), block::ConfigurationRequirementKind::Legality,
        block::ConfigurationRequirementMetric::ResidentWorkers,
        block::ConfigurationRequirementPredicate::Equal, parameter(resident),
        capacity, block::ParameterRefAttr(),
        builder.getStringAttr("cuTile resident workers must match the CTA and occupancy binding"), true));
  }
  return requirements;
}

LogicalResult verifyClosedConfigs(func::FuncOp kernel) {
  auto requirements = collectConfigurationRequirements(kernel);
  if (failed(requirements))
    return failure();
  return block::verifyConfigurationRequirements(kernel, *requirements);
}

} // namespace intent::cutile
