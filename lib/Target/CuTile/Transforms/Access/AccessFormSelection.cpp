#include "AccessFormSelection.h"
#include "Intent/Target/CuTile/IR/CuTileOps.h"
#include "../Configuration/Configurations.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/PhysicalParameters.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;

namespace intent::cutile {

FailureOr<Value> AccessFormSelection::accessFormValue() {
  if (accessForm)
    return accessForm;
  auto parameter = declareProviderParameter(
      kernel, accessFormParameter,
      gpu::ParameterRole::ProviderAccessForm, isLegalAccessForm);
  if (failed(parameter))
    return failure();
  OpBuilder entry(&kernel.front(), kernel.front().begin());
  accessForm = gpu::materializeParameter(entry, kernel.getLoc(), *parameter);
  return accessForm;
}

FailureOr<Value> AccessFormSelection::loadFormCondition() {
  if (preferTileLoads)
    return preferTileLoads;
  FailureOr<Value> form = accessFormValue();
  if (failed(form))
    return failure();
  OpBuilder entry(kernel.getContext());
  entry.setInsertionPointAfter((*form).getDefiningOp());
  Value gather = entry.create<arith::ConstantIndexOp>(kernel.getLoc(),
                                                       gatherAccessForm);
  preferTileLoads = entry.create<gpu::CompareOp>(
      kernel.getLoc(), entry.getI1Type(), *form, gather,
      ComparePredicate::Ne);
  return preferTileLoads;
}

FailureOr<Value> AccessFormSelection::tmaCondition() {
  if (allowNativeTMA)
    return allowNativeTMA;
  FailureOr<Value> form = accessFormValue();
  if (failed(form))
    return failure();
  OpBuilder entry(kernel.getContext());
  entry.setInsertionPointAfter((*form).getDefiningOp());
  Value disabled = entry.create<arith::ConstantIndexOp>(
      kernel.getLoc(), nativeNoTMAForm);
  allowNativeTMA = entry.create<gpu::CompareOp>(
      kernel.getLoc(), entry.getI1Type(), *form, disabled,
      ComparePredicate::Ne);
  return allowNativeTMA;
}

FailureOr<Value> AccessFormSelection::loadPolicyValue() {
  if (!loadPolicy) {
    auto parameter = declareProviderParameter(
        kernel, loadPolicyParameter,
        gpu::ParameterRole::ProviderLoadPolicy, isLegalLoadPolicy);
    if (failed(parameter))
      return failure();
    OpBuilder entry(&kernel.front(), kernel.front().begin());
    loadPolicy = gpu::materializeParameter(entry, kernel.getLoc(), *parameter);
  }
  return loadPolicy;
}

} // namespace intent::cutile
