#include "Accesses.h"
#include "AccessFormSelection.h"
#include "Coordinates.h"
#include "Bounds.h"
#include "TileIndices.h"
#include "Intent/Dialect/GPU/Analysis/IndexRelations.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/IR/FragmentOpInterface.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/PhysicalParameters.h"
#include "Intent/Target/CuTile/Analysis/IndexBounds.h"
#include "Intent/Target/CuTile/IR/CuTileOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

namespace intent::cutile {

namespace {

FailureOr<Value> scalarFill(Operation *owner, Value fill) {
  Value scalar = uniformScalarFill(fill);
  if (!fill || scalar)
    return scalar;
  return owner->emitOpError(
      "cuTile gather padding must be an explicit scalar or splat");
}

bool isZeroFill(Value value) {
  Value scalar = uniformScalarFill(value);
  if (auto cast = scalar ? scalar.getDefiningOp<gpu::CastOp>() : gpu::CastOp())
    if (cast.getValue().getType().isIntOrIndex() &&
        cast.getResult().getType().isIntOrIndex())
      return isZeroFill(cast.getValue());
  Attribute constant = getCompileTimeScalar(scalar ? scalar : value);
  if (!constant)
    return false;
  if (auto integer = dyn_cast<IntegerAttr>(constant))
    return integer.getValue().isZero();
  if (auto floating = dyn_cast<FloatAttr>(constant))
    return floating.getValue().isZero() && !floating.getValue().isNegative();
  return false;
}

} // namespace

LogicalResult formNativeLoads(func::FuncOp kernel,
    ArrayRef<gpu::LoadOp> loads, bool matrixCompute,
    AccessFormSelection &forms, NativeFormRewriter &rewriter) {
  for (gpu::LoadOp load : loads) {
    auto access = cast<gpu::AccessOpInterface>(load.getOperation());
    gpu::PhysicalProgramAnalysis analysis(kernel);
    auto view = dyn_cast<gpu::ViewType>(access.getAccessResource().getType());
    if (!view)
      return load.emitOpError(
          "cuTile native load requires an external view resource");
    OpBuilder builder(load);
    if (!isa<gpu::FragmentType>(access.getAccessResult().getType())) {
      FailureOr<SmallVector<Value>> indices = orderedCoordinates(access);
      FailureOr<Value> fill = scalarFill(load, access.getAccessFill());
      if (failed(indices) || failed(fill) ||
          llvm::any_of(*indices, [](Value value) {
            return isa<gpu::FragmentType>(value.getType());
          }) ||
          (access.getAccessValidity() && isa<gpu::FragmentType>(access.getAccessValidity().getType())))
        return load.emitOpError(
            "cuTile scalar load requires scalar indices and validity");
      bool inBounds = scalarCoordinatesInView(*indices, view, kernel);
      Value validity = access.getAccessValidity();
      Value padding = *fill;
      gpu::PhysicalAccessBoundaryFact boundary =
          analysis.boundaryValidity(load);
      if (inBounds && boundary.isExact()) {
        validity = {};
        padding = {};
      }
      auto replacement = builder.create<ScalarLoadOp>(
          load.getLoc(), access.getAccessResult().getType(), access.getAccessResource(), *indices,
          validity, padding,
          inBounds ? builder.getUnitAttr() : UnitAttr());
      if (Attribute origin = load->getAttr(gpu::originAttr))
        replacement->setAttr(gpu::originAttr, origin);
      rewriter.replace(load, ValueRange{replacement.getResult()});
      continue;
    }
    auto result = cast<gpu::FragmentType>(access.getAccessResult().getType());
    const auto accessBounds = analysis.accessBounds(load);
    bool activeInBounds = accessBounds.isExact();
    // Vector inputs stay register loads, not cluster TMA payloads.
    bool vectorInput = matrixCompute && result.getShape().size() == 1;
    gpu::PhysicalAccessBoundaryFact boundary =
        analysis.boundaryValidity(load, /*allowRangeGuards=*/true);
    FailureOr<NativeTileAccessPlan> plan = analyzeNativeTileAccess(access, kernel, accessBounds);
    FailureOr<MaterializedTileIndices> indices = failure();
    FailureOr<Value> originGuard = failure();
    if (!vectorInput && succeeded(plan) && boundary.isExact() &&
        (!access.getAccessFill() || isZeroFill(access.getAccessFill())))
      indices = materializeTileIndices(builder, load, *plan,
                                       /*allowDynamicAlignment=*/true);
    if (succeeded(indices))
      originGuard = materializeTileOriginGuard(
          builder, load, access.getAccessResource(), *plan, boundary);
    Value rangeGuard = succeeded(indices)
                           ? materializeFullRangeGuard(builder, load.getLoc(),
                                                       boundary)
                           : Value();
    bool guardedNative = succeeded(originGuard) && *originGuard;
    bool native = succeeded(indices) && succeeded(originGuard) &&
                  (!guardedNative ||
                   (access.getAccessFill() && access.getAccessFill().getType() == result));
    if (rangeGuard && guardedNative)
      rangeGuard = builder.create<gpu::BinaryOp>(
          load.getLoc(), builder.getI1Type(), rangeGuard, *originGuard,
          BinaryOperator::LogicalAnd);
    FailureOr<Value> allowTMA = failure();
    if (native) {
      allowTMA = forms.tmaCondition();
      if (failed(allowTMA))
        return failure();
    }
    Value replacementResult;
    SmallVector<Operation *> createdOperations;
    Value loopLatency;
    if (matrixCompute && load->getParentOfType<scf::ForOp>()) {
      auto selected = forms.loadPolicyValue();
      if (failed(selected))
        return failure();
      loopLatency = *selected;
    }
    auto emitNativeLoad = [&](OpBuilder &nested) {
      Value fullTiles = materializeFullTileCondition(
          nested, load.getLoc(), access.getAccessResource(), plan->resourceType);
      auto tile = nested.create<TileLoadOp>(
          load.getLoc(), plan->resourceType, access.getAccessResource(), *allowTMA,
          indices->values, loopLatency, fullTiles);
      Value value = tile.getResult();
      createdOperations.push_back(tile);
      if (plan->resourceToPacked) {
        auto reshape = nested.create<gpu::ReshapeOp>(
            load.getLoc(), plan->packedType, value,
            plan->resourceToPacked);
        value = reshape.getResult();
        createdOperations.push_back(reshape);
      }
      if (!isIdentityPermutation(plan->toComputation)) {
        auto transpose = nested.create<gpu::TransposeOp>(
            load.getLoc(), result, value, plan->toComputation);
        value = transpose.getResult();
        createdOperations.push_back(transpose);
      }
      return value;
    };
    auto emitGatherLoad = [&](OpBuilder &nested) -> FailureOr<Value> {
      Value fill = uniformScalarFill(access.getAccessFill());
      bool fragmentFill = access.getAccessFill() && !fill;
      if (fragmentFill) {
        if (!access.getAccessValidity())
          return load.emitOpError(
              "fragment padding requires an explicit access validity");
        FailureOr<Value> zero = gpu::materializeScalarConstant(
            nested, load.getLoc(), nested.getZeroAttr(view.getElementType()),
            view.getElementType());
        if (failed(zero))
          return failure();
        fill = *zero;
      }
      FailureOr<SmallVector<Value>> materialized =
          materializeCoordinateDomains(nested, access);
      if (failed(materialized))
        return failure();
      auto replacement = nested.create<GatherLoadOp>(
          load.getLoc(), result, access.getAccessResource(), *materialized,
          access.getAccessValidity(), fill, loopLatency, identityAxes(view.getRank()),
          activeInBounds ? nested.getUnitAttr() : UnitAttr());
      createdOperations.push_back(replacement);
      if (fragmentFill) {
        auto selected = nested.create<gpu::SelectOp>(
            load.getLoc(), result, access.getAccessValidity(), replacement.getResult(),
            access.getAccessFill());
        createdOperations.push_back(selected);
        return selected.getResult();
      }
      return replacement.getResult();
    };
    auto emitNativeOrFill = [&](OpBuilder &nested) -> Value {
      if (!guardedNative)
        return emitNativeLoad(nested);
      auto conditional = nested.create<scf::IfOp>(
          load.getLoc(), TypeRange{result}, *originGuard,
          /*withElseRegion=*/true);
      createdOperations.push_back(conditional);
      OpBuilder inBounds = prepareBranch(conditional.getThenRegion());
      Value value = emitNativeLoad(inBounds);
      inBounds.create<scf::YieldOp>(load.getLoc(), value);
      OpBuilder outOfBounds = prepareBranch(conditional.getElseRegion());
      outOfBounds.create<scf::YieldOp>(load.getLoc(), access.getAccessFill());
      return conditional.getResult(0);
    };
    auto emitGuardedNative = [&](OpBuilder &nested) -> FailureOr<Value> {
      Value condition = indices->alignment;
      if (rangeGuard)
        condition = condition ? Value(nested.create<gpu::BinaryOp>(
                                    load.getLoc(), nested.getI1Type(), condition,
                                    rangeGuard, BinaryOperator::LogicalAnd))
                              : rangeGuard;
      if (!condition)
        return emitNativeOrFill(nested);
      auto conditional = nested.create<scf::IfOp>(
          load.getLoc(), TypeRange{result}, condition,
          /*withElseRegion=*/true);
      createdOperations.push_back(conditional);
      OpBuilder aligned = prepareBranch(conditional.getThenRegion());
      Value value = rangeGuard ? emitNativeLoad(aligned)
                              : emitNativeOrFill(aligned);
      aligned.create<scf::YieldOp>(load.getLoc(), value);
      OpBuilder unaligned = prepareBranch(conditional.getElseRegion());
      FailureOr<Value> gathered = emitGatherLoad(unaligned);
      if (failed(gathered))
        return failure();
      unaligned.create<scf::YieldOp>(load.getLoc(), *gathered);
      return conditional.getResult(0);
    };
    if (native) {
      FailureOr<Value> condition = forms.loadFormCondition();
      if (failed(condition))
        return failure();
      auto conditional = builder.create<scf::IfOp>(
          load.getLoc(), TypeRange{result}, *condition,
          /*withElseRegion=*/true);
      createdOperations.push_back(conditional);
      OpBuilder tiled = prepareBranch(conditional.getThenRegion());
      FailureOr<Value> tiledValue = emitGuardedNative(tiled);
      if (failed(tiledValue))
        return failure();
      tiled.create<scf::YieldOp>(load.getLoc(), *tiledValue);
      OpBuilder gathered = prepareBranch(conditional.getElseRegion());
      FailureOr<Value> gatheredValue = emitGatherLoad(gathered);
      if (failed(gatheredValue))
        return failure();
      gathered.create<scf::YieldOp>(load.getLoc(), *gatheredValue);
      replacementResult = conditional.getResult(0);
    } else {
      FailureOr<Value> gathered = emitGatherLoad(builder);
      if (failed(gathered))
        return failure();
      replacementResult = *gathered;
    }
    for (Operation *operation : createdOperations)
      if (Attribute origin = load->getAttr(gpu::originAttr))
        operation->setAttr(gpu::originAttr, origin);
    rewriter.replace(load, ValueRange{replacementResult});
  }
  return success();
}

} // namespace intent::cutile
