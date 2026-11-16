#include "Accesses.h"
#include "AccessFormSelection.h"
#include "Coordinates.h"
#include "Bounds.h"
#include "TileIndices.h"
#include "Intent/Dialect/Block/Analysis/IndexRelations.h"
#include "Intent/Dialect/Block/Analysis/ValueSchema.h"
#include "Intent/Dialect/Block/IR/Program.h"
#include "Intent/Dialect/Block/IR/FragmentOpInterface.h"
#include "Intent/Dialect/Block/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/Block/Transforms/Configuration/PhysicalParameters.h"
#include "Intent/Target/CuTile/Analysis/IndexBounds.h"
#include "Intent/Target/CuTile/IR/CuTileOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

namespace intent::cutile {

LogicalResult formNativeStores(func::FuncOp kernel,
    ArrayRef<block::StoreOp> stores, AccessFormSelection &forms,
    NativeFormRewriter &rewriter) {
  for (block::StoreOp store : stores) {
    auto access = cast<block::AccessOpInterface>(store.getOperation());
    block::PhysicalProgramAnalysis analysis(kernel);
    auto view = dyn_cast<block::ViewType>(access.getAccessResource().getType());
    if (!view)
      return store.emitOpError(
          "cuTile native store requires an external unique-write view");
    OpBuilder builder(store);
    if (!isa<block::FragmentType>(access.getAccessPayloads().front().getType())) {
      FailureOr<SmallVector<Value>> indices = orderedCoordinates(access);
      if (failed(indices) ||
          llvm::any_of(*indices, [](Value value) {
            return isa<block::FragmentType>(value.getType());
          }) ||
          (access.getAccessValidity() && isa<block::FragmentType>(access.getAccessValidity().getType())))
        return store.emitOpError(
            "cuTile scalar store requires scalar indices and validity");
      bool inBounds = scalarCoordinatesInView(*indices, view, kernel);
      bool activeInBounds = inBounds || analysis.accessBounds(store).isExact();
      Value validity = access.getAccessValidity();
      block::PhysicalAccessBoundaryFact boundary =
          analysis.boundaryValidity(store);
      if (inBounds && boundary.isExact())
        validity = {};
      auto replacement = builder.create<ScalarStoreOp>(
          store.getLoc(), access.getAccessResource(), *indices, access.getAccessPayloads().front(),
          validity, activeInBounds ? builder.getUnitAttr() : UnitAttr());
      if (Attribute origin = store->getAttr(block::originAttr))
        replacement->setAttr(block::originAttr, origin);
      rewriter.erase(store);
      continue;
    }
    block::PhysicalAccessBoundaryFact boundary =
        analysis.boundaryValidity(store, /*allowRangeGuards=*/true);
    const auto accessBounds = analysis.accessBounds(store);
    bool activeInBounds = accessBounds.isExact();
    auto computationType =
        cast<block::FragmentType>(access.getAccessPayloads().front().getType());
    FailureOr<NativeTileAccessPlan> plan = analyzeNativeTileAccess(access, kernel, accessBounds);
    FailureOr<MaterializedTileIndices> indices = failure();
    FailureOr<Value> originGuard = failure();
    if (succeeded(plan) && boundary.isExact())
      indices = materializeTileIndices(builder, store, *plan,
                                       /*allowDynamicAlignment=*/true);
    if (succeeded(indices))
      originGuard = materializeTileOriginGuard(
          builder, store, access.getAccessResource(), *plan, boundary);
    bool native = succeeded(indices) && succeeded(originGuard);
    bool guardedNative = native && *originGuard;
    Value nativeCondition;
    if (native) {
      nativeCondition = indices->alignment;
      Value rangeGuard = materializeFullRangeGuard(builder, store.getLoc(),
                                                  boundary);
      if (rangeGuard)
        nativeCondition = nativeCondition
                              ? Value(builder.create<block::BinaryOp>(
                                    store.getLoc(), builder.getI1Type(),
                                    nativeCondition, rangeGuard,
                                    BinaryOperator::LogicalAnd))
                              : rangeGuard;
    }
    FailureOr<Value> allowTMA = failure();
    if (native) {
      allowTMA = forms.tmaCondition(plan->resourceType);
      if (failed(allowTMA))
        return failure();
    }
    SmallVector<Operation *> createdOperations;
    auto emitNativeStore = [&](OpBuilder &nested) {
      Value nativeValue = access.getAccessPayloads().front();
      if (!isIdentityPermutation(plan->toResource)) {
        auto transpose = nested.create<block::TransposeOp>(
            store.getLoc(), plan->packedType, nativeValue,
            plan->toResource);
        nativeValue = transpose.getResult();
        createdOperations.push_back(transpose);
      }
      if (plan->packedToResource) {
        auto reshape = nested.create<block::ReshapeOp>(
            store.getLoc(), plan->resourceType, nativeValue,
            plan->packedToResource);
        nativeValue = reshape.getResult();
        createdOperations.push_back(reshape);
      }
      auto tile = nested.create<TileStoreOp>(
          store.getLoc(), access.getAccessResource(), *allowTMA, indices->values,
          nativeValue, IntegerAttr{});
      createdOperations.push_back(tile);
    };
    auto emitBoundedNativeStore = [&](OpBuilder &nested) {
      if (!guardedNative) {
        emitNativeStore(nested);
        return;
      }
      auto conditional = nested.create<scf::IfOp>(
          store.getLoc(), TypeRange{}, *originGuard,
          /*withElseRegion=*/true);
      createdOperations.push_back(conditional);
      OpBuilder inBounds = prepareBranch(conditional.getThenRegion());
      emitNativeStore(inBounds);
      inBounds.create<scf::YieldOp>(store.getLoc());
      OpBuilder outOfBounds = prepareBranch(conditional.getElseRegion());
      outOfBounds.create<scf::YieldOp>(store.getLoc());
    };
    auto emitScatterStore = [&](OpBuilder &nested) -> LogicalResult {
      FailureOr<SmallVector<Value>> materialized =
          materializeCoordinateDomains(nested, access);
      if (failed(materialized))
        return failure();
      auto replacement = nested.create<ScatterStoreOp>(
          store.getLoc(), access.getAccessResource(), *materialized, access.getAccessPayloads().front(),
          access.getAccessValidity(), identityAxes(view.getRank()),
          activeInBounds ? nested.getUnitAttr() : UnitAttr());
      createdOperations.push_back(replacement);
      return success();
    };
    if (native && nativeCondition) {
      auto conditional = builder.create<scf::IfOp>(
          store.getLoc(), TypeRange{}, nativeCondition,
          /*withElseRegion=*/true);
      createdOperations.push_back(conditional);
      OpBuilder tiled = prepareBranch(conditional.getThenRegion());
      emitBoundedNativeStore(tiled);
      tiled.create<scf::YieldOp>(store.getLoc());
      OpBuilder scattered = prepareBranch(conditional.getElseRegion());
      if (failed(emitScatterStore(scattered)))
        return failure();
      scattered.create<scf::YieldOp>(store.getLoc());
    } else if (native) {
      emitBoundedNativeStore(builder);
    } else if (failed(emitScatterStore(builder))) {
      return failure();
    }
    for (Operation *operation : createdOperations)
      if (Attribute origin = store->getAttr(block::originAttr))
        operation->setAttr(block::originAttr, origin);
    rewriter.erase(store);
  }
  return success();
}

} // namespace intent::cutile
