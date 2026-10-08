#include "Accesses.h"
#include "Coordinates.h"
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

LogicalResult formNativeAtomics(func::FuncOp kernel,
    ArrayRef<gpu::AtomicRMWOp> atomics, NativeFormRewriter &rewriter) {
  for (gpu::AtomicRMWOp atomic : atomics) {
    auto access = cast<gpu::AccessOpInterface>(atomic.getOperation());
    gpu::PhysicalProgramAnalysis analysis(kernel);
    const auto accessBounds = analysis.accessBounds(atomic);
    bool activeInBounds = accessBounds.isExact();
    auto view = dyn_cast<gpu::ViewType>(access.getAccessResource().getType());
    if (!view)
      return atomic.emitOpError(
          "cuTile atomic RMW requires an external view resource");
    if (access.getAccessValidity() &&
        (!access.getAccessResult().use_empty() || view.getRank() == 0))
      return atomic.emitOpError(
          "masked cuTile array atomic requires a ranked view and an unused old value");
    auto tileType = dyn_cast<gpu::FragmentType>(access.getAccessPayloads().front().getType());
    Type element = view.getElementType();
    if (tileType && (element.isF16() || element.isBF16()) &&
        atomic.getKind() == AtomicRMWKind::Add &&
        atomic.getOrdering() == AtomicOrdering::Relaxed &&
        atomic.getSharing() == gpu::AtomicSharingDomain::KernelInvocation &&
        access.getAccessResult().use_empty()) {
      auto boundary = analysis.boundaryValidity(atomic);
      auto plan = analyzeNativeTileAccess(access, kernel, accessBounds);
      OpBuilder builder(atomic);
      FailureOr<MaterializedTileIndices> indices = failure();
      FailureOr<Value> originGuard = failure();
      if (succeeded(plan) && boundary.isExact())
        indices = materializeTileIndices(builder, atomic, *plan,
                                         /*allowDynamicAlignment=*/false);
      if (succeeded(indices) && !indices->alignment)
        originGuard = materializeTileOriginGuard(
            builder, atomic, access.getAccessResource(), *plan, boundary);
      if (succeeded(originGuard)) {
        auto emit = [&](OpBuilder &nested) {
          Value value = access.getAccessPayloads().front();
          if (!isIdentityPermutation(plan->toResource))
            value = nested.create<gpu::TransposeOp>(
                atomic.getLoc(), plan->packedType, value, plan->toResource);
          if (plan->packedToResource)
            value = nested.create<gpu::ReshapeOp>(
                atomic.getLoc(), plan->resourceType, value,
                plan->packedToResource);
          auto replacement = nested.create<TileAtomicAddOp>(
              atomic.getLoc(), access.getAccessResource(), indices->values, value);
          if (Attribute origin = atomic->getAttr(gpu::originAttr))
            replacement->setAttr(gpu::originAttr, origin);
        };
        if (*originGuard) {
          auto conditional = builder.create<scf::IfOp>(
              atomic.getLoc(), TypeRange{}, *originGuard,
              /*withElseRegion=*/false);
          OpBuilder body = prepareBranch(conditional.getThenRegion());
          emit(body);
          body.create<scf::YieldOp>(atomic.getLoc());
        } else {
          emit(builder);
        }
        rewriter.erase(atomic);
        continue;
      }
    }
    OpBuilder builder(atomic);
    auto target = dyn_cast<gpu::FragmentType>(access.getAccessPayloads().front().getType());
    FailureOr<SmallVector<Value>> coordinates = target
        ? materializeCoordinateDomains(builder, access)
        : orderedCoordinates(access);
    if (failed(coordinates))
      return failure();
    if (access.getAccessValidity()) {
      // Native array atomics suppress out-of-bounds lanes but leave their
      // returned old values unspecified. The old value is unused here, so
      // encode the validity in the existing native bounds predicate.
      Type coordinateType = withElementType(access.getAccessPayloads().front().getType(),
                                            builder.getIndexType());
      Value zero = builder.create<arith::ConstantIndexOp>(atomic.getLoc(), 0);
      Value end = builder.create<gpu::DimOp>(
          atomic.getLoc(), builder.getIndexType(), access.getAccessResource(), 0);
      for (auto [axis, coordinate] : llvm::enumerate(*coordinates)) {
        Type indexed = withElementType(coordinate.getType(),
                                       builder.getIndexType());
        if (coordinate.getType() != indexed)
          coordinate = builder.create<gpu::CastOp>(atomic.getLoc(), indexed,
                                                   coordinate);
        auto active = gpu::projectPhysicalValueToSchema(
            builder, atomic.getLoc(), coordinate, coordinateType);
        auto inactive = gpu::projectPhysicalValueToSchema(
            builder, atomic.getLoc(), axis == 0 ? end : zero, coordinateType);
        if (failed(active) || failed(inactive))
          return atomic.emitOpError(
              "masked atomic coordinate cannot adopt its value domain");
        (*coordinates)[axis] = builder.create<gpu::SelectOp>(
            atomic.getLoc(), coordinateType, access.getAccessValidity(), *active,
            *inactive);
      }
    }
    auto replacement = builder.create<AtomicRMWOp>(
        atomic.getLoc(), access.getAccessResult().getType(), access.getAccessResource(),
        *coordinates, access.getAccessPayloads().front(), atomic.getKind(), atomic.getOrdering(),
        atomic.getSharing(), activeInBounds ? builder.getUnitAttr() : UnitAttr());
    if (Attribute origin = atomic->getAttr(gpu::originAttr))
      replacement->setAttr(gpu::originAttr, origin);
    rewriter.replace(atomic, ValueRange{replacement.getResult()});
  }
  return success();
}

} // namespace intent::cutile
