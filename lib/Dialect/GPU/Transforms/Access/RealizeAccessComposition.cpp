#include "AccessComposition.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Transforms/Control/Traversal.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"

using namespace mlir;

namespace intent::gpu {
using namespace access;
namespace {

LogicalResult materializeIndexedFragments(func::FuncOp kernel) {
  SmallVector<GatherOp> gathers;
  kernel.walk([&](GatherOp gather) { gathers.push_back(gather); });
  for (GatherOp gather : gathers) {
    for (auto [coordinate, sourceAxis] :
         llvm::zip(gather.getCoordinates(), gather.getSourceAxes())) {
      PhysicalProgramAnalysis analysis(kernel);
      if (!analysis.axisRealization(gather.getSource(), sourceAxis)
               .constructionScalarSeed)
        continue;
      auto source = cast<FragmentType>(gather.getSource().getType());
      auto extent = cast<PhysicalExprAttr>(source.getShape()[sourceAxis]);
      PhysicalExprAttr bound = queryNonNegativeIndexUpperBound(coordinate);
      if (bound &&
          bound.getKind() == PhysicalExprKind::Constant &&
          extent.getKind() == PhysicalExprKind::Constant &&
          bound.getValue() < extent.getValue())
        continue;
      if (failed(realizeFullCoverageDimension(kernel, gather.getSource(),
                                               sourceAxis)))
        return gather.emitOpError(
            "indexed tensor source has no complete physical extent");
    }
  }
  return success();
}

} // namespace

static LogicalResult realizeAccessCompositionImpl(ModuleOp module) {
  FailureOr<func::FuncOp> physicalKernel = getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  eraseDeadPhysicalValues(*physicalKernel);
  bool changed;
  do {
    changed = false;
    SmallVector<SelectOp> selects;
    physicalKernel->walk([&](SelectOp select) { selects.push_back(select); });
    for (SelectOp select : selects) {
      if (!select->getBlock())
        continue;
      FailureOr<bool> load = composeSelectLoad(select);
      if (failed(load))
        return failure();
      changed |= *load;
    }
    SmallVector<GatherOp> gathers;
    physicalKernel->walk([&](GatherOp gather) { gathers.push_back(gather); });
    for (GatherOp gather : gathers) {
      if (!gather->getBlock())
        continue;
      if (reuseFragmentGather(gather)) {
        changed = true;
        continue;
      }
      FailureOr<bool> identity = composeIdentityFragmentGather(gather);
      if (failed(identity))
        return failure();
      if (*identity) {
        changed = true;
        continue;
      }
      FailureOr<bool> range = composeRangeGather(gather);
      if (failed(range))
        return failure();
      if (*range) {
        changed = true;
        continue;
      }
      FailureOr<bool> reshaped = composeReshapedGather(gather);
      if (failed(reshaped))
        return failure();
      if (*reshaped) {
        changed = true;
        continue;
      }
      FailureOr<bool> pointwise = composePointwiseGather(gather);
      if (failed(pointwise))
        return failure();
      if (*pointwise) {
        changed = true;
        continue;
      }
      FailureOr<bool> reduced = composeReducedGather(gather);
      if (failed(reduced))
        return failure();
      if (*reduced) {
        changed = true;
        continue;
      }
      FailureOr<bool> projection = projectFragmentGather(gather);
      if (failed(projection))
        return failure();
      if (*projection) {
        changed = true;
        continue;
      }
      FailureOr<bool> broadcast = composeBroadcastGather(gather);
      if (failed(broadcast))
        return failure();
      if (*broadcast) {
        changed = true;
        continue;
      }
      FailureOr<bool> load = composeLoadGather(gather);
      if (failed(load))
        return failure();
      changed |= *load;
    }
    SmallVector<ReshapeOp> reshapes;
    physicalKernel->walk([&](ReshapeOp reshape) { reshapes.push_back(reshape); });
    for (ReshapeOp reshape : reshapes) {
      if (composeReshapedPointwise(reshape)) {
        changed = true;
        continue;
      }
      FailureOr<bool> composed = composeReshapedLoad(reshape);
      if (failed(composed))
        return failure();
      changed |= *composed;
    }
    SmallVector<StoreOp> stores;
    physicalKernel->walk([&](StoreOp store) { stores.push_back(store); });
    for (StoreOp store : stores) {
      FailureOr<bool> composed = composeReshapedStore(store);
      if (failed(composed))
        return failure();
      changed |= *composed;
    }
    changed |= reuseStableLoads(*physicalKernel);
    changed |= foldIndexRecompositions(*physicalKernel);
    SmallVector<ReduceOp> reductions;
    physicalKernel->walk([&](ReduceOp reduce) { reductions.push_back(reduce); });
    for (ReduceOp reduce : reductions) {
      FailureOr<bool> composed = composeReductionGathers(reduce);
      if (failed(composed))
        return failure();
      changed |= *composed;
    }
    eraseDeadPhysicalValues(*physicalKernel);
  } while (changed);
  if (failed(materializeIndexedFragments(*physicalKernel)))
    return failure();
  sinkStableLoadChains(*physicalKernel);
  return success();
}

LogicalResult realizeAccessComposition(ModuleOp module) {
  if (failed(realizeAccessCompositionImpl(module))) return failure();
  auto kernel = getPhysicalKernel(module);
  return failed(kernel) ? failure() : closeValueRelations(*kernel);
}

} // namespace intent::gpu
