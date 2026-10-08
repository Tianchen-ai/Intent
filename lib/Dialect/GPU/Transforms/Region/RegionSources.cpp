#include "RegionSources.h"
#include "../Value/SourceReplay.h"
#include "../Value/ReplayPolicy.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Storage/Storage.h"
#include "Intent/Dialect/GPU/Transforms/Control/Traversal.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;

namespace intent::gpu::region {

PhysicalExprAttr parameterExtent(ParameterRefAttr parameter) {
  return PhysicalExprAttr::get(
      parameter.getContext(),
      PhysicalExprKind::Parameter, 0,
      parameter, ArrayAttr::get(parameter.getContext(), {}));
}

FragmentType replaceSliceAxis(FragmentType source, unsigned axis,
                              PhysicalExprAttr extent,
                              AxisMapAttr segmentMapping) {
  SmallVector<Attribute> shape(source.getShape().begin(), source.getShape().end());
  SmallVector<Attribute> mappings(source.getAxisMaps().begin(),
                                  source.getAxisMaps().end());
  shape[axis] = extent;
  mappings[axis] = AxisMapAttr::get(
      source.getContext(), segmentMapping.getSourceId(),
      segmentMapping.getSourceAxis(), segmentMapping.getDimensionId(), axis,
      segmentMapping.getDerived());
  return FragmentType::get(
      source.getContext(), source.getElementType(),
      ArrayAttr::get(source.getContext(), shape),
      ArrayAttr::get(source.getContext(), mappings), source.getValidity(),
      source.getOwner());
}

FragmentType predicateType(FragmentType source) {
  return FragmentType::get(
      source.getContext(), IntegerType::get(source.getContext(), 1),
      source.getShape(), source.getAxisMaps(), source.getValidity(),
      source.getOwner());
}

bool isUnitExtent(Attribute attribute) {
  auto expression = dyn_cast<PhysicalExprAttr>(attribute);
  return expression &&
         expression.getKind() ==
             PhysicalExprKind::Constant &&
         expression.getValue() == 1;
}

namespace {
FailureOr<SourcePlan> analyzeSource(Value source, unsigned sourceAxis,
                                    PhysicalProgramAnalysis &analysis) {
  auto fragment = dyn_cast<FragmentType>(source.getType());
  if (!fragment || sourceAxis >= fragment.getShape().size())
    return failure();
  FailureOr<AxisMapAttr> mapping = queryAxisMap(fragment, sourceAxis);
  if (failed(mapping))
    return failure();
  SourcePlan plan{source, sourceAxisIdentity(*mapping), sourceAxis, false, {}, {}};
  PhysicalRangeFact fact = analysis.axisRanges(source, sourceAxis);
  if (failed(queryExactLogicalRange(fact)))
    return failure();
  plan.ranges.assign(fact.roots.begin(), fact.roots.end());
  UniformValueAnalysis values(describeUniformValue);
  UniformBindings tailValues;
  for (Operation *access : fact.accesses) {
    auto sliced = [&](Value value) {
      auto resultType = dyn_cast<FragmentType>(value.getType());
      return resultType && queryFragmentAxis(resultType, plan.sourceIdentity).isExact();
    };
    if (auto load = dyn_cast<LoadOp>(access)) {
      plan.hasLoads = true;
      if (sliced(load.getResult())) tailValues[load.getResult()] = load.getFill()
          ? values.evaluate(load.getFill()) : uniformZero(uniformElementType(load.getResult().getType()));
      continue;
    }
    if (auto gather = dyn_cast<GatherOp>(access)) {
      plan.hasLoads = true;
      if (sliced(gather.getResult())) tailValues[gather.getResult()] = gather.getFill()
          ? values.evaluate(gather.getFill()) : uniformZero(uniformElementType(gather.getResult().getType()));
      continue;
    }
    return access->emitOpError(
               "region-fold source replay encountered an unsupported access"),
           failure();
  }
  if (!fact.unitStep)
    return plan.ranges.front().emitOpError(
        "region-fold source traversal requires a unit-step physical range");
  if (plan.ranges.empty())
    return failure();
  plan.tailConstant = values.evaluate(source, tailValues);
  return plan;
}
} // namespace

FailureOr<SmallVector<SourcePlan>> prepareRegionSources(
    ValueRange sources, unsigned sourceAxis, Operation *insertionAnchor) {
  auto kernel = insertionAnchor->getParentOfType<func::FuncOp>();
  for (Value source : sources) {
    if (failed(prepareSourceReplay(source, sourceAxis, insertionAnchor,
                                   PhysicalReplayScope::Coordinate)))
      return insertionAnchor->emitOpError(
                 "region source snapshot could not be fully materialized"),
             failure();
  }

  // Coverage materialization can retarget shared ranges and all their users.
  // Build every source plan from the resulting program, after all snapshots.
  PhysicalProgramAnalysis analysis(kernel);
  SmallVector<SourcePlan> plans;
  for (Value source : sources) {
    auto plan = analyzeSource(source, sourceAxis, analysis);
    if (failed(plan))
      return failure();
    plans.push_back(std::move(*plan));
  }
  return plans;
}

LogicalResult buildSourceSlices(OpBuilder &builder, Location location,
                                ArrayRef<SourcePlan> plans,
                                ArrayRef<FragmentType> sliceTypes, Value offset,
                                Value segment, PhysicalExprAttr sliceExtent,
                                bool fullSegment,
                                SmallVectorImpl<Value> &slices,
                                Value &segmentTail, IRMapping &sliceMapping,
                                SmallVectorImpl<std::shared_ptr<IRMapping>>
                                    &sourceMappings,
                                Operation *insertionAnchor, std::string &reason) {
  SmallVector<Value> roots;
  for (const auto &plan : plans) roots.push_back(plan.source);
  ReplayPolicy reuse(insertionAnchor->getParentOfType<func::FuncOp>(), roots,
                     {insertionAnchor}, [&](Value current) {
    return llvm::any_of(plans, [&](const SourcePlan &plan) {
      auto type = cast<FragmentType>(plan.source.getType());
      auto axis = cast<AxisMapAttr>(type.getAxisMaps()[plan.sourceAxis]);
      return queryFragmentAxis(current.getType(), plan.sourceIdentity,
                               axis.getDimensionId()).isExact();
    });
  });
  for (auto [planIndex, plan] : llvm::enumerate(plans)) {
    if (planIndex >= sliceTypes.size() ||
        plan.sourceAxis >= sliceTypes[planIndex].getAxisMaps().size()) {
      reason = "source slice has no helper-local segment coordinate mapping";
      return failure();
    }
    // A shared mapping can reuse common SSA prefixes of distinct source values
    // only under the same actual roots and complete selected coordinate schema.
    // Producer/value identity remains the mapping key; provenance alone is not.
    std::optional<unsigned> sharedRelation;
    for (unsigned previous = 0; previous < planIndex; ++previous) {
      const SourcePlan &candidate = plans[previous];
      auto previousType = sliceTypes[previous], currentType = sliceTypes[planIndex];
      bool sameSliceSchema = previousType.getShape() == currentType.getShape() &&
          previousType.getAxisMaps() == currentType.getAxisMaps() &&
          previousType.getValidity() == currentType.getValidity() &&
          previousType.getOwner() == currentType.getOwner();
      if (sameSliceSchema &&
          candidate.sourceIdentity == plan.sourceIdentity &&
          candidate.sourceAxis == plan.sourceAxis &&
          candidate.ranges == plan.ranges) {
        sharedRelation = previous;
        break;
      }
    }
    std::shared_ptr<IRMapping> ownedMapping =
        sharedRelation ? sourceMappings[*sharedRelation]
                       : std::make_shared<IRMapping>();
    IRMapping &mapping = *ownedMapping;
    auto segmentMapping = cast<AxisMapAttr>(
        sliceTypes[planIndex].getAxisMaps()[plan.sourceAxis]);
    Value tail = sharedRelation ? segmentTail : Value();
    auto buildRange = [&](MakeRangeOp range) -> Value {
      Value start = builder.create<BinaryOp>(
          location, builder.getIndexType(), range.getStart(), offset,
          BinaryOperator::Add);
      auto rangeType = cast<FragmentType>(range.getResult().getType());
      auto blockedRange =
          replaceSliceAxis(rangeType, 0, sliceExtent, segmentMapping);
      Value logicalStart = range.getLogicalStart();
      Value logicalStop = range.getLogicalStop();
      if (fullSegment && isUnitExtent(sliceExtent)) {
        logicalStart = start;
        logicalStop = builder.create<BinaryOp>(
            location, builder.getIndexType(), start, range.getStep(),
            BinaryOperator::Add);
      }
      Value value = builder.create<MakeRangeOp>(
          location, blockedRange, start, segment, range.getStep(),
          logicalStart, logicalStop,
          segmentMapping.getSourceId(), segmentMapping.getSourceAxis(),
          segmentMapping.getDerived());
      for (StringRef name :
           {sourceSubregionAttr, sourceSubregionBoundAttr})
        if (Attribute inherited = range->getAttr(name))
          value.getDefiningOp()->setAttr(name, inherited);
      Value valid;
      if (fullSegment) {
        Value truth = builder.create<arith::ConstantOp>(
            location, builder.getI1Type(), builder.getBoolAttr(true));
        valid = builder.create<SplatOp>(location, predicateType(blockedRange),
                                        truth);
      } else {
        Value stopFragment = builder.create<BroadcastOp>(
            location, blockedRange, range.getLogicalStop());
        auto validComparison = builder.create<CompareOp>(
            location, predicateType(blockedRange), value, stopFragment,
            ComparePredicate::Lt);
        validComparison->setAttr(physicalTailAttr, builder.getUnitAttr());
        valid = validComparison.getResult();
      }
      if (!tail)
        tail = valid;
      if (!segmentTail)
        segmentTail = valid;
      if (!mapping.lookupOrNull(range.getResult()))
        mapping.map(range.getResult(), value);
      if (!sliceMapping.lookupOrNull(range.getResult()))
        sliceMapping.map(range.getResult(), value);
      return value;
    };
    if (!sharedRelation)
      for (MakeRangeOp range : plan.ranges)
        buildRange(range);
    if (!tail) {
      reason = "source slice has no tail predicate";
      return failure();
    }
    ReplayMaterializationOptions replayOptions;
    replayOptions.scope = PhysicalReplayScope::Coordinate;
    replayOptions.fragmentAxis = plan.sourceAxis;
    replayOptions.traversalRanges = plan.ranges;
    replayOptions.segmentTail = tail;
    replayOptions.segmentMapping = segmentMapping;
    replayOptions.materializeZeroFill = true;
    auto replayed = materializeSourceValue(
        builder, location, plan.source, plan.sourceIdentity, sliceExtent,
        mapping, insertionAnchor, replayOptions,
        fullSegment ? Attribute() : plan.tailConstant, reuse);
    if (failed(replayed)) {
      reason = "source pure producer graph cannot be replayed";
      return failure();
    }
    mapping.map(plan.source, *replayed);
    slices.push_back(*replayed);
    if (!sliceMapping.lookupOrNull(plan.source))
      sliceMapping.map(plan.source, *replayed);
    sourceMappings.push_back(std::move(ownedMapping));
  }
  if (!segmentTail) {
    reason = "source slices have no shared physical tail predicate";
    return failure();
  }
  return success();
}

} // namespace intent::gpu::region
