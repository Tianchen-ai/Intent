#include "ReductionAnalysis.h"
#include "ReductionParameters.h"
#include "ReductionRealization.h"
#include "ReductionChains.h"
#include "ReductionValues.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;

namespace intent::gpu {
using namespace reduction;
namespace {

LogicalResult realizeReduce(ReduceOp reduce, func::FuncOp kernel) {
  if (failed(bindReductionFreeAxes(reduce, kernel)))
    return failure();
  const bool oversized = llvm::any_of(
      reduce.getSources(),
      [&](Value source) { return exceedsRegisterFile(source, kernel); });
  const bool needsRealization = requiresPhysicalRealization(reduce);
  const bool boundedSources = !needsRealization &&
      shouldTileReductionSources(reduce, kernel);
  if (!needsRealization && oversized && reduce.getAxes().size() == 1) {
    PhysicalProgramAnalysis analysis(kernel);
    const unsigned axis = reduce.getAxes().front();
    bool complete = llvm::all_of(
        reduce.getSources(),
        [&](Value source) {
          auto fact = analysis.axisRealization(source, axis);
          return fact.isExact() && fact.physicalized;
        });
    if (complete && llvm::any_of(
            reduce.getSources(),
            [&](Value source) {
              auto type = cast<FragmentType>(source.getType());
              auto mapping = cast<AxisMapAttr>(type.getAxisMaps()[axis]);
              return !analysis.replayAt(
                  source, sourceAxisIdentity(mapping),
                  PhysicalReplayScope::ValueGraph, /*allowAccesses=*/true,
                  reduce, IRMapping{}, mapping.getDimensionId()).isReplayable();
            }))
      // Chunking cannot reduce a fully retained, non-replayable producer.
      // Keep its current SSA value for the provider-native reduction.
      return success();
  }
  const bool required = needsRealization || oversized;
  auto unhandled = [&](const Twine &reason) -> LogicalResult {
    return required ? reduce.emitOpError()
                          << "cannot form a complete physical reduction: "
                          << reason
                    : success();
  };
  if (reduce.getAxes().size() != 1 || reduce.getSources().size() == 0)
    return unhandled("requires one reduction axis and at least one source");
  if (hasSelectedSegmentExtent(reduce, kernel))
    return success();
  // A full-coverage fragment grows every component with the runtime axis.
  // Its padding still needs the reduction identity after physicalization.
  // Keep coupled record/tuple accumulators bounded by a real chunk loop.
  if (reduce.getSources().size() == 1) {
    FailureOr<bool> fullCoverage = realizeFullCoverageReduce(reduce, kernel);
    if (failed(fullCoverage))
      return failure();
    if (*fullCoverage)
      return success();
  }
  // A fixed physical axis alone does not require replay. The optional working
  // set schedule proves that existing producers disappear instead of retaining
  // them alongside newly supplied chunks.
  if (!required && !boundedSources)
    return success();
  int64_t reductionAxis = reduce.getAxes().front();
  SmallVector<SourcePlan> sourcePlans;
  SmallVector<LoadOp> sourceLoads;
  SmallVector<MakeRangeOp> sourceRanges;
  SmallVector<unsigned> coordinateIndices;
  PhysicalExprAttr sourceExtent;
  bool hasDerivedSource = false;
  bool hasRuntimeSourceRange = false;
  bool hasConstructionScalarSource = false;
  for (Value source : reduce.getSources()) {
    auto fragment = dyn_cast<FragmentType>(source.getType());
    if (!fragment || reductionAxis < 0 ||
        reductionAxis >= static_cast<int64_t>(fragment.getShape().size()))
      return unhandled("source has no physical reduction axis");
    hasConstructionScalarSource |=
        PhysicalProgramAnalysis(kernel)
            .axisRealization(source, static_cast<unsigned>(reductionAxis))
            .constructionScalarSeed;
    FailureOr<SourcePlan> plan = analyzeSource(source, reductionAxis);
    if (failed(plan)) {
      FailureOr<AxisMapAttr> mapping = queryAxisMap(fragment, reductionAxis);
      bool replayable = succeeded(mapping) &&
                        isReplayableWithoutLoad(
                            source, sourceAxisIdentity(*mapping));
      if (failed(mapping) || !replayable) {
        PhysicalRangeFact ranges =
            PhysicalProgramAnalysis(kernel).axisRanges(source, reductionAxis);
        InFlightDiagnostic diagnostic = reduce.emitOpError(
            "cannot form a complete physical reduction: source is neither load-rooted nor replayable pure data on the reduction axis");
        diagnostic << "; producer="
                   << (source.getDefiningOp()
                           ? source.getDefiningOp()->getName().getStringRef()
                           : StringRef("block argument"))
                   << ", range_state="
                   << static_cast<unsigned>(ranges.state)
                   << ", roots=" << ranges.roots.size()
                   << ", accesses=" << ranges.accesses.size()
                   << ", blockers=" << ranges.blockers.size();
        for (Operation *blocker : ranges.blockers)
          diagnostic << ", blocker=" << blocker->getName();
        return failure();
      }
      plan = SourcePlan{source, sourceAxisIdentity(*mapping),
                        static_cast<unsigned>(reductionAxis), {}, {}, {}};
    }
    if (plan->ranges.empty()) {
      PhysicalProgramAnalysis analysis(kernel);
      DominanceInfo dominance(kernel);
      SmallVector<MakeRangeOp> visible;
      for (MakeRangeOp range :
           analysis.programRanges(plan->sourceIdentity).roots)
        if (dominance.dominates(range.getOperation(), reduce.getOperation()) &&
            range.getResult().getType().getShape()[0] ==
                fragment.getShape()[reductionAxis])
          visible.push_back(range);
      auto traversal = analysis.lockstepRanges(visible);
      if (traversal.isExact())
        plan->ranges.push_back(traversal.authority);
    }
    if (plan->roots.empty() && plan->ranges.empty() &&
        failed(scalarSource(source)))
      return reduce.emitOpError(
                 "pure reduction source has no exact physical range authority")
             << "; source=" << source;
    for (LoadOp root : plan->roots) {
      FailureOr<std::optional<RootAccess>> rootAccess =
          analyzeRoot(root, plan->ranges, plan->reductionAxis);
      if (failed(rootAccess) || !*rootAccess)
        return unhandled(
            "load-rooted producer has no unit-step reduction coordinate");
      hasRuntimeSourceRange |=
          !isCompileTimeValue((**rootAccess).range.getExtent());
    }
    PhysicalExprAttr extent =
        cast<PhysicalExprAttr>(fragment.getShape()[reductionAxis]);
    if (sourceExtent && sourceExtent != extent)
      return unhandled("reduction components disagree on physical extent");
    sourceExtent = extent;
    hasDerivedSource |=
        plan->roots.size() != 1 || plan->source != plan->roots.front().getResult();
    if (!hasDerivedSource) {
      FailureOr<std::optional<RootAccess>> analyzed = analyzeRoot(
          plan->roots.front(), plan->ranges, plan->reductionAxis);
      if (failed(analyzed) || !*analyzed)
        return unhandled(
            "load-rooted producer has no unit-step reduction coordinate");
      RootAccess access = **analyzed;
      PhysicalProgramAnalysis analysis(kernel);
      hasDerivedSource |= llvm::count_if(
          access.load.getCoordinates(), [&](Value coordinate) {
            auto axes = analysis.rangeAxes(coordinate, {access.range});
            return !axes.fragmentAxes.empty();
          }) > 1;
      Value identity = reduce.getIdentities()[sourcePlans.size()];
      if (access.load.getValid() &&
          (!sameScalarValue(access.load.getFill(), identity) ||
           failed(scalarSource(access.load.getValid()))))
        hasDerivedSource = true;
      hasDerivedSource |= !canReplayReadAt(access.load, reduce);
      sourceLoads.push_back(access.load);
      sourceRanges.push_back(access.range);
      coordinateIndices.push_back(access.coordinateIndex);
    }
    sourcePlans.push_back(*plan);
  }
  if (!sourceExtent)
    return unhandled("reduction has no physical source extent");
  auto needsPairedTraversal = [](const SourcePlan &plan) {
    return plan.roots.empty() && plan.ranges.empty();
  };
  if (llvm::any_of(sourcePlans, needsPairedTraversal)) {
    SmallVector<MakeRangeOp> pairedRanges;
    for (const SourcePlan &plan : sourcePlans)
      for (MakeRangeOp range : plan.ranges)
        if (!llvm::is_contained(pairedRanges, range))
          pairedRanges.push_back(range);
    PhysicalLockstepTraversalFact traversal =
        PhysicalProgramAnalysis(kernel).lockstepRanges(pairedRanges);
    if (!traversal.isExact())
      return reduce.emitOpError(
          "uniform reduction components require an exact paired traversal");
    // Reduce pairs these component axes. Uniform scalar broadcasts can use
    // that traversal for replay and padding while retaining their own identity.
    for (SourcePlan &plan : sourcePlans)
      if (needsPairedTraversal(plan))
        plan.ranges.push_back(traversal.authority);
  }
  if (oversized || boundedSources || !isCompileTimeExtent(sourceExtent) || hasDerivedSource ||
      hasRuntimeSourceRange || hasConstructionScalarSource)
    return realizeRuntimeReduce(reduce, sourcePlans, kernel,
                                /*tileProducerFreeAxis=*/false);

  PhysicalExprAttr blockExtent = nextPowerOfTwo(sourceExtent);
  OpBuilder builder(reduce);
  Value physicalExtent;
  if (blockExtent.getKind() ==
      PhysicalExprKind::Constant)
    physicalExtent = builder.create<arith::ConstantIndexOp>(
        reduce.getLoc(), blockExtent.getValue());
  else
    physicalExtent = builder.create<PhysicalExprOp>(
        reduce.getLoc(), builder.getIndexType(), blockExtent);

  SmallVector<Value> blockedSources;
  for (unsigned component = 0; component < reduce.getSources().size(); ++component) {
    LoadOp load = sourceLoads[component];
    MakeRangeOp range = sourceRanges[component];
    auto sourceType = cast<FragmentType>(load.getResult().getType());
    FragmentType blockedSource =
        replaceExtent(sourceType, reductionAxis, blockExtent);
    SmallVector<Attribute> coordinateShape(
        range.getResult().getType().getShape().begin(),
        range.getResult().getType().getShape().end());
    coordinateShape[0] = blockExtent;
    auto blockedCoordinate = FragmentType::get(
        reduce.getContext(), range.getResult().getType().getElementType(),
        ArrayAttr::get(reduce.getContext(), coordinateShape),
        range.getResult().getType().getAxisMaps(),
        range.getResult().getType().getValidity(),
        range.getResult().getType().getOwner());
    Value coordinate = builder.create<MakeRangeOp>(
        reduce.getLoc(), blockedCoordinate, range.getStart(), physicalExtent,
        range.getStep(), range.getLogicalStart(), range.getLogicalStop(),
        range.getSourceId(), range.getSourceAxis(), range.getDerived());
    inheritRangeAuthority(coordinate.getDefiningOp(), range);
    Value stop = builder.create<BinaryOp>(
        reduce.getLoc(), builder.getIndexType(), range.getStart(),
        range.getExtent(), BinaryOperator::Add);
    auto coordinatePredicate = FragmentType::get(
        reduce.getContext(), builder.getI1Type(), blockedCoordinate.getShape(),
        blockedCoordinate.getAxisMaps(), blockedCoordinate.getValidity(),
        blockedCoordinate.getOwner());
    Value stopFragment =
        builder.create<BroadcastOp>(reduce.getLoc(), blockedCoordinate, stop);
    Value valid = builder.create<CompareOp>(
        reduce.getLoc(), coordinatePredicate, coordinate, stopFragment,
        ComparePredicate::Lt);
    auto sourcePredicate = FragmentType::get(
        reduce.getContext(), builder.getI1Type(), blockedSource.getShape(),
        blockedSource.getAxisMaps(), blockedSource.getValidity(),
        blockedSource.getOwner());
    valid = builder.create<BroadcastOp>(reduce.getLoc(), sourcePredicate, valid);
    if (load.getValid()) {
      FailureOr<Value> scalar = scalarSource(load.getValid());
      Value original = builder.create<BroadcastOp>(reduce.getLoc(),
                                                    sourcePredicate, *scalar);
      valid = builder.create<BinaryOp>(reduce.getLoc(), sourcePredicate, valid,
                                       original, BinaryOperator::LogicalAnd);
    }
    Value identity =
        reduce.getIdentities()[component];
    Value fill = identity;
    if (identity.getType() != blockedSource)
      fill = builder.create<BroadcastOp>(reduce.getLoc(), blockedSource, identity);
    SmallVector<Value> coordinates(load.getCoordinates());
    IRMapping coordinateMapping;
    coordinateMapping.map(range.getResult(), coordinate);
    FailureOr<Value> reducedCoordinate = materializeReplayedValue(
        builder, reduce.getLoc(),
        load.getCoordinates()[coordinateIndices[component]],
        sourceAxisIdentity(range), blockExtent, coordinateMapping, reduce);
    if (failed(reducedCoordinate))
      return reduce.emitOpError(
          "could not replay translated full-coverage reduction coordinate");
    coordinates[coordinateIndices[component]] = *reducedCoordinate;
    blockedSources.push_back(builder.create<LoadOp>(
        reduce.getLoc(), blockedSource, load.getResource(), coordinates, valid,
        fill, load.getSourceAxes()));
  }

  auto replacement = builder.create<ReduceOp>(reduce.getLoc(), blockedSources,
      reduce.getIdentities(), reduce.getCaptures(), reduce.getAxes());
  replacement.getCombine().takeBody(reduce.getCombine());
  if (Attribute origin = reduce->getAttr(originAttr))
    replacement->setAttr(originAttr, origin);
  for (auto [oldResult, newResult] :
       llvm::zip(reduce.getResults(), replacement.getResults()))
    oldResult.replaceAllUsesWith(newResult);
  reduce.erase();
  for (LoadOp load : sourceLoads)
    if (load->getBlock() && load.getResult().use_empty())
      load.erase();
  return success();
}

} // namespace

LogicalResult decomposeMultiAxisReductions(ModuleOp module) {
  FailureOr<func::FuncOp> physicalKernel = getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  func::FuncOp kernel = *physicalKernel;
  // Tile consumers while their producers are still structured reductions.
  // Decomposition can clone or erase ancestors, so refresh after every rewrite.
  while (true) {
    SmallVector<ReduceOp> reductions;
    kernel.walk([&](ReduceOp reduce) {
      if (reduce.getAxes().size() > 1)
        reductions.push_back(reduce);
    });
    if (reductions.empty())
      break;
    if (failed(decomposeMultiAxisReduce(reductions.back(), kernel)))
      return failure();
  }
  return success();
}

LogicalResult realizeReductionBlocking(ModuleOp module) {
  FailureOr<func::FuncOp> physicalKernel = getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  func::FuncOp kernel = *physicalKernel;
  while (combineNestedReductions(kernel))
    if (failed(fuseIndependentReductions(module))) return failure();
  // Tile a scalar consumer before its row reductions become loop results.
  // Their existing lowering then runs inside the selected outer traversal.
  while (true) {
    ReduceOp selected;
    std::optional<SourcePlan> source;
    kernel.walk([&](ReduceOp reduce) {
      FailureOr<SourcePlan> candidate =
          nestedScalarReductionSource(reduce, kernel);
      if (failed(candidate))
        return WalkResult::advance();
      selected = reduce;
      source = std::move(*candidate);
      return WalkResult::interrupt();
    });
    if (!selected)
      break;
    if (failed(realizeRuntimeReduce(selected, ArrayRef<SourcePlan>{*source},
                                    kernel, /*tileProducerFreeAxis=*/true)))
      return failure();
  }
  // Keep reductions structured until ownership and the enclosing consumer
  // traversal have been chosen.
  if (failed(decomposeMultiAxisReductions(module)))
    return failure();
  // Pad the remaining reductions before lowering them. Padding can erase dead
  // producers, so rebuild the worklist after each change.
  while (true) {
    SmallVector<ReduceOp> candidates;
    kernel.walk([&](ReduceOp reduce) { candidates.push_back(reduce); });
    bool changed = false;
    for (ReduceOp reduce : candidates) {
      if (sinkReductionIntoSourceIf(reduce, kernel)) {
        changed = true;
        break;
      }
      if (hasSelectedSegmentExtent(reduce, kernel))
        continue;
      FailureOr<bool> padded = realizeStaticPaddingReduce(reduce, kernel);
      if (failed(padded))
        return failure();
      if (*padded) {
        changed = true;
        break;
      }
    }
    if (!changed)
      break;
  }
  SmallVector<ReduceOp> reductions;
  kernel.walk([&](ReduceOp reduce) { reductions.push_back(reduce); });
  for (ReduceOp reduce : reductions) {
    if (reduce.getAxes().size() > 1)
      return reduce.emitOpError(
          "reduction blocking requires prior multi-axis normalization");
    if (reduce->getBlock()) {
      if (failed(neutralizeReductionTails(reduce, kernel)) ||
          failed(realizeReduce(reduce, kernel)))
        return failure();
    }
  }
  SmallVector<scf::ForOp> loops;
  kernel.walk<WalkOrder::PostOrder>(
      [&](scf::ForOp loop) { loops.push_back(loop); });
  bool changed = false;
  for (scf::ForOp loop : loops)
    changed |= hoistNestedReduction(loop, kernel);
  if (changed)
    eraseDeadPhysicalValues(kernel);
  return closeValueRelations(kernel);
}

} // namespace intent::gpu
