#include "Intent/Dialect/GPU/Transforms/Region/Realization.h"
#include "RegionCloning.h"
#include "RegionPredicates.h"
#include "RegionSources.h"
#include "RegionSummary.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Transforms/Control/Traversal.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;

namespace intent::gpu {

namespace {
using namespace region;

LogicalResult realizeScan(RegionScanOp scan, func::FuncOp kernel) {
  foldKnownRecordProjections(scan);
  if (!lookupParameter(kernel, scan.getSegment()))
    return scan.emitOpError("region-scan segment parameter is not declared");
  auto prepared = prepareRegionSources(scan.getSources(), scan.getAxis(), scan);
  if (failed(prepared))
    return scan.emitOpError(
        "region-scan source is not a sliceable unit-step physical value graph");
  SmallVector<SourcePlan> plans = std::move(*prepared);
  PhysicalProgramAnalysis physicalAnalysis(kernel);
  SmallVector<unsigned> sourceAxes(scan.getSources().size(), scan.getAxis());
  PhysicalLockstepTraversalFact traversal =
      physicalAnalysis.lockstepTraversal(scan.getSources(), sourceAxes);
  if (!traversal.isExact()) {
    InFlightDiagnostic diagnostic = scan.emitOpError(
        traversal.state == PhysicalLockstepState::Inconsistent
            ? "region sources have inconsistent physical traversals"
            : "region source traversal is not exactly known");
    for (Operation *blocker : traversal.blockers)
      diagnostic << "; blocker=" << blocker->getName();
    return failure();
  }
  SmallVector<FragmentType> sliceTypes;
  for (BlockArgument argument :
       scan.getSummarize().front().getArguments().take_front(
           scan.getSources().size())) {
    auto fragment = dyn_cast<FragmentType>(argument.getType());
    if (!fragment)
      return scan.emitOpError(
          "region-scan source slice has no physical fragment schema");
    sliceTypes.push_back(fragment);
  }
  SmallVector<SliceRelation> outputRelations;
  for (auto [plan, sliceType] : llvm::zip(plans, sliceTypes)) {
    auto mapping = cast<AxisMapAttr>(
        sliceType.getAxisMaps()[plan.sourceAxis]);
    auto relation = llvm::find_if(
        outputRelations, [&](const SliceRelation &candidate) {
          return candidate.source == plan.sourceIdentity;
        });
    if (relation == outputRelations.end()) {
      outputRelations.push_back({plan.sourceIdentity, mapping});
      continue;
    }
    if (relation->segmentMapping != mapping)
      return scan.emitOpError(
          "region-scan source has inconsistent helper-local slice relations");
  }
  MakeRangeOp master = traversal.authority;
  ValueRange identities = scan.getIdentities();
  if (!scanTailIsIdentity(scan, plans, identities))
    return scan.emitOpError(
        "region-scan summarizer does not prove that zero-filled physical tail members produce its transition identity");

  SmallVector<Operation *> outputConsumers;
  std::string failureReason;
  if (failed(collectScanOutputConsumers(scan, outputConsumers, failureReason)))
    return scan.emitOpError("region-scan physicalization failed: ")
           << failureReason;

  ValueRange initialStates = scan.getInitialStates();
  ValueRange captures = scan.getCaptures();
  OpBuilder builder(scan);
  auto segment = materializeParameter(builder, scan.getLoc(), scan.getSegment());
  Location location = scan.getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
  Value logicalEnd = master.getLogicalStop();
  Value traversalExtent = builder.create<BinaryOp>(
      location, builder.getIndexType(), logicalEnd, master.getStart(),
      BinaryOperator::Subtract);
  PhysicalExprAttr sliceExtent = parameterExtent(scan.getSegment());
  bool bodyFailed = false;
  auto buildBody = [&](OpBuilder &nested, Location nestedLocation, Value offset,
                       ValueRange prefix) {
        SmallVector<Value> slices;
        Value segmentTail;
        IRMapping sliceMapping;
        SmallVector<std::shared_ptr<IRMapping>> sourceMappings;
        if (failed(buildSourceSlices(
                nested, nestedLocation, plans, sliceTypes, offset,
                segment.getResult(),
                sliceExtent, /*fullSegment=*/false, slices, segmentTail,
                sliceMapping, sourceMappings,
                scan, failureReason))) {
          bodyFailed = true;
          return;
        }
        SmallVector<Value> summarizeArguments(slices);
        summarizeArguments.append(captures.begin(), captures.end());
        FailureOr<SmallVector<Value>> summary = inlinePureRegion(
            nested, scan.getSummarize(), summarizeArguments, failureReason);
        if (failed(summary)) {
          bodyFailed = true;
          return;
        }
        SmallVector<Value> applyArguments(prefix.begin(), prefix.end());
        applyArguments.append(initialStates.begin(), initialStates.end());
        FailureOr<SmallVector<Value>> incoming = inlinePureRegion(
            nested, scan.getApply(), applyArguments, failureReason);
        if (failed(incoming)) {
          bodyFailed = true;
          return;
        }
        SmallVector<Value> emitArguments(slices);
        emitArguments.append(incoming->begin(), incoming->end());
        emitArguments.append(captures.begin(), captures.end());
        FailureOr<SmallVector<Value>> emitted = inlinePureRegion(
            nested, scan.getEmit(), emitArguments, failureReason);
        if (failed(emitted)) {
          bodyFailed = true;
          return;
        }
        for (auto [output, slice] : llvm::zip(
                 scan.getEmittedResults(), *emitted))
          sliceMapping.map(output, slice);
        if (failed(cloneScanOutputConsumers(
                nested, nestedLocation, outputConsumers, sliceMapping,
                outputRelations, sliceExtent, segmentTail,
                scan, offset, segment.getResult(), failureReason))) {
          bodyFailed = true;
          return;
        }
        SmallVector<Value> combineArguments(prefix.begin(), prefix.end());
        combineArguments.append(summary->begin(), summary->end());
        FailureOr<SmallVector<Value>> combined = inlinePureRegion(
            nested, scan.getCombine(), combineArguments, failureReason);
        if (failed(combined)) {
          bodyFailed = true;
          return;
        }
        nested.create<scf::YieldOp>(nestedLocation, *combined);
      };
  auto loop = createTraversalLoop(builder, location, zero, traversalExtent,
                                  segment.getResult(), identities, buildBody);
  if (bodyFailed) {
    if (loop->getBlock())
      loop.erase();
    return scan.emitOpError("region-scan physicalization failed: ")
           << failureReason;
  }
  if (Attribute origin = scan->getAttr(originAttr))
    loop->setAttr(originAttr, origin);

  SmallVector<Value> finalArguments(loop.getResults().begin(),
                                    loop.getResults().end());
  finalArguments.append(initialStates.begin(), initialStates.end());
  FailureOr<SmallVector<Value>> finalStates = inlinePureRegion(
      builder, scan.getApply(), finalArguments, failureReason);
  if (failed(finalStates)) {
    loop.erase();
    return scan.emitOpError("region-scan final-state physicalization failed: ")
           << failureReason;
  }
  for (auto [oldResult, newResult] : llvm::zip(
           scan.getFinalStates(), *finalStates))
    oldResult.replaceAllUsesWith(newResult);
  for (Operation *consumer : llvm::reverse(outputConsumers))
    consumer->erase();
  scan.erase();
  eraseDeadPhysicalValues(kernel);
  return success();
}

} // namespace

LogicalResult realizeRegionScans(ModuleOp module) {
  FailureOr<func::FuncOp> physicalKernel = getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  func::FuncOp kernel = *physicalKernel;
  SmallVector<RegionScanOp> scans;
  kernel.walk([&](RegionScanOp scan) { scans.push_back(scan); });
  for (RegionScanOp scan : scans)
    if (scan->getBlock() && failed(realizeScan(scan, kernel)))
      return failure();
  return closeValueRelations(kernel);
}

} // namespace intent::gpu
