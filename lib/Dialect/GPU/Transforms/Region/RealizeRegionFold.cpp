#include "Intent/Dialect/GPU/Transforms/Region/Realization.h"
#include "Intent/Dialect/GPU/Analysis/Helpers.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
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
#include "RegionCoRealization.h"
#include "mlir/IR/Dominance.h"

using namespace mlir;

namespace intent::gpu {

namespace {
using namespace region;

LogicalResult realizeFold(RegionFoldOp fold, func::FuncOp kernel,
                          bool simplifyFirstSummary) {
  foldKnownRecordProjections(fold);
  forwardUnusedRecordFields(fold);
  if (!lookupParameter(kernel, fold.getSegment()))
    return fold.emitOpError("region-fold segment parameter is not declared");
  auto prepared = prepareRegionSources(fold.getSources(), fold.getAxis(), fold);
  if (failed(prepared))
    return fold.emitOpError(
        "region-fold source is not a sliceable unit-step physical value graph");
  SmallVector<SourcePlan> plans = std::move(*prepared);
  PhysicalProgramAnalysis physicalAnalysis(kernel);
  SmallVector<unsigned> sourceAxes(fold.getSources().size(), fold.getAxis());
  PhysicalLockstepTraversalFact traversal =
      physicalAnalysis.lockstepTraversal(fold.getSources(), sourceAxes);
  if (!traversal.isExact()) {
    InFlightDiagnostic diagnostic = fold.emitOpError(
        traversal.state == PhysicalLockstepState::Inconsistent
            ? "region sources have inconsistent physical traversals"
            : "region source traversal is not exactly known");
    for (Operation *blocker : traversal.blockers)
      diagnostic << "; blocker=" << blocker->getName();
    return failure();
  }
  SmallVector<FragmentType> sliceTypes;
  for (BlockArgument argument :
       fold.getSummarize().front().getArguments().take_front(
           fold.getSources().size())) {
    auto fragment = dyn_cast<FragmentType>(argument.getType());
    if (!fragment)
      return fold.emitOpError(
          "region-fold source slice has no physical fragment schema");
    sliceTypes.push_back(fragment);
  }

  SmallVector<AssumeInBoundsOp> sourceAssumptions;
  DominanceInfo dominance(kernel);
  kernel.walk([&](AssumeInBoundsOp assumption) {
    if (!dominance.properlyDominates(assumption.getOperation(),
                                    fold.getOperation()))
      return;
    for (const SourcePlan &plan : plans) {
      PhysicalRangeFact fact = physicalAnalysis.sourceRanges(
          assumption.getIndex(), plan.sourceIdentity);
      FailureOr<MakeRangeOp> range = queryExactLogicalRange(fact);
      if (failed(range) ||
          llvm::none_of(fact.roots, [&](MakeRangeOp root) {
            return llvm::any_of(plan.ranges, [&](MakeRangeOp planned) {
              return sameLogicalRange(planned, root);
            });
          }))
        continue;
      sourceAssumptions.push_back(assumption);
      break;
    }
  });

  MakeRangeOp master = traversal.authority;
  OpBuilder builder(fold);
  auto segment = materializeParameter(builder, fold.getLoc(), fold.getSegment());
  Location location = fold.getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
  Value logicalEnd = master.getLogicalStop();
  Value stop = builder.create<BinaryOp>(
      location, builder.getIndexType(), logicalEnd, master.getStart(),
      BinaryOperator::Subtract);
  PhysicalExprAttr sliceExtent = parameterExtent(fold.getSegment());
  SmallVector<Value> identities(
      fold.getIdentities()
          .begin(),
      fold.getIdentities()
          .end());
  ValueRange captures = fold.getCaptures();
  FailureOr<PredicatePartition> partition = predicatePartition(
      builder, fold, plans, master, stop, identities, segment.getResult());
  bool specializePredicatePrefix =
      succeeded(partition) && partition->prefixSpecializable;
  Value memberPredicate =
      physicalTailMembershipPredicate(fold, identities, fold.getSegment(), plans);
  Value scalarTailStep;
  PhysicalExprAttr scalarTailExtent;
  if (!memberPredicate) {
    scalarTailStep = builder.create<arith::ConstantIndexOp>(location, 1);
    scalarTailExtent = PhysicalExprAttr::get(
        fold.getContext(), PhysicalExprKind::Constant, 1,
        builder.getStringAttr(""), builder.getArrayAttr({}));
  }
  std::optional<SummaryEmptinessPlan> emptiness;
  if (memberPredicate)
    emptiness = summaryEmptinessPlan(fold, identities, memberPredicate);
  std::optional<OnlineRegionPlan> online = onlineRegionPlan(fold, emptiness);
  // A nonempty normalized-exponential summary contains exp(0), hence its
  // accumulated mass cannot be zero. Nonfinite scores produce NaN mass, for
  // which != 0 remains true. Only the empty identity has zero mass. Rebuild
  // validity at its uses instead of carrying it or peeling the first slice.
  // This proof concerns the accumulated mass, not a rescaled incoming slice,
  // whose contribution may legitimately underflow to zero.
  bool massRepresentsValidity = false;
  bool encodeEmptyMaximum = false;
  if (online) {
    auto identity = identities.front().getDefiningOp<MakeRecordOp>();
    massRepresentsValidity =
        identity &&
        isLiteralZeroProjection(
            identity.getFields()[online->summary.massField]) &&
        isLiteralZeroProjection(online->summary.mass.getIdentities().front());
    auto maximumCombine = queryBinaryCombine(online->summary.maximum.getCombine());
    encodeEmptyMaximum = massRepresentsValidity && maximumCombine &&
        maximumCombine->kind() == BinaryOperator::MaximumNum &&
        isLiteralZeroProjection(identity.getFields()[online->summary.maximumField]);
    for (unsigned field : {online->summary.maximumField, online->summary.massField,
                           online->summary.momentField})
      encodeEmptyMaximum &= cast<FragmentType>(identity.getFields()[field].getType())
                                .getElementType().isF32();
  }
  SmallVector<AdditiveRegionContract> additive =
      additiveRegionContracts(fold, identities);
  bool needsSummaryMapping = online || !additive.empty();

  bool bodyFailed = false;
  std::string failureReason;
  auto emitSummary = [&](OpBuilder &nested, Location nestedLocation,
                         Value offset,
                         bool fullSegment,
                         bool predicateIsTrue,
                         bool summaryIsNonempty,
                         IRMapping *summaryMapping)
      -> FailureOr<SmallVector<Value>> {
    SmallVector<Value> slices;
    Value segmentTail;
    IRMapping sliceMapping;
    SmallVector<std::shared_ptr<IRMapping>> sourceMappings;
    bool scalarTail = !fullSegment && !memberPredicate;
    // Without an identity proof, visit the remaining real members as complete
    // unit slices. Padding a custom summary could otherwise turn 0 * NaN into
    // a contribution that did not exist in the author's logical domain.
    Value physicalSegment = scalarTail ? scalarTailStep : segment.getResult();
    PhysicalExprAttr physicalExtent =
        scalarTail ? scalarTailExtent : sliceExtent;
    fullSegment |= scalarTail;
    if (failed(buildSourceSlices(nested, nestedLocation, plans, sliceTypes,
                                 offset,
                                 physicalSegment, physicalExtent,
                                 fullSegment, slices,
                                 segmentTail, sliceMapping, sourceMappings,
                                 fold, failureReason)))
      return failure();
    for (AssumeInBoundsOp assumption : sourceAssumptions) {
      SmallVector<Value> assertedIndices;
      for (const std::shared_ptr<IRMapping> &sourceMapping : sourceMappings) {
        Value index = sourceMapping->lookupOrNull(assumption.getIndex());
        if (!index || llvm::is_contained(assertedIndices, index))
          continue;
        assertedIndices.push_back(index);
        OpBuilder::InsertionGuard guard(nested);
        nested.setInsertionPointAfter(index.getDefiningOp());
        auto replacement = nested.create<AssumeInBoundsOp>(
            nestedLocation, index, assumption.getResource(),
            assumption.getAxis());
        if (Attribute origin = assumption->getAttr(originAttr))
          replacement->setAttr(originAttr, origin);
      }
    }
    SmallVector<Value> summarizeArguments(slices);
    summarizeArguments.append(captures.begin(), captures.end());
    IRMapping localSummaryMapping;
    IRMapping &mapping = summaryMapping ? *summaryMapping : localSummaryMapping;
    if (predicateIsTrue) {
      for (Value predicate : partition->allTruePredicates) {
        auto predicateType = dyn_cast<FragmentType>(predicate.getType());
        if (!predicateType) {
          failureReason = "range predicate is not a physical fragment";
          return failure();
        }
        Value truth = nested.create<arith::ConstantOp>(
            nestedLocation, nested.getI1Type(), nested.getBoolAttr(true));
        mapping.map(predicate,
                    nested.create<SplatOp>(nestedLocation, predicateType, truth));
      }
    }
    Value nonemptySource;
    Value nonemptyTarget;
    if (summaryIsNonempty && emptiness) {
      nonemptySource = emptiness->summarizeValidity;
      Type validityType =
          cast<TypeAttr>(emptiness->fullType.getFieldTypes()[
              emptiness->optionalField])
              .getValue();
      nonemptyTarget = allTrueValue(nested, nestedLocation, validityType);
    }
    return inlinePureRegion(nested, fold.getSummarize(), summarizeArguments,
                            failureReason, {}, {},
                            fullSegment ? Value() : memberPredicate,
                            fullSegment ? Value() : segmentTail,
                            nonemptySource, nonemptyTarget, &mapping);
  };
  auto emitLoop = [&](Value lower, Value upper, ValueRange initial,
                      bool fullSegment, bool predicateIsTrue) {
    Value step = !fullSegment && !memberPredicate ? scalarTailStep
                                                 : segment.getResult();
    auto buildBody = [&](OpBuilder &nested, Location nestedLocation, Value offset,
                         ValueRange carries) {
        IRMapping summaryMapping;
        FailureOr<SmallVector<Value>> summary = emitSummary(
            nested, nestedLocation, offset, fullSegment, predicateIsTrue,
            /*summaryIsNonempty=*/false,
            needsSummaryMapping ? &summaryMapping : nullptr);
        if (failed(summary)) {
          bodyFailed = true;
          return;
        }
        SmallVector<Value> combineArguments(carries.begin(), carries.end());
        if (massRepresentsValidity) {
          FailureOr<Value> restored = restoreOnlineRecord(
              nested, nestedLocation, carries.front(), *emptiness,
              online->summary.massField);
          if (failed(restored)) {
            failureReason = "online mass has no exact validity projection";
            bodyFailed = true;
            return;
          }
          combineArguments.front() = *restored;
        }
        combineArguments.append(summary->begin(), summary->end());
        IRMapping mergeMapping;
        FailureOr<SmallVector<Value>> combined = inlinePureRegion(
            nested, fold.getCombine(), combineArguments, failureReason,
            {}, {}, {}, {}, {}, {},
            needsSummaryMapping ? &mergeMapping : nullptr);
        if (failed(combined)) {
          bodyFailed = true;
          return;
        }
        if (online) {
          FailureOr<Value> direct = coRealizeOnlineRegion(
              nested, nestedLocation, *online, summaryMapping, mergeMapping,
              encodeEmptyMaximum ? combineArguments.front() : Value());
          if (failed(direct) && encodeEmptyMaximum) {
            failureReason = "encoded online maximum could not close its merge graph";
            bodyFailed = true;
            return;
          }
          if (succeeded(direct))
            combined->front() = *direct;
        }
        coRealizeAdditiveRegion(nested, nestedLocation, additive,
                               summaryMapping, mergeMapping);
        if (massRepresentsValidity) {
          FailureOr<Value> payload = stripOptionalRecord(
              nested, nestedLocation, combined->front(), *emptiness);
          if (failed(payload)) {
            bodyFailed = true;
            return;
          }
          combined->front() = *payload;
        }
        nested.create<scf::YieldOp>(nestedLocation, *combined);
      };
    auto loop = createTraversalLoop(builder, location, lower, upper, step,
                                    initial, buildBody);
    if (Attribute origin = fold->getAttr(originAttr))
      loop->setAttr(originAttr, origin);
    return loop;
  };

  auto finish = [&]() -> LogicalResult {
    for (AssumeInBoundsOp assumption : sourceAssumptions)
      assumption.erase();
    simplifyKnownRecordValues(kernel);
    eraseDeadPhysicalValues(kernel);
    return success();
  };
  auto finishResults = [&](ValueRange physicalResults) -> LogicalResult {
    SmallVector<Value> results(physicalResults.begin(), physicalResults.end());
    if (massRepresentsValidity) {
      FailureOr<Value> restored = restoreOnlineRecord(
          builder, location, results.front(), *emptiness,
          online->summary.massField,
          encodeEmptyMaximum ? std::optional<unsigned>(online->summary.maximumField)
                             : std::nullopt);
      if (failed(restored))
        return fold.emitOpError("online result has no validity projection");
      results.front() = *restored;
    }
    for (auto [oldResult, newResult] : llvm::zip(fold.getResults(), results))
      oldResult.replaceAllUsesWith(newResult);
    fold.erase();
    return finish();
  };

  if (segment.getDeclaration().isDeferred() && memberPredicate) {
    FailureOr<SmallVector<Value>> summary =
        emitSummary(builder, location, zero, /*fullSegment=*/false,
                    /*predicateIsTrue=*/false,
                    /*summaryIsNonempty=*/false,
                    /*summaryMapping=*/nullptr);
    if (failed(summary))
      return fold.emitOpError("region-fold physicalization failed: ")
             << failureReason;
    for (auto [oldResult, newResult] :
         llvm::zip(fold.getResults(), *summary))
      oldResult.replaceAllUsesWith(newResult);
    fold.erase();
    return finish();
  }

  if (emptiness && !massRepresentsValidity && specializePredicatePrefix &&
      partition->firstMemberIsActive) {
    stop = partition->effectiveStop;
    Value nonempty = builder.create<CompareOp>(
        location, builder.getI1Type(), stop, zero, ComparePredicate::Gt);
    auto conditional = builder.create<scf::IfOp>(
        location, TypeRange{emptiness->payloadType}, nonempty,
        /*withElseRegion=*/true);
    auto prepareBranch = [](Region &region) {
      Block &block = region.front();
      if (!block.empty() && isa<scf::YieldOp>(block.back()))
        block.back().erase();
      return OpBuilder(&block, block.end());
    };

    OpBuilder nonemptyBuilder = prepareBranch(conditional.getThenRegion());
    FailureOr<SmallVector<Value>> first = emitSummary(
        nonemptyBuilder, location, zero, /*fullSegment=*/false,
        /*predicateIsTrue=*/false, /*summaryIsNonempty=*/true,
        /*summaryMapping=*/nullptr);
    if (succeeded(first) && first->size() != 1) {
      failureReason =
          "summary-emptiness realization requires one summary record";
      bodyFailed = true;
    }
    FailureOr<Value> payload = failure();
    if (succeeded(first) && !bodyFailed) {
      if (simplifyFirstSummary) {
        // The fold's declared identity is neutral for the complete summary
        // tuple. Forward the summary already materialized in this branch.
        payload = stripOptionalRecord(nonemptyBuilder, location, first->front(), *emptiness);
      } else {
        SmallVector<Value> arguments(identities);
        llvm::append_range(arguments, *first);
        auto combined = inlinePureRegion(nonemptyBuilder, fold.getCombine(), arguments, failureReason);
        if (succeeded(combined) && combined->size() == 1)
          payload = stripOptionalRecord(nonemptyBuilder, location, combined->front(), *emptiness);
      }
    }
    if (failed(first) || failed(payload))
      bodyFailed = true;

    if (!bodyFailed)
      nonemptyBuilder.create<scf::YieldOp>(location, *payload);
    OpBuilder emptyBuilder = prepareBranch(conditional.getElseRegion());
    FailureOr<Value> emptyPayload = stripOptionalRecord(
        emptyBuilder, location, identities.front(), *emptiness);
    if (failed(emptyPayload))
      bodyFailed = true;
    else
      emptyBuilder.create<scf::YieldOp>(location, *emptyPayload);
    if (bodyFailed) {
      conditional.erase();
      return fold.emitOpError("region-fold physicalization failed: ")
             << failureReason;
    }

    auto emitPayloadLoop = [&](Value lower, Value upper, Value initial,
                               bool fullSegment,
                               bool predicateIsTrue,
                               bool summaryIsNonempty) -> scf::ForOp {
      auto buildBody = [&](OpBuilder &nested, Location nestedLocation, Value offset,
                           ValueRange carries) {
            IRMapping summaryMapping;
            FailureOr<SmallVector<Value>> summary = emitSummary(
                nested, nestedLocation, offset, fullSegment, predicateIsTrue,
                summaryIsNonempty, needsSummaryMapping ? &summaryMapping
                                                      : nullptr);
            if (failed(summary) || summary->size() != 1 ||
                carries.size() != 1) {
              if (succeeded(summary))
                failureReason =
                    "summary-emptiness loop requires one summary and one payload carry";
              bodyFailed = true;
              return;
            }
            Value fullCarry = restoreOptionalRecord(
                nested, nestedLocation, carries.front(), *emptiness);
            IRMapping mergeMapping;
            FailureOr<SmallVector<Value>> combined = inlinePureRegion(
                nested, fold.getCombine(),
                ValueRange{fullCarry, summary->front()}, failureReason,
                {}, {}, {}, {}, {}, {},
                needsSummaryMapping ? &mergeMapping : nullptr);
            if (failed(combined) || combined->size() != 1) {
              if (succeeded(combined))
                failureReason =
                    "summary-emptiness combine requires one summary record";
              bodyFailed = true;
              return;
            }
            Value combinedValue = combined->front();
            if (online) {
              FailureOr<Value> direct = coRealizeOnlineRegion(
                  nested, nestedLocation, *online, summaryMapping,
                  mergeMapping);
              if (succeeded(direct))
                combinedValue = *direct;
            }
            coRealizeAdditiveRegion(nested, nestedLocation, additive,
                                   summaryMapping, mergeMapping);
            FailureOr<Value> next = stripOptionalRecord(
                nested, nestedLocation, combinedValue, *emptiness);
            if (failed(next)) {
              bodyFailed = true;
              return;
            }
            nested.create<scf::YieldOp>(nestedLocation, *next);
          };
      auto loop = createTraversalLoop(
          builder, location, lower, upper, segment.getResult(),
          ValueRange{initial}, buildBody);
      if (Attribute origin = fold->getAttr(originAttr))
        loop->setAttr(originAttr, origin);
      return loop;
    };

    scf::ForOp allTrueLoop;
    scf::ForOp fullMixedLoop;
    scf::ForOp tailLoop;
    Value current = conditional.getResult(0);
    allTrueLoop = emitPayloadLoop(
        segment.getResult(), partition->allTrueStop, current,
        /*fullSegment=*/true, /*predicateIsTrue=*/true,
        /*summaryIsNonempty=*/true);
    current = allTrueLoop.getResult(0);
    Value mixedStart = builder.create<BinaryOp>(
        location, builder.getIndexType(), partition->allTrueStop,
        segment.getResult(), BinaryOperator::Maximum);
    Value fullMixedSegments = builder.create<BinaryOp>(
        location, builder.getIndexType(), stop, segment.getResult(),
        BinaryOperator::FloorDivide);
    Value fullMixedStop = builder.create<BinaryOp>(
        location, builder.getIndexType(), fullMixedSegments,
        segment.getResult(), BinaryOperator::Multiply);
    if (!bodyFailed) {
      fullMixedLoop = emitPayloadLoop(
          mixedStart, fullMixedStop, current, /*fullSegment=*/true,
          /*predicateIsTrue=*/false,
          /*summaryIsNonempty=*/false);
      current = fullMixedLoop.getResult(0);
    }
    Value tailStart = builder.create<BinaryOp>(
        location, builder.getIndexType(), fullMixedStop,
        segment.getResult(), BinaryOperator::Maximum);
    if (!bodyFailed) {
      tailLoop = emitPayloadLoop(
          tailStart, stop, current, /*fullSegment=*/false,
          /*predicateIsTrue=*/false,
          /*summaryIsNonempty=*/false);
      current = tailLoop.getResult(0);
    }
    if (bodyFailed) {
      if (allTrueLoop && allTrueLoop->getBlock())
        allTrueLoop.erase();
      if (fullMixedLoop && fullMixedLoop->getBlock())
        fullMixedLoop.erase();
      if (tailLoop && tailLoop->getBlock())
        tailLoop.erase();
      return fold.emitOpError("region-fold physicalization failed: ")
             << failureReason;
    }
    Type validityType = cast<TypeAttr>(emptiness->fullType.getFieldTypes()[
                                           emptiness->optionalField])
                            .getValue();
    Value validity = nonempty;
    if (isa<FragmentType>(validityType))
      validity = builder.create<SplatOp>(location, validityType, validity);
    Value result = restoreOptionalRecord(builder, location, current, *emptiness,
                                         validity);
    fold.getResult(0).replaceAllUsesWith(result);
    fold.erase();
    return finish();
  }

  SmallVector<Value> current(identities.begin(), identities.end());
  if (massRepresentsValidity) {
    FailureOr<Value> payload = stripOptionalRecord(
        builder, location, current.front(), *emptiness);
    if (failed(payload))
      return fold.emitOpError("online identity has no payload projection");
    current.front() = *payload;
    if (encodeEmptyMaximum) {
      unsigned field = online->summary.maximumField -
                       (emptiness->optionalField < online->summary.maximumField);
      auto record = current.front().getDefiningOp<MakeRecordOp>();
      OpBuilder::InsertionGuard insertion(builder);
      builder.setInsertionPoint(record);
      Type type = record.getFields()[field].getType();
      Value negativeInfinity = builder.create<arith::ConstantOp>(
          location, FloatAttr::get(builder.getF32Type(),
                                   APFloat::getInf(APFloat::IEEEsingle(), true)));
      Value encoded = builder.create<SplatOp>(location, type, negativeInfinity);
      record->setOperand(field, encoded);
    }
  }
  scf::ForOp allTrueLoop;
  if (specializePredicatePrefix) {
    allTrueLoop = emitLoop(zero, partition->allTrueStop, current,
                           /*fullSegment=*/true,
                           /*predicateIsTrue=*/true);
    if (!bodyFailed)
      current.assign(allTrueLoop.getResults().begin(),
                     allTrueLoop.getResults().end());
  }
  Value mixedStart = specializePredicatePrefix
                         ? partition->allTrueStop
                         : succeeded(partition) ? partition->effectiveStart
                                                : zero;
  stop = succeeded(partition) ? partition->effectiveStop : stop;
  Value fullMixedSegments = builder.create<BinaryOp>(
      location, builder.getIndexType(), stop, segment.getResult(),
      BinaryOperator::FloorDivide);
  Value fullMixedStop = builder.create<BinaryOp>(
      location, builder.getIndexType(), fullMixedSegments, segment.getResult(),
      BinaryOperator::Multiply);
  scf::ForOp leadingMixedLoop;
  if (!specializePredicatePrefix && succeeded(partition)) {
    Value boundedStart = builder.create<BinaryOp>(
        location, builder.getIndexType(), mixedStart, partition->allTrueStart,
        BinaryOperator::Maximum);
    Value interiorStart = builder.create<BinaryOp>(
        location, builder.getIndexType(), fullMixedStop, boundedStart,
        BinaryOperator::Minimum);
    Value boundedStop = builder.create<BinaryOp>(
        location, builder.getIndexType(), interiorStart, partition->allTrueStop,
        BinaryOperator::Maximum);
    Value interiorStop = builder.create<BinaryOp>(
        location, builder.getIndexType(), fullMixedStop, boundedStop,
        BinaryOperator::Minimum);
    leadingMixedLoop = emitLoop(mixedStart, interiorStart, current,
                                /*fullSegment=*/true,
                                /*predicateIsTrue=*/false);
    if (!bodyFailed) {
      current.assign(leadingMixedLoop.getResults().begin(),
                     leadingMixedLoop.getResults().end());
      allTrueLoop = emitLoop(interiorStart, interiorStop, current,
                             /*fullSegment=*/true,
                             /*predicateIsTrue=*/true);
    }
    if (!bodyFailed)
      current.assign(allTrueLoop.getResults().begin(),
                     allTrueLoop.getResults().end());
    mixedStart = interiorStop;
  }
  scf::ForOp fullMixedLoop;
  if (!bodyFailed)
    fullMixedLoop = emitLoop(mixedStart, fullMixedStop, current,
                             /*fullSegment=*/true,
                             /*predicateIsTrue=*/false);
  if (!bodyFailed)
    current.assign(fullMixedLoop.getResults().begin(),
                   fullMixedLoop.getResults().end());
  scf::ForOp tailLoop;
  if (!bodyFailed)
    tailLoop = emitLoop(fullMixedStop, stop, current,
                        /*fullSegment=*/false,
                        /*predicateIsTrue=*/false);
  if (bodyFailed) {
    if (leadingMixedLoop && leadingMixedLoop->getBlock())
      leadingMixedLoop.erase();
    if (allTrueLoop && allTrueLoop->getBlock())
      allTrueLoop.erase();
    if (fullMixedLoop && fullMixedLoop->getBlock())
      fullMixedLoop.erase();
    if (tailLoop && tailLoop->getBlock())
      tailLoop.erase();
    return fold.emitOpError("region-fold physicalization failed: ")
           << failureReason;
  }
  return finishResults(tailLoop.getResults());
}

} // namespace

LogicalResult realizeRegionFolds(ModuleOp module, bool simplifyFirstSummary) {
  FailureOr<func::FuncOp> physicalKernel = getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  func::FuncOp kernel = *physicalKernel;
  SmallVector<RegionFoldOp> folds;
  kernel.walk([&](RegionFoldOp fold) { folds.push_back(fold); });
  for (RegionFoldOp fold : folds)
    if (fold->getBlock() && failed(realizeFold(fold, kernel, simplifyFirstSummary)))
      return failure();
  return closeValueRelations(kernel);
}

} // namespace intent::gpu
