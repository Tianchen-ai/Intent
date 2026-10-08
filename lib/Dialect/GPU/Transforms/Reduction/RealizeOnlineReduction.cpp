#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Analysis/Helpers.h"
#include "Intent/Dialect/GPU/Transforms/Control/Traversal.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/Intent/IR/CompileOptions.h"

#include "OnlineSummary.h"

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Dominance.h"

using namespace mlir;

namespace intent::gpu {
namespace {

struct OnlineSummaryPattern : NormalizedSummaryStructure {
  OnlineSummaryPattern(NormalizedSummaryStructure structure, MakeRangeOp authority,
                       SmallVector<MakeRangeOp> ranges)
      : NormalizedSummaryStructure(std::move(structure)), authority(authority),
        ranges(std::move(ranges)) {}

  MakeRangeOp authority;
  SmallVector<MakeRangeOp> ranges;
};

PhysicalExprAttr parameterExpression(MLIRContext *context, StringRef name) {
  return PhysicalExprAttr::get(
      context, PhysicalExprKind::Parameter, 0,
      ParameterRefAttr::get(context, StringAttr::get(context, name)),
      ArrayAttr::get(context, {}));
}

FragmentType withElementType(FragmentType type, Type elementType) {
  return FragmentType::get(type.getContext(), elementType, type.getShape(),
                           type.getAxisMaps(), type.getValidity(),
                           type.getOwner());
}

FragmentType replaceExtent(FragmentType type, unsigned axis,
                           PhysicalExprAttr extent) {
  SmallVector<Attribute> shape(type.getShape().begin(), type.getShape().end());
  shape[axis] = extent;
  return FragmentType::get(type.getContext(), type.getElementType(),
                           ArrayAttr::get(type.getContext(), shape),
                           type.getAxisMaps(), type.getValidity(),
                           type.getOwner());
}

void inheritRangeAuthority(Operation *target, MakeRangeOp source) {
  for (StringRef name :
       {originAttr, sourceSubregionAttr, sourceSubregionBoundAttr,
        worksetCoordinateRangeAttr})
    if (Attribute value = source->getAttr(name))
      target->setAttr(name, value);
}

Value scalarValue(Value value) {
  while (value) {
    if (auto splat = value.getDefiningOp<SplatOp>()) {
      value = splat.getValue();
      continue;
    }
    if (auto broadcast = value.getDefiningOp<BroadcastOp>()) {
      value = broadcast.getValue();
      continue;
    }
    return value;
  }
  return {};
}

FailureOr<OnlineSummaryPattern> matchOnlineSummary(ContractOp moment,
                                                   func::FuncOp kernel,
                                                   StringRef &reason) {
  auto group = dyn_cast<ExecutionGroupOp>(moment->getParentOp());
  if (!group || group->getBlock() != &kernel.front() ||
      moment->getBlock() != &group.getBody().front())
    return failure();
  FailureOr<NormalizedSummaryStructure> summary =
      matchNormalizedSummaryStructure(moment);
  if (failed(summary))
    return failure();
  DominanceInfo dominance(kernel);
  if (!llvm::all_of(summary->mass.getResult(0).getUsers(), [&](Operation *user) {
        return dominance.dominates(moment.getOperation(), user);
      })) {
    reason = "mass is consumed before the complete normalized summary is available";
    return failure();
  }

  PhysicalProgramAnalysis analysis(kernel);
  SmallVector<MakeRangeOp> ranges;
  for (auto [value, axis] :
       {std::pair<Value, unsigned>{summary->memberValidity,
                                   summary->reductionAxis},
        std::pair<Value, unsigned>{summary->score, summary->reductionAxis},
        std::pair<Value, unsigned>{summary->values,
                                   summary->valueReductionAxis}}) {
    PhysicalRangeFact fact = analysis.axisRanges(value, axis);
    if (fact.roots.empty()) {
      reason = "source axes have no exact current physical range";
      return failure();
    }
    for (MakeRangeOp range : fact.roots)
      if (!llvm::is_contained(ranges, range))
        ranges.push_back(range);
  }
  PhysicalLockstepTraversalFact lockstep = analysis.lockstepRanges(ranges);
  if (!lockstep.isExact() || !lockstep.authority ||
      !llvm::all_of(ranges, [&](MakeRangeOp range) {
        return sourceAxisIdentity(range) == summary->traversal &&
               isUnitStepRange(range);
      })) {
    reason = "sources do not share a unit-step lockstep traversal";
    return failure();
  }
  for (Value value :
       {summary->memberValidity, summary->score, summary->values})
    if (!analysis
             .replayability(value, summary->traversal,
                            PhysicalReplayScope::ValueGraph,
                            /*allowAccesses=*/true)
             .isReplayable()) {
      reason = "source graph cannot be replayed with the same coordinates and read effects";
      return failure();
    }

  return OnlineSummaryPattern(std::move(*summary), lockstep.authority,
                              std::move(ranges));
}

std::optional<StringRef> unsupportedNumerics(OnlineSummaryPattern &pattern) {
  auto element = [](Value value) { return cast<FragmentType>(value.getType()).getElementType(); };
  if (!element(pattern.score).isF32() ||
      !element(pattern.maximum.getResult(0)).isF32() ||
      !element(pattern.mass.getResult(0)).isF32() ||
      !element(pattern.moment.getResult()).isF32())
    return "requires f32 scores, maximum, mass and moment accumulation";
  Type weight = element(pattern.moment.getLhs());
  if ((!weight.isF16() && !weight.isBF16() && !weight.isF32()) ||
      element(pattern.values) != weight)
    return "requires matching f16, bf16 or f32 weight/value operands";
  auto identity = scalarValue(pattern.mass.getIdentities().front()).getDefiningOp<arith::ConstantOp>();
  auto zero = identity ? dyn_cast<FloatAttr>(identity.getValue()) : FloatAttr{};
  if (!zero || !zero.getValue().isZero())
    return "requires an additive-zero mass identity";
  return std::nullopt;
}

LogicalResult realizeOnlineSummary(OnlineSummaryPattern pattern,
                                   func::FuncOp kernel) {
  auto scoreType = cast<FragmentType>(pattern.score.getType());
  std::string parameterName =
      ("REDUCE_CHUNK_" + Twine(pattern.traversal.sourceId) + "_A" +
       Twine(pattern.traversal.sourceAxis) +
       (pattern.traversal.derived ? "_DERIVED" : ""))
          .str();
  SmallVector<int64_t> candidates{8, 16, 32, 64, 128,
                                  256, 512, 1024, 2048, 4096};
  auto chunkReference = getOrCreatePhysicalParameter(
      kernel, parameterName, ParameterRole::Reduction,
      ParameterCategory::Reduction,
      scoreType.getElementType().getIntOrFloatBitWidth(), candidates);
  if (failed(chunkReference))
    return pattern.moment.emitOpError(
        "online reduction has no physical traversal parameter");
  auto declaration = lookupParameter(kernel, *chunkReference);
  auto binding = declaration.getBinding().withSource(PhysicalSourceAttr::get(
      kernel.getContext(), pattern.traversal.sourceId,
      pattern.traversal.sourceAxis, pattern.traversal.derived));
  if (FailureOr<int64_t> dimension = queryRangeDimension(pattern.authority);
      succeeded(dimension))
    binding = binding.withDimension(IntegerAttr::get(
        IntegerType::get(kernel.getContext(), 64), *dimension));
  if (failed(updateParameter(kernel, declaration.withBinding(binding))))
    return failure();
  OpBuilder parameterBuilder(&kernel.front(), kernel.front().begin());
  auto chunk = materializeParameter(parameterBuilder, pattern.moment.getLoc(), *chunkReference);
  PhysicalExprAttr chunkExtent = parameterExpression(
      pattern.moment.getContext(), chunkReference->getName().getValue());

  OpBuilder builder(pattern.moment);
  Location location = pattern.moment.getLoc();
  Value validIdentity = pattern.validity.getIdentities().front();
  Value maximumIdentity = builder.create<SplatOp>(
      location, pattern.maximum.getResult(0).getType(),
      scalarValue(pattern.maximumOrEmpty.getFalseValue()));
  Value maximumReductionIdentity = pattern.maximum.getIdentities().front();
  Value massIdentity = pattern.mass.getIdentities().front();
  Value momentIdentity = pattern.moment.getAccumulator();
  SmallVector<Value> initials{validIdentity, maximumReductionIdentity,
                              massIdentity, momentIdentity};
  bool bodyFailed = false;
  std::string bodyFailure;
  auto loop = createTraversalLoop(
      builder, location, pattern.authority.getStart(), pattern.authority.getLogicalStop(),
      chunk.getResult(), initials,
      [&](OpBuilder &nested, Location nestedLocation, Value chunkStart,
          ValueRange carries) {
        IRMapping mapping;
        ReplayMaterializationOptions options;
        options.traversalRanges = pattern.ranges;
        options.materializeZeroFill = true;
        for (MakeRangeOp range : pattern.ranges) {
          auto original = cast<FragmentType>(range.getResult().getType());
          FragmentType blocked = replaceExtent(original, 0, chunkExtent);
          auto coordinate = nested.create<MakeRangeOp>(
              nestedLocation, blocked, chunkStart, chunk.getResult(),
              range.getStep(), range.getLogicalStart(), range.getLogicalStop(),
              range.getSourceId(), range.getSourceAxis(), range.getDerived());
          inheritRangeAuthority(coordinate, range);
          mapping.map(range.getResult(), coordinate.getResult());
        }
        Value authority = mapping.lookupOrNull(pattern.authority.getResult());
        if (!authority) {
          bodyFailed = true;
          bodyFailure = "lockstep traversal authority was not rematerialized";
          return;
        }
        auto authorityType = cast<FragmentType>(authority.getType());
        Value end = nested.create<BroadcastOp>(
            nestedLocation, authorityType, pattern.authority.getLogicalStop());
        Value tail = nested.create<CompareOp>(
            nestedLocation,
            withElementType(authorityType, nested.getI1Type()), authority, end,
            ComparePredicate::Lt);
        options.segmentTail = tail;

        auto replay = [&](Value value) -> FailureOr<Value> {
          return materializeReplayedValue(
              nested, nestedLocation, value, pattern.traversal, chunkExtent,
              mapping, pattern.moment, options);
        };
        FailureOr<Value> replayedValidity = replay(pattern.memberValidity);
        FailureOr<Value> replayedScore = replay(pattern.score);
        FailureOr<Value> replayedValues = replay(pattern.values);
        if (failed(replayedValidity) || failed(replayedScore) ||
            failed(replayedValues)) {
          bodyFailed = true;
          bodyFailure = "typed member graph could not be replayed once";
          return;
        }
        auto blockedValidityType =
            cast<FragmentType>((*replayedValidity).getType());
        FailureOr<Value> projectedTail = projectPredicateToFragment(
            nested, nestedLocation, tail, blockedValidityType,
            pattern.traversal);
        if (failed(projectedTail)) {
          bodyFailed = true;
          bodyFailure = "traversal tail has no member-predicate projection";
          return;
        }
        Value blockedValidity = nested.create<BinaryOp>(
            nestedLocation, blockedValidityType, *replayedValidity,
            *projectedTail, BinaryOperator::LogicalAnd);
        mapping.map(pattern.memberValidity, blockedValidity);

        auto blockedScoreType = cast<FragmentType>((*replayedScore).getType());
        Value negativeInfinity = nested.create<SplatOp>(
            nestedLocation, blockedScoreType,
            scalarValue(pattern.maskedScore.getFalseValue()));
        Value replayedMasked = nested.create<SelectOp>(
            nestedLocation, blockedScoreType, blockedValidity, *replayedScore,
            negativeInfinity);
        ReduceOp chunkValidity = cloneReductionWithSource(
            nested, nestedLocation, pattern.validity, blockedValidity);
        ReduceOp chunkMaximum = cloneReductionWithSource(
            nested, nestedLocation, pattern.maximum, replayedMasked);
        Value currentMaximum = nested.create<SelectOp>(
            nestedLocation, pattern.maximumOrEmpty.getResult().getType(),
            chunkValidity.getResult(0), chunkMaximum.getResult(0),
            maximumIdentity);
        Value broadcastMaximum = nested.create<BroadcastOp>(
            nestedLocation, blockedScoreType, currentMaximum);
        Value shiftedScore = nested.create<BinaryOp>(
            nestedLocation, blockedScoreType, replayedMasked,
            broadcastMaximum, BinaryOperator::Subtract);
        Value unmaskedProbability = nested.create<UnaryOp>(
            nestedLocation, blockedScoreType, shiftedScore,
            pattern.exponential.getOperatorKind(),
            pattern.exponential.getApproximate(), pattern.exponential.getFlushToZero());
        Value probabilityZero = nested.create<SplatOp>(
            nestedLocation, blockedScoreType,
            scalarValue(pattern.probability.getFalseValue()));
        Value replayedProbability = nested.create<SelectOp>(
            nestedLocation, blockedScoreType, blockedValidity,
            unmaskedProbability, probabilityZero);
        ReduceOp chunkMass = cloneReductionWithSource(
            nested, nestedLocation, pattern.mass, replayedProbability);
        auto originalWeightType =
            cast<FragmentType>(pattern.moment.getLhs().getType());
        FragmentType blockedWeightType = withElementType(
            blockedScoreType, originalWeightType.getElementType());
        Value replayedProbabilityCast = nested.create<CastOp>(
            nestedLocation, blockedWeightType, replayedProbability);
        Value chunkMoment = nested.create<ContractOp>(
            nestedLocation, pattern.moment.getResult().getType(),
            replayedProbabilityCast, *replayedValues, momentIdentity,
            pattern.moment.getLhsReductionAxes(),
            pattern.moment.getRhsReductionAxes(),
            pattern.moment.getLhsBatchAxes(),
            pattern.moment.getRhsBatchAxes());
        if (Attribute origin = pattern.moment->getAttr(originAttr))
          chunkMoment.getDefiningOp()->setAttr(originAttr, origin);
        {
          Value rowValidity = nested.create<BroadcastOp>(
              nestedLocation,
              withElementType(cast<FragmentType>(chunkMoment.getType()),
                              nested.getI1Type()),
              chunkValidity.getResult(0));
          chunkMoment = nested.create<SelectOp>(
              nestedLocation, chunkMoment.getType(), rowValidity,
              chunkMoment, momentIdentity);
        }

        Value bothValid = nested.create<BinaryOp>(
            nestedLocation, carries[0].getType(), carries[0],
            chunkValidity.getResult(0), BinaryOperator::LogicalOr);
        auto maximumCombine = *queryBinaryCombine(pattern.maximum.getCombine());
        Value arguments[] = {carries[1], currentMaximum};
        IRMapping maximumMapping;
        maximumMapping.map(maximumCombine.operation.getLhs(),
                           arguments[maximumCombine.arguments[0]]);
        maximumMapping.map(maximumCombine.operation.getRhs(),
                           arguments[maximumCombine.arguments[1]]);
        auto maximumOfBothOp = cast<BinaryOp>(nested.clone(
            *maximumCombine.operation, maximumMapping));
        maximumOfBothOp.getResult().setType(carries[1].getType());
        Value maximumOfBoth = maximumOfBothOp.getResult();
        Value maximumWithRight = nested.create<SelectOp>(
            nestedLocation, carries[1].getType(), chunkValidity.getResult(0),
            maximumOfBoth, carries[1]);
        Value combinedMaximum = nested.create<SelectOp>(
            nestedLocation, carries[1].getType(), carries[0], maximumWithRight,
            currentMaximum);

        auto scale = [&](Value present, Value previousMaximum) {
          Value finiteReference = nested.create<SelectOp>(
              nestedLocation, previousMaximum.getType(), present,
              previousMaximum, combinedMaximum);
          Value delta = nested.create<BinaryOp>(
              nestedLocation, carries[1].getType(), finiteReference,
              combinedMaximum, BinaryOperator::Subtract);
          Value exponential = nested.create<UnaryOp>(
              nestedLocation, carries[1].getType(), delta,
              pattern.exponential.getOperatorKind(),
              pattern.exponential.getApproximate(), pattern.exponential.getFlushToZero());
          return Value(nested.create<SelectOp>(
              nestedLocation, carries[1].getType(), present, exponential,
              massIdentity));
        };
        Value carryScale = scale(carries[0], carries[1]);
        Value chunkScale =
            scale(chunkValidity.getResult(0), currentMaximum);
        Value scaledCarryMass = nested.create<BinaryOp>(
            nestedLocation, carries[2].getType(), carries[2], carryScale,
            BinaryOperator::Multiply);
        Value scaledChunkMass = nested.create<BinaryOp>(
            nestedLocation, carries[2].getType(), chunkMass.getResult(0),
            chunkScale, BinaryOperator::Multiply);
        Value combinedMass = nested.create<BinaryOp>(
            nestedLocation, carries[2].getType(), scaledCarryMass,
            scaledChunkMass, BinaryOperator::Add);

        auto momentType = cast<FragmentType>(carries[3].getType());
        Value carryMomentScale = nested.create<BroadcastOp>(
            nestedLocation, momentType, carryScale);
        Value chunkMomentScale = nested.create<BroadcastOp>(
            nestedLocation, momentType, chunkScale);
        Value scaledCarryMoment = nested.create<BinaryOp>(
            nestedLocation, momentType, carries[3], carryMomentScale,
            BinaryOperator::Multiply);
        Value scaledChunkMoment = nested.create<BinaryOp>(
            nestedLocation, momentType, chunkMoment, chunkMomentScale,
            BinaryOperator::Multiply);
        Value combinedMoment = nested.create<BinaryOp>(
            nestedLocation, momentType, scaledCarryMoment, scaledChunkMoment,
            BinaryOperator::Add);
        nested.create<scf::YieldOp>(
            nestedLocation,
            ValueRange{bothValid, combinedMaximum, combinedMass,
                       combinedMoment});
      });
  if (bodyFailed) {
    loop.erase();
    return pattern.moment.emitOpError(
               "online reduction could not form one physical traversal: ")
           << bodyFailure;
  }
  if (Attribute origin = pattern.moment->getAttr(originAttr))
    loop->setAttr(originAttr, origin);
  loop->setAttr(
      reductionSourcesAttr,
      builder.getArrayAttr({PhysicalSourceAttr::get(
          pattern.moment.getContext(), pattern.traversal.sourceId,
          pattern.traversal.sourceAxis, pattern.traversal.derived)}));
  Value finalMaximum = builder.create<SelectOp>(
      location, pattern.maximumOrEmpty.getResult().getType(),
      loop.getResult(0), loop.getResult(1), maximumIdentity);
  // Earlier users retain the original value graph. Only uses dominated by the
  // completed summary are redirected; the source graph then dies naturally
  // when its remaining uses are only its own reductions and contraction.
  DominanceInfo dominance(kernel);
  for (auto [original, replacement] :
       {std::pair<Value, Value>{pattern.validity.getResult(0), loop.getResult(0)},
        {pattern.maximumOrEmpty.getResult(), finalMaximum},
        {pattern.mass.getResult(0), loop.getResult(2)},
        {pattern.moment.getResult(), loop.getResult(3)}})
    original.replaceUsesWithIf(replacement, [&](OpOperand &use) {
      return dominance.dominates(replacement, use.getOwner());
    });
  eraseDeadPhysicalValues(kernel);
  return success();
}

} // namespace

static LogicalResult realizeOnlineReductionsImpl(ModuleOp module) {
  auto options = readCompileOptions(module);
  if (failed(options)) return failure();
  if (!options->getOnlineReduction()) {
    if (options->getOptimizationRemarks())
      emitRemark(module.getLoc(), "online-reduction: disabled by compile options");
    return success();
  }
  FailureOr<func::FuncOp> physicalKernel = getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  func::FuncOp kernel = *physicalKernel;
  SmallVector<ContractOp> moments;
  kernel.walk([&](ContractOp moment) { moments.push_back(moment); });
  // A rewrite can erase other producers. Restart discovery from current SSA
  // after each successful rewrite rather than keeping stale operation handles.
  for (size_t index = 0; index < moments.size(); ++index) {
    ContractOp moment = moments[index];
    StringRef reason;
    FailureOr<OnlineSummaryPattern> pattern =
        matchOnlineSummary(moment, kernel, reason);
    if (failed(pattern)) {
      if (options->getOptimizationRemarks() && !reason.empty())
        emitRemark(moment.getLoc()) << "online-reduction: not applied; " << reason;
      continue;
    }
    if (options->getNumerics() != NumericsMode::RelaxedNormalization) {
      if (options->getOptimizationRemarks())
        emitRemark(moment.getLoc(), "online-reduction: not applied; source contract does not permit changing the normalization reference before the weight cast");
      continue;
    }
    if (auto reason = unsupportedNumerics(*pattern)) {
      if (options->getOptimizationRemarks())
        emitRemark(moment.getLoc()) << "online-reduction: not applied; " << *reason;
      continue;
    }
    Location location = moment.getLoc();
    if (failed(realizeOnlineSummary(*pattern, kernel)))
      return failure();
    if (options->getOptimizationRemarks())
      emitRemark(location, "online-reduction: applied under relaxed_normalization; valid scores and every value operand participating in the moment contraction must be finite");
    moments.clear();
    kernel.walk([&](ContractOp current) { moments.push_back(current); });
    index = static_cast<size_t>(-1);
  }
  return success();
}

LogicalResult realizeOnlineReductions(ModuleOp module) {
  if (failed(realizeOnlineReductionsImpl(module))) return failure();
  auto kernel = getPhysicalKernel(module);
  return failed(kernel) ? failure() : closeValueRelations(*kernel);
}

} // namespace intent::gpu
