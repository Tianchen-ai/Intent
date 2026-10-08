#include "RegionCoRealization.h"
#include "RegionSources.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dominance.h"
#include <functional>

using namespace mlir;

namespace intent::gpu::region {

SmallVector<AdditiveRegionContract>
additiveRegionContracts(RegionFoldOp fold, ValueRange identities) {
  SmallVector<AdditiveRegionContract> result;
  if (identities.size() != 1)
    return result;
  auto identity = identities.front().getDefiningOp<MakeRecordOp>();
  auto summaryYield = cast<YieldOp>(fold.getSummarize().front().getTerminator());
  auto mergeYield = cast<YieldOp>(fold.getCombine().front().getTerminator());
  auto summary = summaryYield.getValues().front().getDefiningOp<MakeRecordOp>();
  auto merged = mergeYield.getValues().front().getDefiningOp<MakeRecordOp>();
  if (!identity || !summary || !merged)
    return result;
  Block &combine = fold.getCombine().front();
  for (auto [field, value] : llvm::enumerate(summary.getFields())) {
    Value contracted = stripAdditiveProjection(value, /*singleUse=*/true);
    Value mergedField = stripAdditiveProjection(merged.getFields()[field]);
    auto contract = contracted ? contracted.getDefiningOp<ContractOp>()
                               : ContractOp();
    auto add = mergedField ? mergedField.getDefiningOp<BinaryOp>() : BinaryOp();
    if (!contract || !contract->hasOneUse() || !add ||
        add.getOperatorKind() != BinaryOperator::Add ||
        add.getStrictRounding() ||
        !isRecordField(stripAdditiveProjection(add.getLhs()),
                       combine.getArgument(0), field) ||
        !isRecordField(stripAdditiveProjection(add.getRhs()),
                       combine.getArgument(1), field) ||
        cast<FragmentType>(value.getType()).getElementType() !=
            contract.getResult().getType().getElementType() ||
        !isLiteralZeroProjection(contract.getAccumulator()))
      continue;
    if (isLiteralZeroProjection(identity.getFields()[field]))
      result.push_back({contract, add});
  }
  return result;
}

void coRealizeAdditiveRegion(OpBuilder &builder, Location location,
                            ArrayRef<AdditiveRegionContract> contracts,
                            IRMapping &summaryMapping,
                            IRMapping &mergeMapping) {
  for (AdditiveRegionContract plan : contracts) {
    Value summary = summaryMapping.lookup(plan.summary.getResult());
    Value merged = mergeMapping.lookup(plan.merge.getResult());
    auto contract = summary.getDefiningOp<ContractOp>();
    auto add = merged.getDefiningOp<BinaryOp>();
    DominanceInfo dominance(contract->getParentOfType<func::FuncOp>());
    OpBuilder::InsertionGuard insertion(builder);
    builder.setInsertionPoint(contract);
    IRMapping carryMapping;
    std::function<FailureOr<Value>(Value)> projectCarry =
        [&](Value value) -> FailureOr<Value> {
      if (dominance.dominates(value, contract))
        return value;
      if (Value mapped = carryMapping.lookupOrNull(value))
        return mapped;
      Operation *definition = value.getDefiningOp();
      if (!definition ||
          !isa<ExtractOp, MakeRecordOp, BroadcastOp, ReshapeOp, TransposeOp,
               SplatOp, CastOp, arith::ConstantOp>(definition))
        return failure();
      for (Value operand : definition->getOperands()) {
        FailureOr<Value> mapped = projectCarry(operand);
        if (failed(mapped))
          return failure();
        carryMapping.map(operand, *mapped);
      }
      builder.clone(*definition, carryMapping);
      return carryMapping.lookup(value);
    };
    FailureOr<Value> leftCarry = projectCarry(add.getLhs());
    if (failed(leftCarry))
      continue;
    FailureOr<Value> carry = projectPhysicalValueToSchema(
        builder, location, *leftCarry, contract.getResult().getType());
    if (failed(carry))
      continue;
    // The fold's ordered homomorphism permits carrying the preceding sum
    // through the next slice's contraction. Ordinary adjacent float adds
    // outside this declared reduction boundary are not rewritten here.
    auto direct = builder.create<ContractOp>(
        location, contract.getResult().getType(), contract.getLhs(),
        contract.getRhs(), *carry, contract.getLhsReductionAxes(),
        contract.getRhsReductionAxes(), contract.getLhsBatchAxes(),
        contract.getRhsBatchAxes());
    if (Attribute origin = contract->getAttr(originAttr))
      direct->setAttr(originAttr, origin);
    FailureOr<Value> projected = projectPhysicalValueToSchema(
        builder, location, direct.getResult(), add.getResult().getType());
    if (succeeded(projected))
      add.getResult().replaceAllUsesWith(*projected);
  }
}

std::optional<OnlineRegionPlan>
onlineRegionPlan(RegionFoldOp fold,
                 const std::optional<SummaryEmptinessPlan> &emptiness) {
  if (!emptiness || fold.getSummarize().empty())
    return std::nullopt;
  auto yield = dyn_cast<YieldOp>(fold.getSummarize().front().getTerminator());
  auto record = yield && yield.getValues().size() == 1
                    ? yield.getValues().front().getDefiningOp<MakeRecordOp>()
                    : MakeRecordOp();
  FailureOr<OnlineSummaryStructure> summary =
      matchOnlineSummaryStructure(record);
  if (failed(summary))
    return std::nullopt;
  if (summary->validityField != emptiness->optionalField ||
      summary->record.getResult().getType() != emptiness->fullType)
    return std::nullopt;
  FailureOr<OnlineSummaryMerge> merge =
      matchOnlineSummaryMerge(fold.getCombine(), *summary);
  if (failed(merge))
    return std::nullopt;
  return OnlineRegionPlan{std::move(*summary), std::move(*merge)};
}

FailureOr<Value> coRealizeOnlineRegion(
    OpBuilder &builder, Location location, OnlineRegionPlan &plan,
    IRMapping &summaryMapping, IRMapping &mergeMapping,
    Value encodedCarry) {
  auto summaryValue = [&](Value value) {
    return summaryMapping.lookupOrNull(value);
  };
  auto mergeValue = [&](Value value) {
    return mergeMapping.lookupOrNull(value);
  };

  Value memberValidity = summaryValue(plan.summary.memberValidity);
  Value maskedScore = summaryValue(plan.summary.maskedScore.getResult());
  Value probability = summaryValue(plan.summary.probability.getResult());
  Value probabilityZero =
      summaryValue(plan.summary.probability.getFalseValue());
  Value probabilityCast =
      summaryValue(plan.summary.moment.getLhs());
  Value mass = summaryValue(plan.summary.mass.getResult(0));
  Value moment = summaryValue(plan.summary.moment.getResult());
  Value combinedMaximum = mergeValue(plan.merge.combinedMaximum);
  Value leftMassTerm = mergeValue(plan.merge.leftMassTerm);
  Value leftMomentTerm = mergeValue(plan.merge.leftMomentTerm);
  Value mergedRecord = mergeValue(plan.merge.record.getResult());
  if (!memberValidity || !maskedScore || !probability || !probabilityZero ||
      !probabilityCast || !mass || !moment || !combinedMaximum ||
      !leftMassTerm || !leftMomentTerm || !mergedRecord)
    return failure();

  auto mappedMass = mass.getDefiningOp<ReduceOp>();
  auto mappedMoment = moment.getDefiningOp<ContractOp>();
  if (!mappedMass || !mappedMoment)
    return failure();
  if (encodedCarry) {
    auto carry = encodedCarry.getDefiningOp<MakeRecordOp>();
    Value rawMaximum = summaryValue(plan.summary.maximum.getResult(0));
    if (!carry || !rawMaximum)
      return failure();
    Value oldMaximum = carry.getFields()[plan.summary.maximumField];
    Value oldMass = carry.getFields()[plan.summary.massField];
    Value oldMoment = carry.getFields()[plan.summary.momentField];
    FailureOr<Value> rightMaximum = projectPhysicalValueToSchema(
        builder, location, rawMaximum, oldMaximum.getType());
    if (failed(rightMaximum))
      return failure();
    // maxnum with the -inf reduction identity never returns NaN. Encoding an
    // absent maximum as -inf therefore lets both empty and nonempty summaries
    // use the same maximum operation. Keep the scale's validity select: it
    // protects the empty/empty case without changing Inf/NaN multiplication.
    combinedMaximum = builder.create<BinaryOp>(
        location, oldMaximum.getType(), oldMaximum, *rightMaximum,
        BinaryOperator::MaximumNum);
    Value delta = builder.create<BinaryOp>(
        location, oldMaximum.getType(), oldMaximum, combinedMaximum,
        BinaryOperator::Subtract);
    Value exponential = builder.create<UnaryOp>(
        location, oldMaximum.getType(), delta,
        plan.summary.exponential.getOperatorKind(),
        plan.summary.exponential.getApproximate(),
        plan.summary.exponential.getFlushToZero());
    Value scalarZero = builder.create<arith::ConstantOp>(
        location, builder.getF32FloatAttr(0));
    Value zero = builder.create<SplatOp>(location, oldMaximum.getType(), scalarZero);
    Value scale = builder.create<SelectOp>(
        location, oldMaximum.getType(),
        carry.getFields()[plan.summary.validityField], exponential, zero);
    FailureOr<Value> massScale = projectPhysicalValueToSchema(
        builder, location, scale, oldMass.getType());
    FailureOr<Value> momentScale = projectPhysicalValueToSchema(
        builder, location, scale, mappedMoment.getResult().getType());
    FailureOr<Value> nativeMoment = projectPhysicalValueToSchema(
        builder, location, oldMoment, mappedMoment.getResult().getType());
    if (failed(massScale) || failed(momentScale) || failed(nativeMoment))
      return failure();
    leftMassTerm = builder.create<BinaryOp>(
        location, oldMass.getType(), *massScale, oldMass, BinaryOperator::Multiply);
    leftMomentTerm = builder.create<BinaryOp>(
        location, mappedMoment.getResult().getType(), *momentScale, *nativeMoment,
        BinaryOperator::Multiply);
  }
  FailureOr<Value> projectedMaximum = projectPhysicalValueToSchema(
      builder, location, combinedMaximum, maskedScore.getType());
  FailureOr<Value> projectedZero = projectPhysicalValueToSchema(
      builder, location, probabilityZero, probability.getType());
  if (failed(projectedMaximum) || failed(projectedZero))
    return failure();

  // region_fold declares summarize/combine as an ordered homomorphism.  Keep
  // the same maximum and left carry scale, but associate the right scale with
  // each member before its mass/moment reductions.
  auto shifted = builder.create<BinaryOp>(
      location, maskedScore.getType(), maskedScore, *projectedMaximum,
      BinaryOperator::Subtract);
  auto directExponential = builder.create<UnaryOp>(
      location, probability.getType(), shifted,
      plan.summary.exponential.getOperatorKind(),
      plan.summary.exponential.getApproximate(), plan.summary.exponential.getFlushToZero());
  auto directProbabilityOp = builder.create<SelectOp>(
      location, probability.getType(), memberValidity, directExponential,
      *projectedZero);
  if (Attribute origin = plan.summary.exponential->getAttr(originAttr)) {
    shifted->setAttr(originAttr, origin);
    directExponential->setAttr(originAttr, origin);
    directProbabilityOp->setAttr(originAttr, origin);
  }
  Value directProbability = directProbabilityOp.getResult();

  ReduceOp directMass = cloneReductionWithSource(
      builder, location, mappedMass, directProbability);
  Value directProbabilityCast = directProbability;
  if (directProbability.getType() != probabilityCast.getType()) {
    auto converted = builder.create<CastOp>(
        location, probabilityCast.getType(), directProbability);
    if (plan.summary.probabilityCast)
      if (Attribute origin = plan.summary.probabilityCast->getAttr(originAttr))
        converted->setAttr(originAttr, origin);
    directProbabilityCast = converted.getResult();
  }
  auto mappedRecord = mergedRecord.getDefiningOp<MakeRecordOp>();
  if (!mappedRecord)
    return failure();
  Type massType = mappedRecord.getFields()[plan.summary.massField].getType();
  Type momentType =
      mappedRecord.getFields()[plan.summary.momentField].getType();
  FailureOr<Value> projectedLeftMass = projectPhysicalValueToSchema(
      builder, location, leftMassTerm, massType);
  FailureOr<Value> projectedMass = projectPhysicalValueToSchema(
      builder, location, directMass.getResult(0), massType);
  FailureOr<Value> projectedLeftMoment = projectPhysicalValueToSchema(
      builder, location, leftMomentTerm, mappedMoment.getResult().getType());
  if (failed(projectedLeftMass) || failed(projectedMass) ||
      failed(projectedLeftMoment))
    return failure();
  // The matched moment has a zero accumulator. Keep the ordered left carry
  // inside the contraction instead of materializing a second matrix and add.
  auto directMoment = builder.create<ContractOp>(
      location, mappedMoment.getResult().getType(), directProbabilityCast,
      mappedMoment.getRhs(), *projectedLeftMoment,
      mappedMoment.getLhsReductionAxes(), mappedMoment.getRhsReductionAxes(),
      mappedMoment.getLhsBatchAxes(), mappedMoment.getRhsBatchAxes());
  if (Attribute origin = mappedMoment->getAttr(originAttr))
    directMoment->setAttr(originAttr, origin);
  FailureOr<Value> projectedMoment = projectPhysicalValueToSchema(
      builder, location, directMoment.getResult(), momentType);
  if (failed(projectedMoment))
    return failure();
  Value combinedMoment = *projectedMoment;
  if (plan.summary.momentOrEmpty) {
    Value rowValidity = summaryValue(plan.summary.validity.getResult(0));
    if (!rowValidity)
      return failure();
    UniformValueAnalysis facts(describeUniformValue);
    if (uniformBoolean(facts.evaluate(rowValidity)) != true) {
      FailureOr<Value> condition = projectPhysicalValueToSchema(
          builder, location, rowValidity, predicateType(cast<FragmentType>(momentType)));
      FailureOr<Value> unchangedCarry = projectPhysicalValueToSchema(
          builder, location, leftMomentTerm, momentType);
      if (failed(condition) || failed(unchangedCarry))
        return failure();
      // Preserve the author's zero moment for fully masked rows, including
      // when an inactive value operand contains nonfinite elements.
      combinedMoment = builder.create<SelectOp>(
          location, momentType, *condition, combinedMoment, *unchangedCarry);
    }
  }
  Value combinedMass = builder.create<BinaryOp>(
      location, massType, *projectedLeftMass, *projectedMass,
      BinaryOperator::Add);

  SmallVector<Value> fields;
  fields.reserve(plan.merge.record.getFields().size());
  for (auto [field, original] :
       llvm::enumerate(plan.merge.record.getFields())) {
    if (field == plan.summary.massField) {
      fields.push_back(combinedMass);
      continue;
    }
    if (field == plan.summary.momentField) {
      fields.push_back(combinedMoment);
      continue;
    }
    if (encodedCarry && field == plan.summary.maximumField) {
      fields.push_back(combinedMaximum);
      continue;
    }
    Value mapped = mergeValue(original);
    if (!mapped)
      return failure();
    fields.push_back(mapped);
  }
  auto replacement = builder.create<MakeRecordOp>(
      location, mappedRecord.getResult().getType(), fields);
  if (Attribute origin = plan.merge.record->getAttr(originAttr))
    replacement->setAttr(originAttr, origin);
  return replacement.getResult();
}

} // namespace intent::gpu::region
