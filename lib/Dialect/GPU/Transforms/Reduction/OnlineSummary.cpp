#include "OnlineSummary.h"
#include "Intent/Dialect/GPU/Analysis/Helpers.h"

#include "Intent/Analysis/OnlineSummaryCombine.h"
#include "Intent/Dialect/GPU/IR/FragmentOpInterface.h"
#include "Intent/Dialect/GPU/IR/Program.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace intent::gpu {
namespace {

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
    if (auto reshape = value.getDefiningOp<ReshapeOp>()) {
      value = reshape.getValue();
      continue;
    }
    if (auto transpose = value.getDefiningOp<TransposeOp>()) {
      value = transpose.getValue();
      continue;
    }
    return value;
  }
  return {};
}

bool isBooleanConstant(Value value, bool expected) {
  auto constant = scalarValue(value).getDefiningOp<arith::ConstantOp>();
  auto attribute = constant ? dyn_cast<IntegerAttr>(constant.getValue())
                            : IntegerAttr();
  return attribute && attribute.getType().isInteger(1) &&
         attribute.getValue().getBoolValue() == expected;
}

bool isZero(Value value) {
  auto constant = scalarValue(value).getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return false;
  if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
    return integer.getValue().isZero();
  if (auto floating = dyn_cast<FloatAttr>(constant.getValue()))
    return floating.getValue().isZero();
  return false;
}

bool isNegativeInfinity(Value value) {
  auto constant = scalarValue(value).getDefiningOp<arith::ConstantOp>();
  auto floating = constant ? dyn_cast<FloatAttr>(constant.getValue())
                           : FloatAttr();
  return floating && floating.getValue().isInfinity() &&
         floating.getValue().isNegative();
}

bool sameExecutionSchema(Type lhs, Type rhs) {
  auto left = dyn_cast<FragmentType>(lhs);
  auto right = dyn_cast<FragmentType>(rhs);
  return left && right && left.getShape() == right.getShape() &&
         left.getAxisMaps() == right.getAxisMaps() &&
         left.getValidity() == right.getValidity() &&
         left.getOwner() == right.getOwner();
}

bool isSingleBinaryReduction(ReduceOp reduce, BinaryOperator kind) {
  if (!reduce) return false;
  auto combine = queryBinaryCombine(reduce.getCombine());
  return reduce.getNumResults() == 1 &&
         reduce.getAxes().size() == 1 && reduce.getSources().size() == 1 &&
         reduce.getIdentities().size() == 1 && reduce.getCaptures().size() == 0 &&
         combine && combine->kind() == kind &&
         !combine->operation.getStrictRounding();
}

bool isProjectedFrom(Value value, Value source,
                     SmallPtrSetImpl<Operation *> &visited) {
  if (value == source)
    return true;
  Operation *producer = value.getDefiningOp();
  if (!producer || !visited.insert(producer).second)
    return false;
  if (auto broadcast = dyn_cast<BroadcastOp>(producer))
    return isProjectedFrom(broadcast.getValue(), source, visited);
  if (auto reshape = dyn_cast<ReshapeOp>(producer))
    return isProjectedFrom(reshape.getValue(), source, visited);
  if (auto select = dyn_cast<SelectOp>(producer))
    return isBooleanConstant(select.getCondition(), true) &&
           isZero(select.getFalseValue()) &&
           isProjectedFrom(select.getTrueValue(), source, visited);
  return false;
}

bool isProjectedFrom(Value value, Value source) {
  SmallPtrSet<Operation *, 8> visited;
  return isProjectedFrom(value, source, visited);
}

Value stripProjection(Value value) {
  while (Operation *producer = value.getDefiningOp()) {
    if (auto broadcast = dyn_cast<BroadcastOp>(producer)) {
      value = broadcast.getValue();
      continue;
    }
    if (auto reshape = dyn_cast<ReshapeOp>(producer)) {
      value = reshape.getValue();
      continue;
    }
    break;
  }
  return value;
}

bool isRecordField(Value value, BlockArgument record, unsigned field) {
  value = stripProjection(value);
  auto extract = value.getDefiningOp<ExtractOp>();
  return extract && extract.getRecord() == record && extract.getField() == field;
}

bool sameProjectedValue(Value lhs, Value rhs) {
  if (lhs == rhs)
    return true;
  if (lhs.getType() != rhs.getType())
    return false;
  auto left = lhs.getDefiningOp<BroadcastOp>();
  auto right = rhs.getDefiningOp<BroadcastOp>();
  return left && right && sameProjectedValue(left.getValue(), right.getValue());
}

using SummaryAxes = SmallVector<std::optional<unsigned>>;

FailureOr<SummaryAxes> summaryProjection(Value value, Value summary) {
  auto output = dyn_cast<FragmentType>(value.getType());
  if (!output)
    return failure();
  if (value == summary) {
    SummaryAxes axes;
    for (unsigned axis = 0; axis < output.getShape().size(); ++axis)
      axes.push_back(axis);
    return axes;
  }
  if (auto select = value.getDefiningOp<SelectOp>();
      select && isBooleanConstant(select.getCondition(), true) &&
      isZero(select.getFalseValue()))
    return summaryProjection(select.getTrueValue(), summary);
  Operation *projection = value.getDefiningOp();
  if (!isa_and_nonnull<BroadcastOp, ReshapeOp>(projection))
    return failure();
  auto relations = queryFragmentOperandRelations(projection);
  auto source = summaryProjection(projection->getOperand(0), summary);
  if (failed(relations) || failed(source))
    return failure();
  SummaryAxes axes(output.getShape().size());
  for (const FragmentAxisGroup &group : relations->front().groups) {
    if (group.sourceAxes.size() == 1 && group.resultAxes.size() == 1) {
      axes[group.resultAxes.front()] = (*source)[group.sourceAxes.front()];
      continue;
    }
    // Inserting, dropping or regrouping axes that do not index the summary
    // does not change its reference. A flattened summary axis needs a more
    // general coordinate proof and is deliberately left unchanged here.
    for (unsigned axis : group.sourceAxes)
      if ((*source)[axis])
        return failure();
  }
  return axes;
}

bool hasReductionReference(Value reference, Value summary, unsigned axis) {
  auto projection = summaryProjection(reference, summary);
  if (failed(projection) || axis >= projection->size())
    return false;
  unsigned kept = 0;
  for (auto [position, source] : llvm::enumerate(*projection)) {
    if (position == axis) {
      if (source)
        return false;
    } else if (!source || *source != kept++) {
      return false;
    }
  }
  return kept == cast<FragmentType>(summary.getType()).getShape().size();
}

FailureOr<NormalizedSummaryStructure> matchSummaryComponents(
    ReduceOp validity, SelectOp maximumOrEmpty, ReduceOp mass,
    ContractOp moment, SelectOp momentOrEmpty = {}) {
  if (!isSingleBinaryReduction(validity, BinaryOperator::LogicalOr) ||
      !isSingleBinaryReduction(mass, BinaryOperator::Add) || !maximumOrEmpty)
    return failure();
  auto validityType = dyn_cast<FragmentType>(validity.getResult(0).getType());
  if (!validityType || !validityType.getElementType().isInteger(1))
    return failure();
  auto maximum = maximumOrEmpty.getTrueValue().getDefiningOp<ReduceOp>();
  if (!isSingleBinaryReduction(maximum, BinaryOperator::Maximum) &&
      !isSingleBinaryReduction(maximum, BinaryOperator::MaximumNum))
    return failure();
  if (momentOrEmpty &&
      !isProjectedFrom(momentOrEmpty.getCondition(), validity.getResult(0)))
    return failure();
  Value memberValidity = validity.getSources().front();
  if (maximumOrEmpty.getCondition() != validity.getResult(0) ||
      !isZero(maximumOrEmpty.getFalseValue()) ||
      !isBooleanConstant(validity.getIdentities().front(), false) ||
      !isNegativeInfinity(maximum.getIdentities().front()))
    return failure();

  auto maskedScore = maximum.getSources().front().getDefiningOp<SelectOp>();
  if (!maskedScore ||
      !sameProjectedValue(maskedScore.getCondition(), memberValidity) ||
      !isNegativeInfinity(maskedScore.getFalseValue()))
    return failure();
  Value score = maskedScore.getTrueValue();

  Value probabilityValue = mass.getSources().front();
  auto probability = probabilityValue.getDefiningOp<SelectOp>();
  if (!probability ||
      !sameProjectedValue(probability.getCondition(), memberValidity) ||
      !isZero(probability.getFalseValue()))
    return failure();
  auto exponential = probability.getTrueValue().getDefiningOp<UnaryOp>();
  if (!exponential ||
      (exponential.getOperatorKind() != UnaryOperator::Exp &&
       exponential.getOperatorKind() != UnaryOperator::Exp2))
    return failure();
  auto shift = exponential.getInput().getDefiningOp<BinaryOp>();
  if (!shift || shift.getOperatorKind() != BinaryOperator::Subtract ||
      shift.getStrictRounding() ||
      shift.getLhs() != maskedScore.getResult() ||
      !isProjectedFrom(shift.getRhs(), maximumOrEmpty.getResult()))
    return failure();

  auto probabilityCast = moment.getLhs().getDefiningOp<CastOp>();
  Value weightSource = probabilityCast ? probabilityCast.getValue() : moment.getLhs();
  if (weightSource != probabilityValue ||
      !isZero(moment.getAccumulator()) ||
      moment.getLhsReductionAxes().size() != 1 ||
      moment.getRhsReductionAxes().size() != 1)
    return failure();
  Value values = moment.getRhs();

  int64_t rawReductionAxis = validity.getAxes().front();
  int64_t rawValueReductionAxis = moment.getRhsReductionAxes().front();
  if (rawReductionAxis < 0 || rawValueReductionAxis < 0)
    return failure();
  unsigned reductionAxis = static_cast<unsigned>(rawReductionAxis);
  if (!hasReductionReference(shift.getRhs(), maximumOrEmpty.getResult(),
                             reductionAxis) ||
      validity.getAxes() != maximum.getAxes() ||
      validity.getAxes() != mass.getAxes() ||
      moment.getLhsReductionAxes().front() !=
          static_cast<int64_t>(reductionAxis) ||
      !sameExecutionSchema(validity.getResult(0).getType(),
                           maximum.getResult(0).getType()) ||
      !sameExecutionSchema(maximum.getResult(0).getType(),
                           mass.getResult(0).getType()) ||
      !sameExecutionSchema(memberValidity.getType(), score.getType()) ||
      !sameExecutionSchema(score.getType(), probabilityValue.getType()))
    return failure();
  unsigned valueReductionAxis = static_cast<unsigned>(rawValueReductionAxis);

  auto scoreType = dyn_cast<FragmentType>(score.getType());
  auto memberType = dyn_cast<FragmentType>(memberValidity.getType());
  auto valueType = dyn_cast<FragmentType>(values.getType());
  if (!scoreType || !memberType || !valueType ||
      reductionAxis >= scoreType.getShape().size() ||
      reductionAxis >= memberType.getShape().size() ||
      valueReductionAxis >= valueType.getShape().size())
    return failure();
  FailureOr<AxisMapAttr> scoreMap = queryAxisMap(scoreType, reductionAxis);
  FailureOr<AxisMapAttr> validityMap =
      queryAxisMap(memberType, reductionAxis);
  FailureOr<AxisMapAttr> valueMap =
      queryAxisMap(valueType, valueReductionAxis);
  if (failed(scoreMap) || failed(validityMap) || failed(valueMap))
    return failure();
  PhysicalSourceAxis traversal = sourceAxisIdentity(*scoreMap);
  if (!(sourceAxisIdentity(*validityMap) == traversal) ||
      !(sourceAxisIdentity(*valueMap) == traversal))
    return failure();

  return NormalizedSummaryStructure{
      validity, maximum, mass, moment, maximumOrEmpty, maskedScore,
      probability, probabilityCast, exponential, memberValidity, score, values,
      reductionAxis, valueReductionAxis, traversal, momentOrEmpty};
}

} // namespace

FailureOr<OnlineSummaryStructure>
matchOnlineSummaryStructure(MakeRecordOp record) {
  if (!record || record.getFields().size() != 4)
    return failure();
  SmallVector<std::pair<unsigned, ReduceOp>> validityCandidates;
  SmallVector<std::pair<unsigned, ReduceOp>> massCandidates;
  SmallVector<std::pair<unsigned, SelectOp>> maximumCandidates;
  SmallVector<std::pair<unsigned, ContractOp>> momentCandidates;
  SmallVector<SelectOp> momentGuards;
  for (auto [index, field] : llvm::enumerate(record.getFields())) {
    field = stripProjection(field);
    if (auto reduce = field.getDefiningOp<ReduceOp>()) {
      if (isSingleBinaryReduction(reduce, BinaryOperator::LogicalOr))
        validityCandidates.emplace_back(index, reduce);
      if (isSingleBinaryReduction(reduce, BinaryOperator::Add))
        massCandidates.emplace_back(index, reduce);
    } else if (auto select = field.getDefiningOp<SelectOp>()) {
      auto reduce = select.getTrueValue().getDefiningOp<ReduceOp>();
      if (isSingleBinaryReduction(reduce, BinaryOperator::Maximum) ||
          isSingleBinaryReduction(reduce, BinaryOperator::MaximumNum))
        maximumCandidates.emplace_back(index, select);
      auto contract = stripProjection(select.getTrueValue()).getDefiningOp<ContractOp>();
      if (contract && isZero(select.getFalseValue())) {
        momentCandidates.emplace_back(index, contract);
        momentGuards.push_back(select);
      }
    } else if (auto contract = field.getDefiningOp<ContractOp>()) {
      momentCandidates.emplace_back(index, contract);
      momentGuards.push_back({});
    }
  }
  if (validityCandidates.size() != 1 || massCandidates.size() != 1 ||
      maximumCandidates.size() != 1 || momentCandidates.size() != 1)
    return failure();
  auto [validityField, validity] = validityCandidates.front();
  auto [maximumField, maximum] = maximumCandidates.front();
  auto [massField, mass] = massCandidates.front();
  auto [momentField, moment] = momentCandidates.front();
  auto summary = matchSummaryComponents(validity, maximum, mass, moment,
                                        momentGuards.front());
  if (failed(summary))
    return failure();
  return OnlineSummaryStructure{std::move(*summary), record, validityField,
                                maximumField, massField, momentField};
}

FailureOr<NormalizedSummaryStructure>
matchNormalizedSummaryStructure(ContractOp moment) {
  auto probabilityCast = moment.getLhs().getDefiningOp<CastOp>();
  Value weights = probabilityCast ? probabilityCast.getValue() : moment.getLhs();
  auto probability = weights.getDefiningOp<SelectOp>();
  auto exponential = probability
      ? probability.getTrueValue().getDefiningOp<UnaryOp>() : UnaryOp{};
  auto shift = exponential
      ? exponential.getInput().getDefiningOp<BinaryOp>() : BinaryOp{};
  if (!shift || shift.getOperatorKind() != BinaryOperator::Subtract ||
      shift.getStrictRounding())
    return failure();
  Value reference = shift.getRhs();
  while (Operation *producer = reference.getDefiningOp()) {
    if (auto select = dyn_cast<SelectOp>(producer);
        select && isBooleanConstant(select.getCondition(), true) &&
        isZero(select.getFalseValue())) {
      reference = select.getTrueValue();
      continue;
    }
    Value projected = stripProjection(reference);
    if (projected == reference)
      break;
    reference = projected;
  }
  auto maximum = reference.getDefiningOp<SelectOp>();
  auto validity = maximum
      ? maximum.getCondition().getDefiningOp<ReduceOp>() : ReduceOp{};
  SmallVector<ReduceOp> masses;
  for (Operation *user : probability.getResult().getUsers())
    if (auto reduce = dyn_cast<ReduceOp>(user);
        isSingleBinaryReduction(reduce, BinaryOperator::Add) &&
        reduce.getSources().front() == probability.getResult())
      masses.push_back(reduce);
  if (masses.size() != 1)
    return failure();
  return matchSummaryComponents(validity, maximum, masses.front(), moment);
}

FailureOr<OnlineSummaryMerge>
matchOnlineSummaryMerge(Region &region,
                        const OnlineSummaryStructure &summary) {
  if (region.empty() || !llvm::hasSingleElement(region) ||
      region.front().getNumArguments() != 2)
    return failure();
  bool strict = false;
  region.walk([&](BinaryOp binary) { strict |= binary.getStrictRounding(); });
  if (strict) return failure();
  auto yield = dyn_cast<YieldOp>(region.front().getTerminator());
  auto record = yield && yield.getValues().size() == 1
                    ? yield.getValues().front().getDefiningOp<MakeRecordOp>()
                    : MakeRecordOp();
  if (!record || record.getFields().size() != 4)
    return failure();
  BlockArgument left = region.front().getArgument(0);
  BlockArgument right = region.front().getArgument(1);
  auto massAdd = stripProjection(record.getFields()[summary.massField])
                     .getDefiningOp<BinaryOp>();
  if (!massAdd || massAdd.getOperatorKind() != BinaryOperator::Add)
    return failure();
  SmallVector<Value> candidates;
  for (Value term : {massAdd.getLhs(), massAdd.getRhs()}) {
    auto multiply = stripProjection(term).getDefiningOp<BinaryOp>();
    if (!multiply || multiply.getOperatorKind() != BinaryOperator::Multiply)
      return failure();
    for (Value operand : {multiply.getLhs(), multiply.getRhs()})
      if (stripProjection(operand).getDefiningOp<SelectOp>())
        candidates.push_back(stripProjection(operand));
  }
  // Keep the original first-match order among the mass expression's factors.
  ReduceOp summaryMaximum = summary.maximum;
  auto maximumCombine = queryBinaryCombine(summaryMaximum.getCombine());
  if (!maximumCombine) return failure();
  auto relations = matchOnlineSummaryCombine<BinaryOp, UnaryOp, SelectOp>(
      record.getFields(), left, right,
      {summary.validityField, summary.maximumField, summary.massField, summary.momentField},
      std::optional<BinaryOperator>(maximumCombine->kind()), summary.exponential,
      candidates, stripProjection,
      [](Value value, Value source) { return isProjectedFrom(value, source); },
      isRecordField, isZero);
  if (!relations) return failure();
  return OnlineSummaryMerge{record, relations->combinedMaximum,
                            relations->leftScale, relations->rightScale,
                            relations->leftMassTerm, relations->leftMomentTerm};
}

ReduceOp cloneReductionWithSource(OpBuilder &builder, Location location,
                                  ReduceOp source, Value value) {
  auto result = builder.create<ReduceOp>(location, ValueRange{value},
      source.getIdentities(), source.getCaptures(), source.getAxes());
  IRMapping mapping;
  source.getCombine().cloneInto(&result.getCombine(), mapping);
  if (Attribute origin = source->getAttr(originAttr))
    result->setAttr(originAttr, origin);
  return result;
}

} // namespace intent::gpu
