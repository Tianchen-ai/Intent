#include "Intent/Analysis/OnlineSummary.h"
#include "Intent/Interfaces/StructuredOpInterface.h"
#include "Intent/Analysis/OnlineSummaryCombine.h"
#include "Intent/Analysis/UniformValues.h"
#include <algorithm>

using namespace mlir;
namespace intent {
namespace {
Attribute constant(Value value) {
  return UniformValueAnalysis(describeCanonicalUniformValue).evaluate(value);
}
bool zero(Value value) {
  auto attr = constant(value);
  if (auto number = dyn_cast_or_null<FloatAttr>(attr)) return number.getValue().isZero();
  if (auto number = dyn_cast_or_null<IntegerAttr>(attr)) return number.getValue().isZero();
  return false;
}
bool truth(Value value, bool expected) {
  auto attr = dyn_cast_or_null<IntegerAttr>(constant(value));
  return attr && attr.getType().isInteger(1) && attr.getValue().getBoolValue() == expected;
}
bool negativeInfinity(Value value) {
  auto attr = dyn_cast_or_null<FloatAttr>(constant(value));
  return attr && attr.getValue().isInfinity() && attr.getValue().isNegative();
}
Value projected(Value value) {
  while (Operation *op = value.getDefiningOp()) {
    if (isa<BroadcastOp>(op)) { value = op->getOperand(0); continue; }
    if (auto gather = dyn_cast<GatherOp>(op)) {
      bool insertion = gather.getValid() && truth(gather.getValid(), true);
      unsigned retained = 0;
      for (Attribute attr : gather.getIndex().getTerms()) {
        auto term = cast<IndexTermAttr>(attr);
        insertion &= term.getKind() == 0 || term.getKind() == 1;
        retained += term.getKind() == 0;
      }
      insertion &= retained == gather.getIndex().getSourceRank();
      if (insertion) { value = gather.getSource(); continue; }
    }
    break;
  }
  return value;
}
std::optional<BinaryOperator> reductionKind(ReduceOp reduce) {
  if (!reduce || reduce.getNumResults() != 1 || reduce.getAxes().size() != 1 ||
      reduce.getSources().size() != 1 || reduce.getIdentities().size() != 1 || reduce.getCaptures().size())
    return std::nullopt;
  auto schema = cast<StructuredOpInterface>(reduce.getOperation());
  auto combine = schema.getCombineYields().front().getDefiningOp<BinaryOp>();
  Value lhs = schema.getCombineLhs().front(), rhs = schema.getCombineRhs().front();
  if (!combine || !((combine.getLhs() == lhs && combine.getRhs() == rhs) ||
                    (combine.getLhs() == rhs && combine.getRhs() == lhs))) return std::nullopt;
  return combine.getOperatorKind();
}
bool field(Value value, Value record, unsigned index) {
  auto extract = projected(value).getDefiningOp<ExtractOp>();
  return extract && extract.getProduct() == record && extract.getField() == index;
}
}

std::optional<OnlineSummary> matchOnlineSummary(RegionFoldOp fold) {
  if (fold.getIdentities().size() != 1 || fold.getNumResults() != 1) return std::nullopt;
  auto schema = cast<StructuredOpInterface>(fold.getOperation());
  auto &merge = fold.getCombine().front();
  OnlineSummary plan{};
  plan.summary = schema.getSummarizeYields().front().getDefiningOp<MakeRecordOp>();
  plan.merged = schema.getCombineYields().front().getDefiningOp<MakeRecordOp>();
  if (!plan.summary || !plan.merged || plan.summary.getFields().size() != 4 ||
      plan.merged.getFields().size() != 4 || merge.getNumArguments() != 2) return std::nullopt;
  unsigned validCount = 0, maxCount = 0, massCount = 0, momentCount = 0;
  for (auto [i, value] : llvm::enumerate(plan.summary.getFields())) {
    value = projected(value);
    if (auto reduce = value.getDefiningOp<ReduceOp>()) {
      if (reductionKind(reduce) == BinaryOperator::LogicalOr) { plan.validity = reduce; plan.validField = i; ++validCount; }
      if (reductionKind(reduce) == BinaryOperator::Add) { plan.mass = reduce; plan.massField = i; ++massCount; }
    }
    if (auto select = value.getDefiningOp<SelectOp>()) {
      auto reduce = projected(select.getTrueValue()).getDefiningOp<ReduceOp>();
      auto kind = reductionKind(reduce);
      if ((kind == BinaryOperator::Maximum || kind == BinaryOperator::MaximumNum) && zero(select.getFalseValue())) {
        plan.maximum = reduce; plan.maximumOrEmpty = select; plan.maxField = i; ++maxCount;
      }
      if (auto moment = projected(select.getTrueValue()).getDefiningOp<ContractOp>(); moment && zero(select.getFalseValue())) {
        plan.moment = moment; plan.momentOrEmpty = select; plan.momentField = i; ++momentCount;
      }
    } else if (auto moment = value.getDefiningOp<ContractOp>()) {
      plan.moment = moment; plan.momentField = i; ++momentCount;
    }
  }
  if (validCount != 1 || maxCount != 1 || massCount != 1 || momentCount != 1 ||
      !truth(plan.validity.getIdentities().front(), false) || !negativeInfinity(plan.maximum.getIdentities().front()) ||
      !zero(plan.mass.getIdentities().front()) || plan.maximumOrEmpty.getCondition() != plan.validity.getResult(0) ||
      (plan.momentOrEmpty && projected(plan.momentOrEmpty.getCondition()) != plan.validity.getResult(0))) return std::nullopt;
  auto masked = plan.maximum.getSources().front().getDefiningOp<SelectOp>();
  plan.probability = plan.mass.getSources().front().getDefiningOp<SelectOp>();
  if (!masked || !plan.probability || masked.getCondition() != plan.validity.getSources().front() ||
      !negativeInfinity(masked.getFalseValue()) || plan.probability.getCondition() != masked.getCondition() ||
      !zero(plan.probability.getFalseValue())) return std::nullopt;
  plan.exponential = plan.probability.getTrueValue().getDefiningOp<UnaryOp>();
  if (!plan.exponential || (plan.exponential.getOperatorKind() != UnaryOperator::Exp &&
      plan.exponential.getOperatorKind() != UnaryOperator::Exp2)) return std::nullopt;
  auto subtract = plan.exponential.getInput().getDefiningOp<BinaryOp>();
  plan.probabilityCast = plan.moment.getLhs().getDefiningOp<CastOp>();
  if (!subtract || subtract.getOperatorKind() != BinaryOperator::Subtract || subtract.getLhs() != masked.getResult() ||
      projected(subtract.getRhs()) != plan.maximumOrEmpty.getResult() || !plan.probabilityCast ||
      plan.probabilityCast.getInput() != plan.probability.getResult() || plan.probabilityCast.getRounding() ||
      plan.validity.getAxes() != plan.maximum.getAxes() || plan.mass.getAxes() != plan.maximum.getAxes()) return std::nullopt;
  auto pair = plan.moment.getReduce().size() == 1 ? dyn_cast<ArrayAttr>(plan.moment.getReduce()[0]) : ArrayAttr();
  if (!pair || pair.size() != 2 || !plan.moment.getBatch().empty() ||
      cast<IntegerAttr>(pair[0]).getInt() != cast<IntegerAttr>(plan.maximum.getAxes()[0]).getInt()) return std::nullopt;
  auto scoreType = cast<RankedTensorType>(masked.getType());
  auto valuesType = cast<RankedTensorType>(plan.moment.getRhs().getType());
  auto scoreAxes = dyn_cast<TensorShapeAttr>(scoreType.getEncoding());
  auto valueAxes = dyn_cast<TensorShapeAttr>(valuesType.getEncoding());
  if (!scoreAxes || !valueAxes || scoreAxes.getDimensions()[cast<IntegerAttr>(pair[0]).getInt()] !=
      valueAxes.getDimensions()[cast<IntegerAttr>(pair[1]).getInt()]) return std::nullopt;

  SmallVector<Value> candidates;
  merge.walk([&](SelectOp select) { candidates.push_back(select.getResult()); });
  // Canonical matching historically selects the last matching scale in walk
  // order. The shared matcher selects the first candidate, so reverse here.
  std::reverse(candidates.begin(), candidates.end());
  auto relations = matchOnlineSummaryCombine<BinaryOp, UnaryOp, SelectOp>(
      plan.merged.getFields(), schema.getCombineLhs().front(), schema.getCombineRhs().front(),
      {plan.validField, plan.maxField, plan.massField, plan.momentField},
      reductionKind(plan.maximum), plan.exponential, candidates, projected,
      [](Value value, Value source) { return projected(value) == projected(source); },
      field, zero);
  if (!relations) return std::nullopt;
  plan.combinedMaximum = relations->combinedMaximum;
  plan.leftScale = relations->leftScale;
  plan.rightScale = relations->rightScale;
  plan.leftMassTerm = relations->leftMassTerm;
  plan.leftMomentTerm = relations->leftMomentTerm;
  return plan;
}
}
