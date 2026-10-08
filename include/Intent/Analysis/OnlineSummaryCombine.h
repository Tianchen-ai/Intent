#ifndef INTENT_ANALYSIS_ONLINESUMMARYCOMBINE_H
#define INTENT_ANALYSIS_ONLINESUMMARYCOMBINE_H

#include "Intent/Dialect/Intent/IR/IntentAttrs.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/ArrayRef.h"
#include <optional>

namespace intent {

struct OnlineSummaryFields {
  unsigned validity, maximum, mass, moment;
};

struct OnlineSummaryCombineRelations {
  mlir::Value combinedMaximum, leftScale, rightScale;
  mlir::Value leftMassTerm, leftMomentTerm;
};

// Read only the current typed combine graph. Callers establish the summary
// schema and supply their own projection/constant rules and candidate order;
// this matcher neither chooses physical structure nor rewrites either IR.
template <typename BinaryOp, typename UnaryOp, typename SelectOp,
          typename Project, typename ProjectedFrom, typename Field, typename Zero>
std::optional<OnlineSummaryCombineRelations> matchOnlineSummaryCombine(
    mlir::ValueRange fields, mlir::BlockArgument left, mlir::BlockArgument right,
    OnlineSummaryFields schema, std::optional<BinaryOperator> maximumKind,
    UnaryOp summaryExponential, llvm::ArrayRef<mlir::Value> scaleCandidates,
    Project project, ProjectedFrom projectedFrom, Field field, Zero zero) {
  using mlir::Value;
  auto fieldPair = [&](Value value, BinaryOperator kind, unsigned index) {
    auto binary = project(value).template getDefiningOp<BinaryOp>();
    return binary && binary.getOperatorKind() == kind &&
        ((field(binary.getLhs(), left, index) && field(binary.getRhs(), right, index)) ||
         (field(binary.getRhs(), left, index) && field(binary.getLhs(), right, index)));
  };
  if (!fieldPair(fields[schema.validity], BinaryOperator::LogicalOr, schema.validity) &&
      !fieldPair(fields[schema.validity], BinaryOperator::BitwiseOr, schema.validity))
    return std::nullopt;

  Value combinedMaximum = fields[schema.maximum];
  auto finalMaximum = project(combinedMaximum).template getDefiningOp<SelectOp>();
  if (!finalMaximum || !field(finalMaximum.getCondition(), right, schema.validity))
    return std::nullopt;
  auto initialMaximum = project(finalMaximum.getFalseValue()).template getDefiningOp<SelectOp>();
  if (!initialMaximum || !field(initialMaximum.getCondition(), left, schema.validity) ||
      !field(initialMaximum.getTrueValue(), left, schema.maximum) ||
      !field(initialMaximum.getFalseValue(), right, schema.maximum))
    return std::nullopt;
  auto maximumOfBoth = project(finalMaximum.getTrueValue()).template getDefiningOp<BinaryOp>();
  if (!maximumOfBoth || maximumOfBoth.getOperatorKind() != maximumKind ||
      !((projectedFrom(maximumOfBoth.getLhs(), initialMaximum.getResult()) &&
         field(maximumOfBoth.getRhs(), right, schema.maximum)) ||
        (projectedFrom(maximumOfBoth.getRhs(), initialMaximum.getResult()) &&
         field(maximumOfBoth.getLhs(), right, schema.maximum))))
    return std::nullopt;

  auto matchScale = [&](Value value, mlir::BlockArgument record) -> Value {
    auto scale = project(value).template getDefiningOp<SelectOp>();
    if (!scale || !field(scale.getCondition(), record, schema.validity) ||
        !zero(scale.getFalseValue()))
      return {};
    auto exponential = scale.getTrueValue().template getDefiningOp<UnaryOp>();
    if (!exponential ||
        exponential.getOperatorKind() != summaryExponential.getOperatorKind() ||
        exponential.getApproximate() != summaryExponential.getApproximate() ||
        exponential.getFlushToZero() != summaryExponential.getFlushToZero())
      return {};
    auto delta = exponential.getInput().template getDefiningOp<BinaryOp>();
    if (!delta || delta.getOperatorKind() != BinaryOperator::Subtract ||
        !projectedFrom(delta.getRhs(), combinedMaximum))
      return {};
    auto normalized = project(delta.getLhs()).template getDefiningOp<SelectOp>();
    if (!normalized || !field(normalized.getCondition(), record, schema.validity) ||
        !field(normalized.getTrueValue(), record, schema.maximum) ||
        !projectedFrom(normalized.getFalseValue(), combinedMaximum))
      return {};
    return scale.getResult();
  };
  Value leftScale, rightScale;
  for (Value candidate : scaleCandidates) {
    if (!leftScale) leftScale = matchScale(candidate, left);
    if (!rightScale) rightScale = matchScale(candidate, right);
  }
  if (!leftScale || !rightScale) return std::nullopt;

  auto scaledField = [&](Value value, Value scale, mlir::BlockArgument record, unsigned index) {
    auto multiply = project(value).template getDefiningOp<BinaryOp>();
    return multiply && multiply.getOperatorKind() == BinaryOperator::Multiply &&
        ((projectedFrom(multiply.getLhs(), scale) && field(multiply.getRhs(), record, index)) ||
         (projectedFrom(multiply.getRhs(), scale) && field(multiply.getLhs(), record, index)));
  };
  auto scaledSum = [&](Value value, unsigned index) -> Value {
    auto add = project(value).template getDefiningOp<BinaryOp>();
    if (!add || add.getOperatorKind() != BinaryOperator::Add) return {};
    if (scaledField(add.getLhs(), leftScale, left, index) &&
        scaledField(add.getRhs(), rightScale, right, index))
      return add.getLhs();
    if (scaledField(add.getRhs(), leftScale, left, index) &&
        scaledField(add.getLhs(), rightScale, right, index))
      return add.getRhs();
    return {};
  };
  Value leftMassTerm = scaledSum(fields[schema.mass], schema.mass);
  Value leftMomentTerm = scaledSum(fields[schema.moment], schema.moment);
  if (!leftMassTerm || !leftMomentTerm) return std::nullopt;
  return OnlineSummaryCombineRelations{combinedMaximum, leftScale, rightScale,
                                       leftMassTerm, leftMomentTerm};
}

} // namespace intent

#endif
