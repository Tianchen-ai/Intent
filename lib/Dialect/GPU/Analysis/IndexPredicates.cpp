#include "Intent/Dialect/GPU/Analysis/IndexPredicates.h"
#include "Intent/Dialect/GPU/Analysis/IndexRelations.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalExpressionBounds.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;

namespace intent::gpu {
namespace {

Value withoutProjection(Value value) {
  while (Operation *operation = value.getDefiningOp()) {
    if (!isa<BroadcastOp, SplatOp, ReshapeOp, TransposeOp>(operation))
      break;
    value = operation->getOperand(0);
  }
  return value;
}

std::optional<int64_t> integer(Value value) {
  auto constant = dyn_cast_or_null<IntegerAttr>(
      UniformValueAnalysis(describeUniformValue).evaluate(value));
  return constant && constant.getType().isIndex()
             ? std::optional<int64_t>(constant.getInt()) : std::nullopt;
}

} // namespace

std::optional<bool> proveRangeComparison(CompareOp comparison) {
  Value source = comparison.getLhs();
  Value bound = withoutProjection(comparison.getRhs());
  if (!bound.getType().isIndex())
    return std::nullopt;
  IndexRelations relations;
  if (auto range = withoutProjection(source).getDefiningOp<MakeRangeOp>();
      range && isUnitStepRange(range) &&
      relations.same(range.getStart(), bound) &&
      relations.alignedUnitWindow(range.getStart(), range.getExtent())) {
    // The current aligned window is [start, start + extent - 1] in signed
    // coordinates, even if the arithmetic that produced start was modular.
    // This removes only its redundant lower test, not any resource boundary.
    if (comparison.getPredicate() == ComparePredicate::Ge)
      return true;
    if (comparison.getPredicate() == ComparePredicate::Lt)
      return false;
  }
  if (comparison.getPredicate() == ComparePredicate::Lt &&
      relations.coordinateLessThan(source, bound))
    return true;
  if (comparison.getPredicate() == ComparePredicate::Ge) {
    if (relations.coordinateLessThan(source, bound))
      return false;
    if (integer(bound) == 0 && relations.nonnegative(source))
      return true;
  }
  return std::nullopt;
}

std::optional<IndexComparisonBound> queryIndexComparisonBound(CompareOp comparison) {
  if (comparison.getPredicate() != ComparePredicate::Lt ||
      !comparison.getLhs().getType().isIndex() ||
      !comparison.getRhs().getType().isIndex() ||
      queryLaunchExpression(comparison.getLhs()))
    return std::nullopt;
  auto upper = queryNonNegativeIndexUpperBound(comparison.getLhs());
  auto limit = queryLaunchExpression(comparison.getRhs());
  if (!upper || !limit || !isShapeBound(upper) || !isShapeBound(limit))
    return std::nullopt;
  // A loop-local bound is not automatically safe to evaluate in the host's
  // specialization of an empty loop. Prove the entire checked expression.
  auto kernel = comparison->getParentOfType<func::FuncOp>();
  if (!queryPhysicalExpressionRange(upper, kernel) ||
      !queryPhysicalExpressionRange(limit, kernel))
    return std::nullopt;
  for (Operation *user : comparison.getResult().getUsers()) {
    auto logicalOr = dyn_cast<BinaryOp>(user);
    if (!logicalOr || logicalOr.getOperatorKind() != BinaryOperator::LogicalOr)
      continue;
    Value other = logicalOr.getLhs() == comparison.getResult()
                      ? logicalOr.getRhs() : logicalOr.getLhs();
    auto proof = other.getDefiningOp<CompareOp>();
    if (!proof || proof.getPredicate() != ComparePredicate::Lt) continue;
    auto left = queryLaunchExpression(proof.getLhs());
    auto right = queryLaunchExpression(proof.getRhs());
    if (left && right && left == upper && right == limit)
      return std::nullopt;
  }
  return IndexComparisonBound{upper, limit};
}

} // namespace intent::gpu
