#include "Intent/Analysis/RegionSemantics.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
namespace intent {

std::optional<PredicateIntervals> partitionCoordinatePredicate(
    UniformPredicate predicate, CoordinateInterval source, CoordinateInterval capture,
    Value zero, Value one, const BuildIndexExpression &build) {
  using K = UniformKind;
  bool upper = predicate == UniformPredicate::Less || predicate == UniformPredicate::LessEqual;
  bool lower = predicate == UniformPredicate::Greater || predicate == UniformPredicate::GreaterEqual;
  if (!upper && !lower) return std::nullopt;
  Value length = build(K::Subtract, source.end, source.begin);
  auto clamp = [&](Value value) { return build(K::Minimum, length, build(K::Maximum, zero, value)); };
  Value first = build(K::Subtract, capture.begin, source.begin);
  Value end = build(K::Subtract, capture.end, source.begin);
  if (upper) {
    if (predicate == UniformPredicate::LessEqual) first = build(K::Add, first, one);
    else end = build(K::Subtract, end, one);
    return PredicateIntervals{zero, clamp(first), zero, clamp(end)};
  }
  if (predicate == UniformPredicate::Greater) first = build(K::Add, first, one);
  else end = build(K::Subtract, end, one);
  return PredicateIntervals{clamp(end), length, clamp(first), length};
}

bool isBooleanUnion(const UniformValueAnalysis &values, Value result, Value lhs, Value rhs) {
  auto boolean = IntegerType::get(result.getContext(), 1);
  for (unsigned state = 0; state < 4; ++state) {
    UniformBindings bindings;
    bool left = state & 1, right = state & 2;
    bindings[lhs] = IntegerAttr::get(boolean, left);
    bindings[rhs] = IntegerAttr::get(boolean, right);
    if (uniformBoolean(values.evaluate(result, bindings)) != (left || right)) return false;
  }
  return true;
}

bool hasTrueStateInvariant(const UniformValueAnalysis &values, Value result, Value incomingState) {
  UniformBindings bindings;
  bindings[incomingState] = IntegerAttr::get(IntegerType::get(result.getContext(), 1), 1);
  return uniformBoolean(values.evaluate(result, bindings)) == true;
}

}
