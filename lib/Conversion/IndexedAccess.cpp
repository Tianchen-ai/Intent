#include "Intent/Conversion/IndexedAccess.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

namespace intent {
namespace {

struct ReifiedSlice {
  OpFoldResult begin;
  OpFoldResult end;
  OpFoldResult step;
  OpFoldResult count;
};

FailureOr<ReifiedSlice> reifySlice(
    const IndexTermFact &term, const LogicalShapeReification &reification,
    function_ref<FailureOr<OpFoldResult>()> sourceExtent) {
  if (term.kind != 5 || term.operands.size() != 3 ||
      term.staticValues.size() != 3 || !reification.leaf ||
      !reification.subtract || !reification.ceilDivide ||
      !reification.maximum)
    return failure();

  auto constant = [&](int64_t value) {
    return reification.leaf(TensorExtentFact{value, {}});
  };
  auto bound = [&](unsigned slot) -> FailureOr<OpFoldResult> {
    if (Value value = term.operands[slot])
      return reification.leaf(TensorExtentFact{std::nullopt, value});
    if (auto value = term.staticValues[slot]) return constant(*value);
    if (slot == 1) return sourceExtent();
    return constant(slot == 0 ? 0 : 1);
  };

  auto begin = bound(0);
  auto end = bound(1);
  auto step = bound(2);
  if (failed(begin) || failed(end) || failed(step)) return failure();
  if (auto value = getConstantIntValue(*step); value && *value == 0)
    return failure();
  auto distance = reification.subtract(*end, *begin);
  if (failed(distance)) return failure();
  auto quotient = reification.ceilDivide(*distance, *step);
  auto zero = constant(0);
  if (failed(quotient) || failed(zero)) return failure();
  auto count = reification.maximum(*quotient, *zero);
  if (failed(count)) return failure();
  return ReifiedSlice{*begin, *end, *step, *count};
}

LogicalShapeReification shapeCallbacks(
    const IndexTermMaterialization &materialization) {
  LogicalShapeReification result;
  result.leaf = [&](const TensorExtentFact &fact) -> FailureOr<OpFoldResult> {
    if (fact.constant && materialization.constant) {
      Value value = materialization.constant(*fact.constant);
      if (value) return OpFoldResult(value);
    }
    if (fact.value && materialization.scalar) {
      auto value = materialization.scalar(fact.value);
      if (succeeded(value) && *value) return OpFoldResult(*value);
    }
    return failure();
  };
  using Binary = std::function<FailureOr<Value>(Value, Value)>;
  auto adapt = [](const Binary &binary) {
    return [binary](OpFoldResult lhs, OpFoldResult rhs)
               -> FailureOr<OpFoldResult> {
      if (!binary) return failure();
      auto value = binary(cast<Value>(lhs), cast<Value>(rhs));
      if (failed(value) || !*value) return failure();
      return OpFoldResult(*value);
    };
  };
  result.subtract = adapt(materialization.subtract);
  result.ceilDivide = adapt(materialization.ceilDivide);
  result.maximum = adapt(materialization.maximum);
  return result;
}

} // namespace

FailureOr<ReifiedIndexTerm> materializeIndexTerm(
    Value source, const IndexTermFact &term,
    const IndexTermMaterialization &materialization) {
  if (term.kind == 1) return ReifiedIndexTerm{};
  if (!term.sourceAxis || !source) return failure();
  auto extent = [&]() -> FailureOr<Value> {
    if (!materialization.extent) return failure();
    return materialization.extent(source, *term.sourceAxis);
  };

  switch (term.kind) {
  case 0: {
    if (!materialization.constant) return failure();
    auto count = extent();
    if (failed(count) || !*count) return failure();
    Value zero = materialization.constant(0);
    Value one = materialization.constant(1);
    if (!zero || !one) return failure();
    return ReifiedIndexTerm{{}, {}, IndexRange{zero, *count, one, *count}};
  }
  case 2: {
    if (term.staticValues.size() != 1 || !term.staticValues.front() ||
        !materialization.constant)
      return failure();
    int64_t literal = *term.staticValues.front();
    Value coordinate = materialization.constant(literal);
    if (!coordinate) return failure();
    if (literal < 0) {
      auto size = extent();
      if (failed(size) || !materialization.add) return failure();
      auto adjusted = materialization.add(coordinate, *size);
      if (failed(adjusted) || !*adjusted) return failure();
      coordinate = *adjusted;
    }
    return ReifiedIndexTerm{coordinate, {}, std::nullopt};
  }
  case 3: {
    if (term.operands.size() != 1 || !term.operands.front()) return failure();
    Value index = term.operands.front();
    if (isa<RankedTensorType>(index.getType()))
      return ReifiedIndexTerm{{}, index, std::nullopt};
    if (!materialization.scalar) return failure();
    auto coordinate = materialization.scalar(index);
    if (failed(coordinate) || !*coordinate) return failure();
    return ReifiedIndexTerm{*coordinate, {}, std::nullopt};
  }
  case 4: {
    if (term.operands.size() != 1 || !term.operands.front() ||
        !materialization.domain)
      return failure();
    auto range = materialization.domain(term.operands.front());
    if (failed(range) || !range->begin || !range->end || !range->step ||
        !range->count)
      return failure();
    return ReifiedIndexTerm{{}, {}, *range};
  }
  case 5: {
    auto callbacks = shapeCallbacks(materialization);
    auto slice = reifySlice(term, callbacks, [&]() -> FailureOr<OpFoldResult> {
      auto size = extent();
      if (failed(size) || !*size) return failure();
      return OpFoldResult(*size);
    });
    if (failed(slice)) return failure();
    return ReifiedIndexTerm{
        {}, {}, IndexRange{cast<Value>(slice->begin), cast<Value>(slice->end),
                           cast<Value>(slice->step), cast<Value>(slice->count)}};
  }
  default:
    return failure();
  }
}

FailureOr<SmallVector<Value>> materializeIndexCoordinates(
    const IndexRelationFact &relation, ValueRange members,
    const IndexTermMaterialization &materialization,
    function_ref<FailureOr<Value>(Value, ValueRange)> extractIndex) {
  if (members.size() != relation.resultDimensionIdentities.size())
    return failure();
  SmallVector<Value> coordinates;
  for (const IndexTermFact &term : relation.terms) {
    auto value = materializeIndexTerm(relation.source, term, materialization);
    if (failed(value)) return failure();
    if (!term.sourceAxis) continue;
    Value coordinate = value->coordinate;
    if (value->range) {
      if (term.resultAxes.size() != 1 || term.resultAxes.front() >= members.size() ||
          !materialization.multiply || !materialization.add)
        return failure();
      auto scaled = materialization.multiply(members[term.resultAxes.front()],
                                            value->range->step);
      if (failed(scaled)) return failure();
      auto shifted = materialization.add(value->range->begin, *scaled);
      if (failed(shifted)) return failure();
      coordinate = *shifted;
    } else if (value->tensor) {
      auto type = cast<RankedTensorType>(value->tensor.getType());
      if (term.indexAxes.size() != type.getRank()) return failure();
      SmallVector<Value> indices;
      for (auto [axis, target] : llvm::enumerate(term.indexAxes)) {
        if (target >= members.size()) return failure();
        if (type.getDimSize(axis) == 1) {
          if (!materialization.constant) return failure();
          indices.push_back(materialization.constant(0));
        } else {
          indices.push_back(members[target]);
        }
        if (!indices.back()) return failure();
      }
      auto loaded = extractIndex(value->tensor, indices);
      if (failed(loaded)) return failure();
      coordinate = *loaded;
    }
    if (!coordinate || *term.sourceAxis != coordinates.size()) return failure();
    coordinates.push_back(coordinate);
  }
  if (coordinates.size() != relation.sourceRank) return failure();
  return coordinates;
}

FailureOr<OpFoldResult> reifyIndexSliceExtent(
    CanonicalKernelAnalysis &analysis, Value source, const IndexTermFact &term,
    const LogicalShapeReification &reification) {
  if (!term.sourceAxis) return failure();
  auto slice = reifySlice(term, reification, [&]() {
    return reifyLogicalExtent(analysis, source, *term.sourceAxis, reification);
  });
  if (failed(slice)) return failure();
  return slice->count;
}

} // namespace intent
