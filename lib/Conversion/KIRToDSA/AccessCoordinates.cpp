#include "Construction.h"
#include "Intent/Conversion/ScalarLowering.h"
#include "mlir/IR/TypeUtilities.h"

namespace intent::kir_to_dsa {
namespace {

FailureOr<Value> toIndex(OpBuilder &builder, Location loc, Value value,
                        Type logicalType) {
  // Logical index tensor elements use i64 local storage. Convert that actual
  // carrier back to index; only fixed-width integers override its signedness.
  Type signedness = isa<IntegerType>(logicalType) ? logicalType : Type();
  return castScalarValue(builder, loc, value, builder.getIndexType(), signedness);
}

} // namespace

IndexTermMaterialization Construction::indexMaterialization(Location loc) {
  IndexTermMaterialization result;
  result.constant = [this, loc](int64_t value) { return index(loc, value); };
  result.scalar = [this, loc](Value source) -> FailureOr<Value> {
    return toIndex(b, loc, get(source), source.getType());
  };
  result.extent = [this, loc](Value source, unsigned axis) -> FailureOr<Value> {
    Value extent = logicalExtent(source, axis, loc);
    return extent ? FailureOr<Value>(extent) : FailureOr<Value>(failure());
  };
  result.domain = [this](Value source) -> FailureOr<IndexRange> {
    if (!bindDomain(source)) return failure();
    const auto &domain = domains.find(source)->second;
    return IndexRange{domain.begin, domain.end, domain.step, domain.extent};
  };
  result.add = [this, loc](Value a, Value c) -> FailureOr<Value> { return add(loc, a, c); };
  result.subtract = [this, loc](Value a, Value c) -> FailureOr<Value> { return sub(loc, a, c); };
  result.multiply = [this, loc](Value a, Value c) -> FailureOr<Value> { return mul(loc, a, c); };
  result.ceilDivide = [this, loc](Value a, Value c) -> FailureOr<Value> {
    return Value(b.createOrFold<arith::CeilDivSIOp>(loc, a, c));
  };
  result.maximum = [this, loc](Value a, Value c) -> FailureOr<Value> {
    return Value(b.createOrFold<arith::MaxSIOp>(loc, a, c));
  };
  return result;
}

FailureOr<SmallVector<Value>> Construction::accessCoordinates(
    const IndexRelationFact &relation, const LocalShape &shape,
    ValueRange coordinates, Location loc) {
  if (shape.size() != coordinates.size()) return failure();
  SmallVector<Value> members;
  for (auto [axis, coordinate] : llvm::enumerate(coordinates))
    members.push_back(add(loc, shape[axis].begin, coordinate));
  return materializeIndexCoordinates(relation, members, indexMaterialization(loc),
      [&](Value original, ValueRange indices) -> FailureOr<Value> {
    Value value = get(original);
    auto found = localShapes.find(value);
    if (!value || found == localShapes.end() || found->second.size() != indices.size())
      return failure();
    SmallVector<Value> local;
    for (auto [axis, coordinate] : llvm::enumerate(indices)) {
      const auto &binding = found->second[axis];
      local.push_back(matchPattern(binding.extent, m_One()) ? index(loc, 0)
          : sub(loc, coordinate, binding.begin));
    }
    return toIndex(b, loc, loadLocal(loc, value, local),
                   getElementTypeOrSelf(original.getType()));
  });
}

FailureOr<AffineIndices> Construction::affineAccessTerm(
    Value source, const IndexTermFact &term, unsigned resultRank, Location loc) {
  auto reified = materializeIndexTerm(source, term, indexMaterialization(loc));
  if (failed(reified) || !term.sourceAxis) return failure();
  AffineIndices result{Value(), SmallVector<Value>(resultRank, index(loc, 0))};
  if (reified->range) {
    result.base = reified->range->begin;
    result.steps[term.resultAxes.front()] = reified->range->step;
  } else if (reified->coordinate) {
    result.base = reified->coordinate;
  } else if (reified->tensor) {
    auto input = affineIndex(reified->tensor);
    if (failed(input) || input->steps.size() != term.indexAxes.size()) return failure();
    result.base = input->base;
    for (auto [axis, mapped] : llvm::enumerate(term.indexAxes))
      if (!singletonAxis(reified->tensor, axis))
        result.steps[mapped] = add(loc, result.steps[mapped], input->steps[axis]);
  } else return failure();
  return result;
}

FailureOr<AffineIndices> Construction::affineIndex(Value original) {
  Location loc = original.getLoc();
  auto type = dyn_cast<RankedTensorType>(original.getType());
  if (!type) {
    auto scalar = indexMaterialization(loc).scalar(original);
    if (failed(scalar)) return failure();
    return AffineIndices{*scalar, {}};
  }
  if (!type.getElementType().isIndex() && !type.getElementType().isInteger(64)) return failure();
  if (auto indices = original.getDefiningOp<IndicesOp>()) {
    if (!bindDomain(indices.getSource())) return failure();
    Domain domain = domains.lookup(indices.getSource());
    return AffineIndices{domain.begin, {domain.step}};
  }
  Operation *op = original.getDefiningOp();
  if (!op) return failure();
  if (isa<BroadcastOp, FullOp>(op)) {
    auto source = affineIndex(op->getOperand(0));
    if (failed(source)) return failure();
    AffineIndices result{source->base, SmallVector<Value>(type.getRank(), index(loc, 0))};
    if (auto input = dyn_cast<RankedTensorType>(op->getOperand(0).getType())) {
      if (input.getRank() > type.getRank()) return failure();
      unsigned leading = type.getRank() - input.getRank();
      for (unsigned axis = 0; axis < input.getRank(); ++axis) {
        if (singletonAxis(op->getOperand(0), axis)) continue;
        if (!equalAxisExtent(op->getOperand(0), axis, original, leading + axis)) return failure();
        result.steps[leading + axis] = source->steps[axis];
      }
    }
    return result;
  }
  if (auto cast = dyn_cast<CastOp>(op)) {
    auto input = dyn_cast<RankedTensorType>(cast.getInput().getType());
    if (!input || (!input.getElementType().isIndex() && !input.getElementType().isInteger(64))) return failure();
    return affineIndex(cast.getInput());
  }
  if (auto transpose = dyn_cast<TransposeOp>(op)) {
    auto source = affineIndex(transpose.getInput());
    if (failed(source)) return failure();
    AffineIndices result{source->base, {}};
    for (Attribute axis : transpose.getPermutation()) result.steps.push_back(source->steps[cast<IntegerAttr>(axis).getInt()]);
    return result;
  }
  if (auto gather = dyn_cast<GatherOp>(op)) {
    if (gather.getValid() && !constantTrue(gather.getValid())) return failure();
    auto relation = analysis.indexRelation(gather);
    if (failed(relation)) return failure();
    auto source = affineIndex(relation->source);
    if (failed(source)) return failure();
    AffineIndices result{source->base, SmallVector<Value>(type.getRank(), index(loc, 0))};
    for (const auto &term : relation->terms) {
      if (!term.sourceAxis) continue;
      auto coordinate = affineAccessTerm(relation->source, term, type.getRank(), loc);
      if (failed(coordinate)) return failure();
      Value coefficient = source->steps[*term.sourceAxis];
      result.base = add(loc, result.base, mul(loc, coefficient, coordinate->base));
      for (unsigned axis = 0; axis < result.steps.size(); ++axis)
        result.steps[axis] = add(loc, result.steps[axis], mul(loc, coefficient, coordinate->steps[axis]));
    }
    return result;
  }
  auto binary = dyn_cast<BinaryOp>(op);
  if (!binary) return failure();
  auto lhs = affineIndex(binary.getLhs()), rhs = affineIndex(binary.getRhs());
  if (failed(lhs) || failed(rhs)) return failure();
  if (lhs->steps.empty()) lhs->steps.assign(type.getRank(), index(loc, 0));
  if (rhs->steps.empty()) rhs->steps.assign(type.getRank(), index(loc, 0));
  if (lhs->steps.size() != type.getRank() || rhs->steps.size() != type.getRank()) return failure();
  AffineIndices result{Value(), SmallVector<Value>(type.getRank())};
  if (binary.getOperatorKind() == BinaryOperator::Add || binary.getOperatorKind() == BinaryOperator::Subtract) {
    bool plus = binary.getOperatorKind() == BinaryOperator::Add;
    result.base = plus ? add(loc, lhs->base, rhs->base) : sub(loc, lhs->base, rhs->base);
    for (unsigned axis = 0; axis < type.getRank(); ++axis)
      result.steps[axis] = plus ? add(loc, lhs->steps[axis], rhs->steps[axis]) : sub(loc, lhs->steps[axis], rhs->steps[axis]);
    return result;
  }
  if (binary.getOperatorKind() == BinaryOperator::Multiply) {
    auto uniform = [](const AffineIndices &map) { return llvm::all_of(map.steps, [](Value step) { return matchPattern(step, m_Zero()); }); };
    const AffineIndices *variable = nullptr; Value factor;
    if (uniform(*lhs)) { variable = &*rhs; factor = lhs->base; }
    else if (uniform(*rhs)) { variable = &*lhs; factor = rhs->base; }
    if (variable) {
      result.base = mul(loc, lhs->base, rhs->base);
      for (unsigned axis = 0; axis < type.getRank(); ++axis) result.steps[axis] = mul(loc, factor, variable->steps[axis]);
      return result;
    }
  }
  return failure();
}

} // namespace intent::kir_to_dsa
