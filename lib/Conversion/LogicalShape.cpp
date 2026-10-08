#include "Intent/Conversion/LogicalShape.h"
#include "Intent/Conversion/IndexedAccess.h"
#include "Intent/Dialect/Intent/IR/IntentOps.h"
#include "llvm/ADT/ScopeExit.h"

using namespace mlir;

namespace intent {

FailureOr<OpFoldResult> reifyLogicalExtent(
    CanonicalKernelAnalysis &analysis, Value value, unsigned axis,
    const LogicalShapeReification &reification, ArrayRef<unsigned> fieldPath) {
  struct Query {
    Value value;
    unsigned axis;
    SmallVector<unsigned, 2> path;
  };
  SmallVector<Query> active;
  SmallVector<std::pair<Query, OpFoldResult>> bindings;
  auto lookup = [&](Value source, ArrayRef<unsigned> path,
                    unsigned dimension) -> OpFoldResult {
    if (!reification.lookup) return {};
    for (const auto &entry : bindings)
      if (entry.first.value == source && entry.first.axis == dimension &&
          ArrayRef<unsigned>(entry.first.path) == path)
        return entry.second;
    OpFoldResult bound = reification.lookup(source, path, dimension);
    if (bound)
      bindings.push_back({{source, dimension, {path.begin(), path.end()}}, bound});
    return bound;
  };
  std::function<FailureOr<OpFoldResult>(Value, unsigned, ArrayRef<unsigned>)>
      reify = [&](Value source, unsigned dimension,
                  ArrayRef<unsigned> path) -> FailureOr<OpFoldResult> {
    if (OpFoldResult bound = lookup(source, path, dimension)) return bound;
    if (llvm::any_of(active, [&](const Query &query) {
          return query.value == source && query.axis == dimension &&
                 ArrayRef<unsigned>(query.path) == path;
        })) return failure();
    active.push_back({source, dimension, {path.begin(), path.end()}});
    auto finish = llvm::make_scope_exit([&] { active.pop_back(); });
    TensorExtentFact fact = analysis.tensorExtent(
        source, dimension, path,
        [&](Value node, ArrayRef<unsigned> fields, unsigned axis) {
          return static_cast<bool>(lookup(node, fields, axis));
        });
    if (!fact.isKnown()) return failure();
    if (fact.source)
      if (OpFoldResult bound = lookup(fact.source, fact.fieldPath, fact.axis))
        return bound;
    if (fact.source) {
      Operation *operation = fact.source.getDefiningOp();
      if (isa_and_nonnull<IndexedAccessOpInterface>(operation)) {
        auto relation = analysis.indexRelation(operation);
        if (failed(relation)) return failure();
        for (const IndexTermFact &term : relation->terms)
          if (term.kind == 5 && llvm::is_contained(term.resultAxes, fact.axis))
            return reifyIndexSliceExtent(analysis, relation->source, term,
                                        reification);
      }
    }
    if (!fact.inferred) return reification.leaf(fact);
    auto reshape = fact.source.getDefiningOp<ReshapeOp>();
    if (!reshape || !fact.fieldPath.empty()) return failure();
    Value input = reshape.getInputs().front();
    auto inputType = cast<RankedTensorType>(input.getType());
    auto resultType = cast<RankedTensorType>(reshape.getResult().getType());
    auto one = reification.leaf(TensorExtentFact{1, {}});
    if (failed(one)) return failure();
    OpFoldResult numerator = *one, denominator = *one;
    for (unsigned current = 0; current < inputType.getRank(); ++current) {
      auto extent = reify(input, current, {});
      if (failed(extent)) return failure();
      auto product = reification.multiply(numerator, *extent);
      if (failed(product)) return failure();
      numerator = *product;
    }
    for (unsigned current = 0; current < resultType.getRank(); ++current) {
      if (current == fact.axis) continue;
      auto known = analysis.tensorExtent(reshape.getResult(), current);
      if (known.constant && *known.constant == 0) return failure();
      auto extent = reify(reshape.getResult(), current, {});
      if (failed(extent)) return failure();
      auto product = reification.multiply(denominator, *extent);
      if (failed(product)) return failure();
      denominator = *product;
    }
    return reification.exactDivide(numerator, denominator);
  };
  return reify(value, axis, fieldPath);
}

FailureOr<Value> materializeLogicalExtent(
    CanonicalKernelAnalysis &analysis, Value value, unsigned axis,
    const LogicalShapeMaterialization &materialization,
    ArrayRef<unsigned> fieldPath) {
  LogicalShapeReification callbacks;
  callbacks.lookup = [&](Value source, ArrayRef<unsigned> path, unsigned dimension)
      -> OpFoldResult {
    if (!materialization.lookup) return {};
    Value bound = materialization.lookup(source, path, dimension);
    return bound ? OpFoldResult(bound) : OpFoldResult();
  };
  callbacks.leaf = [&](const TensorExtentFact &fact) -> FailureOr<OpFoldResult> {
    auto result = materialization.leaf(fact);
    if (failed(result)) return failure();
    return OpFoldResult(*result);
  };
  callbacks.multiply = [&](OpFoldResult lhs, OpFoldResult rhs)
      -> FailureOr<OpFoldResult> {
    auto result = materialization.multiply(cast<Value>(lhs), cast<Value>(rhs));
    if (failed(result)) return failure();
    return OpFoldResult(*result);
  };
  callbacks.exactDivide = [&](OpFoldResult lhs, OpFoldResult rhs)
      -> FailureOr<OpFoldResult> {
    auto result = materialization.exactDivide(cast<Value>(lhs), cast<Value>(rhs));
    if (failed(result)) return failure();
    return OpFoldResult(*result);
  };
  using Binary = std::function<FailureOr<Value>(Value, Value)>;
  auto adapt = [](const Binary &binary) {
    return [binary](OpFoldResult lhs, OpFoldResult rhs)
               -> FailureOr<OpFoldResult> {
      if (!binary) return failure();
      auto result = binary(cast<Value>(lhs), cast<Value>(rhs));
      if (failed(result)) return failure();
      return OpFoldResult(*result);
    };
  };
  callbacks.subtract = adapt(materialization.subtract);
  callbacks.ceilDivide = adapt(materialization.ceilDivide);
  callbacks.maximum = adapt(materialization.maximum);
  auto result = reifyLogicalExtent(analysis, value, axis, callbacks, fieldPath);
  if (failed(result)) return failure();
  return cast<Value>(*result);
}

} // namespace intent
