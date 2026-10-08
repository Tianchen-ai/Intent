#include "../Construction.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

namespace intent::kir_to_cpu {

IndexTermMaterialization Construction::indexMaterialization(OpBuilder &nested,
                                                            Location loc) {
  IndexTermMaterialization callbacks;
  callbacks.constant = [&nested, loc](int64_t value) -> Value {
    return nested.create<arith::ConstantIndexOp>(loc, value);
  };
  callbacks.scalar = [this, &nested, loc](Value original) -> FailureOr<Value> {
    Value actual = values.lookupOrNull(original);
    if (!actual) return failure();
    return indexValue(nested, actual, original.getType(), loc);
  };
  callbacks.extent = [this, &nested, loc](Value original,
                                       unsigned axis) -> FailureOr<Value> {
    Value actual = values.lookupOrNull(original);
    if (!actual) return failure();
    return dimension(nested, loc, actual, axis);
  };
  callbacks.domain = [this](Value original) -> FailureOr<IndexRange> {
    auto found = domains.find(original);
    if (found == domains.end()) return failure();
    const Domain &domain = found->second;
    return IndexRange{domain.begin, domain.end, domain.step, domain.extent};
  };
  callbacks.add = [&nested, loc](Value lhs, Value rhs) -> FailureOr<Value> {
    return Value(nested.createOrFold<arith::AddIOp>(loc, lhs, rhs));
  };
  callbacks.subtract = [&nested, loc](Value lhs, Value rhs) -> FailureOr<Value> {
    return Value(nested.createOrFold<arith::SubIOp>(loc, lhs, rhs));
  };
  callbacks.multiply = [&nested, loc](Value lhs, Value rhs) -> FailureOr<Value> {
    return Value(nested.createOrFold<arith::MulIOp>(loc, lhs, rhs));
  };
  callbacks.ceilDivide = [&nested, loc](Value lhs, Value rhs) -> FailureOr<Value> {
    return Value(nested.createOrFold<arith::CeilDivSIOp>(loc, lhs, rhs));
  };
  callbacks.maximum = [&nested, loc](Value lhs, Value rhs) -> FailureOr<Value> {
    return Value(nested.createOrFold<arith::MaxSIOp>(loc, lhs, rhs));
  };
  return callbacks;
}

FailureOr<SmallVector<Value>> Construction::indexedCoordinates(
    const IndexRelationFact &fact, ValueRange members, OpBuilder &nested,
    Location loc) {
  auto callbacks = indexMaterialization(nested, loc);
  auto extract = [this, &nested, loc](Value original,
                                     ValueRange indices) -> FailureOr<Value> {
    Value actual = values.lookupOrNull(original);
    if (!actual) return failure();
    Value scalar = extractElement(nested, loc, actual, indices);
    return indexValue(nested, scalar, getElementTypeOrSelf(original.getType()),
                      loc);
  };
  auto coordinates = materializeIndexCoordinates(fact, members, callbacks,
                                                extract);
  if (failed(coordinates))
    emitError(loc, "CPU construction cannot reify the canonical access coordinates");
  return coordinates;
}

bool Construction::hasTensorIndices(const IndexRelationFact &fact) {
  return llvm::any_of(fact.terms, [](const IndexTermFact &term) {
    return llvm::any_of(term.operands, [](Value value) {
      return value && isa<RankedTensorType>(value.getType());
    });
  });
}

} // namespace intent::kir_to_cpu
