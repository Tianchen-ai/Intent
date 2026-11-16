#include "Lowering.h"

namespace intent::bangc::block_lowering {
LogicalResult Lowering::scaledContraction(block::ScaledContractOp operation) {
  if (!operation.getLhsBatchAxes().empty() || !operation.getRhsBatchAxes().empty() ||
      operation.getLhsReductionAxes() != ArrayRef<int64_t>({1, 2}) ||
      operation.getRhsReductionAxes() != ArrayRef<int64_t>({0, 1}) ||
      operation.getLhsGroupSize() != operation.getRhsGroupSize())
    return operation.emitError("BANG C scaled contraction requires paired group and member axes");
  Value lhs = get(operation.getLhs()), rhs = get(operation.getRhs());
  Value lhsScale = get(operation.getLhsScale()), rhsScale = get(operation.getRhsScale());
  Value result = allocate(operation.getResult().getType());
  if (!result || failed(copy(get(operation.getAccumulator()), result))) return failure();
  auto leftShape = extents[lhs], rightShape = extents[rhs];
  if (leftShape.size() != 3 || rightShape.size() != 3 || leftShape[1] != rightShape[0])
    return operation.emitError("scaled contraction has no paired physical group domain");
  int64_t groupSize = operation.getLhsGroupSize();
  Value left = allocate(b.getF32Type(), {leftShape[0], groupSize});
  Value right = allocate(b.getF32Type(), {groupSize, rightShape[2]});
  auto integer = [&](int64_t value) -> Value { return b.create<arith::ConstantIntOp>(loc, value, 32); };
  auto fp = [&](double value) -> Value { return b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(value)); };
  auto decodeScale = [&](Value value) -> FailureOr<Value> {
    auto converted = castScalarValue(b, loc, value, b.getI32Type(), b.getIntegerType(8, false));
    if (failed(converted)) return failure();
    value = *converted;
    Value bits = b.create<arith::ShLIOp>(loc, value, integer(23));
    Value zero = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, value, integer(0));
    Value nan = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, value, integer(255));
    bits = b.create<arith::SelectOp>(loc, zero, integer(0x00400000), bits);
    bits = b.create<arith::SelectOp>(loc, nan, integer(0x7fc00000), bits);
    return Value(b.create<arith::BitcastOp>(loc, b.getF32Type(), bits));
  };
  if (failed(each({leftShape[1]}, [&](ValueRange group) -> LogicalResult {
    auto unpack = [&](Value input, Value scale, Value destination, bool leftSide, ScaledFormat format) {
      return each(extents[destination], [&](ValueRange coordinates) -> LogicalResult {
        Value k = coordinates[leftSide ? 1 : 0], free = coordinates[leftSide ? 0 : 1];
        Value carrier = format == ScaledFormat::E2M1 ? Value(b.create<arith::DivSIOp>(loc, k, index(2))) : k;
        SmallVector<Value> source = leftSide ? SmallVector<Value>{free, group[0], carrier}
                                            : SmallVector<Value>{group[0], carrier, free};
        Value value = read(input, source);
        if (format == ScaledFormat::E2M1) {
          auto converted = castScalarValue(b, loc, value, b.getI32Type(), b.getIntegerType(8, false));
          if (failed(converted)) return failure();
          Value parity = b.create<arith::RemSIOp>(loc, k, index(2));
          Value shift = b.create<arith::IndexCastOp>(loc, b.getI32Type(), parity);
          shift = b.create<arith::MulIOp>(loc, shift, integer(4));
          value = b.create<arith::ShRUIOp>(loc, *converted, shift);
          Value code = b.create<arith::AndIOp>(loc, value, integer(7));
          Value magnitude = fp(0);
          const double members[] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
          for (int i = 1; i < 8; ++i)
            magnitude = b.create<arith::SelectOp>(loc,
                b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, code, integer(i)), fp(members[i]), magnitude);
          Value sign = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
              b.create<arith::AndIOp>(loc, value, integer(8)), integer(0));
          value = b.create<arith::SelectOp>(loc, sign, b.create<arith::NegFOp>(loc, magnitude), magnitude);
        } else {
          auto converted = castScalarValue(b, loc, value, b.getF32Type());
          if (failed(converted)) return failure();
          value = *converted;
        }
        auto factor = decodeScale(read(scale, ValueRange{free, group[0]}));
        if (failed(factor)) return failure();
        return write(b.create<arith::MulFOp>(loc, value, *factor), destination, coordinates);
      });
    };
    if (failed(unpack(lhs, lhsScale, left, true, operation.getLhsFormat())) ||
        failed(unpack(rhs, rhsScale, right, false, operation.getRhsFormat()))) return failure();
    b.create<physical::MatMulOp>(loc, left, right, result, index(leftShape[0]), index(groupSize), index(rightShape[2]), false);
    return success();
  }))) return failure();
  values.map(operation.getResult(), result);
  return success();
}
}
