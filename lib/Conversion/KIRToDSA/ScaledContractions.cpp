#include "Construction.h"

namespace intent::kir_to_dsa {

LogicalResult Construction::scaledMatMul(ScaledContractOp operation,
                                        const LocalShape &shape) {
  Location loc = operation.getLoc();
  if (shape.size() != 2 || operation.getLhsGroupSize() != operation.getRhsGroupSize() ||
      operation.getLhsGroupSize() <= 0)
    return operation.emitError("scaled matrix requires the canonical paired group schema");
  int64_t groupSize = operation.getLhsGroupSize();
  auto plan = planExecutionSlices(*operation->getBlock(), 0,
      {{operation.getLhs(), 1}, {operation.getRhs(), 0},
       {operation.getLhsScale(), 1}, {operation.getRhsScale(), 1}});
  auto domain = plan ? sliceDomain(*plan, 1, true) : std::nullopt;
  if (!domain) return operation.emitError("scaled matrix group supply has no exact common traversal");
  Value output = allocateTensor(loc, b.getF32Type(), shape);
  auto savedValues = values;
  auto savedProducts = products;
  auto savedSlices = valueSlices;
  auto integer = [&](int64_t value) -> Value { return b.create<arith::ConstantIntOp>(loc, value, 32); };
  auto fp = [&](double value) -> Value { return b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(value)); };
  auto decodeScale = [&](Value value) -> Value {
    value = b.create<arith::ExtUIOp>(loc, b.getI32Type(), value);
    Value bits = b.create<arith::ShLIOp>(loc, value, integer(23));
    Value zero = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, value, integer(0));
    Value nan = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, value, integer(255));
    bits = b.create<arith::SelectOp>(loc, zero, integer(0x00400000), bits);
    bits = b.create<arith::SelectOp>(loc, nan, integer(0x7fc00000), bits);
    return b.create<arith::BitcastOp>(loc, b.getF32Type(), bits);
  };
  auto status = loop(loc, index(loc, 0), domain->extent, index(loc, 1), [&](Value group) -> LogicalResult {
    bindExecutionSlice(*plan, *domain, group, index(loc, 1));
    Value lhs = get(operation.getLhs()), rhs = get(operation.getRhs());
    Value lhsScale = get(operation.getLhsScale()), rhsScale = get(operation.getRhsScale());
    if (!lhs || !rhs || !lhsScale || !rhsScale)
      return operation.emitError("scaled matrix input snapshot is unavailable");
    LocalAxis depth{index(loc, groupSize), index(loc, 0), index(loc, groupSize), groupSize};
    LocalShape aShape{shape[0], depth}, bShape{depth, shape[1]};
    Value left = allocateTensor(loc, b.getF32Type(), aShape);
    Value right = allocateTensor(loc, b.getF32Type(), bShape);
    auto unpack = [&](Value source, Value scale, Value destination,
                      const LocalShape &packed, bool leftSide, ScaledFormat format) {
      return eachElement(loc, packed, [&](ValueRange coordinates) {
        Value k = coordinates[leftSide ? 1 : 0], free = coordinates[leftSide ? 0 : 1];
        Value carrier = format == ScaledFormat::E2M1
            ? Value(b.create<arith::DivSIOp>(loc, k, index(loc, 2))) : k;
        SmallVector<Value> sourceCoordinates = leftSide
            ? SmallVector<Value>{free, index(loc, 0), carrier}
            : SmallVector<Value>{index(loc, 0), carrier, free};
        Value value = loadLocal(loc, source, sourceCoordinates);
        if (format == ScaledFormat::E2M1) {
          value = b.create<arith::ExtUIOp>(loc, b.getI32Type(), value);
          Value parity = b.create<arith::RemSIOp>(loc, k, index(loc, 2));
          Value shift = b.create<arith::IndexCastOp>(loc, b.getI32Type(), parity);
          shift = b.create<arith::MulIOp>(loc, shift, integer(4));
          value = b.create<arith::ShRUIOp>(loc, value, shift);
          Value code = b.create<arith::AndIOp>(loc, value, integer(7));
          Value magnitude = fp(0);
          const double members[] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
          for (int i = 1; i < 8; ++i)
            magnitude = b.create<arith::SelectOp>(loc,
                b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, code, integer(i)),
                fp(members[i]), magnitude);
          Value sign = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
              b.create<arith::AndIOp>(loc, value, integer(8)), integer(0));
          value = b.create<arith::SelectOp>(loc, sign,
              b.create<arith::NegFOp>(loc, magnitude), magnitude);
        } else {
          value = scalarCast(loc, value, b.getF32Type());
        }
        Value factor = decodeScale(loadLocal(loc, scale, ValueRange{free, index(loc, 0)}));
        value = b.create<arith::MulFOp>(loc, value, factor);
        storeLocal(loc, value, destination, coordinates);
        return success();
      });
    };
    if (failed(unpack(lhs, lhsScale, left, aShape, true, operation.getLhsFormat())) ||
        failed(unpack(rhs, rhsScale, right, bShape, false, operation.getRhsFormat())))
      return failure();
    b.create<dsa::MatMulOp>(loc, left, right, output, shape[0].count,
        index(loc, groupSize), shape[1].count, b.getBoolAttr(false));
    return success();
  });
  values = std::move(savedValues);
  products = std::move(savedProducts);
  valueSlices = std::move(savedSlices);
  if (failed(status)) return failure();
  values.map(operation.getResult(), output);
  return success();
}

} // namespace intent::kir_to_dsa
