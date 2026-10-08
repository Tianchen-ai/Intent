#include "Construction.h"

namespace intent::kir_to_dsa {
namespace {

// The record ABI owns byte order and member positions; execution tiling does
// not change them. All arithmetic is explicit standard IR, including each R32.
struct RecordValues {
  OpBuilder &b;
  Location loc;
  Value index(int64_t value) { return b.create<arith::ConstantIndexOp>(loc, value); }
  Value integer(int64_t value, unsigned bits = 32) {
    return b.create<arith::ConstantIntOp>(loc, value, bits);
  }
  Value fp(double value) { return b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(value)); }
  Value read(Value memory, Value record, Value offset) {
    return b.create<memref::LoadOp>(loc, memory, ValueRange{record, offset});
  }
  void write(Value value, Value memory, Value record, Value offset) {
    b.create<memref::StoreOp>(loc, value, memory, ValueRange{record, offset});
  }
  Value littleEndian(Value memory, Value record, int64_t offset, unsigned bytes) {
    Type type = b.getIntegerType(8 * bytes);
    Value value = b.create<arith::ExtUIOp>(loc, type, read(memory, record, index(offset)));
    for (unsigned byte = 1; byte < bytes; ++byte) {
      Value next = b.create<arith::ExtUIOp>(loc, type, read(memory, record, index(offset + byte)));
      next = b.create<arith::ShLIOp>(loc, next, integer(8 * byte, 8 * bytes));
      value = b.create<arith::OrIOp>(loc, value, next);
    }
    return value;
  }
  void bytes(Value value, Value memory, Value record, int64_t offset, unsigned count) {
    for (unsigned byte = 0; byte < count; ++byte) {
      Value part = value;
      if (byte) part = b.create<arith::ShRUIOp>(loc, part,
          integer(8 * byte, cast<IntegerType>(value.getType()).getWidth()));
      part = b.create<arith::TruncIOp>(loc, b.getI8Type(), part);
      write(part, memory, record, index(offset + byte));
    }
  }
  Value field(Value memory, Value record, int field, int member) {
    Value high = b.create<arith::ExtUIOp>(loc, b.getI32Type(),
        read(memory, record, index(4 + field * 4 + member % 4)));
    if (member < 4) return b.create<arith::AndIOp>(loc, high, integer(63));
    high = b.create<arith::ShRUIOp>(loc, high, integer(6));
    high = b.create<arith::ShLIOp>(loc, high, integer(4));
    Value low = b.create<arith::ExtUIOp>(loc, b.getI32Type(),
        read(memory, record, index(12 + member - 4)));
    if (field) low = b.create<arith::ShRUIOp>(loc, low, integer(4));
    low = b.create<arith::AndIOp>(loc, low, integer(15));
    return b.create<arith::OrIOp>(loc, high, low);
  }
};
} // namespace

LogicalResult Construction::quantize(QuantizeOp operation) {
  Location loc = operation.getLoc();
  auto shape = localShape(operation.getResult(), loc);
  if (failed(shape) || shape->size() != 2 || !completeShape(*shape))
    return operation.emitError("quantization requires the complete selected record domain");
  Value input = get(operation.getInput());
  if (!input) return operation.emitError("quantization input is unavailable");
  Value output = allocateTensor(loc, b.getI8Type(), *shape);
  RecordValues p{b, loc};
  Value row = allocate(loc, b.getF32Type(), 1, 256);
  Value high = allocate(loc, b.getF32Type(), 1, 1);
  Value low = allocate(loc, b.getF32Type(), 1, 1);
  Value scratch = allocate(loc, b.getF32Type(), 1, 256);
  Value sum = allocate(loc, b.getI32Type(), 1, 1);
  auto status = loop(loc, index(loc, 0), (*shape)[0].count, index(loc, 1), [&](Value record) {
    b.create<dsa::LoadTileOp>(loc, input, row, mul(loc, record, index(loc, 256)),
        index(loc, 256), index(loc, 1), index(loc, 1), index(loc, 256));
    for (auto [destination, kind, identity] : {
        std::make_tuple(high, BinaryOperator::Maximum, -std::numeric_limits<float>::infinity()),
        std::make_tuple(low, BinaryOperator::Minimum, std::numeric_limits<float>::infinity())})
      b.create<dsa::ReduceOp>(loc, row, destination, scratch, index(loc, 256), p.fp(identity),
          BinaryOperatorAttr::get(b.getContext(), kind), b.getI64IntegerAttr(1));
    Value maximum = loadLocal(loc, high, {}), minimum = loadLocal(loc, low, {});
    Value greater = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT,
        b.create<math::AbsFOp>(loc, maximum), b.create<math::AbsFOp>(loc, minimum));
    Value extreme = b.create<arith::SelectOp>(loc, greater, maximum, minimum);
    Value nonzero = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ONE, extreme, p.fp(0));
    Value denominator = b.create<arith::SelectOp>(loc, nonzero, extreme, p.fp(1));
    Value inverse = b.create<arith::DivFOp>(loc, p.fp(-127), denominator);
    Value scale = b.create<arith::DivFOp>(loc, p.fp(1), inverse);
    inverse = b.create<arith::SelectOp>(loc, nonzero, inverse, p.fp(0));
    scale = b.create<arith::SelectOp>(loc, nonzero, scale, p.fp(0));
    p.bytes(b.create<arith::BitcastOp>(loc, b.getI32Type(), scale), output, record, 0, 4);
    for (int group = 0; group < 16; ++group) {
      storeLocal(loc, p.integer(0), sum, {});
      if (failed(loop(loc, index(loc, 16 * group), index(loc, 16 * (group + 1)), index(loc, 1),
          [&](Value element) {
        Value scaled = b.create<arith::MulFOp>(loc, p.read(row, index(loc, 0), element), inverse);
        scaled = b.create<arith::MaximumFOp>(loc, scaled, p.fp(-128));
        scaled = b.create<arith::MinimumFOp>(loc, scaled, p.fp(127));
        Value floor = b.create<math::FloorOp>(loc, scaled);
        Value integer = b.create<arith::FPToSIOp>(loc, b.getI32Type(), floor);
        Value fraction = b.create<arith::SubFOp>(loc, scaled, floor);
        Value above = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, fraction, p.fp(0.5));
        Value tie = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, fraction, p.fp(0.5));
        Value odd = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
            b.create<arith::AndIOp>(loc, integer, p.integer(1)), p.integer(0));
        Value round = b.create<arith::OrIOp>(loc, above, b.create<arith::AndIOp>(loc, tie, odd));
        integer = b.create<arith::AddIOp>(loc, integer,
            b.create<arith::ExtUIOp>(loc, b.getI32Type(), round));
        p.write(b.create<arith::TruncIOp>(loc, b.getI8Type(), integer), output,
                record, add(loc, element, index(loc, 4)));
        storeLocal(loc, b.create<arith::AddIOp>(loc, loadLocal(loc, sum, {}), integer), sum, {});
        return success();
      }))) return failure();
      Value total = b.create<arith::TruncIOp>(loc, b.getI16Type(), loadLocal(loc, sum, {}));
      p.bytes(total, output, record, 260 + group * 2, 2);
    }
    return success();
  });
  if (failed(status)) return failure();
  values.map(operation.getResult(), output);
  return success();
}

LogicalResult Construction::quantizedDot(QuantizedDotOp operation) {
  Location loc = operation.getLoc();
  Value lhs = get(operation.getLhs()), rhs = get(operation.getRhs());
  if (!lhs || !rhs || !localShapes.count(lhs) || !localShapes.count(rhs))
    return operation.emitError("quantized dot records are unavailable");
  const auto &left = localShapes.find(lhs)->second;
  const auto &right = localShapes.find(rhs)->second;
  if (!completeShape(left) || !completeShape(right) ||
      !sameIndex(left[0].count, right[0].count))
    return operation.emitError("quantized dot requires matching complete record domains");
  Value output = allocateTensor(loc, b.getF32Type(), {});
  Value sums = allocate(loc, b.getI32Type(), 1, 2);
  RecordValues p{b, loc};
  auto status = loop(loc, index(loc, 0), left[0].count, index(loc, 1), [&](Value record) {
    p.write(p.integer(0), sums, index(loc, 0), index(loc, 0));
    p.write(p.integer(0), sums, index(loc, 0), index(loc, 1));
    for (int group = 0; group < 8; ++group) {
      Value scale = p.field(lhs, record, 0, group), minimum = p.field(lhs, record, 1, group);
      if (failed(loop(loc, index(loc, 0), index(loc, 32), index(loc, 1), [&](Value lane) {
        Value offset = add(loc, index(loc, 16 + 32 * (group / 2)), lane);
        Value weight = b.create<arith::ExtUIOp>(loc, b.getI32Type(), p.read(lhs, record, offset));
        if (group % 2) weight = b.create<arith::ShRUIOp>(loc, weight, p.integer(4));
        weight = b.create<arith::AndIOp>(loc, weight, p.integer(15));
        Value value = b.create<arith::ExtSIOp>(loc, b.getI32Type(),
            p.read(rhs, record, add(loc, index(loc, 4 + group * 32), lane)));
        Value product = b.create<arith::MulIOp>(loc, b.create<arith::MulIOp>(loc, weight, value), scale);
        Value previous = p.read(sums, index(loc, 0), index(loc, 0));
        p.write(b.create<arith::AddIOp>(loc, previous, product), sums, index(loc, 0), index(loc, 0));
        return success();
      }))) return failure();
      Value first = b.create<arith::ExtSIOp>(loc, b.getI32Type(), p.littleEndian(rhs, record, 260 + group * 4, 2));
      Value second = b.create<arith::ExtSIOp>(loc, b.getI32Type(), p.littleEndian(rhs, record, 262 + group * 4, 2));
      Value correction = b.create<arith::MulIOp>(loc, minimum, b.create<arith::AddIOp>(loc, first, second));
      Value previous = p.read(sums, index(loc, 0), index(loc, 1));
      p.write(b.create<arith::AddIOp>(loc, previous, correction), sums, index(loc, 0), index(loc, 1));
    }
    Value d = b.create<arith::BitcastOp>(loc, b.getF16Type(), p.littleEndian(lhs, record, 0, 2));
    Value minimum = b.create<arith::BitcastOp>(loc, b.getF16Type(), p.littleEndian(lhs, record, 2, 2));
    d = b.create<arith::ExtFOp>(loc, b.getF32Type(), d);
    minimum = b.create<arith::ExtFOp>(loc, b.getF32Type(), minimum);
    Value s = b.create<arith::SIToFPOp>(loc, b.getF32Type(), p.read(sums, index(loc, 0), index(loc, 0)));
    Value t = b.create<arith::SIToFPOp>(loc, b.getF32Type(), p.read(sums, index(loc, 0), index(loc, 1)));
    Value positive = b.create<arith::MulFOp>(loc, d, s);
    Value negative = b.create<arith::MulFOp>(loc, minimum, t);
    Value difference = b.create<arith::SubFOp>(loc, positive, negative);
    Value ds = b.create<arith::BitcastOp>(loc, b.getF32Type(), p.littleEndian(rhs, record, 0, 4));
    Value value = b.create<arith::MulFOp>(loc, ds, difference);
    storeLocal(loc, b.create<arith::AddFOp>(loc, loadLocal(loc, output, {}), value), output, {});
    return success();
  });
  if (failed(status)) return failure();
  values.map(operation.getResult(), output);
  return success();
}
} // namespace intent::kir_to_dsa
