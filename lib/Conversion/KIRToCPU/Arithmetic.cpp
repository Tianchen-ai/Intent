#include "Construction.h"
#include "Intent/Conversion/ScalarLowering.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

namespace intent::kir_to_cpu {

FailureOr<Value> Construction::arithmetic(Operation *operation, ValueRange arguments, OpBuilder &builder) {
  Location loc = operation->getLoc();
  auto flushF32 = [&](Value value) -> Value {
    Value zero = builder.create<arith::ConstantOp>(loc, builder.getF32FloatAttr(0.0f));
    Value normal = builder.create<arith::ConstantOp>(loc, builder.getF32FloatAttr(0x1.0p-126f));
    Value magnitude = builder.create<math::AbsFOp>(loc, value);
    Value subnormal = builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, magnitude, normal);
    Value signedZero = builder.create<arith::MulFOp>(loc, value, zero);
    return builder.create<arith::SelectOp>(loc, subnormal, signedZero, value);
  };
  if (isa<RandomBitsOp>(operation)) {
    Type u32 = IntegerType::get(builder.getContext(), 32, IntegerType::Unsigned);
    Type u64 = IntegerType::get(builder.getContext(), 64, IntegerType::Unsigned);
    auto literal = [&](Type type, uint64_t value) -> Value {
      return builder.create<arith::ConstantOp>(loc, builder.getIntegerAttr(type, value));
    };
    auto low = [&](Value value) -> Value { return builder.create<arith::TruncIOp>(loc, u32, value); };
    auto high = [&](Value value) -> Value {
      return low(builder.create<arith::ShRUIOp>(loc, value, literal(u64, 32)));
    };
    Value counter = arguments[1];
    if (counter.getType().isIndex()) counter = builder.create<arith::IndexCastUIOp>(loc, u64, counter);
    Value block = builder.create<arith::ShRUIOp>(loc, counter, literal(u64, 2));
    Value word = builder.create<arith::AndIOp>(loc, counter, literal(u64, 3));
    Value key0 = low(arguments[0]), key1 = high(arguments[0]);
    SmallVector<Value> words{low(block), high(block), literal(u32, 0), literal(u32, 0)};
    auto product = [&](Value value, uint64_t multiplier) {
      Value wide = builder.create<arith::ExtUIOp>(loc, u64, value);
      Value result = builder.create<arith::MulIOp>(loc, wide, literal(u64, multiplier));
      return std::make_pair(high(result), low(result));
    };
    auto mix = [&](Value a, Value b, Value key) -> Value {
      return builder.create<arith::XOrIOp>(loc, builder.create<arith::XOrIOp>(loc, a, b), key);
    };
    for (unsigned round = 0; round < 10; ++round) {
      auto [hi0, lo0] = product(words[0], 0xD2511F53);
      auto [hi1, lo1] = product(words[2], 0xCD9E8D57);
      words = {mix(hi1, words[1], key0), lo1, mix(hi0, words[3], key1), lo0};
      key0 = builder.create<arith::AddIOp>(loc, key0, literal(u32, 0x9E3779B9));
      key1 = builder.create<arith::AddIOp>(loc, key1, literal(u32, 0xBB67AE85));
    }
    Value result = words[3];
    for (int i = 2; i >= 0; --i) {
      Value selected = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, word, literal(u64, i));
      result = builder.create<arith::SelectOp>(loc, selected, words[i], result);
    }
    return result;
  } else if (auto binary = dyn_cast<BinaryOp>(operation)) {
    Value a = arguments[0], b = arguments[1];
    if (binary.getApproximate() || binary.getFlushToZero()) {
      if (binary.getOperatorKind() != BinaryOperator::TrueDivide || !binary.getApproximate() ||
          !a.getType().isF32() || !b.getType().isF32())
        return binary.emitError("CPU non-default arithmetic requires the closed f32 approximate division contract"), failure();
      auto literal = [&](float value) -> Value {
        return builder.create<arith::ConstantOp>(loc, builder.getF32FloatAttr(value));
      };
      Value zero = literal(0.0f), negativeZero = literal(-0.0f);
      if (binary.getFlushToZero()) { a = flushF32(a); b = flushF32(b); }
      Value quotient = builder.create<arith::DivFOp>(loc, a, b);
      Value large = builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT,
          builder.create<math::AbsFOp>(loc, b), literal(0x1.0p126f));
      Value negative = builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, b, zero);
      Value denominatorSign = builder.create<arith::SelectOp>(loc, negative, negativeZero, zero);
      Value limit = builder.create<arith::MulFOp>(loc, a, denominatorSign);
      Value result = builder.create<arith::SelectOp>(loc, large, limit, quotient);
      return binary.getFlushToZero() ? flushF32(result) : result;
    }
  } else if (auto unary = dyn_cast<UnaryOp>(operation)) {
    if (unary.getApproximate() || unary.getFlushToZero()) {
      auto kind = unary.getOperatorKind();
      if (!unary.getApproximate() || !arguments[0].getType().isF32() ||
          (kind != UnaryOperator::Exp2 && kind != UnaryOperator::Tanh) ||
          (unary.getFlushToZero() && kind != UnaryOperator::Exp2))
        return unary.emitError("CPU non-default unary arithmetic requires the closed f32 exp2/tanh contract"), failure();
      Value input = unary.getFlushToZero() ? flushF32(arguments[0]) : arguments[0];
      Value result = kind == UnaryOperator::Exp2
          ? Value(builder.create<math::Exp2Op>(loc, input, arith::FastMathFlags::afn))
          : Value(builder.create<math::TanhOp>(loc, input, arith::FastMathFlags::afn));
      return unary.getFlushToZero() ? flushF32(result) : result;
    }
    if (unary.getOperatorKind() == UnaryOperator::Sigmoid) {
      Value one = builder.create<arith::ConstantOp>(loc, builder.getFloatAttr(arguments[0].getType(), 1.0));
      Value negative = builder.create<arith::NegFOp>(loc, arguments[0]);
      Value denominator = builder.create<arith::AddFOp>(loc, one, builder.create<math::ExpOp>(loc, negative));
      return Value(builder.create<arith::DivFOp>(loc, one, denominator));
    }
  } else if (auto cast = dyn_cast<CastOp>(operation)) {
    if (getElementTypeOrSelf(cast.getType()).isInteger(1) &&
        !getElementTypeOrSelf(cast.getInput().getType()).isInteger(1))
      return cast.emitError("CPU numeric-to-bool cast is not implemented"), failure();
  }
  return intent::lowerScalarOperation(operation, arguments, builder);
}

LogicalResult Construction::pointwise(Operation *operation) {
  Type resultType = operation->getResult(0).getType();
  auto tensor = dyn_cast<RankedTensorType>(resultType);
  SmallVector<Value> arguments;
  for (Value input : operation->getOperands())
    arguments.push_back(values.lookup(input));
  if (!tensor) {
    auto result = arithmetic(operation, arguments, builder);
    if (failed(result)) return failure();
    values.map(operation->getResult(0), *result);
    return success();
  }
  auto sizes = extents(operation->getResult(0), operation->getLoc());
  if (failed(sizes)) return failure();
  Value output = emptyTensor(tensor, *sizes, operation->getLoc());
  LogicalResult status = success();
  auto result = builder.create<linalg::GenericOp>(operation->getLoc(), TypeRange{output.getType()}, arguments,
      ValueRange{output}, pointwiseMaps(arguments, tensor.getRank()),
      SmallVector<utils::IteratorType>(tensor.getRank(), utils::IteratorType::parallel),
      [&](OpBuilder &nested, Location loc, ValueRange scalars) {
        auto result = arithmetic(operation, scalars.take_front(arguments.size()), nested);
        if (failed(result)) { status = failure(); return; }
        nested.create<linalg::YieldOp>(loc, *result);
      });
  values.map(operation->getResult(0), result.getResult(0));
  return status;
}

LogicalResult Construction::lower(ConstantOp op) {
  Location loc = op.getLoc();
  Type type = op.getResult().getType();
  TypedAttr attribute;
  if (auto integer = dyn_cast<IntegerAttr>(op.getValue())) {
    attribute = builder.getIntegerAttr(type, integer.getValue());
  } else if (auto floating = dyn_cast<FloatAttr>(op.getValue())) {
    llvm::APFloat value = floating.getValue();
    bool losesInformation;
    value.convert(cast<FloatType>(type).getFloatSemantics(),
                  llvm::APFloat::rmNearestTiesToEven, &losesInformation);
    attribute = FloatAttr::get(type, value);
  } else return op.emitError("CPU constant payload is not a scalar numeric literal");
  values.map(op.getResult(), builder.create<arith::ConstantOp>(loc, type, attribute));
  return success();
}

} // namespace intent::kir_to_cpu
