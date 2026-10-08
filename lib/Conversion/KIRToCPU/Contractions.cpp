#include "Construction.h"
#include "Intent/Analysis/ContractionAxes.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"

using namespace mlir;

namespace intent::kir_to_cpu {

Value Construction::emitContraction(Value lhs, Value rhs, Value destination,
                     ArrayRef<AffineMap> maps, unsigned parallelRank,
                     unsigned reductionRank, Location loc) {
  Type accumulator = cast<RankedTensorType>(destination.getType()).getElementType();
  Value zero = builder.create<arith::ConstantOp>(loc, builder.getZeroAttr(accumulator));
  destination = builder.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{destination}).getResult(0);
  SmallVector<utils::IteratorType> iterators(parallelRank, utils::IteratorType::parallel);
  iterators.append(reductionRank, utils::IteratorType::reduction);
  auto contraction = builder.create<linalg::GenericOp>(loc, TypeRange{destination.getType()}, ValueRange{lhs, rhs}, ValueRange{destination},
      maps, iterators, [](OpBuilder &b, Location loc, ValueRange arguments) {
        Value value;
        if (isa<FloatType>(arguments[2].getType())) {
          Value lhs = arguments[0], rhs = arguments[1];
          if (lhs.getType() != arguments[2].getType())
            lhs = b.create<arith::ExtFOp>(loc, arguments[2].getType(), lhs);
          if (rhs.getType() != arguments[2].getType())
            rhs = b.create<arith::ExtFOp>(loc, arguments[2].getType(), rhs);
          value = b.create<math::FmaOp>(loc, lhs, rhs, arguments[2]);
        } else {
          Value lhs = b.create<arith::ExtSIOp>(loc, arguments[2].getType(), arguments[0]);
          Value rhs = b.create<arith::ExtSIOp>(loc, arguments[2].getType(), arguments[1]);
          value = b.create<arith::AddIOp>(loc,
              b.create<arith::MulIOp>(loc, lhs, rhs), arguments[2]);
        }
        b.create<linalg::YieldOp>(loc, value);
      });
  contraction->setAttr("intent_cpu.reduction_order",
      cpu::ReductionOrderAttr::get(builder.getContext(), true, true));
  return contraction.getResult(0);
}

Value Construction::matrix(Value lhs, Value rhs, Value destination, Location loc) {
  AffineExpr m, n, k;
  bindDims(builder.getContext(), m, n, k);
  SmallVector<AffineMap> maps = {
      AffineMap::get(3, 0, {m, k}, builder.getContext()),
      AffineMap::get(3, 0, {k, n}, builder.getContext()),
      AffineMap::get(3, 0, {m, n}, builder.getContext())};
  return emitContraction(lhs, rhs, destination, maps, 2, 1, loc);
}

LogicalResult Construction::contract(ContractOp operation) {
  auto lhsType = cast<RankedTensorType>(operation.getLhs().getType());
  auto rhsType = cast<RankedTensorType>(operation.getRhs().getType());
  auto resultType = cast<RankedTensorType>(operation.getResult().getType());
  SmallVector<int64_t> lhsReduction, rhsReduction, lhsBatch, rhsBatch;
  auto appendPairs = [](ArrayAttr pairs, SmallVectorImpl<int64_t> &left,
                        SmallVectorImpl<int64_t> &right) {
    for (Attribute attribute : pairs) {
      auto pair = cast<ArrayAttr>(attribute);
      left.push_back(cast<IntegerAttr>(pair[0]).getInt());
      right.push_back(cast<IntegerAttr>(pair[1]).getInt());
    }
  };
  appendPairs(operation.getReduce(), lhsReduction, rhsReduction);
  appendPairs(operation.getBatch(), lhsBatch, rhsBatch);
  std::string reason;
  auto axes = ContractionAxes::get(lhsType.getRank(), rhsType.getRank(),
      lhsReduction, rhsReduction, lhsBatch, rhsBatch, &reason);
  if (!axes)
    return operation.emitError("CPU contraction axis schema: ") << reason;
  if (axes->results.size() != static_cast<size_t>(resultType.getRank()))
    return operation.emitError("CPU contraction result rank disagrees with its axis schema");
  Type inputElement = lhsType.getElementType(), accumulator = resultType.getElementType();
  bool floating = isa<FloatType>(inputElement) && inputElement == rhsType.getElementType() &&
      (accumulator.isF32() || accumulator.isF64()) &&
      inputElement.getIntOrFloatBitWidth() <= accumulator.getIntOrFloatBitWidth();
  bool integer = inputElement.isSignlessInteger(8) &&
      rhsType.getElementType().isSignlessInteger(8) && accumulator.isSignlessInteger(32);
  if (!floating && !integer)
    return operation.emitError(
        "CPU contraction requires lossless floating widening to f32/f64 or i8 to i32 accumulation");

  unsigned parallelRank = resultType.getRank();
  unsigned reductionRank = axes->reduction.size();
  unsigned loopRank = parallelRank + reductionRank;
  SmallVector<AffineExpr> lhsMap(lhsType.getRank()), rhsMap(rhsType.getRank()), resultMap;
  for (auto [axis, result] : llvm::enumerate(axes->lhsResultAxes))
    if (result) lhsMap[axis] = builder.getAffineDimExpr(*result);
  for (auto [axis, result] : llvm::enumerate(axes->rhsResultAxes))
    if (result) rhsMap[axis] = builder.getAffineDimExpr(*result);
  for (auto [number, pair] : llvm::enumerate(axes->reduction)) {
    AffineExpr reduction = builder.getAffineDimExpr(parallelRank + number);
    lhsMap[pair.lhs] = reduction;
    rhsMap[pair.rhs] = reduction;
  }
  for (unsigned axis = 0; axis < parallelRank; ++axis)
    resultMap.push_back(builder.getAffineDimExpr(axis));
  SmallVector<AffineMap> maps = {
      AffineMap::get(loopRank, 0, lhsMap, builder.getContext()),
      AffineMap::get(loopRank, 0, rhsMap, builder.getContext()),
      AffineMap::get(loopRank, 0, resultMap, builder.getContext())};
  Location loc = operation.getLoc();
  auto sizes = extents(operation.getResult(), loc);
  if (failed(sizes)) return failure();
  Value output = emptyTensor(resultType, *sizes, loc);
  Value result = emitContraction(values.lookup(operation.getLhs()), values.lookup(operation.getRhs()),
                  output, maps, parallelRank, reductionRank, loc);
  values.map(operation.getResult(), result);
  return success();
}

LogicalResult Construction::sparseContract(SparseContractOp operation) {
  auto lhsType = cast<RankedTensorType>(operation.getCompressed().getType());
  auto rhsType = cast<RankedTensorType>(operation.getRhs().getType());
  auto resultType = cast<RankedTensorType>(operation.getResult().getType());
  Type element = lhsType.getElementType(), accumulator = resultType.getElementType();
  auto pairs = operation.getReduce();
  if (lhsType.getRank() != 2 || rhsType.getRank() != 2 || !operation.getBatch().empty() ||
      operation.getFormat().getCompressionAxis() != 1 || pairs.size() != 1 ||
      cast<IntegerAttr>(cast<ArrayAttr>(pairs[0])[0]).getInt() != 1 ||
      cast<IntegerAttr>(cast<ArrayAttr>(pairs[0])[1]).getInt() != 0 ||
      !isa<FloatType>(element) || element != rhsType.getElementType() ||
      (!accumulator.isF32() && !accumulator.isF64()) ||
      element.getIntOrFloatBitWidth() > accumulator.getIntOrFloatBitWidth())
    return operation.emitError("CPU sparse contraction requires rank-two compressed matrices and lossless floating widening");
  Location loc = operation.getLoc();
  auto sizes = extents(operation.getResult(), loc);
  if (failed(sizes)) return failure();
  Value output = emptyTensor(resultType, *sizes, loc);
  Value compressed = values.lookup(operation.getCompressed()), rhs = values.lookup(operation.getRhs());
  SmallVector<Value> positions = flattened(ValueRange{operation.getMetadata()});
  int64_t nonzeros = operation.getFormat().getKind() == 0 ? 1 : 2;
  int64_t groupSize = nonzeros * 2;
  Value zero = builder.create<arith::ConstantOp>(loc, builder.getZeroAttr(accumulator));
  output = builder.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{output}).getResult(0);
  AffineExpr m, compressedK, n;
  bindDims(builder.getContext(), m, compressedK, n);
  SmallVector<AffineMap> maps{
      AffineMap::get(3, 0, {m, compressedK}, builder.getContext()),
      AffineMap::get(3, 0, {m, n}, builder.getContext())};
  auto result = builder.create<linalg::GenericOp>(loc, TypeRange{output.getType()}, ValueRange{compressed}, ValueRange{output}, maps,
      SmallVector<utils::IteratorType>{utils::IteratorType::parallel, utils::IteratorType::reduction,
                                      utils::IteratorType::parallel},
      [&](OpBuilder &b, Location loc, ValueRange arguments) {
        auto index = [&](int64_t value) -> Value { return b.create<arith::ConstantIndexOp>(loc, value); };
        Value row = b.create<linalg::IndexOp>(loc, 0), column = b.create<linalg::IndexOp>(loc, 2);
        Value ordinal = b.create<linalg::IndexOp>(loc, 1);
        Value group = b.create<arith::DivSIOp>(loc, ordinal, index(nonzeros));
        auto position = [&](Value memory) -> Value {
          Value value = extractElement(b, loc, memory, ValueRange{row, group});
          if (value.getType().isIndex()) return value;
          return cast<IntegerType>(value.getType()).isUnsigned()
              ? Value(b.create<arith::IndexCastUIOp>(loc, b.getIndexType(), value))
              : Value(b.create<arith::IndexCastOp>(loc, b.getIndexType(), value));
        };
        Value relative = position(positions[0]);
        if (nonzeros == 2) {
          Value first = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
              b.create<arith::RemSIOp>(loc, ordinal, index(nonzeros)), index(0));
          relative = b.create<arith::SelectOp>(loc, first, relative, position(positions[1]));
        }
        Value reduction = b.create<arith::AddIOp>(loc,
            b.create<arith::MulIOp>(loc, group, index(groupSize)), relative);
        Value left = arguments[0], right = extractElement(b, loc, rhs, ValueRange{reduction, column});
        if (element != accumulator) {
          left = b.create<arith::ExtFOp>(loc, accumulator, left);
          right = b.create<arith::ExtFOp>(loc, accumulator, right);
        }
        b.create<linalg::YieldOp>(loc, ValueRange{b.create<math::FmaOp>(loc, left, right, arguments[1])});
      });
  values.map(operation.getResult(), result.getResult(0));
  return success();
}

LogicalResult Construction::scaledContract(ScaledContractOp operation) {
  auto outputType = cast<RankedTensorType>(operation.getResult().getType());
  auto carrier = [](Value value, ScaledFormat format) {
    Type element = cast<RankedTensorType>(value.getType()).getElementType();
    return format == ScaledFormat::E4M3 ? isa<Float8E4M3FNType>(element)
         : format == ScaledFormat::E2M1 && element.isInteger(8);
  };
  auto scale = [](Value value) {
    Type element = cast<RankedTensorType>(value.getType()).getElementType();
    return element.isInteger(8) || element.isF32();
  };
  if (!outputType.getElementType().isF32() ||
      !carrier(operation.getLhs(), operation.getLhsFormat()) ||
      !carrier(operation.getRhs(), operation.getRhsFormat()) ||
      !scale(operation.getLhsScale()) || !scale(operation.getRhsScale()))
    return operation.emitError("CPU scaled contraction requires E4M3 or E2M1 carriers, byte E8M0 or f32 scales and f32 accumulation");
  Location loc = operation.getLoc();
  int64_t groupSize = operation.getLhsGroupSize();
  Value lhs = values.lookup(operation.getLhs()), rhs = values.lookup(operation.getRhs());
  Value groups = dimension(builder, loc, lhs, 1);
  Value depth = builder.create<arith::MulIOp>(loc, groups, constant(loc, groupSize));
  auto sizes = extents(operation.getResult(), loc);
  if (failed(sizes)) return failure();
  auto decode = [&](Value source, Value scales, ScaledFormat format, bool left) {
    SmallVector<Value> shape = left ? SmallVector<Value>{(*sizes)[0], depth}
                                   : SmallVector<Value>{depth, (*sizes)[1]};
    auto type = RankedTensorType::get({ShapedType::kDynamic, ShapedType::kDynamic}, builder.getF32Type());
    Value decoded = emptyTensor(type, shape, loc);
    auto result = builder.create<linalg::GenericOp>(loc, TypeRange{decoded.getType()}, ValueRange{}, ValueRange{decoded},
        SmallVector<AffineMap>{builder.getMultiDimIdentityMap(2)},
        SmallVector<utils::IteratorType>(2, utils::IteratorType::parallel),
        [&](OpBuilder &b, Location loc, ValueRange) {
          auto integer = [&](int64_t value) -> Value { return b.create<arith::ConstantIntOp>(loc, value, 32); };
          auto index = [&](int64_t value) -> Value { return b.create<arith::ConstantIndexOp>(loc, value); };
          Value row = b.create<linalg::IndexOp>(loc, 0), column = b.create<linalg::IndexOp>(loc, 1);
          Value free = left ? row : column, reduction = left ? column : row;
          Value group = b.create<arith::DivSIOp>(loc, reduction, index(groupSize));
          Value inner = b.create<arith::RemSIOp>(loc, reduction, index(groupSize));
          Value position = format == ScaledFormat::E2M1
              ? Value(b.create<arith::DivSIOp>(loc, inner, index(2))) : inner;
          SmallVector<Value> coordinates = left ? SmallVector<Value>{free, group, position}
                                                : SmallVector<Value>{group, position, free};
          Value raw = extractElement(b, loc, source, coordinates);
          Value number;
          if (format == ScaledFormat::E4M3) number = b.create<arith::ExtFOp>(loc, b.getF32Type(), raw);
          else {
            Value bits = b.create<arith::ExtUIOp>(loc, b.getI32Type(), raw);
            Value lane = b.create<arith::IndexCastOp>(loc, b.getI32Type(), b.create<arith::RemSIOp>(loc, inner, index(2)));
            Value shift = b.create<arith::MulIOp>(loc, lane, integer(4));
            Value nibble = b.create<arith::ShRUIOp>(loc, bits, shift);
            Value magnitude = b.create<arith::AndIOp>(loc, nibble, integer(7));
            static constexpr float table[] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
            number = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(table[0]));
            for (int64_t code = 1; code < 8; ++code) {
              Value matches = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, magnitude, integer(code));
              number = b.create<arith::SelectOp>(loc, matches,
                  b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(table[code])), number);
            }
            Value sign = b.create<arith::AndIOp>(loc, nibble, integer(8));
            Value negative = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, sign, integer(0));
            number = b.create<arith::SelectOp>(loc, negative, b.create<arith::NegFOp>(loc, number), number);
          }
          Value rawScale = extractElement(b, loc, scales, ValueRange{free, group});
          Value scale = rawScale;
          if (!scale.getType().isF32()) {
            Value exponent = b.create<arith::ExtUIOp>(loc, b.getI32Type(), rawScale);
            Value bits = b.create<arith::ShLIOp>(loc, exponent, integer(23));
            Value subnormal = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, exponent, integer(0));
            bits = b.create<arith::SelectOp>(loc, subnormal, integer(0x00400000), bits);
            Value nan = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, exponent, integer(255));
            bits = b.create<arith::SelectOp>(loc, nan, integer(0x7fc00000), bits);
            scale = b.create<arith::BitcastOp>(loc, b.getF32Type(), bits);
          }
          b.create<linalg::YieldOp>(loc, ValueRange{b.create<arith::MulFOp>(loc, number, scale)});
        });
    return result.getResult(0);
  };
  Value left = decode(lhs, values.lookup(operation.getLhsScale()), operation.getLhsFormat(), true);
  Value right = decode(rhs, values.lookup(operation.getRhsScale()), operation.getRhsFormat(), false);
  Value output = emptyTensor(outputType, *sizes, loc);
  values.map(operation.getResult(), matrix(left, right, output, loc));
  return success();
}

LogicalResult Construction::lower(QuantizeOp op) {
  Location loc = op.getLoc();
  auto tensor = cast<RankedTensorType>(op.getResult().getType());
  auto sizes = extents(op.getResult(), loc);
  if (failed(sizes)) return failure();
  Value output = emptyTensor(tensor, *sizes, loc);
  auto result = builder.create<cpu::QuantizeOp>(loc, tensorType(tensor),
      values.lookup(op.getInput()), output, op.getFormatAttr());
  values.map(op.getResult(), result.getResult());
  return success();
}

LogicalResult Construction::lower(QuantizedDotOp op) {
  Location loc = op.getLoc();
  Value output = emptyTensor(cast<RankedTensorType>(op.getResult().getType()), {}, loc);
  auto result = builder.create<cpu::QuantizedDotOp>(loc, output.getType(), values.lookup(op.getLhs()), values.lookup(op.getRhs()),
      output, op.getLhsFormatAttr(), op.getRhsFormatAttr());
  values.map(op.getResult(), result.getResult());
  return success();
}

} // namespace intent::kir_to_cpu
