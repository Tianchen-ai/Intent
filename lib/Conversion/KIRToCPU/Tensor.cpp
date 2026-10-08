#include "Construction.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;

namespace intent::kir_to_cpu {

LogicalResult Construction::lower(FullOp op) {
  Location loc = op.getLoc();
  auto tensor = cast<RankedTensorType>(op.getResult().getType());
  auto sizes = extents(op.getResult(), loc);
  if (failed(sizes)) return failure();
  Value output = emptyTensor(tensor, *sizes, loc);
  auto fill = builder.create<linalg::FillOp>(loc, ValueRange{values.lookup(op.getInputs()[0])}, ValueRange{output});
  values.map(op.getResult(), fill.getResult(0));
  return success();
}

LogicalResult Construction::lower(IndicesOp op) {
  Location loc = op.getLoc();
  auto type = cast<RankedTensorType>(op.getResult().getType());
  auto sizes = extents(op.getResult(), loc);
  if (failed(sizes)) return failure();
  Value begin = constant(loc, 0), step = constant(loc, 1);
  int64_t axis = 0;
  if (op.getTensorAxis()) {
    axis = *op.getTensorAxis();
  } else {
    if (!domains.count(op.getSource()))
      return op.emitError("CPU indices requires a realized source domain or a tensor axis");
    Domain domain = domains.lookup(op.getSource());
    begin = domain.begin; step = domain.step;
  }
  Value output = emptyTensor(type, *sizes, loc);
  auto result = builder.create<linalg::GenericOp>(loc, TypeRange{output.getType()}, ValueRange{}, ValueRange{output},
      SmallVector<AffineMap>{builder.getMultiDimIdentityMap(type.getRank())},
      SmallVector<utils::IteratorType>(type.getRank(), utils::IteratorType::parallel),
      [&](OpBuilder &nested, Location location, ValueRange) {
        Value coordinate = nested.create<linalg::IndexOp>(location, axis);
        coordinate = nested.createOrFold<arith::AddIOp>(location, begin,
            nested.createOrFold<arith::MulIOp>(location, coordinate, step));
        if (!type.getElementType().isIndex()) coordinate = nested.create<arith::IndexCastOp>(location, type.getElementType(), coordinate);
        nested.create<linalg::YieldOp>(location, coordinate);
      });
  values.map(op.getResult(), result.getResult(0));
  return success();
}

LogicalResult Construction::lower(JoinOp op) {
  Location loc = op.getLoc();
  auto tensor = cast<RankedTensorType>(op.getResult().getType());
  auto sizes = extents(op.getResult(), loc);
  if (failed(sizes)) return failure();
  Value output = emptyTensor(tensor, *sizes, loc);
  SmallVector<AffineExpr> prefix;
  for (int64_t axis = 0; axis + 1 < tensor.getRank(); ++axis)
    prefix.push_back(builder.getAffineDimExpr(axis));
  auto inputMap = AffineMap::get(tensor.getRank(), 0, prefix, builder.getContext());
  auto result = builder.create<linalg::GenericOp>(loc, TypeRange{output.getType()},
      ValueRange{values.lookup(op.getLhs()), values.lookup(op.getRhs())}, ValueRange{output},
      SmallVector<AffineMap>{inputMap, inputMap, builder.getMultiDimIdentityMap(tensor.getRank())},
      SmallVector<utils::IteratorType>(tensor.getRank(), utils::IteratorType::parallel),
      [&](OpBuilder &nested, Location location, ValueRange inputs) {
        Value component = nested.create<linalg::IndexOp>(location, tensor.getRank() - 1);
        Value first = nested.create<arith::CmpIOp>(location, arith::CmpIPredicate::eq,
            component, nested.create<arith::ConstantIndexOp>(location, 0));
        Value selected = nested.create<arith::SelectOp>(location, first, inputs[0], inputs[1]);
        nested.create<linalg::YieldOp>(location, selected);
      });
  values.map(op.getResult(), result.getResult(0));
  return success();
}

LogicalResult Construction::lower(BroadcastOp op) {
  Location loc = op.getLoc();
  auto tensor = cast<RankedTensorType>(op.getResult().getType());
  auto sizes = extents(op.getResult(), loc);
  if (failed(sizes)) return failure();
  Value input = values.lookup(op.getInputs()[0]);
  Value output = emptyTensor(tensor, *sizes, loc);
  auto result = builder.create<linalg::GenericOp>(loc, TypeRange{output.getType()}, ValueRange{input}, ValueRange{output},
      pointwiseMaps(ValueRange{input}, tensor.getRank()),
      SmallVector<utils::IteratorType>(tensor.getRank(), utils::IteratorType::parallel),
      [](OpBuilder &nested, Location loc, ValueRange scalars) {
        nested.create<linalg::YieldOp>(loc, scalars[0]);
      });
  values.map(op.getResult(), result.getResult(0));
  return success();
}

LogicalResult Construction::tensorShape(Operation *operation) {
  Location loc = operation->getLoc();
  Value input = values.lookup(operation->getOperand(0));
  auto source = cast<RankedTensorType>(input.getType());
  auto tensor = cast<RankedTensorType>(operation->getResult(0).getType());
  auto sizes = extents(operation->getResult(0), loc);
  if (failed(sizes)) return failure();
  if (isa<ReshapeOp>(operation)) {
    Value shape = builder.create<tensor::FromElementsOp>(loc,
        RankedTensorType::get({tensor.getRank()}, builder.getIndexType()), *sizes);
    values.map(operation->getResult(0),
        builder.create<tensor::ReshapeOp>(loc, tensorType(tensor), input, shape));
    return success();
  }
  SmallVector<AffineExpr> coordinates(source.getRank());
  for (auto [axis, permuted] : llvm::enumerate(cast<TransposeOp>(operation).getPermutation()))
    coordinates[cast<IntegerAttr>(permuted).getInt()] = builder.getAffineDimExpr(axis);
  Value output = emptyTensor(tensor, *sizes, loc);
  auto result = builder.create<linalg::GenericOp>(loc, TypeRange{output.getType()}, ValueRange{input}, ValueRange{output},
      SmallVector<AffineMap>{AffineMap::get(tensor.getRank(), 0, coordinates, builder.getContext()), builder.getMultiDimIdentityMap(tensor.getRank())},
      SmallVector<utils::IteratorType>(tensor.getRank(), utils::IteratorType::parallel),
      [](OpBuilder &nested, Location location, ValueRange args) { nested.create<linalg::YieldOp>(location, args[0]); });
  values.map(operation->getResult(0), result.getResult(0));
  return success();
}

} // namespace intent::kir_to_cpu
