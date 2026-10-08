#include "../Construction.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

using namespace mlir;

namespace intent::kir_to_cpu {

FailureOr<Value> Construction::indexed(Operation *operation) {
  auto fact = analysis.indexRelation(operation);
  if (failed(fact))
    return failure();
  Value source = values.lookup(fact->source);
  if (isa<ViewLoadOp, BufferLoadOp, GatherOp>(operation) &&
      (!alwaysValid(operation) || hasTensorIndices(*fact)))
    return indexedRead(operation, *fact, source);
  auto type = cast<ShapedType>(source.getType());
  SmallVector<OpFoldResult> offsets, sizes, strides;
  SmallVector<int64_t> resultShape;
  SmallVector<AffineExpr> projection;
  bool inserted = false;
  auto callbacks = indexMaterialization(builder, operation->getLoc());
  for (const IndexTermFact &term : fact->terms) {
    if (!term.sourceAxis) {
      inserted = true;
      continue;
    }
    auto index = materializeIndexTerm(fact->source, term, callbacks);
    if (failed(index)) {
      operation->emitError("CPU construction cannot reify the canonical access term");
      return failure();
    }
    if (index->range) {
      offsets.push_back(getAsOpFoldResult(index->range->begin));
      sizes.push_back(getAsOpFoldResult(index->range->count));
      strides.push_back(getAsOpFoldResult(index->range->step));
      auto size = getConstantIntValue(sizes.back());
      resultShape.push_back(size.value_or(ShapedType::kDynamic));
      projection.push_back(builder.getAffineDimExpr(term.resultAxes.front()));
    } else if (index->coordinate) {
      offsets.push_back(getAsOpFoldResult(index->coordinate));
      sizes.push_back(builder.getIndexAttr(1));
      strides.push_back(builder.getIndexAttr(1));
    } else {
      operation->emitError("CPU rectangular access requires scalar or range coordinates");
      return failure();
    }
  }
  if (offsets.size() != static_cast<size_t>(type.getRank())) {
    operation->emitError("CPU indexed access must resolve every source axis");
    return failure();
  }
  if (resultShape.empty() && !inserted) {
    SmallVector<Value> indices;
    for (OpFoldResult offset : offsets)
      indices.push_back(isa<Value>(offset) ? cast<Value>(offset)
                         : constant(operation->getLoc(), cast<IntegerAttr>(cast<Attribute>(offset)).getInt()));
    if (isa<ViewLoadOp, BufferLoadOp, GatherOp>(operation) &&
        !isa<RankedTensorType>(operation->getResult(0).getType()))
      return extractElement(builder, operation->getLoc(), source, indices);
  }
  Value selected;
  if (isa<RankedTensorType>(type)) {
    auto resultType = RankedTensorType::get(resultShape, type.getElementType());
    selected = builder.create<tensor::ExtractSliceOp>(operation->getLoc(),
        resultType, source, offsets, sizes, strides);
  } else {
    auto resultType = cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
        resultShape, cast<MemRefType>(type), offsets, sizes, strides));
    selected = builder.create<memref::SubViewOp>(operation->getLoc(), resultType,
                                                source, offsets, sizes, strides);
  }
  if (!inserted) {
    if (isa<ViewLoadOp, BufferLoadOp, GatherOp>(operation) &&
        isa<MemRefType>(selected.getType()))
      return Value(builder.create<cpu::ReadOp>(operation->getLoc(),
          tensorType(operation->getResult(0).getType()), selected));
    if (isa<RankedTensorType>(selected.getType())) {
      auto resultType = tensorType(operation->getResult(0).getType());
      if (selected.getType() != resultType)
        selected = builder.create<tensor::CastOp>(operation->getLoc(),
                                                  resultType, selected);
    }
    return selected;
  }
  if (operation->getNumResults() != 1 || !isa<RankedTensorType>(operation->getResult(0).getType()))
    return operation->emitError("CPU inserted write axes are not implemented"), failure();
  auto tensor = cast<RankedTensorType>(operation->getResult(0).getType());
  auto shape = extents(operation->getResult(0), operation->getLoc());
  if (failed(shape)) return failure();
  Value output = emptyTensor(tensor, *shape, operation->getLoc());
  if (isa<MemRefType>(selected.getType()))
    selected = builder.create<cpu::ReadOp>(operation->getLoc(),
        RankedTensorType::get(resultShape, type.getElementType()), selected);
  auto result = builder.create<linalg::GenericOp>(operation->getLoc(), TypeRange{output.getType()}, ValueRange{selected}, ValueRange{output},
      SmallVector<AffineMap>{AffineMap::get(tensor.getRank(), 0, projection, builder.getContext()), builder.getMultiDimIdentityMap(tensor.getRank())},
      SmallVector<utils::IteratorType>(tensor.getRank(), utils::IteratorType::parallel),
      [](OpBuilder &nested, Location loc, ValueRange scalars) { nested.create<linalg::YieldOp>(loc, scalars[0]); });
  return result.getResult(0);
}


} // namespace intent::kir_to_cpu
