#include "Construction.h"
#include "Intent/Target/BangC/NativeWorkspace.h"

namespace intent::kir_to_dsa {

LogicalResult Construction::tensorAccess(Operation *op) {
  auto relation = analysis.indexRelation(op);
  if (failed(relation)) return failure();
  Location loc = op->getLoc();
  bool store = isa<ViewStoreOp, ScatterUniqueOp>(op);
  bool external = isa<ViewType>(relation->source.getType());
  Value source = get(relation->source);
  if (!source) return op->emitError("DSA indexed source is unavailable");
  SmallVector<ReifiedIndexTerm> terms;
  auto materialization = indexMaterialization(loc);
  for (const auto &term : relation->terms) {
    auto reified = materializeIndexTerm(relation->source, term, materialization);
    if (failed(reified)) return op->emitError("DSA logical access coordinate is unavailable");
    terms.push_back(*reified);
  }
  auto access = cast<IndexedAccessOpInterface>(op);
  Value original = store ? access.getStoredValue() : op->getResult(0);
  auto tensor = dyn_cast<RankedTensorType>(original.getType());
  LocalShape shape;
  if (tensor) {
    auto selected = localShape(original, loc);
    if (failed(selected)) return failure();
    shape = *selected;
  }
  Value data = store ? get(original) : tensor ? allocateTensor(loc, tensor.getElementType(), shape) : Value();
  if (store && !data) return op->emitError("DSA store data is unavailable");
  Value validity = access.getAccessValidity(), fallback = access.getAccessFill();
  bool allValid = !validity || constantTrue(validity);
  if (relation->resultDimensionIdentities.size() != shape.size())
    return op->emitError("DSA index relation and local result ranks disagree");
  // Inserting unit axes changes the logical shape without permuting elements.
  // Retain the independent snapshot and explicitly zero inactive padding.
  if (!external && !store && tensor && localShapes.count(source) &&
      allValid) {
    const auto &inputShape = localShapes.find(source)->second;
    auto inputType = cast<MemRefType>(source.getType()), outputType = cast<MemRefType>(data.getType());
    bool contiguous = inputType.getLayout().isIdentity() && inputType.getElementType() == outputType.getElementType() &&
        inputType.getNumElements() == outputType.getNumElements();
    bool complete = true;
    unsigned sourceAxis = 0;
    for (auto [term, reified] : llvm::zip_equal(relation->terms, terms)) {
      if (term.sourceAxis && !reified.range) { contiguous = false; break; }
      const auto &target = shape[term.resultAxes.front()];
      if (!term.sourceAxis) {
        contiguous &= target.capacity == 1 && matchPattern(target.extent, m_One()) &&
            matchPattern(target.count, m_One()) && matchPattern(target.begin, m_Zero());
      } else {
        if (!term.sourceAxis || *term.sourceAxis != sourceAxis || sourceAxis >= inputShape.size()) {
          contiguous = false; break;
        }
        const auto &input = inputShape[sourceAxis++];
        contiguous &= matchPattern(reified.range->begin, m_Zero()) &&
            matchPattern(reified.range->step, m_One()) &&
            sameIndex(reified.range->count, input.extent);
        complete &= sameIndex(input.count, index(loc, input.capacity));
        contiguous &= input.capacity == target.capacity &&
            sameIndex(input.begin, target.begin) && sameIndex(input.count, target.count) &&
            sameIndex(input.extent, target.extent);
      }
    }
    if (contiguous && sourceAxis == inputShape.size() && (complete || shape.size() <= 2)) {
      Value reshaped = contiguousView(loc, source, shape);
      if (complete) b.create<memref::CopyOp>(loc, reshaped, data);
      else b.create<dsa::LoadTileOp>(loc, reshaped, data, index(loc, 0), index(loc, outputType.getDimSize(1)), index(loc, 1),
          shape.size() == 2 ? shape[0].count : index(loc, 1), shape.empty() ? index(loc, 1) : shape.back().count);
      values.map(original, data);
      return success();
    }
  }
  Value valid = !allValid && validity ? get(validity) : Value();
  Value fill = !allValid && fallback ? get(fallback) : Value();
  if (!allValid && ((validity && !valid) || (fallback && !fill)))
    return op->emitError("DSA access predicate or fill is unavailable");
  // Rectangular external accesses preserve arbitrary view strides. Their
  // selected origins and tails are explicit before BANG C serialization.
  bool rectangle = external && allValid && (tensor || !store) && shape.size() <= 2;
  SmallVector<AffineIndices> affine;
  if (rectangle) {
    for (const auto &term : relation->terms) {
      if (!term.sourceAxis) continue;
      auto coordinate = affineAccessTerm(relation->source, term, shape.size(), loc);
      if (failed(coordinate)) { rectangle = false; break; }
      affine.push_back(*coordinate);
    }
  }
  if (rectangle) {
    Value offset = index(loc, 0);
    SmallVector<Value> strides(shape.size(), index(loc, 0));
    for (auto [axis, geometry] : llvm::enumerate(affine)) {
      Value step = stride(loc, source, axis), coordinate = geometry.base;
      for (unsigned outputAxis = 0; outputAxis < shape.size(); ++outputAxis) {
        Value coefficient = geometry.steps[outputAxis];
        coordinate = add(loc, coordinate, mul(loc, shape[outputAxis].begin, coefficient));
        strides[outputAxis] = add(loc, strides[outputAxis], mul(loc, step, coefficient));
      }
      offset = add(loc, offset, mul(loc, coordinate, step));
    }
    if (!tensor) {
      Value value = b.create<dsa::LoadScalarOp>(loc, cast<MemRefType>(source.getType()).getElementType(), source, offset);
      values.map(original, scalarCast(loc, value, original.getType()));
      return success();
    }
    Value rows = shape.size() == 2 ? shape[0].count : index(loc, 1);
    Value cols = shape.empty() ? index(loc, 1) : shape.back().count;
    Value rowStride = strides.size() == 2 ? strides[0] : index(loc, 0);
    Value colStride = strides.empty() ? index(loc, 0) : strides.back();
    if (store) b.create<dsa::StoreTileOp>(loc, data, source, offset, rowStride, colStride, rows, cols);
    else { b.create<dsa::LoadTileOp>(loc, source, data, offset, rowStride, colStride, rows, cols); values.map(original, data); }
    return success();
  }
  if (!store && !data) data = allocateTensor(loc, original.getType(), shape);
  auto readAt = [&](Value value, ValueRange coordinates) {
    return isa<MemRefType>(value.getType()) ? loadLocal(loc, value, coordinates) : value;
  };
  // Index tensors are immutable operands. Bind their storage outside a masked
  // element's branch; only the selected coordinate is loaded inside that branch.
  for (const auto &term : terms)
    if (term.tensor && !get(term.tensor))
      return op->emitError("DSA index operand is unavailable");
  auto locate = [&](ValueRange coordinates, Value &offset, SmallVectorImpl<Value> &sourceCoordinates) -> LogicalResult {
    auto logical = accessCoordinates(*relation, shape, coordinates, loc);
    if (failed(logical)) return op->emitError("DSA access coordinates could not be materialized");
    offset = index(loc, 0);
    for (auto [axis, coordinate] : llvm::enumerate(*logical)) {
      if (external) offset = add(loc, offset, mul(loc, coordinate, stride(loc, source, axis)));
      else {
        auto local = localShapes.find(source);
        if (local == localShapes.end() || axis >= local->second.size())
          return op->emitError("DSA gather source has no local axis binding");
        sourceCoordinates.push_back(sub(loc, coordinate, local->second[axis].begin));
      }
    }
    return success();
  };
  // Preserve irregular row addresses together with their regular trailing
  // intervals, allowing the target to coalesce rows during local supply.
  if (external && !store && allValid && tensor && shape.size() == 1 &&
      cast<MemRefType>(source.getType()).getElementType() == cast<MemRefType>(data.getType()).getElementType()) {
    int64_t capacity = shape[0].capacity;
    auto offsetShape = bangc::rowOffsetsShape(capacity);
    Value offsets = allocate(loc, b.getI64Type(), offsetShape[0], offsetShape[1]);
    if (failed(loop(loc, index(loc, 0), shape[0].count, index(loc, 1), [&](Value i) {
      Value offset;
      SmallVector<Value> ignored;
      if (failed(locate(ValueRange{i}, offset, ignored))) return failure();
      Value stored = b.create<arith::IndexCastOp>(loc, b.getI64Type(), offset);
      b.create<memref::StoreOp>(loc, stored, offsets, ValueRange{index(loc, 0), i});
      return success();
    }))) return failure();
    auto storage = cast<MemRefType>(data.getType());
    auto transposed = MemRefType::get({capacity, 1}, storage.getElementType(), MemRefLayoutAttrInterface{}, storage.getMemorySpace());
    Value rows = b.create<memref::ReinterpretCastOp>(loc, transposed, data, int64_t(0),
        ArrayRef<int64_t>{capacity, 1}, ArrayRef<int64_t>{1, 1});
    b.create<dsa::GatherRowsOp>(loc, source, offsets, rows, index(loc, 1), shape[0].count, index(loc, 1), Value(), false);
    values.map(original, data);
    return success();
  }
  bool rowTiles = external && !store && allValid && tensor && shape.size() == 2 &&
      cast<MemRefType>(source.getType()).getElementType() == cast<MemRefType>(data.getType()).getElementType();
  int columnTerm = -1;
  for (auto [i, term] : llvm::enumerate(relation->terms)) {
    const auto &reified = terms[i];
    if (reified.range && term.resultAxes.front() == 1) {
      rowTiles &= columnTerm < 0;
      columnTerm = i;
    }
    if (!reified.tensor) continue;
    auto indexType = cast<RankedTensorType>(reified.tensor.getType());
    auto local = localShapes.find(values.lookupOrNull(reified.tensor));
    if (local == localShapes.end()) { rowTiles = false; continue; }
    for (int64_t axis = 0; axis < indexType.getRank(); ++axis) {
      unsigned mapped = term.indexAxes[axis];
      if (mapped == 1 && indexType.getDimSize(axis) != 1 && !matchPattern(local->second[axis].extent, m_One()))
        rowTiles = false;
    }
  }
  if (rowTiles && columnTerm >= 0) {
    const auto &term = relation->terms[columnTerm];
    Value columnStride = mul(loc, stride(loc, source, *term.sourceAxis),
                             terms[columnTerm].range->step);
    auto offsetShape = bangc::rowOffsetsShape(shape[0].capacity);
    Value offsets = allocate(loc, b.getI64Type(), offsetShape[0], offsetShape[1]);
    Value nonempty = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, shape[1].count, index(loc, 0));
    auto branch = b.create<scf::IfOp>(loc, nonempty, false);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(branch.thenBlock());
      if (failed(loop(loc, index(loc, 0), shape[0].count, index(loc, 1), [&](Value i) {
        Value offset;
        SmallVector<Value> ignored;
        if (failed(locate(ValueRange{i, index(loc, 0)}, offset, ignored))) return failure();
        Value stored = b.create<arith::IndexCastOp>(loc, b.getI64Type(), offset);
        b.create<memref::StoreOp>(loc, stored, offsets, ValueRange{index(loc, 0), i});
        return success();
      }))) return failure();
      b.create<dsa::GatherRowsOp>(loc, source, offsets, data, columnStride, shape[0].count, shape[1].count, Value(), false);
    }
    values.map(original, data); return success();
  }
  if (failed(eachElement(loc, shape, [&](ValueRange coordinates) -> LogicalResult {
    if (fill) storeLocal(loc, readAt(fill, coordinates), data, coordinates);
    auto access = [&]() -> LogicalResult {
      Value offset;
      SmallVector<Value> sourceCoordinates;
      if (failed(locate(coordinates, offset, sourceCoordinates))) return failure();
      if (store) {
        Value value = isa<MemRefType>(data.getType()) ? loadLocal(loc, data, coordinates) : data;
        b.create<dsa::StoreScalarOp>(loc, value, source, offset);
      } else {
        Value value = external ? Value(b.create<dsa::LoadScalarOp>(loc, cast<MemRefType>(source.getType()).getElementType(), source, offset))
                               : loadLocal(loc, source, sourceCoordinates);
        storeLocal(loc, value, data, coordinates);
      }
      return success();
    };
    if (!valid) return access();
    auto branch = b.create<scf::IfOp>(loc, readAt(valid, coordinates), false);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(branch.thenBlock());
    return access();
  }))) return failure();
  if (!store) values.map(original, tensor ? data : scalarCast(loc, loadLocal(loc, data, {}), original.getType()));
  return success();
}

Value Construction::stride(Location loc, Value view, int64_t axis) {
  if (auto argument = dyn_cast<BlockArgument>(view); argument && argument.getOwner() == &function.front()) {
    auto interface = intent::getPublicInterface(function);
    auto parameter = intent::getPublicView(interface, argument.getArgNumber());
    if (parameter.getConstraints().getHasStrides())
      if (auto fixed = dyn_cast<IntegerAttr>(parameter.getConstraints().getStrides()[axis]))
        return index(loc, fixed.getInt());
  }
  return b.create<dsa::StrideOp>(loc, b.getIndexType(), view, b.getI64IntegerAttr(axis));
}

} // namespace intent::kir_to_dsa
