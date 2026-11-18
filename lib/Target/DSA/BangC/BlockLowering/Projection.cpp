#include "Lowering.h"
#include "Intent/Dialect/Block/IR/FragmentOpInterface.h"

namespace intent::bangc::block_lowering {
FailureOr<SmallVector<int64_t>> Lowering::projectStrides(
    Operation *operation, ArrayRef<int64_t> sourceStrides) {
  auto relations = block::queryFragmentOperandRelations(operation);
  auto input = shape(operation->getOperand(0).getType());
  auto output = shape(operation->getResult(0).getType());
  if (failed(relations) || relations->size() != 1 || failed(input) ||
      failed(output) || input->size() != sourceStrides.size()) return failure();
  SmallVector<int64_t> strides(output->size(), 0);
  for (const auto &group : relations->front().groups) {
    if (group.kind == block::FragmentAxisRelationKind::Broadcast) continue;
    SmallVector<unsigned> from, to;
    for (unsigned axis : group.sourceAxes) if ((*input)[axis] != 1) from.push_back(axis);
    for (unsigned axis : group.resultAxes) if ((*output)[axis] != 1) to.push_back(axis);
    if (from.size() != to.size()) return failure();
    for (auto [sourceAxis, axis] : llvm::zip(from, to)) {
      if ((*input)[sourceAxis] != (*output)[axis]) return failure();
      strides[axis] = sourceStrides[sourceAxis];
    }
  }
  return strides;
}

LogicalResult Lowering::projection(Operation *operation) {
  Value original = operation->getResult(0);
  if (auto reshape = dyn_cast<block::ReshapeOp>(operation)) {
    Value input = get(reshape.getValue());
    if (!isa<block::FragmentType>(original.getType())) {
      SmallVector<Value> coordinate(extents[input].size(), index(0));
      auto converted = castScalarValue(b, loc, read(input, coordinate), original.getType());
      if (failed(converted)) return failure();
      values.map(original, *converted);
      return success();
    }
    auto dimensions = shape(original.getType());
    if (failed(dimensions)) return failure();
    if (isa<MemRefType>(input.getType())) {
      int64_t rows = 1, columns = dimensions->empty() ? 1 : dimensions->back();
      for (int64_t extent : ArrayRef<int64_t>(*dimensions).drop_back(dimensions->empty() ? 0 : 1)) rows *= extent;
      Value result = view(input, {rows, columns});
      extents[result] = *dimensions;
      values.map(original, result);
      return success();
    }
  }
  Value result = allocate(original.getType());
  if (!result) return failure();
  if (auto splat = dyn_cast<block::SplatOp>(operation)) {
    if (failed(copy(get(splat.getValue()), result))) return failure();
  } else if (auto range = dyn_cast<block::MakeRangeOp>(operation)) {
    if (failed(each(extents[result], [&](ValueRange coordinate) {
      Value value = b.create<arith::AddIOp>(loc, get(range.getStart()),
          b.create<arith::MulIOp>(loc, coordinate.front(), get(range.getStep())));
      return write(value, result, coordinate);
    }))) return failure();
  } else if (auto transpose = dyn_cast<block::TransposeOp>(operation)) {
    Value input = get(transpose.getValue());
    auto dimensions = extents[input];
    if (dimensions.size() == 2 && transpose.getPermutation() == ArrayRef<int64_t>({1, 0})) {
      b.create<physical::TransposeOp>(loc, input, result, index(dimensions[0]), index(dimensions[1]));
    } else if (failed(each(extents[result], [&](ValueRange coordinate) {
      SmallVector<Value> sourceCoordinates(coordinate.size());
      for (auto [axis, sourceAxis] : llvm::enumerate(transpose.getPermutation())) sourceCoordinates[sourceAxis] = coordinate[axis];
      return write(read(input, sourceCoordinates), result, coordinate);
    }))) return failure();
  } else if (auto join = dyn_cast<block::JoinOp>(operation)) {
    if (failed(each(extents[result], [&](ValueRange coordinate) {
      SmallVector<Value> sourceCoordinates(coordinate);
      Value lane = sourceCoordinates[join.getAxis()];
      sourceCoordinates.erase(sourceCoordinates.begin() + join.getAxis());
      Value first = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, lane, index(0));
      Value value = b.create<arith::SelectOp>(loc, first,
          read(get(join.getLhs()), sourceCoordinates), read(get(join.getRhs()), sourceCoordinates));
      return write(value, result, coordinate);
    }))) return failure();
  } else if (isa<block::BroadcastOp, block::ReshapeOp>(operation)) {
    Value input = get(operation->getOperand(0));
    if (!isa<MemRefType>(input.getType())) {
      if (failed(copy(input, result))) return failure();
    } else {
      auto inputType = cast<MemRefType>(input.getType());
      if (inputType.getNumElements() == 1) {
        SmallVector<Value> origin(extents[input].size(), index(0));
        if (failed(copy(read(input, origin), result))) return failure();
        values.map(original, result);
        return success();
      }
      auto relations = block::queryFragmentOperandRelations(operation);
      if (failed(relations) || relations->size() != 1) return failure();
      auto relation = relations->front();
      // Block extents may be symbolic until this complete target configuration
      // is bound. A logical ownership axis that resolves to one is not a
      // physical broadcast, even when its symbolic expression differs from 1.
      // Prove the projection on these bound extents before choosing a transfer.
      {
        SmallVector<int64_t> sourceStrides(extents[input].size());
        int64_t stride = 1;
        for (unsigned axis = sourceStrides.size(); axis-- > 0;) {
          sourceStrides[axis] = stride;
          stride *= extents[input][axis];
        }
        auto projected = projectStrides(operation, sourceStrides);
        bool rectangular = succeeded(projected);
        SmallVector<int64_t> steps = rectangular ? *projected : SmallVector<int64_t>();
        // Local storage collapses all leading logical axes into one row.
        // They must therefore describe one affine row stride; a mixed
        // broadcast/repetition still requires the general projection below.
        int64_t rowStride = 0, rowExtent = 1;
        bool haveRowStride = false;
        for (unsigned axis = steps.empty() ? 0 : steps.size() - 1; axis-- > 0;) {
          if (extents[result][axis] == 1) continue;
          if (!haveRowStride) {
            rowStride = steps[axis];
            haveRowStride = true;
          } else if (steps[axis] != rowStride * rowExtent) {
            rectangular = false;
          }
          rowExtent *= extents[result][axis];
        }
        if (rectangular) {
          auto sizes = cast<MemRefType>(result.getType()).getShape();
          b.create<physical::LoadTileOp>(loc, input, result, index(0),
              index(rowStride), index(steps.empty() ? 0 : steps.back()),
              index(sizes[0]), index(sizes[1]));
          values.map(original, result);
          return success();
        }
      }
      if (failed(each(extents[result], [&](ValueRange coordinate) {
        SmallVector<Value> sourceCoordinates(extents[input].size(), index(0));
        for (const auto &group : relation.groups) {
          if (group.kind == block::FragmentAxisRelationKind::Broadcast) continue;
          Value linear = index(0);
          for (unsigned axis : group.resultAxes)
            linear = b.createOrFold<arith::AddIOp>(loc,
                b.createOrFold<arith::MulIOp>(loc, linear, index(extents[result][axis])), coordinate[axis]);
          for (unsigned axis : llvm::reverse(group.sourceAxes)) {
            sourceCoordinates[axis] = b.create<arith::RemSIOp>(loc, linear, index(extents[input][axis]));
            linear = b.create<arith::DivSIOp>(loc, linear, index(extents[input][axis]));
          }
        }
        return write(read(input, sourceCoordinates), result, coordinate);
      }))) return failure();
    }
  } else return operation->emitError("BANG C projection has no realization");
  values.map(original, result);
  return success();
}
}
