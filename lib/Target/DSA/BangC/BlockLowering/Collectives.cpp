#include "Lowering.h"
#include "Intent/Dialect/Block/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/Block/Analysis/UniformValues.h"
#include "Intent/Dialect/Block/Analysis/Helpers.h"
#include "Intent/Dialect/Block/Transforms/Value/Helpers.h"

namespace intent::bangc::block_lowering {
LogicalResult Lowering::collective(Operation *operation) {
  auto reduce = dyn_cast<block::ReduceOp>(operation);
  auto scan = dyn_cast<block::ScanOp>(operation);
  if (!reduce && !scan) return operation->emitError("structured traversal must be realized before BANG C storage lowering");
  ValueRange sources = reduce ? reduce.getSources() : scan.getSources();
  ValueRange identities = reduce ? reduce.getIdentities() : scan.getIdentities();
  ValueRange captures = reduce ? reduce.getCaptures() : scan.getCaptures();
  Region scalarCombine;
  Region &region = reduce ? reduce.getCombine() : scan.getCombine();
  bool scalarHelper = captures.empty() && succeeded(block::proveLaneWiseHelper(region));
  if (scalarHelper) {
    if (failed(block::scalarizeElementwiseCallback(region, scalarCombine))) return failure();
  }
  Block &combine = scalarHelper ? scalarCombine.front() : (reduce ? reduce.getCombine().front() : scan.getCombine().front());
  SmallVector<int64_t> axes = reduce ? llvm::to_vector(reduce.getAxes()) : SmallVector<int64_t>{static_cast<int64_t>(scan.getAxis())};
  SmallVector<Value> inputs, initials, captured, outputs, counts, finals, resultBuffers;
  SmallVector<SmallVector<int64_t>> stateShapes;
  for (Value input : sources) {
    Value value = get(input);
    SmallVector<int64_t> state;
    for (auto [axis, extent] : llvm::enumerate(extents[value]))
      if (!llvm::is_contained(axes, axis)) state.push_back(extent);
    stateShapes.push_back(state);
    inputs.push_back(view(value, extents[value]));
    for (int64_t extent : extents[value]) counts.push_back(index(extent));
  }
  for (auto [field, identity] : llvm::enumerate(identities)) {
    if (Value uniform = block::uniformScalarSource(identity)) {
      initials.push_back(get(uniform));
      continue;
    }
    Value value = get(identity);
    if (auto type = dyn_cast<MemRefType>(value.getType())) {
      if (type.getNumElements() == 1) initials.push_back(read(value, {}));
      else initials.push_back(view(value, stateShapes[field]));
    } else initials.push_back(value);
  }
  for (Value capture : captures) {
    Value value = get(capture);
    captured.push_back(isa<MemRefType>(value.getType()) ? view(value, extents[value]) : value);
  }
  for (Value original : operation->getResults()) {
    Value output = allocate(original.getType());
    if (!output) return failure();
    resultBuffers.push_back(output);
    outputs.push_back(view(output, extents[output]));
  }
  Region *body;
  if (reduce) {
    auto lowered = b.create<physical::SliceReduceOp>(loc, inputs, initials, captured, outputs, counts, axes);
    body = &lowered.getCombine();
  } else {
    for (Value input : sources) {
      auto sizes = shape(input.getType());
      if (failed(sizes)) return failure();
      sizes->erase(sizes->begin() + scan.getAxis());
      Type element = cast<block::FragmentType>(input.getType()).getElementType();
      Value output = allocate(element, *sizes);
      finals.push_back(view(output, *sizes));
    }
    auto lowered = b.create<physical::ScanOp>(loc, inputs, initials, captured, outputs, counts,
        finals, scan.getAxis(), scan.getInclusive(), scan.getReverse());
    body = &lowered.getCombine();
  }
  for (auto [original, buffer] : llvm::zip_equal(operation->getResults(), resultBuffers)) {
    if (isa<block::FragmentType>(original.getType())) values.map(original, buffer);
    else {
      auto converted = castScalarValue(b, loc, read(buffer, {}), original.getType());
      if (failed(converted)) return failure();
      values.map(original, *converted);
    }
  }
  body->push_back(new Block());
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(&body->front());
  for (BlockArgument original : combine.getArguments()) {
    auto dimensions = shape(original.getType());
    if (failed(dimensions)) return failure();
    Type type = original.getType();
    if (auto fragment = dyn_cast<block::FragmentType>(type)) {
      auto element = fragment.getElementType();
      if (element.isIndex()) element = b.getI64Type();
      auto shape = original.getArgNumber() < 2 * stateShapes.size() ?
          ArrayRef<int64_t>(stateShapes[original.getArgNumber() % stateShapes.size()]) : ArrayRef<int64_t>(*dimensions);
      type = MemRefType::get(shape, element, MemRefLayoutAttrInterface(), b.getI64IntegerAttr(physical::nramSpace));
    }
    Value argument = body->front().addArgument(type, loc);
    if (isa<MemRefType>(type)) {
      int64_t rows = 1, columns = dimensions->empty() ? 1 : dimensions->back();
      for (int64_t size : ArrayRef<int64_t>(*dimensions).drop_back(dimensions->empty() ? 0 : 1)) rows *= size;
      argument = view(argument, {rows, columns});
      extents[argument] = *dimensions;
    }
    values.map(original, argument);
  }
  if (failed(operations(combine))) return failure();
  SmallVector<Value> results;
  for (auto [field, original] : llvm::enumerate(combine.getTerminator()->getOperands())) {
    Value value = get(original);
    results.push_back(isa<MemRefType>(value.getType()) ? view(value, stateShapes[field]) : value);
  }
  b.create<physical::CollectiveYieldOp>(loc, results);
  return success();
}

LogicalResult Lowering::contraction(block::ContractOp operation) {
  auto axes = block::queryContractionAxes(operation);
  if (!axes) return operation.emitError("BANG C contraction has invalid paired axes");
  Value lhs = get(operation.getLhs()), rhs = get(operation.getRhs()), initial = get(operation.getAccumulator());
  Value result = allocate(operation.getResult().getType());
  if (!result || failed(copy(initial, result))) return failure();
  auto leftShape = extents[lhs], rightShape = extents[rhs];
  Type element = cast<MemRefType>(result.getType()).getElementType();
  auto signedByte = [](Type type) {
    auto integer = dyn_cast<IntegerType>(type);
    return integer && integer.getWidth() == 8 && !integer.isUnsigned();
  };
  bool exactInteger = element.isInteger(32) &&
      signedByte(cast<MemRefType>(lhs.getType()).getElementType()) &&
      signedByte(cast<MemRefType>(rhs.getType()).getElementType());
  if (element.isF32() && axes->hasCanonicalMatrixAxes() && axes->batch.empty() && leftShape.size() == 2 && rightShape.size() == 2) {
    b.create<physical::MatMulOp>(loc, lhs, rhs, result, index(leftShape[0]), index(leftShape[1]), index(rightShape[1]), false);
  } else if ((element.isF32() || exactInteger) && axes->reduction.size() == 1 && axes->lhsFree.size() <= 1 && axes->rhsFree.size() <= 1) {
    auto reduction = axes->reduction.front();
    int64_t depth = leftShape[reduction.lhs];
    int64_t rows = axes->lhsFree.empty() ? 1 : leftShape[axes->lhsFree.front()];
    int64_t columns = axes->rhsFree.empty() ? 1 : rightShape[axes->rhsFree.front()];
    int64_t chunk = exactInteger ? std::min<int64_t>(1024, depth) : depth;
    Type inputType = exactInteger ? b.getF16Type() : cast<MemRefType>(lhs.getType()).getElementType();
    Value left = allocate(inputType, {rows, chunk}), right = allocate(inputType, {chunk, columns});
    Value partial = allocate(b.getF32Type(), {rows, columns});
    SmallVector<int64_t> batches;
    for (auto pair : axes->batch) batches.push_back(leftShape[pair.lhs]);
    if (failed(each(batches, [&](ValueRange batch) -> LogicalResult {
      SmallVector<Value> output(extents[result].size(), index(0));
      for (auto [pair, coordinate] : llvm::zip(axes->batch, batch)) output[*axes->lhsResultAxes[pair.lhs]] = coordinate;
      auto bind = [&](ValueRange ij) {
        if (!axes->lhsFree.empty()) output[*axes->lhsResultAxes[axes->lhsFree.front()]] = ij[0];
        if (!axes->rhsFree.empty()) output[*axes->rhsResultAxes[axes->rhsFree.front()]] = ij[1];
      };
      auto loop = b.create<scf::ForOp>(loc, index(0), index(depth), index(chunk));
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(loop.getBody());
      Value active = b.create<arith::MinSIOp>(loc, index(chunk), b.create<arith::SubIOp>(loc, index(depth), loop.getInductionVar()));
      auto pack = [&](Value input, Value destination, bool lhsSide) {
        auto &mapping = lhsSide ? axes->lhsResultAxes : axes->rhsResultAxes;
        auto &free = lhsSide ? axes->lhsFree : axes->rhsFree;
        unsigned reduce = lhsSide ? reduction.lhs : reduction.rhs;
        return each(extents[destination], [&](ValueRange ij) -> LogicalResult {
          SmallVector<Value> coordinate(mapping.size(), index(0));
          for (auto [axis, resultAxis] : llvm::enumerate(mapping)) if (resultAxis) coordinate[axis] = output[*resultAxis];
          if (!free.empty()) coordinate[free.front()] = ij[lhsSide ? 0 : 1];
          Value k = ij[lhsSide ? 1 : 0];
          coordinate[reduce] = b.create<arith::AddIOp>(loc, loop.getInductionVar(), k);
          Value valid = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, k, active);
          if (failed(write(b.create<arith::ConstantOp>(loc, b.getZeroAttr(inputType)), destination, ij))) return failure();
          auto branch = b.create<scf::IfOp>(loc, valid, false);
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(branch.thenBlock());
          return write(read(input, coordinate), destination, ij);
        });
      };
      if (failed(pack(lhs, left, true)) || failed(pack(rhs, right, false))) return failure();
      if (exactInteger) {
        if (failed(copy(b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(0)), partial))) return failure();
      } else if (failed(each({rows, columns}, [&](ValueRange ij) { bind(ij); return write(read(result, output), partial, ij); }))) return failure();
      b.create<physical::MatMulOp>(loc, left, right, partial, index(rows), active, index(columns), false);
      return each({rows, columns}, [&](ValueRange ij) -> LogicalResult {
        bind(ij);
        auto value = castScalarValue(b, loc, read(partial, ij), element);
        if (failed(value)) return failure();
        Value next = exactInteger ? Value(b.create<arith::AddIOp>(loc, read(result, output), *value)) : *value;
        return write(next, result, output);
      });
    }))) return failure();
  } else {
    SmallVector<int64_t> reductions;
    for (auto pair : axes->reduction) reductions.push_back(leftShape[pair.lhs]);
    if (failed(each(extents[result], [&](ValueRange output) {
      SmallVector<Value> left(leftShape.size(), index(0)), right(rightShape.size(), index(0));
      for (unsigned axis = 0; axis < left.size(); ++axis)
        if (axes->lhsResultAxes[axis]) left[axis] = output[*axes->lhsResultAxes[axis]];
      for (unsigned axis = 0; axis < right.size(); ++axis)
        if (axes->rhsResultAxes[axis]) right[axis] = output[*axes->rhsResultAxes[axis]];
      return each(reductions, [&](ValueRange member) -> LogicalResult {
        for (auto [index, pair] : llvm::enumerate(axes->reduction)) {
          left[pair.lhs] = member[index]; right[pair.rhs] = member[index];
        }
        auto a = castScalarValue(b, loc, read(lhs, left), element);
        auto c = castScalarValue(b, loc, read(rhs, right), element);
        if (failed(a) || failed(c)) return failure();
        Value accumulator = read(result, output);
        Value value = isa<FloatType>(element) ? Value(b.create<arith::AddFOp>(loc, b.create<arith::MulFOp>(loc, *a, *c), accumulator)) :
            Value(b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, *a, *c), accumulator));
        return write(value, result, output);
      });
    }))) return failure();
  }
  values.map(operation.getResult(), result);
  return success();
}
}
