#include "Lowering.h"
#include "Intent/Target/DSA/BangC/Physical/IR/Views.h"
#include "Intent/Dialect/Block/IR/FragmentOpInterface.h"
#include "Intent/Dialect/Block/Analysis/IntegerRanges.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/ScopeExit.h"

namespace intent::bangc::block_lowering {
std::optional<ConstantIntRanges> Lowering::integerRange(Value value) {
  auto policy = block::integerRangePolicy();
  auto inferBlock = policy.infer;
  policy.infer = [&](Value current, IntegerRangeAnalysis &analysis) -> std::optional<ConstantIntRanges> {
    APInt number;
    if (Value mapped = values.lookupOrNull(current); mapped && matchPattern(mapped, m_ConstantInt(&number)))
      return ConstantIntRanges::constant(number);
    return inferBlock(current, analysis);
  };
  return IntegerRangeAnalysis(std::move(policy)).range(value);
}

Value Lowering::index(int64_t value) { return b.create<arith::ConstantIndexOp>(loc, value); }
Value Lowering::get(Value value) {
  if (Value result = values.lookupOrNull(value)) return result;
  auto *definition = value.getDefiningOp();
  if (!definition || !deferred.contains(definition)) return {};
  if (!materializing.insert(definition).second) return {};
  auto restore = llvm::make_scope_exit([&] { materializing.erase(definition); });
  OpBuilder::InsertionGuard guard(b);
  Block *scope = deferred.lookup(definition);
  if (b.getInsertionBlock() != scope) {
    Operation *anchor = b.getInsertionBlock()->getParentOp();
    while (anchor && anchor->getBlock() != scope) anchor = anchor->getParentOp();
    if (!anchor) {
      emitError(loc, "deferred Block value escaped its defining execution scope");
      return {};
    }
    b.setInsertionPoint(anchor);
  }
  Location previous = loc;
  LogicalResult status = operation(definition);
  loc = previous;
  return succeeded(status) ? values.lookupOrNull(value) : Value();
}

Value Lowering::allocate(Type element, ArrayRef<int64_t> shape) {
  if (element.isIndex()) element = b.getI64Type();
  int64_t rows = 1, columns = shape.empty() ? 1 : shape.back();
  for (int64_t extent : shape.drop_back(shape.empty() ? 0 : 1)) rows *= extent;
  auto memory = MemRefType::get({rows, columns}, element, MemRefLayoutAttrInterface(), b.getI64IntegerAttr(physical::nramSpace));
  Value result = b.create<memref::AllocaOp>(loc, memory);
  extents[result] = llvm::to_vector(shape);
  return result;
}

Value Lowering::allocate(Type logical) {
  auto dimensions = shape(logical);
  if (failed(dimensions)) return {};
  Type element = logical;
  if (auto fragment = dyn_cast<block::FragmentType>(logical)) element = fragment.getElementType();
  if (auto buffer = dyn_cast<block::BufferType>(logical)) element = buffer.getElementType();
  return allocate(element, *dimensions);
}

SmallVector<Value> Lowering::coordinates(Value memory, ValueRange logical) {
  auto &sizes = extents[memory];
  Value row = index(0);
  for (unsigned axis = 0; axis + 1 < logical.size(); ++axis)
    row = b.createOrFold<arith::AddIOp>(loc,
        b.createOrFold<arith::MulIOp>(loc, row, index(sizes[axis])), logical[axis]);
  return {row, logical.empty() ? index(0) : logical.back()};
}

Value Lowering::read(Value memory, ValueRange logical) {
  if (!isa<MemRefType>(memory.getType())) return memory;
  return b.create<memref::LoadOp>(loc, memory, coordinates(memory, logical));
}

FailureOr<Value> Lowering::element(Value original, Type domain, ValueRange coordinate) {
  auto fragment = dyn_cast<block::FragmentType>(original.getType());
  if (!fragment) {
    if (Value value = get(original)) return value;
    return emitError(loc, "BANG C element references an unbound scalar"), failure();
  }
  auto dimensions = shape(fragment);
  if (failed(dimensions)) return failure();
  SmallVector<Value> source(dimensions->size(), index(0));
  if (auto target = dyn_cast<block::FragmentType>(domain)) {
    auto projection = block::queryAxisProjection(fragment, target);
    if (!projection.isExact()) return emitError(loc, "BANG C element has no unique fragment projection"), failure();
    for (auto [axis, sourceAxis] : llvm::enumerate(projection.targetToSource))
      if (sourceAxis && (*dimensions)[*sourceAxis] != 1) source[*sourceAxis] = coordinate[axis];
  }
  if (auto *definition = original.getDefiningOp(); definition && deferred.contains(definition)) {
    if (auto range = dyn_cast<block::MakeRangeOp>(definition))
      return Value(b.create<arith::AddIOp>(loc, get(range.getStart()),
          b.create<arith::MulIOp>(loc, source.front(), get(range.getStep()))));
    if (isa<block::SplatOp, block::BroadcastOp>(definition))
      return element(definition->getOperand(0), fragment, source);
    SmallVector<Value> operands;
    for (Value operand : definition->getOperands()) {
      auto part = element(operand, fragment, source);
      if (failed(part)) return failure();
      operands.push_back(*part);
    }
    return scalar(definition, operands);
  }
  Value memory = get(original);
  if (!memory) return emitError(loc, "BANG C element references an unbound fragment"), failure();
  return read(memory, source);
}

LogicalResult Lowering::write(Value value, Value memory, ValueRange logical) {
  auto converted = castScalarValue(b, loc, value, cast<MemRefType>(memory.getType()).getElementType());
  if (failed(converted)) return emitError(loc, "BANG C storage conversion has no scalar realization");
  b.create<memref::StoreOp>(loc, *converted, memory, coordinates(memory, logical));
  return success();
}

LogicalResult Lowering::each(ArrayRef<int64_t> dimensions, function_ref<LogicalResult(ValueRange)> body) {
  SmallVector<Value> coordinates;
  std::function<LogicalResult(unsigned)> visit = [&](unsigned axis) -> LogicalResult {
    if (axis == dimensions.size()) return body(coordinates);
    OpBuilder::InsertionGuard guard(b);
    auto loop = b.create<scf::ForOp>(loc, index(0), index(dimensions[axis]), index(1));
    b.setInsertionPointToStart(loop.getBody());
    coordinates.push_back(loop.getInductionVar());
    LogicalResult status = visit(axis + 1);
    coordinates.pop_back();
    return status;
  };
  return visit(0);
}

LogicalResult Lowering::copy(Value input, Value output) {
  if (input == output) return success();
  if (input.getType() == output.getType()) {
    b.create<memref::CopyOp>(loc, input, output);
    return success();
  }
  if (!isa<MemRefType>(input.getType())) {
    auto converted = castScalarValue(b, loc, input, cast<MemRefType>(output.getType()).getElementType());
    if (failed(converted)) return failure();
    b.create<physical::FillOp>(loc, output, *converted);
    return success();
  }
  return emitError(loc, "BANG C state update changed its declared storage shape");
}

FailureOr<Value> Lowering::scalar(Operation *operation, ValueRange inputs) {
  SmallVector<Value> typed;
  for (auto [old, value] : llvm::zip(operation->getOperands(), inputs)) {
    Type type = old.getType();
    if (auto fragment = dyn_cast<block::FragmentType>(type)) type = fragment.getElementType();
    auto converted = castScalarValue(b, loc, value, type);
    if (failed(converted)) return failure();
    typed.push_back(*converted);
  }
  if (auto compare = dyn_cast<block::CompareOp>(operation)) {
    // Bound coordinates use the native i32 compare only with a full-range
    // proof. This also avoids CNCC's incorrect wide sign extraction for a
    // negative address-register intermediate; unbounded i64 data stays wide.
    auto left = integerRange(compare.getLhs()), right = integerRange(compare.getRhs());
    if (left && right && left->smin().isSignedIntN(32) && left->smax().isSignedIntN(32) &&
        right->smin().isSignedIntN(32) && right->smax().isSignedIntN(32)) {
      for (Value &input : typed) {
        auto narrow = castScalarValue(b, loc, input, b.getI32Type());
        if (failed(narrow)) return failure();
        input = *narrow;
      }
    }
  }
  if (auto binary = dyn_cast<block::BinaryOp>(operation);
      binary && binary.getApproximate() && binary.getOperatorKind() == BinaryOperator::TrueDivide && typed[0].getType().isF32()) {
    Value left = allocate(b.getF32Type(), {}), right = allocate(b.getF32Type(), {}), result = allocate(b.getF32Type(), {});
    if (failed(copy(typed[0], left)) || failed(copy(typed[1], right))) return failure();
    b.create<physical::BinaryOp>(loc, left, right, result, binary.getOperatorKindAttr(), binary.getApproximateAttr(), binary.getFlushToZeroAttr(), Value());
    return read(result, {});
  }
  if (auto unary = dyn_cast<block::UnaryOp>(operation);
      unary && unary.getApproximate() && unary.getOperatorKind() == UnaryOperator::Exp2 && typed[0].getType().isF32()) {
    Value input = allocate(b.getF32Type(), {}), result = allocate(b.getF32Type(), {});
    if (failed(copy(typed[0], input))) return failure();
    b.create<physical::UnaryOp>(loc, input, result, unary.getOperatorKindAttr(), unary.getApproximateAttr(), unary.getFlushToZeroAttr(), Value());
    return read(result, {});
  }
  if (auto unary = dyn_cast<block::UnaryOp>(operation);
      unary && !unary.getApproximate() && !unary.getFlushToZero() && unary.getOperatorKind() == UnaryOperator::Sigmoid) {
    Type resultType = typed[0].getType();
    auto input = castScalarValue(b, loc, typed[0], b.getF32Type());
    if (failed(input)) return failure();
    Value zero = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(0));
    Value one = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(1));
    Value exponential = b.create<math::ExpOp>(loc, b.create<arith::NegFOp>(loc, b.create<math::AbsFOp>(loc, *input)));
    Value numerator = b.create<arith::SelectOp>(loc,
        b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, *input, zero), exponential, one);
    return castScalarValue(b, loc, b.create<arith::DivFOp>(loc, numerator, b.create<arith::AddFOp>(loc, one, exponential)), resultType);
  }
  return lowerScalarOperation(operation, typed, b);
}

LogicalResult Lowering::elementwise(Operation *operation) {
  Value original = operation->getResult(0);
  if (!isa<block::FragmentType>(original.getType())) {
    SmallVector<Value> inputs;
    for (Value value : operation->getOperands()) inputs.push_back(get(value));
    auto result = scalar(operation, inputs);
    if (failed(result)) return failure();
    values.map(original, *result);
    return success();
  }
  Value result = allocate(original.getType());
  if (!result) return failure();
  if (auto select = dyn_cast<block::SelectOp>(operation)) {
    auto condition = integerRange(select.getCondition());
    if (condition && condition->umin() == condition->umax()) {
      Value selected = operation->getOperand(condition->umin().isZero() ? 2 : 1);
      if (selected.getType() == original.getType() || !isa<block::FragmentType>(selected.getType())) {
        if (failed(copy(get(selected), result))) return failure();
        values.map(original, result);
        return success();
      }
    }
  }
  if (auto range = integerRange(original); range && range->umin() == range->umax()) {
    Type element = cast<MemRefType>(result.getType()).getElementType();
    Value constant = b.create<arith::ConstantOp>(loc, b.getIntegerAttr(element, range->umin()));
    if (failed(copy(constant, result))) return failure();
    values.map(original, result);
    return success();
  }
  SmallVector<Value> inputs;
  for (Value value : operation->getOperands()) inputs.push_back(get(value));
  bool matching = llvm::all_of(inputs, [&](Value value) {
    auto type = dyn_cast<MemRefType>(value.getType());
    return type && type.getShape() == cast<MemRefType>(result.getType()).getShape();
  });
  Type element = cast<MemRefType>(result.getType()).getElementType();
  auto scalarType = [](Type type) {
    auto fragment = dyn_cast<block::FragmentType>(type);
    return fragment ? fragment.getElementType() : type;
  };
  auto unsignedType = [](Type type) {
    auto integer = dyn_cast<IntegerType>(type);
    return integer && integer.isUnsigned();
  };
  auto nativeCast = [&]() {
    Type input = scalarType(operation->getOperand(0).getType());
    if (isa<Float8E4M3FNType, Float8E5M2Type>(input) || isa<Float8E4M3FNType, Float8E5M2Type>(element)) return false;
    return (!unsignedType(input) && !unsignedType(element)) ||
        (isa<IntegerType>(input) && isa<IntegerType>(element) && input.getIntOrFloatBitWidth() == element.getIntOrFloatBitWidth());
  };
  if (auto op = dyn_cast<block::BinaryOp>(operation); op && matching && isa<FloatType>(element))
    b.create<physical::BinaryOp>(loc, inputs[0], inputs[1], result, op.getOperatorKindAttr(),
                                op.getApproximateAttr(), op.getFlushToZeroAttr(), Value());
  else if (auto op = dyn_cast<block::UnaryOp>(operation); op && matching && isa<FloatType>(element))
    b.create<physical::UnaryOp>(loc, inputs[0], result, op.getOperatorKindAttr(),
                               op.getApproximateAttr(), op.getFlushToZeroAttr(), Value());
  else if (isa<block::CastOp>(operation) && matching && nativeCast())
    b.create<physical::CastOp>(loc, inputs[0], result);
  else if (auto op = dyn_cast<block::CompareOp>(operation); op && matching && inputs[0].getType() == inputs[1].getType() &&
      cast<MemRefType>(inputs[0].getType()).getElementType().isInteger(64) && !unsignedType(scalarType(op.getLhs().getType())))
    b.create<physical::CompareOp>(loc, inputs[0], inputs[1], result, op.getPredicateAttr(), Value());
  else if (isa<block::SelectOp>(operation) && matching)
    b.create<physical::SelectOp>(loc, inputs[0], inputs[1], inputs[2], result, Value());
  else if (failed(each(extents[result], [&](ValueRange coordinates) {
    SmallVector<Value> scalars;
    for (Value input : operation->getOperands()) {
      auto value = this->element(input, original.getType(), coordinates);
      if (failed(value)) return failure();
      scalars.push_back(*value);
    }
    auto scalarValue = scalar(operation, scalars);
    if (failed(scalarValue)) return failure();
    return write(*scalarValue, result, coordinates);
  }))) return failure();
  values.map(original, result);
  return success();
}

Value Lowering::view(Value memory, ArrayRef<int64_t> dimensions) {
  auto type = cast<MemRefType>(memory.getType());
  SmallVector<OpFoldResult> sizes, strides;
  int64_t stride = 1;
  for (int64_t size : dimensions) sizes.push_back(b.getIndexAttr(size));
  for (int64_t size : llvm::reverse(dimensions)) {
    strides.insert(strides.begin(), b.getIndexAttr(stride)); stride *= size;
  }
  auto viewType = MemRefType::get(dimensions, type.getElementType(), MemRefLayoutAttrInterface(), type.getMemorySpace());
  return b.create<memref::ReinterpretCastOp>(loc, viewType, memory, b.getIndexAttr(0), sizes, strides);
}
}
