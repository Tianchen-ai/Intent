#include "CollectiveLowering.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

namespace intent::dsa::collective {

bool Lowering::nativeReduction() {
  if (!scalar || fields.size() != 1 || axes.size() != 1 ||
      !captures.empty() || !llvm::hasSingleElement(body.without_terminator()))
    return false;
  Operation &expression = body.front();
  auto kindOr = binaryKind(&expression);
  if (!kindOr || expression.getNumOperands() != 2 ||
      expression.getOperand(0) != body.getArgument(0) ||
      expression.getOperand(1) != body.getArgument(1) ||
      body.getTerminator()->getOperand(0) != expression.getResult(0))
    return false;
  BinaryOperator kind = *kindOr;
  const Field &field = fields.front();
  auto type = cast<MemRefType>(field.source.getType());
  Type element = type.getElementType();
  unsigned axis = axes.front();
  bool boolean = element.isInteger(1) &&
      (kind == BinaryOperator::LogicalOr || kind == BinaryOperator::LogicalAnd);
  bool numeric = element.isF32() &&
      (kind == BinaryOperator::Add || kind == BinaryOperator::Maximum ||
       kind == BinaryOperator::Minimum || kind == BinaryOperator::MaximumNum ||
       kind == BinaryOperator::MinimumNum);
  if (!boolean && !numeric)
    return false;
  bool shapedInitial = isa<MemRefType>(field.initial.getType()) &&
                       !field.freeShape.empty();
  auto function = operation->getParentOfType<func::FuncOp>();
  auto configuration = function->getAttrOfType<ConfigurationAttr>(
      "intent_dsa.configuration");
  bool boundedRows = type.getRank() == 2 && axis == 1 &&
      type.getDimSize(0) > 1 && type.getDimSize(1) < 65536;
  bool floatRows = boundedRows && numeric &&
      (kind == BinaryOperator::Add ||
       type.getDimSize(0) * (type.getDimSize(1) + 1) <=
           configuration.getLocalBytes() / 16);
  bool floatColumns = type.getRank() == 2 && axis == 0 &&
                       numeric && kind == BinaryOperator::Add;
  bool rowwise = floatRows || floatColumns || (boundedRows && boolean);
  if (!rowwise && axis + 1 != unsigned(type.getRank()))
    return false;

  // Native row reductions take one scalar identity. A carried vector is a
  // seed, not that identity: reduce the current members, then apply the actual
  // helper to the incoming seed exactly once. Use the native arith identity
  // rules (notably NaN for maxnum/minnum); addition needs negative zero so the
  // extra partial-reduction identity does not change signed zero.
  std::optional<TypedAttr> neutral;
  if (rowwise && numeric && shapedInitial) {
    neutral = arith::getNeutralElement(&expression);
    if (kind == BinaryOperator::Add)
      neutral = builder.getFloatAttr(element,
          APFloat::getZero(cast<FloatType>(element).getFloatSemantics(), true));
    if (!neutral) return false;
  }

  Value input = physicalView(builder, location, field.source);
  Value initial = field.initial;
  if (isa<MemRefType>(initial.getType()) && !shapedInitial)
    initial = load(builder, location, initial, {});
  if (boolean) {
    Value promoted = allocate(builder, location, builder.getF32Type(),
                               type.getShape());
    builder.create<CastOp>(location, input,
                           physicalView(builder, location, promoted));
    input = physicalView(builder, location, promoted);
  }
  auto zero = [&]() -> Value {
    return builder.create<arith::ConstantFloatOp>(
        location, APFloat(0.0f), builder.getF32Type());
  };
  if (rowwise) {
    int64_t columns = type.getDimSize(1 - axis);
    Value accumulator = allocate(builder, location, builder.getF32Type(),
                                  {1, columns});
    Value scratch = floatRows && kind != BinaryOperator::Add
        ? allocate(builder, location, builder.getF32Type(),
                   {1, columns * (type.getDimSize(1) + 1)})
        : allocate(builder, location, builder.getF32Type(), {1, columns});
    Value identity = boolean ? zero() : shapedInitial
        ? Value(builder.create<arith::ConstantOp>(location, *neutral)) : initial;
    builder.create<ReduceOp>(location, input, accumulator, scratch,
        field.counts[axis], identity,
        BinaryOperatorAttr::get(builder.getContext(),
                                 boolean ? BinaryOperator::Add : kind),
        builder.getI64IntegerAttr(axis));
    if (boolean) {
      Value integerCount = builder.create<arith::IndexCastOp>(
          location, builder.getI64Type(), field.counts[axis]);
      Value threshold = kind == BinaryOperator::LogicalAnd
          ? Value(builder.create<arith::SIToFPOp>(
                location, builder.getF32Type(), integerCount)) : identity;
      (void)forEach(builder, location, field.freeCounts,
                    [&](ValueRange coordinates) {
        Value sum = load(builder, location, accumulator,
                          {index(builder, location, 0), coordinates.front()});
        Value predicate = builder.create<arith::CmpFOp>(
            location, kind == BinaryOperator::LogicalAnd
                ? arith::CmpFPredicate::OEQ : arith::CmpFPredicate::UNE,
            sum, threshold);
        Value seed = shapedInitial
            ? load(builder, location, initial, coordinates) : initial;
        Value result = kind == BinaryOperator::LogicalAnd
            ? Value(builder.create<arith::AndIOp>(location, seed, predicate))
            : Value(builder.create<arith::OrIOp>(location, seed, predicate));
        store(builder, location, result, field.output, coordinates);
        return success();
      });
    } else {
      if (shapedInitial) {
        auto combined = liftScalarCombine(builder, location, body,
            {initial, view(builder, location, accumulator, field.freeShape)},
            field.freeShape);
        assert(succeeded(combined) && "native binary helper has a tile mapping");
        accumulator = physicalView(builder, location, combined->front());
      }
      builder.create<LoadTileOp>(
          location, accumulator, physicalView(builder, location, field.output),
          index(builder, location, 0), index(builder, location, 0),
          index(builder, location, 1), index(builder, location, 1),
          field.freeCounts.front());
    }
    return true;
  }

  int64_t capacity = llvm::alignTo(type.getDimSize(axis), int64_t(32));
  bool direct = type.getRank() == 1 && type.getDimSize(axis) == capacity;
  Value row = direct ? input : allocate(builder, location, builder.getF32Type(),
                                        {1, capacity});
  Value scratch = allocate(builder, location, builder.getF32Type(),
                            {1, capacity});
  Value reduced = allocate(builder, location, builder.getF32Type(), {1, 1});
  if (boolean)
    kind = kind == BinaryOperator::LogicalOr ? BinaryOperator::Maximum
                                            : BinaryOperator::Minimum;
  (void)forEach(builder, location, field.freeCounts,
                [&](ValueRange coordinates) {
    Value identity = shapedInitial
        ? load(builder, location, initial, coordinates) : initial;
    if (boolean)
      identity = builder.create<arith::UIToFPOp>(location, builder.getF32Type(), identity);
    SmallVector<Value> source(coordinates);
    source.push_back(index(builder, location, 0));
    if (!direct)
      builder.create<LoadTileOp>(
          location, input, row, linearOffset(builder, location, type, source),
          index(builder, location, 0), index(builder, location, 1),
          index(builder, location, 1), field.counts[axis]);
    builder.create<ReduceOp>(location, row, reduced, scratch, field.counts[axis],
        identity, BinaryOperatorAttr::get(builder.getContext(), kind),
        builder.getI64IntegerAttr(1));
    Value result = load(builder, location, reduced,
                         {index(builder, location, 0), index(builder, location, 0)});
    if (boolean)
      result = builder.create<arith::CmpFOp>(location, arith::CmpFPredicate::UNE,
                                             result, zero());
    store(builder, location, result, field.output, coordinates);
    return success();
  });
  return true;
}

bool Lowering::treeReduction() {
  if (!scalar || !fields.front().freeShape.empty() ||
      !canLiftScalarCombine(body))
    return false;
  auto first = cast<MemRefType>(fields.front().source.getType());
  if (axes.size() != unsigned(first.getRank()) || first.getNumElements() < 2)
    return false;
  for (const Field &field : fields) {
    auto type = cast<MemRefType>(field.source.getType());
    if (type.getShape() != first.getShape())
      return false;
    for (auto [axis, count] : llvm::enumerate(field.counts)) {
      APInt value;
      if (!matchPattern(count, m_ConstantInt(&value)) ||
          value.getSExtValue() != type.getDimSize(axis))
        return false;
    }
  }
  auto configuration = operation->getParentOfType<func::FuncOp>()
      ->getAttrOfType<ConfigurationAttr>("intent_dsa.configuration");
  int64_t width = 1;
  while (width * 2 <= std::min(configuration.getRegionTile(),
                              first.getNumElements()))
    width *= 2;
  if (width < 2 || first.getNumElements() % width)
    return false;
  SmallVector<Value> state, next, chunks;
  auto fetch = [&](Value offset) {
    for (auto [field, chunk] : llvm::zip(fields, chunks))
      builder.create<LoadTileOp>(
          location, physicalView(builder, location, field.source), chunk,
          offset, index(builder, location, 0), index(builder, location, 1),
          index(builder, location, 1), index(builder, location, width));
  };
  for (const Field &field : fields) {
    Type element = cast<MemRefType>(field.source.getType()).getElementType();
    state.push_back(allocate(builder, location, element, {1, width}));
    next.push_back(allocate(builder, location, element, {1, width}));
    chunks.push_back(allocate(builder, location, element, {1, width}));
  }
  fetch(index(builder, location, 0));
  for (auto [chunk, slot] : llvm::zip(chunks, state))
    copy(builder, location, chunk, slot);
  auto liftedCombine = [&](ValueRange lhs, ValueRange rhs,
                            int64_t count) -> SmallVector<Value> {
    SmallVector<Value> arguments(lhs);
    llvm::append_range(arguments, rhs);
    llvm::append_range(arguments, captures);
    auto result = liftScalarCombine(builder, location, body, arguments,
                                      {1, count});
    assert(succeeded(result) && "tile mapping was checked before rewriting");
    return *result;
  };
  auto loop = builder.create<scf::ForOp>(
      location, index(builder, location, width),
      index(builder, location, first.getNumElements()),
      index(builder, location, width));
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    fetch(loop.getInductionVar());
    auto combined = liftedCombine(state, chunks, width);
    for (auto [value, destination] : llvm::zip(combined, next))
      copy(builder, location, value, destination);
    for (auto [value, destination] : llvm::zip(next, state))
      copy(builder, location, value, destination);
  }
  SmallVector<Value> partial(state);
  for (int64_t count = width; count > 1; count /= 2) {
    SmallVector<Value> left, right;
    for (Value value : partial) {
      if (!isa<MemRefType>(value.getType())) {
        left.push_back(value);
        right.push_back(value);
        continue;
      }
      Type element = cast<MemRefType>(value.getType()).getElementType();
      Value a = allocate(builder, location, element, {1, count / 2});
      Value b = allocate(builder, location, element, {1, count / 2});
      for (auto [part, offset] : {std::pair{a, int64_t(0)},
                                 std::pair{b, count / 2}})
        builder.create<LoadTileOp>(location, value, part,
            index(builder, location, offset), index(builder, location, 0),
            index(builder, location, 1), index(builder, location, 1),
            index(builder, location, count / 2));
      left.push_back(a);
      right.push_back(b);
    }
    partial = liftedCombine(left, right, count / 2);
  }
  SmallVector<Value> initial;
  for (const Field &field : fields)
    initial.push_back(isa<MemRefType>(field.initial.getType())
        ? load(builder, location, field.initial, {}) : field.initial);
  for (Value &value : partial)
    if (isa<MemRefType>(value.getType()))
      value = load(builder, location, value,
                     {index(builder, location, 0), index(builder, location, 0)});
  auto combined = combine(initial, partial, false);
  assert(succeeded(combined) && "verified scalar helper has matching arguments");
  for (auto [field, value] : llvm::zip(fields, *combined))
    store(builder, location, value, field.output, {});
  return true;
}

} // namespace intent::dsa::collective
