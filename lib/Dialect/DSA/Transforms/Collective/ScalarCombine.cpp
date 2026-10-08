#include "CollectiveLowering.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace intent::dsa::collective {

std::optional<BinaryOperator> binaryKind(Operation *operation) {
  return llvm::TypeSwitch<Operation *, std::optional<BinaryOperator>>(operation)
      .Case<arith::AddFOp>([](auto) { return BinaryOperator::Add; })
      .Case<arith::SubFOp>([](auto) { return BinaryOperator::Subtract; })
      .Case<arith::MulFOp>([](auto) { return BinaryOperator::Multiply; })
      .Case<arith::DivFOp>([](auto) { return BinaryOperator::TrueDivide; })
      .Case<arith::MaximumFOp>([](auto) { return BinaryOperator::Maximum; })
      .Case<arith::MinimumFOp>([](auto) { return BinaryOperator::Minimum; })
      .Case<arith::MaxNumFOp>([](auto) { return BinaryOperator::MaximumNum; })
      .Case<arith::MinNumFOp>([](auto) { return BinaryOperator::MinimumNum; })
      .Case<arith::AndIOp>([](auto op) -> std::optional<BinaryOperator> {
        return op.getType().isInteger(1)
                   ? std::optional(BinaryOperator::LogicalAnd) : std::nullopt;
      })
      .Case<arith::OrIOp>([](auto op) -> std::optional<BinaryOperator> {
        return op.getType().isInteger(1)
                   ? std::optional(BinaryOperator::LogicalOr) : std::nullopt;
      })
      .Default([](Operation *) -> std::optional<BinaryOperator> { return std::nullopt; });
}

namespace {

std::optional<UnaryOperator> unaryKind(Operation *operation) {
  return llvm::TypeSwitch<Operation *, std::optional<UnaryOperator>>(operation)
      .Case<arith::NegFOp>([](auto) { return UnaryOperator::Negate; })
      .Case<math::AbsFOp>([](auto) { return UnaryOperator::Abs; })
      .Case<math::ExpOp>([](auto) { return UnaryOperator::Exp; })
      .Case<math::Exp2Op>([](auto) { return UnaryOperator::Exp2; })
      .Case<math::LogOp>([](auto) { return UnaryOperator::Log; })
      .Case<math::SqrtOp>([](auto) { return UnaryOperator::Sqrt; })
      .Case<math::TanhOp>([](auto) { return UnaryOperator::Tanh; })
      .Default([](Operation *) -> std::optional<UnaryOperator> { return std::nullopt; });
}

std::optional<ComparePredicate> comparison(Operation *operation) {
  auto compare = dyn_cast<arith::CmpFOp>(operation);
  if (!compare)
    return std::nullopt;
  switch (compare.getPredicate()) {
  case arith::CmpFPredicate::OEQ: return ComparePredicate::Eq;
  case arith::CmpFPredicate::UNE: return ComparePredicate::Ne;
  case arith::CmpFPredicate::OLT: return ComparePredicate::Lt;
  case arith::CmpFPredicate::OLE: return ComparePredicate::Le;
  case arith::CmpFPredicate::OGT: return ComparePredicate::Gt;
  case arith::CmpFPredicate::OGE: return ComparePredicate::Ge;
  default: return std::nullopt;
  }
}

bool hasTileMapping(Operation *operation) {
  if (operation->getNumResults() != 1 || operation->getNumRegions() ||
      !isMemoryEffectFree(operation))
    return false;
  Type type = operation->getResult(0).getType();
  if (binaryKind(operation))
    return type.isF32();
  if (unaryKind(operation))
    return type.isF32();
  if (comparison(operation))
    return operation->getOperand(0).getType().isF32();
  if (isa<arith::SelectOp>(operation))
    return type.isF32() || type.isF16() || type.isBF16() || type.isInteger(1);
  if (auto rounding = operation->getAttrOfType<arith::RoundingModeAttr>(
          "roundingmode"))
    if (rounding.getValue() != arith::RoundingMode::to_nearest_even)
      return false;
  return isa<arith::ExtFOp, arith::TruncFOp>(operation) &&
         (type.isF16() || type.isBF16() || type.isF32());
}

} // namespace

bool canLiftScalarCombine(Block &combine) {
  Operation *parent = combine.getParentOp();
  unsigned fields = isa<SliceReduceOp>(parent)
      ? cast<SliceReduceOp>(parent).getSources().size()
      : cast<ScanOp>(parent).getSources().size();
  DenseSet<Value> lanes;
  for (Value argument : combine.getArguments().take_front(fields * 2))
    lanes.insert(argument);
  for (Operation &operation : combine.without_terminator()) {
    if (operation.getNumRegions() || !isMemoryEffectFree(&operation))
      return false;
    bool varying = llvm::any_of(operation.getOperands(),
                                [&](Value value) { return lanes.contains(value); });
    if (!varying)
      continue;
    if (!hasTileMapping(&operation))
      return false;
    lanes.insert(operation.getResults().begin(), operation.getResults().end());
  }
  return true;
}

FailureOr<SmallVector<Value>> liftScalarCombine(
    OpBuilder &builder, Location location, Block &combine, ValueRange arguments,
    ArrayRef<int64_t> shape) {
  IRMapping mapping;
  Operation *parent = combine.getParentOp();
  unsigned fields = isa<SliceReduceOp>(parent)
      ? cast<SliceReduceOp>(parent).getSources().size()
      : cast<ScanOp>(parent).getSources().size();
  DenseSet<Value> lanes;
  for (Value formal : combine.getArguments().take_front(fields * 2))
    lanes.insert(formal);
  for (auto [formal, value] : llvm::zip(combine.getArguments(), arguments))
    mapping.map(formal, value);
  for (Operation &operation : combine.without_terminator()) {
    SmallVector<Value> operands;
    bool varying = false;
    bool laneDependent = false;
    for (Value original : operation.getOperands()) {
      Value value = mapping.lookupOrDefault(original);
      laneDependent |= lanes.contains(original);
      varying |= lanes.contains(original) && isa<MemRefType>(value.getType());
      operands.push_back(value);
    }
    if (laneDependent)
      lanes.insert(operation.getResults().begin(), operation.getResults().end());
    if (!varying) {
      builder.clone(operation, mapping);
      continue;
    }
    if (!hasTileMapping(&operation))
      return operation.emitError("scalar combine has no selected DSA tile mapping"),
             failure();
    auto tile = [&](Value value) {
      if (isa<MemRefType>(value.getType()))
        return physicalView(builder, location, value);
      Value storage = allocate(builder, location, value.getType(), shape);
      fill(builder, location, storage, value);
      return physicalView(builder, location, storage);
    };
    Value result = allocate(builder, location,
                             operation.getResult(0).getType(), shape);
    Value destination = physicalView(builder, location, result);
    auto no = builder.getBoolAttr(false);
    if (auto kind = binaryKind(&operation))
      builder.create<BinaryOp>(location, tile(operands[0]), tile(operands[1]),
          destination, BinaryOperatorAttr::get(builder.getContext(), *kind),
          no, no, Value());
    else if (auto kind = unaryKind(&operation))
      builder.create<UnaryOp>(location, tile(operands[0]), destination,
          UnaryOperatorAttr::get(builder.getContext(), *kind), no, no, Value());
    else if (auto predicate = comparison(&operation))
      builder.create<CompareOp>(location, tile(operands[0]), tile(operands[1]),
          destination, ComparePredicateAttr::get(builder.getContext(), *predicate),
          Value());
    else if (isa<arith::SelectOp>(operation))
      builder.create<SelectOp>(location, tile(operands[0]), tile(operands[1]),
                                tile(operands[2]), destination, Value());
    else
      builder.create<CastOp>(location, tile(operands[0]), destination);
    mapping.map(operation.getResult(0), result);
  }
  SmallVector<Value> results;
  for (Value value : combine.getTerminator()->getOperands())
    results.push_back(mapping.lookupOrDefault(value));
  return results;
}

} // namespace intent::dsa::collective
