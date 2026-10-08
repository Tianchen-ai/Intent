#include "Intent/Conversion/ScalarLowering.h"
#include "Intent/Dialect/Intent/IR/IntentOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

namespace intent {
namespace {

Type scalarType(Type type) {
  type = getElementTypeOrSelf(type);
  return isa<LogicalIndexType>(type) ? IndexType::get(type.getContext()) : type;
}

bool isUnsigned(Type type) {
  auto integer = dyn_cast<IntegerType>(type);
  return integer && (integer.isUnsigned() || integer.getWidth() == 1);
}

Type integerStorageType(Type type) {
  if (auto integer = dyn_cast<IntegerType>(type))
    return IntegerType::get(type.getContext(), integer.getWidth());
  if (auto memory = dyn_cast<MemRefType>(type))
    return MemRefType::get(memory.getShape(),
                          integerStorageType(memory.getElementType()),
                          memory.getLayout(), memory.getMemorySpace());
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    return RankedTensorType::get(tensor.getShape(),
                                integerStorageType(tensor.getElementType()),
                                tensor.getEncoding());
  if (auto function = dyn_cast<FunctionType>(type)) {
    SmallVector<Type> inputs, outputs;
    for (Type input : function.getInputs())
      inputs.push_back(integerStorageType(input));
    for (Type output : function.getResults())
      outputs.push_back(integerStorageType(output));
    return FunctionType::get(type.getContext(), inputs, outputs);
  }
  if (auto vector = dyn_cast<VectorType>(type))
    return VectorType::get(vector.getShape(),
                           integerStorageType(vector.getElementType()),
                           vector.getScalableDims());
  return type;
}

} // namespace

void realizeIntegerStorage(ModuleOp module) {
  module.walk([&](Operation *operation) {
    for (Value result : operation->getResults())
      result.setType(integerStorageType(result.getType()));
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          argument.setType(integerStorageType(argument.getType()));
    // Public interfaces and other nested semantic attributes retain their
    // logical types. Only executable type/constant carriers are rewritten.
    for (NamedAttribute attribute : llvm::to_vector(operation->getAttrs())) {
      if (auto type = dyn_cast<TypeAttr>(attribute.getValue()))
        operation->setAttr(
            attribute.getName(),
            TypeAttr::get(integerStorageType(type.getValue())));
      else if (auto integer = dyn_cast<IntegerAttr>(attribute.getValue()))
        operation->setAttr(
            attribute.getName(),
            IntegerAttr::get(integerStorageType(integer.getType()),
                             integer.getValue()));
    }
  });
}

FailureOr<Value> castScalarValue(OpBuilder &builder, Location loc, Value value,
                                Type destination, Type logicalSource) {
  if (!value)
    return failure();
  Type source = scalarType(logicalSource ? logicalSource : value.getType());
  destination = scalarType(destination);
  if (value.getType() == destination)
    return value;
  if (destination.isInteger(1)) {
    if (isa<FloatType>(source))
      return Value(builder.create<arith::CmpFOp>(
          loc, arith::CmpFPredicate::UNE, value,
          builder.create<arith::ConstantOp>(
              loc, builder.getFloatAttr(value.getType(), 0.0))));
    if (source.isIntOrIndex())
      return Value(builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ne, value,
          builder.create<arith::ConstantOp>(
              loc, builder.getIntegerAttr(value.getType(), 0))));
    return failure();
  }
  if (source.isIndex() && isa<IntegerType>(destination))
    return isUnsigned(destination)
        ? Value(builder.create<arith::IndexCastUIOp>(loc, destination, value))
        : Value(builder.create<arith::IndexCastOp>(loc, destination, value));
  if (isa<IntegerType>(source) && destination.isIndex())
    return isUnsigned(source)
        ? Value(builder.create<arith::IndexCastUIOp>(loc, destination, value))
        : Value(builder.create<arith::IndexCastOp>(loc, destination, value));
  if (source.isIntOrIndex() && isa<FloatType>(destination)) {
    if (source.isIndex())
      value = builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), value);
    return isUnsigned(source)
        ? Value(builder.create<arith::UIToFPOp>(loc, destination, value))
        : Value(builder.create<arith::SIToFPOp>(loc, destination, value));
  }
  if (isa<FloatType>(source) && destination.isIntOrIndex()) {
    Type carrier = destination.isIndex() ? builder.getI64Type() : destination;
    Value result = isUnsigned(destination)
        ? Value(builder.create<arith::FPToUIOp>(loc, carrier, value))
        : Value(builder.create<arith::FPToSIOp>(loc, carrier, value));
    return destination.isIndex()
        ? Value(builder.create<arith::IndexCastOp>(loc, destination, result)) : result;
  }
  if (isa<FloatType>(source) && isa<FloatType>(destination)) {
    if (source.getIntOrFloatBitWidth() < destination.getIntOrFloatBitWidth())
      return Value(builder.create<arith::ExtFOp>(loc, destination, value));
    if (source.getIntOrFloatBitWidth() > destination.getIntOrFloatBitWidth())
      return Value(builder.create<arith::TruncFOp>(loc, destination, value));
    Value widened = builder.create<arith::ExtFOp>(loc, builder.getF32Type(), value);
    return Value(builder.create<arith::TruncFOp>(loc, destination, widened));
  }
  if (isa<IntegerType>(source) && isa<IntegerType>(destination)) {
    unsigned from = source.getIntOrFloatBitWidth();
    unsigned to = destination.getIntOrFloatBitWidth();
    if (from == to)
      return value;
    if (from > to)
      return Value(builder.create<arith::TruncIOp>(loc, destination, value));
    return isUnsigned(source)
        ? Value(builder.create<arith::ExtUIOp>(loc, destination, value))
        : Value(builder.create<arith::ExtSIOp>(loc, destination, value));
  }
  return failure();
}

FailureOr<Value> lowerScalarOperation(Operation *operation, ValueRange operands,
                                     OpBuilder &builder) {
  Location loc = operation->getLoc();
  if (auto constant = dyn_cast<ConstantOp>(operation)) {
    Type type = scalarType(constant.getResult().getType());
    if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
      return Value(builder.create<arith::ConstantOp>(
          loc, builder.getIntegerAttr(type, integer.getValue())));
    if (auto floating = dyn_cast<FloatAttr>(constant.getValue())) {
      auto number = floating.getValue();
      bool losesInformation;
      number.convert(cast<FloatType>(type).getFloatSemantics(),
                     APFloat::rmNearestTiesToEven, &losesInformation);
      return Value(builder.create<arith::ConstantOp>(loc, FloatAttr::get(type, number)));
    }
  } else if (auto cast = dyn_cast<CastOp>(operation)) {
    if (cast.getRounding() && *cast.getRounding() != 0)
      return cast.emitError("unsupported scalar cast rounding"), failure();
    auto result = castScalarValue(builder, loc, operands[0],
                                  cast.getResult().getType(),
                                  cast.getInput().getType());
    if (succeeded(result))
      return result;
  } else if (isa<BitcastOp>(operation)) {
    return Value(builder.create<arith::BitcastOp>(
        loc, scalarType(operation->getResult(0).getType()), operands[0]));
  } else if (auto binary = dyn_cast<BinaryOp>(operation)) {
    if (binary.getApproximate() || binary.getFlushToZero())
      return binary.emitError(
                 "non-default math requires provider numerical lowering"),
             failure();
    Value lhs = operands[0], rhs = operands[1];
    Type type = scalarType(binary.getLhs().getType());
    bool fp = isa<FloatType>(type);
    bool unsignedInteger = isUnsigned(type);
    switch (binary.getOperatorKind()) {
    case BinaryOperator::Add:
      return fp ? Value(builder.create<arith::AddFOp>(loc, lhs, rhs))
                : Value(builder.createOrFold<arith::AddIOp>(loc, lhs, rhs));
    case BinaryOperator::Subtract:
      return fp ? Value(builder.create<arith::SubFOp>(loc, lhs, rhs))
                : Value(builder.createOrFold<arith::SubIOp>(loc, lhs, rhs));
    case BinaryOperator::Multiply:
      return fp ? Value(builder.create<arith::MulFOp>(loc, lhs, rhs))
                : Value(builder.createOrFold<arith::MulIOp>(loc, lhs, rhs));
    case BinaryOperator::TrueDivide:
      return Value(builder.create<arith::DivFOp>(loc, lhs, rhs));
    case BinaryOperator::FloorDivide:
      return unsignedInteger
                 ? Value(builder.create<arith::DivUIOp>(loc, lhs, rhs))
                 : Value(builder.create<arith::FloorDivSIOp>(loc, lhs, rhs));
    case BinaryOperator::Remainder:
      if (unsignedInteger)
        return Value(builder.create<arith::RemUIOp>(loc, lhs, rhs));
      return Value(builder.createOrFold<arith::SubIOp>(
          loc, lhs, builder.createOrFold<arith::MulIOp>(
                        loc, builder.create<arith::FloorDivSIOp>(loc, lhs, rhs),
                        rhs)));
    case BinaryOperator::Maximum:
      if (fp)
        return Value(builder.create<arith::MaximumFOp>(loc, lhs, rhs));
      return unsignedInteger ? Value(builder.create<arith::MaxUIOp>(loc, lhs, rhs))
                             : Value(builder.create<arith::MaxSIOp>(loc, lhs, rhs));
    case BinaryOperator::Minimum:
      if (fp)
        return Value(builder.create<arith::MinimumFOp>(loc, lhs, rhs));
      return unsignedInteger ? Value(builder.create<arith::MinUIOp>(loc, lhs, rhs))
                             : Value(builder.create<arith::MinSIOp>(loc, lhs, rhs));
    case BinaryOperator::MaximumNum:
      return Value(builder.create<arith::MaxNumFOp>(loc, lhs, rhs));
    case BinaryOperator::MinimumNum:
      return Value(builder.create<arith::MinNumFOp>(loc, lhs, rhs));
    case BinaryOperator::Power:
      if (fp)
        return Value(builder.create<math::PowFOp>(loc, lhs, rhs));
      break;
    case BinaryOperator::LogicalAnd:
    case BinaryOperator::BitwiseAnd:
      return Value(builder.create<arith::AndIOp>(loc, lhs, rhs));
    case BinaryOperator::LogicalOr:
    case BinaryOperator::BitwiseOr:
      return Value(builder.create<arith::OrIOp>(loc, lhs, rhs));
    case BinaryOperator::BitwiseXor:
      return Value(builder.create<arith::XOrIOp>(loc, lhs, rhs));
    case BinaryOperator::LeftShift:
      return Value(builder.create<arith::ShLIOp>(loc, lhs, rhs));
    case BinaryOperator::RightShift:
      return unsignedInteger ? Value(builder.create<arith::ShRUIOp>(loc, lhs, rhs))
                             : Value(builder.create<arith::ShRSIOp>(loc, lhs, rhs));
    }
  } else if (auto unary = dyn_cast<UnaryOp>(operation)) {
    if (unary.getApproximate() || unary.getFlushToZero())
      return unary.emitError(
                 "non-default math requires provider numerical lowering"),
             failure();
    Value input = operands[0];
    switch (unary.getOperatorKind()) {
    case UnaryOperator::Rsqrt:
      return Value(builder.create<math::RsqrtOp>(loc, input));
    case UnaryOperator::Sqrt:
      return Value(builder.create<math::SqrtOp>(loc, input));
    case UnaryOperator::Exp:
      return Value(builder.create<math::ExpOp>(loc, input));
    case UnaryOperator::Exp2:
      return Value(builder.create<math::Exp2Op>(loc, input));
    case UnaryOperator::Log:
      return Value(builder.create<math::LogOp>(loc, input));
    case UnaryOperator::Sin:
      return Value(builder.create<math::SinOp>(loc, input));
    case UnaryOperator::Cos:
      return Value(builder.create<math::CosOp>(loc, input));
    case UnaryOperator::Floor:
      return Value(builder.create<math::FloorOp>(loc, input));
    case UnaryOperator::Erf:
      return Value(builder.create<math::ErfOp>(loc, input));
    case UnaryOperator::Tanh:
      return Value(builder.create<math::TanhOp>(loc, input));
    case UnaryOperator::Abs:
      if (isa<FloatType>(input.getType()))
        return Value(builder.create<math::AbsFOp>(loc, input));
      if (isUnsigned(scalarType(unary.getInput().getType())))
        return input;
      return Value(builder.create<math::AbsIOp>(loc, input));
    case UnaryOperator::Negate:
      if (isa<FloatType>(input.getType()))
        return Value(builder.create<arith::NegFOp>(loc, input));
      return Value(builder.create<arith::SubIOp>(
          loc, builder.create<arith::ConstantOp>(
                   loc, builder.getIntegerAttr(input.getType(), 0)),
          input));
    case UnaryOperator::Not:
      return Value(builder.create<arith::XOrIOp>(
          loc, input,
          builder.create<arith::ConstantOp>(loc, builder.getBoolAttr(true))));
    default:
      break;
    }
  } else if (auto compare = dyn_cast<CompareOp>(operation)) {
    static constexpr arith::CmpFPredicate floating[] = {
        arith::CmpFPredicate::OEQ, arith::CmpFPredicate::UNE,
        arith::CmpFPredicate::OLT, arith::CmpFPredicate::OLE,
        arith::CmpFPredicate::OGT, arith::CmpFPredicate::OGE};
    static constexpr arith::CmpIPredicate signedPredicates[] = {
        arith::CmpIPredicate::eq, arith::CmpIPredicate::ne,
        arith::CmpIPredicate::slt, arith::CmpIPredicate::sle,
        arith::CmpIPredicate::sgt, arith::CmpIPredicate::sge};
    static constexpr arith::CmpIPredicate unsignedPredicates[] = {
        arith::CmpIPredicate::eq, arith::CmpIPredicate::ne,
        arith::CmpIPredicate::ult, arith::CmpIPredicate::ule,
        arith::CmpIPredicate::ugt, arith::CmpIPredicate::uge};
    Type type = scalarType(compare.getLhs().getType());
    unsigned predicate = static_cast<unsigned>(compare.getPredicate());
    return isa<FloatType>(type)
        ? Value(builder.create<arith::CmpFOp>(loc, floating[predicate],
                                            operands[0], operands[1]))
        : Value(builder.create<arith::CmpIOp>(
              loc, isUnsigned(type) ? unsignedPredicates[predicate]
                                   : signedPredicates[predicate],
              operands[0], operands[1]));
  } else if (isa<SelectOp>(operation)) {
    return Value(builder.create<arith::SelectOp>(loc, operands[0], operands[1],
                                              operands[2]));
  } else if (isa<MaskOp>(operation)) {
    return Value(builder.create<arith::SelectOp>(loc, operands[1], operands[0],
                                              operands[2]));
  }
  return operation->emitError("operation has no common scalar numerical lowering"), failure();
}

} // namespace intent
