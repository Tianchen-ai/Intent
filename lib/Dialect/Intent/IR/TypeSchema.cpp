#include "TypeSchema.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent {

bool isCanonicalScalarType(Type type) {
  if (isa<IndexType>(type)) return true;
  if (auto integer = dyn_cast<IntegerType>(type))
    return llvm::is_contained({1u, 8u, 16u, 32u, 64u}, integer.getWidth());
  if (auto floating = dyn_cast<FloatType>(type))
    return floating.isF16() || floating.isBF16() || floating.isF32() ||
        floating.isF64() || isa<Float8E4M3FNType, Float8E5M2Type>(floating);
  return false;
}

LogicalResult verifyCanonicalType(function_ref<InFlightDiagnostic()> error, Type type) {
  if (isCanonicalScalarType(type) ||
      isa<LogicalIndexType, DomainType, RegionType, ConstexprType, EnumType>(type))
    return success();
  if (auto tensor = dyn_cast<RankedTensorType>(type)) {
    if (failed(verifyCanonicalType(error, tensor.getElementType()))) return failure();
    auto encoding = dyn_cast_or_null<TensorShapeAttr>(tensor.getEncoding());
    auto dimensions = encoding ? encoding.getDimensions() : DenseI64ArrayAttr();
    if (!dimensions || dimensions.size() != tensor.getRank())
      return error() << "tensor type requires one canonical dimension identity per axis";
    if (llvm::any_of(dimensions.asArrayRef(), [](int64_t identity) { return identity <= 0; }))
      return error() << "tensor dimension identities must be positive for every logical axis";
    return success();
  }
  if (auto view = dyn_cast<ViewType>(type)) return verifyCanonicalType(error, view.getTensor());
  if (auto buffer = dyn_cast<BufferType>(type)) return verifyCanonicalType(error, buffer.getTensor());
  if (auto tuple = dyn_cast<intent::TupleType>(type)) {
    for (Attribute attribute : tuple.getComponentTypes())
      if (failed(verifyCanonicalType(error, cast<TypeAttr>(attribute).getValue()))) return failure();
    return success();
  }
  if (auto record = dyn_cast<RecordType>(type)) {
    for (Attribute attribute : record.getFieldTypes())
      if (failed(verifyCanonicalType(error, cast<TypeAttr>(attribute).getValue()))) return failure();
    return success();
  }
  return error() << "contains non-canonical KIR type " << type;
}

LogicalResult verifyCanonicalType(Operation *owner, Type type) {
  return verifyCanonicalType([&] { return owner->emitOpError(); }, type);
}

} // namespace intent

namespace intent::detail {

bool isIntegerLike(Type type) {
  return isa<IntegerType, IndexType, LogicalIndexType>(type);
}

RankedTensorType getTensorSchema(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    return tensor;
  if (auto view = dyn_cast<ViewType>(type))
    return dyn_cast<RankedTensorType>(view.getTensor());
  if (auto buffer = dyn_cast<BufferType>(type))
    return dyn_cast<RankedTensorType>(buffer.getTensor());
  return {};
}

Type getElementType(Type type) {
  if (auto tensor = getTensorSchema(type))
    return tensor.getElementType();
  return type;
}

std::optional<unsigned> getLogicalRank(Type type) {
  if (auto tensor = getTensorSchema(type))
    return tensor.getRank();
  if (auto domain = dyn_cast<DomainType>(type))
    return domain.getRank();
  if (auto region = dyn_cast<RegionType>(type))
    return region.getRank();
  return std::nullopt;
}

std::optional<uint64_t> getCoordinateSource(Type type) {
  if (auto domain = dyn_cast<DomainType>(type))
    return domain.getOriginId();
  if (auto region = dyn_cast<RegionType>(type))
    return region.getSourceId();
  return std::nullopt;
}

std::optional<int64_t> getConstantInteger(Value value) {
  auto constant = value.getDefiningOp<ConstantOp>();
  if (!constant)
    return std::nullopt;
  if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
    return integer.getInt();
  return std::nullopt;
}

FailureOr<SmallVector<int64_t>> getIntegerArray(Operation *operation,
                                             StringRef name) {
  auto array = operation->getAttrOfType<ArrayAttr>(name);
  if (!array) {
    operation->emitOpError() << "requires array attribute '" << name << "'";
    return failure();
  }
  SmallVector<int64_t> values;
  values.reserve(array.size());
  for (Attribute attribute : array) {
    auto integer = dyn_cast<IntegerAttr>(attribute);
    if (!integer) {
      operation->emitOpError() << "attribute '" << name
                               << "' must contain integers";
      return failure();
    }
    values.push_back(integer.getInt());
  }
  return values;
}

DenseI64ArrayAttr getDimensionIDs(RankedTensorType tensor) {
  auto encoding = dyn_cast_or_null<TensorShapeAttr>(tensor.getEncoding());
  return encoding ? encoding.getDimensions() : DenseI64ArrayAttr();
}

std::optional<int64_t> getDimensionID(RankedTensorType tensor, unsigned axis) {
  if (!ShapedType::isDynamic(tensor.getDimSize(axis)))
    return std::nullopt;
  DenseI64ArrayAttr ids = getDimensionIDs(tensor);
  if (!ids || axis >= ids.size() || ids[axis] <= 0)
    return std::nullopt;
  return ids[axis];
}

bool extentValueMatchesAxis(Value extent, RankedTensorType tensor,
                           unsigned axis) {
  if (auto constant = getConstantInteger(extent))
    return !tensor.isDynamicDim(axis) && tensor.getDimSize(axis) == *constant;
  auto query = extent.getDefiningOp<DimOp>();
  auto dimension = query ? query.getDimensionAttr() : IntegerAttr();
  std::optional<int64_t> tensorDimension = getDimensionID(tensor, axis);
  return dimension && tensorDimension && dimension.getInt() == *tensorDimension;
}

bool sameDimension(RankedTensorType lhs, unsigned lhsAxis,
                   RankedTensorType rhs, unsigned rhsAxis) {
  int64_t left = lhs.getDimSize(lhsAxis);
  int64_t right = rhs.getDimSize(rhsAxis);
  if (!ShapedType::isDynamic(left) || !ShapedType::isDynamic(right))
    return left == right;
  DenseI64ArrayAttr lhsIDs = getDimensionIDs(lhs);
  DenseI64ArrayAttr rhsIDs = getDimensionIDs(rhs);
  return lhsIDs && rhsIDs && lhsIDs[lhsAxis] > 0 &&
         lhsIDs[lhsAxis] == rhsIDs[rhsAxis];
}

bool sameTensorShape(RankedTensorType lhs, RankedTensorType rhs) {
  if (!lhs || !rhs || lhs.getRank() != rhs.getRank())
    return false;
  for (unsigned axis = 0; axis < lhs.getRank(); ++axis)
    if (!sameDimension(lhs, axis, rhs, axis))
      return false;
  return true;
}

bool compatibleElementType(Type lhs, Type rhs) {
  if (lhs == rhs)
    return true;
  return (isa<LogicalIndexType>(lhs) && isa<IndexType>(rhs)) ||
         (isa<IndexType>(lhs) && isa<LogicalIndexType>(rhs));
}

bool sameDataSchema(Type lhs, Type rhs, bool compareElements) {
  auto lhsTensor = dyn_cast<RankedTensorType>(lhs);
  auto rhsTensor = dyn_cast<RankedTensorType>(rhs);
  if (static_cast<bool>(lhsTensor) != static_cast<bool>(rhsTensor))
    return false;
  if (lhsTensor && rhsTensor)
    return sameTensorShape(lhsTensor, rhsTensor) &&
           (!compareElements ||
            compatibleElementType(lhsTensor.getElementType(),
                                  rhsTensor.getElementType()));
  return !compareElements || compatibleElementType(lhs, rhs);
}

bool isBooleanData(Type type) { return getElementType(type).isInteger(1); }

bool isNumericData(Type type) {
  Type element = getElementType(type);
  return isa<IntegerType, IndexType, LogicalIndexType, FloatType>(element) &&
         !element.isInteger(1);
}

FailureOr<SmallVector<unsigned>>
verifyShapeRelation(Operation *operation, ShapeRelationAttr relation,
                    RankedTensorType result, ValueRange operands,
                    bool allowInferred) {
  ArrayAttr axes = relation ? relation.getAxes() : ArrayAttr();
  if (!axes || axes.size() != static_cast<size_t>(result.getRank())) {
    operation->emitOpError("shape relation must describe every result axis");
    return failure();
  }
  SmallVector<unsigned> dynamicOperands;
  unsigned inferredCount = 0;
  for (auto [axis, attribute] : llvm::enumerate(axes)) {
    auto entry = dyn_cast<ShapeExprAttr>(attribute);
    if (!entry) {
      operation->emitOpError("shape relation entry has an invalid kind");
      return failure();
    }
    if (entry.getKind() == 0) {
      if (result.getDimSize(axis) != entry.getPayload()) {
        operation->emitOpError(
            "static shape relation disagrees with result type");
        return failure();
      }
      continue;
    }
    auto ids = getDimensionIDs(result);
    if (entry.getDimension() <= 0 || !ids ||
        ids[axis] != entry.getDimension() ||
        !ShapedType::isDynamic(result.getDimSize(axis))) {
      operation->emitOpError(
          "dynamic/inferred shape relation must bind the result dimension identity");
      return failure();
    }
    if (entry.getKind() == 2) {
      if (!allowInferred || ++inferredCount > 1) {
        operation->emitOpError(
            "shape relation permits at most one inferred reshape extent");
        return failure();
      }
      continue;
    }
    int64_t operand = entry.getPayload();
    if (operand < 0 || operand >= operands.size() ||
        !isIntegerLike(operands[operand].getType())) {
      operation->emitOpError(
          "dynamic shape relation references an invalid extent operand");
      return failure();
    }
    dynamicOperands.push_back(static_cast<unsigned>(operand));
  }
  return dynamicOperands;
}

FailureOr<SmallVector<int64_t>> getExtentDimensions(Operation *operation,
                                                  unsigned rank) {
  FailureOr<SmallVector<int64_t>> dimensions =
      getIntegerArray(operation, "extent_dimensions");
  if (failed(dimensions) || dimensions->size() != rank) {
    operation->emitOpError(
        "logical iteration extent identities must match the source rank");
    return failure();
  }
  for (int64_t identity : *dimensions)
    if (identity < 0) {
      operation->emitOpError(
          "logical iteration extent identity must be non-negative");
      return failure();
    }
  return dimensions;
}

FailureOr<SmallVector<int64_t>> getIterationExtentDimensions(Value source) {
  Operation *definition = source.getDefiningOp();
  auto rank = getLogicalRank(source.getType());
  if (!definition || !rank)
    return failure();
  return getExtentDimensions(definition, *rank);
}

LogicalResult verifyShapeOperands(Operation *operation, ValueRange operands,
                                 ArrayRef<unsigned> dynamicOperands,
                                 unsigned firstShapeOperand) {
  if (operands.size() != firstShapeOperand + dynamicOperands.size())
    return operation->emitOpError(
        "shape extent operands do not match the canonical relation");
  for (auto [offset, position] : llvm::enumerate(dynamicOperands))
    if (position != firstShapeOperand + offset)
      return operation->emitOpError(
          "shape extent operands must be canonical and position ordered");
  return success();
}

} // namespace intent::detail
