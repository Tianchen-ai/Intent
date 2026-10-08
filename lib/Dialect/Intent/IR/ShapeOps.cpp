#include "TypeSchema.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace intent::detail;

namespace intent {

namespace {

FailureOr<RankedTensorType>
verifyShapeOperationInputs(Operation *operation, ShapeRelationAttr relation,
                           ValueRange inputs, bool allowInferred) {
  auto result = dyn_cast<RankedTensorType>(operation->getResult(0).getType());
  if (!result || operation->getNumOperands() == 0)
    return operation->emitOpError(
        "shape operation requires a source/fill and ranked tensor result");
  FailureOr<SmallVector<unsigned>> dynamicOperands =
      verifyShapeRelation(operation, relation, result, inputs, allowInferred);
  if (failed(dynamicOperands) ||
      failed(verifyShapeOperands(operation, inputs, *dynamicOperands, 1)))
    return failure();
  return result;
}

} // namespace

LogicalResult FullOp::verify() {
  Operation *operation = getOperation();
  auto schema =
      verifyShapeOperationInputs(operation, getShape(), getInputs(), false);
  if (failed(schema))
    return failure();
  auto result = *schema;
  Type sourceType = getInputs().front().getType();
  if (!compatibleElementType(sourceType, result.getElementType()) ||
      !isa<IntegerType, IndexType, FloatType, LogicalIndexType>(sourceType))
    return operation->emitOpError(
        "full fill/result element schema is invalid");
  return success();
}

LogicalResult ReshapeOp::verify() {
  Operation *operation = getOperation();
  auto schema =
      verifyShapeOperationInputs(operation, getShape(), getInputs(), true);
  if (failed(schema))
    return failure();
  auto result = *schema;
  Type sourceType = getInputs().front().getType();
  auto source = dyn_cast<RankedTensorType>(sourceType);
  if (!source || source.getElementType() != result.getElementType())
    return operation->emitOpError(
        "reshape must preserve ranked tensor element type");
  if (source.hasStaticShape() && result.hasStaticShape() &&
      source.getNumElements() != result.getNumElements())
    return operation->emitOpError(
        "reshape must preserve logical element count");
  return success();
}

LogicalResult BroadcastOp::verify() {
  Operation *operation = getOperation();
  auto schema =
      verifyShapeOperationInputs(operation, getShape(), getInputs(), false);
  if (failed(schema))
    return failure();
  auto result = *schema;
  Type sourceType = getInputs().front().getType();
  auto source = dyn_cast<RankedTensorType>(sourceType);
  if (!compatibleElementType(getElementType(sourceType),
                             result.getElementType()))
    return operation->emitOpError("broadcast must preserve element type");
  if (!source)
    return isa<IntegerType, IndexType, FloatType, LogicalIndexType>(sourceType)
               ? success()
               : operation->emitOpError(
                     "broadcast source must be scalar or ranked tensor");
  if (source.getRank() > result.getRank())
    return operation->emitOpError("broadcast cannot reduce rank");
  unsigned offset = result.getRank() - source.getRank();
  for (unsigned axis = 0; axis < source.getRank(); ++axis)
    if (source.getDimSize(axis) != 1 &&
        !sameDimension(source, axis, result, axis + offset))
      return operation->emitOpError(
          "broadcast axis is neither size one nor source-identical");
  return success();
}

LogicalResult DimOp::verify() {
  Operation *operation = getOperation();
  auto axis = getAxisAttr();
  auto dimension = getDimensionAttr();
  auto rank = getLogicalRank(getSource().getType());
  if (!axis || axis.getInt() < 0 || !dimension || dimension.getInt() <= 0 ||
      !rank || axis.getInt() >= *rank)
    return operation->emitOpError("dimension query axis is invalid");
  Type sourceType = getSource().getType();
  if (auto source = getTensorSchema(sourceType)) {
    auto ids = getDimensionIDs(source);
    if (!source.isDynamicDim(axis.getInt()) || !ids ||
        ids[axis.getInt()] != dimension.getInt())
      return operation->emitOpError(
          "dimension query identity does not match its tensor source axis");
  } else {
    FailureOr<SmallVector<int64_t>> dimensions =
        getIterationExtentDimensions(getSource());
    if (failed(dimensions) ||
        (*dimensions)[axis.getInt()] != dimension.getInt())
      return operation->emitOpError(
          "dimension query identity does not match its iteration source axis");
  }
  return success();
}

LogicalResult DomainOp::verify() {
  Operation *operation = getOperation();
  auto result = dyn_cast<DomainType>(getResult().getType());
  if (!result || result.getRank() != 1 ||
      (getBounds().size() != 2 && getBounds().size() != 3))
    return operation->emitOpError("domain requires start/stop[/step] and rank one");
  if (failed(detail::getExtentDimensions(operation, 1)))
    return failure();
  for (Value bound : getBounds())
    if (!isIntegerLike(bound.getType()))
      return operation->emitOpError("domain bound must be integer/index typed");
  return success();
}

LogicalResult DomainProductOp::verify() {
  Operation *operation = getOperation();
  auto result = dyn_cast<DomainType>(getResult().getType());
  unsigned rank = 0;
  SmallVector<int64_t> expectedDimensions;
  for (Value operand : getDomains()) {
    auto domain = dyn_cast<DomainType>(operand.getType());
    if (!domain)
      return operation->emitOpError("domain product requires domains");
    rank += domain.getRank();
    FailureOr<SmallVector<int64_t>> dimensions =
        getIterationExtentDimensions(operand);
    if (failed(dimensions))
      return operation->emitOpError(
          "domain product cannot recover a source extent identity");
    expectedDimensions.append(*dimensions);
  }
  if (!result || getDomains().size() == 0 || result.getRank() != rank)
    return operation->emitOpError("domain product result rank is invalid");
  FailureOr<SmallVector<int64_t>> dimensions =
      detail::getExtentDimensions(operation, rank);
  if (failed(dimensions) || *dimensions != expectedDimensions)
    return operation->emitOpError(
        "domain product extent identities must concatenate its operands");
  return success();
}

LogicalResult SubregionOp::verify() {
  Operation *operation = getOperation();
  Type source = getInputs().front().getType();
  auto result = dyn_cast<RegionType>(getResult().getType());
  auto hasStart = getHasStartAttr();
  auto hasStop = getHasStopAttr();
  auto sourceID = getCoordinateSource(source);
  auto rank = getLogicalRank(source);
  unsigned expected = 1 + (hasStart && hasStart.getValue()) +
                      (hasStop && hasStop.getValue());
  if (!result || !hasStart || !hasStop || !sourceID || !rank || *rank != 1 ||
      getInputs().size() != expected || result.getSourceId() != *sourceID ||
      result.getRank() != *rank)
    return operation->emitOpError("subregion source/bound schema is invalid");
  FailureOr<SmallVector<int64_t>> dimensions =
      detail::getExtentDimensions(operation, *rank);
  if (failed(dimensions))
    return failure();
  if (!hasStart.getValue() && !hasStop.getValue()) {
    FailureOr<SmallVector<int64_t>> sourceDimensions =
        getIterationExtentDimensions(getInputs().front());
    if (failed(sourceDimensions) || *dimensions != *sourceDimensions)
      return operation->emitOpError(
          "identity subregion must preserve source extent identities");
  } else if (llvm::any_of(*dimensions,
                         [](int64_t identity) { return identity <= 0; })) {
    return operation->emitOpError(
        "bounded subregion requires fresh dynamic extent identities");
  }
  for (Value bound : getInputs().drop_front())
    if (!isIntegerLike(bound.getType()))
      return operation->emitOpError("subregion bound must be integer/index typed");
  return success();
}

LogicalResult IndicesOp::verify() {
  Operation *operation = getOperation();
  auto sourceRank = getLogicalRank(getSource().getType());
  auto result = dyn_cast<RankedTensorType>(getResult().getType());
  auto axis = getTensorAxisAttr();
  bool tensorSource = isa<RankedTensorType>(getSource().getType());
  if (!sourceRank || !result || !result.getElementType().isIndex() ||
      result.getRank() != *sourceRank || tensorSource != static_cast<bool>(axis) ||
      (axis && (axis.getInt() < 0 || axis.getInt() >= *sourceRank)))
    return operation->emitOpError("indices result/source schema is invalid");
  if (auto source = dyn_cast<RankedTensorType>(getSource().getType());
      source && !sameTensorShape(source, result))
    return operation->emitOpError(
        "tensor indices must preserve the source logical shape");
  if (!tensorSource) {
    FailureOr<SmallVector<int64_t>> dimensions =
        getIterationExtentDimensions(getSource());
    auto resultIDs = getDimensionIDs(result);
    if (failed(dimensions) ||
        dimensions->size() != static_cast<size_t>(result.getRank()))
      return operation->emitOpError(
          "indices cannot recover its source extent identities");
    for (auto [resultAxis, identity] : llvm::enumerate(*dimensions)) {
      if (identity <= 0 || !resultIDs || resultIDs[resultAxis] != identity)
        return operation->emitOpError(
            "indices result extent identity differs from its source");
    }
  }
  return success();
}

LogicalResult RegionEndOp::verify() {
  Operation *operation = getOperation();
  auto rank = getLogicalRank(getSource().getType());
  return rank && *rank == 1
             ? success()
             : operation->emitOpError("region_end requires rank one");
}

LogicalResult TransposeOp::verify() {
  Operation *operation = getOperation();
  auto input = getTensorSchema(getInput().getType());
  auto result = getTensorSchema(getResult().getType());
  FailureOr<SmallVector<int64_t>> permutation =
      getIntegerArray(operation, "permutation");
  if (!input || !result || failed(permutation) ||
      permutation->size() != static_cast<size_t>(input.getRank()) ||
      result.getRank() != input.getRank())
    return operation->emitOpError("transpose permutation/rank is invalid");
  llvm::DenseSet<int64_t> axes;
  for (auto [resultAxis, sourceAxis] : llvm::enumerate(*permutation))
    if (sourceAxis < 0 || sourceAxis >= input.getRank() ||
        !axes.insert(sourceAxis).second ||
        !sameDimension(input, sourceAxis, result, resultAxis))
      return operation->emitOpError(
          "transpose permutation is not a typed bijection");
  return success();
}

LogicalResult JoinOp::verify() {
  Operation *operation = getOperation();
  auto lhs = getTensorSchema(getLhs().getType());
  auto rhs = getTensorSchema(getRhs().getType());
  auto result = getTensorSchema(getResult().getType());
  if (!lhs || !rhs || !result || !sameTensorShape(lhs, rhs) ||
      result.getRank() != lhs.getRank() + 1 ||
      result.getElementType() != lhs.getElementType() ||
      result.getDimSize(result.getRank() - 1) != 2)
    return operation->emitOpError(
        "join requires equal inputs and trailing extent two");
  for (unsigned axis = 0; axis < lhs.getRank(); ++axis)
    if (!sameDimension(lhs, axis, result, axis))
      return operation->emitOpError(
          "join result prefix lost source dimension identity");
  return success();
}

} // namespace intent
