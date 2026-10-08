#include "TypeSchema.h"
#include "RegionVerification.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <limits>

#include "Intent/Dialect/Intent/IR/IndexedAccessOpInterface.cpp.inc"

using namespace mlir;
using namespace intent::detail;

namespace intent {

namespace {

struct IndexRelationInfo {
  unsigned resultRank = 0;
  DenseI64ArrayAttr resultDimensions;
  llvm::DenseSet<unsigned> operandPositions;
};

FailureOr<IndexRelationInfo> verifyIndexRelation(Operation *operation) {
  auto access = cast<IndexedAccessOpInterface>(operation);
  auto relation = access.getIndexRelation();
  if (!relation)
    return operation->emitOpError("requires a typed index relation");
  ArrayAttr terms = relation.getTerms();
  auto indices = access.getIndexOperands();
  auto ranked = getTensorSchema(access.getAccessSource().getType());
  if (!ranked || relation.getSourceRank() != ranked.getRank())
    return operation->emitOpError(
        "index relation source rank does not match its source operand");

  unsigned consumed = 0;
  unsigned basicResultRank = 0;
  unsigned advancedResultRank = 0;
  SmallVector<std::pair<RankedTensorType, unsigned>> advancedShape;
  IndexRelationInfo result;
  result.resultDimensions = relation.getResultDimensions();
  for (Attribute attribute : terms) {
    auto term = dyn_cast<IndexTermAttr>(attribute);
    if (!term)
      return operation->emitOpError("index relation term has an invalid schema");
    int64_t termKind = term.getKind();
    ArrayRef<int64_t> operands = term.getOperandPositions().asArrayRef();
    ArrayRef<int64_t> values = term.getStaticValues().asArrayRef();
    if (termKind != 1)
      ++consumed;
    for (int64_t position : operands) {
      if (position == -1)
        continue;
      if (position < 0 || position >= indices.size())
        return operation->emitOpError(
            "index relation references an invalid indices-segment position");
      Type type = indices[position].getType();
      auto tensor = dyn_cast<RankedTensorType>(type);
      if (termKind != 4 && !isIntegerLike(type) &&
          (!tensor || !isIntegerLike(tensor.getElementType())))
        return operation->emitOpError(
            "dynamic index relation operand must be integer/index typed");
      result.operandPositions.insert(static_cast<unsigned>(position));
    }
    if ((termKind == 0 || termKind == 1) &&
        (!operands.empty() || !values.empty()))
      return operation->emitOpError(
          "full-slice/new-axis terms cannot carry payload");
    if (termKind == 0 || termKind == 1 || termKind == 5)
      ++basicResultRank;
    if (termKind == 2 &&
        (!operands.empty() || values.size() != 1 ||
         values[0] == std::numeric_limits<int64_t>::min()))
      return operation->emitOpError(
          "static-index term requires one integer literal");
    if ((termKind == 3 || termKind == 4) &&
        (operands.size() != 1 || operands[0] < 0 || !values.empty()))
      return operation->emitOpError(
          "value/region index term requires one dynamic operand");
    if (termKind == 4) {
      int64_t position = operands[0];
      if (!isa<DomainType, RegionType>(indices[position].getType()))
        return operation->emitOpError(
            "region index term must reference a domain or subregion operand");
      auto rank = getLogicalRank(indices[position].getType());
      if (!rank)
        return operation->emitOpError("region index term has no logical rank");
      basicResultRank += *rank;
    }
    if (termKind == 3) {
      int64_t position = operands[0];
      if (auto tensor =
              dyn_cast<RankedTensorType>(indices[position].getType())) {
        if (advancedShape.empty()) {
          advancedShape.reserve(tensor.getRank());
          for (unsigned axis = 0; axis < tensor.getRank(); ++axis)
            advancedShape.emplace_back(tensor, axis);
        } else {
          unsigned mergedRank =
              std::max<unsigned>(advancedShape.size(), tensor.getRank());
          SmallVector<std::pair<RankedTensorType, unsigned>> merged(mergedRank);
          for (unsigned trailing = 0; trailing < mergedRank; ++trailing) {
            bool hasLeft = trailing < advancedShape.size();
            bool hasRight = trailing < static_cast<unsigned>(tensor.getRank());
            auto left = hasLeft
                            ? advancedShape[advancedShape.size() - 1 - trailing]
                            : std::pair<RankedTensorType, unsigned>();
            unsigned rightAxis = hasRight ? tensor.getRank() - 1 - trailing : 0;
            bool leftOne = hasLeft && left.first.getDimSize(left.second) == 1;
            bool rightOne = hasRight && tensor.getDimSize(rightAxis) == 1;
            if (hasLeft && hasRight && !leftOne && !rightOne &&
                !sameDimension(left.first, left.second, tensor, rightAxis))
              return operation->emitOpError(
                  "tensor index operands have incompatible broadcast extents");
            if (!hasLeft)
              merged[mergedRank - 1 - trailing] =
                  std::make_pair(tensor, rightAxis);
            else if (!hasRight)
              merged[mergedRank - 1 - trailing] = left;
            else
              merged[mergedRank - 1 - trailing] =
                  leftOne ? std::make_pair(tensor, rightAxis) : left;
          }
          advancedShape = std::move(merged);
        }
        advancedResultRank = advancedShape.size();
      }
    }
    if (termKind == 5) {
      if (operands.size() != 3 || values.size() != 3)
        return operation->emitOpError(
            "slice term requires start/stop/step payload slots");
      constexpr int64_t absent = std::numeric_limits<int64_t>::min();
      for (auto [operand, value] : llvm::zip(operands, values))
        if (operand >= 0 && value != absent)
          return operation->emitOpError(
              "slice payload cannot be dynamic and static simultaneously");
      if (values[2] == 0)
        return operation->emitOpError("slice step cannot be zero");
    }
  }
  if (consumed != relation.getSourceRank())
    return operation->emitOpError(
        "index relation must consume each source axis exactly once");
  result.resultRank = basicResultRank + advancedResultRank;
  if (result.resultRank != relation.getResultRank())
    return operation->emitOpError(
        "index relation result-rank fact disagrees with its typed terms");
  if (result.operandPositions.size() != indices.size())
    return operation->emitOpError(
        "indices segment contains an operand not referenced by its relation");
  return result;
}

std::optional<unsigned> getDataRank(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    return tensor.getRank();
  if (isa<IntegerType, IndexType, FloatType, LogicalIndexType>(type))
    return 0;
  return std::nullopt;
}

LogicalResult verifyIndexedData(Operation *operation, Type type,
                               Type elementType,
                               const IndexRelationInfo &relation,
                               StringRef subject) {
  auto actualRank = getDataRank(type);
  if (!actualRank || *actualRank != relation.resultRank ||
      !compatibleElementType(getElementType(type), elementType))
    return operation->emitOpError()
           << subject << " does not match the indexed element/rank schema";
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor)
    return success();
  DenseI64ArrayAttr actualDimensions = getDimensionIDs(tensor);
  for (unsigned axis = 0; axis < relation.resultRank; ++axis) {
    int64_t expected = relation.resultDimensions[axis];
    if (expected <= 0 || !actualDimensions ||
        actualDimensions[axis] != expected)
      return operation->emitOpError()
             << subject << " lost an index-relation result dimension identity";
  }
  return success();
}

LogicalResult verifyAccess(Operation *operation) {
  auto access = cast<IndexedAccessOpInterface>(operation);
  FailureOr<IndexRelationInfo> relation = verifyIndexRelation(operation);
  if (failed(relation))
    return failure();
  Type sourceType = access.getAccessSource().getType();
  auto source = getTensorSchema(sourceType);
  if (!source)
    return operation->emitOpError(
        "indexed access requires a ranked source/destination schema");

  bool load = isa<ViewLoadOp>(operation) || isa<BufferLoadOp>(operation) ||
              isa<GatherOp>(operation);
  bool viewAccess = isa<ViewLoadOp>(operation) || isa<ViewStoreOp>(operation);
  bool bufferAccess =
      isa<BufferLoadOp>(operation) || isa<BufferStoreOp>(operation);
  bool scatter =
      isa<ScatterUniqueOp>(operation) || isa<ScatterReduceOp>(operation);
  auto view = dyn_cast<ViewType>(sourceType);
  if ((viewAccess || scatter) && !view)
    return operation->emitOpError(
        "external access requires an external view destination/source");
  if (bufferAccess && !isa<BufferType>(sourceType))
    return operation->emitOpError(
        "logical-buffer access requires a logical buffer source");
  if (isa<GatherOp>(operation) && !isa<RankedTensorType>(sourceType))
    return operation->emitOpError(
        "pure gather requires an immutable tensor source");
  if (view && !load && view.getAccess() == 0)
    return operation->emitOpError("In-only external view cannot be written");

  if (load) {
    if (operation->getNumResults() != 1 ||
        failed(verifyIndexedData(operation, operation->getResult(0).getType(),
                                 source.getElementType(), *relation,
                                 "indexed read result")))
      return failure();
    auto valid = access.getAccessValidity();
    auto fill = access.getAccessFill();
    if (static_cast<bool>(valid) != static_cast<bool>(fill))
      return operation->emitOpError(
          "indexed read must provide both validity and fill or neither");
    if (valid) {
      Type resultType = operation->getResult(0).getType();
      Type validType = valid.getType();
      Type fillType = fill.getType();
      if (!isBooleanData(validType) ||
          !sameDataSchema(validType, resultType, false) ||
          !sameDataSchema(fillType, resultType))
        return operation->emitOpError(
            "indexed read validity/fill must match the canonical result shape");
    }
    return success();
  }

  if (failed(verifyIndexedData(operation, access.getStoredValue().getType(),
                               source.getElementType(), *relation,
                               "indexed write value")))
    return failure();
  if (!isa<ScatterReduceOp>(operation))
    return success();
  if (operation->getNumRegions() != 1)
    return operation->emitOpError(
        "scatter-reduce requires one typed combine region");
  Type valueType = source.getElementType();
  SmallVector<Type> arguments{valueType, valueType};
  SmallVector<Type> yielded{valueType};
  return verifyPureYieldSchema(
      operation, cast<ScatterReduceOp>(operation).getCombine(), arguments, yielded);
}

LogicalResult verifyAtomic(Operation *operation) {
  auto access = cast<IndexedAccessOpInterface>(operation);
  FailureOr<IndexRelationInfo> relation = verifyIndexRelation(operation);
  if (failed(relation))
    return failure();
  Type targetType = access.getAccessSource().getType();
  auto target = getTensorSchema(targetType);
  if (!target || !isa<ViewType, BufferType>(targetType))
    return operation->emitOpError("atomic target must be a view or logical buffer");
  auto order = operation->getAttrOfType<AtomicOrderingAttr>("ordering");
  if (!order)
    return operation->emitOpError("atomic ordering is outside the language enum");
  if (auto view = dyn_cast<ViewType>(targetType); view && view.getAccess() != 2)
    return operation->emitOpError(
        "external atomic target must use InOut access semantics");
  if (isa<AtomicLoadOp>(operation)) {
    if (order.getValue() != AtomicOrdering::Relaxed &&
        order.getValue() != AtomicOrdering::Acquire)
      return operation->emitOpError(
          "atomic load allows relaxed or acquire ordering");
    if (failed(verifyIndexedData(operation, operation->getResult(0).getType(),
                                 target.getElementType(), *relation,
                                 "atomic load result")))
      return failure();
    return success();
  }
  Value value =
      access.getCompareValue() ? access.getCompareValue() : access.getStoredValue();
  Type valueType = value.getType();
  if (failed(verifyIndexedData(operation, valueType, target.getElementType(),
                              *relation, "atomic value")))
    return failure();
  if (isa<AtomicStoreOp>(operation)) {
    if (order.getValue() != AtomicOrdering::Relaxed &&
        order.getValue() != AtomicOrdering::Release)
      return operation->emitOpError(
          "atomic store allows relaxed or release ordering");
    return success();
  }
  if (isa<AtomicRMWOp>(operation)) {
    auto kind = operation->getAttrOfType<AtomicRMWKindAttr>("kind");
    if (!kind || operation->getResult(0).getType() != valueType)
      return operation->emitOpError("atomic RMW schema is invalid");
    return success();
  }
  auto desired = access.getReplacementValue();
  auto result = dyn_cast<RecordType>(operation->getResult(0).getType());
  Type successType =
      result && result.getFieldTypes().size() == 2
          ? cast<TypeAttr>(result.getFieldTypes()[1]).getValue()
          : Type();
  if (!desired || desired.getType() != valueType || !result ||
      result.getFieldNames().size() != 2 ||
      cast<StringAttr>(result.getFieldNames()[0]).getValue() != "old_value" ||
      cast<StringAttr>(result.getFieldNames()[1]).getValue() != "success" ||
      cast<TypeAttr>(result.getFieldTypes()[0]).getValue() != valueType ||
      !successType || !getElementType(successType).isInteger(1) ||
      !sameDataSchema(successType, valueType, false))
    return operation->emitOpError("compare-exchange result schema is invalid");
  return success();
}

} // namespace

Value ViewLoadOp::getAccessValidity() { return getValid(); }
Value ViewLoadOp::getAccessFill() { return getFill(); }
Value GatherOp::getAccessValidity() { return getValid(); }
Value GatherOp::getAccessFill() { return getFill(); }
Value ViewStoreOp::getStoredValue() { return getValue(); }
Value BufferStoreOp::getStoredValue() { return getValue(); }
Value ScatterUniqueOp::getStoredValue() { return getValue(); }
Value ScatterReduceOp::getStoredValue() { return getValue(); }
Value AtomicStoreOp::getStoredValue() { return getValue(); }
Value AtomicRMWOp::getStoredValue() { return getValue(); }
Value AtomicCompareExchangeOp::getCompareValue() { return getExpected(); }
Value AtomicCompareExchangeOp::getReplacementValue() { return getDesired(); }

LogicalResult ViewLoadOp::verify() { return verifyAccess(getOperation()); }

LogicalResult ViewStoreOp::verify() { return verifyAccess(getOperation()); }

LogicalResult BufferLoadOp::verify() { return verifyAccess(getOperation()); }

LogicalResult BufferStoreOp::verify() { return verifyAccess(getOperation()); }

LogicalResult GatherOp::verify() { return verifyAccess(getOperation()); }

LogicalResult ScatterUniqueOp::verify() { return verifyAccess(getOperation()); }

LogicalResult ScatterReduceOp::verify() { return verifyAccess(getOperation()); }

LogicalResult AtomicLoadOp::verify() { return verifyAtomic(getOperation()); }

LogicalResult AtomicStoreOp::verify() { return verifyAtomic(getOperation()); }

LogicalResult AtomicRMWOp::verify() { return verifyAtomic(getOperation()); }

LogicalResult AtomicCompareExchangeOp::verify() {
  return verifyAtomic(getOperation());
}

LogicalResult BufferOp::verify() {
  Operation *operation = getOperation();
  auto buffer = dyn_cast<BufferType>(getResult().getType());
  auto tensor = buffer ? dyn_cast<RankedTensorType>(buffer.getTensor())
                       : RankedTensorType();
  if (!buffer || !tensor)
    return operation->emitOpError("logical buffer result schema is invalid");
  FailureOr<SmallVector<unsigned>> dynamicOperands =
      verifyShapeRelation(operation, getShape(), tensor, getExtents(), false);
  if (failed(dynamicOperands))
    return failure();
  for (auto [offset, position] : llvm::enumerate(*dynamicOperands))
    if (position != offset)
      return operation->emitOpError(
          "buffer shape extent operands must be canonical and position ordered");
  auto initial = getInitial();
  if (getExtents().size() != dynamicOperands->size())
    return operation->emitOpError(
        "logical buffer shape/initializer operand partition is invalid");
  if (initial) {
    Type initializer = initial.getType();
    bool scalar = compatibleElementType(initializer, tensor.getElementType());
    auto initializerTensor = dyn_cast<RankedTensorType>(initializer);
    bool complete =
        initializerTensor &&
        initializerTensor.getElementType() == tensor.getElementType() &&
        sameTensorShape(initializerTensor, tensor);
    if (!scalar && !complete)
      return operation->emitOpError(
          "logical buffer initializer must be an element scalar or complete value");
  }
  return success();
}

LogicalResult AssumeInBoundsOp::verify() {
  Operation *operation = getOperation();
  auto axis = getAxisAttr();
  auto rank = getLogicalRank(getView().getType());
  if (!axis || axis.getInt() < 0 || !rank || axis.getInt() >= *rank ||
      !isIntegerLike(getElementType(getIndex().getType())))
    return operation->emitOpError("in-bounds precondition schema is invalid");
  return success();
}

} // namespace intent
