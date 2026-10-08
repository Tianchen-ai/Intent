#include "TypeSchema.h"
#include "RegionVerification.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace intent::detail;

namespace intent {

namespace detail {

LogicalResult verifyPureYieldSchema(Operation *owner, Region &region,
                                    TypeRange arguments, TypeRange yielded) {
  if (!llvm::hasSingleElement(region) || region.front().empty())
    return owner->emitOpError(
        "structured region must contain one non-empty block");
  if (!isa<YieldOp>(region.front().back()))
    return owner->emitOpError("structured region must end in intent.yield");
  Block &block = region.front();
  if (!llvm::equal(block.getArgumentTypes(), arguments))
    return owner->emitOpError(
        "structured region argument schema is not canonical");
  Operation &terminator = block.back();
  if (!llvm::equal(terminator.getOperandTypes(), yielded))
    return owner->emitOpError("structured region yield schema is not canonical");
  WalkResult walk = region.walk([&](Operation *nested) {
    if (isa<YieldOp, AssumeInBoundsOp>(nested))
      return WalkResult::advance();
    if (isa<RandomBitsOp, BufferOp, ViewLoadOp, ViewStoreOp, BufferLoadOp,
            BufferStoreOp, ScatterUniqueOp, ScatterReduceOp, AtomicLoadOp,
            AtomicStoreOp, AtomicRMWOp, AtomicCompareExchangeOp>(nested)) {
      nested->emitOpError("is not legal in a pure structured helper region");
      return WalkResult::interrupt();
    }
    // Recursive control is legal only when every nested operation has already
    // passed this same walk; the container itself carries recursive effects.
    if (isa<IfOp, ForOp, WhileOp, ParallelOp>(nested))
      return WalkResult::advance();
    if (!isMemoryEffectFree(nested)) {
      nested->emitOpError(
          "has observable effects in a pure structured helper region");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return walk.wasInterrupted() ? failure() : success();
}

} // namespace detail

namespace {

LogicalResult verifySliceAssembly(Operation *operation, Type sliceType,
                                 Type resultType, int64_t sliceDimensionID,
                                 RankedTensorType source, unsigned sourceAxis) {
  if (auto sliceTuple = dyn_cast<intent::TupleType>(sliceType)) {
    auto resultTuple = dyn_cast<intent::TupleType>(resultType);
    if (!resultTuple || sliceTuple.getComponentTypes().size() !=
                            resultTuple.getComponentTypes().size())
      return operation->emitOpError(
          "region-scan tuple output/result schemas disagree");
    for (auto [sliceComponent, resultComponent] :
         llvm::zip(sliceTuple.getComponentTypes(),
                   resultTuple.getComponentTypes()))
      if (failed(verifySliceAssembly(
              operation, cast<TypeAttr>(sliceComponent).getValue(),
              cast<TypeAttr>(resultComponent).getValue(), sliceDimensionID,
              source, sourceAxis)))
        return failure();
    return success();
  }
  if (auto sliceRecord = dyn_cast<RecordType>(sliceType)) {
    auto resultRecord = dyn_cast<RecordType>(resultType);
    if (!resultRecord ||
        sliceRecord.getFieldNames() != resultRecord.getFieldNames() ||
        sliceRecord.getFieldTypes().size() != resultRecord.getFieldTypes().size())
      return operation->emitOpError(
          "region-scan record output/result schemas disagree");
    for (auto [sliceField, resultField] :
         llvm::zip(sliceRecord.getFieldTypes(), resultRecord.getFieldTypes()))
      if (failed(verifySliceAssembly(
              operation, cast<TypeAttr>(sliceField).getValue(),
              cast<TypeAttr>(resultField).getValue(), sliceDimensionID,
              source, sourceAxis)))
        return failure();
    return success();
  }
  auto slice = dyn_cast<RankedTensorType>(sliceType);
  auto result = dyn_cast<RankedTensorType>(resultType);
  if (!slice || !result || slice.getRank() != result.getRank() ||
      slice.getElementType() != result.getElementType())
    return operation->emitOpError(
        "region-scan output must be a tensor or structural tensor product");
  std::optional<unsigned> assembledAxis;
  for (unsigned axis = 0; axis < slice.getRank(); ++axis) {
    if (getDimensionID(slice, axis) == sliceDimensionID) {
      if (assembledAxis)
        return operation->emitOpError(
            "region-scan output slice extent must occur exactly once");
      assembledAxis = axis;
      if (!sameDimension(source, sourceAxis, result, axis))
        return operation->emitOpError(
            "region-scan result does not restore the full source extent");
    } else if (!sameDimension(slice, axis, result, axis)) {
      return operation->emitOpError(
          "region-scan output assembly changed a non-source dimension");
    }
  }
  return assembledAxis
             ? success()
             : operation->emitOpError(
                   "region-scan output slice does not carry the source slice extent");
}

FailureOr<SmallVector<std::pair<unsigned, unsigned>>>
getAxisPairs(Operation *operation, StringRef name, unsigned lhsRank,
             unsigned rhsRank) {
  auto array = operation->getAttrOfType<ArrayAttr>(name);
  if (!array) {
    operation->emitOpError() << "requires axis-pair attribute '" << name << "'";
    return failure();
  }
  SmallVector<std::pair<unsigned, unsigned>> pairs;
  for (Attribute attribute : array) {
    auto pair = dyn_cast<ArrayAttr>(attribute);
    if (!pair || pair.size() != 2 || !isa<IntegerAttr>(pair[0]) ||
        !isa<IntegerAttr>(pair[1])) {
      operation->emitOpError() << "attribute '" << name
                               << "' must contain integer axis pairs";
      return failure();
    }
    int64_t lhs = cast<IntegerAttr>(pair[0]).getInt();
    int64_t rhs = cast<IntegerAttr>(pair[1]).getInt();
    if (lhs < 0 || rhs < 0 || lhs >= lhsRank || rhs >= rhsRank) {
      operation->emitOpError() << "attribute '" << name
                               << "' contains an out-of-range axis";
      return failure();
    }
    pairs.emplace_back(static_cast<unsigned>(lhs), static_cast<unsigned>(rhs));
  }
  return pairs;
}

LogicalResult verifyReduceOrScan(StructuredOpInterface structured, bool scan) {
  Operation *operation = structured.getOperation();
  if (failed(verifyStructuredArity(structured)))
    return failure();

  SmallVector<int64_t> axes;
  if (scan) {
    auto axis = operation->getAttrOfType<IntegerAttr>("axis");
    if (!axis || axis.getInt() < 0)
      return operation->emitOpError("scan requires a non-negative axis");
    axes.push_back(axis.getInt());
  } else {
    FailureOr<SmallVector<int64_t>> values = getIntegerArray(operation, "axes");
    if (failed(values) || values->empty())
      return operation->emitOpError("reduce requires one or more axes");
    axes = *values;
  }

  SmallVector<Type> accumulatorTypes;
  RankedTensorType firstSource;
  llvm::DenseSet<int64_t> uniqueAxes;
  for (int64_t axis : axes)
    if (axis < 0 || !uniqueAxes.insert(axis).second)
      return operation->emitOpError(
          "reduce/scan axes must be unique and non-negative");
  for (auto [sourceValue, identityValue, resultValue] : llvm::zip_equal(
           structured.getSources(), structured.getIdentities(),
           operation->getResults())) {
    auto source = dyn_cast<RankedTensorType>(sourceValue.getType());
    Type identity = identityValue.getType();
    if (!source || getElementType(identity) != source.getElementType())
      return operation->emitOpError(
          "reduce/scan source and identity component types disagree");
    for (int64_t axis : axes)
      if (axis >= source.getRank())
        return operation->emitOpError("reduce/scan axis is outside source rank");
    if (firstSource) {
      for (int64_t axis : axes)
        if (!sameDimension(firstSource, axis, source, axis))
          return operation->emitOpError(
              "reduce/scan source components must share each member-axis extent");
    } else {
      firstSource = source;
    }
    Type result = resultValue.getType();
    if (scan) {
      auto identityTensor = dyn_cast<RankedTensorType>(identity);
      bool scalarIdentity = identity == source.getElementType();
      bool sliceIdentity =
          identityTensor && identityTensor.getRank() + 1 == source.getRank() &&
          identityTensor.getElementType() == source.getElementType();
      if (sliceIdentity) {
        unsigned identityAxis = 0;
        for (unsigned sourceAxis = 0; sourceAxis < source.getRank(); ++sourceAxis) {
          if (sourceAxis == static_cast<unsigned>(axes.front()))
            continue;
          if (!sameDimension(source, sourceAxis, identityTensor, identityAxis++))
            sliceIdentity = false;
        }
      }
      if ((!scalarIdentity && !sliceIdentity) || result != source)
        return operation->emitOpError("scan result must preserve source type");
    } else {
      auto resultTensor = dyn_cast<RankedTensorType>(result);
      unsigned resultRank = source.getRank() - axes.size();
      bool scalarIdentity = identity == source.getElementType();
      bool resultIdentity = identity == result;
      if ((resultRank == 0 &&
           (!scalarIdentity || result != source.getElementType())) ||
          (resultRank != 0 &&
           (!resultTensor || resultTensor.getRank() != resultRank ||
            resultTensor.getElementType() != source.getElementType() ||
            (!scalarIdentity && !resultIdentity))))
        return operation->emitOpError(
            "reduce result type does not match removed axes");
      if (resultTensor) {
        unsigned resultAxis = 0;
        for (unsigned sourceAxis = 0; sourceAxis < source.getRank(); ++sourceAxis) {
          if (uniqueAxes.contains(sourceAxis))
            continue;
          if (!sameDimension(source, sourceAxis, resultTensor, resultAxis++))
            return operation->emitOpError(
                "reduce result dimension lost source identity");
        }
      }
    }
    accumulatorTypes.push_back(identity);
  }
  return verifyPureYieldSchema(operation, structured.getCombine(),
                               structured.getCombineArgumentTypes(),
                               accumulatorTypes);
}

LogicalResult verifySegmentedOperation(StructuredOpInterface structured) {
  Operation *operation = structured.getOperation();
  if (failed(verifyStructuredArity(structured)))
    return failure();
  int64_t axis = structured.getIterationAxes().front();
  if (axis < 0)
    return operation->emitOpError("source axis must be non-negative");
  SmallVector<Type> sliceTypes;
  RankedTensorType first;
  for (auto [sourceValue, sliceValue] : llvm::zip_equal(
           structured.getSources(), structured.getSummarizeSources())) {
    auto source = dyn_cast<RankedTensorType>(sourceValue.getType());
    if (!source || axis >= source.getRank())
      return operation->emitOpError("source axis is invalid");
    if (!first)
      first = source;
    else if (!sameDimension(first, axis, source, axis))
      return operation->emitOpError(
          "sources do not share one logical source extent");
    sliceTypes.push_back(sliceValue.getType());
    auto slice = dyn_cast<RankedTensorType>(sliceTypes.back());
    if (!slice || slice.getRank() != source.getRank() ||
        slice.getElementType() != source.getElementType())
      return operation->emitOpError("summarize slice type is invalid");
    for (unsigned dimension = 0; dimension < source.getRank(); ++dimension)
      if (dimension != static_cast<unsigned>(axis) &&
          !sameDimension(source, dimension, slice, dimension))
        return operation->emitOpError(
            "summarize slice changed a non-source dimension");
    if (sliceTypes.size() > 1) {
      auto firstSlice = cast<RankedTensorType>(sliceTypes.front());
      if (getDimensionID(firstSlice, axis) != getDimensionID(slice, axis))
        return operation->emitOpError(
            "source components are not sliced in lockstep");
    }
  }
  auto types = [](ValueRange values) {
    return llvm::to_vector(llvm::map_range(values, [](Value value) {
      return value.getType();
    }));
  };
  SmallVector<Type> identities = types(structured.getIdentities());
  if (failed(verifyPureYieldSchema(
          operation, *structured.getSummarizeRegion(),
          structured.getSummarizeArgumentTypes(sliceTypes), identities)))
    return failure();
  if (failed(verifyPureYieldSchema(operation, structured.getCombine(),
                                   structured.getCombineArgumentTypes(),
                                   identities)))
    return failure();
  if (structured.getStructuredKind() == StructuredOpKind::RegionFold) {
    if (!llvm::equal(identities, operation->getResultTypes()))
      return operation->emitOpError("identity/result schema disagrees");
    return success();
  }
  SmallVector<Type> states = types(structured.getInitialStates());
  SmallVector<Type> finalStates = types(structured.getFinalStates());
  if (!llvm::equal(states, finalStates))
    return operation->emitOpError("final-state schema is inconsistent");
  if (failed(verifyPureYieldSchema(operation, *structured.getApplyRegion(),
                                   structured.getApplyArgumentTypes(), states)))
    return failure();
  Region &emit = *structured.getEmitRegion();
  if (failed(verifyPureYieldSchema(operation, emit,
                                   structured.getEmitArgumentTypes(sliceTypes),
                                   emit.front().back().getOperandTypes())))
    return failure();
  auto firstSlice = cast<RankedTensorType>(sliceTypes.front());
  std::optional<int64_t> sliceDimensionID = getDimensionID(firstSlice, axis);
  if (!sliceDimensionID)
    return operation->emitOpError(
        "region-scan slice axis requires a stable dynamic extent identity");
  for (auto [emitted, output] : llvm::zip_equal(
           structured.getEmitYields(), structured.getEmittedResults()))
    if (failed(verifySliceAssembly(
            operation, emitted.getType(), output.getType(),
            *sliceDimensionID, first, axis)))
      return failure();
  return success();
}

LogicalResult verifyContraction(Operation *operation, Value left, Value right,
                               Value output, SparseContractOp sparse = {}) {
  auto lhs = dyn_cast<RankedTensorType>(left.getType());
  auto rhs = dyn_cast<RankedTensorType>(right.getType());
  auto result = dyn_cast<RankedTensorType>(output.getType());
  if (!lhs || !rhs || !result)
    return operation->emitOpError("contract family requires ranked tensor data");
  FailureOr<SmallVector<std::pair<unsigned, unsigned>>> reductions =
      getAxisPairs(operation, "reduce", lhs.getRank(), rhs.getRank());
  FailureOr<SmallVector<std::pair<unsigned, unsigned>>> batches =
      getAxisPairs(operation, "batch", lhs.getRank(), rhs.getRank());
  if (failed(reductions) || failed(batches) || reductions->empty())
    return operation->emitOpError("contract requires at least one reduction pair");
  llvm::DenseSet<unsigned> lhsUsed;
  llvm::DenseSet<unsigned> rhsUsed;
  auto sparseFormat = sparse ? sparse.getFormat() : SparseFormatAttr();
  for (auto [left, right] : *reductions) {
    if (!lhsUsed.insert(left).second || !rhsUsed.insert(right).second ||
        (!(sparse && sparseFormat &&
           sparseFormat.getCompressionAxis() == left) &&
         !sameDimension(lhs, left, rhs, right)))
      return operation->emitOpError(
          "contract reduction axes must be unique and extent-compatible");
  }
  for (auto [left, right] : *batches) {
    if (!lhsUsed.insert(left).second || !rhsUsed.insert(right).second ||
        !sameDimension(lhs, left, rhs, right))
      return operation->emitOpError(
          "contract batch axes must be unique, disjoint, and compatible");
  }
  unsigned expectedRank = batches->size() + lhs.getRank() + rhs.getRank() -
                          2 * reductions->size() - 2 * batches->size();
  if (result.getRank() != expectedRank)
    return operation->emitOpError("contract result rank is inconsistent");
  SmallVector<std::pair<RankedTensorType, unsigned>> expectedAxes;
  for (unsigned axis = 0; axis < lhs.getRank(); ++axis)
    if (!llvm::any_of(*reductions,
                      [&](auto pair) { return pair.first == axis; }))
      expectedAxes.emplace_back(lhs, axis);
  for (unsigned axis = 0; axis < rhs.getRank(); ++axis) {
    bool reduction = llvm::any_of(
        *reductions, [&](auto pair) { return pair.second == axis; });
    bool batch = llvm::any_of(*batches,
                             [&](auto pair) { return pair.second == axis; });
    if (!reduction && !batch)
      expectedAxes.emplace_back(rhs, axis);
  }
  if (expectedAxes.size() != static_cast<unsigned>(result.getRank()))
    return operation->emitOpError("contract result axis schema is inconsistent");
  for (auto [resultAxis, expected] : llvm::enumerate(expectedAxes))
    if (!sameDimension(expected.first, expected.second, result, resultAxis))
      return operation->emitOpError(
          "contract result dimension lost its free/batch axis identity");
  if (!isNumericData(result))
    return operation->emitOpError(
        "contract accumulator/result element type must be numeric");
  if (sparse) {
    auto format = sparse.getFormat();
    if (!format || format.getCompressionAxis() >= lhs.getRank() ||
        !sparse.getLogicalExtent().getType().isIndex())
      return operation->emitOpError("sparse contract format schema is invalid");
    if (!llvm::any_of(*reductions, [&](auto pair) {
          return pair.first == format.getCompressionAxis();
        }))
      return operation->emitOpError(
          "sparse compression axis must be a contraction reduction axis");
    auto reduction = llvm::find_if(*reductions, [&](auto pair) {
      return pair.first == format.getCompressionAxis();
    });
    Value logicalExtent = sparse.getLogicalExtent();
    if (!extentValueMatchesAxis(logicalExtent, rhs, reduction->second))
      return operation->emitOpError(
          "sparse logical extent must equal its paired dense reduction axis");
    int64_t group = format.getKind() == 0 ? 2 : 4;
    int64_t nonzeros = format.getKind() == 0 ? 1 : 2;
    if (auto constant = getConstantInteger(logicalExtent)) {
      if (*constant < 0 || *constant % group != 0)
        return operation->emitOpError(
            "sparse logical extent violates the closed format group size");
      int64_t compressedExtent = lhs.getDimSize(format.getCompressionAxis());
      if (!ShapedType::isDynamic(compressedExtent) &&
          compressedExtent != *constant / group * nonzeros)
        return operation->emitOpError(
            "sparse compressed extent disagrees with its logical format");
    }
    Type metadata = sparse.getMetadata().getType();
    RankedTensorType positions;
    if (format.getKind() == 0) {
      positions = dyn_cast<RankedTensorType>(metadata);
      if (!positions || !isIntegerLike(positions.getElementType()))
        return operation->emitOpError(
            "one-of-two metadata must be a logical-index tensor");
    } else {
      auto record = dyn_cast<RecordType>(metadata);
      if (!record || record.getFieldNames().size() != 2 ||
          cast<StringAttr>(record.getFieldNames()[0]).getValue() != "first" ||
          cast<StringAttr>(record.getFieldNames()[1]).getValue() != "second")
        return operation->emitOpError(
            "two-of-four metadata must be the {first, second} record");
      auto first = dyn_cast<RankedTensorType>(
          cast<TypeAttr>(record.getFieldTypes()[0]).getValue());
      auto second = dyn_cast<RankedTensorType>(
          cast<TypeAttr>(record.getFieldTypes()[1]).getValue());
      if (!first || !second || !sameTensorShape(first, second) ||
          !isIntegerLike(first.getElementType()) ||
          !isIntegerLike(second.getElementType()))
        return operation->emitOpError(
            "two-of-four metadata fields must be shape-identical logical-index tensors");
      positions = first;
    }
    if (!positions || positions.getRank() != lhs.getRank())
      return operation->emitOpError(
          "sparse metadata must preserve the compressed operand rank");
    for (unsigned axis = 0; axis < lhs.getRank(); ++axis)
      if (axis != format.getCompressionAxis() &&
          !sameDimension(lhs, axis, positions, axis))
        return operation->emitOpError(
            "sparse metadata changed a non-compression operand axis");
    if (auto constant = getConstantInteger(logicalExtent)) {
      int64_t metadataExtent =
          positions.getDimSize(format.getCompressionAxis());
      if (!ShapedType::isDynamic(metadataExtent) &&
          metadataExtent != *constant / group)
        return operation->emitOpError(
            "sparse metadata extent disagrees with its logical groups");
    }
  }
  return success();
}

bool isQuantizationRecord(RankedTensorType type, int64_t bytes) {
  return type.getRank() == 2 && type.getDimSize(1) == bytes &&
         type.getElementType().isUnsignedInteger(8);
}

} // namespace

LogicalResult ReduceOp::verify() {
  return verifyReduceOrScan(cast<StructuredOpInterface>(getOperation()), false);
}

LogicalResult ScanOp::verify() {
  return verifyReduceOrScan(cast<StructuredOpInterface>(getOperation()), true);
}

LogicalResult RegionFoldOp::verify() {
  return verifySegmentedOperation(cast<StructuredOpInterface>(getOperation()));
}

LogicalResult RegionScanOp::verify() {
  return verifySegmentedOperation(cast<StructuredOpInterface>(getOperation()));
}

LogicalResult ContractOp::verify() {
  return verifyContraction(*this, getLhs(), getRhs(), getResult());
}

LogicalResult SparseContractOp::verify() {
  return verifyContraction(*this, getCompressed(), getRhs(), getResult(), *this);
}

LogicalResult ScaledContractOp::verify() {
  Operation *operation = getOperation();
  auto lhs = dyn_cast<RankedTensorType>(getLhs().getType());
  auto rhs = dyn_cast<RankedTensorType>(getRhs().getType());
  auto result = dyn_cast<RankedTensorType>(getResult().getType());
  if (!lhs || !rhs || !result)
    return operation->emitOpError("contract family requires ranked tensor data");
  FailureOr<SmallVector<std::pair<unsigned, unsigned>>> reductions =
      getAxisPairs(operation, "reduce", lhs.getRank(), rhs.getRank());
  FailureOr<SmallVector<std::pair<unsigned, unsigned>>> batches =
      getAxisPairs(operation, "batch", lhs.getRank(), rhs.getRank());
  if (failed(reductions) || failed(batches) || reductions->empty())
    return operation->emitOpError("contract requires at least one reduction pair");
  auto lhsScale = dyn_cast<RankedTensorType>(getLhsScale().getType());
  auto rhsScale = dyn_cast<RankedTensorType>(getRhsScale().getType());
  auto lhsGroup = getLhsGroupSizeAttr();
  auto rhsGroup = getRhsGroupSizeAttr();
  auto lhsFormat = getLhsFormatAttr();
  auto rhsFormat = getRhsFormatAttr();
  bool fixedAxes =
      reductions->size() == 2 && batches->empty() &&
      (*reductions)[0] == std::make_pair(1u, 0u) &&
      (*reductions)[1] == std::make_pair(2u, 1u);
  if (!lhsScale || !rhsScale || lhs.getRank() != 3 ||
      lhsScale.getRank() != 2 || rhs.getRank() != 3 ||
      rhsScale.getRank() != 2 || !fixedAxes || !lhsGroup || !rhsGroup ||
      lhsGroup.getInt() <= 0 || lhsGroup != rhsGroup || !lhsFormat ||
      !rhsFormat)
    return operation->emitOpError(
        "scaled contract requires the closed [M,G,C]/[M,G] x [G,C,N]/[N,G] schema");
  auto carrierExtent = [](ScaledFormat format,
                          int64_t group) -> std::optional<int64_t> {
    int64_t packing = format == ScaledFormat::E2M1 ? 2 : 1;
    if (group % packing != 0)
      return std::nullopt;
    return group / packing;
  };
  std::optional<int64_t> lhsCarrier =
      carrierExtent(lhsFormat.getValue(), lhsGroup.getInt());
  std::optional<int64_t> rhsCarrier =
      carrierExtent(rhsFormat.getValue(), rhsGroup.getInt());
  if (!lhsCarrier || !rhsCarrier || lhs.getDimSize(2) != *lhsCarrier ||
      rhs.getDimSize(1) != *rhsCarrier ||
      !sameDimension(lhs, 0, lhsScale, 0) ||
      !sameDimension(lhs, 1, lhsScale, 1) ||
      !sameDimension(lhs, 1, rhs, 0) ||
      !sameDimension(lhs, 1, rhsScale, 1) ||
      !sameDimension(rhs, 2, rhsScale, 0) || result.getRank() != 2 ||
      !sameDimension(lhs, 0, result, 0) ||
      !sameDimension(rhs, 2, result, 1) || !isNumericData(lhsScale) ||
      !isNumericData(rhsScale) || !isNumericData(result))
    return operation->emitOpError(
        "scaled contract operands violate the closed scale-axis relation");
  return success();
}

LogicalResult QuantizeOp::verify() {
  Operation *operation = getOperation();
  auto input = cast<RankedTensorType>(getInput().getType());
  auto result = cast<RankedTensorType>(getResult().getType());
  if (getFormat() != QuantFormat::Q8K || input.getRank() != 2 ||
      input.getDimSize(1) != 256 || !input.getElementType().isF32() ||
      !isQuantizationRecord(result, 292) || !sameDimension(input, 0, result, 0))
    return operation->emitOpError("Q8_K quantize requires f32[G,256] -> u8[G,292]");
  return success();
}

LogicalResult QuantizedDotOp::verify() {
  Operation *operation = getOperation();
  auto input = cast<RankedTensorType>(getLhs().getType());
  auto result = cast<RankedTensorType>(getResult().getType());
  auto rhs = cast<RankedTensorType>(getRhs().getType());
  if (getLhsFormat() != QuantFormat::Q4K || getRhsFormat() != QuantFormat::Q8K ||
      !isQuantizationRecord(input, 144) || !isQuantizationRecord(rhs, 292) ||
      !sameDimension(input, 0, rhs, 0) ||
      result.getRank() != 0 || !result.getElementType().isF32())
    return operation->emitOpError(
        "quantized dot requires Q4_K u8[G,144] x Q8_K u8[G,292] -> f32[]");
  return success();
}

LogicalResult HistogramOp::verify() {
  Operation *operation = getOperation();
  auto values = dyn_cast<RankedTensorType>(getValues().getType());
  auto valid = dyn_cast<RankedTensorType>(getValid().getType());
  auto result = dyn_cast<RankedTensorType>(getResult().getType());
  auto resultDimension = getResultDimensionAttr();
  if (!values || !valid || !result || !resultDimension ||
      !isIntegerLike(values.getElementType()) ||
      !valid.getElementType().isInteger(1) || !sameTensorShape(values, valid) ||
      result.getRank() != 1 || !isa<IntegerType>(result.getElementType()) ||
      result.getElementType().isInteger(1))
    return operation->emitOpError("histogram value/valid/result schema is invalid");
  Value bins = getBins();
  if (Operation *definition = bins.getDefiningOp()) {
    if (isa<ConstantOp>(definition)) {
      auto value = definition->getAttrOfType<IntegerAttr>("value");
      auto ids = getDimensionIDs(result);
      if (!value || value.getInt() <= 0 || !ids ||
          resultDimension.getInt() <= 0 ||
          ids[0] != resultDimension.getInt() ||
          result.getDimSize(0) != value.getInt())
        return operation->emitOpError(
            "histogram result extent must equal its static bin count");
      return success();
    }
  }
  if (!ShapedType::isDynamic(result.getDimSize(0)))
    return operation->emitOpError(
        "runtime histogram bin count requires a dynamic result extent");
  auto ids = getDimensionIDs(result);
  if (!ids || resultDimension.getInt() <= 0 ||
      ids[0] != resultDimension.getInt())
    return operation->emitOpError(
        "runtime histogram bin count lost its dimension identity");
  return success();
}

} // namespace intent
