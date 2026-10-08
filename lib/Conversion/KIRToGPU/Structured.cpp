#include "Construction.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/PhysicalParameters.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;

namespace intent::kir_to_gpu {

namespace {
void axisPairs(
    ArrayAttr pairs,
    SmallVectorImpl<int64_t> &lhs,
    SmallVectorImpl<int64_t> &rhs) {
  for (Attribute attribute : pairs) {
    auto pair = cast<ArrayAttr>(attribute);
    lhs.push_back(cast<IntegerAttr>(pair[0]).getInt());
    rhs.push_back(cast<IntegerAttr>(pair[1]).getInt());
  }
}

FailureOr<Value> zeroAccumulator(
    OpBuilder &builder,
    Location location,
    FragmentType type) {
  Type element = type.getElementType();
  Value zero;
  if (isa<FloatType>(element))
    zero = builder.create<arith::ConstantOp>(
        location, element, builder.getFloatAttr(element, 0.0));
  else if (auto integer = dyn_cast<IntegerType>(element))
    zero = builder.create<arith::ConstantOp>(
        location, element, builder.getIntegerAttr(integer, 0));
  else
    return failure();
  return Value(builder.create<gpu::SplatOp>(location, type, zero));
}
} // namespace

static FailureOr<FragmentType> convertSegmentSliceType(
    RankedTensorType tensor,
    FragmentType prototype,
    unsigned axis,
    StringRef parameter) {
  if (axis >= prototype.getShape().size() ||
      tensor.getRank() != static_cast<int64_t>(prototype.getShape().size()) ||
      tensor.getElementType() != prototype.getElementType())
    return failure();
  SmallVector<Attribute> shape(prototype.getShape().begin(),
                               prototype.getShape().end());
  SmallVector<Attribute> mappings(prototype.getAxisMaps().begin(),
                                  prototype.getAxisMaps().end());
  DenseI64ArrayAttr dimensions = dimensionIds(tensor);
  if (!dimensions || dimensions.size() != tensor.getRank() ||
      dimensions[axis] <= 0)
    return failure();
  shape[axis] = parameterExpression(tensor.getContext(), parameter);
  // The explicit structured-region boundary creates a helper-local slice
  // dimension.  Keep the source component's non-segment provenance, but give
  // the sliced axis its canonical helper dimension identity.  Otherwise one
  // logical source used simultaneously as an outer ownership axis and an
  // inner segment axis would collapse to one physical extent authority.
  auto mapping = cast<AxisMapAttr>(prototype.getAxisMaps()[axis]);
  mappings[axis] = AxisMapAttr::get(
      tensor.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
      dimensions[axis], axis, mapping.getDerived());
  return FragmentType::get(tensor.getContext(), tensor.getElementType(),
                           ArrayAttr::get(tensor.getContext(), shape),
                           ArrayAttr::get(tensor.getContext(), mappings),
                           prototype.getValidity(),
                           prototype.getOwner());
}

static FailureOr<Type> regionAssemblyType(Type result, Type slice) {
  if (auto resultFragment = dyn_cast<FragmentType>(result)) {
    auto sliceFragment = dyn_cast<FragmentType>(slice);
    if (!sliceFragment ||
        resultFragment.getElementType() != sliceFragment.getElementType() ||
        resultFragment.getShape().size() != sliceFragment.getShape().size() ||
        resultFragment.getOwner() != sliceFragment.getOwner())
      return failure();
    return Type(FragmentType::get(
        result.getContext(), resultFragment.getElementType(),
        resultFragment.getShape(), sliceFragment.getAxisMaps(),
        resultFragment.getValidity(), resultFragment.getOwner()));
  }
  auto resultRecord = dyn_cast<gpu::RecordType>(result);
  auto sliceRecord = dyn_cast<gpu::RecordType>(slice);
  if (!resultRecord || !sliceRecord ||
      resultRecord.getFieldNames() != sliceRecord.getFieldNames() ||
      resultRecord.getFieldTypes().size() !=
          sliceRecord.getFieldTypes().size() ||
      resultRecord.getOwner() != sliceRecord.getOwner())
    return result == slice ? FailureOr<Type>(result)
                           : FailureOr<Type>(failure());
  SmallVector<Attribute> fields;
  for (auto [whole, part] : llvm::zip(resultRecord.getFieldTypes(),
                                      sliceRecord.getFieldTypes())) {
    FailureOr<Type> field = regionAssemblyType(
        cast<TypeAttr>(whole).getValue(), cast<TypeAttr>(part).getValue());
    if (failed(field))
      return failure();
    fields.push_back(TypeAttr::get(*field));
  }
  return Type(gpu::RecordType::get(
      result.getContext(), resultRecord.getFieldNames(),
      ArrayAttr::get(result.getContext(), fields), resultRecord.getOwner()));
}

static FailureOr<Type> convertReductionResultType(
    CanonicalKernelAnalysis &canonicalAnalysis,
    Type logical,
    Operation *origin,
    Value source,
    ArrayRef<int64_t> reducedAxes,
    unsigned resultIndex) {
  auto sourceType = dyn_cast<FragmentType>(source.getType());
  auto logicalType = dyn_cast<RankedTensorType>(logical);
  if (!sourceType || !logicalType)
    return convertDataType(canonicalAnalysis, logical, origin, std::nullopt, 1, resultIndex);
  if (sourceType.getShape().size() < reducedAxes.size())
    return failure();
  if (logicalType.getRank() !=
      static_cast<int64_t>(sourceType.getShape().size() - reducedAxes.size()))
    return failure();
  return gpu::inferCollectiveResultType(sourceType, reducedAxes,
                                        logicalType.getElementType());
}

static FailureOr<Type> convertContractResultType(
    CanonicalKernelAnalysis &canonicalAnalysis,
    OpResult result,
    Value lhs,
    Value rhs) {
  auto tensor = dyn_cast<RankedTensorType>(result.getType());
  auto left = dyn_cast<FragmentType>(lhs.getType());
  auto right = dyn_cast<FragmentType>(rhs.getType());
  if (!tensor || !left || !right || left.getOwner() != right.getOwner())
    return failure();
  auto projections = canonicalAnalysis.operandProjections(result);
  if (failed(projections) || projections->size() != 2)
    return failure();
  SmallVector<Attribute> shape(tensor.getRank()), mappings(tensor.getRank());
  for (auto [index, projection] : llvm::enumerate(*projections)) {
    auto source = index == 0 ? left : right;
    if (source.getShape().size() != projection.resultAxes.size())
      return failure();
    for (auto [axis, resultAxis] : llvm::enumerate(projection.resultAxes)) {
      if (!resultAxis || shape[*resultAxis])
        continue; // The lhs is the declared representative of a batch pair.
      shape[*resultAxis] = source.getShape()[axis];
      auto mapping = cast<AxisMapAttr>(source.getAxisMaps()[axis]);
      mappings[*resultAxis] = AxisMapAttr::get(
          tensor.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
          mapping.getDimensionId(), *resultAxis, mapping.getDerived());
    }
  }
  if (llvm::any_of(shape, [](Attribute extent) { return !extent; }))
    return failure();
  return Type(FragmentType::get(
      tensor.getContext(), tensor.getElementType(),
      ArrayAttr::get(tensor.getContext(), shape),
      ArrayAttr::get(tensor.getContext(), mappings), 1, left.getOwner()));
}

FailureOr<gpu::ParameterAttr> ScalarRegionLowering::getOrCreateRegionSegment(
    Operation *operation) {
  RegionSegmentFact fact = canonicalAnalysis.regionSegment(operation);
  if (!fact.isExact())
    return operation->emitOpError(
        "region segmentation has no exact canonical source relation");
  std::string name =
      ("SEGMENT_N" + Twine(fact.operationIdentity) + "_D" +
       Twine(fact.dimensionIdentity))
          .str();
  Type sourceType = operation->getOperand(0).getType();
  if (auto tensor = dyn_cast<RankedTensorType>(sourceType))
    sourceType = tensor.getElementType();
  if (!isa<IntegerType, FloatType>(sourceType))
    return operation->emitOpError(
        "region segment has no scalar element type for its physical parameter");
  bool contraction = false;
  for (Region &region : operation->getRegions())
    region.walk([&](Operation *nested) {
      contraction |= isa<intent::ContractOp, intent::ScaledContractOp,
                         intent::SparseContractOp>(nested);
    });
  gpu::ParameterCategory category =
      contraction ? gpu::ParameterCategory::RegionContraction
                  : isa<intent::RegionFoldOp>(operation)
                        ? gpu::ParameterCategory::RegionReduction
                        : gpu::ParameterCategory::Scan;
  auto schema = gpu::ParameterAttr::get(
      operation->getContext(), builder.getStringAttr(name), builder.getIndexType(),
      gpu::ParameterRole::ScanChunk,
      category,
      sourceType.getIntOrFloatBitWidth(),
      builder.getDenseI64ArrayAttr(
          {16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
           32768, 65536}),
      gpu::ConfigurationBindingPhase::Shared,
      gpu::ParameterBindingAttr::get(
          operation->getContext(), builder.getI64IntegerAttr(fact.dimensionIdentity),
          {}, {}, {}, false, false));
  func::FuncOp physical = physicalKernel;

  auto reference = gpu::declareParameter(physical, schema);
  if (failed(reference))
    return failure();
  if (!parameters.count(schema.getName())) {
    OpBuilder declarationBuilder(&physical.getBody().front(),
                                 physical.getBody().front().begin());
    auto read = gpu::materializeParameter(
        declarationBuilder, operation->getLoc(), *reference);
    read->setAttr(
        gpu::originAttr,
        declarationBuilder.getI64IntegerAttr(fact.operationIdentity));
    parameters[schema.getName()] = read.getResult();
  }
  return schema;
}

LogicalResult ScalarRegionLowering::lowerPureRegion(
    Region &source,
    Region &target,
    ArrayRef<Type> argumentTypes,
    ArrayRef<Type> resultTypes) {
  if (!target.empty())
    return failure();
  OpBuilder::InsertionGuard guard(builder);
  Block *block = builder.createBlock(
      &target, target.end(), argumentTypes,
      SmallVector<Location>(argumentTypes.size(),
                            source.getParentOp()->getLoc()));
  auto childValues = values;
  for (auto [from, to] :
       llvm::zip(source.front().getArguments(), block->getArguments()))
    childValues[from] = to;
  builder.setInsertionPointToStart(block);
  ScalarRegionLowering child(builder, std::move(childValues), views,
                             abiDimensions, parameters, canonicalAnalysis,
                             physicalKernel);
  FailureOr<SmallVector<Value>> yielded = child.lowerBlock(source.front());
  if (failed(yielded))
    return failure();
  if (!resultTypes.empty()) {
    if (yielded->size() != resultTypes.size())
      return failure();
    for (auto [index, type] : llvm::enumerate(resultTypes)) {
      FailureOr<Value> projected = projectPositionalValue(
          builder, source.getParentOp()->getLoc(), (*yielded)[index], type);
      if (failed(projected))
        return failure();
      (*yielded)[index] = *projected;
    }
  }
  builder.create<gpu::YieldOp>(source.getParentOp()->getLoc(), *yielded);
  return success();
}

LogicalResult ScalarRegionLowering::lowerStructured(
    StructuredOpInterface schema) {
  Operation *operation = schema.getOperation();
  Location location = schema.getLoc();
  auto lowerOperands = [&](ValueRange operands) -> FailureOr<SmallVector<Value>> {
    SmallVector<Value> values;
    for (Value operand : operands) {
      auto value = get(operand);
      if (failed(value))
        return operation->emitOpError("structured physical operand is unavailable"), failure();
      values.push_back(*value);
    }
    return values;
  };
  auto sources = lowerOperands(schema.getSources());
  auto identities = lowerOperands(schema.getIdentities());
  auto initialStates = lowerOperands(schema.getInitialStates());
  auto captures = lowerOperands(schema.getCaptures());
  if (failed(sources) || failed(identities) || failed(initialStates) || failed(captures))
    return failure();
  auto kind = schema.getStructuredKind();
  auto axes = schema.getIterationAxes();
  const bool region = schema.getSummarizeRegion() != nullptr;
  const bool scan = kind == StructuredOpKind::RegionScan;
  SmallVector<Type> results;
  for (auto [index, logical] : llvm::enumerate(operation->getResultTypes())) {
    FailureOr<Type> converted = failure();
    if (kind == StructuredOpKind::Reduce) {
      if (index >= sources->size())
        return operation->emitOpError("reduce result has no corresponding physical source");
      converted = convertReductionResultType(canonicalAnalysis, logical, operation, (*sources)[index], axes, index);
    } else {
      std::optional<Type> prototype;
      if (kind == StructuredOpKind::Scan) {
        if (index >= sources->size())
          return operation->emitOpError("scan result has no corresponding physical source");
        prototype = (*sources)[index].getType();
      } else if (kind == StructuredOpKind::RegionFold) {
        prototype = (*identities)[index].getType();
      } else if (index >= schema.getEmittedResults().size()) {
        unsigned state = index - schema.getEmittedResults().size();
        if (state >= initialStates->size())
          return operation->emitOpError("region-scan final-state result has no matching state operand");
        prototype = (*initialStates)[state].getType();
      }
      converted = convertDataType(canonicalAnalysis, logical, operation, prototype, /*owner=*/1, index);
    }
    if (failed(converted))
      return operation->emitOpError("structured result has no physical schema")
          << "; result=" << index << "; logical_type=" << logical;
    results.push_back(*converted);
  }
  if (!scan) {
    for (auto [index, identity] : llvm::enumerate(*identities)) {
      auto projected = projectPositionalValue(builder, location, identity, results[index]);
      if (failed(projected))
        return operation->emitOpError("structured identity cannot adopt its physical accumulator schema");
      (*identities)[index] = *projected;
    }
  }
  gpu::ParameterAttr segment;
  if (region) {
    auto parameter = getOrCreateRegionSegment(operation);
    if (failed(parameter)) return failure();
    segment = *parameter;
  }
  auto created = [&]() -> FailureOr<Operation *> {
    switch (kind) {
    case StructuredOpKind::Reduce:
      return builder.create<gpu::ReduceOp>(location, *sources, *identities,
          *captures, axes).getOperation();
    case StructuredOpKind::Scan: {
      auto logical = cast<intent::ScanOp>(operation);
      return builder.create<gpu::ScanOp>(location, *sources, *identities, *captures,
          logical.getAxis(), logical.getInclusive(), logical.getReverse()).getOperation();
    }
    case StructuredOpKind::RegionFold:
      return builder.create<gpu::RegionFoldOp>(location, results, *sources, *identities,
          *captures, axes.front(), segment.getReference()).getOperation();
    case StructuredOpKind::RegionScan:
      return builder.create<gpu::RegionScanOp>(location,
          TypeRange(results).take_front(schema.getEmittedResults().size()),
          TypeRange(results).drop_front(schema.getEmittedResults().size()),
          *sources, *identities, *initialStates, *captures, axes.front(), segment.getReference()).getOperation();
    }
    return operation->emitOpError("unknown structured operation kind"), failure();
  }();
  if (failed(created)) return failure();
  Operation *target = *created;
  auto physical = cast<StructuredOpInterface>(target);
  if (!region) {
    if (failed(lowerPureRegion(schema.getCombine(), physical.getCombine(),
            physical.getCombineArgumentTypes(), results)))
      return failure();
  } else {
    SmallVector<Type> slices;
    for (auto [formal, value] : llvm::zip(schema.getSummarizeSources(), *sources)) {
      auto tensor = dyn_cast<RankedTensorType>(formal.getType());
      auto prototype = dyn_cast<FragmentType>(value.getType());
      auto converted = tensor && prototype
          ? convertSegmentSliceType(tensor, prototype, axes.front(), segment.getName().getValue())
          : FailureOr<FragmentType>(failure());
      if (failed(converted))
        return operation->emitOpError("region source slice has no physical fragment schema");
      slices.push_back(*converted);
    }
    SmallVector<Type> summaryTypes;
    if (scan) llvm::append_range(summaryTypes, TypeRange(*identities));
    else summaryTypes = results;
    if (failed(lowerPureRegion(*schema.getSummarizeRegion(), *physical.getSummarizeRegion(),
            physical.getSummarizeArgumentTypes(slices), summaryTypes))) return failure();
    if (failed(lowerPureRegion(schema.getCombine(), physical.getCombine(),
            physical.getCombineArgumentTypes(), summaryTypes))) return failure();
    if (scan) {
      SmallVector<Type> stateTypes;
      llvm::append_range(stateTypes, TypeRange(*initialStates));
      if (failed(lowerPureRegion(*schema.getApplyRegion(), *physical.getApplyRegion(),
              physical.getApplyArgumentTypes(), stateTypes))) return failure();
      if (failed(lowerPureRegion(*schema.getEmitRegion(), *physical.getEmitRegion(),
              physical.getEmitArgumentTypes(slices))))
        return failure();
      auto emitted = physical.getEmitYields();
      if (emitted.size() != physical.getEmittedResults().size())
        return operation->emitOpError("region-scan emitter has no complete physical output schema");
      for (auto [index, result, slice] : llvm::enumerate(physical.getEmittedResults(), emitted)) {
        auto assembled = regionAssemblyType(result.getType(), slice.getType());
        if (failed(assembled))
          return operation->emitOpError("region-scan emitted slice cannot define its assembled output relation")
              << "; result_index=" << index << "; slice=" << slice.getType()
              << "; result=" << result.getType();
        result.setType(*assembled);
      }
    }
  }
  mapResults(operation, target);
  return success();
}

LogicalResult ScalarRegionLowering::lower(intent::ContractOp contract) {
  Operation *operation = contract.getOperation();
  Location location = contract.getLoc();
  FailureOr<Value> lhs = get(contract.getLhs());
  FailureOr<Value> rhs = get(contract.getRhs());
  SmallVector<int64_t> lhsReduction, rhsReduction, lhsBatch, rhsBatch;
  axisPairs(contract.getReduce(), lhsReduction, rhsReduction);
  axisPairs(contract.getBatch(), lhsBatch, rhsBatch);
  FailureOr<Type> result = failed(lhs) || failed(rhs)
                               ? FailureOr<Type>(failure())
                                 : convertContractResultType(canonicalAnalysis,
                                     cast<OpResult>(contract.getResult()), *lhs, *rhs);
  if (failed(lhs) || failed(rhs) || failed(result) ||
      (succeeded(lhs) && !isa<gpu::FragmentType>((*lhs).getType())) ||
      (succeeded(rhs) && !isa<gpu::FragmentType>((*rhs).getType())) ||
      (succeeded(result) && !isa<gpu::FragmentType>(*result))) {
    InFlightDiagnostic diagnostic = contract.emitOpError(
        "contract operands/results have no physical fragment schema");
    diagnostic << "; lhs=";
    if (succeeded(lhs))
      diagnostic << (*lhs).getType();
    else
      diagnostic << "unavailable";
    diagnostic << ", rhs=";
    if (succeeded(rhs))
      diagnostic << (*rhs).getType();
    else
      diagnostic << "unavailable";
    diagnostic << ", result=";
    if (succeeded(result))
      diagnostic << *result;
    else
      diagnostic << "unavailable";
    return failure();
  }
  FailureOr<Value> accumulator =
      zeroAccumulator(builder, location, cast<gpu::FragmentType>(*result));
  if (failed(accumulator))
    return contract.emitOpError("contract accumulator dtype is unsupported");
  auto target = builder.create<gpu::ContractOp>(
      location, *result, *lhs, *rhs, *accumulator, lhsReduction,
      rhsReduction, lhsBatch, rhsBatch);
  mapResults(operation, target);
  return success();
}

LogicalResult ScalarRegionLowering::lower(intent::ScaledContractOp contract) {
  Operation *operation = contract.getOperation();
  Location location = contract.getLoc();
  FailureOr<Value> lhs = get(contract.getLhs());
  FailureOr<Value> lhsScale = get(contract.getLhsScale());
  FailureOr<Value> rhs = get(contract.getRhs());
  FailureOr<Value> rhsScale = get(contract.getRhsScale());
  SmallVector<int64_t> lhsReduction, rhsReduction, lhsBatch, rhsBatch;
  axisPairs(contract.getReduce(), lhsReduction, rhsReduction);
  axisPairs(contract.getBatch(), lhsBatch, rhsBatch);
  FailureOr<Type> result = failed(lhs) || failed(rhs)
                               ? FailureOr<Type>(failure())
                                 : convertContractResultType(canonicalAnalysis,
                                     cast<OpResult>(contract.getResult()), *lhs, *rhs);
  if (failed(lhs) || failed(lhsScale) || failed(rhs) || failed(rhsScale) ||
      failed(result) || !isa<gpu::FragmentType>((*lhs).getType()) ||
      !isa<gpu::FragmentType>((*lhsScale).getType()) ||
      !isa<gpu::FragmentType>((*rhs).getType()) ||
      !isa<gpu::FragmentType>((*rhsScale).getType()) ||
      !isa<gpu::FragmentType>(*result))
    return contract.emitOpError(
        "scaled contract operands/results have no physical fragment schema");
  FailureOr<Value> accumulator =
      zeroAccumulator(builder, location, cast<gpu::FragmentType>(*result));
  if (failed(accumulator))
    return contract.emitOpError(
        "scaled contract accumulator dtype is unsupported");
  auto target = builder.create<gpu::ScaledContractOp>(
      location, *result, *lhs, *lhsScale, *rhs, *rhsScale, *accumulator,
      lhsReduction, rhsReduction, lhsBatch, rhsBatch,
      contract.getLhsFormat(), contract.getRhsFormat(),
      contract.getLhsGroupSize(), contract.getRhsGroupSize());
  mapResults(operation, target);
  return success();
}

LogicalResult ScalarRegionLowering::lower(intent::SparseContractOp contract) {
  Operation *operation = contract.getOperation();
  Location location = contract.getLoc();
  FailureOr<Value> compressed = get(contract.getCompressed());
  FailureOr<Value> metadata = get(contract.getMetadata());
  FailureOr<Value> rhs = get(contract.getRhs());
  FailureOr<Value> logicalExtent = get(contract.getLogicalExtent());
  SmallVector<int64_t> lhsReduction, rhsReduction, lhsBatch, rhsBatch;
  axisPairs(contract.getReduce(), lhsReduction, rhsReduction);
  axisPairs(contract.getBatch(), lhsBatch, rhsBatch);
  FailureOr<Type> result = failed(compressed) || failed(rhs)
                               ? FailureOr<Type>(failure())
                               : convertContractResultType(canonicalAnalysis,
                                     cast<OpResult>(contract.getResult()), *compressed, *rhs);
  if (failed(compressed) || failed(metadata) || failed(rhs) ||
      failed(logicalExtent) || failed(result) ||
      !isa<gpu::FragmentType>((*compressed).getType()) ||
      !isa<gpu::FragmentType>((*rhs).getType()) ||
      !isa<gpu::FragmentType>(*result))
    return contract.emitOpError(
        "sparse contract operands/results have no physical schema");
  FailureOr<Value> accumulator =
      zeroAccumulator(builder, location, cast<gpu::FragmentType>(*result));
  if (failed(accumulator))
    return contract.emitOpError(
        "sparse contract accumulator dtype is unsupported");
  auto format = contract.getFormat();
  auto target = builder.create<gpu::SparseContractOp>(
      location, *result, *compressed, *metadata, *rhs, *logicalExtent,
      *accumulator, format,
      lhsReduction, rhsReduction, lhsBatch, rhsBatch);
  mapResults(operation, target);
  return success();
}

LogicalResult ScalarRegionLowering::lower(intent::HistogramOp histogram) {
  Operation *operation = histogram.getOperation();
  Location location = histogram.getLoc();
  FailureOr<Value> values = get(histogram.getValues());
  FailureOr<Value> bins = get(histogram.getBins());
  FailureOr<Value> valid = get(histogram.getValid());
  FailureOr<Type> result =
      convertDataType(canonicalAnalysis, histogram.getResult().getType(), operation);
  if (failed(values) || failed(bins) || failed(valid) || failed(result) ||
      !isa<gpu::FragmentType>((*values).getType()) ||
      !isa<gpu::FragmentType>((*valid).getType()) ||
      !isa<gpu::FragmentType>(*result))
    return histogram.emitOpError(
        "histogram operands/results have no physical fragment schema");
  auto target = builder.create<gpu::HistogramOp>(
      location, *result, *values, *bins, *valid);
  mapResults(operation, target);
  return success();
}

} // namespace intent::kir_to_gpu
