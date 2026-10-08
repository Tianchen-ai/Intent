#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/IR/ProgramInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseSet.h"
#include "Verification.h"

using namespace mlir;
namespace intent::gpu {
using namespace operation_detail;

namespace {

LogicalResult verifySegmentSlice(Operation *owner, Type sourceType,
                                 Type sliceType, uint64_t axis,
                                 ParameterRefAttr segment) {
  auto declaration = lookupParameterDeclaration(owner, segment);
  if (!declaration || !declaration.isExtent())
    return owner->emitOpError("region segment requires a declared positive index parameter");
  auto source = dyn_cast<FragmentType>(sourceType);
  auto slice = dyn_cast<FragmentType>(sliceType);
  if (!source || !slice || source.getElementType() != slice.getElementType() ||
      source.getOwner() != slice.getOwner() ||
      source.getShape().size() != slice.getShape().size() ||
      axis >= source.getShape().size())
    return owner->emitOpError(
        "physical region source slice lost rank/type/coordinate mapping");
  for (unsigned dimension = 0; dimension < source.getShape().size(); ++dimension) {
    if (dimension == axis) {
      auto extent = dyn_cast<PhysicalExprAttr>(slice.getShape()[dimension]);
      if (!extent ||
          extent.getKind() !=
              PhysicalExprKind::Parameter ||
          extent.getParameterReference() != segment)
        return owner->emitOpError(
                   "physical region slice axis is not bound to its segment parameter: slice_extent=")
               << slice.getShape()[dimension]
               << ", segment=" << segment.getName();
    } else if (source.getShape()[dimension] != slice.getShape()[dimension] ||
               source.getAxisMaps()[dimension] !=
                   slice.getAxisMaps()[dimension]) {
      InFlightDiagnostic diagnostic = owner->emitOpError(
          "physical region slice changed a non-segment extent or coordinate mapping");
      diagnostic << "; axis=" << dimension
                 << ", source_extent=" << source.getShape()[dimension]
                 << ", slice_extent=" << slice.getShape()[dimension]
                 << ", source_mapping=" << source.getAxisMaps()[dimension]
                 << ", slice_mapping=" << slice.getAxisMaps()[dimension];
      return failure();
    }
  }
  return success();
}

bool preservesSliceAssembly(Type slice, Type result) {
  if (auto sliceFragment = dyn_cast<FragmentType>(slice)) {
    auto resultFragment = dyn_cast<FragmentType>(result);
    return resultFragment &&
           sliceFragment.getElementType() == resultFragment.getElementType() &&
           sliceFragment.getShape().size() ==
               resultFragment.getShape().size() &&
           sliceFragment.getAxisMaps() == resultFragment.getAxisMaps();
  }
  auto sliceRecord = dyn_cast<RecordType>(slice);
  auto resultRecord = dyn_cast<RecordType>(result);
  if (!sliceRecord || !resultRecord ||
      sliceRecord.getFieldNames() != resultRecord.getFieldNames() ||
      sliceRecord.getFieldTypes().size() !=
          resultRecord.getFieldTypes().size())
    return false;
  return llvm::all_of(
      llvm::zip(sliceRecord.getFieldTypes(), resultRecord.getFieldTypes()),
      [](auto fields) {
        return preservesSliceAssembly(
            cast<TypeAttr>(std::get<0>(fields)).getValue(),
            cast<TypeAttr>(std::get<1>(fields)).getValue());
      });
}

LogicalResult verifyReduceLike(StructuredOpInterface operation) {
  SmallVector<Type> accumulators;
  for (auto [identity, result] : llvm::zip_equal(operation.getIdentities(), operation->getResults())) {
    if (identity.getType() != result.getType())
      return operation->emitOpError("physical identity/result type disagrees");
    accumulators.push_back(identity.getType());
  }
  return verifyHelperRegion(operation, operation.getCombine(),
                            operation.getCombineArgumentTypes(), accumulators);
}

bool typeCarriesAxis(Type type, int64_t axis) {
  if (auto fragment = dyn_cast<FragmentType>(type))
    return axis >= 0 &&
           axis < static_cast<int64_t>(fragment.getShape().size());
  auto record = dyn_cast<RecordType>(type);
  return record && llvm::all_of(record.getFieldTypes(), [&](Attribute field) {
           return typeCarriesAxis(cast<TypeAttr>(field).getValue(), axis);
         });
}

LogicalResult verifyReductionResultRelations(Operation *owner,
                                              ValueRange sources,
                                              ResultRange results,
                                              ArrayRef<int64_t> axes) {
  if (sources.size() != results.size())
    return owner->emitOpError("physical reduction needs one result relation per source component");
  for (auto [source, result] : llvm::zip_equal(sources, results)) {
    auto expected = inferCollectiveResultType(source.getType(), axes,
                                               elementType(result.getType()));
    if (failed(expected) || result.getType() != *expected)
      return owner->emitOpError("physical reduction result does not preserve source free axes")
             << "; source=" << source.getType() << "; result=" << result.getType();
  }
  return success();
}

} // namespace

FailureOr<Type> inferCollectiveResultType(Type source, ArrayRef<int64_t> reducedAxes,
                                         Type resultElement) {
  auto fragment = dyn_cast<FragmentType>(source);
  if (!fragment) return failure();
  llvm::SmallDenseSet<int64_t> reduced;
  for (int64_t axis : reducedAxes)
    if (axis < 0 || axis >= static_cast<int64_t>(fragment.getShape().size()) ||
        !reduced.insert(axis).second)
      return failure();
  SmallVector<Attribute> shape, mappings;
  for (auto [axis, extent] : llvm::enumerate(fragment.getShape())) {
    if (reduced.contains(axis)) continue;
    shape.push_back(extent);
    auto mapping = cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
    mappings.push_back(AxisMapAttr::get(source.getContext(), mapping.getSourceId(),
        mapping.getSourceAxis(), mapping.getDimensionId(), mappings.size(),
        mapping.getDerived()));
  }
  if (shape.empty()) return resultElement;
  return Type(FragmentType::get(source.getContext(), resultElement,
      ArrayAttr::get(source.getContext(), shape), ArrayAttr::get(source.getContext(), mappings),
      fragment.getValidity(), fragment.getOwner()));
}

LogicalResult inferScalarCollectiveResultTypes(std::optional<Location> location,
    ValueRange sources, int64_t axis, bool scan, SmallVectorImpl<Type> &results) {
  if (sources.empty()) return emitOptionalError(location, "native collective requires a source");
  auto first = dyn_cast<FragmentType>(sources.front().getType());
  bool allAxes = !scan && axis == -1;
  if (!first || (!allAxes && (axis < 0 || axis >= static_cast<int64_t>(first.getShape().size()))))
    return emitOptionalError(location, "collective axis is outside its source fragment");
  SmallVector<int64_t, 1> axes;
  if (allAxes)
    for (int64_t i = 0; i < static_cast<int64_t>(first.getShape().size()); ++i)
      axes.push_back(i);
  else if (!scan) axes.push_back(axis);
  for (Value value : sources) {
    auto source = dyn_cast<FragmentType>(value.getType());
    if (!source || source.getShape() != first.getShape())
      return emitOptionalError(location, "native collective requires equal source fragment shapes");
    auto expected = inferCollectiveResultType(source, axes, source.getElementType());
    if (failed(expected)) return failure();
    results.push_back(*expected);
  }
  return success();
}

LogicalResult verifyScalarCollective(Operation *owner, ValueRange sources,
                                     ValueRange identities, ResultRange results,
                                     Region &combine, int64_t axis, bool scan) {
  unsigned count = sources.size();
  if (!count || identities.size() != count || results.size() != count ||
      !llvm::hasSingleElement(combine))
    return owner->emitOpError(
        "requires paired sources/identities and one scalar combine block");
  SmallVector<Type> expected;
  if (failed(inferScalarCollectiveResultTypes(owner->getLoc(), sources, axis, scan, expected)))
    return failure();
  Block &block = combine.front();
  if (block.empty())
    return owner->emitOpError("scalar combine block must yield its results");
  auto yield = dyn_cast<YieldOp>(block.getTerminator());
  if (block.getNumArguments() != 2 * count || !yield ||
      yield.getValues().size() != count)
    return owner->emitOpError("scalar combine arity disagrees with sources");
  for (unsigned i = 0; i < count; ++i) {
    auto source = dyn_cast<FragmentType>(sources[i].getType());
    if (identities[i].getType() != results[i].getType())
      return owner->emitOpError(
          "native collective requires equal source shapes and exact identity/result types");
    Type element = source.getElementType();
    if (block.getArgument(i).getType() != element ||
        block.getArgument(count + i).getType() != element ||
        yield.getValues()[i].getType() != element)
      return owner->emitOpError(
          "callback arguments and yields must be source element types");
    if (results[i].getType() != expected[i])
      return owner->emitOpError(
          "collective result must preserve its source free-axis relation");
  }
  for (Operation &operation : block) {
    if (operation.getNumRegions())
      return owner->emitOpError(
          "native callback must be a closed elementwise block");
    for (Value result : operation.getResults())
      if (isa<FragmentType>(result.getType()))
        return owner->emitOpError(
            "native callback still contains a fragment value");
  }
  return success();
}

LogicalResult ReduceOp::inferReturnTypes(MLIRContext *context,
    std::optional<Location> location, ValueRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions, SmallVectorImpl<Type> &results) {
  Adaptor adaptor(operands, attributes, properties, regions);
  if (adaptor.getSources().empty() || adaptor.getSources().size() != adaptor.getIdentities().size())
    return failure();
  for (auto [source, identity] : llvm::zip_equal(adaptor.getSources(), adaptor.getIdentities())) {
    auto type = inferCollectiveResultType(source.getType(), adaptor.getAxes(), elementType(identity.getType()));
    if (failed(type)) return failure();
    results.push_back(*type);
  }
  return success();
}

LogicalResult ScanOp::inferReturnTypes(MLIRContext *context,
    std::optional<Location> location, ValueRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions, SmallVectorImpl<Type> &results) {
  Adaptor adaptor(operands, attributes, properties, regions);
  if (adaptor.getSources().empty() || adaptor.getSources().size() != adaptor.getIdentities().size())
    return failure();
  llvm::append_range(results, adaptor.getSources().getTypes());
  return success();
}

LogicalResult ReduceOp::verify() {
  auto structured = cast<StructuredOpInterface>(getOperation());
  if (failed(verifyStructuredArity(structured))) return failure();
  if (getAxes().empty()) return emitOpError("physical reduce requires at least one axis");
  llvm::DenseSet<int64_t> axes;
  for (int64_t axis : getAxes()) {
    if (!axes.insert(axis).second) return emitOpError("physical reduce axes must be unique");
    for (Value source : getSources())
      if (!typeCarriesAxis(source.getType(), axis))
        return emitOpError("physical reduce axis is outside a source schema");
  }
  if (failed(verifyReductionResultRelations(getOperation(), getSources(), getResults(), getAxes())))
    return failure();
  return verifyReduceLike(structured);
}

LogicalResult ScanOp::verify() {
  auto structured = cast<StructuredOpInterface>(getOperation());
  if (failed(verifyStructuredArity(structured))) return failure();
  for (auto [source, result] : llvm::zip_equal(getSources(), getResults())) {
    if (source.getType() != result.getType())
      return emitOpError("physical scan result must preserve its source type");
    if (!typeCarriesAxis(source.getType(), getAxis()))
      return emitOpError("physical scan axis is outside a source schema");
  }
  return verifyReduceLike(structured);
}

LogicalResult RegionFoldOp::verify() {
  auto structured = cast<StructuredOpInterface>(getOperation());
  if (failed(verifyStructuredArity(structured))) return failure();
  SmallVector<Type> summaries;
  for (auto [identity, result] : llvm::zip_equal(getIdentities(), getResults())) {
    if (identity.getType() != result.getType())
      return emitOpError("region-fold identity/result type disagrees: identity=")
             << identity.getType() << ", result=" << result.getType();
    summaries.push_back(identity.getType());
  }
  SmallVector<Type> slices;
  for (auto [source, slice] : llvm::zip_equal(getSources(), structured.getSummarizeSources())) {
    if (failed(verifySegmentSlice(getOperation(), source.getType(), slice.getType(), getAxis(), getSegment())))
      return failure();
    slices.push_back(slice.getType());
  }
  if (failed(verifyHelperRegion(getOperation(), getSummarize(),
                                structured.getSummarizeArgumentTypes(slices), summaries))) return failure();
  return verifyHelperRegion(getOperation(), getCombine(),
                            structured.getCombineArgumentTypes(), summaries);
}

LogicalResult RegionScanOp::verify() {
  auto structured = cast<StructuredOpInterface>(getOperation());
  if (failed(verifyStructuredArity(structured))) return failure();
  SmallVector<Type> slices, transitions, states;
  for (auto [source, slice] : llvm::zip_equal(getSources(), structured.getSummarizeSources())) {
    if (failed(verifySegmentSlice(getOperation(), source.getType(), slice.getType(), getAxis(), getSegment())))
      return failure();
    slices.push_back(slice.getType());
  }
  for (Value identity : getIdentities()) transitions.push_back(identity.getType());
  for (auto [initial, result] : llvm::zip_equal(getInitialStates(), getFinalStates())) {
    if (initial.getType() != result.getType())
      return emitOpError("region-scan final-state type disagrees: input=")
             << initial.getType() << ", result=" << result.getType();
    states.push_back(initial.getType());
  }
  if (failed(verifyHelperRegion(getOperation(), getSummarize(),
                                structured.getSummarizeArgumentTypes(slices), transitions))) return failure();
  if (failed(verifyHelperRegion(getOperation(), getCombine(),
                                structured.getCombineArgumentTypes(), transitions))) return failure();
  if (failed(verifyHelperRegion(getOperation(), getApply(),
                                structured.getApplyArgumentTypes(), states))) return failure();
  SmallVector<Type> emitted(structured.getEmitYields().getTypes());
  if (failed(verifyHelperRegion(getOperation(), getEmit(),
                                structured.getEmitArgumentTypes(slices), emitted))) return failure();
  for (auto [slice, result] : llvm::zip_equal(emitted, getEmittedResults()))
    if (!preservesSliceAssembly(slice, result.getType()))
      return emitOpError("region-scan output assembly relation is invalid");
  return success();
}

LogicalResult HistogramOp::verify() {
  if (!getValues().getType().getElementType().isIntOrIndex() ||
      !getValid().getType().getElementType().isInteger(1) ||
      !isa<IntegerType>(getResult().getType().getElementType()) ||
      getValues().getType().getShape() != getValid().getType().getShape() ||
      getResult().getType().getShape().size() != 1)
    return emitOpError("histogram physical schema is invalid");
  return success();
}

} // namespace intent::gpu
