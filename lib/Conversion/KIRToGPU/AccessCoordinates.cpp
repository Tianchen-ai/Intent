#include "Construction.h"
#include "Intent/Conversion/IndexedAccess.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Control/Traversal.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;

namespace intent::kir_to_gpu {

FailureOr<unsigned> ScalarRegionLowering::physicalResourceAxis(
    Type logicalResource,
    Type physicalResource,
    unsigned logicalAxis) {
  RankedTensorType logicalTensor;
  if (auto view = dyn_cast<intent::ViewType>(logicalResource))
    logicalTensor = dyn_cast<RankedTensorType>(view.getTensor());
  else if (auto buffer = dyn_cast<intent::BufferType>(logicalResource))
    logicalTensor = dyn_cast<RankedTensorType>(buffer.getTensor());
  else
    logicalTensor = dyn_cast<RankedTensorType>(logicalResource);
  if (!logicalTensor || logicalAxis >= static_cast<unsigned>(logicalTensor.getRank()))
    return failure();

  unsigned physicalRank = 0;
  if (auto view = dyn_cast<gpu::ViewType>(physicalResource))
    physicalRank = view.getRank();
  else if (auto buffer = dyn_cast<gpu::BufferType>(physicalResource))
    physicalRank = buffer.getShape().size();
  else if (auto fragment = dyn_cast<gpu::FragmentType>(physicalResource))
    physicalRank = fragment.getShape().size();
  else
    return failure();
  unsigned logicalRank = logicalTensor.getRank();
  if (physicalRank < logicalRank)
    return failure();
  if (!isa<gpu::FragmentType>(physicalResource) && physicalRank != logicalRank)
    return failure();
  return physicalRank - logicalRank + logicalAxis;
}

FailureOr<Value> ScalarRegionLowering::resourceExtent(
    Location location,
    Value resource,
    unsigned axis) {
  PhysicalExprAttr extent;
  if (auto view = dyn_cast<gpu::ViewType>(resource.getType())) {
    if (axis >= view.getRank())
      return failure();
    extent = cast<PhysicalExprAttr>(view.getLayout().getExtents()[axis]);
    if (extent.getKind() ==
        PhysicalExprKind::Constant)
      return physicalExtentValue(location, extent);
    return Value(builder.create<gpu::DimOp>(location, builder.getIndexType(),
                                            resource, axis));
  } else if (auto buffer = dyn_cast<gpu::BufferType>(resource.getType())) {
    if (axis >= buffer.getShape().size())
      return failure();
    extent = cast<PhysicalExprAttr>(buffer.getShape()[axis]);
  } else if (auto fragment = dyn_cast<gpu::FragmentType>(resource.getType())) {
    if (axis >= fragment.getShape().size())
      return failure();
    gpu::PhysicalProgramAnalysis analysis(physicalKernel);
    if (analysis.axisRealization(resource, axis).constructionScalarSeed) {
      FailureOr<gpu::MakeRangeOp> range =
          gpu::queryExactLogicalRange(analysis.axisRanges(resource, axis));
      if (failed(range))
        return failure();
      return rangeExtent(location, range->getLogicalStart(),
                         range->getLogicalStop(), range->getStep());
    }
    extent = cast<PhysicalExprAttr>(fragment.getShape()[axis]);
  } else {
    return failure();
  }
  return physicalExtentValue(location, extent);
}

FailureOr<SmallVector<Value>> ScalarRegionLowering::accessCoordinates(
    Operation *operation, SmallVectorImpl<int64_t> &sourceAxes) {
  auto relation = canonicalAnalysis.indexRelation(operation);
  if (failed(relation))
    return failure();
  SmallVector<Value> coordinates;
  FailureOr<Value> resource = get(relation->source);
  if (failed(resource))
    return failure();
  Value stored = cast<IndexedAccessOpInterface>(operation).getStoredValue();
  gpu::FragmentType valuePrototype;
  if (isa<ViewStoreOp, BufferStoreOp, ScatterUniqueOp, ScatterReduceOp>(operation)) {
    FailureOr<Value> value = get(stored);
    if (succeeded(value))
      valuePrototype = dyn_cast<gpu::FragmentType>((*value).getType());
  }
  auto resultExtent = [&](size_t axis,
                          PhysicalExprAttr fallback)
      -> FailureOr<PhysicalExprAttr> {
    if (valuePrototype) {
      size_t resultRank = relation->resultDimensionIdentities.size();
      if (valuePrototype.getShape().size() < resultRank)
        return failure();
      // A scalar logical index may already have been promoted to a physical
      // fragment.  Those ownership axes lead the value fragment but are not
      // part of the logical access result.  Relation result axis zero must
      // therefore address the first axis after that exact lifted prefix.
      size_t liftedRank = valuePrototype.getShape().size() - resultRank;
      size_t prototypeAxis = liftedRank + axis;
      if (prototypeAxis < valuePrototype.getShape().size())
        return cast<PhysicalExprAttr>(
            valuePrototype.getShape()[prototypeAxis]);
    }
    if (operation->getNumResults() == 0)
      return fallback;
    auto tensor = dyn_cast<RankedTensorType>(operation->getResult(0).getType());
    if (!tensor || axis >= static_cast<size_t>(tensor.getRank()))
      return fallback;
    if (!tensor.isDynamicDim(axis))
      return expression(operation->getContext(), PhysicalExprKind::Constant,
                        tensor.getDimSize(axis));
    FailureOr<PhysicalExprAttr> extent =
        fragmentExtentExpression(canonicalAnalysis, tensor, operation, axis);
    return extent;
  };
  auto makeRange = [&](unsigned sourceAxis, Value start, Value stop,
                       Value step, uint64_t sourceId,
                       int64_t dimensionId,
                       bool derived,
                       PhysicalExprAttr physicalExtent) -> FailureOr<Value> {
    FailureOr<Value> physicalStart = asIndex(operation->getLoc(), start);
    FailureOr<Value> physicalStop = asIndex(operation->getLoc(), stop);
    FailureOr<Value> physicalStep = asIndex(operation->getLoc(), step);
    if (failed(physicalStart) || failed(physicalStop) || failed(physicalStep))
      return failure();
    start = *physicalStart;
    stop = *physicalStop;
    step = *physicalStep;
    FailureOr<Value> extent =
        physicalExtentValue(operation->getLoc(), physicalExtent);
    if (failed(extent))
      return failure();
    auto type = fragmentType(operation->getContext(), builder.getIndexType(),
                             {physicalExtent},
                             {{sourceId, sourceAxis, dimensionId, derived}});
    return Value(builder.create<gpu::MakeRangeOp>(
        operation->getLoc(), type, start, *extent, step, start, stop, sourceId,
        sourceAxis, derived));
  };
  Location location = operation->getLoc();
  IndexTermMaterialization materialization;
  materialization.constant = [&](int64_t value) -> Value {
    return builder.create<arith::ConstantIndexOp>(location, value);
  };
  materialization.scalar = [&](Value logical) -> FailureOr<Value> {
    auto value = get(logical);
    return failed(value) ? FailureOr<Value>(failure())
                         : asLogicalIndex(location, *value);
  };
  materialization.extent = [&](Value logical, unsigned axis) -> FailureOr<Value> {
    return logicalExtent(location, logical, axis);
  };
  materialization.domain = [&](Value logical) -> FailureOr<IndexRange> {
    auto range = get(logical);
    if (failed(range) || !isa<gpu::RangeType>(range->getType())) return failure();
    Value begin = rangeBound(location, *range, 0);
    Value end = rangeBound(location, *range, 1);
    Value step = rangeBound(location, *range, 2);
    return IndexRange{begin, end, step, rangeExtent(location, begin, end, step)};
  };
  auto binary = [&](BinaryOperator kind, Value lhs, Value rhs) -> FailureOr<Value> {
    return createBinary(builder, location, builder.getIndexType(), lhs, rhs, kind);
  };
  materialization.add = [&](Value lhs, Value rhs) { return binary(BinaryOperator::Add, lhs, rhs); };
  materialization.subtract = [&](Value lhs, Value rhs) { return binary(BinaryOperator::Subtract, lhs, rhs); };
  materialization.multiply = [&](Value lhs, Value rhs) { return binary(BinaryOperator::Multiply, lhs, rhs); };
  materialization.maximum = [&](Value lhs, Value rhs) { return binary(BinaryOperator::Maximum, lhs, rhs); };
  materialization.ceilDivide = [&](Value lhs, Value rhs) -> FailureOr<Value> {
    return ceilDivide(location, lhs, rhs);
  };
  for (const IndexTermFact &term : relation->terms) {
    if (!term.sourceAxis)
      continue;
    unsigned sourceAxis = *term.sourceAxis;
    unsigned resultAxis = term.resultAxes.empty() ? 0 : term.resultAxes.front();
    FailureOr<unsigned> physicalSourceAxis = physicalResourceAxis(
        relation->source.getType(), (*resource).getType(), sourceAxis);
    if (failed(physicalSourceAxis))
      return failure();
    sourceAxes.push_back(*physicalSourceAxis);
    auto bindings = materialization;
    if (term.kind == 0)
      if (auto fragment = dyn_cast<gpu::FragmentType>(resource->getType()))
        // The full window of an existing fragment consumes its current physical
        // value. AT and SLICE still bind the logical source extent above.
        bindings.extent = [&, fragment](Value, unsigned) {
          return physicalExtentValue(location,
              cast<PhysicalExprAttr>(fragment.getShape()[*physicalSourceAxis]));
        };
    auto reified = materializeIndexTerm(relation->source, term, bindings);
    if (failed(reified))
      return operation->emitOpError("index term has no coordinate materialization"), failure();
    if (term.kind == 0) {
      auto view = dyn_cast<gpu::ViewType>((*resource).getType());
      auto buffer = dyn_cast<gpu::BufferType>((*resource).getType());
      auto fragment = dyn_cast<gpu::FragmentType>((*resource).getType());
      if ((!view && !buffer && !fragment) ||
          (view && *physicalSourceAxis >= view.getRank()) ||
          (buffer && *physicalSourceAxis >= buffer.getShape().size()) ||
          (fragment &&
           *physicalSourceAxis >= fragment.getShape().size())) {
        operation->emitOpError(
            "full-slice term has no matching physical source axis");
        return failure();
      }
      Value start = reified->range->begin;
      Value stop;
      uint64_t sourceId;
      PhysicalExprAttr extent;
      uint32_t logicalSourceAxis = sourceAxis;
      bool derived = false;
      if (view || buffer) {
        // A buffer is a canonical value result, using the same derived
        // identity convention as resultAxisIdentity.
        sourceId = view ? view.getSourceId() : buffer.getInstance() + 1;
        derived = static_cast<bool>(buffer);
        extent = cast<PhysicalExprAttr>(
            view ? view.getLayout().getExtents()[*physicalSourceAxis]
                 : buffer.getShape()[*physicalSourceAxis]);
        FailureOr<Value> physicalStop =
            physicalExtentValue(operation->getLoc(), extent);
        if (failed(physicalStop)) {
          operation->emitOpError(
              "resource full-slice extent is not materialized in the current program: ")
              << extent;
          return failure();
        }
        stop = *physicalStop;
      } else {
        auto mapping =
            cast<gpu::AxisMapAttr>(
                fragment.getAxisMaps()[*physicalSourceAxis]);
        FailureOr<Value> physicalExtent = physicalExtentValue(
            operation->getLoc(),
            cast<PhysicalExprAttr>(
                fragment.getShape()[*physicalSourceAxis]));
        if (failed(physicalExtent)) {
          operation->emitOpError(
              "fragment full-slice extent is not materialized in the current program: ")
              << fragment.getShape()[*physicalSourceAxis];
          return failure();
        }
        stop = *physicalExtent;
        sourceId = mapping.getSourceId();
        logicalSourceAxis = mapping.getSourceAxis();
        derived = mapping.getDerived();
        extent =
            cast<PhysicalExprAttr>(
                fragment.getShape()[*physicalSourceAxis]);
      }
      int64_t dimension = 0;
      if (view) {
        auto dimensions = view.getLayout().getDimensionIds();
        if (*physicalSourceAxis >= dimensions.size())
          return failure();
        dimension = dimensions[*physicalSourceAxis];
      } else if (buffer) {
        dimension = relation->resultDimensionIdentities[resultAxis];
      } else {
        dimension = cast<gpu::AxisMapAttr>(
                        fragment.getAxisMaps()[*physicalSourceAxis])
                        .getDimensionId();
      }
      if (dimension <= 0)
        return failure();
      // A full slice of an already-physical fragment consumes that
      // fragment's current extent.  Reconstructing the helper-local logical
      // result dimension here would discard an enclosing region segment.
      if (view || buffer) {
        FailureOr<PhysicalExprAttr> resultPhysicalExtent =
            resultExtent(resultAxis, extent);
        if (failed(resultPhysicalExtent))
          return failure();
        extent = *resultPhysicalExtent;
      }
      Value step = reified->range->step;
      FailureOr<Value> coordinate = makeRange(
          logicalSourceAxis, start, stop, step, sourceId, dimension, derived,
          extent);
      if (failed(coordinate))
        return failure();
      coordinates.push_back(*coordinate);
      continue;
    }
    if (reified->coordinate || reified->tensor) {
      Value physicalCoordinate = reified->coordinate;
      if (reified->tensor) {
        auto value = get(reified->tensor);
        if (failed(value)) return failure();
        physicalCoordinate = *value;
      }
      if (auto fragment = dyn_cast<gpu::FragmentType>(physicalCoordinate.getType())) {
        auto logicalCoordinate = reified->tensor
            ? dyn_cast<RankedTensorType>(reified->tensor.getType()) : RankedTensorType();
        if (logicalCoordinate) {
          if (fragment.getShape().size() < term.indexAxes.size())
            return failure();
        }
        unsigned prefix = logicalCoordinate
            ? fragment.getShape().size() - term.indexAxes.size() : 0;
        SmallVector<Attribute> mappings;
        for (auto [axis, attribute] : llvm::enumerate(fragment.getAxisMaps())) {
          auto mapping = cast<gpu::AxisMapAttr>(attribute);
          int64_t dimension = mapping.getDimensionId();
          if (logicalCoordinate && axis >= prefix)
            dimension = relation->resultDimensionIdentities[
                term.indexAxes[axis - prefix]];
          mappings.push_back(gpu::AxisMapAttr::get(
              operation->getContext(), mapping.getSourceId(),
              mapping.getSourceAxis(), dimension,
              mappings.size(), mapping.getDerived()));
        }
        auto target = gpu::FragmentType::get(
            operation->getContext(), fragment.getElementType(),
            fragment.getShape(), builder.getArrayAttr(mappings),
            fragment.getValidity(), fragment.getOwner());
        if (target != fragment) {
          auto reassociation = gpu::inferReshapeReassociation(fragment, target);
          if (failed(reassociation))
            return failure();
          physicalCoordinate = builder.create<gpu::ReshapeOp>(
              operation->getLoc(), target, physicalCoordinate, *reassociation);
        }
      }
      FailureOr<Value> logicalIndex =
          asLogicalIndex(operation->getLoc(), physicalCoordinate);
      if (failed(logicalIndex))
        return failure();
      coordinates.push_back(*logicalIndex);
      continue;
    }
    if (term.kind == 4) {
      FailureOr<Value> range = get(term.operands[0]);
      if (failed(range) || !isa<gpu::RangeType>((*range).getType()))
        return failure();
      Value start = reified->range->begin;
      Value stop = reified->range->end;
      Value step = reified->range->step;
      auto rangeType = cast<gpu::RangeType>((*range).getType());
      PhysicalExprAttr fallback = expression(
          operation->getContext(), PhysicalExprKind::Constant, 1);
      FailureOr<PhysicalExprAttr> physicalExtent =
          resultExtent(resultAxis, fallback);
      if (failed(physicalExtent))
        return failure();
      PhysicalExprAttr extent = *physicalExtent;
      uint64_t sourceId = rangeType.getSourceId();
      uint32_t logicalSourceAxis = rangeType.getSourceAxis();
      bool derived = rangeType.getDerived();
      auto fragment = dyn_cast<gpu::FragmentType>((*resource).getType());
      auto logical =
          dyn_cast<RankedTensorType>(relation->source.getType());
      Value logicalValue = relation->source;
      const bool fragmentIndex = static_cast<bool>(fragment);
      unsigned logicalAxis = sourceAxis;
      unsigned fragmentAxis = *physicalSourceAxis;
      if (!fragment && valuePrototype) {
        auto valueType = dyn_cast<RankedTensorType>(stored.getType());
        if (valueType &&
            valueType.getRank() == relation->resultDimensionIdentities.size()) {
          fragment = valuePrototype;
          logical = valueType;
          logicalValue = stored;
          logicalAxis = resultAxis;
          fragmentAxis =
              fragment.getShape().size() - valueType.getRank() + resultAxis;
        }
      }
      if (fragment && logical && integerConstant(start) == 0 &&
          integerConstant(step) == 1) {
        DenseI64ArrayAttr identities = dimensionIds(logical);
        auto mapping = cast<gpu::AxisMapAttr>(
            fragment.getAxisMaps()[fragmentAxis]);
        bool coversSource = false;
        if (logical.isDynamicDim(logicalAxis)) {
          coversSource = canonicalAnalysis.equalTensorExtents(
              term.operands[0], 0, logicalValue, logicalAxis);
        } else {
          coversSource = integerConstant(stop) == logical.getDimSize(logicalAxis);
        }
        if (coversSource && identities &&
            mapping.getDimensionId() == identities[logicalAxis] &&
            rangeType.getDimensionId() == identities[logicalAxis]) {
          // Full logical indexing preserves the value's axis relation. Reads
          // use local fragment coordinates; writes retain logical bounds so
          // ownership advances their coordinates with the producing tile.
          sourceId = mapping.getSourceId();
          logicalSourceAxis = mapping.getSourceAxis();
          derived = mapping.getDerived();
          extent = cast<PhysicalExprAttr>(fragment.getShape()[fragmentAxis]);
          if (fragmentIndex) {
            FailureOr<Value> fragmentStop =
                physicalExtentValue(operation->getLoc(), extent);
            if (failed(fragmentStop))
              return failure();
            stop = *fragmentStop;
          }
        }
      }
      FailureOr<Value> coordinate = makeRange(
          logicalSourceAxis, start, stop, step, sourceId,
          rangeType.getDimensionId(), derived, extent);
      if (failed(coordinate))
        return failure();
      for (StringRef name : {gpu::sourceSubregionAttr,
                             gpu::sourceSubregionBoundAttr})
        if (Attribute value = (*range).getDefiningOp()->getAttr(name))
          coordinate->getDefiningOp()->setAttr(name, value);
      coordinates.push_back(*coordinate);
      continue;
    }
    if (reified->range) {
      auto view = dyn_cast<gpu::ViewType>((*resource).getType());
      auto buffer = dyn_cast<gpu::BufferType>((*resource).getType());
      auto fragment = dyn_cast<gpu::FragmentType>((*resource).getType());
      if ((!view && !buffer && !fragment) ||
          (view && *physicalSourceAxis >= view.getRank()) ||
          (buffer && *physicalSourceAxis >= buffer.getShape().size()) ||
          (fragment &&
           *physicalSourceAxis >= fragment.getShape().size()))
        return failure();
      PhysicalExprAttr fallback = cast<PhysicalExprAttr>(
          view     ? view.getLayout().getExtents()[*physicalSourceAxis]
          : buffer ? buffer.getShape()[*physicalSourceAxis]
                   : fragment.getShape()[*physicalSourceAxis]);
      FailureOr<PhysicalExprAttr> physicalExtent =
          resultExtent(resultAxis, fallback);
      if (failed(physicalExtent))
        return failure();
      PhysicalExprAttr extent = *physicalExtent;
      uint64_t sourceId;
      uint32_t logicalSourceAxis = sourceAxis;
      int64_t dimension = 0;
      bool derived = false;
      if (view) {
        sourceId = view.getSourceId();
        auto dimensions = view.getLayout().getDimensionIds();
        if (*physicalSourceAxis >= dimensions.size())
          return failure();
        dimension = dimensions[*physicalSourceAxis];
      } else if (buffer) {
        sourceId = buffer.getInstance() + 1;
        derived = true;
      } else {
        auto mapping =
            cast<gpu::AxisMapAttr>(
                fragment.getAxisMaps()[*physicalSourceAxis]);
        sourceId = mapping.getSourceId();
        logicalSourceAxis = mapping.getSourceAxis();
        dimension = mapping.getDimensionId();
        derived = mapping.getDerived();
      }
      dimension = relation->resultDimensionIdentities[resultAxis];
      if (dimension <= 0)
        return failure();
      FailureOr<Value> coordinate = makeRange(
          logicalSourceAxis, reified->range->begin, reified->range->end,
          reified->range->step, sourceId,
          dimension, derived, extent);
      if (failed(coordinate))
        return failure();
      coordinates.push_back(*coordinate);
      continue;
    }
    return failure();
  }
  return coordinates;
}

} // namespace intent::kir_to_gpu
