#include "Construction.h"
#include "Intent/Analysis/ProductSchema.h"
#include "Intent/Conversion/LogicalShape.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/IR/ProgramInterface.h"
#include "Intent/Dialect/GPU/Transforms/Control/Traversal.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;

namespace intent::kir_to_gpu {

std::optional<int64_t> sourceExtentDimension(Value source) {
  while (auto subregion = source.getDefiningOp<intent::SubregionOp>())
    source = subregion.getInputs().front();
  auto domain = source.getDefiningOp<intent::DomainOp>();
  if (!domain || domain.getExtentDimensions().size() != 1)
    return std::nullopt;
  int64_t identity =
      cast<IntegerAttr>(domain.getExtentDimensions()[0]).getInt();
  return identity > 0 ? std::optional<int64_t>(identity) : std::nullopt;
}

std::optional<int64_t> subregionStaticExtentBound(Value source) {
  auto subregion = source ? source.getDefiningOp<intent::SubregionOp>()
                          : intent::SubregionOp();
  if (!subregion || !subregion.getHasStart() || !subregion.getHasStop())
    return std::nullopt;
  Value start = subregion.getInputs()[1];
  Value stop = subregion.getInputs()[2];
  auto addedExtent = [&](Value candidate) -> std::optional<int64_t> {
    auto add = candidate.getDefiningOp<intent::BinaryOp>();
    if (!add || add.getOperatorKind() != BinaryOperator::Add)
      return std::nullopt;
    if (add.getLhs() == start)
      return integerConstant(add.getRhs());
    if (add.getRhs() == start)
      return integerConstant(add.getLhs());
    return std::nullopt;
  };
  std::optional<int64_t> bound = addedExtent(stop);
  if (!bound) {
    auto minimum = stop.getDefiningOp<intent::BinaryOp>();
    if (minimum &&
        (minimum.getOperatorKind() == BinaryOperator::Minimum ||
         minimum.getOperatorKind() == BinaryOperator::MinimumNum)) {
      bound = addedExtent(minimum.getLhs());
      if (!bound)
        bound = addedExtent(minimum.getRhs());
    }
  }
  if (!bound) {
    auto difference = stop.getDefiningOp<intent::BinaryOp>();
    if (difference &&
        difference.getOperatorKind() == BinaryOperator::Subtract) {
      Value base = difference.getRhs();
      auto minimum = difference.getLhs().getDefiningOp<intent::BinaryOp>();
      if (minimum &&
          (minimum.getOperatorKind() == BinaryOperator::Minimum ||
           minimum.getOperatorKind() == BinaryOperator::MinimumNum)) {
        auto distanceFromBase = [&](Value candidate)
            -> std::optional<int64_t> {
          auto add = candidate.getDefiningOp<intent::BinaryOp>();
          if (!add || add.getOperatorKind() != BinaryOperator::Add)
            return std::nullopt;
          if (add.getLhs() == base)
            return integerConstant(add.getRhs());
          if (add.getRhs() == base)
            return integerConstant(add.getLhs());
          return std::nullopt;
        };
        bound = distanceFromBase(minimum.getLhs());
        if (!bound)
          bound = distanceFromBase(minimum.getRhs());
      }
    }
  }
  return bound && *bound > 0 ? bound : std::nullopt;
}

LogicalResult collectIterationAxes(
    Value source,
    SmallVectorImpl<IterationAxis> &axes) {
  if (auto domain = source.getDefiningOp<intent::DomainOp>()) {
    axes.push_back({domain.getBounds()[0], domain.getBounds()[1],
                    domain.getBounds().size() == 3 ? domain.getBounds()[2]
                                                   : Value(),
                    domain.getBounds()[0], source});
    return success();
  }
  if (auto product = source.getDefiningOp<intent::DomainProductOp>()) {
    for (Value component : product.getDomains())
      if (failed(collectIterationAxes(component, axes)))
        return failure();
    return success();
  }
  auto subregion = source.getDefiningOp<intent::SubregionOp>();
  if (!subregion || subregion.getInputs().empty())
    return failure();
  SmallVector<IterationAxis> parent;
  if (failed(collectIterationAxes(subregion.getInputs().front(), parent)) ||
      parent.size() != 1)
    return failure();
  unsigned operand = 1;
  if (subregion.getHasStart())
    parent.front().start = subregion.getInputs()[operand++];
  if (subregion.getHasStop())
    parent.front().stop = subregion.getInputs()[operand++];
  if (operand != subregion.getInputs().size())
    return failure();
  parent.front().source = source;
  axes.append(parent.begin(), parent.end());
  return success();
}

Value ScalarRegionLowering::rangeExtent(
    Location location,
    Value start,
    Value stop,
    Value step) {
  Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
  Value distance = createBinary(builder, location, builder.getIndexType(), stop,
                                start, BinaryOperator::Subtract);
  return createBinary(builder, location, builder.getIndexType(),
                      ceilDivide(location, distance, step), zero,
                      BinaryOperator::Maximum);
}

Value ScalarRegionLowering::ceilDivide(Location location, Value numerator,
                                       Value denominator) {
  Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
  Value one = builder.create<arith::ConstantIndexOp>(location, 1);
  Value quotient = createBinary(builder, location, builder.getIndexType(),
                                 numerator, denominator, BinaryOperator::FloorDivide);
  Value remainder = createBinary(builder, location, builder.getIndexType(),
                                  numerator, denominator, BinaryOperator::Remainder);
  Value hasTail = createCompare(builder, location, builder.getI1Type(),
                               remainder, zero, ComparePredicate::Ne);
  Value tail = builder.create<gpu::SelectOp>(
      location, builder.getIndexType(), hasTail, one, zero);
  // Floor division plus a nonzero-remainder increment is signed ceil division;
  // it does not form an overflowing numerator + denominator - 1.
  return createBinary(builder, location, builder.getIndexType(), quotient, tail,
                      BinaryOperator::Add);
}

Value ScalarRegionLowering::rangeBound(
    Location location,
    Value range,
    unsigned bound) {
  if (auto operation = range.getDefiningOp<gpu::RangeOp>()) {
    if (bound == 0)
      return operation.getStart();
    if (bound == 1)
      return operation.getStop();
    if (bound == 2)
      return operation.getStep();
  }
  return builder.create<gpu::RangeBoundOp>(location, builder.getIndexType(),
                                            range, bound);
}

FailureOr<Value> ScalarRegionLowering::physicalExtentValue(
    Location location,
    PhysicalExprAttr expression) {
  auto kind = expression.getKind();
  if (kind == PhysicalExprKind::Constant)
    return Value(builder.create<arith::ConstantIndexOp>(location,
                                                         expression.getValue()));
  if (kind == PhysicalExprKind::Dimension) {
    int64_t identity = expression.getValue();
    if (identity <= 0)
      return failure();
    auto found = abiDimensions.find(identity);
    return found == abiDimensions.end() ? FailureOr<Value>(failure())
                                     : FailureOr<Value>(found->second);
  }
  if (kind == PhysicalExprKind::Parameter) {
    auto found = parameters.find(expression.getParameterReference().getName());
    return found == parameters.end() ? FailureOr<Value>(failure())
                                     : FailureOr<Value>(found->second);
  }
  func::FuncOp function = physicalKernel;
  if (kind == PhysicalExprKind::ScalarABI) {
    if (auto argument = gpu::resolveArgument(function, expression.getArgumentReference()))
      return Value(argument);
    return failure();
  }
  return Value(builder.create<gpu::PhysicalExprOp>(
      location, builder.getIndexType(), expression));
}

FailureOr<Value> ScalarRegionLowering::logicalExtent(
    Location location, Value value, unsigned axis,
    ArrayRef<unsigned> fieldPath) {
  LogicalShapeMaterialization materialization;
  materialization.leaf = [&](const TensorExtentFact &extent) -> FailureOr<Value> {
    if (extent.constant)
      return Value(builder.create<arith::ConstantIndexOp>(location,
                                                         *extent.constant));
    if (extent.value) {
      auto lowered = get(extent.value);
      return failed(lowered) ? FailureOr<Value>(failure())
                             : asIndex(location, *lowered);
    }
    if (extent.domain) {
      auto lowered = get(extent.domain);
      if (failed(lowered) || !isa<gpu::RangeType>((*lowered).getType()))
        return failure();
      return rangeExtent(location, rangeBound(location, *lowered, 0),
                         rangeBound(location, *lowered, 1),
                         rangeBound(location, *lowered, 2));
    }
    if (!extent.source)
      return failure();
    if (auto argument = dyn_cast<BlockArgument>(extent.source))
      if (auto structured = dyn_cast<StructuredOpInterface>(
              argument.getOwner()->getParentOp()))
        for (const auto &relation : structured.getRegionArgumentRelations(
                 *argument.getOwner()->getParent()))
          if (relation.to == argument &&
              relation.kind == StructuredRelationKind::SourceSlice &&
              llvm::is_contained(relation.axes, extent.axis))
            return emitError(location,
                "compiler-selected source-slice extent is only available to "
                "shape relations"), failure();
    auto lowered = get(extent.source);
    if (failed(lowered))
      return failure();
    Value component = *lowered;
    Type logicalType = extent.source.getType();
    for (unsigned field : extent.fieldPath) {
      auto components = getProductComponents(logicalType);
      auto record = dyn_cast<gpu::RecordType>(component.getType());
      if (!components || field >= components.size() || !record ||
          field >= record.getFieldTypes().size())
        return failure();
      logicalType = cast<TypeAttr>(components[field]).getValue();
      Type physicalType = cast<TypeAttr>(record.getFieldTypes()[field]).getValue();
      component = builder.create<gpu::ExtractOp>(
          location, physicalType, component, field);
    }
    auto physicalAxis = physicalResourceAxis(logicalType, component.getType(),
                                             extent.axis);
    if (failed(physicalAxis))
      return failure();
    if (isa<gpu::ViewType, gpu::BufferType>(component.getType()))
      return resourceExtent(location, component, *physicalAxis);
    if (!isa<gpu::FragmentType>(component.getType()))
      return failure();
    // A fragment's capacity is not its logical shape. Its current coordinate
    // relation must retain the source bounds, including a local subregion.
    gpu::PhysicalProgramAnalysis analysis(physicalKernel);
    auto range = gpu::queryExactLogicalRange(
        analysis.axisRanges(component, *physicalAxis));
    if (failed(range))
      return failure();
    return rangeExtent(location, range->getLogicalStart(),
                       range->getLogicalStop(), range->getStep());
  };
  materialization.multiply = [&](Value lhs, Value rhs) -> FailureOr<Value> {
    return createBinary(builder, location, builder.getIndexType(), lhs, rhs,
                        BinaryOperator::Multiply);
  };
  materialization.exactDivide = [&](Value lhs, Value rhs) -> FailureOr<Value> {
    return createBinary(builder, location, builder.getIndexType(), lhs, rhs,
                        BinaryOperator::FloorDivide);
  };
  materialization.subtract = [&](Value lhs, Value rhs) -> FailureOr<Value> {
    return createBinary(builder, location, builder.getIndexType(), lhs, rhs,
                        BinaryOperator::Subtract);
  };
  materialization.ceilDivide = [&](Value lhs, Value rhs) -> FailureOr<Value> {
    return ceilDivide(location, lhs, rhs);
  };
  materialization.maximum = [&](Value lhs, Value rhs) -> FailureOr<Value> {
    return createBinary(builder, location, builder.getIndexType(), lhs, rhs,
                        BinaryOperator::Maximum);
  };
  return materializeLogicalExtent(canonicalAnalysis, value, axis,
                                  materialization, fieldPath);
}

FailureOr<Value> ScalarRegionLowering::asIndex(Location location, Value value) {
  if (value.getType().isIndex())
    return value;
  if (!isa<IntegerType>(value.getType()))
    return failure();
  return Value(builder.create<gpu::CastOp>(location, builder.getIndexType(),
                                           value));
}

FailureOr<Value> ScalarRegionLowering::asLogicalIndex(
    Location location,
    Value value) {
  if (!isa<gpu::FragmentType>(value.getType()))
    return asIndex(location, value);
  auto source = cast<gpu::FragmentType>(value.getType());
  if (source.getElementType().isIndex())
    return value;
  if (!isa<IntegerType>(source.getElementType()))
    return failure();
  auto target = gpu::FragmentType::get(
      value.getContext(), builder.getIndexType(), source.getShape(),
      source.getAxisMaps(), source.getValidity(), source.getOwner());
  return Value(builder.create<gpu::CastOp>(location, target, value));
}

LogicalResult ScalarRegionLowering::lower(intent::DimOp dim) {
  if (values.count(dim.getResult()))
    return success();
  auto extent = logicalExtent(dim.getLoc(), dim.getSource(), dim.getAxis());
  if (failed(extent))
    return dim.emitOpError("logical source axis has no runtime extent binding");
  values[dim.getResult()] = *extent;
  return success();
}

LogicalResult ScalarRegionLowering::lower(intent::DomainOp domain) {
  Operation *operation = domain.getOperation();
  Location location = domain.getLoc();
  if (values.count(domain.getResult()))
    return success();
  FailureOr<Value> start = get(domain.getBounds()[0]);
  FailureOr<Value> stop = get(domain.getBounds()[1]);
  FailureOr<Value> step =
      domain.getBounds().size() == 3
          ? get(domain.getBounds()[2])
          : FailureOr<Value>(builder.create<arith::ConstantIndexOp>(
                location, 1));
  if (failed(start) || failed(stop) || failed(step))
    return domain.emitOpError("physical domain bounds are unavailable");
  start = asIndex(location, *start);
  stop = asIndex(location, *stop);
  step = asIndex(location, *step);
  if (failed(start) || failed(stop) || failed(step))
    return domain.emitOpError(
        "physical domain bounds are not integer coordinates");
  if (domain.getExtentDimensions().size() != 1)
    return domain.emitOpError(
        "physical domain requires one logical dimension identity");
  int64_t identity =
      cast<IntegerAttr>(domain.getExtentDimensions()[0]).getInt();
  if (identity <= 0)
    return domain.emitOpError(
        "physical domain has no logical dimension identity");
  auto type = gpu::RangeType::get(
      operation->getContext(), domain.getResult().getType().getOriginId(), 0,
      identity, /*derived=*/false);
  auto target =
      builder.create<gpu::RangeOp>(location, type, *start, *stop, *step);
  mapResults(operation, target);
  return success();
}

LogicalResult ScalarRegionLowering::lower(intent::SubregionOp subregion) {
  Operation *operation = subregion.getOperation();
  Location location = subregion.getLoc();
  FailureOr<Value> source = get(subregion.getInputs().front());
  if (failed(source) || !isa<gpu::RangeType>((*source).getType()))
    return subregion.emitOpError(
        "physical subregion source is not an executable range");
  Value start = rangeBound(location, *source, 0);
  Value stop = rangeBound(location, *source, 1);
  Value step = rangeBound(location, *source, 2);
  unsigned operand = 1;
  if (subregion.getHasStart()) {
    FailureOr<Value> explicitStart = get(subregion.getInputs()[operand++]);
    if (failed(explicitStart))
      return failure();
    FailureOr<Value> physicalStart = asIndex(location, *explicitStart);
    if (failed(physicalStart))
      return subregion.emitOpError(
          "subregion start is not an integer coordinate");
    start = *physicalStart;
  }
  if (subregion.getHasStop()) {
    FailureOr<Value> explicitStop = get(subregion.getInputs()[operand]);
    if (failed(explicitStop))
      return failure();
    FailureOr<Value> physicalStop = asIndex(location, *explicitStop);
    if (failed(physicalStop))
      return subregion.emitOpError(
          "subregion stop is not an integer coordinate");
    stop = *physicalStop;
  }
  auto logical = subregion.getResult().getType();
  auto sourceRange = cast<gpu::RangeType>((*source).getType());
  if (subregion.getExtentDimensions().size() != 1)
    return subregion.emitOpError(
        "physical subregion requires one logical extent identity");
  int64_t extentDimension =
      cast<IntegerAttr>(subregion.getExtentDimensions()[0]).getInt();
  if (extentDimension <= 0)
    return subregion.emitOpError(
        "physical subregion has no logical extent identity");
  auto type = gpu::RangeType::get(
      operation->getContext(), logical.getSourceId(), 0,
      extentDimension, sourceRange.getDerived());
  auto target =
      builder.create<gpu::RangeOp>(location, type, start, stop, step);
  if (sourceRange.getDimensionId() <= 0)
    return subregion.emitOpError(
        "physical subregion source has no parent dimension identity");
  target->setAttr(
      gpu::sourceSubregionAttr,
      builder.getI64IntegerAttr(sourceRange.getDimensionId()));
  if (std::optional<int64_t> bound =
          subregionStaticExtentBound(subregion.getResult()))
    target->setAttr(gpu::sourceSubregionBoundAttr,
                    builder.getI64IntegerAttr(*bound));
  mapResults(operation, target);
  return success();
}

LogicalResult ScalarRegionLowering::lower(intent::RegionEndOp end) {
  Location location = end.getLoc();
  FailureOr<Value> source = get(end.getSource());
  if (failed(source) || !isa<gpu::RangeType>((*source).getType()))
    return end.emitOpError("region end source is not a physical range");
  Value target = rangeBound(location, *source, 1);
  values[end.getResult()] = target;
  return success();
}

LogicalResult ScalarRegionLowering::lower(intent::IndicesOp indices) {
  Operation *operation = indices.getOperation();
  Location location = indices.getLoc();
  FailureOr<Value> source = get(indices.getSource());
  if (failed(source))
    return indices.emitOpError("indices physical source is unavailable");
  if (auto fragment = dyn_cast<gpu::FragmentType>((*source).getType())) {
    auto axisAttr = operation->getAttrOfType<IntegerAttr>("tensor_axis");
    if (!axisAttr || axisAttr.getInt() < 0)
      return indices.emitOpError(
          "tensor indices require one explicit logical source axis");
    unsigned logicalAxis = axisAttr.getInt();
    auto physicalAxis = physicalResourceAxis(
        indices.getSource().getType(), fragment, logicalAxis);
    if (failed(physicalAxis))
      return indices.emitOpError("tensor index axis has no physical projection");
    unsigned axis = *physicalAxis;
    auto mapping =
        cast<gpu::AxisMapAttr>(fragment.getAxisMaps()[axis]);
    auto extent = cast<gpu::PhysicalExprAttr>(fragment.getShape()[axis]);
    FailureOr<Value> physicalExtent =
        physicalExtentValue(location, extent);
    if (failed(physicalExtent))
      return indices.emitOpError(
          "tensor index axis extent is not materialized");
    Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
    Value one = builder.create<arith::ConstantIndexOp>(location, 1);
    Value start = zero, logicalStart = zero, step = one, logicalStop;
    gpu::MakeRangeOp sourceRange;
    gpu::PhysicalRangeFact sourceRanges =
        gpu::PhysicalProgramAnalysis(physicalKernel).axisRanges(*source, axis);
    if (!sourceRanges.roots.empty()) {
      FailureOr<gpu::MakeRangeOp> range =
          gpu::queryExactLogicalRange(sourceRanges);
      if (failed(range))
        return indices.emitOpError(
            "tensor index axis has ambiguous logical ranges");
      sourceRange = *range;
      start = range->getStart();
      logicalStart = range->getLogicalStart();
      logicalStop = range->getLogicalStop();
      step = range->getStep();
    } else {
      FailureOr<Value> stop =
          logicalExtent(location, indices.getSource(), logicalAxis);
      if (failed(stop))
        return indices.emitOpError(
            "tensor index axis has no logical extent authority");
      logicalStop = *stop;
    }
    auto coordinateType = gpu::FragmentType::get(
        operation->getContext(), builder.getIndexType(),
        builder.getArrayAttr({extent}),
        builder.getArrayAttr({gpu::AxisMapAttr::get(
            operation->getContext(), mapping.getSourceId(),
            mapping.getSourceAxis(), mapping.getDimensionId(), 0,
            mapping.getDerived())}),
        fragment.getValidity(), fragment.getOwner());
    auto coordinateRange = builder.create<gpu::MakeRangeOp>(
        location, coordinateType, start, *physicalExtent, step, logicalStart,
        logicalStop,
        mapping.getSourceId(), mapping.getSourceAxis(),
        mapping.getDerived());
    if (sourceRange)
      for (StringRef name : {gpu::sourceSubregionAttr,
                             gpu::sourceSubregionBoundAttr})
        if (Attribute attribute = sourceRange->getAttr(name))
          coordinateRange->setAttr(name, attribute);
    Value coordinate = coordinateRange.getResult();
    Value origin = builder.create<gpu::BroadcastOp>(
        location, coordinateType, logicalStart);
    Value stride = builder.create<gpu::BroadcastOp>(
        location, coordinateType, step);
    coordinate = builder.create<gpu::BinaryOp>(
        location, coordinateType, coordinate, origin, BinaryOperator::Subtract);
    coordinate = builder.create<gpu::BinaryOp>(
        location, coordinateType, coordinate, stride,
        BinaryOperator::FloorDivide);
    auto resultType = gpu::FragmentType::get(
        operation->getContext(), builder.getIndexType(), fragment.getShape(),
        fragment.getAxisMaps(), fragment.getValidity(), fragment.getOwner());
    auto target = builder.create<gpu::BroadcastOp>(location, resultType,
                                                   coordinate);
    mapResults(operation, target);
    return success();
  }
  if (!isa<gpu::RangeType>((*source).getType()))
    return indices.emitOpError(
        "indices source is neither a logical range nor a tensor fragment");
  Value start = rangeBound(location, *source, 0);
  Value stop = rangeBound(location, *source, 1);
  Value step = rangeBound(location, *source, 2);
  FailureOr<Type> result =
      convertDataType(canonicalAnalysis, indices.getResult().getType(), operation);
  if (failed(result) || !isa<gpu::FragmentType>(*result))
    return indices.emitOpError("indices result has no physical fragment type");
  auto rangeType = cast<gpu::RangeType>((*source).getType());
  auto genericType = cast<gpu::FragmentType>(*result);
  if (genericType.getShape().size() != 1)
    return indices.emitOpError(
        "range indices require one physical fragment extent");
  FailureOr<Value> extent = physicalExtentValue(
      location, cast<gpu::PhysicalExprAttr>(genericType.getShape()[0]));
  if (failed(extent))
    return indices.emitOpError(
        "range indices physical extent is not materialized");
  auto resultType = gpu::FragmentType::get(
      operation->getContext(), genericType.getElementType(),
      genericType.getShape(),
      builder.getArrayAttr({gpu::AxisMapAttr::get(
          operation->getContext(), rangeType.getSourceId(),
          rangeType.getSourceAxis(), rangeType.getDimensionId(), 0,
          rangeType.getDerived())}),
      genericType.getValidity(), genericType.getOwner());
  auto target = builder.create<gpu::MakeRangeOp>(
      location, resultType, start, *extent, step, start, stop,
      rangeType.getSourceId(), rangeType.getSourceAxis(),
      rangeType.getDerived());
  for (StringRef name : {gpu::sourceSubregionAttr,
                         gpu::sourceSubregionBoundAttr})
    if (Attribute value = (*source).getDefiningOp()->getAttr(name))
      target->setAttr(name, value);
  mapResults(operation, target);
  return success();
}

} // namespace intent::kir_to_gpu
