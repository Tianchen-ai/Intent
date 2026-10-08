#include "Construction.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;

namespace intent::kir_to_gpu {

FailureOr<Type> ScalarRegionLowering::accessResultType(
    Operation *operation,
    Type logical,
    ArrayRef<Value> coordinates,
    std::optional<Type> prototype) {
  auto relation = canonicalAnalysis.indexRelation(operation);
  if (failed(relation))
    return failure();
  SmallVector<Attribute> liftedShape;
  SmallVector<Attribute> liftedMappings;
  uint64_t liftedOwner = 1;
  if (prototype) {
    auto fragment = dyn_cast<gpu::FragmentType>(*prototype);
    size_t logicalRank = relation->resultDimensionIdentities.size();
    if (fragment && fragment.getShape().size() >= logicalRank) {
      size_t prefixRank = fragment.getShape().size() - logicalRank;
      liftedOwner = fragment.getOwner();
      for (unsigned axis = 0; axis < prefixRank; ++axis) {
        auto mapping = cast<gpu::AxisMapAttr>(fragment.getAxisMaps()[axis]);
        liftedShape.push_back(fragment.getShape()[axis]);
        liftedMappings.push_back(gpu::AxisMapAttr::get(
            operation->getContext(), mapping.getSourceId(),
            mapping.getSourceAxis(), mapping.getDimensionId(), axis,
            mapping.getDerived()));
      }
    }
  }
  unsigned coordinateIndex = 0;
  for (const IndexTermFact &term : relation->terms) {
    if (term.kind == 1)
      continue;
    if (coordinateIndex >= coordinates.size())
      return failure();
    Value current = coordinates[coordinateIndex++];
    if (term.kind != 3)
      continue;
    auto fragment = dyn_cast<gpu::FragmentType>(current.getType());
    if (!fragment)
      continue;
    if (fragment.getShape().size() < term.indexAxes.size())
      return failure();
    liftedOwner = fragment.getOwner();
    unsigned prefix = fragment.getShape().size() - term.indexAxes.size();
    for (unsigned axis = 0; axis < prefix; ++axis) {
      Attribute mappingAttribute = fragment.getAxisMaps()[axis];
      auto mapping = cast<gpu::AxisMapAttr>(mappingAttribute);
      bool exists = llvm::any_of(liftedMappings, [&](Attribute existing) {
        auto value = cast<gpu::AxisMapAttr>(existing);
        return value.getSourceId() == mapping.getSourceId() &&
               value.getSourceAxis() == mapping.getSourceAxis() &&
               value.getDerived() == mapping.getDerived();
      });
      if (exists)
        continue;
      liftedShape.push_back(fragment.getShape()[axis]);
      liftedMappings.push_back(gpu::AxisMapAttr::get(
          operation->getContext(), mapping.getSourceId(),
          mapping.getSourceAxis(), mapping.getDimensionId(),
          liftedMappings.size(), mapping.getDerived()));
    }
  }
  FailureOr<Type> converted = failure();
  auto logicalTensor = dyn_cast<RankedTensorType>(logical);
  auto prototypeFragment =
      prototype ? dyn_cast<gpu::FragmentType>(*prototype) : gpu::FragmentType();
  if (logicalTensor && logicalTensor.getRank() == 0) {
    // A logical rank-zero tensor is a scalar physical value. Ownership axes,
    // when present, are appended from the actual coordinates/prototype below.
    converted = logicalTensor.getElementType();
  } else if (logicalTensor && prototypeFragment) {
    // A write value may already carry physical ownership axes introduced by
    // scalar logical indices.  The index relation describes only its result
    // tail.  Split that tail from the prototype before combining it with the
    // lifted prefix below; using the complete prototype here would append the
    // ownership axes twice.
    size_t resultRank = relation->resultDimensionIdentities.size();
    if (prototypeFragment.getShape().size() < resultRank)
      return failure();
    size_t tailStart = prototypeFragment.getShape().size() - resultRank;
    SmallVector<Attribute> tailShape(
        prototypeFragment.getShape().begin() + tailStart,
        prototypeFragment.getShape().end());
    SmallVector<Attribute> tailMappings;
    for (unsigned axis = 0; axis < resultRank; ++axis) {
      auto mapping = cast<gpu::AxisMapAttr>(
          prototypeFragment.getAxisMaps()[tailStart + axis]);
      tailMappings.push_back(gpu::AxisMapAttr::get(
          operation->getContext(), mapping.getSourceId(),
          mapping.getSourceAxis(), mapping.getDimensionId(), axis,
          mapping.getDerived()));
    }
    converted = Type(gpu::FragmentType::get(
        operation->getContext(), logicalTensor.getElementType(),
        builder.getArrayAttr(tailShape), builder.getArrayAttr(tailMappings),
        prototypeFragment.getValidity(), prototypeFragment.getOwner()));
  } else if (logicalTensor) {
    ArrayRef<int64_t> dimensions = relation->resultDimensionIdentities;
    if (dimensions.size() != static_cast<size_t>(logicalTensor.getRank()))
      return failure();
    SmallVector<Attribute> shape;
    SmallVector<Attribute> mappings;
    for (unsigned axis = 0; axis < dimensions.size(); ++axis) {
      PhysicalExprAttr extent;
      if (logicalTensor.isDynamicDim(axis)) {
        FailureOr<PhysicalExprAttr> dynamicExtent =
            fragmentExtentExpression(canonicalAnalysis, logicalTensor, operation, axis);
        if (failed(dynamicExtent))
          return failure();
        extent = *dynamicExtent;
      } else {
        extent = expression(operation->getContext(),
                            PhysicalExprKind::Constant,
                            logicalTensor.getDimSize(axis));
      }
      FailureOr<PhysicalAxisIdentity> identity =
          resultAxisIdentity(operation, /*resultIndex=*/0, axis);
      if (!extent || dimensions[axis] <= 0 || failed(identity))
        return failure();
      shape.push_back(extent);
      mappings.push_back(gpu::AxisMapAttr::get(
          operation->getContext(), identity->sourceId, identity->sourceAxis,
          dimensions[axis], axis, identity->derived));
    }
    converted = Type(gpu::FragmentType::get(
        operation->getContext(), logicalTensor.getElementType(),
        builder.getArrayAttr(shape), builder.getArrayAttr(mappings),
        /*validity=*/1, liftedOwner));
  } else {
    converted = convertDataType(canonicalAnalysis, logical, operation, prototype);
  }
  if (failed(converted)) {
    operation->emitOpError(
        "access result logical type has no physical fragment schema");
    return failure();
  }
  if (!liftedShape.empty()) {
    Type element = *converted;
    SmallVector<Attribute> shape(liftedShape.begin(), liftedShape.end());
    SmallVector<Attribute> mappings(liftedMappings.begin(),
                                   liftedMappings.end());
    uint32_t validity = 1;
    if (auto fragment = dyn_cast<gpu::FragmentType>(*converted)) {
      element = fragment.getElementType();
      validity = fragment.getValidity();
      for (auto [axis, mappingAttribute] :
           llvm::enumerate(fragment.getAxisMaps())) {
        auto mapping = cast<gpu::AxisMapAttr>(mappingAttribute);
        bool exists = llvm::any_of(mappings, [&](Attribute existing) {
          auto value = cast<gpu::AxisMapAttr>(existing);
          return value.getSourceId() == mapping.getSourceId() &&
                 value.getSourceAxis() == mapping.getSourceAxis() &&
                 value.getDerived() == mapping.getDerived();
        });
        if (exists)
          continue;
        shape.push_back(fragment.getShape()[axis]);
        mappings.push_back(gpu::AxisMapAttr::get(
            operation->getContext(), mapping.getSourceId(),
            mapping.getSourceAxis(), mapping.getDimensionId(), mappings.size(), mapping.getDerived()));
      }
    }
    *converted = gpu::FragmentType::get(
        operation->getContext(), element,
        builder.getArrayAttr(shape), builder.getArrayAttr(mappings), validity,
        liftedOwner);
  }
  auto fragment = dyn_cast<gpu::FragmentType>(*converted);
  if (!fragment)
    return *converted;
  SmallVector<Attribute> shape(fragment.getShape().begin(),
                               fragment.getShape().end());
  SmallVector<Attribute> mappings(liftedMappings.begin(),
                                  liftedMappings.end());
  unsigned coordinate = 0;
  unsigned resultAxis = liftedMappings.size();
  const unsigned liftedRank = resultAxis;
  unsigned advancedRank = relation->advancedRank;
  bool advancedMapped = false;
  auto resultDimension = [&](size_t axis) -> FailureOr<int64_t> {
    ArrayRef<int64_t> dimensions = relation->resultDimensionIdentities;
    if (axis < liftedRank || axis - liftedRank >= dimensions.size())
      return failure();
    return dimensions[axis - liftedRank];
  };
  auto resultIdentity = [&](unsigned axis)
      -> FailureOr<PhysicalAxisIdentity> {
    if (operation->getNumResults() != 0)
      return resultAxisIdentity(operation, /*resultIndex=*/0, axis);
    if (axis >= fragment.getAxisMaps().size())
      return failure();
    auto mapping = cast<gpu::AxisMapAttr>(fragment.getAxisMaps()[axis]);
    return PhysicalAxisIdentity{mapping.getSourceId(), mapping.getSourceAxis(),
                                mapping.getDimensionId(),
                                mapping.getDerived()};
  };
  auto appendCoordinate = [&](Value value) -> LogicalResult {
    auto source = dyn_cast<gpu::FragmentType>(value.getType());
    if (!source)
      return failure();
    for (Attribute attribute : source.getAxisMaps()) {
      auto mapping = cast<gpu::AxisMapAttr>(attribute);
      FailureOr<int64_t> dimension = resultDimension(resultAxis);
      if (failed(dimension))
        return failure();
      shape[resultAxis] = source.getShape()[mapping.getFragmentAxis()];
      mappings.push_back(gpu::AxisMapAttr::get(
          operation->getContext(), mapping.getSourceId(),
          mapping.getSourceAxis(), *dimension, resultAxis++,
          mapping.getDerived()));
    }
    return success();
  };
  auto advancedAxisMapping = [&](unsigned advancedAxis)
      -> gpu::AxisMapAttr {
    ArrayRef<int64_t> resultDimensions = relation->resultDimensionIdentities;
    if (advancedAxis >= resultDimensions.size())
      return {};
    gpu::AxisMapAttr selected;
    unsigned coordinateIndex = 0;
    for (const IndexTermFact &term : relation->terms) {
      if (term.kind == 1)
        continue;
      if (coordinateIndex >= coordinates.size())
        return {};
      Value current = coordinates[coordinateIndex++];
      if (term.kind != 3)
        continue;
      if (!isa<RankedTensorType>(term.operands[0].getType()))
        continue;
      auto logical =
          cast<RankedTensorType>(term.operands[0].getType());
      auto source = dyn_cast<gpu::FragmentType>(current.getType());
      DenseI64ArrayAttr logicalDimensions = dimensionIds(logical);
      if (!source || !logicalDimensions ||
          logical.getRank() > static_cast<int64_t>(advancedRank))
        continue;
      unsigned logicalResultAxis = *relation->advancedStart + advancedAxis;
      auto found = llvm::find(term.indexAxes, logicalResultAxis);
      if (found == term.indexAxes.end())
        continue;
      unsigned localAxis = found - term.indexAxes.begin();
      // A singleton advanced-index operand is broadcast along this result
      // axis and therefore carries no coordinate variation for it.  Let the
      // non-singleton operand provide the axis relation instead of treating
      // the two broadcast operands as conflicting coordinate authorities.
      if (!logical.isDynamicDim(localAxis) &&
          logical.getDimSize(localAxis) == 1)
        continue;
      if (logicalDimensions[localAxis] <= 0 ||
          source.getShape().size() <
              static_cast<unsigned>(logical.getRank()))
        return {};
      unsigned physicalAxis =
          source.getShape().size() - logical.getRank() + localAxis;
      auto mapping = cast<gpu::AxisMapAttr>(
          source.getAxisMaps()[physicalAxis]);
      auto sameMapping = [](gpu::AxisMapAttr lhs, gpu::AxisMapAttr rhs) {
        return lhs.getSourceId() == rhs.getSourceId() &&
               lhs.getSourceAxis() == rhs.getSourceAxis() &&
               lhs.getDimensionId() == rhs.getDimensionId() &&
               lhs.getDerived() == rhs.getDerived();
      };
      if (selected && !sameMapping(selected, mapping))
        return {};
      selected = mapping;
    }
    return selected;
  };
  for (const IndexTermFact &term : relation->terms) {
    if (!term.resultAxes.empty() &&
        !(term.kind == 3 && advancedMapped) &&
        resultAxis != liftedRank + term.resultAxes.front())
      return failure();
    if (term.kind == 1) {
      FailureOr<int64_t> dimension = resultDimension(resultAxis);
      FailureOr<PhysicalAxisIdentity> identity = resultIdentity(resultAxis);
      if (failed(dimension) || *dimension <= 0 || failed(identity))
        return failure();
      mappings.push_back(gpu::AxisMapAttr::get(
          operation->getContext(), identity->sourceId, identity->sourceAxis,
          *dimension, resultAxis, identity->derived));
      ++resultAxis;
      continue;
    }
    if (coordinate >= coordinates.size())
      return failure();
    Value current = coordinates[coordinate++];
    if (term.kind == 0 || term.kind == 4 ||
        term.kind == 5) {
      if (failed(appendCoordinate(current))) {
        operation->emitOpError(
            "access range coordinate cannot map a result axis");
        return failure();
      }
      continue;
    }
    if (term.kind == 3 && isa<gpu::FragmentType>(current.getType())) {
      if (!isa<RankedTensorType>(term.operands[0].getType()))
        continue;
      if (!advancedMapped) {
        for (unsigned axis = 0; axis < advancedRank; ++axis) {
          FailureOr<int64_t> dimension = resultDimension(resultAxis);
          if (failed(dimension))
            return failure();
          PhysicalAxisIdentity identity;
          if (auto mapping = advancedAxisMapping(axis)) {
            identity = PhysicalAxisIdentity{
                mapping.getSourceId(), mapping.getSourceAxis(), *dimension,
                mapping.getDerived()};
          } else {
            FailureOr<PhysicalAxisIdentity> derivedIdentity =
                resultIdentity(resultAxis);
            if (failed(derivedIdentity))
              return failure();
            identity = *derivedIdentity;
            identity.dimensionId = *dimension;
          }
          mappings.push_back(gpu::AxisMapAttr::get(
              operation->getContext(), identity.sourceId, identity.sourceAxis,
              *dimension, resultAxis++, identity.derived));
        }
        advancedMapped = true;
      }
      continue;
    }
    if (term.kind != 2 && term.kind != 3)
      return failure();
  }
  if (coordinate != coordinates.size() ||
      resultAxis != fragment.getShape().size()) {
    operation->emitOpError()
        << "access result rank/coordinate partition disagrees: consumed "
        << coordinate << " of " << coordinates.size() << " coordinates and "
        << resultAxis << " of " << fragment.getShape().size()
        << " result axes";
    return failure();
  }
  return Type(gpu::FragmentType::get(
      operation->getContext(), fragment.getElementType(),
      builder.getArrayAttr(shape),
      builder.getArrayAttr(mappings), fragment.getValidity(),
      fragment.getOwner()));
}

FailureOr<Value> ScalarRegionLowering::accessValue(
    Operation *operation,
    Value logical,
    ArrayRef<Value> coordinates) {
  FailureOr<Value> value = get(logical);
  if (failed(value))
    return failure();
  FailureOr<Type> relationType = accessResultType(
      operation, logical.getType(), coordinates, (*value).getType());
  if (failed(relationType))
    return failure();
  if ((*value).getType() == *relationType)
    return *value;
  auto source = dyn_cast<gpu::FragmentType>((*value).getType());
  auto target = dyn_cast<gpu::FragmentType>(*relationType);
  if (!source && target && (*value).getType() == target.getElementType())
    return Value(builder.create<gpu::SplatOp>(operation->getLoc(), target,
                                              *value));
  if (!source || !target) {
    operation->emitOpError()
        << "indexed write value/result relation is not a fragment: value="
        << (*value).getType() << ", relation=" << *relationType;
    return failure();
  }
  if (source.getElementType() != target.getElementType() ||
      source.getShape().size() != target.getShape().size()) {
    operation->emitOpError()
        << "indexed write value cannot adopt its result relation: " << source
        << " vs " << target;
    return failure();
  }
  if (source.getShape() == target.getShape() &&
      !gpu::queryAxisProjection(source, target).isExact()) {
    // Canonical assignment has already aligned logical tensor axes by
    // position.  Rebinding those axes to destination coordinates preserves
    // lane order; matching old domain identities would imply a transpose.
    SmallVector<Attribute> groups;
    for (unsigned axis = 0; axis < source.getShape().size(); ++axis) {
      auto axes = builder.getDenseI64ArrayAttr({static_cast<int64_t>(axis)});
      groups.push_back(gpu::ReshapeGroupAttr::get(
          operation->getContext(), axes, axes));
    }
    auto projected = builder.create<gpu::ReshapeOp>(
        operation->getLoc(), target, *value, builder.getArrayAttr(groups));
    if (Operation *definition = (*value).getDefiningOp())
      if (Attribute origin = definition->getAttr(gpu::originAttr))
        projected->setAttr(gpu::originAttr, origin);
    return projected.getResult();
  }
  return retargetBroadcast(builder, operation->getLoc(), *value, target);
}

FailureOr<Value> ScalarRegionLowering::projectAccessOperand(
    Location location,
    Value value,
    gpu::FragmentType accessType) {
  if (!value)
    return value;
  Type element = value.getType();
  if (auto source = dyn_cast<gpu::FragmentType>(element))
    element = source.getElementType();
  auto target = gpu::FragmentType::get(
      value.getContext(), element, accessType.getShape(),
      accessType.getAxisMaps(), accessType.getValidity(), accessType.getOwner());
  if (value.getType() == target)
    return value;
  if (!isa<gpu::FragmentType>(value.getType()))
    return Value(builder.create<gpu::SplatOp>(location, target, value));
  return retargetBroadcast(builder, location, value, target);
}

SmallVector<bool> ScalarRegionLowering::canonicalAccessAxisProofs(
    Operation *operation,
    Value resource) {
  auto relation = canonicalAnalysis.indexRelation(operation);
  if (failed(relation) || !resource)
    return {};

  unsigned physicalRank = 0;
  if (auto view = dyn_cast<gpu::ViewType>(resource.getType()))
    physicalRank = view.getRank();
  else if (auto buffer = dyn_cast<gpu::BufferType>(resource.getType()))
    physicalRank = buffer.getShape().size();
  else if (auto fragment = dyn_cast<gpu::FragmentType>(resource.getType()))
    physicalRank = fragment.getShape().size();
  else
    return {};
  SmallVector<bool> proven(physicalRank, false);

  for (const IndexTermFact &term : relation->terms) {
    if (!term.sourceAxis) continue;
    auto axis = physicalResourceAxis(relation->source.getType(),
                                     resource.getType(), *term.sourceAxis);
    if (failed(axis) || *axis >= proven.size()) return {};
    // A full slice of an existing fragment addresses a completed SSA value.
    // Logical resource bounds have already participated in its producer.
    proven[*axis] = term.inBounds ||
        (term.kind == 0 && isa<gpu::FragmentType>(resource.getType()));
  }
  return proven;
}

FailureOr<Value> ScalarRegionLowering::materializeAccessValidity(
    Operation *operation,
    Value resource,
    ArrayRef<Value> coordinates,
    ArrayRef<int64_t> sourceAxes,
    Type payloadType,
    Value existing) {
  if (coordinates.size() != sourceAxes.size())
    return failure();
  SmallVector<bool> proven = canonicalAccessAxisProofs(operation, resource);
  if (proven.empty() && !coordinates.empty())
    return failure();

  auto payloadFragment = dyn_cast<gpu::FragmentType>(payloadType);
  Type predicateType = builder.getI1Type();
  if (payloadFragment)
    predicateType = gpu::FragmentType::get(
        operation->getContext(), builder.getI1Type(),
        payloadFragment.getShape(), payloadFragment.getAxisMaps(),
        payloadFragment.getValidity(), payloadFragment.getOwner());

  if (existing) {
    if (payloadFragment) {
      FailureOr<Value> projected =
          projectAccessOperand(operation->getLoc(), existing, payloadFragment);
      if (failed(projected))
        return failure();
      existing = *projected;
    }
    if (existing.getType() != predicateType)
      return failure();
  }

  // Basic index terms have an explicit Cartesian result position.  Preserve
  // that occurrence when equal-length regions share one source identity;
  // broadcasting by the coordinate type alone would select the trailing axis.
  SmallVector<std::optional<unsigned>> coordinateAxes(coordinates.size());
  auto relation = canonicalAnalysis.indexRelation(operation);
  if (payloadFragment && succeeded(relation) &&
      relation->resultDimensionIdentities.size() == payloadFragment.getShape().size()) {
    unsigned coordinateIndex = 0;
    bool cartesian = true;
    for (const IndexTermFact &term : relation->terms) {
      if (!term.sourceAxis)
        continue;
      if (coordinateIndex >= coordinates.size()) {
        cartesian = false;
        break;
      }
      auto type = dyn_cast<gpu::FragmentType>(coordinates[coordinateIndex].getType());
      if (term.kind == 0 || term.kind == 4 || term.kind == 5) {
        unsigned resultAxis = term.resultAxes.front();
        if (!type || type.getShape().size() != 1 ||
            resultAxis >= payloadFragment.getShape().size()) {
          cartesian = false;
          break;
        }
        auto source = cast<gpu::AxisMapAttr>(type.getAxisMaps()[0]);
        auto target = cast<gpu::AxisMapAttr>(payloadFragment.getAxisMaps()[resultAxis]);
        if (!(gpu::sourceAxisIdentity(source) == gpu::sourceAxisIdentity(target)) ||
            source.getDimensionId() != target.getDimensionId() ||
            target.getDimensionId() != relation->resultDimensionIdentities[resultAxis] ||
            type.getShape()[0] != payloadFragment.getShape()[resultAxis] ||
            type.getOwner() != payloadFragment.getOwner()) {
          cartesian = false;
          break;
        }
        coordinateAxes[coordinateIndex] = resultAxis;
      } else if (type) {
        cartesian = false;
        break;
      }
      ++coordinateIndex;
    }
    if (!cartesian || coordinateIndex != coordinates.size())
      std::fill(coordinateAxes.begin(), coordinateAxes.end(), std::nullopt);
  }

  Value valid = existing;
  for (auto [coordinateIndex, coordinate] : llvm::enumerate(coordinates)) {
    int64_t sourceAxis = sourceAxes[coordinateIndex];
    if (sourceAxis < 0 ||
        sourceAxis >= static_cast<int64_t>(proven.size()))
      return failure();
    if (proven[sourceAxis])
      continue;
    FailureOr<Value> extent =
        resourceExtent(operation->getLoc(), resource, sourceAxis);
    if (failed(extent)) {
      operation->emitOpError("indexed resource extent is unavailable")
          << "; resource=" << resource << "; axis=" << sourceAxis;
      return failure();
    }

    Value index = coordinate;
    Value upper = *extent;
    Value zero = builder.create<arith::ConstantIndexOp>(operation->getLoc(), 0);
    FailureOr<Value> logicalIndex =
        asLogicalIndex(operation->getLoc(), index);
    if (failed(logicalIndex))
      return failure();
    index = *logicalIndex;

    Type axisPredicateType = builder.getI1Type();
    if (auto indexType = dyn_cast<gpu::FragmentType>(index.getType())) {
      axisPredicateType = gpu::FragmentType::get(
          operation->getContext(), builder.getI1Type(), indexType.getShape(),
          indexType.getAxisMaps(), indexType.getValidity(),
          indexType.getOwner());
      upper = builder.create<gpu::SplatOp>(operation->getLoc(), indexType,
                                           upper);
      zero = builder.create<gpu::SplatOp>(operation->getLoc(), indexType,
                                          zero);
    }

    Value upperBound = createCompare(
        builder, operation->getLoc(), axisPredicateType, index, upper,
        ComparePredicate::Lt);
    // Construction preserves the complete access domain. Physical rewrites
    // still change these coordinates; only later range simplification may
    // discharge a bound using facts about the final coordinate expression.
    Value lowerBound = createCompare(
        builder, operation->getLoc(), axisPredicateType, index, zero,
        ComparePredicate::Ge);
    Value axisValid = createBinary(builder, operation->getLoc(), axisPredicateType,
                                   lowerBound, upperBound,
                                   BinaryOperator::LogicalAnd);
    if (payloadFragment) {
      auto axis = coordinateAxes[coordinateIndex];
      FailureOr<Value> projected = axis
          ? gpu::projectPredicateToFragmentAxis(builder, operation->getLoc(),
                                                axisValid, payloadFragment, *axis)
          : projectAccessOperand(operation->getLoc(), axisValid, payloadFragment);
      if (failed(projected)) {
        operation->emitOpError("index bounds have no payload projection")
            << "; index=" << index << "; payload=" << payloadFragment;
        return failure();
      }
      axisValid = *projected;
    }
    if (axisValid.getType() != predicateType)
      return failure();
    valid = valid ? createBinary(builder, operation->getLoc(), predicateType,
                                 valid, axisValid,
                                 BinaryOperator::LogicalAnd)
                  : axisValid;
  }
  return valid;
}

FailureOr<Value> ScalarRegionLowering::zeroAccessFill(
    Location location,
    Type type) {
  auto fragment = dyn_cast<gpu::FragmentType>(type);
  Type element = fragment ? fragment.getElementType() : type;
  TypedAttr zero;
  if (auto integer = dyn_cast<IntegerType>(element))
    zero = builder.getIntegerAttr(integer, 0);
  else if (auto floating = dyn_cast<FloatType>(element))
    zero = builder.getFloatAttr(floating, 0.0);
  else if (isa<IndexType>(element))
    zero = builder.getIndexAttr(0);
  if (!zero)
    return failure();
  FailureOr<Value> scalar =
      gpu::materializeScalarConstant(builder, location, zero, element);
  if (failed(scalar))
    return failure();
  if (!fragment)
    return *scalar;
  return Value(builder.create<gpu::SplatOp>(location, fragment, *scalar));
}

} // namespace intent::kir_to_gpu
