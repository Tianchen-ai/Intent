#include "Construction.h"
#include "Intent/Analysis/ProductSchema.h"
#include "llvm/ADT/STLExtras.h"
#include <limits>

using namespace mlir;

namespace intent::kir_to_gpu {

DenseI64ArrayAttr dimensionIds(RankedTensorType tensor) {
  auto shape = dyn_cast_or_null<TensorShapeAttr>(tensor.getEncoding());
  return shape ? shape.getDimensions() : DenseI64ArrayAttr();
}

RankedTensorType viewTensor(Value value) {
  auto view = dyn_cast<ViewType>(value.getType());
  return view ? dyn_cast<RankedTensorType>(view.getTensor()) : RankedTensorType();
}

PhysicalExprAttr expression(
    MLIRContext *context,
    PhysicalExprKind kind,
    int64_t value,
    StringRef symbol,
    ArrayRef<Attribute> operands) {
  return PhysicalExprAttr::get(
      context, kind, value,
      kind == PhysicalExprKind::Parameter
          ? Attribute(gpu::ParameterRefAttr::get(context, StringAttr::get(context, symbol)))
          : Attribute(StringAttr::get(context, symbol)),
      ArrayAttr::get(context, operands));
}

PhysicalExprAttr parameterExpression(MLIRContext *context, StringRef name) {
  return expression(context, PhysicalExprKind::Parameter, 0, name);
}

PhysicalExprAttr binaryExpression(
    MLIRContext *context,
    PhysicalExprKind kind,
    PhysicalExprAttr lhs,
    PhysicalExprAttr rhs) {
  auto leftKind = lhs.getKind();
  auto rightKind = rhs.getKind();
  bool leftConstant = leftKind == PhysicalExprKind::Constant;
  bool rightConstant = rightKind == PhysicalExprKind::Constant;
  if (kind == PhysicalExprKind::Multiply && leftConstant && rightConstant)
    return expression(context, PhysicalExprKind::Constant,
                      lhs.getValue() * rhs.getValue());
  if ((kind == PhysicalExprKind::Add ||
       kind == PhysicalExprKind::Subtract) &&
      rightConstant && rhs.getValue() == 0)
    return lhs;
  if (kind == PhysicalExprKind::Add && leftConstant && lhs.getValue() == 0)
    return rhs;
  if (kind == PhysicalExprKind::Multiply) {
    if ((leftConstant && lhs.getValue() == 0) ||
        (rightConstant && rhs.getValue() == 0))
      return expression(context, PhysicalExprKind::Constant, 0);
    if (leftConstant && lhs.getValue() == 1)
      return rhs;
    if (rightConstant && rhs.getValue() == 1)
      return lhs;
  }
  if ((kind == PhysicalExprKind::CeilDiv ||
       kind == PhysicalExprKind::FloorDiv) &&
      rightConstant && rhs.getValue() == 1)
    return lhs;
  return expression(context, kind, 0, {}, {lhs, rhs});
}

FragmentType fragmentType(
    MLIRContext *context,
    Type element,
    ArrayRef<PhysicalExprAttr> shape,
    ArrayRef<PhysicalAxisIdentity> axes,
    uint64_t owner) {
  SmallVector<Attribute> extents(shape.begin(), shape.end());
  SmallVector<Attribute> mappings;
  for (auto [fragmentAxis, mapping] : llvm::enumerate(axes))
    mappings.push_back(AxisMapAttr::get(
        context, mapping.sourceId, mapping.sourceAxis, mapping.dimensionId,
        fragmentAxis, mapping.derived));
  return FragmentType::get(context, element, ArrayAttr::get(context, extents),
                           ArrayAttr::get(context, mappings), 1, owner);
}

bool samePhysicalShape(gpu::FragmentType lhs, gpu::FragmentType rhs) {
  return lhs.getShape() == rhs.getShape() &&
         lhs.getAxisMaps() == rhs.getAxisMaps() &&
         lhs.getValidity() == rhs.getValidity() &&
         lhs.getOwner() == rhs.getOwner();
}

static FailureOr<PhysicalAxisIdentity> physicalAxisIdentity(
    CanonicalKernelAnalysis &canonicalAnalysis,
    Operation *origin,
    int64_t dimension,
    unsigned axis,
    unsigned resultIndex) {
  if (!origin || dimension <= 0 || resultIndex >= origin->getNumResults())
    return failure();
  Value value = origin->getResult(resultIndex);
  CoordinateProvenance provenance = canonicalAnalysis.axisProvenance(value, axis);
  if (provenance.known && provenance.origins.size() == 1) {
    const CoordinateOrigin &root = provenance.origins.front();
    if (auto domain = dyn_cast<DomainType>(root.source.getType()))
      return PhysicalAxisIdentity{domain.getOriginId(), root.axis, dimension, false};
    if (auto region = dyn_cast<RegionType>(root.source.getType()))
      return PhysicalAxisIdentity{region.getSourceId(), root.axis, dimension, false};
    if (auto argument = dyn_cast<BlockArgument>(root.source))
      if (auto parameter = getSourceParameter(argument))
        return PhysicalAxisIdentity{static_cast<uint64_t>(parameter.getOriginId()) + 1,
                                    root.axis, dimension, false};
    if (auto result = dyn_cast<OpResult>(root.source)) {
      auto identity = resultAxisIdentity(result.getOwner(), result.getResultNumber(),
                                         root.axis);
      if (succeeded(identity)) {
        identity->dimensionId = dimension;
        return *identity;
      }
    }
  }
  // An operation can combine distinct coordinate occurrences. Its own result
  // is the identity of that new axis, never an unrelated same-sized domain.
  auto identity = resultAxisIdentity(origin, resultIndex, axis);
  if (failed(identity)) return failure();
  identity->dimensionId = dimension;
  return *identity;
}

FailureOr<PhysicalExprAttr> fragmentExtentExpression(
    CanonicalKernelAnalysis &canonicalAnalysis,
    RankedTensorType tensor,
    Operation *origin,
    unsigned axis) {
  DenseI64ArrayAttr identities = dimensionIds(tensor);
  if (!origin || !identities || axis >= identities.size() || identities[axis] <= 0)
    return failure();
  Value value = origin->getNumResults() ? origin->getResult(0) : Value();
  if (!value)
    if (auto access = dyn_cast<IndexedAccessOpInterface>(origin))
      value = access.getStoredValue();
  if (std::optional<int64_t> staticBound =
          subregionStaticExtentBound(value ? canonicalAnalysis.tensorExtent(value, axis).domain
                                            : Value()))
    return expression(tensor.getContext(), PhysicalExprKind::Constant,
                      *staticBound);
  // Initial scalar ownership needs no choice between equal-length source
  // occurrences. Their own ranges retain the logical coordinates and bounds.
  return expression(origin->getContext(), PhysicalExprKind::Constant, 1);
}

Type convertScalarType(Type type, uint64_t owner) {
  if (isa<IntegerType, FloatType, IndexType>(type))
    return type;
  if (isa<LogicalIndexType>(type))
    return IndexType::get(type.getContext());
  if (auto record = dyn_cast<intent::RecordType>(type))
    return gpu::RecordType::get(type.getContext(), record.getFieldNames(),
                                record.getFieldTypes(), owner);
  if (auto tuple = dyn_cast<intent::TupleType>(type)) {
    SmallVector<Attribute> names;
    for (unsigned index = 0; index < tuple.getComponentTypes().size(); ++index)
      names.push_back(StringAttr::get(type.getContext(),
                                     ("_" + Twine(index)).str()));
    return gpu::RecordType::get(type.getContext(),
                                ArrayAttr::get(type.getContext(), names),
                                tuple.getComponentTypes(), owner);
  }
  return {};
}

FailureOr<PhysicalAxisIdentity> resultAxisIdentity(
    Operation *operation,
    unsigned resultIndex,
    unsigned axis) {
  if (!operation || resultIndex >= operation->getNumResults())
    return failure();
  auto results = operation->getAttrOfType<ArrayAttr>("intent.result_nodes");
  if (!results || resultIndex >= results.size())
    return failure();
  auto value = dyn_cast<IntegerAttr>(results[resultIndex]);
  if (!value || value.getInt() < 0 || axis > std::numeric_limits<uint32_t>::max())
    return failure();
  return PhysicalAxisIdentity{static_cast<uint64_t>(value.getInt()) + 1,
                              static_cast<uint32_t>(axis),
                              /*dimensionId=*/0, /*derived=*/true};
}

FailureOr<FragmentType> convertTensorType(
    CanonicalKernelAnalysis &canonicalAnalysis,
    RankedTensorType tensor,
    Operation *origin,
    std::optional<FragmentType> prototype,
    uint64_t owner,
    unsigned resultIndex) {
  MLIRContext *context = tensor.getContext();
  DenseI64ArrayAttr dimensions = dimensionIds(tensor);
  SmallVector<PhysicalExprAttr> shape;
  SmallVector<PhysicalAxisIdentity> mappings;
  for (unsigned axis = 0; axis < static_cast<unsigned>(tensor.getRank()); ++axis) {
    if (!dimensions || dimensions.size() != static_cast<unsigned>(tensor.getRank()) ||
        dimensions[axis] <= 0)
      return failure();
    if (tensor.isDynamicDim(axis)) {
      FailureOr<PhysicalExprAttr> extent =
          fragmentExtentExpression(canonicalAnalysis, tensor, origin, axis);
      if (failed(extent))
        return failure();
      shape.push_back(*extent);
      FailureOr<PhysicalAxisIdentity> mapping =
          physicalAxisIdentity(canonicalAnalysis, origin, dimensions[axis], axis, resultIndex);
      if (failed(mapping))
        return failure();
      mappings.push_back(*mapping);
    } else {
      shape.push_back(expression(context, PhysicalExprKind::Constant,
                                 tensor.getDimSize(axis)));
      FailureOr<PhysicalAxisIdentity> mapping =
          physicalAxisIdentity(canonicalAnalysis, origin, dimensions[axis], axis, resultIndex);
      if (failed(mapping))
        return failure();
      mappings.push_back(*mapping);
    }
  }
  if (prototype && prototype->getShape().size() == shape.size()) {
    bool sameExtents = true;
    for (auto [left, right] : llvm::zip(prototype->getShape(), shape))
      sameExtents &= left == right;
    if (sameExtents) {
      SmallVector<Attribute> extents(shape.begin(), shape.end());
      SmallVector<Attribute> remapped;
      for (auto [axis, attribute] :
           llvm::enumerate(prototype->getAxisMaps())) {
        auto mapping = cast<AxisMapAttr>(attribute);
        remapped.push_back(AxisMapAttr::get(
            context, mapping.getSourceId(), mapping.getSourceAxis(),
            dimensions[axis], axis, mapping.getDerived()));
      }
      return FragmentType::get(context, tensor.getElementType(),
                               ArrayAttr::get(context, extents),
                               ArrayAttr::get(context, remapped),
                               prototype->getValidity(),
                               prototype->getOwner());
    }
  }
  return fragmentType(context, tensor.getElementType(), shape, mappings, owner);
}

FailureOr<Type> convertDataType(
    CanonicalKernelAnalysis &canonicalAnalysis,
    Type type,
    Operation *origin,
    std::optional<Type> prototype,
    uint64_t owner,
    unsigned resultIndex) {
  if (isa<IntegerType, FloatType, IndexType, LogicalIndexType>(type)) {
    Type element = convertScalarType(type, owner);
    if (prototype)
      if (auto fragment = dyn_cast<FragmentType>(*prototype))
        return Type(FragmentType::get(
            type.getContext(), element, fragment.getShape(),
            fragment.getAxisMaps(), fragment.getValidity(), fragment.getOwner()));
    return element;
  }
  if (auto tensor = dyn_cast<RankedTensorType>(type)) {
    std::optional<FragmentType> fragment;
    if (prototype)
      if (auto candidate = dyn_cast<FragmentType>(*prototype))
        fragment = candidate;
    FailureOr<FragmentType> converted =
        convertTensorType(canonicalAnalysis, tensor, origin, fragment, owner, resultIndex);
    if (failed(converted))
      return failure();
    return Type(*converted);
  }
  if (ArrayAttr components = getProductComponents(type)) {
    SmallVector<Attribute> names, fields;
    auto record = dyn_cast<intent::RecordType>(type);
    for (unsigned index = 0; index < components.size(); ++index)
      names.push_back(record ? record.getFieldNames()[index]
                             : Attribute(StringAttr::get(type.getContext(),
                                   ("_" + Twine(index)).str())));
    auto fieldNames = ArrayAttr::get(type.getContext(), names);
    auto prototypeRecord = prototype ? dyn_cast<gpu::RecordType>(*prototype)
                                     : gpu::RecordType();
    if (prototypeRecord &&
        (prototypeRecord.getFieldNames() != fieldNames ||
         prototypeRecord.getFieldTypes().size() != components.size()))
      return failure();
    for (auto [index, field] : llvm::enumerate(components)) {
      std::optional<Type> fieldPrototype;
      if (prototypeRecord)
        fieldPrototype = cast<TypeAttr>(prototypeRecord.getFieldTypes()[index]).getValue();
      auto converted = convertDataType(canonicalAnalysis,
          cast<TypeAttr>(field).getValue(), origin, fieldPrototype,
          prototypeRecord ? prototypeRecord.getOwner() : owner, resultIndex);
      if (failed(converted)) return failure();
      fields.push_back(TypeAttr::get(*converted));
    }
    return Type(gpu::RecordType::get(type.getContext(), fieldNames,
        ArrayAttr::get(type.getContext(), fields),
        prototypeRecord ? prototypeRecord.getOwner() : owner));
  }
  return failure();
}

} // namespace intent::kir_to_gpu
