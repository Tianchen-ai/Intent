#include "Construction.h"
#include "ConstructionSchema.h"
#include "Intent/Analysis/ProductSchema.h"
#include "Intent/Dialect/Block/IR/BlockOps.h"
#include "Intent/Dialect/Block/IR/Program.h"
#include "Intent/Dialect/Block/Transforms/Value/ValueMaterialization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;

namespace intent::kir_to_block {

Value createBinary(
    OpBuilder &builder,
    Location location,
    Type result,
    Value lhs,
    Value rhs,
    BinaryOperator kind) {
  return builder.createOrFold<block::BinaryOp>(location, result, lhs, rhs, kind);
}

FailureOr<Value> retargetBroadcast(
    OpBuilder &builder,
    Location location,
    Value value,
    block::FragmentType target) {
  auto source = dyn_cast<block::FragmentType>(value.getType());
  if (!source)
    return failure();
  if (samePhysicalShape(source, target))
    return value;
  auto result = block::FragmentType::get(
      value.getContext(), source.getElementType(), target.getShape(),
      target.getAxisMaps(), target.getValidity(), target.getOwner());
  Operation *replacement = nullptr;
  if (auto broadcast = value.getDefiningOp<block::BroadcastOp>())
    replacement = builder.create<block::BroadcastOp>(location, result,
                                                    broadcast.getValue());
  else if (auto splat = value.getDefiningOp<block::SplatOp>())
    replacement = builder.create<block::SplatOp>(location, result,
                                                splat.getValue());
  else
    replacement = builder.create<block::BroadcastOp>(location, result, value);
  if (Operation *definition = value.getDefiningOp())
    if (Attribute origin = definition->getAttr(block::originAttr))
      replacement->setAttr(block::originAttr, origin);
  return replacement->getResult(0);
}

FailureOr<Value> projectPositionalValue(
    OpBuilder &builder,
    Location location,
    Value identity,
    Type targetType) {
  if (identity.getType() == targetType)
    return identity;
  auto source = dyn_cast<block::FragmentType>(identity.getType());
  auto target = dyn_cast<block::FragmentType>(targetType);
  if (source && target && source.getElementType() == target.getElementType() &&
      source.getShape() == target.getShape() &&
      source.getOwner() == target.getOwner() &&
      source.getValidity() == target.getValidity()) {
    SmallVector<Attribute> groups;
    for (unsigned axis = 0; axis < source.getShape().size(); ++axis)
      groups.push_back(block::ReshapeGroupAttr::get(
          builder.getContext(), builder.getDenseI64ArrayAttr({axis}),
          builder.getDenseI64ArrayAttr({axis})));
    return Value(builder.create<block::ReshapeOp>(
        location, target, identity, builder.getArrayAttr(groups)));
  }
  return block::projectPhysicalValueToSchema(builder, location, identity,
                                           targetType);
}

Value createCompare(
    OpBuilder &builder,
    Location location,
    Type result,
    Value lhs,
    Value rhs,
    ComparePredicate predicate) {
  return builder.create<block::CompareOp>(location, result, lhs, rhs, predicate);
}

std::optional<int64_t> integerConstant(Value value) {
  Attribute attribute;
  if (auto constant = value.getDefiningOp<intent::ConstantOp>())
    attribute = constant.getValue();
  else if (auto constant = value.getDefiningOp<arith::ConstantOp>())
    attribute = constant.getValue();
  auto integer = dyn_cast_or_null<IntegerAttr>(attribute);
  return integer ? std::optional<int64_t>(integer.getInt()) : std::nullopt;
}

LogicalResult ScalarRegionLowering::alignOperands(
    Operation *operation,
    std::initializer_list<Value *> operands,
    ArrayRef<unsigned> positions) {
  Type logical = operation->getResult(0).getType();
  if (isa<intent::JoinOp>(operation))
    logical = operation->getOperand(0).getType();
  auto seed = convertDataType(canonicalAnalysis, logical, operation);
  if (failed(seed)) return failure();
  SmallVector<Value> current;
  for (Value *operand : operands) current.push_back(*operand);
  if (failed(detail::alignPointwiseOperands(
          builder, operation, canonicalAnalysis, physicalKernel,
          dyn_cast<block::FragmentType>(*seed), current, positions)))
    return failure();
  for (auto [operand, value] : llvm::zip(operands, current)) *operand = value;
  return success();
}

FailureOr<Type> ScalarRegionLowering::elementwiseResultType(
    Type logical,
    Operation *origin,
    Value prototype) {
  auto fragment = dyn_cast<block::FragmentType>(prototype.getType());
  if (!fragment)
    return convertDataType(canonicalAnalysis, logical, origin, prototype.getType());
  Type element;
  if (auto tensor = dyn_cast<RankedTensorType>(logical))
    element = tensor.getElementType();
  else if (isa<IntegerType, FloatType, IndexType, LogicalIndexType>(logical))
    element = convertScalarType(logical, fragment.getOwner());
  else
    return failure();
  return Type(block::FragmentType::get(
      logical.getContext(), element, fragment.getShape(),
      fragment.getAxisMaps(), fragment.getValidity(), fragment.getOwner()));
}

LogicalResult ScalarRegionLowering::lower(intent::ConstantOp constant) {
  Operation *operation = constant.getOperation();
  Location location = constant.getLoc();
  FailureOr<Value> value = block::materializeScalarConstant(
      builder, location, constant.getValue(), constant.getResult().getType());
  if (failed(value))
    return constant.emitOpError(
        "constant cannot be represented by its physical result type");
  values[constant.getResult()] = *value;
  if (Operation *target = value->getDefiningOp())
    attachOrigin(operation, target);
  return success();
}

LogicalResult ScalarRegionLowering::lower(intent::FullOp full) {
  Operation *operation = full.getOperation();
  Location location = full.getLoc();
  FailureOr<Value> fill = get(full.getInputs().front());
  FailureOr<Type> result =
      convertDataType(canonicalAnalysis, full.getResult().getType(), operation);
  if (failed(fill) || failed(result) || !isa<block::FragmentType>(*result))
    return full.emitOpError("full has no physical fragment realization");
  auto targetType = cast<block::FragmentType>(*result);
  auto logical = cast<RankedTensorType>(full.getResult().getType());
  SmallVector<Attribute> shape(targetType.getShape().begin(),
                               targetType.getShape().end());
  SmallVector<Attribute> mappings(targetType.getAxisMaps().begin(),
                                  targetType.getAxisMaps().end());
  for (unsigned resultAxis = 0; resultAxis < shape.size(); ++resultAxis) {
    // A bound on a dynamic logical extent is not Full's actual member count.
    // Seed it scalarly unless an explicit tensor shape below supplies the
    // current physical capacity. Static KIR extents retain their exact size.
    if (logical.isDynamicDim(resultAxis))
      shape[resultAxis] = expression(operation->getContext(),
                                     PhysicalExprKind::Constant, 1);
    // A shape explicitly taken from a tensor may reuse that tensor's current
    // physical capacity. This does not materialize its logical .shape value.
    auto extent = cast<ShapeExprAttr>(full.getShape().getAxes()[resultAxis]);
    auto dim = extent.getKind() == 1
        ? full->getOperand(extent.getPayload()).getDefiningOp<intent::DimOp>()
        : intent::DimOp();
    if (!dim)
      continue;
    FailureOr<Value> source = get(dim.getSource());
    auto fragment = succeeded(source)
                        ? dyn_cast<block::FragmentType>((*source).getType())
                        : block::FragmentType();
    auto sourceAxis = fragment
        ? physicalResourceAxis(dim.getSource().getType(), fragment, dim.getAxis())
        : FailureOr<unsigned>(failure());
    if (failed(sourceAxis))
      continue;
    auto sourceMapping =
        cast<block::AxisMapAttr>(fragment.getAxisMaps()[*sourceAxis]);
    shape[resultAxis] = fragment.getShape()[*sourceAxis];
    mappings[resultAxis] = block::AxisMapAttr::get(
        operation->getContext(), sourceMapping.getSourceId(),
        sourceMapping.getSourceAxis(), sourceMapping.getDimensionId(),
        resultAxis, sourceMapping.getDerived());
  }
  targetType = block::FragmentType::get(
      operation->getContext(), targetType.getElementType(),
      builder.getArrayAttr(shape), builder.getArrayAttr(mappings),
      targetType.getValidity(), targetType.getOwner());
  auto target =
      builder.create<block::SplatOp>(location, targetType, *fill);
  mapResults(operation, target);
  return success();
}

LogicalResult ScalarRegionLowering::lower(intent::BroadcastOp broadcastOp) {
  Operation *operation = broadcastOp.getOperation();
  Location location = broadcastOp.getLoc();
  FailureOr<Value> input = get(broadcastOp.getInputs().front());
  FailureOr<Type> result =
      convertDataType(canonicalAnalysis, broadcastOp.getResult().getType(), operation);
  if (failed(input) || failed(result) || !isa<block::FragmentType>(*result))
    return broadcastOp.emitOpError(
        "broadcast has no physical fragment realization");
  auto targetType = cast<block::FragmentType>(*result);
  auto sourceType = dyn_cast<block::FragmentType>((*input).getType());
  auto logicalSource =
      dyn_cast<RankedTensorType>(broadcastOp.getInputs().front().getType());
  auto logicalResult =
      dyn_cast<RankedTensorType>(broadcastOp.getResult().getType());
  if (sourceType && logicalResult) {
    unsigned logicalSourceRank = logicalSource ? logicalSource.getRank() : 0;
    if (logicalSourceRank > static_cast<unsigned>(logicalResult.getRank()))
      return broadcastOp.emitOpError(
          "broadcast source rank exceeds result rank");
    if (sourceType.getShape().size() < logicalSourceRank)
      return broadcastOp.emitOpError(
          "broadcast source lost a logical physical axis");

    // A scalar or tensor value can already be lifted over the current
    // physical workset before the logical broadcast is constructed.  Such
    // axes are not part of the logical source rank, but they remain part of
    // the value and therefore must survive the broadcast.  Prefix the
    // source-only workset axes, then retain the logical result axis order.
    SmallVector<Attribute> shape;
    SmallVector<Attribute> mappings;
    auto appendAxis = [&](Attribute extent, block::AxisMapAttr mapping) {
      mappings.push_back(block::AxisMapAttr::get(
          operation->getContext(), mapping.getSourceId(),
          mapping.getSourceAxis(), mapping.getDimensionId(), mappings.size(), mapping.getDerived()));
      shape.push_back(extent);
    };
    unsigned worksetRank = sourceType.getShape().size() - logicalSourceRank;
    for (unsigned sourceAxis = 0; sourceAxis < worksetRank; ++sourceAxis)
      appendAxis(sourceType.getShape()[sourceAxis],
                 cast<block::AxisMapAttr>(
                     sourceType.getAxisMaps()[sourceAxis]));
    auto projected = canonicalAnalysis.operandProjections(
        cast<OpResult>(broadcastOp.getResult()));
    if (failed(projected)) return failure();
    SmallVector<std::optional<unsigned>> logicalSources(logicalResult.getRank());
    for (const auto &projection : *projected)
      if (projection.operandNumber == 0)
        for (auto [sourceAxis, resultAxis] : llvm::enumerate(projection.resultAxes))
          if (resultAxis) logicalSources[*resultAxis] = sourceAxis;
    for (auto [resultAxis, attribute] :
         llvm::enumerate(targetType.getAxisMaps())) {
      auto resultMapping = cast<block::AxisMapAttr>(attribute);
      Attribute extent = targetType.getShape()[resultAxis];
      if (logicalSource && logicalSources[resultAxis]) {
        unsigned logicalSourceAxis = *logicalSources[resultAxis];
        unsigned physicalSourceAxis = worksetRank + logicalSourceAxis;
        bool expandsSingleton =
            !logicalSource.isDynamicDim(logicalSourceAxis) &&
            logicalSource.getDimSize(logicalSourceAxis) == 1 &&
            (logicalResult.isDynamicDim(resultAxis) ||
             logicalResult.getDimSize(resultAxis) != 1);
        if (!expandsSingleton)
          extent = sourceType.getShape()[physicalSourceAxis];
        if (!expandsSingleton) {
          auto sourceMapping = cast<block::AxisMapAttr>(
              sourceType.getAxisMaps()[physicalSourceAxis]);
          resultMapping = block::AxisMapAttr::get(
              operation->getContext(), sourceMapping.getSourceId(),
              sourceMapping.getSourceAxis(), sourceMapping.getDimensionId(),
              resultAxis, sourceMapping.getDerived());
        }
      }
      appendAxis(extent, resultMapping);
    }
    targetType = block::FragmentType::get(
        operation->getContext(), targetType.getElementType(),
        builder.getArrayAttr(shape), builder.getArrayAttr(mappings),
        targetType.getValidity(), sourceType.getOwner());
  }
  auto target =
      builder.create<block::BroadcastOp>(location, targetType, *input);
  mapResults(operation, target);
  return success();
}

LogicalResult ScalarRegionLowering::lower(intent::ReshapeOp reshape) {
  Operation *operation = reshape.getOperation();
  Location location = reshape.getLoc();
  FailureOr<Value> input = get(reshape.getInputs().front());
  auto source = succeeded(input)
                    ? dyn_cast<block::FragmentType>((*input).getType())
                    : block::FragmentType();
  auto logicalSource =
      dyn_cast<RankedTensorType>(reshape.getInputs().front().getType());
  auto logicalResult =
      dyn_cast<RankedTensorType>(reshape.getResult().getType());
  if (failed(input) || !source || !logicalSource || !logicalResult)
    return reshape.emitOpError(
        "reshape has no physical fragment realization");
  unsigned physicalSourceRank = source.getShape().size();
  unsigned logicalSourceRank = logicalSource.getRank();
  if (physicalSourceRank < logicalSourceRank)
    return reshape.emitOpError("reshape source lost a logical physical axis");
  unsigned worksetRank = physicalSourceRank - logicalSourceRank;
  auto resultMapping = [&](unsigned logicalResultAxis,
                           unsigned physicalResultAxis)
      -> FailureOr<Attribute> {
    DenseI64ArrayAttr dimensions = dimensionIds(logicalResult);
    FailureOr<PhysicalAxisIdentity> identity =
        resultAxisIdentity(operation, /*resultIndex=*/0, logicalResultAxis);
    if (!dimensions || logicalResultAxis >= dimensions.size() ||
        dimensions[logicalResultAxis] <= 0 || failed(identity))
      return failure();
    return Attribute(block::AxisMapAttr::get(
        operation->getContext(), identity->sourceId, identity->sourceAxis,
        dimensions[logicalResultAxis], physicalResultAxis,
        identity->derived));
  };
  SmallVector<Attribute> shape;
  SmallVector<Attribute> mappings;
  for (unsigned axis = 0; axis < worksetRank; ++axis) {
    shape.push_back(source.getShape()[axis]);
    auto mapping = cast<block::AxisMapAttr>(source.getAxisMaps()[axis]);
    mappings.push_back(block::AxisMapAttr::get(
        operation->getContext(), mapping.getSourceId(),
        mapping.getSourceAxis(), mapping.getDimensionId(), mappings.size(), mapping.getDerived()));
  }
  unsigned resultRank = logicalResult.getRank();
  auto groups = canonicalAnalysis.reshapeGroups(reshape);
  if (failed(groups))
    return reshape.emitOpError(
        "canonical reshape has no exact row-major logical relation");
  SmallVector<Attribute> reassociation;
  for (const LogicalReshapeGroup &group : *groups)
    reassociation.push_back(block::ReshapeGroupAttr::get(
        operation->getContext(), builder.getDenseI64ArrayAttr(group.sourceAxes),
        builder.getDenseI64ArrayAttr(group.resultAxes)));
  SmallVector<TensorExtentFact> logicalResultExtents;
  for (unsigned axis = 0; axis < resultRank; ++axis)
    logicalResultExtents.push_back(
        canonicalAnalysis.tensorExtent(reshape.getResult(), axis));

  SmallVector<PhysicalExprAttr> physicalResultExtents(resultRank);
  SmallVector<Attribute> physicalResultMappings;
  physicalResultMappings.reserve(resultRank);
  for (unsigned axis = 0; axis < resultRank; ++axis) {
    FailureOr<Attribute> mapping =
        resultMapping(axis, worksetRank + axis);
    if (failed(mapping))
      return reshape.emitOpError(
          "reshape result axis has no coordinate identity");
    physicalResultMappings.push_back(*mapping);
  }
  for (const LogicalReshapeGroup &group : *groups) {
    PhysicalExprAttr sourceProduct = expression(
        operation->getContext(), PhysicalExprKind::Constant, 1);
    for (int64_t logicalSourceAxis :
         group.sourceAxes)
      sourceProduct = binaryExpression(
          operation->getContext(), PhysicalExprKind::Multiply,
          sourceProduct,
          cast<PhysicalExprAttr>(
              source.getShape()[worksetRank + logicalSourceAxis]));
    PhysicalExprAttr knownProduct = expression(
        operation->getContext(), PhysicalExprKind::Constant, 1);
    SmallVector<unsigned> unresolved;
    for (int64_t logicalResultAxis :
         group.resultAxes) {
      if (logicalResultExtents[logicalResultAxis].constant == 1) {
        physicalResultExtents[logicalResultAxis] = expression(
            operation->getContext(), PhysicalExprKind::Constant, 1);
        continue;
      }
      std::optional<unsigned> matchedSource;
      for (int64_t logicalSourceAxis :
           group.sourceAxes) {
        if (!canonicalAnalysis.equalTensorExtents(
                reshape.getInputs().front(), logicalSourceAxis,
                reshape.getResult(), logicalResultAxis))
          continue;
        if (matchedSource) {
          matchedSource.reset();
          break;
        }
        matchedSource = logicalSourceAxis;
      }
      if (!matchedSource) {
        unresolved.push_back(logicalResultAxis);
        continue;
      }
      unsigned sourceAxis = worksetRank + *matchedSource;
      PhysicalExprAttr physicalExtent = cast<PhysicalExprAttr>(
          source.getShape()[sourceAxis]);
      physicalResultExtents[logicalResultAxis] = physicalExtent;
      knownProduct = binaryExpression(
          operation->getContext(), PhysicalExprKind::Multiply,
          knownProduct, physicalExtent);
      auto mapping = cast<block::AxisMapAttr>(
          source.getAxisMaps()[sourceAxis]);
      physicalResultMappings[logicalResultAxis] = block::AxisMapAttr::get(
          operation->getContext(), mapping.getSourceId(),
          mapping.getSourceAxis(), mapping.getDimensionId(),
          worksetRank + logicalResultAxis, mapping.getDerived());
    }
    if (unresolved.size() > 1) {
      auto knownKind =
          knownProduct.getKind();
      bool allConstant = knownKind == PhysicalExprKind::Constant;
      int64_t logicalConstantProduct =
          allConstant ? knownProduct.getValue() : 0;
      for (unsigned axis : unresolved) {
        auto logicalExtent = logicalResultExtents[axis].constant;
        if (!logicalExtent) {
          allConstant = false;
          continue;
        }
        if (allConstant)
          logicalConstantProduct *= *logicalExtent;
      }
      auto sourceKind = sourceProduct.getKind();
      if (allConstant && sourceKind == PhysicalExprKind::Constant &&
          logicalConstantProduct == sourceProduct.getValue()) {
        for (unsigned axis : unresolved)
          physicalResultExtents[axis] = expression(
              operation->getContext(), PhysicalExprKind::Constant,
              *logicalResultExtents[axis].constant);
        knownProduct = sourceProduct;
        unresolved.clear();
      } else if (sourceKind == PhysicalExprKind::Constant &&
                 sourceProduct.getValue() == 1) {
        PhysicalExprAttr one = expression(
            operation->getContext(), PhysicalExprKind::Constant, 1);
        for (unsigned axis : unresolved)
          physicalResultExtents[axis] = one;
        knownProduct = sourceProduct;
        unresolved.clear();
      } else {
        return reshape.emitOpError(
            "reshape physical group has more than one unbound result axis");
      }
    }
    if (unresolved.size() == 1) {
      physicalResultExtents[unresolved.front()] = binaryExpression(
          operation->getContext(), PhysicalExprKind::FloorDiv,
          sourceProduct, knownProduct);
      knownProduct = sourceProduct;
    }
    if (knownProduct != sourceProduct)
      return reshape.emitOpError(
          "reshape physical group does not preserve row-major elements");
  }
  for (auto [axis, extent] : llvm::enumerate(physicalResultExtents)) {
    if (!extent)
      return reshape.emitOpError(
          "reshape physical result axis has no extent decision");
    shape.push_back(extent);
    mappings.push_back(physicalResultMappings[axis]);
  }

  auto result = block::FragmentType::get(
      operation->getContext(), source.getElementType(),
      builder.getArrayAttr(shape), builder.getArrayAttr(mappings),
      source.getValidity(), source.getOwner());
  auto target = builder.create<block::ReshapeOp>(
      location, result, *input, builder.getArrayAttr(reassociation));
  mapResults(operation, target);
  return success();
}

LogicalResult ScalarRegionLowering::lower(intent::TransposeOp transpose) {
  Operation *operation = transpose.getOperation();
  Location location = transpose.getLoc();
  FailureOr<Value> input = get(transpose.getInput());
  auto source = succeeded(input)
                    ? dyn_cast<block::FragmentType>((*input).getType())
                    : block::FragmentType();
  auto logical = dyn_cast<RankedTensorType>(transpose.getInput().getType());
  if (failed(input) || !source || !logical ||
      source.getShape().size() < static_cast<size_t>(logical.getRank()))
    return transpose.emitOpError(
        "transpose has no physical fragment realization");
  SmallVector<int64_t> permutation;
  for (Attribute axis : transpose.getPermutation())
    permutation.push_back(cast<IntegerAttr>(axis).getInt());
  unsigned prefix = source.getShape().size() - logical.getRank();
  SmallVector<Attribute> shape;
  SmallVector<Attribute> mappings;
  auto appendAxis = [&](unsigned sourceAxis) {
    shape.push_back(source.getShape()[sourceAxis]);
    auto mapping =
        cast<block::AxisMapAttr>(source.getAxisMaps()[sourceAxis]);
    mappings.push_back(block::AxisMapAttr::get(
        operation->getContext(), mapping.getSourceId(),
        mapping.getSourceAxis(), mapping.getDimensionId(), mappings.size(), mapping.getDerived()));
  };
  for (unsigned axis = 0; axis < prefix; ++axis)
    appendAxis(axis);
  SmallVector<int64_t> physicalPermutation;
  for (unsigned axis = 0; axis < prefix; ++axis)
    physicalPermutation.push_back(axis);
  for (int64_t axis : permutation) {
    if (axis < 0 || axis >= logical.getRank())
      return transpose.emitOpError(
          "transpose permutation is outside the logical rank");
    appendAxis(prefix + axis);
    physicalPermutation.push_back(prefix + axis);
  }
  auto result = block::FragmentType::get(
      operation->getContext(), source.getElementType(),
      builder.getArrayAttr(shape), builder.getArrayAttr(mappings),
      source.getValidity(), source.getOwner());
  auto target = builder.create<block::TransposeOp>(
      location, result, *input, physicalPermutation);
  mapResults(operation, target);
  return success();
}

LogicalResult ScalarRegionLowering::lower(intent::JoinOp join) {
  Operation *operation = join.getOperation();
  Location location = join.getLoc();
  FailureOr<Value> lhs = get(join.getLhs());
  FailureOr<Value> rhs = get(join.getRhs());
  if (failed(lhs) || failed(rhs) ||
      failed(alignOperands(operation, {&*lhs, &*rhs}, {0, 1})))
    return join.emitOpError(
        "join operands have no common physical fragment schema");
  auto left = succeeded(lhs) ? dyn_cast<block::FragmentType>((*lhs).getType())
                             : block::FragmentType();
  auto right = succeeded(rhs) ? dyn_cast<block::FragmentType>((*rhs).getType())
                              : block::FragmentType();
  auto logicalLeft = dyn_cast<RankedTensorType>(join.getLhs().getType());
  auto logicalRight = dyn_cast<RankedTensorType>(join.getRhs().getType());
  auto logical = dyn_cast<RankedTensorType>(join.getResult().getType());
  if (!left || !right || left != right ||
      !logicalLeft || !logicalRight || !logical ||
      logicalLeft.getRank() != logicalRight.getRank() ||
      logical.getRank() != logicalLeft.getRank() + 1)
    return join.emitOpError("join has no physical fragment realization")
           << "; lhs=" << (left ? Type(left) : Type())
           << ", rhs=" << (right ? Type(right) : Type())
           << ", logical_lhs=" << join.getLhs().getType()
           << ", logical_result=" << join.getResult().getType();
  FailureOr<PhysicalAxisIdentity> trailingIdentity = resultAxisIdentity(
      operation, /*resultIndex=*/0, logical.getRank() - 1);
  DenseI64ArrayAttr dimensions = dimensionIds(logical);
  if (!dimensions || dimensions.size() != logical.getRank() ||
      dimensions[logical.getRank() - 1] <= 0 || failed(trailingIdentity))
    return join.emitOpError(
        "join trailing axis has no canonical coordinate identity");
  SmallVector<Attribute> shape(left.getShape().begin(),
                               left.getShape().end());
  shape.push_back(expression(operation->getContext(),
                             PhysicalExprKind::Constant, 2));
  SmallVector<Attribute> mappings(left.getAxisMaps().begin(),
                                  left.getAxisMaps().end());
  mappings.push_back(block::AxisMapAttr::get(
      operation->getContext(), trailingIdentity->sourceId,
      trailingIdentity->sourceAxis,
      dimensions[logical.getRank() - 1],
      left.getShape().size(), trailingIdentity->derived));
  auto result = block::FragmentType::get(
      operation->getContext(), left.getElementType(),
      builder.getArrayAttr(shape), builder.getArrayAttr(mappings),
      left.getValidity(), left.getOwner());
  auto target = builder.create<block::JoinOp>(
      location, result, *lhs, *rhs, result.getShape().size() - 1);
  mapResults(operation, target);
  return success();
}

LogicalResult ScalarRegionLowering::lower(intent::UnaryOp unary) {
  Operation *operation = unary.getOperation();
  Location location = unary.getLoc();
  FailureOr<Value> input = get(unary.getInput());
  if (failed(input))
    return failure();
  FailureOr<Type> result =
      elementwiseResultType(unary.getResult().getType(), operation, *input);
  if (failed(result))
    return unary.emitOpError("unary value has no physical data type");
  auto target = builder.create<block::UnaryOp>(location, *result, *input,
      unary.getOperatorKind(), unary.getApproximate(), unary.getFlushToZero());
  mapResults(operation, target);
  return success();
}

LogicalResult ScalarRegionLowering::lower(intent::BinaryOp binary) {
  Operation *operation = binary.getOperation();
  Location location = binary.getLoc();
  FailureOr<Value> lhs = get(binary.getLhs());
  FailureOr<Value> rhs = get(binary.getRhs());
  if (failed(lhs) || failed(rhs))
    return binary.emitOpError("binary operands have no physical data schema");
  if (failed(alignOperands(operation, {&*lhs, &*rhs}, {0, 1})))
    return binary.emitOpError(
               "binary operands have incompatible physical schemas; lhs=")
           << (*lhs).getType() << ", rhs=" << (*rhs).getType();
  FailureOr<Type> result =
      elementwiseResultType(binary.getResult().getType(), operation, *lhs);
  if (failed(result))
    return binary.emitOpError("binary result has no physical data schema");
  auto target = builder.create<block::BinaryOp>(location, *result, *lhs, *rhs,
      binary.getOperatorKind(), binary.getApproximate(), binary.getFlushToZero());
  mapResults(operation, target);
  return success();
}

LogicalResult ScalarRegionLowering::lower(intent::CompareOp compare) {
  Operation *operation = compare.getOperation();
  Location location = compare.getLoc();
  FailureOr<Value> lhs = get(compare.getLhs());
  FailureOr<Value> rhs = get(compare.getRhs());
  if (failed(lhs) || failed(rhs) ||
      failed(alignOperands(operation, {&*lhs, &*rhs}, {0, 1})))
    return compare.emitOpError("comparison operands are unavailable");
  // A comparison changes only the element type.  Its predicate executes
  // over exactly the same physical fragment as the aligned operands.
  FailureOr<Type> result =
      elementwiseResultType(compare.getResult().getType(), operation, *lhs);
  if (failed(result))
    return compare.emitOpError("comparison result is unavailable");
  auto target = builder.create<block::CompareOp>(location, *result, *lhs, *rhs,
                                                compare.getPredicate());
  mapResults(operation, target);
  return success();
}

LogicalResult ScalarRegionLowering::lower(intent::SelectOp select) {
  Operation *operation = select.getOperation();
  Location location = select.getLoc();
  FailureOr<Value> condition = get(select.getCondition());
  FailureOr<Value> trueValue = get(select.getTrueValue());
  FailureOr<Value> falseValue = get(select.getFalseValue());
  if (failed(condition) || failed(trueValue) || failed(falseValue) ||
      failed(alignOperands(operation, {&*condition, &*trueValue, &*falseValue}, {0, 1, 2})))
    return select.emitOpError("select operands are unavailable");
  FailureOr<Type> result = elementwiseResultType(
      select.getResult().getType(), operation, *trueValue);
  if (failed(result))
    return select.emitOpError("select result is unavailable");
  auto target = builder.create<block::SelectOp>(
      location, *result, *condition, *trueValue, *falseValue);
  mapResults(operation, target);
  return success();
}

LogicalResult ScalarRegionLowering::lower(intent::CastOp cast) {
  Operation *operation = cast.getOperation();
  Location location = cast.getLoc();
  FailureOr<Value> input = get(cast.getInput());
  FailureOr<Type> result =
      failed(input) ? FailureOr<Type>(failure())
                    : elementwiseResultType(cast.getResult().getType(),
                                            operation, *input);
  if (failed(input) || failed(result))
    return cast.emitOpError("cast input/result is unavailable");
  if (!isa<block::FragmentType>(*result)) {
    if (auto constant = (*input).getDefiningOp<arith::ConstantOp>()) {
      FailureOr<Value> folded = block::materializeScalarConstant(
          builder, location, constant.getValue(), *result);
      if (succeeded(folded)) {
        values[cast.getResult()] = *folded;
        return success();
      }
    }
  }
  auto target = builder.create<block::CastOp>(location, *result, *input);
  mapResults(operation, target);
  return success();
}

LogicalResult ScalarRegionLowering::lower(intent::BitcastOp bitcast) {
  Operation *operation = bitcast.getOperation();
  Location location = bitcast.getLoc();
  FailureOr<Value> input = get(bitcast.getInput());
  FailureOr<Type> result =
      failed(input) ? FailureOr<Type>(failure())
                    : elementwiseResultType(bitcast.getResult().getType(),
                                            operation, *input);
  if (failed(input) || failed(result))
    return bitcast.emitOpError("bitcast input/result is unavailable");
  auto target = builder.create<block::BitcastOp>(location, *result, *input);
  mapResults(operation, target);
  return success();
}

LogicalResult ScalarRegionLowering::lower(intent::MaskOp mask) {
  Operation *operation = mask.getOperation();
  Location location = mask.getLoc();
  FailureOr<Value> value = get(mask.getValue());
  FailureOr<Value> predicate = get(mask.getPredicate());
  FailureOr<Value> fill = get(mask.getFill());
  if (failed(value) || failed(predicate) || failed(fill) ||
      failed(alignOperands(operation, {&*value, &*predicate, &*fill}, {0, 1, 2})))
    return mask.emitOpError("mask operands are unavailable");
  FailureOr<Type> result =
      failed(value) ? FailureOr<Type>(failure())
                    : elementwiseResultType(mask.getResult().getType(),
                                            operation, *value);
  if (failed(result))
    return mask.emitOpError("mask operands are unavailable");
  auto target = builder.create<block::SelectOp>(
      location, *result, *predicate, *value, *fill);
  mapResults(operation, target);
  return success();
}

LogicalResult ScalarRegionLowering::lower(intent::RandomBitsOp random) {
  Operation *operation = random.getOperation();
  Location location = random.getLoc();
  FailureOr<Value> seed = get(random.getSeed());
  FailureOr<Value> counter = get(random.getLogicalCounter());
  FailureOr<Type> result =
      failed(counter)
          ? FailureOr<Type>(failure())
          : convertDataType(canonicalAnalysis, random.getResult().getType(), operation,
                            (*counter).getType());
  if (failed(seed) || failed(counter) || failed(result))
    return random.emitOpError("Philox operands are unavailable");
  auto seedType = dyn_cast<IntegerType>((*seed).getType());
  if (!seedType || seedType.getWidth() != 64)
    return random.emitOpError("Philox seed is not a 64-bit integer");
  if (seedType != builder.getI64Type())
    *seed = builder.create<block::BitcastOp>(location, builder.getI64Type(),
                                           *seed);
  auto target = builder.create<block::RandomBitsOp>(location, *result, *seed,
                                                   *counter);
  mapResults(operation, target);
  return success();
}

LogicalResult ScalarRegionLowering::lower(intent::MakeRecordOp record) {
  Operation *operation = record.getOperation();
  Location location = record.getLoc();
  SmallVector<Value> fields;
  for (Value field : record.getFields()) {
    FailureOr<Value> lowered = get(field);
    if (failed(lowered))
      return failure();
    fields.push_back(*lowered);
  }
  FailureOr<Type> result =
      convertDataType(canonicalAnalysis, record.getResult().getType(), operation);
  if (failed(result))
    return record.emitOpError("record contains an unphysical field");
  auto schema = cast<block::RecordType>(*result);
  SmallVector<Attribute> fieldTypes;
  for (Value field : fields)
    fieldTypes.push_back(TypeAttr::get(field.getType()));
  auto resultType = block::RecordType::get(
      operation->getContext(), schema.getFieldNames(),
      builder.getArrayAttr(fieldTypes), schema.getOwner());
  auto target =
      builder.create<block::MakeRecordOp>(location, resultType, fields);
  mapResults(operation, target);
  return success();
}

LogicalResult ScalarRegionLowering::lower(intent::MakeTupleOp tuple) {
  Operation *operation = tuple.getOperation();
  Location location = tuple.getLoc();
  SmallVector<Value> fields;
  for (Value field : tuple.getComponents()) {
    FailureOr<Value> lowered = get(field);
    if (failed(lowered))
      return failure();
    fields.push_back(*lowered);
  }
  FailureOr<Type> result =
      convertDataType(canonicalAnalysis, tuple.getResult().getType(), operation);
  if (failed(result))
    return tuple.emitOpError("tuple contains an unphysical field");
  auto schema = cast<block::RecordType>(*result);
  SmallVector<Attribute> fieldTypes;
  for (Value field : fields)
    fieldTypes.push_back(TypeAttr::get(field.getType()));
  auto resultType = block::RecordType::get(
      operation->getContext(), schema.getFieldNames(),
      builder.getArrayAttr(fieldTypes), schema.getOwner());
  auto target =
      builder.create<block::MakeRecordOp>(location, resultType, fields);
  mapResults(operation, target);
  return success();
}

LogicalResult ScalarRegionLowering::lower(intent::ExtractOp extract) {
  Operation *operation = extract.getOperation();
  Location location = extract.getLoc();
  FailureOr<Value> product = get(extract.getProduct());
  if (failed(product))
    return extract.emitOpError("product projection is unavailable");
  auto recordType = dyn_cast<block::RecordType>((*product).getType());
  int64_t field = extract.getField();
  if (!recordType || field < 0 ||
      field >= static_cast<int64_t>(recordType.getFieldTypes().size()))
    return extract.emitOpError("product projection field is unavailable");
  Type result = cast<TypeAttr>(recordType.getFieldTypes()[field]).getValue();
  auto target = builder.create<block::ExtractOp>(location, result, *product,
                                                extract.getField());
  mapResults(operation, target);
  return success();
}

} // namespace intent::kir_to_block
