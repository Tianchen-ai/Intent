#include "Construction.h"
#include "Intent/Conversion/LogicalShape.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/IR/ProgramInterface.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::kir_to_gpu {

PhysicalExprAttr dimensionExpression(func::FuncOp function, int64_t dimension) {
  return PhysicalExprAttr::get(function.getContext(), PhysicalExprKind::Dimension,
      dimension, gpu::ArgumentRefAttr::get(function.getContext(),
          function.getNumArguments() + dimension),
      ArrayAttr::get(function.getContext(), {}));
}

FailureOr<PhysicalABI> buildPhysicalABI(
    func::FuncOp function,
    OpBuilder &builder) {
  PhysicalABI result;
  MLIRContext *context = function.getContext();
  Block &sourceEntry = function.getBody().front();

  auto interface = buildPublicInterface(function);
  if (failed(interface)) return failure();
  result.interface = *interface;
  // Source argument identities and logical dimension identities occupy disjoint
  // ranges. Neither is a physical slot; later appends allocate fresh identities.
  uint64_t nextReference = sourceEntry.getNumArguments() + 1;
  for (BlockArgument argument : sourceEntry.getArguments())
    if (auto tensor = viewTensor(argument))
      if (auto ids = dimensionIds(tensor))
        for (int64_t identity : ids.asArrayRef())
          nextReference = std::max(nextReference,
              uint64_t(sourceEntry.getNumArguments()) + identity + 1);
  SmallVector<std::pair<Type, DictionaryAttr>> strideArguments;
  auto bindingAttrs = [&](gpu::ArgumentBindingAttr binding) {
    return builder.getDictionaryAttr({builder.getNamedAttr(gpu::argumentBindingAttr, binding)});
  };
  unsigned publicOrdinal = 0;

  for (auto [abi, argument] : llvm::enumerate(sourceEntry.getArguments())) {
    auto parameter = getSourceParameter(argument);
    if (!parameter)
      return function.emitError("canonical argument has no source parameter binding");
    if (isa<ConstexprType>(argument.getType())) {
      if (!argument.use_empty())
        return function.emitError("constexpr parameter must be specialized before physical construction");
      result.physicalArgumentForSource.push_back(std::nullopt);
      continue;
    }
    result.physicalArgumentForSource.push_back(result.arguments.size());
    auto reference = gpu::ArgumentRefAttr::get(context, abi + 1);
    auto binding = gpu::ArgumentBindingAttr::get(context, reference,
        gpu::ArgumentKind::Public, builder.getI64IntegerAttr(publicOrdinal++),
        gpu::ArgumentRefAttr{}, IntegerAttr{}, IntegerAttr{});
    result.argumentAttrs.push_back(bindingAttrs(binding));
    auto logicalView = dyn_cast<ViewType>(argument.getType());
    if (!logicalView) {
      Type physicalType = argument.getType();
      if (!isa<IntegerType, IndexType, FloatType>(physicalType))
        return function.emitError("runtime scalar ABI parameter has a non-scalar type");
      result.arguments.push_back(physicalType);
      continue;
    }
    auto tensor = dyn_cast<RankedTensorType>(logicalView.getTensor());
    DenseI64ArrayAttr canonicalIds =
        tensor ? dimensionIds(tensor) : DenseI64ArrayAttr();
    if (!tensor || parameter.getOriginId() < 0 ||
        (canonicalIds && canonicalIds.size() != tensor.getRank()))
      return function.emitError(
          "view lost canonical source or dimension identities");
    uint64_t sourceId = static_cast<uint64_t>(parameter.getOriginId()) + 1;
    SmallVector<int64_t> ids(tensor.getRank(), 0);
    if (canonicalIds)
      llvm::copy(canonicalIds.asArrayRef(), ids.begin());
    for (unsigned axis = 0; axis < static_cast<unsigned>(tensor.getRank()); ++axis)
      if (tensor.isDynamicDim(axis) && ids[axis] <= 0)
        return function.emitError(
            "dynamic view axis lost its canonical dimension identity");
    SmallVector<Attribute> strides;
    SmallVector<Attribute> extents;
    for (unsigned axis = 0; axis < static_cast<unsigned>(tensor.getRank()); ++axis) {
      auto strideReference = gpu::ArgumentRefAttr::get(context, nextReference++);
      strides.push_back(PhysicalExprAttr::get(context, PhysicalExprKind::ScalarABI,
          0, strideReference, builder.getArrayAttr({})));
      strideArguments.emplace_back(builder.getIndexType(), bindingAttrs(
          gpu::ArgumentBindingAttr::get(context, strideReference, gpu::ArgumentKind::Stride,
              IntegerAttr{}, reference, builder.getI64IntegerAttr(axis), IntegerAttr{})));
      int64_t dimension = ids[axis];
      extents.push_back(
          tensor.isDynamicDim(axis)
              ? Attribute(dimensionExpression(function, dimension))
              : Attribute(expression(context, PhysicalExprKind::Constant,
                                     tensor.getDimSize(axis))));
      if (dimension > 0) {
        auto [binding, inserted] = result.dimensions.try_emplace(
            dimension,
            MetadataBinding{dimension, static_cast<unsigned>(abi), axis});
        // Automatic output allocation must derive shared dimensions from an
        // input, including inputs declared after the output in the source ABI.
        if (!inserted && logicalView.getAccess() != 1 &&
            cast<ViewType>(sourceEntry.getArgument(binding->second.sourceABI)
                               .getType()).getAccess() == 1) {
          binding->second.sourceABI = abi;
          binding->second.sourceAxis = axis;
        }
      }
    }
    auto layout = gpu::ViewLayoutAttr::get(
        context, ArrayAttr::get(context, extents),
        DenseI64ArrayAttr::get(context, ids),
        ArrayAttr::get(context, strides));
    result.arguments.push_back(gpu::ViewType::get(
        context, tensor.getElementType(), logicalView.getAccess(),
        sourceId, layout));
  }

  for (auto &entry : result.dimensions)
    result.dimensionOrder.push_back(entry.first);
  llvm::sort(result.dimensionOrder);
  for (int64_t dimension : result.dimensionOrder) {
    MetadataBinding &binding = result.dimensions.find(dimension)->second;
    result.arguments.push_back(builder.getIndexType());
    result.argumentAttrs.push_back(bindingAttrs(gpu::ArgumentBindingAttr::get(
        context, dimensionExpression(function, dimension).getArgumentReference(),
        gpu::ArgumentKind::Dimension, IntegerAttr{},
        gpu::ArgumentRefAttr::get(context, binding.sourceABI + 1),
        builder.getI64IntegerAttr(binding.sourceAxis), builder.getI64IntegerAttr(dimension))));
  }
  for (auto [type, attributes] : strideArguments) {
    result.arguments.push_back(type);
    result.argumentAttrs.push_back(attributes);
  }
  return result;
}

FailureOr<PhysicalExprAttr> launchExtentExpression(
    CanonicalKernelAnalysis &canonicalAnalysis, Value value, unsigned axis,
    func::FuncOp function, ArrayRef<unsigned> fieldPath) {
  MLIRContext *context = value.getContext();
  LogicalShapeReification reification;
  reification.leaf = [&](const TensorExtentFact &extent)
      -> FailureOr<OpFoldResult> {
    if (extent.constant)
      return OpFoldResult(expression(context, PhysicalExprKind::Constant,
                                     *extent.constant));
    if (extent.value) {
      auto result = launchExpression(extent.value, canonicalAnalysis, function);
      if (failed(result)) return failure();
      return OpFoldResult(*result);
    }
    if (extent.domain) {
      SmallVector<IterationAxis> axes;
      if (failed(collectIterationAxes(extent.domain, axes)) ||
          extent.axis >= axes.size())
        return failure();
      const IterationAxis &domain = axes[extent.axis];
      auto begin = launchExpression(domain.start, canonicalAnalysis, function);
      auto end = launchExpression(domain.stop, canonicalAnalysis, function);
      auto step = domain.step
          ? launchExpression(domain.step, canonicalAnalysis, function)
          : FailureOr<PhysicalExprAttr>(
                expression(context, PhysicalExprKind::Constant, 1));
      if (failed(begin) || failed(end) || failed(step)) return failure();
      if (begin->getKind() == PhysicalExprKind::Constant &&
          begin->getValue() == 0 &&
          step->getKind() == PhysicalExprKind::Constant &&
          step->getValue() == 1 &&
          (end->getKind() == PhysicalExprKind::Dimension ||
           (end->getKind() == PhysicalExprKind::Constant && end->getValue() >= 0)))
        return OpFoldResult(*end);
      auto zero = expression(context, PhysicalExprKind::Constant, 0);
      auto distance = binaryExpression(context, PhysicalExprKind::Subtract,
                                       *end, *begin);
      auto count = binaryExpression(context, PhysicalExprKind::CeilDiv,
                                    distance, *step);
      return OpFoldResult(binaryExpression(context, PhysicalExprKind::Maximum,
                                            count, zero));
    }
    auto argument = dyn_cast_or_null<BlockArgument>(extent.source);
    auto tensor = argument ? viewTensor(argument) : RankedTensorType();
    if (!argument || !function ||
        argument.getOwner() != &function.getBody().front() ||
        !getSourceParameter(argument) || !extent.fieldPath.empty() || !tensor ||
        extent.axis >= tensor.getRank())
      return failure();
    auto identities = dimensionIds(tensor);
    if (!identities || identities[extent.axis] <= 0) return failure();
    return OpFoldResult(dimensionExpression(function, identities[extent.axis]));
  };
  auto binary = [&](PhysicalExprKind kind, OpFoldResult lhs, OpFoldResult rhs)
      -> FailureOr<OpFoldResult> {
    return OpFoldResult(binaryExpression(
        context, kind, cast<PhysicalExprAttr>(cast<Attribute>(lhs)),
        cast<PhysicalExprAttr>(cast<Attribute>(rhs))));
  };
  reification.multiply = [&](OpFoldResult lhs, OpFoldResult rhs) {
    return binary(PhysicalExprKind::Multiply, lhs, rhs);
  };
  reification.exactDivide = [&](OpFoldResult lhs, OpFoldResult rhs) {
    return binary(PhysicalExprKind::FloorDiv, lhs, rhs);
  };
  reification.subtract = [&](OpFoldResult lhs, OpFoldResult rhs) {
    return binary(PhysicalExprKind::Subtract, lhs, rhs);
  };
  reification.ceilDivide = [&](OpFoldResult lhs, OpFoldResult rhs) {
    return binary(PhysicalExprKind::CeilDiv, lhs, rhs);
  };
  reification.maximum = [&](OpFoldResult lhs, OpFoldResult rhs) {
    return binary(PhysicalExprKind::Maximum, lhs, rhs);
  };
  auto result = reifyLogicalExtent(canonicalAnalysis, value, axis,
                                  reification, fieldPath);
  if (failed(result)) return failure();
  return cast<PhysicalExprAttr>(cast<Attribute>(*result));
}

FailureOr<PhysicalExprAttr> launchExpression(
    Value value, CanonicalKernelAnalysis &canonicalAnalysis,
    func::FuncOp function) {
  MLIRContext *context = value.getContext();
  Operation *definition = value.getDefiningOp();
  if (!definition) {
    auto argument = dyn_cast<BlockArgument>(value);
    if (!argument || !function || argument.getOwner() != &function.getBody().front())
      return failure();
    if (!getSourceParameter(argument)) return failure();
    Type type = value.getType();
    if (!isa<IntegerType, IndexType>(type))
      return failure();
    return PhysicalExprAttr::get(context, PhysicalExprKind::ScalarABI, 0,
        gpu::ArgumentRefAttr::get(context, argument.getArgNumber() + 1),
        ArrayAttr::get(context, {}));
  }
  if (!definition)
    return failure();
  if (auto constant = dyn_cast<intent::ConstantOp>(definition)) {
    auto integer = dyn_cast<IntegerAttr>(constant.getValue());
    if (!integer)
      return failure();
    return expression(context, PhysicalExprKind::Constant, integer.getInt());
  }
  if (auto dim = dyn_cast<intent::DimOp>(definition)) {
    return launchExtentExpression(canonicalAnalysis, dim.getSource(),
                                  dim.getAxis(), function);
  }
  if (auto binary = dyn_cast<intent::BinaryOp>(definition)) {
    FailureOr<PhysicalExprAttr> lhs =
        launchExpression(binary.getLhs(), canonicalAnalysis, function);
    FailureOr<PhysicalExprAttr> rhs =
        launchExpression(binary.getRhs(), canonicalAnalysis, function);
    if (failed(lhs) || failed(rhs))
      return failure();
    PhysicalExprKind kind;
    switch (binary.getOperatorKind()) {
    case BinaryOperator::Add:
      kind = PhysicalExprKind::Add;
      break;
    case BinaryOperator::Subtract:
      kind = PhysicalExprKind::Subtract;
      break;
    case BinaryOperator::Multiply:
      kind = PhysicalExprKind::Multiply;
      break;
    case BinaryOperator::FloorDivide:
      kind = PhysicalExprKind::FloorDiv;
      break;
    case BinaryOperator::Maximum:
    case BinaryOperator::MaximumNum:
      kind = PhysicalExprKind::Maximum;
      break;
    case BinaryOperator::Minimum:
    case BinaryOperator::MinimumNum:
      kind = PhysicalExprKind::Minimum;
      break;
    default:
      return failure();
    }
    return binaryExpression(context, kind, *lhs, *rhs);
  }
  return failure();
}

} // namespace intent::kir_to_gpu
