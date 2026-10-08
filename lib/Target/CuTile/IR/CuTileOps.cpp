#include "Intent/Target/CuTile/IR/CuTileOps.h"

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/ProgramInterface.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/IR/Program.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallBitVector.h"

#include <functional>

using namespace mlir;

namespace intent::cutile {

namespace {

Type elementType(Type type) {
  if (auto fragment = dyn_cast<gpu::FragmentType>(type))
    return fragment.getElementType();
  return type;
}

bool isIntegerCoordinate(Type type) {
  Type element = elementType(type);
  return element.isIntOrIndex();
}

bool samePhysicalDomain(gpu::FragmentType lhs, gpu::FragmentType rhs) {
  return lhs.getShape() == rhs.getShape() &&
         lhs.getAxisMaps() == rhs.getAxisMaps() &&
         lhs.getValidity() == rhs.getValidity() &&
         lhs.getOwner() == rhs.getOwner();
}

bool sameLogicalAxis(gpu::FragmentType lhs, unsigned lhsAxis,
                     gpu::FragmentType rhs, unsigned rhsAxis) {
  auto left = dyn_cast<gpu::AxisMapAttr>(lhs.getAxisMaps()[lhsAxis]);
  auto right = dyn_cast<gpu::AxisMapAttr>(rhs.getAxisMaps()[rhsAxis]);
  return left && right && left.getDimensionId() > 0 &&
         left.getDimensionId() == right.getDimensionId();
}

LogicalResult verifyCoordinateDomains(Operation *owner, ValueRange coordinates,
                                      Type targetType) {
  auto target = dyn_cast<gpu::FragmentType>(targetType);
  for (Value coordinate : coordinates) {
    if (!isIntegerCoordinate(coordinate.getType()))
      return owner->emitOpError("coordinates must have integer element type");
    auto source = dyn_cast<gpu::FragmentType>(coordinate.getType());
    if (!source)
      continue;
    if (!target || !samePhysicalDomain(source, target))
      return owner->emitOpError(
          "coordinate tile was not materialized on the selected physical domain");
  }
  return success();
}

LogicalResult verifySourceAxes(Operation *owner, ArrayRef<int64_t> sourceAxes,
                               unsigned rank) {
  if (sourceAxes.size() != rank)
    return owner->emitOpError("requires one source-axis mapping per view axis");
  llvm::SmallBitVector seen(rank);
  for (int64_t axis : sourceAxes) {
    if (axis < 0 || axis >= static_cast<int64_t>(rank) || seen.test(axis))
      return owner->emitOpError(
          "source-axis mapping must be a permutation of the view axes");
    seen.set(axis);
  }
  return success();
}

LogicalResult verifyResourceOrderedTile(Operation *owner, unsigned rank,
                                        gpu::FragmentType tile,
                                        ValueRange tileIndices) {
  if (tileIndices.size() != rank || tile.getShape().size() != rank)
    return owner->emitOpError(
        "requires one resource-ordered tile axis and index per view axis");
  for (unsigned axis = 0; axis < rank; ++axis) {
    auto mapping = dyn_cast<gpu::AxisMapAttr>(tile.getAxisMaps()[axis]);
    if (!mapping || mapping.getFragmentAxis() != axis)
      return owner->emitOpError(
          "tile relation is not indexed in resource-axis order");
    Type indexType = tileIndices[axis].getType();
    if (!indexType.isIndex() && !indexType.isSignlessInteger(32))
      return owner->emitOpError("tile-space indices must have index or i32 type");
  }
  return success();
}

LogicalResult verifySerializedCoordinateOrder(
    Operation *owner, ArrayRef<int64_t> sourceAxes) {
  for (auto [position, sourceAxis] : llvm::enumerate(sourceAxes))
    if (position != static_cast<size_t>(sourceAxis))
      return owner->emitOpError(
          "serialized coordinates must be in resource-axis order");
  return success();
}

LogicalResult verifyLoadLatency(Operation *operation, Value latency) {
  if (!latency)
    return success();
  auto parameter = latency.getDefiningOp<gpu::ParameterOp>();
  if (!parameter || parameter.getDeclaration().getRole() !=
                        gpu::ParameterRole::ProviderLoadPolicy ||
      !llvm::all_of(parameter.getDeclaration().getCandidates().asArrayRef(),
                    isLegalLoadPolicy))
    return operation->emitOpError(
        "load policy requires an inferred or explicit latency domain");
  return success();
}

LogicalResult verifyFullTileCondition(TileLoadOp load) {
  if (!load.getFullTiles())
    return success();
  auto view = load.getResource().getType();
  auto shape = load.getResult().getType().getShape();
  llvm::SmallBitVector covered(view.getRank());
  for (auto [axis, attribute] : llvm::enumerate(shape)) {
    auto width = cast<gpu::PhysicalExprAttr>(attribute);
    auto size = cast<gpu::PhysicalExprAttr>(view.getLayout().getExtents()[axis]);
    auto constant = gpu::PhysicalExprKind::Constant;
    covered[axis] = width.getKind() == constant && width.getValue() > 0 &&
                    (width.getValue() == 1 ||
                     (size.getKind() == constant &&
                      size.getValue() % width.getValue() == 0));
  }
  SmallVector<Value> pending{load.getFullTiles()};
  while (!pending.empty()) {
    Value predicate = pending.pop_back_val();
    if (matchPattern(predicate, m_Zero()))
      return success();
    if (matchPattern(predicate, m_One()))
      continue;
    if (auto conjunction = predicate.getDefiningOp<gpu::BinaryOp>();
        conjunction &&
        conjunction.getOperatorKind() == BinaryOperator::LogicalAnd) {
      pending.push_back(conjunction.getLhs());
      pending.push_back(conjunction.getRhs());
      continue;
    }
    auto equal = predicate.getDefiningOp<gpu::CompareOp>();
    if (!equal || equal.getPredicate() != ComparePredicate::Eq ||
        !matchPattern(equal.getRhs(), m_Zero()))
      return load.emitOpError("full tiles require a view-size divisibility proof");
    auto remainder = equal.getLhs().getDefiningOp<gpu::BinaryOp>();
    auto size = remainder ? remainder.getLhs().getDefiningOp<gpu::DimOp>()
                          : gpu::DimOp();
    auto width = remainder ? gpu::queryLaunchExpression(remainder.getRhs())
                           : gpu::PhysicalExprAttr();
    if (!remainder || remainder.getOperatorKind() != BinaryOperator::Remainder ||
        !size || !width || size.getView() != load.getResource() ||
        size.getAxis() >= view.getRank() ||
        width != shape[size.getAxis()])
      return load.emitOpError("full-tile proof must describe this native load");
    covered[size.getAxis()] = true;
  }
  return covered.all()
             ? success()
             : load.emitOpError("full-tile proof does not cover every view axis");
}

FailureOr<gpu::ViewType> collapsedArrayType(gpu::ViewType source,
                                          DenseI64ArrayAttr groups) {
  if (!source || !groups || groups.size() < 2 ||
      groups.size() >= source.getRank() || groups.asArrayRef().back() != source.getRank())
    return failure();
  SmallVector<Attribute> shape, strides;
  SmallVector<int64_t> dimensions;
  auto layout = source.getLayout();
  unsigned begin = 0;
  for (int64_t end : groups.asArrayRef()) {
    if (end <= begin || end > source.getRank())
      return failure();
    Attribute extent = layout.getExtents()[begin];
    for (unsigned axis = begin + 1; axis < static_cast<unsigned>(end); ++axis) {
      auto inner = dyn_cast<gpu::PhysicalExprAttr>(layout.getExtents()[axis]);
      if (!inner || inner.getKind() != gpu::PhysicalExprKind::Constant || inner.getValue() <= 1)
        return failure();
      extent = gpu::PhysicalExprAttr::get(
          source.getContext(), gpu::PhysicalExprKind::Multiply, 0,
          StringAttr::get(source.getContext()),
          ArrayAttr::get(source.getContext(), {extent, inner}));
    }
    shape.push_back(extent);
    strides.push_back(layout.getStrides()[end - 1]);
    dimensions.push_back(end == begin + 1 ? layout.getDimensionIds()[begin] : 0);
    begin = end;
  }
  return gpu::ViewType::get(
      source.getContext(), source.getElementType(), source.getAccess(), source.getSourceId(),
      gpu::ViewLayoutAttr::get(source.getContext(), ArrayAttr::get(source.getContext(), shape),
          DenseI64ArrayAttr::get(source.getContext(), dimensions),
          ArrayAttr::get(source.getContext(), strides)));
}

bool guardedByArrayEligibility(Operation *operation, ArrayViewOp array) {
  for (Region *region = operation->getParentRegion(); region;
       region = region->getParentRegion())
    if (auto branch = dyn_cast_or_null<scf::IfOp>(region->getParentOp());
        branch && region == &branch.getThenRegion() &&
        branch.getCondition() == array.getEligible())
      return true;
  return false;
}

} // namespace

Attribute getCompileTimeScalar(Value value) {
  if (Attribute literal =
          UniformValueAnalysis(gpu::describeUniformValue).evaluate(value))
    return literal;
  while (true) {
    auto description = gpu::describeUniformValue(value);
    if (description.kind == UniformKind::Forward &&
        description.operands.size() == 1) {
      value = description.operands.front();
      continue;
    }
    auto cast = value.getDefiningOp<gpu::CastOp>();
    if (!cast)
      break;
    Type source = gpu::uniformElementType(cast.getValue().getType());
    Type target = gpu::uniformElementType(cast.getResult().getType());
    if (source != target &&
        !(source.isIndex() && target.isSignlessInteger(64)) &&
        !(source.isSignlessInteger(64) && target.isIndex()))
      return {};
    value = cast.getValue();
  }
  auto expression = gpu::queryLaunchExpression(value);
  if (!expression)
    return {};
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  std::function<bool(gpu::PhysicalExprAttr)> isCompileTime =
      [&](gpu::PhysicalExprAttr current) {
    auto kind = current.getKind();
    if (kind == gpu::PhysicalExprKind::ScalarABI) {
      if (!kernel)
        return false;
      auto argument = gpu::resolveArgument(kernel, current.getArgumentReference());
      auto binding = argument ? gpu::getArgumentBinding(argument)
                              : gpu::ArgumentBindingAttr();
      return binding && binding.getKind() == gpu::ArgumentKind::Stride;
    }
    return llvm::all_of(current.getOperands(), [&](Attribute operand) {
      return isCompileTime(cast<gpu::PhysicalExprAttr>(operand));
    });
  };
  return isCompileTime(expression) ? Attribute(expression) : Attribute();
}

LogicalResult ArrayViewOp::verify() {
  auto base = dyn_cast<BlockArgument>(getBase());
  auto kernel = (*this)->getParentOfType<func::FuncOp>();
  auto view = getBase().getType();
  if (!base || !kernel || base.getOwner() != &kernel.getBody().front() ||
      (*this)->getBlock() != base.getOwner() || view.getAccess() != 0)
    return emitOpError(
        "requires a read-only kernel view as its source resource");
  auto expected = collapsedArrayType(view, getGroupEndsAttr());
  if (failed(expected) || getResult().getType() != *expected)
    return emitOpError("result must carry the collapsed source layout with positive non-unit inner extents");
  for (Operation *user : getResult().getUsers())
    if (!isa<TileLoadOp, gpu::DimOp>(user) || !guardedByArrayEligibility(user, *this))
      return emitOpError(
          "conditional array aliases require loads and dimensions inside their eligible control region");
  return success();
}

LogicalResult ArrayViewOp::inferReturnTypes(MLIRContext *context,
    std::optional<Location> location, ValueRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions, SmallVectorImpl<Type> &results) {
  Adaptor operation(operands, attributes, properties, regions);
  if (failed(operation.verify(location.value_or(UnknownLoc::get(context)))))
    return failure();
  auto view = collapsedArrayType(cast<gpu::ViewType>(operation.getBase().getType()),
                                 operation.getGroupEndsAttr());
  if (failed(view))
    return emitOptionalError(location, "invalid contiguous array collapse groups");
  results.push_back(*view);
  results.push_back(IntegerType::get(context, 1));
  return success();
}

Value ArrayViewOp::getViewSource() { return getBase(); }

ArrayAttr getNativeArrayIndexBounds(Value resource) {
  if (auto argument = dyn_cast<BlockArgument>(resource)) {
    auto function = dyn_cast_or_null<func::FuncOp>(argument.getOwner()->getParentOp());
    return function ? function.getArgAttrOfType<ArrayAttr>(argument.getArgNumber(),
                                                         arrayIndexTileBoundsAttr)
                    : ArrayAttr();
  }
  if (auto array = resource.getDefiningOp<ArrayViewOp>())
    return array->getAttrOfType<ArrayAttr>(arrayIndexTileBoundsAttr);
  return {};
}

LogicalResult TileLoadOp::verify() {
  auto view = getResource().getType();
  auto result = getResult().getType();
  if (!getAllowTma().getType().isInteger(1))
    return emitOpError("allow_tma must be a compile-time i1 access decision");
  if (failed(verifyLoadLatency(*this, getLatencyPolicy())))
    return failure();
  if (failed(verifyResourceOrderedTile(*this, view.getRank(), result,
                                       getTileIndices())))
    return failure();
  if (view.getElementType() != result.getElementType())
    return emitOpError("view and tile element types disagree");
  if (auto array = getResource().getDefiningOp<ArrayViewOp>();
      array && !guardedByArrayEligibility(*this, array))
    return emitOpError("collapsed array load requires its eligibility guard");
  return verifyFullTileCondition(*this);
}

void TileLoadOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getResourceMutable());
}

LogicalResult TileStoreOp::verify() {
  auto view = getResource().getType();
  if (!getAllowTma().getType().isInteger(1))
    return emitOpError("allow_tma must be a compile-time i1 access decision");
  if (getResource().getDefiningOp<ArrayViewOp>())
    return emitOpError("conditional array aliases are read-only");
  if (failed(verifyResourceOrderedTile(*this, view.getRank(), getValue().getType(),
                                       getTileIndices())))
    return failure();
  return view.getElementType() == getValue().getType().getElementType()
             ? success()
             : emitOpError("view and tile element types disagree");
}

void TileStoreOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getResourceMutable());
}

LogicalResult TileAtomicAddOp::verify() {
  auto view = getResource().getType();
  Type element = view.getElementType();
  if ((!element.isF16() && !element.isBF16()) ||
      element != getValue().getType().getElementType())
    return emitOpError("requires matching f16 or bf16 view and tile elements");
  if (getResource().getDefiningOp<ArrayViewOp>())
    return emitOpError("conditional array aliases are read-only");
  return verifyResourceOrderedTile(*this, view.getRank(), getValue().getType(),
                                   getTileIndices());
}

void TileAtomicAddOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getResourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getResourceMutable());
}

LogicalResult ScalarLoadOp::verify() {
  auto view = getResource().getType();
  if (getIndices().size() != view.getRank() ||
      getResult().getType() != view.getElementType())
    return emitOpError("requires one scalar index per source axis");
  if (static_cast<bool>(getValid()) != static_cast<bool>(getFill()))
    return emitOpError("requires validity and padding together");
  if (getValid() &&
      (!getValid().getType().isInteger(1) ||
       getFill().getType() != view.getElementType()))
    return emitOpError("validity/padding do not match the scalar load");
  return success();
}

void ScalarLoadOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getResourceMutable());
}

LogicalResult ScalarStoreOp::verify() {
  auto view = getResource().getType();
  if (getIndices().size() != view.getRank() ||
      getValue().getType() != view.getElementType())
    return emitOpError("requires one scalar index per destination axis");
  if (getValid() && !getValid().getType().isInteger(1))
    return emitOpError("scalar store validity must be i1");
  return success();
}

void ScalarStoreOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getResourceMutable());
}

LogicalResult GatherLoadOp::verify() {
  if (failed(verifyLoadLatency(*this, getLatencyPolicy())))
    return failure();
  auto view = getResource().getType();
  if (getCoordinates().size() != view.getRank() ||
      view.getElementType() != getResult().getType().getElementType())
    return emitOpError("requires one advanced coordinate per source axis");
  if (failed(verifySourceAxes(*this, getSourceAxes(), view.getRank())))
    return failure();
  if (failed(verifySerializedCoordinateOrder(*this, getSourceAxes())))
    return failure();
  if (failed(verifyCoordinateDomains(*this, getCoordinates(), getResult().getType())))
    return failure();
  if (static_cast<bool>(getValid()) != static_cast<bool>(getFill()))
    return emitOpError("requires validity and padding together");
  if (getValid()) {
    if (!elementType(getValid().getType()).isInteger(1))
      return emitOpError("mask must be a boolean fragment");
    auto valid = dyn_cast<gpu::FragmentType>(getValid().getType());
    if (valid && !samePhysicalDomain(valid, getResult().getType()))
      return emitOpError("mask does not match the gathered physical domain");
  }
  if (getFill() && getFill().getType() != view.getElementType())
    return emitOpError("padding scalar must match the view element type");
  return success();
}

void GatherLoadOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getResourceMutable());
}

LogicalResult ScatterStoreOp::verify() {
  auto view = getResource().getType();
  if (getCoordinates().size() != view.getRank() ||
      view.getElementType() != getValue().getType().getElementType())
    return emitOpError("requires one advanced coordinate per destination axis");
  if (failed(verifySourceAxes(*this, getSourceAxes(), view.getRank())))
    return failure();
  if (failed(verifySerializedCoordinateOrder(*this, getSourceAxes())))
    return failure();
  if (failed(verifyCoordinateDomains(*this, getCoordinates(), getValue().getType())))
    return failure();
  if (getValid()) {
    if (!elementType(getValid().getType()).isInteger(1))
      return emitOpError("mask must be a boolean fragment");
    auto valid = dyn_cast<gpu::FragmentType>(getValid().getType());
    if (valid && !samePhysicalDomain(valid, getValue().getType()))
      return emitOpError("mask does not match the scattered physical domain");
  }
  return success();
}

void ScatterStoreOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getResourceMutable());
}

LogicalResult AtomicRMWOp::verify() {
  auto view = getResource().getType();
  if (getCoordinates().size() != view.getRank() ||
      getValue().getType() != getResult().getType() ||
      elementType(getValue().getType()) != view.getElementType())
    return emitOpError(
        "requires a view-ranked cuTile atomic address and matching value schema");
  if (failed(verifyCoordinateDomains(*this, getCoordinates(), getValue().getType())))
    return failure();
  return success();
}

void AtomicRMWOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getResourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getResourceMutable());
  if (getOrdering() != AtomicOrdering::Relaxed) {
    effects.emplace_back(MemoryEffects::Read::get());
    effects.emplace_back(MemoryEffects::Write::get());
  }
}

LogicalResult ExtractOp::verify() {
  auto source = getSource().getType();
  unsigned rank = source.getShape().size();
  if (getCoordinates().size() != rank || getExtractionShape().size() != rank ||
      elementType(getResult().getType()) != source.getElementType())
    return emitOpError(
        "requires source-ranked tile coordinates/shape and matching element type");
  llvm::SmallBitVector retained(rank);
  int64_t previous = -1;
  for (int64_t axis : getRetainedAxes()) {
    if (axis <= previous || axis >= static_cast<int64_t>(rank))
      return emitOpError("retained axes must be an ordered subset of source axes");
    retained.set(axis);
    previous = axis;
  }
  for (unsigned axis = 0; axis < rank; ++axis) {
    Value coordinate = getCoordinates()[axis];
    if (!coordinate.getType().isInteger(32))
      return emitOpError("requires i32 scalar tile coordinates");
    auto extent = dyn_cast<gpu::PhysicalExprAttr>(getExtractionShape()[axis]);
    if (!extent)
      return emitOpError("extraction shape must contain physical expressions");
    if (!retained.test(axis) && (extent.getKind() !=
                   gpu::PhysicalExprKind::Constant ||
               extent.getValue() != 1)) {
      return emitOpError("selected axes require unit extraction extents");
    }
  }
  auto result = dyn_cast<gpu::FragmentType>(getResult().getType());
  if (!result)
    return getRetainedAxes().empty()
               ? success()
               : emitOpError("scalar extraction cannot retain source axes");
  if (result.getShape().size() != getRetainedAxes().size() ||
      result.getOwner() != source.getOwner())
    return emitOpError("extracted fragment must preserve retained rank and owner");
  for (auto [resultAxis, sourceAxis] : llvm::enumerate(getRetainedAxes()))
    if (result.getShape()[resultAxis] != getExtractionShape()[sourceAxis] ||
        !sameLogicalAxis(source, sourceAxis, result, resultAxis))
      return emitOpError("extracted fragment must preserve retained axis relations");
  return success();
}

LogicalResult MMAOp::verify() {
  auto lhs = getLhs().getType();
  auto rhs = getRhs().getType();
  auto accumulator = getAccumulator().getType();
  auto result = getResult().getType();
  if (auto chunkAttribute = getReductionChunkAttr()) {
    int64_t chunk = chunkAttribute.getInt();
    if (chunk <= 0 || (chunk & (chunk - 1)) != 0 ||
        !lhs.getElementType().isF32() || !rhs.getElementType().isF32() ||
        !accumulator.getElementType().isF32())
      return emitOpError("chunked MMA requires f32 operands and a positive power-of-two reduction chunk");
    if (!lhs.getShape().empty()) {
      auto reduction = cast<gpu::PhysicalExprAttr>(lhs.getShape().getValue().back());
      if (reduction.getKind() ==
              gpu::PhysicalExprKind::Constant &&
          (reduction.getValue() < chunk || reduction.getValue() % chunk != 0))
        return emitOpError("MMA reduction extent must contain complete chunks");
    }
  }
  const unsigned rank = lhs.getShape().size();
  bool valid = rank >= 2 && rank <= 3 && rhs.getShape().size() == rank &&
               accumulator.getShape().size() == rank &&
               result == accumulator && lhs.getOwner() == rhs.getOwner() &&
               lhs.getOwner() == accumulator.getOwner();
  for (unsigned axis = 0; valid && axis + 2 < rank; ++axis)
    valid = lhs.getShape()[axis] == rhs.getShape()[axis] &&
            lhs.getShape()[axis] == result.getShape()[axis] &&
            sameLogicalAxis(lhs, axis, result, axis);
  if (valid) {
    const unsigned matrixAxis = rank - 2;
    valid = lhs.getShape()[matrixAxis] == result.getShape()[matrixAxis] &&
            lhs.getShape()[matrixAxis + 1] ==
                rhs.getShape()[matrixAxis] &&
            rhs.getShape()[matrixAxis + 1] ==
                result.getShape()[matrixAxis + 1] &&
            sameLogicalAxis(lhs, matrixAxis, result, matrixAxis) &&
            sameLogicalAxis(rhs, matrixAxis + 1, result, matrixAxis + 1);
  }
  if (!valid) {
    InFlightDiagnostic diagnostic = emitOpError(
        "requires a canonical [M,K] x [K,N] or [B,M,K] x [B,K,N] "
        "cuTile MMA form");
    diagnostic << "; lhs=" << lhs << "; rhs=" << rhs
               << "; accumulator=" << accumulator << "; result=" << result;
    return failure();
  }
  return success();
}

LogicalResult ScaledMMAOp::verify() {
  auto lhs = getLhs().getType();
  auto lhsScale = getLhsScale().getType();
  auto rhs = getRhs().getType();
  auto rhsScale = getRhsScale().getType();
  auto accumulator = getAccumulator().getType();
  if (lhs.getShape().size() != 3 || lhsScale.getShape().size() != 2 ||
      rhs.getShape().size() != 3 || rhsScale.getShape().size() != 2 ||
      accumulator.getShape().size() != 2 || getResult().getType() != accumulator ||
      lhs.getOwner() != lhsScale.getOwner() ||
      lhs.getOwner() != rhs.getOwner() ||
      lhs.getOwner() != rhsScale.getOwner() ||
      lhs.getOwner() != accumulator.getOwner() ||
      lhs.getShape()[0] != accumulator.getShape()[0] ||
      lhs.getShape()[0] != lhsScale.getShape()[0] ||
      lhs.getShape()[1] != lhsScale.getShape()[1] ||
      lhs.getShape()[1] != rhs.getShape()[0] ||
      lhs.getShape()[1] != rhsScale.getShape()[0] ||
      lhs.getShape()[2] != rhs.getShape()[1] ||
      rhs.getShape()[2] != accumulator.getShape()[1] ||
      rhs.getShape()[2] != rhsScale.getShape()[1] ||
      !sameLogicalAxis(lhs, 0, accumulator, 0) ||
      !sameLogicalAxis(lhs, 0, lhsScale, 0) ||
      !sameLogicalAxis(lhs, 1, lhsScale, 1) ||
      !sameLogicalAxis(lhs, 1, rhs, 0) ||
      !sameLogicalAxis(lhs, 1, rhsScale, 0) ||
      !sameLogicalAxis(lhs, 2, rhs, 1) ||
      !sameLogicalAxis(rhs, 2, accumulator, 1) ||
      !sameLogicalAxis(rhs, 2, rhsScale, 1))
    return emitOpError("requires [M,G,S]/[M,G] x [G,S,N]/[G,N] scaled MMA operands");
  auto inner = dyn_cast<gpu::PhysicalExprAttr>(lhs.getShape()[2]);
  if (!inner ||
      inner.getKind() !=
          gpu::PhysicalExprKind::Constant ||
      inner.getValue() != static_cast<int64_t>(getLhsGroupSize()) ||
      getLhsGroupSize() != getRhsGroupSize() ||
      getLhsFormat() != ScaledFormat::E4M3 ||
      getRhsFormat() != ScaledFormat::E4M3 ||
      !lhsScale.getElementType().isUnsignedInteger(8) ||
      !rhsScale.getElementType().isUnsignedInteger(8) ||
      !accumulator.getElementType().isF32())
    return emitOpError("requires constant group extent, E8M0 scales and f32 accumulator");
  return success();
}

namespace {

template <typename CollectiveOp>
LogicalResult inferCollectiveTypes(MLIRContext *context,
    std::optional<Location> location, ValueRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions, bool scan,
    SmallVectorImpl<Type> &results) {
  typename CollectiveOp::Adaptor operation(operands, attributes, properties, regions);
  if (failed(operation.verify(location.value_or(UnknownLoc::get(context)))))
    return failure();
  if (operation.getAxisAttr().getInt() < 0)
    return emitOptionalError(location, "cuTile collective requires an explicit source axis");
  return gpu::inferScalarCollectiveResultTypes(
      location, operation.getSources(), operation.getAxisAttr().getInt(), scan, results);
}

LogicalResult verifyLiteralIdentities(Operation *owner, ValueRange identities) {
  for (Value identity : identities)
    if (!getCompileTimeScalar(identity)) {
      InFlightDiagnostic diagnostic = owner->emitOpError(
          "custom collective identity must be a compile-time scalar");
      diagnostic << "; identity type=" << identity.getType();
      if (Operation *producer = identity.getDefiningOp())
        diagnostic << ", producer=" << producer->getName();
      return failure();
    }
  return success();
}

} // namespace

LogicalResult ReduceOp::inferReturnTypes(MLIRContext *context,
    std::optional<Location> location, ValueRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions, SmallVectorImpl<Type> &results) {
  return inferCollectiveTypes<ReduceOp>(context, location, operands, attributes,
                                      properties, regions, false, results);
}

LogicalResult ScanOp::inferReturnTypes(MLIRContext *context,
    std::optional<Location> location, ValueRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions, SmallVectorImpl<Type> &results) {
  return inferCollectiveTypes<ScanOp>(context, location, operands, attributes,
                                    properties, regions, true, results);
}

LogicalResult ReduceOp::verify() {
  if (getAxisAttr().getInt() < 0)
    return emitOpError("cuTile native reduction requires an explicit source axis");
  if (getReverse())
    return emitOpError("native reduction does not reverse logical order");
  if (!getKind()) {
    if (getSources().size() > 1) {
      auto kernel = (*this)->getParentOfType<func::FuncOp>();
      auto capabilities = kernel
          ? kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr)
          : gpu::CapabilitiesAttr();
      if (!capabilities || !capabilities.getNativeTupleReductions() ||
          !capabilities.getNativeTupleReductionRequiresConstantIdentity())
        return emitOpError(
            "cuTile tuple reduction requires constant-identity provider capability");
    }
    if (failed(gpu::verifyScalarCollective(
            getOperation(), getSources(), getIdentities(), getResults(),
            getCombine(), getAxis(), false)))
      return failure();
    return verifyLiteralIdentities(getOperation(), getIdentities());
  }
  auto source = getSources().size() == 1
                    ? dyn_cast<gpu::FragmentType>(getSources().front().getType())
                    : gpu::FragmentType();
  if (getSources().size() != 1 || getNumResults() != 1 || !source ||
      !getIdentities().empty() || !getCombine().empty() ||
      static_cast<size_t>(getAxis()) >= source.getShape().size())
    return emitOpError("has an invalid cuTile native reduction axis/kind");
  switch (*getKind()) {
  case BinaryOperator::Add:
  case BinaryOperator::MaximumNum:
  case BinaryOperator::MinimumNum:
    return success();
  case BinaryOperator::LogicalOr:
  case BinaryOperator::LogicalAnd:
    return elementType(getResult(0).getType()).isInteger(1)
               ? success()
               : emitOpError("logical reduction kind requires an i1 result");
  default:
    return emitOpError("has no semantics-preserving cuTile native reduction");
  }
}

LogicalResult ScanOp::verify() {
  if (!getKind()) {
    if (failed(gpu::verifyScalarCollective(getOperation(), getSources(), getIdentities(),
                                         getResults(), getCombine(), getAxis(), true)))
      return failure();
    return verifyLiteralIdentities(getOperation(), getIdentities());
  }
  auto source = getSources().size() == 1
                    ? dyn_cast<gpu::FragmentType>(getSources().front().getType())
                    : gpu::FragmentType();
  if (getSources().size() != 1 || getNumResults() != 1 || !source ||
      !getIdentities().empty() || !getCombine().empty() ||
      static_cast<size_t>(getAxis()) >= source.getShape().size() ||
      *getKind() != BinaryOperator::Add)
    return emitOpError("cuTile native scan currently requires additive cumsum");
  return getResult(0).getType() == source
             ? success()
             : emitOpError("scan must preserve the physical tile schema");
}

} // namespace intent::cutile

#define GET_OP_CLASSES
#include "Intent/Target/CuTile/IR/CuTileOps.cpp.inc"
