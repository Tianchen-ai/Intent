#include "Gathers.h"
#include "Intent/Target/Triton/IR/TritonOps.h"
#include "llvm/ADT/DenseSet.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "Intent/Dialect/GPU/Transforms/Storage/Storage.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/MapVector.h"
#include <algorithm>
#include <limits>
#include <optional>


using namespace mlir;

namespace intent::triton::detail {

Value stripShapeOnly(Value value) {
  while (true) {
    if (auto broadcast = value.getDefiningOp<gpu::BroadcastOp>()) {
      value = broadcast.getValue();
      continue;
    }
    if (auto reshape = value.getDefiningOp<gpu::ReshapeOp>()) {
      value = reshape.getValue();
      continue;
    }
    if (auto splat = value.getDefiningOp<gpu::SplatOp>()) {
      value = splat.getValue();
      continue;
    }
    return value;
  }
}

FailureOr<Value> zeroLike(OpBuilder &builder, Location location, Type type) {
  Type scalarType = gpu::uniformElementType(type);
  Value zero;
  if (scalarType.isIndex())
    zero = builder.create<arith::ConstantIndexOp>(location, 0);
  else if (auto integer = dyn_cast<IntegerType>(scalarType))
    zero = builder.create<arith::ConstantOp>(
        location, integer, builder.getIntegerAttr(integer, 0));
  else
    return failure();
  if (auto fragment = dyn_cast<gpu::FragmentType>(type))
    return Value(builder.create<gpu::SplatOp>(location, fragment, zero));
  return zero;
}

namespace {

bool isTrueValue(Value value) {
  auto constant = stripShapeOnly(value).getDefiningOp<arith::ConstantOp>();
  auto integer = constant ? dyn_cast<IntegerAttr>(constant.getValue())
                          : IntegerAttr();
  return integer && integer.getType().isInteger(1) && integer.getValue().isOne();
}

std::optional<int64_t> staticGatherCoordinate(gpu::GatherOp gather) {
  if (gather.getCoordinates().size() != 1 ||
      gather.getSourceAxes().size() != 1 || !gather.getValid() ||
      !isTrueValue(gather.getValid()))
    return std::nullopt;
  Value coordinate = stripShapeOnly(gather.getCoordinates().front());
  while (auto cast = coordinate.getDefiningOp<gpu::CastOp>())
    coordinate = cast.getValue();
  auto constant = coordinate.getDefiningOp<arith::ConstantOp>();
  auto integer = constant ? dyn_cast<IntegerAttr>(constant.getValue())
                          : IntegerAttr();
  if (!integer || (integer.getInt() != 0 && integer.getInt() != 1))
    return std::nullopt;

  auto source = dyn_cast<gpu::FragmentType>(gather.getSource().getType());
  auto result = dyn_cast<gpu::FragmentType>(gather.getResult().getType());
  if (!source || !result || source.getShape().size() != result.getShape().size() + 1 ||
      gather.getSourceAxes().front() !=
          static_cast<int64_t>(source.getShape().size() - 1) ||
      source.getElementType() != result.getElementType() ||
      source.getValidity() != result.getValidity() ||
      source.getOwner() != result.getOwner())
    return std::nullopt;
  auto trailing = dyn_cast<gpu::PhysicalExprAttr>(
      source.getShape()[source.getShape().size() - 1]);
  if (!trailing ||
      trailing.getKind() !=
          gpu::PhysicalExprKind::Constant ||
      trailing.getValue() != 2 ||
      !std::equal(result.getShape().begin(), result.getShape().end(),
                  source.getShape().begin()) ||
      !std::equal(result.getAxisMaps().begin(), result.getAxisMaps().end(),
                  source.getAxisMaps().begin()))
    return std::nullopt;
  return integer.getInt();
}

} // namespace

bool belongsToSplitGatherPair(gpu::GatherOp gather) {
  std::optional<int64_t> coordinate = staticGatherCoordinate(gather);
  if (!coordinate || !gather->getBlock())
    return false;
  for (Operation *user : gather.getSource().getUsers()) {
    auto complement = dyn_cast<gpu::GatherOp>(user);
    std::optional<int64_t> other = complement
                                       ? staticGatherCoordinate(complement)
                                       : std::optional<int64_t>();
    if (complement && complement != gather &&
        complement->getBlock() == gather->getBlock() && other &&
        *other == 1 - *coordinate &&
        complement.getResult().getType() == gather.getResult().getType())
      return true;
  }
  return false;
}

LogicalResult legalizeSplitGatherPairs(func::FuncOp kernel) {
  SmallVector<gpu::GatherOp> gathers;
  kernel.walk([&](gpu::GatherOp gather) { gathers.push_back(gather); });
  llvm::SmallPtrSet<Operation *, 8> rewritten;
  bool changed = false;
  for (gpu::GatherOp low : gathers) {
    if (rewritten.contains(low.getOperation()) || !low->getBlock() ||
        staticGatherCoordinate(low) != std::optional<int64_t>(0))
      continue;
    for (gpu::GatherOp high : gathers) {
      if (rewritten.contains(high.getOperation()) || !high->getBlock() ||
          high->getBlock() != low->getBlock() ||
          high.getSource() != low.getSource() ||
          high.getResult().getType() != low.getResult().getType() ||
          staticGatherCoordinate(high) != std::optional<int64_t>(1))
        continue;
      Operation *anchor = low->isBeforeInBlock(high) ? low.getOperation()
                                                    : high.getOperation();
      OpBuilder builder(anchor);
      auto split = builder.create<SplitOp>(
          anchor->getLoc(), low.getResult().getType(), high.getResult().getType(),
          low.getSource());
      low.getResult().replaceAllUsesWith(split.getLow());
      high.getResult().replaceAllUsesWith(split.getHigh());
      rewritten.insert(low.getOperation());
      rewritten.insert(high.getOperation());
      changed = true;
      break;
    }
  }
  if (changed) {
    for (gpu::GatherOp gather : gathers)
      if (rewritten.contains(gather.getOperation()))
        gather.erase();
    gpu::eraseDeadPhysicalValues(kernel);
  }
  return success();
}

LogicalResult legalizeLargeScalarGathers(func::FuncOp kernel) {
  auto capabilities =
      kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
  SmallVector<gpu::GatherOp> gathers;
  kernel.walk([&](gpu::GatherOp gather) { gathers.push_back(gather); });
  for (gpu::GatherOp gather : gathers) {
    auto source = dyn_cast<gpu::FragmentType>(gather.getSource().getType());
    auto result = dyn_cast<gpu::FragmentType>(gather.getResult().getType());
    if (!source || !isa<IntegerType, FloatType>(source.getElementType()) ||
        gather.getCoordinates().empty() ||
        gather.getCoordinates().size() != gather.getSourceAxes().size())
      continue;
    uint64_t bytes = (source.getElementType().getIntOrFloatBitWidth() + 7) / 8;
    uint64_t limit = capabilities.getMaxDynamicSharedMemoryPerBlock();
    bool oversized = bytes > limit;
    bool fixed = true;
    for (Attribute attribute : source.getShape()) {
      auto extent = cast<gpu::PhysicalExprAttr>(attribute);
      if (extent.getKind() != gpu::PhysicalExprKind::Constant ||
          extent.getValue() <= 0) {
        fixed = false;
        break;
      }
      if (!oversized) {
        oversized = bytes > limit / extent.getValue();
        if (!oversized)
          bytes *= extent.getValue();
      }
    }
    if (!fixed || !oversized || (result && result.getShape().empty()))
      continue;
    SmallVector<std::pair<unsigned, Value>> selected;
    SmallVector<std::pair<unsigned, Value>> indexed;
    llvm::SmallDenseSet<unsigned> axes;
    llvm::SmallDenseSet<unsigned> seenAxes;
    bool compatible = true;
    for (auto [coordinate, axis] :
         llvm::zip(gather.getCoordinates(), gather.getSourceAxes())) {
      Value originalCoordinate = coordinate;
      coordinate = stripShapeOnly(coordinate);
      if (axis < 0 || static_cast<unsigned>(axis) >= source.getShape().size() ||
          !seenAxes.insert(axis).second) {
        compatible = false;
        break;
      }
      if (auto fragment = dyn_cast<gpu::FragmentType>(coordinate.getType())) {
        if (!isa<IntegerType, IndexType>(fragment.getElementType())) {
          compatible = false;
          break;
        }
        indexed.emplace_back(axis, originalCoordinate);
        continue;
      }
      if (!isa<IntegerType, IndexType>(coordinate.getType())) {
        compatible = false;
        break;
      }
      axes.insert(axis);
      selected.emplace_back(axis, coordinate);
    }
    if (!compatible || selected.empty())
      continue;
    unsigned resultAxis = 0;
    for (unsigned axis = 0; indexed.empty() && compatible && axis < source.getShape().size(); ++axis)
      if (!axes.contains(axis)) {
        compatible = result && resultAxis < result.getShape().size() &&
                     source.getShape()[axis] == result.getShape()[resultAxis];
        ++resultAxis;
      }
    if (!compatible || (indexed.empty() &&
                        (result ? resultAxis != result.getShape().size() : resultAxis != 0)))
      continue;

    // Exactly one source element contributes to each result lane. Combining
    // its integer representation with zero preserves NaNs and signed zero,
    // while native reductions avoid the whole-source scratch of a gather.
    OpBuilder builder(gather);
    Location location = gather.getLoc();
    auto bits = builder.getIntegerType(source.getElementType().getIntOrFloatBitWidth());
    auto bitsType = gpu::FragmentType::get(
        kernel.getContext(), bits, source.getShape(), source.getAxisMaps(),
        source.getValidity(), source.getOwner());
    Value value = builder.create<gpu::BitcastOp>(location, bitsType, gather.getSource());
    llvm::sort(selected, [](const auto &lhs, const auto &rhs) { return lhs.first > rhs.first; });
    for (auto [axis, coordinate] : selected) {
      // Select one axis at a time so later selections consume the reduced
      // fragment rather than constructing another full-source predicate.
      auto input = cast<gpu::FragmentType>(value.getType());
      auto mapping = cast<gpu::AxisMapAttr>(input.getAxisMaps()[axis]);
      auto ordinalAxis = gpu::AxisMapAttr::get(
          kernel.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
          mapping.getDimensionId(), 0, mapping.getDerived());
      auto rangeType = gpu::FragmentType::get(
          kernel.getContext(), builder.getIndexType(),
          builder.getArrayAttr({input.getShape()[axis]}),
          builder.getArrayAttr({ordinalAxis}), input.getValidity(), input.getOwner());
      Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
      Value one = builder.create<arith::ConstantIndexOp>(location, 1);
      Value size = builder.create<gpu::PhysicalExprOp>(
          location, builder.getIndexType(), cast<gpu::PhysicalExprAttr>(input.getShape()[axis]));
      Value ordinal = builder.create<gpu::MakeRangeOp>(
          location, rangeType, zero, size, one, zero, size,
          mapping.getSourceId(), mapping.getSourceAxis(), mapping.getDerived());
      if (!coordinate.getType().isIndex())
        coordinate = builder.create<gpu::CastOp>(location, builder.getIndexType(), coordinate);
      Value index = builder.create<gpu::BroadcastOp>(location, rangeType, coordinate);
      auto predicate = gpu::FragmentType::get(
          kernel.getContext(), builder.getI1Type(), rangeType.getShape(),
          rangeType.getAxisMaps(), rangeType.getValidity(), rangeType.getOwner());
      Value equal = builder.create<gpu::CompareOp>(
          location, predicate, ordinal, index, ComparePredicate::Eq);
      auto expanded = gpu::projectPredicateToFragmentAxis(
          builder, location, equal, input, axis);
      if (failed(expanded))
        return gather.emitOpError("scalar selection lost its source-axis projection");
      auto emptyBits = zeroLike(builder, location, input);
      if (failed(emptyBits))
        return failure();
      value = builder.create<gpu::SelectOp>(location, input, *expanded, value, *emptyBits);
      auto inferred = gpu::inferCollectiveResultType(input,
          ArrayRef<int64_t>{static_cast<int64_t>(axis)}, bits);
      if (failed(inferred))
        return gather.emitOpError("scalar selection has an invalid reduction schema");
      Type reducedType = *inferred;
      auto identity = zeroLike(builder, location, reducedType);
      if (failed(identity))
        return failure();
      auto reduction = builder.create<gpu::ReduceOp>(location, TypeRange{reducedType},
          ValueRange{value}, ValueRange{*identity}, ValueRange{},
          ArrayRef<int64_t>{static_cast<int64_t>(axis)});
      {
        OpBuilder::InsertionGuard guard(builder);
        Block *body = builder.createBlock(&reduction.getCombine(), {},
            {reducedType, reducedType}, {location, location});
        Value combined = builder.create<gpu::BinaryOp>(
            location, reducedType, body->getArgument(0), body->getArgument(1),
            BinaryOperator::BitwiseOr);
        builder.create<gpu::YieldOp>(location, combined);
      }
      value = reduction.getResult(0);
    }
    Type decodedType = source.getElementType();
    if (auto fragment = dyn_cast<gpu::FragmentType>(value.getType()))
      decodedType = gpu::FragmentType::get(
          kernel.getContext(), source.getElementType(), fragment.getShape(),
          fragment.getAxisMaps(), fragment.getValidity(), fragment.getOwner());
    value = builder.create<gpu::BitcastOp>(location, decodedType, value);
    if (!indexed.empty()) {
      SmallVector<Value> coordinates;
      SmallVector<int64_t> sourceAxes;
      for (auto [axis, coordinate] : indexed) {
        unsigned removed = llvm::count_if(selected, [&](const auto &item) {
          return item.first < axis;
        });
        coordinates.push_back(coordinate);
        sourceAxes.push_back(axis - removed);
      }
      value = builder.create<gpu::GatherOp>(location, gather.getResult().getType(),
          value, coordinates, gather.getValid(), gather.getFill(), sourceAxes);
      if (Attribute origin = gather->getAttr(gpu::originAttr))
        value.getDefiningOp()->setAttr(gpu::originAttr, origin);
      gather.getResult().replaceAllUsesWith(value);
      gather.erase();
      continue;
    }
    if (value.getType() != gather.getResult().getType()) {
      auto reassociation = gpu::inferReshapeReassociation(
          cast<gpu::FragmentType>(value.getType()), result);
      if (failed(reassociation))
        return gather.emitOpError("scalar selection lost its result relation");
      value = builder.create<gpu::ReshapeOp>(location, result, value, *reassociation);
    }
    if (gather.getValid())
      value = builder.create<gpu::SelectOp>(
          location, gather.getResult().getType(), gather.getValid(), value, gather.getFill());
    gather.getResult().replaceAllUsesWith(value);
    gather.erase();
  }
  gpu::eraseDeadPhysicalValues(kernel);
  return success();
}

} // namespace intent::triton::detail
