#include "Gathers.h"
#include "Intent/Target/Triton/Analysis/Configuration.h"
#include "Intent/Target/Triton/IR/Program.h"
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

namespace {

bool samePhysicalShape(Type lhs, Type rhs) {
  auto left = dyn_cast<gpu::FragmentType>(lhs);
  auto right = dyn_cast<gpu::FragmentType>(rhs);
  if (static_cast<bool>(left) != static_cast<bool>(right))
    return false;
  return !left ||
         (left.getShape() == right.getShape() &&
          left.getAxisMaps() == right.getAxisMaps() &&
          left.getValidity() == right.getValidity() &&
          left.getOwner() == right.getOwner());
}

bool sameGatherFreeAxes(gpu::FragmentType source, gpu::FragmentType result,
                       unsigned selectedAxis) {
  if (!source || !result || source.getOwner() != result.getOwner() ||
      source.getShape().size() != result.getShape().size() ||
      selectedAxis >= source.getShape().size())
    return false;
  for (unsigned axis = 0; axis < source.getShape().size(); ++axis)
    if (axis != selectedAxis &&
        (source.getShape()[axis] != result.getShape()[axis] ||
         source.getAxisMaps()[axis] != result.getAxisMaps()[axis]))
      return false;
  return true;
}

FailureOr<Value> projectGatherIndex(OpBuilder &builder, Location location,
                                    Value value, gpu::FragmentType result) {
  if (auto fragment = dyn_cast<gpu::FragmentType>(value.getType())) {
    if (!result) return failure();
    if (!fragment.getElementType().isIndex()) {
      auto converted = gpu::FragmentType::get(
          builder.getContext(), builder.getIndexType(), fragment.getShape(),
          fragment.getAxisMaps(), fragment.getValidity(), fragment.getOwner());
      value = builder.create<gpu::CastOp>(location, converted, value);
    }
  } else if (!value.getType().isIndex()) {
    value = builder.create<gpu::CastOp>(location, builder.getIndexType(), value);
  }
  if (!result) return value;
  auto indices = gpu::FragmentType::get(
      builder.getContext(), builder.getIndexType(), result.getShape(),
      result.getAxisMaps(), result.getValidity(), result.getOwner());
  return gpu::projectPhysicalValueToSchema(builder, location, value, indices);
}

FailureOr<std::pair<Value, Value>> safeGatherIndex(
    OpBuilder &builder, Location location, Value coordinate,
    gpu::PhysicalExprAttr elements, Value valid = {}) {
  if (!gpu::uniformElementType(coordinate.getType()).isIndex()) {
    Type indexType = builder.getIndexType();
    if (auto fragment = dyn_cast<gpu::FragmentType>(coordinate.getType()))
      indexType = gpu::FragmentType::get(
          fragment.getContext(), builder.getIndexType(), fragment.getShape(),
          fragment.getAxisMaps(), fragment.getValidity(), fragment.getOwner());
    coordinate = builder.create<gpu::CastOp>(location, indexType, coordinate);
  }
  Type predicateType = builder.getI1Type();
  Value extent = builder.create<gpu::PhysicalExprOp>(
      location, builder.getIndexType(), elements);
  if (auto fragment = dyn_cast<gpu::FragmentType>(coordinate.getType())) {
    predicateType = gpu::FragmentType::get(
        fragment.getContext(), builder.getI1Type(), fragment.getShape(),
        fragment.getAxisMaps(), fragment.getValidity(), fragment.getOwner());
    auto projected = gpu::projectPhysicalValueToSchema(
        builder, location, extent, fragment);
    if (failed(projected))
      return failure();
    extent = *projected;
  }
  auto zero = zeroLike(builder, location, coordinate.getType());
  if (failed(zero))
    return failure();
  Value lower = builder.create<gpu::CompareOp>(
      location, predicateType, coordinate, *zero, ComparePredicate::Ge);
  Value upper = builder.create<gpu::CompareOp>(
      location, predicateType, coordinate, extent, ComparePredicate::Lt);
  Value bounded = builder.create<gpu::BinaryOp>(
      location, predicateType, lower, upper, BinaryOperator::LogicalAnd);
  if (valid)
    bounded = builder.create<gpu::BinaryOp>(
        location, predicateType, valid, bounded, BinaryOperator::LogicalAnd);
  Value safe = builder.create<gpu::SelectOp>(
      location, coordinate.getType(), bounded, coordinate, *zero);
  return std::pair<Value, Value>{safe, bounded};
}

std::optional<unsigned>
expandedGatherAxis(gpu::FragmentType source, gpu::FragmentType result,
                   unsigned selectedSourceAxis) {
  if (source.getOwner() != result.getOwner() ||
      source.getShape().size() >= result.getShape().size() ||
      selectedSourceAxis >= source.getShape().size())
    return std::nullopt;

  gpu::BroadcastProjection projection = gpu::queryBroadcastProjection(source, result);
  if (!projection.isExact())
    return std::nullopt;
  std::optional<unsigned> selected;
  for (auto [resultIndex, sourceIndex] :
       llvm::enumerate(projection.targetToSource)) {
    if (sourceIndex) {
      if (*sourceIndex == selectedSourceAxis)
        selected = resultIndex;
      continue;
    }
    auto extent = dyn_cast<gpu::PhysicalExprAttr>(result.getShape()[resultIndex]);
    if (!extent ||
        extent.getKind() !=
            gpu::PhysicalExprKind::Constant ||
        extent.getValue() != 1)
      return std::nullopt;
  }
  return selected;
}

} // namespace

LogicalResult legalizeGatherCoordinates(func::FuncOp kernel) {
  auto [nextSource, nextDimension] = gpu::nextPhysicalAxisIdentities(kernel);
  SmallVector<gpu::GatherOp> gathers;
  kernel.walk([&](gpu::GatherOp gather) { gathers.push_back(gather); });
  for (gpu::GatherOp gather : gathers) {
    if (belongsToSplitGatherPair(gather))
      continue;
    auto scalarConstant = [](Value value) -> arith::ConstantOp {
      while (auto broadcast = value.getDefiningOp<gpu::BroadcastOp>())
        value = broadcast.getValue();
      while (auto splat = value.getDefiningOp<gpu::SplatOp>())
        value = splat.getValue();
      while (auto cast = value.getDefiningOp<gpu::CastOp>())
        value = cast.getValue();
      return value.getDefiningOp<arith::ConstantOp>();
    };
    auto source = dyn_cast<gpu::FragmentType>(gather.getSource().getType());
    auto result = dyn_cast<gpu::FragmentType>(gather.getResult().getType());
    if (!gather.getValid() && gather.getCoordinates().size() == 1 &&
        gather.getSourceAxes().size() == 1 &&
        sameGatherFreeAxes(source, result, gather.getSourceAxes().front()) &&
        samePhysicalShape(gather.getCoordinates().front().getType(), result))
      continue;
    if (source && !source.getShape().empty() &&
        gather.getCoordinates().size() == source.getShape().size() &&
        llvm::all_of(gather.getCoordinates(), [](Value coordinate) {
          Type element = coordinate.getType();
          if (auto fragment = dyn_cast<gpu::FragmentType>(element))
            element = fragment.getElementType();
          return isa<IndexType, IntegerType>(element);
        }) &&
        (result || !gather.getValid() || gather.getValid().getType().isInteger(1))) {
      OpBuilder builder(gather);
      Location location = gather.getLoc();
      Type indexType = builder.getIndexType();
      Type predicateType = builder.getI1Type();
      if (result) {
        indexType = gpu::FragmentType::get(
            kernel.getContext(), builder.getIndexType(), result.getShape(),
            result.getAxisMaps(), result.getValidity(), result.getOwner());
        predicateType = gpu::FragmentType::get(
            kernel.getContext(), builder.getI1Type(), result.getShape(),
            result.getAxisMaps(), result.getValidity(), result.getOwner());
      }
      auto projectIndex = [&](Value value) -> FailureOr<Value> {
        return projectGatherIndex(builder, location, value, result);
      };
      SmallVector<Value> coordinates(source.getShape().size());
      for (auto [axis, value] :
           llvm::zip(gather.getSourceAxes(), gather.getCoordinates())) {
        FailureOr<Value> projected = projectIndex(value);
        if (failed(projected))
          return gather.emitOpError("gather coordinate has no exact result-axis projection");
        coordinates[axis] = *projected;
      }
      Value sourceValue = gather.getSource();
      Value coordinate = coordinates.front();
      Value valid = gather.getValid();
      if (valid && result && valid.getType() != predicateType) {
        auto projected = gpu::projectPhysicalValueToSchema(
            builder, location, valid, cast<gpu::FragmentType>(predicateType));
        if (failed(projected))
          return gather.emitOpError("gather validity has no exact result-axis projection");
        valid = *projected;
      }
      Value safeIndex;
      if (source.getShape().size() > 1) {
        auto elements = cast<gpu::PhysicalExprAttr>(source.getShape()[0]);
        auto first = projectIndex(builder.create<arith::ConstantIndexOp>(location, 0));
        if (failed(first))
          return failure();
        coordinate = *first;
        for (auto [axis, value] : llvm::enumerate(coordinates)) {
          auto extent = cast<gpu::PhysicalExprAttr>(source.getShape()[axis]);
          if (axis)
            elements = gpu::PhysicalExprAttr::get(
                kernel.getContext(),
                gpu::PhysicalExprKind::Multiply, 0,
                builder.getStringAttr(""), builder.getArrayAttr({elements, extent}));
          Value size = builder.create<gpu::PhysicalExprOp>(
              location, builder.getIndexType(), extent);
          auto projectedSize = projectIndex(size);
          if (failed(projectedSize))
            return failure();
          coordinate = builder.create<gpu::BinaryOp>(
              location, indexType, coordinate, *projectedSize, BinaryOperator::Multiply);
          coordinate = builder.create<gpu::BinaryOp>(
              location, indexType, coordinate, value, BinaryOperator::Add);
        }
        if (auto count = evaluateCompileTimeExpression(elements))
          elements = gpu::PhysicalExprAttr::get(
              kernel.getContext(),
              gpu::PhysicalExprKind::Constant, *count,
              builder.getStringAttr(""), builder.getArrayAttr({}));
        auto mapping = gpu::AxisMapAttr::get(
            kernel.getContext(), nextSource++, 0, nextDimension++, 0, true);
        auto flatType = gpu::FragmentType::get(
            kernel.getContext(), source.getElementType(),
            builder.getArrayAttr({elements}), builder.getArrayAttr({mapping}),
            source.getValidity(), source.getOwner());
        auto reassociation = gpu::inferReshapeReassociation(source, flatType);
        if (failed(reassociation))
          return gather.emitOpError("scalar gather has no exact row-major linearization");
        sourceValue = builder.create<gpu::ReshapeOp>(
            location, flatType, sourceValue, *reassociation);
        auto safe = safeGatherIndex(builder, location, coordinate, elements, valid);
        if (failed(safe))
          return failure();
        safeIndex = safe->first;
        valid = safe->second;
      } else {
        auto safe = safeGatherIndex(
            builder, location, coordinate,
            cast<gpu::PhysicalExprAttr>(source.getShape()[0]), valid);
        if (failed(safe))
          return gather.emitOpError("scalar gather coordinate has no safe physical index projection");
        safeIndex = safe->first;
        valid = safe->second;
      }
      Type loadedType = gather.getResult().getType();
      if (result && result.getShape().size() > 1) {
        auto elements = cast<gpu::PhysicalExprAttr>(result.getShape()[0]);
        for (Attribute extent : result.getShape().getValue().drop_front())
          elements = gpu::PhysicalExprAttr::get(
              kernel.getContext(),
              gpu::PhysicalExprKind::Multiply, 0,
              builder.getStringAttr(""), builder.getArrayAttr({elements, extent}));
        auto mapping = gpu::AxisMapAttr::get(
            kernel.getContext(), nextSource++, 0, nextDimension++, 0, true);
        auto flatIndex = gpu::FragmentType::get(
            kernel.getContext(), builder.getIndexType(), builder.getArrayAttr({elements}),
            builder.getArrayAttr({mapping}), result.getValidity(), result.getOwner());
        auto relation = gpu::inferReshapeReassociation(
            cast<gpu::FragmentType>(safeIndex.getType()), flatIndex);
        if (failed(relation))
          return gather.emitOpError("gather result has no exact row-major linearization");
        safeIndex = builder.create<gpu::ReshapeOp>(location, flatIndex, safeIndex, *relation);
        loadedType = gpu::FragmentType::get(
            kernel.getContext(), result.getElementType(), flatIndex.getShape(),
            flatIndex.getAxisMaps(), result.getValidity(), result.getOwner());
      }
      auto loaded = builder.create<gpu::GatherOp>(
          gather.getLoc(), loadedType, sourceValue,
          ValueRange{safeIndex}, Value(), Value(), ArrayRef<int64_t>{0});
      Value loadedValue = loaded.getResult();
      if (loadedType != gather.getResult().getType()) {
        auto relation = gpu::inferReshapeReassociation(
            cast<gpu::FragmentType>(loadedType), result);
        if (failed(relation))
          return gather.emitOpError("linear gather has no exact result reassociation");
        loadedValue = builder.create<gpu::ReshapeOp>(location, result, loadedValue, *relation);
      }
      // An unmasked gather already requires active coordinates to be in bounds.
      // Its row-major linearization has no inactive fill to apply. Masked
      // gathers retain their original validity/fill after the safe native read.
      Value replacement = loadedValue;
      if (gather.getValid())
        replacement = builder.create<gpu::SelectOp>(
            gather.getLoc(), gather.getResult().getType(), valid,
            loadedValue, gather.getFill());
      if (Attribute origin = gather->getAttr(gpu::originAttr))
        replacement.getDefiningOp()->setAttr(gpu::originAttr, origin);
      gather.getResult().replaceAllUsesWith(replacement);
      gather.erase();
      continue;
    }
    if (gather.getCoordinates().size() == 1 &&
        gather.getSourceAxes().size() == 1 && source && result &&
        source.getShape().size() < result.getShape().size()) {
      std::optional<unsigned> selectedAxis = expandedGatherAxis(
          source, result, gather.getSourceAxes().front());
      if (selectedAxis) {
        OpBuilder builder(gather);
        auto expandedSourceType = gpu::FragmentType::get(
            gather.getContext(), source.getElementType(), result.getShape(),
            result.getAxisMaps(), result.getValidity(), result.getOwner());
        Value expandedSource = builder.create<gpu::BroadcastOp>(
            gather.getLoc(), expandedSourceType, gather.getSource());
        Value coordinate = gather.getCoordinates().front();
        auto coordinateType = gpu::FragmentType::get(
            gather.getContext(), coordinate.getType(), result.getShape(),
            result.getAxisMaps(), result.getValidity(), result.getOwner());
        if (auto coordinateFragment =
                dyn_cast<gpu::FragmentType>(coordinate.getType())) {
          coordinateType = gpu::FragmentType::get(
              gather.getContext(), coordinateFragment.getElementType(),
              result.getShape(), result.getAxisMaps(), result.getValidity(),
              result.getOwner());
        }
        coordinate = builder.create<gpu::BroadcastOp>(
            gather.getLoc(), coordinateType, coordinate);
        FailureOr<Value> zero =
            zeroLike(builder, gather.getLoc(), coordinateType);
        if (succeeded(zero)) {
          Value safeIndex = coordinate;
          if (gather.getValid())
            safeIndex = builder.create<gpu::SelectOp>(
                gather.getLoc(), coordinateType, gather.getValid(), coordinate,
                *zero);
          auto safeGather = builder.create<gpu::GatherOp>(
              gather.getLoc(), result, expandedSource, ValueRange{safeIndex},
              Value(), Value(), ArrayRef<int64_t>{static_cast<int64_t>(*selectedAxis)});
          Value selected = safeGather.getResult();
          if (gather.getValid())
            selected = builder.create<gpu::SelectOp>(
                gather.getLoc(), result, gather.getValid(), safeGather.getResult(),
                gather.getFill());
          if (Attribute origin = gather->getAttr(gpu::originAttr))
            selected.getDefiningOp()->setAttr(gpu::originAttr, origin);
          gather.getResult().replaceAllUsesWith(selected);
          gather.erase();
          continue;
        }
      }
    }
    if (gather.getCoordinates().size() == 1 &&
        gather.getSourceAxes().size() == 1 && source && result &&
        source.getShape().size() == result.getShape().size() + 1) {
      unsigned selectedAxis = gather.getSourceAxes().front();
      auto selectedExtent =
          selectedAxis < source.getShape().size()
              ? dyn_cast<gpu::PhysicalExprAttr>(source.getShape()[selectedAxis])
              : gpu::PhysicalExprAttr();
      bool compatible = selectedExtent &&
          ((selectedExtent.getKind() == gpu::PhysicalExprKind::Constant &&
            selectedExtent.getValue() > 0) ||
           selectedExtent.getKind() == gpu::PhysicalExprKind::Parameter);
      unsigned resultAxis = 0;
      for (unsigned sourceAxis = 0;
           compatible && sourceAxis < source.getShape().size(); ++sourceAxis) {
        if (sourceAxis == selectedAxis)
          continue;
        compatible = resultAxis < result.getShape().size() &&
                     source.getShape()[sourceAxis] == result.getShape()[resultAxis];
        ++resultAxis;
      }
      if (compatible) {
        OpBuilder builder(gather);
        SmallVector<Attribute> selectedShape(source.getShape().begin(),
                                             source.getShape().end());
        selectedShape[selectedAxis] = gpu::PhysicalExprAttr::get(
            gather.getContext(),
            gpu::PhysicalExprKind::Constant, 1,
            builder.getStringAttr(""), builder.getArrayAttr({}));
        auto coordinateType = gpu::FragmentType::get(
            gather.getContext(), gpu::uniformElementType(gather.getCoordinates().front().getType()),
            builder.getArrayAttr(selectedShape), source.getAxisMaps(),
            source.getValidity(), source.getOwner());
        auto predicateType = gpu::FragmentType::get(
            gather.getContext(), builder.getI1Type(),
            builder.getArrayAttr(selectedShape), source.getAxisMaps(),
            source.getValidity(), source.getOwner());
        auto selectedResultType = gpu::FragmentType::get(
            gather.getContext(), result.getElementType(),
            builder.getArrayAttr(selectedShape), source.getAxisMaps(),
            result.getValidity(), result.getOwner());
        Value coordinate = builder.create<gpu::BroadcastOp>(
            gather.getLoc(), coordinateType, gather.getCoordinates().front());
        Value valid, fill;
        if (gather.getValid()) {
          valid = builder.create<gpu::BroadcastOp>(
              gather.getLoc(), predicateType, gather.getValid());
          fill = builder.create<gpu::BroadcastOp>(
              gather.getLoc(), selectedResultType, gather.getFill());
        }
        FailureOr<Value> zero = zeroLike(builder, gather.getLoc(), coordinateType);
        if (succeeded(zero)) {
          Value safeIndex = coordinate;
          if (valid)
            safeIndex = builder.create<gpu::SelectOp>(
                gather.getLoc(), coordinateType, valid, coordinate, *zero);
          auto safeGather = builder.create<gpu::GatherOp>(
              gather.getLoc(), selectedResultType, gather.getSource(),
              ValueRange{safeIndex}, Value(), Value(), gather.getSourceAxes());
          Value selected = safeGather.getResult();
          if (valid)
            selected = builder.create<gpu::SelectOp>(
                gather.getLoc(), selectedResultType, valid, safeGather.getResult(),
                fill);
          FailureOr<ArrayAttr> reassociation =
              gpu::inferReshapeReassociation(selectedResultType, result);
          if (failed(reassociation))
            return gather.emitOpError(
                "gather fallback has no exact row-major reassociation");
          auto reshaped = builder.create<gpu::ReshapeOp>(
              gather.getLoc(), result, selected, *reassociation);
          if (Attribute origin = gather->getAttr(gpu::originAttr))
            reshaped->setAttr(gpu::originAttr, origin);
          gather.getResult().replaceAllUsesWith(reshaped.getResult());
          gather.erase();
          continue;
        }
      }
    }
    if (!gather.getValid()) {
      if (gather.getCoordinates().size() == 1 &&
          gather.getSourceAxes().size() == 1 &&
          sameGatherFreeAxes(source, result, gather.getSourceAxes().front())) {
        OpBuilder builder(gather);
        auto coordinate = projectGatherIndex(
            builder, gather.getLoc(), gather.getCoordinates().front(), result);
        if (failed(coordinate))
          return gather.emitOpError("gather index has no exact native result-axis projection");
        gather.getCoordinatesMutable().assign(ValueRange{*coordinate});
      } else
        return gather.emitOpError("gather free axes have no exact native coordinate transport");
      continue;
    }
    if (gather.getCoordinates().size() == 1 &&
        isa<gpu::FragmentType>(gather.getCoordinates().front().getType()) &&
        gather.getSourceAxes().size() == 1 && source) {
      unsigned axis = gather.getSourceAxes().front();
      auto coordinate = scalarConstant(gather.getCoordinates().front());
      auto predicate = scalarConstant(gather.getValid());
      auto extent = axis < source.getShape().size()
                        ? dyn_cast<gpu::PhysicalExprAttr>(source.getShape()[axis])
                        : gpu::PhysicalExprAttr();
      auto coordinateValue =
          coordinate ? dyn_cast<IntegerAttr>(coordinate.getValue()) : IntegerAttr();
      auto predicateValue = predicate ? dyn_cast<IntegerAttr>(predicate.getValue())
                                      : IntegerAttr();
      if (extent &&
          extent.getKind() == gpu::PhysicalExprKind::Constant &&
          coordinateValue && coordinateValue.getInt() >= 0 &&
          coordinateValue.getInt() < extent.getValue() && predicateValue &&
          predicateValue.getValue().isOne()) {
        OpBuilder builder(gather);
        auto replacement = builder.create<gpu::GatherOp>(
            gather.getLoc(), gather.getResult().getType(), gather.getSource(),
            gather.getCoordinates(), Value(), Value(), gather.getSourceAxes());
        if (Attribute origin = gather->getAttr(gpu::originAttr))
          replacement->setAttr(gpu::originAttr, origin);
        gather.getResult().replaceAllUsesWith(replacement.getResult());
        gather.erase();
        continue;
      }
    }
    if (gather.getCoordinates().size() != 1)
      continue;
    OpBuilder builder(gather);
    Value coordinate = gather.getCoordinates().front();
    if (!samePhysicalShape(gather.getValid().getType(), coordinate.getType())) {
      auto predicateType = dyn_cast<gpu::FragmentType>(gather.getValid().getType());
      Type indexElement = gpu::uniformElementType(coordinate.getType());
      if (!predicateType || !isa<IntegerType, IndexType>(indexElement))
        continue;
      if (isa<gpu::FragmentType>(coordinate.getType()) &&
          (!source || !result || gather.getSourceAxes().size() != 1 ||
           source.getShape().size() != result.getShape().size()))
        continue;
      auto coordinateType = gpu::FragmentType::get(
          gather.getContext(), indexElement, predicateType.getShape(),
          predicateType.getAxisMaps(), predicateType.getValidity(),
          predicateType.getOwner());
      auto projected = gpu::projectPhysicalValueToSchema(
          builder, gather.getLoc(), coordinate, coordinateType);
      if (failed(projected))
        continue;
      coordinate = *projected;
    }
    if (!samePhysicalShape(gather.getValid().getType(), coordinate.getType()) ||
        !samePhysicalShape(gather.getValid().getType(),
                           gather.getResult().getType()))
      continue;
    if (!source || gather.getSourceAxes().size() != 1)
      continue;
    unsigned axis = gather.getSourceAxes().front();
    auto safe = safeGatherIndex(
        builder, gather.getLoc(), coordinate,
        cast<gpu::PhysicalExprAttr>(source.getShape()[axis]), gather.getValid());
    if (failed(safe))
      return gather.emitOpError("gather coordinate has no safe physical index projection");
    auto safeGather = builder.create<gpu::GatherOp>(
        gather.getLoc(), gather.getResult().getType(), gather.getSource(),
        ValueRange{safe->first}, Value(), Value(), gather.getSourceAxes());
    auto selected = builder.create<gpu::SelectOp>(
        gather.getLoc(), gather.getResult().getType(), gather.getValid(),
        safeGather.getResult(), gather.getFill());
    if (Attribute origin = gather->getAttr(gpu::originAttr))
      selected->setAttr(gpu::originAttr, origin);
    gather.getResult().replaceAllUsesWith(selected.getResult());
    gather.erase();
  }
  return success();
}

LogicalResult legalizeExpandingGathers(func::FuncOp kernel) {
  auto [nextSource, nextDimension] = gpu::nextPhysicalAxisIdentities(kernel);
  SmallVector<gpu::GatherOp> gathers;
  kernel.walk([&](gpu::GatherOp gather) { gathers.push_back(gather); });
  for (gpu::GatherOp gather : gathers) {
    if (gather.getValid() || gather.getCoordinates().size() != 1 ||
        gather.getSourceAxes().size() != 1)
      continue;
    auto source = dyn_cast<gpu::FragmentType>(gather.getSource().getType());
    auto result = dyn_cast<gpu::FragmentType>(gather.getResult().getType());
    auto indices = dyn_cast<gpu::FragmentType>(
        gather.getCoordinates().front().getType());
    if (!source || !result || !indices || source.getShape().size() < 2 ||
        source.getShape().size() != result.getShape().size() ||
        indices.getShape() != result.getShape() ||
        indices.getAxisMaps() != result.getAxisMaps())
      continue;
    unsigned axis = gather.getSourceAxes().front();
    bool compatible = true;
    for (unsigned position = 0; position < source.getShape().size(); ++position)
      compatible &= position == axis ||
                    (source.getShape()[position] == result.getShape()[position] &&
                     source.getAxisMaps()[position] == result.getAxisMaps()[position]);
    auto sourceExtent = cast<gpu::PhysicalExprAttr>(source.getShape()[axis]);
    auto resultExtent = cast<gpu::PhysicalExprAttr>(result.getShape()[axis]);
    if (!compatible || sourceExtent == resultExtent ||
        !isTritonFragmentExtent(sourceExtent) ||
        !isTritonFragmentExtent(resultExtent))
      continue;
    auto constantExtent = [](gpu::PhysicalExprAttr extent) {
      return extent.getKind() ==
             gpu::PhysicalExprKind::Constant;
    };
    bool constantShape = constantExtent(sourceExtent) &&
                         constantExtent(resultExtent);
    if (constantShape && resultExtent.getValue() <= sourceExtent.getValue())
      continue;

    OpBuilder builder(gather);
    Location location = gather.getLoc();
    auto product = [&](ArrayRef<Attribute> shape) {
      auto extent = gpu::PhysicalExprAttr::get(
          kernel.getContext(),
          gpu::PhysicalExprKind::Constant, 1,
          builder.getStringAttr(""), builder.getArrayAttr({}));
      if (shape.empty())
        return extent;
      extent = cast<gpu::PhysicalExprAttr>(shape.front());
      for (Attribute dimension : shape.drop_front())
        extent = gpu::PhysicalExprAttr::get(
            kernel.getContext(),
            gpu::PhysicalExprKind::Multiply, 0,
            builder.getStringAttr(""), builder.getArrayAttr({extent, dimension}));
      return extent;
    };
    auto flatSourceAxis = gpu::AxisMapAttr::get(
        kernel.getContext(), nextSource++, 0, nextDimension++, 0, true);
    auto flatResultAxis = gpu::AxisMapAttr::get(
        kernel.getContext(), nextSource++, 0, nextDimension++, 0, true);
    auto sourceElements = product(source.getShape().getValue());
    auto resultElements = product(result.getShape().getValue());
    auto flatType = [&](gpu::FragmentType original, Type element,
                        gpu::PhysicalExprAttr extent, gpu::AxisMapAttr mapping) {
      return gpu::FragmentType::get(
          kernel.getContext(), element, builder.getArrayAttr({extent}),
          builder.getArrayAttr({mapping}), original.getValidity(),
          original.getOwner());
    };
    auto flatSourceType = flatType(source, source.getElementType(),
                                   sourceElements, flatSourceAxis);
    auto flatResultType = flatType(result, result.getElementType(),
                                   resultElements, flatResultAxis);
    auto flatIndexType = flatType(indices, builder.getIndexType(),
                                  resultElements, flatResultAxis);
    auto linearize = [&](OpBuilder &nested) -> FailureOr<Value> {
      auto reshape = [&](Value value,
                         gpu::FragmentType target) -> FailureOr<Value> {
        auto relation = gpu::inferReshapeReassociation(
            cast<gpu::FragmentType>(value.getType()), target);
        if (failed(relation))
          return failure();
        return Value(nested.create<gpu::ReshapeOp>(location, target, value,
                                                  *relation));
      };
      auto flatOriginalIndexType = flatType(
          indices, indices.getElementType(), resultElements, flatResultAxis);
      FailureOr<Value> flatSource = reshape(gather.getSource(), flatSourceType);
      FailureOr<Value> flatIndices =
          reshape(gather.getCoordinates().front(), flatOriginalIndexType);
      if (failed(flatSource) || failed(flatIndices))
        return failure();
      Value index = *flatIndices;
      if (index.getType() != flatIndexType)
        index = nested.create<gpu::CastOp>(location, flatIndexType, index);
      auto extentValue = [&](gpu::PhysicalExprAttr extent) -> Value {
        return nested.create<gpu::PhysicalExprOp>(
            location, nested.getIndexType(), extent);
      };
      auto broadcastExtent = [&](gpu::PhysicalExprAttr extent) -> Value {
        return nested.create<gpu::BroadcastOp>(location, flatIndexType,
                                               extentValue(extent));
      };
      auto binary = [&](Value lhs, Value rhs, BinaryOperator kind) -> Value {
        return nested.create<gpu::BinaryOp>(location, flatIndexType, lhs, rhs,
                                            kind);
      };
      Value zero = nested.create<arith::ConstantIndexOp>(location, 0);
      Value one = nested.create<arith::ConstantIndexOp>(location, 1);
      Value count = extentValue(resultElements);
      Value ordinal = nested.create<gpu::MakeRangeOp>(
          location, flatIndexType, zero, count, one, zero, count,
          flatResultAxis.getSourceId(), flatResultAxis.getSourceAxis(), true);
      Value inner = broadcastExtent(
          product(result.getShape().getValue().drop_front(axis + 1)));
      Value sourceStride = binary(broadcastExtent(sourceExtent), inner,
                                   BinaryOperator::Multiply);
      Value resultStride = binary(broadcastExtent(resultExtent), inner,
                                   BinaryOperator::Multiply);
      Value outer = binary(ordinal, resultStride, BinaryOperator::FloorDivide);
      Value tail = binary(ordinal, inner, BinaryOperator::Remainder);
      Value offset = binary(outer, sourceStride, BinaryOperator::Multiply);
      offset = binary(offset, binary(index, inner, BinaryOperator::Multiply),
                      BinaryOperator::Add);
      offset = binary(offset, tail, BinaryOperator::Add);
      // The incoming gather already has safe per-axis indices. Preserve that
      // executable safety boundary after row-major linearization as well: an
      // inactive lane may have selected axis index zero while retaining its
      // other coordinates. The original validity/fill selection remains at
      // the gather's consumer.
      auto safe = safeGatherIndex(nested, location, offset, sourceElements);
      if (failed(safe))
        return failure();
      Value selected = nested.create<gpu::GatherOp>(
          location, flatResultType, *flatSource, ValueRange{safe->first}, Value(),
          Value(), ArrayRef<int64_t>{0});
      return reshape(selected, result);
    };

    // Expanding gathers can require cross-warp exchange.  A linear gather
    // retains that meaning without the rank-dependent warp-local rewrite.
    Value replacement;
    if (constantShape) {
      FailureOr<Value> linear = linearize(builder);
      if (failed(linear))
        return gather.emitOpError("expanding gather has no exact linear reshape");
      replacement = *linear;
    } else {
      Value sourceSize = builder.create<gpu::PhysicalExprOp>(
          location, builder.getIndexType(), sourceExtent);
      Value resultSize = builder.create<gpu::PhysicalExprOp>(
          location, builder.getIndexType(), resultExtent);
      Value expanding = builder.create<gpu::CompareOp>(
          location, builder.getI1Type(), resultSize, sourceSize,
          ComparePredicate::Gt);
      auto conditional = builder.create<scf::IfOp>(
          location, TypeRange{result}, expanding, /*withElseRegion=*/true);
      OpBuilder linearBuilder = conditional.getThenBodyBuilder();
      FailureOr<Value> linear = linearize(linearBuilder);
      if (failed(linear))
        return gather.emitOpError("expanding gather has no exact linear reshape");
      linearBuilder.create<scf::YieldOp>(location, *linear);
      OpBuilder originalBuilder = conditional.getElseBodyBuilder();
      Operation *original = originalBuilder.clone(*gather.getOperation());
      originalBuilder.create<scf::YieldOp>(location, original->getResults());
      replacement = conditional.getResult(0);
    }
    if (Attribute origin = gather->getAttr(gpu::originAttr))
      replacement.getDefiningOp()->setAttr(gpu::originAttr, origin);
    gather.getResult().replaceAllUsesWith(replacement);
    gather.erase();
  }
  return success();
}

} // namespace intent::triton::detail
