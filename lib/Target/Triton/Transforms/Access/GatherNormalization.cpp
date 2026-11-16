#include "Gathers.h"
#include "Intent/Target/Triton/Analysis/Configuration.h"
#include "Intent/Target/Triton/IR/Program.h"
#include "llvm/ADT/DenseSet.h"
#include "Intent/Dialect/Block/IR/Program.h"
#include "Intent/Dialect/Block/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/Block/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/Block/Analysis/UniformValues.h"
#include "Intent/Dialect/Block/Analysis/ValueSchema.h"
#include "Intent/Dialect/Block/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/Block/Transforms/Value/ValueRelations.h"
#include "Intent/Dialect/Block/Transforms/Storage/Storage.h"
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
  auto left = dyn_cast<block::FragmentType>(lhs);
  auto right = dyn_cast<block::FragmentType>(rhs);
  if (static_cast<bool>(left) != static_cast<bool>(right))
    return false;
  return !left ||
         (left.getShape() == right.getShape() &&
          left.getAxisMaps() == right.getAxisMaps() &&
          left.getValidity() == right.getValidity() &&
          left.getOwner() == right.getOwner());
}

bool sameGatherFreeAxes(block::FragmentType source, block::FragmentType result,
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
                                    Value value, block::FragmentType result) {
  if (auto fragment = dyn_cast<block::FragmentType>(value.getType())) {
    if (!result) return failure();
    if (!fragment.getElementType().isIndex()) {
      auto converted = block::FragmentType::get(
          builder.getContext(), builder.getIndexType(), fragment.getShape(),
          fragment.getAxisMaps(), fragment.getValidity(), fragment.getOwner());
      value = builder.create<block::CastOp>(location, converted, value);
    }
  } else if (!value.getType().isIndex()) {
    value = builder.create<block::CastOp>(location, builder.getIndexType(), value);
  }
  if (!result) return value;
  auto indices = block::FragmentType::get(
      builder.getContext(), builder.getIndexType(), result.getShape(),
      result.getAxisMaps(), result.getValidity(), result.getOwner());
  return block::projectPhysicalValueToSchema(builder, location, value, indices);
}

FailureOr<std::pair<Value, Value>> safeGatherIndex(
    OpBuilder &builder, Location location, Value coordinate,
    block::PhysicalExprAttr elements, Value valid = {}) {
  if (!block::uniformElementType(coordinate.getType()).isIndex()) {
    Type indexType = builder.getIndexType();
    if (auto fragment = dyn_cast<block::FragmentType>(coordinate.getType()))
      indexType = block::FragmentType::get(
          fragment.getContext(), builder.getIndexType(), fragment.getShape(),
          fragment.getAxisMaps(), fragment.getValidity(), fragment.getOwner());
    coordinate = builder.create<block::CastOp>(location, indexType, coordinate);
  }
  Type predicateType = builder.getI1Type();
  Value extent = builder.create<block::PhysicalExprOp>(
      location, builder.getIndexType(), elements);
  if (auto fragment = dyn_cast<block::FragmentType>(coordinate.getType())) {
    predicateType = block::FragmentType::get(
        fragment.getContext(), builder.getI1Type(), fragment.getShape(),
        fragment.getAxisMaps(), fragment.getValidity(), fragment.getOwner());
    auto projected = block::projectPhysicalValueToSchema(
        builder, location, extent, fragment);
    if (failed(projected))
      return failure();
    extent = *projected;
  }
  auto zero = zeroLike(builder, location, coordinate.getType());
  if (failed(zero))
    return failure();
  Value lower = builder.create<block::CompareOp>(
      location, predicateType, coordinate, *zero, ComparePredicate::Ge);
  Value upper = builder.create<block::CompareOp>(
      location, predicateType, coordinate, extent, ComparePredicate::Lt);
  Value bounded = builder.create<block::BinaryOp>(
      location, predicateType, lower, upper, BinaryOperator::LogicalAnd);
  if (valid)
    bounded = builder.create<block::BinaryOp>(
        location, predicateType, valid, bounded, BinaryOperator::LogicalAnd);
  Value safe = builder.create<block::SelectOp>(
      location, coordinate.getType(), bounded, coordinate, *zero);
  return std::pair<Value, Value>{safe, bounded};
}

std::optional<unsigned>
expandedGatherAxis(block::FragmentType source, block::FragmentType result,
                   unsigned selectedSourceAxis) {
  if (source.getOwner() != result.getOwner() ||
      source.getShape().size() >= result.getShape().size() ||
      selectedSourceAxis >= source.getShape().size())
    return std::nullopt;

  block::BroadcastProjection projection = block::queryBroadcastProjection(source, result);
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
    auto extent = dyn_cast<block::PhysicalExprAttr>(result.getShape()[resultIndex]);
    if (!extent ||
        extent.getKind() !=
            block::PhysicalExprKind::Constant ||
        extent.getValue() != 1)
      return std::nullopt;
  }
  return selected;
}

} // namespace

LogicalResult legalizeGatherCoordinates(func::FuncOp kernel) {
  auto [nextSource, nextDimension] = block::nextPhysicalAxisIdentities(kernel);
  SmallVector<block::GatherOp> gathers;
  kernel.walk([&](block::GatherOp gather) { gathers.push_back(gather); });
  for (block::GatherOp gather : gathers) {
    if (belongsToSplitGatherPair(gather))
      continue;
    auto scalarConstant = [](Value value) -> arith::ConstantOp {
      while (auto broadcast = value.getDefiningOp<block::BroadcastOp>())
        value = broadcast.getValue();
      while (auto splat = value.getDefiningOp<block::SplatOp>())
        value = splat.getValue();
      while (auto cast = value.getDefiningOp<block::CastOp>())
        value = cast.getValue();
      return value.getDefiningOp<arith::ConstantOp>();
    };
    auto source = dyn_cast<block::FragmentType>(gather.getSource().getType());
    auto result = dyn_cast<block::FragmentType>(gather.getResult().getType());
    if (!gather.getValid() && gather.getCoordinates().size() == 1 &&
        gather.getSourceAxes().size() == 1 &&
        sameGatherFreeAxes(source, result, gather.getSourceAxes().front()) &&
        samePhysicalShape(gather.getCoordinates().front().getType(), result))
      continue;
    if (source && !source.getShape().empty() &&
        gather.getCoordinates().size() == source.getShape().size() &&
        llvm::all_of(gather.getCoordinates(), [](Value coordinate) {
          Type element = coordinate.getType();
          if (auto fragment = dyn_cast<block::FragmentType>(element))
            element = fragment.getElementType();
          return isa<IndexType, IntegerType>(element);
        }) &&
        (result || !gather.getValid() || gather.getValid().getType().isInteger(1))) {
      OpBuilder builder(gather);
      Location location = gather.getLoc();
      Type indexType = builder.getIndexType();
      Type predicateType = builder.getI1Type();
      if (result) {
        indexType = block::FragmentType::get(
            kernel.getContext(), builder.getIndexType(), result.getShape(),
            result.getAxisMaps(), result.getValidity(), result.getOwner());
        predicateType = block::FragmentType::get(
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
        auto projected = block::projectPhysicalValueToSchema(
            builder, location, valid, cast<block::FragmentType>(predicateType));
        if (failed(projected))
          return gather.emitOpError("gather validity has no exact result-axis projection");
        valid = *projected;
      }
      Value safeIndex;
      if (source.getShape().size() > 1) {
        auto elements = cast<block::PhysicalExprAttr>(source.getShape()[0]);
        auto first = projectIndex(builder.create<arith::ConstantIndexOp>(location, 0));
        if (failed(first))
          return failure();
        coordinate = *first;
        for (auto [axis, value] : llvm::enumerate(coordinates)) {
          auto extent = cast<block::PhysicalExprAttr>(source.getShape()[axis]);
          if (axis)
            elements = block::PhysicalExprAttr::get(
                kernel.getContext(),
                block::PhysicalExprKind::Multiply, 0,
                builder.getStringAttr(""), builder.getArrayAttr({elements, extent}));
          Value size = builder.create<block::PhysicalExprOp>(
              location, builder.getIndexType(), extent);
          auto projectedSize = projectIndex(size);
          if (failed(projectedSize))
            return failure();
          coordinate = builder.create<block::BinaryOp>(
              location, indexType, coordinate, *projectedSize, BinaryOperator::Multiply);
          coordinate = builder.create<block::BinaryOp>(
              location, indexType, coordinate, value, BinaryOperator::Add);
        }
        if (auto count = evaluateCompileTimeExpression(elements))
          elements = block::PhysicalExprAttr::get(
              kernel.getContext(),
              block::PhysicalExprKind::Constant, *count,
              builder.getStringAttr(""), builder.getArrayAttr({}));
        auto mapping = block::AxisMapAttr::get(
            kernel.getContext(), nextSource++, 0, nextDimension++, 0, true);
        auto flatType = block::FragmentType::get(
            kernel.getContext(), source.getElementType(),
            builder.getArrayAttr({elements}), builder.getArrayAttr({mapping}),
            source.getValidity(), source.getOwner());
        auto reassociation = block::inferReshapeReassociation(source, flatType);
        if (failed(reassociation))
          return gather.emitOpError("scalar gather has no exact row-major linearization");
        sourceValue = builder.create<block::ReshapeOp>(
            location, flatType, sourceValue, *reassociation);
        auto safe = safeGatherIndex(builder, location, coordinate, elements, valid);
        if (failed(safe))
          return failure();
        safeIndex = safe->first;
        valid = safe->second;
      } else {
        auto safe = safeGatherIndex(
            builder, location, coordinate,
            cast<block::PhysicalExprAttr>(source.getShape()[0]), valid);
        if (failed(safe))
          return gather.emitOpError("scalar gather coordinate has no safe physical index projection");
        safeIndex = safe->first;
        valid = safe->second;
      }
      Type loadedType = gather.getResult().getType();
      if (result && result.getShape().size() > 1) {
        auto elements = cast<block::PhysicalExprAttr>(result.getShape()[0]);
        for (Attribute extent : result.getShape().getValue().drop_front())
          elements = block::PhysicalExprAttr::get(
              kernel.getContext(),
              block::PhysicalExprKind::Multiply, 0,
              builder.getStringAttr(""), builder.getArrayAttr({elements, extent}));
        auto mapping = block::AxisMapAttr::get(
            kernel.getContext(), nextSource++, 0, nextDimension++, 0, true);
        auto flatIndex = block::FragmentType::get(
            kernel.getContext(), builder.getIndexType(), builder.getArrayAttr({elements}),
            builder.getArrayAttr({mapping}), result.getValidity(), result.getOwner());
        auto relation = block::inferReshapeReassociation(
            cast<block::FragmentType>(safeIndex.getType()), flatIndex);
        if (failed(relation))
          return gather.emitOpError("gather result has no exact row-major linearization");
        safeIndex = builder.create<block::ReshapeOp>(location, flatIndex, safeIndex, *relation);
        loadedType = block::FragmentType::get(
            kernel.getContext(), result.getElementType(), flatIndex.getShape(),
            flatIndex.getAxisMaps(), result.getValidity(), result.getOwner());
      }
      auto loaded = builder.create<block::GatherOp>(
          gather.getLoc(), loadedType, sourceValue,
          ValueRange{safeIndex}, Value(), Value(), ArrayRef<int64_t>{0});
      Value loadedValue = loaded.getResult();
      if (loadedType != gather.getResult().getType()) {
        auto relation = block::inferReshapeReassociation(
            cast<block::FragmentType>(loadedType), result);
        if (failed(relation))
          return gather.emitOpError("linear gather has no exact result reassociation");
        loadedValue = builder.create<block::ReshapeOp>(location, result, loadedValue, *relation);
      }
      // An unmasked gather already requires active coordinates to be in bounds.
      // Its row-major linearization has no inactive fill to apply. Masked
      // gathers retain their original validity/fill after the safe native read.
      Value replacement = loadedValue;
      if (gather.getValid())
        replacement = builder.create<block::SelectOp>(
            gather.getLoc(), gather.getResult().getType(), valid,
            loadedValue, gather.getFill());
      if (Attribute origin = gather->getAttr(block::originAttr))
        replacement.getDefiningOp()->setAttr(block::originAttr, origin);
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
        auto expandedSourceType = block::FragmentType::get(
            gather.getContext(), source.getElementType(), result.getShape(),
            result.getAxisMaps(), result.getValidity(), result.getOwner());
        Value expandedSource = builder.create<block::BroadcastOp>(
            gather.getLoc(), expandedSourceType, gather.getSource());
        Value coordinate = gather.getCoordinates().front();
        auto coordinateType = block::FragmentType::get(
            gather.getContext(), coordinate.getType(), result.getShape(),
            result.getAxisMaps(), result.getValidity(), result.getOwner());
        if (auto coordinateFragment =
                dyn_cast<block::FragmentType>(coordinate.getType())) {
          coordinateType = block::FragmentType::get(
              gather.getContext(), coordinateFragment.getElementType(),
              result.getShape(), result.getAxisMaps(), result.getValidity(),
              result.getOwner());
        }
        coordinate = builder.create<block::BroadcastOp>(
            gather.getLoc(), coordinateType, coordinate);
        FailureOr<Value> zero =
            zeroLike(builder, gather.getLoc(), coordinateType);
        if (succeeded(zero)) {
          Value safeIndex = coordinate;
          if (gather.getValid())
            safeIndex = builder.create<block::SelectOp>(
                gather.getLoc(), coordinateType, gather.getValid(), coordinate,
                *zero);
          auto safeGather = builder.create<block::GatherOp>(
              gather.getLoc(), result, expandedSource, ValueRange{safeIndex},
              Value(), Value(), ArrayRef<int64_t>{static_cast<int64_t>(*selectedAxis)});
          Value selected = safeGather.getResult();
          if (gather.getValid())
            selected = builder.create<block::SelectOp>(
                gather.getLoc(), result, gather.getValid(), safeGather.getResult(),
                gather.getFill());
          if (Attribute origin = gather->getAttr(block::originAttr))
            selected.getDefiningOp()->setAttr(block::originAttr, origin);
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
              ? dyn_cast<block::PhysicalExprAttr>(source.getShape()[selectedAxis])
              : block::PhysicalExprAttr();
      bool compatible = selectedExtent &&
          ((selectedExtent.getKind() == block::PhysicalExprKind::Constant &&
            selectedExtent.getValue() > 0) ||
           selectedExtent.getKind() == block::PhysicalExprKind::Parameter);
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
        selectedShape[selectedAxis] = block::PhysicalExprAttr::get(
            gather.getContext(),
            block::PhysicalExprKind::Constant, 1,
            builder.getStringAttr(""), builder.getArrayAttr({}));
        auto coordinateType = block::FragmentType::get(
            gather.getContext(), block::uniformElementType(gather.getCoordinates().front().getType()),
            builder.getArrayAttr(selectedShape), source.getAxisMaps(),
            source.getValidity(), source.getOwner());
        auto predicateType = block::FragmentType::get(
            gather.getContext(), builder.getI1Type(),
            builder.getArrayAttr(selectedShape), source.getAxisMaps(),
            source.getValidity(), source.getOwner());
        auto selectedResultType = block::FragmentType::get(
            gather.getContext(), result.getElementType(),
            builder.getArrayAttr(selectedShape), source.getAxisMaps(),
            result.getValidity(), result.getOwner());
        Value coordinate = builder.create<block::BroadcastOp>(
            gather.getLoc(), coordinateType, gather.getCoordinates().front());
        Value valid, fill;
        if (gather.getValid()) {
          valid = builder.create<block::BroadcastOp>(
              gather.getLoc(), predicateType, gather.getValid());
          fill = builder.create<block::BroadcastOp>(
              gather.getLoc(), selectedResultType, gather.getFill());
        }
        FailureOr<Value> zero = zeroLike(builder, gather.getLoc(), coordinateType);
        if (succeeded(zero)) {
          Value safeIndex = coordinate;
          if (valid)
            safeIndex = builder.create<block::SelectOp>(
                gather.getLoc(), coordinateType, valid, coordinate, *zero);
          auto safeGather = builder.create<block::GatherOp>(
              gather.getLoc(), selectedResultType, gather.getSource(),
              ValueRange{safeIndex}, Value(), Value(), gather.getSourceAxes());
          Value selected = safeGather.getResult();
          if (valid)
            selected = builder.create<block::SelectOp>(
                gather.getLoc(), selectedResultType, valid, safeGather.getResult(),
                fill);
          FailureOr<ArrayAttr> reassociation =
              block::inferReshapeReassociation(selectedResultType, result);
          if (failed(reassociation))
            return gather.emitOpError(
                "gather fallback has no exact row-major reassociation");
          auto reshaped = builder.create<block::ReshapeOp>(
              gather.getLoc(), result, selected, *reassociation);
          if (Attribute origin = gather->getAttr(block::originAttr))
            reshaped->setAttr(block::originAttr, origin);
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
        isa<block::FragmentType>(gather.getCoordinates().front().getType()) &&
        gather.getSourceAxes().size() == 1 && source) {
      unsigned axis = gather.getSourceAxes().front();
      auto coordinate = scalarConstant(gather.getCoordinates().front());
      auto predicate = scalarConstant(gather.getValid());
      auto extent = axis < source.getShape().size()
                        ? dyn_cast<block::PhysicalExprAttr>(source.getShape()[axis])
                        : block::PhysicalExprAttr();
      auto coordinateValue =
          coordinate ? dyn_cast<IntegerAttr>(coordinate.getValue()) : IntegerAttr();
      auto predicateValue = predicate ? dyn_cast<IntegerAttr>(predicate.getValue())
                                      : IntegerAttr();
      if (extent &&
          extent.getKind() == block::PhysicalExprKind::Constant &&
          coordinateValue && coordinateValue.getInt() >= 0 &&
          coordinateValue.getInt() < extent.getValue() && predicateValue &&
          predicateValue.getValue().isOne()) {
        OpBuilder builder(gather);
        auto replacement = builder.create<block::GatherOp>(
            gather.getLoc(), gather.getResult().getType(), gather.getSource(),
            gather.getCoordinates(), Value(), Value(), gather.getSourceAxes());
        if (Attribute origin = gather->getAttr(block::originAttr))
          replacement->setAttr(block::originAttr, origin);
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
      auto predicateType = dyn_cast<block::FragmentType>(gather.getValid().getType());
      Type indexElement = block::uniformElementType(coordinate.getType());
      if (!predicateType || !isa<IntegerType, IndexType>(indexElement))
        continue;
      if (isa<block::FragmentType>(coordinate.getType()) &&
          (!source || !result || gather.getSourceAxes().size() != 1 ||
           source.getShape().size() != result.getShape().size()))
        continue;
      auto coordinateType = block::FragmentType::get(
          gather.getContext(), indexElement, predicateType.getShape(),
          predicateType.getAxisMaps(), predicateType.getValidity(),
          predicateType.getOwner());
      auto projected = block::projectPhysicalValueToSchema(
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
        cast<block::PhysicalExprAttr>(source.getShape()[axis]), gather.getValid());
    if (failed(safe))
      return gather.emitOpError("gather coordinate has no safe physical index projection");
    auto safeGather = builder.create<block::GatherOp>(
        gather.getLoc(), gather.getResult().getType(), gather.getSource(),
        ValueRange{safe->first}, Value(), Value(), gather.getSourceAxes());
    auto selected = builder.create<block::SelectOp>(
        gather.getLoc(), gather.getResult().getType(), gather.getValid(),
        safeGather.getResult(), gather.getFill());
    if (Attribute origin = gather->getAttr(block::originAttr))
      selected->setAttr(block::originAttr, origin);
    gather.getResult().replaceAllUsesWith(selected.getResult());
    gather.erase();
  }
  return success();
}

LogicalResult legalizeExpandingGathers(func::FuncOp kernel) {
  auto [nextSource, nextDimension] = block::nextPhysicalAxisIdentities(kernel);
  SmallVector<block::GatherOp> gathers;
  kernel.walk([&](block::GatherOp gather) { gathers.push_back(gather); });
  for (block::GatherOp gather : gathers) {
    if (gather.getValid() || gather.getCoordinates().size() != 1 ||
        gather.getSourceAxes().size() != 1)
      continue;
    auto source = dyn_cast<block::FragmentType>(gather.getSource().getType());
    auto result = dyn_cast<block::FragmentType>(gather.getResult().getType());
    auto indices = dyn_cast<block::FragmentType>(
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
    auto sourceExtent = cast<block::PhysicalExprAttr>(source.getShape()[axis]);
    auto resultExtent = cast<block::PhysicalExprAttr>(result.getShape()[axis]);
    if (!compatible || sourceExtent == resultExtent ||
        !isTritonFragmentExtent(sourceExtent) ||
        !isTritonFragmentExtent(resultExtent))
      continue;
    auto constantExtent = [](block::PhysicalExprAttr extent) {
      return extent.getKind() ==
             block::PhysicalExprKind::Constant;
    };
    bool constantShape = constantExtent(sourceExtent) &&
                         constantExtent(resultExtent);
    if (constantShape && resultExtent.getValue() <= sourceExtent.getValue())
      continue;

    OpBuilder builder(gather);
    Location location = gather.getLoc();
    auto product = [&](ArrayRef<Attribute> shape) {
      auto extent = block::PhysicalExprAttr::get(
          kernel.getContext(),
          block::PhysicalExprKind::Constant, 1,
          builder.getStringAttr(""), builder.getArrayAttr({}));
      if (shape.empty())
        return extent;
      extent = cast<block::PhysicalExprAttr>(shape.front());
      for (Attribute dimension : shape.drop_front())
        extent = block::PhysicalExprAttr::get(
            kernel.getContext(),
            block::PhysicalExprKind::Multiply, 0,
            builder.getStringAttr(""), builder.getArrayAttr({extent, dimension}));
      return extent;
    };
    auto flatSourceAxis = block::AxisMapAttr::get(
        kernel.getContext(), nextSource++, 0, nextDimension++, 0, true);
    auto flatResultAxis = block::AxisMapAttr::get(
        kernel.getContext(), nextSource++, 0, nextDimension++, 0, true);
    auto sourceElements = product(source.getShape().getValue());
    auto resultElements = product(result.getShape().getValue());
    auto flatType = [&](block::FragmentType original, Type element,
                        block::PhysicalExprAttr extent, block::AxisMapAttr mapping) {
      return block::FragmentType::get(
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
                         block::FragmentType target) -> FailureOr<Value> {
        auto relation = block::inferReshapeReassociation(
            cast<block::FragmentType>(value.getType()), target);
        if (failed(relation))
          return failure();
        return Value(nested.create<block::ReshapeOp>(location, target, value,
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
        index = nested.create<block::CastOp>(location, flatIndexType, index);
      auto extentValue = [&](block::PhysicalExprAttr extent) -> Value {
        return nested.create<block::PhysicalExprOp>(
            location, nested.getIndexType(), extent);
      };
      auto broadcastExtent = [&](block::PhysicalExprAttr extent) -> Value {
        return nested.create<block::BroadcastOp>(location, flatIndexType,
                                               extentValue(extent));
      };
      auto binary = [&](Value lhs, Value rhs, BinaryOperator kind) -> Value {
        return nested.create<block::BinaryOp>(location, flatIndexType, lhs, rhs,
                                            kind);
      };
      Value zero = nested.create<arith::ConstantIndexOp>(location, 0);
      Value one = nested.create<arith::ConstantIndexOp>(location, 1);
      Value count = extentValue(resultElements);
      Value ordinal = nested.create<block::MakeRangeOp>(
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
      Value selected = nested.create<block::GatherOp>(
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
      Value sourceSize = builder.create<block::PhysicalExprOp>(
          location, builder.getIndexType(), sourceExtent);
      Value resultSize = builder.create<block::PhysicalExprOp>(
          location, builder.getIndexType(), resultExtent);
      Value expanding = builder.create<block::CompareOp>(
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
    if (Attribute origin = gather->getAttr(block::originAttr))
      replacement.getDefiningOp()->setAttr(block::originAttr, origin);
    gather.getResult().replaceAllUsesWith(replacement);
    gather.erase();
  }
  return success();
}

} // namespace intent::triton::detail
