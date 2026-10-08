#include "Accesses.h"
#include "Coordinates.h"
#include "TileIndices.h"
#include "Intent/Dialect/GPU/Analysis/IndexRelations.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/IR/FragmentOpInterface.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/PhysicalParameters.h"
#include "Intent/Target/CuTile/Analysis/IndexBounds.h"
#include "Intent/Target/CuTile/IR/CuTileOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/ADT/ScopeExit.h"

using namespace mlir;

namespace intent::cutile {

namespace {

FailureOr<std::pair<Value, bool>>
extractionTileIndex(OpBuilder &builder, Location location, Value coordinate,
                    gpu::MakeRangeOp range, Value sourceExtent) {
  if (coordinate == range.getResult()) {
    auto index = tileIndex(builder, location, range.getStart(), range.getExtent());
    if (failed(index))
      return failure();
    return std::pair{*index, true};
  }
  if (auto reshape = coordinate.getDefiningOp<gpu::ReshapeOp>()) {
    auto relations = gpu::queryFragmentOperandRelations(reshape);
    if (failed(relations) || !relations->front().preservesNonUnitAxes())
      return failure();
    return extractionTileIndex(builder, location, reshape.getValue(), range,
                               sourceExtent);
  }
  if (isa_and_nonnull<gpu::BroadcastOp, gpu::TransposeOp>(
          coordinate.getDefiningOp()))
    return extractionTileIndex(builder, location,
                               coordinate.getDefiningOp()->getOperand(0),
                               range, sourceExtent);
  if (Value scalar = uniformScalarFill(coordinate)) {
    if (auto index = tileIndex(builder, location, scalar, range.getExtent());
        succeeded(index))
      return std::pair{*index, false};
    // Native tile extents are powers of two. A source-aligned shift either
    // selects whole sub-tiles, or is periodic in a repeated wider extraction.
    auto index = tileIndex(builder, location, scalar, sourceExtent);
    if (failed(index))
      return failure();
    Value ratio = builder.create<gpu::BinaryOp>(
        location, builder.getIndexType(), sourceExtent, range.getExtent(),
        BinaryOperator::FloorDivide);
    return std::pair{Value(builder.create<gpu::BinaryOp>(
        location, builder.getIndexType(), *index, ratio,
        BinaryOperator::Multiply)), false};
  }
  auto binary = coordinate.getDefiningOp<gpu::BinaryOp>();
  if (!binary || (binary.getOperatorKind() != BinaryOperator::Add &&
                  binary.getOperatorKind() != BinaryOperator::Subtract))
    return failure();
  if (binary.getOperatorKind() == BinaryOperator::Subtract &&
      binary.getLhs() == range.getResult()) {
    if (Value offset = uniformScalarFill(binary.getRhs())) {
      Value start = builder.create<gpu::BinaryOp>(
          location, builder.getIndexType(), range.getStart(), offset,
          BinaryOperator::Subtract);
      if (auto index = tileIndex(builder, location, start, range.getExtent());
          succeeded(index))
        return std::pair{*index, true};
    }
  }
  auto lhs = extractionTileIndex(builder, location, binary.getLhs(), range,
                                 sourceExtent);
  auto rhs = extractionTileIndex(builder, location, binary.getRhs(), range,
                                 sourceExtent);
  if (failed(lhs) || failed(rhs))
    return failure();
  if ((lhs->second && rhs->second) ||
      (rhs->second && binary.getOperatorKind() == BinaryOperator::Subtract))
    return failure();
  return std::pair{Value(builder.create<gpu::BinaryOp>(
      location, builder.getIndexType(), lhs->first, rhs->first,
      binary.getOperatorKind())), lhs->second || rhs->second};
}

Value repeatTileForExtraction(OpBuilder &builder, Location location, Value value,
                              unsigned axis, gpu::PhysicalExprAttr requested) {
  auto source = cast<gpu::FragmentType>(value.getType());
  auto extent = cast<gpu::PhysicalExprAttr>(source.getShape()[axis]);
  if (extent == requested)
    return value;
  auto context = builder.getContext();
  auto expression = [&](gpu::PhysicalExprKind kind,
                         ArrayRef<Attribute> operands) {
    return gpu::PhysicalExprAttr::get(context, kind, 0,
                                     builder.getStringAttr(""),
                                     builder.getArrayAttr(operands));
  };
  auto unit = gpu::PhysicalExprAttr::get(
      context, gpu::PhysicalExprKind::Constant, 1,
      builder.getStringAttr(""), builder.getArrayAttr({}));
  auto repeat = expression(gpu::PhysicalExprKind::FloorDiv,
      {expression(gpu::PhysicalExprKind::Maximum, {extent, requested}), extent});
  auto expanded = expression(gpu::PhysicalExprKind::Multiply, {repeat, extent});
  SmallVector<Attribute> splitShape, splitMaps, splitGroups, mergeGroups;
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  uint64_t repeatSource = gpu::nextPhysicalAxisIdentities(kernel).first;
  for (unsigned original = 0; original < source.getShape().size(); ++original) {
    SmallVector<int64_t> splitAxes;
    if (original == axis) {
      splitAxes.push_back(splitShape.size());
      splitMaps.push_back(gpu::AxisMapAttr::get(
          context, repeatSource, 0, 0, splitShape.size(), true));
      splitShape.push_back(unit);
    }
    splitAxes.push_back(splitShape.size());
    auto mapping = cast<gpu::AxisMapAttr>(source.getAxisMaps()[original]);
    splitMaps.push_back(gpu::AxisMapAttr::get(
        context, mapping.getSourceId(), mapping.getSourceAxis(),
        mapping.getDimensionId(), splitShape.size(), mapping.getDerived()));
    splitShape.push_back(source.getShape()[original]);
    splitGroups.push_back(gpu::ReshapeGroupAttr::get(
        context, builder.getDenseI64ArrayAttr({original}),
        builder.getDenseI64ArrayAttr(splitAxes)));
    mergeGroups.push_back(gpu::ReshapeGroupAttr::get(
        context, builder.getDenseI64ArrayAttr(splitAxes),
        builder.getDenseI64ArrayAttr({original})));
  }
  auto fragment = [&](ArrayRef<Attribute> shape, ArrayRef<Attribute> mappings) {
    return gpu::FragmentType::get(context, source.getElementType(),
        builder.getArrayAttr(shape), builder.getArrayAttr(mappings),
        source.getValidity(), source.getOwner());
  };
  Value split = builder.create<gpu::ReshapeOp>(
      location, fragment(splitShape, splitMaps), value,
      builder.getArrayAttr(splitGroups));
  splitShape[axis] = repeat;
  Value repeated = builder.create<gpu::BroadcastOp>(
      location, fragment(splitShape, splitMaps), split);
  SmallVector<Attribute> resultShape(source.getShape().getValue());
  resultShape[axis] = expanded;
  return builder.create<gpu::ReshapeOp>(
      location, fragment(resultShape, source.getAxisMaps().getValue()), repeated,
      builder.getArrayAttr(mergeGroups));
}

struct RectangularCoordinate {
  gpu::MakeRangeOp range;
  unsigned resultAxis;
  Value sourceExtent;
  Value tileIndex;
};

FailureOr<RectangularCoordinate> rectangularCoordinate(
    OpBuilder &builder, gpu::GatherOp gather, unsigned slot, Value coordinate,
    gpu::PhysicalProgramAnalysis &analysis) {
  auto access = cast<gpu::AccessOpInterface>(gather.getOperation());
  auto source = cast<gpu::FragmentType>(gather.getSource().getType());
  auto result = dyn_cast<gpu::FragmentType>(gather.getResult().getType());
  if (!result) return failure();
  auto ranges = analysis.sourceRanges(coordinate);
  if (ranges.roots.size() != 1 || !gpu::isUnitStepRange(ranges.roots.front()))
    return failure();
  auto range = ranges.roots.front();
  auto resultAxis = nativeAccessRangeAxis(access, slot, range);
  if (failed(resultAxis) || result.getShape()[*resultAxis] !=
          range.getResult().getType().getShape()[0])
    return failure();
  auto fullExtent = cast<gpu::PhysicalExprAttr>(source.getShape()[gather.getSourceAxes()[slot]]);
  Value full;
  if (fullExtent.getKind() == gpu::PhysicalExprKind::Parameter) {
    auto kernel = gather->getParentOfType<func::FuncOp>();
    auto parameter = gpu::queryParameterBySymbol(kernel, fullExtent.getParameterReference().getName());
    if (failed(parameter)) return failure();
    full = gpu::materializeParameter(builder, gather.getLoc(), parameter->getReference());
  } else {
    full = builder.create<gpu::PhysicalExprOp>(gather.getLoc(), builder.getIndexType(), fullExtent);
  }
  auto index = extractionTileIndex(builder, gather.getLoc(), coordinate, range, full);
  if (failed(index) || !index->second) return failure();
  return RectangularCoordinate{range, *resultAxis, full, index->first};
}

} // namespace

bool requiresFragmentStorage(gpu::GatherOp gather) {
  auto kernel = gather->getParentOfType<func::FuncOp>();
  if (!isa<gpu::FragmentType>(gather.getResult().getType())) {
    auto source = cast<gpu::FragmentType>(gather.getSource().getType());
    bool dynamicSelection = false;
    for (auto [coordinate, axis] :
         llvm::zip(gather.getCoordinates(), gather.getSourceAxes())) {
      if (isa<gpu::FragmentType>(coordinate.getType())) return false;
      if (getCompileTimeScalar(coordinate)) continue;
      auto extent = gpu::IndexRelations().constant(
          cast<gpu::PhysicalExprAttr>(source.getShape()[axis]), kernel);
      dynamicSelection |= !extent || *extent != 1;
    }
    // Dynamic scalar lookup uses an array gather from the completed snapshot.
    // Constant selections and unit axes keep the native tile extraction.
    return dynamicSelection;
  }
  gpu::PhysicalProgramAnalysis analysis(kernel);
  OpBuilder builder(gather);
  Operation *previous = gather->getPrevNode();
  auto discardQueryValues = llvm::make_scope_exit([&] {
    while (gather->getPrevNode() != previous) gather->getPrevNode()->erase();
  });
  for (auto [slot, value] : llvm::enumerate(gather.getCoordinates())) {
    while (isa_and_nonnull<gpu::SplatOp, gpu::BroadcastOp>(value.getDefiningOp()))
      value = value.getDefiningOp()->getOperand(0);
    if (isa<gpu::FragmentType>(value.getType()) && !getCompileTimeScalar(value) &&
        failed(rectangularCoordinate(builder, gather, slot, value, analysis)))
      return true;
  }
  return false;
}

LogicalResult formFragmentExtractions(func::FuncOp kernel,
    ArrayRef<gpu::GatherOp> gathers, NativeFormRewriter &rewriter) {
  for (gpu::GatherOp gather : gathers) {
    auto access = cast<gpu::AccessOpInterface>(gather.getOperation());
    gpu::PhysicalProgramAnalysis analysis(kernel);
    auto source = dyn_cast<gpu::FragmentType>(access.getAccessResource().getType());
    auto result = dyn_cast<gpu::FragmentType>(access.getAccessResult().getType());
    if (!source)
      return gather.emitOpError("cuTile tile extraction requires a source fragment");
    if (access.getAccessCoordinates().size() != access.getAccessSourceAxes().size())
      return gather.emitOpError("cuTile tile extraction source axes are incomplete");
    SmallVector<Value> coordinates(source.getShape().size());
    OpBuilder builder(gather);
    auto unit = gpu::PhysicalExprAttr::get(
        kernel.getContext(), gpu::PhysicalExprKind::Constant,
        1, builder.getStringAttr(""), builder.getArrayAttr({}));
    SmallVector<Attribute> extractionShape(source.getShape().size(), unit);
    SmallVector<bool> slicedAxes(source.getShape().size(), false);
    for (auto [slot, originalCoordinate] : llvm::enumerate(access.getAccessCoordinates())) {
      Value coordinate = originalCoordinate;
      int64_t sourceAxis = access.getAccessSourceAxes()[slot];
      if (sourceAxis < 0 ||
          sourceAxis >= static_cast<int64_t>(coordinates.size()) ||
          coordinates[sourceAxis])
        return gather.emitOpError(
            "cuTile tile extraction source axes are not a unique subset");
      while (true) {
        if (auto splat = coordinate.getDefiningOp<gpu::SplatOp>())
          coordinate = splat.getValue();
        else if (auto broadcast = coordinate.getDefiningOp<gpu::BroadcastOp>())
          coordinate = broadcast.getValue();
        else
          break;
      }
      if (result) {
        if (isa<gpu::FragmentType>(coordinate.getType()) &&
            !getCompileTimeScalar(coordinate)) {
          auto selected = rectangularCoordinate(builder, gather, slot, coordinate, analysis);
          if (failed(selected))
            return gather.emitOpError(
                "cuTile coordinate is not an aligned rectangular sub-tile");
          auto range = selected->range;
          Value full = selected->sourceExtent;
          extractionShape[sourceAxis] = result.getShape()[selected->resultAxis];
          Value extractionExtent = builder.create<gpu::BinaryOp>(
              gather.getLoc(), builder.getIndexType(), full, range.getExtent(),
              BinaryOperator::Maximum);
          Value count = builder.create<gpu::BinaryOp>(
              gather.getLoc(), builder.getIndexType(), extractionExtent, range.getExtent(),
              BinaryOperator::FloorDivide);
          Value one = builder.create<arith::ConstantIndexOp>(gather.getLoc(), 1);
          Value zero = builder.create<arith::ConstantIndexOp>(gather.getLoc(), 0);
          Value last = builder.create<gpu::BinaryOp>(
              gather.getLoc(), builder.getIndexType(), count, one,
              BinaryOperator::Subtract);
          Value nonnegative = builder.create<gpu::BinaryOp>(
              gather.getLoc(), builder.getIndexType(), selected->tileIndex, zero,
              BinaryOperator::Maximum);
          // An aligned sub-tile is either entirely inside or outside the
          // physical source. Preserve the gather mask/fill while keeping the
          // native extraction address valid for inactive sub-tiles.
          coordinate = builder.create<gpu::BinaryOp>(
              gather.getLoc(), builder.getIndexType(), nonnegative, last,
              BinaryOperator::Minimum);
          coordinates[sourceAxis] = coordinate;
          slicedAxes[sourceAxis] = true;
          continue;
        }
        auto integer =
            dyn_cast_or_null<IntegerAttr>(getCompileTimeScalar(coordinate));
        if (!integer && !isa<gpu::FragmentType>(coordinate.getType())) {
          // The shared access contract bounds active members. Clamp only the
          // extraction address; the original validity still selects the fill.
          if (!coordinate.getType().isIndex())
            coordinate = builder.create<gpu::CastOp>(
                gather.getLoc(), builder.getIndexType(), coordinate);
          Value extent = builder.create<gpu::PhysicalExprOp>(
              gather.getLoc(), builder.getIndexType(),
              cast<gpu::PhysicalExprAttr>(source.getShape()[sourceAxis]));
          Value zero = builder.create<arith::ConstantIndexOp>(gather.getLoc(), 0);
          Value one = builder.create<arith::ConstantIndexOp>(gather.getLoc(), 1);
          Value last = builder.create<gpu::BinaryOp>(
              gather.getLoc(), builder.getIndexType(), extent, one,
              BinaryOperator::Subtract);
          Value nonnegative = builder.create<gpu::BinaryOp>(
              gather.getLoc(), builder.getIndexType(), coordinate, zero,
              BinaryOperator::Maximum);
          coordinates[sourceAxis] = builder.create<gpu::BinaryOp>(
              gather.getLoc(), builder.getIndexType(), nonnegative, last,
              BinaryOperator::Minimum);
          continue;
        }
        auto extent = gpu::IndexRelations().constant(
            cast<gpu::PhysicalExprAttr>(source.getShape()[sourceAxis]), kernel);
        if (!integer || !extent || integer.getInt() < 0 || integer.getInt() >= *extent)
          return gather.emitOpError(
              "cuTile rectangular tile extraction requires in-bounds constant selected coordinates");
        coordinate = builder.create<arith::ConstantIntOp>(
            gather.getLoc(), integer.getInt(), 32);
      } else {
        if (isa<gpu::FragmentType>(coordinate.getType()))
          return gather.emitOpError("cuTile scalar extraction requires scalar coordinates");
        if (access.getAccessValidity()) {
          Value zero = builder.create<arith::ConstantOp>(
              gather.getLoc(), coordinate.getType(),
              builder.getIntegerAttr(coordinate.getType(), 0));
          coordinate = builder.create<gpu::SelectOp>(
              gather.getLoc(), coordinate.getType(), access.getAccessValidity(), coordinate, zero);
        }
      }
      coordinates[sourceAxis] = coordinate;
    }
    SmallVector<int64_t> retainedAxes;
    for (unsigned axis = 0; axis < coordinates.size(); ++axis) {
      if (coordinates[axis]) {
        if (slicedAxes[axis])
          retainedAxes.push_back(axis);
        continue;
      }
      if (!result)
        return gather.emitOpError("cuTile scalar extraction must select every source axis");
      retainedAxes.push_back(axis);
      extractionShape[axis] = source.getShape()[axis];
      coordinates[axis] = builder.create<arith::ConstantIntOp>(gather.getLoc(), 0, 32);
    }
    // In-bounds coordinates fit i32 under cuTile's tile-size limit.
    // This does not narrow external view dimensions or address arithmetic.
    for (Value &coordinate : coordinates)
      if (!coordinate.getType().isInteger(32))
        coordinate = builder.create<gpu::CastOp>(
            gather.getLoc(), builder.getI32Type(), coordinate);
    Value tile = access.getAccessResource();
    for (unsigned axis = 0; axis < slicedAxes.size(); ++axis)
      if (slicedAxes[axis])
        tile = repeatTileForExtraction(
            builder, gather.getLoc(), tile, axis,
            cast<gpu::PhysicalExprAttr>(extractionShape[axis]));
    Type extractedType = access.getAccessResult().getType();
    if (result) {
      SmallVector<Attribute> shape, maps;
      auto tileType = cast<gpu::FragmentType>(tile.getType());
      for (auto [resultAxis, sourceAxis] : llvm::enumerate(retainedAxes)) {
        shape.push_back(extractionShape[sourceAxis]);
        auto map = cast<gpu::AxisMapAttr>(tileType.getAxisMaps()[sourceAxis]);
        maps.push_back(gpu::AxisMapAttr::get(
            kernel.getContext(), map.getSourceId(), map.getSourceAxis(),
            map.getDimensionId(), resultAxis, map.getDerived()));
      }
      extractedType = gpu::FragmentType::get(
          kernel.getContext(), result.getElementType(), builder.getArrayAttr(shape),
          builder.getArrayAttr(maps), result.getValidity(), result.getOwner());
      if (builder.getArrayAttr(shape) != result.getShape() &&
          !gpu::queryBroadcastProjection(
               cast<gpu::FragmentType>(extractedType), result).isExact())
        return gather.emitOpError(
            "cuTile extraction requires an exact retained-axis broadcast");
    }
    auto replacement = builder.create<ExtractOp>(
        gather.getLoc(), extractedType, tile,
        coordinates, builder.getArrayAttr(extractionShape),
        builder.getDenseI64ArrayAttr(retainedAxes));
    if (Attribute origin = gather->getAttr(gpu::originAttr))
      replacement->setAttr(gpu::originAttr, origin);
    Value value = replacement.getResult();
    if (extractedType != access.getAccessResult().getType()) {
      auto extracted = cast<gpu::FragmentType>(extractedType);
      if (gpu::queryBroadcastProjection(extracted, result).isExact()) {
        value = builder.create<gpu::BroadcastOp>(gather.getLoc(), result, value);
      } else {
        auto reassociation = gpu::inferReshapeReassociation(extracted, result);
        if (failed(reassociation))
          return gather.emitOpError(
              "cuTile extraction has no row-major result domain projection");
        value = builder.create<gpu::ReshapeOp>(
            gather.getLoc(), result, value, *reassociation);
      }
    }
    if (access.getAccessValidity()) {
      auto selected = builder.create<gpu::SelectOp>(
          gather.getLoc(), access.getAccessResult().getType(), access.getAccessValidity(),
          value, access.getAccessFill());
      if (Attribute origin = gather->getAttr(gpu::originAttr))
        selected->setAttr(gpu::originAttr, origin);
      value = selected.getResult();
    }
    rewriter.replace(gather, ValueRange{value});
  }
  return success();
}

} // namespace intent::cutile
