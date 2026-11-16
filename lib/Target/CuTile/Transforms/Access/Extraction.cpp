#include "Accesses.h"
#include "Coordinates.h"
#include "TileIndices.h"
#include "Intent/Dialect/Block/Analysis/IndexRelations.h"
#include "Intent/Dialect/Block/Analysis/ValueSchema.h"
#include "Intent/Dialect/Block/IR/Program.h"
#include "Intent/Dialect/Block/IR/FragmentOpInterface.h"
#include "Intent/Dialect/Block/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/Block/Transforms/Configuration/PhysicalParameters.h"
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
                    block::MakeRangeOp range, Value sourceExtent) {
  if (coordinate == range.getResult()) {
    auto index = tileIndex(builder, location, range.getStart(), range.getExtent());
    if (failed(index))
      return failure();
    return std::pair{*index, true};
  }
  if (auto reshape = coordinate.getDefiningOp<block::ReshapeOp>()) {
    auto relations = block::queryFragmentOperandRelations(reshape);
    if (failed(relations) || !relations->front().preservesNonUnitAxes())
      return failure();
    return extractionTileIndex(builder, location, reshape.getValue(), range,
                               sourceExtent);
  }
  if (isa_and_nonnull<block::BroadcastOp, block::TransposeOp>(
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
    Value ratio = builder.create<block::BinaryOp>(
        location, builder.getIndexType(), sourceExtent, range.getExtent(),
        BinaryOperator::FloorDivide);
    return std::pair{Value(builder.create<block::BinaryOp>(
        location, builder.getIndexType(), *index, ratio,
        BinaryOperator::Multiply)), false};
  }
  auto binary = coordinate.getDefiningOp<block::BinaryOp>();
  if (!binary || (binary.getOperatorKind() != BinaryOperator::Add &&
                  binary.getOperatorKind() != BinaryOperator::Subtract))
    return failure();
  if (binary.getOperatorKind() == BinaryOperator::Subtract &&
      binary.getLhs() == range.getResult()) {
    if (Value offset = uniformScalarFill(binary.getRhs())) {
      Value start = builder.create<block::BinaryOp>(
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
  return std::pair{Value(builder.create<block::BinaryOp>(
      location, builder.getIndexType(), lhs->first, rhs->first,
      binary.getOperatorKind())), lhs->second || rhs->second};
}

struct RectangularCoordinate {
  block::MakeRangeOp range;
  unsigned resultAxis;
  Value sourceExtent;
  Value tileIndex;
};

FailureOr<RectangularCoordinate> rectangularCoordinate(
    OpBuilder &builder, block::GatherOp gather, unsigned slot, Value coordinate,
    block::PhysicalProgramAnalysis &analysis) {
  auto access = cast<block::AccessOpInterface>(gather.getOperation());
  auto source = cast<block::FragmentType>(gather.getSource().getType());
  auto result = dyn_cast<block::FragmentType>(gather.getResult().getType());
  if (!result) return failure();
  auto ranges = analysis.sourceRanges(coordinate);
  if (ranges.roots.size() != 1 || !block::isUnitStepRange(ranges.roots.front()))
    return failure();
  auto range = ranges.roots.front();
  auto resultAxis = nativeAccessRangeAxis(access, slot, range);
  if (failed(resultAxis) || result.getShape()[*resultAxis] !=
          range.getResult().getType().getShape()[0])
    return failure();
  auto fullExtent = cast<block::PhysicalExprAttr>(source.getShape()[gather.getSourceAxes()[slot]]);
  Value full;
  if (fullExtent.getKind() == block::PhysicalExprKind::Parameter) {
    auto kernel = gather->getParentOfType<func::FuncOp>();
    auto parameter = block::queryParameterBySymbol(kernel, fullExtent.getParameterReference().getName());
    if (failed(parameter)) return failure();
    full = block::materializeParameter(builder, gather.getLoc(), parameter->getReference());
  } else {
    full = builder.create<block::PhysicalExprOp>(gather.getLoc(), builder.getIndexType(), fullExtent);
  }
  auto index = extractionTileIndex(builder, gather.getLoc(), coordinate, range, full);
  if (failed(index) || !index->second) return failure();
  return RectangularCoordinate{range, *resultAxis, full, index->first};
}

} // namespace

bool requiresFragmentStorage(block::GatherOp gather) {
  auto kernel = gather->getParentOfType<func::FuncOp>();
  if (!isa<block::FragmentType>(gather.getResult().getType())) {
    auto source = cast<block::FragmentType>(gather.getSource().getType());
    bool dynamicSelection = false;
    for (auto [coordinate, axis] :
         llvm::zip(gather.getCoordinates(), gather.getSourceAxes())) {
      if (isa<block::FragmentType>(coordinate.getType())) return false;
      if (getCompileTimeScalar(coordinate)) continue;
      auto extent = block::IndexRelations().constant(
          cast<block::PhysicalExprAttr>(source.getShape()[axis]), kernel);
      dynamicSelection |= !extent || *extent != 1;
    }
    // Dynamic scalar lookup uses an array gather from the completed snapshot.
    // Constant selections and unit axes keep the native tile extraction.
    return dynamicSelection;
  }
  block::PhysicalProgramAnalysis analysis(kernel);
  OpBuilder builder(gather);
  Operation *previous = gather->getPrevNode();
  auto discardQueryValues = llvm::make_scope_exit([&] {
    while (gather->getPrevNode() != previous) gather->getPrevNode()->erase();
  });
  for (auto [slot, value] : llvm::enumerate(gather.getCoordinates())) {
    while (isa_and_nonnull<block::SplatOp, block::BroadcastOp>(value.getDefiningOp()))
      value = value.getDefiningOp()->getOperand(0);
    if (isa<block::FragmentType>(value.getType()) && !getCompileTimeScalar(value) &&
        failed(rectangularCoordinate(builder, gather, slot, value, analysis)))
      return true;
  }
  return false;
}

LogicalResult formFragmentExtractions(func::FuncOp kernel,
    ArrayRef<block::GatherOp> gathers, NativeFormRewriter &rewriter) {
  for (block::GatherOp gather : gathers) {
    auto access = cast<block::AccessOpInterface>(gather.getOperation());
    block::PhysicalProgramAnalysis analysis(kernel);
    auto source = dyn_cast<block::FragmentType>(access.getAccessResource().getType());
    auto result = dyn_cast<block::FragmentType>(access.getAccessResult().getType());
    if (!source)
      return gather.emitOpError("cuTile tile extraction requires a source fragment");
    if (access.getAccessCoordinates().size() != access.getAccessSourceAxes().size())
      return gather.emitOpError("cuTile tile extraction source axes are incomplete");
    SmallVector<Value> coordinates(source.getShape().size());
    OpBuilder builder(gather);
    auto unit = block::PhysicalExprAttr::get(
        kernel.getContext(), block::PhysicalExprKind::Constant,
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
        if (auto splat = coordinate.getDefiningOp<block::SplatOp>())
          coordinate = splat.getValue();
        else if (auto broadcast = coordinate.getDefiningOp<block::BroadcastOp>())
          coordinate = broadcast.getValue();
        else
          break;
      }
      if (result) {
        if (isa<block::FragmentType>(coordinate.getType()) &&
            !getCompileTimeScalar(coordinate)) {
          auto selected = rectangularCoordinate(builder, gather, slot, coordinate, analysis);
          if (failed(selected))
            return gather.emitOpError(
                "cuTile coordinate is not an aligned rectangular sub-tile");
          auto range = selected->range;
          Value full = selected->sourceExtent;
          extractionShape[sourceAxis] = result.getShape()[selected->resultAxis];
          Value extractionExtent = builder.create<block::BinaryOp>(
              gather.getLoc(), builder.getIndexType(), full, range.getExtent(),
              BinaryOperator::Maximum);
          Value count = builder.create<block::BinaryOp>(
              gather.getLoc(), builder.getIndexType(), extractionExtent, range.getExtent(),
              BinaryOperator::FloorDivide);
          Value one = builder.create<arith::ConstantIndexOp>(gather.getLoc(), 1);
          Value zero = builder.create<arith::ConstantIndexOp>(gather.getLoc(), 0);
          Value last = builder.create<block::BinaryOp>(
              gather.getLoc(), builder.getIndexType(), count, one,
              BinaryOperator::Subtract);
          Value nonnegative = builder.create<block::BinaryOp>(
              gather.getLoc(), builder.getIndexType(), selected->tileIndex, zero,
              BinaryOperator::Maximum);
          // An aligned sub-tile is either entirely inside or outside the
          // physical source. Preserve the gather mask/fill while keeping the
          // native extraction address valid for inactive sub-tiles.
          coordinate = builder.create<block::BinaryOp>(
              gather.getLoc(), builder.getIndexType(), nonnegative, last,
              BinaryOperator::Minimum);
          coordinates[sourceAxis] = coordinate;
          slicedAxes[sourceAxis] = true;
          continue;
        }
        auto integer =
            dyn_cast_or_null<IntegerAttr>(getCompileTimeScalar(coordinate));
        if (!integer && !isa<block::FragmentType>(coordinate.getType())) {
          // The shared access contract bounds active members. Clamp only the
          // extraction address; the original validity still selects the fill.
          if (!coordinate.getType().isIndex())
            coordinate = builder.create<block::CastOp>(
                gather.getLoc(), builder.getIndexType(), coordinate);
          Value extent = builder.create<block::PhysicalExprOp>(
              gather.getLoc(), builder.getIndexType(),
              cast<block::PhysicalExprAttr>(source.getShape()[sourceAxis]));
          Value zero = builder.create<arith::ConstantIndexOp>(gather.getLoc(), 0);
          Value one = builder.create<arith::ConstantIndexOp>(gather.getLoc(), 1);
          Value last = builder.create<block::BinaryOp>(
              gather.getLoc(), builder.getIndexType(), extent, one,
              BinaryOperator::Subtract);
          Value nonnegative = builder.create<block::BinaryOp>(
              gather.getLoc(), builder.getIndexType(), coordinate, zero,
              BinaryOperator::Maximum);
          coordinates[sourceAxis] = builder.create<block::BinaryOp>(
              gather.getLoc(), builder.getIndexType(), nonnegative, last,
              BinaryOperator::Minimum);
          continue;
        }
        auto extent = block::IndexRelations().constant(
            cast<block::PhysicalExprAttr>(source.getShape()[sourceAxis]), kernel);
        if (!integer || !extent || integer.getInt() < 0 || integer.getInt() >= *extent)
          return gather.emitOpError(
              "cuTile rectangular tile extraction requires in-bounds constant selected coordinates");
        coordinate = builder.create<arith::ConstantIntOp>(
            gather.getLoc(), integer.getInt(), 32);
      } else {
        if (isa<block::FragmentType>(coordinate.getType()))
          return gather.emitOpError("cuTile scalar extraction requires scalar coordinates");
        if (access.getAccessValidity()) {
          Value zero = builder.create<arith::ConstantOp>(
              gather.getLoc(), coordinate.getType(),
              builder.getIntegerAttr(coordinate.getType(), 0));
          coordinate = builder.create<block::SelectOp>(
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
        coordinate = builder.create<block::CastOp>(
            gather.getLoc(), builder.getI32Type(), coordinate);
    Value tile = access.getAccessResource();
    for (unsigned axis = 0; axis < slicedAxes.size(); ++axis)
      if (slicedAxes[axis])
        tile = repeatTileAxis(
            builder, gather.getLoc(), tile, axis,
            cast<block::PhysicalExprAttr>(extractionShape[axis]));
    Type extractedType = access.getAccessResult().getType();
    if (result) {
      SmallVector<Attribute> shape, maps;
      auto tileType = cast<block::FragmentType>(tile.getType());
      for (auto [resultAxis, sourceAxis] : llvm::enumerate(retainedAxes)) {
        shape.push_back(extractionShape[sourceAxis]);
        auto map = cast<block::AxisMapAttr>(tileType.getAxisMaps()[sourceAxis]);
        maps.push_back(block::AxisMapAttr::get(
            kernel.getContext(), map.getSourceId(), map.getSourceAxis(),
            map.getDimensionId(), resultAxis, map.getDerived()));
      }
      extractedType = block::FragmentType::get(
          kernel.getContext(), result.getElementType(), builder.getArrayAttr(shape),
          builder.getArrayAttr(maps), result.getValidity(), result.getOwner());
      if (builder.getArrayAttr(shape) != result.getShape() &&
          !block::queryBroadcastProjection(
               cast<block::FragmentType>(extractedType), result).isExact())
        return gather.emitOpError(
            "cuTile extraction requires an exact retained-axis broadcast");
    }
    auto replacement = builder.create<ExtractOp>(
        gather.getLoc(), extractedType, tile,
        coordinates, builder.getArrayAttr(extractionShape),
        builder.getDenseI64ArrayAttr(retainedAxes));
    if (Attribute origin = gather->getAttr(block::originAttr))
      replacement->setAttr(block::originAttr, origin);
    Value value = replacement.getResult();
    if (extractedType != access.getAccessResult().getType()) {
      auto extracted = cast<block::FragmentType>(extractedType);
      if (block::queryBroadcastProjection(extracted, result).isExact()) {
        value = builder.create<block::BroadcastOp>(gather.getLoc(), result, value);
      } else {
        auto reassociation = block::inferReshapeReassociation(extracted, result);
        if (failed(reassociation))
          return gather.emitOpError(
              "cuTile extraction has no row-major result domain projection");
        value = builder.create<block::ReshapeOp>(
            gather.getLoc(), result, value, *reassociation);
      }
    }
    if (access.getAccessValidity()) {
      auto selected = builder.create<block::SelectOp>(
          gather.getLoc(), access.getAccessResult().getType(), access.getAccessValidity(),
          value, access.getAccessFill());
      if (Attribute origin = gather->getAttr(block::originAttr))
        selected->setAttr(block::originAttr, origin);
      value = selected.getResult();
    }
    rewriter.replace(gather, ValueRange{value});
  }
  return success();
}

} // namespace intent::cutile
