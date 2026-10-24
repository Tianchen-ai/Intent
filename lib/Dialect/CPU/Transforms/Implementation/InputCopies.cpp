#include "InputCopies.h"
#include "../Vector/ContiguousMemory.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Intent/Dialect/CPU/Transforms/Structure/LoopBuilders.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace intent::cpu {

linalg::GenericOp createInputCopy(OpBuilder &builder, Location location,
    Value source, Value destination, unsigned innerAxis) {
  auto sourceType = cast<MemRefType>(source.getType());
  unsigned rank = sourceType.getRank();
  if (innerAxis + 1 != rank) {
    SmallVector<AffineExpr> order;
    for (unsigned axis = 0; axis < rank; ++axis)
      if (axis != innerAxis)
        order.push_back(builder.getAffineDimExpr(axis));
    order.push_back(builder.getAffineDimExpr(innerAxis));
    source = builder.create<memref::TransposeOp>(location, source,
        AffineMapAttr::get(AffineMap::get(rank, 0, order, builder.getContext())));
  }
  auto identity = builder.getMultiDimIdentityMap(rank);
  Type element = cast<MemRefType>(destination.getType()).getElementType();
  return builder.create<linalg::GenericOp>(location, ValueRange{source},
      ValueRange{destination}, ArrayRef<AffineMap>{identity, identity},
      SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel),
      [&](OpBuilder &body, Location at, ValueRange arguments) {
        Value value = arguments[0];
        if (value.getType() != element)
          value = body.create<arith::ExtFOp>(at, element, value);
        body.create<linalg::YieldOp>(at, value);
      });
}

namespace {

// Only the complete square uses the standard 16-lane shuffle network. Tails
// keep scalar accesses; no padded source element is introduced by preparation.
constexpr int64_t transposeWidth = 16;

bool supportedElement(Type type) {
  return isa<FloatType, IntegerType>(type) && !type.isInteger(1);
}

Value convert(OpBuilder &builder, Location location, Value value,
              arith::ExtFOp extension, int64_t lanes = 1) {
  if (!extension) return value;
  Type result = extension.getType();
  if (lanes != 1) result = VectorType::get({lanes}, result);
  auto widened = builder.create<arith::ExtFOp>(location, result, value);
  widened->setAttrs(extension->getAttrs());
  return widened;
}

LogicalResult lowerTranspose(scf::ForOp tiles) {
  RewritePatternSet patterns(tiles.getContext());
  vector::VectorTransformsOptions options;
  options.setVectorTransposeLowering(vector::VectorTransposeLowering::Shuffle16x16);
  vector::populateVectorTransposeLoweringPatterns(patterns, options);
  vector::ExtractOp::getCanonicalizationPatterns(patterns, tiles.getContext());
  vector::InsertOp::getCanonicalizationPatterns(patterns, tiles.getContext());
  vector::ShapeCastOp::getCanonicalizationPatterns(patterns, tiles.getContext());
  SmallVector<Operation *> operations;
  tiles.walk([&](Operation *operation) {
    if (operation != tiles.getOperation()) operations.push_back(operation);
  });
  GreedyRewriteConfig config;
  config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;
  config.scope = &tiles.getRegion();
  if (failed(applyOpPatternsGreedily(operations, std::move(patterns), config)))
    return tiles.emitError("prepared input transpose normalization did not converge");
  bool closed = true;
  tiles.walk([&](Operation *operation) {
    if (isa<vector::TransposeOp, vector::ShapeCastOp,
            vector::InsertOp, vector::ExtractOp>(operation)) closed = false;
    for (Type type : operation->getResultTypes())
      if (auto vector = dyn_cast<VectorType>(type); vector && vector.getRank() != 1)
        closed = false;
  });
  if (!closed)
    return tiles.emitError(
        "prepared input transpose must lower completely to rank-one vectors");
  return success();
}

} // namespace

FailureOr<bool> tryMaterializeInputCopy(linalg::GenericOp operation,
    int64_t width, OpBuilder::Listener *listener) {
  if (width < transposeWidth || operation.getNumResults() ||
      operation.getNumLoops() != 2 || operation.getNumReductionLoops() ||
      operation.getInputs().size() != 1 || operation.getOutputs().size() != 1)
    return false;
  Value source = operation.getInputs()[0], target = operation.getOutputs()[0];
  auto inputType = dyn_cast<MemRefType>(source.getType());
  auto outputType = dyn_cast<MemRefType>(target.getType());
  if (!inputType || !outputType || inputType.getRank() != 2 ||
      outputType.getRank() != 2 || !supportedElement(inputType.getElementType()) ||
      !supportedElement(outputType.getElementType())) return false;
  for (AffineMap map : operation.getIndexingMapsArray())
    if (!map.isIdentity()) return false;
  Block &body = operation.getRegion().front();
  if (!body.getArguments().back().use_empty()) return false;
  Value yielded = body.getTerminator()->getOperand(0);
  arith::ExtFOp extension;
  if (yielded != body.getArgument(0)) {
    extension = yielded.getDefiningOp<arith::ExtFOp>();
    if (!extension || extension.getIn() != body.getArgument(0)) return false;
  }
  for (Operation &nested : body.without_terminator())
    if (nested.getNumRegions() || !isMemoryEffectFree(&nested) ||
        !isSpeculatable(&nested)) return false;
  SmallVector<int64_t> inputStrides, outputStrides;
  int64_t inputOffset, outputOffset;
  if (failed(inputType.getStridesAndOffset(inputStrides, inputOffset)) ||
      failed(outputType.getStridesAndOffset(outputStrides, outputOffset)) ||
      (!ShapedType::isDynamic(inputStrides[0]) && inputStrides[0] != 1) ||
      (!ShapedType::isDynamic(outputStrides[1]) && outputStrides[1] != 1))
    return false;
  auto columnBound = constantDimensionUpperBound(source, 1);
  Type element = inputType.getElementType();
  int64_t bits = element.getIntOrFloatBitWidth();
  int64_t groupSize = bits == 8 ? 4 : bits == 16 ? 2 : 0;
  bool wordGroup = !extension && groupSize &&
      outputType.getElementType() == element && outputStrides[0] == groupSize &&
      outputStrides[1] == 1 && columnBound && *columnBound <= groupSize;
  for (unsigned axis = 0; axis < 2; ++axis)
    if (auto bound = constantDimensionUpperBound(source, axis);
        bound && *bound < transposeWidth && !(wordGroup && axis == 1)) return false;
  // Prove reordering at the complete, attached read point, before constructing
  // any new control regions or descriptors used by this copy.
  StorageAnalysis storage(operation->getParentOfType<func::FuncOp>());
  if (!storage.disjoint(source, target)) return false;

  OpBuilder builder(operation, listener);
  Location loc = operation.getLoc();
  Value zero = index(builder, loc, 0);
  Value rows = builder.createOrFold<memref::DimOp>(loc, source, 0);
  Value columns = builder.createOrFold<memref::DimOp>(loc, source, 1);
  auto swapped = builder.create<memref::TransposeOp>(loc, source,
      AffineMapAttr::get(AffineMap::getPermutationMap(
          ArrayRef<int64_t>{1, 0}, builder.getContext())));
  auto contiguous = materializeContiguousMemoryGuard(builder, loc,
      ValueRange{swapped.getResult(), target});
  // Tiling also reorders writes between rows. Unit lanes alone do not prove
  // disjoint rows for an arbitrary strided descriptor.
  Value rowStride = ShapedType::isDynamic(outputStrides[0])
      ? Value(builder.create<memref::ExtractStridedMetadataOp>(loc, target).getStrides()[0])
      : index(builder, loc, outputStrides[0]);
  Value separateRows = builder.create<arith::CmpIOp>(loc,
      arith::CmpIPredicate::sge, rowStride, columns);
  contiguous.condition = contiguous.condition
      ? Value(builder.create<arith::AndIOp>(loc, contiguous.condition, separateRows))
      : separateRows;

  auto scalar = [&](Value rowBegin, Value rowEnd, Value columnBegin, Value columnEnd) {
    loop(builder, loc, rowBegin, rowEnd, 1, [&](Value row) {
      loop(builder, loc, columnBegin, columnEnd, 1, [&](Value column) {
        Value value = builder.create<memref::LoadOp>(loc, source, ValueRange{row, column});
        value = convert(builder, loc, value, extension);
        builder.create<memref::StoreOp>(loc, value, target, ValueRange{row, column});
      });
    });
  };
  if (wordGroup) {
    Value group = index(builder, loc, groupSize);
    Value complete = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, columns, group);
    if (contiguous.condition) complete = builder.create<arith::AndIOp>(loc, complete, contiguous.condition);
    auto choice = builder.create<scf::IfOp>(loc, complete, true);
    builder.setInsertionPointToStart(choice.thenBlock());
    IRMapping mapping;
    contiguous.bind(builder, loc, mapping);
    Value input = mapping.lookupOrDefault(swapped.getResult());
    // This branch proves four-byte rows. Flatten only their destination so a
    // vector store spans complete rows without crossing a logical row bound.
    auto metadata = builder.create<memref::ExtractStridedMetadataOp>(loc, target);
    auto flatType = MemRefType::get({ShapedType::kDynamic}, outputType.getElementType(),
        StridedLayoutAttr::get(builder.getContext(), ShapedType::kDynamic, {1}),
        outputType.getMemorySpace());
    Value size = builder.create<arith::MulIOp>(loc, rows, group);
    Value flat = builder.create<memref::ReinterpretCastOp>(loc, flatType,
        metadata.getBaseBuffer(), OpFoldResult(metadata.getOffset()),
        ArrayRef<OpFoldResult>{size}, ArrayRef<OpFoldResult>{builder.getIndexAttr(1)});
    Value full = builder.create<arith::SubIOp>(loc, rows,
        builder.create<arith::RemSIOp>(loc, rows, index(builder, loc, 16)));
    SmallVector<int64_t> pairMask, groupMask;
    for (int64_t lane = 0; lane < 16; ++lane) {
      pairMask.append({lane, lane + 16});
      groupMask.append({2 * lane, 2 * lane + 1, 2 * lane + 32, 2 * lane + 33});
    }
    auto lanes = VectorType::get({16}, inputType.getElementType());
    loop(builder, loc, zero, full, 16, [&](Value row) {
      SmallVector<Value> values;
      for (int64_t column = 0; column < groupSize; ++column)
        values.push_back(builder.create<vector::LoadOp>(loc, lanes, input,
            ValueRange{index(builder, loc, column), row}));
      Value packed = builder.create<vector::ShuffleOp>(loc, values[0], values[1], pairMask);
      if (groupSize == 4) {
        Value high = builder.create<vector::ShuffleOp>(loc, values[2], values[3], pairMask);
        packed = builder.create<vector::ShuffleOp>(loc, packed, high, groupMask);
      }
      builder.create<vector::StoreOp>(loc, packed, flat,
          ValueRange{builder.create<arith::MulIOp>(loc, row, group)});
    });
    scalar(full, rows, zero, columns);
    builder.setInsertionPointToStart(choice.elseBlock());
    scalar(zero, rows, zero, columns);
    operation.erase();
    return true;
  }
  auto transpose = [&]() -> LogicalResult {
    IRMapping mapping;
    contiguous.bind(builder, loc, mapping);
    Value input = mapping.lookupOrDefault(swapped.getResult());
    Value output = mapping.lookupOrDefault(target);
    Value step = index(builder, loc, transposeWidth);
    Value fullRows = builder.create<arith::SubIOp>(loc, rows,
        builder.create<arith::RemSIOp>(loc, rows, step));
    Value fullColumns = builder.create<arith::SubIOp>(loc, columns,
        builder.create<arith::RemSIOp>(loc, columns, step));
    auto rowTiles = loop(builder, loc, zero, fullRows, transposeWidth, [&](Value row) {
      loop(builder, loc, zero, fullColumns, transposeWidth, [&](Value column) {
        Type element = inputType.getElementType();
        auto laneType = VectorType::get({transposeWidth}, element);
        auto tileType = VectorType::get({transposeWidth, transposeWidth}, element);
        Value tile = builder.create<arith::ConstantOp>(loc, tileType,
            builder.getZeroAttr(tileType));
        for (int64_t lane = 0; lane < transposeWidth; ++lane) {
          Value at = add(builder, loc, column, index(builder, loc, lane));
          Value loaded = builder.create<vector::LoadOp>(loc, laneType, input,
              ValueRange{at, row});
          tile = builder.create<vector::InsertOp>(loc, loaded, tile,
              ArrayRef<int64_t>{lane});
        }
        Value transposed = builder.create<vector::TransposeOp>(loc, tile,
            ArrayRef<int64_t>{1, 0});
        for (int64_t lane = 0; lane < transposeWidth; ++lane) {
          Value value = builder.create<vector::ExtractOp>(loc, transposed,
              ArrayRef<int64_t>{lane});
          value = convert(builder, loc, value, extension, transposeWidth);
          Value at = add(builder, loc, row, index(builder, loc, lane));
          builder.create<vector::StoreOp>(loc, value, output, ValueRange{at, column});
        }
      });
    });
    if (failed(lowerTranspose(rowTiles))) return failure();
    scalar(zero, fullRows, fullColumns, columns);
    scalar(fullRows, rows, zero, columns);
    return success();
  };

  auto transposedOrStrided = [&]() -> LogicalResult {
    if (contiguous.condition) {
      auto choice = builder.create<scf::IfOp>(loc, contiguous.condition, true);
      builder.setInsertionPointToStart(choice.thenBlock());
      if (failed(transpose())) return failure();
      builder.setInsertionPointToStart(choice.elseBlock());
      scalar(zero, rows, zero, columns);
    } else if (failed(transpose())) return failure();
    return success();
  };
  // An unknown leading stride also matches the transpose case above. Do not
  // let that specialization turn ordinary row-contiguous input into scalar
  // packing. Both versions consume the same copy and its declared conversion.
  if (inputStrides[1] == 1 || ShapedType::isDynamic(inputStrides[1])) {
    auto rowContiguous = materializeContiguousMemoryGuard(builder, loc,
                                                        ValueRange{source, target});
    auto copyRows = [&] {
      IRMapping mapping;
      rowContiguous.bind(builder, loc, mapping);
      Value input = mapping.lookupOrDefault(source);
      Value output = mapping.lookupOrDefault(target);
      auto lanes = VectorType::get({width}, inputType.getElementType());
      Value fullColumns = builder.create<arith::SubIOp>(loc, columns,
          builder.create<arith::RemSIOp>(loc, columns, index(builder, loc, width)));
      loop(builder, loc, zero, rows, 1, [&](Value row) {
        loop(builder, loc, zero, fullColumns, width, [&](Value column) {
          Value value = builder.create<vector::LoadOp>(loc, lanes, input, ValueRange{row, column});
          value = convert(builder, loc, value, extension, width);
          builder.create<vector::StoreOp>(loc, value, output, ValueRange{row, column});
        });
        // Finish each row before writing the next, including overlapping rows.
        loop(builder, loc, fullColumns, columns, 1, [&](Value column) {
          Value value = builder.create<memref::LoadOp>(loc, source, ValueRange{row, column});
          value = convert(builder, loc, value, extension);
          builder.create<memref::StoreOp>(loc, value, target, ValueRange{row, column});
        });
      });
    };
    if (rowContiguous.condition) {
      auto choice = builder.create<scf::IfOp>(loc, rowContiguous.condition, true);
      builder.setInsertionPointToStart(choice.thenBlock());
      copyRows();
      builder.setInsertionPointToStart(choice.elseBlock());
      if (failed(transposedOrStrided())) return failure();
    } else copyRows();
  } else if (failed(transposedOrStrided())) return failure();
  operation.erase();
  return true;
}

} // namespace intent::cpu
