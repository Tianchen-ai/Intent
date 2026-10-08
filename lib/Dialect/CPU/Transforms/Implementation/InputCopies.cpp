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
  for (unsigned axis = 0; axis < 2; ++axis)
    if (auto bound = constantDimensionUpperBound(source, axis);
        bound && *bound < transposeWidth) return false;
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

  if (contiguous.condition) {
    auto choice = builder.create<scf::IfOp>(loc, contiguous.condition, true);
    builder.setInsertionPointToStart(choice.thenBlock());
    if (failed(transpose())) return failure();
    builder.setInsertionPointToStart(choice.elseBlock());
    scalar(zero, rows, zero, columns);
  } else if (failed(transpose())) return failure();
  operation.erase();
  return true;
}

} // namespace intent::cpu
