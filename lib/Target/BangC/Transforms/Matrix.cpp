#include "PassDetail.h"
#include "Intent/Dialect/DSA/Transforms/LocalSupplyRelations.h"
#include "Intent/Dialect/DSA/Transforms/MatrixPanels.h"

using namespace mlir;
namespace intent::bangc {
bool paddedInput(Operation *consumer, Value input, Value rows, Value columns) {
  dsa::StorageAnalysis storage(consumer->getParentOfType<func::FuncOp>());
  Value origin = storage.uniqueOrigin(input);
  if (!origin || !dsa::isCompleteStorageViewOf(input, origin)) return false;
  auto matches = [&](Value value) {
    return value.getType() == input.getType() && dsa::isCompleteStorageViewOf(value, origin);
  };
  auto sameCount = [](Value lhs, Value rhs) {
    if (lhs == rhs) return true;
    APInt left, right;
    return matchPattern(lhs, m_ConstantInt(&left)) && matchPattern(rhs, m_ConstantInt(&right)) && left == right;
  };
  if (Operation *previous = storage.lastWriterBefore(input, consumer)) {
    if (auto load = dyn_cast<dsa::LoadTileOp>(previous); load && matches(load.getOutput()))
      return sameCount(load.getRows(), rows) && sameCount(load.getColumns(), columns);
    if (auto gather = dyn_cast<dsa::GatherRowsOp>(previous); gather && matches(gather.getOutput()))
      return sameCount(gather.getRows(), rows) && sameCount(gather.getColumns(), columns);
    if (auto gather = dyn_cast<dsa::GroupGatherRowsOp>(previous); gather && matches(gather.getOutput())) {
      APInt count;
      return sameCount(gather.getRows(), rows) && matchPattern(columns, m_ConstantInt(&count)) &&
          count.getSExtValue() == cast<MemRefType>(input.getType()).getDimSize(1);
    }
    if (auto conversion = dyn_cast<dsa::CastOp>(previous); conversion && matches(conversion.getOutput())) {
      auto source = cast<MemRefType>(conversion.getInput().getType());
      auto destination = cast<MemRefType>(input.getType());
      // Element conversion preserves the zero padding introduced by LoadTile.
      // Prove it before the conversion, independently of later source reuse.
      return storage.disjoint(conversion.getInput(), input) && source.getShape() == destination.getShape() &&
          paddedInput(previous, conversion.getInput(), rows, columns);
    }
  }
  return false;
}

bool consumesMatrixRhs(dsa::MatMulOp matrix) {
  Value input = matrix.getRhs();
  dsa::StorageAnalysis storage(matrix->getParentOfType<func::FuncOp>());
  Value origin = storage.uniqueOrigin(input);
  return origin && origin.getDefiningOp<memref::AllocaOp>() &&
      dsa::isCompleteStorageViewOf(input, origin) &&
      storage.disjoint(input, matrix.getLhs()) &&
      storage.disjoint(input, matrix.getAccumulator()) &&
      storage.allUsesCompleteBefore(origin, matrix, matrix);
}

namespace {
struct MatrixInputSupply {
  bool direct = false, zeroPadded = false;
};
struct MatrixSupply {
  MatrixInputSupply lhs, rhs;
  bool accumulator = false, consumeRhs = false, nonempty = false;
};

bool independentAccumulator(dsa::MatMulOp matrix) {
  dsa::StorageAnalysis storage(matrix->getParentOfType<func::FuncOp>());
  return storage.disjoint(matrix.getAccumulator(), matrix.getLhs()) &&
      storage.disjoint(matrix.getAccumulator(), matrix.getRhs());
}

MatrixSupply matrixSupply(dsa::MatMulOp matrix, int64_t m, int64_t n, int64_t k) {
  auto function = matrix->getParentOfType<func::FuncOp>();
  // Staging an accumulator also protects overlapping input snapshots. Removing
  // that staging requires actual storage independence, not distinct SSA names.
  if (!independentAccumulator(matrix)) return {};
  dsa::LocalSupplyRelations relations(function);
  // Native execution is enclosed by positive For bounds or the active guard.
  // At that execution point an upper bound suffices to prove a single tile;
  // nonpositive counts outside it execute no matrix or preparation operation.
  auto fitsOneTile = [&](Value extent, int64_t capacity) {
    if (relations.equal(extent, capacity)) return true;
    auto range = relations.interval(extent);
    return range && range->second <= capacity;
  };
  auto positive = [&](Value extent, int64_t capacity) {
    if (capacity > 0 && relations.equal(extent, capacity)) return true;
    auto range = relations.interval(extent);
    return range && range->first > 0;
  };
  auto available = [&](Value input, Value rows, Value columns,
                       int64_t r, int64_t c) -> MatrixInputSupply {
    auto type = cast<MemRefType>(input.getType());
    if (!type.getLayout().isIdentity() ||
        type.getShape() != ArrayRef<int64_t>({r, c}) ||
        !fitsOneTile(rows, r) || !fitsOneTile(columns, c)) return {};
    // A complete actual tile has no padding obligation, regardless of which
    // local producer defined it. Partial tiles retain the existing zero proof.
    bool padded = paddedInput(matrix, input, rows, columns);
    return {(relations.equal(rows, r) && relations.equal(columns, c)) || padded, padded};
  };
  bool transposed = matrix.getRhsTransposed();
  MatrixSupply supply;
  supply.nonempty = positive(matrix.getRows(), m) &&
      positive(matrix.getColumns(), n) && positive(matrix.getDepth(), k);
  supply.lhs = available(matrix.getLhs(), matrix.getRows(), matrix.getDepth(), m, k);
  supply.rhs = available(matrix.getRhs(),
      transposed ? matrix.getColumns() : matrix.getDepth(),
      transposed ? matrix.getDepth() : matrix.getColumns(),
      transposed ? n : k, transposed ? k : n);
  auto accumulator = cast<MemRefType>(matrix.getAccumulator().getType());
  supply.accumulator = accumulator.getLayout().isIdentity() &&
      accumulator.getShape() == ArrayRef<int64_t>({m, n}) &&
      relations.equal(matrix.getRows(), m) && relations.equal(matrix.getColumns(), n);
  // A whole MatMul may read its RHS in several M panels. Its last external use
  // does not make that source disposable after the first internal preparation.
  supply.consumeRhs = supply.rhs.direct && fitsOneTile(matrix.getRows(), m) &&
      fitsOneTile(matrix.getColumns(), n) && fitsOneTile(matrix.getDepth(), k) &&
      consumesMatrixRhs(matrix);
  return supply;
}
} // namespace

Value matrixReshapeWorkspace(OpBuilder &b, Location loc, Value input, Value transposed, bool consumeInput) {
  auto type = cast<MemRefType>(transposed.getType());
  if (type.getDimSize(0) == 64) return {};
  if (!consumeInput) return allocate(b, loc, type.getElementType(), type.getShape(), dsa::nramSpace);
  // Packing has already copied the row-major input into an independent
  // transposed tile. Its dead storage can now hold the reshaped filter.
  return b.create<memref::ReinterpretCastOp>(loc, type, input, int64_t(0),
      type.getShape(), ArrayRef<int64_t>{type.getDimSize(1), 1});
}

LogicalResult realizeMatMul(dsa::MatMulOp matrix, dsa::ConfigurationAttr config) {
  Location loc = matrix.getLoc();
  OpBuilder b(matrix);
  auto lhsType = cast<MemRefType>(matrix.getLhs().getType());
  auto rhsType = cast<MemRefType>(matrix.getRhs().getType());
  auto accType = cast<MemRefType>(matrix.getAccumulator().getType());
  bool rhsTransposed = matrix.getRhsTransposed();
  Type element = lhsType.getElementType();
  if (!element.isF16() && !element.isBF16() && !element.isF32())
    return matrix.emitError("MLU370 matrix profile requires f16/bf16/f32 input tiles");
  int64_t bm = config.getTileM(), bn = config.getTileN(), bk = config.getTileK();
  if (bn % 64)
    return matrix.emitError("MLU370 matrix N block must be a multiple of 64");
  if ((bk * element.getIntOrFloatBitWidth() / 8) % 64)
    return matrix.emitError("MLU370 matrix K block must be aligned to 64 bytes");
  bm = std::min(bm, lhsType.getDimSize(0));
  bn = std::min<int64_t>(bn, llvm::alignTo(rhsType.getDimSize(rhsTransposed ? 0 : 1), int64_t(64)));
  bk = std::min<int64_t>(bk, llvm::alignTo(lhsType.getDimSize(1), int64_t(512 / element.getIntOrFloatBitWidth())));
  auto index = [&](int64_t value) -> Value { return b.create<arith::ConstantIndexOp>(loc, value); };
  auto offset = [&](Value row, Value column, int64_t stride) -> Value {
    return b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, row, index(stride)), column);
  };
  auto count = [&](Value extent, Value begin, int64_t tile) -> Value {
    return b.create<arith::MinSIOp>(loc, b.create<arith::SubIOp>(loc, extent, begin), index(tile));
  };
  auto function = matrix->getParentOfType<func::FuncOp>();
  auto full = [&](Value value, int64_t expected) {
    auto interval = integerInterval(value, function);
    return interval && interval->first == expected && interval->second == expected;
  };
  // A full narrow output can use C^T += B^T A^T. The original M=64 axis
  // then fills the matrix engine's N lanes, and A already supplies the packed
  // transposed RHS form. Preserve the initial accumulator through a transpose.
  if (!rhsTransposed && lhsType.getDimSize(0) == 64 && rhsType.getDimSize(1) < 64 &&
      rhsType.getDimSize(1) <= config.getTileM() && lhsType.getDimSize(1) == bk &&
      lhsType.getLayout().isIdentity() && rhsType.getLayout().isIdentity() &&
      accType.getLayout().isIdentity() &&
      full(matrix.getRows(), 64) && full(matrix.getColumns(), rhsType.getDimSize(1)) &&
      full(matrix.getDepth(), lhsType.getDimSize(1)) &&
      independentAccumulator(matrix)) {
    int64_t n = rhsType.getDimSize(1), k = lhsType.getDimSize(1);
    Operation *previous = matrix->getPrevNode();
    Value left = allocate(b, loc, element, {n, k}, dsa::nramSpace);
    Value accumulator = allocate(b, loc, b.getF32Type(), {n, 64}, dsa::nramSpace);
    Value packed = allocate(b, loc, element, {k, 64}, dsa::matrixSpace);
    b.create<dsa::TransposeOp>(loc, matrix.getRhs(), left, index(k), index(n));
    b.create<dsa::TransposeOp>(loc, matrix.getAccumulator(), accumulator, index(64), index(n));
    b.create<dsa::PrepareMatrixOp>(loc, matrix.getLhs(), packed, Value{}, Value{}, b.getBoolAttr(true));
    b.create<dsa::MatrixTileOp>(loc, left, packed, accumulator);
    b.create<dsa::TransposeOp>(loc, accumulator, matrix.getAccumulator(), index(n), index(64));
    int64_t nram = 0, wram = 0; measureStorage(function, nram, wram);
    if (nram <= config.getLocalBytes() && nram <= 768 * 1024 && wram <= 1024 * 1024) {
      matrix.erase(); return success();
    }
    while (matrix->getPrevNode() != previous) matrix->getPrevNode()->erase();
  }
  auto tryDirect = [&](int64_t bm, int64_t bn, int64_t panelDepth) {
    b.setInsertionPoint(matrix);
    MatrixSupply supply = matrixSupply(matrix, bm, bn, panelDepth);
    if (!(dsa::matrixPanelStorage(element, bm, bn, panelDepth) &&
      lhsType.getShape() == ArrayRef<int64_t>({bm, panelDepth}) &&
      rhsType.getShape() == (rhsTransposed ? ArrayRef<int64_t>({bn, panelDepth}) : ArrayRef<int64_t>({panelDepth, bn})) &&
      accType.getShape() == ArrayRef<int64_t>({bm, bn}) &&
      accType.getLayout().isIdentity() &&
      matrix.getAccumulator().getDefiningOp<memref::AllocaOp>() &&
      supply.lhs.direct && supply.rhs.direct &&
      (supply.accumulator || (supply.lhs.zeroPadded && supply.rhs.zeroPadded))))
      return false;
    Operation *previous = matrix->getPrevNode();
    if (!supply.nonempty) {
      Value active = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, matrix.getDepth(), index(0));
      active = b.create<arith::AndIOp>(loc, active,
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, matrix.getRows(), index(0)));
      active = b.create<arith::AndIOp>(loc, active,
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, matrix.getColumns(), index(0)));
      auto guard = b.create<scf::IfOp>(loc, active, false);
      b.setInsertionPointToStart(&guard.getThenRegion().front());
    }
    Value transpose = rhsTransposed ? Value{} : allocate(b, loc, element, {bn, panelDepth}, dsa::nramSpace);
    Value reshaped = rhsTransposed ? (bn == 64 ? Value{} : allocate(b, loc, element, {bn, panelDepth}, dsa::nramSpace))
                                  : matrixReshapeWorkspace(b, loc, matrix.getRhs(), transpose, supply.consumeRhs);
    Value packed = allocate(b, loc, element, {panelDepth, bn}, dsa::matrixSpace);
    b.create<dsa::PrepareMatrixOp>(loc, matrix.getRhs(), packed, transpose, reshaped, b.getBoolAttr(rhsTransposed));
    b.create<dsa::MatrixTileOp>(loc, matrix.getLhs(), packed, matrix.getAccumulator());
    if (storageFitsBudget(function, config, measureStorage(function))) {
      matrix.erase();
      return true;
    }
    // The typed panel remains valid, but its full native preparation does not
    // fit beside current live storage. Retain the original bounded tiling.
    b.setInsertionPoint(matrix);
    while (matrix->getPrevNode() != previous) matrix->getPrevNode()->erase();
    return false;
  };
  auto emitTiled = [&](int64_t bm, int64_t bn, const MatrixSupply &supply) {
    OpBuilder::InsertionGuard insertion(b);
    b.setInsertionPoint(matrix);
    auto rows = b.create<scf::ForOp>(loc, index(0), matrix.getRows(), index(bm));
    b.setInsertionPointToStart(rows.getBody());
    Value m = rows.getInductionVar(), mCount = count(matrix.getRows(), m, bm);
    auto columns = b.create<scf::ForOp>(loc, index(0), matrix.getColumns(), index(bn));
    b.setInsertionPointToStart(columns.getBody());
    Value n = columns.getInductionVar(), nCount = count(matrix.getColumns(), n, bn);
    Value lhs = supply.lhs.direct ? matrix.getLhs() : allocate(b, loc, element, {bm, bk}, dsa::nramSpace);
    Value rhs = supply.rhs.direct ? matrix.getRhs() : allocate(b, loc, element,
        rhsTransposed ? ArrayRef<int64_t>({bn, bk}) : ArrayRef<int64_t>({bk, bn}), dsa::nramSpace);
    Value accumulator = supply.accumulator ? matrix.getAccumulator() :
        allocate(b, loc, b.getF32Type(), {bm, bn}, dsa::nramSpace);
    Value transpose = rhsTransposed ? Value{} : allocate(b, loc, element, {bn, bk}, dsa::nramSpace);
    Value reshaped = rhsTransposed ? (bn == 64 ? Value{} : allocate(b, loc, element, {bn, bk}, dsa::nramSpace))
                                  : matrixReshapeWorkspace(b, loc, rhs, transpose,
                                                           !supply.rhs.direct || supply.consumeRhs);
    Value packed = allocate(b, loc, element, {bk, bn}, dsa::matrixSpace);
    if (!supply.accumulator)
      b.create<dsa::LoadTileOp>(loc, matrix.getAccumulator(), accumulator,
          offset(m, n, accType.getDimSize(1)), index(accType.getDimSize(1)), index(1), mCount, nCount);
    auto reduction = b.create<scf::ForOp>(loc, index(0), matrix.getDepth(), index(bk));
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(reduction.getBody());
      Value k = reduction.getInductionVar(), kCount = count(matrix.getDepth(), k, bk);
      if (!supply.lhs.direct)
        b.create<dsa::LoadTileOp>(loc, matrix.getLhs(), lhs, offset(m, k, lhsType.getDimSize(1)),
            index(lhsType.getDimSize(1)), index(1), mCount, kCount);
      if (!supply.rhs.direct)
        b.create<dsa::LoadTileOp>(loc, matrix.getRhs(), rhs,
            offset(rhsTransposed ? n : k, rhsTransposed ? k : n, rhsType.getDimSize(1)),
            index(rhsType.getDimSize(1)), index(1), rhsTransposed ? nCount : kCount, rhsTransposed ? kCount : nCount);
      b.create<dsa::PrepareMatrixOp>(loc, rhs, packed, transpose, reshaped, b.getBoolAttr(rhsTransposed));
      b.create<dsa::MatrixTileOp>(loc, lhs, packed, accumulator);
    }
    if (!supply.accumulator)
      b.create<dsa::StoreTileOp>(loc, accumulator, matrix.getAccumulator(), offset(m, n, accType.getDimSize(1)),
          index(accType.getDimSize(1)), index(1), mCount, nCount);
  };
  SmallVector<dsa::MatrixPanelShape> candidates;
  if (lhsType.getLayout().isIdentity() && rhsType.getLayout().isIdentity() &&
      accType.getLayout().isIdentity()) {
    dsa::StorageAnalysis storage(function);
    if (storage.disjoint(matrix.getAccumulator(), matrix.getLhs()) &&
        storage.disjoint(matrix.getAccumulator(), matrix.getRhs()))
      candidates = dsa::largerMatrixPanels(config, element,
          lhsType.getDimSize(0), rhsType.getDimSize(rhsTransposed ? 0 : 1),
          bk, {bm, bn});
  }
  for (auto shape : candidates) {
    // Coarsen only independent output rectangles. A new free-axis choice does
    // not combine the original K updates or change their accumulation order.
    if (tryDirect(shape.rows, shape.columns, bk)) return success();
    Operation *previous = matrix->getPrevNode();
    emitTiled(shape.rows, shape.columns, matrixSupply(matrix, shape.rows, shape.columns, bk));
    if (storageFitsBudget(function, config, measureStorage(function))) {
      matrix.erase();
      return success();
    }
    b.setInsertionPoint(matrix);
    while (matrix->getPrevNode() != previous) matrix->getPrevNode()->erase();
  }
  if (tryDirect(bm, bn, lhsType.getDimSize(1))) return success();
  MatrixSupply supply = matrixSupply(matrix, bm, bn, bk);
  Operation *previous = matrix->getPrevNode();
  emitTiled(bm, bn, supply);
  if ((supply.lhs.direct || supply.rhs.direct || supply.accumulator) &&
      !storageFitsBudget(function, config, measureStorage(function))) {
    b.setInsertionPoint(matrix);
    while (matrix->getPrevNode() != previous) matrix->getPrevNode()->erase();
    emitTiled(bm, bn, {});
  }
  matrix.erase();
  return success();
}
} // namespace intent::bangc
