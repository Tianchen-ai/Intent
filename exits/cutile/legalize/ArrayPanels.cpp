#include "intentdsl/levels/execution/block/ir/OwnerProgram.h"
#include "intentdsl/levels/execution/block/ir/TileBuilder.h"
namespace CI_UNIT_NAMESPACE {
using namespace mlir;
using intentdsl::block::Tiles;
namespace {
Value reshape(Tiles &t, Value value, ArrayRef<int64_t> shape) {
  return t.op("reshape", value, t.shaped(shape, t.elem(value.getType())))
      ->getResult(0);
}
Value scalarIndex(Tiles &t, Value value, StringRef op = "min") {
  return isa<RankedTensorType>(value.getType()) ? t.reduce(value, op, 0)
                                                : value;
}
Value blockIndex(Tiles &t, Value index, int64_t width) {
  return t.b.create<arith::DivUIOp>(t.loc, index, t.integer(width));
}
Value load(Tiles &t, Value base, ValueRange at, ArrayRef<int64_t> shape,
           bool tma) {
  SmallVector<Value> args{base};
  llvm::append_range(args, at);
  return t
      .op("rect_read", args,
          t.shaped(shape, cast<MemRefType>(base.getType()).getElementType()),
          {t.b.getNamedAttr("allow_tma", t.b.getBoolAttr(tma))})
      ->getResult(0);
}
bool rectangle(Tiles &t, Operation &op, ArrayRef<Value> coords, int64_t rows,
               Value stored, Value &result) {
  auto rectAxis = op.getAttrOfType<IntegerAttr>("ct_src.owner_rectangle");
  if (!rectAxis)
    return false;
  int64_t pos = rectAxis.getInt();
  auto base = op.getOperand(stored ? 1 : 0);
  int64_t width = t.shape(base).asArrayRef().back();
  SmallVector<Value> at;
  for (auto [idx, coord] : llvm::enumerate(coords)) {
    Value sc = scalarIndex(t, coord);
    at.push_back(int64_t(idx) == pos ? blockIndex(t, sc, rows) : sc);
  }
  at.push_back(t.integer(0));
  SmallVector<int64_t> dims(at.size(), 1);
  dims[pos] = rows;
  dims.back() = width;
  if (stored) {
    SmallVector<Value> args{base, reshape(t, stored, dims)};
    llvm::append_range(args, at);
    t.op("rect_write", args);
  } else {
    bool tma = op.hasAttr("ct_src.owner_tma");
    result = reshape(t, load(t, base, at, dims, tma), {rows, width});
  }
  return true;
}
Value panel(Tiles &t, Operation &op, ArrayRef<Value> coords, Value start,
            int64_t span, int64_t width) {
  if (!llvm::isPowerOf2_64(width))
    return {};
  auto page = op.getAttrOfType<IntegerAttr>("block.page_extent");
  if (!page && !op.hasAttr("block.rectangular_tokens"))
    return {};
  SmallVector<Value> at;
  int axis = page ? 1 : coords.size() - 1;
  for (auto [i, v] : llvm::enumerate(coords)) {
    if (int(i) == axis) {
      auto offset = page ? Value(t.b.create<arith::RemUIOp>(
                               t.loc, start, t.integer(page.getInt())))
                         : start;
      at.push_back(blockIndex(t, offset, span));
    } else
      at.push_back(scalarIndex(t, v, i == 0 && page ? "max" : "min"));
  }
  at.push_back(t.integer(0));
  SmallVector<int64_t> dims(at.size(), 1);
  dims[axis] = span;
  dims.back() = width;
  bool tma = op.hasAttr(page ? "ct_src.page_tma" : "ct_src.panel_tma");
  return reshape(t, load(t, op.getOperand(0), at, dims, tma), {span, width});
}
} // namespace
mlir::Value nativeMatrixRead(Tiles &t, intentdsl::block::MatrixAccess &p,
                             bool left, Value group, Value panel) {
  if (left && p.dot.getNumOperands() > 2)
    return {};
  auto view = p.dot.getOperand(left ? 0 : 1);
  int rank = p.rank(view), batch = p.axis(left ? "lhs_batch" : "rhs_batch"),
      inner = p.axis(left ? "lhs_k" : "rhs_k");
  if ((rank != 2 && rank != 3) || (batch != -1 && batch != 0))
    return {};
  SmallVector<Value> at(rank);
  SmallVector<int64_t> dims(rank);
  if (batch == 0) {
    at[0] = group;
    dims[0] = 1;
  }
  auto free = p.freeAxis(view, inner, batch);
  at[inner] = panel;
  dims[inner] = p.tile[2];
  at[free] = t.pid(left ? 0 : 1);
  dims[free] = p.tile[left ? 0 : 1];
  auto value = load(t, view, at, dims, t.fn->hasAttr("ct_src.tma"));
  if (rank == 3)
    value = reshape(t, value, ArrayRef<int64_t>(dims).drop_front());
  bool transpose = inner != (left ? rank - 1 : rank - 2);
  if (transpose) {
    auto tile = cast<RankedTensorType>(value.getType());
    return t
        .op("transpose", value,
            t.shaped({tile.getDimSize(1), tile.getDimSize(0)},
                     tile.getElementType()))
        ->getResult(0);
  }
  return value;
}
mlir::Value nativeWindowPanels(Tiles &t, intentdsl::block::MatrixAccess &p,
                               Value &batch, Value &chunk, Value &head) {
  auto x = p.dot.getOperand(0), basis = p.dot.getOperand(1),
       dt = p.dot.getOperand(2), decay = p.dot.getOperand(3);
  int64_t bm = p.tile[0], bn = p.tile[1], bk = p.tile[2], k = p.shape[2];
  if (p.rank(x) != 4 || p.rank(basis) != 4 || p.rank(dt) != 4 ||
      p.rank(decay) != 4 || p.shape[0] <= 0 || p.shape[1] <= 0 || k <= 0 ||
      k % bk != 0)
    return {};
  auto &b = t.b;
  auto loc = t.loc;
  int64_t heads = std::max<int64_t>(1, t.shape(dt)[1]),
          chunks = std::max<int64_t>(1, t.shape(dt)[2]),
          panelsPerChunk = k / bk;
  Value gp = t.pid(2);
  Value headIndex = b.create<arith::DivUIOp>(loc, gp, t.integer(heads));
  head = b.create<arith::RemUIOp>(loc, gp, t.integer(heads));
  batch = b.create<arith::DivUIOp>(loc, headIndex, t.integer(chunks));
  chunk = b.create<arith::RemUIOp>(loc, headIndex, t.integer(chunks));
  Value group =
      b.create<arith::DivUIOp>(loc, head, t.integer(p.axis("head_group")));
  Value last = t.read(
      decay,
      t.address(decay, {batch, head, chunk, t.integer(k - 1)}, {-1, -1, -1, -1}));
  bool tma = t.fn->hasAttr("ct_src.tma");
  auto accTy = t.shaped({bm, bn}, b.getF32Type());
  Value init = t.like(t.scalar(0), accTy);
  Value chunkBase =
      b.create<arith::MulIOp>(loc, chunk, t.integer(panelsPerChunk));
  auto lo = b.create<arith::ConstantIndexOp>(loc, 0),
       hi = b.create<arith::ConstantIndexOp>(loc, panelsPerChunk),
       one = b.create<arith::ConstantIndexOp>(loc, 1);
  auto loop = b.create<scf::ForOp>(loc, lo, hi, one, ValueRange{init});
  loop->setAttr(t.symbol("runtime_range"), b.getUnitAttr());
  b.setInsertionPointToStart(loop.getBody());
  Value it =
      b.create<arith::IndexCastOp>(loc, b.getI64Type(), loop.getInductionVar());
  Value seqPanel = b.create<arith::AddIOp>(loc, chunkBase, it);
  Value xTile = reshape(
      t, load(t, x, {batch, seqPanel, head, t.pid(0)}, {1, bk, 1, bm}, tma),
      {bk, bm});
  Value lhs =
      t.op("transpose", xTile, t.shaped({bm, bk}, t.elem(xTile.getType())))
          ->getResult(0);
  Value rhs = reshape(
      t, load(t, basis, {batch, seqPanel, group, t.pid(1)}, {1, bk, 1, bn}, tma),
      {bk, bn});
  Value dtVec = reshape(
      t, load(t, dt, {batch, head, chunk, it}, {1, 1, 1, bk}, false), {bk});
  Value decVec = reshape(
      t, load(t, decay, {batch, head, chunk, it}, {1, 1, 1, bk}, false), {bk});
  auto &body = p.dot.getRegion(0).front();
  IRMapping mapping;
  auto ty = cast<RankedTensorType>(rhs.getType());
  for (auto [arg, val] :
       llvm::zip(body.getArguments(),
                 SmallVector<Value>{rhs, dtVec, decVec, last}))
    mapping.map(arg, t.like(val, ty.clone(t.elem(val.getType())),
                            isa<ShapedType>(val.getType()) ? 0 : -1));
  for (auto &old : body.without_terminator()) {
    Value v;
    if (auto lit = dyn_cast<arith::ConstantOp>(old))
      v = t.like(b.create<arith::ConstantOp>(loc, lit.getValue()),
                 ty.clone(lit.getType()));
    else {
      OperationState state(loc, old.getName());
      for (auto arg : old.getOperands())
        state.addOperands(mapping.lookup(arg));
      state.addTypes(ty.clone(old.getResult(0).getType()));
      state.addAttributes(old.getAttrs());
      v = b.create(state)->getResult(0);
    }
    mapping.map(old.getResult(0), v);
  }
  Value rhsOut = mapping.lookup(body.back().getOperand(0));
  if (p.shape[1] % bn != 0) {
    Value col =
        t.stepped(b.create<arith::MulIOp>(loc, t.pid(1), t.integer(bn)), bn);
    Value validCol = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                             col,
                                             t.like(t.integer(p.shape[1]), col.getType()));
    Value guard = t.like(validCol, ty.clone(b.getI1Type()), 1);
    Value zeroRhs = t.like(
        b.create<arith::ConstantOp>(loc, b.getZeroAttr(ty.getElementType())),
        ty);
    rhsOut = b.create<arith::SelectOp>(loc, guard, rhsOut, zeroRhs);
  }
  Value next = t.dot(lhs, rhsOut, false, loop.getRegionIterArgs()[0]);
  b.create<scf::YieldOp>(loc, next);
  b.setInsertionPointAfter(loop);
  return loop.getResult(0);
}
bool nativeMatrixWrite(Tiles &t, intentdsl::block::MatrixAccess &p, Value y,
                       Value val, ArrayRef<Value> batchCoords) {
  auto arg = dyn_cast<BlockArgument>(y);
  if (!arg || arg.getOwner() != &t.fn.front() || p.shape[0] <= 0 ||
      p.shape[1] <= 0 || p.shape[0] % p.tile[0] != 0 ||
      p.shape[1] % p.tile[1] != 0)
    return false;
  int rank = p.rank(y);
  if (rank < 2 || batchCoords.size() + 2 != size_t(rank))
    return false;
  Value shaped = val;
  if (rank > 2) {
    SmallVector<int64_t> dims(batchCoords.size(), 1);
    dims.push_back(p.tile[0]);
    dims.push_back(p.tile[1]);
    shaped = reshape(t, val, dims);
  }
  SmallVector<Value> args{y, shaped};
  llvm::append_range(args, batchCoords);
  args.push_back(t.pid(0));
  args.push_back(t.pid(1));
  t.op("rect_write", args);
  return true;
}
intentdsl::block::OwnerAccess nativeOwnerViews() { return {rectangle, panel}; }
} // namespace CI_UNIT_NAMESPACE
