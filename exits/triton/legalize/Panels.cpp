#include "../capabilities/Target.h"
#include "compilerinfra/Assembly.h"
#include "compilerinfra/Proofs.h"
#include "intentdsl/levels/algorithm/kir/ir/PanelMap.h"
#include "intentdsl/levels/algorithm/kir/ir/Storage.h"
#include "intentdsl/levels/execution/block/ir/MatrixProgram.h"
#include "intentdsl/levels/execution/block/ir/OwnerProgram.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/MathExtras.h"

namespace CI_UNIT_NAMESPACE {
using namespace mlir;
using intentdsl::block::MatrixAccess;
using intentdsl::block::Tiles;

namespace {
bool tmaViewCompatible(ArrayRef<int64_t> viewShape,
                       ArrayRef<int64_t> viewStrides,
                       ArrayRef<int64_t> blockShape, int64_t elemBytes) {
  if (elemBytes <= 0 || viewShape.size() < 2 || viewShape.size() > 5 ||
      viewStrides.size() != viewShape.size() ||
      blockShape.size() != viewShape.size())
    return false;
  if (viewStrides.back() != 1 || blockShape.back() * elemBytes < 16)
    return false;
  for (int64_t b : blockShape)
    if (b <= 0 || b > 256 || !llvm::isPowerOf2_64(b))
      return false;
  for (int64_t d : viewShape)
    if (d <= 0 || d > INT32_MAX)
      return false;
  for (int64_t s : viewStrides.drop_back()) {
    auto bytes = llvm::checkedMul<int64_t>(s, elemBytes);
    if (s <= 0 || !bytes || *bytes % 16 != 0)
      return false;
  }
  return true;
}

bool directViewLayout(Value view, func::FuncOp fn,
                      SmallVectorImpl<int64_t> &shapeOut,
                      SmallVectorImpl<int64_t> &stridesOut,
                      int64_t &elemBytesOut) {
  auto arg = dyn_cast<BlockArgument>(view);
  if (!arg || arg.getOwner() != &fn.front() ||
      fn.getArgAttr(arg.getArgNumber(), "kir.indexing"))
    return false;
  auto mem = dyn_cast<MemRefType>(arg.getType());
  auto sh = fn.getArgAttrOfType<DenseI64ArrayAttr>(arg.getArgNumber(),
                                                   "block.shape");
  auto st = fn.getArgAttrOfType<DenseI64ArrayAttr>(arg.getArgNumber(),
                                                   "block.strides");
  if (!mem || !sh || !st || sh.size() != size_t(mem.getRank()) ||
      st.size() != sh.size())
    return false;
  elemBytesOut = llvm::divideCeil(mem.getElementTypeBitWidth(), 8);
  shapeOut.assign(sh.asArrayRef().begin(), sh.asArrayRef().end());
  stridesOut.assign(st.asArrayRef().begin(), st.asArrayRef().end());
  return true;
}

Value emitDescRead(Tiles &t, Value base, ValueRange offsets,
                   ArrayRef<int64_t> viewShape, ArrayRef<int64_t> viewStrides,
                   ArrayRef<int64_t> blockShape) {
  SmallVector<Value> args{base};
  llvm::append_range(args, offsets);
  return t
      .op("desc_read", args,
          t.shaped(blockShape, cast<MemRefType>(base.getType()).getElementType()),
          {t.b.getNamedAttr("view_shape",
                            t.b.getDenseI64ArrayAttr(viewShape)),
           t.b.getNamedAttr("view_strides",
                            t.b.getDenseI64ArrayAttr(viewStrides)),
           t.b.getNamedAttr("block_shape",
                            t.b.getDenseI64ArrayAttr(blockShape))})
      ->getResult(0);
}

void emitDescWrite(Tiles &t, Value base, Value val, ValueRange offsets,
                   ArrayRef<int64_t> viewShape, ArrayRef<int64_t> viewStrides,
                   ArrayRef<int64_t> blockShape) {
  SmallVector<Value> args{base, val};
  llvm::append_range(args, offsets);
  t.op("desc_write", args, {},
       {t.b.getNamedAttr("view_shape", t.b.getDenseI64ArrayAttr(viewShape)),
        t.b.getNamedAttr("view_strides", t.b.getDenseI64ArrayAttr(viewStrides)),
        t.b.getNamedAttr("block_shape", t.b.getDenseI64ArrayAttr(blockShape))});
}

Value reshapeTile(Tiles &t, Value v, ArrayRef<int64_t> shape) {
  return t.op("reshape", v, t.shaped(shape, t.elem(v.getType())))->getResult(0);
}

Value transposeTile(Tiles &t, Value v) {
  auto ty = cast<RankedTensorType>(v.getType());
  return t
      .op("transpose", v,
          t.shaped({ty.getDimSize(1), ty.getDimSize(0)}, ty.getElementType()))
      ->getResult(0);
}

Value scalarCoord(Tiles &t, Value v, StringRef alg = "min") {
  return isa<RankedTensorType>(v.getType()) ? t.reduce(v, alg, 0) : v;
}

bool ownerRectangle(Tiles &t, Operation &op, ArrayRef<Value> coords,
                    int64_t rows, Value stored, Value &result) {
  if (!op.hasAttr("triton_src.owner_tma"))
    return false;
  auto rectAxis = op.getAttrOfType<IntegerAttr>("triton_src.owner_rectangle");
  if (!rectAxis)
    return false;
  int64_t pos = rectAxis.getInt();
  Value base = op.getOperand(stored ? 1 : 0);
  SmallVector<int64_t> viewShape, viewStrides;
  int64_t elemBytes = 0;
  if (!directViewLayout(base, t.fn, viewShape, viewStrides, elemBytes))
    return false;
  int64_t width = viewShape.back();
  SmallVector<Value> offsets;
  for (auto [idx, coord] : llvm::enumerate(coords))
    offsets.push_back(scalarCoord(t, coord));
  offsets.push_back(t.integer(0));
  SmallVector<int64_t> blockShape(offsets.size(), 1);
  blockShape[pos] = rows;
  blockShape.back() = width;
  if (!tmaViewCompatible(viewShape, viewStrides, blockShape, elemBytes))
    return false;
  if (stored) {
    emitDescWrite(t, base, reshapeTile(t, stored, blockShape), offsets,
                  viewShape, viewStrides, blockShape);
  } else {
    result = reshapeTile(
        t,
        emitDescRead(t, base, offsets, viewShape, viewStrides, blockShape),
        {rows, width});
  }
  return true;
}

Value ownerPanel(Tiles &t, Operation &op, ArrayRef<Value> coords, Value start,
                 int64_t span, int64_t width) {
  bool pageTma = op.hasAttr("triton_src.page_tma");
  bool panelTma = op.hasAttr("triton_src.panel_tma");
  if ((!pageTma && !panelTma) || !llvm::isPowerOf2_64(width))
    return {};
  auto page = op.getAttrOfType<IntegerAttr>("block.page_extent");
  Value base = op.getOperand(0);
  SmallVector<int64_t> viewShape, viewStrides;
  int64_t elemBytes = 0;
  if (!directViewLayout(base, t.fn, viewShape, viewStrides, elemBytes))
    return {};
  SmallVector<Value> offsets;
  int axis = page ? 1 : int(coords.size()) - 1;
  for (auto [i, v] : llvm::enumerate(coords)) {
    if (int(i) == axis) {
      Value off = page ? Value(t.b.create<arith::RemUIOp>(
                             t.loc, start, t.integer(page.getInt())))
                       : start;
      offsets.push_back(off);
    } else {
      offsets.push_back(scalarCoord(t, v, (i == 0 && page) ? "max" : "min"));
    }
  }
  offsets.push_back(t.integer(0));
  SmallVector<int64_t> blockShape(offsets.size(), 1);
  blockShape[axis] = span;
  blockShape.back() = width;
  if (!tmaViewCompatible(viewShape, viewStrides, blockShape, elemBytes))
    return {};
  return reshapeTile(
      t, emitDescRead(t, base, offsets, viewShape, viewStrides, blockShape),
      {span, width});
}
} // namespace

std::optional<int64_t> tritonOwnerRectangleAxis(Operation *op, func::FuncOp fn) {
  auto kind = op->getName().stripDialect();
  if (kind != "slice_read" && kind != "slice_write")
    return std::nullopt;
  auto &space = fn.front().front();
  auto pair = space.getAttrOfType<IntegerAttr>("block.pair_axis");
  if (!pair || space.hasAttr("triton_src.matrix_rows") ||
      op->getParentOp() != &space)
    return std::nullopt;
  auto &bb = space.getRegion(0).front();
  auto coords = op->getOperands().drop_front(kind == "slice_write" ? 2 : 1);
  if (coords.empty() ||
      op->getAttrOfType<IntegerAttr>("axis").getInt() != int64_t(coords.size()))
    return std::nullopt;
  Value pairArg = bb.getArgument(pair.getInt());
  int64_t pairPos = -1;
  llvm::SmallDenseSet<Value, 4> seen;
  for (auto [idx, v] : llvm::enumerate(coords)) {
    auto arg = dyn_cast<BlockArgument>(v);
    if (!arg || arg.getOwner() != &bb || !seen.insert(v).second)
      return std::nullopt;
    if (v == pairArg)
      pairPos = idx;
  }
  if (pairPos < 0)
    return std::nullopt;
  auto view =
      cast<BlockArgument>(op->getOperand(kind == "slice_write" ? 1 : 0));
  auto shape =
      fn.getArgAttrOfType<DenseI64ArrayAttr>(view.getArgNumber(), "block.shape");
  auto ownerTile =
      space.getAttrOfType<IntegerAttr>("block.owner_tile").getInt();
  if (!shape || shape.size() != coords.size() + 1 ||
      shape[pairPos] < ownerTile ||
      !llvm::isPowerOf2_64(shape.asArrayRef().back()))
    return std::nullopt;
  compilerinfra::TailDimension ownerDim;
  ownerDim.staticExtent = shape[pairPos];
  ownerDim.staticOffset = shape[pairPos] - ownerTile;
  ownerDim.tileSize = ownerTile;
  ownerDim.step = 1;
  ownerDim.tileAlignment = ownerTile;
  compilerinfra::TailDimension laneDim;
  laneDim.staticExtent = shape.asArrayRef().back();
  laneDim.staticOffset = 0;
  laneDim.tileSize = shape.asArrayRef().back();
  laneDim.step = 1;
  laneDim.tileAlignment = shape.asArrayRef().back();
  auto tailProof = compilerinfra::proveTailAccess({ownerDim, laneDim});
  if (!tailProof.proven ||
      tailProof.kind != compilerinfra::TailKind::ExactInBounds)
    return std::nullopt;
  return pairPos;
}

bool tritonOwnerRectPermitsTma(Operation *op, func::FuncOp fn, bool permitTma) {
  if (!permitTma)
    return false;
  auto rectPos = tritonOwnerRectangleAxis(op, fn);
  if (!rectPos)
    return false;
  auto rows = fn.front().front().getAttrOfType<IntegerAttr>("block.owner_tile");
  if (!rows || rows.getInt() < 16)
    return false;
  bool write = op->getName().stripDialect() == "slice_write";
  Value base = op->getOperand(write ? 1 : 0);
  SmallVector<int64_t> viewShape, viewStrides;
  int64_t elemBytes = 0;
  if (!directViewLayout(base, fn, viewShape, viewStrides, elemBytes))
    return false;
  int64_t depth = viewShape.back();
  if (rows.getInt() * depth * elemBytes > 32768)
    return false;
  SmallVector<int64_t> blockShape(viewShape.size(), 1);
  blockShape[*rectPos] = rows.getInt();
  blockShape.back() = depth;
  return tmaViewCompatible(viewShape, viewStrides, blockShape, elemBytes);
}

bool tritonPanelPermitsTma(Operation *op, func::FuncOp fn, bool permitTma) {
  if (!permitTma || !op->hasAttr("block.rectangular_tokens"))
    return false;
  auto *parent = op->getParentOp();
  auto span =
      parent ? parent->getAttrOfType<IntegerAttr>("block.contribution_width")
             : IntegerAttr();
  if (!span || span.getInt() <= 0)
    return false;
  SmallVector<int64_t> viewShape, viewStrides;
  int64_t elemBytes = 0;
  if (!directViewLayout(op->getOperand(0), fn, viewShape, viewStrides,
                        elemBytes) ||
      viewShape.size() < 2)
    return false;
  int64_t depth = viewShape.back();
  if (span.getInt() * depth * elemBytes > 32768)
    return false;
  SmallVector<int64_t> blockShape(viewShape.size(), 1);
  blockShape[viewShape.size() - 2] = span.getInt();
  blockShape.back() = depth;
  return tmaViewCompatible(viewShape, viewStrides, blockShape, elemBytes);
}

bool tritonPagePermitsTma(Operation *op, func::FuncOp fn, bool permitTma) {
  if (!permitTma || !op->hasAttr("block.page_extent"))
    return false;
  auto *parent = op->getParentOp();
  auto span =
      parent ? parent->getAttrOfType<IntegerAttr>("block.contribution_width")
             : IntegerAttr();
  if (!span || span.getInt() <= 0)
    return false;
  SmallVector<int64_t> viewShape, viewStrides;
  int64_t elemBytes = 0;
  if (!directViewLayout(op->getOperand(0), fn, viewShape, viewStrides,
                        elemBytes) ||
      viewShape.size() < 3)
    return false;
  int64_t depth = viewShape.back();
  if (span.getInt() * depth * elemBytes > 32768)
    return false;
  SmallVector<int64_t> blockShape(viewShape.size(), 1);
  blockShape[1] = span.getInt();
  blockShape.back() = depth;
  return tmaViewCompatible(viewShape, viewStrides, blockShape, elemBytes);
}

Value tritonMatrixRead(Tiles &t, MatrixAccess &p, bool left, Value group,
                       Value panel) {
  if (!t.fn->hasAttr("triton_src.tma") || (left && p.dot.getNumOperands() > 2))
    return {};
  Value view = p.dot.getOperand(left ? 0 : 1);
  int rank = p.rank(view), batch = p.axis(left ? "lhs_batch" : "rhs_batch"),
      inner = p.axis(left ? "lhs_k" : "rhs_k");
  if ((rank != 2 && rank != 3) || (batch != -1 && batch != 0))
    return {};
  SmallVector<int64_t> viewShape, viewStrides;
  int64_t elemBytes = 0;
  if (!directViewLayout(view, t.fn, viewShape, viewStrides, elemBytes))
    return {};
  auto &b = t.b;
  auto loc = t.loc;
  int freeAx = p.freeAxis(view, inner, batch);
  int64_t freeTile = p.tile[left ? 0 : 1], kTile = p.tile[2];
  Value freeOff =
      b.create<arith::MulIOp>(loc, t.pid(left ? 0 : 1), t.integer(freeTile));
  Value kOff = b.create<arith::MulIOp>(loc, panel, t.integer(kTile));

  SmallVector<Value> offsets(rank);
  SmallVector<int64_t> blockShape(rank);
  if (batch == 0) {
    offsets[0] = group;
    blockShape[0] = 1;
  }
  offsets[inner] = kOff;
  blockShape[inner] = kTile;
  offsets[freeAx] = freeOff;
  blockShape[freeAx] = freeTile;

  bool needTranspose = inner != (left ? rank - 1 : rank - 2);
  if (viewStrides.back() != 1 && viewStrides[rank - 2] == 1) {
    std::swap(viewShape[rank - 2], viewShape[rank - 1]);
    std::swap(viewStrides[rank - 2], viewStrides[rank - 1]);
    std::swap(blockShape[rank - 2], blockShape[rank - 1]);
    std::swap(offsets[rank - 2], offsets[rank - 1]);
    needTranspose = !needTranspose;
  }
  if (!tmaViewCompatible(viewShape, viewStrides, blockShape, elemBytes))
    return {};
  Value loaded =
      emitDescRead(t, view, offsets, viewShape, viewStrides, blockShape);
  if (rank == 3)
    loaded = reshapeTile(t, loaded, ArrayRef<int64_t>(blockShape).drop_front());
  return needTranspose ? transposeTile(t, loaded) : loaded;
}

Value tritonWindowPanels(Tiles &t, MatrixAccess &p, Value &batch, Value &chunk,
                         Value &head) {
  if (!t.fn->hasAttr("triton_src.tma"))
    return {};
  Value x = p.dot.getOperand(0), basis = p.dot.getOperand(1),
        dt = p.dot.getOperand(2), decay = p.dot.getOperand(3);
  int64_t bm = p.tile[0], bn = p.tile[1], bk = p.tile[2], k = p.shape[2];
  if (p.rank(x) != 4 || p.rank(basis) != 4 || p.rank(dt) != 4 ||
      p.rank(decay) != 4 || p.shape[0] <= 0 || p.shape[1] <= 0 || k <= 0 ||
      k % bk != 0)
    return {};
  SmallVector<int64_t> xShape, xStrides, bShape, bStrides;
  int64_t xBytes = 0, bBytes = 0;
  SmallVector<int64_t> xBlock{1, bk, 1, bm}, bBlock{1, bk, 1, bn};
  if (!directViewLayout(x, t.fn, xShape, xStrides, xBytes) ||
      !directViewLayout(basis, t.fn, bShape, bStrides, bBytes) ||
      !tmaViewCompatible(xShape, xStrides, xBlock, xBytes) ||
      !tmaViewCompatible(bShape, bStrides, bBlock, bBytes))
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
  Value rowOff = b.create<arith::MulIOp>(loc, t.pid(0), t.integer(bm));
  Value colOff = b.create<arith::MulIOp>(loc, t.pid(1), t.integer(bn));
  Value chunkElemBase = b.create<arith::MulIOp>(loc, chunk, t.integer(k));

  auto accTy = t.shaped({bm, bn}, b.getF32Type());
  Value init = t.like(t.scalar(0), accTy);
  auto lo = b.create<arith::ConstantIndexOp>(loc, 0),
       hi = b.create<arith::ConstantIndexOp>(loc, panelsPerChunk),
       one = b.create<arith::ConstantIndexOp>(loc, 1);
  auto loop = b.create<scf::ForOp>(loc, lo, hi, one, ValueRange{init});
  loop->setAttr(t.symbol("runtime_range"), b.getUnitAttr());
  b.setInsertionPointToStart(loop.getBody());
  Value it =
      b.create<arith::IndexCastOp>(loc, b.getI64Type(), loop.getInductionVar());
  Value panelOff = b.create<arith::MulIOp>(loc, it, t.integer(bk));
  Value seqOff = b.create<arith::AddIOp>(loc, chunkElemBase, panelOff);

  Value xTile = reshapeTile(
      t, emitDescRead(t, x, {batch, seqOff, head, rowOff}, xShape, xStrides, xBlock),
      {bk, bm});
  Value lhs = transposeTile(t, xTile);
  Value rhs = reshapeTile(
      t,
      emitDescRead(t, basis, {batch, seqOff, group, colOff}, bShape, bStrides,
                   bBlock),
      {bk, bn});
  Value member = t.stepped(panelOff, bk);
  Value valid = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, member,
                                        t.like(t.integer(k), member.getType()));
  auto readCoeff = [&](Value tensor) {
    return t.read(tensor,
                  t.address(tensor, {batch, head, chunk, member},
                            {-1, -1, -1, 0}),
                  valid);
  };
  auto &body = p.dot.getRegion(0).front();
  IRMapping mapping;
  auto ty = cast<RankedTensorType>(rhs.getType());
  for (auto [arg, val] :
       llvm::zip(body.getArguments(),
                 SmallVector<Value>{rhs, readCoeff(dt), readCoeff(decay), last}))
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
    Value col = t.stepped(colOff, bn);
    Value validCol = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, col,
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

bool tritonMatrixWrite(Tiles &t, MatrixAccess &p, Value y, Value val,
                       ArrayRef<Value> batchCoords) {
  if (!t.fn->hasAttr("triton_src.tma") || p.shape[0] <= 0 || p.shape[1] <= 0)
    return false;
  int rank = p.rank(y);
  if (rank < 2 || batchCoords.size() + 2 != size_t(rank))
    return false;
  SmallVector<int64_t> viewShape, viewStrides;
  int64_t elemBytes = 0;
  if (!directViewLayout(y, t.fn, viewShape, viewStrides, elemBytes))
    return false;
  SmallVector<int64_t> blockShape(batchCoords.size(), 1);
  blockShape.push_back(p.tile[0]);
  blockShape.push_back(p.tile[1]);
  if (!tmaViewCompatible(viewShape, viewStrides, blockShape, elemBytes))
    return false;
  auto &b = t.b;
  auto loc = t.loc;
  Value shaped = rank > 2 ? reshapeTile(t, val, blockShape) : val;
  SmallVector<Value> offsets(batchCoords.begin(), batchCoords.end());
  offsets.push_back(
      b.create<arith::MulIOp>(loc, t.pid(0), t.integer(p.tile[0])));
  offsets.push_back(
      b.create<arith::MulIOp>(loc, t.pid(1), t.integer(p.tile[1])));
  emitDescWrite(t, y, shaped, offsets, viewShape, viewStrides, blockShape);
  return true;
}

Value tritonScaledPanels(Tiles &t, MatrixAccess &p) {
  auto &b = t.b;
  auto loc = t.loc;
  auto format = cast<intentdsl::kir::MX8Type>(
                    p.dot.getAttrOfType<TypeAttr>("kir.storage").getValue())
                    .getLayout();
  int64_t m = p.tile[0], n = p.tile[1], k = p.tile[2], g = k / format.group;
  int64_t totalM = p.shape[0], totalN = p.shape[1], totalK = p.shape[2],
          totalGroups = totalK / format.group;
  auto *dispatch = p.dot.getPrevNode();
  auto range = dispatch->getAttrOfType<DenseI64ArrayAttr>("interval");
  bool native = dispatch->getAttrOfType<StringAttr>("on_match") == "mx";
  bool allowTma = t.fn->hasAttr("triton_src.tma");

  Value a = p.dot.getOperand(0), sa = p.dot.getOperand(1),
        rhs = p.dot.getOperand(2);
  intentdsl::kir::RightScale scale(p.dot);

  auto castTo = [&](StringRef opName, Value v, Type e) {
    OperationState st(loc, opName);
    st.addOperands(v);
    st.addTypes(cast<RankedTensorType>(v.getType()).clone(e));
    return b.create(st)->getResult(0);
  };
  auto cmpLt = [&](Value x, Value bound) {
    return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, x,
                                   t.like(bound, x.getType()));
  };
  auto pairMask = [&](Value r, Value c) {
    auto ty = t.shaped({cast<RankedTensorType>(r.getType()).getDimSize(0),
                        cast<RankedTensorType>(c.getType()).getDimSize(0)},
                       b.getI1Type());
    return b.create<arith::AndIOp>(loc, t.like(r, ty, 0), t.like(c, ty, 1));
  };

  Value rowBase = b.create<arith::MulIOp>(loc, t.pid(0), t.integer(m));
  Value colBase = b.create<arith::MulIOp>(loc, t.pid(1), t.integer(n));
  Value row = t.stepped(rowBase, m);
  Value col = t.stepped(colBase, n);
  Value validRow = cmpLt(row, t.integer(totalM));
  Value validCol = cmpLt(col, t.integer(totalN));

  SmallVector<int64_t> aShape, aStrides, rShape, rStrides;
  int64_t aBytes = 0, rBytes = 0;
  bool aFlatTma = false, rFlatTma = false;
  if (allowTma && directViewLayout(a, t.fn, aShape, aStrides, aBytes) &&
      aShape.size() == 3 && aStrides[1] == format.group * aStrides[2] &&
      aStrides[2] == 1)
    aFlatTma = tmaViewCompatible({totalM, totalK}, {aStrides[0], 1}, {m, k}, 1);
  if (allowTma && directViewLayout(rhs, t.fn, rShape, rStrides, rBytes) &&
      rShape.size() == 3 && rStrides[0] == format.group * rStrides[1] &&
      rStrides[2] == 1)
    rFlatTma = tmaViewCompatible({totalK, totalN}, {rStrides[1], 1}, {k, n}, 1);

  auto matrix = t.shaped({m, n}, b.getF32Type());
  auto loop = b.create<scf::ForOp>(
      loc, t.integer(0), t.integer(llvm::divideCeil(totalK, k)), t.integer(1),
      ValueRange{t.like(t.scalar(0), matrix)});
  loop->setAttr(t.symbol("runtime_range"), b.getUnitAttr());
  b.setInsertionPointToStart(loop.getBody());
  Value iv = loop.getInductionVar(), acc = loop.getRegionIterArgs()[0];

  Value grpBase = b.create<arith::MulIOp>(loc, iv, t.integer(g));
  Value grp = t.stepped(grpBase, g);
  Value validGrp = cmpLt(grp, t.integer(totalGroups));

  Value biasI8 =
      b.create<arith::ConstantOp>(loc, b.getIntegerAttr(b.getI8Type(), format.bias));
  Value maskLS = pairMask(validRow, validGrp);
  Value rawLS = t.read(sa, t.address(sa, {row, grp}, {0, 1}), maskLS);
  Value leftScale = b.create<arith::SelectOp>(
      loc, maskLS, rawLS,
      t.like(biasI8, cast<RankedTensorType>(rawLS.getType())));

  Value rightScaleNG, rightScaleGN;
  if (scale.column == 0) {
    Value maskRS = pairMask(validCol, validGrp);
    Value rawRS =
        t.read(scale.view, t.address(scale.view, {col, grp}, {0, 1}), maskRS);
    rightScaleNG = b.create<arith::SelectOp>(
        loc, maskRS, rawRS,
        t.like(biasI8, cast<RankedTensorType>(rawRS.getType())));
    rightScaleGN = transposeTile(t, rightScaleNG);
  } else {
    Value maskRS = pairMask(validGrp, validCol);
    Value rawRS =
        t.read(scale.view, t.address(scale.view, {grp, col}, {0, 1}), maskRS);
    rightScaleGN = b.create<arith::SelectOp>(
        loc, maskRS, rawRS,
        t.like(biasI8, cast<RankedTensorType>(rawRS.getType())));
    rightScaleNG = transposeTile(t, rightScaleGN);
  }

  Value kBase = b.create<arith::MulIOp>(loc, iv, t.integer(k));
  Value kk = t.stepped(kBase, k);
  Value validK = cmpLt(kk, t.integer(totalK));
  Value groupLit = t.like(t.integer(format.group), kk.getType());
  Value kkGrp = b.create<arith::DivUIOp>(loc, kk, groupLit);
  Value kkLane = b.create<arith::RemUIOp>(loc, kk, groupLit);

  Value lhs =
      aFlatTma ? emitDescRead(t, a, {rowBase, kBase}, {totalM, totalK},
                              {aStrides[0], 1}, {m, k})
               : t.read(a, t.address(a, {row, kkGrp, kkLane}, {0, 1, 1}),
                        pairMask(validRow, validK));
  Value rv =
      rFlatTma ? emitDescRead(t, rhs, {kBase, colBase}, {totalK, totalN},
                              {rStrides[1], 1}, {k, n})
               : t.read(rhs, t.address(rhs, {kkGrp, kkLane, col}, {0, 0, 1}),
                        pairMask(validK, validCol));

  auto inside = [&](Value sTile) {
    auto count = cast<RankedTensorType>(sTile.getType()).getNumElements();
    Value codes =
        castTo("arith.extui", reshapeTile(t, sTile, {count}), b.getI32Type());
    return b.create<arith::AndIOp>(
        loc,
        b.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::sge, t.reduce(codes, "min", 0),
            b.create<arith::ConstantIntOp>(loc, range[0], 32)),
        b.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt, t.reduce(codes, "max", 0),
            b.create<arith::ConstantIntOp>(loc, range[1] + 1, 32)));
  };
  Value validScales =
      b.create<arith::AndIOp>(loc, inside(leftScale), inside(rightScaleNG));
  auto branch = b.create<scf::IfOp>(loc, TypeRange{matrix}, validScales, true);
  auto product = [&](bool fast) -> Value {
    if (native && fast)
      return t
          .op("scaled_mma", {lhs, leftScale, rv, rightScaleNG, acc}, matrix,
              {b.getNamedAttr("ci.implementation",
                              b.getStringAttr("exits/triton::scaled-mma"))})
          ->getResult(0);
    auto rescale = [&](Value x, Value s, ArrayRef<int64_t> dims, int axis) {
      Value codes = castTo("arith.extui", s, b.getI32Type());
      Value factor =
          t.op("e8m0", codes,
               cast<RankedTensorType>(s.getType()).clone(b.getF32Type()))
              ->getResult(0);
      Value val = castTo("arith.extf", reshapeTile(t, x, dims), b.getF32Type());
      Value scaled = t.op("expand_scale", factor, val.getType(),
                          {b.getNamedAttr("axis", b.getI64IntegerAttr(axis))})
                         ->getResult(0);
      Value out = b.create<arith::MulFOp>(loc, val, scaled);
      return fast ? castTo("arith.truncf", out, b.getBF16Type()) : out;
    };
    return t.dot(
        reshapeTile(t, rescale(lhs, leftScale, {m, g, format.group}, 2),
                    {m, k}),
        reshapeTile(t, rescale(rv, rightScaleGN, {g, format.group, n}, 1),
                    {k, n}),
        false, acc);
  };
  b.setInsertionPointToStart(branch.thenBlock());
  b.create<scf::YieldOp>(loc, product(true));
  b.setInsertionPointToStart(branch.elseBlock());
  b.create<scf::YieldOp>(loc, product(false));
  b.setInsertionPointAfter(branch);
  b.create<scf::YieldOp>(loc, branch.getResults());
  b.setInsertionPointAfter(loop);
  return loop.getResult(0);
}

intentdsl::block::OwnerAccess tritonOwnerViews() {
  return {ownerRectangle, ownerPanel};
}

LogicalResult verifyTritonGeometry(func::FuncOp fn,
                                   const compilerinfra::TargetProfile &target) {
  auto cap = target.services.get<intentdsl::itsrc::TritonTarget>();
  if (!cap || fn->hasAttr("triton_src.tma") != cap->permitTma)
    return fn.emitError("Triton TMA capability disagrees with target");
  return success(
      !fn.walk([&](Operation *op) {
           auto n = op->getName().getStringRef();
           auto reject = [&](StringRef msg) {
             op->emitError(msg);
             return WalkResult::interrupt();
           };
           if (n == "triton_src.desc_read" || n == "triton_src.desc_write") {
             if (!cap->permitTma)
               return reject("descriptor transfer without TMA capability");
             bool write = n == "triton_src.desc_write";
             auto arg = dyn_cast<BlockArgument>(op->getOperand(0));
             if (!arg || arg.getOwner() != &fn.front())
               return reject("descriptor access must bind a function view");
             auto mem = cast<MemRefType>(arg.getType());
             auto tile =
                 cast<RankedTensorType>(write ? op->getOperand(1).getType()
                                              : op->getResult(0).getType());
             auto vShape = op->getAttrOfType<DenseI64ArrayAttr>("view_shape");
             auto vStrides =
                 op->getAttrOfType<DenseI64ArrayAttr>("view_strides");
             auto bShape = op->getAttrOfType<DenseI64ArrayAttr>("block_shape");
             if (!vShape || !vStrides || !bShape ||
                 tile.getElementType() != mem.getElementType() ||
                 tile.getShape() != bShape.asArrayRef() ||
                 op->getNumOperands() !=
                     (write ? 2u : 1u) + bShape.size() ||
                 !tmaViewCompatible(
                     vShape.asArrayRef(), vStrides.asArrayRef(),
                     bShape.asArrayRef(),
                     llvm::divideCeil(mem.getElementTypeBitWidth(), 8)))
               return reject("invalid Triton TMA descriptor geometry");
           }
           if (n == "triton_src.reshape" || n == "triton_src.transpose" ||
               n == "triton_src.expand_scale") {
             auto from = cast<RankedTensorType>(op->getOperand(0).getType()),
                  to = cast<RankedTensorType>(op->getResult(0).getType());
             if (!from.hasStaticShape() || !to.hasStaticShape() ||
                 from.getElementType() != to.getElementType())
               return reject("incompatible Triton tile shape or element type");
             if (n == "triton_src.reshape" &&
                 from.getNumElements() != to.getNumElements())
               return reject("reshape changes element count");
             if (n == "triton_src.transpose" &&
                 (from.getRank() != 2 || to.getRank() != 2 ||
                  from.getDimSize(0) != to.getDimSize(1) ||
                  from.getDimSize(1) != to.getDimSize(0)))
               return reject("invalid 2D tile transpose");
             if (n == "triton_src.expand_scale") {
               auto ax = op->getAttrOfType<IntegerAttr>("axis").getInt();
               SmallVector<int64_t> sh(to.getShape());
               if (ax < 0 || ax >= to.getRank())
                 return reject("invalid scale expansion axis");
               sh.erase(sh.begin() + ax);
               if (ArrayRef<int64_t>(sh) != from.getShape())
                 return reject("scale does not match grouped dimensions");
             }
           }
           return WalkResult::advance();
         }).wasInterrupted());
}
} // namespace CI_UNIT_NAMESPACE
