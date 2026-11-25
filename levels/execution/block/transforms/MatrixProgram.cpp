#include "intentdsl/levels/execution/block/ir/MatrixProgram.h"
#include "intentdsl/levels/algorithm/kir/ir/Ordered.h"
#include "intentdsl/levels/execution/block/ir/TileBuilder.h"

namespace intentdsl::block {
using namespace mlir;

void encodedTiles(Tiles &, MatrixAccess &, ScaledPanelLowering,
                  MatrixPanelRead, WindowPanelLowering, MatrixPanelWrite,
                  bool flatRecordBitPlanes);
namespace {
// Lower the chosen panel loop with a loop-carried accumulator. The pointers
// for both transposed and contiguous operands use their verified ABI strides.
void matrixTiles(Tiles &t, intentdsl::block::MatrixAccess &p,
                 MatrixPanelRead native, MatrixPanelWrite writeNative) {
  auto &b = t.b;
  auto loc = t.loc;
  auto [m, n, k] = std::tuple{p.shape[0], p.shape[1], p.shape[2]};
  auto [bm, bn, bk] = std::tuple{p.tile[0], p.tile[1], p.tile[2]};
  Value gp = t.pid(2);
  Value first = b.create<arith::MulIOp>(loc, t.pid(0), t.integer(bm));
  Value row = t.stepped(first, bm),
        col = t.stepped(b.create<arith::MulIOp>(loc, t.pid(1), t.integer(bn)),
                        bn);
  Value end = t.integer(m), groupStart, live;
  auto less = [&](Value x, Value y) -> Value {
    return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, x,
                                   t.like(y, x.getType()));
  };
  bool segmented = p.dot.getNumOperands() > 2;
  scf::IfOp active;
  if (segmented) {
    auto bounds = p.dot.getOperand(2);
    auto next = b.create<arith::AddIOp>(loc, gp, t.integer(1));
    auto at = [&](Value i) -> Value {
      auto raw = t.read(bounds, t.address(bounds, i, {-1}));
      return b.create<arith::ExtSIOp>(loc, b.getI64Type(), raw);
    };
    groupStart = at(gp);
    end = at(next);
    active = b.create<scf::IfOp>(
        loc, less(b.create<arith::AddIOp>(loc, first, groupStart), end), false);
    b.setInsertionPointToStart(active.thenBlock());
    row = b.create<arith::AddIOp>(loc, row, t.like(groupStart, row.getType()));
  }
  Value validRow = less(row, end), validCol = less(col, t.integer(n)),
        dst = row, src = row;
  Value multiplicity;
  if (p.routed()) {
    auto route = p.dot.getOperand(3), y = p.domain.getOperand(0);
    auto ids = t.read(route, t.address(route, row, {0}), validRow);
    dst = b.create<arith::ExtSIOp>(loc, row.getType(), ids);
    multiplicity = t.like(t.integer(t.shape(y)[1]), row.getType());
    src = b.create<arith::DivUIOp>(loc, dst, multiplicity);
  }
  Value a = p.dot.getOperand(0), rhs = p.dot.getOperand(1);
  Type element = cast<MemRefType>(a.getType()).getElementType().isInteger(8)
                     ? Type(b.getI32Type())
                     : Type(b.getF32Type());
  auto mt = RankedTensorType::get({bm, bn}, element);
  auto zero = b.create<arith::ConstantOp>(
      loc, mt, DenseElementsAttr::get(mt, b.getZeroAttr(element)));
  auto bound = p.fold.getAttrOfType<IntegerAttr>("bound").getInt();
  auto step = p.fold.getAttrOfType<IntegerAttr>("step").getInt();
  auto limit =
      b.create<arith::ConstantIndexOp>(loc, llvm::divideCeil(bound, step));
  auto lo = b.create<arith::ConstantIndexOp>(loc, 0),
       one = b.create<arith::ConstantIndexOp>(loc, 1);
  auto loop = b.create<scf::ForOp>(loc, lo, limit, one, ValueRange{zero});
  loop->setAttr(t.symbol("runtime_range"), b.getUnitAttr());
  b.setInsertionPointToStart(loop.getBody());
  auto panel =
      b.create<arith::IndexCastOp>(loc, b.getI64Type(), loop.getInductionVar());
  auto member =
      t.stepped(b.create<arith::MulIOp>(loc, panel, t.integer(step)), bk);
  auto panelAddress = [&](Value base, Value free, int fa, int ka, int ba,
                          int dim) {
    SmallVector<Value> coords(p.rank(base));
    SmallVector<int64_t> axes(p.rank(base), -1);
    coords[fa] = free;
    axes[fa] = dim;
    coords[ka] = member;
    axes[ka] = 1 - dim;
    if (ba >= 0)
      coords[ba] = gp;
    return t.address(base, coords, axes);
  };
  auto mask = [&](Value rows, Value columns, int64_t nr, int64_t nc) -> Value {
    auto maskType = t.shaped({nr, nc}, b.getI1Type());
    return b.create<arith::AndIOp>(loc, t.like(rows, maskType, 0),
                                   t.like(columns, maskType, 1));
  };
  int ak = p.axis("lhs_k"), rk = p.axis("rhs_k"), ab = p.axis("lhs_batch"),
      rb = p.axis("rhs_batch");
  auto km = less(member, t.integer(k));
  auto left = native ? native(t, p, true, gp, panel) : Value();
  if (!left)
    left = t.read(a, panelAddress(a, src, p.freeAxis(a, ak, ab), ak, ab, 0),
                  mask(validRow, km, bm, bk));
  auto right = native ? native(t, p, false, gp, panel) : Value();
  if (!right)
    right =
        t.read(rhs, panelAddress(rhs, col, p.freeAxis(rhs, rk, rb), rk, rb, 1),
               mask(km, validCol, bk, bn));
  auto acc = t.dot(left, right, false, loop.getRegionIterArgs()[0]);
  b.create<scf::YieldOp>(loc, acc);
  b.setInsertionPointAfter(loop);
  IRMapping map;
  map.map(p.fold.getResult(0), loop.getResult(0));
  for (auto &op : p.domain.getRegion(0).front()) {
    auto name = op.getName().stripDialect();
    if (name == "add_columns") {
      auto bias = op.getOperand(1), v = map.lookup(op.getOperand(0));
      auto lane = t.read(bias, t.address(bias, col, {0}), validCol);
      Value values = t.like(lane, v.getType(), 1);
      Value added = getElementTypeOrSelf(v.getType()).isInteger(32)
                        ? Value(b.create<arith::AddIOp>(loc, v, values))
                        : Value(b.create<arith::AddFOp>(loc, v, values));
      map.map(op.getResult(0), added);
    } else if (name == "convert_matrix") {
      Value v = map.lookup(op.getOperand(0));
      auto to = mt.clone(getElementTypeOrSelf(op.getResult(0).getType()));
      if (v.getType() != to) {
        auto from = getElementTypeOrSelf(v.getType());
        if (from.isInteger(32))
          v = b.create<arith::SIToFPOp>(loc, to, v);
        else if (to.getElementType().isInteger(32))
          v = b.create<arith::FPToSIOp>(loc, to, v);
        else if (from.getIntOrFloatBitWidth() < to.getElementTypeBitWidth())
          v = b.create<arith::ExtFOp>(loc, to, v);
        else {
          if (from.getIntOrFloatBitWidth() == to.getElementTypeBitWidth())
            v = b.create<arith::ExtFOp>(loc, mt.clone(b.getF32Type()), v);
          v = b.create<arith::TruncFOp>(loc, to, v);
        }
      }
      map.map(op.getResult(0), v);
    } else if (name == "publish_matrix") {
      auto y = op.getOperand(0), v = map.lookup(op.getOperand(1));
      SmallVector<Value, 1> batches;
      if (!segmented && !p.routed() && p.rank(y) == 3)
        batches.push_back(gp);
      if (!segmented && !p.routed() && writeNative &&
          writeNative(t, p, y, v, batches))
        continue;
      Value addr;
      if (p.routed()) {
        Value a = b.create<arith::DivUIOp>(loc, dst, multiplicity),
              r = b.create<arith::RemUIOp>(loc, dst, multiplicity);
        addr = t.address(y, {a, r, col}, {0, 0, 1});
      } else if (p.rank(y) == 3)
        addr = t.address(y, {gp, dst, col}, {-1, 0, 1});
      else
        addr = t.address(y, {dst, col}, {0, 1});
      t.write(y, addr, v, mask(validRow, validCol, bm, bn));
    }
  }
  if (active)
    b.setInsertionPointAfter(active);
}

} // namespace
void expandMatrixProgram(func::FuncOp fn, StringRef unit,
                         ScaledPanelLowering scaled, MatrixPanelRead native,
                         WindowPanelLowering window, MatrixPanelWrite write,
                         bool flatRecordBitPlanes) {
  fn.getContext()
      ->loadDialect<arith::ArithDialect, scf::SCFDialect, math::MathDialect>();
  auto &root = fn.front().front();
  if (intentdsl::kir::matrixDomain(fn)) {
    intentdsl::block::MatrixAccess p(fn);
    Tiles tiles(fn, "matrix", unit);
    if (p.dot.getName().stripDialect() == "contract")
      matrixTiles(tiles, p, native, write);
    else
      encodedTiles(tiles, p, scaled, native, window, write,
                   flatRecordBitPlanes);
    tiles.op("end");
    root.erase();
  }
}
} // namespace intentdsl::block
