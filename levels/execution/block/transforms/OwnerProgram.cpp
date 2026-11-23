#include "intentdsl/levels/execution/block/ir/OwnerProgram.h"
#include "OwnerTiles.h"
namespace intentdsl::block {
namespace {
void scoreTiles(Tiles &t, Operation &old) {
  auto &b = t.b;
  auto loc = t.loc;
  auto height = old.getAttrOfType<IntegerAttr>("block.owner_tile").getInt();
  IRMapping map;
  Value pid = t.pid(0);
  for (auto &op : old.getRegion(0).front().without_terminator()) {
    auto n = op.getName().stripDialect();
    Value result;
    auto a = [&](unsigned i) { return map.lookupOrDefault(op.getOperand(i)); };
    if (n == "score_contract") {
      auto q = a(0), k = a(1);
      auto qs = t.shape(q), ks = t.shape(k);
      auto tiles = t.integer(qs[2] / height), heads = t.integer(qs[1]);
      auto group = b.create<arith::DivUIOp>(loc, pid, tiles);
      Value batch = b.create<arith::DivUIOp>(loc, group, heads),
            h = b.create<arith::RemUIOp>(loc, group, heads);
      auto kv = b.create<arith::DivUIOp>(
          loc, h, t.integer(op.getAttrOfType<IntegerAttr>("group").getInt()));
      auto block = b.create<arith::RemUIOp>(loc, pid, tiles);
      Value row = t.stepped(
          b.create<arith::MulIOp>(loc, block, t.integer(height)), height);
      map.map(op.getOperand(3), batch);
      map.map(op.getOperand(4), h);
      map.map(op.getOperand(5), row);
      auto d = t.lanes(qs[3]), keys = t.lanes(ks[2]);
      auto lhs = t.read(q, t.address(q, {batch, h, row, d}, {-1, -1, 0, 1}));
      auto rhs = t.read(k, t.address(k, {batch, kv, keys, d}, {-1, -1, 0, 1}));
      auto dot = t.dot(lhs, rhs, true);
      auto scale =
          b.create<arith::MulFOp>(loc, a(2), t.scalar(1.4426950408889634f));
      result = b.create<arith::MulFOp>(loc, dot, t.like(scale, dot.getType()));
    } else if (n == "causal_exp") {
      auto scores = a(0);
      auto mt = cast<RankedTensorType>(scores.getType());
      auto wide = mt.clone(b.getI64Type());
      auto row = t.like(a(1), wide, 0),
           col = t.like(t.lanes(mt.getDimSize(1)), wide, 1);
      auto valid =
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, row, col);
      auto masked = b.create<arith::SelectOp>(loc, valid, scores,
                                              t.like(t.scalar(-INFINITY), mt));
      auto max = t.reduce(masked, "max");
      auto scaled = b.create<arith::SubFOp>(loc, scores, t.like(max, mt, 0));
      auto weight = b.create<arith::SelectOp>(
          loc, valid, b.create<math::Exp2Op>(loc, scaled),
          t.like(t.scalar(0), mt));
      map.map(op.getResult(0), b.create<arith::TruncFOp>(
                                   loc, mt.clone(b.getBF16Type()), weight));
      map.map(op.getResult(1), t.reduce(weight, "sum"));
      continue;
    } else if (n == "value_contract") {
      auto v = a(1);
      auto size = t.shape(v);
      auto kv = b.create<arith::DivUIOp>(
          loc, a(3),
          t.integer(op.getAttrOfType<IntegerAttr>("group").getInt()));
      auto fragment =
          t.read(v, t.address(v, {a(2), kv, t.lanes(size[2]), t.lanes(size[3])},
                              {-1, -1, 0, 1}));
      result = t.dot(a(0), fragment, false);
    } else if (n == "normalize_store") {
      auto sum = a(1), v = a(2), matrix = a(0);
      auto size = t.shape(v);
      auto inv =
          b.create<arith::DivFOp>(loc, t.like(t.scalar(1), sum.getType()), sum);
      auto normalized = b.create<arith::MulFOp>(
          loc, matrix, t.like(inv, matrix.getType(), 0));
      auto cast = b.create<arith::TruncFOp>(
          loc,
          mlir::cast<RankedTensorType>(normalized.getType())
              .clone(b.getBF16Type()),
          normalized);
      t.write(
          v, t.address(v, {a(3), a(4), a(5), t.lanes(size[3])}, {-1, -1, 0, 1}),
          cast);
      continue;
    }
    map.map(op.getResult(0), result);
  }
}
} // namespace
void expandOwnerProgram(func::FuncOp fn, StringRef unit, OwnerAccess access) {
  fn.getContext()
      ->loadDialect<arith::ArithDialect, scf::SCFDialect, math::MathDialect>();
  auto &root = fn.front().front();
  if (root.hasAttr("block.score_panel")) {
    Tiles tiles(fn, "owners", unit);
    scoreTiles(tiles, root);
    tiles.op("end");
    root.erase();
  } else if (intentdsl::kir::orderedOwners(fn) &&
             !root.hasAttr("block.column_prefixes")) {
    OwnerTiles tiles(fn, root, unit, access);
    (void)tiles.build();
    root.erase();
  }
}
} // namespace intentdsl::block
