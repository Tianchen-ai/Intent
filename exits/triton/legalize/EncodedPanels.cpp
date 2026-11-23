#include "TileBuilder.h"
#include "intentdsl/levels/algorithm/kir/ir/PanelMap.h"
#include "intentdsl/levels/algorithm/kir/ir/Storage.h"

namespace CI_UNIT_NAMESPACE {
using namespace mlir;
namespace {
struct Panel {
  Tiles &t;
  intentdsl::block::MatrixAccess &p;
  OpBuilder &b;
  Location loc;
  Value row, col, validRow, validCol, gp;
  Panel(Tiles &t, intentdsl::block::MatrixAccess &p)
      : t(t), p(p), b(t.b), loc(t.loc) {
    row = axis(0, p.tile[0]);
    col = axis(1, p.tile[1]);
    gp = t.pid(2);
    validRow = cmp(arith::CmpIPredicate::slt, row, t.integer(p.shape[0]));
    validCol = cmp(arith::CmpIPredicate::slt, col, t.integer(p.shape[1]));
  }
  Value axis(int dim, int w) {
    return t.stepped(b.create<arith::MulIOp>(loc, t.pid(dim), t.integer(w)), w);
  }
  Value lit(int n, Type type) {
    auto e = t.elem(type);
    auto c = b.create<arith::ConstantOp>(loc, b.getIntegerAttr(e, n));
    return t.like(c, type);
  }
  Value cmp(arith::CmpIPredicate pred, Value a, Value c) {
    return b.create<arith::CmpIOp>(loc, pred, a, t.like(c, a.getType()));
  }
  Value castTo(StringRef operation, Value a, Type e) {
    auto type = a.getType();
    if (auto r = dyn_cast<RankedTensorType>(type))
      type = r.clone(e);
    else
      type = e;
    OperationState s(loc, operation);
    s.addOperands(a);
    s.addTypes(type);
    return b.create(s)->getResult(0);
  }
  Value pairMask(Value r, Value c) {
    auto ty = t.shaped({cast<RankedTensorType>(r.getType()).getDimSize(0),
                        cast<RankedTensorType>(c.getType()).getDimSize(0)},
                       b.getI1Type());
    return b.create<arith::AndIOp>(loc, t.like(r, ty, 0), t.like(c, ty, 1));
  }
  Value fold(int64_t groups, Value init,
             std::function<Value(Value, Value)> body) {
    auto loop = b.create<scf::ForOp>(loc, t.integer(0), t.integer(groups),
                                     t.integer(1), ValueRange{init});
    loop->setAttr("triton_src.runtime_range", b.getUnitAttr());
    b.setInsertionPointToStart(loop.getBody());
    auto next = body(loop.getInductionVar(), loop.getRegionIterArgs()[0]);
    b.create<scf::YieldOp>(loc, next);
    b.setInsertionPointAfter(loop);
    return loop.getResult(0);
  }
  Value choice(Value flag, Value init, std::function<Value()> yes,
               std::function<Value()> no) {
    auto branch =
        b.create<scf::IfOp>(loc, TypeRange{init.getType()}, flag, true);
    b.setInsertionPointToStart(branch.thenBlock());
    b.create<scf::YieldOp>(loc, yes());
    b.setInsertionPointToStart(branch.elseBlock());
    b.create<scf::YieldOp>(loc, no());
    b.setInsertionPointAfter(branch);
    return branch.getResult(0);
  }
  Value zero(ArrayRef<int64_t> dims) {
    return t.like(t.scalar(0), t.shaped(dims, b.getF32Type()));
  }
  Value strict(StringRef code, Value x, Value y) {
    OperationState s(loc, code);
    s.addOperands({x, y});
    s.addTypes(x.getType());
    s.addAttribute("triton_src.round_nearest", b.getUnitAttr());
    return b.create(s)->getResult(0);
  }
  Value record() {
    auto q4 = cast<intentdsl::kir::Q4KType>(
                  p.dot.getAttrOfType<TypeAttr>("kir.storage").getValue())
                  .getLayout();
    auto *encoder = p.dot.getOperand(1).getDefiningOp();
    auto q8 = cast<intentdsl::kir::Q8KType>(
                  encoder->getAttrOfType<TypeAttr>("kir.storage").getValue())
                  .getLayout();
    auto w = p.dot.getOperand(0),
         x = p.dot.getOperand(1).getDefiningOp()->getOperand(0);
    return fold(
        p.shape[2] / q4.elements, zero({p.tile[0]}), [&](Value g, Value acc) {
          auto ix = t.lanes(q8.elements);
          auto v = t.read(x, t.address(x, {g, ix}, {-1, 0}));
          auto abs = b.create<math::AbsFOp>(loc, v);
          auto peak = t.reduce(abs, "max", 0);
          auto eq = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, abs,
                                            t.like(peak, v.getType()));
          auto first =
              t.reduce(b.create<arith::SelectOp>(
                           loc, eq, ix, lit(q4.elements, ix.getType())),
                       "min", 0);
          auto extreme =
              t.reduce(b.create<arith::SelectOp>(
                           loc, cmp(arith::CmpIPredicate::eq, ix, first), v,
                           t.like(t.scalar(0), v.getType())),
                       "sum", 0);
          auto empty = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ,
                                               peak, t.scalar(0));
          auto inv = strict(
              "arith.divf", t.scalar(-127),
              b.create<arith::SelectOp>(loc, empty, t.scalar(1), extreme));
          auto ds = b.create<arith::SelectOp>(
              loc, empty, t.scalar(0), strict("arith.divf", t.scalar(1), inv));
          auto rounded = b.create<math::RoundEvenOp>(
              loc, strict("arith.mulf", t.like(inv, v.getType()), v));
          auto clamp = b.create<arith::MinNumFOp>(
              loc,
              b.create<arith::MaxNumFOp>(loc, rounded,
                                         t.like(t.scalar(-128), v.getType())),
              t.like(t.scalar(127), v.getType()));
          auto q = castTo("arith.fptosi", clamp, b.getI32Type());
          auto byte = [&](Value index, int dim) {
            auto ptr = t.address(w, {row, g, index}, {0, -1, dim});
            auto mask = t.like(
                validRow,
                cast<RankedTensorType>(ptr.getType()).clone(b.getI1Type()), 0);
            return castTo("arith.extui", t.read(w, ptr, mask), b.getI32Type());
          };
          auto half = [&](int offset) {
            return t
                .op("from_le_half",
                    {byte(t.integer(offset), -1),
                     byte(t.integer(offset + 1), -1)},
                    acc.getType())
                ->getResult(0);
          };
          auto d = half(q4.d), dm = half(q4.dmin);
          auto j =
              b.create<arith::DivUIOp>(loc, ix, lit(q4.group, ix.getType()));
          auto slot = b.create<arith::RemUIOp>(
              loc, j, lit(q4.halfGroups, ix.getType()));
          auto field = [&](int base) {
            return byte(
                b.create<arith::AddIOp>(loc, slot, lit(base, slot.getType())),
                1);
          };
          auto low = field(q4.factors),
               high = field(q4.factors + q4.halfGroups),
               packed = field(q4.factors + 2 * q4.halfGroups);
          Type ty = low.getType(),
               maskType = cast<RankedTensorType>(ty).clone(b.getI1Type());
          auto firstHalf = t.like(
              cmp(arith::CmpIPredicate::slt, j, t.integer(q4.halfGroups)),
              maskType, 1);
          auto nibMask = lit((1 << q4.nibbleBits) - 1, ty),
               bits = lit(q4.nibbleBits, ty);
          auto coeff = [&](Value x, Value upper) {
            auto top = b.create<arith::ShLIOp>(
                loc, b.create<arith::ShRUIOp>(loc, x, lit(q4.scaleBits, ty)),
                bits);
            return b.create<arith::SelectOp>(
                loc, firstHalf,
                b.create<arith::AndIOp>(loc, x,
                                        lit((1 << q4.scaleBits) - 1, ty)),
                b.create<arith::OrIOp>(loc, upper, top));
          };
          auto sc = coeff(low, b.create<arith::AndIOp>(loc, packed, nibMask));
          auto mn = coeff(high, b.create<arith::ShRUIOp>(loc, packed, bits));
          auto offset = b.create<arith::AddIOp>(
              loc,
              b.create<arith::MulIOp>(
                  loc, b.create<arith::DivUIOp>(loc, j, lit(2, ix.getType())),
                  lit(q4.group, ix.getType())),
              b.create<arith::RemUIOp>(loc, ix, lit(q4.group, ix.getType())));
          auto raw = byte(b.create<arith::AddIOp>(loc, offset,
                                                  lit(q4.packed, ix.getType())),
                          1);
          auto shift = b.create<arith::MulIOp>(
              loc, b.create<arith::RemUIOp>(loc, j, lit(2, ix.getType())),
              lit(q4.nibbleBits, ix.getType()));
          auto nib = b.create<arith::AndIOp>(
              loc,
              b.create<arith::ShRUIOp>(
                  loc, raw,
                  t.like(castTo("arith.trunci", shift, b.getI32Type()), ty, 1)),
              nibMask);
          auto activation = t.like(q, ty, 1);
          auto sum = t.reduce(
              b.create<arith::MulIOp>(
                  loc, b.create<arith::MulIOp>(loc, sc, nib), activation),
              "sum");
          auto correction =
              t.reduce(b.create<arith::MulIOp>(loc, mn, activation), "sum");
          auto residual = strict(
              "arith.subf",
              strict("arith.mulf", d,
                     castTo("arith.sitofp", sum, b.getF32Type())),
              strict("arith.mulf", dm,
                     castTo("arith.sitofp", correction, b.getF32Type())));
          return b
              .create<arith::AddFOp>(
                  loc, acc,
                  strict("arith.mulf", t.like(ds, acc.getType()), residual))
              .getResult();
        });
  }
  Value scaled() {
    auto mx = cast<intentdsl::kir::MX8Type>(
                  p.dot.getAttrOfType<TypeAttr>("kir.storage").getValue())
                  .getLayout();
    auto a = p.dot.getOperand(0), sa = p.dot.getOperand(1),
         rhs = p.dot.getOperand(2);
    intentdsl::kir::RightScale right(p.dot);
    auto init = zero({p.tile[0], p.tile[1]});
    auto groups = p.shape[2] / mx.group;
    auto rightScale = [&](Value g, bool all) {
      SmallVector<Value> coords(2);
      SmallVector<int64_t> dims(2);
      coords[right.column] = col;
      dims[right.column] = all ? 1 : 0;
      coords[right.group] = g;
      dims[right.group] = all ? 0 : -1;
      return t.address(right.view, coords, dims);
    };
    auto inRange = [&](Value x, int lo, int hi) {
      auto min = t.reduce(x, "min", 0), max = t.reduce(x, "max", 0);
      return b.create<arith::AndIOp>(
          loc, cmp(arith::CmpIPredicate::sge, min, lit(lo, min.getType())),
          cmp(arith::CmpIPredicate::slt, max, lit(hi + 1, max.getType())));
    };
    auto readScale = [&](Value base, Value ptr, Value valid, int axis) {
      auto ix = cast<RankedTensorType>(ptr.getType());
      auto mask = t.like(valid, ix.clone(b.getI1Type()), axis);
      auto v = castTo("arith.extui", t.read(base, ptr, mask), b.getI32Type());
      return b.create<arith::SelectOp>(loc, mask, v, lit(mx.bias, v.getType()))
          .getResult();
    };
    auto general = [&]() {
      return fold(groups, init, [&](Value g, Value acc) {
        auto ax = readScale(sa, t.address(sa, {row, g}, {0, -1}), validRow, 0),
             bx = readScale(right.view, rightScale(g, false), validCol, 0);
        auto k = t.lanes(mx.group);
        auto av =
            t.read(a, t.address(a, {row, g, k}, {0, -1, 1}),
                   t.like(validRow,
                          t.shaped({p.tile[0], mx.group}, b.getI1Type()), 0));
        auto bv =
            t.read(rhs, t.address(rhs, {g, k, col}, {-1, 0, 1}),
                   t.like(validCol,
                          t.shaped({mx.group, p.tile[1]}, b.getI1Type()), 1));
        auto rescale = [&](Value val, Value scale, int axis) -> Value {
          auto fp = castTo("arith.extf", val, b.getF32Type());
          auto factor = t.op("e8m0", scale,
                             cast<RankedTensorType>(scale.getType())
                                 .clone(b.getF32Type()))
                            ->getResult(0);
          return b.create<arith::MulFOp>(loc, fp,
                                         t.like(factor, fp.getType(), axis));
        };
        auto x = rescale(av, ax, 0), y = rescale(bv, bx, 1);
        auto range =
            p.dot.getPrevNode()->getAttrOfType<DenseI64ArrayAttr>("interval");
        auto fast =
            b.create<arith::AndIOp>(loc, inRange(ax, range[0], range[1]),
                                    inRange(bx, range[0], range[1]));
        return choice(
            fast, acc,
            [&]() {
              return t.dot(castTo("arith.truncf", x, b.getBF16Type()),
                           castTo("arith.truncf", y, b.getBF16Type()), false,
                           acc);
            },
            [&]() { return t.dot(x, y, false, acc); });
      });
    };
    if (!p.dot.getAttrOfType<BoolAttr>("triton_src.neutral_scale_dispatch")
             .getValue())
      return general();
    auto g = t.lanes(groups);
    auto ax = readScale(sa, t.address(sa, {row, g}, {0, 1}), validRow, 0),
         bx = readScale(right.view, rightScale(g, true), validCol, 1);
    auto unit = [&](Value s) {
      auto eq = cmp(arith::CmpIPredicate::eq, s, lit(mx.bias, s.getType()));
      return t.reduce(t.reduce(eq, "min", 1), "min", 0);
    };
    return choice(
        b.create<arith::AndIOp>(loc, unit(ax), unit(bx)), init,
        [&]() {
          return fold(
              llvm::divideCeil(p.shape[2], p.tile[2]), init,
              [&](Value it, Value acc) {
                auto k = t.stepped(
                    b.create<arith::MulIOp>(loc, it, t.integer(p.tile[2])),
                    p.tile[2]);
                auto group = b.create<arith::DivUIOp>(
                    loc, k, lit(mx.group, k.getType()));
                auto lane = b.create<arith::RemUIOp>(
                    loc, k, lit(mx.group, k.getType()));
                auto active =
                    cmp(arith::CmpIPredicate::slt, k, t.integer(p.shape[2]));
                auto av = t.read(a, t.address(a, {row, group, lane}, {0, 1, 1}),
                                 pairMask(validRow, active));
                auto bv =
                    t.read(rhs, t.address(rhs, {group, lane, col}, {0, 0, 1}),
                           pairMask(active, validCol));
                auto sum = t.dot(av, bv, false, acc);
                sum.getDefiningOp()->setAttr("triton_src.exact_acc",
                                             b.getUnitAttr());
                return sum;
              });
        },
        general);
  }
  Value window(Value &batch, Value &chunk, Value &head) {
    auto x = p.dot.getOperand(0), basis = p.dot.getOperand(1),
         dt = p.dot.getOperand(2), decay = p.dot.getOperand(3);
    int64_t heads = std::max<int64_t>(1, t.shape(dt)[1]),
            chunks = std::max<int64_t>(1, t.shape(dt)[2]), k = p.shape[2],
            bk = p.tile[2];
    auto headIndex = b.create<arith::DivUIOp>(loc, gp, t.integer(heads));
    head = b.create<arith::RemUIOp>(loc, gp, t.integer(heads));
    batch = b.create<arith::DivUIOp>(loc, headIndex, t.integer(chunks));
    chunk = b.create<arith::RemUIOp>(loc, headIndex, t.integer(chunks));
    Value group =
        b.create<arith::DivUIOp>(loc, head, t.integer(p.axis("head_group")));
    return fold(
        llvm::divideCeil(k, bk), zero({p.tile[0], p.tile[1]}),
        [&](Value it, Value carry) {
          Value member =
              t.stepped(b.create<arith::MulIOp>(loc, it, t.integer(bk)), bk);
          auto start = b.create<arith::MulIOp>(loc, chunk, t.integer(k));
          auto seq = b.create<arith::AddIOp>(
              loc, t.like(start, member.getType()), member);
          auto valid = cmp(arith::CmpIPredicate::slt, member, t.integer(k));
          auto lhs =
              t.read(x, t.address(x, {batch, seq, head, row}, {-1, 1, -1, 0}),
                     pairMask(validRow, valid));
          auto rhs = t.read(
              basis, t.address(basis, {batch, seq, group, col}, {-1, 0, -1, 1}),
              pairMask(valid, validCol));
          auto readCoefficient = [&](Value tensor) {
            return t.read(tensor,
                          t.address(tensor, {batch, head, chunk, member},
                                    {-1, -1, -1, 0}),
                          valid);
          };
          auto last = t.read(
              decay, t.address(decay, {batch, head, chunk, t.integer(k - 1)},
                               {-1, -1, -1, -1}));
          auto &body = p.dot.getRegion(0).front();
          IRMapping mapping;
          auto ty = cast<RankedTensorType>(rhs.getType());
          for (auto [arg, val] :
               llvm::zip(body.getArguments(),
                         SmallVector<Value>{rhs, readCoefficient(dt),
                                            readCoefficient(decay), last}))
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
          auto result = mapping.lookup(body.back().getOperand(0));
          auto guard = t.like(valid, ty.clone(b.getI1Type()), 0);
          auto nil = b.create<arith::ConstantOp>(
              loc, ty,
              DenseElementsAttr::get(ty, b.getZeroAttr(ty.getElementType())));
          return t.dot(lhs, b.create<arith::SelectOp>(loc, guard, result, nil),
                       false, carry);
        });
  }
  void run() {
    bool records = p.dot.getName().stripDialect() == "q4k_dot";
    bool hasWindow = p.dot.getName().stripDialect() == "window_contract";
    Value batch, chunk, head;
    Value acc = records     ? record()
                : hasWindow ? window(batch, chunk, head)
                            : scaled();
    IRMapping map;
    map.map(p.fold.getResult(0), acc);
    for (auto &op : p.domain.getRegion(0).front()) {
      auto code = op.getName().stripDialect();
      if (code == "add_columns") {
        auto view = op.getOperand(1), old = map.lookup(op.getOperand(0));
        auto bias = t.read(view, t.address(view, records ? row : col, {0}),
                           records ? validRow : validCol);
        map.map(op.getResult(0), b.create<arith::AddFOp>(
                                     loc, old, t.like(bias, old.getType(), 1)));
      } else if (code == "convert_matrix") {
        auto v = map.lookup(op.getOperand(0));
        Type to = t.elem(op.getResult(0).getType());
        map.map(op.getResult(0),
                t.elem(v.getType()) == to
                    ? v
                    : castTo(to.isInteger(32) ? "arith.fptosi" : "arith.truncf",
                             v, to));
      } else if (code == "publish_matrix") {
        auto y = op.getOperand(0), value = map.lookup(op.getOperand(1));
        auto ptr = records     ? t.address(y, row, {0})
                   : hasWindow ? t.address(y, {batch, chunk, head, row, col},
                                           {-1, -1, -1, 0, 1})
                               : t.address(y, {row, col}, {0, 1});
        t.write(y, ptr, value,
                records ? validRow : pairMask(validRow, validCol));
      }
    }
  }
};
} // namespace
void encodedTiles(Tiles &t, intentdsl::block::MatrixAccess &access) {
  Panel(t, access).run();
}
} // namespace CI_UNIT_NAMESPACE
