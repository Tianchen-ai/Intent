#include "intentdsl/levels/algorithm/kir/ir/PanelMap.h"
#include "intentdsl/levels/algorithm/kir/ir/Storage.h"
#include "intentdsl/levels/execution/block/ir/MatrixProgram.h"

namespace intentdsl::block {
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
    loop->setAttr(t.symbol("runtime_range"), b.getUnitAttr());
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
    s.addAttribute(t.symbol("round_nearest"), b.getUnitAttr());
    return b.create(s)->getResult(0);
  }
  Value q8kArgmaxKey(Value v, Value ix, int64_t blockElems) {
    auto vTy = cast<RankedTensorType>(v.getType());
    auto i32Ty = vTy.clone(b.getI32Type());
    auto i64Ty = cast<RankedTensorType>(ix.getType());
    Value raw = b.create<arith::BitcastOp>(loc, i32Ty, v);
    Value absBits = b.create<arith::AndIOp>(loc, raw, lit(0x7fffffff, i32Ty));
    Value isNeg = cmp(arith::CmpIPredicate::slt, raw, lit(0, i32Ty));
    Value abs64 = castTo("arith.extsi", absBits, b.getI64Type());
    Value sign64 = b.create<arith::SelectOp>(loc, isNeg, lit(1, i64Ty),
                                             lit(0, i64Ty));
    Value revIx = b.create<arith::SubIOp>(loc, lit(blockElems - 1, i64Ty), ix);
    return b.create<arith::OrIOp>(
        loc,
        b.create<arith::OrIOp>(
            loc, b.create<arith::ShLIOp>(loc, abs64, lit(9, i64Ty)),
            b.create<arith::ShLIOp>(loc, revIx, lit(1, i64Ty))),
        sign64);
  }
  std::pair<Value, Value> q8kScaleFromKey(Value bestKey, float maxQ) {
    Value magBits = castTo(
        "arith.trunci",
        b.create<arith::ShRUIOp>(loc, bestKey, lit(9, bestKey.getType())),
        b.getI32Type());
    Value bestSign = castTo(
        "arith.trunci",
        b.create<arith::AndIOp>(loc, bestKey, lit(1, bestKey.getType())),
        b.getI32Type());
    Value magF = b.create<arith::BitcastOp>(loc, b.getF32Type(), magBits);
    Value negF = b.create<arith::SubFOp>(loc, t.scalar(0), magF);
    Value extreme = b.create<arith::SelectOp>(
        loc, cmp(arith::CmpIPredicate::ne, bestSign, lit(0, bestSign.getType())),
        negF, magF);
    Value empty = cmp(arith::CmpIPredicate::eq, magBits,
                      lit(0, magBits.getType()));
    Value inv = strict(
        "arith.divf", t.scalar(maxQ),
        b.create<arith::SelectOp>(loc, empty, t.scalar(1), extreme));
    Value ds = b.create<arith::SelectOp>(
        loc, empty, t.scalar(0), strict("arith.divf", t.scalar(1), inv));
    return {inv, ds};
  }
  Value q8kQuantizeVec(Value v, Value inv, float loQ, float hiQ) {
    auto rounded = b.create<math::RoundEvenOp>(
        loc, strict("arith.mulf", t.like(inv, v.getType()), v));
    auto clamp = b.create<arith::MinNumFOp>(
        loc,
        b.create<arith::MaxNumFOp>(loc, rounded,
                                   t.like(t.scalar(loQ), v.getType())),
        t.like(t.scalar(hiQ), v.getType()));
    return castTo("arith.fptosi", clamp, b.getI32Type());
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
          auto [inv, ds] = q8kScaleFromKey(
              t.reduce(q8kArgmaxKey(v, ix, q8.elements), "max", 0), -127.0f);
          auto q = q8kQuantizeVec(v, inv, -128.0f, 127.0f);
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
    if (!p.dot.getAttrOfType<BoolAttr>(t.symbol("neutral_scale_dispatch"))
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
                sum.getDefiningOp()->setAttr(t.symbol("exact_acc"),
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
  Value laneShift(Value subLane, const intentdsl::kir::QuantFormatSpec &spec) {
    auto shTy = cast<RankedTensorType>(subLane.getType()).clone(b.getI32Type());
    if (spec.packFactor == 2) {
      Value sub32 = castTo("arith.trunci", subLane, b.getI32Type());
      int diff = spec.laneBitShifts[1] - spec.laneBitShifts[0];
      Value scaled =
          diff == 1 ? sub32 : b.create<arith::MulIOp>(loc, sub32, lit(diff, shTy));
      return spec.laneBitShifts[0] == 0
                 ? scaled
                 : b.create<arith::AddIOp>(loc, scaled,
                                           lit(spec.laneBitShifts[0], shTy));
    }
    if (spec.packFactor == 8) {
      int base = spec.laneBitShifts[0];
      int s0 = spec.laneBitShifts[1] - base;
      int s1 = spec.laneBitShifts[2] - base;
      bool twoLevelAffine = true;
      for (int idx = 0; idx < 8; ++idx) {
        if (spec.laneBitShifts[idx] != base + (idx & 1) * s0 + (idx >> 1) * s1)
          twoLevelAffine = false;
      }
      if (twoLevelAffine) {
        Value sub32 = castTo("arith.trunci", subLane, b.getI32Type());
        Value odd = b.create<arith::AndIOp>(loc, sub32, lit(1, shTy));
        Value hi = b.create<arith::ShRUIOp>(loc, sub32, lit(1, shTy));
        Value term0 = b.create<arith::MulIOp>(loc, odd, lit(s0, shTy));
        Value term1 = b.create<arith::MulIOp>(loc, hi, lit(s1, shTy));
        Value sum = b.create<arith::AddIOp>(loc, term0, term1);
        return base == 0 ? sum : b.create<arith::AddIOp>(loc, sum, lit(base, shTy));
      }
    }
    Value sh = lit(spec.laneBitShifts[0], shTy);
    for (int idx = 1; idx < spec.packFactor; ++idx)
      sh = b.create<arith::SelectOp>(
          loc,
          cmp(arith::CmpIPredicate::eq, subLane,
              lit(idx, subLane.getType())),
          lit(spec.laneBitShifts[idx], shTy), sh);
    return sh;
  }
  Value decodeQuantCode(Value codeI32,
                        const intentdsl::kir::QuantFormatSpec &spec) {
    Type ty = codeI32.getType();
    Type fTy = cast<RankedTensorType>(ty).clone(b.getF32Type());
    if (spec.valueCode == intentdsl::kir::QuantValueCode::Fp4E2M1) {
      Value s = b.create<arith::AndIOp>(
          loc, b.create<arith::ShRUIOp>(loc, codeI32, lit(3, ty)), lit(1, ty));
      Value e = b.create<arith::AndIOp>(
          loc, b.create<arith::ShRUIOp>(loc, codeI32, lit(1, ty)), lit(3, ty));
      Value m = b.create<arith::AndIOp>(loc, codeI32, lit(1, ty));
      Value isSub = cmp(arith::CmpIPredicate::eq, e, lit(0, ty));
      Value safeShift = b.create<arith::SelectOp>(
          loc, isSub, lit(0, ty),
          b.create<arith::SubIOp>(loc, e, lit(1, ty)));
      Value normDouble = b.create<arith::ShLIOp>(
          loc, b.create<arith::AddIOp>(loc, m, lit(2, ty)), safeShift);
      Value magDouble =
          b.create<arith::SelectOp>(loc, isSub, m, normDouble);
      Value magF = castTo("arith.sitofp", magDouble, b.getF32Type());
      Value neg = cmp(arith::CmpIPredicate::ne, s, lit(0, ty));
      Value halfScale = b.create<arith::SelectOp>(
          loc, neg, t.like(t.scalar(-0.5f), fTy), t.like(t.scalar(0.5f), fTy));
      return b.create<arith::MulFOp>(loc, halfScale, magF);
    }
    return castTo("arith.sitofp", codeI32, b.getF32Type());
  }
  Value quantized(MatrixPanelRead readNative, bool flatRecordBitPlanes) {
    auto reshapeTile = [&](Value v, ArrayRef<int64_t> shape) {
      return t.op("reshape", v, t.shaped(shape, t.elem(v.getType())))
          ->getResult(0);
    };
    auto expandAxis = [&](Value v, ArrayRef<int64_t> shape, int64_t axis) {
      return t
          .op("expand_scale", v, t.shaped(shape, t.elem(v.getType())),
              {b.getNamedAttr("axis", b.getI64IntegerAttr(axis))})
          ->getResult(0);
    };
    const auto *spec = intentdsl::kir::quantFormatSpec(p.dot);
    assert(spec && "verified quantized contraction must carry format spec");
    if (spec->roleLayout == intentdsl::kir::QuantRoleLayout::RecordGemv) {
      Value w = p.dot.getOperand(0);
      Value x = spec->q8kActivation
                    ? p.dot.getOperand(1).getDefiningOp()->getOperand(0)
                    : p.dot.getOperand(1);
      bool exactRows = p.shape[0] > 0 && p.shape[0] % p.tile[0] == 0;
      bool quad64Path =
          flatRecordBitPlanes &&
          spec->blockElements == 256 && spec->numBitPlanes == 2 &&
          spec->subScaleOffset >= 0 && spec->subBlockElements == 16 &&
          spec->planes[0].halfStride == 64 &&
          spec->planes[0].quadByteStep == 32 &&
          spec->planes[0].quadShiftMode == 0 &&
          spec->planes[1].halfStride == 32 &&
          spec->planes[1].quadByteStep == 0 &&
          spec->planes[1].quadShiftMode == 1;
      if (quad64Path) {
        const auto &pl0 = spec->planes[0];
        const auto &pl1 = spec->planes[1];
        Value j64 = t.lanes(64);
        Type j64Ty = j64.getType();
        Value lane32 = b.create<arith::AndIOp>(loc, j64, lit(31, j64Ty));
        Value quadBit = b.create<arith::ShRUIOp>(loc, j64, lit(5, j64Ty));
        Value subIdx4 = b.create<arith::ShRUIOp>(loc, j64, lit(4, j64Ty));
        auto addOff = [&](Value base, int64_t off) -> Value {
          return off == 0
                     ? base
                     : b.create<arith::AddIOp>(loc, base, lit(off, j64Ty))
                           .getResult();
        };
        Value qlOff[2], qhOff[2], scOff[2][2], xOff[2][2];
        for (int h = 0; h < 2; ++h) {
          qlOff[h] = addOff(j64, pl0.baseByte + h * 64);
          qhOff[h] = addOff(lane32, pl1.baseByte + h * 32);
          for (int q = 0; q < 2; ++q) {
            scOff[h][q] = addOff(subIdx4, spec->subScaleOffset + h * 8 + q * 4);
            xOff[h][q] = addOff(j64, h * 128 + q * 64);
          }
        }
        Value qhSh1D[2];
        qhSh1D[0] = castTo(
            "arith.trunci",
            b.create<arith::MulIOp>(loc, quadBit, lit(pl1.shiftUnit, j64Ty)),
            b.getI32Type());
        qhSh1D[1] = castTo(
            "arith.trunci",
            b.create<arith::MulIOp>(
                loc, b.create<arith::AddIOp>(loc, quadBit, lit(2, j64Ty)),
                lit(pl1.shiftUnit, j64Ty)),
            b.getI32Type());
        int mask0 = (1 << pl0.bitWidth) - 1;
        int mask1 = (1 << pl1.bitWidth) - 1;
        return fold(
            p.shape[2] / spec->blockElements, zero({p.tile[0]}),
            [&](Value g, Value acc) {
              Value vAct[2][2], qAct[2][2];
              for (int h = 0; h < 2; ++h)
                for (int q = 0; q < 2; ++q)
                  vAct[h][q] =
                      t.read(x, t.address(x, {g, xOff[h][q]}, {-1, 0}));
              Value ds;
              if (spec->q8kActivation) {
                intentdsl::kir::Q8KRecord q8;
                Value k00 = q8kArgmaxKey(vAct[0][0], xOff[0][0], 256);
                Value k01 = q8kArgmaxKey(vAct[0][1], xOff[0][1], 256);
                Value k10 = q8kArgmaxKey(vAct[1][0], xOff[1][0], 256);
                Value k11 = q8kArgmaxKey(vAct[1][1], xOff[1][1], 256);
                Value kMax = b.create<arith::MaxSIOp>(
                    loc, b.create<arith::MaxSIOp>(loc, k00, k01),
                    b.create<arith::MaxSIOp>(loc, k10, k11));
                float maxQ = -float((1 << (q8.elementBits - 1)) - 1);
                float loQ = -float(1 << (q8.elementBits - 1));
                float hiQ = float((1 << (q8.elementBits - 1)) - 1);
                auto [inv, dsVal] =
                    q8kScaleFromKey(t.reduce(kMax, "max", 0), maxQ);
                ds = dsVal;
                for (int h = 0; h < 2; ++h)
                  for (int q = 0; q < 2; ++q)
                    qAct[h][q] = q8kQuantizeVec(vAct[h][q], inv, loQ, hiQ);
              }
              auto byte = [&](Value index, int dim) {
                auto ptr = t.address(w, {row, g, index}, {0, -1, dim});
                Value loaded;
                if (exactRows) {
                  loaded = t.read(w, ptr);
                } else {
                  auto mask = t.like(
                      validRow,
                      cast<RankedTensorType>(ptr.getType()).clone(b.getI1Type()),
                      0);
                  loaded = t.read(w, ptr, mask);
                }
                auto u = castTo("arith.extui", loaded, b.getI32Type());
                return b.create<arith::AndIOp>(loc, u, lit(255, u.getType()))
                    .getResult();
              };
              auto halfAt = [&](int offset) {
                return t
                    .op("from_le_half",
                        {byte(t.integer(offset), -1),
                         byte(t.integer(offset + 1), -1)},
                        acc.getType())
                    ->getResult(0);
              };
              Type tileI32Ty = t.shaped({p.tile[0], 64}, b.getI32Type());
              Value sumLane;
              for (int h = 0; h < 2; ++h) {
                Value ql = byte(qlOff[h], 1);
                Value qh = byte(qhOff[h], 1);
                for (int q = 0; q < 2; ++q) {
                  Value qlShift =
                      q == 0 ? ql
                             : b.create<arith::ShRUIOp>(
                                    loc, ql, lit(pl0.shiftUnit, tileI32Ty))
                                   .getResult();
                  Value p0 = b.create<arith::AndIOp>(loc, qlShift,
                                                     lit(mask0, tileI32Ty));
                  if (pl0.destBitOffset > 0)
                    p0 = b.create<arith::ShLIOp>(
                        loc, p0, lit(pl0.destBitOffset, tileI32Ty));
                  Value p1 = b.create<arith::AndIOp>(
                      loc,
                      b.create<arith::ShRUIOp>(
                          loc, qh, t.like(qhSh1D[q], tileI32Ty, 1)),
                      lit(mask1, tileI32Ty));
                  if (pl1.destBitOffset > 0)
                    p1 = b.create<arith::ShLIOp>(
                        loc, p1, lit(pl1.destBitOffset, tileI32Ty));
                  Value code = b.create<arith::OrIOp>(loc, p0, p1);
                  if (spec->zeroBias != 0)
                    code = b.create<arith::SubIOp>(
                        loc, code, lit(spec->zeroBias, tileI32Ty));
                  Value scU8 = byte(scOff[h][q], 1);
                  Value scS8 = b.create<arith::SubIOp>(
                      loc,
                      b.create<arith::XOrIOp>(loc, scU8, lit(128, tileI32Ty)),
                      lit(128, tileI32Ty));
                  Value wI32 = b.create<arith::MulIOp>(loc, code, scS8);
                  if (spec->q8kActivation) {
                    Value prod = b.create<arith::MulIOp>(
                        loc, wI32, t.like(qAct[h][q], tileI32Ty, 1));
                    sumLane =
                        sumLane
                            ? b.create<arith::AddIOp>(loc, sumLane, prod)
                                  .getResult()
                            : prod;
                  } else {
                    Value fW = castTo("arith.sitofp", wI32, b.getF32Type());
                    Value prod = b.create<arith::MulFOp>(
                        loc, fW, t.like(vAct[h][q], fW.getType(), 1));
                    sumLane =
                        sumLane
                            ? b.create<arith::AddFOp>(loc, sumLane, prod)
                                  .getResult()
                            : prod;
                  }
                }
              }
              Value d = spec->scaleCode ==
                                intentdsl::kir::QuantScaleCode::LeHalfAtOffset
                            ? halfAt(spec->superScaleOffset)
                            : t.like(t.scalar(1.0f), acc.getType());
              if (spec->q8kActivation) {
                Value sumI = t.reduce(sumLane, "sum", 1);
                Value scaledSum = strict(
                    "arith.mulf", d,
                    castTo("arith.sitofp", sumI, b.getF32Type()));
                return b
                    .create<arith::AddFOp>(
                        loc, acc,
                        strict("arith.mulf", t.like(ds, acc.getType()),
                               scaledSum))
                    .getResult();
              }
              Value sumF = t.reduce(sumLane, "sum", 1);
              return b
                  .create<arith::AddFOp>(loc, acc,
                                         strict("arith.mulf", d, sumF))
                  .getResult();
            });
      }
      auto ix = t.lanes(spec->blockElements);
      Value half = b.create<arith::DivUIOp>(loc, ix, lit(128, ix.getType()));
      Value inHalf = b.create<arith::RemUIOp>(loc, ix, lit(128, ix.getType()));
      Value quad = b.create<arith::DivUIOp>(loc, inHalf, lit(32, ix.getType()));
      Value lane = b.create<arith::RemUIOp>(loc, inHalf, lit(32, ix.getType()));
      Value quadOdd = b.create<arith::RemUIOp>(loc, quad, lit(2, ix.getType()));
      Value quadHalf = b.create<arith::DivUIOp>(loc, quad, lit(2, ix.getType()));
      auto makeRangeOff = [&](int baseOffset, int64_t count) -> Value {
        Value idx = t.lanes(count);
        return baseOffset == 0
                   ? idx
                   : b.create<arith::AddIOp>(loc, idx,
                                             lit(baseOffset, idx.getType()))
                         .getResult();
      };
      SmallVector<Value, 2> planeOffs(spec->numBitPlanes);
      SmallVector<Value, 2> planeSh1D(spec->numBitPlanes);
      for (int pi = 0; pi < spec->numBitPlanes; ++pi) {
        const auto &pl = spec->planes[pi];
        if (spec->blockElements == 256 && pl.halfStride == 64 &&
            pl.quadByteStep == 32) {
          planeOffs[pi] = makeRangeOff(pl.baseByte, 128);
        } else if (spec->blockElements == 256 && pl.halfStride == 32 &&
                   pl.quadByteStep == 0) {
          planeOffs[pi] = makeRangeOff(pl.baseByte, 64);
        } else {
          planeOffs[pi] = b.create<arith::AddIOp>(
              loc,
              b.create<arith::AddIOp>(
                  loc, lit(pl.baseByte, ix.getType()),
                  b.create<arith::MulIOp>(loc, half,
                                          lit(pl.halfStride, ix.getType()))),
              b.create<arith::AddIOp>(
                  loc,
                  b.create<arith::MulIOp>(loc, quadOdd,
                                          lit(pl.quadByteStep, ix.getType())),
                  lane));
        }
        Value shFactor = pl.quadShiftMode == 0 ? quadHalf : quad;
        Value sh64 = b.create<arith::MulIOp>(loc, shFactor,
                                             lit(pl.shiftUnit, ix.getType()));
        planeSh1D[pi] = castTo("arith.trunci", sh64, b.getI32Type());
      }
      Value subOffPre;
      int64_t numSubPre = 0;
      bool fastSubScale = false;
      if (spec->subScaleOffset >= 0 && spec->subBlockElements > 0) {
        numSubPre = spec->blockElements / spec->subBlockElements;
        fastSubScale = spec->blockElements % spec->subBlockElements == 0 &&
                       numSubPre > 0 && llvm::isPowerOf2_64(numSubPre);
        subOffPre = fastSubScale
                        ? makeRangeOff(spec->subScaleOffset, numSubPre)
                        : b.create<arith::AddIOp>(
                              loc, lit(spec->subScaleOffset, ix.getType()),
                              b.create<arith::DivUIOp>(
                                  loc, ix,
                                  lit(spec->subBlockElements, ix.getType())))
                              .getResult();
      }
      return fold(
          p.shape[2] / spec->blockElements, zero({p.tile[0]}),
          [&](Value g, Value acc) {
            Value ds, qAct, vAct;
            if (spec->q8kActivation) {
              intentdsl::kir::Q8KRecord q8;
              auto v = t.read(x, t.address(x, {g, ix}, {-1, 0}));
              float maxQ = -float((1 << (q8.elementBits - 1)) - 1);
              float loQ = -float(1 << (q8.elementBits - 1));
              float hiQ = float((1 << (q8.elementBits - 1)) - 1);
              Value inv;
              if (spec->blockElements <= 256) {
                std::tie(inv, ds) = q8kScaleFromKey(
                    t.reduce(q8kArgmaxKey(v, ix, spec->blockElements), "max",
                             0),
                    maxQ);
              } else {
                auto abs = b.create<math::AbsFOp>(loc, v);
                auto peak = t.reduce(abs, "max", 0);
                auto eq = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ,
                                                  abs, t.like(peak, v.getType()));
                auto first = t.reduce(
                    b.create<arith::SelectOp>(
                        loc, eq, ix, lit(spec->blockElements, ix.getType())),
                    "min", 0);
                auto extreme = t.reduce(
                    b.create<arith::SelectOp>(
                        loc, cmp(arith::CmpIPredicate::eq, ix, first), v,
                        t.like(t.scalar(0), v.getType())),
                    "sum", 0);
                auto empty = b.create<arith::CmpFOp>(
                    loc, arith::CmpFPredicate::OEQ, peak, t.scalar(0));
                inv = strict(
                    "arith.divf", t.scalar(maxQ),
                    b.create<arith::SelectOp>(loc, empty, t.scalar(1), extreme));
                ds = b.create<arith::SelectOp>(
                    loc, empty, t.scalar(0),
                    strict("arith.divf", t.scalar(1), inv));
              }
              qAct = q8kQuantizeVec(v, inv, loQ, hiQ);
            } else {
              vAct = t.read(x, t.address(x, {g, ix}, {-1, 0}));
            }
            auto byte = [&](Value index, int dim) {
              auto ptr = t.address(w, {row, g, index}, {0, -1, dim});
              Value loaded;
              if (exactRows) {
                loaded = t.read(w, ptr);
              } else {
                auto mask = t.like(
                    validRow,
                    cast<RankedTensorType>(ptr.getType()).clone(b.getI1Type()),
                    0);
                loaded = t.read(w, ptr, mask);
              }
              auto u = castTo("arith.extui", loaded, b.getI32Type());
              return b.create<arith::AndIOp>(loc, u, lit(255, u.getType()))
                  .getResult();
            };
            auto halfAt = [&](int offset) {
              return t
                  .op("from_le_half",
                      {byte(t.integer(offset), -1),
                       byte(t.integer(offset + 1), -1)},
                      acc.getType())
                  ->getResult(0);
            };
            Value code;
            for (int pi = 0; pi < spec->numBitPlanes; ++pi) {
              const auto &pl = spec->planes[pi];
              Value raw;
              if (spec->blockElements == 256 && pl.halfStride == 64 &&
                  pl.quadByteStep == 32) {
                Value raw128 = byte(planeOffs[pi], 1);
                Value as3D = reshapeTile(raw128, {p.tile[0], 2, 64});
                raw = reshapeTile(
                    expandAxis(as3D, {p.tile[0], 2, 2, 64}, 2),
                    {p.tile[0], 256});
              } else if (spec->blockElements == 256 && pl.halfStride == 32 &&
                         pl.quadByteStep == 0) {
                Value raw64 = byte(planeOffs[pi], 1);
                Value as3D = reshapeTile(raw64, {p.tile[0], 2, 32});
                raw = reshapeTile(
                    expandAxis(as3D, {p.tile[0], 2, 4, 32}, 2),
                    {p.tile[0], 256});
              } else {
                raw = byte(planeOffs[pi], 1);
              }
              Type ty = raw.getType();
              Value sh32 = t.like(planeSh1D[pi], ty, 1);
              Value part = b.create<arith::AndIOp>(
                  loc, b.create<arith::ShRUIOp>(loc, raw, sh32),
                  lit((1 << pl.bitWidth) - 1, ty));
              if (pl.destBitOffset > 0)
                part = b.create<arith::ShLIOp>(
                    loc, part, lit(pl.destBitOffset, ty));
              code = pi == 0 ? part
                             : b.create<arith::OrIOp>(loc, code, part)
                                   .getResult();
            }
            Type ty = code.getType();
            if (spec->zeroBias != 0)
              code = b.create<arith::SubIOp>(loc, code, lit(spec->zeroBias, ty));
            if (spec->subScaleOffset >= 0 && spec->subBlockElements > 0) {
              Value scS8;
              if (fastSubScale) {
                Value scU8Sub = byte(subOffPre, 1);
                Type subTy = scU8Sub.getType();
                Value scS8Sub = b.create<arith::SubIOp>(
                    loc, b.create<arith::XOrIOp>(loc, scU8Sub, lit(128, subTy)),
                    lit(128, subTy));
                scS8 = reshapeTile(
                    expandAxis(scS8Sub,
                               {p.tile[0], numSubPre, spec->subBlockElements}, 2),
                    {p.tile[0], spec->blockElements});
              } else {
                Value scU8 = byte(subOffPre, 1);
                scS8 = b.create<arith::SubIOp>(
                    loc, b.create<arith::XOrIOp>(loc, scU8, lit(128, ty)),
                    lit(128, ty));
              }
              code = b.create<arith::MulIOp>(loc, code, scS8);
            }
            Value d = spec->scaleCode ==
                              intentdsl::kir::QuantScaleCode::LeHalfAtOffset
                          ? halfAt(spec->superScaleOffset)
                          : t.like(t.scalar(1.0f), acc.getType());
            if (spec->q8kActivation) {
              Value prod = b.create<arith::MulIOp>(
                  loc, code, t.like(qAct, ty, 1));
              Value sumI = t.reduce(prod, "sum", 1);
              Value scaledSum = strict(
                  "arith.mulf", d,
                  castTo("arith.sitofp", sumI, b.getF32Type()));
              return b
                  .create<arith::AddFOp>(
                      loc, acc,
                      strict("arith.mulf", t.like(ds, acc.getType()), scaledSum))
                  .getResult();
            }
            Value fWeight = castTo("arith.sitofp", code, b.getF32Type());
            Value fProd = b.create<arith::MulFOp>(
                loc, fWeight, t.like(vAct, fWeight.getType(), 1));
            Value sumF = t.reduce(fProd, "sum", 1);
            return b
                .create<arith::AddFOp>(loc, acc,
                                       strict("arith.mulf", d, sumF))
                .getResult();
          });
    }
    if (spec->roleLayout == intentdsl::kir::QuantRoleLayout::BlockScaledPair) {
      Value a = p.dot.getOperand(0), sa = p.dot.getOperand(1),
            rhs = p.dot.getOperand(2);
      intentdsl::kir::RightScale right(p.dot);
      Value init = zero({p.tile[0], p.tile[1]});
      int64_t groups = p.shape[2] / spec->blockElements;
      auto readScale = [&](Value base, Value ptr, Value valid, int axis) {
        auto ix = cast<RankedTensorType>(ptr.getType());
        auto mask = t.like(valid, ix.clone(b.getI1Type()), axis);
        auto u = castTo("arith.extui", t.read(base, ptr, mask), b.getI32Type());
        auto v = b.create<arith::AndIOp>(loc, u, lit(255, u.getType()));
        return b
            .create<arith::SelectOp>(loc, mask, v,
                                     lit(spec->scaleBias, v.getType()))
            .getResult();
      };
      auto inRange = [&](Value x, int lo, int hi) {
        auto min = t.reduce(x, "min", 0), max = t.reduce(x, "max", 0);
        return b.create<arith::AndIOp>(
            loc, cmp(arith::CmpIPredicate::sge, min, lit(lo, min.getType())),
            cmp(arith::CmpIPredicate::slt, max, lit(hi + 1, max.getType())));
      };
      auto k = t.lanes(spec->blockElements);
      Value bytePos =
          b.create<arith::DivUIOp>(loc, k, lit(spec->packFactor, k.getType()));
      Value subLane =
          b.create<arith::RemUIOp>(loc, k, lit(spec->packFactor, k.getType()));
      Value sh1D = laneShift(subLane, *spec);
      int codeMaskVal = (1 << spec->packBits) - 1;
      return fold(groups, init, [&](Value g, Value acc) {
        SmallVector<Value> rCoords(2);
        SmallVector<int64_t> rDims(2);
        rCoords[right.column] = col;
        rDims[right.column] = 0;
        rCoords[right.group] = g;
        rDims[right.group] = -1;
        Value ax =
            readScale(sa, t.address(sa, {row, g}, {0, -1}), validRow, 0);
        Value bx = readScale(right.view, t.address(right.view, rCoords, rDims),
                             validCol, 0);
        Value aMask = t.like(
            validRow,
            t.shaped({p.tile[0], spec->blockElements}, b.getI1Type()), 0);
        Value bMask = t.like(
            validCol,
            t.shaped({spec->blockElements, p.tile[1]}, b.getI1Type()), 1);
        Value aRaw = castTo(
            "arith.extui",
            t.read(a, t.address(a, {row, g, bytePos}, {0, -1, 1}), aMask),
            b.getI32Type());
        aRaw = b.create<arith::AndIOp>(loc, aRaw, lit(255, aRaw.getType()));
        Value bRaw = castTo(
            "arith.extui",
            t.read(rhs, t.address(rhs, {g, bytePos, col}, {-1, 0, 1}), bMask),
            b.getI32Type());
        bRaw = b.create<arith::AndIOp>(loc, bRaw, lit(255, bRaw.getType()));
        Value aCode = b.create<arith::AndIOp>(
            loc,
            b.create<arith::ShRUIOp>(loc, aRaw,
                                     t.like(sh1D, aRaw.getType(), 1)),
            lit(codeMaskVal, aRaw.getType()));
        Value bCode = b.create<arith::AndIOp>(
            loc,
            b.create<arith::ShRUIOp>(loc, bRaw,
                                     t.like(sh1D, bRaw.getType(), 0)),
            lit(codeMaskVal, bRaw.getType()));
        if (spec->zeroBias != 0) {
          aCode = b.create<arith::SubIOp>(loc, aCode,
                                          lit(spec->zeroBias, aCode.getType()));
          bCode = b.create<arith::SubIOp>(loc, bCode,
                                          lit(spec->zeroBias, bCode.getType()));
        }
        Value aVal = decodeQuantCode(aCode, *spec);
        Value bVal = decodeQuantCode(bCode, *spec);
        auto rescale = [&](Value val, Value scale, int axis) -> Value {
          auto factor = t.op("e8m0", scale,
                             cast<RankedTensorType>(scale.getType())
                                 .clone(b.getF32Type()))
                            ->getResult(0);
          return b.create<arith::MulFOp>(loc, val,
                                         t.like(factor, val.getType(), axis));
        };
        Value x = rescale(aVal, ax, 0), y = rescale(bVal, bx, 1);
        Value fast =
            b.create<arith::AndIOp>(loc, inRange(ax, 2, 252),
                                    inRange(bx, 2, 252));
        return choice(
            fast, acc,
            [&]() {
              return t.dot(castTo("arith.truncf", x, b.getBF16Type()),
                           castTo("arith.truncf", y, b.getBF16Type()), false,
                           acc);
            },
            [&]() { return t.dot(x, y, false, acc); });
      });
    }
    // GroupedWeightRhs: lhs [M, K], qweight [K, N / packFactor],
    // scales [K / blockElements, N], optional qzeros [G, N / packFactor].
    Value a = p.dot.getOperand(0), qw = p.dot.getOperand(1),
          sc = p.dot.getOperand(2);
    Type lhsElem = cast<MemRefType>(a.getType()).getElementType();
    Value init = zero({p.tile[0], p.tile[1]});
    Value subCol = b.create<arith::RemUIOp>(
        loc, col, lit(spec->packFactor, col.getType()));
    Value colShift1D = laneShift(subCol, *spec);
    Type rhsI32Ty = t.shaped({p.tile[2], p.tile[1]}, b.getI32Type());
    int codeMaskVal = (1 << spec->packBits) - 1;
    auto readPackedI32 = [&](Value view, Value ptr, Value mask) -> Value {
      Value raw = mask ? t.read(view, ptr, mask) : t.read(view, ptr);
      if (t.elem(raw.getType()).isInteger(32))
        return raw;
      Value u = castTo("arith.extui", raw, b.getI32Type());
      return b.create<arith::AndIOp>(loc, u, lit(255, u.getType()));
    };
    bool compactPackCols =
        spec->packFactor > 1 && p.tile[1] % spec->packFactor == 0 &&
        llvm::isPowerOf2_64(p.tile[1] / spec->packFactor);
    bool uniformGroup =
        compactPackCols && spec->blockElements >= p.tile[2] &&
        spec->blockElements % p.tile[2] == 0;
    int64_t packCols = compactPackCols ? (p.tile[1] / spec->packFactor) : 0;
    int64_t totalPackCols =
        llvm::divideCeil(p.shape[1], int64_t(std::max(1, spec->packFactor)));
    Value packCol1D, validPackCol, packColFull;
    if (compactPackCols) {
      packCol1D = t.stepped(
          b.create<arith::MulIOp>(loc, t.pid(1), t.integer(packCols)),
          packCols);
      validPackCol =
          cmp(arith::CmpIPredicate::slt, packCol1D, t.integer(totalPackCols));
    } else {
      packColFull = b.create<arith::DivUIOp>(
          loc, col, lit(spec->packFactor, col.getType()));
    }
    bool exactRhsBounds =
        (p.shape[2] > 0 && p.shape[2] % p.tile[2] == 0 && p.shape[1] > 0 &&
         p.shape[1] % p.tile[1] == 0);
    return fold(
        llvm::divideCeil(p.shape[2], p.tile[2]), init,
        [&](Value it, Value acc) {
          Value member = t.stepped(
              b.create<arith::MulIOp>(loc, it, t.integer(p.tile[2])),
              p.tile[2]);
          Value validK =
              cmp(arith::CmpIPredicate::slt, member, t.integer(p.shape[2]));
          Value lhs = readNative ? readNative(t, p, true, gp, it) : Value();
          if (!lhs)
            lhs = t.read(a, t.address(a, {row, member}, {0, 1}),
                         pairMask(validRow, validK));
          Value rhsMask =
              (!compactPackCols || !uniformGroup || !exactRhsBounds)
                  ? pairMask(validK, validCol)
                  : Value();
          Value qwRaw;
          if (compactPackCols) {
            Value qwCompact =
                readNative ? readNative(t, p, false, gp, it) : Value();
            if (!qwCompact)
              qwCompact = readPackedI32(
                  qw, t.address(qw, {member, packCol1D}, {0, 1}),
                  exactRhsBounds ? Value() : pairMask(validK, validPackCol));
            else if (!t.elem(qwCompact.getType()).isInteger(32)) {
              Value u = castTo("arith.extui", qwCompact, b.getI32Type());
              qwCompact =
                  b.create<arith::AndIOp>(loc, u, lit(255, u.getType()));
            }
            qwRaw = reshapeTile(
                expandAxis(qwCompact,
                           {p.tile[2], packCols, spec->packFactor}, 2),
                {p.tile[2], p.tile[1]});
          } else {
            qwRaw = readPackedI32(
                qw, t.address(qw, {member, packColFull}, {0, 1}), rhsMask);
          }
          Value colShift2D = t.like(colShift1D, rhsI32Ty, 1);
          Value wCode = b.create<arith::AndIOp>(
              loc, b.create<arith::ShRUIOp>(loc, qwRaw, colShift2D),
              lit(codeMaskVal, rhsI32Ty));
          Value rhsCast;
          auto rhsTy = RankedTensorType::get({p.tile[2], p.tile[1]}, lhsElem);
          if (uniformGroup) {
            int64_t panelsPerGroup = spec->blockElements / p.tile[2];
            Value grpScalar =
                panelsPerGroup == 1
                    ? it
                    : b.create<arith::DivUIOp>(loc, it,
                                               t.integer(panelsPerGroup))
                          .getResult();
            Value zCode1D;
            if (spec->hasZeroView) {
              Value qz = p.dot.getOperand(3);
              Value qzCompact1D = readPackedI32(
                  qz, t.address(qz, {grpScalar, packCol1D}, {-1, 0}),
                  exactRhsBounds ? Value() : validPackCol);
              Value qz1D = reshapeTile(
                  expandAxis(qzCompact1D, {packCols, spec->packFactor}, 1),
                  {p.tile[1]});
              Type colI32Ty = t.shaped({p.tile[1]}, b.getI32Type());
              zCode1D = b.create<arith::AndIOp>(
                  loc, b.create<arith::ShRUIOp>(loc, qz1D, colShift1D),
                  lit(codeMaskVal, colI32Ty));
            }
            Value scPtr = t.address(sc, {grpScalar, col}, {-1, 0});
            Value sc1D = exactRhsBounds ? t.read(sc, scPtr)
                                        : t.read(sc, scPtr, validCol);
            // For unsigned integer codes of <=10 bits with FP16 scales and LHS,
            // IEEE-754 half-precision bit pattern (0x6400 | code) equals exact
            // 1024.0 + code; subtracting (0x6400 | zero) is exact by Sterbenz's
            // lemma and avoids 32-bit int->f32->f16 conversion traffic.
            bool exactHalfBiasDecode =
                lhsElem.isF16() && t.elem(sc1D.getType()).isF16() &&
                spec->valueCode == kir::QuantValueCode::UnsignedAffine &&
                spec->packBits > 0 && spec->packBits <= 10 &&
                spec->zeroBias >= 0 && spec->zeroBias <= 1023 &&
                !(spec->hasZeroView && spec->zeroBias != 0);
            if (exactHalfBiasDecode) {
              constexpr int32_t kFp16Exp1024 = 0x6400;
              auto biasToF16 = [&](Value uCode) -> Value {
                Value biased = b.create<arith::OrIOp>(
                    loc, uCode, lit(kFp16Exp1024, uCode.getType()));
                Value i16 = castTo("arith.trunci", biased, b.getI16Type());
                auto f16Ty =
                    cast<RankedTensorType>(i16.getType()).clone(b.getF16Type());
                return b.create<arith::BitcastOp>(loc, f16Ty, i16);
              };
              Value wF16 = biasToF16(wCode);
              Value zF16;
              if (spec->hasZeroView) {
                zF16 = t.like(biasToF16(zCode1D), rhsTy, 1);
              } else {
                Value zConst = b.create<arith::ConstantOp>(
                    loc, b.getF16Type(),
                    b.getFloatAttr(b.getF16Type(),
                                   double(1024 + spec->zeroBias)));
                zF16 = t.like(zConst, rhsTy);
              }
              Value diffF16 = b.create<arith::SubFOp>(loc, wF16, zF16);
              rhsCast = b.create<arith::MulFOp>(
                  loc, diffF16, t.like(sc1D, rhsTy, 1));
            } else {
              if (spec->hasZeroView)
                wCode = b.create<arith::SubIOp>(
                    loc, wCode, t.like(zCode1D, rhsI32Ty, 1));
              if (spec->zeroBias != 0)
                wCode = b.create<arith::SubIOp>(loc, wCode,
                                                lit(spec->zeroBias, rhsI32Ty));
              Value wF32 = decodeQuantCode(wCode, *spec);
              if (!t.elem(sc1D.getType()).isF32())
                sc1D = castTo("arith.extf", sc1D, b.getF32Type());
              Value rhsF32 = b.create<arith::MulFOp>(
                  loc, wF32, t.like(sc1D, wF32.getType(), 1));
              rhsCast = lhsElem.isF32()
                            ? rhsF32
                            : castTo("arith.truncf", rhsF32, lhsElem);
            }
          } else {
            Value grp = b.create<arith::DivUIOp>(
                loc, member, lit(spec->blockElements, member.getType()));
            Value packColForZero = compactPackCols
                                       ? b.create<arith::DivUIOp>(
                                             loc, col,
                                             lit(spec->packFactor, col.getType()))
                                             .getResult()
                                       : packColFull;
            if (spec->hasZeroView) {
              Value qz = p.dot.getOperand(3);
              Value qzRaw = readPackedI32(
                  qz, t.address(qz, {grp, packColForZero}, {0, 1}), rhsMask);
              Value zCode = b.create<arith::AndIOp>(
                  loc, b.create<arith::ShRUIOp>(loc, qzRaw, colShift2D),
                  lit(codeMaskVal, rhsI32Ty));
              wCode = b.create<arith::SubIOp>(loc, wCode, zCode);
            }
            if (spec->zeroBias != 0)
              wCode = b.create<arith::SubIOp>(loc, wCode,
                                              lit(spec->zeroBias, rhsI32Ty));
            Value wF32 = decodeQuantCode(wCode, *spec);
            Value scVal =
                t.read(sc, t.address(sc, {grp, col}, {0, 1}), rhsMask);
            if (!t.elem(scVal.getType()).isF32())
              scVal = castTo("arith.extf", scVal, b.getF32Type());
            Value rhsF32 = b.create<arith::MulFOp>(loc, wF32, scVal);
            rhsCast = lhsElem.isF32()
                          ? rhsF32
                          : castTo("arith.truncf", rhsF32, lhsElem);
          }
          if (exactRhsBounds)
            return t.dot(lhs, rhsCast, false, acc);
          auto zeroRhs = b.create<arith::ConstantOp>(
              loc, rhsTy, DenseElementsAttr::get(rhsTy, b.getZeroAttr(lhsElem)));
          Value rhsMasked =
              b.create<arith::SelectOp>(loc, rhsMask, rhsCast, zeroRhs);
          return t.dot(lhs, rhsMasked, false, acc);
        });
  }
  void run(ScaledPanelLowering native, MatrixPanelRead readNative,
           WindowPanelLowering windowNative, MatrixPanelWrite writeNative,
           bool flatRecordBitPlanes) {
    bool records = p.dot.getName().stripDialect() == "q4k_dot";
    bool isQuant = p.dot.getName().stripDialect() == "quant_contract";
    bool hasWindow = p.dot.getName().stripDialect() == "window_contract";
    bool is1D = records || (isQuant && p.rank(p.domain.getOperand(0)) == 1);
    Value batch, chunk, head;
    Value acc;
    if (records)
      acc = record();
    else if (isQuant) {
      const auto *spec = intentdsl::kir::quantFormatSpec(p.dot);
      if (spec &&
          spec->roleLayout == intentdsl::kir::QuantRoleLayout::BlockScaledPair &&
          native)
        acc = native(t, p);
      if (!acc)
        acc = quantized(readNative, flatRecordBitPlanes);
    } else if (hasWindow) {
      if (windowNative)
        acc = windowNative(t, p, batch, chunk, head);
      if (!acc)
        acc = window(batch, chunk, head);
    } else
      acc = native ? native(t, p) : scaled();
    IRMapping map;
    map.map(p.fold.getResult(0), acc);
    for (auto &op : p.domain.getRegion(0).front()) {
      auto code = op.getName().stripDialect();
      if (code == "add_columns") {
        auto view = op.getOperand(1), old = map.lookup(op.getOperand(0));
        auto bias = t.read(view, t.address(view, is1D ? row : col, {0}),
                           is1D ? validRow : validCol);
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
        SmallVector<Value, 3> batches;
        if (hasWindow)
          batches.assign({batch, chunk, head});
        if (!is1D && writeNative && writeNative(t, p, y, value, batches))
          continue;
        auto ptr = is1D      ? t.address(y, row, {0})
                   : hasWindow ? t.address(y, {batch, chunk, head, row, col},
                                           {-1, -1, -1, 0, 1})
                               : t.address(y, {row, col}, {0, 1});
        t.write(y, ptr, value,
                is1D ? validRow : pairMask(validRow, validCol));
      }
    }
  }
};
} // namespace
void encodedTiles(Tiles &t, intentdsl::block::MatrixAccess &access,
                  ScaledPanelLowering native, MatrixPanelRead readNative,
                  WindowPanelLowering windowNative,
                  MatrixPanelWrite writeNative, bool flatRecordBitPlanes) {
  Panel(t, access).run(native, readNative, windowNative, writeNative,
                       flatRecordBitPlanes);
}
} // namespace intentdsl::block
