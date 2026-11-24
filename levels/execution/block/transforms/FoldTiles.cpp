#include "OwnerTiles.h"
using namespace mlir;
namespace intentdsl::block {
namespace {
enum class Axis { Scalar, Owner, Token, Both };
struct Fragment {
  Value value;
  Axis axis;
};
} // namespace
SmallVector<Value> OwnerTiles::panelLoop(scf::ForOp loop, const Values &parent,
                                         Value mask,
                                         const OnlinePanelFusion *online) {
  int64_t span =
      loop->getAttrOfType<IntegerAttr>("block.contribution_width").getInt();
  bool paired = space.hasAttr("block.pair_axis");
  auto asScalar = [&](Value v, StringRef algebra) -> Value {
    auto fragment = dyn_cast<RankedTensorType>(v.getType());
    if (!fragment)
      return v;
    if (rows != logical && fragment.getRank() == 1)
      v = op("tile_prefix", v, shaped({logical}, elem(v.getType())))
              ->getResult(0);
    return reduce(v, algebra, 0);
  };
  Value lower = at(loop.getLowerBound(), parent),
        upper = at(loop.getUpperBound(), parent);
  auto tileType = [&](Type e) { return shaped({rows, span}, e); };
  auto plane = [&](Fragment x) -> Value {
    return broadcast(x.value, tileType(elem(x.value.getType())),
                     x.axis == Axis::Token ? 1 : 0);
  };
  auto uniform = [&](Value v) { return asScalar(v, "min"); };
  bool full = loop->hasAttr("block.full_panels");
  Value width = integer(span), lo = uniform(lower), hi = asScalar(upper, "max");
  Value start =
      full ? lo
           : Value(b.create<arith::MulIOp>(
                 loc, b.create<arith::FloorDivSIOp>(loc, lo, width), width));
  SmallVector<Value> init;
  SmallVector<Type> types;
  if (online) {
    auto mTy = shaped({rows}, b.getF32Type());
    auto seenTy = shaped({rows}, b.getI32Type());
    init.push_back(broadcast(scalar(-INFINITY), mTy));
    types.push_back(mTy);
    init.push_back(
        broadcast(b.create<arith::ConstantIntOp>(loc, 0, 32), seenTy));
    types.push_back(seenTy);
  }
  for (auto v : loop.getInitArgs()) {
    auto t = lifted(v);
    init.push_back(broadcast(at(v, parent), t));
    types.push_back(t);
  }
  Value begin = start, end = hi, stride = width;
  bool counted = loop->hasAttr("block.counted_panels");
  if (counted) {
    auto constI64 = [](Value v) -> std::optional<int64_t> {
      if (auto c = v.getDefiningOp<arith::ConstantOp>())
        if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
          return ia.getInt();
      return std::nullopt;
    };
    if (auto cs = constI64(start))
      begin = b.create<arith::ConstantIntOp>(loc, *cs / span, 32);
    else
      begin = b.create<arith::TruncIOp>(
          loc, b.getI32Type(), b.create<arith::DivUIOp>(loc, start, width));
    if (auto ch = constI64(hi))
      end = b.create<arith::ConstantIntOp>(loc, (*ch + span - 1) / span, 32);
    else {
      auto last = b.create<arith::AddIOp>(loc, hi, integer(span - 1));
      end = b.create<arith::TruncIOp>(
          loc, b.getI32Type(), b.create<arith::DivUIOp>(loc, last, width));
    }
    stride = b.create<arith::ConstantIntOp>(loc, 1, 32);
  }
  auto l = b.create<scf::ForOp>(loc, begin, end, stride, init);
  l->setAttr(symbol("runtime_range"), b.getUnitAttr());
  b.setInsertionPointToStart(l.getBody());
  Value base = l.getInductionVar();
  if (counted)
    base = b.create<arith::MulIOp>(
        loc, b.create<arith::ExtSIOp>(loc, b.getI64Type(), base), width);
  Value token = stepped(base, span);
  auto tokens = Fragment{token, Axis::Token};
  auto tokenType = token.getType();
  Value live, any;
  if (full) {
    auto one = b.create<arith::ConstantOp>(loc, b.getBoolAttr(true));
    any = like(one, shaped({span}, b.getI1Type()));
    live = like(one, tileType(b.getI1Type()));
  } else if (loop->hasAttr("block.uniform_tokens") || !paired) {
    any = b.create<arith::AndIOp>(
        loc,
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, token,
                                like(lo, tokenType)),
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, token,
                                like(hi, tokenType)));
    live = plane({any, Axis::Token});
  } else {
    auto t = plane(tokens);
    live = b.create<arith::AndIOp>(
        loc,
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, t,
                                plane({lower, Axis::Owner})),
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, t,
                                plane({upper, Axis::Owner})));
    any = reduce(live, "max", 0);
  }
  llvm::DenseMap<Value, Fragment> local;
  local[loop.getInductionVar()] = tokens;
  auto value = [&](Value v) {
    if (auto it = local.find(v); it != local.end())
      return it->second;
    auto x = at(v, parent);
    return Fragment{x, isa<RankedTensorType>(x.getType()) ? Axis::Owner
                                                          : Axis::Scalar};
  };
  auto coordinate = [&](Value v) {
    auto x = value(v);
    return x.axis == Axis::Owner ? uniform(x.value) : x.value;
  };
  llvm::SmallPtrSet<Operation *, 16> folded;
  Operation *onlineAnchor = nullptr;
  Value onlineMNew, onlineSeenNew, onlineAlpha;
  if (online) {
    onlineAnchor = online->guardOp ? online->guardOp : online->expOp;
    folded.insert(online->subOp);
    if (onlineAnchor != online->expOp)
      folded.insert(online->expOp);
  }
  auto yielded = loop.getBody()->getTerminator()->getOperands();
  for (Value r : yielded) {
    auto *comb = r.getDefiningOp();
    folded.insert(comb);
    if (comb->hasAttr("block.masked_dot_update")) {
      auto *product = comb->getOperand(1).getDefiningOp();
      folded.insert(product);
      for (Value input : product->getOperands())
        if (auto *op = input.getDefiningOp();
            op && op->getName().stripDialect() == "lane_splat" &&
            input.hasOneUse()) {
          folded.insert(op);
          if (auto *ext = op->getOperand(0).getDefiningOp();
              ext && isa<arith::ExtFOp>(ext) && op->getOperand(0).hasOneUse())
            folded.insert(ext);
        }
    }
  }
  for (auto &o : *loop.getBody())
    if (o.hasAttr("block.pair_dot"))
      folded.insert(o.getOperand(0).getDefiningOp());
  for (auto &o : loop.getBody()->without_terminator())
    if (isa<arith::ExtFOp>(o) && o.getOperand(0).getDefiningOp() &&
        o.getOperand(0).getDefiningOp()->getName().stripDialect() ==
            "slice_read" &&
        !o.use_empty() &&
        llvm::all_of(o.getUsers(),
                     [&](Operation *u) { return folded.contains(u); }))
      folded.insert(&o);
  auto singleMember = [&](Fragment x) -> Value {
    if (x.axis != Axis::Owner)
      return x.value;
    if (auto s = dyn_cast<RankedTensorType>(x.value.getType());
        s && s.getRank() == 2)
      return op("drop_unit_owner", x.value,
                shaped({s.getDimSize(1)}, s.getElementType()))
          ->getResult(0);
    return uniform(x.value);
  };
  auto combineRole = [](ArrayRef<Fragment> inputs) {
    Axis role = Axis::Scalar;
    for (const auto &in : inputs)
      if (in.axis != Axis::Scalar && in.axis != role)
        role = role == Axis::Scalar ? in.axis : Axis::Both;
    return role;
  };
  BlockArgument pairOwner;
  if (auto pairAttr = space.getAttrOfType<IntegerAttr>("block.pair_axis"))
    pairOwner = space.getRegion(0).front().getArgument(pairAttr.getInt());
  auto promoteInputs = [&](ArrayRef<Fragment> inputs, Value out, Axis &role,
                           Type &dst) {
    bool sharedIdx =
        paired && out.getType().isIndex() &&
        intentdsl::block::sharedCoordinate(out, pairOwner, loop, logical);
    SmallVector<Value> args;
    for (auto x : inputs) {
      auto arg = x.value;
      if (!paired) {
        arg = singleMember(x);
        if (isa<RankedTensorType>(out.getType())) {
          auto n = std::max<int64_t>(
              1, llvm::PowerOf2Ceil(
                     intentdsl::kir::memberExtent(fn, out, "block.shape")));
          arg = broadcast(arg, shaped({span, n}, elem(arg.getType())),
                          x.axis == Axis::Token ? 0 : 1);
        }
      } else if (sharedIdx) {
        if (x.axis == Axis::Owner)
          arg = uniform(arg);
        role = llvm::any_of(inputs,
                            [](Fragment f) { return f.axis == Axis::Token; })
                   ? Axis::Token
                   : Axis::Scalar;
      } else if (role == Axis::Both)
        arg = plane(x);
      args.push_back(arg);
      if (auto rt = dyn_cast<RankedTensorType>(arg.getType());
          rt && (!dst || !isa<RankedTensorType>(dst) ||
                 rt.getRank() > cast<RankedTensorType>(dst).getRank()))
        dst = rt;
      else if (!dst)
        dst = arg.getType();
    }
    if (!paired)
      role = isa<RankedTensorType>(dst) ? Axis::Token : Axis::Scalar;
    return args;
  };
  std::function<SmallVector<Fragment>(Block &)> lowerFoldBlock =
      [&](Block &blk) -> SmallVector<Fragment> {
    for (auto &old : blk.without_terminator()) {
      if (folded.contains(&old))
        continue;
      if (&old == onlineAnchor) {
        Value sTile = paired ? plane(value(online->scoreVal))
                             : singleMember(value(online->scoreVal));
        if (!isa<RankedTensorType>(sTile.getType()))
          sTile = broadcast(sTile, shaped({span}, b.getF32Type()));
        Value validMask;
        if (online->condVal) {
          Value cTile = paired ? plane(value(online->condVal))
                               : singleMember(value(online->condVal));
          validMask = broadcast(
              cTile, cast<RankedTensorType>(sTile.getType()).clone(b.getI1Type()));
        }
        if (!full) {
          Value domMask = paired ? live : any;
          validMask = validMask
                          ? Value(b.create<arith::AndIOp>(loc, validMask, domMask))
                          : domMask;
        }
        Value negInfTile = broadcast(scalar(-INFINITY), sTile.getType());
        Value sMasked = validMask ? select(validMask, sTile, negInfTile) : sTile;
        Value mPrev = l.getRegionIterArgs()[0];
        Value seenPrev = l.getRegionIterArgs()[1];
        Value rowValid =
            validMask
                ? broadcast(reduce(castElement(validMask, b.getI32Type()), "max",
                                   paired ? 1 : 0),
                            seenPrev.getType())
                : broadcast(b.create<arith::ConstantIntOp>(loc, 1, 32),
                            seenPrev.getType());
        onlineSeenNew = b.create<arith::OrIOp>(loc, seenPrev, rowValid);
        Value mPanel =
            broadcast(reduce(sMasked, "max", paired ? 1 : 0), mPrev.getType());
        onlineMNew = b.create<arith::MaxNumFOp>(loc, mPrev, mPanel);
        Value negInfRow = broadcast(scalar(-INFINITY), onlineMNew.getType());
        Value zeroRow = broadcast(scalar(0.f), onlineMNew.getType());
        Value allMasked = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ,
                                                  onlineMNew, negInfRow);
        Value mSafe = select(allMasked, zeroRow, onlineMNew);
        Value mDiff = b.create<arith::SubFOp>(loc, mPrev, mSafe);
        onlineAlpha = online->isExp2
                          ? Value(b.create<math::Exp2Op>(loc, mDiff))
                          : Value(b.create<math::ExpOp>(loc, mDiff));
        Value mSafeTile =
            paired ? plane({mSafe, Axis::Owner})
                   : broadcast(singleMember({mSafe, Axis::Owner}), sTile.getType());
        Value sShift = b.create<arith::SubFOp>(loc, sTile, mSafeTile);
        Value wRaw = online->isExp2
                         ? Value(b.create<math::Exp2Op>(loc, sShift))
                         : Value(b.create<math::ExpOp>(loc, sShift));
        Value zeroTile = broadcast(scalar(0.f), sTile.getType());
        Value wTile = validMask ? select(validMask, wRaw, zeroTile) : wRaw;
        local[online->weightVal] = {wTile, paired ? Axis::Both : Axis::Token};
        continue;
      }
      if (auto br = dyn_cast<scf::IfOp>(old)) {
        auto cFrag = value(br.getCondition());
        auto yes = lowerFoldBlock(*br.thenBlock());
        auto no = lowerFoldBlock(*br.elseBlock());
        for (auto [res, tFrag, fFrag] :
             llvm::zip(br.getResults(), yes, no)) {
          SmallVector<Fragment, 3> tri{cFrag, tFrag, fFrag};
          Axis role = combineRole(tri);
          Type dst;
          auto args = promoteInputs(tri, res, role, dst);
          auto ty = elem(res.getType());
          if (ty.isIndex())
            ty = b.getI64Type();
          if (auto s = dyn_cast<RankedTensorType>(dst))
            ty = s.clone(ty);
          local[res] = {
              select(args[0], broadcast(args[1], ty), broadcast(args[2], ty)),
              role};
        }
        continue;
      }
      auto code = old.getName().stripDialect();
      auto out = old.getResult(0);
      auto operand = [&](int i) { return value(old.getOperand(i)); };
      Fragment result;
      if (auto c = dyn_cast<arith::ConstantOp>(old))
        result = {constant(c), Axis::Scalar};
      else if (code == "slice_read" || isa<memref::LoadOp>(old)) {
        auto view = old.getOperand(0);
        SmallVector<Value> coords;
        SmallVector<int64_t> axes;
        for (auto v : old.getOperands().drop_front()) {
          auto ix = widen(coordinate(v));
          coords.push_back(broadcast(ix, tokenType));
          axes.push_back(0);
        }
        if (code == "slice_read" && access.panel) {
          SmallVector<Value> outer;
          for (auto v : old.getOperands().drop_front())
            outer.push_back(widen(coordinate(v)));
          auto n = shape(view)[old.getAttrOfType<IntegerAttr>("axis").getInt()];
          if (Value tile = access.panel(*this, old, outer, base, span, n)) {
            auto zero = castElement(scalar(0), elem(tile.getType()));
            auto maskType =
                cast<RankedTensorType>(tile.getType()).clone(b.getI1Type());
            local[out] = {full ? tile
                               : select(broadcast(any, maskType, 0), tile,
                                        broadcast(zero, tile.getType())),
                          Axis::Token};
            continue;
          }
        }
        Value accessMask = any;
        if (code == "slice_read") {
          auto ax = old.getAttrOfType<IntegerAttr>("axis").getInt(),
               n = shape(view)[ax];
          int64_t padded = std::max<int64_t>(1, llvm::PowerOf2Ceil(n));
          Value k = lanes(padded);
          coords.insert(coords.begin() + ax, k);
          axes.insert(axes.begin() + ax, 1);
          auto m = shaped({span, padded}, b.getI1Type());
          accessMask = broadcast(any, m, 0);
          if (padded != n) {
            auto within = b.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::slt, k,
                like(integer(n), k.getType()));
            accessMask = b.create<arith::AndIOp>(loc, accessMask,
                                                 broadcast(within, m, 1));
          }
        }
        bool whole = full && (code != "slice_read" ||
                              llvm::isPowerOf2_64(shape(view)[
                                  old.getAttrOfType<IntegerAttr>("axis")
                                      .getInt()]));
        result = {read(view, address(view, coords, axes),
                       whole ? Value() : accessMask),
                  Axis::Token};
      } else if (old.hasAttr("block.pair_dot")) {
        auto operands = *intentdsl::block::tileDot(&old, loop);
        result = {
            dot(value(operands.rows).value, value(operands.tokens).value, true),
            Axis::Both};
      } else if (code == "lane_splat") {
        auto x = operand(0);
        int64_t n = std::max<int64_t>(
            1, llvm::PowerOf2Ceil(
                   intentdsl::kir::memberExtent(fn, out, "block.shape")));
        result = {
            broadcast(singleMember(x), shaped({span, n}, elem(out.getType())),
                      0),
            Axis::Token};
      } else if (code == "lane_sum")
        result = {reduce(operand(0).value, "sum", 1), Axis::Token};
      else {
        SmallVector<Fragment> inputs;
        for (Value v : old.getOperands())
          inputs.push_back(value(v));
        Axis role = combineRole(inputs);
        Type dst;
        auto args = promoteInputs(inputs, out, role, dst);
        auto ty = elem(out.getType());
        if (ty.isIndex())
          ty = b.getI64Type();
        if (auto s = dyn_cast<RankedTensorType>(dst))
          ty = s.clone(ty);
        result = {apply(old, args, ty), role};
      }
      local[out] = result;
    }
    SmallVector<Fragment> outs;
    if (&blk != loop.getBody())
      for (Value yv : blk.getTerminator()->getOperands())
        outs.push_back(value(yv));
    return outs;
  };
  (void)lowerFoldBlock(*loop.getBody());
  SmallVector<Value> next;
  if (online) {
    next.push_back(onlineMNew);
    next.push_back(onlineSeenNew);
  }
  auto accIterArgs =
      l.getRegionIterArgs().drop_front(online ? 2 : 0);
  for (auto [ret, rawAcc] : llvm::zip(yielded, accIterArgs)) {
    Value acc = online
                    ? Value(b.create<arith::MulFOp>(
                          loc, rawAcc,
                          broadcast(onlineAlpha, rawAcc.getType(), 0)))
                    : rawAcc;
    auto *comb = ret.getDefiningOp();
    auto rhs = comb->getOperand(1);
    if (paired && comb->hasAttr("block.masked_dot_update")) {
      auto operands = *intentdsl::block::tileAccumulate(rhs, loop);
      auto w = plane(value(operands.weights));
      auto v = value(operands.values).value;
      auto zero = castElement(scalar(0), elem(w.getType()));
      auto guarded = full ? w : select(live, w, broadcast(zero, w.getType()));
      auto w16 = castElement(guarded, elem(v.getType()));
      if (!loop->hasAttr("block.repair_masked_nan")) {
        next.push_back(dot(w16, v, false, acc));
      } else {
        auto eTy = elem(v.getType());
        auto zElem = castElement(scalar(0), eTy);
        auto zW = broadcast(zElem, w16.getType());
        auto zV = broadcast(zElem, v.getType());
        auto diffW = b.create<arith::SubFOp>(loc, w16, w16);
        auto diffV = b.create<arith::SubFOp>(loc, v, v);
        auto finW =
            b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, diffW, zW);
        auto finV =
            b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, diffV, zV);
        auto nfW =
            b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNE, diffW, zW);
        auto nfV =
            b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNE, diffV, zV);
        auto hasNf = b.create<arith::OrIOp>(
            loc, reduce(reduce(nfW, "max", 1), "max", 0),
            reduce(reduce(nfV, "max", 1), "max", 0));
        Value fast =
            dot(select(finW, w16, zW), select(finV, v, zV), false, acc);
        auto fix = b.create<scf::IfOp>(loc, fast.getType(), hasNf, true);
        b.setInsertionPointToStart(fix.thenBlock());
        auto c0 = b.create<arith::ConstantIntOp>(loc, 0, 32),
             cSpan = b.create<arith::ConstantIntOp>(loc, span, 32),
             c1 = b.create<arith::ConstantIntOp>(loc, 1, 32);
        auto fixLoop = b.create<scf::ForOp>(loc, c0, cSpan, c1, ValueRange{fast});
        fixLoop->setAttr(symbol("runtime_range"), b.getUnitAttr());
        b.setInsertionPointToStart(fixLoop.getBody());
        Value cur = fixLoop.getRegionIterArgs()[0];
        Value j64 = b.create<arith::ExtSIOp>(loc, b.getI64Type(),
                                             fixLoop.getInductionVar());
        auto tokAxis = lanes(span);
        auto isJ = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                           tokAxis, like(j64, tokAxis.getType()));
        auto isJW = broadcast(isJ, live.getType(), 1);
        auto liveJ = reduce(b.create<arith::AndIOp>(loc, live, isJW), "max", 1);
        auto wF32 = castElement(w16, b.getF32Type());
        auto zWF32 = broadcast(scalar(0.f), wF32.getType());
        auto wJ = reduce(select(isJW, wF32, zWF32), "sum", 1);
        auto isJV = broadcast(
            isJ, cast<RankedTensorType>(v.getType()).clone(b.getI1Type()), 0);
        auto vF32 = castElement(v, b.getF32Type());
        auto zVF32 = broadcast(scalar(0.f), vF32.getType());
        auto vJ = reduce(select(isJV, vF32, zVF32), "sum", 0);
        auto accMaskTy =
            cast<RankedTensorType>(cur.getType()).clone(b.getI1Type());
        auto classify = [&](Value vec, int axis) {
          auto z1 = broadcast(scalar(0.f), vec.getType());
          auto p1 = broadcast(scalar(INFINITY), vec.getType());
          auto n1 = broadcast(scalar(-INFINITY), vec.getType());
          auto lift = [&](Value p) { return broadcast(p, accMaskTy, axis); };
          Value nan = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNE,
                                              vec, vec);
          Value notNan = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ,
                                                 vec, vec);
          Value pinf = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ,
                                               vec, p1);
          Value ninf = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ,
                                               vec, n1);
          Value inf = b.create<arith::OrIOp>(loc, pinf, ninf);
          Value zero = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ,
                                               vec, z1);
          Value nz = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNE,
                                             vec, z1);
          Value mx = b.create<arith::MaxNumFOp>(loc, vec, z1);
          Value pos = b.create<arith::AndIOp>(
              loc, nz,
              b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, mx, vec));
          Value neg = b.create<arith::AndIOp>(
              loc, b.create<arith::AndIOp>(loc, notNan, nz),
              b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, mx, z1));
          return std::tuple{lift(nan),  lift(pinf), lift(ninf), lift(inf),
                            lift(zero), lift(pos),  lift(neg)};
        };
        auto [wNan, wPInf, wNInf, wInf, wZero, wPos, wNeg] = classify(wJ, 0);
        auto [vNan, vPInf, vNInf, vInf, vZero, vPos, vNeg] = classify(vJ, 1);
        auto and2 = [&](Value a, Value c) -> Value {
          return b.create<arith::AndIOp>(loc, a, c);
        };
        auto or2 = [&](Value a, Value c) -> Value {
          return b.create<arith::OrIOp>(loc, a, c);
        };
        auto live2D = broadcast(liveJ, accMaskTy, 0);
        auto termPInf = and2(
            live2D,
            or2(or2(and2(wPInf, vPos), and2(wPos, vPInf)),
                or2(and2(wNInf, vNeg), and2(wNeg, vNInf))));
        auto termNInf = and2(
            live2D,
            or2(or2(and2(wPInf, vNeg), and2(wPos, vNInf)),
                or2(and2(wNInf, vPos), and2(wNeg, vPInf))));
        auto termNan = and2(
            live2D,
            or2(or2(wNan, vNan), or2(and2(wInf, vZero), and2(wZero, vInf))));
        auto pInfAcc = broadcast(scalar(INFINITY), cur.getType());
        auto nInfAcc = broadcast(scalar(-INFINITY), cur.getType());
        auto nanAcc = broadcast(scalar(NAN), cur.getType());
        auto badP = and2(
            termPInf,
            b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, cur,
                                    nInfAcc));
        auto badN = and2(
            termNInf,
            b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, cur,
                                    pInfAcc));
        auto stayNan =
            b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNE, cur, cur);
        cur = select(termPInf, pInfAcc, cur);
        cur = select(termNInf, nInfAcc, cur);
        cur = select(or2(or2(termNan, stayNan), or2(badP, badN)), nanAcc, cur);
        b.create<scf::YieldOp>(loc, ValueRange{cur});
        b.setInsertionPointAfter(fixLoop);
        b.create<scf::YieldOp>(loc, fixLoop.getResults());
        b.setInsertionPointToStart(fix.elseBlock());
        b.create<scf::YieldOp>(loc, ValueRange{fast});
        b.setInsertionPointAfter(fix);
        next.push_back(fix.getResult(0));
      }
    } else {
      bool max = isa<arith::MaxNumFOp>(comb);
      auto x = paired ? plane(value(rhs)) : singleMember(value(rhs));
      Value keep = paired ? live : any;
      if (!isa<RankedTensorType>(x.getType()))
        x = broadcast(x, shaped({span}, elem(x.getType())));
      Value term =
          full ? x
               : select(keep, x,
                        broadcast(scalar(max ? -INFINITY : 0.f), x.getType()));
      auto partial = reduce(term, max ? "max" : "sum", paired ? 1 : 0);
      if (!paired && isa<ShapedType>(rhs.getType()))
        partial = broadcast(partial, acc.getType(), 1);
      else
        partial = broadcast(partial, acc.getType());
      next.push_back(max ? Value(b.create<arith::MaxNumFOp>(loc, acc, partial))
                         : Value(b.create<arith::AddFOp>(loc, acc, partial)));
    }
  }
  b.create<scf::YieldOp>(loc, next);
  b.setInsertionPointAfter(l);
  if (!online)
    return SmallVector<Value>(l.getResults());
  Value finalM = l.getResult(0);
  Value finalSeen = l.getResult(1);
  Value zeroSeen =
      broadcast(b.create<arith::ConstantIntOp>(loc, 0, 32), finalSeen.getType());
  Value negInfM = broadcast(scalar(-INFINITY), finalM.getType());
  Value hasValid =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, finalSeen, zeroSeen);
  Value mIsNegInf =
      b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, finalM, negInfM);
  Value allValidNegInf = b.create<arith::AndIOp>(loc, hasValid, mIsNegInf);
  SmallVector<Value> outs{finalM};
  for (Value res : l.getResults().drop_front(2)) {
    auto resTy = cast<RankedTensorType>(res.getType());
    Value nanMask = resTy.getRank() == 2
                        ? broadcast(allValidNegInf, resTy.clone(b.getI1Type()), 0)
                        : allValidNegInf;
    outs.push_back(b.create<arith::SelectOp>(
        loc, nanMask, broadcast(scalar(NAN), resTy), res));
  }
  return outs;
}
} // namespace intentdsl::block
