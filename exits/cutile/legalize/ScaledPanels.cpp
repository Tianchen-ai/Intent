#include "intentdsl/levels/algorithm/kir/ir/PanelMap.h"
#include "intentdsl/levels/algorithm/kir/ir/Storage.h"
#include "intentdsl/levels/execution/block/ir/MatrixProgram.h"

namespace CI_UNIT_NAMESPACE {
using namespace mlir;
using intentdsl::block::MatrixAccess;
using intentdsl::block::Tiles;

// The scale interval and packed array views are bound at @legal. At
// @callable the native and general paths are separate, editable SSA regions.
Value scaledPanels(Tiles &t, MatrixAccess &p) {
  auto &b = t.b;
  auto loc = t.loc;
  if (p.dot.getName().stripDialect() == "quant_contract") {
    const auto *spec = intentdsl::kir::quantFormatSpec(p.dot);
    auto flat = p.dot.getAttrOfType<DenseI64ArrayAttr>("ct_src.flat_groups");
    int64_t m = p.tile[0], n = p.tile[1], k = p.tile[2];
    int64_t totalM = p.shape[0], totalN = p.shape[1], totalK = p.shape[2];
    int64_t totalGroups = totalK / 32;
    if (!spec ||
        spec->roleLayout != intentdsl::kir::QuantRoleLayout::BlockScaledPair ||
        spec->valueCode != intentdsl::kir::QuantValueCode::Fp4E2M1 ||
        spec->scaleCode != intentdsl::kir::QuantScaleCode::E8M0 ||
        spec->packFactor != 2 || spec->blockElements != 32 || !flat ||
        flat.size() != 2 || !t.fn->hasAttr("ct_src.native_views") || k < 32 ||
        k % 32 != 0 || totalM <= 0 || totalM % m != 0 || totalN <= 0 ||
        totalN % n != 0 || totalK <= 0 || totalK % k != 0 || totalGroups <= 0 ||
        totalGroups > 256 || !llvm::isPowerOf2_64(totalGroups))
      return {};
    bool native = p.dot.hasAttr("ct_src.scaled_mma");
    int64_t g = k / 32, kBytes = k / 2;
    Value a = p.dot.getOperand(0), sa = p.dot.getOperand(1),
          rhs = p.dot.getOperand(2);
    intentdsl::kir::RightScale scale(p.dot);
    auto reshape = [&](Value v, ArrayRef<int64_t> shape) {
      return t.op("reshape", v, t.shaped(shape, t.elem(v.getType())))
          ->getResult(0);
    };
    auto castTo = [&](StringRef op, Value v, Type e) {
      OperationState st(loc, op);
      st.addOperands(v);
      st.addTypes(cast<RankedTensorType>(v.getType()).clone(e));
      return b.create(st)->getResult(0);
    };
    auto lit = [&](int64_t val, Type ty) {
      return t.like(
          b.create<arith::ConstantOp>(loc, b.getIntegerAttr(t.elem(ty), val)),
          ty);
    };
    auto read = [&](Value base, ArrayRef<int64_t> shape, ValueRange indices,
                    bool allowTma) {
      SmallVector<Value> args{base};
      llvm::append_range(args, indices);
      return t
          .op("rect_read", args,
              t.shaped(shape, cast<MemRefType>(base.getType()).getElementType()),
              {b.getNamedAttr(
                  "allow_tma",
                  b.getBoolAttr(allowTma && t.fn->hasAttr("ct_src.tma")))})
          ->getResult(0);
    };
    auto xblock = t.pid(0), yblock = t.pid(1), zero = t.integer(0);
    auto allLS = read(sa, {m, totalGroups}, {xblock, zero}, false);
    auto allRS = scale.column == 0
                     ? read(scale.view, {n, totalGroups}, {yblock, zero}, false)
                     : read(scale.view, {totalGroups, n}, {zero, yblock}, false);
    auto inside = [&](Value v) {
      auto count = cast<RankedTensorType>(v.getType()).getNumElements();
      Value codes = castTo("arith.extui", reshape(v, {count}), b.getI32Type());
      codes = b.create<arith::AndIOp>(loc, codes, lit(255, codes.getType()));
      return b.create<arith::AndIOp>(
          loc,
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                  t.reduce(codes, "min", 0),
                                  b.create<arith::ConstantIntOp>(loc, 2, 32)),
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                  t.reduce(codes, "max", 0),
                                  b.create<arith::ConstantIntOp>(loc, 253, 32)));
    };
    auto valid = b.create<arith::AndIOp>(loc, inside(allLS), inside(allRS));
    auto matrix = t.shaped({m, n}, b.getF32Type());
    auto branch = b.create<scf::IfOp>(loc, TypeRange{matrix}, valid, true);
    auto emitLoop = [&](bool fast) -> Value {
      auto loop = b.create<scf::ForOp>(
          loc, zero, t.integer(totalK / k), t.integer(1),
          ValueRange{t.like(t.scalar(0), matrix)});
      OpBuilder::InsertionGuard ig(b);
      b.setInsertionPointToStart(loop.getBody());
      Value i = loop.getInductionVar(), acc = loop.getRegionIterArgs()[0];
      auto left = read(sa, {m, g}, {xblock, i}, false);
      auto right = scale.column == 0
                       ? read(scale.view, {n, g}, {yblock, i}, false)
                       : read(scale.view, {g, n}, {i, yblock}, false);
      if (scale.column == 0)
        right = t.op("transpose", right, t.shaped({g, n}, t.elem(right.getType())))
                    ->getResult(0);
      auto lhsBytes = read(a, {m, kBytes}, {xblock, i}, true);
      auto rhsBytes = read(rhs, {kBytes, n}, {i, yblock}, true);
      Value next;
      if (native && fast) {
        next = t
                   .op("scaled_mma",
                       {lhsBytes, left, rhsBytes, right, acc}, matrix,
                       {b.getNamedAttr("format", b.getStringAttr("e2m1")),
                        b.getNamedAttr(
                            "ci.implementation",
                            b.getStringAttr("exits/cutile::scaled-mma"))})
                   ->getResult(0);
      } else {
        Value kk = t.lanes(k);
        Value odd = b.create<arith::RemUIOp>(loc, kk, lit(2, kk.getType()));
        Value sh1D = castTo(
            "arith.trunci",
            b.create<arith::MulIOp>(loc, odd, lit(4, kk.getType())),
            b.getI32Type());
        auto decodeFp4 = [&](Value codeI32) {
          Type ty = codeI32.getType();
          Type fTy = cast<RankedTensorType>(ty).clone(b.getF32Type());
          Value s = b.create<arith::AndIOp>(
              loc, b.create<arith::ShRUIOp>(loc, codeI32, lit(3, ty)), lit(1, ty));
          Value e = b.create<arith::AndIOp>(
              loc, b.create<arith::ShRUIOp>(loc, codeI32, lit(1, ty)), lit(3, ty));
          Value mant = b.create<arith::AndIOp>(loc, codeI32, lit(1, ty));
          Value isSub = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                e, lit(0, ty));
          Value safeShift = b.create<arith::SelectOp>(
              loc, isSub, lit(0, ty),
              b.create<arith::SubIOp>(loc, e, lit(1, ty)));
          Value normDbl = b.create<arith::ShLIOp>(
              loc, b.create<arith::AddIOp>(loc, mant, lit(2, ty)), safeShift);
          Value magF = castTo(
              "arith.sitofp",
              b.create<arith::SelectOp>(loc, isSub, mant, normDbl),
              b.getF32Type());
          Value neg = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, s,
                                              lit(0, ty));
          Value halfS = b.create<arith::SelectOp>(
              loc, neg, t.like(t.scalar(-0.5f), fTy),
              t.like(t.scalar(0.5f), fTy));
          return b.create<arith::MulFOp>(loc, halfS, magF);
        };
        Value aU32 = castTo("arith.extui", lhsBytes, b.getI32Type());
        aU32 = b.create<arith::AndIOp>(loc, aU32, lit(255, aU32.getType()));
        Value aDup = reshape(
            t.op("expand_scale", aU32, t.shaped({m, kBytes, 2}, b.getI32Type()),
                 {b.getNamedAttr("axis", b.getI64IntegerAttr(2))})
                ->getResult(0),
            {m, k});
        Value aNib = b.create<arith::AndIOp>(
            loc,
            b.create<arith::ShRUIOp>(loc, aDup, t.like(sh1D, aDup.getType(), 1)),
            lit(15, aDup.getType()));
        Value rU32 = castTo("arith.extui", rhsBytes, b.getI32Type());
        rU32 = b.create<arith::AndIOp>(loc, rU32, lit(255, rU32.getType()));
        Value rDup = reshape(
            t.op("expand_scale", rU32, t.shaped({kBytes, 2, n}, b.getI32Type()),
                 {b.getNamedAttr("axis", b.getI64IntegerAttr(1))})
                ->getResult(0),
            {k, n});
        Value rNib = b.create<arith::AndIOp>(
            loc,
            b.create<arith::ShRUIOp>(loc, rDup, t.like(sh1D, rDup.getType(), 0)),
            lit(15, rDup.getType()));
        auto rescaleFp4 = [&](Value valF32, Value sTile, ArrayRef<int64_t> dims,
                              int axis) {
          Value codes = castTo("arith.extui", sTile, b.getI32Type());
          codes = b.create<arith::AndIOp>(loc, codes, lit(255, codes.getType()));
          auto factor =
              t.op("e8m0", codes,
                   cast<RankedTensorType>(sTile.getType()).clone(b.getF32Type()))
                  ->getResult(0);
          Value reshaped = reshape(valF32, dims);
          auto scaled = t.op("expand_scale", factor, reshaped.getType(),
                             {b.getNamedAttr("axis", b.getI64IntegerAttr(axis))})
                            ->getResult(0);
          Value out = b.create<arith::MulFOp>(loc, reshaped, scaled);
          return fast ? castTo("arith.truncf", out, b.getBF16Type()) : out;
        };
        next = t.dot(
            reshape(rescaleFp4(decodeFp4(aNib), left, {m, g, 32}, 2), {m, k}),
            reshape(rescaleFp4(decodeFp4(rNib), right, {g, 32, n}, 1), {k, n}),
            false, acc);
      }
      b.create<scf::YieldOp>(loc, next);
      return loop.getResult(0);
    };
    b.setInsertionPointToStart(branch.thenBlock());
    b.create<scf::YieldOp>(loc, emitLoop(true));
    b.setInsertionPointToStart(branch.elseBlock());
    b.create<scf::YieldOp>(loc, emitLoop(false));
    b.setInsertionPointAfter(branch);
    return branch.getResult(0);
  }
  auto format = cast<intentdsl::kir::MX8Type>(
                    p.dot.getAttrOfType<TypeAttr>("kir.storage").getValue())
                    .getLayout();
  auto m = p.tile[0], n = p.tile[1], k = p.tile[2], g = k / format.group;
  auto *dispatch = p.dot.getPrevNode();
  auto range = dispatch->getAttrOfType<DenseI64ArrayAttr>("interval");
  bool native = dispatch->getAttrOfType<StringAttr>("on_match") == "mx";
  auto flat = p.dot.getAttrOfType<DenseI64ArrayAttr>("ct_src.flat_groups");
  auto a = p.dot.getOperand(0), sa = p.dot.getOperand(1),
       rhs = p.dot.getOperand(2);
  intentdsl::kir::RightScale scale(p.dot);
  auto reshape = [&](Value v, ArrayRef<int64_t> shape) {
    return t.op("reshape", v, t.shaped(shape, t.elem(v.getType())))
        ->getResult(0);
  };
  auto castTo = [&](StringRef op, Value v, Type e) {
    OperationState st(loc, op);
    st.addOperands(v);
    st.addTypes(cast<RankedTensorType>(v.getType()).clone(e));
    return b.create(st)->getResult(0);
  };
  auto read = [&](Value base, ArrayRef<int64_t> shape, ValueRange indices,
                  bool allowTma) {
    SmallVector<Value> args{base};
    llvm::append_range(args, indices);
    return t
        .op("rect_read", args,
            t.shaped(shape, cast<MemRefType>(base.getType()).getElementType()),
            {b.getNamedAttr(
                "allow_tma",
                b.getBoolAttr(allowTma && t.fn->hasAttr("ct_src.tma")))})
        ->getResult(0);
  };
  auto xblock = t.pid(0), yblock = t.pid(1), zero = t.integer(0);
  auto matrix = t.shaped({m, n}, b.getF32Type());
  auto loop = b.create<scf::ForOp>(
      loc, zero, t.integer(llvm::divideCeil(p.shape[2], k)), t.integer(1),
      ValueRange{t.like(t.scalar(0), matrix)});
  b.setInsertionPointToStart(loop.getBody());
  Value i = loop.getInductionVar(), acc = loop.getRegionIterArgs()[0];
  auto left = read(sa, {m, g}, {xblock, i}, false);
  auto right = scale.column == 0 ? read(scale.view, {n, g}, {yblock, i}, false)
                                 : read(scale.view, {g, n}, {i, yblock}, false);
  if (scale.column == 0)
    right = t.op("transpose", right, t.shaped({g, n}, t.elem(right.getType())))
                ->getResult(0);
  auto linear = [&](Value view) {
    return llvm::is_contained(flat.asArrayRef(), p.slot(view));
  };
  auto lhs = linear(a)
                 ? read(a, {m, k}, {xblock, i}, native)
                 : read(a, {m, g, format.group}, {xblock, i, zero}, native);
  auto rv = linear(rhs)
                ? read(rhs, {k, n}, {i, yblock}, native)
                : read(rhs, {g, format.group, n}, {i, zero, yblock}, native);
  auto inside = [&](Value v) {
    auto count = cast<RankedTensorType>(v.getType()).getNumElements();
    auto codes = castTo("arith.extui", reshape(v, {count}), b.getI32Type());
    return b.create<arith::AndIOp>(
        loc,
        b.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::sge, t.reduce(codes, "min", 0),
            b.create<arith::ConstantIntOp>(loc, range[0], 32)),
        b.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt, t.reduce(codes, "max", 0),
            b.create<arith::ConstantIntOp>(loc, range[1] + 1, 32)));
  };
  auto valid = b.create<arith::AndIOp>(loc, inside(left), inside(right));
  auto branch = b.create<scf::IfOp>(loc, TypeRange{matrix}, valid, true);
  auto product = [&](bool fast) -> Value {
    if (native && fast)
      return t
          .op("scaled_mma",
              {reshape(lhs, {m, k}), left, reshape(rv, {k, n}), right, acc},
              matrix,
              {b.getNamedAttr("ci.implementation",
                              b.getStringAttr("exits/cutile::scaled-mma"))})
          ->getResult(0);
    auto rescale = [&](Value x, Value s, ArrayRef<int64_t> dims, int axis) {
      auto codes = castTo("arith.extui", s, b.getI32Type());
      auto factor =
          t.op("e8m0", codes,
               cast<RankedTensorType>(s.getType()).clone(b.getF32Type()))
              ->getResult(0);
      auto val = castTo("arith.extf", reshape(x, dims), b.getF32Type());
      auto scaled = t.op("expand_scale", factor, val.getType(),
                         {b.getNamedAttr("axis", b.getI64IntegerAttr(axis))})
                        ->getResult(0);
      Value out = b.create<arith::MulFOp>(loc, val, scaled);
      return fast ? castTo("arith.truncf", out, b.getBF16Type()) : out;
    };
    return t.dot(reshape(rescale(lhs, left, {m, g, format.group}, 2), {m, k}),
                 reshape(rescale(rv, right, {g, format.group, n}, 1), {k, n}),
                 false, acc);
  };
  b.setInsertionPointToStart(branch.thenBlock());
  b.create<scf::YieldOp>(loc, product(dispatch->getAttrOfType<StringAttr>(
                                          "on_match") != "float32"));
  b.setInsertionPointToStart(branch.elseBlock());
  b.create<scf::YieldOp>(loc, product(false));
  b.setInsertionPointAfter(branch);
  b.create<scf::YieldOp>(loc, branch.getResults());
  b.setInsertionPointAfter(loop);
  return loop.getResult(0);
}
} // namespace CI_UNIT_NAMESPACE
