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
  b.create<scf::YieldOp>(loc, product(true));
  b.setInsertionPointToStart(branch.elseBlock());
  b.create<scf::YieldOp>(loc, product(false));
  b.setInsertionPointAfter(branch);
  b.create<scf::YieldOp>(loc, branch.getResults());
  b.setInsertionPointAfter(loop);
  return loop.getResult(0);
}
} // namespace CI_UNIT_NAMESPACE
