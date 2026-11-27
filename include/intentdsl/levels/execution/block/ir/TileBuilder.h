#pragma once
#include "intentdsl/levels/execution/block/ir/MatrixAccess.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"

namespace intentdsl::block {
using namespace mlir;
struct Tiles {
  OpBuilder b;
  Location loc;
  func::FuncOp fn;
  Operation *program;
  Operation *sourceDot = nullptr;
  StringRef dialect, unit;
  std::string symbol(StringRef name) const {
    return (dialect + "." + name).str();
  }
  Tiles(func::FuncOp f, StringRef convention, StringRef unit)
      : b(&f.front().front()), loc(f.getLoc()), fn(f),
        dialect(f.front().front().getName().getDialectNamespace()), unit(unit) {
    auto &original = f.front().front();
    OperationState s(loc, symbol("tile_program"));
    s.addOperands(original.getOperands());
    s.addAttributes(original.getAttrs());
    s.addAttribute("calling_convention", b.getStringAttr(convention));
    s.addRegion();
    program = b.create(s);
    auto *body = b.createBlock(&program->getRegion(0));
    b.setInsertionPointToStart(body);
  }
  Tiles(func::FuncOp f, StringRef convention, StringRef unit,
        StringRef dialectName, ValueRange domainOperands)
      : b(&f.front().front()), loc(f.getLoc()), fn(f), dialect(dialectName),
        unit(unit) {
    OperationState s(loc, symbol("tile_program"));
    s.addOperands(domainOperands);
    s.addAttribute("calling_convention", b.getStringAttr(convention));
    s.addRegion();
    program = b.create(s);
    auto *body = b.createBlock(&program->getRegion(0));
    b.setInsertionPointToStart(body);
  }
  Operation *op(StringRef name, ValueRange args = {}, TypeRange results = {},
                ArrayRef<NamedAttribute> attrs = {}) {
    OperationState s(loc, symbol(name));
    s.addOperands(args);
    s.addTypes(results);
    s.addAttributes(attrs);
    return b.create(s);
  }
  Type elem(Type t) { return getElementTypeOrSelf(t); }
  Type shaped(ArrayRef<int64_t> shape, Type t) {
    return shape.empty() ? t : RankedTensorType::get(shape, t);
  }
  Value like(Value v, Type type, int axis = -1) {
    if (type == v.getType())
      return v;
    return op("tile_broadcast", v, type,
              {b.getNamedAttr("axis", b.getI64IntegerAttr(axis))})
        ->getResult(0);
  }
  Value integer(int64_t i) {
    return b.create<arith::ConstantIntOp>(loc, i, 64);
  }
  Value scalar(float f) {
    return b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(f));
  }
  Value pid(int axis) {
    return op("tile_pid", {}, b.getI64Type(),
              {b.getNamedAttr("axis", b.getI64IntegerAttr(axis))})
        ->getResult(0);
  }
  Value lanes(int64_t n) {
    return op("tile_axis", {}, shaped({n}, b.getI64Type()))->getResult(0);
  }
  Value stepped(Value first, int64_t n) {
    auto r = lanes(n);
    return b.create<arith::AddIOp>(loc, like(first, r.getType()), r);
  }
  Value address(Value base, ValueRange coordinates,
                ArrayRef<int64_t> projection) {
    SmallVector<int64_t> dims;
    for (auto [index, axis] : llvm::zip(coordinates, projection))
      if (axis >= 0) {
        dims.resize(std::max(dims.size(), size_t(axis + 1)), 1);
        dims[axis] = cast<RankedTensorType>(index.getType()).getDimSize(0);
      }
    SmallVector<Value> args{base};
    llvm::append_range(args, coordinates);
    auto strides = fn.getArgAttr(cast<BlockArgument>(base).getArgNumber(),
                                 "block.strides");
    return op("tile_address", args, shaped(dims, b.getI64Type()),
              {b.getNamedAttr("projection", b.getDenseI64ArrayAttr(projection)),
               b.getNamedAttr("strides", strides),
               b.getNamedAttr("index_bits", b.getI32IntegerAttr(64))})
        ->getResult(0);
  }
  Value read(Value base, Value addr, Value mask = {}) {
    SmallVector<Value> args{base, addr};
    if (mask)
      args.push_back(mask);
    auto t = cast<MemRefType>(base.getType()).getElementType();
    if (auto s = dyn_cast<RankedTensorType>(addr.getType()))
      t = s.clone(t);
    return op("tile_read", args, t)->getResult(0);
  }
  void write(Value base, Value addr, Value val, Value mask = {}) {
    SmallVector<Value> args{base, addr, val};
    if (mask)
      args.push_back(mask);
    op("tile_write", args);
  }
  void atomicAdd(Value base, Value addr, Value val, Value mask = {}) {
    SmallVector<Value> args{base, addr, val};
    if (mask)
      args.push_back(mask);
    SmallVector<NamedAttribute, 2> attrs{
        b.getNamedAttr("ci.implementation",
                       b.getStringAttr((unit + "::tile-atomic-add").str())),
        b.getNamedAttr("order", b.getStringAttr("relaxed"))};
    if (isa<FloatType>(elem(val.getType())))
      attrs.push_back(b.getNamedAttr(
          "fastmath", arith::FastMathFlagsAttr::get(
                          b.getContext(), arith::FastMathFlags::reassoc)));
    op("tile_atomic_add", args, {}, attrs);
  }
  Value atomicFetchAdd(Value base, Value addr, Value val, Value mask,
                       StringRef order, StringRef scope) {
    SmallVector<Value> args{base, addr, val};
    if (mask)
      args.push_back(mask);
    SmallVector<NamedAttribute, 4> attrs{
        b.getNamedAttr("order", b.getStringAttr(order)),
        b.getNamedAttr("scope", b.getStringAttr(scope)),
        b.getNamedAttr(
            "ci.implementation",
            b.getStringAttr((unit + "::tile-atomic-fetch-add").str()))};
    if (isa<FloatType>(elem(val.getType())))
      attrs.push_back(b.getNamedAttr(
          "fastmath", arith::FastMathFlagsAttr::get(
                          b.getContext(), arith::FastMathFlags::reassoc)));
    return op("tile_atomic_fetch_add", args, val.getType(), attrs)
        ->getResult(0);
  }
  void atomicXchg(Value base, Value addr, Value val, Value mask,
                  StringRef order, StringRef scope) {
    SmallVector<Value> args{base, addr, val};
    if (mask)
      args.push_back(mask);
    SmallVector<NamedAttribute, 3> attrs{
        b.getNamedAttr("order", b.getStringAttr(order)),
        b.getNamedAttr("scope", b.getStringAttr(scope)),
        b.getNamedAttr("ci.implementation",
                       b.getStringAttr((unit + "::tile-atomic-xchg").str()))};
    op("tile_atomic_xchg", args, {}, attrs);
  }
  Value tileGather(Value src, Value indices, int64_t axis) {
    auto sTy = cast<RankedTensorType>(src.getType());
    auto iTy = cast<RankedTensorType>(indices.getType());
    Type outTy = iTy.clone(sTy.getElementType());
    return op("tile_gather", {src, indices}, outTy,
              {b.getNamedAttr("axis", b.getI64IntegerAttr(axis)),
               b.getNamedAttr(
                   "ci.implementation",
                   b.getStringAttr((unit + "::tile-gather").str()))})
        ->getResult(0);
  }
  Value tileHistogram(Value keys, Value mask, int64_t bins) {
    Type outTy = shaped({bins}, b.getI32Type());
    return op("tile_histogram", {keys, mask}, outTy,
              {b.getNamedAttr("bins", b.getI64IntegerAttr(bins)),
               b.getNamedAttr(
                   "ci.implementation",
                   b.getStringAttr((unit + "::tile-histogram").str()))})
        ->getResult(0);
  }
  Value reshapeTile(Value val, ArrayRef<int64_t> dims) {
    return op("reshape", val, shaped(dims, elem(val.getType())))->getResult(0);
  }
  Value dot(Value a, Value rhs, bool trans, Value acc = {}) {
    auto l = cast<RankedTensorType>(a.getType()),
         r = cast<RankedTensorType>(rhs.getType());
    auto t = shaped({l.getDimSize(0), r.getDimSize(trans ? 0 : 1)},
                    l.getElementType().isInteger(8) ? Type(b.getI32Type())
                                                    : Type(b.getF32Type()));
    auto elemType = l.getElementType();
    bool scaled =
        sourceDot && sourceDot->getName().stripDialect() == "scaled_contract";
    if (sourceDot &&
        (elemType.isBF16() || elemType.isF16() ||
         (scaled && elemType.isF32())) &&
        r.getElementType() == elemType) {
      auto mode =
          fn->getAttrOfType<StringAttr>(scaled              ? "kir.fp8_dot"
                                        : elemType.isBF16() ? "kir.bf16_dot"
                                                            : "kir.f16_dot");
      auto local = sourceDot->getAttrOfType<StringAttr>("denormal-fp-math");
      if (mode == "fp32-product" ||
          (local ? !compilerinfra::allowsSubnormalFlush(sourceDot)
                 : mode == "source")) {
        // PTX does not specify subnormal input handling for narrow MMA.
        // Express the required product rounding before the K reduction.
        auto m = l.getDimSize(0), k = l.getDimSize(1),
             n = r.getDimSize(trans ? 0 : 1);
        SmallVector<int64_t> dims = trans ? SmallVector<int64_t>{m, n, k}
                                          : SmallVector<int64_t>{m, k, n};
        auto expand = [&](Value v, int axis) {
          auto type = cast<RankedTensorType>(v.getType()).clone(b.getF32Type());
          Value wide = v.getType() == type
                           ? v
                           : Value(b.create<arith::ExtFOp>(loc, type, v));
          return op("expand_scale", wide, shaped(dims, b.getF32Type()),
                    {b.getNamedAttr("axis", b.getI64IntegerAttr(axis))})
              ->getResult(0);
        };
        auto lhs3 = expand(a, trans ? 1 : 2), rhs3 = expand(rhs, 0);
        auto sum = reduce(b.create<arith::MulFOp>(loc, lhs3, rhs3), "sum",
                          trans ? 2 : 1);
        return acc ? Value(b.create<arith::AddFOp>(loc, acc, sum)) : sum;
      }
    }
    SmallVector<Value> args{a, rhs};
    if (acc)
      args.push_back(acc);
    return op("tile_dot", args, t,
              {b.getNamedAttr("transpose_rhs", b.getBoolAttr(trans)),
               b.getNamedAttr("ci.implementation",
                              b.getStringAttr((unit + "::native-dot").str()))})
        ->getResult(0);
  }
  Value reduce(Value input, StringRef algebra, int64_t axis = 1) {
    auto t = cast<RankedTensorType>(input.getType());
    SmallVector<int64_t> dims(t.getShape());
    dims.erase(dims.begin() + axis);
    return op("tile_reduce", input, shaped(dims, t.getElementType()),
              {b.getNamedAttr("algebra", b.getStringAttr(algebra)),
               b.getNamedAttr("axis", b.getI64IntegerAttr(axis)),
               b.getNamedAttr(
                   "ci.implementation",
                   b.getStringAttr((unit + "::row-collective").str()))})
        ->getResult(0);
  }
  DenseI64ArrayAttr shape(Value view) {
    return fn.getArgAttrOfType<DenseI64ArrayAttr>(
        cast<BlockArgument>(view).getArgNumber(), "block.shape");
  }
};
} // namespace intentdsl::block
