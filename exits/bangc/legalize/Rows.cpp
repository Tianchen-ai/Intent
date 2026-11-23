#include "Rows.h"
#include "Slab.h"
#include "intentdsl/exits/bangc/ir/Dialect.h"
#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "intentdsl/levels/algorithm/kir/ir/Segment.h"
#include "intentdsl/levels/execution/block/ir/Grid.h"
#include "intentdsl/levels/storage/dsa/ir/Supply.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <set>

namespace CI_UNIT_NAMESPACE {
using namespace mlir;
using namespace compilerinfra;

llvm::Error describeRowArtifact(func::FuncOp fn, llvm::raw_pwrite_stream &out) {
  auto layout = fn->getAttrOfType<DictionaryAttr>("dsa.layout");
  if (!layout)
    return llvm::createStringError("missing supplied DSA layout");
  auto depth = layout.getAs<IntegerAttr>("depth").getInt();
  llvm::json::Array rows;
  for (auto a : layout.getAs<ArrayAttr>("rows")) {
    auto allocation = cast<DictionaryAttr>(a);
    auto id = allocation.getAs<IntegerAttr>("id");
    for (auto attr : fn->getAttrOfType<ArrayAttr>("block.configs")) {
      auto row = cast<DictionaryAttr>(attr);
      if (row.get("id") != id)
        continue;
      unsigned tasks = 0;
      if (row.getAs<StringAttr>("payload").getValue().getAsInteger(10, tasks))
        return llvm::createStringError("invalid physical task count");
      rows.push_back(llvm::json::Object{
          {"id", id.getInt()},
          {"rows", row.getAs<IntegerAttr>("rows").getInt()},
          {"width", row.getAs<IntegerAttr>("width").getInt()},
          {"bytes", allocation.getAs<IntegerAttr>("bytes").getInt()},
          {"pitch", allocation.getAs<IntegerAttr>("pitch").getInt()},
          {"tasks", layout.getAs<IntegerAttr>("participants").getInt()}});
    }
  }
  auto abi =
      llvm::json::parse(fn->getAttrOfType<StringAttr>("kir.abi").getValue());
  if (!abi)
    return abi.takeError();
  auto guard = fn->getAttrOfType<DenseI64ArrayAttr>("block.guard");
  llvm::json::Array axes;
  Builder builder(fn.getContext());
  auto mapping = layout.contains("independent_columns")
                     ? intentdsl::block::grid(builder, true)
                     : fn->getAttrOfType<ArrayAttr>("block.grid");
  for (auto expr : mapping)
    axes.push_back(intentdsl::block::gridExpression(expr));
  out << llvm::json::Value(llvm::json::Object{
      {"version", 2},
      {"source", ""},
      {"domain", intentdsl::publishDomain(fn)},
      {"abi", std::move(*abi)},
      {"rows", std::move(rows)},
      {"depth", depth},
      {"grid", llvm::json::Array{std::move(axes), 1, 1}},
      {"guard",
       llvm::json::Object{{"min_width", guard[0]},
                          {"max_width", guard[1] < 0
                                            ? llvm::json::Value(nullptr)
                                            : llvm::json::Value(guard[1])}}}});
  return llvm::Error::success();
}

namespace {
Type elemOf(Type t) {
  if (auto s = dyn_cast<ShapedType>(t))
    return s.getElementType();
  return t;
}

Type storageElemType(MLIRContext *ctx, Type t) {
  t = elemOf(t);
  if (t.isUnsignedInteger(64))
    return IntegerType::get(ctx, 64);
  return t;
}

MemRefType ptrType(MLIRContext *ctx, Type elem, int64_t space) {
  return MemRefType::get({ShapedType::kDynamic}, storageElemType(ctx, elem),
                         AffineMap{},
                         IntegerAttr::get(IntegerType::get(ctx, 32), space));
}

TupleType momentTuple(MLIRContext *ctx) {
  return TupleType::get(ctx, {IntegerType::get(ctx, 32), Float32Type::get(ctx),
                              Float32Type::get(ctx)});
}

TupleType momentArrayTuple(MLIRContext *ctx) {
  return TupleType::get(ctx, {IndexType::get(ctx), IntegerType::get(ctx, 32),
                              Float32Type::get(ctx), Float32Type::get(ctx)});
}

int rankOf(Type t) {
  if (auto r = dyn_cast<RankedTensorType>(t))
    return r.getRank();
  return 0;
}

struct RowLowerer {
  func::FuncOp fn;
  const TargetProfile &profile;
  OpBuilder b;
  Location loc;
  DictionaryAttr layout;
  bool resident = false;
  bool independent = false;
  bool wide = false;
  int64_t depth = 1;
  int64_t batch = 1;
  bool multiMember = true;
  Operation *constAnchor = nullptr;
  llvm::DenseMap<int64_t, Value> constI64Map;
  llvm::DenseMap<int32_t, Value> constI32Map;
  llvm::DenseMap<std::pair<Value, int64_t>, Value> dimCache;
  llvm::DenseMap<std::pair<Value, int64_t>, Value> strideCache;
  Value storage;
  Value symM, symN, symGrid, symTaskId, symTaskDim, symR, symB, symEB, symW,
      symP, symBytes;
  llvm::DenseMap<Value, Value> mapped;

  RowLowerer(func::FuncOp f, const TargetProfile &p)
      : fn(f), profile(p), b(f.getContext()), loc(f.getLoc()),
        layout(f->getAttrOfType<DictionaryAttr>("dsa.layout")) {
    resident = fn->getAttrOfType<StringAttr>("block.topology") == "resident";
    independent = layout.contains("independent_columns");
    wide = layout.getAs<BoolAttr>("group_temporaries").getValue();
    depth = layout.getAs<IntegerAttr>("depth").getInt();
    batch = layout.getAs<IntegerAttr>("batch")
                ? layout.getAs<IntegerAttr>("batch").getInt()
                : 1;
    int64_t maxRows = 1;
    if (auto rowsAttr = layout.getAs<ArrayAttr>("rows")) {
      for (auto a : rowsAttr) {
        auto id = cast<DictionaryAttr>(a).getAs<IntegerAttr>("id");
        for (auto cfg : fn->getAttrOfType<ArrayAttr>("block.configs")) {
          auto row = cast<DictionaryAttr>(cfg);
          if (row.get("id") == id)
            maxRows = std::max(maxRows, row.getAs<IntegerAttr>("rows").getInt());
        }
      }
    }
    multiMember = (maxRows * batch) > 1;
  }

  Value cI64(int64_t v) {
    if (!constAnchor)
      return b.create<arith::ConstantIntOp>(loc, v, 64);
    if (auto it = constI64Map.find(v); it != constI64Map.end())
      return it->second;
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointAfter(constAnchor);
    Value res = b.create<arith::ConstantIntOp>(loc, v, 64);
    constAnchor = res.getDefiningOp();
    constI64Map[v] = res;
    return res;
  }
  Value cI32(int32_t v) {
    if (!constAnchor)
      return b.create<arith::ConstantIntOp>(loc, v, 32);
    if (auto it = constI32Map.find(v); it != constI32Map.end())
      return it->second;
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointAfter(constAnchor);
    Value res = b.create<arith::ConstantIntOp>(loc, v, 32);
    constAnchor = res.getDefiningOp();
    constI32Map[v] = res;
    return res;
  }
  Value cF32(float v) {
    return b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(v));
  }
  Value toIdx(Value v) {
    return v.getType().isIndex()
               ? v
               : b.create<arith::IndexCastOp>(loc, b.getIndexType(), v)
                     .getResult();
  }
  Value toI64(Value v) {
    if (v.getType().isSignlessInteger(64))
      return v;
    if (v.getType().isIndex())
      return b.create<arith::IndexCastOp>(loc, b.getI64Type(), v).getResult();
    if (isa<FloatType>(v.getType()))
      return b.create<arith::FPToSIOp>(loc, b.getI64Type(), v).getResult();
    return b.create<arith::ExtSIOp>(loc, b.getI64Type(), v).getResult();
  }

  scf::ForOp forLoop(Value lb, Value ub, Value step,
                     llvm::function_ref<void(Value)> bodyFn) {
    Value lbIdx = toIdx(lb), ubIdx = toIdx(ub), stIdx = toIdx(step);
    return b.create<scf::ForOp>(
        loc, lbIdx, ubIdx, stIdx, ValueRange{},
        [&](OpBuilder &nb, Location l, Value iv, ValueRange) {
          OpBuilder::InsertionGuard g(b);
          b.restoreInsertionPoint(nb.saveInsertionPoint());
          bodyFn(toI64(iv));
          b.create<scf::YieldOp>(l);
        });
  }

  scf::ForOp
  forLoopCarry(Value lb, Value ub, Value step, ValueRange inits,
               llvm::function_ref<void(Value, ValueRange)> bodyFn) {
    Value lbIdx = toIdx(lb), ubIdx = toIdx(ub), stIdx = toIdx(step);
    return b.create<scf::ForOp>(
        loc, lbIdx, ubIdx, stIdx, inits,
        [&](OpBuilder &nb, Location, Value iv, ValueRange args) {
          OpBuilder::InsertionGuard g(b);
          b.restoreInsertionPoint(nb.saveInsertionPoint());
          bodyFn(toI64(iv), args);
        });
  }

  void ifThen(Value cond, llvm::function_ref<void()> thenFn) {
    b.create<scf::IfOp>(loc, cond, [&](OpBuilder &nb, Location l) {
      OpBuilder::InsertionGuard g(b);
      b.restoreInsertionPoint(nb.saveInsertionPoint());
      thenFn();
      b.create<scf::YieldOp>(l);
    });
  }

  scf::IfOp ifThenElse(Value cond, llvm::function_ref<void()> thenFn,
                       llvm::function_ref<void()> elseFn,
                       bool autoYield = true) {
    return b.create<scf::IfOp>(
        loc, cond,
        [&](OpBuilder &nb, Location l) {
          OpBuilder::InsertionGuard g(b);
          b.restoreInsertionPoint(nb.saveInsertionPoint());
          thenFn();
          if (autoYield)
            b.create<scf::YieldOp>(l);
        },
        [&](OpBuilder &nb, Location l) {
          OpBuilder::InsertionGuard g(b);
          b.restoreInsertionPoint(nb.saveInsertionPoint());
          elseFn();
          if (autoYield)
            b.create<scf::YieldOp>(l);
        });
  }

  Operation *makeOp(StringRef name, ValueRange operands, TypeRange results = {},
                    ArrayRef<NamedAttribute> attrs = {},
                    StringRef contract = {}) {
    OperationState st(loc, ("bangc_src." + name).str());
    st.addOperands(operands);
    st.addTypes(results);
    st.addAttributes(attrs);
    if (!contract.empty())
      st.addAttribute("ci.implementation",
                      b.getStringAttr(("exits/bangc::" + contract).str()));
    return b.create(st);
  }

  Value makeSym(StringRef s) {
    return makeOp("symbol", {}, {b.getI64Type()},
                  {b.getNamedAttr("name", b.getStringAttr(s))})
        ->getResult(0);
  }
  Value viewDim(Value view, int64_t axis) {
    if (constAnchor && isa<BlockArgument>(view)) {
      auto key = std::make_pair(view, axis);
      if (auto it = dimCache.find(key); it != dimCache.end())
        return it->second;
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointAfter(constAnchor);
      Value res = makeOp("dim", {view}, {b.getI64Type()},
                         {b.getNamedAttr("axis", b.getI64IntegerAttr(axis))})
                      ->getResult(0);
      constAnchor = res.getDefiningOp();
      dimCache[key] = res;
      return res;
    }
    return makeOp("dim", {view}, {b.getI64Type()},
                  {b.getNamedAttr("axis", b.getI64IntegerAttr(axis))})
        ->getResult(0);
  }
  Value viewStride(Value view, int64_t axis) {
    if (constAnchor && isa<BlockArgument>(view)) {
      auto key = std::make_pair(view, axis);
      if (auto it = strideCache.find(key); it != strideCache.end())
        return it->second;
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointAfter(constAnchor);
      Value res = makeOp("stride", {view}, {b.getI64Type()},
                         {b.getNamedAttr("axis", b.getI64IntegerAttr(axis))})
                      ->getResult(0);
      constAnchor = res.getDefiningOp();
      strideCache[key] = res;
      return res;
    }
    return makeOp("stride", {view}, {b.getI64Type()},
                  {b.getNamedAttr("axis", b.getI64IntegerAttr(axis))})
        ->getResult(0);
  }
  Value makeAddr(Value base, ArrayRef<Value> coords, ArrayRef<Value> strides,
                 int64_t space) {
    auto baseTy = cast<MemRefType>(base.getType());
    SmallVector<Value> ops{base};
    ops.append(coords.begin(), coords.end());
    ops.append(strides.begin(), strides.end());
    int32_t n = static_cast<int32_t>(coords.size());
    return makeOp("address", ops,
                  {ptrType(b.getContext(), baseTy.getElementType(), space)},
                  {b.getNamedAttr("index_bits", b.getI32IntegerAttr(64)),
                   b.getNamedAttr("operandSegmentSizes",
                                  b.getDenseI32ArrayAttr({1, n, n}))})
        ->getResult(0);
  }
  Value nramAt(Type elemTy, Value byteOffset) {
    return makeOp("nram_ptr", {storage, byteOffset},
                  {ptrType(b.getContext(), elemTy, 1)})
        ->getResult(0);
  }
  Value phaseByteOffset(DenseI64ArrayAttr parts) {
    Value gTerm = cI64(parts[0] * depth);
    gTerm = b.create<arith::MulIOp>(loc, gTerm, symEB);
    Value lTerm = cI64(parts[1]);
    if (wide)
      lTerm = b.create<arith::MulIOp>(loc, lTerm, symEB);
    Value elems = b.create<arith::AddIOp>(loc, gTerm, lTerm);
    return b.create<arith::MulIOp>(loc, symP, elems);
  }
  Value localVar(Type elemTy, StringRef extent) {
    return makeOp("local_var", {}, {ptrType(b.getContext(), elemTy, 2)},
                  {b.getNamedAttr("extent", b.getStringAttr(extent))})
        ->getResult(0);
  }
  Value loadAt(Value ptr, Value idx, Type resTy = {}) {
    if (!resTy)
      resTy = cast<MemRefType>(ptr.getType()).getElementType();
    return makeOp("load_scalar", {ptr, idx}, {resTy})->getResult(0);
  }
  void storeAt(Value ptr, Value idx, Value val) {
    makeOp("store_scalar", {ptr, idx, val});
  }
  void memcpy1D(Value dst, Value src, Value elems, StringRef dir, bool async) {
    makeOp("memcpy_1d", {dst, src, elems}, {},
           {b.getNamedAttr("direction", b.getStringAttr(dir)),
            b.getNamedAttr("async", b.getBoolAttr(async))});
  }
  void memcpy2D(Value dst, Value src, Value width, Value dstStride,
                Value srcStride, Value segNum, StringRef dir, bool async) {
    makeOp("memcpy_2d", {dst, src, width, dstStride, srcStride, segNum}, {},
           {b.getNamedAttr("direction", b.getStringAttr(dir)),
            b.getNamedAttr("async", b.getBoolAttr(async))});
  }
  void sync(StringRef dom) {
    makeOp("sync", {}, {}, {b.getNamedAttr("domain", b.getStringAttr(dom))});
  }
  void fillVec(Value dst, Value lanes, Value val) {
    makeOp("fill", {dst, lanes, val});
  }

  Value scalarConst(TypedAttr attr) {
    auto ty = attr.getType();
    if (ty.isUnsignedInteger(64))
      return b.create<arith::ConstantOp>(loc, ty, attr);
    return b.create<arith::ConstantOp>(loc, attr);
  }

  Value scalarCast(Value v, Type dstTy) {
    Type srcTy = v.getType();
    if (srcTy == dstTy)
      return v;
    if (srcTy.isBF16() && dstTy.isF32())
      return makeOp("bf16_scalar", {v}, {dstTy},
                    {b.getNamedAttr("mode", b.getStringAttr("widen"))},
                    "bf16-scalar-convert")
          ->getResult(0);
    if (srcTy.isF32() && dstTy.isBF16())
      return makeOp("bf16_scalar", {v}, {dstTy},
                    {b.getNamedAttr("mode", b.getStringAttr("narrow"))},
                    "bf16-scalar-convert")
          ->getResult(0);
    if (srcTy.isBF16() && dstTy.isF16()) {
      Value f = scalarCast(v, b.getF32Type());
      return b.create<arith::TruncFOp>(loc, dstTy, f);
    }
    if (srcTy.isF16() && dstTy.isBF16()) {
      Value f = b.create<arith::ExtFOp>(loc, b.getF32Type(), v);
      return scalarCast(f, dstTy);
    }
    if (isa<FloatType>(srcTy) && isa<FloatType>(dstTy)) {
      return srcTy.getIntOrFloatBitWidth() < dstTy.getIntOrFloatBitWidth()
                 ? b.create<arith::ExtFOp>(loc, dstTy, v).getResult()
                 : b.create<arith::TruncFOp>(loc, dstTy, v).getResult();
    }
    if (srcTy.isInteger(1) && isa<FloatType>(dstTy))
      return b.create<arith::UIToFPOp>(loc, dstTy, v);
    if (srcTy.isInteger(1) && dstTy.isSignlessInteger())
      return b.create<arith::ExtUIOp>(loc, dstTy, v);
    if (srcTy.isSignlessInteger() && isa<FloatType>(dstTy))
      return b.create<arith::SIToFPOp>(loc, dstTy, v);
    if (isa<FloatType>(srcTy) && dstTy.isSignlessInteger())
      return b.create<arith::FPToSIOp>(loc, dstTy, v);
    if (srcTy.isSignlessInteger() && dstTy.isSignlessInteger()) {
      return srcTy.getIntOrFloatBitWidth() < dstTy.getIntOrFloatBitWidth()
                 ? b.create<arith::ExtSIOp>(loc, dstTy, v).getResult()
                 : b.create<arith::TruncIOp>(loc, dstTy, v).getResult();
    }
    return makeOp("scalar_math", {v}, {dstTy},
                  {b.getNamedAttr("op", b.getStringAttr("cast"))})
        ->getResult(0);
  }

  LogicalResult evalScalarExpr(StringRef code, StringRef prim, ValueRange ops,
                               Type dstTy, TypedAttr constAttr, Value &out) {
    if (code == "constant") {
      out = scalarConst(constAttr);
      return success();
    }
    if (prim == "bf16_product") {
      out = makeOp("bf16_scalar", {ops[0], ops[1]}, {dstTy},
                   {b.getNamedAttr("mode", b.getStringAttr("product"))},
                   "bf16-scalar-convert")
                ->getResult(0);
      return success();
    }
    if (prim == "bf16_widen" || prim == "bf16_round" || code == "cast") {
      out = scalarCast(ops[0], dstTy);
      return success();
    }
    if (code == "sqrt" || code == "rsqrt" || code == "exp") {
      Value in = scalarCast(ops[0], b.getF32Type());
      Value r = makeOp("scalar_math", {in}, {b.getF32Type()},
                       {b.getNamedAttr("op", b.getStringAttr(code))})
                    ->getResult(0);
      out = scalarCast(r, dstTy);
      return success();
    }
    if (code == "select") {
      out = b.create<arith::SelectOp>(loc, ops[0], ops[1], ops[2]);
      return success();
    }
    if (code == "ne") {
      if (isa<FloatType>(ops[0].getType()))
        out = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNE, ops[0],
                                      ops[1]);
      else
        out = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, ops[0],
                                      ops[1]);
      return success();
    }
    if (code == "gt") {
      if (isa<FloatType>(ops[0].getType()))
        out = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, ops[0],
                                      ops[1]);
      else
        out = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, ops[0],
                                      ops[1]);
      return success();
    }
    if (code == "and") {
      out = b.create<arith::AndIOp>(loc, ops[0], ops[1]);
      return success();
    }
    if (code == "xor") {
      out = b.create<arith::XOrIOp>(loc, ops[0], ops[1]);
      return success();
    }
    if (code == "shl") {
      out = b.create<arith::ShLIOp>(loc, ops[0], ops[1]);
      return success();
    }
    if (code == "shr") {
      out = b.create<arith::ShRSIOp>(loc, ops[0], ops[1]);
      return success();
    }
    if (code == "maximum") {
      Value a = scalarCast(ops[0], b.getF32Type());
      Value c = scalarCast(ops[1], b.getF32Type());
      Value r = makeOp("scalar_math", {a, c}, {b.getF32Type()},
                       {b.getNamedAttr("op", b.getStringAttr("maximum_nan"))})
                    ->getResult(0);
      out = scalarCast(r, dstTy);
      return success();
    }
    if (code == "combine") {
      if (prim == "sum")
        out = b.create<arith::AddFOp>(loc, ops[0], ops[1]);
      else
        out = makeOp("scalar_math", {ops[0], ops[1]}, {dstTy},
                     {b.getNamedAttr("op", b.getStringAttr("max_num"))})
                  ->getResult(0);
      return success();
    }
    if (code == "add" || code == "sub" || code == "mul" || code == "div") {
      if (dstTy.isSignlessInteger()) {
        out = code == "add" ? b.create<arith::AddIOp>(loc, ops[0], ops[1]).getResult()
              : code == "sub" ? b.create<arith::SubIOp>(loc, ops[0], ops[1]).getResult()
              : code == "mul" ? b.create<arith::MulIOp>(loc, ops[0], ops[1]).getResult()
                              : b.create<arith::DivSIOp>(loc, ops[0], ops[1]).getResult();
        return success();
      }
      Value a = ops[0], c = ops[1];
      if (dstTy.isF16()) {
        a = scalarCast(a, b.getF32Type());
        c = scalarCast(c, b.getF32Type());
      }
      Value r = code == "add"   ? b.create<arith::AddFOp>(loc, a, c).getResult()
                : code == "sub" ? b.create<arith::SubFOp>(loc, a, c).getResult()
                : code == "mul" ? b.create<arith::MulFOp>(loc, a, c).getResult()
                                : b.create<arith::DivFOp>(loc, a, c).getResult();
      out = scalarCast(r, dstTy);
      return success();
    }
    return failure();
  }

  Value buildRowPtr(Operation &op, unsigned start, Value view, Value rowIdx,
                    Value col, Value &outColStride) {
    SmallVector<Value> coords, strides;
    outColStride = cI64(0);
    for (auto [i, c] : llvm::enumerate(op.getOperands().drop_front(start))) {
      auto res = cast<OpResult>(c);
      Value st = viewStride(view, i);
      auto *def = c.getDefiningOp();
      if (def->getName().getStringRef() == "dsa.lookup_axis") {
        Value ids = mapped.lookup(def->getOperand(0));
        Value idsS0 = viewStride(ids, 0);
        Value idsPtr = makeAddr(ids, {rowIdx}, {idsS0}, 0);
        Value loaded = loadAt(idsPtr, cI64(0));
        coords.push_back(toI64(loaded));
        strides.push_back(st);
      } else if (def->getName().getStringRef() == "dsa.member_axis") {
        Value dom = mapped.lookup(def->getOperand(1));
        int64_t k = def->getAttrOfType<IntegerAttr>("axis").getInt();
        int64_t rank = cast<MemRefType>(dom.getType()).getRank();
        Value q = rowIdx;
        for (int64_t d = k + 1; d < rank - 1; ++d)
          q = b.create<arith::DivSIOp>(loc, q, viewDim(dom, d));
        q = b.create<arith::RemSIOp>(loc, q, viewDim(dom, k));
        coords.push_back(q);
        strides.push_back(st);
      } else if (res.getResultNumber() == 0) {
        coords.push_back(rowIdx);
        strides.push_back(st);
      } else {
        outColStride = st;
        coords.push_back(col);
        strides.push_back(st);
      }
    }
    return makeAddr(view, coords, strides, 0);
  }

  Value buildSlabPtr(Operation &op, unsigned first, Value view, Value rowIdx,
                     Value colPlusK) {
    int64_t owner = fn->getAttrOfType<IntegerAttr>("kir.owner_axis").getInt();
    SmallVector<Value> coords, strides;
    for (auto [pos, value] : llvm::enumerate(op.getOperands().drop_front(first))) {
      auto *coord = value.getDefiningOp();
      Value idx = rowIdx;
      if (coord->getName().getStringRef() == "dsa.slab_axis") {
        Value dom = mapped.lookup(coord->getOperand(1));
        int64_t axis = coord->getAttrOfType<IntegerAttr>("axis").getInt();
        int64_t rank = cast<MemRefType>(dom.getType()).getRank();
        idx = colPlusK;
        for (int64_t dim = axis + 1; dim < rank; ++dim)
          if (dim != owner)
            idx = b.create<arith::DivSIOp>(loc, idx, viewDim(dom, dim));
        idx = b.create<arith::RemSIOp>(loc, idx, viewDim(dom, axis));
      }
      coords.push_back(idx);
      strides.push_back(viewStride(view, pos));
    }
    return makeAddr(view, coords, strides, 0);
  }

  void emitSlabTransfer(Operation &op, unsigned first, Value view, Value local,
                        Value rowIdx, Value col, Value live, bool load) {
    auto loop = op.getAttrOfType<DictionaryAttr>("bangc.slab_loop");
    auto getIdx = [&](StringRef key) {
      return loop.getAs<IntegerAttr>(key).getInt();
    };
    int64_t slot = getIdx("extent_slot");
    Value extent = slot < 0 ? cI64(1)
                            : viewDim(mapped.lookup(fn.getArgument(slot)),
                                      getIdx("extent_axis"));
    Value innerStride = getIdx("inner_stride") < 0
                            ? cI64(0)
                            : viewStride(view, getIdx("inner_stride"));
    Value outerStride = getIdx("outer_pitch") < 0
                            ? cI64(0)
                            : viewStride(view, getIdx("outer_pitch"));
    int64_t elemBytes = llvm::divideCeil(
        cast<MemRefType>(view.getType()).getElementTypeBitWidth(), 8u);

    auto w = b.create<scf::WhileOp>(loc, TypeRange{b.getI64Type()},
                                    ValueRange{cI64(0)});
    {
      OpBuilder::InsertionGuard g(b);
      Block *before =
          b.createBlock(&w.getBefore(), {}, {b.getI64Type()}, {loc});
      b.setInsertionPointToStart(before);
      Value k = before->getArgument(0);
      Value cond =
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, k, live);
      b.create<scf::ConditionOp>(loc, cond, ValueRange{k});
    }
    {
      OpBuilder::InsertionGuard g(b);
      Block *after = b.createBlock(&w.getAfter(), {}, {b.getI64Type()}, {loc});
      b.setInsertionPointToStart(after);
      Value k = after->getArgument(0);
      Value colK = b.create<arith::AddIOp>(loc, col, k);
      Value rem = b.create<arith::RemSIOp>(loc, colK, extent);
      Value avail = b.create<arith::SubIOp>(loc, extent, rem);
      Value left = b.create<arith::SubIOp>(loc, live, k);
      Value run = b.create<arith::MinSIOp>(loc, avail, left);
      Value global = buildSlabPtr(op, first, view, rowIdx, colK);
      Value bufK = makeAddr(local, {k}, {cI64(1)}, 1);

      auto instructions = loop.getAs<ArrayAttr>("body");
      bool hasPitched = false;
      SmallVector<StringRef> fallbacks;
      for (auto attr : instructions) {
        auto s = cast<StringAttr>(attr).getValue();
        if (s == "pitched_full_runs")
          hasPitched = true;
        else
          fallbacks.push_back(s);
      }

      std::function<void(size_t)> emitFallbacks = [&](size_t idx) {
        StringRef opcode = fallbacks[idx];
        auto emitOne = [&]() {
          if (opcode == "contiguous") {
            if (load)
              memcpy1D(bufK, global, run, "GDRAM2NRAM", false);
            else
              memcpy1D(global, bufK, run, "NRAM2GDRAM", false);
          } else if (opcode == "broadcast") {
            Value sc = loadAt(global, cI64(0));
            fillVec(bufK, run, sc);
          } else {
            forLoop(cI64(0), run, cI64(1), [&](Value t) {
              Value gOff = b.create<arith::MulIOp>(loc, t, innerStride);
              if (load)
                storeAt(bufK, t, loadAt(global, gOff));
              else
                storeAt(global, gOff, loadAt(bufK, t));
            });
          }
        };
        if (idx + 1 >= fallbacks.size()) {
          emitOne();
          return;
        }
        Value cond = b.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, innerStride,
            cI64(opcode == "contiguous" ? 1 : 0));
        ifThenElse(cond, emitOne, [&]() { emitFallbacks(idx + 1); });
      };

      Value stepK = run;
      if (hasPitched) {
        Value c1 = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                           innerStride, cI64(1));
        Value c2 = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, run,
                                           extent);
        Value twoRun = b.create<arith::MulIOp>(loc, cI64(2), run);
        Value c3 = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, left,
                                           twoRun);
        Value c4 = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                           outerStride, run);
        Value c5 = b.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::sle, outerStride,
            cI64(2147483647LL / std::max<int64_t>(1, elemBytes)));
        Value canPitched = b.create<arith::AndIOp>(
            loc, c1,
            b.create<arith::AndIOp>(
                loc, c2,
                b.create<arith::AndIOp>(
                    loc, c3, b.create<arith::AndIOp>(loc, c4, c5))));
        auto pitchedIf = ifThenElse(
            canPitched,
            [&]() {
              Value cnt = b.create<arith::DivSIOp>(loc, left, run);
              Value seg = b.create<arith::SubIOp>(loc, cnt, cI64(1));
              if (load)
                memcpy2D(bufK, global, run, run, outerStride, seg, "GDRAM2NRAM",
                         false);
              else
                memcpy2D(global, bufK, run, outerStride, run, seg, "NRAM2GDRAM",
                         false);
              Value adv = b.create<arith::MulIOp>(loc, cnt, run);
              b.create<scf::YieldOp>(loc, ValueRange{adv});
            },
            [&]() {
              emitFallbacks(0);
              b.create<scf::YieldOp>(loc, ValueRange{run});
            },
            false);
        stepK = pitchedIf.getResult(0);
      } else {
        emitFallbacks(0);
      }
      Value nextK = b.create<arith::AddIOp>(loc, k, stepK);
      b.create<scf::YieldOp>(loc, ValueRange{nextK});
    }
  }

  std::pair<Value, Value> phaseCoords(Value iterPos, Value outerGroup) {
    if (resident)
      return {b.create<arith::MulIOp>(loc, iterPos, symR), cI64(0)};
    Value col = b.create<arith::MulIOp>(loc, iterPos, symW);
    if (independent) {
      Value nw1 = b.create<arith::SubIOp>(
          loc, b.create<arith::AddIOp>(loc, symN, symW), cI64(1));
      Value nc = b.create<arith::DivSIOp>(loc, nw1, symW);
      Value grpRow = b.create<arith::DivSIOp>(loc, outerGroup, nc);
      return {b.create<arith::MulIOp>(loc, grpRow, symR), col};
    }
    return {b.create<arith::MulIOp>(loc, outerGroup, symR), col};
  }

  void emitIssue(Block &fetch, Value iterPos, Value slot, Value outerGroup,
                 const llvm::DenseMap<Operation *, Value> &copyBuffers) {
    auto [row0, col] = phaseCoords(iterPos, outerGroup);
    Value remN = b.create<arith::SubIOp>(loc, symN, col);
    Value live = b.create<arith::MinSIOp>(loc, remN, symW);
    Value hasLive =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, live, cI64(0));
    ifThen(hasLive, [&]() {
      Value remM = b.create<arith::SubIOp>(loc, symM, row0);
      Value count = b.create<arith::MinSIOp>(
          loc, b.create<arith::MaxSIOp>(loc, remM, cI64(0)), symEB);
      for (auto &op : fetch.without_terminator()) {
        auto code = op.getName().getStringRef();
        if (code == "dsa.window" || code == "dsa.member_axis" ||
            code == "dsa.slab_axis" || code == "dsa.lookup_axis" ||
            op.hasAttr("dsa.shared"))
          continue;
        Value src = mapped.lookup(op.getOperand(0));
        Value dstBase = copyBuffers.lookup(&op);
        StringRef prim =
            op.getAttrOfType<StringAttr>("bangc.primitive").getValue();
        bool async = prim.ends_with("_async");
        int64_t elemBytes = llvm::divideCeil(
            cast<MemRefType>(src.getType()).getElementTypeBitWidth(), 8u);

        auto emitPerRowGather = [&]() {
          forLoop(cI64(0), count, cI64(1), [&](Value j) {
            Value rIdx = b.create<arith::AddIOp>(loc, row0, j);
            Value colStride;
            Value srcPtr = buildRowPtr(op, 2, src, rIdx, col, colStride);
            Value slotRow = b.create<arith::AddIOp>(
                loc, b.create<arith::MulIOp>(loc, slot, symEB), j);
            Value dstPtr = makeAddr(dstBase, {slotRow}, {symP}, 1);
            Value isContig = b.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::eq, colStride, cI64(1));
            ifThenElse(
                isContig,
                [&]() { memcpy1D(dstPtr, srcPtr, live, "GDRAM2NRAM", async); },
                [&]() {
                  Value isBcast = b.create<arith::CmpIOp>(
                      loc, arith::CmpIPredicate::eq, colStride, cI64(0));
                  ifThenElse(
                      isBcast,
                      [&]() {
                        Value sc = loadAt(srcPtr, cI64(0));
                        fillVec(dstPtr, live, sc);
                      },
                      [&]() {
                        forLoop(cI64(0), live, cI64(1), [&](Value k) {
                          Value off =
                              b.create<arith::MulIOp>(loc, k, colStride);
                          storeAt(dstPtr, k, loadAt(srcPtr, off));
                        });
                      });
                });
          });
        };

        if (prim == "indexed_dma") {
          auto *lookup = op.getOperand(2).getDefiningOp();
          Value keys = mapped.lookup(lookup->getOperand(0));
          Type keyElem = cast<MemRefType>(keys.getType()).getElementType();
          Value keysS0 = viewStride(keys, 0);
          Value srcS0 = viewStride(src, 0), srcS1 = viewStride(src, 1);
          Value srcD0 = viewDim(src, 0);
          Value c1 = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                             symEB, cI64(1));
          Value c2 = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                             keysS0, cI64(1));
          Value c3 = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                             srcS1, cI64(1));
          Value c4 = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                             srcD0, cI64(0));
          Value maxSpan =
              cI64(0xffffffffLL / std::max<int64_t>(1, elemBytes));
          Value prod = b.create<arith::MulIOp>(loc, srcS0, srcD0);
          Value c5 = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                             srcS0, cI64(0));
          Value c6 = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle,
                                             prod, maxSpan);
          Value fast = b.create<arith::AndIOp>(
              loc, c1,
              b.create<arith::AndIOp>(
                  loc, c2,
                  b.create<arith::AndIOp>(
                      loc, c3,
                      b.create<arith::AndIOp>(
                          loc, c4, b.create<arith::AndIOp>(loc, c5, c6)))));
          auto idxLoop = op.getAttrOfType<DictionaryAttr>("bangc.indexed_loop");
          int64_t perMember =
              idxLoop.getAs<IntegerAttr>("scratch_per_member").getInt();
          int64_t align =
              idxLoop.getAs<IntegerAttr>("scratch_alignment").getInt();
          ifThenElse(
              fast,
              [&]() {
                Value raw = b.create<arith::AddIOp>(
                    loc, b.create<arith::MulIOp>(loc, symEB, cI64(perMember)),
                    cI64(align - 1));
                Value aligned = b.create<arith::MulIOp>(
                    loc, b.create<arith::DivSIOp>(loc, raw, cI64(align)),
                    cI64(align));
                Value idsOff = b.create<arith::SubIOp>(loc, symBytes, aligned);
                Value idsBuf = nramAt(keyElem, idsOff);
                for (auto stepAttr : idxLoop.getAs<ArrayAttr>("batch")) {
                  StringRef step = cast<StringAttr>(stepAttr).getValue();
                  if (step == "stage_contiguous_indices") {
                    Value kSrc = makeAddr(keys, {row0}, {cI64(1)}, 0);
                    memcpy1D(idsBuf, kSrc, count, "GDRAM2NRAM", false);
                  } else if (step == "async_dma_per_member") {
                    forLoop(cI64(0), count, cI64(1), [&](Value j) {
                      Value idVal = toI64(loadAt(idsBuf, j));
                      Value sPtr =
                          makeAddr(src, {idVal, col}, {srcS0, cI64(1)}, 0);
                      Value slotRow = b.create<arith::AddIOp>(
                          loc, b.create<arith::MulIOp>(loc, slot, symEB), j);
                      Value dPtr = makeAddr(dstBase, {slotRow}, {symP}, 1);
                      memcpy1D(dPtr, sPtr, live, "GDRAM2NRAM", true);
                    });
                  } else {
                    sync("io");
                  }
                }
              },
              emitPerRowGather);
          continue;
        }

        if (prim == "segmented_slab_read") {
          forLoop(cI64(0), count, cI64(1), [&](Value j) {
            Value rIdx = b.create<arith::AddIOp>(loc, row0, j);
            Value slotRow = b.create<arith::AddIOp>(
                loc, b.create<arith::MulIOp>(loc, slot, symEB), j);
            Value dPtr = makeAddr(dstBase, {slotRow}, {symP}, 1);
            emitSlabTransfer(op, 2, src, dPtr, rIdx, col, live, true);
          });
          continue;
        }

        if (prim.starts_with("group_dma") && multiMember) {
          int64_t axis = cast<MemRefType>(src.getType()).getRank() - 2;
          Value dim = viewDim(src, axis), outer = viewStride(src, axis);
          Value colStride;
          Value srcBase = buildRowPtr(op, 2, src, row0, col, colStride);
          Value slotBase = b.create<arith::MulIOp>(loc, slot, symEB);
          Value dstSlot = makeAddr(dstBase, {slotBase}, {symP}, 1);
          Value c1 = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                             count, cI64(1));
          Value c2 = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                             colStride, cI64(1));
          Value c3 = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                             outer, live);
          Value c4 = b.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::sle, outer,
              cI64(2147483647LL / std::max<int64_t>(1, elemBytes)));
          Value wrap = b.create<arith::AddIOp>(
              loc, b.create<arith::RemSIOp>(loc, row0, dim), count);
          Value c5 = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle,
                                             wrap, dim);
          Value canGroup = b.create<arith::AndIOp>(
              loc, c1,
              b.create<arith::AndIOp>(
                  loc, c2,
                  b.create<arith::AndIOp>(
                      loc, c3, b.create<arith::AndIOp>(loc, c4, c5))));
          ifThenElse(
              canGroup,
              [&]() {
                Value flat = b.create<arith::AndIOp>(
                    loc,
                    b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                            outer, live),
                    b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, live,
                                            symP));
                ifThenElse(
                    flat,
                    [&]() {
                      Value total = b.create<arith::MulIOp>(loc, count, live);
                      memcpy1D(dstSlot, srcBase, total, "GDRAM2NRAM", async);
                    },
                    [&]() {
                      Value seg = b.create<arith::SubIOp>(loc, count, cI64(1));
                      memcpy2D(dstSlot, srcBase, live, symP, outer, seg,
                               "GDRAM2NRAM", async);
                    });
              },
              emitPerRowGather);
          continue;
        }

        emitPerRowGather();
      }
    });
  }

  LogicalResult emitCompute(Block &fetch, Block &body, Operation *phaseOp,
                            Value iterPos, Value slot, Value outerGroup,
                            Value carrySlot, bool carryIsMoment, bool packed,
                            const llvm::DenseMap<Operation *, Value> &copyBuffers) {
    auto [row0, col] = phaseCoords(iterPos, outerGroup);
    Value remN = b.create<arith::SubIOp>(loc, symN, col);
    Value live = b.create<arith::MinSIOp>(
        loc, b.create<arith::MaxSIOp>(loc, remN, cI64(0)), symW);
    Value remM = b.create<arith::SubIOp>(loc, symM, row0);
    Value count = b.create<arith::MinSIOp>(
        loc, b.create<arith::MaxSIOp>(loc, remM, cI64(0)), symEB);

    auto emitBodyMembers = [&]() -> LogicalResult {
      Value rowLanes = b.create<arith::MulIOp>(
          loc,
          b.create<arith::DivSIOp>(
              loc, b.create<arith::AddIOp>(loc, live, cI64(63)), cI64(64)),
          cI64(64));
      bool hasTransposeStore = false, needsFullPitch = false;
      for (auto &inst : body) {
        auto pAttr = inst.getAttrOfType<StringAttr>("bangc.primitive");
        StringRef pr = pAttr ? pAttr.getValue() : StringRef{};
        hasTransposeStore |= pr.starts_with("transpose_store");
        StringRef cn = inst.getName().stripDialect();
        needsFullPitch |= cn == "reduce" || cn == "exp" || cn == "div" ||
                          pr == "certified_half_quotient" ||
                          pr == "bf16_product" || pr == "bf16_round";
      }
      Value members = cI64(1);
      if (packed) {
        if (hasTransposeStore && !needsFullPitch) {
          members = count;
          rowLanes = symP;
        } else {
          Value fullPitch = b.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::eq, live, symP);
          members = b.create<arith::SelectOp>(loc, fullPitch, count, cI64(1));
        }
      }
      Value stepMembers = b.create<arith::MaxSIOp>(loc, members, cI64(1));

      llvm::DenseSet<Operation *> skipOps;
      llvm::DenseMap<Operation *, Operation *> fusedGtCast;
      for (auto &inst : body) {
        auto code = inst.getName().getStringRef();
        if (code == "block.linear_index" && inst.hasOneUse() &&
            inst.use_begin()->getOwner()->getName().getStringRef() ==
                "block.uniform" &&
            inst.use_begin()->getOwner()->getBlock() == &body)
          skipOps.insert(&inst);
        if (code == "block.gt" && inst.hasOneUse()) {
          auto *user = inst.use_begin()->getOwner();
          int r0 = rankOf(inst.getOperand(0).getType()),
              r1 = rankOf(inst.getOperand(1).getType());
          if (user->getBlock() == &body &&
              user->getName().getStringRef() == "block.cast" &&
              rankOf(inst.getResult(0).getType()) == 2 &&
              elemOf(inst.getOperand(0).getType()).isF32() &&
              elemOf(inst.getOperand(1).getType()).isF32() &&
              elemOf(user->getResult(0).getType()).isF32() &&
              ((r0 == 2 && (r1 == 2 || r1 == 0)) || (r0 == 0 && r1 == 2))) {
            skipOps.insert(&inst);
            fusedGtCast[user] = &inst;
          }
        }
      }

      LogicalResult bodyStatus = success();
      forLoop(cI64(0), count, stepMembers, [&](Value j) {
        Value lanes = b.create<arith::MulIOp>(loc, members, rowLanes);
        llvm::DenseMap<Value, Value> localMap;
        auto getVal = [&](Value v) -> Value {
          if (localMap.count(v))
            return localMap.lookup(v);
          return mapped.lookup(v);
        };

        unsigned argIdx = 0;
        for (auto &copy : fetch.without_terminator()) {
          auto cn = copy.getName().getStringRef();
          if (cn == "dsa.window" || cn == "dsa.member_axis" ||
              cn == "dsa.slab_axis" || cn == "dsa.lookup_axis")
            continue;
          Value baseBuf = copyBuffers.lookup(&copy);
          Value ptr = baseBuf;
          if (copy.hasAttr("dsa.shared")) {
            if (wide)
              ptr = makeAddr(baseBuf, {j}, {symP}, 1);
          } else {
            Value slotRow = b.create<arith::AddIOp>(
                loc, b.create<arith::MulIOp>(loc, slot, symEB), j);
            ptr = makeAddr(baseBuf, {slotRow}, {symP}, 1);
          }
          localMap[body.getArgument(argIdx++)] = ptr;
        }
        if (carrySlot) {
          Value curCarry = carryIsMoment
                               ? makeOp("moment_get", {carrySlot, j},
                                        {momentTuple(b.getContext())})
                                     ->getResult(0)
                               : carrySlot;
          localMap[body.getArgument(argIdx)] = curCarry;
        }

        for (auto &inst : body) {
          if (failed(bodyStatus))
            break;
          if (skipOps.contains(&inst))
            continue;
          StringRef n = inst.getName().stripDialect();
          if (n == "window" || n == "member_axis" || n == "slab_axis")
            continue;
          if (n == "await") {
            localMap[inst.getResult(0)] = getVal(inst.getOperand(0));
            continue;
          }
          StringRef prim =
              inst.getAttrOfType<StringAttr>("bangc.primitive").getValue();
          if (n == "owner_read") {
            Value view = getVal(inst.getOperand(0));
            Type resElem = elemOf(inst.getResult(0).getType());
            Value arr = localVar(resElem, "EB");
            forLoop(cI64(0), members, cI64(1), [&](Value h) {
              Value jh = b.create<arith::AddIOp>(loc, j, h);
              Value rIdx = b.create<arith::AddIOp>(loc, row0, jh);
              Value ptr = makeAddr(view, {rIdx}, {viewStride(view, 0)}, 0);
              storeAt(arr, jh, loadAt(ptr, cI64(0)));
            });
            localMap[inst.getResult(0)] = arr;
            continue;
          }
          if (n == "owner_write") {
            Value view = getVal(inst.getOperand(0));
            Value val = getVal(inst.getOperand(2));
            int vRank = rankOf(inst.getOperand(2).getType());
            forLoop(cI64(0), members, cI64(1), [&](Value h) {
              Value jh = b.create<arith::AddIOp>(loc, j, h);
              Value rIdx = b.create<arith::AddIOp>(loc, row0, jh);
              Value ptr = makeAddr(view, {rIdx}, {viewStride(view, 0)}, 0);
              storeAt(ptr, cI64(0), vRank == 1 ? loadAt(val, jh) : val);
            });
            continue;
          }
          if (prim == "unique_put") {
            Value base = getVal(inst.getOperand(0));
            Value indices = getVal(inst.getOperand(1));
            Value values = getVal(inst.getOperand(2));
            Value s0 = viewStride(base, 0);
            forLoop(cI64(0), live, cI64(1), [&](Value k) {
              Value idx = toI64(loadAt(indices, k));
              Value dst = makeAddr(base, {idx}, {s0}, 0);
              storeAt(dst, cI64(0), loadAt(values, k));
            });
            continue;
          }
          if (prim == "prefix_endpoints") {
            Value offsets = getVal(inst.getOperand(0));
            Value total = getVal(inst.getOperand(1));
            Value prefix = getVal(inst.getOperand(4));
            Value s0 = viewStride(offsets, 0);
            storeAt(offsets, cI64(0), cI32(0));
            Value contig = b.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::eq, s0, cI64(1));
            ifThenElse(
                contig,
                [&]() {
                  Value dst1 = makeAddr(offsets, {cI64(1)}, {cI64(1)}, 0);
                  memcpy1D(dst1, prefix, live, "NRAM2GDRAM", false);
                },
                [&]() {
                  forLoop(cI64(0), live, cI64(1), [&](Value k) {
                    Value kp1 = b.create<arith::AddIOp>(loc, k, cI64(1));
                    Value dst = makeAddr(offsets, {kp1}, {s0}, 0);
                    storeAt(dst, cI64(0), loadAt(prefix, k));
                  });
                });
            Value lastIdx = b.create<arith::SubIOp>(loc, live, cI64(1));
            storeAt(total, cI64(0), loadAt(prefix, lastIdx));
            continue;
          }
          if (prim == "mark_intervals") {
            Value out = getVal(inst.getOperand(0));
            Value offsets = getVal(inst.getOperand(1));
            int64_t unit = inst.getAttrOfType<IntegerAttr>("unit").getInt();
            Value outS0 = viewStride(out, 0), offS0 = viewStride(offsets, 0);
            Value maxK = b.create<arith::MinSIOp>(
                loc, live,
                b.create<arith::MaxSIOp>(
                    loc,
                    b.create<arith::SubIOp>(
                        loc, b.create<arith::SubIOp>(loc, symN, col), cI64(1)),
                    cI64(0)));
            forLoop(cI64(0), maxK, cI64(1), [&](Value k) {
              Value e = b.create<arith::AddIOp>(loc, col, k);
              Value ep1 = b.create<arith::AddIOp>(loc, e, cI64(1));
              Value sRaw =
                  toI64(loadAt(makeAddr(offsets, {e}, {offS0}, 0), cI64(0)));
              Value eRaw =
                  toI64(loadAt(makeAddr(offsets, {ep1}, {offS0}, 0), cI64(0)));
              Value st = b.create<arith::DivSIOp>(loc, sRaw, cI64(unit));
              Value en = b.create<arith::DivSIOp>(loc, eRaw, cI64(unit));
              Value eI32 = b.create<arith::TruncIOp>(loc, b.getI32Type(), e);
              forLoop(st, en, cI64(1), [&](Value i) {
                Value dst = makeAddr(out, {i}, {outS0}, 0);
                storeAt(dst, cI64(0), eI32);
              });
            });
            continue;
          }
          if (prim == "pack_scatter") {
            Value out = getVal(inst.getOperand(0));
            Value counts = getVal(inst.getOperand(1));
            Value mask = getVal(inst.getOperand(5));
            Value prefix = getVal(inst.getOperand(6));
            Value payload = getVal(inst.getOperand(7));
            int32_t fillVal = inst.getAttrOfType<IntegerAttr>("fill").getInt();
            Value pBytes = b.create<arith::MulIOp>(loc, symP, cI64(4));
            Value packedBuf = nramAt(
                b.getI32Type(), b.create<arith::SubIOp>(loc, symBytes, pBytes));
            Value hasElem = b.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::sgt, live, cI64(0));
            auto cntIf = ifThenElse(
                hasElem,
                [&]() {
                  Value last = b.create<arith::SubIOp>(loc, live, cI64(1));
                  b.create<scf::YieldOp>(loc,
                                         ValueRange{loadAt(prefix, last)});
                },
                [&]() { b.create<scf::YieldOp>(loc, ValueRange{cI32(0)}); },
                false);
            Value totalCnt = cntIf.getResult(0);
            Value totalCnt64 = toI64(totalCnt);
            forLoop(cI64(0), live, cI64(1), [&](Value k) {
              Value p = toI64(loadAt(prefix, k));
              Value keep = loadAt(mask, k);
              ifThenElse(
                  keep,
                  [&]() {
                    Value idx = b.create<arith::SubIOp>(loc, p, cI64(1));
                    storeAt(packedBuf, idx, loadAt(payload, k));
                  },
                  [&]() {
                    Value idx = b.create<arith::SubIOp>(
                        loc, b.create<arith::AddIOp>(loc, totalCnt64, k), p);
                    storeAt(packedBuf, idx, cI32(fillVal));
                  });
            });
            Value rIdx = b.create<arith::AddIOp>(loc, row0, j);
            ifThen(hasElem, [&]() {
              Value outS0 = viewStride(out, 0), outS1 = viewStride(out, 1);
              Value dstRow = makeAddr(out, {rIdx}, {outS0}, 0);
              Value contig = b.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::eq, outS1, cI64(1));
              ifThenElse(
                  contig,
                  [&]() {
                    memcpy1D(dstRow, packedBuf, live, "NRAM2GDRAM", false);
                  },
                  [&]() {
                    forLoop(cI64(0), live, cI64(1), [&](Value k) {
                      Value off = b.create<arith::MulIOp>(loc, k, outS1);
                      storeAt(dstRow, off, loadAt(packedBuf, k));
                    });
                  });
            });
            Value cntPtr =
                makeAddr(counts, {rIdx}, {viewStride(counts, 0)}, 0);
            storeAt(cntPtr, cI64(0), totalCnt);
            continue;
          }
          if (n == "complete") {
            if (inst.getNumOperands() > 1 && carrySlot) {
              Value res = getVal(inst.getOperand(1));
              if (carryIsMoment)
                makeOp("moment_set", {carrySlot, j, res});
              else {
                if (rankOf(inst.getOperand(1).getType()) == 1)
                  res = loadAt(res, j);
                storeAt(carrySlot, j, res);
              }
            }
            continue;
          }
          if (n == "drain") {
            if (llvm::any_of(inst.getOperands(), [](Value v) {
                  auto mode =
                      v.getDefiningOp()->getAttrOfType<BoolAttr>("async");
                  return mode && mode.getValue();
                }))
              sync("all");
            continue;
          }
          if (n == "write_back") {
            Value ptr = getVal(inst.getOperand(0));
            Value src = getVal(inst.getOperand(1));
            bool async = inst.getAttrOfType<BoolAttr>("async").getValue();
            int64_t elemBytes = llvm::divideCeil(
                cast<MemRefType>(ptr.getType()).getElementTypeBitWidth(), 8u);
            if (prim == "segmented_slab_write") {
              Value rIdx = b.create<arith::AddIOp>(loc, row0, j);
              emitSlabTransfer(inst, 3, ptr, src, rIdx, col, live, false);
              continue;
            }
            auto emitMemberStores = [&]() {
              forLoop(cI64(0), members, cI64(1), [&](Value h) {
                Value rIdx = b.create<arith::AddIOp>(
                    loc, row0, b.create<arith::AddIOp>(loc, j, h));
                Value colStride;
                Value outPtr =
                    buildRowPtr(inst, 3, ptr, rIdx, col, colStride);
                Value vPtr = makeAddr(src, {h}, {symP}, 1);
                Value contig = b.create<arith::CmpIOp>(
                    loc, arith::CmpIPredicate::eq, colStride, cI64(1));
                ifThenElse(
                    contig,
                    [&]() {
                      memcpy1D(outPtr, vPtr, live, "NRAM2GDRAM", async);
                    },
                    [&]() {
                      forLoop(cI64(0), live, cI64(1), [&](Value k) {
                        Value off =
                            b.create<arith::MulIOp>(loc, k, colStride);
                        storeAt(outPtr, off, loadAt(vPtr, k));
                      });
                    });
              });
            };
            if (packed && multiMember &&
                (prim == "transpose_store" ||
                 prim == "transpose_store_async")) {
              Type elemTy = cast<MemRefType>(ptr.getType()).getElementType();
              unsigned bits = elemTy.getIntOrFloatBitWidth();
              if (bits == 16 || bits == 32) {
                Value outS0 = viewStride(ptr, 0), outS1 = viewStride(ptr, 1);
                Value rIdx0 = b.create<arith::AddIOp>(loc, row0, j);
                Value outBase = makeAddr(ptr, {col, rIdx0}, {outS0, outS1}, 0);
                Value ebAligned = b.create<arith::CmpIOp>(
                    loc, arith::CmpIPredicate::eq,
                    b.create<arith::RemSIOp>(loc, symEB, cI64(32)), cI64(0));
                Value hasRows = b.create<arith::CmpIOp>(
                    loc, arith::CmpIPredicate::sgt, members, cI64(0));
                Value hasCols = b.create<arith::CmpIOp>(
                    loc, arith::CmpIPredicate::sgt, live, cI64(0));
                Value unitInner = b.create<arith::CmpIOp>(
                    loc, arith::CmpIPredicate::eq, outS1, cI64(1));
                Value validPitch = b.create<arith::AndIOp>(
                    loc,
                    b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                            outS0, members),
                    b.create<arith::CmpIOp>(
                        loc, arith::CmpIPredicate::sle, outS0,
                        cI64(2147483647LL / std::max<int64_t>(1, elemBytes))));
                Value canTrans = b.create<arith::AndIOp>(
                    loc, ebAligned,
                    b.create<arith::AndIOp>(
                        loc, hasRows,
                        b.create<arith::AndIOp>(
                            loc, hasCols,
                            b.create<arith::AndIOp>(loc, unitInner,
                                                    validPitch))));
                Value transScratch = nramAt(
                    elemTy,
                    phaseByteOffset(inst.getAttrOfType<DenseI64ArrayAttr>(
                        "bangc.scratch")));
                ifThenElse(
                    canTrans,
                    [&]() {
                      makeOp("tile_transpose",
                             {transScratch, src, symEB, symP}, {}, {},
                             "nram-transpose");
                      Value oneCol = b.create<arith::CmpIOp>(
                          loc, arith::CmpIPredicate::eq, live, cI64(1));
                      ifThenElse(
                          oneCol,
                          [&]() {
                            memcpy1D(outBase, transScratch, members,
                                     "NRAM2GDRAM", async);
                          },
                          [&]() {
                            Value seg =
                                b.create<arith::SubIOp>(loc, live, cI64(1));
                            memcpy2D(outBase, transScratch, members, outS0,
                                     symEB, seg, "NRAM2GDRAM", async);
                          });
                    },
                    emitMemberStores);
                continue;
              }
            }
            if (packed && multiMember &&
                (prim == "group_store" || prim == "group_store_async")) {
              int64_t axis = cast<MemRefType>(ptr.getType()).getRank() - 2;
              Value stride = viewStride(ptr, axis), dim = viewDim(ptr, axis);
              Value rIdx0 = b.create<arith::AddIOp>(loc, row0, j);
              Value colStride;
              Value outBase =
                  buildRowPtr(inst, 3, ptr, rIdx0, col, colStride);
              Value c1 = b.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::sgt, members, cI64(1));
              Value c2 = b.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::eq, colStride, cI64(1));
              Value c3 = b.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::sge, stride, live);
              Value c4 = b.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::sle, stride,
                  cI64(2147483647LL / std::max<int64_t>(1, elemBytes)));
              Value wrap = b.create<arith::AddIOp>(
                  loc, b.create<arith::RemSIOp>(loc, rIdx0, dim), members);
              Value c5 = b.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::sle, wrap, dim);
              Value canGroup = b.create<arith::AndIOp>(
                  loc, c1,
                  b.create<arith::AndIOp>(
                      loc, c2,
                      b.create<arith::AndIOp>(
                          loc, c3, b.create<arith::AndIOp>(loc, c4, c5))));
              ifThenElse(
                  canGroup,
                  [&]() {
                    Value flat = b.create<arith::AndIOp>(
                        loc,
                        b.create<arith::CmpIOp>(
                            loc, arith::CmpIPredicate::eq, stride, live),
                        b.create<arith::CmpIOp>(
                            loc, arith::CmpIPredicate::eq, live, symP));
                    ifThenElse(
                        flat,
                        [&]() {
                          Value total =
                              b.create<arith::MulIOp>(loc, members, live);
                          memcpy1D(outBase, src, total, "NRAM2GDRAM", async);
                        },
                        [&]() {
                          Value seg =
                              b.create<arith::SubIOp>(loc, members, cI64(1));
                          memcpy2D(outBase, src, live, stride, symP, seg,
                                   "NRAM2GDRAM", async);
                        });
                  },
                  emitMemberStores);
              continue;
            }
            emitMemberStores();
            continue;
          }

          Value resVal = inst.getResult(0);
          if (prim == "welford_f32") {
            Value scratch = nramAt(
                b.getF32Type(),
                phaseByteOffset(
                    inst.getAttrOfType<DenseI64ArrayAttr>("bangc.scratch")));
            Value m = makeOp("summary_reduce",
                             {getVal(inst.getOperand(0)), scratch, live},
                             {momentTuple(b.getContext())}, {},
                             "moments-reduce")
                          ->getResult(0);
            localMap[resVal] = m;
            continue;
          }
          if (prim == "summary_merge") {
            Value m = makeOp("summary_merge",
                             {getVal(inst.getOperand(0)),
                              getVal(inst.getOperand(1))},
                             {momentTuple(b.getContext())}, {}, "moments-merge")
                          ->getResult(0);
            localMap[resVal] = m;
            continue;
          }
          if (prim == "summary_field") {
            int64_t pos =
                inst.getAttrOfType<IntegerAttr>("position").getInt();
            if (pos < 0 || pos > 2) {
              bodyStatus = failure();
              break;
            }
            Type fieldTy = elemOf(resVal.getType());
            Value fVal =
                makeOp("summary_field", {getVal(inst.getOperand(0))},
                       {fieldTy},
                       {b.getNamedAttr("position", b.getI64IntegerAttr(pos))})
                    ->getResult(0);
            if (rankOf(resVal.getType()) == 1) {
              Value arr = localVar(fieldTy, "EB");
              storeAt(arr, j, fVal);
              localMap[resVal] = arr;
            } else {
              localMap[resVal] = fVal;
            }
            continue;
          }
          if (n == "reduce") {
            Value outArr = localVar(b.getF32Type(), "EB");
            Value outSlice = makeAddr(outArr, {j}, {cI64(1)}, 2);
            Value scratch = nramAt(
                b.getF32Type(),
                phaseByteOffset(
                    inst.getAttrOfType<DenseI64ArrayAttr>("bangc.offset")));
            StringRef alg = prim == "row_sum" ? "sum" : "max_num";
            makeOp("row_reduce",
                   {outSlice, scratch, getVal(inst.getOperand(0)), members,
                    symP, live},
                   {}, {b.getNamedAttr("algebra", b.getStringAttr(alg))},
                   "row-reduce");
            localMap[resVal] = outArr;
            continue;
          }

          int resRank = rankOf(resVal.getType());
          Type dstElem = elemOf(resVal.getType());
          if (resRank == 2) {
            Value dst = nramAt(
                dstElem,
                phaseByteOffset(
                    inst.getAttrOfType<DenseI64ArrayAttr>("bangc.offset")));
            localMap[resVal] = dst;

            if (prim == "quotient_for_narrow")
              continue;

            if (prim == "certified_half_quotient") {
              auto *divOp = inst.getOperand(0).getDefiningOp();
              Value qBuf = getVal(inst.getOperand(0));
              Value scratch = nramAt(
                  b.getF32Type(),
                  phaseByteOffset(
                      inst.getAttrOfType<DenseI64ArrayAttr>("bangc.scratch")));
              Value num = getVal(divOp->getOperand(0));
              Value divIn = getVal(divOp->getOperand(1));
              Value divSlice;
              if (rankOf(divOp->getOperand(1).getType()) == 1) {
                divSlice = makeAddr(divIn, {j}, {cI64(1)}, 2);
              } else {
                Value dArr = localVar(b.getF32Type(), "EB");
                forLoop(cI64(0), members, cI64(1), [&](Value h) {
                  storeAt(dArr, b.create<arith::AddIOp>(loc, j, h), divIn);
                });
                divSlice = makeAddr(dArr, {j}, {cI64(1)}, 2);
              }
              makeOp("narrow_quotient",
                     {dst, qBuf, scratch, num, divSlice, members, symP, live,
                      rowLanes},
                     {}, {}, "certified-half-quotient");
              continue;
            }

            if (fusedGtCast.count(&inst)) {
              Operation *gt = fusedGtCast.lookup(&inst);
              int r0 = rankOf(gt->getOperand(0).getType());
              int r1 = rankOf(gt->getOperand(1).getType());
              Value a = getVal(gt->getOperand(0)),
                    c = getVal(gt->getOperand(1));
              if (r0 == 2 && r1 == 2)
                makeOp("vec_binary", {dst, a, c, lanes}, {},
                       {b.getNamedAttr("op", b.getStringAttr("gt"))});
              else if (r0 == 2 && r1 == 0)
                makeOp("vec_scalar", {dst, a, c, lanes}, {},
                       {b.getNamedAttr("op", b.getStringAttr("gt"))});
              else
                makeOp("vec_scalar", {dst, c, a, lanes}, {},
                       {b.getNamedAttr("op", b.getStringAttr("lt"))});
              continue;
            }

            if (prim == "index_read_at") {
              Value base = getVal(inst.getOperand(0));
              Value keys = getVal(inst.getOperand(1));
              Value s0 = viewStride(base, 0);
              forLoop(cI64(0), live, cI64(1), [&](Value k) {
                Value idx = toI64(loadAt(keys, k));
                Value srcPtr = makeAddr(base, {idx}, {s0}, 0);
                storeAt(dst, k, loadAt(srcPtr, cI64(0)));
              });
              continue;
            }

            if (prim == "atomic_fetch_add") {
              Value base = getVal(inst.getOperand(0));
              Value keys = getVal(inst.getOperand(1));
              Value deltaVal = getVal(inst.getOperand(2));
              Value twoPBytes = b.create<arith::MulIOp>(loc, symP, cI64(8));
              Value deltaBuf = nramAt(
                  b.getI32Type(),
                  b.create<arith::SubIOp>(loc, symBytes, twoPBytes));
              Value oldBuf = makeAddr(deltaBuf, {symP}, {cI64(1)}, 1);
              storeAt(deltaBuf, cI64(0), deltaVal);
              Value s0 = viewStride(base, 0);
              forLoop(cI64(0), live, cI64(1), [&](Value k) {
                Value idx = toI64(loadAt(keys, k));
                Value gAddr = makeAddr(base, {idx}, {s0}, 0);
                makeOp("atomic_add", {oldBuf, gAddr, deltaBuf, cI64(1)});
                sync("all");
                storeAt(dst, k, loadAt(oldBuf, cI64(0)));
              });
              continue;
            }

            if (prim == "member_tile") {
              forLoop(cI64(0), live, cI64(1), [&](Value k) {
                Value v = b.create<arith::TruncIOp>(
                    loc, b.getI32Type(), b.create<arith::AddIOp>(loc, col, k));
                storeAt(dst, k, v);
              });
              continue;
            }

            if (prim == "private_bin_count" || prim == "vector_sum_columns") {
              Value ptr = getVal(inst.getOperand(0));
              Type inElem = cast<MemRefType>(ptr.getType()).getElementType();
              int64_t inBytes =
                  llvm::divideCeil(inElem.getIntOrFloatBitWidth(), 8u);
              int64_t extent =
                  inst.getAttrOfType<IntegerAttr>("extent").getInt();
              Value panelOff = b.create<arith::SubIOp>(
                  loc, symBytes,
                  b.create<arith::MulIOp>(loc, symP, cI64(inBytes)));
              Value panel = nramAt(inElem, panelOff);
              fillVec(dst, lanes, cI32(0));
              if (prim == "private_bin_count") {
                int64_t span =
                    inst.getAttrOfType<IntegerAttr>("span").getInt();
                Value rIdx = b.create<arith::AddIOp>(loc, row0, j);
                Value first = b.create<arith::MulIOp>(loc, rIdx, cI64(span));
                Value end = b.create<arith::MinSIOp>(
                    loc, b.create<arith::AddIOp>(loc, first, cI64(span)),
                    cI64(extent));
                Value s0 = viewStride(ptr, 0);
                forLoop(first, end, symP, [&](Value m) {
                  Value take = b.create<arith::MinSIOp>(
                      loc, b.create<arith::SubIOp>(loc, end, m), symP);
                  Value contig = b.create<arith::CmpIOp>(
                      loc, arith::CmpIPredicate::eq, s0, cI64(1));
                  ifThenElse(
                      contig,
                      [&]() {
                        Value sPtr = makeAddr(ptr, {m}, {cI64(1)}, 0);
                        memcpy1D(panel, sPtr, take, "GDRAM2NRAM", false);
                      },
                      [&]() {
                        forLoop(cI64(0), take, cI64(1), [&](Value z) {
                          Value idx = b.create<arith::AddIOp>(loc, m, z);
                          Value sPtr = makeAddr(ptr, {idx}, {s0}, 0);
                          storeAt(panel, z, loadAt(sPtr, cI64(0)));
                        });
                      });
                  forLoop(cI64(0), take, cI64(1), [&](Value z) {
                    Value bin = toI64(loadAt(panel, z));
                    Value ge0 = b.create<arith::CmpIOp>(
                        loc, arith::CmpIPredicate::sge, bin, cI64(0));
                    Value ltL = b.create<arith::CmpIOp>(
                        loc, arith::CmpIPredicate::slt, bin, live);
                    ifThen(b.create<arith::AndIOp>(loc, ge0, ltL), [&]() {
                      Value cur = loadAt(dst, bin);
                      storeAt(dst, bin,
                              b.create<arith::AddIOp>(loc, cur, cI32(1)));
                    });
                  });
                });
              } else {
                Value s0 = viewStride(ptr, 0), s1 = viewStride(ptr, 1);
                forLoop(cI64(0), cI64(extent), cI64(1), [&](Value p) {
                  Value contig = b.create<arith::CmpIOp>(
                      loc, arith::CmpIPredicate::eq, s1, cI64(1));
                  ifThenElse(
                      contig,
                      [&]() {
                        Value sPtr =
                            makeAddr(ptr, {p, col}, {s0, cI64(1)}, 0);
                        memcpy1D(panel, sPtr, live, "GDRAM2NRAM", false);
                      },
                      [&]() {
                        forLoop(cI64(0), live, cI64(1), [&](Value k) {
                          Value ck = b.create<arith::AddIOp>(loc, col, k);
                          Value sPtr = makeAddr(ptr, {p, ck}, {s0, s1}, 0);
                          storeAt(panel, k, loadAt(sPtr, cI64(0)));
                        });
                      });
                  makeOp("vec_binary", {dst, dst, panel, lanes}, {},
                         {b.getNamedAttr("op", b.getStringAttr("add"))});
                });
              }
              continue;
            }

            if (prim == "feature_panel_sum") {
              Value offsetView = getVal(inst.getOperand(0));
              auto &bb = inst.getRegion(0).front();
              Value vView = getVal(
                  bb.back().getOperand(0).getDefiningOp()->getOperand(0));
              Value offS0 = viewStride(offsetView, 0);
              Value rIdx = b.create<arith::AddIOp>(loc, row0, j);
              Value rIdx1 = b.create<arith::AddIOp>(loc, rIdx, cI64(1));
              Value a = toI64(
                  loadAt(makeAddr(offsetView, {rIdx}, {offS0}, 0), cI64(0)));
              Value lim = toI64(
                  loadAt(makeAddr(offsetView, {rIdx1}, {offS0}, 0), cI64(0)));
              int64_t pm =
                  inst.getAttrOfType<IntegerAttr>("bangc.panel_members")
                      .getInt();
              Value panelOff = b.create<arith::SubIOp>(
                  loc, symBytes,
                  b.create<arith::MulIOp>(loc, symP, cI64(pm * 4)));
              Value panel = nramAt(b.getF32Type(), panelOff);
              fillVec(dst, lanes, cF32(0.0f));
              Value vS0 = viewStride(vView, 0), vS1 = viewStride(vView, 1);
              forLoop(a, lim, cI64(pm), [&](Value i) {
                Value cnt = b.create<arith::MinSIOp>(
                    loc, b.create<arith::SubIOp>(loc, lim, i), cI64(pm));
                Value srcBase = makeAddr(vView, {i, col}, {vS0, vS1}, 0);
                Value fast = b.create<arith::AndIOp>(
                    loc,
                    b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, vS1,
                                            cI64(1)),
                    b.create<arith::AndIOp>(
                        loc,
                        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                live, symP),
                        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                vS0, symP)));
                ifThenElse(
                    fast,
                    [&]() {
                      Value tot = b.create<arith::MulIOp>(loc, cnt, symP);
                      memcpy1D(panel, srcBase, tot, "GDRAM2NRAM", false);
                    },
                    [&]() {
                      forLoop(cI64(0), cnt, cI64(1), [&](Value h) {
                        Value pSlice = makeAddr(panel, {h}, {symP}, 1);
                        Value sRow = makeAddr(srcBase, {h}, {vS0}, 0);
                        Value c1 = b.create<arith::CmpIOp>(
                            loc, arith::CmpIPredicate::eq, vS1, cI64(1));
                        ifThenElse(
                            c1,
                            [&]() {
                              memcpy1D(pSlice, sRow, live, "GDRAM2NRAM", false);
                            },
                            [&]() {
                              forLoop(cI64(0), live, cI64(1), [&](Value k) {
                                Value off =
                                    b.create<arith::MulIOp>(loc, k, vS1);
                                storeAt(pSlice, k, loadAt(sRow, off));
                              });
                            });
                      });
                    });
                forLoop(cI64(0), cnt, cI64(1), [&](Value h) {
                  Value pSlice = makeAddr(panel, {h}, {symP}, 1);
                  makeOp("vec_binary", {dst, dst, pSlice, lanes}, {},
                         {b.getNamedAttr("op", b.getStringAttr("add"))});
                });
              });
              continue;
            }

            if (prim == "ordered_members" || prim == "segment_size") {
              bool flat =
                  cast<MemRefType>(
                      fn.getArgument(
                            fn->getAttrOfType<IntegerAttr>("kir.domain_arg")
                                .getInt())
                          .getType())
                      .getRank() == 1;
              Value offsets = getVal(inst.getOperand(0));
              Value offS0 = viewStride(offsets, 0);
              forLoop(cI64(0), live, cI64(1), [&](Value k) {
                Value colK = b.create<arith::AddIOp>(loc, col, k);
                Value owner =
                    flat ? colK : b.create<arith::AddIOp>(loc, row0, j);
                Value owner1 = b.create<arith::AddIOp>(loc, owner, cI64(1));
                Value lo = toI64(loadAt(
                    makeAddr(offsets, {owner}, {offS0}, 0), cI64(0)));
                Value hi = toI64(loadAt(
                    makeAddr(offsets, {owner1}, {offS0}, 0), cI64(0)));
                if (prim == "segment_size") {
                  Value diff = b.create<arith::SubIOp>(loc, hi, lo);
                  storeAt(dst, k,
                          b.create<arith::SIToFPOp>(loc, b.getF32Type(), diff));
                } else {
                  auto accLoop = forLoopCarry(
                      lo, hi, cI64(1), ValueRange{cF32(0.0f)},
                      [&](Value member, ValueRange iter) {
                        auto &terms = inst.getRegion(0).front();
                        llvm::DenseMap<Value, Value> tMap;
                        tMap[terms.getArgument(0)] = member;
                        tMap[terms.getArgument(1)] = colK;
                        auto getT = [&](Value v) {
                          if (tMap.count(v))
                            return tMap.lookup(v);
                          return getVal(v);
                        };
                        Value termVal = cF32(0.0f);
                        for (auto &v : terms) {
                          StringRef tc = v.getName().stripDialect();
                          if (tc == "term") {
                            termVal = getT(v.getOperand(0));
                            break;
                          }
                          if (tc == "member_load") {
                            Value srcV = getT(v.getOperand(0));
                            SmallVector<Value> cs, ss;
                            for (unsigned ci = 1; ci < v.getNumOperands();
                                 ++ci) {
                              cs.push_back(toI64(getT(v.getOperand(ci))));
                              ss.push_back(viewStride(srcV, ci - 1));
                            }
                            Value p = makeAddr(srcV, cs, ss, 0);
                            tMap[v.getResult(0)] = loadAt(p, cI64(0));
                          } else if (tc == "mul") {
                            tMap[v.getResult(0)] =
                                makeOp("scalar_math",
                                       {getT(v.getOperand(0)),
                                        getT(v.getOperand(1))},
                                       {v.getResult(0).getType()},
                                       {b.getNamedAttr(
                                           "op",
                                           b.getStringAttr("volatile_mul"))})
                                    ->getResult(0);
                          } else if (tc == "add") {
                            tMap[v.getResult(0)] = b.create<arith::AddFOp>(
                                loc, getT(v.getOperand(0)),
                                getT(v.getOperand(1)));
                          } else {
                            bodyStatus = failure();
                            break;
                          }
                        }
                        Value nextAcc =
                            b.create<arith::AddFOp>(loc, iter[0], termVal);
                        b.create<scf::YieldOp>(loc, ValueRange{nextAcc});
                      });
                  storeAt(dst, k, accLoop.getResult(0));
                }
              });
              continue;
            }

            if (prim == "row_inclusive_sum") {
              Value scratch = nramAt(
                  dstElem,
                  phaseByteOffset(inst.getAttrOfType<DenseI64ArrayAttr>(
                      "bangc.scan_scratch")));
              makeOp("scan_sum",
                     {dst, getVal(inst.getOperand(0)), scratch, lanes, live},
                     {}, {}, "inclusive-scan");
              continue;
            }

            if (prim == "positive_half_or_nan") {
              makeOp("positive_half_or_nan",
                     {dst, getVal(inst.getOperand(0)), lanes});
              continue;
            }

            if ((prim == "bf16_round" || prim == "bf16_widen") &&
                rankOf(inst.getOperand(0).getType()) == 2) {
              StringRef mode = prim == "bf16_round" ? "narrow" : "widen";
              makeOp("bf16_tile",
                     {dst, getVal(inst.getOperand(0)), live, lanes}, {},
                     {b.getNamedAttr("mode", b.getStringAttr(mode))},
                     "bf16-tile-convert");
              continue;
            }

            if (n == "linear_index") {
              Type storeTy = storageElemType(b.getContext(), dstElem);
              forLoop(cI64(0), members, cI64(1), [&](Value h) {
                Value rIdx = b.create<arith::AddIOp>(
                    loc, row0, b.create<arith::AddIOp>(loc, j, h));
                Value rowBase = b.create<arith::AddIOp>(
                    loc, b.create<arith::MulIOp>(loc, rIdx, symN), col);
                Value dstH = makeAddr(dst, {h}, {symP}, 1);
                forLoop(cI64(0), live, cI64(1), [&](Value k) {
                  Value lin = b.create<arith::AddIOp>(loc, rowBase, k);
                  Value v = storeTy.isInteger(32)
                                ? b.create<arith::TruncIOp>(loc, storeTy, lin)
                                      .getResult()
                                : lin;
                  storeAt(dstH, k, v);
                });
              });
              continue;
            }

            if (prim == "philox4x32_uniform") {
              Value seed = getVal(inst.getOperand(0));
              auto *ctrOp = inst.getOperand(1).getDefiningOp();
              if (ctrOp &&
                  ctrOp->getName().getStringRef() == "block.linear_index") {
                int64_t rScratch =
                    profile.services.get<intentdsl::dsa::LocalSupply>()
                        ->randomScratchBytes;
                int64_t alignedR = llvm::alignTo(rScratch, 128);
                Value scOff =
                    b.create<arith::SubIOp>(loc, symBytes, cI64(alignedR));
                Value scPtr = nramAt(b.getI32Type(), scOff);
                forLoop(cI64(0), members, cI64(1), [&](Value h) {
                  Value rIdx = b.create<arith::AddIOp>(
                      loc, row0, b.create<arith::AddIOp>(loc, j, h));
                  Value base = b.create<arith::AddIOp>(
                      loc, b.create<arith::MulIOp>(loc, rIdx, symN), col);
                  Value dstH = makeAddr(dst, {h}, {symP}, 1);
                  makeOp("philox_uniform", {dstH, scPtr, seed, base, live}, {},
                         {}, "philox4x32-linear");
                });
              } else {
                Value ctrs = getVal(inst.getOperand(1));
                forLoop(cI64(0), members, cI64(1), [&](Value h) {
                  Value dstH = makeAddr(dst, {h}, {symP}, 1);
                  Value ctrH = makeAddr(ctrs, {h}, {symP}, 1);
                  forLoop(cI64(0), live, cI64(1), [&](Value k) {
                    Value cVal = loadAt(ctrH, k);
                    Value u = makeOp("philox_at", {seed, cVal},
                                     {b.getF32Type()}, {}, "philox4x32-element")
                                  ->getResult(0);
                    storeAt(dstH, k, u);
                  });
                });
              }
              continue;
            }

            Type in0Elem = inst.getNumOperands() >= 1
                               ? elemOf(inst.getOperand(0).getType())
                               : Type{};
            if (n == "cast" && rankOf(inst.getOperand(0).getType()) == 2 &&
                ((dstElem.isF16() && in0Elem.isF32()) ||
                 (dstElem.isF32() && in0Elem.isF16()))) {
              StringRef kind = dstElem.isF16() ? "f32_to_f16" : "f16_to_f32";
              makeOp("vec_cast", {dst, getVal(inst.getOperand(0)), lanes}, {},
                     {b.getNamedAttr("kind", b.getStringAttr(kind))});
              continue;
            }
            if (n == "exp" && dstElem.isF32() &&
                rankOf(inst.getOperand(0).getType()) == 2) {
              Value scratch = nramAt(
                  b.getF32Type(),
                  phaseByteOffset(
                      inst.getAttrOfType<DenseI64ArrayAttr>("bangc.scratch")));
              Value oneRow = b.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::eq, members, cI64(1));
              Value activeElems = b.create<arith::SelectOp>(
                  loc, oneRow, live,
                  b.create<arith::MulIOp>(loc, members, symP));
              makeOp("row_exp",
                     {dst, scratch, getVal(inst.getOperand(0)), activeElems,
                      lanes},
                     {}, {}, "validated-exp");
              continue;
            }
            if (n == "sqrt" && dstElem.isF32() &&
                rankOf(inst.getOperand(0).getType()) == 2) {
              makeOp("vec_sqrt_rn", {dst, getVal(inst.getOperand(0)), lanes},
                     {}, {}, "ieee-sqrt-rn");
              continue;
            }
            if (prim == "bf16_product" &&
                rankOf(inst.getOperand(0).getType()) == 2 &&
                rankOf(inst.getOperand(1).getType()) == 2) {
              Value scratch = nramAt(
                  b.getF32Type(),
                  phaseByteOffset(
                      inst.getAttrOfType<DenseI64ArrayAttr>("bangc.scratch")));
              Value oneRow = b.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::eq, members, cI64(1));
              Value activeElems = b.create<arith::SelectOp>(
                  loc, oneRow, live,
                  b.create<arith::MulIOp>(loc, members, symP));
              makeOp("bf16_product",
                     {dst, getVal(inst.getOperand(0)),
                      getVal(inst.getOperand(1)), scratch, activeElems, lanes},
                     {}, {}, "bf16-tile-product");
              continue;
            }
            if (n == "div" && prim != "quotient_for_narrow" &&
                dstElem.isF32()) {
              int r0 = rankOf(inst.getOperand(0).getType());
              int r1 = rankOf(inst.getOperand(1).getType());
              Value scratch = nramAt(
                  b.getF32Type(),
                  phaseByteOffset(
                      inst.getAttrOfType<DenseI64ArrayAttr>("bangc.scratch")));
              Value oneRow = b.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::eq, members, cI64(1));
              Value activeElems = b.create<arith::SelectOp>(
                  loc, oneRow, live,
                  b.create<arith::MulIOp>(loc, members, symP));
              if ((r0 == 2 && r1 == 2) || (r0 == 2 && r1 == 0) ||
                  (r0 == 0 && r1 == 2)) {
                StringRef mode = (r0 == 2 && r1 == 2)   ? "vv"
                                 : (r0 == 2 && r1 == 0) ? "vs"
                                                        : "sv";
                makeOp("vec_div_rn",
                       {dst, getVal(inst.getOperand(0)),
                        getVal(inst.getOperand(1)), scratch, activeElems,
                        lanes},
                       {}, {b.getNamedAttr("mode", b.getStringAttr(mode))},
                       "ieee-div-rn");
                continue;
              }
              if (r0 == 2 && r1 == 1) {
                Value lhsIn = getVal(inst.getOperand(0));
                Value rhsIn = getVal(inst.getOperand(1));
                forLoop(cI64(0), members, cI64(1), [&](Value h) {
                  Value dH = makeAddr(dst, {h}, {symP}, 1);
                  Value sH = makeAddr(lhsIn, {h}, {symP}, 1);
                  Value scVal =
                      loadAt(rhsIn, b.create<arith::AddIOp>(loc, j, h));
                  makeOp("vec_div_rn",
                         {dH, sH, scVal, scratch, live, rowLanes}, {},
                         {b.getNamedAttr("mode", b.getStringAttr("vs"))},
                         "ieee-div-rn");
                });
                continue;
              }
            }
            if ((n == "add" || n == "sub" || n == "mul") &&
                prim != "bf16_product" &&
                (dstElem.isF32() || dstElem.isF16() ||
                 dstElem.isInteger(32))) {
              int r0 = rankOf(inst.getOperand(0).getType());
              int r1 = rankOf(inst.getOperand(1).getType());
              if (r0 == 2 && r1 == 2) {
                makeOp("vec_binary",
                       {dst, getVal(inst.getOperand(0)),
                        getVal(inst.getOperand(1)), lanes},
                       {}, {b.getNamedAttr("op", b.getStringAttr(n))});
                continue;
              }
              if (r0 == 2 && r1 == 0) {
                makeOp("vec_scalar",
                       {dst, getVal(inst.getOperand(0)),
                        getVal(inst.getOperand(1)), lanes},
                       {}, {b.getNamedAttr("op", b.getStringAttr(n))});
                continue;
              }
              if (r0 == 0 && r1 == 2 && (n == "add" || n == "mul")) {
                makeOp("vec_scalar",
                       {dst, getVal(inst.getOperand(1)),
                        getVal(inst.getOperand(0)), lanes},
                       {}, {b.getNamedAttr("op", b.getStringAttr(n))});
                continue;
              }
              if (r0 == 0 && r1 == 2 && n == "sub") {
                Value negOne =
                    dstElem.isF32()
                        ? cF32(-1.0f)
                        : dstElem.isF16()
                              ? scalarCast(cF32(-1.0f), dstElem)
                              : cI32(-1);
                makeOp("vec_scalar",
                       {dst, getVal(inst.getOperand(1)), negOne, lanes}, {},
                       {b.getNamedAttr("op", b.getStringAttr("mul"))});
                makeOp("vec_scalar",
                       {dst, dst, getVal(inst.getOperand(0)), lanes}, {},
                       {b.getNamedAttr("op", b.getStringAttr("add"))});
                continue;
              }
              if ((r0 == 2 && r1 == 1) ||
                  (r0 == 1 && r1 == 2 && (n == "add" || n == "mul"))) {
                Value vecIn = getVal(inst.getOperand(r0 == 2 ? 0 : 1));
                Value rowIn = getVal(inst.getOperand(r0 == 2 ? 1 : 0));
                forLoop(cI64(0), members, cI64(1), [&](Value h) {
                  Value dH = makeAddr(dst, {h}, {symP}, 1);
                  Value sH = makeAddr(vecIn, {h}, {symP}, 1);
                  Value sc =
                      loadAt(rowIn, b.create<arith::AddIOp>(loc, j, h));
                  makeOp("vec_scalar", {dH, sH, sc, rowLanes}, {},
                         {b.getNamedAttr("op", b.getStringAttr(n))});
                });
                continue;
              }
            }

            forLoop(cI64(0), members, cI64(1), [&](Value h) {
              Value dH = makeAddr(dst, {h}, {symP}, 1);
              Value jh = b.create<arith::AddIOp>(loc, j, h);
              forLoop(cI64(0), live, cI64(1), [&](Value k) {
                SmallVector<Value> scOps;
                for (Value opnd : inst.getOperands()) {
                  int rk = rankOf(opnd.getType());
                  Value mv = getVal(opnd);
                  if (rk == 2) {
                    Value sH = makeAddr(mv, {h}, {symP}, 1);
                    scOps.push_back(loadAt(sH, k));
                  } else if (rk == 1) {
                    scOps.push_back(loadAt(mv, jh));
                  } else {
                    scOps.push_back(mv);
                  }
                }
                Value outSc;
                if (failed(evalScalarExpr(
                        n, prim, scOps, dstElem,
                        inst.getAttrOfType<TypedAttr>("value"), outSc))) {
                  bodyStatus = failure();
                  return;
                }
                storeAt(dH, k, outSc);
              });
            });
            continue;
          }

          if (resRank == 1) {
            Value arr = localVar(dstElem, "EB");
            forLoop(cI64(0), members, cI64(1), [&](Value h) {
              Value jh = b.create<arith::AddIOp>(loc, j, h);
              SmallVector<Value> scOps;
              for (Value opnd : inst.getOperands()) {
                Value mv = getVal(opnd);
                scOps.push_back(rankOf(opnd.getType()) == 1 ? loadAt(mv, jh)
                                                            : mv);
              }
              Value outSc;
              if (failed(evalScalarExpr(
                      n, prim, scOps, dstElem,
                      inst.getAttrOfType<TypedAttr>("value"), outSc))) {
                bodyStatus = failure();
                return;
              }
              storeAt(arr, jh, outSc);
            });
            localMap[resVal] = arr;
            continue;
          }

          SmallVector<Value> scOps;
          for (Value opnd : inst.getOperands())
            scOps.push_back(getVal(opnd));
          Value outSc;
          if (failed(evalScalarExpr(n, prim, scOps, dstElem,
                                    inst.getAttrOfType<TypedAttr>("value"),
                                    outSc))) {
            bodyStatus = failure();
            break;
          }
          localMap[resVal] = outSc;
        }
      });
      return bodyStatus;
    };

    if (auto emptyCount =
            phaseOp->getAttrOfType<IntegerAttr>("bangc.empty_count_slot")) {
      Value dest = mapped.lookup(fn.getArgument(emptyCount.getInt()));
      Value destS0 = viewStride(dest, 0);
      Value isEmpty =
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, live, cI64(0));
      LogicalResult status = success();
      ifThenElse(
          isEmpty,
          [&]() {
            forLoop(cI64(0), count, cI64(1), [&](Value j) {
              Value rIdx = b.create<arith::AddIOp>(loc, row0, j);
              Value dPtr = makeAddr(dest, {rIdx}, {destS0}, 0);
              storeAt(dPtr, cI64(0), cI32(0));
            });
          },
          [&]() { status = emitBodyMembers(); });
      return status;
    }
    return emitBodyMembers();
  }

  LogicalResult emitPhase(Operation &phaseOp, Value outerGroup) {
    auto &fetch = phaseOp.getRegion(0).front();
    auto &body = phaseOp.getRegion(1).front();
    auto &sched = phaseOp.getRegion(2).front();
    Value carrySlot;
    bool carryIsMoment = false;
    if (phaseOp.getNumResults()) {
      carrySlot = mapped.lookup(phaseOp.getOperand(0));
      carryIsMoment = isa<TupleType>(phaseOp.getResult(0).getType());
      mapped[phaseOp.getResult(0)] = carrySlot;
    }

    llvm::DenseMap<Operation *, Value> copyBuffers;
    for (auto &copy : fetch.without_terminator()) {
      auto cn = copy.getName().getStringRef();
      if (cn == "dsa.window" || cn == "dsa.member_axis" ||
          cn == "dsa.slab_axis" || cn == "dsa.lookup_axis")
        continue;
      Type elem = elemOf(copy.getResult(0).getType());
      Value buf = nramAt(elem, phaseByteOffset(
                                   copy.getAttrOfType<DenseI64ArrayAttr>(
                                       "bangc.offset")));
      copyBuffers[&copy] = buf;
    }

    llvm::DenseMap<Value, Value> sMap;
    auto emitSteps = [&](auto &self, Block &steps) -> LogicalResult {
      for (auto &op : steps) {
        StringRef n = op.getName().stripDialect();
        if (n == "range") {
          StringRef src = op.getAttrOfType<StringAttr>("source").getValue();
          if (src == "owned_tile") {
            Value nw1 = b.create<arith::SubIOp>(
                loc, b.create<arith::AddIOp>(loc, symN, symW), cI64(1));
            Value nc = b.create<arith::DivSIOp>(loc, nw1, symW);
            Value lo = b.create<arith::RemSIOp>(loc, outerGroup, nc);
            sMap[op.getResult(0)] = lo;
            sMap[op.getResult(1)] = b.create<arith::AddIOp>(loc, lo, cI64(1));
            sMap[op.getResult(2)] = cI64(1);
          } else if (src == "participant_stride") {
            Value start =
                batch > 1
                    ? b.create<arith::MulIOp>(loc, symTaskId, symB).getResult()
                    : symTaskId;
            Value step =
                batch > 1
                    ? b.create<arith::MulIOp>(loc, symTaskDim, symB).getResult()
                    : symTaskDim;
            sMap[op.getResult(0)] = start;
            sMap[op.getResult(1)] = symGrid;
            sMap[op.getResult(2)] = step;
          } else {
            Value nw1 = b.create<arith::SubIOp>(
                loc, b.create<arith::AddIOp>(loc, symN, symW), cI64(1));
            sMap[op.getResult(0)] = cI64(0);
            sMap[op.getResult(1)] = b.create<arith::DivSIOp>(loc, nw1, symW);
            sMap[op.getResult(2)] = cI64(1);
          }
        } else if (n == "load_shared") {
          Value lo = sMap.lookup(op.getOperand(0)),
                hi = sMap.lookup(op.getOperand(1));
          Value cond =
              b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, lo, hi);
          ifThen(cond, [&]() {
            for (auto &copy : fetch.without_terminator()) {
              if (!copy.hasAttr("dsa.shared"))
                continue;
              Value src = mapped.lookup(copy.getOperand(0));
              Value dst = copyBuffers.lookup(&copy);
              Value s0 = viewStride(src, 0);
              Value contig = b.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::eq, s0, cI64(1));
              ifThenElse(
                  contig,
                  [&]() { memcpy1D(dst, src, symN, "GDRAM2NRAM", false); },
                  [&]() {
                    forLoop(cI64(0), symN, cI64(1), [&](Value k) {
                      Value sPtr = makeAddr(src, {k}, {s0}, 0);
                      storeAt(dst, k, loadAt(sPtr, cI64(0)));
                    });
                  });
              if (wide) {
                forLoop(cI64(1), symEB, cI64(1), [&](Value h) {
                  Value dstH = makeAddr(dst, {h}, {symP}, 1);
                  memcpy1D(dstH, dst, symP, "NRAM2NRAM", false);
                });
              }
            }
          });
        } else if (n == "slot_zero") {
          sMap[op.getResult(0)] = cI64(0);
        } else if (n == "issue_if_in_bounds") {
          Value pos = sMap.lookup(op.getOperand(0)),
                end = sMap.lookup(op.getOperand(1)),
                sl = sMap.lookup(op.getOperand(2));
          Value cond =
              b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, pos, end);
          ifThen(cond, [&]() {
            emitIssue(fetch, pos, sl, outerGroup, copyBuffers);
          });
        } else if (n == "iterate") {
          Value lo = sMap.lookup(op.getOperand(0)),
                hi = sMap.lookup(op.getOperand(1)),
                st = sMap.lookup(op.getOperand(2)),
                initSlot = sMap.lookup(op.getOperand(3));
          auto &bb = op.getRegion(0).front();
          LogicalResult innerStatus = success();
          forLoopCarry(lo, hi, st, ValueRange{initSlot},
                       [&](Value posVal, ValueRange iter) {
                         sMap[bb.getArgument(0)] = posVal;
                         sMap[bb.getArgument(1)] = iter[0];
                         innerStatus = self(self, bb);
                       });
          if (failed(innerStatus))
            return failure();
        } else if (n == "wait_slot") {
          sync("all");
        } else if (n == "next") {
          sMap[op.getResult(0)] =
              b.create<arith::AddIOp>(loc, sMap.lookup(op.getOperand(0)),
                                      sMap.lookup(op.getOperand(1)));
        } else if (n == "other_slot") {
          sMap[op.getResult(0)] = b.create<arith::SubIOp>(
              loc, cI64(1), sMap.lookup(op.getOperand(0)));
        } else if (n == "compute_members") {
          bool isPacked = op.getAttrOfType<StringAttr>("bangc.primitive") ==
                          "packed_pointwise";
          if (failed(emitCompute(fetch, body, &phaseOp,
                                 sMap.lookup(op.getOperand(0)),
                                 sMap.lookup(op.getOperand(1)), outerGroup,
                                 carrySlot, carryIsMoment, isPacked,
                                 copyBuffers)))
            return failure();
        } else if (n == "advance") {
          b.create<scf::YieldOp>(loc,
                                 ValueRange{sMap.lookup(op.getOperand(0))});
        } else if (n != "end") {
          return failure();
        }
      }
      return success();
    };
    return emitSteps(emitSteps, sched);
  }

  LogicalResult emitOuterOps(ArrayRef<Operation *> ops, Value outerGroup) {
    llvm::DenseMap<Value, Value> rangeMap;
    for (Operation *opPtr : ops) {
      Operation &op = *opPtr;
      StringRef code = op.getName().stripDialect();
      if (code == "range") {
        rangeMap[op.getResult(0)] = symTaskId;
        rangeMap[op.getResult(1)] = symGrid;
        rangeMap[op.getResult(2)] = symTaskDim;
      } else if (code == "task_groups") {
        Value lo = rangeMap.lookup(op.getOperand(0)),
              hi = rangeMap.lookup(op.getOperand(1)),
              st = rangeMap.lookup(op.getOperand(2));
        SmallVector<Operation *> innerOps;
        for (auto &innerOp : op.getRegion(0).front())
          innerOps.push_back(&innerOp);
        LogicalResult inner = success();
        forLoop(lo, hi, st,
                [&](Value grp) { inner = emitOuterOps(innerOps, grp); });
        if (failed(inner))
          return failure();
      } else if (code == "phase") {
        if (failed(emitPhase(op, outerGroup)))
          return failure();
      } else if (code == "owner_rows") {
        mapped[op.getResult(0)] = outerGroup;
      } else if (code == "owner_read" || code == "owner_write") {
        Value base = mapped.lookup(op.getOperand(0));
        Value s0 = viewStride(base, 0);
        Value gRow0 = b.create<arith::MulIOp>(loc, outerGroup, symR);
        Value cnt = b.create<arith::MinSIOp>(
            loc,
            b.create<arith::MaxSIOp>(
                loc, b.create<arith::SubIOp>(loc, symM, gRow0), cI64(0)),
            symR);
        if (code == "owner_write") {
          Value val = mapped.lookup(op.getOperand(2));
          bool isArr = isa<RankedTensorType>(op.getOperand(2).getType());
          forLoop(cI64(0), cnt, cI64(1), [&](Value j) {
            Value rIdx = b.create<arith::AddIOp>(loc, gRow0, j);
            Value dst = makeAddr(base, {rIdx}, {s0}, 0);
            storeAt(dst, cI64(0), isArr ? loadAt(val, j) : val);
          });
        } else {
          Type resElem = elemOf(op.getResult(0).getType());
          Value arr = localVar(resElem, "R");
          forLoop(cI64(0), cnt, cI64(1), [&](Value j) {
            Value rIdx = b.create<arith::AddIOp>(loc, gRow0, j);
            Value src = makeAddr(base, {rIdx}, {s0}, 0);
            storeAt(arr, j, loadAt(src, cI64(0)));
          });
          mapped[op.getResult(0)] = arr;
        }
      } else if (code == "drain") {
        continue;
      } else if (code == "summary_identity") {
        mapped[op.getResult(0)] =
            makeOp("moment_array", {}, {momentArrayTuple(b.getContext())},
                   {b.getNamedAttr("extent", b.getStringAttr("R"))})
                ->getResult(0);
      } else if (code == "identity") {
        Value arr = localVar(b.getF32Type(), "R");
        bool sum = op.getAttrOfType<StringAttr>("algebra") == "sum";
        Value init = sum ? cF32(0.0f)
                         : cF32(-std::numeric_limits<float>::infinity());
        forLoop(cI64(0), symR, cI64(1),
                [&](Value j) { storeAt(arr, j, init); });
        mapped[op.getResult(0)] = arr;
      } else if (code == "cast" || code == "summary_field" || code == "div" ||
                 code == "mul" || code == "add" || code == "sub" ||
                 code == "rsqrt" || code == "constant") {
        Type resTy = op.getResult(0).getType();
        Type resElem = elemOf(resTy);
        bool perOwner = isa<RankedTensorType>(resTy);
        if (perOwner) {
          Value arr = localVar(resElem, "R");
          Value gRow0 = b.create<arith::MulIOp>(loc, outerGroup, symR);
          Value cnt = b.create<arith::MinSIOp>(
              loc,
              b.create<arith::MaxSIOp>(
                  loc, b.create<arith::SubIOp>(loc, symM, gRow0), cI64(0)),
              symR);
          LogicalResult st = success();
          forLoop(cI64(0), cnt, cI64(1), [&](Value j) {
            if (code == "summary_field") {
              int64_t pos =
                  op.getAttrOfType<IntegerAttr>("position").getInt();
              if (pos < 0 || pos > 2) {
                st = failure();
                return;
              }
              Value mArr = mapped.lookup(op.getOperand(0));
              Value m = makeOp("moment_get", {mArr, j},
                               {momentTuple(b.getContext())})
                            ->getResult(0);
              Value fVal =
                  makeOp("summary_field", {m}, {resElem},
                         {b.getNamedAttr("position", b.getI64IntegerAttr(pos))})
                      ->getResult(0);
              storeAt(arr, j, fVal);
            } else {
              SmallVector<Value> scOps;
              for (Value opnd : op.getOperands()) {
                Value mv = mapped.lookup(opnd);
                scOps.push_back(isa<RankedTensorType>(opnd.getType())
                                    ? loadAt(mv, j)
                                    : mv);
              }
              Value outSc;
              if (failed(evalScalarExpr(
                      code, "", scOps, resElem,
                      op.getAttrOfType<TypedAttr>("value"), outSc))) {
                st = failure();
                return;
              }
              storeAt(arr, j, outSc);
            }
          });
          if (failed(st))
            return failure();
          mapped[op.getResult(0)] = arr;
        } else {
          SmallVector<Value> scOps;
          for (Value opnd : op.getOperands())
            scOps.push_back(mapped.lookup(opnd));
          Value outSc;
          if (failed(evalScalarExpr(code, "", scOps, resElem,
                                    op.getAttrOfType<TypedAttr>("value"),
                                    outSc)))
            return failure();
          mapped[op.getResult(0)] = outSc;
        }
      } else if (code != "return" && code != "end") {
        return failure();
      }
    }
    return success();
  }

  LogicalResult run() {
    for (Value arg : fn.getArguments())
      mapped[arg] = arg;

    SmallVector<Operation *> oldOps;
    for (auto &op : fn.front())
      if (op.getName().getStringRef() != "func.return")
        oldOps.push_back(&op);

    b.setInsertionPointToStart(&fn.front());
    OperationState progState(loc, "bangc_src.row_program");
    progState.addAttribute("batch", b.getI64IntegerAttr(batch));
    progState.addAttribute("depth", b.getI64IntegerAttr(depth));
    progState.addRegion();
    progState.addRegion();
    progState.addRegion();
    Operation *prog = b.create(progState);
    Block *sourcesBlk = b.createBlock(&prog->getRegion(0));
    Block *deviceBlk = b.createBlock(&prog->getRegion(1));
    Block *hostBlk = b.createBlock(&prog->getRegion(2));

    b.setInsertionPointToStart(deviceBlk);
    OperationState dkState(loc, "bangc_src.device_kernel");
    dkState.addOperands(fn.getArguments());
    dkState.addRegion();
    Operation *dk = b.create(dkState);
    Block *kBody = b.createBlock(&dk->getRegion(0));
    b.setInsertionPointToStart(kBody);

    storage =
        makeOp("nram_alloc", {}, {ptrType(b.getContext(), b.getI8Type(), 1)})
            ->getResult(0);
    symM = makeSym("M");
    symN = makeSym("N");
    symGrid = makeSym("grid");
    symTaskId = makeSym("taskId");
    symTaskDim = makeSym("taskDim");
    symR = makeSym("R");
    symB = makeSym("B");
    symEB = makeSym("EB");
    symW = makeSym("W");
    symP = makeSym("P");
    symBytes = makeSym("BYTES");
    constAnchor = symBytes.getDefiningOp();

    if (failed(emitOuterOps(oldOps, cI64(0))))
      return fn.emitError("failed to lower row program to bangc_src");

    for (auto *op : llvm::reverse(oldOps))
      op->erase();

    b.setInsertionPointToStart(hostBlk);
    makeOp("host_enqueue", fn.getArguments());

    std::set<std::string> seenResources;
    SmallVector<std::string> neededImpls;
    prog->walk([&](Operation *op) {
      if (auto attr = op->getAttrOfType<StringAttr>("ci.implementation")) {
        auto key = attr.getValue().str();
        auto it = profile.implementations.find(key);
        if (it != profile.implementations.end() &&
            it->second.source == ImplementationSource::Runtime &&
            seenResources.insert(it->second.resource).second)
          neededImpls.push_back(key);
      }
    });
    b.setInsertionPointToStart(sourcesBlk);
    for (auto &impl : neededImpls)
      makeOp("runtime_source", {}, {},
             {b.getNamedAttr("implementation", b.getStringAttr(impl))});

    return success();
  }
};
} // namespace

LogicalResult lowerRowsToCallable(ModuleOp m, const TargetProfile &profile) {
  auto *ctx = m.getContext();
  ctx->getOrLoadDialect<intentdsl::bangc_src::BangcSrcDialect>();
  ctx->getOrLoadDialect<scf::SCFDialect>();
  ctx->getOrLoadDialect<arith::ArithDialect>();
  auto fn = *m.getOps<func::FuncOp>().begin();
  Builder b(ctx);
  fn->setAttr("bangc.entry", b.getStringAttr("intent_enqueue"));
  RowLowerer lowerer(fn, profile);
  if (failed(lowerer.run()))
    return failure();
  llvm::SmallVector<char> metadata;
  llvm::raw_svector_ostream os(metadata);
  if (auto err = describeRowArtifact(fn, os))
    return fn.emitError() << llvm::toString(std::move(err)), failure();
  m->setAttr("bangc_src.package",
             b.getStringAttr(StringRef(metadata.data(), metadata.size())));
  return success();
}
} // namespace CI_UNIT_NAMESPACE
