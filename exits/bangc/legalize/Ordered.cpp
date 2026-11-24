#include "OwnerVectors.h"
#include "Rows.h"
#include "intentdsl/exits/bangc/ir/Dialect.h"
#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "intentdsl/levels/storage/dsa/ir/Owners.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/JSON.h"
#include <cmath>
#include <limits>
#include <set>

namespace CI_UNIT_NAMESPACE {
using namespace mlir;
using namespace compilerinfra;

namespace {
MemRefType memSpacePtr(MLIRContext *ctx, Type elem, int64_t space) {
  return MemRefType::get({ShapedType::kDynamic}, elem, AffineMap{},
                         IntegerAttr::get(IntegerType::get(ctx, 32), space));
}

struct OrderedLowerer {
  func::FuncOp fn;
  const TargetProfile &profile;
  Operation &space;
  OpBuilder b;
  Location loc;
  Operation *constAnchor = nullptr;
  llvm::DenseMap<int64_t, Value> constI64Map;
  Value symGrid, symTaskId, symTaskDim;

  OrderedLowerer(func::FuncOp f, const TargetProfile &prof)
      : fn(f), profile(prof), space(f.front().front()), b(f.getContext()),
        loc(f.getLoc()) {}

  int64_t dimAttr(Value v, StringRef key, unsigned axis) {
    return fn.getArgAttrOfType<DenseI64ArrayAttr>(
        cast<BlockArgument>(v).getArgNumber(), key)[axis];
  }

  int64_t memberExtent(Value v) {
    return intentdsl::kir::memberExtent(fn, v, "block.shape");
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
    return b.create<arith::ConstantIntOp>(loc, v, 32);
  }

  Value cI1(bool v) {
    return b.create<arith::ConstantIntOp>(loc, v ? 1 : 0, 1);
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
    if (v.getType().isInteger(1))
      return b.create<arith::ExtUIOp>(loc, b.getI64Type(), v).getResult();
    return b.create<arith::ExtSIOp>(loc, b.getI64Type(), v).getResult();
  }

  Value floorDivI64(Value lhs, Value rhs) {
    Value q = b.create<arith::DivSIOp>(loc, lhs, rhs);
    Value r = b.create<arith::RemSIOp>(loc, lhs, rhs);
    Value nz = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, r, cI64(0));
    Value diffSign = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, b.create<arith::XOrIOp>(loc, r, rhs),
        cI64(0));
    Value adjust = b.create<arith::AndIOp>(loc, nz, diffSign);
    return b.create<arith::SelectOp>(
        loc, adjust, b.create<arith::SubIOp>(loc, q, cI64(1)), q);
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

  Value nramBuf(Type elemTy, int64_t elements) {
    return makeOp("nram_buffer", {}, {memSpacePtr(b.getContext(), elemTy, 1)},
                  {b.getNamedAttr("elements", b.getI64IntegerAttr(elements))})
        ->getResult(0);
  }

  Value wramBuf(Type elemTy, int64_t elements) {
    return makeOp("wram_buffer", {}, {memSpacePtr(b.getContext(), elemTy, 3)},
                  {b.getNamedAttr("elements", b.getI64IntegerAttr(elements))})
        ->getResult(0);
  }

  Value nramPtr(Value storageBuf, Value byteOff, Type elemTy) {
    return makeOp("nram_ptr", {storageBuf, byteOff},
                  {memSpacePtr(b.getContext(), elemTy, 1)})
        ->getResult(0);
  }

  Value makeAddr(Value base, ArrayRef<Value> coords, ArrayRef<Value> strides,
                 int64_t spaceId, Type elemOverride = {}) {
    Type elemTy = elemOverride ? elemOverride
                               : cast<MemRefType>(base.getType()).getElementType();
    SmallVector<Value> ops{base};
    ops.append(coords.begin(), coords.end());
    ops.append(strides.begin(), strides.end());
    int32_t n = static_cast<int32_t>(coords.size());
    return makeOp("address", ops, {memSpacePtr(b.getContext(), elemTy, spaceId)},
                  {b.getNamedAttr("index_bits", b.getI32IntegerAttr(64)),
                   b.getNamedAttr("operandSegmentSizes",
                                  b.getDenseI32ArrayAttr({1, n, n}))})
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

  void memcpy1D(Value dst, Value src, Value elems, StringRef dir) {
    makeOp("memcpy_1d", {dst, src, elems}, {},
           {b.getNamedAttr("direction", b.getStringAttr(dir)),
            b.getNamedAttr("async", b.getBoolAttr(false))});
  }

  void memcpy2D(Value dst, Value src, Value width, Value dstStride,
                Value srcStride, Value segNum, StringRef dir) {
    makeOp("memcpy_2d", {dst, src, width, dstStride, srcStride, segNum}, {},
           {b.getNamedAttr("direction", b.getStringAttr(dir)),
            b.getNamedAttr("async", b.getBoolAttr(false))});
  }

  void sync(StringRef dom) {
    makeOp("sync", {}, {}, {b.getNamedAttr("domain", b.getStringAttr(dom))});
  }

  void fillVec(Value dst, Value lanes, Value val) {
    makeOp("fill", {dst, lanes, val});
  }

  scf::ForOp forLoop(Value lb, Value ub, Value step,
                     llvm::function_ref<void(Value)> bodyFn) {
    return b.create<scf::ForOp>(
        loc, toIdx(lb), toIdx(ub), toIdx(step), ValueRange{},
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
    return b.create<scf::ForOp>(
        loc, toIdx(lb), toIdx(ub), toIdx(step), inits,
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

  scf::IfOp ifThenElse(Value cond, TypeRange resultTypes,
                       llvm::function_ref<void()> thenFn,
                       llvm::function_ref<void()> elseFn,
                       bool autoYield = true) {
    auto ifOp = b.create<scf::IfOp>(loc, resultTypes, cond, true);
    {
      OpBuilder::InsertionGuard g(b);
      if (!ifOp.thenBlock()->empty() &&
          isa<scf::YieldOp>(ifOp.thenBlock()->back()))
        ifOp.thenBlock()->back().erase();
      if (!ifOp.elseBlock()->empty() &&
          isa<scf::YieldOp>(ifOp.elseBlock()->back()))
        ifOp.elseBlock()->back().erase();
      b.setInsertionPointToEnd(ifOp.thenBlock());
      thenFn();
      if (autoYield)
        b.create<scf::YieldOp>(loc);
      b.setInsertionPointToEnd(ifOp.elseBlock());
      elseFn();
      if (autoYield)
        b.create<scf::YieldOp>(loc);
    }
    return ifOp;
  }

  Value lowerScalarExpr(Operation &op, llvm::DenseMap<Value, Value> &env) {
    StringRef n = op.getName().getStringRef();
    StringRef kind = op.getName().stripDialect();
    auto a = [&](unsigned i) { return env.lookup(op.getOperand(i)); };
    if (auto d = dyn_cast<memref::DimOp>(op))
      return cI64(dimAttr(d.getSource(), "block.shape", *d.getConstantIndex()));
    if (auto c = dyn_cast<arith::ConstantOp>(op)) {
      if (auto fa = dyn_cast<FloatAttr>(c.getValue()))
        return b.create<arith::ConstantOp>(loc, fa);
      auto ia = cast<IntegerAttr>(c.getValue());
      if (c.getType().isIndex() || c.getType().isSignlessInteger(64))
        return cI64(ia.getInt());
      if (c.getType().isInteger(1))
        return cI1(!ia.getValue().isZero());
      return cI32(static_cast<int32_t>(ia.getInt()));
    }
    if (n == "arith.index_cast") {
      if (op.getResult(0).getType().isIndex())
        return toI64(a(0));
      return b.create<arith::TruncIOp>(loc, op.getResult(0).getType(), toI64(a(0)));
    }
    if (n == "arith.extf") {
      Type from = op.getOperand(0).getType();
      if (from.isBF16())
        return makeOp("bf16_scalar", {a(0)}, {b.getF32Type()},
                      {b.getNamedAttr("mode", b.getStringAttr("widen"))},
                      "bf16-scalar-convert")
            ->getResult(0);
      return b.create<arith::ExtFOp>(loc, op.getResult(0).getType(), a(0));
    }
    if (n == "arith.truncf") {
      Type to = op.getResult(0).getType();
      if (to.isBF16())
        return makeOp("bf16_scalar", {a(0)}, {b.getBF16Type()},
                      {b.getNamedAttr("mode", b.getStringAttr("narrow"))},
                      "bf16-scalar-convert")
            ->getResult(0);
      return b.create<arith::TruncFOp>(loc, to, a(0));
    }
    if (n == "arith.maxnumf")
      return makeOp("scalar_math", {a(0), a(1)}, {b.getF32Type()},
                    {b.getNamedAttr("op", b.getStringAttr("max_num"))})
          ->getResult(0);
    if (n == "math.sqrt" || n == "math.exp" || n == "math.exp2" ||
        n == "math.log")
      return makeOp("scalar_math", {a(0)}, {b.getF32Type()},
                    {b.getNamedAttr("op", b.getStringAttr(kind))})
          ->getResult(0);
    if (n == "arith.addi")
      return b.create<arith::AddIOp>(loc, a(0), a(1));
    if (n == "arith.subi")
      return b.create<arith::SubIOp>(loc, a(0), a(1));
    if (n == "arith.muli")
      return b.create<arith::MulIOp>(loc, a(0), a(1));
    if (n == "arith.floordivsi")
      return floorDivI64(a(0), a(1));
    if (n == "arith.remsi")
      return b.create<arith::RemSIOp>(loc, a(0), a(1));
    if (n == "arith.minsi")
      return b.create<arith::MinSIOp>(loc, a(0), a(1));
    if (n == "arith.addf")
      return b.create<arith::AddFOp>(loc, a(0), a(1));
    if (n == "arith.subf")
      return b.create<arith::SubFOp>(loc, a(0), a(1));
    if (n == "arith.mulf")
      return b.create<arith::MulFOp>(loc, a(0), a(1));
    if (n == "arith.divf")
      return b.create<arith::DivFOp>(loc, a(0), a(1));
    if (auto cmp = dyn_cast<arith::CmpIOp>(op))
      return b.create<arith::CmpIOp>(loc, cmp.getPredicate(), a(0), a(1));
    if (auto sel = dyn_cast<arith::SelectOp>(op))
      return b.create<arith::SelectOp>(loc, a(0), a(1), a(2));
    return Value{};
  }

  // Lower cooperative panel fold ("panel-fold") from DSA-assigned fragment slots.
  LogicalResult lowerPanelFold() {
    auto &bb = space.getRegion(0).front();
    int64_t tile = space.getAttrOfType<IntegerAttr>("block.owner_tile").getInt();
    int64_t owners = space.getAttrOfType<IntegerAttr>("block.owners").getInt();
    int64_t M = space.getAttrOfType<IntegerAttr>("dsa.panel_rows").getInt();
    int64_t S = space.getAttrOfType<IntegerAttr>("dsa.panel_span").getInt();
    int64_t wramBytes =
        space.getAttrOfType<IntegerAttr>("dsa.wram_bytes").getInt();
    int64_t memberBytes =
        space.getAttrOfType<IntegerAttr>("dsa.member_bytes").getInt();
    auto scratchOffs =
        space.getAttrOfType<DenseI64ArrayAttr>("dsa.panel_scratch");

    llvm::DenseMap<StringRef, Operation *> roleOp;
    space.walk([&](Operation *op) {
      if (auto r = op->getAttrOfType<StringAttr>("dsa.panel_role"))
        roleOp[r.getValue()] = op;
    });
    Operation *qSlice = roleOp.lookup("query");
    Operation *peakLoopOp = roleOp.lookup("peak_loop");
    Operation *kSlice1 = roleOp.lookup("key_peak");
    Operation *dot1 = roleOp.lookup("score_peak");
    Operation *sumLoopOp = roleOp.lookup("sum_loop");
    Operation *kSlice2 = roleOp.lookup("key_sum");
    Operation *dot2 = roleOp.lookup("score_sum");
    Operation *weightDef = roleOp.lookup("weight");
    Operation *wTrunc = roleOp.lookup("weight_narrow");
    Operation *vSlice = roleOp.lookup("value");
    Operation *normMul = roleOp.lookup("norm");
    Operation *outTrunc = roleOp.lookup("output_narrow");
    Operation *outWrite = roleOp.lookup("output");
    if (!qSlice || !peakLoopOp || !kSlice1 || !dot1 || !sumLoopOp ||
        !kSlice2 || !dot2 || !weightDef || !wTrunc || !vSlice || !normMul ||
        !outTrunc || !outWrite || !dot1->hasOneUse() || !dot2->hasOneUse())
      return space.emitError("missing DSA panel-fold role bindings");

    scf::ForOp maxLoop = cast<scf::ForOp>(peakLoopOp);
    scf::ForOp sumLoop = cast<scf::ForOp>(sumLoopOp);
    Type halfTy =
        cast<MemRefType>(qSlice->getOperand(0).getType()).getElementType();
    int64_t halfB = halfTy.getIntOrFloatBitWidth() / 8;
    int64_t qRank = cast<MemRefType>(qSlice->getOperand(0).getType()).getRank();
    int64_t D = dimAttr(qSlice->getOperand(0), "block.shape", qRank - 1);

    auto mulSc1 = cast<arith::MulFOp>(*dot1->getUsers().begin());
    Value gain1Src = mulSc1.getLhs() == dot1->getResult(0) ? mulSc1.getRhs()
                                                           : mulSc1.getLhs();
    auto mulSc2 = cast<arith::MulFOp>(*dot2->getUsers().begin());
    Value gain2Src = mulSc2.getLhs() == dot2->getResult(0) ? mulSc2.getRhs()
                                                           : mulSc2.getLhs();
    bool causal = false;
    Operation *expOp = weightDef;
    if (auto br = dyn_cast<scf::IfOp>(weightDef)) {
      causal = true;
      expOp = br.thenBlock()->getTerminator()->getOperand(0).getDefiningOp();
    }

    Value kView = kSlice1->getOperand(0), vView = vSlice->getOperand(0),
          qView = qSlice->getOperand(0), yView = outWrite->getOperand(1);
    int64_t kRank = cast<MemRefType>(kView.getType()).getRank();
    int64_t K = dimAttr(kView, "block.shape", kRank - 2);
    int64_t DV = dimAttr(vView, "block.shape", kRank - 1);
    bool useExp2 = expOp->getName().getStringRef() == "math.exp2";

    Value storage = nramBuf(b.getI8Type(), memberBytes);
    Value filter = wramBuf(halfTy, wramBytes / halfB);

    auto slotAt = [&](Operation *op, unsigned idx, Type elemTy) {
      int64_t off = op->getAttrOfType<DenseI64ArrayAttr>("dsa.member_offsets")[idx];
      return nramPtr(storage, cI64(off), elemTy);
    };

    Value qBuf = slotAt(qSlice, 0, halfTy);
    Value peakBuf = slotAt(maxLoop, 0, b.getF32Type());
    Value kBuf1 = slotAt(kSlice1, 0, halfTy);
    Value scores1Buf = slotAt(dot1, 0, b.getF32Type());
    Value massBuf = slotAt(sumLoop, 0, b.getF32Type());
    Value weightedBuf = slotAt(sumLoop, 1, b.getF32Type());
    Value kBuf2 = slotAt(kSlice2, 0, halfTy);
    Value scores2Buf = slotAt(dot2, 0, b.getF32Type());
    Value probBuf = slotAt(weightDef, 0, b.getF32Type());
    Value pHalfBuf = slotAt(wTrunc, 0, halfTy);
    Value vBuf = slotAt(vSlice, 0, halfTy);
    Value vtBuf = slotAt(vSlice, 1, halfTy);
    Value normBuf = slotAt(normMul, 0, b.getF32Type());
    Value outBuf = slotAt(outTrunc, 0, halfTy);

    Value magScratch = nramPtr(storage, cI64(scratchOffs[0]), b.getBF16Type());
    Value flagScratch = nramPtr(storage, cI64(scratchOffs[1]), b.getBF16Type());
    Value scalarOutBuf = nramPtr(storage, cI64(scratchOffs[2]), b.getF32Type());
    Value reduceScratch =
        nramPtr(storage, cI64(scratchOffs[2] + 128), b.getF32Type());
    Value convTmp =
        nramPtr(storage, cI64(scratchOffs[2] + 512), b.getF32Type());

    auto sliceBasePtr = [&](Value view, ValueRange prefixCoords, Value rowOrTok) {
      int64_t rank = cast<MemRefType>(view.getType()).getRank();
      SmallVector<Value> coords, strides;
      for (int64_t i = 0; i < rank - 2; ++i) {
        coords.push_back(prefixCoords[i]);
        strides.push_back(cI64(dimAttr(view, "block.strides", i)));
      }
      coords.push_back(rowOrTok);
      strides.push_back(cI64(dimAttr(view, "block.strides", rank - 2)));
      return makeAddr(view, coords, strides, 0);
    };

    auto stageRows = [&](Value dstNram, Value view, Value ptr, Value nRows,
                         int64_t maxRows, int64_t cols, bool zeroTail) {
      int64_t rank = cast<MemRefType>(view.getType()).getRank();
      int64_t rowStride = dimAttr(view, "block.strides", rank - 2);
      if (zeroTail) {
        ifThen(
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, nRows,
                                    cI64(maxRows)),
            [&]() {
              fillVec(dstNram, cI64(maxRows * cols),
                      b.create<arith::ConstantOp>(loc, b.getZeroAttr(halfTy)));
            });
      }
      if (rowStride == cols) {
        memcpy1D(dstNram, ptr, b.create<arith::MulIOp>(loc, nRows, cI64(cols)),
                 "GDRAM2NRAM");
      } else {
        ifThenElse(
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, nRows,
                                    cI64(1)),
            {}, [&]() { memcpy1D(dstNram, ptr, cI64(cols), "GDRAM2NRAM"); },
            [&]() {
              memcpy2D(dstNram, ptr, cI64(cols), cI64(cols), cI64(rowStride),
                       b.create<arith::SubIOp>(loc, nRows, cI64(1)),
                       "GDRAM2NRAM");
            });
      }
    };

    auto emitPanelProduct = [&](Value outF32, Value lhsHalf,
                                Value rhsFilterMajor, int64_t Kdim,
                                int64_t Cdim, bool accumulate) {
      auto runWram = [&]() {
        StringRef mKind = halfTy.isBF16() ? "bf16" : "f16";
        StringRef cKey = halfTy.isBF16() ? "wram-conv-bf16" : "wram-conv-f16";
        if (Cdim == 64) {
          sync("compute");
          memcpy1D(filter, rhsFilterMajor, cI64(64 * Kdim), "NRAM2WRAM");
          makeOp("wram_conv", {outF32, lhsHalf, filter, cI1(accumulate)}, {},
                 {b.getNamedAttr("bm", b.getI64IntegerAttr(M)),
                  b.getNamedAttr("bn", b.getI64IntegerAttr(64)),
                  b.getNamedAttr("bk", b.getI64IntegerAttr(Kdim)),
                  b.getNamedAttr("mode", b.getStringAttr(mKind))},
                 cKey);
          sync("compute");
        } else if (Cdim == 32) {
          Value padNram = nramPtr(storage, cI64(scratchOffs[2] + 512), halfTy);
          fillVec(padNram, cI64(64 * Kdim),
                  b.create<arith::ConstantOp>(loc, b.getZeroAttr(halfTy)));
          memcpy1D(padNram, rhsFilterMajor, cI64(32 * Kdim), "NRAM2NRAM");
          sync("io");
          memcpy1D(filter, padNram, cI64(64 * Kdim), "NRAM2WRAM");
          makeOp("wram_conv", {convTmp, lhsHalf, filter, cI1(false)}, {},
                 {b.getNamedAttr("bm", b.getI64IntegerAttr(M)),
                  b.getNamedAttr("bn", b.getI64IntegerAttr(64)),
                  b.getNamedAttr("bk", b.getI64IntegerAttr(Kdim)),
                  b.getNamedAttr("mode", b.getStringAttr(mKind))},
                 cKey);
          sync("compute");
          forLoop(cI64(0), cI64(M), cI64(1), [&](Value r) {
            Value dstRow = makeAddr(outF32, {r}, {cI64(32)}, 1);
            Value srcRow = makeAddr(convTmp, {r}, {cI64(64)}, 1);
            if (accumulate)
              makeOp("vec_binary", {dstRow, dstRow, srcRow, cI64(32)}, {},
                     {b.getNamedAttr("op", b.getStringAttr("add"))});
            else
              memcpy1D(dstRow, srcRow, cI64(32), "NRAM2NRAM");
          });
          sync("compute");
        } else {
          forLoop(cI64(0), cI64(Cdim), cI64(64), [&](Value fOff) {
            Value fltSrc = makeAddr(rhsFilterMajor, {fOff}, {cI64(Kdim)}, 1);
            sync("compute");
            memcpy1D(filter, fltSrc, cI64(64 * Kdim), "NRAM2WRAM");
            makeOp("wram_conv", {convTmp, lhsHalf, filter, cI1(false)}, {},
                   {b.getNamedAttr("bm", b.getI64IntegerAttr(M)),
                    b.getNamedAttr("bn", b.getI64IntegerAttr(64)),
                    b.getNamedAttr("bk", b.getI64IntegerAttr(Kdim)),
                    b.getNamedAttr("mode", b.getStringAttr(mKind))},
                   cKey);
            sync("compute");
            forLoop(cI64(0), cI64(M), cI64(1), [&](Value r) {
              Value dstRow =
                  makeAddr(outF32, {r, fOff}, {cI64(Cdim), cI64(1)}, 1);
              Value srcRow = makeAddr(convTmp, {r}, {cI64(64)}, 1);
              if (accumulate)
                makeOp("vec_binary", {dstRow, dstRow, srcRow, cI64(64)}, {},
                       {b.getNamedAttr("op", b.getStringAttr("add"))});
              else
                memcpy1D(dstRow, srcRow, cI64(64), "NRAM2NRAM");
            });
          });
        }
      };

      if (halfTy.isBF16()) {
        sync("compute");
        Value okL = makeOp("bf16_panel_ok",
                           {lhsHalf, magScratch, flagScratch, cI64(M * Kdim)},
                           {b.getI1Type()}, {}, "bf16-panel-guard")
                        ->getResult(0);
        Value okR =
            makeOp("bf16_panel_ok",
                   {rhsFilterMajor, magScratch, flagScratch, cI64(Cdim * Kdim)},
                   {b.getI1Type()}, {}, "bf16-panel-guard")
                ->getResult(0);
        Value bothOk = b.create<arith::AndIOp>(loc, okL, okR);
        ifThenElse(bothOk, {}, runWram, [&]() {
          if (!accumulate)
            fillVec(outF32, cI64(M * Cdim), cF32(0.0f));
          sync("compute");
          makeOp("bf16_ieee_dot",
                 {outF32, lhsHalf, rhsFilterMajor, cI64(M), cI64(Cdim),
                  cI64(Kdim), cI64(Kdim), cI64(Cdim)},
                 {}, {}, "bf16-ieee-dot");
        });
      } else {
        runWram();
      }
    };

    auto narrowPanel = [&](Value dstHalf, Value srcF32, int64_t count) {
      if (halfTy.isBF16())
        makeOp("bf16_tile", {dstHalf, srcF32, cI64(count), cI64(count)}, {},
               {b.getNamedAttr("mode", b.getStringAttr("narrow"))},
               "bf16-tile-convert");
      else
        makeOp("vec_cast", {dstHalf, srcF32, cI64(count)}, {},
               {b.getNamedAttr("kind", b.getStringAttr("f32_to_f16"))});
    };

    int64_t stepTile = std::max<int64_t>(tile, M);
    auto axes = space.getAttrOfType<DenseI64ArrayAttr>("axes");
    forLoop(symTaskId, symGrid, symTaskDim, [&](Value job) {
      forLoop(cI64(0), cI64(stepTile), cI64(M), [&](Value sub) {
        Value ownerBase = b.create<arith::AddIOp>(
            loc, b.create<arith::MulIOp>(loc, job, cI64(stepTile)), sub);
        ifThen(b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                       ownerBase, cI64(owners)),
               [&]() {
                 llvm::DenseMap<Value, Value> env;
                 for (Value arg : fn.getArguments())
                   env[arg] = arg;
                 int64_t pitch = 1;
                 for (int i = axes.size() - 1; i >= 0; --i) {
                   int64_t ext = std::max<int64_t>(
                       1, dimAttr(space.getOperand(i), "block.shape", axes[i]));
                   Value coord = b.create<arith::RemSIOp>(
                       loc, b.create<arith::DivSIOp>(loc, ownerBase, cI64(pitch)),
                       cI64(ext));
                   env[bb.getArgument(i)] = coord;
                   pitch *= ext;
                 }
                 Value rowBase = env.lookup(bb.getArguments().back());

                 // Evaluate outer scalar setup ops (head group division, scale, etc.).
                 for (auto &op : bb.without_terminator()) {
                   if (isa<scf::ForOp, scf::IfOp>(op) ||
                       op.getName().stripDialect() == "slice_read" ||
                       op.getName().stripDialect() == "slice_write")
                     continue;
                   if (op.getNumResults() == 1 &&
                       !isa<RankedTensorType>(op.getResult(0).getType())) {
                     if (Value sv = lowerScalarExpr(op, env))
                       env[op.getResult(0)] = sv;
                   }
                 }
                 auto evalInLoop = [&](Block &lBody) {
                   for (auto &op : lBody.without_terminator()) {
                     if (isa<scf::ForOp, scf::IfOp>(op))
                       continue;
                     if (op.getNumResults() == 1 &&
                         !isa<RankedTensorType>(op.getResult(0).getType()) &&
                         llvm::all_of(op.getOperands(),
                                      [&](Value u) { return env.count(u); })) {
                       if (Value sv = lowerScalarExpr(op, env))
                         env[op.getResult(0)] = sv;
                     }
                   }
                 };
                 evalInLoop(*maxLoop.getBody());
                 evalInLoop(*sumLoop.getBody());

                 Value gain1Val = env.lookup(gain1Src);
                 Value gain2Val = env.lookup(gain2Src);

                 SmallVector<Value> qPrefix, kPrefix, vPrefix, yPrefix;
                 for (Value c : qSlice->getOperands().drop_front().drop_back())
                   qPrefix.push_back(env.lookup(c));
                 for (Value c : kSlice1->getOperands().drop_front().drop_back())
                   kPrefix.push_back(env.lookup(c));
                 for (Value c : vSlice->getOperands().drop_front().drop_back())
                   vPrefix.push_back(env.lookup(c));
                 for (Value c : outWrite->getOperands().drop_front(2).drop_back())
                   yPrefix.push_back(env.lookup(c));

                 // Stage Q [M, D].
                 stageRows(qBuf, qView, sliceBasePtr(qView, qPrefix, rowBase),
                           cI64(M), M, D, false);
                 sync("io");

                 Value tokLimit =
                     causal ? b.create<arith::MinSIOp>(
                                  loc, b.create<arith::AddIOp>(loc, rowBase, cI64(M)),
                                  cI64(K))
                                  .getResult()
                            : cI64(K);

                 // Pass 1: peak[0..M-1]
                 fillVec(peakBuf, cI64(M),
                         cF32(-std::numeric_limits<float>::infinity()));
                 forLoop(cI64(0), tokLimit, cI64(S), [&](Value t0) {
                   Value nt = b.create<arith::MinSIOp>(
                       loc, b.create<arith::SubIOp>(loc, cI64(K), t0), cI64(S));
                   stageRows(kBuf1, kView, sliceBasePtr(kView, kPrefix, t0), nt,
                             S, D, true);
                   sync("io");
                   emitPanelProduct(scores1Buf, qBuf, kBuf1, D, S, false);
                   makeOp("vec_scalar",
                          {scores1Buf, scores1Buf, gain1Val, cI64(M * S)}, {},
                          {b.getNamedAttr("op", b.getStringAttr("mul"))});
                   sync("compute");
                   forLoop(cI64(0), cI64(M), cI64(1), [&](Value r) {
                     Value live =
                         causal
                             ? b.create<arith::MinSIOp>(
                                   loc,
                                   b.create<arith::MaxSIOp>(
                                       loc,
                                       b.create<arith::SubIOp>(
                                           loc,
                                           b.create<arith::AddIOp>(
                                               loc,
                                               b.create<arith::AddIOp>(
                                                   loc, rowBase, r),
                                               cI64(1)),
                                           t0),
                                       cI64(0)),
                                   nt)
                                   .getResult()
                             : nt;
                     ifThen(b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                                    live, cI64(0)),
                            [&]() {
                              Value sRow =
                                  makeAddr(scores1Buf, {r}, {cI64(S)}, 1);
                              makeOp("row_reduce",
                                     {scalarOutBuf, reduceScratch, sRow,
                                      cI64(1), cI64(S), live},
                                     {},
                                     {b.getNamedAttr("algebra",
                                                     b.getStringAttr("max_num"))},
                                     "row-reduce");
                              Value curP = loadAt(peakBuf, r);
                              Value blkP = loadAt(scalarOutBuf, cI64(0));
                              Value nxtP =
                                  makeOp("scalar_math", {curP, blkP},
                                         {b.getF32Type()},
                                         {b.getNamedAttr(
                                             "op", b.getStringAttr("max_num"))})
                                      ->getResult(0);
                              storeAt(peakBuf, r, nxtP);
                            });
                   });
                 });

                 // Pass 2: mass[0..M-1] and weighted[0..M*DV-1]
                 fillVec(massBuf, cI64(M), cF32(0.0f));
                 fillVec(weightedBuf, cI64(M * DV), cF32(0.0f));
                 forLoop(cI64(0), tokLimit, cI64(S), [&](Value t0) {
                   Value nt = b.create<arith::MinSIOp>(
                       loc, b.create<arith::SubIOp>(loc, cI64(K), t0), cI64(S));
                   stageRows(kBuf2, kView, sliceBasePtr(kView, kPrefix, t0), nt,
                             S, D, true);
                   sync("io");
                   emitPanelProduct(scores2Buf, qBuf, kBuf2, D, S, false);
                   makeOp("vec_scalar",
                          {scores2Buf, scores2Buf, gain2Val, cI64(M * S)}, {},
                          {b.getNamedAttr("op", b.getStringAttr("mul"))});
                   sync("compute");
                   forLoop(cI64(0), cI64(M), cI64(1), [&](Value r) {
                     Value sRow = makeAddr(scores2Buf, {r}, {cI64(S)}, 1);
                     Value pRow = makeAddr(probBuf, {r}, {cI64(S)}, 1);
                     Value live =
                         causal
                             ? b.create<arith::MinSIOp>(
                                   loc,
                                   b.create<arith::MaxSIOp>(
                                       loc,
                                       b.create<arith::SubIOp>(
                                           loc,
                                           b.create<arith::AddIOp>(
                                               loc,
                                               b.create<arith::AddIOp>(
                                                   loc, rowBase, r),
                                               cI64(1)),
                                           t0),
                                       cI64(0)),
                                   nt)
                                   .getResult()
                             : nt;
                     ifThenElse(
                         b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                                 live, cI64(0)),
                         {},
                         [&]() {
                           Value pk = loadAt(peakBuf, r);
                           makeOp("vec_scalar", {pRow, sRow, pk, cI64(S)}, {},
                                  {b.getNamedAttr("op", b.getStringAttr("sub"))});
                           makeOp(useExp2 ? "row_exp2" : "row_exp",
                                  {pRow, reduceScratch, pRow, live, cI64(S)}, {},
                                  {},
                                  useExp2 ? "validated-exp2" : "validated-exp");
                           ifThen(b.create<arith::CmpIOp>(
                                      loc, arith::CmpIPredicate::slt, live,
                                      cI64(S)),
                                  [&]() {
                                    sync("compute");
                                    forLoop(live, cI64(S), cI64(1),
                                            [&](Value j) {
                                              storeAt(pRow, j, cF32(0.0f));
                                            });
                                  });
                           makeOp("row_reduce",
                                  {scalarOutBuf, reduceScratch, pRow, cI64(1),
                                   cI64(S), live},
                                  {},
                                  {b.getNamedAttr("algebra",
                                                  b.getStringAttr("sum"))},
                                  "row-reduce");
                           storeAt(massBuf, r,
                                   b.create<arith::AddFOp>(
                                       loc, loadAt(massBuf, r),
                                       loadAt(scalarOutBuf, cI64(0))));
                         },
                         [&]() { fillVec(pRow, cI64(S), cF32(0.0f)); });
                   });
                   narrowPanel(pHalfBuf, probBuf, M * S);
                   sync("compute");

                   stageRows(vBuf, vView, sliceBasePtr(vView, vPrefix, t0), nt,
                             S, DV, true);
                   sync("io");
                   makeOp("tile_transpose", {vtBuf, vBuf, cI64(S), cI64(DV)},
                          {}, {}, "nram-transpose");
                   sync("compute");
                   emitPanelProduct(weightedBuf, pHalfBuf, vtBuf, S, DV, true);
                 });

                 // Normalize and publish Y [M, DV].
                 forLoop(cI64(0), cI64(M), cI64(1), [&](Value r) {
                   Value mVal = loadAt(massBuf, r);
                   Value inv =
                       K == 0 ? cF32(0.0f)
                              : b.create<arith::DivFOp>(loc, cF32(1.0f), mVal)
                                    .getResult();
                   Value wRow = makeAddr(weightedBuf, {r}, {cI64(DV)}, 1);
                   Value nRow = makeAddr(normBuf, {r}, {cI64(DV)}, 1);
                   makeOp("vec_scalar", {nRow, wRow, inv, cI64(DV)}, {},
                          {b.getNamedAttr("op", b.getStringAttr("mul"))});
                 });
                 narrowPanel(outBuf, normBuf, M * DV);
                 sync("compute");

                 Value yPtr = sliceBasePtr(yView, yPrefix, rowBase);
                 int64_t yRank = cast<MemRefType>(yView.getType()).getRank();
                 int64_t yRowStride =
                     dimAttr(yView, "block.strides", yRank - 2);
                 if (yRowStride == DV) {
                   memcpy1D(yPtr, outBuf, cI64(M * DV), "NRAM2GDRAM");
                 } else {
                   memcpy2D(yPtr, outBuf, cI64(DV), cI64(yRowStride), cI64(DV),
                            cI64(M - 1), "NRAM2GDRAM");
                 }
               });
      });
    });
    return success();
  }

  // Lower multi-owner packed stream recurrence (dsa.owner_pack > 1).
  LogicalResult lowerPackedStreams(int64_t pack) {
    auto &bb = space.getRegion(0).front();
    int64_t tile = space.getAttrOfType<IntegerAttr>("block.owner_tile").getInt();
    int64_t owners = space.getAttrOfType<IntegerAttr>("block.owners").getInt();
    int64_t memberBytes =
        space.getAttrOfType<IntegerAttr>("dsa.member_bytes").getInt();
    int64_t streamBytes =
        space.getAttrOfType<IntegerAttr>("dsa.stream_bytes").getInt();
    int64_t deferredBytes =
        space.getAttrOfType<IntegerAttr>("bangc.deferred_bytes").getInt();

    Value innerOwner = bb.getArguments().back();
    int64_t K = 32;
    int64_t steps = 1;
    scf::ForOp mainLoop;
    for (auto &op : bb.without_terminator()) {
      if (auto l = dyn_cast<scf::ForOp>(op))
        mainLoop = l;
    }
    space.walk([&](Operation *op) {
      for (Value r : op->getResults())
        if (isa<RankedTensorType>(r.getType()))
          K = memberExtent(r);
      if (auto d = op->getAttrOfType<DenseI64ArrayAttr>("dsa.stream"))
        steps = d[2];
    });

    int64_t chunkSteps = std::min<int64_t>(steps, 128);
    Value storage = memberBytes ? nramBuf(b.getI8Type(), memberBytes) : Value{};
    Value streambuf = nramBuf(b.getI8Type(), streamBytes);
    Value deferred =
        deferredBytes ? nramBuf(b.getF32Type(), steps * pack * K) : Value{};
    Value workBuf =
        nramBuf(b.getF32Type(),
                deferredBytes ? chunkSteps * pack * K
                              : std::max<int64_t>(128, pack * K * 2));
    Value scalarOutBuf = nramBuf(b.getF32Type(), 32);

    auto isPackedVal = [&](Value v) {
      return intentdsl::dsa::ownerValueDependsOn(v, innerOwner);
    };

    int64_t stepTile = std::max<int64_t>(tile, pack);
    auto axes = space.getAttrOfType<DenseI64ArrayAttr>("axes");
    forLoop(symTaskId, symGrid, symTaskDim, [&](Value job) {
      forLoop(cI64(0), cI64(stepTile), cI64(pack), [&](Value sub) {
        Value ownerBase = b.create<arith::AddIOp>(
            loc, b.create<arith::MulIOp>(loc, job, cI64(stepTile)), sub);
        ifThen(b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                       ownerBase, cI64(owners)),
               [&]() {
                 llvm::DenseMap<Value, Value> env;
                 for (Value arg : fn.getArguments())
                   env[arg] = arg;
                 int64_t pitch = 1;
                 for (int i = axes.size() - 1; i >= 0; --i) {
                   int64_t ext = std::max<int64_t>(
                       1, dimAttr(space.getOperand(i), "block.shape", axes[i]));
                   Value coord = b.create<arith::RemSIOp>(
                       loc, b.create<arith::DivSIOp>(loc, ownerBase, cI64(pitch)),
                       cI64(ext));
                   env[bb.getArgument(i)] = coord;
                   pitch *= ext;
                 }

                 auto emitStreamTransfers = [&](scf::ForOp loop, bool publish,
                                                llvm::DenseMap<Operation *, Value> &sPtrs) {
                   for (auto &access : *loop.getBody()) {
                     auto d = access.getAttrOfType<DenseI64ArrayAttr>("dsa.stream");
                     if (!d)
                       continue;
                     auto l = dyn_cast<memref::LoadOp>(access);
                     auto w = dyn_cast<memref::StoreOp>(access);
                     bool sliced = access.getName().stripDialect() == "slice_read";
                     Value view =
                         l ? l.getMemref() : w ? w.getMemref() : access.getOperand(0);
                     SmallVector<Value> coords;
                     if (sliced) {
                       llvm::append_range(coords, access.getOperands().drop_front());
                       coords.insert(
                           coords.begin() +
                               access.getAttrOfType<IntegerAttr>("axis").getInt(),
                           Value{});
                     } else {
                       llvm::append_range(coords, l ? l.getIndices() : w.getIndices());
                     }
                     Type elemTy = cast<MemRefType>(view.getType()).getElementType();
                     if (!publish)
                       sPtrs[&access] = nramPtr(streambuf, cI64(d[0]), elemTy);
                     if (bool(w) != publish)
                       continue;
                     Value local = sPtrs.lookup(&access);
                     SmallVector<Value> addrC, addrS;
                     for (auto [ax, cv] : llvm::enumerate(coords)) {
                       if (cv && int64_t(ax) != d[1]) {
                         addrC.push_back(env.lookup(cv));
                         addrS.push_back(cI64(dimAttr(view, "block.strides", ax)));
                       }
                     }
                     Value gPtr = makeAddr(view, addrC, addrS, 0);
                     int64_t width = d[3];
                     int64_t stepStride = dimAttr(view, "block.strides", d[1]);
                     if (width == stepStride) {
                       memcpy1D(publish ? gPtr : local, publish ? local : gPtr,
                                cI64(d[2] * width),
                                publish ? "NRAM2GDRAM" : "GDRAM2NRAM");
                     } else {
                       memcpy2D(publish ? gPtr : local, publish ? local : gPtr,
                                cI64(width),
                                cI64(publish ? stepStride : width),
                                cI64(publish ? width : stepStride),
                                cI64(d[2] - 1),
                                publish ? "NRAM2GDRAM" : "GDRAM2NRAM");
                     }
                   }
                 };

                 // Top-level instructions in bb.
                 for (auto &op : bb.without_terminator()) {
                   StringRef sn = op.getName().stripDialect();
                   if (auto loop = dyn_cast<scf::ForOp>(op)) {
                     llvm::DenseMap<Operation *, Value> streamPtrs;
                     emitStreamTransfers(loop, false, streamPtrs);
                     sync("io");

                     // Bind carried member slots.
                     SmallVector<Value> carryTemps;
                     unsigned offPos = 0;
                     auto loopOffs =
                         op.getAttrOfType<DenseI64ArrayAttr>("dsa.member_offsets");
                     for (auto [initV, iterA, resV] :
                          llvm::zip(loop.getInitArgs(), loop.getRegionIterArgs(),
                                    loop.getResults())) {
                       Type et = cast<RankedTensorType>(resV.getType()).getElementType();
                       int64_t elems = (isPackedVal(resV) ? pack : 1) * K;
                       Value dstSlot = nramPtr(storage, cI64(loopOffs[offPos++]), et);
                       Value tmpSlot = nramPtr(storage, cI64(loopOffs[offPos++]), et);
                       if (isPackedVal(resV) && !isPackedVal(initV)) {
                         for (int64_t pIdx = 0; pIdx < pack; ++pIdx)
                           memcpy1D(makeAddr(dstSlot, {cI64(pIdx)}, {cI64(K)}, 1),
                                    env.lookup(initV), cI64(K), "NRAM2NRAM");
                         sync("io");
                       } else {
                         memcpy1D(dstSlot, env.lookup(initV), cI64(elems),
                                  "NRAM2NRAM");
                         sync("io");
                       }
                       env[iterA] = env[resV] = dstSlot;
                       carryTemps.push_back(tmpSlot);
                     }

                     // Detect optional pre-broadcastable packed scalar load -> lane_splat.
                     Operation *preBcastSplat = nullptr;
                     Operation *preBcastLoad = nullptr;
                     if (deferredBytes && (chunkSteps * pack) % 32 == 0 &&
                         (steps % chunkSteps == 0 ||
                          ((steps % chunkSteps) * pack) % 32 == 0)) {
                       for (auto &bop : *loop.getBody()) {
                         if (bop.getName().stripDialect() == "lane_splat") {
                           Operation *ld = bop.getOperand(0).getDefiningOp();
                           if (ld && isa<memref::LoadOp>(ld) &&
                               ld->hasAttr("dsa.stream") &&
                               ld->getAttrOfType<DenseI64ArrayAttr>("dsa.stream")[4] ==
                                   pack) {
                             if (!preBcastSplat) {
                               preBcastSplat = &bop;
                               preBcastLoad = ld;
                             } else {
                               preBcastSplat = nullptr;
                               break;
                             }
                           }
                         }
                       }
                     }

                     StringRef loopImpl =
                         op.hasAttr("bangc.vector_impl")
                             ? op.getAttrOfType<StringAttr>("bangc.vector_impl")
                                   .getValue()
                             : StringRef();

                     forLoop(cI64(0), cI64(steps), cI64(chunkSteps), [&](Value w0) {
                       Value curSteps = b.create<arith::MinSIOp>(
                           loc, b.create<arith::SubIOp>(loc, cI64(steps), w0),
                           cI64(chunkSteps));
                       Value curRows =
                           b.create<arith::MulIOp>(loc, curSteps, cI64(pack));
                       if (preBcastSplat) {
                         Value vSrc = makeAddr(streamPtrs.lookup(preBcastLoad),
                                               {w0}, {cI64(pack)}, 1);
                         Value tmpArea =
                             makeAddr(deferred, {w0}, {cI64(pack * K)}, 1);
                         memcpy1D(tmpArea, vSrc, curRows, "NRAM2NRAM");
                         for (int64_t d = 1; d < K; d *= 2) {
                           Value chunkElems =
                               b.create<arith::MulIOp>(loc, curRows, cI64(d));
                           Value dstDup = makeAddr(tmpArea, {chunkElems},
                                                   {cI64(1)}, 1);
                           memcpy1D(dstDup, tmpArea, chunkElems, "NRAM2NRAM");
                         }
                         sync("io");
                         makeOp("tile_transpose",
                                {workBuf, tmpArea, cI64(K), curRows}, {}, {},
                                "nram-transpose");
                         sync("compute");
                       }

                       forLoop(cI64(0), curSteps, cI64(1), [&](Value localT) {
                         Value iv = b.create<arith::AddIOp>(loc, w0, localT);
                         env[loop.getInductionVar()] = iv;

                         for (auto &bop : loop.getBody()->without_terminator()) {
                           StringRef bsn = bop.getName().stripDialect();
                           StringRef bfn = bop.getName().getStringRef();
                           StringRef bImpl =
                               bop.hasAttr("bangc.vector_impl")
                                   ? bop.getAttrOfType<StringAttr>(
                                            "bangc.vector_impl")
                                         .getValue()
                                   : StringRef();
                           if (isa<memref::StoreOp>(bop)) {
                             if (bImpl == "deferred-row-reduction")
                               continue;
                             Value sBuf = streamPtrs.lookup(&bop);
                             forLoop(cI64(0), cI64(pack), cI64(1), [&](Value pIdx) {
                               Value idx = b.create<arith::AddIOp>(
                                   loc, b.create<arith::MulIOp>(loc, iv, cI64(pack)),
                                   pIdx);
                               storeAt(sBuf, idx, loadAt(env.lookup(bop.getOperand(0)), pIdx));
                             });
                             continue;
                           }
                           if (auto ld = dyn_cast<memref::LoadOp>(bop)) {
                             auto sd = bop.getAttrOfType<DenseI64ArrayAttr>("dsa.stream");
                             Value sBuf = streamPtrs.lookup(&bop);
                             if (sd[4] == 1)
                               env[bop.getResult(0)] = loadAt(sBuf, iv);
                             else
                               env[bop.getResult(0)] =
                                   makeAddr(sBuf, {iv}, {cI64(pack)}, 1);
                             continue;
                           }
                           if (bsn == "lane_sum") {
                             if (bImpl == "deferred-row-reduction")
                               continue;
                             makeOp("row_reduce",
                                    {scalarOutBuf, workBuf,
                                     env.lookup(bop.getOperand(0)), cI64(pack),
                                     cI64(K), cI64(K)},
                                    {},
                                    {b.getNamedAttr("algebra",
                                                    b.getStringAttr("sum"))},
                                    "row-reduce");
                             env[bop.getResult(0)] = scalarOutBuf;
                             continue;
                           }
                           if (bop.getNumResults() == 1 &&
                               isa<RankedTensorType>(bop.getResult(0).getType())) {
                             Value resV = bop.getResult(0);
                             if (&bop == preBcastSplat) {
                               env[resV] =
                                   makeAddr(workBuf, {localT}, {cI64(pack * K)}, 1);
                               continue;
                             }
                             if (bImpl == "broadcast-multiplier") {
                               env[resV] = env.lookup(bop.getOperand(0));
                               continue;
                             }
                             if (bImpl == "stream-view") {
                               env[resV] = makeAddr(streamPtrs.lookup(&bop),
                                                    {iv}, {cI64(K)}, 1);
                               continue;
                             }
                             Value local =
                                 bImpl == "deferred-row-reduction"
                                     ? makeAddr(deferred, {iv}, {cI64(pack * K)}, 1)
                                     : nramPtr(storage,
                                               cI64(bop.getAttrOfType<DenseI64ArrayAttr>(
                                                   "dsa.member_offsets")[0]),
                                               b.getF32Type());
                             env[resV] = local;
                             if (bsn == "slice_read") {
                               memcpy1D(local,
                                        makeAddr(streamPtrs.lookup(&bop), {iv},
                                                 {cI64(K)}, 1),
                                        cI64(K), "NRAM2NRAM");
                             } else if (bsn == "lane_splat") {
                               Value scIn = env.lookup(bop.getOperand(0));
                               if (isa<MemRefType>(scIn.getType())) {
                                 forLoop(cI64(0), cI64(pack), cI64(1),
                                         [&](Value pIdx) {
                                           Value dstP = makeAddr(local, {pIdx},
                                                                 {cI64(K)}, 1);
                                           fillVec(dstP, cI64(K),
                                                   loadAt(scIn, pIdx));
                                         });
                               } else {
                                 int64_t elems =
                                     (isPackedVal(resV) ? pack : 1) * K;
                                 fillVec(local, cI64(elems), scIn);
                               }
                             } else {
                               StringRef vOp = bfn == "arith.addf"   ? "add"
                                               : bfn == "arith.subf" ? "sub"
                                                                     : "mul";
                               auto isSplat = [&](unsigned idx) {
                                 auto *def = bop.getOperand(idx).getDefiningOp();
                                 auto tag =
                                     def ? def->getAttrOfType<StringAttr>(
                                               "bangc.vector_impl")
                                         : StringAttr();
                                 return tag &&
                                        tag.getValue() ==
                                            "broadcast-multiplier";
                               };
                               bool p0 = isPackedVal(bop.getOperand(0));
                               bool p1 = isPackedVal(bop.getOperand(1));
                               int64_t elems = ((p0 || p1) ? pack : 1) * K;
                               Value v0 = env.lookup(bop.getOperand(0));
                               Value v1 = env.lookup(bop.getOperand(1));
                               if (vOp == "mul" && isSplat(0) && isSplat(1)) {
                                 fillVec(local, cI64(elems),
                                         b.create<arith::MulFOp>(loc, v0, v1));
                               } else if (vOp == "mul" && (isSplat(0) || isSplat(1))) {
                                 makeOp("vec_scalar",
                                        {local, isSplat(0) ? v1 : v0,
                                         isSplat(0) ? v0 : v1, cI64(elems)},
                                        {},
                                        {b.getNamedAttr("op",
                                                        b.getStringAttr("mul"))});
                               } else if (vOp == "mul" && (p0 != p1)) {
                                 makeOp("cycle_mul",
                                        {local, p0 ? v0 : v1, p0 ? v1 : v0,
                                         cI64(pack * K), cI64(K)},
                                        {}, {}, "cycle-mul");
                               } else {
                                 makeOp("vec_binary",
                                        {local, v0, v1, cI64(elems)}, {},
                                        {b.getNamedAttr("op",
                                                        b.getStringAttr(vOp))});
                               }
                             }
                             continue;
                           }
                           if (Value sv = lowerScalarExpr(bop, env))
                             env[bop.getResult(0)] = sv;
                         }

                         if (loopImpl == "single-carry") {
                           Value yv =
                               loop.getBody()->getTerminator()->getOperand(0);
                           int64_t elems =
                               (isPackedVal(loop.getResult(0)) ? pack : 1) * K;
                           memcpy1D(env.lookup(loop.getResult(0)),
                                    env.lookup(yv), cI64(elems), "NRAM2NRAM");
                         } else {
                           for (auto [yv, tmp] : llvm::zip(
                                    loop.getBody()->getTerminator()->getOperands(),
                                    carryTemps)) {
                             int64_t elems = (isPackedVal(yv) ? pack : 1) * K;
                             memcpy1D(tmp, env.lookup(yv), cI64(elems),
                                      "NRAM2NRAM");
                           }
                           for (auto [rv, tmp] :
                                llvm::zip(loop.getResults(), carryTemps)) {
                             int64_t elems = (isPackedVal(rv) ? pack : 1) * K;
                             memcpy1D(env.lookup(rv), tmp, cI64(elems),
                                      "NRAM2NRAM");
                           }
                         }
                       });
                     });

                     if (deferredBytes) {
                       Operation *defStore = nullptr;
                       for (auto &bop : *loop.getBody()) {
                         auto bTag =
                             bop.getAttrOfType<StringAttr>("bangc.vector_impl");
                         if (isa<memref::StoreOp>(bop) && bTag &&
                             bTag.getValue() == "deferred-row-reduction")
                           defStore = &bop;
                       }
                       Value outStream = streamPtrs.lookup(defStore);
                       sync("compute");
                       forLoop(cI64(0), cI64(steps), cI64(chunkSteps),
                               [&](Value w0) {
                                 Value curSteps = b.create<arith::MinSIOp>(
                                     loc,
                                     b.create<arith::SubIOp>(loc, cI64(steps), w0),
                                     cI64(chunkSteps));
                                 Value curRows = b.create<arith::MulIOp>(
                                     loc, curSteps, cI64(pack));
                                 Value srcChunk = makeAddr(deferred, {w0},
                                                           {cI64(pack * K)}, 1);
                                 Value dstChunk =
                                     makeAddr(outStream, {w0}, {cI64(pack)}, 1);
                                 makeOp("tile_transpose",
                                        {workBuf, srcChunk, curRows, cI64(K)},
                                        {}, {}, "nram-transpose");
                                 makeOp("sumpool",
                                        {dstChunk, workBuf, curRows, cI64(K)},
                                        {}, {}, "nram-sumpool");
                               });
                       sync("compute");
                     }
                     emitStreamTransfers(loop, true, streamPtrs);
                     continue;
                   }

                   if (sn == "slice_write") {
                     Value vec = env.lookup(op.getOperand(0));
                     Value view = op.getOperand(1);
                     int64_t sAxis =
                         op.getAttrOfType<IntegerAttr>("axis").getInt();
                     SmallVector<Value> addrC, addrS;
                     unsigned pos = 0;
                     for (Value cv : op.getOperands().drop_front(2)) {
                       if (int64_t(pos) == sAxis)
                         ++pos;
                       addrC.push_back(env.lookup(cv));
                       addrS.push_back(
                           cI64(dimAttr(view, "block.strides", pos)));
                       ++pos;
                     }
                     Value gPtr = makeAddr(view, addrC, addrS, 0);
                     int64_t kStride = dimAttr(view, "block.strides", sAxis);
                     sync("compute");
                     makeOp("tile_transpose", {workBuf, vec, cI64(pack), cI64(K)},
                            {}, {}, "nram-transpose");
                     sync("compute");
                     memcpy2D(gPtr, workBuf, cI64(pack), cI64(kStride),
                              cI64(pack), cI64(K - 1), "NRAM2GDRAM");
                     continue;
                   }

                   if (op.getNumResults() == 1 &&
                       isa<RankedTensorType>(op.getResult(0).getType())) {
                     Value resV = op.getResult(0);
                     int64_t elems = (isPackedVal(resV) ? pack : 1) * K;
                     Value local = nramPtr(
                         storage,
                         cI64(op.getAttrOfType<DenseI64ArrayAttr>(
                             "dsa.member_offsets")[0]),
                         b.getF32Type());
                     env[resV] = local;
                     if (sn == "lane_splat")
                       fillVec(local, cI64(elems), env.lookup(op.getOperand(0)));
                     continue;
                   }
                   if (Value sv = lowerScalarExpr(op, env))
                     env[op.getResult(0)] = sv;
                 }
               });
      });
    });
    return success();
  }

  // General single-owner lowering (direct, stage-slab, or 1-owner streams).
  struct GeneralState {
    Value storage, streambuf, deferred, slabLocal, poolScratch, scalarOut;
    int64_t slabBytes = 0;
    llvm::DenseMap<Value, Value> env;
    llvm::DenseMap<Operation *, Value> streams;
  };

  void emitGeneralStreamTransfers(scf::ForOp loop, bool publish,
                                  GeneralState &st) {
    for (auto &access : *loop.getBody()) {
      auto d = access.getAttrOfType<DenseI64ArrayAttr>("dsa.stream");
      if (!d)
        continue;
      auto l = dyn_cast<memref::LoadOp>(access);
      auto w = dyn_cast<memref::StoreOp>(access);
      bool sliced = access.getName().stripDialect() == "slice_read";
      Value view =
          l ? l.getMemref() : w ? w.getMemref() : access.getOperand(0);
      SmallVector<Value> coords;
      if (sliced) {
        llvm::append_range(coords, access.getOperands().drop_front());
        coords.insert(coords.begin() +
                          access.getAttrOfType<IntegerAttr>("axis").getInt(),
                      Value{});
      } else {
        llvm::append_range(coords, l ? l.getIndices() : w.getIndices());
      }
      Type elemTy = cast<MemRefType>(view.getType()).getElementType();
      if (!publish) {
        Value local = nramPtr(st.streambuf, cI64(d[0]), elemTy);
        st.streams[&access] = local;
      }
      if (bool(w) != publish)
        continue;
      Value local = st.streams.lookup(&access);
      auto accTag = access.getAttrOfType<StringAttr>("bangc.vector_impl");
      if (w && accTag && accTag.getValue() == "deferred-row-reduction") {
        int64_t k = memberExtent(w.getValue().getDefiningOp()->getOperand(0));
        Value transDst = makeAddr(st.deferred, {cI64(d[2] * k)}, {cI64(1)}, 1);
        sync("compute");
        makeOp("tile_transpose", {transDst, st.deferred, cI64(d[2]), cI64(k)},
               {}, {}, "nram-transpose");
        makeOp("sumpool", {local, transDst, cI64(d[2]), cI64(k)}, {}, {},
               "nram-sumpool");
        sync("compute");
      }
      SmallVector<Value> addrC, addrS;
      for (auto [ax, cv] : llvm::enumerate(coords)) {
        if (cv && int64_t(ax) != d[1]) {
          addrC.push_back(st.env.lookup(cv));
          addrS.push_back(cI64(dimAttr(view, "block.strides", ax)));
        }
      }
      Value gPtr = makeAddr(view, addrC, addrS, 0);
      int64_t width = d[3];
      int64_t stepStride = dimAttr(view, "block.strides", d[1]);
      if (width == stepStride) {
        memcpy1D(publish ? gPtr : local, publish ? local : gPtr,
                 cI64(d[2] * width), publish ? "NRAM2GDRAM" : "GDRAM2NRAM");
      } else {
        memcpy2D(publish ? gPtr : local, publish ? local : gPtr, cI64(width),
                 cI64(publish ? stepStride : width),
                 cI64(publish ? width : stepStride), cI64(d[2] - 1),
                 publish ? "NRAM2GDRAM" : "GDRAM2NRAM");
      }
      if (!publish)
        sync("io");
    }
  }

  std::pair<Value, Value> generalScalarPtr(Operation &op, Value view,
                                           ValueRange at, GeneralState &st) {
    if (auto d = op.getAttrOfType<DenseI64ArrayAttr>("dsa.stream"))
      return {st.streams.lookup(&op), st.env.lookup(at[d[1]])};
    SmallVector<Value> coords, strides;
    for (auto [a, idxV] : llvm::enumerate(at)) {
      if (!st.slabBytes || a != 0) {
        coords.push_back(st.env.lookup(idxV));
        strides.push_back(cI64(dimAttr(view, "block.strides", a)));
      }
    }
    Value base = st.slabBytes ? st.slabLocal : st.env.lookup(view);
    int64_t sp = st.slabBytes ? 1 : 0;
    return {makeAddr(base, coords, strides, sp), cI64(0)};
  }

  LogicalResult emitGeneralBlock(Block &blk, GeneralState &st) {
    for (auto &op : blk) {
      StringRef n = op.getName().getStringRef();
      StringRef kind = op.getName().stripDialect();
      if (kind == "end" || n == "scf.yield")
        continue;
      auto arg = [&](unsigned i) { return st.env.lookup(op.getOperand(i)); };
      auto implAttr = op.getAttrOfType<StringAttr>("bangc.vector_impl");
      StringRef impl = implAttr ? implAttr.getValue() : StringRef();

      if (auto loop = dyn_cast<scf::ForOp>(op)) {
        emitGeneralStreamTransfers(loop, false, st);
        unsigned pos = 0;
        auto offsets = op.getAttrOfType<DenseI64ArrayAttr>("dsa.member_offsets");
        SmallVector<Value> scalarInits;
        SmallVector<unsigned> scalarResIdx;
        SmallVector<std::pair<Value, Value>> tensorSlots; // {dst, tmp}
        SmallVector<int64_t> tensorLanes;

        for (auto [idx, tup] : llvm::enumerate(
                 llvm::zip(loop.getInitArgs(), loop.getRegionIterArgs(),
                           loop.getResults()))) {
          auto [initV, iterA, resV] = tup;
          if (isa<RankedTensorType>(resV.getType())) {
            Type et = cast<RankedTensorType>(resV.getType()).getElementType();
            int64_t lanes = memberExtent(resV);
            Value dst = nramPtr(st.storage, cI64(offsets[pos++]), et);
            Value tmp = nramPtr(st.storage, cI64(offsets[pos++]), et);
            memcpy1D(dst, st.env.lookup(initV), cI64(lanes), "NRAM2NRAM");
            st.env[iterA] = st.env[resV] = dst;
            tensorSlots.push_back({dst, tmp});
            tensorLanes.push_back(lanes);
          } else {
            Value initVal = st.env.lookup(initV);
            if (resV.getType().isIndex())
              initVal = toI64(initVal);
            scalarInits.push_back(initVal);
            scalarResIdx.push_back(idx);
          }
        }

        LogicalResult bodySt = success();
        auto termLoop = forLoopCarry(
            arg(0), arg(1), arg(2), scalarInits,
            [&](Value iv, ValueRange scIter) {
              st.env[loop.getInductionVar()] = iv;
              for (auto [si, iterVal] : llvm::zip(scalarResIdx, scIter))
                st.env[loop.getRegionIterArgs()[si]] = iterVal;
              if (failed(emitGeneralBlock(*loop.getBody(), st))) {
                bodySt = failure();
                return;
              }
              auto yieldOps = loop.getBody()->getTerminator()->getOperands();
              if (impl == "single-carry") {
                Value yv = yieldOps[0];
                memcpy1D(st.env.lookup(loop.getResult(0)), st.env.lookup(yv),
                         cI64(memberExtent(yv)), "NRAM2NRAM");
              } else {
                unsigned tIdx = 0;
                for (Value yv : yieldOps) {
                  if (isa<RankedTensorType>(yv.getType())) {
                    memcpy1D(tensorSlots[tIdx].second, st.env.lookup(yv),
                             cI64(tensorLanes[tIdx]), "NRAM2NRAM");
                    ++tIdx;
                  }
                }
                for (unsigned k = 0; k < tensorSlots.size(); ++k)
                  memcpy1D(tensorSlots[k].first, tensorSlots[k].second,
                           cI64(tensorLanes[k]), "NRAM2NRAM");
              }
              SmallVector<Value> scYields;
              for (unsigned si : scalarResIdx) {
                Value yv = st.env.lookup(yieldOps[si]);
                if (loop.getResult(si).getType().isIndex())
                  yv = toI64(yv);
                scYields.push_back(yv);
              }
              b.create<scf::YieldOp>(loc, scYields);
            });
        if (failed(bodySt))
          return failure();
        for (auto [si, resVal] : llvm::zip(scalarResIdx, termLoop.getResults()))
          st.env[loop.getResult(si)] = resVal;
        emitGeneralStreamTransfers(loop, true, st);
        continue;
      }

      if (auto iff = dyn_cast<scf::IfOp>(op)) {
        SmallVector<Type> scTypes;
        SmallVector<unsigned> scIdx;
        for (auto [idx, r] : llvm::enumerate(op.getResults())) {
          if (isa<RankedTensorType>(r.getType()))
            return op.emitError("unexpected tensor scf.if in ordered block");
          scTypes.push_back(r.getType().isIndex() ? Type(b.getI64Type())
                                                  : r.getType());
          scIdx.push_back(idx);
        }
        LogicalResult brSt = success();
        if (iff.getElseRegion().empty()) {
          ifThen(arg(0), [&]() {
            if (failed(emitGeneralBlock(*iff.thenBlock(), st)))
              brSt = failure();
          });
        } else {
          auto termIf = ifThenElse(
              arg(0), scTypes,
              [&]() {
                if (failed(emitGeneralBlock(*iff.thenBlock(), st))) {
                  brSt = failure();
                  return;
                }
                SmallVector<Value> ys;
                for (unsigned si : scIdx) {
                  Value yv =
                      st.env.lookup(iff.thenBlock()->getTerminator()->getOperand(si));
                  if (op.getResult(si).getType().isIndex())
                    yv = toI64(yv);
                  ys.push_back(yv);
                }
                b.create<scf::YieldOp>(loc, ys);
              },
              [&]() {
                if (failed(emitGeneralBlock(*iff.elseBlock(), st))) {
                  brSt = failure();
                  return;
                }
                SmallVector<Value> ys;
                for (unsigned si : scIdx) {
                  Value yv =
                      st.env.lookup(iff.elseBlock()->getTerminator()->getOperand(si));
                  if (op.getResult(si).getType().isIndex())
                    yv = toI64(yv);
                  ys.push_back(yv);
                }
                b.create<scf::YieldOp>(loc, ys);
              },
              false);
          for (auto [si, rv] : llvm::zip(scIdx, termIf.getResults()))
            st.env[op.getResult(si)] = rv;
        }
        if (failed(brSt))
          return failure();
        continue;
      }

      if (auto store = dyn_cast<memref::StoreOp>(op)) {
        if (impl != "deferred-row-reduction") {
          auto [ptr, idx] = generalScalarPtr(op, store.getMemref(),
                                             store.getIndices(), st);
          storeAt(ptr, idx, arg(0));
        }
        continue;
      }

      if (kind == "slice_write" ||
          (op.getNumResults() &&
           isa<RankedTensorType>(op.getResult(0).getType()))) {
        bool write = kind == "slice_write";
        Value v = write ? op.getOperand(0) : op.getResult(0);
        int64_t lanes = memberExtent(v);
        Type elemTy = cast<ShapedType>(v.getType()).getElementType();
        if (impl == "broadcast-multiplier") {
          st.env[v] = arg(0);
          continue;
        }
        if (impl == "stream-view") {
          Value iv =
              st.env.lookup(cast<scf::ForOp>(op.getParentOp()).getInductionVar());
          st.env[v] = makeAddr(st.streams.lookup(&op), {iv}, {cI64(lanes)}, 1);
          continue;
        }
        Value local;
        if (write) {
          local = arg(0);
        } else if (impl == "deferred-row-reduction") {
          Value iv =
              st.env.lookup(cast<scf::ForOp>(op.getParentOp()).getInductionVar());
          local = makeAddr(st.deferred, {iv}, {cI64(lanes)}, 1);
        } else {
          int64_t off =
              op.getAttrOfType<DenseI64ArrayAttr>("dsa.member_offsets")[0];
          local = nramPtr(st.storage, cI64(off), elemTy);
        }
        if (!write)
          st.env[v] = local;

        if (kind == "slice_read" || write) {
          Value base = op.getOperand(write ? 1 : 0);
          int64_t axis = op.getAttrOfType<IntegerAttr>("axis").getInt();
          SmallVector<Value> addrC, addrS;
          for (auto [i, c] :
               llvm::enumerate(op.getOperands().drop_front(write ? 2 : 1))) {
            int64_t realAx = i + (int64_t(i) >= axis);
            addrC.push_back(st.env.lookup(c));
            addrS.push_back(cI64(dimAttr(base, "block.strides", realAx)));
          }
          Value gPtr = makeAddr(st.env.lookup(base), addrC, addrS, 0);
          int64_t step = dimAttr(base, "block.strides", axis);
          if (op.hasAttr("dsa.stream")) {
            Value iv = st.env.lookup(
                cast<scf::ForOp>(op.getParentOp()).getInductionVar());
            memcpy1D(local, makeAddr(st.streams.lookup(&op), {iv}, {cI64(lanes)}, 1),
                     cI64(lanes), "NRAM2NRAM");
          } else if (step == 1) {
            if (write)
              sync("compute");
            memcpy1D(write ? gPtr : local, write ? local : gPtr, cI64(lanes),
                     write ? "NRAM2GDRAM" : "GDRAM2NRAM");
            if (!write)
              sync("io");
          } else {
            if (write)
              sync("compute");
            forLoop(cI64(0), cI64(lanes), cI64(1), [&](Value j) {
              Value elemG = makeAddr(gPtr, {j}, {cI64(step)}, 0);
              if (write)
                storeAt(elemG, cI64(0), loadAt(local, j));
              else
                storeAt(local, j, loadAt(elemG, cI64(0)));
            });
          }
        } else if (kind == "lane_splat") {
          fillVec(local, cI64(lanes), arg(0));
        } else if (impl == "widen-bits") {
          makeOp("bf16_tile", {local, arg(0), cI64(lanes), cI64(lanes)}, {},
                 {b.getNamedAttr("mode", b.getStringAttr("widen"))},
                 "bf16-tile-convert");
        } else if (impl == "widen-half" || impl == "narrow-half") {
          makeOp("vec_cast", {local, arg(0), cI64(lanes)}, {},
                 {b.getNamedAttr(
                     "kind", b.getStringAttr(impl == "widen-half"
                                                 ? "f16_to_f32"
                                                 : "f32_to_f16"))});
        } else if (n == "arith.extf" || n == "arith.truncf") {
          Type from = cast<ShapedType>(op.getOperand(0).getType()).getElementType();
          if (from.isBF16() && elemTy.isF32() && lanes % 32 == 0) {
            makeOp("bf16_tile", {local, arg(0), cI64(lanes), cI64(lanes)}, {},
                   {b.getNamedAttr("mode", b.getStringAttr("widen"))},
                   "bf16-tile-convert");
          } else if (from.isF32() && elemTy.isBF16() && lanes % 32 == 0) {
            makeOp("bf16_tile", {local, arg(0), cI64(lanes), cI64(lanes)}, {},
                   {b.getNamedAttr("mode", b.getStringAttr("narrow"))},
                   "bf16-tile-convert");
          } else {
            sync("compute");
            forLoop(cI64(0), cI64(lanes), cI64(1), [&](Value j) {
              Value vIn = loadAt(arg(0), j);
              Value vOut;
              if (from.isBF16() && elemTy.isF32())
                vOut = makeOp("bf16_scalar", {vIn}, {b.getF32Type()},
                              {b.getNamedAttr("mode", b.getStringAttr("widen"))},
                              "bf16-scalar-convert")
                           ->getResult(0);
              else if (from.isF32() && elemTy.isBF16())
                vOut = makeOp("bf16_scalar", {vIn}, {b.getBF16Type()},
                              {b.getNamedAttr("mode", b.getStringAttr("narrow"))},
                              "bf16-scalar-convert")
                           ->getResult(0);
              else if (n == "arith.extf")
                vOut = b.create<arith::ExtFOp>(loc, elemTy, vIn);
              else
                vOut = b.create<arith::TruncFOp>(loc, elemTy, vIn);
              storeAt(local, j, vOut);
            });
          }
        } else {
          StringRef vOp = n == "arith.addf"   ? "add"
                          : n == "arith.subf" ? "sub"
                          : n == "arith.mulf" ? "mul"
                          : n == "arith.divf" ? "div"
                                              : "";
          if (vOp.empty())
            return op.emitError("unsupported lane instruction ") << n;
          auto isSplat = [&](unsigned idx) {
            auto *def = op.getOperand(idx).getDefiningOp();
            auto tag =
                def ? def->getAttrOfType<StringAttr>("bangc.vector_impl")
                    : StringAttr();
            return tag && tag.getValue() == "broadcast-multiplier";
          };
          if (vOp == "mul" && isSplat(0) && isSplat(1)) {
            fillVec(local, cI64(lanes),
                    b.create<arith::MulFOp>(loc, arg(0), arg(1)));
          } else if (vOp == "mul" && (isSplat(0) || isSplat(1)) &&
                     elemTy.isF32() && lanes % 32 == 0) {
            makeOp("vec_scalar",
                   {local, arg(isSplat(0) ? 1 : 0), arg(isSplat(0) ? 0 : 1),
                    cI64(lanes)},
                   {}, {b.getNamedAttr("op", b.getStringAttr("mul"))});
          } else if (elemTy.isF32() && lanes % 32 == 0 && vOp != "div") {
            makeOp("vec_binary", {local, arg(0), arg(1), cI64(lanes)}, {},
                   {b.getNamedAttr("op", b.getStringAttr(vOp))});
          } else {
            sync("compute");
            forLoop(cI64(0), cI64(lanes), cI64(1), [&](Value j) {
              Value l = loadAt(arg(0), j), r = loadAt(arg(1), j);
              Value res = vOp == "add"
                              ? b.create<arith::AddFOp>(loc, l, r).getResult()
                          : vOp == "sub"
                              ? b.create<arith::SubFOp>(loc, l, r).getResult()
                          : vOp == "mul"
                              ? b.create<arith::MulFOp>(loc, l, r).getResult()
                              : b.create<arith::DivFOp>(loc, l, r).getResult();
              storeAt(local, j, res);
            });
          }
        }
        continue;
      }

      if (kind == "lane_sum") {
        if (impl == "deferred-row-reduction")
          continue;
        int64_t lanes = memberExtent(op.getOperand(0));
        if (impl == "reduce32-reuse-product") {
          st.env[op.getResult(0)] =
              makeOp("reduce32_sum", {arg(0)}, {b.getF32Type()}, {},
                     "reduce32-sum")
                  ->getResult(0);
        } else {
          sync("compute");
          makeOp("row_reduce",
                 {st.scalarOut, st.poolScratch, arg(0), cI64(1), cI64(lanes),
                  cI64(lanes)},
                 {}, {b.getNamedAttr("algebra", b.getStringAttr("sum"))},
                 "row-reduce");
          st.env[op.getResult(0)] = loadAt(st.scalarOut, cI64(0));
        }
        continue;
      }

      if (auto ld = dyn_cast<memref::LoadOp>(op)) {
        auto [ptr, idx] =
            generalScalarPtr(op, ld.getMemref(), ld.getIndices(), st);
        st.env[op.getResult(0)] = loadAt(ptr, idx, op.getResult(0).getType());
        continue;
      }

      Value sv = lowerScalarExpr(op, st.env);
      if (!sv)
        return op.emitError("unhandled ordered BANG instruction ") << n;
      st.env[op.getResult(0)] = sv;
    }
    return success();
  }

  LogicalResult lowerGeneral() {
    auto &bb = space.getRegion(0).front();
    int64_t tile = space.getAttrOfType<IntegerAttr>("block.owner_tile").getInt();
    int64_t owners = space.getAttrOfType<IntegerAttr>("block.owners").getInt();
    int64_t slabBytes =
        space.getAttrOfType<IntegerAttr>("dsa.owner_bytes").getInt();
    int64_t memberBytes =
        space.getAttrOfType<IntegerAttr>("dsa.member_bytes").getInt();
    auto streamAttr = space.getAttrOfType<IntegerAttr>("dsa.stream_bytes");
    auto defAttr = space.getAttrOfType<IntegerAttr>("bangc.deferred_bytes");

    GeneralState st;
    st.slabBytes = slabBytes;
    bool needsReduce = false;
    space.walk([&](Operation *op) {
      if (op->getName().stripDialect() == "lane_sum") {
        auto im = op->getAttrOfType<StringAttr>("bangc.vector_impl");
        if (!im || (im != "reduce32-reuse-product" &&
                    im != "deferred-row-reduction"))
          needsReduce = true;
      }
    });
    if (needsReduce) {
      st.poolScratch = nramBuf(b.getF32Type(), 32);
      st.scalarOut = nramBuf(b.getF32Type(), 32);
    }
    if (memberBytes)
      st.storage = nramBuf(b.getI8Type(), memberBytes);
    if (streamAttr && streamAttr.getInt())
      st.streambuf = nramBuf(b.getI8Type(), streamAttr.getInt());
    if (defAttr && defAttr.getInt())
      st.deferred = nramBuf(b.getF32Type(), defAttr.getInt() / 4);
    if (slabBytes) {
      Type slabElem =
          cast<MemRefType>(fn.getArgument(0).getType()).getElementType();
      int64_t elemB = slabElem.getIntOrFloatBitWidth() / 8;
      st.slabLocal = nramBuf(slabElem, slabBytes / elemB);
    }

    auto axes = space.getAttrOfType<DenseI64ArrayAttr>("axes");
    LogicalResult status = success();
    forLoop(symTaskId, symGrid, symTaskDim, [&](Value job) {
      Value baseOwner = b.create<arith::MulIOp>(loc, job, cI64(tile));
      Value endOwner = b.create<arith::MinSIOp>(
          loc, cI64(owners), b.create<arith::AddIOp>(loc, baseOwner, cI64(tile)));
      forLoop(baseOwner, endOwner, cI64(1), [&](Value owner) {
        st.env.clear();
        st.streams.clear();
        for (Value arg : fn.getArguments())
          st.env[arg] = arg;
        int64_t pitch = 1;
        for (int i = axes.size() - 1; i >= 0; --i) {
          int64_t ext = std::max<int64_t>(
              1, dimAttr(space.getOperand(i), "block.shape", axes[i]));
          Value coord = b.create<arith::RemSIOp>(
              loc, b.create<arith::DivSIOp>(loc, owner, cI64(pitch)), cI64(ext));
          st.env[bb.getArgument(i)] = coord;
          pitch *= ext;
        }
        Value slabGlobal;
        int64_t slabElems = 0;
        if (slabBytes) {
          Value a0 = fn.getArgument(0);
          Type slabElem = cast<MemRefType>(a0.getType()).getElementType();
          slabElems = slabBytes / (slabElem.getIntOrFloatBitWidth() / 8);
          slabGlobal = makeAddr(
              a0, {owner}, {cI64(dimAttr(a0, "block.strides", 0))}, 0);
          memcpy1D(st.slabLocal, slabGlobal, cI64(slabElems), "GDRAM2NRAM");
          sync("io");
        }
        if (failed(emitGeneralBlock(bb, st)))
          status = failure();
        if (slabBytes) {
          sync("compute");
          memcpy1D(slabGlobal, st.slabLocal, cI64(slabElems), "NRAM2GDRAM");
        }
      });
    });
    return status;
  }

  LogicalResult run(ModuleOp m) {
    int64_t tile = space.getAttrOfType<IntegerAttr>("block.owner_tile").getInt();
    int64_t owners = space.getAttrOfType<IntegerAttr>("block.owners").getInt();
    int64_t slabBytes =
        space.getAttrOfType<IntegerAttr>("dsa.owner_bytes").getInt();
    int64_t workers =
        space.getAttrOfType<IntegerAttr>("bangc.workers").getInt();
    StringRef transfers =
        space.getAttrOfType<StringAttr>("dsa.transfers").getValue();
    auto packAttr = space.getAttrOfType<IntegerAttr>("dsa.owner_pack");
    int64_t pack = packAttr ? packAttr.getInt() : 1;
    int64_t panelM =
        transfers == "panel-fold"
            ? space.getAttrOfType<IntegerAttr>("dsa.panel_rows").getInt()
            : 1;
    int64_t launchTile = std::max<int64_t>({tile, pack, panelM});

    b.setInsertionPointToStart(&fn.front());
    OperationState progState(loc, "bangc_src.ordered_program");
    progState.addAttribute("owner_tile", b.getI64IntegerAttr(launchTile));
    progState.addAttribute("owners", b.getI64IntegerAttr(owners));
    progState.addAttribute("participants", b.getI64IntegerAttr(workers));
    progState.addAttribute("owner_bytes", b.getI64IntegerAttr(slabBytes));
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

    symGrid = makeSym("grid");
    symTaskId = makeSym("taskId");
    symTaskDim = makeSym("taskDim");
    constAnchor = symTaskDim.getDefiningOp();

    LogicalResult st = transfers == "panel-fold" ? lowerPanelFold()
                       : pack > 1                ? lowerPackedStreams(pack)
                                                 : lowerGeneral();
    if (failed(st))
      return failure();

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

    auto abiJson =
        llvm::json::parse(fn->getAttrOfType<StringAttr>("kir.abi").getValue());
    if (!abiJson)
      return llvm::consumeError(abiJson.takeError()), failure();

    space.erase();

    llvm::SmallVector<char> metadata;
    llvm::raw_svector_ostream os(metadata);
    os << llvm::json::Value(llvm::json::Object{
        {"version", 2},
        {"source", ""},
        {"abi", std::move(*abiJson)},
        {"domain", intentdsl::publishDomain(fn)},
        {"grid",
         llvm::json::Array{int64_t(llvm::divideCeil(owners, launchTile)), 1,
                           1}},
        {"guard", llvm::json::Object{{"min_width", 0}, {"max_width", nullptr}}},
        {"rows", llvm::json::Array{llvm::json::Object{{"id", 0},
                                                      {"rows", 1},
                                                      {"width", 1},
                                                      {"pitch", 1},
                                                      {"bytes", slabBytes},
                                                      {"tasks", workers}}}}});
    m->setAttr("bangc_src.package",
               b.getStringAttr(StringRef(metadata.data(), metadata.size())));
    return success();
  }
};
} // namespace

LogicalResult lowerOrderedToCallable(ModuleOp m, const TargetProfile &profile) {
  auto *ctx = m.getContext();
  ctx->getOrLoadDialect<intentdsl::bangc_src::BangcSrcDialect>();
  ctx->getOrLoadDialect<scf::SCFDialect>();
  ctx->getOrLoadDialect<arith::ArithDialect>();
  auto fn = *m.getOps<func::FuncOp>().begin();
  auto supply = profile.services.get<intentdsl::dsa::LocalSupply>();
  if (!supply)
    return failure();
  auto &space = fn.front().front();
  space.setAttr("bangc.workers",
                Builder(ctx).getI64IntegerAttr(supply->consumers));
  space.setAttr(
      "bangc.owner_memory",
      StringAttr::get(
          ctx, space.getAttrOfType<IntegerAttr>("dsa.owner_bytes").getInt()
                   ? "private-nram"
                   : "global"));
  if (failed(ownerVectors(fn, true,
                          std::min(supply->budget, supply->capacity))))
    return failure();
  fn->setAttr("bangc.entry", Builder(ctx).getStringAttr("intent_enqueue"));
  OrderedLowerer lowerer(fn, profile);
  return lowerer.run(m);
}
} // namespace CI_UNIT_NAMESPACE
