#include "OwnerVectors.h"
#include "Rows.h"
#include "intentdsl/exits/bangc/ir/Dialect.h"
#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "intentdsl/levels/storage/dsa/ir/Owners.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/JSON.h"
#include <cmath>
#include <limits>
#include <set>
#include <tuple>

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
  struct PendingOrderedTailGuard {
    Operation *guardIf;
    Operation *accessOp;
  };
  SmallVector<PendingOrderedTailGuard, 8> orderedTailGuards;

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

  Value floorDivSI(Value lhs, Value rhs) {
    Type ty = lhs.getType();
    Value zero = ty.isInteger(32) ? cI32(0) : cI64(0);
    Value one = ty.isInteger(32) ? cI32(1) : cI64(1);
    Value q = b.create<arith::DivSIOp>(loc, lhs, rhs);
    Value r = b.create<arith::RemSIOp>(loc, lhs, rhs);
    Value nz = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, r, zero);
    Value diffSign = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, b.create<arith::XOrIOp>(loc, r, rhs),
        zero);
    Value adjust = b.create<arith::AndIOp>(loc, nz, diffSign);
    return b.create<arith::SelectOp>(
        loc, adjust, b.create<arith::SubIOp>(loc, q, one), q);
  }

  Operation *makeOp(StringRef name, ValueRange operands, TypeRange results = {},
                    ArrayRef<NamedAttribute> attrs = {},
                    StringRef contract = {}) {
    OperationState st(loc, ("bangc_src." + name).str());
    st.addOperands(operands);
    st.addTypes(results);
    st.addAttributes(attrs);
    if (!contract.empty()) {
      st.addAttribute("ci.implementation",
                      b.getStringAttr(("exits/bangc::" + contract).str()));
      if (llvm::any_of(operands,
                       [](Value v) { return isa<MemRefType>(v.getType()); }) &&
          llvm::none_of(attrs, [](const NamedAttribute &na) {
            return na.getName() == "alignment";
          }))
        st.addAttribute("alignment", b.getI64IntegerAttr(64));
    }
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

  scf::IfOp ifThen(Value cond, llvm::function_ref<void()> thenFn) {
    return b.create<scf::IfOp>(loc, cond, [&](OpBuilder &nb, Location l) {
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
      Value v64 = toI64(a(0));
      if (op.getResult(0).getType().isSignlessInteger(64))
        return v64;
      return b.create<arith::TruncIOp>(loc, op.getResult(0).getType(), v64);
    }
    if (n == "arith.index_castui") {
      Value u64 = a(0).getType().isSignlessInteger(64)
                      ? a(0)
                      : a(0).getType().isIndex()
                            ? b.create<arith::IndexCastUIOp>(loc, b.getI64Type(),
                                                             a(0))
                                  .getResult()
                            : b.create<arith::ExtUIOp>(loc, b.getI64Type(), a(0))
                                  .getResult();
      if (op.getResult(0).getType().isIndex() ||
          op.getResult(0).getType().isSignlessInteger(64))
        return u64;
      return b.create<arith::TruncIOp>(loc, op.getResult(0).getType(), u64);
    }
    if (n == "arith.extsi")
      return b.create<arith::ExtSIOp>(loc, op.getResult(0).getType(), a(0));
    if (n == "arith.extui")
      return b.create<arith::ExtUIOp>(loc, op.getResult(0).getType(), a(0));
    if (n == "arith.trunci")
      return b.create<arith::TruncIOp>(loc, op.getResult(0).getType(), a(0));
    if (n == "arith.sitofp" || n == "arith.uitofp") {
      Type to = op.getResult(0).getType();
      Type mid = to.isBF16() ? Type(b.getF32Type()) : to;
      Value fp = n == "arith.sitofp"
                     ? b.create<arith::SIToFPOp>(loc, mid, a(0)).getResult()
                     : b.create<arith::UIToFPOp>(loc, mid, a(0)).getResult();
      if (to.isBF16())
        return makeOp("bf16_scalar", {fp}, {b.getBF16Type()},
                      {b.getNamedAttr("mode", b.getStringAttr("narrow"))},
                      "bf16-scalar-convert")
            ->getResult(0);
      return fp;
    }
    if (n == "arith.fptosi" || n == "arith.fptoui") {
      Value src = a(0);
      if (src.getType().isBF16())
        src = makeOp("bf16_scalar", {src}, {b.getF32Type()},
                     {b.getNamedAttr("mode", b.getStringAttr("widen"))},
                     "bf16-scalar-convert")
                  ->getResult(0);
      Type to = op.getResult(0).getType().isIndex() ? Type(b.getI64Type())
                                                    : op.getResult(0).getType();
      return n == "arith.fptosi"
                 ? b.create<arith::FPToSIOp>(loc, to, src).getResult()
                 : b.create<arith::FPToUIOp>(loc, to, src).getResult();
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
    if (n == "arith.minnumf")
      return b.create<arith::MinNumFOp>(loc, a(0), a(1));
    if (n == "math.sqrt" || n == "math.exp" || n == "math.exp2" ||
        n == "math.log" || n == "math.absf" || n == "math.floor") {
      StringRef mOp = kind == "absf" ? "fabs" : kind;
      return makeOp("scalar_math", {a(0)}, {b.getF32Type()},
                    {b.getNamedAttr("op", b.getStringAttr(mOp))})
          ->getResult(0);
    }
    if (n == "arith.addi")
      return b.create<arith::AddIOp>(loc, a(0), a(1));
    if (n == "arith.subi")
      return b.create<arith::SubIOp>(loc, a(0), a(1));
    if (n == "arith.muli")
      return b.create<arith::MulIOp>(loc, a(0), a(1));
    if (n == "arith.divsi")
      return b.create<arith::DivSIOp>(loc, a(0), a(1));
    if (n == "arith.divui")
      return b.create<arith::DivUIOp>(loc, a(0), a(1));
    if (n == "arith.floordivsi")
      return floorDivSI(a(0), a(1));
    if (n == "arith.remsi")
      return b.create<arith::RemSIOp>(loc, a(0), a(1));
    if (n == "arith.remui")
      return b.create<arith::RemUIOp>(loc, a(0), a(1));
    if (n == "arith.minsi")
      return b.create<arith::MinSIOp>(loc, a(0), a(1));
    if (n == "arith.maxsi")
      return b.create<arith::MaxSIOp>(loc, a(0), a(1));
    if (n == "arith.minui")
      return b.create<arith::MinUIOp>(loc, a(0), a(1));
    if (n == "arith.maxui")
      return b.create<arith::MaxUIOp>(loc, a(0), a(1));
    if (n == "arith.andi")
      return b.create<arith::AndIOp>(loc, a(0), a(1));
    if (n == "arith.ori")
      return b.create<arith::OrIOp>(loc, a(0), a(1));
    if (n == "arith.xori")
      return b.create<arith::XOrIOp>(loc, a(0), a(1));
    if (n == "arith.shli")
      return b.create<arith::ShLIOp>(loc, a(0), a(1));
    if (n == "arith.shrsi")
      return b.create<arith::ShRSIOp>(loc, a(0), a(1));
    if (n == "arith.shrui")
      return b.create<arith::ShRUIOp>(loc, a(0), a(1));
    if (n == "arith.addf")
      return b.create<arith::AddFOp>(loc, a(0), a(1));
    if (n == "arith.subf")
      return b.create<arith::SubFOp>(loc, a(0), a(1));
    if (n == "arith.mulf")
      return b.create<arith::MulFOp>(loc, a(0), a(1));
    if (n == "arith.divf")
      return b.create<arith::DivFOp>(loc, a(0), a(1));
    if (n == "arith.negf") {
      Value z = b.create<arith::ConstantOp>(
          loc, b.getZeroAttr(op.getResult(0).getType()));
      return b.create<arith::SubFOp>(loc, z, a(0));
    }
    if (auto cmp = dyn_cast<arith::CmpIOp>(op))
      return b.create<arith::CmpIOp>(loc, cmp.getPredicate(), a(0), a(1));
    if (auto cmpf = dyn_cast<arith::CmpFOp>(op)) {
      Value lhs = a(0), rhs = a(1);
      if (lhs.getType().isBF16()) {
        lhs = makeOp("bf16_scalar", {lhs}, {b.getF32Type()},
                     {b.getNamedAttr("mode", b.getStringAttr("widen"))},
                     "bf16-scalar-convert")
                  ->getResult(0);
        rhs = makeOp("bf16_scalar", {rhs}, {b.getF32Type()},
                     {b.getNamedAttr("mode", b.getStringAttr("widen"))},
                     "bf16-scalar-convert")
                  ->getResult(0);
      }
      return b.create<arith::CmpFOp>(loc, cmpf.getPredicate(), lhs, rhs);
    }
    if (auto sel = dyn_cast<arith::SelectOp>(op))
      return b.create<arith::SelectOp>(loc, a(0), a(1), a(2));
    return Value{};
  }

  // Lower cooperative panel fold ("panel-fold") op-by-op from DSA-assigned
  // per-value roles and live-interval NRAM offsets (no whole-algorithm matching).
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

    Type halfTy;
    space.walk([&](Operation *op) {
      if (halfTy)
        return;
      StringRef sn = op->getName().stripDialect();
      if (sn == "slice_read" || sn == "slice_write") {
        Value view = op->getOperand(sn == "slice_write" ? 1 : 0);
        if (auto mt = dyn_cast<MemRefType>(view.getType())) {
          Type et = mt.getElementType();
          if (et.isBF16() || et.isF16())
            halfTy = et;
        }
      }
    });
    if (!halfTy || !scratchOffs || scratchOffs.size() < 3)
      return space.emitError("missing DSA panel-fold element type or scratch");
    int64_t halfB = halfTy.getIntOrFloatBitWidth() / 8;

    Value storage = nramBuf(b.getI8Type(), memberBytes);
    Value filter = wramBuf(halfTy, wramBytes / halfB);

    auto slotAt = [&](Operation *op, unsigned idx, Type elemTy) {
      int64_t off =
          op->getAttrOfType<DenseI64ArrayAttr>("dsa.member_offsets")[idx];
      return nramPtr(storage, cI64(off), elemTy);
    };

    Value magScratch = nramPtr(storage, cI64(scratchOffs[0]), b.getBF16Type());
    Value flagScratch = nramPtr(storage, cI64(scratchOffs[1]), b.getBF16Type());
    Value scalarOutBuf = nramPtr(storage, cI64(scratchOffs[2]), b.getF32Type());
    Value reduceScratch =
        nramPtr(storage, cI64(scratchOffs[2] + 128), b.getF32Type());
    Value loBuf =
        nramPtr(storage, cI64(scratchOffs[2] + 512), b.getI64Type());
    Value hiBuf =
        nramPtr(storage, cI64(scratchOffs[2] + 1024), b.getI64Type());
    Value convTmp =
        nramPtr(storage, cI64(scratchOffs[2] + 1536), b.getF32Type());

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
                         int64_t maxRows, int64_t cols, bool zeroTail)
        -> Operation * {
      int64_t rank = cast<MemRefType>(view.getType()).getRank();
      int64_t rowStride = dimAttr(view, "block.strides", rank - 2);
      Operation *tailIf = nullptr;
      if (zeroTail) {
        tailIf = ifThen(
                     b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                             nRows, cI64(maxRows)),
                     [&]() {
                       fillVec(dstNram, cI64(maxRows * cols),
                               b.create<arith::ConstantOp>(
                                   loc, b.getZeroAttr(halfTy)));
                     })
                     .getOperation();
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
      return tailIf;
    };

    auto emitPanelProduct = [&](Value outF32, Value lhsHalf,
                                Value rhsFilterMajor, int64_t Kdim,
                                int64_t Cdim, bool accumulate) {
      auto runWram = [&](bool guardedWindow) {
        StringRef mKind = halfTy.isBF16() ? "bf16" : "f16";
        StringRef cKey = halfTy.isBF16() ? "wram-conv-bf16" : "wram-conv-f16";
        auto makeConvAttrs = [&](int64_t bmVal, int64_t bnVal, int64_t bkVal) {
          SmallVector<NamedAttribute, 5> attrs{
              b.getNamedAttr("bm", b.getI64IntegerAttr(bmVal)),
              b.getNamedAttr("bn", b.getI64IntegerAttr(bnVal)),
              b.getNamedAttr("bk", b.getI64IntegerAttr(bkVal)),
              b.getNamedAttr("mode", b.getStringAttr(mKind))};
          if (guardedWindow && halfTy.isBF16())
            attrs.push_back(b.getNamedAttr("bangc.guarded_bf16_window",
                                           b.getBoolAttr(true)));
          return attrs;
        };
        if (Cdim == 64) {
          sync("compute");
          memcpy1D(filter, rhsFilterMajor, cI64(64 * Kdim), "NRAM2WRAM");
          makeOp("wram_conv", {outF32, lhsHalf, filter, cI1(accumulate)}, {},
                 makeConvAttrs(M, 64, Kdim), cKey);
          sync("compute");
        } else if (Cdim == 32) {
          Value padNram =
              nramPtr(storage, cI64(scratchOffs[2] + 1536), halfTy);
          fillVec(padNram, cI64(64 * Kdim),
                  b.create<arith::ConstantOp>(loc, b.getZeroAttr(halfTy)));
          memcpy1D(padNram, rhsFilterMajor, cI64(32 * Kdim), "NRAM2NRAM");
          sync("io");
          memcpy1D(filter, padNram, cI64(64 * Kdim), "NRAM2WRAM");
          makeOp("wram_conv", {convTmp, lhsHalf, filter, cI1(false)}, {},
                 makeConvAttrs(M, 64, Kdim), cKey);
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
                   makeConvAttrs(M, 64, Kdim), cKey);
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

      if (halfTy.isBF16() &&
          !compilerinfra::allowsSubnormalFlush(fn, profile.numerical)) {
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
        ifThenElse(
            bothOk, {}, [&]() { runWram(/*guardedWindow=*/true); },
            [&]() {
              if (!accumulate)
                fillVec(outF32, cI64(M * Cdim), cF32(0.0f));
              sync("compute");
              makeOp("bf16_ieee_dot",
                     {outF32, lhsHalf, rhsFilterMajor, cI64(M), cI64(Cdim),
                      cI64(Kdim), cI64(Kdim), cI64(Cdim)},
                     {}, {}, "bf16-ieee-dot");
            });
      } else {
        runWram(/*guardedWindow=*/false);
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

    auto widenPanel = [&](Value dstF32, Value srcHalf, int64_t count) {
      if (halfTy.isBF16())
        makeOp("bf16_tile", {dstF32, srcHalf, cI64(count), cI64(count)}, {},
               {b.getNamedAttr("mode", b.getStringAttr("widen"))},
               "bf16-tile-convert");
      else
        makeOp("vec_cast", {dstF32, srcHalf, cI64(count)}, {},
               {b.getNamedAttr("kind", b.getStringAttr("f16_to_f32"))});
    };

    auto isFloatConstPred = [](Value v,
                               llvm::function_ref<bool(const APFloat &)> pred) {
      auto c = v ? v.getDefiningOp<arith::ConstantOp>() : arith::ConstantOp{};
      auto fa = c ? dyn_cast<FloatAttr>(c.getValue()) : FloatAttr{};
      return fa && fa.getType().isF32() && pred(fa.getValue());
    };
    auto isNegInfVal = [&](Value v) {
      return isFloatConstPred(
          v, [](const APFloat &x) { return x.isInfinity() && x.isNegative(); });
    };
    auto isZeroVal = [&](Value v) {
      return isFloatConstPred(v, [](const APFloat &x) { return x.isZero(); });
    };

    Value rowArg = bb.getArguments().back();
    int64_t stepTile = std::max<int64_t>(tile, M);
    auto axes = space.getAttrOfType<DenseI64ArrayAttr>("axes");

    forLoop(symTaskId, symGrid, symTaskDim, [&](Value job) {
      forLoop(cI64(0), cI64(stepTile), cI64(M), [&](Value sub) {
        Value ownerBase = b.create<arith::AddIOp>(
            loc, b.create<arith::MulIOp>(loc, job, cI64(stepTile)), sub);
        ifThen(
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, ownerBase,
                                    cI64(owners)),
            [&]() {
              llvm::DenseMap<Value, Value> scalarEnv;
              llvm::DenseMap<Value, Value> slotEnv;
              llvm::DenseMap<Value, int64_t> panelFeat;
              llvm::DenseSet<Value> ownerScalars;
              llvm::DenseSet<Value> causalMaskedStrips;
              llvm::DenseSet<Value> windowZeroedStrips;
              bool curLoopHasWindow = false;
              Value curStripT0;

              for (Value arg : fn.getArguments())
                scalarEnv[arg] = arg;
              int64_t pitch = 1;
              for (int i = axes.size() - 1; i >= 0; --i) {
                int64_t ext = std::max<int64_t>(
                    1, dimAttr(space.getOperand(i), "block.shape", axes[i]));
                Value coord = b.create<arith::RemSIOp>(
                    loc, b.create<arith::DivSIOp>(loc, ownerBase, cI64(pitch)),
                    cI64(ext));
                if (size_t(i) + 1 < bb.getNumArguments())
                  scalarEnv[bb.getArgument(i)] = coord;
                else
                  scalarEnv[rowArg] = coord;
                pitch *= ext;
              }
              Value rowBase = scalarEnv.lookup(rowArg);
              scalarEnv.erase(rowArg);

              auto unwrapBridge = [](Value v) -> Value {
                while (Operation *d = v.getDefiningOp()) {
                  if (d->hasAttr("dsa.panel_role"))
                    break;
                  StringRef sn = d->getName().stripDialect();
                  if (d->getName().getStringRef() == "arith.extf" ||
                      sn == "lane_splat") {
                    v = d->getOperand(0);
                    continue;
                  }
                  break;
                }
                return v;
              };

              auto evalScalarBlockForRow =
                  [&](Block &subBlk, Value r,
                      const llvm::DenseMap<Value, Value> &baseScalars) -> Value {
                llvm::DenseMap<Value, Value> rowEnv = baseScalars;
                rowEnv[rowArg] = b.create<arith::AddIOp>(loc, rowBase, r);
                for (Value os : ownerScalars)
                  rowEnv[os] = loadAt(slotEnv.lookup(os), r);
                for (Operation &inOp : subBlk.without_terminator())
                  if (Value sv = lowerScalarExpr(inOp, rowEnv))
                    rowEnv[inOp.getResult(0)] = sv;
                return rowEnv.lookup(subBlk.getTerminator()->getOperand(0));
              };

              auto emitStripElem = [&](Operation &eop, Value dstSlot,
                                       const llvm::DenseMap<Value, Value> &lEnv,
                                       Value singleRow, Value activeCount) {
                StringRef en = eop.getName().getStringRef();
                auto rowSlice = [&](Value panel, Value r) {
                  return makeAddr(panel, {r}, {cI64(S)}, 1);
                };
                if (en == "math.exp" || en == "math.exp2") {
                  bool isExp2 = en == "math.exp2";
                  Value srcSlot = slotEnv.lookup(eop.getOperand(0));
                  auto emitOne = [&](Value r, Value cnt) {
                    ifThen(b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                                   cnt, cI64(0)),
                           [&]() {
                             Value dR = rowSlice(dstSlot, r);
                             Value sR = rowSlice(srcSlot, r);
                             makeOp(isExp2 ? "row_exp2" : "row_exp",
                                    {dR, reduceScratch, sR, cnt, cI64(S)}, {},
                                    {},
                                    isExp2 ? "validated-exp2" : "validated-exp");
                           });
                  };
                  if (singleRow) {
                    emitOne(singleRow, activeCount);
                  } else if (curLoopHasWindow) {
                    forLoop(cI64(0), cI64(M), cI64(1), [&](Value r) {
                      Value rLo = b.create<arith::MinSIOp>(
                          loc, activeCount,
                          b.create<arith::MaxSIOp>(
                              loc, cI64(0),
                              b.create<arith::SubIOp>(
                                  loc, loadAt(loBuf, r), curStripT0)));
                      Value rHi = b.create<arith::MinSIOp>(
                          loc, activeCount,
                          b.create<arith::MaxSIOp>(
                              loc, rLo,
                              b.create<arith::SubIOp>(
                                  loc, loadAt(hiBuf, r), curStripT0)));
                      Value cnt = b.create<arith::SubIOp>(loc, rHi, rLo);
                      Value dR = rowSlice(dstSlot, r);
                      Value sR = rowSlice(srcSlot, r);
                      ifThenElse(
                          b.create<arith::CmpIOp>(
                              loc, arith::CmpIPredicate::sgt, cnt, cI64(0)),
                          {},
                          [&]() {
                            memcpy1D(convTmp, sR, cI64(S), "NRAM2NRAM");
                            sync("io");
                            forLoop(cI64(0), rLo, cI64(1), [&](Value j) {
                              storeAt(convTmp, j, cF32(0.0f));
                            });
                            forLoop(rHi, cI64(S), cI64(1), [&](Value j) {
                              storeAt(convTmp, j, cF32(0.0f));
                            });
                            sync("compute");
                            makeOp(isExp2 ? "row_exp2" : "row_exp",
                                   {dR, reduceScratch, convTmp, cI64(S),
                                    cI64(S)},
                                   {}, {},
                                   isExp2 ? "validated-exp2" : "validated-exp");
                            sync("compute");
                            forLoop(cI64(0), rLo, cI64(1),
                                    [&](Value j) { storeAt(dR, j, cF32(0.0f)); });
                            forLoop(rHi, cI64(S), cI64(1),
                                    [&](Value j) { storeAt(dR, j, cF32(0.0f)); });
                          },
                          [&]() { fillVec(dR, cI64(S), cF32(0.0f)); });
                    });
                    windowZeroedStrips.insert(eop.getResult(0));
                  } else {
                    forLoop(cI64(0), cI64(M), cI64(1),
                            [&](Value r) { emitOne(r, activeCount); });
                  }
                  return;
                }

                StringRef opKind = en == "arith.addf"   ? "add"
                                   : en == "arith.subf" ? "sub"
                                                        : "mul";
                Value lhs = eop.getOperand(0), rhs = eop.getOperand(1);
                bool lhsStrip = slotEnv.contains(lhs) && !ownerScalars.contains(lhs);
                bool rhsStrip = slotEnv.contains(rhs) && !ownerScalars.contains(rhs);

                if (lhsStrip && rhsStrip) {
                  if (singleRow) {
                    makeOp("vec_binary",
                           {rowSlice(dstSlot, singleRow),
                            rowSlice(slotEnv.lookup(lhs), singleRow),
                            rowSlice(slotEnv.lookup(rhs), singleRow), cI64(S)},
                           {}, {b.getNamedAttr("op", b.getStringAttr(opKind))});
                  } else {
                    makeOp("vec_binary",
                           {dstSlot, slotEnv.lookup(lhs), slotEnv.lookup(rhs),
                            cI64(M * S)},
                           {}, {b.getNamedAttr("op", b.getStringAttr(opKind))});
                  }
                  return;
                }

                Value stripV = lhsStrip ? lhs : rhs;
                Value scV = lhsStrip ? rhs : lhs;
                Value srcSlot = slotEnv.lookup(stripV);
                bool reverseSub = (opKind == "sub" && !lhsStrip);

                auto emitRowScalar = [&](Value dPtr, Value sPtr, Value scVal,
                                         int64_t len) {
                  if (reverseSub) {
                    makeOp("vec_scalar", {dPtr, sPtr, cF32(-1.0f), cI64(len)},
                           {}, {b.getNamedAttr("op", b.getStringAttr("mul"))});
                    makeOp("vec_scalar", {dPtr, dPtr, scVal, cI64(len)}, {},
                           {b.getNamedAttr("op", b.getStringAttr("add"))});
                  } else {
                    makeOp("vec_scalar", {dPtr, sPtr, scVal, cI64(len)}, {},
                           {b.getNamedAttr("op", b.getStringAttr(opKind))});
                  }
                };

                if (ownerScalars.contains(scV)) {
                  Value osBuf = slotEnv.lookup(scV);
                  if (singleRow) {
                    emitRowScalar(rowSlice(dstSlot, singleRow),
                                  rowSlice(srcSlot, singleRow),
                                  loadAt(osBuf, singleRow), S);
                  } else {
                    forLoop(cI64(0), cI64(M), cI64(1), [&](Value r) {
                      emitRowScalar(rowSlice(dstSlot, r), rowSlice(srcSlot, r),
                                    loadAt(osBuf, r), S);
                    });
                  }
                } else {
                  Value invSc = lEnv.lookup(scV);
                  if (singleRow)
                    emitRowScalar(rowSlice(dstSlot, singleRow),
                                  rowSlice(srcSlot, singleRow), invSc, S);
                  else
                    emitRowScalar(dstSlot, srcSlot, invSc, M * S);
                }
              };

              // Walk bb in program order, emitting each op from its DSA role.
              for (Operation &op : bb.without_terminator()) {
                auto roleAttr = op.getAttrOfType<StringAttr>("dsa.panel_role");
                if (!roleAttr) {
                  if (auto ld = dyn_cast<memref::LoadOp>(op)) {
                    if (llvm::all_of(ld.getOperands(),
                                     [&](Value u) { return scalarEnv.count(u); })) {
                      SmallVector<Value> coords, strides;
                      for (auto [ax, cv] : llvm::enumerate(ld.getIndices())) {
                        coords.push_back(scalarEnv.lookup(cv));
                        strides.push_back(
                            cI64(dimAttr(ld.getMemref(), "block.strides", ax)));
                      }
                      scalarEnv[ld.getResult()] = loadAt(
                          makeAddr(scalarEnv.lookup(ld.getMemref()), coords,
                                   strides, 0),
                          cI64(0), ld.getType());
                    }
                    continue;
                  }
                  if (!isa<scf::ForOp, scf::IfOp>(op) &&
                      op.getNumResults() == 1 &&
                      !isa<RankedTensorType>(op.getResult(0).getType()) &&
                      llvm::all_of(op.getOperands(),
                                   [&](Value u) { return scalarEnv.count(u); })) {
                    if (Value sv = lowerScalarExpr(op, scalarEnv))
                      scalarEnv[op.getResult(0)] = sv;
                  }
                  continue;
                }
                StringRef role = roleAttr.getValue();

                if (role == "owner_slice") {
                  Value qView = op.getOperand(0);
                  int64_t rank = cast<MemRefType>(qView.getType()).getRank();
                  int64_t D = dimAttr(qView, "block.shape", rank - 1);
                  SmallVector<Value> prefix;
                  for (Value c : op.getOperands().drop_front().drop_back())
                    prefix.push_back(scalarEnv.lookup(c));
                  Value dst = slotAt(&op, 0, halfTy);
                  stageRows(dst, qView, sliceBasePtr(qView, prefix, rowBase),
                            cI64(M), M, D, false);
                  sync("io");
                  slotEnv[op.getResult(0)] = dst;
                  panelFeat[op.getResult(0)] = D;
                  continue;
                }

                if (role == "owner_widen") {
                  Value src = slotEnv.lookup(op.getOperand(0));
                  int64_t F = panelFeat.lookup(op.getOperand(0));
                  Value dst = slotAt(&op, 0, b.getF32Type());
                  widenPanel(dst, src, M * F);
                  sync("compute");
                  slotEnv[op.getResult(0)] = dst;
                  panelFeat[op.getResult(0)] = F;
                  continue;
                }

                if (role == "owner_scalar") {
                  Type resTy = op.getResult(0).getType();
                  Type slotTy = resTy.isIndex() ? Type(b.getI64Type()) : resTy;
                  Value dst = slotAt(&op, 0, slotTy);
                  forLoop(cI64(0), cI64(M), cI64(1), [&](Value r) {
                    if (auto sif = dyn_cast<scf::IfOp>(op)) {
                      Value condV = sif.getCondition();
                      Value cond = ownerScalars.contains(condV)
                                       ? loadAt(slotEnv.lookup(condV), r)
                                       : scalarEnv.lookup(condV);
                      ifThenElse(
                          cond, {},
                          [&]() {
                            storeAt(dst, r,
                                    evalScalarBlockForRow(*sif.thenBlock(), r,
                                                          scalarEnv));
                          },
                          [&]() {
                            storeAt(dst, r,
                                    evalScalarBlockForRow(*sif.elseBlock(), r,
                                                          scalarEnv));
                          });
                    } else {
                      llvm::DenseMap<Value, Value> rowEnv = scalarEnv;
                      rowEnv[rowArg] = b.create<arith::AddIOp>(loc, rowBase, r);
                      for (Value os : ownerScalars)
                        rowEnv[os] = loadAt(slotEnv.lookup(os), r);
                      storeAt(dst, r, lowerScalarExpr(op, rowEnv));
                    }
                  });
                  sync("compute");
                  slotEnv[op.getResult(0)] = dst;
                  ownerScalars.insert(op.getResult(0));
                  continue;
                }

                if (role == "owner_vec") {
                  Value dst = slotAt(&op, 0, b.getF32Type());
                  StringRef en = op.getName().getStringRef();
                  StringRef opKind = en == "arith.addf"   ? "add"
                                     : en == "arith.subf" ? "sub"
                                                          : "mul";
                  Value lhs = op.getOperand(0), rhs = op.getOperand(1);
                  bool lhsPanel = slotEnv.contains(lhs);
                  bool rhsPanel = slotEnv.contains(rhs);
                  int64_t F =
                      panelFeat.lookup(lhsPanel ? lhs : rhs);
                  if (lhsPanel && rhsPanel) {
                    makeOp("vec_binary",
                           {dst, slotEnv.lookup(lhs), slotEnv.lookup(rhs),
                            cI64(M * F)},
                           {}, {b.getNamedAttr("op", b.getStringAttr(opKind))});
                  } else {
                    Value pVal = lhsPanel ? lhs : rhs;
                    Value spVal = unwrapBridge(lhsPanel ? rhs : lhs);
                    Value pSlot = slotEnv.lookup(pVal);
                    bool revSub = (opKind == "sub" && !lhsPanel);
                    auto emitOne = [&](Value dP, Value sP, Value sc, int64_t n) {
                      if (revSub) {
                        makeOp("vec_scalar", {dP, sP, cF32(-1.0f), cI64(n)}, {},
                               {b.getNamedAttr("op", b.getStringAttr("mul"))});
                        makeOp("vec_scalar", {dP, dP, sc, cI64(n)}, {},
                               {b.getNamedAttr("op", b.getStringAttr("add"))});
                      } else {
                        makeOp("vec_scalar", {dP, sP, sc, cI64(n)}, {},
                               {b.getNamedAttr("op", b.getStringAttr(opKind))});
                      }
                    };
                    if (ownerScalars.contains(spVal)) {
                      Value osBuf = slotEnv.lookup(spVal);
                      forLoop(cI64(0), cI64(M), cI64(1), [&](Value r) {
                        emitOne(makeAddr(dst, {r}, {cI64(F)}, 1),
                                makeAddr(pSlot, {r}, {cI64(F)}, 1),
                                loadAt(osBuf, r), F);
                      });
                    } else {
                      emitOne(dst, pSlot, scalarEnv.lookup(spVal), M * F);
                    }
                  }
                  sync("compute");
                  slotEnv[op.getResult(0)] = dst;
                  panelFeat[op.getResult(0)] = F;
                  continue;
                }

                if (role == "owner_narrow") {
                  Value src = slotEnv.lookup(op.getOperand(0));
                  int64_t F = panelFeat.lookup(op.getOperand(0));
                  Value dst = slotAt(&op, 0, halfTy);
                  narrowPanel(dst, src, M * F);
                  sync("compute");
                  slotEnv[op.getResult(0)] = dst;
                  panelFeat[op.getResult(0)] = F;
                  continue;
                }

                if (role == "owner_write") {
                  Value src = slotEnv.lookup(op.getOperand(0));
                  int64_t F = panelFeat.lookup(op.getOperand(0));
                  Value yView = op.getOperand(1);
                  int64_t yRank = cast<MemRefType>(yView.getType()).getRank();
                  int64_t yStride = dimAttr(yView, "block.strides", yRank - 2);
                  SmallVector<Value> prefix;
                  for (Value c : op.getOperands().drop_front(2).drop_back())
                    prefix.push_back(scalarEnv.lookup(c));
                  Value yPtr = sliceBasePtr(yView, prefix, rowBase);
                  if (yStride == F)
                    memcpy1D(yPtr, src, cI64(M * F), "NRAM2GDRAM");
                  else
                    memcpy2D(yPtr, src, cI64(F), cI64(yStride), cI64(F),
                             cI64(M - 1), "NRAM2GDRAM");
                  continue;
                }

                if (role == "strip_loop") {
                  auto loop = cast<scf::ForOp>(op);
                  int64_t K = -1;
                  bool ubIsFullDimK = false;
                  if (auto dimOp =
                          loop.getUpperBound().getDefiningOp<memref::DimOp>()) {
                    int64_t dRank =
                        cast<MemRefType>(dimOp.getSource().getType()).getRank();
                    if (isa<BlockArgument>(dimOp.getSource()) &&
                        dimOp.getConstantIndex() &&
                        *dimOp.getConstantIndex() == size_t(dRank - 2)) {
                      K = dimAttr(dimOp.getSource(), "block.shape",
                                  *dimOp.getConstantIndex());
                      ubIsFullDimK = true;
                    }
                  }
                  if (K < 0) {
                    for (Operation &bop : loop.getBody()->without_terminator()) {
                      if (bop.getAttrOfType<StringAttr>("dsa.panel_role") ==
                          StringAttr::get(b.getContext(), "strip_slice")) {
                        Value kv = bop.getOperand(0);
                        int64_t rk = cast<MemRefType>(kv.getType()).getRank();
                        K = dimAttr(kv, "block.shape", rk - 2);
                        break;
                      }
                    }
                  }
                  bool hasWindow =
                      !matchPattern(loop.getLowerBound(), m_Zero()) ||
                      !ubIsFullDimK;
                  curLoopHasWindow = hasWindow;
                  windowZeroedStrips.clear();

                  // Initialize all loop-carried slots from initArgs.
                  llvm::DenseMap<Operation *, unsigned> outerAccCarryIdx;
                  auto *yieldOp = loop.getBody()->getTerminator();
                  for (unsigned k = 0; k < loop.getNumResults(); ++k) {
                    Value initV = loop.getInitArgs()[k];
                    Value iterV = loop.getRegionIterArgs()[k];
                    Value resV = loop.getResult(k);
                    Value cSlot = slotAt(&op, k, b.getF32Type());
                    if (resV.getType().isF32()) {
                      if (ownerScalars.contains(initV))
                        memcpy1D(cSlot, slotEnv.lookup(initV), cI64(M),
                                 "NRAM2NRAM");
                      else
                        fillVec(cSlot, cI64(M), scalarEnv.lookup(initV));
                      ownerScalars.insert(resV);
                    } else {
                      int64_t F =
                          intentdsl::kir::memberExtent(fn, resV, "block.shape");
                      if (slotEnv.contains(initV)) {
                        memcpy1D(cSlot, slotEnv.lookup(initV), cI64(M * F),
                                 "NRAM2NRAM");
                      } else {
                        Value sp = unwrapBridge(initV);
                        if (ownerScalars.contains(sp)) {
                          forLoop(cI64(0), cI64(M), cI64(1), [&](Value r) {
                            fillVec(makeAddr(cSlot, {r}, {cI64(F)}, 1), cI64(F),
                                    loadAt(slotEnv.lookup(sp), r));
                          });
                        } else {
                          fillVec(cSlot, cI64(M * F), scalarEnv.lookup(sp));
                        }
                      }
                      panelFeat[iterV] = F;
                      panelFeat[resV] = F;
                      Operation *comb = yieldOp->getOperand(k).getDefiningOp();
                      Value term = comb->getOperand(0) == iterV
                                       ? comb->getOperand(1)
                                       : comb->getOperand(0);
                      outerAccCarryIdx[term.getDefiningOp()] = k;
                    }
                    slotEnv[iterV] = cSlot;
                    slotEnv[resV] = cSlot;
                  }
                  sync("compute");

                  // Pre-evaluate owner/strip-invariant scalar ops in loop body.
                  llvm::DenseMap<Value, Value> loopScalars = scalarEnv;
                  for (Operation &bop : loop.getBody()->without_terminator()) {
                    if (bop.hasAttr("dsa.panel_role") ||
                        isa<scf::ForOp, scf::IfOp>(bop))
                      continue;
                    if (bop.getNumResults() == 1 &&
                        !isa<RankedTensorType>(bop.getResult(0).getType()) &&
                        llvm::all_of(bop.getOperands(), [&](Value u) {
                          return loopScalars.count(u);
                        })) {
                      if (Value sv = lowerScalarExpr(bop, loopScalars))
                        loopScalars[bop.getResult(0)] = sv;
                    }
                  }

                  // Check whether every carry update in this loop is identity-padded
                  // past the causal diagonal so strips t0 >= rowBase + M are no-ops.
                  auto causalPadOf = [&](Value v) -> Value {
                    Operation *d = v.getDefiningOp();
                    if (!d || d->getAttrOfType<StringAttr>("dsa.panel_role") !=
                                  StringAttr::get(b.getContext(),
                                                  "strip_causal_mask"))
                      return Value{};
                    if (auto br = dyn_cast<scf::IfOp>(d))
                      return br.elseBlock()->getTerminator()->getOperand(0);
                    if (auto sel = dyn_cast<arith::SelectOp>(d))
                      return sel.getFalseValue();
                    return Value{};
                  };
                  bool canTrimCausal = true;
                  for (unsigned k = 0; k < loop.getNumResults(); ++k) {
                    Value iterV = loop.getRegionIterArgs()[k];
                    Operation *comb = yieldOp->getOperand(k).getDefiningOp();
                    Value rhs = comb->getOperand(0) == iterV
                                    ? comb->getOperand(1)
                                    : comb->getOperand(0);
                    if (loop.getResult(k).getType().isF32()) {
                      Value pad = causalPadOf(rhs);
                      if (!pad || (isa<arith::MaxNumFOp>(comb) && !isNegInfVal(pad)) ||
                          (isa<arith::AddFOp>(comb) && !isZeroVal(pad)))
                        canTrimCausal = false;
                    } else {
                      Operation *omul = rhs.getDefiningOp();
                      bool okW = false;
                      for (Value u : omul->getOperands()) {
                        Value base = unwrapBridge(u);
                        if (auto tr = base.getDefiningOp<arith::TruncFOp>()) {
                          Value pad = causalPadOf(tr.getIn());
                          if (pad && isZeroVal(pad))
                            okW = true;
                        }
                      }
                      if (!okW)
                        canTrimCausal = false;
                    }
                  }

                  Value tStart = cI64(0);
                  Value tokLimit =
                      canTrimCausal
                          ? b.create<arith::MinSIOp>(
                                 loc,
                                 b.create<arith::AddIOp>(loc, rowBase, cI64(M)),
                                 cI64(K))
                                .getResult()
                          : cI64(K);

                  if (hasWindow) {
                    auto evalBoundAtRow = [&](Value bnd, Value r) -> Value {
                      if (bnd == rowArg)
                        return b.create<arith::AddIOp>(loc, rowBase, r);
                      if (ownerScalars.contains(bnd))
                        return toI64(loadAt(slotEnv.lookup(bnd), r));
                      return toI64(scalarEnv.lookup(bnd));
                    };
                    auto winScan = forLoopCarry(
                        cI64(0), cI64(M), cI64(1),
                        ValueRange{cI64(K), cI64(0)},
                        [&](Value r, ValueRange wSt) {
                          Value rawLo = evalBoundAtRow(loop.getLowerBound(), r);
                          Value rawHi = evalBoundAtRow(loop.getUpperBound(), r);
                          Value cLo = b.create<arith::MinSIOp>(
                              loc, cI64(K),
                              b.create<arith::MaxSIOp>(loc, cI64(0), rawLo));
                          Value cHi = b.create<arith::MinSIOp>(
                              loc, cI64(K),
                              b.create<arith::MaxSIOp>(loc, cLo, rawHi));
                          storeAt(loBuf, r, cLo);
                          storeAt(hiBuf, r, cHi);
                          Value hasActive = b.create<arith::CmpIOp>(
                              loc, arith::CmpIPredicate::slt, cLo, cHi);
                          Value nMin = b.create<arith::SelectOp>(
                              loc, hasActive,
                              b.create<arith::MinSIOp>(loc, wSt[0], cLo),
                              wSt[0]);
                          Value nMax = b.create<arith::SelectOp>(
                              loc, hasActive,
                              b.create<arith::MaxSIOp>(loc, wSt[1], cHi),
                              wSt[1]);
                          b.create<scf::YieldOp>(loc, ValueRange{nMin, nMax});
                        });
                    sync("compute");
                    tStart = b.create<arith::MulIOp>(
                        loc,
                        b.create<arith::DivSIOp>(loc, winScan.getResult(0),
                                                 cI64(S)),
                        cI64(S));
                    tokLimit = b.create<arith::MinSIOp>(
                        loc, tokLimit, winScan.getResult(1));
                  }

                  auto causalLive = [&](Value r, Value t0, Value nt) {
                    return b.create<arith::MinSIOp>(
                        loc,
                        b.create<arith::MaxSIOp>(
                            loc,
                            b.create<arith::SubIOp>(
                                loc,
                                b.create<arith::AddIOp>(
                                    loc,
                                    b.create<arith::AddIOp>(loc, rowBase, r),
                                    cI64(1)),
                                t0),
                            cI64(0)),
                        nt);
                  };

                  forLoop(tStart, tokLimit, cI64(S), [&](Value t0) {
                    curStripT0 = t0;
                    Value nt = b.create<arith::MinSIOp>(
                        loc, b.create<arith::SubIOp>(loc, cI64(K), t0),
                        cI64(S));

                    for (Operation &bop : loop.getBody()->without_terminator()) {
                      auto bRoleAttr =
                          bop.getAttrOfType<StringAttr>("dsa.panel_role");
                      if (!bRoleAttr)
                        continue;
                      StringRef bRole = bRoleAttr.getValue();

                      if (bRole == "strip_index_f32") {
                        Value dst = slotAt(&bop, 0, b.getF32Type());
                        forLoop(cI64(0), cI64(S), cI64(1), [&](Value j) {
                          Value tj = b.create<arith::TruncIOp>(
                              loc, b.getI32Type(),
                              b.create<arith::AddIOp>(loc, t0, j));
                          storeAt(dst, j,
                                  b.create<arith::SIToFPOp>(loc, b.getF32Type(),
                                                            tj));
                        });
                        sync("compute");
                        for (int64_t d = 1; d < M; d *= 2) {
                          int64_t copyRows = std::min<int64_t>(d, M - d);
                          if (copyRows > 0) {
                            memcpy1D(makeAddr(dst, {cI64(d)}, {cI64(S)}, 1), dst,
                                     cI64(copyRows * S), "NRAM2NRAM");
                            sync("io");
                          }
                        }
                        slotEnv[bop.getResult(0)] = dst;
                        continue;
                      }

                      if (bRole == "strip_slice") {
                        Value kView = bop.getOperand(0);
                        int64_t kRank =
                            cast<MemRefType>(kView.getType()).getRank();
                        int64_t kFeat =
                            dimAttr(kView, "block.shape", kRank - 1);
                        SmallVector<Value> prefix;
                        for (Value c : bop.getOperands().drop_front().drop_back())
                          prefix.push_back(loopScalars.lookup(c));
                        Value kSlot = slotAt(&bop, 0, halfTy);
                        Value kPtr = sliceBasePtr(kView, prefix, t0);
                        if (Operation *tailIf = stageRows(
                                kSlot, kView, kPtr, nt, S, kFeat, true)) {
                          if (Operation *addrOp = kPtr.getDefiningOp()) {
                            addrOp->setAttr(
                                "bangc.tile_axes",
                                b.getDenseI64ArrayAttr({kRank - 2}));
                            addrOp->setAttr("bangc.tile_sizes",
                                            b.getDenseI64ArrayAttr({S}));
                            orderedTailGuards.push_back(
                                PendingOrderedTailGuard{tailIf, addrOp});
                          }
                        }
                        sync("io");
                        slotEnv[bop.getResult(0)] = kSlot;
                        panelFeat[bop.getResult(0)] = kFeat;
                        continue;
                      }

                      if (bRole == "strip_transpose_slice") {
                        Value vView = bop.getOperand(0);
                        int64_t vRank =
                            cast<MemRefType>(vView.getType()).getRank();
                        int64_t vFeat =
                            dimAttr(vView, "block.shape", vRank - 1);
                        SmallVector<Value> prefix;
                        for (Value c : bop.getOperands().drop_front().drop_back())
                          prefix.push_back(loopScalars.lookup(c));
                        Value vSlot = slotAt(&bop, 0, halfTy);
                        Value vtSlot = slotAt(&bop, 1, halfTy);
                        stageRows(vSlot, vView,
                                  sliceBasePtr(vView, prefix, t0), nt, S,
                                  vFeat, true);
                        sync("io");
                        makeOp("tile_transpose",
                               {vtSlot, vSlot, cI64(S), cI64(vFeat)}, {}, {},
                               "nram-transpose");
                        sync("compute");
                        slotEnv[bop.getResult(0)] = vtSlot;
                        panelFeat[bop.getResult(0)] = vFeat;
                        continue;
                      }

                      if (bRole == "strip_dot") {
                        auto mul =
                            bop.getOperand(0).getDefiningOp<arith::MulFOp>();
                        Value u0 = unwrapBridge(mul.getLhs());
                        Value u1 = unwrapBridge(mul.getRhs());
                        bool u0IsOwner =
                            u0.getDefiningOp() &&
                            u0.getDefiningOp()->getBlock() == &bb;
                        Value qVal = u0IsOwner ? u0 : u1;
                        Value kVal = u0IsOwner ? u1 : u0;
                        int64_t D = panelFeat.lookup(qVal);
                        Value dst = slotAt(&bop, 0, b.getF32Type());
                        emitPanelProduct(dst, slotEnv.lookup(qVal),
                                         slotEnv.lookup(kVal), D, S, false);
                        slotEnv[bop.getResult(0)] = dst;
                        continue;
                      }

                      if (bRole == "strip_elem") {
                        Value dst = slotAt(&bop, 0, b.getF32Type());
                        emitStripElem(bop, dst, loopScalars, Value{}, nt);
                        sync("compute");
                        slotEnv[bop.getResult(0)] = dst;
                        continue;
                      }

                      if (bRole == "strip_causal_mask") {
                        Value dst = slotAt(&bop, 0, b.getF32Type());
                        Value padSrc;
                        if (auto br = dyn_cast<scf::IfOp>(bop))
                          padSrc =
                              br.elseBlock()->getTerminator()->getOperand(0);
                        else
                          padSrc = cast<arith::SelectOp>(bop).getFalseValue();
                        Value padVal = loopScalars.lookup(padSrc);
                        if (!padVal && padSrc.getDefiningOp())
                          padVal = lowerScalarExpr(*padSrc.getDefiningOp(),
                                                   loopScalars);

                        forLoop(cI64(0), cI64(M), cI64(1), [&](Value r) {
                          Value dRow = makeAddr(dst, {r}, {cI64(S)}, 1);
                          Value live = causalLive(r, t0, nt);
                          ifThenElse(
                              b.create<arith::CmpIOp>(
                                  loc, arith::CmpIPredicate::sgt, live, cI64(0)),
                              {},
                              [&]() {
                                Value thenVal;
                                if (auto br = dyn_cast<scf::IfOp>(bop)) {
                                  for (Operation &inOp :
                                       br.thenBlock()->without_terminator()) {
                                    Value inSlot =
                                        slotAt(&inOp, 0, b.getF32Type());
                                    emitStripElem(inOp, inSlot, loopScalars, r,
                                                  live);
                                    sync("compute");
                                    slotEnv[inOp.getResult(0)] = inSlot;
                                  }
                                  thenVal = br.thenBlock()
                                                ->getTerminator()
                                                ->getOperand(0);
                                } else {
                                  thenVal =
                                      cast<arith::SelectOp>(bop).getTrueValue();
                                }
                                Value sRow = makeAddr(slotEnv.lookup(thenVal),
                                                      {r}, {cI64(S)}, 1);
                                memcpy1D(dRow, sRow, cI64(S), "NRAM2NRAM");
                                sync("io");
                                ifThen(b.create<arith::CmpIOp>(
                                           loc, arith::CmpIPredicate::slt, live,
                                           cI64(S)),
                                       [&]() {
                                         forLoop(live, cI64(S), cI64(1),
                                                 [&](Value j) {
                                                   storeAt(dRow, j, padVal);
                                                 });
                                         sync("compute");
                                       });
                              },
                              [&]() {
                                fillVec(dRow, cI64(S), padVal);
                                sync("compute");
                              });
                        });
                        slotEnv[bop.getResult(0)] = dst;
                        causalMaskedStrips.insert(bop.getResult(0));
                        continue;
                      }

                      if (bRole == "strip_narrow") {
                        Value srcVal = bop.getOperand(0);
                        Value srcSlot = slotEnv.lookup(srcVal);
                        if (hasWindow && !windowZeroedStrips.contains(srcVal) &&
                            !causalMaskedStrips.contains(srcVal)) {
                          sync("compute");
                          forLoop(cI64(0), cI64(M), cI64(1), [&](Value r) {
                            Value sRow = makeAddr(srcSlot, {r}, {cI64(S)}, 1);
                            Value rLo = b.create<arith::MinSIOp>(
                                loc, nt,
                                b.create<arith::MaxSIOp>(
                                    loc, cI64(0),
                                    b.create<arith::SubIOp>(
                                        loc, loadAt(loBuf, r), t0)));
                            Value rHi = b.create<arith::MinSIOp>(
                                loc, nt,
                                b.create<arith::MaxSIOp>(
                                    loc, rLo,
                                    b.create<arith::SubIOp>(
                                        loc, loadAt(hiBuf, r), t0)));
                            forLoop(cI64(0), rLo, cI64(1), [&](Value j) {
                              storeAt(sRow, j, cF32(0.0f));
                            });
                            forLoop(rHi, cI64(S), cI64(1), [&](Value j) {
                              storeAt(sRow, j, cF32(0.0f));
                            });
                          });
                          sync("compute");
                        } else if (!windowZeroedStrips.contains(srcVal) &&
                                   !causalMaskedStrips.contains(srcVal)) {
                          ifThen(b.create<arith::CmpIOp>(
                                     loc, arith::CmpIPredicate::slt, nt,
                                     cI64(S)),
                                 [&]() {
                                   sync("compute");
                                   forLoop(cI64(0), cI64(M), cI64(1),
                                           [&](Value r) {
                                             Value sRow = makeAddr(
                                                 srcSlot, {r}, {cI64(S)}, 1);
                                             forLoop(nt, cI64(S), cI64(1),
                                                     [&](Value j) {
                                                       storeAt(sRow, j,
                                                               cF32(0.0f));
                                                     });
                                           });
                                   sync("compute");
                                 });
                        }
                        Value dst = slotAt(&bop, 0, halfTy);
                        narrowPanel(dst, srcSlot, M * S);
                        sync("compute");
                        slotEnv[bop.getResult(0)] = dst;
                        continue;
                      }

                      if (bRole == "strip_outer_acc") {
                        unsigned k = outerAccCarryIdx.lookup(&bop);
                        Value carrySlot = slotAt(&op, k, b.getF32Type());
                        Value u0 = unwrapBridge(bop.getOperand(0));
                        Value u1 = unwrapBridge(bop.getOperand(1));
                        bool u0IsW =
                            u0.getDefiningOp() &&
                            u0.getDefiningOp()->getAttrOfType<StringAttr>(
                                "dsa.panel_role") ==
                                StringAttr::get(b.getContext(), "strip_narrow");
                        Value wVal = u0IsW ? u0 : u1;
                        Value vVal = u0IsW ? u1 : u0;
                        int64_t DV = panelFeat.lookup(vVal);
                        emitPanelProduct(carrySlot, slotEnv.lookup(wVal),
                                         slotEnv.lookup(vVal), S, DV, true);
                        continue;
                      }
                    }

                    // Update scalar reduction carries at scf.yield.
                    for (unsigned k = 0; k < loop.getNumResults(); ++k) {
                      if (!loop.getResult(k).getType().isF32())
                        continue;
                      Value iterV = loop.getRegionIterArgs()[k];
                      Value carrySlot = slotAt(&op, k, b.getF32Type());
                      Operation *comb = yieldOp->getOperand(k).getDefiningOp();
                      Value stripV = comb->getOperand(0) == iterV
                                         ? comb->getOperand(1)
                                         : comb->getOperand(0);
                      Value stripSlot = slotEnv.lookup(stripV);
                      bool isMax = isa<arith::MaxNumFOp>(comb);
                      bool masked = causalMaskedStrips.contains(stripV);
                      bool zeroed = windowZeroedStrips.contains(stripV);
                      if (hasWindow) {
                        Value padId = isMax
                                          ? cF32(-std::numeric_limits<float>::infinity())
                                          : cF32(0.0f);
                        forLoop(cI64(0), cI64(M), cI64(1), [&](Value r) {
                          Value rLo = b.create<arith::MinSIOp>(
                              loc, nt,
                              b.create<arith::MaxSIOp>(
                                  loc, cI64(0),
                                  b.create<arith::SubIOp>(
                                      loc, loadAt(loBuf, r), t0)));
                          Value hiLim = masked ? causalLive(r, t0, nt) : nt;
                          Value rHi = b.create<arith::MinSIOp>(
                              loc, hiLim,
                              b.create<arith::MaxSIOp>(
                                  loc, rLo,
                                  b.create<arith::SubIOp>(
                                      loc, loadAt(hiBuf, r), t0)));
                          Value cnt = b.create<arith::SubIOp>(loc, rHi, rLo);
                          ifThen(
                              b.create<arith::CmpIOp>(
                                  loc, arith::CmpIPredicate::sgt, cnt, cI64(0)),
                              [&]() {
                                Value sRow =
                                    makeAddr(stripSlot, {r}, {cI64(S)}, 1);
                                Value redIn = sRow;
                                if (isMax || !zeroed || masked) {
                                  memcpy1D(convTmp, sRow, cI64(S), "NRAM2NRAM");
                                  sync("io");
                                  forLoop(cI64(0), rLo, cI64(1), [&](Value j) {
                                    storeAt(convTmp, j, padId);
                                  });
                                  forLoop(rHi, cI64(S), cI64(1), [&](Value j) {
                                    storeAt(convTmp, j, padId);
                                  });
                                  sync("compute");
                                  redIn = convTmp;
                                }
                                makeOp("row_reduce",
                                       {scalarOutBuf, reduceScratch, redIn,
                                        cI64(1), cI64(S), cI64(S)},
                                       {},
                                       {b.getNamedAttr(
                                           "algebra",
                                           b.getStringAttr(isMax ? "max_num"
                                                                 : "sum"))},
                                       "row-reduce");
                                Value cur = loadAt(carrySlot, r);
                                Value blk = loadAt(scalarOutBuf, cI64(0));
                                Value nxt =
                                    isMax
                                        ? makeOp("scalar_math", {cur, blk},
                                                 {b.getF32Type()},
                                                 {b.getNamedAttr(
                                                     "op", b.getStringAttr(
                                                               "max_num"))})
                                              ->getResult(0)
                                        : b.create<arith::AddFOp>(loc, cur, blk)
                                              .getResult();
                                storeAt(carrySlot, r, nxt);
                              });
                        });
                      } else {
                        forLoop(cI64(0), cI64(M), cI64(1), [&](Value r) {
                          Value cnt = masked ? causalLive(r, t0, nt) : nt;
                          ifThen(
                              b.create<arith::CmpIOp>(
                                  loc, arith::CmpIPredicate::sgt, cnt, cI64(0)),
                              [&]() {
                                Value sRow =
                                    makeAddr(stripSlot, {r}, {cI64(S)}, 1);
                                makeOp("row_reduce",
                                       {scalarOutBuf, reduceScratch, sRow,
                                        cI64(1), cI64(S), cnt},
                                       {},
                                       {b.getNamedAttr(
                                           "algebra",
                                           b.getStringAttr(isMax ? "max_num"
                                                                 : "sum"))},
                                       "row-reduce");
                                Value cur = loadAt(carrySlot, r);
                                Value blk = loadAt(scalarOutBuf, cI64(0));
                                Value nxt =
                                    isMax
                                        ? makeOp("scalar_math", {cur, blk},
                                                 {b.getF32Type()},
                                                 {b.getNamedAttr(
                                                     "op", b.getStringAttr(
                                                               "max_num"))})
                                              ->getResult(0)
                                        : b.create<arith::AddFOp>(loc, cur, blk)
                                              .getResult();
                                storeAt(carrySlot, r, nxt);
                              });
                        });
                      }
                    }
                  });
                  curLoopHasWindow = false;
                  continue;
                }
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
    Value storage, streambuf, deferred, slabLocal, poolScratch, scalarOut,
        vecPool;
    int64_t slabBytes = 0;
    llvm::DenseMap<Value, Value> env;
    llvm::DenseMap<Operation *, Value> streams;
  };

  static constexpr int64_t kVecStreamChunk = 1024;
  static constexpr int64_t kVecStreamPoolBytes = 8 * kVecStreamChunk * 4 + 256;

  bool canVectorizeStreamLoop(scf::ForOp loop) {
    if (loop->getParentOp() != &space ||
        !matchPattern(loop.getLowerBound(), m_Zero()) ||
        !matchPattern(loop.getStep(), m_One()))
      return false;
    auto dimOp = loop.getUpperBound().getDefiningOp<memref::DimOp>();
    if (!dimOp || !isa<BlockArgument>(dimOp.getSource()) ||
        !dimOp.getConstantIndex())
      return false;
    int64_t N =
        dimAttr(dimOp.getSource(), "block.shape", *dimOp.getConstantIndex());
    if (N < 64 || N > 0x3fffffffLL || N % 32 != 0)
      return false;

    bool isRed = loop.getNumResults() == 1 && loop->hasAttr("kir.reassociate") &&
                 loop.getResult(0).getType().isF32();
    bool isMap = loop.getNumResults() == 0;
    if (!isRed && !isMap)
      return false;

    Value iv = loop.getInductionVar();
    Block &body = *loop.getBody();
    auto *yieldOp = body.getTerminator();
    Operation *redComb = nullptr;
    if (isRed) {
      redComb = yieldOp->getOperand(0).getDefiningOp();
      Value iterV = loop.getRegionIterArgs()[0];
      if (!redComb || redComb->getBlock() != &body || !redComb->hasOneUse() ||
          !isa<arith::MaxNumFOp, arith::AddFOp>(redComb) ||
          (redComb->getOperand(0) != iterV && redComb->getOperand(1) != iterV))
        return false;
    }

    auto checkContig1DAccess = [&](Value view, ValueRange indices) -> bool {
      if (!isa<BlockArgument>(view))
        return false;
      Type et = cast<MemRefType>(view.getType()).getElementType();
      if (!et.isF32() && !et.isF16() && !et.isBF16())
        return false;
      int64_t rank = cast<MemRefType>(view.getType()).getRank();
      if (rank < 1 || int64_t(indices.size()) != rank ||
          indices.back() != iv ||
          dimAttr(view, "block.strides", rank - 1) != 1 ||
          dimAttr(view, "block.shape", rank - 1) < N)
        return false;
      return llvm::all_of(indices.drop_back(), [&](Value c) {
        return loop.isDefinedOutsideOfLoop(c);
      });
    };

    llvm::DenseSet<Value> vecF32, vecHalf, invSc;
    int numStores = 0;
    for (Operation &bop : body.without_terminator()) {
      if (&bop == redComb)
        continue;
      if (auto ld = dyn_cast<memref::LoadOp>(bop)) {
        if (!checkContig1DAccess(ld.getMemref(), ld.getIndices()))
          return false;
        if (ld.getType().isF32())
          vecF32.insert(ld.getResult());
        else
          vecHalf.insert(ld.getResult());
        continue;
      }
      if (auto stOp = dyn_cast<memref::StoreOp>(bop)) {
        if (!isMap || !checkContig1DAccess(stOp.getMemref(), stOp.getIndices()))
          return false;
        Value val = stOp.getValue();
        if (!vecF32.contains(val) && !vecHalf.contains(val))
          return false;
        ++numStores;
        continue;
      }
      if (auto cmp = dyn_cast<arith::CmpIOp>(bop)) {
        if (cmp.getPredicate() == arith::CmpIPredicate::eq &&
            ((cmp.getLhs() == iv &&
              loop.isDefinedOutsideOfLoop(cmp.getRhs())) ||
             (cmp.getRhs() == iv &&
              loop.isDefinedOutsideOfLoop(cmp.getLhs()))))
          continue;
        return false;
      }
      if (auto iff = dyn_cast<scf::IfOp>(bop)) {
        if (!isMap || iff.getNumResults() != 1 ||
            !iff.getResult(0).getType().isF32() || iff.getElseRegion().empty())
          return false;
        auto cmp = iff.getCondition().getDefiningOp<arith::CmpIOp>();
        if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::eq ||
            !((cmp.getLhs() == iv &&
               loop.isDefinedOutsideOfLoop(cmp.getRhs())) ||
              (cmp.getRhs() == iv &&
               loop.isDefinedOutsideOfLoop(cmp.getLhs()))))
          return false;
        if (!iff.elseBlock()->without_terminator().empty())
          return false;
        Value baseV = iff.elseBlock()->getTerminator()->getOperand(0);
        if (!vecF32.contains(baseV))
          return false;
        Operation *adjOp = nullptr;
        for (Operation &tOp : iff.thenBlock()->without_terminator()) {
          if (isa<arith::ConstantOp>(tOp) && tOp.getResult(0).getType().isF32()) {
            invSc.insert(tOp.getResult(0));
            continue;
          }
          if (!adjOp && isa<arith::AddFOp, arith::SubFOp>(tOp) &&
              tOp.getResult(0).getType().isF32()) {
            adjOp = &tOp;
            continue;
          }
          return false;
        }
        if (!adjOp ||
            iff.thenBlock()->getTerminator()->getOperand(0) !=
                adjOp->getResult(0))
          return false;
        Value o0 = adjOp->getOperand(0), o1 = adjOp->getOperand(1);
        auto isInv = [&](Value v) {
          return loop.isDefinedOutsideOfLoop(v) || invSc.contains(v);
        };
        if (!(o0 == baseV && isInv(o1)) &&
            !(isa<arith::AddFOp>(adjOp) && o1 == baseV && isInv(o0)))
          return false;
        vecF32.insert(iff.getResult(0));
        continue;
      }
      StringRef fnName = bop.getName().getStringRef();
      if (isa<arith::ConstantOp>(bop) && bop.getResult(0).getType().isF32()) {
        invSc.insert(bop.getResult(0));
        continue;
      }
      if (fnName == "arith.extf" && bop.getResult(0).getType().isF32() &&
          vecHalf.contains(bop.getOperand(0))) {
        vecF32.insert(bop.getResult(0));
        continue;
      }
      if (fnName == "arith.truncf" &&
          (bop.getResult(0).getType().isBF16() ||
           bop.getResult(0).getType().isF16()) &&
          vecF32.contains(bop.getOperand(0))) {
        vecHalf.insert(bop.getResult(0));
        continue;
      }
      if ((fnName == "math.exp" || fnName == "math.exp2") &&
          vecF32.contains(bop.getOperand(0))) {
        vecF32.insert(bop.getResult(0));
        continue;
      }
      if ((fnName == "arith.addf" || fnName == "arith.subf" ||
           fnName == "arith.mulf") &&
          bop.getResult(0).getType().isF32()) {
        Value u0 = bop.getOperand(0), u1 = bop.getOperand(1);
        auto okS = [&](Value v) {
          return v.getType().isF32() &&
                 (loop.isDefinedOutsideOfLoop(v) || invSc.contains(v));
        };
        if ((vecF32.contains(u0) && vecF32.contains(u1)) ||
            (vecF32.contains(u0) && okS(u1)) ||
            (okS(u0) && vecF32.contains(u1))) {
          vecF32.insert(bop.getResult(0));
          continue;
        }
      }
      return false;
    }

    if (isRed) {
      Value iterV = loop.getRegionIterArgs()[0];
      Value rhs = redComb->getOperand(0) == iterV ? redComb->getOperand(1)
                                                  : redComb->getOperand(0);
      return vecF32.contains(rhs);
    }
    return numStores > 0;
  }

  bool tryEmitVectorStreamLoop(scf::ForOp loop, GeneralState &st) {
    if (!st.vecPool || !canVectorizeStreamLoop(loop))
      return false;

    auto dimOp = loop.getUpperBound().getDefiningOp<memref::DimOp>();
    int64_t N =
        dimAttr(dimOp.getSource(), "block.shape", *dimOp.getConstantIndex());
    bool isRed = loop.getNumResults() == 1;
    Block &body = *loop.getBody();
    auto *yieldOp = body.getTerminator();
    Operation *redComb =
        isRed ? yieldOp->getOperand(0).getDefiningOp() : nullptr;

    // Partition vecPool: slots 0..5 = work vectors (1024 f32 = 4096B each),
    // slot 6 = exp/reduce scratch (4096B), slot 7 = scalarOut (128B) + narrowScratch (2048B).
    auto workSlot = [&](int idx, Type elemTy) {
      return nramPtr(st.vecPool, cI64((idx % 6) * kVecStreamChunk * 4), elemTy);
    };
    Value expRedScratch =
        nramPtr(st.vecPool, cI64(6 * kVecStreamChunk * 4), b.getF32Type());
    Value redScalarOut =
        nramPtr(st.vecPool, cI64(7 * kVecStreamChunk * 4), b.getF32Type());
    Value narrowScratch = nramPtr(
        st.vecPool, cI64(7 * kVecStreamChunk * 4 + 128), b.getBF16Type());

    auto streamChunkGlobalPtr = [&](Value view, ValueRange indices, Value c0) {
      SmallVector<Value> addrC, addrS;
      for (auto [ax, cv] : llvm::enumerate(indices.drop_back())) {
        addrC.push_back(st.env.lookup(cv));
        addrS.push_back(cI64(dimAttr(view, "block.strides", ax)));
      }
      addrC.push_back(c0);
      addrS.push_back(cI64(1));
      return makeAddr(st.env.lookup(view), addrC, addrS, 0);
    };

    auto emitChunkBody = [&](Value c0, Value curLanes,
                             llvm::DenseMap<Value, Value> &vecEnv) {
      llvm::DenseMap<Value, Value> scEnv = st.env;
      int nextSlot = 0;
      for (Operation &bop : body.without_terminator()) {
        if (&bop == redComb || isa<arith::CmpIOp>(bop))
          continue;
        if (auto cst = dyn_cast<arith::ConstantOp>(bop)) {
          scEnv[cst.getResult()] = lowerScalarExpr(bop, scEnv);
          continue;
        }
        if (auto ld = dyn_cast<memref::LoadOp>(bop)) {
          Value dst = workSlot(nextSlot++, ld.getType());
          memcpy1D(dst,
                   streamChunkGlobalPtr(ld.getMemref(), ld.getIndices(), c0),
                   curLanes, "GDRAM2NRAM");
          sync("io");
          vecEnv[ld.getResult()] = dst;
          continue;
        }
        if (auto stOp = dyn_cast<memref::StoreOp>(bop)) {
          Value src = vecEnv.lookup(stOp.getValue());
          sync("compute");
          memcpy1D(
              streamChunkGlobalPtr(stOp.getMemref(), stOp.getIndices(), c0),
              src, curLanes, "NRAM2GDRAM");
          sync("io");
          continue;
        }
        if (auto iff = dyn_cast<scf::IfOp>(bop)) {
          auto cmp = iff.getCondition().getDefiningOp<arith::CmpIOp>();
          Value targetIdx = cmp.getLhs() == loop.getInductionVar()
                                ? scEnv.lookup(cmp.getRhs())
                                : scEnv.lookup(cmp.getLhs());
          Value baseV = iff.elseBlock()->getTerminator()->getOperand(0);
          Value dst = workSlot(nextSlot++, b.getF32Type());
          memcpy1D(dst, vecEnv.lookup(baseV), curLanes, "NRAM2NRAM");
          sync("io");
          Operation *adjOp = nullptr;
          for (Operation &tOp : iff.thenBlock()->without_terminator()) {
            if (isa<arith::ConstantOp>(tOp))
              scEnv[tOp.getResult(0)] = lowerScalarExpr(tOp, scEnv);
            else
              adjOp = &tOp;
          }
          Value deltaV = adjOp->getOperand(0) == baseV
                             ? scEnv.lookup(adjOp->getOperand(1))
                             : scEnv.lookup(adjOp->getOperand(0));
          Value t64 = toI64(targetIdx);
          Value hi32 = b.create<arith::TruncIOp>(
              loc, b.getI32Type(),
              b.create<arith::ShRUIOp>(loc, t64, cI64(32)));
          Value t32 = b.create<arith::TruncIOp>(loc, b.getI32Type(), t64);
          Value c0_32 = b.create<arith::TruncIOp>(loc, b.getI32Type(), c0);
          Value lanes32 =
              b.create<arith::TruncIOp>(loc, b.getI32Type(), curLanes);
          Value cEnd32 = b.create<arith::AddIOp>(loc, c0_32, lanes32);
          Value inChunk = b.create<arith::AndIOp>(
              loc,
              b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, hi32,
                                      cI32(0)),
              b.create<arith::AndIOp>(
                  loc,
                  b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, t32,
                                          c0_32),
                  b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, t32,
                                          cEnd32)));
          Value rel32 = b.create<arith::SubIOp>(loc, t32, c0_32);
          Value safeRel32 =
              b.create<arith::SelectOp>(loc, inChunk, rel32, cI32(0));
          Value safeRel =
              b.create<arith::ExtUIOp>(loc, b.getI64Type(), safeRel32);
          sync("all");
          Value cur = loadAt(dst, safeRel, b.getF32Type());
          Value adjDelta =
              b.create<arith::SelectOp>(loc, inChunk, deltaV, cF32(0.0f));
          Value upd = isa<arith::SubFOp>(adjOp)
                          ? b.create<arith::SubFOp>(loc, cur, adjDelta)
                                .getResult()
                          : b.create<arith::AddFOp>(loc, cur, adjDelta)
                                .getResult();
          storeAt(dst, safeRel, upd);
          sync("all");
          vecEnv[iff.getResult(0)] = dst;
          continue;
        }
        StringRef fnName = bop.getName().getStringRef();
        if (fnName == "arith.extf") {
          Type fromTy = bop.getOperand(0).getType();
          Value dst = workSlot(nextSlot++, b.getF32Type());
          Value src = vecEnv.lookup(bop.getOperand(0));
          if (fromTy.isBF16())
            makeOp("bf16_tile", {dst, src, curLanes, curLanes}, {},
                   {b.getNamedAttr("mode", b.getStringAttr("widen"))},
                   "bf16-tile-convert");
          else
            makeOp("vec_cast", {dst, src, curLanes}, {},
                   {b.getNamedAttr("kind", b.getStringAttr("f16_to_f32"))});
          sync("compute");
          vecEnv[bop.getResult(0)] = dst;
          continue;
        }
        if (fnName == "arith.truncf") {
          Type toTy = bop.getResult(0).getType();
          Value dst = workSlot(nextSlot++, toTy);
          Value src = vecEnv.lookup(bop.getOperand(0));
          if (toTy.isBF16())
            makeOp("bf16_tile",
                   {dst, src, narrowScratch, curLanes, curLanes}, {},
                   {b.getNamedAttr("mode", b.getStringAttr("narrow"))},
                   "bf16-tile-convert");
          else
            makeOp("vec_cast", {dst, src, curLanes}, {},
                   {b.getNamedAttr("kind", b.getStringAttr("f32_to_f16"))});
          sync("compute");
          vecEnv[bop.getResult(0)] = dst;
          continue;
        }
        if (fnName == "math.exp" || fnName == "math.exp2") {
          bool isExp2 = fnName == "math.exp2";
          Value dst = workSlot(nextSlot++, b.getF32Type());
          Value src = vecEnv.lookup(bop.getOperand(0));
          makeOp(isExp2 ? "row_exp2" : "row_exp",
                 {dst, expRedScratch, src, curLanes, curLanes}, {}, {},
                 isExp2 ? "validated-exp2" : "validated-exp");
          sync("compute");
          vecEnv[bop.getResult(0)] = dst;
          continue;
        }
        StringRef opKind = fnName == "arith.addf"   ? "add"
                           : fnName == "arith.subf" ? "sub"
                                                    : "mul";
        Value u0 = bop.getOperand(0), u1 = bop.getOperand(1);
        bool v0 = vecEnv.contains(u0), v1 = vecEnv.contains(u1);
        Value dst = workSlot(nextSlot++, b.getF32Type());
        if (v0 && v1) {
          makeOp("vec_binary",
                 {dst, vecEnv.lookup(u0), vecEnv.lookup(u1), curLanes}, {},
                 {b.getNamedAttr("op", b.getStringAttr(opKind))});
        } else if (v0) {
          makeOp("vec_scalar",
                 {dst, vecEnv.lookup(u0), scEnv.lookup(u1), curLanes}, {},
                 {b.getNamedAttr("op", b.getStringAttr(opKind))});
        } else if (opKind == "sub") {
          makeOp("vec_scalar",
                 {dst, vecEnv.lookup(u1), cF32(-1.0f), curLanes}, {},
                 {b.getNamedAttr("op", b.getStringAttr("mul"))});
          makeOp("vec_scalar", {dst, dst, scEnv.lookup(u0), curLanes}, {},
                 {b.getNamedAttr("op", b.getStringAttr("add"))});
        } else {
          makeOp("vec_scalar",
                 {dst, vecEnv.lookup(u1), scEnv.lookup(u0), curLanes}, {},
                 {b.getNamedAttr("op", b.getStringAttr(opKind))});
        }
        sync("compute");
        vecEnv[bop.getResult(0)] = dst;
      }
    };

    if (isRed) {
      Value initVal = st.env.lookup(loop.getInitArgs()[0]);
      bool isMax = isa<arith::MaxNumFOp>(redComb);
      Value iterV = loop.getRegionIterArgs()[0];
      Value rhs = redComb->getOperand(0) == iterV ? redComb->getOperand(1)
                                                  : redComb->getOperand(0);
      auto cLoop = forLoopCarry(
          cI64(0), cI64(N), cI64(kVecStreamChunk), ValueRange{initVal},
          [&](Value c0, ValueRange cIter) {
            Value curLanes = b.create<arith::MinSIOp>(
                loc, b.create<arith::SubIOp>(loc, cI64(N), c0),
                cI64(kVecStreamChunk));
            llvm::DenseMap<Value, Value> vecEnv;
            emitChunkBody(c0, curLanes, vecEnv);
            makeOp("row_reduce",
                   {redScalarOut, expRedScratch, vecEnv.lookup(rhs), cI64(1),
                    curLanes, curLanes},
                   {},
                   {b.getNamedAttr("algebra",
                                   b.getStringAttr(isMax ? "max_num" : "sum"))},
                   "row-reduce");
            Value blk = loadAt(redScalarOut, cI64(0));
            Value nxt =
                isMax
                    ? makeOp("scalar_math", {cIter[0], blk}, {b.getF32Type()},
                             {b.getNamedAttr("op", b.getStringAttr("max_num"))})
                          ->getResult(0)
                    : b.create<arith::AddFOp>(loc, cIter[0], blk).getResult();
            b.create<scf::YieldOp>(loc, ValueRange{nxt});
          });
      st.env[loop.getResult(0)] = cLoop.getResult(0);
    } else {
      forLoop(cI64(0), cI64(N), cI64(kVecStreamChunk), [&](Value c0) {
        Value curLanes = b.create<arith::MinSIOp>(
            loc, b.create<arith::SubIOp>(loc, cI64(N), c0),
            cI64(kVecStreamChunk));
        llvm::DenseMap<Value, Value> vecEnv;
        emitChunkBody(c0, curLanes, vecEnv);
      });
    }
    return true;
  }

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
        if (tryEmitVectorStreamLoop(loop, st))
          continue;
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

  // Abstract 2D spatial plane stencil matcher & NRAM DMA+vector lowerer.
  struct StencilTapSpec {
    int64_t k = 0;
    int64_t dh = 0, dw = 0;
    int64_t ohLo = 0, ohHi = 0, owLo = 0, owHi = 0;
  };
  struct PlaneStencilPlan {
    int64_t H = 0, W = 0, WP = 0, padH = 1;
    Value xView, outView, biasVal, weightVal;
    SmallVector<Value, 4> xOuterCoords, outOuterCoords;
    scf::ForOp loop;
    SmallVector<StencilTapSpec, 16> taps;
  };

  std::optional<PlaneStencilPlan> matchPlaneStencil() {
    auto axes = space.getAttrOfType<DenseI64ArrayAttr>("axes");
    auto &bb = space.getRegion(0).front();
    int64_t tile = space.getAttrOfType<IntegerAttr>("block.owner_tile").getInt();
    int64_t owners = space.getAttrOfType<IntegerAttr>("block.owners").getInt();
    int64_t slabBytes =
        space.getAttrOfType<IntegerAttr>("dsa.owner_bytes").getInt();
    if (!axes || axes.size() < 2 || bb.getNumArguments() != axes.size() ||
        slabBytes != 0)
      return std::nullopt;

    unsigned na = axes.size();
    Value hArg = bb.getArgument(na - 2);
    Value wArg = bb.getArgument(na - 1);
    int64_t H = dimAttr(space.getOperand(na - 2), "block.shape", axes[na - 2]);
    int64_t W = dimAttr(space.getOperand(na - 1), "block.shape", axes[na - 1]);
    int64_t WP = llvm::alignTo(W, 32);
    int64_t planeOwners = H * W;
    if (H < 1 || W < 8 || planeOwners <= 0 || H * WP * 4 > 65536 ||
        owners % planeOwners != 0)
      return std::nullopt;

    auto planeInvariant = [&](Value v, Value loopIv = {}) {
      return !intentdsl::dsa::ownerValueDependsOn(v, hArg) &&
             !intentdsl::dsa::ownerValueDependsOn(v, wArg) &&
             (!loopIv || !intentdsl::dsa::ownerValueDependsOn(v, loopIv));
    };

    memref::StoreOp outStore;
    scf::ForOp stencilLoop;
    for (Operation &op : bb.without_terminator()) {
      if (auto stOp = dyn_cast<memref::StoreOp>(op)) {
        if (outStore)
          return std::nullopt;
        outStore = stOp;
      } else if (auto lOp = dyn_cast<scf::ForOp>(op)) {
        if (stencilLoop)
          return std::nullopt;
        stencilLoop = lOp;
      } else if (!isMemoryEffectFree(&op) && !isa<memref::LoadOp>(op)) {
        return std::nullopt;
      } else if (auto ldOp = dyn_cast<memref::LoadOp>(op)) {
        if (!isa<BlockArgument>(ldOp.getMemref()) ||
            llvm::any_of(ldOp.getIndices(),
                         [&](Value c) { return !planeInvariant(c); }))
          return std::nullopt;
      }
    }
    if (!outStore || !stencilLoop || !isa<BlockArgument>(outStore.getMemref()))
      return std::nullopt;

    Value outView = outStore.getMemref();
    if (!cast<MemRefType>(outView.getType()).getElementType().isF32() ||
        outStore.getIndices().size() != na ||
        outStore.getIndices()[na - 2] != hArg ||
        outStore.getIndices()[na - 1] != wArg ||
        dimAttr(outView, "block.strides", na - 1) != 1 ||
        dimAttr(outView, "block.strides", na - 2) != W ||
        llvm::any_of(outStore.getIndices().drop_back(2),
                     [&](Value c) { return !planeInvariant(c); }))
      return std::nullopt;

    Value biasVal;
    Value storedV = outStore.getValue();
    if (storedV != stencilLoop.getResult(0)) {
      auto addOp = storedV.getDefiningOp<arith::AddFOp>();
      if (!addOp)
        return std::nullopt;
      if (addOp.getLhs() == stencilLoop.getResult(0) &&
          planeInvariant(addOp.getRhs()))
        biasVal = addOp.getRhs();
      else if (addOp.getRhs() == stencilLoop.getResult(0) &&
               planeInvariant(addOp.getLhs()))
        biasVal = addOp.getLhs();
      else
        return std::nullopt;
    }

    llvm::APInt kAp;
    if (!stencilLoop->hasAttr("kir.reassociate") ||
        stencilLoop.getNumResults() != 1 ||
        !stencilLoop.getResult(0).getType().isF32() ||
        !matchPattern(stencilLoop.getLowerBound(), m_Zero()) ||
        !matchPattern(stencilLoop.getStep(), m_One()) ||
        !matchPattern(stencilLoop.getUpperBound(), m_ConstantInt(&kAp)))
      return std::nullopt;
    int64_t kTaps = kAp.getSExtValue();
    auto initCst =
        stencilLoop.getInitArgs()[0].getDefiningOp<arith::ConstantOp>();
    auto initFa = initCst ? dyn_cast<FloatAttr>(initCst.getValue()) : FloatAttr{};
    if (kTaps < 1 || kTaps > 64 || !initFa || !initFa.getValue().isZero())
      return std::nullopt;

    Block &lBody = *stencilLoop.getBody();
    Value iv = stencilLoop.getInductionVar();
    Value iterArg = stencilLoop.getRegionIterArgs()[0];
    auto yieldAdd =
        lBody.getTerminator()->getOperand(0).getDefiningOp<arith::AddFOp>();
    if (!yieldAdd ||
        (yieldAdd.getLhs() != iterArg && yieldAdd.getRhs() != iterArg))
      return std::nullopt;
    Value termV =
        yieldAdd.getLhs() == iterArg ? yieldAdd.getRhs() : yieldAdd.getLhs();
    auto termMul = termV.getDefiningOp<arith::MulFOp>();
    if (!termMul)
      return std::nullopt;

    Value sampleVal, weightVal;
    if (!intentdsl::dsa::ownerValueDependsOn(termMul.getRhs(), hArg) &&
        !intentdsl::dsa::ownerValueDependsOn(termMul.getRhs(), wArg)) {
      sampleVal = termMul.getLhs();
      weightVal = termMul.getRhs();
    } else if (!intentdsl::dsa::ownerValueDependsOn(termMul.getLhs(), hArg) &&
               !intentdsl::dsa::ownerValueDependsOn(termMul.getLhs(), wArg)) {
      sampleVal = termMul.getRhs();
      weightVal = termMul.getLhs();
    } else {
      return std::nullopt;
    }

    auto sampleIf = sampleVal.getDefiningOp<scf::IfOp>();
    if (!sampleIf || sampleIf.getNumResults() != 1 ||
        sampleIf.getElseRegion().empty())
      return std::nullopt;
    Value elseY = sampleIf.elseBlock()->getTerminator()->getOperand(0);
    auto elseCst = elseY.getDefiningOp<arith::ConstantOp>();
    auto elseFa = elseCst ? dyn_cast<FloatAttr>(elseCst.getValue()) : FloatAttr{};
    if (!elseFa || !elseFa.getValue().isZero())
      return std::nullopt;

    memref::LoadOp xLoad;
    for (Operation &tOp : sampleIf.thenBlock()->without_terminator()) {
      if (auto ld = dyn_cast<memref::LoadOp>(tOp)) {
        if (xLoad)
          return std::nullopt;
        xLoad = ld;
      } else if (!isMemoryEffectFree(&tOp)) {
        return std::nullopt;
      }
    }
    if (!xLoad ||
        sampleIf.thenBlock()->getTerminator()->getOperand(0) !=
            xLoad.getResult() ||
        !isa<BlockArgument>(xLoad.getMemref()))
      return std::nullopt;

    Value xView = xLoad.getMemref();
    unsigned xRank = cast<MemRefType>(xView.getType()).getRank();
    if (!cast<MemRefType>(xView.getType()).getElementType().isF32() ||
        xRank < 2 || xLoad.getIndices().size() != xRank ||
        dimAttr(xView, "block.shape", xRank - 2) != H ||
        dimAttr(xView, "block.shape", xRank - 1) != W ||
        dimAttr(xView, "block.strides", xRank - 1) != 1 ||
        dimAttr(xView, "block.strides", xRank - 2) != W ||
        llvm::any_of(xLoad.getIndices().drop_back(2),
                     [&](Value c) { return !planeInvariant(c, iv); }))
      return std::nullopt;

    // Verify public ABI disjointness (xView and weight reads must be In, outView Out).
    auto abiParsed =
        llvm::json::parse(fn->getAttrOfType<StringAttr>("kir.abi").getValue());
    if (!abiParsed || !abiParsed->getAsArray()) {
      if (!abiParsed)
        llvm::consumeError(abiParsed.takeError());
      return std::nullopt;
    }
    const auto &abiArr = *abiParsed->getAsArray();
    auto argDir = [&](Value v) -> StringRef {
      auto ba = dyn_cast<BlockArgument>(v);
      if (!ba || ba.getOwner() != &fn.front())
        return "";
      auto *obj = abiArr[ba.getArgNumber()].getAsObject();
      return obj ? obj->getString("direction").value_or("") : "";
    };
    if (argDir(xView) != "In" || argDir(outView) != "Out")
      return std::nullopt;

    // Verify weightVal has no side effects other than reading In views.
    bool weightOk = true;
    lBody.walk([&](Operation *sub) {
      if (sub == xLoad.getOperation())
        return;
      if (auto ld = dyn_cast<memref::LoadOp>(sub)) {
        if (argDir(ld.getMemref()) != "In" ||
            llvm::any_of(ld.getIndices(), [&](Value c) {
              return intentdsl::dsa::ownerValueDependsOn(c, hArg) ||
                     intentdsl::dsa::ownerValueDependsOn(c, wArg);
            }))
          weightOk = false;
      } else if (isa<memref::StoreOp>(sub)) {
        weightOk = false;
      }
    });
    if (!weightOk)
      return std::nullopt;

    // Abstract interpretation of affine spatial expressions & 2D indicator rectangles.
    struct AffVal {
      int kind = -1; // 0 = const, 1 = hArg + c, 2 = wArg + c
      int64_t c = 0;
    };
    struct Rect2D {
      int64_t hLo = 0, hHi = 0, wLo = 0, wHi = 0;
      bool empty() const { return hLo >= hHi || wLo >= wHi; }
    };
    auto intersectRect = [](Rect2D a, Rect2D b) -> Rect2D {
      Rect2D r{std::max(a.hLo, b.hLo), std::min(a.hHi, b.hHi),
               std::max(a.wLo, b.wLo), std::min(a.wHi, b.wHi)};
      if (r.empty())
        return {0, 0, 0, 0};
      return r;
    };

    llvm::DenseMap<Value, AffVal> outerAff;
    outerAff[hArg] = {1, 0};
    outerAff[wArg] = {2, 0};
    for (Operation &topOp : bb.without_terminator()) {
      if (&topOp == stencilLoop.getOperation())
        break;
      if (auto cst = dyn_cast<arith::ConstantOp>(topOp)) {
        if (auto ia = dyn_cast<IntegerAttr>(cst.getValue()))
          outerAff[cst.getResult()] = {0, ia.getInt()};
      } else if (auto dOp = dyn_cast<memref::DimOp>(topOp)) {
        if (isa<BlockArgument>(dOp.getSource()) && dOp.getConstantIndex())
          outerAff[dOp.getResult()] = {
              0, dimAttr(dOp.getSource(), "block.shape",
                         *dOp.getConstantIndex())};
      }
    }

    PlaneStencilPlan plan;
    plan.H = H;
    plan.W = W;
    plan.WP = WP;
    plan.xView = xView;
    plan.outView = outView;
    plan.biasVal = biasVal;
    plan.weightVal = weightVal;
    plan.loop = stencilLoop;
    llvm::append_range(plan.xOuterCoords, xLoad.getIndices().drop_back(2));
    llvm::append_range(plan.outOuterCoords, outStore.getIndices().drop_back(2));

    int64_t maxAbsDh = 1;
    for (int64_t k = 0; k < kTaps; ++k) {
      llvm::DenseMap<Value, AffVal> aff = outerAff;
      llvm::DenseMap<Value, Rect2D> rects;
      aff[iv] = {0, k};

      std::function<bool(Operation &)> evalAffOp =
          [&](Operation &op) -> bool {
        if (auto cst = dyn_cast<arith::ConstantOp>(op)) {
          if (auto ia = dyn_cast<IntegerAttr>(cst.getValue())) {
            int64_t v = ia.getInt();
            aff[cst.getResult()] = {0, v};
            if (v == 1)
              rects[cst.getResult()] = {0, H, 0, W};
            else if (v == 0)
              rects[cst.getResult()] = {0, 0, 0, 0};
            return true;
          }
          return true;
        }
        if (auto dOp = dyn_cast<memref::DimOp>(op)) {
          if (isa<BlockArgument>(dOp.getSource()) && dOp.getConstantIndex()) {
            aff[dOp.getResult()] = {
                0, dimAttr(dOp.getSource(), "block.shape",
                           *dOp.getConstantIndex())};
            return true;
          }
          return false;
        }
        StringRef n = op.getName().getStringRef();
        if (n == "arith.addi" || n == "arith.subi" || n == "arith.floordivsi" ||
            n == "arith.divsi" || n == "arith.remsi" || n == "arith.muli") {
          auto it0 = aff.find(op.getOperand(0)),
               it1 = aff.find(op.getOperand(1));
          if (it0 == aff.end() || it1 == aff.end())
            return true;
          AffVal a0 = it0->second, a1 = it1->second;
          if (n == "arith.addi" && (a0.kind == 0 || a1.kind == 0)) {
            aff[op.getResult(0)] = {a0.kind ? a0.kind : a1.kind, a0.c + a1.c};
          } else if (n == "arith.subi" && a1.kind == 0) {
            aff[op.getResult(0)] = {a0.kind, a0.c - a1.c};
          } else if (a0.kind == 0 && a1.kind == 0 && a1.c != 0) {
            if (n == "arith.muli")
              aff[op.getResult(0)] = {0, a0.c * a1.c};
            else if (n == "arith.divsi")
              aff[op.getResult(0)] = {0, a0.c / a1.c};
            else if (n == "arith.floordivsi") {
              int64_t q = a0.c / a1.c, r = a0.c % a1.c;
              if (r != 0 && ((r ^ a1.c) < 0))
                --q;
              aff[op.getResult(0)] = {0, q};
            } else if (n == "arith.remsi")
              aff[op.getResult(0)] = {0, a0.c % a1.c};
          }
          return true;
        }
        if (auto cmp = dyn_cast<arith::CmpIOp>(op)) {
          Value lhs = cmp.getLhs(), rhs = cmp.getRhs();
          auto pred = cmp.getPredicate();
          if (pred == arith::CmpIPredicate::eq) {
            if (rects.contains(lhs) && aff.contains(rhs) &&
                aff[rhs].kind == 0 && aff[rhs].c == 1) {
              rects[cmp.getResult()] = rects[lhs];
              return true;
            }
            if (rects.contains(rhs) && aff.contains(lhs) &&
                aff[lhs].kind == 0 && aff[lhs].c == 1) {
              rects[cmp.getResult()] = rects[rhs];
              return true;
            }
          }
          if (!aff.contains(lhs) || !aff.contains(rhs))
            return true;
          AffVal a0 = aff[lhs], a1 = aff[rhs];
          if (a0.kind == 0 && a1.kind != 0) {
            std::swap(a0, a1);
            pred = arith::invertPredicate(pred); // wait: swap flips direction!
            return false;
          }
          if ((a0.kind == 1 || a0.kind == 2) && a1.kind == 0) {
            int64_t lim = a0.kind == 1 ? H : W;
            int64_t rhsBound = a1.c - a0.c;
            int64_t lo = 0, hi = lim;
            if (pred == arith::CmpIPredicate::sge)
              lo = std::max<int64_t>(0, rhsBound);
            else if (pred == arith::CmpIPredicate::sgt)
              lo = std::max<int64_t>(0, rhsBound + 1);
            else if (pred == arith::CmpIPredicate::sle)
              hi = std::min<int64_t>(lim, rhsBound + 1);
            else if (pred == arith::CmpIPredicate::slt)
              hi = std::min<int64_t>(lim, rhsBound);
            else
              return false;
            if (lo >= hi)
              rects[cmp.getResult()] = {0, 0, 0, 0};
            else if (a0.kind == 1)
              rects[cmp.getResult()] = {lo, hi, 0, W};
            else
              rects[cmp.getResult()] = {0, H, lo, hi};
            return true;
          }
          return true;
        }
        if (n == "arith.andi") {
          if (rects.contains(op.getOperand(0)) &&
              rects.contains(op.getOperand(1)))
            rects[op.getResult(0)] =
                intersectRect(rects[op.getOperand(0)], rects[op.getOperand(1)]);
          return true;
        }
        if (auto iff = dyn_cast<scf::IfOp>(op)) {
          if (&op == sampleIf.getOperation())
            return true;
          if (iff.getNumResults() != 1 || iff.getElseRegion().empty() ||
              !rects.contains(iff.getCondition()))
            return false;
          for (Operation &sub : iff.thenBlock()->without_terminator())
            if (!evalAffOp(sub))
              return false;
          for (Operation &sub : iff.elseBlock()->without_terminator())
            if (!evalAffOp(sub))
              return false;
          Value tY = iff.thenBlock()->getTerminator()->getOperand(0);
          Value eY = iff.elseBlock()->getTerminator()->getOperand(0);
          if (!rects.contains(tY) || !rects.contains(eY) || !rects[eY].empty())
            return false;
          rects[iff.getResult(0)] =
              intersectRect(rects[iff.getCondition()], rects[tY]);
          return true;
        }
        return true;
      };

      for (Operation &bop : lBody.without_terminator())
        if (!evalAffOp(bop))
          return std::nullopt;
      if (!rects.contains(sampleIf.getCondition()))
        return std::nullopt;
      for (Operation &tOp : sampleIf.thenBlock()->without_terminator())
        if (!isa<memref::LoadOp>(tOp) && !evalAffOp(tOp))
          return std::nullopt;

      Value ihV = xLoad.getIndices()[xRank - 2];
      Value iwV = xLoad.getIndices()[xRank - 1];
      if (!aff.contains(ihV) || !aff.contains(iwV) || aff[ihV].kind != 1 ||
          aff[iwV].kind != 2)
        return std::nullopt;
      Rect2D validR = rects[sampleIf.getCondition()];
      int64_t dh = aff[ihV].c, dw = aff[iwV].c;
      if (!validR.empty()) {
        if (validR.hLo + dh < 0 || validR.hHi - 1 + dh >= H ||
            validR.wLo + dw < 0 || validR.wHi - 1 + dw >= W)
          return std::nullopt;
        maxAbsDh = std::max<int64_t>(maxAbsDh, std::abs(dh));
      }
      plan.taps.push_back(
          {k, dh, dw, validR.hLo, validR.hHi, validR.wLo, validR.wHi});
    }

    plan.padH = maxAbsDh;
    // Count distinct horizontal windows (dw, owLo, owHi).
    SmallVector<std::tuple<int64_t, int64_t, int64_t>, 8> horizKeys;
    for (const auto &tp : plan.taps) {
      if (tp.ohLo >= tp.ohHi || tp.owLo >= tp.owHi)
        continue;
      if (tp.dw == 0 && tp.owLo == 0 && tp.owHi == W)
        continue;
      std::tuple<int64_t, int64_t, int64_t> key{tp.dw, tp.owLo, tp.owHi};
      if (!llvm::is_contained(horizKeys, key))
        horizKeys.push_back(key);
    }
    int64_t paddedPlaneBytes = (H + 2 * plan.padH) * WP * 4;
    int64_t totalStencilNram =
        (1 + int64_t(horizKeys.size())) * llvm::alignTo(paddedPlaneBytes, 128) +
        2 * llvm::alignTo(H * WP * 4, 128);
    auto supply = profile.services.get<intentdsl::dsa::LocalSupply>();
    if (!supply ||
        totalStencilNram > std::min(supply->budget, supply->capacity))
      return std::nullopt;

    return plan;
  }

  LogicalResult lowerPlaneStencil(PlaneStencilPlan plan) {
    auto axes = space.getAttrOfType<DenseI64ArrayAttr>("axes");
    auto &bb = space.getRegion(0).front();
    unsigned na = axes.size();
    int64_t H = plan.H, W = plan.W, WP = plan.WP, padH = plan.padH;
    int64_t owners = space.getAttrOfType<IntegerAttr>("block.owners").getInt();
    int64_t numPlanes = owners / (H * W);
    int64_t paddedRows = H + 2 * padH;

    Value xZero = nramBuf(b.getF32Type(), paddedRows * WP);
    Value outBuf = nramBuf(b.getF32Type(), H * WP);
    Value tmpBuf = nramBuf(b.getF32Type(), H * WP);

    SmallVector<std::tuple<int64_t, int64_t, int64_t>, 8> horizKeys;
    SmallVector<Value, 8> horizBufs;
    for (const auto &tp : plan.taps) {
      if (tp.ohLo >= tp.ohHi || tp.owLo >= tp.owHi)
        continue;
      if (tp.dw == 0 && tp.owLo == 0 && tp.owHi == W)
        continue;
      std::tuple<int64_t, int64_t, int64_t> key{tp.dw, tp.owLo, tp.owHi};
      if (!llvm::is_contained(horizKeys, key)) {
        horizKeys.push_back(key);
        horizBufs.push_back(nramBuf(b.getF32Type(), paddedRows * WP));
      }
    }

    // Zero-fill padded borders once per core before iterating over planes.
    fillVec(xZero, cI64(paddedRows * WP), cF32(0.0f));
    for (Value hb : horizBufs)
      fillVec(hb, cI64(paddedRows * WP), cF32(0.0f));
    sync("compute");

    // Collect loop-body ops needed to evaluate weightVal for a given iv = k.
    SmallVector<Operation *, 16> weightSlice;
    {
      llvm::DenseSet<Operation *> inSlice;
      std::function<void(Value)> collect = [&](Value v) {
        Operation *def = v.getDefiningOp();
        if (!def || def->getBlock() != plan.loop.getBody() ||
            !inSlice.insert(def).second)
          return;
        for (Value u : def->getOperands())
          collect(u);
        weightSlice.push_back(def);
      };
      collect(plan.weightVal);
    }

    forLoop(symTaskId, cI64(numPlanes), symTaskDim, [&](Value planeIdx) {
      GeneralState st;
      for (Value arg : fn.getArguments())
        st.env[arg] = arg;
      int64_t pitch = 1;
      for (int i = int(na) - 3; i >= 0; --i) {
        int64_t ext = std::max<int64_t>(
            1, dimAttr(space.getOperand(i), "block.shape", axes[i]));
        Value coord = b.create<arith::RemSIOp>(
            loc, b.create<arith::DivSIOp>(loc, planeIdx, cI64(pitch)),
            cI64(ext));
        st.env[bb.getArgument(i)] = coord;
        pitch *= ext;
      }

      // Evaluate outer plane-invariant ops in bb (e.g. bias[c]).
      for (Operation &op : bb.without_terminator()) {
        if (&op == plan.loop.getOperation() || isa<memref::StoreOp>(op))
          continue;
        if (llvm::any_of(op.getOperands(),
                         [&](Value u) { return !st.env.count(u); }))
          continue;
        if (auto ld = dyn_cast<memref::LoadOp>(op)) {
          auto [ptr, idx] =
              generalScalarPtr(op, ld.getMemref(), ld.getIndices(), st);
          st.env[ld.getResult()] = loadAt(ptr, idx, ld.getType());
        } else if (Value sv = lowerScalarExpr(op, st.env)) {
          st.env[op.getResult(0)] = sv;
        }
      }

      // Stage x[outer..., :, :] into xZero[padH : padH + H, 0 : W].
      SmallVector<Value, 4> xCoords, xStrides;
      for (auto [ax, cv] : llvm::enumerate(plan.xOuterCoords)) {
        xCoords.push_back(st.env.lookup(cv));
        xStrides.push_back(cI64(dimAttr(plan.xView, "block.strides", ax)));
      }
      Value xGlobal = makeAddr(plan.xView, xCoords, xStrides, 0);
      Value xCenterDst = makeAddr(xZero, {cI64(padH)}, {cI64(WP)}, 1);
      if (H == 1)
        memcpy1D(xCenterDst, xGlobal, cI64(W), "GDRAM2NRAM");
      else
        memcpy2D(xCenterDst, xGlobal, cI64(W), cI64(WP), cI64(W),
                 cI64(H - 1), "GDRAM2NRAM");
      sync("io");

      // Populate horizontally shifted NRAM planes via 2D NRAM2NRAM DMA.
      for (auto [hk, hBuf] : llvm::zip(horizKeys, horizBufs)) {
        auto [dw, owLo, owHi] = hk;
        int64_t width = owHi - owLo;
        Value dstSh = makeAddr(hBuf, {cI64(padH), cI64(owLo)},
                               {cI64(WP), cI64(1)}, 1);
        Value srcSh = makeAddr(xZero, {cI64(padH), cI64(owLo + dw)},
                               {cI64(WP), cI64(1)}, 1);
        if (H == 1)
          memcpy1D(dstSh, srcSh, cI64(width), "NRAM2NRAM");
        else
          memcpy2D(dstSh, srcSh, cI64(width), cI64(WP), cI64(WP),
                   cI64(H - 1), "NRAM2NRAM");
      }
      if (!horizKeys.empty())
        sync("io");

      Value initVal = plan.biasVal ? st.env.lookup(plan.biasVal) : cF32(0.0f);
      fillVec(outBuf, cI64(H * WP), initVal);
      sync("compute");

      for (const auto &tp : plan.taps) {
        if (tp.ohLo >= tp.ohHi || tp.owLo >= tp.owHi)
          continue;
        st.env[plan.loop.getInductionVar()] = cI64(tp.k);
        for (Operation *wOp : weightSlice) {
          if (auto ld = dyn_cast<memref::LoadOp>(*wOp)) {
            auto [ptr, idx] =
                generalScalarPtr(*wOp, ld.getMemref(), ld.getIndices(), st);
            st.env[ld.getResult()] = loadAt(ptr, idx, ld.getType());
          } else if (Value sv = lowerScalarExpr(*wOp, st.env)) {
            st.env[wOp->getResult(0)] = sv;
          }
        }
        Value wScalar = st.env.lookup(plan.weightVal);
        Value srcBase = xZero;
        if (!(tp.dw == 0 && tp.owLo == 0 && tp.owHi == W)) {
          std::tuple<int64_t, int64_t, int64_t> key{tp.dw, tp.owLo, tp.owHi};
          auto it = llvm::find(horizKeys, key);
          srcBase = horizBufs[it - horizKeys.begin()];
        }
        int64_t rowCount = tp.ohHi - tp.ohLo;
        int64_t lanes = rowCount * WP;
        Value srcPtr =
            makeAddr(srcBase, {cI64(tp.ohLo + padH + tp.dh)}, {cI64(WP)}, 1);
        Value dstPtr = makeAddr(outBuf, {cI64(tp.ohLo)}, {cI64(WP)}, 1);
        makeOp("vec_scalar", {tmpBuf, srcPtr, wScalar, cI64(lanes)}, {},
               {b.getNamedAttr("op", b.getStringAttr("mul"))});
        makeOp("vec_binary", {dstPtr, dstPtr, tmpBuf, cI64(lanes)}, {},
               {b.getNamedAttr("op", b.getStringAttr("add"))});
      }
      sync("compute");

      SmallVector<Value, 4> oCoords, oStrides;
      for (auto [ax, cv] : llvm::enumerate(plan.outOuterCoords)) {
        oCoords.push_back(st.env.lookup(cv));
        oStrides.push_back(cI64(dimAttr(plan.outView, "block.strides", ax)));
      }
      Value oGlobal = makeAddr(plan.outView, oCoords, oStrides, 0);
      if (H == 1)
        memcpy1D(oGlobal, outBuf, cI64(W), "NRAM2GDRAM");
      else
        memcpy2D(oGlobal, outBuf, cI64(W), cI64(W), cI64(WP), cI64(H - 1),
                 "NRAM2GDRAM");
    });
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
    bool wantsVecPool = false;
    space.walk([&](Operation *op) {
      if (op->getName().stripDialect() == "lane_sum") {
        auto im = op->getAttrOfType<StringAttr>("bangc.vector_impl");
        if (!im || (im != "reduce32-reuse-product" &&
                    im != "deferred-row-reduction"))
          needsReduce = true;
      }
    });
    for (Operation &topOp : bb.without_terminator()) {
      if (auto lOp = dyn_cast<scf::ForOp>(topOp))
        if (canVectorizeStreamLoop(lOp))
          wantsVecPool = true;
    }
    int64_t usedNram = 0;
    if (needsReduce) {
      st.poolScratch = nramBuf(b.getF32Type(), 32);
      st.scalarOut = nramBuf(b.getF32Type(), 32);
      usedNram += 256;
    }
    if (memberBytes) {
      st.storage = nramBuf(b.getI8Type(), memberBytes);
      usedNram += llvm::alignTo(memberBytes, 128);
    }
    if (streamAttr && streamAttr.getInt()) {
      st.streambuf = nramBuf(b.getI8Type(), streamAttr.getInt());
      usedNram += llvm::alignTo(streamAttr.getInt(), 128);
    }
    if (defAttr && defAttr.getInt()) {
      st.deferred = nramBuf(b.getF32Type(), defAttr.getInt() / 4);
      usedNram += llvm::alignTo(defAttr.getInt(), 128);
    }
    if (slabBytes) {
      Type slabElem =
          cast<MemRefType>(fn.getArgument(0).getType()).getElementType();
      int64_t elemB = slabElem.getIntOrFloatBitWidth() / 8;
      st.slabLocal = nramBuf(slabElem, slabBytes / elemB);
      usedNram += llvm::alignTo(slabBytes, 128);
    }
    if (wantsVecPool) {
      auto supply = profile.services.get<intentdsl::dsa::LocalSupply>();
      int64_t limit = supply ? std::min(supply->budget, supply->capacity) : 0;
      if (usedNram + llvm::alignTo(kVecStreamPoolBytes, 128) <= limit)
        st.vecPool = nramBuf(b.getI8Type(), kVecStreamPoolBytes);
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
    std::optional<PlaneStencilPlan> stencilPlan;
    if (transfers != "panel-fold" && pack <= 1)
      stencilPlan = matchPlaneStencil();
    int64_t stencilTile = stencilPlan ? stencilPlan->H * stencilPlan->W : 1;
    int64_t launchTile = std::max<int64_t>({tile, pack, panelM, stencilTile});

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
                       : stencilPlan             ? lowerPlaneStencil(*stencilPlan)
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

    if (!orderedTailGuards.empty()) {
      compilerinfra::IntegerRangeContext tailRanges(fn.getOperation());
      for (auto &tg : orderedTailGuards) {
        if (compilerinfra::proveTailAccess(tg.accessOp, &tailRanges).kind ==
            compilerinfra::TailKind::ExactInBounds) {
          Value cond = tg.guardIf->getOperand(0);
          tg.guardIf->erase();
          if (cond.use_empty())
            if (Operation *def = cond.getDefiningOp())
              def->erase();
        }
      }
    }

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
  if (failed(lowerer.run(m)))
    return failure();
  stampScfCarryIds(m);
  return success();
}
} // namespace CI_UNIT_NAMESPACE
