#include "OwnerVectors.h"
#include "Rows.h"
#include "intentdsl/exits/bangc/ir/Dialect.h"
#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "intentdsl/levels/execution/block/ir/FoldPanel.h"
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

  void memcpy1D(Value dst, Value src, Value elems, StringRef dir,
                bool isAsync = false) {
    makeOp("memcpy_1d", {dst, src, elems}, {},
           {b.getNamedAttr("direction", b.getStringAttr(dir)),
            b.getNamedAttr("async", b.getBoolAttr(isAsync))});
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
    if (auto ld = dyn_cast<memref::LoadOp>(op)) {
      SmallVector<Value> cs, ss;
      for (auto [i, v] : llvm::enumerate(ld.getIndices())) {
        cs.push_back(env.lookup(v));
        ss.push_back(cI64(dimAttr(ld.getMemref(), "block.strides", i)));
      }
      return loadAt(makeAddr(ld.getMemref(), cs, ss, 0), cI64(0));
    }
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
    auto constI64 = [](Value v) -> std::optional<int64_t> {
      if (!v || !v.getType().isSignlessInteger(64))
        return std::nullopt;
      if (auto cst = v.getDefiningOp<arith::ConstantOp>())
        if (auto ia = dyn_cast<IntegerAttr>(cst.getValue()))
          return ia.getInt();
      return std::nullopt;
    };
    if (n == "arith.addi") {
      if (auto c0 = constI64(a(0)), c1 = constI64(a(1)); c0 && c1)
        return cI64(int64_t(uint64_t(*c0) + uint64_t(*c1)));
      return b.create<arith::AddIOp>(loc, a(0), a(1));
    }
    if (n == "arith.subi") {
      if (auto c0 = constI64(a(0)), c1 = constI64(a(1)); c0 && c1)
        return cI64(int64_t(uint64_t(*c0) - uint64_t(*c1)));
      return b.create<arith::SubIOp>(loc, a(0), a(1));
    }
    if (n == "arith.muli") {
      if (auto c0 = constI64(a(0)), c1 = constI64(a(1)); c0 && c1)
        return cI64(int64_t(uint64_t(*c0) * uint64_t(*c1)));
      return b.create<arith::MulIOp>(loc, a(0), a(1));
    }
    if (n == "arith.divsi") {
      if (auto c0 = constI64(a(0)), c1 = constI64(a(1)); c0 && c1 && *c1 > 0)
        return cI64(*c0 / *c1);
      return b.create<arith::DivSIOp>(loc, a(0), a(1));
    }
    if (n == "arith.divui")
      return b.create<arith::DivUIOp>(loc, a(0), a(1));
    if (n == "arith.floordivsi") {
      if (auto c0 = constI64(a(0)), c1 = constI64(a(1)); c0 && c1 && *c1 > 0) {
        int64_t q = *c0 / *c1, r = *c0 % *c1;
        return cI64(r < 0 ? q - 1 : q);
      }
      return floorDivSI(a(0), a(1));
    }
    if (n == "arith.remsi") {
      if (auto c0 = constI64(a(0)), c1 = constI64(a(1)); c0 && c1 && *c1 > 0)
        return cI64(*c0 % *c1);
      return b.create<arith::RemSIOp>(loc, a(0), a(1));
    }
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
    Value panelWork = nramBuf(b.getF32Type(), M * S);
    Value panelScratch2 = nramBuf(b.getF32Type(), M * S);
    Value panelRedOut = nramBuf(b.getF32Type(), std::max<int64_t>(M, 64));
    bool hasStripIndex = false;
    space.walk([&](Operation *op) {
      if (op->getAttrOfType<StringAttr>("dsa.panel_role") ==
          StringAttr::get(b.getContext(), "strip_index_f32"))
        hasStripIndex = true;
    });
    Value iotaPanel =
        hasStripIndex ? nramBuf(b.getF32Type(), M * S) : Value{};
    if (iotaPanel) {
      forLoop(cI64(0), cI64(S), cI64(1), [&](Value j) {
        Value jf = b.create<arith::SIToFPOp>(
            loc, b.getF32Type(),
            b.create<arith::TruncIOp>(loc, b.getI32Type(), j));
        storeAt(iotaPanel, j, jf);
      });
      sync("all");
      for (int64_t d = 1; d < M; d *= 2) {
        int64_t copyRows = std::min<int64_t>(d, M - d);
        if (copyRows > 0) {
          memcpy1D(makeAddr(iotaPanel, {cI64(d)}, {cI64(S)}, 1), iotaPanel,
                   cI64(copyRows * S), "NRAM2NRAM");
          sync("io");
        }
      }
    }

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
        makeOp("bf16_tile",
               {dstHalf, srcF32, magScratch, cI64(count), cI64(count)}, {},
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

    unsigned rowAxis =
        space.getAttrOfType<IntegerAttr>("dsa.panel_axis").getInt();
    Value rowArg = bb.getArgument(rowAxis);
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
              llvm::DenseMap<Value, Value> transposed;
              llvm::DenseSet<Value> ownerScalars;
              llvm::DenseSet<Value> causalMaskedStrips;
              llvm::DenseSet<Value> windowZeroedStrips;
              bool curLoopHasWindow = false;
              Value curStripT0;
              Value curFullWinStrip;

              for (Value arg : fn.getArguments())
                scalarEnv[arg] = arg;
              int64_t pitch = 1;
              SmallVector<unsigned> ownerAxes{rowAxis};
              for (unsigned i = axes.size(); i-- > 0;)
                if (i != rowAxis)
                  ownerAxes.push_back(i);
              for (unsigned i : ownerAxes) {
                int64_t ext = std::max<int64_t>(
                    1, dimAttr(space.getOperand(i), "block.shape", axes[i]));
                Value coord = b.create<arith::RemSIOp>(
                    loc, b.create<arith::DivSIOp>(loc, ownerBase, cI64(pitch)),
                    cI64(ext));
                if (i != rowAxis)
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
                    Value negInf =
                        cF32(-std::numeric_limits<float>::infinity());
                    ifThenElse(
                        curFullWinStrip, {},
                        [&]() {
                          makeOp(isExp2 ? "row_exp2" : "row_exp",
                                 {dstSlot, panelWork, srcSlot, cI64(M * S),
                                  cI64(M * S)},
                                 {}, {},
                                 isExp2 ? "validated-exp2" : "validated-exp");
                        },
                        [&]() {
                          memcpy1D(panelScratch2, srcSlot, cI64(M * S),
                                   "NRAM2NRAM");
                          sync("io");
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
                            Value pRow =
                                makeAddr(panelScratch2, {r}, {cI64(S)}, 1);
                            forLoop(cI64(0), rLo, cI64(1), [&](Value j) {
                              storeAt(pRow, j, negInf);
                            });
                            forLoop(rHi, cI64(S), cI64(1), [&](Value j) {
                              storeAt(pRow, j, negInf);
                            });
                          });
                          sync("all");
                          makeOp(isExp2 ? "row_exp2" : "row_exp",
                                 {dstSlot, panelWork, panelScratch2,
                                  cI64(M * S), cI64(M * S)},
                                 {}, {},
                                 isExp2 ? "validated-exp2" : "validated-exp");
                        });
                    windowZeroedStrips.insert(eop.getResult(0));
                  } else {
                    ifThenElse(
                        curFullWinStrip, {},
                        [&]() {
                          makeOp(isExp2 ? "row_exp2" : "row_exp",
                                 {dstSlot, panelWork, srcSlot, cI64(M * S),
                                  cI64(M * S)},
                                 {}, {},
                                 isExp2 ? "validated-exp2" : "validated-exp");
                        },
                        [&]() {
                          forLoop(cI64(0), cI64(M), cI64(1),
                                  [&](Value r) { emitOne(r, activeCount); });
                        });
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

                if (role == "owner_group") {
                  scalarEnv[op.getResult(0)] = b.create<arith::DivSIOp>(
                      loc, rowBase, scalarEnv.lookup(op.getOperand(1)));
                  continue;
                }
                if (role == "owner_scalar_write") {
                  auto st = cast<memref::StoreOp>(op);
                  forLoop(cI64(0), cI64(M), cI64(1), [&](Value r) {
                    SmallVector<Value> cs, ss;
                    for (auto [i, v] : llvm::enumerate(st.getIndices())) {
                      cs.push_back(v == rowArg ? Value(b.create<arith::AddIOp>(
                                                     loc, rowBase, r))
                                               : scalarEnv.lookup(v));
                      ss.push_back(
                          cI64(dimAttr(st.getMemref(), "block.strides", i)));
                    }
                    storeAt(makeAddr(st.getMemref(), cs, ss, 0), cI64(0),
                            loadAt(slotEnv.lookup(st.getValue()), r));
                  });
                  continue;
                }
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
                  SmallVector<Value> cs, ss;
                  int64_t yStride = 0;
                  for (auto [i, v] :
                       llvm::enumerate(op.getOperands().drop_front(2))) {
                    cs.push_back(v == rowArg ? rowBase : scalarEnv.lookup(v));
                    auto stride = dimAttr(yView, "block.strides", i);
                    ss.push_back(cI64(stride));
                    if (v == rowArg)
                      yStride = stride;
                  }
                  Value yPtr = makeAddr(yView, cs, ss, 0);
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
                      if (auto r =
                              bop.getAttrOfType<StringAttr>("dsa.panel_role");
                          r && (r == "strip_slice" || r == "strip_dual_slice" ||
                                r == "strip_transpose_slice")) {
                        Value kv = bop.getOperand(0);
                        int64_t rk = cast<MemRefType>(kv.getType()).getRank();
                        K = dimAttr(kv, "block.shape", rk - 2);
                        break;
                      }
                    }
                  }
                  bool paged = llvm::any_of(
                      loop.getBody()->getOperations(), [](Operation &o) {
                        auto r = o.getAttrOfType<StringAttr>("dsa.panel_role");
                        return r && r.getValue().starts_with("paged_");
                      });
                  bool hasWindow =
                      !paged &&
                      (!matchPattern(loop.getLowerBound(), m_Zero()) ||
                       !ubIsFullDimK);
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

                  Value winMaxLo = cI64(0), winMinHi = cI64(K);
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
                        ValueRange{cI64(K), cI64(0), cI64(0), cI64(K)},
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
                          Value nMaxLo =
                              b.create<arith::MaxSIOp>(loc, wSt[2], cLo);
                          Value nMinHi =
                              b.create<arith::MinSIOp>(loc, wSt[3], cHi);
                          b.create<scf::YieldOp>(
                              loc, ValueRange{nMin, nMax, nMaxLo, nMinHi});
                        });
                    sync("all");
                    tStart = b.create<arith::MulIOp>(
                        loc,
                        b.create<arith::DivSIOp>(loc, winScan.getResult(0),
                                                 cI64(S)),
                        cI64(S));
                    tokLimit = b.create<arith::MinSIOp>(
                        loc, tokLimit, winScan.getResult(1));
                    winMaxLo = winScan.getResult(2);
                    winMinHi = winScan.getResult(3);
                  }

                  if (paged) {
                    auto bound = [&](Value v) {
                      return ownerScalars.contains(v)
                                 ? toI64(loadAt(slotEnv.lookup(v), cI64(0)))
                                 : toI64(scalarEnv.lookup(v));
                    };
                    tStart = bound(loop.getLowerBound());
                    tokLimit = bound(loop.getUpperBound());
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
                        loc,
                        b.create<arith::SubIOp>(loc, paged ? tokLimit : cI64(K),
                                                t0),
                        cI64(S));
                    Value fullNt = b.create<arith::CmpIOp>(
                        loc, arith::CmpIPredicate::eq, nt, cI64(S));
                    if (hasWindow) {
                      Value tEnd = b.create<arith::AddIOp>(loc, t0, cI64(S));
                      curFullWinStrip = b.create<arith::AndIOp>(
                          loc, fullNt,
                          b.create<arith::AndIOp>(
                              loc,
                              b.create<arith::CmpIOp>(
                                  loc, arith::CmpIPredicate::sge, t0, winMaxLo),
                              b.create<arith::CmpIOp>(
                                  loc, arith::CmpIPredicate::sle, tEnd,
                                  winMinHi)));
                    } else {
                      curFullWinStrip = fullNt;
                    }
                    Value fullCausalStrip = b.create<arith::AndIOp>(
                        loc, fullNt,
                        b.create<arith::CmpIOp>(
                            loc, arith::CmpIPredicate::sle,
                            b.create<arith::AddIOp>(loc, t0, cI64(S)),
                            b.create<arith::AddIOp>(loc, rowBase, cI64(1))));

                    for (Operation &bop : loop.getBody()->without_terminator()) {
                      auto bRoleAttr =
                          bop.getAttrOfType<StringAttr>("dsa.panel_role");
                      if (!bRoleAttr)
                        continue;
                      StringRef bRole = bRoleAttr.getValue();

                      if (bRole == "strip_index_f32") {
                        Value dst = slotAt(&bop, 0, b.getF32Type());
                        if (iotaPanel && K >= 0 && K <= 16777216) {
                          Value t0F = b.create<arith::SIToFPOp>(
                              loc, b.getF32Type(),
                              b.create<arith::TruncIOp>(loc, b.getI32Type(),
                                                        t0));
                          makeOp("vec_scalar",
                                 {dst, iotaPanel, t0F, cI64(M * S)}, {},
                                 {b.getNamedAttr("op",
                                                 b.getStringAttr("add"))});
                          sync("compute");
                        } else {
                          forLoop(cI64(0), cI64(S), cI64(1), [&](Value j) {
                            Value tj = b.create<arith::TruncIOp>(
                                loc, b.getI32Type(),
                                b.create<arith::AddIOp>(loc, t0, j));
                            storeAt(dst, j,
                                    b.create<arith::SIToFPOp>(loc, b.getF32Type(),
                                                              tj));
                          });
                          sync("all");
                          for (int64_t d = 1; d < M; d *= 2) {
                            int64_t copyRows = std::min<int64_t>(d, M - d);
                            if (copyRows > 0) {
                              memcpy1D(makeAddr(dst, {cI64(d)}, {cI64(S)}, 1),
                                       dst, cI64(copyRows * S), "NRAM2NRAM");
                              sync("io");
                            }
                          }
                        }
                        slotEnv[bop.getResult(0)] = dst;
                        continue;
                      }

                      if (bRole.starts_with("paged_")) {
                        Value view = bop.getOperand(0);
                        auto coords = bop.getOperands().drop_front();
                        int64_t width = dimAttr(view, "block.shape", 1);
                        int64_t f = dimAttr(view, "block.shape", coords.size());
                        Value dst = slotAt(&bop, 0, halfTy);
                        ifThen(b.create<arith::CmpIOp>(
                                   loc, arith::CmpIPredicate::slt, nt, cI64(S)),
                               [&] {
                                 fillVec(dst, cI64(S * f),
                                         b.create<arith::ConstantOp>(
                                             loc, b.getZeroAttr(halfTy)));
                               });
                        Value end = b.create<arith::AddIOp>(loc, t0, nt);
                        Value first =
                            b.create<arith::DivSIOp>(loc, t0, cI64(width));
                        Value last = b.create<arith::AddIOp>(
                            loc,
                            b.create<arith::DivSIOp>(
                                loc, b.create<arith::SubIOp>(loc, end, cI64(1)),
                                cI64(width)),
                            cI64(1));
                        forLoop(first, last, cI64(1), [&](Value pg) {
                          Value start =
                              b.create<arith::MulIOp>(loc, pg, cI64(width));
                          Value from = b.create<arith::MaxSIOp>(loc, t0, start);
                          Value to = b.create<arith::MinSIOp>(
                              loc, end,
                              b.create<arith::AddIOp>(loc, start, cI64(width)));
                          auto env = loopScalars;
                          env[loop.getInductionVar()] = from;
                          auto eval = [&](auto &self, Value v) -> Value {
                            if (auto found = env.lookup(v))
                              return found;
                            auto def = v.getDefiningOp();
                            for (auto operand : def->getOperands())
                              self(self, operand);
                            return env[v] = lowerScalarExpr(*def, env);
                          };
                          SmallVector<Value> cs, ss;
                          for (auto [i, v] : llvm::enumerate(coords)) {
                            cs.push_back(eval(eval, v));
                            ss.push_back(
                                cI64(dimAttr(view, "block.strides", i)));
                          }
                          auto src = makeAddr(view, cs, ss, 0);
                          auto out = makeAddr(
                              dst, {b.create<arith::SubIOp>(loc, from, t0)},
                              {cI64(f)}, 1);
                          Value len = b.create<arith::SubIOp>(loc, to, from);
                          int64_t stride = dimAttr(view, "block.strides", 1);
                          if (stride == f)
                            memcpy1D(out, src,
                                     b.create<arith::MulIOp>(loc, len, cI64(f)),
                                     "GDRAM2NRAM");
                          else
                            memcpy2D(out, src, cI64(f), cI64(f), cI64(stride),
                                     b.create<arith::SubIOp>(loc, len, cI64(1)),
                                     "GDRAM2NRAM");
                        });
                        sync("io");
                        slotEnv[bop.getResult(0)] = dst;
                        panelFeat[bop.getResult(0)] = f;
                        if (bRole == "paged_dual_slice") {
                          auto tr = slotAt(&bop, 1, halfTy);
                          makeOp("tile_transpose", {tr, dst, cI64(S), cI64(f)},
                                 {}, {}, "nram-transpose");
                          sync("compute");
                          transposed[bop.getResult(0)] = tr;
                        }
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

                      if (bRole == "strip_scalar_read") {
                        auto ld = cast<memref::LoadOp>(bop);
                        SmallVector<Value> cs, ss;
                        for (auto [i, v] : llvm::enumerate(ld.getIndices())) {
                          cs.push_back(v == loop.getInductionVar()
                                           ? t0
                                           : loopScalars.lookup(v));
                          ss.push_back(cI64(
                              dimAttr(ld.getMemref(), "block.strides", i)));
                        }
                        Value slot = slotAt(&bop, 0, b.getF32Type());
                        auto ptr = makeAddr(ld.getMemref(), cs, ss, 0);
                        fillVec(slot, cI64(S), cF32(0));
                        if (dimAttr(ld.getMemref(), "block.strides",
                                    cs.size() - 1) == 1)
                          memcpy1D(slot, ptr, nt, "GDRAM2NRAM");
                        else
                          memcpy2D(slot, ptr, cI64(1), cI64(1), ss.back(),
                                   b.create<arith::SubIOp>(loc, nt, cI64(1)),
                                   "GDRAM2NRAM");
                        forLoop(cI64(1), cI64(M), cI64(1), [&](Value r) {
                          memcpy1D(makeAddr(slot, {r}, {cI64(S)}, 1), slot,
                                   cI64(S), "NRAM2NRAM");
                        });
                        slotEnv[ld.getResult()] = slot;
                        continue;
                      }
                      if (bRole == "strip_transpose_slice" ||
                          bRole == "strip_dual_slice") {
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
                        slotEnv[bop.getResult(0)] = vSlot;
                        transposed[bop.getResult(0)] = vtSlot;
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

                        ifThenElse(
                            fullCausalStrip, {},
                            [&]() {
                              Value thenVal;
                              if (auto br = dyn_cast<scf::IfOp>(bop)) {
                                Operation *term = br.thenBlock()->getTerminator();
                                for (Operation &inOp :
                                     br.thenBlock()->without_terminator()) {
                                  bool isYielded =
                                      inOp.getResult(0) == term->getOperand(0);
                                  Value inSlot =
                                      isYielded ? dst
                                                : slotAt(&inOp, 0,
                                                         b.getF32Type());
                                  emitStripElem(inOp, inSlot, loopScalars,
                                                Value{}, cI64(S));
                                  sync("compute");
                                  slotEnv[inOp.getResult(0)] = inSlot;
                                }
                                thenVal = term->getOperand(0);
                              } else {
                                thenVal =
                                    cast<arith::SelectOp>(bop).getTrueValue();
                              }
                              Value srcS = slotEnv.lookup(thenVal);
                              if (srcS != dst) {
                                memcpy1D(dst, srcS, cI64(M * S), "NRAM2NRAM");
                                sync("io");
                              }
                            },
                            [&]() {
                              forLoop(cI64(0), cI64(M), cI64(1), [&](Value r) {
                                Value dRow = makeAddr(dst, {r}, {cI64(S)}, 1);
                                Value live = causalLive(r, t0, nt);
                                ifThenElse(
                                    b.create<arith::CmpIOp>(
                                        loc, arith::CmpIPredicate::sgt, live,
                                        cI64(0)),
                                    {},
                                    [&]() {
                                      Value thenVal;
                                      if (auto br = dyn_cast<scf::IfOp>(bop)) {
                                        for (Operation &inOp :
                                             br.thenBlock()
                                                 ->without_terminator()) {
                                          Value inSlot =
                                              slotAt(&inOp, 0, b.getF32Type());
                                          emitStripElem(inOp, inSlot,
                                                        loopScalars, r, live);
                                          sync("compute");
                                          slotEnv[inOp.getResult(0)] = inSlot;
                                        }
                                        thenVal = br.thenBlock()
                                                      ->getTerminator()
                                                      ->getOperand(0);
                                      } else {
                                        thenVal = cast<arith::SelectOp>(bop)
                                                      .getTrueValue();
                                      }
                                      Value sRow = makeAddr(
                                          slotEnv.lookup(thenVal), {r},
                                          {cI64(S)}, 1);
                                      memcpy1D(dRow, sRow, cI64(S),
                                               "NRAM2NRAM");
                                      sync("io");
                                      ifThen(b.create<arith::CmpIOp>(
                                                 loc, arith::CmpIPredicate::slt,
                                                 live, cI64(S)),
                                             [&]() {
                                               forLoop(live, cI64(S), cI64(1),
                                                       [&](Value j) {
                                                         storeAt(dRow, j,
                                                                 padVal);
                                                       });
                                               sync("all");
                                             });
                                    },
                                    [&]() {
                                      fillVec(dRow, cI64(S), padVal);
                                      sync("compute");
                                    });
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
                          ifThen(
                              b.create<arith::XOrIOp>(loc, curFullWinStrip,
                                                      cI1(true)),
                              [&]() {
                                sync("compute");
                                forLoop(cI64(0), cI64(M), cI64(1), [&](Value r) {
                                  Value sRow =
                                      makeAddr(srcSlot, {r}, {cI64(S)}, 1);
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
                                sync("all");
                              });
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
                                   sync("all");
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
                        auto dense = [&] {
                          emitPanelProduct(carrySlot, slotEnv.lookup(wVal),
                                           transposed.lookup(vVal), S, DV,
                                           true);
                        };
                        if (!hasWindow) {
                          dense();
                          continue;
                        }
                        // Zero weights alone cannot suppress a NaN or infinity
                        // from a token outside this row's logical loop bounds.
                        auto ok = ifThenElse(
                            curFullWinStrip, {b.getI1Type()},
                            [&] { b.create<scf::YieldOp>(loc, cI1(true)); },
                            [&] {
                              auto *finite = makeOp(
                                  "finite_panel",
                                  {transposed.lookup(vVal), magScratch,
                                   cI64(S * DV),
                                   cI32(halfTy.isF16() ? 0x7c00 : 0x7f80)},
                                  {b.getI1Type()}, {}, "finite-panel");
                              b.create<scf::YieldOp>(loc, finite->getResults());
                            },
                            false);
                        ifThenElse(ok.getResult(0), {}, dense, [&] {
                          forLoop(cI64(0), cI64(M), cI64(1), [&](Value r) {
                            Value lo = b.create<arith::MaxSIOp>(
                                loc, cI64(0),
                                b.create<arith::SubIOp>(loc, loadAt(loBuf, r),
                                                        t0));
                            Value hi = b.create<arith::MinSIOp>(
                                loc, nt,
                                b.create<arith::SubIOp>(loc, loadAt(hiBuf, r),
                                                        t0));
                            forLoop(cI64(0), cI64(DV), cI64(1), [&](Value f) {
                              auto out = makeAddr(carrySlot, {r, f},
                                                  {cI64(DV), cI64(1)}, 1);
                              auto lhs = makeAddr(slotEnv.lookup(wVal), {r},
                                                  {cI64(S)}, 1);
                              auto rhs = makeAddr(transposed.lookup(vVal), {f},
                                                  {cI64(S)}, 1);
                              auto sum = forLoopCarry(
                                  lo, hi, cI64(1),
                                  ValueRange{loadAt(out, cI64(0))},
                                  [&](Value j, ValueRange acc) {
                                    Value a = b.create<arith::ExtFOp>(
                                        loc, b.getF32Type(), loadAt(lhs, j));
                                    Value v = b.create<arith::ExtFOp>(
                                        loc, b.getF32Type(), loadAt(rhs, j));
                                    Value prod =
                                        b.create<arith::MulFOp>(loc, a, v);
                                    b.create<scf::YieldOp>(
                                        loc, ValueRange{b.create<arith::AddFOp>(
                                                 loc, acc.front(), prod)});
                                  });
                              storeAt(out, cI64(0), sum.getResult(0));
                            });
                          });
                        });
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
                      Value padId =
                          isMax ? cF32(-std::numeric_limits<float>::infinity())
                                : cF32(0.0f);
                      auto emitPanelReduce = [&](Value redSrc) {
                        makeOp("row_reduce",
                               {panelRedOut, panelWork, redSrc, cI64(M),
                                cI64(S), cI64(S)},
                               {},
                               {b.getNamedAttr("algebra",
                                               b.getStringAttr(isMax ? "max_num"
                                                                     : "sum"))},
                               "row-reduce");
                        if (!isMax) {
                          makeOp("vec_binary",
                                 {carrySlot, carrySlot, panelRedOut, cI64(M)},
                                 {},
                                 {b.getNamedAttr("op",
                                                 b.getStringAttr("add"))});
                          sync("compute");
                        } else {
                          sync("compute");
                          forLoop(cI64(0), cI64(M), cI64(1), [&](Value r) {
                            Value cur = loadAt(carrySlot, r);
                            Value blk = loadAt(panelRedOut, r);
                            Value nxt =
                                makeOp("scalar_math", {cur, blk},
                                       {b.getF32Type()},
                                       {b.getNamedAttr(
                                           "op", b.getStringAttr("max_num"))})
                                    ->getResult(0);
                            storeAt(carrySlot, r, nxt);
                          });
                          sync("all");
                        }
                      };

                      if ((M % 32 == 0) && (S % 32 == 0)) {
                        if (!isMax && ((hasWindow && zeroed && !masked) ||
                                       (!hasWindow && masked))) {
                          emitPanelReduce(stripSlot);
                        } else if (hasWindow) {
                          Value fullRed =
                              masked ? b.create<arith::AndIOp>(
                                           loc, curFullWinStrip,
                                           fullCausalStrip)
                                           .getResult()
                                     : curFullWinStrip;
                          ifThenElse(
                              fullRed, {},
                              [&]() { emitPanelReduce(stripSlot); },
                              [&]() {
                                memcpy1D(panelScratch2, stripSlot, cI64(M * S),
                                         "NRAM2NRAM");
                                sync("io");
                                forLoop(cI64(0), cI64(M), cI64(1), [&](Value r) {
                                  Value rLo = b.create<arith::MinSIOp>(
                                      loc, nt,
                                      b.create<arith::MaxSIOp>(
                                          loc, cI64(0),
                                          b.create<arith::SubIOp>(
                                              loc, loadAt(loBuf, r), t0)));
                                  Value hiLim =
                                      masked ? causalLive(r, t0, nt) : nt;
                                  Value rHi = b.create<arith::MinSIOp>(
                                      loc, hiLim,
                                      b.create<arith::MaxSIOp>(
                                          loc, rLo,
                                          b.create<arith::SubIOp>(
                                              loc, loadAt(hiBuf, r), t0)));
                                  Value pRow = makeAddr(panelScratch2, {r},
                                                        {cI64(S)}, 1);
                                  forLoop(cI64(0), rLo, cI64(1), [&](Value j) {
                                    storeAt(pRow, j, padId);
                                  });
                                  forLoop(rHi, cI64(S), cI64(1), [&](Value j) {
                                    storeAt(pRow, j, padId);
                                  });
                                });
                                sync("all");
                                emitPanelReduce(panelScratch2);
                              });
                        } else {
                          Value fullRed =
                              masked ? fullCausalStrip : curFullWinStrip;
                          ifThenElse(
                              fullRed, {},
                              [&]() { emitPanelReduce(stripSlot); },
                              [&]() {
                                memcpy1D(panelScratch2, stripSlot, cI64(M * S),
                                         "NRAM2NRAM");
                                sync("io");
                                forLoop(cI64(0), cI64(M), cI64(1), [&](Value r) {
                                  Value cnt =
                                      masked ? causalLive(r, t0, nt) : nt;
                                  Value pRow = makeAddr(panelScratch2, {r},
                                                        {cI64(S)}, 1);
                                  forLoop(cnt, cI64(S), cI64(1), [&](Value j) {
                                    storeAt(pRow, j, padId);
                                  });
                                });
                                sync("all");
                                emitPanelReduce(panelScratch2);
                              });
                        }
                        continue;
                      }

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
    llvm::DenseMap<Value, Value> privateRows;
    Value scanOut, scanScratch, selectVals, selectIdx, selectScratch;
    int64_t selectWidth = 0;
    int64_t slabBytes = 0;
    int64_t vecChunk = 1024;
    Value rowCacheView, rowCacheRaw, rowCacheF32, rowCacheExpShift;
    bool rowCacheExpReady = false;
    int64_t rowCacheLen = 0;
    SmallVector<Value, 4> rowCacheOuter;
    llvm::DenseMap<Value, Value> env;
    llvm::DenseMap<Operation *, Value> streams;
  };

  static int32_t streamCmpfPredCode(arith::CmpFPredicate p, bool swapped) {
    if (swapped) {
      switch (p) {
      case arith::CmpFPredicate::OGT:
        p = arith::CmpFPredicate::OLT;
        break;
      case arith::CmpFPredicate::OGE:
        p = arith::CmpFPredicate::OLE;
        break;
      case arith::CmpFPredicate::OLT:
        p = arith::CmpFPredicate::OGT;
        break;
      case arith::CmpFPredicate::OLE:
        p = arith::CmpFPredicate::OGE;
        break;
      default:
        break;
      }
    }
    switch (p) {
    case arith::CmpFPredicate::OEQ:
      return 1;
    case arith::CmpFPredicate::OGT:
      return 2;
    case arith::CmpFPredicate::OGE:
      return 3;
    case arith::CmpFPredicate::OLT:
      return 4;
    case arith::CmpFPredicate::OLE:
      return 5;
    case arith::CmpFPredicate::ONE:
      return 6;
    case arith::CmpFPredicate::UNE:
      return 7;
    default:
      return 0;
    }
  }

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
    if (N < 64 || N > 0x3fffffffLL || N % 64 != 0)
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

    llvm::DenseSet<Value> vecF32, vecHalf, vecMask, invSc;
    auto okS = [&](Value v) {
      return v.getType().isF32() &&
             (loop.isDefinedOutsideOfLoop(v) || invSc.contains(v));
    };

    int slotCount = 0;
    std::function<bool(Operation &, bool)> checkPureOp =
        [&](Operation &bop, bool inSubBlock) -> bool {
      StringRef fnName = bop.getName().getStringRef();
      if (isa<arith::ConstantOp>(bop) && bop.getResult(0).getType().isF32()) {
        invSc.insert(bop.getResult(0));
        return true;
      }
      if (!inSubBlock && fnName == "arith.extf" &&
          bop.getResult(0).getType().isF32() &&
          vecHalf.contains(bop.getOperand(0))) {
        vecF32.insert(bop.getResult(0));
        ++slotCount;
        return true;
      }
      if (!inSubBlock && fnName == "arith.truncf" &&
          (bop.getResult(0).getType().isBF16() ||
           bop.getResult(0).getType().isF16()) &&
          vecF32.contains(bop.getOperand(0))) {
        vecHalf.insert(bop.getResult(0));
        ++slotCount;
        return true;
      }
      if ((fnName == "math.exp" || fnName == "math.exp2" ||
           fnName == "math.log") &&
          vecF32.contains(bop.getOperand(0))) {
        vecF32.insert(bop.getResult(0));
        slotCount += (fnName == "math.log" && inSubBlock) ? 2 : 1;
        return true;
      }
      if (auto cmpf = dyn_cast<arith::CmpFOp>(bop)) {
        Value u0 = cmpf.getLhs(), u1 = cmpf.getRhs();
        bool v0 = vecF32.contains(u0), v1 = vecF32.contains(u1);
        if ((v0 && (v1 || okS(u1))) || (okS(u0) && v1)) {
          if (streamCmpfPredCode(cmpf.getPredicate(), !v0 && v1) != 0) {
            vecMask.insert(cmpf.getResult());
            ++slotCount;
            return true;
          }
        }
        return false;
      }
      if (auto sel = dyn_cast<arith::SelectOp>(bop)) {
        if (sel.getType().isF32() && vecMask.contains(sel.getCondition())) {
          Value tV = sel.getTrueValue(), fV = sel.getFalseValue();
          if ((vecF32.contains(tV) || okS(tV)) &&
              (vecF32.contains(fV) || okS(fV)) &&
              (vecF32.contains(tV) || vecF32.contains(fV))) {
            vecF32.insert(sel.getResult());
            ++slotCount;
            return true;
          }
        }
        return false;
      }
      if ((fnName == "arith.addf" || fnName == "arith.subf" ||
           fnName == "arith.mulf") &&
          bop.getResult(0).getType().isF32()) {
        Value u0 = bop.getOperand(0), u1 = bop.getOperand(1);
        if ((vecF32.contains(u0) && vecF32.contains(u1)) ||
            (vecF32.contains(u0) && okS(u1)) ||
            (okS(u0) && vecF32.contains(u1))) {
          vecF32.insert(bop.getResult(0));
          ++slotCount;
          return true;
        }
      }
      return false;
    };

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
        ++slotCount;
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
        if (iff.getNumResults() == 1 && iff.getResult(0).getType().isF32() &&
            !iff.getElseRegion().empty() &&
            vecMask.contains(iff.getCondition())) {
          for (Operation &tOp : iff.thenBlock()->without_terminator())
            if (!checkPureOp(tOp, true))
              return false;
          for (Operation &eOp : iff.elseBlock()->without_terminator())
            if (!checkPureOp(eOp, true))
              return false;
          Value tY = iff.thenBlock()->getTerminator()->getOperand(0);
          Value eY = iff.elseBlock()->getTerminator()->getOperand(0);
          if ((vecF32.contains(tY) || okS(tY)) &&
              (vecF32.contains(eY) || okS(eY)) &&
              (vecF32.contains(tY) || vecF32.contains(eY))) {
            vecF32.insert(iff.getResult(0));
            ++slotCount;
            continue;
          }
          return false;
        }
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
        if (!(o0 == baseV && okS(o1)) &&
            !(isa<arith::AddFOp>(adjOp) && o1 == baseV && okS(o0)))
          return false;
        vecF32.insert(iff.getResult(0));
        ++slotCount;
        continue;
      }
      if (checkPureOp(bop, false))
        continue;
      return false;
    }

    if (slotCount > 10)
      return false;
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
    int64_t chunk = std::min<int64_t>(st.vecChunk, N);
    bool isRed = loop.getNumResults() == 1;
    Block &body = *loop.getBody();
    auto *yieldOp = body.getTerminator();
    Operation *redComb =
        isRed ? yieldOp->getOperand(0).getDefiningOp() : nullptr;

    // Partition vecPool: slots 0..9 = 10 work vectors (chunk * 4B each),
    // slot 10 = exp/reduce/narrow scratch, tail = scalarOut (128B).
    auto workSlot = [&](int idx, Type elemTy) {
      return nramPtr(st.vecPool, cI64((idx % 10) * st.vecChunk * 4), elemTy);
    };
    Value expRedScratch =
        nramPtr(st.vecPool, cI64(10 * st.vecChunk * 4), b.getF32Type());
    Value narrowScratch =
        nramPtr(st.vecPool, cI64(10 * st.vecChunk * 4), b.getBF16Type());
    Value redScalarOut =
        nramPtr(st.vecPool, cI64(11 * st.vecChunk * 4), b.getF32Type());

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

    auto matchesRowCache = [&](Value view, ValueRange indices) -> bool {
      if (!st.rowCacheView || view != st.rowCacheView || st.rowCacheLen < N)
        return false;
      auto outer = indices.drop_back();
      if (outer.size() != st.rowCacheOuter.size())
        return false;
      for (size_t i = 0; i < outer.size(); ++i)
        if (outer[i] != st.rowCacheOuter[i])
          return false;
      return true;
    };

    // Single-call full-row reduction when reducing directly over rowCacheF32.
    if (isRed && !st.rowCacheExpReady && st.rowCacheView &&
        N == st.rowCacheLen) {
      Value iterV = loop.getRegionIterArgs()[0];
      Value rhs = redComb->getOperand(0) == iterV ? redComb->getOperand(1)
                                                  : redComb->getOperand(0);
      memref::LoadOp redLd;
      if (auto l = rhs.getDefiningOp<memref::LoadOp>())
        redLd = l;
      else if (rhs.getDefiningOp() &&
               rhs.getDefiningOp()->getName().getStringRef() == "arith.extf")
        redLd = rhs.getDefiningOp()->getOperand(0).getDefiningOp<memref::LoadOp>();
      int otherOps = 0;
      for (Operation &bop : body.without_terminator())
        if (&bop != redComb && &bop != redLd.getOperation() &&
            &bop != rhs.getDefiningOp())
          ++otherOps;
      if (redLd && otherOps == 0 &&
          matchesRowCache(redLd.getMemref(), redLd.getIndices())) {
        Value initVal = st.env.lookup(loop.getInitArgs()[0]);
        bool isMax = isa<arith::MaxNumFOp>(redComb);
        makeOp("row_reduce",
               {redScalarOut, expRedScratch, st.rowCacheF32, cI64(1), cI64(N),
                cI64(N)},
               {},
               {b.getNamedAttr("algebra",
                               b.getStringAttr(isMax ? "max_num" : "sum"))},
               "row-reduce");
        Value blk = loadAt(redScalarOut, cI64(0));
        st.env[loop.getResult(0)] =
            isMax ? makeOp("scalar_math", {initVal, blk}, {b.getF32Type()},
                           {b.getNamedAttr("op", b.getStringAttr("max_num"))})
                        ->getResult(0)
                  : b.create<arith::AddFOp>(loc, initVal, blk).getResult();
        return true;
      }
    }

    auto matchShiftedRowExp = [&](scf::ForOp lp, Value &shiftOut,
                                  Operation *&subOut,
                                  Operation *&expOut) -> bool {
      memref::LoadOp hitLd;
      int viewRefs = 0;
      lp.getBody()->walk([&](Operation *sub) {
        if (llvm::is_contained(sub->getOperands(), st.rowCacheView)) {
          ++viewRefs;
          if (auto l = dyn_cast<memref::LoadOp>(sub))
            hitLd = l;
        }
      });
      if (viewRefs != 1 || !hitLd ||
          !matchesRowCache(hitLd.getMemref(), hitLd.getIndices()))
        return false;
      Value rowF32 = hitLd.getResult();
      if (!rowF32.getType().isF32()) {
        if (!rowF32.hasOneUse())
          return false;
        Operation *ext = *rowF32.getUsers().begin();
        if (ext->getName().getStringRef() != "arith.extf" ||
            !ext->getResult(0).getType().isF32())
          return false;
        rowF32 = ext->getResult(0);
      }
      if (!rowF32.hasOneUse())
        return false;
      auto subOp = dyn_cast<arith::SubFOp>(*rowF32.getUsers().begin());
      if (!subOp || subOp.getLhs() != rowF32 ||
          !lp.isDefinedOutsideOfLoop(subOp.getRhs()) ||
          !subOp.getResult().hasOneUse())
        return false;
      Operation *eOp = *subOp.getResult().getUsers().begin();
      if (eOp->getName().getStringRef() != "math.exp")
        return false;
      shiftOut = subOp.getRhs();
      subOut = subOp.getOperation();
      expOut = eOp;
      return true;
    };

    Operation *reuseSubOp = nullptr, *reuseExpOp = nullptr;
    Operation *writeCacheExpOp = nullptr;
    Value writeCacheShift;
    if (st.rowCacheExpReady && st.rowCacheView && N == st.rowCacheLen) {
      Value sh;
      Operation *sOp = nullptr, *eOp = nullptr;
      if (matchShiftedRowExp(loop, sh, sOp, eOp) && sh == st.rowCacheExpShift) {
        reuseSubOp = sOp;
        reuseExpOp = eOp;
      }
    } else if (!st.rowCacheExpReady && st.rowCacheView && N == st.rowCacheLen &&
               loop->getBlock() == &space.getRegion(0).front()) {
      Value sh;
      Operation *sOp = nullptr, *eOp = nullptr;
      if (matchShiftedRowExp(loop, sh, sOp, eOp)) {
        bool allAfterMatch = true;
        int reuseCount = 0;
        for (Operation *cur = loop->getNextNode();
             allAfterMatch && cur && !cur->hasTrait<OpTrait::IsTerminator>();
             cur = cur->getNextNode()) {
          bool touchesView = false;
          cur->walk([&](Operation *w) {
            if (llvm::is_contained(w->getOperands(), st.rowCacheView))
              touchesView = true;
          });
          if (!touchesView)
            continue;
          auto nextLp = dyn_cast<scf::ForOp>(cur);
          Value sh2;
          Operation *s2 = nullptr, *e2 = nullptr;
          if (nextLp && canVectorizeStreamLoop(nextLp) &&
              matchShiftedRowExp(nextLp, sh2, s2, e2) && sh2 == sh) {
            ++reuseCount;
          } else {
            allAfterMatch = false;
          }
        }
        if (allAfterMatch && reuseCount > 0) {
          writeCacheExpOp = eOp;
          writeCacheShift = sh;
        }
      }
    }

    auto emitChunkBody = [&](Value c0, Value curLanes,
                             llvm::DenseMap<Value, Value> &vecEnv) {
      llvm::DenseMap<Value, Value> scEnv = st.env;
      llvm::DenseMap<Value, Value> maskEnv;
      llvm::DenseMap<Value, Value> preWidened;
      int nextSlot = 0;

      auto emitBitSelect = [&](Value mask, Value tV, Value fV) -> Value {
        Value dst = workSlot(nextSlot++, b.getF32Type());
        bool tv = vecEnv.contains(tV), fv = vecEnv.contains(fV);
        if (tv && fv) {
          makeOp("vec_bitselect",
                 {dst, expRedScratch, mask, vecEnv.lookup(tV),
                  vecEnv.lookup(fV), curLanes},
                 {}, {b.getNamedAttr("mode", b.getStringAttr("vv"))},
                 "ieee-vec-bitselect");
        } else if (tv) {
          makeOp("vec_bitselect",
                 {dst, expRedScratch, mask, vecEnv.lookup(tV),
                  scEnv.lookup(fV), curLanes},
                 {}, {b.getNamedAttr("mode", b.getStringAttr("vs"))},
                 "ieee-vec-bitselect");
        } else {
          makeOp("vec_bitselect",
                 {dst, expRedScratch, mask, scEnv.lookup(tV),
                  vecEnv.lookup(fV), curLanes},
                 {}, {b.getNamedAttr("mode", b.getStringAttr("sv"))},
                 "ieee-vec-bitselect");
        }
        return dst;
      };

      std::function<void(Operation &, Value)> emitPureOp =
          [&](Operation &bop, Value activeMask) {
        if (auto cst = dyn_cast<arith::ConstantOp>(bop)) {
          scEnv[cst.getResult()] = lowerScalarExpr(bop, scEnv);
          return;
        }
        StringRef fnName = bop.getName().getStringRef();
        if (fnName == "arith.extf") {
          Value inV = bop.getOperand(0);
          if (preWidened.contains(inV)) {
            vecEnv[bop.getResult(0)] = preWidened.lookup(inV);
            return;
          }
          Type fromTy = inV.getType();
          Value dst = workSlot(nextSlot++, b.getF32Type());
          Value src = vecEnv.lookup(inV);
          if (fromTy.isBF16())
            makeOp("bf16_tile", {dst, src, curLanes, curLanes}, {},
                   {b.getNamedAttr("mode", b.getStringAttr("widen"))},
                   "bf16-tile-convert");
          else
            makeOp("vec_cast", {dst, src, curLanes}, {},
                   {b.getNamedAttr("kind", b.getStringAttr("f16_to_f32"))});
          vecEnv[bop.getResult(0)] = dst;
          return;
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
          vecEnv[bop.getResult(0)] = dst;
          return;
        }
        if (&bop == reuseSubOp)
          return;
        if (&bop == reuseExpOp) {
          vecEnv[bop.getResult(0)] =
              makeAddr(st.rowCacheF32, {c0}, {cI64(1)}, 1);
          return;
        }
        if (fnName == "math.exp" || fnName == "math.exp2") {
          bool isExp2 = fnName == "math.exp2";
          Value dst = (&bop == writeCacheExpOp)
                          ? makeAddr(st.rowCacheF32, {c0}, {cI64(1)}, 1)
                          : workSlot(nextSlot++, b.getF32Type());
          Value src = vecEnv.lookup(bop.getOperand(0));
          makeOp(isExp2 ? "row_exp2" : "row_exp",
                 {dst, expRedScratch, src, curLanes, curLanes}, {}, {},
                 isExp2 ? "validated-exp2" : "validated-exp");
          vecEnv[bop.getResult(0)] = dst;
          return;
        }
        if (fnName == "math.log") {
          Value src = vecEnv.lookup(bop.getOperand(0));
          if (activeMask) {
            Value safeSrc = workSlot(nextSlot++, b.getF32Type());
            makeOp("vec_bitselect",
                   {safeSrc, expRedScratch, activeMask, src, cF32(1.0f),
                    curLanes},
                   {}, {b.getNamedAttr("mode", b.getStringAttr("vs"))},
                   "ieee-vec-bitselect");
            src = safeSrc;
          }
          Value dst = workSlot(nextSlot++, b.getF32Type());
          makeOp("row_log", {dst, expRedScratch, src, curLanes, curLanes}, {},
                 {}, "validated-log");
          vecEnv[bop.getResult(0)] = dst;
          return;
        }
        if (auto cmpf = dyn_cast<arith::CmpFOp>(bop)) {
          Value u0 = cmpf.getLhs(), u1 = cmpf.getRhs();
          bool v0 = vecEnv.contains(u0), v1 = vecEnv.contains(u1);
          bool swapped = (!v0 && v1);
          int32_t pc = streamCmpfPredCode(cmpf.getPredicate(), swapped);
          Value lhsVec = vecEnv.lookup(swapped ? u1 : u0);
          bool rhsIsVec = (v0 && v1);
          Value rhsOp = rhsIsVec ? vecEnv.lookup(u1)
                                 : scEnv.lookup(swapped ? u0 : u1);
          Value mDst = workSlot(nextSlot++, b.getI32Type());
          makeOp("vec_cmpf_mask",
                 {mDst, expRedScratch, lhsVec, rhsOp, curLanes}, {},
                 {b.getNamedAttr("pred_code", b.getI32IntegerAttr(pc)),
                  b.getNamedAttr("rhs_is_vec", b.getBoolAttr(rhsIsVec))},
                 "ieee-vec-cmpf");
          maskEnv[cmpf.getResult()] = mDst;
          return;
        }
        if (auto sel = dyn_cast<arith::SelectOp>(bop)) {
          vecEnv[sel.getResult()] = emitBitSelect(
              maskEnv.lookup(sel.getCondition()), sel.getTrueValue(),
              sel.getFalseValue());
          return;
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
        vecEnv[bop.getResult(0)] = dst;
      };

      for (Operation &bop : body.without_terminator()) {
        if (&bop == redComb || isa<arith::CmpIOp>(bop))
          continue;
        if (auto ld = dyn_cast<memref::LoadOp>(bop)) {
          if (matchesRowCache(ld.getMemref(), ld.getIndices())) {
            if (ld.getType().isF32()) {
              vecEnv[ld.getResult()] =
                  makeAddr(st.rowCacheF32, {c0}, {cI64(1)}, 1);
            } else {
              if (st.rowCacheRaw)
                vecEnv[ld.getResult()] =
                    makeAddr(st.rowCacheRaw, {c0}, {cI64(1)}, 1);
              preWidened[ld.getResult()] =
                  makeAddr(st.rowCacheF32, {c0}, {cI64(1)}, 1);
            }
            continue;
          }
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
          if (maskEnv.contains(iff.getCondition())) {
            Value mVal = maskEnv.lookup(iff.getCondition());
            for (Operation &tOp : iff.thenBlock()->without_terminator())
              emitPureOp(tOp, mVal);
            for (Operation &eOp : iff.elseBlock()->without_terminator())
              emitPureOp(eOp, Value{});
            Value tY = iff.thenBlock()->getTerminator()->getOperand(0);
            Value eY = iff.elseBlock()->getTerminator()->getOperand(0);
            vecEnv[iff.getResult(0)] = emitBitSelect(mVal, tY, eY);
            continue;
          }
          auto cmp = iff.getCondition().getDefiningOp<arith::CmpIOp>();
          Value targetIdx = cmp.getLhs() == loop.getInductionVar()
                                ? scEnv.lookup(cmp.getRhs())
                                : scEnv.lookup(cmp.getLhs());
          Value baseV = iff.elseBlock()->getTerminator()->getOperand(0);
          bool canInPlace = llvm::all_of(baseV.getUsers(), [&](Operation *u) {
            return iff->isProperAncestor(u);
          });
          Value dst = canInPlace ? vecEnv.lookup(baseV)
                                 : workSlot(nextSlot++, b.getF32Type());
          if (!canInPlace) {
            sync("compute");
            memcpy1D(dst, vecEnv.lookup(baseV), curLanes, "NRAM2NRAM");
            sync("io");
          }
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
          ifThen(inChunk, [&]() {
            Value rel32 = b.create<arith::SubIOp>(loc, t32, c0_32);
            Value rel64 = b.create<arith::ExtUIOp>(loc, b.getI64Type(), rel32);
            sync("all");
            Value cur = loadAt(dst, rel64, b.getF32Type());
            Value upd = isa<arith::SubFOp>(adjOp)
                            ? b.create<arith::SubFOp>(loc, cur, deltaV)
                                  .getResult()
                            : b.create<arith::AddFOp>(loc, cur, deltaV)
                                  .getResult();
            storeAt(dst, rel64, upd);
            sync("all");
          });
          vecEnv[iff.getResult(0)] = dst;
          continue;
        }
        emitPureOp(bop, Value{});
      }
    };

    if (isRed) {
      Value initVal = st.env.lookup(loop.getInitArgs()[0]);
      bool isMax = isa<arith::MaxNumFOp>(redComb);
      Value iterV = loop.getRegionIterArgs()[0];
      Value rhs = redComb->getOperand(0) == iterV ? redComb->getOperand(1)
                                                  : redComb->getOperand(0);
      auto cLoop = forLoopCarry(
          cI64(0), cI64(N), cI64(chunk), ValueRange{initVal},
          [&](Value c0, ValueRange cIter) {
            Value curLanes = b.create<arith::MinSIOp>(
                loc, b.create<arith::SubIOp>(loc, cI64(N), c0), cI64(chunk));
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
      forLoop(cI64(0), cI64(N), cI64(chunk), [&](Value c0) {
        Value curLanes = b.create<arith::MinSIOp>(
            loc, b.create<arith::SubIOp>(loc, cI64(N), c0), cI64(chunk));
        llvm::DenseMap<Value, Value> vecEnv;
        emitChunkBody(c0, curLanes, vecEnv);
      });
    }
    if (writeCacheExpOp) {
      sync("compute");
      st.rowCacheExpReady = true;
      st.rowCacheExpShift = writeCacheShift;
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
    if (Value cache = st.privateRows.lookup(view))
      return {cache, st.env.lookup(at.back())};
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
    int64_t sp = 0;
    if (auto mt = dyn_cast<MemRefType>(base.getType()))
      if (auto ms = dyn_cast_or_null<IntegerAttr>(mt.getMemorySpace()))
        sp = ms.getInt();
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
        intentdsl::block::OrderedSelectionInfo sel;
        if (st.selectScratch &&
            loop->getBlock() == &space.getRegion(0).front() &&
            intentdsl::block::matchOrderedSelectionChain(loop, &sel)) {
          auto ld = sel.candidateVal.getDefiningOp<memref::LoadOp>();
          if (ld && ld.getIndices().back() == loop.getInductionVar() &&
              st.privateRows.count(ld.getMemref())) {
            makeOp("ordered_select",
                   {st.selectVals, st.selectIdx,
                    st.privateRows.lookup(ld.getMemref()), st.selectScratch,
                    arg(1), cI64(sel.k), cI64(st.selectWidth)},
                   {}, {}, "ordered-selection");
            for (unsigned i = 0; i < sel.k; ++i) {
              st.env[loop.getResult(i)] = loadAt(st.selectVals, cI64(i));
              st.env[loop.getResult(i + sel.k)] = loadAt(st.selectIdx, cI64(i));
            }
            continue;
          }
        }
        intentdsl::block::ExclusiveScanLoopInfo scan;
        if (st.scanScratch && loop->getBlock() == &space.getRegion(0).front() &&
            intentdsl::block::matchExclusiveScanLoop(loop, fn, nullptr,
                                                     &scan) &&
            loop.getResult(0).getType().isInteger(32)) {
          auto n = scan.extent;
          Value src = st.privateRows.lookup(scan.srcLoad.getMemref());
          makeOp("scan_sum",
                 {st.scanOut, src, st.scanScratch, cI64(n), cI64(n)}, {}, {},
                 "inclusive-scan");
          auto seed = st.env.lookup(loop.getInitArgs()[0]);
          makeOp("vec_scalar", {st.scanOut, st.scanOut, seed, cI64(n)}, {},
                 {b.getNamedAttr("op", b.getStringAttr("add"))});
          for (auto [store, inclusive] : scan.stores) {
            Value dst = st.privateRows.lookup(store.getMemref());
            if (inclusive)
              memcpy1D(dst, st.scanOut, cI64(n), "NRAM2NRAM");
            else {
              storeAt(dst, cI64(0), seed);
              memcpy1D(makeAddr(dst, {cI64(1)}, {cI64(1)}, 1), st.scanOut,
                       cI64(n - 1), "NRAM2NRAM");
            }
          }
          st.env[loop.getResult(0)] = loadAt(st.scanOut, cI64(n - 1));
          continue;
        }
        if (st.privateRows.empty() && tryEmitVectorStreamLoop(loop, st))
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
          if (!write && !op.hasAttr("dsa.stream") && st.rowCacheView &&
              base == st.rowCacheView && elemTy.isF32() &&
              lanes == st.rowCacheLen &&
              axis == cast<MemRefType>(base.getType()).getRank() - 1) {
            auto outerOps = op.getOperands().drop_front(1);
            bool sameOuter = outerOps.size() == st.rowCacheOuter.size();
            for (size_t i = 0; sameOuter && i < outerOps.size(); ++i)
              if (outerOps[i] != st.rowCacheOuter[i])
                sameOuter = false;
            if (sameOuter) {
              st.env[v] = st.rowCacheF32;
              continue;
            }
          }
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
    // Count distinct partial horizontal windows (owLo, owHi) != (0, W).
    SmallVector<std::pair<int64_t, int64_t>, 8> partialWins;
    for (const auto &tp : plan.taps) {
      if (tp.ohLo >= tp.ohHi || tp.owLo >= tp.owHi)
        continue;
      if (std::abs(tp.dw) > 64)
        return std::nullopt;
      if (tp.owLo == 0 && tp.owHi == W)
        continue;
      std::pair<int64_t, int64_t> key{tp.owLo, tp.owHi};
      if (!llvm::is_contained(partialWins, key))
        partialWins.push_back(key);
    }
    int64_t xPadStride =
        llvm::alignTo(64 + (H + 2 * plan.padH) * W + 64, 64);
    int64_t outStride = llvm::alignTo(H * W + 64, 64);
    int64_t totalStencilNram =
        2 * llvm::alignTo(xPadStride * 4, 128) +
        (3 + (partialWins.empty() ? 0 : 1) + int64_t(partialWins.size())) *
            llvm::alignTo(outStride * 4, 128) +
        512;
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
    int64_t H = plan.H, W = plan.W, padH = plan.padH;
    int64_t owners = space.getAttrOfType<IntegerAttr>("block.owners").getInt();
    int64_t numPlanes = owners / (H * W);
    int64_t leadFloats = 64;
    int64_t xPadStride = llvm::alignTo(leadFloats + (H + 2 * padH) * W + 64, 64);
    int64_t outStride = llvm::alignTo(H * W + 64, 64);
    int64_t alignedHW = llvm::alignTo(H * W, 64);

    Value xPadStorage = nramBuf(b.getF32Type(), 2 * xPadStride);
    Value outStorage = nramBuf(b.getF32Type(), 2 * outStride);
    Value tmpBuf = nramBuf(b.getF32Type(), outStride);
    Value maskScratch = nramBuf(b.getI32Type(), outStride);

    SmallVector<std::pair<int64_t, int64_t>, 8> partialWins;
    SmallVector<Value, 8> winMasks;
    SmallVector<int64_t, 8> remCounts;
    SmallVector<Value, 8> remMasks;
    for (const auto &tp : plan.taps) {
      if (tp.ohLo >= tp.ohHi || tp.owLo >= tp.owHi)
        continue;
      if (!(tp.owLo == 0 && tp.owHi == W)) {
        std::pair<int64_t, int64_t> key{tp.owLo, tp.owHi};
        if (!llvm::is_contained(partialWins, key)) {
          partialWins.push_back(key);
          winMasks.push_back(nramBuf(b.getI32Type(), outStride));
        }
      }
      int64_t rem = ((tp.ohHi - tp.ohLo) * W) % 64;
      if (rem > 0 && !llvm::is_contained(remCounts, rem)) {
        remCounts.push_back(rem);
        remMasks.push_back(nramBuf(b.getI32Type(), 64));
      }
    }
    Value winAccBuf =
        !partialWins.empty() ? nramBuf(b.getF32Type(), outStride) : Value{};

    // Zero-fill padded ping-pong input borders and initialize masks once per core.
    fillVec(xPadStorage, cI64(2 * xPadStride), cF32(0.0f));
    fillVec(outStorage, cI64(2 * outStride), cF32(0.0f));
    sync("compute");

    for (auto [pw, mBuf] : llvm::zip(partialWins, winMasks)) {
      auto [owLo, owHi] = pw;
      fillVec(mBuf, cI64(outStride), cI32(0));
      sync("compute");
      forLoop(cI64(0), cI64(W), cI64(1), [&](Value col) {
        Value ok = b.create<arith::AndIOp>(
            loc,
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, col,
                                    cI64(owLo)),
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, col,
                                    cI64(owHi)));
        storeAt(mBuf, col,
                b.create<arith::SelectOp>(loc, ok, cI32(-1), cI32(0)));
      });
      sync("all");
      for (int64_t d = 1; d < H; d *= 2) {
        int64_t copyRows = std::min<int64_t>(d, H - d);
        if (copyRows > 0) {
          memcpy1D(makeAddr(mBuf, {cI64(d * W)}, {cI64(1)}, 1), mBuf,
                   cI64(copyRows * W), "NRAM2NRAM");
          sync("io");
        }
      }
    }
    for (auto [rem, rBuf] : llvm::zip(remCounts, remMasks)) {
      forLoop(cI64(0), cI64(64), cI64(1), [&](Value k) {
        Value ok =
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, k,
                                    cI64(rem));
        storeAt(rBuf, k, b.create<arith::SelectOp>(loc, ok, cI32(-1), cI32(0)));
      });
      sync("all");
    }

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

    // Pre-stage small immutable "In" parameter views (weights, bias) into NRAM once!
    llvm::DenseMap<Value, Value> paramCache;
    if (auto abiParsed = llvm::json::parse(
            fn->getAttrOfType<StringAttr>("kir.abi").getValue())) {
      if (auto *abiArr = abiParsed->getAsArray()) {
        auto supply = profile.services.get<intentdsl::dsa::LocalSupply>();
        int64_t remBudget =
            (supply ? std::min(supply->budget, supply->capacity) : 0) -
            (2 * xPadStride * 4 + 6 * outStride * 4 + 2048);
        auto tryStageParam = [&](memref::LoadOp ld) {
          Value v = ld.getMemref();
          auto ba = dyn_cast<BlockArgument>(v);
          if (!ba || ba.getOwner() != &fn.front() || v == plan.xView ||
              v == plan.outView || paramCache.contains(v))
            return;
          auto *obj = (*abiArr)[ba.getArgNumber()].getAsObject();
          if (!obj || obj->getString("direction").value_or("") != "In")
            return;
          auto mt = cast<MemRefType>(v.getType());
          int64_t rnk = mt.getRank();
          if (rnk < 1 || dimAttr(v, "block.strides", rnk - 1) != 1)
            return;
          int64_t totalElems =
              std::max<int64_t>(1, dimAttr(v, "block.shape", 0)) *
              std::max<int64_t>(1, dimAttr(v, "block.strides", 0));
          int64_t elemBytes = mt.getElementTypeBitWidth() / 8;
          if (totalElems <= 0 || totalElems > 16384 || elemBytes <= 0 ||
              totalElems * elemBytes > remBudget)
            return;
          remBudget -= llvm::alignTo(totalElems * elemBytes, 128);
          Value pBuf =
              nramBuf(mt.getElementType(), llvm::alignTo(totalElems, 64));
          memcpy1D(pBuf, v, cI64(totalElems), "GDRAM2NRAM");
          paramCache[v] = pBuf;
        };
        for (Operation &op : bb.without_terminator())
          if (auto ld = dyn_cast<memref::LoadOp>(op))
            tryStageParam(ld);
        for (Operation *wOp : weightSlice)
          if (auto ld = dyn_cast<memref::LoadOp>(*wOp))
            tryStageParam(ld);
        if (!paramCache.empty())
          sync("io");
      }
    } else {
      llvm::consumeError(abiParsed.takeError());
    }

    auto bindOuterCoords = [&](Value pIdx, llvm::DenseMap<Value, Value> &env) {
      int64_t pitch = 1;
      for (int i = int(na) - 3; i >= 0; --i) {
        int64_t ext = std::max<int64_t>(
            1, dimAttr(space.getOperand(i), "block.shape", axes[i]));
        Value divVal =
            pitch == 1 ? pIdx
            : llvm::isPowerOf2_64(uint64_t(pitch))
                ? b.create<arith::ShRUIOp>(
                       loc, pIdx, cI64(llvm::Log2_64(uint64_t(pitch))))
                      .getResult()
                : b.create<arith::DivSIOp>(loc, pIdx, cI64(pitch)).getResult();
        Value coord =
            ext == 1 ? cI64(0)
            : llvm::isPowerOf2_64(uint64_t(ext))
                ? b.create<arith::AndIOp>(loc, divVal, cI64(ext - 1))
                      .getResult()
                : b.create<arith::RemSIOp>(loc, divVal, cI64(ext)).getResult();
        env[bb.getArgument(i)] = coord;
        pitch *= ext;
      }
    };

    auto planeViewPtr = [&](Value pIdx, Value view, ArrayRef<Value> outerC) {
      llvm::DenseMap<Value, Value> tmpEnv;
      bindOuterCoords(pIdx, tmpEnv);
      SmallVector<Value, 4> coords, strides;
      for (auto [ax, cv] : llvm::enumerate(outerC)) {
        coords.push_back(tmpEnv.lookup(cv));
        strides.push_back(cI64(dimAttr(view, "block.strides", ax)));
      }
      return makeAddr(view, coords, strides, 0);
    };

    auto xCenterSlot = [&](Value bufIdx) {
      Value off = b.create<arith::AddIOp>(
          loc, b.create<arith::MulIOp>(loc, bufIdx, cI64(xPadStride)),
          cI64(leadFloats + padH * W));
      return makeAddr(xPadStorage, {off}, {cI64(1)}, 1);
    };
    auto outSlot = [&](Value bufIdx) {
      Value off = b.create<arith::MulIOp>(loc, bufIdx, cI64(outStride));
      return makeAddr(outStorage, {off}, {cI64(1)}, 1);
    };

    Value remPlanes = b.create<arith::MaxSIOp>(
        loc, cI64(0), b.create<arith::SubIOp>(loc, cI64(numPlanes), symTaskId));
    Value numIters = b.create<arith::DivSIOp>(
        loc,
        b.create<arith::SubIOp>(
            loc, b.create<arith::AddIOp>(loc, remPlanes, symTaskDim), cI64(1)),
        symTaskDim);

    ifThen(
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, numIters,
                                cI64(0)),
        [&]() {
          memcpy1D(xCenterSlot(cI64(0)),
                   planeViewPtr(symTaskId, plan.xView, plan.xOuterCoords),
                   cI64(H * W), "GDRAM2NRAM");
          sync("io");

          forLoop(cI64(0), numIters, cI64(1), [&](Value pIter) {
            Value planeIdx = b.create<arith::AddIOp>(
                loc, symTaskId,
                b.create<arith::MulIOp>(loc, pIter, symTaskDim));
            Value nextPlane =
                b.create<arith::AddIOp>(loc, planeIdx, symTaskDim);
            Value curIdx = b.create<arith::AndIOp>(loc, pIter, cI64(1));
            Value otherIdx = b.create<arith::SubIOp>(loc, cI64(1), curIdx);

            // Overlap BOTH next-plane GDRAM read and prev-plane GDRAM write with compute!
            ifThen(b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                           nextPlane, cI64(numPlanes)),
                   [&]() {
                     memcpy1D(xCenterSlot(otherIdx),
                              planeViewPtr(nextPlane, plan.xView,
                                           plan.xOuterCoords),
                              cI64(H * W), "GDRAM2NRAM", /*isAsync=*/true);
                   });
            ifThen(b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                           pIter, cI64(0)),
                   [&]() {
                     Value prevPlane =
                         b.create<arith::SubIOp>(loc, planeIdx, symTaskDim);
                     memcpy1D(planeViewPtr(prevPlane, plan.outView,
                                           plan.outOuterCoords),
                              outSlot(otherIdx), cI64(H * W), "NRAM2GDRAM",
                              /*isAsync=*/true);
                   });

            GeneralState st;
            for (Value arg : fn.getArguments())
              st.env[arg] = paramCache.contains(arg) ? paramCache.lookup(arg)
                                                     : arg;
            bindOuterCoords(planeIdx, st.env);

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

            Value curOut = outSlot(curIdx);
            Value curXBaseOff =
                b.create<arith::MulIOp>(loc, curIdx, cI64(xPadStride));
            Value initVal =
                plan.biasVal ? st.env.lookup(plan.biasVal) : cF32(0.0f);
            fillVec(curOut, cI64(alignedHW), initVal);

            auto evalTapWeight = [&](int64_t k) -> Value {
              st.env[plan.loop.getInductionVar()] = cI64(k);
              for (Operation *wOp : weightSlice) {
                if (auto ld = dyn_cast<memref::LoadOp>(*wOp)) {
                  auto [ptr, idx] = generalScalarPtr(*wOp, ld.getMemref(),
                                                     ld.getIndices(), st);
                  st.env[ld.getResult()] = loadAt(ptr, idx, ld.getType());
                } else if (Value sv = lowerScalarExpr(*wOp, st.env)) {
                  st.env[wOp->getResult(0)] = sv;
                }
              }
              return st.env.lookup(plan.weightVal);
            };

            auto emitTapAccum = [&](const auto &tp, Value targetAcc,
                                    bool overwrite) {
              Value wScalar = evalTapWeight(tp.k);
              int64_t exactLanes = (tp.ohHi - tp.ohLo) * W;
              int64_t full64 = (exactLanes / 64) * 64;
              int64_t rem64 = exactLanes - full64;
              int64_t alignedLanes = llvm::alignTo(exactLanes, 64);
              int64_t relSrc =
                  leadFloats + (tp.ohLo + padH + tp.dh) * W + tp.dw;
              Value srcPtr = makeAddr(
                  xPadStorage,
                  {b.create<arith::AddIOp>(loc, curXBaseOff, cI64(relSrc))},
                  {cI64(1)}, 1);
              Value dstPtr =
                  makeAddr(targetAcc, {cI64(tp.ohLo * W)}, {cI64(1)}, 1);
              if (overwrite) {
                makeOp("vec_scalar",
                       {dstPtr, srcPtr, wScalar, cI64(alignedLanes)}, {},
                       {b.getNamedAttr("op", b.getStringAttr("mul"))});
                return;
              }
              makeOp("vec_scalar",
                     {tmpBuf, srcPtr, wScalar, cI64(alignedLanes)}, {},
                     {b.getNamedAttr("op", b.getStringAttr("mul"))});
              if (rem64 > 0) {
                auto rIt = llvm::find(remCounts, rem64);
                Value rMask = remMasks[rIt - remCounts.begin()];
                Value tailPtr =
                    makeAddr(tmpBuf, {cI64(full64)}, {cI64(1)}, 1);
                makeOp("vec_bitselect",
                       {tailPtr, maskScratch, rMask, tailPtr, cF32(0.0f),
                        cI64(64)},
                       {}, {b.getNamedAttr("mode", b.getStringAttr("vs"))},
                       "ieee-vec-bitselect");
              }
              makeOp("vec_binary",
                     {dstPtr, dstPtr, tmpBuf, cI64(alignedLanes)}, {},
                     {b.getNamedAttr("op", b.getStringAttr("add"))});
            };

            // 1. Center full-width taps (owLo == 0 && owHi == W).
            for (const auto &tp : plan.taps) {
              if (tp.ohLo >= tp.ohHi || tp.owLo >= tp.owHi)
                continue;
              if (tp.owLo == 0 && tp.owHi == W)
                emitTapAccum(tp, curOut, false);
            }

            // 2. Partial horizontal windows: accumulate taps, bitwise-mask once, add!
            for (auto [pw, wMask] : llvm::zip(partialWins, winMasks)) {
              const auto *seedTap = static_cast<const decltype(&plan.taps[0])>(nullptr);
              for (const auto &tp : plan.taps) {
                if (tp.ohLo >= tp.ohHi || tp.owLo >= tp.owHi)
                  continue;
                if (tp.owLo == pw.first && tp.owHi == pw.second &&
                    tp.ohLo == 0 && tp.ohHi == H && ((H * W) % 64 == 0)) {
                  seedTap = &tp;
                  break;
                }
              }
              if (seedTap)
                emitTapAccum(*seedTap, winAccBuf, true);
              else
                fillVec(winAccBuf, cI64(alignedHW), cF32(0.0f));
              for (const auto &tp : plan.taps) {
                if (tp.ohLo >= tp.ohHi || tp.owLo >= tp.owHi || &tp == seedTap)
                  continue;
                if (tp.owLo == pw.first && tp.owHi == pw.second)
                  emitTapAccum(tp, winAccBuf, false);
              }
              makeOp("vec_bitselect",
                     {winAccBuf, maskScratch, wMask, winAccBuf, cF32(0.0f),
                      cI64(alignedHW)},
                     {}, {b.getNamedAttr("mode", b.getStringAttr("vs"))},
                     "ieee-vec-bitselect");
              makeOp("vec_binary",
                     {curOut, curOut, winAccBuf, cI64(alignedHW)}, {},
                     {b.getNamedAttr("op", b.getStringAttr("add"))});
            }

            sync("compute");
            sync("io");
          });

          Value lastIter =
              b.create<arith::SubIOp>(loc, numIters, cI64(1));
          Value lastCur = b.create<arith::AndIOp>(loc, lastIter, cI64(1));
          Value lastPlane = b.create<arith::AddIOp>(
              loc, symTaskId,
              b.create<arith::MulIOp>(loc, lastIter, symTaskDim));
          memcpy1D(
              planeViewPtr(lastPlane, plan.outView, plan.outOuterCoords),
              outSlot(lastCur), cI64(H * W), "NRAM2GDRAM");
          sync("io");
        });
    return success();
  }

  LogicalResult
  lowerGeneral(int64_t tile,
               const std::optional<intentdsl::dsa::PrivateRows> &privatePlan) {
    auto &bb = space.getRegion(0).front();
    int64_t owners = space.getAttrOfType<IntegerAttr>("block.owners").getInt();
    int64_t slabBytes =
        space.getAttrOfType<IntegerAttr>("dsa.owner_bytes").getInt();
    int64_t memberBytes =
        space.getAttrOfType<IntegerAttr>("dsa.member_bytes").getInt();
    auto streamAttr = space.getAttrOfType<IntegerAttr>("dsa.stream_bytes");
    auto defAttr = space.getAttrOfType<IntegerAttr>("bangc.deferred_bytes");

    GeneralState st;
    st.slabBytes = slabBytes;
    if (privatePlan) {
      for (Value v : privatePlan->views)
        st.privateRows[v] =
            nramBuf(cast<MemRefType>(v.getType()).getElementType(),
                    dimAttr(v, "block.shape", 1));
      if (privatePlan->scanWidth) {
        st.scanOut = nramBuf(b.getI32Type(), privatePlan->scanWidth);
        st.scanScratch = nramBuf(b.getI32Type(), privatePlan->scanWidth);
      }
      if (privatePlan->k) {
        st.selectVals = nramBuf(b.getF32Type(), privatePlan->k);
        st.selectIdx = nramBuf(b.getI32Type(), privatePlan->k);
        st.selectWidth = privatePlan->selectionWidth;
        st.selectScratch = nramBuf(b.getI32Type(), st.selectWidth * 3);
      }
    }
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
    int64_t maxVecN = 0;
    for (Operation &topOp : bb.without_terminator()) {
      if (auto lOp = dyn_cast<scf::ForOp>(topOp)) {
        if (!privatePlan && canVectorizeStreamLoop(lOp)) {
          wantsVecPool = true;
          auto dOp = lOp.getUpperBound().getDefiningOp<memref::DimOp>();
          int64_t lN =
              dimAttr(dOp.getSource(), "block.shape", *dOp.getConstantIndex());
          maxVecN = std::max(maxVecN, lN);
        }
      }
    }
    int64_t usedNram = privatePlan ? privatePlan->bytes : 0;
    if (needsReduce) {
      st.poolScratch = nramBuf(b.getF32Type(), 32);
      st.scalarOut = nramBuf(b.getF32Type(), 32);
      usedNram += 256;
    }
    if (memberBytes) {
      st.storage = nramBuf(b.getI8Type(), memberBytes);
      usedNram += llvm::alignTo(memberBytes, 128);
    }
    bool needsStreamBuf = false;
    space.walk([&](scf::ForOp lOp) {
      if (!canVectorizeStreamLoop(lOp)) {
        for (Operation &bop : *lOp.getBody())
          if (bop.hasAttr("dsa.stream"))
            needsStreamBuf = true;
      }
    });
    if (needsStreamBuf && streamAttr && streamAttr.getInt()) {
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
      constexpr int64_t minPoolBytes = 11 * 1024 * 4 + 128;

      // Identify hot multi-pass contiguous "In" row view to stage once in NRAM.
      if (auto abiParsed = llvm::json::parse(
              fn->getAttrOfType<StringAttr>("kir.abi").getValue())) {
        if (auto *abiArr = abiParsed->getAsArray()) {
          struct CandRow {
            Value view;
            int64_t len = 0;
            SmallVector<Value, 4> outer;
            int hits = 0;
            bool needsRaw = false;
          };
          SmallVector<CandRow, 4> cands;
          auto recordRead = [&](Value v, ValueRange outer, int64_t len,
                                bool rawNeeded) {
            auto ba = dyn_cast<BlockArgument>(v);
            if (!ba || ba.getOwner() != &fn.front() || len < 64 ||
                len > 32768 || (len % 64 != 0))
              return;
            auto *obj = (*abiArr)[ba.getArgNumber()].getAsObject();
            if (!obj || obj->getString("direction").value_or("") != "In")
              return;
            Type et = cast<MemRefType>(v.getType()).getElementType();
            if (!et.isF32() && !et.isF16() && !et.isBF16())
              return;
            if (llvm::any_of(outer, [&](Value c) {
                  auto ca = dyn_cast<BlockArgument>(c);
                  return !ca || ca.getOwner() != &bb;
                }))
              return;
            for (auto &cr : cands) {
              if (cr.view == v && cr.len == len &&
                  cr.outer.size() == outer.size() &&
                  std::equal(cr.outer.begin(), cr.outer.end(), outer.begin())) {
                ++cr.hits;
                cr.needsRaw |= rawNeeded;
                return;
              }
            }
            CandRow nr;
            nr.view = v;
            nr.len = len;
            nr.outer.assign(outer.begin(), outer.end());
            nr.hits = 1;
            nr.needsRaw = rawNeeded;
            cands.push_back(std::move(nr));
          };

          for (Operation &topOp : bb.without_terminator()) {
            if (auto lOp = dyn_cast<scf::ForOp>(topOp)) {
              if (!canVectorizeStreamLoop(lOp))
                continue;
              auto dOp = lOp.getUpperBound().getDefiningOp<memref::DimOp>();
              int64_t lN = dimAttr(dOp.getSource(), "block.shape",
                                   *dOp.getConstantIndex());
              for (Operation &bop : lOp.getBody()->without_terminator()) {
                if (auto ld = dyn_cast<memref::LoadOp>(bop)) {
                  int64_t rnk =
                      cast<MemRefType>(ld.getMemref().getType()).getRank();
                  if (dimAttr(ld.getMemref(), "block.strides", rnk - 1) == 1 &&
                      dimAttr(ld.getMemref(), "block.shape", rnk - 1) == lN) {
                    bool rawNeeded =
                        !ld.getType().isF32() &&
                        llvm::any_of(ld.getResult().getUsers(),
                                     [](Operation *u) {
                                       return u->getName().getStringRef() !=
                                              "arith.extf";
                                     });
                    recordRead(ld.getMemref(), ld.getIndices().drop_back(), lN,
                               rawNeeded);
                  }
                }
              }
            } else if (topOp.getName().stripDialect() == "slice_read" &&
                       !topOp.hasAttr("dsa.stream")) {
              Value base = topOp.getOperand(0);
              int64_t axis = topOp.getAttrOfType<IntegerAttr>("axis").getInt();
              int64_t rnk = cast<MemRefType>(base.getType()).getRank();
              if (axis == rnk - 1 &&
                  dimAttr(base, "block.strides", axis) == 1) {
                int64_t lN = memberExtent(topOp.getResult(0));
                bool rawNeeded = !cast<MemRefType>(base.getType())
                                      .getElementType()
                                      .isF32();
                recordRead(base, topOp.getOperands().drop_front(1), lN,
                           rawNeeded);
              }
            }
          }

          int bestIdx = -1, bestHits = 1;
          for (size_t i = 0; i < cands.size(); ++i) {
            if (cands[i].hits > bestHits) {
              bestHits = cands[i].hits;
              bestIdx = int(i);
            }
          }
          if (bestIdx >= 0) {
            auto &best = cands[bestIdx];
            Type et = cast<MemRefType>(best.view.getType()).getElementType();
            bool allocRaw = !et.isF32() && best.needsRaw;
            int64_t f32Bytes = llvm::alignTo(best.len * 4, 128);
            int64_t rawBytes = allocRaw ? llvm::alignTo(best.len * 2, 128) : 0;
            int64_t minStagePool =
                (!et.isF32() && !allocRaw)
                    ? std::max<int64_t>(minPoolBytes, best.len * 2)
                    : minPoolBytes;
            if (usedNram + f32Bytes + rawBytes +
                    llvm::alignTo(minStagePool, 128) <=
                limit) {
              st.rowCacheView = best.view;
              st.rowCacheLen = best.len;
              st.rowCacheOuter = best.outer;
              if (allocRaw)
                st.rowCacheRaw = nramBuf(et, best.len);
              st.rowCacheF32 = nramBuf(b.getF32Type(), best.len);
              usedNram += f32Bytes + rawBytes;
            }
          }
        }
      } else {
        llvm::consumeError(abiParsed.takeError());
      }

      int64_t reqStageBytes =
          (st.rowCacheView && !st.rowCacheRaw &&
           !cast<MemRefType>(st.rowCacheView.getType())
                .getElementType()
                .isF32())
              ? st.rowCacheLen * 2
              : 0;
      for (int64_t candChunk :
           {int64_t(16384), int64_t(8192), int64_t(4096), int64_t(2048),
            int64_t(1024)}) {
        int64_t effChunk =
            maxVecN > 0 ? std::min<int64_t>(candChunk,
                                            llvm::alignTo(maxVecN, 64))
                        : candChunk;
        int64_t pBytes = std::max<int64_t>(11 * effChunk * 4 + 128,
                                           reqStageBytes);
        if (usedNram + llvm::alignTo(pBytes, 128) <= limit) {
          st.vecChunk = effChunk;
          st.vecPool = nramBuf(b.getI8Type(), pBytes);
          break;
        }
      }
    }

    auto axes = space.getAttrOfType<DenseI64ArrayAttr>("axes");
    LogicalResult status = success();
    auto emitSingleOwner = [&](Value owner) {
      st.env.clear();
      st.streams.clear();
      st.rowCacheExpReady = false;
      st.rowCacheExpShift = Value{};
      for (Value arg : fn.getArguments())
        st.env[arg] = arg;
      int64_t pitch = 1;
      for (int i = axes.size() - 1; i >= 0; --i) {
        int64_t ext = std::max<int64_t>(
            1, dimAttr(space.getOperand(i), "block.shape", axes[i]));
        Value divVal =
            pitch == 1 ? owner
            : llvm::isPowerOf2_64(uint64_t(pitch))
                ? b.create<arith::ShRUIOp>(
                       loc, owner, cI64(llvm::Log2_64(uint64_t(pitch))))
                      .getResult()
                : b.create<arith::DivSIOp>(loc, owner, cI64(pitch))
                      .getResult();
        Value coord =
            ext == 1 ? cI64(0)
            : llvm::isPowerOf2_64(uint64_t(ext))
                ? b.create<arith::AndIOp>(loc, divVal, cI64(ext - 1))
                      .getResult()
                : b.create<arith::RemSIOp>(loc, divVal, cI64(ext))
                      .getResult();
        st.env[bb.getArgument(i)] = coord;
        pitch *= ext;
      }
      if (st.rowCacheView) {
        SmallVector<Value, 4> rcC, rcS;
        for (auto [ax, cv] : llvm::enumerate(st.rowCacheOuter)) {
          rcC.push_back(st.env.lookup(cv));
          rcS.push_back(cI64(dimAttr(st.rowCacheView, "block.strides", ax)));
        }
        Value gPtr = makeAddr(st.rowCacheView, rcC, rcS, 0);
        Type et =
            cast<MemRefType>(st.rowCacheView.getType()).getElementType();
        if (et.isF32()) {
          memcpy1D(st.rowCacheF32, gPtr, cI64(st.rowCacheLen), "GDRAM2NRAM");
          sync("io");
        } else {
          Value rawStaging =
              st.rowCacheRaw ? st.rowCacheRaw
                             : nramPtr(st.vecPool, cI64(0), et);
          memcpy1D(rawStaging, gPtr, cI64(st.rowCacheLen), "GDRAM2NRAM");
          sync("io");
          if (et.isBF16())
            makeOp("bf16_tile",
                   {st.rowCacheF32, rawStaging, cI64(st.rowCacheLen),
                    cI64(st.rowCacheLen)},
                   {}, {b.getNamedAttr("mode", b.getStringAttr("widen"))},
                   "bf16-tile-convert");
          else
            makeOp("vec_cast",
                   {st.rowCacheF32, rawStaging, cI64(st.rowCacheLen)}, {},
                   {b.getNamedAttr("kind", b.getStringAttr("f16_to_f32"))});
          sync("compute");
        }
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
      auto moveRow = [&](Value v, bool store) {
        auto global =
            makeAddr(v, {owner}, {cI64(dimAttr(v, "block.strides", 0))}, 0);
        auto nr = st.privateRows.lookup(v);
        memcpy1D(store ? global : nr, store ? nr : global,
                 cI64(dimAttr(v, "block.shape", 1)),
                 store ? "NRAM2GDRAM" : "GDRAM2NRAM");
      };
      if (privatePlan)
        for (auto v : privatePlan->views)
          moveRow(v, false);
      if (failed(emitGeneralBlock(bb, st)))
        status = failure();
      if (privatePlan)
        for (auto v : privatePlan->views)
          if (privatePlan->writes.contains(v))
            moveRow(v, true);
      if (slabBytes) {
        sync("compute");
        memcpy1D(slabGlobal, st.slabLocal, cI64(slabElems), "NRAM2GDRAM");
      }
    };
    forLoop(symTaskId, symGrid, symTaskDim, [&](Value job) {
      if (tile <= 1) {
        emitSingleOwner(job);
      } else {
        Value baseOwner = b.create<arith::MulIOp>(loc, job, cI64(tile));
        Value endOwner = b.create<arith::MinSIOp>(
            loc, cI64(owners),
            b.create<arith::AddIOp>(loc, baseOwner, cI64(tile)));
        forLoop(baseOwner, endOwner, cI64(1), emitSingleOwner);
      }
    });
    return status;
  }

  LogicalResult lowerIndexedRowAdd(const intentdsl::dsa::IndexedRowAdd &p) {
    auto elem = cast<MemRefType>(p.source.getType()).getElementType();
    auto keys = nramBuf(b.getI32Type(), p.batch);
    auto rows = nramBuf(elem, p.batch * p.width);
    auto old = nramBuf(elem, p.width);
    forLoop(symTaskId, symGrid, symTaskDim, [&](Value job) {
      Value first = b.create<arith::MulIOp>(loc, job, cI64(p.batch));
      Value live = b.create<arith::MinSIOp>(
          loc, cI64(p.batch),
          b.create<arith::SubIOp>(loc, cI64(p.rows), first));
      memcpy1D(keys, makeAddr(p.keys, {first}, {cI64(1)}, 0), live,
               "GDRAM2NRAM");
      Value count = b.create<arith::MulIOp>(loc, live, cI64(p.width));
      memcpy1D(rows, makeAddr(p.source, {first}, {cI64(p.width)}, 0), count,
               "GDRAM2NRAM");
      forLoop(cI64(0), live, cI64(1), [&](Value i) {
        Value target = toI64(loadAt(keys, i));
        auto dest = makeAddr(p.destination, {target}, {cI64(p.width)}, 0);
        auto delta = makeAddr(rows, {i}, {cI64(p.width)}, 1);
        makeOp("atomic_add", {old, dest, delta, cI64(p.width)}, {}, {},
               "native-row-atomic-add");
      });
    });
    return success();
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
    auto indexed = transfers == "indexed-row-add"
                       ? intentdsl::dsa::indexedRowAdd(fn, profile)
                       : std::nullopt;
    if (transfers == "indexed-row-add" && !indexed)
      return failure();
    std::optional<PlaneStencilPlan> stencilPlan;
    if (!indexed && transfers != "panel-fold" && pack <= 1)
      stencilPlan = matchPlaneStencil();
    auto privatePlan = transfers == "private-rows"
                           ? intentdsl::dsa::privateRows(fn, profile)
                           : std::nullopt;
    if (transfers == "private-rows" && !privatePlan)
      return failure();
    bool isGeneral = transfers != "panel-fold" && pack <= 1 && !stencilPlan;
    int64_t stencilTile = stencilPlan ? stencilPlan->H * stencilPlan->W : 1;
    int64_t launchTile =
        isGeneral ? 1 : std::max<int64_t>({tile, pack, panelM, stencilTile});

    if (indexed) {
      owners = indexed->rows;
      launchTile = indexed->batch;
    }
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

    LogicalResult st = indexed ? lowerIndexedRowAdd(*indexed)
                       : transfers == "panel-fold" ? lowerPanelFold()
                       : pack > 1                  ? lowerPackedStreams(pack)
                       : stencilPlan ? lowerPlaneStencil(*stencilPlan)
                                     : lowerGeneral(launchTile, privatePlan);
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
