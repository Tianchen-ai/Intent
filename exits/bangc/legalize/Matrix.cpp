#include "Matrix.h"
#include "Rows.h"
#include "intentdsl/exits/bangc/ir/Dialect.h"
#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "intentdsl/levels/algorithm/kir/ir/PanelMap.h"
#include "intentdsl/levels/algorithm/kir/ir/Storage.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/JSON.h"
#include <limits>
#include <numeric>
#include <set>

namespace CI_UNIT_NAMESPACE {
using namespace mlir;
using namespace compilerinfra;

namespace {
MemRefType memSpacePtr(MLIRContext *ctx, Type elem, int64_t space) {
  return MemRefType::get({ShapedType::kDynamic}, elem, AffineMap{},
                         IntegerAttr::get(IntegerType::get(ctx, 32), space));
}

struct MatrixLowerer {
  func::FuncOp fn;
  const TargetProfile &profile;
  intentdsl::block::MatrixAccess p;
  OpBuilder b;
  Location loc;
  Operation *constAnchor = nullptr;
  llvm::DenseMap<int64_t, Value> constI64Map;
  llvm::DenseMap<int32_t, Value> constI32Map;
  Value symGrid, symTaskId, symTaskDim, symBM, symBN, symBK;
  SmallVector<std::string> argDtypes;
  struct PendingTailGuard {
    Operation *guardIf;
    Operation *accessOp;
  };
  SmallVector<PendingTailGuard, 8> tailGuards;

  MatrixLowerer(func::FuncOp f, const TargetProfile &prof)
      : fn(f), profile(prof), p(f), b(f.getContext()), loc(f.getLoc()) {
    auto parsed =
        llvm::cantFail(llvm::json::parse(fn->getAttrOfType<StringAttr>("kir.abi")));
    for (auto &entry : *parsed.getAsArray())
      argDtypes.push_back(entry.getAsObject()->getString("dtype")->str());
  }

  StringRef abiDtype(Value v) const {
    return argDtypes[p.slot(v)];
  }

  Type carrierElemType(Value v) {
    auto et = cast<MemRefType>(v.getType()).getElementType();
    if (et.isInteger(8) && abiDtype(v) == "i8")
      return IntegerType::get(b.getContext(), 8, IntegerType::Signed);
    return et;
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
    return b.create<arith::ExtSIOp>(loc, b.getI64Type(), v).getResult();
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

  Value localVar(Type elemTy, int64_t extent) {
    return makeOp("local_var", {}, {memSpacePtr(b.getContext(), elemTy, 2)},
                  {b.getNamedAttr("extent",
                                  b.getStringAttr(std::to_string(extent)))})
        ->getResult(0);
  }

  Value makeAddr(Value base, ArrayRef<Value> coords, ArrayRef<Value> strides,
                 int64_t space, Type elemOverride = {}) {
    Type elemTy = elemOverride ? elemOverride
                               : cast<MemRefType>(base.getType()).getElementType();
    SmallVector<Value> ops{base};
    ops.append(coords.begin(), coords.end());
    ops.append(strides.begin(), strides.end());
    int32_t n = static_cast<int32_t>(coords.size());
    return makeOp("address", ops, {memSpacePtr(b.getContext(), elemTy, space)},
                  {b.getNamedAttr("index_bits", b.getI32IntegerAttr(64)),
                   b.getNamedAttr("operandSegmentSizes",
                                  b.getDenseI32ArrayAttr({1, n, n}))})
        ->getResult(0);
  }

  Value globalPtr(Value view, ArrayRef<Value> coords, Type elemOverride = {}) {
    SmallVector<Value> strides;
    for (size_t i = 0; i < coords.size(); ++i)
      strides.push_back(cI64(p.stride(view, i)));
    return makeAddr(view, coords, strides, 0,
                    elemOverride ? elemOverride : carrierElemType(view));
  }

  Value termPtr(Value v, int reduction, int batch, Value coord, Value kVal,
                Value gVal) {
    SmallVector<Value> indices;
    for (int i = 0; i < p.rank(v); ++i)
      indices.push_back(i == reduction ? kVal : i == batch ? gVal : coord);
    return globalPtr(v, indices);
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
      if (!ifOp.thenBlock()->empty() && isa<scf::YieldOp>(ifOp.thenBlock()->back()))
        ifOp.thenBlock()->back().erase();
      if (!ifOp.elseBlock()->empty() && isa<scf::YieldOp>(ifOp.elseBlock()->back()))
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

  Value inputRow(Value r, Value off) {
    Value ri = b.create<arith::AddIOp>(loc, r, off);
    if (!p.routed())
      return ri;
    Value ids = p.dot.getOperand(3), dst = p.domain.getOperand(0);
    int64_t q =
        fn.getArgAttrOfType<DenseI64ArrayAttr>(p.slot(dst), "block.shape")[1];
    Value raw = toI64(loadAt(globalPtr(ids, {ri}), cI64(0), b.getI32Type()));
    return b.create<arith::DivSIOp>(loc, raw, cI64(q));
  }

  // Lower Q8_K activation quantization and Q4_K x Q8_K record dot strictly from
  // KIR storage layout descriptors (intentdsl::kir::Q4KType / Q8KType).
  LogicalResult emitQ4KRecords(Value acc, Value r, Value nr) {
    auto q4Attr = p.dot.getAttrOfType<TypeAttr>("kir.storage");
    auto *encoder = p.dot.getOperand(1).getDefiningOp();
    auto q8Attr =
        encoder ? encoder->getAttrOfType<TypeAttr>("kir.storage") : TypeAttr{};
    if (!q4Attr || !q8Attr)
      return p.dot.emitError("missing KIR storage type on Q4_K contraction");
    auto q4 = cast<intentdsl::kir::Q4KType>(q4Attr.getValue()).getLayout();
    auto q8 = cast<intentdsl::kir::Q8KType>(q8Attr.getValue()).getLayout();

    Value w = p.dot.getOperand(0);
    Value x = encoder->getOperand(0);
    Value rec = nramBuf(b.getI8Type(), llvm::alignTo(q4.bytes, 64));
    int64_t numGroups = q4.elements / q4.group;
    Value qArr = localVar(b.getI32Type(), q8.elements);
    Value sumsArr = localVar(b.getI32Type(), numGroups);

    forLoop(cI64(0), nr, cI64(1),
            [&](Value i) { storeAt(acc, i, cF32(0.0f)); });

    int64_t recordGroups = p.shape[2] / q4.elements;
    forLoop(cI64(0), cI64(recordGroups), cI64(1), [&](Value gq) {
      // Find first signed value with maximum magnitude (GGML tie-breaking).
      auto maxLoop = forLoopCarry(
          cI64(0), cI64(q8.elements), cI64(1), ValueRange{cF32(0.0f)},
          [&](Value k, ValueRange iter) {
            Value xk = loadAt(globalPtr(x, {gq, k}), cI64(0), b.getF32Type());
            Value ax = makeOp("scalar_math", {xk}, {b.getF32Type()},
                              {b.getNamedAttr("op", b.getStringAttr("fabs"))})
                           ->getResult(0);
            Value ae =
                makeOp("scalar_math", {iter[0]}, {b.getF32Type()},
                       {b.getNamedAttr("op", b.getStringAttr("fabs"))})
                    ->getResult(0);
            Value gt =
                b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, ax, ae);
            Value next = b.create<arith::SelectOp>(loc, gt, xk, iter[0]);
            b.create<scf::YieldOp>(loc, ValueRange{next});
          });
      Value e = maxLoop.getResult(0);
      Value isZero = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, e,
                                             cF32(0.0f));
      ifThenElse(
          isZero, {},
          [&]() {
            int64_t st2 = p.stride(w, 2);
            forLoop(cI64(0), nr, cI64(1), [&](Value i) {
              Value ri = b.create<arith::AddIOp>(loc, r, i);
              Value dHiPtr = globalPtr(w, {ri, gq, cI64(q4.d + 1)});
              Value dmHiPtr = globalPtr(w, {ri, gq, cI64(q4.dmin + 1)});
              Value dHi = b.create<arith::ExtUIOp>(
                  loc, b.getI32Type(), loadAt(dHiPtr, cI64(0), b.getI8Type()));
              Value dmHi = b.create<arith::ExtUIOp>(
                  loc, b.getI32Type(), loadAt(dmHiPtr, cI64(0), b.getI8Type()));
              Value c7c = cI32(0x7c);
              Value nf0 = b.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::eq,
                  b.create<arith::AndIOp>(loc, dHi, c7c), c7c);
              Value nf1 = b.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::eq,
                  b.create<arith::AndIOp>(loc, dmHi, c7c), c7c);
              Value nf = b.create<arith::OrIOp>(loc, nf0, nf1);
              Value nanF = cF32(std::numeric_limits<float>::quiet_NaN());
              Value delta =
                  b.create<arith::SelectOp>(loc, nf, nanF, cF32(0.0f));
              storeAt(acc, i,
                      b.create<arith::AddFOp>(loc, loadAt(acc, i), delta));
              (void)st2;
            });
          },
          [&]() {
            Value maxQuant = cF32(-float((1 << (q8.elementBits - 1)) - 1));
            Value inv = b.create<arith::DivFOp>(loc, maxQuant, e);
            Value ds = b.create<arith::DivFOp>(loc, cF32(1.0f), inv);
            forLoop(cI64(0), cI64(numGroups), cI64(1),
                    [&](Value g) { storeAt(sumsArr, g, cI32(0)); });
            forLoop(cI64(0), cI64(q8.elements), cI64(1), [&](Value k) {
              Value xk = loadAt(globalPtr(x, {gq, k}), cI64(0), b.getF32Type());
              Value f = b.create<arith::MulFOp>(loc, xk, inv);
              Value fl =
                  makeOp("scalar_math", {f}, {b.getF32Type()},
                         {b.getNamedAttr("op", b.getStringAttr("floor"))})
                      ->getResult(0);
              Value lo = b.create<arith::FPToSIOp>(loc, b.getI32Type(), fl);
              Value frac = b.create<arith::SubFOp>(loc, f, fl);
              Value gtHalf = b.create<arith::CmpFOp>(
                  loc, arith::CmpFPredicate::OGT, frac, cF32(0.5f));
              Value eqHalf = b.create<arith::CmpFOp>(
                  loc, arith::CmpFPredicate::OEQ, frac, cF32(0.5f));
              Value odd = b.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::ne,
                  b.create<arith::AndIOp>(loc, lo, cI32(1)), cI32(0));
              Value bump = b.create<arith::OrIOp>(
                  loc, gtHalf, b.create<arith::AndIOp>(loc, eqHalf, odd));
              Value inc = b.create<arith::SelectOp>(loc, bump, cI32(1), cI32(0));
              Value rnd = b.create<arith::AddIOp>(loc, lo, inc);
              int32_t qMin = -(1 << (q8.elementBits - 1));
              int32_t qMax = (1 << (q8.elementBits - 1)) - 1;
              Value cl = b.create<arith::MinSIOp>(
                  loc, b.create<arith::MaxSIOp>(loc, rnd, cI32(qMin)),
                  cI32(qMax));
              storeAt(qArr, k, cl);
              Value grp = b.create<arith::DivSIOp>(loc, k, cI64(q4.group));
              storeAt(sumsArr, grp,
                      b.create<arith::AddIOp>(loc, loadAt(sumsArr, grp), cl));
            });

            int64_t st2 = p.stride(w, 2);
            forLoop(cI64(0), nr, cI64(1), [&](Value i) {
              Value ri = b.create<arith::AddIOp>(loc, r, i);
              Value base = globalPtr(w, {ri, gq, cI64(0)});
              if (st2 == 1) {
                memcpy1D(rec, base, cI64(q4.bytes), "GDRAM2NRAM");
              } else {
                forLoop(cI64(0), cI64(q4.bytes), cI64(1), [&](Value z) {
                  Value elemPtr = makeAddr(base, {z}, {cI64(st2)}, 0);
                  storeAt(rec, z, loadAt(elemPtr, cI64(0), b.getI8Type()));
                });
              }
              auto recU32 = [&](Value idx) {
                return b
                    .create<arith::ExtUIOp>(loc, b.getI32Type(),
                                            loadAt(rec, idx, b.getI8Type()))
                    .getResult();
              };
              Value dVal =
                  makeOp("from_le_half",
                         {recU32(cI64(q4.d)), recU32(cI64(q4.d + 1))},
                         {b.getF32Type()})
                      ->getResult(0);
              Value dmVal =
                  makeOp("from_le_half",
                         {recU32(cI64(q4.dmin)), recU32(cI64(q4.dmin + 1))},
                         {b.getF32Type()})
                      ->getResult(0);

              int32_t scaleMask = (1 << q4.scaleBits) - 1;
              int32_t nibMask = (1 << q4.nibbleBits) - 1;
              auto jLoop = forLoopCarry(
                  cI64(0), cI64(numGroups), cI64(1),
                  ValueRange{cI32(0), cI32(0)}, [&](Value j, ValueRange st) {
                    Value slot =
                        b.create<arith::RemSIOp>(loc, j, cI64(q4.halfGroups));
                    Value lByte = recU32(
                        b.create<arith::AddIOp>(loc, slot, cI64(q4.factors)));
                    Value hByte = recU32(b.create<arith::AddIOp>(
                        loc, slot, cI64(q4.factors + q4.halfGroups)));
                    Value pByte = recU32(b.create<arith::AddIOp>(
                        loc, slot, cI64(q4.factors + 2 * q4.halfGroups)));
                    Value firstHalf = b.create<arith::CmpIOp>(
                        loc, arith::CmpIPredicate::slt, j,
                        cI64(q4.halfGroups));
                    auto unpackFactor = [&](Value x, Value lowNib) {
                      Value hi2 = b.create<arith::ShLIOp>(
                          loc,
                          b.create<arith::ShRUIOp>(loc, x,
                                                   cI32(q4.scaleBits)),
                          cI32(q4.nibbleBits));
                      return b
                          .create<arith::SelectOp>(
                              loc, firstHalf,
                              b.create<arith::AndIOp>(loc, x, cI32(scaleMask)),
                              b.create<arith::OrIOp>(loc, lowNib, hi2))
                          .getResult();
                    };
                    Value sc = unpackFactor(
                        lByte,
                        b.create<arith::AndIOp>(loc, pByte, cI32(nibMask)));
                    Value mn = unpackFactor(
                        hByte, b.create<arith::ShRUIOp>(loc, pByte,
                                                        cI32(q4.nibbleBits)));

                    Value packBase = b.create<arith::AddIOp>(
                        loc, cI64(q4.packed),
                        b.create<arith::MulIOp>(
                            loc, b.create<arith::DivSIOp>(loc, j, cI64(2)),
                            cI64(q4.group)));
                    Value shift = b.create<arith::TruncIOp>(
                        loc, b.getI32Type(),
                        b.create<arith::MulIOp>(
                            loc, b.create<arith::RemSIOp>(loc, j, cI64(2)),
                            cI64(q4.nibbleBits)));
                    Value qBase =
                        b.create<arith::MulIOp>(loc, j, cI64(q4.group));

                    auto dotLoop = forLoopCarry(
                        cI64(0), cI64(q4.group), cI64(1), ValueRange{cI32(0)},
                        [&](Value t, ValueRange dIter) {
                          Value bVal = recU32(
                              b.create<arith::AddIOp>(loc, packBase, t));
                          Value nib = b.create<arith::AndIOp>(
                              loc, b.create<arith::ShRUIOp>(loc, bVal, shift),
                              cI32(nibMask));
                          Value qv = loadAt(
                              qArr, b.create<arith::AddIOp>(loc, qBase, t));
                          Value prod = b.create<arith::MulIOp>(loc, nib, qv);
                          b.create<scf::YieldOp>(
                              loc, ValueRange{b.create<arith::AddIOp>(
                                       loc, dIter[0], prod)});
                        });

                    Value nextS = b.create<arith::AddIOp>(
                        loc, st[0],
                        b.create<arith::MulIOp>(loc, dotLoop.getResult(0), sc));
                    Value nextT = b.create<arith::AddIOp>(
                        loc, st[1],
                        b.create<arith::MulIOp>(loc, loadAt(sumsArr, j), mn));
                    b.create<scf::YieldOp>(loc, ValueRange{nextS, nextT});
                  });

              Value sF = b.create<arith::SIToFPOp>(loc, b.getF32Type(),
                                                   jLoop.getResult(0));
              Value tF = b.create<arith::SIToFPOp>(loc, b.getF32Type(),
                                                   jLoop.getResult(1));
              Value pTerm = b.create<arith::MulFOp>(loc, dVal, sF);
              Value nTerm = b.create<arith::MulFOp>(loc, dmVal, tF);
              Value diff = b.create<arith::SubFOp>(loc, pTerm, nTerm);
              Value contrib = b.create<arith::MulFOp>(loc, ds, diff);
              storeAt(acc, i,
                      b.create<arith::AddFOp>(loc, loadAt(acc, i), contrib));
            });
          });
    });
    return success();
  }

  // Decode E4M3FN byte x and E8M0 scale byte strictly from KIR MX8Block layout.
  Value decodeMX8(Value xByte, Value scaleByte,
                  const intentdsl::kir::MX8Block &mx) {
    Value sign = b.create<arith::ShLIOp>(
        loc, b.create<arith::AndIOp>(loc, xByte, cI32(128)), cI32(24));
    Value mag = b.create<arith::AndIOp>(loc, xByte, cI32(127));
    Value m = b.create<arith::AndIOp>(loc, mag, cI32(7));
    Value e = b.create<arith::ShRUIOp>(loc, mag, cI32(3));
    Value isNanVal =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, mag, cI32(127));
    Value normBits = b.create<arith::OrIOp>(
        loc, sign,
        b.create<arith::SelectOp>(
            loc, isNanVal, cI32(0x7fc00000),
            b.create<arith::OrIOp>(
                loc,
                b.create<arith::ShLIOp>(
                    loc, b.create<arith::AddIOp>(loc, e, cI32(120)), cI32(23)),
                b.create<arith::ShLIOp>(loc, m, cI32(20)))));
    Value normF = b.create<arith::BitcastOp>(loc, b.getF32Type(), normBits);
    Value neg = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ne,
        b.create<arith::AndIOp>(loc, xByte, cI32(128)), cI32(0));
    Value sgnF =
        b.create<arith::SelectOp>(loc, neg, cF32(-1.0f), cF32(1.0f));
    Value subF = b.create<arith::MulFOp>(
        loc,
        b.create<arith::MulFOp>(
            loc, sgnF, b.create<arith::SIToFPOp>(loc, b.getF32Type(), m)),
        cF32(0.001953125f));
    Value isNorm =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, e, cI32(0));
    Value valF = b.create<arith::SelectOp>(loc, isNorm, normF, subF);

    Value scNan = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                          scaleByte, cI32(mx.nan));
    Value scZero = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                           scaleByte, cI32(0));
    int32_t zeroExpBits = 1 << (149 - mx.bias);
    Value shiftedSc = b.create<arith::ShLIOp>(
        loc, b.create<arith::AddIOp>(loc, scaleByte, cI32(127 - mx.bias)),
        cI32(23));
    Value factorBits = b.create<arith::SelectOp>(
        loc, scNan, cI32(0x7fc00000),
        b.create<arith::SelectOp>(loc, scZero, cI32(zeroExpBits), shiftedSc));
    Value factorF =
        b.create<arith::BitcastOp>(loc, b.getF32Type(), factorBits);
    return b.create<arith::MulFOp>(loc, valF, factorF);
  }

  Value emitLaneShiftScalar(Value subLane64,
                            const intentdsl::kir::QuantFormatSpec &spec) {
    Value sh = cI32(spec.laneBitShifts[0]);
    for (int idx = 1; idx < spec.packFactor; ++idx) {
      Value eq = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                         subLane64, cI64(idx));
      sh = b.create<arith::SelectOp>(loc, eq, cI32(spec.laneBitShifts[idx]), sh);
    }
    return sh;
  }

  Value decodeScalarQuantCode(Value codeI32,
                              const intentdsl::kir::QuantFormatSpec &spec) {
    if (spec.valueCode == intentdsl::kir::QuantValueCode::Fp4E2M1) {
      Value s = b.create<arith::AndIOp>(
          loc, b.create<arith::ShRUIOp>(loc, codeI32, cI32(3)), cI32(1));
      Value e = b.create<arith::AndIOp>(
          loc, b.create<arith::ShRUIOp>(loc, codeI32, cI32(1)), cI32(3));
      Value m = b.create<arith::AndIOp>(loc, codeI32, cI32(1));
      Value isSub =
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, e, cI32(0));
      Value safeShift = b.create<arith::SelectOp>(
          loc, isSub, cI32(0), b.create<arith::SubIOp>(loc, e, cI32(1)));
      Value normDouble = b.create<arith::ShLIOp>(
          loc, b.create<arith::AddIOp>(loc, m, cI32(2)), safeShift);
      Value magDouble =
          b.create<arith::SelectOp>(loc, isSub, m, normDouble);
      Value magF = b.create<arith::SIToFPOp>(loc, b.getF32Type(), magDouble);
      Value neg =
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, s, cI32(0));
      Value halfScale =
          b.create<arith::SelectOp>(loc, neg, cF32(-0.5f), cF32(0.5f));
      return b.create<arith::MulFOp>(loc, halfScale, magF);
    }
    return b.create<arith::SIToFPOp>(loc, b.getF32Type(), codeI32);
  }

  Value decodeScalarE8M0(Value scaleByte,
                         const intentdsl::kir::QuantFormatSpec &spec) {
    Value scNan = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                          scaleByte, cI32(spec.scaleNan));
    Value scZero =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, scaleByte, cI32(0));
    int32_t zeroExpBits = 1 << (149 - spec.scaleBias);
    Value shiftedSc = b.create<arith::ShLIOp>(
        loc,
        b.create<arith::AddIOp>(loc, scaleByte, cI32(127 - spec.scaleBias)),
        cI32(23));
    Value factorBits = b.create<arith::SelectOp>(
        loc, scNan, cI32(0x7fc00000),
        b.create<arith::SelectOp>(loc, scZero, cI32(zeroExpBits), shiftedSc));
    return b.create<arith::BitcastOp>(loc, b.getF32Type(), factorBits);
  }

  // Descriptor-driven quantized contraction lowering ("format as data"):
  // Handles RecordGemv, BlockScaledPair, and GroupedWeightRhs from QuantFormatSpec
  // with zero per-format branches.
  LogicalResult emitQuantizedFromDescriptor(Value acc, Value r, Value c,
                                            Value nr, Value nc, Value quantLut,
                                            Value quantRecLut,
                                            Value qActAllF32, Value dsAll,
                                            Value zeroAll) {
    const auto *spec = intentdsl::kir::quantFormatSpec(p.dot);
    if (!spec)
      return p.dot.emitError("missing quantized format descriptor");
    int64_t bm = p.tile[0], bn = p.tile[1], bk = p.tile[2];
    int64_t shapeK = p.shape[2];

    if (spec->roleLayout == intentdsl::kir::QuantRoleLayout::RecordGemv) {
      Value w = p.dot.getOperand(0);
      Value x = spec->q8kActivation
                    ? p.dot.getOperand(1).getDefiningOp()->getOperand(0)
                    : p.dot.getOperand(1);
      int64_t recordGroups = shapeK / spec->blockElements;
      int64_t xSt1 = p.stride(x, 1), wSt2 = p.stride(w, 2);

      if (quantRecLut) {
        Value actBuf = qActAllF32 ? Value{} : nramBuf(b.getF32Type(), 256);
        Value qActF32 = qActAllF32 ? Value{} : nramBuf(b.getF32Type(), 256);
        Value recPanel = nramBuf(b.getI8Type(), 64 * 256);
        Value recTrans = nramBuf(b.getI8Type(), 64 * 256);
        Value idx16Buf = nramBuf(b.getI16Type(), 1024);
        Value w16Buf = nramBuf(b.getI16Type(), 1024);
        Value h16Buf = nramBuf(b.getI16Type(), 1024);
        Value wf32Buf = nramBuf(b.getF32Type(), 1024);
        fillVec(acc, cI64(std::max<int64_t>(bm, 64)), cF32(0.0f));
        sync("compute");

        intentdsl::kir::Q8KRecord q8;
        const auto &p0 = spec->planes[0], &p1 = spec->planes[1];
        SmallVector<int32_t, 10> pp = {
            spec->subScaleOffset, spec->superScaleOffset,
            p0.baseByte,          p0.halfStride,
            p0.quadByteStep,      p0.quadShiftMode,
            p1.baseByte,          p1.halfStride,
            p1.quadByteStep,      p1.quadShiftMode};

        forLoop(cI64(0), cI64(recordGroups), cI64(1), [&](Value gq) {
          Value isZero, preDs;
          if (qActAllF32) {
            isZero = b.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::ne,
                loadAt(zeroAll, gq, b.getI32Type()), cI32(0));
            preDs = loadAt(dsAll, gq, b.getF32Type());
          } else {
            if (xSt1 == 1) {
              memcpy1D(actBuf, globalPtr(x, {gq, cI64(0)}), cI64(256),
                       "GDRAM2NRAM");
              sync("io");
            } else {
              forLoop(cI64(0), cI64(256), cI64(1), [&](Value k) {
                storeAt(actBuf, k,
                        loadAt(globalPtr(x, {gq, k}), cI64(0), b.getF32Type()));
              });
              sync("all");
            }
            auto maxLoop = forLoopCarry(
                cI64(0), cI64(256), cI64(1), ValueRange{cF32(0.0f)},
                [&](Value k, ValueRange iter) {
                  Value xk = loadAt(actBuf, k, b.getF32Type());
                  Value ax =
                      makeOp("scalar_math", {xk}, {b.getF32Type()},
                             {b.getNamedAttr("op", b.getStringAttr("fabs"))})
                          ->getResult(0);
                  Value ae =
                      makeOp("scalar_math", {iter[0]}, {b.getF32Type()},
                             {b.getNamedAttr("op", b.getStringAttr("fabs"))})
                          ->getResult(0);
                  Value gt = b.create<arith::CmpFOp>(
                      loc, arith::CmpFPredicate::OGT, ax, ae);
                  Value next =
                      b.create<arith::SelectOp>(loc, gt, xk, iter[0]);
                  b.create<scf::YieldOp>(loc, ValueRange{next});
                });
            Value e = maxLoop.getResult(0);
            isZero = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, e,
                                             cF32(0.0f));
            preDs = e;
          }
          ifThenElse(
              isZero, {},
              [&]() {
                if (spec->scaleCode ==
                    intentdsl::kir::QuantScaleCode::LeHalfAtOffset) {
                  sync("all");
                  forLoop(cI64(0), nr, cI64(1), [&](Value i) {
                    Value ri = b.create<arith::AddIOp>(loc, r, i);
                    Value dHiPtr = globalPtr(
                        w, {ri, gq, cI64(spec->superScaleOffset + 1)});
                    Value dHi = b.create<arith::ExtUIOp>(
                        loc, b.getI32Type(),
                        loadAt(dHiPtr, cI64(0), b.getI8Type()));
                    Value c7c = cI32(0x7c);
                    Value nf = b.create<arith::CmpIOp>(
                        loc, arith::CmpIPredicate::eq,
                        b.create<arith::AndIOp>(loc, dHi, c7c), c7c);
                    Value nanF = cF32(std::numeric_limits<float>::quiet_NaN());
                    Value delta =
                        b.create<arith::SelectOp>(loc, nf, nanF, cF32(0.0f));
                    storeAt(acc, i,
                            b.create<arith::AddFOp>(loc, loadAt(acc, i),
                                                    delta));
                  });
                  sync("all");
                }
              },
              [&]() {
                Value curQAct, ds;
                if (qActAllF32) {
                  curQAct = makeAddr(
                      qActAllF32,
                      {b.create<arith::MulIOp>(loc, gq, cI64(256))}, {cI64(1)},
                      1);
                  ds = preDs;
                } else {
                  Value maxQuant =
                      cF32(-float((1 << (q8.elementBits - 1)) - 1));
                  Value inv = b.create<arith::DivFOp>(loc, maxQuant, preDs);
                  ds = b.create<arith::DivFOp>(loc, cF32(1.0f), inv);
                  int32_t qMin = -(1 << (q8.elementBits - 1));
                  int32_t qMax = (1 << (q8.elementBits - 1)) - 1;
                  forLoop(cI64(0), cI64(256), cI64(1), [&](Value k) {
                    Value xk = loadAt(actBuf, k, b.getF32Type());
                    Value f = b.create<arith::MulFOp>(loc, xk, inv);
                    Value fl =
                        makeOp("scalar_math", {f}, {b.getF32Type()},
                               {b.getNamedAttr("op", b.getStringAttr("floor"))})
                            ->getResult(0);
                    Value lo =
                        b.create<arith::FPToSIOp>(loc, b.getI32Type(), fl);
                    Value frac = b.create<arith::SubFOp>(loc, f, fl);
                    Value gtHalf = b.create<arith::CmpFOp>(
                        loc, arith::CmpFPredicate::OGT, frac, cF32(0.5f));
                    Value eqHalf = b.create<arith::CmpFOp>(
                        loc, arith::CmpFPredicate::OEQ, frac, cF32(0.5f));
                    Value odd = b.create<arith::CmpIOp>(
                        loc, arith::CmpIPredicate::ne,
                        b.create<arith::AndIOp>(loc, lo, cI32(1)), cI32(0));
                    Value bump = b.create<arith::OrIOp>(
                        loc, gtHalf,
                        b.create<arith::AndIOp>(loc, eqHalf, odd));
                    Value inc =
                        b.create<arith::SelectOp>(loc, bump, cI32(1), cI32(0));
                    Value rnd = b.create<arith::AddIOp>(loc, lo, inc);
                    Value cl = b.create<arith::MinSIOp>(
                        loc, b.create<arith::MaxSIOp>(loc, rnd, cI32(qMin)),
                        cI32(qMax));
                    storeAt(qActF32, k,
                            b.create<arith::SIToFPOp>(loc, b.getF32Type(), cl));
                  });
                  sync("all");
                  curQAct = qActF32;
                }
                forLoop(cI64(0), nr, cI64(64), [&](Value sub) {
                  Value curNr = b.create<arith::MinSIOp>(
                      loc, b.create<arith::SubIOp>(loc, nr, sub), cI64(64));
                  Value rSub = b.create<arith::AddIOp>(loc, r, sub);
                  Value accSub = makeAddr(acc, {sub}, {cI64(1)}, 1);
                  Value wBase =
                      globalPtr(w, {rSub, gq, cI64(0)}, b.getI8Type());
                  ifThenElse(
                      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                              curNr, cI64(1)),
                      {},
                      [&]() {
                        memcpy1D(recPanel, wBase, cI64(spec->recordBytes),
                                 "GDRAM2NRAM");
                      },
                      [&]() {
                        memcpy2D(recPanel, wBase, cI64(spec->recordBytes),
                                 cI64(256), cI64(p.stride(w, 0)),
                                 b.create<arith::SubIOp>(loc, curNr, cI64(1)),
                                 "GDRAM2NRAM");
                      });
                  sync("io");
                  makeOp("quant_record_gemv",
                         {accSub, recPanel, recTrans, curQAct, quantRecLut,
                          idx16Buf, w16Buf, h16Buf, wf32Buf, ds, curNr},
                         {},
                         {b.getNamedAttr("plane_params",
                                         b.getDenseI32ArrayAttr(pp))},
                         "quant-record-gemv");
                  sync("compute");
                });
              });
        });
        return success();
      }

      Value rec = nramBuf(b.getI8Type(), llvm::alignTo(spec->recordBytes, 64));
      Value actBuf =
          nramBuf(b.getF32Type(), llvm::alignTo(spec->blockElements, 64));
      Value qArr =
          spec->q8kActivation
              ? nramBuf(b.getI32Type(), llvm::alignTo(spec->blockElements, 64))
              : Value{};
      forLoop(cI64(0), nr, cI64(1),
              [&](Value i) { storeAt(acc, i, cF32(0.0f)); });
      forLoop(cI64(0), cI64(recordGroups), cI64(1), [&](Value gq) {
        if (xSt1 == 1) {
          memcpy1D(actBuf, globalPtr(x, {gq, cI64(0)}),
                   cI64(spec->blockElements), "GDRAM2NRAM");
          sync("io");
        } else {
          forLoop(cI64(0), cI64(spec->blockElements), cI64(1), [&](Value k) {
            storeAt(actBuf, k,
                    loadAt(globalPtr(x, {gq, k}), cI64(0), b.getF32Type()));
          });
        }
        auto emitRecordRows = [&](Value ds) {
          forLoop(cI64(0), nr, cI64(1), [&](Value i) {
            Value ri = b.create<arith::AddIOp>(loc, r, i);
            Value base = globalPtr(w, {ri, gq, cI64(0)});
            if (wSt2 == 1) {
              memcpy1D(rec, base, cI64(spec->recordBytes), "GDRAM2NRAM");
              sync("io");
            } else {
              forLoop(cI64(0), cI64(spec->recordBytes), cI64(1), [&](Value z) {
                Value elemPtr = makeAddr(base, {z}, {cI64(wSt2)}, 0);
                storeAt(rec, z, loadAt(elemPtr, cI64(0), b.getI8Type()));
              });
            }
            auto recU32 = [&](Value idx) {
              return b
                  .create<arith::ExtUIOp>(loc, b.getI32Type(),
                                          loadAt(rec, idx, b.getI8Type()))
                  .getResult();
            };
            Value dVal =
                spec->scaleCode ==
                        intentdsl::kir::QuantScaleCode::LeHalfAtOffset
                    ? makeOp("from_le_half",
                             {recU32(cI64(spec->superScaleOffset)),
                              recU32(cI64(spec->superScaleOffset + 1))},
                             {b.getF32Type()})
                          ->getResult(0)
                    : cF32(1.0f);

            bool hierarchicalPlanes =
                spec->numBitPlanes > 0 && spec->blockElements % 128 == 0 &&
                (spec->subBlockElements <= 0 || 32 % spec->subBlockElements == 0);
            if (hierarchicalPlanes) {
              int64_t halves = spec->blockElements / 128;
              int64_t stepLen = spec->subBlockElements > 0
                                    ? spec->subBlockElements
                                    : 32;
              int64_t subSteps = 32 / stepLen;
              auto extractCodeAt = [&](ArrayRef<Value> planeBases, Value lane,
                                       int64_t q) -> Value {
                Value code;
                for (int pi = 0; pi < spec->numBitPlanes; ++pi) {
                  const auto &pl = spec->planes[pi];
                  Value byteIdx =
                      b.create<arith::AddIOp>(loc, planeBases[pi], lane);
                  Value raw = recU32(byteIdx);
                  int32_t shConst = static_cast<int32_t>(
                      (pl.quadShiftMode == 0 ? (q / 2) : q) * pl.shiftUnit);
                  Value shifted =
                      shConst > 0
                          ? b.create<arith::ShRUIOp>(loc, raw, cI32(shConst))
                                .getResult()
                          : raw;
                  Value part = b.create<arith::AndIOp>(
                      loc, shifted, cI32((1 << pl.bitWidth) - 1));
                  if (pl.destBitOffset > 0)
                    part = b.create<arith::ShLIOp>(loc, part,
                                                   cI32(pl.destBitOffset));
                  code = pi == 0 ? part
                                 : b.create<arith::OrIOp>(loc, code, part)
                                       .getResult();
                }
                if (spec->zeroBias != 0)
                  code =
                      b.create<arith::SubIOp>(loc, code, cI32(spec->zeroBias));
                return code;
              };

              if (spec->q8kActivation) {
                auto halfLoop = forLoopCarry(
                    cI64(0), cI64(halves), cI64(1), ValueRange{cI32(0)},
                    [&](Value half, ValueRange hIter) {
                      Value accI32 = hIter[0];
                      for (int64_t q = 0; q < 4; ++q) {
                        for (int64_t sb = 0; sb < subSteps; ++sb) {
                          int64_t inHalfOff = q * 32 + sb * stepLen;
                          SmallVector<Value, 4> planeBases;
                          for (int pi = 0; pi < spec->numBitPlanes; ++pi) {
                            const auto &pl = spec->planes[pi];
                            int64_t cOff = pl.baseByte +
                                           (q % 2) * pl.quadByteStep +
                                           sb * stepLen;
                            planeBases.push_back(b.create<arith::AddIOp>(
                                loc, cI64(cOff),
                                b.create<arith::MulIOp>(
                                    loc, half, cI64(pl.halfStride))));
                          }
                          Value qBase = b.create<arith::AddIOp>(
                              loc, cI64(inHalfOff),
                              b.create<arith::MulIOp>(loc, half, cI64(128)));
                          auto laneLoop = forLoopCarry(
                              cI64(0), cI64(stepLen), cI64(1),
                              ValueRange{cI32(0)},
                              [&](Value lane, ValueRange lIter) {
                                Value wCode =
                                    extractCodeAt(planeBases, lane, q);
                                Value qv = loadAt(
                                    qArr,
                                    b.create<arith::AddIOp>(loc, qBase, lane));
                                Value prod =
                                    b.create<arith::MulIOp>(loc, wCode, qv);
                                b.create<scf::YieldOp>(
                                    loc, ValueRange{b.create<arith::AddIOp>(
                                             loc, lIter[0], prod)});
                              });
                          Value subSum = laneLoop.getResult(0);
                          if (spec->subScaleOffset >= 0 &&
                              spec->subBlockElements > 0) {
                            int64_t cSub = spec->subScaleOffset +
                                           inHalfOff / spec->subBlockElements;
                            int64_t hStep = 128 / spec->subBlockElements;
                            Value subIdx = b.create<arith::AddIOp>(
                                loc, cI64(cSub),
                                b.create<arith::MulIOp>(loc, half,
                                                        cI64(hStep)));
                            Value scU8 = recU32(subIdx);
                            Value scS8 = b.create<arith::SubIOp>(
                                loc,
                                b.create<arith::XOrIOp>(loc, scU8, cI32(128)),
                                cI32(128));
                            subSum =
                                b.create<arith::MulIOp>(loc, subSum, scS8);
                          }
                          accI32 =
                              b.create<arith::AddIOp>(loc, accI32, subSum);
                        }
                      }
                      b.create<scf::YieldOp>(loc, ValueRange{accI32});
                    });
                Value sumF = b.create<arith::SIToFPOp>(
                    loc, b.getF32Type(), halfLoop.getResult(0));
                Value contrib = b.create<arith::MulFOp>(
                    loc, ds, b.create<arith::MulFOp>(loc, dVal, sumF));
                storeAt(acc, i,
                        b.create<arith::AddFOp>(loc, loadAt(acc, i), contrib));
              } else {
                auto halfLoop = forLoopCarry(
                    cI64(0), cI64(halves), cI64(1), ValueRange{cF32(0.0f)},
                    [&](Value half, ValueRange hIter) {
                      Value accF = hIter[0];
                      for (int64_t q = 0; q < 4; ++q) {
                        for (int64_t sb = 0; sb < subSteps; ++sb) {
                          int64_t inHalfOff = q * 32 + sb * stepLen;
                          SmallVector<Value, 4> planeBases;
                          for (int pi = 0; pi < spec->numBitPlanes; ++pi) {
                            const auto &pl = spec->planes[pi];
                            int64_t cOff = pl.baseByte +
                                           (q % 2) * pl.quadByteStep +
                                           sb * stepLen;
                            planeBases.push_back(b.create<arith::AddIOp>(
                                loc, cI64(cOff),
                                b.create<arith::MulIOp>(
                                    loc, half, cI64(pl.halfStride))));
                          }
                          Value qBase = b.create<arith::AddIOp>(
                              loc, cI64(inHalfOff),
                              b.create<arith::MulIOp>(loc, half, cI64(128)));
                          auto laneLoop = forLoopCarry(
                              cI64(0), cI64(stepLen), cI64(1),
                              ValueRange{cF32(0.0f)},
                              [&](Value lane, ValueRange lIter) {
                                Value wF = b.create<arith::SIToFPOp>(
                                    loc, b.getF32Type(),
                                    extractCodeAt(planeBases, lane, q));
                                Value xv = loadAt(
                                    actBuf,
                                    b.create<arith::AddIOp>(loc, qBase, lane),
                                    b.getF32Type());
                                Value prod =
                                    b.create<arith::MulFOp>(loc, wF, xv);
                                b.create<scf::YieldOp>(
                                    loc, ValueRange{b.create<arith::AddFOp>(
                                             loc, lIter[0], prod)});
                              });
                          Value subSum = laneLoop.getResult(0);
                          if (spec->subScaleOffset >= 0 &&
                              spec->subBlockElements > 0) {
                            int64_t cSub = spec->subScaleOffset +
                                           inHalfOff / spec->subBlockElements;
                            int64_t hStep = 128 / spec->subBlockElements;
                            Value subIdx = b.create<arith::AddIOp>(
                                loc, cI64(cSub),
                                b.create<arith::MulIOp>(loc, half,
                                                        cI64(hStep)));
                            Value scU8 = recU32(subIdx);
                            Value scS8 = b.create<arith::SubIOp>(
                                loc,
                                b.create<arith::XOrIOp>(loc, scU8, cI32(128)),
                                cI32(128));
                            subSum = b.create<arith::MulFOp>(
                                loc, subSum,
                                b.create<arith::SIToFPOp>(loc, b.getF32Type(),
                                                          scS8));
                          }
                          accF = b.create<arith::AddFOp>(loc, accF, subSum);
                        }
                      }
                      b.create<scf::YieldOp>(loc, ValueRange{accF});
                    });
                Value contrib =
                    b.create<arith::MulFOp>(loc, dVal, halfLoop.getResult(0));
                storeAt(acc, i,
                        b.create<arith::AddFOp>(loc, loadAt(acc, i), contrib));
              }
              return;
            }

            auto reconstructWeight = [&](Value idx) -> Value {
              Value half = b.create<arith::DivSIOp>(loc, idx, cI64(128));
              Value inHalf = b.create<arith::RemSIOp>(loc, idx, cI64(128));
              Value quad = b.create<arith::DivSIOp>(loc, inHalf, cI64(32));
              Value lane = b.create<arith::RemSIOp>(loc, inHalf, cI64(32));
              Value quadOdd = b.create<arith::RemSIOp>(loc, quad, cI64(2));
              Value quadHalf = b.create<arith::DivSIOp>(loc, quad, cI64(2));
              Value code;
              for (int pi = 0; pi < spec->numBitPlanes; ++pi) {
                const auto &pl = spec->planes[pi];
                Value byteIdx = b.create<arith::AddIOp>(
                    loc,
                    b.create<arith::AddIOp>(
                        loc, cI64(pl.baseByte),
                        b.create<arith::MulIOp>(loc, half,
                                                cI64(pl.halfStride))),
                    b.create<arith::AddIOp>(
                        loc,
                        b.create<arith::MulIOp>(loc, quadOdd,
                                                cI64(pl.quadByteStep)),
                        lane));
                Value raw = recU32(byteIdx);
                Value shFactor = pl.quadShiftMode == 0 ? quadHalf : quad;
                Value sh32 = b.create<arith::TruncIOp>(
                    loc, b.getI32Type(),
                    b.create<arith::MulIOp>(loc, shFactor,
                                            cI64(pl.shiftUnit)));
                Value part = b.create<arith::AndIOp>(
                    loc, b.create<arith::ShRUIOp>(loc, raw, sh32),
                    cI32((1 << pl.bitWidth) - 1));
                if (pl.destBitOffset > 0)
                  part = b.create<arith::ShLIOp>(loc, part,
                                                 cI32(pl.destBitOffset));
                code = pi == 0
                           ? part
                           : b.create<arith::OrIOp>(loc, code, part).getResult();
              }
              if (spec->zeroBias != 0)
                code =
                    b.create<arith::SubIOp>(loc, code, cI32(spec->zeroBias));
              if (spec->subScaleOffset >= 0 && spec->subBlockElements > 0) {
                Value subIdx = b.create<arith::AddIOp>(
                    loc, cI64(spec->subScaleOffset),
                    b.create<arith::DivSIOp>(loc, idx,
                                             cI64(spec->subBlockElements)));
                Value scU8 = recU32(subIdx);
                Value scS8 = b.create<arith::SubIOp>(
                    loc, b.create<arith::XOrIOp>(loc, scU8, cI32(128)),
                    cI32(128));
                code = b.create<arith::MulIOp>(loc, code, scS8);
              }
              return code;
            };
            if (spec->q8kActivation) {
              auto dotLoop = forLoopCarry(
                  cI64(0), cI64(spec->blockElements), cI64(1),
                  ValueRange{cI32(0)}, [&](Value idx, ValueRange st) {
                    Value wCode = reconstructWeight(idx);
                    Value qv = loadAt(qArr, idx);
                    Value prod = b.create<arith::MulIOp>(loc, wCode, qv);
                    b.create<scf::YieldOp>(
                        loc,
                        ValueRange{b.create<arith::AddIOp>(loc, st[0], prod)});
                  });
              Value sumF = b.create<arith::SIToFPOp>(
                  loc, b.getF32Type(), dotLoop.getResult(0));
              Value contrib = b.create<arith::MulFOp>(
                  loc, ds, b.create<arith::MulFOp>(loc, dVal, sumF));
              storeAt(acc, i,
                      b.create<arith::AddFOp>(loc, loadAt(acc, i), contrib));
            } else {
              auto dotLoop = forLoopCarry(
                  cI64(0), cI64(spec->blockElements), cI64(1),
                  ValueRange{cF32(0.0f)}, [&](Value idx, ValueRange st) {
                    Value wF = b.create<arith::SIToFPOp>(
                        loc, b.getF32Type(), reconstructWeight(idx));
                    Value xv = loadAt(actBuf, idx, b.getF32Type());
                    Value prod = b.create<arith::MulFOp>(loc, wF, xv);
                    b.create<scf::YieldOp>(
                        loc,
                        ValueRange{b.create<arith::AddFOp>(loc, st[0], prod)});
                  });
              Value contrib =
                  b.create<arith::MulFOp>(loc, dVal, dotLoop.getResult(0));
              storeAt(acc, i,
                      b.create<arith::AddFOp>(loc, loadAt(acc, i), contrib));
            }
          });
        };

        if (!spec->q8kActivation) {
          emitRecordRows(cF32(1.0f));
          return;
        }
        intentdsl::kir::Q8KRecord q8;
        auto maxLoop = forLoopCarry(
            cI64(0), cI64(spec->blockElements), cI64(1),
            ValueRange{cF32(0.0f)}, [&](Value k, ValueRange iter) {
              Value xk = loadAt(actBuf, k, b.getF32Type());
              Value ax = makeOp("scalar_math", {xk}, {b.getF32Type()},
                                {b.getNamedAttr("op", b.getStringAttr("fabs"))})
                             ->getResult(0);
              Value ae =
                  makeOp("scalar_math", {iter[0]}, {b.getF32Type()},
                         {b.getNamedAttr("op", b.getStringAttr("fabs"))})
                      ->getResult(0);
              Value gt = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT,
                                                 ax, ae);
              Value next = b.create<arith::SelectOp>(loc, gt, xk, iter[0]);
              b.create<scf::YieldOp>(loc, ValueRange{next});
            });
        Value e = maxLoop.getResult(0);
        Value isZero = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ,
                                               e, cF32(0.0f));
        ifThenElse(
            isZero, {},
            [&]() {
              if (spec->scaleCode ==
                  intentdsl::kir::QuantScaleCode::LeHalfAtOffset) {
                forLoop(cI64(0), nr, cI64(1), [&](Value i) {
                  Value ri = b.create<arith::AddIOp>(loc, r, i);
                  Value dHiPtr =
                      globalPtr(w, {ri, gq, cI64(spec->superScaleOffset + 1)});
                  Value dHi = b.create<arith::ExtUIOp>(
                      loc, b.getI32Type(),
                      loadAt(dHiPtr, cI64(0), b.getI8Type()));
                  Value c7c = cI32(0x7c);
                  Value nf = b.create<arith::CmpIOp>(
                      loc, arith::CmpIPredicate::eq,
                      b.create<arith::AndIOp>(loc, dHi, c7c), c7c);
                  Value nanF = cF32(std::numeric_limits<float>::quiet_NaN());
                  Value delta =
                      b.create<arith::SelectOp>(loc, nf, nanF, cF32(0.0f));
                  storeAt(acc, i,
                          b.create<arith::AddFOp>(loc, loadAt(acc, i), delta));
                });
              }
            },
            [&]() {
              Value maxQuant = cF32(-float((1 << (q8.elementBits - 1)) - 1));
              Value inv = b.create<arith::DivFOp>(loc, maxQuant, e);
              Value ds = b.create<arith::DivFOp>(loc, cF32(1.0f), inv);
              int32_t qMin = -(1 << (q8.elementBits - 1));
              int32_t qMax = (1 << (q8.elementBits - 1)) - 1;
              forLoop(cI64(0), cI64(spec->blockElements), cI64(1),
                      [&](Value k) {
                        Value xk = loadAt(actBuf, k, b.getF32Type());
                        Value f = b.create<arith::MulFOp>(loc, xk, inv);
                        Value fl = makeOp("scalar_math", {f}, {b.getF32Type()},
                                          {b.getNamedAttr(
                                              "op", b.getStringAttr("floor"))})
                                       ->getResult(0);
                        Value lo =
                            b.create<arith::FPToSIOp>(loc, b.getI32Type(), fl);
                        Value frac = b.create<arith::SubFOp>(loc, f, fl);
                        Value gtHalf = b.create<arith::CmpFOp>(
                            loc, arith::CmpFPredicate::OGT, frac, cF32(0.5f));
                        Value eqHalf = b.create<arith::CmpFOp>(
                            loc, arith::CmpFPredicate::OEQ, frac, cF32(0.5f));
                        Value odd = b.create<arith::CmpIOp>(
                            loc, arith::CmpIPredicate::ne,
                            b.create<arith::AndIOp>(loc, lo, cI32(1)), cI32(0));
                        Value bump = b.create<arith::OrIOp>(
                            loc, gtHalf,
                            b.create<arith::AndIOp>(loc, eqHalf, odd));
                        Value inc = b.create<arith::SelectOp>(loc, bump,
                                                              cI32(1), cI32(0));
                        Value rnd = b.create<arith::AddIOp>(loc, lo, inc);
                        Value cl = b.create<arith::MinSIOp>(
                            loc, b.create<arith::MaxSIOp>(loc, rnd, cI32(qMin)),
                            cI32(qMax));
                        storeAt(qArr, k, cl);
                      });
              emitRecordRows(ds);
            });
      });
      return success();
    }

    if (spec->roleLayout ==
        intentdsl::kir::QuantRoleLayout::BlockScaledPair) {
      Value a = p.dot.getOperand(0), sa = p.dot.getOperand(1),
            rhs = p.dot.getOperand(2);
      intentdsl::kir::RightScale rightScale(p.dot);
      int64_t recBytes = spec->recordBytes;
      int64_t groups = shapeK / spec->blockElements;
      int64_t groupsPerPanel = std::gcd(
          groups, std::max<int64_t>(1, bk / spec->blockElements));
      int64_t panelRecBytes = groupsPerPanel * recBytes;
      int64_t panelElems = groupsPerPanel * spec->blockElements;
      int64_t convK = std::max<int64_t>(panelElems, 64);

      Value aRawBuf =
          nramBuf(b.getI8Type(), llvm::alignTo(bm * panelRecBytes, 64));
      Value bRawBuf =
          nramBuf(b.getI8Type(), llvm::alignTo(panelRecBytes * bn, 64));
      Value saBuf =
          nramBuf(b.getI8Type(), llvm::alignTo(bm * groupsPerPanel, 64));
      Value sbBuf =
          nramBuf(b.getI8Type(), llvm::alignTo(bn * groupsPerPanel, 64));
      Value leftF16 = nramBuf(b.getF16Type(), bm * convK);
      Value transF16 = nramBuf(b.getF16Type(), bn * convK);
      Value wFilter = wramBuf(b.getF16Type(), bn * convK);
      Value rightBuf = nramBuf(b.getF32Type(), spec->blockElements * bn);
      Value tmpBuf = nramBuf(b.getF32Type(), bn);
      bool useLutSimd = quantLut && spec->packBits == 4 &&
                        spec->packFactor == 2 && bn == 64 &&
                        panelElems == convK;
      Value idxScratch =
          useLutSimd
              ? nramBuf(b.getI32Type(),
                        llvm::alignTo(std::max(bm, bn) * panelRecBytes, 64))
              : Value{};
      Value scScratch =
          useLutSimd
              ? nramBuf(b.getF16Type(),
                        std::max<int64_t>(
                            8192,
                            llvm::alignTo(std::max(bm, bn) * groupsPerPanel * 2,
                                          64)))
              : Value{};
      Value trScratch =
          useLutSimd ? nramBuf(b.getF16Type(), std::max(bm, bn) * convK)
                     : Value{};
      fillVec(acc, cI64(bm * bn), cF32(0.0f));

      int32_t codeMask = (1 << spec->packBits) - 1;
      int32_t f16MinExp = spec->scaleBias - 13;
      int32_t f16MaxExp = spec->scaleBias + 13;
      forLoop(cI64(0), cI64(groups), cI64(groupsPerPanel), [&](Value u0) {
        if (p.stride(a, 2) == 1 && p.stride(a, 1) == recBytes) {
          Value aBase = globalPtr(a, {r, u0, cI64(0)}, b.getI8Type());
          ifThenElse(
              b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, nr,
                                      cI64(1)),
              {},
              [&]() {
                memcpy1D(aRawBuf, aBase, cI64(panelRecBytes), "GDRAM2NRAM");
              },
              [&]() {
                memcpy2D(aRawBuf, aBase, cI64(panelRecBytes),
                         cI64(panelRecBytes), cI64(p.stride(a, 0)),
                         b.create<arith::SubIOp>(loc, nr, cI64(1)),
                         "GDRAM2NRAM");
              });
        } else {
          forLoop(cI64(0), nr, cI64(1), [&](Value i) {
            Value ri = b.create<arith::AddIOp>(loc, r, i);
            forLoop(cI64(0), cI64(groupsPerPanel), cI64(1), [&](Value g) {
              Value ug = b.create<arith::AddIOp>(loc, u0, g);
              forLoop(cI64(0), cI64(recBytes), cI64(1), [&](Value z) {
                Value idx = b.create<arith::AddIOp>(
                    loc,
                    b.create<arith::MulIOp>(loc, i, cI64(panelRecBytes)),
                    b.create<arith::AddIOp>(
                        loc, b.create<arith::MulIOp>(loc, g, cI64(recBytes)),
                        z));
                storeAt(aRawBuf, idx,
                        loadAt(globalPtr(a, {ri, ug, z}, b.getI8Type()),
                               cI64(0), b.getI8Type()));
              });
            });
          });
        }

        if (p.stride(rhs, 2) == 1 &&
            p.stride(rhs, 0) == recBytes * p.stride(rhs, 1)) {
          Value bBase = globalPtr(rhs, {u0, cI64(0), c}, b.getI8Type());
          if (panelRecBytes == 1)
            memcpy1D(bRawBuf, bBase, nc, "GDRAM2NRAM");
          else
            memcpy2D(bRawBuf, bBase, nc, cI64(bn), cI64(p.stride(rhs, 1)),
                     cI64(panelRecBytes - 1), "GDRAM2NRAM");
        } else {
          forLoop(cI64(0), cI64(groupsPerPanel), cI64(1), [&](Value g) {
            Value ug = b.create<arith::AddIOp>(loc, u0, g);
            forLoop(cI64(0), cI64(recBytes), cI64(1), [&](Value z) {
              Value gz = b.create<arith::AddIOp>(
                  loc, b.create<arith::MulIOp>(loc, g, cI64(recBytes)), z);
              forLoop(cI64(0), nc, cI64(1), [&](Value j) {
                Value cj = b.create<arith::AddIOp>(loc, c, j);
                Value idx = b.create<arith::AddIOp>(
                    loc, b.create<arith::MulIOp>(loc, gz, cI64(bn)), j);
                storeAt(bRawBuf, idx,
                        loadAt(globalPtr(rhs, {ug, z, cj}, b.getI8Type()),
                               cI64(0), b.getI8Type()));
              });
            });
          });
        }

        if (p.stride(sa, 1) == 1) {
          Value saBase = globalPtr(sa, {r, u0}, b.getI8Type());
          ifThenElse(
              b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, nr,
                                      cI64(1)),
              {},
              [&]() {
                memcpy1D(saBuf, saBase, cI64(groupsPerPanel), "GDRAM2NRAM");
              },
              [&]() {
                memcpy2D(saBuf, saBase, cI64(groupsPerPanel),
                         cI64(groupsPerPanel), cI64(p.stride(sa, 0)),
                         b.create<arith::SubIOp>(loc, nr, cI64(1)),
                         "GDRAM2NRAM");
              });
        } else {
          forLoop(cI64(0), nr, cI64(1), [&](Value i) {
            Value ri = b.create<arith::AddIOp>(loc, r, i);
            forLoop(cI64(0), cI64(groupsPerPanel), cI64(1), [&](Value g) {
              Value ug = b.create<arith::AddIOp>(loc, u0, g);
              Value idx = b.create<arith::AddIOp>(
                  loc, b.create<arith::MulIOp>(loc, i, cI64(groupsPerPanel)),
                  g);
              storeAt(saBuf, idx,
                      loadAt(globalPtr(sa, {ri, ug}, b.getI8Type()), cI64(0),
                             b.getI8Type()));
            });
          });
        }

        // Stage rightScale into sbBuf. When rightScale is [G, N] contiguous and
        // nc == bn, load [groupsPerPanel, bn] via a single DMA (col_major = 2).
        Value rhsColMode = cI32(1);
        if (rightScale.column == 0 && p.stride(rightScale.view, 1) == 1) {
          Value sbBase =
              globalPtr(rightScale.view, {c, u0}, b.getI8Type());
          ifThenElse(
              b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, nc,
                                      cI64(1)),
              {},
              [&]() {
                memcpy1D(sbBuf, sbBase, cI64(groupsPerPanel), "GDRAM2NRAM");
              },
              [&]() {
                memcpy2D(sbBuf, sbBase, cI64(groupsPerPanel),
                         cI64(groupsPerPanel),
                         cI64(p.stride(rightScale.view, 0)),
                         b.create<arith::SubIOp>(loc, nc, cI64(1)),
                         "GDRAM2NRAM");
              });
        } else if (useLutSimd && rightScale.column == 1 &&
                   p.stride(rightScale.view, 1) == 1) {
          Value fullCols =
              b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, nc, cI64(bn));
          rhsColMode =
              b.create<arith::SelectOp>(loc, fullCols, cI32(2), cI32(1));
          ifThenElse(
              fullCols, {},
              [&]() {
                Value sbBase =
                    globalPtr(rightScale.view, {u0, c}, b.getI8Type());
                if (groupsPerPanel == 1)
                  memcpy1D(sbBuf, sbBase, cI64(bn), "GDRAM2NRAM");
                else
                  memcpy2D(sbBuf, sbBase, cI64(bn), cI64(bn),
                           cI64(p.stride(rightScale.view, 0)),
                           cI64(groupsPerPanel - 1), "GDRAM2NRAM");
              },
              [&]() {
                forLoop(cI64(0), nc, cI64(1), [&](Value j) {
                  Value cj = b.create<arith::AddIOp>(loc, c, j);
                  forLoop(cI64(0), cI64(groupsPerPanel), cI64(1), [&](Value g) {
                    Value ug = b.create<arith::AddIOp>(loc, u0, g);
                    Value idx = b.create<arith::AddIOp>(
                        loc,
                        b.create<arith::MulIOp>(loc, j, cI64(groupsPerPanel)),
                        g);
                    storeAt(
                        sbBuf, idx,
                        loadAt(globalPtr(rightScale.view, {ug, cj}, b.getI8Type()),
                               cI64(0), b.getI8Type()));
                  });
                });
              });
        } else {
          forLoop(cI64(0), nc, cI64(1), [&](Value j) {
            Value cj = b.create<arith::AddIOp>(loc, c, j);
            forLoop(cI64(0), cI64(groupsPerPanel), cI64(1), [&](Value g) {
              Value ug = b.create<arith::AddIOp>(loc, u0, g);
              SmallVector<Value> sbCoords(2);
              sbCoords[rightScale.column] = cj;
              sbCoords[rightScale.group] = ug;
              Value idx = b.create<arith::AddIOp>(
                  loc, b.create<arith::MulIOp>(loc, j, cI64(groupsPerPanel)),
                  g);
              storeAt(sbBuf, idx,
                      loadAt(globalPtr(rightScale.view, sbCoords, b.getI8Type()),
                             cI64(0), b.getI8Type()));
            });
          });
        }
        sync("io");

        Value okVal;
        if (useLutSimd) {
          Value na = b.create<arith::MulIOp>(loc, nr, cI64(groupsPerPanel));
          Value nb = b.create<arith::MulIOp>(loc, nc, cI64(groupsPerPanel));
          okVal = makeOp("quant_e8m0_ok",
                         {saBuf, na, sbBuf, nb, scScratch, cI32(f16MinExp),
                          cI32(f16MaxExp)},
                         {b.getI1Type()}, {}, "quant-e8m0-guard")
                      ->getResult(0);
        } else {
          auto okA = forLoopCarry(
              cI64(0), b.create<arith::MulIOp>(loc, nr, cI64(groupsPerPanel)),
              cI64(1), ValueRange{cI1(bn == 64)}, [&](Value idx, ValueRange st) {
                Value sb = b.create<arith::ExtUIOp>(
                    loc, b.getI32Type(), loadAt(saBuf, idx, b.getI8Type()));
                Value inWin = b.create<arith::AndIOp>(
                    loc,
                    b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, sb,
                                            cI32(f16MinExp)),
                    b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle, sb,
                                            cI32(f16MaxExp)));
                b.create<scf::YieldOp>(
                    loc, ValueRange{b.create<arith::AndIOp>(loc, st[0], inWin)});
              });
          auto okAB = forLoopCarry(
              cI64(0), b.create<arith::MulIOp>(loc, nc, cI64(groupsPerPanel)),
              cI64(1), ValueRange{okA.getResult(0)},
              [&](Value idx, ValueRange st) {
                Value sb = b.create<arith::ExtUIOp>(
                    loc, b.getI32Type(), loadAt(sbBuf, idx, b.getI8Type()));
                Value inWin = b.create<arith::AndIOp>(
                    loc,
                    b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, sb,
                                            cI32(f16MinExp)),
                    b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle, sb,
                                            cI32(f16MaxExp)));
                b.create<scf::YieldOp>(
                    loc, ValueRange{b.create<arith::AndIOp>(loc, st[0], inWin)});
              });
          okVal = okAB.getResult(0);
        }

        ifThenElse(
            okVal, {},
            [&]() {
              if (useLutSimd) {
                makeOp("quant_decode_block_scaled",
                       {transF16, bRawBuf, sbBuf, quantLut, idxScratch,
                        scScratch, trScratch, nc, cI64(bn),
                        cI64(groupsPerPanel), cI64(spec->blockElements),
                        cI32(spec->scaleBias), rhsColMode},
                       {}, {}, "quant-block-scaled");
                makeOp("quant_decode_block_scaled",
                       {leftF16, aRawBuf, saBuf, quantLut, idxScratch,
                        scScratch, trScratch, nr, cI64(bm),
                        cI64(groupsPerPanel), cI64(spec->blockElements),
                        cI32(spec->scaleBias), cI32(0)},
                       {}, {}, "quant-block-scaled");
                sync("compute");
                memcpy1D(wFilter, transF16, cI64(bn * convK), "NRAM2WRAM");
                Value accum = b.create<arith::CmpIOp>(
                    loc, arith::CmpIPredicate::ne, u0, cI64(0));
                makeOp("wram_conv", {acc, leftF16, wFilter, accum}, {},
                       {b.getNamedAttr("bm", b.getI64IntegerAttr(bm)),
                        b.getNamedAttr("bn", b.getI64IntegerAttr(bn)),
                        b.getNamedAttr("bk", b.getI64IntegerAttr(convK)),
                        b.getNamedAttr("mode", b.getStringAttr("f16"))},
                       "wram-conv-f16");
                sync("compute");
                return;
              }
              Value zeroHalf = b.create<arith::ConstantOp>(
                  loc, b.getZeroAttr(b.getF16Type()));
              if (panelElems < convK) {
                fillVec(leftF16, cI64(bm * convK), zeroHalf);
                fillVec(transF16, cI64(bn * convK), zeroHalf);
              } else {
                ifThen(b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                               nr, cI64(bm)),
                       [&]() { fillVec(leftF16, cI64(bm * convK), zeroHalf); });
                ifThen(b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                               nc, cI64(bn)),
                       [&]() {
                         fillVec(transF16, cI64(bn * convK), zeroHalf);
                       });
              }
              sync("compute");
              forLoop(cI64(0), nc, cI64(1), [&](Value j) {
                Value fltRowBase =
                    b.create<arith::MulIOp>(loc, j, cI64(convK));
                Value sbRowBase =
                    b.create<arith::MulIOp>(loc, j, cI64(groupsPerPanel));
                forLoop(cI64(0), cI64(groupsPerPanel), cI64(1), [&](Value g) {
                  Value sbByte = b.create<arith::ExtUIOp>(
                      loc, b.getI32Type(),
                      loadAt(sbBuf,
                             b.create<arith::AddIOp>(loc, sbRowBase, g),
                             b.getI8Type()));
                  Value sbF = decodeScalarE8M0(sbByte, *spec);
                  Value gRecBase =
                      b.create<arith::MulIOp>(loc, g, cI64(recBytes));
                  Value gElemBase = b.create<arith::AddIOp>(
                      loc, fltRowBase,
                      b.create<arith::MulIOp>(loc, g,
                                              cI64(spec->blockElements)));
                  forLoop(cI64(0), cI64(recBytes), cI64(1), [&](Value z) {
                    Value gz = b.create<arith::AddIOp>(loc, gRecBase, z);
                    Value rawIdx = b.create<arith::AddIOp>(
                        loc, b.create<arith::MulIOp>(loc, gz, cI64(bn)), j);
                    Value raw = b.create<arith::ExtUIOp>(
                        loc, b.getI32Type(),
                        loadAt(bRawBuf, rawIdx, b.getI8Type()));
                    Value zElemOff = b.create<arith::AddIOp>(
                        loc, gElemBase,
                        b.create<arith::MulIOp>(loc, z,
                                                cI64(spec->packFactor)));
                    for (int sub = 0; sub < spec->packFactor; ++sub) {
                      Value shifted =
                          spec->laneBitShifts[sub] > 0
                              ? b.create<arith::ShRUIOp>(
                                     loc, raw, cI32(spec->laneBitShifts[sub]))
                                    .getResult()
                              : raw;
                      Value code =
                          b.create<arith::AndIOp>(loc, shifted, cI32(codeMask));
                      if (spec->zeroBias != 0)
                        code = b.create<arith::SubIOp>(loc, code,
                                                       cI32(spec->zeroBias));
                      Value decF = b.create<arith::MulFOp>(
                          loc, decodeScalarQuantCode(code, *spec), sbF);
                      Value decH =
                          b.create<arith::TruncFOp>(loc, b.getF16Type(), decF);
                      storeAt(transF16,
                              b.create<arith::AddIOp>(loc, zElemOff, cI64(sub)),
                              decH);
                    }
                  });
                });
              });
              forLoop(cI64(0), nr, cI64(1), [&](Value i) {
                Value lhsRowBase =
                    b.create<arith::MulIOp>(loc, i, cI64(convK));
                Value saRowBase =
                    b.create<arith::MulIOp>(loc, i, cI64(groupsPerPanel));
                Value aRowBase =
                    b.create<arith::MulIOp>(loc, i, cI64(panelRecBytes));
                forLoop(cI64(0), cI64(groupsPerPanel), cI64(1), [&](Value g) {
                  Value saByte = b.create<arith::ExtUIOp>(
                      loc, b.getI32Type(),
                      loadAt(saBuf,
                             b.create<arith::AddIOp>(loc, saRowBase, g),
                             b.getI8Type()));
                  Value saF = decodeScalarE8M0(saByte, *spec);
                  Value gRecBase = b.create<arith::AddIOp>(
                      loc, aRowBase,
                      b.create<arith::MulIOp>(loc, g, cI64(recBytes)));
                  Value gElemBase = b.create<arith::AddIOp>(
                      loc, lhsRowBase,
                      b.create<arith::MulIOp>(loc, g,
                                              cI64(spec->blockElements)));
                  forLoop(cI64(0), cI64(recBytes), cI64(1), [&](Value z) {
                    Value raw = b.create<arith::ExtUIOp>(
                        loc, b.getI32Type(),
                        loadAt(aRawBuf,
                               b.create<arith::AddIOp>(loc, gRecBase, z),
                               b.getI8Type()));
                    Value zElemOff = b.create<arith::AddIOp>(
                        loc, gElemBase,
                        b.create<arith::MulIOp>(loc, z,
                                                cI64(spec->packFactor)));
                    for (int sub = 0; sub < spec->packFactor; ++sub) {
                      Value shifted =
                          spec->laneBitShifts[sub] > 0
                              ? b.create<arith::ShRUIOp>(
                                     loc, raw, cI32(spec->laneBitShifts[sub]))
                                    .getResult()
                              : raw;
                      Value code =
                          b.create<arith::AndIOp>(loc, shifted, cI32(codeMask));
                      if (spec->zeroBias != 0)
                        code = b.create<arith::SubIOp>(loc, code,
                                                       cI32(spec->zeroBias));
                      Value decF = b.create<arith::MulFOp>(
                          loc, decodeScalarQuantCode(code, *spec), saF);
                      Value decH =
                          b.create<arith::TruncFOp>(loc, b.getF16Type(), decF);
                      storeAt(leftF16,
                              b.create<arith::AddIOp>(loc, zElemOff, cI64(sub)),
                              decH);
                    }
                  });
                });
              });
              sync("compute");
              memcpy1D(wFilter, transF16, cI64(bn * convK), "NRAM2WRAM");
              makeOp("wram_conv", {acc, leftF16, wFilter, cI1(true)}, {},
                     {b.getNamedAttr("bm", b.getI64IntegerAttr(bm)),
                      b.getNamedAttr("bn", b.getI64IntegerAttr(bn)),
                      b.getNamedAttr("bk", b.getI64IntegerAttr(convK)),
                      b.getNamedAttr("mode", b.getStringAttr("f16"))},
                     "wram-conv-f16");
              sync("compute");
            },
            [&]() {
              forLoop(cI64(0), cI64(groupsPerPanel), cI64(1), [&](Value g) {
                Value gRecBase =
                    b.create<arith::MulIOp>(loc, g, cI64(recBytes));
                forLoop(cI64(0), nc, cI64(1), [&](Value j) {
                  Value sbByte = b.create<arith::ExtUIOp>(
                      loc, b.getI32Type(),
                      loadAt(sbBuf,
                             b.create<arith::AddIOp>(
                                 loc,
                                 b.create<arith::MulIOp>(loc, j,
                                                         cI64(groupsPerPanel)),
                                 g),
                             b.getI8Type()));
                  Value sbF = decodeScalarE8M0(sbByte, *spec);
                  forLoop(cI64(0), cI64(recBytes), cI64(1), [&](Value z) {
                    Value gz = b.create<arith::AddIOp>(loc, gRecBase, z);
                    Value raw = b.create<arith::ExtUIOp>(
                        loc, b.getI32Type(),
                        loadAt(
                            bRawBuf,
                            b.create<arith::AddIOp>(
                                loc, b.create<arith::MulIOp>(loc, gz, cI64(bn)),
                                j),
                            b.getI8Type()));
                    Value tBase =
                        b.create<arith::MulIOp>(loc, z, cI64(spec->packFactor));
                    for (int sub = 0; sub < spec->packFactor; ++sub) {
                      Value shifted =
                          spec->laneBitShifts[sub] > 0
                              ? b.create<arith::ShRUIOp>(
                                     loc, raw, cI32(spec->laneBitShifts[sub]))
                                    .getResult()
                              : raw;
                      Value code =
                          b.create<arith::AndIOp>(loc, shifted, cI32(codeMask));
                      if (spec->zeroBias != 0)
                        code = b.create<arith::SubIOp>(loc, code,
                                                       cI32(spec->zeroBias));
                      Value decF = b.create<arith::MulFOp>(
                          loc, decodeScalarQuantCode(code, *spec), sbF);
                      Value t = b.create<arith::AddIOp>(loc, tBase, cI64(sub));
                      Value idx = b.create<arith::AddIOp>(
                          loc, b.create<arith::MulIOp>(loc, t, cI64(bn)), j);
                      storeAt(rightBuf, idx, decF);
                    }
                  });
                });
                forLoop(cI64(0), nr, cI64(1), [&](Value i) {
                  Value saByte = b.create<arith::ExtUIOp>(
                      loc, b.getI32Type(),
                      loadAt(saBuf,
                             b.create<arith::AddIOp>(
                                 loc,
                                 b.create<arith::MulIOp>(loc, i,
                                                         cI64(groupsPerPanel)),
                                 g),
                             b.getI8Type()));
                  Value saF = decodeScalarE8M0(saByte, *spec);
                  Value accRow = makeAddr(acc, {i}, {cI64(bn)}, 1);
                  Value aBaseOff = b.create<arith::AddIOp>(
                      loc,
                      b.create<arith::MulIOp>(loc, i, cI64(panelRecBytes)),
                      gRecBase);
                  forLoop(cI64(0), cI64(recBytes), cI64(1), [&](Value z) {
                    Value raw = b.create<arith::ExtUIOp>(
                        loc, b.getI32Type(),
                        loadAt(aRawBuf,
                               b.create<arith::AddIOp>(loc, aBaseOff, z),
                               b.getI8Type()));
                    Value tBase =
                        b.create<arith::MulIOp>(loc, z, cI64(spec->packFactor));
                    for (int sub = 0; sub < spec->packFactor; ++sub) {
                      Value shifted =
                          spec->laneBitShifts[sub] > 0
                              ? b.create<arith::ShRUIOp>(
                                     loc, raw, cI32(spec->laneBitShifts[sub]))
                                    .getResult()
                              : raw;
                      Value code =
                          b.create<arith::AndIOp>(loc, shifted, cI32(codeMask));
                      if (spec->zeroBias != 0)
                        code = b.create<arith::SubIOp>(loc, code,
                                                       cI32(spec->zeroBias));
                      Value av = b.create<arith::MulFOp>(
                          loc, decodeScalarQuantCode(code, *spec), saF);
                      Value t = b.create<arith::AddIOp>(loc, tBase, cI64(sub));
                      Value rRow = makeAddr(rightBuf, {t}, {cI64(bn)}, 1);
                      makeOp("vec_scalar", {tmpBuf, rRow, av, cI64(bn)}, {},
                             {b.getNamedAttr("op", b.getStringAttr("mul"))});
                      makeOp("vec_binary", {accRow, accRow, tmpBuf, cI64(bn)},
                             {},
                             {b.getNamedAttr("op", b.getStringAttr("add"))});
                    }
                  });
                });
              });
            });
      });
      return success();
    }

    // GroupedWeightRhs: lhs [M, K], qw [K, N / packFactor],
    // sc [K / blockElements, N], optional qz [K / blockElements, N / packFactor].
    Value lhsView = p.dot.getOperand(0), qwView = p.dot.getOperand(1),
          scView = p.dot.getOperand(2);
    Type lhsElem = carrierElemType(lhsView);
    Type packElem = carrierElemType(qwView);
    Type scElem = carrierElemType(scView);
    int64_t packCols = std::max<int64_t>(1, bn / spec->packFactor);
    int64_t maxGroupsInPanel = bk / spec->blockElements + 2;
    Value leftBuf = nramBuf(lhsElem, bm * bk);
    Value transBuf = nramBuf(lhsElem, bn * bk);
    Value wFilter = lhsElem.isF16() ? wramBuf(lhsElem, bn * bk) : Value{};
    Value qwBuf =
        nramBuf(packElem, llvm::alignTo(bk * packCols, 16));
    int64_t scCap = llvm::alignTo(maxGroupsInPanel * bn, 64);
    Value scBuf = nramBuf(scElem, scCap);
    Value qzBuf = spec->hasZeroView
                      ? nramBuf(packElem,
                                llvm::alignTo(maxGroupsInPanel * packCols, 32))
                      : Value{};
    bool useGrpSimd = quantLut && spec->packBits == 4 &&
                      spec->packFactor == 8 && lhsElem.isF16() &&
                      scElem.isF16() &&
                      packElem.getIntOrFloatBitWidth() == 32 && bn == 64;
    Value idxScratch =
        useGrpSimd ? nramBuf(b.getI32Type(), bk * packCols * 4) : Value{};
    Value workA = useGrpSimd ? nramBuf(b.getF16Type(), bk * bn) : Value{};
    Value workB = useGrpSimd ? nramBuf(b.getF16Type(), bk * bn) : Value{};
    Value qzDummy =
        (useGrpSimd && !spec->hasZeroView) ? nramBuf(b.getI32Type(), 16) : Value{};
    Value tmpF32 = !lhsElem.isF16() ? nramBuf(b.getF32Type(), bn) : Value{};
    Value rightF32 =
        !lhsElem.isF16() ? nramBuf(b.getF32Type(), bk * bn) : Value{};
    fillVec(acc, cI64(bm * bn), cF32(0.0f));
    if (useGrpSimd)
      fillVec(scBuf, cI64(scCap),
              b.create<arith::ConstantOp>(loc, b.getZeroAttr(scElem)));

    int32_t codeMask = (1 << spec->packBits) - 1;
    forLoop(cI64(0), cI64(shapeK), cI64(bk), [&](Value kk) {
      Value nk = b.create<arith::MinSIOp>(
          loc, b.create<arith::SubIOp>(loc, cI64(shapeK), kk), cI64(bk));
      Value zeroLhs =
          b.create<arith::ConstantOp>(loc, b.getZeroAttr(lhsElem));
      Value partialLhs = b.create<arith::OrIOp>(
          loc,
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, nr, cI64(bm)),
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, nk, cI64(bk)));
      ifThen(partialLhs, [&]() { fillVec(leftBuf, cI64(bm * bk), zeroLhs); });
      if (p.stride(lhsView, 1) == 1) {
        Value lBase = globalPtr(lhsView, {r, kk}, lhsElem);
        ifThenElse(
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, nr, cI64(1)),
            {}, [&]() { memcpy1D(leftBuf, lBase, nk, "GDRAM2NRAM"); },
            [&]() {
              memcpy2D(leftBuf, lBase, nk, cI64(bk),
                       cI64(p.stride(lhsView, 0)),
                       b.create<arith::SubIOp>(loc, nr, cI64(1)),
                       "GDRAM2NRAM");
            });
      } else {
        forLoop(cI64(0), nr, cI64(1), [&](Value i) {
          Value ri = b.create<arith::AddIOp>(loc, r, i);
          forLoop(cI64(0), nk, cI64(1), [&](Value t) {
            Value kt = b.create<arith::AddIOp>(loc, kk, t);
            Value idx = b.create<arith::AddIOp>(
                loc, b.create<arith::MulIOp>(loc, i, cI64(bk)), t);
            storeAt(leftBuf, idx,
                    loadAt(globalPtr(lhsView, {ri, kt}, lhsElem), cI64(0),
                           lhsElem));
          });
        });
      }

      Value nPack = b.create<arith::DivSIOp>(
          loc,
          b.create<arith::AddIOp>(loc, nc, cI64(spec->packFactor - 1)),
          cI64(spec->packFactor));
      Value cPackBase =
          b.create<arith::DivSIOp>(loc, c, cI64(spec->packFactor));
      if (p.stride(qwView, 1) == 1 && bn % spec->packFactor == 0) {
        Value qBase = globalPtr(qwView, {kk, cPackBase}, packElem);
        ifThenElse(
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, nk, cI64(1)),
            {}, [&]() { memcpy1D(qwBuf, qBase, nPack, "GDRAM2NRAM"); },
            [&]() {
              memcpy2D(qwBuf, qBase, nPack, cI64(packCols),
                       cI64(p.stride(qwView, 0)),
                       b.create<arith::SubIOp>(loc, nk, cI64(1)),
                       "GDRAM2NRAM");
            });
      } else {
        forLoop(cI64(0), nk, cI64(1), [&](Value t) {
          Value kt = b.create<arith::AddIOp>(loc, kk, t);
          forLoop(cI64(0), nPack, cI64(1), [&](Value pIdx) {
            Value cp = b.create<arith::AddIOp>(loc, cPackBase, pIdx);
            Value dstIdx = b.create<arith::AddIOp>(
                loc, b.create<arith::MulIOp>(loc, t, cI64(packCols)), pIdx);
            storeAt(qwBuf, dstIdx,
                    loadAt(globalPtr(qwView, {kt, cp}, packElem), cI64(0),
                           packElem));
          });
        });
      }

      Value grpBase =
          b.create<arith::DivSIOp>(loc, kk, cI64(spec->blockElements));
      Value grpLast = b.create<arith::DivSIOp>(
          loc,
          b.create<arith::SubIOp>(
              loc, b.create<arith::AddIOp>(loc, kk, nk), cI64(1)),
          cI64(spec->blockElements));
      Value numGrp = b.create<arith::AddIOp>(
          loc, b.create<arith::SubIOp>(loc, grpLast, grpBase), cI64(1));

      if (p.stride(scView, 1) == 1) {
        Value scBase = globalPtr(scView, {grpBase, c}, scElem);
        ifThenElse(
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, numGrp,
                                    cI64(1)),
            {}, [&]() { memcpy1D(scBuf, scBase, nc, "GDRAM2NRAM"); },
            [&]() {
              memcpy2D(scBuf, scBase, nc, cI64(bn), cI64(p.stride(scView, 0)),
                       b.create<arith::SubIOp>(loc, numGrp, cI64(1)),
                       "GDRAM2NRAM");
            });
      } else {
        forLoop(cI64(0), numGrp, cI64(1), [&](Value gLocal) {
          Value grp = b.create<arith::AddIOp>(loc, grpBase, gLocal);
          forLoop(cI64(0), nc, cI64(1), [&](Value j) {
            Value cj = b.create<arith::AddIOp>(loc, c, j);
            Value dstIdx = b.create<arith::AddIOp>(
                loc, b.create<arith::MulIOp>(loc, gLocal, cI64(bn)), j);
            storeAt(scBuf, dstIdx,
                    loadAt(globalPtr(scView, {grp, cj}, scElem), cI64(0),
                           scElem));
          });
        });
      }

      if (spec->hasZeroView) {
        Value qzView = p.dot.getOperand(3);
        if (p.stride(qzView, 1) == 1 && bn % spec->packFactor == 0) {
          Value qzBase = globalPtr(qzView, {grpBase, cPackBase}, packElem);
          ifThenElse(
              b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, numGrp,
                                      cI64(1)),
              {}, [&]() { memcpy1D(qzBuf, qzBase, nPack, "GDRAM2NRAM"); },
              [&]() {
                memcpy2D(qzBuf, qzBase, nPack, cI64(packCols),
                         cI64(p.stride(qzView, 0)),
                         b.create<arith::SubIOp>(loc, numGrp, cI64(1)),
                         "GDRAM2NRAM");
              });
        } else {
          forLoop(cI64(0), numGrp, cI64(1), [&](Value gLocal) {
            Value grp = b.create<arith::AddIOp>(loc, grpBase, gLocal);
            forLoop(cI64(0), nPack, cI64(1), [&](Value pIdx) {
              Value cp = b.create<arith::AddIOp>(loc, cPackBase, pIdx);
              Value dstIdx = b.create<arith::AddIOp>(
                  loc, b.create<arith::MulIOp>(loc, gLocal, cI64(packCols)),
                  pIdx);
              storeAt(qzBuf, dstIdx,
                      loadAt(globalPtr(qzView, {grp, cp}, packElem), cI64(0),
                             packElem));
            });
          });
        }
      }
      sync("io");

      if (useGrpSimd) {
        Value kkMod =
            b.create<arith::RemSIOp>(loc, kk, cI64(spec->blockElements));
        SmallVector<int32_t, 8> lShifts(spec->laneBitShifts,
                                        spec->laneBitShifts + 8);
        makeOp("quant_decode_grouped_rhs",
               {transBuf, qwBuf, scBuf, spec->hasZeroView ? qzBuf : qzDummy,
                quantLut, idxScratch, workA, workB, nk, nc, cI64(bk), cI64(bn),
                cI64(spec->blockElements), kkMod,
                cI32(spec->hasZeroView ? 1 : 0)},
               {},
               {b.getNamedAttr("lane_shifts",
                               b.getDenseI32ArrayAttr(lShifts))},
               "quant-grouped-rhs");
        Value accum =
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, kk, cI64(0));
        sync("compute");
        memcpy1D(wFilter, transBuf, cI64(bk * bn), "NRAM2WRAM");
        makeOp("wram_conv", {acc, leftBuf, wFilter, accum}, {},
               {b.getNamedAttr("bm", b.getI64IntegerAttr(bm)),
                b.getNamedAttr("bn", b.getI64IntegerAttr(bn)),
                b.getNamedAttr("bk", b.getI64IntegerAttr(bk)),
                b.getNamedAttr("mode", b.getStringAttr("f16"))},
               "wram-conv-f16");
        sync("compute");
        return;
      }

      if (lhsElem.isF16() && bn == 64) {
        ifThen(
            b.create<arith::OrIOp>(
                loc,
                b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, nc,
                                        cI64(bn)),
                b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, nk,
                                        cI64(bk))),
            [&]() { fillVec(transBuf, cI64(bn * bk), zeroLhs); });
      } else if (!lhsElem.isF16()) {
        fillVec(rightF32, cI64(bk * bn), cF32(0.0f));
      }

      forLoop(cI64(0), numGrp, cI64(1), [&](Value gLocal) {
        Value grp = b.create<arith::AddIOp>(loc, grpBase, gLocal);
        Value tStart = b.create<arith::MaxSIOp>(
            loc, cI64(0),
            b.create<arith::SubIOp>(
                loc,
                b.create<arith::MulIOp>(loc, grp, cI64(spec->blockElements)),
                kk));
        Value tStop = b.create<arith::MinSIOp>(
            loc, nk,
            b.create<arith::SubIOp>(
                loc,
                b.create<arith::MulIOp>(
                    loc, b.create<arith::AddIOp>(loc, grp, cI64(1)),
                    cI64(spec->blockElements)),
                kk));
        Value scGrpBase = b.create<arith::MulIOp>(loc, gLocal, cI64(bn));
        Value qzGrpBase = b.create<arith::MulIOp>(loc, gLocal, cI64(packCols));

        forLoop(cI64(0), nPack, cI64(1), [&](Value pLocal) {
          Value qzWord;
          if (spec->hasZeroView) {
            qzWord = loadAt(
                qzBuf, b.create<arith::AddIOp>(loc, qzGrpBase, pLocal),
                packElem);
            if (!qzWord.getType().isInteger(32))
              qzWord = b.create<arith::ExtUIOp>(loc, b.getI32Type(), qzWord);
          }
          Value jBase =
              b.create<arith::MulIOp>(loc, pLocal, cI64(spec->packFactor));
          for (int sub = 0; sub < spec->packFactor; ++sub) {
            Value j = b.create<arith::AddIOp>(loc, jBase, cI64(sub));
            auto emitColBody = [&]() {
              Value zCode;
              if (spec->hasZeroView) {
                Value zSh =
                    spec->laneBitShifts[sub] > 0
                        ? b.create<arith::ShRUIOp>(
                               loc, qzWord, cI32(spec->laneBitShifts[sub]))
                              .getResult()
                        : qzWord;
                zCode = b.create<arith::AndIOp>(loc, zSh, cI32(codeMask));
              }
              Value scRaw = loadAt(
                  scBuf, b.create<arith::AddIOp>(loc, scGrpBase, j), scElem);
              Value scF =
                  scElem.isF32()
                      ? scRaw
                      : scElem.isF16()
                            ? b.create<arith::ExtFOp>(loc, b.getF32Type(),
                                                      scRaw)
                                  .getResult()
                            : makeOp("bf16_scalar", {scRaw}, {b.getF32Type()},
                                     {b.getNamedAttr("mode",
                                                     b.getStringAttr("widen"))},
                                     "bf16-scalar-convert")
                                  ->getResult(0);
              Value fltRowBase = b.create<arith::MulIOp>(loc, j, cI64(bk));
              forLoop(tStart, tStop, cI64(1), [&](Value t) {
                Value qIdx = b.create<arith::AddIOp>(
                    loc, b.create<arith::MulIOp>(loc, t, cI64(packCols)),
                    pLocal);
                Value qwRaw = loadAt(qwBuf, qIdx, packElem);
                if (!qwRaw.getType().isInteger(32))
                  qwRaw = b.create<arith::ExtUIOp>(loc, b.getI32Type(), qwRaw);
                Value qwSh =
                    spec->laneBitShifts[sub] > 0
                        ? b.create<arith::ShRUIOp>(
                               loc, qwRaw, cI32(spec->laneBitShifts[sub]))
                              .getResult()
                        : qwRaw;
                Value wCode =
                    b.create<arith::AndIOp>(loc, qwSh, cI32(codeMask));
                if (spec->hasZeroView)
                  wCode = b.create<arith::SubIOp>(loc, wCode, zCode);
                if (spec->zeroBias != 0)
                  wCode = b.create<arith::SubIOp>(loc, wCode,
                                                  cI32(spec->zeroBias));
                Value decF = b.create<arith::MulFOp>(
                    loc, decodeScalarQuantCode(wCode, *spec), scF);
                if (lhsElem.isF16() && bn == 64) {
                  Value decH =
                      b.create<arith::TruncFOp>(loc, b.getF16Type(), decF);
                  storeAt(transBuf,
                          b.create<arith::AddIOp>(loc, fltRowBase, t), decH);
                } else {
                  Value rIdx = b.create<arith::AddIOp>(
                      loc, b.create<arith::MulIOp>(loc, t, cI64(bn)), j);
                  storeAt(rightF32, rIdx, decF);
                }
              });
            };
            if (bn % spec->packFactor == 0 && p.shape[1] % bn == 0)
              emitColBody();
            else
              ifThen(b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, j,
                                             nc),
                     emitColBody);
          }
        });
      });

      if (lhsElem.isF16() && bn == 64) {
        Value accum =
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, kk, cI64(0));
        sync("compute");
        memcpy1D(wFilter, transBuf, cI64(bk * bn), "NRAM2WRAM");
        makeOp("wram_conv", {acc, leftBuf, wFilter, accum}, {},
               {b.getNamedAttr("bm", b.getI64IntegerAttr(bm)),
                b.getNamedAttr("bn", b.getI64IntegerAttr(bn)),
                b.getNamedAttr("bk", b.getI64IntegerAttr(bk)),
                b.getNamedAttr("mode", b.getStringAttr("f16"))},
               "wram-conv-f16");
        sync("compute");
      } else {
        forLoop(cI64(0), nr, cI64(1), [&](Value i) {
          Value accRow = makeAddr(acc, {i}, {cI64(bn)}, 1);
          forLoop(cI64(0), nk, cI64(1), [&](Value t) {
            Value lIdx = b.create<arith::AddIOp>(
                loc, b.create<arith::MulIOp>(loc, i, cI64(bk)), t);
            Value lvRaw = loadAt(leftBuf, lIdx, lhsElem);
            Value lv = lhsElem.isF32()
                           ? lvRaw
                           : lhsElem.isF16()
                                 ? b.create<arith::ExtFOp>(loc, b.getF32Type(),
                                                           lvRaw)
                                       .getResult()
                                 : makeOp("bf16_scalar", {lvRaw},
                                          {b.getF32Type()},
                                          {b.getNamedAttr(
                                              "mode", b.getStringAttr("widen"))},
                                          "bf16-scalar-convert")
                                       ->getResult(0);
            Value rRow = makeAddr(rightF32, {t}, {cI64(bn)}, 1);
            makeOp("vec_scalar", {tmpF32, rRow, lv, cI64(bn)}, {},
                   {b.getNamedAttr("op", b.getStringAttr("mul"))});
            makeOp("vec_binary", {accRow, accRow, tmpF32, cI64(bn)}, {},
                   {b.getNamedAttr("op", b.getStringAttr("add"))});
          });
        });
      }
    });
    return success();
  }

  LogicalResult emitScaledMX8(Value acc, Value r, Value c, Value nr, Value nc) {
    auto mxAttr = p.dot.getAttrOfType<TypeAttr>("kir.storage");
    if (!mxAttr)
      return p.dot.emitError("missing KIR MX8 storage type");
    auto mx = cast<intentdsl::kir::MX8Type>(mxAttr.getValue()).getLayout();
    int64_t bn = p.tile[1];
    Value a = p.dot.getOperand(0), sa = p.dot.getOperand(1),
          rhs = p.dot.getOperand(2);
    intentdsl::kir::RightScale rightScale(p.dot);
    Value rightBuf = nramBuf(b.getF32Type(), mx.group * bn);
    Value tmpBuf = nramBuf(b.getF32Type(), bn);

    fillVec(acc, cI64(p.tile[0] * bn), cF32(0.0f));
    int64_t groups = p.shape[2] / mx.group;
    forLoop(cI64(0), cI64(groups), cI64(1), [&](Value u) {
      forLoop(cI64(0), nc, cI64(1), [&](Value j) {
        Value cj = b.create<arith::AddIOp>(loc, c, j);
        SmallVector<Value> sbCoords(2);
        sbCoords[rightScale.column] = cj;
        sbCoords[rightScale.group] = u;
        Value sbByte = b.create<arith::ExtUIOp>(
            loc, b.getI32Type(),
            loadAt(globalPtr(rightScale.view, sbCoords, b.getI8Type()), cI64(0),
                   b.getI8Type()));
        forLoop(cI64(0), cI64(mx.group), cI64(1), [&](Value t) {
          Value bByte = b.create<arith::ExtUIOp>(
              loc, b.getI32Type(),
              loadAt(globalPtr(rhs, {u, t, cj}, b.getI8Type()), cI64(0),
                     b.getI8Type()));
          Value dec = decodeMX8(bByte, sbByte, mx);
          Value idx = b.create<arith::AddIOp>(
              loc, b.create<arith::MulIOp>(loc, t, cI64(bn)), j);
          storeAt(rightBuf, idx, dec);
        });
      });

      forLoop(cI64(0), nr, cI64(1), [&](Value i) {
        Value ri = b.create<arith::AddIOp>(loc, r, i);
        Value saByte = b.create<arith::ExtUIOp>(
            loc, b.getI32Type(),
            loadAt(globalPtr(sa, {ri, u}, b.getI8Type()), cI64(0),
                   b.getI8Type()));
        Value accRow = makeAddr(acc, {i}, {cI64(bn)}, 1);
        forLoop(cI64(0), cI64(mx.group), cI64(1), [&](Value t) {
          Value aByte = b.create<arith::ExtUIOp>(
              loc, b.getI32Type(),
              loadAt(globalPtr(a, {ri, u, t}, b.getI8Type()), cI64(0),
                     b.getI8Type()));
          Value av = decodeMX8(aByte, saByte, mx);
          Value rRow = makeAddr(rightBuf, {t}, {cI64(bn)}, 1);
          makeOp("vec_scalar", {tmpBuf, rRow, av, cI64(bn)}, {},
                 {b.getNamedAttr("op", b.getStringAttr("mul"))});
          makeOp("vec_binary", {accRow, accRow, tmpBuf, cI64(bn)}, {},
                 {b.getNamedAttr("op", b.getStringAttr("add"))});
        });
      });
    });
    return success();
  }

  // Lower window_contract row-scale sub-DAG on BK-lane NRAM vectors.
  LogicalResult emitVectorRowScale(Value scales, Value stepbuf, Value prefixbuf,
                                   Value lastVal, Value nk, int64_t bk,
                                   Value stopVal, Value expScratch) {
    auto &region = p.dot.getRegion(0).front();
    struct Val {
      Value v;
      bool isVec = false;
    };
    llvm::DenseMap<Value, Val> env;
    env[region.getArgument(1)] = {stepbuf, true};
    env[region.getArgument(2)] = {prefixbuf, true};
    env[region.getArgument(3)] = {lastVal, false};
    if (env.count(stopVal)) {
      Val s = env.lookup(stopVal);
      if (s.isVec)
        memcpy1D(scales, s.v, cI64(bk), "NRAM2NRAM");
      else
        fillVec(scales, cI64(bk), s.v);
    } else {
      for (auto &op : region) {
        StringRef n = op.getName().getStringRef();
        if (op.getName().stripDialect() == "panel_value")
          break;
        if (op.getOperand(0) == region.getArgument(0))
          continue;
        if (n == "arith.constant") {
          float fv =
              op.getAttrOfType<FloatAttr>("value").getValueAsDouble();
          env[op.getResult(0)] = {cF32(fv), false};
        } else if (n == "arith.addf" || n == "arith.subf" ||
                   n == "arith.mulf") {
          if (!env.count(op.getOperand(0)) || !env.count(op.getOperand(1)))
            continue;
          Val a = env.lookup(op.getOperand(0)),
              rhs = env.lookup(op.getOperand(1));
          StringRef k = n == "arith.addf"   ? "add"
                        : n == "arith.subf" ? "sub"
                                            : "mul";
          if (!a.isVec && !rhs.isVec) {
            Value r = n == "arith.addf"
                          ? b.create<arith::AddFOp>(loc, a.v, rhs.v).getResult()
                      : n == "arith.subf"
                          ? b.create<arith::SubFOp>(loc, a.v, rhs.v).getResult()
                          : b.create<arith::MulFOp>(loc, a.v, rhs.v)
                                .getResult();
            env[op.getResult(0)] = {r, false};
          } else {
            Value dst = (op.getResult(0) == stopVal)
                            ? scales
                            : nramBuf(b.getF32Type(), bk);
            if (a.isVec && rhs.isVec) {
              makeOp("vec_binary", {dst, a.v, rhs.v, cI64(bk)}, {},
                     {b.getNamedAttr("op", b.getStringAttr(k))});
            } else if (a.isVec && !rhs.isVec) {
              makeOp("vec_scalar", {dst, a.v, rhs.v, cI64(bk)}, {},
                     {b.getNamedAttr("op", b.getStringAttr(k))});
            } else if (k != "sub") {
              makeOp("vec_scalar", {dst, rhs.v, a.v, cI64(bk)}, {},
                     {b.getNamedAttr("op", b.getStringAttr(k))});
            } else {
              makeOp("vec_scalar", {dst, rhs.v, cF32(-1.0f), cI64(bk)}, {},
                     {b.getNamedAttr("op", b.getStringAttr("mul"))});
              makeOp("vec_scalar", {dst, dst, a.v, cI64(bk)}, {},
                     {b.getNamedAttr("op", b.getStringAttr("add"))});
            }
            env[op.getResult(0)] = {dst, true};
          }
        } else if (n == "arith.minimumf") {
          if (!env.count(op.getOperand(0)) || !env.count(op.getOperand(1)))
            continue;
          Val a = env.lookup(op.getOperand(0)),
              rhs = env.lookup(op.getOperand(1));
          Value dst = (op.getResult(0) == stopVal)
                          ? scales
                          : nramBuf(b.getF32Type(), bk);
          Val vecIn = a.isVec ? a : rhs;
          Val scIn = a.isVec ? rhs : a;
          if (!vecIn.isVec || scIn.isVec)
            return p.dot.emitError("unsupported non-scalar window min bound");
          makeOp("vec_min_scalar",
                 {dst, expScratch, vecIn.v, scIn.v, nk, cI64(bk)}, {}, {},
                 "ieee-vec-min");
          env[op.getResult(0)] = {dst, true};
        } else if (n == "math.exp") {
          if (!env.count(op.getOperand(0)))
            continue;
          Val a = env.lookup(op.getOperand(0));
          Value dst = (op.getResult(0) == stopVal)
                          ? scales
                          : nramBuf(b.getF32Type(), bk);
          makeOp("row_exp", {dst, expScratch, a.v, nk, cI64(bk)}, {}, {},
                 "validated-exp");
          env[op.getResult(0)] = {dst, true};
        }
        if (op.getResult(0) == stopVal)
          break;
      }
    }
    ifThen(b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, nk,
                                   cI64(bk)),
           [&]() {
             forLoop(nk, cI64(bk), cI64(1),
                     [&](Value k) { storeAt(scales, k, cF32(0.0f)); });
           });
    return success();
  }

  LogicalResult emitScalarWindowValue(Value rawVal, Value dVal, Value vVal,
                                      Value lastVal, Value &out) {
    auto &region = p.dot.getRegion(0).front();
    llvm::DenseMap<Value, Value> env;
    env[region.getArgument(0)] = rawVal;
    env[region.getArgument(1)] = dVal;
    env[region.getArgument(2)] = vVal;
    env[region.getArgument(3)] = lastVal;
    for (auto &op : region) {
      if (op.getName().stripDialect() == "panel_value") {
        out = env.lookup(op.getOperand(0));
        return success();
      }
      StringRef n = op.getName().getStringRef();
      if (n == "arith.constant") {
        float fv = op.getAttrOfType<FloatAttr>("value").getValueAsDouble();
        env[op.getResult(0)] = cF32(fv);
      } else if (n == "arith.extf") {
        env[op.getResult(0)] =
            makeOp("bf16_scalar", {env.lookup(op.getOperand(0))},
                   {b.getF32Type()},
                   {b.getNamedAttr("mode", b.getStringAttr("widen"))},
                   "bf16-scalar-convert")
                ->getResult(0);
      } else if (n == "arith.truncf") {
        env[op.getResult(0)] =
            makeOp("bf16_scalar", {env.lookup(op.getOperand(0))},
                   {b.getBF16Type()},
                   {b.getNamedAttr("mode", b.getStringAttr("narrow"))},
                   "bf16-scalar-convert")
                ->getResult(0);
      } else if (n == "math.exp") {
        env[op.getResult(0)] =
            makeOp("scalar_math", {env.lookup(op.getOperand(0))},
                   {b.getF32Type()},
                   {b.getNamedAttr("op", b.getStringAttr("exp"))})
                ->getResult(0);
      } else if (n == "arith.minimumf") {
        env[op.getResult(0)] =
            makeOp("scalar_math",
                   {env.lookup(op.getOperand(0)), env.lookup(op.getOperand(1))},
                   {b.getF32Type()},
                   {b.getNamedAttr("op", b.getStringAttr("min_ieee"))})
                ->getResult(0);
      } else if (n == "arith.addf") {
        env[op.getResult(0)] = b.create<arith::AddFOp>(
            loc, env.lookup(op.getOperand(0)), env.lookup(op.getOperand(1)));
      } else if (n == "arith.subf") {
        env[op.getResult(0)] = b.create<arith::SubFOp>(
            loc, env.lookup(op.getOperand(0)), env.lookup(op.getOperand(1)));
      } else if (n == "arith.mulf") {
        env[op.getResult(0)] = b.create<arith::MulFOp>(
            loc, env.lookup(op.getOperand(0)), env.lookup(op.getOperand(1)));
      } else {
        return p.dot.emitError("unsupported window panel scalar operation");
      }
    }
    return failure();
  }

  LogicalResult emitOperandStage(bool lhs, Value left, Value right, Value trans,
                                 Value mapped, Value stepbuf, Value prefixbuf,
                                 Value scales, Value expScratch, Value r,
                                 Value c, Value gVal, Value kk, Value nr,
                                 Value nc, Value nk, Value batchVal,
                                 Value chunkVal, Value headVal, Value groupVal,
                                 Value lastVal) {
    int64_t bm = p.tile[0], bn = p.tile[1], bk = p.tile[2];
    Value v = p.dot.getOperand(lhs ? 0 : 1);
    auto &feeds = p.fold.getRegion(0).front().front().getRegion(0).front();
    StringRef mode = (lhs ? feeds.front() : *feeds.front().getNextNode())
                         .getAttrOfType<StringAttr>("bangc.transfer")
                         .getValue();

    if (p.dot.getName().stripDialect() == "window_contract") {
      Value seqBase = b.create<arith::AddIOp>(
          loc, b.create<arith::MulIOp>(loc, chunkVal, cI64(p.shape[2])), kk);
      if (mode == "transpose-window-dma") {
        Value needZero = b.create<arith::OrIOp>(
            loc,
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, nr,
                                    cI64(bm)),
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, nk,
                                    cI64(bk)));
        ifThen(needZero, [&]() {
          fillVec(trans, cI64(bk * bm),
                  b.create<arith::ConstantOp>(
                      loc, b.getZeroAttr(b.getBF16Type())));
        });
        Value srcPtr = globalPtr(v, {batchVal, seqBase, headVal, r});
        Value seg = b.create<arith::SubIOp>(loc, nk, cI64(1));
        ifThenElse(
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, nk, cI64(1)),
            {}, [&]() { memcpy1D(trans, srcPtr, nr, "GDRAM2NRAM"); },
            [&]() {
              memcpy2D(trans, srcPtr, nr, cI64(bm), cI64(p.stride(v, 1)), seg,
                       "GDRAM2NRAM");
            });
        makeOp("tile_transpose", {left, trans, cI64(bk), cI64(bm)}, {}, {},
               "nram-transpose");
        return success();
      }
      if (mode == "mapped-filter-major-dma") {
        Value stepView = p.dot.getOperand(2), prefView = p.dot.getOperand(3);
        Value needZero = b.create<arith::OrIOp>(
            loc,
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, nc,
                                    cI64(bn)),
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, nk,
                                    cI64(bk)));
        ifThen(needZero, [&]() {
          fillVec(right, cI64(bk * bn),
                  b.create<arith::ConstantOp>(
                      loc, b.getZeroAttr(b.getBF16Type())));
        });
        Value bPtr = globalPtr(v, {batchVal, seqBase, groupVal, c});
        Value seg = b.create<arith::SubIOp>(loc, nk, cI64(1));
        ifThenElse(
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, nk, cI64(1)),
            {}, [&]() { memcpy1D(right, bPtr, nc, "GDRAM2NRAM"); },
            [&]() {
              memcpy2D(right, bPtr, nc, cI64(bn), cI64(p.stride(v, 1)), seg,
                       "GDRAM2NRAM");
            });
        for (auto [view, buf] :
             {std::pair{stepView, stepbuf}, std::pair{prefView, prefixbuf}}) {
          Value ptr = globalPtr(view, {batchVal, headVal, chunkVal, kk});
          if (p.stride(view, 3) == 1) {
            memcpy1D(buf, ptr, nk, "GDRAM2NRAM");
          } else {
            ifThenElse(
                b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, nk,
                                        cI64(1)),
                {}, [&]() { memcpy1D(buf, ptr, cI64(1), "GDRAM2NRAM"); },
                [&]() {
                  memcpy2D(buf, ptr, cI64(1), cI64(1), cI64(p.stride(view, 3)),
                           seg, "GDRAM2NRAM");
                });
          }
        }
        sync("io");
        Value stopVal = intentdsl::kir::rowScale(p.dot);
        if (failed(emitVectorRowScale(scales, stepbuf, prefixbuf, lastVal, nk,
                                      bk, stopVal, expScratch)))
          return failure();
        makeOp("tile_transpose", {trans, right, cI64(bk), cI64(bn)}, {}, {},
               "nram-transpose");
        makeOp("bf16_tile", {mapped, trans, cI64(bk * bn), cI64(bk * bn)}, {},
               {b.getNamedAttr("mode", b.getStringAttr("widen"))},
               "bf16-tile-convert");
        makeOp("cycle_mul",
               {mapped, mapped, scales, cI64(bn * bk), cI64(bk)}, {}, {},
               "cycle-mul");
        makeOp("bf16_tile", {trans, mapped, cI64(bn * bk), cI64(bn * bk)}, {},
               {b.getNamedAttr("mode", b.getStringAttr("narrow"))},
               "bf16-tile-convert");
        return success();
      }

      Value targetBuf = lhs ? left : right;
      fillVec(targetBuf, cI64(lhs ? bm * bk : bk * bn),
              b.create<arith::ConstantOp>(loc, b.getZeroAttr(b.getBF16Type())));
      if (lhs) {
        forLoop(cI64(0), nk, cI64(1), [&](Value k) {
          Value seqK = b.create<arith::AddIOp>(loc, seqBase, k);
          forLoop(cI64(0), nr, cI64(1), [&](Value i) {
            Value ri = b.create<arith::AddIOp>(loc, r, i);
            Value val = loadAt(globalPtr(v, {batchVal, seqK, headVal, ri}),
                               cI64(0), b.getBF16Type());
            Value dstIdx = b.create<arith::AddIOp>(
                loc, b.create<arith::MulIOp>(loc, i, cI64(bk)), k);
            storeAt(left, dstIdx, val);
          });
        });
      } else {
        Value stepView = p.dot.getOperand(2), prefView = p.dot.getOperand(3);
        LogicalResult st = success();
        forLoop(cI64(0), nk, cI64(1), [&](Value k) {
          Value kkPlusK = b.create<arith::AddIOp>(loc, kk, k);
          Value seqK = b.create<arith::AddIOp>(loc, seqBase, k);
          Value dVal =
              loadAt(globalPtr(stepView, {batchVal, headVal, chunkVal, kkPlusK}),
                     cI64(0), b.getF32Type());
          Value pVal =
              loadAt(globalPtr(prefView, {batchVal, headVal, chunkVal, kkPlusK}),
                     cI64(0), b.getF32Type());
          forLoop(cI64(0), nc, cI64(1), [&](Value j) {
            Value cj = b.create<arith::AddIOp>(loc, c, j);
            Value raw = loadAt(globalPtr(v, {batchVal, seqK, groupVal, cj}),
                               cI64(0), b.getBF16Type());
            Value mappedVal;
            if (failed(
                    emitScalarWindowValue(raw, dVal, pVal, lastVal, mappedVal)))
              st = failure();
            else {
              Value dstIdx = b.create<arith::AddIOp>(
                  loc, b.create<arith::MulIOp>(loc, k, cI64(bn)), j);
              storeAt(right, dstIdx, mappedVal);
            }
          });
        });
        if (failed(st))
          return failure();
      }
      return success();
    }

    int kAx = p.axis(lhs ? "lhs_k" : "rhs_k");
    int bAx = p.axis(lhs ? "lhs_batch" : "rhs_batch");
    int memberAx = p.freeAxis(v, kAx, bAx);
    Type elemTy = carrierElemType(v);
    Type loadTy = elemTy.isSignedInteger(8) ? Type(b.getI32Type()) : elemTy;
    Value zeroElem = elemTy.isSignedInteger(8)
                         ? cI32(0)
                         : b.create<arith::ConstantOp>(loc, b.getZeroAttr(elemTy))
                               .getResult();

    if (mode == "filter-major-dma") {
      Value partial = b.create<arith::OrIOp>(
          loc,
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, nc, cI64(bn)),
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, nk, cI64(bk)));
      auto guardIf =
          ifThen(partial, [&]() { fillVec(trans, cI64(bk * bn), zeroElem); });
      Value base = termPtr(v, kAx, bAx, c, kk, gVal);
      if (Operation *addrOp = base.getDefiningOp()) {
        addrOp->setAttr("bangc.tile_axes",
                        b.getDenseI64ArrayAttr({memberAx, kAx}));
        addrOp->setAttr("bangc.tile_sizes", b.getDenseI64ArrayAttr({bn, bk}));
        tailGuards.push_back(PendingTailGuard{guardIf.getOperation(), addrOp});
      }
      ifThenElse(
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, nc, cI64(1)),
          {}, [&]() { memcpy1D(trans, base, nk, "GDRAM2NRAM"); },
          [&]() {
            memcpy2D(trans, base, nk, cI64(bk), cI64(p.stride(v, memberAx)),
                     b.create<arith::SubIOp>(loc, nc, cI64(1)), "GDRAM2NRAM");
          });
      return success();
    }

    Value buf = lhs ? left : right;
    Value outer = lhs ? nr : nk;
    Value inner = lhs ? nk : nc;
    int64_t outerCap = lhs ? bm : bk;
    int64_t pitch = lhs ? bk : bn;
    Value partial = b.create<arith::OrIOp>(
        loc,
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, outer,
                                cI64(outerCap)),
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, inner,
                                cI64(pitch)));
    auto guardIf = ifThen(
        partial, [&]() { fillVec(buf, cI64(outerCap * pitch), zeroElem); });

    if (mode == "routed-row-dma") {
      forLoop(cI64(0), nr, cI64(1), [&](Value i) {
        Value dstRow = makeAddr(left, {i}, {cI64(bk)}, 1);
        Value srcRow = termPtr(v, kAx, bAx, inputRow(r, i), kk, gVal);
        memcpy1D(dstRow, srcRow, nk, "GDRAM2NRAM");
      });
    } else if (mode == "pitched-dma") {
      Value base =
          termPtr(v, kAx, bAx, lhs ? inputRow(r, cI64(0)) : c, kk, gVal);
      if (!lhs || (!p.routed() && p.dot.getNumOperands() <= 2)) {
        if (Operation *addrOp = base.getDefiningOp()) {
          int64_t outAx = lhs ? memberAx : kAx;
          int64_t inAx = lhs ? kAx : memberAx;
          addrOp->setAttr("bangc.tile_axes",
                          b.getDenseI64ArrayAttr({outAx, inAx}));
          addrOp->setAttr("bangc.tile_sizes",
                          b.getDenseI64ArrayAttr({outerCap, pitch}));
          tailGuards.push_back(
              PendingTailGuard{guardIf.getOperation(), addrOp});
        }
      }
      int64_t outerStride = p.stride(v, lhs ? memberAx : kAx);
      ifThenElse(
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, outer,
                                  cI64(1)),
          {}, [&]() { memcpy1D(buf, base, inner, "GDRAM2NRAM"); },
          [&]() {
            memcpy2D(buf, base, inner, cI64(pitch), cI64(outerStride),
                     b.create<arith::SubIOp>(loc, outer, cI64(1)),
                     "GDRAM2NRAM");
          });
    } else {
      forLoop(cI64(0), outer, cI64(1), [&](Value i) {
        forLoop(cI64(0), inner, cI64(1), [&](Value j) {
          Value freeC =
              lhs ? inputRow(r, i) : b.create<arith::AddIOp>(loc, c, j);
          Value redC = b.create<arith::AddIOp>(loc, kk, lhs ? j : i);
          Value addr = termPtr(v, kAx, bAx, freeC, redC, gVal);
          Value dstIdx = b.create<arith::AddIOp>(
              loc, b.create<arith::MulIOp>(loc, i, cI64(pitch)), j);
          storeAt(buf, dstIdx, loadAt(addr, cI64(0), loadTy));
        });
      });
    }
    return success();
  }

  LogicalResult run(ModuleOp m) {
    int64_t bm = p.tile[0], bn = p.tile[1], bk = p.tile[2];
    int64_t shapeM = p.shape[0], shapeN = p.shape[1], shapeK = p.shape[2],
            shapeB = p.shape[3];
    auto &setup = p.fold.getRegion(0).front().front();
    int64_t tasks = setup.getAttrOfType<IntegerAttr>("participants").getInt();
    int64_t scratchBytes =
        setup.getAttrOfType<IntegerAttr>("scratch_bytes").getInt();
    StringRef mode = p.dot.getAttrOfType<StringAttr>("bangc.matrix").getValue();
    StringRef writeback =
        setup.getAttrOfType<StringAttr>("bangc.writeback").getValue();
    bool hasBf16Guard = p.dot.hasAttr("bangc.bf16_arithmetic");
    bool isWindow = p.dot.getName().stripDialect() == "window_contract";
    bool isRagged = mode != "q4k-q8k" &&
                    p.dot.getName().stripDialect() == "contract" &&
                    p.dot.getNumOperands() > 2;
    bool isRouted = p.routed();
    Value routedIds = isRouted ? p.dot.getOperand(3) : Value{};
    Value raggedOffsets = isRagged ? p.dot.getOperand(2) : Value{};
    Value dst = p.domain.getOperand(0);
    Type outElem = carrierElemType(dst);

    struct EpilogueOp {
      StringRef kind;
      Value bias;
      Type fromTy, toTy;
    };
    SmallVector<EpilogueOp> epilogue;
    for (auto &ep : p.domain.getRegion(0).front()) {
      StringRef en = ep.getName().stripDialect();
      if (en == "add_columns")
        epilogue.push_back(
            {en, ep.getOperand(1),
             cast<ShapedType>(ep.getOperand(0).getType()).getElementType(),
             cast<ShapedType>(ep.getResult(0).getType()).getElementType()});
      else if (en == "convert_matrix")
        epilogue.push_back(
            {en, Value{},
             cast<ShapedType>(ep.getOperand(0).getType()).getElementType(),
             cast<ShapedType>(ep.getResult(0).getType()).getElementType()});
    }

    int64_t headGroup = isWindow ? p.axis("head_group") : 1;
    DenseI64ArrayAttr dstShape =
        fn.getArgAttrOfType<DenseI64ArrayAttr>(p.slot(dst), "block.shape");

    b.setInsertionPointToStart(&fn.front());
    OperationState progState(loc, "bangc_src.matrix_program");
    progState.addAttribute("tile_m", b.getI64IntegerAttr(bm));
    progState.addAttribute("tile_n", b.getI64IntegerAttr(bn));
    progState.addAttribute("tile_k", b.getI64IntegerAttr(bk));
    progState.addAttribute("extents", p.shape);
    progState.addAttribute("participants", b.getI64IntegerAttr(tasks));
    progState.addAttribute("scratch_bytes", b.getI64IntegerAttr(scratchBytes));
    progState.addRegion();
    progState.addRegion();
    progState.addRegion();
    Operation *prog = b.create(progState);
    Block *sourcesBlk = b.createBlock(&prog->getRegion(0));
    Block *deviceBlk = b.createBlock(&prog->getRegion(1));
    Block *hostBlk = b.createBlock(&prog->getRegion(2));

    if (auto lib = setup.getAttrOfType<StringAttr>("library_impl")) {
      auto plan = intentdsl::dsa::matchLibraryMatrix(p, profile);
      if (!plan || plan->impl != lib.getValue())
        return p.dot.emitError("external CNNL matrix plan mismatch");
      prog->setAttr("cnnl", b.getUnitAttr());
      b.setInsertionPointToStart(hostBlk);
      OperationState cs(loc, "bangc_src.cnnl_matmul");
      cs.addOperands(fn.getArguments());
      cs.addAttribute("lhs_slot", b.getI64IntegerAttr(plan->lhsSlot));
      cs.addAttribute("rhs_slot", b.getI64IntegerAttr(plan->rhsSlot));
      cs.addAttribute("dst_slot", b.getI64IntegerAttr(plan->dstSlot));
      cs.addAttribute("batch", b.getI64IntegerAttr(plan->batch));
      cs.addAttribute("m", b.getI64IntegerAttr(plan->m));
      cs.addAttribute("n", b.getI64IntegerAttr(plan->n));
      cs.addAttribute("k", b.getI64IntegerAttr(plan->k));
      cs.addAttribute("trans_a", b.getI32IntegerAttr(plan->transA));
      cs.addAttribute("trans_b", b.getI32IntegerAttr(plan->transB));
      cs.addAttribute("lhs_batch_stride",
                      b.getI64IntegerAttr(plan->lhsBatchStride));
      cs.addAttribute("rhs_batch_stride",
                      b.getI64IntegerAttr(plan->rhsBatchStride));
      cs.addAttribute("dst_batch_stride",
                      b.getI64IntegerAttr(plan->dstBatchStride));
      cs.addAttribute("in_dtype", b.getI32IntegerAttr(plan->inDtype));
      cs.addAttribute("out_dtype", b.getI32IntegerAttr(plan->outDtype));
      cs.addAttribute("ci.implementation", lib);
      b.create(cs);

      uint64_t grid = llvm::divideCeil(uint64_t(shapeM), uint64_t(bm)) *
                      llvm::divideCeil(uint64_t(shapeN), uint64_t(bn)) *
                      uint64_t(shapeB);
      auto abiJson = llvm::json::parse(
          fn->getAttrOfType<StringAttr>("kir.abi").getValue());
      if (!abiJson)
        return llvm::consumeError(abiJson.takeError()), failure();
      p.domain.erase();
      llvm::SmallVector<char> metadata;
      llvm::raw_svector_ostream os(metadata);
      os << llvm::json::Value(llvm::json::Object{
          {"version", 2},
          {"source", ""},
          {"cnnl", true},
          {"abi", std::move(*abiJson)},
          {"domain", intentdsl::publishDomain(fn)},
          {"grid", llvm::json::Array{int64_t(grid), 1, 1}},
          {"guard", llvm::json::Object{{"min_width", 0}, {"max_width", nullptr}}},
          {"rows",
           llvm::json::Array{llvm::json::Object{{"id", 0},
                                                {"rows", bm},
                                                {"width", bn},
                                                {"pitch", bk},
                                                {"bytes", scratchBytes},
                                                {"tasks", tasks}}}}});
      m->setAttr("bangc_src.package",
                 b.getStringAttr(StringRef(metadata.data(), metadata.size())));
      return success();
    }

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
    symBM = makeSym("BM");
    symBN = makeSym("BN");
    symBK = makeSym("BK");
    constAnchor = symBK.getDefiningOp();

    Type accElem = (mode == "exact-i8-conv") ? Type(b.getI32Type())
                                             : Type(b.getF32Type());
    Value acc = nramBuf(accElem, bm * bn);
    Value left, right, trans, filter, partial, flags, mapped, stepbuf,
        prefixbuf, scales, expScratch, biasBuf;
    Type inElem = carrierElemType(p.dot.getOperand(0));
    if (mode == "wram-conv" || mode == "exact-i8-conv" || mode == "f32-outer") {
      left = nramBuf(inElem, bm * bk);
      right = nramBuf(inElem, bk * bn);
      if (mode != "f32-outer") {
        trans = nramBuf(inElem, bk * bn);
        filter = wramBuf(inElem, bk * bn);
      }
      if (mode == "exact-i8-conv")
        partial = nramBuf(b.getF32Type(), bm * bn);
    }
    if (hasBf16Guard)
      flags = nramBuf(b.getBF16Type(), bk * bn);
    if (isWindow) {
      mapped = nramBuf(b.getF32Type(), bk * bn);
      stepbuf = nramBuf(b.getF32Type(), bk);
      prefixbuf = nramBuf(b.getF32Type(), bk);
      scales = nramBuf(b.getF32Type(), bk);
      expScratch = nramBuf(b.getF32Type(), std::max<int64_t>(bk, 64));
    }
    if (writeback == "pitched-dma") {
      for (auto &ep : epilogue)
        if (ep.kind == "add_columns")
          biasBuf = nramBuf(carrierElemType(ep.bias), bn);
    }

    Value quantLut, quantRecLut, qActAllF32, dsAll, zeroAll;
    if (mode == "quant-format") {
      if (const auto *qSpec = intentdsl::kir::quantFormatSpec(p.dot)) {
        if ((qSpec->roleLayout ==
                 intentdsl::kir::QuantRoleLayout::BlockScaledPair ||
             qSpec->roleLayout ==
                 intentdsl::kir::QuantRoleLayout::GroupedWeightRhs) &&
            qSpec->packBits == 4) {
          quantLut = nramBuf(b.getI32Type(), 256);
          int32_t vc =
              qSpec->valueCode == intentdsl::kir::QuantValueCode::Fp4E2M1 ? 1
                                                                          : 0;
          int32_t s0 =
              qSpec->packFactor == 2 ? qSpec->laneBitShifts[0] : 0;
          int32_t s1 =
              qSpec->packFactor == 2 ? qSpec->laneBitShifts[1] : 4;
          makeOp("quant_init_lut",
                 {quantLut, cI32(vc), cI32(qSpec->packBits),
                  cI32(qSpec->zeroBias), cI32(s0), cI32(s1)},
                 {}, {}, "quant-lut-init");
        } else if (qSpec->roleLayout ==
                       intentdsl::kir::QuantRoleLayout::RecordGemv &&
                   qSpec->q8kActivation && qSpec->numBitPlanes == 2 &&
                   qSpec->blockElements == 256 &&
                   qSpec->subBlockElements == 16 && bm <= 128 &&
                   bm * bn >= std::max<int64_t>(bm, 64) &&
                   p.stride(p.dot.getOperand(0), 2) == 1) {
          quantRecLut = nramBuf(b.getI16Type(), 1536);
          const auto &p0 = qSpec->planes[0], &p1 = qSpec->planes[1];
          makeOp("quant_init_record_lut",
                 {quantRecLut, cI32(p0.shiftUnit), cI32(p0.bitWidth),
                  cI32(qSpec->zeroBias), cI32(p1.shiftUnit), cI32(p1.bitWidth),
                  cI32(p1.destBitOffset)},
                 {}, {}, "quant-record-lut-init");
          int64_t recGroups = shapeK / 256;
          if (recGroups > 0 && recGroups <= 64) {
            Value xView = p.dot.getOperand(1).getDefiningOp()->getOperand(0);
            int64_t xSt1 = p.stride(xView, 1);
            Value actTmp = nramBuf(b.getF32Type(), 256);
            qActAllF32 = nramBuf(b.getF32Type(), recGroups * 256);
            dsAll = nramBuf(b.getF32Type(), llvm::alignTo(recGroups, 32));
            zeroAll = nramBuf(b.getI32Type(), llvm::alignTo(recGroups, 32));
            intentdsl::kir::Q8KRecord q8;
            int32_t qMin = -(1 << (q8.elementBits - 1));
            int32_t qMax = (1 << (q8.elementBits - 1)) - 1;
            forLoop(cI64(0), cI64(recGroups), cI64(1), [&](Value gq) {
              if (xSt1 == 1) {
                memcpy1D(actTmp, globalPtr(xView, {gq, cI64(0)}), cI64(256),
                         "GDRAM2NRAM");
                sync("io");
              } else {
                forLoop(cI64(0), cI64(256), cI64(1), [&](Value k) {
                  storeAt(
                      actTmp, k,
                      loadAt(globalPtr(xView, {gq, k}), cI64(0), b.getF32Type()));
                });
                sync("all");
              }
              auto maxLp = forLoopCarry(
                  cI64(0), cI64(256), cI64(1), ValueRange{cF32(0.0f)},
                  [&](Value k, ValueRange iter) {
                    Value xk = loadAt(actTmp, k, b.getF32Type());
                    Value ax =
                        makeOp("scalar_math", {xk}, {b.getF32Type()},
                               {b.getNamedAttr("op", b.getStringAttr("fabs"))})
                            ->getResult(0);
                    Value ae =
                        makeOp("scalar_math", {iter[0]}, {b.getF32Type()},
                               {b.getNamedAttr("op", b.getStringAttr("fabs"))})
                            ->getResult(0);
                    Value gt = b.create<arith::CmpFOp>(
                        loc, arith::CmpFPredicate::OGT, ax, ae);
                    Value next =
                        b.create<arith::SelectOp>(loc, gt, xk, iter[0]);
                    b.create<scf::YieldOp>(loc, ValueRange{next});
                  });
              Value e = maxLp.getResult(0);
              Value isZ = b.create<arith::CmpFOp>(
                  loc, arith::CmpFPredicate::OEQ, e, cF32(0.0f));
              storeAt(zeroAll, gq,
                      b.create<arith::SelectOp>(loc, isZ, cI32(1), cI32(0)));
              ifThenElse(
                  isZ, {},
                  [&]() { storeAt(dsAll, gq, cF32(0.0f)); },
                  [&]() {
                    Value maxQ = cF32(-float((1 << (q8.elementBits - 1)) - 1));
                    Value inv = b.create<arith::DivFOp>(loc, maxQ, e);
                    Value ds = b.create<arith::DivFOp>(loc, cF32(1.0f), inv);
                    storeAt(dsAll, gq, ds);
                    Value gBase = b.create<arith::MulIOp>(loc, gq, cI64(256));
                    forLoop(cI64(0), cI64(256), cI64(1), [&](Value k) {
                      Value xk = loadAt(actTmp, k, b.getF32Type());
                      Value f = b.create<arith::MulFOp>(loc, xk, inv);
                      Value fl =
                          makeOp(
                              "scalar_math", {f}, {b.getF32Type()},
                              {b.getNamedAttr("op", b.getStringAttr("floor"))})
                              ->getResult(0);
                      Value lo =
                          b.create<arith::FPToSIOp>(loc, b.getI32Type(), fl);
                      Value frac = b.create<arith::SubFOp>(loc, f, fl);
                      Value gtHalf = b.create<arith::CmpFOp>(
                          loc, arith::CmpFPredicate::OGT, frac, cF32(0.5f));
                      Value eqHalf = b.create<arith::CmpFOp>(
                          loc, arith::CmpFPredicate::OEQ, frac, cF32(0.5f));
                      Value odd = b.create<arith::CmpIOp>(
                          loc, arith::CmpIPredicate::ne,
                          b.create<arith::AndIOp>(loc, lo, cI32(1)), cI32(0));
                      Value bump = b.create<arith::OrIOp>(
                          loc, gtHalf,
                          b.create<arith::AndIOp>(loc, eqHalf, odd));
                      Value inc = b.create<arith::SelectOp>(loc, bump, cI32(1),
                                                            cI32(0));
                      Value rnd = b.create<arith::AddIOp>(loc, lo, inc);
                      Value cl = b.create<arith::MinSIOp>(
                          loc, b.create<arith::MaxSIOp>(loc, rnd, cI32(qMin)),
                          cI32(qMax));
                      storeAt(qActAllF32, b.create<arith::AddIOp>(loc, gBase, k),
                              b.create<arith::SIToFPOp>(loc, b.getF32Type(), cl));
                    });
                  });
            });
            sync("all");
          }
        }
      }
    }

    int64_t mb =
        std::max<int64_t>(1, llvm::divideCeil(uint64_t(shapeM), uint64_t(bm)));
    int64_t nb =
        std::max<int64_t>(1, llvm::divideCeil(uint64_t(shapeN), uint64_t(bn)));

    LogicalResult status = success();
    forLoop(symTaskId, symGrid, symTaskDim, [&](Value job) {
      Value c = b.create<arith::MulIOp>(
          loc, b.create<arith::RemSIOp>(loc, job, cI64(nb)), cI64(bn));
      Value rInit = b.create<arith::MulIOp>(
          loc,
          b.create<arith::RemSIOp>(
              loc, b.create<arith::DivSIOp>(loc, job, cI64(nb)), cI64(mb)),
          cI64(bm));
      Value gVal = b.create<arith::DivSIOp>(loc, job, cI64(mb * nb));

      auto emitJobWork = [&](Value r, Value endM) {
        Value nr = b.create<arith::MinSIOp>(
            loc, b.create<arith::SubIOp>(loc, endM, r), cI64(bm));
        Value nc = b.create<arith::MinSIOp>(
            loc, b.create<arith::SubIOp>(loc, cI64(shapeN), c), cI64(bn));

        Value batchVal, chunkVal, headVal, groupVal, lastVal;
        if (isWindow) {
          int64_t hDim = std::max<int64_t>(1, dstShape[2]);
          int64_t cDim = std::max<int64_t>(1, dstShape[1]);
          headVal = b.create<arith::RemSIOp>(loc, gVal, cI64(hDim));
          chunkVal = b.create<arith::RemSIOp>(
              loc, b.create<arith::DivSIOp>(loc, gVal, cI64(hDim)), cI64(cDim));
          batchVal = b.create<arith::DivSIOp>(loc, gVal, cI64(hDim * cDim));
          groupVal = b.create<arith::DivSIOp>(loc, headVal, cI64(headGroup));
          if (shapeK > 0) {
            Value prefView = p.dot.getOperand(3);
            lastVal = loadAt(
                globalPtr(prefView,
                          {batchVal, headVal, chunkVal, cI64(shapeK - 1)}),
                cI64(0), b.getF32Type());
          } else {
            lastVal = cF32(0.0f);
          }
        }

        Value strictFlag = cI1(false);
        if (mode == "q4k-q8k") {
          if (failed(emitQ4KRecords(acc, r, nr)))
            status = failure();
        } else if (mode == "scaled-f32") {
          if (failed(emitScaledMX8(acc, r, c, nr, nc)))
            status = failure();
        } else if (mode == "quant-format") {
          if (failed(emitQuantizedFromDescriptor(
                  acc, r, c, nr, nc, quantLut, quantRecLut, qActAllF32, dsAll,
                  zeroAll)))
            status = failure();
        } else {
          fillVec(acc, cI64(bm * bn),
                  mode == "exact-i8-conv" ? cI32(0) : cF32(0.0f));
          auto &feeds = p.fold.getRegion(0).front().front().getRegion(0).front();
          StringRef rhsTransfer = feeds.front()
                                      .getNextNode()
                                      ->getAttrOfType<StringAttr>("bangc.transfer")
                                      .getValue();
          auto kLoop = forLoopCarry(
              cI64(0), cI64(shapeK), cI64(bk), ValueRange{cI1(false)},
              [&](Value kk, ValueRange iter) {
                Value nk = b.create<arith::MinSIOp>(
                    loc, b.create<arith::SubIOp>(loc, cI64(shapeK), kk),
                    cI64(bk));
                if (failed(emitOperandStage(
                        true, left, right, trans, mapped, stepbuf, prefixbuf,
                        scales, expScratch, r, c, gVal, kk, nr, nc, nk,
                        batchVal, chunkVal, headVal, groupVal, lastVal)) ||
                    failed(emitOperandStage(
                        false, left, right, trans, mapped, stepbuf, prefixbuf,
                        scales, expScratch, r, c, gVal, kk, nr, nc, nk,
                        batchVal, chunkVal, headVal, groupVal, lastVal))) {
                  status = failure();
                  b.create<scf::YieldOp>(loc, ValueRange{iter[0]});
                  return;
                }
                if (mode == "wram-conv" || mode == "exact-i8-conv") {
                  if (rhsTransfer != "filter-major-dma" &&
                      rhsTransfer != "mapped-filter-major-dma") {
                    makeOp("tile_transpose",
                           {trans, right, cI64(bk), cI64(bn)}, {}, {},
                           "nram-transpose");
                  }
                  auto runWramConv = [&](bool guardedWindow) {
                    sync("compute");
                    memcpy1D(filter, trans, cI64(bk * bn), "NRAM2WRAM");
                    if (mode == "exact-i8-conv") {
                      makeOp("wram_conv", {acc, partial, left, filter}, {},
                             {b.getNamedAttr("bm", b.getI64IntegerAttr(bm)),
                              b.getNamedAttr("bn", b.getI64IntegerAttr(bn)),
                              b.getNamedAttr("bk", b.getI64IntegerAttr(bk)),
                              b.getNamedAttr("mode", b.getStringAttr("i8"))},
                             "wram-conv-i8");
                    } else {
                      Value accum = b.create<arith::CmpIOp>(
                          loc, arith::CmpIPredicate::ne, kk, cI64(0));
                      StringRef mKind = inElem.isBF16() ? "bf16" : "f16";
                      StringRef cKey =
                          inElem.isBF16() ? "wram-conv-bf16" : "wram-conv-f16";
                      SmallVector<NamedAttribute, 5> convAttrs{
                          b.getNamedAttr("bm", b.getI64IntegerAttr(bm)),
                          b.getNamedAttr("bn", b.getI64IntegerAttr(bn)),
                          b.getNamedAttr("bk", b.getI64IntegerAttr(bk)),
                          b.getNamedAttr("mode", b.getStringAttr(mKind))};
                      if (guardedWindow && inElem.isBF16())
                        convAttrs.push_back(b.getNamedAttr(
                            "bangc.guarded_bf16_window", b.getBoolAttr(true)));
                      makeOp("wram_conv", {acc, left, filter, accum}, {},
                             convAttrs, cKey);
                      sync("compute");
                    }
                  };
                  if (hasBf16Guard) {
                    sync("compute");
                    Value okL =
                        makeOp("bf16_panel_ok",
                               {left, right, flags, cI64(bm * bk)},
                               {b.getI1Type()}, {}, "bf16-panel-guard")
                            ->getResult(0);
                    Value okR =
                        makeOp("bf16_panel_ok",
                               {trans, right, flags, cI64(bn * bk)},
                               {b.getI1Type()}, {}, "bf16-panel-guard")
                            ->getResult(0);
                    Value bothOk = b.create<arith::AndIOp>(loc, okL, okR);
                    Value nextStrict = b.create<arith::OrIOp>(
                        loc, iter[0],
                        b.create<arith::XOrIOp>(loc, bothOk, cI1(true)));
                    ifThenElse(
                        nextStrict, {},
                        [&]() {
                          makeOp("bf16_ieee_dot",
                                 {acc, left, trans, nr, nc, nk, cI64(bk),
                                  cI64(bn)},
                                 {}, {}, "bf16-ieee-dot");
                        },
                        [&]() { runWramConv(/*guardedWindow=*/true); });
                    b.create<scf::YieldOp>(loc, ValueRange{nextStrict});
                    return;
                  }
                  runWramConv(/*guardedWindow=*/false);
                } else {
                  forLoop(cI64(0), nr, cI64(1), [&](Value i) {
                    forLoop(cI64(0), nk, cI64(1), [&](Value t) {
                      Value lv = loadAt(
                          left,
                          b.create<arith::AddIOp>(
                              loc, b.create<arith::MulIOp>(loc, i, cI64(bk)),
                              t),
                          b.getF32Type());
                      forLoop(cI64(0), nc, cI64(1), [&](Value j) {
                        Value rv = loadAt(
                            right,
                            b.create<arith::AddIOp>(
                                loc, b.create<arith::MulIOp>(loc, t, cI64(bn)),
                                j),
                            b.getF32Type());
                        Value aIdx = b.create<arith::AddIOp>(
                            loc, b.create<arith::MulIOp>(loc, i, cI64(bn)), j);
                        Value cur = loadAt(acc, aIdx);
                        storeAt(acc, aIdx,
                                b.create<arith::AddFOp>(
                                    loc, cur,
                                    b.create<arith::MulFOp>(loc, lv, rv)));
                      });
                    });
                  });
                }
                b.create<scf::YieldOp>(loc, ValueRange{iter[0]});
              });
          strictFlag = kLoop.getResult(0);
        }

        // Epilogue and writeback.
        Value accOut = acc;
        auto narrowAcc = [&]() {
          if (outElem.isBF16()) {
            Value bfAcc = makeOp("nram_ptr", {acc, cI64(0)},
                                 {memSpacePtr(b.getContext(), b.getBF16Type(), 1)})
                              ->getResult(0);
            makeOp("bf16_acc_narrow",
                   {bfAcc, acc, cI64(bm * bn),
                    hasBf16Guard ? strictFlag : cI1(false)},
                   {}, {}, "bf16-acc-narrow");
            accOut = bfAcc;
          } else if (outElem.isF16()) {
            Value f16Acc =
                makeOp("nram_ptr", {acc, cI64(0)},
                       {memSpacePtr(b.getContext(), b.getF16Type(), 1)})
                    ->getResult(0);
            makeOp("vec_cast", {f16Acc, acc, cI64(bm * bn)}, {},
                   {b.getNamedAttr("kind", b.getStringAttr("f32_to_f16"))});
            accOut = f16Acc;
          }
        };

        if (writeback == "routed-row-dma") {
          int64_t q = dstShape[1];
          narrowAcc();
          sync("compute");
          forLoop(cI64(0), nr, cI64(1), [&](Value i) {
            Value ri = b.create<arith::AddIOp>(loc, r, i);
            Value dest = toI64(
                loadAt(globalPtr(routedIds, {ri}), cI64(0), b.getI32Type()));
            Value d0 = b.create<arith::DivSIOp>(loc, dest, cI64(q));
            Value d1 = b.create<arith::RemSIOp>(loc, dest, cI64(q));
            Value outPtr = globalPtr(dst, {d0, d1, c});
            Value srcRow = makeAddr(accOut, {i}, {cI64(bn)}, 1);
            memcpy1D(outPtr, srcRow, nc, "NRAM2GDRAM");
          });
        } else if (writeback == "pitched-dma") {
          for (auto &ep : epilogue) {
            if (ep.kind == "add_columns") {
              if (p.stride(ep.bias, 0) == 1) {
                memcpy1D(biasBuf, globalPtr(ep.bias, {c}), nc, "GDRAM2NRAM");
                sync("io");
              } else {
                forLoop(cI64(0), nc, cI64(1), [&](Value j) {
                  Value cj = b.create<arith::AddIOp>(loc, c, j);
                  storeAt(biasBuf, j,
                          loadAt(globalPtr(ep.bias, {cj}), cI64(0)));
                });
              }
              forLoop(cI64(0), nr, cI64(1), [&](Value i) {
                Value aRow = makeAddr(acc, {i}, {cI64(bn)}, 1);
                makeOp("vec_binary", {aRow, aRow, biasBuf, cI64(bn)}, {},
                       {b.getNamedAttr("op", b.getStringAttr("add"))});
              });
            }
          }
          narrowAcc();
          sync("compute");
          if (p.rank(dst) == 1) {
            memcpy1D(globalPtr(dst, {r}), accOut, nr, "NRAM2GDRAM");
          } else {
            Value outPtr =
                p.rank(dst) == 5
                    ? globalPtr(dst, {batchVal, chunkVal, headVal, r, c})
                : p.rank(dst) == 2 ? globalPtr(dst, {r, c})
                                   : globalPtr(dst, {gVal, r, c});
            ifThenElse(
                b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, nr,
                                        cI64(1)),
                {}, [&]() { memcpy1D(outPtr, accOut, nc, "NRAM2GDRAM"); },
                [&]() {
                  memcpy2D(outPtr, accOut, nc,
                           cI64(p.stride(dst, p.rank(dst) - 2)), cI64(bn),
                           b.create<arith::SubIOp>(loc, nr, cI64(1)),
                           "NRAM2GDRAM");
                });
          }
        } else {
          sync("compute");
          forLoop(cI64(0), nr, cI64(1), [&](Value i) {
            Value ri = b.create<arith::AddIOp>(loc, r, i);
            Value dest =
                isRouted ? toI64(loadAt(globalPtr(routedIds, {ri}), cI64(0),
                                        b.getI32Type()))
                         : ri;
            forLoop(cI64(0), nc, cI64(1), [&](Value j) {
              Value cj = b.create<arith::AddIOp>(loc, c, j);
              Value aIdx = p.rank(dst) == 1
                               ? i
                               : b.create<arith::AddIOp>(
                                     loc, b.create<arith::MulIOp>(loc, i, cI64(bn)),
                                     j);
              Value val = loadAt(acc, aIdx);
              for (auto &ep : epilogue) {
                if (ep.kind == "add_columns") {
                  Value bCoord = p.rank(dst) == 1 ? dest : cj;
                  Value bv = loadAt(globalPtr(ep.bias, {bCoord}), cI64(0),
                                    ep.toTy);
                  val = ep.toTy.isInteger(32)
                            ? b.create<arith::AddIOp>(loc, val, bv).getResult()
                            : b.create<arith::AddFOp>(loc, val, bv).getResult();
                } else if (ep.kind == "convert_matrix") {
                  if (ep.toTy.isBF16() && ep.fromTy.isInteger(32)) {
                    val = makeOp("i32_to_bf16", {val}, {b.getBF16Type()}, {},
                                 "i32-to-bf16")
                              ->getResult(0);
                  } else if (ep.toTy.isBF16() && ep.fromTy.isF32()) {
                    val = makeOp("bf16_scalar", {val}, {b.getBF16Type()},
                                 {b.getNamedAttr("mode",
                                                 b.getStringAttr("narrow"))},
                                 "bf16-scalar-convert")
                              ->getResult(0);
                  } else if (ep.toTy.isF16() && ep.fromTy.isF32()) {
                    val = b.create<arith::TruncFOp>(loc, b.getF16Type(), val);
                  } else if (ep.toTy.isF32() && ep.fromTy.isF16()) {
                    val = b.create<arith::ExtFOp>(loc, b.getF32Type(), val);
                  } else if (ep.toTy.isF32() && ep.fromTy.isInteger(32)) {
                    val = b.create<arith::SIToFPOp>(loc, b.getF32Type(), val);
                  }
                }
              }
              Value outPtr;
              if (p.rank(dst) == 1)
                outPtr = globalPtr(dst, {dest});
              else if (p.rank(dst) == 2)
                outPtr = globalPtr(dst, {dest, cj});
              else if (isRouted) {
                int64_t q = dstShape[1];
                outPtr = globalPtr(
                    dst, {b.create<arith::DivSIOp>(loc, dest, cI64(q)),
                          b.create<arith::RemSIOp>(loc, dest, cI64(q)), cj});
              } else if (p.rank(dst) == 5)
                outPtr =
                    globalPtr(dst, {batchVal, chunkVal, headVal, dest, cj});
              else
                outPtr = globalPtr(dst, {gVal, dest, cj});
              storeAt(outPtr, cI64(0), val);
            });
          });
        }
      };

      if (isRagged) {
        Value gStart = toI64(
            loadAt(globalPtr(raggedOffsets, {gVal}), cI64(0), b.getI32Type()));
        Value gEnd = toI64(loadAt(
            globalPtr(raggedOffsets,
                      {b.create<arith::AddIOp>(loc, gVal, cI64(1))}),
            cI64(0), b.getI32Type()));
        Value r = b.create<arith::AddIOp>(loc, rInit, gStart);
        ifThen(
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, r, gEnd),
            [&]() { emitJobWork(r, gEnd); });
      } else {
        emitJobWork(rInit, cI64(shapeM));
      }
    });

    if (failed(status))
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

    uint64_t grid = llvm::divideCeil(uint64_t(shapeM), uint64_t(bm)) *
                    llvm::divideCeil(uint64_t(shapeN), uint64_t(bn)) *
                    uint64_t(shapeB);
    auto abiJson =
        llvm::json::parse(fn->getAttrOfType<StringAttr>("kir.abi").getValue());
    if (!abiJson)
      return llvm::consumeError(abiJson.takeError()), failure();

    p.domain.erase();

    if (!tailGuards.empty()) {
      compilerinfra::IntegerRangeContext tailRanges(fn.getOperation());
      for (auto &tg : tailGuards) {
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
        {"grid", llvm::json::Array{int64_t(grid), 1, 1}},
        {"guard", llvm::json::Object{{"min_width", 0}, {"max_width", nullptr}}},
        {"rows",
         llvm::json::Array{llvm::json::Object{{"id", 0},
                                              {"rows", bm},
                                              {"width", bn},
                                              {"pitch", bk},
                                              {"bytes", scratchBytes},
                                              {"tasks", tasks}}}}});
    m->setAttr("bangc_src.package",
               b.getStringAttr(StringRef(metadata.data(), metadata.size())));
    return success();
  }
};
} // namespace

LogicalResult lowerMatrixToCallable(ModuleOp m, const TargetProfile &profile) {
  auto *ctx = m.getContext();
  ctx->getOrLoadDialect<intentdsl::bangc_src::BangcSrcDialect>();
  ctx->getOrLoadDialect<scf::SCFDialect>();
  ctx->getOrLoadDialect<arith::ArithDialect>();
  auto fn = *m.getOps<func::FuncOp>().begin();
  if (failed(bindMatrix(fn, 2, true, profile)))
    return failure();
  Builder b(ctx);
  fn->setAttr("bangc.entry", b.getStringAttr("intent_enqueue"));
  MatrixLowerer lowerer(fn, profile);
  if (failed(lowerer.run(m)))
    return failure();
  stampScfCarryIds(m);
  return success();
}
} // namespace CI_UNIT_NAMESPACE
