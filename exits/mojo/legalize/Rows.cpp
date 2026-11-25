#include "compilerinfra/Assembly.h"
#include "compilerinfra/Proofs.h"
#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "intentdsl/levels/algorithm/kir/ir/Literal.h"
#include "intentdsl/levels/execution/cpu/ir/Task.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/StringExtras.h"
#include <cmath>
#include <functional>
#include <map>
#include <set>

namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
using namespace mlir;

bool canNarrowMojoAddress(Operation &op);
TerminalPrinter mojoMatrixPrinter();

void lowerMojoScfIfs(Operation *rootOp) {
  SmallVector<scf::IfOp, 16> ifs;
  rootOp->walk<WalkOrder::PostOrder>(
      [&](scf::IfOp ifOp) { ifs.push_back(ifOp); });
  for (scf::IfOp ifOp : ifs) {
    OpBuilder b(ifOp);
    Location loc = ifOp.getLoc();
    SmallVector<Value, 4> slots;
    for (Value r : ifOp.getResults()) {
      std::string tyStr;
      if (auto sh = dyn_cast<ShapedType>(r.getType())) {
        std::string dt = "DType.float32";
        Type et = sh.getElementType();
        if (et.isBF16())
          dt = "DType.bfloat16";
        else if (et.isF16())
          dt = "DType.float16";
        else if (et.isInteger(1))
          dt = "DType.bool";
        else if (et.isUnsignedInteger(64))
          dt = "DType.uint64";
        else if (et.isInteger(64))
          dt = "DType.int64";
        else if (et.isInteger(32))
          dt = "DType.int32";
        tyStr = "SIMD[" + dt + "," + std::to_string(sh.getDimSize(0)) + "]";
      } else {
        Type t = r.getType();
        tyStr = t.isIndex()               ? "Int"
                : t.isF16()               ? "Float16"
                : t.isBF16()              ? "BFloat16"
                : t.isF32()               ? "Float32"
                : t.isInteger(1)          ? "Bool"
                : t.isUnsignedInteger(64) ? "UInt64"
                : t.isInteger(64)         ? "Int64"
                                          : "Int32";
      }
      OperationState declSt(loc, "mojo_row.var_decl");
      declSt.addAttribute("mojo_type", b.getStringAttr(tyStr));
      declSt.addTypes(r.getType());
      Value slot = b.create(declSt)->getResult(0);
      r.replaceAllUsesWith(slot);
      slots.push_back(slot);
    }

    bool hasElse = !ifOp.getElseRegion().empty();
    OperationState newSt(loc, "mojo_row.if");
    newSt.addOperands(ifOp.getCondition());
    for (NamedAttribute attr : ifOp->getDiscardableAttrDictionary())
      newSt.addAttribute(attr.getName(), attr.getValue());
    newSt.addAttribute("has_else", b.getBoolAttr(hasElse));
    newSt.addRegion();
    newSt.addRegion();
    Operation *newIf = b.create(newSt);

    auto convertRegion = [&](Region &srcReg, Region &dstReg) {
      dstReg.takeBody(srcReg);
      Block &bb = dstReg.front();
      if (auto yOp = dyn_cast<scf::YieldOp>(bb.getTerminator())) {
        OpBuilder yb(yOp);
        for (auto [slot, val] : llvm::zip(slots, yOp.getOperands())) {
          OperationState asgSt(loc, "mojo_row.var_assign");
          asgSt.addOperands({slot, val});
          yb.create(asgSt);
        }
        yOp.erase();
      }
      OpBuilder eb = OpBuilder::atBlockEnd(&bb);
      if (bb.empty())
        eb.create(OperationState(loc, "mojo_row.pass"));
      eb.create(OperationState(loc, "mojo_row.end"));
    };

    convertRegion(ifOp.getThenRegion(), newIf->getRegion(0));
    if (hasElse)
      convertRegion(ifOp.getElseRegion(), newIf->getRegion(1));
    ifOp.erase();
  }
}

namespace {
std::string dtypeBare(Type t) {
  if (auto sh = dyn_cast<ShapedType>(t))
    t = sh.getElementType();
  if (t.isInteger(1))
    return "bool";
  if (t.isUnsignedInteger(64))
    return "uint64";
  if (t.isInteger(32))
    return "int32";
  if (t.isSignlessInteger(64) || t.isInteger(64))
    return "int64";
  return t.isBF16() ? "bfloat16" : t.isF16() ? "float16" : "float32";
}

std::string scalarMojoSpec(Type t) {
  if (t.isIndex())
    return "Int";
  if (t.isUnsignedInteger(64))
    return "UInt64";
  if (t.isInteger(64))
    return "Int64";
  if (t.isInteger(32))
    return "Int32";
  if (t.isInteger(1))
    return "Bool";
  if (t.isF16())
    return "Float16";
  if (t.isBF16())
    return "BFloat16";
  return "Float32";
}

Operation *makeOp(OpBuilder &b, Location loc, StringRef name,
                  ValueRange operands = {}, TypeRange types = {},
                  ArrayRef<NamedAttribute> attrs = {},
                  unsigned numRegions = 0) {
  OperationState s(loc, ("mojo_row." + name).str());
  s.addOperands(operands);
  s.addTypes(types);
  s.addAttributes(attrs);
  for (unsigned i = 0; i < numRegions; ++i)
    s.addRegion();
  return b.create(s);
}

Value constIdx(OpBuilder &b, Location loc, int64_t v) {
  return b.create<arith::ConstantIndexOp>(loc, v);
}

Value constI32(OpBuilder &b, Location loc, int32_t v) {
  return b.create<arith::ConstantOp>(loc, b.getI32Type(),
                                     b.getI32IntegerAttr(v));
}

Value constF32(OpBuilder &b, Location loc, float v) {
  return b.create<arith::ConstantOp>(loc, b.getF32Type(), b.getF32FloatAttr(v));
}

Value asInt(OpBuilder &b, Location loc, Value v) {
  if (v.getType().isIndex())
    return v;
  return b.create<arith::IndexCastOp>(loc, b.getIndexType(), v);
}

Value makeStaticAddr(OpBuilder &b, Location loc, Value ptr, ValueRange coords,
                     ArrayRef<int64_t> strides) {
  SmallVector<Value> ops{ptr};
  ops.append(coords.begin(), coords.end());
  return makeOp(b, loc, "address", ops, {b.getIndexType()},
                {b.getNamedAttr("strides", b.getDenseI64ArrayAttr(strides)),
                 b.getNamedAttr("index_bits", b.getI32IntegerAttr(64))})
      ->getResult(0);
}

Value makeDynAddr(OpBuilder &b, Location loc, Value ptr, ValueRange coords,
                  ValueRange strides) {
  SmallVector<Value> ops{ptr};
  ops.append(coords.begin(), coords.end());
  ops.append(strides.begin(), strides.end());
  return makeOp(b, loc, "dyn_address", ops, {b.getIndexType()},
                {b.getNamedAttr("terms",
                                b.getI64IntegerAttr(int64_t(coords.size()))),
                 b.getNamedAttr("index_bits", b.getI32IntegerAttr(64))})
      ->getResult(0);
}

struct ArgSlotInfo {
  bool isView = false;
  Type elemTy;
  unsigned rank = 0;
  Value ptrVal;
  DenseI64ArrayAttr boundShape;
  DenseI64ArrayAttr boundStrides;
  SmallVector<Value, 4> dimVals;    // length rank-1 if rank > 2, else 0
  SmallVector<Value, 4> strideVals; // length rank
};

struct RowLowerer {
  func::FuncOp fn;
  Operation &task;
  OpBuilder &b;
  Location loc;
  int64_t fullWidth = 1;
  int64_t simdWidth = 1;

  SmallVector<ArgSlotInfo> slots;
  Value firstIdx, stopIdx, colsIdx, colBlockIdx, keyLoIdx, keyHiIdx;
  bool projected = false;
  unsigned projectedRank = 0;
  std::optional<int64_t> ownerAxis;
  SmallVector<SmallVector<Value, 4>> hoistedOwnerSizes;
  SmallVector<Value, 4> rowCacheBufs;
  SmallVector<Type, 4> rowCacheElemTypes;
  bool forceUnitStride = false;
  Block *sweepHoistBlock = nullptr;
  Operation *sweepHoistBefore = nullptr;
  std::map<std::pair<void *, int64_t>, Value> sweepSplatCache;

  unsigned slotIndex(Value taskCapture) {
    return cast<BlockArgument>(taskCapture).getArgNumber() - 1;
  }

  Value emitViewAddr(unsigned idx, ArrayRef<Value> coords) {
    auto &info = slots[idx];
    if (info.boundStrides && info.boundStrides.size() == coords.size())
      return makeStaticAddr(b, loc, info.ptrVal, coords,
                            info.boundStrides.asArrayRef());
    if (forceUnitStride && coords.size() == size_t(info.rank)) {
      if (info.rank == 1)
        return makeStaticAddr(b, loc, info.ptrVal, coords, {1});
      SmallVector<Value, 4> st(info.strideVals.begin(),
                               info.strideVals.begin() + coords.size());
      st.back() = constIdx(b, loc, 1);
      return makeDynAddr(b, loc, info.ptrVal, coords, st);
    }
    SmallVector<Value, 4> st(info.strideVals.begin(),
                             info.strideVals.begin() + coords.size());
    return makeDynAddr(b, loc, info.ptrVal, coords, st);
  }

  Value declScalarSlot(Type ty, Value initVal = {}) {
    SmallVector<Value> inits;
    if (initVal)
      inits.push_back(initVal);
    return makeOp(b, loc, "scalar_decl", inits, {ty},
                  {b.getNamedAttr("scalar_type",
                                  b.getStringAttr(scalarMojoSpec(ty)))})
        ->getResult(0);
  }

  Value readScalarSlot(Value slot) {
    return makeOp(b, loc, "acc_read", {slot}, {slot.getType()})->getResult(0);
  }

  void writeScalarSlot(Value slot, Value val) {
    makeOp(b, loc, "acc_write", {slot, val});
  }

  Value declVecSlot(Type vecTy, int64_t w, Value initVal = {}) {
    SmallVector<Value> inits;
    if (initVal)
      inits.push_back(initVal);
    return makeOp(b, loc, "acc_decl", inits, {vecTy},
                  {b.getNamedAttr("dtype", b.getStringAttr(dtypeBare(vecTy))),
                   b.getNamedAttr("width", b.getI64IntegerAttr(w))})
        ->getResult(0);
  }

  Value readVecSlot(Value slot) {
    return makeOp(b, loc, "acc_read", {slot}, {slot.getType()})->getResult(0);
  }

  void writeVecSlot(Value slot, Value val) {
    makeOp(b, loc, "acc_write", {slot, val});
  }

  void precomputeOwnerSizes() {
    if (!ownerAxis)
      return;
    int64_t owner = *ownerAxis;
    hoistedOwnerSizes.resize(slots.size());
    for (size_t idx = 0; idx < slots.size(); ++idx) {
      auto &info = slots[idx];
      if (!info.isView)
        continue;
      int64_t R = info.rank;
      hoistedOwnerSizes[idx].resize(R);
      if (info.boundShape) {
        for (int64_t d = 0; d < R; ++d)
          hoistedOwnerSizes[idx][d] =
              constIdx(b, loc, std::max<int64_t>(1, info.boundShape[d]));
        continue;
      }
      if (R == 2) {
        for (int64_t d = 0; d < R; ++d)
          if (d != owner)
            hoistedOwnerSizes[idx][d] = colsIdx;
        continue;
      }
      Value last = colsIdx;
      for (int64_t d = 0; d + 1 < R; ++d) {
        if (d != owner) {
          hoistedOwnerSizes[idx][d] = info.dimVals[d];
          last = b.create<arith::DivSIOp>(loc, last, info.dimVals[d]);
        }
      }
      if (R - 1 != owner)
        hoistedOwnerSizes[idx][R - 1] = last;
    }
  }

  SmallVector<Value, 4> buildOwnerCoords(unsigned idx, Value r,
                                         Value linearIdx) {
    int64_t R = slots[idx].rank;
    int64_t owner = *ownerAxis;
    int64_t firstNonOwner = -1;
    for (int64_t d = 0; d < R; ++d) {
      if (d != owner) {
        firstNonOwner = d;
        break;
      }
    }
    SmallVector<Value, 4> coords(R);
    coords[owner] = r;
    Value cur = linearIdx;
    for (int64_t d = R - 1; d >= 0; --d) {
      if (d == owner)
        continue;
      if (slots[idx].boundShape) {
        int64_t szVal = std::max<int64_t>(1, slots[idx].boundShape[d]);
        if (szVal == 1) {
          coords[d] = constIdx(b, loc, 0);
          continue;
        }
        if (d == firstNonOwner) {
          coords[d] = cur;
          continue;
        }
        if (llvm::isPowerOf2_64(uint64_t(szVal))) {
          coords[d] =
              b.create<arith::AndIOp>(loc, cur, constIdx(b, loc, szVal - 1));
          cur = b.create<arith::ShRUIOp>(
              loc, cur, constIdx(b, loc, int64_t(llvm::Log2_64(uint64_t(szVal)))));
          continue;
        }
      }
      if (d == firstNonOwner && !slots[idx].boundShape) {
        Value sz = hoistedOwnerSizes[idx][d];
        coords[d] = b.create<arith::RemSIOp>(loc, cur, sz);
        continue;
      }
      Value sz = hoistedOwnerSizes[idx][d];
      coords[d] = b.create<arith::RemSIOp>(loc, cur, sz);
      cur = b.create<arith::DivSIOp>(loc, cur, sz);
    }
    return coords;
  }

  SmallVector<Value, 4> buildStandardCoords(unsigned idx, Value r, Value j,
                                            bool swap) {
    int64_t R = slots[idx].rank;
    if (R == 1)
      return {j};
    if (R == 2)
      return swap ? SmallVector<Value, 4>{j, r} : SmallVector<Value, 4>{r, j};
    SmallVector<Value, 4> coords(R);
    coords[R - 1] = j;
    if (projected && unsigned(R) == projectedRank) {
      for (int64_t d = 0; d < R - 2; ++d)
        coords[d] = constIdx(b, loc, 0);
      coords[R - 2] = b.create<arith::SubIOp>(loc, r, firstIdx);
      return coords;
    }
    Value row = r;
    for (int64_t d = R - 2; d >= 0; --d) {
      Value sz = slots[idx].dimVals[d];
      coords[d] = b.create<arith::RemSIOp>(loc, row, sz);
      row = b.create<arith::DivSIOp>(loc, row, sz);
    }
    return coords;
  }

  void emitTransposeTile() {
    auto &info0 = slots[0];
    auto &info1 = slots[1];
    Value s0_0 = info0.strideVals[0], s0_1 = info0.strideVals[1];
    Value s1_0 = info1.strideVals[0], s1_1 = info1.strideVals[1];
    Value c32 = constIdx(b, loc, 32), c1 = constIdx(b, loc, 1),
          c0 = constIdx(b, loc, 0), c8 = constIdx(b, loc, 8);

    Value fullR =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, stopIdx, c32);
    Value fullC =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, colsIdx, c32);
    Value unitSrc =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, s0_1, c1);
    Value unitDst =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, s1_1, c1);
    Value cond = b.create<arith::AndIOp>(
        loc, b.create<arith::AndIOp>(loc, fullR, fullC),
        b.create<arith::AndIOp>(loc, unitSrc, unitDst));

    auto branch = b.create<scf::IfOp>(loc, TypeRange{}, cond, true);
    {
      OpBuilder::InsertionGuard tg(b);
      b.setInsertionPointToStart(&branch.getThenRegion().front());
      auto *brLoop = makeOp(b, loc, "for_range", {c0, c32, c8}, {}, {}, 1);
      Block &brBb = brLoop->getRegion(0).emplaceBlock();
      Value br = brBb.addArgument(b.getIndexType(), loc);
      b.setInsertionPointToEnd(&brBb);
      auto *bcLoop = makeOp(b, loc, "for_range", {c0, c32, c8}, {}, {}, 1);
      Block &bcBb = bcLoop->getRegion(0).emplaceBlock();
      Value bc = bcBb.addArgument(b.getIndexType(), loc);
      b.setInsertionPointToEnd(&bcBb);
      Value srcOff = emitViewAddr(0, {br, bc});
      Value dstOff = emitViewAddr(1, {bc, br});
      makeOp(b, loc, "transpose_8x8",
             {info0.ptrVal, srcOff, s0_0, info1.ptrVal, dstOff, s1_0}, {},
             {b.getNamedAttr("dtype", b.getStringAttr(dtypeBare(info0.elemTy))),
              b.getNamedAttr("ci.implementation",
                             b.getStringAttr("exits/mojo::simd-transpose-8x8"))});
      makeOp(b, loc, "end");
      b.setInsertionPointToEnd(&brBb);
      makeOp(b, loc, "end");
    }
    {
      OpBuilder::InsertionGuard eg(b);
      b.setInsertionPointToStart(&branch.getElseRegion().front());
      auto *rLoop =
          makeOp(b, loc, "for_range", {firstIdx, stopIdx, c1}, {}, {}, 1);
      Block &rBb = rLoop->getRegion(0).emplaceBlock();
      Value r = rBb.addArgument(b.getIndexType(), loc);
      b.setInsertionPointToEnd(&rBb);
      auto *jLoop = makeOp(b, loc, "for_range", {c0, colsIdx, c1}, {}, {}, 1);
      Block &jBb = jLoop->getRegion(0).emplaceBlock();
      Value j = jBb.addArgument(b.getIndexType(), loc);
      b.setInsertionPointToEnd(&jBb);
      Value srcAddr = emitViewAddr(0, {r, j});
      Value val =
          makeOp(b, loc, "load_scalar", {info0.ptrVal, srcAddr}, {info0.elemTy})
              ->getResult(0);
      Value dstAddr = emitViewAddr(1, {j, r});
      makeOp(b, loc, "store_scalar", {info1.ptrVal, dstAddr, val});
      makeOp(b, loc, "end");
      b.setInsertionPointToEnd(&rBb);
      makeOp(b, loc, "end");
    }
  }

  LogicalResult emitReservations(Block &body) {
    int64_t bins = task.getAttrOfType<IntegerAttr>("cpu.batch_extent").getInt();
    Value c0 = constIdx(b, loc, 0), c1 = constIdx(b, loc, 1),
          cBins = constIdx(b, loc, bins);
    Value i32Zero = constI32(b, loc, 0), i32One = constI32(b, loc, 1);
    Value privBuf;
    int distOffSlot = -1;
    for (Operation &step : body.without_terminator())
      if (step.getName().stripDialect() == "distribute_tickets")
        distOffSlot = int(slotIndex(step.getOperand(1)));

    for (Operation &step : body.without_terminator()) {
      auto n = step.getName().stripDialect();
      if (n == "tally_keys") {
        unsigned kSlot = slotIndex(step.getOperand(0));
        privBuf =
            makeOp(b, loc, "stack_alloc", {}, {b.getIndexType()},
                   {b.getNamedAttr("dtype", b.getStringAttr("int32")),
                    b.getNamedAttr("count", b.getI64IntegerAttr(bins))})
                ->getResult(0);
        {
          auto *zLoop =
              makeOp(b, loc, "for_range", {c0, cBins, c1}, {}, {}, 1);
          Block &zBb = zLoop->getRegion(0).emplaceBlock();
          Value k = zBb.addArgument(b.getIndexType(), loc);
          OpBuilder::InsertionGuard zg(b);
          b.setInsertionPointToEnd(&zBb);
          Value zAddr = makeStaticAddr(b, loc, privBuf, {k}, {1});
          makeOp(b, loc, "store_scalar", {privBuf, zAddr, i32Zero});
          makeOp(b, loc, "end");
        }
        auto emitTallyBody = [&](bool unitK) {
          bool prevFast = forceUnitStride;
          forceUnitStride = unitK;
          auto *rLoop =
              makeOp(b, loc, "for_range", {firstIdx, stopIdx, c1}, {}, {}, 1);
          Block &rBb = rLoop->getRegion(0).emplaceBlock();
          Value r = rBb.addArgument(b.getIndexType(), loc);
          OpBuilder::InsertionGuard rg(b);
          b.setInsertionPointToEnd(&rBb);
          auto *jLoop =
              makeOp(b, loc, "for_range", {c0, colsIdx, c1}, {}, {}, 1);
          Block &jBb = jLoop->getRegion(0).emplaceBlock();
          Value j = jBb.addArgument(b.getIndexType(), loc);
          b.setInsertionPointToEnd(&jBb);
          Value kAddr = emitViewAddr(kSlot, {r, j});
          Value kRaw = makeOp(b, loc, "load_scalar",
                              {slots[kSlot].ptrVal, kAddr},
                              {slots[kSlot].elemTy})
                           ->getResult(0);
          Value kIdx = asInt(b, loc, kRaw);
          Value pAddr = makeStaticAddr(b, loc, privBuf, {kIdx}, {1});
          Value cur =
              makeOp(b, loc, "load_scalar", {privBuf, pAddr}, {b.getI32Type()})
                  ->getResult(0);
          makeOp(b, loc, "store_scalar",
                 {privBuf, pAddr, b.create<arith::AddIOp>(loc, cur, i32One)});
          makeOp(b, loc, "end");
          b.setInsertionPointToEnd(&rBb);
          makeOp(b, loc, "end");
          forceUnitStride = prevFast;
        };
        if (!slots[kSlot].boundStrides && slots[kSlot].rank >= 1) {
          Value eq1 = b.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::eq,
              slots[kSlot].strideVals[slots[kSlot].rank - 1], c1);
          auto ifUnit = b.create<scf::IfOp>(loc, TypeRange{}, eq1, true);
          {
            OpBuilder::InsertionGuard tg(b);
            b.setInsertionPointToStart(&ifUnit.getThenRegion().front());
            emitTallyBody(true);
          }
          {
            OpBuilder::InsertionGuard eg(b);
            b.setInsertionPointToStart(&ifUnit.getElseRegion().front());
            emitTallyBody(false);
          }
        } else {
          emitTallyBody(false);
        }
      } else if (n == "reserve_tickets") {
        unsigned cntSlot = slotIndex(step.getOperand(0));
        Value threadShift = b.create<arith::MulIOp>(
            loc, b.create<arith::ShRSIOp>(loc, firstIdx, constIdx(b, loc, 5)),
            constIdx(b, loc, 16));
        auto *kLoop = makeOp(b, loc, "for_range", {c0, cBins, c1}, {}, {}, 1);
        Block &kBb = kLoop->getRegion(0).emplaceBlock();
        Value kIter = kBb.addArgument(b.getIndexType(), loc);
        OpBuilder::InsertionGuard kg(b);
        b.setInsertionPointToEnd(&kBb);
        Value k = b.create<arith::RemSIOp>(
            loc, b.create<arith::AddIOp>(loc, kIter, threadShift), cBins);
        Value pAddr = makeStaticAddr(b, loc, privBuf, {k}, {1});
        Value count =
            makeOp(b, loc, "load_scalar", {privBuf, pAddr}, {b.getI32Type()})
                ->getResult(0);
        Value nonZero = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                                count, i32Zero);
        auto ifOp = b.create<scf::IfOp>(loc, TypeRange{}, nonZero, false);
        b.setInsertionPointToStart(&ifOp.getThenRegion().front());
        Value gAddr = emitViewAddr(cntSlot, {k});
        Value base =
            makeOp(b, loc, "atomic_fetch_add_i32",
                   {slots[cntSlot].ptrVal, gAddr, count}, {b.getI32Type()})
                ->getResult(0);
        if (distOffSlot >= 0) {
          Value oAddr = emitViewAddr(unsigned(distOffSlot), {k});
          Value oVal =
              makeOp(b, loc, "load_scalar",
                     {slots[distOffSlot].ptrVal, oAddr}, {b.getI32Type()})
                  ->getResult(0);
          base = b.create<arith::AddIOp>(loc, base, oVal);
        }
        makeOp(b, loc, "store_scalar", {privBuf, pAddr, base});
        b.setInsertionPointToEnd(&kBb);
        makeOp(b, loc, "end");
      } else if (n == "distribute_tickets") {
        unsigned kSlot = slotIndex(step.getOperand(0));
        unsigned outSlot = slotIndex(step.getOperand(3));
        auto emitDistBody = [&](bool unitFast) {
          bool prevFast = forceUnitStride;
          forceUnitStride = unitFast;
          auto *rLoop =
              makeOp(b, loc, "for_range", {firstIdx, stopIdx, c1}, {}, {}, 1);
          Block &rBb = rLoop->getRegion(0).emplaceBlock();
          Value r = rBb.addArgument(b.getIndexType(), loc);
          OpBuilder::InsertionGuard rg(b);
          b.setInsertionPointToEnd(&rBb);
          auto *jLoop =
              makeOp(b, loc, "for_range", {c0, colsIdx, c1}, {}, {}, 1);
          Block &jBb = jLoop->getRegion(0).emplaceBlock();
          Value j = jBb.addArgument(b.getIndexType(), loc);
          b.setInsertionPointToEnd(&jBb);
          Value kAddr = emitViewAddr(kSlot, {r, j});
          Value kRaw = makeOp(b, loc, "load_scalar",
                              {slots[kSlot].ptrVal, kAddr}, {slots[kSlot].elemTy})
                           ->getResult(0);
          Value kIdx = asInt(b, loc, kRaw);
          Value pAddr = makeStaticAddr(b, loc, privBuf, {kIdx}, {1});
          Value slotI32 =
              makeOp(b, loc, "load_scalar", {privBuf, pAddr}, {b.getI32Type()})
                  ->getResult(0);
          Value slotIdx = asInt(b, loc, slotI32);
          Value dstAddr = emitViewAddr(outSlot, {slotIdx});
          Value ordIdx = b.create<arith::AddIOp>(
              loc, b.create<arith::MulIOp>(loc, r, colsIdx), j);
          Value ordI32 =
              makeOp(b, loc, "scalar_cast", {ordIdx}, {b.getI32Type()},
                     {b.getNamedAttr("dtype", b.getStringAttr("int32"))})
                  ->getResult(0);
          makeOp(b, loc, "store_scalar",
                 {slots[outSlot].ptrVal, dstAddr, ordI32});
          makeOp(b, loc, "store_scalar",
                 {privBuf, pAddr, b.create<arith::AddIOp>(loc, slotI32, i32One)});
          makeOp(b, loc, "end");
          b.setInsertionPointToEnd(&rBb);
          makeOp(b, loc, "end");
          forceUnitStride = prevFast;
        };
        if (!slots[kSlot].boundStrides && !slots[outSlot].boundStrides &&
            slots[kSlot].rank >= 1 && slots[outSlot].rank >= 1) {
          Value eqK = b.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::eq,
              slots[kSlot].strideVals[slots[kSlot].rank - 1], c1);
          Value eqO = b.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::eq,
              slots[outSlot].strideVals[slots[outSlot].rank - 1], c1);
          auto ifUnit = b.create<scf::IfOp>(
              loc, TypeRange{}, b.create<arith::AndIOp>(loc, eqK, eqO), true);
          {
            OpBuilder::InsertionGuard tg(b);
            b.setInsertionPointToStart(&ifUnit.getThenRegion().front());
            emitDistBody(true);
          }
          {
            OpBuilder::InsertionGuard eg(b);
            b.setInsertionPointToStart(&ifUnit.getElseRegion().front());
            emitDistBody(false);
          }
        } else {
          emitDistBody(false);
        }
      } else {
        return step.emitError("unexpected reservation stage ") << n;
      }
    }
    return success();
  }

  void emitPackMembers(Operation &packOp, Value r) {
    unsigned srcSlot = slotIndex(packOp.getOperand(0));
    unsigned dstSlot = slotIndex(packOp.getOperand(1));
    unsigned cntSlot = slotIndex(packOp.getOperand(2));
    Value c0 = constIdx(b, loc, 0), c1 = constIdx(b, loc, 1);
    Value countSlot = declScalarSlot(b.getIndexType(), c0);
    Value thresh =
        makeOp(b, loc, "scalar_const", {}, {slots[srcSlot].elemTy},
               {b.getNamedAttr("dtype",
                               b.getStringAttr(dtypeBare(slots[srcSlot].elemTy))),
                b.getNamedAttr(
                    "literal",
                    b.getStringAttr(intentdsl::kir::literal(
                        packOp.getAttrOfType<TypedAttr>("threshold"))))})
            ->getResult(0);
    int32_t fillInt = packOp.getAttrOfType<IntegerAttr>("fill").getInt();
    Value fillVal = constI32(b, loc, fillInt);

    {
      auto *jLoop = makeOp(b, loc, "for_range", {c0, colsIdx, c1}, {}, {}, 1);
      Block &jBb = jLoop->getRegion(0).emplaceBlock();
      Value j = jBb.addArgument(b.getIndexType(), loc);
      OpBuilder::InsertionGuard jg(b);
      b.setInsertionPointToEnd(&jBb);
      Value inAddr = emitViewAddr(srcSlot, {r, j});
      Value inVal = makeOp(b, loc, "load_scalar",
                           {slots[srcSlot].ptrVal, inAddr},
                           {slots[srcSlot].elemTy})
                        ->getResult(0);
      Value keep = isa<FloatType>(inVal.getType())
                       ? b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNE,
                                                 inVal, thresh)
                             .getResult()
                       : b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                                 inVal, thresh)
                             .getResult();
      auto ifKeep = b.create<scf::IfOp>(loc, TypeRange{}, keep, false);
      b.setInsertionPointToStart(&ifKeep.getThenRegion().front());
      Value curCnt = readScalarSlot(countSlot);
      Value outAddr = emitViewAddr(dstSlot, {r, curCnt});
      Value jI32 = makeOp(b, loc, "scalar_cast", {j}, {b.getI32Type()},
                          {b.getNamedAttr("dtype", b.getStringAttr("int32"))})
                       ->getResult(0);
      makeOp(b, loc, "store_scalar", {slots[dstSlot].ptrVal, outAddr, jI32});
      writeScalarSlot(countSlot, b.create<arith::AddIOp>(loc, curCnt, c1));
      b.setInsertionPointToEnd(&jBb);
      makeOp(b, loc, "end");
    }
    Value totalCnt = readScalarSlot(countSlot);
    {
      auto *fLoop =
          makeOp(b, loc, "for_range", {totalCnt, colsIdx, c1}, {}, {}, 1);
      Block &fBb = fLoop->getRegion(0).emplaceBlock();
      Value j = fBb.addArgument(b.getIndexType(), loc);
      OpBuilder::InsertionGuard fg(b);
      b.setInsertionPointToEnd(&fBb);
      Value fAddr = emitViewAddr(dstSlot, {r, j});
      makeOp(b, loc, "store_scalar", {slots[dstSlot].ptrVal, fAddr, fillVal});
      makeOp(b, loc, "end");
    }
    Value cntAddr = emitViewAddr(cntSlot, {r});
    Value cntI32 =
        makeOp(b, loc, "scalar_cast", {totalCnt}, {b.getI32Type()},
               {b.getNamedAttr("dtype", b.getStringAttr("int32"))})
            ->getResult(0);
    makeOp(b, loc, "store_scalar", {slots[cntSlot].ptrVal, cntAddr, cntI32});
  }

  LogicalResult emitAtomicMembers(Operation &atomOp, Value r) {
    auto &innerBb = atomOp.getRegion(0).front();
    Value c0 = constIdx(b, loc, 0), c1 = constIdx(b, loc, 1);
    llvm::DenseMap<Value, Value> carries;
    for (Operation &v : innerBb.without_terminator()) {
      if (v.getName().stripDialect() == "scan_sum") {
        Type elemTy = cast<ShapedType>(v.getResult(0).getType()).getElementType();
        Value zero =
            makeOp(b, loc, "scalar_const", {}, {elemTy},
                   {b.getNamedAttr("dtype", b.getStringAttr(dtypeBare(elemTy))),
                    b.getNamedAttr("literal", b.getStringAttr("0"))})
                ->getResult(0);
        carries[v.getResult(0)] = declScalarSlot(elemTy, zero);
      }
    }

    struct PostAction {
      enum Kind { PrefixFill, WriteZeroOffsets } kind;
      unsigned dstSlot = 0;
      unsigned cntSlot = 0;
      Value carrySlot;
      int32_t fill = 0;
    };
    SmallVector<PostAction, 2> postActions;
    bool recordPost = true;
    auto emitMemberLoop = [&](bool unitStride) -> LogicalResult {
      bool savedForce = forceUnitStride;
      forceUnitStride = unitStride;
      auto *jLoop = makeOp(b, loc, "for_range", {c0, colsIdx, c1}, {}, {}, 1);
      Block &jBb = jLoop->getRegion(0).emplaceBlock();
      Value j = jBb.addArgument(b.getIndexType(), loc);
      OpBuilder::InsertionGuard jg(b);
      b.setInsertionPointToEnd(&jBb);
      IRMapping vals;
      for (Operation &v : innerBb.without_terminator()) {
        auto n = v.getName().stripDialect();
        if (n == "read") {
          unsigned sIdx = slotIndex(v.getOperand(0));
          Value addr = slots[sIdx].rank == 1 ? emitViewAddr(sIdx, {j})
                                             : emitViewAddr(sIdx, {r, j});
          Value loaded = makeOp(b, loc, "load_scalar",
                                {slots[sIdx].ptrVal, addr},
                                {slots[sIdx].elemTy})
                             ->getResult(0);
          vals.map(v.getResult(0), loaded);
        } else if (n == "fill_ranges") {
          unsigned dstIdx = slotIndex(v.getOperand(0));
          unsigned offIdx = slotIndex(v.getOperand(1));
          int64_t unit = v.getAttrOfType<IntegerAttr>("unit").getInt();
          Value jNext = b.create<arith::AddIOp>(loc, j, c1);
          Value hasNext = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                                  jNext, colsIdx);
          auto ifNext = b.create<scf::IfOp>(loc, TypeRange{}, hasNext, false);
          {
            OpBuilder::InsertionGuard ng(b);
            b.setInsertionPointToStart(&ifNext.getThenRegion().front());
            Value a0 = emitViewAddr(offIdx, {j});
            Value a1 = emitViewAddr(offIdx, {jNext});
            Value loRaw = makeOp(b, loc, "load_scalar",
                                 {slots[offIdx].ptrVal, a0},
                                 {slots[offIdx].elemTy})
                              ->getResult(0);
            Value hiRaw = makeOp(b, loc, "load_scalar",
                                 {slots[offIdx].ptrVal, a1},
                                 {slots[offIdx].elemTy})
                              ->getResult(0);
            Value uIdx = constIdx(b, loc, unit);
            Value bLo =
                b.create<arith::DivSIOp>(loc, asInt(b, loc, loRaw), uIdx);
            Value bHi =
                b.create<arith::DivSIOp>(loc, asInt(b, loc, hiRaw), uIdx);
            auto *blkLoop =
                makeOp(b, loc, "for_range", {bLo, bHi, c1}, {}, {}, 1);
            Block &blkBb = blkLoop->getRegion(0).emplaceBlock();
            Value blk = blkBb.addArgument(b.getIndexType(), loc);
            b.setInsertionPointToEnd(&blkBb);
            Value dAddr = emitViewAddr(dstIdx, {blk});
            Value jI32 =
                makeOp(b, loc, "scalar_cast", {j}, {b.getI32Type()},
                       {b.getNamedAttr("dtype", b.getStringAttr("int32"))})
                    ->getResult(0);
            makeOp(b, loc, "store_scalar", {slots[dstIdx].ptrVal, dAddr, jI32});
            makeOp(b, loc, "end");
          }
        } else if (n == "constant") {
          Type resTy = v.getResult(0).getType();
          if (auto sh = dyn_cast<ShapedType>(resTy))
            resTy = sh.getElementType();
          Value cst =
              makeOp(b, loc, "scalar_const", {}, {resTy},
                     {b.getNamedAttr("dtype", b.getStringAttr(dtypeBare(resTy))),
                      b.getNamedAttr(
                          "literal",
                          b.getStringAttr(intentdsl::kir::literal(
                              v.getAttrOfType<TypedAttr>("value"))))})
                  ->getResult(0);
          vals.map(v.getResult(0), cst);
        } else if (n == "ne") {
          Value lhs = vals.lookup(v.getOperand(0)),
                rhs = vals.lookup(v.getOperand(1));
          Value res = isa<FloatType>(lhs.getType())
                          ? b.create<arith::CmpFOp>(
                                 loc, arith::CmpFPredicate::UNE, lhs, rhs)
                                .getResult()
                          : b.create<arith::CmpIOp>(
                                 loc, arith::CmpIPredicate::ne, lhs, rhs)
                                .getResult();
          vals.map(v.getResult(0), res);
        } else if (n == "cast") {
          Type dstTy = v.getResult(0).getType();
          if (auto sh = dyn_cast<ShapedType>(dstTy))
            dstTy = sh.getElementType();
          Value res =
              makeOp(b, loc, "scalar_cast", {vals.lookup(v.getOperand(0))},
                     {dstTy},
                     {b.getNamedAttr("dtype", b.getStringAttr(dtypeBare(dstTy)))})
                  ->getResult(0);
          vals.map(v.getResult(0), res);
        } else if (n == "member_index") {
          Value res =
              makeOp(b, loc, "scalar_cast", {j}, {b.getI32Type()},
                     {b.getNamedAttr("dtype", b.getStringAttr("int32"))})
                  ->getResult(0);
          vals.map(v.getResult(0), res);
        } else if (n == "ordinal") {
          Value ordIdx = b.create<arith::AddIOp>(
              loc, b.create<arith::MulIOp>(loc, r, colsIdx), j);
          Value res =
              makeOp(b, loc, "scalar_cast", {ordIdx}, {b.getI32Type()},
                     {b.getNamedAttr("dtype", b.getStringAttr("int32"))})
                  ->getResult(0);
          vals.map(v.getResult(0), res);
        } else if (n == "shr" || n == "shl" || n == "add") {
          Value lhs = vals.lookup(v.getOperand(0)),
                rhs = vals.lookup(v.getOperand(1));
          Value res =
              n == "add"
                  ? (isa<FloatType>(lhs.getType())
                         ? b.create<arith::AddFOp>(loc, lhs, rhs).getResult()
                         : b.create<arith::AddIOp>(loc, lhs, rhs).getResult())
                  : (n == "shl"
                         ? b.create<arith::ShLIOp>(loc, lhs, rhs).getResult()
                         : b.create<arith::ShRSIOp>(loc, lhs, rhs).getResult());
          vals.map(v.getResult(0), res);
        } else if (n == "scan_sum") {
          Value cSlot = carries.lookup(v.getResult(0));
          Value cur = readScalarSlot(cSlot);
          Value inc = vals.lookup(v.getOperand(0));
          Value next = isa<FloatType>(cur.getType())
                           ? b.create<arith::AddFOp>(loc, cur, inc).getResult()
                           : b.create<arith::AddIOp>(loc, cur, inc).getResult();
          writeScalarSlot(cSlot, next);
          vals.map(v.getResult(0), readScalarSlot(cSlot));
        } else if (n == "prefix_scatter") {
          unsigned dstIdx = slotIndex(v.getOperand(0));
          unsigned cntIdx = slotIndex(v.getOperand(1));
          Value pred = vals.lookup(v.getOperand(3));
          Value pref = vals.lookup(v.getOperand(4));
          Value pay = vals.lookup(v.getOperand(5));
          auto ifPred = b.create<scf::IfOp>(loc, TypeRange{}, pred, false);
          {
            OpBuilder::InsertionGuard pg(b);
            b.setInsertionPointToStart(&ifPred.getThenRegion().front());
            Value colPos =
                b.create<arith::SubIOp>(loc, asInt(b, loc, pref), c1);
            Value dAddr = emitViewAddr(dstIdx, {r, colPos});
            makeOp(b, loc, "store_scalar", {slots[dstIdx].ptrVal, dAddr, pay});
          }
          if (recordPost)
            postActions.push_back(
                {PostAction::PrefixFill, dstIdx, cntIdx,
                 carries.lookup(v.getOperand(4)),
                 int32_t(v.getAttrOfType<IntegerAttr>("fill").getInt())});
        } else if (n == "write_offsets") {
          unsigned dstIdx = slotIndex(v.getOperand(0));
          unsigned totIdx = slotIndex(v.getOperand(1));
          Value pref = vals.lookup(v.getOperand(3));
          Value dAddr =
              emitViewAddr(dstIdx, {b.create<arith::AddIOp>(loc, j, c1)});
          makeOp(b, loc, "store_scalar", {slots[dstIdx].ptrVal, dAddr, pref});
          if (recordPost)
            postActions.push_back({PostAction::WriteZeroOffsets, dstIdx, totIdx,
                                   carries.lookup(v.getOperand(3)), 0});
        } else if (n == "atomic_add") {
          unsigned dstIdx = slotIndex(v.getOperand(0));
          Value kIdx = asInt(b, loc, vals.lookup(v.getOperand(1)));
          Value delta = vals.lookup(v.getOperand(2));
          if (keyLoIdx) {
            Value geLo = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                                 kIdx, keyLoIdx);
            Value ltHi = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                                 kIdx, keyHiIdx);
            Value inRange = b.create<arith::AndIOp>(loc, geLo, ltHi);
            auto ifRange = b.create<scf::IfOp>(loc, TypeRange{}, inRange, false);
            b.setInsertionPointToStart(&ifRange.getThenRegion().front());
          }
          Value dAddr = emitViewAddr(dstIdx, {kIdx});
          Value old;
          if (task.hasAttr("cpu.single_dispatcher") || keyLoIdx) {
            old = makeOp(b, loc, "load_scalar", {slots[dstIdx].ptrVal, dAddr},
                         {b.getI32Type()})
                      ->getResult(0);
            makeOp(b, loc, "store_scalar",
                   {slots[dstIdx].ptrVal, dAddr,
                    b.create<arith::AddIOp>(loc, old, delta)});
          } else {
            old = makeOp(b, loc, "atomic_fetch_add_i32",
                         {slots[dstIdx].ptrVal, dAddr, delta}, {b.getI32Type()})
                      ->getResult(0);
          }
          vals.map(v.getResult(0), old);
        } else if (n == "index_read") {
          unsigned sIdx = slotIndex(v.getOperand(0));
          Value kIdx = asInt(b, loc, vals.lookup(v.getOperand(1)));
          Value sAddr = emitViewAddr(sIdx, {kIdx});
          Value val = makeOp(b, loc, "load_scalar",
                             {slots[sIdx].ptrVal, sAddr}, {slots[sIdx].elemTy})
                          ->getResult(0);
          vals.map(v.getResult(0), val);
        } else if (n == "unique_put") {
          unsigned dstIdx = slotIndex(v.getOperand(0));
          Value kIdx = asInt(b, loc, vals.lookup(v.getOperand(1)));
          Value dAddr = emitViewAddr(dstIdx, {kIdx});
          makeOp(b, loc, "store_scalar",
                 {slots[dstIdx].ptrVal, dAddr, vals.lookup(v.getOperand(2))});
        } else {
          forceUnitStride = savedForce;
          return v.emitError("unsupported atomic member instruction ") << n;
        }
      }
      b.setInsertionPointToEnd(&jBb);
      makeOp(b, loc, "end");
      forceUnitStride = savedForce;
      recordPost = false;
      return success();
    };

    Value allContig = b.create<arith::ConstantIntOp>(loc, 1, 1);
    for (auto &sl : slots) {
      if (!sl.strideVals.empty()) {
        Value eq1 = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                            sl.strideVals.back(), c1);
        allContig = b.create<arith::AndIOp>(loc, allContig, eq1);
      }
    }
    auto ifContig = b.create<scf::IfOp>(loc, TypeRange{}, allContig, true);
    {
      OpBuilder::InsertionGuard ig(b);
      b.setInsertionPointToStart(&ifContig.getThenRegion().front());
      if (failed(emitMemberLoop(true)))
        return failure();
      b.setInsertionPointToStart(&ifContig.getElseRegion().front());
      if (failed(emitMemberLoop(false)))
        return failure();
    }

    for (auto &act : postActions) {
      Value finalCarry = readScalarSlot(act.carrySlot);
      if (act.kind == PostAction::PrefixFill) {
        Value startFill = asInt(b, loc, finalCarry);
        Value fillVal = constI32(b, loc, act.fill);
        auto *fLoop =
            makeOp(b, loc, "for_range", {startFill, colsIdx, c1}, {}, {}, 1);
        Block &fBb = fLoop->getRegion(0).emplaceBlock();
        Value j = fBb.addArgument(b.getIndexType(), loc);
        {
          OpBuilder::InsertionGuard fg(b);
          b.setInsertionPointToEnd(&fBb);
          Value fAddr = emitViewAddr(act.dstSlot, {r, j});
          makeOp(b, loc, "store_scalar",
                 {slots[act.dstSlot].ptrVal, fAddr, fillVal});
          makeOp(b, loc, "end");
        }
        Value cntAddr = emitViewAddr(act.cntSlot, {r});
        makeOp(b, loc, "store_scalar",
               {slots[act.cntSlot].ptrVal, cntAddr, finalCarry});
      } else {
        Value z0Addr = emitViewAddr(act.dstSlot, {c0});
        makeOp(b, loc, "store_scalar",
               {slots[act.dstSlot].ptrVal, z0Addr, constI32(b, loc, 0)});
        Value totAddr = emitViewAddr(act.cntSlot, {c0});
        makeOp(b, loc, "store_scalar",
               {slots[act.cntSlot].ptrVal, totAddr, finalCarry});
      }
    }
    return success();
  }

  Value splatIfScalar(Value v, Type targetTy, int64_t w) {
    if (w == 1 || isa<ShapedType>(v.getType()) || !isa<ShapedType>(targetTy))
      return v;
    auto key = std::make_pair(v.getAsOpaquePointer(), w);
    if (sweepHoistBlock) {
      Block *defBlk = v.getParentBlock();
      bool dominates =
          defBlk && (defBlk == sweepHoistBlock ||
                     defBlk->getParentOp()->isProperAncestor(
                         sweepHoistBlock->getParentOp()));
      if (dominates) {
        auto it = sweepSplatCache.find(key);
        if (it != sweepSplatCache.end())
          return it->second;
        OpBuilder::InsertionGuard hg(b);
        if (sweepHoistBefore)
          b.setInsertionPoint(sweepHoistBefore);
        else
          b.setInsertionPointToEnd(sweepHoistBlock);
        auto vecTy = VectorType::get({w}, v.getType());
        Value res =
            makeOp(b, loc, "splat_scalar", {v}, {vecTy},
                   {b.getNamedAttr("dtype",
                                   b.getStringAttr(dtypeBare(v.getType()))),
                    b.getNamedAttr("width", b.getI64IntegerAttr(w))})
                ->getResult(0);
        sweepSplatCache[key] = res;
        return res;
      }
    }
    auto vecTy = VectorType::get({w}, v.getType());
    return makeOp(b, loc, "splat_scalar", {v}, {vecTy},
                  {b.getNamedAttr("dtype", b.getStringAttr(dtypeBare(v.getType()))),
                   b.getNamedAttr("width", b.getI64IntegerAttr(w))})
        ->getResult(0);
  }

  bool proveOwnerFastSpan(unsigned idx, int64_t fast, int64_t w, bool isWrite) {
    if (!slots[idx].boundShape || !slots[idx].boundStrides || w <= 0)
      return false;
    int64_t R = slots[idx].rank;
    int64_t spanConst = std::max<int64_t>(1, slots[idx].boundShape[fast]);
    int64_t stepConst = slots[idx].boundStrides[fast];
    int64_t colsConst = 1;
    for (int64_t ax = 0; ax < R; ++ax)
      if (ax != *ownerAxis)
        colsConst *= std::max<int64_t>(1, slots[idx].boundShape[ax]);
    int64_t fullStopConst = (colsConst / fullWidth) * fullWidth;
    if (fullStopConst < w)
      return false;
    OpBuilder sb(b.getContext());
    Location sloc = b.getUnknownLoc();
    OperationState scopeSt(sloc, "mojo_row.row_fn");
    scopeSt.addAttribute("param_specs", sb.getArrayAttr({}));
    scopeSt.addAttribute("uses_atomic", sb.getBoolAttr(false));
    scopeSt.addAttribute("uses_stack", sb.getBoolAttr(false));
    scopeSt.addAttribute("uses_memcpy", sb.getBoolAttr(false));
    scopeSt.addAttribute("has_key_bounds", sb.getBoolAttr(false));
    Block &sBb = scopeSt.addRegion()->emplaceBlock();
    Value ptrArg = sBb.addArgument(slots[idx].ptrVal.getType(), sloc);
    auto vecTy = VectorType::get({w}, slots[idx].elemTy);
    Value valArg = isWrite ? sBb.addArgument(vecTy, sloc) : Value{};
    OwningOpRef<Operation *> scopeOp(Operation::create(scopeSt));
    sb.setInsertionPointToEnd(&sBb);
    Value z0 = constIdx(sb, sloc, 0), ub = constIdx(sb, sloc, fullStopConst),
          stW = constIdx(sb, sloc, w);
    Operation *jLoop =
        makeOp(sb, sloc, "for_range", {z0, ub, stW}, {}, {}, 1);
    makeOp(sb, sloc, "end");
    Block &jBb = jLoop->getRegion(0).emplaceBlock();
    Value jVar = jBb.addArgument(sb.getIndexType(), sloc);
    sb.setInsertionPointToEnd(&jBb);
    Value fastCoord = jVar;
    if (colsConst != spanConst) {
      if (spanConst % w == 0) {
        Value stepIdx = sb.create<arith::DivSIOp>(sloc, jVar, stW);
        Value spanSteps = constIdx(sb, sloc, spanConst / w);
        Value remSteps = sb.create<arith::RemSIOp>(sloc, stepIdx, spanSteps);
        fastCoord = sb.create<arith::MulIOp>(sloc, remSteps, stW);
      } else {
        fastCoord =
            sb.create<arith::RemSIOp>(sloc, jVar, constIdx(sb, sloc, spanConst));
      }
    }
    Value probeAddr =
        makeOp(sb, sloc, "address", {ptrArg, fastCoord}, {sb.getIndexType()},
               {sb.getNamedAttr("strides", sb.getDenseI64ArrayAttr({stepConst})),
                sb.getNamedAttr("index_bits", sb.getI32IntegerAttr(64)),
                sb.getNamedAttr("extents", sb.getDenseI64ArrayAttr({spanConst}))})
            ->getResult(0);
    Operation *probeMem =
        isWrite
            ? makeOp(sb, sloc, "store_slice", {ptrArg, probeAddr, valArg}, {},
                     {sb.getNamedAttr("slice_axis", sb.getI64IntegerAttr(0))})
            : makeOp(sb, sloc, "load_slice", {ptrArg, probeAddr}, {vecTy},
                     {sb.getNamedAttr("width", sb.getI64IntegerAttr(w)),
                      sb.getNamedAttr("slice_axis", sb.getI64IntegerAttr(0))});
    makeOp(sb, sloc, "end");
    auto tp = compilerinfra::proveTailAccess(probeMem);
    return tp.proven && tp.kind == compilerinfra::TailKind::ExactInBounds;
  }

  Value emitSweepRead(Operation &op, Value r, Value j, int64_t w) {
    unsigned idx = slotIndex(op.getOperand(0));
    bool swap = op.hasAttr("axes");
    Type elemTy = slots[idx].elemTy;
    if (w == 1) {
      auto coords = ownerAxis ? buildOwnerCoords(idx, r, j)
                              : buildStandardCoords(idx, r, j, swap);
      Value addr = emitViewAddr(idx, coords);
      return makeOp(b, loc, "load_scalar", {slots[idx].ptrVal, addr}, {elemTy})
          ->getResult(0);
    }

    auto vecTy = VectorType::get({w}, elemTy);

    if (ownerAxis) {
      int64_t R = slots[idx].rank;
      int64_t fast = R - 1 - (*ownerAxis == R - 1 ? 1 : 0);
      if (slots[idx].boundShape && slots[idx].boundStrides) {
        int64_t stepConst = slots[idx].boundStrides[fast];
        bool exactSpan = proveOwnerFastSpan(idx, fast, w, /*isWrite=*/false);
        if (exactSpan && stepConst == 1) {
          Value baseAddr = emitViewAddr(idx, buildOwnerCoords(idx, r, j));
          return makeOp(b, loc, "load_slice", {slots[idx].ptrVal, baseAddr},
                        {vecTy},
                        {b.getNamedAttr("width", b.getI64IntegerAttr(w))})
              ->getResult(0);
        }
        if (exactSpan && stepConst == 0) {
          Value baseAddr = emitViewAddr(idx, buildOwnerCoords(idx, r, j));
          Value sc = makeOp(b, loc, "load_scalar",
                            {slots[idx].ptrVal, baseAddr}, {elemTy})
                         ->getResult(0);
          return makeOp(b, loc, "splat_scalar", {sc}, {vecTy},
                        {b.getNamedAttr("dtype",
                                        b.getStringAttr(dtypeBare(elemTy))),
                         b.getNamedAttr("width", b.getI64IntegerAttr(w))})
              ->getResult(0);
        }
      }
      Value c0 = constIdx(b, loc, 0), c1 = constIdx(b, loc, 1),
            cW = constIdx(b, loc, w);
      Value resSlot = declVecSlot(vecTy, w);
      Value span = hoistedOwnerSizes[idx][fast];
      Value step = slots[idx].strideVals[fast];
      Value baseAddr = emitViewAddr(idx, buildOwnerCoords(idx, r, j));
      Value rem = b.create<arith::RemSIOp>(loc, j, span);
      Value inSpan = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sle,
          b.create<arith::AddIOp>(loc, rem, cW), span);
      Value cond1 = b.create<arith::AndIOp>(
          loc,
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, step, c1),
          inSpan);
      auto ifContig = b.create<scf::IfOp>(loc, TypeRange{}, cond1, true);
      {
        OpBuilder::InsertionGuard tg(b);
        b.setInsertionPointToStart(&ifContig.getThenRegion().front());
        Value v = makeOp(b, loc, "load_slice", {slots[idx].ptrVal, baseAddr},
                         {vecTy},
                         {b.getNamedAttr("width", b.getI64IntegerAttr(w))})
                      ->getResult(0);
        writeVecSlot(resSlot, v);
      }
      {
        OpBuilder::InsertionGuard eg(b);
        b.setInsertionPointToStart(&ifContig.getElseRegion().front());
        Value cond0 = b.create<arith::AndIOp>(
            loc,
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, step, c0),
            inSpan);
        auto ifBcast = b.create<scf::IfOp>(loc, TypeRange{}, cond0, true);
        {
          OpBuilder::InsertionGuard bg(b);
          b.setInsertionPointToStart(&ifBcast.getThenRegion().front());
          Value sc = makeOp(b, loc, "load_scalar",
                            {slots[idx].ptrVal, baseAddr}, {elemTy})
                         ->getResult(0);
          Value v = makeOp(b, loc, "splat_scalar", {sc}, {vecTy},
                           {b.getNamedAttr("dtype",
                                           b.getStringAttr(dtypeBare(elemTy))),
                            b.getNamedAttr("width", b.getI64IntegerAttr(w))})
                        ->getResult(0);
          writeVecSlot(resSlot, v);
        }
        {
          OpBuilder::InsertionGuard sg(b);
          b.setInsertionPointToStart(&ifBcast.getElseRegion().front());
          auto *lLoop = makeOp(b, loc, "for_range", {c0, cW, c1}, {}, {}, 1);
          Block &lBb = lLoop->getRegion(0).emplaceBlock();
          Value lane = lBb.addArgument(b.getIndexType(), loc);
          b.setInsertionPointToEnd(&lBb);
          Value lAddr = emitViewAddr(
              idx,
              buildOwnerCoords(idx, r, b.create<arith::AddIOp>(loc, j, lane)));
          Value elem = makeOp(b, loc, "load_scalar",
                              {slots[idx].ptrVal, lAddr}, {elemTy})
                           ->getResult(0);
          makeOp(b, loc, "acc_insert_lane", {resSlot, lane, elem});
          makeOp(b, loc, "end");
        }
      }
      return readVecSlot(resSlot);
    }

    int64_t R = slots[idx].rank;
    int64_t fastAxis = R == 2 && swap ? 0 : R - 1;
    if (forceUnitStride) {
      Value baseAddr = emitViewAddr(idx, buildStandardCoords(idx, r, j, swap));
      return makeOp(b, loc, "load_slice", {slots[idx].ptrVal, baseAddr},
                    {vecTy}, {b.getNamedAttr("width", b.getI64IntegerAttr(w))})
          ->getResult(0);
    }
    if (slots[idx].boundStrides) {
      int64_t stepConst = slots[idx].boundStrides[fastAxis];
      if (stepConst == 1) {
        Value baseAddr = emitViewAddr(idx, buildStandardCoords(idx, r, j, swap));
        return makeOp(b, loc, "load_slice", {slots[idx].ptrVal, baseAddr},
                      {vecTy},
                      {b.getNamedAttr("width", b.getI64IntegerAttr(w))})
            ->getResult(0);
      }
      if (stepConst == 0) {
        Value baseAddr = emitViewAddr(idx, buildStandardCoords(idx, r, j, swap));
        Value sc = makeOp(b, loc, "load_scalar", {slots[idx].ptrVal, baseAddr},
                          {elemTy})
                       ->getResult(0);
        return makeOp(b, loc, "splat_scalar", {sc}, {vecTy},
                      {b.getNamedAttr("dtype",
                                      b.getStringAttr(dtypeBare(elemTy))),
                       b.getNamedAttr("width", b.getI64IntegerAttr(w))})
            ->getResult(0);
      }
    }
    Value c0 = constIdx(b, loc, 0), c1 = constIdx(b, loc, 1),
          cW = constIdx(b, loc, w);
    Value resSlot = declVecSlot(vecTy, w);
    Value step = slots[idx].strideVals[fastAxis];
    Value baseAddr = emitViewAddr(idx, buildStandardCoords(idx, r, j, swap));
    Value isContig =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, step, c1);
    auto ifContig = b.create<scf::IfOp>(loc, TypeRange{}, isContig, true);
    {
      OpBuilder::InsertionGuard tg(b);
      b.setInsertionPointToStart(&ifContig.getThenRegion().front());
      Value v = makeOp(b, loc, "load_slice", {slots[idx].ptrVal, baseAddr},
                       {vecTy},
                       {b.getNamedAttr("width", b.getI64IntegerAttr(w))})
                    ->getResult(0);
      writeVecSlot(resSlot, v);
    }
    {
      OpBuilder::InsertionGuard eg(b);
      b.setInsertionPointToStart(&ifContig.getElseRegion().front());
      Value isZero =
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, step, c0);
      auto ifBcast = b.create<scf::IfOp>(loc, TypeRange{}, isZero, true);
      {
        OpBuilder::InsertionGuard bg(b);
        b.setInsertionPointToStart(&ifBcast.getThenRegion().front());
        Value sc = makeOp(b, loc, "load_scalar",
                          {slots[idx].ptrVal, baseAddr}, {elemTy})
                       ->getResult(0);
        Value v = makeOp(b, loc, "splat_scalar", {sc}, {vecTy},
                         {b.getNamedAttr("dtype",
                                         b.getStringAttr(dtypeBare(elemTy))),
                          b.getNamedAttr("width", b.getI64IntegerAttr(w))})
                      ->getResult(0);
        writeVecSlot(resSlot, v);
      }
      {
        OpBuilder::InsertionGuard sg(b);
        b.setInsertionPointToStart(&ifBcast.getElseRegion().front());
        auto *lLoop = makeOp(b, loc, "for_range", {c0, cW, c1}, {}, {}, 1);
        Block &lBb = lLoop->getRegion(0).emplaceBlock();
        Value lane = lBb.addArgument(b.getIndexType(), loc);
        b.setInsertionPointToEnd(&lBb);
        Value lAddr = makeDynAddr(b, loc, slots[idx].ptrVal, {baseAddr, lane},
                                  {c1, step});
        Value elem = makeOp(b, loc, "load_scalar",
                            {slots[idx].ptrVal, lAddr}, {elemTy})
                         ->getResult(0);
        makeOp(b, loc, "acc_insert_lane", {resSlot, lane, elem});
        makeOp(b, loc, "end");
      }
    }
    return readVecSlot(resSlot);
  }

  void emitSweepWrite(Operation &op, Value r, Value j, Value val, int64_t w) {
    unsigned idx = slotIndex(op.getOperand(0));
    bool swap = op.hasAttr("axes");
    Type elemTy = slots[idx].elemTy;
    if (w == 1) {
      auto coords = ownerAxis ? buildOwnerCoords(idx, r, j)
                              : buildStandardCoords(idx, r, j, swap);
      Value addr = emitViewAddr(idx, coords);
      makeOp(b, loc, "store_scalar", {slots[idx].ptrVal, addr, val});
      return;
    }

    if (ownerAxis) {
      int64_t R = slots[idx].rank;
      int64_t fast = R - 1 - (*ownerAxis == R - 1 ? 1 : 0);
      if (slots[idx].boundShape && slots[idx].boundStrides &&
          slots[idx].boundStrides[fast] == 1 &&
          proveOwnerFastSpan(idx, fast, w, /*isWrite=*/true)) {
        Value baseAddr = emitViewAddr(idx, buildOwnerCoords(idx, r, j));
        makeOp(b, loc, "store_slice", {slots[idx].ptrVal, baseAddr, val});
        return;
      }
      Value c0 = constIdx(b, loc, 0), c1 = constIdx(b, loc, 1),
            cW = constIdx(b, loc, w);
      Value span = hoistedOwnerSizes[idx][fast];
      Value step = slots[idx].strideVals[fast];
      Value baseAddr = emitViewAddr(idx, buildOwnerCoords(idx, r, j));
      Value rem = b.create<arith::RemSIOp>(loc, j, span);
      Value inSpan = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sle,
          b.create<arith::AddIOp>(loc, rem, cW), span);
      Value cond1 = b.create<arith::AndIOp>(
          loc,
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, step, c1),
          inSpan);
      auto ifContig = b.create<scf::IfOp>(loc, TypeRange{}, cond1, true);
      {
        OpBuilder::InsertionGuard tg(b);
        b.setInsertionPointToStart(&ifContig.getThenRegion().front());
        makeOp(b, loc, "store_slice", {slots[idx].ptrVal, baseAddr, val});
      }
      {
        OpBuilder::InsertionGuard eg(b);
        b.setInsertionPointToStart(&ifContig.getElseRegion().front());
        auto *lLoop = makeOp(b, loc, "for_range", {c0, cW, c1}, {}, {}, 1);
        Block &lBb = lLoop->getRegion(0).emplaceBlock();
        Value lane = lBb.addArgument(b.getIndexType(), loc);
        b.setInsertionPointToEnd(&lBb);
        Value elem =
            makeOp(b, loc, "extract_lane_dyn", {val, lane}, {elemTy})->getResult(0);
        Value lAddr = emitViewAddr(
            idx, buildOwnerCoords(idx, r, b.create<arith::AddIOp>(loc, j, lane)));
        makeOp(b, loc, "store_scalar", {slots[idx].ptrVal, lAddr, elem});
        makeOp(b, loc, "end");
      }
      return;
    }

    int64_t R = slots[idx].rank;
    int64_t fastAxis = R == 2 && swap ? 0 : R - 1;
    if (forceUnitStride ||
        (slots[idx].boundStrides && slots[idx].boundStrides[fastAxis] == 1)) {
      Value baseAddr = emitViewAddr(idx, buildStandardCoords(idx, r, j, swap));
      makeOp(b, loc, "store_slice", {slots[idx].ptrVal, baseAddr, val});
      return;
    }
    Value c0 = constIdx(b, loc, 0), c1 = constIdx(b, loc, 1),
          cW = constIdx(b, loc, w);
    Value step = slots[idx].strideVals[fastAxis];
    Value baseAddr = emitViewAddr(idx, buildStandardCoords(idx, r, j, swap));
    Value isContig =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, step, c1);
    auto ifContig = b.create<scf::IfOp>(loc, TypeRange{}, isContig, true);
    {
      OpBuilder::InsertionGuard tg(b);
      b.setInsertionPointToStart(&ifContig.getThenRegion().front());
      makeOp(b, loc, "store_slice", {slots[idx].ptrVal, baseAddr, val});
    }
    {
      OpBuilder::InsertionGuard eg(b);
      b.setInsertionPointToStart(&ifContig.getElseRegion().front());
      auto *lLoop = makeOp(b, loc, "for_range", {c0, cW, c1}, {}, {}, 1);
      Block &lBb = lLoop->getRegion(0).emplaceBlock();
      Value lane = lBb.addArgument(b.getIndexType(), loc);
      b.setInsertionPointToEnd(&lBb);
      Value elem =
          makeOp(b, loc, "extract_lane_dyn", {val, lane}, {elemTy})->getResult(0);
      Value lAddr = makeDynAddr(b, loc, slots[idx].ptrVal, {baseAddr, lane},
                                {c1, step});
      makeOp(b, loc, "store_scalar", {slots[idx].ptrVal, lAddr, elem});
      makeOp(b, loc, "end");
    }
  }

  LogicalResult
  lowerSweepBody(Block &bb, Value r, Value j, int64_t w, IRMapping &map,
                 llvm::DenseMap<int64_t, Value> &scanCarries,
                 SmallVectorImpl<Value> &yieldedVals,
                 SmallVectorImpl<std::function<void()>> *deferredStores = nullptr,
                 bool deferViewWrites = false) {
    Value c0 = constIdx(b, loc, 0), c1 = constIdx(b, loc, 1),
          cW = constIdx(b, loc, w);
    int64_t domArg = task.getAttrOfType<IntegerAttr>("domain_arg").getInt();
    bool flatDomain = cast<MemRefType>(fn.getArgument(domArg).getType()).getRank() == 1;
    bool memberTasks = task.hasAttr("cpu.member_tasks");
    llvm::DenseMap<Value, Value> ordinalScalarBases;
    llvm::DenseSet<unsigned> writtenInBb;
    for (Operation &wop : bb) {
      auto wc = wop.getName().stripDialect();
      if (wc == "write" || wc == "owner_write")
        writtenInBb.insert(slotIndex(wop.getOperand(0)));
    }
    std::map<std::pair<unsigned, bool>, Value> readCache;

    for (Operation &op : bb) {
      auto code = op.getName().stripDialect();
      if (code == "yield") {
        for (Value yv : op.getOperands())
          yieldedVals.push_back(map.lookup(yv));
        return success();
      }
      if (op.getNumResults() == 1 && map.contains(op.getResult(0)))
        continue;
      if (code == "row_cache_load") {
        unsigned cSlot = op.getAttrOfType<IntegerAttr>("slot").getInt();
        Value buf = rowCacheBufs[cSlot];
        Type elemTy = rowCacheElemTypes[cSlot];
        Value addr = makeStaticAddr(b, loc, buf, {j}, {1});
        Value res =
            w == 1
                ? makeOp(b, loc, "load_scalar", {buf, addr}, {elemTy})
                      ->getResult(0)
                : makeOp(b, loc, "load_slice", {buf, addr},
                         {VectorType::get({w}, elemTy)},
                         {b.getNamedAttr("width", b.getI64IntegerAttr(w))})
                      ->getResult(0);
        map.map(op.getResult(0), res);
        continue;
      }
      if (code == "row_cache_store") {
        unsigned cSlot = op.getAttrOfType<IntegerAttr>("slot").getInt();
        Value buf = rowCacheBufs[cSlot];
        Value val = map.lookup(op.getOperand(1));
        if (deferredStores) {
          deferredStores->push_back([=, this]() {
            Value addr = makeStaticAddr(b, loc, buf, {j}, {1});
            if (w == 1)
              makeOp(b, loc, "store_scalar", {buf, addr, val});
            else
              makeOp(b, loc, "store_slice", {buf, addr, val});
          });
        } else {
          Value addr = makeStaticAddr(b, loc, buf, {j}, {1});
          if (w == 1)
            makeOp(b, loc, "store_scalar", {buf, addr, val});
          else
            makeOp(b, loc, "store_slice", {buf, addr, val});
        }
        continue;
      }
      if (code == "sum_columns") {
        unsigned sIdx = slotIndex(op.getOperand(0));
        int64_t ext = op.getAttrOfType<IntegerAttr>("extent").getInt();
        Type resTy = w == 1 ? Type(b.getI32Type())
                            : Type(VectorType::get({w}, b.getI32Type()));
        Value acc = w == 1 ? declScalarSlot(b.getI32Type(), constI32(b, loc, 0))
                           : declVecSlot(resTy, w);
        auto *pLoop = makeOp(
            b, loc, "for_range", {c0, constIdx(b, loc, ext), c1}, {}, {}, 1);
        Block &pBb = pLoop->getRegion(0).emplaceBlock();
        Value p = pBb.addArgument(b.getIndexType(), loc);
        {
          OpBuilder::InsertionGuard pg(b);
          b.setInsertionPointToEnd(&pBb);
          if (w == 1) {
            Value addr = emitViewAddr(sIdx, {p, j});
            Value val = makeOp(b, loc, "load_scalar",
                               {slots[sIdx].ptrVal, addr}, {b.getI32Type()})
                            ->getResult(0);
            writeScalarSlot(acc,
                            b.create<arith::AddIOp>(loc, readScalarSlot(acc), val));
          } else {
            Value s1 = slots[sIdx].strideVals[1];
            Value isContig =
                b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, s1, c1);
            auto ifContig = b.create<scf::IfOp>(loc, TypeRange{}, isContig, true);
            {
              OpBuilder::InsertionGuard tg(b);
              b.setInsertionPointToStart(&ifContig.getThenRegion().front());
              Value addr = emitViewAddr(sIdx, {p, j});
              Value v =
                  makeOp(b, loc, "load_slice", {slots[sIdx].ptrVal, addr},
                         {resTy},
                         {b.getNamedAttr("width", b.getI64IntegerAttr(w))})
                      ->getResult(0);
              writeVecSlot(acc,
                           b.create<arith::AddIOp>(loc, readVecSlot(acc), v));
            }
            {
              OpBuilder::InsertionGuard eg(b);
              b.setInsertionPointToStart(&ifContig.getElseRegion().front());
              auto *kLoop = makeOp(b, loc, "for_range", {c0, cW, c1}, {}, {}, 1);
              Block &kBb = kLoop->getRegion(0).emplaceBlock();
              Value k = kBb.addArgument(b.getIndexType(), loc);
              b.setInsertionPointToEnd(&kBb);
              Value addr =
                  emitViewAddr(sIdx, {p, b.create<arith::AddIOp>(loc, j, k)});
              Value elem = makeOp(b, loc, "load_scalar",
                                  {slots[sIdx].ptrVal, addr}, {b.getI32Type()})
                               ->getResult(0);
              Value prev = makeOp(b, loc, "extract_lane_dyn",
                                  {readVecSlot(acc), k}, {b.getI32Type()})
                               ->getResult(0);
              makeOp(b, loc, "acc_insert_lane",
                     {acc, k, b.create<arith::AddIOp>(loc, prev, elem)});
              makeOp(b, loc, "end");
            }
          }
          makeOp(b, loc, "end");
        }
        map.map(op.getResult(0), w == 1 ? readScalarSlot(acc) : readVecSlot(acc));
        continue;
      }
      if (code == "segment_fold" || code == "segment_size") {
        if (map.contains(op.getResult(0)))
          continue;
        unsigned offIdx = slotIndex(op.getOperand(0));
        auto loadOff = [&](Value idxVal) {
          Value a = emitViewAddr(offIdx, {idxVal});
          Value raw = makeOp(b, loc, "load_scalar",
                             {slots[offIdx].ptrVal, a}, {b.getI32Type()})
                          ->getResult(0);
          return asInt(b, loc, raw);
        };
        Type resTy = w == 1 ? Type(b.getF32Type())
                            : Type(VectorType::get({w}, b.getF32Type()));
        if (op.hasAttr("feature_vector")) {
          auto &terms = op.getRegion(0).front();
          Operation *ldOp = terms.back().getOperand(0).getDefiningOp();
          if (ldOp->getName().stripDialect() == "cast")
            ldOp = ldOp->getOperand(0).getDefiningOp();
          unsigned fIdx = slotIndex(ldOp->getOperand(0));
          Type fElem = slots[fIdx].elemTy;
          Value start = loadOff(r);
          Value end = loadOff(b.create<arith::AddIOp>(loc, r, c1));
          Value acc = w == 1 ? declScalarSlot(b.getF32Type(), constF32(b, loc, 0))
                             : declVecSlot(resTy, w);
          auto emitContigSegLoop = [&]() {
            auto *mLoop =
                makeOp(b, loc, "for_range", {start, end, c1}, {}, {}, 1);
            Block &mBb = mLoop->getRegion(0).emplaceBlock();
            Value member = mBb.addArgument(b.getIndexType(), loc);
            OpBuilder::InsertionGuard mg(b);
            b.setInsertionPointToEnd(&mBb);
            Value fAddr = emitViewAddr(fIdx, {member, j});
            auto rawVecTy = VectorType::get({w}, fElem);
            Value raw =
                makeOp(b, loc, "load_slice", {slots[fIdx].ptrVal, fAddr},
                       {rawVecTy},
                       {b.getNamedAttr("width", b.getI64IntegerAttr(w))})
                    ->getResult(0);
            Value fVec = fElem.isF32()
                             ? raw
                             : b.create<arith::ExtFOp>(loc, resTy, raw)
                                   .getResult();
            writeVecSlot(acc, b.create<arith::AddFOp>(loc, readVecSlot(acc), fVec));
            makeOp(b, loc, "end");
          };
          if (w == 1) {
            auto *mLoop =
                makeOp(b, loc, "for_range", {start, end, c1}, {}, {}, 1);
            Block &mBb = mLoop->getRegion(0).emplaceBlock();
            Value member = mBb.addArgument(b.getIndexType(), loc);
            OpBuilder::InsertionGuard mg(b);
            b.setInsertionPointToEnd(&mBb);
            Value fAddr = emitViewAddr(fIdx, {member, j});
            Value raw = makeOp(b, loc, "load_scalar",
                               {slots[fIdx].ptrVal, fAddr}, {fElem})
                            ->getResult(0);
            Value fVal = fElem.isF32()
                             ? raw
                             : b.create<arith::ExtFOp>(loc, b.getF32Type(), raw)
                                   .getResult();
            writeScalarSlot(
                acc, b.create<arith::AddFOp>(loc, readScalarSlot(acc), fVal));
            makeOp(b, loc, "end");
          } else if (forceUnitStride ||
                     (slots[fIdx].boundStrides &&
                      slots[fIdx].boundStrides[1] == 1)) {
            emitContigSegLoop();
          } else {
            Value s1 = slots[fIdx].strideVals[1];
            Value isContig = b.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::eq, s1, c1);
            auto ifContig =
                b.create<scf::IfOp>(loc, TypeRange{}, isContig, true);
            {
              OpBuilder::InsertionGuard tg(b);
              b.setInsertionPointToStart(&ifContig.getThenRegion().front());
              emitContigSegLoop();
            }
            {
              OpBuilder::InsertionGuard eg(b);
              b.setInsertionPointToStart(&ifContig.getElseRegion().front());
              Value cW = constIdx(b, loc, w);
              auto *mLoop =
                  makeOp(b, loc, "for_range", {start, end, c1}, {}, {}, 1);
              Block &mBb = mLoop->getRegion(0).emplaceBlock();
              Value member = mBb.addArgument(b.getIndexType(), loc);
              b.setInsertionPointToEnd(&mBb);
              auto *lLoop =
                  makeOp(b, loc, "for_range", {c0, cW, c1}, {}, {}, 1);
              Block &lBb = lLoop->getRegion(0).emplaceBlock();
              Value lane = lBb.addArgument(b.getIndexType(), loc);
              b.setInsertionPointToEnd(&lBb);
              Value fAddr = emitViewAddr(
                  fIdx, {member, b.create<arith::AddIOp>(loc, j, lane)});
              Value raw = makeOp(b, loc, "load_scalar",
                                 {slots[fIdx].ptrVal, fAddr}, {fElem})
                              ->getResult(0);
              Value fVal =
                  fElem.isF32()
                      ? raw
                      : b.create<arith::ExtFOp>(loc, b.getF32Type(), raw)
                            .getResult();
              Value prev = makeOp(b, loc, "extract_lane_dyn",
                                  {readVecSlot(acc), lane}, {b.getF32Type()})
                               ->getResult(0);
              makeOp(b, loc, "acc_insert_lane",
                     {acc, lane, b.create<arith::AddFOp>(loc, prev, fVal)});
              makeOp(b, loc, "end");
              b.setInsertionPointToEnd(&mBb);
              makeOp(b, loc, "end");
            }
          }
          map.map(op.getResult(0),
                  w == 1 ? readScalarSlot(acc) : readVecSlot(acc));
          continue;
        }

        auto evalTermAt = [&](Value memberVal) -> llvm::Expected<Value> {
          Value colCoord = j;
          IRMapping tMap;
          auto &terms = op.getRegion(0).front();
          tMap.map(terms.getArgument(0), memberVal);
          tMap.map(terms.getArgument(1), colCoord);
          for (Operation &v : terms) {
            auto tn = v.getName().stripDialect();
            if (tn == "term")
              return tMap.lookup(v.getOperand(0));
            if (tn == "member_load") {
              unsigned mIdx = slotIndex(v.getOperand(0));
              SmallVector<Value, 4> mCoords;
              for (Value cv : v.getOperands().drop_front()) {
                Value mv = tMap.lookupOrNull(cv);
                if (!mv)
                  mv = map.lookup(cv);
                mCoords.push_back(asInt(b, loc, mv));
              }
              Value mAddr = (forceUnitStride && slots[mIdx].rank == 1 &&
                             mCoords.size() == 1)
                                ? makeStaticAddr(b, loc, slots[mIdx].ptrVal,
                                                 mCoords, {1})
                                : emitViewAddr(mIdx, mCoords);
              Value loaded =
                  makeOp(b, loc, "load_scalar", {slots[mIdx].ptrVal, mAddr},
                         {slots[mIdx].elemTy})
                      ->getResult(0);
              tMap.map(v.getResult(0), loaded);
            } else if (tn == "cast") {
              Type dt = v.getResult(0).getType();
              Value casted =
                  makeOp(b, loc, "scalar_cast", {tMap.lookup(v.getOperand(0))},
                         {dt},
                         {b.getNamedAttr("dtype", b.getStringAttr(dtypeBare(dt)))})
                      ->getResult(0);
              tMap.map(v.getResult(0), casted);
            } else if (tn == "mul" || tn == "add") {
              Value lhs = tMap.lookup(v.getOperand(0)),
                    rhs = tMap.lookup(v.getOperand(1));
              Value res = tn == "mul"
                              ? b.create<arith::MulFOp>(loc, lhs, rhs).getResult()
                              : b.create<arith::AddFOp>(loc, lhs, rhs).getResult();
              tMap.map(v.getResult(0), res);
            } else {
              return llvm::createStringError("illegal scalar member op " +
                                             tn.str());
            }
          }
          return llvm::createStringError("missing segment_fold term");
        };

        auto emitOneSegLane = [&](Value laneVal) -> llvm::Expected<Value> {
          Value colCoord =
              w == 1 ? j : b.create<arith::AddIOp>(loc, j, laneVal).getResult();
          Value segOwner =
              flatDomain
                  ? (memberTasks
                         ? b.create<arith::AddIOp>(loc, r, colCoord).getResult()
                         : colCoord)
                  : r;
          if (code == "segment_size") {
            Value start = loadOff(segOwner);
            Value end = loadOff(b.create<arith::AddIOp>(loc, segOwner, c1));
            Value diff = b.create<arith::SubIOp>(loc, end, start);
            return makeOp(b, loc, "scalar_cast", {diff}, {b.getF32Type()},
                          {b.getNamedAttr("dtype", b.getStringAttr("float32"))})
                ->getResult(0);
          }
          auto &terms = op.getRegion(0).front();
          Value sAcc = declScalarSlot(b.getF32Type(), constF32(b, loc, 0));

          SmallVector<unsigned, 4> rank1Slots;
          bool allRank1 = slots[offIdx].rank == 1;
          if (allRank1) {
            rank1Slots.push_back(offIdx);
            for (Operation &v : terms) {
              if (v.getName().stripDialect() == "member_load") {
                unsigned mIdx = slotIndex(v.getOperand(0));
                if (slots[mIdx].rank != 1) {
                  allRank1 = false;
                  break;
                }
                if (!llvm::is_contained(rank1Slots, mIdx))
                  rank1Slots.push_back(mIdx);
              }
            }
          }

          auto canVecSegTerms = [&]() -> bool {
            if (!forceUnitStride || !allRank1 || simdWidth < 8)
              return false;
            llvm::DenseSet<Value> vecVals;
            vecVals.insert(terms.getArgument(0));
            for (Operation &v : terms) {
              auto tn = v.getName().stripDialect();
              if (tn == "term")
                return vecVals.contains(v.getOperand(0));
              if (tn == "member_load") {
                if (v.getNumOperands() != 2)
                  return false;
                unsigned mIdx = slotIndex(v.getOperand(0));
                if (slots[mIdx].rank != 1)
                  return false;
                Value cVal = v.getOperand(1);
                if (cVal != terms.getArgument(0))
                  return false;
                vecVals.insert(v.getResult(0));
              } else if (tn == "cast") {
                if (!vecVals.contains(v.getOperand(0)))
                  return false;
                Type srcT = v.getOperand(0).getType(),
                     dstT = v.getResult(0).getType();
                if (!((isa<FloatType>(srcT) && dstT.isF32()) || srcT == dstT))
                  return false;
                vecVals.insert(v.getResult(0));
              } else if (tn == "mul" || tn == "add") {
                if (!v.getResult(0).getType().isF32())
                  return false;
                vecVals.insert(v.getResult(0));
              } else {
                return false;
              }
            }
            return false;
          };

          auto evalVecSegAt = [&](Value m0, Value laneCol, int64_t kW) -> Value {
            IRMapping vMap;
            auto vecF32 = VectorType::get({kW}, b.getF32Type());
            for (Operation &v : terms) {
              auto tn = v.getName().stripDialect();
              if (tn == "term")
                return vMap.lookup(v.getOperand(0));
              if (tn == "member_load") {
                unsigned mIdx = slotIndex(v.getOperand(0));
                Type eTy = slots[mIdx].elemTy;
                auto vTy = VectorType::get({kW}, eTy);
                Value cVal = v.getOperand(1);
                Value loaded;
                if (cVal == terms.getArgument(0)) {
                  Value mAddr =
                      makeStaticAddr(b, loc, slots[mIdx].ptrVal, {m0}, {1});
                  loaded = makeOp(b, loc, "load_slice",
                                  {slots[mIdx].ptrVal, mAddr}, {vTy},
                                  {b.getNamedAttr("width",
                                                  b.getI64IntegerAttr(kW))})
                               ->getResult(0);
                } else {
                  Value idxVec = vMap.lookup(cVal);
                  loaded = makeOp(b, loc, "gather_slice",
                                  {slots[mIdx].ptrVal, idxVec}, {vTy})
                               ->getResult(0);
                }
                vMap.map(v.getResult(0), loaded);
              } else if (tn == "cast") {
                Value inV = vMap.lookup(v.getOperand(0));
                Type inElem = cast<VectorType>(inV.getType()).getElementType();
                vMap.map(v.getResult(0),
                         inElem.isF32()
                             ? inV
                             : b.create<arith::ExtFOp>(loc, vecF32, inV)
                                   .getResult());
              } else if (tn == "mul" || tn == "add") {
                auto getVecOpnd = [&](Value opnd) -> Value {
                  if (Value mapped = vMap.lookupOrNull(opnd))
                    return mapped;
                  Value sc = opnd == terms.getArgument(1) ? laneCol
                                                          : map.lookup(opnd);
                  return makeOp(b, loc, "splat_scalar", {sc}, {vecF32},
                                {b.getNamedAttr("dtype",
                                                b.getStringAttr("float32")),
                                 b.getNamedAttr("width",
                                                b.getI64IntegerAttr(kW))})
                      ->getResult(0);
                };
                Value lhs = getVecOpnd(v.getOperand(0));
                Value rhs = getVecOpnd(v.getOperand(1));
                Value res = tn == "mul"
                                ? b.create<arith::MulFOp>(loc, lhs, rhs)
                                      .getResult()
                                : b.create<arith::AddFOp>(loc, lhs, rhs)
                                      .getResult();
                vMap.map(v.getResult(0), res);
              }
            }
            return {};
          };

          auto emitVec16SegLoopAt = [&](Value start, Value stop16,
                                        Value laneCol, Value accSlot) -> bool {
            if (!canVecSegTerms())
              return false;
            const int64_t kW = simdWidth >= 16 ? 16 : 8;
            Value stepW = constIdx(b, loc, kW);
            auto *mVecLoop =
                makeOp(b, loc, "for_range", {start, stop16, stepW}, {}, {}, 1);
            Block &mvBb = mVecLoop->getRegion(0).emplaceBlock();
            Value m0 = mvBb.addArgument(b.getIndexType(), loc);
            OpBuilder::InsertionGuard mvg(b);
            b.setInsertionPointToEnd(&mvBb);
            Value prodVec = evalVecSegAt(m0, laneCol, kW);
            Value accVal = readScalarSlot(accSlot);
            for (int64_t lane = 0; lane < kW; ++lane) {
              Value lv =
                  makeOp(b, loc, "extract_lane", {prodVec}, {b.getF32Type()},
                         {b.getNamedAttr("lane", b.getI64IntegerAttr(lane))})
                      ->getResult(0);
              accVal = b.create<arith::AddFOp>(loc, accVal, lv);
            }
            writeScalarSlot(accSlot, accVal);
            makeOp(b, loc, "end");
            return true;
          };

          auto emitScalarRangeAt = [&](Value start, Value end, Value laneCol,
                                       Value accSlot) -> llvm::Error {
            Value rawSpan = b.create<arith::SubIOp>(loc, end, start);
            Value span = b.create<arith::MaxSIOp>(loc, c0, rawSpan);
            const int64_t segW = simdWidth >= 16 ? 16 : 8;
            Value alignedW =
                b.create<arith::AndIOp>(loc, span, constIdx(b, loc, -segW));
            Value stopW = b.create<arith::AddIOp>(loc, start, alignedW);
            Value tailStart = start;
            if (emitVec16SegLoopAt(start, stopW, laneCol, accSlot)) {
              tailStart = stopW;
            } else {
              Value aligned8 =
                  b.create<arith::AndIOp>(loc, span, constIdx(b, loc, -8));
              Value stop8 = b.create<arith::AddIOp>(loc, start, aligned8);
              Value c8 = constIdx(b, loc, 8);
              {
                auto *m8Loop =
                    makeOp(b, loc, "for_range", {start, stop8, c8}, {}, {}, 1);
                Block &m8Bb = m8Loop->getRegion(0).emplaceBlock();
                Value m0 = m8Bb.addArgument(b.getIndexType(), loc);
                OpBuilder::InsertionGuard m8g(b);
                b.setInsertionPointToEnd(&m8Bb);
                Value prevJ = j;
                j = laneCol;
                Value tVals[8];
                for (int u = 0; u < 8; ++u) {
                  Value mu = u == 0
                                 ? m0
                                 : b.create<arith::AddIOp>(loc, m0,
                                                           constIdx(b, loc, u))
                                       .getResult();
                  auto tv = evalTermAt(mu);
                  if (!tv) {
                    j = prevJ;
                    return tv.takeError();
                  }
                  tVals[u] = *tv;
                }
                j = prevJ;
                Value accVal = readScalarSlot(accSlot);
                for (int u = 0; u < 8; ++u)
                  accVal = b.create<arith::AddFOp>(loc, accVal, tVals[u]);
                writeScalarSlot(accSlot, accVal);
                makeOp(b, loc, "end");
              }
              tailStart = stop8;
            }
            auto *mLoop =
                makeOp(b, loc, "for_range", {tailStart, end, c1}, {}, {}, 1);
            Block &mBb = mLoop->getRegion(0).emplaceBlock();
            Value member = mBb.addArgument(b.getIndexType(), loc);
            OpBuilder::InsertionGuard mg(b);
            b.setInsertionPointToEnd(&mBb);
            Value prevJ = j;
            j = laneCol;
            auto tVal = evalTermAt(member);
            j = prevJ;
            if (!tVal)
              return tVal.takeError();
            Value cur = readScalarSlot(accSlot);
            writeScalarSlot(accSlot, b.create<arith::AddFOp>(loc, cur, *tVal));
            makeOp(b, loc, "end");
            return llvm::Error::success();
          };

          auto emitScalarRange = [&](Value start, Value end) -> llvm::Error {
            return emitScalarRangeAt(start, end, colCoord, sAcc);
          };

          if (allRank1) {
            auto loadOffUnit = [&](Value idxVal) {
              Value a =
                  makeStaticAddr(b, loc, slots[offIdx].ptrVal, {idxVal}, {1});
              Value raw = makeOp(b, loc, "load_scalar",
                                 {slots[offIdx].ptrVal, a}, {b.getI32Type()})
                              ->getResult(0);
              return asInt(b, loc, raw);
            };
            Value allUnit;
            for (unsigned sIdx : rank1Slots) {
              if (forceUnitStride ||
                  (slots[sIdx].boundStrides && slots[sIdx].boundStrides[0] == 1))
                continue;
              Value eq1 = b.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::eq, slots[sIdx].strideVals[0], c1);
              allUnit = allUnit ? b.create<arith::AndIOp>(loc, allUnit, eq1).getResult()
                                : eq1;
            }
            if (!allUnit) {
              bool prevFast = forceUnitStride;
              forceUnitStride = true;
              Value start = loadOffUnit(segOwner);
              Value end =
                  loadOffUnit(b.create<arith::AddIOp>(loc, segOwner, c1));
              if (auto err = emitScalarRange(start, end)) {
                forceUnitStride = prevFast;
                return std::move(err);
              }
              forceUnitStride = prevFast;
              return readScalarSlot(sAcc);
            }
            auto ifUnit = b.create<scf::IfOp>(loc, TypeRange{}, allUnit, true);
            {
              OpBuilder::InsertionGuard tg(b);
              b.setInsertionPointToStart(&ifUnit.getThenRegion().front());
              bool prevFast = forceUnitStride;
              forceUnitStride = true;
              Value start = loadOffUnit(segOwner);
              Value end = loadOffUnit(b.create<arith::AddIOp>(loc, segOwner, c1));
              if (auto err = emitScalarRange(start, end)) {
                forceUnitStride = prevFast;
                return std::move(err);
              }
              forceUnitStride = prevFast;
            }
            {
              OpBuilder::InsertionGuard eg(b);
              b.setInsertionPointToStart(&ifUnit.getElseRegion().front());
              Value start = loadOff(segOwner);
              Value end = loadOff(b.create<arith::AddIOp>(loc, segOwner, c1));
              if (auto err = emitScalarRange(start, end))
                return std::move(err);
            }
            return readScalarSlot(sAcc);
          }

          Value start = loadOff(segOwner);
          Value end = loadOff(b.create<arith::AddIOp>(loc, segOwner, c1));
          if (auto err = emitScalarRange(start, end))
            return std::move(err);
          return readScalarSlot(sAcc);
        };

        if (w == 1) {
          auto valOrErr = emitOneSegLane(c0);
          if (!valOrErr)
            return op.emitError() << llvm::toString(valOrErr.takeError()),
                   failure();
          map.map(op.getResult(0), *valOrErr);
        } else {
          Value vecSlot = declVecSlot(resTy, w);
          auto &terms = op.getRegion(0).front();
          SmallVector<unsigned, 4> vecRank1Slots;
          bool vecAllRank1 =
              code == "segment_fold" && w >= 4 && w % 4 == 0 &&
              simdWidth >= 8 && slots[offIdx].rank == 1 &&
              cast<VectorType>(resTy).getElementType().isF32();
          if (vecAllRank1) {
            vecRank1Slots.push_back(offIdx);
            for (Operation &v : terms) {
              auto tn = v.getName().stripDialect();
              if (tn == "term")
                break;
              if (tn == "member_load") {
                if (v.getNumOperands() != 2) {
                  vecAllRank1 = false;
                  break;
                }
                unsigned mIdx = slotIndex(v.getOperand(0));
                if (slots[mIdx].rank != 1) {
                  vecAllRank1 = false;
                  break;
                }
                Value cVal = v.getOperand(1);
                if (cVal != terms.getArgument(0) &&
                    !cVal.getType().isInteger(32) &&
                    !cVal.getType().isInteger(64)) {
                  vecAllRank1 = false;
                  break;
                }
                if (!llvm::is_contained(vecRank1Slots, mIdx))
                  vecRank1Slots.push_back(mIdx);
              } else if (tn != "cast" && tn != "mul" && tn != "add") {
                vecAllRank1 = false;
                break;
              }
            }
          }
          auto emitSerialLanes = [&]() -> LogicalResult {
            auto *lLoop = makeOp(b, loc, "for_range", {c0, cW, c1}, {}, {}, 1);
            Block &lBb = lLoop->getRegion(0).emplaceBlock();
            Value lane = lBb.addArgument(b.getIndexType(), loc);
            OpBuilder::InsertionGuard lg(b);
            b.setInsertionPointToEnd(&lBb);
            auto valOrErr = emitOneSegLane(lane);
            if (!valOrErr)
              return op.emitError() << llvm::toString(valOrErr.takeError()),
                     failure();
            makeOp(b, loc, "acc_insert_lane", {vecSlot, lane, *valOrErr});
            makeOp(b, loc, "end");
            return success();
          };
          auto emitLockstep4Lanes = [&]() -> LogicalResult {
            auto loadOffUnit = [&](Value idxVal) {
              Value a =
                  makeStaticAddr(b, loc, slots[offIdx].ptrVal, {idxVal}, {1});
              Value raw = makeOp(b, loc, "load_scalar",
                                 {slots[offIdx].ptrVal, a}, {b.getI32Type()})
                              ->getResult(0);
              return asInt(b, loc, raw);
            };
            const int lockK = (w >= 8 && w % 8 == 0) ? 8 : 4;
            Value c4 = constIdx(b, loc, 4);
            Value mask4 = constIdx(b, loc, -4);
            bool prevFast = forceUnitStride;
            forceUnitStride = true;
            SmallVector<Value, 16> laneVals;
            for (int64_t g = 0; g < w; g += lockK) {
              Value colsK[8], segsK[8], startsK[8], endsK[8], spansK[8], accsK[8];
              for (int u = 0; u < lockK; ++u) {
                Value laneOff = constIdx(b, loc, g + u);
                colsK[u] = (g + u == 0)
                               ? j
                               : b.create<arith::AddIOp>(loc, j, laneOff).getResult();
                segsK[u] =
                    flatDomain
                        ? (memberTasks
                               ? b.create<arith::AddIOp>(loc, r, colsK[u]).getResult()
                               : colsK[u])
                        : r;
                startsK[u] = (u > 0 && flatDomain) ? endsK[u - 1]
                                                   : loadOffUnit(segsK[u]);
                endsK[u] =
                    loadOffUnit(b.create<arith::AddIOp>(loc, segsK[u], c1));
                spansK[u] = b.create<arith::MaxSIOp>(
                    loc, c0, b.create<arith::SubIOp>(loc, endsK[u], startsK[u]));
                accsK[u] = declScalarSlot(b.getF32Type(), constF32(b, loc, 0));
              }
              Value minSpan = spansK[0];
              for (int u = 1; u < lockK; ++u)
                minSpan = b.create<arith::MinSIOp>(loc, minSpan, spansK[u]);
              Value common4 = b.create<arith::AndIOp>(loc, minSpan, mask4);
              {
                auto *m4Loop =
                    makeOp(b, loc, "for_range", {c0, common4, c4}, {}, {}, 1);
                Block &mvBb = m4Loop->getRegion(0).emplaceBlock();
                Value tOff = mvBb.addArgument(b.getIndexType(), loc);
                OpBuilder::InsertionGuard mvg(b);
                b.setInsertionPointToEnd(&mvBb);
                Value curAccs[8], ps[8];
                for (int u = 0; u < lockK; ++u) {
                  curAccs[u] = readScalarSlot(accsK[u]);
                  ps[u] = b.create<arith::AddIOp>(loc, startsK[u], tOff);
                }
                for (int ln = 0; ln < 4; ++ln) {
                  Value lnOff = constIdx(b, loc, ln);
                  for (int u = 0; u < lockK; ++u) {
                    Value mu = ln == 0
                                   ? ps[u]
                                   : b.create<arith::AddIOp>(loc, ps[u], lnOff)
                                         .getResult();
                    Value prevJ = j;
                    j = colsK[u];
                    auto tv = evalTermAt(mu);
                    j = prevJ;
                    if (!tv) {
                      forceUnitStride = prevFast;
                      return op.emitError() << llvm::toString(tv.takeError()),
                             failure();
                    }
                    curAccs[u] = b.create<arith::AddFOp>(loc, curAccs[u], *tv);
                  }
                }
                for (int u = 0; u < lockK; ++u)
                  writeScalarSlot(accsK[u], curAccs[u]);
                makeOp(b, loc, "end");
              }
              for (int u = 0; u < lockK; ++u) {
                Value remStart =
                    b.create<arith::AddIOp>(loc, startsK[u], common4);
                auto *rtLoop = makeOp(b, loc, "for_range",
                                      {remStart, endsK[u], c1}, {}, {}, 1);
                Block &rtBb = rtLoop->getRegion(0).emplaceBlock();
                Value member = rtBb.addArgument(b.getIndexType(), loc);
                {
                  OpBuilder::InsertionGuard rtg(b);
                  b.setInsertionPointToEnd(&rtBb);
                  Value prevJ = j;
                  j = colsK[u];
                  auto tv = evalTermAt(member);
                  j = prevJ;
                  if (!tv) {
                    forceUnitStride = prevFast;
                    return op.emitError() << llvm::toString(tv.takeError()),
                           failure();
                  }
                  writeScalarSlot(
                      accsK[u],
                      b.create<arith::AddFOp>(loc, readScalarSlot(accsK[u]), *tv));
                  makeOp(b, loc, "end");
                }
                laneVals.push_back(readScalarSlot(accsK[u]));
              }
            }
            forceUnitStride = prevFast;
            Value packed =
                makeOp(b, loc, "pack_apply", laneVals, {resTy},
                       {b.getNamedAttr("in_dtype", b.getStringAttr("float32")),
                        b.getNamedAttr("out_dtype", b.getStringAttr("float32")),
                        b.getNamedAttr("kind", b.getStringAttr("id"))})
                    ->getResult(0);
            writeVecSlot(vecSlot, packed);
            return success();
          };
          if (vecAllRank1) {
            Value allUnit;
            for (unsigned sIdx : vecRank1Slots) {
              if (forceUnitStride ||
                  (slots[sIdx].boundStrides && slots[sIdx].boundStrides[0] == 1))
                continue;
              Value eq1 = b.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::eq, slots[sIdx].strideVals[0], c1);
              allUnit = allUnit
                            ? b.create<arith::AndIOp>(loc, allUnit, eq1).getResult()
                            : eq1;
            }
            if (!allUnit) {
              if (failed(emitLockstep4Lanes()))
                return failure();
            } else {
              auto ifUnit = b.create<scf::IfOp>(loc, TypeRange{}, allUnit, true);
              {
                OpBuilder::InsertionGuard tg(b);
                b.setInsertionPointToStart(&ifUnit.getThenRegion().front());
                if (failed(emitLockstep4Lanes()))
                  return failure();
              }
              {
                OpBuilder::InsertionGuard eg(b);
                b.setInsertionPointToStart(&ifUnit.getElseRegion().front());
                if (failed(emitSerialLanes()))
                  return failure();
              }
            }
          } else {
            if (failed(emitSerialLanes()))
              return failure();
          }
          map.map(op.getResult(0), readVecSlot(vecSlot));
        }
        continue;
      }
      if (code == "indexed_read") {
        unsigned tIdx = slotIndex(op.getOperand(0));
        unsigned idIdx = slotIndex(op.getOperand(1));
        Type elemTy = slots[tIdx].elemTy;
        Value idAddr = emitViewAddr(idIdx, {r});
        Value idRaw = makeOp(b, loc, "load_scalar",
                             {slots[idIdx].ptrVal, idAddr},
                             {slots[idIdx].elemTy})
                          ->getResult(0);
        Value rowId = asInt(b, loc, idRaw);
        if (w == 1) {
          Value tAddr = emitViewAddr(tIdx, {rowId, j});
          Value val = makeOp(b, loc, "load_scalar",
                             {slots[tIdx].ptrVal, tAddr}, {elemTy})
                          ->getResult(0);
          map.map(op.getResult(0), val);
        } else {
          auto vecTy = VectorType::get({w}, elemTy);
          Value resSlot = declVecSlot(vecTy, w);
          Value s1 = slots[tIdx].strideVals[1];
          Value isContig =
              b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, s1, c1);
          auto ifContig = b.create<scf::IfOp>(loc, TypeRange{}, isContig, true);
          {
            OpBuilder::InsertionGuard tg(b);
            b.setInsertionPointToStart(&ifContig.getThenRegion().front());
            Value tAddr = emitViewAddr(tIdx, {rowId, j});
            Value v = makeOp(b, loc, "load_slice", {slots[tIdx].ptrVal, tAddr},
                             {vecTy},
                             {b.getNamedAttr("width", b.getI64IntegerAttr(w))})
                          ->getResult(0);
            writeVecSlot(resSlot, v);
          }
          {
            OpBuilder::InsertionGuard eg(b);
            b.setInsertionPointToStart(&ifContig.getElseRegion().front());
            auto *lLoop = makeOp(b, loc, "for_range", {c0, cW, c1}, {}, {}, 1);
            Block &lBb = lLoop->getRegion(0).emplaceBlock();
            Value lane = lBb.addArgument(b.getIndexType(), loc);
            b.setInsertionPointToEnd(&lBb);
            Value tAddr = emitViewAddr(
                tIdx, {rowId, b.create<arith::AddIOp>(loc, j, lane)});
            Value elem = makeOp(b, loc, "load_scalar",
                                {slots[tIdx].ptrVal, tAddr}, {elemTy})
                             ->getResult(0);
            makeOp(b, loc, "acc_insert_lane", {resSlot, lane, elem});
            makeOp(b, loc, "end");
          }
          map.map(op.getResult(0), readVecSlot(resSlot));
        }
        continue;
      }
      if (code == "owner_read" || code == "owner_write") {
        unsigned sIdx = slotIndex(op.getOperand(0));
        Value addr = emitViewAddr(sIdx, {r});
        if (code == "owner_read") {
          Value val = makeOp(b, loc, "load_scalar",
                             {slots[sIdx].ptrVal, addr}, {slots[sIdx].elemTy})
                          ->getResult(0);
          map.map(op.getResult(0), val);
        } else {
          makeOp(b, loc, "store_scalar",
                 {slots[sIdx].ptrVal, addr, map.lookup(op.getOperand(2))});
        }
        continue;
      }
      if (code == "read") {
        unsigned sIdx = slotIndex(op.getOperand(0));
        bool swap = op.hasAttr("axes");
        std::pair<unsigned, bool> rKey{sIdx, swap};
        if (!writtenInBb.contains(sIdx)) {
          auto it = readCache.find(rKey);
          if (it != readCache.end()) {
            map.map(op.getResult(0), it->second);
            continue;
          }
        }
        Value loaded = emitSweepRead(op, r, j, w);
        if (!writtenInBb.contains(sIdx))
          readCache[rKey] = loaded;
        map.map(op.getResult(0), loaded);
        continue;
      }
      if (code == "write") {
        Value rawVal = map.lookup(op.getOperand(3));
        unsigned sIdx = slotIndex(op.getOperand(0));
        Type targetTy = w == 1 ? slots[sIdx].elemTy
                               : Type(VectorType::get({w}, slots[sIdx].elemTy));
        Value splatted = splatIfScalar(rawVal, targetTy, w);
        if (deferredStores && deferViewWrites) {
          Operation *writeOpPtr = &op;
          deferredStores->push_back([=, this]() {
            emitSweepWrite(*writeOpPtr, r, j, splatted, w);
          });
        } else {
          emitSweepWrite(op, r, j, splatted, w);
        }
        continue;
      }
      if (code == "apply") {
        auto opKind = op.getAttrOfType<StringAttr>("kind").getValue();
        Type origResTy = op.getResult(0).getType();
        Type elemResTy = isa<ShapedType>(origResTy)
                             ? cast<ShapedType>(origResTy).getElementType()
                             : origResTy;
        Type targetTy = w == 1 && isa<ShapedType>(origResTy) ? elemResTy
                                                             : origResTy;
        if (opKind == "ordinal") {
          Value baseIdx =
              (flatDomain && memberTasks)
                  ? b.create<arith::AddIOp>(loc, r, j).getResult()
                  : b.create<arith::AddIOp>(
                        loc, b.create<arith::MulIOp>(loc, r, colsIdx), j)
                        .getResult();
          if (w >= 4 && w % 4 == 0 && !op.getResult(0).use_empty() &&
              llvm::all_of(op.getResult(0).getUsers(), [](Operation *u) {
                auto uk = u->getAttrOfType<StringAttr>("kind");
                return u->getName().getStringRef() == "mojo_row.apply" && uk &&
                       uk.getValue() == "uniform";
              })) {
            Value baseU64 =
                makeOp(b, loc, "scalar_cast", {baseIdx},
                       {b.getIntegerType(64, false)},
                       {b.getNamedAttr("dtype", b.getStringAttr("uint64"))})
                    ->getResult(0);
            ordinalScalarBases[op.getResult(0)] = baseU64;
            continue;
          }
          Value res =
              makeOp(b, loc, "simd_iota_add", {baseIdx}, {targetTy},
                     {b.getNamedAttr("dtype",
                                     b.getStringAttr(dtypeBare(elemResTy))),
                      b.getNamedAttr("width", b.getI64IntegerAttr(w))})
                  ->getResult(0);
          map.map(op.getResult(0), res);
          continue;
        }
        if (opKind == "field") {
          int64_t pos = op.getAttrOfType<IntegerAttr>("position").getInt();
          Value res =
              makeOp(b, loc, "tuple_field", {map.lookup(op.getOperand(0))},
                     {elemResTy},
                     {b.getNamedAttr("position", b.getI64IntegerAttr(pos))})
                  ->getResult(0);
          map.map(op.getResult(0), res);
          continue;
        }
        if (opKind == "scan_sum") {
          int64_t cId = op.getAttrOfType<IntegerAttr>("carry").getInt();
          if (!scanCarries.count(cId)) {
            Value z =
                makeOp(b, loc, "scalar_const", {}, {elemResTy},
                       {b.getNamedAttr("dtype",
                                       b.getStringAttr(dtypeBare(elemResTy))),
                        b.getNamedAttr("literal", b.getStringAttr("0"))})
                    ->getResult(0);
            scanCarries[cId] = declScalarSlot(elemResTy, z);
          }
          Value cSlot = scanCarries[cId];
          auto *scOp = makeOp(
              b, loc, "simd_scan_sum",
              {map.lookup(op.getOperand(0)), readScalarSlot(cSlot)},
              {targetTy, elemResTy},
              {b.getNamedAttr("dtype", b.getStringAttr(dtypeBare(elemResTy))),
               b.getNamedAttr("width", b.getI64IntegerAttr(w)),
               b.getNamedAttr("ci.implementation",
                              b.getStringAttr("exits/mojo::simd-scan-sum"))});
          writeScalarSlot(cSlot, scOp->getResult(1));
          map.map(op.getResult(0), scOp->getResult(0));
          continue;
        }
        if (opKind == "uniform") {
          auto itOrd = ordinalScalarBases.find(op.getOperand(1));
          Value cntOperand = itOrd != ordinalScalarBases.end()
                                 ? itOrd->second
                                 : map.lookup(op.getOperand(1));
          Value res =
              makeOp(b, loc, "uniform",
                     {map.lookup(op.getOperand(0)), cntOperand},
                     {targetTy},
                     {b.getNamedAttr("width", b.getI64IntegerAttr(w)),
                      b.getNamedAttr("ci.implementation",
                                     b.getStringAttr("exits/mojo::philox-uniform"))})
                  ->getResult(0);
          map.map(op.getResult(0), res);
          continue;
        }
        if (opKind == "constant") {
          Value res =
              makeOp(b, loc, "scalar_const", {}, {elemResTy},
                     {b.getNamedAttr("dtype",
                                     b.getStringAttr(dtypeBare(elemResTy))),
                      b.getNamedAttr("literal",
                                     b.getStringAttr(intentdsl::kir::literal(
                                         op.getAttrOfType<TypedAttr>("value"))))})
                  ->getResult(0);
          map.map(op.getResult(0), res);
          continue;
        }
        if (opKind == "cast") {
          Value inVal = map.lookup(op.getOperand(0));
          Type outTy = isa<ShapedType>(inVal.getType()) ? targetTy : elemResTy;
          Value res =
              makeOp(b, loc, "scalar_cast", {inVal}, {outTy},
                     {b.getNamedAttr("dtype",
                                     b.getStringAttr(dtypeBare(elemResTy)))})
                  ->getResult(0);
          map.map(op.getResult(0), res);
          continue;
        }
        if (opKind == "sqrt" || opKind == "exp" || opKind == "rsqrt") {
          Value inVal = splatIfScalar(map.lookup(op.getOperand(0)), targetTy, w);
          Value res = opKind == "sqrt"
                          ? b.create<math::SqrtOp>(loc, inVal).getResult()
                          : (opKind == "exp"
                                 ? b.create<math::ExpOp>(loc, inVal).getResult()
                                 : b.create<math::RsqrtOp>(loc, inVal).getResult());
          map.map(op.getResult(0), res);
          continue;
        }
        if (opKind == "select") {
          Value cond = map.lookup(op.getOperand(0));
          Value onT = splatIfScalar(map.lookup(op.getOperand(1)), targetTy, w);
          Value onF = splatIfScalar(map.lookup(op.getOperand(2)), targetTy, w);
          Value res =
              isa<ShapedType>(cond.getType())
                  ? makeOp(b, loc, "simd_select", {cond, onT, onF}, {targetTy})
                        ->getResult(0)
                  : b.create<arith::SelectOp>(loc, cond, onT, onF).getResult();
          map.map(op.getResult(0), res);
          continue;
        }
        if (opKind == "maximum") {
          Value lhs = splatIfScalar(map.lookup(op.getOperand(0)), targetTy, w);
          Value rhs = splatIfScalar(map.lookup(op.getOperand(1)), targetTy, w);
          map.map(op.getResult(0), b.create<arith::MaxNumFOp>(loc, lhs, rhs));
          continue;
        }
        if (opKind == "gt") {
          Value pTy =
              w > 1 && (isa<ShapedType>(map.lookup(op.getOperand(0)).getType()) ||
                        isa<ShapedType>(map.lookup(op.getOperand(1)).getType()))
                  ? splatIfScalar(map.lookup(op.getOperand(0)),
                                  VectorType::get({w}, map.lookup(op.getOperand(0)).getType()),
                                  w)
                  : map.lookup(op.getOperand(0));
          Type cmpOperandTy = pTy.getType();
          Value lhs = splatIfScalar(map.lookup(op.getOperand(0)), cmpOperandTy, w);
          Value rhs = splatIfScalar(map.lookup(op.getOperand(1)), cmpOperandTy, w);
          Value res;
          if (isa<ShapedType>(cmpOperandTy)) {
            res = makeOp(b, loc, "simd_cmp_gt", {lhs, rhs}, {targetTy})
                      ->getResult(0);
          } else if (isa<FloatType>(cmpOperandTy)) {
            res = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, lhs,
                                          rhs);
          } else {
            res =
                b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, lhs, rhs);
          }
          map.map(op.getResult(0), res);
          continue;
        }
        // Binary arithmetic / bitwise ops
        Value raw0 = map.lookup(op.getOperand(0)),
              raw1 = map.lookup(op.getOperand(1));
        Value lhs = splatIfScalar(raw0, targetTy, w);
        Value rhs = splatIfScalar(raw1, targetTy, w);
        bool isFloat = isa<FloatType>(elemResTy);
        Value res;
        if (opKind == "add")
          res = isFloat ? b.create<arith::AddFOp>(loc, lhs, rhs).getResult()
                        : b.create<arith::AddIOp>(loc, lhs, rhs).getResult();
        else if (opKind == "sub")
          res = isFloat ? b.create<arith::SubFOp>(loc, lhs, rhs).getResult()
                        : b.create<arith::SubIOp>(loc, lhs, rhs).getResult();
        else if (opKind == "mul")
          res = isFloat ? b.create<arith::MulFOp>(loc, lhs, rhs).getResult()
                        : b.create<arith::MulIOp>(loc, lhs, rhs).getResult();
        else if (opKind == "div") {
          Value exactInvScalar;
          if (isFloat) {
            if (Operation *rhsDef = op.getOperand(1).getDefiningOp()) {
              if (rhsDef->getName().stripDialect() == "apply" &&
                  rhsDef->getAttrOfType<StringAttr>("kind").getValue() ==
                      "constant") {
                if (auto fAttr = rhsDef->getAttrOfType<FloatAttr>("value")) {
                  llvm::APFloat invVal(fAttr.getValue().getSemantics());
                  if (fAttr.getValue().getExactInverse(&invVal)) {
                    OpBuilder::InsertionGuard cg(b);
                    if (sweepHoistBlock) {
                      if (sweepHoistBefore)
                        b.setInsertionPoint(sweepHoistBefore);
                      else
                        b.setInsertionPointToEnd(sweepHoistBlock);
                    }
                    exactInvScalar =
                        makeOp(
                            b, loc, "scalar_const", {}, {elemResTy},
                            {b.getNamedAttr("dtype",
                                            b.getStringAttr(dtypeBare(elemResTy))),
                             b.getNamedAttr(
                                 "literal",
                                 b.getStringAttr(intentdsl::kir::literal(
                                     FloatAttr::get(elemResTy, invVal))))})
                            ->getResult(0);
                  }
                }
              }
            }
          }
          if (exactInvScalar) {
            Value invRhs = splatIfScalar(exactInvScalar, targetTy, w);
            res = b.create<arith::MulFOp>(loc, lhs, invRhs).getResult();
          } else {
            res = isFloat ? b.create<arith::DivFOp>(loc, lhs, rhs).getResult()
                          : b.create<arith::DivSIOp>(loc, lhs, rhs).getResult();
          }
        }
        else if (opKind == "xor")
          res = b.create<arith::XOrIOp>(loc, lhs, rhs);
        else if (opKind == "and")
          res = b.create<arith::AndIOp>(loc, lhs, rhs);
        else if (opKind == "shl")
          res = b.create<arith::ShLIOp>(loc, lhs, rhs);
        else if (opKind == "shr")
          res = elemResTy.isUnsignedInteger(64)
                    ? b.create<arith::ShRUIOp>(loc, lhs, rhs).getResult()
                    : b.create<arith::ShRSIOp>(loc, lhs, rhs).getResult();
        else
          return op.emitError("unsupported Mojo row apply kind ") << opKind;
        map.map(op.getResult(0), res);
        continue;
      }
      return op.emitError("unrecognized Mojo sweep instruction ")
             << op.getName().getStringRef();
    }
    return success();
  }

  LogicalResult emitSweep(Operation &loop, Value r, IRMapping &rowMap,
                          llvm::DenseMap<int64_t, Value> &scanCarries) {
    auto range = loop.getAttrOfType<StringAttr>("range").getValue();
    Value c0 = constIdx(b, loc, 0), c1 = constIdx(b, loc, 1);

    if (range == "private_bins") {
      auto &countOp = loop.getRegion(0).front().front();
      auto &writeOp = *countOp.getNextNode();
      unsigned srcIdx = slotIndex(countOp.getOperand(0));
      unsigned dstIdx = slotIndex(writeOp.getOperand(0));
      int64_t span = countOp.getAttrOfType<IntegerAttr>("span").getInt();
      int64_t size = countOp.getAttrOfType<IntegerAttr>("extent").getInt();
      Value i32Zero = constI32(b, loc, 0), i32One = constI32(b, loc, 1);
      {
        auto *kLoop = makeOp(b, loc, "for_range", {c0, colsIdx, c1}, {}, {}, 1);
        Block &kBb = kLoop->getRegion(0).emplaceBlock();
        Value k = kBb.addArgument(b.getIndexType(), loc);
        OpBuilder::InsertionGuard kg(b);
        b.setInsertionPointToEnd(&kBb);
        Value dAddr = emitViewAddr(dstIdx, {r, k});
        makeOp(b, loc, "store_scalar",
               {slots[dstIdx].ptrVal, dAddr, i32Zero});
        makeOp(b, loc, "end");
      }
      Value mStart = b.create<arith::MulIOp>(loc, r, constIdx(b, loc, span));
      Value mEnd = b.create<arith::MinSIOp>(
          loc,
          b.create<arith::MulIOp>(loc, b.create<arith::AddIOp>(loc, r, c1),
                                  constIdx(b, loc, span)),
          constIdx(b, loc, size));
      {
        auto *mLoop =
            makeOp(b, loc, "for_range", {mStart, mEnd, c1}, {}, {}, 1);
        Block &mBb = mLoop->getRegion(0).emplaceBlock();
        Value m = mBb.addArgument(b.getIndexType(), loc);
        OpBuilder::InsertionGuard mg(b);
        b.setInsertionPointToEnd(&mBb);
        Value sAddr = emitViewAddr(srcIdx, {m});
        Value kRaw = makeOp(b, loc, "load_scalar",
                            {slots[srcIdx].ptrVal, sAddr},
                            {slots[srcIdx].elemTy})
                         ->getResult(0);
        Value kVal = slots[srcIdx].elemTy.isF32()
                         ? makeOp(b, loc, "scalar_cast", {kRaw},
                                  {b.getI32Type()},
                                  {b.getNamedAttr("dtype",
                                                  b.getStringAttr("int32"))})
                               ->getResult(0)
                         : kRaw;
        Value kIdx = asInt(b, loc, kVal);
        Value inLo =
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, kIdx, c0);
        Value inHi = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                             kIdx, colsIdx);
        auto ifIn = b.create<scf::IfOp>(
            loc, TypeRange{}, b.create<arith::AndIOp>(loc, inLo, inHi), false);
        b.setInsertionPointToStart(&ifIn.getThenRegion().front());
        Value dAddr = emitViewAddr(dstIdx, {r, kIdx});
        Value cur = makeOp(b, loc, "load_scalar",
                           {slots[dstIdx].ptrVal, dAddr}, {b.getI32Type()})
                        ->getResult(0);
        makeOp(b, loc, "store_scalar",
               {slots[dstIdx].ptrVal, dAddr,
                b.create<arith::AddIOp>(loc, cur, i32One)});
        b.setInsertionPointToEnd(&mBb);
        makeOp(b, loc, "end");
      }
      return success();
    }

    if (range == "indexed_copy") {
      auto &readOp = loop.getRegion(0).front().front();
      auto &writeOp = *readOp.getNextNode();
      unsigned srcIdx = slotIndex(readOp.getOperand(0));
      unsigned idIdx = slotIndex(readOp.getOperand(1));
      unsigned dstIdx = slotIndex(writeOp.getOperand(0));
      Type elemTy = slots[srcIdx].elemTy;
      int64_t elemBits = elemTy.getIntOrFloatBitWidth();
      int64_t copyVecW = std::max<int64_t>(1, (fullWidth * 32) / elemBits);
      Value idAddr = emitViewAddr(idIdx, {r});
      Value idRaw = makeOp(b, loc, "load_scalar",
                           {slots[idIdx].ptrVal, idAddr},
                           {slots[idIdx].elemTy})
                        ->getResult(0);
      Value rowId = asInt(b, loc, idRaw);
      Value srcBase = emitViewAddr(srcIdx, {rowId, c0});
      Value dstBase = emitViewAddr(dstIdx, {r, c0});
      Value contig = b.create<arith::AndIOp>(
          loc,
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                  slots[srcIdx].strideVals[1], c1),
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                  slots[dstIdx].strideVals[1], c1));
      auto ifContig = b.create<scf::IfOp>(loc, TypeRange{}, contig, true);
      {
        OpBuilder::InsertionGuard tg(b);
        b.setInsertionPointToStart(&ifContig.getThenRegion().front());
        Value cVecW = constIdx(b, loc, copyVecW);
        auto vecTy = VectorType::get({copyVecW}, elemTy);
        auto *wVec =
            makeOp(b, loc, "while_lt", {c0, colsIdx, cVecW}, {},
                   {b.getNamedAttr("check_step", b.getBoolAttr(true))}, 1);
        Block &wvBb = wVec->getRegion(0).emplaceBlock();
        Value jVec = wvBb.addArgument(b.getIndexType(), loc);
        {
          OpBuilder::InsertionGuard wg(b);
          b.setInsertionPointToEnd(&wvBb);
          Value sOff = b.create<arith::AddIOp>(loc, srcBase, jVec);
          Value dOff = b.create<arith::AddIOp>(loc, dstBase, jVec);
          Value sAddr = makeStaticAddr(b, loc, slots[srcIdx].ptrVal, {sOff}, {1});
          Value dAddr = makeStaticAddr(b, loc, slots[dstIdx].ptrVal, {dOff}, {1});
          Value chunk =
              makeOp(b, loc, "load_slice", {slots[srcIdx].ptrVal, sAddr},
                     {vecTy},
                     {b.getNamedAttr("width", b.getI64IntegerAttr(copyVecW))})
                  ->getResult(0);
          makeOp(b, loc, "store_slice", {slots[dstIdx].ptrVal, dAddr, chunk});
          makeOp(b, loc, "end");
        }
        Value vecDone = b.create<arith::MulIOp>(
            loc, b.create<arith::DivSIOp>(loc, colsIdx, cVecW), cVecW);
        auto *wSca =
            makeOp(b, loc, "while_lt", {vecDone, colsIdx, c1}, {},
                   {b.getNamedAttr("check_step", b.getBoolAttr(false))}, 1);
        Block &wsBb = wSca->getRegion(0).emplaceBlock();
        Value jSca = wsBb.addArgument(b.getIndexType(), loc);
        {
          OpBuilder::InsertionGuard sg(b);
          b.setInsertionPointToEnd(&wsBb);
          Value sOff = b.create<arith::AddIOp>(loc, srcBase, jSca);
          Value dOff = b.create<arith::AddIOp>(loc, dstBase, jSca);
          Value sAddr = makeStaticAddr(b, loc, slots[srcIdx].ptrVal, {sOff}, {1});
          Value dAddr = makeStaticAddr(b, loc, slots[dstIdx].ptrVal, {dOff}, {1});
          Value elem = makeOp(b, loc, "load_scalar",
                              {slots[srcIdx].ptrVal, sAddr}, {elemTy})
                           ->getResult(0);
          makeOp(b, loc, "store_scalar", {slots[dstIdx].ptrVal, dAddr, elem});
          makeOp(b, loc, "end");
        }
      }
      {
        OpBuilder::InsertionGuard eg(b);
        b.setInsertionPointToStart(&ifContig.getElseRegion().front());
        auto *jLoop =
            makeOp(b, loc, "for_range", {c0, colsIdx, c1}, {}, {}, 1);
        Block &jBb = jLoop->getRegion(0).emplaceBlock();
        Value j = jBb.addArgument(b.getIndexType(), loc);
        b.setInsertionPointToEnd(&jBb);
        Value sAddr = emitViewAddr(srcIdx, {rowId, j});
        Value val = makeOp(b, loc, "load_scalar",
                           {slots[srcIdx].ptrVal, sAddr},
                           {slots[srcIdx].elemTy})
                        ->getResult(0);
        Value dAddr = emitViewAddr(dstIdx, {r, j});
        makeOp(b, loc, "store_scalar", {slots[dstIdx].ptrVal, dAddr, val});
        makeOp(b, loc, "end");
      }
      return success();
    }

    if (range == "copy") {
      auto &readOp = loop.getRegion(0).front().front();
      auto &writeOp = *readOp.getNextNode();
      unsigned srcIdx = slotIndex(readOp.getOperand(0));
      unsigned dstIdx = slotIndex(writeOp.getOperand(0));
      bool srcSwap = readOp.hasAttr("axes");
      bool dstSwap = writeOp.hasAttr("axes");
      auto *jLoop = makeOp(b, loc, "for_range", {c0, colsIdx, c1}, {}, {}, 1);
      Block &jBb = jLoop->getRegion(0).emplaceBlock();
      Value j = jBb.addArgument(b.getIndexType(), loc);
      OpBuilder::InsertionGuard jg(b);
      b.setInsertionPointToEnd(&jBb);
      Value sAddr = emitViewAddr(srcIdx, buildStandardCoords(srcIdx, r, j, srcSwap));
      Value val = makeOp(b, loc, "load_scalar",
                         {slots[srcIdx].ptrVal, sAddr}, {slots[srcIdx].elemTy})
                      ->getResult(0);
      Value dAddr = emitViewAddr(dstIdx, buildStandardCoords(dstIdx, r, j, dstSwap));
      makeOp(b, loc, "store_scalar", {slots[dstIdx].ptrVal, dAddr, val});
      makeOp(b, loc, "end");
      return success();
    }

    auto alg = loop.getAttrOfType<StringAttr>("algebra").getValue();
    bool isTail = range == "tail";
    bool isOwners = range == "owners" || range == "owners_vec";
    bool isStore = alg == "store";
    bool isSum = alg == "sum";
    bool isMoments = alg == "welford";
    int64_t w = loop.getAttrOfType<IntegerAttr>("width").getInt();

    Value fullStop = b.create<arith::MulIOp>(
        loc,
        b.create<arith::DivSIOp>(loc, colsIdx, constIdx(b, loc, fullWidth)),
        constIdx(b, loc, fullWidth));

    if (isOwners) {
      SmallVector<Value, 1> dummyVals;
      IRMapping bodyMap = rowMap;
      bodyMap.map(loop.getRegion(0).front().getArgument(0), c0);
      return lowerSweepBody(loop.getRegion(0).front(), r, c0, w, bodyMap,
                            scanCarries, dummyVals);
    }

    // Declare reduction accumulators
    unsigned numReds = loop.getNumResults();
    SmallVector<Value, 4> accSlots;
    Value welfordNSlot, welfordMSlot, welfordS2Slot, welfordTupleSlot;
    auto vecF32Ty = VectorType::get({w}, b.getF32Type());
    if (isMoments) {
      if (!isTail) {
        welfordNSlot = declScalarSlot(b.getI32Type(), constI32(b, loc, 0));
        welfordMSlot = declVecSlot(vecF32Ty, w);
        welfordS2Slot = declVecSlot(vecF32Ty, w);
      } else {
        Value prevTuple = rowMap.lookup(loop.getOperand(0));
        welfordTupleSlot =
            makeOp(b, loc, "tuple_decl", {prevTuple}, {prevTuple.getType()})
                ->getResult(0);
      }
    } else if (!isStore) {
      if (!isTail) {
        Value initVal;
        if (!isSum) {
          Value neginf = constF32(b, loc, -INFINITY);
          initVal =
              makeOp(b, loc, "splat_scalar", {neginf}, {vecF32Ty},
                     {b.getNamedAttr("dtype", b.getStringAttr("float32")),
                      b.getNamedAttr("width", b.getI64IntegerAttr(w))})
                  ->getResult(0);
        }
        for (unsigned k = 0; k < numReds; ++k)
          accSlots.push_back(declVecSlot(vecF32Ty, w, initVal));
      } else {
        for (unsigned k = 0; k < numReds; ++k) {
          Value prevVal = rowMap.lookup(loop.getOperand(k));
          accSlots.push_back(declScalarSlot(b.getF32Type(), prevVal));
        }
      }
    }

    // Hoist row-invariant owner reads, static stride-0 broadcast reads, and
    // pure invariant expressions out of the inner column loop.
    IRMapping hoistedRowMap = rowMap;
    sweepSplatCache.clear();
    bool hasSweepUniform = false;
    {
      llvm::DenseSet<unsigned> writtenSlots;
      for (Operation &sop : loop.getRegion(0).front()) {
        auto sc = sop.getName().stripDialect();
        if (sc == "write" || sc == "owner_write")
          writtenSlots.insert(slotIndex(sop.getOperand(0)));
        if (sc == "apply") {
          auto k = sop.getAttrOfType<StringAttr>("kind");
          if (k && k.getValue() == "uniform")
            hasSweepUniform = true;
        }
      }
      sweepHoistBlock = b.getInsertionBlock();
      sweepHoistBefore = nullptr;
      for (Operation &sop : loop.getRegion(0).front()) {
        auto sc = sop.getName().stripDialect();
        if (sc == "owner_read") {
          unsigned sIdx = slotIndex(sop.getOperand(0));
          if (writtenSlots.contains(sIdx))
            continue;
          Value addr = emitViewAddr(sIdx, {r});
          Value val = makeOp(b, loc, "load_scalar",
                             {slots[sIdx].ptrVal, addr}, {slots[sIdx].elemTy})
                          ->getResult(0);
          hoistedRowMap.map(sop.getResult(0), val);
        } else if (sc == "segment_size" && !ownerAxis &&
                   slots[task.getAttrOfType<IntegerAttr>("domain_arg").getInt()]
                           .rank > 1) {
          unsigned offIdx = slotIndex(sop.getOperand(0));
          if (writtenSlots.contains(offIdx))
            continue;
          auto loadOffH = [&](Value idxVal) {
            Value a = emitViewAddr(offIdx, {idxVal});
            Value raw = makeOp(b, loc, "load_scalar",
                               {slots[offIdx].ptrVal, a}, {b.getI32Type()})
                            ->getResult(0);
            return asInt(b, loc, raw);
          };
          Value diff = b.create<arith::SubIOp>(
              loc, loadOffH(b.create<arith::AddIOp>(loc, r, c1)), loadOffH(r));
          Value scLen =
              makeOp(b, loc, "scalar_cast", {diff}, {b.getF32Type()},
                     {b.getNamedAttr("dtype", b.getStringAttr("float32"))})
                  ->getResult(0);
          Value val =
              w == 1
                  ? scLen
                  : makeOp(b, loc, "splat_scalar", {scLen},
                           {VectorType::get({w}, b.getF32Type())},
                           {b.getNamedAttr("dtype", b.getStringAttr("float32")),
                            b.getNamedAttr("width", b.getI64IntegerAttr(w))})
                        ->getResult(0);
          hoistedRowMap.map(sop.getResult(0), val);
        } else if (sc == "read" && !ownerAxis) {
          unsigned sIdx = slotIndex(sop.getOperand(0));
          if (writtenSlots.contains(sIdx) || !slots[sIdx].boundStrides ||
              slots[sIdx].rank < 1)
            continue;
          bool swap = sop.hasAttr("axes");
          int64_t fAx =
              slots[sIdx].rank == 2 && swap ? 0 : int64_t(slots[sIdx].rank) - 1;
          if (slots[sIdx].boundStrides[fAx] != 0)
            continue;
          Value addr = emitViewAddr(sIdx, buildStandardCoords(sIdx, r, c0, swap));
          Value scVal = makeOp(b, loc, "load_scalar",
                               {slots[sIdx].ptrVal, addr}, {slots[sIdx].elemTy})
                            ->getResult(0);
          Value val =
              w == 1
                  ? scVal
                  : makeOp(b, loc, "splat_scalar", {scVal},
                           {VectorType::get({w}, slots[sIdx].elemTy)},
                           {b.getNamedAttr("dtype",
                                           b.getStringAttr(
                                               dtypeBare(slots[sIdx].elemTy))),
                            b.getNamedAttr("width", b.getI64IntegerAttr(w))})
                        ->getResult(0);
          hoistedRowMap.map(sop.getResult(0), val);
        } else if (sc == "apply") {
          auto opKind = sop.getAttrOfType<StringAttr>("kind").getValue();
          if (opKind != "constant" && opKind != "field" && opKind != "cast" &&
              opKind != "sqrt" && opKind != "exp" && opKind != "rsqrt" &&
              opKind != "add" && opKind != "sub" && opKind != "mul" &&
              opKind != "div")
            continue;
          Type origResTy = sop.getResult(0).getType();
          Type elemResTy = isa<ShapedType>(origResTy)
                               ? cast<ShapedType>(origResTy).getElementType()
                               : origResTy;
          if (opKind == "div" && !isa<FloatType>(elemResTy))
            continue;
          if (!llvm::all_of(sop.getOperands(),
                            [&](Value in) { return hoistedRowMap.contains(in); }))
            continue;
          Type targetTy = w == 1 && isa<ShapedType>(origResTy) ? elemResTy
                                                                : origResTy;
          Value hRes;
          if (opKind == "constant") {
            hRes = makeOp(b, loc, "scalar_const", {}, {elemResTy},
                          {b.getNamedAttr("dtype",
                                          b.getStringAttr(dtypeBare(elemResTy))),
                           b.getNamedAttr("literal",
                                          b.getStringAttr(intentdsl::kir::literal(
                                              sop.getAttrOfType<TypedAttr>(
                                                  "value"))))})
                       ->getResult(0);
          } else if (opKind == "field") {
            int64_t pos = sop.getAttrOfType<IntegerAttr>("position").getInt();
            hRes = makeOp(b, loc, "tuple_field",
                          {hoistedRowMap.lookup(sop.getOperand(0))},
                          {elemResTy},
                          {b.getNamedAttr("position", b.getI64IntegerAttr(pos))})
                       ->getResult(0);
          } else if (opKind == "cast") {
            Value inVal = hoistedRowMap.lookup(sop.getOperand(0));
            Type outTy = isa<ShapedType>(inVal.getType()) ? targetTy : elemResTy;
            hRes = makeOp(b, loc, "scalar_cast", {inVal}, {outTy},
                          {b.getNamedAttr("dtype",
                                          b.getStringAttr(dtypeBare(elemResTy)))})
                       ->getResult(0);
          } else if (opKind == "sqrt" || opKind == "exp" || opKind == "rsqrt") {
            Value inVal =
                splatIfScalar(hoistedRowMap.lookup(sop.getOperand(0)), targetTy, w);
            hRes = opKind == "sqrt"
                       ? b.create<math::SqrtOp>(loc, inVal).getResult()
                       : (opKind == "exp"
                              ? b.create<math::ExpOp>(loc, inVal).getResult()
                              : b.create<math::RsqrtOp>(loc, inVal).getResult());
          } else {
            Value lhs =
                splatIfScalar(hoistedRowMap.lookup(sop.getOperand(0)), targetTy, w);
            Value rhs =
                splatIfScalar(hoistedRowMap.lookup(sop.getOperand(1)), targetTy, w);
            bool isFl = isa<FloatType>(elemResTy);
            if (opKind == "add")
              hRes = isFl ? b.create<arith::AddFOp>(loc, lhs, rhs).getResult()
                          : b.create<arith::AddIOp>(loc, lhs, rhs).getResult();
            else if (opKind == "sub")
              hRes = isFl ? b.create<arith::SubFOp>(loc, lhs, rhs).getResult()
                          : b.create<arith::SubIOp>(loc, lhs, rhs).getResult();
            else if (opKind == "mul")
              hRes = isFl ? b.create<arith::MulFOp>(loc, lhs, rhs).getResult()
                          : b.create<arith::MulIOp>(loc, lhs, rhs).getResult();
            else
              hRes = b.create<arith::DivFOp>(loc, lhs, rhs).getResult();
          }
          hoistedRowMap.map(sop.getResult(0), hRes);
        }
      }
    }

    auto emitInnerColStepTo =
        [&](Value j, ArrayRef<Value> dstAccs,
            SmallVectorImpl<std::function<void()>> *deferredStores = nullptr,
            bool deferViewWrites = false) -> LogicalResult {
      SmallVector<Value, 16> yielded;
      IRMapping bodyMap = hoistedRowMap;
      bodyMap.map(loop.getRegion(0).front().getArgument(0), j);
      if (failed(lowerSweepBody(loop.getRegion(0).front(), r, j, w, bodyMap,
                                scanCarries, yielded, deferredStores,
                                deferViewWrites)))
        return failure();
      if (isStore)
        return success();
      if (isMoments) {
        if (!isTail) {
          Value nextN = b.create<arith::AddIOp>(
              loc, readScalarSlot(welfordNSlot), constI32(b, loc, 1));
          writeScalarSlot(welfordNSlot, nextN);
          auto *stepOp = makeOp(
              b, loc, "welford_vec_step",
              {yielded[0], readVecSlot(welfordMSlot),
               readVecSlot(welfordS2Slot), nextN},
              {vecF32Ty, vecF32Ty},
              {b.getNamedAttr("width", b.getI64IntegerAttr(w)),
               b.getNamedAttr("ci.implementation",
                              b.getStringAttr("exits/mojo::welford-step"))});
          writeVecSlot(welfordMSlot, stepOp->getResult(0));
          writeVecSlot(welfordS2Slot, stepOp->getResult(1));
        } else {
          Value curT =
              makeOp(b, loc, "tuple_read", {welfordTupleSlot},
                     {welfordTupleSlot.getType()})
                  ->getResult(0);
          Value nextT =
              makeOp(b, loc, "welford_scalar_step", {curT, yielded[0]},
                     {welfordTupleSlot.getType()},
                     {b.getNamedAttr(
                         "ci.implementation",
                         b.getStringAttr("exits/mojo::welford-combine"))})
                  ->getResult(0);
          makeOp(b, loc, "tuple_write", {welfordTupleSlot, nextT});
        }
      } else if (!isTail) {
        for (unsigned k = 0; k < numReds; ++k) {
          Value cur = readVecSlot(dstAccs[k]);
          Value next =
              isSum
                  ? b.create<arith::AddFOp>(loc, cur, yielded[k]).getResult()
                  : b.create<arith::MaxNumFOp>(loc, cur, yielded[k]).getResult();
          writeVecSlot(dstAccs[k], next);
        }
      } else {
        for (unsigned k = 0; k < numReds; ++k) {
          Value cur = readScalarSlot(dstAccs[k]);
          Value next =
              isSum
                  ? b.create<arith::AddFOp>(loc, cur, yielded[k]).getResult()
                  : b.create<arith::MaxNumFOp>(loc, cur, yielded[k]).getResult();
          writeVecSlot(dstAccs[k], next);
        }
      }
      return success();
    };
    auto emitInnerColStep = [&](Value j) {
      return emitInnerColStepTo(j, accSlots);
    };

    if (!isTail) {
      Value cW = constIdx(b, loc, w);
      sweepHoistBefore = cW.getDefiningOp();
      Value initVal;
      if (!isStore && !isMoments && !isSum) {
        Value neginf = constF32(b, loc, -INFINITY);
        initVal =
            makeOp(b, loc, "splat_scalar", {neginf}, {vecF32Ty},
                   {b.getNamedAttr("dtype", b.getStringAttr("float32")),
                    b.getNamedAttr("width", b.getI64IntegerAttr(w))})
                ->getResult(0);
      }
      auto estimateUnroll4VecRegisters =
          [&](bool deferCrossLaneStores) -> int64_t {
        Block &sweepBb = loop.getRegion(0).front();
        llvm::DenseSet<Value> invariantVecInputs;
        llvm::DenseSet<Operation *> constApplyOps;
        llvm::DenseMap<Value, int> remUses;
        bool usesExpPoly = false;
        for (Operation &sop : sweepBb) {
          if (sop.getNumResults() == 1 &&
              hoistedRowMap.contains(sop.getResult(0)))
            continue;
          auto sc = sop.getName().stripDialect();
          if (sc == "apply") {
            auto k = sop.getAttrOfType<StringAttr>("kind").getValue();
            if (k == "constant") {
              constApplyOps.insert(&sop);
              continue;
            }
            if (k == "exp")
              usesExpPoly = true;
          }
          if (sc != "yield" && sop.getNumResults() == 1) {
            int uses = 0;
            for (Operation *u : sop.getResult(0).getUsers())
              if (u->getBlock() == &sweepBb)
                ++uses;
            remUses[sop.getResult(0)] = uses;
          }
          for (Value in : sop.getOperands()) {
            if (in == sweepBb.getArgument(0))
              continue;
            if (hoistedRowMap.contains(in)) {
              invariantVecInputs.insert(in);
            } else if (Operation *inDef = in.getDefiningOp();
                       inDef && constApplyOps.contains(inDef)) {
              invariantVecInputs.insert(in);
            }
          }
        }
        int64_t hoistedVecs = invariantVecInputs.size();
        int64_t sharedPolyRegs = usesExpPoly ? 6 : 0;
        int64_t curLive = 0, lanePeak = 0, deferredPerLane = 0;
        for (Operation &sop : sweepBb) {
          if (sop.getNumResults() == 1 &&
              (hoistedRowMap.contains(sop.getResult(0)) ||
               constApplyOps.contains(&sop)))
            continue;
          auto sc = sop.getName().stripDialect();
          int64_t opScratch = 0;
          if (sc == "apply") {
            auto k = sop.getAttrOfType<StringAttr>("kind").getValue();
            if (k == "exp")
              opScratch = 2;
            else if (k == "sqrt" || k == "rsqrt" || k == "div")
              opScratch = 1;
          }
          if (sc != "yield" && sop.getNumResults() == 1 &&
              remUses.lookup(sop.getResult(0)) > 0)
            ++curLive;
          lanePeak = std::max(lanePeak, curLive + opScratch);
          bool isDeferredSink =
              deferCrossLaneStores &&
              ((isStore && sc == "write") ||
               (!isStore && sc == "row_cache_store"));
          if (isDeferredSink)
            ++deferredPerLane;
          for (Value in : sop.getOperands()) {
            auto it = remUses.find(in);
            if (it != remUses.end() && --it->second == 0)
              --curLive;
          }
        }
        int64_t overlapLanes = deferCrossLaneStores ? 4 : 2;
        return hoistedVecs + sharedPolyRegs + 4 * int64_t(numReds) +
               3 * deferredPerLane + overlapLanes * std::max<int64_t>(1, lanePeak);
      };
      const int64_t archVecRegs = fullWidth >= 16 ? 32 : 16;
      int64_t chosenUnroll = 4;
      if (auto uAttr = task.getAttrOfType<IntegerAttr>("cpu.unroll"))
        chosenUnroll = uAttr.getInt();
      auto emitBlockedFullLoop = [&](bool unitFast,
                                     bool allowUnroll4) -> LogicalResult {
        allowUnroll4 = allowUnroll4 && (chosenUnroll >= 4);
        bool prevFast = forceUnitStride;
        forceUnitStride = unitFast;
        if (allowUnroll4 && w > 1 && isStore && !hasSweepUniform && !ownerAxis) {
          Operation *singleFeatSeg = nullptr;
          int segOps = 0, statefulOps = 0, userReads = 0, userWrites = 0;
          bool hasCacheLoad = false;
          for (Operation &sop : loop.getRegion(0).front()) {
            auto sc = sop.getName().stripDialect();
            if (sc == "segment_fold") {
              ++segOps;
              ++userReads;
              if (sop.hasAttr("feature_vector"))
                singleFeatSeg = &sop;
            } else if (sc == "segment_size" || sc == "indexed_read" ||
                       sc == "owner_write" || sc == "key_counts" ||
                       sc == "sum_columns" || sc == "row_cache_store") {
              ++statefulOps;
              ++userReads;
            } else if (sc == "read" || sc == "owner_read") {
              ++userReads;
            } else if (sc == "write") {
              ++userWrites;
            } else if (sc == "row_cache_load") {
              hasCacheLoad = true;
            } else if (sc == "apply" &&
                       sop.getAttrOfType<StringAttr>("kind").getValue() ==
                           "scan_sum") {
              ++statefulOps;
            }
          }
          if (segOps == 0 && statefulOps == 0 &&
              estimateUnroll4VecRegisters(false) <= archVecRegs) {
            bool deferOut =
                hasCacheLoad && userReads == 0 && userWrites == 1 &&
                estimateUnroll4VecRegisters(true) <= archVecRegs;
            Value cW2 = constIdx(b, loc, 2 * w), cW3 = constIdx(b, loc, 3 * w),
                  cW4 = constIdx(b, loc, 4 * w);
            Value unrollStop =
                llvm::isPowerOf2_64(uint64_t(4 * w))
                    ? b.create<arith::AndIOp>(loc, fullStop,
                                              constIdx(b, loc, -(4 * w)))
                          .getResult()
                    : b.create<arith::MulIOp>(
                           loc, b.create<arith::DivSIOp>(loc, fullStop, cW4),
                           cW4)
                          .getResult();
            {
              auto *uLoop =
                  makeOp(b, loc, "for_range", {c0, unrollStop, cW4}, {}, {}, 1);
              Block &uBb = uLoop->getRegion(0).emplaceBlock();
              Value j0 = uBb.addArgument(b.getIndexType(), loc);
              OpBuilder::InsertionGuard ug(b);
              b.setInsertionPointToEnd(&uBb);
              SmallVector<std::function<void()>, 4> deferred;
              auto *defPtr = deferOut ? &deferred : nullptr;
              if (failed(emitInnerColStepTo(j0, accSlots, defPtr, deferOut)) ||
                  failed(emitInnerColStepTo(
                      b.create<arith::AddIOp>(loc, j0, cW), accSlots, defPtr,
                      deferOut)) ||
                  failed(emitInnerColStepTo(
                      b.create<arith::AddIOp>(loc, j0, cW2), accSlots, defPtr,
                      deferOut)) ||
                  failed(emitInnerColStepTo(
                      b.create<arith::AddIOp>(loc, j0, cW3), accSlots, defPtr,
                      deferOut))) {
                forceUnitStride = prevFast;
                return failure();
              }
              for (auto &fn : deferred)
                fn();
              makeOp(b, loc, "end");
            }
            {
              auto *rLoop = makeOp(b, loc, "for_range",
                                   {unrollStop, fullStop, cW}, {}, {}, 1);
              Block &rBb = rLoop->getRegion(0).emplaceBlock();
              Value j = rBb.addArgument(b.getIndexType(), loc);
              OpBuilder::InsertionGuard rg(b);
              b.setInsertionPointToEnd(&rBb);
              if (failed(emitInnerColStep(j))) {
                forceUnitStride = prevFast;
                return failure();
              }
              makeOp(b, loc, "end");
            }
            forceUnitStride = prevFast;
            return success();
          }
          if (segOps == 1 && singleFeatSeg && statefulOps == 0) {
            auto &terms = singleFeatSeg->getRegion(0).front();
            Operation *ldOp = terms.back().getOperand(0).getDefiningOp();
            if (ldOp->getName().stripDialect() == "cast")
              ldOp = ldOp->getOperand(0).getDefiningOp();
            unsigned fIdx = slotIndex(ldOp->getOperand(0));
            if (unitFast ||
                (slots[fIdx].boundStrides && slots[fIdx].boundStrides[1] == 1)) {
              unsigned offIdx = slotIndex(singleFeatSeg->getOperand(0));
              Type fElem = slots[fIdx].elemTy;
              int64_t knownCols = -1;
              if (auto kc = task.getAttrOfType<IntegerAttr>("mojo_row.known_cols"))
                knownCols = kc.getInt();
              int segK = (knownCols >= 8 * w && knownCols % (8 * w) == 0 &&
                          knownCols <= 16 * w)
                             ? 8
                             : 4;
              Value cStep = constIdx(b, loc, segK * w);
              Value unrollStop = b.create<arith::MulIOp>(
                  loc, b.create<arith::DivSIOp>(loc, fullStop, cStep),
                  cStep);
              auto loadOffF = [&](Value idxVal) {
                Value a = emitViewAddr(offIdx, {idxVal});
                Value raw = makeOp(b, loc, "load_scalar",
                                   {slots[offIdx].ptrVal, a}, {b.getI32Type()})
                                ->getResult(0);
                return asInt(b, loc, raw);
              };
              Value start = loadOffF(r);
              Value end = loadOffF(b.create<arith::AddIOp>(loc, r, c1));
              {
                auto *uLoop =
                    makeOp(b, loc, "for_range", {c0, unrollStop, cStep}, {}, {}, 1);
                Block &uBb = uLoop->getRegion(0).emplaceBlock();
                Value j0 = uBb.addArgument(b.getIndexType(), loc);
                OpBuilder::InsertionGuard ug(b);
                b.setInsertionPointToEnd(&uBb);
                SmallVector<Value, 8> segAccs(segK);
                for (int u = 0; u < segK; ++u)
                  segAccs[u] = declVecSlot(vecF32Ty, w);
                {
                  auto *mLoop =
                      makeOp(b, loc, "for_range", {start, end, c1}, {}, {}, 1);
                  Block &mBb = mLoop->getRegion(0).emplaceBlock();
                  Value member = mBb.addArgument(b.getIndexType(), loc);
                  OpBuilder::InsertionGuard mg(b);
                  b.setInsertionPointToEnd(&mBb);
                  Value baseAddr = emitViewAddr(fIdx, {member, j0});
                  auto rawVecTy = VectorType::get({w}, fElem);
                  for (int u = 0; u < segK; ++u) {
                    Value uAddr =
                        u == 0
                            ? baseAddr
                            : b.create<arith::AddIOp>(
                                   loc, baseAddr, constIdx(b, loc, u * w))
                                  .getResult();
                    Value raw =
                        makeOp(b, loc, "load_slice",
                               {slots[fIdx].ptrVal, uAddr}, {rawVecTy},
                               {b.getNamedAttr("width", b.getI64IntegerAttr(w))})
                            ->getResult(0);
                    Value fVec =
                        fElem.isF32()
                            ? raw
                            : b.create<arith::ExtFOp>(loc, vecF32Ty, raw)
                                  .getResult();
                    writeVecSlot(
                        segAccs[u],
                        b.create<arith::AddFOp>(loc, readVecSlot(segAccs[u]),
                                                fVec));
                  }
                  makeOp(b, loc, "end");
                }
                for (int u = 0; u < segK; ++u) {
                  Value ju =
                      u == 0
                          ? j0
                          : b.create<arith::AddIOp>(loc, j0,
                                                    constIdx(b, loc, u * w))
                                .getResult();
                  SmallVector<Value, 4> dummy;
                  IRMapping bodyMap = hoistedRowMap;
                  bodyMap.map(loop.getRegion(0).front().getArgument(0), ju);
                  bodyMap.map(singleFeatSeg->getResult(0),
                              readVecSlot(segAccs[u]));
                  if (failed(lowerSweepBody(loop.getRegion(0).front(), r, ju, w,
                                            bodyMap, scanCarries, dummy))) {
                    forceUnitStride = prevFast;
                    return failure();
                  }
                }
                makeOp(b, loc, "end");
              }
              {
                auto *rLoop = makeOp(b, loc, "for_range",
                                     {unrollStop, fullStop, cW}, {}, {}, 1);
                Block &rBb = rLoop->getRegion(0).emplaceBlock();
                Value j = rBb.addArgument(b.getIndexType(), loc);
                OpBuilder::InsertionGuard rg(b);
                b.setInsertionPointToEnd(&rBb);
                if (failed(emitInnerColStep(j))) {
                  forceUnitStride = prevFast;
                  return failure();
                }
                makeOp(b, loc, "end");
              }
              forceUnitStride = prevFast;
              return success();
            }
          }
        }
        if (allowUnroll4 && w > 1 && !isStore && !isMoments && numReds >= 1 &&
            numReds <= 4 && !hasSweepUniform &&
            estimateUnroll4VecRegisters(false) <= archVecRegs) {
          bool hasCacheStore = llvm::any_of(
              loop.getRegion(0).front(), [](Operation &sop) {
                return sop.getName().stripDialect() == "row_cache_store";
              });
          bool deferCache =
              hasCacheStore &&
              estimateUnroll4VecRegisters(true) <= archVecRegs;
          SmallVector<SmallVector<Value, 4>, 3> extraAccs(3);
          for (int u = 0; u < 3; ++u)
            for (unsigned k = 0; k < numReds; ++k)
              extraAccs[u].push_back(declVecSlot(vecF32Ty, w, initVal));
          Value cW2 = constIdx(b, loc, 2 * w), cW3 = constIdx(b, loc, 3 * w),
                cW4 = constIdx(b, loc, 4 * w);
          Value unrollStop =
              llvm::isPowerOf2_64(uint64_t(4 * w))
                  ? b.create<arith::AndIOp>(loc, fullStop,
                                            constIdx(b, loc, -(4 * w)))
                        .getResult()
                  : b.create<arith::MulIOp>(
                         loc, b.create<arith::DivSIOp>(loc, fullStop, cW4), cW4)
                        .getResult();
          {
            auto *uLoop =
                makeOp(b, loc, "for_range", {c0, unrollStop, cW4}, {}, {}, 1);
            Block &uBb = uLoop->getRegion(0).emplaceBlock();
            Value j0 = uBb.addArgument(b.getIndexType(), loc);
            OpBuilder::InsertionGuard ug(b);
            b.setInsertionPointToEnd(&uBb);
            SmallVector<std::function<void()>, 4> deferred;
            auto *defPtr = deferCache ? &deferred : nullptr;
            if (failed(emitInnerColStepTo(j0, accSlots, defPtr, false)) ||
                failed(emitInnerColStepTo(
                    b.create<arith::AddIOp>(loc, j0, cW), extraAccs[0], defPtr,
                    false)) ||
                failed(emitInnerColStepTo(
                    b.create<arith::AddIOp>(loc, j0, cW2), extraAccs[1], defPtr,
                    false)) ||
                failed(emitInnerColStepTo(
                    b.create<arith::AddIOp>(loc, j0, cW3), extraAccs[2], defPtr,
                    false))) {
              forceUnitStride = prevFast;
              return failure();
            }
            for (auto &fn : deferred)
              fn();
            makeOp(b, loc, "end");
          }
          {
            auto *rLoop =
                makeOp(b, loc, "for_range", {unrollStop, fullStop, cW}, {}, {}, 1);
            Block &rBb = rLoop->getRegion(0).emplaceBlock();
            Value j = rBb.addArgument(b.getIndexType(), loc);
            OpBuilder::InsertionGuard rg(b);
            b.setInsertionPointToEnd(&rBb);
            if (failed(emitInnerColStepTo(j, accSlots))) {
              forceUnitStride = prevFast;
              return failure();
            }
            makeOp(b, loc, "end");
          }
          for (unsigned k = 0; k < numReds; ++k) {
            auto comb = [&](Value a, Value d) {
              return isSum ? b.create<arith::AddFOp>(loc, a, d).getResult()
                           : b.create<arith::MaxNumFOp>(loc, a, d).getResult();
            };
            Value s01 =
                comb(readVecSlot(accSlots[k]), readVecSlot(extraAccs[0][k]));
            Value s23 =
                comb(readVecSlot(extraAccs[1][k]), readVecSlot(extraAccs[2][k]));
            writeVecSlot(accSlots[k], comb(s01, s23));
          }
          forceUnitStride = prevFast;
          return success();
        }
        auto *jLoop =
            makeOp(b, loc, "for_range", {c0, fullStop, cW}, {}, {}, 1);
        Block &jBb = jLoop->getRegion(0).emplaceBlock();
        Value j = jBb.addArgument(b.getIndexType(), loc);
        {
          OpBuilder::InsertionGuard jg(b);
          b.setInsertionPointToEnd(&jBb);
          if (failed(emitInnerColStep(j))) {
            forceUnitStride = prevFast;
            return failure();
          }
          makeOp(b, loc, "end");
        }
        forceUnitStride = prevFast;
        return success();
      };

      SmallVector<std::pair<unsigned, int64_t>, 8> dynFastAxes;
      if (!ownerAxis) {
        for (Operation &sop : loop.getRegion(0).front()) {
          auto scode = sop.getName().stripDialect();
          if (scode == "segment_fold") {
            if (sop.hasAttr("feature_vector")) {
              auto &terms = sop.getRegion(0).front();
              Operation *ldOp = terms.back().getOperand(0).getDefiningOp();
              if (ldOp->getName().stripDialect() == "cast")
                ldOp = ldOp->getOperand(0).getDefiningOp();
              unsigned fIdx = slotIndex(ldOp->getOperand(0));
              if (!slots[fIdx].boundStrides && slots[fIdx].rank == 2) {
                std::pair<unsigned, int64_t> key{fIdx, 1};
                if (!llvm::is_contained(dynFastAxes, key))
                  dynFastAxes.push_back(key);
              }
            } else if (!sop.hasAttr("segment_size")) {
              unsigned oIdx = slotIndex(sop.getOperand(0));
              if (!slots[oIdx].boundStrides && slots[oIdx].rank == 1) {
                std::pair<unsigned, int64_t> key{oIdx, 0};
                if (!llvm::is_contained(dynFastAxes, key))
                  dynFastAxes.push_back(key);
              }
              for (Operation &v : sop.getRegion(0).front()) {
                if (v.getName().stripDialect() == "member_load") {
                  unsigned mIdx = slotIndex(v.getOperand(0));
                  if (!slots[mIdx].boundStrides && slots[mIdx].rank == 1) {
                    std::pair<unsigned, int64_t> key{mIdx, 0};
                    if (!llvm::is_contained(dynFastAxes, key))
                      dynFastAxes.push_back(key);
                  }
                }
              }
            }
            continue;
          }
          if (scode != "read" && scode != "write")
            continue;
          unsigned idx = slotIndex(sop.getOperand(0));
          if (slots[idx].boundStrides || slots[idx].rank < 1)
            continue;
          bool swap = sop.hasAttr("axes");
          int64_t fAx =
              slots[idx].rank == 2 && swap ? 0 : int64_t(slots[idx].rank) - 1;
          std::pair<unsigned, int64_t> key{idx, fAx};
          if (!llvm::is_contained(dynFastAxes, key))
            dynFastAxes.push_back(key);
        }
      }

      if (!dynFastAxes.empty()) {
        Value allUnit;
        for (auto [idx, fAx] : dynFastAxes) {
          Value eq1 = b.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::eq, slots[idx].strideVals[fAx], c1);
          allUnit = allUnit ? b.create<arith::AndIOp>(loc, allUnit, eq1).getResult()
                            : eq1;
        }
        auto ifUnit = b.create<scf::IfOp>(loc, TypeRange{}, allUnit, true);
        {
          OpBuilder::InsertionGuard tg(b);
          b.setInsertionPointToStart(&ifUnit.getThenRegion().front());
          if (failed(emitBlockedFullLoop(true, true))) {
            sweepHoistBlock = nullptr;
            sweepHoistBefore = nullptr;
            return failure();
          }
        }
        {
          OpBuilder::InsertionGuard eg(b);
          b.setInsertionPointToStart(&ifUnit.getElseRegion().front());
          if (failed(emitBlockedFullLoop(false, false))) {
            sweepHoistBlock = nullptr;
            sweepHoistBefore = nullptr;
            return failure();
          }
        }
      } else {
        if (failed(emitBlockedFullLoop(false, !ownerAxis))) {
          sweepHoistBlock = nullptr;
          sweepHoistBefore = nullptr;
          return failure();
        }
      }
    } else {
      sweepHoistBlock = nullptr;
      sweepHoistBefore = nullptr;
      auto *jLoop =
          makeOp(b, loc, "for_range", {fullStop, colsIdx, c1}, {}, {}, 1);
      Block &jBb = jLoop->getRegion(0).emplaceBlock();
      Value j = jBb.addArgument(b.getIndexType(), loc);
      {
        OpBuilder::InsertionGuard jg(b);
        b.setInsertionPointToEnd(&jBb);
        if (failed(emitInnerColStep(j))) {
          return failure();
        }
        makeOp(b, loc, "end");
      }
    }
    sweepHoistBlock = nullptr;
    sweepHoistBefore = nullptr;

    if (!isStore) {
      if (isMoments) {
        if (!isTail) {
          Value red = makeOp(
                          b, loc, "welford_vec_reduce",
                          {readScalarSlot(welfordNSlot),
                           readVecSlot(welfordMSlot),
                           readVecSlot(welfordS2Slot)},
                          {loop.getResult(0).getType()},
                          {b.getNamedAttr("width", b.getI64IntegerAttr(w)),
                           b.getNamedAttr(
                               "ci.implementation",
                               b.getStringAttr("exits/mojo::welford-reduce"))})
                          ->getResult(0);
          rowMap.map(loop.getResult(0), red);
        } else {
          Value finalT =
              makeOp(b, loc, "tuple_read", {welfordTupleSlot},
                     {welfordTupleSlot.getType()})
                  ->getResult(0);
          rowMap.map(loop.getResult(0), finalT);
        }
      } else if (!isTail) {
        for (unsigned k = 0; k < numReds; ++k) {
          Value vecVal = readVecSlot(accSlots[k]);
          Value red =
              isSum
                  ? makeOp(b, loc, "lane_sum", {vecVal}, {b.getF32Type()})
                        ->getResult(0)
                  : makeOp(b, loc, "lane_max", {vecVal}, {b.getF32Type()},
                           {b.getNamedAttr("width", b.getI64IntegerAttr(w))})
                        ->getResult(0);
          rowMap.map(loop.getResult(k), red);
        }
      } else {
        for (unsigned k = 0; k < numReds; ++k)
          rowMap.map(loop.getResult(k), readScalarSlot(accSlots[k]));
      }
    }
    return success();
  }
};
} // namespace

llvm::Error describeMojoRowPackage(func::FuncOp fn, Operation &prog,
                                   llvm::raw_pwrite_stream &out) {
  auto abi = llvm::json::parse(fn->getAttrOfType<StringAttr>("kir.abi"));
  if (!abi)
    return abi.takeError();
  llvm::json::Array rows;
  for (auto a : prog.getAttrOfType<ArrayAttr>("rows")) {
    auto row = cast<DictionaryAttr>(a);
    rows.push_back(llvm::json::Object{
        {"grain", row.getAs<IntegerAttr>("grain").getInt()},
        {"columns", row.getAs<IntegerAttr>("columns").getInt()}});
  }
  out << llvm::json::Value(llvm::json::Object{
      {"version", 2},
      {"abi", std::move(*abi)},
      {"domain", intentdsl::publishDomain(fn)},
      {"rows", std::move(rows)},
      {"returns_status", true},
      {"source", ""},
      {"host", ""}});
  return llvm::Error::success();
}

LogicalResult lowerMojoRow(func::FuncOp fn, int64_t fullWidth,
                           int64_t simdWidth) {
  auto &task = fn.front().front();
  auto &body = task.getRegion(0).front();
  Location loc = task.getLoc();
  OpBuilder b(&fn.front().front());

  bool isTileTranspose = !body.without_terminator().empty() &&
                         body.front().getName().getStringRef() ==
                             "cpu.transpose_tile";
  bool hasKeyBounds = task.hasAttr("cpu.key_extent");
  bool isReservations = task.hasAttr("cpu.batch_extent");
  bool usesAtomic = false, usesMemcpy = false, usesPhilox = false,
       usesMoments = false, usesScan = false, hasSegmentFold = false;
  task.walk([&](Operation *op) {
    auto n = op->getName().stripDialect();
    if (n == "reserve_tickets" ||
        (n == "atomic_members" && !task.hasAttr("cpu.single_dispatcher") &&
         !hasKeyBounds))
      usesAtomic = true;
    if (n == "segment_fold" && op->hasAttr("feature_vector"))
      hasSegmentFold = true;
    if (auto alg = op->getAttrOfType<StringAttr>("algebra");
        alg && alg.getValue() == "welford")
      usesMoments = true;
    if (auto k = op->getAttrOfType<StringAttr>("kind")) {
      if (k.getValue() == "uniform")
        usesPhilox = true;
      if (k.getValue() == "scan_sum")
        usesScan = true;
    }
  });
  auto rowCacheAttr = task.getAttrOfType<ArrayAttr>("mojo_row.row_cache");
  bool usesStack = isReservations || static_cast<bool>(rowCacheAttr);

  auto rowsAttr = task.getAttrOfType<ArrayAttr>("rows");
  OperationState progState(loc, "mojo_row.row_program");
  progState.addAttribute("rows", rowsAttr);
  progState.addRegion();
  progState.addRegion();
  Operation *prog = b.create(progState);

  // Region 0: row.mojo
  Block &kOuter = prog->getRegion(0).emplaceBlock();
  b.setInsertionPointToEnd(&kOuter);
  if (isTileTranspose)
    makeOp(b, loc, "runtime_source", {}, {},
           {b.getNamedAttr("implementation",
                           b.getStringAttr("exits/mojo::simd-transpose-8x8"))});
  if (usesPhilox)
    makeOp(b, loc, "runtime_source", {}, {},
           {b.getNamedAttr("implementation",
                           b.getStringAttr("exits/mojo::philox-uniform"))});
  if (usesMoments)
    makeOp(b, loc, "runtime_source", {}, {},
           {b.getNamedAttr("implementation",
                           b.getStringAttr("exits/mojo::welford-step"))});
  if (usesScan)
    makeOp(b, loc, "runtime_source", {}, {},
           {b.getNamedAttr("implementation",
                           b.getStringAttr("exits/mojo::simd-scan-sum"))});

  SmallVector<Attribute> paramSpecs;
  SmallVector<Type> blockArgTypes;
  for (auto [i, arg] : llvm::enumerate(fn.getArguments())) {
    Type t = arg.getType();
    if (auto m = dyn_cast<MemRefType>(t)) {
      paramSpecs.push_back(
          b.getStringAttr("Pointer[Scalar[DType." + dtypeBare(m.getElementType()) +
                          "], MutUntrackedOrigin]"));
      blockArgTypes.push_back(t);
      if (m.getRank() > 2) {
        for (int64_t j = 0; j < m.getRank() - 1; ++j) {
          paramSpecs.push_back(b.getStringAttr("Int64"));
          blockArgTypes.push_back(b.getI64Type());
        }
      }
      for (int64_t j = 0; j < m.getRank(); ++j) {
        paramSpecs.push_back(b.getStringAttr("Int64"));
        blockArgTypes.push_back(b.getI64Type());
      }
    } else {
      paramSpecs.push_back(b.getStringAttr(scalarMojoSpec(t)));
      blockArgTypes.push_back(t);
    }
  }
  for (int k = 0; k < (hasKeyBounds ? 6 : 4); ++k) {
    paramSpecs.push_back(b.getStringAttr("Int64"));
    blockArgTypes.push_back(b.getI64Type());
  }

  auto *rowFn =
      makeOp(b, loc, "row_fn", {}, {},
             {b.getNamedAttr("param_specs", b.getArrayAttr(paramSpecs)),
              b.getNamedAttr("uses_atomic", b.getBoolAttr(usesAtomic)),
              b.getNamedAttr("uses_stack", b.getBoolAttr(usesStack)),
              b.getNamedAttr("uses_memcpy", b.getBoolAttr(usesMemcpy)),
              b.getNamedAttr("has_key_bounds", b.getBoolAttr(hasKeyBounds))},
             1);
  makeOp(b, loc, "end");

  Block &fnBb = rowFn->getRegion(0).emplaceBlock();
  for (Type bt : blockArgTypes)
    fnBb.addArgument(bt, loc);
  b.setInsertionPointToEnd(&fnBb);

  RowLowerer lowerer{fn, task, b, loc, fullWidth, simdWidth};
  lowerer.projected = fn->hasAttr("kir.iteration") && !fn->hasAttr("kir.owner_axis");
  if (lowerer.projected) {
    if (auto m0 = dyn_cast<MemRefType>(fn.getArgument(0).getType());
        m0 && m0.getRank() > 2)
      lowerer.projectedRank = m0.getRank();
  }
  if (auto oa = fn->getAttrOfType<IntegerAttr>("kir.owner_axis"))
    lowerer.ownerAxis = oa.getInt();

  unsigned argCursor = 0;
  IRMapping baseMap;
  for (auto [i, origCap] : llvm::enumerate(body.getArguments().drop_front())) {
    ArgSlotInfo info;
    Type t = origCap.getType();
    if (auto m = dyn_cast<MemRefType>(t)) {
      info.isView = true;
      info.elemTy = m.getElementType();
      info.rank = m.getRank();
      info.ptrVal = fnBb.getArgument(argCursor++);
      info.boundShape = fn.getArgAttrOfType<DenseI64ArrayAttr>(i, "mojo_row.shape");
      info.boundStrides =
          fn.getArgAttrOfType<DenseI64ArrayAttr>(i, "mojo_row.strides");
      if (m.getRank() > 2) {
        for (int64_t j = 0; j < m.getRank() - 1; ++j) {
          Value rawDim = fnBb.getArgument(argCursor++);
          info.dimVals.push_back(info.boundShape
                                     ? constIdx(b, loc, info.boundShape[j])
                                     : asInt(b, loc, rawDim));
        }
      }
      for (int64_t j = 0; j < m.getRank(); ++j) {
        Value rawSt = fnBb.getArgument(argCursor++);
        info.strideVals.push_back(info.boundStrides
                                      ? constIdx(b, loc, info.boundStrides[j])
                                      : asInt(b, loc, rawSt));
      }
      baseMap.map(origCap, info.ptrVal);
    } else {
      Value scArg = fnBb.getArgument(argCursor++);
      baseMap.map(origCap, scArg);
    }
    lowerer.slots.push_back(std::move(info));
  }
  lowerer.firstIdx = asInt(b, loc, fnBb.getArgument(argCursor++));
  lowerer.stopIdx = asInt(b, loc, fnBb.getArgument(argCursor++));
  lowerer.colsIdx = asInt(b, loc, fnBb.getArgument(argCursor++));
  ++argCursor; // unused column panel parameter
  if (hasKeyBounds) {
    lowerer.keyLoIdx = asInt(b, loc, fnBb.getArgument(argCursor++));
    lowerer.keyHiIdx = asInt(b, loc, fnBb.getArgument(argCursor++));
  }
  lowerer.precomputeOwnerSizes();

  int64_t rowCacheCap = 8192;
  if (auto capAttr = task.getAttrOfType<IntegerAttr>("mojo_row.row_cache_cap"))
    rowCacheCap = capAttr.getInt();
  if (rowCacheAttr) {
    for (Attribute ta : rowCacheAttr) {
      Type et = cast<TypeAttr>(ta).getValue();
      lowerer.rowCacheElemTypes.push_back(et);
      lowerer.rowCacheBufs.push_back(
          makeOp(b, loc, "stack_alloc", {}, {b.getIndexType()},
                 {b.getNamedAttr("dtype", b.getStringAttr(dtypeBare(et))),
                  b.getNamedAttr("count", b.getI64IntegerAttr(rowCacheCap))})
              ->getResult(0));
    }
  }

  bool flatColParallel = false;
  if (!isTileTranspose && !isReservations && !hasKeyBounds &&
      !task.hasAttr("cpu.member_tasks") &&
      !task.hasAttr("cpu.single_dispatcher") && !lowerer.projectedRank &&
      !lowerer.ownerAxis && !rowCacheAttr) {
    auto walkAttrCheck = task.getAttrOfType<StringAttr>("cpu.walk");
    StringRef wCheck = walkAttrCheck ? walkAttrCheck.getValue() : "source";
    bool allViewsRank1 = true;
    for (Type argTy : fn.getArgumentTypes()) {
      if (auto m = dyn_cast<MemRefType>(argTy); m && m.getRank() != 1) {
        allViewsRank1 = false;
        break;
      }
    }
    bool pureStoreSweeps =
        wCheck == "source" && allViewsRank1 &&
        body.without_terminator().begin() != body.without_terminator().end();
    if (pureStoreSweeps) {
      for (Operation &step : body.without_terminator()) {
        if (step.getName().getStringRef() != "mojo_row.sweep") {
          pureStoreSweeps = false;
          break;
        }
        auto alg = step.getAttrOfType<StringAttr>("algebra");
        auto rng = step.getAttrOfType<StringAttr>("range");
        if (!alg || alg.getValue() != "store" || !rng ||
            (rng.getValue() != "full" && rng.getValue() != "tail")) {
          pureStoreSweeps = false;
          break;
        }
      }
    }
    if (pureStoreSweeps) {
      body.walk([&](Operation *sub) {
        auto sn = sub->getName().stripDialect();
        if (sn == "segment_fold" || sn == "segment_size" ||
            sn == "key_counts" || sn == "sum_columns" ||
            sn == "indexed_read" || sn == "owner_read" ||
            sn == "owner_write") {
          pureStoreSweeps = false;
          return;
        }
        if (sn == "apply") {
          auto k = sub->getAttrOfType<StringAttr>("kind");
          if (k && (k.getValue() == "ordinal" || k.getValue() == "scan_sum" ||
                    k.getValue() == "uniform"))
            pureStoreSweeps = false;
        }
      });
    }
    flatColParallel = pureStoreSweeps;
  }

  if (isTileTranspose) {
    lowerer.emitTransposeTile();
  } else if (isReservations) {
    if (failed(lowerer.emitReservations(body)))
      return failure();
  } else {
    auto *rLoop = makeOp(b, loc, "for_range",
                         {lowerer.firstIdx, lowerer.stopIdx, constIdx(b, loc, 1)},
                         {}, {}, 1);
    Block &rBb = rLoop->getRegion(0).emplaceBlock();
    Value r = rBb.addArgument(b.getIndexType(), loc);
    b.setInsertionPointToEnd(&rBb);
    IRMapping rowMap = baseMap;
    rowMap.map(body.getArgument(0), r);
    llvm::DenseMap<int64_t, Value> scanCarries;
    body.walk([&](Operation *sub) {
      if (sub->getName().getStringRef() != "mojo_row.apply")
        return;
      auto k = sub->getAttrOfType<StringAttr>("kind");
      if (!k || k.getValue() != "scan_sum")
        return;
      int64_t cId = sub->getAttrOfType<IntegerAttr>("carry").getInt();
      if (scanCarries.count(cId))
        return;
      Type origTy = sub->getResult(0).getType();
      Type elemTy = isa<ShapedType>(origTy)
                        ? cast<ShapedType>(origTy).getElementType()
                        : origTy;
      Value z =
          makeOp(b, loc, "scalar_const", {}, {elemTy},
                 {b.getNamedAttr("dtype", b.getStringAttr(dtypeBare(elemTy))),
                  b.getNamedAttr("literal", b.getStringAttr("0"))})
              ->getResult(0);
      scanCarries[cId] = lowerer.declScalarSlot(elemTy, z);
    });
    if (rowCacheAttr) {
      Value fitsCache = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sle, lowerer.colsIdx,
          constIdx(b, loc, rowCacheCap));
      auto ifCache = b.create<scf::IfOp>(loc, TypeRange{}, fitsCache, true);
      {
        OpBuilder::InsertionGuard tg(b);
        b.setInsertionPointToStart(&ifCache.getThenRegion().front());
        IRMapping cachedMap = rowMap;
        for (Operation &step : body.without_terminator()) {
          auto mode = step.getAttrOfType<StringAttr>("cache_mode");
          if (!mode || mode.getValue() != "cached")
            continue;
          if (failed(lowerer.emitSweep(step, r, cachedMap, scanCarries)))
            return failure();
        }
      }
      {
        OpBuilder::InsertionGuard eg(b);
        b.setInsertionPointToStart(&ifCache.getElseRegion().front());
        IRMapping uncachedMap = rowMap;
        for (Operation &step : body.without_terminator()) {
          auto mode = step.getAttrOfType<StringAttr>("cache_mode");
          if (!mode || mode.getValue() != "uncached")
            continue;
          if (failed(lowerer.emitSweep(step, r, uncachedMap, scanCarries)))
            return failure();
        }
      }
    } else {
      for (Operation &step : body.without_terminator()) {
        auto sn = step.getName().getStringRef();
        if (sn == "cpu.pack_members") {
          lowerer.emitPackMembers(step, r);
        } else if (step.getName().stripDialect() == "atomic_members") {
          if (failed(lowerer.emitAtomicMembers(step, r)))
            return failure();
        } else if (sn == "mojo_row.sweep") {
          if (failed(lowerer.emitSweep(step, r, rowMap, scanCarries)))
            return failure();
        } else {
          return step.emitError("unexpected row task step ") << sn;
        }
      }
    }
    b.setInsertionPointToEnd(&rBb);
    makeOp(b, loc, "end");
  }
  b.setInsertionPointToEnd(&fnBb);
  makeOp(b, loc, "end");

  // Region 1: entry.c (structured C host function + OpenMP dispatch + call)
  Block &hOuter = prog->getRegion(1).emplaceBlock();
  b.setInsertionPointToEnd(&hOuter);

  bool memberTasks = task.hasAttr("cpu.member_tasks");
  bool hasSeg = task.walk([](Operation *o) {
                  auto n = o->getName().stripDialect();
                  return (n == "ordered_sum" || n == "segment_fold" ||
                          n == "segment_size")
                             ? WalkResult::interrupt()
                             : WalkResult::advance();
                }).wasInterrupted();
  auto walkAttr = task.getAttrOfType<StringAttr>("cpu.walk");
  StringRef walk = walkAttr ? walkAttr.getValue() : "source";
  int64_t domSlot = task.getAttrOfType<IntegerAttr>("domain_arg").getInt();

  SmallVector<std::string> hostSlots, callNames;
  for (auto [i, type] : llvm::enumerate(fn.getArgumentTypes())) {
    std::string n = "a" + std::to_string(i);
    auto view = dyn_cast<MemRefType>(type);
    if (lowerer.projectedRank && view &&
        view.getRank() == int64_t(lowerer.projectedRank)) {
      std::string off = "0";
      for (unsigned ax = 0; ax + 1 < lowerer.projectedRank; ++ax)
        off += "+c" + std::to_string(ax) + "*" + n + "s" + std::to_string(ax);
      callNames.push_back("(char*)" + n + "+(" + off + ")*" +
                          std::to_string(view.getElementTypeBitWidth() / 8));
    } else if (memberTasks && view &&
               (hasSeg ? int64_t(i) == domSlot : view.getRank() == 1)) {
      callNames.push_back("(char*)" + n + "+start*" + n + "s0*" +
                          std::to_string(view.getElementTypeBitWidth() / 8));
    } else if (flatColParallel && view) {
      callNames.push_back("(char*)" + n + "+cstart*" + n + "s0*" +
                          std::to_string(view.getElementTypeBitWidth() / 8));
    } else {
      callNames.push_back(n);
    }
    hostSlots.push_back(std::string(view                         ? "void *"
                                    : type.isUnsignedInteger(64) ? "uint64_t "
                                    : type.isInteger(32)         ? "int32_t "
                                                                 : "float ") +
                        n);
    if (!view)
      continue;
    if (view.getRank() > 2) {
      for (int64_t j = 0; j < view.getRank() - 1; ++j) {
        std::string d = n + "d" + std::to_string(j);
        hostSlots.push_back("int64_t " + d);
        callNames.push_back(d);
      }
    }
    for (int64_t j = 0; j < view.getRank(); ++j) {
      std::string s = n + "s" + std::to_string(j);
      hostSlots.push_back("int64_t " + s);
      callNames.push_back(s);
    }
  }
  std::string baseParams = llvm::join(hostSlots, ",");
  std::string externDecl =
      "extern void intent_row_interval(" + baseParams +
      (baseParams.empty() ? "" : ",") + "int64_t,int64_t,int64_t,int64_t" +
      (hasKeyBounds ? ",int64_t,int64_t" : "") + ");";
  std::string fullHostParams =
      baseParams + (baseParams.empty() ? "" : ",") +
      "int64_t rows,int64_t cols,int64_t grain,int64_t columns,int64_t workers";
  std::string joinedCallArgs = llvm::join(callNames, ",");

  auto *cHost = makeOp(
      b, loc, "c_row_host_fn", {}, {},
      {b.getNamedAttr("params", b.getStringAttr(fullHostParams)),
       b.getNamedAttr("extern_decl", b.getStringAttr(externDecl))},
      1);
  makeOp(b, loc, "end");

  Block &cFnBb = cHost->getRegion(0).emplaceBlock();
  b.setInsertionPointToEnd(&cFnBb);
  if (hasKeyBounds) {
    int64_t keyExt = task.getAttrOfType<IntegerAttr>("cpu.key_extent").getInt();
    auto *kOmp = makeOp(b, loc, "c_key_omp_for", {}, {},
                        {b.getNamedAttr("key_extent", b.getI64IntegerAttr(keyExt))},
                        1);
    makeOp(b, loc, "end");
    Block &kOmpBb = kOmp->getRegion(0).emplaceBlock();
    b.setInsertionPointToEnd(&kOmpBb);
    makeOp(b, loc, "c_row_call", {}, {},
           {b.getNamedAttr("mode", b.getStringAttr("key")),
            b.getNamedAttr("call_args", b.getStringAttr(joinedCallArgs)),
            b.getNamedAttr("elem_bytes", b.getI64IntegerAttr(0))});
    makeOp(b, loc, "end");
  } else if (walk == "tile") {
    int64_t tileSz = task.getAttrOfType<IntegerAttr>("cpu.tile").getInt();
    int64_t elemBytes =
        cast<MemRefType>(fn.getArgument(0).getType()).getElementTypeBitWidth() / 8;
    if (task.hasAttr("cpu.single_dispatcher"))
      makeOp(b, loc, "c_row_prep", {}, {},
             {b.getNamedAttr("single_dispatcher", b.getBoolAttr(true)),
              b.getNamedAttr("member_tasks", b.getBoolAttr(false)),
              b.getNamedAttr("swap_dest", b.getBoolAttr(false))});
    auto *tOmp = makeOp(b, loc, "c_tile_2d_omp", {}, {},
                        {b.getNamedAttr("tile", b.getI64IntegerAttr(tileSz))}, 1);
    makeOp(b, loc, "end");
    Block &tOmpBb = tOmp->getRegion(0).emplaceBlock();
    b.setInsertionPointToEnd(&tOmpBb);
    makeOp(b, loc, "c_row_call", {}, {},
           {b.getNamedAttr("mode", b.getStringAttr("tile")),
            b.getNamedAttr("call_args", b.getStringAttr("")),
            b.getNamedAttr("elem_bytes", b.getI64IntegerAttr(elemBytes))});
    makeOp(b, loc, "end");
  } else if (flatColParallel) {
    int64_t colGrain = std::max<int64_t>(256, 16 * fullWidth);
    auto *fOmp = makeOp(b, loc, "c_flat_col_omp", {}, {},
                        {b.getNamedAttr("col_grain", b.getI64IntegerAttr(colGrain))},
                        1);
    makeOp(b, loc, "end");
    Block &fOmpBb = fOmp->getRegion(0).emplaceBlock();
    b.setInsertionPointToEnd(&fOmpBb);
    makeOp(b, loc, "c_row_call", {}, {},
           {b.getNamedAttr("mode", b.getStringAttr("flat_cols")),
            b.getNamedAttr("call_args", b.getStringAttr(joinedCallArgs)),
            b.getNamedAttr("elem_bytes", b.getI64IntegerAttr(0))});
    makeOp(b, loc, "end");
  } else {
    bool singleDisp = task.hasAttr("cpu.single_dispatcher");
    bool swapDest = walk == "destination";
    if (singleDisp || memberTasks || swapDest) {
      makeOp(b, loc, "c_row_prep", {}, {},
             {b.getNamedAttr("single_dispatcher", b.getBoolAttr(singleDisp)),
              b.getNamedAttr("member_tasks", b.getBoolAttr(memberTasks)),
              b.getNamedAttr("swap_dest", b.getBoolAttr(swapDest))});
    }
    auto emitDispatchBody = [&](Block &dstBb) {
      OpBuilder::InsertionGuard dg(b);
      b.setInsertionPointToEnd(&dstBb);
      if (lowerer.projectedRank) {
        auto *wLoop = makeOp(
            b, loc, "c_projected_while", {}, {},
            {b.getNamedAttr("proj_rank",
                            b.getI64IntegerAttr(lowerer.projectedRank))},
            1);
        makeOp(b, loc, "end");
        Block &wBb = wLoop->getRegion(0).emplaceBlock();
        b.setInsertionPointToEnd(&wBb);
        makeOp(b, loc, "c_row_call", {}, {},
               {b.getNamedAttr("mode", b.getStringAttr("projected")),
                b.getNamedAttr("call_args", b.getStringAttr(joinedCallArgs)),
                b.getNamedAttr("elem_bytes", b.getI64IntegerAttr(0))});
        makeOp(b, loc, "end");
      } else {
        makeOp(b, loc, "c_row_call", {}, {},
               {b.getNamedAttr("mode", b.getStringAttr(memberTasks ? "members"
                                                                   : "standard")),
                b.getNamedAttr("call_args", b.getStringAttr(joinedCallArgs)),
                b.getNamedAttr("elem_bytes", b.getI64IntegerAttr(0))});
        makeOp(b, loc, "end");
      }
    };
    auto *sGuard = makeOp(b, loc, "c_row_single_thread", {}, {}, {}, 1);
    emitDispatchBody(sGuard->getRegion(0).emplaceBlock());
    bool dynChunks =
        hasSegmentFold && !singleDisp && !memberTasks && !lowerer.projectedRank;
    int64_t chunkSize = 1;
    if (dynChunks) {
      int64_t knownRows = -1;
      if (auto kr = task.getAttrOfType<IntegerAttr>("mojo_row.known_rows"))
        knownRows = kr.getInt();
      int64_t repGrain = 1;
      if (rowsAttr && !rowsAttr.empty())
        repGrain =
            cast<DictionaryAttr>(rowsAttr[0]).getAs<IntegerAttr>("grain").getInt();
      if (knownRows > 0 && repGrain > 0) {
        int64_t groups = (knownRows - 1) / repGrain + 1;
        chunkSize = groups > 256 ? 8 : 1;
      }
    }
    auto *rOmp = makeOp(
        b, loc, "c_row_omp_parallel", {}, {},
        {b.getNamedAttr("schedule",
                        b.getStringAttr(dynChunks ? "dynamic" : "static")),
         b.getNamedAttr("chunk_size", b.getI64IntegerAttr(chunkSize))},
        1);
    makeOp(b, loc, "end");
    emitDispatchBody(rOmp->getRegion(0).emplaceBlock());
  }

  SmallVector<Operation *> toErase;
  for (Operation &op : fn.front()) {
    if (&op != prog && op.getName().getStringRef() != "func.return")
      toErase.push_back(&op);
  }
  for (Operation *op : llvm::reverse(toErase))
    op->erase();
  lowerMojoScfIfs(prog);
  return success();
}

bool hasMojoRowProgram(ModuleOp m) {
  auto funcs = m.getOps<func::FuncOp>();
  if (!llvm::hasSingleElement(funcs))
    return false;
  auto fn = *funcs.begin();
  return !fn.empty() && !fn.front().empty() &&
         fn.front().front().getName().getStringRef() == "mojo_row.row_program";
}

LogicalResult verifyMojoRow(ModuleOp m, const TargetProfile &p, bool callable) {
  auto fn = *m.getOps<func::FuncOp>().begin();
  auto &prog = fn.front().front();
  if (prog.getNumRegions() != 2 || prog.getRegion(0).empty() ||
      prog.getRegion(1).empty())
    return prog.emitError("incomplete mojo_row.row_program regions");

  std::set<std::string> emittedResources;
  for (Operation &op : prog.getRegion(0).front()) {
    if (op.getName().getStringRef() == "mojo_row.runtime_source") {
      auto impl = op.getAttrOfType<StringAttr>("implementation");
      if (!impl || !p.implementations.count(impl.getValue().str()))
        return op.emitError("unknown runtime_source implementation");
      emittedResources.insert(
          p.implementations.at(impl.getValue().str()).resource);
    }
  }

  bool badWalk =
      m.walk([&](Operation *op) {
         auto name = op->getName().getStringRef();
         if (name == "mojo_row.address") {
           auto bits = op->getAttrOfType<IntegerAttr>("index_bits");
           if (!bits || (bits.getInt() != 32 && bits.getInt() != 64)) {
             op->emitError("mojo_row.address requires i32 or i64 index_bits");
             return WalkResult::interrupt();
           }
           if (bits.getInt() == 32 && !canNarrowMojoAddress(*op)) {
             op->emitError(
                 "32-bit row address lacks a proven signed i32 range");
             return WalkResult::interrupt();
           }
         } else if (name == "mojo_row.dyn_address") {
           auto bits = op->getAttrOfType<IntegerAttr>("index_bits");
           auto terms = op->getAttrOfType<IntegerAttr>("terms");
           if (!bits || bits.getInt() != 64 || !terms || terms.getInt() < 0 ||
               int64_t(op->getNumOperands()) != 1 + 2 * terms.getInt()) {
             op->emitError(
                 "mojo_row.dyn_address requires 64-bit dynamic stride pairs");
             return WalkResult::interrupt();
           }
         } else if (name == "mojo_row.c_row_omp_parallel") {
           auto sched = op->getAttrOfType<StringAttr>("schedule");
           auto chunk = op->getAttrOfType<IntegerAttr>("chunk_size");
           if (!sched || !chunk || chunk.getInt() < 1 ||
               (sched.getValue() != "static" && sched.getValue() != "dynamic") ||
               (sched.getValue() == "static" && chunk.getInt() != 1)) {
             op->emitError("mojo_row.c_row_omp_parallel requires verified "
                           "schedule and chunk_size");
             return WalkResult::interrupt();
           }
         }
         if (auto bind = op->getAttrOfType<StringAttr>("ci.implementation")) {
           auto it = p.implementations.find(bind.getValue().str());
           if (it != p.implementations.end() &&
               it->second.source == ImplementationSource::Runtime &&
               !emittedResources.count(it->second.resource)) {
             op->emitError("missing runtime_source declaration for ")
                 << bind.getValue();
             return WalkResult::interrupt();
           }
         }
         return WalkResult::advance();
       }).wasInterrupted();
  if (badWalk)
    return failure();

  if (failed(mojoMatrixPrinter().verify(m, p)))
    return failure();

  if (callable) {
    if (fn->getAttrOfType<StringAttr>("mojo_row.entry") != "intent_entry")
      return fn.emitError("missing synchronous Mojo host entry");
    llvm::SmallVector<char> expected;
    llvm::raw_svector_ostream os(expected);
    if (auto err = describeMojoRowPackage(fn, prog, os))
      return fn.emitError() << llvm::toString(std::move(err)), failure();
    auto pkg = m->getAttrOfType<StringAttr>("mojo_row.package");
    if (!pkg || pkg.getValue() != StringRef(expected.data(), expected.size()))
      return fn.emitError("mojo_row.package disagrees with verified row ABI"),
             failure();
  }
  return success();
}
} // namespace CI_UNIT_NAMESPACE
