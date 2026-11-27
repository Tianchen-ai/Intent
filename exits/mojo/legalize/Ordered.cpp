#include "intentdsl/levels/algorithm/kir/ir/Ordered.h"
#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "intentdsl/levels/execution/cpu/ir/FoldProducts.h"
#include "intentdsl/levels/execution/cpu/ir/MatrixTask.h"
#include "intentdsl/levels/execution/cpu/ir/OrderedValue.h"
#include "intentdsl/levels/execution/cpu/ir/OwnerTraversal.h"
#include "intentdsl/levels/execution/cpu/ir/RepeatedScore.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include <cmath>
#include <set>

namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
using namespace mlir;

bool canNarrowMojoAddress(Operation &op);
TerminalPrinter mojoMatrixPrinter();
void lowerMojoScfIfs(Operation *rootOp);

namespace {
void unrollShortWindow(Operation &space) {
  auto &body = space.getRegion(0).front();
  for (auto loop : llvm::make_early_inc_range(body.getOps<scf::ForOp>())) {
    auto hi = getConstantIntValue(loop.getUpperBound());
    if (getConstantIntValue(loop.getLowerBound()) != 0 ||
        getConstantIntValue(loop.getStep()) != 1 || !hi || *hi < 2 || *hi > 16)
      continue;
    // Unrolling a small fixed window exposes constant tap coordinates to
    // the target compiler. Retain the source accumulator and guard order.
    auto variesWithTap = [&](auto &self, Value value) -> bool {
      if (value == loop.getInductionVar())
        return true;
      Operation *def = value.getDefiningOp();
      return def && loop->isAncestor(def) &&
             def->getName().getDialectNamespace() == "arith" &&
             llvm::any_of(def->getOperands(),
                          [&](Value operand) { return self(self, operand); });
    };
    bool guardedTap = false;
    unsigned ops = 0;
    loop.walk([&](Operation *op) {
      ++ops;
      if (auto load = dyn_cast<memref::LoadOp>(op))
        guardedTap |= isa<scf::IfOp>(load->getParentOp()) &&
                      llvm::any_of(load.getIndices(), [&](Value index) {
                        return variesWithTap(variesWithTap, index);
                      });
    });
    if (!guardedTap || ops * *hi > 512)
      continue;
    (void)loopUnrollByFactor(loop, *hi);
  }
}

std::string scalarTypeStr(Type t) {
  if (t.isIndex())
    return "Int";
  if (t.isF16())
    return "Float16";
  if (t.isBF16())
    return "BFloat16";
  if (t.isF32())
    return "Float32";
  if (t.isInteger(1))
    return "Bool";
  if (t.isUnsignedInteger(64))
    return "UInt64";
  if (t.isInteger(64))
    return "Int64";
  return "Int32";
}

std::string dtypeBare(Type t) {
  if (auto sh = dyn_cast<ShapedType>(t))
    t = sh.getElementType();
  if (t.isBF16())
    return "bfloat16";
  if (t.isF16())
    return "float16";
  if (t.isInteger(1))
    return "bool";
  if (t.isUnsignedInteger(64))
    return "uint64";
  if (t.isInteger(64))
    return "int64";
  if (t.isInteger(32))
    return "int32";
  return "float32";
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

Value constF32(OpBuilder &b, Location loc, float v) {
  return b.create<arith::ConstantOp>(loc, b.getF32Type(), b.getF32FloatAttr(v));
}

Value addIdx(OpBuilder &b, Location loc, Value base, int64_t delta) {
  if (delta == 0)
    return base;
  return b.create<arith::AddIOp>(loc, base, constIdx(b, loc, delta));
}

Value makeAddr(OpBuilder &b, Location loc, Value ptr, ValueRange indices,
               ArrayRef<int64_t> strides) {
  SmallVector<Value> args{ptr};
  args.append(indices.begin(), indices.end());
  return makeOp(b, loc, "address", args, {b.getIndexType()},
                {b.getNamedAttr("strides", b.getDenseI64ArrayAttr(strides)),
                 b.getNamedAttr("index_bits", b.getI32IntegerAttr(64))})
      ->getResult(0);
}

struct WideViewRef {
  Value ptr;
  Value baseAddr;
  Type rawElemType;
  int64_t extent = 0;
};

struct OrderedLowerer {
  func::FuncOp fn;
  Operation &space;
  OpBuilder &b;
  Location loc;
  int64_t batch = 1;
  OrderedLowerer(func::FuncOp fn, Operation &space, OpBuilder &b, Location loc,
                 int64_t batch)
      : fn(fn), space(space), b(b), loc(loc), batch(batch) {}
  Value localStackBuf;
  BlockArgument innerOwnerArg;
  bool batchSimd = false;
  bool usesReduce16x16 = false;
  llvm::DenseMap<Value, Value> batchVecs;
  llvm::DenseMap<Value, Value> wideBatchSplats;

  // Chunked L1-resident slice lowering state (active when chunkedMode == true)
  bool chunkedMode = false;
  int64_t invFloatsPerLane = 0;
  int64_t invStride = 0;
  int64_t accExtent = 0;
  int64_t accStride = 0;
  Value invPrepackBuf;
  Value wideAccBuf;
  std::optional<intentdsl::cpu::OrderedDotPanel> dotPanel;
  Value brgHandle, brgScratch, packedQueries, dots;
  Value vAddrBuf, packedValues, weights, tokenWeights;
  std::optional<intentdsl::cpu::OrderedValuePanel> valuePanel;
  bool emittedValuePanel = false;
  bool inTiledReplayPassA = false;
  Operation *tiledVReadOp = nullptr;
  Value tiledVPtr;
  Type tiledVElemTy;
  llvm::DenseMap<Operation *, int64_t> invReadOffset;
  SmallVector<llvm::DenseMap<Value, WideViewRef>, 16> wideViews;
  SmallVector<llvm::DenseMap<Value, Value>, 16> wideSplats;
  llvm::DenseSet<Value> wideBufValues;

  DenseI64ArrayAttr argArray(Value v, StringRef key) {
    return fn.getArgAttrOfType<DenseI64ArrayAttr>(
        cast<BlockArgument>(v).getArgNumber(), key);
  }

  int64_t memberLen(Value origVal) {
    return intentdsl::kir::memberExtent(fn, origVal, "cpu.shape");
  }

  bool isWideVal(Value origVal) {
    if (!chunkedMode || !isa<RankedTensorType>(origVal.getType()))
      return false;
    int64_t k = memberLen(origVal);
    return k >= 64 && k % 16 == 0;
  }

  int64_t simdWidthOf(Value origVal) {
    int64_t k = memberLen(origVal);
    return std::max<int64_t>(
        1, llvm::PowerOf2Ceil(uint64_t(std::max<int64_t>(1, k))));
  }

  void analyzeChunkableWideMode() {
    chunkedMode = false;
    invFloatsPerLane = 0;
    accExtent = 0;
    invReadOffset.clear();
    if (batch < 4)
      return;

    bool hasLargeBatchSlice = false;
    bool ok = true;
    space.walk([&](Operation *op) {
      if (!ok)
        return;
      for (Value res : op->getResults()) {
        if (!isa<RankedTensorType>(res.getType()))
          continue;
        int64_t k = memberLen(res);
        if (k >= 64) {
          if (k % 16 != 0) {
            ok = false;
            return;
          }
          if (batch * k >= 256)
            hasLargeBatchSlice = true;
        }
      }
    });
    if (!ok || !hasLargeBatchSlice)
      return;

    int64_t foundAcc = 0;
    space.walk([&](Operation *op) {
      if (!ok)
        return;
      auto kind = op->getName().stripDialect();
      bool touchesWide = false;
      for (Value v : op->getOperands())
        touchesWide |= isa<RankedTensorType>(v.getType()) && memberLen(v) >= 64;
      for (Value v : op->getResults())
        touchesWide |= isa<RankedTensorType>(v.getType()) && memberLen(v) >= 64;
      if (!touchesWide)
        return;
      if (auto loop = dyn_cast<scf::ForOp>(op)) {
        int64_t wideArgs = 0;
        for (Value init : loop.getInitArgs()) {
          if (isa<RankedTensorType>(init.getType()) && memberLen(init) >= 64) {
            ++wideArgs;
            int64_t k = memberLen(init);
            if (foundAcc != 0 && foundAcc != k)
              ok = false;
            foundAcc = k;
          }
        }
        if (wideArgs > 1)
          ok = false;
        return;
      }
      if (kind == "slice_read" || kind == "slice_write") {
        if (!op->hasAttr("mojo_row.contiguous_simd"))
          ok = false;
      } else if (kind != "extf" && kind != "truncf" && kind != "lane_splat" &&
                 kind != "lane_sum" && kind != "mulf" && kind != "addf" &&
                 !isa<scf::IfOp, scf::YieldOp>(op)) {
        ok = false;
      }
    });
    if (!ok)
      return;

    auto &topBb = space.getRegion(0).front();
    int64_t curOff = 0;
    for (Operation &op : topBb) {
      if (op.getName().stripDialect() == "slice_read") {
        int64_t k = memberLen(op.getResult(0));
        if (k >= 64 && k % 16 == 0) {
          invReadOffset[&op] = curOff;
          curOff += k;
        }
      }
    }
    chunkedMode = true;
    invFloatsPerLane = curOff;
    invStride =
        curOff > 0 ? (curOff + ((curOff % 32 == 0) ? 16 : 0)) : 0;
    accExtent = foundAcc;
    accStride =
        foundAcc > 0 ? (foundAcc + ((foundAcc % 32 == 0) ? 16 : 0)) : 0;
  }

  Value declSlot(Type ty, Value initVal, Value origRef) {
    if (auto rt = dyn_cast<RankedTensorType>(ty)) {
      int64_t w = simdWidthOf(origRef);
      SmallVector<Value> inits;
      if (initVal)
        inits.push_back(initVal);
      return makeOp(b, loc, "acc_decl", inits, {ty},
                    {b.getNamedAttr("dtype", b.getStringAttr(dtypeBare(rt))),
                     b.getNamedAttr("width", b.getI64IntegerAttr(w))})
          ->getResult(0);
    }
    SmallVector<Value> inits;
    if (initVal)
      inits.push_back(initVal);
    return makeOp(b, loc, "scalar_decl", inits, {ty},
                  {b.getNamedAttr("scalar_type",
                                  b.getStringAttr(scalarTypeStr(ty)))})
        ->getResult(0);
  }

  Value readSlot(Value slot) {
    return makeOp(b, loc, "acc_read", {slot}, {slot.getType()})->getResult(0);
  }

  void writeSlot(Value slot, Value val) {
    makeOp(b, loc, "acc_write", {slot, val});
  }

  bool canBatchVecType(Type t) const {
    return batchSimd &&
           (t.isF32() || t.isBF16() || t.isF16() || t.isIndex() ||
            t.isInteger(1) || t.isInteger(32) || t.isInteger(64) ||
            t.isUnsignedInteger(64));
  }

  Type vecElemType(Type t) const {
    return t.isIndex() ? b.getI64Type() : t;
  }

  RankedTensorType batchVecType(Type origTy) const {
    return RankedTensorType::get({batch}, vecElemType(origTy));
  }

  Value declBatchVecSlot(RankedTensorType vecTy, Value initVal) {
    SmallVector<Value, 1> inits;
    if (initVal)
      inits.push_back(initVal);
    return makeOp(b, loc, "acc_decl", inits, {vecTy},
                  {b.getNamedAttr("dtype", b.getStringAttr(dtypeBare(vecTy))),
                   b.getNamedAttr("width", b.getI64IntegerAttr(batch))})
        ->getResult(0);
  }

  Value getBatchVec(Value origVal, ArrayRef<IRMapping> lanes) {
    if (auto it = batchVecs.find(origVal); it != batchVecs.end() && it->second)
      return it->second;
    Type origTy = origVal.getType();
    Type elemTy = vecElemType(origTy);
    auto vecTy = RankedTensorType::get({batch}, elemTy);
    bool allSame = llvm::all_of(lanes, [&](const IRMapping &m) {
      return m.lookup(origVal) == lanes[0].lookup(origVal);
    });
    if (allSame) {
      Value s0 = lanes[0].lookup(origVal);
      if (origTy.isIndex())
        s0 = b.create<arith::IndexCastOp>(loc, b.getI64Type(), s0);
      return makeOp(b, loc, "splat_scalar", {s0}, {vecTy},
                    {b.getNamedAttr("dtype", b.getStringAttr(dtypeBare(elemTy))),
                     b.getNamedAttr("width", b.getI64IntegerAttr(batch))})
          ->getResult(0);
    }
    SmallVector<Value, 16> scalars;
    for (const auto &m : lanes) {
      Value sl = m.lookup(origVal);
      if (origTy.isIndex())
        sl = b.create<arith::IndexCastOp>(loc, b.getI64Type(), sl);
      scalars.push_back(sl);
    }
    return makeOp(b, loc, "pack_apply", scalars, {vecTy},
                  {b.getNamedAttr("in_dtype", b.getStringAttr(dtypeBare(elemTy))),
                   b.getNamedAttr("out_dtype",
                                  b.getStringAttr(dtypeBare(elemTy))),
                   b.getNamedAttr("kind", b.getStringAttr("id"))})
        ->getResult(0);
  }

  void materializeScalarLanes(Value origVal,
                              SmallVectorImpl<IRMapping> &lanes) {
    if (lanes.empty() || lanes[0].contains(origVal))
      return;
    auto it = batchVecs.find(origVal);
    if (it == batchVecs.end() || !it->second)
      return;
    Value vec = it->second;
    Type elemTy = cast<ShapedType>(vec.getType()).getElementType();
    for (size_t l = 0; l < lanes.size(); ++l) {
      Value s = makeOp(b, loc, "extract_lane", {vec}, {elemTy},
                       {b.getNamedAttr("lane", b.getI64IntegerAttr(l))})
                    ->getResult(0);
      if (origVal.getType().isIndex())
        s = b.create<arith::IndexCastOp>(loc, b.getIndexType(), s);
      lanes[l].map(origVal, s);
    }
  }

  bool hasWideSplat(unsigned laneIdx, Value v) const {
    return wideSplats[laneIdx].count(v) || wideBatchSplats.count(v);
  }

  Value getWideSplatScalar(unsigned laneIdx, Value v) {
    if (auto it = wideSplats[laneIdx].find(v);
        it != wideSplats[laneIdx].end())
      return it->second;
    Value vec = wideBatchSplats.lookup(v);
    Type elemTy = cast<ShapedType>(vec.getType()).getElementType();
    return makeOp(b, loc, "extract_lane", {vec}, {elemTy},
                  {b.getNamedAttr("lane", b.getI64IntegerAttr(laneIdx))})
        ->getResult(0);
  }

  bool isLaneUniformVal(Value v, ArrayRef<IRMapping> lanes) const {
    if (isa_and_nonnull<arith::ConstantOp, arith::ConstantIndexOp>(
            v.getDefiningOp()))
      return true;
    if (!lanes[0].contains(v))
      return false;
    Value ref = lanes[0].lookup(v);
    return llvm::all_of(
        lanes, [&](const IRMapping &m) { return m.lookupOrNull(v) == ref; });
  }

  bool isContiguousLaneIndex(Value v, ArrayRef<IRMapping> lanes) const {
    if (!innerOwnerArg || !lanes[0].contains(innerOwnerArg))
      return false;
    if (v == innerOwnerArg)
      return true;
    if (auto add = v.getDefiningOp<arith::AddIOp>())
      return (isContiguousLaneIndex(add.getLhs(), lanes) &&
              isLaneUniformVal(add.getRhs(), lanes)) ||
             (isLaneUniformVal(add.getLhs(), lanes) &&
              isContiguousLaneIndex(add.getRhs(), lanes));
    if (auto sub = v.getDefiningOp<arith::SubIOp>())
      return isContiguousLaneIndex(sub.getLhs(), lanes) &&
             isLaneUniformVal(sub.getRhs(), lanes);
    return false;
  }

  Value evalScalarIndexLane0(Value v, IRMapping &m0) {
    if (m0.contains(v))
      return m0.lookup(v);
    if (auto it = batchVecs.find(v); it != batchVecs.end()) {
      Value s = makeOp(b, loc, "extract_lane", {it->second}, {b.getI64Type()},
                       {b.getNamedAttr("lane", b.getI64IntegerAttr(0))})
                    ->getResult(0);
      if (v.getType().isIndex())
        s = b.create<arith::IndexCastOp>(loc, b.getIndexType(), s);
      m0.map(v, s);
      return s;
    }
    return Value{};
  }

  bool tryLowerBatchVecOp(Operation &op, SmallVectorImpl<IRMapping> &lanes) {
    if (!batchSimd || lanes.size() != size_t(batch) ||
        op.hasAttr("cpu.owner_common"))
      return false;
    auto n = op.getName().getStringRef();
    auto kind = op.getName().stripDialect();
    if (n == "cpu.buffer_store") {
      Value offOrig = op.getOperand(2);
      if (!lanes[0].contains(offOrig) ||
          !canBatchVecType(op.getOperand(0).getType()) ||
          !llvm::all_of(lanes, [&](const IRMapping &m) {
            return m.lookupOrNull(offOrig) == lanes[0].lookup(offOrig);
          }))
        return false;
      Value addr =
          makeAddr(b, loc, localStackBuf, {lanes[0].lookup(offOrig)}, {batch});
      makeOp(b, loc, "store_slice",
             {localStackBuf, addr, getBatchVec(op.getOperand(0), lanes)});
      return true;
    }
    if (n == "cpu.buffer_load") {
      Value offOrig = op.getOperand(1);
      Type resTy = op.getResult(0).getType();
      if (!lanes[0].contains(offOrig) || !canBatchVecType(resTy) ||
          !llvm::all_of(lanes, [&](const IRMapping &m) {
            return m.lookupOrNull(offOrig) == lanes[0].lookup(offOrig);
          }))
        return false;
      Value addr =
          makeAddr(b, loc, localStackBuf, {lanes[0].lookup(offOrig)}, {batch});
      batchVecs[op.getResult(0)] =
          loadContiguousSlice(localStackBuf, addr, vecElemType(resTy), batch);
      return true;
    }
    if (auto load = dyn_cast<memref::LoadOp>(op)) {
      auto strides = argArray(load.getMemref(), "cpu.strides");
      if (!innerOwnerArg || load.getIndices().empty() || strides.empty() ||
          strides[strides.size() - 1] != 1 ||
          !canBatchVecType(load.getResult().getType()) ||
          !isContiguousLaneIndex(load.getIndices().back(), lanes))
        return false;
      for (Value idx : load.getIndices().drop_back()) {
        if (!isLaneUniformVal(idx, lanes))
          return false;
      }
      Value ptr = lanes[0].lookup(load.getMemref());
      SmallVector<Value> idxs;
      for (Value idx : load.getIndices()) {
        Value v0 = evalScalarIndexLane0(idx, lanes[0]);
        if (!v0)
          return false;
        idxs.push_back(v0);
      }
      Value addr = makeAddr(b, loc, ptr, idxs, strides.asArrayRef());
      batchVecs[load.getResult()] = loadContiguousSlice(
          ptr, addr, vecElemType(load.getResult().getType()), batch);
      return true;
    }
    if (auto store = dyn_cast<memref::StoreOp>(op)) {
      auto strides = argArray(store.getMemref(), "cpu.strides");
      if (!innerOwnerArg || store.getIndices().empty() ||
          store.getIndices().back() != innerOwnerArg || strides.empty() ||
          strides[strides.size() - 1] != 1 ||
          !canBatchVecType(store.getValue().getType()) ||
          !lanes[0].contains(innerOwnerArg))
        return false;
      for (Value idx : store.getIndices().drop_back()) {
        if (!lanes[0].contains(idx) ||
            !llvm::all_of(lanes, [&](const IRMapping &m) {
              return m.lookupOrNull(idx) == lanes[0].lookup(idx);
            }))
          return false;
      }
      Value ptr = lanes[0].lookup(store.getMemref());
      SmallVector<Value> idxs;
      for (Value idx : store.getIndices())
        idxs.push_back(lanes[0].lookup(idx));
      Value addr = makeAddr(b, loc, ptr, idxs, strides.asArrayRef());
      makeOp(b, loc, "store_slice",
             {ptr, addr, getBatchVec(store.getValue(), lanes)});
      return true;
    }
    if (kind == "lane_splat" && isWideVal(op.getResult(0)) &&
        canBatchVecType(op.getOperand(0).getType())) {
      wideBatchSplats[op.getResult(0)] = getBatchVec(op.getOperand(0), lanes);
      return true;
    }
    if (op.getNumResults() != 1 || isa<ShapedType>(op.getResult(0).getType()) ||
        !canBatchVecType(op.getResult(0).getType()))
      return false;
    Type resTy = op.getResult(0).getType();
    if (isa<arith::ConstantOp, arith::ConstantIndexOp>(op))
      return false;
    if (isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
            arith::MaxNumFOp, arith::AddIOp, arith::SubIOp, arith::MulIOp,
            arith::AndIOp, arith::OrIOp, arith::XOrIOp>(op)) {
      if (!canBatchVecType(op.getOperand(0).getType()) ||
          !canBatchVecType(op.getOperand(1).getType()))
        return false;
      OperationState st(loc, n);
      st.addOperands({getBatchVec(op.getOperand(0), lanes),
                      getBatchVec(op.getOperand(1), lanes)});
      st.addTypes({batchVecType(resTy)});
      batchVecs[op.getResult(0)] = b.create(st)->getResult(0);
      return true;
    }
    if (auto e2 = dyn_cast<math::Exp2Op>(op);
        e2 && bitEnumContainsAll(e2.getFastmath(), arith::FastMathFlags::afn) &&
        resTy.isF32() && canBatchVecType(e2.getOperand().getType())) {
      auto vTy = batchVecType(resTy);
      Value ln2Scalar = constF32(b, loc, 0.6931471824645996f);
      Value ln2Vec =
          makeOp(b, loc, "splat_scalar", {ln2Scalar}, {vTy},
                 {b.getNamedAttr("dtype", b.getStringAttr("float32")),
                  b.getNamedAttr("width", b.getI64IntegerAttr(batch))})
              ->getResult(0);
      Value scaled = b.create<arith::MulFOp>(
          loc, getBatchVec(e2.getOperand(), lanes), ln2Vec, e2.getFastmath());
      batchVecs[e2.getResult()] =
          b.create<math::ExpOp>(loc, scaled, e2.getFastmath())->getResult(0);
      return true;
    }
    if (isa<math::ExpOp, math::Exp2Op, math::LogOp, math::SqrtOp,
            math::RsqrtOp, arith::ExtFOp, arith::TruncFOp>(op)) {
      if (!canBatchVecType(op.getOperand(0).getType()))
        return false;
      OperationState st(loc, n);
      st.addOperands({getBatchVec(op.getOperand(0), lanes)});
      st.addTypes({batchVecType(resTy)});
      batchVecs[op.getResult(0)] = b.create(st)->getResult(0);
      return true;
    }
    if (isa<arith::IndexCastOp, arith::SIToFPOp, arith::FPToSIOp>(op)) {
      if (!canBatchVecType(op.getOperand(0).getType()))
        return false;
      Value inVec = getBatchVec(op.getOperand(0), lanes);
      auto outVecTy = batchVecType(resTy);
      batchVecs[op.getResult(0)] =
          (inVec.getType() == outVecTy)
              ? inVec
              : makeOp(b, loc, "scalar_cast", {inVec}, {outVecTy},
                       {b.getNamedAttr("dtype",
                                       b.getStringAttr(dtypeBare(outVecTy)))})
                    ->getResult(0);
      return true;
    }
    if (auto cmp = dyn_cast<arith::CmpIOp>(op)) {
      if (!canBatchVecType(cmp.getLhs().getType()))
        return false;
      StringRef pred;
      switch (cmp.getPredicate()) {
      case arith::CmpIPredicate::eq:
        pred = "eq";
        break;
      case arith::CmpIPredicate::ne:
        pred = "ne";
        break;
      case arith::CmpIPredicate::slt:
        pred = "lt";
        break;
      case arith::CmpIPredicate::sle:
        pred = "le";
        break;
      case arith::CmpIPredicate::sgt:
        pred = "gt";
        break;
      case arith::CmpIPredicate::sge:
        pred = "ge";
        break;
      default:
        return false;
      }
      batchVecs[op.getResult(0)] =
          makeOp(b, loc, "simd_cmpi",
                 {getBatchVec(cmp.getLhs(), lanes),
                  getBatchVec(cmp.getRhs(), lanes)},
                 {batchVecType(b.getI1Type())},
                 {b.getNamedAttr("pred", b.getStringAttr(pred))})
              ->getResult(0);
      return true;
    }
    if (auto cmp = dyn_cast<arith::CmpFOp>(op)) {
      if (!canBatchVecType(cmp.getLhs().getType()))
        return false;
      StringRef pred;
      switch (cmp.getPredicate()) {
      case arith::CmpFPredicate::OEQ:
        pred = "eq";
        break;
      case arith::CmpFPredicate::ONE:
        pred = "ne";
        break;
      case arith::CmpFPredicate::OLT:
        pred = "lt";
        break;
      case arith::CmpFPredicate::OLE:
        pred = "le";
        break;
      case arith::CmpFPredicate::OGT:
        pred = "gt";
        break;
      case arith::CmpFPredicate::OGE:
        pred = "ge";
        break;
      default:
        return false;
      }
      batchVecs[op.getResult(0)] =
          makeOp(b, loc, "simd_cmpi",
                 {getBatchVec(cmp.getLhs(), lanes),
                  getBatchVec(cmp.getRhs(), lanes)},
                 {batchVecType(b.getI1Type())},
                 {b.getNamedAttr("pred", b.getStringAttr(pred))})
              ->getResult(0);
      return true;
    }
    if (auto sel = dyn_cast<arith::SelectOp>(op)) {
      if (!sel.getCondition().getType().isInteger(1) ||
          !canBatchVecType(sel.getTrueValue().getType()))
        return false;
      batchVecs[op.getResult(0)] =
          makeOp(b, loc, "simd_select",
                 {getBatchVec(sel.getCondition(), lanes),
                  getBatchVec(sel.getTrueValue(), lanes),
                  getBatchVec(sel.getFalseValue(), lanes)},
                 {batchVecType(resTy)})
              ->getResult(0);
      return true;
    }
    return false;
  }

  void emitZeroWideAccBuf() {
    auto f32x16 = RankedTensorType::get({16}, b.getF32Type());
    Value zeroF = constF32(b, loc, 0.0f);
    Value zeroVec =
        makeOp(b, loc, "splat_scalar", {zeroF}, {f32x16},
               {b.getNamedAttr("dtype", b.getStringAttr("float32")),
                b.getNamedAttr("width", b.getI64IntegerAttr(16))})
            ->getResult(0);
    auto *zLoop = makeOp(
        b, loc, "for_range",
        {constIdx(b, loc, 0), constIdx(b, loc, batch * accStride),
         constIdx(b, loc, 16)},
        {}, {}, 1);
    Block &zBb = zLoop->getRegion(0).emplaceBlock();
    Value c = zBb.addArgument(b.getIndexType(), loc);
    OpBuilder::InsertionGuard zg(b);
    b.setInsertionPointToEnd(&zBb);
    Value addr = makeAddr(b, loc, wideAccBuf, {c}, {1});
    makeOp(b, loc, "store_slice", {wideAccBuf, addr, zeroVec});
    makeOp(b, loc, "end");
  }

  Value loadContiguousSlice(Value ptr, Value addr, Type elemTy, int64_t width) {
    auto vecTy = RankedTensorType::get({width}, elemTy);
    return makeOp(b, loc, "load_slice", {ptr, addr}, {vecTy},
                  {b.getNamedAttr("width", b.getI64IntegerAttr(width))})
        ->getResult(0);
  }

  Value loadWideChunkF32(const WideViewRef &ref, Value cIdx, int64_t chunk) {
    auto f32Ty = RankedTensorType::get({chunk}, b.getF32Type());
    Value addr;
    auto baseCst = ref.baseAddr.getDefiningOp<arith::ConstantIndexOp>();
    auto cCst = cIdx.getDefiningOp<arith::ConstantIndexOp>();
    if (baseCst && cCst)
      addr = makeAddr(b, loc, ref.ptr,
                      {constIdx(b, loc, baseCst.value() + cCst.value())}, {1});
    else if (cCst && cCst.value() == 0)
      addr = makeAddr(b, loc, ref.ptr, {ref.baseAddr}, {1});
    else
      addr = makeAddr(b, loc, ref.ptr, {ref.baseAddr, cIdx}, {1, 1});
    if (ref.rawElemType.isF32())
      return loadContiguousSlice(ref.ptr, addr, b.getF32Type(), chunk);
    Value raw = loadContiguousSlice(ref.ptr, addr, ref.rawElemType, chunk);
    return b.create<arith::ExtFOp>(loc, f32Ty, raw);
  }

  Value emitSliceLoadGeneral(Value ptr, Value addr, Type vecTy, int64_t width,
                             int64_t valid, int64_t step, bool contig) {
    if (contig && valid == width)
      return makeOp(b, loc, "load_slice", {ptr, addr}, {vecTy},
                    {b.getNamedAttr("width", b.getI64IntegerAttr(width))})
          ->getResult(0);
    Type elemTy = cast<ShapedType>(vecTy).getElementType();
    Value slot =
        makeOp(b, loc, "acc_decl", {}, {vecTy},
               {b.getNamedAttr("dtype", b.getStringAttr(dtypeBare(elemTy))),
                b.getNamedAttr("width", b.getI64IntegerAttr(width))})
            ->getResult(0);
    auto *lLoop = makeOp(
        b, loc, "for_range",
        {constIdx(b, loc, 0), constIdx(b, loc, valid), constIdx(b, loc, 1)}, {},
        {}, 1);
    Block &lBb = lLoop->getRegion(0).emplaceBlock();
    Value laneIv = lBb.addArgument(b.getIndexType(), loc);
    {
      OpBuilder::InsertionGuard lg(b);
      b.setInsertionPointToEnd(&lBb);
      Value elemAddr = makeAddr(b, loc, ptr, {addr, laneIv}, {1, step});
      Value elem =
          makeOp(b, loc, "load_scalar", {ptr, elemAddr}, {elemTy})->getResult(0);
      makeOp(b, loc, "acc_insert_lane", {slot, laneIv, elem});
      makeOp(b, loc, "end");
    }
    return readSlot(slot);
  }

  void emitSliceStoreGeneral(Value ptr, Value addr, Value val, int64_t width,
                             int64_t valid, int64_t step, bool contig) {
    if (contig && valid == width) {
      makeOp(b, loc, "store_slice", {ptr, addr, val});
      return;
    }
    Type elemTy = cast<ShapedType>(val.getType()).getElementType();
    auto *sLoop = makeOp(
        b, loc, "for_range",
        {constIdx(b, loc, 0), constIdx(b, loc, valid), constIdx(b, loc, 1)}, {},
        {}, 1);
    Block &sBb = sLoop->getRegion(0).emplaceBlock();
    Value laneIv = sBb.addArgument(b.getIndexType(), loc);
    OpBuilder::InsertionGuard sg(b);
    b.setInsertionPointToEnd(&sBb);
    Value elem =
        makeOp(b, loc, "extract_lane_dyn", {val, laneIv}, {elemTy})->getResult(0);
    Value elemAddr = makeAddr(b, loc, ptr, {addr, laneIv}, {1, step});
    makeOp(b, loc, "store_scalar", {ptr, elemAddr, elem});
    makeOp(b, loc, "end");
  }

  LogicalResult lowerSingleOp(Operation &op, unsigned laneIdx,
                              IRMapping &map) {
    auto n = op.getName().getStringRef();
    auto kind = op.getName().stripDialect();
    if (n == "cpu.local_buffer") {
      map.map(op.getResult(0), localStackBuf);
      return success();
    }
    if (n == "cpu.buffer_store") {
      Value val = map.lookup(op.getOperand(0));
      Value off = map.lookup(op.getOperand(2));
      Value addr =
          batch == 1
              ? makeAddr(b, loc, localStackBuf, {off}, {1})
              : makeAddr(b, loc, localStackBuf,
                         {off, constIdx(b, loc, laneIdx)}, {batch, 1});
      makeOp(b, loc, "store_scalar", {localStackBuf, addr, val});
      return success();
    }
    if (n == "cpu.buffer_load") {
      Value off = map.lookup(op.getOperand(1));
      Value addr =
          batch == 1
              ? makeAddr(b, loc, localStackBuf, {off}, {1})
              : makeAddr(b, loc, localStackBuf,
                         {off, constIdx(b, loc, laneIdx)}, {batch, 1});
      Value res = makeOp(b, loc, "load_scalar", {localStackBuf, addr},
                         {op.getResult(0).getType()})
                      ->getResult(0);
      map.map(op.getResult(0), res);
      return success();
    }
    if (auto load = dyn_cast<memref::LoadOp>(op)) {
      Value ptr = map.lookup(load.getMemref());
      auto strides = argArray(load.getMemref(), "cpu.strides");
      SmallVector<Value> idxs;
      for (Value idx : load.getIndices())
        idxs.push_back(map.lookup(idx));
      Value addr = makeAddr(b, loc, ptr, idxs, strides.asArrayRef());
      Value res = makeOp(b, loc, "load_scalar", {ptr, addr},
                         {load.getResult().getType()})
                      ->getResult(0);
      map.map(load.getResult(), res);
      return success();
    }
    if (auto store = dyn_cast<memref::StoreOp>(op)) {
      Value ptr = map.lookup(store.getMemref());
      Value val = map.lookup(store.getValue());
      auto strides = argArray(store.getMemref(), "cpu.strides");
      SmallVector<Value> idxs;
      for (Value idx : store.getIndices())
        idxs.push_back(map.lookup(idx));
      Value addr = makeAddr(b, loc, ptr, idxs, strides.asArrayRef());
      makeOp(b, loc, "store_scalar", {ptr, addr, val});
      return success();
    }
    if (auto dim = dyn_cast<memref::DimOp>(op)) {
      int64_t extent =
          argArray(dim.getSource(), "cpu.shape")[*dim.getConstantIndex()];
      map.map(dim.getResult(), constIdx(b, loc, extent));
      return success();
    }
    if (kind == "lane_splat") {
      Value outOrig = op.getResult(0);
      Value in = map.lookup(op.getOperand(0));
      if (isWideVal(outOrig)) {
        wideSplats[laneIdx][outOrig] = in;
        return success();
      }
      int64_t w = simdWidthOf(outOrig);
      Value res =
          makeOp(b, loc, "splat_scalar", {in}, {outOrig.getType()},
                 {b.getNamedAttr("dtype",
                                 b.getStringAttr(dtypeBare(outOrig.getType()))),
                  b.getNamedAttr("width", b.getI64IntegerAttr(w))})
              ->getResult(0);
      map.map(outOrig, res);
      return success();
    }
    if (kind == "slice_read" || kind == "slice_write") {
      bool write = kind == "slice_write";
      Value origVec = write ? op.getOperand(0) : op.getResult(0);
      int64_t k = memberLen(origVec);
      Value origBase = op.getOperand(write ? 1 : 0);
      Value ptr = map.lookup(origBase);
      int64_t ax = op.getAttrOfType<IntegerAttr>("axis").getInt();
      auto allStrides = argArray(origBase, "cpu.strides");
      SmallVector<Value> coords;
      SmallVector<int64_t> coordStrides;
      for (auto [j, c] :
           llvm::enumerate(op.getOperands().drop_front(write ? 2 : 1))) {
        coords.push_back(map.lookup(c));
        coordStrides.push_back(allStrides[j + (int64_t(j) >= ax)]);
      }
      Value addr = makeAddr(b, loc, ptr, coords, coordStrides);
      if (isWideVal(origVec)) {
        Type elemTy = cast<MemRefType>(origBase.getType()).getElementType();
        if (!write) {
          if (inTiledReplayPassA && &op == tiledVReadOp && laneIdx == 0 &&
              (!valuePanel || valuePanel->read != &op)) {
            auto loop = op.getParentOfType<scf::ForOp>();
            Value lb = map.lookup(loop.getLowerBound());
            Value iv = map.lookup(loop.getInductionVar());
            Value off = b.create<arith::SubIOp>(loc, iv, lb);
            Value i64Addr =
                b.create<arith::IndexCastOp>(loc, b.getI64Type(), addr);
            Value vOffAddr = makeAddr(b, loc, vAddrBuf, {off}, {1});
            makeOp(b, loc, "store_scalar", {vAddrBuf, vOffAddr, i64Addr});
            tiledVPtr = ptr;
            tiledVElemTy = elemTy;
          }
          wideViews[laneIdx][origVec] = WideViewRef{ptr, addr, elemTy, k};
          return success();
        }
        Value srcVal = origVec;
        if (auto tr = srcVal.getDefiningOp<arith::TruncFOp>())
          srcVal = tr.getIn();
        Value scaleScalar;
        if (auto mul = srcVal.getDefiningOp<arith::MulFOp>()) {
          if (wideBufValues.contains(mul.getLhs()) &&
              hasWideSplat(laneIdx, mul.getRhs())) {
            srcVal = mul.getLhs();
            scaleScalar = getWideSplatScalar(laneIdx, mul.getRhs());
          } else if (wideBufValues.contains(mul.getRhs()) &&
                     hasWideSplat(laneIdx, mul.getLhs())) {
            srcVal = mul.getRhs();
            scaleScalar = getWideSplatScalar(laneIdx, mul.getLhs());
          }
        }
        auto f32x16 = RankedTensorType::get({16}, b.getF32Type());
        Value scaleVec;
        if (scaleScalar) {
          scaleVec =
              makeOp(b, loc, "splat_scalar", {scaleScalar}, {f32x16},
                     {b.getNamedAttr("dtype", b.getStringAttr("float32")),
                      b.getNamedAttr("width", b.getI64IntegerAttr(16))})
                  ->getResult(0);
        }
        auto emitWriteChunk = [&](Value cIdx) {
          auto cCst = cIdx.getDefiningOp<arith::ConstantIndexOp>();
          Value bufAddr =
              cCst ? makeAddr(b, loc, wideAccBuf,
                              {constIdx(b, loc,
                                        int64_t(laneIdx) * accStride +
                                            cCst.value())},
                              {1})
                   : makeAddr(b, loc, wideAccBuf,
                              {constIdx(b, loc, int64_t(laneIdx) * accStride),
                               cIdx},
                              {1, 1});
          Value chunkVal =
              loadContiguousSlice(wideAccBuf, bufAddr, b.getF32Type(), 16);
          if (scaleVec)
            chunkVal = b.create<arith::MulFOp>(loc, chunkVal, scaleVec);
          if (!elemTy.isF32()) {
            auto outTy = RankedTensorType::get({16}, elemTy);
            chunkVal = b.create<arith::TruncFOp>(loc, outTy, chunkVal);
          }
          Value dstAddr = (cCst && cCst.value() == 0)
                              ? makeAddr(b, loc, ptr, {addr}, {1})
                              : makeAddr(b, loc, ptr, {addr, cIdx}, {1, 1});
          makeOp(b, loc, "store_slice", {ptr, dstAddr, chunkVal});
        };
        if (k <= 128) {
          for (int64_t cOff = 0; cOff < k; cOff += 16)
            emitWriteChunk(constIdx(b, loc, cOff));
        } else {
          auto *wLoop = makeOp(
              b, loc, "for_range",
              {constIdx(b, loc, 0), constIdx(b, loc, k), constIdx(b, loc, 16)},
              {}, {}, 1);
          Block &wBb = wLoop->getRegion(0).emplaceBlock();
          Value cIdx = wBb.addArgument(b.getIndexType(), loc);
          OpBuilder::InsertionGuard wg(b);
          b.setInsertionPointToEnd(&wBb);
          emitWriteChunk(cIdx);
          makeOp(b, loc, "end");
        }
        return success();
      }

      int64_t w = std::max<int64_t>(
          1, llvm::PowerOf2Ceil(uint64_t(std::max<int64_t>(1, k))));
      int64_t step = allStrides[ax];
      bool contig = op.hasAttr("mojo_row.contiguous_simd");
      if (write) {
        emitSliceStoreGeneral(ptr, addr, map.lookup(origVec), w, k, step,
                              contig);
      } else {
        Value res = emitSliceLoadGeneral(ptr, addr, origVec.getType(), w, k,
                                         step, contig);
        map.map(origVec, res);
      }
      return success();
    }
    if (kind == "lane_sum") {
      Value origIn = op.getOperand(0);
      Value res =
          makeOp(b, loc, "lane_sum", {map.lookup(origIn)}, {b.getF32Type()})
              ->getResult(0);
      map.map(op.getResult(0), res);
      return success();
    }

    if (op.getNumResults() == 1 && isWideVal(op.getResult(0))) {
      if (auto ext = dyn_cast<arith::ExtFOp>(op)) {
        wideViews[laneIdx][ext.getResult()] = wideViews[laneIdx][ext.getIn()];
        return success();
      }
      if (isa<arith::MulFOp, arith::TruncFOp>(op))
        return success();
    }

    if (auto e2 = dyn_cast<math::Exp2Op>(op);
        e2 && bitEnumContainsAll(e2.getFastmath(), arith::FastMathFlags::afn) &&
        e2.getType().isF32()) {
      Value inVal = map.lookup(e2.getOperand());
      Value ln2Scalar = constF32(b, loc, 0.6931471824645996f);
      Value scaled =
          b.create<arith::MulFOp>(loc, inVal, ln2Scalar, e2.getFastmath());
      Value res =
          b.create<math::ExpOp>(loc, scaled, e2.getFastmath())->getResult(0);
      map.map(e2.getResult(), res);
      return success();
    }
    if (isa<arith::SIToFPOp, arith::FPToSIOp>(op)) {
      Value inVal = map.lookup(op.getOperand(0));
      Type outTy = op.getResult(0).getType();
      Value res = makeOp(b, loc, "scalar_cast", {inVal}, {outTy},
                         {b.getNamedAttr("dtype",
                                         b.getStringAttr(dtypeBare(outTy)))})
                      ->getResult(0);
      map.map(op.getResult(0), res);
      return success();
    }
    StringRef ns = op.getName().getDialectNamespace();
    if (ns == "arith" || ns == "math") {
      Operation *cloned = b.clone(op, map);
      cloned->removeAttr("cpu.owner_common");
      cloned->removeAttr("mojo_row.pack_scalars");
      return success();
    }
    return op.emitError("unsupported Mojo ordered instruction ") << n;
  }

  bool tryLowerVectorizedInnerLoop(scf::ForOp loop, IRMapping &lane0) {
    auto stepCst = loop.getStep().getDefiningOp<arith::ConstantIndexOp>();
    if (!stepCst || stepCst.value() != 1 || loop.getNumRegionIterArgs() > 1)
      return false;

    const bool isRed = loop.getNumRegionIterArgs() == 1;
    bool isAdd = false;
    Operation *redOp = nullptr;
    Value iterArg, stepTerm;
    if (isRed) {
      if (!loop->hasAttr("kir.reassociate") ||
          !loop.getResult(0).getType().isF32())
        return false;
      iterArg = loop.getRegionIterArgs()[0];
      if (!iterArg.hasOneUse())
        return false;
      Value yVal = loop.getBody()->getTerminator()->getOperand(0);
      redOp = yVal.getDefiningOp();
      if (!redOp || redOp->getBlock() != loop.getBody())
        return false;
      if (auto add = dyn_cast<arith::AddFOp>(redOp)) {
        isAdd = true;
        stepTerm = add.getLhs() == iterArg
                       ? add.getRhs()
                       : (add.getRhs() == iterArg ? add.getLhs() : Value{});
      } else if (auto mx = dyn_cast<arith::MaxNumFOp>(redOp)) {
        isAdd = false;
        stepTerm = mx.getLhs() == iterArg
                       ? mx.getRhs()
                       : (mx.getRhs() == iterArg ? mx.getLhs() : Value{});
      } else {
        return false;
      }
      if (!stepTerm || stepTerm == iterArg)
        return false;
    }

    auto vecElemOk = [&](Type t) {
      return t.isF32() || t.isBF16() || t.isF16() || t.isIndex() ||
             t.isInteger(1) || t.isInteger(32) || t.isInteger(64) ||
             t.isUnsignedInteger(64);
    };

    Value ivOrig = loop.getInductionVar();
    llvm::DenseSet<Value> writtenViews;
    for (Operation &bop : loop.getBody()->without_terminator()) {
      if (auto st = dyn_cast<memref::StoreOp>(bop)) {
        if (isRed)
          return false;
        writtenViews.insert(st.getMemref());
      }
    }
    if (!isRed && writtenViews.empty())
      return false;

    llvm::DenseSet<Value> dependsOnIv{ivOrig};
    bool hasContigMem = false;
    for (Operation &bop : loop.getBody()->without_terminator()) {
      if (&bop == redOp)
        continue;
      if (bop.getNumRegions() != 0)
        return false;
      if (isRed && llvm::is_contained(bop.getOperands(), iterArg))
        return false;
      if (auto ld = dyn_cast<memref::LoadOp>(bop)) {
        if (writtenViews.contains(ld.getMemref()))
          return false;
        bool anyIv = llvm::any_of(
            ld.getIndices(), [&](Value v) { return dependsOnIv.contains(v); });
        if (!anyIv)
          continue;
        auto strides = argArray(ld.getMemref(), "cpu.strides");
        if (ld.getIndices().empty() || ld.getIndices().back() != ivOrig ||
            strides.empty() || strides[strides.size() - 1] != 1 ||
            !vecElemOk(ld.getResult().getType()) ||
            llvm::any_of(ld.getIndices().drop_back(),
                         [&](Value v) { return dependsOnIv.contains(v); }))
          return false;
        hasContigMem = true;
        dependsOnIv.insert(ld.getResult());
        continue;
      }
      if (auto st = dyn_cast<memref::StoreOp>(bop)) {
        auto strides = argArray(st.getMemref(), "cpu.strides");
        if (st.getIndices().empty() || st.getIndices().back() != ivOrig ||
            strides.empty() || strides[strides.size() - 1] != 1 ||
            !vecElemOk(st.getValue().getType()) ||
            llvm::any_of(st.getIndices().drop_back(),
                         [&](Value v) { return dependsOnIv.contains(v); }))
          return false;
        hasContigMem = true;
        continue;
      }
      if (isa<memref::DimOp, arith::ConstantOp, arith::ConstantIndexOp>(bop))
        continue;
      if (bop.getNumResults() != 1 ||
          isa<ShapedType>(bop.getResult(0).getType()) ||
          !vecElemOk(bop.getResult(0).getType()))
        return false;
      bool anyIv = llvm::any_of(
          bop.getOperands(), [&](Value v) { return dependsOnIv.contains(v); });
      if (!anyIv)
        continue;
      bool opOk = false;
      if (isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
              arith::MaxNumFOp, arith::AddIOp, arith::SubIOp, arith::MulIOp,
              arith::AndIOp, arith::OrIOp, arith::XOrIOp>(bop)) {
        opOk = vecElemOk(bop.getOperand(0).getType()) &&
               vecElemOk(bop.getOperand(1).getType());
      } else if (isa<math::ExpOp, math::Exp2Op, math::LogOp, math::SqrtOp,
                     math::RsqrtOp, arith::ExtFOp, arith::TruncFOp,
                     arith::IndexCastOp, arith::SIToFPOp,
                     arith::FPToSIOp>(bop)) {
        opOk = vecElemOk(bop.getOperand(0).getType());
      } else if (auto ci = dyn_cast<arith::CmpIOp>(bop)) {
        auto p = ci.getPredicate();
        opOk = vecElemOk(ci.getLhs().getType()) &&
               (p == arith::CmpIPredicate::eq ||
                p == arith::CmpIPredicate::ne ||
                p == arith::CmpIPredicate::slt ||
                p == arith::CmpIPredicate::sle ||
                p == arith::CmpIPredicate::sgt ||
                p == arith::CmpIPredicate::sge);
      } else if (auto cf = dyn_cast<arith::CmpFOp>(bop)) {
        auto p = cf.getPredicate();
        opOk = vecElemOk(cf.getLhs().getType()) &&
               (p == arith::CmpFPredicate::OEQ ||
                p == arith::CmpFPredicate::ONE ||
                p == arith::CmpFPredicate::OLT ||
                p == arith::CmpFPredicate::OLE ||
                p == arith::CmpFPredicate::OGT ||
                p == arith::CmpFPredicate::OGE);
      } else if (auto sel = dyn_cast<arith::SelectOp>(bop)) {
        opOk = sel.getCondition().getType().isInteger(1) &&
               vecElemOk(sel.getTrueValue().getType());
      }
      if (!opOk)
        return false;
      dependsOnIv.insert(bop.getResult(0));
    }
    if (!hasContigMem)
      return false;

    constexpr int64_t W = 16;
    Value lb = lane0.lookup(loop.getLowerBound());
    Value ub = lane0.lookup(loop.getUpperBound());
    Value c0 = constIdx(b, loc, 0), c1 = constIdx(b, loc, 1),
          cW = constIdx(b, loc, W);
    Value span = b.create<arith::SubIOp>(loc, ub, lb);
    Value geW =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, span, cW);
    Value aligned =
        b.create<arith::MulIOp>(loc, b.create<arith::DivSIOp>(loc, span, cW), cW);
    Value vecSpan = b.create<arith::SelectOp>(loc, geW, aligned, c0);
    Value vecEnd = b.create<arith::AddIOp>(loc, lb, vecSpan);

    auto f32xW = RankedTensorType::get({W}, b.getF32Type());
    Value scalarSlot, vAccSlot;
    if (isRed) {
      Value initOrig = loop.getInitArgs()[0];
      scalarSlot = declSlot(b.getF32Type(), lane0.lookup(initOrig), initOrig);
      Value identScalar =
          isAdd ? constF32(b, loc, 0.0f)
                : b.create<arith::ConstantOp>(
                       loc, b.getF32Type(),
                       b.getFloatAttr(b.getF32Type(),
                                      llvm::APFloat::getInf(
                                          llvm::APFloat::IEEEsingle(),
                                          /*Negative=*/true)))
                      ->getResult(0);
      Value identVec =
          makeOp(b, loc, "splat_scalar", {identScalar}, {f32xW},
                 {b.getNamedAttr("dtype", b.getStringAttr("float32")),
                  b.getNamedAttr("width", b.getI64IntegerAttr(W))})
              ->getResult(0);
      vAccSlot =
          makeOp(b, loc, "acc_decl", {identVec}, {f32xW},
                 {b.getNamedAttr("dtype", b.getStringAttr("float32")),
                  b.getNamedAttr("width", b.getI64IntegerAttr(W))})
              ->getResult(0);
    }

    int64_t savedBatch = batch;
    bool savedBatchSimd = batchSimd;
    BlockArgument savedInner = innerOwnerArg;
    auto savedBatchVecs = std::move(batchVecs);
    batch = W;
    batchSimd = true;
    innerOwnerArg = BlockArgument();
    if (auto ba = dyn_cast<BlockArgument>(ivOrig))
      innerOwnerArg = ba;
    batchVecs.clear();

    auto *vLoop = makeOp(b, loc, "for_range", {lb, vecEnd, cW}, {}, {}, 1);
    Block &vBb = vLoop->getRegion(0).emplaceBlock();
    Value ivVec = vBb.addArgument(b.getIndexType(), loc);
    {
      OpBuilder::InsertionGuard vg(b);
      b.setInsertionPointToEnd(&vBb);
      SmallVector<IRMapping, 16> vecLanes(W, lane0);
      for (int64_t l = 0; l < W; ++l)
        vecLanes[l].map(ivOrig, ivVec);
      Value ivI64 = b.create<arith::IndexCastOp>(loc, b.getI64Type(), ivVec);
      batchVecs[ivOrig] =
          makeOp(b, loc, "simd_iota_add", {ivI64},
                 {RankedTensorType::get({W}, b.getI64Type())},
                 {b.getNamedAttr("dtype", b.getStringAttr("int64")),
                  b.getNamedAttr("width", b.getI64IntegerAttr(W))})
              ->getResult(0);
      for (Operation &bop : loop.getBody()->without_terminator()) {
        if (&bop == redOp)
          continue;
        bool dep = isa<memref::StoreOp>(bop) ||
                   (bop.getNumResults() == 1 &&
                    dependsOnIv.contains(bop.getResult(0)));
        if (!dep) {
          if (failed(lowerSingleOp(bop, 0, vecLanes[0]))) {
            batch = savedBatch;
            batchSimd = savedBatchSimd;
            innerOwnerArg = savedInner;
            batchVecs = std::move(savedBatchVecs);
            return false;
          }
          if (bop.getNumResults() == 1)
            for (int64_t l = 1; l < W; ++l)
              vecLanes[l].map(bop.getResult(0),
                              vecLanes[0].lookup(bop.getResult(0)));
          continue;
        }
        bool okVec = tryLowerBatchVecOp(bop, vecLanes);
        (void)okVec;
      }
      if (isRed) {
        Value stepVec = getBatchVec(stepTerm, vecLanes);
        Value curAcc = readSlot(vAccSlot);
        Value nextAcc =
            isAdd ? b.create<arith::AddFOp>(loc, curAcc, stepVec)->getResult(0)
                  : b.create<arith::MaxNumFOp>(loc, curAcc, stepVec)
                        ->getResult(0);
        writeSlot(vAccSlot, nextAcc);
      }
      makeOp(b, loc, "end");
    }

    batch = savedBatch;
    batchSimd = savedBatchSimd;
    innerOwnerArg = savedInner;
    batchVecs = std::move(savedBatchVecs);

    if (isRed) {
      auto redIf = b.create<scf::IfOp>(loc, TypeRange{}, geW, false);
      OpBuilder::InsertionGuard rg(b);
      b.setInsertionPointToStart(&redIf.getThenRegion().front());
      Value vFinal = readSlot(vAccSlot);
      Value vRed =
          isAdd ? makeOp(b, loc, "lane_sum", {vFinal}, {b.getF32Type()})
                      ->getResult(0)
                : makeOp(b, loc, "lane_max", {vFinal}, {b.getF32Type()},
                         {b.getNamedAttr("width", b.getI64IntegerAttr(W))})
                      ->getResult(0);
      Value curS = readSlot(scalarSlot);
      Value nextS =
          isAdd ? b.create<arith::AddFOp>(loc, curS, vRed)->getResult(0)
                : b.create<arith::MaxNumFOp>(loc, curS, vRed)->getResult(0);
      writeSlot(scalarSlot, nextS);
    }

    auto *tLoop = makeOp(b, loc, "for_range", {vecEnd, ub, c1}, {}, {}, 1);
    Block &tBb = tLoop->getRegion(0).emplaceBlock();
    Value ivTail = tBb.addArgument(b.getIndexType(), loc);
    {
      OpBuilder::InsertionGuard tg(b);
      b.setInsertionPointToEnd(&tBb);
      SmallVector<IRMapping, 1> tailLanes{lane0};
      tailLanes[0].map(ivOrig, ivTail);
      if (isRed)
        tailLanes[0].map(iterArg, readSlot(scalarSlot));
      if (failed(lowerBlock(*loop.getBody(), tailLanes)))
        return false;
      if (isRed) {
        Value nextTail =
            tailLanes[0].lookup(loop.getBody()->getTerminator()->getOperand(0));
        writeSlot(scalarSlot, nextTail);
      }
      makeOp(b, loc, "end");
    }
    if (isRed)
      lane0.map(loop.getResult(0), readSlot(scalarSlot));
    return true;
  }

  void emitDotPanel(scf::ForOp loop, SmallVectorImpl<IRMapping> &lanes) {
    auto &p = *dotPanel;
    auto address = [&](Operation *read, unsigned lane) {
      SmallVector<Value> coords;
      for (auto x : read->getOperands().drop_front()) {
        if (x == loop.getInductionVar())
          coords.push_back(constIdx(b, loc, 0));
        else {
          materializeScalarLanes(x, lanes);
          coords.push_back(lanes[lane].lookup(x));
        }
      }
      Value view = read->getOperand(0);
      return makeAddr(
          b, loc, lanes[lane].lookup(view), coords,
          argArray(view, "cpu.strides").asArrayRef().take_front(coords.size()));
    };
    Value q = lanes[0].lookup(p.query->getOperand(0));
    SmallVector<Value> rowBase;
    for (int i = 0; i < batch; ++i)
      rowBase.push_back(address(p.query, i));
    auto *pack = makeOp(b, loc, "for_range",
                        {constIdx(b, loc, 0), constIdx(b, loc, p.depth / 2),
                         constIdx(b, loc, 1)},
                        {}, {}, 1);
    {
      OpBuilder::InsertionGuard guard(b);
      Block &body = pack->getRegion(0).emplaceBlock();
      Value pair = body.addArgument(b.getIndexType(), loc);
      b.setInsertionPointToStart(&body);
      for (int i = 0; i < batch; ++i) {
        Value src = makeAddr(b, loc, q, {rowBase[i], pair}, {1, 2});
        Value dst = makeAddr(b, loc, packedQueries, {pair, constIdx(b, loc, i)},
                             {2 * batch, 2});
        makeOp(b, loc, "store_slice",
               {packedQueries, dst,
                loadContiguousSlice(q, src, b.getBF16Type(), 2)});
      }
      makeOp(b, loc, "end");
    }
    Value keyBase = address(p.key, 0);
    Value keys = makeOp(b, loc, "ptr_offset",
                        {lanes[0].lookup(p.key->getOperand(0)), keyBase},
                        {b.getIndexType()})
                     ->getResult(0);
    Value status =
        makeOp(b, loc, "brgemm_call",
               {brgHandle, keys, packedQueries, dots, brgScratch},
               {b.getI32Type()},
               {b.getNamedAttr("allow_subnormal_ftz", b.getBoolAttr(true)),
                b.getNamedAttr("ci.implementation",
                               p.dot->getAttr("cpu.panel_engine"))})
            ->getResult(0);
    makeOp(b, loc, "check_status", {status});
  }

  LogicalResult lowerBlock(Block &srcBb, SmallVectorImpl<IRMapping> &lanes,
                           Operation *start = nullptr,
                           Operation *stop = nullptr) {
    const bool vecActive = batchSimd && lanes.size() == size_t(batch);
    for (Operation &op : srcBb.without_terminator()) {
      if (&op == stop)
        break;
      if (start && op.isBeforeInBlock(start))
        continue;
      if (auto loop = dyn_cast<scf::ForOp>(op)) {
        if (dotPanel && dotPanel->dot->getParentOp() == &op)
          emitDotPanel(loop, lanes);
        if (!vecActive && lanes.size() == 1 &&
            tryLowerVectorizedInnerLoop(loop, lanes[0]))
          continue;
        materializeScalarLanes(loop.getLowerBound(), lanes);
        materializeScalarLanes(loop.getUpperBound(), lanes);
        materializeScalarLanes(loop.getStep(), lanes);
        unsigned numIter = loop.getNumRegionIterArgs();
        SmallVector<SmallVector<Value, 4>, 16> slots(lanes.size());
        SmallVector<Value, 4> vecSlots(numIter, Value{});
        bool hasWideIter = false;
        for (unsigned k = 0; k < numIter; ++k) {
          Value initOrig = loop.getInitArgs()[k];
          Type resTy = loop.getResult(k).getType();
          if (isWideVal(initOrig)) {
            hasWideIter = true;
            wideBufValues.insert(loop.getRegionIterArgs()[k]);
            wideBufValues.insert(loop.getResult(k));
            for (size_t l = 0; l < lanes.size(); ++l)
              slots[l].push_back(Value{});
            continue;
          }
          if (vecActive && !isa<ShapedType>(resTy) && canBatchVecType(resTy) &&
              !resTy.isInteger(1)) {
            vecSlots[k] = declBatchVecSlot(batchVecType(resTy),
                                           getBatchVec(initOrig, lanes));
            for (size_t l = 0; l < lanes.size(); ++l)
              slots[l].push_back(Value{});
            continue;
          }
          materializeScalarLanes(initOrig, lanes);
          for (size_t l = 0; l < lanes.size(); ++l) {
            Value init = lanes[l].lookup(initOrig);
            slots[l].push_back(declSlot(resTy, init, initOrig));
          }
        }
        // Check if this is a window-replay loop eligible for register-tiled
        // column-outer accumulation (zero L1d vector stores per token!)
        Operation *candVRead = nullptr;
        unsigned wideReads = 0, wideAdds = 0;
        bool canTileReplay = false;
        if (hasWideIter && (vAddrBuf || valuePanel) && batch >= 2 &&
            accExtent >= 32 && accExtent % 32 == 0) {
          if (auto parentIf = dyn_cast<scf::IfOp>(loop->getParentOp())) {
            if (parentIf->hasAttr("cpu.window_branch") &&
                loop->getParentRegion() == &parentIf.getThenRegion()) {
              bool bodyOk = true;
              for (Operation &bop : loop.getBody()->without_terminator()) {
                if (bop.getNumRegions() != 0) {
                  bool nestedWideOrWrite =
                      llvm::any_of(bop.getResults(),
                                   [&](Value v) { return isWideVal(v); }) ||
                      bop.walk([&](Operation *inner) {
                           if (inner == &bop)
                             return WalkResult::advance();
                           if (llvm::any_of(inner->getResults(),
                                            [&](Value v) { return isWideVal(v); }) ||
                               llvm::any_of(inner->getOperands(),
                                            [&](Value v) { return isWideVal(v); }) ||
                               !isMemoryEffectFree(inner))
                             return WalkResult::interrupt();
                           return WalkResult::advance();
                         })
                          .wasInterrupted();
                  if (nestedWideOrWrite)
                    bodyOk = false;
                }
                if (bop.getName().stripDialect() == "slice_read" &&
                    isWideVal(bop.getResult(0))) {
                  ++wideReads;
                  if (bop.hasAttr("cpu.owner_common") &&
                      memberLen(bop.getResult(0)) == accExtent)
                    candVRead = &bop;
                }
                if (auto aop = dyn_cast<arith::AddFOp>(bop);
                    aop && isWideVal(aop.getResult())) {
                  ++wideAdds;
                  if (!wideBufValues.contains(aop.getLhs()) &&
                      !wideBufValues.contains(aop.getRhs()))
                    bodyOk = false;
                }
              }
              if (bodyOk && wideReads == 1 && wideAdds == 1 && candVRead &&
                  (vAddrBuf || candVRead == valuePanel->read))
                canTileReplay = true;
            }
          }
        }

        if (hasWideIter && !canTileReplay)
          emitZeroWideAccBuf();

        Value lb = lanes[0].lookup(loop.getLowerBound());
        Value ub = lanes[0].lookup(loop.getUpperBound());
        Value step = lanes[0].lookup(loop.getStep());
        auto *forOp = makeOp(b, loc, "for_range", {lb, ub, step}, {}, {}, 1);
        Block &fBb = forOp->getRegion(0).emplaceBlock();
        Value iv = fBb.addArgument(b.getIndexType(), loc);
        {
          OpBuilder::InsertionGuard fg(b);
          b.setInsertionPointToEnd(&fBb);
          SmallVector<IRMapping, 16> bodyLanes(lanes.begin(), lanes.end());
          for (size_t l = 0; l < bodyLanes.size(); ++l) {
            bodyLanes[l].map(loop.getInductionVar(), iv);
            for (unsigned k = 0; k < numIter; ++k)
              if (slots[l][k])
                bodyLanes[l].map(loop.getRegionIterArgs()[k],
                                 readSlot(slots[l][k]));
          }
          for (unsigned k = 0; k < numIter; ++k)
            if (vecSlots[k])
              batchVecs[loop.getRegionIterArgs()[k]] = readSlot(vecSlots[k]);
          bool prevTiled = inTiledReplayPassA;
          Operation *prevVRead = tiledVReadOp;
          if (canTileReplay) {
            inTiledReplayPassA = true;
            tiledVReadOp = candVRead;
          }
          if (failed(lowerBlock(*loop.getBody(), bodyLanes)))
            return failure();
          inTiledReplayPassA = prevTiled;
          tiledVReadOp = prevVRead;
          auto *term = loop.getBody()->getTerminator();
          SmallVector<Value, 4> nextVecVals(numIter, Value{});
          for (unsigned k = 0; k < numIter; ++k) {
            if (vecSlots[k])
              nextVecVals[k] = getBatchVec(term->getOperand(k), bodyLanes);
            else if (slots[0][k])
              materializeScalarLanes(term->getOperand(k), bodyLanes);
          }
          for (unsigned k = 0; k < numIter; ++k)
            if (vecSlots[k])
              writeSlot(vecSlots[k], nextVecVals[k]);
          for (size_t l = 0; l < bodyLanes.size(); ++l) {
            SmallVector<Value, 4> nextVals(numIter, Value{});
            for (unsigned k = 0; k < numIter; ++k)
              if (slots[l][k])
                nextVals[k] = bodyLanes[l].lookup(term->getOperand(k));
            for (unsigned k = 0; k < numIter; ++k)
              if (slots[l][k])
                writeSlot(slots[l][k], nextVals[k]);
          }
          makeOp(b, loc, "end");
        }
        if (canTileReplay) {
          Value span = b.create<arith::SubIOp>(loc, ub, lb);
          auto f32x16 = RankedTensorType::get({16}, b.getF32Type());
          if (valuePanel && candVRead == valuePanel->read) {
            emittedValuePanel = true;
            // Pass B consumes only the keys the fold actually visits. The value
            // table is immutable and was packed once before launching the
            // tasks.
            SmallVector<Value> coords;
            for (Value x : candVRead->getOperands().drop_front().drop_back()) {
              materializeScalarLanes(x, lanes);
              coords.push_back(lanes[0].lookup(x));
            }
            auto view = candVRead->getOperand(0);
            coords.push_back(lb);
            auto base = makeAddr(b, loc, packedValues, coords,
                                 argArray(view, "cpu.strides")
                                     .asArrayRef()
                                     .take_front(coords.size()));
            auto ptr = makeOp(b, loc, "ptr_offset", {packedValues, base},
                              {b.getIndexType()})
                           ->getResult(0);
            makeOp(
                b, loc, "transpose_weights", {tokenWeights, weights, span}, {},
                {b.getNamedAttr("columns", b.getI64IntegerAttr(batch)),
                 b.getNamedAttr("pitch", b.getI64IntegerAttr(dotPanel->tokens)),
                 b.getNamedAttr(
                     "ci.implementation",
                     b.getStringAttr("exits/mojo::transpose-bf16-weights"))});
            auto positive = b.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::sgt, span, constIdx(b, loc, 0));
            auto guarded = b.create<scf::IfOp>(loc, positive, true);
            OpBuilder::InsertionGuard cfg(b);
            b.setInsertionPointToStart(guarded.elseBlock());
            emitZeroWideAccBuf();
            b.setInsertionPointToStart(guarded.thenBlock());
            auto args = SmallVector<Value>{weights, ptr,       wideAccBuf,
                                           span,    brgHandle, brgScratch};
            auto status =
                makeOp(
                    b, loc, "value_prefix", args, {b.getI32Type()},
                    {b.getNamedAttr("rows", b.getI64IntegerAttr(batch)),
                     b.getNamedAttr("columns", b.getI64IntegerAttr(accExtent)),
                     b.getNamedAttr("lda",
                                    b.getI64IntegerAttr(dotPanel->tokens)),
                     b.getNamedAttr("ldc", b.getI64IntegerAttr(accStride)),
                     b.getNamedAttr(
                         "ci.implementation",
                         b.getStringAttr("exits/mojo::bf16-value-prefix"))})
                    ->getResult(0);
            makeOp(b, loc, "check_status", {status});
          } else if (accExtent == 64 && lanes.size() == 16) {
            // [6, 6, 4] x 64-col full-width register-blocked accumulation
            // (24, 24, 16 ZMM accumulators: 3 passes instead of 4, 50% fewer w loads)
            const size_t starts[3] = {0, 6, 12};
            const size_t lens[3] = {6, 6, 4};
            for (int grp = 0; grp < 3; ++grp) {
              size_t sub = starts[grp], subLen = lens[grp];
              SmallVector<std::array<Value, 4>, 6> accs(subLen);
              for (size_t s = 0; s < subLen; ++s)
                for (int c = 0; c < 4; ++c)
                  accs[s][c] =
                      makeOp(b, loc, "acc_decl", {}, {f32x16},
                             {b.getNamedAttr("dtype",
                                             b.getStringAttr("float32")),
                              b.getNamedAttr("width",
                                             b.getI64IntegerAttr(16))})
                          ->getResult(0);
              {
                auto *tInner = makeOp(
                    b, loc, "for_range",
                    {constIdx(b, loc, 0), span, constIdx(b, loc, 1)}, {}, {},
                    1);
                Block &tBb = tInner->getRegion(0).emplaceBlock();
                Value off = tBb.addArgument(b.getIndexType(), loc);
                OpBuilder::InsertionGuard tg(b);
                b.setInsertionPointToEnd(&tBb);
                Value vOffAddr = makeAddr(b, loc, vAddrBuf, {off}, {1});
                Value vBase64 = makeOp(b, loc, "load_scalar",
                                       {vAddrBuf, vOffAddr}, {b.getI64Type()})
                                    ->getResult(0);
                Value vBase =
                    b.create<arith::IndexCastOp>(loc, b.getIndexType(), vBase64).getResult();
                WideViewRef vRef{tiledVPtr, vBase, tiledVElemTy, accExtent};
                Value vChunks[4] = {
                    loadWideChunkF32(vRef, constIdx(b, loc, 0), 16),
                    loadWideChunkF32(vRef, constIdx(b, loc, 16), 16),
                    loadWideChunkF32(vRef, constIdx(b, loc, 32), 16),
                    loadWideChunkF32(vRef, constIdx(b, loc, 48), 16)};
                for (size_t s = 0; s < subLen; ++s) {
                  Value wAddr = makeAddr(b, loc, localStackBuf,
                                         {off, constIdx(b, loc, sub + s)},
                                         {batch, 1});
                  Value wScalar =
                      makeOp(b, loc, "load_scalar", {localStackBuf, wAddr},
                             {b.getF32Type()})
                          ->getResult(0);
                  Value wVec =
                      makeOp(b, loc, "splat_scalar", {wScalar}, {f32x16},
                             {b.getNamedAttr("dtype",
                                             b.getStringAttr("float32")),
                              b.getNamedAttr("width",
                                             b.getI64IntegerAttr(16))})
                          ->getResult(0);
                  for (int c = 0; c < 4; ++c)
                    writeSlot(
                        accs[s][c],
                        b.create<arith::AddFOp>(
                            loc, readSlot(accs[s][c]),
                            b.create<arith::MulFOp>(loc, vChunks[c], wVec)));
                }
                makeOp(b, loc, "end");
              }
              for (size_t s = 0; s < subLen; ++s) {
                int64_t laneBase = int64_t(sub + s) * accStride;
                for (int c = 0; c < 4; ++c) {
                  Value dstAddr = makeAddr(
                      b, loc, wideAccBuf,
                      {constIdx(b, loc, laneBase + c * 16)}, {1});
                  makeOp(b, loc, "store_slice",
                         {wideAccBuf, dstAddr, readSlot(accs[s][c])});
                }
              }
            }
          } else {
            int64_t width = 32;
            if (auto geometry =
                    space.getAttrOfType<DictionaryAttr>("cpu.vector_panel")) {
              int64_t planned = geometry.getAs<IntegerAttr>("value_width").getInt();
              if (planned > 0 && planned % 16 == 0 && accExtent % planned == 0)
                width = planned;
            }
            for (size_t sub = 0; sub < lanes.size(); sub += 8) {
              size_t subEnd = std::min<size_t>(lanes.size(), sub + 8);
              size_t subLen = subEnd - sub;
              auto *cOuter = makeOp(
                  b, loc, "for_range",
                  {constIdx(b, loc, 0), constIdx(b, loc, accExtent),
                   constIdx(b, loc, width)},
                  {}, {}, 1);
              Block &cBb = cOuter->getRegion(0).emplaceBlock();
              Value cIdx = cBb.addArgument(b.getIndexType(), loc);
              OpBuilder::InsertionGuard cg(b);
              b.setInsertionPointToEnd(&cBb);
              SmallVector<Value> columns;
              for (int64_t c = 0; c < width; c += 16)
                columns.push_back(c ? addIdx(b, loc, cIdx, c) : cIdx);
              SmallVector<SmallVector<Value>> acc(subLen);
              for (auto &row : acc)
                for (size_t c = 0; c < columns.size(); ++c)
                  row.push_back(
                      makeOp(b, loc, "acc_decl", {}, {f32x16},
                             {b.getNamedAttr("dtype", b.getStringAttr("float32")),
                              b.getNamedAttr("width", b.getI64IntegerAttr(16))})
                          ->getResult(0));
              {
                auto *tInner = makeOp(
                    b, loc, "for_range",
                    {constIdx(b, loc, 0), span, constIdx(b, loc, 1)}, {}, {},
                    1);
                Block &tBb = tInner->getRegion(0).emplaceBlock();
                Value off = tBb.addArgument(b.getIndexType(), loc);
                OpBuilder::InsertionGuard tg(b);
                b.setInsertionPointToEnd(&tBb);
                Value vOffAddr = makeAddr(b, loc, vAddrBuf, {off}, {1});
                Value vBase64 = makeOp(b, loc, "load_scalar",
                                       {vAddrBuf, vOffAddr}, {b.getI64Type()})
                                    ->getResult(0);
                Value vBase =
                    b.create<arith::IndexCastOp>(loc, b.getIndexType(), vBase64).getResult();
                WideViewRef vRef{tiledVPtr, vBase, tiledVElemTy, accExtent};
                SmallVector<Value> values;
                for (Value column : columns)
                  values.push_back(loadWideChunkF32(vRef, column, 16));
                for (size_t s = 0; s < subLen; ++s) {
                  Value wAddr = makeAddr(b, loc, localStackBuf,
                                         {off, constIdx(b, loc, sub + s)},
                                         {batch, 1});
                  Value wScalar =
                      makeOp(b, loc, "load_scalar", {localStackBuf, wAddr},
                             {b.getF32Type()})
                          ->getResult(0);
                  Value wVec =
                      makeOp(b, loc, "splat_scalar", {wScalar}, {f32x16},
                             {b.getNamedAttr("dtype",
                                             b.getStringAttr("float32")),
                              b.getNamedAttr("width",
                                             b.getI64IntegerAttr(16))})
                          ->getResult(0);
                  for (auto [slot, value] : llvm::zip_equal(acc[s], values))
                    writeSlot(slot, b.create<arith::AddFOp>(
                                        loc, readSlot(slot),
                                        b.create<arith::MulFOp>(loc, value, wVec)));
                }
                makeOp(b, loc, "end");
              }
              for (size_t s = 0; s < subLen; ++s) {
                int64_t laneBase = int64_t(sub + s) * accStride;
                for (auto [slot, column] : llvm::zip_equal(acc[s], columns)) {
                  Value addr = makeAddr(b, loc, wideAccBuf,
                                        {constIdx(b, loc, laneBase), column},
                                        {1, 1});
                  makeOp(b, loc, "store_slice", {wideAccBuf, addr, readSlot(slot)});
                }
              }
              makeOp(b, loc, "end");
            }
          }
        }
        for (unsigned k = 0; k < numIter; ++k)
          if (vecSlots[k])
            batchVecs[loop.getResult(k)] = readSlot(vecSlots[k]);
        for (size_t l = 0; l < lanes.size(); ++l)
          for (unsigned k = 0; k < numIter; ++k)
            if (slots[l][k])
              lanes[l].map(loop.getResult(k), readSlot(slots[l][k]));
        continue;
      }

      if (auto branch = dyn_cast<scf::IfOp>(op)) {
        unsigned numRes = branch.getNumResults();
        bool hasElse = !branch.getElseRegion().empty();
        // Pure scalar/vector conditional value selection (`a if cond else b`):
        // lower directly to SIMD select when both regions only materialize
        // side-effect-free constants.
        if (vecActive && numRes == 1 && hasElse &&
            !isWideVal(branch.thenYield().getOperand(0)) &&
            canBatchVecType(branch.getResult(0).getType()) &&
            !branch.getResult(0).getType().isInteger(1) &&
            batchVecs.contains(branch.getCondition())) {
          auto onlyConsts = [](Block &blk) {
            return llvm::all_of(blk.without_terminator(), [](Operation &inner) {
              return isa<arith::ConstantOp, arith::ConstantIndexOp>(inner);
            });
          };
          if (onlyConsts(branch.getThenRegion().front()) &&
              onlyConsts(branch.getElseRegion().front())) {
            for (Operation &cOp :
                 branch.getThenRegion().front().without_terminator())
              for (size_t l = 0; l < lanes.size(); ++l)
                (void)lowerSingleOp(cOp, l, lanes[l]);
            for (Operation &cOp :
                 branch.getElseRegion().front().without_terminator())
              for (size_t l = 0; l < lanes.size(); ++l)
                (void)lowerSingleOp(cOp, l, lanes[l]);
            Type resTy = branch.getResult(0).getType();
            batchVecs[branch.getResult(0)] =
                makeOp(b, loc, "simd_select",
                       {getBatchVec(branch.getCondition(), lanes),
                        getBatchVec(branch.thenYield().getOperand(0), lanes),
                        getBatchVec(branch.elseYield().getOperand(0), lanes)},
                       {batchVecType(resTy)})
                    ->getResult(0);
            continue;
          }
        }
        // Fast path for masked unit-stride stencil/window loads across batched
        // contiguous lanes: when all lanes are in-bounds (interior tiles),
        // issue a single SIMD load_slice; when all lanes are out-of-bounds,
        // splat the else value; only scalarize on partial boundary tiles.
        if (vecActive && numRes == 1 && hasElse &&
            !isWideVal(branch.thenYield().getOperand(0)) &&
            canBatchVecType(branch.getResult(0).getType()) &&
            !branch.getResult(0).getType().isInteger(1) &&
            batchVecs.contains(branch.getCondition()) &&
            branch.getElseRegion().front().without_terminator().empty()) {
          memref::LoadOp theLoad;
          bool thenPurePlusOneLoad = true;
          for (Operation &top :
               branch.getThenRegion().front().without_terminator()) {
            if (auto ld = dyn_cast<memref::LoadOp>(top)) {
              if (theLoad) {
                thenPurePlusOneLoad = false;
                break;
              }
              theLoad = ld;
            } else if (!isa<arith::ConstantOp, arith::ConstantIndexOp,
                            arith::AddIOp, arith::SubIOp,
                            arith::IndexCastOp>(top)) {
              thenPurePlusOneLoad = false;
              break;
            }
          }
          if (thenPurePlusOneLoad && theLoad &&
              branch.thenYield().getOperand(0) == theLoad.getResult()) {
            auto strides = argArray(theLoad.getMemref(), "cpu.strides");
            auto checkContigInThen = [&](auto &self, Value v) -> bool {
              if (!branch->isProperAncestor(
                      v.getParentRegion()->getParentOp()))
                return isContiguousLaneIndex(v, lanes);
              if (auto add = v.getDefiningOp<arith::AddIOp>())
                return (self(self, add.getLhs()) &&
                        isLaneUniformVal(add.getRhs(), lanes)) ||
                       (isLaneUniformVal(add.getLhs(), lanes) &&
                        self(self, add.getRhs()));
              if (auto sub = v.getDefiningOp<arith::SubIOp>())
                return self(self, sub.getLhs()) &&
                       isLaneUniformVal(sub.getRhs(), lanes);
              return false;
            };
            bool leadUniform = llvm::all_of(
                theLoad.getIndices().drop_back(), [&](Value idx) {
                  return !branch->isProperAncestor(
                             idx.getParentRegion()->getParentOp()) &&
                         isLaneUniformVal(idx, lanes);
                });
            if (!strides.empty() && strides[strides.size() - 1] == 1 &&
                !theLoad.getIndices().empty() && leadUniform &&
                checkContigInThen(checkContigInThen,
                                  theLoad.getIndices().back())) {
              Type resTy = branch.getResult(0).getType();
              Value condVec = getBatchVec(branch.getCondition(), lanes);
              Value elseVec =
                  getBatchVec(branch.elseYield().getOperand(0), lanes);
              Value vecSlot = declBatchVecSlot(batchVecType(resTy), elseVec);
              Value allCond =
                  makeOp(b, loc, "lane_all", {condVec}, {b.getI1Type()})
                      ->getResult(0);
              auto fastIf =
                  b.create<scf::IfOp>(loc, TypeRange{}, allCond, true);
              {
                OpBuilder::InsertionGuard fg(b);
                b.setInsertionPointToStart(&fastIf.getThenRegion().front());
                IRMapping m0 = lanes[0];
                for (Value idx : theLoad.getIndices())
                  if (!branch->isProperAncestor(
                          idx.getParentRegion()->getParentOp()))
                    evalScalarIndexLane0(idx, m0);
                bool ok0 = true;
                for (Operation &top :
                     branch.getThenRegion().front().without_terminator()) {
                  if (&top == theLoad.getOperation())
                    continue;
                  for (Value opd : top.getOperands())
                    if (!m0.contains(opd))
                      evalScalarIndexLane0(opd, m0);
                  if (failed(lowerSingleOp(top, 0, m0))) {
                    ok0 = false;
                    break;
                  }
                }
                if (ok0) {
                  Value ptr = m0.lookup(theLoad.getMemref());
                  SmallVector<Value> idxs;
                  for (Value idx : theLoad.getIndices())
                    idxs.push_back(m0.lookup(idx));
                  Value addr = makeAddr(b, loc, ptr, idxs, strides.asArrayRef());
                  Value loadedVec = loadContiguousSlice(
                      ptr, addr, vecElemType(resTy), batch);
                  writeSlot(vecSlot, loadedVec);
                }
              }
              {
                OpBuilder::InsertionGuard eg(b);
                b.setInsertionPointToStart(&fastIf.getElseRegion().front());
                Value anyCond =
                    makeOp(b, loc, "lane_any", {condVec}, {b.getI1Type()})
                        ->getResult(0);
                auto partialIf =
                    b.create<scf::IfOp>(loc, TypeRange{}, anyCond, false);
                OpBuilder::InsertionGuard pg(b);
                b.setInsertionPointToStart(&partialIf.getThenRegion().front());
                materializeScalarLanes(branch.getCondition(), lanes);
                materializeScalarLanes(branch.elseYield().getOperand(0), lanes);
                branch.walk([&](Operation *inner) {
                  for (Value operand : inner->getOperands())
                    if (!branch->isProperAncestor(
                            operand.getParentRegion()->getParentOp()))
                      materializeScalarLanes(operand, lanes);
                });
                Value elseScalar0 =
                    lanes[0].lookup(branch.elseYield().getOperand(0));
                SmallVector<Value, 16> scSlots;
                for (size_t l = 0; l < lanes.size(); ++l)
                  scSlots.push_back(
                      declSlot(resTy, elseScalar0,
                               branch.elseYield().getOperand(0)));
                for (size_t l = 0; l < lanes.size(); ++l) {
                  SmallVector<IRMapping, 1> single{lanes[l]};
                  auto lIf = b.create<scf::IfOp>(
                      loc, TypeRange{}, single[0].lookup(branch.getCondition()),
                      false);
                  OpBuilder::InsertionGuard lg(b);
                  b.setInsertionPointToStart(&lIf.getThenRegion().front());
                  if (succeeded(
                          lowerBlock(branch.getThenRegion().front(), single)))
                    writeSlot(scSlots[l],
                              single[0].lookup(branch.thenYield().getOperand(0)));
                }
                SmallVector<Value, 16> scVals;
                for (size_t l = 0; l < lanes.size(); ++l)
                  scVals.push_back(readSlot(scSlots[l]));
                StringRef dtStr = dtypeBare(vecElemType(resTy));
                Value packed =
                    makeOp(b, loc, "pack_apply", scVals, {batchVecType(resTy)},
                           {b.getNamedAttr("in_dtype", b.getStringAttr(dtStr)),
                            b.getNamedAttr("out_dtype", b.getStringAttr(dtStr)),
                            b.getNamedAttr("kind", b.getStringAttr("id"))})
                        ->getResult(0);
                writeSlot(vecSlot, packed);
              }
              batchVecs[branch.getResult(0)] = readSlot(vecSlot);
              continue;
            }
          }
        }
        materializeScalarLanes(branch.getCondition(), lanes);
        bool sharedCond = llvm::all_of(lanes, [&](IRMapping &m) {
          return m.lookup(branch.getCondition()) ==
                 lanes[0].lookup(branch.getCondition());
        });
        SmallVector<SmallVector<Value, 4>, 16> resSlots(lanes.size());
        SmallVector<Value, 4> vecResSlots(numRes, Value{});
        for (unsigned k = 0; k < numRes; ++k) {
          Value origRef = branch.thenYield().getOperand(k);
          Type resTy = branch.getResult(k).getType();
          if (isWideVal(origRef)) {
            wideBufValues.insert(branch.getResult(k));
            for (size_t l = 0; l < lanes.size(); ++l)
              resSlots[l].push_back(Value{});
            continue;
          }
          if (sharedCond && vecActive && !isa<ShapedType>(resTy) &&
              canBatchVecType(resTy) && !resTy.isInteger(1)) {
            vecResSlots[k] = declBatchVecSlot(batchVecType(resTy), Value{});
            for (size_t l = 0; l < lanes.size(); ++l)
              resSlots[l].push_back(Value{});
            continue;
          }
          for (size_t l = 0; l < lanes.size(); ++l)
            resSlots[l].push_back(declSlot(resTy, Value{}, origRef));
        }
        if (sharedCond) {
          auto newIf = b.create<scf::IfOp>(
              loc, TypeRange{}, lanes[0].lookup(branch.getCondition()), hasElse);
          if (branch->hasAttr("cpu.window_guard"))
            newIf->setAttr("mojo_row.window_guard", b.getUnitAttr());
          if (branch->hasAttr("cpu.window_branch"))
            newIf->setAttr("mojo_row.window_branch", b.getUnitAttr());
          {
            OpBuilder::InsertionGuard tg(b);
            auto &tBb = newIf.getThenRegion().front();
            b.setInsertionPointToStart(&tBb);
            SmallVector<IRMapping, 16> thenLanes(lanes.begin(), lanes.end());
            if (failed(lowerBlock(branch.getThenRegion().front(), thenLanes)))
              return failure();
            auto *tTerm = branch.getThenRegion().front().getTerminator();
            for (unsigned k = 0; k < numRes; ++k) {
              if (vecResSlots[k])
                writeSlot(vecResSlots[k],
                          getBatchVec(tTerm->getOperand(k), thenLanes));
              else if (resSlots[0][k])
                materializeScalarLanes(tTerm->getOperand(k), thenLanes);
            }
            for (size_t l = 0; l < thenLanes.size(); ++l)
              for (unsigned k = 0; k < numRes; ++k)
                if (resSlots[l][k])
                  writeSlot(resSlots[l][k],
                            thenLanes[l].lookup(tTerm->getOperand(k)));
          }
          if (hasElse) {
            OpBuilder::InsertionGuard eg(b);
            auto &eBb = newIf.getElseRegion().front();
            b.setInsertionPointToStart(&eBb);
            SmallVector<IRMapping, 16> elseLanes(lanes.begin(), lanes.end());
            if (failed(lowerBlock(branch.getElseRegion().front(), elseLanes)))
              return failure();
            auto *eTerm = branch.getElseRegion().front().getTerminator();
            for (unsigned k = 0; k < numRes; ++k) {
              if (vecResSlots[k])
                writeSlot(vecResSlots[k],
                          getBatchVec(eTerm->getOperand(k), elseLanes));
              else if (resSlots[0][k])
                materializeScalarLanes(eTerm->getOperand(k), elseLanes);
            }
            for (size_t l = 0; l < elseLanes.size(); ++l)
              for (unsigned k = 0; k < numRes; ++k)
                if (resSlots[l][k])
                  writeSlot(resSlots[l][k],
                            elseLanes[l].lookup(eTerm->getOperand(k)));
          }
        } else {
          branch.walk([&](Operation *inner) {
            for (Value operand : inner->getOperands())
              if (!branch->isProperAncestor(
                      operand.getParentRegion()->getParentOp()))
                materializeScalarLanes(operand, lanes);
          });
          for (size_t l = 0; l < lanes.size(); ++l) {
            SmallVector<IRMapping, 1> single{lanes[l]};
            auto newIf = b.create<scf::IfOp>(
                loc, TypeRange{}, single[0].lookup(branch.getCondition()),
                hasElse);
            {
              OpBuilder::InsertionGuard tg(b);
              b.setInsertionPointToStart(&newIf.getThenRegion().front());
              if (failed(lowerBlock(branch.getThenRegion().front(), single)))
                return failure();
              auto *tTerm = branch.getThenRegion().front().getTerminator();
              for (unsigned k = 0; k < numRes; ++k)
                if (resSlots[l][k])
                  writeSlot(resSlots[l][k],
                            single[0].lookup(tTerm->getOperand(k)));
            }
            if (hasElse) {
              OpBuilder::InsertionGuard eg(b);
              b.setInsertionPointToStart(&newIf.getElseRegion().front());
              if (failed(lowerBlock(branch.getElseRegion().front(), single)))
                return failure();
              auto *eTerm = branch.getElseRegion().front().getTerminator();
              for (unsigned k = 0; k < numRes; ++k)
                if (resSlots[l][k])
                  writeSlot(resSlots[l][k],
                            single[0].lookup(eTerm->getOperand(k)));
            }
          }
        }
        for (unsigned k = 0; k < numRes; ++k)
          if (vecResSlots[k])
            batchVecs[branch.getResult(k)] = readSlot(vecResSlots[k]);
        for (size_t l = 0; l < lanes.size(); ++l)
          for (unsigned k = 0; k < numRes; ++k)
            if (resSlots[l][k])
              lanes[l].map(branch.getResult(k), readSlot(resSlots[l][k]));
        continue;
      }

      // Batched chunked wide dot product: lane_sum(mulf(wideA, wideB))
      if (dotPanel && dotPanel->dot == &op) {
        auto loop = cast<scf::ForOp>(op.getParentOp());
        Value offset = makeAddr(
            b, loc, dots, {lanes[0].lookup(loop.getInductionVar())}, {batch});
        batchVecs[op.getResult(0)] =
            loadContiguousSlice(dots, offset, b.getF32Type(), batch);
        continue;
      }
      if (op.getName().stripDialect() == "lane_sum" &&
          isWideVal(op.getOperand(0))) {
        auto mul = op.getOperand(0).getDefiningOp<arith::MulFOp>();
        if (!mul)
          return op.emitError("expected wide mulf feeding lane_sum");
        Value lhsOrig = mul.getLhs(), rhsOrig = mul.getRhs();
        int64_t kExt = memberLen(lhsOrig);
        bool rhsCommon = llvm::all_of(
            llvm::iota_range<size_t>(0, lanes.size(), false), [&](size_t l) {
              return wideViews[l][rhsOrig].baseAddr ==
                     wideViews[0][rhsOrig].baseAddr;
            });
        bool lhsCommon = llvm::all_of(
            llvm::iota_range<size_t>(0, lanes.size(), false), [&](size_t l) {
              return wideViews[l][lhsOrig].baseAddr ==
                     wideViews[0][lhsOrig].baseAddr;
            });
        auto f32x16 = RankedTensorType::get({16}, b.getF32Type());
        SmallVector<Value, 16> scalarDots;
        for (size_t sub = 0; sub < lanes.size(); sub += 16) {
          size_t subEnd = std::min(lanes.size(), sub + 16);
          size_t subLen = subEnd - sub;
          SmallVector<Value, 16> accVecs(subLen);
          if (kExt <= 128) {
            for (int64_t cOff = 0; cOff < kExt; cOff += 16) {
              Value cIdx = constIdx(b, loc, cOff);
              Value sharedRhs =
                  rhsCommon ? loadWideChunkF32(wideViews[0][rhsOrig], cIdx, 16)
                            : Value{};
              Value sharedLhs =
                  (!rhsCommon && lhsCommon)
                      ? loadWideChunkF32(wideViews[0][lhsOrig], cIdx, 16)
                      : Value{};
              for (size_t l = sub; l < subEnd; ++l) {
                Value lv =
                    sharedLhs ? sharedLhs
                              : loadWideChunkF32(wideViews[l][lhsOrig], cIdx, 16);
                Value rv =
                    sharedRhs ? sharedRhs
                              : loadWideChunkF32(wideViews[l][rhsOrig], cIdx, 16);
                Value prod = b.create<arith::MulFOp>(loc, lv, rv);
                accVecs[l - sub] =
                    (cOff == 0)
                        ? prod
                        : b.create<arith::AddFOp>(loc, accVecs[l - sub], prod);
              }
            }
          } else {
            SmallVector<Value, 16> accSlots;
            for (size_t l = sub; l < subEnd; ++l) {
              accSlots.push_back(
                  makeOp(b, loc, "acc_decl", {}, {f32x16},
                         {b.getNamedAttr("dtype", b.getStringAttr("float32")),
                          b.getNamedAttr("width", b.getI64IntegerAttr(16))})
                      ->getResult(0));
            }
            auto *cLoop = makeOp(
                b, loc, "for_range",
                {constIdx(b, loc, 0), constIdx(b, loc, kExt),
                 constIdx(b, loc, 16)},
                {}, {}, 1);
            Block &cBb = cLoop->getRegion(0).emplaceBlock();
            Value cIdx = cBb.addArgument(b.getIndexType(), loc);
            {
              OpBuilder::InsertionGuard cg(b);
              b.setInsertionPointToEnd(&cBb);
              Value sharedRhs =
                  rhsCommon ? loadWideChunkF32(wideViews[0][rhsOrig], cIdx, 16)
                            : Value{};
              Value sharedLhs =
                  (!rhsCommon && lhsCommon)
                      ? loadWideChunkF32(wideViews[0][lhsOrig], cIdx, 16)
                      : Value{};
              for (size_t l = sub; l < subEnd; ++l) {
                Value lv =
                    sharedLhs ? sharedLhs
                              : loadWideChunkF32(wideViews[l][lhsOrig], cIdx, 16);
                Value rv =
                    sharedRhs ? sharedRhs
                              : loadWideChunkF32(wideViews[l][rhsOrig], cIdx, 16);
                Value prod = b.create<arith::MulFOp>(loc, lv, rv);
                Value sum = b.create<arith::AddFOp>(
                    loc, readSlot(accSlots[l - sub]), prod);
                writeSlot(accSlots[l - sub], sum);
              }
              makeOp(b, loc, "end");
            }
            for (size_t l = sub; l < subEnd; ++l)
              accVecs[l - sub] = readSlot(accSlots[l - sub]);
          }
          if (vecActive && lanes.size() == 16 && subLen == 16) {
            usesReduce16x16 = true;
            batchVecs[op.getResult(0)] =
                makeOp(b, loc, "reduce_sum_16x16", accVecs, {f32x16},
                       {b.getNamedAttr(
                           "ci.implementation",
                           b.getStringAttr("exits/mojo::simd-reduce-16x16"))})
                    ->getResult(0);
          } else {
            for (size_t l = sub; l < subEnd; ++l) {
              Value dot =
                  makeOp(b, loc, "lane_sum", {accVecs[l - sub]},
                         {b.getF32Type()})
                      ->getResult(0);
              lanes[l].map(op.getResult(0), dot);
              scalarDots.push_back(dot);
            }
          }
        }
        if (vecActive && lanes.size() < 16 &&
            scalarDots.size() == lanes.size()) {
          batchVecs[op.getResult(0)] =
              makeOp(b, loc, "pack_apply", scalarDots,
                     {RankedTensorType::get({batch}, b.getF32Type())},
                     {b.getNamedAttr("in_dtype", b.getStringAttr("float32")),
                      b.getNamedAttr("out_dtype", b.getStringAttr("float32")),
                      b.getNamedAttr("kind", b.getStringAttr("id"))})
                  ->getResult(0);
        }
        continue;
      }

      // Batched chunked in-place wide accumulation:
      // addf(accBuf | mulf(accBuf, splat(alpha)), mulf(view, splat(beta)))
      if (auto add = dyn_cast<arith::AddFOp>(op);
          add && isWideVal(add.getResult())) {
        auto isAccTerm = [&](Value v, Value &alphaOrig) -> bool {
          if (wideBufValues.contains(v))
            return true;
          if (auto m = v.getDefiningOp<arith::MulFOp>()) {
            if (wideBufValues.contains(m.getLhs()) &&
                hasWideSplat(0, m.getRhs())) {
              alphaOrig = m.getRhs();
              return true;
            }
            if (wideBufValues.contains(m.getRhs()) &&
                hasWideSplat(0, m.getLhs())) {
              alphaOrig = m.getLhs();
              return true;
            }
          }
          return false;
        };
        Value alphaOrig;
        Value accSide = isAccTerm(add.getLhs(), alphaOrig) ? add.getLhs()
                                                           : add.getRhs();
        if (!wideBufValues.contains(accSide) && !alphaOrig)
          return op.emitError("unsupported wide addf accumulator operand");
        Value mulSide = accSide == add.getLhs() ? add.getRhs() : add.getLhs();
        auto mul = mulSide.getDefiningOp<arith::MulFOp>();
        if (!mul)
          return op.emitError("unsupported wide addf value operand");
        Value viewOrig =
            wideViews[0].count(mul.getLhs()) ? mul.getLhs() : mul.getRhs();
        Value betaOrig =
            viewOrig == mul.getLhs() ? mul.getRhs() : mul.getLhs();
        if (inTiledReplayPassA) {
          auto loop = op.getParentOfType<scf::ForOp>();
          Value lb = lanes[0].lookup(loop.getLowerBound());
          Value iv = lanes[0].lookup(loop.getInductionVar());
          Value off = b.create<arith::SubIOp>(loc, iv, lb);
          if (valuePanel && op.hasAttr("cpu.pv_engine")) {
            auto *splat = betaOrig.getDefiningOp();
            auto ext = splat->getOperand(0).getDefiningOp<arith::ExtFOp>();
            auto raw = getBatchVec(ext.getIn(), lanes);
            Value offset = makeAddr(b, loc, tokenWeights, {off}, {batch});
            makeOp(b, loc, "store_slice", {tokenWeights, offset, raw});
          } else if (vecActive && wideBatchSplats.count(betaOrig)) {
            Value wAddr = makeAddr(b, loc, localStackBuf, {off}, {batch});
            makeOp(b, loc, "store_slice",
                   {localStackBuf, wAddr, wideBatchSplats[betaOrig]});
          } else {
            for (size_t l = 0; l < lanes.size(); ++l) {
              Value wAddr = makeAddr(b, loc, localStackBuf,
                                     {off, constIdx(b, loc, l)}, {batch, 1});
              makeOp(b, loc, "store_scalar",
                     {localStackBuf, wAddr, getWideSplatScalar(l, betaOrig)});
            }
          }
          wideBufValues.insert(add.getResult());
          continue;
        }
        int64_t kExt = memberLen(add.getResult());
        bool viewCommon = llvm::all_of(
            llvm::iota_range<size_t>(0, lanes.size(), false), [&](size_t l) {
              return wideViews[l][viewOrig].baseAddr ==
                     wideViews[0][viewOrig].baseAddr;
            });
        auto f32x16 = RankedTensorType::get({16}, b.getF32Type());
        size_t subStep = alphaOrig ? 8 : 16;
        for (size_t sub = 0; sub < lanes.size(); sub += subStep) {
          size_t subEnd = std::min(lanes.size(), sub + subStep);
          SmallVector<Value, 16> alphaVecs, betaVecs;
          for (size_t l = sub; l < subEnd; ++l) {
            if (alphaOrig) {
              alphaVecs.push_back(
                  makeOp(b, loc, "splat_scalar",
                         {getWideSplatScalar(l, alphaOrig)}, {f32x16},
                         {b.getNamedAttr("dtype", b.getStringAttr("float32")),
                          b.getNamedAttr("width", b.getI64IntegerAttr(16))})
                      ->getResult(0));
            }
            betaVecs.push_back(
                makeOp(b, loc, "splat_scalar", {getWideSplatScalar(l, betaOrig)},
                       {f32x16},
                       {b.getNamedAttr("dtype", b.getStringAttr("float32")),
                        b.getNamedAttr("width", b.getI64IntegerAttr(16))})
                    ->getResult(0));
          }
          auto *cLoop = makeOp(
              b, loc, "for_range",
              {constIdx(b, loc, 0), constIdx(b, loc, kExt),
               constIdx(b, loc, 16)},
              {}, {}, 1);
          Block &cBb = cLoop->getRegion(0).emplaceBlock();
          Value cIdx = cBb.addArgument(b.getIndexType(), loc);
          {
            OpBuilder::InsertionGuard cg(b);
            b.setInsertionPointToEnd(&cBb);
            Value sharedV =
                viewCommon ? loadWideChunkF32(wideViews[0][viewOrig], cIdx, 16)
                           : Value{};
            for (size_t l = sub; l < subEnd; ++l) {
              Value v16 =
                  sharedV ? sharedV
                          : loadWideChunkF32(wideViews[l][viewOrig], cIdx, 16);
              Value bufAddr = makeAddr(
                  b, loc, wideAccBuf,
                  {constIdx(b, loc, int64_t(l) * accStride), cIdx}, {1, 1});
              Value cur16 =
                  loadContiguousSlice(wideAccBuf, bufAddr, b.getF32Type(), 16);
              if (alphaOrig)
                cur16 = b.create<arith::MulFOp>(loc, cur16, alphaVecs[l - sub]);
              Value prod = b.create<arith::MulFOp>(loc, v16, betaVecs[l - sub]);
              Value upd = b.create<arith::AddFOp>(loc, cur16, prod);
              makeOp(b, loc, "store_slice", {wideAccBuf, bufAddr, upd});
            }
            makeOp(b, loc, "end");
          }
        }
        wideBufValues.insert(add.getResult());
        continue;
      }

      if (tryLowerBatchVecOp(op, lanes))
        continue;

      for (Value operand : op.getOperands())
        materializeScalarLanes(operand, lanes);

      if (batch > 1 && op.hasAttr("mojo_row.pack_scalars")) {
        SmallVector<Value, 16> scalars;
        for (auto &m : lanes)
          scalars.push_back(m.lookup(op.getOperand(0)));
        Type inTy = op.getOperand(0).getType();
        Type outTy = op.getResult(0).getType();
        auto packTy = RankedTensorType::get({batch}, outTy);
        Value packed =
            makeOp(b, loc, "pack_apply", scalars, {packTy},
                   {b.getNamedAttr("in_dtype", b.getStringAttr(dtypeBare(inTy))),
                    b.getNamedAttr("out_dtype",
                                   b.getStringAttr(dtypeBare(outTy))),
                    b.getNamedAttr("kind",
                                   b.getStringAttr(op.getName().stripDialect()))})
                ->getResult(0);
        for (size_t l = 0; l < lanes.size(); ++l) {
          Value laneVal =
              makeOp(b, loc, "extract_lane", {packed}, {outTy},
                     {b.getNamedAttr("lane", b.getI64IntegerAttr(l))})
                  ->getResult(0);
          lanes[l].map(op.getResult(0), laneVal);
        }
        continue;
      }

      for (size_t l = 0; l < lanes.size(); ++l) {
        if (l > 0 && (op.hasAttr("cpu.owner_common") ||
                      isa<arith::ConstantOp, arith::ConstantIndexOp>(op))) {
          if (op.getNumResults() == 1 && isWideVal(op.getResult(0))) {
            if (wideViews[0].count(op.getResult(0)))
              wideViews[l][op.getResult(0)] = wideViews[0][op.getResult(0)];
            if (wideSplats[0].count(op.getResult(0)))
              wideSplats[l][op.getResult(0)] = wideSplats[0][op.getResult(0)];
          } else {
            lanes[l].map(op.getResult(0), lanes[0].lookup(op.getResult(0)));
          }
          continue;
        }
        if (failed(lowerSingleOp(op, l, lanes[l])))
          return failure();
      }

      // Pre-expand top-level per-owner wide slices once into L1d f32 invPrepackBuf
      if (invReadOffset.count(&op)) {
        Value origVec = op.getResult(0);
        int64_t kExt = memberLen(origVec);
        int64_t invOff = invReadOffset[&op];
        if (kExt <= 128) {
          for (int64_t cOff = 0; cOff < kExt; cOff += 16) {
            Value cIdx = constIdx(b, loc, cOff);
            for (size_t l = 0; l < lanes.size(); ++l) {
              Value v16 = loadWideChunkF32(wideViews[l][origVec], cIdx, 16);
              Value dstAddr = makeAddr(
                  b, loc, invPrepackBuf,
                  {constIdx(b, loc, int64_t(l) * invStride + invOff + cOff)},
                  {1});
              makeOp(b, loc, "store_slice", {invPrepackBuf, dstAddr, v16});
            }
          }
        } else {
          auto *pLoop = makeOp(
              b, loc, "for_range",
              {constIdx(b, loc, 0), constIdx(b, loc, kExt),
               constIdx(b, loc, 16)},
              {}, {}, 1);
          Block &pBb = pLoop->getRegion(0).emplaceBlock();
          Value cIdx = pBb.addArgument(b.getIndexType(), loc);
          {
            OpBuilder::InsertionGuard pg(b);
            b.setInsertionPointToEnd(&pBb);
            for (size_t l = 0; l < lanes.size(); ++l) {
              Value v16 = loadWideChunkF32(wideViews[l][origVec], cIdx, 16);
              Value dstAddr = makeAddr(
                  b, loc, invPrepackBuf,
                  {constIdx(b, loc, int64_t(l) * invStride + invOff), cIdx},
                  {1, 1});
              makeOp(b, loc, "store_slice", {invPrepackBuf, dstAddr, v16});
            }
            makeOp(b, loc, "end");
          }
        }
        for (size_t l = 0; l < lanes.size(); ++l) {
          wideViews[l][origVec] = WideViewRef{
              invPrepackBuf,
              constIdx(b, loc, int64_t(l) * invStride + invOff),
              b.getF32Type(), kExt};
        }
      }
    }
    return success();
  }
};
} // namespace

llvm::Error describeMojoOrderedPackage(func::FuncOp fn, Operation &prog,
                                       llvm::raw_pwrite_stream &out) {
  auto abi = llvm::json::parse(fn->getAttrOfType<StringAttr>("kir.abi"));
  if (!abi)
    return abi.takeError();
  int64_t grain = prog.getAttrOfType<IntegerAttr>("grain").getInt();
  out << llvm::json::Value(llvm::json::Object{
      {"version", 2},
      {"fp_contract", fn->getAttrOfType<StringAttr>("mojo_row.fp_contract").str()},
      {"abi", std::move(*abi)},
      {"domain", intentdsl::publishDomain(fn)},
      {"rows",
       llvm::json::Array{llvm::json::Object{{"grain", grain}, {"columns", 1}}}},
      {"returns_status", true},
      {"onednn", prog.hasAttr("brgemm")},
      {"source", ""},
      {"host", ""}});
  return llvm::Error::success();
}

#include "FoldProducts.h"

LogicalResult lowerMojoOrdered(func::FuncOp fn,
                               const intentdsl::CallFacts &facts) {
  auto &task = fn.front().front();
  auto &space = task.getRegion(0).front().front();
  unrollShortWindow(space);
  Location loc = space.getLoc();
  OpBuilder b(&fn.front().front());

  int64_t owners = space.getAttrOfType<IntegerAttr>("cpu.owners").getInt();
  int64_t grain = space.getAttrOfType<IntegerAttr>("cpu.grain").getInt();
  int64_t batch = 1;
  if (auto bAttr = space.getAttrOfType<IntegerAttr>("cpu.owner_batch"))
    batch = bAttr.getInt();
  auto windowAttr = space.getAttrOfType<IntegerAttr>("cpu.batch_score_window");

  auto foldTile = space.getAttrOfType<DenseI64ArrayAttr>("cpu.fold_panel");
  auto foldPlan =
      foldTile ? intentdsl::cpu::foldProducts(space, fn, facts) : std::nullopt;
  if (foldTile && !foldPlan)
    return space.emitError("invalid fold panel");
  if (foldTile)
    batch = foldTile[0];
  Operation *foundLocalBuf = nullptr;
  for (Operation &op : space.getRegion(0).front()) {
    if (op.getName().getStringRef() == "cpu.local_buffer") {
      foundLocalBuf = &op;
      break;
    }
  }

  bool hasVariableOwnerTrip =
      space
          .walk([&](scf::ForOp loop) {
            auto isOwnerInvariant = [&](Value v) {
              if (!v)
                return true;
              if (auto *def = v.getDefiningOp()) {
                if (isa<arith::ConstantOp, arith::ConstantIndexOp>(def))
                  return true;
                if (!space.isProperAncestor(def))
                  return true;
                return false;
              }
              return !space.getRegion(0).isAncestor(v.getParentRegion());
            };
            if (!isOwnerInvariant(loop.getLowerBound()) ||
                !isOwnerInvariant(loop.getUpperBound()))
              return WalkResult::interrupt();
            return WalkResult::advance();
          })
          .wasInterrupted();
  if (hasVariableOwnerTrip && owners >= 2 * batch) {
    int64_t targetGrain = std::max<int64_t>(batch, owners / 64);
    targetGrain = ((targetGrain + batch - 1) / batch) * batch;
    grain = std::min(grain, targetGrain);
  }
  bool useCyclic = hasVariableOwnerTrip && owners > grain;

  OrderedLowerer lowerer{fn, space, b, loc, batch};
  lowerer.batchSimd =
      !foldTile && batch >= 2 && llvm::isPowerOf2_64(uint64_t(batch));
  if (!foldTile)
    lowerer.analyzeChunkableWideMode();
  auto innerAxis = space.getAttrOfType<IntegerAttr>("cpu.innermost_owner");
  auto panel = innerAxis ? intentdsl::cpu::orderedDotPanel(
                               space, facts, innerAxis.getInt(), batch)
                         : std::nullopt;
  if (panel && panel->dot->hasAttr("cpu.panel_engine") && lowerer.chunkedMode)
    lowerer.dotPanel = panel;
  bool brg = bool(lowerer.dotPanel) || foldPlan.has_value();
  if (brg && space.hasAttr("cpu.value_width")) {
    auto value = intentdsl::cpu::orderedValuePanel(space, facts, *panel);
    if (!value)
      return space.emitError("weighted contraction has no proven value view");
    lowerer.valuePanel = value;
  }
  bool pv = bool(lowerer.valuePanel);
  if (space.hasAttr("cpu.value_width") && !pv)
    return space.emitError("selected value panel requires wide owner replay");
  lowerer.wideViews.resize(batch);
  lowerer.wideSplats.resize(batch);
  bool usesStack = bool(foldTile) || foundLocalBuf != nullptr ||
                   lowerer.invFloatsPerLane > 0 || lowerer.accExtent > 0;

  OperationState progState(loc, "mojo_row.ordered_program");
  progState.addAttribute("grain", b.getI64IntegerAttr(grain));
  if (windowAttr) {
    progState.addAttribute("owner_batch", b.getI64IntegerAttr(batch));
    progState.addAttribute("window_slots", windowAttr);
  }
  progState.addRegion();
  progState.addRegion();
  Operation *prog = b.create(progState);

  // Region 0: row.mojo
  Block &kOuter = prog->getRegion(0).emplaceBlock();
  b.setInsertionPointToEnd(&kOuter);
  SmallVector<Attribute> mojoParamTypes;
  SmallVector<std::string> externCParams, callArgs;
  for (auto arg : fn.getArguments()) {
    std::string aName = "a" + std::to_string(arg.getArgNumber());
    callArgs.push_back(aName);
    if (auto view = dyn_cast<MemRefType>(arg.getType())) {
      mojoParamTypes.push_back(
          b.getStringAttr("Pointer[" + scalarTypeStr(view.getElementType()) +
                          ",MutUntrackedOrigin]"));
      externCParams.push_back("void*");
    } else {
      mojoParamTypes.push_back(b.getStringAttr(scalarTypeStr(arg.getType())));
      externCParams.push_back(arg.getType().isF32() ? "float" : "int32_t");
    }
  }

  auto *ownerFn =
      makeOp(b, loc, "owner_fn", {}, {},
             {b.getNamedAttr("param_types", b.getArrayAttr(mojoParamTypes)),
              b.getNamedAttr("uses_stack", b.getBoolAttr(usesStack)),
              b.getNamedAttr("owners", b.getI64IntegerAttr(owners))},
             1);
  makeOp(b, loc, "end");

  Block &fnBb = ownerFn->getRegion(0).emplaceBlock();
  SmallVector<Value> paramVals;
  for (auto arg : fn.getArguments())
    paramVals.push_back(fnBb.addArgument(arg.getType(), loc));
  Value loArg = fnBb.addArgument(b.getI64Type(), loc);
  Value hiArg = fnBb.addArgument(b.getI64Type(), loc);
  if (brg) {
    prog->setAttr("brgemm", b.getUnitAttr());
    ownerFn->setAttr("brgemm", b.getUnitAttr());
    lowerer.brgHandle = fnBb.addArgument(b.getIndexType(), loc);
    lowerer.brgScratch = fnBb.addArgument(b.getIndexType(), loc);
    if (pv) {
      ownerFn->setAttr("packed_values", b.getUnitAttr());
      lowerer.packedValues = fnBb.addArgument(b.getIndexType(), loc);
    }
  }

  b.setInsertionPointToEnd(&fnBb);

  if (brg && !foldTile) {
    auto buffer = [&](StringRef dtype, int64_t count) {
      return makeOp(b, loc, "stack_alloc", {}, {b.getIndexType()},
                    {b.getNamedAttr("dtype", b.getStringAttr(dtype)),
                     b.getNamedAttr("count", b.getI64IntegerAttr(count))})
          ->getResult(0);
    };
    lowerer.packedQueries = buffer("bfloat16", batch * panel->depth);
    lowerer.dots = buffer("float32", batch * panel->tokens);
    if (pv) {
      lowerer.weights = buffer("bfloat16", batch * panel->tokens);
      lowerer.tokenWeights = buffer("bfloat16", batch * panel->tokens);
    }
  }

  if (foundLocalBuf) {
    int64_t slots =
        foundLocalBuf->getAttrOfType<IntegerAttr>("elements").getInt();
    // Pad stack buffers by odd cache-line multiples to prevent 4 KiB page-offset
    // memory disambiguation aliasing across localStackBuf, vAddrBuf, and wideAccBuf.
    int64_t paddedScoreCount = batch * slots + ((batch * slots * 4) % 4096 == 0 ? 32 : 0);
    SmallVector<NamedAttribute, 3> scoreAllocAttrs{
        b.getNamedAttr("dtype", b.getStringAttr("float32")),
        b.getNamedAttr("count", b.getI64IntegerAttr(paddedScoreCount))};
    if (auto roleAttr = foundLocalBuf->getAttrOfType<StringAttr>("role"))
      scoreAllocAttrs.push_back(b.getNamedAttr("role", roleAttr));
    lowerer.localStackBuf =
        makeOp(b, loc, "stack_alloc", {}, {b.getIndexType()}, scoreAllocAttrs)
            ->getResult(0);
    if (!pv && lowerer.chunkedMode && lowerer.accExtent >= 32 &&
        lowerer.accExtent % 32 == 0) {
      int64_t paddedAddrCount = slots + ((slots * 8) % 4096 == 0 ? 24 : 0);
      SmallVector<NamedAttribute, 3> addrAllocAttrs{
          b.getNamedAttr("dtype", b.getStringAttr("int64")),
          b.getNamedAttr("count", b.getI64IntegerAttr(paddedAddrCount))};
      if (windowAttr)
        addrAllocAttrs.push_back(
            b.getNamedAttr("role", b.getStringAttr("window_addrs")));
      lowerer.vAddrBuf =
          makeOp(b, loc, "stack_alloc", {}, {b.getIndexType()}, addrAllocAttrs)
              ->getResult(0);
    }
  }
  if (lowerer.invStride > 0) {
    lowerer.invPrepackBuf =
        makeOp(b, loc, "stack_alloc", {}, {b.getIndexType()},
               {b.getNamedAttr("dtype", b.getStringAttr("float32")),
                b.getNamedAttr(
                    "count",
                    b.getI64IntegerAttr(batch * lowerer.invStride))})
            ->getResult(0);
  }
  if (lowerer.accStride > 0) {
    lowerer.wideAccBuf =
        makeOp(b, loc, "stack_alloc", {}, {b.getIndexType()},
               {b.getNamedAttr("dtype", b.getStringAttr("float32")),
                b.getNamedAttr("count",
                               b.getI64IntegerAttr(batch * lowerer.accStride))})
            ->getResult(0);
  }

  Value loInt = b.create<arith::IndexCastOp>(loc, b.getIndexType(), loArg);
  Value hiInt = b.create<arith::IndexCastOp>(loc, b.getIndexType(), hiArg);
  auto *ownerLoop = makeOp(
      b, loc, "for_range", {loInt, hiInt, constIdx(b, loc, batch)}, {}, {}, 1);
  Block &oBb = ownerLoop->getRegion(0).emplaceBlock();
  Value owner = oBb.addArgument(b.getIndexType(), loc);

  b.setInsertionPointToEnd(&oBb);
  SmallVector<IRMapping, 16> lanes(batch);
  for (auto [origArg, newParam] : llvm::zip(fn.getArguments(), paramVals))
    for (auto &m : lanes)
      m.map(origArg, newParam);

  auto &srcBody = space.getRegion(0).front();
  auto axes = space.getAttrOfType<DenseI64ArrayAttr>("axes");
  auto innerAttr = space.getAttrOfType<IntegerAttr>("cpu.innermost_owner");
  int64_t pitch = 1;
  for (unsigned ax : intentdsl::cpu::ownerRadices(space)) {
    int64_t ext = std::max<int64_t>(
        1, lowerer.argArray(space.getOperand(ax), "cpu.shape")[axes[ax]]);
    Value base =
        pitch == 1
            ? owner
            : (llvm::isPowerOf2_64(uint64_t(pitch))
                   ? b.create<arith::ShRUIOp>(
                          loc, owner,
                          constIdx(b, loc, llvm::Log2_64(uint64_t(pitch))))
                         .getResult()
                   : b.create<arith::DivSIOp>(loc, owner, constIdx(b, loc, pitch))
                         .getResult());
    Value coord =
        ext <= 1
            ? constIdx(b, loc, 0)
            : (llvm::isPowerOf2_64(uint64_t(ext))
                   ? b.create<arith::AndIOp>(loc, base, constIdx(b, loc, ext - 1))
                         .getResult()
                   : b.create<arith::RemSIOp>(loc, base, constIdx(b, loc, ext))
                         .getResult());
    for (int64_t l = 0; l < batch; ++l) {
      Value laneCoord = coord;
      if (batch > 1 && pitch == 1 && innerAttr &&
          unsigned(innerAttr.getInt()) == ax && l > 0)
        laneCoord = addIdx(b, loc, coord, l);
      lanes[l].map(srcBody.getArgument(ax), laneCoord);
    }
    if (lowerer.batchSimd && pitch == 1 && innerAttr &&
        unsigned(innerAttr.getInt()) == ax) {
      lowerer.innerOwnerArg = srcBody.getArgument(ax);
      Value baseI64 = b.create<arith::IndexCastOp>(loc, b.getI64Type(), coord);
      auto i64VecTy = RankedTensorType::get({batch}, b.getI64Type());
      lowerer.batchVecs[srcBody.getArgument(ax)] =
          makeOp(b, loc, "simd_iota_add", {baseI64}, {i64VecTy},
                 {b.getNamedAttr("dtype", b.getStringAttr("int64")),
                  b.getNamedAttr("width", b.getI64IntegerAttr(batch))})
              ->getResult(0);
    }
    pitch *= ext;
  }

  if (failed(foldTile ? lowerFoldProducts(lowerer, srcBody, lanes, *foldPlan,
                                          foldTile[1], ownerFn)
                      : lowerer.lowerBlock(srcBody, lanes)))
    return failure();
  if (pv && !lowerer.emittedValuePanel)
    return space.emitError(
        "selected matrix value contraction has no matching replay");
  makeOp(b, loc, "end");
  b.setInsertionPointToEnd(&fnBb);
  if (brg)
    makeOp(b, loc, "panel_return", {}, {},
           {b.getNamedAttr("code", b.getI32IntegerAttr(0))});
  makeOp(b, loc, "end");
  if (lowerer.usesReduce16x16) {
    OpBuilder::InsertionGuard rg(b);
    b.setInsertionPointToStart(&kOuter);
    makeOp(b, loc, "runtime_source", {}, {},
           {b.getNamedAttr("implementation",
                           b.getStringAttr("exits/mojo::simd-reduce-16x16"))});
  }

  // Region 1: entry.c (structured C host function + OpenMP loop + kernel call)
  Block &hBb = prog->getRegion(1).emplaceBlock();
  b.setInsertionPointToEnd(&hBb);
  std::string externDecl =
      "extern void owner_task(" + llvm::join(externCParams, ",") +
      (externCParams.empty() ? "" : ",") + "int64_t,int64_t);";
  if (brg)
    externDecl = "extern int owner_task(" + llvm::join(externCParams, ",") +
                 ",int64_t,int64_t,const void*,void*" +
                 (pv ? ",const void*" : "") +
                 ");\nextern const void "
                 "*intent_brgemm_get(int64_t,int64_t,int64_t,int64_t,int64_t,"
                 "int64_t,int32_t,int32_t,int32_t);\n"
                 "extern int64_t intent_brgemm_scratch_bytes(const void*);\n"
                 "extern int32_t intent_brgemm_thread_enter(const "
                 "void*);\nextern void intent_brgemm_thread_leave(void);";
  if (pv) {

    externDecl +=
        "\n#include <stdlib.h>\nextern void intent_pack_bf16_rows(const "
        "uint16_t*,uint16_t*,int64_t,int64_t,int64_t);";
  }
  if (pv && space.hasAttr("cpu.worker_output_ftz"))
    externDecl += "\n#include <xmmintrin.h>";
  auto *cFn = makeOp(
      b, loc, "c_host_fn", {}, {},
      {b.getNamedAttr("params",
                      b.getStringAttr(intentdsl::cpu::matrixHostParams(fn))),
       b.getNamedAttr("extern_decl", b.getStringAttr(externDecl))},
      1);
  makeOp(b, loc, "end");

  Block &cFnBb = cFn->getRegion(0).emplaceBlock();
  b.setInsertionPointToEnd(&cFnBb);
  if (brg) {
    cFn->setAttr("brgemm", b.getUnitAttr());
    makeOp(b, loc, "owner_brgemm_plan", {}, {},
           {b.getNamedAttr(
                "rows", b.getI64IntegerAttr(foldTile ? batch : panel->tokens)),
            b.getNamedAttr("cols",
                           b.getI64IntegerAttr(foldTile ? foldTile[1] : batch)),
            b.getNamedAttr(
                "dtype",
                b.getI64IntegerAttr(foldPlan && !foldPlan->nativeBf16 ? 2 : 1)),
            b.getNamedAttr("depth",
                           b.getI64IntegerAttr(foldPlan ? foldPlan->depth
                                                        : panel->depth))});
  }
  if (pv) {
    auto &value = *lowerer.valuePanel;
    cFn->setAttr("packed_values", b.getUnitAttr());
    auto arg = cast<BlockArgument>(value.read->getOperand(0));
    makeOp(b, loc, "pack_value_table", {}, {},
           {b.getNamedAttr("ci.implementation",
                           b.getStringAttr("exits/mojo::pack-bf16-rows")),
            b.getNamedAttr("slot", b.getI64IntegerAttr(arg.getArgNumber())),
            b.getNamedAttr("rows",
                           b.getI64IntegerAttr(value.elements / value.columns)),
            b.getNamedAttr("columns", b.getI64IntegerAttr(value.columns)),
            b.getNamedAttr("plane_rows", b.getI64IntegerAttr(value.tokens))});
  }
  auto *ompFor = makeOp(
      b, loc, "c_omp_for", {}, {},
      {b.getNamedAttr("count", b.getI64IntegerAttr(owners)),
       b.getNamedAttr("grain", b.getI64IntegerAttr(grain)),
       b.getNamedAttr("cyclic", b.getBoolAttr(useCyclic))},
      1);
  makeOp(b, loc, "end");

  Block &ompBb = ompFor->getRegion(0).emplaceBlock();
  if (brg)
    ompFor->setAttr("brgemm", b.getUnitAttr());
  b.setInsertionPointToEnd(&ompBb);
  auto *invoke = makeOp(
      b, loc, "c_owner_call", {}, {},
      {b.getNamedAttr("call_args", b.getStringAttr(llvm::join(callArgs, ","))),
       b.getNamedAttr("cyclic", b.getBoolAttr(useCyclic)),
       b.getNamedAttr("count", b.getI64IntegerAttr(owners)),
       b.getNamedAttr("grain", b.getI64IntegerAttr(grain))});
  if (brg)
    invoke->setAttr("brgemm", b.getUnitAttr());
  if (pv)
    invoke->setAttr("packed_values", b.getUnitAttr());
  if (pv && space.hasAttr("cpu.worker_output_ftz"))
    invoke->setAttr("flush_output_denormals", b.getUnitAttr());
  makeOp(b, loc, "end");

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

bool hasMojoOrderedProgram(ModuleOp m) {
  auto funcs = m.getOps<func::FuncOp>();
  if (!llvm::hasSingleElement(funcs))
    return false;
  auto fn = *funcs.begin();
  return !fn.empty() && !fn.front().empty() &&
         fn.front().front().getName().getStringRef() ==
             "mojo_row.ordered_program";
}

LogicalResult verifyMojoOrdered(ModuleOp m, const TargetProfile &p,
                                bool callable) {
  auto fn = *m.getOps<func::FuncOp>().begin();
  auto &prog = fn.front().front();
  if (prog.getNumRegions() != 2 || prog.getRegion(0).empty() ||
      prog.getRegion(1).empty())
    return prog.emitError("incomplete mojo_row.ordered_program regions");

  if (prog.walk([&](Operation *op) {
            if (!op->hasAttr("flush_output_denormals"))
              return WalkResult::advance();
            return op->getName().getStringRef() == "mojo_row.c_owner_call" &&
                           op->hasAttr("packed_values") &&
                           intentdsl::cpu::allowsWorkerOutputFtz(prog, p)
                       ? WalkResult::advance()
                       : WalkResult::interrupt();
          })
          .wasInterrupted())
    return prog.emitError(
        "worker FTZ requires a common signed-output flush mode");

  // Independently verify windowed score buffer, replay, and over-window
  // zero-read invariants on resume (parity with CPU verifyOwnerBuffers).
  SmallVector<Operation *, 2> guards, branches;
  SmallVector<Value, 2> winBufs, addrBufs;
  m.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "mojo_row.if") {
      if (op->hasAttr("mojo_row.window_guard"))
        guards.push_back(op);
      if (op->hasAttr("mojo_row.window_branch"))
        branches.push_back(op);
    } else if (op->getName().getStringRef() == "mojo_row.stack_alloc") {
      if (auto role = op->getAttrOfType<StringAttr>("role")) {
        if (role.getValue() == "score_window")
          winBufs.push_back(op->getResult(0));
        else if (role.getValue() == "window_addrs")
          addrBufs.push_back(op->getResult(0));
      }
    }
  });
  auto winAttr = prog.getAttrOfType<IntegerAttr>("window_slots");
  auto batchAttr = prog.getAttrOfType<IntegerAttr>("owner_batch");
  if (winAttr || batchAttr || !guards.empty() || !branches.empty() ||
      !winBufs.empty() || !addrBufs.empty()) {
    if (!winAttr || !batchAttr || batchAttr.getInt() < 2 ||
        winAttr.getInt() <= 0 ||
        winAttr.getInt() > intentdsl::cpu::scoreSlotLimit(
                               *p.services.get<intentdsl::cpu::RowMachine>()) /
                               batchAttr.getInt() ||
        guards.size() != 1 || branches.size() != 1 || winBufs.size() != 1 ||
        addrBufs.size() > 1)
      return prog.emitError(
          "invalid or incomplete Mojo windowed score metadata");
    int64_t slots = winAttr.getInt(), batch = batchAttr.getInt();
    Value scoreBuf = winBufs[0];
    Operation *scoreAlloc = scoreBuf.getDefiningOp();
    auto scoreDt = scoreAlloc->getAttrOfType<StringAttr>("dtype");
    auto scoreCnt = scoreAlloc->getAttrOfType<IntegerAttr>("count");
    if (!scoreDt || scoreDt.getValue() != "float32" || !scoreCnt ||
        scoreCnt.getInt() < batch * slots)
      return scoreAlloc->emitError("invalid Mojo score_window stack buffer");
    if (!addrBufs.empty()) {
      Operation *addrAlloc = addrBufs[0].getDefiningOp();
      auto addrDt = addrAlloc->getAttrOfType<StringAttr>("dtype");
      auto addrCnt = addrAlloc->getAttrOfType<IntegerAttr>("count");
      if (!addrDt || addrDt.getValue() != "int64" || !addrCnt ||
          addrCnt.getInt() < slots)
        return addrAlloc->emitError("invalid Mojo window_addrs stack buffer");
    }

    auto isConstIdx = [](Value v, int64_t want) {
      auto cst = v ? v.getDefiningOp<arith::ConstantIndexOp>()
                   : arith::ConstantIndexOp();
      return cst && cst.value() == want;
    };
    auto sameBound = [&](auto &self, Value x, Value y) -> bool {
      if (x == y)
        return true;
      auto *px = x ? x.getDefiningOp() : nullptr;
      auto *py = y ? y.getDefiningOp() : nullptr;
      if (!px || !py || px->getName() != py->getName() ||
          px->getNumRegions() != 0 || py->getNumRegions() != 0 ||
          px->getNumResults() != 1 || py->getNumResults() != 1 ||
          x.getType() != y.getType() ||
          px->getPropertiesAsAttribute() != py->getPropertiesAsAttribute() ||
          px->getNumOperands() != py->getNumOperands())
        return false;
      return llvm::all_of(
          llvm::zip(px->getOperands(), py->getOperands()),
          [&](auto pair) { return self(self, std::get<0>(pair), std::get<1>(pair)); });
    };
    auto isSubOf = [&](Value v, Value lhsWant, Value rhsWant) {
      auto sub = v ? v.getDefiningOp<arith::SubIOp>() : arith::SubIOp();
      return sub && sameBound(sameBound, sub.getLhs(), lhsWant) &&
             sameBound(sameBound, sub.getRhs(), rhsWant);
    };
    auto isWindowIndexAt = [&](Operation *memOp, Value buf, Value ivWant,
                               Value loWant, bool isSlice) {
      if (memOp->getNumOperands() < 2 || memOp->getOperand(0) != buf)
        return false;
      Operation *addrOp = memOp->getOperand(1).getDefiningOp();
      if (!addrOp || addrOp->getName().getStringRef() != "mojo_row.address" ||
          addrOp->getOperand(0) != buf)
        return false;
      auto strides = addrOp->getAttrOfType<DenseI64ArrayAttr>("strides");
      if (isSlice) {
        if (addrOp->getNumOperands() != 2 || !strides || strides.size() != 1 ||
            strides[0] != batch ||
            !isSubOf(addrOp->getOperand(1), ivWant, loWant))
          return false;
        if (memOp->getName().getStringRef() == "mojo_row.load_slice") {
          auto w = memOp->getAttrOfType<IntegerAttr>("width");
          return w && w.getInt() == batch;
        }
        auto st = dyn_cast<ShapedType>(memOp->getOperand(2).getType());
        return st && st.getNumElements() == batch;
      }
      return addrOp->getNumOperands() == 3 && strides && strides.size() == 2 &&
             strides[0] == batch && strides[1] == 1 &&
             isSubOf(addrOp->getOperand(1), ivWant, loWant);
    };

    Operation *gIf = guards[0], *bIf = branches[0];
    Operation *loop1 = gIf->getParentOp();
    if (!loop1 || loop1->getName().getStringRef() != "mojo_row.for_range" ||
        !isConstIdx(loop1->getOperand(2), 1) ||
        loop1->getBlock() != bIf->getBlock() || !loop1->isBeforeInBlock(bIf) ||
        !gIf->getRegion(1).empty() || bIf->getRegion(1).empty())
      return prog.emitError("invalid Mojo windowed guard or branch structure");
    Value lo1 = loop1->getOperand(0), hi1 = loop1->getOperand(1),
          iv1 = loop1->getRegion(0).front().getArgument(0);
    auto gCmp = gIf->getOperand(0).getDefiningOp<arith::CmpIOp>();
    bool validGuardCond =
        gCmp && ((gCmp.getPredicate() == arith::CmpIPredicate::slt &&
                  isSubOf(gCmp.getLhs(), iv1, lo1) &&
                  isConstIdx(gCmp.getRhs(), slots)) ||
                 (gCmp.getPredicate() == arith::CmpIPredicate::sgt &&
                  isConstIdx(gCmp.getLhs(), slots) &&
                  isSubOf(gCmp.getRhs(), iv1, lo1)));
    if (!validGuardCond)
      return gIf->emitError(
          "Mojo window guard condition must compare (iv - lo) < window_slots");

    unsigned guardSliceStores = 0, guardScalarStores = 0;
    bool badLoop1Buf = false;
    llvm::SmallPtrSet<Value, 4> guardWrittenBufs;
    loop1->walk([&](Operation *op) {
      StringRef n = op->getName().getStringRef();
      if (n == "mojo_row.store_slice" || n == "mojo_row.store_scalar") {
        bool inGuard = gIf->getRegion(0).isAncestor(op->getParentRegion());
        if (inGuard)
          if (auto *def = op->getOperand(0).getDefiningOp();
              def && def->getName().getStringRef() == "mojo_row.stack_alloc")
            guardWrittenBufs.insert(op->getOperand(0));
        if (op->getOperand(0) == scoreBuf) {
          bool isSlice = (n == "mojo_row.store_slice");
          if (!inGuard || !isWindowIndexAt(op, scoreBuf, iv1, lo1, isSlice))
            badLoop1Buf = true;
          else if (isSlice)
            ++guardSliceStores;
          else
            ++guardScalarStores;
        }
      } else if ((n == "mojo_row.load_slice" || n == "mojo_row.load_scalar") &&
                 op->getOperand(0) == scoreBuf) {
        badLoop1Buf = true;
      }
    });
    bool validGuardStores =
        !badLoop1Buf && guardWrittenBufs.size() == 1 &&
        guardWrittenBufs.contains(scoreBuf) &&
        ((guardSliceStores == 1 && guardScalarStores == 0) ||
         (guardSliceStores == 0 && guardScalarStores == batch));
    if (!validGuardStores)
      return gIf->emitError(
          "Mojo window guard must contain the score_window store indexed at (iv - lo)");

    auto bCmp = bIf->getOperand(0).getDefiningOp<arith::CmpIOp>();
    bool validBranchCond =
        bCmp && ((bCmp.getPredicate() == arith::CmpIPredicate::sle &&
                  isSubOf(bCmp.getLhs(), hi1, lo1) &&
                  isConstIdx(bCmp.getRhs(), slots)) ||
                 (bCmp.getPredicate() == arith::CmpIPredicate::sge &&
                  isConstIdx(bCmp.getLhs(), slots) &&
                  isSubOf(bCmp.getRhs(), hi1, lo1)) ||
                 (bCmp.getPredicate() == arith::CmpIPredicate::slt &&
                  isSubOf(bCmp.getLhs(), hi1, lo1) &&
                  isConstIdx(bCmp.getRhs(), slots + 1)));
    if (!validBranchCond)
      return bIf->emitError(
          "Mojo window branch condition must compare (hi - lo) <= window_slots");

    SmallVector<Operation *, 2> thenLoops, elseLoops;
    for (Operation &op : bIf->getRegion(0).front())
      if (op.getName().getStringRef() == "mojo_row.for_range" &&
          isConstIdx(op.getOperand(2), 1) &&
          sameBound(sameBound, op.getOperand(0), lo1) &&
          sameBound(sameBound, op.getOperand(1), hi1))
        thenLoops.push_back(&op);
    for (Operation &op : bIf->getRegion(1).front())
      if (op.getName().getStringRef() == "mojo_row.for_range" &&
          isConstIdx(op.getOperand(2), 1) &&
          sameBound(sameBound, op.getOperand(0), lo1) &&
          sameBound(sameBound, op.getOperand(1), hi1))
        elseLoops.push_back(&op);
    if (thenLoops.size() != 1 || elseLoops.size() != 1)
      return bIf->emitError(
          "Mojo window branch must wrap single replay and recompute folds over [lo, hi)");

    Operation *thenFor = thenLoops[0];
    Value thenLo = thenFor->getOperand(0),
          thenIv = thenFor->getRegion(0).front().getArgument(0);
    unsigned thenSliceLoads = 0, thenScalarLoads = 0;
    bool validThenLoadIdx = true;
    thenFor->getRegion(0).walk([&](Operation *op) {
      StringRef n = op->getName().getStringRef();
      if ((n == "mojo_row.load_slice" || n == "mojo_row.load_scalar") &&
          op->getOperand(0) == scoreBuf) {
        bool isSlice = (n == "mojo_row.load_slice");
        validThenLoadIdx &=
            isWindowIndexAt(op, scoreBuf, thenIv, thenLo, isSlice);
        if (isSlice)
          ++thenSliceLoads;
        else
          ++thenScalarLoads;
      }
    });
    if (!validThenLoadIdx ||
        !((thenSliceLoads == 1 && thenScalarLoads == 0) ||
          (thenSliceLoads == 0 && thenScalarLoads == batch)))
      return bIf->emitError(
          "Mojo window replay branch must read score_window at (iv - lo)");

    bool elseTouchesWindow =
        bIf->getRegion(1)
            .walk([&](Operation *op) {
              for (Value operand : op->getOperands())
                if (operand == scoreBuf ||
                    (!addrBufs.empty() && operand == addrBufs[0]) ||
                    guardWrittenBufs.contains(operand))
                  return WalkResult::interrupt();
              return WalkResult::advance();
            })
            .wasInterrupted();
    if (elseTouchesWindow)
      return bIf->emitError(
          "Mojo over-window recomputation branch must not read score window");
  }

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

  bool badAddr =
      m.walk([&](Operation *op) {
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
         if (op->getName().getStringRef() != "mojo_row.address")
           return WalkResult::advance();
         auto bits = op->getAttrOfType<IntegerAttr>("index_bits");
         if (!bits || (bits.getInt() != 32 && bits.getInt() != 64)) {
           op->emitError("mojo_row.address requires i32 or i64 index_bits");
           return WalkResult::interrupt();
         }
         if (bits.getInt() == 32 && !canNarrowMojoAddress(*op)) {
           op->emitError(
               "32-bit ordered address lacks a proven signed i32 range");
           return WalkResult::interrupt();
         }
         return WalkResult::advance();
       }).wasInterrupted();
  if (badAddr)
    return failure();

  if (failed(mojoMatrixPrinter().verify(m, p)))
    return failure();

  if (callable) {
    if (fn->getAttrOfType<StringAttr>("mojo_row.entry") != "intent_entry")
      return fn.emitError("missing synchronous Mojo host entry");
    llvm::SmallVector<char> expected;
    llvm::raw_svector_ostream os(expected);
    if (auto err = describeMojoOrderedPackage(fn, prog, os))
      return fn.emitError() << llvm::toString(std::move(err)), failure();
    auto pkg = m->getAttrOfType<StringAttr>("mojo_row.package");
    if (!pkg || pkg.getValue() != StringRef(expected.data(), expected.size()))
      return fn.emitError(
                 "mojo_row.package disagrees with verified ordered ABI"),
             failure();
  }
  return success();
}
} // namespace CI_UNIT_NAMESPACE
