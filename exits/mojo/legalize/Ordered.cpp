#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "intentdsl/levels/algorithm/kir/ir/Ordered.h"
#include "intentdsl/levels/execution/cpu/ir/MatrixTask.h"
#include "intentdsl/levels/execution/cpu/ir/OwnerTraversal.h"
#include "intentdsl/levels/execution/cpu/ir/RepeatedScore.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include <cmath>

namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
using namespace mlir;

bool canNarrowMojoAddress(Operation &op);
TerminalPrinter mojoMatrixPrinter();

namespace {
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
  Value localStackBuf;

  // Chunked L1-resident slice lowering state (active when chunkedMode == true)
  bool chunkedMode = false;
  int64_t invFloatsPerLane = 0;
  int64_t invStride = 0;
  int64_t accExtent = 0;
  int64_t accStride = 0;
  Value invPrepackBuf;
  Value wideAccBuf;
  Value vAddrBuf;
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
    Value addr = makeAddr(b, loc, ref.ptr, {ref.baseAddr, cIdx}, {1, 1});
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
          if (inTiledReplayPassA && &op == tiledVReadOp && laneIdx == 0) {
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
              wideSplats[laneIdx].count(mul.getRhs())) {
            srcVal = mul.getLhs();
            scaleScalar = wideSplats[laneIdx][mul.getRhs()];
          } else if (wideBufValues.contains(mul.getRhs()) &&
                     wideSplats[laneIdx].count(mul.getLhs())) {
            srcVal = mul.getRhs();
            scaleScalar = wideSplats[laneIdx][mul.getLhs()];
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
        auto *wLoop = makeOp(
            b, loc, "for_range",
            {constIdx(b, loc, 0), constIdx(b, loc, k), constIdx(b, loc, 16)},
            {}, {}, 1);
        Block &wBb = wLoop->getRegion(0).emplaceBlock();
        Value cIdx = wBb.addArgument(b.getIndexType(), loc);
        OpBuilder::InsertionGuard wg(b);
        b.setInsertionPointToEnd(&wBb);
        Value bufAddr = makeAddr(
            b, loc, wideAccBuf,
            {constIdx(b, loc, int64_t(laneIdx) * accStride), cIdx}, {1, 1});
        Value chunkVal =
            loadContiguousSlice(wideAccBuf, bufAddr, b.getF32Type(), 16);
        if (scaleVec)
          chunkVal = b.create<arith::MulFOp>(loc, chunkVal, scaleVec);
        if (!elemTy.isF32()) {
          auto outTy = RankedTensorType::get({16}, elemTy);
          chunkVal = b.create<arith::TruncFOp>(loc, outTy, chunkVal);
        }
        Value dstAddr = makeAddr(b, loc, ptr, {addr, cIdx}, {1, 1});
        makeOp(b, loc, "store_slice", {ptr, dstAddr, chunkVal});
        makeOp(b, loc, "end");
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

    StringRef ns = op.getName().getDialectNamespace();
    if (ns == "arith" || ns == "math") {
      Operation *cloned = b.clone(op, map);
      cloned->removeAttr("cpu.owner_common");
      cloned->removeAttr("mojo_row.pack_scalars");
      return success();
    }
    return op.emitError("unsupported Mojo ordered instruction ") << n;
  }

  LogicalResult lowerBlock(Block &srcBb, SmallVectorImpl<IRMapping> &lanes) {
    for (Operation &op : srcBb.without_terminator()) {
      if (auto loop = dyn_cast<scf::ForOp>(op)) {
        unsigned numIter = loop.getNumRegionIterArgs();
        SmallVector<SmallVector<Value, 4>, 16> slots(lanes.size());
        bool hasWideIter = false;
        for (size_t l = 0; l < lanes.size(); ++l) {
          for (unsigned k = 0; k < numIter; ++k) {
            Value initOrig = loop.getInitArgs()[k];
            if (isWideVal(initOrig)) {
              hasWideIter = true;
              wideBufValues.insert(loop.getRegionIterArgs()[k]);
              wideBufValues.insert(loop.getResult(k));
              slots[l].push_back(Value{});
              continue;
            }
            Value init = lanes[l].lookup(initOrig);
            slots[l].push_back(
                declSlot(loop.getResult(k).getType(), init, initOrig));
          }
        }
        // Check if this is a window-replay loop eligible for register-tiled
        // column-outer accumulation (zero L1d vector stores per token!)
        Operation *candVRead = nullptr;
        unsigned wideReads = 0, wideAdds = 0;
        bool canTileReplay = false;
        if (hasWideIter && vAddrBuf && batch >= 2 && accExtent >= 32 &&
            accExtent % 32 == 0) {
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
              if (bodyOk && wideReads == 1 && wideAdds == 1 && candVRead)
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
          for (size_t l = 0; l < lanes.size(); ++l) {
            lanes[l].map(loop.getInductionVar(), iv);
            for (unsigned k = 0; k < numIter; ++k)
              if (slots[l][k])
                lanes[l].map(loop.getRegionIterArgs()[k], readSlot(slots[l][k]));
          }
          bool prevTiled = inTiledReplayPassA;
          Operation *prevVRead = tiledVReadOp;
          if (canTileReplay) {
            inTiledReplayPassA = true;
            tiledVReadOp = candVRead;
          }
          if (failed(lowerBlock(*loop.getBody(), lanes)))
            return failure();
          inTiledReplayPassA = prevTiled;
          tiledVReadOp = prevVRead;
          auto *term = loop.getBody()->getTerminator();
          for (size_t l = 0; l < lanes.size(); ++l) {
            SmallVector<Value, 4> nextVals(numIter, Value{});
            for (unsigned k = 0; k < numIter; ++k)
              if (slots[l][k])
                nextVals[k] = lanes[l].lookup(term->getOperand(k));
            for (unsigned k = 0; k < numIter; ++k)
              if (slots[l][k])
                writeSlot(slots[l][k], nextVals[k]);
          }
          makeOp(b, loc, "end");
        }
        if (canTileReplay) {
          // Pass B: column-outer register-blocked accumulation [8 heads x 32 cols]
          Value span = b.create<arith::SubIOp>(loc, ub, lb);
          auto f32x16 = RankedTensorType::get({16}, b.getF32Type());
          for (size_t sub = 0; sub < lanes.size(); sub += 8) {
            size_t subEnd = std::min<size_t>(lanes.size(), sub + 8);
            size_t subLen = subEnd - sub;
            auto *cOuter = makeOp(
                b, loc, "for_range",
                {constIdx(b, loc, 0), constIdx(b, loc, accExtent),
                 constIdx(b, loc, 32)},
                {}, {}, 1);
            Block &cBb = cOuter->getRegion(0).emplaceBlock();
            Value cIdx = cBb.addArgument(b.getIndexType(), loc);
            OpBuilder::InsertionGuard cg(b);
            b.setInsertionPointToEnd(&cBb);
            Value cIdxHi = addIdx(b, loc, cIdx, 16);
            SmallVector<Value, 8> accLo, accHi;
            for (size_t s = 0; s < subLen; ++s) {
              accLo.push_back(
                  makeOp(b, loc, "acc_decl", {}, {f32x16},
                         {b.getNamedAttr("dtype", b.getStringAttr("float32")),
                          b.getNamedAttr("width", b.getI64IntegerAttr(16))})
                      ->getResult(0));
              accHi.push_back(
                  makeOp(b, loc, "acc_decl", {}, {f32x16},
                         {b.getNamedAttr("dtype", b.getStringAttr("float32")),
                          b.getNamedAttr("width", b.getI64IntegerAttr(16))})
                      ->getResult(0));
            }
            {
              auto *tInner = makeOp(
                  b, loc, "for_range",
                  {constIdx(b, loc, 0), span, constIdx(b, loc, 1)}, {}, {}, 1);
              Block &tBb = tInner->getRegion(0).emplaceBlock();
              Value off = tBb.addArgument(b.getIndexType(), loc);
              OpBuilder::InsertionGuard tg(b);
              b.setInsertionPointToEnd(&tBb);
              Value vOffAddr = makeAddr(b, loc, vAddrBuf, {off}, {1});
              Value vBase64 = makeOp(b, loc, "load_scalar",
                                     {vAddrBuf, vOffAddr}, {b.getI64Type()})
                                  ->getResult(0);
              Value vBase =
                  makeOp(b, loc, "as_int", {vBase64}, {b.getIndexType()})
                      ->getResult(0);
              WideViewRef vRef{tiledVPtr, vBase, tiledVElemTy, accExtent};
              Value vLo = loadWideChunkF32(vRef, cIdx, 16);
              Value vHi = loadWideChunkF32(vRef, cIdxHi, 16);
              for (size_t s = 0; s < subLen; ++s) {
                Value wAddr = makeAddr(b, loc, localStackBuf,
                                       {off, constIdx(b, loc, sub + s)},
                                       {batch, 1});
                Value wScalar = makeOp(b, loc, "load_scalar",
                                       {localStackBuf, wAddr}, {b.getF32Type()})
                                    ->getResult(0);
                Value wVec =
                    makeOp(b, loc, "splat_scalar", {wScalar}, {f32x16},
                           {b.getNamedAttr("dtype", b.getStringAttr("float32")),
                            b.getNamedAttr("width", b.getI64IntegerAttr(16))})
                        ->getResult(0);
                writeSlot(
                    accLo[s],
                    b.create<arith::AddFOp>(
                        loc, readSlot(accLo[s]),
                        b.create<arith::MulFOp>(loc, vLo, wVec)));
                writeSlot(
                    accHi[s],
                    b.create<arith::AddFOp>(
                        loc, readSlot(accHi[s]),
                        b.create<arith::MulFOp>(loc, vHi, wVec)));
              }
              makeOp(b, loc, "end");
            }
            for (size_t s = 0; s < subLen; ++s) {
              int64_t laneBase = int64_t(sub + s) * accStride;
              Value addrLo = makeAddr(
                  b, loc, wideAccBuf, {constIdx(b, loc, laneBase), cIdx},
                  {1, 1});
              Value addrHi = makeAddr(
                  b, loc, wideAccBuf, {constIdx(b, loc, laneBase), cIdxHi},
                  {1, 1});
              makeOp(b, loc, "store_slice",
                     {wideAccBuf, addrLo, readSlot(accLo[s])});
              makeOp(b, loc, "store_slice",
                     {wideAccBuf, addrHi, readSlot(accHi[s])});
            }
            makeOp(b, loc, "end");
          }
        }
        for (size_t l = 0; l < lanes.size(); ++l)
          for (unsigned k = 0; k < numIter; ++k)
            if (slots[l][k])
              lanes[l].map(loop.getResult(k), readSlot(slots[l][k]));
        continue;
      }

      if (auto branch = dyn_cast<scf::IfOp>(op)) {
        unsigned numRes = branch.getNumResults();
        bool hasElse = !branch.getElseRegion().empty();
        SmallVector<SmallVector<Value, 4>, 16> resSlots(lanes.size());
        for (size_t l = 0; l < lanes.size(); ++l) {
          for (unsigned k = 0; k < numRes; ++k) {
            Value origRef = branch.thenYield().getOperand(k);
            if (isWideVal(origRef)) {
              wideBufValues.insert(branch.getResult(k));
              resSlots[l].push_back(Value{});
              continue;
            }
            resSlots[l].push_back(
                declSlot(branch.getResult(k).getType(), Value{}, origRef));
          }
        }
        bool sharedCond = llvm::all_of(lanes, [&](IRMapping &m) {
          return m.lookup(branch.getCondition()) ==
                 lanes[0].lookup(branch.getCondition());
        });
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
            if (failed(lowerBlock(branch.getThenRegion().front(), lanes)))
              return failure();
            auto *tTerm = branch.getThenRegion().front().getTerminator();
            for (size_t l = 0; l < lanes.size(); ++l)
              for (unsigned k = 0; k < numRes; ++k)
                if (resSlots[l][k])
                  writeSlot(resSlots[l][k],
                            lanes[l].lookup(tTerm->getOperand(k)));
          }
          if (hasElse) {
            OpBuilder::InsertionGuard eg(b);
            auto &eBb = newIf.getElseRegion().front();
            b.setInsertionPointToStart(&eBb);
            if (failed(lowerBlock(branch.getElseRegion().front(), lanes)))
              return failure();
            auto *eTerm = branch.getElseRegion().front().getTerminator();
            for (size_t l = 0; l < lanes.size(); ++l)
              for (unsigned k = 0; k < numRes; ++k)
                if (resSlots[l][k])
                  writeSlot(resSlots[l][k],
                            lanes[l].lookup(eTerm->getOperand(k)));
          }
        } else {
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
        for (size_t l = 0; l < lanes.size(); ++l)
          for (unsigned k = 0; k < numRes; ++k)
            if (resSlots[l][k])
              lanes[l].map(branch.getResult(k), readSlot(resSlots[l][k]));
        continue;
      }

      // Batched chunked wide dot product: lane_sum(mulf(wideA, wideB))
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
        for (size_t sub = 0; sub < lanes.size(); sub += 16) {
          size_t subEnd = std::min(lanes.size(), sub + 16);
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
              Value lv = sharedLhs
                             ? sharedLhs
                             : loadWideChunkF32(wideViews[l][lhsOrig], cIdx, 16);
              Value rv = sharedRhs
                             ? sharedRhs
                             : loadWideChunkF32(wideViews[l][rhsOrig], cIdx, 16);
              Value prod = b.create<arith::MulFOp>(loc, lv, rv);
              Value sum =
                  b.create<arith::AddFOp>(loc, readSlot(accSlots[l - sub]), prod);
              writeSlot(accSlots[l - sub], sum);
            }
            makeOp(b, loc, "end");
          }
          for (size_t l = sub; l < subEnd; ++l) {
            Value dot =
                makeOp(b, loc, "lane_sum", {readSlot(accSlots[l - sub])},
                       {b.getF32Type()})
                    ->getResult(0);
            lanes[l].map(op.getResult(0), dot);
          }
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
                wideSplats[0].count(m.getRhs())) {
              alphaOrig = m.getRhs();
              return true;
            }
            if (wideBufValues.contains(m.getRhs()) &&
                wideSplats[0].count(m.getLhs())) {
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
          for (size_t l = 0; l < lanes.size(); ++l) {
            Value wAddr = makeAddr(b, loc, localStackBuf,
                                   {off, constIdx(b, loc, l)}, {batch, 1});
            makeOp(b, loc, "store_scalar",
                   {localStackBuf, wAddr, wideSplats[l][betaOrig]});
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
                  makeOp(b, loc, "splat_scalar", {wideSplats[l][alphaOrig]},
                         {f32x16},
                         {b.getNamedAttr("dtype", b.getStringAttr("float32")),
                          b.getNamedAttr("width", b.getI64IntegerAttr(16))})
                      ->getResult(0));
            }
            betaVecs.push_back(
                makeOp(b, loc, "splat_scalar", {wideSplats[l][betaOrig]},
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
        if (l > 0 && op.hasAttr("cpu.owner_common")) {
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
        auto *pLoop = makeOp(
            b, loc, "for_range",
            {constIdx(b, loc, 0), constIdx(b, loc, kExt), constIdx(b, loc, 16)},
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
      {"abi", std::move(*abi)},
      {"domain", intentdsl::publishDomain(fn)},
      {"rows",
       llvm::json::Array{llvm::json::Object{{"grain", grain}, {"columns", 1}}}},
      {"returns_status", true},
      {"source", ""},
      {"host", ""}});
  return llvm::Error::success();
}

LogicalResult lowerMojoOrdered(func::FuncOp fn) {
  auto &task = fn.front().front();
  auto &space = task.getRegion(0).front().front();
  Location loc = space.getLoc();
  OpBuilder b(&fn.front().front());

  int64_t owners = space.getAttrOfType<IntegerAttr>("cpu.owners").getInt();
  int64_t grain = space.getAttrOfType<IntegerAttr>("cpu.grain").getInt();
  int64_t batch = 1;
  if (auto bAttr = space.getAttrOfType<IntegerAttr>("cpu.owner_batch"))
    batch = bAttr.getInt();
  auto windowAttr = space.getAttrOfType<IntegerAttr>("cpu.batch_score_window");

  Operation *foundLocalBuf = nullptr;
  for (Operation &op : space.getRegion(0).front()) {
    if (op.getName().getStringRef() == "cpu.local_buffer") {
      foundLocalBuf = &op;
      break;
    }
  }

  OrderedLowerer lowerer{fn, space, b, loc, batch};
  lowerer.analyzeChunkableWideMode();
  lowerer.wideViews.resize(batch);
  lowerer.wideSplats.resize(batch);
  bool usesStack = foundLocalBuf != nullptr || lowerer.invFloatsPerLane > 0 ||
                   lowerer.accExtent > 0;

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
              b.getNamedAttr("uses_stack", b.getBoolAttr(usesStack))},
             1);
  makeOp(b, loc, "end");

  Block &fnBb = ownerFn->getRegion(0).emplaceBlock();
  SmallVector<Value> paramVals;
  for (auto arg : fn.getArguments())
    paramVals.push_back(fnBb.addArgument(arg.getType(), loc));
  Value loArg = fnBb.addArgument(b.getI64Type(), loc);
  Value hiArg = fnBb.addArgument(b.getI64Type(), loc);

  b.setInsertionPointToEnd(&fnBb);

  if (foundLocalBuf) {
    int64_t slots =
        foundLocalBuf->getAttrOfType<IntegerAttr>("elements").getInt();
    lowerer.localStackBuf =
        makeOp(b, loc, "stack_alloc", {}, {b.getIndexType()},
               {b.getNamedAttr("dtype", b.getStringAttr("float32")),
                b.getNamedAttr("count", b.getI64IntegerAttr(batch * slots))})
            ->getResult(0);
    if (lowerer.chunkedMode && lowerer.accExtent >= 32 &&
        lowerer.accExtent % 32 == 0) {
      lowerer.vAddrBuf =
          makeOp(b, loc, "stack_alloc", {}, {b.getIndexType()},
                 {b.getNamedAttr("dtype", b.getStringAttr("int64")),
                  b.getNamedAttr("count", b.getI64IntegerAttr(slots))})
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

  Value loInt =
      makeOp(b, loc, "as_int", {loArg}, {b.getIndexType()})->getResult(0);
  Value hiInt =
      makeOp(b, loc, "as_int", {hiArg}, {b.getIndexType()})->getResult(0);
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
    Value coord =
        makeOp(b, loc, "owner_coord", {owner}, {b.getIndexType()},
               {b.getNamedAttr("pitch", b.getI64IntegerAttr(pitch)),
                b.getNamedAttr("extent", b.getI64IntegerAttr(ext))})
            ->getResult(0);
    for (int64_t l = 0; l < batch; ++l) {
      Value laneCoord = coord;
      if (batch > 1 && pitch == 1 && innerAttr &&
          unsigned(innerAttr.getInt()) == ax && l > 0)
        laneCoord = addIdx(b, loc, coord, l);
      lanes[l].map(srcBody.getArgument(ax), laneCoord);
    }
    pitch *= ext;
  }

  if (failed(lowerer.lowerBlock(srcBody, lanes)))
    return failure();
  makeOp(b, loc, "end");
  b.setInsertionPointToEnd(&fnBb);
  makeOp(b, loc, "end");

  // Region 1: entry.c (structured C host function + OpenMP loop + kernel call)
  Block &hBb = prog->getRegion(1).emplaceBlock();
  b.setInsertionPointToEnd(&hBb);
  std::string externDecl =
      "extern void owner_task(" + llvm::join(externCParams, ",") +
      (externCParams.empty() ? "" : ",") + "int64_t,int64_t);";
  auto *cFn = makeOp(
      b, loc, "c_host_fn", {}, {},
      {b.getNamedAttr("params",
                      b.getStringAttr(intentdsl::cpu::matrixHostParams(fn))),
       b.getNamedAttr("extern_decl", b.getStringAttr(externDecl))},
      1);
  makeOp(b, loc, "end");

  Block &cFnBb = cFn->getRegion(0).emplaceBlock();
  b.setInsertionPointToEnd(&cFnBb);
  auto *ompFor = makeOp(
      b, loc, "c_omp_for", {}, {},
      {b.getNamedAttr("count", b.getI64IntegerAttr(owners)),
       b.getNamedAttr("grain", b.getI64IntegerAttr(grain)),
       b.getNamedAttr("cyclic", b.getBoolAttr(false))},
      1);
  makeOp(b, loc, "end");

  Block &ompBb = ompFor->getRegion(0).emplaceBlock();
  b.setInsertionPointToEnd(&ompBb);
  makeOp(b, loc, "c_owner_call", {}, {},
         {b.getNamedAttr("call_args", b.getStringAttr(llvm::join(callArgs, ","))),
          b.getNamedAttr("cyclic", b.getBoolAttr(false))});
  makeOp(b, loc, "end");

  SmallVector<Operation *> toErase;
  for (Operation &op : fn.front()) {
    if (&op != prog && op.getName().getStringRef() != "func.return")
      toErase.push_back(&op);
  }
  for (Operation *op : llvm::reverse(toErase))
    op->erase();
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

  // Independently verify windowed score replay/recompute threshold on resume
  SmallVector<scf::IfOp, 2> guards, branches;
  m.walk([&](scf::IfOp ifOp) {
    if (ifOp->hasAttr("mojo_row.window_guard"))
      guards.push_back(ifOp);
    if (ifOp->hasAttr("mojo_row.window_branch"))
      branches.push_back(ifOp);
  });
  auto winAttr = prog.getAttrOfType<IntegerAttr>("window_slots");
  auto batchAttr = prog.getAttrOfType<IntegerAttr>("owner_batch");
  if (winAttr || batchAttr || !guards.empty() || !branches.empty()) {
    if (!winAttr || !batchAttr || batchAttr.getInt() <= 0 ||
        winAttr.getInt() !=
            intentdsl::cpu::batchScoreSlots / batchAttr.getInt() ||
        guards.size() != 1 || branches.size() != 1)
      return prog.emitError(
          "invalid or incomplete Mojo windowed score metadata");
    int64_t slots = winAttr.getInt();
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
    auto gIf = guards[0], bIf = branches[0];
    Operation *loop1 = gIf->getParentOp();
    if (!loop1 || loop1->getName().getStringRef() != "mojo_row.for_range" ||
        !gIf.getElseRegion().empty() || bIf.getElseRegion().empty())
      return prog.emitError("invalid Mojo windowed guard or branch structure");
    Value lo1 = loop1->getOperand(0), hi1 = loop1->getOperand(1),
          iv1 = loop1->getRegion(0).front().getArgument(0);
    auto gCmp = gIf.getCondition().getDefiningOp<arith::CmpIOp>();
    auto gSub = gCmp ? gCmp.getLhs().getDefiningOp<arith::SubIOp>()
                     : arith::SubIOp();
    if (!gCmp || gCmp.getPredicate() != arith::CmpIPredicate::slt || !gSub ||
        gSub.getLhs() != iv1 || !sameBound(sameBound, gSub.getRhs(), lo1) ||
        !isConstIdx(gCmp.getRhs(), slots))
      return gIf.emitError(
          "Mojo window guard condition must compare (iv - lo) < window_slots");

    Operation *thenFor = nullptr, *elseFor = nullptr;
    for (Operation &op : bIf.getThenRegion().front())
      if (op.getName().getStringRef() == "mojo_row.for_range" &&
          isConstIdx(op.getOperand(2), 1) && !thenFor)
        thenFor = &op;
    for (Operation &op : bIf.getElseRegion().front())
      if (op.getName().getStringRef() == "mojo_row.for_range" &&
          isConstIdx(op.getOperand(2), 1) && !elseFor)
        elseFor = &op;
    auto bCmp = bIf.getCondition().getDefiningOp<arith::CmpIOp>();
    auto bSub = bCmp ? bCmp.getLhs().getDefiningOp<arith::SubIOp>()
                     : arith::SubIOp();
    if (!thenFor || !elseFor || !bCmp ||
        bCmp.getPredicate() != arith::CmpIPredicate::sle || !bSub ||
        !sameBound(sameBound, bSub.getLhs(), hi1) ||
        !sameBound(sameBound, bSub.getRhs(), lo1) ||
        !sameBound(sameBound, thenFor->getOperand(0), lo1) ||
        !sameBound(sameBound, thenFor->getOperand(1), hi1) ||
        !sameBound(sameBound, elseFor->getOperand(0), lo1) ||
        !sameBound(sameBound, elseFor->getOperand(1), hi1) ||
        !isConstIdx(bCmp.getRhs(), slots))
      return bIf.emitError(
          "Mojo window branch condition must compare (hi - lo) <= window_slots");
  }

  bool badAddr =
      m.walk([&](Operation *op) {
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
