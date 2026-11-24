#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "intentdsl/levels/algorithm/kir/ir/Storage.h"
#include "intentdsl/levels/execution/cpu/ir/MatrixTask.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"

namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
using namespace mlir;

bool canNarrowMojoAddress(Operation &op);
TerminalPrinter mojoMatrixPrinter();

namespace {
struct ViewDesc {
  unsigned slot;
  std::string dtype;
  std::string hostOrigin;
  SmallVector<int64_t> panelStrides;
  bool writes = false;
};

std::string mojoDType(StringRef t) {
  return llvm::StringSwitch<std::string>(t)
      .Case("bf16", "bfloat16")
      .Case("f16", "float16")
      .Case("i8", "int8")
      .Case("i32", "int32")
      .Case("u8", "uint8")
      .Case("f8e4m3fn", "float8_e4m3fn")
      .Case("f32", "float32")
      .Default("float32");
}

std::string elemTypeName(Type t) {
  if (auto sh = dyn_cast<ShapedType>(t))
    t = sh.getElementType();
  if (t.isBF16())
    return "bf16";
  if (t.isF16())
    return "f16";
  if (t.isF32())
    return "f32";
  if (t.isInteger(8))
    return "i8";
  return "i32";
}

Operation *makeOp(OpBuilder &b, Location loc, StringRef name,
                  ValueRange operands = {}, TypeRange types = {},
                  ArrayRef<NamedAttribute> attrs = {}) {
  OperationState s(loc, ("mojo_row." + name).str());
  s.addOperands(operands);
  s.addTypes(types);
  s.addAttributes(attrs);
  return b.create(s);
}

Value constIdx(OpBuilder &b, Location loc, int64_t v) {
  return b.create<arith::ConstantIndexOp>(loc, v);
}

Value addOffset(OpBuilder &b, Location loc, Value base, int64_t delta) {
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

struct EpilogueStep {
  enum Kind { Bias, Convert } kind;
  unsigned biasViewIdx = 0;
  Type fromType, toType;
};

Value emitEpilogue(OpBuilder &b, Location loc, Value accVal, Value colCoord,
                   int64_t width, ArrayRef<EpilogueStep> steps,
                   ArrayRef<Value> panelPtrs, ArrayRef<ViewDesc> views) {
  Value cur = accVal;
  for (auto &step : steps) {
    if (step.kind == EpilogueStep::Bias) {
      auto &bv = views[step.biasViewIdx];
      Value bAddr =
          makeAddr(b, loc, panelPtrs[step.biasViewIdx], {colCoord},
                   bv.panelStrides);
      Value bVec =
          makeOp(b, loc, "load_vec", {panelPtrs[step.biasViewIdx], bAddr},
                 {b.getIndexType()},
                 {b.getNamedAttr("dtype", b.getStringAttr(mojoDType(bv.dtype))),
                  b.getNamedAttr("width", b.getI64IntegerAttr(width)),
                  b.getNamedAttr("lane_stride",
                                 b.getI64IntegerAttr(bv.panelStrides.back()))})
              ->getResult(0);
      cur = makeOp(b, loc, "epilogue_add", {cur, bVec}, {b.getIndexType()})
                ->getResult(0);
    } else {
      if (step.toType.isBF16() && step.fromType.isF32()) {
        cur = makeOp(b, loc, "epilogue_round_bf16", {cur}, {b.getIndexType()},
                     {b.getNamedAttr("width", b.getI64IntegerAttr(width))})
                  ->getResult(0);
      } else {
        cur = makeOp(
                  b, loc, "epilogue_cast", {cur}, {b.getIndexType()},
                  {b.getNamedAttr("dtype", b.getStringAttr(mojoDType(
                                               elemTypeName(step.toType))))})
                  ->getResult(0);
      }
    }
  }
  return cur;
}

void emitStoreOut(OpBuilder &b, Location loc, Value outPtr, Value rowCoord,
                  Value colCoord, Value val, int64_t width,
                  const ViewDesc &outView) {
  SmallVector<Value> coords;
  if (outView.panelStrides.size() == 1)
    coords.push_back(rowCoord);
  else {
    coords.push_back(rowCoord);
    coords.push_back(colCoord);
  }
  Value addr = makeAddr(b, loc, outPtr, coords, outView.panelStrides);
  makeOp(b, loc, "store_vec", {outPtr, addr, val}, {},
         {b.getNamedAttr("width", b.getI64IntegerAttr(width)),
          b.getNamedAttr("lane_stride",
                         b.getI64IntegerAttr(outView.panelStrides.back()))});
}
} // namespace

llvm::Error describeMojoMatrixPackage(func::FuncOp fn, Operation &prog,
                                      llvm::raw_pwrite_stream &out) {
  auto abi = llvm::json::parse(fn->getAttrOfType<StringAttr>("kir.abi"));
  if (!abi)
    return abi.takeError();
  auto tile = prog.getAttrOfType<DenseI64ArrayAttr>("tile");
  bool usesBrgemm = prog.getAttrOfType<BoolAttr>("uses_brgemm").getValue();
  out << llvm::json::Value(llvm::json::Object{
      {"version", 2},
      {"source", ""},
      {"host", ""},
      {"returns_status", true},
      {"onednn", usesBrgemm},
      {"allow_subnormal_ftz", intentdsl::kir::allowsSubnormalFtz(fn)},
      {"domain", intentdsl::publishDomain(fn)},
      {"abi", std::move(*abi)},
      {"rows", llvm::json::Array{llvm::json::Object{{"grain", tile[0]},
                                                    {"columns", tile[1]}}}}});
  return llvm::Error::success();
}

LogicalResult lowerMojoMatrix(func::FuncOp fn) {
  auto *task = intentdsl::cpu::findMatrixTask(fn);
  if (!task)
    return failure();
  auto &domain = task->getRegion(0).front().front();
  auto &dot = intentdsl::cpu::calculation(domain);
  auto kind = dot.getName().stripDialect();
  auto extents = domain.getAttrOfType<DenseI64ArrayAttr>("cpu.extents");
  auto tile = domain.getAttrOfType<DenseI64ArrayAttr>("cpu.output_tile");
  auto implAttr = domain.getAttrOfType<StringAttr>("cpu.implementation");
  if (!implAttr)
    return domain.emitError("missing cpu.implementation at mojo@realized");
  StringRef implId = implAttr.getValue();
  auto packingAttr = domain.getAttrOfType<StringAttr>("cpu.packing");
  StringRef packing = packingAttr ? packingAttr.getValue() : StringRef();
  bool routed = domain.hasAttr("cpu.route_storage");
  bool usesBrgemm = implId.contains("onednn-brgemm");
  bool allowSubnormalFtz = intentdsl::kir::allowsSubnormalFtz(&dot);
  bool emitFtz = implId == "exits/mojo::onednn-brgemm-bf16";
  if (emitFtz && !allowSubnormalFtz)
    return dot.emitError(
        "onednn-brgemm-bf16 requires non-ieee denormal-fp-math");

  int64_t M = extents[0], N = extents[1], K = extents[2], G = extents[3];
  int64_t mr = tile[0], nr = tile[1], depth = tile[2], lanes = tile[3];

  auto abiJson =
      llvm::cantFail(llvm::json::parse(fn->getAttrOfType<StringAttr>("kir.abi")));
  auto slotOf = [](Value v) { return cast<BlockArgument>(v).getArgNumber(); };
  auto argDtype = [&](unsigned s) {
    return (*abiJson.getAsArray())[s]
        .getAsObject()
        ->getString("dtype")
        ->str();
  };
  auto argStrides = [&](unsigned s) {
    return fn.getArgAttrOfType<DenseI64ArrayAttr>(s, "cpu.strides");
  };
  auto argShape = [&](unsigned s) {
    return fn.getArgAttrOfType<DenseI64ArrayAttr>(s, "cpu.shape");
  };

  SmallVector<ViewDesc> views;
  auto addOrigin = [&](Value v, ArrayRef<std::string> idxExprs,
                       ArrayRef<int> panelAxes, bool isOut = false) {
    unsigned s = slotOf(v);
    auto st = argStrides(s);
    std::string addr = "0";
    for (auto [ax, expr] : llvm::enumerate(idxExprs))
      if (expr != "0")
        addr += "+(" + expr + ")*" + std::to_string(st[ax]) + "LL";
    SmallVector<int64_t> pSt;
    for (int ax : panelAxes)
      pSt.push_back(st[ax]);
    views.push_back({s, argDtype(s), addr, std::move(pSt), isOut});
  };

  int64_t offsetsSlot = -1, offsetsStride = 0;
  int64_t routesSlot = -1, routesStride = 0, multiplicity = 1;
  bool grouped = false;

  if (kind == "contract") {
    auto addPanel = [&](Value v, int kAx, int bAx, bool isLhs) {
      int rank = cast<MemRefType>(v.getType()).getRank();
      SmallVector<std::string> idx(rank, "0");
      int freeAx = 0;
      while (freeAx == kAx || freeAx == bAx)
        ++freeAx;
      idx[freeAx] = isLhs ? "src" : "c";
      if (bAx >= 0)
        idx[bAx] = "g";
      addOrigin(v, idx,
                isLhs ? ArrayRef<int>{freeAx, kAx}
                      : ArrayRef<int>{kAx, freeAx});
    };
    int ak = dot.getAttrOfType<IntegerAttr>("lhs_k").getInt();
    int bk = dot.getAttrOfType<IntegerAttr>("rhs_k").getInt();
    int ab = dot.getAttrOfType<IntegerAttr>("lhs_batch").getInt();
    int bb = dot.getAttrOfType<IntegerAttr>("rhs_batch").getInt();
    addPanel(dot.getOperand(0), ak, ab, true);
    addPanel(dot.getOperand(1), bk, bb, false);
    if (packing == "f32-panels-per-call") {
      views[0].dtype = views[1].dtype = "f32";
      views[0].panelStrides = {K, 1};
      views[1].panelStrides = {nr, 1};
    } else if (!packing.empty()) {
      views[1].panelStrides = {nr, 1};
    }
    if (dot.getNumOperands() > 2) {
      grouped = true;
      offsetsSlot = slotOf(dot.getOperand(2));
      offsetsStride = argStrides(offsetsSlot)[0];
    }
    if (dot.getNumOperands() > 3) {
      routesSlot = slotOf(dot.getOperand(3));
      routesStride = argStrides(routesSlot)[0];
      multiplicity = argShape(slotOf(domain.getOperand(0)))[1];
    }
  } else if (kind == "window_contract") {
    addOrigin(dot.getOperand(0), {"batch", "chunk*K", "head", "src"}, {3, 1});
    views.push_back({slotOf(dot.getOperand(1)),
                     "bf16",
                     "(g*K)*N+c",
                     {N, 1},
                     false});
    if (packing == "mapped-f32-panels") {
      views[0].dtype = views[1].dtype = "f32";
      views[0].hostOrigin = "(g*M+src)*K";
      views[0].panelStrides = {K, 1};
    }
  } else if (kind == "scaled_contract") {
    views.push_back(
        {slotOf(dot.getOperand(0)), "f32", "src*K", {K, 1}, false});
    views.push_back({slotOf(dot.getOperand(2)), "f32", "c", {N, 1}, false});
  } else {
    addOrigin(dot.getOperand(0), {"src", "0", "0"}, {0, 1, 2});
    Value x = dot.getOperand(1).getDefiningOp()->getOperand(0);
    addOrigin(x, {"0", "0"}, {0, 1});
    views[1].dtype = "i8";
  }

  Value dstArg = domain.getOperand(0);
  SmallVector<EpilogueStep> epilogue;
  bool usesRoundBf16 = false;
  for (auto &op : domain.getRegion(0).front()) {
    auto n = op.getName().stripDialect();
    if (n == "add_columns") {
      unsigned bIdx = views.size();
      addOrigin(
          op.getOperand(1),
          {cast<MemRefType>(dstArg.getType()).getRank() == 1 ? "dest" : "c"},
          {0});
      epilogue.push_back({EpilogueStep::Bias, bIdx, {}, {}});
    } else if (n == "convert_matrix") {
      Type fromT =
          cast<ShapedType>(op.getOperand(0).getType()).getElementType();
      Type toT = cast<ShapedType>(op.getResult(0).getType()).getElementType();
      if (toT.isBF16() && fromT.isF32())
        usesRoundBf16 = true;
      epilogue.push_back({EpilogueStep::Convert, 0, fromT, toT});
    }
  }

  if (kind == "window_contract")
    addOrigin(dstArg, {"batch", "chunk", "head", "dest", "c"}, {3, 4}, true);
  else if (routed)
    addOrigin(dstArg,
              {"(dest/" + std::to_string(multiplicity) + "LL)",
               "(dest%" + std::to_string(multiplicity) + "LL)", "c"},
              {0, 2}, true);
  else {
    int dstRank = cast<MemRefType>(dstArg.getType()).getRank();
    if (dstRank == 1)
      addOrigin(dstArg, {"dest"}, {0}, true);
    else if (dstRank == 2)
      addOrigin(dstArg, {"dest", "c"}, {0, 1}, true);
    else
      addOrigin(dstArg, {"g", "dest", "c"}, {1, 2}, true);
  }
  SmallVector<int64_t> origDstStrides(
      argStrides(slotOf(dstArg)).asArrayRef().begin(),
      argStrides(slotOf(dstArg)).asArrayRef().end());
  if (routed) {
    views.front().panelStrides = {K, 1};
    views.back().panelStrides = {nr, 1};
  }

  // Collect cpu@buffers preparation & release ops before replacing body
  SmallVector<Operation *> allocOps, prepOps, releaseOps;
  for (auto &op : fn.front()) {
    auto n = op.getName().getStringRef();
    if (n == "cpu.alloc_buffer")
      allocOps.push_back(&op);
    else if (n == "cpu.prepare_panels")
      prepOps.push_back(&op);
    else if (n == "cpu.release_buffer")
      releaseOps.push_back(&op);
  }

  Location loc = fn.getLoc();
  OpBuilder b(&fn.front().front());
  OperationState progState(loc, "mojo_row.matrix_program");
  progState.addAttribute("tile", tile);
  progState.addAttribute("extents", extents);
  progState.addAttribute("uses_brgemm", b.getBoolAttr(usesBrgemm));
  progState.addAttribute("implementation", implAttr);
  progState.addRegion();
  progState.addRegion();
  Operation *prog = b.create(progState);

  // ---------------- Region 0: Panel ("row.mojo") ----------------
  Block &panelTop = prog->getRegion(0).emplaceBlock();
  b.setInsertionPointToEnd(&panelTop);
  SmallVector<Attribute> paramDtypes;
  for (auto &v : views)
    paramDtypes.push_back(b.getStringAttr(mojoDType(v.dtype)));
  bool usesVnni = implId == "exits/mojo::vnni-dot4" ||
                  implId == "exits/mojo::q4k-vnni";
  bool usesStack = usesBrgemm;

  OperationState pfnState(loc, "mojo_row.panel_fn");
  pfnState.addAttribute("param_dtypes", b.getArrayAttr(paramDtypes));
  pfnState.addAttribute("uses_brgemm", b.getBoolAttr(usesBrgemm));
  pfnState.addAttribute("uses_round_bf16", b.getBoolAttr(usesRoundBf16));
  pfnState.addAttribute("uses_vnni", b.getBoolAttr(usesVnni));
  pfnState.addAttribute("uses_stack", b.getBoolAttr(usesStack));
  pfnState.addRegion();
  Operation *pfnOp = b.create(pfnState);
  makeOp(b, loc, "end");

  Block &pfnBb = pfnOp->getRegion(0).emplaceBlock();
  SmallVector<Value> panelPtrs;
  for (size_t i = 0; i < views.size(); ++i)
    panelPtrs.push_back(pfnBb.addArgument(b.getIndexType(), loc));
  Value argR = pfnBb.addArgument(b.getI64Type(), loc);
  Value argC = pfnBb.addArgument(b.getI64Type(), loc);
  Value argBrgH, argBrgS;
  if (usesBrgemm) {
    argBrgH = pfnBb.addArgument(b.getIndexType(), loc);
    argBrgS = pfnBb.addArgument(b.getIndexType(), loc);
  }
  b.setInsertionPointToEnd(&pfnBb);

  auto q4kStorageAttr = b.getNamedAttr(
      "kir.storage", TypeAttr::get(intentdsl::kir::Q4KType::get(fn.getContext())));
  auto implNamedAttr = b.getNamedAttr("ci.implementation", implAttr);

  if (implId == "exits/mojo::q4k-simd" || implId == "exits/mojo::q4k-vnni") {
    bool q4kVnni = implId == "exits/mojo::q4k-vnni";
    Value c0 = constIdx(b, loc, 0);
    Value rBound = makeOp(b, loc, "as_int", {argR}, {b.getIndexType()})->getResult(0);
    Value c1 = constIdx(b, loc, 1);
    OperationState rLoop(loc, "mojo_row.for_range");
    rLoop.addOperands({c0, rBound, c1});
    Block &rBb = rLoop.addRegion()->emplaceBlock();
    Value rIv = rBb.addArgument(b.getIndexType(), loc);
    b.create(rLoop);
    b.setInsertionPointToEnd(&rBb);

    Value accSlot =
        makeOp(b, loc, "acc_decl", {}, {b.getIndexType()},
               {b.getNamedAttr("dtype", b.getStringAttr("float32")),
                b.getNamedAttr("width", b.getI64IntegerAttr(1))})
            ->getResult(0);
    Value gBound = constIdx(b, loc, K / 256);
    OperationState gLoop(loc, "mojo_row.for_range");
    gLoop.addOperands({c0, gBound, c1});
    Block &gBb = gLoop.addRegion()->emplaceBlock();
    Value gIv = gBb.addArgument(b.getIndexType(), loc);
    b.create(gLoop);
    b.setInsertionPointToEnd(&gBb);

    Value wBase = makeAddr(b, loc, panelPtrs[0], {rIv, gIv, c0},
                           views[0].panelStrides);
    Value sbTerm =
        makeOp(b, loc, "q4k_superblock",
               {panelPtrs[0], panelPtrs[1], wBase, gIv}, {b.getIndexType()},
               {b.getNamedAttr("byte_stride",
                               b.getI64IntegerAttr(views[0].panelStrides[2])),
                b.getNamedAttr("use_vnni", b.getBoolAttr(q4kVnni)),
                q4kStorageAttr, implNamedAttr})
            ->getResult(0);
    Value prev =
        makeOp(b, loc, "acc_read", {accSlot}, {b.getIndexType()})->getResult(0);
    Value next = makeOp(b, loc, "epilogue_add", {prev, sbTerm}, {b.getIndexType()})
                     ->getResult(0);
    makeOp(b, loc, "acc_write", {accSlot, next});
    makeOp(b, loc, "end");

    b.setInsertionPointToEnd(&rBb);
    Value finalAcc =
        makeOp(b, loc, "acc_read", {accSlot}, {b.getIndexType()})->getResult(0);
    Value epi = emitEpilogue(b, loc, finalAcc, rIv, 1, epilogue, panelPtrs, views);
    emitStoreOut(b, loc, panelPtrs.back(), rIv, c0, epi, 1, views.back());
    makeOp(b, loc, "end");
    b.setInsertionPointToEnd(&pfnBb);
    makeOp(b, loc, "panel_return", {}, {},
           {b.getNamedAttr("code", b.getI32IntegerAttr(0))});
    makeOp(b, loc, "end");
  } else if (implId == "exits/mojo::onednn-brgemm-u8s8") {
    Value c0 = constIdx(b, loc, 0);
    Value c1 = constIdx(b, loc, 1);
    Value rBound = makeOp(b, loc, "as_int", {argR}, {b.getIndexType()})->getResult(0);
    Value u8Buf =
        makeOp(b, loc, "stack_alloc", {}, {b.getIndexType()},
               {b.getNamedAttr("dtype", b.getStringAttr("uint8")),
                b.getNamedAttr("count", b.getI64IntegerAttr(mr * 32))})
            ->getResult(0);
    Value s32Buf =
        makeOp(b, loc, "stack_alloc", {}, {b.getIndexType()},
               {b.getNamedAttr("dtype", b.getStringAttr("int32")),
                b.getNamedAttr("count", b.getI64IntegerAttr(mr))})
            ->getResult(0);
    Value accBuf =
        makeOp(b, loc, "stack_alloc", {}, {b.getIndexType()},
               {b.getNamedAttr("dtype", b.getStringAttr("float32")),
                b.getNamedAttr("count", b.getI64IntegerAttr(mr))})
            ->getResult(0);
    // Zero-init accBuf[0..R-1]
    {
      OperationState zLoop(loc, "mojo_row.for_range");
      zLoop.addOperands({c0, rBound, c1});
      Block &zBb = zLoop.addRegion()->emplaceBlock();
      Value zIv = zBb.addArgument(b.getIndexType(), loc);
      b.create(zLoop);
      b.setInsertionPointToEnd(&zBb);
      Value zVal =
          makeOp(b, loc, "acc_decl", {}, {b.getIndexType()},
                 {b.getNamedAttr("dtype", b.getStringAttr("float32")),
                  b.getNamedAttr("width", b.getI64IntegerAttr(1))})
              ->getResult(0);
      Value zAddr = makeAddr(b, loc, accBuf, {zIv}, {1});
      makeOp(b, loc, "store_vec", {accBuf, zAddr, zVal}, {},
             {b.getNamedAttr("width", b.getI64IntegerAttr(1)),
              b.getNamedAttr("lane_stride", b.getI64IntegerAttr(1))});
      makeOp(b, loc, "end");
      b.setInsertionPointToEnd(&pfnBb);
    }
    Value gBound = constIdx(b, loc, K / 256);
    OperationState gLoop(loc, "mojo_row.for_range");
    gLoop.addOperands({c0, gBound, c1});
    Block &gBb = gLoop.addRegion()->emplaceBlock();
    Value gIv = gBb.addArgument(b.getIndexType(), loc);
    b.create(gLoop);
    b.setInsertionPointToEnd(&gBb);

    Value jBound = constIdx(b, loc, 8);
    OperationState jLoop(loc, "mojo_row.for_range");
    jLoop.addOperands({c0, jBound, c1});
    Block &jBb = jLoop.addRegion()->emplaceBlock();
    Value jIv = jBb.addArgument(b.getIndexType(), loc);
    b.create(jLoop);
    b.setInsertionPointToEnd(&jBb);

    // Unpack nibbles for ri in 0..R-1
    {
      OperationState uLoop(loc, "mojo_row.for_range");
      uLoop.addOperands({c0, rBound, c1});
      Block &uBb = uLoop.addRegion()->emplaceBlock();
      Value uIv = uBb.addArgument(b.getIndexType(), loc);
      b.create(uLoop);
      b.setInsertionPointToEnd(&uBb);
      Value wBase =
          makeAddr(b, loc, panelPtrs[0], {uIv, gIv, c0}, views[0].panelStrides);
      makeOp(b, loc, "q4k_unpack_row",
             {panelPtrs[0], wBase, u8Buf, uIv, jIv}, {},
             {b.getNamedAttr("byte_stride",
                             b.getI64IntegerAttr(views[0].panelStrides[2])),
              q4kStorageAttr});
      makeOp(b, loc, "end");
      b.setInsertionPointToEnd(&jBb);
    }
    // Q8_K subgroup pointer: g * 292 + 4 + j * 32
    Value c4 = constIdx(b, loc, intentdsl::kir::Q8KRecord::values);
    Value qOff = makeAddr(
        b, loc, panelPtrs[1], {gIv, jIv, c4},
        {intentdsl::kir::Q8KRecord::bytes, intentdsl::kir::Q4KRecord::group, 1});
    Value qSubPtr =
        makeOp(b, loc, "ptr_offset", {panelPtrs[1], qOff}, {b.getIndexType()})
            ->getResult(0);
    Value subSt =
        makeOp(b, loc, "brgemm_call",
               {argBrgH, u8Buf, qSubPtr, s32Buf, argBrgS}, {b.getI32Type()},
               {b.getNamedAttr("allow_subnormal_ftz", b.getBoolAttr(false)),
                implNamedAttr})
            ->getResult(0);
    makeOp(b, loc, "check_status", {subSt});
    // Accumulate scaled subgroup for ri in 0..R-1
    {
      OperationState aLoop(loc, "mojo_row.for_range");
      aLoop.addOperands({c0, rBound, c1});
      Block &aBb = aLoop.addRegion()->emplaceBlock();
      Value aIv = aBb.addArgument(b.getIndexType(), loc);
      b.create(aLoop);
      b.setInsertionPointToEnd(&aBb);
      Value wBase =
          makeAddr(b, loc, panelPtrs[0], {aIv, gIv, c0}, views[0].panelStrides);
      makeOp(b, loc, "q4k_accum_row",
             {panelPtrs[0], wBase, panelPtrs[1], s32Buf, accBuf, aIv, gIv, jIv},
             {},
             {b.getNamedAttr("byte_stride",
                             b.getI64IntegerAttr(views[0].panelStrides[2])),
              q4kStorageAttr});
      makeOp(b, loc, "end");
      b.setInsertionPointToEnd(&jBb);
    }
    makeOp(b, loc, "end");
    b.setInsertionPointToEnd(&gBb);
    makeOp(b, loc, "end");
    b.setInsertionPointToEnd(&pfnBb);

    // Epilogue & store for ri in 0..R-1
    {
      OperationState eLoop(loc, "mojo_row.for_range");
      eLoop.addOperands({c0, rBound, c1});
      Block &eBb = eLoop.addRegion()->emplaceBlock();
      Value eIv = eBb.addArgument(b.getIndexType(), loc);
      b.create(eLoop);
      b.setInsertionPointToEnd(&eBb);
      Value aAddr = makeAddr(b, loc, accBuf, {eIv}, {1});
      Value accVal =
          makeOp(b, loc, "load_vec", {accBuf, aAddr}, {b.getIndexType()},
                 {b.getNamedAttr("dtype", b.getStringAttr("float32")),
                  b.getNamedAttr("width", b.getI64IntegerAttr(1)),
                  b.getNamedAttr("lane_stride", b.getI64IntegerAttr(1))})
              ->getResult(0);
      Value epi =
          emitEpilogue(b, loc, accVal, eIv, 1, epilogue, panelPtrs, views);
      emitStoreOut(b, loc, panelPtrs.back(), eIv, c0, epi, 1, views.back());
      makeOp(b, loc, "end");
      b.setInsertionPointToEnd(&pfnBb);
    }
    makeOp(b, loc, "panel_return", {}, {},
           {b.getNamedAttr("code", b.getI32IntegerAttr(0))});
    makeOp(b, loc, "end");
  } else if (usesBrgemm) {
    // BRGeMM tile path (f32, f16, bf16, s8)
    bool isInt = implId == "exits/mojo::onednn-brgemm-s8";
    StringRef accDtype = isInt ? "int32" : "float32";
    Value cBuf =
        makeOp(b, loc, "stack_alloc", {}, {b.getIndexType()},
               {b.getNamedAttr("dtype", b.getStringAttr(accDtype)),
                b.getNamedAttr("count", b.getI64IntegerAttr(mr * nr))})
            ->getResult(0);
    Value tileSt =
        makeOp(b, loc, "brgemm_call",
               {argBrgH, panelPtrs[0], panelPtrs[1], cBuf, argBrgS},
               {b.getI32Type()},
               {b.getNamedAttr("allow_subnormal_ftz", b.getBoolAttr(emitFtz)),
                implNamedAttr})
            ->getResult(0);
    makeOp(b, loc, "check_status", {tileSt});

    // Fast full-tile epilogue when R == mr and C == nr
    {
      OperationState fullIf(loc, "mojo_row.if_full_tile");
      fullIf.addOperands({argR, argC});
      fullIf.addAttribute("rows", b.getI64IntegerAttr(mr));
      fullIf.addAttribute("cols", b.getI64IntegerAttr(nr));
      fullIf.addAttribute("check_cols", b.getBoolAttr(true));
      Block &fBb = fullIf.addRegion()->emplaceBlock();
      b.create(fullIf);
      b.setInsertionPointToEnd(&fBb);
      int64_t vecs = nr / lanes;
      for (int64_t r = 0; r < mr; ++r) {
        Value rIdx = constIdx(b, loc, r);
        for (int64_t v = 0; v < vecs; ++v) {
          Value cIdx = constIdx(b, loc, v * lanes);
          Value cAddr = makeAddr(b, loc, cBuf, {rIdx, cIdx}, {nr, 1});
          Value vec =
              makeOp(b, loc, "load_vec", {cBuf, cAddr}, {b.getIndexType()},
                     {b.getNamedAttr("dtype", b.getStringAttr(accDtype)),
                      b.getNamedAttr("width", b.getI64IntegerAttr(lanes)),
                      b.getNamedAttr("lane_stride", b.getI64IntegerAttr(1))})
                  ->getResult(0);
          Value epi = emitEpilogue(b, loc, vec, cIdx, lanes, epilogue,
                                   panelPtrs, views);
          emitStoreOut(b, loc, panelPtrs.back(), rIdx, cIdx, epi, lanes,
                       views.back());
        }
      }
      makeOp(b, loc, "end");
      b.setInsertionPointToEnd(&pfnBb);
    }

    // Dynamic R x C tail epilogue (note: ldc == C on tail tiles!)
    Value c0 = constIdx(b, loc, 0);
    Value c1 = constIdx(b, loc, 1);
    Value rBound = makeOp(b, loc, "as_int", {argR}, {b.getIndexType()})->getResult(0);
    Value cBound = makeOp(b, loc, "as_int", {argC}, {b.getIndexType()})->getResult(0);
    OperationState rLoop(loc, "mojo_row.for_range");
    rLoop.addOperands({c0, rBound, c1});
    Block &rBb = rLoop.addRegion()->emplaceBlock();
    Value rIv = rBb.addArgument(b.getIndexType(), loc);
    b.create(rLoop);
    b.setInsertionPointToEnd(&rBb);

    Value rowBase = b.create<arith::MulIOp>(loc, rIv, cBound);
    Value laneStep = constIdx(b, loc, lanes);
    // Vector tail loop: while c + lanes <= C
    OperationState wVec(loc, "mojo_row.while_lt");
    wVec.addOperands({c0, cBound, laneStep});
    wVec.addAttribute("check_step", b.getBoolAttr(true));
    Block &wvBb = wVec.addRegion()->emplaceBlock();
    Value cVecIv = wvBb.addArgument(b.getIndexType(), loc);
    b.create(wVec);
    b.setInsertionPointToEnd(&wvBb);
    {
      Value flatIdx = b.create<arith::AddIOp>(loc, rowBase, cVecIv);
      Value cAddr = makeAddr(b, loc, cBuf, {flatIdx}, {1});
      Value vec =
          makeOp(b, loc, "load_vec", {cBuf, cAddr}, {b.getIndexType()},
                 {b.getNamedAttr("dtype", b.getStringAttr(accDtype)),
                  b.getNamedAttr("width", b.getI64IntegerAttr(lanes)),
                  b.getNamedAttr("lane_stride", b.getI64IntegerAttr(1))})
              ->getResult(0);
      Value epi = emitEpilogue(b, loc, vec, cVecIv, lanes, epilogue, panelPtrs,
                               views);
      emitStoreOut(b, loc, panelPtrs.back(), rIv, cVecIv, epi, lanes,
                   views.back());
      makeOp(b, loc, "end");
    }
    b.setInsertionPointToEnd(&rBb);

    // Scalar tail loop: while c < C (starting at (C // lanes) * lanes)
    Value vecDone = b.create<arith::MulIOp>(
        loc, b.create<arith::DivSIOp>(loc, cBound, laneStep), laneStep);
    OperationState wSca(loc, "mojo_row.while_lt");
    wSca.addOperands({vecDone, cBound, c1});
    wSca.addAttribute("check_step", b.getBoolAttr(false));
    Block &wsBb = wSca.addRegion()->emplaceBlock();
    Value cScaIv = wsBb.addArgument(b.getIndexType(), loc);
    b.create(wSca);
    b.setInsertionPointToEnd(&wsBb);
    {
      Value flatIdx = b.create<arith::AddIOp>(loc, rowBase, cScaIv);
      Value cAddr = makeAddr(b, loc, cBuf, {flatIdx}, {1});
      Value sca =
          makeOp(b, loc, "load_vec", {cBuf, cAddr}, {b.getIndexType()},
                 {b.getNamedAttr("dtype", b.getStringAttr(accDtype)),
                  b.getNamedAttr("width", b.getI64IntegerAttr(1)),
                  b.getNamedAttr("lane_stride", b.getI64IntegerAttr(1))})
              ->getResult(0);
      Value epi = emitEpilogue(b, loc, sca, cScaIv, 1, epilogue, panelPtrs,
                               views);
      emitStoreOut(b, loc, panelPtrs.back(), rIv, cScaIv, epi, 1, views.back());
      makeOp(b, loc, "end");
    }
    b.setInsertionPointToEnd(&rBb);
    makeOp(b, loc, "end");
    b.setInsertionPointToEnd(&pfnBb);
    makeOp(b, loc, "panel_return", {}, {},
           {b.getNamedAttr("code", b.getI32IntegerAttr(0))});
    makeOp(b, loc, "end");
  } else if (usesVnni) {
    // Native AVX-512 VNNI dot4 path
    int64_t vectors = nr / 16;
    auto emitVnniBody = [&](int64_t rows, Value rBase, Block *targetBb) {
      b.setInsertionPointToEnd(targetBb);
      SmallVector<Value> accs(rows * vectors);
      for (int64_t v = 0; v < vectors; ++v) {
        Value corrIdx = constIdx(b, loc, K * nr + 64 * v);
        Value corrAddr = makeAddr(b, loc, panelPtrs[1], {corrIdx}, {1});
        Value corr = makeOp(b, loc, "load_bitcast_vec",
                            {panelPtrs[1], corrAddr}, {b.getIndexType()},
                            {b.getNamedAttr("scalar_type",
                                            b.getStringAttr("Int32")),
                             b.getNamedAttr("width", b.getI64IntegerAttr(16))})
                         ->getResult(0);
        for (int64_t m = 0; m < rows; ++m)
          accs[m * vectors + v] =
              makeOp(b, loc, "acc_decl", {corr}, {b.getIndexType()},
                     {b.getNamedAttr("dtype", b.getStringAttr("int32")),
                      b.getNamedAttr("width", b.getI64IntegerAttr(16))})
                  ->getResult(0);
      }
      Value c0 = constIdx(b, loc, 0), kBound = constIdx(b, loc, K),
            c4 = constIdx(b, loc, 4);
      OperationState kLoop(loc, "mojo_row.for_range");
      kLoop.addOperands({c0, kBound, c4});
      Block &kBb = kLoop.addRegion()->emplaceBlock();
      Value kIv = kBb.addArgument(b.getIndexType(), loc);
      b.create(kLoop);
      b.setInsertionPointToEnd(&kBb);

      SmallVector<Value> bVecs(vectors);
      for (int64_t v = 0; v < vectors; ++v) {
        Value vOff = constIdx(b, loc, 64 * v);
        Value bAddr = makeAddr(b, loc, panelPtrs[1], {kIv, vOff}, {nr, 1});
        bVecs[v] =
            makeOp(b, loc, "load_vec", {panelPtrs[1], bAddr},
                   {b.getIndexType()},
                   {b.getNamedAttr("dtype", b.getStringAttr("int8")),
                    b.getNamedAttr("width", b.getI64IntegerAttr(64)),
                    b.getNamedAttr("lane_stride", b.getI64IntegerAttr(1))})
                ->getResult(0);
      }
      for (int64_t m = 0; m < rows; ++m) {
        Value rCoord = addOffset(b, loc, rBase, m);
        Value aAddr = makeAddr(b, loc, panelPtrs[0], {rCoord, kIv},
                               views[0].panelStrides);
        Value aVec =
            makeOp(b, loc, "load_i32_xor", {panelPtrs[0], aAddr},
                   {b.getIndexType()})
                ->getResult(0);
        for (int64_t v = 0; v < vectors; ++v) {
          Value slot = accs[m * vectors + v];
          Value cur = makeOp(b, loc, "acc_read", {slot}, {b.getIndexType()})
                          ->getResult(0);
          Value upd = makeOp(b, loc, "vnni_dot4", {cur, aVec, bVecs[v]},
                             {b.getIndexType()}, {implNamedAttr})
                          ->getResult(0);
          makeOp(b, loc, "acc_write", {slot, upd});
        }
      }
      makeOp(b, loc, "end");
      b.setInsertionPointToEnd(targetBb);
      for (int64_t m = 0; m < rows; ++m) {
        Value rCoord = addOffset(b, loc, rBase, m);
        for (int64_t v = 0; v < vectors; ++v) {
          Value cCoord = constIdx(b, loc, v * 16);
          Value accVal =
              makeOp(b, loc, "acc_read", {accs[m * vectors + v]},
                     {b.getIndexType()})
                  ->getResult(0);
          Value epi = emitEpilogue(b, loc, accVal, cCoord, 16, epilogue,
                                   panelPtrs, views);
          emitStoreOut(b, loc, panelPtrs.back(), rCoord, cCoord, epi, 16,
                       views.back());
        }
      }
    };

    {
      OperationState fullIf(loc, "mojo_row.if_full_tile");
      fullIf.addOperands({argR, argC});
      fullIf.addAttribute("rows", b.getI64IntegerAttr(mr));
      fullIf.addAttribute("cols", b.getI64IntegerAttr(nr));
      fullIf.addAttribute("check_cols", b.getBoolAttr(false));
      Block &fBb = fullIf.addRegion()->emplaceBlock();
      b.create(fullIf);
      b.setInsertionPointToEnd(&fBb);
      emitVnniBody(mr, constIdx(b, loc, 0), &fBb);
      makeOp(b, loc, "end");
      b.setInsertionPointToEnd(&pfnBb);
    }
    Value c0 = constIdx(b, loc, 0), c1 = constIdx(b, loc, 1);
    Value rBound = makeOp(b, loc, "as_int", {argR}, {b.getIndexType()})->getResult(0);
    OperationState rLoop(loc, "mojo_row.for_range");
    rLoop.addOperands({c0, rBound, c1});
    Block &rBb = rLoop.addRegion()->emplaceBlock();
    Value rIv = rBb.addArgument(b.getIndexType(), loc);
    b.create(rLoop);
    emitVnniBody(1, rIv, &rBb);
    makeOp(b, loc, "end");
    b.setInsertionPointToEnd(&pfnBb);
    makeOp(b, loc, "panel_return", {}, {},
           {b.getNamedAttr("code", b.getI32IntegerAttr(0))});
    makeOp(b, loc, "end");
  } else {
    // Native SIMD FMA path
    bool isInt = views[0].dtype == "i8";
    StringRef accDtype = isInt ? "int32" : "float32";
    auto emitSimdBody = [&](int64_t w, int64_t rows, int64_t vectors,
                            Value rBase, Value cBase, Block *targetBb) {
      b.setInsertionPointToEnd(targetBb);
      SmallVector<Value> accs(rows * vectors);
      for (int64_t r = 0; r < rows; ++r)
        for (int64_t v = 0; v < vectors; ++v)
          accs[r * vectors + v] =
              makeOp(b, loc, "acc_decl", {}, {b.getIndexType()},
                     {b.getNamedAttr("dtype", b.getStringAttr(accDtype)),
                      b.getNamedAttr("width", b.getI64IntegerAttr(w))})
                  ->getResult(0);

      Value c0 = constIdx(b, loc, 0), kTotal = constIdx(b, loc, K),
            kStep = constIdx(b, loc, depth), c1 = constIdx(b, loc, 1);
      Block *pBb = nullptr;
      Value kLb = c0, kUb = kTotal;
      if (K > depth) {
        OperationState pLoop(loc, "mojo_row.for_range");
        pLoop.addOperands({c0, kTotal, kStep});
        pBb = &pLoop.addRegion()->emplaceBlock();
        Value pIv = pBb->addArgument(b.getIndexType(), loc);
        b.create(pLoop);
        b.setInsertionPointToEnd(pBb);
        kLb = pIv;
        Value nextP = b.create<arith::AddIOp>(loc, pIv, kStep);
        kUb = (K % depth == 0)
                  ? nextP
                  : b.create<arith::MinSIOp>(loc, nextP, kTotal).getResult();
      }
      OperationState kLoop(loc, "mojo_row.for_range");
      kLoop.addOperands({kLb, kUb, c1});
      Block &kBb = kLoop.addRegion()->emplaceBlock();
      Value kIv = kBb.addArgument(b.getIndexType(), loc);
      b.create(kLoop);
      b.setInsertionPointToEnd(&kBb);

      SmallVector<Value> bVecs(vectors);
      for (int64_t v = 0; v < vectors; ++v) {
        Value cCoord = addOffset(b, loc, cBase, v * w);
        Value bAddr = makeAddr(b, loc, panelPtrs[1], {kIv, cCoord},
                               views[1].panelStrides);
        Value raw =
            makeOp(b, loc, "load_vec", {panelPtrs[1], bAddr},
                   {b.getIndexType()},
                   {b.getNamedAttr("dtype",
                                   b.getStringAttr(mojoDType(views[1].dtype))),
                    b.getNamedAttr("width", b.getI64IntegerAttr(w)),
                    b.getNamedAttr("lane_stride",
                                   b.getI64IntegerAttr(views[1].panelStrides[1]))})
                ->getResult(0);
        bVecs[v] =
            makeOp(b, loc, "splat_cast", {raw}, {b.getIndexType()},
                   {b.getNamedAttr("dtype", b.getStringAttr(accDtype)),
                    b.getNamedAttr("width", b.getI64IntegerAttr(w)),
                    b.getNamedAttr("from_scalar", b.getBoolAttr(false))})
                ->getResult(0);
      }
      bool contigLhsRows = w > 1 && rows >= 2 && (rows & (rows - 1)) == 0 &&
                           views[0].panelStrides[0] == 1 &&
                           (rows * vectors + vectors + 2 <= 32);
      Value aColVec;
      if (contigLhsRows) {
        Value aColAddr =
            makeAddr(b, loc, panelPtrs[0], {rBase, kIv}, views[0].panelStrides);
        Value aRawCol =
            makeOp(b, loc, "load_vec", {panelPtrs[0], aColAddr},
                   {b.getIndexType()},
                   {b.getNamedAttr("dtype",
                                   b.getStringAttr(mojoDType(views[0].dtype))),
                    b.getNamedAttr("width", b.getI64IntegerAttr(rows)),
                    b.getNamedAttr("lane_stride", b.getI64IntegerAttr(1))})
                ->getResult(0);
        aColVec =
            makeOp(b, loc, "splat_cast", {aRawCol}, {b.getIndexType()},
                   {b.getNamedAttr("dtype", b.getStringAttr(accDtype)),
                    b.getNamedAttr("width", b.getI64IntegerAttr(rows)),
                    b.getNamedAttr("from_scalar", b.getBoolAttr(false))})
                ->getResult(0);
      }
      for (int64_t r = 0; r < rows; ++r) {
        Value aVec;
        if (contigLhsRows) {
          aVec =
              makeOp(b, loc, "splat_lane", {aColVec}, {b.getIndexType()},
                     {b.getNamedAttr("dtype", b.getStringAttr(accDtype)),
                      b.getNamedAttr("width", b.getI64IntegerAttr(w)),
                      b.getNamedAttr("lane", b.getI64IntegerAttr(r))})
                  ->getResult(0);
        } else {
          Value rCoord = addOffset(b, loc, rBase, r);
          Value aAddr = makeAddr(b, loc, panelPtrs[0], {rCoord, kIv},
                                 views[0].panelStrides);
          Value aSca =
              makeOp(b, loc, "load_vec", {panelPtrs[0], aAddr},
                     {b.getIndexType()},
                     {b.getNamedAttr("dtype",
                                     b.getStringAttr(mojoDType(views[0].dtype))),
                      b.getNamedAttr("width", b.getI64IntegerAttr(1)),
                      b.getNamedAttr("lane_stride", b.getI64IntegerAttr(1))})
                  ->getResult(0);
          aVec =
              makeOp(b, loc, "splat_cast", {aSca}, {b.getIndexType()},
                     {b.getNamedAttr("dtype", b.getStringAttr(accDtype)),
                      b.getNamedAttr("width", b.getI64IntegerAttr(w)),
                      b.getNamedAttr("from_scalar", b.getBoolAttr(true))})
                  ->getResult(0);
        }
        for (int64_t v = 0; v < vectors; ++v) {
          Value slot = accs[r * vectors + v];
          Value cur = makeOp(b, loc, "acc_read", {slot}, {b.getIndexType()})
                          ->getResult(0);
          Value upd =
              makeOp(b, loc, "simd_fma", {cur, aVec, bVecs[v]},
                     {b.getIndexType()},
                     {b.getNamedAttr("is_int", b.getBoolAttr(isInt)),
                      implNamedAttr})
                  ->getResult(0);
          makeOp(b, loc, "acc_write", {slot, upd});
        }
      }
      makeOp(b, loc, "end");
      if (pBb) {
        b.setInsertionPointToEnd(pBb);
        makeOp(b, loc, "end");
      }
      b.setInsertionPointToEnd(targetBb);

      for (int64_t r = 0; r < rows; ++r) {
        Value rCoord = addOffset(b, loc, rBase, r);
        for (int64_t v = 0; v < vectors; ++v) {
          Value cCoord = addOffset(b, loc, cBase, v * w);
          Value accVal =
              makeOp(b, loc, "acc_read", {accs[r * vectors + v]},
                     {b.getIndexType()})
                  ->getResult(0);
          Value epi = emitEpilogue(b, loc, accVal, cCoord, w, epilogue,
                                   panelPtrs, views);
          emitStoreOut(b, loc, panelPtrs.back(), rCoord, cCoord, epi, w,
                       views.back());
        }
      }
    };

    {
      OperationState fullIf(loc, "mojo_row.if_full_tile");
      fullIf.addOperands({argR, argC});
      fullIf.addAttribute("rows", b.getI64IntegerAttr(mr));
      fullIf.addAttribute("cols", b.getI64IntegerAttr(nr));
      fullIf.addAttribute("check_cols", b.getBoolAttr(true));
      Block &fBb = fullIf.addRegion()->emplaceBlock();
      b.create(fullIf);
      b.setInsertionPointToEnd(&fBb);
      Value z0 = constIdx(b, loc, 0);
      emitSimdBody(lanes, mr, nr / lanes, z0, z0, &fBb);
      makeOp(b, loc, "end");
      b.setInsertionPointToEnd(&pfnBb);
    }

    Value c0 = constIdx(b, loc, 0), c1 = constIdx(b, loc, 1);
    Value rBound = makeOp(b, loc, "as_int", {argR}, {b.getIndexType()})->getResult(0);
    Value cBound = makeOp(b, loc, "as_int", {argC}, {b.getIndexType()})->getResult(0);
    Value laneStep = constIdx(b, loc, lanes);

    OperationState rLoop(loc, "mojo_row.for_range");
    rLoop.addOperands({c0, rBound, c1});
    Block &rBb = rLoop.addRegion()->emplaceBlock();
    Value rIv = rBb.addArgument(b.getIndexType(), loc);
    b.create(rLoop);
    b.setInsertionPointToEnd(&rBb);

    OperationState wVec(loc, "mojo_row.while_lt");
    wVec.addOperands({c0, cBound, laneStep});
    wVec.addAttribute("check_step", b.getBoolAttr(true));
    Block &wvBb = wVec.addRegion()->emplaceBlock();
    Value cVecIv = wvBb.addArgument(b.getIndexType(), loc);
    b.create(wVec);
    emitSimdBody(lanes, 1, 1, rIv, cVecIv, &wvBb);
    makeOp(b, loc, "end");
    b.setInsertionPointToEnd(&rBb);

    Value vecDone = b.create<arith::MulIOp>(
        loc, b.create<arith::DivSIOp>(loc, cBound, laneStep), laneStep);
    OperationState wSca(loc, "mojo_row.while_lt");
    wSca.addOperands({vecDone, cBound, c1});
    wSca.addAttribute("check_step", b.getBoolAttr(false));
    Block &wsBb = wSca.addRegion()->emplaceBlock();
    Value cScaIv = wsBb.addArgument(b.getIndexType(), loc);
    b.create(wSca);
    emitSimdBody(1, 1, 1, rIv, cScaIv, &wsBb);
    makeOp(b, loc, "end");
    b.setInsertionPointToEnd(&rBb);
    makeOp(b, loc, "end");
    b.setInsertionPointToEnd(&pfnBb);
    makeOp(b, loc, "panel_return", {}, {},
           {b.getNamedAttr("code", b.getI32IntegerAttr(0))});
    makeOp(b, loc, "end");
  }

  // ---------------- Region 1: Host ("entry.c") ----------------
  Block &hostTop = prog->getRegion(1).emplaceBlock();
  b.setInsertionPointToEnd(&hostTop);

  bool usesQ8k = packing == "shared-q8k-per-call";
  bool usesBf16Math = packing == "mapped-right-per-call" ||
                      packing == "mapped-f32-panels" ||
                      (packing == "f32-panels-per-call" &&
                       argDtype(slotOf(dot.getOperand(0))) == "bf16");
  bool usesF16Math = packing == "f32-panels-per-call" &&
                     argDtype(slotOf(dot.getOperand(0))) == "f16";
  bool usesMx8Math = packing == "decoded-f32-per-call";

  OperationState hfnState(loc, "mojo_row.host_fn");
  hfnState.addAttribute("params",
                        b.getStringAttr(intentdsl::cpu::matrixHostParams(fn)));
  hfnState.addAttribute("extents", extents);
  hfnState.addAttribute("tile", tile);
  hfnState.addAttribute("uses_brgemm", b.getBoolAttr(usesBrgemm));
  hfnState.addAttribute("uses_q8k", b.getBoolAttr(usesQ8k));
  hfnState.addAttribute("uses_bf16_math", b.getBoolAttr(usesBf16Math));
  hfnState.addAttribute("uses_f16_math", b.getBoolAttr(usesF16Math));
  hfnState.addAttribute("uses_mx8_math", b.getBoolAttr(usesMx8Math));
  hfnState.addAttribute("panel_arity", b.getI64IntegerAttr(views.size()));
  hfnState.addRegion();
  Operation *hfnOp = b.create(hfnState);
  makeOp(b, loc, "end");
  Block &hBb = hfnOp->getRegion(0).emplaceBlock();
  b.setInsertionPointToEnd(&hBb);
  makeOp(b, loc, "end");

  auto openHostBlock = [&](StringRef opName,
                           ArrayRef<NamedAttribute> attrs) -> Block * {
    OperationState s(loc, ("mojo_row." + opName).str());
    s.addAttributes(attrs);
    Block &bb = s.addRegion()->emplaceBlock();
    b.create(s);
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToEnd(&bb);
    makeOp(b, loc, "end");
    return &bb;
  };
  auto enterBlock = [&](Block *bb) { b.setInsertionPoint(bb->getTerminator()); };
  auto emitLet = [&](StringRef name, StringRef expr) {
    makeOp(b, loc, "host_let_i64", {}, {},
           {b.getNamedAttr("name", b.getStringAttr(name)),
            b.getNamedAttr("expr", b.getStringAttr(expr))});
  };
  auto emitPtr = [&](StringRef cTy, StringRef name, StringRef base,
                     StringRef off) {
    makeOp(b, loc, "host_ptr_offset", {}, {},
           {b.getNamedAttr("c_type", b.getStringAttr(cTy)),
            b.getNamedAttr("name", b.getStringAttr(name)),
            b.getNamedAttr("base_expr", b.getStringAttr(base)),
            b.getNamedAttr("offset_expr", b.getStringAttr(off))});
  };
  auto emitSpan = [&](StringRef opKind, StringRef dstE, StringRef srcE,
                      StringRef cntE, int64_t eb) {
    makeOp(b, loc, "host_span_op", {}, {},
           {b.getNamedAttr("op", b.getStringAttr(opKind)),
            b.getNamedAttr("dst_expr", b.getStringAttr(dstE)),
            b.getNamedAttr("src_expr", b.getStringAttr(srcE)),
            b.getNamedAttr("count_expr", b.getStringAttr(cntE)),
            b.getNamedAttr("elem_bytes", b.getI64IntegerAttr(eb))});
  };
  auto emitElem = [&](StringRef opKind, StringRef dstE, StringRef srcE,
                      StringRef auxE, int64_t eb) {
    makeOp(b, loc, "host_elem_op", {}, {},
           {b.getNamedAttr("op", b.getStringAttr(opKind)),
            b.getNamedAttr("dst_expr", b.getStringAttr(dstE)),
            b.getNamedAttr("src_expr", b.getStringAttr(srcE)),
            b.getNamedAttr("aux_expr", b.getStringAttr(auxE)),
            b.getNamedAttr("elem_bytes", b.getI64IntegerAttr(eb))});
  };
  auto makeHostIf = [&](StringRef cond) {
    return openHostBlock("host_if",
                         {b.getNamedAttr("cond", b.getStringAttr(cond))});
  };
  auto makeHostFor = [&](StringRef iv, StringRef lb, StringRef ub,
                         StringRef step) {
    return openHostBlock("host_for",
                         {b.getNamedAttr("iv", b.getStringAttr(iv)),
                          b.getNamedAttr("lb", b.getStringAttr(lb)),
                          b.getNamedAttr("ub", b.getStringAttr(ub)),
                          b.getNamedAttr("step", b.getStringAttr(step))});
  };
  auto makeHostOmpFor = [&](StringRef iv, StringRef lb, StringRef ub,
                            StringRef step, bool parallel, bool nowait,
                            bool simd, StringRef ifCond) {
    return openHostBlock(
        "host_omp_for",
        {b.getNamedAttr("iv", b.getStringAttr(iv)),
         b.getNamedAttr("lb", b.getStringAttr(lb)),
         b.getNamedAttr("ub", b.getStringAttr(ub)),
         b.getNamedAttr("step", b.getStringAttr(step)),
         b.getNamedAttr("parallel", b.getBoolAttr(parallel)),
         b.getNamedAttr("nowait", b.getBoolAttr(nowait)),
         b.getNamedAttr("simd", b.getBoolAttr(simd)),
         b.getNamedAttr("if_cond", b.getStringAttr(ifCond))});
  };

  llvm::DenseMap<Value, std::string> bufNames;
  SmallVector<Attribute> activeHeap;
  int bufCounter = 0;
  enterBlock(&hBb);
  for (Operation *aop : allocOps) {
    StringRef role = aop->getAttrOfType<StringAttr>("role").getValue();
    StringRef place = aop->getAttrOfType<StringAttr>("placement").getValue();
    int64_t elemBytes =
        aop->getAttrOfType<IntegerAttr>("element_bytes").getInt();
    std::string name = role == "lhs-panels"    ? "left"
                       : role == "rhs-panels"  ? "rhs"
                       : role == "q8k-records" ? "packed"
                       : role == "routed-lhs"  ? "members"
                                               : ("buf" + std::to_string(bufCounter++));
    bufNames[aop->getResult(0)] = name;
    std::string cType = name == "packed" ? "struct Record"
                        : elemBytes == 4 ? "float"
                        : elemBytes == 2 ? "uint16_t"
                                         : "unsigned char";
    int64_t count = aop->getAttrOfType<IntegerAttr>("elements").getInt();
    makeOp(b, loc, "host_alloc", {}, {},
           {b.getNamedAttr("buf_name", b.getStringAttr(name)),
            b.getNamedAttr("placement", b.getStringAttr(place)),
            b.getNamedAttr("c_type", b.getStringAttr(cType)),
            b.getNamedAttr("count_expr",
                           b.getStringAttr(std::to_string(count) + "ULL")),
            b.getNamedAttr("elem_bytes", b.getI64IntegerAttr(elemBytes)),
            b.getNamedAttr("cleanup", b.getArrayAttr(activeHeap))});
    if (place == "heap")
      activeHeap.push_back(b.getStringAttr(name));
  }

  enterBlock(&hBb);
  if (usesBrgemm) {
    bool isQ4k = implId == "exits/mojo::onednn-brgemm-u8s8";
    int64_t aDt = implId == "exits/mojo::onednn-brgemm-f32"    ? 0
                  : implId == "exits/mojo::onednn-brgemm-bf16" ? 1
                  : implId == "exits/mojo::onednn-brgemm-f16"  ? 2
                  : implId == "exits/mojo::onednn-brgemm-s8"   ? 3
                                                               : 4;
    int64_t bDt = isQ4k ? 3 : aDt;
    int64_t cDt = (aDt == 3 || isQ4k) ? 5 : 0;
    std::string ldaExpr =
        isQ4k ? "32" : std::to_string(views[0].panelStrides[0]);
    std::string ldbExpr =
        isQ4k ? "1" : std::to_string(views[1].panelStrides[0]);
    makeOp(b, loc, "host_brgemm_plan_decl");
    Block *planIf = makeHostIf("jobs > 0 && K > 0");
    enterBlock(planIf);
    if (grouped) {
      Block *gBb = makeHostFor("g", "0", "G", "1");
      enterBlock(gBb);
      std::string lenE =
          "((const int32_t*)a" + std::to_string(offsetsSlot) + ")[(g+1)*" +
          std::to_string(offsetsStride) + "LL] - ((const int32_t*)a" +
          std::to_string(offsetsSlot) + ")[g*" + std::to_string(offsetsStride) +
          "LL]";
      makeOp(b, loc, "host_brgemm_mark_rows", {}, {},
             {b.getNamedAttr("len_expr", b.getStringAttr(lenE))});
      enterBlock(planIf);
    } else {
      makeOp(b, loc, "host_brgemm_mark_rows", {}, {},
             {b.getNamedAttr("len_expr", b.getStringAttr("M"))});
    }
    Block *ciBb = makeHostFor("ci", "0", "2", "1");
    enterBlock(ciBb);
    emitLet("c_val", "ci == 0 ? (N >= nr ? nr : 0) : (N % nr)");
    Block *cPosIf = makeHostIf("c_val > 0");
    enterBlock(cPosIf);
    Block *rtBb = makeHostFor("rt", "1", "mr + 1", "1");
    enterBlock(rtBb);
    Block *seenIf = makeHostIf("r_seen[rt]");
    enterBlock(seenIf);
    makeOp(b, loc, "host_brgemm_lookup", {}, {},
           {b.getNamedAttr("a_dt", b.getI64IntegerAttr(aDt)),
            b.getNamedAttr("b_dt", b.getI64IntegerAttr(bDt)),
            b.getNamedAttr("c_dt", b.getI64IntegerAttr(cDt)),
            b.getNamedAttr("lda_expr", b.getStringAttr(ldaExpr)),
            b.getNamedAttr("ldb_expr", b.getStringAttr(ldbExpr)),
            b.getNamedAttr("n_expr", b.getStringAttr(isQ4k ? "1" : "c_val")),
            b.getNamedAttr("k_expr", b.getStringAttr(isQ4k ? "32" : "K")),
            b.getNamedAttr("cleanup", b.getArrayAttr(activeHeap))});
  }

  enterBlock(&hBb);
  bool windowed = kind == "window_contract";
  int64_t headExt = 1, chunkExt = 1;
  if (windowed) {
    auto outSh = argShape(slotOf(dstArg));
    headExt = std::max<int64_t>(1, outSh[2]);
    chunkExt = std::max<int64_t>(1, outSh[1]);
  }
  Block *jobsIf = makeHostIf("jobs > 0");
  enterBlock(jobsIf);
  std::string ntExpr =
      prepOps.empty()
          ? "jobs < workers ? jobs : workers"
          : "(G*(M*K + K*N) > 32768) ? workers : (jobs < workers ? jobs : workers)";
  emitLet("nt", ntExpr);
  Block *parBb = openHostBlock(
      "host_omp_parallel",
      {b.getNamedAttr("num_threads", b.getStringAttr("(int)nt")),
       b.getNamedAttr("if_cond", b.getStringAttr("nt > 1")),
       b.getNamedAttr("reduce_rc", b.getBoolAttr(true))});

  for (auto [popIdx, pop] : llvm::enumerate(prepOps)) {
    bool hasNextPrep = popIdx + 1 < prepOps.size();
    OpBuilder::InsertionGuard prepGuard(b);
    enterBlock(parBb);
    StringRef pk = pop->getAttrOfType<StringAttr>("kind").getValue();
    SmallVector<int64_t> opSlots;
    for (Value v : pop->getOperands().drop_back())
      opSlots.push_back(slotOf(v));
    std::string dst = bufNames[pop->getOperands().back()];

    if (pk == "map-window-rhs-f32" || pk == "map-window-rhs-bf16") {
      auto outSh = argShape(slotOf(dstArg));
      int64_t H = std::max<int64_t>(1, outSh[2]);
      int64_t chunks = std::max<int64_t>(1, outSh[1]);
      int64_t hg = pop->getAttrOfType<IntegerAttr>("head_group").getInt();
      auto idxExpr = [&](unsigned idx, ArrayRef<std::string> coords) {
        auto st = argStrides(opSlots[idx]);
        std::string s = "0";
        for (auto [ax, e] : llvm::enumerate(coords))
          s += "+(" + e + ")*" + std::to_string(st[ax]) + "LL";
        return s;
      };
      Block *gBb = makeHostOmpFor("g", "0", "G", "1", false, false, false, "");
      enterBlock(gBb);
      emitLet("head", "g%" + std::to_string(H));
      emitLet("chunk",
              "g/" + std::to_string(H) + "%" + std::to_string(chunks));
      emitLet("batch",
              "g/" + std::to_string(H) + "/" + std::to_string(chunks));
      emitLet("group", "head/" + std::to_string(hg));

      auto &srcBb = pop->getRegion(0).front();
      Operation *term = srcBb.getTerminator();
      llvm::DenseSet<Value> dependsOnCol{srcBb.getArgument(0)};
      bool hasStage1Transcendental = false;
      for (auto &mop : srcBb.without_terminator()) {
        if (llvm::any_of(mop.getOperands(),
                         [&](Value v) { return dependsOnCol.contains(v); })) {
          dependsOnCol.insert(mop.getResult(0));
        } else {
          auto mn = mop.getName().getStringRef();
          if (mn == "math.exp" || mn == "math.exp2" || mn == "math.log")
            hasStage1Transcendental = true;
        }
      }
      SmallVector<Value, 4> kCut;
      bool canFissionK = hasStage1Transcendental && K > 0 && K <= 512;
      if (canFissionK) {
        auto maybeAddCut = [&](Value v) {
          bool usedInStage2 =
              (!dependsOnCol.contains(term->getOperand(0)) &&
               term->getOperand(0) == v) ||
              llvm::any_of(v.getUsers(), [&](Operation *u) {
                return u == term ||
                       (u->getNumResults() == 1 &&
                        dependsOnCol.contains(u->getResult(0)));
              });
          if (!usedInStage2)
            return;
          if (!v.getType().isF32()) {
            canFissionK = false;
            return;
          }
          if (!llvm::is_contained(kCut, v))
            kCut.push_back(v);
        };
        maybeAddCut(srcBb.getArgument(1));
        maybeAddCut(srcBb.getArgument(2));
        for (auto &mop : srcBb.without_terminator()) {
          if (!dependsOnCol.contains(mop.getResult(0)))
            maybeAddCut(mop.getResult(0));
        }
        if (kCut.empty() || kCut.size() > 4)
          canFissionK = false;
      }

      Value ld3 =
          makeOp(b, loc, "host_load_scalar", {}, {srcBb.getArgument(3).getType()},
                 {b.getNamedAttr("slot", b.getI64IntegerAttr(opSlots[2])),
                  b.getNamedAttr("c_type", b.getStringAttr("float")),
                  b.getNamedAttr("index_expr",
                                 b.getStringAttr(idxExpr(
                                     2, {"batch", "head", "chunk", "K-1"})))})
              ->getResult(0);
      if (canFissionK) {
        for (size_t ci = 0; ci < kCut.size(); ++ci) {
          makeOp(b, loc, "host_alloc", {}, {},
                 {b.getNamedAttr("buf_name",
                                 b.getStringAttr("wk_" + std::to_string(ci))),
                  b.getNamedAttr("placement", b.getStringAttr("stack")),
                  b.getNamedAttr("c_type", b.getStringAttr("float")),
                  b.getNamedAttr("count_expr", b.getStringAttr("K")),
                  b.getNamedAttr("elem_bytes", b.getI64IntegerAttr(4)),
                  b.getNamedAttr("cleanup", b.getArrayAttr({}))});
        }
        {
          Block *k1Bb = makeHostFor("k", "0", "K", "1");
          enterBlock(k1Bb);
          IRMapping map1;
          Value ld1 =
              makeOp(b, loc, "host_load_scalar", {},
                     {srcBb.getArgument(1).getType()},
                     {b.getNamedAttr("slot", b.getI64IntegerAttr(opSlots[1])),
                      b.getNamedAttr("c_type", b.getStringAttr("float")),
                      b.getNamedAttr("index_expr",
                                     b.getStringAttr(idxExpr(
                                         1, {"batch", "head", "chunk", "k"})))})
                  ->getResult(0);
          Value ld2 =
              makeOp(b, loc, "host_load_scalar", {},
                     {srcBb.getArgument(2).getType()},
                     {b.getNamedAttr("slot", b.getI64IntegerAttr(opSlots[2])),
                      b.getNamedAttr("c_type", b.getStringAttr("float")),
                      b.getNamedAttr("index_expr",
                                     b.getStringAttr(idxExpr(
                                         2, {"batch", "head", "chunk", "k"})))})
                  ->getResult(0);
          map1.map(srcBb.getArgument(1), ld1);
          map1.map(srcBb.getArgument(2), ld2);
          map1.map(srcBb.getArgument(3), ld3);
          for (auto &mop : srcBb.without_terminator()) {
            if (!dependsOnCol.contains(mop.getResult(0)))
              b.clone(mop, map1);
          }
          for (size_t ci = 0; ci < kCut.size(); ++ci) {
            makeOp(b, loc, "host_panel_yield", {map1.lookup(kCut[ci])}, {},
                   {b.getNamedAttr("widen_bf16", b.getBoolAttr(false)),
                    b.getNamedAttr(
                        "dst_expr",
                        b.getStringAttr("wk_" + std::to_string(ci) + "[k]"))});
          }
        }
        enterBlock(gBb);
        Block *k2Bb = makeHostFor("k", "0", "K", "1");
        enterBlock(k2Bb);
        IRMapping map2;
        map2.map(srcBb.getArgument(3), ld3);
        for (size_t ci = 0; ci < kCut.size(); ++ci) {
          Value wkVal =
              makeOp(
                  b, loc, "host_load_scalar", {}, {kCut[ci].getType()},
                  {b.getNamedAttr("slot", b.getI64IntegerAttr(-1 - int64_t(ci))),
                   b.getNamedAttr("c_type", b.getStringAttr("float")),
                   b.getNamedAttr("index_expr", b.getStringAttr("k"))})
                  ->getResult(0);
          map2.map(kCut[ci], wkVal);
        }
        Block *nBb =
            makeHostOmpFor("n", "0", "N", "1", false, false, true, "");
        enterBlock(nBb);
        Value ld0 =
            makeOp(b, loc, "host_load_scalar", {},
                   {srcBb.getArgument(0).getType()},
                   {b.getNamedAttr("slot", b.getI64IntegerAttr(opSlots[0])),
                    b.getNamedAttr("c_type", b.getStringAttr("uint16_t")),
                    b.getNamedAttr("index_expr",
                                   b.getStringAttr(idxExpr(
                                       0, {"batch", "chunk*K+k", "group", "n"})))})
                ->getResult(0);
        map2.map(srcBb.getArgument(0), ld0);
        for (auto &mop : srcBb.without_terminator()) {
          if (dependsOnCol.contains(mop.getResult(0)))
            b.clone(mop, map2);
        }
        makeOp(b, loc, "host_panel_yield", {map2.lookup(term->getOperand(0))},
               {},
               {b.getNamedAttr("widen_bf16",
                               b.getBoolAttr(pk == "map-window-rhs-f32")),
                b.getNamedAttr("dst_expr",
                               b.getStringAttr(dst + "[(g*K+k)*N+n]"))});
      } else {
        IRMapping mapping;
        Block *kBb = makeHostFor("k", "0", "K", "1");
        enterBlock(kBb);
        Value ld1 =
            makeOp(b, loc, "host_load_scalar", {},
                   {srcBb.getArgument(1).getType()},
                   {b.getNamedAttr("slot", b.getI64IntegerAttr(opSlots[1])),
                    b.getNamedAttr("c_type", b.getStringAttr("float")),
                    b.getNamedAttr("index_expr",
                                   b.getStringAttr(idxExpr(
                                       1, {"batch", "head", "chunk", "k"})))})
                ->getResult(0);
        Value ld2 =
            makeOp(b, loc, "host_load_scalar", {},
                   {srcBb.getArgument(2).getType()},
                   {b.getNamedAttr("slot", b.getI64IntegerAttr(opSlots[2])),
                    b.getNamedAttr("c_type", b.getStringAttr("float")),
                    b.getNamedAttr("index_expr",
                                   b.getStringAttr(idxExpr(
                                       2, {"batch", "head", "chunk", "k"})))})
                ->getResult(0);
        mapping.map(srcBb.getArgument(1), ld1);
        mapping.map(srcBb.getArgument(2), ld2);
        mapping.map(srcBb.getArgument(3), ld3);
        for (auto &mop : srcBb.without_terminator()) {
          if (!dependsOnCol.contains(mop.getResult(0)))
            b.clone(mop, mapping);
        }
        Block *nBb =
            makeHostOmpFor("n", "0", "N", "1", false, false, true, "");
        enterBlock(nBb);
        Value ld0 =
            makeOp(b, loc, "host_load_scalar", {},
                   {srcBb.getArgument(0).getType()},
                   {b.getNamedAttr("slot", b.getI64IntegerAttr(opSlots[0])),
                    b.getNamedAttr("c_type", b.getStringAttr("uint16_t")),
                    b.getNamedAttr("index_expr",
                                   b.getStringAttr(idxExpr(
                                       0, {"batch", "chunk*K+k", "group", "n"})))})
                ->getResult(0);
        mapping.map(srcBb.getArgument(0), ld0);
        for (auto &mop : srcBb.without_terminator()) {
          if (dependsOnCol.contains(mop.getResult(0)))
            b.clone(mop, mapping);
        }
        makeOp(b, loc, "host_panel_yield", {mapping.lookup(term->getOperand(0))},
               {},
               {b.getNamedAttr("widen_bf16",
                               b.getBoolAttr(pk == "map-window-rhs-f32")),
                b.getNamedAttr("dst_expr",
                               b.getStringAttr(dst + "[(g*K+k)*N+n]"))});
      }
    } else if (pk == "widen-window-lhs") {
      auto outSh = argShape(slotOf(dstArg));
      int64_t H = std::max<int64_t>(1, outSh[2]);
      int64_t chunks = std::max<int64_t>(1, outSh[1]);
      auto st = argStrides(opSlots[0]);
      emitLet("k_tiles", "(K + 7) / 8");
      Block *pBb = makeHostOmpFor("p_wl", "0", "G*k_tiles", "1", false,
                                  hasNextPrep, false, "");
      enterBlock(pBb);
      emitLet("g", "k_tiles ? p_wl / k_tiles : 0");
      emitLet("kb", "(k_tiles ? p_wl % k_tiles : 0) * 8");
      emitLet("k_end", "kb + 8 < K ? kb + 8 : K");
      emitLet("batch",
              "g/" + std::to_string(H) + "/" + std::to_string(chunks));
      emitLet("chunk",
              "g/" + std::to_string(H) + "%" + std::to_string(chunks));
      emitLet("head", "g%" + std::to_string(H));
      Block *mbBb = makeHostFor("mb", "0", "M", "8");
      enterBlock(mbBb);
      emitLet("m_end", "mb + 8 < M ? mb + 8 : M");
      Block *kBb = makeHostFor("k", "kb", "k_end", "1");
      enterBlock(kBb);
      emitPtr("const uint16_t", "src_row", "a" + std::to_string(opSlots[0]),
              "batch*" + std::to_string(st[0]) + "LL+(chunk*K+k)*" +
                  std::to_string(st[1]) + "LL+head*" + std::to_string(st[2]) +
                  "LL");
      Block *mBb =
          makeHostOmpFor("m", "mb", "m_end", "1", false, false, true, "");
      enterBlock(mBb);
      emitElem("expand_bf", dst + "[(g*M+m)*K+k]",
               "src_row[m*" + std::to_string(st[3]) + "LL]", "", 4);
    } else if (pk == "widen-lhs-f32") {
      auto st = argStrides(opSlots[0]);
      std::string dt = argDtype(opSlots[0]);
      int64_t ak = pop->getAttrOfType<IntegerAttr>("lhs_k").getInt();
      int64_t ab = pop->getAttrOfType<IntegerAttr>("lhs_batch").getInt();
      int64_t am = 0;
      while (am == ak || am == ab)
        ++am;
      std::string banks = ab < 0 ? "1LL" : "G";
      Block *rBb = makeHostOmpFor("row", "0", "(" + banks + ")*M", "1", false,
                                  hasNextPrep, false, "");
      enterBlock(rBb);
      emitLet("g", "M ? row / M : 0");
      emitLet("m", "M ? row % M : 0");
      emitPtr("const uint16_t", "src", "a" + std::to_string(opSlots[0]),
              "g*" + std::to_string(ab < 0 ? 0 : st[ab]) + "LL + m*" +
                  std::to_string(st[am]) + "LL");
      emitPtr("float", "out", dst, "row*K");
      if (st[ak] == 1) {
        emitSpan(dt == "bf16" ? "widen_bf16" : "widen_f16", "out", "src", "K", 4);
      } else {
        Block *kBb = makeHostFor("k", "0", "K", "1");
        enterBlock(kBb);
        emitElem(dt == "bf16" ? "expand_bf" : "expand_f16", "out[k]",
                 "src[k*" + std::to_string(st[ak]) + "LL]", "", 4);
      }
    } else if (pk == "widen-pack-rhs-f32") {
      auto st = argStrides(opSlots[0]);
      std::string dt = argDtype(opSlots[0]);
      int64_t bk = pop->getAttrOfType<IntegerAttr>("rhs_k").getInt();
      int64_t bb = pop->getAttrOfType<IntegerAttr>("rhs_batch").getInt();
      int64_t bn = 0;
      while (bn == bk || bn == bb)
        ++bn;
      std::string banks = bb < 0 ? "1LL" : "G";
      Block *pBb = makeHostOmpFor("p", "0", "(" + banks + ")*nc", "1", false,
                                  false, false, "");
      enterBlock(pBb);
      emitLet("g", "nc ? p / nc : 0");
      emitLet("c", "(nc ? p % nc : 0)*nr");
      emitLet("valid", "N-c < nr ? N-c : nr");
      Block *kBb = makeHostFor("k", "0", "K", "1");
      enterBlock(kBb);
      emitPtr("const uint16_t", "src", "a" + std::to_string(opSlots[0]),
              "g*" + std::to_string(bb < 0 ? 0 : st[bb]) + "LL + k*" +
                  std::to_string(st[bk]) + "LL + c*" + std::to_string(st[bn]) +
                  "LL");
      emitPtr("float", "out", dst, "(p*K + k)*nr");
      if (st[bn] == 1) {
        emitSpan(dt == "bf16" ? "widen_bf16" : "widen_f16", "out", "src",
                 "valid", 4);
      } else {
        Block *jBb = makeHostFor("j", "0", "valid", "1");
        enterBlock(jBb);
        emitElem(dt == "bf16" ? "expand_bf" : "expand_f16", "out[j]",
                 "src[j*" + std::to_string(st[bn]) + "LL]", "", 4);
        enterBlock(kBb);
      }
      Block *zIf = makeHostIf("valid < nr");
      enterBlock(zIf);
      emitSpan("zero", "out + valid", "", "nr - valid", 4);
    } else if (pk == "pack-rhs-panels") {
      auto st = argStrides(opSlots[0]);
      std::string dt = argDtype(opSlots[0]);
      int64_t bytes =
          dt == "f32" || dt == "i32" ? 4 : dt == "i8" || dt == "u8" ? 1 : 2;
      int64_t bk = pop->getAttrOfType<IntegerAttr>("rhs_k").getInt();
      int64_t bb = pop->getAttrOfType<IntegerAttr>("rhs_batch").getInt();
      int64_t bn = 0;
      while (bn == bk || bn == bb)
        ++bn;
      std::string banks = bb < 0 ? "1LL" : "G";
      Block *pBb = makeHostOmpFor("p", "0", "(" + banks + ")*nc", "1", false,
                                  false, false, "");
      enterBlock(pBb);
      emitLet("g", "nc ? p / nc : 0");
      emitLet("c", "(nc ? p % nc : 0)*nr");
      emitLet("valid", "N-c < nr ? N-c : nr");
      Block *kBb = makeHostFor("k", "0", "K", "1");
      enterBlock(kBb);
      emitPtr("unsigned char", "out", dst,
              "(p*K + k)*nr*" + std::to_string(bytes) + "LL");
      emitPtr("const unsigned char", "src", "a" + std::to_string(opSlots[0]),
              "(g*" + std::to_string(bb < 0 ? 0 : st[bb]) + "LL + k*" +
                  std::to_string(st[bk]) + "LL + c*" + std::to_string(st[bn]) +
                  "LL)*" + std::to_string(bytes) + "LL");
      if (st[bn] == 1) {
        emitSpan("memcpy", "out", "src", "valid", bytes);
      } else {
        Block *jBb = makeHostFor("j", "0", "valid", "1");
        enterBlock(jBb);
        emitElem("memcpy", "out+j*" + std::to_string(bytes) + "LL",
                 "src+j*" + std::to_string(st[bn] * bytes) + "LL", "", bytes);
        enterBlock(kBb);
      }
      Block *zIf = makeHostIf("valid < nr");
      enterBlock(zIf);
      emitSpan("zero", "out + valid*" + std::to_string(bytes) + "LL", "",
               "nr - valid", bytes);
    } else if (pk == "pack-signed-dot4") {
      auto st = argStrides(opSlots[0]);
      int64_t bk = pop->getAttrOfType<IntegerAttr>("rhs_k").getInt();
      int64_t bb = pop->getAttrOfType<IntegerAttr>("rhs_batch").getInt();
      int64_t bn = 0;
      while (bn == bk || bn == bb)
        ++bn;
      std::string banks = bb < 0 ? "1LL" : "G";
      Block *pBb = makeHostOmpFor("p", "0", "(" + banks + ")*nc", "1", false,
                                  false, false, "");
      enterBlock(pBb);
      emitLet("g", "nc ? p / nc : 0");
      emitLet("c", "(nc ? p % nc : 0)*nr");
      emitPtr("unsigned char", "out", dst, "p*nr*(K+4)");
      emitPtr("uint32_t", "corr", "out + K*nr", "0");
      emitSpan("zero", "corr", "", "nr", 4);
      Block *kBb = makeHostFor("k", "0", "K", "4");
      enterBlock(kBb);
      emitPtr("const int8_t", "base", "a" + std::to_string(opSlots[0]),
              "g*" + std::to_string(bb < 0 ? 0 : st[bb]) + "LL + c*" +
                  std::to_string(st[bn]) + "LL + k*" + std::to_string(st[bk]) +
                  "LL");
      emitPtr("unsigned char", "dst_k", "out", "k*nr");
      Block *jBb = makeHostOmpFor("j", "0", "nr", "1", false, false, true, "");
      enterBlock(jBb);
      emitElem("dot4_quad", "dst_k", "base",
               std::to_string(st[bn]) + "LL," + std::to_string(st[bk]) + "LL",
               1);
    } else if (pk == "pack-vnni") {
      auto st = argStrides(opSlots[0]);
      int64_t kBlk = pop->getAttrOfType<IntegerAttr>("k_block").getInt();
      int64_t bytes = kBlk == 4 ? 1 : 2;
      std::string cTy = bytes == 1 ? "uint8_t" : "uint16_t";
      std::string kBlkS = std::to_string(kBlk);
      int64_t bk = pop->getAttrOfType<IntegerAttr>("rhs_k").getInt();
      int64_t bb = pop->getAttrOfType<IntegerAttr>("rhs_batch").getInt();
      int64_t bn = 0;
      while (bn == bk || bn == bb)
        ++bn;
      std::string banks = bb < 0 ? "1LL" : "G";
      Block *pBb = makeHostOmpFor("p", "0", "(" + banks + ")*nc", "1", false,
                                  false, false, "");
      enterBlock(pBb);
      emitLet("g", "nc ? p / nc : 0");
      emitLet("c", "(nc ? p % nc : 0)*nr");
      emitLet("valid", "N-c < nr ? N-c : nr");
      emitLet("kPad", "((K+" + std::to_string(kBlk - 1) + ")/" + kBlkS + ")*" +
                          kBlkS);
      emitLet("kFull", "(K/" + kBlkS + ")*" + kBlkS);
      emitPtr(cTy, "out", dst, "p*kPad*nr");
      {
        Block *zIf = makeHostIf("valid < nr || kPad > K");
        enterBlock(zIf);
        emitSpan("zero", "out", "", "kPad*nr", bytes);
      }
      enterBlock(pBb);
      {
        Block *kbBb = makeHostFor("kb", "0", "kFull", kBlkS);
        enterBlock(kbBb);
        emitPtr("const " + cTy, "base", "a" + std::to_string(opSlots[0]),
                "g*" + std::to_string(bb < 0 ? 0 : st[bb]) + "LL + c*" +
                    std::to_string(st[bn]) + "LL + kb*" +
                    std::to_string(st[bk]) + "LL");
        emitPtr(cTy, "dst_k", "out", "(kb/" + kBlkS + ")*nr*" + kBlkS);
        Block *jBb =
            makeHostOmpFor("j", "0", "valid", "1", false, false, true, "");
        enterBlock(jBb);
        Block *lBb = makeHostFor("lane", "0", kBlkS, "1");
        enterBlock(lBb);
        emitElem("assign", "dst_k[j*" + kBlkS + " + lane]",
                 "base[j*" + std::to_string(st[bn]) + "LL + lane*" +
                     std::to_string(st[bk]) + "LL]",
                 "", bytes);
      }
      enterBlock(pBb);
      {
        Block *kbTail = makeHostFor("kb", "kFull", "K", kBlkS);
        enterBlock(kbTail);
        emitPtr("const " + cTy, "base", "a" + std::to_string(opSlots[0]),
                "g*" + std::to_string(bb < 0 ? 0 : st[bb]) + "LL + c*" +
                    std::to_string(st[bn]) + "LL + kb*" +
                    std::to_string(st[bk]) + "LL");
        Block *jBb = makeHostFor("j", "0", "valid", "1");
        enterBlock(jBb);
        Block *lBb = makeHostFor("lane", "0", kBlkS, "1");
        enterBlock(lBb);
        Block *inIf = makeHostIf("kb+lane < K");
        enterBlock(inIf);
        emitElem("assign",
                 "out[(kb/" + kBlkS + ")*nr*" + kBlkS + " + j*" + kBlkS +
                     " + lane]",
                 "base[j*" + std::to_string(st[bn]) + "LL + lane*" +
                     std::to_string(st[bk]) + "LL]",
                 "", bytes);
      }
    } else if (pk == "decode-mx8-lhs") {
      auto stA = argStrides(opSlots[0]);
      auto stS = argStrides(opSlots[1]);
      Block *rBb = makeHostOmpFor("r", "0", "M", "1", false, hasNextPrep,
                                  false, "");
      enterBlock(rBb);
      Block *gBb = makeHostFor("g", "0", "K/32", "1");
      enterBlock(gBb);
      emitLet("sc", "((const uint8_t*)a" + std::to_string(opSlots[1]) + ")[r*" +
                        std::to_string(stS[0]) + "LL + g*" +
                        std::to_string(stS[1]) + "LL]");
      Block *tBb = makeHostFor("t", "0", "32", "1");
      enterBlock(tBb);
      emitElem("decode_mx8", dst + "[r*K + g*32 + t]",
               "((const uint8_t*)a" + std::to_string(opSlots[0]) + ")[r*" +
                   std::to_string(stA[0]) + "LL + g*" + std::to_string(stA[1]) +
                   "LL + t*" + std::to_string(stA[2]) + "LL]",
               "(unsigned)sc", 4);
    } else if (pk == "decode-mx8-rhs") {
      auto stB = argStrides(opSlots[0]);
      auto stS = argStrides(opSlots[1]);
      int64_t colAx =
          pop->getAttrOfType<IntegerAttr>("scale_column_axis").getInt();
      std::string scIdx = colAx == 0
                              ? ("c*" + std::to_string(stS[0]) + "LL + g*" +
                                 std::to_string(stS[1]) + "LL")
                              : ("g*" + std::to_string(stS[0]) + "LL + c*" +
                                 std::to_string(stS[1]) + "LL");
      Block *gBb =
          makeHostOmpFor("g", "0", "K/32", "1", false, false, false, "");
      enterBlock(gBb);
      Block *tBb = makeHostFor("t", "0", "32", "1");
      enterBlock(tBb);
      Block *cBb = makeHostFor("c", "0", "N", "1");
      enterBlock(cBb);
      emitElem("decode_mx8", dst + "[(g*32+t)*N + c]",
               "((const uint8_t*)a" + std::to_string(opSlots[0]) + ")[g*" +
                   std::to_string(stB[0]) + "LL + t*" + std::to_string(stB[1]) +
                   "LL + c*" + std::to_string(stB[2]) + "LL]",
               "((const uint8_t*)a" + std::to_string(opSlots[1]) + ")[" +
                   scIdx + "]",
               4);
    } else if (pk == "encode-q8k") {
      auto st = argStrides(opSlots[0]);
      Block *gBb =
          makeHostOmpFor("g", "0", "K/256", "1", false, false, false, "");
      enterBlock(gBb);
      emitElem("pack_q8k", "&" + dst + "[g]",
               "(const float*)a" + std::to_string(opSlots[0]) + " + g*" +
                   std::to_string(st[0]) + "LL",
               std::to_string(st[1]) + "LL", 1);
    } else if (pk == "gather-routed-lhs" ||
               pk == "gather-widen-routed-lhs-f32") {
      auto stA = argStrides(opSlots[0]);
      auto stO = argStrides(opSlots[1]);
      auto stR = argStrides(opSlots[2]);
      std::string dt = argDtype(opSlots[0]);
      int64_t ak = pop->getAttrOfType<IntegerAttr>("lhs_k").getInt();
      int64_t mult = pop->getAttrOfType<IntegerAttr>("multiplicity").getInt();
      bool wide = pk == "gather-widen-routed-lhs-f32";
      emitLet("active_begin",
              "((const int32_t*)a" + std::to_string(opSlots[1]) + ")[0]");
      emitLet("active_end", "((const int32_t*)a" + std::to_string(opSlots[1]) +
                                ")[G*" + std::to_string(stO[0]) + "LL]");
      Block *rBb = makeHostOmpFor("r", "active_begin", "active_end", "1", false,
                                  false, false, "");
      enterBlock(rBb);
      emitLet("src", "((const int32_t*)a" + std::to_string(opSlots[2]) +
                         ")[r*" + std::to_string(stR[0]) + "LL] / " +
                         std::to_string(mult) + "LL");
      if (wide) {
        emitPtr("const uint16_t", "from", "a" + std::to_string(opSlots[0]),
                "src*" + std::to_string(stA[1 - ak]) + "LL");
        emitPtr("float", "to", dst, "r*K");
        Block *kBb = makeHostFor("k", "0", "K", "1");
        enterBlock(kBb);
        emitElem(dt == "bf16" ? "expand_bf" : "expand_f16", "to[k]",
                 "from[k*" + std::to_string(stA[ak]) + "LL]", "", 4);
      } else {
        int64_t bytes =
            dt == "f32" || dt == "i32" ? 4 : dt == "i8" || dt == "u8" ? 1 : 2;
        emitPtr("const unsigned char", "from", "a" + std::to_string(opSlots[0]),
                "src*" + std::to_string(stA[1 - ak] * bytes) + "LL");
        emitPtr("unsigned char", "to", dst,
                "r*K*" + std::to_string(bytes) + "LL");
        if (stA[ak] == 1) {
          emitSpan("memcpy", "to", "from", "K", bytes);
        } else {
          Block *kBb = makeHostFor("k", "0", "K", "1");
          enterBlock(kBb);
          emitElem("memcpy", "to+k*" + std::to_string(bytes) + "LL",
                   "from+k*" + std::to_string(stA[ak] * bytes) + "LL", "",
                   bytes);
        }
      }
    } else {
      return pop->emitError("unsupported matrix preparation kind ") << pk;
    }
  }

  enterBlock(parBb);
  if (usesBrgemm) {
    makeOp(b, loc, "host_thread_scratch_alloc", {}, {},
           {b.getNamedAttr("stack_bytes", b.getI64IntegerAttr(8192)),
            b.getNamedAttr("alignment", b.getI64IntegerAttr(64)),
            b.getNamedAttr("size_var", b.getStringAttr("brg_scratch_bytes"))});
    makeOp(b, loc, "host_brgemm_ctx_decl");
  }
  Block *ptBb = makeHostOmpFor("job", "0", "jobs", "1", /*parallel=*/false,
                               /*nowait=*/true, /*simd=*/false, "");
  enterBlock(ptBb);
  if (usesBrgemm)
    makeOp(b, loc, "host_guard_continue", {}, {},
           {b.getNamedAttr("cond", b.getStringAttr("!thr_ok"))});
  makeOp(b, loc, "host_tile_coords", {}, {},
         {b.getNamedAttr("windowed", b.getBoolAttr(windowed)),
          b.getNamedAttr("head_extent", b.getI64IntegerAttr(headExt)),
          b.getNamedAttr("chunk_extent", b.getI64IntegerAttr(chunkExt))});
  if (grouped) {
    makeOp(b, loc, "host_grouped_bounds", {}, {},
           {b.getNamedAttr("offsets_slot", b.getI64IntegerAttr(offsetsSlot)),
            b.getNamedAttr("offsets_stride",
                           b.getI64IntegerAttr(offsetsStride))});
    makeOp(b, loc, "host_guard_continue", {}, {},
           {b.getNamedAttr("cond", b.getStringAttr("r >= end"))});
  }
  makeOp(b, loc, "host_route_coords", {}, {},
         {b.getNamedAttr("routed", b.getBoolAttr(routed)),
          b.getNamedAttr("routes_slot", b.getI64IntegerAttr(routesSlot)),
          b.getNamedAttr("routes_stride", b.getI64IntegerAttr(routesStride)),
          b.getNamedAttr("multiplicity", b.getI64IntegerAttr(multiplicity))});
  if (usesBrgemm)
    makeOp(b, loc, "host_brgemm_ctx_switch");

  for (auto [i, v] : llvm::enumerate(views)) {
    int64_t argElemBytes =
        cast<MemRefType>(fn.getArgument(v.slot).getType()).getElementTypeBitWidth() /
        8;
    std::string sk = "arg", expr;
    if (routed && i == 0) {
      int64_t mBytes = packing == "f32-panels-per-call" ? 4 : argElemBytes;
      sk = "raw";
      expr = "(char*)members+r*" + std::to_string(K * mBytes) + "LL";
    } else if (routed && v.writes) {
      sk = "routed_out";
      expr = std::to_string(mr * nr * argElemBytes);
    } else if (i == 0 && (packing == "mapped-f32-panels" ||
                          packing == "decoded-f32-per-call")) {
      sk = "raw";
      expr = "left+(" + v.hostOrigin + ")";
    } else if (i == 0 && packing == "f32-panels-per-call") {
      int64_t ab = dot.getAttrOfType<IntegerAttr>("lhs_batch").getInt();
      sk = "raw";
      expr = std::string("left+(") + (ab < 0 ? "0LL" : "g*M") + "+src)*K";
    } else if (i == 1 && (packing == "mapped-right-per-call" ||
                          packing == "mapped-f32-panels" ||
                          packing == "decoded-f32-per-call")) {
      sk = "raw";
      expr = "rhs+(" + v.hostOrigin + ")";
    } else if (i == 1 && packing == "shared-q8k-per-call") {
      sk = "raw";
      expr = "packed";
    } else if (i == 1 && packing == "signed-dot4-panels") {
      int64_t bb = dot.getAttrOfType<IntegerAttr>("rhs_batch").getInt();
      sk = "raw";
      expr = std::string("rhs+(") + (bb < 0 ? "0LL" : "g*nc") +
             "+c/nr)*(nr*(K+4))";
    } else if (i == 1 && (packing == "rhs-panels-per-call" ||
                          packing == "f32-panels-per-call" ||
                          packing == "vnni-pack32-panels")) {
      int64_t bb = dot.getAttrOfType<IntegerAttr>("rhs_batch").getInt();
      int64_t pBytes = packing == "f32-panels-per-call" ? 4 : argElemBytes;
      std::string kExpr = packing == "vnni-pack32-panels"
                              ? ("((K+" + std::to_string(argElemBytes == 1 ? 3 : 1) +
                                 ")/" + std::to_string(argElemBytes == 1 ? 4 : 2) +
                                 "*" + std::to_string(argElemBytes == 1 ? 4 : 2) +
                                 ")")
                              : "K";
      sk = "raw";
      expr = std::string("(char*)rhs+((") + (bb < 0 ? "0LL" : "g*nc") +
             "+c/nr)*" + kExpr + "*nr)*" + std::to_string(pBytes) + "LL";
    } else {
      expr = "(char*)a" + std::to_string(v.slot) + "+(" + v.hostOrigin + ")*" +
             std::to_string(argElemBytes) + "LL";
    }
    makeOp(b, loc, "host_view_ptr", {}, {},
           {b.getNamedAttr("index", b.getI64IntegerAttr(i)),
            b.getNamedAttr("source_kind", b.getStringAttr(sk)),
            b.getNamedAttr("expr", b.getStringAttr(expr))});
  }

  makeOp(b, loc, "host_call_panel", {}, {},
         {b.getNamedAttr("arity", b.getI64IntegerAttr(views.size())),
          b.getNamedAttr("uses_brgemm", b.getBoolAttr(usesBrgemm))});

  if (routed) {
    int64_t dstBytes =
        cast<MemRefType>(dstArg.getType()).getElementTypeBitWidth() / 8;
    Block *sBb = makeHostFor("i", "0", "R", "1");
    enterBlock(sBb);
    emitLet("d_i", "((const int32_t*)a" + std::to_string(routesSlot) +
                       ")[(r+i)*" + std::to_string(routesStride) + "LL]");
    emitPtr("unsigned char", "out", "a" + std::to_string(slotOf(dstArg)),
            "((d_i/" + std::to_string(multiplicity) + "LL)*" +
                std::to_string(origDstStrides[0]) + "LL + (d_i%" +
                std::to_string(multiplicity) + "LL)*" +
                std::to_string(origDstStrides[1]) + "LL + c*" +
                std::to_string(origDstStrides[2]) + "LL)*" +
                std::to_string(dstBytes) + "LL");
    emitPtr("const unsigned char", "from", "routed_out",
            "i*nr*" + std::to_string(dstBytes) + "LL");
    if (origDstStrides[2] == 1) {
      emitSpan("memcpy", "out", "from", "C", dstBytes);
    } else {
      Block *jBb = makeHostFor("j", "0", "C", "1");
      enterBlock(jBb);
      emitElem("memcpy", "out+j*" + std::to_string(origDstStrides[2] * dstBytes) + "LL",
               "from+j*" + std::to_string(dstBytes) + "LL", "", dstBytes);
    }
  }

  if (usesBrgemm) {
    enterBlock(parBb);
    makeOp(b, loc, "host_brgemm_ctx_release");
    makeOp(b, loc, "host_thread_scratch_free", {}, {},
           {b.getNamedAttr("stack_bytes", b.getI64IntegerAttr(8192))});
  }

  enterBlock(&hBb);
  for (Operation *rop : releaseOps) {
    makeOp(b, loc, "host_free", {}, {},
           {b.getNamedAttr("buf_name",
                           b.getStringAttr(bufNames[rop->getOperand(0)]))});
  }

  // Erase old cpu@buffers ops (keep prog and func.return)
  SmallVector<Operation *> toErase;
  for (auto &op : fn.front()) {
    if (&op != prog && op.getName().getStringRef() != "func.return")
      toErase.push_back(&op);
  }
  for (Operation *op : llvm::reverse(toErase))
    op->erase();

  return success();
}

bool hasMojoMatrixProgram(ModuleOp m) {
  auto funcs = m.getOps<func::FuncOp>();
  if (!llvm::hasSingleElement(funcs))
    return false;
  auto fn = *funcs.begin();
  return !fn.empty() && !fn.front().empty() &&
         fn.front().front().getName().getStringRef() ==
             "mojo_row.matrix_program";
}

LogicalResult verifyMojoMatrix(ModuleOp m, const TargetProfile &p,
                               bool callable) {
  auto fn = *m.getOps<func::FuncOp>().begin();
  auto &prog = fn.front().front();
  if (prog.getNumRegions() != 2 || prog.getRegion(0).empty() ||
      prog.getRegion(1).empty())
    return prog.emitError("incomplete mojo_row.matrix_program regions");

  bool usesBrgemm = prog.getAttrOfType<BoolAttr>("uses_brgemm").getValue();
  bool sawBrgemmCall = false;

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
                 "32-bit matrix address lacks a proven signed i32 range");
             return WalkResult::interrupt();
           }
         } else if (name == "mojo_row.brgemm_call") {
           sawBrgemmCall = true;
           auto ftz = op->getAttrOfType<BoolAttr>("allow_subnormal_ftz");
           auto impl = op->getAttrOfType<StringAttr>("ci.implementation");
           if (!ftz || !impl) {
             op->emitError(
                 "brgemm_call requires allow_subnormal_ftz and ci.implementation");
             return WalkResult::interrupt();
           }
           bool isBf16 = impl.getValue() == "exits/mojo::onednn-brgemm-bf16";
           if (ftz.getValue() != isBf16) {
             op->emitError(
                 "brgemm_call allow_subnormal_ftz disagrees with implementation contract");
             return WalkResult::interrupt();
           }
           Operation *next = op->getNextNode();
           if (!next || next->getName().getStringRef() != "mojo_row.check_status" ||
               next->getOperand(0) != op->getResult(0)) {
             op->emitError(
                 "brgemm_call status must be immediately checked by mojo_row.check_status");
             return WalkResult::interrupt();
           }
         }
         return WalkResult::advance();
       }).wasInterrupted();
  if (badWalk)
    return failure();

  if (usesBrgemm != sawBrgemmCall)
    return prog.emitError("matrix_program uses_brgemm disagrees with panel body");

  if (usesBrgemm) {
    bool sawScratchAlloc = false, sawScratchFree = false, sawNowait = false,
         sawCtxSwitch = false, sawCtxRelease = false;
    prog.getRegion(1).walk([&](Operation *op) {
      auto n = op->getName().getStringRef();
      if (n == "mojo_row.host_thread_scratch_alloc") {
        auto sb = op->getAttrOfType<IntegerAttr>("stack_bytes");
        auto al = op->getAttrOfType<IntegerAttr>("alignment");
        sawScratchAlloc = sb && sb.getInt() > 0 && al && al.getInt() == 64;
      } else if (n == "mojo_row.host_thread_scratch_free") {
        sawScratchFree = true;
      } else if (n == "mojo_row.host_omp_for" &&
                 op->getAttrOfType<BoolAttr>("nowait").getValue()) {
        sawNowait = true;
      } else if (n == "mojo_row.host_brgemm_ctx_switch") {
        sawCtxSwitch = true;
      } else if (n == "mojo_row.host_brgemm_ctx_release") {
        sawCtxRelease = true;
      }
    });
    if (!sawScratchAlloc || !sawScratchFree || !sawNowait || !sawCtxSwitch ||
        !sawCtxRelease)
      return prog.emitError(
          "BRGeMM host region lacks explicit scratch, nowait, or context IR");
  }

  if (failed(mojoMatrixPrinter().verify(m, p)))
    return failure();

  if (callable) {
    if (fn->getAttrOfType<StringAttr>("mojo_row.entry") != "intent_entry")
      return fn.emitError("missing synchronous Mojo host entry");
    llvm::SmallVector<char> expected;
    llvm::raw_svector_ostream os(expected);
    if (auto err = describeMojoMatrixPackage(fn, prog, os))
      return fn.emitError() << llvm::toString(std::move(err)), failure();
    auto pkg = m->getAttrOfType<StringAttr>("mojo_row.package");
    if (!pkg || pkg.getValue() != StringRef(expected.data(), expected.size()))
      return fn.emitError("mojo_row.package disagrees with verified matrix ABI"),
             failure();
  }
  return success();
}
} // namespace CI_UNIT_NAMESPACE
