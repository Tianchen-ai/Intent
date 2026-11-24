#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "intentdsl/levels/algorithm/kir/ir/Storage.h"
#include "intentdsl/levels/execution/cpu/ir/MatrixTask.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"

namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
using namespace mlir;

bool canNarrowWeftAddress(Operation &op);
TerminalPrinter weftMatrixPrinter();

namespace {
std::string listI64(ArrayRef<int64_t> values) {
  SmallVector<std::string> parts;
  for (int64_t v : values)
    parts.push_back(std::to_string(v));
  return "[" + llvm::join(parts, ",") + "]";
}

struct WeftTypeSpec {
  std::string carrier;
  SmallVector<int64_t> dims, axes;

  std::string element() const {
    return llvm::StringSwitch<std::string>(carrier)
        .Case("i32", "si32")
        .Case("i16", "si16")
        .Case("i8", "si8")
        .Case("u8", "ui8")
        .Case("f8e4m3fn", "f8E4M3FN")
        .Default(carrier);
  }

  std::string format(StringRef kind) const {
    std::string e = carrier == "f8e4m3fn" ? std::string("f8E4M3FN") : carrier;
    if (kind == "value" && dims.empty())
      return element();
    bool record = e == "Q4_K" || e == "Q8_K";
    std::string enc =
        record ? ("!weft_kernel.encoding<\"" + e + "\",\"base\",\"" + e +
                  "\",[]>")
        : kind == "value"
            ? element()
            : ("!weft_kernel.encoding<\"" + e + "\",\"dense\",\"dense." + e +
               "\",[]>");
    return "!weft_kernel." + kind.str() + "<" + enc + "," + listI64(dims) +
           "," + listI64(axes) + ">";
  }
};

std::string cElementType(StringRef carrier) {
  return llvm::StringSwitch<std::string>(carrier)
      .Case("f16", "_Float16")
      .Case("f32", "float")
      .Case("i8", "int8_t")
      .Case("u8", "uint8_t")
      .Case("Q4_K", "uint8_t")
      .Case("Q8_K", "uint8_t")
      .Case("i16", "int16_t")
      .Case("f8e4m3fn", "uint8_t")
      .Case("bf16", "uint16_t")
      .Case("i32", "int32_t")
      .Case("i64", "int64_t")
      .Case("u64", "uint64_t")
      .Default("void");
}

std::string elemCarrierName(Type t) {
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

enum class CoordTag {
  Zero,
  G,
  C,
  CTile,
  R,
  Src,
  Dest,
  DestDivMult,
  DestModMult,
  Batch,
  Chunk,
  ChunkTimesK,
  Head,
};

struct AddrTerm {
  CoordTag coord;
  int64_t stride;
};

struct SlotSpec {
  WeftTypeSpec t;
  std::string baseExpr;
  int64_t byteOffset = 0;
  SmallVector<AddrTerm> addrTerms;
  SmallVector<int64_t> panelStrides;
  bool writes = false;
  bool isRoutedTempOut = false;
  int64_t routedTempBytes = 0;
};

struct EpilogueItem {
  enum Kind { Bias, Convert } kind;
  unsigned biasSlot = 0;
  std::string targetCarrier;
};

bool nativeQ4KStrides(func::FuncOp fn, Operation &dot) {
  auto slot = cast<BlockArgument>(dot.getOperand(0)).getArgNumber();
  auto s = fn.getArgAttrOfType<DenseI64ArrayAttr>(slot, "cpu.strides");
  auto shape = fn.getArgAttrOfType<DenseI64ArrayAttr>(slot, "cpu.shape");
  return s && shape && s.size() == 3 && shape.size() == 3 && s[2] == 1 &&
         s[1] == 144 && s[0] == 144 * shape[1];
}

Operation *makeOp(OpBuilder &b, Location loc, StringRef name,
                  ValueRange operands = {}, TypeRange types = {},
                  ArrayRef<NamedAttribute> attrs = {}) {
  OperationState s(loc, ("weft_host." + name).str());
  s.addOperands(operands);
  s.addTypes(types);
  s.addAttributes(attrs);
  return b.create(s);
}

struct TaskTerm {
  Value v;
  std::string type;
};

struct TaskBuilder {
  OpBuilder &b;
  Location loc;

  TaskTerm idx(int64_t n) {
    Value v = makeOp(b, loc, "task_const", {}, {b.getIndexType()},
                     {b.getNamedAttr("lit", b.getStringAttr(std::to_string(n))),
                      b.getNamedAttr("val_type", b.getStringAttr("index"))})
                  ->getResult(0);
    return {v, "index"};
  }

  TaskTerm lit(StringRef n, StringRef ty) {
    Value v = makeOp(b, loc, "task_const", {}, {b.getIndexType()},
                     {b.getNamedAttr("lit", b.getStringAttr(n)),
                      b.getNamedAttr("val_type", b.getStringAttr(ty))})
                  ->getResult(0);
    return {v, ty.str()};
  }

  TaskTerm newVal(TaskTerm scalar, StringRef valType) {
    Value v =
        makeOp(b, loc, "task_new", {scalar.v}, {b.getIndexType()},
               {b.getNamedAttr("in_type", b.getStringAttr(scalar.type)),
                b.getNamedAttr("val_type", b.getStringAttr(valType))})
            ->getResult(0);
    return {v, valType.str()};
  }

  TaskTerm bin(StringRef kind, TaskTerm lhs, TaskTerm rhs, StringRef resType) {
    Value v =
        makeOp(b, loc, "task_binary", {lhs.v, rhs.v}, {b.getIndexType()},
               {b.getNamedAttr("kind", b.getStringAttr(kind)),
                b.getNamedAttr("lhs_type", b.getStringAttr(lhs.type)),
                b.getNamedAttr("rhs_type", b.getStringAttr(rhs.type)),
                b.getNamedAttr("val_type", b.getStringAttr(resType))})
            ->getResult(0);
    return {v, resType.str()};
  }

  TaskTerm cmp(StringRef pred, TaskTerm lhs, TaskTerm rhs, StringRef resType) {
    Value v =
        makeOp(b, loc, "task_compare", {lhs.v, rhs.v}, {b.getIndexType()},
               {b.getNamedAttr("predicate", b.getStringAttr(pred)),
                b.getNamedAttr("lhs_type", b.getStringAttr(lhs.type)),
                b.getNamedAttr("rhs_type", b.getStringAttr(rhs.type)),
                b.getNamedAttr("val_type", b.getStringAttr(resType))})
            ->getResult(0);
    return {v, resType.str()};
  }

  TaskTerm sel(TaskTerm cond, TaskTerm tVal, TaskTerm fVal) {
    Value v =
        makeOp(b, loc, "task_select", {cond.v, tVal.v, fVal.v},
               {b.getIndexType()},
               {b.getNamedAttr("cond_type", b.getStringAttr(cond.type)),
                b.getNamedAttr("val_type", b.getStringAttr(tVal.type))})
            ->getResult(0);
    return {v, tVal.type};
  }

  TaskTerm castOp(StringRef opKind, TaskTerm in, StringRef dstType) {
    if (opKind == "cast" && in.type == dstType)
      return in;
    Value v = makeOp(b, loc, "task_cast", {in.v}, {b.getIndexType()},
                     {b.getNamedAttr("op_kind", b.getStringAttr(opKind)),
                      b.getNamedAttr("in_type", b.getStringAttr(in.type)),
                      b.getNamedAttr("val_type", b.getStringAttr(dstType))})
                  ->getResult(0);
    return {v, dstType.str()};
  }

  TaskTerm sliceSlot(Value viewArg, const WeftTypeSpec &spec,
                     ArrayRef<std::optional<Value>> coords, bool admit) {
    WeftTypeSpec sub{spec.carrier, {}, {}};
    SmallVector<Attribute> selAttrs;
    SmallVector<Value> args{viewArg};
    for (auto [dimIdx, c] : llvm::enumerate(coords)) {
      if (!c.has_value()) {
        selAttrs.push_back(b.getStringAttr("all"));
        sub.dims.push_back(spec.dims[dimIdx]);
        sub.axes.push_back(spec.axes[dimIdx]);
      } else {
        selAttrs.push_back(b.getStringAttr("index"));
        args.push_back(*c);
      }
    }
    std::string slTy = sub.format("slice");
    std::string vlTy = sub.format("value");
    Value v =
        makeOp(b, loc, "task_slice", args, {b.getIndexType()},
               {b.getNamedAttr("view_type", b.getStringAttr(spec.format("view"))),
                b.getNamedAttr("selectors", b.getArrayAttr(selAttrs)),
                b.getNamedAttr("slice_type", b.getStringAttr(slTy)),
                b.getNamedAttr("val_type", b.getStringAttr(vlTy)),
                b.getNamedAttr("admit", b.getBoolAttr(admit))})
            ->getResult(0);
    return {v, admit ? vlTy : slTy};
  }

  void commit(TaskTerm val, TaskTerm slice) {
    makeOp(b, loc, "task_commit", {val.v, slice.v}, {},
           {b.getNamedAttr("val_type", b.getStringAttr(val.type)),
            b.getNamedAttr("slice_type", b.getStringAttr(slice.type))});
  }
};
} // namespace

llvm::Error describeWeftMatrixPackage(func::FuncOp fn, Operation &prog,
                                      llvm::raw_pwrite_stream &out) {
  auto abi = llvm::json::parse(fn->getAttrOfType<StringAttr>("kir.abi"));
  if (!abi)
    return abi.takeError();
  auto tile = prog.getAttrOfType<DenseI64ArrayAttr>("tile");
  auto slotsAttr = prog.getAttrOfType<ArrayAttr>("task_slots_json");
  auto ranksAttr = prog.getAttrOfType<DenseI64ArrayAttr>("task_ranks");
  auto contigAttr = prog.getAttrOfType<DenseI64ArrayAttr>("contiguous_slots");

  llvm::json::Array slots, ranks, compact;
  for (Attribute a : slotsAttr) {
    auto d = cast<DictionaryAttr>(a);
    slots.push_back(llvm::json::Object{
        {"dtype", d.getAs<StringAttr>("dtype").getValue().str()},
        {"direction", d.getAs<StringAttr>("direction").getValue().str()},
        {"encoding", d.getAs<StringAttr>("encoding").getValue().str()}});
  }
  for (int64_t r : ranksAttr.asArrayRef())
    ranks.push_back(r);
  for (int64_t c : contigAttr.asArrayRef())
    compact.push_back(c);

  llvm::json::Object pkg{
      {"version", 3},
      {"task", ""},
      {"host", ""},
      {"native_symbol", "matrix_panel_instance"},
      {"task_slots", std::move(slots)},
      {"task_ranks", std::move(ranks)},
      {"contiguous_slots", std::move(compact)},
      {"shape_parameters", llvm::json::Array{"R", "C"}},
      {"domain", intentdsl::publishDomain(fn)},
      {"abi", std::move(*abi)},
      {"rows", llvm::json::Array{llvm::json::Object{{"grain", tile[0]},
                                                    {"columns", tile[1]}}}}};
  out << llvm::json::Value(std::move(pkg));
  return llvm::Error::success();
}

LogicalResult lowerWeftMatrix(func::FuncOp fn, bool boundWidening) {
  auto *task = intentdsl::cpu::findMatrixTask(fn);
  if (!task)
    return failure();
  auto &domain = task->getRegion(0).front().front();
  auto &dot = intentdsl::cpu::calculation(domain);
  StringRef kind = dot.getName().stripDialect();
  auto extents = domain.getAttrOfType<DenseI64ArrayAttr>("cpu.extents");
  auto tile = domain.getAttrOfType<DenseI64ArrayAttr>("cpu.output_tile");
  auto implAttr = domain.getAttrOfType<StringAttr>("cpu.implementation");
  if (!implAttr)
    return domain.emitError("missing cpu.implementation at weft@realized");
  StringRef implId = implAttr.getValue();
  StringRef shortImpl = implId.rsplit("::").second;
  if (shortImpl.empty())
    shortImpl = implId;

  auto packingAttr = domain.getAttrOfType<StringAttr>("cpu.packing");
  StringRef packing = packingAttr ? packingAttr.getValue() : StringRef();
  bool routed = domain.hasAttr("cpu.route_storage");
  bool usesBrgemm = shortImpl.starts_with("onednn-brgemm");
  bool allowSubnormalFtz = dot.hasAttr("kir.allow_subnormal_ftz") ||
                           domain.hasAttr("kir.allow_subnormal_ftz");

  int64_t M = extents[0], N = extents[1], K = extents[2], G = extents[3];
  int64_t mr = tile[0], nr = tile[1];
  bool emptyRed = (K == 0);

  auto abiJson =
      llvm::cantFail(llvm::json::parse(fn->getAttrOfType<StringAttr>("kir.abi")));
  auto slotOf = [](Value v) { return cast<BlockArgument>(v).getArgNumber(); };
  auto argDtype = [&](unsigned s) {
    return (*abiJson.getAsArray())[s].getAsObject()->getString("dtype")->str();
  };
  auto argStrides = [&](unsigned s) {
    return fn.getArgAttrOfType<DenseI64ArrayAttr>(s, "cpu.strides");
  };
  auto argShape = [&](unsigned s) {
    return fn.getArgAttrOfType<DenseI64ArrayAttr>(s, "cpu.shape");
  };

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

  Value dstArg = domain.getOperand(0);
  int dstRank = cast<MemRefType>(dstArg.getType()).getRank();
  int64_t offsetsSlot = -1, offsetsStride = 0;
  int64_t routesSlot = -1, routesStride = 0, multiplicity = 1;
  bool grouped = false;
  if (kind == "contract" && dot.getNumOperands() > 2) {
    grouped = true;
    offsetsSlot = slotOf(dot.getOperand(2));
    offsetsStride = argStrides(offsetsSlot)[0];
  }
  if (kind == "contract" && dot.getNumOperands() > 3) {
    routesSlot = slotOf(dot.getOperand(3));
    routesStride = argStrides(routesSlot)[0];
    multiplicity = argShape(slotOf(dstArg))[1];
  }

  int64_t cols = N <= nr ? N : (N % nr == 0 ? nr : -2);
  bool isPack32 = (packing == "vnni-pack32-panels");
  if (isPack32)
    cols = -2;

  bool isQ4k = (kind == "q4k_dot");
  bool useEncodedRecords = !emptyRed && isQ4k &&
                           shortImpl == "weft-q4k-dot" &&
                           nativeQ4KStrides(fn, dot);

  SmallVector<SlotSpec, 8> slots;
  auto addArgSlot = [&](Value v, StringRef overrideCarrier,
                        SmallVector<int64_t> dims, SmallVector<int64_t> axes,
                        ArrayRef<CoordTag> coordTags, ArrayRef<int> panelAxes,
                        bool isWrite = false) {
    unsigned s = slotOf(v);
    auto st = argStrides(s);
    SmallVector<AddrTerm> terms;
    for (auto [ax, tag] : llvm::enumerate(coordTags))
      if (tag != CoordTag::Zero)
        terms.push_back({tag, st[ax]});
    SmallVector<int64_t> pSt;
    for (int ax : panelAxes)
      pSt.push_back(st[ax]);
    std::string carrier =
        overrideCarrier.empty() ? argDtype(s) : overrideCarrier.str();
    slots.push_back({{carrier, std::move(dims), std::move(axes)},
                     "a" + std::to_string(s),
                     0,
                     std::move(terms),
                     std::move(pSt),
                     isWrite,
                     false,
                     0});
  };

  if (!emptyRed) {
    if (isQ4k) {
      int64_t ng = K / 256;
      unsigned wSlot = slotOf(dot.getOperand(0));
      auto wSt = argStrides(wSlot);
      if (useEncodedRecords) {
        slots.push_back({{"Q4_K", {-1, ng, 256}, {1, 4, 3}},
                         "a" + std::to_string(wSlot),
                         0,
                         {{CoordTag::Src, wSt[0]}},
                         {},
                         false});
        slots.push_back(
            {{"Q8_K", {ng, 256}, {4, 3}}, "packed", 0, {}, {}, false});
      } else {
        slots.push_back({{"u8", {-1, ng, 16}, {1, 4, 5}},
                         "a" + std::to_string(wSlot),
                         0,
                         {{CoordTag::Src, wSt[0]}},
                         {wSt[0], wSt[1], wSt[2]},
                         false});
        slots.push_back({{"u8", {-1, ng, 4, 32}, {1, 4, 6, 3}},
                         "a" + std::to_string(wSlot),
                         16 * wSt[2],
                         {{CoordTag::Src, wSt[0]}},
                         {wSt[0], wSt[1], 32 * wSt[2], wSt[2]},
                         false});
        slots.push_back({{"f32", {ng}, {4}}, "packed", 0, {}, {73}, false});
        slots.push_back(
            {{"i8", {ng, 8, 32}, {4, 7, 3}}, "packed", 4, {}, {292, 32, 1}, false});
        slots.push_back(
            {{"i16", {ng, 16}, {4, 8}}, "packed", 260, {}, {146, 1}, false});
      }
    } else if (kind == "contract") {
      int ak = dot.getAttrOfType<IntegerAttr>("lhs_k").getInt();
      int bk = dot.getAttrOfType<IntegerAttr>("rhs_k").getInt();
      int ab = dot.getAttrOfType<IntegerAttr>("lhs_batch").getInt();
      int bb = dot.getAttrOfType<IntegerAttr>("rhs_batch").getInt();

      auto buildContractInput = [&](Value v, int kAx, int bAx, bool isLhs) {
        int rank = cast<MemRefType>(v.getType()).getRank();
        SmallVector<CoordTag> tags(rank, CoordTag::Zero);
        int freeAx = 0;
        while (freeAx == kAx || freeAx == bAx)
          ++freeAx;
        tags[freeAx] = isLhs ? CoordTag::Src : CoordTag::C;
        if (bAx >= 0)
          tags[bAx] = CoordTag::G;
        addArgSlot(v, "",
                   isLhs ? SmallVector<int64_t>{-1, K}
                         : SmallVector<int64_t>{K, cols},
                   isLhs ? SmallVector<int64_t>{1, 3}
                         : SmallVector<int64_t>{3, 2},
                   tags,
                   isLhs ? ArrayRef<int>{freeAx, kAx}
                         : ArrayRef<int>{kAx, freeAx});
      };
      buildContractInput(dot.getOperand(0), ak, ab, true);
      buildContractInput(dot.getOperand(1), bk, bb, false);

      int64_t ncCount = nr > 0 ? (N + nr - 1) / nr : 0;
      if (routed) {
        slots[0].baseExpr = "members";
        slots[0].addrTerms = {{CoordTag::R, K}};
        slots[0].panelStrides = {K, 1};
        if (packing == "f32-panels-per-call")
          slots[0].t.carrier = "f32";
      } else if (packing == "f32-panels-per-call") {
        slots[0].t.carrier = "f32";
        slots[0].baseExpr = "left";
        slots[0].addrTerms = ab < 0
                                 ? SmallVector<AddrTerm>{{CoordTag::Src, K}}
                                 : SmallVector<AddrTerm>{{CoordTag::G, M * K},
                                                         {CoordTag::Src, K}};
        slots[0].panelStrides = {K, 1};
      }

      if (packing == "rhs-panels-per-call" || packing == "f32-panels-per-call") {
        if (packing == "f32-panels-per-call")
          slots[1].t.carrier = "f32";
        slots[1].baseExpr = "rhs";
        slots[1].addrTerms.clear();
        if (bb >= 0)
          slots[1].addrTerms.push_back({CoordTag::G, ncCount * K * nr});
        slots[1].addrTerms.push_back({CoordTag::CTile, K * nr});
        slots[1].panelStrides = {nr, 1};
      } else if (isPack32) {
        int64_t kBlk = slots[0].t.carrier == "i8" ? 4 : 2;
        int64_t kb = K / kBlk;
        int64_t kPad = kb * kBlk;
        int64_t lhsRowSt = slots[0].panelStrides[0];
        slots[0].t.dims = {-1, kb, kBlk};
        slots[0].t.axes = {1, 3, 4};
        slots[0].panelStrides = {lhsRowSt, kBlk, 1};

        slots[1].t.dims = {kb, -2, kBlk};
        slots[1].t.axes = {3, 2, 4};
        slots[1].baseExpr = "rhs";
        slots[1].addrTerms.clear();
        if (bb >= 0)
          slots[1].addrTerms.push_back({CoordTag::G, ncCount * kPad * nr});
        slots[1].addrTerms.push_back({CoordTag::CTile, kPad * nr});
        slots[1].panelStrides = {nr * kBlk, kBlk, 1};
      }
    } else if (kind == "window_contract") {
      addArgSlot(dot.getOperand(0), "", {-1, K}, {1, 3},
                 {CoordTag::Batch, CoordTag::ChunkTimesK, CoordTag::Head,
                  CoordTag::Src},
                 {3, 1});
      slots.push_back({{"bf16", {K, cols}, {3, 2}},
                       "rhs",
                       0,
                       {{CoordTag::G, K * N}, {CoordTag::C, 1}},
                       {N, 1},
                       false});
      if (packing == "mapped-f32-panels") {
        slots[0].t.carrier = slots[1].t.carrier = "f32";
        slots[0].baseExpr = "left";
        slots[0].addrTerms = {{CoordTag::G, M * K}, {CoordTag::Src, K}};
        slots[0].panelStrides = {K, 1};
      }
    } else if (kind == "scaled_contract") {
      slots.push_back({{"f32", {-1, K}, {1, 3}},
                       "left",
                       0,
                       {{CoordTag::Src, K}},
                       {K, 1},
                       false});
      slots.push_back({{"f32", {K, cols}, {3, 2}},
                       "rhs",
                       0,
                       {{CoordTag::C, 1}},
                       {N, 1},
                       false});
    }
  }

  SmallVector<EpilogueItem> epilogue;
  for (auto &op : domain.getRegion(0).front()) {
    auto n = op.getName().stripDialect();
    if (n == "add_columns") {
      unsigned bIdx = slots.size();
      if (dstRank == 1)
        addArgSlot(op.getOperand(1), "", {-1}, {1}, {CoordTag::Dest}, {0});
      else
        addArgSlot(op.getOperand(1), "", {cols}, {2}, {CoordTag::C}, {0});
      epilogue.push_back({EpilogueItem::Bias, bIdx, ""});
    } else if (n == "convert_matrix") {
      Type toT = cast<ShapedType>(op.getResult(0).getType()).getElementType();
      epilogue.push_back({EpilogueItem::Convert, 0, elemCarrierName(toT)});
    }
  }

  if (kind == "window_contract") {
    addArgSlot(dstArg, "", {-1, cols}, {1, 2},
               {CoordTag::Batch, CoordTag::Chunk, CoordTag::Head,
                CoordTag::Dest, CoordTag::C},
               {3, 4}, true);
  } else if (routed) {
    int64_t dstBytes =
        cast<MemRefType>(dstArg.getType()).getElementTypeBitWidth() / 8;
    slots.push_back({{argDtype(slotOf(dstArg)), {-1, cols}, {1, 2}},
                     "routed_out",
                     0,
                     {},
                     {nr, 1},
                     true,
                     true,
                     mr * nr * dstBytes});
  } else if (dstRank == 1) {
    addArgSlot(dstArg, "", {-1}, {1}, {CoordTag::Dest}, {0}, true);
  } else if (dstRank == 2) {
    addArgSlot(dstArg, "", {-1, cols}, {1, 2}, {CoordTag::Dest, CoordTag::C},
               {0, 1}, true);
  } else {
    addArgSlot(dstArg, "", {-1, cols}, {1, 2},
               {CoordTag::G, CoordTag::Dest, CoordTag::C}, {1, 2}, true);
  }

  SmallVector<int64_t> origDstStrides(
      argStrides(slotOf(dstArg)).asArrayRef().begin(),
      argStrides(slotOf(dstArg)).asArrayRef().end());

  Location loc = fn.getLoc();
  OpBuilder b(&fn.front().front());

  SmallVector<Attribute> slotDicts, viewTypes, argWrites;
  SmallVector<int64_t> taskRanks, contigSlots, stridedSlots, flatStrides;
  for (auto [i, sl] : llvm::enumerate(slots)) {
    bool rec = sl.t.carrier == "Q4_K" || sl.t.carrier == "Q8_K";
    slotDicts.push_back(b.getDictionaryAttr(
        {b.getNamedAttr("dtype", b.getStringAttr(rec ? "u8" : sl.t.carrier)),
         b.getNamedAttr("direction", b.getStringAttr(sl.writes ? "Out" : "In")),
         b.getNamedAttr("encoding",
                        b.getStringAttr(rec ? sl.t.carrier
                                            : ("dense." + sl.t.carrier)))}));
    taskRanks.push_back(int64_t(sl.t.dims.size()));
    if (sl.panelStrides.empty())
      contigSlots.push_back(int64_t(i));
    else
      stridedSlots.push_back(int64_t(i));
    for (int64_t pst : sl.panelStrides)
      flatStrides.push_back(pst);
    viewTypes.push_back(b.getStringAttr(sl.t.format("view")));
    argWrites.push_back(b.getBoolAttr(sl.writes));
  }

  OperationState progState(loc, "weft_host.matrix_program");
  progState.addAttribute("tile", tile);
  progState.addAttribute("extents", extents);
  progState.addAttribute("implementation", implAttr);
  progState.addAttribute("uses_brgemm", b.getBoolAttr(usesBrgemm));
  progState.addAttribute("task_slots_json", b.getArrayAttr(slotDicts));
  progState.addAttribute("task_ranks", b.getDenseI64ArrayAttr(taskRanks));
  progState.addAttribute("contiguous_slots", b.getDenseI64ArrayAttr(contigSlots));
  progState.addRegion();
  progState.addRegion();
  Operation *prog = b.create(progState);

  NamedAttribute implNamedAttr = b.getNamedAttr("ci.implementation", implAttr);

  // ---------------- Region 0: Task ("task.weftir") ----------------
  Block &taskTop = prog->getRegion(0).emplaceBlock();
  b.setInsertionPointToEnd(&taskTop);
  if (useEncodedRecords) {
    makeOp(b, loc, "encoding_decl", {}, {},
           {b.getNamedAttr("sym_name", b.getStringAttr("Q4_K")),
            b.getNamedAttr("is_weight", b.getBoolAttr(true)),
            b.getNamedAttr("kir.storage",
                           TypeAttr::get(intentdsl::kir::Q4KType::get(fn.getContext())))});
    makeOp(b, loc, "encoding_decl", {}, {},
           {b.getNamedAttr("sym_name", b.getStringAttr("Q8_K")),
            b.getNamedAttr("is_weight", b.getBoolAttr(false)),
            b.getNamedAttr("kir.storage",
                           TypeAttr::get(intentdsl::kir::Q8KType::get(fn.getContext())))});
  }

  OperationState tkState(loc, "weft_host.task_kernel");
  tkState.addAttribute("sym_name", b.getStringAttr("matrix_panel"));
  tkState.addAttribute("source", b.getStringAttr("CPU output panel"));
  tkState.addAttribute("view_types", b.getArrayAttr(viewTypes));
  tkState.addAttribute("arg_writes", b.getArrayAttr(argWrites));
  tkState.addAttribute("strided_slots", b.getDenseI64ArrayAttr(stridedSlots));
  Block &tkBb = tkState.addRegion()->emplaceBlock();
  SmallVector<Value> viewArgs;
  for (size_t i = 0; i < slots.size(); ++i)
    viewArgs.push_back(tkBb.addArgument(b.getIndexType(), loc));
  Value symR = tkBb.addArgument(b.getIndexType(), loc);
  Value symC = tkBb.addArgument(b.getIndexType(), loc);
  Value rootDom = tkBb.addArgument(b.getIndexType(), loc);
  b.create(tkState);
  makeOp(b, loc, "end");

  b.setInsertionPointToEnd(&tkBb);
  TaskBuilder tb{b, loc};

  auto applyEpilogue = [&](TaskTerm acc, ArrayRef<int64_t> outDims,
                           ArrayRef<int64_t> outAxes,
                           std::optional<Value> biasCoord) -> TaskTerm {
    WeftTypeSpec curSpec{
        acc.type.find("si32") != std::string::npos ? "i32" : "f32",
        SmallVector<int64_t>(outDims.begin(), outDims.end()),
        SmallVector<int64_t>(outAxes.begin(), outAxes.end())};
    TaskTerm cur = acc;
    for (auto &step : epilogue) {
      if (step.kind == EpilogueItem::Bias) {
        SmallVector<std::optional<Value>> bCoords;
        if (biasCoord.has_value())
          bCoords.push_back(biasCoord);
        else
          bCoords.push_back(std::nullopt);
        TaskTerm bv = tb.sliceSlot(viewArgs[step.biasSlot],
                                   slots[step.biasSlot].t, bCoords, true);
        cur = tb.bin("add", cur, bv, cur.type);
      } else {
        curSpec.carrier = step.targetCarrier;
        cur = tb.castOp("cast", cur, curSpec.format("value"));
      }
    }
    return cur;
  };

  if (emptyRed) {
    bool isInt = kind == "contract" && argDtype(slotOf(dot.getOperand(0))) == "i8";
    auto z = tb.idx(0), one = tb.idx(1);
    OperationState rFor(loc, "weft_host.task_for");
    rFor.addOperands({z.v, symR, one.v});
    rFor.addAttribute("iter_types", b.getArrayAttr({}));
    Block &rBb = rFor.addRegion()->emplaceBlock();
    Value rIv = rBb.addArgument(b.getIndexType(), loc);
    b.create(rFor);
    b.setInsertionPointToEnd(&rBb);
    TaskTerm initVal = tb.lit(isInt ? "0" : "0.0", isInt ? "si32" : "f32");
    if (slots.back().t.dims.size() == 2) {
      OperationState cFor(loc, "weft_host.task_for");
      cFor.addOperands({z.v, symC, one.v});
      cFor.addAttribute("iter_types", b.getArrayAttr({}));
      Block &cBb = cFor.addRegion()->emplaceBlock();
      Value cIv = cBb.addArgument(b.getIndexType(), loc);
      b.create(cFor);
      b.setInsertionPointToEnd(&cBb);
      TaskTerm outVal = applyEpilogue(initVal, {}, {}, cIv);
      tb.commit(outVal, tb.sliceSlot(viewArgs.back(), slots.back().t,
                                     {rIv, cIv}, false));
      makeOp(b, loc, "task_yield", {}, {},
             {b.getNamedAttr("val_types", b.getArrayAttr({}))});
      b.setInsertionPointToEnd(&rBb);
    } else {
      TaskTerm outVal = applyEpilogue(initVal, {}, {}, rIv);
      tb.commit(outVal,
                tb.sliceSlot(viewArgs.back(), slots.back().t, {rIv}, false));
    }
    makeOp(b, loc, "task_yield", {}, {},
           {b.getNamedAttr("val_types", b.getArrayAttr({}))});
    b.setInsertionPointToEnd(&tkBb);
  } else if (isQ4k) {
    auto q4StorageAttr = b.getNamedAttr(
        "kir.storage", TypeAttr::get(intentdsl::kir::Q4KType::get(fn.getContext())));
    int64_t ng = K / 256;
    if (useEncodedRecords && !boundWidening) {
      // x86 encoded Q4_K x Q8_K: static [8, 32] subgroup gather + SIMD VNNI
      // contract in ZMM registers (avoids dynamic subquant_level stack spills).
      auto z = tb.idx(0), one = tb.idx(1), endG = tb.idx(ng);
      auto fz = tb.lit("0.0", "f32");
      std::string gIdxTy = "!weft_kernel.value<ui8,[8],[15]>";
      std::string kIdxTy = "!weft_kernel.value<ui8,[32],[16]>";
      std::string coordTy = "!weft_kernel.value<ui8,[8,32],[15,16]>";
      std::string mIdxTy = "!weft_kernel.value<ui32,[8],[17]>";
      Value grpIota =
          makeOp(b, loc, "task_iota", {}, {b.getIndexType()},
                 {b.getNamedAttr("start", b.getI64IntegerAttr(0)),
                  b.getNamedAttr("end", b.getI64IntegerAttr(8)),
                  b.getNamedAttr("val_type", b.getStringAttr(gIdxTy))})
              ->getResult(0);
      Value laneIota =
          makeOp(b, loc, "task_iota", {}, {b.getIndexType()},
                 {b.getNamedAttr("start", b.getI64IntegerAttr(0)),
                  b.getNamedAttr("end", b.getI64IntegerAttr(32)),
                  b.getNamedAttr("val_type", b.getStringAttr(kIdxTy))})
              ->getResult(0);
      TaskTerm grpBase =
          tb.bin("mul", {grpIota, gIdxTy}, tb.lit("32", "ui8"), gIdxTy);
      TaskTerm coord2d = tb.bin("add", grpBase, {laneIota, kIdxTy}, coordTy);
      Value minIota =
          makeOp(b, loc, "task_iota", {}, {b.getIndexType()},
                 {b.getNamedAttr("start", b.getI64IntegerAttr(0)),
                  b.getNamedAttr("end", b.getI64IntegerAttr(8)),
                  b.getNamedAttr("val_type", b.getStringAttr(mIdxTy))})
              ->getResult(0);

      OperationState rFor(loc, "weft_host.task_for");
      rFor.addOperands({z.v, symR, one.v});
      rFor.addAttribute("iter_types", b.getArrayAttr({}));
      Block &rBb = rFor.addRegion()->emplaceBlock();
      Value rIv = rBb.addArgument(b.getIndexType(), loc);
      b.create(rFor);
      b.setInsertionPointToEnd(&rBb);

      OperationState gFor(loc, "weft_host.task_for");
      gFor.addOperands({z.v, endG.v, one.v, fz.v});
      gFor.addTypes({b.getIndexType()});
      gFor.addAttribute("iter_types", b.getArrayAttr({b.getStringAttr("f32")}));
      Block &gBb = gFor.addRegion()->emplaceBlock();
      Value gIv = gBb.addArgument(b.getIndexType(), loc);
      Value gAcc = gBb.addArgument(b.getIndexType(), loc);
      Operation *gLoopOp = b.create(gFor);
      b.setInsertionPointToEnd(&gBb);

      TaskTerm w =
          tb.sliceSlot(viewArgs[0], slots[0].t, {rIv, gIv, std::nullopt}, true);
      TaskTerm x =
          tb.sliceSlot(viewArgs[1], slots[1].t, {gIv, std::nullopt}, true);
      auto field = [&](TaskTerm rec, StringRef name, StringRef elem,
                       int n = 1) -> TaskTerm {
        std::string fTy =
            n == 1 ? elem.str()
                   : ("!weft_kernel.value<" + elem.str() + ",[" +
                      std::to_string(n) + "],[3]>");
        Value fv =
            makeOp(b, loc, "task_field", {rec.v}, {b.getIndexType()},
                   {b.getNamedAttr("field_name", b.getStringAttr(name)),
                    b.getNamedAttr("in_type", b.getStringAttr(rec.type)),
                    b.getNamedAttr("val_type", b.getStringAttr(fTy))})
                ->getResult(0);
        return {fv, fTy};
      };
      auto gather = [&](TaskTerm src, Value idxV, StringRef idxTy,
                        StringRef outTy) -> TaskTerm {
        Value gv =
            makeOp(b, loc, "task_extract", {src.v, idxV}, {b.getIndexType()},
                   {b.getNamedAttr("selector", b.getStringAttr("gather")),
                    b.getNamedAttr("in_type", b.getStringAttr(src.type)),
                    b.getNamedAttr("point_type", b.getStringAttr(idxTy)),
                    b.getNamedAttr("val_type", b.getStringAttr(outTy))})
                ->getResult(0);
        return {gv, outTy.str()};
      };

      std::string wq2dTy = "!weft_kernel.value<ui4,[8,32],[15,16]>";
      std::string xq2dTy = "!weft_kernel.value<si8,[8,32],[15,16]>";
      std::string dot8Ty = "!weft_kernel.value<si32,[8],[15]>";
      TaskTerm wq2d = gather(field(w, "q", "ui4", 256), coord2d.v, coordTy, wq2dTy);
      TaskTerm xq2d = gather(field(x, "q", "si8", 256), coord2d.v, coordTy, xq2dTy);
      Value dot8Val =
          makeOp(b, loc, "task_contract", {wq2d.v, xq2d.v}, {b.getIndexType()},
                 {b.getNamedAttr("kind", b.getStringAttr("contract")),
                  b.getNamedAttr("over", b.getDenseI64ArrayAttr({16})),
                  b.getNamedAttr("acc_type", b.getStringAttr("si32")),
                  b.getNamedAttr("engine", b.getStringAttr("auto")),
                  b.getNamedAttr("allow_subnormal_ftz", b.getBoolAttr(false)),
                  b.getNamedAttr(
                      "in_types",
                      b.getArrayAttr(
                          {b.getStringAttr(wq2dTy), b.getStringAttr(xq2dTy)})),
                  b.getNamedAttr("val_type", b.getStringAttr(dot8Ty)),
                  implNamedAttr, q4StorageAttr})
              ->getResult(0);

      TaskTerm scGather = gather(field(w, "sc", "ui6", 8), grpIota, gIdxTy,
                                 "!weft_kernel.value<ui6,[8],[15]>");
      TaskTerm scW = tb.castOp("widen", scGather, dot8Ty);
      TaskTerm scaledDots = tb.bin("mul", {dot8Val, dot8Ty}, scW, dot8Ty);
      Value posSumV =
          makeOp(b, loc, "task_reduce", {scaledDots.v}, {b.getIndexType()},
                 {b.getNamedAttr("kind", b.getStringAttr("add")),
                  b.getNamedAttr("axis", b.getI64IntegerAttr(0)),
                  b.getNamedAttr("in_type", b.getStringAttr(dot8Ty)),
                  b.getNamedAttr("val_type", b.getStringAttr("si32"))})
              ->getResult(0);

      TaskTerm bsum = field(x, "bsum", "si16", 16);
      std::string grp3Ty = WeftTypeSpec{"i32", {8}, {3}}.format("value");
      Value sumsVal =
          makeOp(b, loc, "task_fold2", {bsum.v}, {b.getIndexType()},
                 {b.getNamedAttr("in_type", b.getStringAttr(bsum.type)),
                  b.getNamedAttr("val_type", b.getStringAttr(grp3Ty))})
              ->getResult(0);
      std::string min8Ty = "!weft_kernel.value<si32,[8],[17]>";
      TaskTerm sums8 = gather({sumsVal, grp3Ty}, minIota, mIdxTy, min8Ty);
      TaskTerm mnGather = gather(field(w, "m", "ui6", 8), minIota, mIdxTy,
                                 "!weft_kernel.value<ui6,[8],[17]>");
      TaskTerm mnW = tb.castOp("widen", mnGather, min8Ty);
      TaskTerm scaledMin = tb.bin("mul", sums8, mnW, min8Ty);
      Value negSumV =
          makeOp(b, loc, "task_reduce", {scaledMin.v}, {b.getIndexType()},
                 {b.getNamedAttr("kind", b.getStringAttr("add")),
                  b.getNamedAttr("axis", b.getI64IntegerAttr(0)),
                  b.getNamedAttr("in_type", b.getStringAttr(min8Ty)),
                  b.getNamedAttr("val_type", b.getStringAttr("si32"))})
              ->getResult(0);

      TaskTerm d = tb.castOp("cast", field(w, "d", "f16"), "f32");
      TaskTerm dm = tb.castOp("cast", field(w, "dmin", "f16"), "f32");
      TaskTerm pos =
          tb.bin("mul", d, tb.castOp("cast", {posSumV, "si32"}, "f32"), "f32");
      TaskTerm neg =
          tb.bin("mul", dm, tb.castOp("cast", {negSumV, "si32"}, "f32"), "f32");
      TaskTerm vrec = tb.bin("mul", field(x, "ds", "f32"),
                             tb.bin("sub", pos, neg, "f32"), "f32");
      TaskTerm nAcc = tb.bin("add", {gAcc, "f32"}, vrec, "f32");
      makeOp(b, loc, "task_yield", {nAcc.v}, {},
             {b.getNamedAttr("val_types", b.getArrayAttr({b.getStringAttr("f32")}))});

      b.setInsertionPointToEnd(&rBb);
      TaskTerm finalRow =
          applyEpilogue({gLoopOp->getResult(0), "f32"}, {}, {}, rIv);
      tb.commit(finalRow,
                tb.sliceSlot(viewArgs.back(), slots.back().t, {rIv}, false));
      makeOp(b, loc, "task_yield", {}, {},
             {b.getNamedAttr("val_types", b.getArrayAttr({}))});
      b.setInsertionPointToEnd(&tkBb);
    } else if (useEncodedRecords) {
      auto z = tb.idx(0), one = tb.idx(1), endG = tb.idx(ng);
      auto fz = tb.lit("0.0", "f32"), iz = tb.lit("0", "si32");
      OperationState rFor(loc, "weft_host.task_for");
      rFor.addOperands({z.v, symR, one.v});
      rFor.addAttribute("iter_types", b.getArrayAttr({}));
      Block &rBb = rFor.addRegion()->emplaceBlock();
      Value rIv = rBb.addArgument(b.getIndexType(), loc);
      b.create(rFor);
      b.setInsertionPointToEnd(&rBb);

      OperationState gFor(loc, "weft_host.task_for");
      gFor.addOperands({z.v, endG.v, one.v, fz.v});
      gFor.addTypes({b.getIndexType()});
      gFor.addAttribute("iter_types", b.getArrayAttr({b.getStringAttr("f32")}));
      Block &gBb = gFor.addRegion()->emplaceBlock();
      Value gIv = gBb.addArgument(b.getIndexType(), loc);
      Value gAcc = gBb.addArgument(b.getIndexType(), loc);
      Operation *gLoopOp = b.create(gFor);
      b.setInsertionPointToEnd(&gBb);

      TaskTerm w = tb.sliceSlot(viewArgs[0], slots[0].t, {rIv, gIv, std::nullopt}, true);
      TaskTerm x = tb.sliceSlot(viewArgs[1], slots[1].t, {gIv, std::nullopt}, true);
      auto field = [&](TaskTerm rec, StringRef name, StringRef elem,
                       int n = 1) -> TaskTerm {
        std::string fTy =
            n == 1 ? elem.str()
                   : ("!weft_kernel.value<" + elem.str() + ",[" +
                      std::to_string(n) + "],[3]>");
        Value fv =
            makeOp(b, loc, "task_field", {rec.v}, {b.getIndexType()},
                   {b.getNamedAttr("field_name", b.getStringAttr(name)),
                    b.getNamedAttr("in_type", b.getStringAttr(rec.type)),
                    b.getNamedAttr("val_type", b.getStringAttr(fTy))})
                ->getResult(0);
        return {fv, fTy};
      };
      TaskTerm d = tb.castOp("cast", field(w, "d", "f16"), "f32");
      TaskTerm dm = tb.castOp("cast", field(w, "dmin", "f16"), "f32");
      auto nonzero = [&](TaskTerm t) {
        auto bits = tb.castOp("bitcast", t, "si32");
        auto mag = tb.bin("and", bits, tb.lit("2147483647", "si32"), "si32");
        return tb.cmp("ne", mag, iz, "i1");
      };
      TaskTerm live = tb.bin("or", nonzero(d), nonzero(dm), "i1");

      OperationState ifSt(loc, "weft_host.task_if");
      ifSt.addOperands({live.v});
      ifSt.addTypes({b.getIndexType()});
      ifSt.addAttribute("res_types", b.getArrayAttr({b.getStringAttr("f32")}));
      Block &thenBb = ifSt.addRegion()->emplaceBlock();
      Block &elseBb = ifSt.addRegion()->emplaceBlock();
      Operation *ifOp = b.create(ifSt);

      b.setInsertionPointToEnd(&thenBb);
      TaskTerm wq = field(w, "q", "ui4", 256), xq = field(x, "q", "si8", 256);
      TaskTerm sc = field(w, "sc", "ui6", 8), mn = field(w, "m", "ui6", 8);
      TaskTerm bsum = field(x, "bsum", "si16", 16);
      std::string grpTy = WeftTypeSpec{"i32", {8}, {3}}.format("value");
      Value sumsVal =
          makeOp(b, loc, "task_fold2", {bsum.v}, {b.getIndexType()},
                 {b.getNamedAttr("in_type", b.getStringAttr(bsum.type)),
                  b.getNamedAttr("val_type", b.getStringAttr(grpTy))})
              ->getResult(0);
      TaskTerm sums{sumsVal, grpTy};

      OperationState lvlSt(loc, "weft_host.task_subquant_level");
      lvlSt.addOperands({rootDom, iz.v, iz.v});
      lvlSt.addTypes({b.getIndexType(), b.getIndexType()});
      Block &lvlBb = lvlSt.addRegion()->emplaceBlock();
      Value ptArg = lvlBb.addArgument(b.getIndexType(), loc);
      Value aS = lvlBb.addArgument(b.getIndexType(), loc);
      Value aT = lvlBb.addArgument(b.getIndexType(), loc);
      Operation *lvlOp = b.create(lvlSt);

      b.setInsertionPointToEnd(&lvlBb);
      std::string ptTy =
          "!weft_kernel.point<<\"subquant.1\",1,0,3,\"blocks\",\"exact\">>";
      auto extract = [&](TaskTerm src, StringRef outTy, bool subDom) -> TaskTerm {
        Value ev =
            makeOp(b, loc, "task_extract", {src.v, ptArg}, {b.getIndexType()},
                   {b.getNamedAttr("selector",
                                   b.getStringAttr(subDom ? "domain"
                                                          : "group_index")),
                    b.getNamedAttr("in_type", b.getStringAttr(src.type)),
                    b.getNamedAttr("point_type", b.getStringAttr(ptTy)),
                    b.getNamedAttr("val_type", b.getStringAttr(outTy))})
                ->getResult(0);
        return {ev, outTy.str()};
      };
      TaskTerm u =
          extract(wq, WeftTypeSpec{"ui4", {32}, {3}}.format("value"), true);
      TaskTerm v =
          extract(xq, WeftTypeSpec{"i8", {32}, {3}}.format("value"), true);
      Value dotVal =
          makeOp(b, loc, "task_contract", {u.v, v.v}, {b.getIndexType()},
                 {b.getNamedAttr("kind", b.getStringAttr("contract")),
                  b.getNamedAttr("over", b.getDenseI64ArrayAttr({3})),
                  b.getNamedAttr("acc_type", b.getStringAttr("si32")),
                  b.getNamedAttr("engine", b.getStringAttr("auto")),
                  b.getNamedAttr("allow_subnormal_ftz", b.getBoolAttr(false)),
                  b.getNamedAttr(
                      "in_types",
                      b.getArrayAttr({b.getStringAttr(u.type),
                                      b.getStringAttr(v.type)})),
                  b.getNamedAttr("val_type", b.getStringAttr("si32")),
                  implNamedAttr, q4StorageAttr})
              ->getResult(0);
      TaskTerm scW = tb.castOp("widen", extract(sc, "ui6", false), "si32");
      TaskTerm mnW = tb.castOp("widen", extract(mn, "ui6", false), "si32");
      TaskTerm nextS = tb.bin("add", {aS, "si32"},
                              tb.bin("mul", {dotVal, "si32"}, scW, "si32"), "si32");
      TaskTerm nextT =
          tb.bin("add", {aT, "si32"},
                 tb.bin("mul", extract(sums, "si32", false), mnW, "si32"), "si32");
      makeOp(b, loc, "task_handoff", {nextS.v, nextT.v});

      b.setInsertionPointToEnd(&thenBb);
      TaskTerm pos = tb.bin(
          "mul", d, tb.castOp("cast", {lvlOp->getResult(0), "si32"}, "f32"), "f32");
      TaskTerm neg = tb.bin(
          "mul", dm, tb.castOp("cast", {lvlOp->getResult(1), "si32"}, "f32"), "f32");
      TaskTerm vrec = tb.bin("mul", field(x, "ds", "f32"),
                             tb.bin("sub", pos, neg, "f32"), "f32");
      makeOp(b, loc, "task_yield", {vrec.v}, {},
             {b.getNamedAttr("val_types", b.getArrayAttr({b.getStringAttr("f32")}))});

      b.setInsertionPointToEnd(&elseBb);
      makeOp(b, loc, "task_yield", {fz.v}, {},
             {b.getNamedAttr("val_types", b.getArrayAttr({b.getStringAttr("f32")}))});

      b.setInsertionPointToEnd(&gBb);
      TaskTerm nAcc =
          tb.bin("add", {gAcc, "f32"}, {ifOp->getResult(0), "f32"}, "f32");
      makeOp(b, loc, "task_yield", {nAcc.v}, {},
             {b.getNamedAttr("val_types", b.getArrayAttr({b.getStringAttr("f32")}))});

      b.setInsertionPointToEnd(&rBb);
      TaskTerm finalRow = applyEpilogue({gLoopOp->getResult(0), "f32"}, {}, {}, rIv);
      tb.commit(finalRow,
                tb.sliceSlot(viewArgs.back(), slots.back().t, {rIv}, false));
      makeOp(b, loc, "task_yield", {}, {},
             {b.getNamedAttr("val_types", b.getArrayAttr({}))});
      b.setInsertionPointToEnd(&tkBb);
    } else if (!boundWidening) {
      // Multi-row Q4_K subgroup contraction across all R rows in parallel!
      // Identical Canonical IR for both native ("auto") and vendor ("onednn-brgemm-u8s8").
      auto z = tb.idx(0), one = tb.idx(1), eight = tb.idx(8), endG = tb.idx(ng);
      auto i0 = tb.idx(0), i1 = tb.idx(1), i2 = tb.idx(2), i3 = tb.idx(3);
      auto fz = tb.lit("0.0", "f32"), iz = tb.lit("0", "si32");
      auto c = [&](int n) { return tb.lit(std::to_string(n), "si32"); };
      std::string fVecTy = WeftTypeSpec{"f32", {-1}, {1}}.format("value");
      std::string iVecTy = WeftTypeSpec{"i32", {-1}, {1}}.format("value");
      std::string u16VecTy = "!weft_kernel.value<ui16,[-1],[1]>";
      std::string f16VecTy = WeftTypeSpec{"f16", {-1}, {1}}.format("value");
      std::string i1VecTy = "!weft_kernel.value<i1,[-1],[1]>";
      TaskTerm zfVec = tb.newVal(fz, fVecTy);
      TaskTerm ziVec = tb.newVal(iz, iVecTy);

      OperationState gFor(loc, "weft_host.task_for");
      gFor.addOperands({z.v, endG.v, one.v, zfVec.v});
      gFor.addTypes({b.getIndexType()});
      gFor.addAttribute("iter_types", b.getArrayAttr({b.getStringAttr(fVecTy)}));
      Block &gBb = gFor.addRegion()->emplaceBlock();
      Value gIv = gBb.addArgument(b.getIndexType(), loc);
      Value gAcc = gBb.addArgument(b.getIndexType(), loc);
      Operation *gLoopOp = b.create(gFor);
      b.setInsertionPointToEnd(&gBb);

      auto byteVec = [&](Value offVal) {
        return tb.castOp(
            "cast",
            tb.sliceSlot(viewArgs[0], slots[0].t, {std::nullopt, gIv, offVal}, true),
            iVecTy);
      };
      auto halfVec = [&](Value loOff, Value hiOff) {
        auto lo = byteVec(loOff), hi = byteVec(hiOff);
        auto bits = tb.bin("or", lo, tb.bin("shl", hi, c(8), iVecTy), iVecTy);
        return tb.castOp("cast",
                         tb.castOp("bitcast", tb.castOp("cast", bits, u16VecTy),
                                   f16VecTy),
                         fVecTy);
      };
      TaskTerm d = halfVec(i0.v, i1.v), dm = halfVec(i2.v, i3.v);
      auto activeVec = [&](TaskTerm x) {
        auto raw = tb.castOp("bitcast", x, iVecTy);
        auto mag = tb.bin("and", raw, c(2147483647), iVecTy);
        return tb.cmp("ne", mag, iz, i1VecTy);
      };
      TaskTerm live = tb.bin("or", activeVec(d), activeVec(dm), i1VecTy);

      OperationState jFor(loc, "weft_host.task_for");
      jFor.addOperands({z.v, eight.v, one.v, ziVec.v, ziVec.v});
      jFor.addTypes({b.getIndexType(), b.getIndexType()});
      jFor.addAttribute(
          "iter_types",
          b.getArrayAttr({b.getStringAttr(iVecTy), b.getStringAttr(iVecTy)}));
      Block &jBb = jFor.addRegion()->emplaceBlock();
      Value jIv = jBb.addArgument(b.getIndexType(), loc);
      Value sCarry = jBb.addArgument(b.getIndexType(), loc);
      Value tCarry = jBb.addArgument(b.getIndexType(), loc);
      Operation *jLoopOp = b.create(jFor);
      b.setInsertionPointToEnd(&jBb);

      TaskTerm ji = tb.castOp("cast", {jIv, "index"}, "si32");
      TaskTerm sub = tb.bin("and", ji, c(3), "si32");
      auto toIdx = [&](TaskTerm t) { return tb.castOp("cast", t, "index").v; };
      TaskTerm l = byteVec(toIdx(tb.bin("add", sub, c(4), "si32")));
      TaskTerm h = byteVec(toIdx(tb.bin("add", sub, c(8), "si32")));
      TaskTerm u = byteVec(toIdx(tb.bin("add", sub, c(12), "si32")));
      TaskTerm mask = tb.cmp("lt", ji, c(4), "i1");
      auto factor = [&](TaskTerm low, TaskTerm high) {
        auto lo6 = tb.bin("and", low, c(63), iVecTy);
        auto hi6 = tb.bin(
            "or", high,
            tb.bin("shl", tb.bin("shr", low, c(6), iVecTy), c(4), iVecTy),
            iVecTy);
        return tb.sel(mask, lo6, hi6);
      };
      TaskTerm sc = factor(l, tb.bin("and", u, c(15), iVecTy));
      TaskTerm mn = factor(h, tb.bin("shr", u, c(4), iVecTy));

      Value jHalf = toIdx(tb.bin("shr", ji, c(1), "si32"));
      TaskTerm raw = tb.sliceSlot(
          viewArgs[1], slots[1].t, {std::nullopt, gIv, jHalf, std::nullopt}, true);
      TaskTerm shift = tb.castOp(
          "cast", tb.bin("shl", tb.bin("and", ji, c(1), "si32"), c(2), "si32"),
          "ui8");
      TaskTerm nib = tb.bin("and", tb.bin("shr", raw, shift, raw.type),
                            tb.lit("15", "ui8"), raw.type);
      TaskTerm q =
          tb.sliceSlot(viewArgs[3], slots[3].t, {gIv, jIv, std::nullopt}, true);

      std::string reqEngine = usesBrgemm ? "onednn-brgemm-u8s8" : "auto";
      TaskTerm nibForDot =
          usesBrgemm
              ? nib
              : tb.castOp("cast", nib,
                          WeftTypeSpec{"i8", {-1, 32}, {1, 3}}.format("value"));
      Value dotRes =
          makeOp(b, loc, "task_contract", {nibForDot.v, q.v}, {b.getIndexType()},
                 {b.getNamedAttr("kind", b.getStringAttr("contract")),
                  b.getNamedAttr("over", b.getDenseI64ArrayAttr({3})),
                  b.getNamedAttr("acc_type", b.getStringAttr("si32")),
                  b.getNamedAttr("engine", b.getStringAttr(reqEngine)),
                  b.getNamedAttr("allow_subnormal_ftz", b.getBoolAttr(false)),
                  b.getNamedAttr(
                      "in_types",
                      b.getArrayAttr({b.getStringAttr(nibForDot.type),
                                      b.getStringAttr(q.type)})),
                  b.getNamedAttr("val_type", b.getStringAttr(iVecTy)),
                  implNamedAttr, q4StorageAttr})
              ->getResult(0);
      TaskTerm nextS = tb.bin(
          "add", {sCarry, iVecTy}, tb.bin("mul", sc, {dotRes, iVecTy}, iVecTy),
          iVecTy);

      TaskTerm jj = tb.bin("shl", ji, c(1), "si32");
      TaskTerm b0 = tb.castOp(
          "cast",
          tb.sliceSlot(viewArgs[4], slots[4].t, {gIv, toIdx(jj)}, true), "si32");
      TaskTerm b1 = tb.castOp(
          "cast",
          tb.sliceSlot(viewArgs[4], slots[4].t,
                       {gIv, toIdx(tb.bin("add", jj, c(1), "si32"))}, true),
          "si32");
      TaskTerm sumq = tb.bin("add", b0, b1, "si32");
      TaskTerm nextT = tb.bin("add", {tCarry, iVecTy},
                              tb.bin("mul", mn, sumq, iVecTy), iVecTy);
      makeOp(b, loc, "task_yield", {nextS.v, nextT.v}, {},
             {b.getNamedAttr(
                 "val_types",
                 b.getArrayAttr(
                     {b.getStringAttr(iVecTy), b.getStringAttr(iVecTy)}))});

      b.setInsertionPointToEnd(&gBb);
      TaskTerm pos = tb.bin(
          "mul", d, tb.castOp("cast", {jLoopOp->getResult(0), iVecTy}, fVecTy),
          fVecTy);
      TaskTerm neg = tb.bin(
          "mul", dm, tb.castOp("cast", {jLoopOp->getResult(1), iVecTy}, fVecTy),
          fVecTy);
      TaskTerm ds = tb.sliceSlot(viewArgs[2], slots[2].t, {gIv}, true);
      TaskTerm contribRaw =
          tb.bin("mul", tb.bin("sub", pos, neg, fVecTy), ds, fVecTy);
      TaskTerm contrib = tb.sel(live, contribRaw, zfVec);
      TaskTerm nAcc = tb.bin("add", {gAcc, fVecTy}, contrib, fVecTy);
      makeOp(b, loc, "task_yield", {nAcc.v}, {},
             {b.getNamedAttr("val_types",
                             b.getArrayAttr({b.getStringAttr(fVecTy)}))});

      b.setInsertionPointToEnd(&tkBb);
      TaskTerm finalVec =
          applyEpilogue({gLoopOp->getResult(0), fVecTy}, {-1}, {1}, std::nullopt);
      tb.commit(finalVec,
                tb.sliceSlot(viewArgs.back(), slots.back().t, {std::nullopt},
                             false));
    } else {
      // Strided RVV fallback: row-wise Q4_K subgroup loop fitting within 32 RVV registers.
      auto z = tb.idx(0), one = tb.idx(1), eight = tb.idx(8), endG = tb.idx(ng);
      auto fz = tb.lit("0.0", "f32"), iz = tb.lit("0", "si32");
      auto c = [&](int n) { return tb.lit(std::to_string(n), "si32"); };
      OperationState rFor(loc, "weft_host.task_for");
      rFor.addOperands({z.v, symR, one.v});
      rFor.addAttribute("iter_types", b.getArrayAttr({}));
      Block &rBb = rFor.addRegion()->emplaceBlock();
      Value rIv = rBb.addArgument(b.getIndexType(), loc);
      b.create(rFor);
      b.setInsertionPointToEnd(&rBb);

      OperationState gFor(loc, "weft_host.task_for");
      gFor.addOperands({z.v, endG.v, one.v, fz.v});
      gFor.addTypes({b.getIndexType()});
      gFor.addAttribute("iter_types", b.getArrayAttr({b.getStringAttr("f32")}));
      Block &gBb = gFor.addRegion()->emplaceBlock();
      Value gIv = gBb.addArgument(b.getIndexType(), loc);
      Value gAcc = gBb.addArgument(b.getIndexType(), loc);
      Operation *gLoopOp = b.create(gFor);
      b.setInsertionPointToEnd(&gBb);

      auto toIdx = [&](TaskTerm t) { return tb.castOp("cast", t, "index").v; };
      auto byteSca = [&](Value offVal) {
        return tb.castOp(
            "cast",
            tb.sliceSlot(viewArgs[0], slots[0].t, {rIv, gIv, offVal}, true),
            "si32");
      };
      auto halfSca = [&](int off) {
        auto lo = byteSca(tb.idx(off).v), hi = byteSca(tb.idx(off + 1).v);
        auto bits = tb.bin("or", lo, tb.bin("shl", hi, c(8), "si32"), "si32");
        return tb.castOp(
            "cast", tb.castOp("bitcast", tb.castOp("cast", bits, "ui16"), "f16"),
            "f32");
      };
      TaskTerm d = halfSca(0), dm = halfSca(2);
      auto activeSca = [&](TaskTerm x) {
        auto raw = tb.castOp("bitcast", x, "si32");
        return tb.cmp("ne", tb.bin("and", raw, c(2147483647), "si32"), iz, "i1");
      };
      TaskTerm live = tb.bin("or", activeSca(d), activeSca(dm), "i1");

      OperationState ifSt(loc, "weft_host.task_if");
      ifSt.addOperands({live.v});
      ifSt.addTypes({b.getIndexType()});
      ifSt.addAttribute("res_types", b.getArrayAttr({b.getStringAttr("f32")}));
      Block &thenBb = ifSt.addRegion()->emplaceBlock();
      Block &elseBb = ifSt.addRegion()->emplaceBlock();
      Operation *ifOp = b.create(ifSt);
      b.setInsertionPointToEnd(&thenBb);

      OperationState jFor(loc, "weft_host.task_for");
      jFor.addOperands({z.v, eight.v, one.v, iz.v, iz.v});
      jFor.addTypes({b.getIndexType(), b.getIndexType()});
      jFor.addAttribute(
          "iter_types",
          b.getArrayAttr({b.getStringAttr("si32"), b.getStringAttr("si32")}));
      Block &jBb = jFor.addRegion()->emplaceBlock();
      Value jIv = jBb.addArgument(b.getIndexType(), loc);
      Value sCarry = jBb.addArgument(b.getIndexType(), loc);
      Value tCarry = jBb.addArgument(b.getIndexType(), loc);
      Operation *jLoopOp = b.create(jFor);
      b.setInsertionPointToEnd(&jBb);

      TaskTerm ji = tb.castOp("cast", {jIv, "index"}, "si32");
      TaskTerm sub = tb.bin("and", ji, c(3), "si32");
      TaskTerm l = byteSca(toIdx(tb.bin("add", sub, c(4), "si32")));
      TaskTerm h = byteSca(toIdx(tb.bin("add", sub, c(8), "si32")));
      TaskTerm u = byteSca(toIdx(tb.bin("add", sub, c(12), "si32")));
      TaskTerm mask = tb.cmp("lt", ji, c(4), "i1");
      auto factor = [&](TaskTerm low, TaskTerm high) {
        return tb.sel(
            mask, tb.bin("and", low, c(63), "si32"),
            tb.bin("or", high,
                   tb.bin("shl", tb.bin("shr", low, c(6), "si32"), c(4), "si32"),
                   "si32"));
      };
      TaskTerm sc = factor(l, tb.bin("and", u, c(15), "si32"));
      TaskTerm mn = factor(h, tb.bin("shr", u, c(4), "si32"));
      TaskTerm raw = tb.sliceSlot(
          viewArgs[1], slots[1].t,
          {rIv, gIv, toIdx(tb.bin("shr", ji, c(1), "si32")), std::nullopt},
          true);
      TaskTerm shift = tb.castOp(
          "cast", tb.bin("shl", tb.bin("and", ji, c(1), "si32"), c(2), "si32"),
          "ui8");
      TaskTerm nib = tb.bin("and", tb.bin("shr", raw, shift, raw.type),
                            tb.lit("15", "ui8"), raw.type);
      TaskTerm nibI8 = tb.castOp(
          "cast", nib, WeftTypeSpec{"i8", {32}, {3}}.format("value"));
      TaskTerm q =
          tb.sliceSlot(viewArgs[3], slots[3].t, {gIv, jIv, std::nullopt}, true);
      Value dotRes =
          makeOp(b, loc, "task_contract", {nibI8.v, q.v}, {b.getIndexType()},
                 {b.getNamedAttr("kind", b.getStringAttr("contract")),
                  b.getNamedAttr("over", b.getDenseI64ArrayAttr({3})),
                  b.getNamedAttr("acc_type", b.getStringAttr("si32")),
                  b.getNamedAttr("engine", b.getStringAttr("auto")),
                  b.getNamedAttr("allow_subnormal_ftz", b.getBoolAttr(false)),
                  b.getNamedAttr(
                      "in_types",
                      b.getArrayAttr({b.getStringAttr(nibI8.type),
                                      b.getStringAttr(q.type)})),
                  b.getNamedAttr("val_type", b.getStringAttr("si32")),
                  implNamedAttr, q4StorageAttr})
              ->getResult(0);
      TaskTerm nextS = tb.bin(
          "add", {sCarry, "si32"}, tb.bin("mul", sc, {dotRes, "si32"}, "si32"),
          "si32");
      TaskTerm jj = tb.bin("shl", ji, c(1), "si32");
      TaskTerm sumq = tb.bin(
          "add",
          tb.castOp("cast",
                    tb.sliceSlot(viewArgs[4], slots[4].t, {gIv, toIdx(jj)}, true),
                    "si32"),
          tb.castOp("cast",
                    tb.sliceSlot(
                        viewArgs[4], slots[4].t,
                        {gIv, toIdx(tb.bin("add", jj, c(1), "si32"))}, true),
                    "si32"),
          "si32");
      TaskTerm nextT = tb.bin("add", {tCarry, "si32"},
                              tb.bin("mul", mn, sumq, "si32"), "si32");
      makeOp(b, loc, "task_yield", {nextS.v, nextT.v}, {},
             {b.getNamedAttr(
                 "val_types",
                 b.getArrayAttr(
                     {b.getStringAttr("si32"), b.getStringAttr("si32")}))});

      b.setInsertionPointToEnd(&thenBb);
      TaskTerm pos = tb.bin(
          "mul", d, tb.castOp("cast", {jLoopOp->getResult(0), "si32"}, "f32"),
          "f32");
      TaskTerm neg = tb.bin(
          "mul", dm, tb.castOp("cast", {jLoopOp->getResult(1), "si32"}, "f32"),
          "f32");
      TaskTerm contrib =
          tb.bin("mul", tb.sliceSlot(viewArgs[2], slots[2].t, {gIv}, true),
                 tb.bin("sub", pos, neg, "f32"), "f32");
      makeOp(b, loc, "task_yield", {contrib.v}, {},
             {b.getNamedAttr("val_types", b.getArrayAttr({b.getStringAttr("f32")}))});
      b.setInsertionPointToEnd(&elseBb);
      makeOp(b, loc, "task_yield", {fz.v}, {},
             {b.getNamedAttr("val_types", b.getArrayAttr({b.getStringAttr("f32")}))});
      b.setInsertionPointToEnd(&gBb);
      TaskTerm nAcc =
          tb.bin("add", {gAcc, "f32"}, {ifOp->getResult(0), "f32"}, "f32");
      makeOp(b, loc, "task_yield", {nAcc.v}, {},
             {b.getNamedAttr("val_types", b.getArrayAttr({b.getStringAttr("f32")}))});
      b.setInsertionPointToEnd(&rBb);
      TaskTerm finalRow = applyEpilogue({gLoopOp->getResult(0), "f32"}, {}, {}, rIv);
      tb.commit(finalRow,
                tb.sliceSlot(viewArgs.back(), slots.back().t, {rIv}, false));
      makeOp(b, loc, "task_yield", {}, {},
             {b.getNamedAttr("val_types", b.getArrayAttr({}))});
      b.setInsertionPointToEnd(&tkBb);
    }
  } else {
    // 2D / 3D matrix contraction
    bool isInt = slots[0].t.carrier == "i8";
    std::string accCarrier = isInt ? "i32" : "f32";
    std::string accElem = isInt ? "si32" : "f32";
    int64_t staticCols = slots.back().t.dims[1];
    int64_t vecLanes = std::max<int64_t>(1, tile[3]);
    int64_t colGroups =
        staticCols > 0 ? (staticCols + vecLanes - 1) / vecLanes : 99;
    bool canUseOuterContract =
        !boundWidening ||
        (slots[0].t.carrier == "f32" && slots.back().t.carrier == "f32" &&
         staticCols > 0 && mr * colGroups <= 16);

    if (canUseOuterContract) {
      SmallVector<std::optional<Value>> aSel(slots[0].t.dims.size(),
                                             std::nullopt);
      SmallVector<std::optional<Value>> bSel(slots[1].t.dims.size(),
                                             std::nullopt);
      TaskTerm va = tb.sliceSlot(viewArgs[0], slots[0].t, aSel, true);
      TaskTerm vb = tb.sliceSlot(viewArgs[1], slots[1].t, bSel, true);
      if (!usesBrgemm && !isPack32 &&
          (slots[0].t.carrier == "bf16" || slots[0].t.carrier == "i8")) {
        va = tb.castOp(
            "cast", va,
            WeftTypeSpec{accCarrier, slots[0].t.dims, slots[0].t.axes}.format(
                "value"));
        vb = tb.castOp(
            "cast", vb,
            WeftTypeSpec{accCarrier, slots[1].t.dims, slots[1].t.axes}.format(
                "value"));
      }
      SmallVector<int64_t> overAxes = isPack32 ? SmallVector<int64_t>{3, 4}
                                               : SmallVector<int64_t>{3};
      int64_t outColDim = slots.back().t.dims[1];
      std::string accValTy =
          WeftTypeSpec{accCarrier, {-1, outColDim}, {1, 2}}.format("value");
      std::string reqEngine = usesBrgemm ? shortImpl.str() : "auto";
      bool emitFtz = usesBrgemm && allowSubnormalFtz && !isInt;
      Value accVal =
          makeOp(b, loc, "task_contract", {va.v, vb.v}, {b.getIndexType()},
                 {b.getNamedAttr("kind", b.getStringAttr("outer_contract")),
                  b.getNamedAttr("over", b.getDenseI64ArrayAttr(overAxes)),
                  b.getNamedAttr("acc_type", b.getStringAttr(accElem)),
                  b.getNamedAttr("engine", b.getStringAttr(reqEngine)),
                  b.getNamedAttr("allow_subnormal_ftz", b.getBoolAttr(emitFtz)),
                  b.getNamedAttr(
                      "in_types",
                      b.getArrayAttr({b.getStringAttr(va.type),
                                      b.getStringAttr(vb.type)})),
                  b.getNamedAttr("val_type", b.getStringAttr(accValTy)),
                  implNamedAttr})
              ->getResult(0);
      TaskTerm outVal = applyEpilogue({accVal, accValTy}, {-1, outColDim},
                                      {1, 2}, std::nullopt);
      tb.commit(outVal, tb.sliceSlot(viewArgs.back(), slots.back().t,
                                     {std::nullopt, std::nullopt}, false));
    } else {
      // RVV register-bounded 1D vector FMA/MAC loop for bf16/f16/i8
      auto z = tb.idx(0), one = tb.idx(1), endK = tb.idx(K);
      int64_t outColDim = slots.back().t.dims[1];
      WeftTypeSpec vecSpec{accCarrier, {outColDim}, {2}};
      std::string vecTy = vecSpec.format("value");
      int64_t regRows =
          ((slots[0].t.carrier == "f32" || slots[0].t.carrier == "i8") &&
           !grouped && M % 2 == 0 && outColDim > 0)
              ? std::min<int64_t>(mr, 2)
              : 1;
      auto stepR = tb.idx(regRows);

      OperationState rFor(loc, "weft_host.task_for");
      rFor.addOperands({z.v, symR, stepR.v});
      rFor.addAttribute("iter_types", b.getArrayAttr({}));
      Block &rBb = rFor.addRegion()->emplaceBlock();
      Value rIv = rBb.addArgument(b.getIndexType(), loc);
      b.create(rFor);
      b.setInsertionPointToEnd(&rBb);

      TaskTerm initVec =
          tb.newVal(tb.lit(isInt ? "0" : "0.0", accElem), vecTy);
      SmallVector<Value> rowCoords{rIv}, initArgs;
      SmallVector<Attribute> iterTypes;
      for (int64_t m = 0; m < regRows; ++m) {
        if (m > 0)
          rowCoords.push_back(
              tb.bin("add", {rIv, "index"}, tb.idx(m), "index").v);
        initArgs.push_back(initVec.v);
        iterTypes.push_back(b.getStringAttr(vecTy));
      }

      OperationState kFor(loc, "weft_host.task_for");
      SmallVector<Value> kForOps{z.v, endK.v, one.v};
      kForOps.append(initArgs.begin(), initArgs.end());
      kFor.addOperands(kForOps);
      SmallVector<Type> kResTys(regRows, b.getIndexType());
      kFor.addTypes(kResTys);
      kFor.addAttribute("iter_types", b.getArrayAttr(iterTypes));
      Block &kBb = kFor.addRegion()->emplaceBlock();
      Value kIv = kBb.addArgument(b.getIndexType(), loc);
      SmallVector<Value> pCarries;
      for (int64_t m = 0; m < regRows; ++m)
        pCarries.push_back(kBb.addArgument(b.getIndexType(), loc));
      Operation *kLoopOp = b.create(kFor);
      b.setInsertionPointToEnd(&kBb);

      TaskTerm bSlice =
          tb.sliceSlot(viewArgs[1], slots[1].t, {kIv, std::nullopt}, true);
      TaskTerm bWide = tb.castOp("cast", bSlice, vecTy);
      SmallVector<Value> yielded;
      for (int64_t m = 0; m < regRows; ++m) {
        TaskTerm aSlice =
            tb.sliceSlot(viewArgs[0], slots[0].t, {rowCoords[m], kIv}, true);
        TaskTerm aWide = tb.castOp("cast", aSlice, accElem);
        Value nextP =
            makeOp(b, loc, "task_contract", {pCarries[m], bWide.v, aWide.v},
                   {b.getIndexType()},
                   {b.getNamedAttr("kind", b.getStringAttr("fma")),
                    b.getNamedAttr("over", b.getDenseI64ArrayAttr({})),
                    b.getNamedAttr("acc_type", b.getStringAttr(accElem)),
                    b.getNamedAttr("engine", b.getStringAttr("auto")),
                    b.getNamedAttr("allow_subnormal_ftz", b.getBoolAttr(false)),
                    b.getNamedAttr("in_types",
                                   b.getArrayAttr({b.getStringAttr(vecTy),
                                                   b.getStringAttr(vecTy),
                                                   b.getStringAttr(accElem)})),
                    b.getNamedAttr("val_type", b.getStringAttr(vecTy)),
                    implNamedAttr})
                ->getResult(0);
        yielded.push_back(nextP);
      }
      makeOp(b, loc, "task_yield", yielded, {},
             {b.getNamedAttr("val_types", b.getArrayAttr(iterTypes))});

      b.setInsertionPointToEnd(&rBb);
      for (int64_t m = 0; m < regRows; ++m) {
        TaskTerm rowOut = applyEpilogue({kLoopOp->getResult(m), vecTy},
                                        {outColDim}, {2}, std::nullopt);
        tb.commit(rowOut,
                  tb.sliceSlot(viewArgs.back(), slots.back().t,
                               {rowCoords[m], std::nullopt}, false));
      }
      makeOp(b, loc, "task_yield", {}, {},
             {b.getNamedAttr("val_types", b.getArrayAttr({}))});
      b.setInsertionPointToEnd(&tkBb);
    }
  }
  makeOp(b, loc, "end");

  // ---------------- Region 1: Host ("host.c") ----------------
  Block &hostTop = prog->getRegion(1).emplaceBlock();
  b.setInsertionPointToEnd(&hostTop);

  bool usesQ8k = (packing == "shared-q8k-per-call");
  bool usesBf16Math =
      packing == "mapped-right-per-call" || packing == "mapped-f32-panels" ||
      (packing == "f32-panels-per-call" &&
       argDtype(slotOf(dot.getOperand(0))) == "bf16");
  bool usesF16Math = packing == "f32-panels-per-call" &&
                     argDtype(slotOf(dot.getOperand(0))) == "f16";
  bool usesMx8Math = (packing == "decoded-f32-per-call");

  OperationState hfnState(loc, "weft_host.host_fn");
  hfnState.addAttribute("params",
                        b.getStringAttr(intentdsl::cpu::matrixHostParams(fn)));
  hfnState.addAttribute("extents", extents);
  hfnState.addAttribute("tile", tile);
  hfnState.addAttribute("uses_q8k", b.getBoolAttr(usesQ8k));
  hfnState.addAttribute("uses_bf16_math", b.getBoolAttr(usesBf16Math));
  hfnState.addAttribute("uses_f16_math", b.getBoolAttr(usesF16Math));
  hfnState.addAttribute("uses_mx8_math", b.getBoolAttr(usesMx8Math));
  Block &hBb = hfnState.addRegion()->emplaceBlock();
  b.create(hfnState);
  makeOp(b, loc, "end");

  b.setInsertionPointToEnd(&hBb);
  llvm::DenseMap<Value, std::string> bufNames;
  SmallVector<Attribute> activeHeap;
  int bufCounter = 0;
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
                                         : "uint8_t";
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

  for (Operation *pop : prepOps) {
    StringRef pk = pop->getAttrOfType<StringAttr>("kind").getValue();
    SmallVector<NamedAttribute> attrs{
        b.getNamedAttr("kind", b.getStringAttr(pk)),
        b.getNamedAttr("dst_buf",
                       b.getStringAttr(bufNames[pop->getOperands().back()])),
        b.getNamedAttr("cleanup", b.getArrayAttr(activeHeap))};
    for (auto na : pop->getAttrs())
      if (na.getName() != "kind")
        attrs.push_back(na);

    SmallVector<int64_t> opSlots;
    for (Value v : pop->getOperands().drop_back())
      opSlots.push_back(slotOf(v));
    attrs.push_back(
        b.getNamedAttr("src_slots", b.getDenseI64ArrayAttr(opSlots)));
    for (auto [idx, s] : llvm::enumerate(opSlots)) {
      attrs.push_back(b.getNamedAttr("src_strides_" + std::to_string(idx),
                                     argStrides(s)));
      attrs.push_back(b.getNamedAttr(
          "src_dtype_" + std::to_string(idx), b.getStringAttr(argDtype(s))));
    }
    if (kind == "window_contract") {
      auto outSh = argShape(slotOf(dstArg));
      attrs.push_back(b.getNamedAttr(
          "head_extent", b.getI64IntegerAttr(std::max<int64_t>(1, outSh[2]))));
      attrs.push_back(b.getNamedAttr(
          "chunk_extent", b.getI64IntegerAttr(std::max<int64_t>(1, outSh[1]))));
    }

    OperationState ps(loc, "weft_host.host_prep");
    ps.addAttributes(attrs);
    if (!pop->getRegions().empty() && !pop->getRegion(0).empty()) {
      Region *invReg = ps.addRegion();
      Block &invBb = invReg->emplaceBlock();
      auto &srcBb = pop->getRegion(0).front();
      IRMapping mapping;
      for (unsigned ai = 1; ai < srcBb.getNumArguments(); ++ai)
        mapping.map(srcBb.getArgument(ai),
                    invBb.addArgument(srcBb.getArgument(ai).getType(), loc));
      llvm::DenseSet<Value> dependsOnCol{srcBb.getArgument(0)};
      OpBuilder ib(fn.getContext());
      ib.setInsertionPointToEnd(&invBb);
      for (auto &mop : srcBb.without_terminator()) {
        if (llvm::any_of(mop.getOperands(), [&](Value v) {
              return dependsOnCol.contains(v);
            })) {
          dependsOnCol.insert(mop.getResult(0));
        } else {
          ib.clone(mop, mapping);
        }
      }
      OperationState colState(loc, "weft_host.host_col_loop");
      Block &varBb = colState.addRegion()->emplaceBlock();
      ib.create(colState);
      makeOp(ib, loc, "end");

      Value colArg = varBb.addArgument(srcBb.getArgument(0).getType(), loc);
      mapping.map(srcBb.getArgument(0), colArg);
      OpBuilder vb(fn.getContext());
      vb.setInsertionPointToEnd(&varBb);
      for (auto &mop : srcBb.without_terminator()) {
        if (dependsOnCol.contains(mop.getResult(0)))
          vb.clone(mop, mapping);
      }
      Operation *term = srcBb.getTerminator();
      makeOp(vb, loc, "host_panel_yield", {mapping.lookup(term->getOperand(0))});
    }
    b.create(ps);
  }

  bool windowed = (kind == "window_contract");
  int64_t headExt = 1, chunkExt = 1;
  if (windowed) {
    auto outSh = argShape(slotOf(dstArg));
    headExt = std::max<int64_t>(1, outSh[2]);
    chunkExt = std::max<int64_t>(1, outSh[1]);
  }

  OperationState ptState(loc, "weft_host.host_parallel_tiles");
  ptState.addAttribute("extents", extents);
  ptState.addAttribute("tile", tile);
  ptState.addAttribute("grouped", b.getBoolAttr(grouped));
  ptState.addAttribute("routed", b.getBoolAttr(routed));
  ptState.addAttribute("windowed", b.getBoolAttr(windowed));
  ptState.addAttribute("head_extent", b.getI64IntegerAttr(headExt));
  ptState.addAttribute("chunk_extent", b.getI64IntegerAttr(chunkExt));
  ptState.addAttribute("multiplicity", b.getI64IntegerAttr(multiplicity));
  ptState.addAttribute("offsets_slot", b.getI64IntegerAttr(offsetsSlot));
  ptState.addAttribute("offsets_stride", b.getI64IntegerAttr(offsetsStride));
  ptState.addAttribute("routes_slot", b.getI64IntegerAttr(routesSlot));
  ptState.addAttribute("routes_stride", b.getI64IntegerAttr(routesStride));

  Block &ptBb = ptState.addRegion()->emplaceBlock();
  // (%g, %c, %r, %src, %dest, %batch, %chunk, %head, %R, %C)
  Value vG = ptBb.addArgument(b.getIndexType(), loc);
  Value vC = ptBb.addArgument(b.getIndexType(), loc);
  Value vR = ptBb.addArgument(b.getIndexType(), loc);
  Value vSrc = ptBb.addArgument(b.getIndexType(), loc);
  Value vDest = ptBb.addArgument(b.getIndexType(), loc);
  Value vBatch = ptBb.addArgument(b.getIndexType(), loc);
  Value vChunk = ptBb.addArgument(b.getIndexType(), loc);
  Value vHead = ptBb.addArgument(b.getIndexType(), loc);
  Value vRows = ptBb.addArgument(b.getIndexType(), loc);
  Value vCols = ptBb.addArgument(b.getIndexType(), loc);
  b.create(ptState);

  b.setInsertionPointToEnd(&ptBb);
  auto cIdx = [&](int64_t val) {
    return b.create<arith::ConstantIndexOp>(loc, val).getResult();
  };
  auto resolveCoord = [&](CoordTag tag) -> Value {
    switch (tag) {
    case CoordTag::Zero:
      return cIdx(0);
    case CoordTag::G:
      return vG;
    case CoordTag::C:
      return vC;
    case CoordTag::CTile:
      return b.create<arith::DivSIOp>(loc, vC, cIdx(std::max<int64_t>(1, nr)));
    case CoordTag::R:
      return vR;
    case CoordTag::Src:
      return vSrc;
    case CoordTag::Dest:
      return vDest;
    case CoordTag::DestDivMult:
      return b.create<arith::DivSIOp>(
          loc, vDest, cIdx(std::max<int64_t>(1, multiplicity)));
    case CoordTag::DestModMult:
      return b.create<arith::RemSIOp>(
          loc, vDest, cIdx(std::max<int64_t>(1, multiplicity)));
    case CoordTag::Batch:
      return vBatch;
    case CoordTag::Chunk:
      return vChunk;
    case CoordTag::ChunkTimesK:
      return b.create<arith::MulIOp>(loc, vChunk, cIdx(K));
    case CoordTag::Head:
      return vHead;
    }
    return cIdx(0);
  };

  for (auto [i, sl] : llvm::enumerate(slots)) {
    std::string cPtrTy =
        (sl.writes ? "" : "const ") + cElementType(sl.t.carrier) + " *";
    if (sl.isRoutedTempOut) {
      makeOp(b, loc, "host_view_ptr", {}, {},
             {b.getNamedAttr("index", b.getI64IntegerAttr(i)),
              b.getNamedAttr("c_ptr_type", b.getStringAttr(cPtrTy)),
              b.getNamedAttr("base_expr", b.getStringAttr("routed_out")),
              b.getNamedAttr("byte_offset",
                             b.getI64IntegerAttr(sl.routedTempBytes)),
              b.getNamedAttr("routed_temp", b.getUnitAttr())});
      continue;
    }
    SmallVector<Value> addrCoords;
    SmallVector<int64_t> addrStrides;
    int64_t elemScale = sl.t.carrier == "Q4_K" ? 1 : 1;
    (void)elemScale;
    for (auto &term : sl.addrTerms) {
      addrCoords.push_back(resolveCoord(term.coord));
      addrStrides.push_back(term.stride);
    }
    SmallVector<Value> ptrOps;
    if (!addrCoords.empty()) {
      Value offVal =
          makeOp(b, loc, "address", addrCoords, {b.getIndexType()},
                 {b.getNamedAttr("strides", b.getDenseI64ArrayAttr(addrStrides)),
                  b.getNamedAttr("index_bits", b.getI32IntegerAttr(64))})
              ->getResult(0);
      ptrOps.push_back(offVal);
    }
    makeOp(b, loc, "host_view_ptr", ptrOps, {},
           {b.getNamedAttr("index", b.getI64IntegerAttr(i)),
            b.getNamedAttr("c_ptr_type", b.getStringAttr(cPtrTy)),
            b.getNamedAttr("base_expr", b.getStringAttr(sl.baseExpr)),
            b.getNamedAttr("byte_offset", b.getI64IntegerAttr(sl.byteOffset))});
  }

  makeOp(b, loc, "host_call_task", {vRows, vCols}, {},
         {b.getNamedAttr("arity", b.getI64IntegerAttr(slots.size())),
          b.getNamedAttr("strides", b.getDenseI64ArrayAttr(flatStrides))});

  if (routed) {
    int64_t dstBytes =
        cast<MemRefType>(dstArg.getType()).getElementTypeBitWidth() / 8;
    makeOp(b, loc, "host_scatter_routed", {}, {},
           {b.getNamedAttr("dst_slot", b.getI64IntegerAttr(slotOf(dstArg))),
            b.getNamedAttr("multiplicity", b.getI64IntegerAttr(multiplicity)),
            b.getNamedAttr("dst_strides",
                           b.getDenseI64ArrayAttr(origDstStrides)),
            b.getNamedAttr("elem_bytes", b.getI64IntegerAttr(dstBytes)),
            b.getNamedAttr("routes_slot", b.getI64IntegerAttr(routesSlot)),
            b.getNamedAttr("routes_stride", b.getI64IntegerAttr(routesStride))});
  }
  makeOp(b, loc, "end");

  b.setInsertionPointToEnd(&hBb);
  for (Operation *rop : releaseOps) {
    makeOp(b, loc, "host_free", {}, {},
           {b.getNamedAttr("buf_name",
                           b.getStringAttr(bufNames[rop->getOperand(0)]))});
  }
  makeOp(b, loc, "end");

  SmallVector<Operation *> toErase;
  for (auto &op : fn.front())
    if (&op != prog && op.getName().getStringRef() != "func.return")
      toErase.push_back(&op);
  for (Operation *op : llvm::reverse(toErase))
    op->erase();

  return success();
}

bool hasWeftMatrixProgram(ModuleOp m) {
  auto funcs = m.getOps<func::FuncOp>();
  if (!llvm::hasSingleElement(funcs))
    return false;
  auto fn = *funcs.begin();
  return !fn.empty() && !fn.front().empty() &&
         fn.front().front().getName().getStringRef() ==
             "weft_host.matrix_program";
}

namespace {
struct PublishWeftMatrixABI
    : PassWrapper<PublishWeftMatrixABI, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PublishWeftMatrixABI)
  void runOnOperation() final {
    ModuleOp m = getOperation();
    if (!hasWeftMatrixProgram(m))
      return;
    auto fn = *m.getOps<func::FuncOp>().begin();
    auto &prog = fn.front().front();
    OpBuilder b(m.getContext());
    fn->setAttr("weft_host.entry", b.getStringAttr("intent_entry"));
    llvm::SmallVector<char> buf;
    llvm::raw_svector_ostream os(buf);
    if (auto err = describeWeftMatrixPackage(fn, prog, os)) {
      m.emitError() << llvm::toString(std::move(err));
      signalPassFailure();
      return;
    }
    m->setAttr("weft_host.package",
               b.getStringAttr(StringRef(buf.data(), buf.size())));
  }
};
} // namespace

void publishWeftMatrixABI(OpPassManager &pm) {
  pm.addPass(std::make_unique<PublishWeftMatrixABI>());
}

LogicalResult verifyWeftMatrix(ModuleOp m, const TargetProfile &p,
                               bool callable) {
  auto fn = *m.getOps<func::FuncOp>().begin();
  auto &prog = fn.front().front();
  if (prog.getNumRegions() != 2 || prog.getRegion(0).empty() ||
      prog.getRegion(1).empty())
    return prog.emitError("incomplete weft_host.matrix_program regions");

  bool badAddr = m.walk([&](Operation *op) {
                    if (op->getName().getStringRef() != "weft_host.address")
                      return WalkResult::advance();
                    auto bits = op->getAttrOfType<IntegerAttr>("index_bits");
                    if (!bits || (bits.getInt() != 32 && bits.getInt() != 64)) {
                      op->emitError(
                          "weft_host.address requires i32 or i64 index_bits");
                      return WalkResult::interrupt();
                    }
                    if (bits.getInt() == 32 && !canNarrowWeftAddress(*op)) {
                      op->emitError(
                          "32-bit Weft matrix address lacks a proven signed i32 range");
                      return WalkResult::interrupt();
                    }
                    return WalkResult::advance();
                  }).wasInterrupted();
  if (badAddr)
    return failure();

  if (failed(weftMatrixPrinter().verify(m, p)))
    return failure();

  if (callable) {
    if (fn->getAttrOfType<StringAttr>("weft_host.entry") != "intent_entry")
      return fn.emitError("missing synchronous Weft matrix host entry");
    llvm::SmallVector<char> expected;
    llvm::raw_svector_ostream os(expected);
    if (auto err = describeWeftMatrixPackage(fn, prog, os))
      return fn.emitError() << llvm::toString(std::move(err)), failure();
    auto pkg = m->getAttrOfType<StringAttr>("weft_host.package");
    if (!pkg || pkg.getValue() != StringRef(expected.data(), expected.size()))
      return fn.emitError("weft_host.package disagrees with verified matrix ABI"),
             failure();
  }
  return success();
}
} // namespace CI_UNIT_NAMESPACE
