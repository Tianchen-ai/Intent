#include "Rows.h"
#include "../capabilities/SegmentGather.h"
#include "Broadcast.h"
#include "Slab.h"
#include "intentdsl/exits/bangc/ir/Dialect.h"
#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "intentdsl/levels/algorithm/kir/ir/Segment.h"
#include "intentdsl/levels/execution/block/ir/Grid.h"
#include "intentdsl/levels/storage/dsa/ir/Matrix.h"
#include "intentdsl/levels/storage/dsa/ir/Supply.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <numeric>
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
  llvm::json::Object pkgObj{
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
                                            : llvm::json::Value(guard[1])}}}};
  if (layout.contains("library_impl"))
    pkgObj["cnnl"] = true;
  out << llvm::json::Value(std::move(pkgObj));
  return llvm::Error::success();
}

namespace {
bool isFilterPrefix(Operation *scan) {
  if (!scan || scan->getName().getStringRef() != "block.scan_sum" ||
      !scan->hasOneUse())
    return false;
  auto *op = *scan->getUsers().begin();
  if (op->getName().getStringRef() != "dsa.scatter_fragment" ||
      op->getOperand(6) != scan->getResult(0))
    return false;
  auto *cast = scan->getOperand(0).getDefiningOp();
  return cast && cast->getName().getStringRef() == "block.cast" &&
         cast->getOperand(0) == op->getOperand(5) &&
         getElementTypeOrSelf(scan->getResult(0).getType()).isInteger(32) &&
         getElementTypeOrSelf(op->getOperand(7).getType()).isInteger(32);
}

Value filterNonzeroInput(Operation *scan) {
  if (!isFilterPrefix(scan))
    return {};
  auto cast = scan->getOperand(0).getDefiningOp();
  auto pred = cast->getOperand(0).getDefiningOp();
  if (!cast->hasOneUse() || !pred ||
      pred->getName().getStringRef() != "block.ne" ||
      std::distance(pred->use_begin(), pred->use_end()) != 2)
    return {};
  auto src = pred->getOperand(0);
  auto zero = pred->getOperand(1).getDefiningOp();
  auto v = zero ? zero->getAttrOfType<FloatAttr>("value") : FloatAttr{};
  if (!v || !v.getValue().isZero() ||
      !getElementTypeOrSelf(src.getType()).isF32() || !src.getDefiningOp() ||
      src.getDefiningOp()->getName().getStringRef() != "dsa.await")
    return {};
  return src;
}

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
  int64_t minPitch = INT64_MAX;
  bool multiMember = true;
  Operation *constAnchor = nullptr;
  llvm::DenseMap<int64_t, Value> constI64Map;
  llvm::DenseMap<int32_t, Value> constI32Map;
  llvm::DenseMap<std::pair<Value, int64_t>, Value> dimCache;
  llvm::DenseMap<std::pair<Value, int64_t>, Value> strideCache;
  llvm::DenseMap<std::tuple<Block *, Value, Value, int64_t>, Value>
      memberCoordCache;
  Value storage;
  Value ticketCounts, ticketBases, ticketValues;
  Value segmentTable, segmentOffsets, segmentScratch;
  Value histogramRaw, histogramKeys, histogramFilters, histogramAxis;
  llvm::DenseMap<Value, Value> localBroadcasts;
  int64_t boundRows = 0, boundColumns = 0;
  int64_t widthMultiple = 0;
  bool fullMembers = false, fullColumns = false;
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
    auto slot = intentdsl::domainSlot(fn);
    int64_t m = 1, n = 1;
    if (slot >= 0 && slot < fn.getNumArguments()) {
      auto domain = fn.getArgument(slot);
      auto shape = dyn_cast<MemRefType>(domain.getType());
      auto owner = fn->getAttrOfType<IntegerAttr>("kir.owner_axis");
      for (int64_t i = 0; shape && i < shape.getRank(); ++i) {
        auto extent = boundViewField(
            domain, i, *p.services.get<intentdsl::CallFacts>(), false);
        bool outer = owner ? i == owner.getInt() : i + 1 < shape.getRank();
        auto &size = outer ? m : n;
        if (!extent || *extent <= 0 ||
            __builtin_mul_overflow(size, *extent, &size)) {
          m = n = 0;
          break;
        }
      }
      fullMembers = shape && m > 0 && (resident || batch == 1);
      fullColumns = shape && n > 0;
      if (shape) {
        boundRows = m;
        boundColumns = n;
      }
    }
    if (auto rowsAttr = layout.getAs<ArrayAttr>("rows")) {
      for (auto a : rowsAttr) {
        auto alloc = cast<DictionaryAttr>(a);
        auto id = alloc.getAs<IntegerAttr>("id");
        minPitch = std::min(minPitch, alloc.getAs<IntegerAttr>("pitch").getInt());
        for (auto cfg : fn->getAttrOfType<ArrayAttr>("block.configs")) {
          auto row = cast<DictionaryAttr>(cfg);
          if (row.get("id") == id) {
            maxRows = std::max(maxRows, row.getAs<IntegerAttr>("rows").getInt());
            int64_t members = row.getAs<IntegerAttr>("rows").getInt() * batch;
            widthMultiple = std::gcd(
                widthMultiple, row.getAs<IntegerAttr>("width").getInt());
            fullMembers &= m % members == 0;
            fullColumns &= n % row.getAs<IntegerAttr>("width").getInt() == 0;
          }
        }
      }
    }
    if (minPitch == INT64_MAX)
      minPitch = 0;
    multiMember = (maxRows * batch) > 1;
  }

  void prepareHistogram(ArrayRef<Operation *> ops) {
    if (auto tickets = layout.getAs<DenseI64ArrayAttr>("ticket_panel")) {
      auto buffer = [&](int64_t n) {
        return makeOp("nram_buffer", {},
                      {ptrType(b.getContext(), b.getI32Type(), 1)},
                      {b.getNamedAttr("elements", b.getI64IntegerAttr(n))})
            ->getResult(0);
      };
      ticketCounts = buffer(tickets[0]);
      ticketBases = buffer(tickets[0]);
      ticketValues = buffer(tickets[1]);
    }
    auto panel = layout.getAs<DenseI64ArrayAttr>("histogram_panel");
    if (!panel)
      return;
    Type elem;
    for (auto *root : ops)
      root->walk([&](Operation *op) {
        if (op->getName().getStringRef() == "block.key_counts")
          elem = elemOf(op->getOperand(0).getType());
      });
    auto samples = panel[0], bins = panel[1];
    auto buffer = [&](Type t, int64_t n, bool filter = false) {
      return makeOp(filter ? "wram_buffer" : "nram_buffer", {},
                    {ptrType(b.getContext(), t, filter ? 3 : 1)},
                    {b.getNamedAttr("elements", b.getI64IntegerAttr(n))})
          ->getResult(0);
    };
    histogramRaw = buffer(elem, samples);
    histogramKeys =
        buffer(b.getI16Type(), samples * (elem.isInteger(64) ? 4 : 1));
    histogramFilters = buffer(b.getI16Type(), 64 * 32, true);
    histogramAxis = buffer(b.getI16Type(), bins * 32);
    forLoop(cI64(0), cI64(bins), cI64(1), [&](Value bin) {
      auto row = makeAddr(histogramAxis, {bin}, {cI64(32)}, 1);
      fillVec(row, cI64(32),
              b.create<arith::TruncIOp>(loc, b.getI16Type(), bin));
    });
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
  Value viewDim(Value view, int64_t axis) {
    if (auto extent = boundViewField(
            view, axis, *profile.services.get<intentdsl::CallFacts>(), false))
      return cI64(*extent);
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
    if (auto stride = boundViewField(
            view, axis, *profile.services.get<intentdsl::CallFacts>(), true))
      return cI64(*stride);
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
  Value phaseByteOffset(DenseI64ArrayAttr parts, int64_t groupedShift = 0) {
    Value gTerm = cI64((parts[0] + groupedShift) * depth);
    gTerm = b.create<arith::MulIOp>(loc, gTerm, symEB);
    Value lTerm = cI64(parts[1]);
    if (wide)
      lTerm = b.create<arith::MulIOp>(loc, lTerm, symEB);
    Value elems = b.create<arith::AddIOp>(loc, gTerm, lTerm);
    return b.create<arith::MulIOp>(loc, symP, elems);
  }
  static bool blockHasCompute(Block &body) {
    for (auto &inst : body) {
      StringRef n = inst.getName().stripDialect();
      if (n != "window" && n != "member_axis" && n != "slab_axis" &&
          n != "await" && n != "write_back" && n != "drain" && n != "complete")
        return true;
    }
    return false;
  }
  Value localVar(Type elemTy, StringRef extent) {
    return makeOp("local_var", {}, {ptrType(b.getContext(), elemTy, 2)},
                  {b.getNamedAttr("extent", b.getStringAttr(extent))})
        ->getResult(0);
  }
  Value loadAt(Value ptr, Value idx, Type resTy = {}) {
    if (auto *address = ptr.getDefiningOp();
        address && address->getName().getStringRef() == "bangc_src.address") {
      if (Value local = localBroadcasts.lookup(address->getOperand(0))) {
        SmallVector<Value> operands(address->getOperands());
        operands.front() = local;
        ptr = makeOp("address", operands, {local.getType()}, address->getAttrs())
                  ->getResult(0);
      }
    }
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

  Value strictQuotient(Value a, Value b) {
    return makeOp("div_rn", {a, b}, {a.getType()}, {}, "ieee-div-rn")
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
    if (prim == "div_rn") {
      out = strictQuotient(ops[0], ops[1]);
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
        auto cacheKey = std::make_tuple(b.getInsertionBlock(), dom, rowIdx, k);
        Value q = memberCoordCache.lookup(cacheKey);
        if (!q) {
          int64_t rank = cast<MemRefType>(dom.getType()).getRank();
          q = rowIdx;
          for (int64_t d = k + 1; d < rank - 1; ++d)
            q = b.create<arith::DivSIOp>(loc, q, viewDim(dom, d));
          if (k > 0)
            q = b.create<arith::RemSIOp>(loc, q, viewDim(dom, k));
          memberCoordCache[cacheKey] = q;
        }
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
        auto cacheKey =
            std::make_tuple(b.getInsertionBlock(), dom, colPlusK, axis);
        idx = memberCoordCache.lookup(cacheKey);
        if (!idx) {
          int64_t rank = cast<MemRefType>(dom.getType()).getRank();
          idx = colPlusK;
          for (int64_t dim = axis + 1; dim < rank; ++dim)
            if (dim != owner)
              idx = b.create<arith::DivSIOp>(loc, idx, viewDim(dom, dim));
          bool isOutermostNonOwner = true;
          for (int64_t dim = 0; dim < axis; ++dim)
            if (dim != owner)
              isOutermostNonOwner = false;
          if (!isOutermostNonOwner)
            idx = b.create<arith::RemSIOp>(loc, idx, viewDim(dom, axis));
          memberCoordCache[cacheKey] = idx;
        }
      }
      coords.push_back(idx);
      strides.push_back(viewStride(view, pos));
    }
    return makeAddr(view, coords, strides, 0);
  }

  void emitSlabTransfer(Operation &op, unsigned first, Value view, Value local,
                        Value rowIdx, Value col, Value live, bool load,
                        bool async = false) {
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

    // For complete tiles starting on a member boundary, all source runs
    // have the same length. Lower the entire rectangle to one strided DMA.
    const auto &facts = *profile.services.get<intentdsl::CallFacts>();
    auto inner = getIdx("inner_stride"), outer = getIdx("outer_pitch");
    if (fullColumns && widthMultiple > 0 && slot >= 0 && outer >= 0) {
      auto length = boundViewField(fn.getArgument(slot), getIdx("extent_axis"),
                                   facts, false);
      auto stride = inner < 0 ? std::optional<int64_t>(0)
                              : boundViewField(view, inner, facts, true);
      auto pitch = boundViewField(view, outer, facts, true);
      if (length && *length > 0 && widthMultiple % *length == 0 && stride &&
          pitch && *pitch >= 0 && *pitch <= INT32_MAX / elemBytes &&
          *length <= INT32_MAX / elemBytes) {
        Value base = buildSlabPtr(op, first, view, rowIdx, col);
        Value runs = b.create<arith::DivSIOp>(loc, live, extent);
        if (*stride == 1 && *pitch >= *length) {
          Value repeats = b.create<arith::SubIOp>(loc, runs, cI64(1));
          if (load)
            memcpy2D(local, base, extent, extent, outerStride, repeats,
                     "GDRAM2NRAM", async);
          else
            memcpy2D(base, local, extent, outerStride, extent, repeats,
                     "NRAM2GDRAM", false);
          return;
        }
        if (*stride == 0 && load) {
          forLoop(cI64(0), runs, cI64(1), [&](Value i) {
            auto dst = makeAddr(local, {i}, {extent}, 1);
            Value index = b.create<arith::MulIOp>(loc, i, outerStride);
            fillVec(dst, extent, loadAt(base, index));
          });
          return;
        }
      }
    }

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
              memcpy1D(bufK, global, run, "GDRAM2NRAM", async);
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
                         async);
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

  // Projected slices of one public operand often differ only by a channel
  // offset. Issue one pitched DMA across those slices for each logical row.
  llvm::DenseSet<Operation *>
  issueProjectedRuns(Block &fetch, Value row0, Value col, Value slot,
                     Value count, Value live,
                     const llvm::DenseMap<Operation *, Value> &buffers) {
    struct Slice {
      Operation *copy;
      int64_t start, dst;
    };
    const auto &facts = *profile.services.get<intentdsl::CallFacts>();
    std::map<int64_t, SmallVector<Slice>> groups;
    auto coordEqual = [](Value a, Value c) {
      if (a == c)
        return true;
      auto *x = a.getDefiningOp(), *y = c.getDefiningOp();
      return x && y && x->getName().getStringRef() == "dsa.member_axis" &&
             x->getName() == y->getName() &&
             x->getAttr("axis") == y->getAttr("axis") &&
             llvm::equal(x->getOperands(), y->getOperands());
    };
    for (auto &op : fetch.without_terminator()) {
      if (op.getName().getStringRef() != "dsa.copy_in" ||
          op.hasAttr("dsa.shared"))
        continue;
      auto arg = dyn_cast<BlockArgument>(op.getOperand(0));
      auto pr = op.getAttrOfType<StringAttr>("bangc.primitive");
      if (!arg || !pr || !pr.getValue().starts_with("group_dma"))
        continue;
      unsigned i = arg.getArgNumber();
      auto src = fn.getArgAttrOfType<IntegerAttr>(i, "kir.source");
      auto map = fn.getArgAttrOfType<AffineMapAttr>(i, "kir.indexing");
      auto layout = intentdsl::projectedLayout(fn, i, facts, false);
      if (!src || !map || !layout || layout.empty() ||
          layout.asArrayRef().back() != 1 ||
          op.getNumOperands() != 2 + layout.size() || src.getInt() < 0 ||
          size_t(src.getInt()) >= facts.arguments.size())
        continue;
      auto last = dyn_cast<OpResult>(op.getOperands().back());
      if (!last ||
          last.getDefiningOp()->getName().getStringRef() != "dsa.window" ||
          last.getResultNumber() != 1)
        continue;
      auto *record = facts.arguments[src.getInt()].getAsObject();
      auto *st = record ? record->getArray("strides") : nullptr;
      if (!st || st->size() != map.getValue().getNumResults())
        continue;
      int64_t offset = 0;
      bool valid = true;
      for (auto [axis, expr] : llvm::enumerate(map.getValue().getResults())) {
        SmallVector<int64_t> coeff(map.getValue().getNumDims() + 1, 0);
        int64_t term;
        auto stride = (*st)[axis].getAsInteger();
        if (!stride || !intentdsl::coefficients(expr, coeff) ||
            __builtin_mul_overflow(coeff.back(), *stride, &term) ||
            __builtin_add_overflow(offset, term, &offset)) {
          valid = false;
          break;
        }
      }
      auto addr = op.getAttrOfType<DenseI64ArrayAttr>("bangc.offset");
      if (valid && addr && addr[1] == 0 && offset >= 0)
        groups[src.getInt()].push_back({&op, offset, addr[0]});
    }
    llvm::DenseSet<Operation *> issued;
    for (auto &[source, slices] : groups) {
      llvm::sort(slices, [](const auto &a, const auto &c) {
        return a.start < c.start;
      });
      for (size_t i = 0; i + 2 < slices.size();) {
        auto head = slices[i];
        auto *first = head.copy;
        auto arg = cast<BlockArgument>(first->getOperand(0));
        auto strides =
            intentdsl::projectedLayout(fn, arg.getArgNumber(), facts, false);
        int64_t elem =
            cast<MemRefType>(arg.getType()).getElementTypeBitWidth() / 8;
        int64_t srcStep = slices[i + 1].start - head.start,
                dstStep = slices[i + 1].dst - head.dst;
        if (!elem || srcStep <= 0 || dstStep <= 0 || dstStep % elem ||
            srcStep > INT32_MAX / elem) {
          ++i;
          continue;
        }
        size_t j = i + 1;
        for (; j < slices.size(); ++j) {
          auto *next = slices[j].copy;
          auto v = cast<BlockArgument>(next->getOperand(0));
          if (slices[j].start - slices[j - 1].start != srcStep ||
              slices[j].dst - slices[j - 1].dst != dstStep ||
              arg.getType() != v.getType() ||
              intentdsl::projectedLayout(fn, v.getArgNumber(), facts, false) !=
                  strides ||
              first->getAttr("bangc.primitive") !=
                  next->getAttr("bangc.primitive") ||
              !std::equal(first->operand_begin() + 1, first->operand_end(),
                          next->operand_begin() + 1, next->operand_end(),
                          coordEqual))
            break;
        }
        if (j - i < 3) {
          ++i;
          continue;
        }
        auto dPitch = b.create<arith::MulIOp>(
            loc, symP,
            b.create<arith::MulIOp>(loc, symEB, cI64(depth * dstStep / elem)));
        auto target = buffers.lookup(first);
        forLoop(cI64(0), count, cI64(1), [&](Value member) {
          Value ignored;
          auto row = b.create<arith::AddIOp>(loc, row0, member);
          Value global =
              buildRowPtr(*first, 2, mapped.lookup(arg), row, col, ignored);
          auto localRow = b.create<arith::AddIOp>(
              loc, b.create<arith::MulIOp>(loc, slot, symEB), member);
          Value dest = makeAddr(target, {localRow}, {symP}, 1);
          memcpy2D(dest, global, live, dPitch, cI64(srcStep), cI64(j - i - 1),
                   "GDRAM2NRAM",
                   first->getAttrOfType<StringAttr>("bangc.primitive")
                       .getValue()
                       .ends_with("_async"));
        });
        for (size_t k = i; k < j; ++k)
          issued.insert(slices[k].copy);
        i = j;
      }
    }
    return issued;
  }

  void emitIssue(Block &fetch, Value iterPos, Value slot, Value outerGroup,
                 const llvm::DenseMap<Operation *, Value> &copyBuffers) {
    auto [row0, col] = phaseCoords(iterPos, outerGroup);
    Value remN = b.create<arith::SubIOp>(loc, symN, col);
    Value live = b.create<arith::MinSIOp>(loc, remN, symW);
    if (fullColumns)
      live = symW;
    Value hasLive =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, live, cI64(0));
    ifThen(hasLive, [&]() {
      Value remM = b.create<arith::SubIOp>(loc, symM, row0);
      Value count = b.create<arith::MinSIOp>(
          loc, b.create<arith::MaxSIOp>(loc, remM, cI64(0)), symEB);
      if (fullMembers)
        count = symEB;
      auto issued =
          issueProjectedRuns(fetch, row0, col, slot, count, live, copyBuffers);
      for (auto &op : fetch.without_terminator()) {
        if (issued.contains(&op))
          continue;
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
                Value aligned =
                    llvm::isPowerOf2_64(align)
                        ? b.create<arith::AndIOp>(loc, raw, cI64(-align))
                              .getResult()
                        : b.create<arith::MulIOp>(
                               loc,
                               b.create<arith::DivSIOp>(loc, raw, cI64(align)),
                               cI64(align))
                              .getResult();
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

        if (prim.starts_with("segmented_slab_read")) {
          forLoop(cI64(0), count, cI64(1), [&](Value j) {
            Value rIdx = b.create<arith::AddIOp>(loc, row0, j);
            Value slotRow = b.create<arith::AddIOp>(
                loc, b.create<arith::MulIOp>(loc, slot, symEB), j);
            Value dPtr = makeAddr(dstBase, {slotRow}, {symP}, 1);
            emitSlabTransfer(op, 2, src, dPtr, rIdx, col, live, true,
                             async);
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
          Value baseCoord =
              axis == 0 ? row0 : b.create<arith::RemSIOp>(loc, row0, dim);
          Value wrap = b.create<arith::AddIOp>(loc, baseCoord, count);
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
                            const llvm::DenseMap<Operation *, Value> &copyBuffers,
                            const llvm::DenseMap<Operation *, Value> &writeBuffers,
                            int64_t groupedShift = 0) {
    auto [row0, col] = phaseCoords(iterPos, outerGroup);
    Value remN = b.create<arith::SubIOp>(loc, symN, col);
    Value live = b.create<arith::MinSIOp>(
        loc, b.create<arith::MaxSIOp>(loc, remN, cI64(0)), symW);
    Value remM = b.create<arith::SubIOp>(loc, symM, row0);
    Value count = b.create<arith::MinSIOp>(
        loc, b.create<arith::MaxSIOp>(loc, remM, cI64(0)), symEB);
    if (fullColumns)
      live = symW;
    if (fullMembers)
      count = symEB;
    if (ticketCounts) {
      Operation *atom = nullptr, *copy = nullptr;
      for (auto &inst : body)
        if (inst.getName().getStringRef() == "block.atomic_add_at")
          atom = &inst;
      for (auto &inst : fetch)
        if (inst.getName().getStringRef() == "dsa.copy_in")
          copy = &inst;
      if (!atom || !copy)
        return failure();
      auto start = b.create<arith::MulIOp>(loc, slot, symEB);
      auto keys = makeAddr(copyBuffers.lookup(copy), {start}, {symP}, 1);
      makeOp("reserve_tickets",
             {mapped.lookup(atom->getOperand(0)), keys, ticketCounts,
              ticketBases, ticketValues, count, live, symP,
              cI64(layout.getAs<DenseI64ArrayAttr>("ticket_panel")[0]),
              b.create<arith::ConstantIntOp>(loc, !atom->use_empty(), 1)},
             {}, {}, "key-reservation");
    }
    bool bodyHasCompute = blockHasCompute(body);
    int64_t numRowSums = 0;
    for (auto &inst : body) {
      if (inst.getName().stripDialect() == "reduce" &&
          inst.getAttrOfType<StringAttr>("bangc.primitive") == "row_sum")
        ++numRowSums;
    }
    int64_t maxStageSlots = minPitch / 32;

    auto emitBodyMembers = [&]() -> LogicalResult {
      Value rowLanes = b.create<arith::AndIOp>(
          loc, b.create<arith::AddIOp>(loc, live, cI64(63)), cI64(-64));
      bool hasTransposeStore = false, needsFullPitch = false;
      for (auto &inst : body) {
        auto pAttr = inst.getAttrOfType<StringAttr>("bangc.primitive");
        StringRef pr = pAttr ? pAttr.getValue() : StringRef{};
        hasTransposeStore |= pr.starts_with("transpose_store");
        StringRef cn = inst.getName().stripDialect();
        needsFullPitch |= cn == "reduce" || cn == "exp" || cn == "div" ||
                          cn == "sqrt" || pr == "bf16_product" ||
                          pr == "bf16_round";
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
        if (filterNonzeroInput(&inst)) {
          auto *cast = inst.getOperand(0).getDefiningOp();
          skipOps.insert(cast);
          skipOps.insert(cast->getOperand(0).getDefiningOp());
        }
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

        llvm::DenseMap<Operation *, Value> directWriteDst;
        llvm::DenseMap<Operation *, Value> writeStageSlots;
        for (auto &inst : body) {
          if (inst.getName().getStringRef() != "dsa.write_back")
            continue;
          auto asyncAttr = inst.getAttrOfType<BoolAttr>("async");
          if (!asyncAttr || !asyncAttr.getValue())
            continue;
          Value wBase = writeBuffers.lookup(&inst);
          Value slotRow = b.create<arith::AddIOp>(
              loc, b.create<arith::MulIOp>(loc, slot, symEB), j);
          Value stagePtr = makeAddr(wBase, {slotRow}, {symP}, 1);
          writeStageSlots[&inst] = stagePtr;
          StringRef wPrim =
              inst.getAttrOfType<StringAttr>("bangc.primitive").getValue();
          if (wPrim.starts_with("transpose_store"))
            continue;
          Operation *srcDef = inst.getOperand(1).getDefiningOp();
          if (srcDef && srcDef->getBlock() == &body &&
              srcDef->getName().stripDialect() != "await" &&
              rankOf(srcDef->getResult(0).getType()) == 2 &&
              !directWriteDst.count(srcDef)) {
            directWriteDst[srcDef] = stagePtr;
          }
        }

        struct DeferredWrite {
          Operation *inst;
          Value ptr;
          Value src;
        };
        SmallVector<DeferredWrite> deferredAsyncWrites;
        auto emitWriteBack = [&](Operation &inst, Value ptr, Value src,
                                 bool async) {
          StringRef prim =
              inst.getAttrOfType<StringAttr>("bangc.primitive").getValue();
          int64_t elemBytes = llvm::divideCeil(
              cast<MemRefType>(ptr.getType()).getElementTypeBitWidth(), 8u);
          if (prim == "segmented_slab_write") {
            Value rIdx = b.create<arith::AddIOp>(loc, row0, j);
            emitSlabTransfer(inst, 3, ptr, src, rIdx, col, live, false);
            return;
          }
          auto emitMemberStores = [&]() {
            forLoop(cI64(0), members, cI64(1), [&](Value h) {
              Value rIdx = b.create<arith::AddIOp>(
                  loc, row0, b.create<arith::AddIOp>(loc, j, h));
              Value colStride;
              Value outPtr = buildRowPtr(inst, 3, ptr, rIdx, col, colStride);
              Value vPtr = makeAddr(src, {h}, {symP}, 1);
              Value contig = b.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::eq, colStride, cI64(1));
              ifThenElse(
                  contig,
                  [&]() { memcpy1D(outPtr, vPtr, live, "NRAM2GDRAM", async); },
                  [&]() {
                    forLoop(cI64(0), live, cI64(1), [&](Value k) {
                      Value off = b.create<arith::MulIOp>(loc, k, colStride);
                      storeAt(outPtr, off, loadAt(vPtr, k));
                    });
                  });
            });
          };
          if (packed && multiMember &&
              (prim == "transpose_store" || prim == "transpose_store_async")) {
            Type elemTy = cast<MemRefType>(ptr.getType()).getElementType();
            unsigned bits = elemTy.getIntOrFloatBitWidth();
            if (bits == 16 || bits == 32) {
              Value outS0 = viewStride(ptr, 0), outS1 = viewStride(ptr, 1);
              Value rIdx0 = b.create<arith::AddIOp>(loc, row0, j);
              Value outBase = makeAddr(ptr, {col, rIdx0}, {outS0, outS1}, 0);
              Value ebAligned = b.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::eq,
                  b.create<arith::AndIOp>(loc, symEB, cI64(31)), cI64(0));
              Value hasRows = b.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::sgt, members, cI64(0));
              Value hasCols = b.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::sgt, live, cI64(0));
              Value unitInner = b.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::eq, outS1, cI64(1));
              Value validPitch = b.create<arith::AndIOp>(
                  loc,
                  b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, outS0,
                                          members),
                  b.create<arith::CmpIOp>(
                      loc, arith::CmpIPredicate::sle, outS0,
                      cI64(2147483647LL / std::max<int64_t>(1, elemBytes))));
              Value canTrans = b.create<arith::AndIOp>(
                  loc, ebAligned,
                  b.create<arith::AndIOp>(
                      loc, hasRows,
                      b.create<arith::AndIOp>(
                          loc, hasCols,
                          b.create<arith::AndIOp>(loc, unitInner, validPitch))));
              Value transScratch =
                  async ? writeStageSlots.lookup(&inst)
                        : nramAt(elemTy,
                                 phaseByteOffset(
                                     inst.getAttrOfType<DenseI64ArrayAttr>(
                                         "bangc.scratch"),
                                     groupedShift));
              ifThenElse(
                  canTrans,
                  [&]() {
                    makeOp("tile_transpose", {transScratch, src, symEB, symP},
                           {}, {}, "nram-transpose");
                    if (async)
                      sync("compute");
                    Value oneCol = b.create<arith::CmpIOp>(
                        loc, arith::CmpIPredicate::eq, live, cI64(1));
                    ifThenElse(
                        oneCol,
                        [&]() {
                          memcpy1D(outBase, transScratch, members, "NRAM2GDRAM",
                                   async);
                        },
                        [&]() {
                          Value seg =
                              b.create<arith::SubIOp>(loc, live, cI64(1));
                          memcpy2D(outBase, transScratch, members, outS0, symEB,
                                   seg, "NRAM2GDRAM", async);
                        });
                  },
                  emitMemberStores);
              return;
            }
          }
          if (packed && multiMember &&
              (prim == "group_store" || prim == "group_store_async")) {
            int64_t axis = cast<MemRefType>(ptr.getType()).getRank() - 2;
            Value stride = viewStride(ptr, axis), dim = viewDim(ptr, axis);
            Value rIdx0 = b.create<arith::AddIOp>(loc, row0, j);
            Value colStride;
            Value outBase = buildRowPtr(inst, 3, ptr, rIdx0, col, colStride);
            Value c1 = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                               members, cI64(1));
            Value c2 = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                               colStride, cI64(1));
            Value c3 = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                               stride, live);
            Value c4 = b.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::sle, stride,
                cI64(2147483647LL / std::max<int64_t>(1, elemBytes)));
            Value baseCoord =
                axis == 0 ? rIdx0 : b.create<arith::RemSIOp>(loc, rIdx0, dim);
            Value wrap = b.create<arith::AddIOp>(loc, baseCoord, members);
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
                                              stride, live),
                      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                              live, symP));
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
            return;
          }
          emitMemberStores();
        };

        struct StagedRowSum {
          Value resVal;
          Value outSlice;
          Value slotScratch;
          Value inputPtr;
          Attribute inputOffsetAttr;
        };
        SmallVector<StagedRowSum, 16> stagedSums;
        SmallVector<Operation *, 16> deferredRank1Ops;
        auto emitRank1Expr = [&](Operation &r1Inst) {
          StringRef r1Name = r1Inst.getName().stripDialect();
          StringRef r1Prim =
              r1Inst.getAttrOfType<StringAttr>("bangc.primitive").getValue();
          Value r1Res = r1Inst.getResult(0);
          Type r1Elem = elemOf(r1Res.getType());
          Value arr = localVar(r1Elem, "EB");
          forLoop(cI64(0), members, cI64(1), [&](Value h) {
            Value jh = b.create<arith::AddIOp>(loc, j, h);
            SmallVector<Value> scOps;
            for (Value opnd : r1Inst.getOperands()) {
              Value mv = getVal(opnd);
              scOps.push_back(rankOf(opnd.getType()) == 1 ? loadAt(mv, jh)
                                                          : mv);
            }
            Value outSc;
            if (failed(evalScalarExpr(
                    r1Name, r1Prim, scOps, r1Elem,
                    r1Inst.getAttrOfType<TypedAttr>("value"), outSc))) {
              bodyStatus = failure();
              return;
            }
            storeAt(arr, jh, outSc);
          });
          localMap[r1Res] = arr;
        };
        auto flushStagedSums = [&]() {
          if (stagedSums.empty())
            return;
          sync("compute");
          for (auto &sr : stagedSums) {
            makeOp("row_reduce",
                   {sr.outSlice, sr.slotScratch, sr.inputPtr, members, symP,
                    live},
                   {},
                   {b.getNamedAttr("algebra", b.getStringAttr("sum")),
                    b.getNamedAttr("mode", b.getStringAttr("finish"))},
                   "row-reduce");
          }
          stagedSums.clear();
          for (Operation *defInst : deferredRank1Ops) {
            if (failed(bodyStatus))
              break;
            emitRank1Expr(*defInst);
          }
          deferredRank1Ops.clear();
        };

        for (auto &inst : body) {
          if (failed(bodyStatus))
            break;
          if (skipOps.contains(&inst))
            continue;
          StringRef n = inst.getName().stripDialect();
          if (n == "window" || n == "member_axis" || n == "slab_axis")
            continue;
          if (!stagedSums.empty()) {
            if (inst.getNumResults() == 1 &&
                rankOf(inst.getResult(0).getType()) == 1 &&
                (n == "add" || n == "sub" || n == "mul")) {
              deferredRank1Ops.push_back(&inst);
              continue;
            }
            bool needFlush = false;
            for (Value opnd : inst.getOperands()) {
              for (auto &sr : stagedSums)
                if (opnd == sr.resVal)
                  needFlush = true;
              for (Operation *defOp : deferredRank1Ops)
                if (opnd == defOp->getResult(0))
                  needFlush = true;
            }
            if (!needFlush) {
              if (n == "await") {
                needFlush = false;
              } else if (n == "reduce" &&
                         inst.getAttrOfType<StringAttr>("bangc.primitive") ==
                             "row_sum") {
                if (static_cast<int64_t>(stagedSums.size()) >= maxStageSlots)
                  needFlush = true;
              } else if ((n == "add" || n == "sub" || n == "mul" ||
                          n == "cast") &&
                         inst.getNumResults() == 1 &&
                         rankOf(inst.getResult(0).getType()) == 2) {
                StringRef pr =
                    inst.getAttrOfType<StringAttr>("bangc.primitive").getValue();
                Type dEl = elemOf(inst.getResult(0).getType());
                bool pureVec = false;
                if (n == "cast") {
                  Type sEl = elemOf(inst.getOperand(0).getType());
                  pureVec = (pr != "bf16_round" && pr != "bf16_widen" &&
                             ((dEl.isF16() && sEl.isF32()) ||
                              (dEl.isF32() && sEl.isF16())));
                } else {
                  int r0 = rankOf(inst.getOperand(0).getType());
                  int r1 = rankOf(inst.getOperand(1).getType());
                  pureVec = pr != "bf16_product" &&
                            (dEl.isF32() || dEl.isF16() || dEl.isInteger(32)) &&
                            ((r0 == 2 && r1 == 2) || (r0 == 2 && r1 == 0) ||
                             (r0 == 0 && r1 == 2) || (r0 == 2 && r1 == 1) ||
                             (r0 == 1 && r1 == 2 && (n == "add" || n == "mul")));
                }
                if (!pureVec) {
                  needFlush = true;
                } else if (!directWriteDst.count(&inst)) {
                  Attribute dstOff = inst.getAttr("bangc.offset");
                  for (auto &sr : stagedSums)
                    if (sr.inputOffsetAttr && dstOff == sr.inputOffsetAttr)
                      needFlush = true;
                }
              } else {
                needFlush = true;
              }
            }
            if (needFlush)
              flushStagedSums();
          }
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
            Value totalCnt;
            if (isFilterPrefix(inst.getOperand(6).getDefiningOp())) {
              totalCnt =
                  ifThenElse(
                      hasElem,
                      [&] {
                        fillVec(packedBuf, lanes, cI32(fillVal));
                        Value src = filterNonzeroInput(
                            inst.getOperand(6).getDefiningOp());
                        auto *op =
                            makeOp("stable_filter",
                                   {packedBuf, payload,
                                    src ? getVal(src) : prefix, prefix, live},
                                   {b.getI32Type()}, {}, "stable-filter");
                        b.create<scf::YieldOp>(loc, op->getResults());
                      },
                      [&] { b.create<scf::YieldOp>(loc, ValueRange{cI32(0)}); },
                      false)
                      .getResult(0);
            } else {
              auto cntIf = ifThenElse(
                  hasElem,
                  [&]() {
                    Value last = b.create<arith::SubIOp>(loc, live, cI64(1));
                    b.create<scf::YieldOp>(loc,
                                           ValueRange{loadAt(prefix, last)});
                  },
                  [&]() { b.create<scf::YieldOp>(loc, ValueRange{cI32(0)}); },
                  false);
              totalCnt = cntIf.getResult(0);
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
            }
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
            if (!deferredAsyncWrites.empty()) {
              if (bodyHasCompute)
                sync("compute");
              for (auto &dw : deferredAsyncWrites)
                emitWriteBack(*dw.inst, dw.ptr, dw.src, true);
              deferredAsyncWrites.clear();
            }
            continue;
          }
          if (n == "write_back") {
            Value ptr = getVal(inst.getOperand(0));
            Value src = getVal(inst.getOperand(1));
            bool async = inst.getAttrOfType<BoolAttr>("async").getValue();
            if (async)
              deferredAsyncWrites.push_back({&inst, ptr, src});
            else
              emitWriteBack(inst, ptr, src, false);
            continue;
          }

          Value resVal = inst.getResult(0);
          if (prim == "welford_f32") {
            Value scratch = nramAt(
                b.getF32Type(),
                phaseByteOffset(
                    inst.getAttrOfType<DenseI64ArrayAttr>("bangc.scratch"),
                    groupedShift));
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
            Value srcMom = getVal(inst.getOperand(0));
            if (srcMom.getType() == momentArrayTuple(b.getContext())) {
              if (rankOf(resVal.getType()) == 1) {
                Value arr = localVar(fieldTy, "EB");
                forLoop(cI64(0), members, cI64(1), [&](Value h) {
                  Value jh = b.create<arith::AddIOp>(loc, j, h);
                  Value m = makeOp("moment_get", {srcMom, jh},
                                   {momentTuple(b.getContext())})
                                ->getResult(0);
                  Value fv =
                      makeOp("summary_field", {m}, {fieldTy},
                             {b.getNamedAttr("position",
                                             b.getI64IntegerAttr(pos))})
                          ->getResult(0);
                  storeAt(arr, jh, fv);
                });
                localMap[resVal] = arr;
              } else {
                Value m = makeOp("moment_get", {srcMom, j},
                                 {momentTuple(b.getContext())})
                              ->getResult(0);
                localMap[resVal] =
                    makeOp("summary_field", {m}, {fieldTy},
                           {b.getNamedAttr("position",
                                           b.getI64IntegerAttr(pos))})
                        ->getResult(0);
              }
              continue;
            }
            Value fVal =
                makeOp("summary_field", {srcMom}, {fieldTy},
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
                    inst.getAttrOfType<DenseI64ArrayAttr>("bangc.offset"),
                    groupedShift));
            StringRef alg = prim == "row_sum" ? "sum" : "max_num";
            Value inPtr = getVal(inst.getOperand(0));
            if (alg == "sum" && numRowSums >= 2 && maxStageSlots >= 2) {
              int64_t slotIdx = static_cast<int64_t>(stagedSums.size());
              Value slotScratch =
                  slotIdx == 0
                      ? scratch
                      : makeAddr(
                            scratch,
                            {b.create<arith::MulIOp>(loc, members,
                                                     cI64(slotIdx * 32))},
                            {cI64(1)}, 1);
              makeOp("row_reduce",
                     {outSlice, slotScratch, inPtr, members, symP, live}, {},
                     {b.getNamedAttr("algebra", b.getStringAttr("sum")),
                      b.getNamedAttr("mode", b.getStringAttr("stage"))},
                     "row-reduce");
              Attribute inOffAttr;
              if (Operation *inDef = inst.getOperand(0).getDefiningOp())
                inOffAttr = inDef->getAttr("bangc.offset");
              stagedSums.push_back(
                  {resVal, outSlice, slotScratch, inPtr, inOffAttr});
              localMap[resVal] = outArr;
              continue;
            }
            makeOp("row_reduce",
                   {outSlice, scratch, inPtr, members, symP, live}, {},
                   {b.getNamedAttr("algebra", b.getStringAttr(alg))},
                   "row-reduce");
            localMap[resVal] = outArr;
            continue;
          }

          int resRank = rankOf(resVal.getType());
          Type dstElem = elemOf(resVal.getType());
          if (resRank == 2) {
            Value dst =
                directWriteDst.count(&inst)
                    ? directWriteDst.lookup(&inst)
                    : nramAt(dstElem,
                             phaseByteOffset(
                                 inst.getAttrOfType<DenseI64ArrayAttr>(
                                     "bangc.offset"),
                                 groupedShift));
            localMap[resVal] = dst;

            if (n == "gt" && rankOf(inst.getOperand(0).getType()) == 2 &&
                elemOf(inst.getOperand(0).getType()).isF32() &&
                inst.getOperand(1).getType().isF32()) {
              auto off = inst.getAttrOfType<DenseI64ArrayAttr>("bangc.scratch");
              Value mask =
                  nramAt(b.getI32Type(), phaseByteOffset(off, groupedShift));
              Value scratch = nramAt(
                  b.getI32Type(),
                  phaseByteOffset(b.getDenseI64ArrayAttr({off[0], off[1] + 4}),
                                  groupedShift));
              makeOp("vec_cmpf_mask",
                     {mask, scratch, getVal(inst.getOperand(0)),
                      getVal(inst.getOperand(1)), lanes},
                     {},
                     {b.getNamedAttr("pred_code", b.getI32IntegerAttr(2)),
                      b.getNamedAttr("rhs_is_vec", b.getBoolAttr(false))},
                     "ieee-vec-cmpf");
              makeOp("vec_scalar", {mask, mask, cI32(-1), lanes}, {},
                     {b.getNamedAttr("op", b.getStringAttr("mul"))});
              makeOp("vec_cast", {dst, mask, lanes}, {},
                     {b.getNamedAttr("kind", b.getStringAttr("i32_to_u8"))});
              continue;
            }
            if (n == "select" && dstElem.isF16() &&
                rankOf(inst.getOperand(0).getType()) == 2 &&
                rankOf(inst.getOperand(1).getType()) == 2 &&
                inst.getOperand(2).getType().isF16()) {
              auto *def = inst.getOperand(2).getDefiningOp();
              auto zero =
                  def ? def->getAttrOfType<FloatAttr>("value") : FloatAttr{};
              if (zero && zero.getValue().isPosZero()) {
                Value scratch = nramAt(
                    b.getI16Type(),
                    phaseByteOffset(
                        inst.getAttrOfType<DenseI64ArrayAttr>("bangc.scratch"),
                        groupedShift));
                makeOp("mask_zero_b16",
                       {dst, scratch, getVal(inst.getOperand(0)),
                        getVal(inst.getOperand(1)), lanes});
                continue;
              }
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
              if (ticketCounts) {
                localMap[resVal] = makeAddr(ticketValues, {j}, {live}, 1);
                continue;
              }
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
                makeOp("atomic_add", {oldBuf, gAddr, deltaBuf, cI64(1)}, {}, {},
                       "native-row-atomic-add");
                sync("all");
                storeAt(dst, k, loadAt(oldBuf, cI64(0)));
              });
              continue;
            }

            if (prim == "member_tile") {
              makeOp("member_ramp", {dst, col, live}, {}, {}, "member-ramp");
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
                if (histogramRaw) {
                  Value take = b.create<arith::MaxSIOp>(
                      loc, cI64(0), b.create<arith::SubIOp>(loc, end, first));
                  ifThen(b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                                 take, cI64(0)),
                         [&] {
                           memcpy1D(histogramRaw,
                                    makeAddr(ptr, {first}, {cI64(1)}, 0), take,
                                    "GDRAM2NRAM", false);
                           makeOp("key_histogram",
                                  {dst, histogramRaw, histogramKeys,
                                   histogramFilters, histogramAxis, take,
                                   cI64(layout.getAs<DenseI64ArrayAttr>(
                                       "histogram_panel")[1])},
                                  {}, {}, "wram-key-count");
                         });
                  continue;
                }
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

            if (segmentTable && prim == "ordered_members") {
              auto a = indexedProduct(&inst);
              auto ptr =
                  makeAddr(getVal(inst.getOperand(0)), {col}, {cI64(1)}, 0);
              makeOp(
                  "segment_product",
                  {dst, ptr, getVal(a->weights), getVal(a->indices),
                   segmentTable, segmentOffsets, segmentScratch, live,
                   cI64(layout.getAs<DenseI64ArrayAttr>("segment_gather")[2]),
                   cI64(llvm::alignTo(
                       layout.getAs<DenseI64ArrayAttr>("segment_gather")[0],
                       64))},
                  {}, {}, "segmented-indexed-product");
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
              if (isFilterPrefix(&inst)) {
                if (!filterNonzeroInput(&inst))
                  memcpy1D(dst, getVal(inst.getOperand(0)), live, "NRAM2NRAM",
                           false);
                continue;
              }
              Value scratch = nramAt(
                  dstElem,
                  phaseByteOffset(
                      inst.getAttrOfType<DenseI64ArrayAttr>(
                          "bangc.scan_scratch"),
                      groupedShift));
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
              if (prim == "bf16_round") {
                Value scratch = nramAt(
                    b.getBF16Type(),
                    phaseByteOffset(
                        inst.getAttrOfType<DenseI64ArrayAttr>("bangc.scratch"),
                        groupedShift));
                Value oneRow = b.create<arith::CmpIOp>(
                    loc, arith::CmpIPredicate::eq, members, cI64(1));
                Value activeElems = b.create<arith::SelectOp>(
                    loc, oneRow, live,
                    b.create<arith::MulIOp>(loc, members, symP));
                makeOp("bf16_tile",
                       {dst, getVal(inst.getOperand(0)), scratch, activeElems,
                        lanes},
                       {}, {b.getNamedAttr("mode", b.getStringAttr("narrow"))},
                       "bf16-tile-convert");
              } else {
                makeOp("bf16_tile",
                       {dst, getVal(inst.getOperand(0)), live, lanes}, {},
                       {b.getNamedAttr("mode", b.getStringAttr("widen"))},
                       "bf16-tile-convert");
              }
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
                if (storeTy.isInteger(32)) {
                  makeOp("member_ramp", {dstH, rowBase, live}, {}, {},
                         "member-ramp");
                  return;
                }
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
            if (n == "cast" && in0Elem.isInteger(32) && dstElem.isF32() &&
                rankOf(inst.getOperand(0).getType()) == 2) {
              makeOp("vec_cast", {dst, getVal(inst.getOperand(0)), lanes}, {},
                     {b.getNamedAttr("kind", b.getStringAttr("i32_to_f32"))});
              continue;
            }
            if ((n == "xor" || n == "and" || n == "shl" || n == "shr") &&
                dstElem.isInteger(32) &&
                rankOf(inst.getOperand(0).getType()) == 2) {
              int rhs = rankOf(inst.getOperand(1).getType());
              if ((n == "xor" || n == "and") && (rhs == 0 || rhs == 2)) {
                makeOp(
                    rhs ? "vec_binary" : "vec_scalar",
                    {dst, getVal(inst.getOperand(0)),
                     getVal(inst.getOperand(1)), lanes},
                    {},
                    {b.getNamedAttr(
                        "op", b.getStringAttr(n == "xor" ? "bxor" : "band"))});
                continue;
              }
              if ((n == "shr" || n == "shl") && rhs == 0) {
                makeOp(
                    "vec_scalar",
                    {dst, getVal(inst.getOperand(0)),
                     getVal(inst.getOperand(1)), lanes},
                    {},
                    {b.getNamedAttr(
                        "op", b.getStringAttr(n == "shr" ? "ashr" : "shl"))});
                continue;
              }
            }
            if (n == "cast" && rankOf(inst.getOperand(0).getType()) == 2 &&
                ((dstElem.isF16() && in0Elem.isF32()) ||
                 (dstElem.isF32() && in0Elem.isF16()))) {
              bool flushSignedZero =
                  compilerinfra::allowsOutputSubnormalFlush(&inst) &&
                  compilerinfra::hasFastMath(&inst, arith::FastMathFlags::nsz);
              StringRef kind = !dstElem.isF16()  ? "f16_to_f32"
                               : flushSignedZero ? "f32_to_f16"
                                                 : "f32_to_f16_ieee";
              makeOp("vec_cast", {dst, getVal(inst.getOperand(0)), lanes}, {},
                     {b.getNamedAttr("kind", b.getStringAttr(kind))});
              continue;
            }
            if (n == "exp" && dstElem.isF32() &&
                rankOf(inst.getOperand(0).getType()) == 2) {
              Value scratch = nramAt(
                  b.getF32Type(),
                  phaseByteOffset(
                      inst.getAttrOfType<DenseI64ArrayAttr>("bangc.scratch"),
                      groupedShift));
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
              if (prim == "approximate_sqrt") {
                makeOp("vec_sqrt", {dst, getVal(inst.getOperand(0)), lanes}, {},
                       {b.getNamedAttr(
                            "fastmath",
                            arith::FastMathFlagsAttr::get(
                                b.getContext(),
                                compilerinfra::effectiveFastMath(&inst))),
                        b.getNamedAttr(
                            "denormal-fp-math",
                            b.getStringAttr(
                                intentdsl::kir::getDenormalFpMath(&inst)))},
                       "approximate-sqrt");
                continue;
              }
              Value scratch = nramAt(
                  b.getF32Type(),
                  phaseByteOffset(
                      inst.getAttrOfType<DenseI64ArrayAttr>("bangc.scratch"),
                      groupedShift));
              Value oneRow = b.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::eq, members, cI64(1));
              Value activeElems = b.create<arith::SelectOp>(
                  loc, oneRow, live,
                  b.create<arith::MulIOp>(loc, members, symP));
              makeOp("vec_sqrt_rn",
                     {dst, getVal(inst.getOperand(0)), scratch, activeElems,
                      lanes},
                     {}, {}, "ieee-sqrt-rn");
              continue;
            }
            if (prim == "bf16_product" &&
                rankOf(inst.getOperand(0).getType()) == 2 &&
                rankOf(inst.getOperand(1).getType()) == 2) {
              Value scratch = nramAt(
                  b.getF32Type(),
                  phaseByteOffset(
                      inst.getAttrOfType<DenseI64ArrayAttr>("bangc.scratch"),
                      groupedShift));
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
            if (n == "div" && dstElem.isF32()) {
              int r0 = rankOf(inst.getOperand(0).getType());
              int r1 = rankOf(inst.getOperand(1).getType());
              if (r0 == 2 && r1 == 0) {
                auto *input = inst.getOperand(0).getDefiningOp();
                auto *den = inst.getOperand(1).getDefiningOp();
                auto fa =
                    den ? den->getAttrOfType<FloatAttr>("value") : FloatAttr{};
                if (input && input->getName().getStringRef() == "block.cast" &&
                    elemOf(input->getOperand(0).getType()).isInteger(32) &&
                    fa) {
                  int e;
                  float f = fa.getValueAsDouble();
                  if (std::frexp(f, &e) == 0.5f && e >= -94 && e <= 126) {
                    makeOp("vec_scalar",
                           {dst, getVal(inst.getOperand(0)), cF32(1.0f / f),
                            lanes},
                           {}, {b.getNamedAttr("op", b.getStringAttr("mul"))});
                    continue;
                  }
                }
              }
              bool approximate = prim == "approximate_div";
              if (!approximate && prim != "guarded_reciprocal" && prim != "div_rn") {
                bodyStatus = inst.emitError(
                    "missing resolved f32 quotient implementation");
                return;
              }
              Value a = getVal(inst.getOperand(0)),
                    divisor = getVal(inst.getOperand(1));
              StringRef mode = r0 != 2 ? "sv" : r1 != 2 ? "vs" : "vv";
              auto divide = [&](Value output, Value x, Value y, Value active, Value length) {
                if (approximate) {
                  Value scratch = output;
                  if (mode == "vv")
                    scratch = nramAt(
                        b.getF32Type(),
                        phaseByteOffset(inst.getAttrOfType<DenseI64ArrayAttr>(
                                            "bangc.scratch"),
                                        groupedShift));
                  makeOp("vec_div_approx", {output, x, y, scratch, length}, {},
                         {b.getNamedAttr(
                              "fastmath",
                              arith::FastMathFlagsAttr::get(
                                  b.getContext(),
                                  compilerinfra::effectiveFastMath(&inst))),
                          b.getNamedAttr(
                              "denormal-fp-math",
                              b.getStringAttr(
                                  intentdsl::kir::getDenormalFpMath(&inst)))},
                         "reciprocal-product");
                  return;
                }
                Value scratch = nramAt(b.getF32Type(), phaseByteOffset(
                    inst.getAttrOfType<DenseI64ArrayAttr>("bangc.scratch"), groupedShift));
                if (prim == "guarded_reciprocal") {
                  makeOp("vec_guarded_div", {output, x, y, scratch, active, length}, {},
                         {b.getNamedAttr("fastmath", arith::FastMathFlagsAttr::get(
                              b.getContext(), compilerinfra::effectiveFastMath(&inst))),
                          b.getNamedAttr("denormal-fp-math", b.getStringAttr(intentdsl::kir::getDenormalFpMath(&inst)))},
                         "guarded-reciprocal-product");
                  return;
                }
                makeOp("vec_div_rn", {output, x, y, scratch, active, length}, {},
                       {b.getNamedAttr("mode", b.getStringAttr(mode))},
                       ("ieee-div-rn-" + mode).str());
              };
              if (r0 != 1 && r1 != 1) {
                Value active = b.create<arith::SelectOp>(loc,
                    b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, members, cI64(1)),
                    live, b.create<arith::MulIOp>(loc, members, symP));
                divide(dst, a, divisor, active, lanes);
              } else {
                forLoop(cI64(0), members, cI64(1), [&](Value h) {
                  Value row = b.create<arith::AddIOp>(loc, j, h);
                  auto input = [&](Value v, int rank) {
                    return rank == 2   ? makeAddr(v, {h}, {symP}, 1)
                           : rank == 1 ? loadAt(v, row)
                                       : v;
                  };
                  divide(makeAddr(dst, {h}, {symP}, 1), input(a, r0),
                         input(divisor, r1), live, rowLanes);
                });
              }
              continue;
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
        flushStagedSums();
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

  static bool sameCopyInSpec(Operation &a, Operation &b) {
    if (a.getOperand(0) != b.getOperand(0) ||
        a.getNumOperands() != b.getNumOperands() ||
        a.getAttr("bangc.primitive") != b.getAttr("bangc.primitive") ||
        a.hasAttr("dsa.shared") || b.hasAttr("dsa.shared"))
      return false;
    for (unsigned i = 2; i < a.getNumOperands(); ++i) {
      auto ra = cast<OpResult>(a.getOperand(i));
      auto rb = cast<OpResult>(b.getOperand(i));
      Operation *da = ra.getOwner(), *db = rb.getOwner();
      if (da->getName() != db->getName() ||
          ra.getResultNumber() != rb.getResultNumber())
        return false;
      StringRef dn = da->getName().getStringRef();
      if (dn == "dsa.lookup_axis")
        return false;
      if ((dn == "dsa.member_axis" || dn == "dsa.slab_axis") &&
          (da->getOperand(1) != db->getOperand(1) ||
           da->getAttr("axis") != db->getAttr("axis")))
        return false;
    }
    return true;
  }

  static bool isPureReductionPhase(Operation &phaseOp, int64_t &outGrouped,
                                   int64_t &outMaxLocal) {
    if (phaseOp.getName().getStringRef() != "dsa.phase" ||
        phaseOp.getNumResults() != 1 ||
        isa<TupleType>(phaseOp.getResult(0).getType()) ||
        phaseOp.hasAttr("bangc.empty_count_slot"))
      return false;
    auto &fetch = phaseOp.getRegion(0).front();
    auto &body = phaseOp.getRegion(1).front();
    outGrouped = 0;
    for (auto &c : fetch.without_terminator()) {
      StringRef cn = c.getName().getStringRef();
      if (cn == "dsa.window" || cn == "dsa.member_axis" ||
          cn == "dsa.slab_axis")
        continue;
      if (cn != "dsa.copy_in" || c.hasAttr("dsa.shared"))
        return false;
      outGrouped += llvm::divideCeil(
          cast<MemRefType>(c.getOperand(0).getType()).getElementTypeBitWidth(),
          8u);
    }
    outMaxLocal = 0;
    for (auto &inst : body) {
      StringRef n = inst.getName().stripDialect();
      if (n == "window" || n == "member_axis" || n == "slab_axis" ||
          n == "await" || n == "complete" || n == "combine")
        continue;
      if (n == "drain") {
        if (inst.getNumOperands() != 0)
          return false;
        continue;
      }
      if (n == "reduce") {
        if (auto off = inst.getAttrOfType<DenseI64ArrayAttr>("bangc.offset"))
          outMaxLocal = std::max<int64_t>(outMaxLocal, off[1] + 4);
        continue;
      }
      if (n == "add" || n == "sub" || n == "mul" || n == "cast" ||
          n == "constant") {
        StringRef pr =
            inst.getAttrOfType<StringAttr>("bangc.primitive").getValue();
        if (pr == "bf16_product" || pr == "bf16_round")
          return false;
        if (auto off = inst.getAttrOfType<DenseI64ArrayAttr>("bangc.offset")) {
          int64_t eb = llvm::divideCeil(
              elemOf(inst.getResult(0).getType()).getIntOrFloatBitWidth(), 8u);
          outMaxLocal = std::max<int64_t>(outMaxLocal, off[1] + eb);
        }
        continue;
      }
      return false;
    }
    return outGrouped > 0;
  }

  struct FusedFollower {
    Operation *op;
    int64_t groupedShift;
    Value carrySlot;
    llvm::DenseMap<Operation *, Value> copyBuffers;
  };

  LogicalResult emitPhase(Operation &phaseOp, Value outerGroup,
                          ArrayRef<std::pair<Operation *, int64_t>> followers = {}) {
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

    SmallVector<FusedFollower, 4> fInfos;
    for (auto [fOp, gShift] : followers) {
      FusedFollower fi;
      fi.op = fOp;
      fi.groupedShift = gShift;
      fi.carrySlot = mapped.lookup(fOp->getOperand(0));
      mapped[fOp->getResult(0)] = fi.carrySlot;
      for (auto &fCopy : fOp->getRegion(0).front().without_terminator()) {
        if (fCopy.getName().getStringRef() != "dsa.copy_in")
          continue;
        for (auto &lCopy : fetch.without_terminator()) {
          if (lCopy.getName().getStringRef() == "dsa.copy_in" &&
              sameCopyInSpec(lCopy, fCopy)) {
            fi.copyBuffers[&fCopy] = copyBuffers.lookup(&lCopy);
            break;
          }
        }
      }
      fInfos.push_back(std::move(fi));
    }

    bool bodyHasCompute = blockHasCompute(body);
    bool phaseHasAsyncWrite = false;
    llvm::DenseMap<Operation *, Value> writeBuffers;
    for (auto &inst : body) {
      if (inst.getName().getStringRef() != "dsa.write_back")
        continue;
      auto asyncAttr = inst.getAttrOfType<BoolAttr>("async");
      if (!asyncAttr || !asyncAttr.getValue())
        continue;
      phaseHasAsyncWrite = true;
      Type elem = cast<MemRefType>(inst.getOperand(0).getType()).getElementType();
      writeBuffers[&inst] = nramAt(
          elem, phaseByteOffset(
                    inst.getAttrOfType<DenseI64ArrayAttr>("bangc.write_offset")));
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
          if (phaseHasAsyncWrite && bodyHasCompute && &steps == &sched)
            sync("compute");
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
          sync(phaseHasAsyncWrite ? "io" : "all");
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
          Value curPos = sMap.lookup(op.getOperand(0));
          Value curSlot = sMap.lookup(op.getOperand(1));
          if (failed(emitCompute(fetch, body, &phaseOp, curPos, curSlot,
                                 outerGroup, carrySlot, carryIsMoment, isPacked,
                                 copyBuffers, writeBuffers)))
            return failure();
          llvm::DenseMap<Operation *, Value> emptyWrites;
          for (auto &fi : fInfos) {
            bool fPacked = isPacked;
            fi.op->getRegion(2).front().walk([&](Operation *sOp) {
              if (sOp->getName().stripDialect() == "compute_members")
                fPacked = sOp->getAttrOfType<StringAttr>("bangc.primitive") ==
                          "packed_pointwise";
            });
            if (failed(emitCompute(fi.op->getRegion(0).front(),
                                   fi.op->getRegion(1).front(), fi.op, curPos,
                                   curSlot, outerGroup, fi.carrySlot, false,
                                   fPacked, fi.copyBuffers, emptyWrites,
                                   fi.groupedShift)))
              return failure();
          }
        } else if (n == "advance") {
          b.create<scf::YieldOp>(loc,
                                 ValueRange{sMap.lookup(op.getOperand(0))});
        } else if (n == "end") {
          if (phaseHasAsyncWrite && &steps == &sched)
            sync("io");
        } else {
          return failure();
        }
      }
      return success();
    };
    return emitSteps(emitSteps, sched);
  }

  LogicalResult emitOuterOps(ArrayRef<Operation *> ops, Value outerGroup) {
    // Allocate independent carry identities before their consumers. This lets
    // adjacent reductions share one fetch even when their initializers were
    // interleaved with the phases in program order.
    for (auto *op : ops) {
      if (op->getName().getStringRef() != "block.identity")
        continue;
      Value arr = localVar(b.getF32Type(), "R");
      bool sum = op->getAttrOfType<StringAttr>("algebra") == "sum";
      Value init = sum ? cF32(0.0f)
                       : cF32(-std::numeric_limits<float>::infinity());
      forLoop(cI64(0), symR, cI64(1), [&](Value j) { storeAt(arr, j, init); });
      mapped[op->getResult(0)] = arr;
    }
    llvm::DenseMap<Value, Value> rangeMap;
    for (size_t idx = 0; idx < ops.size(); ++idx) {
      Operation &op = *ops[idx];
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
        SmallVector<std::pair<Operation *, int64_t>, 4> followers;
        int64_t g0 = 0, l0 = 0;
        if (!resident && !independent && isPureReductionPhase(op, g0, l0)) {
          int64_t leaderUnits = g0 * depth + l0;
          SmallVector<Value, 4> fusedResults{op.getResult(0)};
          SmallVector<Value, 4> fusedCarries{op.getOperand(0)};
          while (idx + 1 < ops.size()) {
            Operation &cand = *ops[idx + 1];
            if (cand.getName().getStringRef() == "block.identity") {
              ++idx;
              continue;
            }
            int64_t gc = 0, lc = 0;
            if (!isPureReductionPhase(cand, gc, lc) || gc > g0)
              break;
            if (llvm::is_contained(fusedCarries, cand.getOperand(0)))
              break;
            bool usesPrior = false;
            cand.walk([&](Operation *uOp) {
              for (Value v : uOp->getOperands())
                if (llvm::is_contained(fusedResults, v))
                  usesPrior = true;
            });
            if (usesPrior)
              break;
            if ((g0 - gc) * depth + lc > leaderUnits)
              break;
            bool allCopiesMatch = true;
            for (auto &fCopy :
                 cand.getRegion(0).front().without_terminator()) {
              if (fCopy.getName().getStringRef() != "dsa.copy_in")
                continue;
              bool matched = false;
              for (auto &lCopy : op.getRegion(0).front().without_terminator()) {
                if (lCopy.getName().getStringRef() == "dsa.copy_in" &&
                    sameCopyInSpec(lCopy, fCopy)) {
                  matched = true;
                  break;
                }
              }
              if (!matched) {
                allCopiesMatch = false;
                break;
              }
            }
            if (!allCopiesMatch)
              break;
            followers.push_back({&cand, g0 - gc});
            fusedResults.push_back(cand.getResult(0));
            fusedCarries.push_back(cand.getOperand(0));
            ++idx;
          }
        }
        if (failed(emitPhase(op, outerGroup, followers)))
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
        continue;
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

    if (auto lib = layout.getAs<StringAttr>("library_impl")) {
      auto kind = layout.getAs<StringAttr>("library_kind");
      if (!kind || oldOps.size() != 1 ||
          oldOps[0]->getName().getStringRef() != "dsa.library_call")
        return fn.emitError("expected single dsa.library_call op");
      auto inMr = cast<MemRefType>(fn.getArgument(0).getType());
      Type elem = inMr.getElementType();
      int32_t elemBytes =
          static_cast<int32_t>(llvm::divideCeil(inMr.getElementTypeBitWidth(), 8u));
      int32_t dtypeCode = elem.isF16()    ? 1
                          : elem.isF32()  ? 2
                          : elem.isBF16() ? 17
                                          : 0;
      b.setInsertionPointToStart(hostBlk);
      OperationState cs(loc, "bangc_src.cnnl_row_op");
      cs.addOperands(fn.getArguments());
      cs.addAttribute("kind", kind);
      cs.addAttribute("dtype_code", b.getI32IntegerAttr(dtypeCode));
      cs.addAttribute("elem_bytes", b.getI32IntegerAttr(elemBytes));
      cs.addAttribute("ci.implementation", lib);
      if (auto fm = oldOps[0]->getAttr("fastmath"))
        cs.addAttribute("fastmath", fm);
      b.create(cs);
      oldOps[0]->erase();
      return success();
    }

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
    symM = boundRows ? cI64(boundRows) : makeSym("M");
    symN = boundColumns ? cI64(boundColumns) : makeSym("N");
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

    for (auto [arg, elements] : broadcastSpans(
             fn, *profile.services.get<intentdsl::CallFacts>(),
             *profile.services.get<intentdsl::dsa::LocalSupply>())) {
      Type elem = cast<MemRefType>(arg.getType()).getElementType();
      Value local = makeOp("nram_buffer", {}, {ptrType(b.getContext(), elem, 1)},
                           {b.getNamedAttr("elements", b.getI64IntegerAttr(elements))})
                        ->getResult(0);
      memcpy1D(local, arg, cI64(elements), "GDRAM2NRAM", true);
      localBroadcasts[arg] = local;
    }
    if (!localBroadcasts.empty())
      sync("io");

    if (auto spec = layout.getAs<DenseI64ArrayAttr>("segment_gather")) {
      auto buffer = [&](Type type, int64_t n) {
        return makeOp("nram_buffer", {}, {ptrType(b.getContext(), type, 1)},
                      {b.getNamedAttr("elements", b.getI64IntegerAttr(n))})
            ->getResult(0);
      };
      segmentTable = buffer(b.getF32Type(), llvm::alignTo(spec[0], 64));
      segmentOffsets = buffer(b.getI32Type(), spec[1] + 1);
      segmentScratch = buffer(b.getI32Type(), 3 * spec[2]);
      for (auto *root : oldOps)
        root->walk([&](Operation *op) {
          if (op->getName().getStringRef() == "block.segment_fold") {
            auto a = indexedProduct(op);
            memcpy1D(segmentTable, a->table, cI64(spec[0]), "GDRAM2NRAM",
                     false);
          }
        });
    }
    prepareHistogram(oldOps);
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
  stampScfCarryIds(m);
  llvm::SmallVector<char> metadata;
  llvm::raw_svector_ostream os(metadata);
  if (auto err = describeRowArtifact(fn, os))
    return fn.emitError() << llvm::toString(std::move(err)), failure();
  m->setAttr("bangc_src.package",
             b.getStringAttr(StringRef(metadata.data(), metadata.size())));
  return success();
}
} // namespace CI_UNIT_NAMESPACE
