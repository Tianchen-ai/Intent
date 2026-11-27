#include "Plan.h"
#include "intentdsl/levels/algorithm/kir/ir/Matrix.h"
#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
#include "intentdsl/levels/algorithm/kir/ir/Segment.h"
#include "intentdsl/levels/storage/dsa/ir/Owners.h"
#include "llvm/Support/MathExtras.h"
#include <climits>

namespace intentdsl::dsa {
using namespace mlir;

// A DMA slot holds a complete group. Vector temporaries and collective scratch
// are reused after each member's compute and drain.
struct Footprint {
  int64_t inputBytes = 0, sharedBytes = 0, outputBytes = 0, vectorBytes = 0,
          collectiveBytes = 0;
  static Footprint of(Block &body, const LocalSupply &c, bool reuse = false,
                      bool persistent = false, bool asyncWrite = false,
                      bool approximate = false) {
    auto reductionScratch=c.reductionScratchPerColumn;
    Footprint f;
    for (auto &op : body) {
      auto code = op.getName().stripDialect();
      if (code == "await" || code == "active" || code == "window" ||
          code == "member_axis" || code == "slab_axis" || code == "rows" || code == "columns")
        continue;
      if (asyncWrite && (code == "store" || code == "write_back")) {
        f.outputBytes += llvm::divideCeil(
            cast<MemRefType>(op.getOperand(0).getType()).getElementTypeBitWidth(),
            8u);
      }
      if (code == "reduce" || code == "exp") {
        f.collectiveBytes = std::max<int64_t>(f.collectiveBytes, reductionScratch);
      } else if (code == "summary_reduce") {
        f.collectiveBytes =
            std::max<int64_t>(f.collectiveBytes, 2 * reductionScratch);
      } else if (projectsTransposed2D(&op)) {
        int64_t elemBytes = llvm::divideCeil(
            cast<MemRefType>(op.getOperand(0).getType()).getElementTypeBitWidth(),
            8u);
        f.collectiveBytes = std::max<int64_t>(f.collectiveBytes, elemBytes);
      } else if (op.getNumResults() == 1) {
        if(code=="scan_sum")f.collectiveBytes=std::max<int64_t>(f.collectiveBytes,reductionScratch);
        if(code=="atomic_add_at")f.collectiveBytes=std::max<int64_t>(f.collectiveBytes,8);
        if(code=="key_counts" || code=="sum_columns")
          f.collectiveBytes=std::max<int64_t>(f.collectiveBytes,cast<MemRefType>(op.getOperand(0).getType()).getElementTypeBitWidth()/8);
        if(code=="segment_fold" && kir::contiguousFeatureRead(&op))
          f.collectiveBytes=std::max<int64_t>(f.collectiveBytes,c.featurePanelMembers*int64_t(sizeof(float)));
        auto t = dyn_cast<RankedTensorType>(op.getResult(0).getType());
        if (!t || t.getRank() < 1 || !t.getElementType().isIntOrFloat())
          continue;
        if (t.getRank() == 2 && code == "mul" && t.getElementType().isBF16())
          f.collectiveBytes =
              std::max<int64_t>(f.collectiveBytes, 2 * int64_t(sizeof(float)));
        if (t.getRank() == 2 && code == "cast" && t.getElementType().isBF16())
          f.collectiveBytes =
              std::max<int64_t>(f.collectiveBytes, int64_t(sizeof(uint16_t)));
        auto fastScratch = approximate && c.approxScratch ? c.approxScratch(&op) : std::nullopt;
        if (fastScratch) {
          f.collectiveBytes = std::max(f.collectiveBytes, *fastScratch);
        } else if (t.getRank() == 2 && (code == "sqrt" || code == "div") &&
                   t.getElementType().isF32())
          f.collectiveBytes = std::max<int64_t>(f.collectiveBytes,
                                                6 * int64_t(sizeof(float)));
        // Rank-one calculation results in the block contract are per-row
        // summaries. Projected coordinates have no local storage; copied
        // fragments do.
        if (t.getRank() == 1 && code != "load" && code != "copy_in")
          continue;
        if (code == "load" || code == "copy_in" || code=="indexed_tile_read") {
          int64_t elemBytes = llvm::divideCeil(t.getElementTypeBitWidth(), 8u);
          if (persistent && code != "indexed_tile_read" &&
              cast<MemRefType>(op.getOperand(0).getType()).getRank() == 1)
            f.sharedBytes += elemBytes;
          else
            f.inputBytes += elemBytes;
        } else
          f.vectorBytes += llvm::divideCeil(t.getElementTypeBitWidth(), 8u);
      }
    }
    if(reuse) f.vectorBytes=placeTemporaries(body).peak;
    return f;
  }
};

bool reusableInput(func::FuncOp fn){
  if(fn->getAttrOfType<StringAttr>("block.topology")!="resident")return false;
  auto abi=llvm::json::parse(fn->getAttrOfType<StringAttr>("kir.abi").getValue());
  if(!abi){llvm::consumeError(abi.takeError());return false;}
  for(auto &v:*abi->getAsArray())if(v.getAsObject()->getString("direction")=="InOut")return false;
  bool found=false;
  fn.walk([&](Operation *op){auto n=op->getName().getStringRef();
    if(n=="block.load" || n=="dsa.copy_in")found|=cast<MemRefType>(op->getOperand(0).getType()).getRank()==1;
  });return found;
}
bool batchableInstances(func::FuncOp fn) {
  if (fn->getAttrOfType<StringAttr>("block.topology") != "resident" ||
      fn->hasAttr("kir.owner_axis"))
    return false;
  bool ok = true;
  fn.walk([&](Operation *op) {
    auto ns = op->getName().getDialectNamespace();
    if (ns == "builtin" || ns == "func")
      return;
    auto code = op->getName().stripDialect();
    if (code == "reduce") {
      auto alg = op->getAttrOfType<StringAttr>("algebra");
      ok &= alg && (alg == "sum" || alg == "max_num");
      return;
    }
    ok &= llvm::is_contained(
        ArrayRef<StringRef>{
            "rows",     "columns",          "pid",          "active",
            "zero",     "load",             "store",        "window",
            "member_axis", "lookup_axis",   "indexed_tile_read",
            "copy_in",  "await",            "write_back",
            "issued",   "drain",            "complete",     "yield",
            "phase",    "range",            "slot_zero",    "iterate",
            "issue_if_in_bounds",           "wait_slot",    "next",
            "other_slot",                   "compute_members",
            "advance",  "end",              "load_shared",  "add",
            "sub",      "mul",              "div",          "exp",
            "rsqrt",    "sqrt",             "cast",         "maximum",
            "constant", "select",           "gt",           "ne",
            "and",      "xor",              "shl",          "shr",
            "linear_index",                 "uniform"},
        code);
  });
  return ok;
}
bool canAsyncWriteBack(func::FuncOp fn, int64_t depth) {
  if (depth != 2 ||
      fn->getAttrOfType<StringAttr>("block.topology") != "resident" ||
      fn->hasAttr("kir.owner_axis"))
    return false;
  auto sig = fn->getAttrOfType<StringAttr>("kir.abi");
  if (!sig)
    return false;
  auto abi = llvm::json::parse(sig.getValue());
  if (!abi) {
    llvm::consumeError(abi.takeError());
    return false;
  }
  auto *arr = abi->getAsArray();
  if (!arr)
    return false;
  for (auto &v : *arr) {
    auto *obj = v.getAsObject();
    if (!obj || obj->getString("direction") == "InOut")
      return false;
  }
  unsigned stores = 0;
  bool otherEffects = false;
  fn.walk([&](Operation *op) {
    auto code = op->getName().stripDialect();
    if (code == "store" || code == "write_back")
      ++stores;
    else if (code == "scatter_fragment" || code == "publish_offsets" ||
             code == "unique_put" || code == "unique_put_at" ||
             code == "fill_ranges_at" || code == "owner_store" ||
             code == "owner_write" || code == "atomic_add_at")
      otherEffects = true;
  });
  return stores > 0 && !otherEffects;
}
static llvm::Expected<DictionaryAttr>
planStorage(func::FuncOp fn, const LocalSupply &c, int64_t depth,
            int64_t participants, bool persistent, bool reuse, int64_t batch,
            bool approximate) {
  if (persistent && !reusableInput(fn))
    return llvm::createStringError(
        "persistent panel requires an immutable row-invariant input");
  if (batch < 1 || batch > 4 || !llvm::isPowerOf2_64(batch) ||
      (batch > 1 && !batchableInstances(fn)))
    return llvm::createStringError("invalid supply instance batch");
  if (participants <= 0 || participants > c.consumers)
    return llvm::createStringError("invalid supply work partition");
  Builder b(fn.getContext());
  bool asyncWrite = c.asynchronous && canAsyncWriteBack(fn, depth);
  // A group of equally shaped lane-wise expressions can keep all member
  // results in private storage until the final rectangular transfer.
  bool wide = fn->hasAttr("kir.iteration") &&
              fn->getAttrOfType<StringAttr>("block.topology")=="resident";
  for (auto t:fn.getArgumentTypes()) {
    auto view=dyn_cast<MemRefType>(t);
    wide &= view && view.getRank()>2 && view.getElementType().isF16();
  }
  unsigned arithmetic=0;
  bool hasUniform = false;
  bool permuted2D = false, hasCollective = false;
  fn.walk([&](Operation *op){
    auto code=op->getName().stripDialect();
    hasUniform |= code == "uniform";
    permuted2D |= projectsTransposed2D(op);
    hasCollective |= code == "reduce" || code == "summary_reduce" ||
                     code == "scan_sum";
    if(code=="add" || code=="sub" || code=="mul") {++arithmetic;return;}
    if(code=="reduce" || (op->getName().getDialectNamespace()=="block" &&
       code!="rows" && code!="columns" && code!="pid" && code!="active" &&
       code!="load" && code!="store" && code!="zero")) wide=false;
  });
  auto configs = fn->getAttrOfType<ArrayAttr>("block.configs");
  bool multiRow = false;
  if (configs)
    for (auto attr : configs)
      if (auto r = dyn_cast<DictionaryAttr>(attr))
        if (auto rows = r.getAs<IntegerAttr>("rows"))
          multiRow |= (rows.getInt() * batch) > 1;
  wide = (wide && arithmetic != 0) || batch > 1 ||
         (multiRow && batchableInstances(fn)) ||
         (permuted2D && !hasCollective);
  SmallVector<Footprint> phases;
  bool indexed=false;
  fn.walk([&](Operation *op) {
    indexed |= op->getName().getStringRef()=="block.indexed_tile_read" || op->getName().getStringRef()=="dsa.lookup_axis";
    if (op->getName().getStringRef() == "dsa.phase") {
      auto f = Footprint::of(op->getRegion(1).front(), c, reuse, persistent,
                             asyncWrite, approximate);
      auto fetchF =
          Footprint::of(op->getRegion(0).front(), c, false, persistent, false);
      f.inputBytes = fetchF.inputBytes;
      f.sharedBytes = fetchF.sharedBytes;
      phases.push_back(f);
    }
  });
  if (phases.empty() &&
      fn->getAttrOfType<StringAttr>("block.topology") == "resident")
    phases.push_back(
        Footprint::of(fn.front(), c, reuse, persistent, asyncWrite, approximate));
  else if (phases.empty())
    fn.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "block.tiles")
        phases.push_back(Footprint::of(op->getRegion(0).front(), c, reuse,
                                       persistent, asyncWrite, approximate));
    });
  if(indexed && fn->getAttrOfType<StringAttr>("block.topology")=="resident" &&
     phases.size()==1 && !phases.front().vectorBytes && !phases.front().collectiveBytes)
    wide=true;
  if (!configs || phases.empty() || (depth != 1 && depth != 2))
    return llvm::createStringError("expected a realized tile traversal");
  SmallVector<Attribute> accepted, rejected;
  for (auto attr : configs) {
    auto row = dyn_cast<DictionaryAttr>(attr);
    if (!row || !row.getAs<IntegerAttr>("id") ||
        !row.getAs<IntegerAttr>("rows") || !row.getAs<IntegerAttr>("width"))
      return llvm::createStringError("malformed source parameter row");
    auto r = row.getAs<IntegerAttr>("rows").getInt(),
         w = row.getAs<IntegerAttr>("width").getInt();
    auto effR = r * batch;
    // A pitched row must align even the narrowest supported element (f16).
    auto pitch = llvm::alignTo(w, c.alignment / 2);
    int64_t bytes = 0;
    for (auto p : phases) {
      int64_t scratchBytes = (wide ? effR : 1) * pitch * p.collectiveBytes;
      if (p.collectiveBytes)
        scratchBytes =
            std::max<int64_t>(scratchBytes, 32 * effR * int64_t(sizeof(float)));
      bytes = std::max<int64_t>(
          bytes, pitch * (depth * effR * (p.inputBytes + p.outputBytes) +
                          (wide ? effR : 1) * (p.sharedBytes + p.vectorBytes)) +
                     scratchBytes);
    }
    if (indexed)
      bytes = llvm::alignTo(bytes, c.alignment) +
              llvm::alignTo(c.indexScratchPerMember * effR, c.alignment);
    if (hasUniform && c.randomScratchBytes > 0)
      bytes = llvm::alignTo(bytes, c.alignment) +
              llvm::alignTo(c.randomScratchBytes, c.alignment);
    auto reason = (depth == 2 && !c.asynchronous)
                      ? "no participant-local DMA completion"
                  : bytes > std::min(c.budget, c.capacity)
                      ? "private staging exceeds available budget"
                      : "";
    NamedAttrList attrs;
    attrs.set("id", row.get("id"));
    attrs.set("bytes", b.getI64IntegerAttr(bytes));
    attrs.set("pitch", b.getI64IntegerAttr(pitch));
    if (*reason) {
      attrs.set("reason", b.getStringAttr(reason));
      rejected.push_back(attrs.getDictionary(fn.getContext()));
    } else
      accepted.push_back(attrs.getDictionary(fn.getContext()));
  }
  NamedAttrList out;
  if (fn->getAttrOfType<StringAttr>("block.topology") == "stream" &&
      phases.size() == 1 && !fn->hasAttr("kir.owner_axis") &&
      !fn.walk([](Operation *op) {
           auto n = op->getName().stripDialect();
           return (n == "reduce" || n == "combine" || n == "owner_store" ||
                   n == "owner_load" || n == "summary_reduce" ||
                   n == "summary_merge" || n == "scan_sum" ||
                   n == "publish_offsets" || n == "fill_ranges_at" ||
                   n == "scatter_fragment" || n == "unique_put" ||
                   n == "unique_put_at" || n == "atomic_add_at" ||
                   n == "key_counts")
                      ? WalkResult::interrupt()
                      : WalkResult::advance();
         }).wasInterrupted())
    out.set("independent_columns",b.getUnitAttr());
  if (asyncWrite)
    out.set("async_writeback", b.getUnitAttr());
  if(reuse)out.set("reuse_temporaries", b.getUnitAttr());
  if (approximate) out.set("approximate_math", b.getUnitAttr());
  out.set("group_temporaries",b.getBoolAttr(wide));
  out.set("persistent",b.getBoolAttr(persistent));
  out.set("participants",b.getI64IntegerAttr(participants));
  out.set("batch", b.getI64IntegerAttr(batch));
  out.set("budget", b.getI64IntegerAttr(std::min(c.budget, c.capacity)));
  out.set("depth", b.getI32IntegerAttr(depth));
  out.set("rows", b.getArrayAttr(accepted));
  out.set("rejected", b.getArrayAttr(rejected));
  return out.getDictionary(fn.getContext());
}
llvm::Expected<DictionaryAttr> plan(func::FuncOp fn, const LocalSupply &c,
                                    int64_t depth, int64_t participants,
                                    bool persistent, int64_t batch, bool approximate) {
  if (approximate && (!c.approxScratch || !fn.walk([&](Operation *op) {
        return c.approxScratch(op) ? WalkResult::interrupt() : WalkResult::advance();
      }).wasInterrupted()))
    return llvm::createStringError("approximate math requires an authorized vector operation");
  auto ordinary = planStorage(fn, c, depth, participants, persistent, false, batch, approximate);
  if (!ordinary || ordinary->getAs<ArrayAttr>("rejected").empty())
    return ordinary;
  auto compact = planStorage(fn, c, depth, participants, persistent, true, batch, approximate);
  if (!compact)
    return compact;
  if (compact->getAs<ArrayAttr>("rows").size() >
      ordinary->getAs<ArrayAttr>("rows").size())
    return compact;
  return ordinary;
}

namespace {
bool isUnprojectedArg(func::FuncOp fn, unsigned slot) {
  return !fn.getArgAttr(slot, "kir.indexing") &&
         !fn.getArgAttr(slot, "kir.source");
}

std::optional<int64_t> aliasGroupOf(const CallFacts &facts, unsigned slot) {
  if (slot >= facts.arguments.size())
    return std::nullopt;
  auto *obj = facts.arguments[slot].getAsObject();
  return obj ? obj->getInteger("alias_group") : std::nullopt;
}

bool isContiguous2D(func::FuncOp fn, const CallFacts &facts, unsigned slot,
                    int64_t d0, int64_t d1) {
  if (slot >= fn.getNumArguments() || slot >= facts.arguments.size() ||
      !isUnprojectedArg(fn, slot))
    return false;
  auto mr = dyn_cast<MemRefType>(fn.getArgument(slot).getType());
  auto *obj = facts.arguments[slot].getAsObject();
  auto *shape = obj ? obj->getArray("shape") : nullptr;
  auto *strides = obj ? obj->getArray("strides") : nullptr;
  if (!mr || mr.getRank() != 2 || !shape || !strides || shape->size() != 2 ||
      strides->size() != 2)
    return false;
  auto s0 = (*shape)[0].getAsInteger(), s1 = (*shape)[1].getAsInteger();
  auto st0 = (*strides)[0].getAsInteger(), st1 = (*strides)[1].getAsInteger();
  if (!s0 || !s1 || !st0 || !st1 || *s0 != d0 || *s1 != d1)
    return false;
  return (d1 <= 1 || *st1 == 1) && (d0 <= 1 || *st0 == d1);
}

bool isContiguous1D(func::FuncOp fn, const CallFacts &facts, unsigned slot,
                    int64_t d0) {
  if (slot >= fn.getNumArguments() || slot >= facts.arguments.size() ||
      !isUnprojectedArg(fn, slot))
    return false;
  auto mr = dyn_cast<MemRefType>(fn.getArgument(slot).getType());
  auto *obj = facts.arguments[slot].getAsObject();
  auto *shape = obj ? obj->getArray("shape") : nullptr;
  auto *strides = obj ? obj->getArray("strides") : nullptr;
  if (!mr || mr.getRank() != 1 || !shape || !strides || shape->size() != 1 ||
      strides->size() != 1)
    return false;
  auto s0 = (*shape)[0].getAsInteger(), st0 = (*strides)[0].getAsInteger();
  if (!s0 || !st0 || *s0 != d0)
    return false;
  return d0 <= 1 || *st0 == 1;
}

Value stripCastTo(Value v, Type targetElem) {
  if (auto *op = v.getDefiningOp();
      op && op->getName().stripDialect() == "cast" && op->getNumOperands() == 1) {
    auto rTy = dyn_cast<RankedTensorType>(v.getType());
    if (rTy && rTy.getElementType() == targetElem)
      return op->getOperand(0);
  }
  return v;
}

bool isLoadOf2D(Value v, Value expectedArg, Type workElem) {
  Value raw = stripCastTo(v, workElem);
  auto *op = raw.getDefiningOp();
  if (!op || op->getName().getStringRef() != "block.load" ||
      op->getOperand(0) != expectedArg)
    return false;
  if (auto axes = op->getAttrOfType<DenseI64ArrayAttr>("block.member_axes"))
    return axes.asArrayRef() == ArrayRef<int64_t>({0, 1});
  return true;
}

bool isLoadOf1D(Value v, Value expectedArg, Type workElem) {
  Value raw = stripCastTo(v, workElem);
  auto *op = raw.getDefiningOp();
  return op && op->getName().getStringRef() == "block.load" &&
         op->getOperand(0) == expectedArg;
}

bool gatherLayout(func::FuncOp fn, const CallFacts &facts, int64_t m, int64_t n) {
  if (fn.getNumArguments() != 3 || m * n > UINT32_MAX ||
      intentdsl::domainSlot(fn) != 2)
    return false;
  auto table = dyn_cast<MemRefType>(fn.getArgument(0).getType());
  auto ids = dyn_cast<MemRefType>(fn.getArgument(1).getType());
  auto out = dyn_cast<MemRefType>(fn.getArgument(2).getType());
  if (!table || !ids || !out || table.getElementType() != out.getElementType() ||
      !(ids.getElementType().isSignlessInteger(32) ||
        ids.getElementType().isSignlessInteger(64)))
    return false;
  auto dims = facts.shapeOf(fn.getArgument(0));
  auto rows = dims && dims->size() == 2 ? (*dims)[0].getAsInteger() : std::nullopt;
  auto bits = table.getElementTypeBitWidth();
  auto t = aliasGroupOf(facts, 0), i = aliasGroupOf(facts, 1), o = aliasGroupOf(facts, 2);
  return (bits == 8 || bits == 16 || bits == 32 || bits == 64) &&
         rows && *rows > 0 && *rows <= INT32_MAX &&
         t && i && o && t != o && i != o &&
         isContiguous2D(fn, facts, 0, *rows, n) &&
         isContiguous1D(fn, facts, 1, m) && isContiguous2D(fn, facts, 2, m, n);
}

bool matchGatherRows(func::FuncOp fn) {
  Operation *read = nullptr, *store = nullptr;
  auto walk = fn.walk([&](Operation *op) {
    StringRef name = op->getName().getStringRef();
    if (name == "block.indexed_tile_read" && !read)
      read = op;
    else if (name == "block.store" && !store)
      store = op;
    else if (!llvm::is_contained(ArrayRef<StringRef>{"func.func", "func.return",
                "block.pid", "block.zero", "block.rows", "block.columns",
                "block.active", "block.tiles", "block.yield"}, name))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (walk.wasInterrupted() || !read || !store || read->getBlock() != store->getBlock() ||
      read->getOperand(0) != fn.getArgument(0) || read->getOperand(1) != fn.getArgument(1) ||
      store->getOperand(0) != fn.getArgument(2) || store->getOperand(4) != read->getResult(0) ||
      store->hasAttr("block.member_axes"))
    return false;
  auto *active = store->getOperand(3).getDefiningOp();
  auto *rows = store->getOperand(1).getDefiningOp();
  auto *columns = store->getOperand(2).getDefiningOp();
  return active && active->getName().getStringRef() == "block.active" &&
         rows && rows->getName().getStringRef() == "block.rows" &&
         columns && columns->getName().getStringRef() == "block.columns" &&
         active->getOperand(0) == fn.getArgument(2) &&
         active->getOperand(1) == store->getOperand(1) &&
         active->getOperand(2) == store->getOperand(2) &&
         read->getOperands().drop_front(2) == store->getOperands().slice(1, 3);
}

bool matchTransposePattern(func::FuncOp fn, const CallFacts &facts, int64_t m,
                           int64_t n) {
  if (fn.getNumArguments() != 2)
    return false;
  auto inTy = dyn_cast<MemRefType>(fn.getArgument(0).getType());
  auto outTy = dyn_cast<MemRefType>(fn.getArgument(1).getType());
  if (!inTy || !outTy || inTy.getElementType() != outTy.getElementType())
    return false;
  unsigned bits = inTy.getElementTypeBitWidth();
  if (bits != 8 && bits != 16 && bits != 32)
    return false;
  auto g0 = aliasGroupOf(facts, 0), g1 = aliasGroupOf(facts, 1);
  if (!g0 || !g1 || *g0 == *g1 || !isContiguous2D(fn, facts, 0, m, n) ||
      !isContiguous2D(fn, facts, 1, n, m))
    return false;
  unsigned loads = 0, stores = 0;
  Operation *loadOp = nullptr, *storeOp = nullptr;
  bool badOp = false;
  fn.walk([&](Operation *op) {
    auto ns = op->getName().getDialectNamespace();
    if (ns == "builtin" || ns == "func")
      return;
    auto code = op->getName().stripDialect();
    if (code == "load") {
      ++loads;
      loadOp = op;
    } else if (code == "store") {
      ++stores;
      storeOp = op;
    } else if (!llvm::is_contained(
                   ArrayRef<StringRef>{"rows", "columns", "zero", "pid",
                                       "active", "tiles", "yield"},
                   code)) {
      badOp = true;
    }
  });
  if (badOp || loads != 1 || stores != 1 || !loadOp || !storeOp ||
      loadOp->getBlock() != storeOp->getBlock() ||
      loadOp->getOperand(0) != fn.getArgument(0) ||
      storeOp->getOperand(0) != fn.getArgument(1) ||
      storeOp->getOperand(4) != loadOp->getResult(0) ||
      !projectsTransposed2D(storeOp))
    return false;
  if (auto axes = loadOp->getAttrOfType<DenseI64ArrayAttr>("block.member_axes"))
    if (axes.asArrayRef() != ArrayRef<int64_t>({0, 1}))
      return false;
  return true;
}

bool matchSoftmaxPattern(func::FuncOp fn, const CallFacts &facts, int64_t m,
                         int64_t n) {
  if (fn.getNumArguments() != 2)
    return false;
  auto inTy = dyn_cast<MemRefType>(fn.getArgument(0).getType());
  auto outTy = dyn_cast<MemRefType>(fn.getArgument(1).getType());
  if (!inTy || !outTy || inTy.getElementType() != outTy.getElementType())
    return false;
  Type elem = inTy.getElementType();
  if (!elem.isF16() && !elem.isBF16() && !elem.isF32())
    return false;
  auto g0 = aliasGroupOf(facts, 0), g1 = aliasGroupOf(facts, 1);
  if (!g0 || !g1 || *g0 == *g1 || !isContiguous2D(fn, facts, 0, m, n) ||
      !isContiguous2D(fn, facts, 1, m, n))
    return false;

  Type f32 = Float32Type::get(fn.getContext());
  Value inArg = fn.getArgument(0), outArg = fn.getArgument(1);
  unsigned stores = 0;
  Operation *storeOp = nullptr;
  bool badOp = false;
  fn.walk([&](Operation *op) {
    auto ns = op->getName().getDialectNamespace();
    if (ns == "builtin" || ns == "func")
      return;
    auto code = op->getName().stripDialect();
    if (code == "store") {
      ++stores;
      storeOp = op;
    } else if (!llvm::is_contained(
                   ArrayRef<StringRef>{"rows", "columns", "zero", "pid",
                                       "active", "load", "cast", "reduce",
                                       "sub", "exp", "div", "tiles", "identity",
                                       "combine", "yield"},
                   code)) {
      badOp = true;
    }
  });
  if (badOp || stores != 1 || !storeOp || storeOp->getOperand(0) != outArg ||
      projectsTransposed2D(storeOp))
    return false;
  if (auto axes = storeOp->getAttrOfType<DenseI64ArrayAttr>("block.member_axes"))
    if (axes.asArrayRef() != ArrayRef<int64_t>({0, 1}))
      return false;

  Value storedVal = storeOp->getOperand(4);
  if (!elem.isF32()) {
    auto *cOp = storedVal.getDefiningOp();
    if (!cOp || cOp->getName().stripDialect() != "cast")
      return false;
    storedVal = cOp->getOperand(0);
  }
  auto *divOp = storedVal.getDefiningOp();
  if (!divOp || divOp->getName().stripDialect() != "div")
    return false;
  auto matchShiftedExp = [&](Value v, Value &outMax) -> bool {
    auto *expOp = v.getDefiningOp();
    if (!expOp || expOp->getName().stripDialect() != "exp")
      return false;
    auto *subOp = expOp->getOperand(0).getDefiningOp();
    if (!subOp || subOp->getName().stripDialect() != "sub" ||
        !isLoadOf2D(subOp->getOperand(0), inArg, f32))
      return false;
    outMax = subOp->getOperand(1);
    return true;
  };
  Value maxVal;
  if (!matchShiftedExp(divOp->getOperand(0), maxVal))
    return false;
  Value sumVal = divOp->getOperand(1);

  if (fn->getAttrOfType<StringAttr>("block.topology") == "resident") {
    auto *maxRed = maxVal.getDefiningOp();
    auto *sumRed = sumVal.getDefiningOp();
    if (!maxRed || !sumRed || maxRed->getName().stripDialect() != "reduce" ||
        sumRed->getName().stripDialect() != "reduce" ||
        maxRed->getAttrOfType<StringAttr>("algebra") != "max_num" ||
        sumRed->getAttrOfType<StringAttr>("algebra") != "sum" ||
        !isLoadOf2D(maxRed->getOperand(0), inArg, f32) ||
        sumRed->getOperand(0) != divOp->getOperand(0))
      return false;
    return true;
  }

  auto *maxLoop = maxVal.getDefiningOp();
  auto *sumLoop = sumVal.getDefiningOp();
  if (!maxLoop || !sumLoop || maxLoop->getName().stripDialect() != "tiles" ||
      sumLoop->getName().stripDialect() != "tiles" ||
      maxLoop->getNumResults() != 1 || sumLoop->getNumResults() != 1)
    return false;
  auto *maxComb =
      maxLoop->getRegion(0).front().getTerminator()->getOperand(0).getDefiningOp();
  auto *sumComb =
      sumLoop->getRegion(0).front().getTerminator()->getOperand(0).getDefiningOp();
  if (!maxComb || !sumComb || maxComb->getName().stripDialect() != "combine" ||
      sumComb->getName().stripDialect() != "combine" ||
      maxComb->getAttrOfType<StringAttr>("algebra") != "max_num" ||
      sumComb->getAttrOfType<StringAttr>("algebra") != "sum")
    return false;
  auto *maxRed = maxComb->getOperand(1).getDefiningOp();
  auto *sumRed = sumComb->getOperand(1).getDefiningOp();
  Value sumMax;
  return maxRed && sumRed && maxRed->getName().stripDialect() == "reduce" &&
         sumRed->getName().stripDialect() == "reduce" &&
         isLoadOf2D(maxRed->getOperand(0), inArg, f32) &&
         matchShiftedExp(sumRed->getOperand(0), sumMax) && sumMax == maxVal;
}

bool matchLayerNormPattern(func::FuncOp fn, const CallFacts &facts, int64_t m,
                           int64_t n) {
  if (fn.getNumArguments() != 5 || !fn.getArgument(4).getType().isF32())
    return false;
  auto xTy = dyn_cast<MemRefType>(fn.getArgument(0).getType());
  auto gTy = dyn_cast<MemRefType>(fn.getArgument(1).getType());
  auto bTy = dyn_cast<MemRefType>(fn.getArgument(2).getType());
  auto yTy = dyn_cast<MemRefType>(fn.getArgument(3).getType());
  if (!xTy || !gTy || !bTy || !yTy)
    return false;
  Type elem = xTy.getElementType();
  if (elem != gTy.getElementType() || elem != bTy.getElementType() ||
      elem != yTy.getElementType() ||
      (!elem.isF16() && !elem.isBF16() && !elem.isF32()))
    return false;
  auto ax = aliasGroupOf(facts, 0), ag = aliasGroupOf(facts, 1),
       ab = aliasGroupOf(facts, 2), ay = aliasGroupOf(facts, 3);
  if (!ax || !ag || !ab || !ay || *ay == *ax || *ay == *ag || *ay == *ab ||
      !isContiguous2D(fn, facts, 0, m, n) ||
      !isContiguous1D(fn, facts, 1, n) || !isContiguous1D(fn, facts, 2, n) ||
      !isContiguous2D(fn, facts, 3, m, n))
    return false;

  Type f32 = Float32Type::get(fn.getContext());
  Value xArg = fn.getArgument(0), gArg = fn.getArgument(1),
        bArg = fn.getArgument(2), yArg = fn.getArgument(3),
        epsArg = fn.getArgument(4);
  unsigned stores = 0, summaryReds = 0;
  Operation *storeOp = nullptr, *sumRedOp = nullptr;
  bool badOp = false;
  fn.walk([&](Operation *op) {
    auto ns = op->getName().getDialectNamespace();
    if (ns == "builtin" || ns == "func")
      return;
    auto code = op->getName().stripDialect();
    if (code == "store") {
      ++stores;
      storeOp = op;
    } else if (code == "summary_reduce") {
      ++summaryReds;
      sumRedOp = op;
    } else if (!llvm::is_contained(
                   ArrayRef<StringRef>{"rows", "columns", "zero", "pid",
                                       "active", "load", "cast", "sub", "mul",
                                       "add", "div", "rsqrt", "tiles",
                                       "summary_identity", "summary_merge",
                                       "summary_field", "yield"},
                   code)) {
      badOp = true;
    }
  });
  if (badOp || stores != 1 || summaryReds != 1 || !storeOp || !sumRedOp ||
      storeOp->getOperand(0) != yArg || projectsTransposed2D(storeOp) ||
      !isLoadOf2D(sumRedOp->getOperand(0), xArg, f32))
    return false;

  Value momVal = sumRedOp->getResult(0);
  if (fn->getAttrOfType<StringAttr>("block.topology") != "resident") {
    auto *loopOp = sumRedOp->getParentOp();
    while (loopOp && loopOp->getName().stripDialect() != "tiles")
      loopOp = loopOp->getParentOp();
    if (!loopOp || loopOp->getNumResults() != 1)
      return false;
    momVal = loopOp->getResult(0);
  }

  auto isSummaryField = [&](Value v, int64_t pos) {
    auto *op = v.getDefiningOp();
    auto pAttr = op ? op->getAttrOfType<IntegerAttr>("position") : IntegerAttr{};
    return op && op->getName().stripDialect() == "summary_field" &&
           op->getOperand(0) == momVal && pAttr && pAttr.getInt() == pos;
  };
  auto isInvStd = [&](Value v) {
    auto *rsq = v.getDefiningOp();
    if (!rsq || rsq->getName().stripDialect() != "rsqrt")
      return false;
    auto *add = rsq->getOperand(0).getDefiningOp();
    if (!add || add->getName().stripDialect() != "add")
      return false;
    Value varVal = add->getOperand(0) == epsArg   ? add->getOperand(1)
                   : add->getOperand(1) == epsArg ? add->getOperand(0)
                                                  : Value{};
    auto *div = varVal ? varVal.getDefiningOp() : nullptr;
    if (!div || div->getName().stripDialect() != "div" ||
        !isSummaryField(div->getOperand(0), 2))
      return false;
    auto *cntCast = div->getOperand(1).getDefiningOp();
    return cntCast && cntCast->getName().stripDialect() == "cast" &&
           isSummaryField(cntCast->getOperand(0), 0);
  };
  auto isCentered = [&](Value v) {
    auto *sub = v.getDefiningOp();
    return sub && sub->getName().stripDialect() == "sub" &&
           isLoadOf2D(sub->getOperand(0), xArg, f32) &&
           isSummaryField(sub->getOperand(1), 1);
  };

  Value yVal = storeOp->getOperand(4);
  if (!elem.isF32()) {
    auto *cOp = yVal.getDefiningOp();
    if (!cOp || cOp->getName().stripDialect() != "cast")
      return false;
    yVal = cOp->getOperand(0);
  }
  auto *biasAdd = yVal.getDefiningOp();
  if (!biasAdd || biasAdd->getName().stripDialect() != "add")
    return false;
  Value scaledVal = isLoadOf1D(biasAdd->getOperand(1), bArg, f32)
                        ? biasAdd->getOperand(0)
                    : isLoadOf1D(biasAdd->getOperand(0), bArg, f32)
                        ? biasAdd->getOperand(1)
                        : Value{};
  if (!scaledVal)
    return false;

  SmallVector<Value, 3> factors;
  std::function<void(Value)> collectMul = [&](Value v) {
    if (auto *mOp = v.getDefiningOp();
        mOp && mOp->getName().stripDialect() == "mul" && factors.size() < 4) {
      collectMul(mOp->getOperand(0));
      collectMul(mOp->getOperand(1));
      return;
    }
    factors.push_back(v);
  };
  collectMul(scaledVal);
  if (factors.size() != 3)
    return false;
  bool gotCentered = false, gotInv = false, gotGain = false;
  for (Value f : factors) {
    if (!gotCentered && isCentered(f))
      gotCentered = true;
    else if (!gotInv && isInvStd(f))
      gotInv = true;
    else if (!gotGain && isLoadOf1D(f, gArg, f32))
      gotGain = true;
    else
      return false;
  }
  return gotCentered && gotInv && gotGain;
}
} // namespace

std::optional<DictionaryAttr> planLibrary(func::FuncOp fn,
                                          const compilerinfra::TargetProfile &p) {
  auto *caps = p.services.get<LocalSupply>();
  auto *facts = p.services.get<CallFacts>();
  if (!caps || !facts || caps->libraryEstimator.empty() ||
      fn->hasAttr("kir.owner_axis"))
    return std::nullopt;
  auto configs = fn->getAttrOfType<ArrayAttr>("block.configs");
  if (!configs || configs.empty())
    return std::nullopt;
  auto firstRow = dyn_cast<DictionaryAttr>(configs[0]);
  if (!firstRow || !firstRow.getAs<IntegerAttr>("id"))
    return std::nullopt;

  auto [mD, nD] = facts->geometry(fn);
  int64_t m = int64_t(mD), n = int64_t(nD);
  if (m <= 0 || n <= 0 || m > INT32_MAX || n > INT32_MAX)
    return std::nullopt;

  std::string kind, impl;
  if (!fn.empty() && !fn.front().empty() &&
      fn.front().front().getName().getStringRef() == "dsa.library_call") {
    auto &call = fn.front().front();
    auto kAttr = call.getAttrOfType<StringAttr>("kind");
    auto iAttr = call.getAttrOfType<StringAttr>("implementation");
    if (!kAttr || !iAttr || call.getOperands() != ValueRange(fn.getArguments()))
      return std::nullopt;
    kind = kAttr.getValue().str();
    impl = iAttr.getValue().str();
    if (kind == "gather_rows") {
      if (impl != caps->libraryGatherImpl || !gatherLayout(fn, *facts, m, n))
        return std::nullopt;
    } else if (kind == "transpose") {
      if (impl != caps->libraryTransposeImpl || fn.getNumArguments() != 2 ||
          !isContiguous2D(fn, *facts, 0, m, n) ||
          !isContiguous2D(fn, *facts, 1, n, m))
        return std::nullopt;
    } else if (kind == "softmax") {
      if (impl != caps->librarySoftmaxImpl || fn.getNumArguments() != 2 ||
          !isContiguous2D(fn, *facts, 0, m, n) ||
          !isContiguous2D(fn, *facts, 1, m, n))
        return std::nullopt;
    } else if (kind == "layer_norm") {
      if (impl != caps->libraryLayerNormImpl || fn.getNumArguments() != 5 ||
          !isContiguous2D(fn, *facts, 0, m, n) ||
          !isContiguous1D(fn, *facts, 1, n) ||
          !isContiguous1D(fn, *facts, 2, n) ||
          !isContiguous2D(fn, *facts, 3, m, n))
        return std::nullopt;
    } else {
      return std::nullopt;
    }
  } else if (!caps->libraryGatherImpl.empty() && gatherLayout(fn, *facts, m, n) &&
             matchGatherRows(fn)) {
    kind = "gather_rows";
    impl = caps->libraryGatherImpl;
  } else if (!caps->libraryTransposeImpl.empty() &&
             matchTransposePattern(fn, *facts, m, n)) {
    kind = "transpose";
    impl = caps->libraryTransposeImpl;
  } else if (!caps->librarySoftmaxImpl.empty() &&
             matchSoftmaxPattern(fn, *facts, m, n)) {
    kind = "softmax";
    impl = caps->librarySoftmaxImpl;
  } else if (!caps->libraryLayerNormImpl.empty() &&
             matchLayerNormPattern(fn, *facts, m, n)) {
    kind = "layer_norm";
    impl = caps->libraryLayerNormImpl;
  } else {
    return std::nullopt;
  }

  Builder b(fn.getContext());
  NamedAttrList rowAttrs;
  rowAttrs.set("id", firstRow.get("id"));
  rowAttrs.set("bytes", b.getI64IntegerAttr(0));
  rowAttrs.set("pitch", b.getI64IntegerAttr(64));

  NamedAttrList out;
  out.set("library_impl", b.getStringAttr(impl));
  out.set("library_kind", b.getStringAttr(kind));
  out.set("group_temporaries", b.getBoolAttr(false));
  out.set("persistent", b.getBoolAttr(false));
  out.set("participants", b.getI64IntegerAttr(1));
  out.set("batch", b.getI64IntegerAttr(1));
  out.set("budget", b.getI64IntegerAttr(std::min(caps->budget, caps->capacity)));
  out.set("depth", b.getI32IntegerAttr(1));
  out.set("rows", b.getArrayAttr({rowAttrs.getDictionary(fn.getContext())}));
  out.set("rejected", b.getArrayAttr({}));
  return out.getDictionary(fn.getContext());
}
} // namespace intentdsl::dsa
namespace CI_UNIT_NAMESPACE {
using namespace mlir;
using namespace compilerinfra;
void supply(Unit &u) {
  DecisionFamily family;
  family.name = "supply";
  family.applies = [](Operation *op) {
    return isa<func::FuncOp>(op) &&
           !intentdsl::kir::orderedOwners(cast<func::FuncOp>(op)) &&
           !intentdsl::kir::matrixDomain(cast<func::FuncOp>(op));
  };
  family.read = [](Operation *op) { return op->getAttr("dsa.pending"); };
  family.write = [](Operation *op, Attribute a) {
    op->setAttr("dsa.pending", a);
    return llvm::Error::success();
  };
  family.verify = [](Operation *op, Attribute a,
                     const TargetProfile &p) -> llvm::Error {
    auto facts = p.services.get<intentdsl::dsa::LocalSupply>();
    auto d = dyn_cast<DictionaryAttr>(a);
    if (!d || !facts || !d.getAs<IntegerAttr>("depth") ||
        !d.getAs<IntegerAttr>("participants") ||
        !d.getAs<BoolAttr>("persistent") || !d.getAs<IntegerAttr>("batch"))
      return llvm::createStringError("missing private supply facts");
    if (auto lib = d.getAs<StringAttr>("library_impl")) {
      auto expected = intentdsl::dsa::planLibrary(cast<func::FuncOp>(op), p);
      if (!expected || *expected != d)
        return llvm::createStringError("library supply layout mismatch");
      return p.checkImplementationForOp(lib.getValue(), op);
    }
    auto check = intentdsl::dsa::plan(
        cast<func::FuncOp>(op), *facts, d.getAs<IntegerAttr>("depth").getInt(),
        d.getAs<IntegerAttr>("participants").getInt(),
        d.getAs<BoolAttr>("persistent").getValue(),
        d.getAs<IntegerAttr>("batch").getInt(), d.contains("approximate_math"));
    if (!check)
      return check.takeError();
    if(!d.getAs<ArrayAttr>("rows") || d.getAs<ArrayAttr>("rows").empty())return llvm::createStringError("no legal private storage row or DMA completion");
    if ((d.contains("histogram_panel") || d.contains("ticket_panel") ||
         d.contains("segment_gather")) &&
        facts->keyPanel)
      *check = facts->keyPanel(cast<func::FuncOp>(op), p, *check);
    if (*check != d)
      return llvm::createStringError(
          "supply row costs, exclusion reasons or wait capability changed");
    return llvm::Error::success();
  };
  family.verifyCandidate=[](Operation*,const StructuralCandidate &c,const TargetProfile &)->llvm::Error{
    auto d=cast<DictionaryAttr>(c.value);
    if (auto lib = d.getAs<StringAttr>("library_impl")) {
      if (c.id != "library-cnnl" || c.topology != "external-library" ||
          c.preconditions != std::vector<std::string>{"exact-library-abi"} ||
          c.implementations != std::vector<std::string>{lib.getValue().str()})
        return llvm::createStringError("library supply candidate mismatch");
      return llvm::Error::success();
    }
    if (!d.getAs<IntegerAttr>("batch"))
      return llvm::createStringError("missing supply batch factor");
    auto depth=d.getAs<IntegerAttr>("depth").getInt(),
         count=d.getAs<IntegerAttr>("participants").getInt(),
         batch=d.getAs<IntegerAttr>("batch").getInt();
    std::string topology=depth==1?"direct":"double-buffer";
    if(d.getAs<BoolAttr>("persistent").getValue())topology+="-shared";
    std::string expectedId = topology + "-" + std::to_string(count) +
                             (batch > 1 ? "-b" + std::to_string(batch) : "");
    if (d.contains("histogram_panel") || d.contains("ticket_panel") ||
        d.contains("segment_gather"))
      expectedId += "-key-panel";
    if (d.contains("approximate_math")) expectedId += "-approx";
    if(c.topology!=topology || c.id!=expectedId || c.preconditions!=std::vector<std::string>{"legal-private-supply"})return llvm::createStringError("supply topology or participants mismatch");
    return llvm::Error::success();};
  u.families.push_back({{std::move(family), "dsa@formed", "dsa@supplied"},
                        COMPILERINFRA_SOURCE});
  for (int depth : {1, 2}) {
    Strategy s;
    s.name = depth == 1 ? "direct" : "double-buffer";
    s.family = u.qualify("supply");
    s.enumerate = [depth](Operation *op, const TargetProfile &p,
                        AnalysisManager &) -> llvm::Expected<std::vector<StructuralCandidate>> {
      auto facts = p.services.get<intentdsl::dsa::LocalSupply>();
      if (!facts)
        return llvm::createStringError(
            "target has no private transfer resources");
      auto fn = cast<func::FuncOp>(op);
      bool canBatch = intentdsl::dsa::batchableInstances(fn);
      bool canApproximate = facts->approxScratch && fn.walk([&](Operation *i) {
        return facts->approxScratch(i) ? WalkResult::interrupt() : WalkResult::advance();
      }).wasInterrupted();
      std::vector<StructuralCandidate> cs;
      for(bool persistent:{false,true})for(int64_t n=facts->consumers;n>=1;n/=2){
        if(persistent && !intentdsl::dsa::reusableInput(fn))continue;
        for (int64_t batch : {1, 2, 4}) {
          if (batch > 1 && !canBatch)
            continue;
          auto d=intentdsl::dsa::plan(fn,*facts,depth,n,persistent,batch);
          if(!d)return d.takeError();
          std::string topology=depth==1?"direct":"double-buffer";
          if(persistent)topology+="-shared";
          std::string id = topology + "-" + std::to_string(n) +
                           (batch > 1 ? "-b" + std::to_string(batch) : "");
          cs.push_back({id,topology,*d,{"legal-private-supply"}});
          if (facts->keyPanel)
            if (auto panel = facts->keyPanel(fn, p, *d))
              cs.push_back({id + "-key-panel",
                            topology,
                            panel,
                            {"legal-private-supply"}});
          if (canApproximate) {
            auto fast = intentdsl::dsa::plan(fn, *facts, depth, n, persistent, batch, true);
            if (!fast) return fast.takeError();
            cs.push_back({id+"-approx",topology,*fast,{"legal-private-supply"}});
          }
        }
      }return cs;
    };
    u.strategies.push_back({std::move(s), COMPILERINFRA_SOURCE});
  }
  Strategy libS;
  libS.name = "library";
  libS.family = u.qualify("supply");
  libS.enumerate = [](Operation *op, const TargetProfile &p, AnalysisManager &)
      -> llvm::Expected<std::vector<StructuralCandidate>> {
    std::vector<StructuralCandidate> cs;
    if (auto d = intentdsl::dsa::planLibrary(cast<func::FuncOp>(op), p)) {
      auto impl = d->getAs<StringAttr>("library_impl").getValue().str();
      cs.push_back({"library-cnnl",
                    "external-library",
                    *d,
                    {"exact-library-abi"},
                    {impl}});
    }
    return cs;
  };
  u.strategies.push_back({std::move(libS), COMPILERINFRA_SOURCE});
  Realizer r;
  r.name = "stage-transfers";
  r.family = u.qualify("supply");
  r.apply = intentdsl::dsa::stageTransfers;
  u.realizers.push_back({std::move(r), COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
