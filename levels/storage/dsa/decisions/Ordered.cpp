#include "intentdsl/levels/storage/dsa/ir/Owners.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace compilerinfra;

namespace intentdsl::dsa {
namespace {
// Allocate private NRAM slots for carried and intermediate member fragments
// from their actual SSA live intervals, reusing non-overlapping lifetimes.
LogicalResult memberStorage(func::FuncOp fn, const TargetProfile &p,
                            int64_t pack, bool panelFold, bool assign,
                            int64_t *required = nullptr) {
  auto caps = p.services.get<LocalSupply>();
  if (!caps || caps->alignment <= 0)
    return failure();
  Builder b(fn.getContext());
  auto &space = fn.front().front();

  if (panelFold) {
    llvm::DenseMap<Operation *, SmallVector<int64_t>> offsets;
    auto pf = panelFoldSupply(fn, p, &offsets);
    if (!pf)
      return failure();
    if (required) {
      *required = pf->localBytes;
      return success();
    }
    bool ok = true;
    space.walk([&](Operation *op) {
      if (op == &space)
        return;
      auto it = offsets.find(op);
      if (it != offsets.end()) {
        auto expected = b.getDenseI64ArrayAttr(it->second);
        if (assign)
          op->setAttr("dsa.member_offsets", expected);
        else if (op->getAttr("dsa.member_offsets") != expected)
          ok = false;
      } else if (op->hasAttr("dsa.member_offsets")) {
        ok = false;
      }
    });
    auto scratchAttr = b.getDenseI64ArrayAttr(pf->scratchOffsets);
    auto bytesAttr = b.getI64IntegerAttr(pf->localBytes);
    if (assign) {
      space.setAttr("dsa.panel_scratch", scratchAttr);
      space.setAttr("dsa.member_bytes", bytesAttr);
    }
    return success(ok && space.getAttr("dsa.panel_scratch") == scratchAttr &&
                   space.getAttr("dsa.member_bytes") == bytesAttr);
  }

  auto &bb = space.getRegion(0).front();
  Value innerOwner = bb.getNumArguments() ? bb.getArguments().back() : Value{};
  llvm::DenseMap<Operation *, unsigned> startOrd, endOrd;
  unsigned nextOrd = 0;
  std::function<void(Block &)> numberBlock = [&](Block &blk) {
    for (auto &op : blk) {
      startOrd[&op] = nextOrd++;
      for (auto &reg : op.getRegions())
        for (auto &sub : reg)
          numberBlock(sub);
      endOrd[&op] = nextOrd++;
    }
  };
  numberBlock(bb);

  struct LiveSlot {
    unsigned start, end;
    int64_t offset, size;
  };
  SmallVector<LiveSlot> slots;
  int64_t alignment = std::max<int64_t>(64, caps->alignment);
  int64_t peak = 0;

  auto allocInterval = [&](unsigned start, unsigned end,
                           int64_t bytes) -> std::optional<int64_t> {
    int64_t alignedSize = 0;
    if (__builtin_add_overflow(bytes, alignment - 1, &alignedSize))
      return {};
    alignedSize = llvm::alignTo(bytes, alignment);
    int64_t off = 0;
    for (const auto &s : slots) {
      if (s.end < start || end < s.start)
        continue;
      off = llvm::alignTo(off, alignment);
      if (off + alignedSize <= s.offset)
        break;
      int64_t cand = 0;
      if (__builtin_add_overflow(s.offset, s.size, &cand))
        return {};
      off = std::max<int64_t>(off, llvm::alignTo(cand, alignment));
    }
    int64_t nextEnd = 0;
    if (__builtin_add_overflow(off, alignedSize, &nextEnd) ||
        nextEnd > std::min(caps->budget, caps->capacity))
      return {};
    peak = std::max(peak, nextEnd);
    auto pos = llvm::lower_bound(slots, off, [](const LiveSlot &s, int64_t key) {
      return s.offset < key;
    });
    slots.insert(pos, {start, end, off, alignedSize});
    return off;
  };

  auto lastUseOrd = [&](Value v, Operation *defOp) {
    unsigned hi = endOrd.lookup(defOp);
    for (Operation *u : v.getUsers()) {
      Operation *top = u;
      while (top->getBlock() != defOp->getBlock() && top->getParentOp())
        top = top->getParentOp();
      hi = std::max(hi, endOrd.lookup(top));
    }
    return hi;
  };

  auto status = space.walk<WalkOrder::PreOrder>([&](Operation *op) {
    SmallVector<int64_t> offsets;
    for (auto v : op->getResults()) {
      auto t = dyn_cast<RankedTensorType>(v.getType());
      if (!t)
        continue;
      if (!kir::memberAxis(v))
        return WalkResult::interrupt();
      auto k = kir::memberExtent(fn, v, "block.shape");
      int64_t factor =
          (pack > 1 && innerOwner && ownerValueDependsOn(v, innerOwner)) ? pack
                                                                         : 1;
      int64_t elems = 0, bytes = 0;
      if (k < 0 || __builtin_mul_overflow(factor, k, &elems) ||
          __builtin_mul_overflow(std::max<int64_t>(64, elems),
                                 int64_t(t.getElementTypeBitWidth() / 8),
                                 &bytes))
        return WalkResult::interrupt();
      unsigned sOrd = startOrd.lookup(op);
      unsigned eOrd = lastUseOrd(v, op);
      auto slot0 = allocInterval(sOrd, eOrd, bytes);
      if (!slot0)
        return WalkResult::interrupt();
      offsets.push_back(*slot0);
      if (op->getName().getStringRef() == "scf.for") {
        auto slot1 = allocInterval(endOrd.lookup(op) - 1, endOrd.lookup(op), bytes);
        if (!slot1)
          return WalkResult::interrupt();
        offsets.push_back(*slot1);
      }
    }
    if (!offsets.empty()) {
      if (required)
        return WalkResult::advance();
      auto expected = b.getDenseI64ArrayAttr(offsets);
      if (assign)
        op->setAttr("dsa.member_offsets", expected);
      else if (op->getAttr("dsa.member_offsets") != expected)
        return WalkResult::interrupt();
    } else if (op->hasAttr("dsa.member_offsets")) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (status.wasInterrupted())
    return space.emitError(
        "member state exceeds local capacity or disagrees with placement");
  if (required) {
    *required = peak;
    return success();
  }
  auto size = b.getI64IntegerAttr(peak);
  if (assign)
    space.setAttr("dsa.member_bytes", size);
  return success(space.getAttr("dsa.member_bytes") == size);
}

int64_t privateBytes(func::FuncOp fn, const TargetProfile &p) {
  if (fn.getNumArguments() != 1)
    return 0;
  auto &space = fn.front().front();
  auto axes = space.getAttrOfType<DenseI64ArrayAttr>("axes");
  auto view = cast<MemRefType>(fn.getArgument(0).getType());
  auto el = view.getElementTypeBitWidth() / 8;
  auto shape = fn.getArgAttrOfType<DenseI64ArrayAttr>(0, "block.shape"),
       stride = fn.getArgAttrOfType<DenseI64ArrayAttr>(0, "block.strides");
  if (axes.size() != 1 || axes[0] != 0 || view.getRank() != 3 || !shape ||
      !stride || stride[2] != 1 || stride[1] != shape[2])
    return 0;
  auto row = space.getRegion(0).front().getArgument(0);
  if (space
          .walk([&](Operation *o) {
            if (auto l = dyn_cast<memref::LoadOp>(o))
              return l.getIndices()[0] != row ? WalkResult::interrupt()
                                              : WalkResult::advance();
            if (auto s = dyn_cast<memref::StoreOp>(o))
              return s.getIndices()[0] != row ? WalkResult::interrupt()
                                              : WalkResult::advance();
            return WalkResult::advance();
          })
          .wasInterrupted())
    return 0;
  auto caps = p.services.get<LocalSupply>();
  int64_t bytes = 0;
  if (!caps || __builtin_mul_overflow(shape[1], shape[2], &bytes) ||
      __builtin_mul_overflow(bytes, (int64_t)el, &bytes) || bytes <= 0 ||
      bytes > std::min(caps->budget, caps->capacity))
    return 0;
  return bytes;
}

bool fits(func::FuncOp fn, const TargetProfile &p, int64_t extra,
          int64_t pack = 1, bool panelFold = false) {
  int64_t members = 0;
  auto cap = p.services.get<LocalSupply>();
  if (!cap || extra < 0)
    return false;
  if (panelFold) {
    auto pf = panelFoldSupply(fn, p);
    auto ms = p.services.get<MatrixStorage>();
    return pf && ms && extra == pf->wramBytes && pf->wramBytes <= ms->wramBytes &&
           pf->localBytes <= std::min(cap->budget, cap->capacity);
  }
  return succeeded(memberStorage(fn, p, pack, false, false, &members)) &&
         members <= std::min(cap->budget, cap->capacity) - extra;
}
} // namespace

LogicalResult verifyOwners(func::FuncOp fn, const TargetProfile &p,
                           bool supplied) {
  if (failed(block::verifyOwnerMap(fn, p, "block", "realized")))
    return failure();
  auto &space = fn.front().front();
  if (!supplied) {
    auto d = space.getAttrOfType<DictionaryAttr>("dsa.owner_choice");
    auto b = d ? d.getAs<IntegerAttr>("bytes") : IntegerAttr();
    auto m = d ? d.getAs<StringAttr>("mode") : StringAttr();
    if (!b || !m)
      return space.emitError("missing supply decision");
    if (m == "panel-fold") {
      auto pf = panelFoldSupply(fn, p);
      return success(pf && b.getInt() == pf->wramBytes &&
                     fits(fn, p, b.getInt(), 1, true));
    }
    auto streams = m == "streamed-owner-views" ? ownerStreams(fn, p)
                                               : OwnerStreams{};
    int64_t pack = m == "streamed-owner-views" ? streams.pack : 1;
    if (!fits(fn, p, b.getInt(), pack, false))
      return space.emitError("insufficient private capacity for supply choice");
    return success((m == "direct" && b.getInt() == 0) ||
                   (m == "stage-slab" && b.getInt() > 0 &&
                    b.getInt() == privateBytes(fn, p)) ||
                   (m == "streamed-owner-views" && b.getInt() > 0 &&
                    b.getInt() == streams.bytes));
  }

  auto bytes = space.getAttrOfType<IntegerAttr>("dsa.owner_bytes");
  if (!bytes || (bytes.getInt() != 0 && bytes.getInt() != privateBytes(fn, p)))
    return space.emitError("invalid private owner slab");
  auto transfers = space.getAttrOfType<StringAttr>("dsa.transfers");
  if (!transfers)
    return space.emitError("missing ordered transfers");

  if (transfers == "panel-fold") {
    auto pf = panelFoldSupply(fn, p);
    auto rows = space.getAttrOfType<IntegerAttr>("dsa.panel_rows");
    auto span = space.getAttrOfType<IntegerAttr>("dsa.panel_span");
    auto wram = space.getAttrOfType<IntegerAttr>("dsa.wram_bytes");
    auto sched = space.getAttrOfType<StringAttr>("dsa.schedule");
    if (!pf || !rows || !span || !wram || bytes.getInt() != 0 ||
        rows.getInt() != pf->rows || span.getInt() != pf->span ||
        wram.getInt() != pf->wramBytes || !sched ||
        sched.getValue() != "participant-stride")
      return space.emitError("invalid cooperative panel fold supply");
    return memberStorage(fn, p, 1, true, false);
  }

  auto streaming = transfers == "streamed-owner-views";
  int64_t pack = 1;
  if (streaming) {
    auto plan = ownerStreams(fn, p);
    auto size = space.getAttrOfType<IntegerAttr>("dsa.stream_bytes");
    auto packAttr = space.getAttrOfType<IntegerAttr>("dsa.owner_pack");
    if (!size || !plan.bytes || size.getInt() != plan.bytes || !packAttr ||
        packAttr.getInt() != plan.pack)
      return space.emitError("invalid stream capacity or owner pack");
    pack = plan.pack;
    llvm::DenseSet<Operation *> known;
    for (auto &t : plan.transfers) {
      known.insert(t.access);
      auto d = t.access->getAttrOfType<DenseI64ArrayAttr>("dsa.stream");
      if (!d || d.asArrayRef() !=
                    ArrayRef<int64_t>{t.offset, t.axis, t.steps, t.width, t.pack})
        return t.access->emitError("incorrect stream transfer");
    }
    if (space
            .walk([&](Operation *op) {
              return op->hasAttr("dsa.stream") && !known.contains(op)
                         ? WalkResult::interrupt()
                         : WalkResult::advance();
            })
            .wasInterrupted())
      return space.emitError("unverified stream transfer");
  }
  auto sched = space.getAttrOfType<StringAttr>("dsa.schedule");
  if (!sched || sched.getValue() != "participant-stride" ||
      (!streaming &&
       transfers.getValue() !=
           (bytes.getInt() ? "read-before-work/write-on-completion" : "direct")))
    return space.emitError("missing owner transfer or completion");
  return memberStorage(fn, p, pack, false, false);
}
} // namespace intentdsl::dsa

namespace CI_UNIT_NAMESPACE {
using namespace intentdsl;
void ownerSupply(Unit &u) {
  DecisionFamily f;
  f.name = "ordered-supply";
  f.applies = [](Operation *o) {
    return o->getName().getStringRef() == "block.owner_space" &&
           !o->hasAttr("dsa.schedule");
  };
  f.read = [](Operation *o) { return o->getAttr("dsa.owner_choice"); };
  f.write = [](Operation *o, Attribute a) {
    o->setAttr("dsa.owner_choice", a);
    return llvm::Error::success();
  };
  f.verify = [](Operation *o, Attribute a,
                const TargetProfile &p) -> llvm::Error {
    auto d = dyn_cast<DictionaryAttr>(a);
    auto b = d ? d.getAs<IntegerAttr>("bytes") : IntegerAttr();
    auto fn = o->getParentOfType<func::FuncOp>();
    auto mode = d ? d.getAs<StringAttr>("mode") : StringAttr();
    if (!b || !mode)
      return llvm::createStringError("missing ordered supply mode");
    if (mode == "panel-fold") {
      auto pf = dsa::panelFoldSupply(fn, p);
      if (!pf || b.getInt() != pf->wramBytes ||
          !dsa::fits(fn, p, b.getInt(), 1, true))
        return llvm::createStringError("panel fold exceeds NRAM/WRAM supply");
      return llvm::Error::success();
    }
    auto streams =
        mode == "streamed-owner-views" ? dsa::ownerStreams(fn, p)
                                       : dsa::OwnerStreams{};
    int64_t pack = mode == "streamed-owner-views" ? streams.pack : 1;
    if (!dsa::fits(fn, p, b.getInt(), pack, false) ||
        !((mode == "direct" && !b.getInt()) ||
          (mode == "stage-slab" && b.getInt() > 0 &&
           b.getInt() == dsa::privateBytes(fn, p)) ||
          (mode == "streamed-owner-views" && b.getInt() > 0 &&
           b.getInt() == streams.bytes)))
      return llvm::createStringError(
          "insufficient private capacity or noncontiguous owner");
    return llvm::Error::success();
  };
  f.verifyCandidate = [](Operation *, const StructuralCandidate &c,
                         const TargetProfile &) -> llvm::Error {
    auto d = dyn_cast<DictionaryAttr>(c.value);
    if (!d || !d.getAs<StringAttr>("mode") ||
        d.getAs<StringAttr>("mode") != c.id || c.topology != c.id ||
        c.preconditions != std::vector<std::string>{"exclusive-owner"})
      return llvm::createStringError("invalid ordered supply candidate");
    return llvm::Error::success();
  };
  u.families.push_back(
      {{std::move(f), "dsa@formed", "dsa@supplied"}, COMPILERINFRA_SOURCE});

  Strategy s;
  s.name = "owner-working-set";
  s.family = u.qualify("ordered-supply");
  s.enumerate = [](Operation *o, const TargetProfile &p, AnalysisManager &)
      -> llvm::Expected<std::vector<StructuralCandidate>> {
    auto b = Builder(o->getContext());
    std::vector<StructuralCandidate> choices;
    auto add = [&](std::string mode, int64_t bytes) {
      choices.push_back(
          {mode,
           mode,
           b.getDictionaryAttr(
               {b.getNamedAttr("mode", b.getStringAttr(mode)),
                b.getNamedAttr("bytes", b.getI64IntegerAttr(bytes))}),
           {"exclusive-owner"}});
    };
    add("direct", 0);
    auto fn = o->getParentOfType<func::FuncOp>();
    if (auto bytes = dsa::privateBytes(fn, p);
        bytes && dsa::fits(fn, p, bytes))
      add("stage-slab", bytes);
    if (auto streams = dsa::ownerStreams(fn, p);
        streams.bytes && dsa::fits(fn, p, streams.bytes, streams.pack))
      add("streamed-owner-views", streams.bytes);
    if (auto pf = dsa::panelFoldSupply(fn, p);
        pf && dsa::fits(fn, p, pf->wramBytes, 1, true))
      add("panel-fold", pf->wramBytes);
    return choices;
  };
  u.strategies.push_back({std::move(s), COMPILERINFRA_SOURCE});

  Realizer r;
  r.name = "bind-owner-supply";
  r.family = u.qualify("ordered-supply");
  r.apply = [](Operation *o, Attribute a, const TargetProfile &p,
               RewriterBase &b) -> llvm::Error {
    auto fn = o->getParentOfType<func::FuncOp>();
    auto d = cast<DictionaryAttr>(a);
    auto mode = d.getAs<StringAttr>("mode");
    auto bytes = d.getAs<IntegerAttr>("bytes");

    if (mode == "panel-fold") {
      auto pf = dsa::panelFoldSupply(fn, p);
      if (!pf || failed(dsa::memberStorage(fn, p, 1, true, true)))
        return llvm::createStringError("cannot place panel fold fragments");
      o->setAttr("dsa.owner_bytes", b.getI64IntegerAttr(0));
      o->setAttr("dsa.panel_rows", b.getI64IntegerAttr(pf->rows));
      o->setAttr("dsa.panel_span", b.getI64IntegerAttr(pf->span));
      o->setAttr("dsa.wram_bytes", b.getI64IntegerAttr(pf->wramBytes));
      o->removeAttr("dsa.owner_choice");
      o->setAttr("dsa.schedule", b.getStringAttr("participant-stride"));
      o->setAttr("dsa.transfers", mode);
      return llvm::Error::success();
    }

    auto streams =
        mode == "streamed-owner-views" ? dsa::ownerStreams(fn, p)
                                       : dsa::OwnerStreams{};
    int64_t pack = mode == "streamed-owner-views" ? streams.pack : 1;
    if (failed(dsa::memberStorage(fn, p, pack, false, true)))
      return llvm::createStringError("cannot place carried member values");
    o->setAttr("dsa.owner_bytes",
               mode == "stage-slab" ? bytes : b.getI64IntegerAttr(0));
    o->removeAttr("dsa.owner_choice");
    if (mode == "streamed-owner-views") {
      auto caps = p.services.get<dsa::LocalSupply>();
      if (o->getAttrOfType<IntegerAttr>("dsa.member_bytes").getInt() >
          std::min(caps->capacity, caps->budget) - bytes.getInt())
        return llvm::createStringError(
            "stream panels and live carries exceed local memory");
      for (auto &t : streams.transfers)
        t.access->setAttr("dsa.stream",
                          b.getDenseI64ArrayAttr(
                              {t.offset, t.axis, t.steps, t.width, t.pack}));
      o->setAttr("dsa.stream_bytes", bytes);
      o->setAttr("dsa.owner_pack", b.getI64IntegerAttr(pack));
    }
    o->setAttr("dsa.schedule", b.getStringAttr("participant-stride"));
    o->setAttr("dsa.transfers",
               mode == "stage-slab"
                   ? b.getStringAttr("read-before-work/write-on-completion")
                   : mode);
    return llvm::Error::success();
  };
  u.realizers.push_back({std::move(r), COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
