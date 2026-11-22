#include "intentdsl/levels/storage/dsa/ir/Owners.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
using namespace mlir;
using namespace compilerinfra;
namespace intentdsl::dsa {
namespace {
// Successive SSA values have distinct storage, including loop results which
// hold the carried state. The second slot per loop result enables parallel
// yield assignment, even for exchanged carry values.
LogicalResult memberStorage(func::FuncOp fn, const TargetProfile &p,
                            bool assign, int64_t *required = nullptr) {
  auto caps = p.services.get<LocalSupply>();
  if (!caps || caps->alignment <= 0)
    return failure();
  int64_t offset = 0;
  Builder b(fn.getContext());
  auto &space = fn.front().front();
  auto status = space.walk([&](Operation *op) {
    SmallVector<int64_t> offsets;
    for (auto v : op->getResults()) {
      auto t = dyn_cast<RankedTensorType>(v.getType());
      if (!t)
        continue;
      if (!kir::memberAxis(v))
        return WalkResult::interrupt();
      auto k = kir::memberExtent(fn, v, "block.shape");
      int64_t bytes = 0;
      if (k < 0 || __builtin_mul_overflow(
                       std::max<int64_t>(64, k),
                       int64_t(t.getElementTypeBitWidth() / 8), &bytes))
        return WalkResult::interrupt();
      for (int i = 0; i < (op->getName().getStringRef() == "scf.for" ? 2 : 1);
           ++i) {
        int64_t alignment = std::max<int64_t>(64, caps->alignment);
        offset = llvm::alignTo(offset, alignment);
        offsets.push_back(offset);
        if (__builtin_add_overflow(offset, bytes, &offset) ||
            offset > std::min(caps->budget, caps->capacity))
          return WalkResult::interrupt();
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
    } else if (op->hasAttr("dsa.member_offsets"))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (status.wasInterrupted())
    return space.emitError(
        "member state exceeds local capacity or disagrees with placement");
  if (required) {
    *required = offset;
    return success();
  }
  auto size = b.getI64IntegerAttr(offset);
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
bool fits(func::FuncOp fn, const TargetProfile &p, int64_t extra) {
  int64_t members = 0;
  auto cap = p.services.get<LocalSupply>();
  return cap && extra >= 0 &&
         succeeded(memberStorage(fn, p, false, &members)) &&
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
    if (!b || !d.getAs<StringAttr>("mode") || !fits(fn, p, b.getInt()))
      return space.emitError("missing supply decision");
    auto m = d.getAs<StringAttr>("mode");
    return success((m == "direct" && b.getInt() == 0) ||
                   (m == "stage-slab" && b.getInt() > 0 &&
                    b.getInt() == privateBytes(fn, p)) ||
                   (m == "streamed-owner-views" && b.getInt() > 0 &&
                    b.getInt() == ownerStreams(fn, p).bytes));
  }
  auto bytes = space.getAttrOfType<IntegerAttr>("dsa.owner_bytes");
  if (!bytes || (bytes.getInt() != 0 && bytes.getInt() != privateBytes(fn, p)))
    return space.emitError("invalid private owner slab");
  auto transfers = space.getAttrOfType<StringAttr>("dsa.transfers");
  if (!transfers)
    return space.emitError("missing ordered transfers");
  auto streaming = transfers == "streamed-owner-views";
  if (streaming) {
    auto plan = ownerStreams(fn, p);
    auto size = space.getAttrOfType<IntegerAttr>("dsa.stream_bytes");
    if (!size || !plan.bytes || size.getInt() != plan.bytes)
      return space.emitError("invalid stream capacity");
    llvm::DenseSet<Operation *> known;
    for (auto &t : plan.transfers) {
      known.insert(t.access);
      auto d = t.access->getAttrOfType<DenseI64ArrayAttr>("dsa.stream");
      if (!d || d.asArrayRef() !=
                    ArrayRef<int64_t>{t.offset, t.axis, t.steps, t.width})
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
  if (supplied && (space.getAttrOfType<StringAttr>("dsa.schedule") !=
                       "participant-stride" ||
                   (!streaming &&
                    space.getAttrOfType<StringAttr>("dsa.transfers") !=
                        (bytes.getInt() ? "read-before-work/write-on-completion"
                                        : "direct"))))
    return space.emitError("missing owner transfer or completion");
  return supplied ? memberStorage(fn, p, false) : success();
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
    if (!b || !mode || !dsa::fits(fn, p, b.getInt()) ||
        !(mode == "direct" && !b.getInt() ||
          mode == "stage-slab" && b.getInt() > 0 &&
              b.getInt() == dsa::privateBytes(fn, p) ||
          mode == "streamed-owner-views" && b.getInt() > 0 &&
              b.getInt() == dsa::ownerStreams(fn, p).bytes))
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
    if (auto bytes = dsa::privateBytes(fn, p); bytes && dsa::fits(fn, p, bytes))
      add("stage-slab", bytes);
    if (auto bytes = dsa::ownerStreams(fn, p).bytes;
        bytes && dsa::fits(fn, p, bytes))
      add("streamed-owner-views", bytes);
    return choices;
  };
  u.strategies.push_back({std::move(s), COMPILERINFRA_SOURCE});
  Realizer r;
  r.name = "bind-owner-supply";
  r.family = u.qualify("ordered-supply");
  r.apply = [](Operation *o, Attribute a, const TargetProfile &p,
               RewriterBase &b) -> llvm::Error {
    if (failed(dsa::memberStorage(o->getParentOfType<func::FuncOp>(), p, true)))
      return llvm::createStringError("cannot place carried member values");
    auto d = cast<DictionaryAttr>(a);
    auto mode = d.getAs<StringAttr>("mode");
    auto bytes = d.getAs<IntegerAttr>("bytes");
    o->setAttr("dsa.owner_bytes",
               mode == "stage-slab" ? bytes : b.getI64IntegerAttr(0));
    o->removeAttr("dsa.owner_choice");
    if (mode == "streamed-owner-views") {
      auto caps = p.services.get<dsa::LocalSupply>();
      if (o->getAttrOfType<IntegerAttr>("dsa.member_bytes").getInt() >
          std::min(caps->capacity, caps->budget) - bytes.getInt())
        return llvm::createStringError(
            "stream panels and live carries exceed local memory");
      auto plan = dsa::ownerStreams(o->getParentOfType<func::FuncOp>(), p);
      for (auto &t : plan.transfers)
        t.access->setAttr(
            "dsa.stream",
            b.getDenseI64ArrayAttr({t.offset, t.axis, t.steps, t.width}));
      o->setAttr("dsa.stream_bytes", bytes);
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
