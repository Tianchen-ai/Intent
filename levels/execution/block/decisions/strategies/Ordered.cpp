#include "intentdsl/levels/execution/block/ir/PanelBounds.h"
#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "intentdsl/levels/execution/block/ir/FoldIssue.h"
#include "intentdsl/levels/execution/block/ir/FoldPanel.h"
#include "intentdsl/levels/execution/block/ir/Owners.h"
#include "intentdsl/levels/execution/block/ir/SharedReads.h"
#include "mlir/IR/PatternMatch.h"
using namespace mlir;
using namespace compilerinfra;
namespace intentdsl::block {
static bool isLargeScalarOwnerSpace(Operation *space, const TargetProfile &p,
                                    int64_t owners) {
  auto caps = p.services.get<OwnerMachine>();
  if (!caps || caps->parallelMembers < 128 || owners < 65536)
    return false;
  auto constIdx = [](Value v) -> std::optional<int64_t> {
    while (auto cast = v.getDefiningOp<arith::IndexCastOp>())
      v = cast.getIn();
    if (auto c = v.getDefiningOp<arith::ConstantOp>())
      if (auto ia = dyn_cast<IntegerAttr>(c.getValue());
          ia && (ia.getType().isIndex() ||
                 (ia.getType().isIntOrFloat() &&
                  ia.getType().getIntOrFloatBitWidth() <= 64)))
        return ia.getInt();
    return std::nullopt;
  };
  return !space
              ->walk([&](Operation *inst) {
                auto n = inst->getName().stripDialect();
                if (n == "slice_read" || n == "slice_write" ||
                    n == "lane_sum" || n == "lane_splat")
                  return WalkResult::interrupt();
                if (auto loop = dyn_cast<scf::ForOp>(inst)) {
                  auto lb = constIdx(loop.getLowerBound());
                  auto ub = constIdx(loop.getUpperBound());
                  auto st = constIdx(loop.getStep());
                  if (!lb || !ub || !st || *st <= 0 ||
                      (*ub - *lb + *st - 1) / *st > 16)
                    return WalkResult::interrupt();
                  for (Type rt : loop.getResultTypes())
                    if (isa<ShapedType>(rt))
                      return WalkResult::interrupt();
                }
                return WalkResult::advance();
              })
              .wasInterrupted();
}
static bool legalWidth(IntegerAttr width, int64_t owners, int64_t maxW = 128) {
  return width && width.getInt() > 0 && width.getInt() <= maxW &&
         llvm::isPowerOf2_64(width.getInt()) &&
         width.getInt() <= std::max<int64_t>(1, llvm::PowerOf2Ceil(owners));
}
LogicalResult verifyOwnerMap(func::FuncOp fn, const TargetProfile &p,
                             StringRef dialect, StringRef phase) {
  if (auto dm = fn->getAttr("denormal-fp-math")) {
    auto s = dyn_cast<StringAttr>(dm);
    if (!s || !intentdsl::kir::validDenormalMode(s.getValue()))
      return fn.emitError("invalid denormal-fp-math mode");
  }
  bool columns = fn.front().front().hasAttr("block.column_prefixes");
  if (columns) {
    if (phase == "formed" || phase == "decided")
      return fn.emitError("unexpected realized columns");
    if (failed(verifyColumns(fn, dialect)))
      return failure();
  } else if (failed(kir::verifyOwners(fn, dialect)))
    return failure();
  auto &space = fn.front().front();
  auto facts = p.services.get<CallFacts>();
  if (!facts)
    return fn.emitError("missing native argument layout");
  if (failed(kir::verifyProjectedOwners(space, *facts)))
    return failure();
  if (failed(verifySharedReads(&space, *facts)))
    return failure();
  auto span = space.getAttrOfType<IntegerAttr>("block.token_span");
  auto pairAxis = space.getAttrOfType<IntegerAttr>("block.pair_axis");
  auto validPair = [&](IntegerAttr width, IntegerAttr s, IntegerAttr axis) {
    if (!width || !s || !axis)
      return false;
    auto result = issueAxis(&space, p, width.getInt(), s.getInt());
    return result && *result == axis.getInt();
  };
  if (pairAxis &&
      !validPair(space.getAttrOfType<IntegerAttr>("block.owner_tile"), span,
                 pairAxis))
    return space.emitError("invalid projection for cooperative fold");
  if (span &&
      (!panelFolds(&space, p, span.getInt()) || phase == "formed" ||
       phase == "decided" ||
       (!pairAxis && space.getAttr("block.owner_tile") !=
                         Builder(fn.getContext()).getI64IntegerAttr(1))))
    return space.emitError("invalid fold contribution panel");
  if (space
          .walk([&](Operation *v) {
            bool required = span &&
                            v->getBlock() == &space.getRegion(0).front() &&
                            isa<scf::ForOp>(v) && v->hasAttr("kir.reassociate");
            if (pairAxis && required) {
              auto loop = cast<scf::ForOp>(v);
              auto w =
                  space.getAttrOfType<IntegerAttr>("block.owner_tile").getInt();
              if (v->hasAttr("block.uniform_tokens") !=
                  sharedBounds(loop, pairAxis.getInt(), w))
                return WalkResult::interrupt();
              if(v->hasAttr("block.full_panels")!=(v->hasAttr("block.uniform_tokens") && fullFoldPanels(loop,*facts,span.getInt())))
                return WalkResult::interrupt();
              if(v->hasAttr("block.counted_panels")!=countedPanel(v,*facts))return WalkResult::interrupt();
              bool replay =
                  !v->hasAttr("block.uniform_tokens") &&
                  llvm::any_of(
                      loop.getBody()->getTerminator()->getOperands(),
                      [](Value y) {
                        return isa<ShapedType>(y.getType()) &&
                               (!y.getDefiningOp() ||
                                !intentdsl::kir::hasFastMath(
                                    y.getDefiningOp(),
                                    arith::FastMathFlags::afn));
                      });
              if (v->hasAttr("block.repair_masked_nan") != replay)
                return WalkResult::interrupt();
            } else if (v->hasAttr("block.counted_panels") || v->hasAttr("block.full_panels") || v->hasAttr("block.uniform_tokens") ||
                       v->hasAttr("block.repair_masked_nan"))
              return WalkResult::interrupt();
            auto loop = v->getParentOfType<scf::ForOp>();
            int64_t page =
                span && loop && loop->hasAttr("block.contribution_width") &&
                        v->getBlock() == loop.getBody()
                    ? alignedPageExtent(*v, loop, *facts, span.getInt())
                    : 0;
            if (v->getAttr("block.page_extent") !=
                (page ? Attribute(
                            Builder(fn.getContext()).getI64IntegerAttr(page))
                      : Attribute()))
              return WalkResult::interrupt();
            bool pair = pairAxis && loop &&
                        loop->hasAttr("block.contribution_width") &&
                        v->getBlock() == loop.getBody();
            if (v->hasAttr("block.rectangular_tokens") !=
                bool(span && loop &&
                     loop->hasAttr("block.contribution_width") &&
                     v->getBlock() == loop.getBody() &&
                     rectangularTokenLoad(*v, loop)))
              return WalkResult::interrupt();
            if (v->hasAttr("block.pair_dot") != bool(pair && tileDot(v, loop)))
              return WalkResult::interrupt();
            bool update = pair && isa<arith::AddFOp>(v) &&
                          isa<RankedTensorType>(v->getResult(0).getType()) &&
                          llvm::is_contained(
                              loop.getBody()->getTerminator()->getOperands(),
                              v->getResult(0));
            if (v->hasAttr("block.masked_dot_update") != update)
              return WalkResult::interrupt();
            return v->getAttr("block.contribution_width") !=
                           (required ? Attribute(span) : Attribute())
                       ? WalkResult::interrupt()
                       : WalkResult::advance();
          })
          .wasInterrupted())
    return space.emitError("fold panel differs from owner decision");
  auto owners = ownerCount(&space, *facts);
  if (owners < 0)
    return space.emitError("owner space exceeds signed index range");
  auto tile = space.getAttrOfType<IntegerAttr>("block.owner_tile");
  auto choice = space.getAttrOfType<DictionaryAttr>("block.owner_choice");
  if (columns) {
    auto caps = p.services.get<OwnerMachine>();
    auto sh = fn.getArgAttrOfType<DenseI64ArrayAttr>(0, "block.shape");
    if (!tile || tile.getInt() != 1 || !caps || !sh || sh.size() != 3 ||
        sh[1] <= 0 || sh[1] > caps->residentElements / sh[1] ||
        !llvm::isPowerOf2_64(sh[1]))
      return space.emitError(
          "column tile requires a supported private square matrix");
  }
  if (phase == "formed")
    return success(!tile && !choice && !space.hasAttr("block.fold_split"));
  int64_t maxDirectWidth =
      isLargeScalarOwnerSpace(&space, p, owners) ? 512 : 128;
  if (phase == "decided")
    return success(
        !tile && choice && !space.hasAttr("block.fold_split") &&
        legalWidth(choice.getAs<IntegerAttr>("width"), owners,
                   choice.getAs<StringAttr>("memory") == "direct"
                       ? maxDirectWidth
                       : 128) &&
        choice.getAs<StringAttr>("memory") &&
        (choice.getAs<StringAttr>("memory") == "direct" ||
         (choice.getAs<StringAttr>("memory") == "scatter-fold" &&
          choice.getAs<IntegerAttr>("split") &&
          canSplitAssociativeScatter(
              &space, facts, choice.getAs<IntegerAttr>("width").getInt(),
              choice.getAs<IntegerAttr>("split").getInt())) ||
         (choice.getAs<StringAttr>("memory") == "matrix-fold" &&
          validPair(choice.getAs<IntegerAttr>("width"),
                    choice.getAs<IntegerAttr>("span"),
                    choice.getAs<IntegerAttr>("axis"))) ||
         (choice.getAs<StringAttr>("memory") == "token" &&
          choice.getAs<IntegerAttr>("span") &&
          choice.getAs<IntegerAttr>("width").getInt() == 1 &&
          panelFolds(&space, p, choice.getAs<IntegerAttr>("span").getInt())) ||
         (choice.getAs<StringAttr>("memory") == "shared" &&
          legalSharedReads(&space, p, choice)) ||
         (choice.getAs<StringAttr>("memory") == "forward" &&
          canForwardOwners(&space, p)) ||
         (choice.getAs<StringAttr>("memory") == "columns" &&
          columnPrefixes(&space, p) &&
          choice.getAs<IntegerAttr>("width").getInt() == 1) ||
         (choice.getAs<StringAttr>("memory") == "prefetch" &&
          choice.getAs<IntegerAttr>("prefetch") &&
          (choice.getAs<IntegerAttr>("prefetch").getInt() == 4 ||
           choice.getAs<IntegerAttr>("prefetch").getInt() == 16) &&
          canGroupReads(&space, p,
                        choice.getAs<IntegerAttr>("prefetch").getInt()))));
  if (auto fs = space.getAttrOfType<IntegerAttr>("block.fold_split"))
    if (!tile || !canSplitAssociativeScatter(&space, facts, tile.getInt(),
                                             fs.getInt()))
      return space.emitError("invalid associative scatter fold split");
  if (choice || !legalWidth(tile, owners, maxDirectWidth) ||
      space.getAttr("block.owners") !=
          Builder(fn.getContext()).getI64IntegerAttr(owners))
    return space.emitError("unrealized or invalid owner tile");
  for (auto arg : fn.getArguments())
    if (isa<MemRefType>(arg.getType())) {
      auto *layout = facts->arguments[arg.getArgNumber()].getAsObject();
      for (auto key : {"shape", "strides"}) {
        auto attr = fn.getArgAttrOfType<DenseI64ArrayAttr>(
            arg.getArgNumber(), std::string("block.") + key);
        auto *values = layout ? layout->getArray(key) : nullptr;
        if (!values || !attr || attr.size() != values->size())
          return fn.emitError("missing specialized owner layout");
        for (auto [a, v] : llvm::zip(attr.asArrayRef(), *values))
          if (v.getAsInteger() != a)
            return fn.emitError("owner layout differs from the call");
      }
      auto expectedAlias =
          layout ? layout->getInteger("alias_group") : std::nullopt;
      auto actualAlias =
          fn.getArgAttrOfType<IntegerAttr>(arg.getArgNumber(), "block.alias_group");
      if (expectedAlias.has_value() != bool(actualAlias) ||
          (expectedAlias && actualAlias.getInt() != *expectedAlias))
        return fn.emitError("owner alias group differs from the call");
    }
  return success();
}
llvm::Error emitOwnerArtifact(func::FuncOp fn, StringRef src,
                              ArrayRef<llvm::json::Object> natives,
                              unsigned version, llvm::raw_pwrite_stream &out) {
  auto &sp = fn.front().front();
  auto tile = sp.getAttrOfType<IntegerAttr>("block.owner_tile").getInt();
  auto count = sp.getAttrOfType<IntegerAttr>("block.owners").getInt();
  int64_t foldSplit = 1;
  if (auto fs = sp.getAttrOfType<IntegerAttr>("block.fold_split"))
    foldSplit = std::max<int64_t>(1, fs.getInt());
  auto abi = llvm::json::parse(fn->getAttrOfType<StringAttr>("kir.abi"));
  if (!abi)
    return abi.takeError();
  llvm::json::Array rows;
  for (auto [idx, cfg] : llvm::enumerate(natives))
    rows.push_back(llvm::json::Object{{"id", int64_t(idx)},
                                      {"rows", tile},
                                      {"width", 1},
                                      {"native", llvm::json::Object(cfg)}});
  out << llvm::json::Value(llvm::json::Object{
      {"source", src.str()},
      {"entry", "kernel"},
      {"version", version},
      {"abi", std::move(*abi)},
      {"domain", publishDomain(fn)},
      {"guard", llvm::json::Object{{"min_width", 0}, {"max_width", nullptr}}},
      {"grid",
       llvm::json::Array{int64_t(llvm::divideCeil(count, tile)), foldSplit, 1}},
      {"direct_views", true},
      {"rows", std::move(rows)}});
  return llvm::Error::success();
}
llvm::Error emitOwnerArtifact(func::FuncOp fn, StringRef src,
                              llvm::json::Object native, unsigned version,
                              llvm::raw_pwrite_stream &out) {
  return emitOwnerArtifact(fn, src, ArrayRef<llvm::json::Object>{native},
                           version, out);
}
} // namespace intentdsl::block
namespace CI_UNIT_NAMESPACE {
void registerOrdered(Unit &u) {
  using namespace intentdsl;
  DecisionFamily f;
  f.name = "ordered-owners";
  f.applies = [](Operation *o) {
    return o->getName().getStringRef() == "block.owner_space" &&
           !o->hasAttr("block.owner_tile");
  };
  f.read = [](Operation *o) { return o->getAttr("block.owner_choice"); };
  f.write = [](Operation *o, Attribute a) {
    o->setAttr("block.owner_choice", a);
    return llvm::Error::success();
  };
  f.verify = [](Operation *o, Attribute a,
                const TargetProfile &p) -> llvm::Error {
    auto count = block::ownerCount(o, *p.services.get<CallFacts>());
    auto d = dyn_cast<DictionaryAttr>(a);
    if (d && d.getAs<StringAttr>("memory") == "matrix-fold") {
      auto w = d.getAs<IntegerAttr>("width"), s = d.getAs<IntegerAttr>("span"),
           ax = d.getAs<IntegerAttr>("axis");
      if (w && s && ax) {
        auto allowed = block::issueAxis(o, p, w.getInt(), s.getInt());
        if (allowed && *allowed == ax.getInt())
          return llvm::Error::success();
      }
    }
    if (d && d.getAs<StringAttr>("memory") == "token" &&
        d.getAs<IntegerAttr>("width") &&
        d.getAs<IntegerAttr>("width").getInt() == 1 &&
        d.getAs<IntegerAttr>("span") &&
        block::panelFolds(o, p, d.getAs<IntegerAttr>("span").getInt()))
      return llvm::Error::success();
    if (d && d.getAs<StringAttr>("memory") == "scatter-fold" &&
        count >= 0 && block::legalWidth(d.getAs<IntegerAttr>("width"), count) &&
        d.getAs<IntegerAttr>("split") &&
        block::canSplitAssociativeScatter(
            o, p.services.get<CallFacts>(),
            d.getAs<IntegerAttr>("width").getInt(),
            d.getAs<IntegerAttr>("split").getInt()))
      return llvm::Error::success();
    int64_t maxDirectW =
        block::isLargeScalarOwnerSpace(o, p, count) ? 512 : 128;
    if (count < 0 || !d || !d.getAs<StringAttr>("memory") ||
        !block::legalWidth(d.getAs<IntegerAttr>("width"), count,
                           d.getAs<StringAttr>("memory") == "direct"
                               ? maxDirectW
                               : 128) ||
        !(d.getAs<StringAttr>("memory") == "direct" ||
          (d.getAs<StringAttr>("memory") == "shared" &&
           block::legalSharedReads(o, p, d)) ||
          (d.getAs<StringAttr>("memory") == "forward" &&
           block::canForwardOwners(o, p)) ||
          (d.getAs<StringAttr>("memory") == "columns" &&
           d.getAs<IntegerAttr>("width").getInt() == 1 &&
           block::columnPrefixes(o, p)) ||
          (d.getAs<StringAttr>("memory") == "prefetch" &&
           d.getAs<IntegerAttr>("prefetch") &&
           (d.getAs<IntegerAttr>("prefetch").getInt() == 4 ||
            d.getAs<IntegerAttr>("prefetch").getInt() == 16) &&
           block::canGroupReads(o, p,
                                d.getAs<IntegerAttr>("prefetch").getInt()))))
      return llvm::createStringError("invalid owner geometry or storage");
    return llvm::Error::success();
  };
  f.verifyCandidate = [](Operation *, const StructuralCandidate &c,
                         const TargetProfile &) -> llvm::Error {
    auto d = dyn_cast<DictionaryAttr>(c.value);
    auto w = d ? d.getAs<IntegerAttr>("width") : IntegerAttr();
    if (!d || !d.getAs<StringAttr>("memory"))
      return llvm::createStringError("missing owner memory mode");
    if (d.getAs<StringAttr>("memory") == "matrix-fold") {
      auto span = d.getAs<IntegerAttr>("span");
      if (w && span && c.topology == "fold-matrix-issue" &&
          c.id == "pair-" + std::to_string(w.getInt()) + "x" +
                      std::to_string(span.getInt()) &&
          c.preconditions == std::vector<std::string>{"disjoint-owners"})
        return llvm::Error::success();
      return llvm::createStringError("invalid cooperative fold candidate");
    }
    if (d.getAs<StringAttr>("memory") == "token") {
      auto span = d.getAs<IntegerAttr>("span");
      if (w && w.getInt() == 1 && span &&
          c.topology == "parallel-fold-contributions" &&
          c.id == "tokens-" + std::to_string(span.getInt()) &&
          c.preconditions == std::vector<std::string>{"disjoint-owners"})
        return llvm::Error::success();
      return llvm::createStringError("invalid fold candidate");
    }
    if (d.getAs<StringAttr>("memory") == "scatter-fold") {
      auto split = d.getAs<IntegerAttr>("split");
      if (w && split && c.topology == "parallel-scatter-fold" &&
          c.id == "scatter-" + std::to_string(w.getInt()) + "x" +
                      std::to_string(split.getInt()) &&
          c.preconditions ==
              std::vector<std::string>{"disjoint-owners",
                                       "reassociable-scatter"})
        return llvm::Error::success();
      return llvm::createStringError("invalid scatter fold candidate");
    }
    bool forward = d.getAs<StringAttr>("memory") == "forward";
    bool shared = d.getAs<StringAttr>("memory") == "shared";
    bool columns = d && d.getAs<StringAttr>("memory") == "columns";
    auto fetch = d ? d.getAs<IntegerAttr>("prefetch") : IntegerAttr();
    if (!w ||
        c.topology != (fetch     ? "grouped-reads"
                       : shared  ? "shared-owner-reads"
                       : columns ? "column-prefix"
                       : forward ? "resident-owner-slab"
                                 : "ordered-owner-tile") ||
        c.id != (fetch     ? "prefetch" + std::to_string(fetch.getInt()) + "-"
                 : shared  ? "shared-"
                 : columns ? "columns-"
                 : forward ? "resident-"
                           : "owners-") +
                    std::to_string(w.getInt()) ||
        c.preconditions != std::vector<std::string>{"disjoint-owners"})
      return llvm::createStringError("invalid owner candidate");
    return llvm::Error::success();
  };
  u.families.push_back({{std::move(f), "block@decided", "block@realized"},
                        COMPILERINFRA_SOURCE});
  Strategy s;
  s.name = "owner-widths";
  s.family = u.qualify("ordered-owners");
  s.enumerate = [](Operation *o, const TargetProfile &p, AnalysisManager &)
      -> llvm::Expected<std::vector<StructuralCandidate>> {
    auto count = block::ownerCount(o, *p.services.get<CallFacts>());
    std::vector<StructuralCandidate> cs;
    bool eligible = block::canForwardOwners(o, p);
    auto b = Builder(o->getContext());
    if (block::columnPrefixes(o, p)) {
      auto d = b.getDictionaryAttr(
          {b.getNamedAttr("width", b.getI64IntegerAttr(1)),
           b.getNamedAttr("memory", b.getStringAttr("columns"))});
      cs.push_back({"columns-1", "column-prefix", d, {"disjoint-owners"}});
    }
    int64_t maxDirectW =
        block::isLargeScalarOwnerSpace(o, p, count) ? 512 : 128;
    for (int w : {1, 4, 8, 16, 32, 64, 128, 256, 512}) {
      auto a = b.getI64IntegerAttr(w);
      if (!block::legalWidth(a, count, maxDirectW))
        continue;
      if (w <= 32)
        for (auto group : {4, 16})
          if (block::canGroupReads(o, p, group)) {
            auto d = b.getDictionaryAttr(
                {b.getNamedAttr("width", a),
                 b.getNamedAttr("memory", b.getStringAttr("prefetch")),
                 b.getNamedAttr("prefetch", b.getI64IntegerAttr(group))});
            cs.push_back(
                {"prefetch" + std::to_string(group) + "-" + std::to_string(w),
                 "grouped-reads",
                 d,
                 {"disjoint-owners"}});
          }
      for (bool local : {false, true}) {
        if (local && (!eligible || w > 8))
          continue;
        auto d = b.getDictionaryAttr(
            {b.getNamedAttr("width", a),
             b.getNamedAttr("memory",
                            b.getStringAttr(local ? "forward" : "direct"))});
        cs.push_back({(local ? "resident-" : "owners-") + std::to_string(w),
                      local ? "resident-owner-slab" : "ordered-owner-tile",
                      d,
                      {"disjoint-owners"}});
      }
      if (w >= 32 && w <= 128) {
        for (int64_t split : {16, 64, 256, 512, 1024, 2048}) {
          if (!block::canSplitAssociativeScatter(
                  o, p.services.get<CallFacts>(), w, split))
            continue;
          auto sd = b.getDictionaryAttr(
              {b.getNamedAttr("width", a),
               b.getNamedAttr("memory", b.getStringAttr("scatter-fold")),
               b.getNamedAttr("split", b.getI64IntegerAttr(split))});
          cs.push_back(
              {"scatter-" + std::to_string(w) + "x" + std::to_string(split),
               "parallel-scatter-fold",
               sd,
               {"disjoint-owners", "reassociable-scatter"}});
        }
      }
    }
    return cs;
  };
  u.strategies.push_back({std::move(s), COMPILERINFRA_SOURCE});
  Realizer r;
  r.name = "map-owner-tile";
  r.family = u.qualify("ordered-owners");
  r.apply = [](Operation *o, Attribute a, const TargetProfile &p,
               RewriterBase &b) {
    auto f = p.services.get<CallFacts>();
    auto fn = o->getParentOfType<func::FuncOp>();
    for (auto arg : fn.getArguments())
      if (isa<MemRefType>(arg.getType())) {
        auto *obj = f->arguments[arg.getArgNumber()].getAsObject();
        for (auto key : {"shape", "strides"}) {
          SmallVector<int64_t> ds;
          for (auto &v : *obj->getArray(key))
            ds.push_back(*v.getAsInteger());
          fn.setArgAttr(arg.getArgNumber(), std::string("block.") + key,
                        b.getDenseI64ArrayAttr(ds));
        }
        if (auto ag = obj->getInteger("alias_group"))
          fn.setArgAttr(arg.getArgNumber(), "block.alias_group",
                        b.getI64IntegerAttr(*ag));
      }
    auto d = cast<DictionaryAttr>(a);
    o->removeAttr("block.owner_choice");
    o->setAttr("block.owner_tile", d.getAs<IntegerAttr>("width"));
    o->setAttr("block.owners", b.getI64IntegerAttr(block::ownerCount(o, *f)));
    if (d.getAs<StringAttr>("memory") == "scatter-fold")
      o->setAttr("block.fold_split", d.get("split"));
    if (d.getAs<StringAttr>("memory") == "matrix-fold")
      o->setAttr("block.pair_axis", d.get("axis"));
    if (d.getAs<StringAttr>("memory") == "token" ||
        d.getAs<StringAttr>("memory") == "matrix-fold") {
      o->setAttr("block.token_span", d.get("span"));
      if(o->hasAttr("block.pair_axis"))
        block::partitionOwnerFolds(o,d.getAs<IntegerAttr>("axis").getInt(),d.getAs<IntegerAttr>("width").getInt(),d.getAs<IntegerAttr>("span").getInt());
      for (auto loop : o->getRegion(0).front().getOps<scf::ForOp>())
        if (loop->hasAttr("kir.reassociate")) {
          loop->setAttr("block.contribution_width", d.get("span"));
          // A second traversal rereads the score inputs. One panel already
          // fits locally; streaming several panels can amortize rescaling.
          auto lo = block::panelRange(loop.getLowerBound(), *o, *f);
          auto hi = block::panelRange(loop.getUpperBound(), *o, *f);
          if (fn->getAttrOfType<StringAttr>("kir.normalization") == "online-eligible" &&
              !o->hasAttr("block.fold_split") && lo && hi &&
              (__int128(hi->hi) - lo->lo) > d.getAs<IntegerAttr>("span").getInt())
            loop->setAttr("block.online_normalization", b.getUnitAttr());
          for (auto &op : loop.getBody()->without_terminator()) {
            if (block::rectangularTokenLoad(op, loop))
              op.setAttr("block.rectangular_tokens", b.getUnitAttr());
            if (auto page = block::alignedPageExtent(
                    op, loop, *f, d.getAs<IntegerAttr>("span").getInt()))
              op.setAttr("block.page_extent", b.getI64IntegerAttr(page));
          }
          if (o->hasAttr("block.pair_axis")) {
            if (block::countedPanel(loop, *f))
              loop->setAttr("block.counted_panels", b.getUnitAttr());
            if (block::sharedBounds(loop, d.getAs<IntegerAttr>("axis").getInt(),
                                    d.getAs<IntegerAttr>("width").getInt())) {
              loop->setAttr("block.uniform_tokens", b.getUnitAttr());
              if (block::fullFoldPanels(
                      loop, *f, d.getAs<IntegerAttr>("span").getInt()))
                loop->setAttr("block.full_panels", b.getUnitAttr());
            } else if (llvm::any_of(
                         loop.getBody()->getTerminator()->getOperands(),
                         [](Value y) {
                           return isa<ShapedType>(y.getType()) &&
                                  (!y.getDefiningOp() ||
                                   !intentdsl::kir::hasFastMath(
                                       y.getDefiningOp(),
                                       arith::FastMathFlags::afn));
                         }))
              loop->setAttr("block.repair_masked_nan", b.getUnitAttr());
            for (auto &op : loop.getBody()->without_terminator())
              if (block::tileDot(&op, loop))
                op.setAttr("block.pair_dot", b.getUnitAttr());
            for (auto next : loop.getBody()->getTerminator()->getOperands())
              if (isa<ShapedType>(next.getType()))
                next.getDefiningOp()->setAttr("block.masked_dot_update",
                                              b.getUnitAttr());
          }
        }
    }
    if (d.getAs<StringAttr>("memory") == "shared") {
      auto reads = block::sharedReads(o, *f);
      o->setAttr("block.group_axis", d.get("axis"));
      for (auto *s : reads->slices)
        s->setAttr("block.broadcast_read", b.getUnitAttr());
    }
    if (d.getAs<StringAttr>("memory") == "forward")
      o->setAttr("block.forward_slab", b.getUnitAttr());
    if (d.getAs<StringAttr>("memory") == "columns") {
      o->setAttr("block.column_prefixes", b.getUnitAttr());
      block::realizeColumns(fn);
    }
    if (auto group = d.getAs<IntegerAttr>("prefetch"))
      block::groupReads(fn, group.getInt());
    return llvm::Error::success();
  };
  u.realizers.push_back({std::move(r), COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
