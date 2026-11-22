#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "intentdsl/levels/execution/block/ir/Owners.h"
#include "mlir/IR/PatternMatch.h"
using namespace mlir;
using namespace compilerinfra;
namespace intentdsl::block {
static bool legalWidth(IntegerAttr width, int64_t owners) {
  return width && width.getInt() > 0 && width.getInt() <= 128 &&
         llvm::isPowerOf2_64(width.getInt()) &&
         width.getInt() <= std::max<int64_t>(1, llvm::PowerOf2Ceil(owners));
}
LogicalResult verifyOwnerMap(func::FuncOp fn, const TargetProfile &p,
                             StringRef dialect, StringRef phase) {
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
    return success(!tile && !choice);
  if (phase == "decided")
    return success(
        !tile && choice &&
        legalWidth(choice.getAs<IntegerAttr>("width"), owners) &&
        choice.getAs<StringAttr>("memory") &&
        (choice.getAs<StringAttr>("memory") == "direct" ||
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
  if (choice || !legalWidth(tile, owners) ||
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
    }
  return success();
}
llvm::Error emitOwnerArtifact(func::FuncOp fn, StringRef src,
                              llvm::json::Object native, unsigned version,
                              llvm::raw_pwrite_stream &out) {
  auto &sp = fn.front().front();
  auto tile = sp.getAttrOfType<IntegerAttr>("block.owner_tile").getInt();
  auto count = sp.getAttrOfType<IntegerAttr>("block.owners").getInt();
  auto abi = llvm::json::parse(fn->getAttrOfType<StringAttr>("kir.abi"));
  if (!abi)
    return abi.takeError();
  out << llvm::json::Value(llvm::json::Object{
      {"source", src.str()},
      {"entry", "kernel"},
      {"version", version},
      {"abi", std::move(*abi)},
      {"domain", publishDomain(fn)},
      {"guard", llvm::json::Object{{"min_width", 0}, {"max_width", nullptr}}},
      {"grid", llvm::json::Array{int64_t(llvm::divideCeil(count, tile)), 1, 1}},
      {"direct_views", true},
      {"rows",
       llvm::json::Array{llvm::json::Object{{"id", 0},
                                            {"rows", tile},
                                            {"width", 1},
                                            {"native", std::move(native)}}}}});
  return llvm::Error::success();
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
    if (count < 0 || !d || !d.getAs<StringAttr>("memory") ||
        !block::legalWidth(d.getAs<IntegerAttr>("width"), count) ||
        !(d.getAs<StringAttr>("memory") == "direct" ||
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
    bool forward = d.getAs<StringAttr>("memory") == "forward";
    bool columns = d && d.getAs<StringAttr>("memory") == "columns";
    auto fetch = d ? d.getAs<IntegerAttr>("prefetch") : IntegerAttr();
    if (!w ||
        c.topology != (fetch     ? "grouped-reads"
                       : columns ? "column-prefix"
                       : forward ? "resident-owner-slab"
                                 : "ordered-owner-tile") ||
        c.id != (fetch     ? "prefetch" + std::to_string(fetch.getInt()) + "-"
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
    for (int w : {1, 4, 8, 32, 64, 128}) {
      auto a = b.getI64IntegerAttr(w);
      if (!block::legalWidth(a, count))
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
      if (isa<MemRefType>(arg.getType()))
        for (auto key : {"shape", "strides"}) {
          SmallVector<int64_t> ds;
          for (auto &v :
               *f->arguments[arg.getArgNumber()].getAsObject()->getArray(key))
            ds.push_back(*v.getAsInteger());
          fn.setArgAttr(arg.getArgNumber(), std::string("block.") + key,
                        b.getDenseI64ArrayAttr(ds));
        }
    auto d = cast<DictionaryAttr>(a);
    if (d.getAs<StringAttr>("memory") == "forward")
      o->setAttr("block.forward_slab", b.getUnitAttr());
    if (d.getAs<StringAttr>("memory") == "columns") {
      o->setAttr("block.column_prefixes", b.getUnitAttr());
      block::realizeColumns(fn);
    }
    if (auto group = d.getAs<IntegerAttr>("prefetch"))
      block::groupReads(fn, group.getInt());
    o->removeAttr("block.owner_choice");
    o->setAttr("block.owner_tile", d.getAs<IntegerAttr>("width"));
    o->setAttr("block.owners", b.getI64IntegerAttr(block::ownerCount(o, *f)));
    return llvm::Error::success();
  };
  u.realizers.push_back({std::move(r), COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
