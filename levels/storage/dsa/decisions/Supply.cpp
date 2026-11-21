#include "Plan.h"
#include "llvm/Support/MathExtras.h"

namespace intentdsl::dsa {
using namespace mlir;

// A DMA slot holds a complete group. Vector temporaries and collective scratch
// are reused after each member's compute and drain.
struct Footprint {
  int64_t inputBytes = 0, vectorBytes = 0, collectiveBytes = 0;
  static Footprint of(Block &body, int64_t reductionScratch) {
    Footprint f;
    for (auto &op : body) {
      auto code = op.getName().stripDialect();
      if (code == "await" || code == "active" || code == "window" ||
          code == "member_axis" || code == "rows" || code == "columns")
        continue;
      if (code == "reduce") {
        f.collectiveBytes = reductionScratch;
      } else if (op.getNumResults() == 1) {
        auto t = dyn_cast<RankedTensorType>(op.getResult(0).getType());
        if (!t || t.getRank() < 1 || !t.getElementType().isIntOrFloat())
          continue;
        // Rank-one calculation results in the block contract are per-row
        // summaries. Projected coordinates have no local storage; copied
        // fragments do.
        if (t.getRank() == 1 && code != "load" && code != "copy_in")
          continue;
        if (code == "load" || code == "copy_in")
          f.inputBytes += llvm::divideCeil(t.getElementTypeBitWidth(), 8u);
        else
          f.vectorBytes += llvm::divideCeil(t.getElementTypeBitWidth(), 8u);
      }
    }
    return f;
  }
};

llvm::Expected<DictionaryAttr> plan(func::FuncOp fn, const LocalSupply &c,
                                    int64_t depth) {
  Builder b(fn.getContext());
  SmallVector<Footprint> phases;
  fn.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "dsa.phase") {
      auto f =
          Footprint::of(op->getRegion(1).front(), c.reductionScratchPerColumn);
      f.inputBytes =
          Footprint::of(op->getRegion(0).front(), c.reductionScratchPerColumn)
              .inputBytes;
      phases.push_back(f);
    }
  });
  if (phases.empty() &&
      fn->getAttrOfType<StringAttr>("block.topology") == "resident")
    phases.push_back(Footprint::of(fn.front(), c.reductionScratchPerColumn));
  else
    fn.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "block.tiles")
        phases.push_back(Footprint::of(op->getRegion(0).front(),
                                       c.reductionScratchPerColumn));
    });
  auto configs = fn->getAttrOfType<ArrayAttr>("block.configs");
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
    // A pitched row must align even the narrowest supported element (f16).
    auto pitch = llvm::alignTo(w, c.alignment / 2);
    int64_t bytes = 0;
    for (auto p : phases)
      bytes =
          std::max<int64_t>(bytes, pitch * (depth * r * p.inputBytes +
                                            p.vectorBytes + p.collectiveBytes));
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
  if (accepted.empty())
    return llvm::createStringError(
        "all parameter rows exceed the supply budget or lack local completion");
  NamedAttrList out;
  out.set("budget", b.getI64IntegerAttr(std::min(c.budget, c.capacity)));
  out.set("depth", b.getI32IntegerAttr(depth));
  out.set("rows", b.getArrayAttr(accepted));
  out.set("rejected", b.getArrayAttr(rejected));
  return out.getDictionary(fn.getContext());
}
} // namespace intentdsl::dsa
namespace CI_UNIT_NAMESPACE {
using namespace mlir;
using namespace compilerinfra;
void supply(Unit &u) {
  DecisionFamily family;
  family.name = "supply";
  family.applies = [](Operation *op) { return isa<func::FuncOp>(op); };
  family.read = [](Operation *op) { return op->getAttr("dsa.pending"); };
  family.write = [](Operation *op, Attribute a) {
    op->setAttr("dsa.pending", a);
    return llvm::Error::success();
  };
  family.verify = [](Operation *op, Attribute a,
                     const TargetProfile &p) -> llvm::Error {
    auto facts = p.services.get<intentdsl::dsa::LocalSupply>();
    auto d = dyn_cast<DictionaryAttr>(a);
    if (!d || !facts || !d.getAs<IntegerAttr>("depth"))
      return llvm::createStringError("missing private supply facts");
    auto check = intentdsl::dsa::plan(cast<func::FuncOp>(op), *facts,
                                      d.getAs<IntegerAttr>("depth").getInt());
    if (!check)
      return check.takeError();
    if (*check != d)
      return llvm::createStringError(
          "supply row costs, exclusion reasons or wait capability changed");
    return llvm::Error::success();
  };
  u.families.push_back({{std::move(family), "dsa@formed", "dsa@supplied"},
                        COMPILERINFRA_SOURCE});
  for (int depth : {1, 2}) {
    Strategy s;
    s.name = depth == 1 ? "direct" : "double-buffer";
    s.family = u.qualify("supply");
    s.propose = [depth](Operation *op, const TargetProfile &p,
                        AnalysisManager &) -> llvm::Expected<Attribute> {
      auto facts = p.services.get<intentdsl::dsa::LocalSupply>();
      if (!facts)
        return llvm::createStringError(
            "target has no private transfer resources");
      auto d = intentdsl::dsa::plan(cast<func::FuncOp>(op), *facts, depth);
      if (!d)
        return d.takeError();
      return Attribute(*d);
    };
    u.strategies.push_back({std::move(s), COMPILERINFRA_SOURCE});
  }
  Realizer r;
  r.name = "stage-transfers";
  r.family = u.qualify("supply");
  r.apply = intentdsl::dsa::stageTransfers;
  u.realizers.push_back({std::move(r), COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
