#include "Plan.h"
#include "intentdsl/levels/algorithm/kir/ir/Matrix.h"
#include "intentdsl/levels/algorithm/kir/ir/Segment.h"
#include "intentdsl/levels/storage/dsa/ir/Owners.h"
#include "llvm/Support/MathExtras.h"

namespace intentdsl::dsa {
using namespace mlir;

// A DMA slot holds a complete group. Vector temporaries and collective scratch
// are reused after each member's compute and drain.
struct Footprint {
  int64_t inputBytes = 0, vectorBytes = 0, collectiveBytes = 0;
  static Footprint of(Block &body, const LocalSupply &c, bool reuse=false) {
    auto reductionScratch=c.reductionScratchPerColumn;
    Footprint f;
    for (auto &op : body) {
      auto code = op.getName().stripDialect();
      if (code == "await" || code == "active" || code == "window" ||
          code == "member_axis" || code == "slab_axis" || code == "rows" || code == "columns")
        continue;
      if (code == "reduce") {
        f.collectiveBytes = reductionScratch;
      } else if (op.getNumResults() == 1) {
        if(code=="scan_sum")f.collectiveBytes=reductionScratch;
        if(code=="atomic_add_at")f.collectiveBytes=std::max<int64_t>(f.collectiveBytes,8);
        if(code=="key_counts" || code=="sum_columns")
          f.collectiveBytes=std::max<int64_t>(f.collectiveBytes,cast<MemRefType>(op.getOperand(0).getType()).getElementTypeBitWidth()/8);
        if(code=="segment_fold" && kir::contiguousFeatureRead(&op))
          f.collectiveBytes=std::max<int64_t>(f.collectiveBytes,c.featurePanelMembers*int64_t(sizeof(float)));
        auto t = dyn_cast<RankedTensorType>(op.getResult(0).getType());
        if (!t || t.getRank() < 1 || !t.getElementType().isIntOrFloat())
          continue;
        // Rank-one calculation results in the block contract are per-row
        // summaries. Projected coordinates have no local storage; copied
        // fragments do.
        if (t.getRank() == 1 && code != "load" && code != "copy_in")
          continue;
        if (code == "load" || code == "copy_in" || code=="indexed_tile_read")
          f.inputBytes += llvm::divideCeil(t.getElementTypeBitWidth(), 8u);
        else f.vectorBytes += llvm::divideCeil(t.getElementTypeBitWidth(), 8u);
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
static llvm::Expected<DictionaryAttr> planStorage(func::FuncOp fn, const LocalSupply &c,
                                    int64_t depth,int64_t participants,bool persistent, bool reuse) {
  if(persistent && !reusableInput(fn))return llvm::createStringError("persistent panel requires an immutable row-invariant input");
  if(participants<=0 || participants>c.consumers)return llvm::createStringError("invalid supply work partition");
  Builder b(fn.getContext());
  // A group of equally shaped lane-wise expressions can keep all member
  // results in private storage until the final rectangular transfer.
  bool wide = fn->hasAttr("kir.iteration") &&
              fn->getAttrOfType<StringAttr>("block.topology")=="resident";
  for (auto t:fn.getArgumentTypes()) {
    auto view=dyn_cast<MemRefType>(t);
    wide &= view && view.getRank()>2 && view.getElementType().isF16();
  }
  unsigned arithmetic=0;
  fn.walk([&](Operation *op){
    auto code=op->getName().stripDialect();
    if(code=="add" || code=="sub" || code=="mul") {++arithmetic;return;}
    if(code=="reduce" || (op->getName().getDialectNamespace()=="block" &&
       code!="rows" && code!="columns" && code!="pid" && code!="active" &&
       code!="load" && code!="store" && code!="zero")) wide=false;
  });
  wide &= arithmetic!=0;
  SmallVector<Footprint> phases;
  bool indexed=false;
  fn.walk([&](Operation *op) {
    indexed |= op->getName().getStringRef()=="block.indexed_tile_read" || op->getName().getStringRef()=="dsa.lookup_axis";
    if (op->getName().getStringRef() == "dsa.phase") {
      auto f =
          Footprint::of(op->getRegion(1).front(),c,reuse);
      f.inputBytes =
          Footprint::of(op->getRegion(0).front(),c)
              .inputBytes;
      phases.push_back(f);
    }
  });
  if (phases.empty() &&
      fn->getAttrOfType<StringAttr>("block.topology") == "resident")
    phases.push_back(Footprint::of(fn.front(),c,reuse));
  else
    fn.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "block.tiles")
        phases.push_back(Footprint::of(op->getRegion(0).front(),
                                       c,reuse));
    });
  auto configs = fn->getAttrOfType<ArrayAttr>("block.configs");
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
    // A pitched row must align even the narrowest supported element (f16).
    auto pitch = llvm::alignTo(w, c.alignment / 2);
    int64_t bytes = 0;
    for (auto p : phases)
      bytes =
          std::max<int64_t>(bytes, pitch * (depth * r * p.inputBytes +
                                            (wide ? r : 1)*p.vectorBytes + p.collectiveBytes));
    if(indexed)bytes=llvm::alignTo(bytes,c.alignment)+llvm::alignTo(c.indexScratchPerMember*r,c.alignment);
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
  if(fn->getAttrOfType<StringAttr>("block.topology")=="stream" && phases.size()==1 &&
     fn.walk([](Operation *op){return op->getName().stripDialect()=="segment_fold"?WalkResult::interrupt():WalkResult::advance();}).wasInterrupted() &&
     !fn.walk([](Operation *op){auto n=op->getName().stripDialect();return n=="reduce" || n=="combine" || n=="owner_store" || n=="summary_reduce"?WalkResult::interrupt():WalkResult::advance();}).wasInterrupted())
    out.set("independent_columns",b.getUnitAttr());
  if(reuse)out.set("reuse_temporaries", b.getUnitAttr());
  out.set("group_temporaries",b.getBoolAttr(wide));
  out.set("persistent",b.getBoolAttr(persistent));
  out.set("participants",b.getI64IntegerAttr(participants));
  out.set("budget", b.getI64IntegerAttr(std::min(c.budget, c.capacity)));
  out.set("depth", b.getI32IntegerAttr(depth));
  out.set("rows", b.getArrayAttr(accepted));
  out.set("rejected", b.getArrayAttr(rejected));
  return out.getDictionary(fn.getContext());
}
llvm::Expected<DictionaryAttr> plan(func::FuncOp fn, const LocalSupply &c,
                                    int64_t depth,int64_t participants,bool persistent) {
  auto ordinary=planStorage(fn,c,depth,participants,persistent,false);
  if(!ordinary || !ordinary->getAs<ArrayAttr>("rows").empty())return ordinary;
  auto compact=planStorage(fn,c,depth,participants,persistent,true);
  if(!compact || !compact->getAs<ArrayAttr>("rows").empty())return compact;
  return ordinary;
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
    if (!d || !facts || !d.getAs<IntegerAttr>("depth") || !d.getAs<IntegerAttr>("participants") || !d.getAs<BoolAttr>("persistent"))
      return llvm::createStringError("missing private supply facts");
    auto check = intentdsl::dsa::plan(cast<func::FuncOp>(op), *facts,
                                      d.getAs<IntegerAttr>("depth").getInt(),d.getAs<IntegerAttr>("participants").getInt(),d.getAs<BoolAttr>("persistent") && d.getAs<BoolAttr>("persistent").getValue());
    if (!check)
      return check.takeError();
    if(!d.getAs<ArrayAttr>("rows") || d.getAs<ArrayAttr>("rows").empty())return llvm::createStringError("no legal private storage row or DMA completion");
    if (*check != d)
      return llvm::createStringError(
          "supply row costs, exclusion reasons or wait capability changed");
    return llvm::Error::success();
  };
  family.verifyCandidate=[](Operation*,const StructuralCandidate &c,const TargetProfile &)->llvm::Error{
    auto d=cast<DictionaryAttr>(c.value);auto depth=d.getAs<IntegerAttr>("depth").getInt(),count=d.getAs<IntegerAttr>("participants").getInt();
    std::string topology=depth==1?"direct":"double-buffer";
    if(d.getAs<BoolAttr>("persistent").getValue())topology+="-shared";
    if(c.topology!=topology || c.id!=topology+"-"+std::to_string(count) || c.preconditions!=std::vector<std::string>{"legal-private-supply"})return llvm::createStringError("supply topology or participants mismatch");
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
      std::vector<StructuralCandidate> cs;
      for(bool persistent:{false,true})for(int64_t n=facts->consumers;n>=1;n/=2){
        if(persistent && !intentdsl::dsa::reusableInput(cast<func::FuncOp>(op)))continue;
        auto d=intentdsl::dsa::plan(cast<func::FuncOp>(op),*facts,depth,n,persistent);
        if(!d)return d.takeError();std::string topology=depth==1?"direct":"double-buffer";if(persistent)topology+="-shared";
        cs.push_back({topology+"-"+std::to_string(n),topology,*d,{"legal-private-supply"}});
      }return cs;
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
