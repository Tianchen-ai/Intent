#include "intentdsl/levels/algorithm/kir/ir/FastMath.h"
#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
#include "intentdsl/levels/execution/cpu/ir/Task.h"
#include "intentdsl/levels/execution/cpu/ir/Traversal.h"
#include <algorithm>
#include <climits>
namespace CI_UNIT_NAMESPACE {
using namespace mlir;
using namespace compilerinfra;
void registerSegment(Unit &u) {
  Strategy s;
  s.name = "segment-reduction";
  s.family = u.qualify("partition");
  std::string sid = u.qualify("segment-reduction");
  s.enumerate = [sid](Operation *op, const TargetProfile &p,
                      AnalysisManager &)
      -> llvm::Expected<std::vector<StructuralCandidate>> {
    Operation *segOp = nullptr;
    op->walk([&](Operation *v) {
      if (v->getName().stripDialect() == "ordered_sum") {
        segOp = v;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (!segOp)
      return std::vector<StructuralCandidate>{};
    auto *machine = p.services.get<intentdsl::cpu::RowMachine>();
    if (!machine)
      return llvm::createStringError("missing task capacity for segment outputs");
    Builder b(op->getContext());
    ArrayAttr rows = machine->configurations(op->getContext(), false);
    bool flatRank1 = cast<MemRefType>(op->getOperand(0).getType()).getRank() == 1;
    bool reassoc = intentdsl::kir::hasFastMath(
        segOp, arith::FastMathFlags::reassoc);
    std::vector<StructuralCandidate> out;
    if (reassoc && machine->segmentWidths.size() > 1 &&
        segOp->getNumRegions() == 1 && !segOp->getRegion(0).empty()) {
      int64_t minExt = INT64_MAX;
      bool allRank1 = true;
      auto *facts = p.services.get<intentdsl::CallFacts>();
      if (facts) {
        for (Operation &scOp : segOp->getRegion(0).front()) {
          if (scOp.getName().stripDialect() != "member_load")
            continue;
          auto vt = dyn_cast<MemRefType>(scOp.getOperand(0).getType());
          auto *shape = facts->shapeOf(scOp.getOperand(0));
          if (!vt || vt.getRank() != 1 || !shape || shape->size() != 1) {
            allRank1 = false;
            break;
          }
          if (auto ext = (*shape)[0].getAsInteger())
            minExt = std::min(minExt, *ext);
          else {
            allRank1 = false;
            break;
          }
        }
      } else {
        allRank1 = false;
      }
      int64_t baseW = machine->segmentWidths.front();
      for (int64_t w : machine->segmentWidths) {
        if (w > baseW && (!allRank1 || minExt == INT64_MAX || minExt <= w))
          continue;
        NamedAttrList layout(intentdsl::cpu::partition(b, "source", rows));
        if (flatRank1)
          layout.set("member_tasks", b.getUnitAttr());
        layout.set("segment_width", b.getI64IntegerAttr(w));
        std::string id =
            w == baseW ? sid : (sid + "-w" + std::to_string(w));
        out.push_back(
            {id, sid, layout.getDictionary(op->getContext()), {}});
      }
    }
    if (out.empty()) {
      NamedAttrList layout(intentdsl::cpu::partition(b, "source", rows));
      if (flatRank1)
        layout.set("member_tasks", b.getUnitAttr());
      out.push_back({sid, sid, layout.getDictionary(op->getContext()), {}});
    }
    return out;
  };
  u.strategies.push_back({std::move(s), COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
