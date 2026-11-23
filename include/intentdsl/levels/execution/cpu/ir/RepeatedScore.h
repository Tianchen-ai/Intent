#pragma once
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
namespace intentdsl::cpu {
// The 32 KiB per-task budget includes all owners of this candidate.
inline constexpr int64_t batchScoreSlots = 32 * 1024 / sizeof(float);
struct RepeatedScore {
  mlir::Operation *record, *replay;
};
inline std::optional<RepeatedScore> repeatedScore(mlir::Operation &space) {
  using namespace mlir;
  auto &bb = space.getRegion(0).front();
  auto loops = llvm::to_vector(bb.getOps<scf::ForOp>());
  if (loops.size() != 2)
    return {};
  auto a = loops[0], b = loops[1];
  if (a.getNumResults() != 1 || !a.getResult(0).getType().isF32() ||
      a.getLowerBound() != b.getLowerBound() ||
      a.getUpperBound() != b.getUpperBound() ||
      !matchPattern(a.getStep(), m_One()) ||
      !matchPattern(b.getStep(), m_One()))
    return {};
  // Every load of the matched expression must see the same input in both folds.
  for (Operation *op = a; op != b; op = op->getNextNode()) {
    if (op->walk([](Operation *v) {
            if (v->getNumRegions() ||
                v->getName().stripDialect() == "slice_read" ||
                isa<memref::LoadOp>(v))
              return WalkResult::advance();
            return isMemoryEffectFree(v) ? WalkResult::advance()
                                         : WalkResult::interrupt();
          }).wasInterrupted())
      return {};
  }
  auto peak = a.getBody()
                  ->getTerminator()
                  ->getOperand(0)
                  .getDefiningOp<arith::MaxNumFOp>();
  if (!peak || peak.getLhs() != a.getRegionIterArgs()[0])
    return {};
  auto score = peak.getRhs();
  auto *producer = score.getDefiningOp();
  if (!producer || producer->getBlock() != a.getBody())
    return {};
  auto same = [&](auto &self, Value x, Value y) -> bool {
    if (x == y || (x == a.getInductionVar() && y == b.getInductionVar()))
      return true;
    auto *p = x.getDefiningOp(), *q = y.getDefiningOp();
    if (!p || !q || p->getBlock() != a.getBody() ||
        q->getBlock() != b.getBody() || p->getName() != q->getName() ||
        p->getNumRegions() || p->getNumResults() != 1 ||
        q->getNumResults() != 1 || x.getType() != y.getType() ||
        p->getAttrs() != q->getAttrs() ||
        p->getPropertiesAsAttribute() != q->getPropertiesAsAttribute() ||
        p->getNumOperands() != q->getNumOperands())
      return false;
    return llvm::all_of(
        llvm::zip(p->getOperands(), q->getOperands()), [&](auto pair) {
          return self(self, std::get<0>(pair), std::get<1>(pair));
        });
  };
  for (auto sub : b.getBody()->getOps<arith::SubFOp>()) {
    if (sub.getRhs() == a.getResult(0) && same(same, score, sub.getLhs()))
      return RepeatedScore{producer, sub.getLhs().getDefiningOp()};
  }
  return {};
}
} // namespace intentdsl::cpu
