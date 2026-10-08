#include "PassDetail.h"
#include "Intent/Dialect/DSA/Transforms/LocalSupplyRelations.h"

namespace intent::bangc {
namespace {

bool positiveZero(Value value) {
  FloatAttr constant;
  return value && matchPattern(value, m_Constant(&constant)) &&
      constant.getValue().isZero() && !constant.getValue().isNegative();
}

bool hasFirstIteration(scf::ForOp loop, dsa::LocalSupplyRelations &relations) {
  if (!loop.getInductionVar().getType().isIndex()) return false;
  auto lower = relations.interval(loop.getLowerBound());
  auto upper = relations.interval(loop.getUpperBound());
  auto step = relations.interval(loop.getStep());
  int64_t next;
  if (!lower || !upper || !step || step->first <= 0 ||
      llvm::AddOverflow(lower->second, step->second, next))
    return false;
  if (lower->second < upper->first) return true;
  auto difference = dyn_cast<AffineConstantExpr>(
      relations.difference(loop.getUpperBound(), loop.getLowerBound()));
  return difference && difference.getValue() > 0;
}

bool firstWrite(dsa::MatrixTileOp matrix, scf::ForOp loop,
                dsa::StorageAnalysis &storage, DominanceInfo &dominance,
                dsa::UniformMemoryAnalysis &uniforms) {
  Value accumulator = matrix.getAccumulator();
  if (!loop.getInitArgs().empty() || matrix->getBlock() != loop.getBody() ||
      !dominance.properlyDominates(accumulator, loop)) return false;
  // No earlier operation of this iteration may have changed the seed. Reads
  // are retained; ordinary dead-write elimination decides whether Fill dies.
  for (Operation &operation : *loop.getBody()) {
    if (&operation == matrix.getOperation()) break;
    if (!storage.preservesContents(&operation, accumulator)) return false;
  }
  return positiveZero(uniforms.read(accumulator, loop));
}

void initializeFirstIteration(scf::ForOp loop, dsa::MatrixTileOp matrix) {
  OpBuilder builder(loop);
  Value next = builder.createOrFold<arith::AddIOp>(
      loop.getLoc(), loop.getLowerBound(), loop.getStep());
  IRMapping mapping;
  mapping.map(loop.getInductionVar(), loop.getLowerBound());
  for (Operation &operation : loop.getBody()->without_terminator()) {
    Operation *cloned = builder.clone(operation, mapping);
    if (&operation == matrix.getOperation())
      cast<dsa::MatrixTileOp>(cloned).setAccumulate(false);
  }
  loop.setLowerBound(next);
}

} // namespace

bool initializeMatrixStorage(func::FuncOp function) {
  dsa::StorageAnalysis storage(function);
  DominanceInfo dominance(function);
  dsa::LocalSupplyRelations relations(function);
  dsa::UniformMemoryAnalysis uniforms(function, storage);
  SmallVector<dsa::MatrixTileOp> matrices;
  function.walk([&](dsa::MatrixTileOp matrix) { matrices.push_back(matrix); });
  for (dsa::MatrixTileOp matrix : matrices) {
    if (!matrix.getAccumulate()) continue;
    if (positiveZero(uniforms.read(matrix.getAccumulator(), matrix))) {
      matrix.setAccumulate(false);
      return true;
    }
    auto loop = dyn_cast<scf::ForOp>(matrix->getParentOp());
    if (!loop || !hasFirstIteration(loop, relations)) continue;
    auto effects = storage.effects(loop);
    if (!effects.complete || effects.ordered ||
        !firstWrite(matrix, loop, storage, dominance, uniforms)) continue;
    initializeFirstIteration(loop, matrix);
    return true;
  }
  return false;
}

} // namespace intent::bangc
