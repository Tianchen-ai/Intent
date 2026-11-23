#include "intentdsl/levels/execution/block/ir/FoldIssue.h"
using namespace mlir;
namespace intentdsl::block {
namespace {
Value widened(Value v) {
  auto cast = v.getDefiningOp<arith::ExtFOp>();
  return cast ? cast.getIn() : Value();
}
bool sharedCoordinate(Value v, BlockArgument owner, scf::ForOp loop,
                      int64_t width) {
  if (v == owner)
    return false;
  if (v == loop.getInductionVar() ||
      loop.isDefinedOutsideOfLoop(v) && isa<BlockArgument>(v))
    return true;
  auto *op = v.getDefiningOp();
  if (!op)
    return false;
  if (auto branch = dyn_cast<scf::IfOp>(op)) {
    return sharedCoordinate(branch.getCondition(), owner, loop, width) &&
           llvm::all_of(branch.thenYield().getOperands(),
                        [&](Value u) {
                          return sharedCoordinate(u, owner, loop, width);
                        }) &&
           llvm::all_of(branch.elseYield().getOperands(), [&](Value u) {
             return sharedCoordinate(u, owner, loop, width);
           });
  }
  if (auto d = dyn_cast<arith::FloorDivSIOp>(op)) {
    auto group = d.getRhs().getDefiningOp<arith::ConstantIndexOp>();
    if (d.getLhs() == owner && group && group.value() > 0 &&
        group.value() % width == 0)
      return true;
  }
  if (!isa<memref::LoadOp, memref::DimOp>(op) &&
      op->getName().getDialectNamespace() != "arith")
    return false;
  return llvm::all_of(op->getOperands(), [&](Value u) {
    return sharedCoordinate(u, owner, loop, width);
  });
}
} // namespace
std::optional<TileDot> tileDot(Operation *s, scf::ForOp loop) {
  if (s->getName().stripDialect() != "lane_sum")
    return {};
  auto m = s->getOperand(0).getDefiningOp<arith::MulFOp>();
  if (!m)
    return {};
  for (unsigned j : {0u, 1u}) {
    Value a = widened(m->getOperand(j)), b = widened(m->getOperand(1 - j));
    if (a && b && a.getType() == b.getType() &&
        loop.isDefinedOutsideOfLoop(a) && b.getDefiningOp() &&
        b.getDefiningOp()->getBlock() == loop.getBody() &&
        b.getDefiningOp()->getName().stripDialect() == "slice_read")
      return TileDot{a, b};
  }
  return {};
}
bool sharedBounds(scf::ForOp loop, unsigned axis, int64_t rows) {
  auto member = loop->getBlock()->getArgument(axis);
  return sharedCoordinate(loop.getLowerBound(), member, loop, rows) &&
         sharedCoordinate(loop.getUpperBound(), member, loop, rows);
}
std::optional<TileAccumulate> tileAccumulate(Value v, scf::ForOp loop) {
  auto m = v.getDefiningOp<arith::MulFOp>();
  if (!m)
    return {};
  for (unsigned j : {0u, 1u}) {
    auto *s = m->getOperand(j).getDefiningOp();
    Value a = widened(m->getOperand(1 - j));
    if (!a || !a.getDefiningOp() ||
        a.getDefiningOp()->getName().stripDialect() != "slice_read" ||
        a.getDefiningOp()->getBlock() != loop.getBody() || !s ||
        s->getName().stripDialect() != "lane_splat")
      continue;
    Value w = widened(s->getOperand(0));
    if (w && w.getType() == cast<ShapedType>(a.getType()).getElementType())
      return TileAccumulate{a, w};
  }
  return {};
}
std::optional<unsigned> issueAxis(Operation *op,
                                  const compilerinfra::TargetProfile &p,
                                  int64_t rows, int64_t span) {
  if ((rows != 16 && rows != 32 && rows != 64 && rows != 128) ||
      !panelFolds(op, p, span))
    return {};
  auto &bb = op->getRegion(0).front();
  auto facts = p.services.get<CallFacts>();
  auto axes = op->getAttrOfType<DenseI64ArrayAttr>("axes");
  for (int j = bb.getNumArguments() - 1; j >= 0; --j) {
    auto *shape = facts->shapeOf(op->getOperand(j));
    auto extent = (*shape)[axes[j]].getAsInteger();
    if (!extent || *extent < rows || *extent % rows)
      continue;
    bool dot = false, good = true;
    int64_t maxDepth = 0;
    for (auto loop : bb.getOps<scf::ForOp>())
      if (loop->hasAttr("kir.reassociate")) {
        // Disjoint split intervals have no key panel to share. A lower bound
        // equal to the owner coordinate is the overlapping causal suffix.
        good &= loop.getLowerBound() == bb.getArgument(j) ||
                sharedCoordinate(loop.getLowerBound(), bb.getArgument(j), loop,
                                 rows);
        for (auto &v : loop.getBody()->without_terminator()) {
          auto name = v.getName().stripDialect();
          if (name == "slice_read" || isa<memref::LoadOp>(v)) {
            for (auto coord : v.getOperands().drop_front())
              good &= sharedCoordinate(coord, bb.getArgument(j), loop, rows);
          }
          if (name == "lane_sum") {
            auto d = tileDot(&v, loop);
            if (!d) {
              good = false;
              continue;
            }
            auto axis = kir::memberAxis(d->tokens);
            auto *tensorShape = facts->shapeOf(axis->first);
            auto n = (*tensorShape)[axis->second].getAsInteger();
            good &= n && *n >= 16 && *n <= 512 && llvm::isPowerOf2_64(*n);
            if (n)
              maxDepth = std::max(maxDepth, *n);
            dot = true;
          }
        }
        for (auto next : loop.getBody()->getTerminator()->getOperands()) {
          if (!isa<ShapedType>(next.getType()))
            continue;
          auto update =
              tileAccumulate(next.getDefiningOp()->getOperand(1), loop);
          if (!update) {
            good = false;
            continue;
          }
          auto axis = kir::memberAxis(update->values);
          auto n = (*facts->shapeOf(axis->first))[axis->second].getAsInteger();
          good &= n && *n >= 16 && *n <= 512 && llvm::isPowerOf2_64(*n);
        }
      }
    auto caps = p.services.get<OwnerMachine>();
    // Operand staging and score/probability exchanges share the block-local
    // budget. The compiler also needs a lane fallback for non-finite values
    // when causal rows cover different token intervals.
    auto transferBytes =
        2 * (rows + span) * maxDepth + 8 * rows * span + 2 * span * maxDepth;
    if (!caps->panelBytes || transferBytes > caps->panelBytes)
      good = false;
    if (caps->contributionLimit &&
        (rows + span) * maxDepth > caps->contributionLimit / 2)
      good = false;
    for (auto loop : bb.getOps<scf::ForOp>())
      if (loop->hasAttr("kir.reassociate") && caps->contributionLimit &&
          !sharedBounds(loop, j, rows) &&
          rows * span * maxDepth > 4 * caps->contributionLimit)
        good = false;
    if (good && dot)
      return j;
  }
  return {};
}
} // namespace intentdsl::block
