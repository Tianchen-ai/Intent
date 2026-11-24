#include "intentdsl/levels/execution/block/ir/FoldIssue.h"
#include "mlir/IR/IRMapping.h"
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
int64_t alignedPageExtent(Operation &read, scf::ForOp loop,
                          const CallFacts &facts, int64_t span) {
  if (read.getName().stripDialect() != "slice_read" ||
      read.getNumOperands() < 3)
    return 0;
  auto coords = read.getOperands().drop_front();
  auto axis = read.getAttrOfType<IntegerAttr>("axis");
  if (!axis || axis.getInt() != int64_t(coords.size()) ||
      !llvm::all_of(coords.drop_front(2),
                    [&](Value v) { return loop.isDefinedOutsideOfLoop(v); }))
    return 0;
  auto rem = coords[1].getDefiningOp<arith::RemSIOp>();
  auto page = coords[0].getDefiningOp<arith::IndexCastOp>();
  if (!rem || rem.getLhs() != loop.getInductionVar() || !page)
    return 0;
  auto table = page.getIn().getDefiningOp<memref::LoadOp>();
  if (!table || table.getIndices().size() != 1)
    return 0;
  auto quotient = [&](Value v) {
    auto div = v.getDefiningOp<arith::FloorDivSIOp>();
    return div && div.getLhs() == rem.getLhs() && div.getRhs() == rem.getRhs();
  };
  auto index = table.getIndices().front();
  auto add = index.getDefiningOp<arith::AddIOp>();
  if (!quotient(index) &&
      !(add && ((quotient(add.getLhs()) &&
                 loop.isDefinedOutsideOfLoop(add.getRhs())) ||
                (quotient(add.getRhs()) &&
                 loop.isDefinedOutsideOfLoop(add.getLhs())))))
    return 0;
  auto shape = facts.shapeOf(read.getOperand(0));
  if (!shape || shape->size() < 3)
    return 0;
  auto width = (*shape)[1].getAsInteger().value_or(0);
  if (width < span || width % span)
    return 0;
  if (auto c = rem.getRhs().getDefiningOp<arith::ConstantIndexOp>())
    return c.value() == width ? width : 0;
  auto dim = rem.getRhs().getDefiningOp<memref::DimOp>();
  if (!dim || !dim.getConstantIndex())
    return 0;
  auto sh = facts.shapeOf(dim.getSource());
  return sh && (*sh)[*dim.getConstantIndex()].getAsInteger() == width ? width
                                                                      : 0;
}
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
// Propagate divisibility through the actual bounds. The proof applies to
// the entire interval, including the last panel, independently of input data.
bool fullFoldPanels(scf::ForOp loop,const CallFacts &facts,int64_t span) {
  if(span<=0 || !matchPattern(loop.getStep(),m_One()))return false;
  std::function<bool(Value)> multiple=[&](Value v) {
    if(auto c=v.getDefiningOp<arith::ConstantIndexOp>())return c.value()%span==0;
    if(auto dim=v.getDefiningOp<memref::DimOp>()){
      auto ax=dim.getConstantIndex();
      auto shape=facts.shapeOf(dim.getSource());
      if(!ax || !shape)return false;
      auto n=(*shape)[*ax].getAsInteger();return n && *n%span==0;
    }
    if(auto m=v.getDefiningOp<arith::MulIOp>())return multiple(m.getLhs())||multiple(m.getRhs());
    if(auto a=v.getDefiningOp<arith::AddIOp>())return multiple(a.getLhs())&&multiple(a.getRhs());
    if(auto m=v.getDefiningOp<arith::MinSIOp>())return multiple(m.getLhs())&&multiple(m.getRhs());
    return false;
  };
  return multiple(loop.getLowerBound()) && multiple(loop.getUpperBound());
}
bool sharedBounds(scf::ForOp loop, unsigned axis, int64_t rows) {
  auto member = loop->getBlock()->getArgument(axis);
  return sharedCoordinate(loop.getLowerBound(), member, loop, rows) &&
         sharedCoordinate(loop.getUpperBound(), member, loop, rows);
}
// Split the common interval from the ragged fringe in increasing token
// order. Only the fringe can mix excluded and included non-finite operands.
// Distinct memref.dim operations can denote the same immutable extent.
static bool sameExtent(Value first,Value second){
  if(first==second)return true;
  auto lhs=first.getDefiningOp<memref::DimOp>(),rhs=second.getDefiningOp<memref::DimOp>();
  return lhs && rhs && lhs.getSource()==rhs.getSource() && lhs.getConstantIndex() &&
         lhs.getConstantIndex()==rhs.getConstantIndex();
}
Value causalPrefixOwner(scf::ForOp loop) {
  auto branch = loop.getUpperBound().getDefiningOp<scf::IfOp>();
  if (!branch || !matchPattern(loop.getLowerBound(), m_Zero()) ||
      branch.getNumResults() != 1 || branch.getElseRegion().empty() ||
      loop.getUpperBound() != branch.getResult(0))
    return {};
  auto cmp = branch.getCondition().getDefiningOp<arith::CmpIOp>();
  auto next = branch.thenYield().getOperand(0).getDefiningOp<arith::AddIOp>();
  if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::slt || !next ||
      next.getLhs() != cmp.getLhs() || !matchPattern(next.getRhs(), m_One()) ||
      !sameExtent(cmp.getRhs(), branch.elseYield().getOperand(0)))
    return {};
  return cmp.getLhs();
}
void partitionOwnerFolds(Operation *space,unsigned axis,int64_t rows,int64_t span){
  auto &body=space->getRegion(0).front();
  auto owner=body.getArgument(axis);
  for(auto loop:llvm::make_early_inc_range(body.getOps<scf::ForOp>())){
    if(!loop->hasAttr("kir.reassociate"))continue;
    Value lo=loop.getLowerBound(),hi=loop.getUpperBound(),cap;
    bool suffix=lo==owner && sharedCoordinate(hi,owner,loop,rows);
    if(suffix)cap=hi;
    else{
      if (causalPrefixOwner(loop) != owner)
        continue;
      auto branch = hi.getDefiningOp<scf::IfOp>();
      cap = branch.getCondition().getDefiningOp<arith::CmpIOp>().getRhs();
      if (!sharedCoordinate(cap, owner, loop, rows))
        continue;
    }
    auto loc=loop.getLoc();OpBuilder b(loop);
    Value width=b.create<arith::ConstantIndexOp>(loc,std::max(rows,span));
    Value base=b.create<arith::MulIOp>(loc,b.create<arith::FloorDivSIOp>(loc,owner,width),width);
    if(suffix)base=b.create<arith::AddIOp>(loc,base,width);
    Value cut=b.create<arith::MinSIOp>(loc,cap,base);
    IRMapping map;auto first=cast<scf::ForOp>(b.clone(*loop,map));
    first.setUpperBound(cut);
    loop.setLowerBound(cut);
    loop.getInitArgsMutable().assign(first.getResults());
  }
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
  if ((rows != 4 && rows != 8 && rows != 16 && rows != 32 && rows != 64 &&
       rows != 128) ||
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
    int64_t maxDepth = 0, operandBits = 32, shapedCarries = 0;
    bool sawOperand = false;
    for (auto loop : bb.getOps<scf::ForOp>())
      if (loop->hasAttr("kir.reassociate")) {
        // Disjoint split intervals have no key panel to share. A lower bound
        // equal to the owner coordinate is the overlapping causal suffix.
        good &= loop.getLowerBound() == bb.getArgument(j) ||
                sharedCoordinate(loop.getLowerBound(), bb.getArgument(j), loop,
                                 rows);
        shapedCarries = std::max<int64_t>(
            shapedCarries,
            llvm::count_if(loop.getResultTypes(),
                           [](Type t) { return isa<ShapedType>(t); }));
        for (auto &v : loop.getBody()->without_terminator()) {
          auto name = v.getName().stripDialect();
          if (name == "slice_read" || isa<memref::LoadOp>(v)) {
            for (auto coord : v.getOperands().drop_front())
              good &= sharedCoordinate(coord, bb.getArgument(j), loop, rows);
            if (name == "slice_read") {
              int64_t bits =
                  cast<MemRefType>(v.getOperand(0).getType())
                      .getElementTypeBitWidth();
              operandBits = sawOperand ? std::min(operandBits, bits) : bits;
              sawOperand = true;
            }
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
    if (rows < 16 && !llvm::all_of(
                         bb.getOps<scf::ForOp>(),
                         [&](scf::ForOp l) {
                           return !l->hasAttr("kir.reassociate") ||
                                  sharedBounds(l, j, rows);
                         }))
      good = false;
    // Operand staging and score/probability exchanges share the block-local
    // budget. The compiler also needs a lane fallback for non-finite values
    // when causal rows cover different token intervals.
    auto physical = std::max<int64_t>(16, rows);
    auto transferBytes = 2 * (physical + span) * maxDepth +
                         8 * physical * span + 2 * span * maxDepth;
    if (!caps->panelBytes || transferBytes > caps->panelBytes)
      good = false;
    int64_t liveWords =
        ((physical + span) * maxDepth * operandBits + 31) / 32 +
        std::max<int64_t>(2, shapedCarries) * physical * span +
        shapedCarries * physical * maxDepth;
    if (caps->contributionLimit && liveWords > caps->contributionLimit)
      good = false;
    // A masked edge replays at most one member vector per owner. The score
    // matrix and operand panels are live separately; no [row,token,depth]
    // tensor is produced even when two causal intervals differ.
    if (good && dot)
      return j;
  }
  return {};
}
} // namespace intentdsl::block
