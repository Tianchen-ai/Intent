#include "intentdsl/levels/execution/block/ir/FoldIssue.h"
#include "compilerinfra/Proofs.h"
#include "mlir/IR/IRMapping.h"
using namespace mlir;
namespace intentdsl::block {
namespace {
Value widened(Value v) {
  auto cast = v.getDefiningOp<arith::ExtFOp>();
  return cast ? cast.getIn() : Value();
}
} // namespace
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
// Prove exact full-panel coverage via CompilerInfra's tail-access proof.
bool fullFoldPanels(scf::ForOp loop, const CallFacts &facts, int64_t span) {
  if (span <= 0 || !matchPattern(loop.getStep(), m_One()))
    return false;
  auto staticVal = [&](Value v) -> std::optional<int64_t> {
    if (auto c = v.getDefiningOp<arith::ConstantIndexOp>())
      return c.value();
    if (auto c = v.getDefiningOp<arith::ConstantOp>())
      if (auto ia = dyn_cast<IntegerAttr>(c.getValue());
          ia && ia.getValue().getSignificantBits() <= 64)
        return ia.getValue().getSExtValue();
    if (auto dim = v.getDefiningOp<memref::DimOp>()) {
      auto ax = dim.getConstantIndex();
      auto shape = facts.shapeOf(dim.getSource());
      if (ax && shape)
        return (*shape)[*ax].getAsInteger();
    }
    return std::nullopt;
  };
  auto staticCap = [&](Value v) -> std::optional<int64_t> {
    if (auto c = staticVal(v))
      return c;
    if (auto mn = v.getDefiningOp<arith::MinSIOp>()) {
      if (auto c = staticVal(mn.getLhs()))
        return c;
      if (auto c = staticVal(mn.getRhs()))
        return c;
    }
    if (auto sel = v.getDefiningOp<arith::SelectOp>()) {
      auto a = staticVal(sel.getTrueValue()), b = staticVal(sel.getFalseValue());
      if (a && b)
        return std::max(*a, *b);
    }
    return std::nullopt;
  };
  auto loCst = staticVal(loop.getLowerBound());
  auto hiCst = staticVal(loop.getUpperBound());
  if (loCst && hiCst && *loCst == *hiCst && *loCst >= 0 && *loCst % span == 0)
    return true;
  auto cap = staticCap(loop.getUpperBound());
  if (!cap || *cap < span)
    return false;
  if (loCst && (*loCst < 0 || *loCst > *cap))
    return false;
  if (hiCst && (*hiCst < 0 || *hiCst > *cap))
    return false;
  if (loCst && hiCst && *loCst > *hiCst - span)
    return false;
  auto *ctx = loop->getContext();
  ctx->getOrLoadDialect<arith::ArithDialect>();
  ctx->getOrLoadDialect<memref::MemRefDialect>();
  ctx->getOrLoadDialect<scf::SCFDialect>();
  OpBuilder b(loop);
  Location loc = loop.getLoc();
  SmallVector<Operation *, 16> tempOps;
  auto track = [&](Operation *op) {
    tempOps.push_back(op);
    return op;
  };
  // Materialize static CallFacts extents (memref.dim), owner_space arguments
  // (via unit-step scf.for ranges so MLIR IntegerRangeAnalysis propagates
  // through arithmetic), and arith.minsi -> arith.select in temporary probe IR
  // so CompilerInfra's proveTailAccess verifies alignment and bounds from IR.
  Operation *space = loop->getParentOp();
  auto spaceAxes =
      space ? space->getAttrOfType<DenseI64ArrayAttr>("axes") : DenseI64ArrayAttr();
  llvm::DenseMap<Value, Value> ownerRemap;
  SmallVector<scf::ForOp, 4> ownerLoops;
  auto trackLocal = [&](Operation *op) {
    if (ownerLoops.empty())
      tempOps.push_back(op);
    return op;
  };
  std::function<void(Value, unsigned)> collectOwners = [&](Value v,
                                                           unsigned depth) {
    if (depth > 8 || ownerRemap.count(v))
      return;
    if (auto barg = dyn_cast<BlockArgument>(v)) {
      if (space && space->getName().stripDialect() == "owner_space" &&
          spaceAxes && barg.getOwner() == &space->getRegion(0).front()) {
        unsigned idx = barg.getArgNumber();
        if (idx < spaceAxes.size() && idx < space->getNumOperands()) {
          if (auto sh = facts.shapeOf(space->getOperand(idx))) {
            if (auto ext = (*sh)[spaceAxes[idx]].getAsInteger(); ext && *ext > 0) {
              Value c0 =
                  trackLocal(b.create<arith::ConstantIndexOp>(loc, 0))
                      ->getResult(0);
              Value cExt =
                  trackLocal(b.create<arith::ConstantIndexOp>(loc, *ext))
                      ->getResult(0);
              Value c1 =
                  trackLocal(b.create<arith::ConstantIndexOp>(loc, 1))
                      ->getResult(0);
              auto ol = b.create<scf::ForOp>(loc, c0, cExt, c1);
              ownerLoops.push_back(ol);
              ownerRemap[v] = ol.getInductionVar();
              b.setInsertionPointToStart(ol.getBody());
            }
          }
        }
      }
      return;
    }
    if (Operation *def = v.getDefiningOp()) {
      if (isa<arith::MinSIOp, arith::AddIOp, arith::SubIOp, arith::MulIOp,
              arith::FloorDivSIOp, arith::DivUIOp, arith::RemSIOp,
              arith::RemUIOp, arith::IndexCastOp>(def))
        for (Value opnd : def->getOperands())
          collectOwners(opnd, depth + 1);
      else if (auto sel = dyn_cast<arith::SelectOp>(def)) {
        collectOwners(sel.getTrueValue(), depth + 1);
        collectOwners(sel.getFalseValue(), depth + 1);
      }
    }
  };
  auto buf = b.create<memref::AllocOp>(
      loc, MemRefType::get({*cap}, b.getI8Type()));
  track(buf);
  Value cCap = track(b.create<arith::ConstantIndexOp>(loc, *cap))->getResult(0);
  Value cSpan =
      track(b.create<arith::ConstantIndexOp>(loc, span))->getResult(0);
  collectOwners(loop.getLowerBound(), 0);
  collectOwners(loop.getUpperBound(), 0);
  std::function<Value(Value, unsigned)> materialize =
      [&](Value v, unsigned depth) -> Value {
    if (depth > 8)
      return v;
    if (auto it = ownerRemap.find(v); it != ownerRemap.end())
      return it->second;
    if (auto c = staticVal(v)) {
      if (v.getDefiningOp<arith::ConstantIndexOp>())
        return v;
      return trackLocal(b.create<arith::ConstantIndexOp>(loc, *c))
          ->getResult(0);
    }
    if (auto mn = v.getDefiningOp<arith::MinSIOp>()) {
      Value lhs = materialize(mn.getLhs(), depth + 1);
      Value rhs = materialize(mn.getRhs(), depth + 1);
      Value cmp =
          trackLocal(b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                             lhs, rhs))
              ->getResult(0);
      return trackLocal(b.create<arith::SelectOp>(loc, cmp, lhs, rhs))
          ->getResult(0);
    }
    if (auto sel = v.getDefiningOp<arith::SelectOp>()) {
      Value t = materialize(sel.getTrueValue(), depth + 1);
      Value f = materialize(sel.getFalseValue(), depth + 1);
      if (t != sel.getTrueValue() || f != sel.getFalseValue())
        return trackLocal(b.create<arith::SelectOp>(loc, sel.getCondition(), t,
                                                    f))
            ->getResult(0);
    }
    if (auto fd = v.getDefiningOp<arith::FloorDivSIOp>()) {
      Value lhs = materialize(fd.getLhs(), depth + 1);
      Value rhs = materialize(fd.getRhs(), depth + 1);
      if (lhs != fd.getLhs() || rhs != fd.getRhs())
        return trackLocal(b.create<arith::FloorDivSIOp>(loc, lhs, rhs))
            ->getResult(0);
    }
    if (auto add = v.getDefiningOp<arith::AddIOp>()) {
      Value lhs = materialize(add.getLhs(), depth + 1);
      Value rhs = materialize(add.getRhs(), depth + 1);
      if (lhs != add.getLhs() || rhs != add.getRhs())
        return trackLocal(b.create<arith::AddIOp>(loc, lhs, rhs))->getResult(0);
    }
    if (auto sub = v.getDefiningOp<arith::SubIOp>()) {
      Value lhs = materialize(sub.getLhs(), depth + 1);
      Value rhs = materialize(sub.getRhs(), depth + 1);
      if (lhs != sub.getLhs() || rhs != sub.getRhs())
        return trackLocal(b.create<arith::SubIOp>(loc, lhs, rhs))->getResult(0);
    }
    if (auto mul = v.getDefiningOp<arith::MulIOp>()) {
      Value lhs = materialize(mul.getLhs(), depth + 1);
      Value rhs = materialize(mul.getRhs(), depth + 1);
      if (lhs != mul.getLhs() || rhs != mul.getRhs())
        return trackLocal(b.create<arith::MulIOp>(loc, lhs, rhs))->getResult(0);
    }
    if (!v.getType().isIndex())
      return trackLocal(b.create<arith::IndexCastOp>(loc, b.getIndexType(), v))
          ->getResult(0);
    return v;
  };
  auto proveAligned = [&](Value bound) -> bool {
    OpBuilder::InsertionGuard guard(b);
    Value startIdx;
    if (auto bc = staticVal(bound); bc && *bc == *cap) {
      startIdx =
          trackLocal(b.create<arith::ConstantIndexOp>(loc, 0))->getResult(0);
    } else {
      startIdx = materialize(bound, 0);
      if (!startIdx.getType().isIndex())
        startIdx = trackLocal(
                       b.create<arith::IndexCastOp>(loc, b.getIndexType(),
                                                    startIdx))
                       ->getResult(0);
    }
    auto probe = b.create<scf::ForOp>(loc, startIdx, cCap, cSpan);
    b.setInsertionPointToStart(probe.getBody());
    auto sub = b.create<memref::SubViewOp>(
        loc, buf, ArrayRef<OpFoldResult>{probe.getInductionVar()},
        ArrayRef<OpFoldResult>{b.getIndexAttr(span)},
        ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
    auto proof = compilerinfra::proveTailAccess(sub.getOperation());
    probe.erase();
    return proof.proven && proof.kind == compilerinfra::TailKind::ExactInBounds;
  };
  bool ok = proveAligned(loop.getLowerBound()) &&
            proveAligned(loop.getUpperBound());
  if (!ownerLoops.empty())
    ownerLoops.front().erase();
  for (auto it = tempOps.rbegin(); it != tempOps.rend(); ++it)
    (*it)->erase();
  return ok;
}
bool overlappingWindowBound(Value v, BlockArgument owner, scf::ForOp loop,
                            int64_t width) {
  if (v == owner || sharedCoordinate(v, owner, loop, width))
    return true;
  Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  if (auto add = dyn_cast<arith::AddIOp>(def))
    return (overlappingWindowBound(add.getLhs(), owner, loop, width) &&
            sharedCoordinate(add.getRhs(), owner, loop, width)) ||
           (sharedCoordinate(add.getLhs(), owner, loop, width) &&
            overlappingWindowBound(add.getRhs(), owner, loop, width));
  if (auto sub = dyn_cast<arith::SubIOp>(def))
    return overlappingWindowBound(sub.getLhs(), owner, loop, width) &&
           sharedCoordinate(sub.getRhs(), owner, loop, width);
  if (isa<arith::MinSIOp, arith::MaxSIOp>(def))
    return overlappingWindowBound(def->getOperand(0), owner, loop, width) &&
           overlappingWindowBound(def->getOperand(1), owner, loop, width);
  if (auto sel = dyn_cast<arith::SelectOp>(def))
    return overlappingWindowBound(sel.getTrueValue(), owner, loop, width) &&
           overlappingWindowBound(sel.getFalseValue(), owner, loop, width);
  if (auto br = dyn_cast<scf::IfOp>(def);
      br && br.getNumResults() == 1 && !br.getElseRegion().empty())
    return overlappingWindowBound(br.thenYield().getOperand(0), owner, loop,
                                  width) &&
           overlappingWindowBound(br.elseYield().getOperand(0), owner, loop,
                                  width);
  return false;
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
Value causalMaskOwner(scf::ForOp loop, bool *inclusive) {
  if (!matchPattern(loop.getLowerBound(), m_Zero()) ||
      !matchPattern(loop.getStep(), m_One()) || loop.getNumResults() == 0)
    return {};
  auto isConstF32 = [](Value v, llvm::function_ref<bool(const APFloat &)> p) {
    auto c = v ? v.getDefiningOp<arith::ConstantOp>() : arith::ConstantOp();
    auto f = c ? dyn_cast<FloatAttr>(c.getValue()) : FloatAttr();
    return f && f.getType().isF32() && p(f.getValue());
  };
  auto isNegInf = [&](Value v) {
    return isConstF32(
        v, [](const APFloat &x) { return x.isInfinity() && x.isNegative(); });
  };
  auto isPosZero = [&](Value v) {
    return isConstF32(
        v, [](const APFloat &x) { return x.isZero() && !x.isNegative(); });
  };
  Value cond;
  auto y0 = loop.getBody()->getTerminator()->getOperand(0);
  if (auto mx = y0.getDefiningOp<arith::MaxNumFOp>()) {
    if (loop.getNumResults() != 1 || mx.getLhs() != loop.getRegionIterArgs()[0])
      return {};
    Value cand = mx.getRhs();
    if (auto br = cand.getDefiningOp<scf::IfOp>()) {
      if (br.getNumResults() != 1 || br.getElseRegion().empty() ||
          !isNegInf(br.elseYield().getOperand(0)))
        return {};
      cond = br.getCondition();
    } else if (auto sel = cand.getDefiningOp<arith::SelectOp>()) {
      if (!isNegInf(sel.getFalseValue()))
        return {};
      cond = sel.getCondition();
    }
  } else if (auto ad = y0.getDefiningOp<arith::AddFOp>()) {
    if (ad.getLhs() != loop.getRegionIterArgs()[0])
      return {};
    Value rhs = ad.getRhs();
    if (auto br = rhs.getDefiningOp<scf::IfOp>()) {
      if (br.getNumResults() != 1 || br.getElseRegion().empty() ||
          !isPosZero(br.elseYield().getOperand(0)))
        return {};
      cond = br.getCondition();
    } else if (auto sel = rhs.getDefiningOp<arith::SelectOp>()) {
      if (!isPosZero(sel.getFalseValue()))
        return {};
      cond = sel.getCondition();
    }
  }
  auto cmp = cond ? cond.getDefiningOp<arith::CmpIOp>() : arith::CmpIOp();
  if (!cmp)
    return {};
  Value iv = loop.getInductionVar(), lhs = cmp.getLhs(), rhs = cmp.getRhs();
  auto pred = cmp.getPredicate();
  if (rhs == iv && loop.isDefinedOutsideOfLoop(lhs) &&
      (pred == arith::CmpIPredicate::sge || pred == arith::CmpIPredicate::sgt)) {
    if (inclusive)
      *inclusive = (pred == arith::CmpIPredicate::sge);
    return lhs;
  }
  if (lhs == iv && loop.isDefinedOutsideOfLoop(rhs) &&
      (pred == arith::CmpIPredicate::sle || pred == arith::CmpIPredicate::slt)) {
    if (inclusive)
      *inclusive = (pred == arith::CmpIPredicate::sle);
    return rhs;
  }
  return {};
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
        // Disjoint split intervals have no key panel to share, whereas a
        // 1-Lipschitz owner-shifted lower bound (causal suffix or sliding
        // window) shares overlapping token panels across adjacent owners.
        good &= overlappingWindowBound(loop.getLowerBound(), bb.getArgument(j),
                                       loop, rows);
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
    int64_t contribLimit = caps->contributionLimit;
    if (contribLimit && physical >= 128 && caps->maxWarpgroups >= 2)
      contribLimit = contribLimit * 7 / 4;
    if (contribLimit && liveWords > contribLimit)
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
