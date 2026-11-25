#include "OwnerTiles.h"
#include "compilerinfra/Proofs.h"
#include "intentdsl/levels/execution/block/ir/FoldPanel.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
using namespace mlir;
namespace intentdsl::block {
namespace {
Type retype(Type t, Type elem) {
  if (auto shaped = dyn_cast<RankedTensorType>(t))
    return shaped.clone(elem);
  return elem;
}
} // namespace
OwnerTiles::OwnerTiles(func::FuncOp f, Operation &s, StringRef unit,
                       OwnerAccess access)
    : Tiles(f, "owners", unit), space(s), access(access) {
  logical = s.getAttrOfType<IntegerAttr>("block.owner_tile").getInt();
  auto native = s.getAttrOfType<IntegerAttr>(symbol("matrix_rows"));
  rows = native ? native.getInt() : logical;
}
Type OwnerTiles::lifted(Value v) {
  auto typ = v.getType(), e = elem(typ);
  if (e.isIndex())
    e = b.getI64Type();
  if (isa<RankedTensorType>(typ)) {
    int64_t w = std::max<int64_t>(
        1,
        llvm::PowerOf2Ceil(intentdsl::kir::memberExtent(fn, v, "block.shape")));
    return shaped({rows, w}, e);
  }
  return shaped({rows}, e);
}
Value OwnerTiles::castElement(Value v, Type to) {
  if (to.isIndex())
    to = b.getI64Type();
  Type from = elem(v.getType()), dst = retype(v.getType(), to);
  if (from == to)
    return v;
  if (isa<FloatType>(from) && isa<FloatType>(to)) {
    if (from.getIntOrFloatBitWidth() < to.getIntOrFloatBitWidth())
      return b.create<arith::ExtFOp>(loc, dst, v);
    return b.create<arith::TruncFOp>(loc, dst, v);
  }
  if (from.getIntOrFloatBitWidth() > to.getIntOrFloatBitWidth())
    return b.create<arith::TruncIOp>(loc, dst, v);
  if (from.isInteger(1))
    return b.create<arith::ExtUIOp>(loc, dst, v);
  return b.create<arith::ExtSIOp>(loc, dst, v);
}
Value OwnerTiles::widen(Value v) { return castElement(v, b.getI64Type()); }
Value OwnerTiles::broadcast(Value v, Type dest, int axis) {
  return like(v, dest, isa<ShapedType>(v.getType()) ? axis : -1);
}
static std::optional<int64_t> constIntVal(Value v) {
  Operation *op = v.getDefiningOp();
  while (op && op->getName().stripDialect() == "tile_broadcast") {
    v = op->getOperand(0);
    op = v.getDefiningOp();
  }
  auto c = dyn_cast_or_null<arith::ConstantOp>(op);
  if (!c)
    return std::nullopt;
  if (auto ia = dyn_cast<IntegerAttr>(c.getValue());
      ia && (ia.getType().isIndex() ||
             (ia.getType().isIntOrFloat() &&
              ia.getType().getIntOrFloatBitWidth() <= 64)))
    return ia.getInt();
  if (auto da = dyn_cast<DenseIntElementsAttr>(c.getValue());
      da && da.isSplat() && da.getElementType().getIntOrFloatBitWidth() <= 64)
    return da.getSplatValue<APInt>().getSExtValue();
  return std::nullopt;
}
Value OwnerTiles::select(Value mask, Value yes, Value no) {
  Value condBool = broadcast(mask, retype(yes.getType(), b.getI1Type()));
  if (auto inner = yes.getDefiningOp<arith::SelectOp>()) {
    auto kF1 = constIntVal(inner.getFalseValue());
    auto kF2 = constIntVal(no);
    if ((inner.getFalseValue() == no || (kF1 && kF2 && *kF1 == *kF2)) &&
        inner.getCondition().getType() == condBool.getType()) {
      Value both =
          b.create<arith::AndIOp>(loc, inner.getCondition(), condBool);
      return b.create<arith::SelectOp>(loc, both, inner.getTrueValue(), no);
    }
  }
  return b.create<arith::SelectOp>(loc, condBool, yes, no);
}
Value OwnerTiles::constant(arith::ConstantOp c) {
  if (auto i = dyn_cast<IntegerAttr>(c.getValue())) {
    auto t = c.getType().isIndex() ? Type(b.getI64Type()) : c.getType();
    return b.create<arith::ConstantOp>(loc, b.getIntegerAttr(t, i.getInt()));
  }
  return b.create<arith::ConstantOp>(loc, c.getValue());
}
Value OwnerTiles::at(Value old, const Values &env) {
  auto it = env.find(old);
  return it == env.end() ? old : it->second;
}
Value OwnerTiles::apply(Operation &old, ValueRange args, Type dst) {
  auto code = old.getName().stripDialect();
  if (code == "index_cast" || code == "extf" || code == "truncf")
    return castElement(args[0], elem(dst));
  if (!isa<ShapedType>(dst) && args.size() == 2) {
    auto c0 = args[0].getDefiningOp<arith::ConstantOp>();
    auto c1 = args[1].getDefiningOp<arith::ConstantOp>();
    auto i0 = c0 ? dyn_cast<IntegerAttr>(c0.getValue()) : IntegerAttr();
    auto i1 = c1 ? dyn_cast<IntegerAttr>(c1.getValue()) : IntegerAttr();
    if (i0 && i1 && i0.getType().isInteger(64) && i1.getType().isInteger(64)) {
      int64_t a = i0.getInt(), rhsVal = i1.getInt();
      auto floorDiv64 = [](int64_t x, int64_t d) -> int64_t {
        int64_t q = x / d, r = x % d;
        return (r != 0 && ((r < 0) != (d < 0))) ? q - 1 : q;
      };
      if (dst.isInteger(64)) {
        if (code == "addi")
          return integer(a + rhsVal);
        if (code == "subi")
          return integer(a - rhsVal);
        if (code == "muli")
          return integer(a * rhsVal);
        if (code == "divui" && a >= 0 && rhsVal > 0)
          return integer(a / rhsVal);
        if (code == "remui" && a >= 0 && rhsVal > 0)
          return integer(a % rhsVal);
        if (code == "floordivsi" && rhsVal > 0)
          return integer(floorDiv64(a, rhsVal));
        if (code == "remsi" && rhsVal > 0)
          return integer(a % rhsVal);
      } else if (dst.isInteger(1) && isa<arith::CmpIOp>(old)) {
        auto pred = cast<arith::CmpIOp>(old).getPredicate();
        bool res = false;
        switch (pred) {
        case arith::CmpIPredicate::eq: res = (a == rhsVal); break;
        case arith::CmpIPredicate::ne: res = (a != rhsVal); break;
        case arith::CmpIPredicate::slt: res = (a < rhsVal); break;
        case arith::CmpIPredicate::sle: res = (a <= rhsVal); break;
        case arith::CmpIPredicate::sgt: res = (a > rhsVal); break;
        case arith::CmpIPredicate::sge: res = (a >= rhsVal); break;
        default: res = false; break;
        }
        if (pred == arith::CmpIPredicate::eq || pred == arith::CmpIPredicate::ne ||
            pred == arith::CmpIPredicate::slt || pred == arith::CmpIPredicate::sle ||
            pred == arith::CmpIPredicate::sgt || pred == arith::CmpIPredicate::sge)
          return b.create<arith::ConstantOp>(loc, b.getBoolAttr(res));
      }
    }
  }
  if (args.size() == 2 && elem(dst).isIntOrIndex()) {
    auto c0Val = constIntVal(args[0]);
    auto c1Val = constIntVal(args[1]);
    if (code == "addi") {
      if (c1Val && *c1Val == 0)
        return broadcast(args[0], dst);
      if (c0Val && *c0Val == 0)
        return broadcast(args[1], dst);
    } else if (code == "subi" && c1Val && *c1Val == 0) {
      return broadcast(args[0], dst);
    } else if (elem(dst).isInteger(1) && isa<arith::CmpIOp>(old)) {
      auto pred = cast<arith::CmpIOp>(old).getPredicate();
      if ((pred == arith::CmpIPredicate::eq ||
           pred == arith::CmpIPredicate::ne) &&
          c1Val) {
        if (auto sel = args[0].getDefiningOp<arith::SelectOp>()) {
          auto kT = constIntVal(sel.getTrueValue());
          auto kF = constIntVal(sel.getFalseValue());
          if (kT && kF && *kT != *kF) {
            bool isEq = pred == arith::CmpIPredicate::eq;
            if ((*c1Val == *kT && isEq) || (*c1Val == *kF && !isEq))
              return broadcast(sel.getCondition(), dst);
            if ((*c1Val == *kF && isEq) || (*c1Val == *kT && !isEq)) {
              Value cB = broadcast(sel.getCondition(), dst);
              Value one = like(b.create<arith::ConstantOp>(loc, b.getBoolAttr(true)),
                               dst);
              return b.create<arith::XOrIOp>(loc, cB, one);
            }
          }
        }
      }
    }
  }
  OperationState st(loc, old.getName());
  st.addAttributes(old.getAttrs());
  st.addTypes(dst);
  for (auto arg : args)
    st.addOperands(broadcast(arg, retype(dst, elem(arg.getType()))));
  return b.create(st)->getResult(0);
}
Value OwnerTiles::slice(Operation &old, ArrayRef<Value> outerCoords, Value mask,
                        Value stored) {
  auto view = old.getOperand(stored ? 1 : 0);
  int axis = old.getAttrOfType<IntegerAttr>("axis").getInt();
  int64_t n = shape(view)[axis],
          w = std::max<int64_t>(1, llvm::PowerOf2Ceil(n));
  Value result;
  if (access.rectangle &&
      access.rectangle(*this, old, outerCoords, rows, stored, result))
    return result;
  auto lanesK = lanes(w);
  if (old.hasAttr("block.broadcast_read") && !stored) {
    // The block verifier proved equality of every owner's source coordinates.
    SmallVector<Value> ix;
    for (auto coord : outerCoords)
      ix.push_back(isa<RankedTensorType>(coord.getType())
                       ? reduce(coord, "min", 0)
                       : coord);
    ix.insert(ix.begin() + axis, lanesK);
    SmallVector<int64_t> axes(ix.size(), -1);
    axes[axis] = 0;
    auto ownersAttr = program->getAttrOfType<IntegerAttr>("block.owners");
    bool guaranteedActive =
        mask == active && ownersAttr && ownersAttr.getInt() > 0;
    Value keep;
    if (!guaranteedActive || n != w) {
      Value inside =
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, lanesK,
                                  like(integer(n), lanesK.getType()));
      if (guaranteedActive) {
        keep = inside;
      } else {
        auto anyActive = reduce(mask, "max", 0);
        keep = b.create<arith::AndIOp>(loc, like(anyActive, inside.getType()),
                                       inside);
      }
    }
    auto v = read(view, address(view, ix, axes), keep);
    return like(v, lifted(old.getResult(0)), 1);
  }
  SmallVector<Value> coords(outerCoords);
  coords.insert(coords.begin() + axis, lanesK);
  SmallVector<int64_t> projections;
  for (auto [i, c] : llvm::enumerate(coords)) {
    if (i != size_t(axis) && !isa<ShapedType>(c.getType()))
      c = broadcast(c, shaped({rows}, b.getI64Type()));
    projections.push_back(i == size_t(axis) ? 1 : 0);
  }
  auto ptr = address(view, coords, projections);
  auto maskType = shaped({rows, w}, b.getI1Type());
  auto valid = broadcast(mask, maskType);
  if (n != w) {
    auto k = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, lanesK,
                                     like(integer(n), lanesK.getType()));
    valid = b.create<arith::AndIOp>(loc, valid, broadcast(k, maskType, 1));
  }
  if (stored) {
    write(view, ptr, stored, valid);
    return {};
  }
  return read(view, ptr, valid);
}
bool OwnerTiles::canVectorizeIndependentLoop(scf::ForOp loop) {
  if (loop.getNumResults() != 0 ||
      !matchPattern(loop.getLowerBound(), m_Zero()) ||
      !matchPattern(loop.getStep(), m_One()))
    return false;
  if (!loop.isDefinedOutsideOfLoop(loop.getUpperBound()))
    return false;
  if (rows > 1) {
    // Multi-owner tiles require the loop bound to be invariant across owners.
    if (auto barg = dyn_cast<BlockArgument>(loop.getUpperBound());
        barg && barg.getOwner() == &space.getRegion(0).front())
      return false;
    if (!isa<arith::ConstantOp, arith::ConstantIndexOp, memref::DimOp>(
            loop.getUpperBound().getDefiningOp()))
      return false;
  }
  Value iv = loop.getInductionVar();
  SmallVector<memref::LoadOp, 4> loads;
  SmallVector<memref::StoreOp, 4> stores;
  std::function<bool(Block &, bool)> checkBlock = [&](Block &blk,
                                                      bool top) -> bool {
    for (Operation &inst : blk.without_terminator()) {
      if (auto br = dyn_cast<scf::IfOp>(inst)) {
        if (br.getNumResults() == 0 || br.getElseRegion().empty() ||
            !checkBlock(*br.thenBlock(), false) ||
            !checkBlock(*br.elseBlock(), false))
          return false;
        continue;
      }
      if (inst.getNumRegions() != 0)
        return false;
      if (auto ld = dyn_cast<memref::LoadOp>(inst)) {
        if (!top)
          return false;
        auto arg = dyn_cast<BlockArgument>(ld.getMemRef());
        if (!arg || arg.getOwner() != &fn.front())
          return false;
        int ivCount = 0;
        for (Value idx : ld.getIndices()) {
          if (idx == iv)
            ++ivCount;
          else if (!loop.isDefinedOutsideOfLoop(idx))
            return false;
        }
        if (ivCount > 1)
          return false;
        loads.push_back(ld);
        continue;
      }
      if (auto st = dyn_cast<memref::StoreOp>(inst)) {
        if (!top)
          return false;
        auto arg = dyn_cast<BlockArgument>(st.getMemRef());
        if (!arg || arg.getOwner() != &fn.front() || st.getIndices().empty() ||
            st.getIndices().back() != iv)
          return false;
        for (Value idx : st.getIndices().drop_back())
          if (!loop.isDefinedOutsideOfLoop(idx))
            return false;
        stores.push_back(st);
        continue;
      }
      if (auto cmp = dyn_cast<arith::CmpIOp>(inst)) {
        if (!supportedPanelCmpI(cmp))
          return false;
        continue;
      }
      if (!llvm::StringSwitch<bool>(inst.getName().getStringRef())
               .Cases("arith.index_cast", "arith.constant", true)
               .Cases("arith.sitofp", "arith.fptosi", "arith.cmpf", true)
               .Cases("arith.extf", "arith.truncf", "arith.maxnumf", true)
               .Cases("arith.minnumf", "arith.minimumf", "arith.minsi", true)
               .Cases("arith.addf", "arith.mulf", "arith.subf", "arith.divf",
                      true)
               .Cases("arith.addi", "arith.subi", "arith.muli", true)
               .Cases("arith.andi", "arith.ori", "arith.xori", true)
               .Cases("arith.select", "math.exp", "math.exp2", "math.log",
                      "math.absf", true)
               .Default(false))
        return false;
    }
    return true;
  };
  if (!checkBlock(*loop.getBody(), true) || stores.empty())
    return false;
  auto aliasGroup = [&](Value memrefVal) -> std::optional<int64_t> {
    auto arg = dyn_cast<BlockArgument>(memrefVal);
    if (!arg || arg.getOwner() != &fn.front())
      return std::nullopt;
    auto attr =
        fn.getArgAttrOfType<IntegerAttr>(arg.getArgNumber(), "block.alias_group");
    return attr ? std::optional<int64_t>(attr.getInt()) : std::nullopt;
  };
  for (size_t i = 0; i < stores.size(); ++i) {
    auto sg = aliasGroup(stores[i].getMemRef());
    if (!sg)
      return false;
    for (size_t j = i + 1; j < stores.size(); ++j) {
      auto og = aliasGroup(stores[j].getMemRef());
      if (!og || (*sg == *og &&
                  (stores[i].getMemRef() != stores[j].getMemRef() ||
                   stores[i].getIndices() != stores[j].getIndices())))
        return false;
    }
    for (auto ld : loads) {
      auto lg = aliasGroup(ld.getMemRef());
      if (!lg)
        return false;
      if (*sg == *lg) {
        if (stores[i].getMemRef() != ld.getMemRef() ||
            stores[i].getIndices() != ld.getIndices() ||
            !ld->isBeforeInBlock(stores[i]))
          return false;
      }
    }
  }
  return true;
}

void OwnerTiles::vectorizedLoop(scf::ForOp loop, const Values &parent,
                                Value mask) {
  int64_t baseSpan =
      space.hasAttr("block.token_span")
          ? space.getAttrOfType<IntegerAttr>("block.token_span").getInt()
          : 512;
  int64_t W = std::max<int64_t>(16, std::min<int64_t>(4096, baseSpan) / rows);
  Value rawHi = widen(at(loop.getUpperBound(), parent));
  Value hiScalar = isa<RankedTensorType>(rawHi.getType())
                       ? reduce(rawHi, "max", 0)
                       : rawHi;
  auto l = b.create<scf::ForOp>(loc, integer(0), hiScalar, integer(W));
  l->setAttr(symbol("runtime_range"), b.getUnitAttr());
  b.setInsertionPointToStart(l.getBody());
  Value base = l.getInductionVar();
  Value colVec = stepped(base, W);
  Type tileBool = shaped({rows, W}, b.getI1Type());
  Value inBounds = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                           colVec, like(hiScalar, colVec.getType()));
  Value valid2D = b.create<arith::AndIOp>(loc, broadcast(mask, tileBool, 0),
                                          broadcast(inBounds, tileBool, 1));
  Value iv = loop.getInductionVar();
  llvm::DenseMap<Value, Value> vecEnv;
  auto lift2D = [&](Value orig) -> Value {
    if (orig == iv)
      return broadcast(colVec, shaped({rows, W}, b.getI64Type()), 1);
    if (auto it = vecEnv.find(orig); it != vecEnv.end())
      return it->second;
    Value pv = at(orig, parent);
    Type e = elem(pv.getType());
    if (e.isIndex())
      e = b.getI64Type();
    if (elem(pv.getType()) != e)
      pv = castElement(pv, e);
    return broadcast(pv, shaped({rows, W}, e), 0);
  };
  std::function<SmallVector<Value>(Block &)> lowerVecBlock =
      [&](Block &blk) -> SmallVector<Value> {
    for (Operation &inst : blk.without_terminator()) {
      if (auto c = dyn_cast<arith::ConstantOp>(inst)) {
        Value cv = constant(c);
        vecEnv[inst.getResult(0)] =
            broadcast(cv, shaped({rows, W}, elem(cv.getType())), 0);
        continue;
      }
      if (auto br = dyn_cast<scf::IfOp>(inst)) {
        Value cond2D = lift2D(br.getCondition());
        auto yes = lowerVecBlock(*br.thenBlock());
        auto no = lowerVecBlock(*br.elseBlock());
        for (auto [res, tv, fv] : llvm::zip(br.getResults(), yes, no))
          vecEnv[res] = select(cond2D, tv, fv);
        continue;
      }
      if (auto ld = dyn_cast<memref::LoadOp>(inst)) {
        Value view = ld.getMemRef();
        SmallVector<Value, 4> coords;
        SmallVector<int64_t, 4> axes;
        bool usesIv = false;
        for (Value idx : ld.getIndices()) {
          if (idx == iv) {
            coords.push_back(colVec);
            axes.push_back(1);
            usesIv = true;
          } else {
            coords.push_back(
                broadcast(widen(at(idx, parent)), shaped({rows}, b.getI64Type())));
            axes.push_back(0);
          }
        }
        if (usesIv) {
          vecEnv[ld.getResult()] =
              read(view, address(view, coords, axes), valid2D);
        } else {
          Value rv = read(view, address(view, coords, axes), mask);
          vecEnv[ld.getResult()] =
              broadcast(rv, shaped({rows, W}, elem(rv.getType())), 0);
        }
        continue;
      }
      if (auto st = dyn_cast<memref::StoreOp>(inst)) {
        Value view = st.getMemRef();
        SmallVector<Value, 4> coords;
        SmallVector<int64_t, 4> axes;
        for (Value idx : st.getIndices().drop_back()) {
          coords.push_back(
              broadcast(widen(at(idx, parent)), shaped({rows}, b.getI64Type())));
          axes.push_back(0);
        }
        coords.push_back(colVec);
        axes.push_back(1);
        write(view, address(view, coords, axes), lift2D(st.getValueToStore()),
              valid2D);
        continue;
      }
      SmallVector<Value, 3> args;
      for (Value opnd : inst.getOperands())
        args.push_back(lift2D(opnd));
      Type e = elem(inst.getResult(0).getType());
      if (e.isIndex())
        e = b.getI64Type();
      vecEnv[inst.getResult(0)] = apply(inst, args, shaped({rows, W}, e));
    }
    SmallVector<Value> outs;
    if (&blk != loop.getBody())
      for (Value yv : blk.getTerminator()->getOperands())
        outs.push_back(lift2D(yv));
    return outs;
  };
  (void)lowerVecBlock(*loop.getBody());
  b.setInsertionPointAfter(l);
}

void OwnerTiles::triangularLoop(scf::ForOp loop,
                                const TriangularRecurrenceInfo &tri,
                                const Values &parent, Value mask) {
  int64_t N = tri.extent;
  int64_t W = std::max<int64_t>(2, llvm::PowerOf2Ceil(N));
  Type accTy = shaped({rows, W}, b.getF32Type());
  Type rowF32 = shaped({rows}, b.getF32Type());
  Type rowI64 = shaped({rows}, b.getI64Type());
  Type tileBool = shaped({rows, W}, b.getI1Type());

  Value zero2D = broadcast(scalar(0.0f), accTy);
  Value lanesRPre = lanes(W);
  Value inBoundsPre =
      (N == W) ? Value()
               : Value(b.create<arith::CmpIOp>(
                     loc, arith::CmpIPredicate::slt, lanesRPre,
                     like(integer(N), lanesRPre.getType())));
  Value validState2D = broadcast(mask, tileBool, 0);
  if (inBoundsPre)
    validState2D = b.create<arith::AndIOp>(
        loc, validState2D, broadcast(inBoundsPre, tileBool, 1));

  auto colLoop = b.create<scf::ForOp>(loc, integer(0), integer(N), integer(1),
                                      ValueRange{zero2D});
  colLoop->setAttr(symbol("runtime_range"), b.getUnitAttr());
  b.setInsertionPointToStart(colLoop.getBody());

  Value c = colLoop.getInductionVar();
  Value acc = colLoop.getRegionIterArgs()[0];

  Value isC1D =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, lanesRPre,
                              like(c, lanesRPre.getType()));
  Value isC2D = broadcast(isC1D, tileBool, 1);
  Value accC =
      reduce(b.create<arith::SelectOp>(loc, isC2D, acc, zero2D), "sum", 1);

  SmallVector<Value, 4> stCoords;
  SmallVector<int64_t, 4> stAxes;
  for (auto [i, orig] : llvm::enumerate(tri.stateCoords)) {
    Value coordVal = (int(i) == tri.stateAxis) ? c : widen(at(orig, parent));
    stCoords.push_back(broadcast(coordVal, rowI64));
    stAxes.push_back(0);
  }
  Value stPtr = address(tri.state, stCoords, stAxes);
  Value rhsC = read(tri.state, stPtr, mask);
  Value diffC = apply(*tri.subOp, {rhsC, accC}, rowF32);
  Value solC = diffC;
  if (tri.divOp) {
    SmallVector<Value, 4> dCoords;
    SmallVector<int64_t, 4> dAxes;
    for (auto [i, orig] : llvm::enumerate(tri.coeffCoords)) {
      Value coordVal =
          (int(i) == tri.coeffRowAxis || int(i) == tri.coeffColAxis)
              ? c
              : widen(at(orig, parent));
      dCoords.push_back(broadcast(coordVal, rowI64));
      dAxes.push_back(0);
    }
    Value diagC = read(tri.coeff, address(tri.coeff, dCoords, dAxes), mask);
    solC = apply(*tri.divOp, {diffC, diagC}, rowF32);
  }
  write(tri.state, stPtr, solC, mask);
  Value sol2D = broadcast(solC, accTy, 0);

  bool uniformCoeff = true;
  for (auto [i, orig] : llvm::enumerate(tri.coeffCoords)) {
    if (int(i) == tri.coeffRowAxis || int(i) == tri.coeffColAxis)
      continue;
    if (isa<ShapedType>(at(orig, parent).getType()))
      uniformCoeff = false;
  }

  Value lCol2D;
  if (access.scalarBroadcastRead && uniformCoeff && rows > 1) {
    SmallVector<Value, 4> lCoords;
    SmallVector<int64_t, 4> lAxes;
    for (auto [i, orig] : llvm::enumerate(tri.coeffCoords)) {
      if (int(i) == tri.coeffRowAxis) {
        lCoords.push_back(lanesRPre);
        lAxes.push_back(0);
      } else if (int(i) == tri.coeffColAxis) {
        lCoords.push_back(c);
        lAxes.push_back(-1);
      } else {
        lCoords.push_back(widen(at(orig, parent)));
        lAxes.push_back(-1);
      }
    }
    auto ownersAttr = program->getAttrOfType<IntegerAttr>("block.owners");
    bool guaranteedActive =
        mask == active && ownersAttr && ownersAttr.getInt() > 0;
    Value keep1D;
    if (!guaranteedActive || N != W) {
      Value inside1D =
          inBoundsPre
              ? inBoundsPre
              : Value(b.create<arith::CmpIOp>(
                    loc, arith::CmpIPredicate::slt, lanesRPre,
                    like(integer(N), lanesRPre.getType())));
      if (guaranteedActive) {
        keep1D = inside1D;
      } else {
        Value anyActive = reduce(mask, "max", 0);
        keep1D = b.create<arith::AndIOp>(
            loc, like(anyActive, inside1D.getType()), inside1D);
      }
    }
    Value v1D = read(tri.coeff, address(tri.coeff, lCoords, lAxes), keep1D);
    lCol2D = broadcast(v1D, accTy, 1);
  } else {
    SmallVector<Value, 4> lCoords;
    SmallVector<int64_t, 4> lAxes;
    for (auto [i, orig] : llvm::enumerate(tri.coeffCoords)) {
      if (int(i) == tri.coeffRowAxis) {
        lCoords.push_back(lanesRPre);
        lAxes.push_back(1);
      } else if (int(i) == tri.coeffColAxis) {
        lCoords.push_back(c);
        lAxes.push_back(-1);
      } else {
        lCoords.push_back(broadcast(widen(at(orig, parent)), rowI64));
        lAxes.push_back(0);
      }
    }
    lCol2D = read(tri.coeff, address(tri.coeff, lCoords, lAxes), validState2D);
  }

  Value prod =
      (tri.mulOp->getOperand(0).getDefiningOp<memref::LoadOp>().getMemRef() ==
       tri.coeff)
          ? apply(*tri.mulOp, {lCol2D, sol2D}, accTy)
          : apply(*tri.mulOp, {sol2D, lCol2D}, accTy);
  Value nextAcc = (tri.addOp->getOperand(0) ==
                   cast<scf::ForOp>(tri.addOp->getParentOp())
                       .getRegionIterArgs()[0])
                      ? apply(*tri.addOp, {acc, prod}, accTy)
                      : apply(*tri.addOp, {prod, acc}, accTy);
  Value gtC1D =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, lanesRPre,
                              like(c, lanesRPre.getType()));
  Value belowDiag = b.create<arith::AndIOp>(loc, validState2D,
                                            broadcast(gtC1D, tileBool, 1));
  Value updated = b.create<arith::SelectOp>(loc, belowDiag, nextAcc, acc);
  b.create<scf::YieldOp>(loc, ValueRange{updated});
  b.setInsertionPointAfter(colLoop);
}

SmallVector<Value> OwnerTiles::exclusiveScanLoop(
    scf::ForOp loop, const ExclusiveScanLoopInfo &scan, const Values &parent,
    Value mask) {
  int64_t B = scan.extent;
  int64_t W = std::max<int64_t>(2, llvm::PowerOf2Ceil(B));
  Type elemT = loop.getResult(0).getType();
  Type vecTy = shaped({W}, elemT);
  Type scan2DTy = shaped({1, W}, elemT);
  Type idxVecTy = shaped({W}, b.getI64Type());

  auto toScalar = [&](Value orig) -> Value {
    Value pv = at(orig, parent);
    if (isa<RankedTensorType>(pv.getType()))
      pv = reduce(pv, "max", 0);
    if (pv.getType().isIndex())
      pv = castElement(pv, b.getI64Type());
    return pv;
  };

  Value activeScalar = reduce(mask, "max", 0);
  Value lanesB = lanes(W);
  Value inB = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, lanesB,
                                      like(integer(B), idxVecTy));
  Value validB =
      b.create<arith::AndIOp>(loc, like(activeScalar, inB.getType()), inB);

  memref::LoadOp srcLd = scan.srcLoad;
  SmallVector<Value, 4> ldCoords;
  SmallVector<int64_t, 4> ldAxes;
  for (Value idx : srcLd.getIndices().drop_back()) {
    ldCoords.push_back(like(toScalar(idx), idxVecTy));
    ldAxes.push_back(0);
  }
  ldCoords.push_back(lanesB);
  ldAxes.push_back(0);
  Value cntVec = read(srcLd.getMemRef(),
                      address(srcLd.getMemRef(), ldCoords, ldAxes),
                      validB);

  Value cnt2D = broadcast(cntVec, scan2DTy, 1);
  Value incl2D =
      op("scan_sum", cnt2D, scan2DTy,
         {b.getNamedAttr("ci.implementation",
                         b.getStringAttr((unit + "::inclusive-scan").str()))})
          ->getResult(0);
  Value inclRaw = reduce(incl2D, "max", 0);
  Value exclRaw = elemT.isF32()
                      ? Value(b.create<arith::SubFOp>(loc, inclRaw, cntVec))
                      : Value(b.create<arith::SubIOp>(loc, inclRaw, cntVec));

  Value initScalar = toScalar(loop.getInitArgs()[0]);
  Value initVec = like(initScalar, vecTy);
  Value exclVec = apply(*scan.addOp, {initVec, exclRaw}, vecTy);
  Value inclVec = apply(*scan.addOp, {initVec, inclRaw}, vecTy);

  for (auto [st, isIncl] : scan.stores) {
    SmallVector<Value, 4> stCoords;
    SmallVector<int64_t, 4> stAxes;
    for (Value idx : st.getIndices().drop_back()) {
      stCoords.push_back(like(toScalar(idx), idxVecTy));
      stAxes.push_back(0);
    }
    stCoords.push_back(lanesB);
    stAxes.push_back(0);
    write(st.getMemRef(), address(st.getMemRef(), stCoords, stAxes),
          isIncl ? inclVec : exclVec, validB);
  }

  Value totalSum = reduce(cntVec, "sum", 0);
  Value nextScalar = b.create<arith::SelectOp>(
      loc, activeScalar, apply(*scan.addOp, {initScalar, totalSum}, elemT),
      initScalar);
  return {like(nextScalar, shaped({1}, elemT))};
}

void OwnerTiles::boundedHistogramLoop(scf::ForOp loop,
                                      const BoundedHistogramLoopInfo &hist,
                                      const Values &parent, Value mask) {
  int64_t B = hist.bins;
  int64_t B_pad = std::max<int64_t>(2, llvm::PowerOf2Ceil(B));
  int64_t P = 64;
  Type elemT = hist.addOp->getResult(0).getType();
  Type histVecTy = shaped({B_pad}, elemT);
  Type pIdxTy = shaped({P}, b.getI64Type());
  Type pI32Ty = shaped({P}, b.getI32Type());
  Type bIdxTy = shaped({B_pad}, b.getI64Type());
  Type bI32Ty = shaped({B_pad}, b.getI32Type());
  Type bpBoolTy = shaped({B_pad, P}, b.getI1Type());
  Type bpI32Ty = shaped({B_pad, P}, b.getI32Type());

  auto toScalar = [&](Value orig) -> Value {
    Value pv = at(orig, parent);
    if (isa<RankedTensorType>(pv.getType()))
      pv = reduce(pv, "max", 0);
    if (pv.getType().isIndex())
      pv = castElement(pv, b.getI64Type());
    return pv;
  };

  Value activeScalar = reduce(mask, "max", 0);
  Value lanesB = lanes(B_pad);
  Value inB = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, lanesB,
                                      like(integer(B), bIdxTy));
  Value validB =
      b.create<arith::AndIOp>(loc, like(activeScalar, inB.getType()), inB);

  memref::LoadOp histLd = hist.histLoad;
  memref::StoreOp histSt = hist.histStore;
  SmallVector<Value, 4> hCoords;
  SmallVector<int64_t, 4> hAxes;
  for (Value idx : histSt.getIndices().drop_back()) {
    hCoords.push_back(like(toScalar(idx), bIdxTy));
    hAxes.push_back(0);
  }
  hCoords.push_back(lanesB);
  hAxes.push_back(0);
  Value histInit = read(histSt.getMemRef(),
                        address(histSt.getMemRef(), hCoords, hAxes),
                        validB);

  Value hiScalar = toScalar(loop.getUpperBound());
  auto pLoop = b.create<scf::ForOp>(loc, integer(0), hiScalar, integer(P),
                                    ValueRange{histInit});
  pLoop->setAttr(symbol("runtime_range"), b.getUnitAttr());
  b.setInsertionPointToStart(pLoop.getBody());

  Value base = pLoop.getInductionVar();
  Value curHist = pLoop.getRegionIterArgs()[0];
  Value colVec = stepped(base, P);
  Value inExt = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, colVec,
                                        like(hiScalar, pIdxTy));
  Value validP =
      b.create<arith::AndIOp>(loc, like(activeScalar, inExt.getType()), inExt);

  Value iv = loop.getInductionVar();
  llvm::DenseMap<Value, Value> vecEnv;
  auto lift1D = [&](Value orig) -> Value {
    if (orig == iv)
      return colVec;
    if (auto it = vecEnv.find(orig); it != vecEnv.end())
      return it->second;
    Value sv = toScalar(orig);
    return like(sv, shaped({P}, elem(sv.getType())));
  };

  for (Operation &inst : loop.getBody()->without_terminator()) {
    if (&inst == histLd.getOperation() ||
        &inst == histSt.getOperation() || &inst == hist.addOp)
      continue;
    if (auto c = dyn_cast<arith::ConstantOp>(inst)) {
      Value cv = constant(c);
      vecEnv[inst.getResult(0)] = like(cv, shaped({P}, elem(cv.getType())));
      continue;
    }
    if (auto ld = dyn_cast<memref::LoadOp>(inst)) {
      SmallVector<Value, 4> coords;
      SmallVector<int64_t, 4> axes;
      bool usesIv = false;
      for (Value idx : ld.getIndices()) {
        if (idx == iv) {
          coords.push_back(colVec);
          axes.push_back(0);
          usesIv = true;
        } else {
          coords.push_back(like(toScalar(idx), pIdxTy));
          axes.push_back(0);
        }
      }
      if (usesIv) {
        vecEnv[ld.getResult()] =
            read(ld.getMemRef(), address(ld.getMemRef(), coords, axes), validP);
      } else {
        SmallVector<Value, 4> sCoords;
        SmallVector<int64_t, 4> sAxes;
        for (Value idx : ld.getIndices()) {
          sCoords.push_back(like(toScalar(idx), shaped({1}, b.getI64Type())));
          sAxes.push_back(0);
        }
        Value rv =
            read(ld.getMemRef(), address(ld.getMemRef(), sCoords, sAxes), mask);
        Value rScalar = reduce(rv, "max", 0);
        vecEnv[ld.getResult()] =
            like(rScalar, shaped({P}, elem(rScalar.getType())));
      }
      continue;
    }
    SmallVector<Value, 3> args;
    for (Value opnd : inst.getOperands())
      args.push_back(lift1D(opnd));
    Type e = elem(inst.getResult(0).getType());
    if (e.isIndex())
      e = b.getI64Type();
    vecEnv[inst.getResult(0)] = apply(inst, args, shaped({P}, e));
  }

  Value bin1D = lift1D(hist.binIdx);
  Value ge0 = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, bin1D,
                                      like(integer(0), pIdxTy));
  Value ltB = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, bin1D,
                                      like(integer(B), pIdxTy));
  Value validBinP = b.create<arith::AndIOp>(
      loc, validP, b.create<arith::AndIOp>(loc, ge0, ltB));

  Value binI32 = b.create<arith::TruncIOp>(loc, pI32Ty, bin1D);
  Value lanesBI32 = b.create<arith::TruncIOp>(loc, bI32Ty, lanes(B_pad));
  Value matchBP = b.create<arith::AndIOp>(
      loc,
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                              broadcast(lanesBI32, bpI32Ty, 0),
                              broadcast(binI32, bpI32Ty, 1)),
      broadcast(validBinP, bpBoolTy, 1));

  Value panelCounts;
  if (elemT.isInteger(32) && matchPattern(hist.deltaVal, m_One())) {
    Value matchI32 = b.create<arith::ExtUIOp>(loc, bpI32Ty, matchBP);
    panelCounts = reduce(matchI32, "sum", 1);
  } else {
    Type bpValTy = shaped({B_pad, P}, elemT);
    Value zeroScalar = elemT.isF32()
                           ? scalar(0.0f)
                           : Value(b.create<arith::ConstantIntOp>(loc, 0, 32));
    Value delta2D = broadcast(lift1D(hist.deltaVal), bpValTy, 1);
    Value contrib = b.create<arith::SelectOp>(loc, matchBP, delta2D,
                                              like(zeroScalar, bpValTy));
    panelCounts = reduce(contrib, "sum", 1);
  }
  Value nextHist = apply(*hist.addOp, {curHist, panelCounts}, histVecTy);
  b.create<scf::YieldOp>(loc, ValueRange{nextHist});
  b.setInsertionPointAfter(pLoop);

  write(histSt.getMemRef(),
        address(histSt.getMemRef(), hCoords, hAxes), pLoop.getResult(0),
        validB);
}

void OwnerTiles::boundedScatterLoop(scf::ForOp loop,
                                    const BoundedScatterLoopInfo &scat,
                                    const Values &parent, Value mask) {
  int64_t B = scat.bins;
  int64_t B_pad = std::max<int64_t>(2, llvm::PowerOf2Ceil(B));
  int64_t P = 32;
  Type pIdxTy = shaped({P}, b.getI64Type());
  Type pI32Ty = shaped({P}, b.getI32Type());
  Type bIdxTy = shaped({B_pad}, b.getI64Type());
  Type bI32Ty = shaped({B_pad}, b.getI32Type());
  Type bpBoolTy = shaped({B_pad, P}, b.getI1Type());
  Type bpI32Ty = shaped({B_pad, P}, b.getI32Type());
  Type ppBoolTy = shaped({P, P}, b.getI1Type());
  Type ppI32Ty = shaped({P, P}, b.getI32Type());

  auto toScalar = [&](Value orig) -> Value {
    Value pv = at(orig, parent);
    if (isa<RankedTensorType>(pv.getType()))
      pv = reduce(pv, "max", 0);
    if (pv.getType().isIndex())
      pv = castElement(pv, b.getI64Type());
    return pv;
  };

  Value activeScalar = reduce(mask, "max", 0);
  Value lanesB = lanes(B_pad);
  Value inB = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, lanesB,
                                      like(integer(B), bIdxTy));
  Value validB =
      b.create<arith::AndIOp>(loc, like(activeScalar, inB.getType()), inB);

  memref::LoadOp curLd = scat.cursorLoad;
  memref::StoreOp curSt = scat.cursorStore;
  SmallVector<Value, 4> cCoords;
  SmallVector<int64_t, 4> cAxes;
  for (Value idx : curSt.getIndices().drop_back()) {
    cCoords.push_back(like(toScalar(idx), bIdxTy));
    cAxes.push_back(0);
  }
  cCoords.push_back(lanesB);
  cAxes.push_back(0);
  Value curInit = read(curSt.getMemRef(),
                       address(curSt.getMemRef(), cCoords, cAxes),
                       validB);

  Value hiScalar = toScalar(loop.getUpperBound());
  auto pLoop = b.create<scf::ForOp>(loc, integer(0), hiScalar, integer(P),
                                    ValueRange{curInit});
  pLoop->setAttr(symbol("runtime_range"), b.getUnitAttr());
  b.setInsertionPointToStart(pLoop.getBody());

  Value base = pLoop.getInductionVar();
  Value curVec = pLoop.getRegionIterArgs()[0];
  Value colVec = stepped(base, P);
  Value inExt = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, colVec,
                                        like(hiScalar, pIdxTy));
  Value validP =
      b.create<arith::AndIOp>(loc, like(activeScalar, inExt.getType()), inExt);

  Value iv = loop.getInductionVar();
  llvm::DenseMap<Value, Value> vecEnv;
  auto lift1D = [&](Value orig) -> Value {
    if (orig == iv)
      return colVec;
    if (auto it = vecEnv.find(orig); it != vecEnv.end())
      return it->second;
    Value sv = toScalar(orig);
    return like(sv, shaped({P}, elem(sv.getType())));
  };

  for (Operation &inst : loop.getBody()->without_terminator()) {
    if (&inst == curLd.getOperation() ||
        &inst == curSt.getOperation() || &inst == scat.cursorAdd ||
        &inst == scat.posCast || isa<memref::StoreOp>(inst))
      continue;
    if (auto c = dyn_cast<arith::ConstantOp>(inst)) {
      Value cv = constant(c);
      vecEnv[inst.getResult(0)] = like(cv, shaped({P}, elem(cv.getType())));
      continue;
    }
    if (auto ld = dyn_cast<memref::LoadOp>(inst)) {
      SmallVector<Value, 4> coords;
      SmallVector<int64_t, 4> axes;
      bool usesIv = false;
      for (Value idx : ld.getIndices()) {
        if (idx == iv) {
          coords.push_back(colVec);
          axes.push_back(0);
          usesIv = true;
        } else {
          coords.push_back(like(toScalar(idx), pIdxTy));
          axes.push_back(0);
        }
      }
      if (usesIv) {
        vecEnv[ld.getResult()] =
            read(ld.getMemRef(), address(ld.getMemRef(), coords, axes), validP);
      } else {
        SmallVector<Value, 4> sCoords;
        SmallVector<int64_t, 4> sAxes;
        for (Value idx : ld.getIndices()) {
          sCoords.push_back(like(toScalar(idx), shaped({1}, b.getI64Type())));
          sAxes.push_back(0);
        }
        Value rv =
            read(ld.getMemRef(), address(ld.getMemRef(), sCoords, sAxes), mask);
        Value rScalar = reduce(rv, "max", 0);
        vecEnv[ld.getResult()] =
            like(rScalar, shaped({P}, elem(rScalar.getType())));
      }
      continue;
    }
    SmallVector<Value, 3> args;
    for (Value opnd : inst.getOperands())
      args.push_back(lift1D(opnd));
    Type e = elem(inst.getResult(0).getType());
    if (e.isIndex())
      e = b.getI64Type();
    vecEnv[inst.getResult(0)] = apply(inst, args, shaped({P}, e));
  }

  Value bin1D = lift1D(scat.binIdx);
  Value ge0 = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, bin1D,
                                      like(integer(0), pIdxTy));
  Value ltB = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, bin1D,
                                      like(integer(B), pIdxTy));
  Value validBinP = b.create<arith::AndIOp>(
      loc, validP, b.create<arith::AndIOp>(loc, ge0, ltB));

  Value binI32 = b.create<arith::TruncIOp>(loc, pI32Ty, bin1D);
  Value lanesBI32 = b.create<arith::TruncIOp>(loc, bI32Ty, lanes(B_pad));
  Value matchBP = b.create<arith::AndIOp>(
      loc,
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                              broadcast(lanesBI32, bpI32Ty, 0),
                              broadcast(binI32, bpI32Ty, 1)),
      broadcast(validBinP, bpBoolTy, 1));

  Value zeroBP =
      like(b.create<arith::ConstantIntOp>(loc, 0, 32), bpI32Ty);
  Value baseCursor1D = reduce(
      b.create<arith::SelectOp>(loc, matchBP, broadcast(curVec, bpI32Ty, 0),
                                zeroBP),
      "sum", 0);

  Value lanesPI32 = b.create<arith::TruncIOp>(loc, pI32Ty, lanes(P));
  Value sameBinPP =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                              broadcast(binI32, ppI32Ty, 0),
                              broadcast(binI32, ppI32Ty, 1));
  Value afterPP =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                              broadcast(lanesPI32, ppI32Ty, 0),
                              broadcast(lanesPI32, ppI32Ty, 1));
  Value priorValidPP = b.create<arith::AndIOp>(
      loc, b.create<arith::AndIOp>(loc, sameBinPP, afterPP),
      broadcast(validBinP, ppBoolTy, 1));
  Value rank1D = reduce(
      b.create<arith::ExtUIOp>(loc, ppI32Ty, priorValidPP), "sum", 1);

  Value dstI32 = b.create<arith::AddIOp>(loc, baseCursor1D, rank1D);
  Value dstI64 = b.create<arith::ExtSIOp>(loc, pIdxTy, dstI32);

  for (auto st : scat.scatterStores) {
    SmallVector<Value, 4> sCoords;
    SmallVector<int64_t, 4> sAxes;
    for (Value idx : st.getIndices().drop_back()) {
      sCoords.push_back(like(toScalar(idx), pIdxTy));
      sAxes.push_back(0);
    }
    sCoords.push_back(dstI64);
    sAxes.push_back(0);
    write(st.getMemRef(), address(st.getMemRef(), sCoords, sAxes),
          lift1D(st.getValue()), validBinP);
  }

  Value panelCounts =
      reduce(b.create<arith::ExtUIOp>(loc, bpI32Ty, matchBP), "sum", 1);
  Value nextCur = b.create<arith::AddIOp>(loc, curVec, panelCounts);
  b.create<scf::YieldOp>(loc, ValueRange{nextCur});
  b.setInsertionPointAfter(pLoop);

  write(curSt.getMemRef(),
        address(curSt.getMemRef(), cCoords, cAxes),
        pLoop.getResult(0), validB);
}

SmallVector<Value> OwnerTiles::selectionLoop(scf::ForOp loop,
                                             const OrderedSelectionInfo &sel,
                                             const Values &parent, Value mask) {
  Value rawHi = widen(at(loop.getUpperBound(), parent));
  Value hiScalar = isa<RankedTensorType>(rawHi.getType())
                       ? reduce(rawHi, "max", 0)
                       : rawHi;
  int64_t W = 1024;
  if (auto c = hiScalar.getDefiningOp<arith::ConstantOp>())
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
      if (ia.getInt() > 0)
        W = std::clamp<int64_t>(llvm::PowerOf2Ceil(ia.getInt()), 64, 1024);
  unsigned k = sel.k;
  Type v1Ty = shaped({1}, b.getF32Type());
  Type i1Ty = shaped({1}, b.getI32Type());
  Type v2Ty = shaped({1, W}, b.getF32Type());
  Type i2Ty = shaped({1, W}, b.getI32Type());
  Type b1Ty = shaped({1}, b.getI1Type());
  Type b2Ty = shaped({1, W}, b.getI1Type());

  Value negInf1D = broadcast(scalar(-INFINITY), v1Ty);
  Value negInf2D = broadcast(scalar(-INFINITY), v2Ty);
  Value negOne1D = broadcast(b.create<arith::ConstantIntOp>(loc, -1, 32), i1Ty);
  Value intMax2D =
      broadcast(b.create<arith::ConstantIntOp>(loc, INT32_MAX, 32), i2Ty);

  SmallVector<Value> init;
  for (unsigned s = 0; s < k; ++s)
    init.push_back(negInf1D);
  for (unsigned s = 0; s < k; ++s)
    init.push_back(negOne1D);

  auto l = b.create<scf::ForOp>(loc, integer(0), hiScalar, integer(W), init);
  l->setAttr(symbol("runtime_range"), b.getUnitAttr());
  b.setInsertionPointToStart(l.getBody());

  Value base = l.getInductionVar();
  Value colVec = stepped(base, W);
  Value inBounds = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                           colVec, like(hiScalar, colVec.getType()));
  Value valid2D = b.create<arith::AndIOp>(loc, broadcast(mask, b2Ty, 0),
                                          broadcast(inBounds, b2Ty, 1));
  Value iv = loop.getInductionVar();
  llvm::DenseMap<Value, Value> vecEnv;
  auto lift2D = [&](Value orig) -> Value {
    if (orig == iv)
      return broadcast(colVec, shaped({1, W}, b.getI64Type()), 1);
    if (auto it = vecEnv.find(orig); it != vecEnv.end())
      return it->second;
    Value pv = at(orig, parent);
    Type e = elem(pv.getType());
    if (e.isIndex())
      e = b.getI64Type();
    if (elem(pv.getType()) != e)
      pv = castElement(pv, e);
    return broadcast(pv, shaped({1, W}, e), 0);
  };
  for (Operation &inst : loop.getBody()->without_terminator()) {
    if (sel.chainOps.contains(&inst))
      continue;
    if (auto c = dyn_cast<arith::ConstantOp>(inst)) {
      Value cv = constant(c);
      vecEnv[inst.getResult(0)] =
          broadcast(cv, shaped({1, W}, elem(cv.getType())), 0);
      continue;
    }
    if (auto ld = dyn_cast<memref::LoadOp>(inst)) {
      Value view = ld.getMemRef();
      SmallVector<Value, 4> coords;
      SmallVector<int64_t, 4> axes;
      for (Value idx : ld.getIndices().drop_back()) {
        coords.push_back(
            broadcast(widen(at(idx, parent)), shaped({1}, b.getI64Type())));
        axes.push_back(0);
      }
      coords.push_back(colVec);
      axes.push_back(1);
      vecEnv[ld.getResult()] = read(view, address(view, coords, axes), valid2D);
      continue;
    }
    SmallVector<Value, 3> args;
    for (Value opnd : inst.getOperands())
      args.push_back(lift2D(opnd));
    Type e = elem(inst.getResult(0).getType());
    if (e.isIndex())
      e = b.getI64Type();
    vecEnv[inst.getResult(0)] = apply(inst, args, shaped({1, W}, e));
  }

  Value xPanel = lift2D(sel.candidateVal);
  Value finiteGtNegInf =
      b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, xPanel, negInf2D);
  Value validElem = b.create<arith::AndIOp>(loc, valid2D, finiteGtNegInf);
  Value curX = select(validElem, xPanel, negInf2D);
  Value colI32_1D = b.create<arith::TruncIOp>(
      loc, RankedTensorType::get({W}, b.getI32Type()), colVec);
  Value colI32 = broadcast(colI32_1D, i2Ty, 1);

  SmallVector<Value, 16> runV, runI;
  for (unsigned s = 0; s < k; ++s) {
    runV.push_back(l.getRegionIterArgs()[s]);
    runI.push_back(l.getRegionIterArgs()[k + s]);
  }

  for (unsigned r = 0; r < k; ++r) {
    Value mx = reduce(curX, "max", 1);
    Value hasValid =
        b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, mx, negInf1D);
    Value isMax = b.create<arith::AndIOp>(
        loc, broadcast(hasValid, b2Ty, 0),
        b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, curX,
                                broadcast(mx, v2Ty, 0)));
    Value winIdx = reduce(select(isMax, colI32, intMax2D), "min", 1);
    if (r + 1 < k) {
      Value isWin = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, colI32, broadcast(winIdx, i2Ty, 0));
      curX = select(isWin, negInf2D, curX);
    }
    Value candV = mx;
    Value candI = winIdx;
    for (unsigned s = r; s < k; ++s) {
      Value cmp = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, candV,
                                          runV[s]);
      Value nv = select(cmp, candV, runV[s]);
      Value ni = select(cmp, candI, runI[s]);
      if (s + 1 < k) {
        candV = select(cmp, runV[s], candV);
        candI = select(cmp, runI[s], candI);
      }
      runV[s] = nv;
      runI[s] = ni;
    }
  }

  SmallVector<Value> yielded;
  yielded.append(runV.begin(), runV.end());
  yielded.append(runI.begin(), runI.end());
  b.create<scf::YieldOp>(loc, yielded);
  b.setInsertionPointAfter(l);
  return SmallVector<Value>(l.getResults());
}

void OwnerTiles::associativeScatterLoop(
    scf::ForOp loop, const AssociativeScatterLoopInfo &info,
    const Values &parent, Value mask) {
  auto fsAttr = space.getAttrOfType<IntegerAttr>("block.fold_split");
  int64_t foldSplit = fsAttr ? std::max<int64_t>(1, fsAttr.getInt()) : 1;
  Value rawHi = widen(at(loop.getUpperBound(), parent));
  Value hiScalar = isa<RankedTensorType>(rawHi.getType())
                       ? reduce(rawHi, "max", 0)
                       : rawHi;

  llvm::SmallPtrSet<Operation *, 4> skipOps;
  llvm::DenseMap<Operation *, const AssociativeScatterUpdate *> storeUpdates;
  for (const auto &u : info.updates) {
    memref::LoadOp pLd = u.pairedLoad;
    memref::StoreOp sOp = u.store;
    skipOps.insert(pLd.getOperation());
    skipOps.insert(u.combiner);
    storeUpdates[sOp.getOperation()] = &u;
  }

  // Check whether all memory coordinate expressions separate fold-axis (iv)
  // and owner-axis dependencies so each CTA step can issue a 2D [P_n, rows]
  // coalesced panel.
  llvm::DenseSet<Value> ivDep{loop.getInductionVar()};
  llvm::DenseSet<Value> ownerDep;
  for (auto [k, v] : parent)
    if (isa<RankedTensorType>(v.getType()))
      ownerDep.insert(k);
  bool separable2D =
      (access.scalarBroadcastRead && rows >= 16 && info.extent >= 32);
  for (Operation &inst : loop.getBody()->without_terminator()) {
    if (skipOps.contains(&inst))
      continue;
    if (auto st = dyn_cast<memref::StoreOp>(inst)) {
      for (Value c : st.getIndices())
        if (ivDep.contains(c) && ownerDep.contains(c))
          separable2D = false;
      continue;
    }
    if (auto ld = dyn_cast<memref::LoadOp>(inst)) {
      bool anyIv = false, anyOwn = false;
      for (Value c : ld.getIndices()) {
        bool ci = ivDep.contains(c), co = ownerDep.contains(c);
        if (ci && co)
          separable2D = false;
        anyIv |= ci;
        anyOwn |= co;
      }
      if (anyIv)
        ivDep.insert(ld.getResult());
      if (anyOwn)
        ownerDep.insert(ld.getResult());
      continue;
    }
    if (inst.getNumResults() == 1) {
      bool anyIv = llvm::any_of(inst.getOperands(),
                                [&](Value a) { return ivDep.contains(a); });
      bool anyOwn = llvm::any_of(inst.getOperands(),
                                 [&](Value a) { return ownerDep.contains(a); });
      if (anyIv)
        ivDep.insert(inst.getResult(0));
      if (anyOwn)
        ownerDep.insert(inst.getResult(0));
    }
  }

  int64_t totalOwners =
      space.getAttrOfType<IntegerAttr>("block.owners").getInt();
  bool exactOwners = (mask == active && rows == logical && totalOwners > 0 &&
                      totalOwners % rows == 0);

  if (separable2D) {
    constexpr int64_t P = 8;
    int64_t itemsPerCta = llvm::divideCeil(info.extent, foldSplit);
    int64_t stepsPerCta = std::max<int64_t>(1, llvm::divideCeil(itemsPerCta, P));
    int64_t ctaSpan = stepsPerCta * P;
    bool exactFold = (foldSplit * ctaSpan == info.extent);
    bool unrollConst = (stepsPerCta <= 8);

    Value ctaBase = foldSplit > 1
                        ? Value(b.create<arith::MulIOp>(loc, pid(1),
                                                        integer(ctaSpan)))
                        : integer(0);
    scf::ForOp l = b.create<scf::ForOp>(loc, integer(0), integer(stepsPerCta),
                                        integer(1));
    if (!unrollConst)
      l->setAttr(symbol("runtime_range"), b.getUnitAttr());
    b.setInsertionPointToStart(l.getBody());

    Value stepBase = b.create<arith::AddIOp>(
        loc, ctaBase,
        b.create<arith::MulIOp>(loc, l.getInductionVar(), integer(P)));
    Value nVec = stepped(stepBase, P);
    Type bool2D = shaped({P, rows}, b.getI1Type());
    Value validN;
    if (!exactFold)
      validN = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, nVec,
                                       like(hiScalar, nVec.getType()));
    Value valid2D;
    if (!exactFold || !exactOwners) {
      Value mOwn = broadcast(mask, bool2D, 1);
      valid2D = exactFold ? mOwn
                : exactOwners
                    ? broadcast(validN, bool2D, 0)
                    : Value(b.create<arith::AndIOp>(
                          loc, broadcast(validN, bool2D, 0), mOwn));
    }

    Type pIdxTy = shaped({P}, b.getI64Type());
    Type rIdxTy = shaped({rows}, b.getI64Type());
    Values env = parent;
    env[loop.getInductionVar()] = nVec;

    llvm::DenseSet<Value> isAxis0{nVec};
    llvm::DenseSet<Value> isAxis1;
    for (auto [k, v] : parent)
      if (isa<RankedTensorType>(v.getType()))
        isAxis1.insert(v);

    auto projectTo2D = [&](Value v) -> Value {
      Type elemTy = elem(v.getType());
      Type dst = shaped({P, rows}, elemTy);
      auto rt = dyn_cast<RankedTensorType>(v.getType());
      if (!rt)
        return like(v, dst, -1);
      if (rt.getRank() == 2)
        return v;
      int ax = isAxis0.contains(v) ? 0 : 1;
      return like(v, dst, ax);
    };

    for (Operation &inst : loop.getBody()->without_terminator()) {
      if (skipOps.contains(&inst))
        continue;
      if (auto itUpd = storeUpdates.find(&inst); itUpd != storeUpdates.end()) {
        const auto *u = itUpd->second;
        memref::StoreOp stOp = u->store;
        Value base = u->targetView;
        SmallVector<Value, 4> coords;
        SmallVector<int64_t, 4> axes;
        bool saw0 = false, saw1 = false;
        for (Value c : stOp.getIndices()) {
          Value mc = widen(at(c, env));
          if (ivDep.contains(c)) {
            mc = isa<RankedTensorType>(mc.getType()) ? mc : like(mc, pIdxTy, -1);
            coords.push_back(mc);
            axes.push_back(0);
            saw0 = true;
          } else if (ownerDep.contains(c)) {
            mc = isa<RankedTensorType>(mc.getType()) ? mc : like(mc, rIdxTy, -1);
            coords.push_back(mc);
            axes.push_back(1);
            saw1 = true;
          } else {
            coords.push_back(mc);
            axes.push_back(-1);
          }
        }
        if (!saw0) {
          // Ensure rank-2 [P, rows] tile if store index had no iv coord
          coords[0] = like(coords[0], pIdxTy, -1);
          axes[0] = 0;
        }
        (void)saw1;
        Value ptr = address(base, coords, axes);
        Value contrib2D = projectTo2D(at(u->contribution, env));
        atomicAdd(base, ptr, contrib2D, valid2D);
        continue;
      }
      if (auto c = dyn_cast<arith::ConstantOp>(inst)) {
        env[inst.getResult(0)] = constant(c);
        continue;
      }
      if (auto d = dyn_cast<memref::DimOp>(inst)) {
        env[inst.getResult(0)] =
            integer(shape(d.getSource())[*d.getConstantIndex()]);
        continue;
      }
      if (auto ld = dyn_cast<memref::LoadOp>(inst)) {
        Value base = ld.getMemRef();
        SmallVector<Value, 4> coords;
        SmallVector<int64_t, 4> axes;
        bool has0 = false, has1 = false;
        for (Value c : ld.getIndices()) {
          if (ivDep.contains(c))
            has0 = true;
          else if (ownerDep.contains(c))
            has1 = true;
        }
        for (Value c : ld.getIndices()) {
          Value mc = widen(at(c, env));
          if (ivDep.contains(c)) {
            mc = isa<RankedTensorType>(mc.getType()) ? mc : like(mc, pIdxTy, -1);
            coords.push_back(mc);
            axes.push_back(0);
          } else if (ownerDep.contains(c)) {
            mc = isa<RankedTensorType>(mc.getType()) ? mc : like(mc, rIdxTy, -1);
            coords.push_back(mc);
            axes.push_back(has0 ? 1 : 0);
          } else {
            coords.push_back(mc);
            axes.push_back(-1);
          }
        }
        Value ldMask;
        if (has0 && has1)
          ldMask = valid2D;
        else if (has0)
          ldMask = validN;
        else if (has1 && !exactOwners)
          ldMask = mask;
        Value res = read(base, address(base, coords, axes), ldMask);
        if (has0 && !has1)
          isAxis0.insert(res);
        else if (has1 && !has0)
          isAxis1.insert(res);
        env[inst.getResult(0)] = res;
        continue;
      }
      Type e = elem(inst.getResult(0).getType());
      if (e.isIndex())
        e = b.getI64Type();
      bool res0 = ivDep.contains(inst.getResult(0));
      bool res1 = ownerDep.contains(inst.getResult(0));
      SmallVector<Value, 4> args;
      Type dstTy = e;
      if (res0 && res1) {
        dstTy = shaped({P, rows}, e);
        for (Value a : inst.getOperands())
          args.push_back(projectTo2D(at(a, env)));
      } else if (res0 || res1) {
        int64_t len = res0 ? P : rows;
        dstTy = shaped({len}, e);
        for (Value a : inst.getOperands()) {
          Value av = at(a, env);
          args.push_back(isa<RankedTensorType>(av.getType())
                             ? av
                             : like(av, shaped({len}, elem(av.getType())), -1));
        }
      } else {
        for (Value a : inst.getOperands())
          args.push_back(at(a, env));
      }
      Value outVal = apply(inst, args, dstTy);
      if (res0 && !res1)
        isAxis0.insert(outVal);
      else if (res1 && !res0)
        isAxis1.insert(outVal);
      env[inst.getResult(0)] = outVal;
    }
    b.setInsertionPointAfter(l);
    return;
  }

  Value loScalar = integer(0), limScalar = hiScalar;
  if (foldSplit > 1) {
    int64_t chunk = llvm::divideCeil(info.extent, foldSplit);
    loScalar = b.create<arith::MulIOp>(loc, pid(1), integer(chunk));
    limScalar = b.create<arith::MinSIOp>(
        loc, b.create<arith::AddIOp>(loc, loScalar, integer(chunk)), hiScalar);
  }
  auto l = b.create<scf::ForOp>(loc, loScalar, limScalar, integer(1));
  l->setAttr(symbol("runtime_range"), b.getUnitAttr());
  b.setInsertionPointToStart(l.getBody());

  Type ixTy = shaped({rows}, b.getI64Type());

  Values env = parent;
  env[loop.getInductionVar()] = l.getInductionVar();
  for (Operation &inst : loop.getBody()->without_terminator()) {
    if (skipOps.contains(&inst))
      continue;
    if (auto itUpd = storeUpdates.find(&inst); itUpd != storeUpdates.end()) {
      const auto *u = itUpd->second;
      memref::StoreOp stOp = u->store;
      Value base = u->targetView;
      SmallVector<Value, 4> coords;
      SmallVector<int64_t, 4> axes;
      for (Value c : stOp.getIndices()) {
        coords.push_back(broadcast(widen(at(c, env)), ixTy));
        axes.push_back(0);
      }
      Value ptr = address(base, coords, axes);
      Type valTy =
          shaped({rows}, cast<MemRefType>(base.getType()).getElementType());
      Value contribVal = broadcast(at(u->contribution, env), valTy);
      atomicAdd(base, ptr, contribVal, exactOwners ? Value{} : mask);
      continue;
    }
    if (auto c = dyn_cast<arith::ConstantOp>(inst)) {
      env[inst.getResult(0)] = constant(c);
      continue;
    }
    if (auto d = dyn_cast<memref::DimOp>(inst)) {
      env[inst.getResult(0)] =
          integer(shape(d.getSource())[*d.getConstantIndex()]);
      continue;
    }
    if (auto ld = dyn_cast<memref::LoadOp>(inst)) {
      Value base = ld.getMemRef();
      SmallVector<Value, 4> mapped;
      bool allScalar = true;
      for (Value c : ld.getIndices()) {
        Value mc = widen(at(c, env));
        if (isa<RankedTensorType>(mc.getType()))
          allScalar = false;
        mapped.push_back(mc);
      }
      if (allScalar && access.scalarBroadcastRead) {
        SmallVector<int64_t, 4> sAxes(mapped.size(), -1);
        Value sVal = read(base, address(base, mapped, sAxes));
        env[inst.getResult(0)] = broadcast(
            sVal,
            shaped({rows}, cast<MemRefType>(base.getType()).getElementType()));
      } else {
        SmallVector<Value, 4> coords;
        SmallVector<int64_t, 4> axes;
        for (Value mc : mapped) {
          coords.push_back(broadcast(mc, ixTy));
          axes.push_back(0);
        }
        env[inst.getResult(0)] = read(base, address(base, coords, axes),
                                      exactOwners ? Value{} : mask);
      }
      continue;
    }
    SmallVector<Value, 4> args;
    Type shapeHint;
    for (Value a : inst.getOperands()) {
      Value av = at(a, env);
      args.push_back(av);
      if (auto tt = dyn_cast<RankedTensorType>(av.getType());
          tt && (!shapeHint ||
                 tt.getRank() > cast<RankedTensorType>(shapeHint).getRank()))
        shapeHint = tt;
    }
    Type e = elem(inst.getResult(0).getType());
    if (e.isIndex())
      e = b.getI64Type();
    env[inst.getResult(0)] =
        apply(inst, args, shapeHint ? retype(shapeHint, e) : e);
  }
  b.setInsertionPointAfter(l);
}

SmallVector<Value> OwnerTiles::serialLoop(scf::ForOp loop, const Values &parent,
                                          Value mask) {
  if (space.hasAttr("block.fold_split")) {
    AssociativeScatterLoopInfo assocInfo;
    if (matchAssociativeScatterLoop(loop, fn, nullptr, &assocInfo)) {
      associativeScatterLoop(loop, assocInfo, parent, mask);
      return {};
    }
  }
  if (canVectorizeIndependentLoop(loop)) {
    vectorizedLoop(loop, parent, mask);
    return {};
  }
  TriangularRecurrenceInfo tri;
  if (matchTriangularRecurrence(loop, fn, nullptr, &tri)) {
    triangularLoop(loop, tri, parent, mask);
    return {};
  }
  if (rows == 1) {
    OrderedSelectionInfo sel;
    if (matchOrderedSelectionChain(loop, &sel))
      return selectionLoop(loop, sel, parent, mask);
    ExclusiveScanLoopInfo scanInfo;
    if (matchExclusiveScanLoop(loop, fn, nullptr, &scanInfo))
      return exclusiveScanLoop(loop, scanInfo, parent, mask);
    BoundedHistogramLoopInfo histInfo;
    if (matchBoundedHistogramLoop(loop, fn, nullptr, &histInfo)) {
      boundedHistogramLoop(loop, histInfo, parent, mask);
      return {};
    }
    BoundedScatterLoopInfo scatInfo;
    if (matchBoundedScatterLoop(loop, fn, nullptr, &scatInfo)) {
      boundedScatterLoop(loop, scatInfo, parent, mask);
      return {};
    }
  }
  auto lo = at(loop.getLowerBound(), parent),
       hi = at(loop.getUpperBound(), parent), step = at(loop.getStep(), parent);
  bool ragged = isa<RankedTensorType>(lo.getType()) ||
                isa<RankedTensorType>(hi.getType());
  if (!ragged &&
      llvm::none_of(loop.getInitArgs(),
                    [](Value v) { return isa<ShapedType>(v.getType()); })) {
    auto cLo = lo.getDefiningOp<arith::ConstantOp>();
    auto cHi = hi.getDefiningOp<arith::ConstantOp>();
    auto cStep = step.getDefiningOp<arith::ConstantOp>();
    auto iLo = cLo ? dyn_cast<IntegerAttr>(cLo.getValue()) : IntegerAttr();
    auto iHi = cHi ? dyn_cast<IntegerAttr>(cHi.getValue()) : IntegerAttr();
    auto iStep =
        cStep ? dyn_cast<IntegerAttr>(cStep.getValue()) : IntegerAttr();
    if (iLo && iHi && iStep && iLo.getInt() == 0 && iStep.getInt() == 1 &&
        iHi.getInt() >= 1 && iHi.getInt() <= 16) {
      unsigned bodyOps = 0;
      bool canUnroll = true;
      loop.walk([&](Operation *inner) {
        if (inner == loop.getOperation())
          return WalkResult::advance();
        ++bodyOps;
        if (bodyOps > 48 || isa<scf::ForOp, memref::StoreOp>(inner) ||
            inner->getName().stripDialect() == "slice_write") {
          canUnroll = false;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (canUnroll) {
        SmallVector<Value> state;
        for (Value v : loop.getInitArgs())
          state.push_back(broadcast(at(v, parent), lifted(v)));
        int64_t tripCount = iHi.getInt();
        for (int64_t stepIdx = 0; stepIdx < tripCount; ++stepIdx) {
          Values stepEnv = parent;
          stepEnv[loop.getInductionVar()] = integer(stepIdx);
          for (auto [a, v] : llvm::zip(loop.getRegionIterArgs(), state))
            stepEnv[a] = v;
          auto next = lowerBody(*loop.getBody(), stepEnv, mask);
          for (size_t i = 0; i < state.size(); ++i)
            state[i] = broadcast(next[i], state[i].getType());
        }
        return state;
      }
    }
  }
  auto ownerType = shaped({rows}, b.getI64Type());
  Value start = lo, limit = hi;
  if (ragged) {
    lo = broadcast(lo, ownerType);
    hi = broadcast(hi, ownerType);
    start = integer(0);
    auto span = b.create<arith::SubIOp>(loc, hi, lo);
    limit = reduce(select(mask, span, broadcast(start, ownerType)), "max", 0);
  }
  SmallVector<Value> init;
  for (Value v : loop.getInitArgs())
    init.push_back(broadcast(at(v, parent), lifted(v)));
  auto l = b.create<scf::ForOp>(loc, start, limit, step, init);
  l->setAttr(symbol("runtime_range"), b.getUnitAttr());
  b.setInsertionPointToStart(l.getBody());
  Values env = parent;
  Value iv = l.getInductionVar(), live = mask;
  if (ragged) {
    iv = b.create<arith::AddIOp>(loc, lo, broadcast(iv, ownerType));
    live = b.create<arith::AndIOp>(
        loc, mask,
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, iv, hi));
  }
  env[loop.getInductionVar()] = iv;
  for (auto [a, v] : llvm::zip(loop.getRegionIterArgs(), l.getRegionIterArgs()))
    env[a] = v;
  auto next = lowerBody(*loop.getBody(), env, live);
  for (auto [i, arg] : llvm::enumerate(l.getRegionIterArgs())) {
    next[i] = broadcast(next[i], arg.getType());
    if (ragged)
      next[i] = select(live, next[i], arg);
  }
  if (!next.empty())
    b.create<scf::YieldOp>(loc, next);
  b.setInsertionPointAfter(l);
  return SmallVector<Value>(l.getResults());
}
std::optional<OwnerTiles::OnlinePanelFusion>
OwnerTiles::matchOnlinePanelPair(scf::ForOp maxLoop, scf::ForOp sumLoop) {
  if (!maxLoop->hasAttr("kir.reassociate") ||
      !sumLoop->hasAttr("kir.reassociate") ||
      maxLoop->getAttr("block.contribution_width") !=
          sumLoop->getAttr("block.contribution_width") ||
      maxLoop->hasAttr("block.uniform_tokens") !=
          sumLoop->hasAttr("block.uniform_tokens") ||
      maxLoop->hasAttr("block.full_panels") !=
          sumLoop->hasAttr("block.full_panels") ||
      maxLoop->hasAttr("block.counted_panels") !=
          sumLoop->hasAttr("block.counted_panels"))
    return std::nullopt;
  auto sameBound = [](Value a, Value b) {
    if (a == b)
      return true;
    if (auto c1 = a.getDefiningOp<arith::ConstantOp>(),
             c2 = b.getDefiningOp<arith::ConstantOp>();
        c1 && c2)
      return c1.getValue() == c2.getValue();
    auto d1 = a.getDefiningOp<memref::DimOp>(),
         d2 = b.getDefiningOp<memref::DimOp>();
    return d1 && d2 && d1.getSource() == d2.getSource() &&
           d1.getConstantIndex() &&
           d1.getConstantIndex() == d2.getConstantIndex();
  };
  if (!sameBound(maxLoop.getLowerBound(), sumLoop.getLowerBound()) ||
      !sameBound(maxLoop.getUpperBound(), sumLoop.getUpperBound()))
    return std::nullopt;
  auto floatIs = [](Value v, llvm::function_ref<bool(const APFloat &)> pred) {
    auto c = v ? v.getDefiningOp<arith::ConstantOp>() : arith::ConstantOp();
    auto f = c ? dyn_cast<FloatAttr>(c.getValue()) : FloatAttr();
    return f && f.getType().isF32() && pred(f.getValue());
  };
  auto isNegInf = [&](Value v) {
    return floatIs(
        v, [](const APFloat &x) { return x.isInfinity() && x.isNegative(); });
  };
  auto isPosZero = [&](Value v) {
    return floatIs(
        v, [](const APFloat &x) { return x.isZero() && !x.isNegative(); });
  };
  if (maxLoop.getNumResults() != 1 || !isNegInf(maxLoop.getInitArgs()[0]) ||
      sumLoop.getNumResults() == 0)
    return std::nullopt;
  Value maxY = maxLoop.getBody()->getTerminator()->getOperand(0);
  auto maxComb = maxY.getDefiningOp<arith::MaxNumFOp>();
  if (!maxComb || maxComb.getLhs() != maxLoop.getRegionIterArgs()[0] ||
      !intentdsl::kir::hasFastMath(maxComb, arith::FastMathFlags::reassoc))
    return std::nullopt;
  Value maxCand = maxComb.getRhs();
  Value score1 = maxCand, cond1;
  if (auto br1 = maxCand.getDefiningOp<scf::IfOp>()) {
    if (br1.getNumResults() != 1 || br1.getElseRegion().empty() ||
        !br1.thenBlock()->without_terminator().empty() ||
        !isNegInf(br1.elseBlock()->getTerminator()->getOperand(0)))
      return std::nullopt;
    cond1 = br1.getCondition();
    score1 = br1.thenBlock()->getTerminator()->getOperand(0);
  } else if (auto sel1 = maxCand.getDefiningOp<arith::SelectOp>()) {
    if (!isNegInf(sel1.getFalseValue()))
      return std::nullopt;
    cond1 = sel1.getCondition();
    score1 = sel1.getTrueValue();
  }
  if (!score1.getType().isF32())
    return std::nullopt;
  for (Value initV : sumLoop.getInitArgs()) {
    if (isPosZero(initV))
      continue;
    auto *sp = initV.getDefiningOp();
    if (!sp || sp->getName().stripDialect() != "lane_splat" ||
        !isPosZero(sp->getOperand(0)))
      return std::nullopt;
  }
  bool paired = space.hasAttr("block.pair_axis");
  Value wVal;
  llvm::SmallPtrSet<Operation *, 8> allowedWUsers;
  for (auto [idx, y] :
       llvm::enumerate(sumLoop.getBody()->getTerminator()->getOperands())) {
    auto add = y.getDefiningOp<arith::AddFOp>();
    if (!add || add.getLhs() != sumLoop.getRegionIterArgs()[idx] ||
        !intentdsl::kir::hasFastMath(add, arith::FastMathFlags::reassoc))
      return std::nullopt;
    Value rhs = add.getRhs();
    if (!isa<ShapedType>(rhs.getType())) {
      if (!rhs.getType().isF32() || (wVal && wVal != rhs))
        return std::nullopt;
      wVal = rhs;
      allowedWUsers.insert(add);
    } else if (paired && add->hasAttr("block.masked_dot_update")) {
      auto accOps = intentdsl::block::tileAccumulate(rhs, sumLoop);
      if (!accOps)
        return std::nullopt;
      auto tr = accOps->weights.getDefiningOp<arith::TruncFOp>();
      if (!tr || !accOps->weights.hasOneUse())
        return std::nullopt;
      allowedWUsers.insert(tr);
      Value wSrc = tr.getIn();
      if (wVal && wVal != wSrc)
        return std::nullopt;
      wVal = wSrc;
    } else if (!paired) {
      auto mul = rhs.getDefiningOp<arith::MulFOp>();
      if (!mul || !rhs.hasOneUse())
        return std::nullopt;
      Value foundW;
      for (int s : {0, 1}) {
        auto *sp = mul->getOperand(s).getDefiningOp();
        if (!sp || sp->getName().stripDialect() != "lane_splat" ||
            !mul->getOperand(s).hasOneUse())
          continue;
        Value ws = sp->getOperand(0);
        if (auto ext = ws.getDefiningOp<arith::ExtFOp>();
            ext && ws.hasOneUse()) {
          if (auto tr = ext.getIn().getDefiningOp<arith::TruncFOp>();
              tr && ext.getIn().hasOneUse()) {
            allowedWUsers.insert(tr);
            foundW = tr.getIn();
            break;
          }
        }
        allowedWUsers.insert(sp);
        foundW = ws;
        break;
      }
      if (!foundW || (wVal && wVal != foundW))
        return std::nullopt;
      wVal = foundW;
    } else {
      return std::nullopt;
    }
  }
  if (!wVal || !wVal.getType().isF32())
    return std::nullopt;
  for (Operation *u : wVal.getUsers())
    if (!allowedWUsers.contains(u))
      return std::nullopt;
  Value peakRes = maxLoop.getResult(0);
  Operation *guardOp = nullptr;
  Value expOut = wVal, cond2;
  if (auto br2 = wVal.getDefiningOp<scf::IfOp>()) {
    if (!cond1 || br2.getNumResults() != 1 || br2.getElseRegion().empty() ||
        !isPosZero(br2.elseBlock()->getTerminator()->getOperand(0)))
      return std::nullopt;
    cond2 = br2.getCondition();
    expOut = br2.thenBlock()->getTerminator()->getOperand(0);
    guardOp = br2;
    for (Operation &inner : br2.thenBlock()->without_terminator()) {
      Operation *eDef = expOut.getDefiningOp();
      Operation *argDef =
          eDef && eDef->getNumOperands() > 0 ? eDef->getOperand(0).getDefiningOp()
                                             : nullptr;
      bool isAllowedSub = false;
      if (argDef) {
        for (Value v : argDef->getOperands())
          if (v.getDefiningOp() == &inner)
            isAllowedSub = true;
      }
      if (&inner != eDef && &inner != argDef && !isAllowedSub)
        return std::nullopt;
    }
  } else if (auto sel2 = wVal.getDefiningOp<arith::SelectOp>()) {
    if (!cond1 || !isPosZero(sel2.getFalseValue()))
      return std::nullopt;
    cond2 = sel2.getCondition();
    expOut = sel2.getTrueValue();
    guardOp = sel2;
  } else if (cond1) {
    return std::nullopt;
  }
  Operation *expOp = expOut.getDefiningOp();
  if (!expOp || !isa<math::ExpOp, math::Exp2Op>(expOp) ||
      (guardOp && !expOut.hasOneUse()) ||
      !intentdsl::kir::hasFastMath(expOp, arith::FastMathFlags::reassoc))
    return std::nullopt;
  Value expArg = expOp->getOperand(0);
  if (!expArg.hasOneUse())
    return std::nullopt;
  Operation *mulOp = nullptr;
  Value scaleVal;
  arith::FastMathFlags scaleFlags = arith::FastMathFlags::none;
  if (auto m = expArg.getDefiningOp<arith::MulFOp>();
      m && intentdsl::kir::hasFastMath(m, arith::FastMathFlags::reassoc)) {
    for (int side : {0, 1}) {
      Value candSub = m->getOperand(side);
      Value other = m->getOperand(1 - side);
      if (candSub.getDefiningOp<arith::SubFOp>() && candSub.hasOneUse() &&
          sumLoop.isDefinedOutsideOfLoop(other) && other.getType().isF32()) {
        mulOp = m;
        expArg = candSub;
        scaleVal = other;
        scaleFlags = intentdsl::kir::getFastMathFlags(m);
        break;
      }
    }
  }
  auto subOp = expArg.getDefiningOp<arith::SubFOp>();
  if (!subOp || !expArg.hasOneUse() || subOp.getRhs() != peakRes ||
      !intentdsl::kir::hasFastMath(subOp, arith::FastMathFlags::reassoc))
    return std::nullopt;
  Value score2 = subOp.getLhs();
  if (guardOp && score2.getDefiningOp() &&
      guardOp->isProperAncestor(score2.getDefiningOp()))
    return std::nullopt;
  for (OpOperand &use : peakRes.getUses())
    if (sumLoop->isProperAncestor(use.getOwner()) && use.getOwner() != subOp)
      return std::nullopt;
  llvm::DenseMap<std::pair<Value, Value>, bool> memo;
  std::function<bool(Value, Value)> sameVal = [&](Value a, Value b) -> bool {
    if (a.getType() != b.getType())
      return false;
    if (auto it = memo.find({a, b}); it != memo.end())
      return it->second;
    bool ok = false;
    if (a == maxLoop.getInductionVar() || b == sumLoop.getInductionVar()) {
      ok = (a == maxLoop.getInductionVar() && b == sumLoop.getInductionVar());
    } else if (maxLoop.isDefinedOutsideOfLoop(a) &&
               sumLoop.isDefinedOutsideOfLoop(b)) {
      ok = sameBound(a, b);
    } else {
      Operation *da = a.getDefiningOp(), *db = b.getDefiningOp();
      if (da && db && da->getBlock() == maxLoop.getBody() &&
          db->getBlock() == sumLoop.getBody() &&
          da->getName() == db->getName() &&
          da->getResultTypes() == db->getResultTypes() &&
          da->getNumOperands() == db->getNumOperands() &&
          da->getNumRegions() == 0 &&
          da->getAttrDictionary() == db->getAttrDictionary() &&
          cast<OpResult>(a).getResultNumber() ==
              cast<OpResult>(b).getResultNumber()) {
        ok = true;
        for (unsigned i = 0; i < da->getNumOperands() && ok; ++i)
          ok = sameVal(da->getOperand(i), db->getOperand(i));
      }
    }
    memo[{a, b}] = ok;
    return ok;
  };
  if (!sameVal(score1, score2) || (cond1 && !sameVal(cond1, cond2)))
    return std::nullopt;
  // Prove via CompilerInfra's value-replay proof that no memory effect in
  // maxLoop, between maxLoop and sumLoop, or in sumLoop can clobber or alias
  // the memory locations read by maxLoop.
  SmallVector<Value, 4> readViews;
  llvm::SmallDenseSet<Value, 4> seenViews;
  bool pureReads = true;
  maxLoop.walk([&](Operation *inst) {
    if (inst == maxLoop.getOperation())
      return WalkResult::advance();
    auto mem = dyn_cast<MemoryEffectOpInterface>(inst);
    if (!mem) {
      if (inst->getNumRegions() > 0 &&
          inst->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
        return WalkResult::advance();
      pureReads = false;
      return WalkResult::interrupt();
    }
    SmallVector<MemoryEffects::EffectInstance, 4> effects;
    mem.getEffects(effects);
    for (const auto &eff : effects) {
      if (isa<MemoryEffects::Write, MemoryEffects::Allocate,
              MemoryEffects::Free>(eff.getEffect())) {
        pureReads = false;
        return WalkResult::interrupt();
      }
      if (isa<MemoryEffects::Read>(eff.getEffect())) {
        Value v = eff.getValue();
        if (!v) {
          pureReads = false;
          return WalkResult::interrupt();
        }
        if (seenViews.insert(v).second)
          readViews.push_back(v);
      }
    }
    return WalkResult::advance();
  });
  if (!pureReads ||
      (!readViews.empty() &&
       !compilerinfra::proveValueReplay(maxLoop, sumLoop, readViews).proven))
    return std::nullopt;
  bool causalInclusive = false;
  Value causalUpper = causalMaskOwner(sumLoop, &causalInclusive);
  return OnlinePanelFusion{maxLoop,
                           score2,
                           cond2,
                           subOp,
                           mulOp,
                           scaleVal,
                           expOp,
                           guardOp,
                           wVal,
                           isa<math::Exp2Op>(expOp),
                           intentdsl::kir::getFastMathFlags(maxComb),
                           intentdsl::kir::getFastMathFlags(subOp),
                           scaleFlags,
                           intentdsl::kir::getFastMathFlags(expOp),
                           causalUpper,
                           causalInclusive};
}
SmallVector<Value> OwnerTiles::lowerBody(Block &body, Values env, Value mask) {
  for (auto it = body.begin(); it != body.end(); ++it) {
    Operation &op = *it;
    auto code = op.getName().stripDialect();
    auto arg = [&](unsigned i) { return at(op.getOperand(i), env); };
    Value v;
    if (auto y = dyn_cast<scf::YieldOp>(op)) {
      SmallVector<Value> values;
      for (auto a : y.getOperands())
        values.push_back(at(a, env));
      return values;
    }
    if (code == "end")
      break;
    if (isa<arith::ExtFOp>(op) && !op.use_empty() &&
        llvm::all_of(op.getUsers(), [](Operation *u) {
          auto fl = dyn_cast_or_null<scf::ForOp>(u->getParentOp());
          return fl && fl->hasAttr("block.contribution_width") &&
                 !u->use_empty() &&
                 llvm::all_of(u->getUsers(), [](Operation *s) {
                   return s->hasAttr("block.pair_dot");
                 });
        }))
      continue;
    if (auto c = dyn_cast<arith::ConstantOp>(op))
      v = constant(c);
    else if (auto d = dyn_cast<memref::DimOp>(op))
      v = integer(shape(d.getSource())[*d.getConstantIndex()]);
    else if (auto loop = dyn_cast<scf::ForOp>(op)) {
      if (op.hasAttr("block.contribution_width")) {
        SmallVector<Operation *, 4> midOps;
        auto scan = std::next(it);
        bool canScan = true;
        while (scan != body.end() && !isa<scf::ForOp>(*scan)) {
          auto mCode = scan->getName().stripDialect();
          if ((!isa<arith::ConstantOp, memref::DimOp>(*scan) &&
               mCode != "lane_splat") ||
              llvm::any_of(scan->getOperands(), [&](Value opnd) {
                return llvm::is_contained(loop.getResults(), opnd);
              })) {
            canScan = false;
            break;
          }
          midOps.push_back(&*scan);
          ++scan;
        }
        if (canScan && scan != body.end())
          if (auto sumLoop = dyn_cast<scf::ForOp>(*scan))
            if (auto online = matchOnlinePanelPair(loop, sumLoop)) {
              for (Operation *mid : midOps) {
                if (auto mc = dyn_cast<arith::ConstantOp>(*mid))
                  env[mid->getResult(0)] = constant(mc);
                else if (auto md = dyn_cast<memref::DimOp>(*mid))
                  env[mid->getResult(0)] =
                      integer(shape(md.getSource())[*md.getConstantIndex()]);
                else
                  env[mid->getResult(0)] = broadcast(
                      at(mid->getOperand(0), env), lifted(mid->getResult(0)));
              }
              auto fused = panelLoop(sumLoop, env, mask, &*online);
              env[loop.getResult(0)] = fused.front();
              for (auto [old, val] : llvm::zip(
                       sumLoop.getResults(), ArrayRef<Value>(fused).drop_front()))
                env[old] = val;
              it = scan;
              continue;
            }
      }
      auto values = op.hasAttr("block.contribution_width")
                        ? panelLoop(loop, env, mask)
                        : serialLoop(loop, env, mask);
      for (auto [old, val] : llvm::zip(op.getResults(), values))
        env[old] = val;
      continue;
    } else if (auto branch = dyn_cast<scf::IfOp>(op)) {
      Value rawCond = arg(0);
      if (!isa<ShapedType>(rawCond.getType())) {
        if (auto cst = rawCond.getDefiningOp<arith::ConstantOp>())
          if (auto ba = dyn_cast<BoolAttr>(cst.getValue())) {
            SmallVector<Value> chosen;
            if (ba.getValue())
              chosen = lowerBody(*branch.thenBlock(), env, mask);
            else if (!branch.getElseRegion().empty())
              chosen = lowerBody(*branch.elseBlock(), env, mask);
            for (auto [ret, val] : llvm::zip(op.getResults(), chosen))
              env[ret] = broadcast(val, lifted(ret));
            continue;
          }
      }
      auto hasEffects = [](Region &reg) {
        return reg
            .walk([](Operation *inst) {
              return (!isa<scf::YieldOp>(inst) && !isMemoryEffectFree(inst))
                         ? WalkResult::interrupt()
                         : WalkResult::advance();
            })
            .wasInterrupted();
      };
      auto cond = broadcast(rawCond, mask.getType());
      bool thenEff = hasEffects(branch.getThenRegion());
      bool elseEff = !branch.getElseRegion().empty() &&
                     hasEffects(branch.getElseRegion());
      Value pos = thenEff
                      ? Value(b.create<arith::AndIOp>(loc, mask, cond))
                      : mask;
      Value neg = elseEff
                      ? Value(b.create<arith::XOrIOp>(
                            loc, mask,
                            thenEff ? pos
                                    : Value(b.create<arith::AndIOp>(loc, mask,
                                                                    cond))))
                      : mask;
      auto yes = lowerBody(*branch.thenBlock(), env, pos);
      SmallVector<Value> no;
      if (!branch.getElseRegion().empty())
        no = lowerBody(*branch.elseBlock(), env, neg);
      for (auto [ret, t, f] : llvm::zip(op.getResults(), yes, no)) {
        auto type = lifted(ret);
        env[ret] = select(cond, broadcast(t, type), broadcast(f, type));
      }
      continue;
    } else if (code == "slice_read" || code == "slice_write") {
      bool w = code == "slice_write";
      SmallVector<Value> coords;
      for (Value coord : op.getOperands().drop_front(w ? 2 : 1))
        coords.push_back(widen(at(coord, env)));
      v = slice(op, coords, mask, w ? arg(0) : Value());
      if (w)
        continue;
    } else if (code == "lane_splat")
      v = broadcast(arg(0), lifted(op.getResult(0)));
    else if (code == "lane_sum") {
      Value value = arg(0);
      auto n =
          intentdsl::kir::memberExtent(fn, op.getOperand(0), "block.shape");
      auto mat = cast<RankedTensorType>(value.getType());
      if (n != mat.getDimSize(1)) {
        auto r = lanes(mat.getDimSize(1));
        auto valid = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, r,
                                             like(integer(n), r.getType()));
        value = select(broadcast(valid, mat.clone(b.getI1Type()), 1), value,
                       broadcast(scalar(0), mat));
      }
      v = reduce(value, "sum", 1);
    } else if (isa<memref::LoadOp, memref::StoreOp>(op)) {
      bool w = isa<memref::StoreOp>(op);
      auto base = arg(w ? 1 : 0);
      SmallVector<Value, 4> mapped;
      bool allScalar = true;
      for (auto c : op.getOperands().drop_front(w ? 2 : 1)) {
        Value mc = widen(at(c, env));
        if (isa<RankedTensorType>(mc.getType()))
          allScalar = false;
        mapped.push_back(mc);
      }
      auto ownersAttr = program->getAttrOfType<IntegerAttr>("block.owners");
      if (!w && allScalar && access.scalarBroadcastRead && mask == active &&
          ownersAttr && ownersAttr.getInt() > 0) {
        SmallVector<int64_t, 4> sAxes(mapped.size(), -1);
        Value sVal = read(base, address(base, mapped, sAxes));
        v = broadcast(
            sVal,
            shaped({rows}, cast<MemRefType>(base.getType()).getElementType()));
      } else {
        SmallVector<Value> coords;
        SmallVector<int64_t> axes;
        auto ix = shaped({rows}, b.getI64Type());
        for (auto mc : mapped) {
          coords.push_back(broadcast(mc, ix));
          axes.push_back(0);
        }
        auto ptr = address(base, coords, axes);
        if (w) {
          write(base, ptr,
                broadcast(arg(0), shaped({rows}, elem(arg(0).getType()))),
                mask);
          continue;
        }
        v = read(base, ptr, mask);
      }
    } else {
      SmallVector<Value> args;
      Type shapeHint;
      for (Value a : op.getOperands()) {
        auto av = at(a, env);
        args.push_back(av);
        if (auto tt = dyn_cast<RankedTensorType>(av.getType());
            tt && (!shapeHint ||
                   tt.getRank() > cast<RankedTensorType>(shapeHint).getRank()))
          shapeHint = tt;
      }
      auto e = elem(op.getResult(0).getType());
      if (e.isIndex())
        e = b.getI64Type();
      v = apply(op, args, shapeHint ? retype(shapeHint, e) : e);
    }
    env[op.getResult(0)] = v;
  }
  return {};
}

LogicalResult OwnerTiles::build() {
  Value blockId = pid(0);
  int64_t count = space.getAttrOfType<IntegerAttr>("block.owners").getInt();
  auto axes = space.getAttrOfType<DenseI64ArrayAttr>("axes");
  if (auto pair = space.getAttrOfType<IntegerAttr>("block.pair_axis")) {
    auto &bb = space.getRegion(0).front();
    auto pairArg = bb.getArgument(pair.getInt());
    bool relaxed = hasReassociativeNormalization(space);
    bool suffix = false, prefix = false;
    for (auto loop : bb.getOps<scf::ForOp>())
      if (loop->hasAttr("kir.reassociate")) {
        suffix |= loop.getLowerBound() == pairArg;
        prefix |= causalPrefixOwner(loop) == pairArg ||
                  (relaxed && causalMaskOwner(loop) == pairArg);
        if (auto branch = loop.getUpperBound().getDefiningOp<scf::IfOp>();
            branch && branch.getNumResults() == 1 &&
            !branch.getElseRegion().empty())
          if (auto cmp = branch.getCondition().getDefiningOp<arith::CmpIOp>();
              cmp && cmp.getPredicate() == arith::CmpIPredicate::slt &&
              cmp.getLhs() == pairArg)
            prefix = true;
      }
    int64_t pairExtent =
        std::max<int64_t>(1, shape(space.getOperand(pair.getInt()))[axes[pair.getInt()]]);
    if ((suffix != prefix) && pairExtent >= logical &&
        pairExtent % logical == 0 && count > pairExtent &&
        count % pairExtent == 0) {
      int64_t rowTiles = pairExtent / logical, outerGroups = count / pairExtent;
      Value outerIdx =
          b.create<arith::RemUIOp>(loc, blockId, integer(outerGroups));
      Value tileRank =
          b.create<arith::DivUIOp>(loc, blockId, integer(outerGroups));
      Value rowTile = prefix ? Value(b.create<arith::SubIOp>(
                                   loc, integer(rowTiles - 1), tileRank))
                             : tileRank;
      blockId = b.create<arith::AddIOp>(
          loc, b.create<arith::MulIOp>(loc, outerIdx, integer(rowTiles)),
          rowTile);
    }
  }
  Value start = b.create<arith::MulIOp>(loc, blockId, integer(logical));
  auto r = lanes(rows);
  auto lane = rows == logical
                  ? r
                  : Value(b.create<arith::RemUIOp>(
                        loc, r, like(integer(logical), r.getType())));
  auto owner =
      b.create<arith::AddIOp>(loc, broadcast(start, r.getType()), lane);
  active = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, owner,
                                   broadcast(integer(count), r.getType()));
  if (rows != logical)
    active = b.create<arith::AndIOp>(
        loc, active,
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, r,
                                like(integer(logical), r.getType())));
  auto group = space.getAttrOfType<IntegerAttr>("block.pair_axis");
  if (!group)
    group = space.getAttrOfType<IntegerAttr>("block.group_axis");
  SmallVector<int> order;
  if (group)
    order.push_back(group.getInt());
  for (int i = axes.size() - 1; i >= 0; --i)
    if (!group || i != group.getInt())
      order.push_back(i);
  int64_t pitch = 1;
  for (int i : order) {
    auto size = std::max<int64_t>(1, shape(space.getOperand(i))[axes[i]]);
    if (pitch == 1 && rows == logical && count > 0 && size % logical == 0) {
      int64_t rowTiles = size / logical;
      Value tileIdx =
          count == size
              ? blockId
              : Value(b.create<arith::RemUIOp>(loc, blockId, integer(rowTiles)));
      Value tileStart =
          b.create<arith::MulIOp>(loc, tileIdx, integer(logical));
      outer[space.getRegion(0).front().getArgument(i)] =
          b.create<arith::AddIOp>(loc, broadcast(tileStart, r.getType()), lane);
      pitch *= size;
      continue;
    }
    // The first owner is aligned to logical. Dividing its consecutive
    // lanes by a multiple of logical gives one shared outer coordinate.
    Value base = pitch % logical == 0 ? start : Value(owner);
    if (count > 0 && count <= INT32_MAX && pitch <= INT32_MAX &&
        size <= INT32_MAX) {
      Value base32 = castElement(base, b.getI32Type());
      auto c32 = [&](int64_t val) {
        return like(b.create<arith::ConstantIntOp>(loc, val, 32),
                    base32.getType());
      };
      Value coord32 =
          pitch == 1
              ? base32
              : Value(b.create<arith::DivUIOp>(loc, base32, c32(pitch)));
      coord32 = b.create<arith::RemUIOp>(loc, coord32, c32(size));
      outer[space.getRegion(0).front().getArgument(i)] =
          b.create<arith::ExtUIOp>(loc, retype(coord32.getType(), b.getI64Type()),
                                   coord32);
    } else {
      auto coord =
          pitch == 1
              ? base
              : Value(b.create<arith::DivUIOp>(loc, base,
                                               like(integer(pitch), base.getType())));
      outer[space.getRegion(0).front().getArgument(i)] =
          b.create<arith::RemUIOp>(loc, coord,
                                   like(integer(size), base.getType()));
    }
    pitch *= size;
  }
  lowerBody(space.getRegion(0).front(), outer, active);
  op("end");
  return success();
}
} // namespace intentdsl::block
