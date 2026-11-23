#include "OwnerTiles.h"
using namespace mlir;
namespace CI_UNIT_NAMESPACE {
namespace {
Type retype(Type t, Type elem) {
  if (auto shaped = dyn_cast<RankedTensorType>(t))
    return shaped.clone(elem);
  return elem;
}
} // namespace
OwnerTiles::OwnerTiles(func::FuncOp f, Operation &s)
    : Tiles(f, "owners"), space(s) {
  logical = s.getAttrOfType<IntegerAttr>("block.owner_tile").getInt();
  auto native = s.getAttrOfType<IntegerAttr>("triton_src.matrix_rows");
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
Value OwnerTiles::select(Value mask, Value yes, Value no) {
  return b.create<arith::SelectOp>(
      loc, broadcast(mask, retype(yes.getType(), b.getI1Type())), yes, no);
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
  auto lanesK = lanes(w);
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
SmallVector<Value> OwnerTiles::serialLoop(scf::ForOp loop, const Values &parent,
                                          Value mask) {
  auto lo = at(loop.getLowerBound(), parent),
       hi = at(loop.getUpperBound(), parent), step = at(loop.getStep(), parent);
  bool ragged = isa<RankedTensorType>(lo.getType()) ||
                isa<RankedTensorType>(hi.getType());
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
  l->setAttr("triton_src.runtime_range", b.getUnitAttr());
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
SmallVector<Value> OwnerTiles::lowerBody(Block &body, Values env, Value mask) {
  for (Operation &op : body) {
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
    if (auto c = dyn_cast<arith::ConstantOp>(op))
      v = constant(c);
    else if (auto d = dyn_cast<memref::DimOp>(op))
      v = integer(shape(d.getSource())[*d.getConstantIndex()]);
    else if (auto loop = dyn_cast<scf::ForOp>(op)) {
      auto values = op.hasAttr("block.contribution_width")
                        ? panelLoop(loop, env, mask)
                        : serialLoop(loop, env, mask);
      for (auto [old, val] : llvm::zip(op.getResults(), values))
        env[old] = val;
      continue;
    } else if (auto branch = dyn_cast<scf::IfOp>(op)) {
      auto cond = broadcast(arg(0), mask.getType());
      auto pos = b.create<arith::AndIOp>(loc, mask, cond);
      auto neg = b.create<arith::XOrIOp>(loc, mask, pos);
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
      SmallVector<Value> coords;
      SmallVector<int64_t> axes;
      auto ix = shaped({rows}, b.getI64Type());
      for (auto c : op.getOperands().drop_front(w ? 2 : 1)) {
        coords.push_back(broadcast(widen(at(c, env)), ix));
        axes.push_back(0);
      }
      auto ptr = address(base, coords, axes);
      if (w) {
        write(base, ptr,
              broadcast(arg(0), shaped({rows}, elem(arg(0).getType()))), mask);
        continue;
      }
      v = read(base, ptr, mask);
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
  Value start = b.create<arith::MulIOp>(loc, pid(0), integer(logical));
  auto r = lanes(rows);
  auto lane = rows == logical
                  ? r
                  : Value(b.create<arith::RemUIOp>(
                        loc, r, like(integer(logical), r.getType())));
  auto owner =
      b.create<arith::AddIOp>(loc, broadcast(start, r.getType()), lane);
  int64_t count = space.getAttrOfType<IntegerAttr>("block.owners").getInt();
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
  auto axes = space.getAttrOfType<DenseI64ArrayAttr>("axes");
  SmallVector<int> order;
  if (group)
    order.push_back(group.getInt());
  for (int i = axes.size() - 1; i >= 0; --i)
    if (!group || i != group.getInt())
      order.push_back(i);
  int64_t pitch = 1;
  for (int i : order) {
    auto size = std::max<int64_t>(1, shape(space.getOperand(i))[axes[i]]);
    Value base = group && pitch % logical == 0 ? start : Value(owner);
    auto coord = b.create<arith::DivUIOp>(loc, base,
                                          like(integer(pitch), base.getType()));
    outer[space.getRegion(0).front().getArgument(i)] = b.create<arith::RemUIOp>(
        loc, coord, like(integer(size), base.getType()));
    pitch *= size;
  }
  lowerBody(space.getRegion(0).front(), outer, active);
  op("end");
  return success();
}
} // namespace CI_UNIT_NAMESPACE
