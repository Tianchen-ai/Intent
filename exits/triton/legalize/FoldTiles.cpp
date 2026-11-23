#include "OwnerTiles.h"
using namespace mlir;
namespace CI_UNIT_NAMESPACE {
namespace {
enum class Axis { Scalar, Owner, Token, Both };
struct Fragment {
  Value value;
  Axis axis;
};
} // namespace
SmallVector<Value> OwnerTiles::panelLoop(scf::ForOp loop, const Values &parent,
                                         Value mask) {
  int64_t span =
      loop->getAttrOfType<IntegerAttr>("block.contribution_width").getInt();
  bool paired = space.hasAttr("block.pair_axis");
  auto asScalar = [&](Value v, StringRef algebra) -> Value {
    auto fragment = dyn_cast<RankedTensorType>(v.getType());
    if (!fragment)
      return v;
    if (rows != logical && fragment.getRank() == 1)
      v = op("tile_prefix", v, shaped({logical}, elem(v.getType())))
              ->getResult(0);
    return reduce(v, algebra, 0);
  };
  Value lower = at(loop.getLowerBound(), parent),
        upper = at(loop.getUpperBound(), parent);
  auto tileType = [&](Type e) { return shaped({rows, span}, e); };
  auto plane = [&](Fragment x) -> Value {
    return broadcast(x.value, tileType(elem(x.value.getType())),
                     x.axis == Axis::Token ? 1 : 0);
  };
  auto uniform = [&](Value v) { return asScalar(v, "min"); };
  Value width = integer(span), lo = uniform(lower), hi = asScalar(upper, "max");
  Value start = b.create<arith::MulIOp>(
      loc, b.create<arith::FloorDivSIOp>(loc, lo, width), width);
  SmallVector<Value> init;
  SmallVector<Type> types;
  for (auto v : loop.getInitArgs()) {
    auto t = lifted(v);
    init.push_back(broadcast(at(v, parent), t));
    types.push_back(t);
  }
  auto l = b.create<scf::ForOp>(loc, start, hi, width, init);
  l->setAttr("triton_src.runtime_range", b.getUnitAttr());
  b.setInsertionPointToStart(l.getBody());
  Value token = stepped(l.getInductionVar(), span);
  auto tokens = Fragment{token, Axis::Token};
  auto tokenType = token.getType();
  Value live, any;
  if (loop->hasAttr("block.uniform_tokens") || !paired) {
    any = b.create<arith::AndIOp>(
        loc,
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, token,
                                like(lo, tokenType)),
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, token,
                                like(hi, tokenType)));
    live = plane({any, Axis::Token});
  } else {
    auto t = plane(tokens);
    live = b.create<arith::AndIOp>(
        loc,
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, t,
                                plane({lower, Axis::Owner})),
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, t,
                                plane({upper, Axis::Owner})));
    any = reduce(live, "max", 0);
  }
  llvm::DenseMap<Value, Fragment> local;
  local[loop.getInductionVar()] = tokens;
  auto value = [&](Value v) {
    if (auto it = local.find(v); it != local.end())
      return it->second;
    auto x = at(v, parent);
    return Fragment{x, isa<RankedTensorType>(x.getType()) ? Axis::Owner
                                                          : Axis::Scalar};
  };
  auto coordinate = [&](Value v) {
    auto x = value(v);
    return x.axis == Axis::Owner ? uniform(x.value) : x.value;
  };
  llvm::SmallPtrSet<Operation *, 16> folded;
  auto yielded = loop.getBody()->getTerminator()->getOperands();
  for (Value r : yielded) {
    auto *comb = r.getDefiningOp();
    folded.insert(comb);
    if (comb->hasAttr("block.masked_dot_update")) {
      auto *product = comb->getOperand(1).getDefiningOp();
      folded.insert(product);
      for (Value input : product->getOperands())
        if (auto *op = input.getDefiningOp();
            op && op->getName().stripDialect() == "lane_splat" &&
            input.hasOneUse())
          folded.insert(op);
    }
  }
  for (auto &o : *loop.getBody())
    if (o.hasAttr("block.pair_dot"))
      folded.insert(o.getOperand(0).getDefiningOp());
  auto singleMember = [&](Fragment x) -> Value {
    if (x.axis != Axis::Owner)
      return x.value;
    if (auto s = dyn_cast<RankedTensorType>(x.value.getType());
        s && s.getRank() == 2)
      return op("drop_unit_owner", x.value,
                shaped({s.getDimSize(1)}, s.getElementType()))
          ->getResult(0);
    return uniform(x.value);
  };
  for (auto &old : loop.getBody()->without_terminator()) {
    if (folded.contains(&old))
      continue;
    auto code = old.getName().stripDialect();
    auto out = old.getResult(0);
    auto operand = [&](int i) { return value(old.getOperand(i)); };
    Fragment result;
    if (auto c = dyn_cast<arith::ConstantOp>(old))
      result = {constant(c), Axis::Scalar};
    else if (code == "slice_read" || isa<memref::LoadOp>(old)) {
      auto view = old.getOperand(0);
      SmallVector<Value> coords;
      SmallVector<int64_t> axes;
      for (auto v : old.getOperands().drop_front()) {
        auto ix = widen(coordinate(v));
        coords.push_back(broadcast(ix, tokenType));
        axes.push_back(0);
      }
      Value accessMask = any;
      if (code == "slice_read") {
        auto ax = old.getAttrOfType<IntegerAttr>("axis").getInt(),
             n = shape(view)[ax];
        int64_t padded = std::max<int64_t>(1, llvm::PowerOf2Ceil(n));
        Value k = lanes(padded);
        coords.insert(coords.begin() + ax, k);
        axes.insert(axes.begin() + ax, 1);
        auto m = shaped({span, padded}, b.getI1Type());
        accessMask = broadcast(any, m, 0);
        if (padded != n) {
          auto within = b.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::slt, k, like(integer(n), k.getType()));
          accessMask =
              b.create<arith::AndIOp>(loc, accessMask, broadcast(within, m, 1));
        }
      }
      result = {read(view, address(view, coords, axes), accessMask),
                Axis::Token};
    } else if (old.hasAttr("block.pair_dot")) {
      auto operands = *intentdsl::block::tileDot(&old, loop);
      result = {
          dot(value(operands.rows).value, value(operands.tokens).value, true),
          Axis::Both};
    } else if (code == "lane_splat") {
      auto x = operand(0);
      int64_t n =
          std::max<int64_t>(1, llvm::PowerOf2Ceil(intentdsl::kir::memberExtent(
                                   fn, out, "block.shape")));
      result = {
          broadcast(singleMember(x), shaped({span, n}, elem(out.getType())), 0),
          Axis::Token};
    } else if (code == "lane_sum")
      result = {reduce(operand(0).value, "sum", 1), Axis::Token};
    else {
      SmallVector<Fragment> inputs;
      for (Value v : old.getOperands())
        inputs.push_back(value(v));
      Axis role = inputs.front().axis;
      if (inputs.size() == 2 && inputs[1].axis != Axis::Scalar &&
          inputs[1].axis != role)
        role = role == Axis::Scalar ? inputs[1].axis : Axis::Both;
      SmallVector<Value> args;
      Type dst;
      for (auto x : inputs) {
        auto arg = x.value;
        if (!paired) {
          arg = singleMember(x);
          role = Axis::Token;
          // A single owner's member vector is broadcast across token rows.
          if (isa<RankedTensorType>(old.getResult(0).getType())) {
            auto n = std::max<int64_t>(
                1, llvm::PowerOf2Ceil(
                       intentdsl::kir::memberExtent(fn, out, "block.shape")));
            arg = broadcast(arg, shaped({span, n}, elem(arg.getType())),
                            x.axis == Axis::Token ? 0 : 1);
          }
        } else if (out.getType().isIndex()) {
          if (x.axis == Axis::Owner)
            arg = uniform(arg);
          role = llvm::any_of(inputs,
                              [](Fragment f) { return f.axis == Axis::Token; })
                     ? Axis::Token
                     : Axis::Scalar;
        } else if (role == Axis::Both)
          arg = plane(x);
        args.push_back(arg);
        if (!dst || isa<RankedTensorType>(arg.getType()))
          dst = arg.getType();
      }
      auto ty = elem(out.getType());
      if (ty.isIndex())
        ty = b.getI64Type();
      if (auto s = dyn_cast<RankedTensorType>(dst))
        ty = s.clone(ty);
      result = {apply(old, args, ty), role};
    }
    local[out] = result;
  }
  SmallVector<Value> next;
  for (auto [ret, acc] : llvm::zip(yielded, l.getRegionIterArgs())) {
    auto *comb = ret.getDefiningOp();
    auto rhs = comb->getOperand(1);
    if (paired && comb->hasAttr("block.masked_dot_update")) {
      auto operands = *intentdsl::block::tileAccumulate(rhs, loop);
      auto w = plane(value(operands.weights));
      auto v = value(operands.values).value;
      auto zero = castElement(scalar(0), elem(w.getType()));
      auto guarded = select(live, w, broadcast(zero, w.getType()));
      next.push_back(
          dot(castElement(guarded, elem(v.getType())), v, false, acc));
    } else {
      bool max = isa<arith::MaxNumFOp>(comb);
      auto x = paired ? plane(value(rhs)) : singleMember(value(rhs));
      Value keep = paired ? live : any;
      if (!isa<RankedTensorType>(x.getType()))
        x = broadcast(x, shaped({span}, elem(x.getType())));
      Value term = select(
          keep, x, broadcast(scalar(max ? -INFINITY : 0.f), x.getType()));
      auto partial = reduce(term, max ? "max" : "sum", paired ? 1 : 0);
      if (!paired && isa<ShapedType>(rhs.getType()))
        partial = broadcast(partial, acc.getType(), 1);
      else
        partial = broadcast(partial, acc.getType());
      next.push_back(max ? Value(b.create<arith::MaxNumFOp>(loc, acc, partial))
                         : Value(b.create<arith::AddFOp>(loc, acc, partial)));
    }
  }
  b.create<scf::YieldOp>(loc, next);
  b.setInsertionPointAfter(l);
  if (!loop->hasAttr("block.repair_masked_nan"))
    return SmallVector<Value>(l.getResults());
  Value bad;
  for (auto [i, v] : llvm::enumerate(l.getResults()))
    if (cast<RankedTensorType>(v.getType()).getRank() == 2) {
      auto nan = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNE, v, v);
      auto anyNan = reduce(reduce(nan, "max", 1), "max", 0);
      bad = bad ? Value(b.create<arith::OrIOp>(loc, bad, anyNan)) : anyNan;
    }
  auto branch = b.create<scf::IfOp>(loc, types, bad, true);
  b.setInsertionPointToStart(branch.thenBlock());
  b.create<scf::YieldOp>(loc, serialLoop(loop, parent, mask));
  b.setInsertionPointToStart(branch.elseBlock());
  b.create<scf::YieldOp>(loc, l.getResults());
  b.setInsertionPointAfter(branch);
  return SmallVector<Value>(branch.getResults());
}
} // namespace CI_UNIT_NAMESPACE
