// Private lowering over the ordered task builder. A tile covers only the
// intersection of the owners' iteration ranges; each fringe retains its loop.
namespace {
struct ProductPanel {
  OrderedLowerer &l;
  intentdsl::cpu::FoldProducts &p;
  int64_t tokens;
  TaskTerm t;
  int64_t axes[3] = {};
  enum { Owner = 1, Step = 2, Depth = 4 };
  struct Tile {
    TaskTerm term;
    unsigned axes;
  };
  llvm::DenseMap<Value, Tile> cache;

  std::string type(StringRef elem, unsigned mask) {
    if (!mask)
      return elem.str();
    SmallVector<std::string> shape, ids;
    int64_t sizes[] = {l.batch, tokens, p.depth};
    for (unsigned i = 0; i < 3; ++i)
      if (mask & (1u << i)) {
        shape.push_back(std::to_string(sizes[i]));
        ids.push_back(std::to_string(axes[i]));
      }
    return "!weft_kernel.value<" + elem.str() + ",[" + llvm::join(shape, ",") +
           "],[" + llvm::join(ids, ",") + "]>";
  }

  Tile read(Operation *op, bool vector) {
    Value view = op->getOperand(0);
    auto coords = op->getOperands().drop_front();
    bool own = coords.back() == p.owner;
    unsigned axis = own ? Owner : Step;
    auto offset = own ? l.lanes[0].at(p.owner) : t;
    int64_t count = own ? l.batch : tokens;
    auto mem = cast<MemRefType>(view.getType());
    auto elem = l.scalarType(mem.getElementType());
    if (vector) {
      auto base = l.slice2D(view, coords.drop_back(), mem.getRank() - 2,
                            mem.getRank() - 1);
      auto val = l.subview2D(base, offset.v, l.idx(0).v, count, p.depth, true,
                             type(elem, axis | Depth));
      return {val, axis | Depth};
    }
    auto base =
        l.sliceView(0, view, coords.drop_back(), mem.getRank() - 1, false);
    auto &b = l.b;
    auto result = type(elem, axis);
    auto *sub =
        makeOp(b, l.loc, "task_subview", {base.v, offset.v}, {b.getIndexType()},
               {b.getNamedAttr("static_extent", b.getI64IntegerAttr(count)),
                b.getNamedAttr("slice_type", b.getStringAttr(base.type)),
                b.getNamedAttr("val_type", b.getStringAttr(result)),
                b.getNamedAttr("admit", b.getBoolAttr(true))});
    return {{sub->getResult(0), result}, axis};
  }
  Tile evaluate(Value v) {
    auto it = cache.find(v);
    if (it != cache.end())
      return it->second;
    auto val = emit(v);
    cache[v] = val;
    return val;
  }
  Tile emit(Value v) {
    auto *op = v.getDefiningOp();
    if (!op)
      return {l.lanes[0].at(v), 0};
    auto kind = op->getName().stripDialect();
    if (kind == "slice_read" || isa<memref::LoadOp>(op))
      return read(op, kind == "slice_read");
    if (auto c = dyn_cast<arith::ConstantOp>(op))
      return {l.lit(stripTypeSuffix(c.getValue()), l.scalarType(v.getType())),
              0};
    if (kind == "lane_sum") {
      auto *product = op->getOperand(0).getDefiningOp();
      auto a = evaluate(product->getOperand(0).getDefiningOp()->getOperand(0)),
           b = evaluate(product->getOperand(1).getDefiningOp()->getOperand(0));
      if (b.axes & Owner)
        std::swap(a, b);
      return {l.outerContractOp(
                  a.term, b.term, axes[2], type("f32", Owner | Step),
                  l.b.getStringAttr(p.nativeBf16
                                        ? "exits/weft::onednn-brgemm-bf16"
                                        : "exits/weft::onednn-brgemm-f16")),
              Owner | Step};
    }
    auto a = evaluate(op->getOperand(0));
    if (isa<arith::ExtFOp, arith::TruncFOp>(op))
      return {
          l.castOp(a.term, type(l.scalarType(getElementTypeOrSelf(v.getType())),
                                a.axes)),
          a.axes};
    if (isa<math::Exp2Op>(op))
      return {l.unaryOp("exp2", a.term), a.axes};
    auto b = evaluate(op->getOperand(1));
    auto axes = a.axes | b.axes;
    auto element = l.scalarType(v.getType());
    auto result = type(element, axes);
    // Weft broadcasts in left-then-right axis order. Expand a step-only
    // left operand in owner order without arithmetic or changing its bits.
    if (a.axes == Step && (b.axes & Owner))
      a.term = l.sel(l.lit("false", "i1"), b.term, a.term, result);
    return {l.bin(isa<arith::MulFOp>(op)   ? "mul"
                  : isa<arith::SubFOp>(op) ? "sub"
                                           : "add",
                  a.term, b.term, result),
            axes};
  }
};

LogicalResult lowerFoldProducts(OrderedLowerer &l, Block &body, int64_t tokens,
                                const intentdsl::CallFacts &facts) {
  auto plan = intentdsl::cpu::foldProducts(l.space, l.fn, facts);
  if (!plan)
    return l.space.emitError("cannot realize the selected fold panel");
  auto &p = *plan;
  auto &b = l.b;
  auto before = [&](int lane) {
    for (auto &op : body) {
      if (&op == p.loop.getOperation())
        break;
      if (failed(l.lowerOp(op, 1, lane)))
        return failure();
    }
    return success();
  };
  auto owner0 = l.lanes[0].at(p.owner);
  auto zero = l.idx(0), step = l.idx(tokens);
  // Evaluate each scalar bound. This also handles clamped and empty rows.
  TaskTerm common;
  for (int i = 0; i < l.batch; ++i) {
    if (p.monotoneBound && i != (p.prefix ? 0 : l.batch - 1))
      continue;
    for (auto *op : p.boundOps)
      if (failed(l.lowerOp(*op, 1, i)))
        return failure();
    auto bound = l.lanes[i].at(p.prefix ? p.loop.getUpperBound()
                                        : p.loop.getLowerBound());
    if (!common.v)
      common = bound;
    else
      common = l.sel(l.cmp(p.prefix ? "lt" : "gt", bound, common, "i1"), bound,
                     common, "index");
  }
  auto bound = common;
  if (!p.prefix)
    bound = l.bin("add", bound, l.idx(tokens - 1), "index");
  bound = l.bin("mul", l.bin("div", bound, step, "index"), step, "index");
  bound = l.sel(l.cmp("lt", bound, zero, "i1"), zero, bound, "index");
  auto extent = l.idx(p.steps);
  bound = l.sel(l.cmp("gt", bound, extent, "i1"), extent, bound, "index");
  auto low = p.prefix ? zero : bound, high = p.prefix ? bound : extent;
  ProductPanel panel{l, p, tokens, {}};
  auto setAxes = [&](Operation *op) {
    if (op->getName().stripDialect() != "slice_read")
      return;
    auto view = op->getOperand(0);
    auto a = l.axesMap.at(view);
    auto last = op->getOperands().back();
    if (last == p.owner || last == p.loop.getInductionVar()) {
      panel.axes[last == p.owner ? 0 : 1] = a[a.size() - 2];
      panel.axes[2] = a.back();
    }
  };
  for (Value operand :
       p.products.front()->getOperand(0).getDefiningOp()->getOperands())
    setAxes(operand.getDefiningOp()->getOperand(0).getDefiningOp());
  auto accTy = panel.type("f32", ProductPanel::Owner | ProductPanel::Depth);
  auto init = l.newVec(l.lit("0.0", "f32"), accTy);
  SmallVector<Value> args{low.v, high.v, step.v};
  SmallVector<Attribute> tys;
  SmallVector<Type> results;
  for (Value v : p.loop.getInitArgs()) {
    args.push_back(init.v);
    tys.push_back(b.getStringAttr(accTy));
    results.push_back(b.getIndexType());
  }
  auto types = b.getArrayAttr(tys);
  auto *fold = makeOp(b, l.loc, "task_for", args, results,
                      {b.getNamedAttr("iter_types", types)}, 1);
  {
    OpBuilder::InsertionGuard g(b);
    auto &bb = fold->getRegion(0).emplaceBlock();
    b.setInsertionPointToStart(&bb);
    panel.t = {bb.addArgument(b.getIndexType(), l.loc), "index"};
    SmallVector<Value> values;
    for (Value term : p.loop.getBody()->getTerminator()->getOperands()) {
      auto add = term.getDefiningOp<arith::AddFOp>();
      auto mul = add.getRhs().getDefiningOp<arith::MulFOp>();
      TaskTerm old{bb.addArgument(b.getIndexType(), l.loc), accTy};
      auto factor = mul.getRhs().getDefiningOp()->getOperand(0);
      auto factors = panel.evaluate(factor.getDefiningOp()->getOperand(0));
      auto data = panel.evaluate(mul.getLhs().getDefiningOp()->getOperand(0));
      auto prod = l.outerContractOp(
          factors.term, data.term, panel.axes[1], accTy,
          b.getStringAttr(p.nativeBf16 ? "exits/weft::onednn-brgemm-bf16"
                                       : "exits/weft::onednn-brgemm-f16"));
      auto next = l.bin("add", old, prod, accTy);
      next.v.getDefiningOp()->setAttr("fastmath", add->getAttr("fastmath"));
      values.push_back(next.v);
    }
    makeOp(b, l.loc, "task_yield", values, {},
           {b.getNamedAttr("val_types", types)});
  }
  auto *fringe =
      makeOp(b, l.loc, "task_for", {zero.v, l.idx(l.batch).v, l.idx(1).v}, {},
             {b.getNamedAttr("iter_types", b.getArrayAttr({}))}, 1);
  auto &fb = fringe->getRegion(0).emplaceBlock();
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointToStart(&fb);
  TaskTerm lane{fb.addArgument(b.getIndexType(), l.loc), "index"};
  l.lanes[0][p.owner] = l.bin("add", owner0, lane, "index");
  if (failed(before(0)))
    return failure();
  intentdsl::cpu::FoldRemainder fringeArgs{
      p.prefix ? bound.v : l.lanes[0].at(p.loop.getLowerBound()).v,
      p.prefix ? l.lanes[0].at(p.loop.getUpperBound()).v : bound.v,
      {}};
  for (auto [i, initial] : llvm::enumerate(p.loop.getInitArgs())) {
    auto ty = l.vectorType(initial);
    auto *row = makeOp(b, l.loc, "task_extract", {fold->getResult(i), lane.v},
                       {b.getIndexType()},
                       {b.getNamedAttr("selector", b.getStringAttr("index")),
                        b.getNamedAttr("keep_columns", b.getUnitAttr()),
                        b.getNamedAttr("in_type", b.getStringAttr(accTy)),
                        b.getNamedAttr("point_type", b.getStringAttr("index")),
                        b.getNamedAttr("val_type", b.getStringAttr(ty))});
    fringeArgs.initial.push_back(row->getResult(0));
  }
  if (failed(l.lowerOp(*p.loop, 1, 0, &fringeArgs)))
    return failure();
  for (Operation *op = p.loop->getNextNode(); op; op = op->getNextNode())
    if (failed(l.lowerOp(*op, 1, 0)))
      return failure();
  makeOp(b, l.loc, "task_yield", {}, {},
         {b.getNamedAttr("val_types", b.getArrayAttr({}))});
  return success();
}
} // namespace
