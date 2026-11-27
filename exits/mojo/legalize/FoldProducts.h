namespace {
LogicalResult lowerFoldProducts(OrderedLowerer &l, Block &body,
                                SmallVectorImpl<IRMapping> &lanes,
                                intentdsl::cpu::FoldProducts &p, int64_t T,
                                Operation *fn) {
  auto &b = l.b;
  auto loc = l.loc;
  int64_t B = l.batch, D = p.depth;
  Value zero = constIdx(b, loc, 0), one = constIdx(b, loc, 1),
        tile = constIdx(b, loc, T);
  auto sum = [&](Value a, Value v) {
    return b.create<arith::AddIOp>(loc, a, v).getResult();
  };
  auto times = [&](Value a, int64_t n) {
    return b.create<arith::MulIOp>(loc, a, constIdx(b, loc, n)).getResult();
  };
  auto range = [&](Value lb, Value ub, int64_t step, auto apply) {
    auto *op = makeOp(b, loc, "for_range", {lb, ub, constIdx(b, loc, step)}, {},
                      {}, 1);
    OpBuilder::InsertionGuard guard(b);
    auto &bb = op->getRegion(0).emplaceBlock();
    b.setInsertionPointToStart(&bb);
    auto result = apply(bb.addArgument(b.getIndexType(), loc));
    makeOp(b, loc, "end");
    return result;
  };
  auto buf = [&](StringRef dtype, int64_t n) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&fn->getRegion(0).front());
    return makeOp(b, loc, "stack_alloc", {}, {b.getIndexType()},
                  {b.getNamedAttr("dtype", b.getStringAttr(dtype)),
                   b.getNamedAttr("count", b.getI64IntegerAttr(n))})
        ->getResult(0);
  };
  auto splat = [&](Value x, int64_t w) {
    return makeOp(b, loc, "splat_scalar", {x},
                  {RankedTensorType::get({w}, x.getType())},
                  {b.getNamedAttr("dtype",
                                  b.getStringAttr(dtypeBare(x.getType()))),
                   b.getNamedAttr("width", b.getI64IntegerAttr(w))})
        ->getResult(0);
  };
  auto store = [&](Value ptr, Value off, Value v) {
    makeOp(b, loc, "store_slice", {ptr, off, v});
  };
  Value base = lanes.front().lookup(p.owner);
  auto address = [&](Operation *read, Value row, Value key) {
    Value view = read->getOperand(0);
    SmallVector<Value> indices;
    for (Value v : read->getOperands().drop_front())
      indices.push_back(v == p.owner ? row
                        : v == p.loop.getInductionVar()
                            ? key
                            : lanes.front().lookup(v));
    return makeAddr(b, loc, lanes.front().lookup(view), indices,
                    l.argArray(view, "cpu.strides")
                        .asArrayRef()
                        .take_front(indices.size()));
  };
  auto ptr = [&](Operation *read, Value row, Value key) {
    return makeOp(b, loc, "ptr_offset",
                  {lanes.front().lookup(read->getOperand(0)),
                   address(read, row, key)},
                  {b.getIndexType()})
        ->getResult(0);
  };
  auto gemm = [&](Value lhs, Value rhs, Value out, bool update) {
    auto i64 = [&](StringRef key, int64_t n) {
      return b.getNamedAttr(key, b.getI64IntegerAttr(n));
    };
    Value rc =
        makeOp(
            b, loc, "fold_gemm", {l.brgHandle, l.brgScratch, lhs, rhs, out},
            {b.getI32Type()},
            {i64("rows", B), i64("cols", update ? D : T),
             i64("depth", update ? T : D), i64("lda", update ? T : D),
             i64("ldb", D), i64("add_c", update),
             b.getNamedAttr(
                 "ci.implementation",
                 b.getStringAttr(p.nativeBf16 ? "exits/mojo::bf16-fold-gemm"
                                              : "exits/mojo::f16-fold-gemm"))})
            ->getResult(0);
    makeOp(b, loc, "check_status", {rc});
  };
  SmallVector<Value> accum, weights;
  llvm::MapVector<Value, Value> scores;
  for (auto *op : p.products)
    scores[op->getResult(0)] = buf("float32", B * T);
  auto zeroVec = splat(
      b.create<arith::ConstantFloatOp>(loc, APFloat(0.f), b.getF32Type()), 16);
  for (Value v : p.loop.getInitArgs()) {
    auto out = buf("float32", B * D);
    accum.push_back(out);
    weights.push_back(buf(p.nativeBf16 ? "bfloat16" : "float16", B * T));
    (void)range(zero, constIdx(b, loc, B * D), 16, [&](Value j) {
      store(out, j, zeroVec);
      return success();
    });
  }
  Value bound;
  // Bounds can vary by owner. Reduce them before issuing any rectangular read.
  for (int i = 0; i < B; ++i) {
    SmallVector<IRMapping, 1> m{lanes[0]};
    m[0].map(p.owner, addIdx(b, loc, base, i));
    if (p.monotoneBound && i != (p.prefix ? 0 : B - 1))
      continue;
    for (auto *op : p.boundOps)
      if (failed(l.lowerBlock(body, m, op, op->getNextNode())))
        return failure();
    auto v =
        m[0].lookup(p.prefix ? p.loop.getUpperBound() : p.loop.getLowerBound());
    bound = !bound     ? v
            : p.prefix ? b.create<arith::MinSIOp>(loc, bound, v).getResult()
                       : b.create<arith::MaxSIOp>(loc, bound, v).getResult();
  }
  if (!p.prefix)
    bound = addIdx(b, loc, bound, T - 1);
  bound = times(b.create<arith::DivSIOp>(loc, bound, tile), T);
  bound = b.create<arith::MaxSIOp>(loc, zero, bound);
  bound = b.create<arith::MinSIOp>(loc, constIdx(b, loc, p.steps), bound);
  if (failed(range(
          p.prefix ? zero : bound, p.prefix ? bound : constIdx(b, loc, p.steps),
          T, [&](Value t) {
            for (auto [dot, out] : scores) {
              auto *mul = dot.getDefiningOp()->getOperand(0).getDefiningOp();
              auto read = [&](unsigned i) {
                return mul->getOperand(i)
                    .getDefiningOp()
                    ->getOperand(0)
                    .getDefiningOp();
              };
              auto *q = read(0), *k = read(1);
              if (k->getOperands().back() == p.owner)
                std::swap(q, k);
              gemm(ptr(q, base, t), ptr(k, base, t), out, false);
            }
            (void)range(zero, constIdx(b, loc, B), 1, [&](Value i) {
              return range(zero, tile, 16, [&](Value j) {
                Value off = sum(times(i, T), j);
                llvm::DenseMap<Value, Value> values;
                auto weight = [&](auto &self, Value v) -> Value {
                  if (Value x = values.lookup(v))
                    return x;
                  Value x;
                  auto *op = v.getDefiningOp();
                  auto ty = RankedTensorType::get({16}, v.getType());
                  if (scores.contains(v))
                    x = l.loadContiguousSlice(scores.lookup(v), off,
                                              b.getF32Type(), 16);
                  else if (!op)
                    x = splat(lanes[0].lookup(v), 16);
                  else if (auto c = dyn_cast<arith::ConstantOp>(op))
                    x = splat(b.clone(*c)->getResult(0), 16);
                  else if (auto ld = dyn_cast<memref::LoadOp>(op)) {
                    Value addr = address(op, sum(base, i), sum(t, j)),
                          view = lanes[0].lookup(ld.getMemref());
                    if (ld.getIndices().back() == p.owner)
                      x = splat(makeOp(b, loc, "load_scalar", {view, addr},
                                       {b.getF32Type()})
                                    ->getResult(0),
                                16);
                    else
                      x = l.loadContiguousSlice(view, addr, b.getF32Type(), 16);
                  } else {
                    SmallVector<Value> operands;
                    for (Value v : op->getOperands())
                      operands.push_back(self(self, v));
                    OperationState state(loc, op->getName());
                    state.addOperands(operands);
                    state.addTypes(ty);
                    state.addAttributes(op->getAttrs());
                    x = b.create(state)->getResult(0);
                  }
                  values[v] = x;
                  return x;
                };
                for (auto [n, y] : llvm::enumerate(
                         p.loop.getBody()->getTerminator()->getOperands())) {
                  auto mul = y.getDefiningOp()->getOperand(1).getDefiningOp();
                  auto half = mul->getOperand(1)
                                  .getDefiningOp()
                                  ->getOperand(0)
                                  .getDefiningOp()
                                  ->getOperand(0);
                  store(weights[n], off, weight(weight, half));
                }
                return success();
              });
            });
            for (auto [n, y] : llvm::enumerate(
                     p.loop.getBody()->getTerminator()->getOperands())) {
              auto mul = y.getDefiningOp()->getOperand(1).getDefiningOp();
              auto *read = mul->getOperand(0)
                               .getDefiningOp()
                               ->getOperand(0)
                               .getDefiningOp();
              gemm(weights[n], ptr(read, base, t), accum[n], true);
            }
            return success();
          })))
    return failure();
  return range(zero, constIdx(b, loc, B), 1, [&](Value i) {
    SmallVector<IRMapping, 1> row{lanes[0]};
    row[0].map(p.owner, sum(base, i));
    if (failed(l.lowerBlock(body, row, nullptr, p.loop)))
      return failure();
    for (auto [n, init] : llvm::enumerate(p.loop.getInitArgs()))
      row[0].map(init, l.loadContiguousSlice(accum[n], times(i, D),
                                             b.getF32Type(), D));
    auto fringeBound =
        p.prefix ? p.loop.getLowerBound() : p.loop.getUpperBound();
    auto original = row[0].lookup(fringeBound);
    row[0].map(fringeBound, bound);
    if (failed(l.lowerBlock(body, row, p.loop, p.loop->getNextNode())))
      return failure();
    row[0].map(fringeBound, original);
    return l.lowerBlock(body, row, p.loop->getNextNode());
  });
}
} // namespace
