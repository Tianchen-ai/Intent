// Logical roles for a panel: query owners (rows), contributions (tokens),
// and independent vector members. A dot contracts only the member axis.
llvm::Error emitPairFold(mlir::scf::ForOp loop, unsigned depth, int64_t span) {
  using namespace mlir;
  auto lib = language().str(), sz = std::to_string(span);
  auto ex = [&](std::string v, int axis) {
    return lib + ".expand_dims(" + v + "," + std::to_string(axis) + ")";
  };
  auto uniform = [&](std::string v) {
    return lib + ".min(" + broadcast(v) + ",axis=0)";
  };
  auto token = [&](std::string v) {
    return lib + ".broadcast_to(" + v + ",(" + sz + ",))";
  };
  auto extent = [&](Value v) {
    return kir::memberExtent(fn, v, "block.shape");
  };
  auto lo = name(loop.getLowerBound()), hi = name(loop.getUpperBound()),
       base = fresh(), iv = fresh(), live = fresh(), any = fresh();
  auto &bb = *loop.getBody();
  auto yields = bb.getTerminator()->getOperands();
  llvm::SmallPtrSet<Operation *, 16> deferred;
  SmallVector<std::string> accum;
  for (auto [init, ret, next] :
       llvm::zip(loop.getInitArgs(), loop.getResults(), yields)) {
    auto id = fresh();
    accum.push_back(id);
    names[ret] = id;
    line(depth, id + "=" +
                    (isa<ShapedType>(ret.getType()) ? name(init)
                                                    : broadcast(name(init))));
    deferred.insert(next.getDefiningOp());
    if (next.getDefiningOp()->hasAttr("block.masked_dot_update")) {
      auto *mul = next.getDefiningOp()->getOperand(1).getDefiningOp();
      deferred.insert(mul);
    }
  }
  for (auto &op : bb)
    if (op.hasAttr("block.pair_dot"))
      deferred.insert(op.getOperand(0).getDefiningOp());
  line(depth, base + "=(" +
                  cast(uniform(lo), IndexType::get(loop.getContext())) + "//" +
                  sz + ")*" + sz);
  line(depth,
       "while " + base + "<" + lib + ".max(" + broadcast(hi) + ",axis=0):");
  line(depth + 1, iv + "=" + base + "+" + memberRange(span));
  if(loop->hasAttr("block.uniform_tokens")){
    line(depth+1,any+"=("+iv+">="+uniform(lo)+")&("+iv+"<"+uniform(hi)+")");
    line(depth+1,live+"="+lib+".broadcast_to("+ex(any,0)+",(ROWS,"+sz+"))");
  }else{
  line(depth + 1, live + "=(" + ex(iv, 0) + ">=" + ex(broadcast(lo), 1) +
                      ")&(" + ex(iv, 0) + "<" + ex(broadcast(hi), 1) + ")");
  line(depth + 1, any + "=" + lib + ".max(" + live + ",axis=0)!=0");
  }
  enum Role { owner, column, pair, scalar };
  struct Tile {
    std::string name;
    Role role;
  };
  llvm::DenseMap<Value, Tile> local;
  local[loop.getInductionVar()] = {iv, column};
  auto value = [&](Value v) {
    auto it = local.find(v);
    if (it != local.end())
      return it->second;
    return Tile{name(v), owner};
  };
  auto matrix = [&](Tile x) {
    if (x.role == owner)
      return ex(broadcast(x.name), 1);
    if (x.role == column)
      return ex(token(x.name), 0);
    return x.name;
  };
  auto coordinate = [&](Value v) {
    auto x = value(v);
    return x.role == owner ? uniform(x.name) : x.name;
  };
  auto asMatrix = [&](std::string v) {
    return lib + ".broadcast_to(" + v + ",(ROWS," + sz + "))";
  };
  for (auto &op : bb.without_terminator()) {
    if (deferred.contains(&op))
      continue;
    auto out = op.getResult(0);
    auto kind = op.getName().stripDialect();
    Tile result{"", scalar};
    auto operand = [&](unsigned i) { return value(op.getOperand(i)); };
    if (kind == "slice_read") {
      SmallVector<std::string> coords;
      for (auto c : op.getOperands().drop_front())
        coords.push_back(ex(token(coordinate(c)), 1));
      auto axis = op.getAttrOfType<IntegerAttr>("axis").getInt();
      auto n = extent(out);
      coords.insert(coords.begin() + axis, ex(memberRange(n), 0));
      std::string read;
      if (op.hasAttr("block.rectangular_tokens")) {
        SmallVector<std::string> outer;
        for (auto c : op.getOperands().drop_front().drop_back())
          outer.push_back(coordinate(c));
        read = loadMemberPanel(op, outer, base, span, n, ex(any, 1));
      }
      result = {read.empty() ? memberAccess(op, coords, ex(any, 1), false)
                             : read,
                column};
    } else if (auto ld = dyn_cast<memref::LoadOp>(op)) {
      SmallVector<std::string> at;
      for (auto c : ld.getIndices())
        at.push_back(coordinate(c));
      result = {gatherScalars(ld, at, any), column};
    } else if (kind == "lane_splat") {
      // These broadcasts feed a materialized mixed matrix contraction.
      if (!llvm::all_of(out.getUsers(),
                        [&](Operation *u) { return deferred.contains(u); }))
        return llvm::createStringError(
            "unissued member broadcast inside matrix fold");
      continue;
    } else if (kind == "lane_sum") {
      auto dot = tileDot(&op, loop);
      if (!dot || !op.hasAttr("block.pair_dot"))
        return llvm::createStringError("missing matrix issue for member dot");
      result = {matrixProduct(value(dot->rows).name, value(dot->tokens).name,
                              true, sz),
                pair};
    } else if (auto c = dyn_cast<arith::ConstantOp>(op)) {
      if (auto n = dyn_cast<IntegerAttr>(c.getValue()))
        result.name = std::to_string(n.getInt());
      else {
        llvm::SmallString<24> s;
        mlir::cast<FloatAttr>(c.getValue()).getValue().toString(s);
        result.name = s.str().str();
      }
    } else if (kind == "index_cast" || kind == "extf" || kind == "truncf") {
      Type t = out.getType();
      if (auto sh = dyn_cast<ShapedType>(t))
        t = sh.getElementType();
      auto a = operand(0);
      result = {cast(a.name, t), a.role};
    } else if (kind == "exp2" || kind == "exp") {
      auto a = operand(0);
      result = {lib + "." + kind.str() + "(" + a.name + ")", a.role};
    } else {
      auto symbol = llvm::StringSwitch<StringRef>(kind)
                        .Cases("addi", "addf", "+")
                        .Cases("subi", "subf", "-")
                        .Cases("muli", "mulf", "*")
                        .Case("floordivsi", "//")
                        .Case("remsi", "%")
                        .Default("");
      if (symbol.empty())
        return llvm::createStringError("unexpected matrix fold operation " +
                                       kind);
      auto a = operand(0), b = operand(1);
      bool index = out.getType().isIndex();
      if (index) {
        result = {coordinate(op.getOperand(0)) + symbol.str() +
                      coordinate(op.getOperand(1)),
                  a.role == column || b.role == column ? column : scalar};
      } else if (a.role == pair || b.role == pair ||
                 (a.role != scalar && b.role != scalar && a.role != b.role)) {
        result = {matrix(a) + symbol.str() + matrix(b), pair};
      } else
        result = {a.name + symbol.str() + b.name,
                  a.role == scalar ? b.role : a.role};
    }
    auto id = fresh();
    line(depth + 1, id + "=" + result.name);
    result.name = id;
    local[out] = result;
  }
  for (auto [acc, v] : llvm::zip(accum, yields)) {
    auto *combine = v.getDefiningOp();
    auto rhs = combine->getOperand(1);
    if (auto update = tileAccumulate(rhs, loop)) {
      if (!combine->hasAttr("block.masked_dot_update"))
        return llvm::createStringError("missing masked matrix update");
      auto w = fresh(), vec = value(update->values).name;
      line(depth + 1, w + "=" + lib + ".where(" + live + "," +
                          asMatrix(matrix(value(update->weights))) + ",0.)");
      auto t =
          mlir::cast<ShapedType>(update->values.getType()).getElementType();
      auto fast = acc + "=" + matrixAccumulate(cast(w, t), vec, acc);
      line(depth + 1, fast);
    } else {
      auto term = asMatrix(matrix(value(rhs)));
      bool maximum = isa<arith::MaxNumFOp>(combine);
      auto x = lib + ".where(" + live + "," + term + "," +
               (maximum ? "-float('inf')" : "0.") + ")";
      auto sum = lib + (maximum ? ".max(" : ".sum(") + x + ",axis=1)";
      line(depth + 1, acc + "=" +
                          (maximum ? lib + ".maximum(" + acc + "," + sum + ")"
                                   : acc + "+" + sum));
    }
  }
  line(depth + 1, base + "=" + base + "+" + sz);
  if (loop->hasAttr("block.repair_masked_nan")) {
    SmallVector<std::string> bad;
    for (auto [i, ret] : llvm::enumerate(loop.getResults()))
      if (isa<ShapedType>(ret.getType())) {
        auto x = accum[i];
        bad.push_back("(" + lib + ".max(" + lib + ".max(" + x + "!=" + x +
                      ",axis=1),axis=0)!=0)");
      }
    line(depth, "if " + llvm::join(bad, "|") + ":");
    if (auto err = emit(*loop->getBlock(), depth + 1, "active", loop))
      return err;
    for (auto [init, ret] : llvm::zip(accum, loop.getResults())) {
      line(depth + 1, init + "=" + name(ret));
      names[ret] = init;
    }
  }
  return llvm::Error::success();
}
