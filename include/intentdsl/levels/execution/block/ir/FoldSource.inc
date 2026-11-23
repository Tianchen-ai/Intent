// Spell a verified associative fold with one owner and a tile of independent
// contributions. Ordinary ordered loops never enter this path.
llvm::Error emitFold(mlir::scf::ForOp loop, unsigned depth, int64_t span) {
  using namespace mlir;
  auto lib = language().str(), sz = std::to_string(span);
  auto extent = [&](Value v) {
    return std::to_string(std::max<uint64_t>(
        1, llvm::PowerOf2Ceil(kir::memberExtent(fn, v, "block.shape"))));
  };
  auto scalar = [&](StringRef v) {
    return lib + ".reshape(" + broadcast(v) + ",())";
  };
  auto lo = scalar(name(loop.getLowerBound())),
       hi = scalar(name(loop.getUpperBound()));
  auto base = fresh(), iv = fresh(), live = fresh();
  auto &bb = *loop.getBody();
  auto yield = bb.getTerminator();
  SmallVector<std::string> accum;
  llvm::DenseSet<Operation *> combiners;
  for (auto [init, ret, next] :
       llvm::zip(loop.getInitArgs(), loop.getResults(), yield->getOperands())) {
    auto id = fresh();
    accum.push_back(id);
    names[ret] = id;
    line(depth, id + "=" +
                    (isa<ShapedType>(ret.getType()) ? name(init)
                                                    : broadcast(name(init))));
    combiners.insert(next.getDefiningOp());
  }
  auto ix = IndexType::get(loop.getContext());
  line(depth, base + "=(" + cast(lo, ix) + "//" + sz + ")*" + sz);
  line(depth, "while " + base + "<" + hi + ":");
  line(depth + 1, iv + "=" + base + "+" + memberRange(span));
  line(depth + 1, live + "=(" + iv + "<" + hi + ")&(" + iv + ">=" + lo + ")");
  llvm::DenseMap<Value, std::string> local;
  local[loop.getInductionVar()] = iv;
  auto value = [&](Value v) {
    auto it = local.find(v);
    if (it != local.end())
      return it->second;
    if (isa<MemRefType>(v.getType()))
      return name(v);
    return isa<RankedTensorType>(v.getType())
               ? lib + ".reshape(" + name(v) + ",(1," + extent(v) + "))"
               : scalar(name(v));
  };
  auto expand = [&](std::string x) {
    return lib + ".expand_dims(" + lib + ".broadcast_to(" + x + ",(" + sz +
           ",)),1)";
  };
  for (auto &op : bb.without_terminator()) {
    if (combiners.contains(&op))
      continue;
    auto v = op.getResult(0);
    auto kind = op.getName().stripDialect();
    auto arg = [&](unsigned i) { return value(op.getOperand(i)); };
    std::string e;
    if (kind == "slice_read") {
      auto view = op.getOperand(0);
      auto axis = op.getAttrOfType<IntegerAttr>("axis").getInt();
      SmallVector<std::string> coords;
      for (auto c : op.getOperands().drop_front())
        coords.push_back(expand(value(c)));
      auto k = kir::memberExtent(fn, v, "block.shape");
      auto range =
          lib + ".expand_dims(" + memberRange(std::stoll(extent(v))) + ",0)";
      coords.insert(coords.begin() + axis, range);
      if (op.hasAttr("block.rectangular_tokens") && llvm::isPowerOf2_64(k)) {
        SmallVector<std::string> outer;
        for (auto c : op.getOperands().drop_front().drop_back())
          outer.push_back(value(c));
        e = loadMemberPanel(op, outer, base, span, k, expand(live));
      }
      if (op.hasAttr("block.page_extent") && llvm::isPowerOf2_64(k)) {
        SmallVector<std::string> at{value(op.getOperand(1))};
        for (auto c : op.getOperands().drop_front(3))
          at.push_back(value(c));
        e = loadPagePanel(op, at, base, span, k, expand(live));
      }
      if (e.empty())
        e = memberAccess(
            op, coords,
            expand(live) + "&(" + range + "<" + std::to_string(k) + ")", false);
    } else if (kind == "lane_sum") {
      auto t = op.getOperand(0);
      auto n = kir::memberExtent(fn, t, "block.shape");
      auto r =
          lib + ".expand_dims(" + memberRange(std::stoll(extent(t))) + ",0)";
      e = lib + ".sum(" + lib + ".where(" + r + "<" + std::to_string(n) + "," +
          arg(0) + ",0.),axis=1)";
    } else if (kind == "lane_splat")
      e = lib + ".broadcast_to(" + expand(arg(0)) + ",(" + sz + "," +
          extent(v) + "))";
    else if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(op)) {
      SmallVector<std::string> coords;
      for (auto i : load.getIndices())
        coords.push_back(value(i));
      e = gatherScalars(load, coords, live);
    } else if (auto c = dyn_cast<arith::ConstantOp>(op)) {
      if (auto i = dyn_cast<IntegerAttr>(c.getValue()))
        e = std::to_string(i.getInt());
      else {
        auto f = mlir::cast<FloatAttr>(c.getValue()).getValue();
        llvm::SmallString<32> s;
        f.toString(s);
        e = f.isInfinity() ? (f.isNegative() ? "-float('inf')" : "float('inf')")
                           : s.str().str();
      }
    } else if (kind == "index_cast" || kind == "extf" || kind == "truncf") {
      Type t = v.getType();
      if (auto st = dyn_cast<ShapedType>(t))
        t = st.getElementType();
      e = cast(arg(0), t);
    } else if (kind == "exp" || kind == "exp2")
      e = lib + "." + kind.str() + "(" + arg(0) + ")";
    else {
      auto tok = llvm::StringSwitch<StringRef>(kind)
                     .Cases("addi", "addf", "+")
                     .Cases("subi", "subf", "-")
                     .Cases("muli", "mulf", "*")
                     .Case("remsi", "%")
                     .Case("floordivsi", "//")
                     .Default("");
      if (tok.empty())
        return llvm::createStringError("unhandled fold contribution: " + kind);
      e = arg(0) + tok.str() + arg(1);
    }
    auto id = fresh();
    line(depth + 1, id + "=" + e);
    local[v] = id;
  }
  for (auto [dest, v] : llvm::zip(accum, yield->getOperands())) {
    auto *comb = v.getDefiningOp();
    bool maximum = isa<arith::MaxNumFOp>(comb),
         vector = isa<ShapedType>(v.getType());
    auto rhs = value(comb->getOperand(1));
    auto filtered = lib + ".where(" + (vector ? expand(live) : live) + "," +
                    rhs + "," + (maximum ? "-float('inf')" : "0.") + ")";
    auto part = lib + (maximum ? ".max(" : ".sum(") + filtered + ",axis=0)";
    line(depth + 1, dest + "=" +
                        (maximum ? lib + ".maximum(" + dest + "," + part + ")"
                                 : dest + "+" + part));
  }
  line(depth + 1, base + "=" + base + "+" + sz);
  return llvm::Error::success();
}
