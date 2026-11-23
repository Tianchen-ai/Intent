// Lower a task's verified packed owner interval. Each logical owner has
// private SSA values; the CPU common-value marks identify shared loads.
using Bindings = llvm::DenseMap<Value, std::string>;
std::optional<intentdsl::cpu::RepeatedScore> reusable;
bool replayWindow = false;
std::string window, offset;
llvm::Error issueTogether(Block &bb, unsigned depth,
                          SmallVectorImpl<Bindings> &lanes) {
  for (auto &op : bb.without_terminator()) {
    if (auto loop = dyn_cast<scf::ForOp>(op)) {
      bool replay = reusable && reusable->replay->getBlock() == loop.getBody();
      for (auto &lane : lanes) {
        for (auto [arg, ret, init] :
             llvm::zip(loop.getRegionIterArgs(), loop.getResults(),
                       loop.getInitArgs())) {
          auto carry = fresh();
          line(depth, "var " + carry + "=" + lane.at(init));
          lane[arg] = lane[ret] = carry;
        }
      }
      auto iv = fresh();
      auto &first = lanes.front();
      auto start = first.at(loop.getLowerBound());
      auto head = "for " + iv + " in range(" + start + "," +
                  first.at(loop.getUpperBound()) + "," +
                  first.at(loop.getStep()) + "):";
      if (replay)
        line(depth, "if " + first.at(loop.getUpperBound()) + "-" + start +
                        "<=" + window + ":");
      for (unsigned variant = 0; variant < (replay ? 2u : 1u); ++variant) {
        if (replay && variant)
          line(depth, "else:");
        unsigned level = depth + int(replay);
        line(level, head);
        offset = "(" + iv + "-" + start + ")";
        replayWindow = replay && !variant;
        for (auto &lane : lanes)
          lane[loop.getInductionVar()] = iv;
        if (auto err = issueTogether(*loop.getBody(), level + 1, lanes))
          return err;
        for (auto &lane : lanes) {
          SmallVector<std::string> next;
          for (auto v : loop.getBody()->getTerminator()->getOperands()) {
            auto tmp = fresh();
            line(level + 1, "var " + tmp + "=" + lane.at(v));
            next.push_back(tmp);
          }
          for (auto [r, t] : llvm::zip(loop.getResults(), next))
            line(level + 1, lane.at(r) + "=" + t);
        }
      }
      replayWindow = false;
    } else if (replayWindow && reusable->replay == &op) {
      for (auto [i, lane] : llvm::enumerate(lanes))
        lane[op.getResult(0)] = "saved[unsafe_offset=" + offset + "*" +
                                std::to_string(lanes.size()) + "+" +
                                std::to_string(i) + "]";
    } else if (op.hasAttr("mojo_row.pack_scalars")) {
      Value input = op.getOperand(0), result = op.getResult(0);
      auto pack = fresh();
      SmallVector<std::string> scalars;
      for (auto &lane : lanes)
        scalars.push_back(lane.at(input));
      auto simd = "SIMD[" + dtype(input.getType()) + "," +
                  std::to_string(lanes.size()) + "](" +
                  llvm::join(scalars, ",") + ")";
      auto expr = op.getName().stripDialect() == "exp"
                      ? "exp(" + simd + ")"
                      : simd + ".cast[" + dtype(result.getType()) + "]()";
      line(depth, "var " + pack + "=" + expr);
      for (auto [i, lane] : llvm::enumerate(lanes))
        lane[result] = pack + "[" + std::to_string(i) + "]";
    } else {
      for (auto [i, lane] : llvm::enumerate(lanes)) {
        if (i && op.hasAttr("cpu.owner_common")) {
          auto v = op.getResult(0);
          lane[v] = lanes.front().at(v);
        } else {
          values.swap(lane);
          auto err = body(bb, depth, &op);
          values.swap(lane);
          if (err)
            return err;
        }
      }
    }
    if (reusable && &op == reusable->record) {
      line(depth, "if " + offset + "<" + window + ":");
      for (auto [i, lane] : llvm::enumerate(lanes))
        line(depth + 1, "saved[unsafe_offset=" + offset + "*" +
                            std::to_string(lanes.size()) + "+" +
                            std::to_string(i) +
                            "]=" + lane.at(op.getResult(0)));
    }
  }
  return llvm::Error::success();
}
llvm::Error issueBatch(unsigned batch) {
  auto &bb = space.getRegion(0).front();
  SmallVector<Bindings, 2> lanes(batch, values);
  if (auto slots = space.getAttrOfType<IntegerAttr>("cpu.batch_score_window")) {
    reusable = intentdsl::cpu::repeatedScore(space);
    window = std::to_string(slots.getInt());
    line(1, "var saved=unsafe_stack_allocation[" +
                std::to_string(batch * slots.getInt()) + ",Float32]()");
  }
  line(1, "for owner in range(Int(lo),Int(hi)," + std::to_string(batch) + "):");
  auto axes = space.getAttrOfType<DenseI64ArrayAttr>("axes");
  int64_t pitch = 1;
  for (unsigned i : intentdsl::cpu::ownerRadices(space)) {
    int64_t n =
        std::max<int64_t>(1, axis(space.getOperand(i), "cpu.shape", axes[i]));
    auto id = fresh();
    line(2, "var " + id + "=(owner//" + std::to_string(pitch) + ")%" +
                std::to_string(n));
    for (auto [lane, map] : llvm::enumerate(lanes)) {
      auto coord = id;
      if (pitch == 1 &&
          i == unsigned(space.getAttrOfType<IntegerAttr>("cpu.innermost_owner")
                            .getInt()) &&
          lane) {
        coord = fresh();
        line(2, "var " + coord + "=" + id + "+" + std::to_string(lane));
      }
      map[bb.getArgument(i)] = coord;
    }
    pitch *= n;
  }
  return issueTogether(bb, 2, lanes);
}
