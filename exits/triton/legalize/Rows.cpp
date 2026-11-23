#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"

namespace CI_UNIT_NAMESPACE {
using namespace mlir;
namespace {
struct Row {
  OpBuilder b;
  Location l;
  Operation &old;
  func::FuncOp fn;
  const Type i64, i32, i1, f32;
  Row(Operation &o)
      : b(&o), l(o.getLoc()), old(o), fn(o.getParentOfType<func::FuncOp>()),
        i64(b.getI64Type()), i32(b.getI32Type()), i1(b.getI1Type()),
        f32(b.getF32Type()) {}
  Type tile(int rank, Type e) {
    return rank ? Type(RankedTensorType::get(
                      SmallVector<int64_t>(rank, ShapedType::kDynamic), e))
                : e;
  }
  int rank(Value v) {
    auto t = dyn_cast<ShapedType>(v.getType());
    return t ? t.getRank() : 0;
  }
  Operation *op(StringRef n, ValueRange args, TypeRange result,
                ArrayRef<NamedAttribute> attrs = {}) {
    OperationState s(l, ("triton_src." + n).str());
    s.addOperands(args);
    s.addTypes(result);
    s.addAttributes(attrs);
    return b.create(s);
  }
  NamedAttribute str(StringRef k, StringRef v) {
    return b.getNamedAttr(k, b.getStringAttr(v));
  }
  NamedAttribute num(StringRef k, int64_t v) {
    return b.getNamedAttr(k, b.getI64IntegerAttr(v));
  }
  Value arg(int i) { return old.getOperand(i); }
  int64_t attr(StringRef n) {
    return old.getAttrOfType<IntegerAttr>(n).getInt();
  }
  Value constant(int64_t i) { return b.create<arith::ConstantIntOp>(l, i, 64); }
  Value meta(StringRef n) {
    return op("dimension", {}, i64, str("name", n))->getResult(0);
  }
  Value lanes(int64_t size) {
    return op("range", {}, tile(1, i64), num("size", size))->getResult(0);
  }
  Value expand(Value val, int i) {
    return op("expand", val,
              tile(rank(val) + 1, getElementTypeOrSelf(val.getType())),
              num("axis", i))
        ->getResult(0);
  }
  Value map(StringRef n, ValueRange args, Type element = {}) {
    int dims = 0;
    for (auto v : args)
      dims = std::max(dims, rank(v));
    return op("map", args,
              tile(dims, element
                             ? element
                             : getElementTypeOrSelf(args.front().getType())),
              str("kind", n))
        ->getResult(0);
  }
  Value binary(StringRef kind, Value x, Value y, Type e = {}) {
    return map(kind, {x, y}, e);
  }
  Value reduce(StringRef alg, Value v, int ax, Type type) {
    return op("axis_reduce", v, type,
              {str("algebra", alg), num("axis", ax),
               str("ci.implementation", "exits/triton::dynamic-collective")})
        ->getResult(0);
  }
  Value splat(Value v, ArrayRef<StringRef> dimensions) {
    return op("splat", v,
              tile(dimensions.size(), getElementTypeOrSelf(v.getType())),
              b.getNamedAttr("dimensions", b.getStrArrayAttr(dimensions)))
        ->getResult(0);
  }
  Value offset(Value view, ValueRange ix) {
    SmallVector<Value> args{view};
    for (auto x : ix)
      args.push_back(map("i64", x, i64));
    int dims = 0;
    for (Value v : ix)
      dims = std::max(dims, rank(v));
    return op("view_offset", args, tile(dims, i64),
              b.getNamedAttr("index_bits", b.getI32IntegerAttr(64)))
        ->getResult(0);
  }
  Value read(Value view, ValueRange ix, Value mask, int fill = 0) {
    Value address = offset(view, ix);
    return op("read_at", {view, address, mask},
              tile(std::max(rank(mask), rank(address)),
                   cast<MemRefType>(view.getType()).getElementType()),
              num("fill", fill))
        ->getResult(0);
  }
  void store(Value view, ValueRange ix, Value v, Value mask) {
    op("write_at", {view, offset(view, ix), v, mask}, {});
  }
  void replace(Value v) {
    old.getResult(0).replaceAllUsesWith(v);
    old.erase();
  }
  Value loop(Value limit, Value init, std::function<Value(Value, Value)> body,
             bool unroll = false) {
    auto step = b.create<arith::ConstantIndexOp>(l, 1),
         lo = b.create<arith::ConstantIndexOp>(l, 0);
    Value hi;
    if (auto c = limit.getDefiningOp<arith::ConstantIntOp>())
      hi = b.create<arith::ConstantIndexOp>(l, c.value());
    else
      hi = b.create<arith::IndexCastOp>(l, b.getIndexType(), limit);
    auto loop = b.create<scf::ForOp>(l, lo, hi, step,
                                     init ? ValueRange{init} : ValueRange{});
    if (!unroll)
      loop->setAttr("triton_src.runtime_range", b.getUnitAttr());
    b.setInsertionPointToStart(loop.getBody());
    auto val = body(loop.getInductionVar(),
                    init ? loop.getRegionIterArgs()[0] : Value());
    if (init)
      b.create<scf::YieldOp>(l, val);
    b.setInsertionPointAfter(loop);
    return init ? loop.getResult(0) : Value();
  }
  void bins(bool privateCounts) {
    int64_t n = attr("extent"), width = attr("native_panel"),
            span = privateCounts ? attr("span") : n;
    auto zero = b.create<arith::ConstantIntOp>(l, 0, 32);
    auto init = splat(zero, {"WIDTH"});
    auto part = loop(
        constant(llvm::divideCeil(span, width)), init,
        [&](Value piece, Value carry) {
          auto index = binary(
              "add", binary("mul", map("i64", piece, i64), constant(width)),
              lanes(width));
          Value partial;
          if (privateCounts) {
            auto pid = op("tile_pid", {}, i64, num("axis", 0))->getResult(0);
            auto pos = binary("add", binary("mul", pid, constant(span)), index);
            auto mask = binary("and", binary("lt", index, constant(span), i1),
                               binary("lt", pos, constant(n), i1));
            auto key = read(arg(0), pos, mask, -1);
            auto keep = binary("and", binary("ge", key, constant(0), i1),
                               binary("lt", key, meta("N"), i1));
            // Redirect masked lanes to bin zero and subtract their
            // contribution. This also works on devices which clamp invalid
            // histogram lanes.
            auto guarded = map("select", {keep, key, constant(0)},
                               getElementTypeOrSelf(key.getType()));
            auto counts =
                op("bin_count", map("i32", guarded, i32), tile(1, i32),
                   str("ci.implementation", "exits/triton::histogram"))
                    ->getResult(0);
            auto invalid = map("i32", binary("eq", keep, constant(0), i1), i32);
            auto extra = reduce("sum", invalid, 0, i32);
            auto empty = binary("eq", arg(2), constant(0), i1);
            partial = binary("sub", counts,
                             map("select", {empty, extra, zero}, i32), i32);
          } else {
            auto r = expand(index, 1), c = expand(arg(2), 0);
            auto mask = binary("and", binary("lt", r, constant(n), i1),
                               binary("lt", c, meta("N"), i1));
            partial =
                reduce("sum", read(arg(0), {r, c}, mask), 0, tile(1, i32));
          }
          return binary("add", carry, partial);
        });
    replace(expand(part, 0));
  }
  void fill() {
    Value col = arg(2), after = binary("add", col, constant(1)),
          live = binary("lt", after, meta("N"), i1);
    auto at = [&](Value i) {
      return binary("floordiv", map("i64", read(arg(1), i, live), i64),
                    constant(attr("unit")));
    };
    Value start = at(col), end = at(after),
          limit = reduce("max", binary("sub", end, start), -1, i64);
    loop(limit, {}, [&](Value it, Value) {
      auto cell = binary("add", start, map("i64", it, i64));
      store(arg(0), cell, map("i32", col, i32), binary("lt", cell, end, i1));
      return Value();
    });
    old.erase();
  }
  void segment(bool sizeOnly) {
    auto slot = fn->getAttrOfType<IntegerAttr>("kir.domain_arg").getInt();
    bool flat = cast<MemRefType>(fn.getArgument(slot).getType()).getRank() == 1;
    Value owner = expand(arg(flat ? 2 : 1), flat ? 0 : 1),
          owners = binary("lt", owner, meta(flat ? "N" : "M"), i1);
    auto cursor = map("i64", read(arg(0), owner, owners), i64),
         end =
             map("i64", read(arg(0), binary("add", owner, constant(1)), owners),
                 i64);
    auto length = binary("sub", end, cursor);
    if (sizeOnly) {
      replace(map("f32", length, f32));
      return;
    }
    auto panel = old.getAttrOfType<IntegerAttr>("member_batch");
    auto w = panel ? panel.getInt() : 1;
    auto limit = binary("ceildiv", reduce("max", length, -1, i64), constant(w));
    auto zero = b.create<arith::ConstantOp>(l, b.getF32FloatAttr(0));
    auto result = loop(
        limit, splat(zero, {"ROWS", "WIDTH"}), [&](Value piece, Value acc) {
          auto start =
              binary("add", cursor,
                     binary("mul", map("i64", piece, i64), constant(w)));
          auto live = binary("and", binary("lt", start, end, i1), arg(3));
          auto pos = start, cols = expand(arg(2), 0);
          if (panel) {
            pos =
                binary("add", expand(start, 2), expand(expand(lanes(w), 0), 0));
            live = binary("and", binary("lt", pos, expand(end, 2), i1),
                          expand(arg(3), 2));
            cols = expand(cols, 2);
          }
          auto &body = old.getRegion(0).front();
          IRMapping mapping;
          mapping.map(body.getArgument(0), pos);
          mapping.map(body.getArgument(1), cols);
          for (auto &term : body.without_terminator()) {
            SmallVector<Value> args;
            for (auto v : term.getOperands())
              args.push_back(mapping.lookupOrDefault(v));
            auto code = term.getName().stripDialect();
            Value val =
                code == "member_load"
                    ? read(args.front(), ArrayRef<Value>(args).drop_front(),
                           live)
                    : map(code == "mul" ? "mul_rn" : "add_rn", args, f32);
            mapping.map(term.getResult(0), val);
          }
          auto value = mapping.lookup(body.back().getOperand(0));
          auto add = [&](Value prev, Value delta, Value valid) {
            return map("select", {valid, binary("add_rn", prev, delta), prev},
                       f32);
          };
          if (!panel)
            return add(acc, value, live);
          return loop(
              constant(w), acc,
              [&](Value lane, Value prev) {
                auto v = op("extract_lane", {value, lane}, tile(2, f32))
                             ->getResult(0);
                auto mask = binary(
                    "and",
                    binary("lt", binary("add", start, map("i64", lane, i64)),
                           end, i1),
                    arg(3));
                return add(prev, v, mask);
              },
              true);
        });
    replace(result);
  }
  void publishOffsets() {
    auto key = expand(arg(2), 0), next = binary("add", key, constant(1));
    auto total = arg(4);
    Value yes = b.create<arith::ConstantIntOp>(l, 1, 1);
    store(arg(0), constant(0), constant(0), yes);
    store(arg(0), next, total, arg(3));
    auto final = binary("eq", next, meta("N"), i1);
    store(arg(1), binary("mul", key, constant(0)), total,
          binary("and", arg(3), final, i1));
    old.erase();
  }
  void indirect(StringRef code) {
    if (code == "indexed_tile_read") {
      auto key = read(arg(1), arg(2), binary("lt", arg(2), meta("M"), i1));
      replace(read(arg(0), {expand(key, 1), expand(arg(3), 0)}, arg(4)));
    } else if (code == "owner_load")
      replace(read(arg(0), arg(1), binary("lt", arg(1), meta("M"), i1)));
    else if (code == "owner_store") {
      store(arg(0), arg(1), arg(2), binary("lt", arg(1), meta("M"), i1));
      old.erase();
    } else if (code == "index_read_at")
      replace(read(arg(0), arg(1), arg(2)));
    else if (code == "unique_put_at") {
      store(arg(0), arg(1), arg(2), arg(3));
      old.erase();
    } else
      replace(op("fetch_add", {arg(0), offset(arg(0), arg(1)), arg(2), arg(3)},
                 old.getResult(0).getType(), str("scope", "gpu"))
                  ->getResult(0));
  }
  void atomicBins() {
    Value source = arg(0), target = arg(1), col = arg(2),
          partition = op("tile_pid", {}, i64, num("axis", 0))->getResult(0);
    auto dst = [&](Value index) {
      return old.getAttrOfType<BoolAttr>("transposed").getValue()
                 ? SmallVector<Value>{index, partition}
                 : SmallVector<Value>{partition, index};
    };
    auto width = attr("native_panel"), span = attr("span"),
         extent = attr("extent");
    // The legal boundary verifies the clear/barrier/panel sequence before
    // specializing its address projections and the load/atomic pair.
    for (Operation &item : old.getRegion(0).front()) {
      auto n = item.getName().stripDialect();
      if (n == "clear_bins")
        store(target, dst(col), constant(0), binary("lt", col, meta("N"), i1));
      else if (n == "cta_barrier")
        op("cta_barrier", {}, {});
      else if (n == "bin_panels") {
        auto panels = item.getAttrOfType<IntegerAttr>("count").getInt();
        loop(constant(panels), {}, [&](Value piece, Value) {
          auto off = binary(
              "add", binary("mul", map("i64", piece, i64), constant(width)),
              lanes(width));
          auto pos =
              binary("add", binary("mul", partition, constant(span)), off);
          auto mask = binary("and", binary("lt", off, constant(span), i1),
                             binary("lt", pos, constant(extent), i1));
          auto key = map("i64", read(source, pos, mask, -1), i64);
          auto keep = binary("and", binary("ge", key, constant(0), i1),
                             binary("lt", key, meta("N"), i1));
          op("fetch_add", {target, offset(target, dst(key)), constant(1), keep},
             tile(1, i32), str("scope", "cta"));
          return Value();
        });
      }
    }
    old.erase();
  }
  void scatter() {
    auto zero = b.create<arith::ConstantIntOp>(l, 0, 32);
    auto total = reduce("max", map("select", {arg(4), arg(6), zero}, i32), 1,
                        tile(1, i32));
    auto slots =
        map("select",
            {arg(5), binary("sub", arg(6), constant(1)),
             binary("sub", binary("add", expand(total, 1), expand(arg(3), 0)),
                    arg(6))},
            i64);
    auto v = map("select", {arg(5), arg(7), constant(attr("fill"))}, i32);
    store(arg(0), {expand(arg(2), 1), slots}, v, arg(4));
    store(arg(1), arg(2), total, binary("lt", arg(2), meta("M"), i1));
    old.erase();
  }
};
} // namespace
static StringRef binding(StringRef op) {
  return llvm::StringSwitch<StringRef>(op)
      .Case("triton_src.summary_reduce", "moments-reduction")
      .Case("triton_src.scan_sum", "inclusive-scan")
      .Case("triton_src.reduce", "native-row-reduce")
      .Default("");
}
void lowerRows(func::FuncOp fn) {
  fn.walk<WalkOrder::PostOrder>([](Operation *op) {
    auto code = op->getName().getStringRef();
    auto implementation = binding(code);
    if (!implementation.empty()) {
      Builder b(op->getContext());
      op->setAttr("ci.implementation",
                  b.getStringAttr(("exits/triton::" + implementation).str()));
      if (code == "triton_src.summary_reduce")
        op->getParentOfType<func::FuncOp>()->setAttr("triton_src.moments",
                                                     b.getUnitAttr());
    }
    if (code == "triton_src.offsets")
      op->setAttr("index_bits",
                  Builder(op->getContext()).getI32IntegerAttr(64));
    auto kind = op->getName().stripDialect();
    if (code == "triton_src.atomic_bins")
      Row(*op).atomicBins();
    else if (op->getName().getDialectNamespace() == "triton_src" &&
             (kind == "indexed_tile_read" || kind == "owner_load" ||
              kind == "owner_store" || kind == "unique_put_at" ||
              kind == "index_read_at" || kind == "atomic_add_at"))
      Row(*op).indirect(kind);
    else if (code == "triton_src.publish_offsets")
      Row(*op).publishOffsets();
    else if (code == "triton_src.fill_ranges_at")
      Row(*op).fill();
    else if (code == "triton_src.key_counts" ||
             code == "triton_src.sum_columns")
      Row(*op).bins(code == "triton_src.key_counts");
    else if (code == "triton_src.segment_fold" ||
             code == "triton_src.segment_size")
      Row(*op).segment(code == "triton_src.segment_size");
    else if (code == "triton_src.scatter_fragment")
      Row(*op).scatter();
  });
}
LogicalResult verifyDynamicRows(func::FuncOp fn) {
  bool moments = false;
  auto valid = fn.walk([&](Operation *op) {
    auto code = op->getName().getStringRef();
    auto reject = [&](StringRef reason) {
      op->emitError(reason);
      return WalkResult::interrupt();
    };
    if (code == "triton_src.summary_reduce")
      moments = true;
    if (code == "triton_src.view_offset" || code == "triton_src.offsets") {
      if (op->getAttr("index_bits") !=
          Builder(op->getContext()).getI32IntegerAttr(64))
        return reject("row view requires wide indices");
      if (code == "triton_src.view_offset") {
        if (op->getNumOperands() !=
            cast<MemRefType>(op->getOperand(0).getType()).getRank() + 1)
          return reject("row view coordinate count disagrees with ABI");
        for (auto x : op->getOperands().drop_front())
          if (!getElementTypeOrSelf(x.getType()).isInteger(64))
            return reject("row address coordinate must be i64");
      }
    }
    if (code == "triton_src.dimension") {
      auto n = op->getAttrOfType<StringAttr>("name");
      if (n != "M" && n != "N")
        return reject("unknown bound row dimension");
    }
    if (code == "triton_src.splat")
      for (auto a : op->getAttrOfType<ArrayAttr>("dimensions")) {
        auto d = cast<StringAttr>(a);
        if (d != "ROWS" && d != "WIDTH")
          return reject("unknown launch-row extent");
      }
    if (code == "triton_src.read_at" || code == "triton_src.write_at" ||
        code == "triton_src.fetch_add") {
      if (code == "triton_src.fetch_add") {
        auto scope = op->getAttrOfType<StringAttr>("scope");
        if (scope != "gpu" && scope != "cta")
          return reject("invalid atomic scope");
      }
      bool read = code == "triton_src.read_at";
      auto *offset = op->getOperand(1).getDefiningOp();
      if (!offset ||
          offset->getName().getStringRef() != "triton_src.view_offset" ||
          offset->getOperand(0) != op->getOperand(0) ||
          !getElementTypeOrSelf(op->getOperand(read ? 2 : 3).getType())
               .isInteger(1))
        return reject("invalid row address or access mask");
    }
    if (code == "triton_src.map") {
      auto k = op->getAttrOfType<StringAttr>("kind").getValue();
      auto arity = llvm::StringSwitch<int>(k)
                       .Cases("i64", "i32", "f32", 1)
                       .Case("select", 3)
                       .Cases("add", "sub", "mul", "floordiv", "ceildiv", 2)
                       .Cases("and", "lt", "ge", "eq", "add_rn", "mul_rn", 2)
                       .Default(0);
      if (!arity || op->getNumOperands() != arity)
        return reject("invalid typed row expression");
    }
    return WalkResult::advance();
  });
  return success(!valid.wasInterrupted() &&
                 moments == fn->hasAttr("triton_src.moments"));
}
} // namespace CI_UNIT_NAMESPACE
