#include "RecordFormat.h"
#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "intentdsl/levels/execution/cpu/ir/MatrixPanels.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
using namespace mlir;
namespace CI_UNIT_NAMESPACE {
namespace {
using intentdsl::cpu::MatrixPanels;
std::string list(ArrayRef<int64_t> values) {
  SmallVector<std::string> s;
  for (auto v : values)
    s.push_back(std::to_string(v));
  return "[" + llvm::join(s, ",") + "]";
}
struct Types {
  std::string carrier;
  SmallVector<int64_t> dims, axes;
  std::string element() const {
    return llvm::StringSwitch<std::string>(carrier)
        .Case("i32", "si32")
        .Case("i16", "si16")
        .Case("i8", "si8")
        .Case("u8", "ui8")
        .Case("f8e4m3fn", "f8E4M3FN")
        .Default(carrier);
  }
  std::string type(StringRef kind) const {
    auto e = carrier == "f8e4m3fn" ? std::string("f8E4M3FN") : carrier;
    if (kind == "value" && dims.empty())
      return element();
    bool record = e == "Q4_K" || e == "Q8_K";
    auto enc = record ? "!weft_kernel.encoding<\"" + e + "\",\"base\",\"" + e +
                            "\",[]>"
               : kind == "value" ? element()
                                 : "!weft_kernel.encoding<\"" + e +
                                       "\",\"dense\",\"dense." + e + "\",[]>";
    return "!weft_kernel." + kind.str() + "<" + enc + "," + list(dims) + "," +
           list(axes) + ">";
  }
};
struct Term {
  std::string id, type;
};
struct Slot {
  Types t;
  std::string address;
  SmallVector<int64_t> strides;
  bool writes;
};
struct Writer {
  MatrixPanels &p;
  SmallVector<Slot, 8> slots;
  std::string text;
  llvm::raw_string_ostream s{text};
  Term root;
  unsigned next = 0;
  bool encoded() const {
    return p.fn.front().front().hasAttr("weft_host.record_fields");
  }
  bool emptyReduction() const {
    return p.fn.front().front().hasAttr("weft_host.empty_reduction");
  }
  bool packetDot() const {
    auto attr = p.domain.getAttrOfType<StringAttr>("cpu.packing");
    return attr && attr == "reduction-packets";
  }
  Term inst(StringRef name, ArrayRef<Term> args, StringRef ty,
            StringRef attrs = {}) {
    SmallVector<std::string> names, types;
    for (auto &a : args) {
      names.push_back(a.id);
      types.push_back(a.type);
    }
    std::string id = "%v" + std::to_string(next++);
    if (ty != "()")
      s << id << "=";
    s << "\"weft_kernel." << name << "\"(" << llvm::join(names, ",") << ")";
    if (!attrs.empty())
      s << " <{" << attrs << "}>";
    s << ":(" << llvm::join(types, ",") << ")->" << ty << "\n";
    return {id, ty.str()};
  }
  Term op(StringRef name, Term a, Term b, StringRef t) {
    return inst("binary", {a, b}, t, ("kind=\"" + name + "\"").str());
  }
  Term convert(Term a, StringRef t) {
    return a.type == t ? a : inst("cast", {a}, t);
  }
  Term lit(StringRef n, StringRef t) {
    return inst("constant", {}, t, ("value=" + n + ":" + t).str());
  }
  Term idx(int64_t n) {
    auto id = "%i" + std::to_string(next++);
    s << id << "=arith.constant " << n << ":index\n";
    return {id, "index"};
  }
  Term slice(unsigned i, ArrayRef<std::string> indices, bool read = true) {
    auto &v = slots[i].t;
    Types sub{v.carrier, {}, {}};
    SmallVector<std::string> sel;
    SmallVector<Term> args{{"%a" + std::to_string(i), v.type("view")}};
    for (auto [n, j] : llvm::enumerate(indices)) {
      if (j == "*") {
        sel.push_back("\"all\"");
        sub.dims.push_back(v.dims[n]);
        sub.axes.push_back(v.axes[n]);
      } else {
        args.push_back({j, "index"});
        sel.push_back("\"index\"");
      }
    }
    auto t = inst("slice", args, sub.type("slice"),
                  "selectors=[" + llvm::join(sel, ",") + "]");
    return read ? inst("admit", {t}, sub.type("value")) : t;
  }
  void signatures() {
    auto add = [&](unsigned i, SmallVector<int64_t> dims,
                   SmallVector<int64_t> axes) {
      auto &v = p.views[i];
      slots.push_back({{v.dtype, dims, axes},
                       "p" + std::to_string(i),
                       SmallVector<int64_t>(v.strides.begin(), v.strides.end()),
                       v.writes});
    };
    auto k = p.shape[2], g = k / 32;
    auto n = p.dot.getName().stripDialect();
    if (emptyReduction()) {
      unsigned last = p.views.size() - 1;
      for (unsigned i = 2; i < last; ++i)
        if (p.views[last].strides.size() == 1)
          add(i, {-1}, {1});
        else
          add(i, {-2}, {2});
      if (p.views[last].strides.size() == 1)
        add(last, {-1}, {1});
      else
        add(last, {-1, -2}, {1, 2});
      return;
    }
    if (n == "q4k_dot") {
      if (encoded()) {
        slots.push_back(
            {{"Q4_K", {-1, k / 256, 256}, {1, 4, 3}}, "p0", {}, false});
        slots.push_back({{"Q8_K", {k / 256, 256}, {4, 3}}, "p1", {}, false});
        for (unsigned i = 2; i < p.views.size(); ++i)
          add(i, {-1}, {1});
        return;
      }
      auto a = p.views[0].strides;
      slots.push_back({{"u8", {-1, k / 256, 16}, {1, 4, 5}},
                       "p0",
                       {a[0], a[1], a[2]},
                       false});
      slots.push_back({{"u8", {-1, k / 256, 4, 32}, {1, 4, 6, 3}},
                       "(uint8_t*)p0+" + std::to_string(16 * a[2]),
                       {a[0], a[1], 32 * a[2], a[2]},
                       false});
      slots.push_back({{"f32", {k / 256}, {4}}, "p1", {73}, false});
      slots.push_back({{"i8", {k / 256, 8, 32}, {4, 7, 3}},
                       "(int8_t*)p1+4",
                       {292, 32, 1},
                       false});
      slots.push_back({{"i16", {k / 256, 16}, {4, 8}},
                       "(int16_t*)p1+130",
                       {146, 1},
                       false});
      for (unsigned i = 2; i < p.views.size(); ++i)
        add(i, {-1}, {1});
      return;
    }
    int64_t cols = p.shape[1] <= p.tile[1]       ? p.shape[1]
                   : p.shape[1] % p.tile[1] == 0 ? p.tile[1]
                                                 : -2;
    if (packetDot()) {
      add(0, {-1, k}, {1, 3});
      slots.push_back({{"i8", {cols, k}, {2, 3}}, "p1", {}, false});
      for (unsigned i = 2; i + 1 < p.views.size(); ++i)
        add(i, {cols}, {2});
      add(p.views.size() - 1, {-1, cols}, {1, 2});
      return;
    }
    add(0, {-1, k}, {1, 3});
    add(1, {k, cols}, {3, 2});
    for (unsigned i = slots.size(); i + 1 < p.views.size(); ++i)
      add(i, {cols}, {2});
    add(p.views.size() - 1, {-1, cols}, {1, 2});
  }
  void begin() {
    SmallVector<std::string> names, effects, aliases, params, strided;
    for (auto [i, v] : llvm::enumerate(slots)) {
      auto a = "a" + std::to_string(i);
      names.push_back("\"" + a + "\"");
      params.push_back("%" + a + ":" + v.t.type("view"));
      effects.push_back(v.writes ? "\"write\"" : "\"read\"");
      aliases.push_back("0");
      if (!v.strides.empty())
        strided.push_back(names.back());
    }
    s << "module {\n";
    if (encoded()) {
      declareRecord(s, true);
      declareRecord(s, false);
    }
    s << "\"weft_kernel.kernel\"() <{sym_name=\"matrix_panel\",source=\"CPU "
         "output panel\",arg_names=["
      << llvm::join(names, ",") << "],arg_access=[" << llvm::join(effects, ",")
      << "],arg_alias_sets=array<i64:" << llvm::join(aliases, ",")
      << ">,shape_symbols=[\"R\",\"C\"],strided_args=["
      << llvm::join(strided, ",") << "]}> ({\n^bb0(" << llvm::join(params, ",")
      << "):\n";
    for (auto arg : {"R", "C"})
      s << "%" << arg
        << "=\"weft_kernel.symbol\"() "
           "<{choices=array<i64>,kind=\"shape\",name=\""
        << arg << "\"}>:()->index\n";
    root = inst("root_domain", {},
                "!weft_kernel.domain<\"root\",0,-1,0,\"root\",\"exact\">");
  }
  Term dot(Term a, Term b, const Types &result) {
    return inst("contract", {a, b}, result.type("value"),
                "over=array<i64:3>,acc_type=" + result.element() +
                    ",engine=\"auto\"");
  }
  void ordinary() {
    if (packetDot()) {
      packetIntegers();
      return;
    }
    auto z = idx(0), one = idx(1), end = idx(p.shape[2]);
    auto cols = slots.back().t.dims[1];
    bool integer = p.views[0].dtype == "i8";
    Types vec{integer ? "i32" : "f32", {cols}, {2}};
    auto t = vec.type("value"), et = vec.element();
    int nr = p.fn.front()
                 .front()
                 .getAttrOfType<IntegerAttr>("weft_host.register_rows")
                 .getInt();
    auto grain = idx(nr);
    s << "scf.for %r=" << z.id << " to %R step " << grain.id << " {\n";
    SmallVector<std::string> rows{"%r"}, carries, types, computed;
    auto init =
        inst("new", {lit(integer ? "0" : "0.0", et)}, t, "initialized=true");
    for (int m = 0; m < nr; ++m) {
      if (m)
        rows.push_back(op("add", {"%r", "index"}, idx(m), "index").id);
      carries.push_back("%p" + std::to_string(m) + "=" + init.id);
      types.push_back(t);
    }
    s << "%acc" << (nr == 1 ? "" : ":" + std::to_string(nr))
      << "=scf.for %k=" << z.id << " to " << end.id << " step " << one.id
      << " iter_args(" << llvm::join(carries, ",") << ")->("
      << llvm::join(types, ",") << "){\n";
    auto b = convert(slice(1, {"%k", "*"}), t);
    for (int m = 0; m < nr; ++m) {
      auto a = convert(slice(0, {rows[m], "%k"}), et);
      computed.push_back(
          op("add", {"%p" + std::to_string(m), t}, op("mul", b, a, t), t).id);
    }
    s << "scf.yield " << llvm::join(computed, ",") << ":"
      << llvm::join(types, ",") << "\n}\n";
    for (int m = 0; m < nr; ++m) {
      Term acc{nr == 1 ? "%acc" : "%acc#" + std::to_string(m), t};
      unsigned bias = 2;
      auto dst = vec;
      for (auto &o : p.domain.getRegion(0).front()) {
        auto n = o.getName().stripDialect();
        if (n == "add_columns")
          acc = op("add", acc, slice(bias++, {"*"}), acc.type);
        if (n == "convert_matrix") {
          auto ty = cast<ShapedType>(o.getResult(0).getType()).getElementType();
          dst.carrier = ty.isF16()    ? "f16"
                        : ty.isBF16() ? "bf16"
                        : ty.isF32()  ? "f32"
                                      : "i32";
          acc = convert(acc, dst.type("value"));
        }
      }
      inst("commit", {acc, slice(slots.size() - 1, {rows[m], "*"}, false)},
           "()");
    }
    s << "}\n";
  }
  void publishIdentity() {
    auto start = idx(0), one = idx(1);
    auto &dst = slots.back().t;
    bool integer = p.dot.getName().stripDialect() == "contract" &&
                   p.views[0].dtype == "i8";
    s << "scf.for %r=" << start.id << " to %R step " << one.id << " {\n";
    Term value = lit(integer ? "0" : "0.0", integer ? "si32" : "f32");
    if (dst.dims.size() == 2) {
      s << "scf.for %c=" << start.id << " to %C step " << one.id << " {\n";
      unsigned bias = 0;
      for (auto &instruction : p.domain.getRegion(0).front()) {
        auto n = instruction.getName().stripDialect();
        if (n == "add_columns")
          value = op("add", value, slice(bias++, {"%c"}), value.type);
        if (n == "convert_matrix") {
          auto t = cast<ShapedType>(instruction.getResult(0).getType())
                       .getElementType();
          value = convert(value, t.isBF16()  ? "bf16"
                                 : t.isF16() ? "f16"
                                 : t.isF32() ? "f32"
                                             : "si32");
        }
      }
      inst("commit", {value, slice(slots.size() - 1, {"%r", "%c"}, false)},
           "()");
      s << "}\n";
    } else
      writeRecord(value, 0);
    s << "}\n";
  }
  void writeRecord(Term value, unsigned bias) {
    for (auto &inst : p.domain.getRegion(0).front()) {
      auto name = inst.getName().stripDialect();
      if (name == "add_columns")
        value = op("add", value, slice(bias++, {"%r"}), value.type);
      if (name == "convert_matrix") {
        auto t = cast<ShapedType>(inst.getResult(0).getType()).getElementType();
        value = convert(value, t.isBF16()  ? "bf16"
                               : t.isF16() ? "f16"
                               : t.isF32() ? "f32"
                                           : "si32");
      }
    }
    inst("commit", {value, slice(slots.size() - 1, {"%r"}, false)}, "()");
  }
  void packetIntegers() {
    auto z = idx(0), one = idx(1);
    s << "scf.for %r=" << z.id << " to %R step " << one.id << " {\n";
    auto a = slice(0, {"%r", "*"});
    s << "scf.for %c=" << z.id << " to %C step " << one.id << " {\n";
    auto acc = dot(a, slice(1, {"%c", "*"}), {"i32", {}, {}});
    for (unsigned i = 2; i + 1 < slots.size(); ++i)
      acc = op("add", acc, slice(i, {"%c"}), acc.type);
    inst("commit", {acc, slice(slots.size() - 1, {"%r", "%c"}, false)}, "()");
    s << "}\n}\n";
  }
  void records() {
    if (encoded()) {
      nativeRecords();
      return;
    }
    auto z = idx(0), one = idx(1), ng = idx(p.shape[2] / 256), eight = idx(8);
    auto fz = lit("0.0", "f32"), iz = lit("0", "si32");
    auto c = [&](int n) { return lit(std::to_string(n), "si32"); };
    s << "scf.for %r=" << z.id << " to %R step " << one.id
      << " {\n%acc=scf.for %g=" << z.id << " to " << ng.id << " step " << one.id
      << " iter_args(%a=" << fz.id << ")->(f32){\n";
    auto two = idx(2), four = idx(4);
    auto aidx = [&](Term t) { return convert(t, "index").id; };
    auto byte = [&](std::string off) {
      return convert(slice(0, {"%r", "%g", off}), "si32");
    };
    auto half = [&](int off) {
      auto lo = byte(idx(off).id), hi = byte(idx(off + 1).id);
      auto bits = op("or", lo, op("shl", hi, c(8), "si32"), "si32");
      return convert(inst("bitcast", {convert(bits, "ui16")}, "f16"), "f32");
    };
    auto d = half(0), dm = half(2);
    auto active = [&](Term x) {
      auto raw = inst("bitcast", {x}, "si32");
      return inst("compare", {op("and", raw, c(2147483647), "si32"), iz}, "i1",
                  "predicate=\"ne\"");
    };
    auto nonzero = active(d), minNonzero = active(dm);
    auto contributes = op("or", nonzero, minNonzero, "i1");
    s << "%record=scf.if " << contributes.id << " -> (f32){\n";
    s << "%sums:2=scf.for %j=" << z.id << " to " << eight.id << " step "
      << one.id << " iter_args(%S=" << iz.id << ",%T=" << iz.id
      << ")->(si32,si32){\n";
    auto j = convert({"%j", "index"}, "si32"), sub = op("and", j, c(3), "si32");
    auto l = byte(aidx(op("add", sub, c(4), "si32"))),
         h = byte(aidx(op("add", sub, c(8), "si32"))),
         u = byte(aidx(op("add", sub, c(12), "si32")));
    auto mask = inst("compare", {j, c(4)}, "i1", "predicate=\"lt\"");
    auto factor = [&](Term low, Term high) {
      return inst(
          "select",
          {mask, op("and", low, c(63), "si32"),
           op("or", high, op("shl", op("shr", low, c(6), "si32"), c(4), "si32"),
              "si32")},
          "si32");
    };
    auto sc = factor(l, op("and", u, c(15), "si32")),
         mn = factor(h, op("shr", u, c(4), "si32"));
    auto raw = slice(1, {"%r", "%g", aidx(op("shr", j, c(1), "si32")), "*"});
    auto shift =
        convert(op("shl", op("and", j, c(1), "si32"), c(2), "si32"), "ui8");
    auto nib =
        op("and", op("shr", raw, shift, raw.type), lit("15", "ui8"), raw.type);
    auto q = slice(3, {"%g", "%j", "*"});
    auto summed = dot(convert(nib, Types{"i8", {32}, {3}}.type("value")), q,
                      {"i32", {}, {}});
    auto s0 = op("add", {"%S", "si32"}, op("mul", sc, summed, "si32"), "si32");
    auto jj = op("shl", j, c(1), "si32");
    auto sumq =
        op("add", convert(slice(4, {"%g", aidx(jj)}), "si32"),
           convert(slice(4, {"%g", aidx(op("add", jj, c(1), "si32"))}), "si32"),
           "si32");
    auto s1 = op("add", {"%T", "si32"}, op("mul", mn, sumq, "si32"), "si32");
    s << "scf.yield " << s0.id << "," << s1.id << ":si32,si32\n}\n";
    auto pos = op("mul", d, convert({"%sums#0", "si32"}, "f32"), "f32"),
         neg = op("mul", dm, convert({"%sums#1", "si32"}, "f32"), "f32");
    auto contribution =
        op("mul", slice(2, {"%g"}), op("sub", pos, neg, "f32"), "f32");
    s << "scf.yield " << contribution.id << ":f32\n}else{scf.yield " << fz.id
      << ":f32\n}\n";
    auto acc = op("add", {"%a", "f32"}, {"%record", "f32"}, "f32");
    s << "scf.yield " << acc.id << ":f32\n}\n";
    writeRecord({"%acc", "f32"}, 5);
    s << "}\n";
  }
  void nativeRecords() {
    auto zero = idx(0), one = idx(1), end = idx(p.shape[2] / 256),
         fz = lit("0.0", "f32");
    s << "scf.for %r=" << zero.id << " to %R step " << one.id
      << " {\n%sum=scf.for %g=" << zero.id << " to " << end.id << " step "
      << one.id << " iter_args(%acc=" << fz.id << ")->(f32){\n";
    auto w = slice(0, {"%r", "%g", "*"}), x = slice(1, {"%g", "*"});
    auto field = [&](Term v, StringRef name, StringRef element, int n = 1) {
      auto type = n == 1 ? element.str()
                         : "!weft_kernel.value<" + element.str() + ",[" +
                               std::to_string(n) + "],[3]>";
      return inst("field", {v}, type, ("name=\"" + name + "\"").str());
    };
    auto groups = Types{"i32", {8}, {3}}.type("value");
    auto d = convert(field(w, "d", "f16"), "f32"),
         dm = convert(field(w, "dmin", "f16"), "f32");
    auto nonzero = [&](Term t) {
      auto bits = inst("bitcast", {t}, "si32");
      auto mag = op("and", bits, lit("2147483647", "si32"), "si32");
      return inst("compare", {mag, lit("0", "si32")}, "i1", "predicate=\"ne\"");
    };
    auto live = op("or", nonzero(d), nonzero(dm), "i1");
    s << "%value=scf.if " << live.id << " -> (f32){\n";
    std::string spec = "\"subquant.1\",1,0,3,\"blocks\",\"exact\"";
    auto extent = idx(256), sub = idx(32);
    s << "%chunks=arith.ceildivui " << extent.id << "," << sub.id << ":index\n";
    auto dom = inst("domain", {root, extent, sub, {"%chunks", "index"}},
                    "!weft_kernel.domain<" + spec + ">");
    auto point = "!weft_kernel.point<<" + spec + ">>";
    auto wq = field(w, "q", "ui4", 256), xq = field(x, "q", "si8", 256),
         sc = field(w, "sc", "ui6", 8), mn = field(w, "m", "ui6", 8);
    auto sums = inst("fold2", {field(x, "bsum", "si16", 16)}, groups);
    auto iz = lit("0", "si32");
    s << "%parts:2=\"weft_kernel.level\"(" << dom.id << "," << iz.id << ","
      << iz.id << ") (";
    for (int region = 0; region < 2; ++region)
      s << "{^bb0(%birth" << region << ":" << point
        << "):\n\"weft_kernel.births_yield\"():()->()\n},";
    s << "{^bb0(%j:" << point << ",%aS:si32,%aT:si32):\n";
    auto extract = [&](Term v, StringRef type, bool sub) {
      return inst("extract", {v, {"%j", point}}, type,
                  sub ? "selectors=[\"domain\"]"
                      : "selectors=[\"group_index\"]");
    };
    auto u = extract(wq, Types{"ui4", {32}, {3}}.type("value"), true),
         v = extract(xq, Types{"i8", {32}, {3}}.type("value"), true);
    auto dotValue = dot(u, v, {"i32", {}, {}});
    auto S = op("add", {"%aS", "si32"},
                op("mul", dotValue,
                   inst("widen", {extract(sc, "ui6", false)}, "si32"), "si32"),
                "si32");
    auto T = op("add", {"%aT", "si32"},
                op("mul", extract(sums, "si32", false),
                   inst("widen", {extract(mn, "ui6", false)}, "si32"), "si32"),
                "si32");
    inst("handoff", {S, T}, "()");
    s << "}):(" << dom.type << ",si32,si32)->(si32,si32)\n";
    auto pos = op("mul", d, convert({"%parts#0", "si32"}, "f32"), "f32"),
         neg = op("mul", dm, convert({"%parts#1", "si32"}, "f32"), "f32");
    auto vrec =
        op("mul", field(x, "ds", "f32"), op("sub", pos, neg, "f32"), "f32");
    s << "scf.yield " << vrec.id << ":f32\n}else{scf.yield " << fz.id
      << ":f32\n}\n";
    auto acc = op("add", {"%acc", "f32"}, {"%value", "f32"}, "f32");
    s << "scf.yield " << acc.id << ":f32\n}\n";
    writeRecord({"%sum", "f32"}, 2);
    s << "}\n";
  }
};
} // namespace
llvm::Error emitMatrix(func::FuncOp fn, llvm::raw_pwrite_stream &out) {
  MatrixPanels p(fn);
  Writer w{p};
  w.signatures();
  w.begin();
  if (w.emptyReduction())
    w.publishIdentity();
  else if (p.dot.getName().stripDialect() == "q4k_dot")
    w.records();
  else
    w.ordinary();
  w.s << "\"weft_kernel.return\"():()->()\n}):()->()\n}\n";
  SmallVector<std::string> args, steps;
  llvm::json::Array slots, ranks, compact;
  for (auto [i, v] : llvm::enumerate(w.slots)) {
    args.push_back(v.address);
    for (auto x : v.strides)
      steps.push_back(std::to_string(x));
    if (v.strides.empty())
      compact.push_back(i);
    bool record = v.t.carrier == "Q4_K" || v.t.carrier == "Q8_K";
    ranks.push_back(int64_t(v.t.dims.size()));
    slots.push_back(llvm::json::Object{
        {"dtype", record ? "u8" : v.t.carrier},
        {"direction", v.writes ? "Out" : "In"},
        {"encoding", record ? v.t.carrier : "dense." + v.t.carrier}});
  }
  args.push_back("R");
  args.push_back("C");
  llvm::append_range(args, steps);
  auto abi = llvm::json::parse(fn->getAttrOfType<StringAttr>("kir.abi"));
  if (!abi)
    return abi.takeError();
  out << llvm::json::Value(llvm::json::Object{
      {"version", 3},
      {"task", w.text},
      {"native_symbol", "matrix_panel_instance"},
      {"task_slots", std::move(slots)},
      {"task_ranks", std::move(ranks)},
      {"contiguous_slots", std::move(compact)},
      {"shape_parameters", llvm::json::Array{"R", "C"}},
      {"host",
       p.dispatch("#include \"task_bridge.h\"",
                  "matrix_panel_instance(" + llvm::join(args, ",") + ");",
                  true)},
      {"domain", intentdsl::publishDomain(fn)},
      {"abi", std::move(*abi)},
      {"rows", llvm::json::Array{llvm::json::Object{{"grain", p.tile[0]},
                                                    {"columns", p.tile[1]}}}}});
  return llvm::Error::success();
}
} // namespace CI_UNIT_NAMESPACE
