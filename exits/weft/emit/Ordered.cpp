#include "../legalize/MemberStore.h"
#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "intentdsl/levels/execution/cpu/ir/Owners.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/StringExtras.h"
using namespace mlir;
namespace CI_UNIT_NAMESPACE {
namespace {
struct Term {
  std::string id, type;
};
struct Ordered {
  func::FuncOp fn;
  Operation &space;
  llvm::DenseMap<Value, Term> values;
  Term scoreCache, scoreMask, scoreZero;
  llvm::DenseMap<Value, SmallVector<int64_t>> axes;
  std::string text;
  llvm::raw_string_ostream s{text};
  unsigned counter = 0;
  explicit Ordered(func::FuncOp f)
      : fn(f), space(f.front().front().getRegion(0).front().front()) {}
  std::string fresh() { return "%v" + std::to_string(counter++); }
  static std::string type(Type t) {
    return t.isIndex()      ? "index"
           : t.isF32()      ? "f32"
           : t.isF16()      ? "f16"
           : t.isBF16()     ? "bf16"
           : t.isInteger(1) ? "i1"
                            : "si32";
  }
  static std::string carrier(Type t) {
    return t.isInteger(32) ? "i32" : type(t);
  }
  int64_t dim(Value v, StringRef key, unsigned i) {
    return fn.getArgAttrOfType<DenseI64ArrayAttr>(
        cast<BlockArgument>(v).getArgNumber(), key)[i];
  }
  Term inst(StringRef op, ArrayRef<Term> args, std::string ty,
            std::string attrs = {}) {
    SmallVector<std::string> ids, types;
    for (auto &a : args) {
      ids.push_back(a.id);
      types.push_back(a.type);
    }
    auto id = fresh();
    if (ty != "()")
      s << id << "=";
    s << "\"weft_kernel." << op << "\"(" << llvm::join(ids, ",") << ")";
    if (!attrs.empty())
      s << " <{" << attrs << "}>";
    s << ":(" << llvm::join(types, ",") << ")->" << ty << "\n";
    return {id, ty};
  }
  Term binary(StringRef op, Term a, Term b) {
    return inst("binary", {a, b}, a.type, "kind=\"" + op.str() + "\"");
  }
  Term idx(int64_t n) {
    auto id = fresh();
    s << id << "=arith.constant " << n << ":index\n";
    return {id, "index"};
  }
  std::string vectorType(Value v) {
    auto [base, axis] = *intentdsl::kir::memberAxis(v);
    return "!weft_kernel.value<" +
           type(cast<ShapedType>(v.getType()).getElementType()) + ",[" +
           std::to_string(dim(base, "cpu.shape", axis)) + "],[" +
           std::to_string(axes.at(base)[axis]) + "]>";
  }
  Term slice(Value v, ValueRange coords, int member = -1) {
    auto elem = carrier(cast<MemRefType>(v.getType()).getElementType());
    SmallVector<Term> args{values.at(v)};
    SmallVector<std::string> axes;
    for (auto j : coords) {
      args.push_back(values.at(j));
      axes.push_back("\"index\"");
    }
    std::string extent, axis;
    if (member >= 0) {
      axes.insert(axes.begin() + member, "\"all\"");
      extent = std::to_string(dim(v, "cpu.shape", member));
      axis = std::to_string(this->axes.at(v)[member]);
    }
    return inst("slice", args,
                "!weft_kernel.slice<!weft_kernel.encoding<\"" + elem +
                    "\",\"dense\",\"dense." + elem + "\",[]>,[" + extent +
                    "],[" + axis + "]>",
                "selectors=[" + llvm::join(axes, ",") + "]");
  }
  llvm::Error region(Block &body) {
    for (auto &op : body) {
      auto n = op.getName().getStringRef();
      if (op.getName().stripDialect() == "end")
        continue;
      if (n == "scf.yield") {
        SmallVector<std::string> ys, ts;
        for (auto v : op.getOperands()) {
          ys.push_back(values.at(v).id);
          ts.push_back(values.at(v).type);
        }
        if (!ys.empty())
          s << "scf.yield " << llvm::join(ys, ",") << ":" << llvm::join(ts, ",")
            << "\n";
        continue;
      }
      auto val = [&](unsigned i) { return values.at(op.getOperand(i)); };
      Term result;
      if (op.hasAttr("cpu.replay_score")) {
        auto iv = values.at(op.getParentOfType<scf::ForOp>().getInductionVar());
        values[op.getResult(0)] =
            inst("pointer_load",
                 {scoreCache, inst("cast", {iv}, "si64"), scoreMask, scoreZero},
                 "f32");
        continue;
      }
      if (auto loop = dyn_cast<scf::ForOp>(op)) {
        SmallVector<std::string> outs, bindings, types;
        auto iv = fresh();
        values[loop.getInductionVar()] = {iv, "index"};
        for (auto [ret, init, arg] :
             llvm::zip(loop.getResults(), loop.getInitArgs(),
                       loop.getRegionIterArgs())) {
          auto t = values.at(init);
          Term r{fresh(), t.type}, a{fresh(), t.type};
          values[ret] = r;
          values[arg] = a;
          outs.push_back(r.id);
          bindings.push_back(a.id + "=" + t.id);
          types.push_back(t.type);
        }
        if (!outs.empty())
          s << llvm::join(outs, ",") << "=";
        s << "scf.for " << iv << "=" << val(0).id << " to " << val(1).id
          << " step " << val(2).id;
        if (!outs.empty())
          s << " iter_args(" << llvm::join(bindings, ",") << ")->("
            << llvm::join(types, ",") << ")";
        s << "{\n";
        if (auto e = region(*loop.getBody()))
          return e;
        s << "}\n";
        continue;
      }
      if (auto branch = dyn_cast<scf::IfOp>(op)) {
        SmallVector<std::string> outs, types;
        for (auto v : op.getResults()) {
          Term a{fresh(), type(v.getType())};
          values[v] = a;
          outs.push_back(a.id);
          types.push_back(a.type);
        }
        s << llvm::join(outs, ",") << "=scf.if " << val(0).id << " -> ("
          << llvm::join(types, ",") << "){\n";
        if (auto e = region(*branch.thenBlock()))
          return e;
        s << "}else{\n";
        if (auto e = region(*branch.elseBlock()))
          return e;
        s << "}\n";
        continue;
      }
      auto kind = op.getName().stripDialect();
      auto narrowed = [](Operation *o) {
        return isa<arith::TruncFOp>(o) && o->hasOneUse() &&
               (*o->getUsers().begin())->hasAttr("weft_host.member_store");
      };
      auto multiplied = [&](Operation *o) {
        return isa<arith::MulFOp>(o) && o->hasOneUse() &&
               narrowed(*o->getUsers().begin());
      };
      if (narrowed(&op) || multiplied(&op))
        continue;
      if (kind == "slice_read")
        result = inst("admit",
                      {slice(op.getOperand(0), op.getOperands().drop_front(),
                             op.getAttrOfType<IntegerAttr>("axis").getInt())},
                      vectorType(op.getResult(0)));
      else if (kind == "slice_write") {
        if (op.hasAttr("weft_host.member_store")) {
          auto plan = *scaledMemberStore(&op);
          auto lo = idx(0),
               hi = idx(
                   intentdsl::kir::memberExtent(fn, plan.vector, "cpu.shape")),
               step = idx(1);
          auto iv = fresh();
          s << "scf.for " << iv << "=" << lo.id << " to " << hi.id << " step "
            << step.id << " {\n";
          auto elem = inst("extract", {values.at(plan.vector), {iv, "index"}},
                           "f32", "selectors=[\"index\"]");
          auto scaled = binary("mul", elem, values.at(plan.scale));
          auto base = op.getOperand(1);
          auto t = type(cast<MemRefType>(base.getType()).getElementType());
          auto dst = inst("cast", {scaled}, t);
          auto slot = op.getAttrOfType<IntegerAttr>("axis").getInt();
          SmallVector<Term> at{values.at(base)};
          for (auto c : op.getOperands().drop_front(2))
            at.push_back(values.at(c));
          at.insert(at.begin() + 1 + slot, {iv, "index"});
          auto enc = carrier(cast<MemRefType>(base.getType()).getElementType());
          SmallVector<std::string> selectors(at.size() - 1, "\"index\"");
          auto addr = inst("slice", at,
                           "!weft_kernel.slice<!weft_kernel.encoding<\"" + enc +
                               "\",\"dense\",\"dense." + enc + "\",[]>,[],[]>",
                           "selectors=[" + llvm::join(selectors, ",") + "]");
          inst("commit", {dst, addr}, "()");
          s << "}\n";
          continue;
        }
        inst("commit",
             {val(0), slice(op.getOperand(1), op.getOperands().drop_front(2),
                            op.getAttrOfType<IntegerAttr>("axis").getInt())},
             "()");
        continue;
      } else if (kind == "lane_splat")
        result = inst("new", {val(0)}, vectorType(op.getResult(0)),
                      "initialized=true");
      else if (kind == "lane_sum")
        result = inst("reduce", {val(0)}, "f32", "kind=\"add\",axis=0:i64");
      else if (auto load = dyn_cast<memref::LoadOp>(op))
        result = inst("admit", {slice(load.getMemref(), load.getIndices())},
                      type(load.getType()));
      else if (auto store = dyn_cast<memref::StoreOp>(op)) {
        inst("commit",
             {values.at(store.getValue()),
              slice(store.getMemref(), store.getIndices())},
             "()");
        continue;
      } else if (auto d = dyn_cast<memref::DimOp>(op))
        result = idx(dim(d.getSource(), "cpu.shape", *d.getConstantIndex()));
      else if (auto c = dyn_cast<arith::ConstantOp>(op)) {
        if (c.getType().isIndex())
          result = idx(cast<IntegerAttr>(c.getValue()).getInt());
        else {
          std::string literal;
          llvm::raw_string_ostream(literal) << c.getValue();
          result = inst("constant", {}, type(c.getType()), "value=" + literal);
        }
      } else if (n == "arith.extf" || n == "arith.truncf" ||
                 n == "arith.index_cast")
        result = inst("cast", {val(0)},
                      isa<RankedTensorType>(op.getResult(0).getType())
                          ? vectorType(op.getResult(0))
                          : type(op.getResult(0).getType()));
      else if (n == "arith.maxnumf")
        result = binary("max", val(0), val(1));
      else if (n == "math.exp" || n == "math.sqrt" || n == "math.exp2" ||
               n == "math.log")
        result = inst("unary", {val(0)}, val(0).type,
                      "kind=\"" + op.getName().stripDialect().str() + "\"");
      else if (auto cmp = dyn_cast<arith::CmpIOp>(op)) {
        auto a = val(0), b = val(1);
        auto p = cmp.getPredicate();
        if (a.type == "index" && p != arith::CmpIPredicate::eq) {
          a = inst("cast", {a}, "si64");
          b = inst("cast", {b}, "si64");
        }
        result = inst("compare", {a, b}, "i1",
                      std::string("predicate=\"") +
                          (p == arith::CmpIPredicate::sge   ? "ge"
                           : p == arith::CmpIPredicate::slt ? "lt"
                                                            : "eq") +
                          "\"");
      } else {
        auto tok = llvm::StringSwitch<StringRef>(n)
                       .Cases("arith.addf", "arith.addi", "add")
                       .Cases("arith.subf", "arith.subi", "sub")
                       .Cases("arith.mulf", "arith.muli", "mul")
                       .Cases("arith.divf", "arith.floordivsi", "div")
                       .Case("arith.remsi", "mod")
                       .Default("");
        if (tok.empty())
          return llvm::createStringError("unhandled ordered Weft instruction " +
                                         n);
        result = binary(tok, val(0), val(1));
      }
      values[op.getResult(0)] = result;
      if (op.hasAttr("cpu.record_score")) {
        auto iv = values.at(op.getParentOfType<scf::ForOp>().getInductionVar());
        inst("pointer_store",
             {scoreCache, inst("cast", {iv}, "si64"), scoreMask, result}, "()");
      }
    }
    return llvm::Error::success();
  }
};
} // namespace
llvm::Error emitOrdered(func::FuncOp fn, llvm::raw_pwrite_stream &output) {
  Ordered w(fn);
  auto abi = llvm::json::parse(fn->getAttrOfType<StringAttr>("kir.abi"));
  if (!abi)
    return abi.takeError();
  SmallVector<std::string> names, access, alias, params, strided, bridge, steps;
  llvm::json::Array ranks;
  bool emptyExtent = false;
  std::map<std::pair<std::string, unsigned>, int64_t> dimNames;
  for (auto [i, v] : llvm::enumerate(fn.getArguments())) {
    auto a = "a" + std::to_string(i);
    names.push_back("\"" + a + "\"");
    bridge.push_back(a);
    auto view = dyn_cast<MemRefType>(v.getType());
    auto rank = view ? view.getRank() : 0;
    auto elem = view ? Ordered::carrier(view.getElementType())
                     : Ordered::type(v.getType());
    auto dir = (*abi->getAsArray())[i].getAsObject()->getString("direction");
    alias.push_back(rank ? "0" : "-1");
    access.push_back(!rank            ? "\"none\""
                     : dir == "Out"   ? "\"write\""
                     : dir == "InOut" ? "\"readwrite\""
                                      : "\"read\"");
    auto ty = elem;
    ranks.push_back(rank);
    if (rank) {
      SmallVector<std::string> shape, axes;
      std::map<std::string, unsigned> occurrences;
      auto labels = (*abi->getAsArray())[i].getAsObject()->getArray("dims");
      for (int j = 0; j < rank; ++j) {
        std::string label;
        llvm::raw_string_ostream(label) << (*labels)[j];
        auto [it, added] = dimNames.try_emplace({label, occurrences[label]++},
                                                dimNames.size() + 1);
        w.axes[v].push_back(it->second);
        auto extent = w.dim(v, "cpu.shape", j);
        // Weft represents an empty extent through an invocation symbol.
        // LO and HI occupy the first two shape slots.
        emptyExtent |= extent == 0;
        shape.push_back(extent ? std::to_string(extent) : "-3");
        axes.push_back(std::to_string(it->second));
        steps.push_back(a + "s" + std::to_string(j));
      }
      ty = "!weft_kernel.view<!weft_kernel.encoding<\"" + elem +
           "\",\"dense\",\"dense." + elem + "\",[]>,[" +
           llvm::join(shape, ",") + "],[" + llvm::join(axes, ",") + "]>";
      strided.push_back(names.back());
    }
    w.values[v] = {"%" + a, ty};
    params.push_back("%" + a + ":" + ty);
  }
  w.s << "module {\n\"weft_kernel.kernel\"() "
         "<{sym_name=\"owner_task\",source=\"independent ordered "
         "owners\",arg_names=["
      << llvm::join(names, ",") << "],arg_access=[" << llvm::join(access, ",")
      << "],arg_alias_sets=array<i64:" << llvm::join(alias, ",")
      << ">,shape_symbols=[\"LO\",\"HI\"" << (emptyExtent ? ",\"EMPTY\"" : "")
      << "],strided_args=[" << llvm::join(strided, ",") << "]}> ({^bb0("
      << llvm::join(params, ",") << "):\n";
  SmallVector<StringRef> symbols{"LO", "HI"};
  if (emptyExtent)
    symbols.push_back("EMPTY");
  llvm::json::Array shapeParameters;
  for (auto key : symbols)
    shapeParameters.push_back(key.str());
  for (auto key : symbols)
    w.s << "%" << key
        << "=\"weft_kernel.symbol\"() "
           "<{choices=array<i64>,kind=\"shape\",name=\""
        << key << "\"}>:()->index\n";
  w.inst("root_domain", {},
         "!weft_kernel.domain<\"root\",0,-1,0,\"root\",\"exact\">");
  auto step = w.idx(1);
  if (auto k = w.space.getAttrOfType<IntegerAttr>("cpu.score_cache")) {
    w.scoreCache = w.inst(
        "local_alloc", {}, "!weft_kernel.pointer<f32>",
        "alignment=64:i64,elements=" + std::to_string(k.getInt()) + ":i64");
    w.scoreZero = w.inst("constant", {}, "f32", "value=0.0:f32");
    w.scoreMask = w.inst("compare", {step, step}, "i1", "predicate=\"eq\"");
  }
  w.s << "scf.for %owner=%LO to %HI step " << step.id << " {\n";
  auto &space = w.space;
  auto axes = space.getAttrOfType<DenseI64ArrayAttr>("axes");
  auto &bb = space.getRegion(0).front();
  Term cursor{"%owner", "index"};
  for (int i = axes.size() - 1; i >= 0; --i) {
    auto d = w.idx(
        std::max<int64_t>(1, w.dim(space.getOperand(i), "cpu.shape", axes[i])));
    w.values[bb.getArgument(i)] = w.binary("mod", cursor, d);
    cursor = w.binary("div", cursor, d);
  }
  if (auto err = w.region(bb))
    return err;
  w.s << "}\n\"weft_kernel.return\"():()->()\n}):()->()\n}\n";
  bridge.push_back("lo");
  bridge.push_back("hi");
  if (emptyExtent)
    bridge.push_back("0");
  llvm::append_range(bridge, steps);
  auto host = intentdsl::cpu::ownerDispatcher(
      fn, "#include \"task_bridge.h\"",
      "owner_task_instance(" + llvm::join(bridge, ",") + ")");
  output << llvm::json::Value(llvm::json::Object{
      {"version", 3},
      {"task", w.text},
      {"abi", std::move(*abi)},
      {"host", host},
      {"native_symbol", "owner_task_instance"},
      {"shape_parameters", std::move(shapeParameters)},
      {"task_ranks", std::move(ranks)},
      {"domain", intentdsl::publishDomain(fn)},
      {"rows",
       llvm::json::Array{llvm::json::Object{
           {"grain", space.getAttrOfType<IntegerAttr>("cpu.grain").getInt()},
           {"columns", 1}}}}});
  return llvm::Error::success();
}
} // namespace CI_UNIT_NAMESPACE
