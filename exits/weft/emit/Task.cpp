#include "intentdsl/levels/execution/cpu/ir/Task.h"
#include "llvm/ADT/StringExtras.h"
using namespace mlir;
namespace CI_UNIT_NAMESPACE {
namespace {
std::string printType(Type t) {
  std::string s;
  llvm::raw_string_ostream out(s);
  out << t;
  return s;
}
std::string regionType(StringRef kind, StringRef element, int rank,
                       int columnDim = -2) {
  std::string shape =
      rank == 2 ? "[-1,-2],[1,2]" : "[" + std::to_string(columnDim) + "],[2]";
  if (kind == "value")
    return "!weft_kernel.value<" + element.str() + "," + shape + ">";
  return "!weft_kernel." + kind.str() + "<!weft_kernel.encoding<\"" +
         element.str() + "\",\"dense\",\"dense." + element.str() + "\",[]>," +
         shape + ">";
}
struct Term {
  std::string id, type;
};
class CanonicalTask {
  llvm::raw_ostream &s;
  llvm::DenseMap<Value, Term> terms;
  unsigned serial = 0;
  Term inst(StringRef op, ArrayRef<Term> args, std::string type,
            std::string props = "") {
    std::string id = "%v" + std::to_string(serial++);
    if (type != "()")
      s << id << " = ";
    s << "\"weft_kernel." << op << "\"(";
    llvm::interleaveComma(args, s, [&](auto &a) { s << a.id; });
    s << ") ";
    if (!props.empty())
      s << "<{" << props << "}> ";
    s << ": (";
    llvm::interleaveComma(args, s, [&](auto &a) { s << a.type; });
    s << ") -> " << type << "\n";
    return {id, type};
  }

  llvm::Error writeBody(Block &body, StringRef start, Term active,
                        int dim = -2) {
    for (auto &op : body.without_terminator()) {
      auto code = op.getName().stripDialect();
      SmallVector<Term> args;
      if (code == "read" || code == "write" || code == "read_panel" ||
          code == "write_panel") {
        auto view = op.getOperand(0);
        auto vt = cast<MemRefType>(view.getType());
        std::string e = printType(vt.getElementType());
        args.push_back(terms.lookup(view));
        if (vt.getRank() == 2)
          args.push_back({"%r", "index"});
        auto slice = inst("slice", args, regionType("slice", e, 1),
                          vt.getRank() == 2 ? "selectors=[\"index\",\"all\"]"
                                            : "selectors=[\"all\"]");
        auto window =
            start.empty()
                ? slice
                : (dim == 1
                       ? inst("subview", {slice, {start.str(), "index"}},
                              regionType("slice", e, 1, 1),
                              "offsets=array<i64:-1>,extents=array<i64:1>")
                       : inst("subview",
                              {slice, {start.str(), "index"}, active},
                              regionType("slice", e, 1, dim),
                              "offsets=array<i64:-1>,extents=array<i64:-1>"));
        if (code == "write_panel" || code == "write")
          inst("commit",
               {terms.lookup(op.getOperand(op.getNumOperands() - 1)), window},
               "()");
        else
          terms[op.getResult(0)] =
              inst("admit", {window}, regionType("value", e, 1, dim));
        continue;
      }
      for (auto operand : op.getOperands())
        args.push_back(terms.lookup(operand));
      auto v = op.getResult(0);
      auto t = v.getType();
      std::string result =
          isa<RankedTensorType>(t)
              ? regionType(
                    "value",
                    printType(cast<RankedTensorType>(t).getElementType()), 1,
                    dim)
              : printType(t);
      std::string kind = code.str(), operation;
      if (code == "max_num" || code == "sum") {
        operation = "reduce";
        kind = code == "max_num" ? "max" : "add";
      } else if (code == "cast")
        operation = "cast";
      else if (code == "rsqrt" || code == "exp")
        operation = "unary";
      else if (code == "mul" || code == "add" || code == "div" || code == "sub")
        operation = "binary";
      else
        return llvm::createStringError("unsupported Weft task operation " +
                                       kind);
      auto p = operation == "cast" ? "" : "kind=\"" + kind + "\"";
      if (operation == "reduce")
        p += ",axis=0:i64";
      terms[v] = inst(operation, args, result, p);
    }
    return llvm::Error::success();
  }

public:
  explicit CanonicalTask(llvm::raw_ostream &s) : s(s) {}
  llvm::Error write(Operation *task, const llvm::json::Array &abi) {
    SmallVector<std::string> names, effects, aliases, strided, params;
    auto &bb = task->getRegion(0).front();
    for (auto [i, arg] : llvm::enumerate(bb.getArguments().drop_front())) {
      auto a = abi[i].getAsObject();
      auto n = "a" + std::to_string(i);
      auto elem = a->getString("dtype")->str();
      int rank = a->getArray("dims")->size();
      auto t = rank ? regionType("view", elem, rank) : elem;
      terms[arg] = {"%" + n, t};
      names.push_back("\"" + n + "\"");
      params.push_back("%" + n + ": " + t);
      effects.push_back(
          rank ? (a->getString("direction") == "Out" ? "\"write\"" : "\"read\"")
               : "\"none\"");
      aliases.push_back(rank ? "0" : "-1");
      if (rank)
        strided.push_back(names.back());
    }
    s << "module {\n\"weft_kernel.kernel\"() <{sym_name=\""
      << task->getAttrOfType<StringAttr>("task_symbol").getValue()
      << "\",source=\"CPU row interval\",arg_names=[" << llvm::join(names, ",")
      << "],arg_access=[" << llvm::join(effects, ",")
      << "],arg_alias_sets=array<i64:" << llvm::join(aliases, ",")
      << ">,shape_symbols=[\"T\",\"N\",\"C\"],strided_args=["
      << llvm::join(strided, ",") << "]}> ({\n^bb0(" << llvm::join(params, ",")
      << "):\n";
    auto extent = inst("symbol", {}, "index",
                       "choices=array<i64>,kind=\"shape\",name=\"T\"");
    auto columns = inst("symbol", {}, "index",
                        "choices=array<i64>,kind=\"shape\",name=\"N\"");
    auto panelWidth = inst("symbol", {}, "index",
                           "choices=array<i64>,kind=\"shape\",name=\"C\"");
    inst("root_domain", {},
         "!weft_kernel.domain<\"root\",0,-1,0,\"root\",\"exact\">");
    s << "%lo=arith.constant 0 : index\n%step=arith.constant 1 : index\n";
    bool whole = cast<DictionaryAttr>(task->getAttrOfType<ArrayAttr>("rows")[0])
                     .getAs<IntegerAttr>("columns")
                     .getInt() == 0;
    Term panelCount, tailStart;
    if (!whole) {
      panelCount =
          inst("binary", {columns, panelWidth}, "index", "kind=\"div\"");
      tailStart =
          inst("binary", {panelCount, panelWidth}, "index", "kind=\"mul\"");
    }
    s << "scf.for %r = %lo to " << extent.id << " step %step {\n";
    if (whole) {
      if (auto e = writeBody(bb, {}, {}))
        return e;
    } else
      for (auto &sweep : bb.without_terminator()) {
        auto combine = sweep.getAttrOfType<StringAttr>("combine");
        bool store = combine == "store";
        std::string previous;
        if (!store) {
          previous = "%neutral" + std::to_string(serial++);
          s << previous << "=arith.constant "
            << (combine == "sum" ? "0.0" : "0xFF800000") << " : f32\n";
        }
        auto &body = sweep.getRegion(0).front();
        for (bool tail : {false, true}) {
          auto id = std::to_string(serial++), induction = "%index" + id,
               acc = "%acc" + id, summary = "%summary" + id;
          if (!store)
            s << summary << " = ";
          s << "scf.for " << induction << " = " << (tail ? tailStart.id : "%lo")
            << " to " << (tail ? columns.id : panelCount.id) << " step %step";
          if (!store)
            s << " iter_args(" << acc << "=" << previous << ") -> (f32)";
          s << " {\n";
          auto offset = tail
                            ? Term{induction, "index"}
                            : inst("binary", {{induction, "index"}, panelWidth},
                                   "index", "kind=\"mul\"");
          if (auto e = writeBody(body, offset.id, panelWidth, tail ? 1 : -3))
            return e;
          if (!store) {
            auto partial = terms.lookup(body.back().getOperand(0));
            auto next =
                inst("binary", {{acc, "f32"}, partial}, "f32",
                     combine == "sum" ? "kind=\"add\"" : "kind=\"max\"");
            s << "scf.yield " << next.id << " : f32\n";
            previous = summary;
          }
          s << "}\n";
        }
        if (!store)
          terms[sweep.getResult(0)] = {previous, "f32"};
      }
    s << "}\n\"weft_kernel.return\"():()->()\n}):()->()\n}\n";
    return llvm::Error::success();
  }
};
} // namespace
llvm::Expected<std::string> taskSource(Operation *op,
                                       const llvm::json::Array &abi) {
  std::string s;
  llvm::raw_string_ostream out(s);
  if (auto e = CanonicalTask(out).write(op, abi))
    return e;
  return s;
}

std::string hostSource(Operation *dispatch, const llvm::json::Array &abi) {
  auto slots =
      dispatch->getAttrOfType<DenseI64ArrayAttr>("weft_host.slot_ranks");
  auto symbol = dispatch->getAttrOfType<StringAttr>("task_symbol").str();
  SmallVector<std::string> params, args, strides;
  for (auto [i, entry] : llvm::enumerate(abi)) {
    auto a = entry.getAsObject();
    auto n = "a" + std::to_string(i);
    auto rank = slots[i];
    auto type = a->getString("dtype") == "f16" ? "_Float16" : "float";
    params.push_back(std::string(type) + (rank ? " *" : " ") + n);
    for (size_t axis = 0; axis < rank; ++axis) {
      auto stride = n + "s" + std::to_string(axis);
      params.push_back("int64_t " + stride);
      strides.push_back(stride);
    }
    args.push_back(n + (rank == 2 ? " + start*" + n + "s0" : ""));
  }
  args.push_back("count");
  args.push_back("cols");
  args.push_back("panel ? panel : cols");
  args.append(strides);
  return "#include <stdint.h>\n#include \"task_bridge.h\"\nvoid intent_entry(" +
         llvm::join(params, ",") +
         ", int64_t rows, int64_t cols, int64_t grain, int64_t panel, int64_t "
         "workers){\nif "
         "(!cols) return;\n#pragma omp parallel for num_threads(workers) "
         "schedule(static) if(workers>1)\n"
         "for(int64_t start=0;start<rows;start+=grain){int64_t "
         "count=rows-start<grain?rows-start:grain; " +
         symbol + "_instance(" + llvm::join(args, ",") + ");}\n}\n";
}
} // namespace CI_UNIT_NAMESPACE
