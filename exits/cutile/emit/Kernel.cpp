#include "intentdsl/levels/algorithm/kir/ir/Literal.h"
#include "compilerinfra/Assembly.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
namespace CI_UNIT_NAMESPACE {
using namespace mlir;
namespace {
class Kernel {
  llvm::raw_ostream &out;
  llvm::DenseMap<Value, std::string> names;
  unsigned sequence = 0;
  std::string indexType;
  std::string local() { return "v" + std::to_string(sequence++); }
  void line(int depth, const std::string &s) {
    out.indent(depth * 4) << s << "\n";
  }
  llvm::Error body(Block &block, int depth) {
    for (auto &op : block) {
      auto code = op.getName().stripDialect();
      auto arg = [&](unsigned n) { return names.lookup(op.getOperand(n)); };
      auto algebra = op.getAttrOfType<StringAttr>("algebra");
      bool sum = algebra && algebra == "sum";
      if (code == "return")
        continue;
      if (code == "yield") {
        if (op.getNumOperands())
          line(depth, names.lookup(block.getArgument(1)) + " = " + arg(0));
        continue;
      }
      if (code == "tiles") {
        auto &inner = op.getRegion(0).front();
        auto i = local();
        names[inner.getArgument(0)] = i;
        if (op.getNumResults()) {
          auto carry = local();
          names[inner.getArgument(1)] = names[op.getResult(0)] = carry;
          line(depth, carry + " = " + arg(1));
        }
        line(depth,
             "for " + i + " in range(ct.cdiv(" + arg(0) + ".shape[-1], W)):");
        if (auto err = body(inner, depth + 1))
          return err;
        continue;
      }
      std::string expr;
      if (code == "linear_index")
        expr = "ct.astype(ct.expand_dims(" + arg(1) + ",1)*" + arg(0) +
               ".shape[1]+ct.expand_dims(" + arg(2) + ",0),ct.int32)";
      else if (code == "select")
        expr = "ct.where(" + arg(0) + "," + arg(1) + "," + arg(2) + ")";
      else if (code == "xor" || code == "and" || code == "shr" ||
               code == "shl" || code == "gt") {
        auto token = code == "xor"   ? " ^ "
                     : code == "and" ? " & "
                     : code == "shl" ? " << "
                     : code == "shr" ? " >> "
                                     : " > ";
        expr = arg(0) + token + arg(1);
      } else if (code == "constant")
        expr = intentdsl::kir::literal(op.getAttrOfType<TypedAttr>("value"));
      else if (code == "maximum")
        expr = "ct.maximum(" + arg(0) + "," + arg(1) + ",propagate_nan=True)";
      else if (code == "pid")
        expr = "ct.astype(ct.bid(0), " + indexType + ")";
      else if (code == "zero")
        expr = "0";
      else if (code == "rows")
        expr = arg(0) + " * R + ct.arange(R, dtype=" + indexType + ")";
      else if (code == "columns")
        expr = "ct.astype(" + arg(0) + ", " + indexType +
               ") * W + ct.arange(W, dtype=" + indexType + ")";
      else if (code == "active" &&
               cast<MemRefType>(op.getOperand(0).getType()).getRank() == 1)
        expr = "(ct.expand_dims(" + arg(1) + ",1)<1) & (ct.expand_dims(" +
               arg(2) + ",0)<" + arg(0) + ".shape[0])";
      else if (code == "active")
        expr = "(ct.expand_dims(" + arg(1) + ", 1) < " + arg(0) +
               ".shape[0]) & (ct.expand_dims(" + arg(2) + ", 0) < " + arg(0) +
               ".shape[1])";
      else if (code == "tile_load" || code == "tile_store") {
        bool matrix = op.getAttrOfType<BoolAttr>("matrix").getValue();
        std::string hint = op.getAttrOfType<BoolAttr>("allow_tma").getValue()
                               ? ""
                               : ", allow_tma=False";
        std::string index =
            matrix ? "(" + arg(1) + ", " + arg(2) + ")" : "(" + arg(2) + ",)";
        if (code == "tile_store") {
          auto axes = op.getAttrOfType<DenseI64ArrayAttr>("block.member_axes");
          bool swap = axes && axes.size() == 2 && axes[0] == 1;
          if (swap)
            index = "(" + arg(2) + "," + arg(1) + ")";
          line(depth, "ct.store(" + arg(0) + ", " + index + ", " +
                          (swap      ? "ct.transpose(" + arg(3) + ")"
                           : !matrix ? "ct.reshape(" + arg(3) + ",(W,))"
                                     : arg(3)) +
                          hint + ")");
          continue;
        }
        expr = "ct.load(" + arg(0) + ", " + index +
               ", shape=" + (matrix ? "(R,W)" : "(W,)") +
               ", padding_mode=ct.PaddingMode.ZERO" + hint + ")";
        if (!matrix)
          expr = "ct.broadcast_to(ct.expand_dims(" + expr + ",0),(R,W))";
      } else if (code == "reduce")
        expr = std::string(sum ? "ct.sum" : "ct.max") + "(ct.where(" + arg(1) +
               ", " + arg(0) + ", " + (sum ? "0.0" : "-float('inf')") +
               "), axis=1)";
      else if (code == "identity")
        expr = std::string("ct.full((R,), ") + (sum ? "0.0" : "-float('inf')") +
               ", ct.float32)";
      else if (code == "combine")
        expr = sum ? arg(0) + " + " + arg(1)
                   : "ct.maximum(" + arg(0) + ", " + arg(1) + ")";
      else if (code == "broadcast_rows")
        expr = "ct.expand_dims(" + arg(0) + ",1)";
      else if (code == "mul" || code == "sub" || code == "add")
        expr = arg(0) +
               (code == "mul"   ? " * "
                : code == "sub" ? " - "
                                : " + ") +
               arg(1);
      else if (code == "div")
        expr = "ct.truediv(" + arg(0) + ", " + arg(1) +
               ", rounding_mode=ct.RoundingMode.RN)";
      else if (code == "sqrt" || code == "exp" || code == "rsqrt")
        expr = "ct." + code.str() + "(" + arg(0) + ")";
      else if (code == "cast") {
        auto t = op.getResult(0).getType();
        if (auto f = dyn_cast<ShapedType>(t))
          t = f.getElementType();
        if (!t.isF32() && !t.isF16() && !t.isBF16() && !t.isInteger(32))
          return llvm::createStringError("unsupported cuTile conversion");
        expr = "ct.astype(" + arg(0) + ", ct." +
               std::string(t.isInteger(32) ? "int32"
                           : t.isBF16()    ? "bfloat16"
                           : t.isF16()     ? "float16"
                                           : "float32") +
               ")";
      } else
        return llvm::createStringError("unexpected cuTile operation " + code);
      auto id = local();
      names[op.getResult(0)] = id;
      line(depth, id + " = " + expr);
    }
    return llvm::Error::success();
  }

public:
  explicit Kernel(llvm::raw_ostream &out) : out(out) {}
  llvm::Error write(func::FuncOp fn) {
    bool wide =
        fn->getAttrOfType<IntegerAttr>("ct_src.index_bits").getInt() == 64;
    indexType = wide ? "ct.int64" : "ct.int32";
    out << "import cuda.tile as ct\n\n@ct.kernel\ndef kernel(";
    for (auto a : fn.getArguments()) {
      auto n = "a" + std::to_string(a.getArgNumber());
      names[a] = n;
      out << n;
      if (isa<MemRefType>(a.getType()))
        out << (wide ? ": ct.IndexedWithInt64[ct.Array]" : ": ct.Array");
      out << ", ";
    }
    out << "R: ct.Constant[int], W: ct.Constant[int]):\n";
    return body(fn.front(), 1);
  }
};
} // namespace
llvm::Error emit(ModuleOp m, llvm::raw_pwrite_stream &out) {
  auto fn = *m.getOps<func::FuncOp>().begin();
  auto abi = llvm::json::parse(fn->getAttrOfType<StringAttr>("kir.abi"));
  if (!abi)
    return abi.takeError();
  llvm::json::Array rows;
  for (auto a : fn->getAttrOfType<ArrayAttr>("block.configs")) {
    auto d = cast<DictionaryAttr>(a);
    auto hints = llvm::json::parse(d.getAs<StringAttr>("payload").getValue());
    if (!hints)
      return hints.takeError();
    rows.push_back(
        llvm::json::Object{{"id", d.getAs<IntegerAttr>("id").getInt()},
                           {"rows", d.getAs<IntegerAttr>("rows").getInt()},
                           {"width", d.getAs<IntegerAttr>("width").getInt()},
                           {"native", std::move(*hints)}});
  }
  llvm::json::Array formula;
  for (auto a : fn->getAttrOfType<ArrayAttr>("block.grid"))
    formula.push_back(cast<StringAttr>(a).str());
  auto guard = fn->getAttrOfType<DenseI64ArrayAttr>("block.guard");
  std::string src;
  llvm::raw_string_ostream stream(src);
  if (auto err = Kernel(stream).write(fn))
    return err;
  out << llvm::json::Value(llvm::json::Object{
      {"abi", std::move(*abi)},
      {"version", 1},
      {"source", src},
      {"entry", fn->getAttrOfType<StringAttr>("ct_src.entry").str()},
      {"grid", llvm::json::Array{std::move(formula), 1, 1}},
      {"rows", std::move(rows)},
      {"guard",
       llvm::json::Object{{"min_width", guard[0]},
                          {"max_width", guard[1] < 0
                                            ? llvm::json::Value(nullptr)
                                            : llvm::json::Value(guard[1])}}}});
  return llvm::Error::success();
}
} // namespace CI_UNIT_NAMESPACE
