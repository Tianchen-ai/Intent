#include "intentdsl/levels/execution/cpu/ir/Task.h"
#include "llvm/ADT/StringExtras.h"
using namespace mlir;
namespace CI_UNIT_NAMESPACE {
namespace {
class Source {
  llvm::raw_ostream &out;
  llvm::DenseMap<Value, std::string> vars;
  int next = 0;
  unsigned depth = 0;
  void line(unsigned n, const std::string &s) {
    out.indent((n + depth) * 4) << s << "\n";
  }
  std::string var(Value v) { return vars.lookup(v); }
  std::string temp() { return "v" + std::to_string(next++); }
  std::string dtype(Type t) {
    if (auto v = dyn_cast<VectorType>(t))
      t = v.getElementType();
    return t.isF16() ? "float16" : "float32";
  }

public:
  explicit Source(llvm::raw_ostream &stream) : out(stream) {}
  llvm::Error write(Operation *task) {
    SmallVector<std::string> params;
    auto &body = task->getRegion(0).front();
    vars[body.getArgument(0)] = "r";
    for (auto [i, v] : llvm::enumerate(body.getArguments().drop_front())) {
      std::string n = "a" + std::to_string(i);
      vars[v] = n;
      if (auto m = dyn_cast<MemRefType>(v.getType())) {
        params.push_back(n + ": Pointer[Scalar[DType." +
                         dtype(m.getElementType()) + "], MutUntrackedOrigin]");
        for (int j = 0; j < m.getRank(); ++j)
          params.push_back(n + "s" + std::to_string(j) + ": Int64");
      } else
        params.push_back(n + ": Float32");
    }
    out << "from std.math import exp, rsqrt, "
           "inf\n@export(\"intent_row_interval\")\ndef intent_row_interval("
        << llvm::join(params, ",")
        << ", first: Int64, stop: Int64, cols: Int64, columns_per_block: "
           "Int64) abi(\"C\"):\n";
    line(1, "for r in range(Int(first),Int(stop)):");
    auto fullWidth =
        task->getAttrOfType<IntegerAttr>("mojo_row.lanes").getInt();
    for (auto &loop : body.without_terminator()) {
      if (loop.getName().getStringRef() != "mojo_row.sweep")
        return llvm::createStringError("Mojo requires realized sweeps");
      bool tail = loop.getAttrOfType<StringAttr>("range") == "tail";
      auto kind = loop.getAttrOfType<StringAttr>("algebra");
      bool sum = kind == "sum", reduction = kind != "store";
      auto lanes =
          std::to_string(loop.getAttrOfType<IntegerAttr>("width").getInt());
      std::string result = temp(), acc = result + "_acc";
      if (reduction)
        line(2, "var " + acc + " = SIMD[DType.float32, " + lanes + "](" +
                    (sum ? "0.0" : "-inf[DType.float32]()") + ")");
      std::string stop = "Int(cols)//" + std::to_string(fullWidth) + "*" +
                         std::to_string(fullWidth);
      if (!tail) {
        line(2, "for base in range(0, " + stop + ", Int(columns_per_block)):");
        depth = 1;
      }
      line(2, "for j in range(" +
                  (tail ? stop + ", Int(cols), 1"
                        : "base, min(base + Int(columns_per_block), " + stop +
                              "), " + lanes) +
                  "):");
      auto &bb = loop.getRegion(0).front();
      vars[bb.getArgument(0)] = "j";
      for (auto &op : bb) {
        auto code = op.getName().stripDialect();
        auto arg = [&](unsigned i) { return var(op.getOperand(i)); };
        auto lhs = temp();
        if (code == "yield") {
          if (reduction)
            line(3, acc + " = " +
                        (sum ? acc + " + " + arg(0)
                             : "max(" + acc + "," + arg(0) + ")"));
        } else if (code == "read" || code == "write") {
          auto view = cast<MemRefType>(op.getOperand(0).getType());
          bool matrix = view.getRank() == 2;
          auto a = arg(0),
               stride = "Int(" + a + "s" + (matrix ? "1" : "0") + ")";
          auto offset = (matrix ? "r*Int(" + a + "s0)+" : "") +
                        std::string("j*") + stride;
          if (code == "read") {
            line(3, "var " + lhs + " = SIMD[DType." +
                        dtype(view.getElementType()) + "," + lanes + "](0)");
            line(3, "if " + stride + " == 1:");
            line(4, lhs + " = " + a + ".unsafe_offset(" + offset +
                        ").unsafe_load[width=" + lanes + "]()");
            line(3, "else:");
            line(4, "for lane in range(" + lanes + "):");
            line(5, lhs + "[lane] = " + a + "[unsafe_offset=" + offset +
                        "+lane*" + stride + "]");
          } else {
            line(3, "if " + stride + " == 1:");
            line(4, a + ".unsafe_offset(" + offset + ").unsafe_store(" +
                        arg(3) + ")");
            line(3, "else:");
            line(4, "for lane in range(" + lanes + "):");
            line(5, a + "[unsafe_offset=" + offset + "+lane*" + stride +
                        "] = " + arg(3) + "[lane]");
          }
        } else if (code == "apply") {
          auto opKind = op.getAttrOfType<StringAttr>("kind").getValue();
          std::string expr;
          if (opKind == "cast")
            expr = arg(0) + ".cast[DType." + dtype(op.getResult(0).getType()) +
                   "]()";
          else if (opKind == "exp" || opKind == "rsqrt")
            expr = opKind.str() + "(" + arg(0) + ")";
          else {
            auto token = llvm::StringSwitch<StringRef>(opKind)
                             .Case("add", "+")
                             .Case("sub", "-")
                             .Case("mul", "*")
                             .Case("div", "/")
                             .Default("");
            if (token.empty())
              return llvm::createStringError("unsupported Mojo expression");
            auto expand = [&](unsigned i) {
              return isa<VectorType>(op.getResult(0).getType()) &&
                             !isa<VectorType>(op.getOperand(i).getType())
                         ? "SIMD[DType." + dtype(op.getOperand(i).getType()) +
                               "," + lanes + "](" + arg(i) + ")"
                         : arg(i);
            };
            expr = expand(0) + " " + token.str() + " " + expand(1);
          }
          line(3, "var " + lhs + " = " + expr);
        } else
          return llvm::createStringError("unrecognized Mojo sweep instruction");
        if (op.getNumResults())
          vars[op.getResult(0)] = lhs;
      }
      depth = 0;
      if (reduction) {
        auto value = acc + (sum ? ".reduce_add()" : ".reduce_max()");
        if (loop.getNumOperands())
          value = sum ? var(loop.getOperand(0)) + " + " + value
                      : "max(" + var(loop.getOperand(0)) + "," + value + ")";
        line(2, "var " + result + " = " + value);
        vars[loop.getResult(0)] = result;
      }
    }

    return llvm::Error::success();
  }
};
} // namespace
llvm::Expected<std::string> mojoSource(Operation *task) {
  std::string s;
  llvm::raw_string_ostream out(s);
  if (auto e = Source(out).write(task))
    return e;
  return s;
}
} // namespace CI_UNIT_NAMESPACE
