#include "compilerinfra/Assembly.h"
#include "intentdsl/exits/triton/ir/Dialect.h"
#include "intentdsl/levels/execution/block/decisions/RowCapacity.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringMap.h"

namespace CI_UNIT_NAMESPACE {
using namespace mlir;
using namespace compilerinfra;
void legalize(OpPassManager &);
llvm::Expected<TargetProfile> capabilities(const llvm::json::Object &);
namespace {
struct BindCallable : PassWrapper<BindCallable, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BindCallable)
  std::optional<uint64_t> width;
  explicit BindCallable(std::optional<uint64_t> width) : width(width) {}
  void runOnOperation() final {
    if (!width) {
      getOperation().emitError("callable requires bound device resources");
      return signalPassFailure();
    }
    auto bound = *width;
    auto fn = *getOperation().getOps<func::FuncOp>().begin();
    Builder b(&getContext());
    fn->setAttr("triton_src.min_width",
                b.getI64IntegerAttr(
                    bound == 0 ? 8193 : (bound == 1 ? 0 : bound / 2 + 1)));
    fn->setAttr("triton_src.max_width", b.getI64IntegerAttr(bound));
  }
};
class Writer {
  llvm::raw_ostream &out;
  llvm::DenseMap<Value, std::string> values;
  unsigned next = 0;
  void line(unsigned depth, const std::string &s) {
    out.indent(4 * depth) << s << "\n";
  }
  std::string name(Value v) { return values.lookup(v); }

public:
  explicit Writer(llvm::raw_ostream &out) : out(out) {}
  llvm::Error write(func::FuncOp fn) {
    out << "import triton\nimport triton.language as tl\n\n@triton.jit\ndef "
           "kernel(";
    for (auto arg : fn.getArguments()) {
      auto id = "a" + std::to_string(arg.getArgNumber());
      values[arg] = id;
      out << id << ", ";
      if (auto v = dyn_cast<MemRefType>(arg.getType()))
        for (int i = 0; i < v.getRank(); ++i)
          out << id << "_s" << i << ", ";
    }
    out << "M, N, ROWS: tl.constexpr, WIDTH: tl.constexpr):\n";
    return body(fn.front(), 1);
  }
  llvm::Error body(Block &bb, unsigned depth) {
    for (auto &op : bb) {
      auto n = op.getName().stripDialect();
      if (op.getName().getStringRef() == "func.return")
        continue;
      if (op.getName().getDialectNamespace() != "triton_src")
        return llvm::createStringError("unexpected operation at emit");
      auto arg = [&](unsigned i) { return name(op.getOperand(i)); };
      auto alg = op.getAttrOfType<StringAttr>("algebra");
      bool sum = alg && alg == "sum";
      if (n == "tiles") {
        auto &loop = op.getRegion(0).front();
        auto k = "k" + std::to_string(next++);
        values[loop.getArgument(0)] = k;
        if (op.getNumResults()) {
          auto carry = "c" + std::to_string(next++);
          values[loop.getArgument(1)] = values[op.getResult(0)] = carry;
          line(depth, carry + " = " + arg(1));
        }
        line(depth, "for " + k + " in range(tl.cdiv(N, WIDTH)):");
        if (auto error = body(loop, depth + 1))
          return error;
        continue;
      }
      if (n == "yield") {
        if (op.getNumOperands())
          line(depth, name(bb.getArgument(1)) + " = " + arg(0));
        continue;
      }
      std::string e;
      if (n == "pid")
        e = "tl.program_id(0).to(tl.int64)";
      else if (n == "rows")
        e = arg(0) + " * ROWS + tl.arange(0, ROWS)";
      else if (n == "zero")
        e = "0";
      else if (n == "columns")
        e = "tl.arange(0, WIDTH).to(tl.int64) + " + arg(0) + " * WIDTH";
      else if (n == "active")
        e = "(" + arg(1) + "[:, None] < M) & (" + arg(2) + "[None, :] < N)";
      else if (n == "offsets") {
        auto rank = cast<MemRefType>(op.getOperand(0).getType()).getRank();
        auto base = rank == 2 ? arg(1) + "[:, None] * " + arg(0) + "_s0"
                              : "tl.full((ROWS, 1), 0, tl.int64)";
        e = base + " + " + arg(2) + "[None, :] * " + arg(0) + "_s" +
            std::to_string(rank - 1);
      } else if (n == "load")
        e = "tl.load(" + arg(0) + " + " + arg(1) + ", mask=" + arg(2) +
            ", other=0)";
      else if (n == "store") {
        line(depth, "tl.store(" + arg(0) + " + " + arg(1) + ", " + arg(3) +
                        ", mask=" + arg(2) + ")");
        continue;
      } else if (n == "neutralize")
        e = "tl.where(" + arg(1) + ", " + arg(0) + ", " +
            (sum ? "0.0" : "-float('inf')") + ")";
      else if (n == "reduce")
        e = std::string(sum ? "tl.sum" : "tl.max") + "(" + arg(0) + ", 1)";
      else if (n == "identity")
        e = std::string("tl.full((ROWS,), ") + (sum ? "0.0" : "-float('inf')") +
            ", tl.float32)";
      else if (n == "combine")
        e = sum ? arg(0) + " + " + arg(1)
                : "tl.maximum(" + arg(0) + ", " + arg(1) + ")";
      else if (n == "broadcast_rows")
        e = arg(0) + "[:, None]";
      else if (n == "add" || n == "sub" || n == "mul")
        e = arg(0) + (n == "add" ? " + " : n == "sub" ? " - " : " * ") + arg(1);
      else if (n == "div")
        e = "tl.div_rn(" + arg(0) + ", " + arg(1) + ")";
      else if (n == "exp" || n == "rsqrt")
        e = "tl." + n.str() + "(" + arg(0) + ")";
      else if (n == "cast") {
        auto t = op.getResult(0).getType();
        if (auto shaped = dyn_cast<ShapedType>(t))
          t = shaped.getElementType();
        e = arg(0) + ".to(tl." +
            std::string(t.isF16() ? "float16" : "float32") + ")";
      } else
        return llvm::createStringError("unhandled legal operation " + n.str());
      auto id = "v" + std::to_string(next++);
      values[op.getResult(0)] = id;
      line(depth, id + " = " + e);
    }
    return llvm::Error::success();
  }
};
llvm::Error emit(ModuleOp m, llvm::raw_pwrite_stream &out) {
  auto fn = *m.getOps<func::FuncOp>().begin();
  auto args = llvm::json::parse(fn->getAttrOfType<StringAttr>("kir.abi"));
  if (!args)
    return args.takeError();
  llvm::json::Array configs;
  for (Attribute a : fn->getAttrOfType<ArrayAttr>("block.configs")) {
    auto d = cast<DictionaryAttr>(a);
    llvm::json::Object row;
    for (StringRef k : {"id", "rows", "width"})
      row[k] = d.getAs<IntegerAttr>(k).getInt();
    auto native = llvm::json::parse(d.getAs<StringAttr>("payload").getValue());
    if (!native)
      return native.takeError();
    row["native"] = std::move(*native);
    configs.push_back(std::move(row));
  }
  std::string src;
  llvm::raw_string_ostream stream(src);
  if (auto err = Writer(stream).write(fn))
    return err;
  out << llvm::json::Value(llvm::json::Object{
      {"version", 1},
      {"entry", "kernel"},
      {"source", src},
      {"abi", std::move(*args)},
      {"rows", std::move(configs)},
      {"topology", fn->getAttrOfType<StringAttr>("block.topology").str()},
      {"guard",
       llvm::json::Object{
           {"min_width",
            fn->getAttrOfType<IntegerAttr>("triton_src.min_width").getInt()},
           {"max_width",
            fn->getAttrOfType<IntegerAttr>("triton_src.max_width").getInt()
                ? llvm::json::Value(
                      fn->getAttrOfType<IntegerAttr>("triton_src.max_width")
                          .getInt())
                : llvm::json::Value(nullptr)}}}});
  return llvm::Error::success();
}
} // namespace
void bind(Unit &unit) {
  unit.dialects = [](DialectRegistry &r) {
    r.insert<intentdsl::itsrc::ItsrcDialect>();
  };
  for (StringRef phase : {"legal", "callable"})
    unit.boundaries.push_back(
        {{phase.str(),
          [phase](OpPassManager &pm,
                  const TargetProfile &profile) -> llvm::Error {
            if (phase == "legal")
              legalize(pm);
            else {
              auto *caps = profile.services.get<intentdsl::RowCapacity>();
              pm.addPass(std::make_unique<BindCallable>(
                  caps ? std::optional<uint64_t>{caps->upperBound} : std::nullopt));
            }
            return llvm::Error::success();
          },
          [phase](ModuleOp m, const TargetProfile &) {
            auto funcs = m.getOps<func::FuncOp>();
            if (!llvm::hasSingleElement(funcs))
              return failure();
            auto fn = *funcs.begin();
            if (!fn->getAttrOfType<StringAttr>("kir.abi") ||
                !fn->getAttrOfType<ArrayAttr>("block.configs") ||
                !fn->getAttrOfType<StringAttr>("block.topology"))
              return failure();
            if (phase == "callable" &&
                (!fn->getAttrOfType<IntegerAttr>("triton_src.min_width") ||
                 !fn->getAttrOfType<IntegerAttr>("triton_src.max_width")))
              return failure();
            bool valid = true;
            m.walk([&](Operation *op) {
              auto ns = op->getName().getDialectNamespace();
              valid &= ns == "builtin" || ns == "func" || ns == "triton_src";
            });
            return success(valid);
          }},
         COMPILERINFRA_SOURCE});
  unit.exit =
      Located<ExitImplementation>{{emit, capabilities}, COMPILERINFRA_SOURCE};
}
} // namespace CI_UNIT_NAMESPACE
