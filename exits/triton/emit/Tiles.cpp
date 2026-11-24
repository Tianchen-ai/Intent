#include "../ir/TileShape.h"
#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/FastMath.h"
#include "intentdsl/levels/algorithm/kir/ir/Literal.h"
#include "intentdsl/levels/algorithm/kir/ir/Storage.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/StringExtras.h"

namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
using namespace mlir;
void registerColumnRules(TerminalPrinter &);
namespace {
std::string tuple(Type t) {
  std::vector<std::string> dims;
  for (auto d : shape(t))
    dims.push_back(std::to_string(d));
  return "(" + llvm::join(dims, ",") + (dims.size() == 1 ? "," : "") + ")";
}
std::string dtype(Type t) {
  t = getElementTypeOrSelf(t);
  if (t.isBF16())
    return "tl.bfloat16";
  if (t.isF16())
    return "tl.float16";
  if (isa<Float8E4M3FNType>(t))
    return "tl.float8e4nv";
  if (t.isF32())
    return "tl.float32";
  if (t.isInteger(32))
    return "tl.int32";
  if (t.isIndex() || t.isInteger(64))
    return "tl.int64";
  if (t.isInteger(8))
    return "tl.int8";
  return "tl.int1";
}
TerminalPrinter tilePrinter() {
  TerminalPrinter p;
  registerColumnRules(p);
  auto leaf = [&](StringRef name, auto rule) {
    p.operations[name.str()] = [rule](Operation &op, PrintPoint at,
                                      PrintContext &c) {
      if (at != PrintPoint::Enter)
        return llvm::Error::success();
      std::vector<std::string> a;
      for (Value v : op.getOperands())
        a.push_back(c.name(v));
      auto expr = rule(op, a);
      if (!expr.empty())
        c.line(op.getNumResults() ? c.name(op.getResult(0)) + " = " + expr
                                  : expr);
      return llvm::Error::success();
    };
  };
  leaf("triton_src.tile_program", [](auto &, auto &) { return std::string(); });
  leaf("triton_src.tile_axis", [](Operation &op, auto &) {
    return "tl.arange(0," +
           std::to_string(shape(op.getResult(0).getType())[0]) +
           ").to(tl.int64)";
  });
  leaf("triton_src.tile_pid", [](Operation &op, auto &) {
    return "tl.program_id(" +
           std::to_string(op.getAttrOfType<IntegerAttr>("axis").getInt()) +
           ").to(tl.int64)";
  });
  leaf("triton_src.drop_unit_owner", [](Operation &op, auto &a) {
    return "tl.reshape(" + a[0] + "," + tuple(op.getResult(0).getType()) + ")";
  });
  leaf("triton_src.tile_prefix", [](Operation &op, auto &a) {
    return "tl.gather(" + a[0] + ",tl.arange(0," +
           std::to_string(shape(op.getResult(0).getType())[0]) + "),0)";
  });
  leaf("triton_src.tile_address", [](Operation &op, auto &a) {
    auto strides = op.getAttrOfType<DenseI64ArrayAttr>("strides");
    auto axes = op.getAttrOfType<DenseI64ArrayAttr>("projection");
    auto rank = shape(op.getResult(0).getType()).size();
    std::vector<std::string> terms;
    for (size_t i = 0; i < axes.size(); ++i) {
      std::string projection = rank == 2 && axes[i] >= 0
                                   ? (axes[i] == 0 ? "[:,None]" : "[None,:]")
                                   : "";
      auto width = op.getAttrOfType<IntegerAttr>("index_bits").getInt();
      terms.push_back("tl.cast(" + a[i + 1] + ",tl.int" +
                      std::to_string(width) + ")" + projection + "*" +
                      std::to_string(strides[i]));
    }
    auto sum = llvm::join(terms, " + ");
    return op.getAttrOfType<IntegerAttr>("index_bits").getInt() == 32
               ? "tl.cast(" + sum + ",tl.int64)"
               : sum;
  });
  leaf("triton_src.tile_read", [](Operation &op, auto &a) {
    auto e = getElementTypeOrSelf(op.getResult(0).getType());
    bool fp8 = isa<Float8E4M3FNType>(e);
    return "tl.load(" + a[0] + (fp8 ? ".to(tl.pointer_type(tl.uint8))" : "") +
           "+" + a[1] + (a.size() == 3 ? ",mask=" + a[2] + ",other=0" : "") +
           ")" + (fp8 ? ".to(tl.float8e4nv,bitcast=True)" : "");
  });
  leaf("triton_src.from_le_half", [](auto &, auto &a) {
    return "(" + a[0] + "|(" + a[1] +
           "<<8)).to(tl.uint16).to(tl.float16,bitcast=True).to(tl.float32)";
  });
  leaf("triton_src.e8m0", [](Operation &op, auto &a) {
    using F = intentdsl::kir::MX8Block;
    return "tl.where(" + a[0] + "==" + std::to_string(F::nan) +
           ",float('nan'),libdevice.ldexp(tl.full(" +
           tuple(op.getResult(0).getType()) + ",1.,tl.float32)," + a[0] + "-" +
           std::to_string(F::bias) + "))";
  });
  leaf("triton_src.tile_write", [](auto &, auto &a) {
    return "tl.store(" + a[0] + "+" + a[1] + "," + a[2] +
           (a.size() == 4 ? ",mask=" + a[3] : "") + ")";
  });
  leaf("triton_src.tile_broadcast", [](Operation &op, auto &a) {
    auto axis = op.getAttrOfType<IntegerAttr>("axis").getInt();
    auto v = "tl.cast(" + a[0] + "," + dtype(op.getOperand(0).getType()) + ")" +
             (axis < 0    ? ""
              : axis == 0 ? "[:,None]"
                          : "[None,:]");
    return "tl.broadcast_to(" + v + "," + tuple(op.getResult(0).getType()) +
           ")";
  });
  leaf("triton_src.tile_dot", [](Operation &op, auto &a) {
    auto rhs = op.getAttrOfType<BoolAttr>("transpose_rhs").getValue()
                   ? "tl.trans(" + a[1] + ")"
                   : a[1];
    return "tl.dot(" + a[0] + "," + rhs + (a.size() == 3 ? "," + a[2] : "") +
           (op.hasAttr("triton_src.exact_acc") ? ",max_num_imprecise_acc=0"
                                               : "") +
           ",input_precision='ieee',out_dtype=" +
           dtype(op.getResult(0).getType()) + ")";
  });
  leaf("triton_src.tile_reduce", [](Operation &op, auto &a) {
    auto expression =
        "tl." + op.getAttrOfType<StringAttr>("algebra").str() + "(" + a[0] +
        "," + std::to_string(op.getAttrOfType<IntegerAttr>("axis").getInt()) +
        ")";
    return getElementTypeOrSelf(op.getResult(0).getType()).isInteger(1)
               ? expression + ".to(tl.int1)"
               : expression;
  });
  leaf("arith.constant", [](Operation &op, auto &) {
    auto v = op.getAttr("value");
    if (auto i = dyn_cast<IntegerAttr>(v)) {
      if (i.getType().isInteger(1))
        return std::string(i.getValue().isZero() ? "False" : "True");
      return std::to_string(i.getInt());
    }
    if (auto f = dyn_cast<FloatAttr>(v)) {
      auto num = f.getValue();
      if (num.isInfinity())
        return std::string(num.isNegative() ? "-float('inf')" : "float('inf')");
      if (num.isNaN())
        return std::string("float('nan')");
      llvm::SmallString<32> txt;
      num.toString(txt);
      std::string number = txt.str().str();
      if (number.find_first_of(".eE") == std::string::npos)
        number += ".0";
      return number;
    }
    auto type = op.getResult(0).getType();
    auto dense = cast<DenseElementsAttr>(v);
    auto attr = dense.getSplatValue<Attribute>();
    std::string literal = intentdsl::kir::literal(cast<TypedAttr>(attr));
    return "tl.full(" + tuple(type) + "," + literal + "," + dtype(type) + ")";
  });
  for (auto [name, token] : {std::pair{"arith.muli", "*"},
                             {"arith.addi", "+"},
                             {"arith.divui", "//"},
                             {"arith.remui", "%"},
                             {"arith.mulf", "*"},
                             {"arith.divf", "/"},
                             {"arith.subf", "-"},
                             {"arith.addf", "+"},
                             {"arith.andi", "&"}})
    leaf(name, [token](Operation &op, auto &v) {
      if (op.hasAttr("triton_src.round_nearest"))
        return "libdevice." + op.getName().stripDialect().drop_back().str() +
               "_rn(" + llvm::join(v, ",") + ")";
      if (op.getName().getStringRef() == "arith.divf" &&
          !intentdsl::kir::hasAnyFastMath(
              &op, arith::FastMathFlags::arcp | arith::FastMathFlags::afn))
        return "tl.div_rn(" + llvm::join(v, ",") + ")";
      return v[0] + token + v[1];
    });
  leaf("arith.cmpi", [](Operation &op, auto &a) {
    auto pred = cast<arith::CmpIOp>(op).getPredicate();
    auto token = pred == arith::CmpIPredicate::sge  ? ">="
                 : pred == arith::CmpIPredicate::sle ? "<="
                 : pred == arith::CmpIPredicate::sgt ? ">"
                 : pred == arith::CmpIPredicate::eq  ? "=="
                 : pred == arith::CmpIPredicate::ne  ? "!="
                                                     : "<";
    return a[0] + token + a[1];
  });
  leaf("arith.cmpf", [](Operation &op, auto &a) {
    return a[0] +
           (cast<arith::CmpFOp>(op).getPredicate() == arith::CmpFPredicate::OEQ
                ? "=="
                : "!=") +
           a[1];
  });
  leaf("arith.minimumf", [](auto &, auto &a) {
    return "tl.minimum(" + llvm::join(a, ",") +
           ",propagate_nan=tl.PropagateNan.ALL)";
  });
  leaf("math.absf", [](auto &, auto &a) { return "tl.abs(" + a[0] + ")"; });
  leaf("math.roundeven",
       [](auto &, auto &a) { return "libdevice.rint(" + a[0] + ")"; });
  leaf("arith.minnumf", [](auto &, auto &a) {
    return "tl.minimum(" + llvm::join(a, ",") + ")";
  });
  leaf("arith.shli", [](auto &, auto &a) { return a[0] + "<<" + a[1]; });
  leaf("arith.shrui", [](auto &, auto &a) {
    return "(" + a[0] + ".to(tl.uint32)>>" + a[1] + ").to(tl.int32)";
  });
  for (auto [name, op] :
       {std::pair{"arith.maxnumf", "maximum"}, {"arith.minsi", "minimum"}})
    leaf(name, [op](auto &, auto &a) {
      return "tl." + std::string(op) + "(" + llvm::join(a, ",") + ")";
    });
  for (auto [name, token] : {std::pair{"arith.subi", "-"},
                             {"arith.floordivsi", "//"},
                             {"arith.remsi", "%"},
                             {"arith.ori", "|"},
                             {"arith.xori", "^"}})
    leaf(name, [token](auto &, auto &a) { return a[0] + token + a[1]; });
  for (auto name : {"exp", "log"})
    leaf("math." + std::string(name), [name](auto &, auto &a) {
      return "tl." + std::string(name) + "(" + a[0] + ")";
    });
  leaf("math.sqrt", [](Operation &op, auto &a) {
    return intentdsl::kir::hasFastMath(&op, arith::FastMathFlags::afn) &&
                   !op.hasAttr("triton_src.round_nearest")
               ? "tl.sqrt(" + a[0] + ")"
               : "tl.sqrt_rn(" + a[0] + ")";
  });
  leaf("arith.select",
       [](auto &, auto &a) { return "tl.where(" + llvm::join(a, ",") + ")"; });
  p.operations["scf.if"] = [](Operation &op, PrintPoint at, PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("if " + c.name(op.getOperand(0)) + ":");
    if (at == PrintPoint::BeginRegion && !op.getRegion(c.region).empty()) {
      if (c.region)
        c.line("else:");
      ++c.indent;
    } else if (at == PrintPoint::EndRegion && !op.getRegion(c.region).empty())
      --c.indent;
    return llvm::Error::success();
  };
  for (auto name :
       {"arith.truncf", "arith.extf", "arith.extsi", "arith.sitofp",
        "arith.fptosi", "arith.index_cast", "arith.trunci", "arith.extui"})
    leaf(name, [](Operation &op, auto &a) {
      return "tl.cast(" + a[0] + "," + dtype(op.getResult(0).getType()) + ")";
    });
  leaf("math.exp2", [](auto &, auto &a) { return "tl.exp2(" + a[0] + ")"; });
  p.package = [](DictionaryAttr, const auto &files,
                 llvm::raw_pwrite_stream &out) {
    out << files.at("kernel.py");
    return llvm::Error::success();
  };
  return p;
}
} // namespace
void addRowRules(TerminalPrinter &);
TerminalPrinter tritonPrinter() {
  auto printer = tilePrinter();
  addRowRules(printer);
  printer.operations["builtin.module"] = [](Operation &, PrintPoint at,
                                            PrintContext &p) {
    if (at == PrintPoint::Enter) {
      p.file = "kernel.py";
      p.line("import triton");
      p.line("import triton.language as tl");
      p.line("from triton.language.extra.cuda import libdevice");
    }
    return llvm::Error::success();
  };
  printer.operations["triton_src.runtime_source"] =
      [](Operation &op, PrintPoint at, PrintContext &p) -> llvm::Error {
    if (at == PrintPoint::Enter) {
      auto impl = op.getAttrOfType<StringAttr>("implementation");
      if (!impl)
        return llvm::createStringError("missing runtime_source implementation");
      auto helper = p.source(impl.getValue());
      if (!helper)
        return helper.takeError();
      p.line(*helper);
    }
    return llvm::Error::success();
  };
  printer.package = [](DictionaryAttr attrs, const auto &files,
                       llvm::raw_pwrite_stream &os) -> llvm::Error {
    auto package = attrs.getAs<StringAttr>("triton_src.package");
    if (!package)
      return llvm::createStringError("missing verified Triton ABI");
    auto data = llvm::json::parse(package.getValue());
    if (!data)
      return data.takeError();
    auto file = files.find("kernel.py");
    if (!data->getAsObject() || file == files.end())
      return llvm::createStringError("incomplete kernel artifact");
    (*data->getAsObject())["source"] = file->second;
    os << *data;
    return llvm::Error::success();
  };
  return printer;
}
} // namespace CI_UNIT_NAMESPACE
