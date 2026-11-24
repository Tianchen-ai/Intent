#include "../ir/TileShape.h"
#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/Literal.h"
#include "intentdsl/levels/algorithm/kir/ir/Storage.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/TypeUtilities.h"
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
bool viewHasAnyFastMath(const LocalOpView &v, arith::FastMathFlags mask) {
  auto attr = v.attrOfType<arith::FastMathFlagsAttr>("fastmath");
  return attr && bitEnumContainsAny(attr.getValue(), mask);
}
bool viewHasFastMath(const LocalOpView &v, arith::FastMathFlags req) {
  auto attr = v.attrOfType<arith::FastMathFlagsAttr>("fastmath");
  return attr && bitEnumContainsAll(attr.getValue(), req);
}

TerminalPrinter tilePrinter() {
  TerminalPrinter p;
  registerColumnRules(p);
  auto leaf = [&](StringRef name, auto rule) {
    p.operations[name.str()] = PrintRule::local(
        [rule](const LocalOpView &view, PrintPoint at, PrintContext &c) {
          if (at != PrintPoint::Enter)
            return llvm::Error::success();
          auto expr = rule(view);
          if (!expr.empty())
            c.line(view.numResults() ? view.result(0).str() + " = " + expr
                                     : expr);
          return llvm::Error::success();
        });
  };

  // Declarative templates
  p.operations["triton_src.tile_program"] = PrintRule::format("");
  p.operations["triton_src.tile_pid"] =
      PrintRule::format("$r0 = tl.program_id({axis}).to(tl.int64)");
  p.operations["triton_src.from_le_half"] = PrintRule::format(
      "$r0 = ($0|($1<<8)).to(tl.uint16).to(tl.float16,bitcast=True).to(tl.float32)");
  p.operations["arith.minimumf"] = PrintRule::format(
      "$r0 = tl.minimum($operands,propagate_nan=tl.PropagateNan.ALL)");
  p.operations["math.absf"] = PrintRule::format("$r0 = tl.abs($0)");
  p.operations["math.roundeven"] = PrintRule::format("$r0 = libdevice.rint($0)");
  p.operations["arith.minnumf"] = PrintRule::format("$r0 = tl.minimum($operands)");
  p.operations["arith.shli"] = PrintRule::format("$r0 = $0<<$1");
  p.operations["arith.shrui"] =
      PrintRule::format("$r0 = ($0.to(tl.uint32)>>$1).to(tl.int32)");
  p.operations["arith.maxnumf"] = PrintRule::format("$r0 = tl.maximum($operands)");
  p.operations["arith.minsi"] = PrintRule::format("$r0 = tl.minimum($operands)");
  p.operations["arith.subi"] = PrintRule::format("$r0 = $0-$1");
  p.operations["arith.floordivsi"] = PrintRule::format("$r0 = $0//$1");
  p.operations["arith.remsi"] = PrintRule::format("$r0 = $0%$1");
  p.operations["arith.ori"] = PrintRule::format("$r0 = $0|$1");
  p.operations["arith.xori"] = PrintRule::format("$r0 = $0^$1");
  p.operations["math.log"] = PrintRule::format("$r0 = tl.log($0)");
  p.operations["math.exp2"] = PrintRule::format("$r0 = tl.exp2($0)");
  p.operations["arith.select"] = PrintRule::format("$r0 = tl.where($operands)");
  p.operations["triton_src.transpose"] = PrintRule::format("$r0 = tl.trans($0)");
  p.operations["triton_src.scaled_mma"] = PrintRule::format(
      "$r0 = tl.dot_scaled($0,$1.to(tl.uint8),'e4m3',$2,$3.to(tl.uint8),'e4m3',$4)");

  // Local rules (LocalOpView only)
  leaf("triton_src.reshape", [](const LocalOpView &v) {
    return ("tl.reshape(" + v.operand(0) + "," + tuple(v.resultType(0)) + ")")
        .str();
  });
  leaf("triton_src.expand_scale", [](const LocalOpView &v) {
    auto ax = v.attrOfType<IntegerAttr>("axis").getInt();
    return ("tl.broadcast_to(tl.expand_dims(" + v.operand(0) + "," +
            std::to_string(ax) + ")," + tuple(v.resultType(0)) + ")")
        .str();
  });
  leaf("triton_src.make_desc", [](const LocalOpView &v) {
    auto joinI64 = [](DenseI64ArrayAttr a) {
      std::vector<std::string> parts;
      for (int64_t x : a.asArrayRef())
        parts.push_back(std::to_string(x));
      return "[" + llvm::join(parts, ",") + "]";
    };
    return ("tl.make_tensor_descriptor(" + v.operand(0).str() + ",shape=" +
            joinI64(v.attrOfType<DenseI64ArrayAttr>("view_shape")) +
            ",strides=" +
            joinI64(v.attrOfType<DenseI64ArrayAttr>("view_strides")) +
            ",block_shape=" +
            joinI64(v.attrOfType<DenseI64ArrayAttr>("block_shape")) + ")");
  });
  leaf("triton_src.desc_read", [](const LocalOpView &v) {
    std::vector<std::string> offs;
    for (unsigned i = 1; i < v.numOperands(); ++i)
      offs.push_back("tl.cast(" + v.operand(i).str() + ",tl.int32)");
    return (v.operand(0).str() + ".load([" + llvm::join(offs, ",") + "])");
  });
  leaf("triton_src.desc_write", [](const LocalOpView &v) {
    std::vector<std::string> offs;
    for (unsigned i = 2; i < v.numOperands(); ++i)
      offs.push_back("tl.cast(" + v.operand(i).str() + ",tl.int32)");
    return (v.operand(0).str() + ".store([" + llvm::join(offs, ",") + "]," +
            v.operand(1).str() + ")");
  });
  leaf("math.exp", [](const LocalOpView &v) {
    if (viewHasFastMath(v, arith::FastMathFlags::afn))
      return ("tl.exp2((" + v.operand(0) + ")*1.4426950408889634)").str();
    return ("tl.exp(" + v.operand(0) + ")").str();
  });
  leaf("triton_src.tile_axis", [](const LocalOpView &v) {
    return "tl.arange(0," + std::to_string(shape(v.resultType(0))[0]) +
           ").to(tl.int64)";
  });
  leaf("triton_src.drop_unit_owner", [](const LocalOpView &v) {
    return ("tl.reshape(" + v.operand(0) + "," + tuple(v.resultType(0)) + ")")
        .str();
  });
  leaf("triton_src.tile_prefix", [](const LocalOpView &v) {
    return ("tl.gather(" + v.operand(0) + ",tl.arange(0," +
            std::to_string(shape(v.resultType(0))[0]) + "),0)")
        .str();
  });
  leaf("triton_src.tile_address", [](const LocalOpView &v) {
    auto strides = v.attrOfType<DenseI64ArrayAttr>("strides");
    auto axes = v.attrOfType<DenseI64ArrayAttr>("projection");
    auto rank = shape(v.resultType(0)).size();
    auto width = v.attrOfType<IntegerAttr>("index_bits").getInt();
    std::vector<std::string> terms;
    for (size_t i = 0; i < axes.size(); ++i) {
      std::string projection =
          rank == 2 && axes[i] >= 0 ? (axes[i] == 0 ? "[:,None]" : "[None,:]")
                                    : "";
      terms.push_back("tl.cast(" + v.operand(i + 1).str() + ",tl.int" +
                      std::to_string(width) + ")" + projection + "*" +
                      std::to_string(strides[i]));
    }
    auto sum = llvm::join(terms, " + ");
    return width == 32 ? "tl.cast(" + sum + ",tl.int64)" : sum;
  });
  leaf("triton_src.tile_read", [](const LocalOpView &v) {
    auto e = getElementTypeOrSelf(v.resultType(0));
    bool fp8 = isa<Float8E4M3FNType>(e);
    return ("tl.load(" + v.operand(0) +
            (fp8 ? ".to(tl.pointer_type(tl.uint8))" : "") + "+" + v.operand(1) +
            (v.numOperands() == 3 ? ",mask=" + v.operand(2) + ",other=0" : "") +
            ")" + (fp8 ? ".to(tl.float8e4nv,bitcast=True)" : ""))
        .str();
  });
  leaf("triton_src.e8m0", [](const LocalOpView &v) {
    using F = intentdsl::kir::MX8Block;
    return ("tl.where(" + v.operand(0) + "==" + std::to_string(F::nan) +
            ",float('nan'),libdevice.ldexp(tl.full(" + tuple(v.resultType(0)) +
            ",1.,tl.float32)," + v.operand(0) + "-" + std::to_string(F::bias) +
            "))")
        .str();
  });
  leaf("triton_src.tile_write", [](const LocalOpView &v) {
    return ("tl.store(" + v.operand(0) + "+" + v.operand(1) + "," +
            v.operand(2) +
            (v.numOperands() == 4 ? ",mask=" + v.operand(3) : "") + ")")
        .str();
  });
  leaf("triton_src.tile_broadcast", [](const LocalOpView &v) {
    auto axis = v.attrOfType<IntegerAttr>("axis").getInt();
    auto castExpr = "tl.cast(" + v.operand(0).str() + "," +
                    dtype(v.operandType(0)) + ")" +
                    (axis < 0    ? ""
                     : axis == 0 ? "[:,None]"
                                 : "[None,:]");
    return "tl.broadcast_to(" + castExpr + "," + tuple(v.resultType(0)) + ")";
  });
  leaf("triton_src.tile_dot", [](const LocalOpView &v) {
    auto rhs = v.attrOfType<BoolAttr>("transpose_rhs").getValue()
                   ? ("tl.trans(" + v.operand(1) + ")").str()
                   : v.operand(1).str();
    return ("tl.dot(" + v.operand(0) + "," + rhs +
            (v.numOperands() == 3 ? "," + v.operand(2) : "") +
            (v.attr("triton_src.exact_acc") ? ",max_num_imprecise_acc=0" : "") +
            ",input_precision='ieee',out_dtype=" + dtype(v.resultType(0)) + ")")
        .str();
  });
  leaf("triton_src.tile_reduce", [](const LocalOpView &v) {
    auto expression = "tl." + v.attrOfType<StringAttr>("algebra").str() + "(" +
                      v.operand(0).str() + "," +
                      std::to_string(v.attrOfType<IntegerAttr>("axis").getInt()) +
                      ")";
    return getElementTypeOrSelf(v.resultType(0)).isInteger(1)
               ? expression + ".to(tl.int1)"
               : expression;
  });
  leaf("arith.constant", [](const LocalOpView &v) {
    auto val = v.attr("value");
    if (auto i = dyn_cast<IntegerAttr>(val)) {
      if (i.getType().isInteger(1))
        return std::string(i.getValue().isZero() ? "False" : "True");
      if (!isa<ShapedType>(v.resultType(0)) && v.resultType(0).isIndex())
        return std::to_string(i.getInt());
      return std::to_string(i.getInt());
    }
    if (auto f = dyn_cast<FloatAttr>(val)) {
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
    auto type = v.resultType(0);
    auto dense = cast<DenseElementsAttr>(val);
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
    leaf(name, [token](const LocalOpView &v) {
      if (v.attr("triton_src.round_nearest")) {
        auto stem = v.operation().split('.').second.drop_back();
        return ("libdevice." + stem + "_rn(" + llvm::join(v.operands(), ",") +
                ")")
            .str();
      }
      if (v.operation() == "arith.divf" &&
          !viewHasAnyFastMath(v, arith::FastMathFlags::arcp |
                                     arith::FastMathFlags::afn))
        return "tl.div_rn(" + llvm::join(v.operands(), ",") + ")";
      return (v.operand(0) + token + v.operand(1)).str();
    });
  leaf("arith.cmpi", [](const LocalOpView &v) {
    auto raw = v.attrOfType<IntegerAttr>("predicate").getValue().getZExtValue();
    auto pred = arith::symbolizeCmpIPredicate(raw).value_or(
        arith::CmpIPredicate::slt);
    auto token = pred == arith::CmpIPredicate::sge   ? ">="
                 : pred == arith::CmpIPredicate::sle ? "<="
                 : pred == arith::CmpIPredicate::sgt ? ">"
                 : pred == arith::CmpIPredicate::eq  ? "=="
                 : pred == arith::CmpIPredicate::ne  ? "!="
                                                     : "<";
    return (v.operand(0) + token + v.operand(1)).str();
  });
  leaf("arith.cmpf", [](const LocalOpView &v) {
    auto raw = v.attrOfType<IntegerAttr>("predicate").getValue().getZExtValue();
    auto pred = arith::symbolizeCmpFPredicate(raw).value_or(
        arith::CmpFPredicate::UNE);
    auto token = pred == arith::CmpFPredicate::OEQ   ? "=="
                 : pred == arith::CmpFPredicate::OGT ? ">"
                 : pred == arith::CmpFPredicate::OGE ? ">="
                 : pred == arith::CmpFPredicate::OLT ? "<"
                 : pred == arith::CmpFPredicate::OLE ? "<="
                                                     : "!=";
    return (v.operand(0) + token + v.operand(1)).str();
  });
  leaf("math.sqrt", [](const LocalOpView &v) {
    return (viewHasFastMath(v, arith::FastMathFlags::afn) &&
                    !v.attr("triton_src.round_nearest")
                ? "tl.sqrt(" + v.operand(0) + ")"
                : "tl.sqrt_rn(" + v.operand(0) + ")")
        .str();
  });
  p.operations["scf.if"] = PrintRule::local(
      [](const LocalOpView &v, PrintPoint at, PrintContext &c) {
        if (at == PrintPoint::Enter)
          c.line("if " + v.operand(0) + ":");
        if (at == PrintPoint::BeginRegion &&
            c.region < v.numRegions() && v.numBlocks(c.region) > 0) {
          if (c.region)
            c.line("else:");
          ++c.indent;
        } else if (at == PrintPoint::EndRegion &&
                   c.region < v.numRegions() && v.numBlocks(c.region) > 0) {
          --c.indent;
        }
        return llvm::Error::success();
      });
  for (auto name :
       {"arith.truncf", "arith.extf", "arith.extsi", "arith.sitofp",
        "arith.fptosi", "arith.index_cast", "arith.trunci"})
    leaf(name, [](const LocalOpView &v) {
      return ("tl.cast(" + v.operand(0) + "," + dtype(v.resultType(0)) + ")")
          .str();
    });
  leaf("arith.extui", [](const LocalOpView &v) {
    auto inElem = getElementTypeOrSelf(v.operandType(0));
    std::string uTy = inElem.isInteger(8)    ? "tl.uint8"
                      : inElem.isInteger(16) ? "tl.uint16"
                      : inElem.isInteger(32) ? "tl.uint32"
                                             : dtype(v.operandType(0));
    return ("tl.cast(tl.cast(" + v.operand(0).str() + "," + uTy + ")," +
            dtype(v.resultType(0)) + ")");
  });
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
  printer.operations["builtin.module"] = PrintRule::local(
      [](const LocalOpView &, PrintPoint at, PrintContext &p) {
        if (at == PrintPoint::Enter) {
          p.file = "kernel.py";
          p.line("import triton");
          p.line("import triton.language as tl");
          p.line("from triton.language.extra.cuda import libdevice");
        }
        return llvm::Error::success();
      });
  printer.operations["triton_src.runtime_source"] = PrintRule::local(
      [](const LocalOpView &v, PrintPoint at, PrintContext &p) -> llvm::Error {
        if (at == PrintPoint::Enter) {
          auto impl = v.attrOfType<StringAttr>("implementation");
          if (!impl)
            return llvm::createStringError(
                "missing runtime_source implementation");
          auto helper = p.source(impl.getValue());
          if (!helper)
            return helper.takeError();
          p.line(*helper);
        }
        return llvm::Error::success();
      });
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
