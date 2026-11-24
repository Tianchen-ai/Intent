#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/Storage.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/StringExtras.h"

namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
using namespace mlir;

namespace {
std::string mojoScalarType(Type t) {
  if (t.isIndex())
    return "Int";
  if (t.isF16())
    return "Float16";
  if (t.isBF16())
    return "BFloat16";
  if (t.isF32())
    return "Float32";
  if (t.isInteger(1))
    return "Bool";
  if (t.isUnsignedInteger(64))
    return "UInt64";
  if (t.isInteger(64))
    return "Int64";
  return "Int32";
}
std::string mojoDTypeName(Type t) {
  if (auto sh = dyn_cast<ShapedType>(t))
    t = sh.getElementType();
  if (t.isBF16())
    return "DType.bfloat16";
  if (t.isF16())
    return "DType.float16";
  if (t.isInteger(1))
    return "DType.bool";
  if (t.isUnsignedInteger(64))
    return "DType.uint64";
  if (t.isInteger(64))
    return "DType.int64";
  if (t.isInteger(32))
    return "DType.int32";
  return "DType.float32";
}
} // namespace

TerminalPrinter mojoMatrixPrinter() {
  TerminalPrinter p;
  auto noop = [](Operation &, PrintPoint, PrintContext &) {
    return llvm::Error::success();
  };
  p.operations["builtin.module"] = noop;
  p.operations["func.func"] = noop;
  p.operations["func.return"] = noop;
  p.operations["mojo_row.end"] = noop;

  auto twoFileProgram = [](Operation &, PrintPoint at, PrintContext &c) {
    if (at == PrintPoint::BeginRegion) {
      c.file = c.region == 0 ? "row.mojo" : "entry.c";
      c.indent = 0;
    }
    return llvm::Error::success();
  };
  p.operations["mojo_row.matrix_program"] = twoFileProgram;
  p.operations["mojo_row.ordered_program"] = twoFileProgram;
  p.operations["mojo_row.row_program"] = twoFileProgram;

  p.operations["mojo_row.runtime_source"] =
      [](Operation &op, PrintPoint at, PrintContext &c) -> llvm::Error {
    if (at == PrintPoint::Enter) {
      auto key = op.getAttrOfType<StringAttr>("implementation").getValue();
      auto src = c.source(key);
      if (!src)
        return src.takeError();
      c.line(*src);
    }
    return llvm::Error::success();
  };

  // ---------------- Arithmetic & Index Ops ----------------
  p.operations["arith.constant"] = [](Operation &op, PrintPoint at,
                                      PrintContext &c) -> llvm::Error {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    if (c.file == "entry.c") {
      llvm::SmallString<24> num;
      op.getAttrOfType<FloatAttr>("value").getValue().toString(num);
      std::string s = num.str().str();
      if (s.find_first_of(".Ee") == std::string::npos)
        s += ".0";
      c.line("const float " + c.name(op.getResult(0)) + " = " + s + "f;");
      return llvm::Error::success();
    }
    Type t = op.getResult(0).getType();
    if (auto fAttr = op.getAttrOfType<FloatAttr>("value")) {
      auto apf = fAttr.getValue();
      std::string expr;
      if (apf.isInfinity()) {
        expr = apf.isNegative() ? "bitcast[DType.float32](UInt32(0xff800000))"
                                : "bitcast[DType.float32](UInt32(0x7f800000))";
        if (!t.isF32())
          expr = mojoScalarType(t) + "(" + expr + ")";
      } else if (apf.isNaN()) {
        expr = "bitcast[DType.float32](UInt32(0x7fc00000))";
        if (!t.isF32())
          expr = mojoScalarType(t) + "(" + expr + ")";
      } else {
        llvm::SmallString<32> num;
        apf.toString(num);
        expr = mojoScalarType(t) + "(" + num.str().str() + ")";
      }
      c.line("var " + c.name(op.getResult(0)) + " = " + expr);
      return llvm::Error::success();
    }
    auto iAttr = op.getAttrOfType<IntegerAttr>("value");
    if (t.isInteger(1)) {
      c.line("var " + c.name(op.getResult(0)) + " = " +
             (iAttr.getInt() ? "True" : "False"));
      return llvm::Error::success();
    }
    if (t.isUnsignedInteger(64)) {
      c.line("var " + c.name(op.getResult(0)) + " = UInt64(" +
             std::to_string(iAttr.getUInt()) + ")");
      return llvm::Error::success();
    }
    c.line("var " + c.name(op.getResult(0)) + " = " + mojoScalarType(t) + "(" +
           std::to_string(iAttr.getInt()) + ")");
    return llvm::Error::success();
  };

  auto binIdx = [&](StringRef opName, StringRef mojoSym, StringRef cSym) {
    p.operations[opName.str()] = [mojoSym, cSym](Operation &op, PrintPoint at,
                                                 PrintContext &c) {
      if (at != PrintPoint::Enter)
        return llvm::Error::success();
      auto lhs = c.name(op.getOperand(0)), rhs = c.name(op.getOperand(1));
      auto res = c.name(op.getResult(0));
      if (c.file == "entry.c")
        c.line("const float " + res + " = " + lhs + " " + cSym.str() + " " +
               rhs + ";");
      else
        c.line("var " + res + " = (" + lhs + " " + mojoSym.str() + " " + rhs +
               ")");
      return llvm::Error::success();
    };
  };
  binIdx("arith.addi", "+", "+");
  binIdx("arith.subi", "-", "-");
  binIdx("arith.muli", "*", "*");
  binIdx("arith.divsi", "//", "/");
  binIdx("arith.floordivsi", "//", "/");
  binIdx("arith.remsi", "%", "%");
  binIdx("arith.addf", "+", "+");
  binIdx("arith.subf", "-", "-");
  binIdx("arith.mulf", "*", "*");
  binIdx("arith.divf", "/", "/");
  binIdx("arith.xori", "^", "^");
  binIdx("arith.ori", "|", "|");
  binIdx("arith.shli", "<<", "<<");
  binIdx("arith.shrsi", ">>", ">>");
  binIdx("arith.shrui", ">>", ">>");

  p.operations["arith.andi"] = [](Operation &op, PrintPoint at,
                                  PrintContext &c) {
    if (at == PrintPoint::Enter) {
      auto lhs = c.name(op.getOperand(0)), rhs = c.name(op.getOperand(1));
      std::string opSym = op.getResult(0).getType().isInteger(1) ? "and" : "&";
      c.line("var " + c.name(op.getResult(0)) + " = (" + lhs + " " + opSym +
             " " + rhs + ")");
    }
    return llvm::Error::success();
  };

  p.operations["arith.cmpf"] = [](Operation &op, PrintPoint at,
                                  PrintContext &c) -> llvm::Error {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    auto cmp = cast<arith::CmpFOp>(op);
    StringRef tok;
    switch (cmp.getPredicate()) {
    case arith::CmpFPredicate::OEQ:
    case arith::CmpFPredicate::UEQ:
      tok = "==";
      break;
    case arith::CmpFPredicate::ONE:
    case arith::CmpFPredicate::UNE:
      tok = "!=";
      break;
    case arith::CmpFPredicate::OLT:
    case arith::CmpFPredicate::ULT:
      tok = "<";
      break;
    case arith::CmpFPredicate::OLE:
    case arith::CmpFPredicate::ULE:
      tok = "<=";
      break;
    case arith::CmpFPredicate::OGT:
    case arith::CmpFPredicate::UGT:
      tok = ">";
      break;
    case arith::CmpFPredicate::OGE:
    case arith::CmpFPredicate::UGE:
      tok = ">=";
      break;
    default:
      return llvm::createStringError("unsupported arith.cmpf predicate");
    }
    c.line("var " + c.name(op.getResult(0)) + " = (" +
           c.name(op.getOperand(0)) + " " + tok.str() + " " +
           c.name(op.getOperand(1)) + ")");
    return llvm::Error::success();
  };

  p.operations["arith.select"] = [](Operation &op, PrintPoint at,
                                    PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("var " + c.name(op.getResult(0)) + " = (" +
             c.name(op.getOperand(1)) + " if " + c.name(op.getOperand(0)) +
             " else " + c.name(op.getOperand(2)) + ")");
    return llvm::Error::success();
  };

  p.operations["arith.minsi"] = [](Operation &op, PrintPoint at,
                                   PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("var " + c.name(op.getResult(0)) + " = min(" +
             c.name(op.getOperand(0)) + ", " + c.name(op.getOperand(1)) + ")");
    return llvm::Error::success();
  };

  p.operations["arith.cmpi"] = [](Operation &op, PrintPoint at,
                                  PrintContext &c) -> llvm::Error {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    auto cmp = cast<arith::CmpIOp>(op);
    StringRef tok;
    switch (cmp.getPredicate()) {
    case arith::CmpIPredicate::eq:
      tok = "==";
      break;
    case arith::CmpIPredicate::ne:
      tok = "!=";
      break;
    case arith::CmpIPredicate::slt:
    case arith::CmpIPredicate::ult:
      tok = "<";
      break;
    case arith::CmpIPredicate::sle:
    case arith::CmpIPredicate::ule:
      tok = "<=";
      break;
    case arith::CmpIPredicate::sgt:
    case arith::CmpIPredicate::ugt:
      tok = ">";
      break;
    case arith::CmpIPredicate::sge:
    case arith::CmpIPredicate::uge:
      tok = ">=";
      break;
    }
    c.line("var " + c.name(op.getResult(0)) + " = (" +
           c.name(op.getOperand(0)) + " " + tok.str() + " " +
           c.name(op.getOperand(1)) + ")");
    return llvm::Error::success();
  };

  p.operations["arith.index_cast"] = [](Operation &op, PrintPoint at,
                                        PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("var " + c.name(op.getResult(0)) + " = " +
             mojoScalarType(op.getResult(0).getType()) + "(" +
             c.name(op.getOperand(0)) + ")");
    return llvm::Error::success();
  };

  p.operations["arith.maxsi"] = [](Operation &op, PrintPoint at,
                                   PrintContext &c) {
    if (at == PrintPoint::Enter) {
      auto lhs = c.name(op.getOperand(0)), rhs = c.name(op.getOperand(1));
      c.line("var " + c.name(op.getResult(0)) + " = (" + lhs + " if " + lhs +
             " > " + rhs + " else " + rhs + ")");
    }
    return llvm::Error::success();
  };

  p.operations["arith.maxnumf"] = [](Operation &op, PrintPoint at,
                                     PrintContext &c) {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    auto lhs = c.name(op.getOperand(0)), rhs = c.name(op.getOperand(1)),
         res = c.name(op.getResult(0));
    if (isa<ShapedType>(op.getResult(0).getType()))
      c.line("var " + res + " = " + lhs + ".ne(" + lhs + ").select(" + lhs +
             ", " + rhs + ".ne(" + rhs + ").select(" + rhs + ", max(" + lhs +
             ", " + rhs + ")))");
    else
      c.line("var " + res + " = " + lhs + " if isnan(" + rhs + ") else (" +
             rhs + " if isnan(" + lhs + ") else max(" + lhs + ", " + rhs +
             "))");
    return llvm::Error::success();
  };

  // Scalar panel map ops in entry.c + Mojo casts/math in row.mojo
  p.operations["arith.extf"] = [](Operation &op, PrintPoint at,
                                  PrintContext &c) {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    if (c.file == "entry.c") {
      c.line("const float " + c.name(op.getResult(0)) + " = expand_bf(" +
             c.name(op.getOperand(0)) + ");");
      return llvm::Error::success();
    }
    auto t = op.getResult(0).getType();
    std::string expr = isa<ShapedType>(t)
                           ? (c.name(op.getOperand(0)) + ".cast[" +
                              mojoDTypeName(t) + "]()")
                           : (mojoScalarType(t) + "(" +
                              c.name(op.getOperand(0)) + ")");
    c.line("var " + c.name(op.getResult(0)) + " = " + expr);
    return llvm::Error::success();
  };
  p.operations["arith.truncf"] = [](Operation &op, PrintPoint at,
                                    PrintContext &c) {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    if (c.file == "entry.c") {
      c.line("const uint16_t " + c.name(op.getResult(0)) + " = narrow_bf(" +
             c.name(op.getOperand(0)) + ");");
      return llvm::Error::success();
    }
    auto t = op.getResult(0).getType();
    std::string expr = isa<ShapedType>(t)
                           ? (c.name(op.getOperand(0)) + ".cast[" +
                              mojoDTypeName(t) + "]()")
                           : (mojoScalarType(t) + "(" +
                              c.name(op.getOperand(0)) + ")");
    c.line("var " + c.name(op.getResult(0)) + " = " + expr);
    return llvm::Error::success();
  };
  p.operations["arith.minimumf"] = [](Operation &op, PrintPoint at,
                                      PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("const float " + c.name(op.getResult(0)) + " = min_ieee(" +
             c.name(op.getOperand(0)) + ", " + c.name(op.getOperand(1)) + ");");
    return llvm::Error::success();
  };
  p.operations["math.exp"] = [](Operation &op, PrintPoint at, PrintContext &c) {
    if (at == PrintPoint::Enter) {
      if (c.file == "entry.c")
        c.line("const float " + c.name(op.getResult(0)) + " = expf(" +
               c.name(op.getOperand(0)) + ");");
      else
        c.line("var " + c.name(op.getResult(0)) + " = exp(" +
               c.name(op.getOperand(0)) + ")");
    }
    return llvm::Error::success();
  };
  p.operations["math.exp2"] = [](Operation &op, PrintPoint at,
                                 PrintContext &c) {
    if (at == PrintPoint::Enter) {
      auto x = c.name(op.getOperand(0));
      if (isa<ShapedType>(op.getResult(0).getType()))
        c.line("var " + c.name(op.getResult(0)) + " = " + x + ".ne(" + x +
               ").select(" + x + ", exp2(" + x + "))");
      else
        c.line("var " + c.name(op.getResult(0)) + " = " + x + " if isnan(" + x +
               ") else exp2(" + x + ")");
    }
    return llvm::Error::success();
  };
  p.operations["math.log"] = [](Operation &op, PrintPoint at, PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("var " + c.name(op.getResult(0)) + " = log(" +
             c.name(op.getOperand(0)) + ")");
    return llvm::Error::success();
  };
  p.operations["math.sqrt"] = [](Operation &op, PrintPoint at,
                                 PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("var " + c.name(op.getResult(0)) + " = sqrt(" +
             c.name(op.getOperand(0)) + ")");
    return llvm::Error::success();
  };
  p.operations["math.rsqrt"] = [](Operation &op, PrintPoint at,
                                  PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("var " + c.name(op.getResult(0)) + " = rsqrt(" +
             c.name(op.getOperand(0)) + ")");
    return llvm::Error::success();
  };

  p.operations["scf.if"] = [](Operation &op, PrintPoint at, PrintContext &c) {
    if (at == PrintPoint::Enter) {
      for (auto r : op.getResults()) {
        std::string tyStr = mojoScalarType(r.getType());
        if (auto sh = dyn_cast<ShapedType>(r.getType()))
          tyStr = "SIMD[" + mojoDTypeName(sh) + "," +
                  std::to_string(sh.getDimSize(0)) + "]";
        c.line("var " + c.name(r) + ": " + tyStr);
      }
      c.line("if " + c.name(op.getOperand(0)) + ":");
    } else if (at == PrintPoint::BeginRegion) {
      if (c.region == 1 && !op.getRegion(1).empty())
        c.line("else:");
    } else if (at == PrintPoint::BeginBlock) {
      c.indent++;
      auto &bb = op.getRegion(c.region).front();
      if (bb.without_terminator().empty() && op.getNumResults() == 0)
        c.line("pass");
    } else if (at == PrintPoint::EndBlock) {
      c.indent--;
    }
    return llvm::Error::success();
  };
  p.operations["scf.yield"] = [](Operation &op, PrintPoint at,
                                 PrintContext &c) {
    if (at == PrintPoint::Enter && isa<scf::IfOp>(op.getParentOp())) {
      for (auto [r, v] :
           llvm::zip(op.getParentOp()->getResults(), op.getOperands()))
        c.line(c.name(r) + " = " + c.name(v));
    }
    return llvm::Error::success();
  };
  p.operations["mojo_row.host_panel_yield"] = [](Operation &op, PrintPoint at,
                                                 PrintContext &c) {
    if (at == PrintPoint::Enter) {
      bool wide = op.getAttrOfType<BoolAttr>("widen_bf16").getValue();
      std::string dst = op.getAttrOfType<StringAttr>("dst_expr").str();
      std::string v = c.name(op.getOperand(0));
      c.line(dst + " = " + (wide ? "expand_bf(" + v + ")" : v) + ";");
    }
    return llvm::Error::success();
  };

  // ---------------- Panel ("row.mojo") Ops ----------------
  p.operations["mojo_row.panel_fn"] = [](Operation &op, PrintPoint at,
                                         PrintContext &c) -> llvm::Error {
    if (at == PrintPoint::Enter) {
      c.line("from std.memory import bitcast");
      c.line("from std.math import fma");
      if (op.getAttrOfType<BoolAttr>("uses_stack").getValue())
        c.line("from std.memory import unsafe_stack_allocation");
      if (op.getAttrOfType<BoolAttr>("uses_brgemm").getValue())
        c.line("from std.ffi import external_call");
      if (op.getAttrOfType<BoolAttr>("uses_vnni").getValue())
        c.line("from std.sys import llvm_intrinsic");
      if (op.getAttrOfType<BoolAttr>("uses_round_bf16").getValue()) {
        c.line("@always_inline");
        c.line("def round_bf16[W:Int](x:SIMD[DType.float32,W])->SIMD[DType.bfloat16,W]:");
        c.line("    var bits=bitcast[DType.uint32,W](x)");
        c.line("    var high=bits>>16");
        c.line("    var rounded=(bits+0x7fff+(high&1))>>16");
        c.line("    var result=(bits&0x7fffffff).gt(0x7f800000).select(high|0x40,rounded)");
        c.line("    return bitcast[DType.bfloat16,W](result.cast[DType.uint16]())");
      }
      auto &bb = op.getRegion(0).front();
      auto dtypes = op.getAttrOfType<ArrayAttr>("param_dtypes");
      bool usesBrg = op.getAttrOfType<BoolAttr>("uses_brgemm").getValue();
      SmallVector<std::string> params;
      for (size_t i = 0; i < dtypes.size(); ++i) {
        auto dt = cast<StringAttr>(dtypes[i]).getValue().str();
        params.push_back(c.name(bb.getArgument(i)) +
                         ": Pointer[Scalar[DType." + dt +
                         "],MutUntrackedOrigin]");
      }
      size_t base = dtypes.size();
      params.push_back(c.name(bb.getArgument(base)) + ": Int64");
      params.push_back(c.name(bb.getArgument(base + 1)) + ": Int64");
      if (usesBrg) {
        params.push_back(c.name(bb.getArgument(base + 2)) +
                         ": Pointer[UInt8,MutUntrackedOrigin]");
        params.push_back(c.name(bb.getArgument(base + 3)) +
                         ": Pointer[UInt8,MutUntrackedOrigin]");
      }
      c.line("@export(\"intent_panel\")");
      c.line("def intent_panel(" + llvm::join(params, ", ") +
             ") abi(\"C\") -> Int32:");
    } else if (at == PrintPoint::BeginBlock) {
      c.indent++;
    } else if (at == PrintPoint::EndBlock) {
      c.indent--;
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.if_full_tile"] = [](Operation &op, PrintPoint at,
                                             PrintContext &c) {
    if (at == PrintPoint::Enter) {
      std::string cond =
          "Int(" + c.name(op.getOperand(0)) + ") == " +
          std::to_string(op.getAttrOfType<IntegerAttr>("rows").getInt());
      if (op.getAttrOfType<BoolAttr>("check_cols").getValue())
        cond += " and Int(" + c.name(op.getOperand(1)) + ") == " +
                std::to_string(op.getAttrOfType<IntegerAttr>("cols").getInt());
      c.line("if " + cond + ":");
    } else if (at == PrintPoint::BeginBlock) {
      c.indent++;
    } else if (at == PrintPoint::EndBlock) {
      c.line("return Int32(0)");
      c.indent--;
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.for_range"] = [](Operation &op, PrintPoint at,
                                          PrintContext &c) {
    if (at == PrintPoint::Enter) {
      auto iv = c.name(op.getRegion(0).front().getArgument(0));
      c.line("for " + iv + " in range(" + c.name(op.getOperand(0)) + ", " +
             c.name(op.getOperand(1)) + ", " + c.name(op.getOperand(2)) + "):");
    } else if (at == PrintPoint::BeginBlock) {
      c.indent++;
    } else if (at == PrintPoint::EndBlock) {
      c.indent--;
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.while_lt"] = [](Operation &op, PrintPoint at,
                                         PrintContext &c) {
    auto iv = c.name(op.getRegion(0).front().getArgument(0));
    if (at == PrintPoint::Enter) {
      c.line("var " + iv + " = " + c.name(op.getOperand(0)));
      if (op.getAttrOfType<BoolAttr>("check_step").getValue())
        c.line("while " + iv + " + " + c.name(op.getOperand(2)) +
               " <= " + c.name(op.getOperand(1)) + ":");
      else
        c.line("while " + iv + " < " + c.name(op.getOperand(1)) + ":");
    } else if (at == PrintPoint::BeginBlock) {
      c.indent++;
    } else if (at == PrintPoint::EndBlock) {
      c.line(iv + " += " + c.name(op.getOperand(2)));
      c.indent--;
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.as_int"] = [](Operation &op, PrintPoint at,
                                       PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("var " + c.name(op.getResult(0)) + " = Int(" +
             c.name(op.getOperand(0)) + ")");
    return llvm::Error::success();
  };

  p.operations["mojo_row.address"] = [](Operation &op, PrintPoint at,
                                        PrintContext &c) {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    auto strides = op.getAttrOfType<DenseI64ArrayAttr>("strides");
    int32_t bits = op.getAttrOfType<IntegerAttr>("index_bits").getInt();
    std::string iTy = bits == 32 ? "Int32" : "Int64";
    SmallVector<std::string> terms{iTy + "(0)"};
    for (auto [idx, st] :
         llvm::zip(op.getOperands().drop_front(), strides.asArrayRef()))
      terms.push_back(iTy + "(" + c.name(idx) + ") * " + iTy + "(" +
                      std::to_string(st) + ")");
    c.line("var " + c.name(op.getResult(0)) + " = Int(" +
           llvm::join(terms, " + ") + ")");
    return llvm::Error::success();
  };

  p.operations["mojo_row.stack_alloc"] = [](Operation &op, PrintPoint at,
                                            PrintContext &c) {
    if (at == PrintPoint::Enter) {
      auto dt = op.getAttrOfType<StringAttr>("dtype").getValue();
      auto cnt = op.getAttrOfType<IntegerAttr>("count").getInt();
      c.line("var " + c.name(op.getResult(0)) + " = unsafe_stack_allocation[" +
             std::to_string(cnt) + ", Scalar[DType." + dt + "]]()");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.acc_decl"] = [](Operation &op, PrintPoint at,
                                         PrintContext &c) {
    if (at == PrintPoint::Enter) {
      auto dt = op.getAttrOfType<StringAttr>("dtype").getValue();
      auto w = op.getAttrOfType<IntegerAttr>("width").getInt();
      if (op.getNumOperands() > 0)
        c.line("var " + c.name(op.getResult(0)) + " = " +
               c.name(op.getOperand(0)));
      else
        c.line("var " + c.name(op.getResult(0)) + " = SIMD[DType." + dt + "," +
               std::to_string(w) + "](0)");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.acc_read"] = [](Operation &op, PrintPoint at,
                                         PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("var " + c.name(op.getResult(0)) + " = " +
             c.name(op.getOperand(0)));
    return llvm::Error::success();
  };

  p.operations["mojo_row.acc_write"] = [](Operation &op, PrintPoint at,
                                          PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line(c.name(op.getOperand(0)) + " = " + c.name(op.getOperand(1)));
    return llvm::Error::success();
  };

  p.operations["mojo_row.load_vec"] = [](Operation &op, PrintPoint at,
                                         PrintContext &c) {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    auto ptr = c.name(op.getOperand(0)), addr = c.name(op.getOperand(1));
    auto res = c.name(op.getResult(0));
    auto dt = op.getAttrOfType<StringAttr>("dtype").getValue().str();
    int64_t w = op.getAttrOfType<IntegerAttr>("width").getInt();
    int64_t st = op.getAttrOfType<IntegerAttr>("lane_stride").getInt();
    if (w == 1 || st == 1) {
      c.line("var " + res + " = " + ptr + ".unsafe_offset(" + addr +
             ").unsafe_load[width=" + std::to_string(w) + "]()");
    } else {
      c.line("var " + res + " = SIMD[DType." + dt + "," + std::to_string(w) +
             "](0)");
      c.line("comptime for lane in range(" + std::to_string(w) + "):");
      c.indent++;
      c.line(res + "[lane] = " + ptr + "[unsafe_offset=" + addr + "+lane*" +
             std::to_string(st) + "]");
      c.indent--;
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.load_i32_xor"] = [](Operation &op, PrintPoint at,
                                             PrintContext &c) {
    if (at == PrintPoint::Enter) {
      auto ptr = c.name(op.getOperand(0)), addr = c.name(op.getOperand(1));
      c.line("var " + c.name(op.getResult(0)) +
             " = bitcast[DType.uint8,64](SIMD[DType.int32,16](" + ptr +
             ".unsafe_offset(" + addr +
             ").bitcast[Int32]().unsafe_load() ^ Int32(-2139062144)))");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.load_bitcast_vec"] = [](Operation &op, PrintPoint at,
                                                 PrintContext &c) {
    if (at == PrintPoint::Enter) {
      auto ptr = c.name(op.getOperand(0)), addr = c.name(op.getOperand(1));
      auto st = op.getAttrOfType<StringAttr>("scalar_type").getValue();
      auto w = op.getAttrOfType<IntegerAttr>("width").getInt();
      c.line("var " + c.name(op.getResult(0)) + " = " + ptr +
             ".unsafe_offset(" + addr + ").bitcast[" + st +
             "]().unsafe_load[width=" + std::to_string(w) + "]()");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.splat_cast"] = [](Operation &op, PrintPoint at,
                                           PrintContext &c) {
    if (at == PrintPoint::Enter) {
      auto in = c.name(op.getOperand(0)), res = c.name(op.getResult(0));
      auto dt = op.getAttrOfType<StringAttr>("dtype").getValue();
      auto w = op.getAttrOfType<IntegerAttr>("width").getInt();
      if (op.getAttrOfType<BoolAttr>("from_scalar").getValue())
        c.line("var " + res + " = SIMD[DType." + dt + "," + std::to_string(w) +
               "](" + in + "[0].cast[DType." + dt + "]())");
      else
        c.line("var " + res + " = " + in + ".cast[DType." + dt + "]()");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.store_vec"] = [](Operation &op, PrintPoint at,
                                          PrintContext &c) {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    auto ptr = c.name(op.getOperand(0)), addr = c.name(op.getOperand(1)),
         val = c.name(op.getOperand(2));
    int64_t w = op.getAttrOfType<IntegerAttr>("width").getInt();
    int64_t st = op.getAttrOfType<IntegerAttr>("lane_stride").getInt();
    if (w == 1 || st == 1) {
      c.line(ptr + ".unsafe_offset(" + addr + ").unsafe_store(" + val + ")");
    } else {
      c.line("comptime for lane in range(" + std::to_string(w) + "):");
      c.indent++;
      c.line(ptr + "[unsafe_offset=" + addr + "+lane*" + std::to_string(st) +
             "] = " + val + "[lane]");
      c.indent--;
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.simd_fma"] = [](Operation &op, PrintPoint at,
                                         PrintContext &c) {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    auto acc = c.name(op.getOperand(0)), lhs = c.name(op.getOperand(1)),
         rhs = c.name(op.getOperand(2)), res = c.name(op.getResult(0));
    if (op.getAttrOfType<BoolAttr>("is_int").getValue())
      c.line("var " + res + " = " + acc + " + " + lhs + " * " + rhs);
    else
      c.line("var " + res + " = fma(" + lhs + ", " + rhs + ", " + acc + ")");
    return llvm::Error::success();
  };

  p.operations["mojo_row.vnni_dot4"] = [](Operation &op, PrintPoint at,
                                          PrintContext &c) {
    if (at == PrintPoint::Enter) {
      auto acc = c.name(op.getOperand(0)), lhs = c.name(op.getOperand(1)),
           rhs = c.name(op.getOperand(2));
      c.line("var " + c.name(op.getResult(0)) +
             " = llvm_intrinsic[\"llvm.x86.avx512.vpdpbusd.512\",SIMD[DType."
             "int32,16]](" +
             acc + ", " + lhs + ", " + rhs + ")");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.q4k_superblock"] = [](Operation &op, PrintPoint at,
                                               PrintContext &c) {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    using Q4 = intentdsl::kir::Q4KRecord;
    using Q8 = intentdsl::kir::Q8KRecord;
    auto wp = c.name(op.getOperand(0)), qp = c.name(op.getOperand(1)),
         wb = c.name(op.getOperand(2)), g = c.name(op.getOperand(3)),
         res = c.name(op.getResult(0));
    int64_t bs = op.getAttrOfType<IntegerAttr>("byte_stride").getInt();
    auto wByte = [&](const std::string &off) {
      return "UInt16(" + wp + "[unsafe_offset=" + wb + "+(" + off + ")*" +
             std::to_string(bs) + "])";
    };
    c.line("var " + res + "_ds = " + qp + ".bitcast[Float32]()[unsafe_offset=" +
           g + "*" + std::to_string(Q8::bytes / 4) + "]");
    c.line("var " + res + "_d = Float32(bitcast[DType.float16](" +
           wByte(std::to_string(Q4::d)) + " | (" +
           wByte(std::to_string(Q4::d + 1)) + " << 8)))");
    c.line("var " + res + "_dm = Float32(bitcast[DType.float16](" +
           wByte(std::to_string(Q4::dmin)) + " | (" +
           wByte(std::to_string(Q4::dmin + 1)) + " << 8)))");
    c.line("var " + res + "_S = Int32(0)");
    c.line("var " + res + "_T = Int32(0)");
    c.line("comptime for j in range(8):");
    c.indent++;
    c.line("var l = Int32(" + wByte(std::to_string(Q4::factors) + "+j%4") + ")");
    c.line("var h = Int32(" + wByte(std::to_string(Q4::factors + 4) + "+j%4") +
           ")");
    c.line("var u = Int32(" + wByte(std::to_string(Q4::factors + 8) + "+j%4") +
           ")");
    c.line("var sc = (l&63) if j<4 else (u&15)|((l>>6)<<4)");
    c.line("var mn = (h&63) if j<4 else (u>>4)|((h>>6)<<4)");
    c.line("var q = " + qp + ".unsafe_offset(" + g + "*" +
           std::to_string(Q8::bytes) + "+" + std::to_string(Q8::values) +
           "+j*" + std::to_string(Q4::group) +
           ").unsafe_load[width=32]().cast[DType.int32]()");
    if (bs == 1) {
      c.line("var raw = " + wp + ".unsafe_offset(" + wb + "+" +
             std::to_string(Q4::packed) +
             "+(j//2)*32).unsafe_load[width=32]()");
    } else {
      c.line("var raw = SIMD[DType.uint8,32](0)");
      c.line("comptime for k in range(32):");
      c.indent++;
      c.line("raw[k] = " + wp + "[unsafe_offset=" + wb + "+(" +
             std::to_string(Q4::packed) + "+(j//2)*32+k)*" +
             std::to_string(bs) + "]");
      c.indent--;
    }
    c.line("var nib = ((raw>>SIMD[DType.uint8,32](UInt8(4*(j%2))))&15).cast["
           "DType.int32]()");
    c.line("var sums = " + qp + ".bitcast[Int16]().unsafe_offset(" + g + "*" +
           std::to_string(Q8::bytes / 2) + "+" + std::to_string(Q8::sums / 2) +
           "+2*j)");
    c.line(res + "_S += sc*(q*nib).reduce_add()");
    c.line(res + "_T += mn*(Int32(sums[unsafe_offset=0])+Int32(sums["
                 "unsafe_offset=1]))");
    c.indent--;
    c.line("var " + res + " = SIMD[DType.float32,1](" + res + "_ds*(" + res +
           "_d*Float32(" + res + "_S)-" + res + "_dm*Float32(" + res + "_T)))");
    return llvm::Error::success();
  };

  p.operations["mojo_row.ptr_offset"] = [](Operation &op, PrintPoint at,
                                           PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("var " + c.name(op.getResult(0)) + " = " +
             c.name(op.getOperand(0)) + ".unsafe_offset(" +
             c.name(op.getOperand(1)) + ")");
    return llvm::Error::success();
  };

  p.operations["mojo_row.brgemm_call"] = [](Operation &op, PrintPoint at,
                                            PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("var " + c.name(op.getResult(0)) +
             " = external_call[\"intent_brgemm_tile\", Int32](" +
             c.name(op.getOperand(0)) + ", " + c.name(op.getOperand(1)) + ", " +
             c.name(op.getOperand(2)) + ", " + c.name(op.getOperand(3)) + ", " +
             c.name(op.getOperand(4)) + ")");
    return llvm::Error::success();
  };

  p.operations["mojo_row.check_status"] = [](Operation &op, PrintPoint at,
                                             PrintContext &c) {
    if (at == PrintPoint::Enter) {
      auto st = c.name(op.getOperand(0));
      c.line("if " + st + " != Int32(0):");
      c.indent++;
      c.line("return " + st);
      c.indent--;
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.panel_return"] = [](Operation &op, PrintPoint at,
                                             PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("return Int32(" +
             std::to_string(op.getAttrOfType<IntegerAttr>("code").getInt()) +
             ")");
    return llvm::Error::success();
  };

  p.operations["mojo_row.q4k_unpack_row"] = [](Operation &op, PrintPoint at,
                                               PrintContext &c) {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    using Q4 = intentdsl::kir::Q4KRecord;
    auto wp = c.name(op.getOperand(0)), wb = c.name(op.getOperand(1)),
         ubuf = c.name(op.getOperand(2)), ri = c.name(op.getOperand(3)),
         sub = c.name(op.getOperand(4));
    int64_t bs = op.getAttrOfType<IntegerAttr>("byte_stride").getInt();
    if (bs == 1) {
      c.line("var raw_u = " + wp + ".unsafe_offset(" + wb + "+" +
             std::to_string(Q4::packed) + "+(" + sub +
             "//2)*32).unsafe_load[width=32]()");
    } else {
      c.line("var raw_u = SIMD[DType.uint8,32](0)");
      c.line("comptime for k in range(32):");
      c.indent++;
      c.line("raw_u[k] = " + wp + "[unsafe_offset=" + wb + "+(" +
             std::to_string(Q4::packed) + "+(" + sub + "//2)*32+k)*" +
             std::to_string(bs) + "]");
      c.indent--;
    }
    c.line("var nib_u = (raw_u >> SIMD[DType.uint8,32](UInt8(4*(" + sub +
           "%2)))) & 15");
    c.line(ubuf + ".unsafe_offset(" + ri + "*32).unsafe_store(nib_u)");
    return llvm::Error::success();
  };

  p.operations["mojo_row.q4k_accum_row"] = [](Operation &op, PrintPoint at,
                                              PrintContext &c) {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    using Q4 = intentdsl::kir::Q4KRecord;
    using Q8 = intentdsl::kir::Q8KRecord;
    auto wp = c.name(op.getOperand(0)), wb = c.name(op.getOperand(1)),
         qp = c.name(op.getOperand(2)), sbuf = c.name(op.getOperand(3)),
         abuf = c.name(op.getOperand(4)), ri = c.name(op.getOperand(5)),
         g = c.name(op.getOperand(6)), sub = c.name(op.getOperand(7));
    int64_t bs = op.getAttrOfType<IntegerAttr>("byte_stride").getInt();
    auto wByte = [&](const std::string &off) {
      return "UInt16(" + wp + "[unsafe_offset=" + wb + "+(" + off + ")*" +
             std::to_string(bs) + "])";
    };
    c.line("var ds_a = " + qp + ".bitcast[Float32]()[unsafe_offset=" + g + "*" +
           std::to_string(Q8::bytes / 4) + "]");
    c.line("var d_a = Float32(bitcast[DType.float16](" +
           wByte(std::to_string(Q4::d)) + " | (" +
           wByte(std::to_string(Q4::d + 1)) + " << 8)))");
    c.line("var dm_a = Float32(bitcast[DType.float16](" +
           wByte(std::to_string(Q4::dmin)) + " | (" +
           wByte(std::to_string(Q4::dmin + 1)) + " << 8)))");
    c.line("var l_a = Int32(" + wByte(std::to_string(Q4::factors) + "+" + sub + "%4") + ")");
    c.line("var h_a = Int32(" + wByte(std::to_string(Q4::factors + 4) + "+" + sub + "%4") + ")");
    c.line("var u_a = Int32(" + wByte(std::to_string(Q4::factors + 8) + "+" + sub + "%4") + ")");
    c.line("var sc_a = (l_a&63) if " + sub + "<4 else (u_a&15)|((l_a>>6)<<4)");
    c.line("var mn_a = (h_a&63) if " + sub + "<4 else (u_a>>4)|((h_a>>6)<<4)");
    c.line("var sums_a = " + qp + ".bitcast[Int16]().unsafe_offset(" + g + "*" +
           std::to_string(Q8::bytes / 2) + "+" + std::to_string(Q8::sums / 2) +
           "+2*" + sub + ")");
    c.line("var dot_a = sc_a * " + sbuf + "[unsafe_offset=" + ri + "]");
    c.line("var min_a = mn_a * (Int32(sums_a[unsafe_offset=0]) + Int32(sums_a[unsafe_offset=1]))");
    c.line(abuf + "[unsafe_offset=" + ri + "] += ds_a * (d_a * Float32(dot_a) - dm_a * Float32(min_a))");
    return llvm::Error::success();
  };

  p.operations["mojo_row.epilogue_add"] = [](Operation &op, PrintPoint at,
                                             PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("var " + c.name(op.getResult(0)) + " = " +
             c.name(op.getOperand(0)) + " + " + c.name(op.getOperand(1)));
    return llvm::Error::success();
  };

  p.operations["mojo_row.epilogue_round_bf16"] = [](Operation &op,
                                                    PrintPoint at,
                                                    PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("var " + c.name(op.getResult(0)) + " = round_bf16[" +
             std::to_string(op.getAttrOfType<IntegerAttr>("width").getInt()) +
             "](" + c.name(op.getOperand(0)) + ")");
    return llvm::Error::success();
  };

  p.operations["mojo_row.epilogue_cast"] = [](Operation &op, PrintPoint at,
                                              PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("var " + c.name(op.getResult(0)) + " = " +
             c.name(op.getOperand(0)) + ".cast[DType." +
             op.getAttrOfType<StringAttr>("dtype").getValue() + "]()");
    return llvm::Error::success();
  };

  // ---------------- Host ("entry.c") Ops ----------------
  p.operations["mojo_row.host_fn"] = [](Operation &op, PrintPoint at,
                                        PrintContext &c) {
    if (at == PrintPoint::Enter) {
      c.line("#include <stdint.h>");
      c.line("#include <stddef.h>");
      c.line("#include <math.h>");
      c.line("#include <stdlib.h>");
      c.line("#include <string.h>");
      c.line("#include <omp.h>");
      if (op.getAttrOfType<BoolAttr>("uses_f16_math").getValue() ||
          op.getAttrOfType<BoolAttr>("uses_bf16_math").getValue()) {
        c.line("#if defined(__AVX512F__) || defined(__F16C__)");
        c.line("#include <immintrin.h>");
        c.line("#endif");
      }
      int64_t arity = op.getAttrOfType<IntegerAttr>("panel_arity").getInt();
      bool usesBrg = op.getAttrOfType<BoolAttr>("uses_brgemm").getValue();
      SmallVector<std::string> ptypes(arity, "void*");
      ptypes.push_back("int64_t");
      ptypes.push_back("int64_t");
      if (usesBrg) {
        ptypes.push_back("const void*");
        ptypes.push_back("void*");
        c.line("extern const void *intent_brgemm_get(int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int32_t,int32_t,int32_t);");
        c.line("extern int64_t intent_brgemm_scratch_bytes(const void*);");
        c.line("extern int32_t intent_brgemm_thread_enter(const void*);");
        c.line("extern void intent_brgemm_thread_leave(void);");
      }
      c.line("extern int32_t intent_panel(" + llvm::join(ptypes, ",") + ");");
      c.line("typedef struct { void *ptr; size_t cap; int busy; } IntentWsSlot;");
      c.line("static IntentWsSlot g_intent_ws[4];");
      c.line("static inline void *intent_ws_acquire(int slot, size_t bytes) {");
      c.line("  size_t need = bytes ? ((bytes + 63u) & ~(size_t)63u) : 64u;");
      c.line("  if ((unsigned)slot < 4u && __atomic_exchange_n(&g_intent_ws[slot].busy, 1, __ATOMIC_ACQUIRE) == 0) {");
      c.line("    if (g_intent_ws[slot].cap < need) {");
      c.line("      void *np = aligned_alloc(64, need);");
      c.line("      if (!np) { __atomic_store_n(&g_intent_ws[slot].busy, 0, __ATOMIC_RELEASE); return NULL; }");
      c.line("      free(g_intent_ws[slot].ptr); g_intent_ws[slot].ptr = np; g_intent_ws[slot].cap = need;");
      c.line("    }");
      c.line("    return g_intent_ws[slot].ptr;");
      c.line("  }");
      c.line("  return aligned_alloc(64, need);");
      c.line("}");
      c.line("static inline void intent_ws_release(int slot, void *p) {");
      c.line("  if (!p) return;");
      c.line("  if ((unsigned)slot < 4u && p == g_intent_ws[slot].ptr) { __atomic_store_n(&g_intent_ws[slot].busy, 0, __ATOMIC_RELEASE); return; }");
      c.line("  free(p);");
      c.line("}");

      if (op.getAttrOfType<BoolAttr>("uses_q8k").getValue()) {
        using Q8 = intentdsl::kir::Q8KRecord;
        c.line("struct Record { float d; int8_t q[" +
               std::to_string(Q8::elements) + "]; int16_t sums[" +
               std::to_string(Q8::elements / Q8::sumGroup) + "]; };");
        c.line("_Static_assert(sizeof(struct Record)==" +
               std::to_string(Q8::bytes) + ", \"Q8_K record ABI\");");
        c.line("static inline void pack_q8_record(struct Record *out, const float *x, int64_t stride) {");
        c.line("  float extreme=0.0f;");
        c.line("  for (int64_t k=0; k<256; ++k) { float v=x[k*stride]; if (fabsf(v)>fabsf(extreme)) extreme=v; }");
        c.line("  float inv=extreme==0.0f ? 0.0f : -127.0f/extreme;");
        c.line("  out->d=extreme==0.0f ? 0.0f : 1.0f/inv;");
        c.line("  for (int group=0; group<16; ++group) {");
        c.line("    int sum=0;");
        c.line("    #pragma omp simd reduction(+:sum)");
        c.line("    for (int64_t k=16*group; k<16*group+16; ++k) {");
        c.line("      float v=x[k*stride]*inv; int q=(int)v; float fraction=fabsf(v-(float)q);");
        c.line("      if (fraction>0.5f || (fraction==0.5f && (q&1))) q+=v<0 ? -1 : 1;");
        c.line("      out->q[k]=(int8_t)(q>127 ? 127 : q); sum+=out->q[k];");
        c.line("    }");
        c.line("    out->sums[group]=(int16_t)sum;");
        c.line("  }");
        c.line("}");
      }
      if (op.getAttrOfType<BoolAttr>("uses_bf16_math").getValue()) {
        c.line("static inline float expand_bf(uint16_t v){uint32_t b=(uint32_t)v<<16;float f;memcpy(&f,&b,4);return f;}");
        c.line("static inline uint16_t narrow_bf(float f){uint32_t b;memcpy(&b,&f,4);if((b&0x7fffffff)>0x7f800000)return (b>>16)|64;return (b+32767+((b>>16)&1))>>16;}");
        c.line("static inline float min_ieee(float a,float b){return isnan(a)?a:isnan(b)?b:fminf(a,b);}");
        c.line("static inline void widen_bf16_span(float *out, const uint16_t *src, int64_t n){");
        c.line("  if (n <= 0) return; size_t i=0, un=(size_t)n;");
        c.line("#if defined(__AVX512F__)");
        c.line("  for (; i + 16 <= un; i += 16) { __m256i h = _mm256_loadu_si256((const __m256i*)(src+i)); _mm512_storeu_ps(out+i, _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16))); }");
        c.line("#endif");
        c.line("  for (; i<un; ++i) out[i] = expand_bf(src[i]);");
        c.line("}");
      }
      if (op.getAttrOfType<BoolAttr>("uses_f16_math").getValue()) {
        c.line("static inline float expand_f16(uint16_t h){");
        c.line("#if defined(__F16C__)");
        c.line("  return _cvtsh_ss(h);");
        c.line("#else");
        c.line("  uint32_t sign=(uint32_t)(h&0x8000u)<<16, exp=(h>>10)&0x1fu, mant=h&0x03ffu, bits;");
        c.line("  if(exp==0){ if(mant==0){bits=sign;float z;memcpy(&z,&bits,4);return z;} return (h&0x8000u?-1.0f:1.0f)*(float)mant*(1.0f/16777216.0f); }");
        c.line("  if(exp==31) bits=sign|0x7f800000u|(mant<<13)|(mant?0x00400000u:0u);");
        c.line("  else bits=sign|((exp+112u)<<23)|(mant<<13);");
        c.line("  float f; memcpy(&f,&bits,4); return f;");
        c.line("#endif");
        c.line("}");
        c.line("static inline void widen_f16_span(float *out, const uint16_t *src, int64_t n){");
        c.line("  if (n <= 0) return; size_t i=0, un=(size_t)n;");
        c.line("#if defined(__AVX512F__) && defined(__F16C__)");
        c.line("  for (; i + 16 <= un; i += 16) { __m256i h = _mm256_loadu_si256((const __m256i*)(src+i)); _mm512_storeu_ps(out+i, _mm512_cvtph_ps(h)); }");
        c.line("#endif");
        c.line("  for (; i<un; ++i) out[i] = expand_f16(src[i]);");
        c.line("}");
      }
      if (op.getAttrOfType<BoolAttr>("uses_mx8_math").getValue()) {
        using MX = intentdsl::kir::MX8Block;
        c.line("static inline float decode_mx8(unsigned x,unsigned scale){");
        c.line("  unsigned mag=x&127,ex=mag>>3,m=mag&7,sign=(x&128)<<24;");
        c.line("  uint32_t bits=sign|(mag==127?0x7fc00000U:((ex+120)<<23)|(m<<20));");
        c.line("  float v;if(ex==0)v=(x&128?-1.0f:1.0f)*m*0.001953125f;else memcpy(&v,&bits,4);");
        c.line("  uint32_t sbits=scale==" + std::to_string(MX::nan) +
               "?0x7fc00000U:scale==0?0x00400000U:scale<<23;");
        c.line("  float factor;memcpy(&factor,&sbits,4);return v*factor;");
        c.line("}");
      }

      auto ext = op.getAttrOfType<DenseI64ArrayAttr>("extents");
      auto tl = op.getAttrOfType<DenseI64ArrayAttr>("tile");
      c.line("int intent_entry(" +
             op.getAttrOfType<StringAttr>("params").str() + ") {");
      c.indent++;
      c.line("const int64_t M=" + std::to_string(ext[0]) +
             ", N=" + std::to_string(ext[1]) + ", K=" + std::to_string(ext[2]) +
             ", G=" + std::to_string(ext[3]) + ";");
      c.line("const int64_t mr=" + std::to_string(tl[0]) +
             ", nr=" + std::to_string(tl[1]) + ";");
      c.line("const int64_t mb=(M+mr-1)/mr, nc=(N+nr-1)/nr, jobs=G*mb*nc;");
      c.line("int rc=0;");
      c.indent--;
    } else if (at == PrintPoint::BeginBlock) {
      c.indent++;
    } else if (at == PrintPoint::EndBlock) {
      c.line("return rc;");
      c.indent--;
      c.line("}");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.host_alloc"] = [](Operation &op, PrintPoint at,
                                           PrintContext &c) {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    auto name = op.getAttrOfType<StringAttr>("buf_name").str();
    auto place = op.getAttrOfType<StringAttr>("placement").getValue();
    auto cType = op.getAttrOfType<StringAttr>("c_type").str();
    auto count = op.getAttrOfType<StringAttr>("count_expr").str();
    int wsSlot = name == "left"      ? 0
                 : name == "rhs"     ? 1
                 : name == "members" ? 2
                                     : 3;
    if (place == "stack") {
      c.line("_Alignas(64) " + cType + " " + name + "[" + count + "];");
    } else {
      std::string frees;
      for (Attribute a : op.getAttrOfType<ArrayAttr>("cleanup")) {
        std::string prev = cast<StringAttr>(a).str();
        int pSlot = prev == "left" ? 0 : prev == "rhs" ? 1 : prev == "members" ? 2 : 3;
        frees += "intent_ws_release(" + std::to_string(pSlot) + ", " + prev + ");";
      }
      c.line(cType + " *" + name + " = (" + cType + "*)intent_ws_acquire(" +
             std::to_string(wsSlot) + ", (size_t)(" + count + ") * sizeof(" +
             cType + "));");
      c.line("if (!" + name + ") { " + frees + "return 2; }");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.host_if"] = [](Operation &op, PrintPoint at,
                                        PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("if (" + op.getAttrOfType<StringAttr>("cond").str() + ") {");
    else if (at == PrintPoint::BeginBlock)
      c.indent++;
    else if (at == PrintPoint::EndBlock) {
      c.indent--;
      c.line("}");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.host_omp_parallel"] = [](Operation &op, PrintPoint at,
                                                  PrintContext &c) {
    if (at == PrintPoint::Enter) {
      std::string nt = op.getAttrOfType<StringAttr>("num_threads").str();
      std::string ic = op.getAttrOfType<StringAttr>("if_cond").str();
      std::string red = op.getAttrOfType<BoolAttr>("reduce_rc").getValue()
                            ? " reduction(|:rc)"
                            : "";
      c.line("#pragma omp parallel num_threads(" + nt + ") if(" + ic + ")" +
             red);
      c.line("{");
    } else if (at == PrintPoint::BeginBlock) {
      c.indent++;
    } else if (at == PrintPoint::EndBlock) {
      c.indent--;
      c.line("}");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.host_omp_for"] = [](Operation &op, PrintPoint at,
                                             PrintContext &c) {
    if (at == PrintPoint::Enter) {
      bool par = op.getAttrOfType<BoolAttr>("parallel").getValue();
      bool nw = op.getAttrOfType<BoolAttr>("nowait").getValue();
      bool simd = op.getAttrOfType<BoolAttr>("simd").getValue();
      std::string ic = op.getAttrOfType<StringAttr>("if_cond").str();
      std::string iv = op.getAttrOfType<StringAttr>("iv").str();
      std::string lb = op.getAttrOfType<StringAttr>("lb").str();
      std::string ub = op.getAttrOfType<StringAttr>("ub").str();
      std::string st = op.getAttrOfType<StringAttr>("step").str();
      if (simd)
        c.line("#pragma omp simd");
      else if (par)
        c.line("#pragma omp parallel for num_threads(workers) schedule(static)" +
               (ic.empty() ? "" : " if(" + ic + ")"));
      else
        c.line(std::string("#pragma omp for schedule(static)") +
               (nw ? " nowait" : ""));
      c.line("for (int64_t " + iv + "=" + lb + "; " + iv + "<" + ub + "; " +
             iv + "+=" + st + ") {");
    } else if (at == PrintPoint::BeginBlock) {
      c.indent++;
    } else if (at == PrintPoint::EndBlock) {
      c.indent--;
      c.line("}");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.host_for"] = [](Operation &op, PrintPoint at,
                                         PrintContext &c) {
    if (at == PrintPoint::Enter) {
      std::string iv = op.getAttrOfType<StringAttr>("iv").str();
      std::string lb = op.getAttrOfType<StringAttr>("lb").str();
      std::string ub = op.getAttrOfType<StringAttr>("ub").str();
      std::string st = op.getAttrOfType<StringAttr>("step").str();
      c.line("for (int64_t " + iv + "=" + lb + "; " + iv + "<" + ub + "; " +
             iv + "+=" + st + ") {");
    } else if (at == PrintPoint::BeginBlock) {
      c.indent++;
    } else if (at == PrintPoint::EndBlock) {
      c.indent--;
      c.line("}");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.host_guard_continue"] = [](Operation &op, PrintPoint at,
                                                    PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("if (" + op.getAttrOfType<StringAttr>("cond").str() +
             ") continue;");
    return llvm::Error::success();
  };

  p.operations["mojo_row.host_let_i64"] = [](Operation &op, PrintPoint at,
                                             PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("const int64_t " + op.getAttrOfType<StringAttr>("name").str() +
             " = " + op.getAttrOfType<StringAttr>("expr").str() + ";");
    return llvm::Error::success();
  };

  p.operations["mojo_row.host_load_scalar"] = [](Operation &op, PrintPoint at,
                                                 PrintContext &c) {
    if (at == PrintPoint::Enter) {
      int64_t s = op.getAttrOfType<IntegerAttr>("slot").getInt();
      std::string cTy = op.getAttrOfType<StringAttr>("c_type").str();
      std::string idx = op.getAttrOfType<StringAttr>("index_expr").str();
      c.line("const " + cTy + " " + c.name(op.getResult(0)) +
             " = ((const " + cTy + "*)a" + std::to_string(s) + ")[" + idx +
             "];");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.host_ptr_offset"] = [](Operation &op, PrintPoint at,
                                                PrintContext &c) {
    if (at == PrintPoint::Enter) {
      std::string cTy = op.getAttrOfType<StringAttr>("c_type").str();
      std::string nm = op.getAttrOfType<StringAttr>("name").str();
      std::string bs = op.getAttrOfType<StringAttr>("base_expr").str();
      std::string of = op.getAttrOfType<StringAttr>("offset_expr").str();
      c.line(cTy + " *" + nm + " = (" + cTy + "*)(" + bs + ") + (" + of + ");");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.host_span_op"] = [](Operation &op, PrintPoint at,
                                             PrintContext &c) {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    auto kind = op.getAttrOfType<StringAttr>("op").getValue();
    std::string dst = op.getAttrOfType<StringAttr>("dst_expr").str();
    std::string src = op.getAttrOfType<StringAttr>("src_expr").str();
    std::string cnt = op.getAttrOfType<StringAttr>("count_expr").str();
    int64_t eb = op.getAttrOfType<IntegerAttr>("elem_bytes").getInt();
    if (kind == "widen_bf16")
      c.line("widen_bf16_span(" + dst + ", " + src + ", " + cnt + ");");
    else if (kind == "widen_f16")
      c.line("widen_f16_span(" + dst + ", " + src + ", " + cnt + ");");
    else if (kind == "memcpy")
      c.line("memcpy(" + dst + ", " + src + ", (size_t)(" + cnt + ") * " +
             std::to_string(eb) + "ULL);");
    else if (kind == "zero")
      c.line("memset(" + dst + ", 0, (size_t)(" + cnt + ") * " +
             std::to_string(eb) + "ULL);");
    return llvm::Error::success();
  };

  p.operations["mojo_row.host_elem_op"] = [](Operation &op, PrintPoint at,
                                             PrintContext &c) {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    auto kind = op.getAttrOfType<StringAttr>("op").getValue();
    std::string dst = op.getAttrOfType<StringAttr>("dst_expr").str();
    std::string src = op.getAttrOfType<StringAttr>("src_expr").str();
    std::string aux = op.getAttrOfType<StringAttr>("aux_expr").str();
    int64_t eb = op.getAttrOfType<IntegerAttr>("elem_bytes").getInt();
    if (kind == "expand_bf")
      c.line(dst + " = expand_bf(" + src + ");");
    else if (kind == "expand_f16")
      c.line(dst + " = expand_f16(" + src + ");");
    else if (kind == "assign")
      c.line(dst + " = " + src + ";");
    else if (kind == "memcpy")
      c.line("memcpy(" + dst + ", " + src + ", " + std::to_string(eb) + ");");
    else if (kind == "decode_mx8")
      c.line(dst + " = decode_mx8(" + src + ", " + aux + ");");
    else if (kind == "pack_q8k")
      c.line("pack_q8_record(" + dst + ", " + src + ", " + aux + ");");
    else if (kind == "dot4_quad") {
      auto comma = StringRef(aux).split(',');
      std::string sbn = comma.first.str(), sbk = comma.second.str();
      c.line("int8_t v0 = " + src + "[j*" + sbn + "], v1 = " + src + "[j*" +
             sbn + " + " + sbk + "], v2 = " + src + "[j*" + sbn + " + 2*" +
             sbk + "], v3 = " + src + "[j*" + sbn + " + 3*" + sbk + "];");
      c.line(dst + "[4*j] = (uint8_t)v0; " + dst + "[4*j+1] = (uint8_t)v1; " +
             dst + "[4*j+2] = (uint8_t)v2; " + dst + "[4*j+3] = (uint8_t)v3;");
      c.line("corr[j] -= (uint32_t)(((int)v0 + (int)v1 + (int)v2 + (int)v3) * 128);");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.host_brgemm_plan_decl"] = [](Operation &, PrintPoint at,
                                                      PrintContext &c) {
    if (at == PrintPoint::Enter) {
      c.line("const void *brg_plan[2*(mr+1)]; memset(brg_plan, 0, sizeof(brg_plan));");
      c.line("int64_t brg_scratch_bytes = 64;");
      c.line("uint8_t r_seen[mr+1]; memset(r_seen, 0, sizeof(r_seen));");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.host_brgemm_mark_rows"] = [](Operation &op,
                                                      PrintPoint at,
                                                      PrintContext &c) {
    if (at == PrintPoint::Enter) {
      std::string le = op.getAttrOfType<StringAttr>("len_expr").str();
      c.line("{ const int64_t _len = " + le +
             "; if (_len >= mr) r_seen[mr] = 1; if (_len > 0 && _len % mr) "
             "r_seen[_len % mr] = 1; }");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.host_brgemm_lookup"] = [](Operation &op, PrintPoint at,
                                                   PrintContext &c) {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    int64_t aDt = op.getAttrOfType<IntegerAttr>("a_dt").getInt();
    int64_t bDt = op.getAttrOfType<IntegerAttr>("b_dt").getInt();
    int64_t cDt = op.getAttrOfType<IntegerAttr>("c_dt").getInt();
    std::string lda = op.getAttrOfType<StringAttr>("lda_expr").str();
    std::string ldb = op.getAttrOfType<StringAttr>("ldb_expr").str();
    std::string nE = op.getAttrOfType<StringAttr>("n_expr").str();
    std::string kE = op.getAttrOfType<StringAttr>("k_expr").str();
    std::string frees;
    for (Attribute a : op.getAttrOfType<ArrayAttr>("cleanup")) {
      std::string prev = cast<StringAttr>(a).str();
      int pSlot = prev == "left" ? 0 : prev == "rhs" ? 1 : prev == "members" ? 2 : 3;
      frees += "intent_ws_release(" + std::to_string(pSlot) + ", " + prev + ");";
    }
    c.line("const void *h = intent_brgemm_get(rt, " + nE + ", " + kE + ", " +
           lda + ", " + ldb + ", " + nE + ", " + std::to_string(aDt) + ", " +
           std::to_string(bDt) + ", " + std::to_string(cDt) + ");");
    c.line("int64_t sb = intent_brgemm_scratch_bytes(h);");
    c.line("if (!h || sb < 0) { " + frees + "return 3; }");
    c.line("brg_plan[ci*(mr+1) + rt] = h;");
    c.line("if (sb > brg_scratch_bytes) brg_scratch_bytes = sb;");
    return llvm::Error::success();
  };

  p.operations["mojo_row.host_thread_scratch_alloc"] =
      [](Operation &op, PrintPoint at, PrintContext &c) {
    if (at == PrintPoint::Enter) {
      int64_t sb = op.getAttrOfType<IntegerAttr>("stack_bytes").getInt();
      int64_t al = op.getAttrOfType<IntegerAttr>("alignment").getInt();
      std::string sv = op.getAttrOfType<StringAttr>("size_var").str();
      std::string sbS = std::to_string(sb);
      c.line("_Alignas(" + std::to_string(al) +
             ") unsigned char brg_scratch_stk[" + sbS + "];");
      c.line("void *brg_scratch = " + sv + " <= " + sbS +
             " ? (void*)brg_scratch_stk : aligned_alloc(" + std::to_string(al) +
             ", (size_t)" + sv + ");");
      c.line("int thr_ok = brg_scratch != NULL;");
      c.line("if (!thr_ok) rc |= 2;");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.host_thread_scratch_free"] =
      [](Operation &, PrintPoint at, PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("if (brg_scratch != (void*)brg_scratch_stk) free(brg_scratch);");
    return llvm::Error::success();
  };

  p.operations["mojo_row.host_brgemm_ctx_decl"] = [](Operation &, PrintPoint at,
                                                     PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("const void *brg_active = NULL;");
    return llvm::Error::success();
  };

  p.operations["mojo_row.host_brgemm_ctx_switch"] =
      [](Operation &, PrintPoint at, PrintContext &c) {
    if (at == PrintPoint::Enter) {
      c.line("const void *bh = brg_plan[(C == nr ? 0 : 1)*(mr+1) + R];");
      c.line("if (bh != brg_active) { if (intent_brgemm_thread_enter(bh) != 0) "
             "{ rc |= 3; thr_ok = 0; continue; } brg_active = bh; }");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.host_brgemm_ctx_release"] =
      [](Operation &, PrintPoint at, PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("if (brg_active) intent_brgemm_thread_leave();");
    return llvm::Error::success();
  };

  p.operations["mojo_row.host_tile_coords"] = [](Operation &op, PrintPoint at,
                                                 PrintContext &c) {
    if (at == PrintPoint::Enter) {
      c.line("const int64_t g = job / (nc * mb), rem = job % (nc * mb);");
      c.line("const int64_t c = (rem / mb) * nr;");
      c.line("int64_t r = (rem % mb) * mr, end = M;");
      if (op.getAttrOfType<BoolAttr>("windowed").getValue()) {
        int64_t H = op.getAttrOfType<IntegerAttr>("head_extent").getInt();
        int64_t chunks = op.getAttrOfType<IntegerAttr>("chunk_extent").getInt();
        c.line("const int64_t head=g%" + std::to_string(H) + ", chunk=g/" +
               std::to_string(H) + "%" + std::to_string(chunks) + ", batch=g/" +
               std::to_string(H) + "/" + std::to_string(chunks) + ";");
      }
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.host_grouped_bounds"] =
      [](Operation &op, PrintPoint at, PrintContext &c) {
    if (at == PrintPoint::Enter) {
      int64_t oSlot = op.getAttrOfType<IntegerAttr>("offsets_slot").getInt();
      int64_t oSt = op.getAttrOfType<IntegerAttr>("offsets_stride").getInt();
      c.line("r += ((const int32_t*)a" + std::to_string(oSlot) + ")[g*" +
             std::to_string(oSt) + "LL];");
      c.line("end = ((const int32_t*)a" + std::to_string(oSlot) + ")[(g+1)*" +
             std::to_string(oSt) + "LL];");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.host_route_coords"] = [](Operation &op, PrintPoint at,
                                                  PrintContext &c) {
    if (at == PrintPoint::Enter) {
      if (op.getAttrOfType<BoolAttr>("routed").getValue()) {
        int64_t rSlot = op.getAttrOfType<IntegerAttr>("routes_slot").getInt();
        int64_t rSt = op.getAttrOfType<IntegerAttr>("routes_stride").getInt();
        int64_t mult = op.getAttrOfType<IntegerAttr>("multiplicity").getInt();
        c.line("const int64_t dest = ((const int32_t*)a" +
               std::to_string(rSlot) + ")[r*" + std::to_string(rSt) +
               "LL], src = dest / " + std::to_string(mult) + "LL;");
      } else {
        c.line("const int64_t dest = r, src = r;");
      }
      c.line("const int64_t R = end-r < mr ? end-r : mr, C = N-c < nr ? N-c : nr;");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.host_view_ptr"] = [](Operation &op, PrintPoint at,
                                              PrintContext &c) {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    int64_t idx = op.getAttrOfType<IntegerAttr>("index").getInt();
    auto sk = op.getAttrOfType<StringAttr>("source_kind").getValue();
    auto expr = op.getAttrOfType<StringAttr>("expr").str();
    if (sk == "routed_out") {
      c.line("_Alignas(64) unsigned char routed_out[" + expr +
             "]; void *p" + std::to_string(idx) + " = routed_out;");
    } else {
      c.line("void *p" + std::to_string(idx) + " = (void*)(" + expr + ");");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.host_call_panel"] = [](Operation &op, PrintPoint at,
                                                PrintContext &c) {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    int64_t arity = op.getAttrOfType<IntegerAttr>("arity").getInt();
    bool usesBrg = op.getAttrOfType<BoolAttr>("uses_brgemm").getValue();
    SmallVector<std::string> args;
    for (int64_t i = 0; i < arity; ++i)
      args.push_back("p" + std::to_string(i));
    args.push_back("R");
    args.push_back("C");
    if (usesBrg) {
      args.push_back("bh");
      args.push_back("brg_scratch");
      c.line("const int32_t panel_rc = intent_panel(" +
             llvm::join(args, ", ") + ");");
      c.line("if (panel_rc != 0) { rc |= (int)panel_rc; thr_ok = 0; continue; }");
    } else {
      c.line("const int32_t panel_rc = intent_panel(" +
             llvm::join(args, ", ") + ");");
      c.line("if (panel_rc != 0) { rc |= (int)panel_rc; continue; }");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.host_free"] = [](Operation &op, PrintPoint at,
                                          PrintContext &c) {
    if (at == PrintPoint::Enter) {
      std::string name = op.getAttrOfType<StringAttr>("buf_name").str();
      int wsSlot = name == "left"      ? 0
                   : name == "rhs"     ? 1
                   : name == "members" ? 2
                                       : 3;
      c.line("intent_ws_release(" + std::to_string(wsSlot) + ", " + name + ");");
    }
    return llvm::Error::success();
  };

  // ---------------- Ordered ("row.mojo" & "entry.c") Ops ----------------
  p.operations["mojo_row.owner_fn"] = [](Operation &op, PrintPoint at,
                                         PrintContext &c) {
    if (at == PrintPoint::Enter) {
      c.line("from std.math import exp, sqrt, rsqrt, exp2, log, max, isnan");
      c.line("from std.memory import bitcast");
      if (op.getAttrOfType<BoolAttr>("uses_stack").getValue())
        c.line("from std.memory import unsafe_stack_allocation");
      auto &bb = op.getRegion(0).front();
      auto ptypes = op.getAttrOfType<ArrayAttr>("param_types");
      SmallVector<std::string> params;
      for (size_t i = 0; i < ptypes.size(); ++i) {
        params.push_back(c.name(bb.getArgument(i)) + ": " +
                         cast<StringAttr>(ptypes[i]).getValue().str());
      }
      size_t base = ptypes.size();
      params.push_back(c.name(bb.getArgument(base)) + ": Int64");
      params.push_back(c.name(bb.getArgument(base + 1)) + ": Int64");
      c.line("@export(\"owner_task\")");
      c.line("def owner_task(" + llvm::join(params, ", ") + ") abi(\"C\"):");
    } else if (at == PrintPoint::BeginBlock) {
      c.indent++;
    } else if (at == PrintPoint::EndBlock) {
      c.indent--;
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.owner_coord"] = [](Operation &op, PrintPoint at,
                                            PrintContext &c) {
    if (at == PrintPoint::Enter) {
      int64_t pitch = op.getAttrOfType<IntegerAttr>("pitch").getInt();
      int64_t ext = op.getAttrOfType<IntegerAttr>("extent").getInt();
      std::string base = pitch == 1
                             ? c.name(op.getOperand(0))
                             : "(" + c.name(op.getOperand(0)) + " // " +
                                   std::to_string(pitch) + ")";
      c.line("var " + c.name(op.getResult(0)) + " = " + base + " % " +
             std::to_string(ext));
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.scalar_decl"] = [](Operation &op, PrintPoint at,
                                            PrintContext &c) {
    if (at == PrintPoint::Enter) {
      auto st = op.getAttrOfType<StringAttr>("scalar_type").getValue();
      if (op.getNumOperands() > 0)
        c.line("var " + c.name(op.getResult(0)) + " = " +
               c.name(op.getOperand(0)));
      else
        c.line("var " + c.name(op.getResult(0)) + ": " + st.str());
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.load_scalar"] = [](Operation &op, PrintPoint at,
                                            PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("var " + c.name(op.getResult(0)) + " = " +
             c.name(op.getOperand(0)) + "[unsafe_offset=" +
             c.name(op.getOperand(1)) + "]");
    return llvm::Error::success();
  };

  p.operations["mojo_row.store_scalar"] = [](Operation &op, PrintPoint at,
                                             PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line(c.name(op.getOperand(0)) + "[unsafe_offset=" +
             c.name(op.getOperand(1)) + "] = " + c.name(op.getOperand(2)));
    return llvm::Error::success();
  };

  p.operations["mojo_row.load_slice"] = [](Operation &op, PrintPoint at,
                                           PrintContext &c) {
    if (at == PrintPoint::Enter) {
      int64_t w = op.getAttrOfType<IntegerAttr>("width").getInt();
      c.line("var " + c.name(op.getResult(0)) + " = " +
             c.name(op.getOperand(0)) + ".unsafe_offset(" +
             c.name(op.getOperand(1)) + ").unsafe_load[width=" +
             std::to_string(w) + "]()");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.store_slice"] = [](Operation &op, PrintPoint at,
                                            PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line(c.name(op.getOperand(0)) + ".unsafe_offset(" +
             c.name(op.getOperand(1)) + ").unsafe_store(" +
             c.name(op.getOperand(2)) + ")");
    return llvm::Error::success();
  };

  p.operations["mojo_row.acc_insert_lane"] = [](Operation &op, PrintPoint at,
                                                PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line(c.name(op.getOperand(0)) + "[" + c.name(op.getOperand(1)) +
             "] = " + c.name(op.getOperand(2)));
    return llvm::Error::success();
  };

  p.operations["mojo_row.extract_lane_dyn"] = [](Operation &op, PrintPoint at,
                                                 PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("var " + c.name(op.getResult(0)) + " = " +
             c.name(op.getOperand(0)) + "[" + c.name(op.getOperand(1)) + "]");
    return llvm::Error::success();
  };

  p.operations["mojo_row.splat_scalar"] = [](Operation &op, PrintPoint at,
                                             PrintContext &c) {
    if (at == PrintPoint::Enter) {
      auto dt = op.getAttrOfType<StringAttr>("dtype").getValue();
      int64_t w = op.getAttrOfType<IntegerAttr>("width").getInt();
      c.line("var " + c.name(op.getResult(0)) + " = SIMD[DType." + dt.str() +
             "," + std::to_string(w) + "](" + c.name(op.getOperand(0)) + ")");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.lane_sum"] = [](Operation &op, PrintPoint at,
                                         PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("var " + c.name(op.getResult(0)) + " = " +
             c.name(op.getOperand(0)) + ".reduce_add()");
    return llvm::Error::success();
  };

  p.operations["mojo_row.lane_max"] = [](Operation &op, PrintPoint at,
                                         PrintContext &c) {
    if (at == PrintPoint::Enter) {
      auto in = c.name(op.getOperand(0)), res = c.name(op.getResult(0));
      c.line("var " + res +
             " = bitcast[DType.float32](UInt32(0x7fc00000)) if " + in + ".ne(" +
             in + ").reduce_or() else " + in + ".reduce_max()");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.reduce_sum_16x16"] = [](Operation &op, PrintPoint at,
                                                 PrintContext &c) {
    if (at == PrintPoint::Enter) {
      SmallVector<std::string, 16> args;
      for (Value v : op.getOperands())
        args.push_back(c.name(v));
      c.line("var " + c.name(op.getResult(0)) + " = reduce_sum_16x16(" +
             llvm::join(args, ", ") + ")");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.simd_cmpi"] = [](Operation &op, PrintPoint at,
                                          PrintContext &c) {
    if (at == PrintPoint::Enter) {
      auto pred = op.getAttrOfType<StringAttr>("pred").getValue().str();
      c.line("var " + c.name(op.getResult(0)) + " = " +
             c.name(op.getOperand(0)) + "." + pred + "(" +
             c.name(op.getOperand(1)) + ")");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.pack_apply"] = [](Operation &op, PrintPoint at,
                                           PrintContext &c) {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    SmallVector<std::string> args;
    for (auto v : op.getOperands())
      args.push_back(c.name(v));
    auto inDt = op.getAttrOfType<StringAttr>("in_dtype").getValue().str();
    auto outDt = op.getAttrOfType<StringAttr>("out_dtype").getValue().str();
    auto kind = op.getAttrOfType<StringAttr>("kind").getValue();
    std::string simd = "SIMD[DType." + inDt + "," +
                       std::to_string(args.size()) + "](" +
                       llvm::join(args, ", ") + ")";
    std::string expr =
        kind == "id"     ? simd
        : kind == "exp"  ? ("exp(" + simd + ")")
        : kind == "exp2" ? ("exp2(" + simd + ")")
                         : (simd + ".cast[DType." + outDt + "]()");
    c.line("var " + c.name(op.getResult(0)) + " = " + expr);
    return llvm::Error::success();
  };

  p.operations["mojo_row.extract_lane"] = [](Operation &op, PrintPoint at,
                                             PrintContext &c) {
    if (at == PrintPoint::Enter) {
      int64_t lane = op.getAttrOfType<IntegerAttr>("lane").getInt();
      c.line("var " + c.name(op.getResult(0)) + " = " +
             c.name(op.getOperand(0)) + "[" + std::to_string(lane) + "]");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.c_host_fn"] = [](Operation &op, PrintPoint at,
                                          PrintContext &c) {
    if (at == PrintPoint::Enter) {
      c.line("#include <stdint.h>");
      c.line("#include <stddef.h>");
      c.line(op.getAttrOfType<StringAttr>("extern_decl").str());
      c.line("int intent_entry(" +
             op.getAttrOfType<StringAttr>("params").str() + ") {");
    } else if (at == PrintPoint::BeginBlock) {
      c.indent++;
    } else if (at == PrintPoint::EndBlock) {
      c.line("return 0;");
      c.indent--;
      c.line("}");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.c_omp_for"] = [](Operation &op, PrintPoint at,
                                          PrintContext &c) {
    int64_t count = op.getAttrOfType<IntegerAttr>("count").getInt();
    int64_t grain = op.getAttrOfType<IntegerAttr>("grain").getInt();
    bool cyclic = op.getAttrOfType<BoolAttr>("cyclic").getValue();
    if (at == PrintPoint::Enter) {
      if (cyclic) {
        int64_t chunks = (count + grain - 1) / grain;
        c.line("#pragma omp parallel for num_threads(workers) schedule(dynamic, 1) "
               "if(workers>1 && " + std::to_string(chunks) + ">1)");
        c.line("for (int64_t ch=0; ch<" + std::to_string(chunks) + "LL; ++ch) {");
      } else {
        c.line("#pragma omp parallel for num_threads(workers) schedule(static) "
               "if(workers>1 && " + std::to_string(count) + ">" +
               std::to_string(grain) + ")");
        c.line("for (int64_t lo=0; lo<" + std::to_string(count) +
               "LL; lo+=" + std::to_string(grain) + "LL) {");
      }
    } else if (at == PrintPoint::BeginBlock) {
      c.indent++;
    } else if (at == PrintPoint::EndBlock) {
      c.indent--;
      c.line("}");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.c_owner_call"] = [](Operation &op, PrintPoint at,
                                             PrintContext &c) {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    auto *omp = op.getParentOp();
    int64_t count = omp->getAttrOfType<IntegerAttr>("count").getInt();
    int64_t grain = omp->getAttrOfType<IntegerAttr>("grain").getInt();
    bool cyclic = op.getAttrOfType<BoolAttr>("cyclic").getValue();
    auto callArgs = op.getAttrOfType<StringAttr>("call_args").str();
    std::string loExpr =
        cyclic ? ("ch*" + std::to_string(grain) + "LL") : "lo";
    std::string hiExpr = "(" + std::to_string(count) + "LL-" + loExpr + "<" +
                         std::to_string(grain) + "LL?" +
                         std::to_string(count) + "LL:" + loExpr + "+" +
                         std::to_string(grain) + "LL)";
    c.line("owner_task(" + callArgs + (callArgs.empty() ? "" : ",") + loExpr +
           ", " + hiExpr + ");");
    return llvm::Error::success();
  };

  // ---------------- Row ("row.mojo" & "entry.c") Ops ----------------
  p.operations["mojo_row.row_fn"] = [](Operation &op, PrintPoint at,
                                       PrintContext &c) {
    if (at == PrintPoint::Enter) {
      c.line("from std.math import exp, sqrt, rsqrt, max, isnan");
      c.line("from std.memory import bitcast");
      if (op.getAttrOfType<BoolAttr>("uses_atomic").getValue())
        c.line("from std.atomic import Atomic");
      if (op.getAttrOfType<BoolAttr>("uses_stack").getValue())
        c.line("from std.memory import unsafe_stack_allocation");
      if (op.getAttrOfType<BoolAttr>("uses_memcpy").getValue())
        c.line("from std.memory import unsafe_memcpy");
      auto &bb = op.getRegion(0).front();
      auto specs = op.getAttrOfType<ArrayAttr>("param_specs");
      SmallVector<std::string> params;
      for (size_t i = 0; i < specs.size(); ++i) {
        params.push_back(c.name(bb.getArgument(i)) + ": " +
                         cast<StringAttr>(specs[i]).getValue().str());
      }
      c.line("@export(\"intent_row_interval\")");
      c.line("def intent_row_interval(" + llvm::join(params, ", ") +
             ") abi(\"C\"):");
    } else if (at == PrintPoint::BeginBlock) {
      c.indent++;
    } else if (at == PrintPoint::EndBlock) {
      c.indent--;
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.dyn_address"] = [](Operation &op, PrintPoint at,
                                            PrintContext &c) {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    int64_t terms = op.getAttrOfType<IntegerAttr>("terms").getInt();
    SmallVector<std::string> parts{"Int64(0)"};
    for (int64_t i = 0; i < terms; ++i) {
      parts.push_back("Int64(" + c.name(op.getOperand(1 + i)) + ") * Int64(" +
                      c.name(op.getOperand(1 + terms + i)) + ")");
    }
    c.line("var " + c.name(op.getResult(0)) + " = Int(" +
           llvm::join(parts, " + ") + ")");
    return llvm::Error::success();
  };

  p.operations["mojo_row.atomic_fetch_add_i32"] =
      [](Operation &op, PrintPoint at, PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("var " + c.name(op.getResult(0)) +
             " = Atomic[DType.int32].fetch_add(" + c.name(op.getOperand(0)) +
             ".unsafe_offset(" + c.name(op.getOperand(1)) + "), " +
             c.name(op.getOperand(2)) + ")");
    return llvm::Error::success();
  };

  p.operations["mojo_row.memcpy_elems"] = [](Operation &op, PrintPoint at,
                                             PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("unsafe_memcpy(dest=" + c.name(op.getOperand(0)) +
             ".unsafe_offset(" + c.name(op.getOperand(1)) + "), src=" +
             c.name(op.getOperand(2)) + ".unsafe_offset(" +
             c.name(op.getOperand(3)) + "), count=" + c.name(op.getOperand(4)) +
             ")");
    return llvm::Error::success();
  };

  p.operations["mojo_row.simd_iota_add"] = [](Operation &op, PrintPoint at,
                                              PrintContext &c) {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    auto dt = op.getAttrOfType<StringAttr>("dtype").getValue().str();
    int64_t w = op.getAttrOfType<IntegerAttr>("width").getInt();
    auto base = c.name(op.getOperand(0)), res = c.name(op.getResult(0));
    if (w == 1) {
      c.line("var " + res + " = Scalar[DType." + dt + "](" + base + ")");
    } else {
      SmallVector<std::string> seq;
      for (int64_t i = 0; i < w; ++i)
        seq.push_back(std::to_string(i));
      std::string sTy = "SIMD[DType." + dt + "," + std::to_string(w) + "]";
      c.line("var " + res + " = " + sTy + "(" + base + ") + " + sTy + "(" +
             llvm::join(seq, ",") + ")");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.simd_select"] = [](Operation &op, PrintPoint at,
                                            PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("var " + c.name(op.getResult(0)) + " = " +
             c.name(op.getOperand(0)) + ".select(" + c.name(op.getOperand(1)) +
             ", " + c.name(op.getOperand(2)) + ")");
    return llvm::Error::success();
  };

  p.operations["mojo_row.simd_cmp_gt"] = [](Operation &op, PrintPoint at,
                                            PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("var " + c.name(op.getResult(0)) + " = " +
             c.name(op.getOperand(0)) + ".gt(" + c.name(op.getOperand(1)) + ")");
    return llvm::Error::success();
  };

  p.operations["mojo_row.scalar_cast"] = [](Operation &op, PrintPoint at,
                                            PrintContext &c) {
    if (at == PrintPoint::Enter) {
      auto dt = op.getAttrOfType<StringAttr>("dtype").getValue().str();
      auto in = c.name(op.getOperand(0)), res = c.name(op.getResult(0));
      Type inTy = op.getOperand(0).getType(), outTy = op.getResult(0).getType();
      if (isa<ShapedType>(outTy)) {
        c.line("var " + res + " = " + in + ".cast[DType." + dt + "]()");
      } else if (outTy.isInteger(1)) {
        c.line("var " + res + " = Bool(" + in + " != 0)");
      } else if (inTy.isInteger(1)) {
        c.line("var " + res + " = Scalar[DType." + dt + "](1) if " + in +
               " else Scalar[DType." + dt + "](0)");
      } else if (inTy.isIndex()) {
        c.line("var " + res + " = Int64(" + in + ").cast[DType." + dt + "]()");
      } else {
        c.line("var " + res + " = " + in + ".cast[DType." + dt + "]()");
      }
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.scalar_const"] = [](Operation &op, PrintPoint at,
                                             PrintContext &c) {
    if (at == PrintPoint::Enter) {
      auto dt = op.getAttrOfType<StringAttr>("dtype").getValue().str();
      auto lit = op.getAttrOfType<StringAttr>("literal").getValue().str();
      c.line("var " + c.name(op.getResult(0)) + " = Scalar[DType." + dt +
             "](" + lit + ")");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.tuple_decl"] = [](Operation &op, PrintPoint at,
                                           PrintContext &c) {
    if (at == PrintPoint::Enter) {
      if (op.getNumOperands() == 3)
        c.line("var " + c.name(op.getResult(0)) + " = (" +
               c.name(op.getOperand(0)) + ", " + c.name(op.getOperand(1)) +
               ", " + c.name(op.getOperand(2)) + ")");
      else
        c.line("var " + c.name(op.getResult(0)) + " = " +
               c.name(op.getOperand(0)));
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.tuple_read"] = [](Operation &op, PrintPoint at,
                                           PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("var " + c.name(op.getResult(0)) + " = " +
             c.name(op.getOperand(0)));
    return llvm::Error::success();
  };

  p.operations["mojo_row.tuple_write"] = [](Operation &op, PrintPoint at,
                                            PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line(c.name(op.getOperand(0)) + " = " + c.name(op.getOperand(1)));
    return llvm::Error::success();
  };

  p.operations["mojo_row.tuple_field"] = [](Operation &op, PrintPoint at,
                                            PrintContext &c) {
    if (at == PrintPoint::Enter) {
      int64_t pos = op.getAttrOfType<IntegerAttr>("position").getInt();
      c.line("var " + c.name(op.getResult(0)) + " = " +
             c.name(op.getOperand(0)) + "[" + std::to_string(pos) + "]");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.uniform"] = [](Operation &op, PrintPoint at,
                                        PrintContext &c) {
    if (at == PrintPoint::Enter) {
      int64_t w = op.getAttrOfType<IntegerAttr>("width").getInt();
      if (w > 1 && !isa<ShapedType>(op.getOperand(1).getType())) {
        c.line("var " + c.name(op.getResult(0)) + " = uniform_contiguous[" +
               std::to_string(w) + "](" + c.name(op.getOperand(0)) + ", " +
               c.name(op.getOperand(1)) + ")");
      } else {
        std::string pos =
            w == 1 ? ("SIMD[DType.uint64,1](" + c.name(op.getOperand(1)) + ")")
                   : c.name(op.getOperand(1));
        c.line("var " + c.name(op.getResult(0)) + " = uniform_at[" +
               std::to_string(w) + "](" + c.name(op.getOperand(0)) + ", " +
               pos + ")");
      }
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.welford_vec_step"] = [](Operation &op, PrintPoint at,
                                                 PrintContext &c) {
    if (at == PrintPoint::Enter) {
      int64_t w = op.getAttrOfType<IntegerAttr>("width").getInt();
      auto r0 = c.name(op.getResult(0)), r1 = c.name(op.getResult(1));
      c.line("var " + r0 + "_p = welford_vec_step[" + std::to_string(w) + "](" +
             c.name(op.getOperand(0)) + ", " + c.name(op.getOperand(1)) + ", " +
             c.name(op.getOperand(2)) + ", " + c.name(op.getOperand(3)) + ")");
      c.line("var " + r0 + " = " + r0 + "_p[0]");
      c.line("var " + r1 + " = " + r0 + "_p[1]");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.welford_vec_reduce"] = [](Operation &op, PrintPoint at,
                                                   PrintContext &c) {
    if (at == PrintPoint::Enter) {
      int64_t w = op.getAttrOfType<IntegerAttr>("width").getInt();
      c.line("var " + c.name(op.getResult(0)) + " = welford_vec_reduce[" +
             std::to_string(w) + "](" + c.name(op.getOperand(0)) + ", " +
             c.name(op.getOperand(1)) + ", " + c.name(op.getOperand(2)) + ")");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.welford_scalar_step"] =
      [](Operation &op, PrintPoint at, PrintContext &c) {
    if (at == PrintPoint::Enter) {
      auto st = c.name(op.getOperand(0));
      c.line("var " + c.name(op.getResult(0)) + " = welford_scalar_step(" + st +
             "[0], " + st + "[1], " + st + "[2], " + c.name(op.getOperand(1)) +
             ")");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.transpose_8x8"] = [](Operation &op, PrintPoint at,
                                              PrintContext &c) {
    if (at == PrintPoint::Enter) {
      auto dt = op.getAttrOfType<StringAttr>("dtype").getValue().str();
      c.line("transpose_tile_8x8[DType." + dt + "](" +
             c.name(op.getOperand(0)) + ", " + c.name(op.getOperand(1)) + ", " +
             c.name(op.getOperand(2)) + ", " + c.name(op.getOperand(3)) + ", " +
             c.name(op.getOperand(4)) + ", " + c.name(op.getOperand(5)) + ")");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.simd_scan_sum"] = [](Operation &op, PrintPoint at,
                                              PrintContext &c) {
    if (at == PrintPoint::Enter) {
      auto dt = op.getAttrOfType<StringAttr>("dtype").getValue().str();
      int64_t w = op.getAttrOfType<IntegerAttr>("width").getInt();
      auto r0 = c.name(op.getResult(0)), r1 = c.name(op.getResult(1));
      std::string inExpr = w == 1 ? ("SIMD[DType." + dt + ",1](" +
                                     c.name(op.getOperand(0)) + ")")
                                  : c.name(op.getOperand(0));
      c.line("var " + r0 + "_p = simd_scan_sum[DType." + dt + "," +
             std::to_string(w) + "](" + inExpr + ", Scalar[DType." + dt + "](" +
             c.name(op.getOperand(1)) + "))");
      c.line("var " + r0 + " = " + (w == 1 ? (r0 + "_p[0][0]") : (r0 + "_p[0]")));
      c.line("var " + r1 + " = " + r0 + "_p[1]");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.c_row_host_fn"] = [](Operation &op, PrintPoint at,
                                              PrintContext &c) {
    if (at == PrintPoint::Enter) {
      c.line("#include <stdint.h>");
      c.line("#include <omp.h>");
      c.line(op.getAttrOfType<StringAttr>("extern_decl").str());
      c.line("int intent_entry(" +
             op.getAttrOfType<StringAttr>("params").str() + ") {");
      c.indent++;
      c.line("if (rows == 0) return 0;");
      c.indent--;
    } else if (at == PrintPoint::BeginBlock) {
      c.indent++;
    } else if (at == PrintPoint::EndBlock) {
      c.line("return 0;");
      c.indent--;
      c.line("}");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.c_row_prep"] = [](Operation &op, PrintPoint at,
                                           PrintContext &c) {
    if (at == PrintPoint::Enter) {
      if (op.getAttrOfType<BoolAttr>("single_dispatcher").getValue())
        c.line("workers = 1;");
      if (op.getAttrOfType<BoolAttr>("member_tasks").getValue())
        c.line("rows = cols;");
      if (op.getAttrOfType<BoolAttr>("swap_dest").getValue()) {
        c.line("int64_t t = rows; rows = cols; cols = t;");
        c.line("t = a0s0; a0s0 = a0s1; a0s1 = t;");
        c.line("t = a1s0; a1s0 = a1s1; a1s1 = t;");
      }
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.c_key_omp_for"] = [](Operation &op, PrintPoint at,
                                              PrintContext &c) {
    int64_t keys = op.getAttrOfType<IntegerAttr>("key_extent").getInt();
    if (at == PrintPoint::Enter) {
      c.line("const int64_t keys = " + std::to_string(keys) +
             "LL, lines = (keys + 15) / 16, n = workers < lines ? workers : lines;");
      c.line("#pragma omp parallel for num_threads(n) schedule(static) if(n>1)");
      c.line("for (int64_t b = 0; b < n; ++b) {");
    } else if (at == PrintPoint::BeginBlock) {
      c.indent++;
    } else if (at == PrintPoint::EndBlock) {
      c.indent--;
      c.line("}");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.c_tile_2d_omp"] = [](Operation &op, PrintPoint at,
                                              PrintContext &c) {
    int64_t step = op.getAttrOfType<IntegerAttr>("tile").getInt();
    std::string st = std::to_string(step) + "LL";
    if (at == PrintPoint::Enter) {
      c.line("const int64_t nt = (cols - 1) / " + st +
             " + 1, mt = (rows - 1) / " + st + " + 1;");
      c.line("#pragma omp parallel for schedule(static) num_threads(workers) "
             "if(workers>1)");
      c.line("for (int64_t g = 0; g < mt * nt; ++g) {");
    } else if (at == PrintPoint::BeginBlock) {
      c.indent++;
      c.line("const int64_t r = (g / nt) * " + st + ", c = (g % nt) * " + st + ";");
      c.line("const int64_t nr = rows - r < " + st + " ? rows - r : " + st +
             ", nc = cols - c < " + st + " ? cols - c : " + st + ";");
    } else if (at == PrintPoint::EndBlock) {
      c.indent--;
      c.line("}");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.c_row_omp_parallel"] = [](Operation &, PrintPoint at,
                                                   PrintContext &c) {
    if (at == PrintPoint::Enter) {
      c.line("const int64_t groups = (rows - 1) / grain + 1;");
      c.line("const int64_t threads = groups < workers ? groups : workers;");
      c.line("#pragma omp parallel num_threads(threads) if(threads>1)");
      c.line("{");
    } else if (at == PrintPoint::BeginBlock) {
      c.indent++;
      c.line("int64_t b = omp_get_thread_num(), nt = omp_get_num_threads();");
      c.line("int64_t q = groups / nt, rem = groups % nt;");
      c.line("int64_t start = (q * b + (b < rem ? b : rem)) * grain;");
      c.line("int64_t stop = b + 1 == nt ? rows : start + (q + (b < rem)) * grain;");
    } else if (at == PrintPoint::EndBlock) {
      c.indent--;
      c.line("}");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.c_flat_col_omp"] = [](Operation &op, PrintPoint at,
                                               PrintContext &c) {
    int64_t cg = op.getAttrOfType<IntegerAttr>("col_grain").getInt();
    std::string cgs = std::to_string(cg) + "LL";
    if (at == PrintPoint::Enter) {
      c.line("if (cols <= 0) return 0;");
      c.line("const int64_t cgroups = (cols - 1) / " + cgs + " + 1;");
      c.line("const int64_t threads = (rows == 1 && cgroups < workers) ? cgroups : (rows == 1 ? workers : 1);");
      c.line("#pragma omp parallel num_threads(threads) if(threads>1)");
      c.line("{");
    } else if (at == PrintPoint::BeginBlock) {
      c.indent++;
      c.line("int64_t b = omp_get_thread_num(), nt = omp_get_num_threads();");
      c.line("int64_t q = cgroups / nt, rem = cgroups % nt;");
      c.line("int64_t cstart = (q * b + (b < rem ? b : rem)) * " + cgs + ";");
      c.line("int64_t cstop = b + 1 == nt ? cols : cstart + (q + (b < rem)) * " + cgs + ";");
    } else if (at == PrintPoint::EndBlock) {
      c.indent--;
      c.line("}");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.c_projected_while"] = [](Operation &op, PrintPoint at,
                                                  PrintContext &c) {
    int64_t pRank = op.getAttrOfType<IntegerAttr>("proj_rank").getInt();
    if (at == PrintPoint::Enter) {
      c.line("while (start < stop) {");
    } else if (at == PrintPoint::BeginBlock) {
      c.indent++;
      c.line("int64_t outer = start;");
      for (int64_t ax = pRank - 2; ax >= 0; --ax)
        c.line("int64_t c" + std::to_string(ax) + " = outer % a0d" +
               std::to_string(ax) + "; outer /= a0d" + std::to_string(ax) + ";");
      c.line("int64_t span = a0d" + std::to_string(pRank - 2) + " - c" +
             std::to_string(pRank - 2) + ";");
      c.line("int64_t end = stop - start < span ? stop : start + span;");
    } else if (at == PrintPoint::EndBlock) {
      c.line("start = end;");
      c.indent--;
      c.line("}");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.c_row_call"] = [](Operation &op, PrintPoint at,
                                           PrintContext &c) {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    auto mode = op.getAttrOfType<StringAttr>("mode").getValue();
    auto args = op.getAttrOfType<StringAttr>("call_args").str();
    int64_t bytes = op.getAttrOfType<IntegerAttr>("elem_bytes").getInt();
    std::string pre = args.empty() ? "" : (args + ", ");
    if (mode == "key") {
      c.line("intent_row_interval(" + pre +
             "0, rows, cols, columns, 16*(lines*b/n), 16*(lines*(b+1)/n));");
    } else if (mode == "tile") {
      std::string bs = std::to_string(bytes) + "LL";
      c.line("intent_row_interval((char*)a0+(r*a0s0+c*a0s1)*" + bs +
             ", a0s0, a0s1, (char*)a1+(c*a1s0+r*a1s1)*" + bs +
             ", a1s0, a1s1, 0, nr, nc, columns);");
    } else if (mode == "projected") {
      c.line("intent_row_interval(" + pre + "start, end, cols, columns);");
    } else if (mode == "members") {
      c.line("intent_row_interval(" + pre +
             "start, start + 1, stop - start, columns);");
    } else if (mode == "flat_cols") {
      c.line("intent_row_interval(" + pre +
             "0, rows, cstop - cstart, columns);");
    } else {
      c.line("intent_row_interval(" + pre + "start, stop, cols, columns);");
    }
    return llvm::Error::success();
  };

  p.package = [](DictionaryAttr attrs,
                 const std::map<std::string, std::string> &files,
                 llvm::raw_pwrite_stream &out) -> llvm::Error {
    auto pkgAttr = attrs.getAs<StringAttr>("mojo_row.package");
    if (!pkgAttr)
      return llvm::createStringError("missing mojo_row.package attribute");
    auto parsed = llvm::json::parse(pkgAttr.getValue());
    if (!parsed)
      return parsed.takeError();
    auto *obj = parsed->getAsObject();
    (*obj)["source"] = files.at("row.mojo");
    (*obj)["host"] = files.at("entry.c");
    out << llvm::json::Value(std::move(*obj));
    return llvm::Error::success();
  };

  return p;
}
} // namespace CI_UNIT_NAMESPACE
