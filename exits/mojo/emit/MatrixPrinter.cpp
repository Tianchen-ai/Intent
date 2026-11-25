#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/Storage.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"

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
  auto &rules = p.operations;

  auto enterLocal =
      [&](StringRef name,
          std::function<llvm::Error(const LocalOpView &, PrintContext &)> fn) {
        rules[name.str()] = PrintRule::local(
            [fn = std::move(fn)](const LocalOpView &view, PrintPoint pt,
                                 PrintContext &c) -> llvm::Error {
              if (pt != PrintPoint::Enter)
                return llvm::Error::success();
              return fn(view, c);
            });
      };

  // ---------------- Declarative Format & RegionFormat Rules ----------------
  rules["builtin.module"] = PrintRule::format("");
  rules["func.func"] = PrintRule::format("");
  rules["func.return"] = PrintRule::format("");
  rules["mojo_row.end"] = PrintRule::format("");

  rules["arith.select"] = PrintRule::format("var $r0 = ($1 if $0 else $2)");
  rules["arith.minsi"] = PrintRule::format("var $r0 = min($0, $1)");
  rules["arith.maxsi"] = PrintRule::format("var $r0 = ($0 if $0 > $1 else $1)");
  rules["arith.minimumf"] =
      PrintRule::format("const float $r0 = min_ieee($0, $1);");
  rules["math.log"] = PrintRule::format("var $r0 = log($0)");
  rules["math.sqrt"] = PrintRule::format("var $r0 = sqrt($0)");
  rules["math.rsqrt"] = PrintRule::format("var $r0 = rsqrt($0)");

  rules["mojo_row.as_int"] = PrintRule::format("var $r0 = Int($0)");
  rules["mojo_row.stack_alloc"] = PrintRule::format(
      "var $r0 = unsafe_stack_allocation[{count}, Scalar[DType.{dtype}], alignment=64]()");
  rules["mojo_row.acc_read"] = PrintRule::format("var $r0 = $0");
  rules["mojo_row.acc_write"] = PrintRule::format("$0 = $1");
  rules["mojo_row.load_i32_xor"] = PrintRule::format(
      "var $r0 = bitcast[DType.uint8,64](SIMD[DType.int32,16]($0.unsafe_offset($1).bitcast[Int32]().unsafe_load() ^ Int32(-2139062144)))");
  rules["mojo_row.load_bitcast_vec"] = PrintRule::format(
      "var $r0 = $0.unsafe_offset($1).bitcast[{scalar_type}]().unsafe_load[width={width}]()");
  rules["mojo_row.splat_lane"] =
      PrintRule::format("var $r0 = SIMD[DType.{dtype},{width}]($0[{lane}])");
  rules["mojo_row.vnni_dot4"] = PrintRule::format(
      "var $r0 = llvm_intrinsic[\"llvm.x86.avx512.vpdpbusd.512\",SIMD[DType.int32,16]]($0, $1, $2)");
  rules["mojo_row.ptr_offset"] =
      PrintRule::format("var $r0 = $0.unsafe_offset($1)");
  rules["mojo_row.brgemm_call"] = PrintRule::format(
      "var $r0 = external_call[\"intent_brgemm_tile\", Int32]($0, $1, $2, $3, $4)");
  rules["mojo_row.panel_return"] = PrintRule::format("return Int32({code})");
  rules["mojo_row.epilogue_add"] = PrintRule::format("var $r0 = $0 + $1");
  rules["mojo_row.epilogue_round_bf16"] =
      PrintRule::format("var $r0 = round_bf16[{width}]($0)");
  rules["mojo_row.epilogue_cast"] =
      PrintRule::format("var $r0 = $0.cast[DType.{dtype}]()");

  rules["mojo_row.host_if"] =
      PrintRule::regionFormat("if ({cond}) {{", "", "}}", "");
  rules["mojo_row.host_for"] = PrintRule::regionFormat(
      "for (int64_t {iv}={lb}; {iv}<{ub}; {iv}+={step}) {{", "", "}}", "");
  rules["mojo_row.host_guard_continue"] =
      PrintRule::format("if ({cond}) continue;");
  rules["mojo_row.host_let_i64"] =
      PrintRule::format("const int64_t {name} = {expr};");
  rules["mojo_row.host_ptr_offset"] = PrintRule::format(
      "{c_type} *{name} = ({c_type}*)({base_expr}) + ({offset_expr});");
  rules["mojo_row.host_thread_scratch_free"] = PrintRule::format(
      "if (brg_scratch != (void*)brg_scratch_stk) free(brg_scratch);");
  rules["mojo_row.host_brgemm_ctx_decl"] =
      PrintRule::format("const void *brg_active = NULL;");
  rules["mojo_row.host_brgemm_acc_call"] = PrintRule::format(
      "if (intent_brgemm_tile(bh, {a_expr}, {b_expr}, {c_expr}, brg_scratch) != 0) {{ rc |= 4; thr_ok = 0; continue; }}");
  rules["mojo_row.host_brgemm_ctx_release"] =
      PrintRule::format("if (brg_active) intent_brgemm_thread_leave();");
  rules["mojo_row.host_brgemm_mark_rows"] = PrintRule::format(
      "{{ const int64_t _len = {len_expr}; if (_len >= mr) r_seen[mr] = 1; if (_len > 0 && _len % mr) r_seen[_len % mr] = 1; }}");

  rules["mojo_row.load_scalar"] =
      PrintRule::format("var $r0 = $0[unsafe_offset=$1]");
  rules["mojo_row.store_scalar"] =
      PrintRule::format("$0[unsafe_offset=$1] = $2");
  rules["mojo_row.load_slice"] = PrintRule::format(
      "var $r0 = $0.unsafe_offset($1).unsafe_load[width={width}]()");
  rules["mojo_row.gather_slice"] =
      PrintRule::format("var $r0 = $0.gather($1)");
  rules["mojo_row.store_slice"] =
      PrintRule::format("$0.unsafe_offset($1).unsafe_store($2)");
  rules["mojo_row.acc_insert_lane"] = PrintRule::format("$0[$1] = $2");
  rules["mojo_row.extract_lane_dyn"] = PrintRule::format("var $r0 = $0[$1]");
  rules["mojo_row.splat_scalar"] =
      PrintRule::format("var $r0 = SIMD[DType.{dtype},{width}]($0)");
  rules["mojo_row.lane_sum"] = PrintRule::format("var $r0 = $0.reduce_add()");
  rules["mojo_row.lane_max"] = PrintRule::format(
      "var $r0 = bitcast[DType.float32](UInt32(0x7fc00000)) if $0.ne($0).reduce_or() else $0.reduce_max()");
  rules["mojo_row.reduce_sum_16x16"] =
      PrintRule::format("var $r0 = reduce_sum_16x16($operands)");
  rules["mojo_row.simd_cmpi"] = PrintRule::format("var $r0 = $0.{pred}($1)");
  rules["mojo_row.extract_lane"] = PrintRule::format("var $r0 = $0[{lane}]");
  rules["mojo_row.atomic_fetch_add_i32"] = PrintRule::format(
      "var $r0 = Atomic[DType.int32].fetch_add($0.unsafe_offset($1), $2)");
  rules["mojo_row.memcpy_elems"] = PrintRule::format(
      "unsafe_memcpy(dest=$0.unsafe_offset($1), src=$2.unsafe_offset($3), count=$4)");
  rules["mojo_row.simd_select"] =
      PrintRule::format("var $r0 = $0.select($1, $2)");
  rules["mojo_row.simd_cmp_gt"] = PrintRule::format("var $r0 = $0.gt($1)");
  rules["mojo_row.scalar_const"] =
      PrintRule::format("var $r0 = Scalar[DType.{dtype}]({literal})");
  rules["mojo_row.tuple_read"] = PrintRule::format("var $r0 = $0");
  rules["mojo_row.tuple_write"] = PrintRule::format("$0 = $1");
  rules["mojo_row.tuple_field"] = PrintRule::format("var $r0 = $0[{position}]");
  rules["mojo_row.welford_vec_reduce"] = PrintRule::format(
      "var $r0 = welford_vec_reduce[{width}]($0, $1, $2)");
  rules["mojo_row.welford_scalar_step"] = PrintRule::format(
      "var $r0 = welford_scalar_step($0[0], $0[1], $0[2], $1)");
  rules["mojo_row.transpose_8x8"] = PrintRule::format(
      "transpose_tile_8x8[DType.{dtype}]($0, $1, $2, $3, $4, $5)");

  // ---------------- Strictly Local Rules (LocalOpView) ----------------
  auto twoFileProgram = PrintRule::local(
      [](const LocalOpView &, PrintPoint at, PrintContext &c) -> llvm::Error {
        if (at == PrintPoint::BeginRegion) {
          c.file = c.region == 0 ? "row.mojo" : "entry.c";
          c.indent = 0;
        }
        return llvm::Error::success();
      });
  rules["mojo_row.matrix_program"] = twoFileProgram;
  rules["mojo_row.ordered_program"] = twoFileProgram;
  rules["mojo_row.row_program"] = twoFileProgram;

  enterLocal("mojo_row.runtime_source",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto key =
                   view.attrOfType<StringAttr>("implementation").getValue();
               auto src = c.source(key);
               if (!src)
                 return src.takeError();
               c.line(*src);
               return llvm::Error::success();
             });

  enterLocal("arith.constant",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               std::string res = view.result(0).str();
               if (c.file == "entry.c") {
                 llvm::SmallString<24> num;
                 view.attrOfType<FloatAttr>("value").getValue().toString(num);
                 std::string s = num.str().str();
                 if (s.find_first_of(".Ee") == std::string::npos)
                   s += ".0";
                 c.line("const float " + res + " = " + s + "f;");
                 return llvm::Error::success();
               }
               Type t = view.resultType(0);
               if (auto fAttr = view.attrOfType<FloatAttr>("value")) {
                 auto apf = fAttr.getValue();
                 std::string expr;
                 if (apf.isInfinity()) {
                   expr = apf.isNegative()
                              ? "bitcast[DType.float32](UInt32(0xff800000))"
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
                 c.line("var " + res + " = " + expr);
                 return llvm::Error::success();
               }
               auto iAttr = view.attrOfType<IntegerAttr>("value");
               if (t.isInteger(1)) {
                 c.line("var " + res + " = " +
                        (iAttr.getInt() ? "True" : "False"));
                 return llvm::Error::success();
               }
               if (t.isUnsignedInteger(64)) {
                 c.line("var " + res + " = UInt64(" +
                        std::to_string(iAttr.getUInt()) + ")");
                 return llvm::Error::success();
               }
               c.line("var " + res + " = " + mojoScalarType(t) + "(" +
                      std::to_string(iAttr.getInt()) + ")");
               return llvm::Error::success();
             });

  auto binIdx = [&](StringRef opName, StringRef mojoSym, StringRef cSym) {
    enterLocal(opName,
               [mojoSym, cSym](const LocalOpView &view,
                               PrintContext &c) -> llvm::Error {
                 auto lhs = view.operand(0).str(), rhs = view.operand(1).str();
                 auto res = view.result(0).str();
                 if (c.file == "entry.c")
                   c.line("const float " + res + " = " + lhs + " " +
                          cSym.str() + " " + rhs + ";");
                 else
                   c.line("var " + res + " = (" + lhs + " " + mojoSym.str() +
                          " " + rhs + ")");
                 return llvm::Error::success();
               });
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

  enterLocal("arith.andi",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               std::string opSym =
                   view.resultType(0).isInteger(1) ? "and" : "&";
               c.line("var " + view.result(0).str() + " = (" +
                      view.operand(0).str() + " " + opSym + " " +
                      view.operand(1).str() + ")");
               return llvm::Error::success();
             });

  enterLocal("arith.cmpf",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto predAttr = view.attrOfType<IntegerAttr>("predicate");
               if (!predAttr)
                 return llvm::createStringError("arith.cmpf missing predicate");
               auto pred = static_cast<arith::CmpFPredicate>(predAttr.getInt());
               StringRef tok;
               switch (pred) {
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
                 return llvm::createStringError(
                     "unsupported arith.cmpf predicate");
               }
               c.line("var " + view.result(0).str() + " = (" +
                      view.operand(0).str() + " " + tok.str() + " " +
                      view.operand(1).str() + ")");
               return llvm::Error::success();
             });

  enterLocal("arith.cmpi",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto predAttr = view.attrOfType<IntegerAttr>("predicate");
               if (!predAttr)
                 return llvm::createStringError("arith.cmpi missing predicate");
               auto pred = static_cast<arith::CmpIPredicate>(predAttr.getInt());
               StringRef tok;
               switch (pred) {
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
               c.line("var " + view.result(0).str() + " = (" +
                      view.operand(0).str() + " " + tok.str() + " " +
                      view.operand(1).str() + ")");
               return llvm::Error::success();
             });

  enterLocal("arith.index_cast",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               c.line("var " + view.result(0).str() + " = " +
                      mojoScalarType(view.resultType(0)) + "(" +
                      view.operand(0).str() + ")");
               return llvm::Error::success();
             });

  enterLocal("arith.maxnumf",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto lhs = view.operand(0).str(), rhs = view.operand(1).str(),
                    res = view.result(0).str();
               if (isa<ShapedType>(view.resultType(0)))
                 c.line("var " + res + " = " + lhs + ".ne(" + lhs +
                        ").select(" + lhs + ", " + rhs + ".ne(" + rhs +
                        ").select(" + rhs + ", max(" + lhs + ", " + rhs + ")))");
               else
                 c.line("var " + res + " = " + lhs + " if isnan(" + rhs +
                        ") else (" + rhs + " if isnan(" + lhs +
                        ") else max(" + lhs + ", " + rhs + "))");
               return llvm::Error::success();
             });

  enterLocal("arith.extf",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               if (c.file == "entry.c") {
                 c.line("const float " + view.result(0).str() +
                        " = expand_bf(" + view.operand(0).str() + ");");
                 return llvm::Error::success();
               }
               auto t = view.resultType(0);
               std::string expr =
                   isa<ShapedType>(t)
                       ? (view.operand(0).str() + ".cast[" + mojoDTypeName(t) +
                          "]()")
                       : (mojoScalarType(t) + "(" + view.operand(0).str() + ")");
               c.line("var " + view.result(0).str() + " = " + expr);
               return llvm::Error::success();
             });

  enterLocal("arith.truncf",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               if (c.file == "entry.c") {
                 c.line("const uint16_t " + view.result(0).str() +
                        " = narrow_bf(" + view.operand(0).str() + ");");
                 return llvm::Error::success();
               }
               auto t = view.resultType(0);
               std::string expr =
                   isa<ShapedType>(t)
                       ? (view.operand(0).str() + ".cast[" + mojoDTypeName(t) +
                          "]()")
                       : (mojoScalarType(t) + "(" + view.operand(0).str() + ")");
               c.line("var " + view.result(0).str() + " = " + expr);
               return llvm::Error::success();
             });

  enterLocal("math.exp",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               if (c.file == "entry.c")
                 c.line("const float " + view.result(0).str() + " = expf(" +
                        view.operand(0).str() + ");");
               else
                 c.line("var " + view.result(0).str() + " = exp(" +
                        view.operand(0).str() + ")");
               return llvm::Error::success();
             });

  enterLocal("math.exp2",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto x = view.operand(0).str();
               if (isa<ShapedType>(view.resultType(0)))
                 c.line("var " + view.result(0).str() + " = " + x + ".ne(" + x +
                        ").select(" + x + ", exp2(" + x + "))");
               else
                 c.line("var " + view.result(0).str() + " = " + x +
                        " if isnan(" + x + ") else exp2(" + x + ")");
               return llvm::Error::success();
             });

  rules["mojo_row.var_decl"] = PrintRule::format("var $r0: {mojo_type}");
  rules["mojo_row.var_assign"] = PrintRule::format("$0 = $1");
  rules["mojo_row.pass"] = PrintRule::format("pass");
  rules["mojo_row.if"] = PrintRule::local(
      [](const LocalOpView &view, PrintPoint at, PrintContext &c) -> llvm::Error {
        if (at == PrintPoint::Enter) {
          c.line("if " + view.operand(0).str() + ":");
        } else if (at == PrintPoint::BeginRegion) {
          if (c.region == 1 && view.attrOfType<BoolAttr>("has_else").getValue())
            c.line("else:");
        } else if (at == PrintPoint::BeginBlock) {
          c.indent++;
        } else if (at == PrintPoint::EndBlock) {
          c.indent--;
        }
        return llvm::Error::success();
      });

  enterLocal("mojo_row.host_panel_yield",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               bool wide = view.attrOfType<BoolAttr>("widen_bf16").getValue();
               std::string dst = view.attrOfType<StringAttr>("dst_expr").str();
               std::string v = view.operand(0).str();
               c.line(dst + " = " + (wide ? "expand_bf(" + v + ")" : v) + ";");
               return llvm::Error::success();
             });

  // ---------------- Panel ("row.mojo") Local Ops ----------------
  rules["mojo_row.panel_fn"] = PrintRule::local(
      [](const LocalOpView &view, PrintPoint at, PrintContext &c) -> llvm::Error {
        if (at == PrintPoint::BeginBlock) {
          c.line("from std.memory import bitcast");
          c.line("from std.math import fma");
          if (view.attrOfType<BoolAttr>("uses_stack").getValue())
            c.line("from std.memory import unsafe_stack_allocation");
          if (view.attrOfType<BoolAttr>("uses_brgemm").getValue())
            c.line("from std.ffi import external_call");
          if (view.attrOfType<BoolAttr>("uses_vnni").getValue())
            c.line("from std.sys import llvm_intrinsic");
          if (view.attrOfType<BoolAttr>("uses_round_bf16").getValue()) {
            c.line("@always_inline");
            c.line("def round_bf16[W:Int](x:SIMD[DType.float32,W])->SIMD[DType.bfloat16,W]:");
            c.line("    var bits=bitcast[DType.uint32,W](x)");
            c.line("    var high=bits>>16");
            c.line("    var rounded=(bits+0x7fff+(high&1))>>16");
            c.line("    var result=(bits&0x7fffffff).gt(0x7f800000).select(high|0x40,rounded)");
            c.line("    return bitcast[DType.bfloat16,W](result.cast[DType.uint16]())");
          }
          auto bArgs = view.blockArguments();
          auto dtypes = view.attrOfType<ArrayAttr>("param_dtypes");
          bool usesBrg = view.attrOfType<BoolAttr>("uses_brgemm").getValue();
          SmallVector<std::string> params;
          for (size_t i = 0; i < dtypes.size(); ++i) {
            auto dt = cast<StringAttr>(dtypes[i]).getValue().str();
            params.push_back(bArgs[i] + ": Pointer[Scalar[DType." + dt +
                             "],MutUntrackedOrigin]");
          }
          size_t base = dtypes.size();
          params.push_back(bArgs[base] + ": Int64");
          params.push_back(bArgs[base + 1] + ": Int64");
          if (usesBrg) {
            params.push_back(bArgs[base + 2] +
                             ": Pointer[UInt8,MutUntrackedOrigin]");
            params.push_back(bArgs[base + 3] +
                             ": Pointer[UInt8,MutUntrackedOrigin]");
            if (auto extC = view.attrOfType<BoolAttr>("external_c_buf");
                extC && extC.getValue()) {
              params.push_back(
                  bArgs[base + 4] +
                  ": Pointer[Scalar[DType.float32],MutUntrackedOrigin]");
            }
          }
          c.line("@export(\"intent_panel\")");
          c.line("def intent_panel(" + llvm::join(params, ", ") +
                 ") abi(\"C\") -> Int32:");
          c.indent++;
        } else if (at == PrintPoint::EndBlock) {
          c.indent--;
        }
        return llvm::Error::success();
      });

  rules["mojo_row.if_full_tile"] = PrintRule::local(
      [](const LocalOpView &view, PrintPoint at, PrintContext &c) -> llvm::Error {
        if (at == PrintPoint::Enter) {
          std::string cond =
              "Int(" + view.operand(0).str() + ") == " +
              std::to_string(view.attrOfType<IntegerAttr>("rows").getInt());
          if (view.attrOfType<BoolAttr>("check_cols").getValue())
            cond +=
                " and Int(" + view.operand(1).str() + ") == " +
                std::to_string(view.attrOfType<IntegerAttr>("cols").getInt());
          c.line("if " + cond + ":");
        } else if (at == PrintPoint::BeginBlock) {
          c.indent++;
        } else if (at == PrintPoint::EndBlock) {
          c.line("return Int32(0)");
          c.indent--;
        }
        return llvm::Error::success();
      });

  rules["mojo_row.for_range"] = PrintRule::local(
      [](const LocalOpView &view, PrintPoint at, PrintContext &c) -> llvm::Error {
        if (at == PrintPoint::BeginBlock) {
          const std::string &iv = view.blockArguments()[0];
          c.line("for " + iv + " in range(" + view.operand(0).str() + ", " +
                 view.operand(1).str() + ", " + view.operand(2).str() + "):");
          c.indent++;
        } else if (at == PrintPoint::EndBlock) {
          c.indent--;
        }
        return llvm::Error::success();
      });

  rules["mojo_row.while_lt"] = PrintRule::local(
      [](const LocalOpView &view, PrintPoint at, PrintContext &c) -> llvm::Error {
        if (at == PrintPoint::BeginBlock) {
          const std::string &iv = view.blockArguments()[0];
          c.line("var " + iv + " = " + view.operand(0).str());
          if (view.attrOfType<BoolAttr>("check_step").getValue())
            c.line("while " + iv + " + " + view.operand(2).str() +
                   " <= " + view.operand(1).str() + ":");
          else
            c.line("while " + iv + " < " + view.operand(1).str() + ":");
          c.indent++;
        } else if (at == PrintPoint::EndBlock) {
          const std::string &iv = view.blockArguments()[0];
          c.line(iv + " += " + view.operand(2).str());
          c.indent--;
        }
        return llvm::Error::success();
      });

  enterLocal("mojo_row.address",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto strides = view.attrOfType<DenseI64ArrayAttr>("strides");
               int32_t bits =
                   view.attrOfType<IntegerAttr>("index_bits").getInt();
               std::string iTy = bits == 32 ? "Int32" : "Int64";
               SmallVector<std::string> terms{iTy + "(0)"};
               for (auto [idxName, st] :
                    llvm::zip(view.operands().drop_front(),
                              strides.asArrayRef()))
                 terms.push_back(iTy + "(" + idxName + ") * " + iTy + "(" +
                                 std::to_string(st) + ")");
               c.line("var " + view.result(0).str() + " = Int(" +
                      llvm::join(terms, " + ") + ")");
               return llvm::Error::success();
             });

  enterLocal("mojo_row.acc_decl",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto dt = view.attrOfType<StringAttr>("dtype").getValue();
               auto w = view.attrOfType<IntegerAttr>("width").getInt();
               if (view.numOperands() > 0)
                 c.line("var " + view.result(0).str() + " = " +
                        view.operand(0).str());
               else
                 c.line("var " + view.result(0).str() + " = SIMD[DType." +
                        dt.str() + "," + std::to_string(w) + "](0)");
               return llvm::Error::success();
             });

  enterLocal("mojo_row.load_vec",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto ptr = view.operand(0).str(), addr = view.operand(1).str();
               auto res = view.result(0).str();
               auto dt = view.attrOfType<StringAttr>("dtype").getValue().str();
               int64_t w = view.attrOfType<IntegerAttr>("width").getInt();
               int64_t st =
                   view.attrOfType<IntegerAttr>("lane_stride").getInt();
               if (w == 1 || st == 1) {
                 c.line("var " + res + " = " + ptr + ".unsafe_offset(" + addr +
                        ").unsafe_load[width=" + std::to_string(w) + "]()");
               } else {
                 c.line("var " + res + " = SIMD[DType." + dt + "," +
                        std::to_string(w) + "](0)");
                 c.line("comptime for lane in range(" + std::to_string(w) +
                        "):");
                 c.indent++;
                 c.line(res + "[lane] = " + ptr + "[unsafe_offset=" + addr +
                        "+lane*" + std::to_string(st) + "]");
                 c.indent--;
               }
               return llvm::Error::success();
             });

  enterLocal("mojo_row.splat_cast",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto in = view.operand(0).str(), res = view.result(0).str();
               auto dt = view.attrOfType<StringAttr>("dtype").getValue().str();
               auto w = view.attrOfType<IntegerAttr>("width").getInt();
               if (view.attrOfType<BoolAttr>("from_scalar").getValue())
                 c.line("var " + res + " = SIMD[DType." + dt + "," +
                        std::to_string(w) + "](" + in + "[0].cast[DType." + dt +
                        "]())");
               else
                 c.line("var " + res + " = " + in + ".cast[DType." + dt +
                        "]()");
               return llvm::Error::success();
             });

  enterLocal("mojo_row.store_vec",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto ptr = view.operand(0).str(), addr = view.operand(1).str(),
                    val = view.operand(2).str();
               int64_t w = view.attrOfType<IntegerAttr>("width").getInt();
               int64_t st =
                   view.attrOfType<IntegerAttr>("lane_stride").getInt();
               if (w == 1 || st == 1) {
                 c.line(ptr + ".unsafe_offset(" + addr + ").unsafe_store(" +
                        val + ")");
               } else {
                 c.line("comptime for lane in range(" + std::to_string(w) +
                        "):");
                 c.indent++;
                 c.line(ptr + "[unsafe_offset=" + addr + "+lane*" +
                        std::to_string(st) + "] = " + val + "[lane]");
                 c.indent--;
               }
               return llvm::Error::success();
             });

  enterLocal("mojo_row.simd_fma",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto acc = view.operand(0).str(), lhs = view.operand(1).str(),
                    rhs = view.operand(2).str(), res = view.result(0).str();
               if (view.attrOfType<BoolAttr>("is_int").getValue())
                 c.line("var " + res + " = " + acc + " + " + lhs + " * " + rhs);
               else
                 c.line("var " + res + " = fma(" + lhs + ", " + rhs + ", " +
                        acc + ")");
               return llvm::Error::success();
             });

  enterLocal("mojo_row.q4k_superblock",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               using Q4 = intentdsl::kir::Q4KRecord;
               using Q8 = intentdsl::kir::Q8KRecord;
               auto wp = view.operand(0).str(), qp = view.operand(1).str(),
                    wb = view.operand(2).str(), g = view.operand(3).str(),
                    res = view.result(0).str();
               int64_t bs =
                   view.attrOfType<IntegerAttr>("byte_stride").getInt();
               bool useVnni =
                   view.attrOfType<BoolAttr>("use_vnni").getValue();
               auto wByte = [&](const std::string &off) {
                 return "UInt16(" + wp + "[unsafe_offset=" + wb + "+(" + off +
                        ")*" + std::to_string(bs) + "])";
               };
               c.line("var " + res + "_ds = " + qp +
                      ".bitcast[Float32]()[unsafe_offset=" + g + "*" +
                      std::to_string(Q8::bytes / 4) + "]");
               c.line("var " + res + "_d = Float32(bitcast[DType.float16](" +
                      wByte(std::to_string(Q4::d)) + " | (" +
                      wByte(std::to_string(Q4::d + 1)) + " << 8)))");
               c.line("var " + res + "_dm = Float32(bitcast[DType.float16](" +
                      wByte(std::to_string(Q4::dmin)) + " | (" +
                      wByte(std::to_string(Q4::dmin + 1)) + " << 8)))");
               c.line("var " + res + "_T = Int32(0)");
               if (useVnni) {
                 c.line("var " + res + "_accS = SIMD[DType.int32,16](0)");
                 c.line("comptime for p in range(4):");
                 c.indent++;
                 c.line("comptime j0 = 2 * p");
                 c.line("comptime j1 = 2 * p + 1");
                 c.line("var l0 = Int32(" +
                        wByte(std::to_string(Q4::factors) + "+j0%4") + ")");
                 c.line("var h0 = Int32(" +
                        wByte(std::to_string(Q4::factors + 4) + "+j0%4") + ")");
                 c.line("var u0 = Int32(" +
                        wByte(std::to_string(Q4::factors + 8) + "+j0%4") + ")");
                 c.line("var sc0 = (l0&63) if j0<4 else (u0&15)|((l0>>6)<<4)");
                 c.line("var mn0 = (h0&63) if j0<4 else (u0>>4)|((h0>>6)<<4)");
                 c.line("var l1 = Int32(" +
                        wByte(std::to_string(Q4::factors) + "+j1%4") + ")");
                 c.line("var h1 = Int32(" +
                        wByte(std::to_string(Q4::factors + 4) + "+j1%4") + ")");
                 c.line("var u1 = Int32(" +
                        wByte(std::to_string(Q4::factors + 8) + "+j1%4") + ")");
                 c.line("var sc1 = (l1&63) if j1<4 else (u1&15)|((l1>>6)<<4)");
                 c.line("var mn1 = (h1&63) if j1<4 else (u1>>4)|((h1>>6)<<4)");
                 if (bs == 1) {
                   c.line("var raw = " + wp + ".unsafe_offset(" + wb + "+" +
                          std::to_string(Q4::packed) +
                          "+p*32).unsafe_load[width=32]()");
                 } else {
                   c.line("var raw = SIMD[DType.uint8,32](0)");
                   c.line("comptime for k in range(32):");
                   c.indent++;
                   c.line("raw[k] = " + wp + "[unsafe_offset=" + wb + "+(" +
                          std::to_string(Q4::packed) + "+p*32+k)*" +
                          std::to_string(bs) + "]");
                   c.indent--;
                 }
                 c.line("var nib64 = (raw & 15).join(raw >> 4)");
                 c.line("var q64 = " + qp + ".unsafe_offset(" + g + "*" +
                        std::to_string(Q8::bytes) + "+" +
                        std::to_string(Q8::values) +
                        "+p*64).unsafe_load[width=64]()");
                 c.line("var dot16 = llvm_intrinsic[\"llvm.x86.avx512.vpdpbusd.512\",SIMD[DType.int32,16]](SIMD[DType.int32,16](0), nib64, q64)");
                 c.line(res + "_accS += dot16 * SIMD[DType.int32,8](sc0).join(SIMD[DType.int32,8](sc1))");
                 c.line("var sums = " + qp +
                        ".bitcast[Int16]().unsafe_offset(" + g + "*" +
                        std::to_string(Q8::bytes / 2) + "+" +
                        std::to_string(Q8::sums / 2) + "+4*p)");
                 c.line(res + "_T += mn0*(Int32(sums[unsafe_offset=0])+Int32(sums[unsafe_offset=1])) + mn1*(Int32(sums[unsafe_offset=2])+Int32(sums[unsafe_offset=3]))");
                 c.indent--;
                 c.line("var " + res + "_S = " + res + "_accS.reduce_add()");
               } else {
                 c.line("var " + res + "_accS = SIMD[DType.int32,32](0)");
                 c.line("comptime for j in range(8):");
                 c.indent++;
                 c.line("var l = Int32(" +
                        wByte(std::to_string(Q4::factors) + "+j%4") + ")");
                 c.line("var h = Int32(" +
                        wByte(std::to_string(Q4::factors + 4) + "+j%4") + ")");
                 c.line("var u = Int32(" +
                        wByte(std::to_string(Q4::factors + 8) + "+j%4") + ")");
                 c.line("var sc = (l&63) if j<4 else (u&15)|((l>>6)<<4)");
                 c.line("var mn = (h&63) if j<4 else (u>>4)|((h>>6)<<4)");
                 c.line("var q = " + qp + ".unsafe_offset(" + g + "*" +
                        std::to_string(Q8::bytes) + "+" +
                        std::to_string(Q8::values) + "+j*" +
                        std::to_string(Q4::group) +
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
                 c.line("var nib = ((raw>>SIMD[DType.uint8,32](UInt8(4*(j%2))))&15).cast[DType.int32]()");
                 c.line("var sums = " + qp +
                        ".bitcast[Int16]().unsafe_offset(" + g + "*" +
                        std::to_string(Q8::bytes / 2) + "+" +
                        std::to_string(Q8::sums / 2) + "+2*j)");
                 c.line(res + "_accS += SIMD[DType.int32,32](sc) * (q * nib)");
                 c.line(res + "_T += mn*(Int32(sums[unsafe_offset=0])+Int32(sums[unsafe_offset=1]))");
                 c.indent--;
                 c.line("var " + res + "_S = " + res + "_accS.reduce_add()");
               }
               c.line("var " + res + " = SIMD[DType.float32,1](" + res +
                      "_ds*(" + res + "_d*Float32(" + res + "_S)-" + res +
                      "_dm*Float32(" + res + "_T)))");
               return llvm::Error::success();
             });

  enterLocal("mojo_row.check_status",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto st = view.operand(0).str();
               c.line("if " + st + " != Int32(0):");
               c.indent++;
               c.line("return " + st);
               c.indent--;
               return llvm::Error::success();
             });

  enterLocal("mojo_row.q4k_unpack_row",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               using Q4 = intentdsl::kir::Q4KRecord;
               auto wp = view.operand(0).str(), wb = view.operand(1).str(),
                    ubuf = view.operand(2).str(), ri = view.operand(3).str(),
                    sub = view.operand(4).str();
               int64_t bs =
                   view.attrOfType<IntegerAttr>("byte_stride").getInt();
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
               c.line("var nib_u = (raw_u >> SIMD[DType.uint8,32](UInt8(4*(" +
                      sub + "%2)))) & 15");
               c.line(ubuf + ".unsafe_offset(" + ri +
                      "*32).unsafe_store(nib_u)");
               return llvm::Error::success();
             });

  enterLocal("mojo_row.q4k_accum_row",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               using Q4 = intentdsl::kir::Q4KRecord;
               using Q8 = intentdsl::kir::Q8KRecord;
               auto wp = view.operand(0).str(), wb = view.operand(1).str(),
                    qp = view.operand(2).str(), sbuf = view.operand(3).str(),
                    abuf = view.operand(4).str(), ri = view.operand(5).str(),
                    g = view.operand(6).str(), sub = view.operand(7).str();
               int64_t bs =
                   view.attrOfType<IntegerAttr>("byte_stride").getInt();
               auto wByte = [&](const std::string &off) {
                 return "UInt16(" + wp + "[unsafe_offset=" + wb + "+(" + off +
                        ")*" + std::to_string(bs) + "])";
               };
               c.line("var ds_a = " + qp +
                      ".bitcast[Float32]()[unsafe_offset=" + g + "*" +
                      std::to_string(Q8::bytes / 4) + "]");
               c.line("var d_a = Float32(bitcast[DType.float16](" +
                      wByte(std::to_string(Q4::d)) + " | (" +
                      wByte(std::to_string(Q4::d + 1)) + " << 8)))");
               c.line("var dm_a = Float32(bitcast[DType.float16](" +
                      wByte(std::to_string(Q4::dmin)) + " | (" +
                      wByte(std::to_string(Q4::dmin + 1)) + " << 8)))");
               c.line("var l_a = Int32(" +
                      wByte(std::to_string(Q4::factors) + "+" + sub + "%4") +
                      ")");
               c.line("var h_a = Int32(" +
                      wByte(std::to_string(Q4::factors + 4) + "+" + sub +
                            "%4") +
                      ")");
               c.line("var u_a = Int32(" +
                      wByte(std::to_string(Q4::factors + 8) + "+" + sub +
                            "%4") +
                      ")");
               c.line("var sc_a = (l_a&63) if " + sub +
                      "<4 else (u_a&15)|((l_a>>6)<<4)");
               c.line("var mn_a = (h_a&63) if " + sub +
                      "<4 else (u_a>>4)|((h_a>>6)<<4)");
               c.line("var sums_a = " + qp +
                      ".bitcast[Int16]().unsafe_offset(" + g + "*" +
                      std::to_string(Q8::bytes / 2) + "+" +
                      std::to_string(Q8::sums / 2) + "+2*" + sub + ")");
               c.line("var dot_a = sc_a * " + sbuf + "[unsafe_offset=" + ri +
                      "]");
               c.line("var min_a = mn_a * (Int32(sums_a[unsafe_offset=0]) + Int32(sums_a[unsafe_offset=1]))");
               c.line(abuf + "[unsafe_offset=" + ri +
                      "] += ds_a * (d_a * Float32(dot_a) - dm_a * Float32(min_a))");
               return llvm::Error::success();
             });

  // ---------------- Host ("entry.c") Local Ops ----------------
  rules["mojo_row.host_fn"] = PrintRule::local(
      [](const LocalOpView &view, PrintPoint at, PrintContext &c) -> llvm::Error {
        if (at == PrintPoint::Enter) {
          c.line("#include <stdint.h>");
          c.line("#include <stddef.h>");
          c.line("#include <math.h>");
          c.line("#include <stdlib.h>");
          c.line("#include <string.h>");
          c.line("#include <omp.h>");
          if (view.attrOfType<BoolAttr>("uses_f16_math").getValue() ||
              view.attrOfType<BoolAttr>("uses_bf16_math").getValue()) {
            c.line("#if defined(__AVX512F__) || defined(__F16C__)");
            c.line("#include <immintrin.h>");
            c.line("#endif");
          }
          int64_t arity =
              view.attrOfType<IntegerAttr>("panel_arity").getInt();
          bool usesBrg = view.attrOfType<BoolAttr>("uses_brgemm").getValue();
          SmallVector<std::string> ptypes(arity, "void*");
          ptypes.push_back("int64_t");
          ptypes.push_back("int64_t");
          if (usesBrg) {
            ptypes.push_back("const void*");
            ptypes.push_back("void*");
            if (auto extC = view.attrOfType<BoolAttr>("external_c_buf");
                extC && extC.getValue())
              ptypes.push_back("float*");
            c.line("extern const void *intent_brgemm_get(int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int32_t,int32_t,int32_t);");
            c.line("extern const void *intent_brgemm_get_ex(int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int32_t,int32_t,int32_t,int32_t);");
            c.line("extern int64_t intent_brgemm_scratch_bytes(const void*);");
            c.line("extern int32_t intent_brgemm_thread_enter(const void*);");
            c.line("extern void intent_brgemm_thread_leave(void);");
            c.line("extern int32_t intent_brgemm_tile(const void*,const void*,const void*,void*,void*);");
          }
          c.line("extern int32_t intent_panel(" + llvm::join(ptypes, ",") +
                 ");");
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

          if (view.attrOfType<BoolAttr>("uses_q8k").getValue()) {
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
            c.line("  #pragma omp simd");
            c.line("  for (int64_t k=0; k<256; ++k) {");
            c.line("    float v=x[k*stride]*inv; int q=(int)v; float fraction=fabsf(v-(float)q);");
            c.line("    if (fraction>0.5f || (fraction==0.5f && (q&1))) q+=v<0 ? -1 : 1;");
            c.line("    out->q[k]=(int8_t)(q>127 ? 127 : q);");
            c.line("  }");
            c.line("  for (int group=0; group<16; ++group) {");
            c.line("    int sum=0;");
            c.line("    #pragma omp simd reduction(+:sum)");
            c.line("    for (int64_t k=16*group; k<16*group+16; ++k) sum+=out->q[k];");
            c.line("    out->sums[group]=(int16_t)sum;");
            c.line("  }");
            c.line("}");
          }
          if (view.attrOfType<BoolAttr>("uses_bf16_math").getValue() ||
              view.attrOfType<BoolAttr>("uses_f16_math").getValue()) {
            c.line("#if defined(__AVX512F__)");
            c.line("static inline void transpose16x16_ps(const __m512 *r, float *out, int64_t ldo){");
            c.line("  __m512 t[16], u[16];");
            c.line("  for (int m = 0; m < 8; ++m) { t[2*m] = _mm512_unpacklo_ps(r[2*m], r[2*m+1]); t[2*m+1] = _mm512_unpackhi_ps(r[2*m], r[2*m+1]); }");
            c.line("  for (int g = 0; g < 4; ++g) {");
            c.line("    u[4*g+0] = _mm512_shuffle_ps(t[4*g+0], t[4*g+2], 0x44); u[4*g+1] = _mm512_shuffle_ps(t[4*g+0], t[4*g+2], 0xEE);");
            c.line("    u[4*g+2] = _mm512_shuffle_ps(t[4*g+1], t[4*g+3], 0x44); u[4*g+3] = _mm512_shuffle_ps(t[4*g+1], t[4*g+3], 0xEE);");
            c.line("  }");
            c.line("  for (int j = 0; j < 4; ++j) {");
            c.line("    __m512 v0 = _mm512_shuffle_f32x4(u[j], u[4+j], 0x88), v1 = _mm512_shuffle_f32x4(u[8+j], u[12+j], 0x88);");
            c.line("    __m512 v2 = _mm512_shuffle_f32x4(u[j], u[4+j], 0xDD), v3 = _mm512_shuffle_f32x4(u[8+j], u[12+j], 0xDD);");
            c.line("    _mm512_storeu_ps(out + (0+j)*ldo, _mm512_shuffle_f32x4(v0, v1, 0x88));");
            c.line("    _mm512_storeu_ps(out + (4+j)*ldo, _mm512_shuffle_f32x4(v2, v3, 0x88));");
            c.line("    _mm512_storeu_ps(out + (8+j)*ldo, _mm512_shuffle_f32x4(v0, v1, 0xDD));");
            c.line("    _mm512_storeu_ps(out + (12+j)*ldo, _mm512_shuffle_f32x4(v2, v3, 0xDD));");
            c.line("  }");
            c.line("}");
            c.line("#endif");
          }
          if (view.attrOfType<BoolAttr>("uses_bf16_math").getValue()) {
            c.line("static inline float expand_bf(uint16_t v){uint32_t b=(uint32_t)v<<16;float f;memcpy(&f,&b,4);return f;}");
            c.line("static inline uint16_t narrow_bf(float f){uint32_t b;memcpy(&b,&f,4);if((b&0x7fffffff)>0x7f800000)return (b>>16)|64;return (b+32767+((b>>16)&1))>>16;}");
            c.line("static inline float min_ieee(float a,float b){return isnan(a)?a:isnan(b)?b:fminf(a,b);}");
            c.line("static inline void widen_bf16_span(float *out, const uint16_t *src, int64_t n){");
            c.line("  if (n <= 0) return; size_t i=0, un=(size_t)n;");
            c.line("#if defined(__AVX512F__)");
            c.line("  for (; i + 64 <= un; i += 64) {");
            c.line("    for (size_t u = 0; u < 64; u += 16) { __m256i h = _mm256_loadu_si256((const __m256i*)(src+i+u)); _mm512_storeu_ps(out+i+u, _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16))); }");
            c.line("  }");
            c.line("  for (; i + 16 <= un; i += 16) { __m256i h = _mm256_loadu_si256((const __m256i*)(src+i)); _mm512_storeu_ps(out+i, _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16))); }");
            c.line("#endif");
            c.line("  for (; i<un; ++i) out[i] = expand_bf(src[i]);");
            c.line("}");
            c.line("static inline void widen_transpose_bf16_panel(float *out, const uint16_t *src, int64_t K, int64_t valid, int64_t nr, int64_t ldn){");
            c.line("  int64_t jb = 0;");
            c.line("#if defined(__AVX512F__)");
            c.line("  for (; jb + 16 <= valid; jb += 16) {");
            c.line("    int64_t kb = 0;");
            c.line("    for (; kb + 16 <= K; kb += 16) {");
            c.line("      __m512 r[16];");
            c.line("      for (int i = 0; i < 16; ++i) { __m256i h = _mm256_loadu_si256((const __m256i*)(src + (jb + i)*ldn + kb)); r[i] = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16)); }");
            c.line("      transpose16x16_ps(r, out + kb*nr + jb, nr);");
            c.line("    }");
            c.line("    for (; kb < K; ++kb) { float *dst = out + kb*nr + jb; for (int i = 0; i < 16; ++i) dst[i] = expand_bf(src[(jb + i)*ldn + kb]); }");
            c.line("  }");
            c.line("#endif");
            c.line("  if (jb < valid || valid < nr) { for (int64_t k = 0; k < K; ++k) { float *dst = out + k*nr; for (int64_t j = jb; j < valid; ++j) dst[j] = expand_bf(src[j*ldn + k]); if (valid < nr) memset(dst + valid, 0, (size_t)(nr - valid) * sizeof(float)); } }");
            c.line("}");
          }
          if (view.attrOfType<BoolAttr>("uses_f16_math").getValue()) {
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
            c.line("  for (; i + 64 <= un; i += 64) {");
            c.line("    for (size_t u = 0; u < 64; u += 16) { __m256i h = _mm256_loadu_si256((const __m256i*)(src+i+u)); _mm512_storeu_ps(out+i+u, _mm512_cvtph_ps(h)); }");
            c.line("  }");
            c.line("  for (; i + 16 <= un; i += 16) { __m256i h = _mm256_loadu_si256((const __m256i*)(src+i)); _mm512_storeu_ps(out+i, _mm512_cvtph_ps(h)); }");
            c.line("#endif");
            c.line("  for (; i<un; ++i) out[i] = expand_f16(src[i]);");
            c.line("}");
            c.line("static inline void widen_transpose_f16_panel(float *out, const uint16_t *src, int64_t K, int64_t valid, int64_t nr, int64_t ldn){");
            c.line("  int64_t jb = 0;");
            c.line("#if defined(__AVX512F__) && defined(__F16C__)");
            c.line("  for (; jb + 16 <= valid; jb += 16) {");
            c.line("    int64_t kb = 0;");
            c.line("    for (; kb + 16 <= K; kb += 16) {");
            c.line("      __m512 r[16];");
            c.line("      for (int i = 0; i < 16; ++i) { __m256i h = _mm256_loadu_si256((const __m256i*)(src + (jb + i)*ldn + kb)); r[i] = _mm512_cvtph_ps(h); }");
            c.line("      transpose16x16_ps(r, out + kb*nr + jb, nr);");
            c.line("    }");
            c.line("    for (; kb < K; ++kb) { float *dst = out + kb*nr + jb; for (int i = 0; i < 16; ++i) dst[i] = expand_f16(src[(jb + i)*ldn + kb]); }");
            c.line("  }");
            c.line("#endif");
            c.line("  if (jb < valid || valid < nr) { for (int64_t k = 0; k < K; ++k) { float *dst = out + k*nr; for (int64_t j = jb; j < valid; ++j) dst[j] = expand_f16(src[j*ldn + k]); if (valid < nr) memset(dst + valid, 0, (size_t)(nr - valid) * sizeof(float)); } }");
            c.line("}");
          }
          if (view.attrOfType<BoolAttr>("uses_mx8_math").getValue()) {
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

          auto ext = view.attrOfType<DenseI64ArrayAttr>("extents");
          auto tl = view.attrOfType<DenseI64ArrayAttr>("tile");
          c.line("int intent_entry(" +
                 view.attrOfType<StringAttr>("params").str() + ") {");
          c.indent++;
          c.line("const int64_t M=" + std::to_string(ext[0]) +
                 ", N=" + std::to_string(ext[1]) + ", K=" +
                 std::to_string(ext[2]) + ", G=" + std::to_string(ext[3]) + ";");
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
      });

  enterLocal("mojo_row.host_alloc",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto name = view.attrOfType<StringAttr>("buf_name").str();
               auto place = view.attrOfType<StringAttr>("placement").getValue();
               auto cType = view.attrOfType<StringAttr>("c_type").str();
               auto count = view.attrOfType<StringAttr>("count_expr").str();
               int wsSlot = name == "left"      ? 0
                            : name == "rhs"     ? 1
                            : name == "members" ? 2
                                                : 3;
               if (place == "stack") {
                 c.line("_Alignas(64) " + cType + " " + name + "[" + count +
                        "];");
               } else {
                 std::string frees;
                 for (Attribute a : view.attrOfType<ArrayAttr>("cleanup")) {
                   std::string prev = cast<StringAttr>(a).str();
                   int pSlot = prev == "left"      ? 0
                               : prev == "rhs"     ? 1
                               : prev == "members" ? 2
                                                   : 3;
                   frees += "intent_ws_release(" + std::to_string(pSlot) +
                            ", " + prev + ");";
                 }
                 c.line(cType + " *" + name + " = (" + cType +
                        "*)intent_ws_acquire(" + std::to_string(wsSlot) +
                        ", (size_t)(" + count + ") * sizeof(" + cType + "));");
                 c.line("if (!" + name + ") { " + frees + "return 2; }");
               }
               return llvm::Error::success();
             });

  rules["mojo_row.host_omp_parallel"] = PrintRule::local(
      [](const LocalOpView &view, PrintPoint at, PrintContext &c) -> llvm::Error {
        if (at == PrintPoint::Enter) {
          std::string nt = view.attrOfType<StringAttr>("num_threads").str();
          std::string ic = view.attrOfType<StringAttr>("if_cond").str();
          std::string red =
              view.attrOfType<BoolAttr>("reduce_rc").getValue()
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
      });

  rules["mojo_row.host_omp_for"] = PrintRule::local(
      [](const LocalOpView &view, PrintPoint at, PrintContext &c) -> llvm::Error {
        if (at == PrintPoint::Enter) {
          bool par = view.attrOfType<BoolAttr>("parallel").getValue();
          bool nw = view.attrOfType<BoolAttr>("nowait").getValue();
          bool simd = view.attrOfType<BoolAttr>("simd").getValue();
          std::string ic = view.attrOfType<StringAttr>("if_cond").str();
          std::string iv = view.attrOfType<StringAttr>("iv").str();
          std::string lb = view.attrOfType<StringAttr>("lb").str();
          std::string ub = view.attrOfType<StringAttr>("ub").str();
          std::string st = view.attrOfType<StringAttr>("step").str();
          if (simd)
            c.line("#pragma omp simd");
          else if (par) {
            auto scAttr = view.attrOfType<StringAttr>("schedule");
            std::string sc =
                (scAttr && !scAttr.getValue().empty()) ? scAttr.getValue().str()
                                                       : "static";
            c.line("#pragma omp parallel for num_threads(workers) schedule(" +
                   sc + ")" + (ic.empty() ? "" : " if(" + ic + ")"));
          } else {
            auto scAttr = view.attrOfType<StringAttr>("schedule");
            std::string sc =
                (scAttr && !scAttr.getValue().empty()) ? scAttr.getValue().str()
                                                       : "static";
            c.line("#pragma omp for schedule(" + sc + ")" +
                   (nw ? " nowait" : ""));
          }
          c.line("for (int64_t " + iv + "=" + lb + "; " + iv + "<" + ub + "; " +
                 iv + "+=" + st + ") {");
        } else if (at == PrintPoint::BeginBlock) {
          c.indent++;
        } else if (at == PrintPoint::EndBlock) {
          c.indent--;
          c.line("}");
        }
        return llvm::Error::success();
      });

  enterLocal("mojo_row.host_load_scalar",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               int64_t s = view.attrOfType<IntegerAttr>("slot").getInt();
               std::string cTy = view.attrOfType<StringAttr>("c_type").str();
               std::string idx =
                   view.attrOfType<StringAttr>("index_expr").str();
               std::string base = s >= 0 ? ("a" + std::to_string(s))
                                         : ("wk_" + std::to_string(-1 - s));
               c.line("const " + cTy + " " + view.result(0).str() +
                      " = ((const " + cTy + "*)" + base + ")[" + idx + "];");
               return llvm::Error::success();
             });

  enterLocal("mojo_row.host_span_op",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto kind = view.attrOfType<StringAttr>("op").getValue();
               std::string dst = view.attrOfType<StringAttr>("dst_expr").str();
               std::string src = view.attrOfType<StringAttr>("src_expr").str();
               std::string cnt =
                   view.attrOfType<StringAttr>("count_expr").str();
               int64_t eb = view.attrOfType<IntegerAttr>("elem_bytes").getInt();
               if (kind == "widen_bf16")
                 c.line("widen_bf16_span(" + dst + ", " + src + ", " + cnt +
                        ");");
               else if (kind == "widen_f16")
                 c.line("widen_f16_span(" + dst + ", " + src + ", " + cnt +
                        ");");
               else if (kind == "widen_transpose_bf16")
                 c.line("widen_transpose_bf16_panel(" + dst + ", " + src +
                        ", K, valid, nr, " + cnt + ");");
               else if (kind == "widen_transpose_f16")
                 c.line("widen_transpose_f16_panel(" + dst + ", " + src +
                        ", K, valid, nr, " + cnt + ");");
               else if (kind == "memcpy")
                 c.line("memcpy(" + dst + ", " + src + ", (size_t)(" + cnt +
                        ") * " + std::to_string(eb) + "ULL);");
               else if (kind == "zero")
                 c.line("memset(" + dst + ", 0, (size_t)(" + cnt + ") * " +
                        std::to_string(eb) + "ULL);");
               return llvm::Error::success();
             });

  enterLocal("mojo_row.host_elem_op",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto kind = view.attrOfType<StringAttr>("op").getValue();
               std::string dst = view.attrOfType<StringAttr>("dst_expr").str();
               std::string src = view.attrOfType<StringAttr>("src_expr").str();
               std::string aux = view.attrOfType<StringAttr>("aux_expr").str();
               int64_t eb = view.attrOfType<IntegerAttr>("elem_bytes").getInt();
               if (kind == "expand_bf")
                 c.line(dst + " = expand_bf(" + src + ");");
               else if (kind == "expand_f16")
                 c.line(dst + " = expand_f16(" + src + ");");
               else if (kind == "assign")
                 c.line(dst + " = " + src + ";");
               else if (kind == "memcpy")
                 c.line("memcpy(" + dst + ", " + src + ", " +
                        std::to_string(eb) + ");");
               else if (kind == "decode_mx8")
                 c.line(dst + " = decode_mx8(" + src + ", " + aux + ");");
               else if (kind == "pack_q8k")
                 c.line("pack_q8_record(" + dst + ", " + src + ", " + aux +
                        ");");
               else if (kind == "dot4_quad") {
                 auto comma = StringRef(aux).split(',');
                 std::string sbn = comma.first.str(), sbk = comma.second.str();
                 c.line("int8_t v0 = " + src + "[j*" + sbn + "], v1 = " + src +
                        "[j*" + sbn + " + " + sbk + "], v2 = " + src + "[j*" +
                        sbn + " + 2*" + sbk + "], v3 = " + src + "[j*" + sbn +
                        " + 3*" + sbk + "];");
                 c.line(dst + "[4*j] = (uint8_t)v0; " + dst +
                        "[4*j+1] = (uint8_t)v1; " + dst +
                        "[4*j+2] = (uint8_t)v2; " + dst +
                        "[4*j+3] = (uint8_t)v3;");
                 c.line("corr[j] -= (uint32_t)(((int)v0 + (int)v1 + (int)v2 + (int)v3) * 128);");
               }
               return llvm::Error::success();
             });

  enterLocal("mojo_row.host_brgemm_plan_decl",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               c.line("const void *brg_plan[2*(mr+1)]; memset(brg_plan, 0, sizeof(brg_plan));");
               if (auto kc = view.attrOfType<BoolAttr>("kc_split");
                   kc && kc.getValue()) {
                 c.line("const void *brg_kc0[2*(mr+1)]; memset(brg_kc0, 0, sizeof(brg_kc0));");
                 c.line("const void *brg_kc1[2*(mr+1)]; memset(brg_kc1, 0, sizeof(brg_kc1));");
               }
               c.line("int64_t brg_scratch_bytes = 64;");
               c.line("uint8_t r_seen[mr+1]; memset(r_seen, 0, sizeof(r_seen));");
               return llvm::Error::success();
             });

  enterLocal("mojo_row.host_brgemm_lookup",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               int64_t aDt = view.attrOfType<IntegerAttr>("a_dt").getInt();
               int64_t bDt = view.attrOfType<IntegerAttr>("b_dt").getInt();
               int64_t cDt = view.attrOfType<IntegerAttr>("c_dt").getInt();
               std::string lda = view.attrOfType<StringAttr>("lda_expr").str();
               std::string ldb = view.attrOfType<StringAttr>("ldb_expr").str();
               std::string nE = view.attrOfType<StringAttr>("n_expr").str();
               std::string kE = view.attrOfType<StringAttr>("k_expr").str();
               auto pvAttr = view.attrOfType<StringAttr>("plan_var");
               std::string pVar =
                   (pvAttr && !pvAttr.getValue().empty()) ? pvAttr.str()
                                                          : "brg_plan";
               auto addCAttr = view.attrOfType<IntegerAttr>("add_c");
               int64_t addC = addCAttr ? addCAttr.getInt() : 0;
               std::string frees;
               for (Attribute a : view.attrOfType<ArrayAttr>("cleanup")) {
                 std::string prev = cast<StringAttr>(a).str();
                 int pSlot = prev == "left"      ? 0
                             : prev == "rhs"     ? 1
                             : prev == "members" ? 2
                                                 : 3;
                 frees += "intent_ws_release(" + std::to_string(pSlot) + ", " +
                          prev + ");";
               }
               c.line("{ const void *h = intent_brgemm_get_ex(rt, " + nE + ", " +
                      kE + ", " + lda + ", " + ldb + ", " + nE + ", " +
                      std::to_string(aDt) + ", " + std::to_string(bDt) + ", " +
                      std::to_string(cDt) + ", " + std::to_string(addC) + ");");
               c.line("  int64_t sb = intent_brgemm_scratch_bytes(h);");
               c.line("  if (!h || sb < 0) { " + frees + "return 3; }");
               c.line("  " + pVar + "[ci*(mr+1) + rt] = h;");
               c.line("  if (sb > brg_scratch_bytes) brg_scratch_bytes = sb; }");
               return llvm::Error::success();
             });

  enterLocal("mojo_row.host_thread_scratch_alloc",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               int64_t sb =
                   view.attrOfType<IntegerAttr>("stack_bytes").getInt();
               int64_t al = view.attrOfType<IntegerAttr>("alignment").getInt();
               std::string sv = view.attrOfType<StringAttr>("size_var").str();
               std::string sbS = std::to_string(sb);
               c.line("_Alignas(" + std::to_string(al) +
                      ") unsigned char brg_scratch_stk[" + sbS + "];");
               c.line("void *brg_scratch = " + sv + " <= " + sbS +
                      " ? (void*)brg_scratch_stk : aligned_alloc(" +
                      std::to_string(al) + ", (size_t)" + sv + ");");
               c.line("int thr_ok = brg_scratch != NULL;");
               c.line("if (!thr_ok) rc |= 2;");
               return llvm::Error::success();
             });

  enterLocal("mojo_row.host_brgemm_ctx_switch",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto pvAttr = view.attrOfType<StringAttr>("plan_var");
               std::string pVar =
                   (pvAttr && !pvAttr.getValue().empty()) ? pvAttr.str()
                                                          : "brg_plan";
               c.line("const void *bh = " + pVar + "[(C == nr ? 0 : 1)*(mr+1) + R];");
               c.line("if (bh != brg_active) { if (intent_brgemm_thread_enter(bh) != 0) { rc |= 3; thr_ok = 0; continue; } brg_active = bh; }");
               return llvm::Error::success();
             });

  enterLocal("mojo_row.host_tile_coords",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               bool win = view.attrOfType<BoolAttr>("windowed").getValue();
               bool grp = view.attrOfType<BoolAttr>("grouped").getValue();
               int64_t cg = view.attrOfType<IntegerAttr>("col_group").getInt();
               if (!win && !grp && cg > 1) {
                 std::string cgS = std::to_string(cg);
                 c.line("const int64_t g = job / (nc * mb), rem = job % (nc * mb);");
                 c.line("const int64_t full_cg = (nc / " + cgS + ") * " + cgS + ";");
                 c.line("int64_t c, r, end = M;");
                 c.line("if (rem < full_cg * mb) {");
                 c.line("  const int64_t cg_idx = rem / (" + cgS + " * mb), cg_rem = rem % (" + cgS + " * mb);");
                 c.line("  c = (cg_idx * " + cgS + " + cg_rem % " + cgS + ") * nr; r = (cg_rem / " + cgS + ") * mr;");
                 c.line("} else {");
                 c.line("  const int64_t trem = rem - full_cg * mb;");
                 c.line("  c = (full_cg + trem / mb) * nr; r = (trem % mb) * mr;");
                 c.line("}");
               } else {
                 c.line("const int64_t g = job / (nc * mb), rem = job % (nc * mb);");
                 c.line("const int64_t c = (rem / mb) * nr;");
                 c.line("int64_t r = (rem % mb) * mr, end = M;");
               }
               if (win) {
                 int64_t H =
                     view.attrOfType<IntegerAttr>("head_extent").getInt();
                 int64_t chunks =
                     view.attrOfType<IntegerAttr>("chunk_extent").getInt();
                 c.line("const int64_t head=g%" + std::to_string(H) +
                        ", chunk=g/" + std::to_string(H) + "%" +
                        std::to_string(chunks) + ", batch=g/" +
                        std::to_string(H) + "/" + std::to_string(chunks) + ";");
               }
               return llvm::Error::success();
             });

  enterLocal("mojo_row.host_grouped_bounds",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               int64_t oSlot =
                   view.attrOfType<IntegerAttr>("offsets_slot").getInt();
               int64_t oSt =
                   view.attrOfType<IntegerAttr>("offsets_stride").getInt();
               c.line("r += ((const int32_t*)a" + std::to_string(oSlot) +
                      ")[g*" + std::to_string(oSt) + "LL];");
               c.line("end = ((const int32_t*)a" + std::to_string(oSlot) +
                      ")[(g+1)*" + std::to_string(oSt) + "LL];");
               return llvm::Error::success();
             });

  enterLocal("mojo_row.host_route_coords",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               if (view.attrOfType<BoolAttr>("routed").getValue()) {
                 int64_t rSlot =
                     view.attrOfType<IntegerAttr>("routes_slot").getInt();
                 int64_t rSt =
                     view.attrOfType<IntegerAttr>("routes_stride").getInt();
                 int64_t mult =
                     view.attrOfType<IntegerAttr>("multiplicity").getInt();
                 c.line("const int64_t dest = ((const int32_t*)a" +
                        std::to_string(rSlot) + ")[r*" + std::to_string(rSt) +
                        "LL], src = dest / " + std::to_string(mult) + "LL;");
               } else {
                 c.line("const int64_t dest = r, src = r;");
               }
               c.line("const int64_t R = end-r < mr ? end-r : mr, C = N-c < nr ? N-c : nr;");
               return llvm::Error::success();
             });

  enterLocal("mojo_row.host_view_ptr",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               int64_t idx = view.attrOfType<IntegerAttr>("index").getInt();
               auto sk = view.attrOfType<StringAttr>("source_kind").getValue();
               auto expr = view.attrOfType<StringAttr>("expr").str();
               if (sk == "routed_out") {
                 c.line("_Alignas(64) unsigned char routed_out[" + expr +
                        "]; void *p" + std::to_string(idx) + " = routed_out;");
               } else {
                 c.line("void *p" + std::to_string(idx) + " = (void*)(" + expr +
                        ");");
               }
               return llvm::Error::success();
             });

  enterLocal("mojo_row.host_call_panel",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               int64_t arity = view.attrOfType<IntegerAttr>("arity").getInt();
               bool usesBrg =
                   view.attrOfType<BoolAttr>("uses_brgemm").getValue();
               SmallVector<std::string> args;
               for (int64_t i = 0; i < arity; ++i)
                 args.push_back("p" + std::to_string(i));
               args.push_back("R");
               args.push_back("C");
               if (usesBrg) {
                 args.push_back("bh");
                 args.push_back("brg_scratch");
                 if (auto cbe = view.attrOfType<StringAttr>("c_buf_expr");
                     cbe && !cbe.getValue().empty())
                   args.push_back(cbe.str());
                 c.line("const int32_t panel_rc = intent_panel(" +
                        llvm::join(args, ", ") + ");");
                 c.line("if (panel_rc != 0) { rc |= (int)panel_rc; thr_ok = 0; continue; }");
               } else {
                 c.line("const int32_t panel_rc = intent_panel(" +
                        llvm::join(args, ", ") + ");");
                 c.line("if (panel_rc != 0) { rc |= (int)panel_rc; continue; }");
               }
               return llvm::Error::success();
             });

  enterLocal("mojo_row.host_free",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               std::string name = view.attrOfType<StringAttr>("buf_name").str();
               int wsSlot = name == "left"      ? 0
                            : name == "rhs"     ? 1
                            : name == "members" ? 2
                                                : 3;
               c.line("intent_ws_release(" + std::to_string(wsSlot) + ", " +
                      name + ");");
               return llvm::Error::success();
             });

  // ---------------- Ordered ("row.mojo" & "entry.c") Local Ops ----------------
  rules["mojo_row.owner_fn"] = PrintRule::local(
      [](const LocalOpView &view, PrintPoint at, PrintContext &c) -> llvm::Error {
        if (at == PrintPoint::BeginBlock) {
          c.line("from std.math import exp, sqrt, rsqrt, exp2, log, max, isnan");
          c.line("from std.memory import bitcast");
          if (view.attrOfType<BoolAttr>("uses_stack").getValue())
            c.line("from std.memory import unsafe_stack_allocation");
          auto bArgs = view.blockArguments();
          auto ptypes = view.attrOfType<ArrayAttr>("param_types");
          SmallVector<std::string> params;
          for (size_t i = 0; i < ptypes.size(); ++i)
            params.push_back(bArgs[i] + ": " +
                             cast<StringAttr>(ptypes[i]).getValue().str());
          size_t base = ptypes.size();
          params.push_back(bArgs[base] + ": Int64");
          params.push_back(bArgs[base + 1] + ": Int64");
          c.line("@export(\"owner_task\")");
          c.line("def owner_task(" + llvm::join(params, ", ") + ") abi(\"C\"):");
          c.indent++;
        } else if (at == PrintPoint::EndBlock) {
          c.indent--;
        }
        return llvm::Error::success();
      });

  enterLocal("mojo_row.owner_coord",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               int64_t pitch = view.attrOfType<IntegerAttr>("pitch").getInt();
               int64_t ext = view.attrOfType<IntegerAttr>("extent").getInt();
               std::string base =
                   pitch == 1
                       ? view.operand(0).str()
                       : "(" + view.operand(0).str() + " // " +
                             std::to_string(pitch) + ")";
               c.line("var " + view.result(0).str() + " = " + base + " % " +
                      std::to_string(ext));
               return llvm::Error::success();
             });

  enterLocal("mojo_row.scalar_decl",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto st =
                   view.attrOfType<StringAttr>("scalar_type").getValue().str();
               if (view.numOperands() > 0)
                 c.line("var " + view.result(0).str() + " = " +
                        view.operand(0).str());
               else
                 c.line("var " + view.result(0).str() + ": " + st);
               return llvm::Error::success();
             });

  enterLocal("mojo_row.pack_apply",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto args = view.operands();
               auto inDt =
                   view.attrOfType<StringAttr>("in_dtype").getValue().str();
               auto outDt =
                   view.attrOfType<StringAttr>("out_dtype").getValue().str();
               auto kind = view.attrOfType<StringAttr>("kind").getValue();
               std::string simd = "SIMD[DType." + inDt + "," +
                                  std::to_string(args.size()) + "](" +
                                  llvm::join(args, ", ") + ")";
               std::string expr = kind == "id"     ? simd
                                  : kind == "exp"  ? ("exp(" + simd + ")")
                                  : kind == "exp2" ? ("exp2(" + simd + ")")
                                                   : (simd + ".cast[DType." +
                                                      outDt + "]()");
               c.line("var " + view.result(0).str() + " = " + expr);
               return llvm::Error::success();
             });

  rules["mojo_row.c_host_fn"] = PrintRule::local(
      [](const LocalOpView &view, PrintPoint at, PrintContext &c) -> llvm::Error {
        if (at == PrintPoint::Enter) {
          c.line("#include <stdint.h>");
          c.line("#include <stddef.h>");
          c.line(view.attrOfType<StringAttr>("extern_decl").str());
          c.line("int intent_entry(" +
                 view.attrOfType<StringAttr>("params").str() + ") {");
        } else if (at == PrintPoint::BeginBlock) {
          c.indent++;
        } else if (at == PrintPoint::EndBlock) {
          c.line("return 0;");
          c.indent--;
          c.line("}");
        }
        return llvm::Error::success();
      });

  rules["mojo_row.c_omp_for"] = PrintRule::local(
      [](const LocalOpView &view, PrintPoint at, PrintContext &c) -> llvm::Error {
        int64_t count = view.attrOfType<IntegerAttr>("count").getInt();
        int64_t grain = view.attrOfType<IntegerAttr>("grain").getInt();
        bool cyclic = view.attrOfType<BoolAttr>("cyclic").getValue();
        if (at == PrintPoint::Enter) {
          if (cyclic) {
            int64_t chunks = (count + grain - 1) / grain;
            c.line("#pragma omp parallel for num_threads(workers) schedule(dynamic, 1) if(workers>1 && " +
                   std::to_string(chunks) + ">1)");
            c.line("for (int64_t ch=0; ch<" + std::to_string(chunks) +
                   "LL; ++ch) {");
          } else {
            c.line("#pragma omp parallel for num_threads(workers) schedule(static) if(workers>1 && " +
                   std::to_string(count) + ">" + std::to_string(grain) + ")");
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
      });

  enterLocal("mojo_row.c_owner_call",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               int64_t count = view.attrOfType<IntegerAttr>("count").getInt();
               int64_t grain = view.attrOfType<IntegerAttr>("grain").getInt();
               bool cyclic = view.attrOfType<BoolAttr>("cyclic").getValue();
               auto callArgs = view.attrOfType<StringAttr>("call_args").str();
               std::string loExpr =
                   cyclic ? ("ch*" + std::to_string(grain) + "LL") : "lo";
               std::string hiExpr =
                   "(" + std::to_string(count) + "LL-" + loExpr + "<" +
                   std::to_string(grain) + "LL?" + std::to_string(count) +
                   "LL:" + loExpr + "+" + std::to_string(grain) + "LL)";
               c.line("owner_task(" + callArgs +
                      (callArgs.empty() ? "" : ",") + loExpr + ", " + hiExpr +
                      ");");
               return llvm::Error::success();
             });

  // ---------------- Row ("row.mojo" & "entry.c") Local Ops ----------------
  rules["mojo_row.row_fn"] = PrintRule::local(
      [](const LocalOpView &view, PrintPoint at, PrintContext &c) -> llvm::Error {
        if (at == PrintPoint::BeginBlock) {
          c.line("from std.math import exp, sqrt, rsqrt, max, isnan");
          c.line("from std.memory import bitcast");
          if (view.attrOfType<BoolAttr>("uses_atomic").getValue())
            c.line("from std.atomic import Atomic");
          if (view.attrOfType<BoolAttr>("uses_stack").getValue())
            c.line("from std.memory import unsafe_stack_allocation");
          if (view.attrOfType<BoolAttr>("uses_memcpy").getValue())
            c.line("from std.memory import unsafe_memcpy");
          auto bArgs = view.blockArguments();
          auto specs = view.attrOfType<ArrayAttr>("param_specs");
          SmallVector<std::string> params;
          for (size_t i = 0; i < specs.size(); ++i)
            params.push_back(bArgs[i] + ": " +
                             cast<StringAttr>(specs[i]).getValue().str());
          c.line("@export(\"intent_row_interval\")");
          c.line("def intent_row_interval(" + llvm::join(params, ", ") +
                 ") abi(\"C\"):");
          c.indent++;
        } else if (at == PrintPoint::EndBlock) {
          c.indent--;
        }
        return llvm::Error::success();
      });

  enterLocal("mojo_row.dyn_address",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               int64_t terms = view.attrOfType<IntegerAttr>("terms").getInt();
               SmallVector<std::string> parts{"Int64(0)"};
               for (int64_t i = 0; i < terms; ++i)
                 parts.push_back("Int64(" + view.operand(1 + i).str() +
                                 ") * Int64(" +
                                 view.operand(1 + terms + i).str() + ")");
               c.line("var " + view.result(0).str() + " = Int(" +
                      llvm::join(parts, " + ") + ")");
               return llvm::Error::success();
             });

  enterLocal("mojo_row.simd_iota_add",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto dt = view.attrOfType<StringAttr>("dtype").getValue().str();
               int64_t w = view.attrOfType<IntegerAttr>("width").getInt();
               auto base = view.operand(0).str(), res = view.result(0).str();
               if (w == 1) {
                 c.line("var " + res + " = Scalar[DType." + dt + "](" + base +
                        ")");
               } else {
                 SmallVector<std::string> seq;
                 for (int64_t i = 0; i < w; ++i)
                   seq.push_back(std::to_string(i));
                 std::string sTy =
                     "SIMD[DType." + dt + "," + std::to_string(w) + "]";
                 c.line("var " + res + " = " + sTy + "(" + base + ") + " + sTy +
                        "(" + llvm::join(seq, ",") + ")");
               }
               return llvm::Error::success();
             });

  enterLocal("mojo_row.scalar_cast",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto dt = view.attrOfType<StringAttr>("dtype").getValue().str();
               auto in = view.operand(0).str(), res = view.result(0).str();
               Type inTy = view.operandType(0), outTy = view.resultType(0);
               if (isa<ShapedType>(outTy))
                 c.line("var " + res + " = " + in + ".cast[DType." + dt +
                        "]()");
               else if (outTy.isInteger(1))
                 c.line("var " + res + " = Bool(" + in + " != 0)");
               else if (inTy.isInteger(1))
                 c.line("var " + res + " = Scalar[DType." + dt + "](1) if " +
                        in + " else Scalar[DType." + dt + "](0)");
               else if (inTy.isIndex())
                 c.line("var " + res + " = Int64(" + in + ").cast[DType." + dt +
                        "]()");
               else
                 c.line("var " + res + " = " + in + ".cast[DType." + dt +
                        "]()");
               return llvm::Error::success();
             });

  enterLocal("mojo_row.tuple_decl",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               if (view.numOperands() == 3)
                 c.line("var " + view.result(0).str() + " = (" +
                        view.operand(0).str() + ", " + view.operand(1).str() +
                        ", " + view.operand(2).str() + ")");
               else
                 c.line("var " + view.result(0).str() + " = " +
                        view.operand(0).str());
               return llvm::Error::success();
             });

  enterLocal("mojo_row.uniform",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               int64_t w = view.attrOfType<IntegerAttr>("width").getInt();
               if (w > 1 && !isa<ShapedType>(view.operandType(1))) {
                 c.line("var " + view.result(0).str() +
                        " = uniform_contiguous[" + std::to_string(w) + "](" +
                        view.operand(0).str() + ", " + view.operand(1).str() +
                        ")");
               } else {
                 std::string pos =
                     w == 1
                         ? ("SIMD[DType.uint64,1](" + view.operand(1).str() +
                            ")")
                         : view.operand(1).str();
                 c.line("var " + view.result(0).str() + " = uniform_at[" +
                        std::to_string(w) + "](" + view.operand(0).str() +
                        ", " + pos + ")");
               }
               return llvm::Error::success();
             });

  enterLocal("mojo_row.welford_vec_step",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               int64_t w = view.attrOfType<IntegerAttr>("width").getInt();
               auto r0 = view.result(0).str(), r1 = view.result(1).str();
               c.line("var " + r0 + "_p = welford_vec_step[" +
                      std::to_string(w) + "](" + view.operand(0).str() + ", " +
                      view.operand(1).str() + ", " + view.operand(2).str() +
                      ", " + view.operand(3).str() + ")");
               c.line("var " + r0 + " = " + r0 + "_p[0]");
               c.line("var " + r1 + " = " + r0 + "_p[1]");
               return llvm::Error::success();
             });

  enterLocal("mojo_row.simd_scan_sum",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto dt = view.attrOfType<StringAttr>("dtype").getValue().str();
               int64_t w = view.attrOfType<IntegerAttr>("width").getInt();
               auto r0 = view.result(0).str(), r1 = view.result(1).str();
               std::string inExpr =
                   w == 1
                       ? ("SIMD[DType." + dt + ",1](" + view.operand(0).str() +
                          ")")
                       : view.operand(0).str();
               c.line("var " + r0 + "_p = simd_scan_sum[DType." + dt + "," +
                      std::to_string(w) + "](" + inExpr + ", Scalar[DType." +
                      dt + "](" + view.operand(1).str() + "))");
               c.line("var " + r0 + " = " +
                      (w == 1 ? (r0 + "_p[0][0]") : (r0 + "_p[0]")));
               c.line("var " + r1 + " = " + r0 + "_p[1]");
               return llvm::Error::success();
             });

  rules["mojo_row.c_row_host_fn"] = PrintRule::local(
      [](const LocalOpView &view, PrintPoint at, PrintContext &c) -> llvm::Error {
        if (at == PrintPoint::Enter) {
          c.line("#include <stdint.h>");
          c.line("#include <omp.h>");
          c.line(view.attrOfType<StringAttr>("extern_decl").str());
          c.line("int intent_entry(" +
                 view.attrOfType<StringAttr>("params").str() + ") {");
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
      });

  enterLocal("mojo_row.c_row_prep",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               if (view.attrOfType<BoolAttr>("single_dispatcher").getValue())
                 c.line("workers = 1;");
               if (view.attrOfType<BoolAttr>("member_tasks").getValue())
                 c.line("rows = cols;");
               if (view.attrOfType<BoolAttr>("swap_dest").getValue()) {
                 c.line("int64_t t = rows; rows = cols; cols = t;");
                 c.line("t = a0s0; a0s0 = a0s1; a0s1 = t;");
                 c.line("t = a1s0; a1s0 = a1s1; a1s1 = t;");
               }
               return llvm::Error::success();
             });

  rules["mojo_row.c_key_omp_for"] = PrintRule::local(
      [](const LocalOpView &view, PrintPoint at, PrintContext &c) -> llvm::Error {
        int64_t keys = view.attrOfType<IntegerAttr>("key_extent").getInt();
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
      });

  rules["mojo_row.c_tile_2d_omp"] = PrintRule::local(
      [](const LocalOpView &view, PrintPoint at, PrintContext &c) -> llvm::Error {
        int64_t step = view.attrOfType<IntegerAttr>("tile").getInt();
        std::string st = std::to_string(step) + "LL";
        if (at == PrintPoint::Enter) {
          c.line("const int64_t nt = (cols - 1) / " + st +
                 " + 1, mt = (rows - 1) / " + st + " + 1;");
          c.line("#pragma omp parallel for schedule(static) num_threads(workers) if(workers>1)");
          c.line("for (int64_t g = 0; g < mt * nt; ++g) {");
        } else if (at == PrintPoint::BeginBlock) {
          c.indent++;
          c.line("const int64_t r = (g / nt) * " + st + ", c = (g % nt) * " +
                 st + ";");
          c.line("const int64_t nr = rows - r < " + st + " ? rows - r : " + st +
                 ", nc = cols - c < " + st + " ? cols - c : " + st + ";");
        } else if (at == PrintPoint::EndBlock) {
          c.indent--;
          c.line("}");
        }
        return llvm::Error::success();
      });

  rules["mojo_row.c_row_single_thread"] = PrintRule::local(
      [](const LocalOpView &, PrintPoint at, PrintContext &c) -> llvm::Error {
        if (at == PrintPoint::Enter) {
          c.line("const int64_t groups = (rows - 1) / grain + 1;");
          c.line("const int64_t threads = groups < workers ? groups : workers;");
          c.line("if (threads <= 1) {");
        } else if (at == PrintPoint::BeginBlock) {
          c.indent++;
          c.line("int64_t start = 0, stop = rows;");
        } else if (at == PrintPoint::EndBlock) {
          c.line("return 0;");
          c.indent--;
          c.line("}");
        }
        return llvm::Error::success();
      });

  rules["mojo_row.c_row_omp_parallel"] = PrintRule::local(
      [](const LocalOpView &view, PrintPoint at, PrintContext &c) -> llvm::Error {
        bool dyn =
            view.attrOfType<StringAttr>("schedule").getValue() == "dynamic";
        int64_t chunk = view.attrOfType<IntegerAttr>("chunk_size").getInt();
        if (at == PrintPoint::Enter) {
          if (dyn) {
            c.line("#pragma omp parallel for num_threads(threads) schedule(dynamic, " +
                   std::to_string(chunk) + ")");
            c.line("for (int64_t g = 0; g < groups; ++g) {");
          } else {
            c.line("#pragma omp parallel num_threads(threads)");
            c.line("{");
          }
        } else if (at == PrintPoint::BeginBlock) {
          c.indent++;
          if (dyn) {
            c.line("int64_t start = g * grain;");
            c.line("int64_t stop = start + grain < rows ? start + grain : rows;");
          } else {
            c.line("int64_t b = omp_get_thread_num(), nt = omp_get_num_threads();");
            c.line("int64_t q = groups / nt, rem = groups % nt;");
            c.line("int64_t start = (q * b + (b < rem ? b : rem)) * grain;");
            c.line("int64_t stop = b + 1 == nt ? rows : start + (q + (b < rem)) * grain;");
          }
        } else if (at == PrintPoint::EndBlock) {
          c.indent--;
          c.line("}");
        }
        return llvm::Error::success();
      });

  rules["mojo_row.c_flat_col_omp"] = PrintRule::local(
      [](const LocalOpView &view, PrintPoint at, PrintContext &c) -> llvm::Error {
        int64_t cg = view.attrOfType<IntegerAttr>("col_grain").getInt();
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
          c.line("int64_t cstart = (q * b + (b < rem ? b : rem)) * " + cgs +
                 ";");
          c.line("int64_t cstop = b + 1 == nt ? cols : cstart + (q + (b < rem)) * " +
                 cgs + ";");
        } else if (at == PrintPoint::EndBlock) {
          c.indent--;
          c.line("}");
        }
        return llvm::Error::success();
      });

  rules["mojo_row.c_projected_while"] = PrintRule::local(
      [](const LocalOpView &view, PrintPoint at, PrintContext &c) -> llvm::Error {
        int64_t pRank = view.attrOfType<IntegerAttr>("proj_rank").getInt();
        if (at == PrintPoint::Enter) {
          c.line("while (start < stop) {");
        } else if (at == PrintPoint::BeginBlock) {
          c.indent++;
          c.line("int64_t outer = start;");
          for (int64_t ax = pRank - 2; ax >= 0; --ax)
            c.line("int64_t c" + std::to_string(ax) + " = outer % a0d" +
                   std::to_string(ax) + "; outer /= a0d" + std::to_string(ax) +
                   ";");
          c.line("int64_t span = a0d" + std::to_string(pRank - 2) + " - c" +
                 std::to_string(pRank - 2) + ";");
          c.line("int64_t end = stop - start < span ? stop : start + span;");
        } else if (at == PrintPoint::EndBlock) {
          c.line("start = end;");
          c.indent--;
          c.line("}");
        }
        return llvm::Error::success();
      });

  enterLocal("mojo_row.c_row_call",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto mode = view.attrOfType<StringAttr>("mode").getValue();
               auto args = view.attrOfType<StringAttr>("call_args").str();
               int64_t bytes =
                   view.attrOfType<IntegerAttr>("elem_bytes").getInt();
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
                 c.line("intent_row_interval(" + pre +
                        "start, end, cols, columns);");
               } else if (mode == "members") {
                 c.line("intent_row_interval(" + pre +
                        "start, start + 1, stop - start, columns);");
               } else if (mode == "flat_cols") {
                 c.line("intent_row_interval(" + pre +
                        "0, rows, cstop - cstart, columns);");
               } else {
                 c.line("intent_row_interval(" + pre +
                        "start, stop, cols, columns);");
               }
               return llvm::Error::success();
             });

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
