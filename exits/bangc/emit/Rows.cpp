#include "../legalize/Rows.h"
#include "compilerinfra/Product.h"
#include "intentdsl/levels/algorithm/kir/ir/Literal.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"
#include <climits>

namespace CI_UNIT_NAMESPACE {
using namespace mlir;
using namespace compilerinfra;

namespace {
std::string ctype(Type t) {
  if (isa<TupleType>(t))
    return "Moment";
  if (auto s = dyn_cast<ShapedType>(t))
    t = s.getElementType();
  if (t.isBF16())
    return "unsigned short";
  if (t.isInteger(1))
    return "unsigned char";
  if (t.isSignedInteger(8))
    return "signed char";
  if (t.isInteger(8) || isa<Float8E4M3FNType>(t))
    return "unsigned char";
  if (t.isInteger(16))
    return t.isUnsignedInteger() ? "unsigned short" : "short";
  if (t.isInteger(32))
    return t.isUnsignedInteger() ? "unsigned int" : "int";
  if (t.isIndex() || t.isSignlessInteger(64) || t.isSignedInteger(64))
    return "long long";
  if (t.isUnsignedInteger(64))
    return "unsigned long long";
  return t.isF16() ? "half" : "float";
}

std::string unsignedCtype(Type t) {
  if (t.isInteger(1) || t.isInteger(8))
    return "unsigned char";
  if (t.isInteger(16))
    return "unsigned short";
  if (t.isInteger(32))
    return "unsigned int";
  return "unsigned long long";
}

std::string formatArgs(const LocalOpView &view, bool decl) {
  std::vector<std::string> parts;
  for (unsigned i = 0; i < view.numOperands(); ++i) {
    auto n = view.operand(i).str();
    Type ty = view.operandType(i);
    auto m = dyn_cast<MemRefType>(ty);
    if (decl)
      parts.push_back(ctype(ty) + (m ? " *" : " ") + n);
    else
      parts.push_back(n);
    if (m) {
      for (int j = 0; j < m.getRank(); ++j)
        parts.push_back((decl ? "long long " : "") + n + "_d" +
                        std::to_string(j));
      for (int j = 0; j < m.getRank(); ++j)
        parts.push_back((decl ? "long long " : "") + n + "_s" +
                        std::to_string(j));
    }
  }
  if (decl) {
    parts.push_back("long long M");
    parts.push_back("long long N");
    parts.push_back("long long grid");
  } else {
    parts.push_back("M");
    parts.push_back("N");
    parts.push_back("grid");
  }
  return llvm::join(parts, ", ");
}

void loadRuntimeSources(TerminalPrinter &p) {
  auto manifest = llvm::json::parse(ci_manifest());
  if (!manifest) {
    llvm::consumeError(manifest.takeError());
    return;
  }
  auto *obj = manifest->getAsObject();
  auto *ks = obj ? obj->getObject("kernel_sources") : nullptr;
  if (!ks)
    return;
  static const std::pair<const char *, const char *> bindings[] = {
      {"exits/bangc::row-reduce", "runtime/bangc/kernels/reduce.mlu"},
      {"exits/bangc::validated-exp", "runtime/bangc/kernels/exp.mlu"},
      {"exits/bangc::validated-exp2", "runtime/bangc/kernels/exp.mlu"},
      {"exits/bangc::validated-log", "runtime/bangc/kernels/exp.mlu"},
      {"exits/bangc::ieee-vec-cmpf", "runtime/bangc/kernels/exp.mlu"},
      {"exits/bangc::ieee-vec-bitselect", "runtime/bangc/kernels/exp.mlu"},
      {"exits/bangc::quant-lut-init", "runtime/bangc/kernels/quant.mlu"},
      {"exits/bangc::quant-record-lut-init", "runtime/bangc/kernels/quant.mlu"},
      {"exits/bangc::quant-e8m0-guard", "runtime/bangc/kernels/quant.mlu"},
      {"exits/bangc::quant-block-scaled", "runtime/bangc/kernels/quant.mlu"},
      {"exits/bangc::quant-grouped-rhs", "runtime/bangc/kernels/quant.mlu"},
      {"exits/bangc::quant-record-gemv", "runtime/bangc/kernels/quant.mlu"},
      {"exits/bangc::nram-transpose", "runtime/bangc/kernels/transpose.mlu"},
      {"exits/bangc::certified-half-quotient",
       "runtime/bangc/kernels/quotient.mlu"},
      {"exits/bangc::bf16-tile-convert", "runtime/bangc/kernels/bf16.mlu"},
      {"exits/bangc::bf16-tile-product", "runtime/bangc/kernels/bf16.mlu"},
      {"exits/bangc::bf16-scalar-convert", "runtime/bangc/kernels/bf16.mlu"},
      {"exits/bangc::ieee-sqrt-rn", "runtime/bangc/kernels/ieee_math.mlu"},
      {"exits/bangc::ieee-div-rn", "runtime/bangc/kernels/ieee_math.mlu"},
      {"exits/bangc::inclusive-scan", "runtime/bangc/kernels/scan.mlu"},
      {"exits/bangc::moments-reduce", "runtime/bangc/kernels/moments.mlu"},
      {"exits/bangc::moments-merge", "runtime/bangc/kernels/moments.mlu"},
      {"exits/bangc::philox4x32-linear", "runtime/bangc/kernels/philox.mlu"},
      {"exits/bangc::philox4x32-element", "runtime/bangc/kernels/philox.mlu"},
      {"exits/bangc::ieee-vec-min", "runtime/bangc/kernels/matrix.mlu"},
      {"exits/bangc::bf16-panel-guard", "runtime/bangc/kernels/matrix.mlu"},
      {"exits/bangc::bf16-ieee-dot", "runtime/bangc/kernels/matrix.mlu"},
      {"exits/bangc::bf16-acc-narrow", "runtime/bangc/kernels/matrix.mlu"},
      {"exits/bangc::i32-to-bf16", "runtime/bangc/kernels/matrix.mlu"},
  };
  for (auto [key, res] : bindings)
    if (auto txt = ks->getString(res))
      p.runtimeSources[key] = txt->str();
}
} // namespace

TerminalPrinter rowPrinter() {
  TerminalPrinter p;
  loadRuntimeSources(p);
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

  // Declarative format rules (zero C++ logic needed).
  rules["builtin.module"] = PrintRule::format("");
  rules["func.return"] = PrintRule::format("");
  rules["bangc_src.row_program"] = PrintRule::format(
      "#include <bang.h>\n#include <cnrt.h>\n#include <math.h>\n"
      "constexpr int R = INTENT_TILE_R, B = {batch}, EB = R * B, W = "
      "INTENT_TILE_W, P = INTENT_TILE_P, BYTES = INTENT_TILE_BYTES;");
  rules["bangc_src.matrix_program"] = PrintRule::format(
      "#include <bang.h>\n#include <cnrt.h>\n#include <math.h>\n"
      "constexpr int BM = {tile_m}, BN = {tile_n}, BK = {tile_k};");
  rules["bangc_src.ordered_program"] = PrintRule::format(
      "#include <bang.h>\n#include <cnrt.h>\n#include <math.h>");
  rules["bangc_src.nram_alloc"] = PrintRule::format(
      "__nram__ unsigned char $r0[BYTES] __attribute__((aligned(128)));");
  rules["bangc_src.is_mpu"] =
      PrintRule::format("unsigned char $r0 = __is_mpu();");
  rules["bangc_src.dim"] = PrintRule::format("long long $r0 = $0_d{axis};");
  rules["bangc_src.stride"] = PrintRule::format("long long $r0 = $0_s{axis};");
  rules["bangc_src.symbol"] = PrintRule::format("long long $r0 = {name};");
  rules["bangc_src.moment_array"] =
      PrintRule::format("Moment $r0[{extent}] = {{}};");
  rules["bangc_src.moment_get"] = PrintRule::format("Moment $r0 = $0[$1];");
  rules["bangc_src.moment_set"] = PrintRule::format("$0[$1] = $2;");
  rules["bangc_src.summary_zero"] =
      PrintRule::format("Moment $r0 = {{0, 0.0f, 0.0f}};");
  rules["bangc_src.summary_reduce"] =
      PrintRule::format("Moment $r0 = welford_row($0, $1, (int)($2));");
  rules["bangc_src.summary_merge"] =
      PrintRule::format("Moment $r0 = merge_moments($0, $1);");
  rules["bangc_src.atomic_add"] =
      PrintRule::format("__bang_atomic_add($0, $1, $2, (int)($3));");
  rules["bangc_src.vec_binary"] =
      PrintRule::format("__bang_{op}($0, $1, $2, (int)($3));");
  rules["bangc_src.positive_half_or_nan"] = PrintRule::format(
      "__bang_ge_scalar((short *)$0, (const short *)$1, (short)-1023, "
      "(int)($2)); __bang_mul((short *)$0, (const short *)$1, (const short "
      "*)$0, (int)($2));");
  rules["bangc_src.row_exp"] =
      PrintRule::format("row_exp($0, $1, $2, (int)($3), (int)($4));");
  rules["bangc_src.row_exp2"] =
      PrintRule::format("row_exp2($0, $1, $2, (int)($3), (int)($4));");
  rules["bangc_src.row_log"] =
      PrintRule::format("row_log($0, $1, $2, (int)($3), (int)($4));");
  rules["bangc_src.quant_init_lut"] = PrintRule::format(
      "quant_init_byte_lut_f16($0, $1, $2, $3, $4, $5);");
  rules["bangc_src.quant_init_record_lut"] = PrintRule::format(
      "quant_init_record_2plane_lut($0, $1, $2, $3, $4, $5, $6);");
  rules["bangc_src.quant_e8m0_ok"] = PrintRule::format(
      "unsigned char $r0 = quant_e8m0_panel_ok((const unsigned char *)$0, "
      "(int)($1), (const unsigned char *)$2, (int)($3), (short *)$4, $5, $6);");
  rules["bangc_src.quant_decode_block_scaled"] = PrintRule::format(
      "quant_decode_block_scaled_f16($0, (const unsigned char *)$1, "
      "(unsigned char *)$2, $3, (unsigned int *)$4, $5, $6, (int)($7), "
      "(int)($8), (int)($9), (int)($10), $11, $12);");
  rules["bangc_src.reduce32_sum"] = PrintRule::format(
      "__bang_reduce_sum($0, $0, 32); __sync_compute(); float $r0 = ((volatile "
      "float *)$0)[0];");
  rules["bangc_src.sumpool"] = PrintRule::format(
      "__bang_sumpool($0, $1, (int)($2), (int)($3), 1, (int)($3), 1, 1, 1);");
  rules["bangc_src.cycle_mul"] =
      PrintRule::format("__bang_cycle_mul($0, $1, $2, (int)($3), (int)($4));");
  rules["bangc_src.vec_min_scalar"] = PrintRule::format(
      "vec_min_scalar_ieee($0, $1, $2, $3, (int)($4), (int)($5));");
  rules["bangc_src.bf16_panel_ok"] = PrintRule::format(
      "unsigned char $r0 = ordinary_panel((const unsigned short *)$0, "
      "(int)($3), (unsigned short *)$1, (unsigned short *)$2);");
  rules["bangc_src.bf16_ieee_dot"] = PrintRule::format(
      "bf16_ieee_dot_panel($0, (const unsigned short *)$1, (const unsigned "
      "short *)$2, (int)($3), (int)($4), (int)($5), (int)($6), (int)($7));");
  rules["bangc_src.bf16_acc_narrow"] = PrintRule::format(
      "bf16_acc_narrow((unsigned short *)$0, $1, (int)($2), $3);");
  rules["bangc_src.i32_to_bf16"] =
      PrintRule::format("unsigned short $r0 = signed_to_bf16($0);");
  rules["bangc_src.from_le_half"] = PrintRule::format(
      "float $r0; {{ union {{ unsigned short u; half h; }} _h; _h.u = "
      "(unsigned short)(((unsigned)($0) & 0xffu) | (((unsigned)($1) & 0xffu) "
      "<< 8)); $r0 = (float)_h.h; }}");
  rules["bangc_src.narrow_quotient"] = PrintRule::format(
      "group_narrow_quotient($0, $1, $2, $3, $4, (int)($5), (int)($6), "
      "(int)($7), (int)($8));");
  rules["bangc_src.bf16_product"] = PrintRule::format(
      "bf16_product_tile($0, $1, $2, $3, (int)($4), (int)($5));");
  rules["bangc_src.vec_sqrt_rn"] =
      PrintRule::format("vec_sqrt_rn($0, $1, $2, (int)($3), (int)($4));");
  rules["bangc_src.vec_sqrt"] = PrintRule::format("__bang_sqrt($0, $1, (int)($2));");
  rules["bangc_src.vec_div_approx"] =
      PrintRule::format("reciprocal_product($0, $1, $2, $3, (int)($4));");
  rules["bangc_src.scan_sum"] =
      PrintRule::format("tile_prefix($0, $1, $2, (int)($3), (int)($4));");
  rules["bangc_src.philox_uniform"] = PrintRule::format(
      "philox4x32_linear($0, (unsigned *)$1, (unsigned long long)($2), "
      "(unsigned long long)($3), (int)($4));");
  rules["bangc_src.philox_at"] = PrintRule::format(
      "float $r0 = uniform_at((unsigned long long)($0), (unsigned long "
      "long)($1));");

  // Strictly local rules consuming only LocalOpView.
  rules["func.func"] = PrintRule::local(
      [](const LocalOpView &, PrintPoint pt, PrintContext &c) -> llvm::Error {
        if (pt == PrintPoint::Enter)
          c.file = "kernel.mlu";
        return llvm::Error::success();
      });

  enterLocal("bangc_src.runtime_source",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto key = view.attrOfType<StringAttr>("implementation");
               if (!key)
                 return llvm::createStringError(
                     "runtime_source missing implementation attribute");
               auto src = c.source(key.getValue());
               if (!src)
                 return src.takeError();
               c.line(*src);
               return llvm::Error::success();
             });

  rules["bangc_src.device_kernel"] =
      PrintRule::local([](const LocalOpView &view, PrintPoint pt,
                          PrintContext &c) -> llvm::Error {
        if (pt == PrintPoint::Enter) {
          c.line("__mlu_global__ void tile(" + formatArgs(view, true) + ") {");
          ++c.indent;
        } else if (pt == PrintPoint::Leave) {
          --c.indent;
          c.line("}");
        }
        return llvm::Error::success();
      });

  enterLocal("bangc_src.host_enqueue",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               c.line("extern \"C\" int intent_enqueue(" +
                      formatArgs(view, true) + ", int cores, void *queue) {");
               ++c.indent;
               c.line("if (grid == 0) return 0;");
               auto ft = view.attrOfType<StringAttr>("func_type");
               if (ft && ft.getValue() == "union4") {
                 c.line("cnrtDim3_t dim = {(unsigned)cores, 1, 1};");
                 c.line(
                     "tile<<<dim, cnrtFuncTypeUnion4, (cnrtQueue_t)queue>>>(" +
                     formatArgs(view, false) + ");");
               } else {
                 c.line("cnrtDim3_t dim = {(unsigned)(grid < cores ? grid : "
                        "cores), 1, 1};");
                 c.line("tile<<<dim, cnrtFuncTypeBlock, (cnrtQueue_t)queue>>>(" +
                        formatArgs(view, false) + ");");
               }
               c.line("return (int)cnrtGetLastError();");
               --c.indent;
               c.line("}");
               return llvm::Error::success();
             });

  enterLocal("bangc_src.cnnl_matmul",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto ls = view.attrOfType<IntegerAttr>("lhs_slot").getInt();
               auto rs = view.attrOfType<IntegerAttr>("rhs_slot").getInt();
               auto ds = view.attrOfType<IntegerAttr>("dst_slot").getInt();
               auto B = view.attrOfType<IntegerAttr>("batch").getInt();
               auto M = view.attrOfType<IntegerAttr>("m").getInt();
               auto N = view.attrOfType<IntegerAttr>("n").getInt();
               auto K = view.attrOfType<IntegerAttr>("k").getInt();
               auto ta = view.attrOfType<IntegerAttr>("trans_a").getInt();
               auto tb = view.attrOfType<IntegerAttr>("trans_b").getInt();
               auto lbs =
                   view.attrOfType<IntegerAttr>("lhs_batch_stride").getInt();
               auto rbs =
                   view.attrOfType<IntegerAttr>("rhs_batch_stride").getInt();
               auto dbs =
                   view.attrOfType<IntegerAttr>("dst_batch_stride").getInt();
               auto inDt = view.attrOfType<IntegerAttr>("in_dtype").getInt();
               auto outDt = view.attrOfType<IntegerAttr>("out_dtype").getInt();
               c.line("extern \"C\" int intent_cnnl_matmul(const void *, const "
                      "void *, void *, long long, long long, long long, long "
                      "long, int, int, long long, long long, long long, int, "
                      "int, void *);");
               c.line("extern \"C\" int intent_enqueue(" +
                      formatArgs(view, true) + ", int cores, void *queue) {");
               ++c.indent;
               c.line("(void)M; (void)N; (void)grid; (void)cores;");
               c.line("return intent_cnnl_matmul(" +
                      view.operand(ls).str() + ", " + view.operand(rs).str() +
                      ", " + view.operand(ds).str() + ", " + std::to_string(B) +
                      "LL, " + std::to_string(M) + "LL, " + std::to_string(N) +
                      "LL, " + std::to_string(K) + "LL, " + std::to_string(ta) +
                      ", " + std::to_string(tb) + ", " + std::to_string(lbs) +
                      "LL, " + std::to_string(rbs) + "LL, " +
                      std::to_string(dbs) + "LL, " + std::to_string(inDt) +
                      ", " + std::to_string(outDt) + ", queue);");
               --c.indent;
               c.line("}");
               return llvm::Error::success();
             });

  enterLocal("bangc_src.cnnl_row_op",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto kind = view.attrOfType<StringAttr>("kind").getValue();
               auto dt = view.attrOfType<IntegerAttr>("dtype_code").getInt();
               auto eb = view.attrOfType<IntegerAttr>("elem_bytes").getInt();
               if (kind == "gather_rows") {
                 auto table = view.operand(0).str();
                 auto ids = cast<MemRefType>(view.operandType(1));
                 c.line("extern \"C\" int intent_cnnl_gather_rows(const void *, const void *, "
                        "void *, long long, long long, long long, int, int, void *);");
                 c.line("extern \"C\" int intent_enqueue(" + formatArgs(view, true) +
                        ", int cores, void *queue) {");
                 ++c.indent;
                 c.line("return intent_cnnl_gather_rows(" + table + ", " + view.operand(1).str() +
                        ", " + view.operand(2).str() + ", " + table + "_d0, M, N, " +
                        std::to_string(eb) + ", " +
                        std::to_string(ids.getElementTypeBitWidth()) + ", queue);");
                 --c.indent;
                 c.line("}");
                 return llvm::Error::success();
               }
               if (kind == "transpose") {
                 c.line("extern \"C\" int intent_cnnl_transpose(const void *, "
                        "void *, long long, long long, int, void *);");
                 c.line("extern \"C\" int intent_enqueue(" +
                        formatArgs(view, true) + ", int cores, void *queue) {");
                 ++c.indent;
                 c.line("(void)grid; (void)cores;");
                 c.line("return intent_cnnl_transpose(" +
                        view.operand(0).str() + ", " + view.operand(1).str() +
                        ", M, N, " + std::to_string(eb) + ", queue);");
                 --c.indent;
                 c.line("}");
                 return llvm::Error::success();
               }
               if (kind == "softmax") {
                 c.line("extern \"C\" int intent_cnnl_softmax(const void *, "
                        "void *, long long, long long, int, void *);");
                 c.line("extern \"C\" int intent_enqueue(" +
                        formatArgs(view, true) + ", int cores, void *queue) {");
                 ++c.indent;
                 c.line("(void)grid; (void)cores;");
                 c.line("return intent_cnnl_softmax(" + view.operand(0).str() +
                        ", " + view.operand(1).str() + ", M, N, " +
                        std::to_string(dt) + ", queue);");
                 --c.indent;
                 c.line("}");
                 return llvm::Error::success();
               }
               if (kind == "layer_norm") {
                 c.line("extern \"C\" int intent_cnnl_layer_norm(const void *, "
                        "const void *, const void *, void *, long long, long "
                        "long, float, int, void *);");
                 c.line("extern \"C\" int intent_enqueue(" +
                        formatArgs(view, true) + ", int cores, void *queue) {");
                 ++c.indent;
                 c.line("(void)grid; (void)cores;");
                 c.line("return intent_cnnl_layer_norm(" +
                        view.operand(0).str() + ", " + view.operand(1).str() +
                        ", " + view.operand(2).str() + ", " +
                        view.operand(3).str() + ", M, N, (float)(" +
                        view.operand(4).str() + "), " + std::to_string(dt) +
                        ", queue);");
                 --c.indent;
                 c.line("}");
                 return llvm::Error::success();
               }
               return llvm::createStringError("unknown cnnl_row_op kind");
             });

  enterLocal("bangc_src.nram_buffer",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto t = ctype(view.resultType(0));
               auto elems = view.attrOfType<IntegerAttr>("elements").getInt();
               c.line("__nram__ " + t + " " + view.result(0).str() + "[" +
                      std::to_string(elems) + "] __attribute__((aligned(128)));");
               return llvm::Error::success();
             });

  enterLocal("bangc_src.wram_buffer",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto t = ctype(view.resultType(0));
               auto elems = view.attrOfType<IntegerAttr>("elements").getInt();
               c.line("__wram__ " + t + " " + view.result(0).str() + "[" +
                      std::to_string(elems) + "] __attribute__((aligned(128)));");
               return llvm::Error::success();
             });

  enterLocal("bangc_src.sram_buffer",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto t = ctype(view.resultType(0));
               auto elems = view.attrOfType<IntegerAttr>("elements").getInt();
               c.line("__mlu_shared__ " + t + " " + view.result(0).str() + "[" +
                      std::to_string(elems) + "] __attribute__((aligned(128)));");
               return llvm::Error::success();
             });

  enterLocal("bangc_src.nram_ptr",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto t = ctype(view.resultType(0));
               c.line(t + " *" + view.result(0).str() + " = (" + t + " *)(" +
                      view.operand(0).str() + " + " + view.operand(1).str() +
                      ");");
               return llvm::Error::success();
             });

  enterLocal("bangc_src.address",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto segs =
                   view.attrOfType<DenseI32ArrayAttr>("operandSegmentSizes");
               unsigned n = segs[1];
               int64_t bits =
                   view.attrOfType<IntegerAttr>("index_bits").getInt();
               std::string sum;
               if (n == 0) {
                 sum = "0LL";
               } else {
                 std::vector<std::string> terms;
                 for (unsigned i = 0; i < n; ++i) {
                   auto cv = view.operand(1 + i).str();
                   auto sv = view.operand(1 + n + i).str();
                   if (bits == 32)
                     terms.push_back("((int)(" + cv + ") * (int)(" + sv + "))");
                   else
                     terms.push_back("((long long)(" + cv +
                                     ") * (long long)(" + sv + "))");
                 }
                 sum = llvm::join(terms, " + ");
                 if (bits == 32)
                   sum = "(long long)(" + sum + ")";
               }
               auto t = ctype(view.resultType(0));
               c.line(t + " *" + view.result(0).str() + " = (" + t + " *)" +
                      view.operand(0).str() + " + (" + sum + ");");
               return llvm::Error::success();
             });

  enterLocal("bangc_src.local_var",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto t = ctype(view.resultType(0));
               auto ext = view.attrOfType<StringAttr>("extent").str();
               c.line(t + " " + view.result(0).str() + "[" + ext + "];");
               return llvm::Error::success();
             });

  enterLocal("bangc_src.summary_field",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto pos = view.attrOfType<IntegerAttr>("position").getInt();
               std::string field = pos == 0 ? ".n" : pos == 1 ? ".mean" : ".m2";
               c.line(ctype(view.resultType(0)) + " " + view.result(0).str() +
                      " = " + view.operand(0).str() + field + ";");
               return llvm::Error::success();
             });

  enterLocal("bangc_src.memcpy_1d",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               bool async = view.attrOfType<BoolAttr>("async").getValue();
               auto dir = view.attrOfType<StringAttr>("direction").str();
               auto elemTy = ctype(view.operandType(0));
               c.line(std::string(async ? "__memcpy_async" : "__memcpy") + "(" +
                      view.operand(0).str() + ", " + view.operand(1).str() +
                      ", (int)(" + view.operand(2).str() + ") * (int)sizeof(" +
                      elemTy + "), " + dir + ");");
               return llvm::Error::success();
             });

  enterLocal("bangc_src.memcpy_2d",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               bool async = view.attrOfType<BoolAttr>("async").getValue();
               auto dir = view.attrOfType<StringAttr>("direction").str();
               auto elemTy = ctype(view.operandType(0));
               c.line(std::string(async ? "__memcpy_async" : "__memcpy") + "(" +
                      view.operand(0).str() + ", " + view.operand(1).str() +
                      ", (int)(" + view.operand(2).str() + ") * (int)sizeof(" +
                      elemTy + "), " + dir + ", (int)(" +
                      view.operand(3).str() + ") * (int)sizeof(" + elemTy +
                      "), (int)(" + view.operand(4).str() + ") * (int)sizeof(" +
                      elemTy + "), (int)(" + view.operand(5).str() + "));");
               return llvm::Error::success();
             });

  enterLocal("bangc_src.load_scalar",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto t = ctype(view.resultType(0));
               c.line(t + " " + view.result(0).str() + " = (" + t + ")(" +
                      view.operand(0).str() + "[" + view.operand(1).str() +
                      "]);");
               return llvm::Error::success();
             });

  enterLocal("bangc_src.store_scalar",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto elemTy = ctype(view.operandType(0));
               c.line(view.operand(0).str() + "[" + view.operand(1).str() +
                      "] = (" + elemTy + ")(" + view.operand(2).str() + ");");
               return llvm::Error::success();
             });

  enterLocal("bangc_src.sync",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto dom = view.attrOfType<StringAttr>("domain").getValue();
               c.line(dom == "io"             ? "__sync_io();"
                      : dom == "compute"      ? "__sync_compute();"
                      : dom == "cluster"      ? "__sync_cluster();"
                      : dom == "sram_to_nram" ? "__sync_copy_sram_to_nram();"
                      : dom == "sram_to_wram" ? "__sync_copy_sram_to_wram();"
                      : dom == "dram_to_nram" ? "__sync_copy_dram_to_nram();"
                      : dom == "move"         ? "__sync_move();"
                                              : "__sync();");
               return llvm::Error::success();
             });

  enterLocal("bangc_src.fill",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto t = ctype(view.operandType(0));
               c.line("__bang_write_value(" + view.operand(0).str() +
                      ", (int)(" + view.operand(1).str() + "), (" + t + ")(" +
                      view.operand(2).str() + "));");
               return llvm::Error::success();
             });

  enterLocal("bangc_src.vec_scalar",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto k = view.attrOfType<StringAttr>("op").getValue();
               auto sc = view.operand(2).str();
               std::string fnName = (k == "sub") ? "add" : k.str();
               std::string scExpr = (k == "sub") ? "-(" + sc + ")" : sc;
               c.line("__bang_" + fnName + "_scalar(" + view.operand(0).str() +
                      ", " + view.operand(1).str() + ", " + scExpr + ", (int)(" +
                      view.operand(3).str() + "));");
               return llvm::Error::success();
             });

  enterLocal("bangc_src.vec_cast",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto k = view.attrOfType<StringAttr>("kind").getValue();
               std::string fnName = k == "f32_to_f16" ? "__bang_float2half_rn"
                                                      : "__bang_half2float";
               c.line(fnName + "(" + view.operand(0).str() + ", " +
                      view.operand(1).str() + ", (int)(" +
                      view.operand(2).str() + "));");
               return llvm::Error::success();
             });

  enterLocal("bangc_src.scalar_math",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto k = view.attrOfType<StringAttr>("op").getValue();
               auto res = view.result(0).str();
               auto t = ctype(view.resultType(0));
               auto a0 = view.operand(0).str();
               if (k == "volatile_mul") {
                 c.line("volatile " + t + " " + res + "_v = " + a0 + " * " +
                        view.operand(1).str() + "; " + t + " " + res + " = " +
                        res + "_v;");
                 return llvm::Error::success();
               }
               std::string expr;
               if (k == "sqrt")
                 expr = "sqrtf(" + a0 + ")";
               else if (k == "rsqrt")
                 expr = "(1.0f / sqrtf(" + a0 + "))";
               else if (k == "exp")
                 expr = "expf(" + a0 + ")";
               else if (k == "exp2")
                 expr = "exp2f(" + a0 + ")";
               else if (k == "log")
                 expr = "logf(" + a0 + ")";
               else if (k == "fabs")
                 expr = "fabsf(" + a0 + ")";
               else if (k == "floor")
                 expr = "floorf(" + a0 + ")";
               else if (k == "min_ieee") {
                 auto a1 = view.operand(1).str();
                 expr = "(isnan((float)(" + a0 + ")) ? (" + a0 +
                        ") : (isnan((float)(" + a1 + ")) ? (" + a1 +
                        ") : fminf(" + a0 + ", " + a1 + ")))";
               } else if (k == "max_num")
                 expr = "fmaxf(" + a0 + ", " + view.operand(1).str() + ")";
               else if (k == "maximum_nan") {
                 auto a1 = view.operand(1).str();
                 expr = "(isnan((float)(" + a0 + ")) || isnan((float)(" + a1 +
                        ")) ? NAN : fmaxf(" + a0 + ", " + a1 + "))";
               } else
                 expr = "(" + t + ")(" + a0 + ")";
               c.line(t + " " + res + " = " + expr + ";");
               return llvm::Error::success();
             });

  enterLocal("bangc_src.row_reduce",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto alg = view.attrOfType<StringAttr>("algebra").getValue();
               auto modeAttr = view.attrOfType<StringAttr>("mode");
               StringRef mode = modeAttr ? modeAttr.getValue() : "full";
               if (alg == "sum" && mode == "stage") {
                 c.line("group_sum_stage(" + view.operand(1).str() + ", " +
                        view.operand(2).str() + ", (int)(" +
                        view.operand(3).str() + "), (int)(" +
                        view.operand(4).str() + "), (int)(" +
                        view.operand(5).str() + "));");
                 return llvm::Error::success();
               }
               if (alg == "sum" && mode == "finish") {
                 c.line("group_sum_finish(" + view.operand(0).str() + ", " +
                        view.operand(1).str() + ", " + view.operand(2).str() +
                        ", (int)(" + view.operand(3).str() + "), (int)(" +
                        view.operand(4).str() + "), (int)(" +
                        view.operand(5).str() + "));");
                 return llvm::Error::success();
               }
               std::string fnName =
                   alg == "sum" ? "group_sum" : "group_max_num";
               c.line(fnName + "(" + view.operand(0).str() + ", " +
                      view.operand(1).str() + ", " + view.operand(2).str() +
                      ", (int)(" + view.operand(3).str() + "), (int)(" +
                      view.operand(4).str() + "), (int)(" +
                      view.operand(5).str() + "));");
               return llvm::Error::success();
             });

  enterLocal("bangc_src.tile_transpose",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto ty = cast<MemRefType>(view.operandType(0));
               unsigned bits = ty.getElementTypeBitWidth();
               std::string fnName = bits == 8    ? "transpose_tile_b8"
                                    : bits == 16 ? "transpose_tile_b16"
                                                 : "transpose_tile_b32";
               c.line(fnName + "(" + view.operand(0).str() + ", " +
                      view.operand(1).str() + ", (int)(" +
                      view.operand(2).str() + "), (int)(" +
                      view.operand(3).str() + "));");
               return llvm::Error::success();
             });

  enterLocal("bangc_src.wram_conv",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto mode = view.attrOfType<StringAttr>("mode").getValue();
               auto bm =
                   std::to_string(view.attrOfType<IntegerAttr>("bm").getInt());
               auto bn =
                   std::to_string(view.attrOfType<IntegerAttr>("bn").getInt());
               auto bk =
                   std::to_string(view.attrOfType<IntegerAttr>("bk").getInt());
               if (mode == "i8") {
                 auto acc = view.operand(0).str(),
                      part = view.operand(1).str(),
                      lhs = view.operand(2).str(),
                      flt = view.operand(3).str();
                 c.line("__bang_conv(" + part + ", (signed char *)" + lhs +
                        ", (signed char *)" + flt + ", " + bk + ", " + bm +
                        ", 1, 1, 1, 1, 1, " + bn + ", 0);");
                 c.line("__bang_float2int32_rn((int *)" + part + ", " + part +
                        ", " + bm + " * " + bn + ", 0);");
                 c.line("__bang_add(" + acc + ", " + acc + ", (int *)" + part +
                        ", " + bm + " * " + bn + ");");
                 return llvm::Error::success();
               }
               auto acc = view.operand(0).str(), lhs = view.operand(1).str(),
                    flt = view.operand(2).str(), accum = view.operand(3).str();
               std::string ptrTy = mode == "bf16" ? "bfloat16_t" : "half";
               c.line("if (" + accum + ") __bang_conv_partial(" + acc + ", (" +
                      ptrTy + " *)" + lhs + ", (" + ptrTy + " *)" + flt +
                      ", " + acc + ", " + bk + ", " + bm +
                      ", 1, 1, 1, 1, 1, " + bn + "); else __bang_conv(" + acc +
                      ", (" + ptrTy + " *)" + lhs + ", (" + ptrTy + " *)" +
                      flt + ", " + bk + ", " + bm + ", 1, 1, 1, 1, 1, " + bn +
                      ");");
               return llvm::Error::success();
             });

  enterLocal("bangc_src.bf16_tile",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto mode = view.attrOfType<StringAttr>("mode").getValue();
               unsigned n = view.numOperands();
               if (mode == "widen")
                 c.line("widen_bf16_tile(" + view.operand(0).str() + ", " +
                        view.operand(1).str() + ", (int)(" +
                        view.operand(n - 1).str() + "));");
               else if (n == 5)
                 c.line("narrow_bf16_tile_fast(" + view.operand(0).str() +
                        ", " + view.operand(1).str() + ", (unsigned short *)" +
                        view.operand(2).str() + ", (int)(" +
                        view.operand(3).str() + "), (int)(" +
                        view.operand(4).str() + "));");
               else
                 c.line("narrow_bf16_tile(" + view.operand(0).str() + ", " +
                        view.operand(1).str() + ", (int)(" +
                        view.operand(2).str() + "), (int)(" +
                        view.operand(3).str() + "));");
               return llvm::Error::success();
             });

  enterLocal("bangc_src.vec_div_rn",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto mode = view.attrOfType<StringAttr>("mode").getValue();
               if (mode == "vg") {
                 c.line("group_div_scalar_rn(" + view.operand(0).str() + ", " +
                        view.operand(1).str() + ", " + view.operand(2).str() +
                        ", " + view.operand(3).str() + ", (int)(" +
                        view.operand(4).str() + "), (int)(" +
                        view.operand(5).str() + "), (int)(" +
                        view.operand(6).str() + "), (int)(" +
                        view.operand(7).str() + "));");
                 return llvm::Error::success();
               }
               std::string fnName = mode == "vv"   ? "vec_div_rn"
                                    : mode == "vs" ? "vec_div_scalar_rn"
                                                   : "scalar_div_vec_rn";
               c.line(fnName + "(" + view.operand(0).str() + ", " +
                      view.operand(1).str() + ", " + view.operand(2).str() +
                      ", " + view.operand(3).str() + ", (int)(" +
                      view.operand(4).str() + "), (int)(" +
                      view.operand(5).str() + "));");
               return llvm::Error::success();
             });

  enterLocal("bangc_src.vec_cmpf_mask",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               bool isVec = view.attrOfType<BoolAttr>("rhs_is_vec").getValue();
               int32_t pc =
                   view.attrOfType<IntegerAttr>("pred_code").getInt();
               std::string fnName =
                   isVec ? "vec_cmpf_mask_vv" : "vec_cmpf_mask_vs";
               c.line(fnName + "((int *)" + view.operand(0).str() +
                      ", (int *)" + view.operand(1).str() + ", " +
                      view.operand(2).str() + ", " + view.operand(3).str() +
                      ", (int)(" + view.operand(4).str() + "), " +
                      std::to_string(pc) + ");");
               return llvm::Error::success();
             });

  enterLocal("bangc_src.vec_bitselect",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto mode = view.attrOfType<StringAttr>("mode").getValue();
               std::string fnName = mode == "vv"   ? "vec_bitselect_vv"
                                    : mode == "vs" ? "vec_bitselect_vs"
                                                   : "vec_bitselect_sv";
               c.line(fnName + "(" + view.operand(0).str() + ", (int *)" +
                      view.operand(1).str() + ", (const int *)" +
                      view.operand(2).str() + ", " + view.operand(3).str() +
                      ", " + view.operand(4).str() + ", (int)(" +
                      view.operand(5).str() + "));");
               return llvm::Error::success();
             });

  enterLocal("bangc_src.quant_decode_grouped_rhs",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto sh = view.attrOfType<DenseI32ArrayAttr>("lane_shifts");
               if (!sh || sh.size() != 8)
                 return llvm::createStringError(
                     "quant_decode_grouped_rhs requires 8 lane_shifts");
               std::string shArgs;
               for (int i = 0; i < 8; ++i)
                 shArgs += ", " + std::to_string(sh[i]);
               c.line("quant_decode_grouped_rhs_f16(" + view.operand(0).str() +
                      ", (const unsigned char *)" + view.operand(1).str() +
                      ", " + view.operand(2).str() + ", (const int *)" +
                      view.operand(3).str() + ", " + view.operand(4).str() +
                      ", (unsigned int *)" + view.operand(5).str() + ", " +
                      view.operand(6).str() + ", " + view.operand(7).str() +
                      ", (int)(" + view.operand(8).str() + "), (int)(" +
                      view.operand(9).str() + "), (int)(" +
                      view.operand(10).str() + "), (int)(" +
                      view.operand(11).str() + "), (int)(" +
                      view.operand(12).str() + "), (int)(" +
                      view.operand(13).str() + "), " + view.operand(14).str() +
                      shArgs + ");");
               return llvm::Error::success();
             });

  enterLocal("bangc_src.quant_record_gemv",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto pp = view.attrOfType<DenseI32ArrayAttr>("plane_params");
               if (!pp || pp.size() != 10)
                 return llvm::createStringError(
                     "quant_record_gemv requires 10 plane_params");
               std::string pArgs;
               for (int i = 0; i < 10; ++i)
                 pArgs += ", " + std::to_string(pp[i]);
               c.line("quant_record_gemv_2plane_q8k(" + view.operand(0).str() +
                      ", (const unsigned char *)" + view.operand(1).str() +
                      ", (unsigned char *)" + view.operand(2).str() + ", " +
                      view.operand(3).str() + ", " + view.operand(4).str() +
                      ", (unsigned short *)" + view.operand(5).str() + ", " +
                      view.operand(6).str() + ", " + view.operand(7).str() +
                      ", " + view.operand(8).str() + ", " +
                      view.operand(9).str() + ", (int)(" +
                      view.operand(10).str() + ")" + pArgs + ");");
               return llvm::Error::success();
             });

  enterLocal("bangc_src.bf16_scalar",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto mode = view.attrOfType<StringAttr>("mode").getValue();
               auto res = view.result(0).str();
               auto t = ctype(view.resultType(0));
               auto a0 = view.operand(0).str();
               std::string expr =
                   mode == "widen"    ? "widen_bf16(" + a0 + ")"
                   : mode == "narrow" ? "round_bf16(" + a0 + ")"
                                      : "round_bf16(widen_bf16(" + a0 +
                                            ") * widen_bf16(" +
                                            view.operand(1).str() + "))";
               c.line(t + " " + res + " = " + expr + ";");
               return llvm::Error::success();
             });

  // Standard SCF structured control-flow rules using LocalOpView + carry_id.
  rules["scf.for"] = PrintRule::local(
      [](const LocalOpView &view, PrintPoint pt, PrintContext &c) -> llvm::Error {
        unsigned nRes = view.numResults();
        auto idAttr = view.attrOfType<IntegerAttr>("bangc.carry_id");
        if (nRes > 0 && !idAttr)
          return llvm::createStringError(
              "scf.for with iter_args requires bangc.carry_id");
        std::string prefix =
            idAttr ? "_carry_" + std::to_string(idAttr.getInt()) + "_" : "";
        if (pt == PrintPoint::Enter) {
          for (unsigned i = 0; i < nRes; ++i)
            c.line(ctype(view.resultType(i)) + " " + prefix +
                   std::to_string(i) + " = " + view.operand(3 + i).str() + ";");
        } else if (pt == PrintPoint::BeginBlock) {
          auto args = view.blockArguments();
          auto argTypes = view.blockArgumentTypes();
          if (args.size() != 1 + nRes)
            return llvm::createStringError("scf.for block arity mismatch");
          auto iv = args[0];
          c.line("for (long long " + iv + " = " + view.operand(0).str() + "; " +
                 iv + " < " + view.operand(1).str() + "; " + iv +
                 " += " + view.operand(2).str() + ") {");
          ++c.indent;
          for (unsigned i = 0; i < nRes; ++i)
            c.line(ctype(argTypes[1 + i]) + " " + args[1 + i] + " = " + prefix +
                   std::to_string(i) + ";");
        } else if (pt == PrintPoint::EndBlock) {
          --c.indent;
          c.line("}");
        } else if (pt == PrintPoint::Leave) {
          for (unsigned i = 0; i < nRes; ++i)
            c.line(ctype(view.resultType(i)) + " " + view.result(i).str() +
                   " = " + prefix + std::to_string(i) + ";");
        }
        return llvm::Error::success();
      });

  rules["scf.if"] = PrintRule::local(
      [](const LocalOpView &view, PrintPoint pt, PrintContext &c) -> llvm::Error {
        unsigned nRes = view.numResults();
        auto idAttr = view.attrOfType<IntegerAttr>("bangc.carry_id");
        if (nRes > 0 && !idAttr)
          return llvm::createStringError(
              "scf.if with results requires bangc.carry_id");
        std::string prefix =
            idAttr ? "_carry_" + std::to_string(idAttr.getInt()) + "_" : "";
        bool hasElse = view.numRegions() > 1 && view.numBlocks(1) > 0;
        if (pt == PrintPoint::Enter) {
          for (unsigned i = 0; i < nRes; ++i)
            c.line(ctype(view.resultType(i)) + " " + prefix +
                   std::to_string(i) + ";");
          c.line("if (" + view.operand(0).str() + ") {");
        } else if (pt == PrintPoint::BeginRegion) {
          if (c.region == 0) {
            ++c.indent;
          } else if (hasElse) {
            c.line("else {");
            ++c.indent;
          }
        } else if (pt == PrintPoint::EndRegion) {
          if (c.region == 0 || hasElse) {
            --c.indent;
            c.line("}");
          }
        } else if (pt == PrintPoint::Leave) {
          for (unsigned i = 0; i < nRes; ++i)
            c.line(ctype(view.resultType(i)) + " " + view.result(i).str() +
                   " = " + prefix + std::to_string(i) + ";");
        }
        return llvm::Error::success();
      });

  rules["scf.while"] = PrintRule::local(
      [](const LocalOpView &view, PrintPoint pt, PrintContext &c) -> llvm::Error {
        unsigned nRes = view.numResults();
        auto idAttr = view.attrOfType<IntegerAttr>("bangc.carry_id");
        if (nRes > 0 && !idAttr)
          return llvm::createStringError(
              "scf.while with results requires bangc.carry_id");
        std::string prefix =
            idAttr ? "_carry_" + std::to_string(idAttr.getInt()) + "_" : "";
        if (pt == PrintPoint::Enter) {
          for (unsigned i = 0; i < nRes; ++i)
            c.line(ctype(view.resultType(i)) + " " + prefix +
                   std::to_string(i) + " = " + view.operand(i).str() + ";");
          c.line("while (true) {");
        } else if (pt == PrintPoint::BeginBlock) {
          ++c.indent;
          auto args = view.blockArguments();
          auto argTypes = view.blockArgumentTypes();
          for (unsigned i = 0; i < args.size() && i < nRes; ++i)
            c.line(ctype(argTypes[i]) + " " + args[i] + " = " + prefix +
                   std::to_string(i) + ";");
        } else if (pt == PrintPoint::EndBlock) {
          --c.indent;
        } else if (pt == PrintPoint::Leave) {
          c.line("}");
          for (unsigned i = 0; i < nRes; ++i)
            c.line(ctype(view.resultType(i)) + " " + view.result(i).str() +
                   " = " + prefix + std::to_string(i) + ";");
        }
        return llvm::Error::success();
      });

  enterLocal("scf.condition",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               unsigned nFwd = view.numOperands() > 0 ? view.numOperands() - 1 : 0;
               if (nFwd > 0) {
                 auto idAttr = view.attrOfType<IntegerAttr>("bangc.carry_id");
                 if (!idAttr)
                   return llvm::createStringError(
                       "value-carrying scf.condition requires bangc.carry_id");
                 std::string prefix =
                     "_carry_" + std::to_string(idAttr.getInt()) + "_";
                 for (unsigned i = 0; i < nFwd; ++i)
                   c.line("auto _cond_" + std::to_string(i) + " = " +
                          view.operand(1 + i).str() + ";");
                 for (unsigned i = 0; i < nFwd; ++i)
                   c.line(prefix + std::to_string(i) + " = _cond_" +
                          std::to_string(i) + ";");
               }
               c.line("if (!(" + view.operand(0).str() + ")) break;");
               return llvm::Error::success();
             });

  enterLocal("scf.yield",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               unsigned nYld = view.numOperands();
               if (nYld == 0)
                 return llvm::Error::success();
               auto idAttr = view.attrOfType<IntegerAttr>("bangc.carry_id");
               if (!idAttr)
                 return llvm::createStringError(
                     "value-carrying scf.yield requires bangc.carry_id");
               std::string prefix =
                   "_carry_" + std::to_string(idAttr.getInt()) + "_";
               for (unsigned i = 0; i < nYld; ++i)
                 c.line("auto _yld_" + std::to_string(i) + " = " +
                        view.operand(i).str() + ";");
               for (unsigned i = 0; i < nYld; ++i)
                 c.line(prefix + std::to_string(i) + " = _yld_" +
                        std::to_string(i) + ";");
               return llvm::Error::success();
             });

  // Standard Arith scalar rules (all LocalOpView).
  enterLocal("arith.constant",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto v = view.attr("value");
               auto resTy = view.resultType(0);
               auto t = ctype(resTy);
               std::string lit;
               if (auto ia = dyn_cast<IntegerAttr>(v)) {
                 if (resTy.isInteger(1))
                   lit = ia.getValue().isZero() ? "0" : "1";
                 else if (resTy.isUnsignedInteger(64))
                   lit = std::to_string(ia.getUInt()) + "ULL";
                 else if (resTy.isSignedInteger()) {
                   int64_t val = ia.getSInt();
                   lit = std::to_string(val);
                 } else if (resTy.isInteger(8) || resTy.isInteger(16)) {
                   lit = std::to_string(ia.getInt());
                 } else if (resTy.isInteger(32)) {
                   int64_t val = ia.getInt();
                   lit = val == INT32_MIN ? "(-2147483647 - 1)"
                                          : std::to_string(val);
                 } else {
                   int64_t val = ia.getInt();
                   lit = val == LLONG_MIN ? "(-9223372036854775807LL - 1LL)"
                                          : std::to_string(val) + "LL";
                 }
               } else {
                 auto fa = cast<FloatAttr>(v);
                 if (resTy.isBF16() || isa<Float8E4M3FNType>(resTy)) {
                   lit = std::to_string(
                             fa.getValue().bitcastToAPInt().getZExtValue()) +
                         "U";
                 } else if (fa.getValue().isInfinity()) {
                   lit = fa.getValue().isNegative() ? "-INFINITY" : "INFINITY";
                 } else if (fa.getValue().isNaN()) {
                   lit = "NAN";
                 } else {
                   lit = intentdsl::kir::literal(fa) + "f";
                 }
               }
               c.line(t + " " + view.result(0).str() + " = (" + t + ")(" + lit +
                      ");");
               return llvm::Error::success();
             });

  for (auto [opName, tok] :
       {std::pair{"arith.addi", "+"}, {"arith.subi", "-"}, {"arith.muli", "*"}}) {
    enterLocal(opName,
               [tok](const LocalOpView &view, PrintContext &c) -> llvm::Error {
                 auto resTy = view.resultType(0);
                 auto t = ctype(resTy);
                 auto a = view.operand(0).str(), b = view.operand(1).str();
                 std::string expr = resTy.isInteger(32)
                                        ? "(int)((unsigned)(" + a + ") " + tok +
                                              " (unsigned)(" + b + "))"
                                        : "(" + a + ") " + tok + " (" + b + ")";
                 c.line(t + " " + view.result(0).str() + " = " + expr + ";");
                 return llvm::Error::success();
               });
  }

  for (auto [opName, tok] : {std::pair{"arith.divsi", "/"},
                             {"arith.remsi", "%"},
                             {"arith.andi", "&"},
                             {"arith.ori", "|"},
                             {"arith.xori", "^"},
                             {"arith.addf", "+"},
                             {"arith.subf", "-"},
                             {"arith.mulf", "*"},
                             {"arith.divf", "/"}}) {
    enterLocal(opName,
               [tok](const LocalOpView &view, PrintContext &c) -> llvm::Error {
                 auto t = ctype(view.resultType(0));
                 c.line(t + " " + view.result(0).str() + " = (" + t + ")((" +
                        view.operand(0).str() + ") " + tok + " (" +
                        view.operand(1).str() + "));");
                 return llvm::Error::success();
               });
  }

  for (auto [opName, tok] :
       {std::pair{"arith.divui", "/"}, {"arith.remui", "%"}}) {
    enterLocal(opName,
               [tok](const LocalOpView &view, PrintContext &c) -> llvm::Error {
                 auto resTy = view.resultType(0);
                 auto t = ctype(resTy);
                 auto u = unsignedCtype(resTy);
                 c.line(t + " " + view.result(0).str() + " = (" + t + ")((" +
                        u + ")(" + view.operand(0).str() + ") " + tok + " (" +
                        u + ")(" + view.operand(1).str() + "));");
                 return llvm::Error::success();
               });
  }

  enterLocal("arith.shli",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto resTy = view.resultType(0);
               auto t = ctype(resTy);
               auto u = unsignedCtype(resTy);
               c.line(t + " " + view.result(0).str() + " = (" + t + ")((" + u +
                      ")(" + view.operand(0).str() + ") << (" +
                      view.operand(1).str() + "));");
               return llvm::Error::success();
             });

  enterLocal("arith.shrsi",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto t = ctype(view.resultType(0));
               c.line(t + " " + view.result(0).str() + " = (" +
                      view.operand(0).str() + ") >> (" +
                      view.operand(1).str() + ");");
               return llvm::Error::success();
             });

  enterLocal("arith.shrui",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto resTy = view.resultType(0);
               auto t = ctype(resTy);
               auto u = unsignedCtype(resTy);
               c.line(t + " " + view.result(0).str() + " = (" + t + ")((" + u +
                      ")(" + view.operand(0).str() + ") >> (" +
                      view.operand(1).str() + "));");
               return llvm::Error::success();
             });

  enterLocal("arith.bitcast",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto fromT = ctype(view.operandType(0));
               auto toT = ctype(view.resultType(0));
               auto res = view.result(0).str();
               c.line(toT + " " + res + "; { union { " + fromT + " a; " + toT +
                      " b; } _bc; _bc.a = " + view.operand(0).str() + "; " +
                      res + " = _bc.b; }");
               return llvm::Error::success();
             });

  enterLocal("arith.minsi",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto t = ctype(view.resultType(0));
               auto a = view.operand(0).str(), b = view.operand(1).str();
               c.line(t + " " + view.result(0).str() + " = (" + a + " < " + b +
                      " ? " + a + " : " + b + ");");
               return llvm::Error::success();
             });

  enterLocal("arith.maxsi",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto t = ctype(view.resultType(0));
               auto a = view.operand(0).str(), b = view.operand(1).str();
               c.line(t + " " + view.result(0).str() + " = (" + a + " > " + b +
                      " ? " + a + " : " + b + ");");
               return llvm::Error::success();
             });

  enterLocal("arith.minui",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto resTy = view.resultType(0);
               auto t = ctype(resTy), u = unsignedCtype(resTy);
               auto a = view.operand(0).str(), b = view.operand(1).str();
               c.line(t + " " + view.result(0).str() + " = ((" + u + ")(" + a +
                      ") < (" + u + ")(" + b + ") ? " + a + " : " + b + ");");
               return llvm::Error::success();
             });

  enterLocal("arith.maxui",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto resTy = view.resultType(0);
               auto t = ctype(resTy), u = unsignedCtype(resTy);
               auto a = view.operand(0).str(), b = view.operand(1).str();
               c.line(t + " " + view.result(0).str() + " = ((" + u + ")(" + a +
                      ") > (" + u + ")(" + b + ") ? " + a + " : " + b + ");");
               return llvm::Error::success();
             });

  enterLocal("arith.maxnumf",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto t = ctype(view.resultType(0));
               c.line(t + " " + view.result(0).str() + " = (" + t + ")fmaxf(" +
                      view.operand(0).str() + ", " + view.operand(1).str() +
                      ");");
               return llvm::Error::success();
             });

  enterLocal("arith.minnumf",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto t = ctype(view.resultType(0));
               c.line(t + " " + view.result(0).str() + " = (" + t + ")fminf(" +
                      view.operand(0).str() + ", " + view.operand(1).str() +
                      ");");
               return llvm::Error::success();
             });

  enterLocal("arith.cmpi",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto pAttr =
                   view.attrOfType<arith::CmpIPredicateAttr>("predicate");
               if (!pAttr)
                 return llvm::createStringError("arith.cmpi missing predicate");
               auto pred = pAttr.getValue();
               bool isUns = pred == arith::CmpIPredicate::ult ||
                            pred == arith::CmpIPredicate::ule ||
                            pred == arith::CmpIPredicate::ugt ||
                            pred == arith::CmpIPredicate::uge;
               std::string tok = pred == arith::CmpIPredicate::eq    ? "=="
                                 : pred == arith::CmpIPredicate::ne  ? "!="
                                 : pred == arith::CmpIPredicate::slt ? "<"
                                 : pred == arith::CmpIPredicate::sle ? "<="
                                 : pred == arith::CmpIPredicate::sgt ? ">"
                                 : pred == arith::CmpIPredicate::sge ? ">="
                                 : pred == arith::CmpIPredicate::ult ? "<"
                                 : pred == arith::CmpIPredicate::ule ? "<="
                                 : pred == arith::CmpIPredicate::ugt ? ">"
                                                                     : ">=";
               std::string a = "(" + view.operand(0).str() + ")";
               std::string b = "(" + view.operand(1).str() + ")";
               if (isUns) {
                 auto u = unsignedCtype(view.operandType(0));
                 a = "((" + u + ")" + a + ")";
                 b = "((" + u + ")" + b + ")";
               }
               c.line("unsigned char " + view.result(0).str() + " = (" + a +
                      " " + tok + " " + b + ");");
               return llvm::Error::success();
             });

  enterLocal("arith.cmpf",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto pAttr =
                   view.attrOfType<arith::CmpFPredicateAttr>("predicate");
               if (!pAttr)
                 return llvm::createStringError("arith.cmpf missing predicate");
               auto pred = pAttr.getValue();
               auto a = "((float)(" + view.operand(0).str() + "))";
               auto b = "((float)(" + view.operand(1).str() + "))";
               std::string expr;
               switch (pred) {
               case arith::CmpFPredicate::AlwaysFalse:
                 expr = "0";
                 break;
               case arith::CmpFPredicate::AlwaysTrue:
                 expr = "1";
                 break;
               case arith::CmpFPredicate::OEQ:
                 expr = "(" + a + " == " + b + ")";
                 break;
               case arith::CmpFPredicate::OGT:
                 expr = "(" + a + " > " + b + ")";
                 break;
               case arith::CmpFPredicate::OGE:
                 expr = "(" + a + " >= " + b + ")";
                 break;
               case arith::CmpFPredicate::OLT:
                 expr = "(" + a + " < " + b + ")";
                 break;
               case arith::CmpFPredicate::OLE:
                 expr = "(" + a + " <= " + b + ")";
                 break;
               case arith::CmpFPredicate::ONE:
                 expr = "(!isnan(" + a + ") && !isnan(" + b + ") && " + a +
                        " != " + b + ")";
                 break;
               case arith::CmpFPredicate::ORD:
                 expr = "(!isnan(" + a + ") && !isnan(" + b + "))";
                 break;
               case arith::CmpFPredicate::UEQ:
                 expr = "(isnan(" + a + ") || isnan(" + b + ") || " + a +
                        " == " + b + ")";
                 break;
               case arith::CmpFPredicate::UGT:
                 expr = "(isnan(" + a + ") || isnan(" + b + ") || " + a +
                        " > " + b + ")";
                 break;
               case arith::CmpFPredicate::UGE:
                 expr = "(isnan(" + a + ") || isnan(" + b + ") || " + a +
                        " >= " + b + ")";
                 break;
               case arith::CmpFPredicate::ULT:
                 expr = "(isnan(" + a + ") || isnan(" + b + ") || " + a +
                        " < " + b + ")";
                 break;
               case arith::CmpFPredicate::ULE:
                 expr = "(isnan(" + a + ") || isnan(" + b + ") || " + a +
                        " <= " + b + ")";
                 break;
               case arith::CmpFPredicate::UNE:
                 expr = "(" + a + " != " + b + ")";
                 break;
               case arith::CmpFPredicate::UNO:
                 expr = "(isnan(" + a + ") || isnan(" + b + "))";
                 break;
               }
               c.line("unsigned char " + view.result(0).str() + " = " + expr +
                      ";");
               return llvm::Error::success();
             });

  enterLocal("arith.select",
             [](const LocalOpView &view, PrintContext &c) -> llvm::Error {
               auto resTy = view.resultType(0);
               auto t = ctype(resTy);
               if (isa<MemRefType>(resTy)) {
                 c.line(t + " *" + view.result(0).str() + " = (" +
                        view.operand(0).str() + " ? (" + t + " *)" +
                        view.operand(1).str() + " : (" + t + " *)" +
                        view.operand(2).str() + ");");
                 return llvm::Error::success();
               }
               c.line(t + " " + view.result(0).str() + " = (" +
                      view.operand(0).str() + " ? " + view.operand(1).str() +
                      " : " + view.operand(2).str() + ");");
               return llvm::Error::success();
             });

  for (auto castOp :
       {"arith.extsi", "arith.extui", "arith.trunci", "arith.index_cast",
        "arith.index_castui", "arith.sitofp", "arith.uitofp", "arith.fptosi",
        "arith.fptoui", "arith.extf", "arith.truncf"}) {
    enterLocal(castOp,
               [castOp](const LocalOpView &view, PrintContext &c) -> llvm::Error {
                 auto t = ctype(view.resultType(0));
                 std::string src = view.operand(0).str();
                 llvm::StringRef opRef(castOp);
                 if (opRef == "arith.extui" || opRef == "arith.index_castui" ||
                     opRef == "arith.uitofp") {
                   auto u = unsignedCtype(view.operandType(0));
                   src = "(" + u + ")(" + src + ")";
                 }
                 c.line(t + " " + view.result(0).str() + " = (" + t + ")(" +
                        src + ");");
                 return llvm::Error::success();
               });
  }

  p.package = [](DictionaryAttr attrs,
                 const std::map<std::string, std::string> &files,
                 llvm::raw_pwrite_stream &os) -> llvm::Error {
    auto pkg = attrs.getAs<StringAttr>("bangc_src.package");
    if (!pkg)
      return llvm::createStringError("missing verified BANG C package");
    auto data = llvm::json::parse(pkg.getValue());
    if (!data)
      return data.takeError();
    auto file = files.find("kernel.mlu");
    if (!data->getAsObject() || file == files.end())
      return llvm::createStringError("incomplete BANG C kernel artifact");
    (*data->getAsObject())["source"] = file->second;
    os << *data;
    return llvm::Error::success();
  };

  return p;
}
} // namespace CI_UNIT_NAMESPACE
