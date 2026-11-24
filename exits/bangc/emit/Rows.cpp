#include "../legalize/Rows.h"
#include "compilerinfra/Product.h"
#include "intentdsl/levels/algorithm/kir/ir/Literal.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"
#include <climits>

namespace CI_UNIT_NAMESPACE {
using namespace mlir;
using namespace compilerinfra;

namespace {
std::string ctype(Type t) {
  if (auto tup = dyn_cast<TupleType>(t))
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
  if (t.isInteger(32))
    return "int";
  if (t.isIndex() || t.isSignlessInteger(64))
    return "long long";
  if (t.isUnsignedInteger(64))
    return "unsigned long long";
  return t.isF16() ? "half" : "float";
}

std::string formatArgs(Operation &op, PrintContext &c, bool decl) {
  std::vector<std::string> parts;
  for (Value a : op.getOperands()) {
    auto n = c.name(a);
    auto m = dyn_cast<MemRefType>(a.getType());
    if (decl)
      parts.push_back(ctype(a.getType()) + (m ? " *" : " ") + n);
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

  auto enter = [&](StringRef name,
                   std::function<llvm::Error(Operation &, PrintContext &)> fn) {
    rules[name.str()] = [fn = std::move(fn)](Operation &op, PrintPoint pt,
                                             PrintContext &c) -> llvm::Error {
      if (pt != PrintPoint::Enter)
        return llvm::Error::success();
      return fn(op, c);
    };
  };

  rules["builtin.module"] = [](Operation &, PrintPoint, PrintContext &) {
    return llvm::Error::success();
  };
  rules["func.func"] = [](Operation &op, PrintPoint pt, PrintContext &c) {
    if (pt == PrintPoint::Enter) {
      c.file = "kernel.mlu";
      for (Value arg : cast<func::FuncOp>(op).getArguments())
        (void)c.name(arg);
    }
    return llvm::Error::success();
  };
  enter("func.return", [](Operation &, PrintContext &) {
    return llvm::Error::success();
  });

  rules["bangc_src.row_program"] = [](Operation &op, PrintPoint pt,
                                      PrintContext &c) {
    if (pt == PrintPoint::Enter) {
      c.line("#include <bang.h>");
      c.line("#include <cnrt.h>");
      c.line("#include <math.h>");
      auto batch = op.getAttrOfType<IntegerAttr>("batch").getInt();
      c.line("constexpr int R = INTENT_TILE_R, B = " + std::to_string(batch) +
             ", EB = R * B, W = INTENT_TILE_W, P = INTENT_TILE_P, BYTES = "
             "INTENT_TILE_BYTES;");
    }
    return llvm::Error::success();
  };

  rules["bangc_src.matrix_program"] = [](Operation &op, PrintPoint pt,
                                         PrintContext &c) {
    if (pt == PrintPoint::Enter) {
      c.line("#include <bang.h>");
      c.line("#include <cnrt.h>");
      c.line("#include <math.h>");
      auto bm = op.getAttrOfType<IntegerAttr>("tile_m").getInt();
      auto bn = op.getAttrOfType<IntegerAttr>("tile_n").getInt();
      auto bk = op.getAttrOfType<IntegerAttr>("tile_k").getInt();
      c.line("constexpr int BM = " + std::to_string(bm) +
             ", BN = " + std::to_string(bn) +
             ", BK = " + std::to_string(bk) + ";");
    }
    return llvm::Error::success();
  };

  enter("bangc_src.runtime_source",
        [](Operation &op, PrintContext &c) -> llvm::Error {
          auto key = op.getAttrOfType<StringAttr>("implementation").getValue();
          auto src = c.source(key);
          if (!src)
            return src.takeError();
          c.line(*src);
          return llvm::Error::success();
        });

  rules["bangc_src.device_kernel"] = [](Operation &op, PrintPoint pt,
                                        PrintContext &c) {
    if (pt == PrintPoint::Enter) {
      c.line("__mlu_global__ void tile(" + formatArgs(op, c, true) + ") {");
      ++c.indent;
    } else if (pt == PrintPoint::Leave) {
      --c.indent;
      c.line("}");
    }
    return llvm::Error::success();
  };

  enter("bangc_src.host_enqueue", [](Operation &op, PrintContext &c) {
    c.line("extern \"C\" int intent_enqueue(" + formatArgs(op, c, true) +
           ", int cores, void *queue) {");
    ++c.indent;
    c.line("if (grid == 0) return 0;");
    c.line("cnrtDim3_t dim = {(unsigned)(grid < cores ? grid : cores), 1, 1};");
    c.line("tile<<<dim, cnrtFuncTypeBlock, (cnrtQueue_t)queue>>>(" +
           formatArgs(op, c, false) + ");");
    c.line("return (int)cnrtGetLastError();");
    --c.indent;
    c.line("}");
    return llvm::Error::success();
  });

  enter("bangc_src.nram_alloc", [](Operation &op, PrintContext &c) {
    c.line("__nram__ unsigned char " + c.name(op.getResult(0)) +
           "[BYTES] __attribute__((aligned(128)));");
    return llvm::Error::success();
  });

  enter("bangc_src.nram_buffer", [](Operation &op, PrintContext &c) {
    auto t = ctype(op.getResult(0).getType());
    auto elems = op.getAttrOfType<IntegerAttr>("elements").getInt();
    c.line("__nram__ " + t + " " + c.name(op.getResult(0)) + "[" +
           std::to_string(elems) + "] __attribute__((aligned(128)));");
    return llvm::Error::success();
  });

  enter("bangc_src.wram_buffer", [](Operation &op, PrintContext &c) {
    auto t = ctype(op.getResult(0).getType());
    auto elems = op.getAttrOfType<IntegerAttr>("elements").getInt();
    c.line("__wram__ " + t + " " + c.name(op.getResult(0)) + "[" +
           std::to_string(elems) + "] __attribute__((aligned(128)));");
    return llvm::Error::success();
  });

  enter("bangc_src.nram_ptr", [](Operation &op, PrintContext &c) {
    auto t = ctype(op.getResult(0).getType());
    c.line(t + " *" + c.name(op.getResult(0)) + " = (" + t + " *)(" +
           c.name(op.getOperand(0)) + " + " + c.name(op.getOperand(1)) + ");");
    return llvm::Error::success();
  });

  enter("bangc_src.address", [](Operation &op, PrintContext &c) {
    auto segs = op.getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes");
    unsigned n = segs[1];
    auto coords = op.getOperands().slice(1, n);
    auto strides = op.getOperands().slice(1 + n, n);
    int64_t bits = op.getAttrOfType<IntegerAttr>("index_bits").getInt();
    std::string sum;
    if (n == 0) {
      sum = "0LL";
    } else {
      std::vector<std::string> terms;
      for (auto [cv, sv] : llvm::zip(coords, strides)) {
        if (bits == 32)
          terms.push_back("((int)(" + c.name(cv) + ") * (int)(" + c.name(sv) +
                          "))");
        else
          terms.push_back("((long long)(" + c.name(cv) + ") * (long long)(" +
                          c.name(sv) + "))");
      }
      sum = llvm::join(terms, " + ");
      if (bits == 32)
        sum = "(long long)(" + sum + ")";
    }
    auto t = ctype(op.getResult(0).getType());
    c.line(t + " *" + c.name(op.getResult(0)) + " = (" + t + " *)" +
           c.name(op.getOperand(0)) + " + (" + sum + ");");
    return llvm::Error::success();
  });

  enter("bangc_src.dim", [](Operation &op, PrintContext &c) {
    c.line("long long " + c.name(op.getResult(0)) + " = " +
           c.name(op.getOperand(0)) + "_d" +
           std::to_string(op.getAttrOfType<IntegerAttr>("axis").getInt()) +
           ";");
    return llvm::Error::success();
  });

  enter("bangc_src.stride", [](Operation &op, PrintContext &c) {
    c.line("long long " + c.name(op.getResult(0)) + " = " +
           c.name(op.getOperand(0)) + "_s" +
           std::to_string(op.getAttrOfType<IntegerAttr>("axis").getInt()) +
           ";");
    return llvm::Error::success();
  });

  enter("bangc_src.symbol", [](Operation &op, PrintContext &c) {
    c.line("long long " + c.name(op.getResult(0)) + " = " +
           op.getAttrOfType<StringAttr>("name").str() + ";");
    return llvm::Error::success();
  });

  enter("bangc_src.local_var", [](Operation &op, PrintContext &c) {
    auto t = ctype(op.getResult(0).getType());
    auto ext = op.getAttrOfType<StringAttr>("extent").str();
    c.line(t + " " + c.name(op.getResult(0)) + "[" + ext + "];");
    return llvm::Error::success();
  });

  enter("bangc_src.moment_array", [](Operation &op, PrintContext &c) {
    auto ext = op.getAttrOfType<StringAttr>("extent").str();
    c.line("Moment " + c.name(op.getResult(0)) + "[" + ext + "] = {};");
    return llvm::Error::success();
  });

  enter("bangc_src.moment_get", [](Operation &op, PrintContext &c) {
    c.line("Moment " + c.name(op.getResult(0)) + " = " +
           c.name(op.getOperand(0)) + "[" + c.name(op.getOperand(1)) + "];");
    return llvm::Error::success();
  });

  enter("bangc_src.moment_set", [](Operation &op, PrintContext &c) {
    c.line(c.name(op.getOperand(0)) + "[" + c.name(op.getOperand(1)) +
           "] = " + c.name(op.getOperand(2)) + ";");
    return llvm::Error::success();
  });

  enter("bangc_src.summary_zero", [](Operation &op, PrintContext &c) {
    c.line("Moment " + c.name(op.getResult(0)) + " = {0, 0.0f, 0.0f};");
    return llvm::Error::success();
  });

  enter("bangc_src.summary_reduce", [](Operation &op, PrintContext &c) {
    c.line("Moment " + c.name(op.getResult(0)) + " = welford_row(" +
           c.name(op.getOperand(0)) + ", " + c.name(op.getOperand(1)) +
           ", (int)(" + c.name(op.getOperand(2)) + "));");
    return llvm::Error::success();
  });

  enter("bangc_src.summary_merge", [](Operation &op, PrintContext &c) {
    c.line("Moment " + c.name(op.getResult(0)) + " = merge_moments(" +
           c.name(op.getOperand(0)) + ", " + c.name(op.getOperand(1)) + ");");
    return llvm::Error::success();
  });

  enter("bangc_src.summary_field", [](Operation &op, PrintContext &c) {
    auto pos = op.getAttrOfType<IntegerAttr>("position").getInt();
    std::string field = pos == 0 ? ".n" : pos == 1 ? ".mean" : ".m2";
    c.line(ctype(op.getResult(0).getType()) + " " + c.name(op.getResult(0)) +
           " = " + c.name(op.getOperand(0)) + field + ";");
    return llvm::Error::success();
  });

  enter("bangc_src.memcpy_1d", [](Operation &op, PrintContext &c) {
    bool async = op.getAttrOfType<BoolAttr>("async").getValue();
    auto dir = op.getAttrOfType<StringAttr>("direction").str();
    auto elemTy = ctype(op.getOperand(0).getType());
    auto dst = c.name(op.getOperand(0)), src = c.name(op.getOperand(1)),
         cnt = c.name(op.getOperand(2));
    c.line(std::string(async ? "__memcpy_async" : "__memcpy") + "(" + dst +
           ", " + src + ", (int)(" + cnt + ") * (int)sizeof(" + elemTy + "), " +
           dir + ");");
    return llvm::Error::success();
  });

  enter("bangc_src.memcpy_2d", [](Operation &op, PrintContext &c) {
    bool async = op.getAttrOfType<BoolAttr>("async").getValue();
    auto dir = op.getAttrOfType<StringAttr>("direction").str();
    auto elemTy = ctype(op.getOperand(0).getType());
    auto dst = c.name(op.getOperand(0)), src = c.name(op.getOperand(1)),
         w = c.name(op.getOperand(2)), ds = c.name(op.getOperand(3)),
         ss = c.name(op.getOperand(4)), seg = c.name(op.getOperand(5));
    c.line(std::string(async ? "__memcpy_async" : "__memcpy") + "(" + dst +
           ", " + src + ", (int)(" + w + ") * (int)sizeof(" + elemTy + "), " +
           dir + ", (int)(" + ds + ") * (int)sizeof(" + elemTy + "), (int)(" +
           ss + ") * (int)sizeof(" + elemTy + "), (int)(" + seg + "));");
    return llvm::Error::success();
  });

  enter("bangc_src.load_scalar", [](Operation &op, PrintContext &c) {
    auto t = ctype(op.getResult(0).getType());
    c.line(t + " " + c.name(op.getResult(0)) + " = (" + t + ")(" +
           c.name(op.getOperand(0)) + "[" + c.name(op.getOperand(1)) + "]);");
    return llvm::Error::success();
  });

  enter("bangc_src.store_scalar", [](Operation &op, PrintContext &c) {
    auto elemTy = ctype(op.getOperand(0).getType());
    c.line(c.name(op.getOperand(0)) + "[" + c.name(op.getOperand(1)) +
           "] = (" + elemTy + ")(" + c.name(op.getOperand(2)) + ");");
    return llvm::Error::success();
  });

  enter("bangc_src.sync", [](Operation &op, PrintContext &c) {
    auto dom = op.getAttrOfType<StringAttr>("domain").getValue();
    c.line(dom == "io"        ? "__sync_io();"
           : dom == "compute" ? "__sync_compute();"
                              : "__sync();");
    return llvm::Error::success();
  });

  enter("bangc_src.fill", [](Operation &op, PrintContext &c) {
    auto t = ctype(op.getOperand(0).getType());
    c.line("__bang_write_value(" + c.name(op.getOperand(0)) + ", (int)(" +
           c.name(op.getOperand(1)) + "), (" + t + ")(" +
           c.name(op.getOperand(2)) + "));");
    return llvm::Error::success();
  });

  enter("bangc_src.atomic_add", [](Operation &op, PrintContext &c) {
    c.line("__bang_atomic_add(" + c.name(op.getOperand(0)) + ", " +
           c.name(op.getOperand(1)) + ", " + c.name(op.getOperand(2)) +
           ", (int)(" + c.name(op.getOperand(3)) + "));");
    return llvm::Error::success();
  });

  enter("bangc_src.vec_binary", [](Operation &op, PrintContext &c) {
    auto k = op.getAttrOfType<StringAttr>("op").str();
    c.line("__bang_" + k + "(" + c.name(op.getOperand(0)) + ", " +
           c.name(op.getOperand(1)) + ", " + c.name(op.getOperand(2)) +
           ", (int)(" + c.name(op.getOperand(3)) + "));");
    return llvm::Error::success();
  });

  enter("bangc_src.vec_scalar", [](Operation &op, PrintContext &c) {
    auto k = op.getAttrOfType<StringAttr>("op").getValue();
    auto sc = c.name(op.getOperand(2));
    std::string fnName = (k == "sub") ? "add" : k.str();
    std::string scExpr = (k == "sub") ? "-(" + sc + ")" : sc;
    c.line("__bang_" + fnName + "_scalar(" + c.name(op.getOperand(0)) + ", " +
           c.name(op.getOperand(1)) + ", " + scExpr + ", (int)(" +
           c.name(op.getOperand(3)) + "));");
    return llvm::Error::success();
  });

  enter("bangc_src.vec_cast", [](Operation &op, PrintContext &c) {
    auto k = op.getAttrOfType<StringAttr>("kind").getValue();
    std::string fnName =
        k == "f32_to_f16" ? "__bang_float2half_rn" : "__bang_half2float";
    c.line(fnName + "(" + c.name(op.getOperand(0)) + ", " +
           c.name(op.getOperand(1)) + ", (int)(" + c.name(op.getOperand(2)) +
           "));");
    return llvm::Error::success();
  });

  enter("bangc_src.positive_half_or_nan", [](Operation &op, PrintContext &c) {
    auto dst = c.name(op.getOperand(0)), src = c.name(op.getOperand(1)),
         lanes = c.name(op.getOperand(2));
    c.line("__bang_ge_scalar((short *)" + dst + ", (const short *)" + src +
           ", (short)-1023, (int)(" + lanes + "));");
    c.line("__bang_mul((short *)" + dst + ", (const short *)" + src +
           ", (const short *)" + dst + ", (int)(" + lanes + "));");
    return llvm::Error::success();
  });

  enter("bangc_src.scalar_math", [](Operation &op, PrintContext &c) {
    auto k = op.getAttrOfType<StringAttr>("op").getValue();
    auto res = c.name(op.getResult(0));
    auto t = ctype(op.getResult(0).getType());
    auto a0 = c.name(op.getOperand(0));
    if (k == "volatile_mul") {
      c.line("volatile " + t + " " + res + "_v = " + a0 + " * " +
             c.name(op.getOperand(1)) + "; " + t + " " + res + " = " + res +
             "_v;");
      return llvm::Error::success();
    }
    std::string expr;
    if (k == "sqrt")
      expr = "sqrtf(" + a0 + ")";
    else if (k == "rsqrt")
      expr = "(1.0f / sqrtf(" + a0 + "))";
    else if (k == "exp")
      expr = "expf(" + a0 + ")";
    else if (k == "fabs")
      expr = "fabsf(" + a0 + ")";
    else if (k == "floor")
      expr = "floorf(" + a0 + ")";
    else if (k == "min_ieee") {
      auto a1 = c.name(op.getOperand(1));
      expr = "(isnan((float)(" + a0 + ")) ? (" + a0 + ") : (isnan((float)(" +
             a1 + ")) ? (" + a1 + ") : fminf(" + a0 + ", " + a1 + ")))";
    } else if (k == "max_num")
      expr = "fmaxf(" + a0 + ", " + c.name(op.getOperand(1)) + ")";
    else if (k == "maximum_nan") {
      auto a1 = c.name(op.getOperand(1));
      expr = "(isnan((float)(" + a0 + ")) || isnan((float)(" + a1 +
             ")) ? NAN : fmaxf(" + a0 + ", " + a1 + "))";
    } else
      expr = "(" + t + ")(" + a0 + ")";
    c.line(t + " " + res + " = " + expr + ";");
    return llvm::Error::success();
  });

  enter("bangc_src.row_reduce", [](Operation &op, PrintContext &c) {
    auto alg = op.getAttrOfType<StringAttr>("algebra").getValue();
    std::string fnName = alg == "sum" ? "group_sum" : "group_max_num";
    c.line(fnName + "(" + c.name(op.getOperand(0)) + ", " +
           c.name(op.getOperand(1)) + ", " + c.name(op.getOperand(2)) +
           ", (int)(" + c.name(op.getOperand(3)) + "), (int)(" +
           c.name(op.getOperand(4)) + "), (int)(" + c.name(op.getOperand(5)) +
           "));");
    return llvm::Error::success();
  });

  enter("bangc_src.row_exp", [](Operation &op, PrintContext &c) {
    c.line("row_exp(" + c.name(op.getOperand(0)) + ", " +
           c.name(op.getOperand(1)) + ", " + c.name(op.getOperand(2)) +
           ", (int)(" + c.name(op.getOperand(3)) + "), (int)(" +
           c.name(op.getOperand(4)) + "));");
    return llvm::Error::success();
  });

  enter("bangc_src.tile_transpose", [](Operation &op, PrintContext &c) {
    auto ty = cast<MemRefType>(op.getOperand(0).getType());
    unsigned bits = ty.getElementTypeBitWidth();
    std::string fnName = bits == 8    ? "transpose_tile_b8"
                         : bits == 16 ? "transpose_tile_b16"
                                      : "transpose_tile_b32";
    c.line(fnName + "(" + c.name(op.getOperand(0)) + ", " +
           c.name(op.getOperand(1)) + ", (int)(" + c.name(op.getOperand(2)) +
           "), (int)(" + c.name(op.getOperand(3)) + "));");
    return llvm::Error::success();
  });

  enter("bangc_src.wram_conv", [](Operation &op, PrintContext &c) {
    auto mode = op.getAttrOfType<StringAttr>("mode").getValue();
    auto bm = std::to_string(op.getAttrOfType<IntegerAttr>("bm").getInt());
    auto bn = std::to_string(op.getAttrOfType<IntegerAttr>("bn").getInt());
    auto bk = std::to_string(op.getAttrOfType<IntegerAttr>("bk").getInt());
    if (mode == "i8") {
      auto acc = c.name(op.getOperand(0)), part = c.name(op.getOperand(1)),
           lhs = c.name(op.getOperand(2)), flt = c.name(op.getOperand(3));
      c.line("__bang_conv(" + part + ", (signed char *)" + lhs +
             ", (signed char *)" + flt + ", " + bk + ", " + bm +
             ", 1, 1, 1, 1, 1, " + bn + ", 0);");
      c.line("__bang_float2int32_rn((int *)" + part + ", " + part + ", " + bm +
             " * " + bn + ", 0);");
      c.line("__bang_add(" + acc + ", " + acc + ", (int *)" + part + ", " + bm +
             " * " + bn + ");");
      return llvm::Error::success();
    }
    auto acc = c.name(op.getOperand(0)), lhs = c.name(op.getOperand(1)),
         flt = c.name(op.getOperand(2)), accum = c.name(op.getOperand(3));
    std::string ptrTy = mode == "bf16" ? "bfloat16_t" : "half";
    c.line("if (" + accum + ") __bang_conv_partial(" + acc + ", (" + ptrTy +
           " *)" + lhs + ", (" + ptrTy + " *)" + flt + ", " + acc + ", " + bk +
           ", " + bm + ", 1, 1, 1, 1, 1, " + bn +
           "); else __bang_conv(" + acc + ", (" + ptrTy + " *)" + lhs + ", (" +
           ptrTy + " *)" + flt + ", " + bk + ", " + bm + ", 1, 1, 1, 1, 1, " +
           bn + ");");
    return llvm::Error::success();
  });

  enter("bangc_src.cycle_mul", [](Operation &op, PrintContext &c) {
    c.line("__bang_cycle_mul(" + c.name(op.getOperand(0)) + ", " +
           c.name(op.getOperand(1)) + ", " + c.name(op.getOperand(2)) +
           ", (int)(" + c.name(op.getOperand(3)) + "), (int)(" +
           c.name(op.getOperand(4)) + "));");
    return llvm::Error::success();
  });

  enter("bangc_src.vec_min_scalar", [](Operation &op, PrintContext &c) {
    c.line("vec_min_scalar_ieee(" + c.name(op.getOperand(0)) + ", " +
           c.name(op.getOperand(1)) + ", " + c.name(op.getOperand(2)) + ", " +
           c.name(op.getOperand(3)) + ", (int)(" + c.name(op.getOperand(4)) +
           "), (int)(" + c.name(op.getOperand(5)) + "));");
    return llvm::Error::success();
  });

  enter("bangc_src.bf16_panel_ok", [](Operation &op, PrintContext &c) {
    c.line("unsigned char " + c.name(op.getResult(0)) +
           " = ordinary_panel((const unsigned short *)" +
           c.name(op.getOperand(0)) + ", (int)(" + c.name(op.getOperand(3)) +
           "), (unsigned short *)" + c.name(op.getOperand(1)) +
           ", (unsigned short *)" + c.name(op.getOperand(2)) + ");");
    return llvm::Error::success();
  });

  enter("bangc_src.bf16_ieee_dot", [](Operation &op, PrintContext &c) {
    c.line("bf16_ieee_dot_panel(" + c.name(op.getOperand(0)) +
           ", (const unsigned short *)" + c.name(op.getOperand(1)) +
           ", (const unsigned short *)" + c.name(op.getOperand(2)) +
           ", (int)(" + c.name(op.getOperand(3)) + "), (int)(" +
           c.name(op.getOperand(4)) + "), (int)(" + c.name(op.getOperand(5)) +
           "), (int)(" + c.name(op.getOperand(6)) + "), (int)(" +
           c.name(op.getOperand(7)) + "));");
    return llvm::Error::success();
  });

  enter("bangc_src.bf16_acc_narrow", [](Operation &op, PrintContext &c) {
    c.line("bf16_acc_narrow((unsigned short *)" + c.name(op.getOperand(0)) +
           ", " + c.name(op.getOperand(1)) + ", (int)(" +
           c.name(op.getOperand(2)) + "), " + c.name(op.getOperand(3)) + ");");
    return llvm::Error::success();
  });

  enter("bangc_src.i32_to_bf16", [](Operation &op, PrintContext &c) {
    c.line("unsigned short " + c.name(op.getResult(0)) + " = signed_to_bf16(" +
           c.name(op.getOperand(0)) + ");");
    return llvm::Error::success();
  });

  enter("bangc_src.from_le_half", [](Operation &op, PrintContext &c) {
    auto res = c.name(op.getResult(0));
    auto lo = c.name(op.getOperand(0)), hi = c.name(op.getOperand(1));
    c.line("float " + res +
           "; { union { unsigned short u; half h; } _h; _h.u = (unsigned "
           "short)(((unsigned)(" +
           lo + ") & 0xffu) | (((unsigned)(" + hi + ") & 0xffu) << 8)); " +
           res + " = (float)_h.h; }");
    return llvm::Error::success();
  });

  enter("bangc_src.narrow_quotient", [](Operation &op, PrintContext &c) {
    c.line("group_narrow_quotient(" + c.name(op.getOperand(0)) + ", " +
           c.name(op.getOperand(1)) + ", " + c.name(op.getOperand(2)) + ", " +
           c.name(op.getOperand(3)) + ", " + c.name(op.getOperand(4)) +
           ", (int)(" + c.name(op.getOperand(5)) + "), (int)(" +
           c.name(op.getOperand(6)) + "), (int)(" + c.name(op.getOperand(7)) +
           "), (int)(" + c.name(op.getOperand(8)) + "));");
    return llvm::Error::success();
  });

  enter("bangc_src.bf16_tile", [](Operation &op, PrintContext &c) {
    auto mode = op.getAttrOfType<StringAttr>("mode").getValue();
    if (mode == "widen")
      c.line("widen_bf16_tile(" + c.name(op.getOperand(0)) + ", " +
             c.name(op.getOperand(1)) + ", (int)(" + c.name(op.getOperand(3)) +
             "));");
    else
      c.line("narrow_bf16_tile(" + c.name(op.getOperand(0)) + ", " +
             c.name(op.getOperand(1)) + ", (int)(" + c.name(op.getOperand(2)) +
             "), (int)(" + c.name(op.getOperand(3)) + "));");
    return llvm::Error::success();
  });

  enter("bangc_src.bf16_product", [](Operation &op, PrintContext &c) {
    c.line("bf16_product_tile(" + c.name(op.getOperand(0)) + ", " +
           c.name(op.getOperand(1)) + ", " + c.name(op.getOperand(2)) + ", " +
           c.name(op.getOperand(3)) + ", (int)(" + c.name(op.getOperand(4)) +
           "), (int)(" + c.name(op.getOperand(5)) + "));");
    return llvm::Error::success();
  });

  enter("bangc_src.vec_sqrt_rn", [](Operation &op, PrintContext &c) {
    c.line("vec_sqrt_rn(" + c.name(op.getOperand(0)) + ", " +
           c.name(op.getOperand(1)) + ", (int)(" + c.name(op.getOperand(2)) +
           "));");
    return llvm::Error::success();
  });

  enter("bangc_src.vec_div_rn", [](Operation &op, PrintContext &c) {
    auto mode = op.getAttrOfType<StringAttr>("mode").getValue();
    std::string fnName = mode == "vv"   ? "vec_div_rn"
                         : mode == "vs" ? "vec_div_scalar_rn"
                                        : "scalar_div_vec_rn";
    c.line(fnName + "(" + c.name(op.getOperand(0)) + ", " +
           c.name(op.getOperand(1)) + ", " + c.name(op.getOperand(2)) + ", " +
           c.name(op.getOperand(3)) + ", (int)(" + c.name(op.getOperand(4)) +
           "), (int)(" + c.name(op.getOperand(5)) + "));");
    return llvm::Error::success();
  });

  enter("bangc_src.bf16_scalar", [](Operation &op, PrintContext &c) {
    auto mode = op.getAttrOfType<StringAttr>("mode").getValue();
    auto res = c.name(op.getResult(0));
    auto t = ctype(op.getResult(0).getType());
    auto a0 = c.name(op.getOperand(0));
    std::string expr =
        mode == "widen"    ? "widen_bf16(" + a0 + ")"
        : mode == "narrow" ? "round_bf16(" + a0 + ")"
                           : "round_bf16(widen_bf16(" + a0 +
                                 ") * widen_bf16(" + c.name(op.getOperand(1)) +
                                 "))";
    c.line(t + " " + res + " = " + expr + ";");
    return llvm::Error::success();
  });

  enter("bangc_src.scan_sum", [](Operation &op, PrintContext &c) {
    c.line("tile_prefix(" + c.name(op.getOperand(0)) + ", " +
           c.name(op.getOperand(1)) + ", " + c.name(op.getOperand(2)) +
           ", (int)(" + c.name(op.getOperand(3)) + "), (int)(" +
           c.name(op.getOperand(4)) + "));");
    return llvm::Error::success();
  });

  enter("bangc_src.philox_uniform", [](Operation &op, PrintContext &c) {
    c.line("philox4x32_linear(" + c.name(op.getOperand(0)) +
           ", (unsigned *)" + c.name(op.getOperand(1)) +
           ", (unsigned long long)(" + c.name(op.getOperand(2)) +
           "), (unsigned long long)(" + c.name(op.getOperand(3)) +
           "), (int)(" + c.name(op.getOperand(4)) + "));");
    return llvm::Error::success();
  });

  enter("bangc_src.philox_at", [](Operation &op, PrintContext &c) {
    c.line("float " + c.name(op.getResult(0)) +
           " = uniform_at((unsigned long long)(" + c.name(op.getOperand(0)) +
           "), (unsigned long long)(" + c.name(op.getOperand(1)) + "));");
    return llvm::Error::success();
  });

  // Standard SCF structured control-flow rules.
  rules["scf.for"] = [](Operation &op, PrintPoint pt, PrintContext &c) {
    auto loop = cast<scf::ForOp>(op);
    if (pt == PrintPoint::Enter) {
      for (auto [res, init] : llvm::zip(loop.getResults(), loop.getInitArgs()))
        c.line(ctype(res.getType()) + " " + c.name(res) + " = " +
               c.name(init) + ";");
      auto iv = c.name(loop.getInductionVar());
      c.line("for (long long " + iv + " = " + c.name(loop.getLowerBound()) +
             "; " + iv + " < " + c.name(loop.getUpperBound()) + "; " + iv +
             " += " + c.name(loop.getStep()) + ") {");
      ++c.indent;
      for (auto [arg, res] :
           llvm::zip(loop.getRegionIterArgs(), loop.getResults()))
        c.line(ctype(arg.getType()) + " " + c.name(arg) + " = " + c.name(res) +
               ";");
    } else if (pt == PrintPoint::Leave) {
      --c.indent;
      c.line("}");
    }
    return llvm::Error::success();
  };

  rules["scf.if"] = [](Operation &op, PrintPoint pt, PrintContext &c) {
    if (pt == PrintPoint::Enter) {
      for (Value res : op.getResults())
        c.line(ctype(res.getType()) + " " + c.name(res) + ";");
      c.line("if (" + c.name(op.getOperand(0)) + ") {");
    } else if (pt == PrintPoint::BeginRegion) {
      if (c.region == 0) {
        ++c.indent;
      } else if (!op.getRegion(1).empty()) {
        c.line("else {");
        ++c.indent;
      }
    } else if (pt == PrintPoint::EndRegion) {
      if (c.region == 0 || !op.getRegion(1).empty()) {
        --c.indent;
        c.line("}");
      }
    }
    return llvm::Error::success();
  };

  rules["scf.while"] = [](Operation &op, PrintPoint pt, PrintContext &c) {
    auto w = cast<scf::WhileOp>(op);
    if (pt == PrintPoint::Enter) {
      for (auto [res, init] : llvm::zip(w.getResults(), w.getInits()))
        c.line(ctype(res.getType()) + " " + c.name(res) + " = " +
               c.name(init) + ";");
      c.line("while (true) {");
    } else if (pt == PrintPoint::BeginRegion) {
      ++c.indent;
      Block &blk = op.getRegion(c.region).front();
      for (auto [arg, res] : llvm::zip(blk.getArguments(), w.getResults()))
        c.line(ctype(arg.getType()) + " " + c.name(arg) + " = " + c.name(res) +
               ";");
    } else if (pt == PrintPoint::EndRegion) {
      --c.indent;
    } else if (pt == PrintPoint::Leave) {
      c.line("}");
    }
    return llvm::Error::success();
  };

  enter("scf.condition", [](Operation &op, PrintContext &c) {
    auto *parent = op.getParentOp();
    auto forwarded = op.getOperands().drop_front();
    for (auto [i, v] : llvm::enumerate(forwarded))
      c.line("auto _cond_" + std::to_string(i) + " = " + c.name(v) + ";");
    for (auto [i, res] : llvm::enumerate(parent->getResults()))
      c.line(c.name(res) + " = _cond_" + std::to_string(i) + ";");
    c.line("if (!(" + c.name(op.getOperand(0)) + ")) break;");
    return llvm::Error::success();
  });

  enter("scf.yield", [](Operation &op, PrintContext &c) {
    if (op.getNumOperands() == 0)
      return llvm::Error::success();
    auto *parent = op.getParentOp();
    for (auto [i, v] : llvm::enumerate(op.getOperands()))
      c.line("auto _yld_" + std::to_string(i) + " = " + c.name(v) + ";");
    for (auto [i, res] : llvm::enumerate(parent->getResults()))
      c.line(c.name(res) + " = _yld_" + std::to_string(i) + ";");
    return llvm::Error::success();
  });

  // Standard Arith scalar rules.
  enter("arith.constant", [](Operation &op, PrintContext &c) {
    auto v = op.getAttr("value");
    auto resTy = op.getResult(0).getType();
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
      } else if (resTy.isInteger(8)) {
        lit = std::to_string(ia.getInt());
      } else if (resTy.isInteger(32)) {
        int64_t val = ia.getInt();
        lit = val == INT32_MIN ? "(-2147483647 - 1)" : std::to_string(val);
      } else {
        int64_t val = ia.getInt();
        lit = val == LLONG_MIN ? "(-9223372036854775807LL - 1LL)"
                               : std::to_string(val) + "LL";
      }
    } else {
      auto fa = cast<FloatAttr>(v);
      if (resTy.isBF16() || isa<Float8E4M3FNType>(resTy)) {
        lit = std::to_string(fa.getValue().bitcastToAPInt().getZExtValue()) +
              "U";
      } else if (fa.getValue().isInfinity()) {
        lit = fa.getValue().isNegative() ? "-INFINITY" : "INFINITY";
      } else if (fa.getValue().isNaN()) {
        lit = "NAN";
      } else {
        lit = intentdsl::kir::literal(fa) + "f";
      }
    }
    c.line(t + " " + c.name(op.getResult(0)) + " = (" + t + ")(" + lit + ");");
    return llvm::Error::success();
  });

  for (auto [opName, tok] : {std::pair{"arith.addi", "+"},
                             {"arith.subi", "-"},
                             {"arith.muli", "*"}}) {
    enter(opName, [tok](Operation &op, PrintContext &c) {
      auto resTy = op.getResult(0).getType();
      auto t = ctype(resTy);
      auto a = c.name(op.getOperand(0)), b = c.name(op.getOperand(1));
      std::string expr = resTy.isInteger(32)
                             ? "(int)((unsigned)(" + a + ") " + tok +
                                   " (unsigned)(" + b + "))"
                             : "(" + a + ") " + tok + " (" + b + ")";
      c.line(t + " " + c.name(op.getResult(0)) + " = " + expr + ";");
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
    enter(opName, [tok](Operation &op, PrintContext &c) {
      auto t = ctype(op.getResult(0).getType());
      c.line(t + " " + c.name(op.getResult(0)) + " = (" + t + ")((" +
             c.name(op.getOperand(0)) + ") " + tok + " (" +
             c.name(op.getOperand(1)) + "));");
      return llvm::Error::success();
    });
  }

  enter("arith.shli", [](Operation &op, PrintContext &c) {
    auto resTy = op.getResult(0).getType();
    auto t = ctype(resTy);
    std::string u = resTy.isInteger(32) ? "unsigned" : "unsigned long long";
    c.line(t + " " + c.name(op.getResult(0)) + " = (" + t + ")((" + u + ")(" +
           c.name(op.getOperand(0)) + ") << (" + c.name(op.getOperand(1)) +
           "));");
    return llvm::Error::success();
  });

  enter("arith.shrsi", [](Operation &op, PrintContext &c) {
    auto t = ctype(op.getResult(0).getType());
    c.line(t + " " + c.name(op.getResult(0)) + " = (" + c.name(op.getOperand(0)) +
           ") >> (" + c.name(op.getOperand(1)) + ");");
    return llvm::Error::success();
  });

  enter("arith.shrui", [](Operation &op, PrintContext &c) {
    auto resTy = op.getResult(0).getType();
    auto t = ctype(resTy);
    std::string u = resTy.isInteger(32) ? "unsigned" : "unsigned long long";
    c.line(t + " " + c.name(op.getResult(0)) + " = (" + t + ")((" + u + ")(" +
           c.name(op.getOperand(0)) + ") >> (" + c.name(op.getOperand(1)) +
           "));");
    return llvm::Error::success();
  });

  enter("arith.bitcast", [](Operation &op, PrintContext &c) {
    auto fromT = ctype(op.getOperand(0).getType());
    auto toT = ctype(op.getResult(0).getType());
    auto res = c.name(op.getResult(0));
    c.line(toT + " " + res + "; { union { " + fromT + " a; " + toT +
           " b; } _bc; _bc.a = " + c.name(op.getOperand(0)) + "; " + res +
           " = _bc.b; }");
    return llvm::Error::success();
  });

  enter("arith.minsi", [](Operation &op, PrintContext &c) {
    auto t = ctype(op.getResult(0).getType());
    auto a = c.name(op.getOperand(0)), b = c.name(op.getOperand(1));
    c.line(t + " " + c.name(op.getResult(0)) + " = (" + a + " < " + b + " ? " +
           a + " : " + b + ");");
    return llvm::Error::success();
  });

  enter("arith.maxsi", [](Operation &op, PrintContext &c) {
    auto t = ctype(op.getResult(0).getType());
    auto a = c.name(op.getOperand(0)), b = c.name(op.getOperand(1));
    c.line(t + " " + c.name(op.getResult(0)) + " = (" + a + " > " + b + " ? " +
           a + " : " + b + ");");
    return llvm::Error::success();
  });

  enter("arith.cmpi", [](Operation &op, PrintContext &c) {
    auto pred = cast<arith::CmpIOp>(op).getPredicate();
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
    c.line("unsigned char " + c.name(op.getResult(0)) + " = ((" +
           c.name(op.getOperand(0)) + ") " + tok + " (" +
           c.name(op.getOperand(1)) + "));");
    return llvm::Error::success();
  });

  enter("arith.cmpf", [](Operation &op, PrintContext &c) {
    auto pred = cast<arith::CmpFOp>(op).getPredicate();
    auto a = "((float)(" + c.name(op.getOperand(0)) + "))";
    auto b = "((float)(" + c.name(op.getOperand(1)) + "))";
    std::string expr;
    switch (pred) {
    case arith::CmpFPredicate::OEQ:
      expr = "(" + a + " == " + b + ")";
      break;
    case arith::CmpFPredicate::ONE:
      expr = "(!isnan(" + a + ") && !isnan(" + b + ") && " + a + " != " + b +
             ")";
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
    case arith::CmpFPredicate::UNE:
      expr = "(" + a + " != " + b + ")";
      break;
    case arith::CmpFPredicate::UNO:
      expr = "(isnan(" + a + ") || isnan(" + b + "))";
      break;
    default:
      expr = "(" + a + " > " + b + ")";
      break;
    }
    c.line("unsigned char " + c.name(op.getResult(0)) + " = " + expr + ";");
    return llvm::Error::success();
  });

  enter("arith.select", [](Operation &op, PrintContext &c) {
    auto t = ctype(op.getResult(0).getType());
    c.line(t + " " + c.name(op.getResult(0)) + " = (" +
           c.name(op.getOperand(0)) + " ? " + c.name(op.getOperand(1)) + " : " +
           c.name(op.getOperand(2)) + ");");
    return llvm::Error::success();
  });

  for (auto castOp :
       {"arith.extsi", "arith.extui", "arith.trunci", "arith.index_cast",
        "arith.sitofp", "arith.uitofp", "arith.fptosi", "arith.extf",
        "arith.truncf"}) {
    enter(castOp, [castOp](Operation &op, PrintContext &c) {
      auto t = ctype(op.getResult(0).getType());
      std::string src = c.name(op.getOperand(0));
      if (llvm::StringRef(castOp) == "arith.extui" ||
          llvm::StringRef(castOp) == "arith.uitofp") {
        auto inTy = op.getOperand(0).getType();
        if (inTy.isInteger(1) || inTy.isInteger(8))
          src = "(unsigned char)(" + src + ")";
        else if (inTy.isInteger(32))
          src = "(unsigned int)(" + src + ")";
        else
          src = "(unsigned long long)(" + src + ")";
      }
      c.line(t + " " + c.name(op.getResult(0)) + " = (" + t + ")(" + src + ");");
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
