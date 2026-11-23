#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/Storage.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "llvm/ADT/StringExtras.h"

namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
using namespace mlir;

TerminalPrinter mojoMatrixPrinter() {
  TerminalPrinter p;
  auto noop = [](Operation &, PrintPoint, PrintContext &) {
    return llvm::Error::success();
  };
  p.operations["builtin.module"] = noop;
  p.operations["func.func"] = noop;
  p.operations["func.return"] = noop;
  p.operations["mojo_row.end"] = noop;

  p.operations["mojo_row.matrix_program"] = [](Operation &, PrintPoint at,
                                               PrintContext &c) {
    if (at == PrintPoint::BeginRegion) {
      c.file = c.region == 0 ? "row.mojo" : "entry.c";
      c.indent = 0;
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
    int64_t val = op.getAttrOfType<IntegerAttr>("value").getInt();
    c.line("var " + c.name(op.getResult(0)) + " = Int(" + std::to_string(val) +
           ")");
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
  binIdx("arith.remsi", "%", "%");
  binIdx("arith.addf", "+", "+");
  binIdx("arith.subf", "-", "-");
  binIdx("arith.mulf", "*", "*");

  p.operations["arith.minsi"] = [](Operation &op, PrintPoint at,
                                   PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("var " + c.name(op.getResult(0)) + " = min(" +
             c.name(op.getOperand(0)) + ", " + c.name(op.getOperand(1)) + ")");
    return llvm::Error::success();
  };

  // Scalar panel map ops in entry.c
  p.operations["arith.extf"] = [](Operation &op, PrintPoint at,
                                  PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("const float " + c.name(op.getResult(0)) + " = expand_bf(" +
             c.name(op.getOperand(0)) + ");");
    return llvm::Error::success();
  };
  p.operations["arith.truncf"] = [](Operation &op, PrintPoint at,
                                    PrintContext &c) {
    if (at == PrintPoint::Enter)
      c.line("const uint16_t " + c.name(op.getResult(0)) + " = narrow_bf(" +
             c.name(op.getOperand(0)) + ");");
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
    if (at == PrintPoint::Enter)
      c.line("const float " + c.name(op.getResult(0)) + " = expf(" +
             c.name(op.getOperand(0)) + ");");
    return llvm::Error::success();
  };
  p.operations["mojo_row.host_panel_yield"] = [](Operation &op, PrintPoint at,
                                                 PrintContext &c) {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    auto *prep = op.getParentOp()->getParentOp();
    bool wide = prep->getAttrOfType<StringAttr>("kind").getValue() ==
                "map-window-rhs-f32";
    std::string dst = prep->getAttrOfType<StringAttr>("dst_buf").str();
    std::string v = c.name(op.getOperand(0));
    c.line(dst + "[p*N+n] = " + (wide ? "expand_bf(" + v + ")" : v) + ";");
    return llvm::Error::success();
  };
  p.operations["mojo_row.host_col_loop"] = [](Operation &op, PrintPoint at,
                                              PrintContext &c) {
    auto *prep = op.getParentOp();
    auto slots = prep->getAttrOfType<DenseI64ArrayAttr>("src_slots");
    auto st = prep->getAttrOfType<DenseI64ArrayAttr>("src_strides_0");
    if (at == PrintPoint::Enter) {
      auto &varBb = op.getRegion(0).front();
      c.line("#pragma omp simd");
      c.line("for (int64_t n=0; n<N; ++n) {");
      c.indent++;
      c.line("const uint16_t " + c.name(varBb.getArgument(0)) +
             " = ((const uint16_t*)a" + std::to_string(slots[0]) + ")[batch*" +
             std::to_string(st[0]) + "LL+(chunk*K+k)*" +
             std::to_string(st[1]) + "LL+group*" + std::to_string(st[2]) +
             "LL+n*" + std::to_string(st[3]) + "LL];");
    } else if (at == PrintPoint::EndBlock) {
      c.indent--;
      c.line("}");
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
      c.line("def intent_panel(" + llvm::join(params, ", ") + ") abi(\"C\"):");
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
      c.line("return");
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
    if (at == PrintPoint::Enter) {
      c.line("var " + c.name(op.getResult(0)) +
             " = external_call[\"intent_brgemm_tile\", Int32](" +
             c.name(op.getOperand(0)) + ", " + c.name(op.getOperand(1)) + ", " +
             c.name(op.getOperand(2)) + ", " + c.name(op.getOperand(3)) + ", " +
             c.name(op.getOperand(4)) + ")");
      c.line("_ = " + c.name(op.getResult(0)));
    }
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
      c.line("extern void intent_panel(" + llvm::join(ptypes, ",") + ");");
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
        c.line("  if (n <= 0) return; int64_t i=0;");
        c.line("#if defined(__AVX512F__)");
        c.line("  for (; i <= n - 16; i += 16) { __m256i h = _mm256_loadu_si256((const __m256i*)(src+i)); _mm512_storeu_ps(out+i, _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16))); }");
        c.line("#endif");
        c.line("  for (; i<n; ++i) out[i] = expand_bf(src[i]);");
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
        c.line("  if (n <= 0) return; int64_t i=0;");
        c.line("#if defined(__AVX512F__) && defined(__F16C__)");
        c.line("  for (; i <= n - 16; i += 16) { __m256i h = _mm256_loadu_si256((const __m256i*)(src+i)); _mm512_storeu_ps(out+i, _mm512_cvtph_ps(h)); }");
        c.line("#endif");
        c.line("  for (; i<n; ++i) out[i] = expand_f16(src[i]);");
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

  p.operations["mojo_row.host_prep"] = [](Operation &op, PrintPoint at,
                                          PrintContext &c) -> llvm::Error {
    auto pk = op.getAttrOfType<StringAttr>("kind").getValue();
    bool isMap = pk == "map-window-rhs-f32" || pk == "map-window-rhs-bf16";
    if (isMap) {
      auto slots = op.getAttrOfType<DenseI64ArrayAttr>("src_slots");
      auto access = [&](int idx, StringRef cTy, ArrayRef<std::string> coords) {
        auto st = op.getAttrOfType<DenseI64ArrayAttr>("src_strides_" +
                                                      std::to_string(idx));
        std::string s =
            "((" + cTy.str() + "*)a" + std::to_string(slots[idx]) + ")[0";
        for (auto [ax, e] : llvm::enumerate(coords))
          s += "+(" + e + ")*" + std::to_string(st[ax]) + "LL";
        return s + "]";
      };
      if (at == PrintPoint::Enter) {
        int64_t H = op.getAttrOfType<IntegerAttr>("head_extent").getInt();
        int64_t chunks = op.getAttrOfType<IntegerAttr>("chunk_extent").getInt();
        int64_t hg = op.getAttrOfType<IntegerAttr>("head_group").getInt();
        c.line("#pragma omp parallel for num_threads(workers) schedule(static) if(G*K>1)");
        c.line("for (int64_t p=0; p<G*K; ++p) {");
        c.indent++;
        c.line("const int64_t g=p/K, k=p%K, head=g%" + std::to_string(H) +
               ", chunk=g/" + std::to_string(H) + "%" + std::to_string(chunks) +
               ", batch=g/" + std::to_string(H) + "/" + std::to_string(chunks) +
               ", group=head/" + std::to_string(hg) + ";");
        auto &invBb = op.getRegion(0).front();
        c.line("const float " + c.name(invBb.getArgument(0)) + " = " +
               access(1, "float", {"batch", "head", "chunk", "k"}) + ";");
        c.line("const float " + c.name(invBb.getArgument(1)) + " = " +
               access(2, "float", {"batch", "head", "chunk", "k"}) + ";");
        c.line("const float " + c.name(invBb.getArgument(2)) + " = " +
               access(2, "float", {"batch", "head", "chunk", "K-1"}) + ";");
      } else if (at == PrintPoint::EndBlock) {
        c.indent--;
        c.line("}");
      }
      return llvm::Error::success();
    }

    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    auto slots = op.getAttrOfType<DenseI64ArrayAttr>("src_slots");
    auto dst = op.getAttrOfType<StringAttr>("dst_buf").str();

    if (pk == "widen-window-lhs") {
      int64_t H = op.getAttrOfType<IntegerAttr>("head_extent").getInt();
      int64_t chunks = op.getAttrOfType<IntegerAttr>("chunk_extent").getInt();
      auto st = op.getAttrOfType<DenseI64ArrayAttr>("src_strides_0");
      c.line("#pragma omp parallel for num_threads(workers) schedule(static) if(G>1)");
      c.line("for (int64_t g=0; g<G; ++g) {");
      c.line("  const int64_t batch=g/" + std::to_string(H) + "/" +
             std::to_string(chunks) + ", chunk=g/" + std::to_string(H) + "%" +
             std::to_string(chunks) + ", head=g%" + std::to_string(H) + ";");
      c.line("  for (int64_t m=0; m<M; ++m) for (int64_t k=0; k<K; ++k)");
      c.line("    " + dst + "[(g*M+m)*K+k] = expand_bf(((const uint16_t*)a" +
             std::to_string(slots[0]) + ")[batch*" + std::to_string(st[0]) +
             "LL+(chunk*K+k)*" + std::to_string(st[1]) + "LL+head*" +
             std::to_string(st[2]) + "LL+m*" + std::to_string(st[3]) + "LL]);");
      c.line("}");
    } else if (pk == "widen-lhs-f32") {
      auto st = op.getAttrOfType<DenseI64ArrayAttr>("src_strides_0");
      auto dt = op.getAttrOfType<StringAttr>("src_dtype_0").getValue();
      int64_t ak = op.getAttrOfType<IntegerAttr>("lhs_k").getInt();
      int64_t ab = op.getAttrOfType<IntegerAttr>("lhs_batch").getInt();
      int64_t am = 0;
      while (am == ak || am == ab)
        ++am;
      std::string banks = ab < 0 ? "1LL" : "G";
      std::string fnWiden = dt == "bf16" ? "expand_bf" : "expand_f16";
      std::string fnSpan = dt == "bf16" ? "widen_bf16_span" : "widen_f16_span";
      c.line("#pragma omp parallel for num_threads(workers) schedule(static) if((" +
             banks + ")*M*K > 131072)");
      c.line("for (int64_t row=0; row<(" + banks + ")*M; ++row) {");
      c.line("  const int64_t g = M ? row / M : 0, m = M ? row % M : 0;");
      c.line("  const uint16_t *src = (const uint16_t*)a" +
             std::to_string(slots[0]) + " + g*" +
             std::to_string(ab < 0 ? 0 : st[ab]) + "LL + m*" +
             std::to_string(st[am]) + "LL;");
      c.line("  float *out = " + dst + " + row*K;");
      if (st[ak] == 1) {
        c.line("  " + fnSpan + "(out, src, K);");
      } else {
        c.line("  for (int64_t k=0; k<K; ++k) out[k] = " + fnWiden + "(src[k*" +
               std::to_string(st[ak]) + "LL]);");
      }
      c.line("}");
    } else if (pk == "widen-pack-rhs-f32") {
      auto st = op.getAttrOfType<DenseI64ArrayAttr>("src_strides_0");
      auto dt = op.getAttrOfType<StringAttr>("src_dtype_0").getValue();
      int64_t bk = op.getAttrOfType<IntegerAttr>("rhs_k").getInt();
      int64_t bb = op.getAttrOfType<IntegerAttr>("rhs_batch").getInt();
      int64_t bn = 0;
      while (bn == bk || bn == bb)
        ++bn;
      std::string banks = bb < 0 ? "1LL" : "G";
      std::string fnWiden = dt == "bf16" ? "expand_bf" : "expand_f16";
      std::string fnSpan = dt == "bf16" ? "widen_bf16_span" : "widen_f16_span";
      c.line("#pragma omp parallel for num_threads(workers) schedule(static) if((" +
             banks + ")*nc*K*nr > 131072 && (" + banks + ")*nc >= 4)");
      c.line("for (int64_t p=0; p<(" + banks + ")*nc; ++p) {");
      c.line("  const int64_t g = nc ? p / nc : 0, c = (nc ? p % nc : 0)*nr, valid = N-c < nr ? N-c : nr;");
      c.line("  for (int64_t k=0; k<K; ++k) {");
      c.line("    const uint16_t *src = (const uint16_t*)a" +
             std::to_string(slots[0]) + " + g*" +
             std::to_string(bb < 0 ? 0 : st[bb]) + "LL + k*" +
             std::to_string(st[bk]) + "LL + c*" + std::to_string(st[bn]) +
             "LL;");
      c.line("    float *out = " + dst + " + (p*K + k)*nr;");
      if (st[bn] == 1) {
        c.line("    " + fnSpan + "(out, src, valid);");
      } else {
        c.line("    for (int64_t j=0; j<valid; ++j) out[j] = " + fnWiden +
               "(src[j*" + std::to_string(st[bn]) + "LL]);");
      }
      c.line("    for (int64_t j=valid; j<nr; ++j) out[j] = 0.0f;");
      c.line("  }");
      c.line("}");
    } else if (pk == "pack-rhs-panels") {
      auto st = op.getAttrOfType<DenseI64ArrayAttr>("src_strides_0");
      auto dt = op.getAttrOfType<StringAttr>("src_dtype_0").getValue();
      int64_t bytes = dt == "f32" || dt == "i32" ? 4 : dt == "i8" || dt == "u8" ? 1 : 2;
      int64_t bk = op.getAttrOfType<IntegerAttr>("rhs_k").getInt();
      int64_t bb = op.getAttrOfType<IntegerAttr>("rhs_batch").getInt();
      int64_t bn = 0;
      while (bn == bk || bn == bb)
        ++bn;
      std::string banks = bb < 0 ? "1LL" : "G";
      c.line("#pragma omp parallel for num_threads(workers) schedule(static) if((" +
             banks + ")*nc*K*nr > 131072 && (" + banks + ")*nc >= 4)");
      c.line("for (int64_t p=0; p<(" + banks + ")*nc; ++p) {");
      c.line("  const int64_t g = nc ? p / nc : 0, c = (nc ? p % nc : 0)*nr, valid = N-c < nr ? N-c : nr;");
      c.line("  for (int64_t k=0; k<K; ++k) {");
      c.line("    unsigned char *out = (unsigned char*)" + dst + " + (p*K + k)*nr*" +
             std::to_string(bytes) + "LL;");
      c.line("    const unsigned char *src = (const unsigned char*)a" +
             std::to_string(slots[0]) + " + (g*" +
             std::to_string(bb < 0 ? 0 : st[bb]) + "LL + k*" +
             std::to_string(st[bk]) + "LL + c*" + std::to_string(st[bn]) +
             "LL)*" + std::to_string(bytes) + "LL;");
      if (st[bn] == 1)
        c.line("    memcpy(out, src, valid*" + std::to_string(bytes) + "LL);");
      else
        c.line("    for (int64_t j=0; j<valid; ++j) memcpy(out+j*" +
               std::to_string(bytes) + "LL, src+j*" +
               std::to_string(st[bn] * bytes) + "LL, " +
               std::to_string(bytes) + ");");
      c.line("    if (valid < nr) memset(out + valid*" + std::to_string(bytes) +
             "LL, 0, (nr - valid)*" + std::to_string(bytes) + "LL);");
      c.line("  }");
      c.line("}");
    } else if (pk == "pack-signed-dot4") {
      auto st = op.getAttrOfType<DenseI64ArrayAttr>("src_strides_0");
      int64_t bk = op.getAttrOfType<IntegerAttr>("rhs_k").getInt();
      int64_t bb = op.getAttrOfType<IntegerAttr>("rhs_batch").getInt();
      int64_t bn = 0;
      while (bn == bk || bn == bb)
        ++bn;
      std::string banks = bb < 0 ? "1LL" : "G";
      c.line("#pragma omp parallel for num_threads(workers) schedule(static) if((" +
             banks + ")*nc*K*nr > 131072 && (" + banks + ")*nc >= 4)");
      c.line("for (int64_t p=0; p<(" + banks + ")*nc; ++p) {");
      c.line("  const int64_t g = nc ? p / nc : 0, c = (nc ? p % nc : 0)*nr;");
      c.line("  unsigned char *out = (unsigned char*)" + dst + " + p*nr*(K+4);");
      c.line("  uint32_t *corr = (uint32_t*)(out + K*nr); memset(corr, 0, nr*sizeof(uint32_t));");
      c.line("  for (int64_t k=0; k<K; k+=4) {");
      c.line("    const int8_t *base = (const int8_t*)a" +
             std::to_string(slots[0]) + " + g*" +
             std::to_string(bb < 0 ? 0 : st[bb]) + "LL + c*" +
             std::to_string(st[bn]) + "LL + k*" + std::to_string(st[bk]) + "LL;");
      c.line("    const int8_t *r0 = base, *r1 = base + " +
             std::to_string(st[bk]) + "LL, *r2 = base + " +
             std::to_string(2 * st[bk]) + "LL, *r3 = base + " +
             std::to_string(3 * st[bk]) + "LL;");
      c.line("    unsigned char *dst_k = out + k*nr;");
      c.line("    #pragma omp simd");
      c.line("    for (int64_t j=0; j<nr; ++j) {");
      c.line("      int8_t v0 = r0[j*" + std::to_string(st[bn]) + "LL], v1 = r1[j*" +
             std::to_string(st[bn]) + "LL], v2 = r2[j*" +
             std::to_string(st[bn]) + "LL], v3 = r3[j*" +
             std::to_string(st[bn]) + "LL];");
      c.line("      dst_k[4*j] = (uint8_t)v0; dst_k[4*j+1] = (uint8_t)v1; dst_k[4*j+2] = (uint8_t)v2; dst_k[4*j+3] = (uint8_t)v3;");
      c.line("      corr[j] -= (uint32_t)(((int)v0 + (int)v1 + (int)v2 + (int)v3) * 128);");
      c.line("    }");
      c.line("  }");
      c.line("}");
    } else if (pk == "pack-vnni") {
      auto st = op.getAttrOfType<DenseI64ArrayAttr>("src_strides_0");
      int64_t kBlk = op.getAttrOfType<IntegerAttr>("k_block").getInt();
      int64_t bytes = kBlk == 4 ? 1 : 2;
      std::string cTy = bytes == 1 ? "uint8_t" : "uint16_t";
      int64_t bk = op.getAttrOfType<IntegerAttr>("rhs_k").getInt();
      int64_t bb = op.getAttrOfType<IntegerAttr>("rhs_batch").getInt();
      int64_t bn = 0;
      while (bn == bk || bn == bb)
        ++bn;
      std::string banks = bb < 0 ? "1LL" : "G";
      c.line("#pragma omp parallel for num_threads(workers) schedule(static) if((" +
             banks + ")*nc*K*nr > 131072 && (" + banks + ")*nc >= 4)");
      c.line("for (int64_t p=0; p<(" + banks + ")*nc; ++p) {");
      c.line("  const int64_t g = nc ? p / nc : 0, c = (nc ? p % nc : 0)*nr, valid = N-c < nr ? N-c : nr;");
      c.line("  const int64_t kPad = ((K+" + std::to_string(kBlk - 1) + ")/" +
             std::to_string(kBlk) + ")*" + std::to_string(kBlk) + ";");
      c.line("  " + cTy + " *out = (" + cTy + "*)" + dst + " + p*kPad*nr;");
      c.line("  if (valid < nr || kPad > K) memset(out, 0, (size_t)(kPad*nr)*sizeof(" + cTy + "));");
      c.line("  int64_t kb = 0;");
      c.line("  for (; kb + " + std::to_string(kBlk) + " <= K; kb += " +
             std::to_string(kBlk) + ") {");
      c.line("    const " + cTy + " *base = (const " + cTy + "*)a" +
             std::to_string(slots[0]) + " + g*" +
             std::to_string(bb < 0 ? 0 : st[bb]) + "LL + c*" +
             std::to_string(st[bn]) + "LL + kb*" + std::to_string(st[bk]) + "LL;");
      c.line("    " + cTy + " *dst_k = out + (kb/" + std::to_string(kBlk) +
             ")*nr*" + std::to_string(kBlk) + ";");
      c.line("    #pragma omp simd");
      c.line("    for (int64_t j=0; j<valid; ++j)");
      c.line("      for (int lane=0; lane<" + std::to_string(kBlk) + "; ++lane)");
      c.line("        dst_k[j*" + std::to_string(kBlk) +
             " + lane] = base[j*" + std::to_string(st[bn]) + "LL + lane*" +
             std::to_string(st[bk]) + "LL];");
      c.line("  }");
      c.line("  for (; kb < K; kb += " + std::to_string(kBlk) + ") {");
      c.line("    const " + cTy + " *base = (const " + cTy + "*)a" +
             std::to_string(slots[0]) + " + g*" +
             std::to_string(bb < 0 ? 0 : st[bb]) + "LL + c*" +
             std::to_string(st[bn]) + "LL + kb*" + std::to_string(st[bk]) + "LL;");
      c.line("    for (int64_t j=0; j<valid; ++j)");
      c.line("      for (int lane=0; lane<" + std::to_string(kBlk) + "; ++lane)");
      c.line("        if (kb+lane < K) out[(kb/" + std::to_string(kBlk) +
             ")*nr*" + std::to_string(kBlk) + " + j*" + std::to_string(kBlk) +
             " + lane] = base[j*" + std::to_string(st[bn]) + "LL + lane*" +
             std::to_string(st[bk]) + "LL];");
      c.line("  }");
      c.line("}");
    } else if (pk == "decode-mx8-lhs") {
      auto stA = op.getAttrOfType<DenseI64ArrayAttr>("src_strides_0");
      auto stS = op.getAttrOfType<DenseI64ArrayAttr>("src_strides_1");
      c.line("#pragma omp parallel for num_threads(workers) schedule(static) if(M*K>4096)");
      c.line("for (int64_t r=0; r<M; ++r) for (int64_t g=0; g<K/32; ++g) {");
      c.line("  unsigned sc = ((const uint8_t*)a" + std::to_string(slots[1]) +
             ")[r*" + std::to_string(stS[0]) + "LL + g*" +
             std::to_string(stS[1]) + "LL];");
      c.line("  for (int t=0; t<32; ++t) " + dst + "[r*K + g*32 + t] = decode_mx8(((const uint8_t*)a" +
             std::to_string(slots[0]) + ")[r*" + std::to_string(stA[0]) +
             "LL + g*" + std::to_string(stA[1]) + "LL + t*" +
             std::to_string(stA[2]) + "LL], sc);");
      c.line("}");
    } else if (pk == "decode-mx8-rhs") {
      auto stB = op.getAttrOfType<DenseI64ArrayAttr>("src_strides_0");
      auto stS = op.getAttrOfType<DenseI64ArrayAttr>("src_strides_1");
      int64_t colAx =
          op.getAttrOfType<IntegerAttr>("scale_column_axis").getInt();
      int64_t grpAx =
          op.getAttrOfType<IntegerAttr>("scale_group_axis").getInt();
      std::string scIdx = colAx == 0
                              ? ("c*" + std::to_string(stS[0]) + "LL + g*" +
                                 std::to_string(stS[1]) + "LL")
                              : ("g*" + std::to_string(stS[0]) + "LL + c*" +
                                 std::to_string(stS[1]) + "LL");
      c.line("#pragma omp parallel for num_threads(workers) schedule(static) if(N*K>4096)");
      c.line("for (int64_t g=0; g<K/32; ++g) for (int t=0; t<32; ++t) for (int64_t c=0; c<N; ++c)");
      c.line("  " + dst + "[(g*32+t)*N + c] = decode_mx8(((const uint8_t*)a" +
             std::to_string(slots[0]) + ")[g*" + std::to_string(stB[0]) +
             "LL + t*" + std::to_string(stB[1]) + "LL + c*" +
             std::to_string(stB[2]) + "LL], ((const uint8_t*)a" +
             std::to_string(slots[1]) + ")[" + scIdx + "]);");
      (void)grpAx;
    } else if (pk == "encode-q8k") {
      auto st = op.getAttrOfType<DenseI64ArrayAttr>("src_strides_0");
      c.line("for (int64_t g=0; g<K/256; ++g)");
      c.line("  pack_q8_record(&" + dst + "[g], (const float*)a" +
             std::to_string(slots[0]) + " + g*" + std::to_string(st[0]) +
             "LL, " + std::to_string(st[1]) + "LL);");
    } else if (pk == "gather-routed-lhs" ||
               pk == "gather-widen-routed-lhs-f32") {
      auto stA = op.getAttrOfType<DenseI64ArrayAttr>("src_strides_0");
      auto stO = op.getAttrOfType<DenseI64ArrayAttr>("src_strides_1");
      auto stR = op.getAttrOfType<DenseI64ArrayAttr>("src_strides_2");
      auto dt = op.getAttrOfType<StringAttr>("src_dtype_0").getValue();
      int64_t ak = op.getAttrOfType<IntegerAttr>("lhs_k").getInt();
      int64_t mult = op.getAttrOfType<IntegerAttr>("multiplicity").getInt();
      bool wide = pk == "gather-widen-routed-lhs-f32";
      c.line("const int64_t active_begin = ((const int32_t*)a" +
             std::to_string(slots[1]) + ")[0], active_end = ((const int32_t*)a" +
             std::to_string(slots[1]) + ")[G*" + std::to_string(stO[0]) + "LL];");
      c.line("#pragma omp parallel for num_threads(workers) schedule(static) if(active_end > active_begin + 1)");
      c.line("for (int64_t r=active_begin; r<active_end; ++r) {");
      c.line("  const int64_t src = ((const int32_t*)a" +
             std::to_string(slots[2]) + ")[r*" + std::to_string(stR[0]) +
             "LL] / " + std::to_string(mult) + "LL;");
      if (wide) {
        std::string fnWiden = dt == "bf16" ? "expand_bf" : "expand_f16";
        c.line("  const uint16_t *from = (const uint16_t*)a" +
               std::to_string(slots[0]) + " + src*" +
               std::to_string(stA[1 - ak]) + "LL;");
        c.line("  float *to = " + dst + " + r*K;");
        c.line("  for (int64_t k=0; k<K; ++k) to[k] = " + fnWiden + "(from[k*" +
               std::to_string(stA[ak]) + "LL]);");
      } else {
        int64_t bytes = dt == "f32" || dt == "i32" ? 4 : dt == "i8" || dt == "u8" ? 1 : 2;
        c.line("  const unsigned char *from = (const unsigned char*)a" +
               std::to_string(slots[0]) + " + src*" +
               std::to_string(stA[1 - ak] * bytes) + "LL;");
        c.line("  unsigned char *to = (unsigned char*)" + dst + " + r*K*" +
               std::to_string(bytes) + "LL;");
        if (stA[ak] == 1)
          c.line("  memcpy(to, from, K*" + std::to_string(bytes) + "LL);");
        else
          c.line("  for (int64_t k=0; k<K; ++k) memcpy(to+k*" +
                 std::to_string(bytes) + "LL, from+k*" +
                 std::to_string(stA[ak] * bytes) + "LL, " +
                 std::to_string(bytes) + ");");
      }
      c.line("}");
    }
    return llvm::Error::success();
  };

  p.operations["mojo_row.host_brgemm_plan"] = [](Operation &op, PrintPoint at,
                                                 PrintContext &c) {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    int64_t aDt = op.getAttrOfType<IntegerAttr>("a_dt").getInt();
    int64_t bDt = op.getAttrOfType<IntegerAttr>("b_dt").getInt();
    int64_t cDt = op.getAttrOfType<IntegerAttr>("c_dt").getInt();
    auto lda = op.getAttrOfType<StringAttr>("lda_expr").str();
    auto ldb = op.getAttrOfType<StringAttr>("ldb_expr").str();
    bool isQ4k = op.getAttrOfType<BoolAttr>("is_q4k").getValue();
    bool grouped = op.getAttrOfType<BoolAttr>("grouped").getValue();
    std::string frees;
    for (Attribute a : op.getAttrOfType<ArrayAttr>("cleanup")) {
      std::string prev = cast<StringAttr>(a).str();
      int pSlot = prev == "left" ? 0 : prev == "rhs" ? 1 : prev == "members" ? 2 : 3;
      frees += "intent_ws_release(" + std::to_string(pSlot) + ", " + prev + ");";
    }

    c.line("const void *brg_plan[2*(mr+1)]; memset(brg_plan, 0, sizeof(brg_plan));");
    c.line("const void *brg_first = NULL; int64_t brg_scratch_bytes = 64;");
    c.line("if (jobs > 0 && K > 0) {");
    c.indent++;
    c.line("uint8_t r_seen[mr+1]; memset(r_seen, 0, sizeof(r_seen));");
    if (grouped) {
      int64_t oSlot = op.getAttrOfType<IntegerAttr>("offsets_slot").getInt();
      int64_t oSt = op.getAttrOfType<IntegerAttr>("offsets_stride").getInt();
      c.line("for (int64_t g=0; g<G; ++g) {");
      c.line("  int64_t len = ((const int32_t*)a" + std::to_string(oSlot) +
             ")[(g+1)*" + std::to_string(oSt) + "LL] - ((const int32_t*)a" +
             std::to_string(oSt == 0 ? oSlot : oSlot) + ")[g*" +
             std::to_string(oSt) + "LL];");
      c.line("  if (len >= mr) r_seen[mr] = 1;");
      c.line("  if (len > 0 && len % mr) r_seen[len % mr] = 1;");
      c.line("}");
    } else {
      c.line("if (M >= mr) r_seen[mr] = 1;");
      c.line("if (M % mr) r_seen[M % mr] = 1;");
    }
    c.line("int64_t c_vals[2] = { N >= nr ? nr : 0, N % nr };");
    c.line("for (int ci=0; ci<2; ++ci) if (c_vals[ci] > 0) {");
    c.indent++;
    c.line("for (int64_t rt=1; rt<=mr; ++rt) if (r_seen[rt]) {");
    c.indent++;
    std::string kArg = isQ4k ? "32" : "K";
    std::string nArg = isQ4k ? "1" : "c_vals[ci]";
    c.line("const void *h = intent_brgemm_get(rt, " + nArg + ", " + kArg +
           ", " + lda + ", " + ldb + ", " + nArg + ", " + std::to_string(aDt) +
           ", " + std::to_string(bDt) + ", " + std::to_string(cDt) + ");");
    c.line("if (!h) { " + frees + "return 3; }");
    c.line("brg_plan[ci*(mr+1) + rt] = h;");
    c.line("if (!brg_first) brg_first = h;");
    c.line("int64_t sb = intent_brgemm_scratch_bytes(h); if (sb > brg_scratch_bytes) brg_scratch_bytes = sb;");
    c.indent--;
    c.line("}");
    c.indent--;
    c.line("}");
    c.indent--;
    c.line("}");
    return llvm::Error::success();
  };

  p.operations["mojo_row.host_parallel_tiles"] = [](Operation &op, PrintPoint at,
                                                    PrintContext &c) {
    bool usesBrg = op.getAttrOfType<BoolAttr>("uses_brgemm").getValue();
    if (at == PrintPoint::Enter) {
      bool grouped = op.getAttrOfType<BoolAttr>("grouped").getValue();
      bool routed = op.getAttrOfType<BoolAttr>("routed").getValue();
      bool windowed = op.getAttrOfType<BoolAttr>("windowed").getValue();
      c.line("if (jobs > 0) {");
      c.indent++;
      c.line("const int nt = jobs < workers ? (int)jobs : (int)workers;");
      if (usesBrg) {
        c.line("#pragma omp parallel num_threads(nt) if(nt > 1) reduction(|:rc)");
        c.line("{");
        c.indent++;
        c.line("_Alignas(64) unsigned char brg_scratch_stk[8192];");
        c.line("void *brg_scratch = brg_scratch_bytes <= 8192 ? (void*)brg_scratch_stk : aligned_alloc(64, (size_t)brg_scratch_bytes);");
        c.line("const void *brg_active = NULL;");
        c.line("int thr_ok = brg_scratch != NULL;");
        c.line("if (!thr_ok) rc |= 2;");
        c.line("#pragma omp for schedule(static) nowait");
      } else {
        c.line("#pragma omp parallel for num_threads(nt) schedule(static) if(nt > 1)");
      }
      c.line("for (int64_t job=0; job<jobs; ++job) {");
      c.indent++;
      if (usesBrg)
        c.line("if (!thr_ok) continue;");
      c.line("const int64_t g = job / (nc * mb), rem = job % (nc * mb);");
      c.line("const int64_t c = (rem / mb) * nr;");
      c.line("int64_t r = (rem % mb) * mr, end = M;");
      if (windowed) {
        int64_t H = op.getAttrOfType<IntegerAttr>("head_extent").getInt();
        int64_t chunks = op.getAttrOfType<IntegerAttr>("chunk_extent").getInt();
        c.line("const int64_t head=g%" + std::to_string(H) + ", chunk=g/" +
               std::to_string(H) + "%" + std::to_string(chunks) + ", batch=g/" +
               std::to_string(H) + "/" + std::to_string(chunks) + ";");
      }
      if (grouped) {
        int64_t oSlot = op.getAttrOfType<IntegerAttr>("offsets_slot").getInt();
        int64_t oSt = op.getAttrOfType<IntegerAttr>("offsets_stride").getInt();
        c.line("r += ((const int32_t*)a" + std::to_string(oSlot) + ")[g*" +
               std::to_string(oSt) + "LL];");
        c.line("end = ((const int32_t*)a" + std::to_string(oSlot) +
               ")[(g+1)*" + std::to_string(oSt) + "LL];");
        c.line("if (r >= end) continue;");
      }
      if (routed) {
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
      if (usesBrg) {
        c.line("const void *bh = brg_plan[(C == nr ? 0 : 1)*(mr+1) + R];");
        c.line("if (bh != brg_active) { if (intent_brgemm_thread_enter(bh) != 0) { rc |= 3; thr_ok = 0; continue; } brg_active = bh; }");
      }
    } else if (at == PrintPoint::EndBlock) {
      c.indent--;
      c.line("}");
      if (usesBrg) {
        c.line("if (brg_active) intent_brgemm_thread_leave();");
        c.line("if (brg_scratch != (void*)brg_scratch_stk) free(brg_scratch);");
        c.indent--;
        c.line("}");
      }
      c.indent--;
      c.line("}");
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
    SmallVector<std::string> args;
    for (int64_t i = 0; i < arity; ++i)
      args.push_back("p" + std::to_string(i));
    args.push_back("R");
    args.push_back("C");
    if (op.getAttrOfType<BoolAttr>("uses_brgemm").getValue()) {
      args.push_back("bh");
      args.push_back("brg_scratch");
    }
    c.line("intent_panel(" + llvm::join(args, ", ") + ");");
    return llvm::Error::success();
  };

  p.operations["mojo_row.host_scatter_routed"] = [](Operation &op,
                                                    PrintPoint at,
                                                    PrintContext &c) {
    if (at != PrintPoint::Enter)
      return llvm::Error::success();
    int64_t dstSlot = op.getAttrOfType<IntegerAttr>("dst_slot").getInt();
    int64_t mult = op.getAttrOfType<IntegerAttr>("multiplicity").getInt();
    auto st = op.getAttrOfType<DenseI64ArrayAttr>("dst_strides");
    int64_t bytes = op.getAttrOfType<IntegerAttr>("elem_bytes").getInt();
    int64_t rSlot = op.getAttrOfType<IntegerAttr>("routes_slot").getInt();
    int64_t rSt = op.getAttrOfType<IntegerAttr>("routes_stride").getInt();
    c.line("for (int64_t i=0; i<R; ++i) {");
    c.line("  const int64_t d_i = ((const int32_t*)a" + std::to_string(rSlot) +
           ")[(r+i)*" + std::to_string(rSt) + "LL];");
    c.line("  unsigned char *out = (unsigned char*)a" +
           std::to_string(dstSlot) + " + ((d_i/" + std::to_string(mult) +
           "LL)*" + std::to_string(st[0]) + "LL + (d_i%" +
           std::to_string(mult) + "LL)*" + std::to_string(st[1]) + "LL + c*" +
           std::to_string(st[2]) + "LL)*" + std::to_string(bytes) + "LL;");
    c.line("  const unsigned char *from = routed_out + i*nr*" +
           std::to_string(bytes) + "LL;");
    if (st[2] == 1)
      c.line("  memcpy(out, from, C*" + std::to_string(bytes) + "LL);");
    else
      c.line("  for (int64_t j=0; j<C; ++j) memcpy(out+j*" +
             std::to_string(st[2] * bytes) + "LL, from+j*" +
             std::to_string(bytes) + "LL, " + std::to_string(bytes) + ");");
    c.line("}");
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
