#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "intentdsl/levels/execution/cpu/ir/MatrixPanels.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
using namespace mlir;
namespace CI_UNIT_NAMESPACE {
namespace {
using intentdsl::cpu::MatrixPanels;
using intentdsl::cpu::PanelView;
std::string dtype(StringRef t) {
  return llvm::StringSwitch<std::string>(t)
      .Case("bf16", "bfloat16")
      .Case("f16", "float16")
      .Case("i8", "int8")
      .Case("i32", "int32")
      .Case("u8", "uint8")
      .Case("f8e4m3fn", "float8_e4m3fn")
      .Case("f32", "float32")
      .Default("");
}
struct Writer {
  MatrixPanels &p;
  std::string source;
  llvm::raw_string_ostream o{source};
  void line(unsigned depth, const Twine &text) {
    o.indent(depth * 4) << text << "\n";
  }
  std::string address(unsigned slot, ArrayRef<std::string> indices) {
    std::string a = "0";
    for (auto [s, i] : llvm::zip_equal(p.views[slot].strides, indices))
      a += "+" + i + "*" + std::to_string(s);
    return a;
  }
  std::string load(unsigned slot, ArrayRef<std::string> axes) {
    return "p" + std::to_string(slot) +
           "[unsafe_offset=" + address(slot, axes) + "]";
  }
  void records() {
    line(1, "for r in range(Int(R)):");
    line(2, "var acc=Float32(0)");
    line(2, "for g in range(" + std::to_string(p.shape[2] / 256) + "):");
    line(3, "var ds=p1.bitcast[Float32]()[unsafe_offset=g*73]");
    auto byte = [&](std::string k) {
      return "UInt16(" + load(0, {"r", "g", k}) + ")";
    };
    line(3, "var d=Float32(bitcast[DType.float16](" + byte("0") + " | (" +
                byte("1") + " << 8)))");
    line(3, "var dm=Float32(bitcast[DType.float16](" + byte("2") + " | (" +
                byte("3") + " << 8)))");
    line(3, "var S=Int32(0)");
    line(3, "var T=Int32(0)");
    line(3, "comptime for j in range(8):");
    line(4, "var l=Int32(" + byte("(4+j%4)") + ")");
    line(4, "var h=Int32(" + byte("(8+j%4)") + ")");
    line(4, "var u=Int32(" + byte("(12+j%4)") + ")");
    line(4, "var sc=(l&63) if j<4 else (u&15)|((l>>6)<<4)");
    line(4, "var mn=(h&63) if j<4 else (u>>4)|((h>>6)<<4)");
    line(4, "var "
            "q=p1.unsafe_offset(g*292+4+j*32).unsafe_load[width=32]().cast["
            "DType.int32]()");
    if (p.views[0].strides.back() == 1)
      line(4, "var raw=p0.unsafe_offset(" +
                  address(0, {"r", "g", "(16+(j//2)*32)"}) +
                  ").unsafe_load[width=32]()");
    else {
      line(4, "var raw=SIMD[DType.uint8,32](0)");
      line(4, "comptime for k in range(32):");
      line(5, "raw[k]=" + load(0, {"r", "g", "(16+(j//2)*32+k)"}));
    }
    line(4, "var "
            "nib=((raw>>SIMD[DType.uint8,32](UInt8(4*(j%2))))&15).cast[DType."
            "int32]()");
    line(4, "var sums=p1.bitcast[Int16]().unsafe_offset(g*146+130+2*j)");
    line(4, "S+=sc*(q*nib).reduce_add()");
    line(4,
         "T+=mn*(Int32(sums[unsafe_offset=0])+Int32(sums[unsafe_offset=1]))");
    line(3, "acc+=ds*(d*Float32(S)-dm*Float32(T))");
    std::string value = "acc";
    unsigned bias = 2, stage = 0;
    for (auto &op : p.domain.getRegion(0).front()) {
      if (op.getName().stripDialect() == "add_columns")
        line(2, value + "+=" + load(bias++, {"r"}));
      if (op.getName().stripDialect() == "convert_matrix") {
        auto t = cast<ShapedType>(op.getResult(0).getType()).getElementType();
        auto next = "v" + std::to_string(++stage);
        if (t.isBF16() && cast<ShapedType>(op.getOperand(0).getType())
                              .getElementType()
                              .isF32())
          line(2, "var " + next + "=round_bf16[1](" + value + ")");
        else
          line(2, "var " + next + "=" + value + ".cast[DType." +
                      (t.isBF16()  ? "bfloat16"
                       : t.isF16() ? "float16"
                       : t.isF32() ? "float32"
                                   : "int32") +
                      "]()");
        value = next;
      }
    }
    line(2, load(p.views.size() - 1, {"r"}) + "=" + value);
  }
  void vectorBody(unsigned depth, int lanes, int rows = 1, int vectors = 1) {
    unsigned last = p.views.size() - 1;
    auto accum = p.views[0].dtype == "i8" ? "int32" : "float32";
    std::string W = std::to_string(lanes),
                vec = "SIMD[DType." + std::string(accum) + "," + W + "]",
                cast = ".cast[DType." + std::string(accum) + "]()";
    auto loadColumns = [&](unsigned i, ArrayRef<std::string> axes,
                           std::string local, unsigned d) {
      auto t = dtype(p.views[i].dtype), base = address(i, axes);
      if (lanes == 1 || p.views[i].strides.back() == 1) {
        line(d, "var " + local + "=p" + std::to_string(i) + ".unsafe_offset(" +
                    base + ").unsafe_load[width=" + W + "]()");
      } else {
        line(d, "var " + local + "=SIMD[DType." + t + "," + W + "](0)");
        line(d, "comptime for lane in range(" + W + "):");
        line(d + 1, local + "[lane]=p" + std::to_string(i) +
                        "[unsafe_offset=" + base + "+lane*" +
                        std::to_string(p.views[i].strides.back()) + "]");
      }
    };
    auto id = [](int r, int c) {
      return std::to_string(r) + "_" + std::to_string(c);
    };
    auto coord = [](const char *base, int delta) {
      return "(" + std::string(base) + "+" + std::to_string(delta) + ")";
    };
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < vectors; ++c)
        line(depth, "var acc" + id(r, c) + "=" + vec + "(0)");
    line(depth, "for panel in range(0," + std::to_string(p.shape[2]) + "," +
                    std::to_string(p.tile[2]) + "):");
    line(depth + 1, "for k in range(panel,min(panel+" +
                        std::to_string(p.tile[2]) + "," +
                        std::to_string(p.shape[2]) + ")):");
    for (int c = 0; c < vectors; ++c) {
      auto rhs = "right" + std::to_string(c);
      loadColumns(1, {"k", coord("c", c * lanes)}, rhs, depth + 2);
      line(depth + 2, "var b" + std::to_string(c) + "=" + rhs + cast);
    }
    for (int r = 0; r < rows; ++r) {
      auto left = "left" + std::to_string(r);
      line(depth + 2, "var " + left + "=" + vec + "(" +
                          load(0, {coord("r", r), "k"}) + cast + ")");
      for (int c = 0; c < vectors; ++c) {
        auto dst = "acc" + id(r, c), right = "b" + std::to_string(c);
        line(depth + 2,
             p.views[0].dtype == "i8"
                 ? dst + "+=" + left + "*" + right
                 : dst + "=fma(" + left + "," + right + "," + dst + ")");
      }
    }
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < vectors; ++c) {
        auto key = id(r, c), col = coord("c", c * lanes);
        std::string result = "acc" + key;
        unsigned bias = 2, converts = 0;
        for (auto &op : p.domain.getRegion(0).front()) {
          auto n = op.getName().stripDialect();
          if (n == "add_columns") {
            auto v = "bias" + std::to_string(bias) + "_" + key;
            loadColumns(bias++, {col}, v, depth);
            line(depth, result + "+=" + v);
          }
          if (n == "convert_matrix") {
            auto t = mlir::cast<ShapedType>(op.getResult(0).getType())
                         .getElementType();
            auto dest = "converted" + std::to_string(converts++) + "_" + key;
            if (t.isBF16() && mlir::cast<ShapedType>(op.getOperand(0).getType())
                                  .getElementType()
                                  .isF32())
              line(depth,
                   "var " + dest + "=round_bf16[" + W + "](" + result + ")");
            else
              line(depth, "var " + dest + "=" + result + ".cast[DType." +
                              (t.isF16()    ? "float16"
                               : t.isBF16() ? "bfloat16"
                               : t.isF32()  ? "float32"
                                            : "int32") +
                              "]()");
            result = dest;
          }
        }
        auto addr = address(last, {coord("r", r), col});
        if (p.views[last].strides.back() == 1 || lanes == 1)
          line(depth, "p" + std::to_string(last) + ".unsafe_offset(" + addr +
                          ").unsafe_store(" + result + ")");
        else {
          line(depth, "comptime for lane in range(" + W + "):");
          line(depth + 1, "p" + std::to_string(last) +
                              "[unsafe_offset=" + addr + "+lane*" +
                              std::to_string(p.views[last].strides.back()) +
                              "]=" + result + "[lane]");
        }
      }
  }
  void write() {
    SmallVector<std::string> params;
    for (auto [i, v] : llvm::enumerate(p.views))
      params.push_back("p" + std::to_string(i) + ": Pointer[Scalar[DType." +
                       (i == 1 && p.dot.getName().stripDialect() == "q4k_dot"
                            ? "int8"
                            : dtype(v.dtype)) +
                       "],MutUntrackedOrigin]");
    o << "from std.memory import bitcast\nfrom std.math import fma\n";
    o << R"mojo(
@always_inline
def round_bf16[W:Int](x:SIMD[DType.float32,W])->SIMD[DType.bfloat16,W]:
    var bits=bitcast[DType.uint32,W](x)
    var high=bits>>16
    var rounded=(bits+0x7fff+(high&1))>>16
    var result=(bits&0x7fffffff).gt(0x7f800000).select(high|0x40,rounded)
    return bitcast[DType.bfloat16,W](result.cast[DType.uint16]())
)mojo";
    auto packing = p.domain.getAttrOfType<StringAttr>("cpu.packing");
    bool dot4 = packing && packing == "signed-dot4-panels";
    if (dot4)
      o << "from std.sys import llvm_intrinsic\n";
    o << "@export(\"intent_panel\")\ndef intent_panel("
      << llvm::join(params, ",") << ", R:Int64,C:Int64) abi(\"C\"):\n";
    if (p.dot.getName().stripDialect() == "q4k_dot") {
      records();
      return;
    }
    if (dot4) {
      integerDot();
      return;
    }
    line(1, "if Int(R)==" + std::to_string(p.tile[0]) +
                " and Int(C)==" + std::to_string(p.tile[1]) + ":");
    line(2, "var r=0");
    line(2, "var c=0");
    vectorBody(2, p.tile[3], p.tile[0], p.tile[1] / p.tile[3]);
    line(2, "return");
    line(1, "for r in range(Int(R)):");
    line(2, "var c=0");
    line(2, "while c+" + std::to_string(p.tile[3]) + "<=Int(C):");
    vectorBody(3, p.tile[3]);
    line(3, "c+=" + std::to_string(p.tile[3]));
    line(2, "while c<Int(C):");
    vectorBody(3, 1);
    line(3, "c+=1");
  }
  void integerDot() {
    int vectors = p.tile[1] / 16;
    auto body = [&](int rows) {
      auto id = [](int r, int v) {
        return std::to_string(r) + "_" + std::to_string(v);
      };
      for (int v = 0; v < vectors; ++v) {
        auto offset = std::to_string(p.shape[2] * p.tile[1] + 64 * v);
        line(2, "var correction" + std::to_string(v) + "=p1.unsafe_offset(" +
                    offset + ").bitcast[Int32]().unsafe_load[width=16]()");
        for (int m = 0; m < rows; ++m)
          line(2, "var sum" + id(m, v) + "=correction" + std::to_string(v));
      }
      line(2, "for k in range(0," + std::to_string(p.shape[2]) + ",4):");
      for (int v = 0; v < vectors; ++v)
        line(3, "var b" + std::to_string(v) + "=p1.unsafe_offset(k*" +
                    std::to_string(p.tile[1]) + "+" + std::to_string(64 * v) +
                    ").unsafe_load[width=64]()");
      for (int m = 0; m < rows; ++m) {
        auto mi = std::to_string(m);
        line(3, "var word" + mi + "=p0.unsafe_offset(" +
                    address(0, {"(r+" + mi + ")", "k"}) +
                    ").bitcast[Int32]().unsafe_load() ^ Int32(-2139062144)");
        line(3, "var a" + mi +
                    "=bitcast[DType.uint8,64](SIMD[DType.int32,16](word" + mi +
                    "))");
        for (int v = 0; v < vectors; ++v) {
          auto name = "sum" + id(m, v);
          line(3, name +
                      "=llvm_intrinsic[\"llvm.x86.avx512.vpdpbusd.512\",SIMD["
                      "DType.int32,16]](" +
                      name + ",a" + mi + ",b" + std::to_string(v) + ")");
        }
      }
      for (int v = 0; v < vectors; ++v) {
        unsigned bias = 2, last = p.views.size() - 1;
        auto col = std::to_string(v * 16);
        for (auto &op : p.domain.getRegion(0).front())
          if (op.getName().stripDialect() == "add_columns") {
            auto key = "bias" + std::to_string(bias) + "_" + std::to_string(v);
            if (p.views[bias].strides[0] == 1)
              line(2, "var " + key + "=p" + std::to_string(bias) +
                          ".unsafe_offset(" + col +
                          ").unsafe_load[width=16]()");
            else {
              line(2, "var " + key + "=SIMD[DType.int32,16](0)");
              line(2, "comptime for j in range(16):");
              line(3, key + "[j]=" + load(bias, {"(" + col + "+j)"}));
            }
            for (int m = 0; m < rows; ++m)
              line(2, "sum" + id(m, v) + "+=" + key);
            ++bias;
          }
        for (int m = 0; m < rows; ++m) {
          auto row = "(r+" + std::to_string(m) + ")";
          if (p.views[last].strides[1] == 1)
            line(2, "p" + std::to_string(last) + ".unsafe_offset(" +
                        address(last, {row, col}) + ").unsafe_store(sum" +
                        id(m, v) + ")");
          else {
            line(2, "comptime for j in range(16):");
            line(3, load(last, {row, "(" + col + "+j)"}) + "=sum" + id(m, v) +
                        "[j]");
          }
        }
      }
    };
    line(1, "if Int(R)==" + std::to_string(p.tile[0]) + ":");
    line(2, "var r=0");
    body(p.tile[0]);
    line(2, "return");
    line(1, "for r in range(Int(R)):");
    body(1);
  }
};
} // namespace
static MatrixPanels::PackedRight packedRight(MatrixPanels &p) {
  auto mode=p.domain.getAttrOfType<StringAttr>("cpu.packing");
  if(!mode || mode!="signed-dot4-panels")return {};
  MatrixPanels::PackedRight pack;llvm::raw_string_ostream s(pack.prepare);
  auto rhs = p.dot.getOperand(1);
    auto i = cast<BlockArgument>(rhs).getArgNumber();
    auto t = p.fn.getArgAttrOfType<DenseI64ArrayAttr>(i, "cpu.strides");
    int k = p.dot.getAttrOfType<IntegerAttr>("rhs_k").getInt(),
        b = p.dot.getAttrOfType<IntegerAttr>("rhs_batch").getInt(), n = 0;
    while (n == k || n == b)
      ++n;
    s << "const int64_t K=" << p.shape[2]
      << ",tiles=N/nr,panels=" << (b < 0 ? "1" : "G") << "*tiles;\n"
      << "const size_t bytes=nr*(K+4);unsigned char "
         "*rhs=calloc(panels?panels:1,bytes);if(!rhs)return 2;\n"
      << "#pragma omp parallel for num_threads(workers) schedule(static) "
         "if(panels>1)\n"
      << "for(int64_t panel=0;panel<panels;panel++){int64_t "
         "g=panel/tiles,c=panel%tiles*nr;unsigned char *p=rhs+panel*bytes;"
         "uint32_t *correction=(uint32_t*)(p+K*nr);"
         "for(int64_t k=0;k<K;k+=4){const int8_t *base=(int8_t*)a"
      << i << "+g*" << (b < 0 ? 0 : t[b]) << "+c*" << t[n] << "+k*" << t[k]
      << ";"
         "for(int j=0;j<nr;j++){int sum=0;for(int lane=0;lane<4;lane++){int8_t "
         "x=base[j*"
      << t[n] << "+lane*" << t[k]
      << "];"
         "p[k*nr+4*j+lane]=(uint8_t)x;sum+=x;}correction[j]-=(uint32_t)(sum*"
         "128);}}}\n";
  pack.address=std::string("rhs+(")+(b<0?"0":"g")+"*tiles+c/nr)*bytes";pack.release="free(rhs);";return pack;
}
llvm::Error emitMatrix(func::FuncOp fn, llvm::raw_pwrite_stream &out) {
  MatrixPanels p(fn);
  Writer w{p};
  w.write();
  SmallVector<std::string> args, declarations;
  for (size_t i = 0; i < p.views.size(); ++i) {
    args.push_back("p" + std::to_string(i));
    declarations.push_back("void*");
  }
  auto host =
      p.dispatch("extern void intent_panel(" + llvm::join(declarations, ",") +
                     ",int64_t,int64_t);",
                 "intent_panel(" + llvm::join(args, ",") + ",R,C);", true,packedRight(p));
  auto abi = llvm::json::parse(fn->getAttrOfType<StringAttr>("kir.abi"));
  if (!abi)
    return abi.takeError();
  out << llvm::json::Value(llvm::json::Object{
      {"version", 2},
      {"source", w.source},
      {"host", host},
      {"returns_status", true},
      {"domain", intentdsl::publishDomain(fn)},
      {"abi", std::move(*abi)},
      {"rows", llvm::json::Array{llvm::json::Object{{"grain", p.tile[0]},
                                                    {"columns", p.tile[1]}}}}});
  return llvm::Error::success();
}
} // namespace CI_UNIT_NAMESPACE
