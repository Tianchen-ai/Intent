#include "intentdsl/levels/storage/dsa/ir/Matrix.h"
#include "llvm/ADT/StringExtras.h"
using namespace mlir;
namespace CI_UNIT_NAMESPACE {
namespace {
struct Matrix : intentdsl::block::MatrixAccess {
  using MatrixAccess::MatrixAccess;
  std::string src;
  llvm::raw_string_ostream s{src};
  static StringRef scalar(Type t) {
    return t.isF32()    ? "float"
           : t.isF16()  ? "half"
           : t.isBF16() ? "bfloat16_t"
                        : "int";
  }
  std::string abi(bool types) {
    SmallVector<std::string> vals;
    for (auto v : fn.getArguments()) {
      vals.push_back((types ? "void* " : "") + name(v));
      for (auto tag : {"d", "s"})
        for (int i = 0; i < rank(v); ++i)
          vals.push_back((types ? "long long " : "") + name(v) + tag +
                         std::to_string(i));
    }
    for (auto n : {"outer", "inner", "grid"})
      vals.push_back((types ? "long long " : "") + std::string(n));
    return llvm::join(vals, ",");
  }
  std::string ptr(Value v, StringRef element, ArrayRef<std::string> axes) {
    std::string result = "((" + element.str() + "*)" + name(v) + ")";
    for (auto [i, a] : llvm::enumerate(axes))
      result += "+((long long)(" + a + "))*" + std::to_string(stride(v, i));
    return "(" + result + ")";
  }
  std::string term(Value v, StringRef ty, int reduction, int batch,
                   StringRef coord, StringRef k) {
    SmallVector<std::string> indices;
    for (int i = 0; i < rank(v); ++i)
      indices.push_back(i == reduction ? k.str()
                        : i == batch   ? "g"
                                       : coord.str());
    return ptr(v, ty, indices);
  }
  std::string inputRow(StringRef i) {
    if (!routed())
      return "(r+" + i.str() + ")";
    auto ids = dot.getOperand(3), dst = domain.getOperand(0);
    auto q =
        fn.getArgAttrOfType<DenseI64ArrayAttr>(slot(dst), "block.shape")[1];
    return "(*" + ptr(ids, "int", {"r+" + i.str()}) + "/" + std::to_string(q) +
           ")";
  }
  void records() {
    auto w = dot.getOperand(0),
         x = dot.getOperand(1).getDefiningOp()->getOperand(0);
    s << "for(int i=0;i<nr;i++)acc[i]=0;\n"
      << "for(long long gq=0;gq<K/256;++gq){float e=0;\n"
      << "for(int k=0;k<256;++k){float x=*" << ptr(x, "float", {"gq", "k"})
      << ";if(fabsf(x)>fabsf(e)||(fabsf(x)==fabsf(e)&&x<e))e=x;}\n"
      << "if(e==0){for(int i=0;i<nr;++i){const unsigned char *header="
      << ptr(w,"unsigned char",{"r+i","gq","0"}) << ";"
      << "bool nonfinite=(header["<<stride(w,2)<<"LL]&0x7c)==0x7c || (header[3LL*"
      <<stride(w,2)<<"]&0x7c)==0x7c;acc[i]+=nonfinite?NAN:0.f;}continue;}\n"
      << "float inv=-127.f/e,ds=1.f/inv;int q[256],sums[8]={};\n"
      << "for(int k=0;k<256;++k){float f=*" << ptr(x, "float", {"gq", "k"})
      << "*inv;int lo=(int)floorf(f);f-=lo;lo+=(f>.5f||(f==.5f&&(lo&1)));"
         "q[k]=lo>127?127:lo< -128?-128:lo;sums[k/32]+=q[k];}\n"
      << "for(int i=0;i<nr;++i){unsigned char *base="
      << ptr(w, "unsigned char", {"r+i", "gq", "0"}) << ";\n";
    auto st = stride(w, 2);
    if (st == 1)
      s << "__memcpy(rec,base,144,GDRAM2NRAM);";
    else
      s << "for(long long z=0;z<144;z++)rec[z]=base[z*" << st << "];";
    s << "union{unsigned short u;half h;} "
         "d,dm;d.u=rec[0]|(rec[1]<<8);dm.u=rec[2]|(rec[3]<<8);int S=0,T=0;\n"
      << "for(int j=0;j<8;j++){int "
         "l=rec[4+j%4],h=rec[8+j%4],p=rec[12+j%4],sc=j<4?l&63:(p&15)|((l>>6)<<"
         "4),mn=j<4?h&63:(p>>4)|((h>>6)<<4);"
         "int dot=0;for(int "
         "t=0;t<32;t++)dot+=((rec[16+(j/"
         "2)*32+t]>>(4*(j%2)))&15)*q[j*32+t];S+=dot*sc;T+=sums[j]*mn;}"
         "acc[i]+=ds*((float)d.h*(float)S-(float)dm.h*(float)T);}}\n";
  }
  void operand(bool lhs, StringRef type) {
    auto v = dot.getOperand(lhs ? 0 : 1);
    if (dot.getName().stripDialect() == "window_contract") {
      auto &feeds = fold.getRegion(0).front().front().getRegion(0).front();
      auto transfer = (lhs ? feeds.front() : *feeds.front().getNextNode())
                          .getAttrOfType<StringAttr>("bangc.transfer");
      if (transfer == "transpose-window-dma") {
        if (type != "bfloat16_t")
          llvm_unreachable("verified BF16 window");
        s << "if(nr<BM || "
             "nk<BK)__bang_write_value((short*)trans,BK*BM,(short)0);"
          << "__memcpy(trans,"
          << ptr(v, type, {"batch", "chunk*K+kk", "head", "r"})
          << ",nr*2,GDRAM2NRAM,BM*2," << stride(v, 1) << "*2LL,nk-1);"
          << "__bang_transpose((short*)left,(short*)trans,BK,BM);\n";
        return;
      }
      if (transfer == "mapped-filter-major-dma") {
        auto step = dot.getOperand(2), prefix = dot.getOperand(3);
        s << "if(nc<BN || "
             "nk<BK)__bang_write_value((short*)right,BK*BN,(short)0);"
          << "__memcpy(right,"
          << ptr(v, type, {"batch", "chunk*K+kk", "group", "c"})
          << ",nc*2,GDRAM2NRAM,BN*2," << stride(v, 1) << "*2LL,nk-1);\n";
        for (auto [from, name] :
             {std::pair{step, "stepbuf"}, std::pair{prefix, "prefixbuf"}}) {
          if (stride(from, 3) == 1)
            s << "__memcpy(" << name << ","
              << ptr(from, "float", {"batch", "head", "chunk", "kk"})
              << ",nk*sizeof(float),GDRAM2NRAM);\n";
          else
            s << "__memcpy(" << name << ","
              << ptr(from, "float", {"batch", "head", "chunk", "kk"})
              << ",sizeof(float),GDRAM2NRAM,sizeof(float)," << stride(from, 3)
              << "*4LL,nk-1);\n";
        }
        auto scale = intentdsl::kir::panelExpression(
            dot, {"", "stepbuf[k]", "prefixbuf[k]", "last"},
            {"widen_value", "narrow_value", "min_ieee", "expf"},
            intentdsl::kir::rowScale(dot));
        s << "for(int k=0;k<BK;++k)scales[k]=k<nk?" << scale << ":0.f;\n"
          << "__bang_transpose((short*)trans,(short*)right,BK,BN);"
          << "widen_bf16_tile(mapped,(const unsigned short*)trans,BK*BN);"
          << "__bang_cycle_mul(mapped,mapped,scales,BN*BK,BK);"
          << "narrow_bf16_tile((unsigned short*)trans,mapped,BN*BK,BN*BK);\n";
        return;
      }
      s << "__bang_write_value((unsigned short*)"
        << (lhs ? "left,BM*BK" : "right,BK*BN") << ",(unsigned short)0);\n";
      if (lhs)
        s << "for(int k=0;k<nk;++k)for(int i=0;i<nr;++i)left[i*BK+k]=*"
          << ptr(v, type, {"batch", "chunk*K+kk+k", "head", "r+i"}) << ";\n";
      else {
        auto step = dot.getOperand(2), prefix = dot.getOperand(3);
        s << "for(int k=0;k<nk;++k){float d=*"
          << ptr(step, "float", {"batch", "head", "chunk", "kk+k"}) << ",v=*"
          << ptr(prefix, "float", {"batch", "head", "chunk", "kk+k"}) << ";\n"
          << "for(int j=0;j<nc;++j){auto raw=*"
          << ptr(v, type, {"batch", "chunk*K+kk+k", "group", "c+j"})
          << ";right[k*BN+j]="
          << intentdsl::kir::panelExpression(
                 dot, {"raw", "d", "v", "last"},
                 {"widen_value", "narrow_value", "min_ieee", "expf"})
          << ";}}\n";
      }
      return;
    }
    int k = axis(lhs ? "lhs_k" : "rhs_k"),
        b = axis(lhs ? "lhs_batch" : "rhs_batch"), member = freeAxis(v, k, b);
    auto &ops = fold.getRegion(0).front().front().getRegion(0).front();
    auto mode = (lhs ? ops.front() : *ops.front().getNextNode())
                    .getAttrOfType<StringAttr>("bangc.transfer");
    if (mode == "filter-major-dma") {
      s << "if(nc!=BN || nk!=BK)__bang_write_value((unsigned "
           "char*)trans,BK*BN*sizeof("
        << type << "),(unsigned char)0);\n"
        << "__memcpy(trans," << term(v, type, k, b, "c", "kk") << ",nk*sizeof("
        << type << "),GDRAM2NRAM,BK*sizeof(" << type << "),"
        << stride(v, member) << "*sizeof(" << type << "),nc-1);\n";
      return;
    }
    std::string buf = lhs ? "left" : "right", inner = lhs ? "nk" : "nc",
                outer = lhs ? "nr" : "nk", pitch = lhs ? "BK" : "BN";
    s << "if(" << outer << "!=" << (lhs ? "BM" : "BK") << "||" << inner
      << "!=" << pitch << ")__bang_write_value((unsigned char*)" << buf << ","
      << (lhs ? "BM*BK" : "BK*BN") << "*sizeof(" << type
      << "),(unsigned char)0);\n";
    auto base = term(v, type, k, b, lhs ? inputRow("0") : "c", "kk");
    if (mode == "routed-row-dma") {
      s << "for(int i=0;i<nr;++i)__memcpy(left+i*BK,"
        << term(v, type, k, b, inputRow("i"), "kk") << ",nk*sizeof(" << type
        << "),GDRAM2NRAM);\n";
    } else if (mode == "pitched-dma") {
      s << "__memcpy(" << buf << "," << base << "," << inner << "*sizeof("
        << type << "),GDRAM2NRAM," << pitch << "*sizeof(" << type << "),"
        << stride(v, lhs ? member : k) << "*sizeof(" << type << ")," << outer
        << "-1);\n";
    } else {
      auto address = term(v, type, k, b, lhs ? inputRow("i") : "(c+j)",
                          lhs ? "(kk+j)" : "(kk+i)");
      s << "for(int i=0;i<" << outer << ";i++)for(int j=0;j<" << inner
        << ";j++)" << buf << "[i*" << pitch << "+j]=*" << address << ";\n";
    }
  }
  void dense(StringRef mode, StringRef type) {
    if (mode == "exact-i8-conv")
      s << "__bang_write_value(acc,BM*BN,(int)0);\n";
    else
      s << "__bang_write_value(acc,BM*BN,0.f);\n";
    s << "for(long long kk=0;kk<K;kk+=BK){int nk=K-kk<BK?K-kk:BK;\n";
    operand(true, type);
    operand(false, type);
    if (mode == "wram-conv" || mode == "exact-i8-conv") {
      auto mode = fold.getRegion(0)
                      .front()
                      .front()
                      .getRegion(0)
                      .front()
                      .front()
                      .getNextNode()
                      ->getAttrOfType<StringAttr>("bangc.transfer");
      if (mode != "filter-major-dma" && mode != "mapped-filter-major-dma") {
        if (type == "bfloat16_t")
          s << "__bang_transpose((short*)trans,(short*)right,BK,BN);";
        else
          s << "__bang_transpose(trans,right,BK,BN);";
      }
      if (dot.hasAttr("bangc.bf16_arithmetic"))
        s << "__sync_compute();strict = strict || !ordinary_panel((unsigned "
             "short*)left,BM*BK,(unsigned short*)right,flags) || "
             "!ordinary_panel((unsigned short*)trans,BN*BK,(unsigned "
             "short*)right,flags);\n"
             "if(strict){for(int i=0;i<nr;i++)for(int j=0;j<nc;j++){unsigned "
             "z=((unsigned*)acc)[i*BN+j];"
             "for(int k=0;k<nk;k++)z=add_binary32(z,multiply_bf16(((unsigned "
             "short*)left)[i*BK+k],((unsigned short*)trans)[j*BK+k]));"
             "((unsigned*)acc)[i*BN+j]=z;}continue;}\n";
      s << "__sync_compute();__memcpy(filter,trans,BK*BN*sizeof(" << type
        << "),NRAM2WRAM);\n";
      if (dot.getAttrOfType<StringAttr>("bangc.matrix") == "exact-i8-conv") {
        // Legalization proves BK * 128 * 128 <= 2^24 before using this
        // exact partial. Panels and bias combine in the i32 accumulator.
        s << "__bang_conv(partial,left,filter,BK,BM,1,1,1,1,1,BN,0);"
             "__bang_float2int32_rn((int*)partial,partial,BM*BN,0);__bang_add("
             "acc,acc,(int*)partial,BM*BN);\n";
      } else
        s << "if(kk==0)__bang_conv(acc,left,filter,BK,BM,1,1,1,1,1,BN);"
             "else "
             "__bang_conv_partial(acc,left,filter,acc,BK,BM,1,1,1,1,1,BN);__"
             "sync_compute();\n";
    } else {
      s << "for(int i=0;i<nr;i++)for(int t=0;t<nk;t++)for(int "
           "j=0;j<nc;j++)acc[i*BN+j]+=("
        << "float)left[i*BK+t]*right[t*BN+j];\n";
    }
    s << "}\n";
  }
  void scaled() {
    auto a = dot.getOperand(0), sa = dot.getOperand(1), b = dot.getOperand(2),
         sb = intentdsl::kir::RightScale(dot).view;
    s << "__bang_write_value(acc,BM*BN,0.f);\nfor(long long u=0;u<K/32;++u){\n"
      << "for(int j=0;j<nc;++j){unsigned scale=*"
      << ptr(sb, "unsigned char", intentdsl::kir::RightScale(dot).index("c+j","u"))
      << ";"
         "for(int t=0;t<32;t++)right[t*BN+j]=decode8(*"
      << ptr(b, "unsigned char", {"u", "t", "c+j"}) << ",scale);}\n"
      << "for(int i=0;i<nr;++i){unsigned scale=*"
      << ptr(sa, "unsigned char", {"r+i", "u"})
      << ";for(int t=0;t<32;t++){"
         "float v=decode8(*"
      << ptr(a, "unsigned char", {"r+i", "u", "t"})
      << ",scale);"
         "__bang_mul_scalar(tmp,right+t*BN,v,BN);__bang_add(acc+i*BN,acc+i*BN,"
         "tmp,BN);}}}\n";
  }
  void write() {
    auto mode = dot.getAttrOfType<StringAttr>("bangc.matrix").getValue();
    auto a = dot.getOperand(0);
    auto elem = cast<MemRefType>(a.getType()).getElementType();
    StringRef type = elem.isBF16()       ? "bfloat16_t"
                     : elem.isF16()      ? "half"
                     : elem.isInteger(8) ? "signed char"
                                         : "float";
    auto dst = domain.getOperand(0);
    auto output = cast<MemRefType>(dst.getType()).getElementType();
    StringRef ot = output.isBF16()        ? "bfloat16_t"
                   : output.isF16()       ? "half"
                   : output.isInteger(32) ? "int"
                                          : "float";
    s << "#include <bang.h>\n#include <cnrt.h>\n#include <math.h>\n"
      << "#define BM " << tile[0] << "\n#define BN " << tile[1]
      << "\n#define BK " << tile[2] << "\n";
    if (dot.hasAttr("bangc.bf16_arithmetic"))
      s <<
#include "BFloatPanel.inc"
          ;
    if(mode=="exact-i8-conv")s<<R"mlu(
__mlu_func__ bfloat16_t signed_to_bf16(int x){
  union { unsigned short bits; bfloat16_t value; } out;
  unsigned mag=x<0?0u-(unsigned)x:(unsigned)x;
  if(!mag){out.bits=0;return out.value;}
  int exponent=0;
  for(unsigned v=mag;v>1;v>>=1)++exponent;
  int shift=exponent-7;
  unsigned sig=shift>0?mag>>shift:mag<<(-shift);
  if(shift>0){
    unsigned midpoint=1u<<(shift-1), discarded=mag&((1u<<shift)-1);
    if(discarded>midpoint || (discarded==midpoint && (sig&1)))++sig;
  }
  out.bits=(x<0?32768u:0u)|(((unsigned)(exponent+126)<<7)+sig);
  return out.value;
}
)mlu";
    if (dot.getName().stripDialect() == "window_contract") {
      s <<
#include "RowMath.inc"
          ;
      s << R"mlu(
__mlu_func__ float widen_value(bfloat16_t v){union{bfloat16_t b;unsigned short u;}x;union{unsigned u;float f;}f;x.b=v;f.u=(unsigned)x.u<<16;return f.f;}
__mlu_func__ float min_ieee(float a,float b){return isnan(a)?a:isnan(b)?b:fminf(a,b);}
)mlu";
    }
    if (mode == "scaled-f32")
      s << R"mlu(
__mlu_func__ float decode8(unsigned x,unsigned scale){
  unsigned mag=x&127,m=mag&7,e=mag>>3;
  union {unsigned u;float f;} val,factor;
  val.u=((x&128)<<24)|(mag==127?0x7fc00000u:((e+120)<<23)|(m<<20));
  factor.u=scale==255?0x7fc00000u:scale==0?0x00400000u:scale<<23;
  return (e?val.f:(x&128?-1.f:1.f)*m*.001953125f)*factor.f;
}
)mlu";
    s << "__mlu_global__ void matrix(" << abi(true)
      << "){constexpr long long K=" << shape[2] << ";"
      << "__nram__ " << (mode == "exact-i8-conv" ? "int" : "float")
      << " acc[BM*BN];\n";
    if (mode == "wram-conv" || mode == "exact-i8-conv" || mode == "f32-outer") {
      s << "__nram__ " << type << " left[BM*BK],right[BK*BN];\n";
      if (mode != "f32-outer")
        s << "__nram__ " << type << " trans[BK*BN];__wram__ " << type
          << " filter[BK*BN];\n";
      if (mode == "exact-i8-conv")
        s << "__nram__ float partial[BM*BN];\n";
    } else if (mode == "scaled-f32")
      s << "__nram__ float right[32*BN],tmp[BN];\n";
    else if (mode == "q4k-q8k")
      s << "__nram__ unsigned char rec[144];\n";
    if (dot.hasAttr("bangc.bf16_arithmetic"))
      s << "__nram__ unsigned short flags[BK*BN];\n";
    if (dot.getName().stripDialect() == "window_contract")
      s << "__nram__ float "
           "mapped[BK*BN],stepbuf[BK],prefixbuf[BK],scales[BK];\n";
    int64_t mb = llvm::divideCeil(uint64_t(shape[0]), uint64_t(tile[0])),
            nb = llvm::divideCeil(uint64_t(shape[1]), uint64_t(tile[1]));
    s << "for(long long job=taskId;job<grid;job+=taskDim){long long c=job%"
      << std::max<int64_t>(1, nb) << "*BN,r=(job/" << std::max<int64_t>(1, nb)
      << ")%" << std::max<int64_t>(1, mb) << "*BM,g=job/"
      << std::max<int64_t>(1, mb * nb) << ",end=" << shape[0] << ";\n";
    if (mode != "q4k-q8k" && dot.getName().stripDialect() == "contract" &&
        dot.getNumOperands() > 2) {
      auto o = dot.getOperand(2);
      s << "r+=*" << ptr(o, "int", {"g"}) << ";end=*" << ptr(o, "int", {"g+1"})
        << ";if(r>=end)continue;\n";
    }
    s << "int nr=end-r<BM?end-r:BM,nc=" << shape[1] << "-c<BN?" << shape[1]
      << "-c:BN;\n";
    if (dot.getName().stripDialect() == "window_contract") {
      auto sh =
          fn.getArgAttrOfType<DenseI64ArrayAttr>(slot(dst), "block.shape");
      auto h = std::max<int64_t>(1, sh[2]), c = std::max<int64_t>(1, sh[1]);
      s << "long long head=g%" << h << ",chunk=g/" << h << "%" << c
        << ",batch=g/" << h << "/" << c << ",group=head/" << axis("head_group")
        << ";"
        << "float last=K?*"
        << ptr(dot.getOperand(3), "float", {"batch", "head", "chunk", "K-1"})
        << ":0.f;\n";
    }
    if (dot.hasAttr("bangc.bf16_arithmetic"))
      s << "bool strict=false;\n";
    if (mode == "q4k-q8k")
      records();
    else if (mode == "scaled-f32")
      scaled();
    else
      dense(mode, type);
    auto &setup = fold.getRegion(0).front().front();
    auto writeback = setup.getAttrOfType<StringAttr>("bangc.writeback");
    auto narrow = [&]() {
      if (dot.hasAttr("bangc.bf16_arithmetic"))
        s << "if(strict){for(int i=0;i<BM*BN;i++)((unsigned "
             "short*)acc)[i]=narrow_bits(((unsigned*)acc)[i]);}else ";
      s << "__bang_float2bfloat16_rn((bfloat16_t*)acc,acc,BM*BN);\n";
    };
    if (writeback == "routed-row-dma") {
      auto q = std::to_string(
          fn.getArgAttrOfType<DenseI64ArrayAttr>(slot(dst), "block.shape")[1]);
      narrow();
      s << "for(int "
           "i=0;i<nr;i++){long long dest=*"
        << ptr(dot.getOperand(3), "int", {"r+i"}) << ";__memcpy("
        << ptr(dst, ot, {"dest/" + q, "dest%" + q, "c"})
        << ",((bfloat16_t*)acc)+i*BN,nc*sizeof(bfloat16_t),NRAM2GDRAM);}\n";
    } else if (writeback == "pitched-dma") {
      for (auto &ep : domain.getRegion(0).front())
        if (ep.getName().stripDialect() == "add_columns") {
          auto v = ep.getOperand(1);
          auto dtype = scalar(cast<MemRefType>(v.getType()).getElementType());
          s << "{__nram__ " << dtype << " bias[BN];";
          if (stride(v, 0) == 1)
            s << "__memcpy(bias," << ptr(v, dtype, {"c"}) << ",nc*sizeof("
              << dtype << "),GDRAM2NRAM);";
          else
            s << "for(int j=0;j<nc;j++)bias[j]=*" << ptr(v, dtype, {"c+j"})
              << ";";
          s << "for(int i=0;i<nr;i++)__bang_add(acc+i*BN,acc+i*BN,bias,BN);}\n";
        }
      if (ot == "bfloat16_t")
        narrow();
      if (ot == "half")
        s << "__bang_float2half_rn((half*)acc,acc,BM*BN);\n";
      if (rank(dst) == 1)
        s << "__memcpy(" << ptr(dst, ot, {"r"}) << ",acc,nr*sizeof(" << ot
          << "),NRAM2GDRAM);";
      else
        s << "__memcpy("
          << (rank(dst) == 5
                  ? ptr(dst, ot, {"batch", "chunk", "head", "r", "c"})
              : rank(dst) == 2 ? ptr(dst, ot, {"r", "c"})
                               : ptr(dst, ot, {"g", "r", "c"}))
          << ",acc,nc*sizeof(" << ot << "),NRAM2GDRAM,"
          << stride(dst, rank(dst) - 2) << "*sizeof(" << ot << "),BN*sizeof("
          << ot << "),nr-1);";
      s << "\n";
    } else {
      s << "for(int i=0;i<nr;i++){\n";
      if (routed())
        s << "long long dest=*" << ptr(dot.getOperand(3), "int", {"r+i"})
          << ";\n";
      else
        s << "long long dest=r+i;\n";
      s << "for(int j=0;j<nc;j++){auto value=acc["
        << (rank(dst) == 1 ? "i" : "i*BN+j") << "];\n";
      std::string val = "value";
      unsigned stage = 0;
      for (auto &op : domain.getRegion(0).front()) {
        auto n = op.getName().stripDialect();
        if (n != "add_columns" && n != "convert_matrix")
          continue;
        auto type = scalar(
            cast<ShapedType>(op.getResult(0).getType()).getElementType());
        auto next = "v" + std::to_string(stage++);
        if(n=="convert_matrix" && type=="bfloat16_t" &&
           cast<ShapedType>(op.getOperand(0).getType()).getElementType().isInteger(32)){
          s<<"bfloat16_t "<<next<<"=signed_to_bf16("<<val<<");\n"; val=next;continue;
        }
        if (n == "convert_matrix" && type == "bfloat16_t" &&
            dot.hasAttr("bangc.bf16_arithmetic") &&
            cast<ShapedType>(op.getOperand(0).getType())
                .getElementType()
                .isF32()) {
          s << "bfloat16_t " << next << "=narrow_value(" << val << ");\n";
          val = next;
          continue;
        }
        s << type << " " << next << "=(" << type << ")(";
        if (n == "add_columns")
          s << (type == "int" ? "(unsigned)" : "") << val << "+*"
            << ptr(op.getOperand(1), type, {rank(dst) == 1 ? "dest" : "c+j"});
        else
          s << val;
        s << ");\n";
        val = next;
      }
      if (rank(dst) == 1)
        s << "*" << ptr(dst, ot, {"dest"});
      else if (rank(dst) == 2)
        s << "*" << ptr(dst, ot, {"dest", "c+j"});
      else if (routed()) {
        auto q = std::to_string(fn.getArgAttrOfType<DenseI64ArrayAttr>(
            slot(dst), "block.shape")[1]);
        s << "*" << ptr(dst, ot, {"dest/" + q, "dest%" + q, "c+j"});
      } else if (rank(dst) == 5)
        s << "*" << ptr(dst, ot, {"batch", "chunk", "head", "dest", "c+j"});
      else
        s << "*" << ptr(dst, ot, {"g", "dest", "c+j"});
      s << "=(" << ot << ")" << val << ";}}";
    }
    s << "}}\nextern \"C\" int intent_enqueue(" << abi(true)
      << ",int cores,void *queue){if(!grid)return 0;cnrtDim3_t "
         "d={(unsigned)(grid<cores?grid:cores),1,1};"
      << "matrix<<<d,cnrtFuncTypeBlock,(cnrtQueue_t)queue>>>(" << abi(false)
      << ");return (int)cnrtGetLastError();}\n";
  }
};
} // namespace
llvm::Error emitMatrix(func::FuncOp fn, llvm::raw_pwrite_stream &out) {
  Matrix m(fn);
  m.write();
  auto &s = m.fold.getRegion(0).front().front();
  auto a = llvm::json::parse(fn->getAttrOfType<StringAttr>("kir.abi"));
  if (!a)
    return a.takeError();
  auto tasks = s.getAttrOfType<IntegerAttr>("participants").getInt(),
       bytes = s.getAttrOfType<IntegerAttr>("scratch_bytes").getInt();
  auto grid = llvm::divideCeil(uint64_t(m.shape[0]), uint64_t(m.tile[0])) *
              llvm::divideCeil(uint64_t(m.shape[1]), uint64_t(m.tile[1])) *
              m.shape[3];
  out << llvm::json::Value(llvm::json::Object{
      {"version", 2},
      {"source", m.src},
      {"abi", std::move(*a)},
      {"domain", intentdsl::publishDomain(fn)},
      {"grid", llvm::json::Array{int64_t(grid), 1, 1}},
      {"guard", llvm::json::Object{{"min_width", 0}, {"max_width", nullptr}}},
      {"rows", llvm::json::Array{llvm::json::Object{{"id", 0},
                                                    {"rows", m.tile[0]},
                                                    {"width", m.tile[1]},
                                                    {"pitch", m.tile[2]},
                                                    {"bytes", bytes},
                                                    {"tasks", tasks}}}}});
  return llvm::Error::success();
}
} // namespace CI_UNIT_NAMESPACE
