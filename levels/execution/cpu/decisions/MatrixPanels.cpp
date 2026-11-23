#include "intentdsl/levels/execution/cpu/ir/MatrixPanels.h"
#include "llvm/ADT/StringExtras.h"
using namespace mlir;
namespace intentdsl::cpu {
MatrixPanels::MatrixPanels(func::FuncOp f)
    : fn(f), domain(f.front().front().getRegion(0).front().front()),
      dot(calculation(domain)),
      shape(domain.getAttrOfType<DenseI64ArrayAttr>("cpu.extents")),
      tile(domain.getAttrOfType<DenseI64ArrayAttr>("cpu.output_tile")) {
  auto sig = llvm::cantFail(
      llvm::json::parse(fn->getAttrOfType<StringAttr>("kir.abi")));
  auto slot = [](Value v) { return cast<BlockArgument>(v).getArgNumber(); };
  auto origin = [&](Value v, ArrayRef<std::string> indices, ArrayRef<int> axes,
                    bool out = false) {
    auto i = slot(v);
    auto s = fn.getArgAttrOfType<DenseI64ArrayAttr>(i, "cpu.strides");
    std::string address = "0";
    for (auto [ax, coord] : llvm::enumerate(indices))
      if (coord != "0")
        address += "+" + coord + "*" + std::to_string(s[ax]);
    std::vector<int64_t> step;
    for (auto ax : axes)
      step.push_back(s[ax]);
    views.push_back(
        {i, (*sig.getAsArray())[i].getAsObject()->getString("dtype")->str(),
         address, step, out});
  };
  auto n = dot.getName().stripDialect();
  if (n == "contract") {
    auto panel = [&](Value v, int k, int batch, bool lhs) {
      auto rank = cast<MemRefType>(v.getType()).getRank();
      SmallVector<std::string> idx(rank, "0");
      int free = 0;
      while (free == k || free == batch)
        ++free;
      idx[free] = lhs ? "src" : "c";
      if (batch >= 0)
        idx[batch] = "g";
      origin(v, idx,
             lhs ? SmallVector<int>{free, k} : SmallVector<int>{k, free});
    };
    auto axis = [&](StringRef s) {
      return dot.getAttrOfType<IntegerAttr>(s).getInt();
    };
    panel(dot.getOperand(0), axis("lhs_k"), axis("lhs_batch"), true);
    panel(dot.getOperand(1), axis("rhs_k"), axis("rhs_batch"), false);
    if (domain.hasAttr("cpu.packing"))
      views[1].strides = {tile[1], 1};
    auto ids = [&](Value v) {
      auto i = slot(v);
      auto s = fn.getArgAttrOfType<DenseI64ArrayAttr>(i, "cpu.strides")[0];
      return "((int32_t*)a" + std::to_string(i) + ")[%s*" + std::to_string(s) +
             "]";
    };
    if (dot.getNumOperands() > 2)
      offsets = ids(dot.getOperand(2));
    if (dot.getNumOperands() > 3) {
      routes = ids(dot.getOperand(3));
      multiplicity = fn.getArgAttrOfType<DenseI64ArrayAttr>(
          slot(domain.getOperand(0)), "cpu.shape")[1];
    }
  } else if (n == "window_contract") {
    origin(dot.getOperand(0), {"batch", "chunk*K", "head", "src"}, {3, 1});
    views.push_back(
        {slot(dot.getOperand(1)), "bf16", "(g*K)*N+c", {shape[1], 1}, false});
    if (domain.getAttrOfType<StringAttr>("cpu.packing") ==
        "mapped-f32-panels") {
      views[0].dtype = views[1].dtype = "f32";
      views[0].origin = "(g*M+src)*K";
      views[0].strides = {shape[2], 1};
    }
  } else if (n == "scaled_contract") {
    views.push_back({slot(dot.getOperand(0)),
                     "f32",
                     "src*" + std::to_string(shape[2]),
                     {shape[2], 1},
                     false});
    views.push_back(
        {slot(dot.getOperand(2)), "f32", "c", {shape[1], 1}, false});
  } else {
    origin(dot.getOperand(0), {"src", "0", "0"}, {0, 1, 2});
    origin(dot.getOperand(1).getDefiningOp()->getOperand(0), {"0", "0"},
           {0, 1});
  }
  auto dst = domain.getOperand(0);
  for (auto &op : domain.getRegion(0).front())
    if (op.getName().stripDialect() == "add_columns")
      origin(op.getOperand(1),
             {cast<MemRefType>(dst.getType()).getRank() == 1 ? "dest" : "c"},
             {0});
  if (n == "window_contract")
    origin(dst, {"batch", "chunk", "head", "dest", "c"}, {3, 4}, true);
  else if (!routes.empty())
    origin(dst,
           {"(dest/" + std::to_string(multiplicity) + ")",
            "(dest%" + std::to_string(multiplicity) + ")", "c"},
           {0, 2}, true);
  else
    switch (cast<MemRefType>(dst.getType()).getRank()) {
    case 1:
      origin(dst, {"dest"}, {0}, true);
      break;
    case 2:
      origin(dst, {"dest", "c"}, {0, 1}, true);
      break;
    case 3:
      origin(dst, {"g", "dest", "c"}, {1, 2}, true);
      break;
    }
  if (domain.hasAttr("cpu.route_storage")) {
    views.front().strides = {shape[2], 1};
    views.back().strides = {tile[1], 1};
  }
}
std::string MatrixPanels::dispatch(StringRef decl, StringRef invoke,
                                   bool status,const PackedRight &native) {
  std::string text;
  llvm::raw_string_ostream s(text);
  auto index = [](std::string expr, StringRef var) {
    expr.replace(expr.find("%s"), 2, var);
    return expr;
  };
  auto packing = domain.getAttrOfType<StringAttr>("cpu.packing");
  bool pack = packing && packing == "shared-q8k-per-call";
  bool decoded = packing && packing == "decoded-f32-per-call";
  bool widened = packing && packing == "mapped-f32-panels";
  bool mapped = widened || (packing && packing == "mapped-right-per-call");
  bool packRhs = packing && packing == "rhs-panels-per-call";
  auto placement=domain.getAttrOfType<StringAttr>("cpu.record_storage");
  const bool heap=placement && placement=="heap";
  auto releaseRight=packRhs?std::string("free(rhs);"):native.release;
  s << "#include <stdint.h>\n#include <stddef.h>\n#include <math.h>\n#include "
       "<stdlib.h>\n#include <string.h>\n#include <omp.h>\n"
    << decl << "\n";
  if (mapped)
    s << R"c(
static inline float expand_bf(uint16_t v){uint32_t b=(uint32_t)v<<16;float f;memcpy(&f,&b,4);return f;}
static inline uint16_t narrow_bf(float f){uint32_t b;memcpy(&b,&f,4);if((b&0x7fffffff)>0x7f800000)return (b>>16)|64;return (b+32767+((b>>16)&1))>>16;}
static inline float min_ieee(float a,float b){return isnan(a)?a:isnan(b)?b:fminf(a,b);}
)c";
  if (decoded)
    s << R"c(
static inline float decode(unsigned x,unsigned scale){
  unsigned mag=x&127,ex=mag>>3,m=mag&7,sign=(x&128)<<24;
  uint32_t bits=sign|(mag==127?0x7fc00000U:((ex+120)<<23)|(m<<20));
  float v;if(ex==0)v=(x&128?-1.0f:1.0f)*m*0.001953125f;else memcpy(&v,&bits,4);
  uint32_t sbits=scale==255?0x7fc00000U:scale==0?0x00400000U:scale<<23;
  float factor;memcpy(&factor,&sbits,4);return v*factor;
}
)c";
  s << (status ? "int" : "void") << " intent_entry(" << matrixHostParams(fn)
    << "){\n"
    << "const int64_t M=" << shape[0] << ",N=" << shape[1] << ",G=" << shape[3]
    << ";int rc=0;\n"
    << "const int64_t mr=" << tile[0] << ",nr=" << tile[1] << ";\n"
    << "const int64_t jobs=G*((M+mr-1)/mr)*((N+nr-1)/nr);\n";
  if (mapped) {
    auto output = domain.getOperand(0);
    auto i = cast<BlockArgument>(output).getArgNumber();
    auto sh = fn.getArgAttrOfType<DenseI64ArrayAttr>(i, "cpu.shape");
    auto access = [&](Value v, StringRef type, ArrayRef<std::string> idx) {
      auto i = cast<BlockArgument>(v).getArgNumber();
      auto strides = fn.getArgAttrOfType<DenseI64ArrayAttr>(i, "cpu.strides");
      std::string base = "((" + type.str() + "*)a" + std::to_string(i) + ")[0";
      for (auto [j, k] : llvm::enumerate(idx))
        base += "+(" + k + ")*" + std::to_string(strides[j]);
      return base + "]";
    };
    s << "const int64_t H=" << std::max<int64_t>(1, sh[2])
      << ",chunks=" << std::max<int64_t>(1, sh[1]) << ",K=" << shape[2] << ";\n"
      << "size_t elems;if(__builtin_mul_overflow((size_t)G,(size_t)K,&elems) "
         "|| __builtin_mul_overflow(elems,(size_t)N,&elems))return 2;"
      << (widened ? "float" : "uint16_t")
      << " *right=calloc(elems?elems:1,sizeof(*right));if(!right)return 2;\n";
    if (widened) {
      s << "size_t lsz;if(__builtin_mul_overflow((size_t)G,(size_t)M,&lsz) || "
           "__builtin_mul_overflow(lsz,(size_t)K,&lsz)){free(right);return 2;}"
        << "float "
           "*left=calloc(lsz?lsz:1,sizeof(*left));if(!left){free(right);return "
           "2;}\n"
        << "#pragma omp parallel for num_threads(workers) schedule(static) "
           "if(G>1)\n"
        << "for(int64_t g=0;g<G;++g){int64_t "
           "batch=g/H/chunks,chunk=g/H%chunks,head=g%H;"
        << "for(int64_t m=0;m<M;++m)for(int64_t "
           "k=0;k<K;++k)left[(g*M+m)*K+k]=expand_bf("
        << access(dot.getOperand(0), "uint16_t",
                  {"batch", "chunk*K+k", "head", "m"})
        << ");}\n";
    }
    s << "#pragma omp parallel for num_threads(workers) schedule(static) "
         "if(G*K>1)\n"
      << "for(int64_t p=0;p<G*K;++p){int64_t "
         "g=p/K,k=p%K,head=g%H,chunk=g/H%chunks,batch=g/H/chunks,group=head/"
      << dot.getAttrOfType<IntegerAttr>("head_group").getInt() << ";\n";
    auto a = access(dot.getOperand(1), "uint16_t",
                    {"batch", "chunk*K+k", "group", "n"});
    auto c =
        access(dot.getOperand(2), "float", {"batch", "head", "chunk", "k"});
    auto d =
        access(dot.getOperand(3), "float", {"batch", "head", "chunk", "k"});
    auto l =
        access(dot.getOperand(3), "float", {"batch", "head", "chunk", "K-1"});
    kir::PanelMath math{"expand_bf", "narrow_bf", "min_ieee", "expf"};
    if (auto scalar = kir::rowScale(dot)) {
      s << "const float factor="
        << kir::panelExpression(dot, {a, c, d, l}, math, scalar)
        << ";\n#pragma omp simd\n"
        << "for(int64_t n=0;n<N;++n)right[p*N+n]="
        << (widened ? "expand_bf(" : "(") << "narrow_bf(expand_bf(" << a
        << ")*factor));}\n";
    } else
      s << "for(int64_t n=0;n<N;++n)right[p*N+n]="
        << (widened ? "expand_bf(" : "(")
        << kir::panelExpression(dot, {a, c, d, l}, math) << ");}\n";
  }
  s<<native.prepare;
  if (packRhs) {
    auto b = dot.getOperand(1);
    auto i = cast<BlockArgument>(b).getArgNumber();
    auto stride = fn.getArgAttrOfType<DenseI64ArrayAttr>(i, "cpu.strides");
    int k = dot.getAttrOfType<IntegerAttr>("rhs_k").getInt(),
        batch = dot.getAttrOfType<IntegerAttr>("rhs_batch").getInt(), n = 0;
    while (n == k || n == batch)
      ++n;
    int bytes = cast<MemRefType>(b.getType()).getElementTypeBitWidth() / 8;
    s << "const int64_t K=" << shape[2] << ",banks=" << (batch < 0 ? "1" : "G")
      << ",tiles=(N+nr-1)/nr;\n"
      << "size_t "
         "count;if(__builtin_mul_overflow((size_t)(banks*tiles),(size_t)K,&"
         "count))return 2;\n"
      << "unsigned char *rhs=calloc(count?count:1,nr*" << bytes
      << ");if(!rhs)return 2;\n"
      << "#pragma omp parallel for num_threads(workers) schedule(static) "
         "if(banks*tiles>1)\n"
      << "for(int64_t b=0;b<banks*tiles;b++){int64_t "
         "g=b/tiles,c=b%tiles*nr,nc=N-c<nr?N-c:nr;for(int64_t k=0;k<K;k++){"
      << "unsigned char *dst=rhs+(b*K+k)*nr*" << bytes
      << ";unsigned char *src=(unsigned char*)a" << i << "+(g*"
      << (batch < 0 ? 0 : stride[batch]) << "+k*" << stride[k] << "+c*"
      << stride[n] << ")*" << bytes << ";";
    if (stride[n] == 1)
      s << "memcpy(dst,src,nc*" << bytes << ");";
    else
      s << "for(int64_t j=0;j<nc;j++)memcpy(dst+j*" << bytes << ",src+j*"
        << stride[n] * bytes << "," << bytes << ");";
    s << "}}\n";
  }
  if (decoded) {
    int64_t k = shape[2];
    s << "const int64_t K=" << k << ";if(!M || !N)return 0;\n"
      << "float* workspace=malloc(sizeof(float)*(size_t)K*(M+N));if(K && "
         "!workspace)return 2;\n"
      << "float* left=workspace,*right=workspace+M*K;\n";
    auto acc = [&](Value v, ArrayRef<std::string> axes) {
      auto i = cast<BlockArgument>(v).getArgNumber();
      auto t = fn.getArgAttrOfType<DenseI64ArrayAttr>(i, "cpu.strides");
      std::string a = "0";
      for (auto [ax, j] : llvm::enumerate(axes))
        a += "+(int64_t)(" + j + ")*" + std::to_string(t[ax]);
      return "((uint8_t*)a" + std::to_string(i) + ")[" + a + "]";
    };
    s << "#pragma omp parallel for num_threads(workers) schedule(static) "
         "if(M*K>4096)\n"
      << "for(int64_t r=0;r<M;r++)for(int64_t g=0;g<K/32;g++){unsigned scale="
      << acc(dot.getOperand(1), {"r", "g"}) << ";\n"
      << "for(int t=0;t<32;++t)left[r*K+g*32+t]=decode("
      << acc(dot.getOperand(0), {"r", "g", "t"}) << ",scale);}\n"
      << "#pragma omp parallel for num_threads(workers) schedule(static) "
         "if(N*K>4096)\n"
      << "for(int64_t g=0;g<K/32;g++)for(int t=0;t<32;t++)for(int64_t "
         "c=0;c<N;c++)right[(g*32+t)*N+c]=decode("
      << acc(dot.getOperand(2), {"g", "t", "c"}) << ","
      << acc(kir::RightScale(dot).view, kir::RightScale(dot).index("c","g"))
      << ");\n";
  }
  if (pack) {
    auto &x = views[1];
    int64_t groups = shape[2] / 256;
    s << "struct Record {float d;int8_t q[256];int16_t sums[16];};\n"
      << "_Static_assert(sizeof(struct Record)==292,\"Q8_K record ABI\");\n";
    if (heap)
      s << "struct Record *packed=calloc(" << groups
        << ",sizeof(*packed));if(!packed)return 2;\n";
    else
      s << "struct Record packed[" << std::max<int64_t>(1, groups) << "];\n";
    s << "for(int64_t g=0;g<" << groups << ";++g){float *x=(float*)a" << x.slot
      << "+g*" << x.strides[0] << ";uint32_t peak=0;\n"
      << "#pragma omp simd reduction(max:peak)\n"
      << "for(int64_t k=0;k<256;++k){uint32_t bits;memcpy(&bits,x+k*"
      << x.strides[1]
      << ",4);uint32_t key=(bits<<1)|(bits>>31);if(key>peak)peak=key;}\n"
      << "uint32_t bits=(peak>>1)|(peak<<31);float e;memcpy(&e,&bits,4);\n"
      << "float inv=e==0?0.0f:-127.0f/e;packed[g].d=e==0?0.0f:1.0f/inv;\n"
      << "#pragma omp simd\n"
      << "for(int64_t k=0;k<256;++k){float v=x[k*" << x.strides[1]
      << "]*inv;int trunc=(int)v;float f=fabsf(v-(float)trunc);"
         "int round=(f>0.5f)||(f==0.5f&&(trunc&1));int "
         "q=trunc+(v<0?-round:round);packed[g].q[k]=q>127?127:q< "
         "-128?-128:q;}\n"
         "for(int j=0;j<16;++j){int sum=0;for(int "
         "k=j*16;k<(j+1)*16;++k)sum+=packed[g].q[k];packed[g].sums[j]=sum;}}\n";
  }
  bool routed = domain.hasAttr("cpu.route_storage");
  auto elemBytes = [&](unsigned i) {
    return cast<MemRefType>(fn.getArgument(i).getType())
               .getElementTypeBitWidth() /
           8;
  };
  if (routed) {
    auto &input = views.front();
    auto strides =
        fn.getArgAttrOfType<DenseI64ArrayAttr>(input.slot, "cpu.strides");
    auto k = dot.getAttrOfType<IntegerAttr>("lhs_k").getInt(), width = shape[2];
    auto bytes = elemBytes(input.slot);
    s << "size_t elements;if(__builtin_mul_overflow((size_t)M,(size_t)" << width
      << ",&elements)){"
      << releaseRight
      << "return 2;}"
         "unsigned char *members=calloc(elements?elements:1,"
      << bytes << ");if(!members){"
      << releaseRight
      << "return 2;}\n"
      << "int64_t active_begin="<<index(offsets,"0")<<",active_end="<<index(offsets,"G")<<";\n"
      << "#pragma omp parallel for num_threads(workers) schedule(static) "
         "if(active_end>active_begin+1)\n"
         "for(int64_t r=active_begin;r<active_end;r++){int64_t src="
      << index(routes, "r") << "/" << multiplicity
      << ";unsigned char *from=(unsigned char*)a" << input.slot << "+src*"
      << strides[1 - k] * bytes << ",*to=members+r*" << width * bytes << ";";
    if (strides[k] == 1)
      s << "memcpy(to,from," << width * bytes << ");}\n";
    else
      s << "for(int64_t k=0;k<" << width << ";++k)memcpy(to+k*" << bytes
        << ",from+k*" << strides[k] * bytes << "," << bytes << ");}\n";
  }
  s << "if(jobs){int nt=jobs<workers?jobs:workers;\n#pragma omp parallel for "
       "num_threads(nt) schedule(static) if(nt>1) reduction(|:rc)\n"
    << "for(int64_t job=0;job<jobs;++job){int64_t "
       "nc=(N+nr-1)/nr,mb=(M+mr-1)/mr;\n"
    << "int64_t g=job/(nc*mb),c=(job%nc)*nr,r=(job/nc%mb)*mr,end=M;\n";
  if (mapped)
    s << "int64_t head=g%H,chunk=g/H%chunks,batch=g/H/chunks;\n";
  if (!offsets.empty())
    s << "r+=" << index(offsets, "g") << ";end=" << index(offsets, "(g+1)")
      << ";if(r>=end)continue;\n";
  s << "int64_t dest=" << (routes.empty() ? "r" : index(routes, "r")) << ",src="
    << (routes.empty() ? "r" : "dest/" + std::to_string(multiplicity)) << ";\n"
    << "int64_t R=end-r<mr?end-r:mr,C=N-c<nr?N-c:nr;\n";
  for (auto [i, v] : llvm::enumerate(views)) {
    if (widened && !i) {
      s << "void *p0=left+" << v.origin << ";\n";
      continue;
    }
    if (mapped && i == 1) {
      s << "void *p1=right+" << v.origin << ";\n";
      continue;
    }
    if (routed && i == 0) {
      s << "void *p0=members+r*" << shape[2] * elemBytes(v.slot) << ";\n";
      continue;
    }
    if (routed && v.writes) {
      s << "_Alignas(64) unsigned char output["
        << tile[0] * tile[1] * elemBytes(v.slot) << "];void *p" << i
        << "=output;\n";
      continue;
    }
    if(!native.address.empty() && i==1){s<<"void *p1="<<native.address<<";\n";continue;}
    if (i == 1 && packRhs) {
      auto bytes = cast<MemRefType>(fn.getArgument(v.slot).getType())
                       .getElementTypeBitWidth() /
                   8;
      s << "void *p1=rhs+(("
        << (dot.getAttrOfType<IntegerAttr>("rhs_batch").getInt() < 0 ? "0"
                                                                     : "g")
        << "*tiles+c/nr)*K)*nr*" << bytes << ";\n";
      continue;
    }
    if (decoded && i < 2) {
      s << "float *p" << i << "=" << (i == 0 ? "left" : "right") << "+("
        << v.origin << ");\n";
      continue;
    }
    if (pack && i == 1) {
      s << "void *p1=packed;\n";
      continue;
    }
    int bytes = cast<MemRefType>(fn.getFunctionType().getInput(v.slot))
                    .getElementTypeBitWidth() /
                8;
    s << "void *p" << i << "=(char*)a" << v.slot << "+(" << v.origin << ")*"
      << bytes << ";\n";
  }
  s << invoke << "\n";
  if (routed) {
    auto &dst = views.back();
    auto bytes = elemBytes(dst.slot);
    auto strides =
        fn.getArgAttrOfType<DenseI64ArrayAttr>(dst.slot, "cpu.strides");
    s << "for(int64_t i=0;i<R;i++){int64_t dest=" << index(routes, "(r+i)")
      << "; unsigned char *out=(unsigned char*)a" << dst.slot << "+("
      << dst.origin << ")*" << bytes << ",*from=output+i*nr*" << bytes << ";";
    if (strides[2] == 1)
      s << "memcpy(out,from,C*" << bytes << ");}\n";
    else
      s << "for(int64_t j=0;j<C;j++)memcpy(out+j*" << strides[2] * bytes
        << ",from+j*" << bytes << "," << bytes << ");}\n";
  }
  s << "}}" << (routed ? "free(members);" : "") << (heap ? "free(packed);" : "")
    << (decoded ? "free(workspace);" : "") << (mapped ? "free(right);" : "")
    << (widened ? "free(left);" : "")
    << releaseRight
    << (status ? "return rc;" : "") << "}\n";
  return text;
}
} // namespace intentdsl::cpu
