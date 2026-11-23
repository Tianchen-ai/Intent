#pragma once
#include "intentdsl/levels/storage/dsa/ir/ScoreSupply.h"
namespace CI_UNIT_NAMESPACE {
inline void scoreBody(llvm::raw_ostream &out, mlir::Operation &space,
                      llvm::StringRef params) {
  auto plan = intentdsl::dsa::scoreSupply(space);
  auto &dot = space.getRegion(0).front().front();
  out << "#define M " << plan.m << "\n#define N " << plan.n << "\n#define D "
      << plan.d << "\n#define V " << plan.v << "\n#define BK D\n#define BN N\n";
  out <<
#include "BFloatPanel.inc"
      ;
  out << R"mlu(
template<int K,int C> __mlu_func__ void score_product(float *out,unsigned short *x,unsigned short *y,
                bfloat16_t *filter,unsigned short *tmp,unsigned short *flags){
  if(ordinary_panel(x,M*K,tmp,flags) && ordinary_panel(y,C*K,tmp,flags)){
    constexpr int F=C<64?C:64;
    for(int f=0;f<C;f+=F){
      __memcpy(filter,y+f*K,K*F*2,NRAM2WRAM);
      float *dest=C==F?out:(float*)tmp;
      __bang_conv(dest,(bfloat16_t*)x,filter,K,M,1,1,1,1,1,F);
      __sync_compute();
      if(C!=F)__memcpy(out+f,dest,F*4,NRAM2NRAM,C*4,F*4,M-1);
    }
  }else{
    for(int i=0;i<M;++i)for(int j=0;j<C;++j){
      unsigned sum=0;
      for(int t=0;t<K;++t)
        sum=add_binary32(sum,multiply_bf16(x[i*K+t],y[j*K+t]));
      ((unsigned*)out)[i*C+j]=sum;
    }
  }
}
)mlu";
  out << "__mlu_global__ void ordered(" << params << "){\n";
  auto slots = space.getAttrOfType<mlir::DenseI64ArrayAttr>("dsa.panel_slots");
  out << "__nram__ unsigned char storage["
      << space.getAttrOfType<mlir::IntegerAttr>("dsa.member_bytes").getInt()
      << "];\n"
      << "__wram__ bfloat16_t filter[" << plan.filterBytes() / 2 << "];\n";
  int64_t offset = 0;
  const char *names[] = {"q",      "k",       "v",       "vt",
                         "scores", "prob",    "rounded", "value",
                         "out",    "scratch", "tmp",     "flags"};
  for (auto [i, n] : llvm::enumerate(names)) {
    const char *type =
        (i == 4 || i == 5 || i == 7 || i == 9) ? "float" : "unsigned short";
    out << type << " *" << n << "=(" << type << "*)(storage+" << offset
        << ");\n";
    offset += llvm::alignTo(slots[i], 128);
  }
  out << "for(int64_t job=taskId;job<grid;job+=taskDim){int64_t "
         "row=job%(a0d2/M)*M,h=job/(a0d2/M)%a0d1,b=job/(a0d2/M)/a0d1,head=h/"
      << dot.getAttrOfType<mlir::IntegerAttr>("group").getInt() << ";\n";
  out << R"mlu(
    __memcpy(q,a0+b*a0s0+h*a0s1+row*a0s2,M*D*2,GDRAM2NRAM);
    __memcpy(k,a1+b*a1s0+head*a1s1,N*D*2,GDRAM2NRAM);
    score_product<D,N>(scores,q,k,filter,tmp,flags);
    __bang_mul_scalar(scores,scores,a4*1.4426950408889634f,M*N);
    __sync_compute();
    for(int r=0;r<M;++r){
      float *s=scores+r*N,*p=prob+r*N;
      int live=row+r+1<N?(int)(row+r+1):N;
      float peak=row_max_num(scratch,s,live);
      for(int j=0;j<live;++j)p[j]=exp2f(s[j]-peak);
      for(int j=live;j<N;++j)p[j]=0;
      float mass=row_sum(scratch,p,N);
      narrow_bf16_tile(rounded+r*N,p,N,N);
      __sync_compute();
      scores[r*N]=1.f/mass;
    }
    __memcpy(v,a2+b*a2s0+head*a2s1,N*V*2,GDRAM2NRAM);
    __bang_transpose((short*)vt,(short*)v,N,V);
    __sync_compute();
    score_product<N,V>(value,rounded,vt,filter,tmp,flags);
    for(int r=0;r<M;++r){
      __bang_mul_scalar(value+r*V,value+r*V,scores[r*N],V);
      narrow_bf16_tile(out+r*V,value+r*V,V,V);
    }
    __sync_compute();
    __memcpy(a3+b*a3s0+h*a3s1+row*a3s2,out,M*V*2,NRAM2GDRAM);
  }
}
)mlu";
}
} // namespace CI_UNIT_NAMESPACE
