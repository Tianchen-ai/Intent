R"mlu(
// Values in the fast interval have a minimum quantum of 2^-55. A sum
// of their products cannot enter the binary32 subnormal range. K fits
// i32, so even the largest sum stays finite.
__mlu_func__ bool ordinary_panel(const unsigned short *v,int n,unsigned short *mag,unsigned short *flag){
  for(int i=0;i<n;i+=BK*BN){
    int count=n-i<BK*BN?n-i:BK*BN;
    __bang_band_scalar(mag,v+i,(unsigned short)0x7fff,count);
    __bang_sub_scalar(flag,mag,(unsigned short)((127-48)<<7),count);
    __bang_ge_scalar(flag,flag,(unsigned short)(95<<7),count);
    __bang_mul(flag,mag,flag,count);
    if(__bang_count(flag,count))return false;
  }
  return true;
}
__mlu_func__ unsigned sticky(unsigned v,int n){
  if(n==0)return v;
  if(n>=32)return v!=0;
  return (v>>n)|((v<<(32-n))!=0);
}
__mlu_func__ int high_bit(unsigned v){
  int top=0;
  if(v>>16){top=16;v>>=16;}
  if(v>>8){top+=8;v>>=8;}
  if(v>>4){top+=4;v>>=4;}
  if(v>>2){top+=2;v>>=2;}
  return top+(int)(v>>1);
}
__mlu_func__ unsigned multiply_bf16(unsigned a,unsigned b){
  unsigned sign=((a^b)&0x8000u)<<16;
  unsigned x=a&32767u,y=b&32767u,ae=x>>7,be=y>>7;
  if(x>0x7f80 || y>0x7f80 || (x==0x7f80 && !y) || (!x && y==0x7f80))return 0x7fc00000u;
  if(ae==255 || be==255)return sign|0x7f800000u;
  if(!x || !y)return sign;
  unsigned v=((x&127)+(ae?128:0))*((y&127)+(be?128:0));
  int offset=(int)(ae?ae:1)+(int)(be?be:1)-268;
  int top=high_bit(v),exponent=offset+top+127;
  if(exponent>=255)return sign|0x7f800000u;
  if(exponent>0)return sign|((unsigned)exponent<<23)|((v<<(23-top))&0x7fffffu);
  int shift=-(offset+149);
  if(shift<=0)return sign|(v<<(-shift));
  if(shift>16)return sign;
  unsigned q=v>>shift, tail=v&((1u<<shift)-1),halfway=1u<<(shift-1);
  return sign|(q+(tail>halfway || (tail==halfway && (q&1))));
}
__mlu_func__ unsigned add_binary32(unsigned a,unsigned b){
  unsigned x=a&0x7fffffffu,y=b&0x7fffffffu;
  if(x<y){unsigned z=x;x=y;y=z;z=a;a=b;b=z;}
  if(x>=0x7f800000u){
    if(x>0x7f800000u || (y==x && ((a^b)>>31)))return 0x7fc00000u;
    return a;
  }
  if(!y)return x?a:(a&b);
  int e=x>>23,f=y>>23;
  unsigned m=((x&0x7fffff)|(e?1u<<23:0))<<3;
  unsigned n=((y&0x7fffff)|(f?1u<<23:0))<<3;
  if(!e)e=1;if(!f)f=1;n=sticky(n,e-f);
  if((a^b)>>31){
    m-=n;if(!m)return 0;
    int shift=26-high_bit(m);if(shift>e-1)shift=e-1;
    m<<=shift;e-=shift;
  }else{
    m+=n;if(m&(1u<<27)){m=sticky(m,1);++e;}
  }
  unsigned tail=m&7;m=(m>>3)+(tail>4 || (tail==4 && (m&8)));
  if(m&(1u<<24)){m>>=1;++e;}
  if(e>=255)return (a&0x80000000u)|0x7f800000u;
  return (a&0x80000000u)|(m<(1u<<23)?m:((unsigned)e<<23)|(m&0x7fffffu));
}
__mlu_func__ unsigned short narrow_bits(unsigned w){
  if((w&0x7fffffffu)>0x7f800000u)return (w>>16)|0x40;
  return (w+0x7fffu+((w>>16)&1u))>>16;
}
__mlu_func__ bfloat16_t narrow_value(float value){
  union {unsigned u;float f;} x;
  union {unsigned short u;bfloat16_t b;} y;
  x.f=value;y.u=narrow_bits(x.u);return y.b;
}
)mlu"
