R"mlu(
__mlu_func__ float row_sum(float *scratch, const float *input, int count) {
  int full = count / 32 * 32;
  float value = 0.0f;
  if (full) {
    __bang_sumpool(scratch, input, 32, full/32, 1, full/32, 1, 1, 1);
    __bang_reduce_sum(scratch, scratch, 32);
    __sync_compute();
    value = ((volatile float *)scratch)[0];
  }
  for (int k=full; k<count; ++k) value += input[k];
  return value;
}

__mlu_func__ float row_max_num(float *scratch, const float *input, int count) {
  if (count >= 4 && !isnan(input[0])) {
    __bang_argmax(scratch, input, count);
    __sync_compute();
    union { float f; unsigned u; } high;
    high.f = ((volatile float *)scratch)[0];
    unsigned mag = high.u & 0x7fffffff;
    if (mag >= 0x00800000 && mag <= 0x7f800000) return high.f;
  }
  float value = -INFINITY;
  for (int k=0; k<count; ++k) value = fmaxf(value,input[k]);
  return value;
}

// The native high-precision exponential needs a repair for extreme inputs.
// Inspect only active members when determining whether that repair is needed.
__mlu_func__ void row_exp(float *out, const float *in, int live, int lanes) {
  bool repair = true;
  if (live >= 4) {
    __bang_abs(out,in,lanes);
    __bang_argmax(out,out,live);
    __sync_compute();
    repair = !(((volatile float *)out)[0] <= 80.0f);
  }
  __bang_active_exphp(out,in,lanes);
  if (repair) {
    __sync_compute();
    for(int k=0;k<live;++k) {
      float x=in[k];
      if(x < -80.0f || x > 80.0f)
        out[k]=x==-INFINITY ? 0.0f : (x==INFINITY ? INFINITY : expf(x));
    }
  }
}

__mlu_func__ unsigned short round_half_bits(float numerator,float divisor) {
  union {float f; unsigned u;} quotient, left, right;
  left.f=numerator; right.f=divisor;
  quotient.f=__cn_scalar_div_f32_rn(numerator,divisor);
  unsigned mag=quotient.u & 0x7fffffff,sign=((left.u^right.u)>>16)&0x8000;
  if(mag>=0x7f800000)return sign|(mag==0x7f800000?0x7c00:0x7e00);
  if(mag>=0x477ff000)return sign|0x7c00;
  if(mag<0x33000000)return sign;
  unsigned exp=mag>>23;
  unsigned shift=exp<113?126-exp:13;
  unsigned mant=exp<113?((mag&0x7fffff)|0x800000):mag-0x38000000;
  unsigned base=mant>>shift, remainder=mant&((1u<<shift)-1);
  unsigned halfWay=1u<<(shift-1);
  return sign|(base+(remainder>halfWay || (remainder==halfWay && (base&1))));
}
__mlu_func__ void row_exact_half(half *out, float *tmp,const float *x,float d,int live,int lanes) {
  for(int k=0;k<live;++k)tmp[k]=__cn_scalar_div_f32_rn(x[k],d);
  __bang_float2half_rn(out,tmp,lanes);
  __sync_compute();
  union {float f; unsigned bits;} den;den.f=d;
  for(int k=0;k<live;++k){
    if((((volatile unsigned short*)out)[k]&0x7fff)==0){
      union {float f; unsigned bits;} num;num.bits=((const volatile unsigned*)x)[k];
      ((volatile unsigned short*)out)[k]=(num.bits^den.bits)>>16 & 0x8000;
    }
  }
}
// For 1 <= d <= 2^20 the reciprocal is normal. Two FP32 RN operations
// (reciprocal and multiply) differ from RN(x/d) by at most four adjacent
// float representations, including binade transitions. Underflow in this
// domain rounds to signed half zero. Compare the endpoint encodings as bits:
// equal endpoints certify the final half rounding; ambiguous lanes use x/d.
__mlu_func__ void row_narrow_quotient(half *out,float *q,float *tmp,
                                      const float *x,float d,int live,int lanes) {
  if (!(d >= 1.0f && d <= 1048576.0f)) {
    row_exact_half(out,tmp,x,d,live,lanes);
    return;
  }
  __bang_mul_scalar(q,x,__cn_scalar_div_f32_rn(1.f,d),lanes);
  __bang_add_scalar((int*)tmp,(const int*)q,-4,lanes);
  __bang_float2half_rn(out,tmp,lanes);
  __bang_add_scalar((int*)q,(const int*)q,4,lanes);
  half *upper=(half*)tmp;
  __bang_float2half_rn(upper,q,lanes);
  __bang_bxor((short*)upper,(short*)upper,(short*)out,lanes);
  __bang_int162float(q,(const short*)upper,lanes,0);
  unsigned ambiguous=__bang_count(q,live);
  if(ambiguous>16) {
    row_exact_half(out,tmp,x,d,live,lanes);
    return;
  }
  for(unsigned k=0;k<ambiguous;++k) {
    unsigned i=__bang_findfirst1(q,live);
    ((volatile unsigned short*)out)[i]=round_half_bits(x[i],d);
    ((volatile float*)q)[i]=0;
  }
}
)mlu"
