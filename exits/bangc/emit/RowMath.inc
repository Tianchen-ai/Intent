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
)mlu"
