static constexpr const char *tileImplementations = R"BANG(
#include <bang.h>
#include <bang_math_decls.h>
#include <cnrt.h>
#include <stdint.h>
#include <math.h>

__mlu_func__ void intent_sync_local() {
  __sync_compute();
  __sync_move();
}

__mlu_func__ float intent_bf16_to_float(uint16_t value) {
  __bang_bfloat16_raw raw;
  raw.x = value;
  return __bfloat162float__(bfloat16_t(raw));
}
__mlu_func__ uint16_t intent_float_to_bf16(float value) {
  __bang_bfloat16_raw raw = __float2bfloat16__(value);
  return raw.x;
}
__mlu_func__ uint64_t intent_round_shift(uint64_t value, int shift) {
  if (shift <= 0) return value << (-shift);
  if (shift > 64) return 0;
  if (shift == 64) return value > (uint64_t(1) << 63) ? 1 : 0;
  const uint64_t quotient = value >> shift;
  const uint64_t remainder = value & ((uint64_t(1) << shift) - 1);
  const uint64_t halfway = uint64_t(1) << (shift - 1);
  return quotient + (remainder > halfway || (remainder == halfway && (quotient & 1)));
}
template<class To, class From>
__mlu_func__ To intent_scalar_bitcast(From value) {
  static_assert(sizeof(To) == sizeof(From), "bitcast preserves storage bits");
  union { From from; To to; } bits;
  bits.from = value;
  return bits.to;
}
template<int E, int M, int Bias, bool Finite>
__mlu_func__ float intent_fp8_to_float(uint8_t value) {
  const unsigned fraction = value & ((1u << M) - 1);
  const unsigned exponent = (value >> M) & ((1u << E) - 1);
  float magnitude;
  if (exponent == ((1u << E) - 1) && (!Finite || fraction == ((1u << M) - 1)))
    magnitude = Finite || fraction ? NAN : INFINITY;
  else if (!exponent)
    magnitude = ldexpf(float(fraction), 1 - Bias - M);
  else
    magnitude = ldexpf(float((1u << M) | fraction), int(exponent) - Bias - M);
  return value & 128 ? -magnitude : magnitude;
}
template<int E, int M, int Bias, bool Finite>
__mlu_func__ uint8_t intent_number_to_fp8(double value) {
  const uint64_t bits = intent_scalar_bitcast<uint64_t>(value);
  const uint8_t sign = uint8_t(bits >> 56) & 128;
  const unsigned exponent = unsigned((bits >> 52) & 2047);
  const uint64_t fraction = bits & ((uint64_t(1) << 52) - 1);
  const uint8_t infinity = uint8_t(((1u << E) - 1) << M);
  if (exponent == 2047) return sign | (Finite || fraction ? 127 : infinity);
  if (!exponent) return sign;
  int e = int(exponent) - 1023;
  const uint64_t significand = (uint64_t(1) << 52) | fraction;
  if (e < 1 - Bias)
    return sign | uint8_t(intent_round_shift(significand, 53 - Bias - M - e));
  if (e > int((1u << E) - 1) - Bias) return sign | (Finite ? 127 : infinity);
  uint64_t rounded = intent_round_shift(significand, 52 - M);
  if (rounded == (uint64_t(1) << (M + 1))) { rounded >>= 1; ++e; }
  const unsigned encoded = unsigned(e + Bias) * (1u << M) + unsigned(rounded - (1u << M));
  if (encoded >= (Finite ? 127u : unsigned(infinity)))
    return sign | (Finite ? 127 : infinity);
  return sign | uint8_t(encoded);
}
__mlu_func__ uint16_t intent_number_to_bf16(double value) {
  union { double value; uint64_t bits; } converted;
  converted.value = value;
  const uint16_t sign = uint16_t(converted.bits >> 48) & 0x8000u;
  const unsigned exponent = unsigned((converted.bits >> 52) & 0x7ffu);
  const uint64_t fraction = converted.bits & ((uint64_t(1) << 52) - 1);
  if (exponent == 0x7ffu) return fraction ? 0x7fffu : (sign | 0x7f80u);
  if (exponent == 0) return sign;
  int e = int(exponent) - 1023;
  const uint64_t significand = (uint64_t(1) << 52) | fraction;
  if (e < -126) return sign | uint16_t(intent_round_shift(significand, -e - 81));
  if (e > 127) return sign | 0x7f80u;
  uint64_t rounded = intent_round_shift(significand, 45);
  if (rounded == 256) { rounded = 128; ++e; }
  if (e > 127) return sign | 0x7f80u;
  return sign | uint16_t((e + 127) << 7) | uint16_t(rounded - 128);
}
__mlu_func__ uint16_t intent_number_to_bf16(uint64_t value) {
  if (!value) return 0;
  int exponent = 0;
  for (uint64_t remaining = value; remaining > 1; remaining >>= 1) ++exponent;
  uint64_t rounded = intent_round_shift(value, exponent - 7);
  if (rounded == 256) { rounded = 128; ++exponent; }
  return uint16_t((exponent + 127) << 7) | uint16_t(rounded - 128);
}
__mlu_func__ uint16_t intent_number_to_bf16(int64_t value) {
  const uint64_t magnitude = value < 0 ? uint64_t(0) - uint64_t(value) : uint64_t(value);
  return (value < 0 ? 0x8000u : 0) | intent_number_to_bf16(magnitude);
}
__mlu_func__ uint16_t intent_number_to_bf16(float value) { return intent_float_to_bf16(value); }
__mlu_func__ uint16_t intent_number_to_bf16(half value) { return intent_float_to_bf16(float(value)); }
__mlu_func__ uint16_t intent_number_to_bf16(int32_t value) { return intent_number_to_bf16(int64_t(value)); }
__mlu_func__ uint16_t intent_number_to_bf16(uint32_t value) { return intent_number_to_bf16(uint64_t(value)); }
__mlu_func__ uint16_t intent_number_to_bf16(bool value) { return value ? 0x3f80u : 0; }
template<class From, class To> struct IntentLocalCast {
  __mlu_func__ static To apply(From value) { return static_cast<To>(value); }
};
template<class To> struct IntentLocalCast<uint16_t, To> {
  __mlu_func__ static To apply(uint16_t value) { return static_cast<To>(intent_bf16_to_float(value)); }
};
template<class From> struct IntentLocalCast<From, uint16_t> {
  __mlu_func__ static uint16_t apply(From value) { return intent_number_to_bf16(value); }
};

template<class T> __mlu_func__ T intent_read_local(const T *p) { return __load_nram(p); }
template<class T> __mlu_func__ void intent_write_local(T *p, T v) { __store_nram(p, v); }
template<> __mlu_func__ bool intent_read_local(const bool *p) {
  return __load_nram(reinterpret_cast<const uint8_t *>(p)) != 0;
}
template<> __mlu_func__ void intent_write_local(bool *p, bool v) {
  __store_nram(reinterpret_cast<uint8_t *>(p), uint8_t(v));
}
template<class T, int N>
__mlu_func__ void intent_copy_local(T *output, const T *input) {
  if (output == input) return;
  if (N == 1) intent_write_local(output, intent_read_local(input));
  else if (N > 1) __memcpy(output, input, N * sizeof(T), NRAM2NRAM);
}
template<class From, class To, int N>
__mlu_func__ void intent_cast_local(To *output, const From *input) {
  for (int i = 0; i < N; ++i)
    intent_write_local(output + i, IntentLocalCast<From, To>::apply(intent_read_local(input + i)));
}
template<int N>
__mlu_func__ void intent_bool_to_f32_tile(float *output, const bool *input) {
  const int prefix = N / 64 * 64;
  if (prefix) {
    __bang_uint82float(output, reinterpret_cast<const uint8_t *>(input), prefix, 0);
    __sync_compute();
    __bang_ne_scalar(output, output, 0.0f, prefix);
    __sync_compute();
  }
  for (int i = prefix; i < N; ++i)
    intent_write_local(output + i, float(intent_read_local(input + i)));
}
template<int N> __mlu_func__ void intent_bf16_to_f32_tile(float *output, const uint16_t *input) {
  __bang_bfloat162float(output, reinterpret_cast<const bfloat16_t *>(input), N);
}
template<int N> __mlu_func__ void intent_f32_to_bf16_tile(uint16_t *output, const float *input) {
  __bang_float2bfloat16_rn(reinterpret_cast<bfloat16_t *>(output), input, N);
}
__mlu_func__ float intent_f32_bits(uint32_t bits) {
  union { uint32_t bits; float value; } result;
  result.bits = bits;
  return result.value;
}
__mlu_func__ void intent_exp2_ftz_range(float *output, const float *input, int32_t *scratch, int count);
template<int N, int W, bool BaseTwo = false>
__mlu_func__ void intent_exp_f32_tile(float *output, const float *input, float *workspace) {
  float *hi = workspace, *lo = hi + W, *temporary = lo + W, *scale = temporary + W;
  auto *hi_bits = reinterpret_cast<uint32_t *>(hi);
  auto *lo_bits = reinterpret_cast<uint32_t *>(lo);
  auto *temporary_bits = reinterpret_cast<uint32_t *>(temporary);
  const float ln2_lo = intent_f32_bits(0x3805fdf4u);
  for (int first = 0; first < N; first += W) {
    const int count = N - first < W ? N - first : W;
    float *out = output + first;
    const float *x = input + first;
    const auto *x_bits = reinterpret_cast<const uint32_t *>(x);
    if (count < 4) {
      if constexpr (BaseTwo) intent_exp2_ftz_range(out, x, reinterpret_cast<int32_t *>(hi), count);
      else __cn_vector_exp_f32(count, out, x);
      __sync_compute(); continue;
    }
    // Keep the near-underflow/overflow interval in the SDK implementation.
    // Nonfinite lanes are repaired bitwise after the independent arithmetic.
    asm volatile("fuse.nram.u32 [%[dst]], %[count], [%[src]], .and(0x7fffffff), .lt(%[limit]), .mul(0x3f800000);"
        :: [dst]"r"(temporary), [src]"r"(x), [count]"r"(count),
           [limit]"r"(BaseTwo ? 0x42f00001u : 0x42a00001u));
    __bang_argmin(hi, temporary, count);
    __sync();
    const bool finite = intent_read_local(hi) == 1.0f;
    if constexpr (BaseTwo) {
      if (!finite) {
        intent_exp2_ftz_range(out, x, reinterpret_cast<int32_t *>(hi), count);
        continue;
      }
    }
    if (!finite) {
      __bang_band_scalar(hi_bits, x_bits, 0x7fffffffu, count);
      __bang_ge_scalar(hi_bits, hi_bits, 0x7f800000u, count);
      __bang_mul_scalar(hi_bits, hi_bits, 0x3f800000u, count);
      __bang_maximum(temporary, temporary, hi, count);
      __bang_argmin(hi, temporary, count);
      __sync();
      if (intent_read_local(hi) != 1.0f) {
        if constexpr (BaseTwo) intent_exp2_ftz_range(out, x, reinterpret_cast<int32_t *>(hi), count);
        else __cn_vector_exp_f32(count, out, x);
        __sync_compute();
        continue;
      }
    }
    // Reduce by a split ln(2). With |n|<=116, n*ln2_hi is exact, and x minus
    // that product is exact by Sterbenz (or n=0). The natural residual error
    // is below 1.52e-8. On |r| < a=ln(2)/2+1e-5, replace the seventh/eighth
    // Taylor terms by their degree-six Chebyshev projections. Keep c0=c1=1;
    // the omitted low terms, T7/T8 residuals, coefficient rounding and Taylor
    // remainder total <1.69e-8. Including independent RN operations and range
    // reduction gives <5.60e-8 before final rounding on r<0, below one output
    // ULP (2^-24); the positive side has more margin. Including reference and
    // final rounding stays within two output ULP. Scaling is exact.
    const float *finite_x = x;
    if (!finite) {
      asm volatile("fuse.nram.u32 [%[dst]], %[count], [%[src]], .and(0x7fffffff), .lt(0x7f800000), .mul(0xffffffff), .and([%[src]]);"
          :: [dst]"r"(hi), [src]"r"(x), [count]"r"(count));
      finite_x = hi;
    }
    // The biased sum has unit spacing for |n| <= 116. Its integer encoding
    // simultaneously supplies n and the exact power-of-two scale.
    if constexpr (BaseTwo) __bang_add_scalar(temporary, finite_x, 0x1.8p23f, count);
    else asm volatile("fuse.nram.f32 [%[dst]], %[count], [%[src]], .mul(0x3fb8aa3b), .add(0x4b400000);"
        :: [dst]"r"(temporary), [src]"r"(finite_x), [count]"r"(count));
    __bang_sub_scalar(out, temporary, 0x1.8p23f, count);
    asm volatile("fuse.nram.u32 [%[dst]], %[count], [%[src]], .sub(0x4b3fff81), .sll(23);"
        :: [dst]"r"(scale), [src]"r"(temporary), [count]"r"(count));
    if constexpr (BaseTwo) {
      // x-n is exact. Multiplication by RN(ln2) contributes <1.59e-8,
      // retaining the six-degree polynomial's two-ULP bound. Subnormal x
      // produces exactly one, as required by input FTZ; |x|<=120 keeps every
      // output normal, so no output FTZ pass is needed on this branch.
      asm volatile("fuse.nram.f32 [%[dst]], %[count], [%[x]], .sub([%[n]]), .mul(0x3f317218);"
          :: [dst]"r"(out), [x]"r"(finite_x), [n]"r"(out), [count]"r"(count));
    } else {
      __bang_mul_scalar(lo, out, ln2_lo, count);
      asm volatile("fuse.nram.f32 [%[dst]], %[count], [%[n]], .mul(0x3f317000), .sub([%[x]]), .add([%[lo]]), .mul(0xbf800000);"
          :: [dst]"r"(out), [n]"r"(out), [x]"r"(finite_x), [lo]"r"(lo), [count]"r"(count));
    }
    // c2=1/2+a^6/161280, c3=1/6-a^4/5760, c4=1/24-a^4/32256,
    // c5=1/120+a^2/2880, c6=1/720+a^2/20160, each rounded to f32.
    asm volatile("fuse.nram.f32 [%[dst]], %[count], [%[r]], .mul(0x3ab6d34e), .add(0x3c093779), .mul([%[r]]), .add(0x3d2aaa33);"
        :: [dst]"r"(lo), [r]"r"(out), [count]"r"(count));
    asm volatile("fuse.nram.f32 [%[dst]], %[count], [%[dst]], .mul([%[r]]), .add(0x3e2aaa03), .mul([%[r]]), .add(0x3f000000);"
        :: [dst]"r"(lo), [r]"r"(out), [count]"r"(count));
    asm volatile("fuse.nram.f32 [%[dst]], %[count], [%[p]], .mul([%[r]]), .mul([%[r]]), .add([%[r]]), .add(0x3f800000);"
        :: [dst]"r"(temporary), [p]"r"(lo), [r]"r"(out), [count]"r"(count));
    __bang_mul(out, temporary, scale, count);
    if (!finite) {
      asm volatile("fuse.nram.u32 [%[dst]], %[count], [%[src]], .and(0x7fffffff), .lt(0x7f800000), .xor(1), .mul(0xffffffff);"
          :: [dst]"r"(hi), [src]"r"(x), [count]"r"(count));
      asm volatile("fuse.nram.u32 [%[dst]], %[count], [%[src]], .eq(0xff800000), .xor(1), .mul(0xffffffff), .and([%[src]]);"
          :: [dst]"r"(lo), [src]"r"(x), [count]"r"(count));
      asm volatile("fuse.nram.u32 [%[dst]], %[count], [%[dst]], .xor([%[special]]), .and([%[mask]]), .xor([%[dst]]);"
          :: [dst]"r"(out), [special]"r"(lo), [mask]"r"(hi), [count]"r"(count));
    }
    __sync_compute();
  }
}
__mlu_func__ float intent_flush_subnormal(float value) {
  union { float value; uint32_t bits; } converted;
  converted.value = value;
  if ((converted.bits & 0x7f800000u) == 0) converted.bits &= 0x80000000u;
  return converted.value;
}
__mlu_func__ float intent_exp2_subnormal(float value) {
  union { float value; uint32_t bits; } converted;
  converted.bits = 0;
  if (value < -150.0f) return converted.value;
  // The scaled result is normal. Round its mantissa directly into the f32
  // encoding, so a device's arithmetic FTZ mode cannot erase subnormal output.
  // compute_30 scalar exp2 has a documented 1-ULP bound; the final integer
  // rounding stays within the source approximate-exp2 contract's 2 ULP.
  const float scaled = exp2f(value + 149.0f);
  uint32_t mantissa = static_cast<uint32_t>(scaled);
  const float remainder = scaled - float(mantissa);
  if (remainder > 0.5f || (remainder == 0.5f && (mantissa & 1u))) ++mantissa;
  converted.bits = mantissa;
  return converted.value;
}
template<int N, bool Flush>
__mlu_func__ void intent_exp2_tile(float *output, const float *input) {
  if (Flush) {
    for (int i = 0; i < N; ++i) intent_write_local(output + i, intent_flush_subnormal(intent_read_local(input + i)));
    __sync();
    __cn_vector_exp2_f32(N, output, output);
  } else {
    __cn_vector_exp2_f32(N, output, input);
  }
  __sync_compute();
  for (int i = 0; i < N; ++i) {
    float value = intent_read_local(output + i);
    if (Flush) value = intent_flush_subnormal(value);
    else {
      const float x = intent_read_local(input + i);
      if (x < -126.0f) value = intent_exp2_subnormal(x);
    }
    intent_write_local(output + i, value);
  }
}
template<class T, int N>
__mlu_func__ void intent_fill_local(T *output, T value) {
  if (N == 0) return;
  if (sizeof(T) == 1 && N % 64 == 0) {
    __bang_write_value(reinterpret_cast<unsigned char *>(output), N,
                       static_cast<unsigned char>(value));
    return;
  }
  if (sizeof(T) == 4 && N % 32 == 0) {
    union { T value; uint32_t bits; } word;
    word.value = value;
    __bang_write_value(reinterpret_cast<uint32_t *>(output), N, word.bits);
    return;
  }
  if (sizeof(T) == 8 && N % 32 == 0) {
    union { T value; uint64_t bits; } word;
    word.value = value;
    const uint32_t low = uint32_t(word.bits), high = uint32_t(word.bits >> 32);
    const int lanes = __vv_get_length() / sizeof(uint32_t);
    asm volatile("vv.begin;" ::: "memory");
    vv_uint32 pattern;
    __vv_index(pattern, 0u, 1u);
    __vv_and(pattern, pattern, 1u);
    __vv_mul(pattern, pattern, high - low);
    __vv_add(pattern, pattern, low);
    for (int first = 0; first < 2 * N; first += lanes)
      __vv_store(reinterpret_cast<uint32_t *>(output) + first, pattern);
    asm volatile("vv.end;" ::: "memory");
    return;
  }
  intent_write_local(output, value);
  if (N == 1) return;
  __sync();
  for (int first = 1; first < N; first += 65536) {
    const int take = N - first < 65536 ? N - first : 65536;
    __memcpy(output + first, output, sizeof(T), NRAM2NRAM,
             sizeof(T), 0, take - 1);
  }
  __sync_move();
}
template<int N>
__mlu_func__ void intent_iota_local(float *output) {
  const int lanes = __vv_get_length() / sizeof(float);
  asm volatile("vv.begin;" ::: "memory");
  for (int first = 0; first < N; first += lanes) {
    const unsigned take = N - first < lanes ? N - first : lanes;
    vv_float values;
    __vv_index(values, float(first), 1.0f);
    if (take == lanes) __vv_store(output + first, values);
    else __vv_store(output + first, values, take);
  }
  asm volatile("vv.end;" ::: "memory");
}
template<int N>
__mlu_func__ void intent_iota_local(int64_t *output) {
  const int lanes = __vv_get_length() / sizeof(uint32_t);
  asm volatile("vv.begin;" ::: "memory");
  for (int first = 0; first < 2 * N; first += lanes) {
    const unsigned take = 2 * N - first < lanes ? 2 * N - first : lanes;
    vv_uint32 words, low, mask;
    __vv_index(words, unsigned(first), 1u);
    __vv_srl(low, words, 1u);
    __vv_and(mask, words, 1u); __vv_eq(mask, mask, 0u);
    __vv_mul(low, low, mask);
    if (take == lanes) __vv_store(reinterpret_cast<uint32_t *>(output) + first, low);
    else __vv_store(reinterpret_cast<uint32_t *>(output) + first, low, take);
  }
  asm volatile("vv.end;" ::: "memory");
}
template<int N>
__mlu_func__ void intent_index_add(uint32_t *out, const uint32_t *a, const uint32_t *b) {
  __bang_add(out, a, b, N);
  asm volatile("fuse.nram.u32 [%[hi]], %[n], [%[lo]], .lt([%[a]]);"
      :: [hi]"r"(out + N), [n]"r"(N), [lo]"r"(out), [a]"r"(a));
  asm volatile("fuse.nram.u32 [%[hi]], %[n], [%[hi]], .add([%[ah]]), .add([%[bh]]);"
      :: [hi]"r"(out + N), [n]"r"(N), [ah]"r"(a + N), [bh]"r"(b + N));
}
template<int N>
__mlu_func__ void intent_index_add(uint32_t *out, const uint32_t *a, int64_t b) {
  const uint32_t low = uint32_t(b), high = uint32_t(uint64_t(b) >> 32);
  __bang_add_scalar(out, a, low, N);
  asm volatile("fuse.nram.u32 [%[hi]], %[n], [%[lo]], .lt([%[a]]);"
      :: [hi]"r"(out + N), [n]"r"(N), [lo]"r"(out), [a]"r"(a));
  asm volatile("fuse.nram.u32 [%[hi]], %[n], [%[hi]], .add([%[ah]]), .add(%[bh]);"
      :: [hi]"r"(out + N), [n]"r"(N), [ah]"r"(a + N), [bh]"r"(high));
}
template<int N>
__mlu_func__ void intent_index_sub(uint32_t *out, const uint32_t *a, const uint32_t *b) {
  __bang_sub(out, a, b, N);
  asm volatile("fuse.nram.u32 [%[hi]], %[n], [%[a]], .lt([%[b]]);"
      :: [hi]"r"(out + N), [n]"r"(N), [a]"r"(a), [b]"r"(b));
  asm volatile("fuse.nram.u32 [%[hi]], %[n], [%[ah]], .sub([%[bh]]), .sub([%[hi]]);"
      :: [hi]"r"(out + N), [n]"r"(N), [ah]"r"(a + N), [bh]"r"(b + N));
}
template<int N>
__mlu_func__ void intent_index_sub(uint32_t *out, const uint32_t *a, int64_t b) {
  const uint32_t low = uint32_t(b), high = uint32_t(uint64_t(b) >> 32);
  __bang_sub_scalar(out, a, low, N);
  asm volatile("fuse.nram.u32 [%[hi]], %[n], [%[a]], .lt(%[b]);"
      :: [hi]"r"(out + N), [n]"r"(N), [a]"r"(a), [b]"r"(low));
  asm volatile("fuse.nram.u32 [%[hi]], %[n], [%[ah]], .sub(%[bh]), .sub([%[hi]]);"
      :: [hi]"r"(out + N), [n]"r"(N), [ah]"r"(a + N), [bh]"r"(high));
}
template<int N>
__mlu_func__ void intent_index_mul(uint32_t *out, const uint32_t *a, const uint32_t *b) {
  __bang_mul(out, a, b, N);
  __bang_mulh(out + N, a, b, N);
  asm volatile("fuse.nram.u32 [%[hi]], %[n], [%[ah]], .mul([%[b]]), .add([%[hi]]);"
      :: [hi]"r"(out + N), [n]"r"(N), [ah]"r"(a + N), [b]"r"(b));
  asm volatile("fuse.nram.u32 [%[hi]], %[n], [%[a]], .mul([%[bh]]), .add([%[hi]]);"
      :: [hi]"r"(out + N), [n]"r"(N), [a]"r"(a), [bh]"r"(b + N));
}
template<int N>
__mlu_func__ void intent_index_mul(uint32_t *out, const uint32_t *a, int64_t b) {
  const uint32_t low = uint32_t(b), high = uint32_t(uint64_t(b) >> 32);
  __bang_mul_scalar(out, a, low, N);
  __bang_mulh_scalar(out + N, a, low, N);
  asm volatile("fuse.nram.u32 [%[hi]], %[n], [%[ah]], .mul(%[b]), .add([%[hi]]);"
      :: [hi]"r"(out + N), [n]"r"(N), [ah]"r"(a + N), [b]"r"(low));
  asm volatile("fuse.nram.u32 [%[hi]], %[n], [%[a]], .mul(%[bh]), .add([%[hi]]);"
      :: [hi]"r"(out + N), [n]"r"(N), [a]"r"(a), [bh]"r"(high));
}
template<int N, int S>
__mlu_func__ void intent_index_shl(uint32_t *out, const uint32_t *a) {
  if constexpr (S == 0) __bang_add_scalar(out, a, uint32_t(0), 2 * N);
  else if constexpr (S < 32) {
    __bang_srl(out, a, 32 - S, N);
    asm volatile("fuse.nram.u32 [%[hi]], %[n], [%[ah]], .sll(%[s]), .or([%[lo]]);"
        :: [hi]"r"(out + N), [n]"r"(N), [ah]"r"(a + N), [s]"r"(S), [lo]"r"(out));
    __bang_sll(reinterpret_cast<int32_t *>(out), reinterpret_cast<const int32_t *>(a), S, N);
  } else {
    __bang_write_value(out, N, uint32_t(0));
    __bang_sll(reinterpret_cast<int32_t *>(out + N), reinterpret_cast<const int32_t *>(a), S - 32, N);
  }
}
template<int N, int S>
__mlu_func__ void intent_index_sar(uint32_t *out, const uint32_t *a) {
  if constexpr (S == 0) __bang_add_scalar(out, a, uint32_t(0), 2 * N);
  else if constexpr (S < 32) {
    __bang_sll(reinterpret_cast<int32_t *>(out), reinterpret_cast<const int32_t *>(a + N), 32 - S, N);
    asm volatile("fuse.nram.u32 [%[lo]], %[n], [%[a]], .srl(%[s]), .or([%[lo]]);"
        :: [lo]"r"(out), [n]"r"(N), [a]"r"(a), [s]"r"(S));
    __bang_sra(reinterpret_cast<int32_t *>(out + N), reinterpret_cast<const int32_t *>(a + N), S, N);
  } else {
    __bang_sra(reinterpret_cast<int32_t *>(out), reinterpret_cast<const int32_t *>(a + N), S - 32, N);
    __bang_sra(reinterpret_cast<int32_t *>(out + N), reinterpret_cast<const int32_t *>(a + N), 31, N);
  }
}
template<int R, int C>
__mlu_func__ void intent_compare_range(bool *output, const int64_t *coordinates,
                                     int64_t rows, int64_t base) {
  for (int row = 0; row < R; ++row) {
    int prefix = C;
    if (row < rows) {
      const int64_t query = intent_read_local(coordinates + row);
      prefix = query < base ? 0 : query >= base + C - 1 ? C : int(query - base) + 1;
    }
    auto *destination = reinterpret_cast<unsigned char *>(output) + row * C;
    if (prefix) __bang_write_value(destination, prefix, static_cast<unsigned char>(1));
    if (prefix < C) __bang_write_value(destination + prefix, C - prefix, static_cast<unsigned char>(0));
  }
}
template<int R, int C, class Mask>
__mlu_func__ void intent_compare_ramp(Mask *output, float *scratch, int64_t row_begin, int64_t base) {
  // Clamp an unsigned magnitude before converting to a bounded signed offset;
  // subtraction of extreme signed coordinates never overflows here.
  int delta;
  if (row_begin >= base) {
    uint64_t distance = uint64_t(row_begin) - uint64_t(base);
    delta = distance >= uint64_t(C) ? C : int(distance);
  } else {
    uint64_t distance = uint64_t(base) - uint64_t(row_begin);
    delta = distance >= uint64_t(R) ? -R : -int(distance);
  }
  if (delta >= C - 1 || delta <= -R) {
    if constexpr (sizeof(Mask) == 4) __bang_write_value(output, R * C, delta >= C - 1 ? -1 : 0);
    else __bang_write_value(reinterpret_cast<uint8_t *>(output), R * C, uint8_t(delta >= C - 1));
    return;
  }
  constexpr int shift = __builtin_ctz(C);
  const int lanes = __vv_get_length() / sizeof(int32_t);
  asm volatile("vv.begin;" ::: "memory");
  for (int first = 0; first < R * C; first += lanes) {
    vv_int32 index, row, column, mask;
    __vv_index(index, first, 1);
    __vv_sra(row, index, shift);
    __vv_and(column, index, C - 1);
    __vv_add(row, row, delta);
    __vv_ge(mask, row, column);
    __vv_mul(mask, mask, sizeof(Mask) == 4 ? -1 : 0x3f800000);
    const unsigned take = R * C - first < lanes ? R * C - first : lanes;
    auto *destination = sizeof(Mask) == 4 ? reinterpret_cast<int32_t *>(output) : reinterpret_cast<int32_t *>(scratch);
    if (take == lanes) __vv_store(destination + first, mask);
    else __vv_store(destination + first, mask, take);
  }
  asm volatile("vv.end;" ::: "memory");
  if constexpr (sizeof(Mask) == 1) {
    __sync_compute();
    __bang_float2uint8_rn(reinterpret_cast<uint8_t *>(output), scratch, R * C, 0);
  }
}
template<int N>
__mlu_func__ void intent_masked_fill(float *output, const float *input, const int32_t *mask, float value) {
  union { float value; uint32_t bits; } fill;
  fill.value = value;
  if (fill.bits == 0) {
    __bang_band(reinterpret_cast<uint32_t *>(output), reinterpret_cast<const uint32_t *>(input),
                reinterpret_cast<const uint32_t *>(mask), N);
  } else {
    asm volatile("fuse.nram.u32 [%[dst]], %[n], [%[src]], .xor(%[fill]), .and([%[mask]]), .xor(%[fill]);"
        :: [dst]"r"(output), [src]"r"(input), [mask]"r"(mask), [fill]"r"(fill.bits), [n]"r"(N));
  }
}
template<int N, int Width, bool CheckPositiveNormal = false>
__mlu_func__ void intent_flush_subnormal_tile(float *output, const float *input, int32_t *scratch) {
  for (int first = 0; first < N; first += Width) {
    const int take = N - first < Width ? N - first : Width;
    if constexpr (CheckPositiveNormal) {
      if (output == input && take >= 4) {
        // exp2 produces nonnegative values. A normal-or-infinite tile needs
        // no bit repair; exceptional chunks retain the complete FTZ path.
        auto *accepted = reinterpret_cast<float *>(scratch);
        __bang_ge_scalar(accepted, input + first, 0x1p-126f, take);
        __bang_argmin(accepted, accepted, take);
        __sync_compute();
        if (intent_read_local(accepted) == 1.0f) continue;
      }
    }
    // This dependency chain stays on Compute; wait once before the workspace
    // is reused or the following math primitive consumes the completed tile.
    asm volatile("fuse.nram.u32 [%[dst]], %[count], [%[src]], .and(0x7f800000), .eq(0), .mul(0x007fffff), .xor(0xffffffff);"
        :: [dst]"r"(scratch), [src]"r"(input + first), [count]"r"(take));
    __bang_band(reinterpret_cast<unsigned int *>(output + first),
                reinterpret_cast<const unsigned int *>(input + first),
                reinterpret_cast<const unsigned int *>(scratch), take);
    __sync_compute();
  }
}
template<int N, int Width, bool InputNonSubnormal = false>
__mlu_func__ void intent_exp2_ftz_tile(float *output, const float *input, int32_t *scratch) {
  bool normal_output = false;
  if constexpr (InputNonSubnormal && N >= 4) {
    __sync_compute();
    // A non-NaN witness makes the native minimum well-defined. At x >= -125,
    // even the primitive's 1-ULP error cannot produce a subnormal result.
    // NaN lanes remain NaN and do not require output FTZ either.
    if (!isnan(intent_read_local(input))) {
      auto *minimum = reinterpret_cast<float *>(scratch);
      __bang_argmin(minimum, input, N);
      __sync_compute();
      normal_output = intent_read_local(minimum) >= -125.0f;
    }
  }
  if constexpr (InputNonSubnormal) __cn_vector_exp2_f32(N, output, input);
  else {
    intent_flush_subnormal_tile<N, Width>(output, input, scratch);
    __cn_vector_exp2_f32(N, output, output);
  }
  __sync_compute();
  if (!normal_output) intent_flush_subnormal_tile<N, Width, InputNonSubnormal>(output, output, scratch);
}
__mlu_func__ void intent_exp2_ftz_range(float *output, const float *input, int32_t *scratch, int count) {
  asm volatile("fuse.nram.u32 [%[dst]], %[count], [%[src]], .and(0x7f800000), .eq(0), .mul(0x007fffff), .xor(0xffffffff);"
      :: [dst]"r"(scratch), [src]"r"(input), [count]"r"(count));
  __bang_band(reinterpret_cast<uint32_t *>(output), reinterpret_cast<const uint32_t *>(input),
              reinterpret_cast<const uint32_t *>(scratch), count);
  __sync_compute();
  __cn_vector_exp2_f32(count, output, output);
  __sync_compute();
  asm volatile("fuse.nram.u32 [%[dst]], %[count], [%[src]], .and(0x7f800000), .eq(0), .mul(0x007fffff), .xor(0xffffffff);"
      :: [dst]"r"(scratch), [src]"r"(output), [count]"r"(count));
  __bang_band(reinterpret_cast<uint32_t *>(output), reinterpret_cast<const uint32_t *>(output),
              reinterpret_cast<const uint32_t *>(scratch), count);
  __sync_compute();
}
template<int N> __mlu_func__ void intent_zero_tile(float *output) { __bang_write_value(output, N, 0.0f); }
template<int N> __mlu_func__ void intent_zero_tile(half *output) { __bang_write_value(output, N, half(0)); }
template<int N, class T> __mlu_func__ void intent_zero_tile(T *output) { intent_fill_local<T, N>(output, T(0)); }
template<int N, int W>
__mlu_func__ void intent_reciprocal_ftz(float *output, const float *input, int32_t *workspace) {
  // compute_30 reciprocal has a 1-ULP contract. Restrict the bulk acceptance
  // interval so its result stays normal even with that error; uncommon inputs
  // follow the explicit approximate-fdiv and FTZ classifications below.
  __cn_vector_rcp_f32_rn(N, output, input);
  float *accepted = reinterpret_cast<float *>(workspace);
  float *minimum = accepted + W;
  for (int first = 0; first < N; first += W) {
    const int count = N - first < W ? N - first : W;
    if (count >= 4) {
      asm volatile("fuse.nram.u32 [%[dst]], %[count], [%[src]], .and(0x7fffffff), .sub(0x00800000), .lt(0x7d800001), .mul(0x3f800000);"
          :: [dst]"r"(accepted), [src]"r"(input + first), [count]"r"(count));
      __bang_argmin(minimum, accepted, count);
      intent_sync_local();
      if (intent_read_local(minimum) == 1.0f) continue;
    } else __sync_compute();
    for (int i = first; i < first + count; ++i) {
      const uint32_t bits = intent_read_local(reinterpret_cast<const uint32_t *>(input + i));
      const uint32_t magnitude = bits & 0x7fffffffu, sign = bits & 0x80000000u;
      if (magnitude >= 0x00800000u && magnitude <= 0x7e000000u) continue;
      uint32_t result;
      if (magnitude > 0x7f800000u) result = 0x7fc00000u;
      else if (magnitude < 0x00800000u) result = sign | 0x7f800000u;
      else if (magnitude > 0x7e800000u) result = sign;
      else {
        result = intent_read_local(reinterpret_cast<const uint32_t *>(output + i));
        if (!(result & 0x7f800000u)) result &= 0x80000000u;
      }
      intent_write_local(reinterpret_cast<uint32_t *>(output + i), result);
    }
  }
}
template<class T, int R, int C, int Kind>
__mlu_func__ void intent_binary_rows(T *output, const T *lhs, const T *rhs) {
  for (int row = 0; row < R; ++row) {
    T value = intent_read_local(rhs + row);
    if (Kind == 0) __bang_add_scalar(output + row * C, lhs + row * C, value, C);
    if (Kind == 1) __bang_sub_scalar(output + row * C, lhs + row * C, value, C);
    if (Kind == 2) __bang_mul_scalar(output + row * C, lhs + row * C, value, C);
  }
}
template<class T, int N>
__mlu_func__ void intent_divide_local(T *output, const T *lhs, const T *rhs) {
  // MLU370 __bang_div is an abs/reciprocal/sign/multiply sequence. Use scalar
  // division here to retain signed-zero, infinity and NaN behavior.
  for (int i = 0; i < N; ++i)
    intent_write_local(output + i, T(intent_read_local(lhs + i) / intent_read_local(rhs + i)));
}
template<int N, bool Approximate, bool Flush>
__mlu_func__ void intent_divide_f32(float *output, const float *lhs, const float *rhs) {
  if (!Approximate && !Flush) {
    // compute_30 implements this with the same IEEE fdiv as scalar div.rn.
    __cn_vector_div_f32_rn(N, output, lhs, rhs);
    return;
  }
  for (int i = 0; i < N; ++i) {
    float a = intent_read_local(lhs + i), b = intent_read_local(rhs + i);
    if (Flush) { a = intent_flush_subnormal(a); b = intent_flush_subnormal(b); }
    union { float value; uint32_t bits; } left, right, result;
    left.value = a; right.value = b;
    const uint32_t sign = (left.bits ^ right.bits) & 0x80000000u;
    const uint32_t aa = left.bits & 0x7fffffffu, bb = right.bits & 0x7fffffffu;
    if (aa > 0x7f800000u || bb > 0x7f800000u) result.bits = 0x7fc00000u;
    else if (Approximate && bb > 0x7e800000u) result.bits = aa == 0x7f800000u ? 0x7fc00000u : sign;
    else if (bb == 0) result.bits = aa == 0 ? 0x7fc00000u : sign | 0x7f800000u;
    else if (aa == 0x7f800000u) result.bits = bb == 0x7f800000u ? 0x7fc00000u : sign | 0x7f800000u;
    else if (bb == 0x7f800000u || aa == 0) result.bits = sign;
    else result.value = __cn_scalar_div_f32_rn(a, b);
    // compute_30 div.rn is documented as RNE with no error. Approximate mode
    // additionally owns the large-denominator rule and the local FTZ boundary.
    intent_write_local(output + i, Flush ? intent_flush_subnormal(result.value) : result.value);
  }
}
__mlu_func__ float intent_divisor_lane(const float *rhs, int i) { return intent_read_local(rhs + i); }
__mlu_func__ float intent_divisor_lane(float rhs, int) { return rhs; }
template<int N>
__mlu_func__ void intent_divisor_magnitude(float *output, const float *rhs) { __bang_abs(output, rhs, N); }
template<int N>
__mlu_func__ void intent_divisor_magnitude(float *output, float rhs) { __bang_write_value(output, N, fabsf(rhs)); }
__mlu_func__ int intent_division_vector_guards(const float *) { return 2; }
// Once the scalar divisor/reciprocal and the candidate range are certified,
// the RN product bound also covers a subnormal numerator without FTZ.
__mlu_func__ int intent_division_vector_guards(float) { return 0; }
__mlu_func__ unsigned intent_division_radius(const float *) { return 2; }
__mlu_func__ unsigned intent_division_radius(float) { return 1; }
template<int N>
__mlu_func__ void intent_division_normal_guard(float *accepted, const float *operand) {
  asm volatile("fuse.nram.u32 [%[dst]], %[count], [%[src]], .and(0x7fffffff), .sub(0x00800000), .lt(0x7f000000), .mul([%[dst]]);"
      :: [dst]"r"(accepted), [src]"r"(operand), [count]"r"(N));
}
template<int N>
__mlu_func__ void intent_division_normal_guard(float *, float) {}
template<int N>
__mlu_func__ bool intent_divide_candidate(float *output, const float *lhs, const float *rhs, float *) {
  __cn_vector_fast_div_f32_rn(N, output, lhs, rhs);
  return true;
}
template<int N>
__mlu_func__ bool intent_divide_candidate(float *output, const float *lhs, float rhs, float *temporary) {
  // The uniform divisor is checked once, before any per-lane certification.
  union { float value; uint32_t bits; } divisor;
  divisor.value = rhs;
  const uint32_t divisor_exponent = divisor.bits & 0x7f800000u;
  if (!divisor_exponent || divisor_exponent == 0x7f800000u) return false;
  union { float value; uint32_t bits; } reciprocal;
  intent_write_local(temporary, rhs);
  __sync();
  asm volatile("recip.nram.f32 [%[dst]], [%[src]], %[count];"
      :: [dst]"r"(temporary + 1), [src]"r"(temporary), [count]"r"(1));
  __sync_compute();
  reciprocal.value = intent_read_local(temporary + 1);
  const uint32_t exponent = reciprocal.bits & 0x7f800000u;
  if (!exponent || exponent == 0x7f800000u) return false;
  uint32_t q = reciprocal.bits & 0x7fffffffu;
  const uint32_t fraction = q & 0x7fffffu;
  const int shift = 278 - int((divisor.bits >> 23) & 255u) - int(q >> 23);
  bool certified = false;
  if (fraction >= 2 && fraction < 0x7ffffeu && (shift == 24 || shift == 25)) {
    const uint64_t left = uint64_t(0x800000u) << shift;
    const uint32_t my = (divisor.bits & 0x7fffffu) | 0x800000u;
    uint32_t mq = (q & 0x7fffffu) | 0x800000u;
    uint64_t lower = uint64_t(my) * (2u * mq - 1u), upper = uint64_t(my) * (2u * mq + 1u);
    if (left > upper || (left == upper && (q & 1u))) ++q;
    else if (left < lower || (left == lower && (q & 1u))) --q;
    mq = (q & 0x7fffffu) | 0x800000u;
    lower = uint64_t(my) * (2u * mq - 1u); upper = uint64_t(my) * (2u * mq + 1u);
    certified = (left > lower || (left == lower && !(q & 1u))) &&
                (left < upper || (left == upper && !(q & 1u)));
  }
  if (certified) reciprocal.bits = q | (divisor.bits & 0x80000000u);
  else reciprocal.value = __cn_scalar_div_f32_rn(1.0f, rhs);
  asm volatile("mult.scalar.nram.rn.f32 [%[dst]], [%[src]], %[factor], %[count];"
      :: [dst]"r"(output), [src]"r"(lhs), [factor]"r"(reciprocal.bits), [count]"r"(N));
  return true;
}
template<int N>
__mlu_func__ void intent_strict_divide(float *output, const float *lhs, const float *rhs) {
  __cn_vector_div_f32_rn(N, output, lhs, rhs);
}
template<int N>
__mlu_func__ void intent_strict_divide(float *output, const float *lhs, float rhs) {
  for (int i = 0; i < N; ++i)
    intent_write_local(output + i, __cn_scalar_div_f32_rn(intent_read_local(lhs + i), rhs));
}
__mlu_func__ uint16_t intent_narrow_half_bits(uint32_t bits) {
  const uint16_t sign = uint16_t(bits >> 16) & 0x8000u;
  const uint32_t magnitude = bits & 0x7fffffffu;
  const unsigned exponent = magnitude >> 23;
  if (exponent > 142) return sign | (magnitude > 0x7f800000u ? 0x7e00u : 0x7c00u);
  if (exponent < 102) return sign;
  if (exponent >= 113)
    return sign | uint16_t(((magnitude + 0xfffu + ((magnitude >> 13) & 1u)) >> 13) - 0x1c000u);
  const uint32_t mantissa = (magnitude & 0x7fffffu) | 0x800000u;
  const unsigned shift = 126 - exponent;
  return sign | uint16_t((mantissa + (1u << (shift - 1)) - 1u + ((mantissa >> shift) & 1u)) >> shift);
}
__mlu_func__ uint32_t intent_divide_half_boundary(float lhs, float rhs, uint16_t lower, uint16_t upper) {
  union { float value; uint32_t bits; } x, y, midpoint;
  x.value = lhs;
  y.value = rhs;
  const unsigned xe = (x.bits >> 23) & 255u;
  const unsigned ye = (y.bits >> 23) & 255u;
  if (!xe || xe == 255 || !ye || ye == 255) return 0x10000u;
  const uint16_t sign = uint16_t((x.bits ^ y.bits) >> 16) & 0x8000u;
  const unsigned h = lower & 0x7fffu;
  if ((lower & 0x8000u) != sign || (upper & 0x8000u) != sign ||
      (upper & 0x7fffu) != h + 1 || h + 1 >= 0x7c00u) return 0x10000u;
  if (h >= 0x400u) midpoint.bits = ((h + 0x1c000u) << 13) + 0x1000u;
  else {
    const uint32_t odd = 2u * h + 1u;
    unsigned leading = 0, remaining = odd;
    if (remaining >= 256) { remaining >>= 8; leading += 8; }
    if (remaining >= 16) { remaining >>= 4; leading += 4; }
    if (remaining >= 4) { remaining >>= 2; leading += 2; }
    if (remaining >= 2) ++leading;
    midpoint.bits = ((leading + 102u) << 23) | ((odd << (23u - leading)) & 0x7fffffu);
  }
  const unsigned me = (midpoint.bits >> 23) & 255u;
  const int shift = int(xe) - int(ye) - int(me) + 151;
  if (shift < 0 || shift > 39) return 0x10000u;
  // The half midpoint has an even f32 significand. RN32 followed by RN16
  // moves the decision boundary by half an f32 ULP toward the odd half value.
  // Compare x/y with that boundary exactly using at most 49-bit products.
  const unsigned mx = (x.bits & 0x7fffffu) | 0x800000u;
  const unsigned my = (y.bits & 0x7fffffu) | 0x800000u;
  const unsigned mm = (midpoint.bits & 0x7fffffu) | 0x800000u;
  const unsigned threshold = (mm << 1) + 1 - ((h & 1u) << 1);
  const uint64_t left = uint64_t(mx) << shift;
  const uint64_t right = uint64_t(my) * threshold;
  const bool round_lower = left < right || (left == right && !(h & 1u));
  return sign | (h + (round_lower ? 0 : 1));
}
template<int N, class Divisor>
__mlu_func__ void intent_divide_cast_f16(half *output, const float *lhs, Divisor rhs,
    float *quotient, float *bounds, float *accepted, half *narrow_bounds, const float *lane_indices) {
  // The vector candidate's 1-ULP bound needs at most two adjacent encodings.
  // For a normal RN reciprocal r, |lhs*r-lhs/rhs| <= 2^-24*|lhs/rhs|:
  // less than one spacing inside a binade, and less than the 1.5 lower-spacing
  // distance between rounding boundaries across a power of two. Rounding both
  // products therefore differs by at most one encoding, including subnormal
  // lhs when the guarded quotient is normal. The native multiply is RN, no FTZ.
  // Every finite half rounding boundary is on a 4096-encoding grid in normal
  // f32 magnitudes (subnormal half boundaries form a subset of that grid).
  // An open grid cell therefore certifies a constant observable half result.
  if (!intent_divide_candidate<N>(quotient, lhs, rhs, bounds)) {
    intent_strict_divide<N>(quotient, lhs, rhs);
    __sync();
    __bang_float2half_rn(output, quotient, N);
    return;
  }
  __sync();
  const unsigned radius = intent_division_radius(rhs);
  // All certification operations below use the same Compute queue. Their
  // NRAM dependencies remain ordered until compaction returns to scalar code.
  __bang_float2half_rn(output, quotient, N);
  asm volatile("fuse.nram.u32 [%[dst]], %[count], [%[src]], .and(0xfff), .sub(%[start]), .lt(%[width]), .mul(0x3f800000);"
      :: [dst]"r"(accepted), [src]"r"(quotient), [count]"r"(N),
         [start]"r"(radius + 1), [width]"r"(4095u - 2u * radius));
  asm volatile("fuse.nram.u32 [%[dst]], %[count], [%[src]], .and(0x7fffffff), .sub(0x01000000), .lt(0x7d800001), .mul([%[dst]]);"
      :: [dst]"r"(accepted), [src]"r"(quotient), [count]"r"(N));
  if (intent_division_vector_guards(rhs)) {
    intent_division_normal_guard<N>(accepted, lhs);
    intent_division_normal_guard<N>(accepted, rhs);
  }
  // Compact the exceptional positions once. The compiler initializes the
  // exact f32 lane-index row once per program and keeps it across row calls.
  __bang_eq_scalar(accepted, accepted, 0.0f, N);
  const unsigned failures = __bang_filter(bounds, lane_indices, accepted, N);
  __sync();
  const unsigned repair_limit = N / 4 < 256 ? N / 4 : 256;
  if (failures <= repair_limit) {
    unsigned strict_count = 0;
    for (unsigned repaired = 0; repaired < failures; ++repaired) {
      const unsigned index = static_cast<unsigned>(intent_read_local(bounds + repaired));
      const float left = intent_read_local(lhs + index);
      const float right = intent_divisor_lane(rhs, index);
      union { float value; uint32_t bits; } candidate, lower, upper;
      candidate.value = intent_read_local(quotient + index);
      lower.bits = candidate.bits - radius;
      upper.bits = candidate.bits + radius;
      const uint16_t lo = intent_narrow_half_bits(lower.bits), hi = intent_narrow_half_bits(upper.bits);
      union { float value; uint32_t bits; } a, b;
      a.value = left; b.value = right;
      const uint32_t magnitude = candidate.bits & 0x7fffffffu;
      const bool normal_operands = !intent_division_vector_guards(rhs) ||
          ((((a.bits >> 23) & 255u) - 1u < 254u) && (((b.bits >> 23) & 255u) - 1u < 254u));
      const bool certified_range = magnitude >= 0x01000000u && magnitude <= 0x7e800000u && normal_operands;
      const uint32_t narrowed = !certified_range ? 0x10000u : lo == hi ? lo :
          intent_divide_half_boundary(left, right, lo, hi);
      if (narrowed <= 0xffffu) {
        intent_write_local(reinterpret_cast<uint16_t *>(output) + index, uint16_t(narrowed));
        continue;
      }
      intent_write_local(quotient + strict_count, __cn_scalar_div_f32_rn(left, right));
      intent_write_local(bounds + strict_count, float(index));
      ++strict_count;
    }
    if (strict_count) {
      __sync();
      __bang_float2half_rn(narrow_bounds, quotient, strict_count);
      __sync_compute();
      for (unsigned repaired = 0; repaired < strict_count; ++repaired) {
        const unsigned index = static_cast<unsigned>(intent_read_local(bounds + repaired));
        intent_write_local(reinterpret_cast<uint16_t *>(output) + index,
                           intent_read_local(reinterpret_cast<const uint16_t *>(narrow_bounds) + repaired));
      }
    }
    return;
  }
  // Dense exceptional inputs retain the same strict operation sequence.
  intent_strict_divide<N>(quotient, lhs, rhs);
  __sync();
  __bang_float2half_rn(output, quotient, N);
}
__mlu_func__ const float *intent_divide_rhs(float *temporary, const float *rhs, int first, int count) {
  return rhs + first;
}
__mlu_func__ const float *intent_divide_rhs(float *temporary, float rhs, int first, int count) {
  __bang_write_value(temporary, count, rhs);
  return temporary;
}
template<bool Upper>
__mlu_func__ void intent_divide_boundary(vv_uint32 &result, vv_uint32 q,
    vv_uint32 my, vv_uint32 left_low, vv_uint32 left_high) {
  vv_uint32 right_low, right_high, temporary, odd;
  __vv_and(temporary, q, 0x007fffffu);
  __vv_or(temporary, temporary, 0x00800000u);
  __vv_mul(temporary, temporary, 2u);
  if (Upper) __vv_add(temporary, temporary, 1u);
  else __vv_sub(temporary, temporary, 1u);
  __vv_mul(right_low, my, temporary);
  __vv_mulh(right_high, my, temporary);
  if (Upper) {
    __vv_lt(result, right_high, left_high);
    __vv_lt(temporary, right_low, left_low);
  } else {
    __vv_lt(result, left_high, right_high);
    __vv_lt(temporary, left_low, right_low);
  }
  __vv_eq(right_high, left_high, right_high);
  __vv_eq(right_low, left_low, right_low);
  __vv_and(odd, q, 1u);
  __vv_and(right_low, right_low, odd);
  __vv_or(temporary, temporary, right_low);
  __vv_and(temporary, temporary, right_high);
  __vv_or(result, result, temporary);
}
template<int N, int W, class Divisor>
__mlu_func__ void intent_divide_rn(float *output, const float *lhs, Divisor rhs,
                                 int32_t *workspace, const float *lane_indices) {
  auto *invalid = reinterpret_cast<uint32_t *>(workspace);
  auto *positions = reinterpret_cast<float *>(workspace + W);
  auto *uniform_divisor = reinterpret_cast<float *>(workspace + 2 * W);
  auto *reciprocal = reinterpret_cast<float *>(workspace + 3 * W);
  const int lanes = __vv_get_length() / sizeof(uint32_t);
  for (int first = 0; first < N; first += W) {
    const int count = N - first < W ? N - first : W;
    const float *divisor = intent_divide_rhs(uniform_divisor, rhs, first, count);
    asm volatile("recip.nram.f32 [%[dst]], [%[src]], %[count];"
        :: [dst]"r"(reciprocal), [src]"r"(divisor), [count]"r"(count));
    __bang_mul(output + first, lhs + first, reciprocal, count);
    __sync();
    asm volatile("vv.begin;" ::: "memory");
    for (int offset = 0; offset < count; offset += lanes) {
      const unsigned take = count - offset < lanes ? count - offset : lanes;
      vv_uint32 x, y, q, sign, ex, ey, eq, valid, temporary;
      if (take == lanes) {
        __vv_load(x, reinterpret_cast<const uint32_t *>(lhs + first + offset));
        __vv_load(y, reinterpret_cast<const uint32_t *>(divisor + offset));
        __vv_load(q, reinterpret_cast<const uint32_t *>(output + first + offset));
      } else {
        __vv_load(x, reinterpret_cast<const uint32_t *>(lhs + first + offset), take);
        __vv_load(y, reinterpret_cast<const uint32_t *>(divisor + offset), take);
        __vv_load(q, reinterpret_cast<const uint32_t *>(output + first + offset), take);
      }
      __vv_xor(sign, x, y);
      __vv_and(sign, sign, 0x80000000u);
      __vv_srl(ex, x, 23u); __vv_and(ex, ex, 255u);
      __vv_srl(ey, y, 23u); __vv_and(ey, ey, 255u);
      __vv_srl(eq, q, 23u); __vv_and(eq, eq, 255u);
      __vv_sub(temporary, ex, 1u); __vv_lt(valid, temporary, 254u);
      __vv_sub(temporary, ey, 1u); __vv_lt(temporary, temporary, 254u);
      __vv_and(valid, valid, temporary);
      __vv_sub(temporary, eq, 1u); __vv_lt(temporary, temporary, 254u);
      __vv_and(valid, valid, temporary);
      vv_uint32 shift, mx, my, left_low, left_high, above, below;
      __vv_sub(shift, ex, ey); __vv_sub(shift, shift, eq); __vv_add(shift, shift, 151u);
      __vv_sub(temporary, shift, 24u); __vv_lt(temporary, temporary, 2u);
      __vv_and(valid, valid, temporary);
      __vv_and(q, q, 0x7fffffffu);
      // Guard both the candidate and its one-step correction from binade edges.
      __vv_and(temporary, q, 0x007fffffu); __vv_sub(temporary, temporary, 2u);
      __vv_lt(temporary, temporary, 0x007ffffcu); __vv_and(valid, valid, temporary);
      __vv_and(mx, x, 0x007fffffu); __vv_or(mx, mx, 0x00800000u);
      __vv_and(my, y, 0x007fffffu); __vv_or(my, my, 0x00800000u);
      __vv_and(shift, shift, 31u);
      __vv_sll(left_low, mx, shift);
      __vv_move(temporary, 32u); __vv_sub(temporary, temporary, shift);
      __vv_and(temporary, temporary, 31u); __vv_srl(left_high, mx, temporary);
      intent_divide_boundary<true>(above, q, my, left_low, left_high);
      intent_divide_boundary<false>(below, q, my, left_low, left_high);
      __vv_add(q, q, above); __vv_sub(q, q, below);
      // Certify the corrected result against both exact RN boundaries. This
      // does not assume that reciprocal-times-product was within one encoding;
      // candidates outside the certified interval use strict scalar division.
      intent_divide_boundary<true>(above, q, my, left_low, left_high);
      intent_divide_boundary<false>(below, q, my, left_low, left_high);
      __vv_or(temporary, above, below); __vv_eq(temporary, temporary, 0u);
      __vv_and(valid, valid, temporary); __vv_or(q, q, sign);
      __vv_eq(valid, valid, 0u); __vv_mul(valid, valid, 0x3f800000u);
      if (take == lanes) {
        __vv_store(reinterpret_cast<uint32_t *>(output + first + offset), q);
        __vv_store(invalid + offset, valid);
      } else {
        __vv_store(reinterpret_cast<uint32_t *>(output + first + offset), q, take);
        __vv_store(invalid + offset, valid, take);
      }
    }
    asm volatile("vv.end;" ::: "memory");
    const unsigned failures = __bang_filter(positions, lane_indices,
                                            reinterpret_cast<float *>(invalid), count);
    __sync();
    for (unsigned repair = 0; repair < failures; ++repair) {
      const unsigned index = first + unsigned(intent_read_local(positions + repair));
      intent_write_local(output + index, __cn_scalar_div_f32_rn(intent_read_local(lhs + index), intent_divisor_lane(rhs, index)));
    }
    __sync();
  }
}
template<class T, int N>
__mlu_func__ void intent_select_local(T *output, bool condition, const T *lhs, const T *rhs) {
  intent_copy_local<T, N>(output, condition ? lhs : rhs);
}
template<class T, int N>
__mlu_func__ void intent_select_local(T *output, const bool *condition, const T *lhs, const T *rhs) {
  for (int i = 0; i < N; ++i) intent_write_local(output + i, intent_read_local((intent_read_local(condition + i) ? lhs : rhs) + i));
}
template<int N, int W>
__mlu_func__ void intent_select_bits(float *output, const bool *condition, const float *lhs,
                                   const float *rhs, int32_t *scratch) {
  if (lhs == rhs) { intent_copy_local<float, N>(output, lhs); return; }
  auto *mask = reinterpret_cast<uint32_t *>(scratch), *difference = mask + W;
  constexpr int prefix = N / 64 * 64;
  for (int first = 0; first < prefix; first += W) {
    const int count = prefix - first < W ? prefix - first : W;
    const auto *left = reinterpret_cast<const uint32_t *>(lhs + first);
    const auto *right = reinterpret_cast<const uint32_t *>(rhs + first);
    __bang_uint82float(reinterpret_cast<float *>(mask), reinterpret_cast<const uint8_t *>(condition + first), count, 0);
    asm volatile("fuse.nram.u32 [%[mask]], %[count], [%[mask]], .eq(0), .xor(1), .mul(0xffffffff);"
        :: [mask]"r"(mask), [count]"r"(count));
    __bang_bxor(difference, left, right, count);
    asm volatile("fuse.nram.u32 [%[out]], %[count], [%[diff]], .and([%[mask]]), .xor([%[right]]);"
        :: [out]"r"(output + first), [diff]"r"(difference), [mask]"r"(mask), [right]"r"(right), [count]"r"(count));
  }
  __sync_compute();
  for (int i = prefix; i < N; ++i)
    intent_write_local(output + i, intent_read_local((intent_read_local(condition + i) ? lhs : rhs) + i));
}
template<class T, int N>
__mlu_func__ void intent_select_local(T *output, bool condition, const T *lhs, T rhs) {
  if (condition) intent_copy_local<T, N>(output, lhs);
  else intent_fill_local<T, N>(output, rhs);
}
template<class T, int N>
__mlu_func__ void intent_select_local(T *output, const bool *condition, const T *lhs, T rhs) {
  for (int i = 0; i < N; ++i)
    intent_write_local(output + i, intent_read_local(condition + i) ? intent_read_local(lhs + i) : rhs);
}
template<int N, int W>
__mlu_func__ void intent_select_bits(float *output, const bool *condition, const float *lhs,
                                   float rhs, int32_t *scratch) {
  union { float value; uint32_t bits; } right;
  right.value = rhs;
  constexpr int prefix = N / 64 * 64;
  for (int first = 0; first < prefix; first += W) {
    const int count = prefix - first < W ? prefix - first : W;
    __bang_uint82float(reinterpret_cast<float *>(scratch), reinterpret_cast<const uint8_t *>(condition + first), count, 0);
    if (right.bits == 0) {
      asm volatile("fuse.nram.u32 [%[out]], %[count], [%[mask]], .eq(0), .xor(1), .mul(0xffffffff), .and([%[left]]);"
          :: [out]"r"(output + first), [mask]"r"(scratch), [left]"r"(lhs + first), [count]"r"(count));
    } else {
      asm volatile("fuse.nram.u32 [%[mask]], %[count], [%[mask]], .eq(0), .xor(1), .mul(0xffffffff);"
          :: [mask]"r"(scratch), [count]"r"(count));
      asm volatile("fuse.nram.u32 [%[out]], %[count], [%[left]], .xor(%[right]), .and([%[mask]]), .xor(%[right]);"
          :: [out]"r"(output + first), [left]"r"(lhs + first), [right]"r"(right.bits), [mask]"r"(scratch), [count]"r"(count));
    }
  }
  __sync_compute();
  for (int i = prefix; i < N; ++i)
    intent_write_local(output + i, intent_read_local(condition + i) ? intent_read_local(lhs + i) : rhs);
}
template<class T> struct IntentExtremaBits;
template<> struct IntentExtremaBits<float> { using Type = uint32_t; enum : uint32_t { magnitude = 0x7fffffffu, first_nan = 0x7f800001u }; };
template<> struct IntentExtremaBits<half> { using Type = uint16_t; enum : uint16_t { magnitude = 0x7fffu, first_nan = 0x7c01u }; };
template<class T, int N, int W, bool Maximum>
__mlu_func__ void intent_propagating_extrema(T *output, const T *lhs, const T *rhs, T *workspace) {
  using Bits = typename IntentExtremaBits<T>::Type;
  for (int first = 0; first < N; first += W) {
    const int count = N - first < W ? N - first : W;
    auto *bits = reinterpret_cast<Bits *>(workspace);
    const auto *left = reinterpret_cast<const Bits *>(lhs + first);
    const auto *right = reinterpret_cast<const Bits *>(rhs + first);
    __bang_band_scalar(bits, right, Bits(IntentExtremaBits<T>::magnitude), count);
    __bang_ge_scalar(bits, bits, Bits(IntentExtremaBits<T>::first_nan), count);
    __bang_mul_scalar(bits, bits, Bits(~Bits(0)), count);
    __bang_band(bits, bits, right, count);
    __bang_bor(bits, bits, left, count);
    // Both NaNs now reach src0. The native extrema keeps src0 on unordered
    // lanes; its only in-place destination is the independent workspace.
    if constexpr (Maximum) __bang_maxequal(workspace, workspace, rhs + first, count);
    else __bang_minequal(workspace, workspace, rhs + first, count);
    __sync_compute();
    __memcpy(output + first, workspace, count * sizeof(T), NRAM2NRAM);
    __sync_move();
  }
}
template<class T, int N, int W, bool Maximum>
__mlu_func__ void intent_numeric_extrema(T *output, const T *lhs, const T *rhs, T *workspace) {
  // On compute_30 the SDK's maximum/minimum wrappers overwrite src1 during
  // NaN repair despite its const-qualified signature. Keep logical inputs live.
  for (int first = 0; first < N; first += W) {
    const int count = N - first < W ? N - first : W;
    __memcpy(workspace, rhs + first, count * sizeof(T), NRAM2NRAM);
    __sync_move();
    if (Maximum) __bang_maximum(output + first, lhs + first, workspace, count);
    else __bang_minimum(output + first, lhs + first, workspace, count);
    __sync_compute();
  }
}
template<int N, int W, int Predicate>
__mlu_func__ void intent_compare_i64(bool *output, const int64_t *lhs, const int64_t *rhs, int32_t *workspace) {
  uint32_t *left_lo = reinterpret_cast<uint32_t *>(workspace), *right_lo = left_lo + 2 * W;
  uint32_t *result = right_lo + 2 * W, *equal_hi = result + W;
  for (int first = 0; first < N; first += W) {
    const int count = N - first < W ? N - first : W;
    int32_t *left_hi = reinterpret_cast<int32_t *>(left_lo + count);
    int32_t *right_hi = reinterpret_cast<int32_t *>(right_lo + count);
    const uint32_t *a = reinterpret_cast<const uint32_t *>((Predicate == 4 || Predicate == 5 ? rhs : lhs) + first);
    const uint32_t *b = reinterpret_cast<const uint32_t *>((Predicate == 4 || Predicate == 5 ? lhs : rhs) + first);
    __bang_transpose(left_lo, a, count, 2);
    __bang_transpose(right_lo, b, count, 2);
    if (Predicate == 0 || Predicate == 1) {
      __bang_eq(equal_hi, reinterpret_cast<const uint32_t *>(left_hi), reinterpret_cast<const uint32_t *>(right_hi), count);
      __bang_eq(result, left_lo, right_lo, count);
      __bang_band(result, result, equal_hi, count);
      if (Predicate == 1) __bang_bxor_scalar(result, result, 1u, count);
    } else {
      __bang_lt(reinterpret_cast<int32_t *>(result), left_hi, right_hi, count);
      __bang_eq(reinterpret_cast<int32_t *>(equal_hi), left_hi, right_hi, count);
      // The lower word is unsigned, even when the complete i64 is signed.
      if (Predicate == 3 || Predicate == 5) __bang_ge(left_lo, right_lo, left_lo, count);
      else __bang_lt(left_lo, left_lo, right_lo, count);
      __bang_band(equal_hi, equal_hi, left_lo, count);
      __bang_bor(result, result, equal_hi, count);
    }
    __sync_compute();
    __cn_vector_cast_u32_to_u8(count, reinterpret_cast<uint8_t *>(output + first), result);
    __sync();
  }
}
template<class T, int R, int C>
__mlu_func__ void intent_stage_tile(T *dst, const T *src, int64_t offset,
    int64_t row_stride, int64_t column_stride, int64_t rows, int64_t columns) {
  if (!rows || !columns) return;
  if (column_stride == 1 && (rows == 1 || (columns == C && row_stride == C))) {
    __memcpy_async(dst, src + offset, rows * columns * sizeof(T), GDRAM2SRAM);
    return;
  }
  if (column_stride == 1 && row_stride >= columns &&
      uint64_t(row_stride) <= 0x7fffffffu / sizeof(T)) {
    for (int64_t first = 0; first < rows; first += 65536) {
      const int64_t take = rows - first < 65536 ? rows - first : 65536;
      __memcpy_async(dst + first * C, src + offset + first * row_stride,
                     columns * sizeof(T), GDRAM2SRAM, C * sizeof(T), row_stride * sizeof(T), take - 1);
    }
    return;
  }
  for (int64_t row = 0; row < rows; ++row) {
    if (column_stride == 1)
      __memcpy_async(dst + row * C, src + offset + row * row_stride, columns * sizeof(T), GDRAM2SRAM);
    else for (int64_t col = 0; col < columns; ++col)
      __memcpy_async(dst + row * C + col, src + offset + row * row_stride + col * column_stride, sizeof(T), GDRAM2SRAM);
  }
}
template<class T, int R, int C, bool Async = false, bool Shared = false>
__mlu_func__ void intent_load_tile(T *dst, const T *src, int64_t offset,
    int64_t row_stride, int64_t column_stride, int64_t rows, int64_t columns) {
  constexpr mluMemcpyDirection_t direction = Shared ? SRAM2NRAM : GDRAM2NRAM;
  if (rows != R || columns != C) {
    intent_zero_tile<R * C>(dst);
    __sync_compute();
  }
  if (!rows || !columns) return;
  if (column_stride == 1 && (rows == 1 || (columns == C && row_stride == C))) {
    if (Async) __memcpy_async(dst, src + offset, rows * columns * sizeof(T), direction);
    else __memcpy(dst, src + offset, rows * columns * sizeof(T), direction);
    return;
  }
  if (column_stride == 1 && row_stride >= columns &&
      uint64_t(row_stride) <= 0x7fffffffu / sizeof(T)) {
    // The native two-dimensional transfer counts additional rows after the
    // first one. MLU370 permits at most 65535 additional rows per transfer.
    for (int64_t first = 0; first < rows; first += 65536) {
      const int64_t take = rows - first < 65536 ? rows - first : 65536;
      if (Async)
        __memcpy_async(dst + first * C, src + offset + first * row_stride,
                       columns * sizeof(T), direction,
                       C * sizeof(T), row_stride * sizeof(T), take - 1);
      else
        __memcpy(dst + first * C, src + offset + first * row_stride,
                 columns * sizeof(T), direction,
                 C * sizeof(T), row_stride * sizeof(T), take - 1);
    }
    return;
  }
  for (int64_t r = 0; r < rows; ++r) {
    if (column_stride == 1) {
      if (Async) __memcpy_async(dst + r * C, src + offset + r * row_stride,
                               columns * sizeof(T), direction);
      else __memcpy(dst + r * C, src + offset + r * row_stride,
                    columns * sizeof(T), direction);
    } else {
      for (int64_t c = 0; c < columns; ++c) {
        if (Shared) __memcpy(dst + r * C + c, src + offset + r * row_stride + c * column_stride, sizeof(T), direction);
        else intent_write_local(dst + r * C + c, src[offset + r * row_stride + c * column_stride]);
      }
    }
  }
}
template<class T, int R, int C>
__mlu_func__ void intent_broadcast_rows(T *dst, const T *src, T *scratch,
                                      int64_t offset, int64_t rows, int64_t columns) {
  if (rows != R || columns != C) { intent_zero_tile<R * C>(scratch); __sync_compute(); }
  if (rows && columns) {
    __memcpy(scratch, src + offset, rows * sizeof(T), NRAM2NRAM,
             R * sizeof(T), 0, columns - 1);
    __sync_move();
  }
  __bang_transpose(dst, scratch, C, R);
}
template<class T, int R, int C>
__mlu_func__ void intent_gather_rows(T *dst, const T *src, const int64_t *offsets,
    int64_t column_stride, int64_t rows, int64_t columns) {
  if (rows != R || columns != C) {
    intent_zero_tile<R * C>(dst);
    __sync_compute();
  }
  if (!rows || !columns) return;
  int64_t first = 0;
  while (first < rows) {
    const int64_t offset = intent_read_local(offsets + first);
    int64_t take = 1;
    if (column_stride == 1) {
      int64_t pitch = 0;
      if (first + 1 < rows) {
        pitch = intent_read_local(offsets + first + 1) - offset;
        if (pitch >= 0 && uint64_t(pitch) <= 0x7fffffffu / sizeof(T)) {
          take = 2;
          int64_t previous = offset + pitch;
          while (first + take < rows && take < 65536) {
            const int64_t next = intent_read_local(offsets + first + take);
            if (next - previous != pitch) break;
            previous = next;
            ++take;
          }
        }
      }
      if (take > 1 && pitch == 0) {
        // Repeated logical rows share one external snapshot. Supply it once,
        // then replicate from the first local row into disjoint output rows.
        __memcpy(dst + first * C, src + offset, columns * sizeof(T), GDRAM2NRAM);
        __sync();
        __memcpy(dst + (first + 1) * C, dst + first * C, columns * sizeof(T), NRAM2NRAM,
                 C * sizeof(T), 0, take - 2);
      } else if (take > 1)
        __memcpy(dst + first * C, src + offset, columns * sizeof(T), GDRAM2NRAM,
                 C * sizeof(T), pitch * sizeof(T), take - 1);
      else
        __memcpy(dst + first * C, src + offset, columns * sizeof(T), GDRAM2NRAM);
    } else {
      for (int64_t column = 0; column < columns; ++column)
        intent_write_local(dst + first * C + column, src[offset + column * column_stride]);
    }
    first += take;
  }
}
template<int R>
__mlu_func__ void intent_prepare_gather_runs(const int64_t *offsets, int64_t rows,
    int64_t *workspace, const float *indices) {
  unsigned runs = 0;
  int64_t pitch = 0;
  bool consecutive_repeats = false;
  if (rows == R) {
    int64_t *expected = workspace;
    bool *flags = reinterpret_cast<bool *>(workspace + R);
    float *selected = reinterpret_cast<float *>(workspace) + 3 * R;
    int32_t *positions = reinterpret_cast<int32_t *>(selected + R);
    const int64_t offset = intent_read_local(offsets);
    pitch = int64_t(uint64_t(intent_read_local(offsets + 1)) - uint64_t(offset));
    // A descriptor is dtype-independent; use the largest supported element
    // size when checking the DMA byte-stride limit.
    if (pitch >= 0 && uint64_t(pitch) <= 0x7fffffffu / 8) {
      // Consecutive rows with the same address difference form a strided DMA
      // run. Compact its ends in parallel instead of inspecting every offset
      // with scalar loads. Zero pitch also covers repeated snapshots.
      __memcpy(expected, offsets + 1, (R - 1) * sizeof(int64_t), NRAM2NRAM);
      __sync();
      __cn_vector_sub_s64(R - 1, expected, expected, offsets);
      __sync();
      __cn_vector_ne_scalar_s64(R - 1, flags, expected, pitch);
      __sync();
      intent_write_local(flags + R - 1, true);
      __sync();
      intent_bool_to_f32_tile<R>(selected, flags);
      runs = __bang_filter(reinterpret_cast<float *>(positions), indices, selected, R);
      __sync();
      if (runs > R / 2) runs = 0;
      if (runs) {
        __bang_float2int32_tz(positions, reinterpret_cast<const float *>(positions), runs, 0);
        __sync_compute();
      }
      if (runs && !pitch) {
        consecutive_repeats = true;
        int begin = 0;
        for (unsigned run = 0; run < runs; ++run) {
          if (uint64_t(intent_read_local(offsets + begin)) - uint64_t(offset) != run) {
            consecutive_repeats = false; break;
          }
          begin = int(intent_read_local(positions + run)) + 1;
        }
      }
    }
  }
  // The temporary differences are dead here. Keep the compact positions in
  // the final R i32 elements and publish the descriptor in their place.
  intent_write_local(workspace, pitch);
  intent_write_local(workspace + 1, int64_t(runs));
  intent_write_local(workspace + 2, int64_t(consecutive_repeats));
}
template<class T, int R, int C>
__mlu_func__ void intent_gather_runs(T *dst, const T *src, const int64_t *offsets,
    int64_t column_stride, int64_t rows, int64_t columns, const int64_t *plan) {
  const unsigned runs = unsigned(intent_read_local(plan + 1));
  if (runs && rows == R && columns > 0 && column_stride == 1) {
    const int64_t pitch = intent_read_local(plan);
    const int32_t *positions = reinterpret_cast<const int32_t *>(plan) + 4 * R;
    if constexpr (C == 1 && sizeof(T) == 4) {
      if (intent_read_local(plan + 2)) {
        // Fetch each distinct consecutive scalar once, in one DMA. Forward
        // expansion from the output's tail cannot overwrite a future source:
        // each remaining run contributes at least one remaining output row.
        __memcpy(dst + R - runs, src + intent_read_local(offsets), runs * sizeof(T), GDRAM2NRAM);
        __sync();
        int begin = 0;
        for (unsigned run = 0; run < runs; ++run) {
          const int end = int(intent_read_local(positions + run)) + 1;
          const uint32_t value = intent_read_local(reinterpret_cast<const uint32_t *>(dst + R - runs + run));
          __bang_write_value(reinterpret_cast<uint32_t *>(dst + begin), end - begin, value);
          begin = end;
        }
        return;
      }
    }
    if (columns != C) { intent_zero_tile<R * C>(dst); __sync_compute(); }
    int begin = 0;
    for (unsigned run = 0; run < runs; ++run) {
      const int end = int(intent_read_local(positions + run)) + 1;
      const int64_t source = intent_read_local(offsets + begin);
      if (pitch) {
        __memcpy_async(dst + begin * C, src + source, columns * sizeof(T), GDRAM2NRAM,
                       C * sizeof(T), pitch * sizeof(T), end - begin - 1);
      } else {
        __memcpy(dst + begin * C, src + source, columns * sizeof(T), GDRAM2NRAM);
        if (end - begin > 1) {
          __sync();
          __memcpy(dst + (begin + 1) * C, dst + begin * C, columns * sizeof(T), NRAM2NRAM,
                   C * sizeof(T), 0, end - begin - 2);
        }
      }
      begin = end;
    }
    return;
  }
  intent_gather_rows<T, R, C>(dst, src, offsets, column_stride, rows, columns);
}
template<class T, int R, int C>
__mlu_func__ void intent_group_gather_rows(T *output, const T *source,
    const int64_t *offsets, const int64_t *plan, T *shared_data,
    int64_t *shared_metadata, int64_t rows, int64_t lane) {
  // The complete collective owns publication, supply and consumption. Fences
  // precede each barrier, so another participant never observes pending DMA.
  __sync();
  if (!__is_mpu() && lane == 0) {
    __memcpy_async(shared_metadata, offsets, R * sizeof(int64_t), NRAM2SRAM);
    __memcpy_async(shared_metadata + R, plan, 3 * R * sizeof(int64_t), NRAM2SRAM);
  }
  __sync(); __sync_cluster();
  if (__is_mpu()) {
    const int64_t *published_plan = shared_metadata + R;
    const unsigned runs = unsigned(__load_sram(published_plan + 1));
    if (runs && rows == R) {
      const int64_t pitch = __load_sram(published_plan);
      const int32_t *positions = reinterpret_cast<const int32_t *>(published_plan) + 4 * R;
      int begin = 0;
      for (unsigned run = 0; run < runs; ++run) {
        const int end = int(__load_sram(positions + run)) + 1;
        const int64_t offset = __load_sram(shared_metadata + begin);
        __memcpy_async(shared_data + begin * 4 * C, source + offset, 4 * C * sizeof(T), GDRAM2SRAM,
                       4 * C * sizeof(T), pitch * sizeof(T), end - begin - 1);
        begin = end;
      }
    } else {
      for (int row = 0; row < rows; ++row) {
        const int64_t offset = __load_sram(shared_metadata + row);
        __memcpy_async(shared_data + row * 4 * C, source + offset, 4 * C * sizeof(T), GDRAM2SRAM);
      }
    }
  }
  __sync(); __sync_cluster();
  if (!__is_mpu())
    intent_load_tile<T, R, C, true, true>(output, shared_data, lane * C, 4 * C, 1, rows, C);
  __sync(); __sync_cluster();
}
template<class T, int K, int N, int SliceK, int SliceN>
__mlu_func__ void intent_prepare_matrix_view(T *filter, const T *source, T *input,
    T *transposed, int64_t offset, int64_t row_stride, int64_t column_stride, int64_t columns) {
  static_assert(SliceN % 64 == 0 && N % SliceN == 0 && K % SliceK == 0,
                "resident filters use complete reduction slices and 64-channel groups");
  constexpr int bank_bytes = __MLU_WRAM_SIZE__ * 1024 / 16;
  for (int n = 0; n < N; n += SliceN) {
    const int active = columns <= n ? 0 : columns - n < SliceN ? columns - n : SliceN;
    for (int k = 0; k < K; k += SliceK) {
      intent_load_tile<T, SliceK, SliceN>(input, source, offset + k * row_stride + n * column_stride,
                                    row_stride, column_stride, SliceK, active);
      __sync();
      __bang_transpose(transposed, input, SliceK, SliceN);
      __sync_compute();
      // Each LT holds four channels per 64-channel group. Scatter K slices
      // into those full-K rows; all offsets below are bytes, including WRAM.
      for (int group = 0; group < SliceN / 64; ++group) {
        auto *destination = reinterpret_cast<unsigned char *>(filter) + ((n / 64 + group) * 4 * K + k) * sizeof(T);
        __memcpy_async(destination, transposed + group * 64 * SliceK, SliceK * sizeof(T), NRAM2WRAM,
                       bank_bytes, 15, K * sizeof(T), 3,
                       4 * SliceK * sizeof(T), 15, SliceK * sizeof(T), 3);
      }
      __sync_move();
    }
  }
}
template<class T, int R, int C>
__mlu_func__ void intent_store_tile(T *dst, const T *src, int64_t offset,
    int64_t row_stride, int64_t column_stride, int64_t rows, int64_t columns) {
  if (!rows || !columns) return;
  if (column_stride == 1 && (rows == 1 || (columns == C && row_stride == C))) {
    __memcpy(dst + offset, src, rows * columns * sizeof(T), NRAM2GDRAM);
    return;
  }
  if (column_stride == 1 && row_stride >= columns &&
      uint64_t(row_stride) <= 0xffffffffu / sizeof(T)) {
    for (int64_t first = 0; first < rows; first += 65536) {
      const int64_t take = rows - first < 65536 ? rows - first : 65536;
      __memcpy(dst + offset + first * row_stride, src + first * C,
               columns * sizeof(T), NRAM2GDRAM,
               row_stride * sizeof(T), C * sizeof(T), take - 1);
    }
    return;
  }
  for (int64_t r = 0; r < rows; ++r) {
    if (column_stride == 1) {
      if (columns) __memcpy(dst + offset + r * row_stride, src + r * C,
                            columns * sizeof(T), NRAM2GDRAM);
    } else {
      for (int64_t c = 0; c < columns; ++c)
        dst[offset + r * row_stride + c * column_stride] = intent_read_local(src + r * C + c);
    }
  }
}
template<class T>
__mlu_func__ void intent_broadcast_local_row(T *dst, const T *src, int count) {
  __memcpy(dst, src, sizeof(T), NRAM2NRAM, sizeof(T), 0, count - 1);
}
__mlu_func__ void intent_broadcast_local_row(float *dst, const float *src, int count) {
  __bang_write_value(dst, count, intent_read_local(src));
}
__mlu_func__ void intent_broadcast_local_row(half *dst, const half *src, int count) {
  __bang_write_value(dst, count, intent_read_local(src));
}
template<class T, int R, int C>
__mlu_func__ void intent_load_local_tile(T *dst, const T *src, int64_t offset,
    int64_t row_stride, int64_t column_stride, int64_t rows, int64_t columns) {
  if (rows != R || columns != C) {
    intent_zero_tile<R * C>(dst);
    __sync_compute();
  }
  if (!rows || !columns) return;
  if (column_stride == 1 && (row_stride == 0 || row_stride >= columns)) {
    for (int64_t first = 0; first < rows; first += 65536) {
      const int64_t take = rows - first < 65536 ? rows - first : 65536;
      __memcpy(dst + first * C, src + offset + first * row_stride,
               columns * sizeof(T), NRAM2NRAM,
               C * sizeof(T), row_stride * sizeof(T), take - 1);
    }
    return;
  }
  for (int64_t r = 0; r < rows; ++r) {
    if (column_stride >= 0) {
      for (int64_t first = 0; first < columns; first += 65536) {
        const int64_t take = columns - first < 65536 ? columns - first : 65536;
        if (column_stride == 0)
          intent_broadcast_local_row(dst + r * C + first, src + offset + r * row_stride, take);
        else __memcpy(dst + r * C + first, src + offset + r * row_stride + first * column_stride,
                      sizeof(T), NRAM2NRAM, sizeof(T), column_stride * sizeof(T), take - 1);
      }
    } else for (int64_t c = 0; c < columns; ++c)
      intent_write_local(dst + r * C + c, intent_read_local(src + offset + r * row_stride + c * column_stride));
  }
}
template<class T, int R, int C>
__mlu_func__ void intent_store_local_tile(T *dst, const T *src, int64_t offset,
    int64_t row_stride, int64_t column_stride, int64_t rows, int64_t columns) {
  if (!rows || !columns) return;
  if (column_stride == 1 && row_stride >= columns) {
    __memcpy(dst + offset, src, columns * sizeof(T), NRAM2NRAM,
             row_stride * sizeof(T), C * sizeof(T), rows - 1);
    return;
  }
  for (int64_t r = 0; r < rows; ++r) {
    if (column_stride > 0)
      __memcpy(dst + offset + r * row_stride, src + r * C, sizeof(T), NRAM2NRAM,
               column_stride * sizeof(T), sizeof(T), columns - 1);
    else for (int64_t c = 0; c < columns; ++c)
      intent_write_local(dst + offset + r * row_stride + c * column_stride, intent_read_local(src + r * C + c));
  }
}
template<class T>
__mlu_func__ void intent_transpose_full(T *output, const T *input, int rows, int columns) {
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < columns; ++c)
      intent_write_local(output + c * rows + r, intent_read_local(input + r * columns + c));
}
__mlu_func__ void intent_transpose_full(float *output, const float *input, int rows, int columns) {
  __bang_transpose(output, input, rows, columns);
}
__mlu_func__ void intent_transpose_full(half *output, const half *input, int rows, int columns) {
  __bang_transpose(output, input, rows, columns);
}
__mlu_func__ void intent_transpose_full(uint16_t *output, const uint16_t *input, int rows, int columns) {
  __bang_transpose(output, input, rows, columns);
}
__mlu_func__ void intent_transpose_full(int32_t *output, const int32_t *input, int rows, int columns) {
  __bang_transpose(output, input, rows, columns);
}
__mlu_func__ void intent_transpose_full(bool *output, const bool *input, int rows, int columns) {
  __bang_transpose(reinterpret_cast<unsigned char *>(output), reinterpret_cast<const unsigned char *>(input), rows, columns);
}
template<class T, int R, int C>
__mlu_func__ void intent_transpose_tile(T *output, const T *input, int64_t rows, int64_t columns) {
  if (rows == R && columns == C) {
    intent_transpose_full(output, input, R, C);
    return;
  }
  intent_zero_tile<R * C>(output);
  __sync();
  for (int64_t r = 0; r < rows; ++r)
    for (int64_t c = 0; c < columns; ++c)
      intent_write_local(output + c * R + r, intent_read_local(input + r * C + c));
}
template<class T, int K, int N, bool InputTransposed>
__mlu_func__ void intent_prepare_matrix(T *packed, const T *row_major, T *transposed, T *reshaped) {
  if (!InputTransposed) {
    __bang_transpose(transposed, row_major, K, N);
    __sync_compute();
  }
  const T *filter = InputTransposed ? row_major : transposed;
  static_assert(N % 64 == 0, "matrix RHS output channels must be 64 aligned");
  if (N != 64) {
    __bang_reshape_filter(reshaped, filter, N, 1, 1, K);
    __sync_compute();
  }
  __memcpy_async(packed, N == 64 ? filter : reshaped, K * N * sizeof(T), NRAM2WRAM);
  __sync_move();
}
template<class T> __mlu_func__ void intent_matrix_primitive(float *output, const T *a, const T *b, int m, int k, int n) {
  __bang_conv_partial(output, a, b, output, k, m, 1, 1, 1, 1, 1, n);
}
template<> __mlu_func__ void intent_matrix_primitive(float *output, const uint16_t *a, const uint16_t *b, int m, int k, int n) {
  __bang_conv_partial(output, reinterpret_cast<const bfloat16_t *>(a), reinterpret_cast<const bfloat16_t *>(b),
                      output, k, m, 1, 1, 1, 1, 1, n);
}
template<class T> __mlu_func__ void intent_matrix_initialize(float *output, const T *a, const T *b, int m, int k, int n) {
  __bang_conv(output, a, b, k, m, 1, 1, 1, 1, 1, n);
}
template<> __mlu_func__ void intent_matrix_initialize(float *output, const uint16_t *a, const uint16_t *b, int m, int k, int n) {
  __bang_conv(output, reinterpret_cast<const bfloat16_t *>(a), reinterpret_cast<const bfloat16_t *>(b),
              k, m, 1, 1, 1, 1, 1, n);
}
template<class T, int M, int K, int N, bool Accumulate = true>
__mlu_func__ void intent_matmul(float *accumulator, const T *a, const T *packed_b) {
  if constexpr (Accumulate) intent_matrix_primitive(accumulator, a, packed_b, M, K, N);
  else intent_matrix_initialize(accumulator, a, packed_b, M, K, N);
}
template<int C>
__mlu_func__ void intent_reduce_rows(float *out, const float *input, float *scratch,
                                   int64_t count, float identity) {
  __bang_write_value(out, C, identity);
  __sync_compute();
  for (int64_t first = 0; first < count; first += 65535) {
    const int take = count - first < 65535 ? count - first : 65535;
    __bang_sumpool(scratch, input + first * C, C, take, 1, take, 1, 1, 1);
    __sync_compute();
    __bang_add(out, out, scratch, C);
    __sync_compute();
  }
}
template<int R, int C>
__mlu_func__ void intent_reduce_row_sums(float *out, const float *input, float *scratch,
                                       int64_t count, float identity) {
  static_assert(R > 1 && C > 0 && C < 65536, "row pooling requires bounded local rows");
  __bang_write_value(out, R, identity);
  __sync_compute();
  if (count == 0) return;
  // HWC with one channel: each height position is an independent row, and
  // width stride C selects exactly one active-column window from that row.
  __bang_sumpool(scratch, input, 1, R, C, 1, count, C, 1);
  __sync_compute();
  __bang_add(out, out, scratch, R);
}
template<int N, int Kind>
__mlu_func__ void intent_reduce(float *out, const float *input, float *scratch,
                              int64_t count, float identity) {
  // The scratch and result storage are explicit operands in the DSA program.
  // Preserve 32 parallel channels while pooling a long row, then reduce the
  // resulting 128-byte group in the provided workspace.
  if (Kind == 0) {
    static_assert(N >= 32 && N % 32 == 0, "reduce_sum requires 128-byte input groups");
    float value = identity;
    int64_t processed = 0;
    if (N >= 256 && count >= 256 && scratch != input) {
      while (processed + 256 <= count) {
        const int64_t available = (count - processed) / 32;
        const int groups = available < 65535 ? available : 65535;
        __bang_sumpool(scratch, input + processed, 32, groups, 1, groups, 1, 1, 1);
        __sync_compute();
        __bang_reduce_sum(scratch, scratch, 32);
        __sync_compute();
        value += intent_read_local(scratch);
        processed += int64_t(groups) * 32;
      }
    }
    const int remaining = count - processed;
    if (remaining) {
      const int padded = (remaining + 31) / 32 * 32;
      if (padded != remaining) {
        __bang_write_value(scratch, padded, 0.0f);
        __sync_compute();
      }
      __memcpy(scratch, input + processed, remaining * sizeof(float), NRAM2NRAM);
      __sync_move();
      __bang_reduce_sum(scratch, scratch, padded);
      __sync_compute();
      for (int i = 0; i < padded / 32; ++i) value += intent_read_local(scratch + i * 32);
    }
    intent_write_local(out, value);
  } else {
    if ((Kind == 9 || Kind == 10) && count >= 4 && scratch != input && !isnan(identity)) {
      // One observed non-NaN value certifies the native reduction's domain.
      // When it cannot certify the tile, retain the identity-sentinel path.
      const bool sentinel = isnan(intent_read_local(input));
      const float *candidates = input;
      if (sentinel) {
        __memcpy(scratch, input, (count - 1) * sizeof(float), NRAM2NRAM);
        intent_write_local(scratch + count - 1, identity);
        intent_sync_local(); candidates = scratch;
      }
      if (Kind == 9) __bang_argmax(scratch, candidates, count);
      else __bang_argmin(scratch, candidates, count);
      intent_sync_local();
      union { float value; uint32_t bits; } reduced;
      reduced.value = intent_read_local(scratch);
      if (sentinel) {
        if (Kind == 9) reduced.value = fmaxf(reduced.value, intent_read_local(input + count - 1));
        else reduced.value = fminf(reduced.value, intent_read_local(input + count - 1));
      }
      reduced.value = Kind == 9 ? fmaxf(identity, reduced.value) : fminf(identity, reduced.value);
      if (reduced.bits & 0x7f800000u) { intent_write_local(out, reduced.value); return; }
    }
    if (N >= 4 && count > 0) {
      // The hardware extrema primitive excludes NaNs. An ordered equality
      // produces exact 0/1 values, so its minimum certifies this invocation.
      __bang_eq(scratch, input, input, count);
      __sync_compute();
      __bang_argmin(scratch, scratch, count);
      intent_sync_local();
      if (intent_read_local(scratch) == 1.0f) {
        if (Kind == 7 || Kind == 9) __bang_argmax(scratch, input, count);
        else __bang_argmin(scratch, input, count);
        intent_sync_local();
        union { float value; uint32_t bits; } reduced;
        reduced.value = intent_read_local(scratch);
        // Preserve signed-zero and subnormal ordering through the original
        // scalar combine where the hardware tie/denormal rules are unspecified.
        if (reduced.bits & 0x7f800000u) {
          float value;
          if (Kind == 7) value = isnan(identity) ? NAN : fmaxf(identity, reduced.value);
          if (Kind == 8) value = isnan(identity) ? NAN : fminf(identity, reduced.value);
          if (Kind == 9) value = fmaxf(identity, reduced.value);
          if (Kind == 10) value = fminf(identity, reduced.value);
          intent_write_local(out, value);
          return;
        }
      }
    }
    float value = identity;
    for (int64_t i = 0; i < count; ++i) {
      const float next = intent_read_local(input + i);
      if (Kind == 7) value = isnan(value) || isnan(next) ? NAN : fmaxf(value, next);
      if (Kind == 8) value = isnan(value) || isnan(next) ? NAN : fminf(value, next);
      if (Kind == 9) value = fmaxf(value, next);
      if (Kind == 10) value = fminf(value, next);
    }
    intent_write_local(out, value);
  }
}
template<int R, int C, int Kind>
__mlu_func__ void intent_reduce_half_extrema(float *out, const half *input, half *scratch,
                                           int64_t count, float identity) {
  for (int row = 0; row < R; ++row) {
    const half *source = input + row * C;
    if (count >= 4 && !isnan(identity)) {
      const bool sentinel = isnan(float(intent_read_local(source)));
      const half *candidates = source;
      if (sentinel) {
        __memcpy(scratch, source, (count - 1) * sizeof(half), NRAM2NRAM);
        intent_write_local(scratch + count - 1, half(Kind == 9 ? -INFINITY : INFINITY));
        intent_sync_local(); candidates = scratch;
      }
      if constexpr (Kind == 9) __bang_argmax(scratch, candidates, count);
      else __bang_argmin(scratch, candidates, count);
      intent_sync_local();
      union { half value; uint16_t bits; } selected;
      selected.value = intent_read_local(scratch);
      // Normal half extrema retain exact f32 order. Handle zeros and half
      // subnormals using the original widened comparisons below.
      if (selected.bits & 0x7c00u) {
        float value = float(selected.value);
        if (sentinel) {
          float last = float(intent_read_local(source + count - 1));
          value = Kind == 9 ? fmaxf(value, last) : fminf(value, last);
        }
        value = Kind == 9 ? fmaxf(identity, value) : fminf(identity, value);
        intent_write_local(out + row, value); continue;
      }
    }
    float value = identity;
    for (int64_t i = 0; i < count; ++i) {
      const float next = float(intent_read_local(source + i));
      value = Kind == 9 ? fmaxf(value, next) : fminf(value, next);
    }
    intent_write_local(out + row, value);
  }
}
template<int R, int C, int Kind>
__mlu_func__ void intent_reduce_row_tiles(float *out, const float *input, float *scratch,
                                        int64_t count, float identity) {
  if constexpr (Kind == 0 && C % 32 == 0 && C >= R * 32) {
    if (count == C) {
      // Keep 32 channels in each row while reducing every row in one pool.
      // The existing C-element workspace holds the R groups of partial sums.
      __bang_sumpool(scratch, input, 32, R, C / 32, 1, C / 32, C / 32, 1);
      __sync_compute();
      __bang_reduce_sum(scratch, scratch, R * 32);
      __sync_compute();
      __memcpy(out, scratch, sizeof(float), NRAM2NRAM,
               sizeof(float), 32 * sizeof(float), R - 1);
      __sync_move();
      __bang_add_scalar(out, out, identity, R);
      return;
    }
  }
  for (int row = 0; row < R; ++row) intent_reduce<C, Kind>(out + row, input + row * C, scratch, count, identity);
}
template<int R, int C, int Kind>
__mlu_func__ void intent_reduce_row_extrema(float *out, const float *input, float *scratch,
                                          int64_t count, float identity) {
  static_assert(R > 1 && C > 0 && C < 65536, "row pooling requires bounded local rows");
  if (count == 0) { __bang_write_value(out, R, identity); return; }
  // Workspace contains R*C comparison values followed by R row flags. The
  // flags remain live while exceptional rows reuse the comparison workspace.
  float *valid = scratch + R * C;
  if (R >= 32 && R % 32 == 0) {
    // Put independent rows in pool channels so they reduce in parallel.
    __bang_transpose(scratch, input, R, C);
    if (Kind == 7 || Kind == 9) __bang_maxpool(out, scratch, R, count, 1, count, 1, 1, 1);
    else __bang_minpool(out, scratch, R, count, 1, count, 1, 1, 1);
    __bang_eq(scratch, scratch, scratch, R * C);
    __bang_minpool(valid, scratch, R, count, 1, count, 1, 1, 1);
  } else {
    __bang_eq(scratch, input, input, R * C);
    __bang_minpool(valid, scratch, 1, R, C, 1, count, C, 1);
    if (Kind == 7 || Kind == 9) __bang_maxpool(out, input, 1, R, C, 1, count, C, 1);
    else __bang_minpool(out, input, 1, R, C, 1, count, C, 1);
  }
  __sync();
  if (R >= 32 && ((Kind == 7 || Kind == 9) ? identity == -INFINITY : identity == INFINITY)) {
    auto *bits = reinterpret_cast<uint32_t *>(scratch);
    __bang_band_scalar(bits, reinterpret_cast<const uint32_t *>(out), 0x7f800000u, R);
    __bang_ge_scalar(bits, bits, 1u, R);
    __bang_mul_scalar(bits, bits, 0x3f800000u, R);
    __bang_mul(scratch, scratch, valid, R);
    __bang_argmin(scratch, scratch, R);
    __sync();
    if (intent_read_local(scratch) == 1.0f) return;
  }
  for (int row = 0; row < R; ++row) {
    union { float value; uint32_t bits; } reduced;
    reduced.value = intent_read_local(out + row);
    if (intent_read_local(valid + row) == 1.0f && (reduced.bits & 0x7f800000u)) {
      float value;
      if (Kind == 7) value = isnan(identity) ? NAN : fmaxf(identity, reduced.value);
      if (Kind == 8) value = isnan(identity) ? NAN : fminf(identity, reduced.value);
      if (Kind == 9) value = fmaxf(identity, reduced.value);
      if (Kind == 10) value = fminf(identity, reduced.value);
      intent_write_local(out + row, value);
    } else {
      __sync();
      intent_reduce<C, Kind>(out + row, input + row * C, scratch, count, identity);
      __sync();
    }
  }
}
)BANG";
