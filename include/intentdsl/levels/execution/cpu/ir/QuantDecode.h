#pragma once
#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/StorageLayout.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include <cctype>
#include <string>

namespace intentdsl::cpu {

inline std::string sanitizeQuantIdent(llvm::StringRef name) {
  std::string out;
  out.reserve(name.size());
  for (char ch : name)
    out.push_back(std::isalnum(static_cast<unsigned char>(ch)) ? ch : '_');
  return out;
}

inline void emitPackTransposeU16HelperC(compilerinfra::PrintContext &c) {
  c.line("#ifndef INTENT_PACK_TRANSPOSE_U16_DEFINED");
  c.line("#define INTENT_PACK_TRANSPOSE_U16_DEFINED");
  c.line("static inline void pack_transpose_u16_panel(uint16_t *out, const uint16_t *src, int64_t K, int64_t valid, int64_t nr, int64_t ldn) {");
  c.line("  int64_t jb = 0;");
  c.line("#if defined(__x86_64__)");
  c.line("  for (; jb + 8 <= valid; jb += 8) {");
  c.line("    int64_t kb = 0;");
  c.line("    for (; kb + 8 <= K; kb += 8) {");
  c.line("      __m128i r0 = _mm_loadu_si128((const __m128i*)(src + (jb + 0)*ldn + kb));");
  c.line("      __m128i r1 = _mm_loadu_si128((const __m128i*)(src + (jb + 1)*ldn + kb));");
  c.line("      __m128i r2 = _mm_loadu_si128((const __m128i*)(src + (jb + 2)*ldn + kb));");
  c.line("      __m128i r3 = _mm_loadu_si128((const __m128i*)(src + (jb + 3)*ldn + kb));");
  c.line("      __m128i r4 = _mm_loadu_si128((const __m128i*)(src + (jb + 4)*ldn + kb));");
  c.line("      __m128i r5 = _mm_loadu_si128((const __m128i*)(src + (jb + 5)*ldn + kb));");
  c.line("      __m128i r6 = _mm_loadu_si128((const __m128i*)(src + (jb + 6)*ldn + kb));");
  c.line("      __m128i r7 = _mm_loadu_si128((const __m128i*)(src + (jb + 7)*ldn + kb));");
  c.line("      __m128i t0 = _mm_unpacklo_epi16(r0, r1), t1 = _mm_unpackhi_epi16(r0, r1);");
  c.line("      __m128i t2 = _mm_unpacklo_epi16(r2, r3), t3 = _mm_unpackhi_epi16(r2, r3);");
  c.line("      __m128i t4 = _mm_unpacklo_epi16(r4, r5), t5 = _mm_unpackhi_epi16(r4, r5);");
  c.line("      __m128i t6 = _mm_unpacklo_epi16(r6, r7), t7 = _mm_unpackhi_epi16(r6, r7);");
  c.line("      __m128i u0 = _mm_unpacklo_epi32(t0, t2), u1 = _mm_unpackhi_epi32(t0, t2);");
  c.line("      __m128i u2 = _mm_unpacklo_epi32(t1, t3), u3 = _mm_unpackhi_epi32(t1, t3);");
  c.line("      __m128i u4 = _mm_unpacklo_epi32(t4, t6), u5 = _mm_unpackhi_epi32(t4, t6);");
  c.line("      __m128i u6 = _mm_unpacklo_epi32(t5, t7), u7 = _mm_unpackhi_epi32(t5, t7);");
  c.line("      _mm_storeu_si128((__m128i*)(out + (kb + 0)*nr + jb), _mm_unpacklo_epi64(u0, u4));");
  c.line("      _mm_storeu_si128((__m128i*)(out + (kb + 1)*nr + jb), _mm_unpackhi_epi64(u0, u4));");
  c.line("      _mm_storeu_si128((__m128i*)(out + (kb + 2)*nr + jb), _mm_unpacklo_epi64(u1, u5));");
  c.line("      _mm_storeu_si128((__m128i*)(out + (kb + 3)*nr + jb), _mm_unpackhi_epi64(u1, u5));");
  c.line("      _mm_storeu_si128((__m128i*)(out + (kb + 4)*nr + jb), _mm_unpacklo_epi64(u2, u6));");
  c.line("      _mm_storeu_si128((__m128i*)(out + (kb + 5)*nr + jb), _mm_unpackhi_epi64(u2, u6));");
  c.line("      _mm_storeu_si128((__m128i*)(out + (kb + 6)*nr + jb), _mm_unpacklo_epi64(u3, u7));");
  c.line("      _mm_storeu_si128((__m128i*)(out + (kb + 7)*nr + jb), _mm_unpackhi_epi64(u3, u7));");
  c.line("    }");
  c.line("    for (; kb < K; ++kb) { uint16_t *dst = out + kb*nr + jb; for (int i = 0; i < 8; ++i) dst[i] = src[(jb + i)*ldn + kb]; }");
  c.line("  }");
  c.line("#endif");
  c.line("  if (jb < valid || valid < nr) {");
  c.line("    for (int64_t k = 0; k < K; ++k) {");
  c.line("      uint16_t *dst = out + k*nr;");
  c.line("      for (int64_t j = jb; j < valid; ++j) dst[j] = src[j*ldn + k];");
  c.line("      if (valid < nr) memset(dst + valid, 0, (size_t)(nr - valid) * sizeof(uint16_t));");
  c.line("    }");
  c.line("  }");
  c.line("}");
  c.line("#endif");
}

inline void emitPackQ8KHelperC(compilerinfra::PrintContext &c) {
  using Q8 = intentdsl::kir::Q8KRecord;
  c.line("struct Record { float d; int8_t q[" + std::to_string(Q8::elements) +
         "]; int16_t sums[" + std::to_string(Q8::elements / Q8::sumGroup) +
         "]; };");
  c.line("_Static_assert(sizeof(struct Record)==" + std::to_string(Q8::bytes) +
         ", \"Q8_K record ABI\");");
  c.line("#if defined(__x86_64__)");
  c.line("__attribute__((target(\"avx512f,avx512bw\")))");
  c.line("static void pack_q8_record_avx512(struct Record *out, const float *x) {");
  c.line("  __m512 vmax = _mm512_setzero_ps();");
  c.line("  for (int g = 0; g < 16; ++g) vmax = _mm512_max_ps(vmax, _mm512_abs_ps(_mm512_loadu_ps(x + 16 * g)));");
  c.line("  float max_abs = _mm512_reduce_max_ps(vmax);");
  c.line("  if (max_abs == 0.0f) { out->d = 0.0f; memset(out->q, 0, 256); memset(out->sums, 0, 32); return; }");
  c.line("  float extreme = 0.0f;");
  c.line("  const __m512 vtarget = _mm512_set1_ps(max_abs);");
  c.line("  for (int g = 0; g < 16; ++g) {");
  c.line("    __mmask16 m = _mm512_cmp_ps_mask(_mm512_abs_ps(_mm512_loadu_ps(x + 16 * g)), vtarget, _CMP_EQ_OQ);");
  c.line("    if (m != 0) { extreme = x[16 * g + __builtin_ctz((unsigned)m)]; break; }");
  c.line("  }");
  c.line("  const float inv = -127.0f / extreme;");
  c.line("  out->d = 1.0f / inv;");
  c.line("  const __m512 vinv = _mm512_set1_ps(inv);");
  c.line("  const __m512i v127 = _mm512_set1_epi32(127);");
  c.line("  for (int g = 0; g < 16; ++g) {");
  c.line("    __m512 vx = _mm512_loadu_ps(x + 16 * g);");
  c.line("    __m512i vq = _mm512_min_epi32(_mm512_cvt_roundps_epi32(_mm512_mul_ps(vx, vinv), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), v127);");
  c.line("    _mm_storeu_si128((__m128i *)(out->q + 16 * g), _mm512_cvtsepi32_epi8(vq));");
  c.line("    out->sums[g] = (int16_t)_mm512_reduce_add_epi32(vq);");
  c.line("  }");
  c.line("}");
  c.line("#elif defined(__riscv_vector)");
  c.line("static inline void pack_q8_record_rvv(struct Record *out, const float *x) {");
  c.line("  float max_abs = 0.0f;");
  c.line("  for (size_t i = 0; i < 256; ) {");
  c.line("    size_t vl = __riscv_vsetvl_e32m4(256 - i);");
  c.line("    vfloat32m4_t va = __riscv_vfabs_v_f32m4(__riscv_vle32_v_f32m4(x + i, vl), vl);");
  c.line("    vfloat32m1_t vseed = __riscv_vfmv_s_f_f32m1(max_abs, 1);");
  c.line("    max_abs = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m4_f32m1(va, vseed, vl));");
  c.line("    i += vl;");
  c.line("  }");
  c.line("  if (max_abs == 0.0f) { out->d = 0.0f; memset(out->q, 0, 256); memset(out->sums, 0, 32); return; }");
  c.line("  float extreme = 0.0f;");
  c.line("  for (size_t i = 0; i < 256; ) {");
  c.line("    size_t vl = __riscv_vsetvl_e32m4(256 - i);");
  c.line("    vfloat32m4_t va = __riscv_vfabs_v_f32m4(__riscv_vle32_v_f32m4(x + i, vl), vl);");
  c.line("    vbool8_t m = __riscv_vmfeq_vf_f32m4_b8(va, max_abs, vl);");
  c.line("    long first = __riscv_vfirst_m_b8(m, vl);");
  c.line("    if (first >= 0) { extreme = x[i + (size_t)first]; break; }");
  c.line("    i += vl;");
  c.line("  }");
  c.line("  const float inv = -127.0f / extreme;");
  c.line("  out->d = 1.0f / inv;");
  c.line("  for (int g = 0; g < 16; ++g) {");
  c.line("    int sum = 0;");
  c.line("    for (size_t k = 0; k < 16; ) {");
  c.line("      size_t vl = __riscv_vsetvl_e32m4(16 - k);");
  c.line("      vfloat32m4_t vx = __riscv_vle32_v_f32m4(x + 16 * g + k, vl);");
  c.line("      vint32m4_t vq = __riscv_vfcvt_x_f_v_i32m4_rm(__riscv_vfmul_vf_f32m4(vx, inv, vl), __RISCV_FRM_RNE, vl);");
  c.line("      vq = __riscv_vmin_vx_i32m4(vq, 127, vl);");
  c.line("      vq = __riscv_vmax_vx_i32m4(vq, -128, vl);");
  c.line("      vint16m2_t vq16 = __riscv_vnclip_wx_i16m2(vq, 0, __RISCV_VXRM_RNE, vl);");
  c.line("      vint8m1_t vq8 = __riscv_vnclip_wx_i8m1(vq16, 0, __RISCV_VXRM_RNE, vl);");
  c.line("      __riscv_vse8_v_i8m1(out->q + 16 * g + k, vq8, vl);");
  c.line("      vint32m1_t vzero32 = __riscv_vmv_s_x_i32m1(0, 1);");
  c.line("      sum += (int)__riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m4_i32m1(vq, vzero32, vl));");
  c.line("      k += vl;");
  c.line("    }");
  c.line("    out->sums[g] = (int16_t)sum;");
  c.line("  }");
  c.line("}");
  c.line("#endif");
  c.line("static inline void pack_q8_record(struct Record *out, const float *x, int64_t stride) {");
  c.line("#if defined(__x86_64__)");
  c.line("  if (stride == 1 && __builtin_cpu_supports(\"avx512f\") && __builtin_cpu_supports(\"avx512bw\")) { pack_q8_record_avx512(out, x); return; }");
  c.line("#elif defined(__riscv_vector)");
  c.line("  if (stride == 1) { pack_q8_record_rvv(out, x); return; }");
  c.line("#endif");
  c.line("  float extreme=0.0f;");
  c.line("  for (int64_t k=0; k<256; ++k) { float v=x[k*stride]; if (fabsf(v)>fabsf(extreme)) extreme=v; }");
  c.line("  float inv=extreme==0.0f ? 0.0f : -127.0f/extreme;");
  c.line("  out->d=extreme==0.0f ? 0.0f : 1.0f/inv;");
  c.line("  for (int group=0; group<16; ++group) {");
  c.line("    int sum=0;");
  c.line("    for (int64_t k=16*group; k<16*group+16; ++k) {");
  c.line("      float v=x[k*stride]*inv; int q=(int)v; float fraction=fabsf(v-(float)q);");
  c.line("      if (fraction>0.5f || (fraction==0.5f && (q&1))) q+=v<0 ? -1 : 1;");
  c.line("      out->q[k]=(int8_t)(q>127 ? 127 : q); sum+=out->q[k];");
  c.line("    }");
  c.line("    out->sums[group]=(int16_t)sum;");
  c.line("  }");
  c.line("}");
}

inline void emitQuantHostHelpersC(
    llvm::ArrayRef<const intentdsl::kir::QuantFormatSpec *> specs,
    compilerinfra::PrintContext &c) {
  if (specs.empty())
    return;
  c.line("#ifndef INTENT_QUANT_COMMON_HELPERS_DEFINED");
  c.line("#define INTENT_QUANT_COMMON_HELPERS_DEFINED");
  c.line("static inline float intent_qf16_to_f32(uint16_t h) {");
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
  c.line("static inline float intent_qbf16_to_f32(uint16_t v) { uint32_t b=(uint32_t)v<<16; float f; memcpy(&f,&b,4); return f; }");
  c.line("static inline float intent_qe2m1_val(unsigned c) {");
  c.line("  static const float k_e2m1[16] = {0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f,-0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f};");
  c.line("  return k_e2m1[c & 15u];");
  c.line("}");
  c.line("static inline float intent_qread_fview(const void *base, int64_t idx, int dt) {");
  c.line("  if (dt == 1) return intent_qf16_to_f32(((const uint16_t*)base)[idx]);");
  c.line("  if (dt == 2) return intent_qbf16_to_f32(((const uint16_t*)base)[idx]);");
  c.line("  return ((const float*)base)[idx];");
  c.line("}");
  c.line("static inline uint64_t intent_qread_word(const void *base, int64_t idx, int word_bits) {");
  c.line("  if (word_bits == 8) return ((const uint8_t*)base)[idx];");
  c.line("  if (word_bits == 16) return ((const uint16_t*)base)[idx];");
  c.line("  if (word_bits == 32) return ((const uint32_t*)base)[idx];");
  c.line("  return ((const uint64_t*)base)[idx];");
  c.line("}");
  c.line("#endif");

  for (const auto *spec : specs) {
    if (!spec)
      continue;
    std::string id = sanitizeQuantIdent(spec->name);
    const unsigned codeMask = (1u << spec->packBits) - 1u;
    const unsigned numCodes = 1u << (spec->packBits <= 8 ? spec->packBits : 8);
    c.line("static _Alignas(64) float g_qcode_" + id + "[256];");
    c.line("static _Alignas(64) float g_qscale_" + id + "[256];");
    c.line("static int g_qready_" + id + " = 0;");
    c.line("static void ensure_quant_tables_" + id + "(void) {");
    c.line("  if (__atomic_load_n(&g_qready_" + id + ", __ATOMIC_ACQUIRE)) return;");
    c.line("  for (unsigned u = 0; u < 256u; ++u) {");
    if (spec->valueCode == kir::QuantValueCode::Fp4E2M1) {
      c.line("    g_qcode_" + id + "[u] = intent_qe2m1_val(u & " +
             std::to_string(codeMask) + "u);");
    } else {
      c.line("    g_qcode_" + id + "[u] = (float)((int)(u & " +
             std::to_string(codeMask) + "u) - " +
             std::to_string(spec->zeroBias) + ");");
    }
    if (spec->scaleCode == kir::QuantScaleCode::E8M0) {
      if (spec->scaleNan >= 0) {
        c.line("    if ((int)u == " + std::to_string(spec->scaleNan) +
               ") { uint32_t nb = 0x7fc00000u; memcpy(&g_qscale_" + id +
               "[u], &nb, 4); } else g_qscale_" + id +
               "[u] = ldexpf(1.0f, (int)u - " +
               std::to_string(spec->scaleBias) + ");");
      } else {
        c.line("    g_qscale_" + id + "[u] = ldexpf(1.0f, (int)u - " +
               std::to_string(spec->scaleBias) + ");");
      }
    } else {
      c.line("    g_qscale_" + id + "[u] = 1.0f;");
    }
    c.line("  }");
    (void)numCodes;
    c.line("  __atomic_store_n(&g_qready_" + id + ", 1, __ATOMIC_RELEASE);");
    c.line("}");

    if (spec->roleLayout == kir::QuantRoleLayout::BlockScaledPair) {
      const bool fast4x2E8M0 =
          spec->packBits == 4 && spec->packFactor == 2 &&
          spec->scaleCode == kir::QuantScaleCode::E8M0 &&
          spec->laneBitShifts[0] == 0 && spec->laneBitShifts[1] == 4;
      llvm::SmallVector<std::string, 8> shStrs;
      for (int ln = 0; ln < spec->packFactor; ++ln)
        shStrs.push_back(std::to_string(spec->laneBitShifts[ln]));
      std::string shInit = "{" + llvm::join(shStrs, ",") + "}";

      if (fast4x2E8M0 && spec->recordBytes == 16 && spec->blockElements == 32) {
        c.line("#if defined(__x86_64__)");
        c.line("__attribute__((target(\"avx512f\")))");
        c.line("static void intent_decode_block_lhs_row_avx512_" + id +
               "(float *dst_r, const uint8_t *src_r, int64_t stA1, const uint8_t *sc_u8, int64_t stS1, int64_t G_cnt) {");
        c.line("  const __m512 lut = _mm512_loadu_ps(g_qcode_" + id + ");");
        c.line("  const __m128i m0f = _mm_set1_epi8(0x0f);");
        c.line("  for (int64_t g = 0; g < G_cnt; ++g) {");
        c.line("    const __m512 vsc = _mm512_set1_ps(g_qscale_" + id + "[sc_u8[g * stS1]]);");
        c.line("    const __m128i raw = _mm_loadu_si128((const __m128i*)(src_r + g * stA1));");
        c.line("    const __m128i lo = _mm_and_si128(raw, m0f);");
        c.line("    const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), m0f);");
        c.line("    const __m128i p0 = _mm_unpacklo_epi8(lo, hi);");
        c.line("    const __m128i p1 = _mm_unpackhi_epi8(lo, hi);");
        c.line("    _mm512_storeu_ps(dst_r + g * 32, _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_cvtepu8_epi32(p0), lut), vsc));");
        c.line("    _mm512_storeu_ps(dst_r + g * 32 + 16, _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_cvtepu8_epi32(p1), lut), vsc));");
        c.line("  }");
        c.line("}");
        c.line("__attribute__((target(\"avx512f\")))");
        c.line("static void intent_decode_block_rhs_panel_avx512_" + id +
               "(float *panel_out, const uint8_t *src_b, int64_t stB0, int64_t stB1, const uint8_t *sc_u8, int64_t stS0, int64_t stS1, int64_t colAx, int64_t c0, int64_t nr, int64_t G_cnt) {");
        c.line("  const __m512 lut = _mm512_loadu_ps(g_qcode_" + id + ");");
        c.line("  const __m128i m0f = _mm_set1_epi8(0x0f);");
        c.line("  for (int64_t g = 0; g < G_cnt; ++g) {");
        c.line("    for (int64_t jc = 0; jc < nr; jc += 16) {");
        c.line("      float sc_tmp[16];");
        c.line("      for (int l = 0; l < 16; ++l) { int64_t c = c0 + jc + l; sc_tmp[l] = g_qscale_" +
               id + "[sc_u8[colAx == 0 ? (c * stS0 + g * stS1) : (g * stS0 + c * stS1)]]; }");
        c.line("      const __m512 vsc = _mm512_loadu_ps(sc_tmp);");
        c.line("      const uint8_t *b_g = src_b + g * stB0 + (c0 + jc);");
        c.line("      float *out_g = panel_out + g * 32 * nr + jc;");
        c.line("      for (int64_t t = 0; t < 16; ++t) {");
        c.line("        const __m128i raw = _mm_loadu_si128((const __m128i*)(b_g + t * stB1));");
        c.line("        const __m128i lo = _mm_and_si128(raw, m0f);");
        c.line("        const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), m0f);");
        c.line("        _mm512_storeu_ps(out_g + (2 * t) * nr, _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_cvtepu8_epi32(lo), lut), vsc));");
        c.line("        _mm512_storeu_ps(out_g + (2 * t + 1) * nr, _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_cvtepu8_epi32(hi), lut), vsc));");
        c.line("      }");
        c.line("    }");
        c.line("  }");
        c.line("}");
        c.line("#endif");
      }

      c.line("static inline void intent_decode_block_lhs_row_" + id +
             "(float *dst_r, const uint8_t *src_r, int64_t stA1, int64_t stA2, const void *sc_r, int64_t stS1, int sc_dt, int64_t K) {");
      c.line("  const int64_t G_cnt = K / " +
             std::to_string(spec->blockElements) + "LL;");
      if (fast4x2E8M0 && spec->recordBytes == 16 && spec->blockElements == 32) {
        c.line("#if defined(__x86_64__)");
        c.line("  if (stA2 == 1 && __builtin_cpu_supports(\"avx512f\")) { intent_decode_block_lhs_row_avx512_" +
               id + "(dst_r, src_r, stA1, (const uint8_t*)sc_r, stS1, G_cnt); return; }");
        c.line("#endif");
      }
      c.line("  static const int k_sh[" + std::to_string(spec->packFactor) +
             "] = " + shInit + ";");
      c.line("  for (int64_t g = 0; g < G_cnt; ++g) {");
      if (spec->scaleCode == kir::QuantScaleCode::E8M0)
        c.line("    const float sc = g_qscale_" + id +
               "[((const uint8_t*)sc_r)[g * stS1]];");
      else if (spec->scaleCode == kir::QuantScaleCode::FloatView)
        c.line("    const float sc = intent_qread_fview(sc_r, g * stS1, sc_dt);");
      else
        c.line("    const float sc = 1.0f;");
      c.line("    float *dst_g = dst_r + g * " +
             std::to_string(spec->blockElements) + "LL;");
      c.line("    const uint8_t *src_g = src_r + g * stA1;");
      c.line("    for (int64_t t = 0; t < " +
             std::to_string(spec->recordBytes) + "LL; ++t) {");
      c.line("      const unsigned raw = src_g[t * stA2];");
      c.line("      for (int ln = 0; ln < " +
             std::to_string(spec->packFactor) + "; ++ln)");
      c.line("        dst_g[t * " + std::to_string(spec->packFactor) +
             "LL + ln] = g_qcode_" + id + "[(raw >> k_sh[ln]) & " +
             std::to_string(codeMask) + "u] * sc;");
      c.line("    }");
      c.line("  }");
      c.line("}");

      c.line("static inline void intent_decode_block_rhs_panel_" + id +
             "(float *panel_out, const uint8_t *src_b, int64_t stB0, int64_t stB1, int64_t stB2, const void *sc_b, int64_t stS0, int64_t stS1, int sc_dt, int64_t colAx, int64_t c0, int64_t valid, int64_t nr, int64_t K) {");
      c.line("  const int64_t G_cnt = K / " +
             std::to_string(spec->blockElements) + "LL;");
      if (fast4x2E8M0 && spec->recordBytes == 16 && spec->blockElements == 32) {
        c.line("#if defined(__x86_64__)");
        c.line("  if (stB2 == 1 && valid == nr && (nr % 16 == 0) && __builtin_cpu_supports(\"avx512f\")) { intent_decode_block_rhs_panel_avx512_" +
               id + "(panel_out, src_b, stB0, stB1, (const uint8_t*)sc_b, stS0, stS1, colAx, c0, nr, G_cnt); return; }");
        c.line("#endif");
      }
      c.line("  static const int k_sh[" + std::to_string(spec->packFactor) +
             "] = " + shInit + ";");
      c.line("  for (int64_t g = 0; g < G_cnt; ++g) {");
      c.line("    float sc_row[256];");
      c.line("    for (int64_t jc = 0; jc < valid; ++jc) {");
      c.line("      const int64_t c = c0 + jc;");
      c.line("      const int64_t sidx = colAx == 0 ? (c * stS0 + g * stS1) : (g * stS0 + c * stS1);");
      if (spec->scaleCode == kir::QuantScaleCode::E8M0)
        c.line("      sc_row[jc] = g_qscale_" + id +
               "[((const uint8_t*)sc_b)[sidx]];");
      else if (spec->scaleCode == kir::QuantScaleCode::FloatView)
        c.line("      sc_row[jc] = intent_qread_fview(sc_b, sidx, sc_dt);");
      else
        c.line("      sc_row[jc] = 1.0f;");
      c.line("    }");
      c.line("    for (int64_t t = 0; t < " +
             std::to_string(spec->recordBytes) + "LL; ++t) {");
      c.line("      const uint8_t *b_gt = src_b + g * stB0 + t * stB1 + c0 * stB2;");
      c.line("      for (int ln = 0; ln < " +
             std::to_string(spec->packFactor) + "; ++ln) {");
      c.line("        float *dst_k = panel_out + (g * " +
             std::to_string(spec->blockElements) + "LL + t * " +
             std::to_string(spec->packFactor) + "LL + ln) * nr;");
      c.line("        const int sh = k_sh[ln];");
      c.line("        for (int64_t jc = 0; jc < valid; ++jc)");
      c.line("          dst_k[jc] = g_qcode_" + id +
             "[(b_gt[jc * stB2] >> sh) & " + std::to_string(codeMask) +
             "u] * sc_row[jc];");
      c.line("        if (valid < nr) memset(dst_k + valid, 0, (size_t)(nr - valid) * sizeof(float));");
      c.line("      }");
      c.line("    }");
      c.line("  }");
      c.line("}");
    } else if (spec->roleLayout == kir::QuantRoleLayout::GroupedWeightRhs) {
      const int wordBits = spec->packFactor * spec->packBits;
      const bool fast8x4 = wordBits == 32 && spec->packFactor == 8 &&
                           spec->packBits == 4;
      llvm::SmallVector<std::string, 8> shStrs;
      for (int ln = 0; ln < spec->packFactor; ++ln)
        shStrs.push_back(std::to_string(spec->laneBitShifts[ln]));
      std::string shInit = "{" + llvm::join(shStrs, ",") + "}";

      if (fast8x4) {
        std::string sh16 = llvm::join(shStrs, ",") + "," + llvm::join(shStrs, ",");
        c.line("#if defined(__x86_64__)");
        c.line("__attribute__((target(\"avx512f,f16c\")))");
        c.line("static void intent_decode_grouped_rhs_panel_avx512_" + id +
               "(float *panel_out, const uint32_t *qw_u32, int64_t stQw0, const void *sc_base, int64_t stSc0, int64_t stSc1, int sc_dt, const uint32_t *qz_u32, int64_t stQz0, int64_t stQz1, int64_t c0, int64_t nr, int64_t K) {");
        c.line("  const __m512 lut = _mm512_loadu_ps(g_qcode_" + id + ");");
        c.line("  const __m512i vsh = _mm512_setr_epi32(" + sh16 + ");");
        c.line("  const __m512i vm = _mm512_set1_epi32(15);");
        c.line("  const int64_t G_cnt = K / " +
               std::to_string(spec->blockElements) + "LL;");
        c.line("  for (int64_t g = 0; g < G_cnt; ++g) {");
        c.line("    const int64_t k0 = g * " +
               std::to_string(spec->blockElements) + "LL;");
        c.line("    for (int64_t jc = 0; jc < nr; jc += 16) {");
        c.line("      const int64_t c = c0 + jc, np = c / 8;");
        c.line("      __m512 vsc;");
        c.line("      if (sc_dt == 1 && stSc1 == 1) vsc = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)((const uint16_t*)sc_base + g * stSc0 + c)));");
        c.line("      else if (sc_dt == 0 && stSc1 == 1) vsc = _mm512_loadu_ps((const float*)sc_base + g * stSc0 + c);");
        c.line("      else { float stmp[16]; for (int l = 0; l < 16; ++l) stmp[l] = intent_qread_fview(sc_base, g * stSc0 + (c + l) * stSc1, sc_dt); vsc = _mm512_loadu_ps(stmp); }");
        if (spec->hasZeroView) {
          c.line("      const uint32_t z0 = qz_u32[g * stQz0 + np * stQz1], z1 = qz_u32[g * stQz0 + (np + 1) * stQz1];");
          c.line("      const __m512i vz_raw = _mm512_inserti64x4(_mm512_set1_epi32((int)z0), _mm256_set1_epi32((int)z1), 1);");
          c.line("      const __m512 vzp = _mm512_cvtepi32_ps(_mm512_and_si512(_mm512_srlv_epi32(vz_raw, vsh), vm));");
        } else {
          c.line("      const __m512 vzp = _mm512_setzero_ps();");
        }
        c.line("      const uint32_t *qw_col = qw_u32 + np;");
        c.line("      float *dst_col = panel_out + jc;");
        c.line("      for (int64_t tk = 0; tk < " +
               std::to_string(spec->blockElements) + "LL; ++tk) {");
        c.line("        const uint32_t *qw_r = qw_col + (k0 + tk) * stQw0;");
        c.line("        const __m512i vw_raw = _mm512_inserti64x4(_mm512_set1_epi32((int)qw_r[0]), _mm256_set1_epi32((int)qw_r[1]), 1);");
        c.line("        const __m512i vc = _mm512_and_si512(_mm512_srlv_epi32(vw_raw, vsh), vm);");
        c.line("        const __m512 vwf = _mm512_permutexvar_ps(vc, lut);");
        c.line("        _mm512_storeu_ps(dst_col + (k0 + tk) * nr, _mm512_mul_ps(_mm512_sub_ps(vwf, vzp), vsc));");
        c.line("      }");
        c.line("    }");
        c.line("  }");
        c.line("}");
        c.line("#endif");
      }

      c.line("static inline void intent_decode_grouped_rhs_panel_" + id +
             "(float *panel_out, const void *qw_base, int64_t stQw0, int64_t stQw1, const void *sc_base, int64_t stSc0, int64_t stSc1, int sc_dt, const void *qz_base, int64_t stQz0, int64_t stQz1, int64_t c0, int64_t valid, int64_t nr, int64_t K) {");
      if (fast8x4) {
        c.line("#if defined(__x86_64__)");
        c.line("  if (stQw1 == 1 && (c0 % 8 == 0) && valid == nr && (nr % 16 == 0) && __builtin_cpu_supports(\"avx512f\") && __builtin_cpu_supports(\"f16c\")) {");
        c.line("    intent_decode_grouped_rhs_panel_avx512_" + id +
               "(panel_out, (const uint32_t*)qw_base, stQw0, sc_base, stSc0, stSc1, sc_dt, (const uint32_t*)qz_base, stQz0, stQz1, c0, nr, K);");
        c.line("    return;");
        c.line("  }");
        c.line("#endif");
      }
      c.line("  static const int k_sh[" + std::to_string(spec->packFactor) +
             "] = " + shInit + ";");
      c.line("  const int64_t G_cnt = K / " +
             std::to_string(spec->blockElements) + "LL;");
      c.line("  for (int64_t g = 0; g < G_cnt; ++g) {");
      c.line("    float sc_row[256], zp_row[256];");
      c.line("    for (int64_t jc = 0; jc < valid; ++jc) {");
      c.line("      const int64_t c = c0 + jc, np = c / " +
             std::to_string(spec->packFactor) + "LL, ln = c % " +
             std::to_string(spec->packFactor) + "LL;");
      c.line("      sc_row[jc] = intent_qread_fview(sc_base, g * stSc0 + c * stSc1, sc_dt);");
      if (spec->hasZeroView) {
        c.line("      uint64_t zw = intent_qread_word(qz_base, g * stQz0 + np * stQz1, " +
               std::to_string(wordBits) + ");");
        c.line("      zp_row[jc] = (float)((unsigned)((zw >> k_sh[ln]) & " +
               std::to_string(codeMask) + "u));");
      } else {
        c.line("      zp_row[jc] = 0.0f;");
      }
      c.line("    }");
      c.line("    for (int64_t tk = 0; tk < " +
             std::to_string(spec->blockElements) + "LL; ++tk) {");
      c.line("      const int64_t k = g * " +
             std::to_string(spec->blockElements) + "LL + tk;");
      c.line("      float *dst_k = panel_out + k * nr;");
      c.line("      for (int64_t jc = 0; jc < valid; ++jc) {");
      c.line("        const int64_t c = c0 + jc, np = c / " +
             std::to_string(spec->packFactor) + "LL, ln = c % " +
             std::to_string(spec->packFactor) + "LL;");
      c.line("        uint64_t qw = intent_qread_word(qw_base, k * stQw0 + np * stQw1, " +
             std::to_string(wordBits) + ");");
      c.line("        unsigned code = (unsigned)((qw >> k_sh[ln]) & " +
             std::to_string(codeMask) + "u);");
      c.line("        dst_k[jc] = (g_qcode_" + id + "[code] - zp_row[jc]) * sc_row[jc];");
      c.line("      }");
      c.line("      if (valid < nr) memset(dst_k + valid, 0, (size_t)(nr - valid) * sizeof(float));");
      c.line("    }");
      c.line("  }");
      c.line("}");
    }
  }
}

inline void emitMojoQuantRecordSuperblock(
    const intentdsl::kir::QuantFormatSpec &spec, llvm::StringRef wpS,
    llvm::StringRef qpS, llvm::StringRef wbS, llvm::StringRef gS,
    llvm::StringRef resS, int64_t bs, bool useVnni,
    compilerinfra::PrintContext &c) {
  using Q8 = intentdsl::kir::Q8KRecord;
  std::string wp = wpS.str(), qp = qpS.str(), wb = wbS.str(), g = gS.str(),
              res = resS.str();
  auto wByte = [&](const std::string &off) {
    return "UInt16(" + wp + "[unsafe_offset=" + wb + "+(" + off + ")*" +
           std::to_string(bs) + "])";
  };

  if (spec.scaleCode == kir::QuantScaleCode::LeHalfAtOffset &&
      spec.superScaleOffset >= 0) {
    c.line("var " + res + "_d = Float32(bitcast[DType.float16](" +
           wByte(std::to_string(spec.superScaleOffset)) + " | (" +
           wByte(std::to_string(spec.superScaleOffset + 1)) + " << 8)))");
  } else {
    c.line("var " + res + "_d = Float32(1.0)");
  }

  if (spec.q8kActivation) {
    c.line("var " + res + "_ds = " + qp +
           ".bitcast[Float32]()[unsafe_offset=" + g + "*" +
           std::to_string(Q8::bytes / 4) + "]");
  }

  auto emitLoad32 = [&](const std::string &varName, const std::string &offExpr) {
    if (bs == 1) {
      c.line("var " + varName + " = " + wp + ".unsafe_offset(" + wb + "+" +
             offExpr + ").unsafe_load[width=32]()");
    } else {
      c.line("var " + varName + " = SIMD[DType.uint8,32](0)");
      c.line("comptime for _k in range(32):");
      c.indent++;
      c.line(varName + "[_k] = " + wp + "[unsafe_offset=" + wb + "+(" +
             offExpr + "+_k)*" + std::to_string(bs) + "]");
      c.indent--;
    }
  };

  if (spec.q8kActivation && spec.blockElements == 256 &&
      spec.subBlockElements == 16 &&
      spec.valueCode == kir::QuantValueCode::UnsignedAffine) {
    if (useVnni && spec.packBits <= 7) {
      bool canZmmPair = (bs == 1);
      for (int p = 0; p < spec.numBitPlanes; ++p) {
        if (spec.planes[p].quadByteStep != 32 &&
            spec.planes[p].quadByteStep != 0)
          canZmmPair = false;
      }
      if (canZmmPair) {
        c.line("var " + res + "_accS = SIMD[DType.int32,16](0)");
        if (spec.subScaleOffset >= 0) {
          c.line("var " + res + "_sc16 = " + wp + ".unsafe_offset(" + wb + "+" +
                 std::to_string(spec.subScaleOffset) +
                 ").bitcast[Int8]().unsafe_load[width=16]().cast[DType.int32]()");
        } else {
          c.line("var " + res + "_sc16 = SIMD[DType.int32,16](1)");
        }
        c.line("var " + res + "_bs16 = " + qp + ".bitcast[Int16]().unsafe_offset(" +
               g + "*" + std::to_string(Q8::bytes / 2) + "+" +
               std::to_string(Q8::sums / 2) +
               ").unsafe_load[width=16]().cast[DType.int32]()");
        c.line("comptime for _h in range(2):");
        c.indent++;
        std::string expr01, expr23;
        for (int p = 0; p < spec.numBitPlanes; ++p) {
          const auto &pl = spec.planes[p];
          int mask = (1 << pl.bitWidth) - 1;
          std::string rawVar = "_zp_" + std::to_string(p);
          std::string p01, p23;
          if (pl.quadByteStep == 32) {
            c.line("var " + rawVar + " = " + wp + ".unsafe_offset(" + wb + "+" +
                   std::to_string(pl.baseByte) + "+_h*" +
                   std::to_string(pl.halfStride) + ").unsafe_load[width=64]()");
            if (pl.quadShiftMode == 0) {
              p01 = "(" + rawVar + " & " + std::to_string(mask) + ")";
              int sh23 = pl.shiftUnit;
              p23 = sh23 > 0 ? ("((" + rawVar + " >> " + std::to_string(sh23) +
                                ") & " + std::to_string(mask) + ")")
                             : p01;
            } else {
              std::string lo = rawVar + ".slice[32, offset=0]()";
              std::string hi = rawVar + ".slice[32, offset=32]()";
              p01 = "(" + lo + ".join(" + hi + " >> " +
                    std::to_string(pl.shiftUnit) + ") & " +
                    std::to_string(mask) + ")";
              p23 = "((" + lo + " >> " + std::to_string(2 * pl.shiftUnit) +
                    ").join(" + hi + " >> " + std::to_string(3 * pl.shiftUnit) +
                    ") & " + std::to_string(mask) + ")";
            }
          } else {
            c.line("var " + rawVar + " = " + wp + ".unsafe_offset(" + wb + "+" +
                   std::to_string(pl.baseByte) + "+_h*" +
                   std::to_string(pl.halfStride) + ").unsafe_load[width=32]()");
            if (pl.quadShiftMode == 0) {
              std::string j = rawVar + ".join(" + rawVar + ")";
              p01 = "(" + j + " & " + std::to_string(mask) + ")";
              p23 = pl.shiftUnit > 0
                        ? ("((" + j + " >> " + std::to_string(pl.shiftUnit) +
                           ") & " + std::to_string(mask) + ")")
                        : p01;
            } else {
              std::string q1 = pl.shiftUnit > 0
                                   ? ("(" + rawVar + " >> " +
                                      std::to_string(pl.shiftUnit) + ")")
                                   : rawVar;
              std::string q2 = pl.shiftUnit > 0
                                   ? ("(" + rawVar + " >> " +
                                      std::to_string(2 * pl.shiftUnit) + ")")
                                   : rawVar;
              std::string q3 = pl.shiftUnit > 0
                                   ? ("(" + rawVar + " >> " +
                                      std::to_string(3 * pl.shiftUnit) + ")")
                                   : rawVar;
              p01 = "(" + rawVar + ".join(" + q1 + ") & " +
                    std::to_string(mask) + ")";
              p23 = "(" + q2 + ".join(" + q3 + ") & " + std::to_string(mask) +
                    ")";
            }
          }
          if (pl.destBitOffset > 0) {
            p01 = "(" + p01 + " << " + std::to_string(pl.destBitOffset) + ")";
            p23 = "(" + p23 + " << " + std::to_string(pl.destBitOffset) + ")";
          }
          expr01 = expr01.empty() ? p01 : ("(" + expr01 + " | " + p01 + ")");
          expr23 = expr23.empty() ? p23 : ("(" + expr23 + " | " + p23 + ")");
        }
        c.line("var _u01 = " +
               (expr01.empty() ? std::string("SIMD[DType.uint8,64](0)")
                               : expr01));
        c.line("var _u23 = " +
               (expr23.empty() ? std::string("SIMD[DType.uint8,64](0)")
                               : expr23));
        c.line("var _q01 = " + qp + ".unsafe_offset(" + g + "*" +
               std::to_string(Q8::bytes) + "+" + std::to_string(Q8::values) +
               "+_h*128).unsafe_load[width=64]()");
        c.line("var _q23 = " + qp + ".unsafe_offset(" + g + "*" +
               std::to_string(Q8::bytes) + "+" +
               std::to_string(Q8::values + 64) +
               "+_h*128).unsafe_load[width=64]()");
        c.line("var _d01 = llvm_intrinsic[\"llvm.x86.avx512.vpdpbusd.512\",SIMD[DType.int32,16]](SIMD[DType.int32,16](0), _u01, _q01)");
        c.line("var _d23 = llvm_intrinsic[\"llvm.x86.avx512.vpdpbusd.512\",SIMD[DType.int32,16]](SIMD[DType.int32,16](0), _u23, _q23)");
        c.line("var _sch = " + res + "_sc16.slice[8, offset=_h*8]().join(" +
               res + "_sc16.slice[8, offset=_h*8]())");
        c.line("var _sc01 = _sch.shuffle[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3](_sch)");
        c.line("var _sc23 = _sch.shuffle[4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7](_sch)");
        c.line(res + "_accS += _d01 * _sc01 + _d23 * _sc23");
        c.indent--;
        c.line("var " + res + "_tot = (" + res + "_accS - Int32(" +
               std::to_string(spec.zeroBias) + ") * (" + res + "_sc16 * " +
               res + "_bs16)).reduce_add()");
        c.line("var " + res + " = SIMD[DType.float32,1](" + res + "_ds * " +
               res + "_d * Float32(" + res + "_tot))");
        return;
      }
      c.line("var " + res + "_accS = SIMD[DType.int32,16](0)");
      c.line("var " + res + "_accB = SIMD[DType.int32,8](0)");
      c.line("comptime for _h in range(2):");
      c.indent++;
      llvm::SmallVector<std::pair<int, int>, 4> loadedPlanes;
      for (int q = 0; q < 4; ++q) {
        for (int p = 0; p < spec.numBitPlanes; ++p) {
          const auto &pl = spec.planes[p];
          int baseOff = pl.baseByte + (q & 1) * pl.quadByteStep;
          std::pair<int, int> key{p, baseOff};
          if (!llvm::is_contained(loadedPlanes, key)) {
            loadedPlanes.push_back(key);
            std::string rawVar =
                "_rp_" + std::to_string(p) + "_" + std::to_string(baseOff);
            emitLoad32(rawVar, std::to_string(baseOff) + "+_h*" +
                                   std::to_string(pl.halfStride));
          }
        }
      }
      for (int q = 0; q < 4; ++q) {
        std::string uVar = "_u" + std::to_string(q);
        std::string expr;
        for (int p = 0; p < spec.numBitPlanes; ++p) {
          const auto &pl = spec.planes[p];
          int baseOff = pl.baseByte + (q & 1) * pl.quadByteStep;
          int shift = pl.quadShiftMode == 0 ? pl.shiftUnit * (q >> 1)
                                            : pl.shiftUnit * q;
          int mask = (1 << pl.bitWidth) - 1;
          std::string rawVar =
              "_rp_" + std::to_string(p) + "_" + std::to_string(baseOff);
          std::string part = rawVar;
          if (shift > 0)
            part = "(" + part + " >> " + std::to_string(shift) + ")";
          part = "(" + part + " & " + std::to_string(mask) + ")";
          if (pl.destBitOffset > 0)
            part = "(" + part + " << " + std::to_string(pl.destBitOffset) + ")";
          expr = expr.empty() ? part : ("(" + expr + " | " + part + ")");
        }
        c.line("var " + uVar + " = " +
               (expr.empty() ? std::string("SIMD[DType.uint8,32](0)") : expr));
      }
      c.line("var _u01 = _u0.join(_u1)");
      c.line("var _u23 = _u2.join(_u3)");
      c.line("var _q01 = " + qp + ".unsafe_offset(" + g + "*" +
             std::to_string(Q8::bytes) + "+" + std::to_string(Q8::values) +
             "+_h*128).unsafe_load[width=64]()");
      c.line("var _q23 = " + qp + ".unsafe_offset(" + g + "*" +
             std::to_string(Q8::bytes) + "+" +
             std::to_string(Q8::values + 64) +
             "+_h*128).unsafe_load[width=64]()");
      c.line("var _d01 = llvm_intrinsic[\"llvm.x86.avx512.vpdpbusd.512\",SIMD[DType.int32,16]](SIMD[DType.int32,16](0), _u01, _q01)");
      c.line("var _d23 = llvm_intrinsic[\"llvm.x86.avx512.vpdpbusd.512\",SIMD[DType.int32,16]](SIMD[DType.int32,16](0), _u23, _q23)");
      if (spec.subScaleOffset >= 0) {
        if (bs == 1) {
          c.line("var _sc8 = " + wp + ".unsafe_offset(" + wb + "+" +
                 std::to_string(spec.subScaleOffset) +
                 "+_h*8).bitcast[Int8]().unsafe_load[width=8]().cast[DType.int32]()");
        } else {
          c.line("var _sc8 = SIMD[DType.int32,8](0)");
          c.line("comptime for _s in range(8):");
          c.indent++;
          c.line("_sc8[_s] = Int32(bitcast[DType.int8](" + wp +
                 "[unsafe_offset=" + wb + "+(" +
                 std::to_string(spec.subScaleOffset) + "+_h*8+_s)*" +
                 std::to_string(bs) + "]))");
          c.indent--;
        }
      } else {
        c.line("var _sc8 = SIMD[DType.int32,8](1)");
      }
      c.line("var _sc16 = _sc8.join(_sc8)");
      c.line("var _sc01 = _sc16.shuffle[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3](_sc16)");
      c.line("var _sc23 = _sc16.shuffle[4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7](_sc16)");
      c.line(res + "_accS += _d01 * _sc01 + _d23 * _sc23");
      c.line("var _bs8 = " + qp + ".bitcast[Int16]().unsafe_offset(" + g +
             "*" + std::to_string(Q8::bytes / 2) + "+" +
             std::to_string(Q8::sums / 2) +
             "+_h*8).unsafe_load[width=8]().cast[DType.int32]()");
      c.line(res + "_accB += _sc8 * _bs8");
      c.indent--;
      c.line("var " + res + "_tot = " + res + "_accS.reduce_add() - Int32(" +
             std::to_string(spec.zeroBias) + ") * " + res +
             "_accB.reduce_add()");
      c.line("var " + res + " = SIMD[DType.float32,1](" + res + "_ds * " +
             res + "_d * Float32(" + res + "_tot))");
      return;
    }

    // Portable SIMD path (no VNNI required)
    c.line("var " + res + "_accS = SIMD[DType.int32,16](0)");
    c.line("comptime for _h in range(2):");
    c.indent++;
    for (int q = 0; q < 4; ++q) {
      std::string uVar = "_u" + std::to_string(q);
      c.line("var " + uVar + " = SIMD[DType.uint8,32](0)");
      for (int p = 0; p < spec.numBitPlanes; ++p) {
        const auto &pl = spec.planes[p];
        int baseOff = pl.baseByte + (q & 1) * pl.quadByteStep;
        int shift = pl.quadShiftMode == 0 ? pl.shiftUnit * (q >> 1)
                                          : pl.shiftUnit * q;
        int mask = (1 << pl.bitWidth) - 1;
        std::string rawVar =
            "_r_" + std::to_string(q) + "_" + std::to_string(p);
        emitLoad32(rawVar, std::to_string(baseOff) + "+_h*" +
                               std::to_string(pl.halfStride));
        std::string part = rawVar;
        if (shift > 0)
          part = "(" + part + " >> " + std::to_string(shift) + ")";
        part = "(" + part + " & " + std::to_string(mask) + ")";
        if (pl.destBitOffset > 0)
          part = "(" + part + " << " + std::to_string(pl.destBitOffset) + ")";
        c.line(uVar + " = " + uVar + " | " + part);
      }
      c.line("var _w" + std::to_string(q) + " = " + uVar +
             ".cast[DType.int32]() - Int32(" + std::to_string(spec.zeroBias) +
             ")");
      c.line("var _q" + std::to_string(q) + " = " + qp + ".unsafe_offset(" +
             g + "*" + std::to_string(Q8::bytes) + "+" +
             std::to_string(Q8::values + q * 32) +
             "+_h*128).unsafe_load[width=32]().cast[DType.int32]()");
      c.line("var _p" + std::to_string(q) + " = _w" + std::to_string(q) +
             " * _q" + std::to_string(q));
      for (int isub = 0; isub < 2; ++isub) {
        int subIdx = 2 * q + isub;
        std::string scExpr =
            spec.subScaleOffset >= 0
                ? ("Int32(bitcast[DType.int8](" + wp + "[unsafe_offset=" + wb +
                   "+(" + std::to_string(spec.subScaleOffset + subIdx) +
                   "+_h*8)*" + std::to_string(bs) + "]))")
                : std::string("Int32(1)");
        c.line(res + "_accS += _p" + std::to_string(q) + ".slice[16, offset=" +
               std::to_string(isub * 16) + "]() * " + scExpr);
      }
    }
    c.indent--;
    c.line("var " + res + " = SIMD[DType.float32,1](" + res + "_ds * " + res +
           "_d * Float32(" + res + "_accS.reduce_add()))");
    return;
  }

  // Fully general element-loop fallback for arbitrary RecordGemv descriptors
  c.line("var " + res + "_sum = Float32(0.0)");
  c.line("for _i in range(0, " + std::to_string(spec.blockElements) + "):");
  c.indent++;
  c.line("var _half = _i // 128");
  c.line("var _inh = _i % 128");
  c.line("var _quad = _inh // 32");
  c.line("var _rem = _inh % 32");
  c.line("var _code = Int32(0)");
  for (int p = 0; p < spec.numBitPlanes; ++p) {
    const auto &pl = spec.planes[p];
    std::string off = std::to_string(pl.baseByte) + "+_half*" +
                      std::to_string(pl.halfStride) + "+(_quad&1)*" +
                      std::to_string(pl.quadByteStep) + "+_rem";
    std::string sh = pl.quadShiftMode == 0
                         ? ("(_quad>>1)*" + std::to_string(pl.shiftUnit))
                         : ("_quad*" + std::to_string(pl.shiftUnit));
    c.line("var _b" + std::to_string(p) + " = Int32(" + wp +
           "[unsafe_offset=" + wb + "+(" + off + ")*" + std::to_string(bs) +
           "])");
    c.line("_code = _code | (((_b" + std::to_string(p) + " >> " + sh + ") & " +
           std::to_string((1 << pl.bitWidth) - 1) + ") << " +
           std::to_string(pl.destBitOffset) + ")");
  }
  std::string subSc =
      (spec.subScaleOffset >= 0 && spec.subBlockElements > 0)
          ? ("Float32(bitcast[DType.int8](" + wp + "[unsafe_offset=" + wb +
             "+(" + std::to_string(spec.subScaleOffset) + "+_i//" +
             std::to_string(spec.subBlockElements) + ")*" +
             std::to_string(bs) + "]))")
          : std::string("Float32(1.0)");
  std::string xExpr =
      spec.q8kActivation
          ? (res + "_ds * Float32(" + qp + "[unsafe_offset=" + g + "*" +
             std::to_string(Q8::bytes) + "+" + std::to_string(Q8::values) +
             "+_i])")
          : (qp + "[unsafe_offset=" + g + "*" +
             std::to_string(spec.blockElements) + "+_i]");
  c.line(res + "_sum += Float32(_code - " + std::to_string(spec.zeroBias) +
         ") * " + subSc + " * (" + xExpr + ")");
  c.indent--;
  c.line("var " + res + " = SIMD[DType.float32,1](" + res + "_d * " + res +
         "_sum)");
}

} // namespace intentdsl::cpu
