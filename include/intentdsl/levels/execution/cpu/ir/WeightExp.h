#pragma once
#include "compilerinfra/Assembly.h"

namespace intentdsl::cpu {
inline void registerWeightExp(compilerinfra::Unit &u) {
  u.semantics.push_back(
      {{"reduced-weight-exp",
        "vector FP32 exponential feeding BF16 contraction weights",
        "ATen CPU FlashAttentionKernel::_exp_reduce_sum_fusion_kernel",
        "fexp_u20",
        "PyTorch 449b1768 "
        "aten/src/ATen/native/cpu/FlashAttentionKernel.cpp:96-108 and "
        "ATen/cpu/vec/vec512/vec512_float.h Vectorized<float>::fexp_u20; "
        "CppVecOverrides.exp uses Vectorized<float>::exp (Sleef_expf16_u10) "
        "for ordinary exp.",
        [](const compilerinfra::TargetProfile &p,
           const llvm::json::Object &opts)
            -> std::optional<compilerinfra::SemanticChoice> {
          bool configured =
              p.numerical.explicitDenormal || p.numerical.explicitFastMath ||
              opts.get("denormal_fp_math") || opts.get("fpmath_mode") ||
              opts.get("fast_math") || opts.getString("numerics") == "source";
          return compilerinfra::SemanticChoice{
              !configured && p.features.count("avx512f") ? "fexp_u20"
                                                         : "source",
              configured ? "configuration" : "reference", "cpu.weight_exp",
              configured ? "Explicit floating-point policy" : ""};
        }},
       COMPILERINFRA_SOURCE});
}

// ATen AVX-512 fexp_u20, Malossi et al. cubic correction to the mantissa.
// Select the lower and upper range results and preserve NaN inputs.
template <typename V, typename Literal, typename Binary, typename Fma,
          typename Compare, typename Select, typename Floor, typename Convert,
          typename IsNaN>
V reducedWeightExp(V x, Literal lit, Binary bin, Fma fma, Compare cmp,
                   Select sel, Floor floor, Convert convert, IsNaN isNaN) {
  V zero = lit(0.f);
  V tiny = cmp("lt", x, lit(-87.3365478515625f));
  V overflow = cmp("gt", x, lit(88.72283935546875f));
  V y = bin("mul", x, lit(1.4426950408889634f));
  V frac = bin("sub", y, floor(y));
  V poly = fma(frac, lit(-0.079204240219773236f), lit(-0.22433836478672356f));
  poly = fma(frac, poly, lit(0.30354260500649682f));
  poly = fma(frac, poly, lit(0.00010703434948458272f));
  V bits = convert(fma(lit(8388608.f), bin("sub", y, poly), lit(1065353216.f)),
                   "int32");
  return sel(isNaN(x), x,
             sel(overflow, lit(std::numeric_limits<float>::infinity()),
                 sel(tiny, zero, convert(bits, "bitcast"))));
}
} // namespace intentdsl::cpu
