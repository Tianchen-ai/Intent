#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/FastMath.h"

namespace CI_UNIT_NAMESPACE {
using namespace compilerinfra;
using mlir::arith::FastMathFlags;
using llvm::json::Object;

void registerNumericalReferences(Unit &u) {
  u.semantics.push_back({{"denormal", "floating-point operations", "LLVM denormal-fp-math", "ieee",
    "LLVM LangRef: denormal-fp-math (output,input)",
    [](const TargetProfile &p, const Object &o) -> std::optional<SemanticChoice> {
      auto mode = intentdsl::kir::canonicalDenormalName(p.numerical.defaultDenormal);
      if (const auto *policy = p.services.get<intentdsl::kir::FastMathPolicy>();
          policy && (o.get("denormal_fp_math") || o.get("fpmath_mode") || o.getBoolean("fast_math").value_or(false)))
        mode = policy->denormalFpMath;
      bool configured = p.numerical.explicitDenormal || o.get("denormal_fp_math") || o.get("fpmath_mode") ||
                        o.getBoolean("fast_math").value_or(false);
      return SemanticChoice{mode, configured ? "configuration" : "reference", "denormal-fp-math",
                            configured ? "Explicit floating-point environment" : ""};
    }}, COMPILERINFRA_SOURCE});
  u.semantics.push_back({{"normalization", "parallel reduction chain",
    "torch._inductor.config.online_softmax", "online-eligible",
    "torch._inductor.config.online_softmax and lowering._softmax",
    [](const TargetProfile &p, const Object &o) -> std::optional<SemanticChoice> {
      bool strict = o.getString("fpmath_mode") == "strict" ||
                    (p.numerical.explicitFastMath && p.numerical.defaultFastMath == FastMathFlags::none);
      bool source = o.getString("numerics") == "source" ||
                    (strict && o.getString("numerics") != "relaxed_normalization");
      return SemanticChoice{source ? "source" : "online-eligible", source || o.get("numerics") ? "configuration" : "reference",
          "kir.normalization", source ? "Explicit source-order normalization" : ""};
    }}, COMPILERINFRA_SOURCE});
  u.semantics.push_back({{"matmul-precision", "f32 contraction inputs",
    "torch.set_float32_matmul_precision", "highest",
    "torch.set_float32_matmul_precision: highest and high",
    [](const TargetProfile &, const Object &o) -> std::optional<SemanticChoice> {
      auto value = o.getString("float32_matmul_precision");
      return SemanticChoice{value.value_or("highest").str(), value ? "configuration" : "reference",
                            "kir.matmul_precision", value ? "Explicit input precision" : ""};
    }}, COMPILERINFRA_SOURCE});
  auto contraction = [&](const char *name, const char *scope, const char *upstream,
                          const char *reference, const char *citation) {
    u.semantics.push_back({{name, "floating multiply and add", upstream, reference, citation,
      [scope, reference](const TargetProfile &p, const Object &o) -> std::optional<SemanticChoice> {
        if (p.name != scope)
          return std::nullopt;
        bool configured = p.numerical.explicitFastMath || o.get("fpmath_mode") || o.getBoolean("fast_math").value_or(false);
        auto flags = p.numerical.defaultFastMath;
        if (auto *policy = p.services.get<intentdsl::kir::FastMathPolicy>(); policy && policy->explicitFastMath)
          flags = policy->defaultFlags;
        auto value = configured ? (mlir::arith::bitEnumContainsAll(flags, FastMathFlags::contract) ? "on" : "off") : reference;
        return SemanticChoice{value, configured ? "configuration" : "reference", "kir.fp_contract",
                              configured ? "Explicit contraction permission" : ""};
      }}, COMPILERINFRA_SOURCE});
  };
  contraction("triton-fp-contract", "triton", "Triton CUDAOptions.enable_fp_fusion",
              "on", "triton.backends.nvidia.compiler.CUDAOptions.enable_fp_fusion");
  contraction("cutile-fp-contract", "cutile", "cuTile RoundingMode", "on",
              "cuTile Python: add and mul, unspecified rounding permits contraction");
  contraction("weft-fp-contract", "weft", "Clang -ffp-contract=on", "expression",
              "Clang Users Manual, Controlling Floating-Point Behavior");
  contraction("mojo-fp-contract", "mojo", "Clang -ffp-contract=on", "expression",
              "Clang Users Manual -ffp-contract=on; Mojo build --fp-mode=contract=off with explicit std.math.fma");
  contraction("bangc-fp-contract", "bangc", "cncc -ffp-contract=on", "expression",
              "cncc --help: -ffp-contract, default on outside CUDA/HIP");
  u.semantics.push_back({{"division", "floating-point quotients", "LLVM arcp", "rn",
    "LLVM LangRef: Allow Reciprocal (arcp)",
    [](const TargetProfile &p, const Object &) -> std::optional<SemanticChoice> {
      auto flags = p.numerical.defaultFastMath;
      if (auto *policy = p.services.get<intentdsl::kir::FastMathPolicy>(); policy && policy->explicitFastMath)
        flags = policy->defaultFlags;
      bool arcp = mlir::arith::bitEnumContainsAll(flags, FastMathFlags::arcp);
      return SemanticChoice{arcp ? "reciprocal-eligible" : "rn", arcp ? "configuration" : "reference",
                            "fastmath.arcp", arcp ? "Explicit reciprocal permission, subject to target precision" : ""};
    }, {"exits/mojo/legalize/Rows.cpp: proven exact inverse of a constant divisor",
        "runtime/bangc/kernels/ieee_math.mlu: certified quotient and scalar fallback"}}, COMPILERINFRA_SOURCE});
}
} // namespace CI_UNIT_NAMESPACE
