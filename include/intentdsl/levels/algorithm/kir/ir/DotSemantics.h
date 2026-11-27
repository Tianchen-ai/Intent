#pragma once
#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/FastMath.h"

namespace intentdsl::kir {
inline void registerDotSemantics(compilerinfra::Unit &unit, const char *backend,
                                 const char *format, const char *upstream,
                                 const char *reference, const char *citation,
                                 const char *nativeFeature = nullptr) {
  std::string name = std::string(format) + "-dot-denormal";
  auto attribute = "kir." + std::string(format) + "_dot";
  unit.semantics.push_back(
      {{name, std::string(format) + " contraction, FP32 accumulation", upstream,
        reference, citation,
        [=](const compilerinfra::TargetProfile &p,
            const llvm::json::Object &options)
            -> std::optional<compilerinfra::SemanticChoice> {
          if (p.name != backend)
            return std::nullopt;
          auto *policy = p.services.get<FastMathPolicy>();
          if (options.getString("float32_matmul_precision") == "highest" ||
              (p.numerical.explicitFastMath &&
               !mlir::arith::bitEnumContainsAll(
                   p.numerical.defaultFastMath,
                   mlir::arith::FastMathFlags::contract)) ||
              options.getString("fpmath_mode") == "strict")
            return compilerinfra::SemanticChoice{
                "fp32-product", "configuration", attribute,
                "Explicit FP32 product semantics"};
          auto mode = p.numerical.defaultDenormal;
          if (policy && options.get("denormal_fp_math"))
            mode = llvm::parseDenormalFPAttribute(policy->denormalFpMath);
          if ((options.get("denormal_fp_math") ||
               p.numerical.explicitDenormal) &&
              mode != llvm::DenormalMode::getPreserveSign() &&
              mode != llvm::DenormalMode::getPositiveZero())
            return compilerinfra::SemanticChoice{
                "source", "configuration", attribute,
                "Explicit denormal environment"};
          auto value = nativeFeature && !p.features.count(nativeFeature)
                           ? "source"
                           : reference;
          return compilerinfra::SemanticChoice{value, "reference", attribute,
                                               ""};
        }},
       COMPILERINFRA_SOURCE});
}
} // namespace intentdsl::kir
