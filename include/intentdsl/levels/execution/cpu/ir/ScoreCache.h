#pragma once
#include "intentdsl/levels/algorithm/kir/ir/Normalized.h"
namespace intentdsl::cpu {
inline std::optional<kir::ScorePanel>
reusableScores(mlir::Operation *o, const compilerinfra::TargetProfile &p) {
  auto facts = p.services.get<CallFacts>();
  if (!facts)
    return {};
  auto s = kir::normalizedContractions(o, *facts);
  // Each owner retains one fp32 score per key on the task's stack.
  if (s && s->keys <= 1024)
    return s;
  return {};
}
} // namespace intentdsl::cpu
