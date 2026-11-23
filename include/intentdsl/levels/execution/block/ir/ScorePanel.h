#pragma once
#include "intentdsl/levels/algorithm/kir/ir/Normalized.h"
#include "intentdsl/levels/execution/block/ir/Owners.h"
namespace intentdsl::block {
using kir::ScorePanel;
struct ScoreResources {
  std::function<bool(const ScorePanel &, int64_t, const CallFacts &)> supports;
};
std::optional<ScorePanel> scorePanel(mlir::Operation *,
                                     const compilerinfra::TargetProfile &);
bool legalScoreTile(const ScorePanel &, int64_t);
bool fitsScoreTile(const ScorePanel &, int64_t,
                   const compilerinfra::TargetProfile &);
void realizeScoreTile(mlir::Operation *, const ScorePanel &, int64_t);
mlir::LogicalResult verifyScoreTile(mlir::func::FuncOp, llvm::StringRef,
                                    const compilerinfra::TargetProfile &);
} // namespace intentdsl::block
