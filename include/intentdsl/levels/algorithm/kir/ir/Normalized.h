#pragma once
#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
namespace intentdsl::kir {
struct ScorePanel {
  mlir::Value q,k,v,y,scale;
  int64_t group,queries,keys,depth,features;
  mlir::Value maxScore,weightScore;
};
bool scoreShape(ScorePanel &,const CallFacts &);
std::optional<ScorePanel> normalizedContractions(mlir::Operation *,const CallFacts &);
}
