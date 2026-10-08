#ifndef INTENT_ANALYSIS_ONLINESUMMARY_H
#define INTENT_ANALYSIS_ONLINESUMMARY_H

#include "Intent/Dialect/Intent/IR/IntentOps.h"
#include <optional>

namespace intent {
struct OnlineSummary {
  MakeRecordOp summary, merged;
  ReduceOp validity, maximum, mass;
  ContractOp moment;
  SelectOp maximumOrEmpty, probability, momentOrEmpty;
  CastOp probabilityCast;
  UnaryOp exponential;
  mlir::Value combinedMaximum, leftScale, rightScale;
  mlir::Value leftMassTerm, leftMomentTerm;
  unsigned validField, maxField, massField, momentField;
};

// The descriptor refers only to current canonical helper computations. Target
// construction supplies source slicing, storage and accumulator realization.
std::optional<OnlineSummary> matchOnlineSummary(RegionFoldOp fold);
}
#endif
