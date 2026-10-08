#ifndef INTENT_DIALECT_CPU_ANALYSIS_CONTRACTIONS_H
#define INTENT_DIALECT_CPU_ANALYSIS_CONTRACTIONS_H

#include "Intent/Analysis/ContractionAxes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace intent::cpu {

// The current linalg program's paired axes and exact supported multiply-add
// body. Shape, storage organization and implementation selection are separate.
std::optional<ContractionAxes> queryContractionAxes(mlir::linalg::GenericOp operation);

} // namespace intent::cpu
#endif
