#ifndef INTENT_Block_ANALYSIS_INDEXBOUNDS_H
#define INTENT_Block_ANALYSIS_INDEXBOUNDS_H

#include "Intent/Dialect/Block/Analysis/PhysicalProgram.h"

namespace intent::block::detail {

struct IndexBounds {
  bool nonNegative = false;
  PhysicalExprAttr upper;
  int64_t lower = 0;
};

mlir::Value stripIntegerIndexCasts(mlir::Value value);
bool coordinateKnownNonNegative(mlir::Value coordinate);
bool coordinateKnownPositive(mlir::Value coordinate);
std::optional<std::pair<int64_t, int64_t>>
nonNegativeExtentBounds(mlir::func::FuncOp kernel, PhysicalExprAttr extent);
std::optional<std::pair<int64_t, int64_t>>
positiveExtentBounds(mlir::func::FuncOp kernel, PhysicalExprAttr extent);
IndexBounds queryIndexBounds(mlir::Value value);

} // namespace intent::block::detail

#endif
