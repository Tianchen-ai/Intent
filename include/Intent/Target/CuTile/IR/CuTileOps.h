#ifndef INTENT_TARGET_CUTILE_IR_CUTILEOPS_H
#define INTENT_TARGET_CUTILE_IR_CUTILEOPS_H

#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Target/CuTile/IR/CuTileDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#define GET_OP_CLASSES
#include "Intent/Target/CuTile/IR/CuTileOps.h.inc"

namespace intent::cutile {
// The provider-default policy leaves the worker count to the lower compiler.
inline constexpr int64_t inferredWorkerWarps = 1;
inline constexpr int64_t inferredLoadPolicy = 11;
inline bool isLegalLoadPolicy(int64_t value) {
  return value >= 1 && value <= inferredLoadPolicy;
}
mlir::Attribute getCompileTimeScalar(mlir::Value value);
mlir::ArrayAttr getNativeArrayIndexBounds(mlir::Value resource);
}

#endif
