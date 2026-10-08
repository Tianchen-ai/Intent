#ifndef INTENT_TARGET_TRITON_IR_TRITONOPS_H
#define INTENT_TARGET_TRITON_IR_TRITONOPS_H

#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Target/Triton/IR/TritonDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace intent::triton {
inline constexpr llvm::StringLiteral loopStagesAttr = "intent_triton.loop_stages";
}

#define GET_OP_CLASSES
#include "Intent/Target/Triton/IR/TritonOps.h.inc"

#endif
