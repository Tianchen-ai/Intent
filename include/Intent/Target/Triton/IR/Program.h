#ifndef INTENT_TARGET_TRITON_IR_PROGRAM_H
#define INTENT_TARGET_TRITON_IR_PROGRAM_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"

namespace intent::triton {

inline constexpr llvm::StringLiteral contractFormAttr =
    "intent_gpu.triton.contract_form";
inline constexpr llvm::StringLiteral loopUnrollFactorAttr =
    "intent_gpu.triton.loop_unroll_factor";

mlir::LogicalResult verifyProgramAttributes(mlir::Operation *operation);

} // namespace intent::triton

#endif
