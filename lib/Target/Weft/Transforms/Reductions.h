#ifndef INTENT_TARGET_WEFT_TRANSFORMS_REDUCTIONS_H
#define INTENT_TARGET_WEFT_TRANSFORMS_REDUCTIONS_H

#include "Intent/Dialect/CPU/IR/CPUAttrs.h"
#include "mlir/IR/Block.h"
#include "llvm/ADT/StringRef.h"

namespace intent::weft_provider {

struct NativeReduction {
  mlir::Operation *combine;
  mlir::Value contribution;
  llvm::StringRef kind;
};

// Query one current scalar body. Both shaped DPS and scalar SSA reductions
// must satisfy this native-combine contract before mapping to Weft reduce.
mlir::FailureOr<NativeReduction> queryNativeReduction(
    mlir::Operation *owner, mlir::Block &body, mlir::Value accumulator,
    cpu::ReductionOrderAttr order);

} // namespace intent::weft_provider
#endif
