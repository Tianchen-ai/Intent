#ifndef INTENT_CPU_TRANSFORMS_VECTOR_CONTIGUOUSMEMORY_H
#define INTENT_CPU_TRANSFORMS_VECTOR_CONTIGUOUSMEMORY_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"

namespace intent::cpu {

// The caller has proved vector access eligibility for these ranked strided
// memories. Only their dynamic innermost strides remain to be checked.
struct ContiguousMemoryGuard {
  mlir::Value condition;
  llvm::SmallVector<mlir::memref::ExtractStridedMetadataOp> descriptors;

  // Call only in the condition's true branch. Retain the actual descriptor,
  // refining its innermost stride without relying on a removable memref.cast.
  void bind(mlir::OpBuilder &builder, mlir::Location location,
            mlir::IRMapping &mapping) const;
};

ContiguousMemoryGuard materializeContiguousMemoryGuard(
    mlir::OpBuilder &builder, mlir::Location location, mlir::ValueRange memories);

} // namespace intent::cpu

#endif
