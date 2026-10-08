#ifndef INTENT_TARGET_BANGC_NATIVE_WORKSPACE_H
#define INTENT_TARGET_BANGC_NATIVE_WORKSPACE_H
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace intent::bangc {
mlir::Type arithmeticStorageType(mlir::Type element);
llvm::SmallVector<int64_t, 2> exponentialScratchShape(mlir::Type element, int64_t elements);
llvm::SmallVector<int64_t, 2> extremaScratchShape(mlir::Type element, int64_t elements);
llvm::SmallVector<int64_t, 2> selectionScratchShape(mlir::Type element,
    int64_t elements, bool tensorFalseValue);
llvm::SmallVector<int64_t, 2> rowOffsetsShape(int64_t rows);
struct GatherWorkspace {
  llvm::SmallVector<int64_t, 2> plan, indices;
  int64_t internalBytes;
};
std::optional<GatherWorkspace> gatherWorkspace(int64_t rows);
struct NativeWorkspace {
  llvm::SmallVector<mlir::MemRefType> buffers;
  int64_t internalBytes = 0;
};
// Additional storage of the existing primitive at these actual local extents.
// Unknown implementations do not authorize a larger workset.
std::optional<NativeWorkspace> queryNativeWorkspace(
    mlir::Operation *operation, llvm::ArrayRef<int64_t> shape);
} // namespace intent::bangc
#endif
