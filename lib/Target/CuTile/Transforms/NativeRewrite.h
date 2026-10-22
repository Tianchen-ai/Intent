#ifndef INTENT_TARGET_CUTILE_TRANSFORMS_NATIVEREWRITE_H
#define INTENT_TARGET_CUTILE_TRANSFORMS_NATIVEREWRITE_H

#include "Intent/Dialect/GPU/IR/GPUOps.h"

namespace intent::cutile {

// Input operations retain their original def-use until native forms are ready.
// The transaction and its replacements live only within one formation phase.
class NativeFormRewriter {
public:
  void replace(mlir::Operation *operation, mlir::ValueRange values);
  void erase(mlir::Operation *operation);
  void commit();

private:
  struct Replacement {
    mlir::Operation *operation;
    llvm::SmallVector<mlir::Value> values;
  };
  llvm::SmallVector<Replacement> replacements;
};

mlir::Type withElementType(mlir::Type type, mlir::Type elementType);

// Repeat complete tiles on an axis without reloading or changing elements.
// requested and the original extent must be compatible power-of-two extents.
mlir::Value repeatTileAxis(mlir::OpBuilder &builder, mlir::Location location,
    mlir::Value value, unsigned axis, gpu::PhysicalExprAttr requested);

} // namespace intent::cutile

#endif
