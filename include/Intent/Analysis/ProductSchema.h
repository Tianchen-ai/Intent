#ifndef INTENT_ANALYSIS_PRODUCTSCHEMA_H
#define INTENT_ANALYSIS_PRODUCTSCHEMA_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <cstddef>
#include <string>

namespace intent {

// All queries consume verified canonical types. Product fields are visited in
// declaration order, recursively; a non-product has one leaf and an empty tuple
// has none. These are SSA component positions, never memory offsets.
mlir::ArrayAttr getProductComponents(mlir::Type type);
mlir::Type getProductComponentType(mlir::Type type,
                                   llvm::ArrayRef<unsigned> fieldPath);

void walkProductLeaves(
    mlir::Type type,
    llvm::function_ref<void(mlir::Type, llvm::ArrayRef<unsigned>)> visit);

void appendProductLeafTypes(mlir::Type type,
                           llvm::SmallVectorImpl<mlir::Type> &types);
void appendProductLeafTypes(mlir::TypeRange roots,
                           llvm::SmallVectorImpl<mlir::Type> &types);

struct ProductLeafRange {
  size_t offset;
  size_t size;
};

mlir::FailureOr<ProductLeafRange>
getProductLeafRange(mlir::Type type, llvm::ArrayRef<unsigned> fieldPath);

llvm::SmallVector<ProductLeafRange>
getProductLeafRanges(mlir::TypeRange roots);

// Diagnostic component names use record field names and tuple positions.
// The structural path above, rather than this spelling, identifies the field.
std::string getProductPathName(mlir::Type type,
                               llvm::ArrayRef<unsigned> fieldPath);

} // namespace intent

#endif
