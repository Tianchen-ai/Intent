#pragma once
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

namespace intentdsl::block {
enum class RowProjection { Columns, Matrix };
inline mlir::FailureOr<RowProjection> projection(mlir::Type type) {
  auto view = mlir::dyn_cast<mlir::MemRefType>(type);
  if (view && view.getRank() == 1)
    return RowProjection::Columns;
  if (view && view.getRank() == 2)
    return RowProjection::Matrix;
  return mlir::failure();
}
mlir::LogicalResult verifyRowAccess(mlir::Operation *op);
} // namespace intentdsl::block
