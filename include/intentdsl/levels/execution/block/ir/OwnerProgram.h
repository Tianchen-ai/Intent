#pragma once
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace compilerinfra {
struct TargetProfile;
}
namespace intentdsl::block {
mlir::LogicalResult
verifyTileProgram(mlir::func::FuncOp, const compilerinfra::TargetProfile &,
                  llvm::function_ref<bool(mlir::Operation &)> provesNarrow);
struct Tiles;
struct OwnerAccess {
  // Return false when elementwise coordinates must be retained.
  bool (*rectangle)(Tiles &, mlir::Operation &, llvm::ArrayRef<mlir::Value>,
                    int64_t rows, mlir::Value stored,
                    mlir::Value &result) = nullptr;
  mlir::Value (*panel)(Tiles &, mlir::Operation &, llvm::ArrayRef<mlir::Value>,
                       mlir::Value start, int64_t span,
                       int64_t width) = nullptr;
};
void expandOwnerProgram(mlir::func::FuncOp fn, llvm::StringRef unit,
                        OwnerAccess access = {});
} // namespace intentdsl::block
