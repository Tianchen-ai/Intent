#pragma once
#include "intentdsl/levels/algorithm/kir/ir/ContractionShape.h"
#include "intentdsl/levels/algorithm/kir/ir/Matrix.h"
#include "intentdsl/levels/execution/cpu/ir/Task.h"
namespace intentdsl::cpu {
struct MatrixImplementation {
  int64_t rows, cols, depth, lanes;
};
struct MatrixMachine {
  std::vector<MatrixImplementation> choices, records;
  std::string integerPacking, estimate;
};
bool matrixTask(mlir::func::FuncOp);
mlir::LogicalResult verifyMatrixPartition(mlir::func::FuncOp, mlir::Operation &,
                                          const compilerinfra::TargetProfile &);
mlir::Operation &calculation(mlir::Operation &domain);
mlir::LogicalResult verifyMatrixTask(mlir::func::FuncOp,
                                     const compilerinfra::TargetProfile &,
                                     llvm::StringRef storage);
// Host ABI follows runtime/cpu's typed slot order for CPU containers.
std::string matrixHostParams(mlir::func::FuncOp);
} // namespace intentdsl::cpu
