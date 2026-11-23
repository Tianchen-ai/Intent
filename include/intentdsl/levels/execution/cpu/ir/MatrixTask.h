#pragma once
#include "intentdsl/levels/algorithm/kir/ir/ContractionShape.h"
#include "intentdsl/levels/algorithm/kir/ir/Matrix.h"
#include "intentdsl/levels/execution/cpu/ir/Task.h"
namespace intentdsl::cpu {
struct MatrixImplementation {
  int64_t rows, cols, depth, lanes;
  std::string id = {};
  std::string estimator = {};
  std::string packing = {};
  std::string dtype = {};
  int64_t kAlign = 1;
  bool requiresContiguousLhsK = false;
  bool flushesSubnormals = false;
};
struct MatrixMachine {
  std::vector<MatrixImplementation> choices, records;
  std::string integerPacking, estimate;
  std::vector<MatrixImplementation> contracts;
};
mlir::Operation *findMatrixTask(mlir::func::FuncOp);
bool matrixTask(mlir::func::FuncOp);
mlir::LogicalResult verifyMatrixPartition(mlir::func::FuncOp, mlir::Operation &,
                                          const compilerinfra::TargetProfile &);
mlir::Operation &calculation(mlir::Operation &domain);
mlir::LogicalResult verifyMatrixTask(mlir::func::FuncOp,
                                     const compilerinfra::TargetProfile &,
                                     llvm::StringRef storage);
llvm::Error materializeMatrixBuffers(mlir::func::FuncOp, mlir::RewriterBase &);
mlir::LogicalResult verifyMatrixBuffers(mlir::func::FuncOp,
                                        const compilerinfra::TargetProfile &);
// Host ABI follows runtime/cpu's typed slot order for CPU containers.
std::string matrixHostParams(mlir::func::FuncOp);
} // namespace intentdsl::cpu
