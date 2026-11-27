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
llvm::Error materializeMatrixBuffers(
    mlir::func::FuncOp, mlir::RewriterBase &,
    llvm::StringRef realization = "private-values");
mlir::LogicalResult verifyMatrixBuffers(
    mlir::func::FuncOp, const compilerinfra::TargetProfile &,
    llvm::StringRef realization = "private-values");
// Pad f32-widened LHS row stride by 16 floats (64 B) when K is a large multiple
// of 128 floats (512 B) to break 4K/8K L1D set-associativity aliasing.
inline int64_t paddedLhsPanelStride(int64_t K) {
  return (K >= 512 && K % 128 == 0) ? (K + 16) : K;
}
// Transpose complete packed words in source cache-line tiles. The caller
// handles the final partial word and column padding with the regular packer.
template <class For, class Enter, class Element>
void emitVnniWordTranspose(int64_t tile, int64_t block, int64_t nStride,
                           For &loop, Enter &enter, Element &element) {
  std::string step = std::to_string(block), width = std::to_string(tile);
  enter(loop("kt", "0", "kFull", width));
  enter(loop("j", "0", "valid", "1"));
  enter(loop("k", "kt", "(kFull-kt < " + width + " ? kFull : kt+" + width + ")",
             step));
  element("memcpy", "out + (k/" + step + "*nr+j)*" + step,
          "src + j*" + std::to_string(nStride) + "LL + k", "", 4);
}
// Host ABI follows runtime/cpu's typed slot order for CPU containers.
std::string matrixHostParams(mlir::func::FuncOp);
} // namespace intentdsl::cpu
