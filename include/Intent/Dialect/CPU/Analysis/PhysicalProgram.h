#ifndef INTENT_DIALECT_CPU_ANALYSIS_PHYSICALPROGRAM_H
#define INTENT_DIALECT_CPU_ANALYSIS_PHYSICALPROGRAM_H

#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include <optional>

namespace intent::cpu {

enum class CPUProgramStage { Values, Buffers, Realized };

struct AllocationFacts {
  mlir::Value value;
  mlir::Operation *owner;
  mlir::Operation *writer;
  std::optional<int64_t> bytes;
  bool stack;
};

// A query object belongs to the current IR snapshot; recreate after a rewrite.
class PhysicalProgramAnalysis {
public:
  explicit PhysicalProgramAnalysis(mlir::func::FuncOp function) : function(function) {}
  llvm::SmallVector<AllocationFacts> allocations();
  mlir::LogicalResult verify(CPUProgramStage stage);

private:
  mlir::func::FuncOp function;
};

bool isMatrixContraction(mlir::linalg::GenericOp operation);
std::optional<llvm::SmallVector<std::pair<unsigned, unsigned>>>
unitReshapeAxes(mlir::Operation *operation);
bool supportsVectorScan(ScanOp operation);
mlir::LogicalResult verifyCPUProgram(mlir::ModuleOp module, CPUProgramStage stage);

}
#endif
