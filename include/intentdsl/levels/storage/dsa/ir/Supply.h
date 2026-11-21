#pragma once
#include "compilerinfra/Assembly.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
namespace intentdsl::dsa {
struct LocalSupply {
  int64_t capacity, budget, alignment, consumers, reductionScratchPerColumn;
  bool asynchronous;
};
bool projectsConsecutiveRows(mlir::Operation *access, unsigned maskIndex);
mlir::LogicalResult verifyProgram(mlir::ModuleOp,
                                  const compilerinfra::TargetProfile &,
                                  bool supplied);
} // namespace intentdsl::dsa
