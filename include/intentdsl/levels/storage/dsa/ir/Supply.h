#pragma once
#include "compilerinfra/Assembly.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
namespace intentdsl::dsa {
struct LocalSupply {
  int64_t capacity, budget, alignment, consumers, reductionScratchPerColumn;
  bool asynchronous;
};
mlir::LogicalResult verifyProgram(mlir::ModuleOp,
                                  const compilerinfra::TargetProfile &,
                                  bool supplied);
} // namespace intentdsl::dsa
