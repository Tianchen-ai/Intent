#pragma once
#include "compilerinfra/Assembly.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
namespace intentdsl::dsa {
struct LocalSupply {
  int64_t capacity, budget, alignment, consumers, reductionScratchPerColumn;
  bool asynchronous;
  int64_t featurePanelMembers;
  int64_t indexScratchPerMember;
};
bool projectsConsecutiveRows(mlir::Operation *access, unsigned maskIndex);
struct TemporaryLayout {
  llvm::DenseMap<mlir::Operation *, int64_t> offsets;
  int64_t peak = 0;
};
TemporaryLayout placeTemporaries(mlir::Block &calculations);
mlir::LogicalResult verifyProgram(mlir::ModuleOp,
                                  const compilerinfra::TargetProfile &,
                                  bool supplied);
} // namespace intentdsl::dsa
