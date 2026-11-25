#pragma once
#include "compilerinfra/Assembly.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
namespace intentdsl::dsa {
struct LocalSupply {
  int64_t capacity, budget, alignment, consumers, reductionScratchPerColumn;
  bool asynchronous;
  int64_t featurePanelMembers;
  int64_t indexScratchPerMember;
  int64_t randomScratchBytes = 0;
  std::string libraryTransposeImpl;
  std::string librarySoftmaxImpl;
  std::string libraryLayerNormImpl;
  std::string libraryEstimator;
  std::string libraryGatherImpl;
  // Scratch bytes per lane for an authorized alternative to certified math.
  std::optional<int64_t> (*approxScratch)(mlir::Operation *) = nullptr;
};
bool projectsConsecutiveRows(mlir::Operation *access, unsigned maskIndex);
bool projectsTransposed2D(mlir::Operation *op);
struct TemporaryLayout {
  llvm::DenseMap<mlir::Operation *, int64_t> offsets;
  int64_t peak = 0;
};
TemporaryLayout placeTemporaries(mlir::Block &calculations);
mlir::LogicalResult verifyProgram(mlir::ModuleOp,
                                  const compilerinfra::TargetProfile &,
                                  bool supplied);
} // namespace intentdsl::dsa
