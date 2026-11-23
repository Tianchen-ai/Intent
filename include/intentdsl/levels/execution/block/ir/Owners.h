#pragma once
#include "intentdsl/levels/algorithm/kir/ir/Ordered.h"
#include "intentdsl/levels/algorithm/kir/ir/Performance.h"

namespace intentdsl::block {
struct OwnerMachine {
  int parallelMembers;
  int residentElements;
  int64_t panelBytes = 0;
  int64_t contributionLimit = 0;
};
bool canForwardOwners(mlir::Operation *, const compilerinfra::TargetProfile &);
mlir::LogicalResult forwardOwners(mlir::func::FuncOp);
bool columnPrefixes(mlir::Operation *, const compilerinfra::TargetProfile &);
void realizeColumns(mlir::func::FuncOp);
mlir::LogicalResult verifyColumns(mlir::func::FuncOp, llvm::StringRef);
bool canGroupReads(mlir::Operation *, const compilerinfra::TargetProfile &,
                   unsigned);
void groupReads(mlir::func::FuncOp, unsigned);
using kir::ownerCount;
mlir::LogicalResult verifyOwnerMap(mlir::func::FuncOp,
                                   const compilerinfra::TargetProfile &,
                                   llvm::StringRef, llvm::StringRef);
llvm::Error emitOwnerArtifact(mlir::func::FuncOp, llvm::StringRef,
                              llvm::json::Object, unsigned,
                              llvm::raw_pwrite_stream &);
} // namespace intentdsl::block
