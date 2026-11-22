#pragma once
#include "intentdsl/levels/execution/block/ir/Owners.h"
#include "intentdsl/levels/storage/dsa/ir/Supply.h"
namespace intentdsl::dsa {
mlir::LogicalResult verifyOwners(mlir::func::FuncOp,
                                 const compilerinfra::TargetProfile &,
                                 bool supplied);
struct StreamTransfer {
  mlir::Operation *access;
  int64_t offset, axis, steps, width;
};
struct OwnerStreams {
  llvm::SmallVector<StreamTransfer, 8> transfers;
  int64_t bytes = 0;
};
OwnerStreams ownerStreams(mlir::func::FuncOp,
                          const compilerinfra::TargetProfile &);
} // namespace intentdsl::dsa
