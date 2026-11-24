#pragma once
#include "intentdsl/levels/execution/block/ir/Owners.h"
#include "intentdsl/levels/storage/dsa/ir/Matrix.h"
#include "intentdsl/levels/storage/dsa/ir/Supply.h"
#include "llvm/ADT/DenseMap.h"

namespace intentdsl::dsa {
mlir::LogicalResult verifyOwners(mlir::func::FuncOp,
                                 const compilerinfra::TargetProfile &,
                                 bool supplied);

bool ownerValueDependsOn(mlir::Value v, mlir::Value target);

struct StreamTransfer {
  mlir::Operation *access;
  int64_t offset, axis, steps, width, pack;
};

struct OwnerStreams {
  llvm::SmallVector<StreamTransfer, 8> transfers;
  int64_t bytes = 0;
  int64_t pack = 1;
};

struct PanelFoldSupply {
  int64_t rows = 0;
  int64_t span = 0;
  int64_t depth = 0;
  int64_t features = 0;
  int64_t wramBytes = 0;
  int64_t localBytes = 0;
  llvm::SmallVector<int64_t, 3> scratchOffsets;
};

OwnerStreams ownerStreams(mlir::func::FuncOp,
                          const compilerinfra::TargetProfile &);

std::optional<PanelFoldSupply> panelFoldSupply(
    mlir::func::FuncOp, const compilerinfra::TargetProfile &,
    llvm::DenseMap<mlir::Operation *, llvm::SmallVector<int64_t>> *opOffsets =
        nullptr);
} // namespace intentdsl::dsa
