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

struct PrivateRows {
  llvm::SmallVector<mlir::Value> views;
  llvm::DenseSet<mlir::Value> writes;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value>> prefixes;
  int64_t lanes = 1;
  int64_t bytes = 0, scanWidth = 0, selectionWidth = 0, k = 0;
};
std::optional<PrivateRows> privateRows(mlir::func::FuncOp,
                                       const compilerinfra::TargetProfile &);
struct IndexedRowAdd {
  mlir::Value keys, source, destination;
  int64_t rows, width, batch;
};
std::optional<IndexedRowAdd>
indexedRowAdd(mlir::func::FuncOp, const compilerinfra::TargetProfile &);
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
  unsigned rowAxis = 0;
  int64_t rows = 0;
  int64_t span = 0;
  int64_t wramBytes = 0;
  int64_t localBytes = 0;
  int64_t ownerTransferSites = 0;
  int64_t ownerTransferBytes = 0;
  int64_t stripTransferSites = 0;
  int64_t stripTransferBytes = 0;
  llvm::SmallVector<int64_t, 3> scratchOffsets;
  llvm::DenseMap<mlir::Operation *, llvm::StringRef> roles;
};

OwnerStreams ownerStreams(mlir::func::FuncOp,
                          const compilerinfra::TargetProfile &);

std::optional<PanelFoldSupply> panelFoldSupply(
    mlir::func::FuncOp, const compilerinfra::TargetProfile &,
    llvm::DenseMap<mlir::Operation *, llvm::SmallVector<int64_t>> *opOffsets =
        nullptr);
} // namespace intentdsl::dsa
