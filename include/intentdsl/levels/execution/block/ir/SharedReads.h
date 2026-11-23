#pragma once
#include "intentdsl/levels/execution/block/ir/Owners.h"

namespace intentdsl::block {
// The owner axis divided by a constant identifies a group with identical
// read coordinates. Only slices whose entire address is group-invariant can
// be loaded once and broadcast to those owners.
struct SharedReads {
  unsigned axis;
  int64_t group;
  llvm::SmallVector<mlir::Operation *> slices;
};
std::optional<SharedReads> sharedReads(mlir::Operation *, const CallFacts &);
bool legalSharedReads(mlir::Operation *, const compilerinfra::TargetProfile &,
                      mlir::DictionaryAttr);
mlir::LogicalResult verifySharedReads(mlir::Operation *, const CallFacts &);
} // namespace intentdsl::block
