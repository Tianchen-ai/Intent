#pragma once
#include "intentdsl/levels/execution/cpu/ir/Owners.h"
namespace intentdsl::cpu {
// Consecutive owners may visit one input interval even when the author's
// innermost owner is a split of that interval. Writes remain owner-local.
struct OwnerReuse {
  unsigned axis;
  int64_t group, extent;
  double readBytes;
  mlir::Value quotient;
};
std::optional<OwnerReuse> sharedInterval(mlir::Operation *, const CallFacts &,
                                         const RowMachine &);
bool invariantOwnerValue(mlir::Value, mlir::Operation &, const OwnerReuse &);
llvm::SmallVector<unsigned> ownerRadices(mlir::Operation &);
} // namespace intentdsl::cpu
