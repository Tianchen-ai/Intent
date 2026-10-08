#ifndef INTENT_CPU_TRANSFORMS_STORAGE_ACCESSALIASES_H
#define INTENT_CPU_TRANSFORMS_STORAGE_ACCESSALIASES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace intent::cpu {

// Expose scalar accesses to private storage in the allocation's coordinates so
// the existing producer replay policies can inspect the actual indexed value.
mlir::LogicalResult foldPrivateAccessAliases(mlir::func::FuncOp function);

// Compose scalar access coordinates through loop-local subviews, stopping at
// invariant descriptors so their selected shape and contiguous layout survive.
mlir::LogicalResult foldLoopAccessSubviews(mlir::scf::ForOp loop);

} // namespace intent::cpu

#endif
