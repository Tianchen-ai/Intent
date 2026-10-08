#ifndef INTENT_CPU_TRANSFORMS_TASK_ATOMICTASKOWNERSHIP_H
#define INTENT_CPU_TRANSFORMS_TASK_ATOMICTASKOWNERSHIP_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace intent::cpu {

// Assign complete destination slices to tasks, then execute each original
// parallel point once in its owner. Unknown ownership retains the original IR.
mlir::scf::ParallelOp partitionAtomicWorkset(mlir::func::FuncOp function,
                                          mlir::scf::ParallelOp root,
                                          int64_t grain);

} // namespace intent::cpu
#endif
