#ifndef INTENT_TARGET_TRITON_TRANSFORMS_SUPPLY_SUPPLY_H
#define INTENT_TARGET_TRITON_TRANSFORMS_SUPPLY_SUPPLY_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace intent::triton::detail {

bool hasOrderedViewDependencies(mlir::func::FuncOp kernel);
mlir::LogicalResult legalizeOrderedViewDependencies(mlir::func::FuncOp kernel);
llvm::SmallVector<mlir::scf::ForOp> findLoadPipelineLoops(mlir::func::FuncOp kernel);
void selectRecurrencePipelineStages(mlir::func::FuncOp kernel);
void selectOrderedLoadUnrolling(mlir::func::FuncOp kernel);

} // namespace intent::triton::detail

#endif
