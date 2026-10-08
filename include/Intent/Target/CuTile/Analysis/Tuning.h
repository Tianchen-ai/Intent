#ifndef INTENT_TARGET_CUTILE_ANALYSIS_TUNING_H
#define INTENT_TARGET_CUTILE_ANALYSIS_TUNING_H

#include "llvm/ADT/SmallBitVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::cutile {

// Indices in the current physical ABI whose scalar values must distinguish
// autotuning requests. Only a closed, data-only use graph may omit a value;
// symbolic shape/launch references and unknown uses retain it. This query does
// not choose configurations or change native specialization/argument passing.
llvm::SmallBitVector getTuningKeyScalarArguments(mlir::func::FuncOp kernel);

} // namespace intent::cutile

#endif
