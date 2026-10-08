#ifndef INTENT_TARGET_TRITON_TRANSFORMS_VALUE_VALUES_H
#define INTENT_TARGET_TRITON_TRANSFORMS_VALUE_VALUES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::triton::detail {

void selectContractForms(mlir::func::FuncOp kernel);
void canonicalizeBroadcastProjections(mlir::func::FuncOp kernel);
void sinkSelectProducers(mlir::func::FuncOp kernel);

} // namespace intent::triton::detail

#endif
