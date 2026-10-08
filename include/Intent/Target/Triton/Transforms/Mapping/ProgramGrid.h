#ifndef INTENT_TARGET_TRITON_TRANSFORMS_MAPPING_PROGRAMGRID_H
#define INTENT_TARGET_TRITON_TRANSFORMS_MAPPING_PROGRAMGRID_H

#include "mlir/IR/BuiltinOps.h"

namespace intent::triton {

mlir::LogicalResult legalizeProgramGrid(mlir::ModuleOp module);

} // namespace intent::triton

#endif
