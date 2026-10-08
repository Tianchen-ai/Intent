#ifndef INTENT_CONVERSION_KIRTOCPU_KIRTOCPU_H
#define INTENT_CONVERSION_KIRTOCPU_KIRTOCPU_H

#include "mlir/IR/BuiltinOps.h"

namespace intent {
enum class CPUEntryLayout { Contiguous, StridedInputs };

mlir::LogicalResult lowerCanonicalKIRToCPU(mlir::ModuleOp module,
                                         CPUEntryLayout entryLayout);
}

#endif
