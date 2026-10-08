#pragma once

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

namespace intent::weft_provider {

// Close the generated task's executable interface before forming its host
// call. The retained positions map native arguments back to CPU task captures.
mlir::LogicalResult finalizeTaskInterface(
    ::weft::kernel::KernelOp kernel,
    llvm::SmallVectorImpl<unsigned> &argumentPositions);

} // namespace intent::weft_provider
