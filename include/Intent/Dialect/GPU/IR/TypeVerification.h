#ifndef INTENT_DIALECT_GPU_IR_TYPEVERIFICATION_H
#define INTENT_DIALECT_GPU_IR_TYPEVERIFICATION_H

#include "mlir/Support/LogicalResult.h"

namespace mlir {
class Operation;
}

namespace intent::gpu {

// Recheck stored GPU types and their shape/provenance attributes using the same
// invariants as checked construction and parsing. Run before operation verifiers
// that consume those schemas; this does not repair IR or check program relations.
mlir::LogicalResult verifyGPUTypeInvariants(mlir::Operation *root);

} // namespace intent::gpu

#endif
