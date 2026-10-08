#ifndef INTENT_DIALECT_CPU_TRANSFORMS_VECTOR_VECTORIZATION_H
#define INTENT_DIALECT_CPU_TRANSFORMS_VECTOR_VECTORIZATION_H

#include "mlir/Dialect/SCF/IR/SCF.h"

namespace intent::cpu {

mlir::LogicalResult vectorizeLoop(mlir::scf::ForOp loop, int64_t width,
                                 int64_t replicas);

} // namespace intent::cpu
#endif
