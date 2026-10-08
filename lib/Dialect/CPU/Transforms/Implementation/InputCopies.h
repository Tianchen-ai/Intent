#ifndef INTENT_CPU_TRANSFORMS_INPUTCOPIES_H
#define INTENT_CPU_TRANSFORMS_INPUTCOPIES_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"

namespace intent::cpu {

// Copy the actual source window into prepared axis order: all other source axes
// followed by innerAxis. Keep the read structured until producer supply folds.
mlir::linalg::GenericOp createInputCopy(mlir::OpBuilder &builder,
    mlir::Location location, mlir::Value source, mlir::Value destination,
    unsigned innerAxis);

// Optional materialization for an already selected native-vector consumer.
// False leaves the copy unchanged; true replaces it completely. A matched copy
// whose standard vector lowering cannot close is a legalization failure.
mlir::FailureOr<bool> tryMaterializeInputCopy(mlir::linalg::GenericOp operation,
    int64_t width, mlir::OpBuilder::Listener *listener);

} // namespace intent::cpu
#endif
