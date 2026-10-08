#ifndef INTENT_CONVERSION_KIRTOGPU_CONSTRUCTIONSCHEMA_H
#define INTENT_CONVERSION_KIRTOGPU_CONSTRUCTIONSCHEMA_H

#include "Intent/Analysis/CanonicalKernel.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "mlir/IR/Builders.h"

namespace intent::detail {

// The immutable canonical relation determines which logical axes correspond.
// Current physical values supply execution prefixes and extent authorities.
// The resulting Reshape/Broadcast edges make that proof local to GPU IR; no
// subsequent pass consults canonical KIR.
mlir::LogicalResult alignPointwiseOperands(
    mlir::OpBuilder &builder, mlir::Operation *canonical,
    CanonicalKernelAnalysis &analysis, mlir::func::FuncOp kernel,
    gpu::FragmentType logicalSeed, mlir::MutableArrayRef<mlir::Value> values,
    llvm::ArrayRef<unsigned> operandNumbers);

} // namespace intent::detail
#endif
