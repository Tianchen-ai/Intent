#ifndef INTENT_TARGET_TRITON_SERIALIZATION_NUMERICAL_H
#define INTENT_TARGET_TRITON_SERIALIZATION_NUMERICAL_H

#include "Intent/Dialect/GPU/Serialization/PythonEmitter.h"

namespace intent::triton {

// The same typed translation owns source eligibility, spelling and imports.
void addNumericalOperations(gpu::PythonEmitter::Emitters &emitters);
mlir::LogicalResult emitNumericalConstant(mlir::arith::ConstantOp operation,
                                        gpu::PythonEmitter &emitter);

std::string integerDivision(gpu::PhysicalExprKind kind, llvm::StringRef lhs,
                            llvm::StringRef rhs);

mlir::FailureOr<std::string>
numericalCast(mlir::Operation *diagnostic, mlir::Type source, mlir::Type target,
             llvm::StringRef value, bool bitcast);

} // namespace intent::triton

#endif
