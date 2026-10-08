#ifndef INTENT_TARGET_CUTILE_SERIALIZATION_NUMERICAL_H
#define INTENT_TARGET_CUTILE_SERIALIZATION_NUMERICAL_H

#include "Intent/Dialect/GPU/Serialization/PythonEmitter.h"

namespace intent::cutile {

inline constexpr llvm::StringLiteral libraryMathImport =
    "from intent.runtime.cutile import math as cutile_math";

void addNumericalOperations(gpu::PythonEmitter::Emitters &emitters);

std::string integerDivision(gpu::PhysicalExprKind kind, llvm::StringRef lhs,
                            llvm::StringRef rhs);

// Internal source constructions use the same cast/selection spelling as the
// registered numeric operations; they do not select a different computation.
std::string numericalCast(gpu::PythonEmitter &emitter, mlir::Value value,
                          mlir::Type result, bool bitcast);
std::string numericalSelect(gpu::PythonEmitter &emitter, gpu::SelectOp operation);

} // namespace intent::cutile

#endif
