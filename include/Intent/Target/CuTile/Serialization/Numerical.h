#ifndef INTENT_TARGET_CUTILE_SERIALIZATION_NUMERICAL_H
#define INTENT_TARGET_CUTILE_SERIALIZATION_NUMERICAL_H

#include "Intent/Dialect/Block/Serialization/PythonEmitter.h"

namespace intent::cutile {

inline constexpr llvm::StringLiteral libraryMathImport =
    "from intent.runtime.cutile import math as cutile_math";

void addNumericalOperations(block::PythonEmitter::Emitters &emitters);

std::string integerDivision(block::PhysicalExprKind kind, llvm::StringRef lhs,
                            llvm::StringRef rhs);

// Internal source constructions use the same cast/selection spelling as the
// registered numeric operations; they do not select a different computation.
std::string numericalCast(block::PythonEmitter &emitter, mlir::Value value,
                          mlir::Type result, bool bitcast);
std::string numericalSelect(block::PythonEmitter &emitter, block::SelectOp operation);

} // namespace intent::cutile

#endif
