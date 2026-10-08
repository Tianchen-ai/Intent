#ifndef INTENT_ANALYSIS_REGIONSEMANTICS_H
#define INTENT_ANALYSIS_REGIONSEMANTICS_H

#include "Intent/Analysis/UniformValues.h"

namespace intent {

struct CoordinateInterval {
  mlir::Value begin, end;
};
struct PredicateIntervals {
  mlir::Value allTrueBegin, allTrueEnd;
  mlir::Value possibleBegin, possibleEnd;
};
using BuildIndexExpression = std::function<mlir::Value(UniformKind, mlir::Value, mlir::Value)>;

// Adapters supply unit-step, nonempty capture intervals in the current index
// domain. Coordinate differences must be representable; results are source-relative.
std::optional<PredicateIntervals> partitionCoordinatePredicate(
    UniformPredicate predicate, CoordinateInterval source, CoordinateInterval capture,
    mlir::Value zero, mlir::Value one, const BuildIndexExpression &build);

bool isBooleanUnion(const UniformValueAnalysis &values, mlir::Value result,
                    mlir::Value lhs, mlir::Value rhs);
bool hasTrueStateInvariant(const UniformValueAnalysis &values, mlir::Value result,
                           mlir::Value incomingState);

}
#endif
