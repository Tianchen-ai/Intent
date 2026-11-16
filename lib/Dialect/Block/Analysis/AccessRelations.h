#ifndef INTENT_Block_ANALYSIS_ACCESSRELATIONS_H
#define INTENT_Block_ANALYSIS_ACCESSRELATIONS_H

#include "Intent/Dialect/Block/Analysis/PhysicalProgram.h"

namespace intent::block::detail {

bool matchesResourceExtent(mlir::Value value, mlir::Value resource, unsigned axis);
bool capacityCoversResourceExtent(mlir::Value value, mlir::Value resource,
                                  unsigned axis);
bool upperBoundWithinResource(mlir::Value value, mlir::Value resource,
                              unsigned axis);
bool derivesFromAccessCoordinate(mlir::Value value, mlir::Value coordinate);
bool isInclusiveCoordinateUpperBound(mlir::Value value, mlir::Value coordinate);
bool coordinateRangeWithinResource(mlir::Value coordinate, mlir::Value resource,
                                   unsigned axis);
bool hasExactPhysicalRangeCoverage(mlir::Value coordinate, mlir::Value upperBound);

} // namespace intent::block::detail

#endif
