#ifndef INTENT_GPU_ANALYSIS_ACCESSRELATIONS_H
#define INTENT_GPU_ANALYSIS_ACCESSRELATIONS_H

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"

namespace intent::gpu::detail {

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

} // namespace intent::gpu::detail

#endif
