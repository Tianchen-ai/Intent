#ifndef INTENT_DIALECT_CPU_ANALYSIS_REGIONPARTITION_H
#define INTENT_DIALECT_CPU_ANALYSIS_REGIONPARTITION_H

#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "llvm/ADT/DenseMap.h"
#include <optional>

namespace intent::cpu {

// A proof over one current region program, consumed before any IR mutation.
struct RegionPartition {
  llvm::DenseMap<mlir::Value, unsigned> axes;
  llvm::DenseMap<mlir::Operation *, unsigned> loops;
  int64_t width;
};

std::optional<RegionPartition> analyzeRegionPartition(RegionOpInterface program,
                                                     int64_t tileM, int64_t tileN);

}
#endif
