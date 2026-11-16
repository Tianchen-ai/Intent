#ifndef INTENT_TARGET_CUTILE_TRANSFORMS_ACCESS_TILEPLAN_H
#define INTENT_TARGET_CUTILE_TRANSFORMS_ACCESS_TILEPLAN_H

#include "Intent/Dialect/Block/Analysis/PhysicalProgram.h"

namespace intent::cutile {

struct NativeTileAxisPlan {
  llvm::SmallVector<unsigned> computationAxes;
  mlir::Value scalarIndex;
  int64_t divisor = 0;
  mlir::OpFoldResult modulus;
  llvm::SmallVector<std::pair<block::MakeRangeOp, int64_t>> ranges;
  llvm::SmallVector<std::pair<mlir::Value, int64_t>> offsets;
  bool originInBounds = false;
};

struct NativeTileAccessPlan {
  block::FragmentType resourceType;
  block::FragmentType packedType;
  mlir::ArrayAttr resourceToPacked;
  mlir::ArrayAttr packedToResource;
  llvm::SmallVector<NativeTileAxisPlan> axes;
  llvm::SmallVector<int64_t> toComputation;
  llvm::SmallVector<int64_t> toResource;
};

// Current-IR eligibility and packing relation; consumed within native formation.
mlir::FailureOr<NativeTileAccessPlan> analyzeNativeTileAccess(
    block::AccessOpInterface access, mlir::func::FuncOp kernel,
    const block::PhysicalAccessBoundsFact &accessBounds);

} // namespace intent::cutile

#endif
