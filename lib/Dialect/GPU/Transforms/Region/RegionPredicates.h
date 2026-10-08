#ifndef INTENT_GPU_TRANSFORMS_REGIONPREDICATES_H
#define INTENT_GPU_TRANSFORMS_REGIONPREDICATES_H

#include "RegionSources.h"

namespace intent::gpu::region {

struct PredicatePartition {
  mlir::Value allTrueStart;
  mlir::Value allTrueStop;
  mlir::Value effectiveStart;
  mlir::Value effectiveStop;
  llvm::SmallVector<mlir::Value> allTruePredicates;
  bool firstMemberIsActive = false;
  bool prefixSpecializable = false;
};

mlir::Value physicalTailMembershipPredicate(
    RegionFoldOp fold, mlir::ValueRange identities, ParameterRefAttr segment,
    llvm::ArrayRef<SourcePlan> plans);
bool scanTailIsIdentity(RegionScanOp scan, llvm::ArrayRef<SourcePlan> plans,
                        mlir::ValueRange identities);
mlir::FailureOr<PredicatePartition>
predicatePartition(mlir::OpBuilder &builder, RegionFoldOp fold,
                   llvm::ArrayRef<SourcePlan> plans, MakeRangeOp master,
                   mlir::Value masterExtent, mlir::ValueRange identities,
                   mlir::Value segment);

} // namespace intent::gpu::region
#endif
