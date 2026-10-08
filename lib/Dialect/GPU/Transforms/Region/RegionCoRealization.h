#ifndef INTENT_GPU_TRANSFORMS_REGIONCOREALIZATION_H
#define INTENT_GPU_TRANSFORMS_REGIONCOREALIZATION_H

#include "../Reduction/OnlineSummary.h"
#include "RegionSummary.h"
#include "mlir/IR/IRMapping.h"

namespace intent::gpu::region {

struct OnlineRegionPlan {
  OnlineSummaryStructure summary;
  OnlineSummaryMerge merge;
};

struct AdditiveRegionContract {
  ContractOp summary;
  BinaryOp merge;
};

llvm::SmallVector<AdditiveRegionContract>
additiveRegionContracts(RegionFoldOp fold, mlir::ValueRange identities);
void coRealizeAdditiveRegion(
    mlir::OpBuilder &builder, mlir::Location location,
    llvm::ArrayRef<AdditiveRegionContract> contracts,
    mlir::IRMapping &summaryMapping, mlir::IRMapping &mergeMapping);
std::optional<OnlineRegionPlan>
onlineRegionPlan(RegionFoldOp fold,
                 const std::optional<SummaryEmptinessPlan> &emptiness);
mlir::FailureOr<mlir::Value> coRealizeOnlineRegion(
    mlir::OpBuilder &builder, mlir::Location location, OnlineRegionPlan &plan,
    mlir::IRMapping &summaryMapping, mlir::IRMapping &mergeMapping,
    mlir::Value encodedCarry = {});

} // namespace intent::gpu::region
#endif
