#ifndef INTENT_GPU_TRANSFORMS_REGIONCLONING_H
#define INTENT_GPU_TRANSFORMS_REGIONCLONING_H

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"

namespace intent::gpu::region {

struct SliceRelation {
  PhysicalSourceAxis source;
  AxisMapAttr segmentMapping;
};

mlir::FailureOr<llvm::SmallVector<mlir::Value>> inlinePureRegion(
    mlir::OpBuilder &builder, mlir::Region &region, mlir::ValueRange arguments,
    std::string &reason, mlir::Value substituteSource = {},
    mlir::Value substituteTarget = {}, mlir::Value conjunctSource = {},
    mlir::Value conjunctPredicate = {}, mlir::Value additionalSource = {},
    mlir::Value additionalTarget = {}, mlir::IRMapping *resultMapping = nullptr);

mlir::LogicalResult collectScanOutputConsumers(
    RegionScanOp scan, llvm::SmallVectorImpl<mlir::Operation *> &ordered,
    std::string &reason);
mlir::LogicalResult cloneScanOutputConsumers(
    mlir::OpBuilder &builder, mlir::Location location,
    llvm::ArrayRef<mlir::Operation *> consumers, mlir::IRMapping &mapping,
    llvm::ArrayRef<SliceRelation> relations, PhysicalExprAttr sliceExtent,
    mlir::Value segmentTail, RegionScanOp scan, mlir::Value offset,
    mlir::Value segment, std::string &reason);

} // namespace intent::gpu::region
#endif
