#ifndef INTENT_GPU_TRANSFORMS_REGIONSOURCES_H
#define INTENT_GPU_TRANSFORMS_REGIONSOURCES_H

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include <memory>

namespace intent::gpu::region {

struct SourcePlan {
  mlir::Value source;
  PhysicalSourceAxis sourceIdentity;
  unsigned sourceAxis;
  bool hasLoads;
  llvm::SmallVector<MakeRangeOp> ranges;
  mlir::Attribute tailConstant;
};

PhysicalExprAttr parameterExtent(ParameterRefAttr parameter);

FragmentType replaceSliceAxis(FragmentType source, unsigned axis,
                              PhysicalExprAttr extent,
                              AxisMapAttr segmentMapping);
FragmentType predicateType(FragmentType source);
bool isUnitExtent(mlir::Attribute attribute);

mlir::FailureOr<llvm::SmallVector<SourcePlan>>
prepareRegionSources(mlir::ValueRange sources, unsigned sourceAxis,
                     mlir::Operation *insertionAnchor);

// periodStart/period may be supplied only for a proved periodic traversal;
// active members must stay within that period even in the final partial slice.
mlir::LogicalResult buildSourceSlices(
    mlir::OpBuilder &builder, mlir::Location location,
    llvm::ArrayRef<SourcePlan> plans, llvm::ArrayRef<FragmentType> sliceTypes,
    mlir::Value offset, mlir::Value segment, PhysicalExprAttr sliceExtent,
    bool fullSegment, llvm::SmallVectorImpl<mlir::Value> &slices,
    mlir::Value &segmentTail, mlir::IRMapping &sliceMapping,
    llvm::SmallVectorImpl<std::shared_ptr<mlir::IRMapping>> &sourceMappings,
    mlir::Operation *insertionAnchor, std::string &reason,
    mlir::Value traversalStop = {}, mlir::Value periodStart = {},
    mlir::Value period = {});

} // namespace intent::gpu::region

#endif
