#include "Intent/Dialect/GPU/Transforms/Mapping/ExecutionGroups.h"

using namespace mlir;

namespace intent::gpu {

ExecutionGroupOp createExecutionGroup(
    OpBuilder &builder, Location location, Value linear, ValueRange extents,
    ArrayAttr launchExtents, DenseI64ArrayAttr roles, int64_t identity,
    PhysicalExprAttr offset, PhysicalExprAttr length) {
  auto group = builder.create<ExecutionGroupOp>(
      location, linear, extents, launchExtents, roles,
      builder.getI64IntegerAttr(identity), offset, length);
  Block *body = new Block;
  group.getBody().push_back(body);
  for (Value extent : extents)
    body->addArgument(extent.getType(), location);
  OpBuilder::atBlockEnd(body).create<ExecutionGroupYieldOp>(location);
  return group;
}

ExecutionGroupOp rebuildExecutionGroup(
    OpBuilder &builder, ExecutionGroupOp group, ValueRange extents,
    ArrayAttr launchExtents, DenseI64ArrayAttr roles, PhysicalExprAttr length) {
  auto replacement = createExecutionGroup(
      builder, group.getLoc(), group.getLinear(), extents, launchExtents,
      roles, group.getGroupId(), group.getSegmentOffset(), length);
  for (Operation &operation :
       llvm::make_early_inc_range(group.getBody().front().without_terminator()))
    operation.moveBefore(replacement.getBody().front().getTerminator());
  return replacement;
}

LogicalResult lowerExecutionGroups(ModuleOp module) {
  SmallVector<ExecutionGroupOp> groups;
  module.walk<WalkOrder::PostOrder>(
      [&](ExecutionGroupOp group) { groups.push_back(group); });
  for (ExecutionGroupOp group : groups) {
    OpBuilder builder(group);
    auto decoded = builder.create<DelinearizeOp>(
        group.getLoc(), SmallVector<Type>(group.getCoordinates().size(),
                                         builder.getIndexType()),
        group.getLinear(), group.getExtents());
    for (auto [coordinate, value] :
         llvm::zip(group.getCoordinates(), decoded.getCoordinates()))
      coordinate.replaceAllUsesWith(value);
    for (Operation &operation :
         llvm::make_early_inc_range(group.getBody().front().without_terminator()))
      operation.moveBefore(group);
    group.erase();
  }
  return success();
}

} // namespace intent::gpu
