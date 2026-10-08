#include "Intent/Dialect/GPU/Transforms/Storage/Workspace.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Analysis/Workspace.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/IR/PhysicalExpressions.h"
#include "Intent/Dialect/GPU/Transforms/Mapping/ProgramInterface.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace intent::gpu {
namespace {

FailureOr<BlockArgument> bindWorkspace(func::FuncOp kernel, BufferOp allocation,
                                       ArrayAttr shape, uint64_t source) {
  OpBuilder builder(kernel.getContext());
  auto type = allocation.getResult().getType();
  auto workspaceType = BufferType::get(
      kernel.getContext(), type.getElementType(), shape,
      BufferScopeAttr::get(kernel.getContext(), BufferScope::InvocationWorkspace),
      type.getInstance(), type.getOwner(), type.getInitialization(),
      type.getVisibility());
  auto binding = ArgumentBindingAttr::get(
      kernel.getContext(), nextArgumentReference(kernel), ArgumentKind::Workspace,
      IntegerAttr{}, ArgumentRefAttr{}, IntegerAttr{}, IntegerAttr{});
  // Build the allocation and its stride slots together. At the transformation
  // boundary the argument has a complete ViewType and host allocation binding.
  auto workspace = appendArgument(kernel, workspaceType, binding);
  if (failed(workspace)) return failure();
  SmallVector<int64_t> dimensions;
  SmallVector<Attribute> strides;
  for (auto [axis, attribute] : llvm::enumerate(shape)) {
    auto extent = cast<PhysicalExprAttr>(attribute);
    dimensions.push_back(extent.getKind() == PhysicalExprKind::Dimension
                             ? extent.getValue() : 0);
    auto strideBinding = ArgumentBindingAttr::get(
        kernel.getContext(), nextArgumentReference(kernel), ArgumentKind::Stride,
        IntegerAttr{}, getArgumentReference(*workspace),
        builder.getI64IntegerAttr(axis), IntegerAttr{});
    auto stride = appendArgument(kernel, builder.getIndexType(), strideBinding);
    if (failed(stride)) return failure();
    strides.push_back(queryArgumentExpression(*stride));
  }
  auto layout = ViewLayoutAttr::get(
      kernel.getContext(), shape, builder.getDenseI64ArrayAttr(dimensions),
      builder.getArrayAttr(strides));
  if (failed(setArgumentType(*workspace, ViewType::get(
          kernel.getContext(), type.getElementType(), /*access=*/2, source,
          layout))))
    return failure();
  return *workspace;
}

LogicalResult verifyAllocationUses(BufferOp allocation) {
  if (allocation.getInitialValue())
    return allocation.emitOpError(
        "workspace lowering requires initialization to be explicit writes");
  for (OpOperand &use : allocation.getResult().getUses()) {
    if (isa<DimOp, AssumeInBoundsOp>(use.getOwner())) continue;
    auto access = dyn_cast<AccessOpInterface>(use.getOwner());
    if (!access || !access.isMemoryAccess() ||
        &use != &access.getAccessResourceOperand())
      return use.getOwner()->emitOpError(
          "workspace allocation may only have explicit resource accesses");
  }
  return success();
}

} // namespace

LogicalResult lowerWorkspaceAllocations(ModuleOp module) {
  auto physical = getPhysicalKernel(module);
  if (failed(physical)) return failure();
  func::FuncOp kernel = *physical;
  SmallVector<BufferOp> allocations;
  kernel.walk([&](BufferOp buffer) { allocations.push_back(buffer); });
  if (allocations.empty()) return success();
  auto space = kernel->getAttrOfType<ArrayAttr>(programSpaceAttr);
  if (!space || space.empty())
    return kernel.emitError("workspace lowering requires the current program space");
  bool singleton = llvm::all_of(space, [](Attribute attribute) {
    return constantPhysicalExpression(cast<PhysicalExprAttr>(attribute)) == 1;
  });
  bool privateSlices = !singleton && llvm::any_of(allocations, [](BufferOp buffer) {
    return !buffer.getResult().getType().isInvocationWorkspace();
  });
  SmallVector<Value> programCoordinates;
  if (privateSlices) {
    OpBuilder entry(&kernel.front(), kernel.front().begin());
    for (unsigned axis = 0; axis < space.size(); ++axis)
      programCoordinates.push_back(entry.create<ProgramIdOp>(
          kernel.getLoc(), entry.getIndexType(), axis));
  }
  uint64_t nextSource = nextPhysicalAxisIdentities(kernel).first;
  for (BufferOp allocation : allocations) {
    if (!isProgramAllocationContext(allocation, kernel))
      return allocation.emitOpError(
          "workspace ABI cannot be captured inside an isolated or pure helper");
    if (failed(verifyAllocationUses(allocation))) return failure();
    auto type = allocation.getResult().getType();
    bool prefix = privateSlices && !type.isInvocationWorkspace();
    SmallVector<Attribute> extents;
    if (prefix) llvm::append_range(extents, space);
    llvm::append_range(extents, type.getShape());
    auto shape = workspaceAllocationShape(kernel, ArrayAttr::get(
        kernel.getContext(), extents));
    if (failed(shape)) return failure();
    auto workspace = bindWorkspace(kernel, allocation, *shape, nextSource++);
    if (failed(workspace)) return failure();
    unsigned prefixRank = prefix ? programCoordinates.size() : 0;

    // Retain the chosen dimensions, not the larger envelope allocated before
    // the tuning candidate is selected.
    for (Operation *user : llvm::make_early_inc_range(allocation.getResult().getUsers())) {
      // The backing resource now has invocation visibility. Private slices
      // remain disjoint; widening the atomic scope preserves every ordering and
      // collision guarantee of the original program-local allocation.
      llvm::TypeSwitch<Operation *>(user)
          .Case<ScatterReduceOp, AtomicLoadOp, AtomicStoreOp, AtomicRMWOp,
                AtomicCompareExchangeOp>([](auto access) {
            access.setSharing(AtomicSharingDomain::KernelInvocation);
          });
      if (auto dimension = dyn_cast<DimOp>(user)) {
        OpBuilder builder(dimension);
        Value size = builder.create<PhysicalExprOp>(
            dimension.getLoc(), builder.getIndexType(),
            cast<PhysicalExprAttr>(type.getShape()[dimension.getAxis()]));
        dimension.replaceAllUsesWith(size);
        dimension.erase();
      } else if (prefix) {
        if (auto access = dyn_cast<AccessOpInterface>(user)) {
          SmallVector<Value> coordinates(programCoordinates);
          llvm::append_range(coordinates, access.getAccessCoordinates());
          SmallVector<int64_t> axes;
          for (unsigned axis = 0; axis < prefixRank; ++axis) axes.push_back(axis);
          for (int64_t axis : access.getAccessSourceAxes())
            axes.push_back(axis + prefixRank);
          access.getAccessCoordinatesMutable().assign(coordinates);
          access.setAccessSourceAxes(axes);
        } else if (auto bounds = dyn_cast<AssumeInBoundsOp>(user)) {
          bounds.setAxis(bounds.getAxis() + prefixRank);
        }
      }
    }
    OpBuilder builder(allocation);
    if (prefix)
      for (auto [axis, coordinate] : llvm::enumerate(programCoordinates))
        // Grid coordinates fit every allocation shape in the complete tuple
        // envelope. This states the program-private ownership established here.
        builder.create<AssumeInBoundsOp>(allocation.getLoc(), coordinate,
                                         *workspace, axis);
    allocation.getResult().replaceAllUsesWith(*workspace);
    allocation.erase();
  }
  eraseDeadPhysicalValues(kernel);
  return verifyGPUProgram(module);
}

} // namespace intent::gpu
