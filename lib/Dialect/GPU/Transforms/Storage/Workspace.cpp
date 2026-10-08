#include "Intent/Dialect/GPU/Transforms/Storage/Workspace.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;

namespace intent::gpu {
namespace {

BufferOp allocate(OpBuilder &builder, Location location, func::FuncOp kernel,
                  Type elementType, ArrayAttr shape, BufferScope scope,
                  uint64_t owner) {
  llvm::DenseSet<uint64_t> occupied;
  kernel.walk([&](BufferOp buffer) {
    occupied.insert(buffer.getResult().getType().getInstance());
  });
  // Allocation identities are local to this program, not monotonically growing
  // origin IDs or positions in a separately maintained resource table.
  uint64_t instance = 1;
  while (occupied.contains(instance)) ++instance;
  auto type = BufferType::get(
      kernel.getContext(), elementType, shape,
      BufferScopeAttr::get(kernel.getContext(), scope), instance, owner,
      BufferInitializationAttr::get(kernel.getContext(),
                                     BufferInitialization::FirstWrite),
      /*visibility=*/1);
  return builder.create<BufferOp>(location, type, Value{});
}

} // namespace

bool isProgramAllocationContext(Operation *operation, func::FuncOp kernel) {
  for (Operation *parent = operation->getParentOp(); parent != kernel;
       parent = parent->getParentOp())
    if (!parent || parent->hasTrait<OpTrait::IsIsolatedFromAbove>() ||
        !parent->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
      return false;
  return true;
}

BufferOp createProgramBuffer(OpBuilder &builder, Location location,
                             Type elementType, ArrayAttr shape, uint64_t owner) {
  auto kernel = builder.getInsertionBlock()->getParentOp();
  auto function = dyn_cast<func::FuncOp>(kernel);
  if (!function) function = kernel->getParentOfType<func::FuncOp>();
  return allocate(builder, location, function, elementType, shape,
                  BufferScope::ProgramPrivate, owner);
}

BufferOp createInvocationBuffer(func::FuncOp kernel, Location location,
                                Type elementType, ArrayAttr shape,
                                uint64_t owner) {
  OpBuilder builder(&kernel.front(), kernel.front().begin());
  return allocate(builder, location, kernel, elementType, shape,
                  BufferScope::InvocationWorkspace, owner);
}

} // namespace intent::gpu
