#include "Intent/Dialect/GPU/Transforms/Mapping/ProgramInterface.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;

namespace intent::gpu {

ArgumentRefAttr nextArgumentReference(func::FuncOp kernel) {
  llvm::DenseSet<ArgumentRefAttr> occupied;
  for (BlockArgument argument : kernel.getArguments())
    if (auto reference = getArgumentReference(argument))
      occupied.insert(reference);
  // A free identity exists among the first N+1 positive values for N arguments;
  // no increment of an arbitrary imported uint64 identity is necessary.
  for (uint64_t identity = 1; identity <= uint64_t(kernel.getNumArguments()) + 1; ++identity) {
    auto reference = ArgumentRefAttr::get(kernel.getContext(), identity);
    if (!occupied.contains(reference)) return reference;
  }
  llvm_unreachable("finite argument set must have a free identity");
}

FailureOr<BlockArgument> appendArgument(func::FuncOp kernel, Type type,
                                        ArgumentBindingAttr binding) {
  if (!binding || resolveArgument(kernel, binding.getReference()))
    return kernel.emitError("cannot append an absent or duplicate physical argument binding");
  if (failed(ArgumentBindingAttr::verify([&] { return kernel.emitError(); },
      binding.getReference(), binding.getKind(), binding.getPublicOrdinal(),
      binding.getSource(), binding.getAxis(), binding.getDimension()))) return failure();
  OpBuilder builder(kernel.getContext());
  unsigned position = kernel.getNumArguments();
  kernel.insertArgument(position, type, builder.getDictionaryAttr({
      builder.getNamedAttr(argumentBindingAttr, binding)}), kernel.getLoc());
  return kernel.getArgument(position);
}

LogicalResult setArgumentType(BlockArgument argument, Type type) {
  auto binding = getArgumentBinding(argument);
  if (!binding) return failure();
  auto kernel = cast<func::FuncOp>(argument.getOwner()->getParentOp());
  if (binding.getKind() != ArgumentKind::Workspace)
    return kernel.emitError("physical argument type replacement requires a private workspace");
  argument.setType(type);
  kernel.setType(FunctionType::get(kernel.getContext(), kernel.front().getArgumentTypes(),
                                    kernel.getResultTypes()));
  return success();
}

} // namespace intent::gpu
