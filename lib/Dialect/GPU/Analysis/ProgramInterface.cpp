#include "Intent/Dialect/GPU/Analysis/ProgramInterface.h"

using namespace mlir;

namespace intent::gpu {

FailureOr<ProgramInterface> ProgramInterface::read(func::FuncOp kernel) {
  if (failed(verifyProgramInterface(kernel))) return failure();
  ProgramInterface result;
  result.interface = intent::getPublicInterface(kernel);
  for (BlockArgument argument : kernel.getArguments()) {
    auto binding = getArgumentBinding(argument);
    result.entries.push_back({argument, binding});
    result.references.try_emplace(binding.getReference(), argument);
  }
  return result;
}

BlockArgument ProgramInterface::resolve(ArgumentRefAttr reference) const {
  return references.lookup(reference);
}
BlockArgument ProgramInterface::resolve(PhysicalExprAttr expression) const {
  return expression ? resolve(expression.getArgumentReference()) : BlockArgument{};
}
BlockArgument ProgramInterface::publicArgument(unsigned ordinal) const {
  for (const auto &entry : entries)
    if (entry.binding.getKind() == ArgumentKind::Public &&
        entry.binding.getPublicOrdinal().getInt() == ordinal)
      return entry.value;
  return {};
}
BlockArgument ProgramInterface::dimension(int64_t identity) const {
  for (const auto &entry : entries)
    if (entry.binding.getKind() == ArgumentKind::Dimension &&
        entry.binding.getDimension().getInt() == identity)
      return entry.value;
  return {};
}
BlockArgument ProgramInterface::stride(ArgumentRefAttr view, unsigned axis) const {
  for (const auto &entry : entries)
    if (entry.binding.getKind() == ArgumentKind::Stride &&
        entry.binding.getSource() == view && entry.binding.getAxis().getInt() == axis)
      return entry.value;
  return {};
}
intent::PublicParameterAttr ProgramInterface::publicParameter(ArgumentRefAttr reference) const {
  return intent::gpu::publicParameter(resolve(reference));
}
intent::PublicParameterAttr ProgramInterface::publicParameter(Value value) const {
  return intent::gpu::publicParameter(value);
}

} // namespace intent::gpu
