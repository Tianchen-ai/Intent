#ifndef INTENT_TARGET_TRITON_IR_TARGET_H
#define INTENT_TARGET_TRITON_IR_TARGET_H

#include "Intent/Dialect/Block/IR/BlockAttrs.h"
#include "Intent/Dialect/Block/IR/Program.h"
#include "mlir/IR/DialectInterface.h"

namespace intent::triton {
enum class ApproximateMath { Divide, Exp2, Tanh };
enum class LaunchOption { Warps, Stages, CTAs };

/// The Triton surface is shared by different device compilers. Device options,
/// resource estimates and numerical intrinsics belong to the selected target.
class TargetInterface : public mlir::DialectInterface::Base<TargetInterface> {
public:
  explicit TargetInterface(mlir::Dialect *dialect) : Base(dialect) {}
  virtual llvm::StringRef provider() const = 0;
  virtual bool legalOption(mlir::Attribute target, LaunchOption role,
                           int64_t value, bool singleGroup) const = 0;
  virtual int64_t valueWordsPerWarp(mlir::Attribute target) const = 0;
  virtual mlir::FailureOr<std::string> approximateMath(
      mlir::Operation *operation, mlir::Attribute target, ApproximateMath kind,
      llvm::ArrayRef<std::string> operands, bool flushToZero) const = 0;
};

inline const TargetInterface *targetInterface(mlir::Attribute target) {
  return target ? target.getDialect().getRegisteredInterface<TargetInterface>() : nullptr;
}

inline mlir::Attribute targetOf(mlir::Operation *operation) {
  for (; operation; operation = operation->getParentOp())
    if (auto capabilities = operation->getAttrOfType<block::CapabilitiesAttr>(block::capabilitiesAttr))
      return capabilities.getTarget();
  return {};
}
} // namespace intent::triton
#endif
