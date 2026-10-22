#ifndef INTENT_TARGET_BANGC_RESOURCES_H
#define INTENT_TARGET_BANGC_RESOURCES_H
#include "Intent/Dialect/DSA/IR/DSAOps.h"

namespace intent::bangc {
inline dsa::CapabilitiesAttr capabilities(mlir::Operation *operation) {
  for (auto *current = operation; current; current = current->getParentOp())
    if (auto value = current->getAttrOfType<dsa::CapabilitiesAttr>("intent_dsa.capabilities"))
      return value;
  llvm_unreachable("verified BANG C program must carry DSA capabilities");
}

inline int64_t usableSramBytes(mlir::Operation *operation) {
  // The MLU370 implementation reserves 128 KiB for SDK/runtime use. This is
  // separate from the physical SRAM capacity reported by CNRT.
  return capabilities(operation).getSramBytes() - 128 * 1024;
}
} // namespace intent::bangc
#endif
