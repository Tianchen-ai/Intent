#ifndef INTENT_TARGET_BANGC_RESOURCES_H
#define INTENT_TARGET_BANGC_RESOURCES_H
#include "Intent/Target/DSA/BangC/Physical/IR/DSAOps.h"
#include "Intent/Dialect/DSA/IR/Resources.h"
#include "mlir/IR/Builders.h"

namespace intent::bangc {
inline bangc::physical::CapabilitiesAttr capabilities(mlir::Operation *operation) {
  for (auto *current = operation; current; current = current->getParentOp())
    if (auto value = current->getAttrOfType<bangc::physical::CapabilitiesAttr>("intent_bangc.capabilities"))
      return value;
  llvm_unreachable("verified BANG C program must carry DSA capabilities");
}

inline dsa::StorageResource storageResource(mlir::Operation *operation, int64_t space) {
  mlir::Builder builder(operation->getContext());
  auto resource = dsa::lookupStorage(capabilities(operation), builder.getI64IntegerAttr(space));
  assert(mlir::succeeded(resource) && "verified BANG C storage resource");
  return *resource;
}
inline int64_t usableSramBytes(mlir::Operation *operation) {
  return storageResource(operation, physical::sharedSpace).usableBytes();
}
} // namespace intent::bangc
#endif
