#include "Intent/Dialect/Block/IR/Target.h"
#include "mlir/IR/Dialect.h"

namespace intent::block {
const TargetInterface *targetInterface(mlir::Attribute target) {
  if (!target) return nullptr;
  auto *interface = target.getDialect().getRegisteredInterface<TargetInterface>();
  return interface && interface->supports(target) ? interface : nullptr;
}

Resources queryResources(mlir::Attribute target) {
  auto *interface = targetInterface(target);
  assert(interface && "block capabilities must have a verified target");
  return interface->resources(target);
}
}
