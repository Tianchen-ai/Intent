#include "Intent/Dialect/DSA/IR/Resources.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
using namespace mlir;
namespace intent::dsa {
FailureOr<StorageResource> lookupStorage(Attribute target, Attribute space) {
  if (!target) return failure();
  auto *interface = target.getDialect().getRegisteredInterface<ResourceInterface>();
  if (!interface || !interface->supports(target)) return failure();
  for (auto resource : interface->storage(target))
    if (resource.space == space) return resource;
  return failure();
}
LogicalResult verifyResources(Attribute target, function_ref<InFlightDiagnostic()> error) {
  auto *interface = target.getDialect().getRegisteredInterface<ResourceInterface>();
  if (!interface || !interface->supports(target))
    return error() << "DSA target must implement its storage resource contract";
  llvm::DenseSet<Attribute> spaces;
  for (const auto &resource : interface->storage(target)) {
    if (!resource.space || !spaces.insert(resource.space).second)
      return error() << "DSA storage resources require unique address-space identities";
    if (resource.capacityBytes <= 0 || resource.reservedBytes < 0 ||
        resource.reservedBytes > resource.capacityBytes || resource.alignmentBytes <= 0 ||
        (resource.alignmentBytes & (resource.alignmentBytes - 1)))
      return error() << "DSA storage requires physical capacity, a separate valid reservation and power-of-two alignment";
  }
  return success();
}
}
