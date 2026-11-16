#ifndef INTENT_DSA_RESOURCES_H
#define INTENT_DSA_RESOURCES_H
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectInterface.h"
#include "llvm/ADT/SmallVector.h"

namespace intent::dsa {
enum class Visibility { Participant, Group, Invocation };
struct StorageResource {
  mlir::Attribute space;
  Visibility visibility;
  int64_t capacityBytes;
  int64_t reservedBytes;
  int64_t alignmentBytes;
  int64_t usableBytes() const { return capacityBytes - reservedBytes; }
  bool fits(int64_t liveBytes, int64_t budgetBytes) const {
    return liveBytes >= 0 && budgetBytes >= 0 && liveBytes <= budgetBytes &&
           liveBytes <= usableBytes();
  }
};
/// The concrete target owns address-space identities, SDK reservations and
/// visibility. Shared DSA clients do not infer these from allocation rank.
class ResourceInterface : public mlir::DialectInterface::Base<ResourceInterface> {
public:
  explicit ResourceInterface(mlir::Dialect *dialect) : Base(dialect) {}
  virtual bool supports(mlir::Attribute target) const = 0;
  virtual llvm::SmallVector<StorageResource> storage(mlir::Attribute target) const = 0;
};
mlir::FailureOr<StorageResource> lookupStorage(mlir::Attribute target, mlir::Attribute space);
mlir::LogicalResult verifyResources(mlir::Attribute target,
    llvm::function_ref<mlir::InFlightDiagnostic()> error);
}
#endif
