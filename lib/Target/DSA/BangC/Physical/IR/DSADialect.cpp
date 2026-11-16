#include "Intent/Target/DSA/BangC/Physical/IR/DSAOps.h"
#include "Intent/Dialect/DSA/IR/Resources.h"
#include "Intent/Dialect/Block/IR/Target.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"
using namespace mlir;
using namespace intent::bangc::physical;
#include "Intent/Target/DSA/BangC/Physical/IR/DSADialect.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "Intent/Target/DSA/BangC/Physical/IR/DSAAttrs.cpp.inc"
namespace {
class StorageResources final : public intent::dsa::ResourceInterface {
public:
  explicit StorageResources(Dialect *dialect) : ResourceInterface(dialect) {}
  bool supports(Attribute target) const final { return isa<CapabilitiesAttr>(target); }
  SmallVector<intent::dsa::StorageResource> storage(Attribute target) const final {
    auto device = cast<CapabilitiesAttr>(target);
    Builder builder(target.getContext());
    using intent::dsa::Visibility;
    return {
        {builder.getI64IntegerAttr(nramSpace), Visibility::Participant, device.getNramBytes(), 0, 128},
        {builder.getI64IntegerAttr(matrixSpace), Visibility::Participant, device.getWramBytes(), 0, 64},
        {builder.getI64IntegerAttr(sharedSpace), Visibility::Group, device.getSramBytes(), 128 * 1024, 128}};
  }
};
class BlockResources final : public intent::block::TargetInterface {
public:
  explicit BlockResources(Dialect *dialect) : TargetInterface(dialect) {}
  bool supports(Attribute target) const final { return isa<CapabilitiesAttr>(target); }
  intent::block::Resources resources(Attribute target) const final {
    auto device = cast<CapabilitiesAttr>(target);
    return {device.getClusters() * device.getCoresPerCluster(), device.getNramBytes() / 4,
            device.getNramBytes(), 0, 0};
  }
  llvm::json::Object serialize(Attribute target) const final {
    auto device = cast<CapabilitiesAttr>(target);
    return llvm::json::Object{{"clusters", device.getClusters()}, {"cores_per_cluster", device.getCoresPerCluster()},
        {"nram_bytes", device.getNramBytes()}, {"wram_bytes", device.getWramBytes()}, {"sram_bytes", device.getSramBytes()}};
  }
};
}
void IntentBangCPhysicalDialect::initialize() {
  addInterfaces<StorageResources, BlockResources>();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Intent/Target/DSA/BangC/Physical/IR/DSAAttrs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "Intent/Target/DSA/BangC/Physical/IR/DSAOps.cpp.inc"
      >();
}
LogicalResult EntryRequirementsAttr::verify(function_ref<InFlightDiagnostic()> error,
                                            bool disjointOutputs) {
  return disjointOutputs ? success() : error() << "DSA entry requires disjoint writable views";
}
LogicalResult CapabilitiesAttr::verify(function_ref<InFlightDiagnostic()> error,
    int64_t clusters, int64_t cores, int64_t nram, int64_t wram, int64_t sram) {
  if (clusters <= 0 || cores <= 0 || nram <= 0 || wram <= 0 || sram <= 0)
    return error() << "DSA hardware capabilities require positive topology and storage capacities";
  return success();
}
LogicalResult ConfigurationAttr::verify(function_ref<InFlightDiagnostic()> error,
    int64_t tile, int64_t m, int64_t n, int64_t k, int64_t region, int64_t tasks, int64_t bytes) {
  if (tile <= 0 || tile % 64 || m <= 0 || n <= 0 || k <= 0 || region <= 0 || tasks <= 0 || bytes <= 0)
    return error() << "DSA configuration requires positive blocks/resources and a vector tile divisible by 64";
  return success();
}
