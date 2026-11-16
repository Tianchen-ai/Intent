#include "Intent/Target/NVIDIA/IR/Target.h"
#include "Intent/Dialect/Block/IR/BlockAttrs.h"
#include "Intent/Dialect/Block/IR/Program.h"
#include "Intent/Dialect/Block/IR/Target.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
#include "Intent/Target/NVIDIA/IR/TargetDialect.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "Intent/Target/NVIDIA/IR/TargetAttrs.cpp.inc"

namespace intent::nvidia {
namespace {
class Resources final : public block::TargetInterface {
public:
  explicit Resources(Dialect *dialect) : TargetInterface(dialect) {}
  bool supports(Attribute target) const final { return isa<DeviceAttr>(target); }
  block::Resources resources(Attribute target) const final {
    auto device = cast<DeviceAttr>(target);
    return {device.getComputeUnits(), device.getRegistersPerUnit(),
            device.getMaxDynamicSharedMemoryPerBlock(), device.getMaxThreadsPerBlock(),
            device.getSingleToDoublePrecisionPerfRatio()};
  }
  llvm::json::Object serialize(Attribute target) const final {
    auto device = cast<DeviceAttr>(target);
    return llvm::json::Object{{"compute_units", device.getComputeUnits()},
      {"shared_memory_per_unit", device.getSharedMemoryPerUnit()},
      {"max_dynamic_shared_memory_per_block", device.getMaxDynamicSharedMemoryPerBlock()},
      {"registers_per_unit", device.getRegistersPerUnit()},
      {"max_threads_per_block", device.getMaxThreadsPerBlock()},
      {"compute_capability_major", device.getComputeCapabilityMajor()},
      {"compute_capability_minor", device.getComputeCapabilityMinor()},
      {"single_to_double_precision_perf_ratio", device.getSingleToDoublePrecisionPerfRatio()},
      {"matrix_units", device.getMatrixUnits()}, {"dynamic_vector_width", device.getDynamicVectorWidth()}};
  }
};
}

void IntentNVIDIADialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Intent/Target/NVIDIA/IR/TargetAttrs.cpp.inc"
      >();
  addInterfaces<Resources>();
}

LogicalResult DeviceAttr::verify(function_ref<InFlightDiagnostic()> error,
    int64_t computeUnits, int64_t sharedMemoryPerUnit, int64_t maxDynamicSharedMemoryPerBlock,
    int64_t registersPerUnit, int64_t maxThreadsPerBlock, int64_t computeCapabilityMajor,
    int64_t computeCapabilityMinor, int64_t singleToDoublePrecisionPerfRatio,
    bool matrixUnits, bool dynamicVectorWidth) {
  if (computeUnits <= 0 || sharedMemoryPerUnit <= 0 || maxDynamicSharedMemoryPerBlock <= 0 ||
      registersPerUnit <= 0 || maxThreadsPerBlock <= 0 || computeCapabilityMajor <= 0 ||
      computeCapabilityMinor < 0 || singleToDoublePrecisionPerfRatio <= 0)
    return error() << "NVIDIA device requires complete positive hardware limits";
  return success();
}

DeviceAttr deviceOf(Operation *operation) {
  while (operation) {
    if (auto capabilities = operation->getAttrOfType<block::CapabilitiesAttr>(block::capabilitiesAttr))
      return dyn_cast<DeviceAttr>(capabilities.getTarget());
    operation = operation->getParentOp();
  }
  return {};
}
}
