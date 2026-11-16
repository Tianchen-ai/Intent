#include "Intent/Target/NVIDIA/IR/Target.h"
#include "Intent/Dialect/Block/IR/BlockAttrs.h"
#include "Intent/Dialect/Block/IR/Program.h"
#include "Intent/Dialect/Block/IR/Target.h"
#include "Intent/Target/Triton/IR/Target.h"
#include "llvm/Support/MathExtras.h"
#include <limits>
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
class TritonTarget final : public triton::TargetInterface {
public:
  explicit TritonTarget(Dialect *dialect) : TargetInterface(dialect) {}
  StringRef provider() const final { return "triton"; }
  bool legalOption(Attribute target, triton::LaunchOption role, int64_t value,
                   bool singleGroup) const final {
    auto device = cast<DeviceAttr>(target);
    if (value <= 0) return false;
    switch (role) {
    case triton::LaunchOption::Warps:
      return llvm::isPowerOf2_64(value) && value <= device.getMaxThreadsPerBlock() / 32;
    case triton::LaunchOption::Stages:
      return value <= std::numeric_limits<int32_t>::max();
    case triton::LaunchOption::CTAs:
      return llvm::isPowerOf2_64(value) && value <= 16 && (!singleGroup || value == 1) &&
          (value == 1 || (device.getComputeCapabilityMajor() >= 9 && device.getComputeCapabilityMajor() != 12));
    default: return false;
    }
  }
  int64_t valueWordsPerWarp(Attribute) const final { return 32 * 255; }
  FailureOr<std::string> approximateMath(Operation *operation, Attribute target,
      triton::ApproximateMath kind, ArrayRef<std::string> operands, bool ftz) const final {
    std::string instruction, constraints, arguments = operands.front();
    switch (kind) {
    case triton::ApproximateMath::Divide:
      instruction = "div.approx" + std::string(ftz ? ".ftz" : "") + ".f32 $0, $1, $2;";
      constraints = "=f,f,f";
      arguments += ", " + operands[1];
      break;
    case triton::ApproximateMath::Exp2:
      instruction = "ex2.approx" + std::string(ftz ? ".ftz" : "") + ".f32 $0, $1;";
      constraints = "=f,f";
      break;
    case triton::ApproximateMath::Tanh: {
      auto device = cast<DeviceAttr>(target);
      if (10 * device.getComputeCapabilityMajor() + device.getComputeCapabilityMinor() < 75)
        return operation->emitOpError("native approximate tanh requires compute capability 7.5 or newer"), failure();
      instruction = "tanh.approx.f32 $0, $1;";
      constraints = "=f,f";
      break;
    }
    }
    return "tl.inline_asm_elementwise(\"" + instruction + "\", constraints=\"" + constraints +
        "\", args=[" + arguments + "], dtype=tl.float32, is_pure=True, pack=1)";
  }
};
}

void IntentNVIDIADialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Intent/Target/NVIDIA/IR/TargetAttrs.cpp.inc"
      >();
  addInterfaces<Resources, TritonTarget>();
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
