#include "Intent/Target/DSA/Ascend/IR/Target.h"
#include "Intent/Dialect/Block/IR/Target.h"
#include "Intent/Target/Triton/IR/Target.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"
#include <limits>

using namespace mlir;
#include "Intent/Target/DSA/Ascend/IR/TargetDialect.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "Intent/Target/DSA/Ascend/IR/TargetAttrs.cpp.inc"

namespace intent::ascend {
namespace {
class Resources final : public block::TargetInterface {
public:
  explicit Resources(Dialect *dialect) : TargetInterface(dialect) {}
  bool supports(Attribute target) const final { return isa<DeviceAttr>(target); }
  block::Resources resources(Attribute target) const final {
    auto device = cast<DeviceAttr>(target);
    // CANN owns UB/L0 allocation. The driver does not expose these capacities;
    // in particular its compatibility max_shared_mem=1 is not a capacity.
    return {device.getVectorCores(), 0, 0, 0, 0};
  }
  llvm::json::Object serialize(Attribute target) const final {
    auto device = cast<DeviceAttr>(target);
    return llvm::json::Object{{"architecture", device.getArchitecture().getValue()},
            {"cube_cores", device.getCubeCores()}, {"vector_cores", device.getVectorCores()}};
  }
};

class TritonTarget final : public triton::TargetInterface {
public:
  explicit TritonTarget(Dialect *dialect) : TargetInterface(dialect) {}
  StringRef provider() const final { return "triton_ascend"; }
  bool legalOption(Attribute, block::ParameterRole role, int64_t value, bool) const final {
    if (value <= 0) return false;
    switch (role) {
    case block::ParameterRole::ProviderWarps: return llvm::isPowerOf2_64(value);
    case block::ParameterRole::ProviderStages: return value <= std::numeric_limits<int32_t>::max();
    case block::ParameterRole::ProviderCTAs: return value == 1;
    default: return false;
    }
  }
  int64_t valueWordsPerWarp(Attribute) const final { return 0; }
  FailureOr<std::string> approximateMath(Operation *, Attribute,
      triton::ApproximateMath kind, ArrayRef<std::string> operands, bool ftz) const final {
    auto bits = [](StringRef value) { return "tl.cast(" + value.str() + ", tl.uint32, bitcast=True)"; };
    auto floating = [](StringRef value) { return "tl.cast(" + value.str() + ", tl.float32, bitcast=True)"; };
    auto flush = [&](StringRef value) {
      std::string encoded = bits(value);
      return floating("tl.where((" + encoded + " & 2147483647) < 8388608, " +
          encoded + " & 2147483648, " + encoded + ")");
    };
    std::string lhs = ftz ? flush(operands[0]) : operands[0], result;
    switch (kind) {
    case triton::ApproximateMath::Divide: {
      std::string rhs = ftz ? flush(operands[1]) : operands[1];
      std::string sign = "((" + bits(lhs) + " ^ " + bits(rhs) + ") & 2147483648)";
      std::string large = "((tl.abs(" + rhs + ") > 8.507059173023462e37) & (tl.abs(" + rhs + ") < float('inf')))";
      std::string exceptional = "tl.where(tl.abs(" + lhs + ") < float('inf'), " + floating(sign) + ", float('nan'))";
      result = "tl.where(" + large + ", " + exceptional + ", libdevice.div_rn(" + lhs + ", " + rhs + "))";
      break;
    }
    case triton::ApproximateMath::Exp2:
      result = "libdevice.exp2(" + lhs + ")";
      break;
    case triton::ApproximateMath::Tanh:
      result = "tl.where((" + bits(lhs) + " & 2147483647) < 8388608, " + lhs + ", libdevice.tanh(" + lhs + "))";
      break;
    }
    return ftz ? flush(result) : result;
  }
};
}

void IntentAscendDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Intent/Target/DSA/Ascend/IR/TargetAttrs.cpp.inc"
      >();
  addInterfaces<Resources, TritonTarget>();
}

LogicalResult DeviceAttr::verify(function_ref<InFlightDiagnostic()> error,
    StringAttr architecture, int64_t cubeCores, int64_t vectorCores) {
  if (!architecture || architecture.getValue().empty() || cubeCores < 0 || vectorCores <= 0)
    return error() << "Ascend requires its actual architecture and core counts";
  return success();
}
}
