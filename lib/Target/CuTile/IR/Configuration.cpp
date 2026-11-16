#include "Intent/Target/CuTile/IR/Configuration.h"
#include "Intent/Target/CuTile/IR/CuTileOps.h"
namespace intent::cutile {
bool isLegalAccessForm(int64_t value) {
  return value == nativeAccessForm || value == gatherAccessForm ||
         value == nativeNoTMAForm;
}

bool isLegalOccupancy(int64_t value) { return value >= 1 && value <= 32; }

bool isLegalWorkerWarps(int64_t value) {
  return value == inferredWorkerWarps || value == 4 || value == 8;
}

bool isLegalCTAs(int64_t value) {
  return value >= 1 && value <= 16 && llvm::isPowerOf2_64(value);
}

bool isCuTileProviderParameter(block::ParameterAttr parameter) {
  auto name = parameter.getName().getValue();
  return parameter.getRole() == block::ParameterRole::ProviderOption &&
      (name == accessFormParameter || name == occupancyParameter ||
       name == loadPolicyParameter || name == workerWarpsParameter || name == ctasParameter);
}

}
