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

bool isCuTileProviderRole(gpu::ParameterRole role) {
  return role == gpu::ParameterRole::ProviderAccessForm ||
         role == gpu::ParameterRole::ProviderOccupancy ||
         role == gpu::ParameterRole::ProviderLoadPolicy ||
         role == gpu::ParameterRole::ProviderWarps ||
         role == gpu::ParameterRole::ProviderCTAs;
}

}
