#ifndef INTENT_TARGET_CUTILE_IR_CONFIGURATION_H
#define INTENT_TARGET_CUTILE_IR_CONFIGURATION_H
#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "llvm/Support/MathExtras.h"
namespace intent::cutile {
inline constexpr llvm::StringLiteral accessFormParameter = "CUTILE_ACCESS_FORM";
inline constexpr llvm::StringLiteral occupancyParameter = "CUTILE_OCCUPANCY";
inline constexpr llvm::StringLiteral loadPolicyParameter = "CUTILE_LOAD_POLICY";
inline constexpr llvm::StringLiteral ctasParameter = "CUTILE_CTAS";
inline constexpr llvm::StringLiteral workerWarpsParameter = "CUTILE_WORKER_WARPS";
inline constexpr int64_t nativeAccessForm = 1;
inline constexpr int64_t gatherAccessForm = 2;
inline constexpr int64_t nativeNoTMAForm = 3;


bool isLegalAccessForm(int64_t value);
bool isLegalOccupancy(int64_t value);
bool isLegalWorkerWarps(int64_t value);
bool isLegalCTAs(int64_t value);
bool isCuTileProviderRole(gpu::ParameterRole role);
}
#endif
