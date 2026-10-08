#ifndef INTENT_CONVERSION_KIRTOGPU_KIRTOGPU_H
#define INTENT_CONVERSION_KIRTOGPU_KIRTOGPU_H

#include "mlir/IR/BuiltinOps.h"

namespace intent {

struct GPUCapabilities {
  int64_t computeUnits;
  int64_t sharedMemoryPerUnit;
  int64_t maxDynamicSharedMemoryPerBlock;
  int64_t registersPerUnit;
  int64_t maxThreadsPerBlock;
  int64_t computeCapabilityMajor;
  int64_t computeCapabilityMinor;
  int64_t singleToDoublePrecisionPerfRatio;
  bool matrixUnits;
  bool dynamicVectorWidth;
  bool nativeTupleReductions;
  bool nativeTupleReductionRequiresConstantIdentity;
  bool nativeFragmentGather;
};

/// Consumes canonical KIR and replaces it with one complete conservative
/// provider-neutral executable GPU program.  The result never references KIR.
mlir::LogicalResult lowerCanonicalKIRToGPU(mlir::ModuleOp module,
                                           const GPUCapabilities &capabilities);

} // namespace intent

#endif
