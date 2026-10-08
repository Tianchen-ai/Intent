#ifndef INTENT_TARGET_TRITON_IR_CONFIGURATION_H
#define INTENT_TARGET_TRITON_IR_CONFIGURATION_H

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/SmallVector.h"

namespace intent::triton {

// Native compiler/device limits, independent of profile selection and row
// correlation. The same predicate filters supplied values and verifies IR.
bool isLegalDeviceOption(gpu::ParameterRole role, int64_t value,
                         gpu::CapabilitiesAttr capabilities,
                         bool requiresSingleCTA);

// Names from the current program, with no candidate values or tuning policy.
// A stage parameter can serve both a native option and a kernel constexpr.
// Coverage parameters are kernel constexprs but remain deferred in config rows.
struct ConfigurationSchema {
  mlir::StringAttr warps;
  mlir::StringAttr stages;
  mlir::StringAttr ctas;
  llvm::SmallVector<mlir::StringAttr> kernelParameters;

  static mlir::FailureOr<ConfigurationSchema> read(mlir::func::FuncOp kernel);
};

} // namespace intent::triton

#endif
