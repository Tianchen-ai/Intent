#ifndef INTENT_TARGET_TRITON_TRANSFORMS_CONFIGURATIONFACTS_H
#define INTENT_TARGET_TRITON_TRANSFORMS_CONFIGURATIONFACTS_H

#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/SmallVector.h"

namespace intent::triton {

// One read of the current graph, before provider declarations or native forms
// mutate it. This is not retained by a pass or reused by later legalization.
struct ProgramConfigurationFacts {
  explicit ProgramConfigurationFacts(mlir::func::FuncOp kernel);

  llvm::SmallVector<gpu::ParameterCategory> categories;
  bool hasContraction = false;
  bool allContractionsF32 = true;
  bool straightLinePointwise = false;
  bool pipelineStagesAffectProgram = false;
};

} // namespace intent::triton
#endif
