#ifndef INTENT_TARGET_CUTILE_ANALYSIS_PROGRAM_H
#define INTENT_TARGET_CUTILE_ANALYSIS_PROGRAM_H
#include "Intent/Dialect/Block/IR/BlockAttrs.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLFunctionalExtras.h"
namespace intent::cutile {
struct NativeProgramFeatures {
  bool matrixCompute = false;
  bool occupancySensitive = false;
  void observe(mlir::Operation *operation);
};
NativeProgramFeatures queryNativeProgramFeatures(mlir::func::FuncOp kernel);
bool hasLoopVariantIndexedMatrixSupply(mlir::func::FuncOp kernel);
bool supportsE8M0ScaledMMA(block::CapabilitiesAttr capabilities);
// Validate current executable facts with the terminal operation registry. The
// callback selects no forms and this analysis does not depend on serialization.
mlir::LogicalResult verifyCuTileProgram(mlir::ModuleOp module,
    llvm::function_ref<mlir::LogicalResult(mlir::Operation *)> verifySourceOperation);
}
#endif
