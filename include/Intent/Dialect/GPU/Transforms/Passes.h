#ifndef INTENT_DIALECT_GPU_TRANSFORMS_PASSES_H
#define INTENT_DIALECT_GPU_TRANSFORMS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"

namespace intent::gpu {

#define GEN_PASS_DECL
#include "Intent/Dialect/GPU/Transforms/Passes.h.inc"

void registerGPUPasses();
void buildSharedGPUPipeline(mlir::OpPassManager &manager,
                            unsigned preferredResidentPrograms = 1);

mlir::LogicalResult verifyGPUProgram(mlir::ModuleOp module);
mlir::LogicalResult realizeAccessComposition(mlir::ModuleOp module);
mlir::LogicalResult simplifyMaskedAccessCoordinates(mlir::ModuleOp module);
mlir::LogicalResult predicateScalarControl(mlir::ModuleOp module);
mlir::LogicalResult realizeScanConsumerTraversals(mlir::ModuleOp module);
mlir::LogicalResult realizeReductionConsumerTraversals(
    mlir::ModuleOp module, unsigned preferredResidentPrograms);
mlir::LogicalResult materializeRetainedValues(mlir::ModuleOp module);
mlir::LogicalResult vectorizeBufferLoops(mlir::ModuleOp module);
mlir::LogicalResult promoteBufferValues(mlir::ModuleOp module);
mlir::LogicalResult schedulePrivateStores(mlir::ModuleOp module);
mlir::LogicalResult realizeContractionBlocking(mlir::ModuleOp module);
mlir::LogicalResult orientLoopContractions(mlir::ModuleOp module);
mlir::LogicalResult realizeVectorContractions(mlir::ModuleOp module);
mlir::LogicalResult normalizeContractionSources(mlir::ModuleOp module);
mlir::LogicalResult decomposeMultiAxisReductions(mlir::ModuleOp module);
mlir::LogicalResult realizeOnlineReductions(mlir::ModuleOp module);
mlir::LogicalResult realizeReductionBlocking(mlir::ModuleOp module);
mlir::LogicalResult realizePointwiseOwnership(mlir::ModuleOp module);
mlir::LogicalResult realizePointwiseBlocking(mlir::ModuleOp module);
mlir::LogicalResult refineProgramMapping(mlir::ModuleOp module);
// Value cleanup never changes the execution scope of a read. The optional
// placement group owns canonicalization, bounded invariant motion and CSE in
// that order. Shared/provider pass entries verify their respective surfaces.
mlir::LogicalResult eliminateCommonValues(mlir::ModuleOp module);
mlir::LogicalResult hoistLoopInvariantValues(mlir::ModuleOp module);
mlir::LogicalResult simplifyRangePredicates(mlir::ModuleOp module);
mlir::LogicalResult fuseIndependentReductions(mlir::ModuleOp module);
mlir::LogicalResult fuseIndependentTraversals(mlir::ModuleOp module,
                                             bool hoistLoopInvariants);
mlir::LogicalResult completeGPUProgramConstruction(mlir::ModuleOp module);

} // namespace intent::gpu

#endif
