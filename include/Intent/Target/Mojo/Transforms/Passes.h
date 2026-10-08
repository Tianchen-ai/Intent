#ifndef INTENT_TARGET_MOJO_TRANSFORMS_PASSES_H
#define INTENT_TARGET_MOJO_TRANSFORMS_PASSES_H
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "Intent/Dialect/CPU/Transforms/Implementation/Implementation.h"
namespace intent::mojo {
#define GEN_PASS_DECL
#include "Intent/Target/Mojo/Transforms/Passes.h.inc"

void registerMojoPasses();
void buildMojoPipeline(mlir::OpPassManager &manager);
cpu::ImplementationRegistry implementations();
int64_t registerContractionRows(mlir::linalg::GenericOp operation);
mlir::LogicalResult materializeRegisterContractions(mlir::func::FuncOp function);
}
#endif
