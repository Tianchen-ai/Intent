#ifndef INTENT_DIALECT_GPU_TRANSFORMS_PREDICATION_H
#define INTENT_DIALECT_GPU_TRANSFORMS_PREDICATION_H

#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"

namespace intent::gpu {

bool canPredicateScalarBlock(mlir::Block &block);
bool canPredicateScalarWhile(mlir::scf::WhileOp loop);
mlir::LogicalResult clonePredicatedScalarOperation(mlir::OpBuilder &builder,
                                   mlir::Operation *operation,
                                   mlir::IRMapping &mapping,
                                   mlir::Value predicate,
                                   FragmentType shape = {},
                                   bool nonemptyIterations = false);
mlir::LogicalResult guardInactivePredicatedLoops(mlir::ModuleOp module);

} // namespace intent::gpu

#endif
