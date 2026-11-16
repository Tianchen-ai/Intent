#ifndef INTENT_Block_ANALYSIS_PHYSICALPROGRAMDETAIL_H
#define INTENT_Block_ANALYSIS_PHYSICALPROGRAMDETAIL_H

#include "Intent/Analysis/ControlFlow.h"
#include "Intent/Dialect/Block/Analysis/PhysicalProgram.h"

namespace intent::block::detail {

bool isCoordinateReplayNode(mlir::Operation *operation);
bool isValueReplayNode(mlir::Operation *operation);
bool isAccessNode(mlir::Operation *operation);
void appendUnique(llvm::SmallVectorImpl<MakeRangeOp> &destination,
                  MakeRangeOp range);
void appendUnique(llvm::SmallVectorImpl<mlir::Operation *> &destination,
                  mlir::Operation *operation);
mlir::OpOperand *singleControlInput(mlir::Value target, ControlFlowEdgeKind kind,
                                    mlir::Region *sourceRegion);

} // namespace intent::block::detail

#endif
