#ifndef INTENT_TARGET_DSA_BANGC_MICROKERNELS_H
#define INTENT_TARGET_DSA_BANGC_MICROKERNELS_H
#include "Intent/Target/DSA/BangC/Physical/IR/DSAOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::bangc {
/// Realize one typed contraction using its active extents, initial accumulator,
/// explicit local supply and bounded configuration. Packing/scratch become IR
/// allocations and operands; MatrixTile/PrepareMatrix carry the native boundary.
mlir::LogicalResult realizeMatMul(physical::MatMulOp, physical::ConfigurationAttr);
/// An optional row/column collective implementation with explicit transpose
/// storage. Unknown or excessive resource requirements keep the original op.
void realizeRowSumChannels(mlir::func::FuncOp, physical::ConfigurationAttr);
}
#endif
