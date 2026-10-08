#ifndef INTENT_DIALECT_CPU_TRANSFORMS_LOOPBUILDERS_H
#define INTENT_DIALECT_CPU_TRANSFORMS_LOOPBUILDERS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"

namespace intent::cpu {

inline mlir::Value index(mlir::OpBuilder &b, mlir::Location loc, int64_t value) {
  return b.create<mlir::arith::ConstantIndexOp>(loc, value);
}

inline mlir::Value add(mlir::OpBuilder &b, mlir::Location loc,
                       mlir::Value a, mlir::Value c) {
  return b.create<mlir::arith::AddIOp>(loc, a, c);
}

inline mlir::Value multiply(mlir::OpBuilder &b, mlir::Location loc,
                            mlir::Value a, mlir::Value c) {
  return b.create<mlir::arith::MulIOp>(loc, a, c);
}

inline mlir::scf::ForOp loop(mlir::OpBuilder &b, mlir::Location loc,
    mlir::Value begin, mlir::Value end, int64_t step,
    llvm::function_ref<void(mlir::Value)> body) {
  auto result = b.create<mlir::scf::ForOp>(loc, begin, end, index(b, loc, step));
  mlir::OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(result.getBody());
  body(result.getInductionVar());
  return result;
}

}

#endif
