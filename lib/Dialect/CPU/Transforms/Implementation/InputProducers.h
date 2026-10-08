#ifndef INTENT_CPU_TRANSFORMS_INPUTPRODUCERS_H
#define INTENT_CPU_TRANSFORMS_INPUTPRODUCERS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/SmallVector.h"

namespace intent::cpu {

// Actual reads emitted by input preparation. Kept only until the containing
// structured consumers have been replaced, so version observations can decide
// whether the original materialization really disappears.
class InputProducerCopies {
public:
  unsigned begin(mlir::Block *block, mlir::Value source);
  void record(unsigned preparation, mlir::Operation *read);
  mlir::LogicalResult fold(mlir::func::FuncOp function);

private:
  struct Preparation {
    mlir::Block *block;
    mlir::Value source;
    llvm::SmallVector<mlir::Operation *> reads;
  };
  llvm::SmallVector<Preparation> preparations;
};

} // namespace intent::cpu
#endif
