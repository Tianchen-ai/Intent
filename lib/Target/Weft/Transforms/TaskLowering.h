#ifndef INTENT_TARGET_WEFT_TRANSFORMS_TASKLOWERING_H
#define INTENT_TARGET_WEFT_TRANSFORMS_TASKLOWERING_H

#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include <memory>

namespace intent::cpu { class ImplementationRegistry; }

namespace intent::weft_provider {

class TaskConversion;

// One host candidate supplies the axis identities, shape symbols and storage
// interpretation for its task kernels. This state is consumed within conversion;
// the resulting program's calls and typed task bindings carry the final ABI.
class TaskLowering {
public:
  TaskLowering(mlir::func::FuncOp function, mlir::ModuleOp device,
      const llvm::DenseMap<mlir::Value, intent::QuantFormat> &formats,
      const cpu::ImplementationRegistry &implementations);
  ~TaskLowering();

  mlir::LogicalResult normalizeComputations();

  llvm::SmallVector<mlir::Value> shapeArguments(cpu::TasksOp task,
                                              mlir::OpBuilder &builder);
  mlir::LogicalResult lower(cpu::TasksOp task, llvm::StringRef name,
                            llvm::SmallVectorImpl<unsigned> &argumentPositions);

private:
  std::unique_ptr<TaskConversion> implementation;
};

} // namespace intent::weft_provider
#endif
