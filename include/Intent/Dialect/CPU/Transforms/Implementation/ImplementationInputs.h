#ifndef INTENT_DIALECT_CPU_TRANSFORMS_IMPLEMENTATIONINPUTS_H
#define INTENT_DIALECT_CPU_TRANSFORMS_IMPLEMENTATIONINPUTS_H

#include "Intent/Dialect/CPU/Transforms/Implementation/Implementation.h"
#include <memory>

namespace mlir::memref {
class LoadOp;
}

namespace intent::cpu {

// Preparation at this point is outside concurrent work items and can own an
// addressable allocation captured by the compute tasks formed afterwards.
bool hasInvocationInputScope(mlir::Operation *operation);

// One context per function owns prepared input snapshots and their reuse.
class ImplementationInputs {
public:
  explicit ImplementationInputs(mlir::func::FuncOp function);
  ~ImplementationInputs();

  // Once the structured consumers have been expanded, compose complete
  // producer versions into the actual preparation reads and remove only the
  // versions whose observations have all been replaced.
  mlir::LogicalResult fuseProducerCopies();

  bool hasReusableScope(mlir::linalg::GenericOp operation,
                        llvm::ArrayRef<InputRequirement> requirements);

  mlir::FailureOr<llvm::SmallVector<InputSupply>> prepare(
      mlir::linalg::GenericOp operation,
      llvm::ArrayRef<InputRequirement> requirements);
  // Query actual prepared storage after prepare(): every complete allocation
  // must be outside concurrent work items and live at this future workset.
  bool preparedOutsideWorkset(mlir::Operation *point,
                             llvm::ArrayRef<InputSupply> supplies);
  mlir::FailureOr<InputSupply> prepareCaptured(
      mlir::linalg::GenericOp operation, mlir::memref::LoadOp input,
      const InputRequirement &requirement);
  // Allocate one private slot from the caller's checked source-axis capacities.
  // Its lexical owner must enclose the synchronous preparations and consumers.
  mlir::Value createPrivateStorage(mlir::OpBuilder &builder, mlir::Location loc,
      const InputRequirement &requirement, llvm::ArrayRef<int64_t> capacities);
  // Prepare this actual descriptor in that slot at its original read point.
  // The caller completes every prior consumer before reusing the slot; only the
  // actual window is read or initialized. Begins use source axes.
  mlir::FailureOr<InputSupply> prepareAt(
      mlir::Value window, mlir::ValueRange begins,
      const InputRequirement &requirement, llvm::ArrayRef<int64_t> capacities,
      mlir::Value storage, mlir::Operation *scope);
  mlir::FailureOr<llvm::SmallVector<InputSupply>> prepareGroup(
      mlir::OpBuilder &builder, mlir::linalg::GenericOp operation,
      const ContractionTile &tile, ConfigurationAttr configuration,
      llvm::ArrayRef<InputRequirement> requirements);

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

mlir::LogicalResult reusePreparedInputs(
    mlir::func::FuncOp function, const ImplementationRegistry &implementations);
mlir::LogicalResult groupQuantizedDots(
    mlir::func::FuncOp function, const ImplementationRegistry &implementations);

} // namespace intent::cpu
#endif
