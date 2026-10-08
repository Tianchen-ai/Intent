#ifndef INTENT_DIALECT_CPU_TRANSFORMS_PASSSUPPORT_H
#define INTENT_DIALECT_CPU_TRANSFORMS_PASSSUPPORT_H

#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/CPU/IR/CPUDialect.h"
#include "Intent/Dialect/CPU/Transforms/Implementation/Implementation.h"
#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

namespace intent::cpu::detail {

inline mlir::LogicalResult verifyPassInput(
    mlir::ModuleOp module, llvm::StringRef name, CPUProgramStage stage) {
  if (mlir::failed(verifyCPUProgram(module, stage)))
    return module.emitError() << "CPU precondition failed: " << name;
  return mlir::success();
}

inline mlir::LogicalResult finishTransform(
    mlir::ModuleOp module, llvm::StringRef name, mlir::LogicalResult result,
    CPUProgramStage stage = CPUProgramStage::Buffers) {
  if (mlir::failed(result))
    return module.emitError() << "CPU transformation failed: " << name;
  if (mlir::failed(verifyCPUProgram(module, stage)))
    return module.emitError() << "CPU postcondition failed: " << name;
  return mlir::success();
}

// These adapters consume materialized buffers. They neither construct a
// configuration nor retain analyses across changes to the current program.
inline mlir::LogicalResult transformFunctions(
    mlir::ModuleOp module, llvm::StringRef name,
    llvm::function_ref<mlir::LogicalResult(mlir::func::FuncOp)> transform) {
  if (mlir::failed(verifyPassInput(module, name, CPUProgramStage::Buffers)))
    return mlir::failure();
  for (auto function : module.getOps<mlir::func::FuncOp>()) {
    if (function.isExternal())
      continue;
    if (mlir::failed(transform(function)))
      return finishTransform(module, name, mlir::failure());
  }
  return finishTransform(module, name, mlir::success());
}

inline mlir::LogicalResult transformWithImplementations(
    mlir::ModuleOp module, llvm::StringRef name, llvm::StringRef provider,
    llvm::function_ref<mlir::LogicalResult(
        mlir::func::FuncOp, const ImplementationRegistry &)> transform) {
  auto implementations = lookupImplementationProvider(module, provider);
  if (mlir::failed(implementations) ||
      mlir::failed((**implementations).verifyBindings(module)))
    return mlir::failure();
  return transformFunctions(module, name, [&](mlir::func::FuncOp function) {
    return transform(function, **implementations);
  });
}

inline mlir::FailureOr<Configuration>
currentConfiguration(mlir::func::FuncOp function) {
  auto binding = function->getAttrOfType<ConfigurationAttr>(
      "intent_cpu.configuration");
  if (!binding)
    return function.emitError(
               "CPU physical transformation requires a bound configuration"),
           mlir::failure();
  return Configuration{binding.getTaskGrain(), binding.getTileM(),
                       binding.getTileN(), binding.getTileK(),
                       binding.getRegionSize(), {}};
}

} // namespace intent::cpu::detail

#endif
