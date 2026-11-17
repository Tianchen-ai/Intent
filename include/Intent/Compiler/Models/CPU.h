#ifndef INTENT_COMPILER_MODELS_CPU_H
#define INTENT_COMPILER_MODELS_CPU_H
#include "Intent/Compiler/Backend.h"
#include "Intent/Dialect/CPU/Transforms/Implementation/Implementation.h"
#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"

namespace intent::compiler {
struct CPUBackend final : ExecutionModel {
  using ProgramPipeline = void (*)(mlir::OpPassManager &, const cpu::CPUCompilationOptions &);
  CPUBackend(bool strided, llvm::StringRef filename, cpu::ImplementationRegistry implementations,
             cpu::CPUProgramStage stage, ProgramPipeline pipeline)
      : stridedInputs(strided), files{filename}, implementations(std::move(implementations)),
        inputStage(stage), pipeline(pipeline) {}
  llvm::StringRef name() const final { return "cpu"; }
  llvm::ArrayRef<llvm::StringRef> profileFiles() const final { return files; }
  const cpu::ImplementationRegistry *cpuImplementations() const final { return &implementations; }
  void buildConstruction(mlir::OpPassManager &, const Request &) const final;
  void buildShared(mlir::OpPassManager &, const Request &, llvm::StringRef) const final;
  mlir::LogicalResult verifySharedInput(mlir::ModuleOp, const Request &, llvm::StringRef) const final;
  bool stridedInputs;
  llvm::StringRef files[1];
  cpu::ImplementationRegistry implementations;
  cpu::CPUProgramStage inputStage;
  ProgramPipeline pipeline;
};
mlir::LogicalResult configureCPUTarget(mlir::ModuleOp, Request &);
} // namespace intent::compiler
#endif
