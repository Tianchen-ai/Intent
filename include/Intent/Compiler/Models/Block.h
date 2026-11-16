#ifndef INTENT_COMPILER_MODELS_BLOCK_H
#define INTENT_COMPILER_MODELS_BLOCK_H
#include "Intent/Compiler/Backend.h"

namespace intent::compiler {
struct BlockBackend final : ExecutionModel {
  BlockBackend(unsigned residency, block::TuningProfileSchema profiles,
               llvm::StringRef filename)
      : preferredReductionResidentPrograms(residency), profiles(profiles), files{filename} {}
  llvm::StringRef name() const final { return "block"; }
  llvm::ArrayRef<llvm::StringRef> profileFiles() const final { return files; }
  std::optional<block::TuningProfileSchema> blockProfiles() const final { return profiles; }
  void buildConstruction(mlir::OpPassManager &, const Request &) const final;
  void buildShared(mlir::OpPassManager &, const Request &, llvm::StringRef) const final;
  mlir::LogicalResult verifySharedInput(mlir::ModuleOp, const Request &, llvm::StringRef) const final;
  unsigned preferredReductionResidentPrograms;
  block::TuningProfileSchema profiles;
  llvm::StringRef files[1];
};
} // namespace intent::compiler
#endif
