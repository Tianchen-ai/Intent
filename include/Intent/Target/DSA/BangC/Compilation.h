#ifndef INTENT_TARGET_BANGC_COMPILATION_H
#define INTENT_TARGET_BANGC_COMPILATION_H
#include "Intent/Compiler/Backend.h"

namespace intent::compiler {
mlir::FailureOr<mlir::DictionaryAttr> parseDSAParameterBindings(
    mlir::ModuleOp, llvm::StringRef text, bool shapes);
mlir::FailureOr<mlir::ArrayAttr> readDSAConfigurations(
    mlir::ModuleOp, llvm::StringRef directory, llvm::StringRef overrides);
mlir::FailureOr<mlir::Attribute> readDSACapabilities(mlir::ModuleOp, llvm::StringRef text);
struct BangCBackend final : ExecutionModel {
  llvm::StringRef name() const final { return "dsa"; }
  llvm::ArrayRef<llvm::StringRef> profileFiles() const final { return files; }
  void buildConstruction(mlir::OpPassManager &, const Request &) const final;
  void buildShared(mlir::OpPassManager &, const Request &, llvm::StringRef) const final;
  mlir::LogicalResult verifySharedInput(mlir::ModuleOp, const Request &, llvm::StringRef) const final;
  llvm::StringRef files[1] = {"bangc.json"};
};
} // namespace intent::compiler
#endif
