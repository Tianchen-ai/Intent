#ifndef INTENT_COMPILER_BACKEND_H
#define INTENT_COMPILER_BACKEND_H

#include "Intent/Compiler/Compiler.h"
#include "Intent/Dialect/Block/Transforms/Configuration/TuningProfiles.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"
#include <memory>

namespace intent::cpu { class ImplementationRegistry; }

namespace intent::compiler {

/// Components choose a lowering boundary, independently of device identity.
/// The driver never inspects the concrete model or its target options.
struct ExecutionModel {
  virtual ~ExecutionModel() = default;
  virtual llvm::StringRef name() const = 0;
  virtual llvm::ArrayRef<llvm::StringRef> profileFiles() const = 0;
  virtual void buildConstruction(mlir::OpPassManager &, const Request &) const = 0;
  virtual void buildShared(mlir::OpPassManager &, const Request &, llvm::StringRef provider) const = 0;
  virtual mlir::LogicalResult verifySharedInput(mlir::ModuleOp, const Request &, llvm::StringRef provider) const = 0;
  virtual std::optional<block::TuningProfileSchema> blockProfiles() const { return std::nullopt; }
  virtual const cpu::ImplementationRegistry *cpuImplementations() const { return nullptr; }
};

/// A compiled adapter declaration. Unavailable adapters are discoverable by
/// name but are rejected before their hooks or implementation factory are used.
/// Every provider pipeline transforms the current module, including any typed
/// host/device container needed by its terminal serializer.
struct Backend {
  Provider provider;
  llvm::StringRef name;
  bool available;
  std::shared_ptr<const ExecutionModel> model;
  void (*registerDialects)(mlir::DialectRegistry &);
  void (*registerPasses)();
  void (*buildPipeline)(mlir::OpPassManager &, const Request &);
  mlir::LogicalResult (*serialize)(mlir::ModuleOp, std::string &source, std::string &metadata);
  mlir::LogicalResult (*configure)(mlir::ModuleOp, Request &);

  llvm::SmallVector<std::string> profilePaths(llvm::StringRef directory) const;
  void buildConstruction(mlir::OpPassManager &, const Request &) const;
  void buildShared(mlir::OpPassManager &, const Request &) const;
  mlir::LogicalResult verifySharedInput(mlir::ModuleOp, const Request &) const;
};

llvm::ArrayRef<Backend> backends();
const Backend &backend(Provider provider);
llvm::SmallVector<block::TuningProfileSource> blockProfileSources(
    llvm::StringRef directory, llvm::StringRef provider);

} // namespace intent::compiler
#endif
