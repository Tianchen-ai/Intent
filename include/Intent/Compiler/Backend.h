#ifndef INTENT_COMPILER_BACKEND_H
#define INTENT_COMPILER_BACKEND_H

#include "Intent/Compiler/Compiler.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/TuningProfiles.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"
#include <variant>

namespace intent::cpu { class ImplementationRegistry; }

namespace intent::compiler {

enum class Family { GPU, CPU, DSA };

struct GPUBackend {
  static constexpr Family family = Family::GPU;
  bool nativeTupleReductions;
  bool nativeTupleReductionRequiresConstantIdentity;
  bool nativeFragmentGather;
  // Native-adapter profitability preference, not a register-allocation promise.
  unsigned preferredReductionResidentPrograms;
  gpu::TuningProfileSchema profiles;
  llvm::StringRef profileFilename;
  void buildConstruction(mlir::OpPassManager &, const Request &) const;
  void buildShared(mlir::OpPassManager &, const Request &, llvm::StringRef provider) const;
  mlir::LogicalResult verifySharedInput(mlir::ModuleOp, const Request &, llvm::StringRef provider) const;
};

struct CPUBackend {
  static constexpr Family family = Family::CPU;
  bool stridedInputs;
  llvm::StringRef profileFilename;
  cpu::ImplementationRegistry (*implementations)();
  void buildConstruction(mlir::OpPassManager &, const Request &) const;
  void buildShared(mlir::OpPassManager &, const Request &, llvm::StringRef provider) const;
  mlir::LogicalResult verifySharedInput(mlir::ModuleOp, const Request &, llvm::StringRef provider) const;
};

struct DSABackend {
  static constexpr Family family = Family::DSA;
  void buildConstruction(mlir::OpPassManager &, const Request &) const;
  void buildShared(mlir::OpPassManager &, const Request &, llvm::StringRef provider) const;
  mlir::LogicalResult verifySharedInput(mlir::ModuleOp, const Request &, llvm::StringRef provider) const;
};

/// A compiled adapter declaration. Unavailable adapters are discoverable by
/// name but are rejected before their hooks or implementation factory are used.
/// Every provider pipeline transforms the current module, including any typed
/// host/device container needed by its terminal serializer.
struct Backend {
  Provider provider;
  llvm::StringRef name;
  bool available;
  std::variant<GPUBackend, CPUBackend, DSABackend> model;
  void (*registerDialects)(mlir::DialectRegistry &);
  void (*registerPasses)();
  void (*buildPipeline)(mlir::OpPassManager &, const Request &);
  mlir::LogicalResult (*serialize)(mlir::ModuleOp, std::string &source, std::string &metadata);

  Family family() const;
  llvm::SmallVector<std::string> profilePaths(llvm::StringRef directory) const;
  void buildConstruction(mlir::OpPassManager &, const Request &) const;
  void buildShared(mlir::OpPassManager &, const Request &) const;
  mlir::LogicalResult verifySharedInput(mlir::ModuleOp, const Request &) const;
};

llvm::ArrayRef<Backend> backends();
const Backend &backend(Provider provider);
llvm::SmallVector<gpu::TuningProfileSource> gpuProfileSources(
    llvm::StringRef directory, llvm::StringRef provider);
mlir::FailureOr<mlir::DictionaryAttr> parseDSAParameterBindings(
    mlir::ModuleOp module, llvm::StringRef text, bool shapes);
mlir::FailureOr<mlir::ArrayAttr> readDSAConfigurations(
    mlir::ModuleOp module, llvm::StringRef directory, llvm::StringRef overrides);

} // namespace intent::compiler
#endif
