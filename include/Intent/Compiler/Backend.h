#ifndef INTENT_COMPILER_BACKEND_H
#define INTENT_COMPILER_BACKEND_H

#include "Intent/Compiler/Compiler.h"
#include "Intent/Dialect/Block/Transforms/Configuration/TuningProfiles.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"
#include <variant>

namespace intent::cpu { class ImplementationRegistry; }

namespace intent::compiler {

enum class Family { Block, CPU, DSA };

struct BlockBackend {
  static constexpr Family family = Family::Block;
  bool nativeTupleReductions;
  bool nativeTupleReductionRequiresConstantIdentity;
  bool nativeFragmentGather;
  // Native-adapter profitability preference, not a register-allocation promise.
  unsigned preferredReductionResidentPrograms;
  block::TuningProfileSchema profiles;
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
  llvm::StringRef profileFilename;
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
  std::variant<BlockBackend, CPUBackend, DSABackend> model;
  void (*registerDialects)(mlir::DialectRegistry &);
  void (*registerPasses)();
  void (*buildPipeline)(mlir::OpPassManager &, const Request &);
  mlir::LogicalResult (*serialize)(mlir::ModuleOp, std::string &source, std::string &metadata);
  mlir::LogicalResult (*configure)(mlir::ModuleOp, Request &);

  Family family() const;
  llvm::SmallVector<std::string> profilePaths(llvm::StringRef directory) const;
  void buildConstruction(mlir::OpPassManager &, const Request &) const;
  void buildShared(mlir::OpPassManager &, const Request &) const;
  mlir::LogicalResult verifySharedInput(mlir::ModuleOp, const Request &) const;
};

llvm::ArrayRef<Backend> backends();
const Backend &backend(Provider provider);
llvm::SmallVector<block::TuningProfileSource> blockProfileSources(
    llvm::StringRef directory, llvm::StringRef provider);
mlir::FailureOr<mlir::DictionaryAttr> parseDSAParameterBindings(
    mlir::ModuleOp module, llvm::StringRef text, bool shapes);
mlir::FailureOr<mlir::ArrayAttr> readDSAConfigurations(
    mlir::ModuleOp module, llvm::StringRef directory, llvm::StringRef overrides);
mlir::FailureOr<mlir::Attribute> readDSACapabilities(mlir::ModuleOp module, llvm::StringRef text);
mlir::LogicalResult configureNVIDIATarget(mlir::ModuleOp module, Request &request);
mlir::LogicalResult configureCPUTarget(mlir::ModuleOp module, Request &request);
mlir::LogicalResult configureBangCTarget(mlir::ModuleOp module, Request &request);

} // namespace intent::compiler
#endif
