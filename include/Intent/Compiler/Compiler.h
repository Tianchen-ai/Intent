#ifndef INTENT_COMPILER_COMPILER_H
#define INTENT_COMPILER_COMPILER_H

#include "Intent/Dialect/Intent/IR/CompileOptions.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/Support/JSON.h"
#include <optional>
#include <string>

namespace intent::compiler {

using Provider = std::string;
enum class InputStage { Kernel, Shared };
enum class Stage { Kernel, Shared, Provider };
enum class Failure {
  None = 0, Invocation = 1, KernelIR = 2, Construction = 3,
  SharedPipeline = 4, UnavailableProvider = 5, ProviderPipeline = 6,
  Translation = 7, Output = 8
};

struct CompilationOptions {
  NumericsMode numerics = NumericsMode::Source;
  bool onlineReduction = true;
  bool optimizationRemarks = false;
};

/// Compile-call inputs. Algorithms and execution decisions remain in the IR.
struct Request {
  std::optional<Provider> provider;
  InputStage inputStage = InputStage::Kernel;
  Stage stopAfter = Stage::Provider;
  /// Validated component-owned compile inputs. Construction transfers these
  /// facts into the executable IR; subsequent passes consume that IR only.
  mlir::Attribute target;
  mlir::DictionaryAttr bindings;
  std::string profileDirectory;
  std::string tuningConfig;
  std::string targetOptions;
  std::optional<CompilationOptions> options;
};

/// Owns the current module that produced source/metadata, including typed
/// host/device containers used by providers with separate device programs.
/// On failure it retains the current module for diagnostics, never a success artifact.
struct Result {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  Stage stage = Stage::Kernel;
  Failure failure = Failure::None;
  std::string source, metadata;
};

std::optional<Provider> parseProvider(llvm::StringRef name);
llvm::StringRef providerName(Provider provider);
llvm::StringRef stageName(Stage stage);
llvm::StringRef failureStage(Failure failure);
bool isProviderAvailable(Provider provider);
llvm::json::Object information(llvm::StringRef profileDirectory);
Result compile(mlir::OwningOpRef<mlir::ModuleOp> module, const Request &request);

} // namespace intent::compiler
#endif
