#ifndef INTENT_DIALECT_CPU_TRANSFORMS_IMPLEMENTATION_H
#define INTENT_DIALECT_CPU_TRANSFORMS_IMPLEMENTATION_H

#include "Intent/Dialect/CPU/IR/CPUAttrs.h"
#include "Intent/Dialect/CPU/Transforms/Configuration/Configuration.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <array>
#include <functional>
#include <optional>
#include <string>
#include <variant>

namespace intent::cpu {

enum class InputReuse { Group, Consumers };
enum class InputStorageScope { Consumer, Invocation };

// Storage order is [panel, unsplit source axes..., lane within panel]. A group
// supply covers one compute group; consumer reuse preserves a source snapshot.
struct InputRequirement {
  unsigned operand;
  mlir::Type elementType;
  unsigned panelAxis;
  int64_t panelSize;
  int64_t alignment;
  InputReuse reuse;
  int64_t windowAlignment; // Required panel-axis origin multiple for non-singleton windows.
  // Invocation storage remains addressable across task invocations. Consumer
  // storage may instead become a provider's task-local value representation.
  InputStorageScope storageScope = InputStorageScope::Consumer;
};

std::optional<std::string> checkInputRequirement(
    mlir::Value source, mlir::Value element, const InputRequirement &requirement);
std::optional<std::string> checkInputRequirements(
    mlir::linalg::GenericOp operation, llvm::ArrayRef<InputRequirement> requirements);

struct InputSupply {
  unsigned operand;
  unsigned panelAxis;
  int64_t panelSize;
  mlir::Value storage;
  llvm::SmallVector<mlir::Value> begins; // Logical source coordinates of the supplied window.
};

struct ContractionTile {
  // Output is the actual mCount-by-nCount tile, indexed from zero. Input
  // windows and supplied representations retain the logical m/n/k origins.
  mlir::Value lhs, rhs, output, initial;
  mlir::Value mBegin, mCount, nBegin, nCount, kBegin, depth;
  bool first;
  llvm::ArrayRef<InputSupply> inputs;
};

struct ContractionRequirements {
  bool completePrivateInitialization = false;
  bool staticReductionExtent = false;
  bool staticParallelExtent = false;
  std::array<bool, 2> unitInnerStride{false, false};

  bool acceptsInputLayout(unsigned operand, mlir::MemRefType type) const;
};

struct ImplementationParameterDomain {
  bool powerOfTwo = false;
  std::optional<int64_t> maximum;
  int64_t laneBits = 0;
  llvm::SmallVector<int64_t, 2> values;
};

struct ImplementationParameter {
  struct Local { llvm::StringRef name; };
  struct Constant { int64_t value; };
  enum class Axis { TileM, TileN };
  struct Minimum { int64_t limit; llvm::SmallVector<Axis, 2> axes; };

  llvm::StringRef name;
  std::variant<Local, Constant, Minimum> source;
  ImplementationParameterDomain domain;

  static ImplementationParameter local(llvm::StringRef name, ImplementationParameterDomain domain = {});
  static ImplementationParameter alias(llvm::StringRef name, llvm::StringRef source);
  static ImplementationParameter constant(llvm::StringRef name, int64_t value);
  static ImplementationParameter minimum(llvm::StringRef name, int64_t limit, llvm::ArrayRef<Axis> axes);
};

// Services supplied by the provider's current native conversion. A borrowed
// view preserves the CPU storage's declared axes; read/write/bind distinguish
// value materialization, destination updates and source SSA replacement.
class ImplementationExpansion {
public:
  virtual ~ImplementationExpansion() = default;
  virtual mlir::OpBuilder &builder() = 0;
  virtual int64_t &nextAxis() = 0;
  virtual mlir::FailureOr<mlir::Value> view(mlir::Value source) = 0;
  virtual mlir::FailureOr<mlir::Value> read(mlir::Value source) = 0;
  virtual mlir::LogicalResult write(mlir::Value destination,
                                    mlir::Value value) = 0;
  virtual mlir::LogicalResult bind(mlir::Value sourceResult,
                                   mlir::Value nativeValue) = 0;
};

struct WorksetRows {
  // Task grain counts these original work items. Replicas only permit adjacent
  // items to share a computation inside the interval assigned to one worker.
  int64_t unit = 1;
  int64_t replicas = 1;
};

struct Implementation {
  llvm::StringRef name;
  // Optional per-instance refinement of the registered operation family.
  std::function<bool(mlir::Operation *)> applicable;
  std::function<std::optional<std::string>(mlir::Operation *, CapabilitiesAttr,
                                         const Configuration &)> check;
  llvm::SmallVector<ImplementationParameter, 5> parameters;
  std::function<mlir::LogicalResult(mlir::OpBuilder &, mlir::linalg::GenericOp,
      const ContractionTile &, ConfigurationAttr, ImplementationAttr)> formTile;
  std::function<mlir::LogicalResult(mlir::Operation *, ImplementationExpansion &)> expand;
  ContractionRequirements contraction;
  std::function<int64_t(ImplementationAttr)> parallelWindow;
  bool requiresMatrixI8I32 = false;
  std::function<llvm::SmallVector<InputRequirement>(mlir::linalg::GenericOp,
      ConfigurationAttr, ImplementationAttr)> inputs;
  std::function<WorksetRows(mlir::linalg::GenericOp, ImplementationAttr)> worksetRows;
  std::function<std::optional<std::string>(mlir::DictionaryAttr, CapabilitiesAttr)> parameterRelations;
  std::function<mlir::LogicalResult(mlir::Operation *, ImplementationAttr)> materialize;
  std::function<mlir::LogicalResult(mlir::scf::ForOp, ImplementationAttr)> vectorize;
  // The registered operation family is independent of per-instance legality.
  // A recognized computation with no legal candidate must still be diagnosed.
  std::function<bool(mlir::Operation *)> operationKind;

  bool matches(mlir::Operation *operation) const {
    return operationKind(operation) && (!applicable || applicable(operation));
  }

  // A read-only query of the selected implementation's surrounding supply.
  // An empty result requires no preparation; it is not a performance estimate.
  llvm::SmallVector<InputRequirement> inputRequirements(
      mlir::linalg::GenericOp operation, ConfigurationAttr configuration,
      ImplementationAttr binding) const;
};

class ImplementationRegistry {
public:
  std::function<llvm::StringRef(mlir::func::FuncOp)> profile;
  void addProfile(llvm::StringRef name,
                  llvm::ArrayRef<llvm::StringRef> localParameters);
  std::optional<llvm::ArrayRef<llvm::StringRef>>
  profileParameters(llvm::StringRef name) const;
  bool hasImplementation(llvm::StringRef name) const;
  /// Local configuration inputs declared by the explicitly selected members,
  /// including members whose typed computation is absent from this program.
  mlir::FailureOr<llvm::SmallVector<llvm::StringRef>>
  localParameters(mlir::ArrayAttr selection) const;
  template <typename... Operations>
  void add(Implementation implementation) {
    static_assert(sizeof...(Operations) > 0, "declare the implementation's operation family");
    implementation.operationKind = [](mlir::Operation *operation) {
      return mlir::isa<Operations...>(operation);
    };
    implementations.push_back(std::move(implementation));
  }
  bool needsImplementation(mlir::Operation *operation) const;
  mlir::FailureOr<const Implementation *> lookup(mlir::Operation *operation) const;
  mlir::LogicalResult materialize(mlir::Operation *operation) const;
  // Validate retained bindings without rerunning pre-expansion applicability.
  mlir::FailureOr<const Implementation *> verifyBinding(
      ImplementationAttr binding, CapabilitiesAttr capabilities,
      ConfigurationAttr configuration, mlir::Operation *diagnostic) const;
  mlir::LogicalResult verifyBindings(mlir::ModuleOp module) const;
  mlir::FailureOr<llvm::SmallVector<ImplementationAttr>> select(
      mlir::func::FuncOp function, CapabilitiesAttr capabilities,
      const Configuration &configuration,
      llvm::function_ref<void(mlir::Operation *, llvm::StringRef, llvm::StringRef)> rejected) const;
  mlir::LogicalResult bind(mlir::func::FuncOp function, CapabilitiesAttr capabilities,
                           const Configuration &configuration,
                           llvm::ArrayRef<ImplementationAttr> bindings) const;

private:
  struct ProfileSchema {
    llvm::StringRef name;
    llvm::SmallVector<llvm::StringRef> localParameters;
  };
  llvm::SmallVector<ProfileSchema> profiles;
  llvm::SmallVector<Implementation, 0> implementations;
};

/// Resolve a provider from this compiler context. Missing registration and an
/// absent/unknown provider are errors, never a request for a default registry.
mlir::FailureOr<const ImplementationRegistry *>
lookupImplementationProvider(mlir::Operation *operation, llvm::StringRef provider);
mlir::LogicalResult verifyImplementationBindings(mlir::ModuleOp module, llvm::StringRef provider);

int64_t implementationParameter(ImplementationAttr binding, llvm::StringRef name);

}
#endif
