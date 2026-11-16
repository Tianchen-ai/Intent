#ifndef INTENT_DSA_EXECUTION_INTERFACES_H
#define INTENT_DSA_EXECUTION_INTERFACES_H
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <optional>

namespace intent::dsa {
enum class CompletionScope { Participant, Group, Invocation };
struct CompletionDomain {
  mlir::Attribute engine;
  CompletionScope scope;
  bool covers(const CompletionDomain &pending) const {
    return engine == pending.engine && scope >= pending.scope;
  }
};
/// Ordering state is separate from buffer contents. Async issue and completion
/// both read/write this resource so ordinary MLIR transforms retain the edges.
struct ExecutionOrderResource : mlir::SideEffects::Resource::Base<ExecutionOrderResource> {
  llvm::StringRef getName() final { return "IntentDSAExecutionOrder"; }
};
std::optional<CompletionDomain> pendingCompletion(mlir::Operation *);
bool completes(mlir::Operation *, const CompletionDomain &);
/// Locate a completion on every path issuing this use, without crossing a loop
/// backedge or assuming unrelated participants share a wait.
mlir::FailureOr<mlir::Operation *> completionOfUse(mlir::Operation *);
}
#include "Intent/Dialect/DSA/IR/ExecutionInterfaces.h.inc"
#endif
