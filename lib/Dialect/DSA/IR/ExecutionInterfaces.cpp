#include "Intent/Dialect/DSA/IR/ExecutionInterfaces.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "Intent/Dialect/DSA/IR/ExecutionInterfaces.cpp.inc"
using namespace mlir;
namespace intent::dsa {
std::optional<CompletionDomain> pendingCompletion(Operation *operation) {
  auto interface = dyn_cast<AsyncEffectOpInterface>(operation);
  return interface ? interface.getPendingCompletion() : std::nullopt;
}
bool completes(Operation *operation, const CompletionDomain &pending) {
  auto interface = dyn_cast<CompletionOpInterface>(operation);
  return interface && llvm::any_of(interface.getCompletedDomains(),
      [&](const CompletionDomain &domain) { return domain.covers(pending); });
}
FailureOr<Operation *> completionOfUse(Operation *use) {
  auto pending = pendingCompletion(use);
  if (!pending) return use;
  Operation *cursor = use;
  while (cursor) {
    for (Operation *next = cursor->getNextNode(); next; next = next->getNextNode())
      if (completes(next, *pending)) return next;
    auto branch = dyn_cast_or_null<scf::IfOp>(cursor->getParentOp());
    if (!branch) return failure();
    cursor = branch;
  }
  return failure();
}
}
