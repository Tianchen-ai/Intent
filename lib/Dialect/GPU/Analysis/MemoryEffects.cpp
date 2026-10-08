#include "Intent/Dialect/GPU/Analysis/MemoryEffects.h"
#include "Intent/Dialect/GPU/Analysis/ResourceAlias.h"
#include "Intent/Dialect/GPU/IR/AccessOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::gpu {
namespace {

bool hasOrderedAccess(Operation *operation) {
  return operation
      ->walk([](Operation *nested) {
        auto access = dyn_cast<AccessOpInterface>(nested);
        if (!access)
          return WalkResult::advance();
        switch (access.getAccessKind()) {
        case AccessKind::AtomicLoad:
        case AccessKind::AtomicStore:
        case AccessKind::AtomicRMW:
        case AccessKind::AtomicCompareExchange:
          return WalkResult::interrupt();
        default:
          return WalkResult::advance();
        }
      })
      .wasInterrupted();
}

} // namespace

bool hasOnlyReadEffects(Operation *operation) {
  if (hasOrderedAccess(operation))
    return false;
  auto effects = getEffectsRecursively(operation);
  return effects && llvm::all_of(*effects, [](const auto &effect) {
           return isa<MemoryEffects::Read>(effect.getEffect());
         });
}

bool preservesMemoryReads(Operation *reads, Operation *intervening,
                          ResourceAliasAnalysis &aliases,
                          llvm::function_ref<bool(Value, Value)> disjointInContext) {
  if (hasOrderedAccess(reads) || hasOrderedAccess(intervening))
    return false;
  auto readEffects = getEffectsRecursively(reads);
  auto effects = getEffectsRecursively(intervening);
  if (!readEffects || !effects ||
      !llvm::all_of(*readEffects, [](const auto &effect) {
        return isa<MemoryEffects::Read>(effect.getEffect());
      }))
    return false;
  for (const auto &effect : *effects) {
    if (isa<MemoryEffects::Read, MemoryEffects::Allocate>(effect.getEffect()))
      continue;
    if (!isa<MemoryEffects::Write, MemoryEffects::Free>(effect.getEffect()))
      return false;
    for (const auto &read : *readEffects) {
      if (read.getResource() != effect.getResource())
        continue;
      if (!read.getValue() || !effect.getValue())
        return false;
      if (!aliases.alias(read.getValue(), effect.getValue()).isNo() &&
          !(disjointInContext &&
            disjointInContext(read.getValue(), effect.getValue())))
        return false;
    }
  }
  return true;
}

} // namespace intent::gpu
