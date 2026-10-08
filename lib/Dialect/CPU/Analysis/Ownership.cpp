#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;

namespace intent::cpu {
namespace {

bool ownsHeapStorage(Value value) {
  Operation *owner = value.getDefiningOp();
  return owner && !isa<memref::AllocaOp>(owner) &&
         hasEffect<MemoryEffects::Allocate>(owner, value);
}

} // namespace

LogicalResult verifyStorageOwnership(func::FuncOp function) {
  StorageAnalysis storage(function);
  llvm::DenseSet<Value> releasedOrigins;
  bool invalid = false;
  auto release = [&](Operation *operation, Value memory, Value condition) {
    if (condition && matchPattern(condition, m_Zero())) return;
    auto origins = storage.origins(memory);
    if (!origins.complete) {
      operation->emitError("CPU release has an unresolved storage origin");
      invalid = true;
      return;
    }
    bool owned = false, borrowed = false;
    for (Value origin : origins.values)
      if (ownsHeapStorage(origin)) {
        releasedOrigins.insert(origin);
        owned = true;
      } else {
        borrowed = true;
      }
    if (!owned) {
      operation->emitError("CPU release cannot take ownership of borrowed or automatic storage");
      invalid = true;
    } else if (borrowed && condition && matchPattern(condition, m_One())) {
      operation->emitError("CPU unconditional release may select borrowed or automatic storage");
      invalid = true;
    }
  };

  function.walk([&](Operation *operation) {
    if (auto dealloc = dyn_cast<bufferization::DeallocOp>(operation)) {
      // The native operation owns its condition, alias-deduplication and
      // retained-value semantics. This audit checks that its release operands
      // connect to declared storage, not the correctness of the native RAII
      // algorithm or an equivalent reconstruction of its boolean dataflow.
      for (auto [memory, condition] :
           llvm::zip(dealloc.getMemrefs(), dealloc.getConditions()))
        release(operation, memory, condition);
    } else if (auto dealloc = dyn_cast<memref::DeallocOp>(operation)) {
      release(operation, dealloc.getMemref(), {});
      auto allocation = dealloc.getMemref().getDefiningOp<memref::AllocOp>();
      if (!allocation || allocation->getBlock() != dealloc->getBlock()) return;
      // An unconditional lexical free must precede no later access to that
      // exact allocation. Descriptor-only metadata uses do not access storage.
      auto aliases = storage.aliases(allocation);
      for (Operation *user : aliases.users) {
        Operation *local = dealloc->getBlock()->findAncestorOpInBlock(*user);
        if (!local || local == dealloc || !dealloc->isBeforeInBlock(local)) continue;
        for (const StorageEffect &entry : storage.effects(user).entries) {
          Value affected = entry.effect.getValue();
          if (!affected || storage.uniqueOrigin(affected) != allocation.getResult())
            continue;
          if (isa<MemoryEffects::Read, MemoryEffects::Write,
                  MemoryEffects::Free>(entry.effect.getEffect())) {
            user->emitError("CPU storage is accessed after its unconditional lexical release");
            invalid = true;
          }
        }
      }
    }
  });
  function.walk([&](Operation *operation) {
    for (Value value : operation->getResults()) {
      if (!isa<BaseMemRefType>(value.getType()) || !ownsHeapStorage(value)) continue;
      if (!releasedOrigins.contains(value)) {
        operation->emitError("CPU heap storage is disconnected from every explicit release");
        invalid = true;
      }
    }
  });
  return failure(invalid);
}

} // namespace intent::cpu
