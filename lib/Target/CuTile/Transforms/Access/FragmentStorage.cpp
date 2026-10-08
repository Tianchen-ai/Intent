#include "FragmentStorage.h"
#include "Accesses.h"
#include "Intent/Dialect/GPU/Transforms/Storage/FragmentSnapshot.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/MapVector.h"

using namespace mlir;

namespace intent::cutile {

LogicalResult materializeFragmentStorage(func::FuncOp kernel) {
  llvm::MapVector<Value, SmallVector<gpu::GatherOp>> readers;
  kernel.walk([&](gpu::GatherOp gather) {
    if (requiresFragmentStorage(gather)) readers[gather.getSource()].push_back(gather);
  });
  IRMapping replacements;
  for (auto &[original, gathers] : readers) {
    Value source = replacements.lookupOrDefault(original);
    Value enabled = gpu::materializeFragmentSnapshotReadDomain(source, gathers);
    auto snapshot = gpu::materializeFragmentSnapshot(source, enabled);
    if (failed(snapshot)) return failure();
    for (gpu::GatherOp gather : gathers) {
      auto loaded = gpu::loadFragmentSnapshot(gather, *snapshot);
      if (failed(loaded)) return failure();
      replacements.map(gather.getResult(), *loaded);
      gather.getResult().replaceAllUsesWith(*loaded);
      gather.erase();
    }
  }
  return success();
}

} // namespace intent::cutile
