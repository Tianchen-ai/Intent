#include "FragmentStorage.h"
#include "Accesses.h"
#include "Intent/Dialect/Block/Transforms/Storage/FragmentSnapshot.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/MapVector.h"

using namespace mlir;

namespace intent::cutile {

LogicalResult materializeFragmentStorage(func::FuncOp kernel) {
  llvm::MapVector<Value, SmallVector<block::GatherOp>> readers;
  kernel.walk([&](block::GatherOp gather) {
    if (requiresFragmentStorage(gather)) readers[gather.getSource()].push_back(gather);
  });
  IRMapping replacements;
  for (auto &[original, gathers] : readers) {
    Value source = replacements.lookupOrDefault(original);
    Value enabled = block::materializeFragmentSnapshotReadDomain(source, gathers);
    auto snapshot = block::materializeFragmentSnapshot(source, enabled);
    if (failed(snapshot)) return failure();
    for (block::GatherOp gather : gathers) {
      auto loaded = block::loadFragmentSnapshot(gather, *snapshot);
      if (failed(loaded)) return failure();
      replacements.map(gather.getResult(), *loaded);
      gather.getResult().replaceAllUsesWith(*loaded);
      gather.erase();
    }
  }
  return success();
}

} // namespace intent::cutile
