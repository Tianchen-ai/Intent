#ifndef INTENT_DSA_TRANSFORMS_LOCAL_TRANSFERS_H
#define INTENT_DSA_TRANSFORMS_LOCAL_TRANSFERS_H

#include "Intent/Dialect/DSA/Transforms/LocalSupplyRelations.h"

namespace intent::dsa::detail {

struct LocalTransfer {
  mlir::Operation *operation;
  mlir::Value source, destination, sourceOwner;
  bool identity;
};

// The complete destination version of one synchronous local transfer. A
// nonidentity transfer retains its explicit active rectangle and zero padding.
std::optional<LocalTransfer> queryLocalTransfer(mlir::Operation *operation,
    StorageAnalysis &storage, LocalSupplyRelations &relations);

struct LocalTransferRead {
  LoadTileOp reader;
  int64_t row, column;
  bool broadcastRows, broadcastColumns, keepRows, keepColumns;
};

std::optional<LocalTransferRead> composeTransferRead(
    const LocalTransfer &transfer, LoadTileOp reader,
    LocalSupplyRelations &relations);
void applyTransferRead(const LocalTransfer &transfer,
                       const LocalTransferRead &read);

} // namespace intent::dsa::detail
#endif
