#ifndef INTENT_DIALECT_Block_IR_BlockTYPES_H
#define INTENT_DIALECT_Block_IR_BlockTYPES_H

#include "Intent/Dialect/Block/IR/BlockDialect.h"
#include "Intent/Dialect/Block/IR/BlockAttrs.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

#define GET_TYPEDEF_CLASSES
#include "Intent/Dialect/Block/IR/BlockTypes.h.inc"

namespace intent::block {

enum class BroadcastProjectionState { Exact, Unknown, Ambiguous };

struct BroadcastProjection {
  BroadcastProjectionState state = BroadcastProjectionState::Unknown;
  llvm::SmallVector<std::optional<unsigned>, 4> targetToSource;

  bool isExact() const { return state == BroadcastProjectionState::Exact; }
};

BroadcastProjection queryAxisProjection(FragmentType source,
                                        FragmentType target);
BroadcastProjection queryBroadcastProjection(FragmentType source,
                                              FragmentType target);
std::optional<llvm::SmallVector<int64_t>>
queryAxisPermutation(FragmentType source, FragmentType target);

} // namespace intent::block

#endif
