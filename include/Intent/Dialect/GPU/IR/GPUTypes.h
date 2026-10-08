#ifndef INTENT_DIALECT_GPU_IR_GPUTYPES_H
#define INTENT_DIALECT_GPU_IR_GPUTYPES_H

#include "Intent/Dialect/GPU/IR/GPUDialect.h"
#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

#define GET_TYPEDEF_CLASSES
#include "Intent/Dialect/GPU/IR/GPUTypes.h.inc"

namespace intent::gpu {

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

} // namespace intent::gpu

#endif
