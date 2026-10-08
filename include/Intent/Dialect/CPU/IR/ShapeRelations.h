#ifndef INTENT_DIALECT_CPU_IR_SHAPERELATIONS_H
#define INTENT_DIALECT_CPU_IR_SHAPERELATIONS_H

#include "mlir/IR/Value.h"
#include <optional>

namespace intent::cpu {

struct PublicDimension {
  mlir::BlockArgument argument;
  int64_t axis;
  std::optional<int64_t> constant;
};

// Public ABI dimension identity is an extent equality, not an alias or an
// equality between the elements at the corresponding coordinates.
std::optional<PublicDimension>
queryPublicDimension(mlir::BlockArgument argument, int64_t axis);

} // namespace intent::cpu
#endif
