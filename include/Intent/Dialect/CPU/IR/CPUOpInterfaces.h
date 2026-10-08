#ifndef INTENT_DIALECT_CPU_IR_CPUOPINTERFACES_H
#define INTENT_DIALECT_CPU_IR_CPUOPINTERFACES_H

#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <optional>

namespace intent::cpu {

enum class RegionArgumentKind {
  Sources, Captures, LeftSummary, RightSummary, Summary, State, Destinations
};

struct RegionArgumentSchema {
  mlir::BlockArgument argument;
  mlir::Value prototype;
  RegionArgumentKind kind;
  std::optional<int64_t> sliceAxis;
};

namespace detail {
mlir::FailureOr<llvm::SmallVector<RegionArgumentSchema>>
regionArgumentSchema(mlir::Operation *operation, mlir::Region &region);
mlir::Block::BlockArgListType regionArguments(
    mlir::Operation *operation, mlir::Region &region, RegionArgumentKind kind);
}
}

#include "Intent/Dialect/CPU/IR/CPUOpInterfaces.h.inc"

#endif
