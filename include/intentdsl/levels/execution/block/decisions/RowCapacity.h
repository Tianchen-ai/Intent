#pragma once
#include "llvm/ADT/SmallVector.h"
#include <string>

namespace intentdsl {
struct RowCapacity {
  uint64_t upperBound;
  struct Row {
    unsigned id, rows, width;
    std::string payload;
  };
  llvm::SmallVector<Row> rows;
};
} // namespace intentdsl
