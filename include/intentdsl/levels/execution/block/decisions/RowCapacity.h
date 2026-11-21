#pragma once
#include "llvm/ADT/SmallVector.h"
#include <string>

namespace intentdsl {
struct RowCapacity {
  int64_t observedWidth;
  struct Row {
    unsigned id, rows, width;
    std::string payload;
  };
  llvm::SmallVector<Row> rows;
};
} // namespace intentdsl
