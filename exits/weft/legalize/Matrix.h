#pragma once
#include "intentdsl/levels/algorithm/kir/ir/StorageLayout.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cstdint>
namespace CI_UNIT_NAMESPACE {
struct RecordFields {
  bool grouped;
  int64_t vectorRegisters = 32;
  int64_t maxLmulEighths = 8;
  bool zvfbfmin = false;
  int64_t privateStackBytes = 65536;

  static int64_t subgroupWidth(const intentdsl::kir::QuantFormatSpec &format) {
    return format.subBlockElements > 0 ? format.subBlockElements
                                       : intentdsl::kir::Q8KRecord::sumGroup;
  }

  int64_t decodedRows(const intentdsl::kir::QuantFormatSpec &format,
                      int64_t vectorBits, int64_t rows) const {
    if (grouped)
      return 1;
    // The activation and its subblock sums are shared. Each overlapping
    // weight row uses the expanded planes, merged codes, and i32 dot sums.
    int64_t codes = llvm::divideCeil(format.blockElements * 8, vectorBits);
    int64_t subgroups = format.blockElements / subgroupWidth(format);
    int64_t sums = llvm::divideCeil(subgroups * 32, vectorBits);
    int64_t perRow = (format.numBitPlanes + 1) * codes + sums;
    int64_t available =
        std::max<int64_t>(1, (vectorRegisters - codes - sums) / perRow);
    return int64_t(1) << llvm::Log2_64(std::min(rows, available));
  }
};
} // namespace CI_UNIT_NAMESPACE
