#pragma once
namespace intentdsl::kir {
// Byte offsets in little-endian records. Field width names describe encoded
// storage, not the type used when accumulating a dot.
struct Q4KRecord {
  static constexpr int elements = 256, bytes = 144, group = 32;
  static constexpr int d = 0, dmin = 2, factors = 4, packed = 16;
  static constexpr int scaleBits = 6, nibbleBits = 4, halfGroups = 4;
};
struct Q8KRecord {
  static constexpr int elements = 256, bytes = 292, d = 0, values = 4,
                       sums = 260;
  static constexpr int sumGroup = 16, scaleBits = 32, elementBits = 8;
};
struct MX8Block {
  static constexpr int group = 32, bias = 127, nan = 255;
  // A[M,G,k], B[G,k,N]; both scale views are logically [output,G].
  static constexpr int lhsGroupAxis = 1, rhsGroupAxis = 0, scaleGroupAxis = 1;
};
} // namespace intentdsl::kir
