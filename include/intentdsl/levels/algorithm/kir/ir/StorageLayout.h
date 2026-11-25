#pragma once
#include "llvm/ADT/StringRef.h"
#include <cstdint>

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

// Declarative quantized format descriptor ("format as data"):
// New formats add one pure-data entry in kQuantFormats without exit changes.
enum class QuantRoleLayout : uint8_t {
  // Super-block record view [N, G, recordBytes] x activation [G, blockElements]
  // (or Q8_K encoded activation), producing [N].
  RecordGemv = 0,
  // Symmetric block-scaled pair:
  //   lhs [M, G, recordBytes], lhsScale [M, G],
  //   rhs [G, recordBytes, N], rhsScale [N, G] (or transposed [G, N]),
  // producing [M, N].
  BlockScaledPair = 1,
  // Grouped weight-only RHS along K:
  //   lhs [M, K], qweight [K, N / packFactor], scales [K / blockElements, N],
  //   optional qzeros [K / blockElements, N / packFactor], producing [M, N].
  GroupedWeightRhs = 2,
};

enum class QuantValueCode : uint8_t {
  UnsignedAffine = 0, // val = f32(code - zeroBias - zeroPoint)
  Fp4E2M1 = 1,        // OCP 4-bit float E2M1: +/- {0, 0.5, 1, 1.5, 2, 3, 4, 6}
};

enum class QuantScaleCode : uint8_t {
  None = 0,
  LeHalfAtOffset = 1, // IEEE f16 at little-endian byte offset superScaleOffset
  E8M0 = 2,           // 8-bit shared power-of-two exponent 2^(e - scaleBias)
  FloatView = 3,      // Direct floating-point element from scale view
};

struct BitPlaneSpec {
  int baseByte;      // byte offset of plane in super-block record
  int halfStride;    // byte stride per 128-element half
  int quadByteStep;  // byte step for (quad & 1) where quad = (i % 128) / 32
  int quadShiftMode; // 0: shift = shiftUnit * (quad >> 1); 1: shiftUnit * quad
  int shiftUnit;     // bit shift step per quad unit
  int bitWidth;      // extracted bit width
  int destBitOffset; // bit offset in reconstructed integer code
};

struct QuantFormatSpec {
  const char *name;
  QuantRoleLayout roleLayout;
  QuantValueCode valueCode;
  QuantScaleCode scaleCode;
  int blockElements;
  int subBlockElements;
  int recordBytes;
  int packBits;
  int packFactor;
  int numBitPlanes;
  BitPlaneSpec planes[2];
  int subScaleOffset;
  int superScaleOffset;
  int zeroBias;
  int scaleBias;
  int scaleNan;
  bool hasZeroView;
  bool q8kActivation;
  int laneBitShifts[8];
};

inline constexpr QuantFormatSpec kQ6KSpec = {
    "q6_k",
    QuantRoleLayout::RecordGemv,
    QuantValueCode::UnsignedAffine,
    QuantScaleCode::LeHalfAtOffset,
    /*blockElements=*/256,
    /*subBlockElements=*/16,
    /*recordBytes=*/210,
    /*packBits=*/6,
    /*packFactor=*/1,
    /*numBitPlanes=*/2,
    /*planes=*/
    {{/*baseByte=*/0, /*halfStride=*/64, /*quadByteStep=*/32,
      /*quadShiftMode=*/0, /*shiftUnit=*/4, /*bitWidth=*/4,
      /*destBitOffset=*/0},
     {/*baseByte=*/128, /*halfStride=*/32, /*quadByteStep=*/0,
      /*quadShiftMode=*/1, /*shiftUnit=*/2, /*bitWidth=*/2,
      /*destBitOffset=*/4}},
    /*subScaleOffset=*/192,
    /*superScaleOffset=*/208,
    /*zeroBias=*/32,
    /*scaleBias=*/0,
    /*scaleNan=*/-1,
    /*hasZeroView=*/false,
    /*q8kActivation=*/true,
    /*laneBitShifts=*/{0, 0, 0, 0, 0, 0, 0, 0},
};

inline constexpr QuantFormatSpec kMXFP4Spec = {
    "mxfp4",
    QuantRoleLayout::BlockScaledPair,
    QuantValueCode::Fp4E2M1,
    QuantScaleCode::E8M0,
    /*blockElements=*/32,
    /*subBlockElements=*/0,
    /*recordBytes=*/16,
    /*packBits=*/4,
    /*packFactor=*/2,
    /*numBitPlanes=*/0,
    /*planes=*/{{0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0}},
    /*subScaleOffset=*/-1,
    /*superScaleOffset=*/-1,
    /*zeroBias=*/0,
    /*scaleBias=*/127,
    /*scaleNan=*/255,
    /*hasZeroView=*/false,
    /*q8kActivation=*/false,
    /*laneBitShifts=*/{0, 4, 0, 0, 0, 0, 0, 0},
};

inline constexpr QuantFormatSpec kAWQInt4Spec = {
    "awq_int4",
    QuantRoleLayout::GroupedWeightRhs,
    QuantValueCode::UnsignedAffine,
    QuantScaleCode::FloatView,
    /*blockElements=*/128,
    /*subBlockElements=*/0,
    /*recordBytes=*/0,
    /*packBits=*/4,
    /*packFactor=*/8,
    /*numBitPlanes=*/0,
    /*planes=*/{{0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0}},
    /*subScaleOffset=*/-1,
    /*superScaleOffset=*/-1,
    /*zeroBias=*/0,
    /*scaleBias=*/0,
    /*scaleNan=*/-1,
    /*hasZeroView=*/true,
    /*q8kActivation=*/false,
    /*laneBitShifts=*/{0, 16, 4, 20, 8, 24, 12, 28},
};

inline constexpr QuantFormatSpec kQuantFormats[] = {
    kQ6KSpec,
    kMXFP4Spec,
    kAWQInt4Spec,
};

inline const QuantFormatSpec *lookupQuantFormat(llvm::StringRef name) {
  for (const auto &spec : kQuantFormats)
    if (name == spec.name)
      return &spec;
  return nullptr;
}
} // namespace intentdsl::kir
