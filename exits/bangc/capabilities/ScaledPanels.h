#pragma once
#include "intentdsl/levels/algorithm/kir/ir/Storage.h"
#include "intentdsl/levels/storage/dsa/ir/Matrix.h"
#include <numeric>

namespace CI_UNIT_NAMESPACE {
// Each iteration decodes a whole number of scale groups and consumes an
// equal fraction of the reduction. WRAM row padding is applied at publication.
inline int64_t sharedDecodeGroups(const intentdsl::kir::QuantFormatSpec *spec,
                                  int64_t groups, int64_t depth) {
  if (!spec ||
      spec->roleLayout != intentdsl::kir::QuantRoleLayout::BlockScaledPair ||
      spec->packBits != 4 || spec->packFactor != 2 || groups <= 0)
    return 0;
  return std::gcd(groups, std::max<int64_t>(1, depth / spec->blockElements));
}

// Number of reduction groups to bring on chip per outer transfer. Retain the
// streaming plan when the view cannot be coalesced or the larger rectangle
// would exceed the capacity of its physical owner. Count alignment per slot.
inline int64_t residentQuantGroups(intentdsl::block::MatrixAccess &a,
                                   llvm::StringRef placement,
                                   const intentdsl::dsa::MatrixStorage &ms,
                                   int64_t privateBudget) {
  using namespace intentdsl::kir;
  auto *fmt = quantFormatSpec(a.dot);
  if (!fmt || a.shape[2] <= 0)
    return 0;
  int64_t m = a.tile[0], n = a.tile[1], k = a.tile[2];
  int64_t groups = a.shape[2] / fmt->blockElements;
  if (fmt->roleLayout == QuantRoleLayout::RecordGemv) {
    if (placement != "private" || !fmt->q8kActivation || fmt->numBitPlanes != 2 ||
        fmt->blockElements != 256 || fmt->subBlockElements != 16 || m > 128 ||
        a.stride(a.dot.getOperand(0), 2) != 1 ||
        a.stride(a.dot.getOperand(0), 1) != fmt->recordBytes)
      return 0;
    auto available = privateBudget - ms.privateBytes(m, n, k, "quant_contract");
    auto slab = std::min(groups, (available - 127) / (m * fmt->recordBytes));
    return slab >= 2 ? slab : 0;
  }
  auto fits = [&](int64_t steps, int64_t widthBytes, int64_t occupied, int slots) {
    auto remaining = ms.sramBytes - occupied - slots * 127;
    return remaining > 0 && steps <= remaining / widthBytes;
  };
  if (a.shape[0] % m || a.shape[1] % n)
    return 0;
  if (fmt->roleLayout == QuantRoleLayout::BlockScaledPair &&
      placement == "shared-tile") {
    int64_t step = sharedDecodeGroups(fmt, groups, k);
    auto lhs = a.dot.getOperand(0), sa = a.dot.getOperand(1), rhs = a.dot.getOperand(2);
    RightScale scale(a.dot);
    int64_t occupied = (m+n)*std::max<int64_t>(64, step*fmt->blockElements)*2+128;
    if (step && groups > step && a.stride(lhs, 2) == 1 &&
        a.stride(lhs, 1) == fmt->recordBytes && a.stride(sa, 1) == 1 &&
        a.stride(rhs, 2) == 1 && a.stride(rhs, 1) > 0 &&
        a.stride(rhs, 0) / fmt->recordBytes == a.stride(rhs, 1) &&
        a.stride(rhs, 0) % fmt->recordBytes == 0 && scale.column == 1 &&
        a.stride(scale.view, 1) == 1 &&
        fits(groups, (m+n)*(fmt->recordBytes+1), occupied, 4))
      return groups;
  } else if (fmt->roleLayout == QuantRoleLayout::GroupedWeightRhs &&
             placement == "shared-lhs") {
    int64_t columns = ms.coresPerCluster * n;
    int64_t wordBytes = mlir::cast<mlir::MemRefType>(
        a.dot.getOperand(1).getType()).getElementTypeBitWidth() / 8;
    if (a.shape[2] > k && a.shape[1] % columns == 0 &&
        a.stride(a.dot.getOperand(1), 1) == 1 &&
        fits(a.shape[2], m*2 + columns/fmt->packFactor*wordBytes, 0, 2))
      return groups;
  }
  return 0;
}
} // namespace CI_UNIT_NAMESPACE
