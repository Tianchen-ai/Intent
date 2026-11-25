#pragma once
#include "intentdsl/levels/algorithm/kir/ir/Storage.h"
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
} // namespace CI_UNIT_NAMESPACE
