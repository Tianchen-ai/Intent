#pragma once
#include <tuple>

namespace intentdsl::kir {
// Merge a tile's weights into the running FP32 normalization. Both the block
// and CPU exits use the author's subtraction, exponential and reduction. The
// value contraction, when present, uses the same weights and old-state scale.
template <class V, class Origin, class Scale, class Project, class Mul, class Add>
auto mergeNormalizationPanel(V oldPeak, V newPeak, V oldMass, Origin originFor,
                             Scale scaleFor, Project project, Mul mul, Add add) {
  V origin = originFor(newPeak);
  V scale = scaleFor(oldPeak, origin);
  V previous = mul(oldMass, scale);
  auto [mass, weights] = project(origin);
  return std::tuple{scale, add(previous, mass), weights};
}
} // namespace intentdsl::kir
