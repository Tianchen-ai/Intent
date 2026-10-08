#include "Intent/Analysis/ContractionAxes.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/STLExtras.h"

namespace intent {

std::optional<ContractionAxes> ContractionAxes::get(
    unsigned lhsRank, unsigned rhsRank, llvm::ArrayRef<int64_t> lhsReduction,
    llvm::ArrayRef<int64_t> rhsReduction, llvm::ArrayRef<int64_t> lhsBatch,
    llvm::ArrayRef<int64_t> rhsBatch, std::string *failureReason) {
  auto reject = [&](const std::string &reason) -> std::optional<ContractionAxes> {
    if (failureReason) *failureReason = reason;
    return std::nullopt;
  };
  if (lhsReduction.empty())
    return reject("requires at least one reduction pair");
  if (lhsReduction.size() != rhsReduction.size())
    return reject("lhs/rhs reduction pair counts disagree");
  if (lhsBatch.size() != rhsBatch.size())
    return reject("lhs/rhs batch pair counts disagree");

  ContractionAxes result;
  llvm::SmallBitVector lhsUsed(lhsRank), rhsUsed(rhsRank);
  auto appendPairs = [&](llvm::ArrayRef<int64_t> left,
                         llvm::ArrayRef<int64_t> right,
                         llvm::SmallVectorImpl<ContractionAxisPair> &pairs,
                         const char *role) -> std::optional<std::string> {
    for (unsigned index = 0; index < left.size(); ++index) {
      if (left[index] < 0 || static_cast<uint64_t>(left[index]) >= lhsRank)
        return std::string("lhs ") + role + " axis is out of range: " +
               std::to_string(left[index]);
      if (right[index] < 0 || static_cast<uint64_t>(right[index]) >= rhsRank)
        return std::string("rhs ") + role + " axis is out of range: " +
               std::to_string(right[index]);
      unsigned lhs = left[index], rhs = right[index];
      if (lhsUsed.test(lhs) || rhsUsed.test(rhs))
        return std::string(role) + " axis is repeated or overlaps another pair";
      lhsUsed.set(lhs);
      rhsUsed.set(rhs);
      pairs.push_back({lhs, rhs});
    }
    return std::nullopt;
  };
  if (auto error = appendPairs(lhsReduction, rhsReduction, result.reduction,
                               "reduction"))
    return reject(*error);
  if (auto error = appendPairs(lhsBatch, rhsBatch, result.batch, "batch"))
    return reject(*error);

  result.lhsResultAxes.resize(lhsRank);
  result.rhsResultAxes.resize(rhsRank);
  llvm::SmallBitVector lhsReduced(lhsRank);
  for (const auto &pair : result.reduction) lhsReduced.set(pair.lhs);
  for (unsigned axis = 0; axis < lhsRank; ++axis) {
    if (!lhsUsed.test(axis)) result.lhsFree.push_back(axis);
    if (lhsReduced.test(axis)) continue;
    result.lhsResultAxes[axis] = result.results.size();
    result.results.push_back({ContractionOperand::Lhs, axis});
  }
  for (const auto &pair : result.batch)
    result.rhsResultAxes[pair.rhs] = result.lhsResultAxes[pair.lhs];
  for (unsigned axis = 0; axis < rhsRank; ++axis) {
    if (rhsUsed.test(axis)) continue;
    result.rhsFree.push_back(axis);
    result.rhsResultAxes[axis] = result.results.size();
    result.results.push_back({ContractionOperand::Rhs, axis});
  }
  return result;
}

bool ContractionAxes::hasCanonicalMatrixAxes() const {
  unsigned rank = lhsResultAxes.size();
  if (rank < 2 || rhsResultAxes.size() != rank || reduction.size() != 1 ||
      batch.size() != rank - 2 || reduction.front().lhs != rank - 1 ||
      reduction.front().rhs != rank - 2)
    return false;
  for (unsigned axis = 0; axis < batch.size(); ++axis)
    if (batch[axis].lhs != axis || batch[axis].rhs != axis)
      return false;
  return true;
}

std::optional<ProductContractionAxes> ProductContractionAxes::get(
    llvm::ArrayRef<std::optional<unsigned>> lhsProjection,
    llvm::ArrayRef<bool> lhsUnitAxes,
    llvm::ArrayRef<std::optional<unsigned>> rhsProjection,
    llvm::ArrayRef<bool> rhsUnitAxes,
    llvm::ArrayRef<int64_t> reductionAxes, std::string *failureReason) {
  auto reject = [&](const char *reason) -> std::optional<ProductContractionAxes> {
    if (failureReason) *failureReason = reason;
    return std::nullopt;
  };
  unsigned rank = lhsProjection.size();
  if (rhsProjection.size() != rank || reductionAxes.empty())
    return reject("product projections require one common domain and nonempty reduction axes");
  llvm::SmallBitVector reduced(rank);
  for (int64_t axis : reductionAxes) {
    if (axis < 0 || static_cast<uint64_t>(axis) >= rank || reduced.test(axis))
      return reject("product reduction axis is out of range or repeated");
    reduced.set(axis);
  }
  auto invert = [&](llvm::ArrayRef<std::optional<unsigned>> projection,
                    llvm::ArrayRef<bool> units)
      -> std::optional<llvm::SmallVector<std::optional<unsigned>>> {
    llvm::SmallVector<std::optional<unsigned>> sourceToProduct(units.size());
    for (auto [product, source] : llvm::enumerate(projection)) {
      if (!source) continue;
      if (*source >= units.size() || sourceToProduct[*source]) return std::nullopt;
      sourceToProduct[*source] = product;
    }
    for (unsigned source = 0; source < units.size(); ++source)
      if (!sourceToProduct[source] && !units[source]) return std::nullopt;
    return sourceToProduct;
  };
  auto lhsDomain = invert(lhsProjection, lhsUnitAxes);
  auto rhsDomain = invert(rhsProjection, rhsUnitAxes);
  if (!lhsDomain || !rhsDomain)
    return reject("product projection must map each non-unit operand axis exactly once");

  llvm::SmallBitVector leftActive(rank), rightActive(rank);
  for (unsigned axis = 0; axis < rank; ++axis) {
    auto lhs = lhsProjection[axis], rhs = rhsProjection[axis];
    bool left = lhs && !lhsUnitAxes[*lhs];
    bool right = rhs && !rhsUnitAxes[*rhs];
    if (reduced.test(axis) && (!left || !right))
      return reject("contraction requires both operands to vary along every reduction axis");
    // A common singleton free axis still belongs to the result. Retain one
    // actual operand axis to represent it, without inventing an extent.
    if (!reduced.test(axis) && !left && !right) {
      if (lhs) left = true;
      else if (rhs) right = true;
      else return reject("unit result axis has no operand representative");
    }
    if (left) leftActive.set(axis);
    if (right) rightActive.set(axis);
  }

  ProductContractionAxes result;
  auto retain = [&](llvm::ArrayRef<std::optional<unsigned>> sourceToProduct,
                    const llvm::SmallBitVector &active,
                    llvm::SmallVectorImpl<int64_t> &kept) {
    llvm::SmallVector<std::optional<unsigned>> positions(rank);
    for (auto [source, product] : llvm::enumerate(sourceToProduct)) {
      if (!product || !active.test(*product)) continue;
      positions[*product] = kept.size();
      kept.push_back(source);
    }
    return positions;
  };
  auto lhsPositions = retain(*lhsDomain, leftActive, result.lhsKept);
  auto rhsPositions = retain(*rhsDomain, rightActive, result.rhsKept);
  llvm::SmallVector<int64_t> lhsReduction, rhsReduction, lhsBatch, rhsBatch;
  for (int64_t axis : reductionAxes) {
    lhsReduction.push_back(*lhsPositions[axis]);
    rhsReduction.push_back(*rhsPositions[axis]);
  }
  for (unsigned axis = 0; axis < rank; ++axis)
    if (!reduced.test(axis) && lhsPositions[axis] && rhsPositions[axis]) {
      lhsBatch.push_back(*lhsPositions[axis]);
      rhsBatch.push_back(*rhsPositions[axis]);
    }
  auto axes = ContractionAxes::get(result.lhsKept.size(), result.rhsKept.size(),
      lhsReduction, rhsReduction, lhsBatch, rhsBatch, failureReason);
  if (!axes) return std::nullopt;
  result.axes = std::move(*axes);
  for (auto output : result.axes.results) {
    bool left = output.operand == ContractionOperand::Lhs;
    unsigned source = left ? result.lhsKept[output.axis] : result.rhsKept[output.axis];
    result.resultProductAxes.push_back(*(left ? (*lhsDomain)[source] : (*rhsDomain)[source]));
  }
  for (unsigned axis = 0; axis < rank; ++axis) {
    if (reduced.test(axis)) continue;
    auto position = llvm::find(result.resultProductAxes, axis);
    if (position == result.resultProductAxes.end())
      return reject("contraction does not preserve every reduced-product result axis");
    result.resultPermutation.push_back(position - result.resultProductAxes.begin());
  }
  return result;
}

} // namespace intent
