#ifndef INTENT_ANALYSIS_CONTRACTIONAXES_H
#define INTENT_ANALYSIS_CONTRACTIONAXES_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <optional>
#include <string>

namespace intent {

enum class ContractionOperand { Lhs, Rhs };

struct ContractionAxisPair {
  unsigned lhs, rhs;
};

struct ContractionResultAxis {
  ContractionOperand operand;
  unsigned axis;
};

// Structural contraction semantics only: no values, extents, provenance or
// execution policy. Result positions follow the lhs non-reduction axes, then
// the rhs axes that are neither reduced nor paired batches. Repeated source
// identities do not merge these positions. Recompute when axis lists change.
struct ContractionAxes {
  static std::optional<ContractionAxes> get(
      unsigned lhsRank, unsigned rhsRank,
      llvm::ArrayRef<int64_t> lhsReduction,
      llvm::ArrayRef<int64_t> rhsReduction,
      llvm::ArrayRef<int64_t> lhsBatch, llvm::ArrayRef<int64_t> rhsBatch,
      std::string *failureReason = nullptr);

  llvm::SmallVector<ContractionAxisPair> reduction, batch;
  llvm::SmallVector<unsigned> lhsFree, rhsFree;
  llvm::SmallVector<ContractionResultAxis> results;
  // Reduction axes have no result position; both members of a batch pair map
  // to the same position, whose representative is the lhs member.
  llvm::SmallVector<std::optional<unsigned>> lhsResultAxes, rhsResultAxes;

  // Shape/dtype/capability checks remain with each consumer. This predicate
  // only recognizes the existing [..., M, K] x [..., K, N] axis convention.
  bool hasCanonicalMatrixAxes() const;
};

// Relate two operands broadcast into one product domain to a contraction.
// Each projection maps a product axis to an operand axis, or to no axis for
// an introduced broadcast. Unit facts are indexed by operand axis and must
// describe logical singleton domains, not one-element physical tiles.
// This query neither proves fusion legality nor chooses a physical orientation.
struct ProductContractionAxes {
  static std::optional<ProductContractionAxes> get(
      llvm::ArrayRef<std::optional<unsigned>> lhsProjection,
      llvm::ArrayRef<bool> lhsUnitAxes,
      llvm::ArrayRef<std::optional<unsigned>> rhsProjection,
      llvm::ArrayRef<bool> rhsUnitAxes,
      llvm::ArrayRef<int64_t> reductionAxes,
      std::string *failureReason = nullptr);

  // Operand axes retained in their original source order. The contraction
  // axis positions below refer to these compressed operands.
  llvm::SmallVector<int64_t> lhsKept, rhsKept;
  ContractionAxes axes;
  // Formal contraction result -> product axis, and reduced product result
  // -> formal contraction result. The latter restores the original output.
  llvm::SmallVector<int64_t> resultProductAxes, resultPermutation;
};

} // namespace intent

#endif
