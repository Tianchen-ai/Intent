#ifndef INTENT_DIALECT_Block_IR_PROGRAM_H
#define INTENT_DIALECT_Block_IR_PROGRAM_H

#include "Intent/Dialect/Block/IR/BlockAttrs.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"

#include <optional>

namespace intent::block {

// The source supplies the physical axis relation; the caller owns the declared
// accumulator element type. An empty reduction-axis set preserves every axis.
mlir::FailureOr<mlir::Type> inferCollectiveResultType(
    mlir::Type source, llvm::ArrayRef<int64_t> reducedAxes,
    mlir::Type resultElement);

// Native reductions may use axis -1 for a whole-fragment scalar reduction.
// Scans always require an explicit nonnegative source axis.
mlir::LogicalResult inferScalarCollectiveResultTypes(
    std::optional<mlir::Location> location, mlir::ValueRange sources,
    int64_t axis, bool scan, llvm::SmallVectorImpl<mlir::Type> &results);

mlir::LogicalResult verifyScalarCollective(
    mlir::Operation *owner, mlir::ValueRange sources,
    mlir::ValueRange identities, mlir::ResultRange results, mlir::Region &combine,
    int64_t axis, bool scan);

inline constexpr llvm::StringLiteral kernelAttr = "intent_block.kernel";
inline constexpr llvm::StringLiteral capabilitiesAttr =
    "intent_block.capabilities";
inline constexpr llvm::StringLiteral programSpaceAttr =
    "intent_block.program_space";
inline constexpr llvm::StringLiteral gridRankAttr = "intent_block.grid_rank";
inline constexpr llvm::StringLiteral originAttr = "intent_block.origin";
inline constexpr llvm::StringLiteral effectOriginsAttr =
    "intent_block.effect_origins";
inline constexpr llvm::StringLiteral argumentBindingAttr = "intent_block.argument_binding";
inline constexpr llvm::StringLiteral dimensionAttr = "intent_block.dimension";
inline constexpr llvm::StringLiteral sourceSubregionAttr =
    "intent_block.source_subregion";
inline constexpr llvm::StringLiteral sourceSubregionBoundAttr =
    "intent_block.source_subregion_bound";
inline constexpr llvm::StringLiteral worksetCoordinateRangeAttr =
    "intent_block.workset_coordinate_range";
inline constexpr llvm::StringLiteral worksetAxisAttr =
    "intent_block.workset_axis";
inline constexpr llvm::StringLiteral reductionSourcesAttr =
    "intent_block.reduction_sources";
inline constexpr llvm::StringLiteral independentIterationAttr =
    "intent_block.independent_iteration";
inline constexpr llvm::StringLiteral physicalTailAttr =
    "intent_block.physical_tail";
inline constexpr llvm::StringLiteral programBoundedOriginAttr =
    "intent_block.program_bounded_origin";
inline constexpr llvm::StringLiteral configurationsAttr =
    "intent_block.configurations";
inline constexpr llvm::StringLiteral parametersAttr = "intent_block.parameters";
inline constexpr llvm::StringLiteral tuningProfilesAttr =
    "intent_block.tuning_profiles";

enum class CoordinateRole : int64_t {
  Unspecified = -1,
  Workset = 0,
  PointwiseOwnership = 1,
  TiledWorkset = 2,
  ContractionM = 3,
  ContractionN = 4,
  TraversalWorker = 5,
  IndirectTraversal = 6,
};

} // namespace intent::block

#endif
