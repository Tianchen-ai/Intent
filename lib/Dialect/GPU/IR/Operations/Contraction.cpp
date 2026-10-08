#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/IR/ProgramInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseSet.h"
#include "Verification.h"

using namespace mlir;
namespace intent::gpu {
using namespace operation_detail;

namespace {

LogicalResult verifyContractAxes(Operation *owner, FragmentType lhs,
                                 FragmentType rhs, FragmentType result,
                                 ArrayRef<int64_t> lhsReduction,
                                 ArrayRef<int64_t> rhsReduction,
                                 ArrayRef<int64_t> lhsBatch,
                                 ArrayRef<int64_t> rhsBatch,
                                 std::optional<int64_t> lhsExtentException =
                                     std::nullopt) {
  if (lhsReduction.empty() || lhsReduction.size() != rhsReduction.size() ||
      lhsBatch.size() != rhsBatch.size())
    return owner->emitOpError("physical contract axis-pair counts disagree");
  llvm::DenseSet<int64_t> lhsAxes;
  llvm::DenseSet<int64_t> rhsAxes;
  auto verifyPairs = [&](ArrayRef<int64_t> leftAxes,
                         ArrayRef<int64_t> rightAxes,
                         StringRef role) -> LogicalResult {
    for (auto [left, right] : llvm::zip(leftAxes, rightAxes)) {
      if (left < 0 || right < 0 ||
          static_cast<size_t>(left) >= lhs.getShape().size() ||
          static_cast<size_t>(right) >= rhs.getShape().size())
        return owner->emitOpError()
               << "physical contract " << role
               << " axis is out of range: lhs_axis=" << left
               << ", rhs_axis=" << right;
      if (!lhsAxes.insert(left).second || !rhsAxes.insert(right).second)
        return owner->emitOpError()
               << "physical contract " << role << " axis is repeated";
      if (lhs.getShape()[left] != rhs.getShape()[right] &&
          (!lhsExtentException || left != *lhsExtentException))
        return owner->emitOpError()
               << "physical contract " << role
               << " extents disagree: lhs_axis=" << left
               << ", lhs_extent=" << lhs.getShape()[left]
               << ", rhs_axis=" << right
               << ", rhs_extent=" << rhs.getShape()[right];
    }
    return success();
  };
  if (failed(verifyPairs(lhsReduction, rhsReduction, "reduction")) ||
      failed(verifyPairs(lhsBatch, rhsBatch, "batch")))
    return failure();
  size_t expectedRank = lhsBatch.size() +
                        (lhs.getShape().size() - lhsAxes.size()) +
                        (rhs.getShape().size() - rhsAxes.size());
  return result.getShape().size() == expectedRank
             ? success()
             : owner->emitOpError(
                   "physical contract result rank disagrees with batch/free axes");
}

} // namespace

LogicalResult ContractOp::verify() {
  auto lhs = getLhs().getType();
  auto rhs = getRhs().getType();
  auto accumulator = getAccumulator().getType();
  auto result = getResult().getType();
  if (accumulator != result || lhs.getOwner() != rhs.getOwner() ||
      lhs.getOwner() != result.getOwner()) {
    InFlightDiagnostic diagnostic =
        emitOpError("physical contract relation/ownership is inconsistent");
    diagnostic << "; lhs_owner=" << lhs.getOwner()
               << ", rhs_owner=" << rhs.getOwner()
               << ", result_owner=" << result.getOwner()
               << ", accumulator=" << accumulator << ", result=" << result;
    return failure();
  }
  return verifyContractAxes(getOperation(), lhs, rhs, result,
                            getLhsReductionAxes(), getRhsReductionAxes(),
                            getLhsBatchAxes(), getRhsBatchAxes());
}

LogicalResult ScaledContractOp::verify() {
  auto lhs = getLhs().getType();
  auto lhsScale = getLhsScale().getType();
  auto rhs = getRhs().getType();
  auto rhsScale = getRhsScale().getType();
  auto result = getResult().getType();
  auto carrierMatchesFormat = [](Type element, ScaledFormat format) {
    return format == ScaledFormat::E4M3
        ? isa<Float8E4M3FNType>(element) || element.isUnsignedInteger(8)
        : element.isUnsignedInteger(8);
  };
  if (!carrierMatchesFormat(lhs.getElementType(), getLhsFormat()) ||
      !carrierMatchesFormat(rhs.getElementType(), getRhsFormat()) ||
      !lhsScale.getElementType().isUnsignedInteger(8) ||
      !rhsScale.getElementType().isUnsignedInteger(8))
    return emitOpError(
        "scaled-contract requires format-matched carriers and u8 E8M0 scales");
  auto sameLogicalAxis = [](FragmentType left, unsigned leftAxis,
                            FragmentType right, unsigned rightAxis) {
    auto lhsMap = cast<AxisMapAttr>(left.getAxisMaps()[leftAxis]);
    auto rhsMap = cast<AxisMapAttr>(right.getAxisMaps()[rightAxis]);
    return lhsMap.getDimensionId() > 0 &&
           lhsMap.getDimensionId() == rhsMap.getDimensionId();
  };
  auto samePhysicalAxis = [](FragmentType left, unsigned leftAxis,
                             FragmentType right, unsigned rightAxis) {
    return left.getShape()[leftAxis] == right.getShape()[rightAxis];
  };
  auto constantExtent = [](FragmentType value,
                           unsigned axis) -> std::optional<int64_t> {
    auto extent = cast<PhysicalExprAttr>(value.getShape()[axis]);
    return extent.getKind() ==
                   PhysicalExprKind::Constant
               ? std::optional<int64_t>(extent.getValue())
               : std::nullopt;
  };
  auto carrierExtent = [](ScaledFormat format,
                          uint64_t group) -> std::optional<int64_t> {
    uint64_t packing = format == ScaledFormat::E2M1 ? 2 : 1;
    return group % packing == 0
               ? std::optional<int64_t>(group / packing)
               : std::nullopt;
  };
  ArrayRef<int64_t> lhsReduction = getLhsReductionAxes();
  ArrayRef<int64_t> rhsReduction = getRhsReductionAxes();
  bool fixedAxes = lhsReduction.size() == 2 && rhsReduction.size() == 2 &&
                   lhsReduction[0] == 1 && lhsReduction[1] == 2 &&
                   rhsReduction[0] == 0 && rhsReduction[1] == 1 &&
                   getLhsBatchAxes().empty() && getRhsBatchAxes().empty();
  if (getLhsGroupSize() == 0 ||
      getLhsGroupSize() != getRhsGroupSize() ||
      getAccumulator().getType() != result || lhs.getOwner() != rhs.getOwner() ||
      lhs.getOwner() != lhsScale.getOwner() ||
      lhs.getOwner() != rhsScale.getOwner() || lhs.getOwner() != result.getOwner() ||
      lhs.getShape().size() != 3 || lhsScale.getShape().size() != 2 ||
      rhs.getShape().size() != 3 || rhsScale.getShape().size() != 2 ||
      result.getShape().size() != 2 || !fixedAxes)
    return emitOpError("scaled-contract physical schema is invalid");
  std::optional<int64_t> lhsCarrier =
      carrierExtent(getLhsFormat(), getLhsGroupSize());
  std::optional<int64_t> rhsCarrier =
      carrierExtent(getRhsFormat(), getRhsGroupSize());
  if (!lhsCarrier || !rhsCarrier || constantExtent(lhs, 2) != lhsCarrier ||
      constantExtent(rhs, 1) != rhsCarrier ||
      !sameLogicalAxis(lhs, 0, lhsScale, 0) ||
      !samePhysicalAxis(lhs, 0, lhsScale, 0) ||
      !sameLogicalAxis(lhs, 1, lhsScale, 1) ||
      !samePhysicalAxis(lhs, 1, lhsScale, 1) ||
      !sameLogicalAxis(lhs, 1, rhs, 0) ||
      !samePhysicalAxis(lhs, 1, rhs, 0) ||
      !sameLogicalAxis(lhs, 1, rhsScale, 1) ||
      !samePhysicalAxis(lhs, 1, rhsScale, 1) ||
      !sameLogicalAxis(rhs, 2, rhsScale, 0) ||
      !samePhysicalAxis(rhs, 2, rhsScale, 0) ||
      !sameLogicalAxis(lhs, 0, result, 0) ||
      !samePhysicalAxis(lhs, 0, result, 0) ||
      !sameLogicalAxis(rhs, 2, result, 1) ||
      !samePhysicalAxis(rhs, 2, result, 1))
    return emitOpError("scaled-contract physical scale-axis relation is invalid");
  return success();
}

LogicalResult SparseContractOp::verify() {
  auto lhs = getCompressed().getType();
  auto rhs = getRhs().getType();
  auto result = getResult().getType();
  if (!getFormat() ||
      getFormat().getCompressionAxis() >=
          getCompressed().getType().getShape().size() ||
      getAccumulator().getType() != result || lhs.getOwner() != rhs.getOwner() ||
      lhs.getOwner() != result.getOwner())
    return emitOpError("sparse-contract physical schema is invalid");
  return verifyContractAxes(getOperation(), lhs, rhs, result,
                            getLhsReductionAxes(), getRhsReductionAxes(),
                            getLhsBatchAxes(), getRhsBatchAxes(),
                            static_cast<int64_t>(
                                getFormat().getCompressionAxis()));
}

} // namespace intent::gpu
