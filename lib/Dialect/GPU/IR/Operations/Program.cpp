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

ParameterAttr ParameterOp::getDeclaration() {
  return lookupParameterDeclaration(getOperation(), getReference());
}

namespace {

bool valueMatchesPhysicalExtent(Value value, PhysicalExprAttr extent) {
  if (auto physical = value.getDefiningOp<PhysicalExprOp>())
    return physical.getExpression() == extent;
  auto kind = extent.getKind();
  if (kind == PhysicalExprKind::Constant) {
    auto constant = value.getDefiningOp<arith::ConstantIndexOp>();
    return constant && constant.value() == extent.getValue();
  }
  if (kind == PhysicalExprKind::Parameter) {
    auto parameter = value.getDefiningOp<ParameterOp>();
    return parameter && parameter.getReference() == extent.getParameterReference();
  }
  return false;
}

} // namespace

LogicalResult ParameterOp::verify() {
  ParameterAttr declaration = getDeclaration();
  if (!declaration)
    return emitOpError("references an undeclared kernel parameter") << "; reference=" << getReference();
  if (!declaration.isExtent())
    return emitOpError("index parameter read requires a positive index declaration");
  return success();
}

LogicalResult PhysicalExprOp::verify() { return success(); }

LogicalResult ProgramIdOp::verify() {
  return success();
}

LogicalResult WorksetCoordinateOp::verify() {
  const int64_t sourceId = getSourceIdAttr().getInt();
  const int64_t sourceAxis = getSourceAxisAttr().getInt();
  const int64_t sourceRank = getSourceRankAttr().getInt();
  const int64_t dimensionId = getDimensionIdAttr().getInt();
  if (sourceId <= 0)
    return emitOpError(
        "workset coordinate requires logical source provenance");
  if (sourceRank <= 0 || sourceAxis < 0 || sourceAxis >= sourceRank)
    return emitOpError(
        "workset coordinate source axis is outside its logical source rank");
  if (dimensionId <= 0)
    return emitOpError(
        "workset coordinate requires a logical dimension identity");
  auto worksetAxis = (*this)->getAttrOfType<IntegerAttr>(worksetAxisAttr);
  if (!worksetAxis || worksetAxis.getInt() < 0)
    return emitOpError(
        "workset coordinate requires a non-negative physical workset axis");
  return success();
}

LogicalResult DelinearizeOp::verify() {
  if (getExtents().empty() || getExtents().size() != getCoordinates().size())
    return emitOpError(
        "delinearization requires one runtime extent per coordinate");
  return success();
}

std::optional<DecodedCoordinate> queryDecodedCoordinate(Value value) {
  if (auto argument = dyn_cast<BlockArgument>(value))
    if (auto group = dyn_cast<ExecutionGroupOp>(argument.getOwner()->getParentOp()))
      return DecodedCoordinate{group.getLinear(), group.getExtents(),
                               argument.getArgNumber()};
  if (auto result = dyn_cast<OpResult>(value))
    if (auto decode = dyn_cast<DelinearizeOp>(result.getOwner()))
      return DecodedCoordinate{decode.getLinear(), decode.getExtents(),
                               result.getResultNumber()};
  return std::nullopt;
}

void ExecutionGroupOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent())
    regions.emplace_back(&getBody());
  else
    regions.emplace_back();
}

LogicalResult ExecutionGroupOp::verify() {
  if (getBody().empty() || !llvm::hasSingleElement(getBody()))
    return emitOpError("requires one execution block");
  if (getExtents().empty() || getExtents().size() != getCoordinates().size() ||
      getLaunchExtents().size() != getExtents().size() ||
      getCoordinateRoles().size() != getExtents().size())
    return emitOpError("requires one runtime extent, launch extent, role and block argument per coordinate");
  for (BlockArgument coordinate : getCoordinates())
    if (!coordinate.getType().isIndex())
      return emitOpError("execution coordinates must have index type");
  for (Attribute extent : getLaunchExtents())
    if (!isa<PhysicalExprAttr>(extent))
      return emitOpError("launch extents must be typed expressions");
  for (int64_t role : getCoordinateRoles())
    if (role < static_cast<int64_t>(CoordinateRole::Unspecified) ||
        role > static_cast<int64_t>(CoordinateRole::IndirectTraversal))
      return emitOpError("coordinate role is invalid");
  if (getGroupId() < 0)
    return emitOpError("requires a nonnegative execution group identity");
  return success();
}

LogicalResult DimOp::verify() {
  return static_cast<uint64_t>(getAxis()) < getView().getType().getRank()
             ? success()
             : emitOpError("view dimension axis is out of range");
}

LogicalResult RangeOp::verify() {
  return getResult().getType().getSourceId() != 0
             ? success()
             : emitOpError("physical range requires logical source provenance");
}

LogicalResult ViewOverlapOp::verify() {
  auto kernel = (*this)->getParentOfType<func::FuncOp>();
  if (!kernel || !kernel->hasAttr(kernelAttr) ||
      (*this)->getBlock() != &kernel.front())
    return emitOpError("must be declared in a physical GPU kernel entry block");
  if (getLhs() == getRhs())
    return emitOpError("requires two distinct external view arguments");
  for (Value value : getOperands()) {
    auto argument = dyn_cast<BlockArgument>(value);
    if (!argument || argument.getOwner() != &kernel.front())
      return emitOpError("requires external view ABI arguments");
    if (!getPublicView(argument))
      return emitOpError("requires external view ABI arguments");
  }
  return success();
}

LogicalResult RangeBoundOp::verify() {
  return getBound() <= 2 ? success()
                         : emitOpError("range bound selector is invalid");
}

LogicalResult MakeRangeOp::verify() {
  auto result = getResult().getType();
  if (result.getShape().size() != 1 ||
      result.getElementType() != getStart().getType() ||
      getExtent().getType() != getStart().getType() ||
      getStep().getType() != getStart().getType() ||
      getLogicalStart().getType() != getStart().getType() ||
      getLogicalStop().getType() != getStart().getType())
    return emitOpError("physical range must produce a rank-one index fragment");
  auto mapping = dyn_cast<AxisMapAttr>(result.getAxisMaps()[0]);
  if (!mapping || mapping.getSourceId() != static_cast<uint64_t>(getSourceId()) ||
      mapping.getSourceAxis() != static_cast<uint32_t>(getSourceAxis()) ||
      mapping.getDerived() != getDerived())
    return emitOpError("physical range result lost logical provenance");
  auto physicalExtent = dyn_cast<PhysicalExprAttr>(result.getShape()[0]);
  if (!physicalExtent ||
      !valueMatchesPhysicalExtent(getExtent(), physicalExtent)) {
    InFlightDiagnostic diagnostic = emitOpError(
        "physical range extent does not match its result fragment extent");
    diagnostic << "; result_extent=" << result.getShape()[0]
               << ", extent_operand=" << getExtent()
               << ", source_id=" << getSourceId()
               << ", source_axis=" << getSourceAxis();
    if (Operation *producer = getExtent().getDefiningOp())
      diagnostic << ", extent_producer=" << producer->getName();
    if (Operation *parent = (*this)->getParentOp())
      diagnostic << ", parent=" << parent->getName();
    return failure();
  }
  return success();
}

} // namespace intent::gpu
