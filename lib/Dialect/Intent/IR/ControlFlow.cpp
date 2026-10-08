#include "Intent/Dialect/Intent/IR/IntentOps.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent {
namespace {

template <typename Terminator>
LogicalResult verifyTerminator(Operation *owner, Region &region) {
  if (!llvm::hasSingleElement(region) || region.front().empty() ||
      !isa<Terminator>(region.front().back()))
    return owner->emitOpError("requires a single block with the declared terminator");
  return success();
}

LogicalResult verifyCoordinates(Operation *owner, Value source, Region &region,
                                unsigned carryCount) {
  unsigned rank;
  uint64_t sourceId;
  if (auto domain = dyn_cast<DomainType>(source.getType())) {
    rank = domain.getRank();
    sourceId = domain.getOriginId();
  } else {
    auto subregion = cast<RegionType>(source.getType());
    rank = subregion.getRank();
    sourceId = subregion.getSourceId();
  }
  if (failed(verifyTerminator<YieldOp>(owner, region))) return failure();
  Block &body = region.front();
  if (body.getNumArguments() != rank + carryCount)
    return owner->emitOpError("body requires source coordinates followed by loop carries");
  for (auto [axis, argument] : llvm::enumerate(body.getArguments().take_front(rank))) {
    auto coordinate = dyn_cast<LogicalIndexType>(argument.getType());
    if (!coordinate || coordinate.getSourceId() != sourceId ||
        coordinate.getAxis() != axis)
      return owner->emitOpError("iteration coordinate lost its source/axis provenance");
  }
  return success();
}

} // namespace

LogicalResult ParallelOp::verify() {
  if (failed(verifyCoordinates(*this, getSource(), getBody(), 0))) return failure();
  if (!cast<YieldOp>(getBody().front().back()).getInputs().empty())
    return emitOpError("parallel iteration cannot carry SSA values");
  return success();
}

LogicalResult ForOp::verify() {
  if (getInitArgs().size() != getResults().size())
    return emitOpError("requires one result per initial loop carry");
  return verifyCoordinates(*this, getSource(), getBody(), getInitArgs().size());
}

OperandRange ForOp::getEntrySuccessorOperands(RegionBranchPoint point) {
  return getInitArgs();
}

void ForOp::getSuccessorRegions(RegionBranchPoint point,
                                SmallVectorImpl<RegionSuccessor> &regions) {
  // The source may be empty. Initial carries therefore reach either the body
  // or the results; yielded carries reach the next iteration or the results.
  regions.emplace_back(&getBody(), getRegionIterArgs());
  regions.emplace_back(getResults());
}

LogicalResult IfOp::verify() {
  for (Region &region : getOperation()->getRegions()) {
    if (failed(verifyTerminator<YieldOp>(*this, region))) return failure();
    if (region.front().getNumArguments())
      return emitOpError("conditional branches do not receive block arguments");
  }
  return success();
}

void IfOp::getSuccessorRegions(RegionBranchPoint point,
                               SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent()) {
    regions.emplace_back(&getThenRegion());
    regions.emplace_back(&getElseRegion());
  } else {
    regions.emplace_back(getResults());
  }
}

LogicalResult WhileOp::verify() {
  // Intent carries keep one stable schema, while generic scf.while permits
  // different before/after schemas. The remaining edges use the MLIR verifier.
  if (!llvm::equal(getInitArgs().getTypes(), getResultTypes()))
    return emitOpError("while carries must keep their initial/result schema");
  if (failed(verifyTerminator<ConditionOp>(*this, getBefore())) ||
      failed(verifyTerminator<YieldOp>(*this, getAfter())))
    return failure();
  return success();
}

OperandRange WhileOp::getEntrySuccessorOperands(RegionBranchPoint point) {
  return getInitArgs();
}

void WhileOp::getSuccessorRegions(RegionBranchPoint point,
                                  SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent() || point.getRegionOrNull() == &getAfter()) {
    regions.emplace_back(&getBefore(), getBeforeArguments());
  } else {
    regions.emplace_back(&getAfter(), getAfterArguments());
    regions.emplace_back(getResults());
  }
}

LogicalResult ConditionOp::verify() {
  auto loop = cast<WhileOp>(getOperation()->getParentOp());
  if (getOperation()->getParentRegion() != &loop.getBefore())
    return emitOpError("must terminate the before region of a while loop");
  return success();
}

MutableOperandRange ConditionOp::getMutableSuccessorOperands(RegionBranchPoint point) {
  return getArgsMutable();
}

} // namespace intent
