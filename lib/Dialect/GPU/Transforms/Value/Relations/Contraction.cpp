#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/AccessOpInterface.h"
#include "Intent/Dialect/GPU/IR/FragmentOpInterface.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Worklist.h"

using namespace mlir;

namespace intent::gpu::value_relations {

LogicalResult alignContractAccumulator(Operation *operation,
                                       RelationWorklist &changes) {
  auto align = [&](Operation *owner, OpOperand &accumulatorOperand,
                   Value result) -> LogicalResult {
    Value accumulator = accumulatorOperand.get();
    if (accumulator.getType() == result.getType())
      return success();
    auto source = dyn_cast<FragmentType>(accumulator.getType());
    auto target = dyn_cast<FragmentType>(result.getType());
    if (!source || !target ||
        source.getElementType() != target.getElementType() ||
        source.getOwner() != target.getOwner())
      return owner->emitOpError("pointwise ownership cannot preserve the "
                                "contract accumulator relation");
    if (isLiteralZeroProjection(accumulator)) {
      OpBuilder builder(owner);
      builder.setListener(&changes);
      auto projected = projectPhysicalValueToSchema(
          builder, owner->getLoc(), accumulator, target, changes.typeChanged());
      if (failed(projected))
        return owner->emitOpError(
            "contract zero accumulator cannot adopt its result schema");
      accumulatorOperand.set(*projected);
      return success();
    }
    if (source.getAxisMaps() != target.getAxisMaps())
      return owner->emitOpError("pointwise ownership cannot preserve the "
                                "contract accumulator relation");
    auto isUnit = [](Attribute attribute) {
      auto expression = cast<PhysicalExprAttr>(attribute);
      return expression.getKind() == PhysicalExprKind::Constant &&
             expression.getValue() == 1;
    };
    SmallVector<Attribute> shape(target.getShape().begin(),
                                 target.getShape().end());
    for (unsigned axis = 0; axis < shape.size(); ++axis) {
      if (source.getShape()[axis] == target.getShape()[axis])
        continue;
      bool sourceUnit = isUnit(source.getShape()[axis]);
      bool targetUnit = isUnit(target.getShape()[axis]);
      if (sourceUnit == targetUnit)
        return owner->emitOpError(
            "pointwise ownership found two non-equivalent contract extents");
      if (targetUnit)
        shape[axis] = source.getShape()[axis];
    }
    auto aligned = FragmentType::get(
        target.getContext(), target.getElementType(),
        ArrayAttr::get(target.getContext(), shape), target.getAxisMaps(),
        target.getValidity(), target.getOwner());
    for (auto [axis, mapping] : llvm::enumerate(aligned.getAxisMaps())) {
      if (source.getShape()[axis] == aligned.getShape()[axis] &&
          target.getShape()[axis] == aligned.getShape()[axis])
        continue;
      int64_t dimension = cast<AxisMapAttr>(mapping).getDimensionId();
      if (dimension <= 0)
        return owner->emitOpError(
            "contract accumulator alignment has no dimension authority");
      if (failed(retargetFragmentAxisExtent(
              result, axis,
              cast<PhysicalExprAttr>(aligned.getShape()[axis]),
              changes.typeChanged(), &changes)))
        return failure();
      if (failed(retargetFragmentAxisExtent(
              accumulator, axis,
              cast<PhysicalExprAttr>(aligned.getShape()[axis]),
              changes.typeChanged(), &changes)))
        return failure();
    }
    changes.setType(accumulator, aligned);
    changes.setType(result, aligned);
    return success();
  };
  OpOperand *accumulator;
  Value output;
  if (auto contract = dyn_cast<ContractOp>(operation)) {
    accumulator = &contract.getAccumulatorMutable();
    output = contract.getResult();
  } else if (auto contract = dyn_cast<ScaledContractOp>(operation)) {
    accumulator = &contract.getAccumulatorMutable();
    output = contract.getResult();
  } else if (auto contract = dyn_cast<SparseContractOp>(operation)) {
    accumulator = &contract.getAccumulatorMutable();
    output = contract.getResult();
  } else {
    return success();
  }
  return align(operation, *accumulator, output);
}

WalkResult alignContractOperands(Operation *operation,
                                 RelationWorklist &changes) {
  auto contract = dyn_cast<ContractOp>(operation);
  if (!contract)
    return WalkResult::advance();
  auto kernel = operation->getParentOfType<func::FuncOp>();
  auto isUnit = [](Attribute attribute) {
    auto extent = cast<PhysicalExprAttr>(attribute);
    return extent.getKind() == PhysicalExprKind::Constant &&
           extent.getValue() == 1;
  };
  auto isUniformBatch = [&](Value value, unsigned axis) {
    auto broadcast = value.getDefiningOp<BroadcastOp>();
    auto source = broadcast
                      ? dyn_cast<FragmentType>(broadcast.getValue().getType())
                      : FragmentType();
    if (!source)
      return false;
    auto relations = queryFragmentOperandRelations(broadcast);
    if (failed(relations) || !relations->front().hasCompatibleExtents())
      return false;
    const auto *group = relations->front().groupForResultAxis(axis);
    if (!group || !group->sourceAxes.empty())
      return false;
    auto ranges = PhysicalProgramAnalysis(kernel).axisRanges(value, axis);
    return ranges.isExact() && ranges.roots.empty() && ranges.blockers.empty();
  };
  auto batchExtentAuthority = [&](Value value, unsigned axis) {
    std::pair<Value, unsigned> original{value, axis}, authority = original;
    while (Operation *operation = authority.first.getDefiningOp()) {
      if (!isa<ReshapeOp, TransposeOp>(operation))
        break;
      auto relations = queryFragmentOperandRelations(operation);
      if (failed(relations) || relations->size() != 1)
        return original;
      auto sourceAxis =
          relations->front().correspondingSourceAxis(authority.second);
      if (!sourceAxis)
        return original;
      authority = {operation->getOperand(relations->front().operandNumber),
                   *sourceAxis};
    }
    auto load = authority.first.getDefiningOp<LoadOp>();
    if (!load)
      return original;
    PhysicalProgramAnalysis analysis(kernel);
    auto isUniform = [&](Value dependency, const BroadcastProjection &relation) {
      if (!dependency || !isa<FragmentType>(dependency.getType()))
        return true;
      if (!relation.isExact() || authority.second >= relation.targetToSource.size())
        return false;
      auto sourceAxis = relation.targetToSource[authority.second];
      if (!sourceAxis)
        return true;
      auto ranges = analysis.axisRanges(dependency, *sourceAxis);
      return ranges.isExact() && ranges.roots.empty() && ranges.blockers.empty();
    };
    auto access = cast<AccessOpInterface>(load.getOperation());
    for (auto [slot, coordinate] : llvm::enumerate(load.getCoordinates()))
      if (!isUniform(coordinate, queryAccessCoordinateAxes(access, slot)))
        return original;
    for (Value dependency : {load.getValid(), load.getFill()})
      if (dependency)
        if (auto type = dyn_cast<FragmentType>(dependency.getType()))
          if (!isUniform(dependency, queryAxisProjection(
                                         type, cast<FragmentType>(
                                                   load.getResult().getType()))))
            return original;
    return authority;
  };

  auto alignPairs = [&](Value lhs, Value rhs, ArrayRef<int64_t> lhsAxes,
                        ArrayRef<int64_t> rhsAxes,
                        bool batch) -> LogicalResult {
    if (lhsAxes.size() != rhsAxes.size())
      return failure();
    for (auto [lhsAxis, rhsAxis] : llvm::zip(lhsAxes, rhsAxes)) {
      auto lhsType = cast<FragmentType>(lhs.getType());
      auto rhsType = cast<FragmentType>(rhs.getType());
      if (lhsAxis < 0 || rhsAxis < 0 ||
          lhsAxis >= static_cast<int64_t>(lhsType.getShape().size()) ||
          rhsAxis >= static_cast<int64_t>(rhsType.getShape().size()))
        return failure();
      Attribute lhsExtent = lhsType.getShape()[lhsAxis];
      Attribute rhsExtent = rhsType.getShape()[rhsAxis];
      if (lhsExtent == rhsExtent)
        continue;
      bool lhsUnit = isUnit(lhsExtent);
      bool rhsUnit = isUnit(rhsExtent);
      bool rebindLhs = lhsUnit;
      bool lhsUniform = batch && isUniformBatch(lhs, lhsAxis);
      bool rhsUniform = batch && isUniformBatch(rhs, rhsAxis);
      if (lhsUnit == rhsUnit && lhsUniform == rhsUniform)
        return contract.emitOpError("ordinary contract paired axes have "
                                    "conflicting physical extents")
               << "; lhs_axis=" << lhsAxis << "; lhs_extent=" << lhsExtent
               << "; rhs_axis=" << rhsAxis << "; rhs_extent=" << rhsExtent;
      if (lhsUnit == rhsUnit)
        rebindLhs = lhsUniform;
      if (rebindLhs) {
        auto authority = batch ? batchExtentAuthority(lhs, lhsAxis)
                               : std::pair<Value, unsigned>{
                                     lhs, static_cast<unsigned>(lhsAxis)};
        if (failed(retargetFragmentAxisExtent(
                authority.first, authority.second,
                cast<PhysicalExprAttr>(rhsExtent), changes.typeChanged(),
                &changes)))
          return failure();
      } else {
        auto authority = batch ? batchExtentAuthority(rhs, rhsAxis)
                               : std::pair<Value, unsigned>{
                                     rhs, static_cast<unsigned>(rhsAxis)};
        if (failed(retargetFragmentAxisExtent(
                authority.first, authority.second,
                cast<PhysicalExprAttr>(lhsExtent), changes.typeChanged(),
                &changes)))
          return failure();
      }
    }
    return success();
  };
  if (failed(alignPairs(contract.getLhs(), contract.getRhs(),
                        contract.getLhsReductionAxes(),
                        contract.getRhsReductionAxes(), false)) ||
      failed(alignPairs(contract.getLhs(), contract.getRhs(),
                        contract.getLhsBatchAxes(), contract.getRhsBatchAxes(),
                        true)))
    return WalkResult::interrupt();
  return WalkResult::advance();
}

} // namespace intent::gpu::value_relations
