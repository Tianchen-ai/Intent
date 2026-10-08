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

static FailureOr<ArrayAttr> inferReshapeReassociationImpl(
    MLIRContext *context, ArrayRef<Attribute> sourceShape,
    ArrayRef<Attribute> resultShape, ArrayRef<Attribute> sourceMappings,
    ArrayRef<Attribute> resultMappings) {
  unsigned sourceRank = sourceShape.size();
  unsigned resultRank = resultShape.size();
  unsigned sourceBegin = 0;
  unsigned resultBegin = 0;
  SmallVector<Attribute> groups;
  if (sourceShape == resultShape) {
    for (unsigned axis = 0; axis < sourceRank; ++axis) {
      auto axes = DenseI64ArrayAttr::get(context, {static_cast<int64_t>(axis)});
      groups.push_back(ReshapeGroupAttr::get(context, axes, axes));
    }
    return ArrayAttr::get(context, groups);
  }
  auto relationScore = [&](unsigned sourceBegin, unsigned sourceEnd,
                           unsigned resultBegin, unsigned resultEnd) {
    if (sourceMappings.size() != sourceShape.size() ||
        resultMappings.size() != resultShape.size())
      return 0u;
    unsigned score = 0;
    for (unsigned resultAxis = resultBegin; resultAxis < resultEnd;
         ++resultAxis) {
      if (haveEqualPhysicalElementCounts(ArrayRef<Attribute>(),
                           resultShape.slice(resultAxis, 1)))
        continue;
      auto resultMap = dyn_cast<AxisMapAttr>(resultMappings[resultAxis]);
      if (!resultMap)
        continue;
      unsigned best = 0;
      for (unsigned sourceAxis = sourceBegin; sourceAxis < sourceEnd;
           ++sourceAxis) {
        auto sourceMap = dyn_cast<AxisMapAttr>(sourceMappings[sourceAxis]);
        if (!sourceMap)
          continue;
        if (sourceMap.getSourceId() == resultMap.getSourceId() &&
            sourceMap.getSourceAxis() == resultMap.getSourceAxis() &&
            sourceMap.getDerived() == resultMap.getDerived()) {
          best = 2;
          break;
        }
        if (sourceMap.getDimensionId() > 0 &&
            sourceMap.getDimensionId() == resultMap.getDimensionId())
          best = std::max(best, 1u);
      }
      score += best;
    }
    return score;
  };
  while (sourceBegin < sourceRank && resultBegin < resultRank) {
    bool sourceUnit = haveEqualPhysicalElementCounts(
        sourceShape.slice(sourceBegin, 1), ArrayRef<Attribute>());
    bool resultUnit = haveEqualPhysicalElementCounts(
        ArrayRef<Attribute>(), resultShape.slice(resultBegin, 1));
    // Unit axes are real row-major structure, not an ambiguous factor of the
    // neighboring dynamic extent.  Preserve insertion/removal explicitly so a
    // later physical extent refinement cannot turn `[N] -> [1, N]` into
    // `[N] -> [N, N]` merely because both result axes carry the same logical
    // dimension identity.
    if (resultUnit && !sourceUnit) {
      groups.push_back(ReshapeGroupAttr::get(
          context, DenseI64ArrayAttr::get(context, {}),
          DenseI64ArrayAttr::get(context,
                                 {static_cast<int64_t>(resultBegin)})));
      ++resultBegin;
      continue;
    }
    if (sourceUnit && !resultUnit) {
      groups.push_back(ReshapeGroupAttr::get(
          context,
          DenseI64ArrayAttr::get(context,
                                 {static_cast<int64_t>(sourceBegin)}),
          DenseI64ArrayAttr::get(context, {})));
      ++sourceBegin;
      continue;
    }
    std::optional<std::pair<unsigned, unsigned>> match;
    unsigned bestScore = 0;
    for (unsigned sourceEnd = sourceBegin + 1;
         sourceEnd <= sourceRank; ++sourceEnd) {
      for (unsigned resultEnd = resultBegin + 1;
           resultEnd <= resultRank; ++resultEnd) {
        if (!haveEqualPhysicalElementCounts(
                sourceShape.slice(sourceBegin, sourceEnd - sourceBegin),
                resultShape.slice(resultBegin, resultEnd - resultBegin)))
          continue;
        unsigned score =
            relationScore(sourceBegin, sourceEnd, resultBegin, resultEnd);
        if (!match || score > bestScore) {
          match = std::make_pair(sourceEnd, resultEnd);
          bestScore = score;
        }
      }
    }
    if (!match)
      return failure();
    SmallVector<int64_t> sourceAxes;
    SmallVector<int64_t> resultAxes;
    for (unsigned axis = sourceBegin; axis < match->first; ++axis)
      sourceAxes.push_back(axis);
    for (unsigned axis = resultBegin; axis < match->second; ++axis)
      resultAxes.push_back(axis);
    groups.push_back(ReshapeGroupAttr::get(
        context, DenseI64ArrayAttr::get(context, sourceAxes),
        DenseI64ArrayAttr::get(context, resultAxes)));
    sourceBegin = match->first;
    resultBegin = match->second;
  }
  if (sourceBegin < sourceRank) {
    ArrayRef<Attribute> remaining = sourceShape.drop_front(sourceBegin);
    if (!haveEqualPhysicalElementCounts(remaining, ArrayRef<Attribute>()))
      return failure();
    SmallVector<int64_t> sourceAxes;
    for (unsigned axis = sourceBegin; axis < sourceRank; ++axis)
      sourceAxes.push_back(axis);
    groups.push_back(ReshapeGroupAttr::get(
        context, DenseI64ArrayAttr::get(context, sourceAxes),
        DenseI64ArrayAttr::get(context, {})));
    sourceBegin = sourceRank;
  }
  if (resultBegin < resultRank) {
    ArrayRef<Attribute> remaining = resultShape.drop_front(resultBegin);
    if (!haveEqualPhysicalElementCounts(ArrayRef<Attribute>(), remaining))
      return failure();
    SmallVector<int64_t> resultAxes;
    for (unsigned axis = resultBegin; axis < resultRank; ++axis)
      resultAxes.push_back(axis);
    groups.push_back(ReshapeGroupAttr::get(
        context, DenseI64ArrayAttr::get(context, {}),
        DenseI64ArrayAttr::get(context, resultAxes)));
    resultBegin = resultRank;
  }
  if (sourceBegin != sourceRank || resultBegin != resultRank)
    return failure();
  return ArrayAttr::get(context, groups);
}

FailureOr<ArrayAttr>
inferReshapeReassociation(MLIRContext *context,
                          ArrayRef<Attribute> sourceShape,
                          ArrayRef<Attribute> resultShape) {
  return inferReshapeReassociationImpl(context, sourceShape, resultShape, {},
                                       {});
}

FailureOr<ArrayAttr>
inferReshapeReassociation(FragmentType source, FragmentType result,
                          unsigned sourcePrefix, unsigned resultPrefix) {
  if (!source || !result || sourcePrefix > source.getShape().size() ||
      resultPrefix > result.getShape().size())
    return failure();
  return inferReshapeReassociationImpl(
      source.getContext(), source.getShape().getValue().drop_front(sourcePrefix),
      result.getShape().getValue().drop_front(resultPrefix),
      source.getAxisMaps().getValue().drop_front(sourcePrefix),
      result.getAxisMaps().getValue().drop_front(resultPrefix));
}

LogicalResult ReshapeOp::verify() {
  auto source = dyn_cast<FragmentType>(getValue().getType());
  auto result = dyn_cast<FragmentType>(getResult().getType());
  if (!source || !result || source.getElementType() != result.getElementType() ||
      source.getOwner() != result.getOwner() || !getReassociation()) {
    return emitOpError("reshape physical schema is invalid: source=")
           << getValue().getType() << ", result=" << getResult().getType()
           << ", reassociation=" << getReassociation();
  }
  auto relations = queryFragmentOperandRelations(getOperation());
  if (failed(relations))
    return emitOpError(
        "reshape reassociation must cover consecutive axes and preserve the "
        "execution-prefix rank");
  const auto &relation = relations->front();
  for (const auto &group : relation.groups)
    if (group.kind == FragmentAxisRelationKind::Corresponding &&
        source.getAxisMaps()[group.sourceAxes[0]] !=
            result.getAxisMaps()[group.resultAxes[0]])
      return emitOpError("reshape changed a physical execution-prefix axis");
  if (!relation.hasCompatibleExtents())
    return emitOpError(
        "reshape reassociation group does not preserve row-major elements");
  return success();
}

LogicalResult TransposeOp::verify() {
  auto source = getValue().getType();
  auto result = getResult().getType();
  if (source.getShape().size() != result.getShape().size() ||
      getPermutation().size() != source.getShape().size() ||
      source.getElementType() != result.getElementType() ||
      source.getOwner() != result.getOwner())
    return emitOpError("transpose physical rank/type/owner is invalid");
  auto relations = queryFragmentOperandRelations(getOperation());
  if (failed(relations) || !relations->front().hasCompatibleExtents())
    return emitOpError("transpose permutation does not map physical extents");
  return success();
}

LogicalResult JoinOp::verify() {
  auto lhs = getLhs().getType();
  auto rhs = getRhs().getType();
  auto result = getResult().getType();
  if (lhs != rhs || getAxis() != lhs.getShape().size() ||
      result.getShape().size() != lhs.getShape().size() + 1 ||
      result.getElementType() != lhs.getElementType() ||
      result.getOwner() != lhs.getOwner())
    return emitOpError("join physical schema is invalid");
  for (unsigned axis = 0; axis < lhs.getShape().size(); ++axis)
    if (result.getShape()[axis] != lhs.getShape()[axis])
      return emitOpError("join changed a pre-existing physical extent");
  auto trailing = dyn_cast<PhysicalExprAttr>(
      result.getShape()[result.getShape().size() - 1]);
  if (!trailing ||
      trailing.getKind() != PhysicalExprKind::Constant ||
      trailing.getValue() != 2)
    return emitOpError("join trailing physical extent must be exactly two");
  return success();
}

} // namespace intent::gpu
