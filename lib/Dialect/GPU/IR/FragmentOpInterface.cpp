#include "Intent/Dialect/GPU/IR/FragmentOpInterface.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/MathExtras.h"

#include "Intent/Dialect/GPU/IR/FragmentOpInterface.cpp.inc"

using namespace mlir;

namespace intent::gpu {
namespace {
using Relations = SmallVector<FragmentOperandRelation>;
using Kind = FragmentAxisRelationKind;

bool unit(Attribute extent) {
  auto expression = dyn_cast<PhysicalExprAttr>(extent);
  return expression && expression.getKind() == PhysicalExprKind::Constant &&
         expression.getValue() == 1;
}

SmallVector<Attribute> extents(FragmentType type, ArrayRef<unsigned> axes) {
  SmallVector<Attribute> values;
  for (unsigned axis : axes)
    values.push_back(type.getShape()[axis]);
  return values;
}

// Compare the existing ordered symbolic product; this does not commute source
// axes or introduce division as a way to infer a missing reshape dimension.
struct ExtentProduct {
  llvm::APInt constant = llvm::APInt(256, 1);
  SmallVector<PhysicalExprAttr> terms;
  bool valid = true;
};
void appendProduct(PhysicalExprAttr expression, ExtentProduct &product) {
  if (!product.valid)
    return;
  if (expression.getKind() == PhysicalExprKind::Multiply) {
    if (expression.getOperands().size() != 2) {
      product.valid = false;
      return;
    }
    for (Attribute child : expression.getOperands())
      appendProduct(cast<PhysicalExprAttr>(child), product);
  } else if (expression.getKind() == PhysicalExprKind::Constant) {
    if (expression.getValue() < 0) {
      product.valid = false;
      return;
    }
    product.constant *= llvm::APInt(256, expression.getValue());
  } else {
    product.terms.push_back(expression);
  }
}

PhysicalExprAttr product(MLIRContext *context, ArrayRef<Attribute> values) {
  auto result = PhysicalExprAttr::get(
      context, PhysicalExprKind::Constant, 1, StringAttr::get(context, ""),
      ArrayAttr::get(context, {}));
  for (Attribute value : values) {
    auto extent = cast<PhysicalExprAttr>(value);
    if (unit(result)) {
      result = extent;
      continue;
    }
    if (unit(extent))
      continue;
    if (result.getKind() == PhysicalExprKind::Constant &&
        extent.getKind() == PhysicalExprKind::Constant) {
      int64_t combined;
      if (!llvm::MulOverflow(result.getValue(), extent.getValue(), combined)) {
        result = PhysicalExprAttr::get(
            context, PhysicalExprKind::Constant, combined,
            StringAttr::get(context, ""), ArrayAttr::get(context, {}));
        continue;
      }
    }
    result = PhysicalExprAttr::get(
        context, PhysicalExprKind::Multiply, 0, StringAttr::get(context, ""),
        ArrayAttr::get(context, {result, extent}));
  }
  return result;
}

FailureOr<Relations> positional(TypeRange operands, Type result,
                               unsigned resultNumber, bool sourceSchema) {
  if (resultNumber != 0)
    return failure();
  auto output = dyn_cast<FragmentType>(result);
  Relations relations;
  for (auto [index, type] : llvm::enumerate(operands)) {
    FragmentOperandRelation relation{static_cast<unsigned>(index), type,
                                     result, {}, sourceSchema};
    auto source = dyn_cast<FragmentType>(type);
    if (source && output) {
      if (source.getShape().size() != output.getShape().size()) {
        if (!sourceSchema) {
          auto projection = queryAxisProjection(source, output);
          if (!projection.isExact())
            return failure();
          for (auto [axis, input] : llvm::enumerate(projection.targetToSource))
            relation.groups.push_back({
                input ? Kind::Corresponding : Kind::Broadcast,
                input ? SmallVector<unsigned>{*input} : SmallVector<unsigned>{},
                {static_cast<unsigned>(axis)}});
        }
      } else {
        for (unsigned axis = 0; axis < source.getShape().size(); ++axis)
          relation.groups.push_back({Kind::Corresponding, {axis}, {axis}});
      }
    } else if (source || output) {
      // A scalar is uniform with respect to every fragment coordinate. This
      // describes a relation during ownership rewriting, not op legality.
      if (source && !sourceSchema)
        return failure();
      if (output)
        for (unsigned axis = 0; axis < output.getShape().size(); ++axis)
          relation.groups.push_back({Kind::Broadcast, {}, {axis}});
    }
    relations.push_back(std::move(relation));
  }
  return relations;
}

FailureOr<Relations> broadcast(TypeRange operands, Type result,
                               unsigned resultNumber) {
  if (resultNumber != 0 || operands.size() != 1)
    return failure();
  auto output = dyn_cast<FragmentType>(result);
  if (!output)
    return failure();
  auto source = dyn_cast<FragmentType>(operands[0]);
  if (!source)
    return positional(operands, result, resultNumber, false);
  auto projection = queryAxisProjection(source, output);
  if (!projection.isExact())
    return failure();
  FragmentOperandRelation relation{0, source, output, {}};
  for (auto [axis, input] : llvm::enumerate(projection.targetToSource)) {
    FragmentAxisGroup group{Kind::Broadcast, {}, {static_cast<unsigned>(axis)}};
    if (input) {
      group.sourceAxes.push_back(*input);
      if (!unit(source.getShape()[*input]) ||
          source.getShape()[*input] == output.getShape()[axis])
        group.kind = Kind::Corresponding;
    }
    relation.groups.push_back(std::move(group));
  }
  return Relations{std::move(relation)};
}

Type withShape(FragmentType type, ArrayRef<Attribute> shape) {
  return FragmentType::get(type.getContext(), type.getElementType(),
                           ArrayAttr::get(type.getContext(), shape),
                           type.getAxisMaps(), type.getValidity(),
                           type.getOwner());
}
} // namespace

bool haveEqualPhysicalElementCounts(ArrayRef<Attribute> lhs,
                                    ArrayRef<Attribute> rhs) {
  ExtentProduct left, right;
  for (Attribute value : lhs)
    appendProduct(cast<PhysicalExprAttr>(value), left);
  for (Attribute value : rhs)
    appendProduct(cast<PhysicalExprAttr>(value), right);
  return left.valid && right.valid && left.constant == right.constant &&
         left.terms == right.terms;
}

const FragmentAxisGroup *
FragmentOperandRelation::groupForResultAxis(unsigned axis) const {
  for (const auto &group : groups)
    if (llvm::is_contained(group.resultAxes, axis))
      return &group;
  return nullptr;
}

std::optional<unsigned>
FragmentOperandRelation::correspondingSourceAxis(unsigned axis) const {
  const auto *group = groupForResultAxis(axis);
  if (!group || group->kind == Kind::Broadcast || group->sourceAxes.size() != 1 ||
      group->resultAxes.size() != 1)
    return std::nullopt;
  return group->sourceAxes.front();
}

bool FragmentOperandRelation::isIntroducedUnitAxis(unsigned axis) const {
  const auto *group = groupForResultAxis(axis);
  auto target = dyn_cast<FragmentType>(resultType);
  return group && group->sourceAxes.empty() && target &&
         unit(target.getShape()[axis]);
}

bool FragmentOperandRelation::isUnitAxisInsertion() const {
  auto source = dyn_cast<FragmentType>(sourceType);
  auto target = dyn_cast<FragmentType>(resultType);
  if (!source || !target || source.getShape().size() >= target.getShape().size())
    return false;
  unsigned nextSource = 0, nextResult = 0;
  for (const auto &group : groups) {
    if (group.resultAxes.empty())
      return false;
    for (unsigned axis : group.resultAxes)
      if (axis != nextResult++)
        return false;
    if (group.sourceAxes.empty()) {
      for (unsigned axis : group.resultAxes)
        if (!unit(target.getShape()[axis]))
          return false;
    } else if (group.sourceAxes.size() != 1 || group.resultAxes.size() != 1 ||
               group.sourceAxes[0] != nextSource++ ||
               source.getShape()[group.sourceAxes[0]] !=
                   target.getShape()[group.resultAxes[0]]) {
      return false;
    }
  }
  if (nextSource != source.getShape().size() ||
      nextResult != target.getShape().size())
    return false;
  auto projection = queryBroadcastProjection(source, target);
  if (!projection.isExact())
    return false;
  nextSource = 0;
  for (auto [axis, input] : llvm::enumerate(projection.targetToSource)) {
    if (input) {
      if (*input != nextSource++ ||
          source.getShape()[*input] != target.getShape()[axis])
        return false;
    } else if (!unit(target.getShape()[axis])) {
      return false;
    }
  }
  return nextSource == source.getShape().size();
}

bool FragmentOperandRelation::preservesNonUnitAxes() const {
  auto source = dyn_cast<FragmentType>(sourceType);
  auto target = dyn_cast<FragmentType>(resultType);
  if (!source || !target)
    return false;
  for (const auto &group : groups) {
    auto before = extents(source, group.sourceAxes);
    auto after = extents(target, group.resultAxes);
    llvm::erase_if(before, unit);
    llvm::erase_if(after, unit);
    if (before.size() > 1 || before != after)
      return false;
  }
  return true;
}

bool FragmentOperandRelation::hasCompatibleExtents() const {
  auto source = dyn_cast<FragmentType>(sourceType);
  auto target = dyn_cast<FragmentType>(resultType);
  if (!target)
    return !source;
  for (const auto &group : groups) {
    if (group.kind == Kind::Broadcast) {
      if (source && llvm::any_of(group.sourceAxes, [&](unsigned axis) {
            return !unit(source.getShape()[axis]);
          }))
        return false;
    } else if (group.kind == Kind::Corresponding) {
      if (!source || source.getShape()[group.sourceAxes[0]] !=
                         target.getShape()[group.resultAxes[0]])
        return false;
    } else {
      if (!source ||
          !haveEqualPhysicalElementCounts(extents(source, group.sourceAxes),
                                           extents(target, group.resultAxes)))
        return false;
    }
  }
  return true;
}

FailureOr<Relations>
queryFragmentOperandRelations(Operation *operation, TypeRange operandTypes,
                              Type resultType, unsigned resultNumber) {
  auto interface = dyn_cast_or_null<FragmentOpInterface>(operation);
  if (!interface || resultNumber >= operation->getNumResults() ||
      operandTypes.size() != operation->getNumOperands())
    return failure();
  return interface.getFragmentOperandRelations(resultNumber, operandTypes,
                                               resultType);
}

FailureOr<Relations> queryFragmentOperandRelations(Operation *operation,
                                                   unsigned resultNumber) {
  if (!operation || resultNumber >= operation->getNumResults())
    return failure();
  return queryFragmentOperandRelations(operation, operation->getOperandTypes(),
                                       operation->getResult(resultNumber).getType(),
                                       resultNumber);
}

FailureOr<Relations> queryFragmentOperandRelations(OpResult result) {
  return queryFragmentOperandRelations(result.getOwner(),
                                       result.getResultNumber());
}

#define POSITIONAL_RELATIONS(OP, SOURCE_SCHEMA)                                 \
  FailureOr<Relations> OP::getFragmentOperandRelations(                         \
      unsigned resultNumber, TypeRange operands, Type result) {                \
    return positional(operands, result, resultNumber, SOURCE_SCHEMA);           \
  }
POSITIONAL_RELATIONS(UnaryOp, true)
POSITIONAL_RELATIONS(CastOp, true)
POSITIONAL_RELATIONS(BitcastOp, true)
POSITIONAL_RELATIONS(BinaryOp, false)
POSITIONAL_RELATIONS(CompareOp, false)
POSITIONAL_RELATIONS(SelectOp, false)
#undef POSITIONAL_RELATIONS

FailureOr<Relations> RandomBitsOp::getFragmentOperandRelations(
    unsigned resultNumber, TypeRange operands, Type result) {
  if (operands.size() != 2)
    return failure();
  // The scalar seed does not define the counter's lane schema.
  auto relations = positional(operands.drop_front(), result, resultNumber, true);
  if (failed(relations))
    return failure();
  relations->front().operandNumber = 1;
  return relations;
}

FailureOr<Relations> JoinOp::getFragmentOperandRelations(
    unsigned resultNumber, TypeRange operands, Type result) {
  auto target = dyn_cast<FragmentType>(result);
  if (resultNumber != 0 || operands.size() != 2 || !target)
    return failure();
  Relations relations;
  for (auto [index, type] : llvm::enumerate(operands)) {
    auto source = dyn_cast<FragmentType>(type);
    if (!source || target.getShape().size() != source.getShape().size() + 1 ||
        getAxis() != source.getShape().size())
      return failure();
    FragmentOperandRelation relation{static_cast<unsigned>(index), source,
                                     target, {}};
    for (unsigned axis = 0; axis < source.getShape().size(); ++axis)
      relation.groups.push_back({Kind::Corresponding, {axis}, {axis}});
    relation.invariantResultAxes.push_back(source.getShape().size());
    relations.push_back(std::move(relation));
  }
  return relations;
}

FailureOr<Relations> SplatOp::getFragmentOperandRelations(
    unsigned resultNumber, TypeRange operands, Type result) {
  return broadcast(operands, result, resultNumber);
}
FailureOr<Relations> BroadcastOp::getFragmentOperandRelations(
    unsigned resultNumber, TypeRange operands, Type result) {
  return broadcast(operands, result, resultNumber);
}

FailureOr<Relations> TransposeOp::getFragmentOperandRelations(
    unsigned resultNumber, TypeRange operands, Type result) {
  if (resultNumber != 0 || operands.size() != 1)
    return failure();
  auto source = dyn_cast<FragmentType>(operands[0]);
  auto target = dyn_cast<FragmentType>(result);
  if (!source || !target || source.getShape().size() != target.getShape().size() ||
      getPermutation().size() != target.getShape().size())
    return failure();
  FragmentOperandRelation relation{0, source, target, {}};
  llvm::DenseSet<int64_t> seen;
  for (auto [axis, input] : llvm::enumerate(getPermutation())) {
    if (input < 0 || static_cast<size_t>(input) >= source.getShape().size() ||
        !seen.insert(input).second)
      return failure();
    relation.groups.push_back({Kind::Corresponding,
                               {static_cast<unsigned>(input)},
                               {static_cast<unsigned>(axis)}});
  }
  return Relations{std::move(relation)};
}

FailureOr<Relations> ReshapeOp::getFragmentOperandRelations(
    unsigned resultNumber, TypeRange operands, Type result) {
  if (resultNumber != 0 || operands.size() != 1)
    return failure();
  auto source = dyn_cast<FragmentType>(operands[0]);
  auto target = dyn_cast<FragmentType>(result);
  if (!source || !target)
    return failure();
  unsigned sourceRank = 0, resultRank = 0;
  for (Attribute attribute : getReassociation()) {
    auto group = dyn_cast<ReshapeGroupAttr>(attribute);
    if (!group)
      return failure();
    for (int64_t axis : group.getSourceAxes().asArrayRef())
      if (axis < 0 || static_cast<uint64_t>(axis) != sourceRank++)
        return failure();
    for (int64_t axis : group.getResultAxes().asArrayRef())
      if (axis < 0 || static_cast<uint64_t>(axis) != resultRank++)
        return failure();
  }
  if (sourceRank > source.getShape().size() ||
      resultRank > target.getShape().size())
    return failure();
  unsigned sourcePrefix = source.getShape().size() - sourceRank;
  unsigned resultPrefix = target.getShape().size() - resultRank;
  if (sourcePrefix != resultPrefix)
    return failure();
  FragmentOperandRelation relation{0, source, target, {}};
  for (unsigned axis = 0; axis < sourcePrefix; ++axis) {
    relation.groups.push_back({Kind::Corresponding, {axis}, {axis}});
  }
  for (Attribute attribute : getReassociation()) {
    auto group = cast<ReshapeGroupAttr>(attribute);
    FragmentAxisGroup projected{Kind::Reassociation, {}, {}};
    for (int64_t axis : group.getSourceAxes().asArrayRef())
      projected.sourceAxes.push_back(sourcePrefix + axis);
    for (int64_t axis : group.getResultAxes().asArrayRef())
      projected.resultAxes.push_back(resultPrefix + axis);
    relation.groups.push_back(std::move(projected));
  }
  return Relations{std::move(relation)};
}

FailureOr<Type>
transportFragmentResultType(ArrayRef<FragmentOperandRelation> relations,
                            TypeRange operandTypes, Type declaredResult,
                            FragmentBroadcastRefinement refineBroadcast) {
  auto target = dyn_cast<FragmentType>(declaredResult);
  if (relations.size() == 1 && relations.front().preservesSourceSchema) {
    const auto &relation = relations.front();
    if (relation.operandNumber >= operandTypes.size())
      return failure();
    auto source = dyn_cast<FragmentType>(operandTypes[relation.operandNumber]);
    Type element = target ? target.getElementType() : declaredResult;
    if (!source)
      return target ? FailureOr<Type>(failure())
                    : FailureOr<Type>(declaredResult);
    return Type(FragmentType::get(
        source.getContext(), element, source.getShape(), source.getAxisMaps(),
        source.getValidity(), source.getOwner()));
  }
  if (!target)
    return declaredResult;
  SmallVector<Attribute> shape(target.getShape().getValue());
  SmallVector<bool> assigned(shape.size(), false);
  auto assign = [&](unsigned axis, Attribute value) -> LogicalResult {
    if (axis >= shape.size() || (assigned[axis] && shape[axis] != value))
      return failure();
    shape[axis] = value;
    assigned[axis] = true;
    return success();
  };
  for (const auto &relation : relations) {
    if (relation.operandNumber >= operandTypes.size())
      return failure();
    auto source = dyn_cast<FragmentType>(operandTypes[relation.operandNumber]);
    auto previous = dyn_cast<FragmentType>(relation.sourceType);
    auto oldResult = dyn_cast<FragmentType>(relation.resultType);
    if (!oldResult || oldResult.getShape().size() != shape.size())
      return failure();
    for (unsigned axis : relation.invariantResultAxes)
      if (shape[axis] != oldResult.getShape()[axis])
        return failure();
    if (!previous) {
      if (source)
        return failure();
      continue;
    }
    if (!source || source.getShape().size() != previous.getShape().size())
      return failure();
    for (const auto &group : relation.groups) {
      if (group.kind == Kind::Broadcast) {
        if (llvm::all_of(group.sourceAxes, [&](unsigned axis) {
              return unit(source.getShape()[axis]);
            }))
          continue;
        if (group.sourceAxes.size() != 1 || group.resultAxes.size() != 1 ||
            !refineBroadcast)
          return failure();
        unsigned sourceAxis = group.sourceAxes.front();
        unsigned resultAxis = group.resultAxes.front();
        if (!refineBroadcast(relation.operandNumber, sourceAxis, resultAxis) ||
            source.getShape()[sourceAxis] != target.getShape()[resultAxis] ||
            failed(assign(resultAxis, source.getShape()[sourceAxis])))
          return failure();
        continue;
      }
      if (group.kind == Kind::Corresponding) {
        if (failed(assign(group.resultAxes[0],
                          source.getShape()[group.sourceAxes[0]])))
          return failure();
        continue;
      }
      auto inputs = extents(source, group.sourceAxes);
      if (group.resultAxes.size() == 1) {
        if (failed(assign(group.resultAxes[0],
                          product(target.getContext(), inputs))))
          return failure();
        continue;
      }
      SmallVector<unsigned> nonUnit;
      for (unsigned axis : group.resultAxes)
        if (!unit(shape[axis]))
          nonUnit.push_back(axis);
      if (group.sourceAxes.size() == 1 && nonUnit.size() == 1) {
        if (failed(assign(nonUnit.front(), inputs.front())))
          return failure();
      } else {
        SmallVector<Attribute> outputs;
        for (unsigned axis : group.resultAxes)
          outputs.push_back(shape[axis]);
        if (!haveEqualPhysicalElementCounts(inputs, outputs))
          return failure();
      }
    }
  }
  return withShape(target, shape);
}

FailureOr<Type>
transportFragmentOperandType(const FragmentOperandRelation &relation,
                             Type newResultType, Type declaredOperand) {
  auto target = dyn_cast<FragmentType>(newResultType);
  auto previous = dyn_cast<FragmentType>(relation.resultType);
  auto source = dyn_cast<FragmentType>(declaredOperand);
  if (!source)
    return declaredOperand;
  auto originalSource = dyn_cast<FragmentType>(relation.sourceType);
  if (!target || !previous || !originalSource ||
      source.getShape().size() != originalSource.getShape().size() ||
      target.getShape().size() != previous.getShape().size())
    return failure();
  SmallVector<Attribute> shape(source.getShape().getValue());
  for (unsigned axis : relation.invariantResultAxes)
    if (target.getShape()[axis] != previous.getShape()[axis])
      return failure();
  for (const auto &group : relation.groups) {
    if (group.kind == Kind::Broadcast)
      continue;
    if (group.sourceAxes.size() == 1 && group.resultAxes.size() == 1) {
      shape[group.sourceAxes[0]] = target.getShape()[group.resultAxes[0]];
      continue;
    }
    for (unsigned axis : group.resultAxes)
      if (target.getShape()[axis] != previous.getShape()[axis])
        return failure();
  }
  return withShape(source, shape);
}

} // namespace intent::gpu
