#include "Intent/Dialect/GPU/Transforms/Value/ExecutionSchema.h"
#include "Intent/Dialect/GPU/Analysis/Helpers.h"
#include "Intent/Dialect/GPU/IR/FragmentOpInterface.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::gpu {
namespace {

bool sameAxis(Attribute lhs, Attribute rhs) {
  auto first = cast<AxisMapAttr>(lhs), second = cast<AxisMapAttr>(rhs);
  return first.getSourceId() == second.getSourceId() &&
         first.getSourceAxis() == second.getSourceAxis() &&
         first.getDimensionId() == second.getDimensionId() &&
         first.getDerived() == second.getDerived();
}

Type elementType(Type type) {
  auto fragment = dyn_cast<FragmentType>(type);
  return fragment ? fragment.getElementType() : type;
}

FragmentType withElement(FragmentType schema, Type element) {
  return FragmentType::get(schema.getContext(), element, schema.getShape(),
                           schema.getAxisMaps(), schema.getValidity(), schema.getOwner());
}

// Equal-rank schema transport retains operation positions. Rank expansion
// embeds old occurrences one-to-one, even if an identity occurs repeatedly.
FailureOr<SmallVector<unsigned>> axisEmbedding(FragmentType before,
                                                FragmentType after) {
  unsigned oldRank = before.getShape().size(), newRank = after.getShape().size();
  if (oldRank > newRank) return failure();
  if (oldRank == newRank) return llvm::to_vector(llvm::seq<unsigned>(0, oldRank));
  // ExecutionSchema prefixes newly selected occurrences and retains every old
  // occurrence in its ordered suffix. Prefer that explicit positional relation:
  // the prefix may deliberately repeat an identity already in the suffix.
  unsigned prefix = newRank - oldRank;
  if (llvm::all_of(llvm::seq<unsigned>(0, oldRank), [&](unsigned axis) {
        return sameAxis(before.getAxisMaps()[axis], after.getAxisMaps()[prefix + axis]);
      })) {
    SmallVector<unsigned> mapping;
    for (unsigned axis = 0; axis < oldRank; ++axis) mapping.push_back(prefix + axis);
    return mapping;
  }
  // A separately selected permutation is also usable when each occurrence has
  // one unambiguous destination. Equal extents do not identify an occurrence.
  SmallVector<unsigned> mapping;
  SmallVector<bool> used(newRank, false);
  for (Attribute axis : before.getAxisMaps()) {
    std::optional<unsigned> match;
    for (unsigned current = 0; current < newRank; ++current)
      if (!used[current] && sameAxis(axis, after.getAxisMaps()[current])) {
        if (match) return failure();
        match = current;
      }
    if (!match) return failure();
    used[*match] = true;
    mapping.push_back(*match);
  }
  return mapping;
}

FailureOr<SmallVector<unsigned>> axisEmbedding(Type before, Type after) {
  if (auto fragment = dyn_cast<FragmentType>(before)) {
    auto target = dyn_cast<FragmentType>(after);
    return target ? axisEmbedding(fragment, target)
                  : FailureOr<SmallVector<unsigned>>(failure());
  }
  auto record = dyn_cast<RecordType>(before), target = dyn_cast<RecordType>(after);
  if (!record || !target || record.getFieldNames() != target.getFieldNames() ||
      record.getFieldTypes().size() != target.getFieldTypes().size())
    return failure();
  std::optional<SmallVector<unsigned>> result;
  for (auto [field, next] : llvm::zip(record.getFieldTypes(), target.getFieldTypes())) {
    auto mapping = axisEmbedding(cast<TypeAttr>(field).getValue(),
                                 cast<TypeAttr>(next).getValue());
    if (failed(mapping) || (result && *result != *mapping)) return failure();
    result = std::move(*mapping);
  }
  return result ? FailureOr<SmallVector<unsigned>>(*result)
                : FailureOr<SmallVector<unsigned>>(failure());
}

FailureOr<SmallVector<int64_t>> mapAxes(ArrayRef<int64_t> axes,
                                       ArrayRef<unsigned> embedding) {
  SmallVector<int64_t> result;
  for (int64_t axis : axes) {
    if (axis < 0 || static_cast<uint64_t>(axis) >= embedding.size()) return failure();
    result.push_back(embedding[axis]);
  }
  return result;
}

FailureOr<FragmentType> commonSchema(ValueRange operands) {
  FragmentType schema;
  for (Value operand : operands)
    if (auto fragment = dyn_cast<FragmentType>(operand.getType())) {
      if (!schema || fragment.getShape().size() > schema.getShape().size())
        schema = fragment;
      else if (fragment.getShape().size() == schema.getShape().size() &&
               (fragment.getShape() != schema.getShape() ||
                fragment.getAxisMaps() != schema.getAxisMaps() ||
                fragment.getOwner() != schema.getOwner() ||
                fragment.getValidity() != schema.getValidity()))
        return failure();
    }
  if (schema)
    for (Value operand : operands)
      if (auto fragment = dyn_cast<FragmentType>(operand.getType()); fragment &&
          !queryAxisProjection(fragment, withElement(schema, fragment.getElementType())).isExact())
        return failure();
  return schema;
}

} // namespace

FailureOr<LiftedFragmentSchema> ExecutionSchema::project(FragmentType original) const {
  if (!selectedAxes) return failure();
  LiftedFragmentSchema result;
  unsigned oldRank = original.getShape().size();
  SmallVector<bool> used(oldRank, false);
  SmallVector<std::optional<unsigned>> existing;
  SmallVector<Attribute> shape, axes;
  auto append = [&](Attribute extent, Attribute attribute) {
    auto axis = cast<AxisMapAttr>(attribute);
    shape.push_back(extent);
    axes.push_back(AxisMapAttr::get(original.getContext(), axis.getSourceId(),
        axis.getSourceAxis(), axis.getDimensionId(), axes.size(), axis.getDerived()));
  };
  for (auto [index, axis] : llvm::enumerate(selectedAxes.getAxisMaps())) {
    std::optional<unsigned> match;
    for (unsigned old = 0; old < oldRank; ++old)
      if (!used[old] && sameAxis(axis, original.getAxisMaps()[old])) {
        match = old;
        used[old] = true;
        break;
      }
    existing.push_back(match);
    result.executionToNew.push_back(shape.size());
    if (!match) append(selectedAxes.getShape()[index], axis);
  }
  unsigned prefix = shape.size();
  for (auto [index, axis] : llvm::enumerate(original.getAxisMaps())) {
    result.oldToNew.push_back(shape.size());
    append(original.getShape()[index], axis);
  }
  for (auto [index, old] : llvm::enumerate(existing))
    if (old) result.executionToNew[index] = prefix + *old;
  result.type = FragmentType::get(original.getContext(), original.getElementType(),
      ArrayAttr::get(original.getContext(), shape), ArrayAttr::get(original.getContext(), axes),
      original.getValidity(), original.getOwner());
  return result;
}

FailureOr<Type> ExecutionSchema::lift(Type original) const {
  if (!selectedAxes) return failure();
  if (auto fragment = dyn_cast<FragmentType>(original)) {
    auto projected = project(fragment);
    return succeeded(projected) ? FailureOr<Type>(projected->type) : FailureOr<Type>(failure());
  }
  if (auto record = dyn_cast<RecordType>(original)) {
    SmallVector<Attribute> fields;
    for (Attribute field : record.getFieldTypes()) {
      auto type = lift(cast<TypeAttr>(field).getValue());
      if (failed(type)) return failure();
      fields.push_back(TypeAttr::get(*type));
    }
    return Type(RecordType::get(original.getContext(), record.getFieldNames(),
                               ArrayAttr::get(original.getContext(), fields), record.getOwner()));
  }
  if (isa<IntegerType, IndexType, FloatType>(original))
    return Type(withElement(selectedAxes, original));
  return original;
}

FailureOr<SmallVector<Type>> inferElementwiseSchema(Operation *source,
                                                   ValueRange operands) {
  if (source->getNumOperands() != operands.size()) return failure();
  if (auto record = dyn_cast<MakeRecordOp>(source)) {
    SmallVector<Attribute> fields;
    for (Value value : operands) fields.push_back(TypeAttr::get(value.getType()));
    auto original = record.getResult().getType();
    return SmallVector<Type>{RecordType::get(source->getContext(), original.getFieldNames(),
        ArrayAttr::get(source->getContext(), fields), original.getOwner())};
  }
  if (auto extract = dyn_cast<ExtractOp>(source)) {
    auto record = dyn_cast<RecordType>(operands.front().getType());
    if (!record || extract.getField() >= record.getFieldTypes().size()) return failure();
    return SmallVector<Type>{cast<TypeAttr>(record.getFieldTypes()[extract.getField()]).getValue()};
  }
  if (isa<arith::ConstantOp>(source)) return llvm::to_vector(source->getResultTypes());
  if (!isLaneWisePointwiseOperation(source) || source->getNumResults() != 1) return failure();
  auto schema = commonSchema(operands);
  if (failed(schema)) return failure();
  Type type = elementType(source->getResult(0).getType());
  return SmallVector<Type>{*schema ? Type(withElement(*schema, type)) : type};
}

LogicalResult remapSchemaAxes(Operation *operation, TypeRange originalOperands,
                              TypeRange originalResults) {
  if (originalOperands.size() != operation->getNumOperands() ||
      originalResults.size() != operation->getNumResults())
    return operation->emitOpError("schema mutation lost its original operand/result slots");
  auto embedding = [&](unsigned operand) {
    return axisEmbedding(originalOperands[operand], operation->getOperand(operand).getType());
  };
  auto remap = [&](unsigned operand, ArrayRef<int64_t> axes) -> FailureOr<SmallVector<int64_t>> {
    auto map = embedding(operand);
    return succeeded(map) ? mapAxes(axes, *map) : FailureOr<SmallVector<int64_t>>(failure());
  };
  auto context = operation->getContext();
  if (auto reduce = dyn_cast<ReduceOp>(operation)) {
    auto axes = remap(0, reduce.getAxes());
    if (failed(axes)) return reduce.emitOpError("cannot embed reduction axes in selected schema");
    for (unsigned source = 1; source < reduce.getSources().size(); ++source) {
      auto other = remap(source, reduce.getAxes());
      if (failed(other) || *other != *axes)
        return reduce.emitOpError("reduction components disagree on selected axis occurrences");
    }
    reduce.setAxesAttr(DenseI64ArrayAttr::get(context, *axes));
  } else if (isa<ScanOp, RegionFoldOp, RegionScanOp>(operation)) {
    auto axis = operation->getAttrOfType<IntegerAttr>("axis");
    auto axes = remap(0, {axis.getInt()});
    if (failed(axes)) return operation->emitOpError("cannot embed structured iteration axis");
    operation->setAttr("axis", IntegerAttr::get(axis.getType(), axes->front()));
  } else if (auto gather = dyn_cast<GatherOp>(operation)) {
    auto axes = remap(0, gather.getSourceAxes());
    if (failed(axes)) return gather.emitOpError("cannot embed gathered source axes");
    gather.setSourceAxesAttr(DenseI64ArrayAttr::get(context, *axes));
  } else if (auto transpose = dyn_cast<TransposeOp>(operation)) {
    auto input = embedding(0);
    auto output = axisEmbedding(originalResults[0], transpose.getType());
    if (failed(input) || failed(output)) return transpose.emitOpError("cannot embed transpose axes");
    SmallVector<int64_t> permutation(transpose.getType().getShape().size(), -1);
    for (auto [old, from] : llvm::enumerate(transpose.getPermutation())) {
      if (old >= output->size() || from < 0 || static_cast<uint64_t>(from) >= input->size())
        return failure();
      permutation[(*output)[old]] = (*input)[from];
    }
    SmallVector<bool> used(transpose.getValue().getType().getShape().size(), false);
    for (int64_t axis : permutation) if (axis >= 0) used[axis] = true;
    for (auto [axis, source] : llvm::enumerate(permutation)) {
      if (source >= 0) continue;
      for (unsigned candidate = 0; candidate < used.size(); ++candidate)
        if (!used[candidate] && sameAxis(transpose.getType().getAxisMaps()[axis],
                                        transpose.getValue().getType().getAxisMaps()[candidate])) {
          source = candidate;
          used[candidate] = true;
          break;
        }
      if (source < 0) return transpose.emitOpError("new transpose axis has no input occurrence");
    }
    transpose.setPermutationAttr(DenseI64ArrayAttr::get(context, permutation));
  } else if (auto reshape = dyn_cast<ReshapeOp>(operation)) {
    auto relations = queryFragmentOperandRelations(operation, originalOperands, originalResults[0]);
    auto input = embedding(0);
    auto output = axisEmbedding(originalResults[0], reshape.getType());
    if (failed(relations) || failed(input) || failed(output))
      return reshape.emitOpError("cannot transport original reshape reassociation");
    auto inputType = cast<FragmentType>(reshape.getValue().getType());
    auto outputType = cast<FragmentType>(reshape.getType());
    SmallVector<Attribute> groups;
    SmallVector<unsigned> newInputs, newOutputs;
    for (unsigned axis = 0; axis < inputType.getShape().size(); ++axis)
      if (!llvm::is_contained(*input, axis)) newInputs.push_back(axis);
    for (unsigned axis = 0; axis < outputType.getShape().size(); ++axis)
      if (!llvm::is_contained(*output, axis)) newOutputs.push_back(axis);
    if (newInputs.size() != newOutputs.size()) return failure();
    for (auto [from, to] : llvm::zip(newInputs, newOutputs)) {
      if (!sameAxis(inputType.getAxisMaps()[from], outputType.getAxisMaps()[to])) return failure();
      groups.push_back(ReshapeGroupAttr::get(context, DenseI64ArrayAttr::get(context, {from}),
                                           DenseI64ArrayAttr::get(context, {to})));
    }
    for (const auto &group : relations->front().groups) {
      SmallVector<int64_t> from, to;
      for (unsigned axis : group.sourceAxes) from.push_back((*input)[axis]);
      for (unsigned axis : group.resultAxes) to.push_back((*output)[axis]);
      groups.push_back(ReshapeGroupAttr::get(context, DenseI64ArrayAttr::get(context, from),
                                           DenseI64ArrayAttr::get(context, to)));
    }
    reshape.setReassociationAttr(ArrayAttr::get(context, groups));
  } else if (auto contract = dyn_cast<ContractOp>(operation)) {
    auto lhs = embedding(0), rhs = embedding(1);
    if (failed(lhs) || failed(rhs)) return contract.emitOpError("cannot embed contraction axes");
    auto lr = mapAxes(contract.getLhsReductionAxes(), *lhs);
    auto rr = mapAxes(contract.getRhsReductionAxes(), *rhs);
    auto lb = mapAxes(contract.getLhsBatchAxes(), *lhs);
    auto rb = mapAxes(contract.getRhsBatchAxes(), *rhs);
    if (failed(lr) || failed(rr) || failed(lb) || failed(rb)) return failure();
    SmallVector<int64_t> leftBatch, rightBatch;
    auto left = contract.getLhs().getType(), right = contract.getRhs().getType();
    for (unsigned axis = 0; axis < left.getShape().size(); ++axis) {
      if (llvm::is_contained(*lhs, axis)) continue;
      for (unsigned other = 0; other < right.getShape().size(); ++other) {
        if (llvm::is_contained(*rhs, other) || llvm::is_contained(rightBatch, other) ||
            !sameAxis(left.getAxisMaps()[axis], right.getAxisMaps()[other])) continue;
        leftBatch.push_back(axis);
        rightBatch.push_back(other);
        break;
      }
    }
    llvm::append_range(leftBatch, *lb);
    llvm::append_range(rightBatch, *rb);
    contract.setLhsReductionAxesAttr(DenseI64ArrayAttr::get(context, *lr));
    contract.setRhsReductionAxesAttr(DenseI64ArrayAttr::get(context, *rr));
    contract.setLhsBatchAxesAttr(DenseI64ArrayAttr::get(context, leftBatch));
    contract.setRhsBatchAxesAttr(DenseI64ArrayAttr::get(context, rightBatch));
  } else if (auto join = dyn_cast<JoinOp>(operation)) {
    auto source = cast<FragmentType>(join.getLhs().getType());
    join.setAxis(source.getShape().size());
  }
  return success();
}

} // namespace intent::gpu
