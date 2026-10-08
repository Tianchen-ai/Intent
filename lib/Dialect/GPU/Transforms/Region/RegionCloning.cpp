#include "RegionCloning.h"
#include "RegionSources.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Value/ExecutionSchema.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace intent::gpu::region {
namespace {

struct ExtentBinding {
  uint64_t sourceId;
  uint64_t sourceAxis;
  int64_t dimensionId;
  bool derived;
  Attribute extent;
  uint64_t actualSourceId;
  uint64_t actualSourceAxis;
  int64_t actualDimensionId;
  bool actualDerived;
};

const ExtentBinding *findBinding(ArrayRef<ExtentBinding> bindings,
                                 uint64_t sourceId, uint64_t sourceAxis,
                                 int64_t dimensionId, bool derived) {
  auto found = llvm::find_if(bindings, [&](const ExtentBinding &binding) {
    return binding.sourceId == sourceId &&
           binding.sourceAxis == sourceAxis &&
           binding.dimensionId == dimensionId && binding.derived == derived;
  });
  return found == bindings.end() ? nullptr : &*found;
}

FailureOr<SmallVector<std::optional<unsigned>>>
helperAxisProjection(FragmentType expected, FragmentType actual) {
  SmallVector<std::optional<unsigned>> actualAxes(expected.getShape().size());
  if (expected.getShape().size() == actual.getShape().size()) {
    // Formal/actual and original/clone pairs explicitly preserve positions.
    for (unsigned axis = 0; axis < actualAxes.size(); ++axis)
      actualAxes[axis] = axis;
    return actualAxes;
  }
  auto projection = queryAxisProjection(expected, actual);
  if (!projection.isExact()) return failure();
  for (auto [axis, input] : llvm::enumerate(projection.targetToSource)) {
    if (!input) continue;
    if (*input >= actualAxes.size() || actualAxes[*input]) return failure();
    actualAxes[*input] = axis;
  }
  return actualAxes;
}

LogicalResult collectExtentBindings(Type expected, Type actual,
                                    SmallVectorImpl<ExtentBinding> &bindings,
                                    std::string &reason,
                                    ArrayRef<unsigned> selectedAxes = {}) {
  if (auto expectedFragment = dyn_cast<FragmentType>(expected)) {
    auto actualFragment = dyn_cast<FragmentType>(actual);
    if (!actualFragment)
      return success();
    auto actualAxes = helperAxisProjection(expectedFragment, actualFragment);
    if (failed(actualAxes)) {
      reason = "helper schema refinement has no unique axis projection";
      return failure();
    }
    for (auto [expectedAxis, attribute] :
         llvm::enumerate(expectedFragment.getAxisMaps())) {
      if (!selectedAxes.empty() &&
          !llvm::is_contained(selectedAxes, expectedAxis))
        continue;
      auto expectedMap = cast<AxisMapAttr>(attribute);
      std::optional<unsigned> actualAxis = (*actualAxes)[expectedAxis];
      if (!actualAxis) {
        reason = "helper schema refinement dropped a selected source axis";
        return failure();
      }
      Attribute extent = actualFragment.getShape()[*actualAxis];
      auto actualMap = cast<AxisMapAttr>(
          actualFragment.getAxisMaps()[*actualAxis]);
      if (const ExtentBinding *existing =
              findBinding(bindings, expectedMap.getSourceId(),
                          expectedMap.getSourceAxis(),
                          expectedMap.getDimensionId(),
                          expectedMap.getDerived())) {
        if (existing->extent != extent ||
            existing->actualSourceId != actualMap.getSourceId() ||
            existing->actualSourceAxis != actualMap.getSourceAxis() ||
            existing->actualDimensionId != actualMap.getDimensionId() ||
            existing->actualDerived != actualMap.getDerived()) {
          reason.clear();
          llvm::raw_string_ostream message(reason);
          message << "helper axis has incompatible physical bindings; expected_axis="
                  << expectedAxis << "; expected_mapping=" << expectedMap
                  << "; expected_extent="
                  << expectedFragment.getShape()[expectedAxis]
                  << "; actual_axis=" << *actualAxis
                  << "; actual_mapping=" << actualMap
                  << "; actual_extent=" << extent
                  << "; previous_mapping=(" << existing->actualSourceId
                  << ", " << existing->actualSourceAxis << ", "
                  << existing->actualDimensionId << ", "
                  << existing->actualDerived << ")"
                  << "; previous_extent=" << existing->extent;
          return failure();
        }
        continue;
      }
      bindings.push_back({expectedMap.getSourceId(),
                          expectedMap.getSourceAxis(),
                          expectedMap.getDimensionId(), expectedMap.getDerived(),
                          extent,
                          actualMap.getSourceId(),
                          actualMap.getSourceAxis(),
                          actualMap.getDimensionId(), actualMap.getDerived()});
    }
    return success();
  }
  auto expectedRecord = dyn_cast<RecordType>(expected);
  auto actualRecord = dyn_cast<RecordType>(actual);
  if (!expectedRecord || !actualRecord ||
      expectedRecord.getFieldNames() != actualRecord.getFieldNames() ||
      expectedRecord.getFieldTypes().size() !=
          actualRecord.getFieldTypes().size())
    return success();
  for (auto [expectedField, actualField] :
       llvm::zip(expectedRecord.getFieldTypes(),
                 actualRecord.getFieldTypes()))
    if (failed(collectExtentBindings(cast<TypeAttr>(expectedField).getValue(),
                                     cast<TypeAttr>(actualField).getValue(),
                                     bindings, reason)))
      return failure();
  return success();
}

bool carriesAxis(Type type, AxisMapAttr expected) {
  auto fragment = dyn_cast<FragmentType>(type);
  return fragment && llvm::any_of(fragment.getAxisMaps(), [&](Attribute attribute) {
           auto mapping = cast<AxisMapAttr>(attribute);
           return mapping.getSourceId() == expected.getSourceId() &&
                  mapping.getSourceAxis() == expected.getSourceAxis() &&
                  mapping.getDerived() == expected.getDerived();
         });
}

const ExtentBinding *boundAxis(FragmentType fragment, unsigned axis,
                              ArrayRef<ExtentBinding> bindings,
                              Operation *producer) {
  auto mapping = cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
  const ExtentBinding *binding =
      findBinding(bindings, mapping.getSourceId(), mapping.getSourceAxis(),
                  mapping.getDimensionId(), mapping.getDerived());
  if (!binding)
    return nullptr;
  if (auto reshape = dyn_cast_or_null<ReshapeOp>(producer))
    if (isUnitExtent(fragment.getShape()[axis]) &&
        !carriesAxis(reshape.getValue().getType(), mapping))
      return nullptr;
  return binding;
}

bool sameBoundDomain(const ExtentBinding &source, const ExtentBinding &result) {
  return source.actualSourceId == result.actualSourceId &&
         source.actualSourceAxis == result.actualSourceAxis &&
         source.actualDimensionId == result.actualDimensionId &&
         source.actualDerived == result.actualDerived &&
         source.extent == result.extent;
}

Type bindPhysicalExtents(Type type, ArrayRef<ExtentBinding> bindings,
                         Operation *producer = nullptr) {
  if (auto fragment = dyn_cast<FragmentType>(type)) {
    SmallVector<Attribute> shape(fragment.getShape().begin(),
                                 fragment.getShape().end());
    SmallVector<Attribute> mappings(fragment.getAxisMaps().begin(),
                                    fragment.getAxisMaps().end());
    bool changed = false;
    for (auto [axis, attribute] : llvm::enumerate(fragment.getAxisMaps())) {
      auto mapping = cast<AxisMapAttr>(attribute);
      const ExtentBinding *binding =
          boundAxis(fragment, axis, bindings, producer);
      if (!binding)
        continue;
      if (shape[axis] != binding->extent) {
        shape[axis] = binding->extent;
        changed = true;
      }
      if (mapping.getSourceId() != binding->actualSourceId ||
          mapping.getSourceAxis() != binding->actualSourceAxis ||
          mapping.getDerived() != binding->actualDerived) {
        mappings[axis] = AxisMapAttr::get(
            type.getContext(), binding->actualSourceId,
            binding->actualSourceAxis, binding->actualDimensionId, axis,
            binding->actualDerived);
        changed = true;
      }
    }
    if (!changed)
      return type;
    return FragmentType::get(
        type.getContext(), fragment.getElementType(),
        ArrayAttr::get(type.getContext(), shape),
        ArrayAttr::get(type.getContext(), mappings),
        fragment.getValidity(), fragment.getOwner());
  }
  auto record = dyn_cast<RecordType>(type);
  if (!record)
    return type;
  SmallVector<Attribute> fields;
  bool changed = false;
  for (Attribute attribute : record.getFieldTypes()) {
    Type field = cast<TypeAttr>(attribute).getValue();
    Type replacement = bindPhysicalExtents(field, bindings);
    fields.push_back(TypeAttr::get(replacement));
    changed |= replacement != field;
  }
  return changed ? Type(RecordType::get(type.getContext(), record.getFieldNames(),
                                        ArrayAttr::get(type.getContext(), fields),
                                        record.getOwner()))
                 : type;
}

// Preserve extents learned from actual producers except where this helper
// invocation already supplied an explicit axis binding. Such a binding is a
// projection request, not permission to change the producer or its other uses.
Type boundProjectionType(Type original, Type actual,
                         ArrayRef<ExtentBinding> bindings,
                         Operation *producer) {
  if (auto source = dyn_cast<FragmentType>(original)) {
    auto target = dyn_cast<FragmentType>(actual);
    if (!target)
      return actual == source.getElementType()
          ? bindPhysicalExtents(source, bindings, producer) : Type();
    auto actualAxes = helperAxisProjection(source, target);
    if (failed(actualAxes)) return {};
    auto selected = cast<FragmentType>(
        bindPhysicalExtents(source, bindings, producer));
    SmallVector<Attribute> shape(target.getShape().getValue());
    SmallVector<Attribute> mappings(target.getAxisMaps().getValue());
    for (unsigned axis = 0; axis < source.getShape().size(); ++axis)
      if (boundAxis(source, axis, bindings, producer)) {
        if (!(*actualAxes)[axis]) return {};
        unsigned targetAxis = *(*actualAxes)[axis];
        shape[targetAxis] = selected.getShape()[axis];
        auto mapping = cast<AxisMapAttr>(selected.getAxisMaps()[axis]);
        mappings[targetAxis] = AxisMapAttr::get(
            actual.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
            mapping.getDimensionId(), targetAxis, mapping.getDerived());
      }
    return FragmentType::get(
        actual.getContext(), target.getElementType(),
        ArrayAttr::get(actual.getContext(), shape),
        ArrayAttr::get(actual.getContext(), mappings),
        target.getValidity(), target.getOwner());
  }
  auto source = dyn_cast<RecordType>(original);
  if (!source) return actual;
  auto target = dyn_cast<RecordType>(actual);
  if (!target || source.getFieldNames() != target.getFieldNames() ||
      source.getFieldTypes().size() != target.getFieldTypes().size())
    return {};
  SmallVector<Attribute> fields;
  for (auto [before, after] :
       llvm::zip(source.getFieldTypes(), target.getFieldTypes())) {
    Type field = boundProjectionType(cast<TypeAttr>(before).getValue(),
                                    cast<TypeAttr>(after).getValue(),
                                    bindings, producer);
    if (!field) return {};
    fields.push_back(TypeAttr::get(field));
  }
  return RecordType::get(actual.getContext(), target.getFieldNames(),
                         ArrayAttr::get(actual.getContext(), fields),
                         target.getOwner());
}

void bindClonedRanges(Operation *source, const IRMapping &mapping,
                      OpBuilder::InsertPoint insertion, Operation *previous) {
  source->walk([&](MakeRangeOp original) {
    Value mapped = mapping.lookupOrNull(original.getResult());
    auto range = mapped ? mapped.getDefiningOp<MakeRangeOp>() : MakeRangeOp{};
    if (!range || range == original) return;
    Operation *created = insertion.getBlock()->findAncestorOpInBlock(*range);
    if (!created || (previous && !previous->isBeforeInBlock(created)) ||
        (insertion.getPoint() != insertion.getBlock()->end() &&
         !created->isBeforeInBlock(&*insertion.getPoint()))) return;
    auto originalExtent =
        range.getExtent().getDefiningOp<arith::ConstantIndexOp>();
    bool coveredIntroducedUnitDomain =
        originalExtent && originalExtent.value() == 1 &&
        samePhysicalScalarExpression(range.getStart(), range.getLogicalStart()) &&
        samePhysicalScalarExpression(range.getExtent(), range.getLogicalStop());
    auto fragment = cast<FragmentType>(range.getResult().getType());
    auto extent = cast<PhysicalExprAttr>(fragment.getShape()[0]);
    OpBuilder builder(range);
    Value physicalExtent;
    if (extent.getKind() == PhysicalExprKind::Constant)
      physicalExtent = builder.create<arith::ConstantIndexOp>(
          range.getLoc(), extent.getValue());
    else
      physicalExtent = builder.create<PhysicalExprOp>(
          range.getLoc(), builder.getIndexType(), extent);
    range.getExtentMutable().assign(physicalExtent);
    if (coveredIntroducedUnitDomain)
      range->setOperand(4, physicalExtent);
  });
}

} // namespace

FailureOr<SmallVector<Value>> inlinePureRegion(OpBuilder &builder, Region &region,
                                               ValueRange arguments,
                                               std::string &reason,
                                               Value substituteSource,
                                               Value substituteTarget,
                                               Value conjunctSource,
                                               Value conjunctPredicate,
                                               Value additionalSource,
                                               Value additionalTarget,
                                               IRMapping *resultMapping) {
  if (region.empty() || !llvm::hasSingleElement(region) ||
      region.front().getNumArguments() != arguments.size()) {
    reason = "helper argument schema mismatch";
    return failure();
  }
  auto yield = dyn_cast<YieldOp>(region.front().getTerminator());
  if (!yield) {
    reason = "helper has no physical yield";
    return failure();
  }
  // Keep the invocation's published mapping and insertion block unchanged if
  // any later field projection or tail binding rejects the reconstructed helper.
  IRMapping mapping = resultMapping ? *resultMapping : IRMapping{};
  auto insertion = builder.saveInsertionPoint();
  Operation *previous = insertion.getPoint() == insertion.getBlock()->begin()
      ? nullptr : &*std::prev(insertion.getPoint());
  auto rollback = llvm::make_scope_exit([&] {
    auto end = insertion.getPoint();
    while (end != insertion.getBlock()->begin()) {
      Operation *operation = &*std::prev(end);
      if (operation == previous) break;
      operation->erase();
    }
    builder.restoreInsertionPoint(insertion);
  });
  SmallVector<ExtentBinding> extentBindings;
  for (auto [argument, value] :
       llvm::zip(region.front().getArguments(), arguments)) {
    if (failed(collectExtentBindings(argument.getType(), value.getType(),
                                     extentBindings, reason))) {
      reason = "helper argument " + std::to_string(argument.getArgNumber()) +
               ": " + reason;
      return failure();
    }
    mapping.map(argument, value);
  }
  auto conjoin = [&](Value value) -> FailureOr<Value> {
    auto target = dyn_cast<FragmentType>(value.getType());
    if (!target || !target.getElementType().isInteger(1)) {
      reason = "summary membership predicate is not a boolean fragment";
      return failure();
    }
    Value predicate = conjunctPredicate;
    auto source = dyn_cast<FragmentType>(predicate.getType());
    if (source && source.getShape().size() == 1 &&
        target.getShape().size() > 1) {
      std::optional<unsigned> targetAxis;
      for (auto [axis, extent] : llvm::enumerate(target.getShape())) {
        if (extent != source.getShape()[0])
          continue;
        if (targetAxis) {
          targetAxis.reset();
          break;
        }
        targetAxis = axis;
      }
      if (!targetAxis) {
        reason = "physical tail extent does not identify one summary membership axis";
        return failure();
      }
      auto targetMapping =
          cast<AxisMapAttr>(target.getAxisMaps()[*targetAxis]);
      auto projected = FragmentType::get(
          target.getContext(), source.getElementType(), source.getShape(),
          ArrayAttr::get(
              target.getContext(),
              {AxisMapAttr::get(target.getContext(),
                                targetMapping.getSourceId(),
                                targetMapping.getSourceAxis(),
                                targetMapping.getDimensionId(), 0,
                                targetMapping.getDerived())}),
          target.getValidity(), target.getOwner());
      if (predicate.getType() != projected)
        predicate = builder.create<BroadcastOp>(value.getLoc(), projected,
                                                predicate);
    }
    if (predicate.getType() != target)
      predicate = builder.create<BroadcastOp>(value.getLoc(), target, predicate);
    return Value(builder.create<BinaryOp>(value.getLoc(), target, value,
                                          predicate,
                                          BinaryOperator::LogicalAnd));
  };
  if (substituteSource && substituteTarget) {
    if (substituteSource == conjunctSource) {
      FailureOr<Value> combined = conjoin(substituteTarget);
      if (failed(combined))
        return failure();
      substituteTarget = *combined;
    }
    mapping.map(substituteSource, substituteTarget);
  }
  if (additionalSource && additionalTarget)
    mapping.map(additionalSource, additionalTarget);
  auto projectBound = [&](Value original, Value actual) -> FailureOr<Value> {
    Type target = boundProjectionType(original.getType(), actual.getType(),
                                     extentBindings, original.getDefiningOp());
    auto fail = [&](StringRef message) -> FailureOr<Value> {
      reason.clear();
      llvm::raw_string_ostream diagnostic(reason);
      diagnostic << message << "; source=";
      if (auto result = dyn_cast<OpResult>(original))
        diagnostic << result.getOwner()->getName() << " result "
                   << result.getResultNumber();
      else
        diagnostic << "argument " << cast<BlockArgument>(original).getArgNumber();
      diagnostic << "; original_type=" << original.getType()
                 << "; actual_type=" << actual.getType();
      if (target) diagnostic << "; target_type=" << target;
      return failure();
    };
    if (!target)
      return fail("helper replacement has no exact bound fragment or product projection");
    auto projected = projectPhysicalValueToSchema(
        builder, original.getLoc(), actual, target);
    if (failed(projected))
      return fail("helper replacement cannot adopt its invocation's bound schema");
    return projected;
  };
  for (BlockArgument argument : region.front().getArguments()) {
    auto projected = projectBound(argument, mapping.lookup(argument));
    if (failed(projected)) return failure();
    if (argument == conjunctSource &&
        !(argument == substituteSource && substituteTarget)) {
      projected = conjoin(*projected);
      if (failed(projected)) return failure();
    }
    mapping.map(argument, *projected);
  }
  for (Operation &operation : region.front().without_terminator()) {
    if (operation.getNumResults() == 1 &&
        mapping.lookupOrNull(operation.getResult(0))) {
      Value result = operation.getResult(0);
      auto replacement = projectBound(result, mapping.lookup(result));
      if (failed(replacement)) return failure();
      mapping.map(result, *replacement);
      continue;
    }
    SmallVector<Type> selectedTypes;
    for (Value result : operation.getResults())
      selectedTypes.push_back(bindPhysicalExtents(result.getType(), extentBindings,
                                                 &operation));
    auto cloneInsertion = builder.saveInsertionPoint();
    Operation *clonePrevious = cloneInsertion.getPoint() == cloneInsertion.getBlock()->begin()
        ? nullptr : &*std::prev(cloneInsertion.getPoint());
    auto cloned = cloneWithPhysicalSchema(
            builder, &operation, mapping,
            [&](Value original) {
              return bindPhysicalExtents(original.getType(), extentBindings,
                                         original.getDefiningOp());
            },
            [&](OpOperand &operand, unsigned sourceAxis, OpResult result,
                unsigned resultAxis) {
              auto sourceType = dyn_cast<FragmentType>(operand.get().getType());
              auto resultType = dyn_cast<FragmentType>(result.getType());
              if (!sourceType || !resultType)
                return false;
              const ExtentBinding *input =
                  boundAxis(sourceType, sourceAxis, extentBindings,
                            operand.get().getDefiningOp());
              const ExtentBinding *output =
                  boundAxis(resultType, resultAxis, extentBindings,
                            result.getOwner());
              return input && output && sameBoundDomain(*input, *output);
            });
    if (failed(cloned)) {
      reason = "helper operands cannot transport the declared fragment schema";
      return failure();
    }
    bindClonedRanges(&operation, mapping, cloneInsertion, clonePrevious);
    for (unsigned index = 0; index < operation.getNumResults(); ++index) {
      Value originalValue = operation.getResult(index);
      auto projected = projectBound(originalValue, (*cloned)[index]);
      if (failed(projected)) return failure();
      mapping.map(originalValue, *projected);
      auto original = dyn_cast<FragmentType>(operation.getResult(index).getType());
      auto selected = dyn_cast<FragmentType>(selectedTypes[index]);
      auto type = dyn_cast<FragmentType>(projected->getType());
      if (!original || !selected || !type || type == selected)
        continue;
      SmallVector<unsigned> changedAxes;
      bool changedRank = selected.getShape().size() != type.getShape().size();
      if (!changedRank)
        for (auto [axis, extent] : llvm::enumerate(selected.getShape()))
          if (extent != type.getShape()[axis] ||
              selected.getAxisMaps()[axis] != type.getAxisMaps()[axis])
            changedAxes.push_back(axis);
      if (changedRank || !changedAxes.empty()) {
        if (failed(collectExtentBindings(original, type, extentBindings, reason,
                                        changedAxes))) {
          reason = "cloned " + operation.getName().getStringRef().str() +
                   " result " + std::to_string(index) + ": " + reason;
          return failure();
        }
      }
    }
    for (auto [source, result] :
         llvm::zip(operation.getResults(), *cloned)) {
      if (source == conjunctSource) {
        Value mapped = mapping.lookupOrNull(source);
        FailureOr<Value> combined = conjoin(mapped ? mapped : result);
        if (failed(combined))
          return failure();
        mapping.map(source, *combined);
      } else if (!mapping.lookupOrNull(source)) {
        mapping.map(source, result);
      }
    }
  }
  SmallVector<Value> results;
  for (Value value : yield.getValues()) {
    Value mapped = mapping.lookupOrNull(value);
    if (!mapped) {
      reason = "helper yield is outside the cloned value graph";
      return failure();
    }
    auto projected = projectBound(value, mapped);
    if (failed(projected)) return failure();
    results.push_back(*projected);
  }
  if (resultMapping) *resultMapping = std::move(mapping);
  rollback.release();
  return results;
}

} // namespace intent::gpu::region
