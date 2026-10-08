#include "RegionCloning.h"
#include "RegionSources.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Value/ExecutionSchema.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/ScopeExit.h"

using namespace mlir;

namespace intent::gpu::region {
namespace {

struct SliceProjection {
  std::optional<unsigned> axis;
  AxisMapAttr segmentMapping;
};

FailureOr<SliceProjection>
querySliceProjection(FragmentType fragment, ArrayRef<SliceRelation> relations,
                     Operation *producer) {
  SliceProjection selected;
  for (const SliceRelation &relation : relations) {
    PhysicalAxisProjection axis = queryFragmentAxis(fragment, relation.source);
    if (axis.state == PhysicalFactState::Ambiguous)
      return failure();
    if (!axis.isExact())
      continue;
    if (selected.axis &&
        (*selected.axis != axis.fragmentAxis ||
         selected.segmentMapping != relation.segmentMapping))
      return failure();
    selected = {axis.fragmentAxis, relation.segmentMapping};
  }
  if (!selected.axis)
    return selected;
  if (auto reshape = dyn_cast_or_null<ReshapeOp>(producer)) {
    bool introducedUnitAxis =
        isUnitExtent(fragment.getShape()[*selected.axis]) &&
        llvm::none_of(relations, [&](const SliceRelation &relation) {
          return !queryFragmentAxes(reshape.getValue().getType(), relation.source)
                      .empty();
        });
    if (introducedUnitAxis)
      return SliceProjection{};
  }
  return selected;
}

FailureOr<Type> slicedType(Type type, ArrayRef<SliceRelation> relations,
                          PhysicalExprAttr extent,
                          Operation *producer = nullptr) {
  if (auto fragment = dyn_cast<FragmentType>(type)) {
    auto selected = querySliceProjection(fragment, relations, producer);
    if (failed(selected))
      return failure();
    if (!selected->axis)
      return type;
    return Type(replaceSliceAxis(fragment, *selected->axis, extent,
                                selected->segmentMapping));
  }
  if (auto record = dyn_cast<RecordType>(type)) {
    SmallVector<Attribute> fields;
    for (Attribute field : record.getFieldTypes()) {
      FailureOr<Type> converted =
          slicedType(cast<TypeAttr>(field).getValue(), relations, extent);
      if (failed(converted))
        return failure();
      fields.push_back(TypeAttr::get(*converted));
    }
    return Type(RecordType::get(type.getContext(), record.getFieldNames(),
                                ArrayAttr::get(type.getContext(), fields),
                                record.getOwner()));
  }
  return type;
}

FailureOr<SmallVector<Value>> cloneSlicedOperation(
    OpBuilder &builder, Operation *source, IRMapping &mapping,
    ArrayRef<SliceRelation> relations, PhysicalExprAttr extent) {
  return cloneWithPhysicalSchema(
      builder, source, mapping,
      [&](Value original) -> Type {
        auto selected = slicedType(original.getType(), relations, extent,
                                   original.getDefiningOp());
        return succeeded(selected) ? *selected : Type{};
      },
      [&](OpOperand &operand, unsigned sourceAxis, OpResult result,
          unsigned resultAxis) {
        auto sourceType = dyn_cast<FragmentType>(operand.get().getType());
        auto resultType = dyn_cast<FragmentType>(result.getType());
        if (!sourceType || !resultType)
          return false;
        auto input = querySliceProjection(sourceType, relations,
                                           operand.get().getDefiningOp());
        auto output =
            querySliceProjection(resultType, relations, result.getOwner());
        // The selected member domain, not equal capacities, makes this a
        // refinement of the same slice on both sides of the projection.
        return succeeded(input) && succeeded(output) &&
               input->axis == sourceAxis && output->axis == resultAxis &&
               input->segmentMapping == output->segmentMapping;
      });
}

bool isScanOutputPureConsumer(Operation *operation) {
  return isPhysicalReplayNode(operation, PhysicalReplayScope::Coordinate,
                              /*allowAccesses=*/false);
}

} // namespace

LogicalResult collectScanOutputConsumers(RegionScanOp scan,
                                         SmallVectorImpl<Operation *> &ordered,
                                         std::string &reason) {
  llvm::SetVector<Operation *> closure;
  SmallVector<Operation *> worklist;
  for (Value output : scan.getEmittedResults())
    llvm::append_range(worklist, output.getUsers());
  while (!worklist.empty()) {
    Operation *operation = worklist.pop_back_val();
    if (!closure.insert(operation))
      continue;
    if (operation->getBlock() != scan->getBlock()) {
      reason = "region-scan output escapes its physical execution block";
      return failure();
    }
    if (isa<StoreOp>(operation))
      continue;
    if (!isScanOutputPureConsumer(operation)) {
      reason = "region-scan output consumer is not a slice-preserving pure graph ending in a store";
      return failure();
    }
    for (Value result : operation->getResults())
      llvm::append_range(worklist, result.getUsers());
  }
  if (closure.empty()) {
    reason = "region-scan output has no source-aligned physical assembly";
    return failure();
  }
  for (Operation &operation : *scan->getBlock())
    if (closure.contains(&operation))
      ordered.push_back(&operation);
  if (!llvm::any_of(ordered, [](Operation *operation) {
        return isa<StoreOp>(operation);
      })) {
    reason = "region-scan output assembly has no observable store";
    return failure();
  }
  return success();
}

namespace {

Value mappedValue(IRMapping &mapping, Value value) {
  if (!value)
    return {};
  if (Value mapped = mapping.lookupOrNull(value))
    return mapped;
  return value;
}

bool availableAtInsertionPoint(Value value, OpBuilder &builder,
                               DominanceInfo &dominance) {
  if (auto argument = dyn_cast<BlockArgument>(value))
    return dominance.dominates(argument.getOwner(), builder.getInsertionBlock());
  Operation *definition = value.getDefiningOp();
  return definition && dominance.properlyDominates(
      definition->getBlock(), definition->getIterator(),
      builder.getInsertionBlock(), builder.getInsertionPoint(),
      /*enclosingOk=*/false);
}

bool isPureScalarComputation(Operation *operation) {
  return operation->getNumRegions() == 0 && isPure(operation) &&
         llvm::all_of(operation->getResultTypes(), [](Type type) {
           return isa<IntegerType, IndexType, FloatType>(type);
         });
}

Value dominatingEquivalent(Value value, OpBuilder &builder, IRMapping &mapping,
                          DominanceInfo &dominance) {
  auto result = cast<OpResult>(value);
  Operation *definition = result.getOwner();
  SmallVector<Value> operands;
  for (Value operand : definition->getOperands())
    operands.push_back(mappedValue(mapping, operand));
  for (Block *block = builder.getInsertionBlock(); block;) {
    for (Operation &candidate : *block) {
      if (candidate.getName() != definition->getName() ||
          candidate.getNumResults() != definition->getNumResults() ||
          !llvm::equal(candidate.getOperands(), operands) ||
          !availableAtInsertionPoint(candidate.getResult(result.getResultNumber()),
                                     builder, dominance))
        continue;
      if (OperationEquivalence::isEquivalentTo(
              &candidate, definition,
              OperationEquivalence::ignoreValueEquivalence,
              nullptr, OperationEquivalence::IgnoreLocations))
        return candidate.getResult(result.getResultNumber());
    }
    Operation *parent = block->getParentOp();
    if (!parent || parent->hasTrait<OpTrait::IsIsolatedFromAbove>())
      break;
    block = parent->getBlock();
  }
  return {};
}

FailureOr<Value> materializeScanConsumerValue(
    OpBuilder &builder, Location location, RegionScanOp scan, Value value,
    IRMapping &mapping, ArrayRef<SliceRelation> relations,
    PhysicalExprAttr sliceExtent,
    Value offset, Value segment, DominanceInfo &dominance, std::string &reason) {
  if (!value)
    return Value();
  if (Value mapped = mapping.lookupOrNull(value)) {
    if (availableAtInsertionPoint(mapped, builder, dominance))
      return mapped;
    reason = "region-scan mapped output address does not dominate its use";
    return failure();
  }
  Operation *definition = value.getDefiningOp();
  bool available = availableAtInsertionPoint(value, builder, dominance);
  if (!definition || definition->getBlock() != scan->getBlock()) {
    if (available)
      return value;
    reason = "region-scan output address is outside the segment insertion scope";
    return failure();
  }
  bool changed = false;
  for (Value operand : definition->getOperands()) {
    FailureOr<Value> materialized = materializeScanConsumerValue(
        builder, location, scan, operand, mapping, relations, sliceExtent,
        offset, segment, dominance, reason);
    if (failed(materialized))
      return failure();
    changed |= *materialized != operand;
    if (!mapping.lookupOrNull(operand) && *materialized != operand)
      mapping.map(operand, *materialized);
  }
  if (auto range = dyn_cast<MakeRangeOp>(definition)) {
    auto type = cast<FragmentType>(range.getResult().getType());
    auto relation = llvm::find_if(relations, [&](const SliceRelation &candidate) {
      return sourceAxisIdentity(range) == candidate.source;
    });
    bool isSourceAxis = relation != relations.end();
    if (available && !isSourceAxis && !changed)
      return value;
    Value physicalStart = mappedValue(mapping, range.getStart());
    Value physicalExtent = mappedValue(mapping, range.getExtent());
    FragmentType physicalType = type;
    if (isSourceAxis) {
      physicalStart = builder.create<BinaryOp>(location, builder.getIndexType(),
                                               physicalStart, offset,
                                               BinaryOperator::Add);
      physicalExtent = segment;
      physicalType = replaceSliceAxis(type, 0, sliceExtent,
                                      relation->segmentMapping);
    }
    Value result = builder.create<MakeRangeOp>(
        location, physicalType, physicalStart, physicalExtent,
        mappedValue(mapping, range.getStep()),
        mappedValue(mapping, range.getLogicalStart()),
        mappedValue(mapping, range.getLogicalStop()),
        range.getSourceId(), range.getSourceAxis(), range.getDerived());
    mapping.map(value, result);
    return result;
  }
  FailureOr<Type> sliced =
      slicedType(value.getType(), relations, sliceExtent, definition);
  if (failed(sliced)) {
    reason = "region-scan output address has no unique segment relation";
    return failure();
  }
  if (available && !changed && *sliced == value.getType())
    return value;
  bool scalarComputation = isPureScalarComputation(definition);
  if (scalarComputation)
    if (Value existing = dominatingEquivalent(value, builder, mapping, dominance)) {
      mapping.map(value, existing);
      return existing;
    }
  if (!scalarComputation && !isa<arith::ConstantOp>(definition) &&
      !isScanOutputPureConsumer(definition)) {
    reason =
        "region-scan output address depends on an operation that cannot be moved into the segment loop: " +
        definition->getName().getStringRef().str();
    return failure();
  }
  auto cloned = cloneSlicedOperation(builder, definition, mapping, relations, sliceExtent);
  if (failed(cloned)) {
    reason = "region-scan output address cannot transport its sliced schema";
    return failure();
  }
  Value result = mapping.lookupOrNull(value);
  return result ? FailureOr<Value>(result) : FailureOr<Value>(failure());
}

} // namespace

LogicalResult cloneScanOutputConsumers(
    OpBuilder &builder, Location location, ArrayRef<Operation *> consumers,
    IRMapping &resultMapping, ArrayRef<SliceRelation> relations,
    PhysicalExprAttr sliceExtent,
    Value segmentTail, RegionScanOp scan, Value offset, Value segment,
    std::string &reason) {
  IRMapping mapping(resultMapping);
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
  DominanceInfo dominance(scan->getParentOfType<func::FuncOp>());
  for (Operation *operation : consumers) {
    for (Value operand : operation->getOperands()) {
      if (mapping.lookupOrNull(operand))
        continue;
      FailureOr<Value> materialized = materializeScanConsumerValue(
          builder, location, scan, operand, mapping, relations, sliceExtent,
          offset, segment, dominance, reason);
      if (failed(materialized))
        return failure();
      if (*materialized != operand && !mapping.lookupOrNull(operand))
        mapping.map(operand, *materialized);
    }
    if (auto store = dyn_cast<StoreOp>(operation)) {
      Value value = mappedValue(mapping, store.getValue());
      auto fragment = dyn_cast<FragmentType>(value.getType());
      if (!fragment) {
        reason = "region-scan slice store value is not a fragment";
        return failure();
      }
      Value valid = mappedValue(mapping, store.getValid());
      Value tail = segmentTail;
      FragmentType predicate = predicateType(fragment);
      if (tail.getType() != predicate)
        tail = builder.create<BroadcastOp>(location, predicate, tail);
      if (valid) {
        if (valid.getType() != predicate)
          valid = builder.create<BroadcastOp>(location, predicate, valid);
        valid = builder.create<BinaryOp>(location, predicate, valid, tail,
                                         BinaryOperator::LogicalAnd);
      } else {
        valid = tail;
      }
      SmallVector<Value> coordinates;
      for (Value coordinate : store.getCoordinates())
        coordinates.push_back(mappedValue(mapping, coordinate));
      auto replacement = builder.create<StoreOp>(
          location, mappedValue(mapping, store.getResource()), coordinates,
          value, valid, store.getSourceAxes());
      if (Attribute origin = store->getAttr(originAttr))
        replacement->setAttr(originAttr, origin);
      continue;
    }
    if (failed(cloneSlicedOperation(builder, operation, mapping, relations, sliceExtent))) {
      reason = "region-scan output consumer cannot transport its sliced schema";
      return failure();
    }
  }
  resultMapping = std::move(mapping);
  rollback.release();
  return success();
}

} // namespace intent::gpu::region
