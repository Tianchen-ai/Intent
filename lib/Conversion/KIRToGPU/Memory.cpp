#include "Construction.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Control/Traversal.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;

namespace intent::kir_to_gpu {

LogicalResult ScalarRegionLowering::lower(intent::BufferOp buffer) {
  Operation *operation = buffer.getOperation();
  Location location = buffer.getLoc();
  auto logical = cast<intent::BufferType>(buffer.getResult().getType());
  auto tensor = cast<RankedTensorType>(logical.getTensor());
  SmallVector<Attribute> shape;
  for (unsigned axis = 0; axis < tensor.getRank(); ++axis) {
    auto extent = launchExtentExpression(
        canonicalAnalysis, buffer.getResult(), axis,
        operation->getParentOfType<func::FuncOp>());
    if (failed(extent))
      return buffer.emitOpError(
          "logical buffer allocation requires a launch-visible extent");
    shape.push_back(*extent);
  }
  LogicalBufferFact allocation = canonicalAnalysis.logicalBuffer(operation);
  if (!allocation.isExact())
    return buffer.emitOpError(
        "logical buffer has no exact canonical allocation fact");
  gpu::BufferScope scope =
      allocation.scope == LogicalBufferScope::ProgramPrivate
          ? gpu::BufferScope::ProgramPrivate
          : gpu::BufferScope::IterationPrivate;
  auto physicalType = gpu::BufferType::get(
      operation->getContext(), tensor.getElementType(), builder.getArrayAttr(shape),
      gpu::BufferScopeAttr::get(operation->getContext(), scope),
      allocation.instanceIdentity,
      /*owner=*/1,
      gpu::BufferInitializationAttr::get(
          operation->getContext(), gpu::BufferInitialization::FirstWrite),
      /*visibility=*/0);
  Value initial;
  if (buffer.getInitial()) {
    FailureOr<Value> lowered =
        get(buffer.getInitial());
    if (failed(lowered))
      return buffer.emitOpError("logical buffer initializer is unavailable");
    initial = *lowered;
  }
  // Materialize initialization as an ordered write in the shared program.
  if (initial && !isa<gpu::FragmentType>(initial.getType()) && !shape.empty()) {
    auto payload = convertTensorType(canonicalAnalysis, tensor, operation);
    if (failed(payload))
      return buffer.emitOpError("scalar initializer has no buffer shape");
    initial = builder.create<gpu::SplatOp>(location, *payload, initial);
  }
  if (initial)
    for (unsigned axis = 0; axis < shape.size(); ++axis)
      if (cast<gpu::FragmentType>(initial.getType()).getShape()[axis] !=
              shape[axis] &&
          failed(gpu::realizeFullCoverageDimension(physicalKernel, initial,
                                                  axis)))
        return buffer.emitOpError(
            "buffer initializer has no complete physical value");
  auto target = builder.create<gpu::BufferOp>(location, physicalType, Value());
  if (initial) {
    auto payload = dyn_cast<gpu::FragmentType>(initial.getType());
    SmallVector<Value> coordinates;
    SmallVector<int64_t> axes;
    Value valid;
    Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
    Value one = builder.create<arith::ConstantIndexOp>(location, 1);
    for (unsigned axis = 0; axis < shape.size(); ++axis) {
      auto mapping = cast<gpu::AxisMapAttr>(payload.getAxisMaps()[axis]);
      auto coordinateType = gpu::FragmentType::get(
          operation->getContext(), builder.getIndexType(),
          builder.getArrayAttr({payload.getShape()[axis]}),
          builder.getArrayAttr({gpu::AxisMapAttr::get(
              operation->getContext(), mapping.getSourceId(),
              mapping.getSourceAxis(), mapping.getDimensionId(), 0,
              mapping.getDerived())}),
          payload.getValidity(), payload.getOwner());
      Value stop = builder.create<gpu::PhysicalExprOp>(
          location, builder.getIndexType(), cast<PhysicalExprAttr>(shape[axis]));
      Value width = builder.create<gpu::PhysicalExprOp>(
          location, builder.getIndexType(),
          cast<PhysicalExprAttr>(payload.getShape()[axis]));
      Value coordinate = builder.create<gpu::MakeRangeOp>(
          location, coordinateType, zero, width, one, zero, stop,
          mapping.getSourceId(), mapping.getSourceAxis(), mapping.getDerived());
      coordinates.push_back(coordinate);
      axes.push_back(axis);
      auto predicateType = gpu::FragmentType::get(
          operation->getContext(), builder.getI1Type(),
          coordinateType.getShape(), coordinateType.getAxisMaps(),
          coordinateType.getValidity(), coordinateType.getOwner());
      Value end = builder.create<gpu::SplatOp>(location, coordinateType, stop);
      Value predicate = builder.create<gpu::CompareOp>(
          location, predicateType, coordinate, end, ComparePredicate::Lt);
      auto projected = gpu::projectPredicateToFragmentAxis(
          builder, location, predicate, payload, axis);
      if (failed(projected))
        return buffer.emitOpError("buffer initializer has no tail projection");
      valid = valid ? createBinary(builder, location, (*projected).getType(),
                                    valid, *projected, BinaryOperator::LogicalAnd)
                    : *projected;
    }
    auto initialized = builder.create<gpu::StoreOp>(
        location, target, coordinates, initial, valid, axes);
    attachOrigin(operation, initialized);
  }
  mapResults(operation, target);
  return success();
}

LogicalResult ScalarRegionLowering::lower(intent::GatherOp gather) {
  return lowerIndexedRead(gather);
}

LogicalResult ScalarRegionLowering::lowerIndexedRead(Operation *operation) {
  auto access = cast<IndexedAccessOpInterface>(operation);
  bool gather = isa<intent::GatherOp>(operation);
  Location location = operation->getLoc();
  FailureOr<Value> source = get(access.getAccessSource());
  SmallVector<int64_t> axes;
  FailureOr<SmallVector<Value>> coordinates = accessCoordinates(operation, axes);
  FailureOr<Type> result = failed(coordinates)
                               ? FailureOr<Type>(failure())
                               : accessResultType(
                                     operation, operation->getResult(0).getType(),
                                     *coordinates);
  if (failed(source))
    return operation->emitOpError("indexed read physical source is unavailable");
  if (failed(coordinates))
    return operation->emitOpError(
        "indexed read has no physical coordinate realization");
  if (failed(result))
    return operation->emitOpError(
        "indexed read result cannot preserve the physical coordinate relation");
  Value valid;
  Value fill;
  if (access.getAccessValidity()) {
    FailureOr<Value> lowered =
        get(access.getAccessValidity());
    if (failed(lowered))
      return failure();
    valid = *lowered;
  }
  if (access.getAccessFill()) {
    FailureOr<Value> lowered =
        get(access.getAccessFill());
    if (failed(lowered))
      return failure();
    fill = *lowered;
  }
  if (auto target = dyn_cast<gpu::FragmentType>(*result)) {
    for (Value *operand : {&valid, &fill}) {
      if (!*operand)
        continue;
      FailureOr<Value> aligned =
          projectAccessOperand(location, *operand, target);
      if (failed(aligned))
        return operation->emitOpError(
            "indexed read validity/fill cannot adopt its result relation");
      *operand = *aligned;
    }
  }
  FailureOr<Value> bounded = materializeAccessValidity(
      operation, *source, *coordinates, axes, *result, valid);
  if (failed(bounded))
    return operation->emitOpError(
        "indexed read bounds cannot be materialized in its result relation");
  valid = *bounded;
  if (valid && !fill) {
    FailureOr<Value> zero = zeroAccessFill(location, *result);
    if (failed(zero))
      return operation->emitOpError(
          "indexed read bounds validity has no typed fill");
    fill = *zero;
  }
  if (gather && coordinates->empty() && axes.empty() && source->getType() == *result) {
    Value target = *source;
    if (valid) {
      target = builder.create<gpu::SelectOp>(location, *result, valid,
                                              target, fill);
      attachOrigin(operation, target.getDefiningOp());
    }
    values[operation->getResult(0)] = target;
    return success();
  }
  if (auto scalarTensor = dyn_cast<gpu::FragmentType>(source->getType());
      gather && scalarTensor && scalarTensor.getShape().empty() &&
      coordinates->empty() && axes.empty()) {
    FailureOr<PhysicalAxisIdentity> identity = resultAxisIdentity(operation, 0, 0);
    if (failed(identity))
      return operation->emitOpError("scalar extraction has no physical value identity");
    auto singleton = fragmentType(
        operation->getContext(), scalarTensor.getElementType(),
        {expression(operation->getContext(), PhysicalExprKind::Constant, 1)},
        {*identity}, scalarTensor.getOwner());
    auto reassociation = builder.getArrayAttr({gpu::ReshapeGroupAttr::get(
        operation->getContext(), builder.getDenseI64ArrayAttr({}),
        builder.getDenseI64ArrayAttr({0}))});
    source = Value(builder.create<gpu::ReshapeOp>(
        location, singleton, *source, reassociation));
    coordinates->push_back(builder.create<arith::ConstantIndexOp>(location, 0));
    axes.push_back(0);
  }
  if (gather)
    mapResults(operation, builder.create<gpu::GatherOp>(
        location, *result, *source, *coordinates, valid, fill, axes));
  else
    mapResults(operation, builder.create<gpu::LoadOp>(
        location, *result, *source, *coordinates, valid, fill, axes));
  return success();
}

LogicalResult ScalarRegionLowering::lower(intent::ViewLoadOp load) {
  return lowerIndexedRead(load);
}

LogicalResult ScalarRegionLowering::lower(intent::BufferLoadOp load) {
  return lowerIndexedRead(load);
}

LogicalResult ScalarRegionLowering::lower(intent::ViewStoreOp store) {
  return lowerIndexedWrite(store);
}

LogicalResult ScalarRegionLowering::lowerIndexedWrite(Operation *operation) {
  auto access = cast<IndexedAccessOpInterface>(operation);
  Location location = operation->getLoc();
  FailureOr<Value> resource = get(access.getAccessSource());
  SmallVector<int64_t> axes;
  FailureOr<SmallVector<Value>> coordinates = accessCoordinates(operation, axes);
  FailureOr<Value> value =
      failed(coordinates)
          ? FailureOr<Value>(failure())
          : accessValue(operation,
                        access.getStoredValue(),
                        *coordinates);
  if (failed(resource))
    return operation->emitOpError("indexed write physical resource is unavailable");
  if (failed(coordinates))
    return operation->emitOpError(
        "indexed write has no physical coordinate realization");
  if (failed(value))
    return operation->emitOpError(
        "indexed write value cannot preserve its physical result relation");
  FailureOr<Value> valid = materializeAccessValidity(
      operation, *resource, *coordinates, axes, (*value).getType(), Value());
  if (failed(valid))
    return operation->emitOpError(
        "indexed write bounds cannot be materialized in its value relation");
  auto target = builder.create<gpu::StoreOp>(
      location, *resource, *coordinates, *value, *valid, axes);
  attachOrigin(operation, target);
  return success();
}

LogicalResult ScalarRegionLowering::lower(intent::BufferStoreOp store) {
  return lowerIndexedWrite(store);
}

LogicalResult ScalarRegionLowering::lower(intent::ScatterUniqueOp store) {
  return lowerIndexedWrite(store);
}

LogicalResult ScalarRegionLowering::lower(intent::ScatterReduceOp scatter) {
  Operation *operation = scatter.getOperation();
  Location location = scatter.getLoc();
  FailureOr<Value> resource = get(scatter.getSource());
  SmallVector<int64_t> axes;
  FailureOr<SmallVector<Value>> coordinates = accessCoordinates(operation, axes);
  FailureOr<Value> value =
      failed(coordinates)
          ? FailureOr<Value>(failure())
          : accessValue(operation,
                        scatter.getValue(),
                        *coordinates);
  if (failed(resource) || failed(coordinates) ||
      failed(value))
    return scatter.emitOpError(
        "scatter-reduce physical relation is unavailable");
  FailureOr<Value> valid = materializeAccessValidity(
      operation, *resource, *coordinates, axes, (*value).getType(), Value());
  if (failed(valid))
    return scatter.emitOpError(
        "scatter-reduce resource bounds cannot be materialized in its value relation");
  OperationState state(location, gpu::ScatterReduceOp::getOperationName());
  state.addOperands(*resource);
  state.addOperands(*coordinates);
  state.addOperands(*value);
  if (*valid)
    state.addOperands(*valid);
  state.addAttribute("source_axes", builder.getDenseI64ArrayAttr(axes));
  state.addAttribute(
      "sharing", gpu::AtomicSharingDomainAttr::get(
                     operation->getContext(),
                     isa<gpu::ViewType>((*resource).getType())
                         ? gpu::AtomicSharingDomain::KernelInvocation
                         : gpu::AtomicSharingDomain::ProgramInstance));
  state.addAttribute("operandSegmentSizes",
                     builder.getDenseI32ArrayAttr(
                         {1, static_cast<int32_t>(coordinates->size()), 1,
                          static_cast<int32_t>(static_cast<bool>(*valid))}));
  state.addRegion();
  Operation *raw = builder.create(state);
  auto target = cast<gpu::ScatterReduceOp>(raw);
  SmallVector<Type> arguments{(*value).getType(), (*value).getType()};
  if (failed(lowerPureRegion(scatter.getCombine(), target.getCombine(),
                             arguments)))
    return failure();
  attachOrigin(operation, raw);
  return success();
}

LogicalResult ScalarRegionLowering::lower(intent::AtomicLoadOp atomic) {
  Operation *operation = atomic.getOperation();
  Location location = atomic.getLoc();
  FailureOr<Value> resource = get(atomic.getSource());
  SmallVector<int64_t> axes;
  FailureOr<SmallVector<Value>> coordinates = accessCoordinates(operation, axes);
  FailureOr<Type> result =
      failed(coordinates)
          ? FailureOr<Type>(failure())
          : accessResultType(operation, atomic.getResult().getType(),
                             *coordinates);
  if (failed(resource) || failed(coordinates) ||
      failed(result))
    return atomic.emitOpError("atomic load address/result is unavailable");
  gpu::AtomicSharingDomain sharing =
      isa<gpu::ViewType>((*resource).getType())
          ? gpu::AtomicSharingDomain::KernelInvocation
          : gpu::AtomicSharingDomain::ProgramInstance;
  FailureOr<Value> valid = materializeAccessValidity(
      operation, *resource, *coordinates, axes, *result, Value());
  if (failed(valid))
    return atomic.emitOpError(
        "atomic-load resource bounds cannot be materialized in its result relation");
  auto target = builder.create<gpu::AtomicLoadOp>(
      location, *result, *resource, *coordinates, *valid,
      atomic.getOrdering(), sharing, axes);
  mapResults(operation, target);
  return success();
}

LogicalResult ScalarRegionLowering::lower(intent::AtomicStoreOp atomic) {
  Operation *operation = atomic.getOperation();
  Location location = atomic.getLoc();
  FailureOr<Value> resource = get(atomic.getSource());
  SmallVector<int64_t> axes;
  FailureOr<SmallVector<Value>> coordinates = accessCoordinates(operation, axes);
  FailureOr<Value> value =
      failed(coordinates)
          ? FailureOr<Value>(failure())
          : accessValue(operation,
                        atomic.getValue(),
                        *coordinates);
  if (failed(resource) || failed(coordinates) ||
      failed(value))
    return atomic.emitOpError("atomic store address/value is unavailable");
  gpu::AtomicSharingDomain sharing =
      isa<gpu::ViewType>((*resource).getType())
          ? gpu::AtomicSharingDomain::KernelInvocation
          : gpu::AtomicSharingDomain::ProgramInstance;
  FailureOr<Value> valid = materializeAccessValidity(
      operation, *resource, *coordinates, axes, (*value).getType(), Value());
  if (failed(valid))
    return atomic.emitOpError(
        "atomic-store resource bounds cannot be materialized in its value relation");
  auto target = builder.create<gpu::AtomicStoreOp>(
      location, *resource, *coordinates, *value, *valid,
      atomic.getOrdering(), sharing, axes);
  if (Attribute node = operation->getAttr("intent.node"))
    target->setAttr(gpu::originAttr, node);
  return success();
}

LogicalResult ScalarRegionLowering::lower(intent::AtomicRMWOp atomic) {
  Operation *operation = atomic.getOperation();
  Location location = atomic.getLoc();
  FailureOr<Value> resource = get(atomic.getSource());
  SmallVector<int64_t> axes;
  FailureOr<SmallVector<Value>> coordinates = accessCoordinates(operation, axes);
  FailureOr<Value> value =
      failed(coordinates)
          ? FailureOr<Value>(failure())
          : accessValue(operation,
                        atomic.getValue(),
                        *coordinates);
  FailureOr<Type> result = failed(value)
                               ? FailureOr<Type>(failure())
                               : FailureOr<Type>((*value).getType());
  if (failed(resource) || failed(coordinates) ||
      failed(value) || failed(result))
    return atomic.emitOpError("atomic RMW address/value is unavailable");
  gpu::AtomicSharingDomain sharing =
      isa<gpu::ViewType>((*resource).getType())
          ? gpu::AtomicSharingDomain::KernelInvocation
          : gpu::AtomicSharingDomain::ProgramInstance;
  FailureOr<Value> valid = materializeAccessValidity(
      operation, *resource, *coordinates, axes, (*value).getType(), Value());
  if (failed(valid))
    return atomic.emitOpError(
        "atomic-RMW resource bounds cannot be materialized in its value relation");
  auto target = builder.create<gpu::AtomicRMWOp>(
      location, *result, *resource, *coordinates, *value, *valid,
      atomic.getKind(), atomic.getOrdering(), sharing, axes);
  mapResults(operation, target);
  return success();
}

LogicalResult ScalarRegionLowering::lower(
    intent::AtomicCompareExchangeOp atomic) {
  Operation *operation = atomic.getOperation();
  Location location = atomic.getLoc();
  FailureOr<Value> resource = get(atomic.getSource());
  SmallVector<int64_t> axes;
  FailureOr<SmallVector<Value>> coordinates = accessCoordinates(operation, axes);
  FailureOr<Value> expected =
      failed(coordinates)
          ? FailureOr<Value>(failure())
          : accessValue(operation,
                        atomic.getExpected(),
                        *coordinates);
  FailureOr<Value> desired =
      failed(coordinates)
          ? FailureOr<Value>(failure())
          : accessValue(operation,
                        atomic.getDesired(),
                        *coordinates);
  FailureOr<Type> result =
      convertDataType(canonicalAnalysis, atomic.getResult().getType(), operation);
  if (failed(resource) || failed(coordinates) ||
      failed(expected) || failed(desired) || failed(result))
    return atomic.emitOpError("compare-exchange address/value is unavailable");
  gpu::AtomicSharingDomain sharing =
      isa<gpu::ViewType>((*resource).getType())
          ? gpu::AtomicSharingDomain::KernelInvocation
          : gpu::AtomicSharingDomain::ProgramInstance;
  FailureOr<Value> valid = materializeAccessValidity(
      operation, *resource, *coordinates, axes, (*expected).getType(),
      Value());
  if (failed(valid))
    return atomic.emitOpError(
        "compare-exchange resource bounds cannot be materialized in its value relation");
  auto target = builder.create<gpu::AtomicCompareExchangeOp>(
      location, *result, *resource, *coordinates, *expected, *desired,
      *valid, atomic.getOrdering(), sharing, axes);
  mapResults(operation, target);
  return success();
}

LogicalResult ScalarRegionLowering::lower(intent::AssumeInBoundsOp assumption) {
  Operation *operation = assumption.getOperation();
  Location location = assumption.getLoc();
  FailureOr<Value> index = get(assumption.getIndex());
  FailureOr<Value> resource = get(assumption.getView());
  if (failed(index) || failed(resource))
    return assumption.emitOpError(
        "in-bounds assertion lost its physical value or resource");
  index = asLogicalIndex(location, *index);
  if (failed(index))
    return assumption.emitOpError(
        "in-bounds assertion index has no physical logical-index schema");
  FailureOr<unsigned> axis = physicalResourceAxis(
      assumption.getView().getType(), (*resource).getType(),
      assumption.getAxis());
  if (failed(axis))
    return assumption.emitOpError(
        "in-bounds assertion axis has no physical resource mapping");
  auto target = builder.create<gpu::AssumeInBoundsOp>(
      location, *index, *resource, *axis);
  attachOrigin(operation, target);
  return success();
}

} // namespace intent::kir_to_gpu
