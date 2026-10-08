#include "PredicationDetail.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Value/ExecutionSchema.h"
#include "Intent/Dialect/GPU/Transforms/Control/Predication.h"
#include "Intent/Dialect/GPU/Transforms/Value/SchemaMutation.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <functional>

using namespace mlir;

namespace intent::gpu {
namespace predication {

bool isFloatToIntegerCast(CastOp cast) {
  auto elementType = [](Type type) {
    auto fragment = dyn_cast<FragmentType>(type);
    return fragment ? fragment.getElementType() : type;
  };
  return isa<FloatType>(elementType(cast.getValue().getType())) &&
         elementType(cast.getType()).isIntOrIndex();
}

FailureOr<Value> selectScalarProduct(OpBuilder &builder, Location location,
                                     Value condition, Value lhs, Value rhs) {
  auto record = dyn_cast<RecordType>(lhs.getType());
  if (!record) {
    if (auto fragment = dyn_cast<FragmentType>(lhs.getType())) {
      auto predicate = FragmentType::get(
          fragment.getContext(), builder.getI1Type(), fragment.getShape(),
          fragment.getAxisMaps(), fragment.getValidity(), fragment.getOwner());
      auto projected = projectPhysicalValueToSchema(builder, location,
                                                     condition, predicate);
      if (failed(projected)) return failure();
      condition = *projected;
    }
    return Value(builder.create<SelectOp>(location, lhs.getType(), condition, lhs, rhs));
  }
  auto field = [&](Value value, unsigned index, Type type) -> Value {
    if (auto made = value.getDefiningOp<MakeRecordOp>())
      return made.getFields()[index];
    return builder.create<ExtractOp>(location, type, value, index);
  };
  SmallVector<Value> fields;
  for (auto [index, attribute] : llvm::enumerate(record.getFieldTypes())) {
    Type type = cast<TypeAttr>(attribute).getValue();
    auto selected = selectScalarProduct(builder, location, condition,
                                         field(lhs, index, type),
                                         field(rhs, index, type));
    if (failed(selected)) return failure();
    fields.push_back(*selected);
  }
  return Value(builder.create<MakeRecordOp>(location, record, fields));
}

Value anyActiveLane(OpBuilder &builder, Location location, Value predicate) {
  auto shape = dyn_cast<FragmentType>(predicate.getType());
  if (!shape)
    return predicate;
  Value identity = builder.create<arith::ConstantOp>(location,
                                                    builder.getBoolAttr(false));
  SmallVector<int64_t> axes;
  for (unsigned axis = 0; axis < shape.getShape().size(); ++axis)
    axes.push_back(axis);
  auto any = builder.create<ReduceOp>(
      location, ValueRange{predicate},
      ValueRange{identity}, ValueRange{}, axes);
  OpBuilder::InsertionGuard guard(builder);
  Block *combine = builder.createBlock(
      &any.getCombine(), {}, {builder.getI1Type(), builder.getI1Type()},
      {location, location});
  Value joined = builder.create<BinaryOp>(
      location, builder.getI1Type(), combine->getArgument(0),
      combine->getArgument(1), BinaryOperator::LogicalOr);
  builder.create<YieldOp>(location, joined);
  return any.getResult(0);
}

} // namespace predication

using namespace predication;

LogicalResult clonePredicatedScalarOperation(
    OpBuilder &builder, Operation *operation, IRMapping &mapping,
    Value predicate, FragmentType shape, bool nonemptyIterations) {
  Location location = operation->getLoc();
  ExecutionSchema schema(shape);
  auto resultType = [&](Type type) -> FailureOr<Type> {
    return shape ? schema.lift(type) : FailureOr<Type>(type);
  };
  std::function<FailureOr<bool>(Type)> hasIterationAxes =
      [&](Type type) -> FailureOr<bool> {
    if (!shape) return false;
    if (auto record = dyn_cast<RecordType>(type)) {
      for (Attribute field : record.getFieldTypes()) {
        auto present = hasIterationAxes(cast<TypeAttr>(field).getValue());
        if (failed(present)) return failure();
        if (*present) return true;
      }
      return false;
    }
    auto fragment = dyn_cast<FragmentType>(type);
    if (!fragment) return false;
    auto projected = schema.project(fragment);
    if (failed(projected)) return failure();
    return projected->type == fragment;
  };
  auto project = [&](Value value, Type target) {
    return projectPhysicalValueToSchema(builder, location, value, target);
  };
  auto lift = [&](Value value) -> FailureOr<Value> {
    auto target = resultType(value.getType());
    return failed(target) ? FailureOr<Value>(failure()) : project(value, *target);
  };
  auto mapped = [&](Value value) { return mapping.lookupOrDefault(value); };
  auto liftGroup = [&](ValueRange values) -> FailureOr<SmallVector<Value>> {
    SmallVector<Value> lifted;
    for (Value value : values) {
      auto replacement = lift(mapped(value));
      if (failed(replacement)) return failure();
      lifted.push_back(*replacement);
    }
    return lifted;
  };
  auto liftTypes = [&](TypeRange types) -> FailureOr<SmallVector<Type>> {
    SmallVector<Type> lifted;
    for (Type type : types) {
      auto replacement = resultType(type);
      if (failed(replacement)) return failure();
      lifted.push_back(*replacement);
    }
    return lifted;
  };
  auto maskedValidity = [&](Value valid, Type dataType) -> FailureOr<Value> {
    auto lifted = resultType(dataType);
    if (failed(lifted)) return failure();
    Type maskType = builder.getI1Type();
    if (auto fragment = dyn_cast<FragmentType>(*lifted))
      maskType = FragmentType::get(
          fragment.getContext(), builder.getI1Type(), fragment.getShape(),
          fragment.getAxisMaps(), fragment.getValidity(), fragment.getOwner());
    auto condition = project(predicate, maskType);
    if (failed(condition)) return failure();
    if (!valid) return *condition;
    auto original = project(mapped(valid), maskType);
    if (failed(original)) return failure();
    return Value(builder.create<BinaryOp>(location, maskType, *condition,
                                          *original, BinaryOperator::LogicalAnd));
  };
  auto fill = [&](Value value, Type type) -> FailureOr<Value> {
    if (value) {
      auto target = resultType(type);
      return failed(target) ? FailureOr<Value>(failure())
                            : project(mapped(value), *target);
    }
    auto fragment = dyn_cast<FragmentType>(type);
    Type element = fragment ? fragment.getElementType() : type;
    Value zero = builder.create<arith::ConstantOp>(
        location, builder.getZeroAttr(element));
    if (fragment) zero = builder.create<SplatOp>(location, fragment, zero);
    return lift(zero);
  };

  // A traversal can bind a member read to an already selected lane value.
  // Apply this read's own fill and current control predicate here, including
  // reads nested in scalar branches; the binding does not authorize a load.
  if (auto gather = dyn_cast<GatherOp>(operation))
    if (Value member = mapping.lookupOrNull(gather.getResult())) {
      auto type = resultType(gather.getType());
      auto valid = maskedValidity(gather.getValid(), gather.getType());
      auto inactive = fill(gather.getFill(), gather.getType());
      if (failed(type) || failed(valid) || failed(inactive)) return failure();
      auto selected = project(member, *type);
      if (failed(selected)) return failure();
      mapping.map(gather.getResult(), builder.create<SelectOp>(
          location, *type, *valid, *selected, *inactive).getResult());
      return success();
    }

  Operation *clone;
  if (auto loop = dyn_cast<scf::ForOp>(operation)) {
    auto initial = liftGroup(loop.getInitArgs());
    if (failed(initial)) return failure();
    auto result = builder.create<scf::ForOp>(
        location, mapped(loop.getLowerBound()), mapped(loop.getUpperBound()),
        mapped(loop.getStep()), *initial);
    result->setAttrs(loop->getAttrs());
    if (!result.getBody()->empty()) result.getBody()->back().erase();
    IRMapping bodyMapping(mapping);
    bodyMapping.map(loop.getInductionVar(), result.getInductionVar());
    bodyMapping.map(loop.getRegionIterArgs(), result.getRegionIterArgs());
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(result.getBody());
      for (Operation &nested : loop.getBody()->without_terminator())
        if (failed(clonePredicatedScalarOperation(
                builder, &nested, bodyMapping, predicate, shape,
                nonemptyIterations)))
          return failure();
      SmallVector<Value> yielded;
      for (auto [value, carried] :
           llvm::zip(loop.getBody()->getTerminator()->getOperands(),
                     result.getRegionIterArgs())) {
        auto lifted = lift(bodyMapping.lookupOrDefault(value));
        if (failed(lifted)) return failure();
        auto selected = selectScalarProduct(builder, location, predicate,
                                             *lifted, carried);
        if (failed(selected)) return failure();
        yielded.push_back(*selected);
      }
      builder.create<scf::YieldOp>(location, yielded);
    }
    clone = result;
  } else if (auto loop = dyn_cast<scf::WhileOp>(operation)) {
    auto lifted = liftGroup(loop.getInits());
    if (failed(lifted)) return failure();
    SmallVector<Value> initial = std::move(*lifted);
    if (shape) {
      auto active = lift(predicate);
      if (failed(active)) return failure();
      initial.push_back(*active);
    }
    SmallVector<Type> types;
    for (Value value : initial) types.push_back(value.getType());
    auto result = builder.create<scf::WhileOp>(location, types, initial);
    result->setAttrs(loop->getAttrs());
    OpBuilder::InsertionGuard guard(builder);
    Block *before = builder.createBlock(
        &result.getBefore(), {}, types, SmallVector<Location>(types.size(), location));
    IRMapping beforeMapping(mapping);
    beforeMapping.map(loop.getBeforeArguments(),
                      before->getArguments().take_front(loop.getNumResults()));
    Value active = shape ? Value(before->getArguments().back()) : predicate;
    for (Operation &nested : loop.getBefore().front().without_terminator())
      if (failed(clonePredicatedScalarOperation(
              builder, &nested, beforeMapping, active, shape)))
        return failure();
    auto condition = cast<scf::ConditionOp>(loop.getBefore().front().getTerminator());
    auto projected = project(beforeMapping.lookupOrDefault(condition.getCondition()),
                             active.getType());
    if (failed(projected)) return failure();
    Value nextActive = builder.create<BinaryOp>(
        location, active.getType(), active, *projected, BinaryOperator::LogicalAnd);
    Value anyActive = anyActiveLane(builder, location, nextActive);
    SmallVector<Value> forwarded(before->getArguments().take_front(loop.getNumResults()));
    if (shape) forwarded.push_back(nextActive);
    builder.create<scf::ConditionOp>(location, anyActive, forwarded);
    Block *after = builder.createBlock(
        &result.getAfter(), {}, types, SmallVector<Location>(types.size(), location));
    IRMapping afterMapping(mapping);
    afterMapping.map(loop.getAfterArguments(),
                     after->getArguments().take_front(loop.getNumResults()));
    active = shape ? Value(after->getArguments().back()) : predicate;
    for (Operation &nested : loop.getAfter().front().without_terminator())
      if (failed(clonePredicatedScalarOperation(
              builder, &nested, afterMapping, active, shape)))
        return failure();
    SmallVector<Value> yielded;
    for (auto [value, carried] :
         llvm::zip(loop.getAfter().front().getTerminator()->getOperands(),
                   after->getArguments().take_front(loop.getNumResults()))) {
      auto replacement = lift(afterMapping.lookupOrDefault(value));
      if (failed(replacement)) return failure();
      auto selected = selectScalarProduct(builder, location, active,
                                           *replacement, carried);
      if (failed(selected)) return failure();
      yielded.push_back(*selected);
    }
    if (shape) yielded.push_back(active);
    builder.create<scf::YieldOp>(location, yielded);
    clone = result;
  } else if (auto branch = dyn_cast<scf::IfOp>(operation)) {
    // An iteration-invariant condition stays lazy. Only the selected branch's
    // accesses receive the active iteration predicate.
    if (shape && mapped(branch.getCondition()).getType().isInteger(1)) {
      auto types = liftTypes(branch.getResultTypes());
      if (failed(types)) return failure();
      auto result = builder.create<scf::IfOp>(
          location, *types, mapped(branch.getCondition()),
          !branch.getElseRegion().empty());
      for (auto [original, region] :
           llvm::zip(branch->getRegions(), result->getRegions())) {
        if (original.empty()) continue;
        Block &body = region.front();
        if (!body.empty()) body.back().erase();
        IRMapping branchMapping(mapping);
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&body);
        for (Operation &nested : original.front().without_terminator())
          if (failed(clonePredicatedScalarOperation(
                  builder, &nested, branchMapping, predicate, shape,
                  nonemptyIterations)))
            return failure();
        SmallVector<Value> yielded;
        for (Value value : original.front().getTerminator()->getOperands()) {
          auto replacement = lift(branchMapping.lookupOrDefault(value));
          if (failed(replacement)) return failure();
          yielded.push_back(*replacement);
        }
        builder.create<scf::YieldOp>(location, yielded);
      }
      if (Attribute origin = operation->getAttr(originAttr))
        result->setAttr(originAttr, origin);
      mapping.map(branch.getResults(), result.getResults());
      return success();
    }
    // Complete chunks can have scalar true validity and lane-varying branches.
    auto activePredicate = lift(predicate);
    auto liftedCondition = lift(mapped(branch.getCondition()));
    if (failed(activePredicate) || failed(liftedCondition)) return failure();
    auto condition = project(*liftedCondition, (*activePredicate).getType());
    if (failed(condition)) return failure();
    Value zero = builder.create<arith::ConstantOp>(location, builder.getBoolAttr(false));
    if (auto fragment = dyn_cast<FragmentType>((*condition).getType()))
      zero = builder.create<SplatOp>(location, fragment, zero);
    Value inverse = builder.create<CompareOp>(
        location, (*condition).getType(), *condition, zero, ComparePredicate::Eq);
    auto cloneBranch = [&](Region &region, Value selected)
        -> FailureOr<SmallVector<Value>> {
      SmallVector<Value> results;
      if (region.empty()) return results;
      Value active = builder.create<BinaryOp>(
          location, (*activePredicate).getType(), *activePredicate, selected,
          BinaryOperator::LogicalAnd);
      IRMapping branchMapping(mapping);
      for (Operation &nested : region.front().without_terminator())
        // A branch may have no active iteration, even in a nonempty chunk.
        if (failed(clonePredicatedScalarOperation(
                builder, &nested, branchMapping, active, shape)))
          return failure();
      for (Value value : region.front().getTerminator()->getOperands()) {
        auto replacement = lift(branchMapping.lookupOrDefault(value));
        if (failed(replacement)) return failure();
        results.push_back(*replacement);
      }
      return results;
    };
    auto thenValues = cloneBranch(branch.getThenRegion(), *condition);
    auto elseValues = cloneBranch(branch.getElseRegion(), inverse);
    if (failed(thenValues) || failed(elseValues)) return failure();
    for (auto [result, lhs, rhs] :
         llvm::zip(branch.getResults(), *thenValues, *elseValues)) {
      auto selected = selectScalarProduct(builder, location, *condition, lhs, rhs);
      if (failed(selected)) return failure();
      mapping.map(result, *selected);
    }
    return success();
  } else if (auto reduce = dyn_cast<ReduceOp>(operation)) {
    auto sources = liftGroup(reduce.getSources());
    auto identities = liftGroup(reduce.getIdentities());
    auto captures = liftGroup(reduce.getCaptures());
    auto types = liftTypes(reduce.getResultTypes());
    auto arguments = liftTypes(reduce.getCombine().front().getArgumentTypes());
    if (failed(sources) || failed(identities) || failed(captures) ||
        failed(types) || failed(arguments))
      return failure();
    auto result = builder.create<ReduceOp>(location, *types, *sources, *identities,
                                           *captures, reduce.getAxes());
    for (NamedAttribute attribute : reduce->getDiscardableAttrs())
      result->setAttr(attribute.getName(), attribute.getValue());
    {
      OpBuilder::InsertionGuard guard(builder);
      Block *body = builder.createBlock(
          &result.getCombine(), {}, *arguments,
          SmallVector<Location>(arguments->size(), location));
      IRMapping nestedMapping(mapping);
      nestedMapping.map(reduce.getCombine().front().getArguments(), body->getArguments());
      for (Operation &nested : reduce.getCombine().front().without_terminator())
        if (failed(clonePredicatedScalarOperation(
                builder, &nested, nestedMapping, predicate, shape)))
          return failure();
      SmallVector<Value> yielded;
      for (Value value : reduce.getCombine().front().getTerminator()->getOperands()) {
        auto replacement = lift(nestedMapping.lookupOrDefault(value));
        if (failed(replacement)) return failure();
        yielded.push_back(*replacement);
      }
      builder.create<YieldOp>(location, yielded);
    }
    if (failed(remapSchemaAxes(result, reduce->getOperandTypes(),
                               reduce->getResultTypes())))
      return failure();
    clone = result;
  } else if (auto access = dyn_cast<AccessOpInterface>(operation);
             access && (access.getAccessKind() == AccessKind::Load ||
                        access.getAccessKind() == AccessKind::Gather ||
                        access.getAccessKind() == AccessKind::Store)) {
    // In a nonempty independent chunk, an invariant original read is already
    // required. Keep its own domain instead of repeating it for the new lanes.
    bool plainLoad = access.getAccessKind() == AccessKind::Load;
    bool safeRead = plainLoad && shape && nonemptyIterations;
    if (safeRead)
      for (Value operand : operation->getOperands()) {
        auto varying = hasIterationAxes(mapped(operand).getType());
        if (failed(varying)) return failure();
        if (*varying) { safeRead = false; break; }
      }
    auto view = dyn_cast<ViewType>(access.getAccessResource().getType());
    if (plainLoad && !shape && isa<FragmentType>(access.getAccessValueType()) &&
        view && view.getAccess() == 0) {
      PhysicalProgramAnalysis analysis(operation->getParentOfType<func::FuncOp>());
      auto bounds = analysis.accessBounds(operation);
      safeRead = bounds.isExact() && bounds.assumedAxes.empty();
    }
    if (safeRead) {
      clone = builder.clone(*operation, mapping);
    } else {
      SmallVector<Value> coordinates, payloads;
      for (Value coordinate : access.getAccessCoordinates())
        coordinates.push_back(mapped(coordinate));
      for (Value payload : access.getAccessPayloads()) {
        auto target = resultType(payload.getType());
        if (failed(target)) return failure();
        auto projected = project(mapped(payload), *target);
        if (failed(projected)) return failure();
        payloads.push_back(*projected);
      }
      auto validity = maskedValidity(access.getAccessValidity(), access.getAccessValueType());
      auto types = liftTypes(operation->getResultTypes());
      if (failed(validity) || failed(types)) return failure();
      Value inactive;
      if (access.getAccessFillMutable()) {
        auto value = fill(access.getAccessFill(), access.getAccessValueType());
        if (failed(value)) return failure();
        inactive = *value;
      }
      clone = builder.clone(*operation, mapping);
      for (auto [result, type] : llvm::zip_equal(clone->getResults(), *types))
        setPhysicalValueType(result, type);
      // Remap the original operand slots before optional mask/fill insertion
      // changes the access's operand segments.
      if (failed(remapSchemaAxes(clone, operation->getOperandTypes(),
                                 operation->getResultTypes())))
        return failure();
      auto replacement = cast<AccessOpInterface>(clone);
      replacement.getAccessCoordinatesMutable().assign(coordinates);
      replacement.getAccessPayloadsMutable().assign(payloads);
      replacement.getAccessValidityMutable().assign(*validity);
      if (auto fillGroup = replacement.getAccessFillMutable()) fillGroup->assign(inactive);
    }
  } else if (auto conversion = dyn_cast<CastOp>(operation);
             conversion && isFloatToIntegerCast(conversion)) {
    // Inactive lanes may be NaN/out of range. Guard the input before conversion.
    auto value = lift(mapped(conversion.getValue()));
    auto type = resultType(conversion.getType());
    if (failed(value) || failed(type)) return failure();
    auto fragment = dyn_cast<FragmentType>((*value).getType());
    Type element = fragment ? fragment.getElementType() : (*value).getType();
    Value zero = builder.create<arith::ConstantOp>(location, builder.getZeroAttr(element));
    auto projectedZero = project(zero, (*value).getType());
    if (failed(projectedZero)) return failure();
    auto guarded = selectScalarProduct(builder, location, predicate, *value, *projectedZero);
    if (failed(guarded)) return failure();
    IRMapping guardedMapping(mapping);
    guardedMapping.map(conversion.getValue(), *guarded);
    auto results = cloneWithSchema(builder, operation, guardedMapping, TypeRange{*type});
    if (failed(results)) return failure();
    mapping.map(operation->getResults(), *results);
    return success();
  } else {
    bool varying = false;
    if (shape)
      for (Value operand : operation->getOperands()) {
        auto present = hasIterationAxes(mapped(operand).getType());
        if (failed(present)) return failure();
        varying |= *present;
      }
    if (varying) {
      SmallVector<Type> selected;
      // Product fields are inferred from the actual mapped operands. An extract
      // preserves the selected field even when other record fields vary.
      if (!isa<MakeRecordOp, ExtractOp>(operation)) {
        auto types = liftTypes(operation->getResultTypes());
        if (failed(types)) return failure();
        selected = std::move(*types);
      }
      auto results = cloneWithSchema(builder, operation, mapping, selected);
      if (failed(results)) return failure();
      mapping.map(operation->getResults(), *results);
      return success();
    }
    clone = builder.clone(*operation, mapping);
  }
  if (Attribute origin = operation->getAttr(originAttr)) clone->setAttr(originAttr, origin);
  for (auto [previous, replacement] :
       llvm::zip(operation->getResults(), clone->getResults()))
    mapping.map(previous, replacement);
  return success();
}

} // namespace intent::gpu
