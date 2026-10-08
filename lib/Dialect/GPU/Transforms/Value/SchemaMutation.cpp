#include "Intent/Dialect/GPU/Transforms/Value/SchemaMutation.h"
#include "Intent/Analysis/ControlFlow.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/FragmentOpInterface.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <functional>

using namespace mlir;

namespace intent::gpu {

void setPhysicalValueType(Value value, Type type, ValueTypeChangeCallback changed) {
  Type previous = value.getType();
  if (previous == type)
    return;
  value.setType(type);
  if (changed)
    changed(value, previous);
}

namespace {
struct SchemaChanges {
  ValueTypeChangeCallback callback;
  void setType(Value value, Type type) {
    setPhysicalValueType(value, type, callback);
  }
  ValueTypeChangeCallback typeChanged() { return callback; }
};

bool carriesSchema(Type type) { return isa<FragmentType, RecordType>(type); }

struct ControlSchemaGroup {
  SmallVector<Value> targets;
  SmallVector<ControlFlowEdge> incoming;
};

FailureOr<SmallVector<ControlSchemaGroup>> controlGroups(Operation *operation) {
  SmallVector<ControlSchemaGroup> groups;
  SmallVector<Value> targets(operation->getResults());
  for (Region &region : operation->getRegions())
    for (Block &block : region)
      llvm::append_range(targets, block.getArguments());
  if (isa<BranchOpInterface>(operation))
    for (OpOperand &operand : operation->getOpOperands())
      for (const ControlFlowEdge &edge : queryControlFlowOutgoing(operand).edges)
        if (edge.target && !llvm::is_contained(targets, edge.target))
          targets.push_back(edge.target);
  for (Value target : targets) {
    auto incoming = queryControlFlowIncoming(target);
    if (!carriesSchema(target.getType()) &&
        llvm::none_of(incoming.edges, [](const ControlFlowEdge &edge) {
          return edge.operand && carriesSchema(edge.operand->get().getType());
        }))
      continue;
    if (!incoming.complete || incoming.edges.empty())
      return operation->emitOpError("physical control schema has a produced or "
                                    "unknown incoming slot"),
             failure();
    ControlSchemaGroup combined{{target}, {}};
    combined.incoming = std::move(incoming.edges);
    // One operand slot may feed both a region formal and its parent result.
    // Identical SSA values in two distinct slots do not connect components.
    for (unsigned index = 0; index < groups.size();) {
      bool sharedSlot = llvm::any_of(
          groups[index].incoming, [&](const ControlFlowEdge &edge) {
            return llvm::any_of(
                combined.incoming, [&](const ControlFlowEdge &candidate) {
                  return edge.operand == candidate.operand;
                });
          });
      if (!sharedSlot) {
        ++index;
        continue;
      }
      llvm::append_range(combined.targets, groups[index].targets);
      llvm::append_range(combined.incoming, groups[index].incoming);
      groups.erase(groups.begin() + index);
      index = 0;
    }
    groups.push_back(std::move(combined));
  }
  return groups;
}

LogicalResult projectControlGroup(Operation *operation,
                                  const ControlSchemaGroup &group, Type type,
                                  ValueTypeChangeCallback changed,
                                  OpBuilder::Listener *listener) {
  llvm::SmallPtrSet<OpOperand *, 8> seen;
  for (const ControlFlowEdge &edge : group.incoming) {
    if (!edge.operand || !seen.insert(edge.operand).second)
      continue;
    OpBuilder builder(edge.operand->getOwner());
    builder.setListener(listener);
    auto projected = projectPhysicalValueToSchema(
        builder, operation->getLoc(), edge.operand->get(), type, changed);
    if (failed(projected))
      return operation->emitOpError(
          "incoming control slot cannot adopt its physical boundary schema");
    edge.operand->set(*projected);
  }
  for (Value target : group.targets)
    setPhysicalValueType(target, type, changed);
  return success();
}
} // namespace

bool hasSchemaBoundary(Operation *operation) {
  return isa<MakeRecordOp, ExtractOp, RegionFoldOp, RegionScanOp,
             RegionBranchOpInterface, BranchOpInterface>(operation);
}

LogicalResult verifySelectedControlSchemas(Operation *operation) {
  auto groups = controlGroups(operation);
  if (failed(groups))
    return failure();
  for (const ControlSchemaGroup &group : *groups) {
    Type type = group.targets.front().getType();
    if (llvm::any_of(group.targets,
                     [&](Value target) { return target.getType() != type; }))
      return operation->emitOpError(
          "selected control slots disagree on their physical schema");
  }
  return success();
}

LogicalResult projectSelectedControlSchemas(Operation *operation,
                                             OpBuilder::Listener *listener) {
  if (failed(verifySelectedControlSchemas(operation)))
    return failure();
  auto groups = controlGroups(operation);
  if (failed(groups))
    return failure();
  for (const ControlSchemaGroup &group : *groups)
    if (failed(projectControlGroup(operation, group,
                                    group.targets.front().getType(), {}, listener)))
      return failure();
  return success();
}

LogicalResult projectSchemaBoundary(Value target,
                                    ValueTypeChangeCallback changed,
                                    OpBuilder::Listener *listener) {
  Operation *owner = target.getDefiningOp();
  if (!owner)
    owner = cast<BlockArgument>(target).getOwner()->getParentOp();
  auto project = [&](OpOperand &operand, Type type) -> LogicalResult {
    OpBuilder builder(operand.getOwner());
    builder.setListener(listener);
    auto value = projectPhysicalValueToSchema(
        builder, owner->getLoc(), operand.get(), type, changed);
    if (failed(value))
      return failure();
    operand.set(*value);
    return success();
  };
  if (auto record = dyn_cast<MakeRecordOp>(owner)) {
    auto type = cast<RecordType>(target.getType());
    for (auto [index, field] : llvm::enumerate(type.getFieldTypes()))
      if (failed(project(owner->getOpOperand(index),
                         cast<TypeAttr>(field).getValue())))
        return owner->emitOpError(
            "product field cannot adopt its selected schema");
    return success();
  }
  if (auto extract = dyn_cast<ExtractOp>(owner)) {
    auto type = extract.getRecord().getType();
    SmallVector<Attribute> fields(type.getFieldTypes().begin(),
                                 type.getFieldTypes().end());
    fields[extract.getField()] = TypeAttr::get(target.getType());
    auto selected = RecordType::get(type.getContext(), type.getFieldNames(),
                                    ArrayAttr::get(type.getContext(), fields),
                                    type.getOwner());
    return project(owner->getOpOperand(0), selected);
  }
  auto groups = controlGroups(owner);
  if (failed(groups))
    return failure();
  for (const ControlSchemaGroup &group : *groups)
    if (llvm::is_contained(group.targets, target))
      return projectControlGroup(owner, group, target.getType(), changed,
                                 listener);
  return owner->emitOpError("selected control value has no schema boundary slot");
}

LogicalResult closeSchemaBoundary(Operation *operation,
                                  ValueTypeChangeCallback changed,
                                  OpBuilder::Listener *listener) {
  auto kernel = operation->getParentOfType<func::FuncOp>();
  SchemaChanges changes{changed};
  if (auto record = dyn_cast<MakeRecordOp>(operation)) {
    RecordType current = record.getResult().getType();
    SmallVector<Attribute> fields;
    fields.reserve(record.getFields().size());
    for (Value field : record.getFields())
      fields.push_back(TypeAttr::get(field.getType()));
    changes.setType(
        record.getResult(),
        RecordType::get(kernel.getContext(), current.getFieldNames(),
                        ArrayAttr::get(kernel.getContext(), fields),
                        current.getOwner()));
    return success();
  }
  std::function<FailureOr<Type>(Type, Type, Type)> joinTypes =
      [&](Type current, Type lhs, Type rhs) -> FailureOr<Type> {
    if (auto lhsFragment = dyn_cast<FragmentType>(lhs)) {
      auto rhsFragment = dyn_cast<FragmentType>(rhs);
      auto currentFragment = dyn_cast<FragmentType>(current);
      if (!rhsFragment || !currentFragment ||
          lhsFragment.getElementType() != rhsFragment.getElementType() ||
          lhsFragment.getElementType() != currentFragment.getElementType() ||
          lhsFragment.getShape().size() != rhsFragment.getShape().size() ||
          lhsFragment.getShape().size() != currentFragment.getShape().size() ||
          lhsFragment.getAxisMaps() != rhsFragment.getAxisMaps() ||
          lhsFragment.getValidity() != rhsFragment.getValidity() ||
          lhsFragment.getOwner() != rhsFragment.getOwner())
        return failure();
      SmallVector<Attribute> shape;
      for (auto [left, right] :
           llvm::zip(lhsFragment.getShape(), rhsFragment.getShape())) {
        if (left == right) {
          shape.push_back(left);
          continue;
        }
        auto leftExtent = cast<PhysicalExprAttr>(left);
        auto rightExtent = cast<PhysicalExprAttr>(right);
        bool leftUnit = leftExtent.getKind() == PhysicalExprKind::Constant &&
                        leftExtent.getValue() == 1;
        bool rightUnit = rightExtent.getKind() == PhysicalExprKind::Constant &&
                         rightExtent.getValue() == 1;
        if (leftUnit == rightUnit)
          return failure();
        shape.push_back(leftUnit ? right : left);
      }
      return Type(FragmentType::get(
          kernel.getContext(), lhsFragment.getElementType(),
          ArrayAttr::get(kernel.getContext(), shape), lhsFragment.getAxisMaps(),
          lhsFragment.getValidity(), lhsFragment.getOwner()));
    }
    auto lhsRecord = dyn_cast<RecordType>(lhs);
    auto rhsRecord = dyn_cast<RecordType>(rhs);
    auto currentRecord = dyn_cast<RecordType>(current);
    if (lhsRecord || rhsRecord || currentRecord) {
      if (!lhsRecord || !rhsRecord || !currentRecord ||
          lhsRecord.getFieldNames() != rhsRecord.getFieldNames() ||
          lhsRecord.getFieldNames() != currentRecord.getFieldNames() ||
          lhsRecord.getFieldTypes().size() != rhsRecord.getFieldTypes().size() ||
          lhsRecord.getFieldTypes().size() !=
              currentRecord.getFieldTypes().size() ||
          lhsRecord.getOwner() != rhsRecord.getOwner())
        return failure();
      SmallVector<Attribute> fields;
      for (auto [base, left, right] :
           llvm::zip(currentRecord.getFieldTypes(), lhsRecord.getFieldTypes(),
                     rhsRecord.getFieldTypes())) {
        FailureOr<Type> joined = joinTypes(
            cast<TypeAttr>(base).getValue(), cast<TypeAttr>(left).getValue(),
            cast<TypeAttr>(right).getValue());
        if (failed(joined))
          return failure();
        fields.push_back(TypeAttr::get(*joined));
      }
      return Type(RecordType::get(
          kernel.getContext(), lhsRecord.getFieldNames(),
          ArrayAttr::get(kernel.getContext(), fields), lhsRecord.getOwner()));
    }
    return current == lhs && lhs == rhs ? FailureOr<Type>(current)
                                        : FailureOr<Type>(failure());
  };
  if (isa<RegionBranchOpInterface, BranchOpInterface>(operation)) {
    auto groups = controlGroups(operation);
    if (failed(groups))
      return failure();
    for (const ControlSchemaGroup &group : *groups) {
      SmallVector<OpOperand *> producers, seeds;
      llvm::SmallPtrSet<OpOperand *, 8> seen;
      for (const ControlFlowEdge &edge : group.incoming) {
        if (!edge.operand || !seen.insert(edge.operand).second)
          continue;
        bool seed = edge.kind == ControlFlowEdgeKind::Entry ||
                    edge.kind == ControlFlowEdgeKind::Bypass;
        (seed ? seeds : producers).push_back(edge.operand);
      }
      if (producers.empty())
        producers = seeds;
      if (producers.empty())
        return failure();
      Type target = producers.front()->get().getType();
      auto joiningType = [&](Value value, bool hasNonUniform) -> Type {
        Type type = value.getType();
        if (!hasNonUniform)
          return type;
        UniformValueAnalysis uniform(describeUniformValue);
        if (!uniform.evaluate(value))
          return type;
        auto fragment = dyn_cast<FragmentType>(type);
        if (!fragment)
          return type;
        auto unit = PhysicalExprAttr::get(
            kernel.getContext(), PhysicalExprKind::Constant, 1,
            StringAttr::get(kernel.getContext(), ""),
            ArrayAttr::get(kernel.getContext(), {}));
        SmallVector<Attribute> units(fragment.getShape().size(), unit);
        return FragmentType::get(kernel.getContext(), fragment.getElementType(),
                                 ArrayAttr::get(kernel.getContext(), units),
                                 fragment.getAxisMaps(), fragment.getValidity(),
                                 fragment.getOwner());
      };
      if (producers.size() > 1) {
        UniformValueAnalysis uniform(describeUniformValue);
        bool hasNonUniform = llvm::any_of(producers, [&](OpOperand *operand) {
          return !uniform.evaluate(operand->get());
        });
        target = joiningType(producers.front()->get(), hasNonUniform);
        for (OpOperand *operand : llvm::drop_begin(producers)) {
          auto joined = joinTypes(group.targets.front().getType(), target,
                                  joiningType(operand->get(), hasNonUniform));
          if (failed(joined))
            return operation->emitOpError(
                "control inputs have no unique physical boundary schema");
          target = *joined;
        }
      }
      if (failed(
              projectControlGroup(operation, group, target, changed, listener)))
        return failure();
    }
    return success();
  }
  if (isa<RegionFoldOp, RegionScanOp>(operation)) {
    auto structured = cast<StructuredOpInterface>(operation);
    if (failed(verifyStructuredArity(structured)))
      return failure();
    OpBuilder builder(operation);
    builder.setListener(listener);
    for (const StructuredSchemaGroup &group :
         queryStructuredSchemaGroups(operation)) {
      Type target = group.producer->get().getType();
      for (Value result : group.results)
        if (failed(retargetValueExtents(result, target, changes.typeChanged(),
                                       listener)))
          return failure();
      for (BlockArgument argument : group.arguments)
        if (failed(retargetValueExtents(argument, target, changes.typeChanged(),
                                       listener)))
          return failure();
      FailureOr<Value> projected = projectPhysicalValueToSchema(
          builder, operation->getLoc(), operation->getOperand(group.seedOperand),
          target, changes.typeChanged());
      if (failed(projected)) {
        operation->emitOpError("structured seed cannot adopt its producer schema")
            << "; seed_operand=" << group.seedOperand;
        return failure();
      }
      operation->setOperand(group.seedOperand, *projected);
      for (Value result : group.results)
        changes.setType(result, target);
      for (BlockArgument argument : group.arguments)
        changes.setType(argument, target);
      for (OpOperand *yield : group.yields) {
        OpBuilder yieldBuilder(yield->getOwner());
        yieldBuilder.setListener(listener);
        auto projectedYield = projectPhysicalValueToSchema(
            yieldBuilder, yield->getOwner()->getLoc(), yield->get(), target,
            changes.typeChanged());
        if (failed(projectedYield)) {
          operation->emitOpError(
              "structured combine cannot adopt its producer schema")
              << "; seed_operand=" << group.seedOperand;
          return failure();
        }
        yield->set(*projectedYield);
      }
    }
    return success();
  }
  // Structured arguments/results are the record-schema authority.  Refresh
  // projections only after those schemas have been aligned; doing this before
  // the region owner leaves combine-body fields one refinement behind.
  if (auto extract = dyn_cast<ExtractOp>(operation)) {
    RecordType record = extract.getRecord().getType();
    if (extract.getField() < record.getFieldTypes().size())
      changes.setType(
          extract.getResult(),
          cast<TypeAttr>(record.getFieldTypes()[extract.getField()]).getValue());
  }
  return success();
}

} // namespace intent::gpu
