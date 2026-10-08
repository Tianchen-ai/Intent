#include "Intent/Dialect/GPU/Transforms/Value/ExecutionSchema.h"

#include "Intent/Analysis/ControlFlow.h"
#include "Intent/Dialect/GPU/Analysis/Helpers.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/AccessOpInterface.h"
#include "Intent/Dialect/GPU/IR/FragmentOpInterface.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Value/SchemaMutation.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Intent/Interfaces/StructuredOpInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;

namespace intent::gpu {
namespace {

// Build at the real insertion point: projection queries can still see the
// current function, parameters and dominance. Only newly inserted operations
// and their owned regions are removed on failure; existing values stay intact.
class CloneInsertion {
public:
  explicit CloneInsertion(OpBuilder &builder)
      : builder(builder), guard(builder), block(builder.getInsertionBlock()),
        before(builder.getInsertionPoint() == block->end()
                   ? nullptr : &*builder.getInsertionPoint()),
        previous(builder.getInsertionPoint() == block->begin()
                     ? nullptr : &*std::prev(builder.getInsertionPoint())) {}

  ~CloneInsertion() {
    if (committed)
      return;
    SmallVector<Operation *> inserted;
    Operation *operation = previous ? previous->getNextNode()
                                    : (block->empty() ? nullptr : &block->front());
    for (; operation != before; operation = operation->getNextNode())
      inserted.push_back(operation);
    IRRewriter rewriter(builder.getContext());
    rewriter.setListener(builder.getListener());
    for (Operation *created : llvm::reverse(inserted))
      rewriter.eraseOp(created);
  }

  void commit() { committed = true; }

private:
  OpBuilder &builder;
  OpBuilder::InsertionGuard guard;
  Block *block;
  Operation *before;
  Operation *previous;
  bool committed = false;
};

Type elementType(Type type) {
  if (auto fragment = dyn_cast<FragmentType>(type))
    return fragment.getElementType();
  return type;
}

Type withElement(Type schema, Type element) {
  if (auto fragment = dyn_cast<FragmentType>(schema))
    return FragmentType::get(fragment.getContext(), element, fragment.getShape(),
                             fragment.getAxisMaps(), fragment.getValidity(),
                             fragment.getOwner());
  return element;
}

bool sameElements(Type original, Type selected) {
  auto before = dyn_cast<RecordType>(original);
  auto after = dyn_cast<RecordType>(selected);
  if (before || after) {
    if (!before || !after || before.getFieldNames() != after.getFieldNames() ||
        before.getFieldTypes().size() != after.getFieldTypes().size())
      return false;
    return llvm::all_of(llvm::zip(before.getFieldTypes(), after.getFieldTypes()),
                        [](auto fields) {
      return sameElements(cast<TypeAttr>(std::get<0>(fields)).getValue(),
                          cast<TypeAttr>(std::get<1>(fields)).getValue());
    });
  }
  return elementType(original) == elementType(selected);
}

LogicalResult projectOperand(OpBuilder &builder, Operation *source,
                             SmallVectorImpl<Value> &operands, unsigned slot,
                             Type target) {
  if (slot >= operands.size())
    return failure();
  auto projected = projectPhysicalValueToSchema(
      builder, source->getLoc(), operands[slot], target);
  if (failed(projected))
    return failure();
  operands[slot] = *projected;
  return success();
}

// Operation-owned operand roles. This does not select coordinates, range
// bounds, accumulator precision, or whether a read may move to this scope.
LogicalResult projectOperands(OpBuilder &builder, Operation *source,
                              SmallVectorImpl<Value> &operands,
                              TypeRange results) {
  if (isLaneWisePointwiseOperation(source)) {
    if (results.size() != 1)
      return failure();
    for (unsigned index = 0; index < operands.size(); ++index)
      if (failed(projectOperand(builder, source, operands, index,
                                withElement(results.front(),
                                            elementType(operands[index].getType())))))
        return failure();
  } else if (isa<MakeRecordOp>(source)) {
    auto record = results.size() == 1 ? dyn_cast<RecordType>(results.front())
                                      : RecordType();
    if (!record || record.getFieldTypes().size() != operands.size())
      return failure();
    for (auto [index, field] : llvm::enumerate(record.getFieldTypes()))
      if (failed(projectOperand(builder, source, operands, index,
                                cast<TypeAttr>(field).getValue())))
        return failure();
  }
  OpOperand *accumulator = nullptr;
  if (auto contract = dyn_cast<ContractOp>(source))
    accumulator = &contract.getAccumulatorMutable();
  else if (auto contract = dyn_cast<ScaledContractOp>(source))
    accumulator = &contract.getAccumulatorMutable();
  else if (auto contract = dyn_cast<SparseContractOp>(source))
    accumulator = &contract.getAccumulatorMutable();
  if (accumulator &&
      (results.size() != 1 ||
       failed(projectOperand(builder, source, operands,
                             accumulator->getOperandNumber(), results.front()))))
    return failure();
  if (auto access = dyn_cast<AccessOpInterface>(source);
      access && (access.getAccessKind() == AccessKind::Load ||
                 access.getAccessKind() == AccessKind::Gather)) {
    if (results.size() != 1)
      return failure();
    OperandRange valid = access.getAccessValidityMutable();
    if (!valid.empty() &&
        failed(projectOperand(builder, source, operands,
                              valid.getBeginOperandIndex(),
                              withElement(results.front(), builder.getI1Type()))))
      return failure();
    if (auto fill = access.getAccessFillMutable(); fill && !fill->empty()) {
      OperandRange values = *fill;
      if (failed(projectOperand(builder, source, operands,
                                values.getBeginOperandIndex(), results.front())))
        return failure();
    }
  }
  return success();
}

FailureOr<SmallVector<Value>> rebuildLeaf(
    OpBuilder &builder, Operation *source, ValueRange actualOperands,
    IRMapping &mapping, TypeRange selectedResults) {
  if (source->getNumRegions() ||
      actualOperands.size() != source->getNumOperands())
    return failure();
  for (auto [before, after] : llvm::zip(source->getOperands(), actualOperands))
    if (!sameElements(before.getType(), after.getType()))
      return failure();
  SmallVector<Value> operands(actualOperands);
  SmallVector<Type> types(selectedResults);
  if (types.empty() && source->getNumResults()) {
    auto inferred = inferElementwiseSchema(source, operands);
    if (failed(inferred))
      return failure();
    types = std::move(*inferred);
  }
  if (types.size() != source->getNumResults())
    return failure();
  for (auto [original, selected] : llvm::zip(source->getResultTypes(), types))
    if (!selected || !sameElements(original, selected))
      return failure();
  auto publish = [&](ValueRange values) -> SmallVector<Value> {
    mapping.map(source->getResults(), values);
    return llvm::to_vector(values);
  };
  if (isa<SplatOp, BroadcastOp>(source)) {
    auto projected = projectPhysicalValueToSchema(
        builder, source->getLoc(), operands.front(), types.front());
    if (failed(projected))
      return failure();
    mapping.erase(source);
    return publish(ValueRange{*projected});
  }
  if (failed(projectOperands(builder, source, operands, types)))
    return failure();
  IRMapping local(mapping);
  Operation *copy = builder.clone(*source, local);
  copy->setOperands(operands);
  if (auto extract = dyn_cast<ExtractOp>(source)) {
    auto record = dyn_cast<RecordType>(operands.front().getType());
    if (!record || extract.getField() >= record.getFieldTypes().size())
      return failure();
    copy->getResult(0).setType(
        cast<TypeAttr>(record.getFieldTypes()[extract.getField()]).getValue());
  } else if (!isa<arith::ConstantOp>(source)) {
    for (auto [result, type] : llvm::zip(copy->getResults(), types))
      result.setType(type);
  }
  if (failed(remapSchemaAxes(copy, source->getOperandTypes(),
                            source->getResultTypes())))
    return failure();
  SmallVector<Value> values;
  for (auto [result, type] : llvm::zip(copy->getResults(), types)) {
    auto projected = projectPhysicalValueToSchema(builder, source->getLoc(),
                                                 result, type);
    if (failed(projected))
      return failure();
    values.push_back(*projected);
  }
  mapping.map(source, copy);
  return publish(values);
}

FailureOr<SmallVector<Type>> selectedTypes(
    Operation *source, ValueRange operands,
    llvm::function_ref<Type(Value)> select,
    ClonedAxisRefinement refineBroadcast, func::FuncOp kernel) {
  SmallVector<Type> types;
  SmallVector<Type> operandTypes;
  for (Value operand : operands)
    operandTypes.push_back(operand.getType());
  for (OpResult result : source->getResults()) {
    Type type = select(result);
    if (!type || !sameElements(result.getType(), type))
      return failure();
    if (auto fragment = dyn_cast<FragmentType>(type);
        fragment && isLaneWisePointwiseOperation(source)) {
      if (!kernel)
        return failure();
      auto refined = queryValueSchema(kernel, fragment, operands);
      if (failed(refined))
        return failure();
      type = *refined;
    } else if (!isa<SplatOp, BroadcastOp>(source) &&
               isa<FragmentOpInterface>(source) &&
               isa<FragmentType>(result.getType()) && isa<FragmentType>(type)) {
      auto relations = queryFragmentOperandRelations(result);
      if (failed(relations))
        return failure();
      auto transported = transportFragmentResultType(
          *relations, operandTypes, type,
          [&](unsigned operand, unsigned sourceAxis, unsigned resultAxis) {
            return refineBroadcast && refineBroadcast(
                source->getOpOperand(operand), sourceAxis, result, resultAxis);
          });
      if (failed(transported))
        return failure();
      type = *transported;
    }
    types.push_back(type);
  }
  return types;
}

LogicalResult projectSlot(OpOperand &slot, Type type,
                           OpBuilder::Listener *listener) {
  OpBuilder builder(slot.getOwner(), listener);
  auto projected = projectPhysicalValueToSchema(
      builder, slot.getOwner()->getLoc(), slot.get(), type);
  if (failed(projected))
    return failure();
  slot.set(*projected);
  return success();
}

LogicalResult prepareStructuredSignature(StructuredOpInterface structured,
                                         OpBuilder::Listener *listener) {
  Operation *operation = structured.getOperation();
  if (failed(verifyStructuredArity(structured)))
    return failure();
  if (isa<ReduceOp, ScanOp>(operation)) {
    OperandRange identities = isa<ReduceOp>(operation)
        ? cast<ReduceOp>(operation).getIdentities()
        : cast<ScanOp>(operation).getIdentities();
    for (unsigned index = 0; index < identities.size(); ++index)
      if (failed(projectSlot(
              operation->getOpOperand(identities.getBeginOperandIndex() + index),
              operation->getResult(index).getType(), listener)))
        return failure();
  } else {
    // These are the operation's existing seed/formal/result groups, not a
    // shape-identity search across unrelated fields or captures.
    for (const StructuredSchemaGroup &group :
         queryStructuredSchemaGroups(operation)) {
      Type type = !group.results.empty() ? group.results.front().getType()
                                         : group.arguments.front().getType();
      if (llvm::any_of(group.results,
                       [&](Value value) { return value.getType() != type; }) ||
          llvm::any_of(group.arguments,
                       [&](Value value) { return value.getType() != type; }))
        return operation->emitOpError(
            "selected structured slots disagree on their physical schema");
      if (failed(projectSlot(operation->getOpOperand(group.seedOperand), type,
                             listener)))
        return failure();
    }
  }
  for (Region &region : operation->getRegions()) {
    SmallVector<Type> expected;
    if (&region == &structured.getCombine()) {
      expected = structured.getCombineArgumentTypes();
    } else if (&region == structured.getSummarizeRegion()) {
      SmallVector<Type> slices;
      for (Value source : structured.getSummarizeSources())
        slices.push_back(source.getType());
      expected = structured.getSummarizeArgumentTypes(slices);
    } else if (&region == structured.getApplyRegion()) {
      expected = structured.getApplyArgumentTypes();
    } else if (&region == structured.getEmitRegion()) {
      SmallVector<Type> slices;
      for (Value source : structured.getEmitSources())
        slices.push_back(source.getType());
      expected = structured.getEmitArgumentTypes(slices);
    } else {
      return failure();
    }
    if (expected.size() != region.front().getNumArguments())
      return failure();
    for (auto [argument, type] :
         llvm::zip(region.front().getArguments(), expected)) {
      if (!sameElements(argument.getType(), type))
        return failure();
      argument.setType(type);
    }
  }
  return success();
}

LogicalResult projectStructuredYields(StructuredOpInterface structured,
                                      OpBuilder::Listener *listener) {
  auto project = [&](Region &region, ValueRange expected) -> LogicalResult {
    Operation *yield = region.front().getTerminator();
    if (yield->getNumOperands() != expected.size())
      return failure();
    for (auto [slot, value] : llvm::zip(yield->getOpOperands(), expected))
      if (failed(projectSlot(slot, value.getType(), listener)))
        return failure();
    return success();
  };
  if (failed(project(structured.getCombine(), structured.getIdentities())))
    return failure();
  if (Region *summarize = structured.getSummarizeRegion())
    if (failed(project(*summarize, structured.getIdentities())))
      return failure();
  if (Region *apply = structured.getApplyRegion())
    if (failed(project(*apply, structured.getInitialStates())))
      return failure();
  // Emission is source-aligned, not a same-schema state edge. Its slice width
  // remains the caller's selected member domain, checked by the group verifier.
  return success();
}

// The immutable source and its native clone have identical regions/blocks at
// entry. Rebuild only the clone, replacing each old cloned leaf by real values.
LogicalResult rebuildRegionOperation(
    Operation *source, Operation *copy, IRMapping &mapping,
    llvm::function_ref<Type(Value)> select,
    ClonedAxisRefinement refineBroadcast, OpBuilder::Listener *listener) {
  for (auto [original, actual] :
       llvm::zip(source->getOperands(), copy->getOperands()))
    if (!sameElements(original.getType(), actual.getType()))
      return failure();
  auto types = selectedTypes(source, copy->getOperands(), select, refineBroadcast,
                             copy->getParentOfType<func::FuncOp>());
  if (failed(types))
    return failure();
  OpBuilder builder(copy, listener);
  if (!source->getNumRegions()) {
    auto values = rebuildLeaf(builder, source, copy->getOperands(), mapping, *types);
    if (failed(values))
      return failure();
    IRRewriter rewriter(builder.getContext());
    rewriter.setListener(listener);
    rewriter.replaceOp(copy, *values);
    return success();
  }
  SmallVector<Type> originalOperands(source->getOperandTypes());
  for (auto [result, type] : llvm::zip(copy->getResults(), *types))
    result.setType(type);
  if (failed(remapSchemaAxes(copy, originalOperands, source->getResultTypes())))
    return failure();
  // Establish the complete formal signature before reconstructing any users.
  for (auto [sourceRegion, targetRegion] :
       llvm::zip(source->getRegions(), copy->getRegions())) {
    for (auto [sourceBlock, targetBlock] : llvm::zip(sourceRegion, targetRegion)) {
      for (auto [before, after] :
           llvm::zip(sourceBlock.getArguments(), targetBlock.getArguments())) {
        Type type = select(before);
        if (!type || !sameElements(before.getType(), type))
          return failure();
        after.setType(type);
      }
    }
  }
  if (isa<RegionBranchOpInterface>(copy) &&
      failed(verifySelectedControlSchemas(copy)))
    return failure();
  auto structured = dyn_cast<StructuredOpInterface>(copy);
  if (structured && failed(prepareStructuredSignature(structured, listener)))
    return failure();
  for (auto [sourceRegion, targetRegion] :
       llvm::zip(source->getRegions(), copy->getRegions())) {
    for (auto [sourceBlock, targetBlock] : llvm::zip(sourceRegion, targetRegion)) {
      SmallVector<Operation *> original, cloned;
      for (Operation &operation : sourceBlock)
        original.push_back(&operation);
      for (Operation &operation : targetBlock)
        cloned.push_back(&operation);
      for (auto [before, after] : llvm::zip(original, cloned))
        if (failed(rebuildRegionOperation(before, after, mapping, select,
                                          refineBroadcast, listener)))
          return failure();
    }
  }
  if (isa<RegionBranchOpInterface>(copy) &&
      failed(projectSelectedControlSchemas(copy, listener)))
    return failure();
  if (structured && failed(projectStructuredYields(structured, listener)))
    return failure();
  mapping.map(source->getResults(), copy->getResults());
  return success();
}

} // namespace

FailureOr<SmallVector<Value>> cloneWithSchema(
    OpBuilder &builder, Operation *source, IRMapping &mapping,
    TypeRange selectedResults) {
  SmallVector<Value> operands;
  for (Value operand : source->getOperands())
    operands.push_back(mapping.lookupOrDefault(operand));
  return cloneWithSchema(builder, source, operands, mapping, selectedResults);
}

FailureOr<SmallVector<Value>> cloneWithSchema(
    OpBuilder &builder, Operation *source, ValueRange operands,
    IRMapping &mapping, TypeRange selectedResults) {
  if (!builder.getInsertionBlock())
    return failure();
  CloneInsertion insertion(builder);
  IRMapping local(mapping);
  auto values = rebuildLeaf(builder, source, operands, local, selectedResults);
  if (failed(values))
    return failure();
  mapping = std::move(local);
  insertion.commit();
  return values;
}

FailureOr<SmallVector<Value>> cloneWithPhysicalSchema(
    OpBuilder &builder, Operation *source, IRMapping &mapping,
    llvm::function_ref<Type(Value)> select,
    ClonedAxisRefinement refineBroadcast) {
  if (!builder.getInsertionBlock())
    return failure();
  CloneInsertion insertion(builder);
  IRMapping local(mapping);
  Operation *copy = builder.clone(*source, local);
  if (failed(rebuildRegionOperation(source, copy, local, select,
                                    refineBroadcast, builder.getListener())))
    return failure();
  SmallVector<Value> values;
  for (Value result : source->getResults())
    values.push_back(local.lookup(result));
  mapping = std::move(local);
  insertion.commit();
  return values;
}

} // namespace intent::gpu
