#include "Intent/Dialect/CPU/Analysis/UniformValues.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/CPU/Analysis/ExtentRelations.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;
namespace intent::cpu {

Value canonicalUniformMemory(Value value) {
  while (true) {
    if (auto cast = value.getDefiningOp<memref::CastOp>()) {
      value = cast.getSource();
      continue;
    }
    auto view = value.getDefiningOp<memref::SubViewOp>();
    if (!view || view.getSourceType().getRank() != view.getType().getRank())
      return value;
    for (unsigned axis = 0; axis < view.getSourceType().getRank(); ++axis)
      if (getConstantIntValue(view.getMixedOffsets()[axis]) != 0 ||
          getConstantIntValue(view.getMixedStrides()[axis]) != 1 ||
          !haveEqualExtents(ValueBoundsConstraintSet::Variable(view.getMixedSizes()[axis]),
                            ValueBoundsConstraintSet::Variable(view.getSource(), axis)))
        return value;
    value = view.getSource();
  }
}

Attribute foldUniformComputation(linalg::GenericOp operation,
    const UniformBindings &operands, const UniformBindings &scalarFacts,
    unsigned outputIndex) {
  if (outputIndex >= operation.getOutputs().size() || operation.getNumResults()) return {};
  Block &body = operation.getRegion().front();
  if (llvm::any_of(body.without_terminator(), [](Operation &nested) {
        return nested.getNumRegions() || !isMemoryEffectFree(&nested);
      })) return {};
  UniformValueAnalysis values(describeScalarValue);
  auto maps = operation.getIndexingMapsArray();
  Value output = operation.getOutputs()[outputIndex];
  UniformBindings bindings = scalarFacts;
  for (auto &operand : operands) bindings[operand.first] = operand.second;
  UniformExpression expression;
  expression.type = cast<MemRefType>(output.getType()).getElementType();
  if (isMatrixContraction(operation)) {
    expression.kind = UniformKind::Contract;
    expression.operands.assign(operation.getInputs().begin(), operation.getInputs().end());
    expression.operands.push_back(output);
    return values.fold(expression, bindings);
  }
  if (!operation.getNumReductionLoops() &&
      maps[operation.getNumDpsInputs() + outputIndex].isPermutation()) {
    for (auto [argument, operand] : llvm::zip(body.getArguments(), operation->getOperands()))
      bindings[argument] = operands.lookup(operand);
    return values.evaluate(body.getTerminator()->getOperand(outputIndex), bindings);
  }
  if (!operation.getNumReductionLoops()) return {};
  expression.kind = UniformKind::Fold;
  expression.stateCount = operation.getNumDpsInits();
  expression.result = outputIndex;
  llvm::append_range(expression.operands, operation.getOutputs());
  llvm::append_range(expression.operands, operation.getInputs());
  llvm::append_range(expression.parameters,
      body.getArguments().drop_front(operation.getNumDpsInputs()));
  llvm::append_range(expression.parameters,
      body.getArguments().take_front(operation.getNumDpsInputs()));
  llvm::append_range(expression.yields, body.getTerminator()->getOperands());
  auto iterators = operation.getIteratorTypesArray();
  SmallVector<bool> positive(iterators.size(), false);
  for (auto [operand, map] : llvm::zip(operation->getOperands(), maps)) {
    auto type = dyn_cast<MemRefType>(operand.getType());
    if (!type) continue;
    for (auto [axis, coordinate] : llvm::enumerate(map.getResults()))
      if (auto dimension = dyn_cast<AffineDimExpr>(coordinate))
        positive[dimension.getPosition()] = positive[dimension.getPosition()] || type.getDimSize(axis) > 0;
  }
  expression.nonempty = llvm::all_of(llvm::enumerate(iterators), [&](auto iterator) {
    return iterator.value() != utils::IteratorType::reduction || positive[iterator.index()];
  });
  return values.fold(expression, bindings);
}

UniformMemoryAnalysis::UniformMemoryAnalysis(StorageAnalysis &storage,
                                           UniformBindings scalarFacts)
    : storage(&storage), scalarFacts(std::move(scalarFacts)),
      values(describeScalarValue) {}

Attribute UniformMemoryAnalysis::read(Value value) const {
  if (!isa<MemRefType>(value.getType())) return values.evaluate(value, scalarFacts);
  Type element = cast<MemRefType>(value.getType()).getElementType();
  Value key = canonicalUniformMemory(value);
  auto found = memory.find(key);
  Value origin = storage->uniqueOrigin(key);
  Attribute constant = found != memory.end() ? found->second
      : origin ? memory.lookup(origin) : Attribute{};
  auto typed = dyn_cast_or_null<TypedAttr>(constant);
  return typed && typed.getType() == element ? constant : Attribute{};
}

void UniformMemoryAnalysis::write(Value value, Attribute constant) {
  if (!isa<MemRefType>(value.getType())) {
    if (constant) scalarFacts[value] = constant;
    else scalarFacts.erase(value);
    return;
  }
  SmallVector<Value> invalidated;
  for (const auto &fact : memory)
    if (!storage->disjoint(fact.first, value)) invalidated.push_back(fact.first);
  for (Value alias : invalidated) memory.erase(alias);
  if (constant) memory[canonicalUniformMemory(value)] = constant;
}

void UniformMemoryAnalysis::invalidate(const StorageEffects &effects) {
  if (!effects.complete || effects.ordered) {
    memory.clear();
    return;
  }
  for (const StorageEffect &entry : effects.entries) {
    if (isa<MemoryEffects::Read, MemoryEffects::Allocate>(entry.effect.getEffect()))
      continue;
    Value target = entry.effect.getValue();
    if (!target || !isa<MemRefType>(target.getType())) memory.clear();
    else write(target, {});
  }
}

void UniformMemoryAnalysis::invalidate(Operation *operation) {
  invalidate(storage->effects(operation));
}

UniformComputationFacts
UniformMemoryAnalysis::evaluate(linalg::GenericOp operation) const {
  UniformComputationFacts facts;
  facts.outputs.resize(operation.getNumDpsInits());
  if (operation.getNumResults()) return facts;
  Block &body = operation.getRegion().front();
  if (llvm::any_of(body.without_terminator(), [](Operation &nested) {
        return nested.getNumRegions() || !isMemoryEffectFree(&nested);
      })) return facts;
  auto effects = storage->effects(operation);
  if (!effects.complete || effects.ordered) return facts;

  // All components observe the same incoming state, including coupled product
  // reductions. Publishing one output before evaluating another is incorrect.
  for (Value operand : operation->getOperands()) facts.operands[operand] = read(operand);
  auto maps = operation.getIndexingMapsArray();
  facts.canSubstituteInputs = true;
  for (auto [index, output] : llvm::enumerate(operation.getOutputs())) {
    AffineMap outputMap = maps[operation.getNumDpsInputs() + index];
    bool unsafeAlias = false;
    for (auto [number, input] : llvm::enumerate(operation.getInputs())) {
      if (!isa<MemRefType>(input.getType()) || storage->disjoint(input, output))
        continue;
      bool sameElement = input == output && !operation.getNumReductionLoops() &&
          maps[number] == outputMap && outputMap.isPermutation();
      unsafeAlias |= !sameElement;
    }
    for (auto [other, destination] : llvm::enumerate(operation.getOutputs()))
      if (index != other && !storage->disjoint(output, destination)) unsafeAlias = true;
    facts.canSubstituteInputs &= !unsafeAlias;
    if (!unsafeAlias)
      facts.outputs[index] = foldUniformComputation(operation, facts.operands,
                                                    scalarFacts, index);
  }
  return facts;
}

UniformComputationFacts UniformMemoryAnalysis::visit(linalg::GenericOp operation) {
  auto facts = evaluate(operation);
  invalidate(operation);
  for (auto [output, constant] : llvm::zip(operation.getOutputs(), facts.outputs))
    write(output, constant);
  return facts;
}

void UniformMemoryAnalysis::visit(Operation *operation) {
  if (auto fill = dyn_cast<linalg::FillOp>(operation)) {
    write(fill.getOutputs()[0], read(fill.getInputs()[0]));
  } else if (auto copy = dyn_cast<memref::CopyOp>(operation)) {
    write(copy.getTarget(), read(copy.getSource()));
  } else if (auto generic = dyn_cast<linalg::GenericOp>(operation)) {
    visit(generic);
  } else if (auto store = dyn_cast<memref::StoreOp>(operation)) {
    write(store.getMemref(), store.getIndices().empty() ? read(store.getValue()) : Attribute{});
  } else {
    invalidate(operation);
  }
}

}
