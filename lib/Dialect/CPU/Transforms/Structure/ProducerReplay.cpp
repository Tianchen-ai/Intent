#include "Intent/Dialect/CPU/Transforms/Structure/ProducerReplay.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;

namespace intent::cpu {
namespace {

LogicalResult checkFrontier(const ProducerReplay &payload, OpBuilder &builder,
                            IRMapping &mapping, Value requested = {}) {
  Block *block = builder.getInsertionBlock();
  if (!block || !block->getParentOp()) return failure();
  auto function = dyn_cast<func::FuncOp>(block->getParentOp());
  if (!function) function = block->getParentOp()->getParentOfType<func::FuncOp>();
  if (!function || function != payload.scope->getParentOfType<func::FuncOp>())
    return failure();
  DominanceInfo dominance(function);
  auto available = [&](Value value) {
    if (auto argument = dyn_cast<BlockArgument>(value))
      return dominance.dominates(argument.getOwner(), block);
    Operation *definition = value.getDefiningOp();
    return definition && dominance.properlyDominates(
        definition->getBlock(), definition->getIterator(), block,
        builder.getInsertionPoint(), /*enclosingOk=*/false);
  };
  for (Value value : payload.frontier)
    if (!mapping.contains(value) || mapping.lookup(value).getType() != value.getType() ||
        !available(mapping.lookup(value)))
      return failure();
  for (auto [original, replacement] : mapping.getValueMap())
    if ((original == requested || llvm::is_contained(payload.frontier, original) ||
         llvm::is_contained(payload.externalValues, original) ||
         llvm::is_contained(payload.operations, original.getDefiningOp())) &&
        (original.getType() != replacement.getType() || !available(replacement)))
      return failure();
  for (Value value : payload.externalValues)
    if (!available(mapping.lookupOrDefault(value))) return failure();
  return success();
}

void cloneNode(Operation *operation, const ProducerReplay &payload,
               OpBuilder &builder, IRMapping &mapping) {
  // Native region cloning remaps nested definitions, including linalg.index.
  // Restore explicitly bound coordinates instead of letting those definitions
  // silently become indices of the new consumer's loop domain.
  SmallVector<std::pair<Value, Value>> bound;
  for (Value value : payload.frontier)
    if (Operation *definition = value.getDefiningOp();
        definition && operation->isAncestor(definition))
      bound.emplace_back(value, mapping.lookup(value));
  builder.clone(*operation, mapping);
  for (auto [original, replacement] : bound) {
    Value cloned = mapping.lookup(original);
    if (cloned == replacement) continue;
    Operation *definition = cloned.getDefiningOp();
    cloned.replaceAllUsesWith(replacement);
    mapping.map(original, replacement);
    if (definition && isOpTriviallyDead(definition)) definition->erase();
  }
}

FailureOr<Value> materialize(const ProducerReplay &payload, Value value,
                            OpBuilder &builder, IRMapping &mapping) {
  if (mapping.contains(value)) return mapping.lookup(value);
  Operation *operation = value.getDefiningOp();
  if (!operation || !llvm::is_contained(payload.nodes, operation)) {
    if (llvm::is_contained(payload.externalValues, value)) return value;
    return failure();
  }
  if (!llvm::is_contained(payload.operations, operation)) return failure();
  auto status = operation->walk([&](Operation *nested) -> WalkResult {
    for (Value operand : nested->getOperands()) {
      Operation *owner = operand.getDefiningOp();
      if (!owner) owner = cast<BlockArgument>(operand).getOwner()->getParentOp();
      if (operation->isAncestor(owner)) continue;
      auto replayed = materialize(payload, operand, builder, mapping);
      if (failed(replayed)) return WalkResult::interrupt();
      mapping.map(operand, *replayed);
    }
    return WalkResult::advance();
  });
  if (status.wasInterrupted()) return failure();
  cloneNode(operation, payload, builder, mapping);
  return mapping.lookup(value);
}

bool canMaterialize(const ProducerReplay &payload, Value value, IRMapping &mapping) {
  if (mapping.contains(value) || llvm::is_contained(payload.externalValues, value))
    return true;
  Operation *operation = value.getDefiningOp();
  if (!operation || !llvm::is_contained(payload.operations, operation)) return false;
  return !operation->walk([&](Operation *nested) -> WalkResult {
    for (Value operand : nested->getOperands()) {
      Operation *owner = operand.getDefiningOp();
      if (!owner) owner = cast<BlockArgument>(operand).getOwner()->getParentOp();
      if (!operation->isAncestor(owner) && !canMaterialize(payload, operand, mapping))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  }).wasInterrupted();
}

} // namespace

LogicalResult cloneProducerPayload(const ProducerReplay &payload,
                                   OpBuilder &builder, IRMapping &mapping) {
  if (failed(checkFrontier(payload, builder, mapping))) return failure();
  for (Operation *operation : payload.operations) {
    if (operation->getNumResults() && llvm::all_of(operation->getResults(),
        [&](Value value) { return mapping.contains(value); })) continue;
    cloneNode(operation, payload, builder, mapping);
  }
  return success();
}

FailureOr<Value> materializeProducerValue(const ProducerReplay &payload,
                                         Value value, OpBuilder &builder,
                                         IRMapping &mapping) {
  if (failed(checkFrontier(payload, builder, mapping, value)) ||
      !canMaterialize(payload, value, mapping)) return failure();
  return materialize(payload, value, builder, mapping);
}

} // namespace intent::cpu
