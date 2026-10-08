#include "Intent/Dialect/CPU/Analysis/ProducerReplay.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace intent::cpu {
namespace {

bool inside(Value value, Operation *scope) {
  if (Operation *definition = value.getDefiningOp())
    return scope->isAncestor(definition);
  return scope->isAncestor(cast<BlockArgument>(value).getOwner()->getParentOp());
}

class Collector {
public:
  Collector(Operation *scope, ValueRange frontier, StorageAnalysis &storage)
      : result{scope, llvm::to_vector(frontier), {}, {}, {}, {}}, storage(storage) {}

  LogicalResult value(Value value) {
    if (llvm::is_contained(result.frontier, value)) return success();
    if (!inside(value, result.scope)) {
      if (!llvm::is_contained(result.externalValues, value))
        result.externalValues.push_back(value);
      return success();
    }
    Operation *definition = value.getDefiningOp();
    // An enclosing loop's induction variable is not an external SSA capture.
    // The caller must explicitly supply its coordinate binding.
    return definition ? operation(definition) : failure();
  }

  LogicalResult operation(Operation *operation) {
    if (seen.contains(operation)) return success();
    auto effects = storage.effects(operation);
    if (!effects.complete || effects.ordered) return failure();
    for (const StorageEffect &entry : effects.entries) {
      if (!isa<MemoryEffects::Read>(entry.effect.getEffect()) ||
          !entry.effect.getValue() ||
          !isa<MemRefType>(entry.effect.getValue().getType()))
        return failure();
      Value memory = entry.effect.getValue();
      if (!llvm::is_contained(result.reads, memory)) result.reads.push_back(memory);
    }
    auto status = operation->walk([&](Operation *nested) -> WalkResult {
      if (nested->getNumRegions() && !isa<RegionBranchOpInterface>(nested))
        return WalkResult::interrupt();
      for (Value operand : nested->getOperands()) {
        if (inside(operand, operation)) continue;
        if (failed(value(operand))) return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (status.wasInterrupted()) return failure();
    operation->walk([&](Operation *nested) {
      if (seen.insert(nested).second) result.nodes.push_back(nested);
    });
    result.operations.push_back(operation);
    return success();
  }

  ProducerReplay result;

private:
  StorageAnalysis &storage;
  llvm::SmallPtrSet<Operation *, 32> seen;
};

} // namespace

FailureOr<ProducerReplay> analyzeProducerPayload(
    Block &body, ValueRange frontier, ValueRange implicitReads,
    StorageAnalysis &storage) {
  Collector collector(body.getParentOp(), frontier, storage);
  for (Value memory : implicitReads) collector.result.reads.push_back(memory);
  for (Operation &operation : body.without_terminator()) {
    if (operation.getNumResults() && llvm::all_of(operation.getResults(),
        [&](Value value) { return llvm::is_contained(frontier, value); })) {
      if (!isMemoryEffectFree(&operation) || operation.getNumRegions())
        return failure();
      continue;
    }
    if (failed(collector.operation(&operation))) return failure();
  }
  for (Value value : body.getTerminator()->getOperands())
    if (failed(collector.value(value))) return failure();
  return std::move(collector.result);
}

FailureOr<ProducerReplay> analyzeProducerValue(
    Value value, Operation *scope, ValueRange frontier, StorageAnalysis &storage) {
  Collector collector(scope, frontier, storage);
  if (failed(collector.value(value))) return failure();
  return std::move(collector.result);
}

FailureOr<ProducerReplay> analyzeLinalgProducer(linalg::GenericOp producer,
                                               StorageAnalysis &storage) {
  Block &body = producer.getRegion().front();
  SmallVector<Value> frontier, reads;
  for (BlockArgument argument : body.getArguments())
    if (!argument.use_empty()) frontier.push_back(argument);
  body.walk([&](linalg::IndexOp index) { frontier.push_back(index.getResult()); });
  for (Value input : producer.getInputs())
    if (isa<MemRefType>(input.getType())) reads.push_back(input);
  for (auto [number, output] : llvm::enumerate(producer.getOutputs()))
    if (!body.getArgument(producer.getInputs().size() + number).use_empty() &&
        isa<MemRefType>(output.getType())) reads.push_back(output);
  auto payload = analyzeProducerPayload(body, frontier, reads, storage);
  if (failed(payload)) return failure();
  for (Value input : producer.getInputs())
    if (!isa<MemRefType>(input.getType())) payload->externalValues.push_back(input);
  return payload;
}

bool canReplayProducerAt(const ProducerReplay &payload, Operation *from,
                         Operation *to, StorageAnalysis &storage,
                         ValueRange disjointFrom) {
  if (!from || !to || !from->isAncestor(payload.scope) || from->isAncestor(to))
    return false;
  DominanceInfo dominance(from->getParentOfType<func::FuncOp>());
  if (!dominance.dominates(from, to)) return false;
  for (Value value : payload.externalValues)
    if (!dominance.dominates(value, to)) return false;
  for (Value memory : payload.reads) {
    Value observed = memory;
    if (inside(memory, payload.scope)) {
      observed = storage.uniqueOrigin(memory);
      if (!observed || inside(observed, payload.scope)) return false;
    }
    if (!storage.readStable(observed, from, to)) return false;
    for (Value written : disjointFrom)
      if (!storage.disjoint(memory, written)) return false;
  }
  return true;
}

} // namespace intent::cpu
