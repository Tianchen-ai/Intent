#include "Worklist.h"
#include "Intent/Dialect/GPU/IR/FragmentOpInterface.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;

namespace intent::gpu::value_relations {

RelationWorklist::RelationWorklist(func::FuncOp kernel,
                                   ValueRelationScope scope)
    : kernel(kernel), scope(scope),
      callback([this](Value value, Type previous) {
        typeChanged(value, previous);
      }) {}

ValueTypeChangeCallback RelationWorklist::typeChanged() { return callback; }

void RelationWorklist::setType(Value value, Type type) {
  setPhysicalValueType(value, type, callback);
}

void RelationWorklist::notifyOperationInserted(Operation *operation,
                                               OpBuilder::InsertPoint) {
  operation->walk<WalkOrder::PreOrder>([&](Operation *nested) {
    live.insert(nested);
    enqueue(nested);
  });
}

void RelationWorklist::notifyOperationErased(Operation *operation) {
  // Removing a use can change projection/authority queries on its producers.
  for (Value operand : operation->getOperands())
    affected(operand);
  enqueue(operation->getParentOp());
  operation->walk([&](Operation *nested) {
    live.erase(nested);
    for (auto &pending : queued)
      pending.erase(nested);
    for (Value result : nested->getResults())
      transitions.erase(result);
    for (Region &region : nested->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          transitions.erase(argument);
  });
}

void RelationWorklist::notifyOperationReplaced(Operation *operation,
                                               ValueRange replacements) {
  // Rewriter notifications precede RAUW, so the original result users are
  // still available here. They will observe the new edges when dequeued.
  for (Value result : operation->getResults())
    affected(result);
  for (Value value : replacements)
    affected(value);
}

void RelationWorklist::notifyOperationModified(Operation *operation) {
  enqueue(operation);
  for (Value operand : operation->getOperands())
    affected(operand);
  for (Value result : operation->getResults())
    affected(result);
  for (Region &region : operation->getRegions())
    for (Block &block : region)
      for (BlockArgument argument : block.getArguments())
        affected(argument);
}

bool RelationWorklist::enabled(Rule rule) const {
  switch (scope) {
  case ValueRelationScope::Complete:
    return true;
  case ValueRelationScope::Pointwise:
    return rule == Pointwise || rule == Reshape || rule == ReductionResult ||
           rule == ReductionIdentity;
  case ValueRelationScope::Contracts:
    return rule == ContractOperands || rule == ContractAccumulator;
  case ValueRelationScope::AccessResults:
    return rule == AccessResult;
  case ValueRelationScope::ReductionInputs:
    return rule == ReductionResult || rule == ReductionIdentity;
  }
  llvm_unreachable("unknown value relation scope");
}

void RelationWorklist::push(Operation *operation, Rule rule) {
  if (enabled(rule) && queued[rule].insert(operation).second)
    worklists[priority(rule)].emplace_back(operation, rule);
}

unsigned RelationWorklist::priority(Rule rule) {
  // Access results, reshapes and elementwise values share the same SSA queue:
  // a reshape must run between its producer and its consumers, not in a
  // separate whole-program sweep after those consumers have been visited.
  if (rule == AccessResult || rule == Pointwise || rule == Reshape)
    return AccessResult;
  return rule;
}

void RelationWorklist::enqueue(Operation *operation) {
  if (!operation || !live.contains(operation))
    return;
  if (isa<RegionFoldOp, RegionScanOp>(operation))
    push(operation, Captures);
  if (isa<ReduceOp, ScanOp>(operation)) {
    push(operation, ReductionResult);
    push(operation, ReductionIdentity);
    push(operation, ReductionYield);
  }
  if (hasSchemaBoundary(operation))
    push(operation, Aggregate);
  if (auto access = dyn_cast<AccessOpInterface>(operation)) {
    AccessKind kind = access.getAccessKind();
    if (kind == AccessKind::Load || kind == AccessKind::Gather)
      push(operation, AccessResult);
    if (kind == AccessKind::Load || kind == AccessKind::Gather ||
        kind == AccessKind::Store)
      push(operation, AccessValue);
  }
  if (isa<FragmentOpInterface>(operation) &&
      !isa<SplatOp, ReshapeOp>(operation))
    push(operation, Pointwise);
  if (isa<ReshapeOp>(operation))
    push(operation, Reshape);
  if (isa<ContractOp>(operation))
    push(operation, ContractOperands);
  if (isa<ContractOp, ScaledContractOp, SparseContractOp>(operation))
    push(operation, ContractAccumulator);
  // A yield/capture change affects its structured owner even though those
  // schema edges are not ordinary SSA result uses.
  Operation *parent = operation->getParentOp();
  if (parent && parent != kernel)
    enqueue(parent);
}

void RelationWorklist::affected(Value value) {
  if (auto argument = dyn_cast<BlockArgument>(value))
    enqueue(argument.getOwner()->getParentOp());
  else
    enqueue(value.getDefiningOp());
  for (Operation *user : value.getUsers())
    enqueue(user);
}

void RelationWorklist::typeChanged(Value value, Type previous) {
  if (previous == value.getType())
    return;
  auto transition = std::make_pair(previous, value.getType());
  auto &history = transitions[value];
  // Detect an actual repeated conflicting refinement, rather than silently
  // stopping after an arbitrary number of iterations.
  if (llvm::is_contained(history, transition)) {
    if (!conflict) {
      Operation *owner = value.getDefiningOp();
      if (!owner)
        owner = cast<BlockArgument>(value).getOwner()->getParentOp();
      owner->emitOpError(
          "physical relation closure repeated a conflicting type refinement")
          << "; previous=" << previous << "; selected=" << value.getType();
    }
    conflict = true;
  } else {
    history.push_back(transition);
  }
  affected(value);
}

LogicalResult RelationWorklist::run() {
  kernel.walk<WalkOrder::PreOrder>([&](Operation *operation) {
    live.insert(operation);
    enqueue(operation);
  });
  while (true) {
    unsigned selected = 0;
    while (selected != RuleCount && worklists[selected].empty())
      ++selected;
    if (selected == RuleCount)
      return success();
    auto [operation, rule] = worklists[selected].front();
    worklists[selected].pop_front();
    if (!queued[rule].erase(operation) || !live.contains(operation))
      continue;

    // Rules can replace operands on an owner or its region terminators. Track
    // these edges in addition to explicit type notifications, so a new
    // projection or a reverse refinement requeues every affected relation.
    SmallVector<std::pair<Operation *, SmallVector<Value>>> boundaries;
    auto remember = [&](Operation *boundary) {
      boundaries.emplace_back(boundary,
                              llvm::to_vector(boundary->getOperands()));
    };
    remember(operation);
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        if (!block.empty())
          remember(block.getTerminator());

    LogicalResult status = success();
    auto checked = [](WalkResult result) {
      return failure(result.wasInterrupted());
    };
    switch (rule) {
    case Captures:
      status = checked(alignStructuredCaptures(operation, *this));
      break;
    case ReductionResult:
      status = checked(alignReductionResultRelation(operation, *this));
      break;
    case ReductionIdentity:
      status = checked(alignReductionIdentityRelation(operation, *this));
      break;
    case Aggregate:
      status = closeSchemaBoundary(operation, typeChanged(), this);
      break;
    case AccessResult:
      status = checked(alignAccessResult(operation, *this));
      break;
    case Pointwise:
      status = checked(alignPointwiseValue(operation, *this));
      break;
    case ReductionYield:
      status = checked(alignReductionYield(operation, *this));
      break;
    case AccessValue:
      status = alignAccessValue(operation, *this);
      break;
    case Reshape:
      status = refreshReshapeRelation(cast<ReshapeOp>(operation), *this);
      break;
    case ContractOperands:
      status = checked(alignContractOperands(operation, *this));
      break;
    case ContractAccumulator:
      status = alignContractAccumulator(operation, *this);
      break;
    case RuleCount:
      llvm_unreachable("invalid relation rule");
    }
    if (failed(status))
      return operation->emitOpError(
          "failed to close its physical value relation");
    if (conflict)
      return failure();
    for (auto &[boundary, before] : boundaries) {
      if (!live.contains(boundary) ||
          llvm::equal(before, boundary->getOperands()))
        continue;
      enqueue(boundary);
      for (Value value : before)
        affected(value);
      for (Value value : boundary->getOperands())
        affected(value);
    }
  }
}

} // namespace intent::gpu::value_relations

namespace intent::gpu {

void eraseDeadPhysicalValues(func::FuncOp kernel) {
  bool changed = false;
  do {
    changed = false;
    kernel.walk<WalkOrder::PostOrder>([&](Operation *operation) {
      if (!operation->getBlock() || operation == kernel.getOperation() ||
          !operation->getNumResults() ||
          !llvm::all_of(operation->getResults(),
                        [](Value value) { return value.use_empty(); }))
        return;
      bool unusedReadOnlyLoop =
          isa<scf::ForOp>(operation) &&
          !operation
               ->walk([](Operation *nested) {
                 if (isa<scf::WhileOp>(nested))
                   return WalkResult::interrupt();
                 return isa<scf::ForOp, scf::IfOp, LoadOp, GatherOp>(nested) ||
                                isMemoryEffectFree(nested)
                            ? WalkResult::advance()
                            : WalkResult::interrupt();
               })
               .wasInterrupted();
      if (isMemoryEffectFree(operation) || isa<LoadOp, GatherOp>(operation) ||
          unusedReadOnlyLoop) {
        operation->erase();
        changed = true;
      }
    });
  } while (changed);
}

LogicalResult closeValueRelations(func::FuncOp kernel,
                                  ValueRelationScope scope) {
  return value_relations::RelationWorklist(kernel, scope).run();
}

} // namespace intent::gpu
