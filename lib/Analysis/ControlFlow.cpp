#include "Intent/Analysis/ControlFlow.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent {
namespace {

bool targets(RegionSuccessor successor, Value value) {
  if (auto argument = dyn_cast<BlockArgument>(value))
    return successor.getSuccessor() == argument.getOwner()->getParent();
  return successor.isParent();
}

ControlFlowEdgeKind kind(Region *source, Region *target) {
  if (!source)
    return target ? ControlFlowEdgeKind::Entry : ControlFlowEdgeKind::Bypass;
  return target ? ControlFlowEdgeKind::RegionTransfer : ControlFlowEdgeKind::Exit;
}

// MLIR 20 exposes forwarding ranges; their operand offsets preserve slot
// identity even when one SSA value supplies several unrelated carried values.
void appendRegionEdges(ControlFlowEdges &result, Operation *source,
                       Region *sourceRegion, RegionSuccessor successor,
                       OperandRange operands, Value selectedTarget,
                       OpOperand *selectedOperand) {
  if (selectedTarget && !targets(successor, selectedTarget))
    return;
  ValueRange inputs = successor.getSuccessorInputs();
  Region *targetRegion = successor.getSuccessor();
  auto edgeKind = kind(sourceRegion, targetRegion);
  if (inputs.size() != operands.size()) {
    result.complete = false;
    if (selectedTarget)
      result.edges.push_back(
          {nullptr, selectedTarget, sourceRegion, targetRegion, edgeKind});
    return;
  }
  bool found = false;
  for (auto [index, input] : llvm::enumerate(inputs)) {
    if (selectedTarget && selectedTarget != input)
      continue;
    found = true;
    auto &operand = source->getOpOperand(operands.getBeginOperandIndex() + index);
    if (selectedOperand && selectedOperand != &operand)
      continue;
    result.edges.push_back({&operand, input, sourceRegion, targetRegion, edgeKind});
  }
  if (selectedTarget && !found) {
    // Induction variables and other internally produced arguments do not
    // belong to the forwarded successor inputs.
    result.complete = false;
    result.edges.push_back(
        {nullptr, selectedTarget, sourceRegion, targetRegion, edgeKind});
  }
}

void appendTerminatorEdges(ControlFlowEdges &result, Operation *terminator,
                           Value selectedTarget, OpOperand *selectedOperand) {
  auto branch = dyn_cast<RegionBranchTerminatorOpInterface>(terminator);
  if (!branch) {
    result.complete = false;
    return;
  }
  SmallVector<RegionSuccessor> successors;
  SmallVector<Attribute> unknownOperands(terminator->getNumOperands());
  branch.getSuccessorRegions(unknownOperands, successors);
  for (RegionSuccessor successor : successors)
    appendRegionEdges(result, terminator, terminator->getParentRegion(),
                      successor, branch.getSuccessorOperands(successor),
                      selectedTarget, selectedOperand);
}

ControlFlowEdges regionEdges(RegionBranchOpInterface owner, Value selectedTarget,
                             OpOperand *selectedOperand) {
  ControlFlowEdges result{{}, true};
  if (!hasCompleteControlFlowRegions(owner))
    return {};
  if (!selectedOperand || selectedOperand->getOwner() == owner.getOperation()) {
    SmallVector<RegionSuccessor> successors;
    owner.getSuccessorRegions(RegionBranchPoint::parent(), successors);
    for (RegionSuccessor successor : successors)
      appendRegionEdges(result, owner.getOperation(), nullptr, successor,
                        owner.getEntrySuccessorOperands(successor),
                        selectedTarget, selectedOperand);
  }
  if (selectedOperand) {
    if (selectedOperand->getOwner() != owner.getOperation())
      appendTerminatorEdges(result, selectedOperand->getOwner(), {},
                            selectedOperand);
    return result;
  }
  for (Region &region : owner->getRegions()) {
    SmallVector<RegionSuccessor> successors;
    owner.getSuccessorRegions(&region, successors);
    if (!llvm::any_of(successors, [&](RegionSuccessor successor) {
          return targets(successor, selectedTarget);
        }))
      continue;
    for (Block &block : region) {
      if (block.empty() || !block.back().hasTrait<OpTrait::IsTerminator>()) {
        result.complete = false;
        continue;
      }
      Operation *terminator = block.getTerminator();
      if (terminator->getNumSuccessors())
        continue;
      appendTerminatorEdges(result, terminator, selectedTarget, nullptr);
    }
  }
  return result;
}

void appendBranchEdges(ControlFlowEdges &result, BranchOpInterface branch,
                       unsigned successorIndex, BlockArgument selectedTarget,
                       OpOperand *selectedOperand) {
  Block *target = branch->getSuccessor(successorIndex);
  auto operands = branch.getSuccessorOperands(successorIndex);
  if (operands.size() != target->getNumArguments()) {
    result.complete = false;
    return;
  }
  for (BlockArgument argument : target->getArguments()) {
    if (selectedTarget && argument != selectedTarget)
      continue;
    unsigned index = argument.getArgNumber();
    OpOperand *operand =
        operands.isOperandProduced(index)
            ? nullptr
            : &branch->getOpOperand(operands.getOperandIndex(index));
    if (selectedOperand && operand != selectedOperand)
      continue;
    if (!operand)
      result.complete = false;
    result.edges.push_back({operand, argument, branch->getParentRegion(),
                            target->getParent(), ControlFlowEdgeKind::Branch});
  }
}

} // namespace

bool hasCompleteControlFlowRegions(Operation *operation) {
  if (!operation) return false;
  for (Region &region : operation->getRegions()) {
    if (region.empty()) {
      // A result-free scf.if has a legitimately absent else region. Required
      // loop/helper regions cannot be treated as an empty, known control path.
      auto branch = dyn_cast<scf::IfOp>(operation);
      if (branch && !branch.getNumResults() && &region == &branch.getElseRegion())
        continue;
      return false;
    }
    for (Block &block : region)
      if (block.empty() || !block.back().hasTrait<OpTrait::IsTerminator>())
        return false;
  }
  return true;
}

ControlFlowEdges queryControlFlowIncoming(Value target) {
  if (auto argument = dyn_cast<BlockArgument>(target)) {
    Block *block = argument.getOwner();
    ControlFlowEdges result{{}, true};
    if (block->isEntryBlock()) {
      auto owner = dyn_cast<RegionBranchOpInterface>(block->getParentOp());
      if (owner)
        result = regionEdges(owner, target, nullptr);
      else
        result.complete = false;
    }
    llvm::DenseSet<Block *> visitedPredecessors;
    for (Block *predecessor : block->getPredecessors()) {
      if (!visitedPredecessors.insert(predecessor).second)
        continue;
      if (predecessor->empty() ||
          !predecessor->back().hasTrait<OpTrait::IsTerminator>()) {
        result.complete = false;
        continue;
      }
      auto branch = dyn_cast<BranchOpInterface>(predecessor->getTerminator());
      if (!branch) {
        result.complete = false;
        continue;
      }
      for (unsigned index = 0; index < branch->getNumSuccessors(); ++index)
        if (branch->getSuccessor(index) == block)
          appendBranchEdges(result, branch, index, argument, nullptr);
    }
    return result;
  }
  auto owner = dyn_cast_or_null<RegionBranchOpInterface>(target.getDefiningOp());
  return owner ? regionEdges(owner, target, nullptr) : ControlFlowEdges{};
}

ControlFlowEdges queryControlFlowOutgoing(OpOperand &operand) {
  Operation *operation = operand.getOwner();
  if (auto owner = dyn_cast<RegionBranchOpInterface>(operation))
    return regionEdges(owner, {}, &operand);
  if (isa<RegionBranchTerminatorOpInterface>(operation)) {
    auto owner = dyn_cast_or_null<RegionBranchOpInterface>(operation->getParentOp());
    return owner ? regionEdges(owner, {}, &operand) : ControlFlowEdges{};
  }
  if (auto branch = dyn_cast<BranchOpInterface>(operation)) {
    ControlFlowEdges result{{}, true};
    for (unsigned index = 0; index < branch->getNumSuccessors(); ++index)
      appendBranchEdges(result, branch, index, {}, &operand);
    return result;
  }
  return {};
}

} // namespace intent
