#include "Intent/Dialect/GPU/Analysis/ResourceAlias.h"
#include "Intent/Analysis/ControlFlow.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/ProgramInterface.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::gpu {
namespace {

bool isResource(Value value) {
  return value && isa<ViewType, BufferType>(value.getType());
}

bool isAllocationRoot(Value value) {
  if (value.getDefiningOp<BufferOp>())
    return true;
  auto argument = dyn_cast<BlockArgument>(value);
  auto function = argument
                      ? dyn_cast<func::FuncOp>(argument.getOwner()->getParentOp())
                      : func::FuncOp();
  if (!function || argument.getOwner() != &function.front())
    return false;
  auto binding = getArgumentBinding(argument);
  return binding && (binding.getKind() == ArgumentKind::Workspace ||
                     (binding.getKind() == ArgumentKind::Public &&
                      bool(getPublicView(argument))));
}

bool predecessors(Value value, SmallVectorImpl<Value> &incoming) {
  Operation *definition = value.getDefiningOp();
  if (auto view = dyn_cast_or_null<ViewLikeOpInterface>(definition)) {
    incoming.push_back(view.getViewSource());
    return true;
  }
  if (auto select = dyn_cast_or_null<SelectLikeOpInterface>(definition)) {
    incoming.append({select.getTrueValue(), select.getFalseValue()});
    return true;
  }
  auto edges = queryControlFlowIncoming(value);
  for (const ControlFlowEdge &edge : edges.edges)
    if (edge.operand) incoming.push_back(edge.operand->get());
  return edges.complete && !incoming.empty();
}

bool disjointRoots(Value lhs, Value rhs) {
  if (lhs == rhs)
    return false;
  // Every BufferOp and hidden workspace owns a fresh allocation. A different
  // SSA handle alone is insufficient; both inputs here are proven roots.
  if (lhs.getDefiningOp<BufferOp>() || rhs.getDefiningOp<BufferOp>() ||
      isInvocationWorkspace(lhs) || isInvocationWorkspace(rhs))
    return true;
  auto left = getPublicView(lhs);
  auto right = getPublicView(rhs);
  return left && right && (left.getConstraints().getNoalias() ||
                           right.getConstraints().getNoalias());
}

bool impliesDisjointViews(Value condition, bool truth, Value lhs, Value rhs,
                          llvm::DenseSet<std::pair<Value, unsigned>> &visited) {
  if (!condition.getType().isInteger(1) ||
      !visited.insert({condition, truth}).second)
    return false;
  if (auto overlap = condition.getDefiningOp<ViewOverlapOp>())
    return !truth &&
           ((overlap.getLhs() == lhs && overlap.getRhs() == rhs) ||
            (overlap.getLhs() == rhs && overlap.getRhs() == lhs));
  UniformExpression expression = describeUniformValue(condition);
  if (expression.kind == UniformKind::Not && expression.operands.size() == 1)
    return impliesDisjointViews(expression.operands.front(), !truth, lhs, rhs,
                                visited);
  if (expression.operands.size() != 2) return false;
  if ((expression.kind == UniformKind::And && truth) ||
      (expression.kind == UniformKind::Or && !truth))
    return llvm::any_of(expression.operands, [&](Value operand) {
      return impliesDisjointViews(operand, truth, lhs, rhs, visited);
    });
  if (expression.kind != UniformKind::Compare ||
      (expression.predicate != UniformPredicate::Equal &&
       expression.predicate != UniformPredicate::NotEqual) ||
      !llvm::all_of(expression.operands,
                    [](Value value) { return value.getType().isInteger(1); }))
    return false;
  UniformValueAnalysis uniform(describeUniformValue);
  for (unsigned index = 0; index < 2; ++index) {
    auto constant = uniformBoolean(uniform.evaluate(expression.operands[index]));
    if (!constant) continue;
    bool equal = truth == (expression.predicate == UniformPredicate::Equal);
    if (impliesDisjointViews(expression.operands[1 - index],
                            equal ? *constant : !*constant, lhs, rhs, visited))
      return true;
  }
  return false;
}

} // namespace

const ResourceAliasAnalysis::Roots &ResourceAliasAnalysis::roots(Value value) {
  auto found = cache.find(value);
  if (found != cache.end())
    return found->second;
  Roots result;
  SmallVector<Value> pending{value};
  llvm::DenseSet<Value> visited;
  while (!pending.empty()) {
    Value current = pending.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    if (!isResource(current)) {
      result.complete = false;
      continue;
    }
    if (isAllocationRoot(current)) {
      result.allocations.push_back(current);
      continue;
    }
    SmallVector<Value> incoming;
    if (!predecessors(current, incoming))
      result.complete = false;
    llvm::append_range(pending, incoming);
  }
  result.complete &= !result.allocations.empty();
  return cache.try_emplace(value, std::move(result)).first->second;
}

AliasResult ResourceAliasAnalysis::alias(Value lhs, Value rhs) {
  if (!isResource(lhs) || !isResource(rhs))
    return AliasResult::MayAlias;
  if (lhs == rhs)
    return AliasResult::MustAlias;
  // Querying the second value may grow the DenseMap; do not keep references
  // into it across insertion.
  roots(lhs);
  roots(rhs);
  const Roots &left = cache.find(lhs)->second;
  const Roots &right = cache.find(rhs)->second;
  if (!left.complete || !right.complete)
    return AliasResult::MayAlias;
  for (Value a : left.allocations)
    for (Value b : right.allocations)
      if (!disjointRoots(a, b))
        return AliasResult::MayAlias;
  return AliasResult::NoAlias;
}

bool ResourceAliasAnalysis::disjointAt(Value lhs, Value rhs,
                                     Operation *context) {
  if (!context || !isResource(lhs) || !isResource(rhs) || lhs == rhs)
    return false;
  if (alias(lhs, rhs).isNo()) return true;
  auto kernel = context->getParentOfType<func::FuncOp>();
  auto left = dyn_cast<BlockArgument>(lhs), right = dyn_cast<BlockArgument>(rhs);
  if (!kernel || !left || !right || left.getOwner() != &kernel.front() ||
      right.getOwner() != &kernel.front() || !getPublicView(lhs) ||
      !getPublicView(rhs))
    return false;
  // ViewOverlap describes immutable invocation geometry. Its false value is
  // useful along an executed arm, including for earlier memory effects, but
  // says nothing about overlapping paths or a joined control result.
  for (Region *region = context->getParentRegion(); region;) {
    Operation *owner = region->getParentOp();
    if (!owner || owner == kernel) break;
    if (auto branch = dyn_cast<scf::IfOp>(owner)) {
      bool truth;
      if (region == &branch.getThenRegion()) truth = true;
      else if (region == &branch.getElseRegion()) truth = false;
      else return false;
      llvm::DenseSet<std::pair<Value, unsigned>> visited;
      if (impliesDisjointViews(branch.getCondition(), truth, lhs, rhs, visited))
        return true;
    }
    region = owner->getParentRegion();
  }
  return false;
}

} // namespace intent::gpu
