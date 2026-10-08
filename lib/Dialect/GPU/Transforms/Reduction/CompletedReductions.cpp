#include "CompletedReductions.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include <functional>

using namespace mlir;

namespace intent::gpu::reduction {
namespace {

bool independentMember(Value value, scf::ForOp loop,
                       llvm::DenseSet<Value> &visited) {
  if (llvm::is_contained(loop.getRegionIterArgs(), value)) return false;
  if (!visited.insert(value).second) return true;
  Operation *definition = value.getDefiningOp();
  if (!definition || !loop->isAncestor(definition)) return true;
  if (definition->getNumRegions()) return false;
  return llvm::all_of(definition->getOperands(), [&](Value operand) {
    return independentMember(operand, loop, visited);
  });
}

bool neutralMember(Value value, Value identity,
    ArrayRef<std::pair<MakeRangeOp, Value>> tails,
    PhysicalProgramAnalysis &analysis) {
  if (sameScalarValue(value, identity)) return true;
  UniformValueAnalysis constants(describeUniformValue);
  Attribute expected = constants.evaluate(identity);
  if (!expected) return false;
  UniformBindings bindings;
  SmallVector<Value> pending{value};
  llvm::DenseSet<Value> visited;
  while (!pending.empty()) {
    Value current = pending.pop_back_val();
    if (!visited.insert(current).second) continue;
    Operation *operation = current.getDefiningOp();
    // Bind only an actual upper-tail comparison. A constant true predicate is
    // also accepted by isTailPredicate, but does not become false in padding.
    if (auto compare = dyn_cast_or_null<CompareOp>(operation);
        compare && compare.getPredicate() == ComparePredicate::Lt &&
        analysis.isTailPredicate(current, tails)) {
      bindings[current] = IntegerAttr::get(IntegerType::get(value.getContext(), 1), 0);
      continue;
    }
    if (operation && !operation->getNumRegions())
      llvm::append_range(pending, operation->getOperands());
  }
  return equalUniformConstants(constants.evaluate(value, bindings), expected);
}

} // namespace

std::optional<CompletedReduction> queryCompletedReduction(ReduceOp reduce) {
  if (reduce.getSources().empty()) return std::nullopt;
  auto loop = reduce.getSources().front().getDefiningOp<scf::ForOp>();
  if (!loop || !llvm::equal(reduce.getSources(), loop.getResults()) ||
      loop.getNumRegionIterArgs() != reduce.getSources().size()) return std::nullopt;
  for (auto [initial, identity] : llvm::zip(loop.getInitArgs(), reduce.getIdentities()))
    if (!sameScalarValue(initial, identity)) return std::nullopt;
  SmallVector<Value> members(reduce.getSources().size());
  llvm::SmallPtrSet<Operation *, 32> updates;
  if (!matchReductionCombine(reduce, loop.getBody(), loop.getRegionIterArgs(), members,
          cast<scf::YieldOp>(loop.getBody()->getTerminator()).getOperands(), updates))
    return std::nullopt;
  for (Value member : members) {
    llvm::DenseSet<Value> visited;
    if (!independentMember(member, loop, visited)) return std::nullopt;
  }
  auto kernel = reduce->getParentOfType<func::FuncOp>();
  PhysicalProgramAnalysis analysis(kernel);
  auto type = dyn_cast<FragmentType>(reduce.getSources().front().getType());
  if (!type) return std::nullopt;
  for (unsigned axis = 0; axis < type.getShape().size(); ++axis) {
    if (queryLaunchExpression(loop.getStep()) != type.getShape()[axis]) continue;
    SmallVector<std::pair<MakeRangeOp, Value>> tails;
    bool exact = true;
    for (Value member : members) {
      auto memberType = dyn_cast<FragmentType>(member.getType());
      if (!memberType || memberType.getShape() != type.getShape() ||
          memberType.getAxisMaps() != type.getAxisMaps()) { exact = false; break; }
      auto ranges = analysis.axisRanges(member, axis);
      if (!ranges.blockers.empty()) { exact = false; break; }
      for (MakeRangeOp range : ranges.roots) {
        if (!isUnitStepRange(range) ||
            !samePhysicalScalarExpression(range.getStart(), loop.getInductionVar()) ||
            !samePhysicalScalarExpression(range.getExtent(), loop.getStep()) ||
            !samePhysicalScalarExpression(range.getLogicalStop(), loop.getUpperBound())) {
          exact = false; break;
        }
        if (!llvm::is_contained(tails, std::pair<MakeRangeOp, Value>{range, loop.getUpperBound()}))
          tails.emplace_back(range, loop.getUpperBound());
      }
      if (!exact) break;
    }
    if (!exact || tails.empty()) continue;
    if (llvm::all_of(llvm::zip(members, reduce.getIdentities()), [&](auto component) {
          return neutralMember(std::get<0>(component), std::get<1>(component), tails, analysis);
        }))
      return CompletedReduction{loop, std::move(members), axis};
  }
  return std::nullopt;
}

} // namespace intent::gpu::reduction
