#include "ReductionRealization.h"
#include "ReductionValues.h"
#include "CompletedReductions.h"
#include "ReductionChains.h"
#include "Intent/Dialect/GPU/Analysis/Helpers.h"
#include "Intent/Dialect/GPU/Analysis/IndexRelations.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/PhysicalExpressions.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Value/Helpers.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/MathExtras.h"
#include <functional>
#include <numeric>

using namespace mlir;

namespace intent::gpu::reduction {
namespace {

struct CoverageCheck {
  Value start, stop, extent;
};

bool sameCheck(const CoverageCheck &a, const CoverageCheck &b) {
  return samePhysicalScalarExpression(a.start, b.start) &&
         samePhysicalScalarExpression(a.stop, b.stop) &&
         samePhysicalScalarExpression(a.extent, b.extent);
}

bool completeRange(MakeRangeOp range, Operation *anchor,
    DominanceInfo &dominance, IndexRelations &relations,
    SmallVectorImpl<CoverageCheck> &checks) {
  if (!isUnitStepRange(range) ||
      !samePhysicalScalarExpression(range.getStart(), range.getLogicalStart()))
    return false;
  if (relations.constant(range.getLogicalStart()) == 0 &&
      samePhysicalScalarExpression(range.getLogicalStop(), range.getExtent())) return true;
  auto length = constantLogicalRangeCardinality(range);
  auto extent = relations.constant(range.getExtent());
  if (length && extent && *length == *extent && *extent > 0) return true;
  // The subtraction in the guard is exact signed index arithmetic. Nothing
  // from a completed traversal body is rematerialized at its final consumer.
  CoverageCheck check{range.getLogicalStart(), range.getLogicalStop(), range.getExtent()};
  if (!relations.nonnegative(check.start) || !relations.atMost(check.start, check.stop) ||
      !llvm::all_of(ArrayRef<Value>{check.start, check.stop, check.extent}, [&](Value value) {
        return dominance.dominates(value, anchor) && bool(queryLaunchExpression(value));
      })) return false;
  if (llvm::none_of(checks, [&](const CoverageCheck &old) { return sameCheck(old, check); }))
    checks.push_back(check);
  return true;
}

bool completeAxis(ValueRange sources, unsigned axis, PhysicalExprAttr extent,
    Operation *anchor, PhysicalProgramAnalysis &analysis, DominanceInfo &dominance,
    IndexRelations &relations, SmallVectorImpl<CoverageCheck> &checks) {
  SmallVector<MakeRangeOp> roots;
  for (Value source : sources) {
    auto ranges = analysis.axisRanges(source, axis);
    if (!ranges.blockers.empty() || (!ranges.roots.empty() && !ranges.isExact() &&
                                    !analysis.lockstepRanges(ranges.roots).isExact())) return false;
    for (MakeRangeOp range : ranges.roots) {
      if (range.getResult().getType().getShape()[0] != extent) return false;
      if (!llvm::is_contained(roots, range)) roots.push_back(range);
    }
  }
  return !roots.empty() && llvm::all_of(roots, [&](MakeRangeOp range) {
    return completeRange(range, anchor, dominance, relations, checks);
  });
}

bool completeMask(SelectOp select, Operation *anchor,
    PhysicalProgramAnalysis &analysis, DominanceInfo &dominance,
    IndexRelations &relations, SmallVectorImpl<CoverageCheck> &checks) {
  SmallVector<Value> pending{select.getCondition()};
  llvm::DenseSet<Value> visited;
  SmallVector<std::pair<MakeRangeOp, Value>> tails;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    if (!visited.insert(value).second) continue;
    Operation *operation = value.getDefiningOp();
    if (auto range = dyn_cast_or_null<MakeRangeOp>(operation)) {
      // Tail predicates may also contain the standard coordinate >= 0 check.
      // Complete cardinality alone does not make that lower bound true.
      if (!relations.nonnegative(range.getStart()) ||
          !completeRange(range, anchor, dominance, relations, checks)) return false;
      tails.emplace_back(range, range.getLogicalStop());
      continue;
    }
    if (operation && !operation->getNumRegions() && isMemoryEffectFree(operation))
      llvm::append_range(pending, operation->getOperands());
  }
  return !tails.empty() && analysis.isTailPredicate(select.getCondition(), tails);
}

bool sameCardinality(Value value, const CoverageCheck &check, IndexRelations &relations) {
  if (relations.constant(check.start) == 0)
    return samePhysicalScalarExpression(value, check.stop);
  if (auto difference = value.getDefiningOp<BinaryOp>();
      difference && difference.getOperatorKind() == BinaryOperator::Subtract)
    return samePhysicalScalarExpression(difference.getLhs(), check.stop) &&
           samePhysicalScalarExpression(difference.getRhs(), check.start);
  return false;
}

// A false branch of an existing subset of these equalities cannot satisfy the
// new conjunction. Recognize the actual control rather than marking the IR,
// so repeated standalone runs do not nest the same specialization forever.
bool excludedByEnclosingGuard(Operation *anchor, ArrayRef<CoverageCheck> checks,
                              ArrayRef<PhysicalExprAttr> unitExtents,
                              IndexRelations &relations) {
  std::function<bool(Value)> contained = [&](Value value) {
    if (auto conjunction = value.getDefiningOp<BinaryOp>();
        conjunction && conjunction.getOperatorKind() == BinaryOperator::LogicalAnd)
      return contained(conjunction.getLhs()) && contained(conjunction.getRhs());
    auto compare = value.getDefiningOp<CompareOp>();
    if (!compare || compare.getPredicate() != ComparePredicate::Eq) return false;
    bool coverage = llvm::any_of(checks, [&](const CoverageCheck &check) {
      return (samePhysicalScalarExpression(compare.getRhs(), check.extent) &&
              sameCardinality(compare.getLhs(), check, relations)) ||
             (samePhysicalScalarExpression(compare.getLhs(), check.extent) &&
              sameCardinality(compare.getRhs(), check, relations));
    });
    auto unit = [&](Value extent, Value one) {
      return relations.constant(one) == 1 &&
             llvm::is_contained(unitExtents, queryLaunchExpression(extent));
    };
    return coverage || unit(compare.getLhs(), compare.getRhs()) ||
           unit(compare.getRhs(), compare.getLhs());
  };
  for (Operation *current = anchor; Operation *parent = current->getParentOp(); current = parent)
    if (auto branch = dyn_cast<scf::IfOp>(parent);
        branch && current->getParentRegion() == &branch.getElseRegion() &&
        contained(branch.getCondition())) return true;
  return false;
}

struct StaticCollapse {
  unsigned axis;
  SmallVector<FragmentType> types;
  ArrayAttr groups;
};

// Keep the existing ordered-reshape domain: complete, already physicalized
// constant axes in one contiguous group. This proof does not infer completed
// loop partials or discard identity masks. In that domain the retained axes do
// not add a new layout obligation compared with the existing native program.
std::optional<StaticCollapse> staticCollapse(ReduceOp first, ReduceOp outer,
    ArrayRef<unsigned> reduced, func::FuncOp kernel,
    PhysicalProgramAnalysis &analysis, IndexRelations &relations) {
  for (unsigned i = 1; i < reduced.size(); ++i)
    if (reduced[i] != reduced.front() + i) return std::nullopt;
  auto original = cast<FragmentType>(first.getSources().front().getType());
  int64_t capacity = 1;
  for (unsigned axis : reduced) {
    auto extent = constantPhysicalExpression(cast<PhysicalExprAttr>(original.getShape()[axis]));
    if (!extent || *extent <= 0 || llvm::MulOverflow(capacity, *extent, capacity))
      return std::nullopt;
    for (Value source : first.getSources()) {
      auto realization = analysis.axisRealization(source, axis);
      auto ranges = analysis.axisRanges(source, axis);
      if (!realization.isExact() || !realization.physicalized ||
          realization.constructionScalarSeed || !ranges.isExact() ||
          !ranges.blockers.empty() || ranges.roots.empty() ||
          !llvm::all_of(ranges.roots, [&](MakeRangeOp range) {
            return isUnitStepRange(range) &&
                range.getResult().getType().getShape()[0] == original.getShape()[axis] &&
                samePhysicalScalarExpression(range.getStart(), range.getLogicalStart()) &&
                relations.atMost(range.getLogicalStart(), range.getLogicalStop()) &&
                constantLogicalRangeCardinality(range) == extent;
          })) return std::nullopt;
    }
  }
  MLIRContext *context = kernel.getContext();
  auto [sourceId, dimensionId] = nextPhysicalAxisIdentities(kernel);
  unsigned mergedAxis = reduced.front();
  SmallVector<Attribute> shape, mappings, groups;
  for (unsigned axis = 0; axis < original.getShape().size(); ++axis) {
    unsigned targetAxis = shape.size();
    if (axis == mergedAxis) {
      shape.push_back(expression(context, PhysicalExprKind::Constant, capacity));
      mappings.push_back(AxisMapAttr::get(context, sourceId, 0, dimensionId, targetAxis, true));
      SmallVector<int64_t> axes(reduced.begin(), reduced.end());
      groups.push_back(ReshapeGroupAttr::get(context, DenseI64ArrayAttr::get(context, axes),
          DenseI64ArrayAttr::get(context, {static_cast<int64_t>(targetAxis)})));
      axis = reduced.back();
    } else {
      shape.push_back(original.getShape()[axis]);
      auto mapping = cast<AxisMapAttr>(original.getAxisMaps()[axis]);
      mappings.push_back(AxisMapAttr::get(context, mapping.getSourceId(), mapping.getSourceAxis(),
          mapping.getDimensionId(), targetAxis, mapping.getDerived()));
      groups.push_back(ReshapeGroupAttr::get(context,
          DenseI64ArrayAttr::get(context, {static_cast<int64_t>(axis)}),
          DenseI64ArrayAttr::get(context, {static_cast<int64_t>(targetAxis)})));
    }
  }
  StaticCollapse plan{mergedAxis, {}, ArrayAttr::get(context, groups)};
  for (auto [source, result] : llvm::zip(first.getSources(), outer.getResults())) {
    auto type = cast<FragmentType>(source.getType());
    auto collapsed = FragmentType::get(context, type.getElementType(),
        ArrayAttr::get(context, shape), ArrayAttr::get(context, mappings),
        type.getValidity(), type.getOwner());
    auto expected = inferCollectiveResultType(collapsed,
        ArrayRef<int64_t>{static_cast<int64_t>(mergedAxis)}, dataElementType(result.getType()));
    if (failed(expected) || *expected != result.getType()) return std::nullopt;
    plan.types.push_back(collapsed);
  }
  return plan;
}

bool collapseChain(ReduceOp outer, func::FuncOp kernel) {
  if (outer.getAxes().size() != 1 || outer.getSources().empty())
    return false;
  Region outerCombine;
  if (!scalarReductionCombine(outer.getCombine(), outerCombine))
    return false;

  PhysicalProgramAnalysis analysis(kernel);
  IndexRelations relations;
  DominanceInfo dominance(kernel);
  SmallVector<CoverageCheck> checks;
  SmallVector<SelectOp> masks;
  SmallVector<ReduceOp> chain{outer};
  while (true) {
    ReduceOp consumer = chain.back();
    Value input = consumer.getSources().front();
    if (auto select = input.getDefiningOp<SelectOp>()) input = select.getTrueValue();
    auto inner = input.getDefiningOp<ReduceOp>();
    if (!inner || inner.getAxes().size() != 1 || inner->getBlock() != outer->getBlock() ||
        !inner->isBeforeInBlock(consumer) ||
        inner.getNumResults() != consumer.getSources().size() ||
        llvm::any_of(inner.getResults(), [](Value value) { return !value.hasOneUse(); }) ||
        !sameReductionContract(inner, outer, outerCombine))
      break;
    SmallVector<SelectOp> currentMasks;
    SmallVector<CoverageCheck> currentChecks(checks);
    bool connected = true;
    for (auto [source, result, identity] :
         llvm::zip(consumer.getSources(), inner.getResults(), consumer.getIdentities())) {
      if (source == result) continue;
      auto select = source.getDefiningOp<SelectOp>();
      if (!select || select->getBlock() != outer->getBlock() ||
          !select.getResult().hasOneUse() || select.getTrueValue() != result ||
          !sameScalarValue(select.getFalseValue(), identity) ||
          !completeMask(select, outer, analysis, dominance, relations, currentChecks)) {
        connected = false;
        break;
      }
      currentMasks.push_back(select);
    }
    if (!connected) break;
    checks = std::move(currentChecks);
    llvm::append_range(masks, currentMasks);
    chain.push_back(inner);
  }
  if (chain.size() < 2)
    return false;

  ReduceOp first = chain.back();
  auto original = dyn_cast<FragmentType>(first.getSources().front().getType());
  if (!original)
    return false;
  SmallVector<unsigned> remaining(original.getShape().size());
  std::iota(remaining.begin(), remaining.end(), 0);
  SmallVector<unsigned> reduced;
  for (ReduceOp reduce : llvm::reverse(chain)) {
    unsigned axis = reduce.getAxes().front();
    if (axis >= remaining.size())
      return false;
    reduced.push_back(remaining[axis]);
    remaining.erase(remaining.begin() + axis);
  }
  llvm::sort(reduced);

  // Only a scalar reduction domain exposes the native whole-tuple primitive.
  // Merging across a nonunit retained axis would require an order-preserving
  // reshape that can introduce layout work between the completed traversal and
  // the tree. Keep that domain's original trees, including every configuration.
  SmallVector<PhysicalExprAttr> unitExtents;
  bool scalarDomain = true;
  for (unsigned axis : remaining) {
    auto extent = cast<PhysicalExprAttr>(original.getShape()[axis]);
    auto bounds = queryPositiveExtentBounds(extent, kernel);
    if (!bounds || bounds->first > 1) {
      scalarDomain = false;
      break;
    }
    if (bounds->second != 1 && !llvm::is_contained(unitExtents, extent))
      unitExtents.push_back(extent);
  }
  SmallVector<Value> identities;
  for (Value identity : outer.getIdentities()) {
    auto scalar = scalarSource(identity);
    if (failed(scalar) || !(*scalar).getType().isIntOrIndexOrFloat()) {
      scalarDomain = false;
      break;
    }
    identities.push_back(*scalar);
  }

  for (auto [source, result] : llvm::zip(first.getSources(), outer.getResults())) {
    auto type = dyn_cast<FragmentType>(source.getType());
    if (!type || type.getShape() != original.getShape() ||
        type.getAxisMaps() != original.getAxisMaps() ||
        type.getValidity() != original.getValidity() ||
        type.getOwner() != original.getOwner() ||
        type.getElementType() != dataElementType(result.getType()))
      return false;
  }
  auto ordered = masks.empty() ? staticCollapse(first, outer, reduced, kernel, analysis, relations)
                               : std::nullopt;
  if (!scalarDomain && !ordered) return false;
  if (!scalarDomain) unitExtents.clear();
  // Direct ordinary trees reduce exactly the same physical member set. Their
  // associativity/commutativity contract does not depend on how the input SSA
  // was produced, including nested traversal results. Only removing intervening
  // identity masks needs the additional logical/completed-source proof.
  if (!masks.empty()) {
    auto completed = queryCompletedReduction(first);
    ValueRange coverageSources = completed ? ValueRange(completed->members) : first.getSources();
    for (unsigned axis : reduced) {
      if (completed && completed->axis == axis) continue;
      if (!completeAxis(coverageSources, axis,
              cast<PhysicalExprAttr>(original.getShape()[axis]), outer,
              analysis, dominance, relations, checks)) return false;
    }
  }
  if (scalarDomain && excludedByEnclosingGuard(outer, checks, unitExtents, relations)) return false;
  int64_t maximumCapacity = 1;
  if (scalarDomain) for (Attribute attribute : original.getShape()) {
    auto extent = cast<PhysicalExprAttr>(attribute);
    auto bounds = queryPositiveExtentBounds(extent, kernel);
    if (!bounds || llvm::MulOverflow(maximumCapacity, bounds->second, maximumCapacity)) {
      scalarDomain = false;
      unitExtents.clear();
      break;
    }
  }
  if (!scalarDomain && !ordered) return false;

  SmallVector<int64_t> allAxes(original.getShape().size());
  std::iota(allAxes.begin(), allAxes.end(), 0);

  // No source computation or memory access moves. The equality branch changes
  // only native trees and never reinterprets one field's mask as a tuple identity.
  OpBuilder builder(outer);
  auto orderedCollapse = [&](OpBuilder &at) {
    IRMapping mapping;
    for (auto [source, oldInput, type] :
         llvm::zip(first.getSources(), outer.getSources(), ordered->types)) {
      Value value = at.create<ReshapeOp>(outer.getLoc(), type, source, ordered->groups);
      mapping.map(oldInput, value);
    }
    auto replacement = cast<ReduceOp>(at.clone(*outer, mapping));
    replacement.setAxesAttr(at.getDenseI64ArrayAttr({static_cast<int64_t>(ordered->axis)}));
    return SmallVector<Value>(replacement.getResults());
  };
  auto collapsed = [&](OpBuilder &at) {
    auto replacement = at.create<ReduceOp>(outer.getLoc(), first.getSources(),
        identities, outer.getCaptures(), allAxes);
    replacement->setDiscardableAttrs(llvm::to_vector(outer->getDiscardableAttrs()));
    IRMapping mapping;
    outerCombine.cloneInto(&replacement.getCombine(), mapping);
    SmallVector<Value> values;
    for (auto [value, result] : llvm::zip(replacement.getResults(), outer.getResults())) {
      if (isa<FragmentType>(result.getType()))
        values.push_back(at.create<BroadcastOp>(outer.getLoc(), result.getType(), value));
      else values.push_back(value);
    }
    return values;
  };
  llvm::SmallPtrSet<Operation *, 16> selected;
  for (ReduceOp reduce : chain) selected.insert(reduce);
  for (SelectOp mask : masks) selected.insert(mask);
  SmallVector<Operation *> originals;
  for (Operation &operation : *outer->getBlock())
    if (selected.contains(&operation)) originals.push_back(&operation);
  SmallVector<Value> results;
  if (!scalarDomain) {
    results = orderedCollapse(builder);
  } else if (checks.empty() && unitExtents.empty()) {
    results = collapsed(builder);
  } else {
    Value guard;
    for (const CoverageCheck &check : checks) {
      Value length = relations.constant(check.start) == 0 ? check.stop
          : builder.create<BinaryOp>(outer.getLoc(), check.stop.getType(),
                                    check.stop, check.start, BinaryOperator::Subtract).getResult();
      Value equal = builder.create<CompareOp>(outer.getLoc(), builder.getI1Type(),
                                              length, check.extent, ComparePredicate::Eq);
      guard = guard ? Value(builder.create<BinaryOp>(outer.getLoc(), builder.getI1Type(),
                                  guard, equal, BinaryOperator::LogicalAnd)) : equal;
    }
    for (PhysicalExprAttr extent : unitExtents) {
      Value size = builder.create<PhysicalExprOp>(outer.getLoc(), builder.getIndexType(), extent);
      Value one = builder.create<arith::ConstantIndexOp>(outer.getLoc(), 1);
      Value equal = builder.create<CompareOp>(outer.getLoc(), builder.getI1Type(),
                                              size, one, ComparePredicate::Eq);
      guard = guard ? Value(builder.create<BinaryOp>(outer.getLoc(), builder.getI1Type(),
                                  guard, equal, BinaryOperator::LogicalAnd)) : equal;
    }
    auto branch = builder.create<scf::IfOp>(outer.getLoc(), outer.getResultTypes(), guard, true);
    OpBuilder thenBuilder = OpBuilder::atBlockBegin(&branch.getThenRegion().front());
    auto values = collapsed(thenBuilder);
    thenBuilder.create<scf::YieldOp>(outer.getLoc(), values);
    OpBuilder elseBuilder = OpBuilder::atBlockBegin(&branch.getElseRegion().front());
    if (ordered) values = orderedCollapse(elseBuilder);
    else {
      IRMapping mapping;
      for (Operation *operation : originals) elseBuilder.clone(*operation, mapping);
      values.clear();
      for (Value result : outer.getResults()) values.push_back(mapping.lookup(result));
    }
    elseBuilder.create<scf::YieldOp>(outer.getLoc(), values);
    results.assign(branch.getResults().begin(), branch.getResults().end());
  }
  for (auto [before, after] : llvm::zip(outer.getResults(), results))
    before.replaceAllUsesWith(after);
  for (Operation *operation : llvm::reverse(originals)) operation->erase();
  return true;
}

} // namespace

bool normalizeCompletedReductions(func::FuncOp kernel) {
  bool changed = false;
  while (true) {
    SmallVector<ReduceOp> reductions;
    kernel.walk([&](ReduceOp reduce) { reductions.push_back(reduce); });
    bool rewritten = false;
    for (ReduceOp reduce : llvm::reverse(reductions))
      if (collapseChain(reduce, kernel)) {
        rewritten = changed = true;
        break;
      }
    if (!rewritten)
      return changed;
  }
}

} // namespace intent::gpu::reduction
