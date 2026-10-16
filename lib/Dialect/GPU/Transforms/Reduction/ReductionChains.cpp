#include "ReductionChains.h"
#include "ReductionAnalysis.h"
#include "ReductionValues.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Value/Helpers.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OperationSupport.h"
#include <numeric>

using namespace mlir;

namespace intent::gpu::reduction {

// Compare the complete typed tuple callback, including field coupling and
// numeric attributes, independently of its current retained-axis shape.
bool scalarReductionCombine(Region &source, Region &result) {
  SmallVector<Type> arguments, results;
  for (Type type : source.front().getArgumentTypes())
    arguments.push_back(scalarCallbackType(type));
  for (Type type : source.front().getTerminator()->getOperandTypes())
    results.push_back(scalarCallbackType(type));
  std::string reason;
  if (failed(cloneLaneWiseHelper(source, result, arguments, results, reason)))
    return false;
  result.walk([](Operation *operation) { operation->removeAttr(originAttr); });
  return true;
}

bool sameReductionContract(ReduceOp inner, ReduceOp outer, Region &outerCombine) {
  if (inner.getSources().size() != outer.getSources().size() ||
      !llvm::equal(inner.getCaptures(), outer.getCaptures()) ||
      llvm::any_of(inner.getCaptures(), [](Value value) {
        return !value.getType().isIntOrIndexOrFloat();
      })) return false;
  for (auto [a, b] : llvm::zip(inner.getIdentities(), outer.getIdentities()))
    if (!sameScalarValue(a, b)) return false;
  Region innerCombine;
  return scalarReductionCombine(inner.getCombine(), innerCombine) &&
      OperationEquivalence::isRegionEquivalentTo(
          &innerCombine, &outerCombine, OperationEquivalence::IgnoreLocations);
}

namespace {

bool join(ReduceOp outer, bool fullOnly) {
  if (outer.getSources().empty()) return false;
  Value first = outer.getSources().front();
  if (fullOnly)
    if (auto mask = first.getDefiningOp<SelectOp>()) first = mask.getTrueValue();
  auto inner = first.getDefiningOp<ReduceOp>();
  if (!inner || inner->getBlock() != outer->getBlock() ||
      !inner->isBeforeInBlock(outer) ||
      inner.getNumResults() != outer.getSources().size() ||
      llvm::any_of(inner.getResults(), [](Value value) { return !value.hasOneUse(); }))
    return false;
  SmallVector<SelectOp> masks;
  Value predicate;
  for (auto [source, result, identity] :
       llvm::zip(outer.getSources(), inner.getResults(), outer.getIdentities())) {
    if (source == result) {
      masks.push_back({});
      continue;
    }
    auto mask = fullOnly ? source.getDefiningOp<SelectOp>() : SelectOp();
    if (!mask || mask->getBlock() != outer->getBlock() ||
        !mask.getResult().hasOneUse() || mask.getTrueValue() != result ||
        !sameScalarValue(mask.getFalseValue(), identity) ||
        (predicate && predicate != mask.getCondition())) return false;
    predicate = mask.getCondition();
    masks.push_back(mask);
  }
  // Coupled tuple components must select the same complete identity together.
  if (predicate && llvm::any_of(masks, [](SelectOp mask) { return !mask; })) return false;
  Region combine;
  if (!scalarReductionCombine(outer.getCombine(), combine) ||
      !sameReductionContract(inner, outer, combine)) return false;
  auto source = cast<FragmentType>(inner.getSources().front().getType());
  SmallVector<int64_t> retained;
  for (unsigned axis = 0; axis < source.getShape().size(); ++axis)
    if (!llvm::is_contained(inner.getAxes(), axis)) retained.push_back(axis);
  SmallVector<int64_t> axes(inner.getAxes());
  for (int64_t axis : outer.getAxes()) {
    if (axis < 0 || axis >= static_cast<int64_t>(retained.size())) return false;
    axes.push_back(retained[axis]);
  }
  llvm::sort(axes);
  if (fullOnly && axes.size() != source.getShape().size()) return false;
  for (auto [value, result] : llvm::zip(inner.getSources(), outer.getResults())) {
    auto expected = inferCollectiveResultType(value.getType(), axes,
                                               dataElementType(result.getType()));
    if (failed(expected) || *expected != result.getType()) return false;
  }
  // Source values keep their original read epoch. Single-use summaries exclude
  // predicates computed from those summaries. Pull an identity mask over the
  // inner axes without removing it or replaying any member computation.
  OpBuilder builder(outer);
  SmallVector<Value> sources(inner.getSources());
  if (predicate) {
    for (auto [source, identity] : llvm::zip(sources, outer.getIdentities())) {
      auto type = cast<FragmentType>(source.getType());
      if (type.getElementType() != dataElementType(identity.getType())) return false;
    }
    Operation *previous = outer->getPrevNode();
    auto rollback = [&] {
      while (outer->getPrevNode() != previous) outer->getPrevNode()->erase();
    };
    for (auto [index, source] : llvm::enumerate(sources)) {
      auto type = cast<FragmentType>(source.getType());
      auto predicateType = FragmentType::get(type.getContext(), builder.getI1Type(),
          type.getShape(), type.getAxisMaps(), type.getValidity(), type.getOwner());
      auto selected = projectPhysicalValueToSchema(builder, outer.getLoc(), predicate, predicateType);
      auto identity = projectPhysicalValueToSchema(builder, outer.getLoc(),
                                                   outer.getIdentities()[index], type);
      if (failed(selected) || failed(identity)) {
        rollback();
        return false;
      }
      sources[index] = builder.create<SelectOp>(outer.getLoc(), type, *selected, source, *identity);
    }
  }
  auto replacement = builder.create<ReduceOp>(outer.getLoc(), sources,
      outer.getIdentities(), outer.getCaptures(), axes);
  replacement->setDiscardableAttrs(llvm::to_vector(outer->getDiscardableAttrs()));
  IRMapping mapping;
  outer.getCombine().cloneInto(&replacement.getCombine(), mapping);
  outer.replaceAllUsesWith(replacement.getResults());
  outer.erase();
  for (SelectOp mask : masks) if (mask) mask.erase();
  inner.erase();
  return true;
}

} // namespace

bool combineNestedReductions(func::FuncOp kernel, bool fullOnly) {
  bool changed = false;
  while (true) {
    SmallVector<ReduceOp> reductions;
    kernel.walk([&](ReduceOp reduce) { reductions.push_back(reduce); });
    bool joined = false;
    for (ReduceOp reduce : llvm::reverse(reductions))
      if (join(reduce, fullOnly)) { changed = joined = true; break; }
    if (!joined) return changed;
  }
}

} // namespace intent::gpu::reduction
