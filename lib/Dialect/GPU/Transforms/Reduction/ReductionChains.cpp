#include "ReductionChains.h"
#include "ReductionAnalysis.h"
#include "ReductionValues.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Value/Helpers.h"
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

bool join(ReduceOp outer) {
  if (outer.getSources().empty()) return false;
  auto inner = outer.getSources().front().getDefiningOp<ReduceOp>();
  if (!inner || inner->getBlock() != outer->getBlock() ||
      !inner->isBeforeInBlock(outer) ||
      !llvm::equal(inner.getResults(), outer.getSources()) ||
      llvm::any_of(inner.getResults(), [](Value value) { return !value.hasOneUse(); }))
    return false;
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
  for (auto [value, result] : llvm::zip(inner.getSources(), outer.getResults())) {
    auto expected = inferCollectiveResultType(value.getType(), axes,
                                               dataElementType(result.getType()));
    if (failed(expected) || *expected != result.getType()) return false;
  }
  // Source values keep their original read epoch. There is no intervening
  // cast, mask, member computation or observable use of the inner summaries.
  OpBuilder builder(outer);
  auto replacement = builder.create<ReduceOp>(outer.getLoc(), inner.getSources(),
      outer.getIdentities(), outer.getCaptures(), axes);
  replacement->setDiscardableAttrs(llvm::to_vector(outer->getDiscardableAttrs()));
  IRMapping mapping;
  outer.getCombine().cloneInto(&replacement.getCombine(), mapping);
  outer.replaceAllUsesWith(replacement.getResults());
  outer.erase();
  inner.erase();
  return true;
}

} // namespace

bool combineNestedReductions(func::FuncOp kernel) {
  bool changed = false;
  while (true) {
    SmallVector<ReduceOp> reductions;
    kernel.walk([&](ReduceOp reduce) { reductions.push_back(reduce); });
    bool joined = false;
    for (ReduceOp reduce : llvm::reverse(reductions))
      if (join(reduce)) { changed = joined = true; break; }
    if (!joined) return changed;
  }
}

} // namespace intent::gpu::reduction
