#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Analysis/IndexPredicates.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;

namespace intent::gpu {

LogicalResult simplifyRangePredicates(ModuleOp module) {
  auto kernel = getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  SmallVector<std::pair<CompareOp, bool>> proven;
  SmallVector<std::pair<CompareOp, IndexComparisonBound>> conditional;
  kernel->walk([&](CompareOp comparison) {
    if (auto value = proveRangeComparison(comparison))
      proven.emplace_back(comparison, *value);
    else if (auto bounds = queryIndexComparisonBound(comparison))
      conditional.emplace_back(comparison, *bounds);
  });
  for (auto [comparison, value] : proven) {
    OpBuilder builder(comparison);
    Value replacement = builder.create<arith::ConstantIntOp>(
        comparison.getLoc(), value, 1);
    if (auto fragment = dyn_cast<FragmentType>(comparison.getType()))
      replacement = builder.create<SplatOp>(comparison.getLoc(), fragment, replacement);
    replacement.getDefiningOp()->setDiscardableAttrs(
        llvm::to_vector(comparison->getDiscardableAttrs()));
    comparison.getResult().replaceAllUsesWith(replacement);
    comparison.erase();
  }
  for (auto [comparison, bounds] : conditional) {
    // Keep the original predicate when the runtime specialization cannot prove
    // the stronger shape relationship. No input assumption is introduced.
    OpBuilder builder(comparison);
    auto loc = comparison.getLoc();
    Value upper = builder.create<PhysicalExprOp>(loc, builder.getIndexType(), bounds.upper);
    Value limit = builder.create<PhysicalExprOp>(loc, builder.getIndexType(), bounds.limit);
    Value proof = builder.create<CompareOp>(loc, builder.getI1Type(), upper, limit,
                                           ComparePredicate::Lt);
    builder.setInsertionPointAfter(comparison);
    auto refined = builder.create<BinaryOp>(loc, builder.getI1Type(), proof,
        comparison.getResult(), BinaryOperator::LogicalOr);
    comparison.getResult().replaceAllUsesExcept(refined.getResult(), refined);
  }
  return success();
}

} // namespace intent::gpu
