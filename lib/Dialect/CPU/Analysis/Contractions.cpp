#include "Intent/Dialect/CPU/Analysis/Contractions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace intent::cpu {
namespace {

bool supportedBody(linalg::GenericOp operation) {
  Block &body = operation.getRegion().front();
  auto fma = body.getTerminator()->getOperand(0).getDefiningOp<math::FmaOp>();
  if (fma) {
    if (fma.getC() != body.getArgument(2)) return false;
    llvm::SmallPtrSet<Operation *, 4> computation{fma};
    auto input = [&](Value value, Value argument) {
      if (value == argument) return true;
      auto widen = value.getDefiningOp<arith::ExtFOp>();
      if (!widen || widen.getIn() != argument || widen.getType() != fma.getType()) return false;
      computation.insert(widen);
      return true;
    };
    return input(fma.getA(), body.getArgument(0)) && input(fma.getB(), body.getArgument(1)) &&
        computation.size() == static_cast<size_t>(std::distance(body.begin(), body.end()) - 1);
  }
  auto add = body.getTerminator()->getOperand(0).getDefiningOp<arith::AddIOp>();
  if (!add || !body.getArgument(0).getType().isSignlessInteger(8) ||
      !body.getArgument(1).getType().isSignlessInteger(8) ||
      !body.getArgument(2).getType().isSignlessInteger(32) ||
      std::distance(body.begin(), body.end()) != 5 || add.getRhs() != body.getArgument(2)) return false;
  auto product = add.getLhs().getDefiningOp<arith::MulIOp>();
  if (!product) return false;
  auto lhs = product.getLhs().getDefiningOp<arith::ExtSIOp>();
  auto rhs = product.getRhs().getDefiningOp<arith::ExtSIOp>();
  return lhs && rhs && lhs.getIn() == body.getArgument(0) && rhs.getIn() == body.getArgument(1);
}

} // namespace

std::optional<ContractionAxes> queryContractionAxes(linalg::GenericOp operation) {
  if (operation.getInputs().size() != 2 || operation.getOutputs().size() != 1 ||
      operation.getNumResults() || !supportedBody(operation)) return std::nullopt;
  auto maps = operation.getIndexingMapsArray();
  SmallVector<SmallVector<std::optional<unsigned>>> projections;
  for (AffineMap map : ArrayRef(maps).take_front(2)) {
    if (map.getNumSymbols()) return std::nullopt;
    auto &axes = projections.emplace_back(operation.getNumLoops());
    for (auto [axis, expression] : llvm::enumerate(map.getResults())) {
      auto dim = dyn_cast<AffineDimExpr>(expression);
      if (!dim || axes[dim.getPosition()]) return std::nullopt;
      axes[dim.getPosition()] = axis;
    }
  }
  SmallVector<int64_t> reduction;
  for (auto [loop, iterator] : llvm::enumerate(operation.getIteratorTypesArray())) {
    if (iterator == utils::IteratorType::reduction) reduction.push_back(loop);
    else if (iterator != utils::IteratorType::parallel) return std::nullopt;
  }
  // Explicit contraction maps already bind every operand axis. Do not squeeze
  // their singleton axes: K=1 and unit M/N remain valid paired dimensions.
  auto relation = ProductContractionAxes::get(projections[0],
      SmallVector<bool>(maps[0].getNumResults(), false), projections[1],
      SmallVector<bool>(maps[1].getNumResults(), false), reduction);
  if (!relation || maps[2].getNumSymbols() || relation->axes.results.size() != maps[2].getNumResults())
    return std::nullopt;
  for (auto [position, productAxis] : llvm::enumerate(relation->resultProductAxes))
    if (maps[2].getResult(position) != getAffineDimExpr(productAxis, operation.getContext())) return std::nullopt;
  return std::move(relation->axes);
}

} // namespace intent::cpu
