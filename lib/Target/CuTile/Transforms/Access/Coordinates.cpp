#include "Coordinates.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Analysis/IndexRelations.h"
#include "Intent/Dialect/GPU/Analysis/ProgramInterface.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/PhysicalExpressions.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Target/CuTile/Analysis/IndexBounds.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

namespace intent::cutile {

namespace {

bool sameScalarFill(Value lhs, Value rhs) {
  if (lhs == rhs)
    return true;
  auto lhsConstant = lhs.getDefiningOp<arith::ConstantOp>();
  auto rhsConstant = rhs.getDefiningOp<arith::ConstantOp>();
  return lhsConstant && rhsConstant && lhs.getType() == rhs.getType() &&
         lhsConstant.getValue() == rhsConstant.getValue();
}

} // namespace

Value uniformScalarFill(Value fill) {
  if (!fill)
    return fill;
  while (isa<gpu::FragmentType>(fill.getType())) {
    if (auto splat = fill.getDefiningOp<gpu::SplatOp>()) {
      fill = splat.getValue();
      continue;
    }
    if (auto broadcast = fill.getDefiningOp<gpu::BroadcastOp>()) {
      fill = broadcast.getValue();
      continue;
    }
    if (auto transpose = fill.getDefiningOp<gpu::TransposeOp>()) {
      fill = transpose.getValue();
      continue;
    }
    if (auto reshape = fill.getDefiningOp<gpu::ReshapeOp>()) {
      fill = reshape.getValue();
      continue;
    }
    if (auto select = fill.getDefiningOp<gpu::SelectOp>()) {
      Value trueFill = uniformScalarFill(select.getTrueValue());
      Value falseFill = uniformScalarFill(select.getFalseValue());
      return trueFill && falseFill && sameScalarFill(trueFill, falseFill)
                 ? trueFill
                 : Value();
    }
    return Value();
  }
  return fill;
}

FailureOr<SmallVector<Value>> orderedCoordinates(gpu::AccessOpInterface access) {
  Operation *owner = access.getOperation();
  auto view = cast<gpu::ViewType>(access.getAccessResource().getType());
  auto coordinates = access.getAccessCoordinates();
  auto sourceAxes = access.getAccessSourceAxes();
  SmallVector<Value> result(view.getRank());
  for (auto [coordinate, sourceAxis] : llvm::zip(coordinates, sourceAxes))
    result[sourceAxis] = coordinate;
  if (llvm::any_of(result, [](Value value) { return !value; }))
    return owner->emitOpError("cuTile advanced access source axes are incomplete");
  return result;
}

FailureOr<SmallVector<Value>> materializeCoordinateDomains(
    OpBuilder &builder, gpu::AccessOpInterface access) {
  Operation *owner = access.getOperation();
  auto sourceAxes = access.getAccessSourceAxes();
  auto view = cast<gpu::ViewType>(access.getAccessResource().getType());
  SmallVector<Value> results(view.getRank());
  for (auto [slot, resourceAxis] : llvm::enumerate(sourceAxes)) {
    auto coordinate = gpu::materializeAccessCoordinate(builder, access, slot);
    if (failed(coordinate))
      return failure();
    results[resourceAxis] = *coordinate;
  }
  if (llvm::any_of(results, [](Value value) { return !value; }))
    return owner->emitOpError("cuTile coordinates do not cover every native resource axis");
  return results;
}

OpBuilder prepareBranch(Region &region) {
  Block &block = region.front();
  if (!block.empty() && isa<scf::YieldOp>(block.back()))
    block.back().erase();
  return OpBuilder(&block, block.end());
}

bool isIdentityPermutation(ArrayRef<int64_t> permutation) {
  return llvm::all_of(llvm::enumerate(permutation),
                      [](auto item) {
                        return static_cast<int64_t>(item.index()) ==
                               item.value();
                      });
}

SmallVector<int64_t> identityAxes(unsigned rank) {
  SmallVector<int64_t> result;
  result.reserve(rank);
  for (unsigned axis = 0; axis < rank; ++axis)
    result.push_back(axis);
  return result;
}

FailureOr<unsigned> nativeAccessRangeAxis(gpu::AccessOpInterface access,
                                         unsigned coordinateIndex,
                                         gpu::MakeRangeOp range) {
  auto projection = gpu::queryAccessCoordinateProjection(access, coordinateIndex);
  auto coordinate = dyn_cast<gpu::FragmentType>(
      access.getAccessCoordinates()[coordinateIndex].getType());
  if (!projection.isExact() || !coordinate)
    return failure();
  auto ranges = gpu::queryRangeProjections(coordinate, range);
  if (ranges.size() != 1)
    return failure();
  std::optional<unsigned> result;
  for (auto [targetAxis, sourceAxis] : llvm::enumerate(projection.targetToSource)) {
    if (!sourceAxis || *sourceAxis != ranges.front().fragmentAxis)
      continue;
    if (result)
      return failure();
    result = targetAxis;
  }
  if (result)
    return *result;
  return failure();
}

} // namespace intent::cutile
