#include "InputWindows.h"
#include "Intent/Dialect/CPU/Analysis/ExtentRelations.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
namespace intent::cpu {
namespace {

bool panelAligned(OpFoldResult offset, int64_t panel) {
  if (panel == 1) return true;
  llvm::DenseMap<Value, bool> known;
  std::function<bool(OpFoldResult)> aligned = [&](OpFoldResult value) {
    if (auto constant = getConstantIntValue(value)) return *constant % panel == 0;
    if (!llvm::isPowerOf2_64(panel)) return false;
    Value dynamic = cast<Value>(value);
    auto found = known.find(dynamic);
    if (found != known.end()) return found->second;
    bool result = false;
    if (auto argument = dyn_cast<BlockArgument>(dynamic)) {
      if (auto loop = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
          loop && argument == loop.getInductionVar())
        result = aligned(loop.getLowerBound()) && aligned(loop.getStep());
    } else if (auto product = dynamic.getDefiningOp<arith::MulIOp>()) {
      result = aligned(product.getLhs()) || aligned(product.getRhs());
    } else if (auto subtract = dynamic.getDefiningOp<arith::SubIOp>()) {
      auto remainder = subtract.getRhs().getDefiningOp<arith::RemSIOp>();
      auto divisor = remainder ? getConstantIntValue(remainder.getRhs()) : std::nullopt;
      result = remainder && remainder.getLhs() == subtract.getLhs() && divisor &&
          *divisor > 0 && *divisor % panel == 0;
      result |= aligned(subtract.getLhs()) && aligned(subtract.getRhs());
    } else if (auto select = dynamic.getDefiningOp<arith::SelectOp>()) {
      result = aligned(select.getTrueValue()) && aligned(select.getFalseValue());
    } else if (Operation *operation = dynamic.getDefiningOp();
        isa_and_nonnull<arith::AddIOp, arith::MinSIOp, arith::MaxSIOp>(operation)) {
      result = llvm::all_of(operation->getOperands(), [&](Value operand) { return aligned(operand); });
    }
    known[dynamic] = result;
    return result;
  };
  return aligned(offset);
}


} // namespace

std::optional<InputWindowDependencies> inputWindowDependencies(
    ValueRange values, Operation *scope, ValueRange frontier) {
  llvm::SetVector<Operation *> operations;
  llvm::SetVector<Value> usedFrontier;
  std::function<bool(Value)> visit = [&](Value value) {
    if (llvm::is_contained(frontier, value)) {
      usedFrontier.insert(value);
      return true;
    }
    Operation *owner = value.getParentRegion()->getParentOp();
    if (owner != scope && !scope->isAncestor(owner)) return true;
    Operation *definition = value.getDefiningOp();
    if (!definition || definition->getNumRegions() ||
        !isMemoryEffectFree(definition) || !isSpeculatable(definition))
      return false;
    if (operations.contains(definition)) return true;
    if (!llvm::all_of(definition->getOperands(), visit)) return false;
    operations.insert(definition);
    return true;
  };
  if (!llvm::all_of(values, visit)) return std::nullopt;
  return InputWindowDependencies{{operations.begin(), operations.end()},
                                 {usedFrontier.begin(), usedFrontier.end()}};
}

std::optional<ConsumerWindow> consumerWindow(Value source, const InputRequirement &requirement,
                                            Operation *consumer) {
  if (requirement.reuse != InputReuse::Consumers || requirement.panelAxis >= 2 ||
      requirement.panelSize <= 0 || requirement.windowAlignment <= 0) return std::nullopt;
  auto view = source.getDefiningOp<memref::SubViewOp>();
  bool transposed = false;
  if (!view && source.getDefiningOp<memref::AllocOp>()) {
    StorageAnalysis storage(consumer->getParentOfType<func::FuncOp>());
    auto swap = AffineMap::getPermutationMap(ArrayRef<unsigned>{1, 0}, source.getContext());
    for (Operation *user : source.getUsers()) {
      auto producer = dyn_cast<linalg::GenericOp>(user);
      if (!producer || producer.getNumResults() || producer.getInputs().size() != 1 ||
          producer.getOutputs().size() != 1 || producer.getOutputs()[0] != source ||
          producer.getIteratorTypesArray() != SmallVector<utils::IteratorType>{
              utils::IteratorType::parallel, utils::IteratorType::parallel}) continue;
      Block &body = producer.getRegion().front();
      auto maps = producer.getIndexingMapsArray();
      if (body.getOperations().size() != 1 || body.getTerminator()->getOperand(0) != body.getArgument(0) ||
          !maps[0].isPermutation() || !maps[1].isPermutation() ||
          maps[0].compose(inversePermutation(maps[1])) != swap ||
          !storage.unchangedBetween(source, producer, consumer)) continue;
      view = producer.getInputs()[0].getDefiningOp<memref::SubViewOp>();
      if (view) { transposed = true; break; }
    }
  }
  if (!view || view.getType().getRank() != 2 || view.getSourceType().getRank() != 2 ||
      llvm::any_of(view.getMixedStrides(), [](OpFoldResult stride) { return getConstantIntValue(stride) != 1; }))
    return std::nullopt;
  auto full = [&](unsigned axis) {
    return getConstantIntValue(view.getMixedOffsets()[axis]) == 0 &&
        haveEqualExtents(ValueBoundsConstraintSet::Variable(view.getSource(), axis),
                         ValueBoundsConstraintSet::Variable(view.getMixedSizes()[axis]));
  };
  unsigned axis;
  if (full(1)) axis = 0;
  else if (full(0)) axis = 1;
  else return std::nullopt;
  unsigned panelAxis = transposed ? 1 - requirement.panelAxis : requirement.panelAxis;
  if (axis == panelAxis && getConstantIntValue(view.getMixedSizes()[axis]) != 1 &&
      !panelAligned(view.getMixedOffsets()[axis], requirement.windowAlignment)) return std::nullopt;
  return ConsumerWindow{view, axis, transposed};
}

bool hasIndependentWindowCoordinates(memref::SubViewOp window, Operation *scope, Value groupCoordinate) {
  SmallVector<Value> frontier, values;
  if (groupCoordinate) frontier.push_back(groupCoordinate);
  scope->walk([&](scf::ForOp loop) {
    if (loop != scope) frontier.push_back(loop.getInductionVar());
  });
  for (OpFoldResult value : llvm::concat<const OpFoldResult>(
           window.getMixedOffsets(), window.getMixedSizes()))
    if (auto dynamic = dyn_cast<Value>(value)) values.push_back(dynamic);
  // Inner traversals still identify actual source rows. A window that advances
  // directly with this consumer loop does not provide reuse across its iterations.
  return inputWindowDependencies(values, scope, frontier).has_value();
}

std::optional<InputWindowBounds> boundInputWindow(
    const ConsumerWindow &window, Operation *scope,
    const InputRequirement &requirement) {
  using Bounds = ValueBoundsConstraintSet;
  using Variable = Bounds::Variable;
  auto view = window.view;
  DominanceInfo dominance(scope->getParentOfType<func::FuncOp>());
  auto available = [&](Value value, std::optional<int64_t> dimension, Bounds &) {
    if (dimension) {
      auto type = dyn_cast<MemRefType>(value.getType());
      if (!type || *dimension < 0 || *dimension >= type.getRank()) return false;
    } else if (!value.getType().isIndex()) {
      return false;
    }
    return dominance.properlyDominates(value, scope);
  };
  auto offsets = view.getMixedOffsets(), sizes = view.getMixedSizes();
  InputWindowBounds result;
  if (failed(Bounds::computeBound(result.lower, result.lowerOperands,
          presburger::BoundType::LB, Variable(offsets[window.axis]), available)))
    return std::nullopt;
  auto context = scope->getContext();
  auto sum = AffineMap::get(2, 0,
      getAffineDimExpr(0, context) + getAffineDimExpr(1, context));
  Variable end(sum, ArrayRef<Variable>{Variable(offsets[window.axis]),
                                      Variable(sizes[window.axis])});
  if (failed(Bounds::computeBound(result.upper, result.upperOperands,
          presburger::BoundType::UB, end, available, /*closedUB=*/true)))
    return std::nullopt;
  // Keep the implementation's panel origins aligned after rebasing the cache.
  unsigned logicalAxis = window.transposed ? 1 - window.axis : window.axis;
  if (logicalAxis == requirement.panelAxis) {
    AffineExpr lower = result.lower.getResult(0);
    result.lower = AffineMap::get(result.lower.getNumDims(),
        result.lower.getNumSymbols(),
        lower.floorDiv(requirement.panelSize) * requirement.panelSize);
  }
  auto variable = [](AffineMap map, const ValueDimList &operands) {
    SmallVector<Variable> variables;
    for (auto [value, dimension] : operands)
      variables.emplace_back(value, dimension);
    return Variable(map, variables);
  };
  Variable lower = variable(result.lower, result.lowerOperands);
  Variable upper = variable(result.upper, result.upperOperands);
  Variable zero(IntegerAttr::get(IndexType::get(context), 0));
  if (!Bounds::compare(lower, Bounds::GE, zero) ||
      !Bounds::compare(lower, Bounds::LE, upper))
    return std::nullopt;
  return result;
}

Value materializeInputWindowBound(OpBuilder &builder, Location location,
    AffineMap bound, const ValueDimList &operands) {
  SmallVector<Value> values;
  for (auto [value, dimension] : operands)
    values.push_back(dimension ? builder.createOrFold<memref::DimOp>(
                                    location, value, *dimension) : value);
  auto expanded = affine::expandAffineMap(builder, location, bound, values);
  assert(expanded && expanded->size() == 1 && "computed affine window bound");
  return expanded->front();
}


} // namespace intent::cpu
