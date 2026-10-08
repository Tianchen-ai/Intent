#include "Intent/Target/CuTile/Analysis/IndexBounds.h"
#include "Intent/Target/CuTile/IR/CuTileOps.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/IndexRelations.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
namespace intent::cutile {

bool isProvably(Value value, int64_t expected) {
  std::optional<int64_t> actual = gpu::IndexRelations().constant(value);
  return actual && *actual == expected;
}

Value stripIndexIdentities(Value value) {
  while (auto binary = value.getDefiningOp<gpu::BinaryOp>()) {
    switch (binary.getOperatorKind()) {
    case BinaryOperator::Add:
      if (isProvably(binary.getLhs(), 0)) {
        value = binary.getRhs();
        continue;
      }
      if (isProvably(binary.getRhs(), 0)) {
        value = binary.getLhs();
        continue;
      }
      break;
    case BinaryOperator::Subtract:
      if (isProvably(binary.getRhs(), 0)) {
        value = binary.getLhs();
        continue;
      }
      break;
    case BinaryOperator::Multiply:
      if (isProvably(binary.getLhs(), 1)) {
        value = binary.getRhs();
        continue;
      }
      if (isProvably(binary.getRhs(), 1)) {
        value = binary.getLhs();
        continue;
      }
      break;
    case BinaryOperator::FloorDivide:
      if (isProvably(binary.getRhs(), 1)) {
        value = binary.getLhs();
        continue;
      }
      break;
    default:
      break;
    }
    break;
  }
  return value;
}

static gpu::PhysicalExprAttr arrayIndexTileBound(
    gpu::PhysicalExprAttr expression, func::FuncOp kernel,
    llvm::DenseSet<gpu::ParameterRefAttr> &active) {
  auto kind = expression.getKind();
  if (kind == gpu::PhysicalExprKind::Constant)
    return expression.getValue() > 0 ? expression : gpu::PhysicalExprAttr();
  if (kind == gpu::PhysicalExprKind::Parameter) {
    auto parameter = gpu::queryParameterBySymbol(kernel, expression.getParameterReference().getName());
    if (failed(parameter))
      return {};
    if (parameter->isDeferred()) {
      auto bound = parameter->getBinding().getCoverageBound();
      if (!bound) return {};
      bool referencesParameters = false;
      AttrTypeWalker walker;
      walker.addWalk([&](gpu::ParameterRefAttr) { referencesParameters = true; });
      walker.walk(bound);
      if (!referencesParameters) return expression;
      // A minimum selects one operand's value. When all operand domains are
      // included, deferred binding is exact rather than a covering round-up.
      bool exact = bound.getKind() == gpu::PhysicalExprKind::Minimum;
      for (Attribute attribute : bound.getOperands()) {
        auto operand = cast<gpu::PhysicalExprAttr>(attribute);
        auto source = operand.getKind() == gpu::PhysicalExprKind::Parameter
            ? gpu::lookupParameter(kernel, operand.getParameterReference()) : gpu::ParameterAttr{};
        exact &= source && llvm::all_of(source.getCandidates().asArrayRef(), [&](int64_t candidate) {
          return llvm::is_contained(parameter->getCandidates().asArrayRef(), candidate);
        });
      }
      if (exact) {
        auto reference = parameter->getReference();
        if (!active.insert(reference).second) return {};
        auto result = arrayIndexTileBound(bound, kernel, active);
        active.erase(reference);
        return result;
      }
      return gpu::PhysicalExprAttr::get(kernel.getContext(), gpu::PhysicalExprKind::Constant,
          *llvm::max_element(parameter->getCandidates().asArrayRef()), StringAttr::get(kernel.getContext()),
          ArrayAttr::get(kernel.getContext(), {}));
    }
    auto configurations = kernel->getAttrOfType<gpu::ConfigurationSetAttr>(gpu::configurationsAttr);
    if (!configurations || configurations.getStage() != gpu::ConfigurationStage::Complete)
      return {};
    int64_t maximum = 0;
    for (Attribute configuration : configurations.getRows()) {
      auto tuple = dyn_cast<DictionaryAttr>(configuration);
      auto value = tuple ? tuple.getAs<IntegerAttr>(expression.getParameterReference().getName()) : IntegerAttr();
      if (!value || value.getInt() <= 0)
        return {};
      maximum = std::max(maximum, value.getInt());
    }
    return gpu::PhysicalExprAttr::get(
        kernel.getContext(), gpu::PhysicalExprKind::Constant,
        maximum, StringAttr::get(kernel.getContext()),
        ArrayAttr::get(kernel.getContext(), {}));
  }
  if (kind != gpu::PhysicalExprKind::Add &&
      kind != gpu::PhysicalExprKind::Multiply &&
      kind != gpu::PhysicalExprKind::Minimum &&
      kind != gpu::PhysicalExprKind::Maximum)
    return {};
  SmallVector<Attribute> operands;
  for (Attribute operand : expression.getOperands()) {
    auto bound = arrayIndexTileBound(cast<gpu::PhysicalExprAttr>(operand), kernel, active);
    if (!bound && kind != gpu::PhysicalExprKind::Minimum)
      return {};
    if (bound)
      operands.push_back(bound);
  }
  // A bounded tile min(chunk, runtime_shape) is no wider than chunk. The
  // unknown shape operand must not discard that already proven upper bound.
  if (operands.empty())
    return {};
  if (operands.size() == 1)
    return cast<gpu::PhysicalExprAttr>(operands.front());
  return gpu::PhysicalExprAttr::get(
      kernel.getContext(), expression.getKind(), expression.getValue(),
      expression.getSymbol(), ArrayAttr::get(kernel.getContext(), operands));
}

gpu::PhysicalExprAttr arrayIndexTileBound(gpu::PhysicalExprAttr expression,
                                         func::FuncOp kernel) {
  llvm::DenseSet<gpu::ParameterRefAttr> active;
  return arrayIndexTileBound(expression, kernel, active);
}

std::optional<ArrayIndexBounds> arrayIndexTileBounds(func::FuncOp kernel) {
  MLIRContext *context = kernel.getContext();
  auto one = gpu::PhysicalExprAttr::get(
      context, gpu::PhysicalExprKind::Constant, 1,
      StringAttr::get(context), ArrayAttr::get(context, {}));
  llvm::DenseMap<Value, SmallVector<Attribute>> bounds;
  SmallVector<Value> resources;
  for (BlockArgument argument : kernel.getArguments())
    if (auto view = dyn_cast<gpu::ViewType>(argument.getType())) {
      bounds[argument].assign(view.getRank(), one);
      resources.push_back(argument);
    }
  kernel.walk([&](ArrayViewOp array) {
    Value resource = array.getResult();
    bounds[resource].assign(array.getResult().getType().getRank(), one);
    resources.push_back(resource);
  });
  bool hasArrayAccess = false;
  auto result = kernel.walk([&](Operation *operation) {
    Value resource;
    gpu::FragmentType tile;
    if (auto load = dyn_cast<TileLoadOp>(operation)) {
      resource = load.getResource();
      tile = load.getResult().getType();
    } else if (auto store = dyn_cast<TileStoreOp>(operation)) {
      resource = store.getResource();
      tile = store.getValue().getType();
    } else if (auto atomic = dyn_cast<TileAtomicAddOp>(operation)) {
      resource = atomic.getResource();
      tile = atomic.getValue().getType();
    } else if (isa<ScalarLoadOp, ScalarStoreOp, GatherLoadOp, ScatterStoreOp,
                   AtomicRMWOp>(operation)) {
      // Gather/scatter use element coordinates.  Their active coordinates must
      // already be in bounds before narrowing the array's index arithmetic.
      if (!operation->hasAttr("in_bounds"))
        return WalkResult::interrupt();
      resource = operation->getOperand(0);
    } else {
      return WalkResult::advance();
    }
    hasArrayAccess = true;
    auto known = bounds.find(resource);
    if (known == bounds.end())
      return WalkResult::interrupt();
    if (!tile)
      return WalkResult::advance();
    auto &viewBounds = known->second;
    if (tile.getShape().size() != viewBounds.size())
      return WalkResult::interrupt();
    for (auto [axis, extent] : llvm::enumerate(tile.getShape())) {
      auto bound = arrayIndexTileBound(cast<gpu::PhysicalExprAttr>(extent), kernel);
      if (!bound)
        return WalkResult::interrupt();
      if (viewBounds[axis] == one || viewBounds[axis] == bound)
        viewBounds[axis] = bound;
      else
        viewBounds[axis] = gpu::PhysicalExprAttr::get(
            context, gpu::PhysicalExprKind::Maximum, 0,
            StringAttr::get(context), ArrayAttr::get(context, {viewBounds[axis], bound}));
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted() || !hasArrayAccess)
    return {};
  ArrayIndexBounds resultBounds;
  for (Value resource : resources)
    resultBounds.emplace_back(resource, ArrayAttr::get(context, bounds.find(resource)->second));
  return resultBounds;
}

} // namespace intent::cutile
