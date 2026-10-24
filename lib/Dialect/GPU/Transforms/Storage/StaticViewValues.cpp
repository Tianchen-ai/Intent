#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/AccessOpInterface.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <map>
#include <set>
#include <vector>

using namespace mlir;

namespace intent::gpu {
namespace {

struct StaticView {
  Value resource;
  Value lanes;
  unsigned laneAxis = 0;
  SmallVector<int64_t> shape;
};

std::optional<int64_t> constant(Value value) {
  auto attr = dyn_cast_or_null<IntegerAttr>(
      UniformValueAnalysis(describeUniformValue).evaluate(value));
  return attr && attr.getValue().getBitWidth() <= 64
      ? std::optional<int64_t>(attr.getInt()) : std::nullopt;
}

std::optional<StaticView> staticView(scf::ForOp root) {
  if (root.getNumResults()) return std::nullopt;
  auto lower = constant(root.getLowerBound()), upper = constant(root.getUpperBound());
  if (!lower || !upper || *lower < 0 || *upper <= *lower ||
      *upper - *lower > 16 || constant(root.getStep()) != 1) return std::nullopt;
  StaticView result;
  bool reads = false, writes = false, nested = false;
  auto status = root.walk([&](Operation *op) {
    if (isa<scf::ForOp>(op)) {
      nested |= op != root.getOperation();
      return WalkResult::advance();
    }
    if (isa<scf::YieldOp>(op)) return WalkResult::advance();
    if (!isa<LoadOp, StoreOp>(op))
      return !op->getNumRegions() && isMemoryEffectFree(op)
          ? WalkResult::advance() : WalkResult::interrupt();
    auto access = cast<AccessOpInterface>(op);
    auto view = dyn_cast<ViewType>(access.getAccessResource().getType());
    if (!view || (result.resource && result.resource != access.getAccessResource()))
      return WalkResult::interrupt();
    result.resource = access.getAccessResource();
    auto coordinates = access.getAccessCoordinates();
    auto axes = access.getAccessSourceAxes();
    if (coordinates.size() != view.getRank()) return WalkResult::interrupt();
    unsigned vectorAxes = 0;
    for (auto [axis, coordinate] : llvm::enumerate(coordinates)) {
      if (axes[axis] != static_cast<int64_t>(axis)) return WalkResult::interrupt();
      if (!isa<FragmentType>(coordinate.getType())) continue;
      auto range = coordinate.getDefiningOp<MakeRangeOp>();
      if (!range || !isUnitStepRange(range) || !root.isDefinedOutsideOfLoop(coordinate) ||
          (result.lanes && (result.lanes != coordinate || result.laneAxis != axis)))
        return WalkResult::interrupt();
      result.lanes = coordinate;
      result.laneAxis = axis;
      ++vectorAxes;
    }
    if (vectorAxes != 1) return WalkResult::interrupt();
    Value payload = isa<LoadOp>(op) ? op->getResult(0) : cast<StoreOp>(op).getValue();
    auto type = dyn_cast<FragmentType>(payload.getType());
    auto lanes = cast<FragmentType>(result.lanes.getType());
    if (!type || type.getShape() != lanes.getShape() ||
        type.getAxisMaps() != lanes.getAxisMaps() ||
        type.getElementType() != view.getElementType()) return WalkResult::interrupt();
    reads |= isa<LoadOp>(op);
    writes |= isa<StoreOp>(op);
    return WalkResult::advance();
  });
  if (status.wasInterrupted() || !reads || !writes || !nested) return std::nullopt;
  int64_t elements = 1;
  for (auto [axis, attr] : llvm::enumerate(cast<ViewType>(result.resource.getType()).getLayout().getExtents())) {
    if (axis == result.laneAxis) { result.shape.push_back(0); continue; }
    auto extent = constantPhysicalExpression(cast<PhysicalExprAttr>(attr));
    if (!extent || *extent <= 0 || *extent > 256 / elements) return std::nullopt;
    elements *= *extent;
    result.shape.push_back(*extent);
  }
  return result;
}

// Expand only statically evaluable control, keeping every carried value and the
// original order. The caller owns a disposable region until the whole expansion
// and the view-layout proof succeed; a partial expansion is never adopted.
LogicalResult expand(Block &source, OpBuilder &b, IRMapping &mapping, unsigned &budget) {
  for (Operation &op : source.without_terminator()) {
    if (!budget--) return failure();
    if (auto loop = dyn_cast<scf::ForOp>(op)) {
      auto lower = constant(mapping.lookupOrDefault(loop.getLowerBound()));
      auto upper = constant(mapping.lookupOrDefault(loop.getUpperBound()));
      auto step = constant(mapping.lookupOrDefault(loop.getStep()));
      if (!lower || !upper || !step || *lower < 0 || *step != 1 ||
          static_cast<__int128>(*upper) - *lower > 16) return failure();
      SmallVector<Value> carried;
      for (Value init : loop.getInitArgs()) carried.push_back(mapping.lookupOrDefault(init));
      for (int64_t i = *lower; i < *upper; ++i) {
        IRMapping iteration(mapping);
        Value induction = b.create<arith::ConstantOp>(loop.getLoc(),
            b.getIntegerAttr(loop.getInductionVar().getType(), i));
        iteration.map(loop.getInductionVar(), induction);
        for (auto [arg, value] : llvm::zip(loop.getRegionIterArgs(), carried))
          iteration.map(arg, value);
        if (failed(expand(*loop.getBody(), b, iteration, budget))) return failure();
        carried.clear();
        for (Value value : loop.getBody()->getTerminator()->getOperands())
          carried.push_back(iteration.lookupOrDefault(value));
      }
      for (auto [old, value] : llvm::zip(loop.getResults(), carried)) mapping.map(old, value);
      continue;
    }
    Operation *copy = b.clone(op, mapping);
    if (copy->getNumResults() == 1 && copy->getResult(0).getType().isIntOrIndex()) {
      auto value = UniformValueAnalysis(describeUniformValue).evaluate(copy->getResult(0));
      if (value && !isa<arith::ConstantOp>(copy)) {
        Value folded = b.create<arith::ConstantOp>(copy->getLoc(), cast<TypedAttr>(value));
        mapping.map(op.getResult(0), folded);
        copy->erase();
      }
    }
  }
  return success();
}

FailureOr<unsigned> forward(Block &block, const StaticView &view) {
  struct State { Value value, valid; };
  std::map<std::vector<int64_t>, State> cells;
  std::set<std::pair<int64_t, std::vector<int64_t>>> writtenCells;
  unsigned forwarded = 0;
  Operation *firstWrite = nullptr;
  for (Operation &op : block)
    if (isa<StoreOp>(op)) { firstWrite = &op; break; }
  auto hoistOperand = [&](auto &&self, Value value) -> bool {
    Operation *producer = value.getDefiningOp();
    if (!producer || producer->getBlock() != &block ||
        producer->isBeforeInBlock(firstWrite)) return true;
    if (producer->getNumRegions() || !isMemoryEffectFree(producer)) return false;
    for (Value operand : producer->getOperands())
      if (!self(self, operand)) return false;
    producer->moveBefore(firstWrite);
    return true;
  };
  for (Operation &op : llvm::make_early_inc_range(block.without_terminator())) {
    if (!isa<LoadOp, StoreOp>(op)) continue;
    auto access = cast<AccessOpInterface>(&op);
    std::vector<int64_t> key;
    for (auto [axis, coordinate] : llvm::enumerate(access.getAccessCoordinates())) {
      if (axis == view.laneAxis) {
        if (coordinate != view.lanes) return failure();
      } else {
        auto value = constant(coordinate);
        if (!value || *value < 0 || *value >= view.shape[axis]) return failure();
        key.push_back(*value);
      }
    }
    if (auto store = dyn_cast<StoreOp>(op)) {
      auto origin = store->getAttrOfType<IntegerAttr>(originAttr);
      // The shared effect contract can distinguish disjoint unrolled member
      // domains. Repeated updates of one member retain their ordered loop.
      if (!origin || !writtenCells.emplace(origin.getInt(), key).second)
        return failure();
      cells[key] = {store.getValue(), store.getValid()};
      continue;
    }
    auto load = cast<LoadOp>(op);
    auto found = cells.find(key);
    if (found != cells.end() && found->second.value.getType() == load.getType() &&
        (!found->second.valid || found->second.valid == load.getValid() ||
         (load.getValid() && samePhysicalScalarExpression(found->second.valid, load.getValid())))) {
      Value value = found->second.value;
      if (load.getValid()) {
        if (!load.getFill()) return failure();
        OpBuilder b(load);
        value = b.create<SelectOp>(load.getLoc(), load.getType(), load.getValid(), value, load.getFill());
      }
      load.replaceAllUsesWith(value);
      load.erase();
      ++forwarded;
    } else {
      // A cell absent from the table has not been written in this region. With
      // an injective layout, its initial read can cross writes to other cells.
      // Keep data-dependent predicates in place when their reads cannot move.
      if (found == cells.end() && firstWrite && firstWrite->isBeforeInBlock(load) &&
          llvm::all_of(load->getOperands(), [&](Value operand) {
            return hoistOperand(hoistOperand, operand);
          }))
        load->moveBefore(firstWrite);
      cells[key] = {load.getResult(), load.getValid()};
    }
  }
  return forwarded;
}

LogicalResult promote(scf::ForOp root, func::FuncOp kernel) {
  auto view = staticView(root);
  if (!view) return success();
  OpBuilder b(root);
  auto placeholder = b.create<arith::ConstantIntOp>(root.getLoc(), 1, 1);
  auto branch = b.create<scf::IfOp>(root.getLoc(), placeholder, true);
  b.setInsertionPointToStart(branch.thenBlock());
  IRMapping mapping;
  unsigned budget = 32768;
  // A block wrapper lets the same expansion preserve the outer loop's order.
  auto copy = cast<scf::ForOp>(b.clone(*root));
  Block source;
  copy->moveBefore(&source, source.end());
  OpBuilder end(root.getContext());
  end.setInsertionPointToEnd(&source);
  end.create<scf::YieldOp>(root.getLoc());
  auto expanded = expand(source, b, mapping, budget);
  auto forwarded = succeeded(expanded) ? forward(*branch.thenBlock(), *view)
                                       : FailureOr<unsigned>(failure());
  if (failed(forwarded) || *forwarded == 0) {
    branch.erase();
    placeholder.erase();
    return success();
  }
  auto injective = materializeNonOverlappingView(kernel, view->resource);
  if (failed(injective)) {
    branch.erase();
    placeholder.erase();
    return success();
  }
  branch.getConditionMutable().assign(*injective);
  placeholder.erase();
  root->moveBefore(branch.elseBlock()->getTerminator());
  return success();
}

} // namespace

LogicalResult promoteStaticViewValues(ModuleOp module) {
  auto kernel = getPhysicalKernel(module);
  if (failed(kernel)) return failure();
  SmallVector<scf::ForOp> roots;
  kernel->walk([&](scf::ForOp loop) {
    if (!loop->getParentOfType<scf::ForOp>()) roots.push_back(loop);
  });
  for (scf::ForOp root : roots)
    if (failed(promote(root, *kernel))) return failure();
  if (failed(eliminateCommonValues(module))) return failure();
  return closeValueRelations(*kernel);
}

} // namespace intent::gpu
