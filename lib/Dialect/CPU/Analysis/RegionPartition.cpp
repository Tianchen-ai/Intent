#include "Intent/Dialect/CPU/Analysis/RegionPartition.h"
#include "Intent/Dialect/CPU/Analysis/ExtentRelations.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

using namespace mlir;
namespace intent::cpu {
namespace {

class Axes {
public:
  DenseMap<Value, SmallVector<unsigned>> values;
  SmallVector<unsigned> parents, forbidden;
  SmallVector<linalg::GenericOp> contractions;
  bool effectsKnown = true;

  ArrayRef<unsigned> get(Value value) {
    auto found = values.find(value);
    if (found != values.end()) return found->second;
    auto type = dyn_cast<MemRefType>(value.getType());
    SmallVector<unsigned> axes;
    if (type)
      for (int64_t i = 0; i < type.getRank(); ++i) {
        axes.push_back(parents.size()); parents.push_back(parents.size());
        if (type.getDimSize(i) == 1) forbidden.push_back(axes.back());
      }
    return values.try_emplace(value, std::move(axes)).first->second;
  }
  unsigned root(unsigned axis) {
    if (parents[axis] != axis) parents[axis] = root(parents[axis]);
    return parents[axis];
  }
  void join(unsigned left, unsigned right) { parents[root(left)] = root(right); }
  void bind(Value left, Value right) {
    SmallVector<unsigned> first(get(left));
    auto second = get(right);
    for (auto [a, b] : llvm::zip(first, second)) join(a, b);
  }
  void reject(Value value) { llvm::append_range(forbidden, get(value)); }

  bool fullSubviewAxis(memref::SubViewOp view, unsigned axis) {
    if (getConstantIntValue(view.getMixedOffsets()[axis]) != 0 ||
        getConstantIntValue(view.getMixedStrides()[axis]) != 1) return false;
    return haveEqualExtents(ValueBoundsConstraintSet::Variable(view.getMixedSizes()[axis]),
                            ValueBoundsConstraintSet::Variable(view.getSource(), axis));
  }

  void inspect(Operation *operation) {
    if (auto generic = dyn_cast<linalg::GenericOp>(operation)) {
      auto maps = generic.getIndexingMapsArray();
      SmallVector<SmallVector<unsigned>> loops(generic.getNumLoops());
      for (auto [operand, map] : llvm::zip(generic->getOperands(), maps)) {
        auto axes = get(operand);
        for (auto [axis, expression] : llvm::zip(axes, map.getResults())) {
          if (auto dim = dyn_cast<AffineDimExpr>(expression)) loops[dim.getPosition()].push_back(axis);
          else forbidden.push_back(axis);
        }
      }
      for (auto [loop, kind] : llvm::zip(loops, generic.getIteratorTypesArray())) {
        if (loop.empty()) continue;
        for (unsigned axis : ArrayRef(loop).drop_front()) join(loop.front(), axis);
        if (kind == utils::IteratorType::reduction) forbidden.push_back(loop.front());
      }
      if (isMatrixContraction(generic)) contractions.push_back(generic);
      return;
    }
    if (auto fill = dyn_cast<linalg::FillOp>(operation)) { (void)get(fill.getOutputs()[0]); return; }
    if (auto copy = dyn_cast<memref::CopyOp>(operation)) { bind(copy.getSource(), copy.getTarget()); return; }
    if (auto cast = dyn_cast<memref::CastOp>(operation)) { bind(cast.getSource(), cast.getResult()); return; }
    if (auto axes = unitReshapeAxes(operation)) {
      SmallVector<unsigned> source(get(operation->getOperand(0)));
      auto result = get(operation->getResult(0));
      for (auto [from, to] : *axes) join(source[from], result[to]);
      return;
    }
    if (auto view = dyn_cast<memref::SubViewOp>(operation)) {
      SmallVector<unsigned> source(get(view.getSource()));
      auto result = get(view.getResult());
      auto dropped = view.getDroppedDims();
      unsigned target = 0;
      for (auto [axis, id] : llvm::enumerate(source)) {
        if (dropped.test(axis)) { forbidden.push_back(id); continue; }
        join(id, result[target++]);
        if (!fullSubviewAxis(view, axis)) forbidden.push_back(id);
      }
      return;
    }
    if (isa<memref::DimOp, memref::AllocOp, memref::DeallocOp>(operation)) {
      for (Value value : operation->getOperands()) (void)get(value);
      for (Value value : operation->getResults()) (void)get(value);
      return;
    }
    for (Value value : operation->getOperands()) reject(value);
    for (Value value : operation->getResults()) reject(value);
    if (!isMemoryEffectFree(operation)) effectsKnown = false;
  }

  std::optional<RegionPartition> select(RegionOpInterface program, unsigned selected, int64_t width) {
    selected = root(selected);
    if (width <= 1 || llvm::any_of(forbidden, [&](unsigned axis) { return root(axis) == selected; })) return std::nullopt;
    RegionPartition result;
    result.width = width;
    for (auto &item : values)
      for (auto [axis, id] : llvm::enumerate(item.second)) {
        if (root(id) != selected) continue;
        if (!result.axes.try_emplace(item.first, axis).second) return std::nullopt;
      }
    for (Value value : program.getIdentities()) if (!result.axes.count(value)) return std::nullopt;
    for (Value value : program.getInitialStates()) if (!result.axes.count(value)) return std::nullopt;
    for (Value value : program.getDestinations()) if (!result.axes.count(value)) return std::nullopt;
    for (Value value : program.getSources())
      if (auto found = result.axes.find(value); found != result.axes.end() && found->second == program.getAxis())
        return std::nullopt;
    bool legal = true;
    for (Region &helper : program.getOperation()->getRegions()) {
      helper.walk([&](Operation *operation) {
        if (auto generic = dyn_cast<linalg::GenericOp>(operation)) {
          for (auto [operand, map] : llvm::zip(generic->getOperands(), generic.getIndexingMapsArray()))
            if (auto found = result.axes.find(operand); found != result.axes.end())
              result.loops[generic] = cast<AffineDimExpr>(map.getResult(found->second)).getPosition();
        }
        auto dimension = dyn_cast<memref::DimOp>(operation);
        if (!dimension) return;
        auto found = result.axes.find(dimension.getSource());
        if (found == result.axes.end()) return;
        auto axis = dimension.getConstantIndex();
        if (!axis) { legal = false; return; }
        if (*axis != found->second) return;
        // A logical extent used numerically cannot be replaced by a tile size.
        for (OpOperand &use : dimension.getResult().getUses()) {
          if (auto allocation = dyn_cast<memref::AllocOp>(use.getOwner())) {
            auto target = result.axes.find(allocation.getResult());
            unsigned dynamic = 0;
            bool matches = false;
            for (int64_t i = 0; i < allocation.getType().getRank(); ++i) {
              if (!allocation.getType().isDynamicDim(i)) continue;
              if (dynamic++ == use.getOperandNumber())
                matches = target != result.axes.end() && target->second == i;
            }
            legal &= matches;
          } else if (auto view = dyn_cast<memref::SubViewOp>(use.getOwner())) {
            auto source = result.axes.find(view.getSource());
            unsigned operand = 1;
            for (OpFoldResult offset : view.getMixedOffsets()) operand += isa<Value>(offset);
            bool matches = false;
            for (auto [axis, size] : llvm::enumerate(view.getMixedSizes())) {
              if (!isa<Value>(size)) continue;
              if (operand++ == use.getOperandNumber())
                matches = source != result.axes.end() && source->second == axis && fullSubviewAxis(view, axis);
            }
            legal &= matches;
          } else legal = false;
        }
      });
    }
    if (!legal) return std::nullopt;
    return result;
  }
};

}

std::optional<RegionPartition> analyzeRegionPartition(RegionOpInterface program,
                                                     int64_t tileM, int64_t tileN) {
  Axes axes;
  // Nested control needs its own block-argument/loop-carried coordinate proof.
  for (Region &helper : program.getOperation()->getRegions())
    for (Operation &operation : helper.front().without_terminator())
      if (operation.getNumRegions() && !isa<linalg::LinalgOp>(operation)) return std::nullopt;
  for (Region &helper : program->getRegions()) {
    auto schema = program.getRegionSchema(helper);
    if (failed(schema)) return std::nullopt;
    for (const auto &relation : *schema) axes.bind(relation.argument, relation.prototype);
    helper.walk([&](Operation *operation) { axes.inspect(operation); });
  }
  for (auto [value, output] : llvm::zip(program.getInitialValues(), program.getFinalDestinations()))
    axes.bind(value, output);
  if (!axes.effectsKnown) return std::nullopt;
  const int64_t widths[] = {tileM, tileN};
  for (auto contraction : axes.contractions) {
    SmallVector<unsigned> output(axes.get(contraction.getOutputs()[0]));
    for (auto [axis, width] : llvm::zip(output, widths))
      if (auto partition = axes.select(program, axis, width)) return partition;
  }
  return std::nullopt;
}

}
