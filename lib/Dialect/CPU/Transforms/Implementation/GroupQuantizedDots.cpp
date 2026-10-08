#include "Intent/Dialect/CPU/Transforms/Implementation/ImplementationInputs.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "Intent/Dialect/CPU/IR/CPUAttrs.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "Intent/Dialect/CPU/Transforms/Implementation/Implementation.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Intent/Dialect/CPU/Transforms/Structure/LoopBuilders.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;
namespace intent::cpu {
namespace {

Value withoutCasts(Value value) {
  while (auto cast = value.getDefiningOp<memref::CastOp>()) value = cast.getSource();
  return value;
}

bool rowProjection(memref::SubViewOp view, Value row, unsigned rank) {
  if (!view || view.getSourceType().getRank() != rank ||
      view.getType().getRank() != rank - 1 || !view.getDroppedDims().test(0) ||
      view.getDroppedDims().count() != 1 || view.getMixedOffsets()[0] != OpFoldResult(row) ||
      getConstantIntValue(view.getMixedSizes()[0]) != 1)
    return false;
  for (OpFoldResult step : view.getMixedStrides())
    if (getConstantIntValue(step) != 1) return false;
  for (unsigned axis = 1; axis < rank; ++axis) {
    if (getConstantIntValue(view.getMixedOffsets()[axis]) != 0) return false;
    auto size = view.getMixedSizes()[axis];
    if (!view.getSourceType().isDynamicDim(axis)) {
      if (getConstantIntValue(size) != view.getSourceType().getDimSize(axis)) return false;
    } else {
      auto value = dyn_cast<Value>(size);
      auto dimension = value ? value.getDefiningOp<memref::DimOp>() : memref::DimOp();
      if (!dimension || dimension.getSource() != view.getSource() ||
          dimension.getConstantIndex() != axis) return false;
    }
  }
  return true;
}

LogicalResult group(scf::ParallelOp parallel, const ImplementationRegistry &implementations) {
  if (parallel.getNumLoops() != 1 || parallel.getNumResults() ||
      !matchPattern(parallel.getLowerBound()[0], m_Zero()) ||
      !matchPattern(parallel.getStep()[0], m_One())) return success();
  QuantizedDotOp dot;
  for (Operation &operation : parallel.getBody()->without_terminator()) {
    if (auto candidate = dyn_cast<QuantizedDotOp>(operation)) {
      if (dot) return success();
      dot = candidate;
    } else if (operation.getNumRegions() || !isMemoryEffectFree(&operation)) return success();
  }
  if (!dot || cast<MemRefType>(dot.getOutput().getType()).getRank()) return success();
  auto implementation = implementations.lookup(dot);
  if (failed(implementation)) return failure();
  if (!(*implementation)->parallelWindow) return success();
  auto binding = dot->getAttrOfType<ImplementationAttr>("intent_cpu.implementation");
  int64_t width = (*implementation)->parallelWindow(binding);
  if (width <= 1) return success();
  Value row = parallel.getInductionVars()[0];
  auto lhs = withoutCasts(dot.getLhs()).getDefiningOp<memref::SubViewOp>();
  auto output = withoutCasts(dot.getOutput()).getDefiningOp<memref::SubViewOp>();
  if (!rowProjection(lhs, row, 3) || !rowProjection(output, row, 1)) return success();
  auto function = parallel->getParentOfType<func::FuncOp>();
  DominanceInfo dominance(function);
  const Value sources[] = {lhs.getSource(), dot.getRhs(), output.getSource()};
  for (Value source : sources)
    if (!dominance.dominates(source, parallel)) return success();
  StorageAnalysis analysis(function);
  auto interface = function->getAttrOfType<EntryRequirementsAttr>(entryRequirementsAttr);
  auto destination = analysis.externalView(output.getSource());
  // Group only the existing independent external writes. Captured preparation
  // remains outside the workset and is read once per record by the local group.
  if (!interface.getDisjointOutputs() || !destination || destination.getAccess() != 1 ||
      !analysis.isReadOnly(lhs.getSource()) ||
      !analysis.disjoint(dot.getRhs(), output.getSource())) return success();

  OpBuilder b(parallel);
  Location loc = parallel.getLoc();
  Value zero = index(b, loc, 0), one = index(b, loc, 1), block = index(b, loc, width);
  Value count = b.create<arith::DivSIOp>(loc, parallel.getUpperBound()[0], block);
  Value full = multiply(b, loc, count, block);
  Value tail = b.create<arith::SubIOp>(loc, parallel.getUpperBound()[0], full);
  auto emit = [&](Value extent, bool grouped) {
    auto loop = b.create<scf::ParallelOp>(loc, ValueRange{zero}, ValueRange{extent}, ValueRange{one});
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(loop.getBody());
    Value begin = grouped ? multiply(b, loc, loop.getInductionVars()[0], block)
                          : add(b, loc, full, loop.getInductionVars()[0]);
    IRMapping mapping;
    mapping.map(row, begin);
    for (Operation &operation : parallel.getBody()->without_terminator()) {
      if (&operation != dot || !grouped) {
        b.clone(operation, mapping);
        continue;
      }
      auto window = [&](memref::SubViewOp source) -> Value {
        SmallVector<OpFoldResult> offsets{begin}, sizes{b.getIndexAttr(width)};
        for (unsigned axis = 1; axis < source.getSourceType().getRank(); ++axis) {
          offsets.push_back(b.getIndexAttr(0));
          auto size = source.getMixedSizes()[axis];
          sizes.push_back(isa<Value>(size) ? OpFoldResult(mapping.lookupOrDefault(cast<Value>(size))) : size);
        }
        return b.create<memref::SubViewOp>(loc, source.getSource(), offsets, sizes,
            SmallVector<OpFoldResult>(sizes.size(), b.getIndexAttr(1)));
      };
      auto groupedDot = b.create<QuantizedDotOp>(loc, Type{}, window(lhs), dot.getRhs(), window(output),
          dot.getLhsFormatAttr(), dot.getRhsFormatAttr());
      groupedDot->setAttr("intent_cpu.implementation", binding);
    }
  };
  emit(count, true);
  emit(tail, false);
  parallel.erase();
  return success();
}

}

LogicalResult groupQuantizedDots(func::FuncOp function, const ImplementationRegistry &implementations) {
  SmallVector<scf::ParallelOp> worksets;
  function.walk([&](scf::ParallelOp parallel) { worksets.push_back(parallel); });
  for (auto parallel : llvm::reverse(worksets))
    if (failed(group(parallel, implementations))) return failure();
  return success();
}

}
