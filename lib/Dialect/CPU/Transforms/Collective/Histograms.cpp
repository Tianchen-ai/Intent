#include "Intent/Dialect/CPU/IR/CPUAttrs.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "Intent/Dialect/CPU/Transforms/Collective/Collectives.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "Intent/Dialect/CPU/Transforms/Structure/LoopBuilders.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;

namespace intent::cpu {

LogicalResult realizeHistograms(func::FuncOp function) {
  auto capabilities = function->getParentOfType<ModuleOp>()->getAttrOfType<CapabilitiesAttr>("intent_cpu.capabilities");
  auto configuration = function->getAttrOfType<ConfigurationAttr>("intent_cpu.configuration");
  SmallVector<HistogramOp> histograms;
  function.walk([&](HistogramOp operation) { histograms.push_back(operation); });
  for (HistogramOp operation : histograms) {
    bool nestedParallel = bool(operation->getParentOfType<scf::ParallelOp>());
    if (!nestedParallel && (!capabilities || !configuration))
      return operation.emitOpError(
          "CPU histogram grouping requires target capabilities and a bound configuration");
    OpBuilder b(operation);
    Location loc = operation.getLoc();
    auto inputType = cast<MemRefType>(operation.getValues().getType());
    auto outputType = cast<MemRefType>(operation.getOutput().getType());
    int64_t groups = nestedParallel ? 1
        : capabilities.getWorkers() * configuration.getTaskGrain();
    Value zero = index(b, loc, 0), one = index(b, loc, 1), groupCount = index(b, loc, groups);
    Value size = one;
    SmallVector<Value> extents;
    for (int64_t axis = 0; axis < inputType.getRank(); ++axis) {
      extents.push_back(b.create<memref::DimOp>(loc, operation.getValues(), axis));
      size = multiply(b, loc, size, extents.back());
    }
    Value bins = b.create<memref::DimOp>(loc, operation.getOutput(), 0);
    SmallVector<Value> dynamicSizes;
    if (outputType.isDynamicDim(0)) dynamicSizes.push_back(bins);
    Value partials = b.create<memref::AllocOp>(loc,
        MemRefType::get({groups, outputType.getDimSize(0)}, outputType.getElementType()), dynamicSizes);
    Value countZero = b.create<arith::ConstantOp>(loc, b.getIntegerAttr(outputType.getElementType(), 0));
    Value countOne = b.create<arith::ConstantOp>(loc, b.getIntegerAttr(outputType.getElementType(), 1));
    b.create<linalg::FillOp>(loc, ValueRange{countZero}, ValueRange{partials});
    Value chunk = b.create<arith::CeilDivSIOp>(loc, size, groupCount);
    auto tasks = b.create<scf::ParallelOp>(loc, ValueRange{zero}, ValueRange{groupCount}, ValueRange{one});
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(tasks.getBody());
      Value group = tasks.getInductionVars()[0];
      Value begin = multiply(b, loc, group, chunk);
      Value end = b.create<arith::MinSIOp>(loc, add(b, loc, begin, chunk), size);
      loop(b, loc, begin, end, 1, [&](Value ordinal) {
        SmallVector<Value> coordinates(inputType.getRank());
        Value remaining = ordinal;
        for (int64_t axis = inputType.getRank() - 1; axis >= 0; --axis) {
          coordinates[axis] = axis ? Value(b.create<arith::RemSIOp>(loc, remaining, extents[axis])) : remaining;
          if (axis) remaining = b.create<arith::DivSIOp>(loc, remaining, extents[axis]);
        }
        Value active = b.create<memref::LoadOp>(loc, operation.getValid(), coordinates);
        auto conditional = b.create<scf::IfOp>(loc, active, false);
        OpBuilder::InsertionGuard predicateGuard(b);
        b.setInsertionPointToStart(conditional.thenBlock());
        Value value = b.create<memref::LoadOp>(loc, operation.getValues(), coordinates);
        if (!value.getType().isIndex()) {
          if (operation.getUnsignedValues()) value = b.create<arith::IndexCastUIOp>(loc, b.getIndexType(), value);
          else value = b.create<arith::IndexCastOp>(loc, b.getIndexType(), value);
        }
        SmallVector<Value> bin{group, value};
        Value previous = b.create<memref::LoadOp>(loc, partials, bin);
        Value next = b.create<arith::AddIOp>(loc, previous, countOne);
        b.create<memref::StoreOp>(loc, next, partials, bin);
      });
    }
    b.create<linalg::FillOp>(loc, ValueRange{countZero}, ValueRange{operation.getOutput()});
    auto bin = b.getAffineDimExpr(0), group = b.getAffineDimExpr(1);
    b.create<linalg::GenericOp>(loc, ValueRange{partials}, ValueRange{operation.getOutput()},
        SmallVector<AffineMap>{AffineMap::get(2, 0, {group, bin}, b.getContext()),
                              AffineMap::get(2, 0, {bin}, b.getContext())},
        SmallVector<utils::IteratorType>{utils::IteratorType::parallel, utils::IteratorType::reduction},
        [](OpBuilder &nested, Location location, ValueRange arguments) {
          nested.create<linalg::YieldOp>(location, nested.create<arith::AddIOp>(location, arguments[0], arguments[1]).getResult());
        });
    b.create<memref::DeallocOp>(loc, partials);
    operation.erase();
  }
  return success();
}

} // namespace intent::cpu
