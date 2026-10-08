#include "Intent/Dialect/CPU/Transforms/Task/Tasks.h"
#include "Intent/Dialect/CPU/IR/CPUAttrs.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "Intent/Dialect/CPU/Transforms/Implementation/Implementation.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/CPU/Transforms/Structure/LoopBuilders.h"
#include "Intent/Dialect/CPU/Transforms/Structure/ProducerVersions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"

using namespace mlir;
namespace intent::cpu {
namespace {

LogicalResult block(linalg::GenericOp operation, int64_t width) {
  if (operation.getOutputs().empty() || operation.getNumResults())
    return operation.emitError("bounded structured implementation requires destination buffers");
  auto maps = operation.getIndexingMapsArray();
  if (llvm::any_of(maps, [](AffineMap map) {
        return map.getNumSymbols() || llvm::any_of(map.getResults(), [](AffineExpr expression) {
          return !isa<AffineDimExpr, AffineConstantExpr>(expression);
        });
      })) return operation.emitError("bounded structured implementation requires projected coordinate maps");
  OpBuilder b(operation);
  Location loc = operation.getLoc();
  auto iterators = operation.getIteratorTypesArray();
  SmallVector<Value> extents(iterators.size()), offsets(iterators.size());
  SmallVector<OpFoldResult> sizes(iterators.size());
  for (auto [operand, map] : llvm::zip(operation->getOperands(), maps)) {
    auto type = dyn_cast<MemRefType>(operand.getType());
    if (!type) continue;
    for (auto [axis, expression] : llvm::enumerate(map.getResults()))
      if (auto dim = dyn_cast<AffineDimExpr>(expression)) {
        unsigned dimension = dim.getPosition();
        if (extents[dimension]) continue;
        if (!type.isDynamicDim(axis)) extents[dimension] = index(b, loc, type.getDimSize(axis));
        else extents[dimension] = b.create<memref::DimOp>(loc, operand, axis);
      }
  }
  if (llvm::any_of(extents, [](Value extent) { return !extent; }))
    return operation.emitError("bounded structured traversal has an unbound iteration axis");
  Value destination = operation.getOutputs()[0];
  bool allParallel = llvm::all_of(iterators, [](utils::IteratorType type) { return type == utils::IteratorType::parallel; });
  Value root = destination;
  while (true) {
    if (auto view = root.getDefiningOp<memref::SubViewOp>()) root = view.getSource();
    else if (auto cast = root.getDefiningOp<memref::CastOp>()) root = cast.getSource();
    else break;
  }
  if (allParallel && operation.getRegion().front().getArguments().back().use_empty() &&
      isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(root.getDefiningOp()) &&
      !completelyWritesBuffer(operation, root)) {
    DominanceInfo dominance(operation->getParentOfType<func::FuncOp>());
    bool initialized = llvm::any_of(root.getUsers(), [&](Operation *user) {
      bool complete = false;
      if (auto fill = dyn_cast<linalg::FillOp>(user)) complete = fill.getOutputs()[0] == root;
      if (auto copy = dyn_cast<memref::CopyOp>(user)) complete = copy.getTarget() == root;
      return complete && dominance.properlyDominates(user, operation);
    });
    if (!initialized) {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointAfter(root.getDefiningOp());
      Type element = cast<MemRefType>(root.getType()).getElementType();
      Value zero = b.create<arith::ConstantOp>(loc, element, b.getZeroAttr(element));
      // Functional tile assembly needs a complete value before partial writes.
      // The original producer, not this initializer, defines observable data.
      b.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{root});
    }
  }
  int64_t lastParallel = -1;
  for (auto [axis, type] : llvm::enumerate(iterators))
    if (type == utils::IteratorType::parallel) lastParallel = axis;
  std::function<void(unsigned)> visit = [&](unsigned axis) {
    if (axis == iterators.size()) {
      SmallVector<Value> inputs, outputs;
      for (auto [number, operand] : llvm::enumerate(operation->getOperands())) {
        Value selected = operand;
        if (auto type = dyn_cast<MemRefType>(operand.getType())) {
          SmallVector<OpFoldResult> origins, shape, strides(type.getRank(), b.getIndexAttr(1));
          for (auto expression : maps[number].getResults()) {
            if (auto dim = dyn_cast<AffineDimExpr>(expression)) {
              origins.push_back(offsets[dim.getPosition()]); shape.push_back(sizes[dim.getPosition()]);
            } else {
              origins.push_back(b.getIndexAttr(cast<AffineConstantExpr>(expression).getValue()));
              shape.push_back(b.getIndexAttr(1));
            }
          }
          selected = b.create<memref::SubViewOp>(loc, operand, origins, shape, strides);
        }
        (number < static_cast<size_t>(operation.getNumDpsInputs()) ? inputs : outputs).push_back(selected);
      }
      auto tile = b.create<linalg::GenericOp>(loc, inputs, outputs, maps, iterators,
          [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange arguments) {
            IRMapping mapping;
            mapping.map(operation.getRegion().front().getArguments(), arguments);
            for (Operation &nested : operation.getRegion().front().without_terminator()) {
              if (auto coordinate = dyn_cast<linalg::IndexOp>(nested)) {
                Value local = nestedBuilder.create<linalg::IndexOp>(nestedLoc, coordinate.getDim());
                mapping.map(coordinate.getResult(), nestedBuilder.create<arith::AddIOp>(nestedLoc, local, offsets[coordinate.getDim()]));
              } else nestedBuilder.clone(nested, mapping);
            }
            SmallVector<Value> results;
            for (Value value : operation.getRegion().front().getTerminator()->getOperands()) results.push_back(mapping.lookupOrDefault(value));
            nestedBuilder.create<linalg::YieldOp>(nestedLoc, results);
          });
      tile->setAttrs(operation->getAttrs());
      return;
    }
    if (iterators[axis] == utils::IteratorType::reduction) {
      offsets[axis] = index(b, loc, 0);
      if (auto constant = extents[axis].getDefiningOp<arith::ConstantIndexOp>()) sizes[axis] = b.getIndexAttr(constant.value());
      else sizes[axis] = extents[axis];
      visit(axis + 1);
      return;
    }
    int64_t panel = allParallel && axis == lastParallel ? width : 1;
    Value zero = index(b, loc, 0), step = index(b, loc, panel);
    Value end = b.create<arith::SubIOp>(loc, extents[axis], b.create<arith::RemSIOp>(loc, extents[axis], step));
    loop(b, loc, zero, end, panel, [&](Value coordinate) {
      offsets[axis] = coordinate; sizes[axis] = b.getIndexAttr(panel); visit(axis + 1);
    });
    if (panel != 1)
      loop(b, loc, end, extents[axis], 1, [&](Value coordinate) {
        offsets[axis] = coordinate; sizes[axis] = b.getIndexAttr(1); visit(axis + 1);
      });
  };
  visit(0);
  operation.erase();
  return success();
}

}

LogicalResult blockStructuredComputations(func::FuncOp function, const ImplementationRegistry &implementations) {
  SmallVector<linalg::GenericOp> operations;
  function.walk([&](linalg::GenericOp operation) {
    if (!isMatrixContraction(operation) && operation->hasAttr("intent_cpu.implementation")) operations.push_back(operation);
  });
  for (auto operation : operations) {
    auto implementation = implementations.lookup(operation);
    if (failed(implementation)) return failure();
    if (!(*implementation)->parallelWindow) continue;
    auto binding = operation->getAttrOfType<ImplementationAttr>("intent_cpu.implementation");
    int64_t width = (*implementation)->parallelWindow(binding);
    if (width <= 0) return operation.emitError("implementation requires a positive bounded parallel window");
    if (failed(block(operation, width))) return failure();
  }
  return success();
}

}
