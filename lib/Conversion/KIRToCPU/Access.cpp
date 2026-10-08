#include "Construction.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;

namespace intent::kir_to_cpu {

bool Construction::alwaysValid(Operation *operation) {
  Value predicate = cast<IndexedAccessOpInterface>(operation).getAccessValidity();
  if (!predicate) return true;
  while (auto producer = predicate.getDefiningOp()) {
    if (auto literal = dyn_cast<ConstantOp>(producer)) {
      auto value = dyn_cast<IntegerAttr>(literal.getValue());
      return value && value.getType().isInteger(1) && !value.getValue().isZero();
    }
    if (isa<FullOp, BroadcastOp, ReshapeOp>(producer)) predicate = producer->getOperand(0);
    else return false;
  }
  return false;
}

LogicalResult Construction::lower(BufferOp op) {
  Location loc = op.getLoc();
  if (!analysis.logicalBuffer(op).isExact())
    return op.emitError("CPU logical buffer requires an exact lexical allocation fact");
  auto tensor = cast<RankedTensorType>(cast<BufferType>(op.getResult().getType()).getTensor());
  auto sizes = extents(op.getResult(), loc);
  if (failed(sizes)) return failure();
  SmallVector<Value> dynamic;
  for (auto [axis, size] : llvm::enumerate(*sizes))
    if (tensor.isDynamicDim(axis)) dynamic.push_back(size);
  Value storage = builder.create<memref::AllocOp>(loc,
      MemRefType::get(tensor.getShape(), tensor.getElementType()), dynamic);
  if (op.getInitial()) {
    Value initial = values.lookup(op.getInitial());
    if (isa<RankedTensorType>(initial.getType()))
      builder.create<cpu::WriteOp>(loc, initial, storage);
    else builder.create<linalg::FillOp>(loc, ValueRange{initial}, ValueRange{storage});
  }
  values.map(op.getResult(), storage);
  return success();
}

LogicalResult Construction::load(Operation *operation) {
  auto value = indexed(operation);
  if (failed(value)) return failure();
  values.map(operation->getResult(0), *value);
  return success();
}

LogicalResult Construction::store(Operation *operation) {
  Location loc = operation->getLoc();
  auto fact = analysis.indexRelation(operation);
  if (failed(fact)) return failure();
  if (isa<ScatterUniqueOp>(operation) || hasTensorIndices(*fact) ||
      llvm::any_of(fact->terms, [](const IndexTermFact &term) {
        return !term.sourceAxis;
      })) return indexedWrite(operation);
  auto destination = indexed(operation);
  if (failed(destination)) return failure();
  Value input = values.lookup(cast<IndexedAccessOpInterface>(operation).getStoredValue());
  if (isa<RankedTensorType>(input.getType()))
    builder.create<cpu::WriteOp>(loc, input, *destination);
  else
    builder.create<memref::StoreOp>(loc, input, *destination, ValueRange{});
  return success();
}

} // namespace intent::kir_to_cpu
