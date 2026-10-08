#include "CollectiveLowering.h"
#include "Intent/Dialect/DSA/IR/Views.h"
#include "llvm/ADT/STLExtras.h"
#include <functional>

using namespace mlir;

namespace intent::dsa::collective {

Value index(OpBuilder &builder, Location location, int64_t value) {
  return builder.create<arith::ConstantIndexOp>(location, value);
}

Value view(OpBuilder &builder, Location location, Value storage,
           ArrayRef<int64_t> shape) {
  auto source = cast<MemRefType>(storage.getType());
  auto type = MemRefType::get(shape, source.getElementType(),
                             MemRefLayoutAttrInterface(),
                             source.getMemorySpace());
  auto result = materializeCollectiveView(builder, location, storage, type);
  assert(succeeded(result) && "collective lowering preserves complete storage");
  return *result;
}

Value physicalView(OpBuilder &builder, Location location, Value storage) {
  auto type = cast<MemRefType>(storage.getType());
  int64_t columns = type.getRank() ? type.getShape().back() : 1;
  return view(builder, location, storage,
              {type.getNumElements() / columns, columns});
}

Value allocate(OpBuilder &builder, Location location, Type element,
               ArrayRef<int64_t> shape) {
  int64_t rows = 1;
  for (unsigned axis = 0; axis + 1 < shape.size(); ++axis)
    rows *= shape[axis];
  int64_t columns = shape.empty() ? 1 : shape.back();
  auto type = MemRefType::get({rows, columns}, element,
                             MemRefLayoutAttrInterface(),
                             builder.getI64IntegerAttr(nramSpace));
  auto allocation = builder.create<memref::AllocaOp>(location, type);
  allocation.setAlignment(128);
  return view(builder, location, allocation, shape);
}

void fill(OpBuilder &builder, Location location, Value storage, Value scalar) {
  Type element = cast<MemRefType>(storage.getType()).getElementType();
  if (scalar.getType().isIndex() && element.isInteger(64))
    scalar = builder.create<arith::IndexCastOp>(location, element, scalar);
  builder.create<FillOp>(location, physicalView(builder, location, storage),
                         scalar);
}

void copy(OpBuilder &builder, Location location, Value source,
          Value destination) {
  if (source == destination)
    return;
  if (isa<MemRefType>(source.getType()))
    builder.create<memref::CopyOp>(location, source, destination);
  else
    fill(builder, location, destination, source);
}

LogicalResult forEach(
    OpBuilder &builder, Location location, ValueRange counts,
    llvm::function_ref<LogicalResult(ValueRange)> body) {
  SmallVector<Value> coordinates;
  std::function<LogicalResult(unsigned)> visit = [&](unsigned axis) {
    if (axis == counts.size())
      return body(coordinates);
    auto loop = builder.create<scf::ForOp>(
        location, index(builder, location, 0), counts[axis],
        index(builder, location, 1));
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    coordinates.push_back(loop.getInductionVar());
    LogicalResult result = visit(axis + 1);
    coordinates.pop_back();
    return result;
  };
  return visit(0);
}

Value load(OpBuilder &builder, Location location, Value storage,
           ValueRange coordinates) {
  return builder.create<memref::LoadOp>(location, storage, coordinates);
}

void store(OpBuilder &builder, Location location, Value value, Value storage,
           ValueRange coordinates) {
  Type element = cast<MemRefType>(storage.getType()).getElementType();
  if (value.getType().isIndex() && element.isInteger(64))
    value = builder.create<arith::IndexCastOp>(location, element, value);
  builder.create<memref::StoreOp>(location, value, storage, coordinates);
}

Value linearOffset(OpBuilder &builder, Location location, MemRefType type,
                   ValueRange coordinates) {
  Value offset = index(builder, location, 0);
  for (auto [axis, coordinate] : llvm::enumerate(coordinates)) {
    offset = builder.createOrFold<arith::MulIOp>(
        location, offset, index(builder, location, type.getDimSize(axis)));
    offset = builder.createOrFold<arith::AddIOp>(location, offset, coordinate);
  }
  return offset;
}

} // namespace intent::dsa::collective
