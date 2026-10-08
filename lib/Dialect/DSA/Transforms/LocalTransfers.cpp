#include "LocalTransfers.h"
#include "Intent/Dialect/DSA/IR/Views.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

namespace intent::dsa::detail {
namespace {

std::optional<int64_t> capacity(MemRefType type) {
  if (!type.hasStaticShape()) return std::nullopt;
  int64_t count = 1;
  for (int64_t extent : type.getShape())
    if (extent <= 0 || llvm::MulOverflow(count, extent, count))
      return std::nullopt;
  return count;
}

std::optional<int64_t> constant(Value value, LocalSupplyRelations &relations) {
  auto interval = relations.interval(value);
  return interval && interval->first == interval->second
      ? std::optional<int64_t>(interval->first) : std::nullopt;
}

bool boundedCount(Value count, int64_t capacity, LocalSupplyRelations &relations) {
  auto interval = relations.interval(count);
  return interval && interval->first >= 0 && interval->second <= capacity;
}

bool keepCount(Value supplied, Value requested, int64_t first, bool broadcast,
               LocalSupplyRelations &relations) {
  if (!first && !broadcast && relations.equal(supplied, requested)) return true;
  auto source = relations.interval(supplied), result = relations.interval(requested);
  return source && result && first < source->first &&
      (broadcast || result->second <= source->first - first);
}

// Prove each new typed arithmetic operation before forming an offset. This is
// the original producer address evaluated at the consumer's first coordinate.
bool offsetFits(LoadTileOp source, int64_t row, int64_t column,
                LocalSupplyRelations &relations) {
  if (!row && !column) return true;
  auto bounds = relations.interval(source.getOffset());
  if (!bounds) return false;
  auto current = ConstantIntRanges::fromSigned(APInt(64, bounds->first),
                                               APInt(64, bounds->second));
  for (auto [stride, position] : {std::pair{source.getRowStride(), row},
                                  std::pair{source.getColumnStride(), column}}) {
    if (!position) continue;
    auto interval = relations.interval(stride);
    if (!interval) return false;
    auto value = ConstantIntRanges::fromSigned(APInt(64, interval->first),
                                               APInt(64, interval->second));
    auto factor = ConstantIntRanges::constant(APInt(64, position));
    if (!provesSignedNoWrap(BinaryOperator::Multiply, value, factor)) return false;
    auto product = inferIntegerBinary(BinaryOperator::Multiply, stride.getType(), value, factor);
    if (!product || !provesSignedNoWrap(BinaryOperator::Add, current, *product)) return false;
    auto sum = inferIntegerBinary(BinaryOperator::Add, stride.getType(), current, *product);
    if (!sum) return false;
    current = *sum;
  }
  return true;
}

} // namespace

std::optional<LocalTransfer> queryLocalTransfer(Operation *operation,
    StorageAnalysis &storage, LocalSupplyRelations &relations) {
  Value source, destination;
  if (auto copy = dyn_cast<memref::CopyOp>(operation)) {
    source = copy.getSource(); destination = copy.getTarget();
  } else {
    auto load = dyn_cast<LoadTileOp>(operation);
    if (!load || load.getAsynchronous()) return std::nullopt;
    source = load.getSource(); destination = load.getOutput();
  }
  auto input = cast<MemRefType>(source.getType());
  auto output = cast<MemRefType>(destination.getType());
  auto inputCapacity = capacity(input), outputCapacity = capacity(output);
  Value origin = storage.uniqueOrigin(source);
  if (!inputCapacity || !outputCapacity || *inputCapacity > *outputCapacity ||
      input.getElementType() != output.getElementType() || !origin ||
      !origin.getDefiningOp<memref::AllocaOp>() ||
      !destination.getDefiningOp<memref::AllocaOp>() ||
      !storage.disjoint(source, destination) ||
      !isCompleteLocalStorageView(source) || !isCompleteLocalStorageView(destination) ||
      !isCompleteStorageViewOf(source, origin) ||
      !storage.aliases(origin).complete || !storage.aliases(destination).complete)
    return std::nullopt;
  bool identity = *inputCapacity == *outputCapacity;
  if (auto load = dyn_cast<LoadTileOp>(operation)) {
    if (!boundedCount(load.getRows(), output.getDimSize(0), relations) ||
        !boundedCount(load.getColumns(), output.getDimSize(1), relations)) return std::nullopt;
    identity &= relations.equal(load.getOffset(), 0) &&
        relations.equal(load.getRows(), output.getDimSize(0)) &&
        relations.equal(load.getColumns(), output.getDimSize(1)) &&
        (output.getDimSize(0) == 1 || relations.equal(load.getRowStride(), output.getDimSize(1))) &&
        (output.getDimSize(1) == 1 || relations.equal(load.getColumnStride(), 1));
  }
  return LocalTransfer{operation, source, destination, origin, identity};
}

std::optional<LocalTransferRead> composeTransferRead(
    const LocalTransfer &transfer, LoadTileOp reader,
    LocalSupplyRelations &relations) {
  auto producer = dyn_cast<LoadTileOp>(transfer.operation);
  if (!producer || !reader || reader.getAsynchronous() ||
      reader->getBlock() != producer->getBlock() ||
      reader.getSource().getType() != transfer.destination.getType()) return std::nullopt;
  auto shape = cast<MemRefType>(transfer.destination.getType());
  auto result = cast<MemRefType>(reader.getOutput().getType());
  if (!boundedCount(reader.getRows(), result.getDimSize(0), relations) ||
      !boundedCount(reader.getColumns(), result.getDimSize(1), relations)) return std::nullopt;
  auto offset = constant(reader.getOffset(), relations);
  if (!offset || *offset < 0 || *offset >= *capacity(shape)) return std::nullopt;
  int64_t row = *offset / shape.getDimSize(1), column = *offset % shape.getDimSize(1);
  bool broadcastRows = relations.equal(reader.getRowStride(), 0);
  bool broadcastColumns = relations.equal(reader.getColumnStride(), 0);
  if ((!broadcastRows && !relations.equal(reader.getRowStride(), shape.getDimSize(1))) ||
      (!broadcastColumns && !relations.equal(reader.getColumnStride(), 1))) return std::nullopt;
  // No carry from the selected column into another row: the actual rectangular
  // image must remain inside the intermediate descriptor before composing it.
  auto rows = relations.interval(reader.getRows());
  auto columns = relations.interval(reader.getColumns());
  if ((!broadcastRows && rows->second > shape.getDimSize(0) - row) ||
      (!broadcastColumns && columns->second > shape.getDimSize(1) - column) ||
      !offsetFits(producer, row, column, relations)) return std::nullopt;
  return LocalTransferRead{reader, row, column, broadcastRows, broadcastColumns,
      keepCount(producer.getRows(), reader.getRows(), row, broadcastRows, relations),
      keepCount(producer.getColumns(), reader.getColumns(), column, broadcastColumns, relations)};
}

void applyTransferRead(const LocalTransfer &transfer, const LocalTransferRead &read) {
  auto producer = cast<LoadTileOp>(transfer.operation);
  auto reader = read.reader;
  OpBuilder builder(reader);
  Location location = reader.getLoc();
  auto index = [&](int64_t value) -> Value {
    return builder.create<arith::ConstantIndexOp>(location, value);
  };
  Value offset = producer.getOffset();
  auto add = [&](Value stride, int64_t position) {
    if (!position) return;
    Value term = position == 1 ? stride : builder.createOrFold<arith::MulIOp>(location, stride, index(position));
    offset = builder.createOrFold<arith::AddIOp>(location, offset, term);
  };
  add(producer.getRowStride(), read.row);
  add(producer.getColumnStride(), read.column);
  auto count = [&](Value supplied, Value requested, int64_t first, bool broadcast, bool keep) {
    if (keep) return requested;
    Value begin = index(first), zero = index(0);
    if (broadcast) {
      Value active = builder.createOrFold<arith::CmpIOp>(location, arith::CmpIPredicate::sgt, supplied, begin);
      return builder.createOrFold<arith::SelectOp>(location, active, requested, zero);
    }
    // supplied is in [0, capacity] and first in [0, capacity), so this
    // subtraction is representable even when the active intersection is empty.
    Value available = builder.createOrFold<arith::SubIOp>(location, supplied, begin);
    available = builder.createOrFold<arith::MaxSIOp>(location, available, zero);
    return builder.createOrFold<arith::MinSIOp>(location, requested, available);
  };
  Value rows = count(producer.getRows(), reader.getRows(), read.row, read.broadcastRows, read.keepRows);
  Value columns = count(producer.getColumns(), reader.getColumns(), read.column,
                        read.broadcastColumns, read.keepColumns);
  reader.getSourceMutable().assign(transfer.source);
  reader.getOffsetMutable().assign(offset);
  reader.getRowStrideMutable().assign(read.broadcastRows ? index(0) : producer.getRowStride());
  reader.getColumnStrideMutable().assign(read.broadcastColumns ? index(0) : producer.getColumnStride());
  reader.getRowsMutable().assign(rows);
  reader.getColumnsMutable().assign(columns);
}

} // namespace intent::dsa::detail
