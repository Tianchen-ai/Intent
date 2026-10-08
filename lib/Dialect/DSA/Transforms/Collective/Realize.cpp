#include "CollectiveLowering.h"
#include "Intent/Dialect/DSA/Analysis/UniformValues.h"
#include "Intent/Dialect/DSA/Transforms/Collective/Collectives.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::dsa::collective {

Lowering::Lowering(SliceReduceOp op)
    : operation(op), builder(op), location(op.getLoc()),
      body(op.getCombine().front()), captures(op.getCaptures()),
      axes(op.getAxes()) {
  initialize(op.getSources(), op.getInitials(), op.getOutputs(), op.getCounts(),
             {});
}

Lowering::Lowering(ScanOp op)
    : operation(op), builder(op), location(op.getLoc()),
      body(op.getCombine().front()), captures(op.getCaptures()),
      axes{static_cast<int64_t>(op.getAxis())}, scan(true),
      inclusive(op.getInclusive()), reverse(op.getReverse()) {
  initialize(op.getSources(), op.getInitials(), op.getOutputs(), op.getCounts(),
             op.getFinals());
}

void Lowering::initialize(ValueRange sources, ValueRange initials,
                          ValueRange outputs, ValueRange counts,
                          ValueRange finals) {
  auto function = operation->getParentOfType<func::FuncOp>();
  StorageAnalysis storage(function);
  UniformMemoryAnalysis uniforms(function, storage);
  unsigned offset = 0;
  for (auto [ordinal, source] : llvm::enumerate(sources)) {
    auto type = cast<MemRefType>(source.getType());
    Value initial = initials[ordinal];
    if (isa<MemRefType>(initial.getType()))
      if (Value scalar = uniforms.read(initial, operation)) initial = scalar;
    Field field{source, initial, outputs[ordinal],
                finals.empty() ? Value() : finals[ordinal],
                body.getArgument(ordinal).getType(), {}, {}, {}};
    llvm::append_range(field.counts, counts.slice(offset, type.getRank()));
    offset += type.getRank();
    for (int64_t axis = 0; axis < type.getRank(); ++axis) {
      if (llvm::is_contained(axes, axis))
        continue;
      field.freeShape.push_back(type.getDimSize(axis));
      field.freeCounts.push_back(field.counts[axis]);
    }
    fields.push_back(std::move(field));
  }
  scalar = llvm::none_of(fields, [](const Field &field) {
    return isa<MemRefType>(field.formalType);
  });
}

SmallVector<Value> Lowering::sourceCoordinates(const Field &field,
                                               ValueRange free,
                                               ValueRange reduced) const {
  SmallVector<Value> result;
  unsigned freeAxis = 0;
  for (unsigned axis = 0; axis < field.counts.size(); ++axis) {
    auto found = llvm::find(axes, axis);
    result.push_back(found == axes.end() ? free[freeAxis++]
                                         : reduced[found - axes.begin()]);
  }
  return result;
}

FailureOr<SmallVector<Value>> Lowering::combine(ValueRange left,
                                               ValueRange right,
                                               bool lift) {
  SmallVector<Value> arguments(left);
  llvm::append_range(arguments, right);
  llvm::append_range(arguments, captures);
  if (lift)
    return liftScalarCombine(builder, location, body, arguments,
                             fields.front().freeShape);
  IRMapping mapping;
  for (auto [formal, value] : llvm::zip(body.getArguments(), arguments)) {
    if (formal.getType().isIndex() && value.getType().isInteger(64))
      value = builder.create<arith::IndexCastOp>(location, formal.getType(), value);
    if (formal.getType() != value.getType())
      return operation->emitError("DSA collective realization lost a helper argument type"),
             failure();
    mapping.map(formal, value);
  }
  for (Operation &nested : body.without_terminator())
    builder.clone(nested, mapping);
  SmallVector<Value> results;
  for (Value value : body.getTerminator()->getOperands())
    results.push_back(mapping.lookupOrDefault(value));
  return results;
}

bool Lowering::transferFreeAxis(const Field &field, Value slot,
                               ValueRange reduced, bool read) {
  auto source = cast<MemRefType>(field.source.getType());
  // One contiguous member vector is already the helper's physical component.
  // Keep arbitrary higher-rank/strided free slices on the existing coordinate
  // path rather than flattening away their actual geometry.
  if (field.freeShape.size() != 1 || source.getRank() == 0 ||
      llvm::is_contained(axes, source.getRank() - 1)) return false;
  Value zero = index(builder, location, 0), one = index(builder, location, 1);
  auto coordinates = sourceCoordinates(field, ValueRange{zero}, reduced);
  Value offset = linearOffset(builder, location, source, coordinates);
  if (read)
    builder.create<LoadTileOp>(location, physicalView(builder, location, field.source),
        physicalView(builder, location, slot), offset, zero, one, one,
        field.freeCounts.front());
  else
    builder.create<StoreTileOp>(location, physicalView(builder, location, slot),
        physicalView(builder, location, field.output), offset, zero, one, one,
        field.freeCounts.front());
  // LoadTile defines inactive physical lanes as zero, exactly as the previous
  // full-slot fill followed by active member copies. StoreTile writes only the
  // active free-axis members, including for exclusive and reverse scans.
  return true;
}

LogicalResult Lowering::realizeAt(ValueRange independent) {
  SmallVector<Value> state, next, items;
  SmallVector<bool> shaped;
  for (const Field &field : fields) {
    bool whole = lifted || isa<MemRefType>(field.formalType);
    ArrayRef<int64_t> shape = whole ? ArrayRef(field.freeShape)
                                    : ArrayRef<int64_t>();
    Type element = cast<MemRefType>(field.source.getType()).getElementType();
    state.push_back(allocate(builder, location, element, shape));
    next.push_back(allocate(builder, location, element, shape));
    items.push_back(allocate(builder, location, element, shape));
    shaped.push_back(whole);
    Value initial = field.initial;
    if (!whole && isa<MemRefType>(initial.getType()))
      initial = load(builder, location, initial, independent);
    copy(builder, location, initial, state.back());
  }

  SmallVector<Value> reducedCounts;
  for (int64_t axis : axes)
    reducedCounts.push_back(fields.front().counts[axis]);
  auto traverse = forEach(builder, location, reducedCounts,
                         [&](ValueRange positions) -> LogicalResult {
    SmallVector<Value> reduced(positions);
    if (scan && reverse)
      reduced[0] = builder.createOrFold<arith::SubIOp>(
          location,
          builder.createOrFold<arith::SubIOp>(
              location, reducedCounts[0], index(builder, location, 1)),
          reduced[0]);
    auto transfer = [&](bool read) -> LogicalResult {
      for (auto [ordinal, field] : llvm::enumerate(fields)) {
        Value slot = read ? items[ordinal] : state[ordinal];
        if (shaped[ordinal] && transferFreeAxis(field, slot, reduced, read))
          continue;
        if (read && shaped[ordinal]) {
          auto element = cast<MemRefType>(slot.getType()).getElementType();
          fill(builder, location, slot,
               builder.create<arith::ConstantOp>(
                   location, builder.getZeroAttr(element)));
        }
        ValueRange extents = shaped[ordinal] ? ValueRange(field.freeCounts)
                                             : ValueRange();
        if (failed(forEach(builder, location, extents,
                           [&](ValueRange free) -> LogicalResult {
          ValueRange sourceFree = shaped[ordinal] ? free : independent;
          auto coordinates = sourceCoordinates(field, sourceFree, reduced);
          if (read)
            store(builder, location,
                  load(builder, location, field.source, coordinates), slot,
                  free);
          else
            store(builder, location, load(builder, location, slot, free),
                  field.output, coordinates);
          return success();
        })))
          return failure();
      }
      return success();
    };
    if (scan && !inclusive && failed(transfer(false)))
      return failure();
    if (failed(transfer(true)))
      return failure();
    SmallVector<Value> left, right;
    for (auto [ordinal, field] : llvm::enumerate(fields)) {
      left.push_back(shaped[ordinal] ? state[ordinal]
                                    : load(builder, location, state[ordinal], {}));
      right.push_back(shaped[ordinal] ? items[ordinal]
                                     : load(builder, location, items[ordinal], {}));
    }
    auto updated = combine(left, right, lifted);
    if (failed(updated))
      return failure();
    // Results can borrow any incoming field. Snapshot all of them before
    // publishing any updated state, including fields whose values are swapped.
    for (auto [value, destination] : llvm::zip(*updated, next))
      copy(builder, location, value, destination);
    for (auto [value, destination] : llvm::zip(next, state))
      copy(builder, location, value, destination);
    return scan && inclusive ? transfer(false) : success();
  });
  if (failed(traverse))
    return failure();
  for (auto [ordinal, field] : llvm::enumerate(fields)) {
    Value destination = scan ? field.final : field.output;
    if (!destination)
      continue;
    if (shaped[ordinal])
      copy(builder, location, state[ordinal], destination);
    else
      store(builder, location, load(builder, location, state[ordinal], {}),
            destination, independent);
  }
  return success();
}

LogicalResult Lowering::realize() {
  lifted = scalar && !fields.front().freeShape.empty() &&
           canLiftScalarCombine(body);
  if (scalar && !lifted)
    return forEach(builder, location, fields.front().freeCounts,
                   [&](ValueRange coordinates) { return realizeAt(coordinates); });
  return realizeAt({});
}

LogicalResult Lowering::run() {
  if (!scan && (nativeReduction() || treeReduction())) {
    operation->erase();
    return success();
  }
  if (failed(realize()))
    return failure();
  operation->erase();
  return success();
}

} // namespace intent::dsa::collective

namespace intent::dsa {

LogicalResult realizeCollectives(func::FuncOp function) {
  while (true) {
    Operation *next = nullptr;
    function.walk<WalkOrder::PreOrder>([&](Operation *operation) {
      if (isa<SliceReduceOp, ScanOp>(operation)) {
        next = operation;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (!next)
      return success();
    LogicalResult result = isa<SliceReduceOp>(next)
        ? collective::Lowering(cast<SliceReduceOp>(next)).run()
        : collective::Lowering(cast<ScanOp>(next)).run();
    if (failed(result))
      return failure();
  }
}

} // namespace intent::dsa
