#include "Construction.h"

namespace intent::kir_to_dsa {

LogicalResult Construction::atomic(AtomicRMWOp operation) {
  if (operation.getKind() != AtomicRMWKind::Add ||
      operation.getOrdering() != AtomicOrdering::Relaxed ||
      !getElementTypeOrSelf(operation.getValue().getType()).isInteger(32))
    return operation.emitError("DSA atomic realization requires relaxed i32 add");
  auto relation = analysis.indexRelation(operation);
  if (failed(relation)) return failure();
  Location loc = operation.getLoc();
  Value memory = get(relation->source), input = get(operation.getValue());
  if (!memory || !input) return operation.emitError("atomic operands are unavailable");
  LocalShape shape;
  bool tensor = isa<RankedTensorType>(operation.getResult().getType());
  if (tensor) {
    auto selected = localShape(operation.getResult(), loc);
    if (failed(selected)) return failure();
    shape = *selected;
  }
  Value output = allocateTensor(loc, b.getI32Type(), shape);
  Value old = allocate(loc, b.getI32Type(), 1, 1);
  if (failed(eachElement(loc, shape, [&](ValueRange coordinates) {
    auto indices = accessCoordinates(*relation, shape, coordinates, loc);
    if (failed(indices)) return failure();
    Value offset = index(loc, 0);
    for (auto [axis, coordinate] : llvm::enumerate(*indices))
      offset = add(loc, offset, mul(loc, coordinate, stride(loc, memory, axis)));
    Value value = isa<MemRefType>(input.getType())
        ? loadLocal(loc, input, coordinates) : input;
    b.create<dsa::AtomicAddOp>(loc, memory, offset, value, old);
    storeLocal(loc, loadLocal(loc, old, {}), output, coordinates);
    return success();
  }))) return failure();
  values.map(operation.getResult(), tensor ? output : loadLocal(loc, output, {}));
  return success();
}

LogicalResult Construction::histogram(HistogramOp operation) {
  Location loc = operation.getLoc();
  auto shape = localShape(operation.getResult(), loc);
  if (failed(shape)) return failure();
  auto element = cast<RankedTensorType>(operation.getResult().getType()).getElementType();
  Value output = allocateTensor(loc, element, *shape);
  auto accumulate = [&]() -> LogicalResult {
    Value input = get(operation.getValues()), valid = get(operation.getValid());
    if (!input || !valid) return operation.emitError("histogram source is unavailable");
    LocalShape sourceShape = localShapes.lookup(input);
    return eachElement(loc, sourceShape, [&](ValueRange coordinates) {
      Value active = isa<MemRefType>(valid.getType())
          ? loadLocal(loc, valid, coordinates) : valid;
      auto branch = b.create<scf::IfOp>(loc, active, false);
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(branch.thenBlock());
      // Only active values form an index. Their in-range obligation is the
      // histogram operation's contract, not an assumption about inactive data.
      Value bin = asIndex(loadLocal(loc, input, coordinates), loc);
      Value old = loadLocal(loc, output, ValueRange{bin});
      Value one = b.create<arith::ConstantOp>(loc, b.getIntegerAttr(old.getType(), 1));
      storeLocal(loc, b.create<arith::AddIOp>(loc, old, one), output, ValueRange{bin});
      return success();
    });
  };
  auto inputType = cast<RankedTensorType>(operation.getValues().getType());
  std::optional<WorksetTiling> plan;
  if (inputType.getRank())
    plan = planExecutionSlices(*operation->getBlock(), 0,
        {{operation.getValues(), 0}, {operation.getValid(), 0}});
  auto domain = plan ? sliceDomain(*plan, config.getTile(), true) : std::nullopt;
  if (domain) {
    auto savedValues = values;
    auto savedProducts = products;
    auto savedSlices = valueSlices;
    auto status = loop(loc, index(loc, 0), domain->extent, index(loc, domain->capacity),
        [&](Value begin) {
      Value count = b.create<arith::MinSIOp>(loc,
          sub(loc, domain->extent, begin), index(loc, domain->capacity));
      bindExecutionSlice(*plan, *domain, begin, count);
      return accumulate();
    });
    values = std::move(savedValues);
    products = std::move(savedProducts);
    valueSlices = std::move(savedSlices);
    if (failed(status)) return failure();
  } else if (failed(accumulate())) return failure();
  values.map(operation.getResult(), output);
  return success();
}

} // namespace intent::kir_to_dsa
