#include "Lowering.h"
#include "Intent/Dialect/Block/IR/AccessOpInterface.h"
#include "Intent/Target/DSA/BangC/Physical/IR/Views.h"

namespace intent::bangc::block_lowering {
LogicalResult Lowering::access(Operation *operation) {
  auto access = cast<block::AccessOpInterface>(operation);
  using Kind = block::AccessKind;
  if (access.getAccessKind() != Kind::Load && access.getAccessKind() != Kind::Gather &&
      access.getAccessKind() != Kind::Store && access.getAccessKind() != Kind::AtomicRMW)
    return operation->emitError("BANG C has no realization for this access ordering or atomic primitive");
  auto logicalShape = shape(access.getAccessValueType());
  if (failed(logicalShape)) return failure();
  Value resource = get(access.getAccessResource());
  bool external = cast<MemRefType>(resource.getType()).getMemorySpaceAsInt() == 0;
  Value result;
  if (operation->getNumResults()) {
    result = allocate(operation->getResult(0).getType());
    if (!result) return failure();
  }
  auto rectangular = rectangularAccess(operation, resource, result);
  if (failed(rectangular)) return failure();
  if (*rectangular) return success();
  auto status = each(*logicalShape, [&](ValueRange lane) -> LogicalResult {
    auto laneValue = [&](Value value) { return element(value, access.getAccessValueType(), lane); };
    Value active;
    if (access.getAccessValidity()) {
      auto value = laneValue(access.getAccessValidity());
      if (failed(value)) return failure();
      active = *value;
    }
    if (result && active) {
      Value fill;
      if (access.getAccessFill()) {
        auto value = laneValue(access.getAccessFill());
        if (failed(value)) return failure();
        fill = *value;
      } else fill = b.create<arith::ConstantOp>(loc, b.getZeroAttr(cast<MemRefType>(result.getType()).getElementType()));
      if (failed(write(fill, result, lane))) return failure();
    }
    auto effect = [&]() -> LogicalResult {
      unsigned rank = external ? cast<MemRefType>(resource.getType()).getRank() : extents[resource].size();
      SmallVector<Value> coordinate(rank, index(0));
      for (auto [sourceAxis, value] : llvm::zip(access.getAccessSourceAxes(), access.getAccessCoordinates())) {
        auto part = laneValue(value);
        if (failed(part)) return failure();
        auto scalar = castScalarValue(b, loc, *part, b.getIndexType());
        if (failed(scalar)) return failure();
        coordinate[sourceAxis] = *scalar;
      }
      Value offset = index(0);
      if (external)
        for (auto [axis, value] : llvm::enumerate(coordinate)) {
          Value step = b.create<physical::StrideOp>(loc, b.getIndexType(), resource, axis);
          offset = b.createOrFold<arith::AddIOp>(loc, offset,
              b.createOrFold<arith::MulIOp>(loc, step, value));
        }
      if (access.getAccessKind() == Kind::Store) {
        auto value = laneValue(access.getAccessPayloads().front());
        if (failed(value)) return failure();
        if (external) b.create<physical::StoreScalarOp>(loc, *value, resource, offset);
        else if (failed(write(*value, resource, coordinate))) return failure();
      } else if (auto atomic = dyn_cast<block::AtomicRMWOp>(operation)) {
        if (!external || atomic.getKind() != AtomicRMWKind::Add ||
            atomic.getOrdering() != AtomicOrdering::Relaxed ||
            !cast<MemRefType>(resource.getType()).getElementType().isInteger(32))
          return atomic.emitError("BANG C global atomics currently implement relaxed i32 addition");
        Value slot = allocate(b.getI32Type(), {});
        auto value = laneValue(atomic.getValue());
        if (failed(value)) return failure();
        b.create<physical::AtomicAddOp>(loc, resource, offset, *value, slot);
        if (failed(write(read(slot, {}), result, lane))) return failure();
      } else {
        Value value = external ? Value(b.create<physical::LoadScalarOp>(loc,
            cast<MemRefType>(resource.getType()).getElementType(), resource, offset)) : read(resource, coordinate);
        if (failed(write(value, result, lane))) return failure();
      }
      return success();
    };
    if (!active) return effect();
    auto branch = b.create<scf::IfOp>(loc, active, false);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(branch.thenBlock());
    return effect();
  });
  if (failed(status)) return failure();
  if (result) {
    Value bound = result;
    if (!isa<block::FragmentType>(operation->getResult(0).getType())) bound = read(result, {});
    values.map(operation->getResult(0), bound);
  }
  return success();
}
}
