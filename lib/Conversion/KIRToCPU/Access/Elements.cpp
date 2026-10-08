#include "../Construction.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/TypeUtilities.h"
#include <functional>

using namespace mlir;

namespace intent::kir_to_cpu {

LogicalResult Construction::indexedWrite(Operation *operation) {
  auto fact = analysis.indexRelation(operation);
  if (failed(fact)) return failure();
  auto access = cast<IndexedAccessOpInterface>(operation);
  Value input = values.lookup(access.getStoredValue()), destination = values.lookup(fact->source);
  Location loc = operation->getLoc();
  SmallVector<Value> sizes, members;
  if (auto type = dyn_cast<RankedTensorType>(input.getType()))
    for (int64_t axis = 0; axis < type.getRank(); ++axis)
      sizes.push_back(dimension(builder, loc, input, axis));
  std::function<LogicalResult(unsigned)> traverse = [&](unsigned axis) -> LogicalResult {
    if (axis == sizes.size()) {
      auto coordinates = indexedCoordinates(*fact, members, builder, loc);
      if (failed(coordinates)) return failure();
      Value value = elementAt(input, members, builder, loc);
      builder.create<memref::StoreOp>(loc, value, destination, *coordinates);
      return success();
    }
    auto loop = builder.create<scf::ForOp>(loc, constant(loc, 0), sizes[axis], constant(loc, 1));
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    members.push_back(loop.getInductionVar());
    LogicalResult result = traverse(axis + 1);
    members.pop_back();
    return result;
  };
  return traverse(0);
}

FailureOr<Value> Construction::indexedRead(Operation *operation, const IndexRelationFact &fact, Value source) {
  auto access = cast<IndexedAccessOpInterface>(operation);
  Type resultType = operation->getResult(0).getType();
  auto tensor = dyn_cast<RankedTensorType>(resultType);
  auto read = [&](OpBuilder &nested, Location loc,
                  ValueRange members) -> FailureOr<Value> {
    auto load = [&](OpBuilder &activeBuilder) -> FailureOr<Value> {
      auto coordinates = indexedCoordinates(fact, members, activeBuilder, loc);
      if (failed(coordinates)) return failure();
      return extractElement(activeBuilder, loc, source, *coordinates);
    };
    if (alwaysValid(operation)) return load(nested);
    Value active = elementAt(values.lookup(access.getAccessValidity()), members, nested, loc);
    auto conditional = nested.create<scf::IfOp>(loc, TypeRange{getElementTypeOrSelf(resultType)}, active, true);
    OpBuilder activeBuilder = OpBuilder::atBlockBegin(conditional.thenBlock());
    auto value = load(activeBuilder);
    if (failed(value)) {
      conditional.erase();
      return failure();
    }
    activeBuilder.create<scf::YieldOp>(loc, *value);
    OpBuilder inactiveBuilder = OpBuilder::atBlockBegin(conditional.elseBlock());
    Value fill = elementAt(values.lookup(access.getAccessFill()), members,
                           inactiveBuilder, loc);
    inactiveBuilder.create<scf::YieldOp>(loc, fill);
    return conditional.getResult(0);
  };
  if (!tensor) return read(builder, operation->getLoc(), {});
  auto shape = extents(operation->getResult(0), operation->getLoc());
  if (failed(shape)) return failure();
  Value output = emptyTensor(tensor, *shape, operation->getLoc());
  LogicalResult status = success();
  auto result = builder.create<linalg::GenericOp>(operation->getLoc(), TypeRange{output.getType()}, ValueRange{}, ValueRange{output},
      SmallVector<AffineMap>{builder.getMultiDimIdentityMap(tensor.getRank())},
      SmallVector<utils::IteratorType>(tensor.getRank(), utils::IteratorType::parallel),
      [&](OpBuilder &nested, Location loc, ValueRange) {
        SmallVector<Value> members;
        for (int64_t axis = 0; axis < tensor.getRank(); ++axis)
          members.push_back(nested.create<linalg::IndexOp>(loc, axis));
        auto value = read(nested, loc, members);
        if (failed(value)) {
          status = failure();
          return;
        }
        nested.create<linalg::YieldOp>(loc, *value);
      });
  if (failed(status)) {
    result.erase();
    return failure();
  }
  return result.getResult(0);
}


} // namespace intent::kir_to_cpu
