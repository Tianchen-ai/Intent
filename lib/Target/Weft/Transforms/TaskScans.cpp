#include "TaskConversion.h"
#include "ScalarValues.h"
#include "Intent/Target/Weft/Transforms/Scans.h"

using namespace mlir;
namespace wk = ::weft::kernel;

namespace intent::weft_provider {
using namespace task_detail;

LogicalResult TaskConversion::scan(cpu::ScanOp operation) {
  if (!isNativeScanTile(operation))
    return operation.emitError("native prefix requires a bounded selected scan tile");
  Location loc = operation.getLoc();
  SmallVector<Value> inputs, initial;
  SmallVector<Type> types;
  int64_t scanAxis = nextAxis++;
  for (Value source : operation.getSources()) {
    auto input = read(source);
    if (failed(input)) return failure();
    auto type = valueType(element((*input).getType()), shape((*input).getType()), {scanAxis});
    Value aligned = b.create<wk::ReshapeOp>(loc, type, *input, array(axes((*input).getType())));
    inputs.push_back(aligned);
    types.push_back(type);
  }
  for (Value source : operation.getInitials()) initial.push_back(values.lookup(source));
  auto combine = [&](ValueRange left, ValueRange right) -> FailureOr<SmallVector<Value>> {
    Block &body = operation.getCombine().front();
    IRMapping mapping;
    for (auto [old, value] : llvm::zip(body.getArguments().take_front(left.size()), left))
      mapping.map(old, value);
    for (auto [old, value] : llvm::zip(body.getArguments().drop_front(left.size()), right))
      mapping.map(old, value);
    for (Operation &nested : body.without_terminator()) {
      auto result = expression(&nested, mapping);
      if (failed(result)) return failure();
      mapping.map(nested.getResult(0), *result);
    }
    SmallVector<Value> results;
    for (Value value : body.getTerminator()->getOperands()) results.push_back(mapping.lookup(value));
    return results;
  };
  auto prefix = b.create<wk::ScanOp>(loc, types, inputs, 0, operation.getReverse());
  {
    OpBuilder::InsertionGuard guard(b);
    auto *body = new Block;
    prefix.getCombine().push_back(body);
    SmallVector<Value> left, right;
    for (Type type : types) left.push_back(body->addArgument(element(type), loc));
    for (Type type : types) right.push_back(body->addArgument(element(type), loc));
    b.setInsertionPointToStart(body);
    auto results = combine(left, right);
    if (failed(results)) return failure();
    b.create<wk::ScanYieldOp>(loc, *results);
  }
  // The native collective has an implicit neutral start. The source initial
  // state, including an incoming block carry, remains an explicit combine.
  auto inclusive = combine(initial, prefix.getResults());
  if (failed(inclusive)) return failure();
  SmallVector<Value> result = *inclusive;
  if (!operation.getInclusive()) {
    auto inputType = cast<wk::ValueType>(inputs.front().getType());
    int64_t size = inputType.getShape()[0];
    auto indicesType = wk::ValueType::get(b.getContext(), b.getIndexType(),
        inputType.getShape(), inputType.getAxisIds());
    Value ordinal = b.create<wk::IotaOp>(loc, indicesType, 0, size);
    Value boundary = index(loc, operation.getReverse() ? size - 1 : 0);
    auto predicateType = wk::ValueType::get(b.getContext(), b.getI1Type(),
        inputType.getShape(), inputType.getAxisIds());
    Value first = b.create<wk::CompareOp>(loc, predicateType, ordinal, boundary, "eq");
    auto adjacent = binary(loc, ordinal, index(loc, 1), operation.getReverse() ? "add" : "sub");
    if (failed(adjacent)) return failure();
    Value safe = b.create<wk::SelectOp>(loc, indicesType, first, boundary, *adjacent);
    for (auto [number, value] : llvm::enumerate(result)) {
      Value shifted = b.create<wk::LookupOp>(loc, value.getType(), value, safe, "in_bounds");
      result[number] = b.create<wk::SelectOp>(loc, value.getType(), first, initial[number], shifted);
    }
  }
  for (auto [destination, value] : llvm::zip(operation.getOutputs(), result)) {
    auto type = resultType(destination);
    if (failed(type)) return failure();
    Value aligned = b.create<wk::ReshapeOp>(loc, *type, value, array(axes(value.getType())));
    if (failed(write(destination, aligned))) return failure();
  }
  return success();
}
} // namespace intent::weft_provider
