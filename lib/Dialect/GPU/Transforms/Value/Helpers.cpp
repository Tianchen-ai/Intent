#include "Intent/Dialect/GPU/Transforms/Value/Helpers.h"
#include "Intent/Dialect/GPU/Analysis/Helpers.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Transforms/Value/ExecutionSchema.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;

namespace intent::gpu {

LogicalResult cloneLaneWiseHelper(Region &source, Region &target,
                                  TypeRange argumentTypes, TypeRange resultTypes,
                                  std::string &reason) {
  if (!target.empty()) {
    reason = "lane-wise helper target region must be empty";
    return failure();
  }
  if (failed(proveLaneWiseHelper(source, &reason))) return failure();
  Block &body = source.front();
  auto yield = cast<YieldOp>(body.getTerminator());
  if (body.getNumArguments() != argumentTypes.size() ||
      yield.getNumOperands() != resultTypes.size()) {
    reason = "selected lane-wise helper signature has the wrong arity";
    return failure();
  }
  for (auto [before, after] : llvm::zip(body.getArgumentTypes(), argumentTypes))
    if (scalarCallbackType(before) != scalarCallbackType(after)) {
      reason = "selected lane-wise helper formal changes its scalar/product schema";
      return failure();
    }
  for (auto [before, after] : llvm::zip(yield.getOperandTypes(), resultTypes))
    if (scalarCallbackType(before) != scalarCallbackType(after)) {
      reason = "selected lane-wise helper yield changes its scalar/product schema";
      return failure();
    }
  Region temporary;
  auto *block = new Block();
  temporary.push_back(block);
  IRMapping mapping;
  for (auto [argument, type] : llvm::zip(body.getArguments(), argumentTypes))
    mapping.map(argument, block->addArgument(type, argument.getLoc()));
  OpBuilder builder(yield.getContext());
  builder.setInsertionPointToEnd(block);
  for (Operation &operation : body.without_terminator()) {
    if (Value forwarded = laneWiseProjectionSource(&operation)) {
      mapping.map(operation.getResult(0), mapping.lookup(forwarded));
      continue;
    }
    if (failed(cloneWithSchema(builder, &operation, mapping))) {
      reason = "lane-wise operation cannot adopt the selected formal schema: " +
               operation.getName().getStringRef().str();
      return failure();
    }
  }
  SmallVector<Value> results;
  for (auto [value, type] : llvm::zip(yield.getValues(), resultTypes)) {
    auto projected = projectPhysicalValueToSchema(
        builder, yield.getLoc(), mapping.lookup(value), type);
    if (failed(projected)) {
      reason = "lane-wise helper result cannot adopt the selected yield schema";
      return failure();
    }
    results.push_back(*projected);
  }
  builder.create<YieldOp>(yield.getLoc(), results);
  target.takeBody(temporary);
  return success();
}

LogicalResult scalarizeElementwiseCallback(Region &source, Region &target) {
  std::string reason;
  auto reject = [&]() -> LogicalResult {
    if (Operation *owner = source.getParentOp()) return owner->emitOpError(reason);
    return failure();
  };
  if (failed(proveLaneWiseHelper(source, &reason))) return reject();
  SmallVector<Type> arguments, results;
  for (Type type : source.front().getArgumentTypes())
    arguments.push_back(scalarCallbackType(type));
  for (Type type : source.front().getTerminator()->getOperandTypes())
    results.push_back(scalarCallbackType(type));
  if (failed(cloneLaneWiseHelper(source, target, arguments, results, reason)))
    return reject();
  return success();
}

LogicalResult liftCombineRegion(Region &source, Region &target,
                                TypeRange accumulatorTypes,
                                std::string &reason) {
  if (failed(proveLaneWiseHelper(source, &reason))) return failure();
  if (source.front().getNumArguments() < accumulatorTypes.size() * 2) {
    reason = "combine argument schema is incomplete";
    return failure();
  }
  SmallVector<Type> arguments(accumulatorTypes);
  llvm::append_range(arguments, accumulatorTypes);
  for (BlockArgument capture : source.front().getArguments().drop_front(
           accumulatorTypes.size() * 2))
    arguments.push_back(capture.getType());
  return cloneLaneWiseHelper(source, target, arguments, accumulatorTypes, reason);
}

FailureOr<Value> projectLaneWiseReduction(OpBuilder &builder, Location location,
                                          ReduceOp reduce, FragmentType target) {
  if (reduce.getSources().size() != 1 || reduce.getIdentities().size() != 1 ||
      !reduce.getCaptures().empty() || reduce.getNumResults() != 1 ||
      failed(proveLaneWiseHelper(reduce.getCombine()))) return failure();
  auto source = dyn_cast<FragmentType>(reduce.getResult(0).getType());
  auto input = dyn_cast<FragmentType>(reduce.getSources().front().getType());
  if (!source || !input || source.getOwner() != target.getOwner() ||
      source.getShape().size() != target.getShape().size()) return failure();
  auto relation = queryAxisProjection(source, target);
  if (!relation.isExact() || !llvm::all_of(
          llvm::enumerate(relation.targetToSource), [](auto item) {
            return item.value() && *item.value() == item.index();
          })) return failure();
  SmallVector<Attribute> shape(input.getShape().getValue());
  unsigned resultAxis = 0;
  for (unsigned axis = 0; axis < shape.size(); ++axis)
    if (!llvm::is_contained(reduce.getAxes(), static_cast<int64_t>(axis)))
      shape[axis] = target.getShape()[resultAxis++];
  auto inputTarget = FragmentType::get(
      target.getContext(), input.getElementType(), builder.getArrayAttr(shape),
      input.getAxisMaps(), input.getValidity(), input.getOwner());
  auto resultTarget = FragmentType::get(
      target.getContext(), target.getElementType(), target.getShape(),
      source.getAxisMaps(), target.getValidity(), target.getOwner());

  Region temporary;
  auto *block = new Block();
  temporary.push_back(block);
  OpBuilder staged(builder.getContext());
  staged.setInsertionPointToEnd(block);
  auto inputValue = projectPhysicalValueToSchema(
      staged, location, reduce.getSources().front(), inputTarget);
  auto identity = projectPhysicalValueToSchema(
      staged, location, reduce.getIdentities().front(), resultTarget);
  if (failed(inputValue) || failed(identity)) return failure();
  Region combine;
  std::string reason;
  if (failed(liftCombineRegion(reduce.getCombine(), combine,
                              TypeRange{resultTarget}, reason))) return failure();
  IRMapping mapping;
  mapping.map(reduce.getSources().front(), *inputValue);
  mapping.map(reduce.getIdentities().front(), *identity);
  auto replacement = cast<ReduceOp>(staged.clone(*reduce, mapping));
  replacement.getCombine().takeBody(combine);
  replacement.getResult(0).setType(resultTarget);
  Value result = replacement.getResult(0);
  if (resultTarget != target)
    result = staged.create<BroadcastOp>(location, target, result);
  while (!block->empty()) {
    Operation *operation = &block->front();
    operation->remove();
    builder.insert(operation);
  }
  return result;
}

} // namespace intent::gpu
