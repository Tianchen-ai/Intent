#include "Construction.h"
#include "Intent/Dialect/DSA/IR/Views.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"

namespace intent::kir_to_dsa {

Value Construction::collectiveView(Location loc, Value buffer,
                                   const LocalShape &shape) {
  auto memory = cast<MemRefType>(buffer.getType());
  SmallVector<int64_t> capacities;
  for (const LocalAxis &axis : shape) capacities.push_back(axis.capacity);
  auto type = MemRefType::get(capacities, memory.getElementType(),
      MemRefLayoutAttrInterface{}, memory.getMemorySpace());
  auto view = dsa::materializeCollectiveView(b, loc, buffer, type);
  if (failed(view)) return {};
  return *view;
}

LogicalResult Construction::buildCollectiveHelper(
    Operation *operation, Block &source, TypeRange stateTypes,
    ValueRange states, ValueRange captures) {
  SmallVector<Type> fields;
  appendProductLeafTypes(stateTypes, fields);
  if (fields.size() != states.size())
    return operation->emitError("DSA collective state schema is incomplete");

  SmallVector<Type> formalTypes;
  for (auto [type, state] : llvm::zip(fields, states)) {
    if (isa<RankedTensorType>(type)) {
      if (!completeShape(localShapes.lookup(state)))
        return operation->emitError(
            "DSA full-slice combine requires complete retained axes");
      SmallVector<int64_t> capacities;
      for (const LocalAxis &axis : localShapes.lookup(state))
        capacities.push_back(axis.capacity);
      auto memory = cast<MemRefType>(state.getType());
      formalTypes.push_back(MemRefType::get(capacities, memory.getElementType(),
          MemRefLayoutAttrInterface{}, memory.getMemorySpace()));
    } else {
      formalTypes.push_back(cast<MemRefType>(state.getType()).getElementType());
    }
  }
  Region &region = operation->getRegion(0);
  Block &body = region.emplaceBlock();
  Location loc = operation->getLoc();
  for (unsigned side = 0; side < 2; ++side)
    for (Type type : formalTypes) body.addArgument(type, loc);
  for (Value capture : captures) body.addArgument(capture.getType(), loc);

  auto savedValues = values;
  auto savedProducts = products;
  auto savedDomains = domains;
  auto savedAxes = axisBindings;
  auto savedShapes = localShapes;
  auto savedSlices = valueSlices;
  auto savedStreamed = streamedOperations;
  auto restore = llvm::make_scope_exit([&] {
    values = std::move(savedValues);
    products = std::move(savedProducts);
    domains = std::move(savedDomains);
    axisBindings = std::move(savedAxes);
    localShapes = std::move(savedShapes);
    valueSlices = std::move(savedSlices);
    streamedOperations = std::move(savedStreamed);
  });
  OpBuilder::InsertionGuard insertion(b);
  b.setInsertionPointToStart(&body);
  SmallVector<Value> flatArguments;
  for (unsigned side = 0; side < 2; ++side)
    for (auto [field, state] : llvm::enumerate(states)) {
      Value argument = body.getArgument(side * fields.size() + field);
      if (isa<MemRefType>(argument.getType())) {
        LocalShape shape = localShapes.lookup(state);
        auto physical = cast<MemRefType>(state.getType());
        auto view = dsa::materializeCollectiveView(b, loc, argument, physical);
        if (failed(view)) return failure();
        argument = *view;
        localShapes[argument] = std::move(shape);
      }
      flatArguments.push_back(argument);
    }
  for (auto [number, capture] : llvm::enumerate(captures)) {
    Value argument = body.getArgument(2 * fields.size() + number);
    if (auto shape = localShapes.find(capture); shape != localShapes.end())
      localShapes[argument] = shape->second;
    flatArguments.push_back(argument);
  }
  SmallVector<Type> argumentTypes;
  appendProductLeafTypes(source.getArgumentTypes(), argumentTypes);
  if (flatArguments.size() != argumentTypes.size())
    return operation->emitError("DSA collective helper arguments are incomplete");
  auto arguments = splitFields(source.getArgumentTypes(), flatArguments);
  auto result = helper(source, arguments);
  if (failed(result) || result->size() != fields.size()) return failure();
  SmallVector<Value> yielded;
  for (auto [field, value] : llvm::enumerate(*result)) {
    if (isa<MemRefType>(formalTypes[field])) {
      auto shape = localShapes.find(value);
      if (shape == localShapes.end())
        return operation->emitError("DSA collective yielded slice has no local shape");
      value = collectiveView(loc, value, shape->second);
    } else {
      value = scalarCast(loc, value, formalTypes[field]);
    }
    if (!value || value.getType() != formalTypes[field])
      return operation->emitError("DSA collective helper changes its state schema");
    yielded.push_back(value);
  }
  b.create<dsa::CollectiveYieldOp>(loc, yielded);

  // Shape arithmetic and existing immutable snapshots are explicit captures of
  // this physical helper. No construction environment survives this boundary.
  llvm::SetVector<Value> external;
  getUsedValuesDefinedAbove(region, external);
  for (Value value : external) {
    Value formal = body.addArgument(value.getType(), loc);
    if (auto reduce = dyn_cast<dsa::SliceReduceOp>(operation))
      reduce.getCapturesMutable().append(value);
    else cast<dsa::ScanOp>(operation).getCapturesMutable().append(value);
    value.replaceUsesWithIf(formal, [&](OpOperand &use) {
      return region.isAncestor(use.getOwner()->getParentRegion());
    });
  }
  return success();
}

LogicalResult Construction::emitSliceReduction(
    ReduceOp source, ValueRange inputs, ValueRange initials, ValueRange captures,
    ValueRange outputs, ArrayRef<int64_t> axes) {
  Location loc = source.getLoc();
  if (inputs.size() != outputs.size() || initials.size() != outputs.size())
    return source.emitError("DSA collective field counts disagree");
  SmallVector<Value> views, initialViews, outputViews, counts;
  for (Value input : inputs) {
    auto found = localShapes.find(input);
    if (found == localShapes.end())
      return source.emitError("DSA collective source has no local shape");
    LocalShape shape = found->second;
    views.push_back(collectiveView(loc, input, shape));
    if (!views.back()) return failure();
    for (const LocalAxis &axis : shape) counts.push_back(axis.count);
  }
  for (auto [initial, output] : llvm::zip(initials, outputs)) {
    LocalShape shape = localShapes.find(output)->second;
    outputViews.push_back(collectiveView(loc, output, shape));
    initialViews.push_back(isa<MemRefType>(initial.getType())
        ? collectiveView(loc, initial, shape)
        : scalarCast(loc, initial, cast<MemRefType>(output.getType()).getElementType()));
    if (!outputViews.back() || !initialViews.back()) return failure();
  }
  auto operation = b.create<dsa::SliceReduceOp>(loc, views, initialViews,
      captures, outputViews, counts, b.getDenseI64ArrayAttr(axes));
  return buildCollectiveHelper(operation, source.getCombine().front(),
      source.getIdentities().getTypes(), outputs, captures);
}

LogicalResult Construction::emitScan(
    ScanOp source, ValueRange inputs, ValueRange initials, ValueRange captures,
    ValueRange outputs, ValueRange finals) {
  Location loc = source.getLoc();
  if (inputs.size() != outputs.size() || initials.size() != finals.size() ||
      inputs.size() != finals.size())
    return source.emitError("DSA scan field counts disagree");
  SmallVector<Value> views, initialViews, outputViews, finalViews, counts;
  for (auto [input, output] : llvm::zip(inputs, outputs)) {
    auto found = localShapes.find(input);
    if (found == localShapes.end())
      return source.emitError("DSA scan source has no local shape");
    LocalShape shape = found->second;
    views.push_back(collectiveView(loc, input, shape));
    outputViews.push_back(collectiveView(loc, output, shape));
    if (!views.back() || !outputViews.back()) return failure();
    for (const LocalAxis &axis : shape) counts.push_back(axis.count);
  }
  for (auto [initial, final] : llvm::zip(initials, finals)) {
    LocalShape shape = localShapes.find(final)->second;
    finalViews.push_back(collectiveView(loc, final, shape));
    initialViews.push_back(isa<MemRefType>(initial.getType())
        ? collectiveView(loc, initial, shape)
        : scalarCast(loc, initial, cast<MemRefType>(final.getType()).getElementType()));
    if (!finalViews.back() || !initialViews.back()) return failure();
  }
  auto operation = b.create<dsa::ScanOp>(loc, views, initialViews, captures,
      outputViews, counts, finalViews, source.getAxisAttr(),
      source.getInclusiveAttr(), source.getReverseAttr());
  return buildCollectiveHelper(operation, source.getCombine().front(),
      source.getIdentities().getTypes(), finals, captures);
}

} // namespace intent::kir_to_dsa
