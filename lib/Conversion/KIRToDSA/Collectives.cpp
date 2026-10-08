#include "Construction.h"

namespace intent::kir_to_dsa {

void Construction::bindHelperValue(Value formal, ArrayRef<Value> fields, bool rebase, int64_t sourceAxis) {
  SmallVector<Type> types; appendProductLeafTypes(formal.getType(), types);
  SmallVector<Value> bound;
  for (auto [type, field] : llvm::zip(types, fields)) {
    Value value = field;
    if (!isa<RankedTensorType>(type) && isa<MemRefType>(field.getType()))
      value = scalarCast(formal.getLoc(), loadLocal(formal.getLoc(), field, {}), type);
    bound.push_back(value);
  }
  bindProduct(formal, bound);
  for (auto [type, field] : llvm::zip(types, fields)) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    if (!tensor || !localShapes.count(field)) continue;
    auto shape = localShapes.lookup(field);
    auto ids = cast<TensorShapeAttr>(tensor.getEncoding()).getDimensions();
    for (int64_t axis = 0; axis < tensor.getRank(); ++axis) {
      LocalAxis binding = shape[axis];
      if (rebase && axis == sourceAxis) {
        binding.extent = binding.count;
        binding.begin = index(formal.getLoc(), 0);
        shape[axis] = binding;
      }
      axisBindings[ids[axis]] = binding;
    }
    if (rebase) localShapes[field] = shape;
  }
}

FailureOr<SmallVector<Value>> Construction::helper(Block &block, ArrayRef<SmallVector<Value>> arguments,
                                    unsigned sourceCount, int64_t sourceAxis) {
  if (arguments.size() != block.getNumArguments()) return block.getParentOp()->emitError("DSA helper argument schema mismatch"), failure();
  auto savedValues = values; auto savedProducts = products;
  auto savedAxes = axisBindings;
  auto savedShapes = localShapes;
  auto savedStreamed = streamedOperations;
  for (auto [i, formal] : llvm::enumerate(block.getArguments()))
    bindHelperValue(formal, arguments[i], i < sourceCount, sourceAxis);
  LogicalResult status = lowerOperations(block);
  SmallVector<Value> result;
  if (succeeded(status)) result = flatten(block.getTerminator()->getOperands());
  values = std::move(savedValues); products = std::move(savedProducts);
  axisBindings = std::move(savedAxes);
  streamedOperations = std::move(savedStreamed);
  for (auto &entry : savedShapes) localShapes[entry.first] = entry.second;
  if (failed(status) || result.empty() || llvm::is_contained(result, Value())) return failure();
  return result;
}

SmallVector<SmallVector<Value>> Construction::splitFields(TypeRange types, ValueRange fields) {
  SmallVector<SmallVector<Value>> result;
  for (ProductLeafRange range : getProductLeafRanges(types))
    result.push_back(llvm::to_vector(fields.slice(range.offset, range.size)));
  return result;
}

LogicalResult Construction::streamReduction(ReduceOp reduce, unsigned axis,
    int64_t dimension, const LocalShape &shape) {
  Location loc = reduce.getLoc();
  auto initial = flatten(reduce.getIdentities());
  auto captures = flatten(reduce.getCaptures());
  auto sourceFields = logicalComponents(reduce.getSources());
  SmallVector<Value> outputs;
  if (initial.size() != sourceFields.size()) return failure();
  for (auto [field, value] : llvm::zip(sourceFields, initial)) {
    auto tensor = cast<RankedTensorType>(field.type);
    auto local = localShape(field.value, loc, field.path);
    if (failed(local)) return failure();
    local->erase(local->begin() + axis);
    Value output = allocateTensor(loc, tensor.getElementType(), *local);
    if (isa<MemRefType>(value.getType())) {
      if (failed(copyTo(value, output, loc))) return failure();
    } else {
      value = scalarCast(loc, value,
          cast<MemRefType>(output.getType()).getElementType());
      if (!value) return failure();
      b.create<dsa::FillOp>(loc, output, value);
    }
    outputs.push_back(output);
  }
  auto savedValues = values;
  auto savedProducts = products;
  auto savedBindings = axisBindings;
  int64_t capacity = std::min(config.getRegionTile(), shape[axis].capacity);
  LogicalResult status = loop(loc, index(loc, 0), shape[axis].count,
      index(loc, capacity), [&](Value begin) -> LogicalResult {
    values = savedValues;
    products = savedProducts;
    axisBindings = savedBindings;
    Value count = b.create<arith::MinSIOp>(loc,
        sub(loc, shape[axis].count, begin), index(loc, capacity));
    axisBindings[dimension] = {shape[axis].extent, begin, count, capacity};
    DenseSet<Value> visited;
    for (Value source : reduce.getSources()) forgetReplayedTensors(source, visited);
    auto inputs = flatten(reduce.getSources());
    if (inputs.size() != outputs.size() || llvm::is_contained(inputs, Value()))
      return reduce.emitError("DSA streamed reduction source fields are unavailable");
    return emitSliceReduction(reduce, inputs, outputs, captures, outputs,
                              {static_cast<int64_t>(axis)});
  });
  values = std::move(savedValues);
  products = std::move(savedProducts);
  axisBindings = std::move(savedBindings);
  if (failed(status)) return failure();
  SmallVector<Type> resultTypes;
  appendProductLeafTypes(reduce->getResultTypes(), resultTypes);
  SmallVector<Value> results;
  for (auto [type, output] : llvm::zip(resultTypes, outputs))
    results.push_back(isa<RankedTensorType>(type) ? output
        : scalarCast(loc, loadLocal(loc, output, {}), type));
  auto grouped = splitFields(reduce->getResultTypes(), results);
  for (auto [result, fields] : llvm::zip(reduce->getResults(), grouped))
    bindProduct(result, fields);
  return success();
}

std::optional<LogicalResult> Construction::stageProductReduction(ReduceOp reduce) {
  SmallVector<Value> sources;
  std::function<bool(Value)> collect = [&](Value source) {
    while (auto extract = source.getDefiningOp<ExtractOp>()) {
      Operation *aggregate = extract.getProduct().getDefiningOp();
      if (!isa_and_nonnull<MakeRecordOp, MakeTupleOp>(aggregate)) break;
      source = aggregate->getOperand(extract.getField());
    }
    if (!getProductComponents(source.getType())) {
      sources.push_back(source);
      return true;
    }
    Operation *definition = source.getDefiningOp();
    return isa_and_nonnull<MakeRecordOp, MakeTupleOp>(definition) &&
        llvm::all_of(definition->getOperands(), collect);
  };
  SmallVector<Type> stateTypes;
  appendProductLeafTypes(reduce.getIdentities().getTypes(), stateTypes);
  if (!llvm::all_of(reduce.getSources(), collect) || sources.size() < 2 ||
      llvm::any_of(stateTypes, [](Type type) { return isa<RankedTensorType>(type); }) ||
      reduce.getAxes().size() != 2 ||
      cast<IntegerAttr>(reduce.getAxes()[0]).getInt() != 0 ||
      cast<IntegerAttr>(reduce.getAxes()[1]).getInt() != 1) return std::nullopt;
  auto first = dyn_cast<RankedTensorType>(sources.front().getType());
  if (!first || first.getRank() != 2) return std::nullopt;
  auto shape = localShape(sources.front(), reduce.getLoc());
  if (failed(shape)) return failure();
  if (!completeShape(*shape)) return std::nullopt;
  for (const LocalAxis &axis : *shape) {
    APInt count;
    if (!matchPattern(axis.count, m_ConstantInt(&count)) ||
        count.getSExtValue() != axis.capacity) return std::nullopt;
  }
  auto ids = cast<TensorShapeAttr>(first.getEncoding()).getDimensions();
  int64_t width = 1;
  while (width * 2 <= std::min(config.getRegionTile(), (*shape)[1].capacity))
    width *= 2;
  if (width < 2 || (*shape)[1].capacity % width) return std::nullopt;
  for (Value source : sources) {
    auto type = dyn_cast<RankedTensorType>(source.getType());
    if (!type || type.getRank() != 2 ||
        cast<TensorShapeAttr>(type.getEncoding()).getDimensions() != ids ||
        !equalAxisExtent(sources.front(), 0, source, 0) ||
        !equalAxisExtent(sources.front(), 1, source, 1)) return std::nullopt;
    for (int64_t dimension : ids.asArrayRef()) {
      DenseSet<Value> visited;
      if (!replayableSlice(source, dimension, visited)) return std::nullopt;
    }
  }
  Location loc = reduce.getLoc();
  auto initial = flatten(reduce.getIdentities());
  auto captures = flatten(reduce.getCaptures());
  if (initial.size() != sources.size()) return failure();
  auto savedValues = values;
  auto savedProducts = products;
  auto savedBindings = axisBindings;
  auto load = [&](Value linear) -> FailureOr<SmallVector<Value>> {
    values = savedValues;
    products = savedProducts;
    axisBindings = savedBindings;
    Value spatial = index(loc, (*shape)[1].capacity);
    Value batch = b.create<arith::DivSIOp>(loc, linear, spatial);
    Value column = b.create<arith::RemSIOp>(loc, linear, spatial);
    axisBindings[ids[0]] = {(*shape)[0].extent, batch, index(loc, 1), 1};
    axisBindings[ids[1]] = {(*shape)[1].extent, column, index(loc, width), width};
    DenseSet<Value> visited;
    for (Value source : sources) forgetReplayedTensors(source, visited);
    SmallVector<Value> inputs;
    for (Value source : sources) {
      Value input = get(source);
      if (!input) return failure();
      inputs.push_back(input);
    }
    return inputs;
  };
  auto firstTile = load(index(loc, 0));
  if (failed(firstTile)) return failure();
  Value lanes = index(loc, width);
  LocalShape laneShape{{lanes, index(loc, 0), lanes, width}};
  SmallVector<Value> states;
  for (Value input : *firstTile) {
    Value state = allocateTensor(loc,
        cast<MemRefType>(input.getType()).getElementType(), laneShape);
    if (failed(copyTo(input, state, loc))) return failure();
    states.push_back(state);
  }
  Value total = index(loc, (*shape)[0].capacity * (*shape)[1].capacity);
  LogicalResult status = loop(loc, index(loc, width), total, index(loc, width),
      [&](Value position) -> LogicalResult {
    auto inputs = load(position);
    if (failed(inputs)) return failure();
    // A one-member reduction preserves the lane states. The same current-IR
    // helper is subsequently reduced across lanes; no second combine evaluator.
    return emitSliceReduction(reduce, *inputs, states, captures, states, {0});
  });
  values = std::move(savedValues);
  products = std::move(savedProducts);
  axisBindings = std::move(savedBindings);
  if (failed(status)) return failure();
  SmallVector<Value> outputs;
  for (Value state : states)
    outputs.push_back(allocateTensor(loc,
        cast<MemRefType>(state.getType()).getElementType(), {}));
  if (failed(emitSliceReduction(reduce, states, initial, captures, outputs, {0})))
    return failure();
  SmallVector<Type> resultTypes;
  appendProductLeafTypes(reduce->getResultTypes(), resultTypes);
  SmallVector<Value> results;
  for (auto [type, output] : llvm::zip(resultTypes, outputs))
    results.push_back(isa<RankedTensorType>(type) ? output
        : scalarCast(loc, loadLocal(loc, output, {}), type));
  auto grouped = splitFields(reduce->getResultTypes(), results);
  for (auto [result, fields] : llvm::zip(reduce->getResults(), grouped))
    bindProduct(result, fields);
  return success();
}

LogicalResult Construction::reduceTensor(ReduceOp reduce) {
  Location loc = reduce.getLoc();
  if (auto staged = stageProductReduction(reduce)) return *staged;
  SmallVector<int64_t> axes;
  for (Attribute axis : reduce.getAxes())
    axes.push_back(cast<IntegerAttr>(axis).getInt());
  SmallVector<Type> sourceTypes;
  appendProductLeafTypes(reduce.getSources().getTypes(), sourceTypes);
  if (axes.size() == 1 && valueSlices.empty() &&
      isa<RankedTensorType>(reduce.getSources().front().getType())) {
    auto first = cast<RankedTensorType>(sourceTypes.front());
    unsigned axis = axes.front();
    auto shape = localShape(reduce.getSources().front(), loc);
    if (failed(shape)) return failure();
    // A reduction consumes the materialized producer DAG, not only its final
    // tensor. Count distinct local snapshots before selecting an unsliced row.
    int64_t sourceBytes = 0;
    DenseSet<Value> sized;
    std::function<LogicalResult(Value)> account = [&](Value value) {
      if (!sized.insert(value).second) return success();
      if (auto type = dyn_cast<RankedTensorType>(value.getType())) {
        auto local = localShape(value, loc);
        if (failed(local)) return failure();
        int64_t bytes = std::max<int64_t>(1,
            (storageElement(type.getElementType()).getIntOrFloatBitWidth() + 7) / 8);
        for (const LocalAxis &axis : *local) {
          if (bytes > config.getLocalBytes() / axis.capacity) {
            sourceBytes = config.getLocalBytes();
            return success();
          }
          bytes *= axis.capacity;
        }
        sourceBytes += std::min<int64_t>(bytes, config.getLocalBytes() - sourceBytes);
      }
      Operation *producer = value.getDefiningOp();
      if (!producer || producer->getNumRegions() || values.lookupOrNull(value))
        return success();
      for (Value operand : producer->getOperands())
        if (failed(account(operand))) return failure();
      return success();
    };
    for (Value source : reduce.getSources())
      if (failed(account(source))) return failure();
    int64_t dimension = cast<TensorShapeAttr>(first.getEncoding()).getDimensions()[axis];
    bool stream = (*shape)[axis].capacity > config.getRegionTile() &&
        sourceBytes > config.getLocalBytes() / 2 && completeShape(*shape);
    for (const auto &field : logicalComponents(reduce.getSources())) {
      auto type = dyn_cast<RankedTensorType>(field.type);
      if (!type || axis >= type.getRank()) { stream = false; break; }
      auto ids = cast<TensorShapeAttr>(type.getEncoding()).getDimensions();
      stream &= ids[axis] == dimension && llvm::count(ids.asArrayRef(), dimension) == 1;
      stream &= equalAxisExtent(reduce.getSources().front(), axis,
                                 field.value, axis, {}, field.path);
    }
    DenseSet<Value> visited;
    for (Value source : reduce.getSources()) stream &= replayableSlice(source, dimension, visited);
    if (stream) return streamReduction(reduce, axis, dimension, *shape);
  }
  auto inputs = flatten(reduce.getSources());
  auto initial = flatten(reduce.getIdentities());
  auto captures = flatten(reduce.getCaptures());
  if (inputs.empty() || inputs.size() != initial.size() ||
      llvm::is_contained(inputs, Value()))
    return reduce.emitError("DSA reduction has unbound source fields");
  SmallVector<Value> outputs;
  for (Value input : inputs) {
    auto found = localShapes.find(input);
    if (found == localShapes.end())
      return reduce.emitError("DSA reduction source has no bounded tensor shape");
    LocalShape state;
    for (auto [axis, dimension] : llvm::enumerate(found->second))
      if (!llvm::is_contained(axes, axis)) state.push_back(dimension);
    outputs.push_back(allocateTensor(loc,
        cast<MemRefType>(input.getType()).getElementType(), state));
  }
  if (failed(emitSliceReduction(reduce, inputs, initial, captures, outputs, axes)))
    return failure();
  SmallVector<Type> resultTypes;
  appendProductLeafTypes(reduce->getResultTypes(), resultTypes);
  SmallVector<Value> results;
  for (auto [type, output] : llvm::zip(resultTypes, outputs))
    results.push_back(isa<RankedTensorType>(type) ? output
        : scalarCast(loc, loadLocal(loc, output, {}), type));
  auto grouped = splitFields(reduce->getResultTypes(), results);
  for (auto [result, fields] : llvm::zip(reduce->getResults(), grouped))
    bindProduct(result, fields);
  return success();
}

LogicalResult Construction::scanTensor(ScanOp scan) {
  Location loc = scan.getLoc();
  auto inputs = flatten(scan.getSources());
  auto initial = flatten(scan.getIdentities());
  auto captures = flatten(scan.getCaptures());
  if (inputs.empty() || inputs.size() != initial.size() ||
      llvm::is_contained(inputs, Value()))
    return scan.emitError("DSA scan has unbound source or identity fields");
  SmallVector<Value> outputs, finals;
  for (Value input : inputs) {
    auto found = localShapes.find(input);
    if (found == localShapes.end() || scan.getAxis() >= found->second.size())
      return scan.emitError("DSA scan source requires a bounded local shape");
    LocalShape shape = found->second;
    Type element = cast<MemRefType>(input.getType()).getElementType();
    outputs.push_back(allocateTensor(loc, element, shape));
    shape.erase(shape.begin() + scan.getAxis());
    finals.push_back(allocateTensor(loc, element, shape));
  }
  if (failed(emitScan(scan, inputs, initial, captures, outputs, finals)))
    return failure();
  auto grouped = splitFields(scan->getResultTypes(), outputs);
  for (auto [result, fields] : llvm::zip(scan->getResults(), grouped))
    bindProduct(result, fields);
  return success();
}

bool Construction::canStreamScan(ScanOp scan, Block &block) {
  unsigned axis = scan.getAxis();
  auto sources = scan.getSources();
  auto identities = scan.getIdentities();
  auto first = cast<RankedTensorType>(sources.front().getType());
  int64_t dimension = cast<TensorShapeAttr>(first.getEncoding()).getDimensions()[axis];
  if (dimension <= 0 || axisBindings.count(dimension) ||
      llvm::none_of(identities, [](Value value) { return isa<RankedTensorType>(value.getType()); })) return false;
  for (auto [source, identity] : llvm::zip(sources, identities)) {
    auto type = cast<RankedTensorType>(source.getType());
    auto ids = cast<TensorShapeAttr>(type.getEncoding()).getDimensions();
    auto state = dyn_cast<RankedTensorType>(identity.getType());
    if (axis >= type.getRank() || ids[axis] != dimension || llvm::count(ids.asArrayRef(), dimension) != 1 ||
        type.getRank() != (state ? state.getRank() : 0) + 1) return false;
    DenseSet<Value> visited;
    if (!replayableSlice(source, dimension, visited)) return false;
  }
  // All observable consumers must keep the prefix position free. A pair of
  // prefix axes, a reduction over it, or a nonlocal lookup needs materialization.
  DenseSet<Operation *> dependent;
  SmallVector<Value> pending(scan->getResults());
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    for (Operation *user : value.getUsers()) {
      if (user->getBlock() != &block || user == block.getTerminator()) return false;
      if (dependent.insert(user).second) llvm::append_range(pending, user->getResults());
    }
  }
  auto ids = [](Value value) -> ArrayRef<int64_t> {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    return type ? cast<TensorShapeAttr>(type.getEncoding()).getDimensions().asArrayRef() : ArrayRef<int64_t>();
  };
  bool after = false, hasOutput = false;
  for (Operation &op : block.without_terminator()) {
    if (&op == scan) { after = true; continue; }
    if (auto store = dyn_cast<ViewStoreOp>(op)) {
      if (!after || !dependent.contains(&op) || llvm::count(ids(store.getValue()), dimension) != 1)
        return false;
      hasOutput = true;
    } else if (auto load = dyn_cast<ViewLoadOp>(op)) {
      if (cast<ViewType>(load.getSource().getType()).getAccess() != 0) return false;
    } else if (!isMemoryEffectFree(&op) && !isa<AssumeInBoundsOp>(op)) return false;
    if (isa<ScanOp, RegionFoldOp, RegionScanOp, ForOp, WhileOp, IfOp, ParallelOp, ReshapeOp, JoinOp>(op)) return false;
    for (Type type : op.getResultTypes()) {
      SmallVector<Type> fields; appendProductLeafTypes(type, fields);
      for (Type field : fields) if (auto tensor = dyn_cast<RankedTensorType>(field))
        if (llvm::count(cast<TensorShapeAttr>(tensor.getEncoding()).getDimensions().asArrayRef(), dimension) > 1) return false;
    }
    if (auto reduce = dyn_cast<ReduceOp>(op))
      for (Value source : reduce.getSources()) {
        SmallVector<Type> fields; appendProductLeafTypes(source.getType(), fields);
        for (Type field : fields) if (auto tensor = dyn_cast<RankedTensorType>(field))
          for (Attribute reduced : reduce.getAxes())
            if (cast<TensorShapeAttr>(tensor.getEncoding()).getDimensions()[cast<IntegerAttr>(reduced).getInt()] == dimension) return false;
      }
    if (auto matrix = dyn_cast<ContractOp>(op)) {
      if (!matrix.getBatch().empty()) return false;
      for (Attribute entry : matrix.getReduce()) {
        auto pair = cast<ArrayAttr>(entry);
        if (ids(matrix.getLhs())[cast<IntegerAttr>(pair[0]).getInt()] == dimension ||
            ids(matrix.getRhs())[cast<IntegerAttr>(pair[1]).getInt()] == dimension) return false;
      }
    }
    if (auto gather = dyn_cast<GatherOp>(op)) {
      auto relation = analysis.indexRelation(gather);
      if (failed(relation)) return false;
      for (const auto &term : relation->terms)
        if (term.sourceAxis && ids(relation->source)[*term.sourceAxis] == dimension && term.kind != 0) return false;
    }
  }
  return hasOutput;
}

LogicalResult Construction::streamScan(ScanOp scan, Block &block) {
  Location loc = scan.getLoc();
  unsigned axis = scan.getAxis();
  auto sources = scan.getSources();
  auto first = cast<RankedTensorType>(sources.front().getType());
  int64_t dimension = cast<TensorShapeAttr>(first.getEncoding()).getDimensions()[axis];
  for (Operation &op : block.without_terminator()) {
    if (&op == scan) break;
    if (!canDefer(&op) && failed(lowerOperation(&op))) return failure();
  }
  Value size = logicalExtent(scan.getSources().front(), axis, loc);
  if (!size) return scan.emitError("DSA scan traversal extent is unavailable");
  auto initial = flatten(scan.getIdentities());
  auto states = makeSlots(scan.getIdentities(), loc);
  if (failed(states) || initial.size() != states->size()) return failure();
  for (auto [value, state] : llvm::zip(initial, *states))
    if (failed(copyTo(value, state, loc))) return failure();
  auto captures = flatten(scan.getCaptures());
  auto savedValues = values;
  auto savedProducts = products;
  auto savedAxes = axisBindings;
  LogicalResult status = loop(loc, index(loc, 0), size, index(loc, 1),
      [&](Value position) -> LogicalResult {
    Value i = scan.getReverse()
        ? sub(loc, sub(loc, size, index(loc, 1)), position) : position;
    axisBindings[dimension] = {size, i, index(loc, 1), 1};
    DenseSet<Value> replayed;
    for (Value source : sources) forgetReplayedTensors(source, replayed);
    auto inputs = flatten(sources);
    if (inputs.size() != states->size() || llvm::is_contained(inputs, Value()))
      return failure();
    SmallVector<Value> outputs;
    for (Value input : inputs)
      outputs.push_back(allocateTensor(loc,
          cast<MemRefType>(input.getType()).getElementType(), localShapes.lookup(input)));
    if (failed(emitScan(scan, inputs, *states, captures, outputs, *states)))
      return failure();
    auto grouped = splitFields(scan->getResultTypes(), outputs);
    for (auto [result, fields] : llvm::zip(scan->getResults(), grouped))
      bindProduct(result, fields);
    for (Operation *op = scan->getNextNode(); op && op != block.getTerminator();
         op = op->getNextNode())
      if (!canDefer(op) && failed(lowerOperation(op))) return failure();
    return success();
  });
  values = std::move(savedValues);
  products = std::move(savedProducts);
  axisBindings = std::move(savedAxes);
  return status;
}

} // namespace intent::kir_to_dsa
