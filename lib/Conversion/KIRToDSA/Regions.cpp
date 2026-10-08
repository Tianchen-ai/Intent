#include "Construction.h"

namespace intent::kir_to_dsa {

FailureOr<Value> Construction::partitionQueryAxis(Block &block, RegionFoldOp fold) {
  auto sourceType = cast<RankedTensorType>(fold.getOperands().front().getType());
  auto sourceIds = cast<TensorShapeAttr>(sourceType.getEncoding()).getDimensions();
  int64_t query = 0;
  Value querySource;
  for (Operation &op : block.without_terminator()) if (auto store = dyn_cast<ViewStoreOp>(op)) {
    auto type = dyn_cast<RankedTensorType>(store.getValue().getType());
    if (!type || !type.getRank()) continue;
    int64_t candidate = cast<TensorShapeAttr>(type.getEncoding()).getDimensions()[0];
    if (candidate > 0 && !llvm::is_contained(sourceIds.asArrayRef(), candidate)) {
      if (query && query != candidate)
        return store.emitError("DSA fold outputs require different work ownership projections"), failure();
      query = candidate;
      querySource = store.getValue();
    }
  }
  if (!query) return Value();
  // Slicing is legal only when this axis stays free throughout the author's
  // helpers. Reducing it or indexing it nonlocally requires another plan.
  bool independent = true;
  block.walk([&](Operation *op) {
    auto ids = [&](Value value) -> ArrayRef<int64_t> {
      auto type = dyn_cast<RankedTensorType>(value.getType());
      return type ? cast<TensorShapeAttr>(type.getEncoding()).getDimensions().asArrayRef() : ArrayRef<int64_t>();
    };
    if (isa<ForOp, WhileOp, IfOp, ScanOp, BufferLoadOp, BufferStoreOp>(op)) independent = false;
    if (!op->hasTrait<OpTrait::IsTerminator>() && !isa<ViewLoadOp, ViewStoreOp, AssumeInBoundsOp>(op) &&
        !isMemoryEffectFree(op)) independent = false;
    for (Type result : op->getResultTypes()) {
      SmallVector<Type> fields; appendProductLeafTypes(result, fields);
      for (Type field : fields) if (auto tensor = dyn_cast<RankedTensorType>(field))
        if (llvm::count(cast<TensorShapeAttr>(tensor.getEncoding()).getDimensions().asArrayRef(), query) > 1) independent = false;
    }
    if (auto store = dyn_cast<ViewStoreOp>(op))
      if (llvm::count(ids(store.getValue()), query) != 1) independent = false;
    if (isa<RegionFoldOp, RegionScanOp>(op) && analysis.regionSegment(op).dimensionIdentity == query) independent = false;
    if (auto load = dyn_cast<ViewLoadOp>(op))
      if (cast<ViewType>(load.getSource().getType()).getAccess() != 0) independent = false;
    if (auto reduce = dyn_cast<ReduceOp>(op))
      for (Value source : reduce.getSources()) {
        SmallVector<Type> fields; appendProductLeafTypes(source.getType(), fields);
        for (Type field : fields) if (auto tensor = dyn_cast<RankedTensorType>(field))
          for (Attribute axis : reduce.getAxes())
            if (cast<TensorShapeAttr>(tensor.getEncoding()).getDimensions()[cast<IntegerAttr>(axis).getInt()] == query) independent = false;
      }
    if (auto matrix = dyn_cast<ContractOp>(op)) {
      for (Attribute entry : matrix.getReduce()) {
        auto pair = cast<ArrayAttr>(entry);
        independent &= ids(matrix.getLhs())[cast<IntegerAttr>(pair[0]).getInt()] != query;
        independent &= ids(matrix.getRhs())[cast<IntegerAttr>(pair[1]).getInt()] != query;
      }
      if (!matrix.getBatch().empty()) independent = false;
    }
    if (isa<GatherOp, ViewLoadOp, ViewStoreOp>(op)) {
      auto fact = analysis.indexRelation(op);
      if (failed(fact)) { independent = false; return; }
      if (auto store = dyn_cast<ViewStoreOp>(op)) {
        auto outputIds = ids(store.getValue());
        if (ArrayRef<int64_t>(fact->resultDimensionIdentities) != outputIds) {
          independent = false;
          return;
        }
        unsigned ownedAxes = 0;
        for (const auto &term : fact->terms) {
          unsigned queryAxes = llvm::count_if(term.resultAxes, [&](unsigned axis) {
            return outputIds[axis] == query;
          });
          if (!queryAxes) continue;
          bool interval = false;
          if (term.kind == 4 && term.operands.size() == 1) {
            Operation *domain = term.operands.front().getDefiningOp();
            if (isa_and_nonnull<DomainOp, SubregionOp>(domain)) {
              auto extents = domain->getAttrOfType<ArrayAttr>("extent_dimensions");
              interval = extents && extents.size() == 1 && cast<IntegerAttr>(extents[0]).getInt() == query;
            }
          }
          independent &= term.sourceAxis.has_value() && (term.kind == 0 || interval);
          ownedAxes += queryAxes;
        }
        independent &= ownedAxes == 1;
      } else if (auto sourceType = dyn_cast<RankedTensorType>(fact->source.getType())) {
        auto dims = ids(fact->source);
        for (const auto &term : fact->terms)
          if (term.sourceAxis && dims[*term.sourceAxis] == query && term.kind != 0) independent = false;
      }
    }
    if (isa<ReshapeOp, JoinOp>(op)) independent = false;
  });
  if (!independent)
    return fold.emitError("DSA query partition requires an axis preserved by the region value graph"), failure();
  return querySource;
}

bool Construction::replayableSlice(Value value, int64_t dimension, DenseSet<Value> &visited) {
  if (!visited.insert(value).second) return true;
  auto tensor = dyn_cast<RankedTensorType>(value.getType());
  if (!tensor && !getProductComponents(value.getType())) return true;
  Operation *op = value.getDefiningOp();
  if (!op) return false;
  if (auto load = dyn_cast<ViewLoadOp>(op)) {
    if (cast<ViewType>(load.getSource().getType()).getAccess() != 0) return false;
  } else if (!isMemoryEffectFree(op)) return false;
  if (isa<RegionFoldOp, RegionScanOp, ScanOp, ReshapeOp, JoinOp>(op)) return false;
  auto containsAxis = [&](Value input, unsigned axis) {
    SmallVector<Type> fields; appendProductLeafTypes(input.getType(), fields);
    for (Type field : fields) if (auto tensor = dyn_cast<RankedTensorType>(field))
      if (cast<TensorShapeAttr>(tensor.getEncoding()).getDimensions()[axis] == dimension) return true;
    return false;
  };
  if (auto reduce = dyn_cast<ReduceOp>(op))
    for (Value input : reduce.getSources())
      for (Attribute axis : reduce.getAxes()) if (containsAxis(input, cast<IntegerAttr>(axis).getInt())) return false;
  if (auto matrix = dyn_cast<ContractOp>(op)) {
    if (!matrix.getBatch().empty()) return false;
    for (Attribute entry : matrix.getReduce()) {
      auto pair = cast<ArrayAttr>(entry);
      if (containsAxis(matrix.getLhs(), cast<IntegerAttr>(pair[0]).getInt()) ||
          containsAxis(matrix.getRhs(), cast<IntegerAttr>(pair[1]).getInt())) return false;
    }
  }
  if (auto gather = dyn_cast<GatherOp>(op)) {
    auto fact = analysis.indexRelation(gather);
    if (failed(fact)) return false;
    for (const auto &term : fact->terms)
      if (term.sourceAxis && containsAxis(fact->source, *term.sourceAxis) && term.kind != 0) return false;
  }
  for (Value operand : op->getOperands()) if (!replayableSlice(operand, dimension, visited)) return false;
  return true;
}

void Construction::forgetReplayedTensors(Value value, DenseSet<Value> &visited) {
  if (!visited.insert(value).second || (!isa<RankedTensorType>(value.getType()) && !getProductComponents(value.getType()))) return;
  Operation *op = value.getDefiningOp();
  if (!op) return;
  values.erase(value); products.erase(value);
  for (Value operand : op->getOperands()) forgetReplayedTensors(operand, visited);
}

FailureOr<SmallVector<Operation *>> Construction::regionConsumers(Operation *region, unsigned outputCount, int64_t dimension) {
  DenseSet<Operation *> consumers;
  SmallVector<Value> pending(region->getResults().take_front(outputCount));
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    for (Operation *user : value.getUsers()) {
      if (user->getBlock() != region->getBlock() || user == region->getBlock()->getTerminator())
        return region->emitError("DSA streamed region outputs require consumers in the same workset"), failure();
      if (!consumers.insert(user).second) continue;
      bool write = isa<ViewStoreOp, ScatterUniqueOp>(user);
      if (!write && ((!isMemoryEffectFree(user) && !isa<AssumeInBoundsOp>(user)) || user->getNumRegions()))
        return user->emitError("DSA streamed region consumer requires a pure position-preserving computation"), failure();
      if (isa<ReshapeOp, JoinOp>(user))
        return user->emitError("DSA streamed region consumer needs an explicit position mapping"), failure();
      for (Type type : user->getResultTypes()) {
        SmallVector<Type> fields; appendProductLeafTypes(type, fields);
        for (Type field : fields) if (auto tensor = dyn_cast<RankedTensorType>(field))
          if (llvm::count(cast<TensorShapeAttr>(tensor.getEncoding()).getDimensions().asArrayRef(), dimension) > 1)
            return user->emitError("DSA streamed region consumer couples distinct source positions"), failure();
      }
      if (auto matrix = dyn_cast<ContractOp>(user)) {
        for (Attribute entry : matrix.getReduce()) {
          auto pair = cast<ArrayAttr>(entry);
          for (auto [input, axis] : llvm::zip(matrix->getOperands(), pair)) {
            auto ids = cast<TensorShapeAttr>(cast<RankedTensorType>(input.getType()).getEncoding()).getDimensions();
            if (ids[cast<IntegerAttr>(axis).getInt()] == dimension)
              return user->emitError("DSA streamed region consumer reduces the source-position axis"), failure();
          }
        }
      }
      if (auto gather = dyn_cast<GatherOp>(user)) {
        auto relation = analysis.indexRelation(gather);
        if (failed(relation)) return failure();
        auto ids = cast<TensorShapeAttr>(cast<RankedTensorType>(relation->source.getType()).getEncoding()).getDimensions();
        for (const auto &term : relation->terms)
          if (term.sourceAxis && ids[*term.sourceAxis] == dimension && term.kind != 0)
            return user->emitError("DSA streamed region consumer performs a nonlocal lookup"), failure();
      }
      if (write) {
        Value data = cast<IndexedAccessOpInterface>(user).getStoredValue();
        auto type = dyn_cast<RankedTensorType>(data.getType());
        if (!type || llvm::count(cast<TensorShapeAttr>(type.getEncoding()).getDimensions().asArrayRef(), dimension) != 1)
          return user->emitError("DSA streamed region output must preserve one source-position axis"), failure();
      }
      llvm::append_range(pending, user->getResults());
    }
  }
  SmallVector<Operation *> ordered;
  DenseSet<Value> checkedFinal;
  std::function<bool(Value)> requiresFinal = [&](Value value) {
    if (value.getDefiningOp() == region) return cast<OpResult>(value).getResultNumber() >= outputCount;
    if (!checkedFinal.insert(value).second) return false;
    Operation *definition = value.getDefiningOp();
    return definition && llvm::any_of(definition->getOperands(), requiresFinal);
  };
  for (Operation *op = region->getNextNode(); op && op != region->getBlock()->getTerminator(); op = op->getNextNode()) {
    if (consumers.contains(op)) {
      for (Value operand : op->getOperands()) {
        if (requiresFinal(operand))
          return op->emitError("DSA region output consumer requires the final state before traversal completes"), failure();
        if (operand.getDefiningOp() == region) {
          continue;
        }
        if (consumers.contains(operand.getDefiningOp())) continue;
        DenseSet<Value> visited;
        if (!replayableSlice(operand, dimension, visited))
          return op->emitError("DSA region consumer input needs complete materialization before streaming"), failure();
      }
      ordered.push_back(op);
    }
  }
  if (!ordered.empty()) {
    for (Operation *between = region->getNextNode(); between != ordered.back(); between = between->getNextNode()) {
      if (consumers.contains(between) || isMemoryEffectFree(between) || isa<AssumeInBoundsOp>(between)) continue;
      if (auto load = dyn_cast<ViewLoadOp>(between))
        if (cast<ViewType>(load.getSource().getType()).getAccess() == 0) continue;
      return between->emitError("DSA streaming output cannot cross an observable access"), failure();
    }
  }
  return ordered;
}

Value Construction::regionTraversalEnd(RegionFoldOp fold, Value size, int64_t sourceDimension) {
  Block &body = fold.getSummarize().front();
  auto schema = cast<StructuredOpInterface>(fold.getOperation());
  auto coordinate = [&](Value value) -> std::optional<Domain> {
    DenseSet<Value> visited;
    while (value && visited.insert(value).second) {
      if (auto argument = dyn_cast<BlockArgument>(value)) {
        if (argument.getOwner() != &body) return {};
        Value source;
        for (const auto &relation : schema.getValueRelations())
          if (relation.to == argument &&
              (relation.kind == StructuredRelationKind::Capture ||
               relation.kind == StructuredRelationKind::SourceSlice)) {
            source = relation.from;
            break;
          }
        if (!source) return std::nullopt;
        value = source;
        continue;
      }
      Operation *definition = value.getDefiningOp();
      if (isa_and_nonnull<BroadcastOp, ReshapeOp, TransposeOp>(definition)) {
        value = definition->getOperand(0); continue;
      }
      if (auto gather = dyn_cast_or_null<GatherOp>(definition)) {
        auto relation = analysis.indexRelation(gather);
        Value valid = gather.getValid();
        if (failed(relation) || (valid && !constantTrue(valid)) ||
            llvm::any_of(relation->terms, [](const IndexTermFact &term) {
              return !term.operands.empty() || (term.kind != 0 && term.kind != 1);
            })) return {};
        value = relation->source; continue;
      }
      auto indices = dyn_cast_or_null<IndicesOp>(definition);
      if (!indices) return {};
      auto found = domains.find(indices.getSource());
      if (found == domains.end() || !matchPattern(found->second.step, m_One())) return {};
      return found->second;
    }
    return {};
  };
  UniformValueAnalysis facts(describeCanonicalUniformValue);
  Value end = size;
  body.walk([&](CompareOp comparison) {
    auto lhs = coordinate(comparison.getLhs()), rhs = coordinate(comparison.getRhs());
    if (!lhs || !rhs) return;
    auto predicate = comparison.getPredicate();
    if (rhs->dimension == sourceDimension) {
      std::swap(lhs, rhs);
      switch (predicate) {
      case ComparePredicate::Gt: predicate = ComparePredicate::Lt; break;
      case ComparePredicate::Ge: predicate = ComparePredicate::Le; break;
      default: return;
      }
    }
    if (lhs->dimension != sourceDimension || rhs->dimension == sourceDimension ||
        (predicate != ComparePredicate::Lt && predicate != ComparePredicate::Le)) return;
    auto owner = axisBindings.find(rhs->dimension);
    if (owner == axisBindings.end()) return;
    UniformBindings bindings;
    bindings[comparison.getResult()] = b.getBoolAttr(false);
    auto identities = fold.getIdentities();
    auto yields = body.getTerminator()->getOperands();
    if (yields.size() != identities.size()) return;
    for (auto [part, identity] : llvm::zip(yields, identities))
      if (!equalUniformConstants(facts.evaluate(part, bindings), facts.evaluate(identity))) return;
    Location loc = comparison.getLoc();
    auto build = [&](UniformKind kind, Value a, Value c) -> Value {
      switch (kind) {
      case UniformKind::Add: return add(loc, a, c);
      case UniformKind::Subtract: return sub(loc, a, c);
      case UniformKind::Minimum: return b.create<arith::MinSIOp>(loc, a, c);
      case UniformKind::Maximum: return b.create<arith::MaxSIOp>(loc, a, c);
      default: llvm_unreachable("invalid coordinate interval expression");
      }
    };
    Value captureBegin = add(loc, rhs->begin, owner->second.begin);
    auto bounds = partitionCoordinatePredicate(
        predicate == ComparePredicate::Lt ? UniformPredicate::Less : UniformPredicate::LessEqual,
        {lhs->begin, lhs->end}, {captureBegin, add(loc, captureBegin, owner->second.count)},
        index(loc, 0), index(loc, 1), build);
    if (bounds) {
      // Preserve each retained source slice; only skip complete identity-only
      // tail slices. The logical source extent and its numerical tiles stay.
      Value tile = index(loc, config.getRegionTile());
      Value aligned = mul(loc, b.create<arith::CeilDivSIOp>(loc, bounds->possibleEnd, tile), tile);
      end = b.create<arith::MinSIOp>(loc, end, aligned);
    }
  });
  return end;
}

FailureOr<SmallVector<Value>> Construction::jointSummary(RegionFoldOp fold, OnlineSummary plan,
    ArrayRef<SmallVector<Value>> arguments, ArrayRef<Value> state) {
  auto savedValues = values; auto savedProducts = products;
  auto savedAxes = axisBindings;
  auto savedShapes = localShapes; auto savedStreamed = streamedOperations;
  auto restore = llvm::make_scope_exit([&] {
    values = std::move(savedValues); products = std::move(savedProducts);
    axisBindings = std::move(savedAxes);
    streamedOperations = std::move(savedStreamed);
    for (auto &entry : savedShapes) localShapes[entry.first] = entry.second;
  });
  Location loc = fold.getLoc();
  auto &summary = fold.getSummarize().front();
  auto schema = cast<StructuredOpInterface>(fold.getOperation());
  for (auto [i, formal] : llvm::enumerate(summary.getArguments()))
    bindHelperValue(formal, arguments[i], llvm::is_contained(schema.getSummarizeSources(), formal), fold.getAxis());
  Value valid = get(plan.validity.getResult(0));
  Value maximum = get(plan.maximumOrEmpty.getResult());
  if (!valid || !maximum) return failure();
  bindHelperValue(schema.getCombineLhs().front(), state, false, -1);
  SmallVector<Value> part(state);
  part[plan.validField] = valid; part[plan.maxField] = maximum;
  bindHelperValue(schema.getCombineRhs().front(), part, false, -1);
  Value combinedMaximum = get(plan.combinedMaximum);
  Value massSeed = get(plan.leftMassTerm), momentSeed = get(plan.leftMomentTerm);
  if (!combinedMaximum || !massSeed || !momentSeed) return failure();
  // The region's probabilities now use the same reference as the running
  // summary. Preserve the author's exp/cast contract and source membership.
  values.map(plan.maximumOrEmpty.getResult(), combinedMaximum);
  Value mass = get(plan.mass.getResult(0));
  if (!mass || !get(plan.probabilityCast.getResult())) return failure();
  auto shape = localShape(plan.moment.getResult(), loc);
  if (failed(shape)) return failure();
  Value moment = allocateTensor(loc, b.getF32Type(), *shape);
  if (failed(copyTo(momentSeed, moment, loc)) ||
      failed(localMatMul(plan.moment, *shape, moment, false))) return failure();
  if (plan.momentOrEmpty) {
    values.map(plan.momentOrEmpty.getFalseValue(), momentSeed);
    moment = get(plan.momentOrEmpty.getResult());
    if (!moment) return failure();
  }
  Value denominator = allocateLike(loc, mass);
  b.create<dsa::BinaryOp>(loc, massSeed, mass, denominator,
      BinaryOperatorAttr::get(b.getContext(), BinaryOperator::Add),
      b.getBoolAttr(false), b.getBoolAttr(false), Value());
  localShapes[denominator] = localShapes.lookup(mass);
  Value combinedValid = get(plan.merged.getFields()[plan.validField]);
  if (!combinedValid) return failure();
  SmallVector<Value> result(4);
  result[plan.validField] = combinedValid; result[plan.maxField] = combinedMaximum;
  result[plan.massField] = denominator; result[plan.momentField] = moment;
  return result;
}

LogicalResult Construction::lowerRegion(Operation *op) {
  Location loc = op->getLoc();
  auto schema = cast<StructuredOpInterface>(op);
  bool scan = schema.getStructuredKind() == StructuredOpKind::RegionScan;
  ValueRange sources = schema.getSources(), identities = schema.getIdentities();
  ValueRange states = schema.getInitialStates(), captures = schema.getCaptures();
  unsigned sourceCount = sources.size();
  unsigned outputCount = schema.getEmittedResults().size();
  int64_t axis = schema.getIterationAxes().front();
  auto fact = analysis.regionSegment(op);
  if (!fact.isExact()) return op->emitError("DSA region source has no exact canonical segment relation");
  auto sourceType = cast<RankedTensorType>(sources.front().getType());
  int64_t dimension = cast<TensorShapeAttr>(sourceType.getEncoding()).getDimensions()[axis];
  for (Value source : sources) {
    auto type = cast<RankedTensorType>(source.getType());
    auto ids = cast<TensorShapeAttr>(type.getEncoding()).getDimensions();
    DenseSet<Value> visited;
    if (llvm::count(ids.asArrayRef(), dimension) != 1 || !replayableSlice(source, dimension, visited))
      return op->emitError("DSA source slicing requires an independent axis and immutable replayable inputs; this source needs explicit snapshot materialization");
  }
  Value size = logicalExtent(sources.front(), axis, loc);
  if (!size) return op->emitError("DSA region source has no bound logical extent");
  if (auto bound = axisBindings.find(dimension); bound != axisBindings.end())
    if (!matchPattern(bound->second.begin, m_Zero()) || !sameIndex(bound->second.count, size) || !sameIndex(bound->second.extent, size))
      return op->emitError("DSA region source requires an unpartitioned logical traversal");
  APInt staticSize;
  if (!scan && matchPattern(size, m_ConstantInt(&staticSize)) && staticSize.isStrictlyPositive() &&
      staticSize.getSExtValue() <= config.getRegionTile()) {
    // One nonempty region already is the final summary. Keep the helper's
    // tensor values directly instead of materializing identity/carry/next
    // buffers and copying the only summary through those buffers.
    SmallVector<SmallVector<Value>> captureFields;
    for (Value capture : captures) captureFields.push_back(flatten(ValueRange{capture}));
    auto savedValues = values; auto savedProducts = products; auto savedAxes = axisBindings;
    axisBindings[dimension] = {size, index(loc, 0), size, config.getRegionTile()};
    DenseSet<Value> replayed;
    for (Value source : sources) forgetReplayedTensors(source, replayed);
    SmallVector<SmallVector<Value>> arguments;
    for (Value source : sources) {
      auto fields = flatten(ValueRange{source});
      if (fields.empty() || llvm::is_contained(fields, Value())) return failure();
      arguments.push_back(std::move(fields));
    }
    llvm::append_range(arguments, captureFields);
    auto part = helper(schema.getSummarizeRegion()->front(), arguments, sourceCount, axis);
    SmallVector<Type> resultFields;
    for (Type type : op->getResultTypes()) appendProductLeafTypes(type, resultFields);
    if (failed(part) || part->size() != resultFields.size()) return failure();
    values = std::move(savedValues); products = std::move(savedProducts); axisBindings = std::move(savedAxes);
    auto fields = splitFields(op->getResultTypes(), *part);
    for (auto [result, group] : llvm::zip(op->getResults(), fields)) bindProduct(result, group);
    return success();
  }
  auto initial = flatten(identities), initialState = flatten(states);
  auto slots = makeSlots(identities, loc), next = makeSlots(identities, loc);
  if (failed(slots) || failed(next) || initial.size() != slots->size()) return failure();
  for (auto [value, slot] : llvm::zip(initial, *slots)) if (failed(copyTo(value, slot, loc))) return failure();
  SmallVector<SmallVector<Value>> captureFields;
  for (Value capture : captures) captureFields.push_back(flatten(ValueRange{capture}));
  auto consumers = regionConsumers(op, outputCount, dimension);
  if (failed(consumers)) return failure();
  auto savedValues = values; auto savedProducts = products; auto savedAxes = axisBindings;
  Value traversalEnd = scan ? size : regionTraversalEnd(cast<RegionFoldOp>(op), size, dimension);
  auto joint = !scan ? matchOnlineSummary(cast<RegionFoldOp>(op)) : std::nullopt;
  if (failed(loop(loc, index(loc, 0), traversalEnd, index(loc, config.getRegionTile()), [&](Value begin) -> LogicalResult {
    Value count = b.create<arith::MinSIOp>(loc, sub(loc, size, begin), index(loc, config.getRegionTile()));
    axisBindings[dimension] = {size, begin, count, config.getRegionTile()};
    SmallVector<SmallVector<Value>> slices;
    DenseSet<Value> replayed;
    for (Value source : sources) forgetReplayedTensors(source, replayed);
    for (Value source : sources) {
      // Source expressions are replayed from immutable views in this slice.
      auto fields = flatten(ValueRange{source});
      if (fields.empty() || llvm::is_contained(fields, Value())) return failure();
      slices.push_back(std::move(fields));
    }
    auto arguments = slices; llvm::append_range(arguments, captureFields);
    if (joint) {
      auto updated = jointSummary(cast<RegionFoldOp>(op), *joint, arguments, *slots);
      if (failed(updated) || updated->size() != slots->size()) return failure();
      for (auto [value, slot] : llvm::zip(*updated, *next)) if (failed(copyTo(value, slot, loc))) return failure();
      for (auto [value, slot] : llvm::zip(*next, *slots)) if (failed(copyTo(value, slot, loc))) return failure();
      return success();
    }
    auto part = helper(schema.getSummarizeRegion()->front(), arguments, sourceCount, axis);
    if (failed(part) || part->size() != slots->size()) return failure();
    if (scan) {
      auto applied = splitFields(identities.getTypes(), *slots);
      llvm::append_range(applied, splitFields(states.getTypes(), initialState));
      auto incoming = helper(schema.getApplyRegion()->front(), applied);
      if (failed(incoming)) return failure();
      auto emittedArgs = slices;
      llvm::append_range(emittedArgs, splitFields(states.getTypes(), *incoming));
      llvm::append_range(emittedArgs, captureFields);
      auto emitted = helper(schema.getEmitRegion()->front(), emittedArgs, sourceCount, axis);
      ValueRange outputValues = schema.getEmittedResults();
      auto outputComponents = logicalComponents(outputValues);
      if (failed(emitted) || emitted->size() != outputComponents.size()) return op->emitError("DSA scan output schema is unavailable");
      for (unsigned i = 0; i < outputComponents.size(); ++i) {
        // Reattach the helper-local slice to the complete output's source
        // coordinates before its view consumers form destination addresses.
        const auto &component = outputComponents[i];
        auto outputAxis = analysis.emissionAxis(cast<OpResult>(component.value),
                                                component.path);
        if (failed(outputAxis))
          return op->emitError("DSA scan output has no unique source member axis");
        LocalShape outputShape = localShapes.lookup((*emitted)[i]);
        outputShape[*outputAxis] = {size, begin, count, config.getRegionTile()};
        localShapes[(*emitted)[i]] = std::move(outputShape);
      }
      auto outputFields = splitFields(outputValues.getTypes(), *emitted);
      for (auto [result, fields] : llvm::zip(outputValues, outputFields)) bindProduct(result, fields);
      for (Operation *consumer : *consumers)
        if (!canDefer(consumer) && failed(lowerOperation(consumer))) return failure();
    }
    // Region composition declares a neutral identity. The first nonempty
    // slice supplies the running summary directly; subsequent slices combine.
    // A single-slice realization therefore needs no summary update program.
    Value first = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, begin, index(loc, 0));
    auto update = b.create<scf::IfOp>(loc, first, true);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(update.thenBlock());
      for (auto [value, slot] : llvm::zip(*part, *slots)) if (failed(copyTo(value, slot, loc))) return failure();
      b.setInsertionPointToStart(update.elseBlock());
      auto combinedArgs = splitFields(identities.getTypes(), *slots);
      llvm::append_range(combinedArgs, splitFields(identities.getTypes(), *part));
      auto updated = helper(schema.getCombine().front(), combinedArgs);
      if (failed(updated) || updated->size() != slots->size()) return failure();
      for (auto [value, slot] : llvm::zip(*updated, *next)) if (failed(copyTo(value, slot, loc))) return failure();
      for (auto [value, slot] : llvm::zip(*next, *slots)) if (failed(copyTo(value, slot, loc))) return failure();
    }
    return success();
  }))) return failure();
  values = std::move(savedValues); products = std::move(savedProducts); axisBindings = std::move(savedAxes);
  if (!scan) { bindSlots(op->getResults(), *slots, loc); return success(); }
  auto applied = splitFields(identities.getTypes(), *slots);
  llvm::append_range(applied, splitFields(states.getTypes(), initialState));
  auto finalState = helper(schema.getApplyRegion()->front(), applied);
  if (failed(finalState)) return failure();
  auto grouped = splitFields(states.getTypes(), *finalState);
  for (auto [result, fields] : llvm::zip(schema.getFinalStates(), grouped)) bindProduct(result, fields);
  for (Operation *consumer : *consumers) streamedOperations.insert(consumer);
  return success();
}

} // namespace intent::kir_to_dsa
