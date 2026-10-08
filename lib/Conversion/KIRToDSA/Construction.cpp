#include "Construction.h"
#include "Intent/Dialect/DSA/IR/Views.h"
#include "Intent/Dialect/DSA/Analysis/PhysicalProgram.h"

namespace intent::kir_to_dsa {

Construction::Construction(ModuleOp original, ModuleOp target, dsa::ConfigurationAttr configuration,
             DictionaryAttr shapes, DictionaryAttr strides)
    : analysis(original), target(target), b(target.getContext()), config(configuration),
      shapeBindings(shapes), strideBindings(strides) {}

LogicalResult Construction::lower(func::FuncOp source) {
  auto publicInterface = buildPublicInterface(source);
  if (failed(publicInterface)) return failure();
  DenseMap<int64_t, int64_t> fixedDimensions;
  llvm::SmallSet<StringRef, 8> boundNames;
  for (BlockArgument argument : source.getArguments()) {
    auto name = getSourceParameter(argument).getName();
    auto binding = shapeBindings ? shapeBindings.getAs<DenseI64ArrayAttr>(name.getValue()) : DenseI64ArrayAttr();
    if (!binding) continue;
    auto view = dyn_cast<ViewType>(argument.getType());
    if (!view) return source.emitError("DSA shape binding must name a view parameter");
    auto tensor = cast<RankedTensorType>(view.getTensor());
    if (binding.size() != tensor.getRank()) return source.emitError("DSA shape binding rank mismatch for ") << name;
    auto identities = cast<TensorShapeAttr>(tensor.getEncoding()).getDimensions();
    for (int64_t axis = 0; axis < tensor.getRank(); ++axis) {
      int64_t extent = binding[axis];
      if (extent < -1) return source.emitError("DSA shape bindings use -1 for an unspecialized axis");
      if (extent < 0) continue;
      if (!tensor.isDynamicDim(axis) && tensor.getDimSize(axis) != extent)
        return source.emitError("DSA shape binding contradicts a source static extent");
      if (identities[axis] > 0) {
        auto [entry, inserted] = fixedDimensions.try_emplace(identities[axis], extent);
        if (!inserted && entry->second != extent) return source.emitError("DSA shape bindings disagree on a logical dimension");
      }
    }
    boundNames.insert(name.getValue());
  }
  if (shapeBindings && boundNames.size() != shapeBindings.size())
    return source.emitError("DSA shape binding names an unknown parameter");
  SmallVector<Type> arguments;
  SmallVector<Attribute> interface(publicInterface->getArguments().getValue());
  SmallVector<Value> sourceArguments;
  llvm::SmallSet<StringRef, 8> boundStrideNames;
  for (BlockArgument argument : source.getArguments()) {
    auto name = getSourceParameter(argument).getName();
    if (isa<ConstexprType>(argument.getType()) && argument.use_empty()) continue;
    if (auto view = dyn_cast<ViewType>(argument.getType())) {
      auto tensor = cast<RankedTensorType>(view.getTensor());
      Type element = tensor.getElementType();
      if (!dsa::isStorageElementType(element))
        return source.emitError("DSA construction does not implement this view storage type");
      auto shape = dyn_cast_or_null<TensorShapeAttr>(tensor.getEncoding());
      if (!shape) return source.emitError("DSA view has no dimension identities");
      SmallVector<int64_t> specialized(tensor.getShape());
      for (int64_t axis = 0; axis < tensor.getRank(); ++axis)
        if (auto found = fixedDimensions.find(shape.getDimensions()[axis]); found != fixedDimensions.end()) specialized[axis] = found->second;
      auto constraints = view.getConstraints();
      if (auto binding = strideBindings ? strideBindings.getAs<DenseI64ArrayAttr>(name.getValue()) : DenseI64ArrayAttr()) {
        if (binding.size() != tensor.getRank()) return source.emitError("DSA stride binding rank mismatch for ") << name;
        SmallVector<Attribute> strides;
        for (int64_t axis = 0; axis < tensor.getRank(); ++axis) {
          if (constraints.getHasStrides())
            if (auto fixed = dyn_cast<IntegerAttr>(constraints.getStrides()[axis]); fixed && fixed.getInt() != binding[axis])
              return source.emitError("DSA stride binding contradicts a source stride constraint for ") << name;
          strides.push_back(b.getI64IntegerAttr(binding[axis]));
        }
        // These facts constrain this compiled variant, not the source view.
        constraints = ViewConstraintsAttr::get(b.getContext(), true, b.getArrayAttr(strides),
            constraints.getAlias(), constraints.getNoalias());
        boundStrideNames.insert(name.getValue());
      }
      SmallVector<int64_t> physicalStrides(tensor.getRank(), ShapedType::kDynamic);
      if (constraints.getHasStrides())
        for (auto [axis, constraint] : llvm::enumerate(constraints.getStrides()))
          if (auto fixed = dyn_cast<IntegerAttr>(constraint))
            physicalStrides[axis] = fixed.getInt();
      arguments.push_back(MemRefType::get(specialized, tensor.getElementType(),
          StridedLayoutAttr::get(b.getContext(), 0, physicalStrides)));
      auto compiledTensor = RankedTensorType::get(specialized, tensor.getElementType(), shape);
      auto compiledView = intent::ViewType::get(b.getContext(), compiledTensor, view.getAccess(), constraints);
      interface[sourceArguments.size()] = PublicParameterAttr::get(b.getContext(), name, compiledView);
    } else if (argument.getType().isF32() || argument.getType().isIndex() || argument.getType().isInteger(64) || argument.getType().isInteger(32) || argument.getType().isInteger(1)) {
      arguments.push_back(argument.getType());
    } else return source.emitError("unsupported DSA scalar ABI type");
    sourceArguments.push_back(argument);
  }
  if (strideBindings && boundStrideNames.size() != strideBindings.size())
    return source.emitError("DSA stride binding must name a view parameter");
  b.setInsertionPointToEnd(target.getBody());
  function = b.create<func::FuncOp>(source.getLoc(), source.getName(), b.getFunctionType(arguments, {}));
  auto compiledInterface = InterfaceAttr::getChecked([&] { return source.emitError(); },
      b.getContext(), b.getArrayAttr(interface));
  if (!compiledInterface) return failure();
  function->setAttr(interfaceAttr, compiledInterface);
  function->setAttr(dsa::entryRequirementsAttr, dsa::EntryRequirementsAttr::get(b.getContext(), true));
  function->setAttr("intent_dsa.configuration", config);
  function.addEntryBlock();
  b.setInsertionPointToStart(&function.front());
  for (auto [old, value] : llvm::zip(sourceArguments, function.getArguments())) {
    values.map(old, value);
    if (isa<ViewType>(old.getType())) {
      auto type = cast<MemRefType>(value.getType());
      auto &sizes = publicExtents[old];
      for (int64_t axis = 0; axis < type.getRank(); ++axis)
        sizes.push_back(b.createOrFold<memref::DimOp>(source.getLoc(), value, axis));
    }
  }
  taskId = b.create<dsa::TaskIdOp>(source.getLoc(), b.getIndexType());
  taskCount = b.create<dsa::TaskCountOp>(source.getLoc(), b.getIndexType());
  SmallVector<ParallelOp> roots;
  source.walk([&](ParallelOp op) { if (!op->getParentOfType<ParallelOp>()) roots.push_back(op); });
  if (!roots.empty()) {
    if (roots.size() != 1 || roots.front()->getBlock() != &source.front())
      return source.emitError("DSA block tasks currently require one outer parallel workset; multiple task phases need a target-wide join realization");
    auto worksets = analysis.logicalWorksets(source);
    if (failed(worksets)) return failure();
    if (worksets->size() == 1 && worksets->front().isExact() &&
        !worksets->front().singleton && independentDomains(worksets->front())) {
      distributedWorkset = worksets->front();
      distributedRoot = roots.front();
    }
    for (Operation &op : source.front().without_terminator()) {
      if (isa<ViewStoreOp>(op)) return op.emitError("DSA outer writes need an explicit task owner outside the parallel workset");
      if (auto load = dyn_cast<ViewLoadOp>(op)) {
        auto relation = analysis.indexRelation(load);
        if (failed(relation)) return failure();
        if (cast<ViewType>(relation->source.getType()).getAccess() != 0)
          return load.emitError("DSA distributed workset cannot duplicate a mutable input snapshot");
      }
    }
    if (failed(lowerBlock(source.front()))) return failure();
  } else {
    if (auto plan = planExecutionSlices(source.front())) {
      if (failed(lowerTiledWorkset(source.front(), *plan, false, {}, true))) return failure();
    } else {
      auto owner = b.create<arith::CmpIOp>(source.getLoc(), arith::CmpIPredicate::eq, taskId, index(source.getLoc(), 0));
      auto single = b.create<scf::IfOp>(source.getLoc(), owner, false);
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(single.thenBlock());
      if (failed(lowerBlock(source.front()))) return failure();
    }
  }
  b.create<func::ReturnOp>(source.getLoc());
  function->setAttr(dsa::fullExtentDimensionsAttr, b.getDenseI64ArrayAttr(fullExtents));
  return success();
}

LogicalResult Construction::loop(Location loc, Value begin, Value end, Value step,
                   const std::function<LogicalResult(Value)> &body) {
  auto op = b.create<scf::ForOp>(loc, begin, end, step);
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(op.getBody());
  return body(op.getInductionVar());
}

LogicalResult Construction::orderedControl(Operation *operation) {
  Location loc = operation->getLoc();
  auto savedDomains = domains;
  auto savedAxes = axisBindings;
  auto savedSlices = valueSlices;
  auto savedValues = values;
  auto savedProducts = products;
  auto savedStreamed = streamedOperations;
  auto restore = [&]() {
    domains = savedDomains;
    axisBindings = savedAxes; valueSlices = savedSlices;
    values = savedValues; products = savedProducts;
    streamedOperations = savedStreamed;
  };
  ValueRange stateSources = operation->getResults();
  if (auto loop = dyn_cast<ForOp>(operation)) stateSources = loop.getInitArgs();
  if (auto loop = dyn_cast<WhileOp>(operation)) stateSources = loop.getInitArgs();
  auto slots = makeSlots(stateSources, loc);
  if (failed(slots)) return failure();
  if (auto conditional = dyn_cast<IfOp>(operation)) {
    auto target = b.create<scf::IfOp>(loc, get(conditional.getCondition()), true);
    for (auto [source, destination] : llvm::zip(operation->getRegions(), target->getRegions())) {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(&destination.front());
      if (failed(lowerResults(source.front(), *slots))) return failure();
      restore();
    }
  } else {
    auto forLoop = dyn_cast<ForOp>(operation);
    auto whileLoop = dyn_cast<WhileOp>(operation);
    auto initial = flatten(forLoop ? forLoop.getInitArgs() : whileLoop.getInitArgs());
    if (initial.size() != slots->size()) return operation->emitError("DSA initial and result state schemas differ");
    auto next = makeSlots(stateSources, loc);
    if (failed(next)) return failure();
    for (auto [value, slot] : llvm::zip(initial, *slots)) if (failed(copyTo(value, slot, loc))) return failure();
    savedValues = values; savedProducts = products;
    auto advance = [&]() -> LogicalResult {
      for (auto [value, slot] : llvm::zip(*next, *slots)) if (failed(copyTo(value, slot, loc))) return failure();
      return success();
    };
    if (forLoop) {
      if (!domains.count(forLoop.getSource())) return operation->emitError("DSA ordered for requires a bound interval");
      Domain domain = domains.lookup(forLoop.getSource());
      if (failed(loop(loc, domain.begin, domain.end, domain.step, [&](Value iv) {
        Block &body = forLoop.getBody().front();
        values.map(forLoop.getInductionVars().front(), iv);
        bindSlots(forLoop.getRegionIterArgs(), *slots, loc);
        if (failed(lowerResults(body, *next))) return failure();
        return advance();
      }))) return failure();
    } else {
      auto target = b.create<scf::WhileOp>(loc, TypeRange{}, ValueRange{});
      target.getBefore().emplaceBlock(); target.getAfter().emplaceBlock();
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(&target.getBefore().front());
        Block &before = whileLoop.getBefore().front();
        bindSlots(whileLoop.getBeforeArguments(), *slots, loc);
        auto predicate = lowerResults(before, *next);
        if (failed(predicate) || !*predicate || failed(advance())) return failure();
        b.create<scf::ConditionOp>(loc, *predicate, ValueRange{});
      }
      restore();
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(&target.getAfter().front());
        Block &after = whileLoop.getAfter().front();
        bindSlots(whileLoop.getAfterArguments(), *slots, loc);
        if (failed(lowerResults(after, *next)) || failed(advance())) return failure();
        b.create<scf::YieldOp>(loc);
      }
    }
  }
  restore();
  bindSlots(operation->getResults(), *slots, loc);
  return success();
}

LogicalResult Construction::requireFullExtent(Value extent, int64_t dimension) {
  // A bounded internal extent is discharged by the physical program. Only an
  // actual public dimension can become a caller-visible tile obligation.
  auto interval = dsa::integerInterval(extent, function);
  if (interval) {
    if (interval->second <= config.getTile()) return success();
    if (interval->first > config.getTile())
      return emitError(extent.getLoc(),
          "DSA full row extent exceeds the selected tile capacity");
  }
  if (auto bound = upperDistance(extent, index(extent.getLoc(), 0));
      bound && *bound <= config.getTile()) return success();

  auto interface = getPublicInterface(function);
  auto require = [&](int64_t identity) {
    if (!llvm::is_contained(fullExtents, identity)) fullExtents.push_back(identity);
    return success();
  };
  for (Attribute attribute : interface.getArguments()) {
    auto view = dyn_cast<ViewType>(cast<PublicParameterAttr>(attribute).getType());
    if (!view) continue;
    auto tensor = publicViewTensor(view);
    for (auto [axis, identity] : llvm::enumerate(publicViewDimensions(view).asArrayRef()))
      if (identity == dimension && tensor.isDynamicDim(axis))
        return require(identity);
  }

  Value current = extent;
  while (true) {
    if (auto cast = current.getDefiningOp<arith::IndexCastOp>(); cast &&
        (cast.getIn().getType().isIndex() || cast.getIn().getType().isInteger(64)) &&
        (cast.getType().isIndex() || cast.getType().isInteger(64))) {
      current = cast.getIn();
      continue;
    }
    if (auto maximum = current.getDefiningOp<arith::MaxSIOp>()) {
      if (matchPattern(maximum.getLhs(), m_Zero())) {
        current = maximum.getRhs();
        continue;
      }
      if (matchPattern(maximum.getRhs(), m_Zero())) {
        current = maximum.getLhs();
        continue;
      }
    }
    break;
  }
  // A view extent is nonnegative, so the domain's max(extent, 0) preserves
  // that public size. Arithmetic combinations of different sizes do not.
  if (auto query = current.getDefiningOp<memref::DimOp>()) {
    auto argument = dyn_cast<BlockArgument>(query.getSource());
    auto axis = query.getConstantIndex();
    if (argument && argument.getOwner() == &function.front() && axis) {
      auto view = getPublicView(interface, argument.getArgNumber());
      if (view) {
        int64_t identity = publicViewDimensions(view)[*axis];
        if (identity > 0) return require(identity);
      }
    }
  }
  return emitError(extent.getLoc(),
      "DSA full row requires a bounded extent or a public dimension; "
      "specialize derived extents with compile-call shape bindings");
}

LogicalResult Construction::lowerBlock(Block &block) {
  bool rowCollective = false, rowsOnly = true;
  SmallVector<Value> tensors;
  for (Operation &operation : block.without_terminator()) {
    rowCollective |= isa<ReduceOp, ScanOp>(operation);
    for (Value result : operation.getResults())
      if (auto type = dyn_cast<RankedTensorType>(result.getType())) {
        rowsOnly &= type.getRank() == 1;
        if (type.getRank() == 1) tensors.push_back(result);
      }
  }
  SmallVector<int64_t> bound;
  auto restore = llvm::make_scope_exit([&] {
    for (int64_t dimension : bound) axisBindings.erase(dimension);
  });
  if (rowCollective && rowsOnly)
    for (Value value : tensors) {
      auto type = cast<RankedTensorType>(value.getType());
      int64_t dimension = cast<TensorShapeAttr>(type.getEncoding()).getDimensions()[0];
      if (dimension <= 0 || selectedAxis(value, 0)) continue;
      Value size = logicalExtent(value, 0, value.getLoc());
      if (!size) return emitError(value.getLoc(), "DSA full row has no logical extent");
      auto bounded = upperDistance(size, index(value.getLoc(), 0));
      if (!bounded && failed(requireFullExtent(size, dimension))) return failure();
      int64_t capacity = bounded ? std::max<int64_t>(1, *bounded) : config.getTile();
      axisBindings[dimension] = {size, index(value.getLoc(), 0), size, capacity};
      bound.push_back(dimension);
    }
  return lowerStructuredBlock(block);
}

LogicalResult Construction::lowerOperations(Block &block) {
  for (Operation &op : block.without_terminator()) {
    if (canDefer(&op)) continue;
    if (failed(lowerOperation(&op))) return failure();
  }
  return success();
}

LogicalResult Construction::lowerOperation(Operation *operation) {
  if (streamedOperations.contains(operation)) return success();
  Location loc = operation->getLoc();
  {
    if (auto store = dyn_cast<ViewStoreOp>(operation); store && valueSlices.empty()) {
      Value data = store.getValue();
      auto type = dyn_cast<RankedTensorType>(data.getType());
      if (type && type.getRank() > 0) {
        auto shape = localShape(data, loc);
        if (failed(shape)) return failure();
        int64_t elements = 1;
        for (const auto &axis : *shape) elements *= axis.capacity;
        if (completeShape(*shape) && elements > config.getLocalBytes() / 16) {
          std::pair<Value, unsigned> root{data, 0};
          if (auto plan = planExecutionSlices(*store->getBlock(), 0, {root})) {
            plan->writes = {store};
            return lowerTiledWorkset(*store->getBlock(), *plan, true, data);
          }
        }
      }
    }
    if (isa<RegionFoldOp, RegionScanOp>(operation)) return lowerRegion(operation);
    if (auto histogramOp = dyn_cast<HistogramOp>(operation)) return histogram(histogramOp);
    if (auto atomicOp = dyn_cast<AtomicRMWOp>(operation)) return atomic(atomicOp);
    if (auto quantizeOp = dyn_cast<QuantizeOp>(operation)) return quantize(quantizeOp);
    if (auto dot = dyn_cast<QuantizedDotOp>(operation)) return quantizedDot(dot);
    if (auto reduce = dyn_cast<ReduceOp>(operation)) return reduceTensor(reduce);
    if (auto scan = dyn_cast<ScanOp>(operation)) return scanTensor(scan);
    if (isa<ViewLoadOp, ViewStoreOp, GatherOp, ScatterUniqueOp>(operation)) return tensorAccess(operation);
    if (operation->getNumResults() == 1 && isa<RankedTensorType>(operation->getResult(0).getType()) &&
        !isa<ExtractOp, IfOp, ForOp, WhileOp, ScanOp>(operation)) return tensorOperation(operation);
  }
  if (isa<IfOp, ForOp, WhileOp>(operation)) return orderedControl(operation);
  if (isa<MakeTupleOp, MakeRecordOp>(operation)) {
    products[operation->getResult(0)] = flatten(operation->getOperands());
    return success();
  }
  if (auto extract = dyn_cast<ExtractOp>(operation)) {
    if (!products.count(extract.getProduct())) {
      auto def = extract.getProduct().getDefiningOp();
      if (!def || failed(materialize(def))) return extract.emitError("DSA record has no bound producer");
    }
    auto field = getProductLeafRange(extract.getProduct().getType(),
        {static_cast<unsigned>(extract.getField())});
    if (failed(field)) return extract.emitError("DSA product field path is invalid");
    auto values = products.lookup(extract.getProduct());
    if (field->offset + field->size > values.size()) return extract.emitError("DSA record has unavailable components");
    bindProduct(extract.getResult(), ArrayRef<Value>(values).slice(field->offset, field->size));
    return success();
  }
  if (isa<ConstantOp, CastOp, BinaryOp, CompareOp, SelectOp, MaskOp, UnaryOp>(operation))
    return lowerScalarOperation(operation);
  if (auto dim = dyn_cast<DimOp>(operation)) {
    Value value = logicalExtent(dim.getSource(), dim.getAxis(), loc);
    if (!value) return dim.emitError("DSA dimension has no runtime binding");
    values.map(dim.getResult(), value); return success();
  }
  if (auto domain = dyn_cast<DomainOp>(operation)) {
    if (domain.getBounds().size() < 2 || domain.getBounds().size() > 3 || domain.getExtentDimensions().size() != 1)
      return domain.emitError("DSA construction requires rank-one interval domains");
    Value begin = asIndex(get(domain.getBounds()[0]), loc), end = asIndex(get(domain.getBounds()[1]), loc);
    Value step = domain.getBounds().size() == 3 ? asIndex(get(domain.getBounds()[2]), loc) : index(loc, 1);
    if (!begin || !end || !step || !matchPattern(step, m_One())) return domain.emitError("DSA construction requires a unit-step interval");
    int64_t identity = cast<IntegerAttr>(domain.getExtentDimensions()[0]).getInt();
    Value count = b.createOrFold<arith::MaxSIOp>(loc, sub(loc, end, begin), index(loc, 0));
    domains[domain.getResult()] = {begin, end, step, count, identity, upperDistance(end, begin)};
    return success();
  }
  if (auto region = dyn_cast<SubregionOp>(operation)) {
    if (!bindDomain(region.getInputs().front()) || region.getExtentDimensions().size() != 1)
      return region.emitError("DSA subregion needs one bound source interval");
    Domain source = domains.lookup(region.getInputs().front());
    unsigned operand = 1;
    Value begin = region.getHasStart() ? asIndex(get(region.getInputs()[operand++]), loc) : source.begin;
    Value end = region.getHasStop() ? asIndex(get(region.getInputs()[operand]), loc) : source.end;
    if (!begin || !end) return region.emitError("DSA subregion bounds are unavailable");
    int64_t dimension = cast<IntegerAttr>(region.getExtentDimensions()[0]).getInt();
    auto capacity = upperDistance(end, begin);
    if (source.capacity) capacity = capacity ? std::min(*capacity, *source.capacity) : source.capacity;
    Value count = sub(loc, end, begin);
    domains[region.getResult()] = {begin, end, source.step, count, dimension, capacity};
    if (capacity) axisBindings[dimension] = {count, index(loc, 0), count, std::max<int64_t>(1, *capacity)};
    return success();
  }
  if (auto end = dyn_cast<RegionEndOp>(operation)) {
    if (!bindDomain(end.getSource())) return end.emitError("DSA region end needs a bound source interval");
    values.map(end.getResult(), domains.lookup(end.getSource()).end); return success();
  }
  if (auto parallel = dyn_cast<ParallelOp>(operation)) {
    if (parallel == distributedRoot) return distributeWorkset(loc);
    if (!domains.count(parallel.getSource()) || parallel.getBody().front().getNumArguments() != 1)
      return parallel.emitError("DSA parallel work requires a single interval");
    auto domain = domains.lookup(parallel.getSource());
    bool distribute = parallelDepth++ == 0;
    Value begin = distribute ? add(loc, domain.begin, mul(loc, taskId, domain.step)) : domain.begin;
    Value step = distribute ? mul(loc, taskCount, domain.step) : domain.step;
    LogicalResult result = loop(loc, begin, domain.end, step, [&](Value i) {
      values.map(parallel.getBody().front().getArgument(0), i);
      return lowerBlock(parallel.getBody().front());
    });
    --parallelDepth; return result;
  }
  if (isa<AssumeInBoundsOp>(operation)) return success();
  return operation->emitError("operation has no DSA construction implementation");
}

} // namespace intent::kir_to_dsa
