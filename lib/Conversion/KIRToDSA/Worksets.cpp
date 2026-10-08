#include "Construction.h"

namespace intent::kir_to_dsa {

bool Construction::singletonAxis(Value value, unsigned axis) {
  return knownExtent(value, axis).constant == 1;
}

bool Construction::equalAxisExtent(Value lhs, unsigned a, Value rhs, unsigned c,
                                    ArrayRef<unsigned> lhsPath,
                                    ArrayRef<unsigned> rhsPath) {
  if (analysis.equalTensorExtents(lhs, a, rhs, c, lhsPath, rhsPath)) return true;
  auto left = knownExtent(lhs, a, lhsPath), right = knownExtent(rhs, c, rhsPath);
  if (left.constant && right.constant) return left.constant == right.constant;
  return left.value && right.value && left.value == right.value;
}

std::optional<WorksetTiling> Construction::planExecutionSlices(Block &block, unsigned selectedAxis,
    ArrayRef<std::pair<Value, unsigned>> sources) {
  WorksetTiling plan;
  plan.axis = selectedAxis;
  DenseSet<Value> written;
  RankedTensorType outputType;
  if (sources.empty()) for (Operation &op : block.without_terminator()) {
    if (op.getNumRegions()) return std::nullopt;
    if (isa<ViewStoreOp, ScatterUniqueOp>(op)) {
      auto relation = analysis.indexRelation(&op);
      Value data = cast<IndexedAccessOpInterface>(&op).getStoredValue();
      auto type = dyn_cast<RankedTensorType>(data.getType());
      // Distinct writable views are disjoint in this target's launch ABI.
      // Reordering separate writes to one view needs an effect-footprint proof.
      if (failed(relation) || !type || selectedAxis >= type.getRank() || !written.insert(relation->source).second) return std::nullopt;
      if (outputType && !equalAxisExtent(plan.extentSource, selectedAxis, data, selectedAxis)) return std::nullopt;
      if (!outputType) {
        outputType = type;
        plan.extentSource = data;
        plan.extentAxis = selectedAxis;
      }
      plan.writes.push_back(&op);
    } else if (auto load = dyn_cast<ViewLoadOp>(op)) {
      if (cast<ViewType>(load.getSource().getType()).getAccess() != 0) return std::nullopt;
    } else if (!isMemoryEffectFree(&op) && !isa<AssumeInBoundsOp>(op)) return std::nullopt;
  }
  if (sources.empty() && plan.writes.empty()) return std::nullopt;
  if (!sources.empty()) {
    plan.extentSource = sources.front().first;
    plan.extentAxis = sources.front().second;
  }
  std::function<bool(Value, AxisRequirements)> require;
  std::function<bool(Value)> scalar;
  std::function<bool(Operation *, Value, const AxisRequirements &)> access;
  DenseSet<Value> scalarSeen;
  scalar = [&](Value value) -> bool {
    if (auto tensor = dyn_cast<RankedTensorType>(value.getType())) return require(value, AxisRequirements(tensor.getRank(), false));
    if (getProductComponents(value.getType())) return false;
    if (!scalarSeen.insert(value).second || values.lookupOrNull(value)) return true;
    Operation *op = value.getDefiningOp();
    if (!op || isa<DimOp>(op)) return true;
    if (op->getNumRegions()) return false;
    if (auto load = dyn_cast<ViewLoadOp>(op)) {
      if (cast<ViewType>(load.getSource().getType()).getAccess() != 0) return false;
    } else if (!isMemoryEffectFree(op) && !isa<DomainOp, SubregionOp>(op)) return false;
    return llvm::all_of(op->getOperands(), scalar);
  };
  access = [&](Operation *op, Value result, const AxisRequirements &requested) -> bool {
    auto relation = analysis.indexRelation(op);
    if (failed(relation)) return false;
    if (relation->resultDimensionIdentities.size() != requested.size()) return false;
    auto local = dyn_cast<RankedTensorType>(relation->source.getType());
    AxisRequirements sourceAxes(local ? local.getRank() : 0, false);
    for (const auto &term : relation->terms) {
      if (term.kind == 0 && local) sourceAxes[*term.sourceAxis] = requested[term.resultAxes.front()];
      if (!term.indexAxes.empty()) {
        Value indices = term.operands.front();
        auto type = cast<RankedTensorType>(indices.getType());
        AxisRequirements indexAxes(type.getRank(), false);
        for (unsigned axis = 0; axis < type.getRank(); ++axis) {
          unsigned mapped = term.indexAxes[axis];
          if (!singletonAxis(indices, axis) && requested[mapped]) {
            if (!equalAxisExtent(indices, axis, result, mapped)) return false;
            indexAxes[axis] = true;
          }
        }
        if (!require(indices, indexAxes)) return false;
      } else for (Value operand : term.operands) if (operand && !scalar(operand)) return false;
    }
    // Nonlocal gathers retain the complete indexed source axis. Full slices
    // project one result axis directly; inserted axes never consume a source axis.
    if (local && !require(relation->source, sourceAxes)) return false;
    auto indexed = cast<IndexedAccessOpInterface>(op);
    for (Value value : {indexed.getAccessValidity(), indexed.getAccessFill()})
      if (value) {
        if (isa<RankedTensorType>(value.getType())) {
          if (!require(value, requested)) return false;
        } else if (!scalar(value)) return false;
      }
    return true;
  };
  require = [&](Value value, AxisRequirements requested) -> bool {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type || requested.size() != type.getRank()) return false;
    for (unsigned axis = 0; axis < requested.size(); ++axis)
      if (singletonAxis(value, axis)) requested[axis] = false;
    auto [entry, inserted] = plan.requirements.try_emplace(value, requested);
    if (!inserted) return entry->second == requested;
    // A pre-existing immutable SSA snapshot can be projected locally.
    if (values.lookupOrNull(value)) return true;
    Operation *op = value.getDefiningOp();
    if (!op || op->getNumRegions()) return false;
    if (isa<ViewLoadOp, GatherOp>(op)) {
      if (auto load = dyn_cast<ViewLoadOp>(op))
        if (cast<ViewType>(load.getSource().getType()).getAccess() != 0) return false;
      return access(op, value, requested);
    }
    if (!isMemoryEffectFree(op)) return false;
    if (isa<IndicesOp, FullOp>(op)) return llvm::all_of(op->getOperands(), scalar);
    if (auto broadcast = dyn_cast<BroadcastOp>(op)) {
      Value input = broadcast->getOperand(0);
      auto source = dyn_cast<RankedTensorType>(input.getType());
      if (!source) return scalar(input);
      if (source.getRank() > type.getRank()) return false;
      AxisRequirements inputAxes(source.getRank(), false);
      unsigned leading = type.getRank() - source.getRank();
      for (unsigned axis = 0; axis < source.getRank(); ++axis)
        if (requested[leading + axis] && !singletonAxis(input, axis)) {
          if (!equalAxisExtent(input, axis, value, leading + axis)) return false;
          inputAxes[axis] = true;
        }
      return require(input, inputAxes);
    }
    if (auto transpose = dyn_cast<TransposeOp>(op)) {
      AxisRequirements inputAxes(type.getRank(), false);
      for (auto [axis, perm] : llvm::enumerate(transpose.getPermutation())) inputAxes[cast<IntegerAttr>(perm).getInt()] = requested[axis];
      return require(transpose.getInput(), inputAxes);
    }
    if (auto matrix = dyn_cast<ContractOp>(op)) {
      auto lhs = cast<RankedTensorType>(matrix.getLhs().getType()), rhs = cast<RankedTensorType>(matrix.getRhs().getType());
      AxisRequirements l(lhs.getRank(), false), r(rhs.getRank(), false);
      auto axes = contractionAxes(matrix);
      if (!axes || axes->results.size() != requested.size()) return false;
      for (auto [axis, result] : llvm::enumerate(axes->lhsResultAxes))
        if (result) l[axis] = requested[*result];
      for (auto [axis, result] : llvm::enumerate(axes->rhsResultAxes))
        if (result) r[axis] = requested[*result];
      return require(matrix.getLhs(), l) && require(matrix.getRhs(), r);
    }
    if (auto matrix = dyn_cast<ScaledContractOp>(op)) {
      if (requested.size() != 2) return false;
      return require(matrix.getLhs(), {requested[0], false, false}) &&
          require(matrix.getLhsScale(), {requested[0], false}) &&
          require(matrix.getRhs(), {false, false, requested[1]}) &&
          require(matrix.getRhsScale(), {requested[1], false});
    }
    if (!isa<UnaryOp, BinaryOp, CompareOp, SelectOp, MaskOp, CastOp>(op)) return false;
    for (Value input : op->getOperands()) {
      if (isa<RankedTensorType>(input.getType())) {
        if (!require(input, requested)) return false;
      } else if (!scalar(input)) return false;
    }
    return true;
  };
  for (Operation *write : plan.writes) {
    Value data = cast<IndexedAccessOpInterface>(write).getStoredValue();
    auto type = cast<RankedTensorType>(data.getType());
    AxisRequirements requested(type.getRank(), false); requested[selectedAxis] = true;
    if (!require(data, requested) || !access(write, data, requested)) return std::nullopt;
  }
  for (auto [source, axis] : sources) {
    auto type = dyn_cast<RankedTensorType>(source.getType());
    if (!type || axis >= type.getRank()) return std::nullopt;
    AxisRequirements requested(type.getRank(), false); requested[axis] = true;
    if (!require(source, requested)) return std::nullopt;
  }
  if (sources.empty()) for (Operation &op : block.without_terminator()) {
    if (canDefer(&op) || isa<ViewStoreOp, ScatterUniqueOp, DimOp, AssumeInBoundsOp>(op)) continue;
    for (Value result : op.getResults()) if (!scalar(result)) return std::nullopt;
  }
  return plan;
}

std::optional<TileDomain> Construction::sliceDomain(const WorksetTiling &plan,
                                     int64_t capacity, bool clampCapacity) {
  Location loc = plan.extentSource.getLoc();
  auto sourceType = dyn_cast<RankedTensorType>(plan.extentSource.getType());
  if (!sourceType || plan.extentAxis >= sourceType.getRank()) return std::nullopt;
  Value size = logicalExtent(plan.extentSource, plan.extentAxis, loc);
  if (!size) return std::nullopt;
  for (const auto &entry : plan.requirements)
    for (auto [axis, requested] : llvm::enumerate(entry.second)) {
      if (!requested) continue;
      Value extent = logicalExtent(entry.first, axis, loc);
      if (!extent || (!sameIndex(size, extent) &&
          !analysis.equalTensorExtents(plan.extentSource, plan.extentAxis,
                                       entry.first, axis))) return std::nullopt;
      if (auto current = selectedAxis(entry.first, axis)) {
        if (!matchPattern(current->begin, m_Zero()) ||
            !sameIndex(current->count, current->extent)) return std::nullopt;
        if (clampCapacity) capacity = std::min(capacity, current->capacity);
      }
    }
  APInt constant;
  if (matchPattern(size, m_ConstantInt(&constant))) {
    if (constant.isNegative()) return std::nullopt;
    if (clampCapacity) capacity = std::min(capacity, std::max<int64_t>(1, constant.getSExtValue()));
  }
  return TileDomain{size, capacity};
}

void Construction::bindExecutionSlice(const WorksetTiling &plan, TileDomain domain,
                        Value begin, Value count) {
  for (const auto &entry : plan.requirements)
    for (auto [axis, requested] : llvm::enumerate(entry.second))
      if (requested)
        valueSlices[entry.first][axis] = {domain.extent, begin, count, domain.capacity};
}

LogicalResult Construction::lowerTiledWorkset(Block &block, const WorksetTiling &plan, bool prepared,
    Value selectedOutput, bool distribute) {
  Location loc = block.getParentOp()->getLoc();
  // Scalar bounds and readonly scalar inputs keep their original evaluation
  // order. Tensor producers remain lazy until their selected slice is needed.
  if (!prepared) for (Operation &op : block.without_terminator()) {
    if (canDefer(&op) || isa<ViewStoreOp, ScatterUniqueOp>(op)) continue;
    if (failed(lowerOperation(&op))) return failure();
  }
  SmallVector<WorksetTiling> slices{plan};
  for (unsigned axis = plan.axis + 1;; ++axis) {
    std::optional<WorksetTiling> next;
    if (selectedOutput) {
      std::pair<Value, unsigned> root{selectedOutput, axis};
      next = planExecutionSlices(block, axis, {root});
    } else next = planExecutionSlices(block, axis);
    if (!next) break;
    next->writes = plan.writes;
    slices.push_back(std::move(*next));
  }
  SmallVector<TileDomain> domains;
  for (const WorksetTiling &slice : slices) {
    auto rank = cast<RankedTensorType>(slice.extentSource.getType()).getRank();
    int64_t capacity = rank == 1 ? config.getTile()
        : slice.axis == 0 ? config.getTileM() : config.getTileN();
    for (const auto &[value, selected] : slice.requirements) {
      auto contract = value.getDefiningOp<ContractOp>();
      if (!contract) continue;
      auto axes = contractionAxes(contract);
      if (!axes) continue;
      for (auto [axis, requested] : llvm::enumerate(selected)) {
        if (!requested) continue;
        const auto &result = axes->results[axis];
        bool batch = result.operand == ContractionOperand::Lhs &&
            llvm::any_of(axes->batch, [&](auto pair) { return pair.lhs == result.axis; });
        capacity = std::min<int64_t>(capacity, batch ? 1 :
            result.operand == ContractionOperand::Lhs ? config.getTileM() : config.getTileN());
      }
    }
    auto domain = sliceDomain(slice, capacity, !distribute);
    if (!domain) return emitError(loc, "DSA workset tiling has no complete common logical axis");
    domains.push_back(*domain);
  }
  coarsenMatrixWorkset(block, slices, domains, distribute);
  auto savedValues = values; auto savedProducts = products; auto savedSlices = valueSlices;
  bool savedDistributed = distributedTiles;
  auto restore = llvm::make_scope_exit([&] {
    values = std::move(savedValues); products = std::move(savedProducts);
    valueSlices = std::move(savedSlices); distributedTiles = savedDistributed;
  });
  auto emitWrites = [&]() -> LogicalResult {
    for (Operation *write : plan.writes) if (failed(lowerOperation(write))) return failure();
    return success();
  };
  auto bind = [&](unsigned axis, Value begin) {
    TileDomain domain = domains[axis];
    Value count = b.create<arith::MinSIOp>(loc, sub(loc, domain.extent, begin), index(loc, domain.capacity));
    bindExecutionSlice(slices[axis], domain, begin, count);
  };
  if (distribute) {
    SmallVector<Value> grids;
    for (TileDomain domain : domains)
      grids.push_back(b.create<arith::CeilDivSIOp>(loc, domain.extent, index(loc, domain.capacity)));
    SmallVector<Value> suffix(grids.size(), index(loc, 1));
    for (unsigned axis = grids.size() - 1; axis > 0; --axis)
      suffix[axis - 1] = mul(loc, grids[axis], suffix[axis]);
    Value tasks = mul(loc, grids.front(), suffix.front());
    distributedTiles = true;
    return loop(loc, taskId, tasks, taskCount, [&](Value task) {
      for (unsigned axis = 0; axis < grids.size(); ++axis) {
        Value ordinal = axis + 1 == grids.size() ? task
            : Value(b.create<arith::DivSIOp>(loc, task, suffix[axis]));
        if (axis) ordinal = b.create<arith::RemSIOp>(loc, ordinal, grids[axis]);
        bind(axis, mul(loc, ordinal, index(loc, domains[axis].capacity)));
      }
      return emitWrites();
    });
  }
  std::function<LogicalResult(unsigned)> nest = [&](unsigned axis) {
    if (axis == domains.size()) return emitWrites();
    TileDomain domain = domains[axis];
    return loop(loc, index(loc, 0), domain.extent, index(loc, domain.capacity), [&](Value begin) {
      bind(axis, begin);
      return nest(axis + 1);
    });
  };
  return nest(0);
}

Value Construction::independentRowControl(Block &block) {
  Value row;
  RankedTensorType rowType;
  bool control = false;
  auto tensorType = [&](Type type) {
    SmallVector<Type> fields;
    appendProductLeafTypes(type, fields);
    for (Type field : fields)
      if (auto tensor = dyn_cast<RankedTensorType>(field)) {
        if (tensor.getRank() != 1) return false;
        auto identities = cast<TensorShapeAttr>(tensor.getEncoding()).getDimensions();
        if (identities[0] <= 0 || (rowType &&
            (tensor.getShape() != rowType.getShape() || tensor.getEncoding() != rowType.getEncoding())))
          return false;
        if (!rowType) rowType = tensor;
      }
    return true;
  };
  auto containsTensor = [](Type type) {
    SmallVector<Type> fields;
    appendProductLeafTypes(type, fields);
    return llvm::any_of(fields, [](Type field) { return isa<RankedTensorType>(field); });
  };
  DenseSet<Value> scalarVisited;
  std::function<bool(Value)> scalar = [&](Value value) {
    if (containsTensor(value.getType())) return false;
    if (!scalarVisited.insert(value).second) return true;
    Operation *producer = value.getDefiningOp();
    if (!producer) return true;
    if (auto load = dyn_cast<ViewLoadOp>(producer)) {
      if (cast<ViewType>(load.getSource().getType()).getAccess() != 0) return false;
    } else if (!isMemoryEffectFree(producer) && !isa<DomainOp, SubregionOp>(producer)) return false;
    return llvm::all_of(producer->getOperands(), scalar);
  };
  WalkResult proof = block.walk([&](Operation *op) {
    for (Type type : op->getOperandTypes())
      if (!tensorType(type)) return WalkResult::interrupt();
    for (Value result : op->getResults()) {
      if (!tensorType(result.getType())) return WalkResult::interrupt();
      if (!row && isa<RankedTensorType>(result.getType())) row = result;
    }
    for (Region &region : op->getRegions())
      for (Block &body : region)
        for (Type type : body.getArgumentTypes())
          if (!tensorType(type)) return WalkResult::interrupt();
    if (op->hasTrait<OpTrait::IsTerminator>()) return WalkResult::advance();
    if (isa<IfOp, ForOp, WhileOp>(op)) {
      control = true;
      for (Value value : op->getOperands())
        if (!containsTensor(value.getType()) && !scalar(value)) return WalkResult::interrupt();
      return WalkResult::advance();
    }
    if (op->getNumRegions() || isa<ReduceOp, ScanOp, GatherOp, ContractOp,
                                  BufferLoadOp, BufferStoreOp>(op))
      return WalkResult::interrupt();
    if (auto indexed = dyn_cast<IndexedAccessOpInterface>(op)) {
      if (!isa<ViewLoadOp, ViewStoreOp>(op)) return WalkResult::interrupt();
      auto relation = analysis.indexRelation(op);
      if (failed(relation)) return WalkResult::interrupt();
      auto view = dyn_cast<ViewType>(relation->source.getType());
      bool write = isa<ViewStoreOp>(op);
      if (!view || (!write && view.getAccess() != 0)) return WalkResult::interrupt();
      Value data = write ? indexed.getStoredValue() : op->getResult(0);
      auto type = dyn_cast<RankedTensorType>(data.getType());
      if (write && !type) return WalkResult::interrupt();
      unsigned projected = 0;
      int64_t dimension = type ? cast<TensorShapeAttr>(type.getEncoding()).getDimensions()[0] : 0;
      for (const auto &term : relation->terms) {
        // The canonical access owns the result-axis occurrence. Row slicing
        // needs one coordinate-producing result axis and scalar dependencies;
        // it does not reinterpret the source domain's defining operation.
        if (!term.resultAxes.empty()) {
          if (!type || !term.sourceAxis || !term.indexAxes.empty() ||
              term.resultAxes.size() != 1 || term.resultAxes.front() != 0 ||
              relation->resultDimensionIdentities.front() != dimension)
            return WalkResult::interrupt();
          ++projected;
        }
        for (Value operand : term.operands)
          if (operand && !scalar(operand)) return WalkResult::interrupt();
      }
      return projected == unsigned(bool(type)) ? WalkResult::advance() : WalkResult::interrupt();
    }
    if (isa<AssumeInBoundsOp>(op)) return WalkResult::advance();
    if (!isMemoryEffectFree(op)) return WalkResult::interrupt();
    bool tensor = llvm::any_of(op->getResultTypes(), containsTensor);
    if (tensor && !isa<UnaryOp, BinaryOp, CompareOp, SelectOp, MaskOp, CastOp,
                       BroadcastOp, FullOp, MakeTupleOp, MakeRecordOp, ExtractOp>(op))
      return WalkResult::interrupt();
    for (Value value : op->getOperands())
      if (!containsTensor(value.getType()) && !scalar(value))
        return WalkResult::interrupt();
    if (!tensor && llvm::any_of(op->getResults(), [&](Value value) { return !scalar(value); }))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (proof.wasInterrupted() || !control || !row) return {};
  if (rowType && rowType.isDynamicDim(0)) {
    auto extent = knownExtent(row, 0);
    if (!extent.constant && !extent.value) return {};
  }
  return row;
}

LogicalResult Construction::lowerStructuredBlock(Block &block) {
  for (Operation &op : block.without_terminator())
    if (auto scan = dyn_cast<ScanOp>(op); scan && canStreamScan(scan, block)) return streamScan(scan, block);
  RegionFoldOp fold;
  for (Operation &op : block.without_terminator()) if (auto region = dyn_cast<RegionFoldOp>(op)) {
    if (fold) return op.emitError("DSA multiple region folds in one workset need an explicit shared projection");
    fold = region;
  }
  if (!fold) {
    if (auto plan = planExecutionSlices(block)) return lowerTiledWorkset(block, *plan);
    if (Value row = independentRowControl(block)) {
      auto type = cast<RankedTensorType>(row.getType());
      int64_t dimension = cast<TensorShapeAttr>(type.getEncoding()).getDimensions()[0];
      if (!axisBindings.count(dimension) && !selectedAxis(row, 0)) {
        Location loc = block.getParentOp()->getLoc();
        Value size = logicalExtent(row, 0, loc);
        if (!size) return emitError(loc, "DSA independent row carry has no logical extent");
        auto savedValues = values; auto savedProducts = products;
        auto savedDomains = domains;
        auto savedAxes = axisBindings; auto savedSlices = valueSlices;
        auto restore = llvm::make_scope_exit([&] {
          values = std::move(savedValues); products = std::move(savedProducts);
          domains = std::move(savedDomains);
          axisBindings = std::move(savedAxes); valueSlices = std::move(savedSlices);
        });
        // An empty row still executes the author's scalar control once. Its
        // zero-count tensor slice has no externally visible members.
        Value end = b.create<arith::MaxSIOp>(loc, size, index(loc, 1));
        return loop(loc, index(loc, 0), end, index(loc, config.getTile()), [&](Value begin) {
          Value count = b.create<arith::MinSIOp>(loc, sub(loc, size, begin), index(loc, config.getTile()));
          axisBindings[dimension] = {size, begin, count, config.getTile()};
          return lowerOperations(block);
        });
      }
    }
    return lowerOperations(block);
  }
  auto partition = partitionQueryAxis(block, fold);
  if (failed(partition)) return failure();
  Value source = *partition;
  if (!source) return lowerOperations(block);
  int64_t query = cast<TensorShapeAttr>(
      logicalTensorType(source).getEncoding()).getDimensions()[0];
  if (axisBindings.count(query)) return lowerOperations(block);
  Value size = logicalExtent(source, 0, fold.getLoc());
  if (!size) return fold.emitError("DSA query extent is unavailable");
  auto savedValues = values; auto savedProducts = products;
  auto savedAxes = axisBindings;
  auto status = loop(fold.getLoc(), index(fold.getLoc(), 0), size, index(fold.getLoc(), config.getTileM()), [&](Value begin) {
    Value count = b.create<arith::MinSIOp>(fold.getLoc(), sub(fold.getLoc(), size, begin), index(fold.getLoc(), config.getTileM()));
    axisBindings[query] = {size, begin, count, config.getTileM()};
    return lowerOperations(block);
  });
  values = std::move(savedValues); products = std::move(savedProducts);
  axisBindings = std::move(savedAxes);
  return status;
}

bool Construction::bindDomain(Value value) {
  if (domains.count(value)) return true;
  Operation *definition = value.getDefiningOp();
  return definition && isa<DomainOp, SubregionOp>(definition) && succeeded(lowerOperation(definition));
}

// Bounds constrain the author's logical interval. They size local storage
// without replacing the runtime extent or the source coordinate origin.
std::optional<int64_t> Construction::upperDistance(Value end, Value begin) {
  if (sameIndex(end, begin)) return 0;
  APInt a, c;
  if (matchPattern(end, m_ConstantInt(&a)) && matchPattern(begin, m_ConstantInt(&c))) {
    APInt distance = a.sext(128) - c.sext(128);
    if (distance.isSignedIntN(64)) return std::max<int64_t>(0, distance.getSExtValue());
  }
  if (auto sum = end.getDefiningOp<arith::AddIOp>()) {
    Value extra;
    if (sameIndex(sum.getLhs(), begin)) extra = sum.getRhs();
    if (sameIndex(sum.getRhs(), begin)) extra = sum.getLhs();
    if (extra && matchPattern(extra, m_ConstantInt(&a)) && !a.isNegative()) return a.getSExtValue();
  }
  auto tighter = [](std::optional<int64_t> lhs, std::optional<int64_t> rhs) {
    return lhs && rhs ? std::optional<int64_t>(std::min(*lhs, *rhs)) : lhs ? lhs : rhs;
  };
  if (auto minimum = end.getDefiningOp<arith::MinSIOp>())
    return tighter(upperDistance(minimum.getLhs(), begin), upperDistance(minimum.getRhs(), begin));
  if (auto maximum = begin.getDefiningOp<arith::MaxSIOp>())
    return tighter(upperDistance(end, maximum.getLhs()), upperDistance(end, maximum.getRhs()));
  return std::nullopt;
}

bool Construction::independentDomains(const LogicalWorksetFact &workset) {
  DenseSet<Value> seen;
  std::function<bool(Value)> varies = [&](Value value) {
    if (llvm::is_contained(workset.coordinates, value)) return true;
    if (!seen.insert(value).second) return false;
    Operation *op = value.getDefiningOp();
    return op && llvm::any_of(op->getOperands(), varies);
  };
  for (Value domain : workset.domains) if (varies(domain)) return false;
  Operation *child = workset.parallel;
  while (Operation *parent = child->getParentOp()) {
    if (!isa<ParallelOp>(parent)) break;
    for (Operation &op : cast<ParallelOp>(parent).getBody().front().without_terminator()) {
      if (&op == child) break;
      if (auto load = dyn_cast<ViewLoadOp>(op)) {
        if (cast<ViewType>(load.getSource().getType()).getAccess() == 0) continue;
      }
      if (!isMemoryEffectFree(&op) && !isa<AssumeInBoundsOp>(op)) return false;
    }
    child = parent;
  }
  return true;
}

LogicalResult Construction::distributeWorkset(Location loc) {
  const auto &workset = *distributedWorkset;
  SmallVector<Domain> intervals;
  SmallVector<Value> counts;
  Value total = index(loc, 1);
  for (Value source : workset.domains) {
    if (!bindDomain(source)) return emitError(loc, "DSA independent task domain is unavailable");
    Domain domain = domains.lookup(source);
    Value count = domain.extent;
    counts.push_back(count); intervals.push_back(domain); total = mul(loc, total, count);
  }
  // A free result axis supplies additional independent task owners. Its
  // region traversal and summary remain local to each selected result tile.
  int64_t query = 0;
  Value querySize, queryTiles;
  auto folds = llvm::to_vector(workset.body->getOps<RegionFoldOp>());
  if (folds.size() == 1) {
    auto axis = partitionQueryAxis(*workset.body, folds.front());
    if (failed(axis)) return failure();
    if (Value source = *axis) {
      int64_t dimension = cast<TensorShapeAttr>(
          logicalTensorType(source).getEncoding()).getDimensions()[0];
      if (!axisBindings.count(dimension)) {
        querySize = logicalExtent(source, 0, loc);
        if (querySize) {
          query = dimension;
          queryTiles = b.createOrFold<arith::CeilDivSIOp>(loc, querySize, index(loc, config.getTileM()));
          total = mul(loc, total, queryTiles);
        }
      }
    }
  }
  if (matchPattern(total, m_Zero())) return success();
  auto savedValues = values; auto savedProducts = products;
  auto savedAxes = axisBindings;
  ++parallelDepth;
  LogicalResult status = loop(loc, taskId, total, taskCount, [&](Value task) {
    Value remaining = task;
    if (query) {
      Value tile = b.create<arith::RemSIOp>(loc, remaining, queryTiles);
      remaining = b.create<arith::DivSIOp>(loc, remaining, queryTiles);
      Value begin = mul(loc, tile, index(loc, config.getTileM()));
      Value count = b.create<arith::MinSIOp>(loc, sub(loc, querySize, begin), index(loc, config.getTileM()));
      axisBindings[query] = {querySize, begin, count, config.getTileM()};
    }
    for (int64_t axis = intervals.size() - 1; axis >= 0; --axis) {
      Value coordinate = axis ? Value(b.create<arith::RemSIOp>(loc, remaining, counts[axis])) : remaining;
      values.map(workset.coordinates[axis], add(loc, intervals[axis].begin, coordinate));
      if (axis) remaining = b.create<arith::DivSIOp>(loc, remaining, counts[axis]);
    }
    return lowerBlock(*workset.body);
  });
  --parallelDepth;
  values = std::move(savedValues); products = std::move(savedProducts);
  axisBindings = std::move(savedAxes);
  return status;
}

} // namespace intent::kir_to_dsa
