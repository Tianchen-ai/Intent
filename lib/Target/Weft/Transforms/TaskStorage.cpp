#include "TaskConversion.h"
#include "ScalarValues.h"
#include "Views.h"
#include "Intent/Analysis/ControlFlow.h"
#include "Weft/Dialect/Kernel/IR/SubviewBounds.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;
namespace wk = ::weft::kernel;

namespace intent::weft_provider {
using namespace task_detail;

Value TaskConversion::localRoot(Value memory) {
  if (controlOwners.contains(memory)) return memory;
  if (auto found = localReferences.find(memory); found != localReferences.end())
    return found->second;
  if (auto cast = memory.getDefiningOp<memref::CastOp>()) return localRoot(cast.getSource());
  if (auto view = memory.getDefiningOp<memref::SubViewOp>()) return localRoot(view.getSource());
  if (memory.getDefiningOp() && isAxisView(memory.getDefiningOp()))
    return localRoot(axisViewSource(memory.getDefiningOp()));
  return storage->uniqueOrigin(memory);
}

bool TaskConversion::sameStorageShape(Value first, Value second) {
  auto left = dyn_cast<MemRefType>(first.getType());
  auto right = dyn_cast<MemRefType>(second.getType());
  if (!left || !right || left.getRank() != right.getRank() ||
      left.getElementType() != right.getElementType()) return false;
  for (int64_t axis = 0; axis < left.getRank(); ++axis) {
    if (!left.isDynamicDim(axis) && !right.isDynamicDim(axis) &&
        left.getDimSize(axis) == right.getDimSize(axis)) continue;
    if (!cpu::haveEqualExtents(ValueBoundsConstraintSet::Variable(first, axis),
                              ValueBoundsConstraintSet::Variable(second, axis))) return false;
  }
  return true;
}

Value TaskConversion::fullLocalOwner(Value memory) {
  Value root = storage->uniqueOrigin(memory);
  if (!root || !isLocal(root)) return {};
  if (!sameStorageShape(memory, root)) return {};
  llvm::SmallDenseSet<Value> visited;
  std::function<bool(Value)> complete = [&](Value value) {
    if (value == root || !visited.insert(value).second) return true;
    if (auto cast = value.getDefiningOp<memref::CastOp>())
      return complete(cast.getSource());
    if (auto view = value.getDefiningOp<memref::SubViewOp>()) {
      if (view.getDroppedDims().any()) return false;
      for (auto [axis, size] : llvm::enumerate(view.getMixedSizes()))
        if (!sameBound(view.getMixedOffsets()[axis], b.getIndexAttr(0)) ||
            !sameBound(view.getMixedStrides()[axis], b.getIndexAttr(1)) ||
            !cpu::haveEqualExtents(ValueBoundsConstraintSet::Variable(size),
                ValueBoundsConstraintSet::Variable(view.getSource(), axis)))
          return false;
      return complete(view.getSource());
    }
    auto incoming = intent::queryControlFlowIncoming(value);
    return incoming.complete && !incoming.edges.empty() &&
        llvm::all_of(incoming.edges, [&](const intent::ControlFlowEdge &edge) {
          return edge.operand && complete(edge.operand->get());
        });
  };
  return complete(memory) ? root : Value{};
}

bool TaskConversion::isLocal(Value memory) {
  Value root = localRoot(memory);
  if (controlOwners.contains(root)) return true;
  Operation *owner = root ? root.getDefiningOp() : nullptr;
  return isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(owner) &&
         currentTask->isProperAncestor(owner);
}

LogicalResult TaskConversion::allocateLocal(Value memory) {
  if (!isLocal(memory))
    return emitError(memory.getLoc(), "Weft local allocation requires a task-private owner");
  auto aliases = storage->aliases(memory);
  if (!aliases.complete || llvm::any_of(aliases.users, [&](Operation *user) {
        return !currentTask->isProperAncestor(user);
      }))
    return emitError(memory.getLoc(), "Weft local allocation has an escaping storage alias");
  auto type = cast<MemRefType>(memory.getType());
  SmallVector<OpFoldResult> sizes;
  SmallVector<Value> extents;
  SmallVector<int64_t> dimensions;
  unsigned dynamic = 0;
  for (auto [axis, dimension] : llvm::enumerate(type.getShape())) {
    if (!ShapedType::isDynamic(dimension) && dimension != 0) {
      sizes.push_back(b.getIndexAttr(dimension));
      extents.push_back(index(memory.getLoc(), dimension));
      dimensions.push_back(dimension);
      continue;
    }
    OpFoldResult sourceExtent = dimension == 0 ? OpFoldResult(b.getIndexAttr(0))
        : OpFoldResult(memory.getDefiningOp()->getOperand(dynamic++));
    Value extent = dimension == 0 ? index(memory.getLoc(), 0)
        : values.lookupOrNull(cast<Value>(sourceExtent));
    if (!extent)
      return emitError(memory.getLoc(), "private allocation extent has no current SSA value");
    auto bound = dimensionExtent(memory, axis);
    if (!bound || *bound == 0) {
      int64_t id = shapeValues.size() + 1;
      extent = b.create<wk::LocalShapeOp>(memory.getLoc(), b.getIndexType(), extent, id);
      shapeValues.push_back(extent);
      localShapeBindings.emplace_back(sourceExtent, id);
      bound = -id;
    }
    if (*bound < 0) extent = shapeValues[-*bound - 1];
    sizes.push_back(sourceExtent);
    extents.push_back(extent);
    dimensions.push_back(*bound);
  }
  Type scalar = type.getElementType();
  if (scalar.isIndex()) scalar = IntegerType::get(b.getContext(), 64, IntegerType::Signed);
  auto value = wk::ValueType::get(b.getContext(), nativeScalarType(scalar),
      array(dimensions), array(memoryAxes(memory)));
  locals[memory] = {b.create<wk::EmptyOp>(memory.getLoc(), value, extents),
                    std::move(sizes)};
  return success();
}

FailureOr<Value> TaskConversion::readNative(Value memory) {
  if (!isa<MemRefType>(memory.getType())) return values.lookup(memory);
  if (!isLocal(memory)) {
    auto supply = readOnlySupplies.find(memory);
    if (supply != readOnlySupplies.end()) return supply->second;
    auto previous = operandReads.find(memory);
    if (previous != operandReads.end()) return previous->second;
    auto region = view(memory);
    if (failed(region)) return failure();
    Type element = cast<MemRefType>(memory.getType()).getElementType();
    if (element.isIndex()) element = IntegerType::get(b.getContext(), 64, IntegerType::Signed);
    Value loaded = b.create<wk::AdmitOp>(memory.getLoc(),
        valueType(element, shape((*region).getType()), axes((*region).getType())), *region);
    // Cache only the admitted snapshot. A positional read and a named-axis
    // contraction may consume this same value in different axis orders.
    operandReads[memory] = loaded;
    if (storage->preserves(currentTask, memory)) readOnlySupplies[memory] = loaded;
    return loaded;
  }
  Value root = localRoot(memory);
  auto found = locals.find(root);
  if (found == locals.end()) {
    emitError(memory.getLoc(), "Weft local value is read before a dominating complete supply");
    return failure();
  }
  if (!isa<wk::ValueType>(found->second.value.getType()) &&
      cast<MemRefType>(root.getType()).getRank() != 0) {
    // A complete uniform definition needs only its scalar until a consumer
    // selects a domain. In particular, a panel read never creates the full
    // allocation's register value just to extract that panel again.
    auto type = resultType(memory);
    if (failed(type)) return failure();
    auto supplied = alignValue(found->second.value, *type, memory.getLoc());
    if (failed(supplied)) return failure();
    auto &mapping = viewAxes[memory];
    mapping.clear();
    for (int64_t axis = 0; axis < cast<MemRefType>(memory.getType()).getRank(); ++axis)
      mapping.push_back(axis);
    return supplied;
  }
  if (memory == root || localReferences.lookup(memory) == root) {
    auto type = cast<MemRefType>(root.getType());
    unsigned dynamic = 0;
    for (auto [axis, extent] : llvm::enumerate(type.getShape())) {
      OpFoldResult full = controlOwners.contains(root) ? found->second.sizes[axis] : ShapedType::isDynamic(extent)
          ? OpFoldResult(root.getDefiningOp()->getOperand(dynamic++))
          : OpFoldResult(b.getIndexAttr(extent));
      if (!sameBound(found->second.sizes[axis], full))
        return emitError(memory.getLoc(), "Weft local root read requires a complete initialized region"), failure();
    }
    auto &mapping = viewAxes[memory];
    mapping.clear();
    for (int64_t axis = 0; axis < type.getRank(); ++axis) mapping.push_back(axis);
    return found->second.value;
  }
  if (auto cast = memory.getDefiningOp<memref::CastOp>()) {
    auto source = readNative(cast.getSource());
    if (failed(source)) return failure();
    viewAxes[memory] = viewAxes.at(cast.getSource());
    return source;
  }
  if (isStaticShapeView(memory.getDefiningOp())) {
    Value source = axisViewSource(memory.getDefiningOp());
    auto supplied = read(source);
    auto target = resultType(memory);
    if (failed(supplied) || failed(target)) return failure();
    auto reshaped = reshapeViewValue(memory, *supplied, *target, axes((*supplied).getType()));
    if (failed(reshaped)) return failure();
    auto &mapping = viewAxes[memory];
    mapping.clear();
    for (unsigned axis = 0; axis < cast<MemRefType>(memory.getType()).getRank(); ++axis)
      mapping.push_back(axis);
    return reshaped;
  }
  auto projection = localProjection(memory, found->second);
  if (failed(projection)) return failure();
  Value selected = b.create<wk::ExtractOp>(memory.getLoc(), projection->type, found->second.value,
                                          projectionIndices(*projection, memory.getLoc()), b.getArrayAttr(projection->selectors));
  return selected;
}

FailureOr<Value> TaskConversion::read(Value memory) {
  auto native = readNative(memory);
  if (failed(native) || !isa<MemRefType>(memory.getType())) return native;
  return projectViewValue(memory, *native, (*native).getType());
}

FailureOr<Value> TaskConversion::readNamedAxes(Value memory) {
  auto native = readNative(memory);
  if (failed(native) || !isa<MemRefType>(memory.getType())) return native;
  auto logical = resultType(memory);
  if (failed(logical)) return failure();
  auto logicalAxes = axes(*logical), logicalShape = shape(*logical);
  auto nativeShape = shape((*native).getType());
  auto mapping = viewAxes.at(memory);
  // Unit dimensions carry only coordinate zero. Reuse an existing unused
  // native unit before inserting one: [1,K] must stay [1,K], because even
  // an element-preserving [1,K] -> [K,1] can change the target partition.
  // This choice is local to the named-value representation; the descriptor
  // projection used by positional reads and writes remains unchanged.
  for (auto [logicalPosition, position] : llvm::enumerate(mapping)) {
    if (position >= 0 || logicalShape[logicalPosition] != 1) continue;
    for (unsigned candidate = 0; candidate < nativeShape.size(); ++candidate)
      if (nativeShape[candidate] == 1 && !llvm::is_contained(mapping, candidate)) {
        mapping[logicalPosition] = candidate;
        break;
      }
  }
  SmallVector<int64_t> dimensions, ids;
  for (unsigned position = 0; position < nativeShape.size(); ++position) {
    auto found = llvm::find(mapping, position);
    if (found == mapping.end()) {
      if (nativeShape[position] != 1)
        return emitError(memory.getLoc(), "named-axis supply lost a non-unit storage dimension"), failure();
      continue;
    }
    unsigned logicalPosition = found - mapping.begin();
    dimensions.push_back(nativeShape[position]);
    ids.push_back(logicalAxes[logicalPosition]);
  }
  for (auto [logicalPosition, position] : llvm::enumerate(mapping))
    if (position < 0) {
      if (logicalShape[logicalPosition] != 1)
        return emitError(memory.getLoc(), "named-axis supply can only insert unit dimensions"), failure();
      dimensions.push_back(1); ids.push_back(logicalAxes[logicalPosition]);
    }
  return reshapeViewValue(memory, *native,
      valueType(element((*native).getType()), dimensions, ids), axes((*native).getType()));
}

SmallVector<Value> TaskConversion::projectionIndices(const LocalProjection &projection, Location loc) {
  SmallVector<Value> result;
  auto dimensions = shape(projection.type), ids = axes(projection.type);
  unsigned retained = 0;
  for (auto [selector, offset] : llvm::zip_equal(projection.selectors, projection.offsets)) {
    StringRef kind = cast<StringAttr>(selector).getValue();
    if (kind == "index") { result.push_back(offset); continue; }
    if (kind == "gather") {
      Type unsignedIndex = IntegerType::get(b.getContext(), 64, IntegerType::Unsigned);
      auto type = cast<wk::ValueType>(valueType(unsignedIndex, {dimensions[retained]}, {ids[retained]}));
      Value lane = b.create<wk::IotaOp>(loc, type, 0, dimensions[retained]);
      Value base = b.create<wk::CastOp>(loc, unsignedIndex, offset);
      result.push_back(b.create<wk::BinaryOp>(loc, type, lane, base, "add"));
    }
    ++retained;
  }
  return result;
}

FailureOr<TaskConversion::LocalProjection> TaskConversion::localProjection(Value memory, const LocalValue &state) {
  Value root = localRoot(memory);
  auto rootType = cast<MemRefType>(root.getType());
  SmallVector<Value> origins(rootType.getRank());
  SmallVector<bool> zeroOrigins(rootType.getRank(), true);
  for (Value &origin : origins) origin = index(memory.getLoc(), 0);
  SmallVector<OpFoldResult> sizes(state.sizes);
  SmallVector<int64_t> kept;
  for (unsigned axis = 0; axis < rootType.getRank(); ++axis) kept.push_back(axis);
  SmallVector<Value> chain;
  for (Value current = memory; current != root;) {
    if (auto found = localReferences.find(current); found != localReferences.end()) current = found->second;
    else if (auto cast = current.getDefiningOp<memref::CastOp>()) current = cast.getSource();
    else if (auto view = current.getDefiningOp<memref::SubViewOp>()) { chain.push_back(current); current = view.getSource(); }
    else if (current.getDefiningOp() && isAxisView(current.getDefiningOp())) {
      auto projection = queryAxisView(current);
      if (failed(projection)) return failure();
      chain.push_back(current); current = projection->source;
    }
    else return emitError(memory.getLoc(), "local window has no composed subview relation"), failure();
  }
  for (Value current : llvm::reverse(chain)) {
    SmallVector<int64_t> next;
    if (isAxisView(current.getDefiningOp())) {
      auto projection = queryAxisView(current);
      if (failed(projection)) return failure();
      for (auto source : projection->sourceAxes) next.push_back(source ? kept[*source] : -1);
      kept = std::move(next);
      continue;
    }
    auto view = current.getDefiningOp<memref::SubViewOp>();
    for (unsigned axis = 0; axis < kept.size(); ++axis) {
      int64_t original = kept[axis];
      if (!sameBound(view.getMixedStrides()[axis], b.getIndexAttr(1)))
        return view.emitError("private windows require unit coordinate steps"), failure();
      OpFoldResult offset = view.getMixedOffsets()[axis];
      if (original < 0) {
        if (!sameBound(offset, b.getIndexAttr(0)) || !sameBound(view.getMixedSizes()[axis], b.getIndexAttr(1)))
          return view.emitError("private inserted unit-axis window must retain its single element"), failure();
        if (!view.getDroppedDims().test(axis)) next.push_back(-1);
        continue;
      }
      zeroOrigins[original] = zeroOrigins[original] && sameBound(offset, b.getIndexAttr(0));
      Value value = isa<Attribute>(offset) ? index(memory.getLoc(), cast<IntegerAttr>(cast<Attribute>(offset)).getInt())
                                          : values.lookup(cast<Value>(offset));
      origins[original] = b.create<wk::BinaryOp>(memory.getLoc(), b.getIndexType(), origins[original], value, "add");
      sizes[original] = view.getMixedSizes()[axis];
      if (!view.getDroppedDims().test(axis)) next.push_back(original);
    }
    kept = std::move(next);
  }
  LocalProjection result;
  SmallVector<int64_t> dimensions, ids;
  auto rootAxes = memoryAxes(root);
  auto sourceAxes = axes(state.value.getType());
  auto sourceShape = shape(state.value.getType());
  SmallVector<int64_t> selectedRoots;
  for (auto [position, axis] : llvm::enumerate(sourceAxes)) {
    auto found = llvm::find(rootAxes, axis);
    if (found == rootAxes.end()) return emitError(memory.getLoc(), "private state axis lost its destination relation"), failure();
    unsigned original = found - rootAxes.begin();
    result.offsets.push_back(origins[original]);
    if (!llvm::is_contained(kept, original)) {
      result.selectors.push_back(b.getStringAttr("index"));
      continue;
    }
    selectedRoots.push_back(original);
    if (sameBound(sizes[original], state.sizes[original]) && zeroOrigins[original]) {
      result.selectors.push_back(b.getStringAttr("all"));
      dimensions.push_back(sourceShape[position]); ids.push_back(axis);
      continue;
    }
    std::optional<int64_t> size;
    if (auto attribute = dyn_cast<Attribute>(sizes[original])) size = cast<IntegerAttr>(attribute).getInt();
    else { llvm::APInt constant; if (matchPattern(cast<Value>(sizes[original]), m_ConstantInt(&constant))) size = constant.getSExtValue(); }
    if (!size || *size <= 0)
      return emitError(memory.getLoc(), "private window extent must be statically bounded by the selected implementation"), failure();
    result.selectors.push_back(b.getStringAttr("gather"));
    dimensions.push_back(*size); ids.push_back(axis);
  }
  result.type = valueType(element(state.value.getType()), dimensions, ids);
  auto &mapping = viewAxes[memory];
  mapping.clear();
  for (int64_t original : kept) {
    auto found = llvm::find(selectedRoots, original);
    mapping.push_back(original < 0 ? -1 : found - selectedRoots.begin());
  }
  return result;
}

FailureOr<Value> TaskConversion::alignValue(Value value, Type target, Location loc) {
  if (element(value.getType()).isIndex() && element(target).isSignedInteger(64))
    value = b.create<wk::CastOp>(loc, valueType(element(target), shape(value.getType()), axes(value.getType())), value);
  if (value.getType() == target) return value;
  auto result = dyn_cast<wk::ValueType>(target);
  if (result && value.getType() == result.getElementType())
    return Value(b.create<wk::NewOp>(loc, target, value, true));
  auto input = dyn_cast<wk::ValueType>(value.getType());
  if (input && result && input.getElementType() == result.getElementType() &&
      llvm::all_of(input.getShape().asArrayRef(), [](int64_t extent) { return extent > 0; }) &&
      llvm::all_of(result.getShape().asArrayRef(), [](int64_t extent) { return extent > 0; })) {
    auto preserves = [](wk::ValueType from, wk::ValueType to) {
      for (auto [axis, extent] : llvm::zip(from.getAxisIds().asArrayRef(), from.getShape().asArrayRef())) {
        auto position = llvm::find(to.getAxisIds().asArrayRef(), axis);
        if (extent != 1 && (position == to.getAxisIds().asArrayRef().end() ||
            to.getShape()[position - to.getAxisIds().asArrayRef().begin()] != extent)) return false;
      }
      return true;
    };
    if (preserves(input, result) && preserves(result, input)) {
      SmallVector<int64_t> order;
      for (int64_t axis : result.getAxisIds().asArrayRef())
        if (llvm::is_contained(input.getAxisIds().asArrayRef(), axis)) order.push_back(axis);
      for (int64_t axis : input.getAxisIds().asArrayRef())
        if (!llvm::is_contained(order, axis)) order.push_back(axis);
      SmallVector<int64_t> inputDomain, resultDomain;
      for (auto [axis, extent] : llvm::zip(input.getAxisIds().asArrayRef(), input.getShape().asArrayRef()))
        if (extent != 1) inputDomain.push_back(axis);
      for (auto [axis, extent] : llvm::zip(result.getAxisIds().asArrayRef(), result.getShape().asArrayRef()))
        if (extent != 1) resultDomain.push_back(axis);
      if (inputDomain == resultDomain) order = llvm::to_vector(input.getAxisIds().asArrayRef());
      return Value(b.create<wk::ReshapeOp>(loc, result, value, array(order)));
    }
  }
  emitError(loc) << "Weft value does not preserve destination axes/extents: " << value.getType() << " -> " << target;
  return failure();
}

LogicalResult TaskConversion::materializeLocal(Value root) {
  auto found = locals.find(root);
  if (found == locals.end() || isa<wk::ValueType>(found->second.value.getType()) ||
      cast<MemRefType>(root.getType()).getRank() == 0) return success();
  auto type = resultType(root);
  if (failed(type)) return failure();
  auto supplied = alignValue(found->second.value, *type, root.getLoc());
  if (failed(supplied)) return failure();
  found->second.value = *supplied;
  return success();
}

Value TaskConversion::fillLocal(Value owner, Value scalar, Location loc,
                const LocalProjection *projection) {
  if (!isa<wk::ValueType>(owner.getType())) return scalar;
  auto dimensions = shape(projection ? projection->type : owner.getType());
  SmallVector<Value> coordinates;
  SmallVector<Attribute> selectors(shape(owner.getType()).size(), b.getStringAttr("index"));
  std::function<Value(unsigned, Value)> fill = [&](unsigned axis, Value current) -> Value {
    if (axis == dimensions.size()) {
      SmallVector<Value> selected;
      if (projection) {
        unsigned coordinate = 0;
        for (auto [selector, offset] : llvm::zip_equal(projection->selectors, projection->offsets)) {
          StringRef kind = cast<StringAttr>(selector).getValue();
          if (kind == "index") selected.push_back(offset);
          else selected.push_back(b.create<wk::BinaryOp>(loc, b.getIndexType(), offset,
                                                         coordinates[coordinate++], "add"));
        }
      } else selected = coordinates;
      for (Value &coordinate : selected)
        if (!coordinate.getType().isIndex())
          coordinate = b.create<wk::CastOp>(loc, b.getIndexType(), coordinate);
      return b.create<wk::UpdateOp>(loc, current.getType(), current, scalar,
                                   selected, b.getArrayAttr(selectors));
    }
    Value end = dimensions[axis] < 0 ? shapeValues[-dimensions[axis] - 1]
                                     : index(loc, dimensions[axis]);
    auto loop = b.create<scf::ForOp>(loc, index(loc, 0), end, index(loc, 1), ValueRange{current});
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(loop.getBody());
      coordinates.push_back(loop.getInductionVar());
      Value updated = fill(axis + 1, loop.getRegionIterArgs().front());
      coordinates.pop_back();
      b.create<scf::YieldOp>(loc, ValueRange{updated});
    }
    return loop.getResult(0);
  };
  return fill(0, owner);
}

LogicalResult TaskConversion::write(Value memory, Value value, SmallVector<OpFoldResult> sizes) {
  if (auto found = localReferences.find(memory); found != localReferences.end())
    return write(found->second, value, std::move(sizes));
  if (!isLocal(memory)) {
    auto region = view(memory);
    if (failed(region)) return failure();
    auto logicalType = resultType(memory);
    if (failed(logicalType)) return failure();
    auto logical = alignValue(value, *logicalType, memory.getLoc());
    if (failed(logical)) return failure();
    auto aligned = projectViewValue(memory, *logical,
        valueType(element((*logical).getType()), shape((*region).getType()), axes((*region).getType())), true);
    if (failed(aligned)) return failure();
    auto dimensions = shape((*region).getType()), ids = axes((*region).getType());
    if (wk::hasDynamicSubviewExtent<wk::SubviewOp, wk::SliceOp, wk::FieldOp>(*region) &&
        llvm::any_of(dimensions, [](int64_t extent) { return extent < 0; })) {
      SmallVector<Attribute> selectors;
      SmallVector<int64_t> fixedShape, fixedAxes;
      for (auto [extent, axis] : llvm::zip(dimensions, ids)) {
        selectors.push_back(b.getStringAttr(extent < 0 ? "index" : "all"));
        if (extent >= 0) { fixedShape.push_back(extent); fixedAxes.push_back(axis); }
      }
      auto selectedType = sliceType(*region, fixedShape, fixedAxes);
      auto selectedValue = valueType(element((*aligned).getType()), fixedShape, fixedAxes);
      SmallVector<Value> indices;
      // Keep the window's runtime extents in loop bounds and issue only
      // fixed-shape projections, as required by the Weft memory consumer.
      std::function<void(unsigned)> project = [&](unsigned axis) {
        if (axis == dimensions.size()) {
          Value selected = b.create<wk::SliceOp>(memory.getLoc(), selectedType, *region,
              indices, b.getArrayAttr(selectors));
          Value payload = b.create<wk::ExtractOp>(memory.getLoc(), selectedValue, *aligned,
              indices, b.getArrayAttr(selectors));
          b.create<wk::CommitOp>(memory.getLoc(), payload, selected);
          return;
        }
        if (dimensions[axis] >= 0) { project(axis + 1); return; }
        Value end = b.create<wk::ExtentOp>(memory.getLoc(), b.getIndexType(), *region, axis);
        auto loop = b.create<scf::ForOp>(memory.getLoc(), index(memory.getLoc(), 0), end, index(memory.getLoc(), 1));
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(loop.getBody());
        indices.push_back(loop.getInductionVar());
        project(axis + 1);
        indices.pop_back();
      };
      project(0);
      return success();
    }
    b.create<wk::CommitOp>(memory.getLoc(), *aligned, *region);
    return success();
  }
  Value root = localRoot(memory);
  if (memory != root) {
    if (auto cast = memory.getDefiningOp<memref::CastOp>()) return write(cast.getSource(), value, sizes);
    if (isStaticShapeView(memory.getDefiningOp())) {
      if (!sizes.empty())
        return emitError(memory.getLoc(), "partial shape-view definition requires an explicit subview");
      Value source = axisViewSource(memory.getDefiningOp());
      auto target = resultType(source), logical = resultType(memory);
      if (failed(target) || failed(logical)) return failure();
      auto supplied = alignValue(value, *logical, memory.getLoc());
      if (failed(supplied)) return failure();
      auto reshaped = reshapeViewValue(source, *supplied, *target, axes((*supplied).getType()));
      return failed(reshaped) ? failure() : write(source, *reshaped);
    }
    auto projection = memory.getDefiningOp<memref::SubViewOp>();
    auto found = locals.find(root);
    if (found != locals.end()) {
      if (failed(materializeLocal(root))) return failure();
      auto projected = localProjection(memory, found->second);
      if (failed(projected)) return failure();
      if (!isa<wk::ValueType>(value.getType())) {
        auto scalar = alignValue(value, element(projected->type), memory.getLoc());
        if (failed(scalar)) return failure();
        found->second.value = fillLocal(found->second.value, *scalar, memory.getLoc(), &*projected);
        return success();
      }
      auto logicalType = resultType(memory);
      if (failed(logicalType)) return failure();
      auto logical = alignValue(value, *logicalType, memory.getLoc());
      if (failed(logical)) return failure();
      auto aligned = projectViewValue(memory, *logical, projected->type, true);
      if (failed(aligned)) return failure();
      found->second.value = b.create<wk::UpdateOp>(memory.getLoc(), found->second.value.getType(),
          found->second.value, *aligned, projectionIndices(*projected, memory.getLoc()), b.getArrayAttr(projected->selectors));
      return success();
    }
    if (isAxisView(memory.getDefiningOp())) {
      auto axisView = queryAxisView(memory);
      if (failed(axisView)) return failure();
      auto sourceType = resultType(axisView->source), logicalType = resultType(memory);
      if (failed(sourceType) || failed(logicalType)) return failure();
      auto &mapping = viewAxes[memory];
      mapping.clear();
      for (auto axis : axisView->sourceAxes) mapping.push_back(axis ? *axis : -1);
      auto logical = alignValue(value, *logicalType, memory.getLoc());
      if (failed(logical)) return failure();
      auto supplied = projectViewValue(memory, *logical, *sourceType, true);
      return failed(supplied) ? failure() : write(axisView->source, *supplied);
    }
    if (!projection || projection.getSource() != root || projection.getDroppedDims().any() ||
        llvm::any_of(projection.getMixedOffsets(), [&](OpFoldResult offset) { return !sameBound(offset, b.getIndexAttr(0)); }))
      return emitError(memory.getLoc(), "Weft private supply must define one zero-based complete active region");
    sizes = projection.getMixedSizes();
  }
  if (sizes.empty()) {
    if (controlOwners.contains(root)) {
      auto found = locals.find(root);
      if (found == locals.end()) return emitError(memory.getLoc(), "private control write has no incoming value");
      sizes = found->second.sizes;
    } else {
    auto type = cast<MemRefType>(root.getType());
    Operation *allocation = root.getDefiningOp();
    unsigned dynamic = 0;
    for (int64_t extent : type.getShape()) {
      if (ShapedType::isDynamic(extent)) sizes.push_back(allocation->getOperand(dynamic++));
      else sizes.push_back(b.getIndexAttr(extent));
    }
    }
  }
  auto type = resultType(memory);
  if (failed(type)) return failure();
  auto current = locals.find(root);
  auto empty = current == locals.end() ? wk::EmptyOp{}
      : current->second.value.getDefiningOp<wk::EmptyOp>();
  bool replacesEmpty = empty && empty->getBlock() == b.getInsertionBlock() &&
      sameStorageShape(memory, root) && current->second.sizes.size() == sizes.size() &&
      llvm::all_of(llvm::zip(current->second.sizes, sizes), [&](auto bounds) {
        return cpu::haveEqualExtents(
            ValueBoundsConstraintSet::Variable(std::get<0>(bounds)),
            ValueBoundsConstraintSet::Variable(std::get<1>(bounds)));
      });
  if (memory == root && !isa<wk::ValueType>(value.getType())) {
    Type scalar = element(*type);
    auto aligned = alignValue(value, scalar, memory.getLoc());
    if (failed(aligned)) return failure();
    if (replacesEmpty)
      current->second = {*aligned, sizes};
    else if (current != locals.end() && isa<wk::ValueType>(current->second.value.getType()))
      current->second.value = fillLocal(current->second.value, *aligned, memory.getLoc());
    else locals[root] = {*aligned, sizes};
    return success();
  }
  auto aligned = alignValue(value, *type, memory.getLoc());
  if (failed(aligned)) return failure();
  if (replacesEmpty) {
    current->second = {*aligned, sizes};
    return success();
  }
  if (current != locals.end() && isa<wk::ValueType>(*type)) {
    if ((isa<wk::ValueType>(current->second.value.getType()) &&
         current->second.value.getType() != *type) || current->second.sizes.size() != sizes.size() ||
        !llvm::all_of(llvm::zip(current->second.sizes, sizes), [&](auto bounds) {
          return sameBound(std::get<0>(bounds), std::get<1>(bounds));
        }))
      return emitError(memory.getLoc(), "complete private write must preserve its initialized owner and extents");
    if (isa<wk::ValueType>(current->second.value.getType()))
      current->second.value = b.create<wk::UpdateOp>(memory.getLoc(), *type,
          current->second.value, *aligned, ValueRange{},
          b.getArrayAttr(SmallVector<Attribute>(shape(*type).size(), b.getStringAttr("all"))));
    else current->second.value = *aligned;
    return success();
  }
  locals[root] = {*aligned, sizes};
  return success();
}

LogicalResult TaskConversion::lower(memref::CopyOp copy) {
  Location loc = copy.getLoc();
  auto value = read(copy.getSource());
  if (failed(value)) return failure();
  auto target = resultType(copy.getTarget());
  if (failed(target)) return failure();
  if (shape((*value).getType()) != shape(*target) || element((*value).getType()) != element(*target))
    return copy.emitError("positional copy requires the same ordered extents and element type");
  if (axes((*value).getType()) != axes(*target))
    *value = b.create<wk::ReshapeOp>(loc, *target, *value, array(axes((*value).getType())));
  if (isLocal(copy.getTarget())) {
    Value root = localRoot(copy.getTarget());
    if (root == copy.getTarget() && locals.count(root)) {
      if (failed(materializeLocal(root))) return failure();
      return write(copy.getTarget(), *value);
    }
    *value = b.create<wk::MaterializeOp>(loc, (*value).getType(), *value);
  }
  return write(copy.getTarget(), *value);
}

LogicalResult TaskConversion::lower(linalg::FillOp fill) {
  Location loc = fill.getLoc();
  Value destination = fill.getOutputs()[0];
  Value initial = values.lookup(fill.getInputs()[0]);
  if (isLocal(destination)) {
    auto type = resultType(destination);
    if (failed(type)) return failure();
    if (initial.getType().isIndex() && element(*type).isSignedInteger(64))
      initial = b.create<wk::CastOp>(loc, element(*type), initial);
    return write(destination, initial);
  }
  auto region = view(destination);
  if (failed(region)) return failure();
  auto dimensions = shape((*region).getType());
  SmallVector<Value> indices;
  std::function<void(unsigned)> fillAxis = [&](unsigned axis) {
    if (axis == dimensions.size()) {
      SmallVector<Attribute> selectors(dimensions.size(), b.getStringAttr("index"));
      Value selected = b.create<wk::SliceOp>(loc,
          sliceType(*region, {}, {}),
          *region, indices, b.getArrayAttr(selectors));
      b.create<wk::CommitOp>(loc, initial, selected);
      return;
    }
    Value end = b.create<wk::ExtentOp>(loc, b.getIndexType(), *region, axis);
    auto loop = b.create<scf::ForOp>(loc, index(loc, 0), end, index(loc, 1));
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(loop.getBody());
    indices.push_back(loop.getInductionVar());
    fillAxis(axis + 1);
    indices.pop_back();
  };
  fillAxis(0);
  return success();
}

LogicalResult TaskConversion::lower(memref::StoreOp store) {
  Location loc = store.getLoc();
  if (isLocal(store.getMemref()) && store.getIndices().empty())
    return write(store.getMemref(), values.lookup(store.getValue()));
  if (isLocal(store.getMemref())) {
    Value root = localRoot(store.getMemref());
    if (failed(materializeLocal(root))) return failure();
    auto current = locals.find(root);
    if (current == locals.end())
      return store.emitError("indexed private write requires a dominating initialized owner");
    auto indices = localIndices(store.getMemref(), store.getIndices(), current->second, values);
    if (failed(indices)) return failure();
    auto scalar = alignValue(values.lookup(store.getValue()),
        element(current->second.value.getType()), loc);
    if (failed(scalar)) return failure();
    current->second.value = b.create<wk::UpdateOp>(loc, current->second.value.getType(),
        current->second.value, *scalar, *indices,
        b.getArrayAttr(SmallVector<Attribute>(indices->size(), b.getStringAttr("index"))));
    return success();
  }
  auto region = view(store.getMemref());
  if (failed(region)) return failure();
  auto indices = projectIndices(store.getMemref(), store.getIndices(), shape((*region).getType()).size(), values);
  Value selected = b.create<wk::SliceOp>(loc,
      sliceType(*region, {}, {}),
      *region, indices, b.getArrayAttr(SmallVector<Attribute>(indices.size(), b.getStringAttr("index"))));
  b.create<wk::CommitOp>(loc, values.lookup(store.getValue()), selected);
  return success();
}

LogicalResult TaskConversion::lower(memref::LoadOp load) {
  auto result = indexedRead(load, values);
  if (failed(result)) return failure();
  values.map(load.getResult(), *result);
  return success();
}

LogicalResult TaskConversion::lower(cpu::AtomicRMWOp atomic) {
  if (atomic.getKind() != AtomicRMWKind::Add ||
      (!atomic.getValue().getType().isInteger(32) &&
       !atomic.getValue().getType().isInteger(64)))
    return atomic.emitError("Weft atomic update requires a supported integer add");
  if (isLocal(atomic.getTarget()))
    return atomic.emitError("Weft atomic update requires an addressable shared view");
  auto region = view(atomic.getTarget());
  if (failed(region)) return failure();
  auto indices = projectIndices(atomic.getTarget(), atomic.getIndices(),
      shape((*region).getType()).size(), values);
  Value selected = b.create<wk::SliceOp>(atomic.getLoc(), sliceType(*region, {}, {}),
      *region, indices,
      b.getArrayAttr(SmallVector<Attribute>(indices.size(), b.getStringAttr("index"))));
  StringRef order;
  switch (atomic.getOrdering()) {
  case AtomicOrdering::Relaxed: order = "relaxed"; break;
  case AtomicOrdering::Acquire: order = "acquire"; break;
  case AtomicOrdering::Release: order = "release"; break;
  case AtomicOrdering::AcquireRelease: order = "acq_rel"; break;
  }
  Value result = b.create<wk::AtomicRMWOp>(atomic.getLoc(),
      nativeScalarType(atomic.getOldValue().getType()),
      values.lookup(atomic.getValue()), selected, "add", order);
  values.map(atomic.getOldValue(), result);
  return success();
}

LogicalResult TaskConversion::lower(bufferization::DeallocOp release) {
  if (!release.getRetained().empty() || release.getNumResults())
    return release.emitError("Weft value release cannot synthesize retained ownership results");
  for (Value memory : release.getMemrefs()) {
    auto origins = storage->origins(memory);
    if (!origins.complete || origins.values.empty())
      return release.emitError("Weft value release requires complete storage origins");
    for (Value origin : origins.values) {
      if (immutableBorrow(origin)) continue;
      Operation *allocation = origin.getDefiningOp();
      if (!isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(allocation) ||
          !currentTask->isProperAncestor(allocation))
        return release.emitError("Weft task cannot release host-owned addressable storage");
      auto aliases = storage->aliases(origin);
      if (!aliases.complete || llvm::any_of(aliases.users, [&](Operation *user) {
            return !currentTask->isProperAncestor(user);
          }))
        return release.emitError("Weft value release has an escaping private storage alias");
    }
  }
  // These private resources are represented by local SSA owners. Their native
  // lifetimes follow the resulting uses; no source heap pointer exists to free.
  return success();
}

FailureOr<Value> TaskConversion::indexedRead(memref::LoadOp load,
                                             const IRMapping &mapping) {
  Location loc = load.getLoc();
  bool local = isLocal(load.getMemref());
  Value base;
  SmallVector<Value> indices;
  if (local) {
    auto current = locals.find(localRoot(load.getMemref()));
    if (current == locals.end())
      return load.emitError("indexed private read has no dominating initialized owner"), failure();
    base = current->second.value;
    if (isa<wk::ValueType>(base.getType())) {
      auto projected = localIndices(load.getMemref(), load.getIndices(), current->second, mapping);
      if (failed(projected)) return failure();
      indices = std::move(*projected);
    } else {
      for (Value coordinate : load.getIndices()) indices.push_back(mapping.lookup(coordinate));
    }
  } else {
    auto region = view(load.getMemref());
    if (failed(region)) return failure();
    base = *region;
    indices = projectIndices(load.getMemref(), load.getIndices(), shape(base.getType()).size(), mapping);
  }
  SmallVector<Attribute> selectors;
  SmallVector<int64_t> dimensions, ids;
  for (Value &coordinate : indices) {
    auto type = dyn_cast<wk::ValueType>(coordinate.getType());
    if (!type) {
      if (!coordinate.getType().isIndex())
        coordinate = b.create<wk::CastOp>(loc, b.getIndexType(), coordinate);
      selectors.push_back(b.getStringAttr("index"));
      continue;
    }
    if (!local)
      return load.emitError("shaped memory indexing requires its explicit scalar address traversal"), failure();
    Type unsignedIndex = IntegerType::get(b.getContext(), 64, IntegerType::Unsigned);
    coordinate = b.create<wk::CastOp>(loc,
        valueType(unsignedIndex, shape(type), axes(type)), coordinate);
    selectors.push_back(b.getStringAttr("gather"));
    for (auto [axis, extent] : llvm::zip(type.getAxisIds().asArrayRef(), type.getShape().asArrayRef())) {
      auto found = llvm::find(ids, axis);
      if (found == ids.end()) { ids.push_back(axis); dimensions.push_back(extent); }
      else if (dimensions[found - ids.begin()] != extent)
        return load.emitError("indexed read coordinates disagree on their logical domain"), failure();
    }
  }
  Type storageElement = local ? element(base.getType())
      : nativeScalarType(cast<MemRefType>(load.getMemref().getType()).getElementType());
  if (storageElement.isIndex()) storageElement = IntegerType::get(b.getContext(), 64, IntegerType::Signed);
  Type loadedType = valueType(storageElement, dimensions, ids);
  Value result;
  if (local && !isa<wk::ValueType>(base.getType())) {
    auto aligned = alignValue(base, loadedType, loc);
    if (failed(aligned)) return failure();
    result = *aligned;
  } else if (local) {
    result = b.create<wk::ExtractOp>(loc, loadedType, base, indices, b.getArrayAttr(selectors));
  } else {
    Value selected = b.create<wk::SliceOp>(loc, sliceType(base, dimensions, ids),
        base, indices, b.getArrayAttr(selectors));
    result = b.create<wk::AdmitOp>(loc, loadedType, selected);
  }
  Type expected = valueType(load.getType(), dimensions, ids);
  if (result.getType() != expected)
    result = b.create<wk::CastOp>(loc, expected, result);
  return result;
}

FailureOr<SmallVector<Value>> TaskConversion::localIndices(
    Value memory, ValueRange indices, const LocalValue &state,
    const IRMapping &mapping) {
  auto projection = localProjection(memory, state);
  if (failed(projection)) return failure();
  auto relative = projectIndices(memory, indices, shape(projection->type).size(), mapping);
  SmallVector<Value> coordinates;
  unsigned retained = 0;
  for (auto [selector, offset] : llvm::zip_equal(projection->selectors, projection->offsets)) {
    if (cast<StringAttr>(selector).getValue() == "index")
      coordinates.push_back(offset);
    else {
      auto coordinate = binary(memory.getLoc(), offset, relative[retained++], "add");
      if (failed(coordinate)) return failure();
      coordinates.push_back(*coordinate);
    }
  }
  return coordinates;
}

} // namespace intent::weft_provider
