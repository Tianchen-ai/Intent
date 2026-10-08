#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "Intent/Dialect/CPU/IR/CPUAttrs.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "Intent/Dialect/CPU/Transforms/Implementation/ImplementationInputs.h"
#include "InputWindows.h"
#include "InputProducers.h"
#include "InputCopies.h"
#include "Intent/Dialect/CPU/Analysis/ExtentRelations.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Intent/Dialect/CPU/Transforms/Structure/LoopBuilders.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>

using namespace mlir;
namespace intent::cpu {

bool hasInvocationInputScope(Operation *operation) {
  return !operation->getParentOfType<scf::ParallelOp>() &&
         !operation->getParentOfType<TasksOp>() &&
         !operation->getParentOfType<TaskDispatchOp>();
}

struct ImplementationInputs::Impl {
  explicit Impl(func::FuncOp function) : function(function) {}

  FailureOr<SmallVector<InputSupply>> prepare(linalg::GenericOp operation,
      ArrayRef<InputRequirement> requirements);
  FailureOr<InputSupply> prepareCaptured(linalg::GenericOp operation,
      memref::LoadOp input, const InputRequirement &requirement);
  FailureOr<SmallVector<InputSupply>> prepareGroup(OpBuilder &builder,
      linalg::GenericOp operation, const ContractionTile &tile,
      ConfigurationAttr configuration, ArrayRef<InputRequirement> requirements);

  std::optional<InputSupply> prepareWindow(Value source, const InputRequirement &requirement,
      linalg::GenericOp operation);
  struct PreparationScope {
    Operation *owner;
    SmallVector<Operation *> dependencies;
    bool guarded = false;
    std::optional<InputWindowBounds> bounds;
  };
  void guardLoop(scf::ForOp loop);
  PreparationScope findPreparationScope(Value source, linalg::GenericOp operation,
      const InputRequirement &requirement,
      const ConsumerWindow *window = nullptr);
  void placePreparation(const PreparationScope &scope);
  InputSupply materialize(Value source, const InputRequirement &requirement,
                          Operation *scope, Operation *readPoint = nullptr);

  struct Prepared {
    Value source;
    InputRequirement requirement;
    memref::AllocOp allocation;
    memref::AllocOp initialized;
    Operation *scope;
  };
  struct PreparedWindow {
    Value source;
    InputRequirement requirement;
    unsigned axis;
    bool transposed;
    scf::ForOp scope;
    memref::AllocOp storage;
    memref::AllocOp initialized;
    Value begin;
    AffineMap lower, upper;
    ValueDimList lowerOperands, upperOperands;
  };
  func::FuncOp function;
  SmallVector<Prepared> prepared;
  SmallVector<PreparedWindow> windows;
  SmallVector<Operation *> guardedLoops;
  InputProducerCopies producerCopies;
};

namespace {

bool sameRepresentation(const InputRequirement &first, const InputRequirement &second) {
  return first.elementType == second.elementType && first.panelAxis == second.panelAxis &&
      first.panelSize == second.panelSize && first.alignment == second.alignment &&
      first.reuse == second.reuse && first.storageScope == second.storageScope;
}

SmallVector<OpFoldResult> sourceExtents(OpBuilder &b, Location loc, Value source) {
  auto type = cast<MemRefType>(source.getType());
  SmallVector<OpFoldResult> sizes;
  for (int64_t axis = 0; axis < type.getRank(); ++axis) {
    if (type.isDynamicDim(axis))
      sizes.push_back(b.createOrFold<memref::DimOp>(loc, source, axis));
    else sizes.push_back(b.getIndexAttr(type.getDimSize(axis)));
  }
  return sizes;
}

memref::AllocOp allocateRepresentation(OpBuilder &b, Location loc,
    ArrayRef<OpFoldResult> capacities, const InputRequirement &requirement) {
  SmallVector<Value> sourceSizes;
  for (OpFoldResult extent : capacities)
    sourceSizes.push_back(getValueOrCreateConstantIndexOp(b, loc, extent));
  Value panel = index(b, loc, requirement.panelSize);
  Value count = b.create<arith::CeilDivSIOp>(loc, sourceSizes[requirement.panelAxis], panel);
  int64_t staticExtent = getConstantIntValue(capacities[requirement.panelAxis])
                            .value_or(ShapedType::kDynamic);
  SmallVector<int64_t> shape{ShapedType::isDynamic(staticExtent) ? ShapedType::kDynamic
      : static_cast<int64_t>(llvm::divideCeil(static_cast<uint64_t>(staticExtent),
                                            static_cast<uint64_t>(requirement.panelSize)))};
  SmallVector<Value> sizes{count};
  for (unsigned axis = 0; axis < capacities.size(); ++axis) {
    if (axis == requirement.panelAxis) continue;
    shape.push_back(getConstantIntValue(capacities[axis]).value_or(ShapedType::kDynamic));
    sizes.push_back(sourceSizes[axis]);
  }
  shape.push_back(requirement.panelSize);
  SmallVector<Value> dynamic;
  for (auto [axis, size] : llvm::enumerate(sizes))
    if (ShapedType::isDynamic(shape[axis])) dynamic.push_back(size);
  auto storage = b.create<memref::AllocOp>(loc, MemRefType::get(shape, requirement.elementType), dynamic);
  storage.setAlignment(requirement.alignment);
  return storage;
}

MemRefType privateRepresentationType(const InputRequirement &requirement,
                                    ArrayRef<int64_t> capacities) {
  int64_t extent = capacities[requirement.panelAxis];
  SmallVector<int64_t> shape{
      extent / requirement.panelSize + (extent % requirement.panelSize != 0)};
  for (auto [axis, capacity] : llvm::enumerate(capacities))
    if (axis != requirement.panelAxis) shape.push_back(capacity);
  shape.push_back(requirement.panelSize);
  return MemRefType::get(shape, requirement.elementType);
}

bool copyTransposedRepresentation(OpBuilder &b, Location loc, Value source,
    Value storage, const InputRequirement &requirement, Operation *point,
    InputProducerCopies &producers, unsigned preparation) {
  auto type = cast<MemRefType>(source.getType());
  if (type.getRank() != 2 || requirement.panelAxis != 1 ||
      requirement.panelSize != 1 || type.getElementType().isInteger(1) ||
      requirement.elementType.isInteger(1))
    return false;
  auto capabilities = point->getParentOfType<ModuleOp>()
      ->getAttrOfType<CapabilitiesAttr>("intent_cpu.capabilities");
  auto bits = [](Type type) -> int64_t {
    return type.isIndex() ? 64 : type.getIntOrFloatBitWidth();
  };
  int64_t width = capabilities.getVectorBits() /
      std::max<int64_t>(8, std::max(bits(type.getElementType()),
                                  bits(requirement.elementType)));
  if (width <= 1) return false;

  Value zero = index(b, loc, 0), one = index(b, loc, 1);
  Value step = index(b, loc, width);
  Value rows = b.create<memref::DimOp>(loc, source, 0);
  Value columns = b.create<memref::DimOp>(loc, source, 1);
  Value fullRows = b.create<arith::SubIOp>(loc, rows,
      b.create<arith::RemSIOp>(loc, rows, step));
  Value columnTiles = b.create<arith::CeilDivSIOp>(loc, columns, step);
  auto slab = [&](Value row, Value column, int64_t count) {
    auto view = [&](Value memory, ArrayRef<OpFoldResult> offsets,
                    ArrayRef<OpFoldResult> sizes) -> Value {
      SmallVector<OpFoldResult> strides(sizes.size(), b.getIndexAttr(1));
      auto result = cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
          ArrayRef<int64_t>{count}, cast<MemRefType>(memory.getType()),
          offsets, sizes, strides));
      return b.create<memref::SubViewOp>(loc, result, memory, offsets, sizes, strides);
    };
    Value input = view(source, {row, column},
        {b.getIndexAttr(count), b.getIndexAttr(1)});
    Value output = view(storage, {column, row, b.getIndexAttr(0)},
        {b.getIndexAttr(1), b.getIndexAttr(count), b.getIndexAttr(1)});
    producers.record(preparation, createInputCopy(b, loc, input, output, 0));
  };
  auto tile = [&](Value ordinal) {
    Value begin = multiply(b, loc, ordinal, step);
    Value remaining = b.create<arith::SubIOp>(loc, columns, begin);
    Value end = add(b, loc, begin, b.create<arith::MinSIOp>(loc, step, remaining));
    // Bound both sides of the layout exchange. Each shaped copy holds only one
    // vector of rows, while the surrounding column traversal reuses the same
    // source rectangle and writes contiguous prepared slices.
    loop(b, loc, zero, fullRows, width, [&](Value row) {
      loop(b, loc, begin, end, 1, [&](Value column) { slab(row, column, width); });
    });
    loop(b, loc, fullRows, rows, 1, [&](Value row) {
      loop(b, loc, begin, end, 1, [&](Value column) { slab(row, column, 1); });
    });
  };
  if (point->getParentOfType<scf::ForOp>() || point->getParentOfType<scf::ParallelOp>() ||
      point->getParentOfType<TasksOp>() || point->getParentOfType<TaskDispatchOp>()) {
    loop(b, loc, zero, columnTiles, 1, tile);
  } else {
    auto parallel = b.create<scf::ParallelOp>(loc, ValueRange{zero},
        ValueRange{columnTiles}, ValueRange{one});
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(parallel.getBody());
    tile(parallel.getInductionVars()[0]);
  }
  return true;
}

void copyRepresentation(OpBuilder &b, Location loc, Value source,
    Value storage, const InputRequirement &requirement, Operation *point,
    InputProducerCopies &producers) {
  unsigned preparation = producers.begin(b.getInsertionBlock(), source);
  if (copyTransposedRepresentation(b, loc, source, storage, requirement, point,
                                   producers, preparation))
    return;
  auto type = cast<MemRefType>(source.getType());
  Value zero = index(b, loc, 0), one = index(b, loc, 1);
  Value panel = index(b, loc, requirement.panelSize);
  SmallVector<OpFoldResult> sourceSizes;
  for (int64_t axis = 0; axis < type.getRank(); ++axis)
    sourceSizes.push_back(getAsOpFoldResult(b.createOrFold<memref::DimOp>(loc, source, axis)));
  Value extent = getValueOrCreateConstantIndexOp(b, loc, sourceSizes[requirement.panelAxis]);
  auto copyPanel = [&](Value ordinal, Value width) {
    Value begin = multiply(b, loc, ordinal, panel);
    SmallVector<OpFoldResult> offsets(type.getRank(), b.getIndexAttr(0));
    SmallVector<OpFoldResult> sizes(sourceSizes);
    offsets[requirement.panelAxis] = getAsOpFoldResult(begin);
    sizes[requirement.panelAxis] = getAsOpFoldResult(width);
    SmallVector<OpFoldResult> strides(type.getRank(), b.getIndexAttr(1));
    Value input = b.create<memref::SubViewOp>(loc, source, offsets, sizes, strides);
    SmallVector<OpFoldResult> packedOffsets(type.getRank() + 1, b.getIndexAttr(0));
    packedOffsets.front() = getAsOpFoldResult(ordinal);
    SmallVector<OpFoldResult> packedSizes{b.getIndexAttr(1)};
    SmallVector<int64_t> shape;
    for (unsigned axis = 0; axis < sizes.size(); ++axis) {
      if (axis == requirement.panelAxis) continue;
      packedSizes.push_back(sizes[axis]);
      shape.push_back(getConstantIntValue(sizes[axis]).value_or(ShapedType::kDynamic));
    }
    packedSizes.push_back(sizes[requirement.panelAxis]);
    shape.push_back(getConstantIntValue(sizes[requirement.panelAxis]).value_or(ShapedType::kDynamic));
    SmallVector<OpFoldResult> packedStrides(packedSizes.size(), b.getIndexAttr(1));
    auto packedType = cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
        shape, cast<MemRefType>(storage.getType()), packedOffsets, packedSizes, packedStrides));
    Value output = b.create<memref::SubViewOp>(loc, packedType, storage,
        packedOffsets, packedSizes, packedStrides);
    producers.record(preparation,
        createInputCopy(b, loc, input, output, requirement.panelAxis));
  };
  Value full = b.create<arith::DivSIOp>(loc, extent, panel);
  Value tail = b.create<arith::RemSIOp>(loc, extent, panel);
  if (point->getParentOfType<scf::ForOp>() || point->getParentOfType<scf::ParallelOp>() ||
      point->getParentOfType<TasksOp>() || point->getParentOfType<TaskDispatchOp>()) {
    loop(b, loc, zero, full, 1, [&](Value ordinal) { copyPanel(ordinal, panel); });
  } else {
    auto parallel = b.create<scf::ParallelOp>(loc, ValueRange{zero}, ValueRange{full}, ValueRange{one});
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(parallel.getBody());
    copyPanel(parallel.getInductionVars()[0], panel);
  }
  auto hasTail = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, tail, zero);
  auto remainder = b.create<scf::IfOp>(loc, hasTail, false);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(remainder.thenBlock());
    copyPanel(full, tail);
  }
}

} // namespace

ImplementationInputs::ImplementationInputs(func::FuncOp function)
    : impl(std::make_unique<Impl>(function)) {}

ImplementationInputs::~ImplementationInputs() = default;

LogicalResult ImplementationInputs::fuseProducerCopies() {
  return impl->producerCopies.fold(impl->function);
}

bool ImplementationInputs::hasReusableScope(linalg::GenericOp operation,
    ArrayRef<InputRequirement> requirements) {
  StorageAnalysis storage(impl->function);
  auto interface = impl->function->getAttrOfType<EntryRequirementsAttr>(entryRequirementsAttr);
  for (const InputRequirement &requirement : requirements) {
    if (requirement.reuse != InputReuse::Consumers) continue;
    Value source = operation.getInputs()[requirement.operand];
    if (interface && interface.getDisjointOutputs()) {
      if (auto window = consumerWindow(source, requirement, operation);
          window && storage.isReadOnly(window->view.getSource())) {
        if (impl->findPreparationScope(window->view.getSource(), operation,
                requirement, &*window).owner != operation)
          return true;
      }
    }
    if (impl->findPreparationScope(source, operation, requirement).owner != operation)
      return true;
  }
  return false;
}

FailureOr<SmallVector<InputSupply>> ImplementationInputs::prepare(
    linalg::GenericOp operation, ArrayRef<InputRequirement> requirements) {
  return impl->prepare(operation, requirements);
}

bool ImplementationInputs::preparedOutsideWorkset(
    Operation *point, ArrayRef<InputSupply> supplies) {
  if (supplies.empty() || !hasInvocationInputScope(point)) return false;
  StorageAnalysis storage(impl->function);
  DominanceInfo dominance(impl->function);
  for (const InputSupply &supply : supplies) {
    if (!dominance.properlyDominates(supply.storage, point)) return false;
    auto origins = storage.origins(supply.storage);
    if (!origins.complete || origins.values.empty()) return false;
    for (Value origin : origins.values) {
      auto allocation = origin.getDefiningOp<memref::AllocOp>();
      if (!allocation || !hasInvocationInputScope(allocation) ||
          !dominance.properlyDominates(origin, point)) return false;
      auto lifetime = storage.lifetime(allocation);
      if (!lifetime || !lifetime->aliases.complete || !lifetime->contains(point))
        return false;
    }
    if (!storage.preserves(point, supply.storage)) return false;
  }
  return true;
}

FailureOr<InputSupply> ImplementationInputs::prepareCaptured(
    linalg::GenericOp operation, memref::LoadOp input,
    const InputRequirement &requirement) {
  return impl->prepareCaptured(operation, input, requirement);
}

Value ImplementationInputs::createPrivateStorage(OpBuilder &builder, Location loc,
    const InputRequirement &requirement, ArrayRef<int64_t> capacities) {
  auto storage = builder.create<memref::AllocaOp>(loc,
      privateRepresentationType(requirement, capacities));
  storage.setAlignment(requirement.alignment);
  return storage;
}

FailureOr<InputSupply> ImplementationInputs::prepareAt(Value window,
    ValueRange begins, const InputRequirement &requirement,
    ArrayRef<int64_t> capacities, Value storage, Operation *scope) {
  if (requirement.storageScope != InputStorageScope::Consumer)
    return scope->emitError("bounded private preparation requires consumer-owned storage"), failure();
  auto type = dyn_cast<MemRefType>(window.getType());
  DominanceInfo dominance(impl->function);
  if (!type || capacities.size() != static_cast<size_t>(type.getRank()) ||
      begins.size() != capacities.size() || requirement.panelAxis >= capacities.size() ||
      requirement.panelSize <= 0 ||
      llvm::any_of(capacities, [](int64_t extent) { return extent <= 0; }) ||
      !dominance.properlyDominates(window, scope) ||
      !dominance.properlyDominates(storage, scope) ||
      llvm::any_of(begins, [&](Value begin) {
        return !dominance.properlyDominates(begin, scope);
      }))
    return scope->emitError("selected input window and origins must dominate its preparation scope"), failure();
  if (storage.getType() != privateRepresentationType(requirement, capacities))
    return scope->emitError("prepared input slot does not match its proved capacity"), failure();
  for (auto [axis, capacity] : llvm::enumerate(capacities)) {
    if (!type.isDynamicDim(axis) && type.getDimSize(axis) > capacity)
      return scope->emitError("prepared input window exceeds its proved capacity"), failure();
  }
  OpBuilder builder(scope);
  copyRepresentation(builder, scope->getLoc(), window, storage, requirement, scope,
                     impl->producerCopies);
  return InputSupply{requirement.operand, requirement.panelAxis, requirement.panelSize,
                     storage, SmallVector<Value>(begins.begin(), begins.end())};
}

FailureOr<SmallVector<InputSupply>> ImplementationInputs::prepareGroup(
    OpBuilder &builder, linalg::GenericOp operation, const ContractionTile &tile,
    ConfigurationAttr configuration, ArrayRef<InputRequirement> requirements) {
  return impl->prepareGroup(builder, operation, tile, configuration, requirements);
}


FailureOr<SmallVector<InputSupply>> ImplementationInputs::Impl::prepare(linalg::GenericOp operation,
    ArrayRef<InputRequirement> requirements) {
  if (auto reason = checkInputRequirements(operation, requirements))
    return operation.emitError(*reason), failure();
  SmallVector<InputSupply> supplies;
  for (auto requirement : requirements) {
    Value source = operation.getInputs()[requirement.operand];
    if (requirement.reuse == InputReuse::Group) continue;
    if (requirement.storageScope == InputStorageScope::Invocation &&
        !hasInvocationInputScope(operation))
      return operation.emitError("invocation input preparation must precede its concurrent work items"), failure();
    if (auto supply = prepareWindow(source, requirement, operation)) supplies.push_back(*supply);
    else {
      auto scope = findPreparationScope(source, operation, requirement);
      placePreparation(scope);
      supplies.push_back(materialize(source, requirement, scope.owner,
                                     scope.guarded ? operation : nullptr));
    }
  }
  return supplies;
}

void ImplementationInputs::Impl::guardLoop(scf::ForOp loop) {
  if (llvm::is_contained(guardedLoops, loop.getOperation())) return;
  OpBuilder b(loop);
  Value nonempty = b.create<arith::CmpIOp>(loop.getLoc(), arith::CmpIPredicate::slt,
      loop.getLowerBound(), loop.getUpperBound());
  auto guard = b.create<scf::IfOp>(loop.getLoc(), nonempty, false);
  loop->moveBefore(guard.thenBlock()->getTerminator());
  guardedLoops.push_back(loop);
}

std::optional<InputSupply> ImplementationInputs::Impl::prepareWindow(Value source,
    const InputRequirement &requirement, linalg::GenericOp operation) {
  auto window = consumerWindow(source, requirement, operation);
  if (!window) return std::nullopt;
  auto view = window->view;
  Value base = view.getSource();
  StorageAnalysis storage(function);
  auto interface = function->getAttrOfType<EntryRequirementsAttr>(entryRequirementsAttr);
  if (!interface || !interface.getDisjointOutputs() || !storage.isReadOnly(base)) return std::nullopt;
  unsigned axis = window->axis;
  unsigned logicalAxis = window->transposed ? 1 - axis : axis;
  OpBuilder b(operation);
  auto sizes = view.getMixedSizes(), offsets = view.getMixedOffsets();

  auto placement = findPreparationScope(base, operation, requirement, &*window);
  auto scope = dyn_cast<scf::ForOp>(placement.owner);
  if (!scope) return std::nullopt;
  const auto &bounds = *placement.bounds;

  PreparedWindow *prepared = nullptr;
  for (auto &candidate : windows)
    if (candidate.source == base && candidate.scope == scope && candidate.axis == axis &&
        candidate.transposed == window->transposed && sameRepresentation(candidate.requirement, requirement) &&
        candidate.lower == bounds.lower && candidate.upper == bounds.upper &&
        candidate.lowerOperands == bounds.lowerOperands && candidate.upperOperands == bounds.upperOperands) {
      prepared = &candidate;
      break;
    }
  Location loc = operation.getLoc();
  if (!prepared) {
    placePreparation(placement);
    b.setInsertionPoint(scope);
    Value lower = materializeInputWindowBound(b, loc, bounds.lower, bounds.lowerOperands);
    Value upper = materializeInputWindowBound(b, loc, bounds.upper, bounds.upperOperands);
    Value extent = b.create<arith::SubIOp>(loc, upper, lower);
    auto capacities = sourceExtents(b, loc, base);
    capacities[axis] = extent;
    if (window->transposed) std::swap(capacities[0], capacities[1]);
    // Capacity is an independent scratch envelope, not an access to a larger
    // source view. Only the original guarded window below is ever read.
    auto storage = allocateRepresentation(b, loc, capacities, requirement);
    auto initialized = b.create<memref::AllocOp>(loc, MemRefType::get({ShapedType::kDynamic}, b.getI1Type()),
        ValueRange{extent});
    Value zero = index(b, loc, 0), clear = b.create<arith::ConstantIntOp>(loc, 0, 1);
    loop(b, loc, zero, extent, 1, [&](Value row) {
      b.create<memref::StoreOp>(loc, clear, initialized, row);
    });
    b.setInsertionPointAfter(scope);
    b.create<memref::DeallocOp>(loc, initialized);
    b.create<memref::DeallocOp>(loc, storage);
    windows.push_back({base, requirement, axis, window->transposed, scope, storage,
        initialized, lower, bounds.lower, bounds.upper, bounds.lowerOperands, bounds.upperOperands});
    prepared = &windows.back();
  }
  b.setInsertionPoint(operation);
  Value zero = index(b, loc, 0), panel = index(b, loc, requirement.panelSize);
  Value begin = getValueOrCreateConstantIndexOp(b, loc, offsets[axis]);
  Value count = getValueOrCreateConstantIndexOp(b, loc, sizes[axis]);
  Value width = getValueOrCreateConstantIndexOp(b, loc, sizes[1 - axis]);
  Value clear = b.create<arith::ConstantIntOp>(loc, 0, 1), ready = b.create<arith::ConstantIntOp>(loc, 1, 1);
  unsigned preparation = producerCopies.begin(b.getInsertionBlock(), view);
  // A marker covers one complete source slice, including only its valid tail.
  // Fill at the original consumer so guards never introduce extra input reads.
  loop(b, loc, zero, count, 1, [&](Value row) {
    Value coordinate = b.create<arith::SubIOp>(loc, add(b, loc, begin, row), prepared->begin);
    Value initialized = b.create<memref::LoadOp>(loc, prepared->initialized, coordinate);
    Value needed = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, initialized, clear);
    auto fill = b.create<scf::IfOp>(loc, needed, false);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(fill.thenBlock());
    auto copy = [&](Value other, Value ordinal, Value storageRow, Value lane) {
      SmallVector<Value> logical(2);
      logical[axis] = row;
      logical[1 - axis] = other;
      auto load = b.create<memref::LoadOp>(loc, view, logical);
      producerCopies.record(preparation, load);
      Value value = load;
      if (value.getType() != requirement.elementType) value = b.create<arith::ExtFOp>(loc, requirement.elementType, value);
      b.create<memref::StoreOp>(loc, value, prepared->storage, ValueRange{ordinal, storageRow, lane});
    };
    if (logicalAxis == requirement.panelAxis) {
      Value ordinal = b.create<arith::DivSIOp>(loc, coordinate, panel);
      Value lane = b.create<arith::RemSIOp>(loc, coordinate, panel);
      loop(b, loc, zero, width, 1, [&](Value other) { copy(other, ordinal, other, lane); });
    } else {
      auto copyPanel = [&](Value ordinal, Value lanes) {
        Value start = multiply(b, loc, ordinal, panel);
        loop(b, loc, zero, lanes, 1, [&](Value lane) { copy(add(b, loc, start, lane), ordinal, coordinate, lane); });
      };
      Value full = b.create<arith::DivSIOp>(loc, width, panel), tail = b.create<arith::RemSIOp>(loc, width, panel);
      loop(b, loc, zero, full, 1, [&](Value ordinal) { copyPanel(ordinal, panel); });
      copyPanel(full, tail);
    }
    b.create<memref::StoreOp>(loc, ready, prepared->initialized, coordinate);
  });
  SmallVector<Value> begins(2, zero);
  begins[logicalAxis] = b.create<arith::SubIOp>(loc, prepared->begin, begin);
  return InputSupply{requirement.operand, requirement.panelAxis, requirement.panelSize, prepared->storage, std::move(begins)};
}

ImplementationInputs::Impl::PreparationScope
ImplementationInputs::Impl::findPreparationScope(Value source, linalg::GenericOp operation,
    const InputRequirement &requirement, const ConsumerWindow *window) {
  PreparationScope scope{operation, {}, false, std::nullopt};
  bool crossesGuard = false;
  StorageAnalysis storage(function);
  DominanceInfo dominance(function);
  for (Operation *parent = operation->getParentOp(); parent != function; parent = parent->getParentOp()) {
    // Lazy storage may surround a guard; its input reads remain at the original
    // guarded use. Eager preparation must retain that execution condition.
    if (isa<scf::IfOp>(parent)) {
      crossesGuard = true;
      continue;
    }
    auto loop = dyn_cast<scf::ForOp>(parent);
    if (!loop || loop.getNumResults()) break;
    auto step = getConstantIntValue(loop.getStep());
    if (!step || *step <= 0 || !storage.preserves(loop, source)) break;
    if (window) {
      if (!dominance.properlyDominates(source, loop) ||
          !loop->isAncestor(window->view) ||
          !hasIndependentWindowCoordinates(window->view, loop)) continue;
      auto candidate = boundInputWindow(*window, loop, requirement);
      if (!candidate) continue;
      scope.bounds = std::move(*candidate);
    } else {
      // A descriptor created only under a guard need not describe a valid
      // allocation on the other path, even when its construction is pure.
      if (crossesGuard && !dominance.properlyDominates(source, loop)) break;
      auto candidate = inputWindowDependencies(ValueRange{source}, loop);
      if (!candidate) break;
      scope.dependencies = std::move(candidate->operations);
    }
    scope.owner = loop;
    scope.guarded = crossesGuard;
    // A guard on the outer loop does not prove an inner loop executes. Only
    // cross another loop when every already-crossed traversal is nonempty.
    if (!window && !crossesGuard && !ValueBoundsConstraintSet::compare(
            ValueBoundsConstraintSet::Variable(loop.getLowerBound()),
            ValueBoundsConstraintSet::LT,
            ValueBoundsConstraintSet::Variable(loop.getUpperBound()))) break;
  }
  return scope;
}

void ImplementationInputs::Impl::placePreparation(const PreparationScope &scope) {
  if (auto loop = dyn_cast<scf::ForOp>(scope.owner)) guardLoop(loop);
  for (Operation *dependency : scope.dependencies) dependency->moveBefore(scope.owner);
}

FailureOr<InputSupply> ImplementationInputs::Impl::prepareCaptured(linalg::GenericOp operation,
    memref::LoadOp input, const InputRequirement &requirement) {
  if (input->getBlock() != &operation.getRegion().front() ||
      requirement.reuse != InputReuse::Consumers)
    return operation.emitError("captured input supply requires its consumer load and enclosing scope"), failure();
  if (requirement.storageScope == InputStorageScope::Invocation &&
      !hasInvocationInputScope(operation))
    return operation.emitError("invocation input preparation must precede its concurrent work items"), failure();
  Value source = input.getMemref();
  if (auto reason = checkInputRequirement(source, input.getResult(), requirement))
    return operation.emitError(*reason), failure();
  auto scope = findPreparationScope(source, operation, requirement);
  placePreparation(scope);
  return materialize(source, requirement, scope.owner, scope.guarded ? operation : nullptr);
}

InputSupply ImplementationInputs::Impl::materialize(Value source, const InputRequirement &requirement,
                                                    Operation *scope, Operation *readPoint) {
  StorageAnalysis analysis(function);
  DominanceInfo dominance(function);
  auto type = cast<MemRefType>(source.getType());
  memref::AllocOp storage;
  memref::AllocOp initialized;
  for (auto &previous : prepared) {
    Operation *consumer = previous.allocation->getBlock()->findAncestorOpInBlock(*scope);
    if (previous.source != source || previous.requirement.panelAxis != requirement.panelAxis ||
        previous.requirement.elementType != requirement.elementType ||
        previous.requirement.panelSize != requirement.panelSize ||
        previous.requirement.storageScope != requirement.storageScope ||
        previous.requirement.alignment < requirement.alignment ||
        (previous.initialized && (!readPoint || previous.scope != scope)) ||
        !consumer || !dominance.dominates(previous.allocation.getOperation(), scope) ||
        !previous.allocation->isBeforeInBlock(consumer) ||
        (previous.allocation->getBlock() != scope->getBlock() && !analysis.isReadOnly(source)) ||
        !analysis.readStable(source, previous.allocation, consumer)) continue;
    auto lifetime = analysis.lifetime(previous.allocation);
    if (!lifetime || !lifetime->aliases.complete) continue;
    storage = previous.allocation;
    initialized = previous.initialized;
    if (lifetime->end->isBeforeInBlock(consumer)) lifetime->end->moveAfter(consumer);
    break;
  }
  if (!storage) {
    OpBuilder b(scope);
    Location loc = scope->getLoc();
    storage = allocateRepresentation(b, loc, sourceExtents(b, loc, source), requirement);
    if (readPoint) {
      initialized = b.create<memref::AllocOp>(loc, MemRefType::get({}, b.getI1Type()));
      b.create<memref::StoreOp>(loc, b.create<arith::ConstantIntOp>(loc, 0, 1),
                                initialized, ValueRange{});
    } else copyRepresentation(b, loc, source, storage, requirement, scope, producerCopies);
    b.setInsertionPointAfter(scope);
    if (initialized) b.create<memref::DeallocOp>(loc, initialized);
    b.create<memref::DeallocOp>(loc, storage);
    prepared.push_back({source, requirement, storage, initialized, scope});
  }
  if (initialized) {
    // A full-descriptor snapshot can share the enclosing loop's lifetime while
    // preserving the exact execution guard of its first actual consumer.
    OpBuilder b(readPoint);
    Location loc = readPoint->getLoc();
    Value ready = b.create<memref::LoadOp>(loc, initialized, ValueRange{});
    Value needed = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
        ready, b.create<arith::ConstantIntOp>(loc, 0, 1));
    auto fill = b.create<scf::IfOp>(loc, needed, false);
    b.setInsertionPointToStart(fill.thenBlock());
    copyRepresentation(b, loc, source, storage, requirement, readPoint, producerCopies);
    b.create<memref::StoreOp>(loc, b.create<arith::ConstantIntOp>(loc, 1, 1),
                              initialized, ValueRange{});
  }
  OpBuilder b(scope);
  SmallVector<Value> begins(type.getRank(), index(b, scope->getLoc(), 0));
  return {requirement.operand, requirement.panelAxis, requirement.panelSize, storage, std::move(begins)};
}

FailureOr<SmallVector<InputSupply>> ImplementationInputs::Impl::prepareGroup(OpBuilder &b,
    linalg::GenericOp operation, const ContractionTile &tile, ConfigurationAttr configuration,
    ArrayRef<InputRequirement> requirements) {
  SmallVector<InputSupply> supplies;
  for (auto requirement : requirements) {
    if (requirement.reuse != InputReuse::Group) continue;
    auto loc = operation.getLoc();
    Value source = operation.getInputs()[requirement.operand];
    auto sourceType = cast<MemRefType>(source.getType());
    auto map = operation.getIndexingMapsArray()[requirement.operand];
    if (sourceType.getRank() != 2 || !map.isProjectedPermutation())
      return operation.emitError("grouped input supply requires a projected matrix input"), failure();
    SmallVector<Value> starts{tile.mBegin, tile.nBegin, tile.kBegin};
    SmallVector<Value> counts{tile.mCount, tile.nCount, tile.depth};
    SmallVector<int64_t> capacities{configuration.getTileM(), configuration.getTileN(), configuration.getTileK()};
    SmallVector<OpFoldResult> offsets, sizes;
    SmallVector<Value> begins;
    unsigned otherAxis = 1 - requirement.panelAxis;
    for (AffineExpr expression : map.getResults()) {
      auto axis = cast<AffineDimExpr>(expression).getPosition();
      offsets.push_back(starts[axis]);
      sizes.push_back(counts[axis]);
      begins.push_back(starts[axis]);
    }
    auto other = cast<AffineDimExpr>(map.getResult(otherAxis)).getPosition();
    int64_t capacity = capacities[other];
    // A valid tile window cannot exceed its current source dimension. Keep the
    // panel pitch/alignment unchanged; only remove unused rows of its storage.
    // Empty dimensions retain the existing allocation form, which remains under
    // the zero-work loop guard; this optimization does not introduce zero slots.
    if (auto bound = constantDimensionUpperBound(source, otherAxis); bound && *bound > 0)
      capacity = std::min(capacity, *bound);
    auto storage = b.create<memref::AllocaOp>(loc,
        MemRefType::get({1, capacity, requirement.panelSize}, requirement.elementType));
    storage.setAlignment(requirement.alignment);
    SmallVector<OpFoldResult> strides(2, b.getIndexAttr(1));
    Value window = b.create<memref::SubViewOp>(loc, source, offsets, sizes, strides);
    SmallVector<OpFoldResult> packedOffsets(3, b.getIndexAttr(0));
    SmallVector<OpFoldResult> packedSizes{b.getIndexAttr(1), sizes[otherAxis], sizes[requirement.panelAxis]};
    SmallVector<OpFoldResult> packedStrides(3, b.getIndexAttr(1));
    auto resultType = cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
        {ShapedType::kDynamic, ShapedType::kDynamic}, storage.getType(), packedOffsets, packedSizes, packedStrides));
    Value destination = b.create<memref::SubViewOp>(loc, resultType, storage, packedOffsets, packedSizes, packedStrides);
    unsigned preparation = producerCopies.begin(b.getInsertionBlock(), window);
    producerCopies.record(preparation,
        createInputCopy(b, loc, window, destination, requirement.panelAxis));
    supplies.push_back({requirement.operand, requirement.panelAxis, requirement.panelSize, storage, std::move(begins)});
  }
  return supplies;
}

}
