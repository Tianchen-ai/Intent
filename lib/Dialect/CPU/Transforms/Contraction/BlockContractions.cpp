#include "Intent/Dialect/CPU/Transforms/Configuration/Configuration.h"
#include "Intent/Dialect/CPU/Transforms/Contraction/Contraction.h"
#include "Intent/Dialect/CPU/Transforms/Storage/Storage.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/CPU/IR/CPUAttrs.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "Intent/Dialect/CPU/Transforms/Implementation/Implementation.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Intent/Dialect/CPU/Transforms/Implementation/ImplementationInputs.h"
#include "Contractions.h"
#include "ContractionEpilogues.h"
#include "Intent/Dialect/CPU/Transforms/Structure/LoopBuilders.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include <limits>

using namespace mlir;

namespace intent::cpu {
namespace {

Value subview(OpBuilder &b, Location loc, Value source,
              ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes) {
  auto type = cast<MemRefType>(source.getType());
  SmallVector<OpFoldResult> strides(type.getRank(), b.getIndexAttr(1));
  return b.create<memref::SubViewOp>(loc, source, offsets, sizes, strides);
}

struct InputCohort {
  InputRequirement requirement;
  unsigned freeAxis;
  SmallVector<int64_t, 2> capacities;
  int64_t preparedBytes;
};

std::optional<InputCohort> inputCohort(linalg::GenericOp operation,
    ArrayRef<InputRequirement> requirements, const Configuration &configuration,
    CapabilitiesAttr capabilities, StorageAnalysis &storage) {
  // Two supplied operands have different reuse axes. Keep their actual shared
  // representations rather than repeatedly preparing one to share the other.
  if (requirements.size() != 1 || requirements.front().reuse != InputReuse::Consumers ||
      requirements.front().storageScope == InputStorageScope::Invocation)
    return std::nullopt;
  auto requirement = requirements.front();
  Value source = operation.getInputs()[requirement.operand];
  auto type = cast<MemRefType>(source.getType());
  auto map = operation.getIndexingMapsArray()[requirement.operand];
  if (type.getRank() != 2 || !map.isProjectedPermutation() ||
      map.getNumResults() != 2 || requirement.panelAxis >= 2)
    return std::nullopt;
  SmallVector<int64_t, 2> capacities;
  std::optional<unsigned> freeAxis;
  bool reduction = false;
  for (AffineExpr expression : map.getResults()) {
    auto dimension = dyn_cast<AffineDimExpr>(expression);
    if (!dimension || dimension.getPosition() > 2) return std::nullopt;
    unsigned axis = dimension.getPosition();
    if (axis == 2) reduction = true;
    else freeAxis = axis;
    capacities.push_back(axis == 0 ? configuration.tileM
                         : axis == 1 ? configuration.tileN : configuration.tileK);
  }
  if (!freeAxis || !reduction ||
      llvm::any_of(operation.getInputs(), [&](Value input) {
        return !storage.disjoint(input, operation.getOutputs()[0]);
      })) return std::nullopt;
  // This is the fixed slot used by prepareAt, including panel padding and
  // alignment. A bounded accumulator cohort accounts for it at the same time.
  int64_t bits = requirement.elementType.isIndex()
      ? 64 : requirement.elementType.getIntOrFloatBitWidth();
  int64_t bytes = (bits + 7) / 8;
  if (bytes <= 0 || requirement.panelSize <= 0) return std::nullopt;
  int64_t elements = capabilities.getPrivateBytes() / bytes;
  int64_t extent = capacities[requirement.panelAxis];
  int64_t panels = extent / requirement.panelSize + (extent % requirement.panelSize != 0);
  if (panels > elements / requirement.panelSize) return std::nullopt;
  int64_t packed = panels * requirement.panelSize;
  if (packed <= 0 || capacities[1 - requirement.panelAxis] > elements / packed)
    return std::nullopt;
  int64_t preparedBytes = packed * capacities[1 - requirement.panelAxis] * bytes;
  if (requirement.alignment <= 0) return std::nullopt;
  int64_t padding = preparedBytes % requirement.alignment
      ? requirement.alignment - preparedBytes % requirement.alignment : 0;
  if (padding > capabilities.getPrivateBytes() - preparedBytes) return std::nullopt;
  return InputCohort{requirement, *freeAxis, std::move(capacities), preparedBytes + padding};
}

int64_t accumulatorCohortCapacity(linalg::GenericOp operation,
    const Configuration &configuration, const InputCohort &cohort,
    CapabilitiesAttr capabilities) {
  Value output = operation.getOutputs()[0];
  auto type = cast<MemRefType>(output.getType());
  int64_t bytes = (type.getElementTypeBitWidth() + 7) / 8;
  int64_t available = capabilities.getPrivateBytes() - cohort.preparedBytes;
  int64_t rows = configuration.tileM, columns = configuration.tileN;
  if (available <= 0 || bytes <= 0 || rows <= 0 || columns <= 0 ||
      rows > available / bytes / columns) return 0;
  int64_t size = rows * columns * bytes;
  auto attr = output.getDefiningOp()->getAttrOfType<IntegerAttr>("alignment");
  int64_t alignment = attr ? attr.getInt() : 1;
  if (alignment <= 0) return 0;
  int64_t padding = size % alignment ? alignment - size % alignment : 0;
  if (padding > available - size) return 0;
  int64_t capacity = available / (size + padding);
  // A single output has no consumer reuse. Keep the existing supply order and
  // backing unless at least two complete accumulator slots fit with the input.
  return capacity >= 2 ? capacity : 0;
}

std::optional<std::array<int64_t, 2>> compactAccumulatorShape(
    linalg::GenericOp operation, const Configuration &configuration,
    ArrayRef<InputRequirement> requirements, CapabilitiesAttr capabilities) {
  // Only preparations allocated in this workset belong in its private budget.
  // Callers omit external supplies only after proving their actual owners.
  if (llvm::any_of(requirements, [](const InputRequirement &requirement) {
        return requirement.reuse != InputReuse::Group ||
               requirement.storageScope != InputStorageScope::Consumer;
      })) return std::nullopt;
  std::array<int64_t, 2> shape{configuration.tileM, configuration.tileN};
  int64_t remaining = capabilities.getPrivateBytes();
  auto reserve = [&](int64_t rows, int64_t columns, Type element,
                     int64_t alignment) {
    if (!element.isIntOrIndexOrFloat() || rows <= 0 || columns <= 0 ||
        alignment <= 0) return false;
    int64_t width = element.isIndex() ? 8 : (element.getIntOrFloatBitWidth() + 7) / 8;
    if (width <= 0 || rows > remaining / width / columns) return false;
    int64_t size = rows * columns * width;
    int64_t padding = size % alignment ? alignment - size % alignment : 0;
    if (padding > remaining - size) return false;
    remaining -= size + padding;
    return true;
  };
  Value output = operation.getOutputs()[0];
  auto alignment = output.getDefiningOp()->getAttrOfType<IntegerAttr>("alignment");
  if (!reserve(shape[0], shape[1], cast<MemRefType>(output.getType()).getElementType(),
               alignment ? alignment.getInt() : 1)) return std::nullopt;
  SmallVector<int64_t> capacities{
      configuration.tileM, configuration.tileN, configuration.tileK};
  // Account for every simultaneously prepared Group input, using the same
  // storage extent as prepareGroup. This bounds these new working buffers; it
  // does not claim to model the target's complete register or cache footprint.
  for (const InputRequirement &requirement : requirements) {
    Value source = operation.getInputs()[requirement.operand];
    auto map = operation.getIndexingMapsArray()[requirement.operand];
    unsigned otherAxis = 1 - requirement.panelAxis;
    unsigned other = cast<AffineDimExpr>(map.getResult(otherAxis)).getPosition();
    int64_t capacity = capacities[other];
    if (auto bound = constantDimensionUpperBound(source, otherAxis); bound && *bound > 0)
      capacity = std::min(capacity, *bound);
    if (!reserve(capacity, requirement.panelSize, requirement.elementType,
                 requirement.alignment)) return std::nullopt;
  }
  return shape;
}

bool canBlockAccumulator(linalg::GenericOp operation,
                         const Configuration &configuration) {
  Value output = operation.getOutputs()[0];
  // Addressable storage retains its original lexical owner. A task-local
  // allocation may instead be a provider's whole numeric value; do not enlarge
  // that value's live representation by changing its allocation layout.
  if (!isa_and_nonnull<memref::AllocOp>(output.getDefiningOp()) ||
      !hasInvocationInputScope(operation)) return false;
  auto type = cast<MemRefType>(output.getType());
  int64_t bytes = (type.getElementTypeBitWidth() + 7) / 8;
  int64_t limit = std::numeric_limits<int64_t>::max();
  if (bytes <= 0 || configuration.tileM <= 0 || configuration.tileN <= 0 ||
      configuration.tileM > limit / bytes / configuration.tileN) return false;
  if (type.hasStaticShape()) {
    int64_t elements = 1;
    const int64_t tiles[] = {configuration.tileM, configuration.tileN};
    for (auto [extent, tile] : llvm::zip(type.getShape(), tiles)) {
      int64_t count = extent / tile + (extent % tile != 0);
      int64_t capacity = std::min(extent, tile);
      if (capacity && count > limit / bytes / capacity) return false;
      int64_t padded = count * capacity;
      if (padded && elements > limit / bytes / padded) return false;
      elements *= padded;
    }
  }
  return true;
}

Value createBlockedAccumulator(Value original,
                               const Configuration &configuration) {
  auto allocation = original.getDefiningOp<memref::AllocOp>();
  auto type = allocation.getType();
  OpBuilder b(allocation);
  Location loc = allocation.getLoc();
  Value zero = index(b, loc, 0), one = index(b, loc, 1);
  SmallVector<int64_t, 4> shape(4, ShapedType::kDynamic);
  SmallVector<Value, 4> extents(4), dynamicSizes;
  const int64_t tiles[] = {configuration.tileM, configuration.tileN};
  for (unsigned axis = 0; axis < 2; ++axis) {
    int64_t tile = tiles[axis];
    if (!type.isDynamicDim(axis)) {
      int64_t extent = type.getDimSize(axis);
      shape[axis] = extent / tile + (extent % tile != 0);
      shape[axis + 2] = std::min(extent, tile);
      continue;
    }
    // Read the original allocation operands, not a descriptor defined later
    // at the contraction. The replacement stays at the same lifetime boundary.
    Value extent = allocation.getDynamicSizes()[type.getDynamicDimIndex(axis)];
    Value width = index(b, loc, tile);
    Value remainder = b.create<arith::RemSIOp>(loc, extent, width);
    Value extra = b.create<arith::SelectOp>(loc,
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, remainder, zero),
        one, zero);
    extents[axis] = add(b, loc,
        b.create<arith::DivSIOp>(loc, extent, width), extra);
    extents[axis + 2] = b.create<arith::MinSIOp>(loc, extent, width);
  }
  for (unsigned axis = 0; axis < 4; ++axis)
    if (ShapedType::isDynamic(shape[axis])) dynamicSizes.push_back(extents[axis]);
  auto backing = b.create<memref::AllocOp>(loc,
      MemRefType::get(shape, type.getElementType(), MemRefLayoutAttrInterface{},
                     type.getMemorySpace()), dynamicSizes);
  if (Attribute alignment = allocation->getAttr("alignment"))
    backing->setAttr("alignment", alignment);
  return backing;
}

Value blockedAccumulatorTile(OpBuilder &b, Location loc, Value backing,
                             const Configuration &configuration,
                             Value m, Value n, Value rows, Value columns) {
  Value bm = index(b, loc, configuration.tileM);
  Value bn = index(b, loc, configuration.tileN);
  SmallVector<OpFoldResult> offsets{
      b.create<arith::DivSIOp>(loc, m, bm).getResult(),
      b.create<arith::DivSIOp>(loc, n, bn).getResult(),
      b.create<arith::RemSIOp>(loc, m, bm).getResult(),
      b.create<arith::RemSIOp>(loc, n, bn).getResult()};
  SmallVector<OpFoldResult> sizes{b.getIndexAttr(1), b.getIndexAttr(1),
                                getAsOpFoldResult(rows), getAsOpFoldResult(columns)};
  SmallVector<OpFoldResult> strides(4, b.getIndexAttr(1));
  const int64_t shape[] = {
      getConstantIntValue(rows).value_or(ShapedType::kDynamic),
      getConstantIntValue(columns).value_or(ShapedType::kDynamic)};
  auto selected = cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
      shape, cast<MemRefType>(backing.getType()), offsets, sizes, strides));
  return b.create<memref::SubViewOp>(loc, selected, backing, offsets, sizes, strides);
}

Value privateAccumulator(OpBuilder &b, Location loc, Value original,
                         ArrayRef<int64_t> shape) {
  auto type = cast<MemRefType>(original.getType());
  auto workspace = b.create<memref::AllocaOp>(loc,
      MemRefType::get(shape, type.getElementType(), MemRefLayoutAttrInterface{},
                     type.getMemorySpace()));
  if (Attribute alignment = original.getDefiningOp()->getAttr("alignment"))
    workspace->setAttr("alignment", alignment);
  return workspace;
}

LogicalResult block(linalg::GenericOp operation, const Configuration &config,
                    const ImplementationRegistry &implementations, ImplementationInputs &inputs,
                    bool parallelTiles = false) {
  auto implementation = implementations.lookup(operation);
  if (failed(implementation)) return failure();
  auto binding = operation->getAttrOfType<ImplementationAttr>("intent_cpu.implementation");
  if (!(*implementation)->formTile)
    return operation.emitError("selected contraction implementation has no tile expansion");
  auto shared = operation->getParentOfType<func::FuncOp>()->getAttrOfType<ConfigurationAttr>(
      "intent_cpu.configuration");
  Value lhs = operation.getInputs()[0], rhs = operation.getInputs()[1];
  Value output = operation.getOutputs()[0];
  auto initialization = findContractionInitialization(operation);
  if (!initialization)
    return operation.emitError("CPU contraction accumulator initialization is missing");
  bool keepInitialization;
  bool compactVersion = false;
  std::optional<ContractionEpilogue> epilogue;
  {
    StorageAnalysis storage(operation->getParentOfType<func::FuncOp>());
    Value root = storage.uniqueOrigin(output);
    keepInitialization = !initialization->erasable ||
        ((*implementation)->contraction.completePrivateInitialization && root &&
         isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(root.getDefiningOp()));
    epilogue = queryContractionEpilogue(operation, initialization->operation, storage);
    if (epilogue && initialization->erasable)
      compactVersion = canCompactContractionAccumulator(
          operation, initialization->operation, *epilogue, storage);
  }
  Value initial = initialization->value;
  auto requirements = (*implementation)->inputRequirements(operation, shared, binding);
  if (auto reason = checkInputRequirements(operation, requirements))
    return operation.emitError(*reason);
  auto capabilities = operation->getParentOfType<ModuleOp>()->getAttrOfType<CapabilitiesAttr>("intent_cpu.capabilities");
  bool serialTiles = !parallelTiles && (operation->getParentOfType<scf::ForOp>() ||
                                       operation->getParentOfType<scf::ParallelOp>());
  int64_t groupM = 0, groupN = 0;
  for (auto requirement : requirements) {
    if (requirement.reuse != InputReuse::Group) continue;
    auto map = operation.getIndexingMapsArray()[requirement.operand];
    auto axis = dyn_cast<AffineDimExpr>(map.getResult(requirement.panelAxis));
    if (!axis || axis.getPosition() > 1)
      return operation.emitError("input supply grouping requires a free contraction axis");
    auto &group = axis.getPosition() == 0 ? groupM : groupN;
    group = group ? std::min(group, requirement.panelSize) : requirement.panelSize;
  }
  std::optional<std::array<int64_t, 2>> compact;
  if (compactVersion)
    compact = compactAccumulatorShape(operation, config, requirements,
                                      capabilities);
  bool earlyCompact = compact.has_value();
  if (compact) keepInitialization = false;
  if (epilogue)
    for (Operation *dependency : epilogue->dependencies)
      dependency->moveBefore(operation);
  OpBuilder b(operation);
  Location loc = operation.getLoc();
  Value zero = index(b, loc, 0), one = index(b, loc, 1);
  Value mSize = b.create<memref::DimOp>(loc, lhs, 0);
  Value nSize = b.create<memref::DimOp>(loc, rhs, 1);
  Value kSize = b.create<memref::DimOp>(loc, lhs, 1);
  bool staticParallel = (*implementation)->contraction.staticParallelExtent;
  Value bm, bn, bk, mTasks, nTasks, fullM, fullN;
  auto materializeGeometry = [&] {
    bm = index(b, loc, config.tileM);
    bn = index(b, loc, config.tileN);
    bk = index(b, loc, config.tileK);
    mTasks = b.create<arith::CeilDivSIOp>(loc, mSize, bm);
    nTasks = b.create<arith::CeilDivSIOp>(loc, nSize, bn);
    if (staticParallel) {
      fullM = b.create<arith::DivSIOp>(loc, mSize, bm);
      fullN = b.create<arith::DivSIOp>(loc, nSize, bn);
      mTasks = add(b, loc, fullM, b.create<arith::RemSIOp>(loc, mSize, bm));
      nTasks = add(b, loc, fullN, b.create<arith::RemSIOp>(loc, nSize, bn));
    }
  };
  Value nonempty = b.create<arith::AndIOp>(loc,
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, mSize, zero),
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, nSize, zero));
  if (!compact)
    nonempty = b.create<arith::AndIOp>(loc, nonempty,
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, kSize, zero));
  auto active = b.create<scf::IfOp>(loc, nonempty,
      !compact && (!keepInitialization || epilogue.has_value()));
  scf::IfOp emptyReduction;
  if (!compact && (!keepInitialization || epilogue)) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(active.elseBlock());
    emptyReduction = b.create<scf::IfOp>(loc,
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, kSize, zero), false);
  }
  operation->moveBefore(active.thenBlock()->getTerminator());
  b.setInsertionPoint(operation);
  std::optional<InputCohort> cohort;
  if (!inputs.hasReusableScope(operation, requirements)) {
    StorageAnalysis activeStorage(operation->getParentOfType<func::FuncOp>());
    cohort = inputCohort(operation, requirements, config, capabilities, activeStorage);
  }
  SmallVector<InputSupply> supplies;
  if (!cohort) {
    auto prepared = inputs.prepare(operation, requirements);
    if (failed(prepared)) return failure();
    supplies = std::move(*prepared);
  }
  // Consumers already prepared outside the future workset do not require a
  // K-outer cohort here. Preserve those snapshots and their preparation count;
  // only Group scratch remains live alongside this macro accumulator.
  if (compactVersion && !compact && !cohort &&
      inputs.preparedOutsideWorkset(operation, supplies)) {
    SmallVector<InputRequirement> localRequirements;
    for (const InputRequirement &requirement : requirements)
      if (requirement.reuse == InputReuse::Group)
        localRequirements.push_back(requirement);
    compact = compactAccumulatorShape(operation, config, localRequirements,
                                      capabilities);
  }
  int64_t cohortCapacity = compactVersion && cohort
      ? accumulatorCohortCapacity(operation, config, *cohort, capabilities) : 0;
  bool blocked = compactVersion && !compact && !cohortCapacity &&
                 canBlockAccumulator(operation, config);
  bool initializeBacking = blocked && keepInitialization;
  Value backing;
  if (blocked) {
    backing = createBlockedAccumulator(output, config);
    if (initializeBacking) {
      OpBuilder initialize(initialization->operation);
      initialize.create<linalg::FillOp>(initialization->operation->getLoc(),
          ValueRange{initial}, ValueRange{backing});
    }
  }
  if (compact || cohortCapacity || blocked) keepInitialization = false;
  bool tiledEmptyReduction = !earlyCompact && (compact || cohortCapacity || blocked);
  if (tiledEmptyReduction) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPoint(active);
    materializeGeometry();
  } else {
    materializeGeometry();
  }
  if (emptyReduction && !tiledEmptyReduction) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(emptyReduction.thenBlock());
    if (!keepInitialization)
      b.create<linalg::FillOp>(loc, ValueRange{initial}, ValueRange{output});
    if (epilogue) b.clone(*epilogue->consumer);
  }
  LogicalResult status = success();
  auto group = [&](Value begin, Value extent, int64_t size,
                   const std::function<void(Value, Value)> &body) {
    if (!size) { body(begin, extent); return; }
    Value step = index(b, loc, size);
    Value full = multiply(b, loc, b.create<arith::DivSIOp>(loc, extent, step), step);
    loop(b, loc, zero, full, size, [&](Value offset) { body(add(b, loc, begin, offset), step); });
    Value tail = b.create<arith::SubIOp>(loc, extent, full);
    auto branch = b.create<scf::IfOp>(loc, b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, tail, zero), false);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(branch.thenBlock());
    body(add(b, loc, begin, full), tail);
  };
  auto issue = [&](Value m, Value n, Value rows, Value columns,
                   Value destination, Value kBegin, Value depth, bool first) {
    ContractionTile tile{lhs, rhs, destination, initial, m, rows, n, columns,
                         kBegin, depth, first, {}};
    auto local = inputs.prepareGroup(b, operation, tile, shared, requirements);
    if (failed(local)) { status = failure(); return; }
    llvm::append_range(*local, supplies);
    tile.inputs = *local;
    if (failed((*implementation)->formTile(b, operation, tile, shared, binding))) status = failure();
  };
  auto tileBlock = [&](Value mBegin, Value nBegin, Value mCount, Value nCount,
                       Value destination, Value kBegin, Value depth, bool first) {
    group(mBegin, mCount, groupM, [&](Value m, Value rows) {
      group(nBegin, nCount, groupN, [&](Value n, Value columns) {
        Value selected = subview(b, loc, destination,
            {b.create<arith::SubIOp>(loc, m, mBegin).getResult(),
             b.create<arith::SubIOp>(loc, n, nBegin).getResult()}, {rows, columns});
        issue(m, n, rows, columns, selected, kBegin, depth, first);
      });
    });
  };
  auto reduction = [&](llvm::function_ref<void(Value, Value, bool)> kBlock) {
    if ((*implementation)->contraction.staticReductionExtent) {
      Value fullEnd = b.create<arith::SubIOp>(loc, kSize, b.create<arith::RemSIOp>(loc, kSize, bk));
      Value firstEnd = b.create<arith::MinSIOp>(loc, fullEnd, bk);
      loop(b, loc, zero, firstEnd, config.tileK, [&](Value k) { kBlock(k, bk, true); });
      loop(b, loc, bk, fullEnd, config.tileK, [&](Value k) { kBlock(k, bk, false); });
      Value noFullTile = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, fullEnd, zero);
      Value firstTailEnd = b.create<arith::SelectOp>(loc, noFullTile,
          b.create<arith::MinSIOp>(loc, kSize, one), zero);
      loop(b, loc, zero, firstTailEnd, 1, [&](Value k) { kBlock(k, one, true); });
      Value tailBegin = b.create<arith::MaxSIOp>(loc, fullEnd, one);
      loop(b, loc, tailBegin, kSize, 1, [&](Value k) { kBlock(k, one, false); });
    } else {
      Value firstEnd = b.create<arith::MinSIOp>(loc, kSize, bk);
      auto dynamicBlock = [&](Value k, bool first) {
        Value depth = b.create<arith::MinSIOp>(loc, b.create<arith::SubIOp>(loc, kSize, k), bk);
        kBlock(k, depth, first);
      };
      loop(b, loc, zero, firstEnd, config.tileK, [&](Value k) { dynamicBlock(k, true); });
      loop(b, loc, firstEnd, kSize, config.tileK, [&](Value k) { dynamicBlock(k, false); });
    }
  };
  auto coordinate = [&](unsigned axis, Value ordinal,
                        llvm::function_ref<void(Value, Value)> body) {
    Value size = axis == 0 ? mSize : nSize;
    Value block = axis == 0 ? bm : bn;
    if (!staticParallel) {
      Value begin = multiply(b, loc, ordinal, block);
      Value count = b.create<arith::MinSIOp>(loc,
          b.create<arith::SubIOp>(loc, size, begin), block);
      body(begin, count);
      return;
    }
    Value full = axis == 0 ? fullM : fullN;
    Value complete = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, ordinal, full);
    auto branch = b.create<scf::IfOp>(loc, complete, true);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(branch.thenBlock());
    body(multiply(b, loc, ordinal, block), block);
    b.setInsertionPointToStart(branch.elseBlock());
    Value begin = add(b, loc, multiply(b, loc, full, block), b.create<arith::SubIOp>(loc, ordinal, full));
    body(begin, one);
  };
  auto emitTile = [&](Value m, Value n) {
    coordinate(0, m, [&](Value mBegin, Value mCount) {
      coordinate(1, n, [&](Value nBegin, Value nCount) {
        Value destination;
        if (compact) {
          Value workspace = privateAccumulator(b, loc, output, *compact);
          destination = subview(b, loc, workspace,
              {b.getIndexAttr(0), b.getIndexAttr(0)}, {mCount, nCount});
          if ((*implementation)->contraction.completePrivateInitialization) {
            b.create<linalg::FillOp>(loc, ValueRange{initial}, ValueRange{workspace});
          } else if (earlyCompact) {
            auto empty = b.create<scf::IfOp>(loc,
                b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, kSize, zero), false);
            OpBuilder::InsertionGuard guard(b);
            b.setInsertionPointToStart(empty.thenBlock());
            b.create<linalg::FillOp>(loc, ValueRange{initial}, ValueRange{destination});
          }
        } else if (blocked) {
          destination = blockedAccumulatorTile(b, loc, backing, config,
              mBegin, nBegin, mCount, nCount);
        } else {
          destination = subview(b, loc, output, {mBegin, nBegin}, {mCount, nCount});
        }
        reduction([&](Value kBegin, Value depth, bool first) {
          tileBlock(mBegin, nBegin, mCount, nCount, destination, kBegin, depth, first);
        });
        if (epilogue && failed(emitContractionEpilogue(
                b, *epilogue, ValueRange{mBegin, nBegin}, ValueRange{mCount, nCount},
                compact || blocked ? destination : Value{})))
          status = failure();
      });
    });
  };
  if (cohort) {
    unsigned freeAxis = cohort->freeAxis, reuseAxis = 1 - freeAxis;
    Value freeTasks = freeAxis == 0 ? mTasks : nTasks;
    Value reuseTasks = reuseAxis == 0 ? mTasks : nTasks;
    Value span = reuseTasks;
    if (!serialTiles) {
      // Share as many independent tiles as possible while retaining at least
      // min(original tile count, workers) independent tasks. All divisors are
      // positive inside the complete M/N/K execution guard above.
      Value needed = b.create<arith::CeilDivSIOp>(loc,
          index(b, loc, capabilities.getWorkers()), freeTasks);
      span = b.create<arith::MaxSIOp>(loc, one,
          b.create<arith::DivSIOp>(loc, reuseTasks, needed));
    }
    if (cohortCapacity)
      span = b.create<arith::MinSIOp>(loc, span, index(b, loc, cohortCapacity));
    Value groups = b.create<arith::CeilDivSIOp>(loc, reuseTasks, span);
    auto emitCohort = [&](Value free, Value group) {
      Value groupBegin = multiply(b, loc, group, span);
      Value groupCount = b.create<arith::MinSIOp>(loc, span,
          b.create<arith::SubIOp>(loc, reuseTasks, groupBegin));
      Value groupEnd = add(b, loc, groupBegin, groupCount);
      coordinate(freeAxis, free, [&](Value freeBegin, Value freeCount) {
        Value accumulators;
        if (cohortCapacity) {
          accumulators = privateAccumulator(b, loc, output,
              {cohortCapacity, config.tileM, config.tileN});
          if ((*implementation)->contraction.completePrivateInitialization)
            b.create<linalg::FillOp>(loc, ValueRange{initial}, ValueRange{accumulators});
        }
        Value preparedStorage = inputs.createPrivateStorage(b, loc,
            cohort->requirement, cohort->capacities);
        // Every output tile sees the same ascending K chunks and first flag.
        // Only independent output tiles are interleaved to share preparation.
        // A K chunk's synchronous consumers finish before the next copy reuses
        // this cohort's slot; the source reads remain at each original K point.
        reduction([&](Value kBegin, Value depth, bool first) {
          Value last;
          if (epilogue)
            last = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, depth,
                b.create<arith::SubIOp>(loc, kSize, kBegin));
          SmallVector<Value> begins(2), counts(2);
          auto map = operation.getIndexingMapsArray()[cohort->requirement.operand];
          for (auto [axis, expression] : llvm::enumerate(map.getResults())) {
            bool paired = cast<AffineDimExpr>(expression).getPosition() == 2;
            begins[axis] = paired ? kBegin : freeBegin;
            counts[axis] = paired ? depth : freeCount;
          }
          Value source = operation.getInputs()[cohort->requirement.operand];
          Value window = subview(b, loc, source,
              SmallVector<OpFoldResult>(begins.begin(), begins.end()),
              SmallVector<OpFoldResult>(counts.begin(), counts.end()));
          auto consumers = b.create<scf::ForOp>(loc, groupBegin, groupEnd, one);
          auto supply = inputs.prepareAt(window, begins, cohort->requirement,
                                         cohort->capacities, preparedStorage, consumers);
          if (failed(supply)) { status = failure(); return; }
          supplies.assign(1, *supply);
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(consumers.getBody());
          coordinate(reuseAxis, consumers.getInductionVar(), [&](Value reuseBegin, Value reuseCount) {
            Value mBegin = freeAxis == 0 ? freeBegin : reuseBegin;
            Value nBegin = freeAxis == 1 ? freeBegin : reuseBegin;
            Value mCount = freeAxis == 0 ? freeCount : reuseCount;
            Value nCount = freeAxis == 1 ? freeCount : reuseCount;
            Value destination;
            if (cohortCapacity) {
              SmallVector<OpFoldResult> offsets{
                  b.create<arith::SubIOp>(loc, consumers.getInductionVar(), groupBegin).getResult(),
                  b.getIndexAttr(0), b.getIndexAttr(0)};
              SmallVector<OpFoldResult> sizes{
                  b.getIndexAttr(1), getAsOpFoldResult(mCount), getAsOpFoldResult(nCount)};
              SmallVector<OpFoldResult> strides(3, b.getIndexAttr(1));
              const int64_t shape[] = {
                  getConstantIntValue(mCount).value_or(ShapedType::kDynamic),
                  getConstantIntValue(nCount).value_or(ShapedType::kDynamic)};
              auto selected = cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
                  shape, cast<MemRefType>(accumulators.getType()), offsets, sizes, strides));
              destination = b.create<memref::SubViewOp>(loc, selected, accumulators,
                                                       offsets, sizes, strides);
            } else if (blocked) {
              destination = blockedAccumulatorTile(b, loc, backing, config,
                  mBegin, nBegin, mCount, nCount);
            } else {
              destination = subview(b, loc, output, {mBegin, nBegin}, {mCount, nCount});
            }
            tileBlock(freeAxis == 0 ? freeBegin : reuseBegin,
                      freeAxis == 1 ? freeBegin : reuseBegin,
                      freeAxis == 0 ? freeCount : reuseCount,
                      freeAxis == 1 ? freeCount : reuseCount,
                      destination, kBegin, depth, first);
            if (epilogue) {
              auto completed = b.create<scf::IfOp>(loc, last, false);
              OpBuilder::InsertionGuard guard(b);
              b.setInsertionPointToStart(completed.thenBlock());
              if (failed(emitContractionEpilogue(b, *epilogue,
                  ValueRange{freeAxis == 0 ? freeBegin : reuseBegin,
                             freeAxis == 1 ? freeBegin : reuseBegin},
                  ValueRange{freeAxis == 0 ? freeCount : reuseCount,
                             freeAxis == 1 ? freeCount : reuseCount},
                  blocked || cohortCapacity ? destination : Value{})))
                status = failure();
            }
          });
        });
      });
    };
    if (serialTiles) {
      loop(b, loc, zero, freeTasks, 1, [&](Value free) {
        if (cohortCapacity)
          loop(b, loc, zero, groups, 1, [&](Value group) { emitCohort(free, group); });
        else
          emitCohort(free, zero);
      });
    } else {
      auto parallel = b.create<scf::ParallelOp>(loc, ValueRange{zero, zero},
          ValueRange{freeTasks, groups}, ValueRange{one, one});
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(parallel.getBody());
      emitCohort(parallel.getInductionVars()[0], parallel.getInductionVars()[1]);
    }
  } else if (serialTiles) {
    loop(b, loc, zero, mTasks, 1, [&](Value m) {
      loop(b, loc, zero, nTasks, 1, [&](Value n) { emitTile(m, n); });
    });
  } else {
    auto parallel = b.create<scf::ParallelOp>(loc, ValueRange{zero, zero},
        ValueRange{mTasks, nTasks}, ValueRange{one, one});
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(parallel.getBody());
    emitTile(parallel.getInductionVars()[0], parallel.getInductionVars()[1]);
  }
  if (tiledEmptyReduction) {
    // No source preparation or contraction executes for K=0. Complete exactly
    // the logical output tiles using the original scalar initializer/epilogue.
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(emptyReduction.thenBlock());
    auto emptyTile = [&](Value m, Value n) {
      coordinate(0, m, [&](Value mBegin, Value mCount) {
        coordinate(1, n, [&](Value nBegin, Value nCount) {
          Value destination;
          if (blocked) {
            destination = blockedAccumulatorTile(b, loc, backing, config,
                mBegin, nBegin, mCount, nCount);
          } else {
            Value workspace = privateAccumulator(b, loc, output,
                {config.tileM, config.tileN});
            destination = subview(b, loc, workspace,
                {b.getIndexAttr(0), b.getIndexAttr(0)}, {mCount, nCount});
          }
          if (!initializeBacking)
            b.create<linalg::FillOp>(loc, ValueRange{initial}, ValueRange{destination});
          if (failed(emitContractionEpilogue(b, *epilogue,
                  ValueRange{mBegin, nBegin}, ValueRange{mCount, nCount},
                  destination))) status = failure();
        });
      });
    };
    if (serialTiles) {
      loop(b, loc, zero, mTasks, 1, [&](Value m) {
        loop(b, loc, zero, nTasks, 1, [&](Value n) { emptyTile(m, n); });
      });
    } else {
      auto parallel = b.create<scf::ParallelOp>(loc, ValueRange{zero, zero},
          ValueRange{mTasks, nTasks}, ValueRange{one, one});
      b.setInsertionPointToStart(parallel.getBody());
      emptyTile(parallel.getInductionVars()[0], parallel.getInductionVars()[1]);
    }
  }
  if (failed(status)) return failure();
  if (epilogue) epilogue->consumer.erase();
  if (!keepInitialization) initialization->operation->erase();
  operation.erase();
  if ((compact || cohortCapacity || blocked) &&
      failed(eraseContractionAccumulator(output, backing)))
    return failure();
  return success();
}

LogicalResult exposeGroupedTiles(func::FuncOp function, const Configuration &config,
    const ImplementationRegistry &implementations, ImplementationInputs &inputs) {
  auto capabilities = function->getParentOfType<ModuleOp>()->getAttrOfType<CapabilitiesAttr>("intent_cpu.capabilities");
  if (capabilities.getWorkers() <= 1) return success();
  SmallVector<scf::ParallelOp> groups;
  function.walk([&](scf::ParallelOp loop) {
    if (loop->getParentOfType<scf::ParallelOp>() || loop->getParentOfType<scf::ForOp>() ||
        loop.getNumLoops() != 1 || loop.getNumResults() ||
        !matchPattern(loop.getLowerBound()[0], m_Zero()) || !matchPattern(loop.getStep()[0], m_One())) return;
    if (llvm::any_of(loop.getBody()->getOps<linalg::GenericOp>(), [](linalg::GenericOp operation) {
          return isMatrixContraction(operation) && !operation->hasAttr("intent_cpu.microtile");
        })) groups.push_back(loop);
  });
  for (auto group : groups) {
    OpBuilder b(group);
    Location loc = group.getLoc();
    Value tasks = b.create<arith::CeilDivSIOp>(loc, group.getUpperBound()[0], index(b, loc, config.taskGrain));
    Value fewGroups = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
        tasks, index(b, loc, capabilities.getWorkers()));
    auto selection = b.create<scf::IfOp>(loc, fewGroups, true);
    b.setInsertionPointToStart(selection.thenBlock());
    auto serial = b.create<scf::ForOp>(loc, group.getLowerBound()[0], group.getUpperBound()[0], group.getStep()[0]);
    b.setInsertionPointToStart(serial.getBody());
    IRMapping mapping;
    mapping.map(group.getInductionVars()[0], serial.getInductionVar());
    SmallVector<linalg::GenericOp> contractions;
    for (Operation &operation : group.getBody()->without_terminator()) {
      Operation *copy = b.clone(operation, mapping);
      if (auto contraction = dyn_cast<linalg::GenericOp>(copy);
          contraction && isMatrixContraction(contraction) && !contraction->hasAttr("intent_cpu.microtile"))
        contractions.push_back(contraction);
    }
    group->moveBefore(selection.elseBlock()->getTerminator());
    // Group preparation and epilogues retain their lexical owner. Tile tasks
    // finish before the group consumes or releases their shared buffers.
    for (auto contraction : contractions)
      if (failed(block(contraction, config, implementations, inputs, true))) return failure();
  }
  return success();
}

}

LogicalResult blockContractions(func::FuncOp function, const Configuration &configuration,
                                const ImplementationRegistry &implementations) {
  SmallVector<linalg::GenericOp> contractions;
  ImplementationInputs inputs(function);
  if (failed(exposeGroupedTiles(function, configuration, implementations, inputs))) return failure();
  function.walk([&](linalg::GenericOp operation) {
    if (isMatrixContraction(operation) && !operation->hasAttr("intent_cpu.microtile"))
      contractions.push_back(operation);
  });
  for (linalg::GenericOp operation : contractions)
    if (failed(block(operation, configuration, implementations, inputs))) return failure();
  if (failed(inputs.fuseProducerCopies())) return failure();
  eraseDeadPrivateBuffers(function);
  return success();
}

}
