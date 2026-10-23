#include "ContractionEpilogues.h"
#include "Intent/Dialect/CPU/Analysis/Contractions.h"
#include "Intent/Dialect/CPU/Analysis/ExtentRelations.h"
#include "Intent/Dialect/CPU/Analysis/ViewRelations.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

namespace intent::cpu {
namespace {

bool sameExtent(Value first, int64_t firstAxis, Value second,
                int64_t secondAxis) {
  return haveEqualExtents(ValueBoundsConstraintSet::Variable(first, firstAxis),
                         ValueBoundsConstraintSet::Variable(second, secondAxis));
}

bool sameDomain(linalg::GenericOp consumer, unsigned input, Value accumulator) {
  auto maps = consumer.getIndexingMapsArray();
  AffineMap source = maps[input];
  if (!source.isPermutation() || !maps.back().isPermutation()) return false;
  SmallVector<unsigned> sourceAxes(consumer.getNumLoops());
  for (auto [axis, expression] : llvm::enumerate(source.getResults()))
    sourceAxes[cast<AffineDimExpr>(expression).getPosition()] = axis;
  for (auto [operand, map] : llvm::zip(consumer->getOperands(), maps)) {
    auto type = dyn_cast<MemRefType>(operand.getType());
    if (!type) continue;
    if (map.getNumSymbols() || !map.isProjectedPermutation(/*allowZeroInResults=*/true))
      return false;
    for (auto [axis, expression] : llvm::enumerate(map.getResults())) {
      if (auto dimension = dyn_cast<AffineDimExpr>(expression)) {
        if (!sameExtent(operand, axis, accumulator,
                        sourceAxes[dimension.getPosition()])) return false;
      } else if (!haveEqualExtents(
                     ValueBoundsConstraintSet::Variable(operand, axis),
                     ValueBoundsConstraintSet::Variable(IntegerAttr::get(
                         IndexType::get(consumer.getContext()), 1)))) {
        return false;
      }
    }
  }
  return true;
}

} // namespace

std::optional<ContractionEpilogue> queryContractionEpilogue(
    linalg::GenericOp contraction, Operation *initialization,
    StorageAnalysis &storage) {
  Value accumulator = contraction.getOutputs()[0];
  auto type = cast<MemRefType>(accumulator.getType());
  auto axes = queryContractionAxes(contraction);
  if (!axes || !type.getRank()) return std::nullopt;
  for (unsigned operand = 0; operand != 2; ++operand) {
    const auto &resultAxes = operand ? axes->rhsResultAxes : axes->lhsResultAxes;
    for (auto [axis, result] : llvm::enumerate(resultAxes))
      if (result && !sameExtent(accumulator, *result,
                               contraction.getInputs()[operand], axis))
        return std::nullopt;
  }

  auto aliases = storage.aliases(accumulator);
  if (!aliases.complete) return std::nullopt;
  linalg::GenericOp consumer;
  for (Operation *user : aliases.users) {
    if (user == contraction || user == initialization ||
        isa<memref::DeallocOp, memref::DimOp, memref::ExtractStridedMetadataOp>(user) ||
        isStorageAliasOperation(user)) continue;
    auto candidate = dyn_cast<linalg::GenericOp>(user);
    if (!candidate || consumer) return std::nullopt;
    consumer = candidate;
  }
  if (!consumer || consumer->getBlock() != contraction->getBlock() ||
      !contraction->isBeforeInBlock(consumer) || consumer.getNumResults() ||
      consumer.getOutputs().size() != 1 || consumer.getNumLoops() != type.getRank() ||
      consumer.getNumReductionLoops() ||
      !consumer.getRegion().front().getArguments().back().use_empty())
    return std::nullopt;
  std::optional<unsigned> sourceInput;
  for (auto [number, input] : llvm::enumerate(consumer.getInputs()))
    if (input == accumulator) { sourceInput = number; break; }
  if (!sourceInput || !sameDomain(consumer, *sourceInput, accumulator))
    return std::nullopt;
  for (Operation &operation : consumer.getRegion().front().without_terminator())
    if (operation.getNumRegions() || !isMemoryEffectFree(&operation))
      return std::nullopt;

  // The original consumer runs after every contraction read. Publishing tiles
  // early is legal only when none of those pending reads can observe the write.
  // In particular, an external Out direction is not itself a disjointness fact.
  Value destination = consumer.getOutputs()[0];
  bool inplace = destination == accumulator;
  auto maps = consumer.getIndexingMapsArray();
  if (inplace && maps.back() != maps[*sourceInput]) return std::nullopt;
  bool privateAccumulator = storage.uniqueOrigin(accumulator) == accumulator &&
      isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(accumulator.getDefiningOp());
  // Output forwarding may already have placed C in its public destination.
  // The same-coordinate epilogue can finish each non-overlapping rectangle
  // there, provided no still-pending contraction read observes those writes.
  if (!privateAccumulator && (!inplace || !isContiguousDescriptor(accumulator)))
    return std::nullopt;
  for (Value input : contraction->getOperands()) {
    if (inplace && input == accumulator) continue;
    if (!storage.disjoint(destination, input)) return std::nullopt;
  }
  auto computation = storage.effects(contraction);
  if (!computation.complete || computation.ordered) return std::nullopt;
  for (const StorageEffect &entry : computation.entries) {
    Value memory = entry.effect.getValue();
    if (inplace && memory == accumulator) continue;
    if (!memory || (isa<BaseMemRefType>(memory.getType()) &&
                    !storage.disjoint(memory, destination)))
      return std::nullopt;
  }
  for (auto [number, input] : llvm::enumerate(consumer.getInputs())) {
    if (!isa<BaseMemRefType>(input.getType())) continue;
    if (!(inplace && input == accumulator) && !storage.disjoint(destination, input))
      return std::nullopt;
    // Repeated accumulator inputs must read this same completed rectangle,
    // not a transpose or another tile that is still being accumulated.
    if (input == accumulator && maps[number] != maps[*sourceInput])
      return std::nullopt;
    if (input != accumulator &&
        (!storage.preserves(contraction, input) ||
         !storage.unchangedBetween(input, contraction, consumer)))
      return std::nullopt;
  }
  for (Operation *between = contraction->getNextNode(); between != consumer;
       between = between->getNextNode()) {
    if (between == destination.getDefiningOp() &&
        isa<memref::AllocOp, memref::AllocaOp>(between)) continue;
    auto effects = storage.effects(between);
    if (!effects.complete || effects.ordered) return std::nullopt;
    for (const StorageEffect &entry : effects.entries) {
      Value memory = entry.effect.getValue();
      if (!memory || (isa<BaseMemRefType>(memory.getType()) &&
                      !storage.disjoint(memory, destination)))
        return std::nullopt;
    }
  }

  DominanceInfo dominance(contraction->getParentOfType<func::FuncOp>());
  llvm::SetVector<Operation *> dependencies;
  std::function<bool(Value)> available = [&](Value value) {
    if (dominance.properlyDominates(value, contraction)) return true;
    Operation *definition = value.getDefiningOp();
    if (!definition || definition->getBlock() != contraction->getBlock() ||
        definition->getNumRegions()) return false;
    // A consumer's private destination may be allocated just after the
    // contraction. Its independent size/descriptor DAG can move before it;
    // other memory effects and dependent allocations cannot.
    bool allocation = isa<memref::AllocOp, memref::AllocaOp>(definition);
    if (!allocation && (!isMemoryEffectFree(definition) ||
                        !isSpeculatable(definition))) return false;
    if (dependencies.contains(definition)) return true;
    if (!llvm::all_of(definition->getOperands(), available)) return false;
    dependencies.insert(definition);
    return true;
  };
  if (!llvm::all_of(consumer->getOperands(), available)) return std::nullopt;
  llvm::SetVector<Value> captures;
  getUsedValuesDefinedAbove(consumer.getRegion(), captures);
  for (Value capture : captures)
    if (isa<BaseMemRefType>(capture.getType()) || !available(capture))
      return std::nullopt;
  return ContractionEpilogue{consumer, *sourceInput,
                            {dependencies.begin(), dependencies.end()}};
}

bool canCompactContractionAccumulator(linalg::GenericOp contraction,
    Operation *initialization, ContractionEpilogue &epilogue,
    StorageAnalysis &storage) {
  Value accumulator = contraction.getOutputs()[0];
  auto type = cast<MemRefType>(accumulator.getType());
  if (!type.getLayout().isIdentity() ||
      !isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(accumulator.getDefiningOp()))
    return false;
  for (Value input : contraction.getInputs())
    if (!storage.disjoint(input, accumulator)) return false;
  for (Operation *user : accumulator.getUsers()) {
    if (user == contraction || user == initialization ||
        user == epilogue.consumer || isa<memref::DeallocOp>(user)) continue;
    auto dimension = dyn_cast<memref::DimOp>(user);
    if (!dimension || !dimension.getConstantIndex()) return false;
  }
  return true;
}

LogicalResult eraseContractionAccumulator(Value accumulator,
                                          Value replacementStorage) {
  Operation *allocation = accumulator.getDefiningOp();
  auto type = cast<MemRefType>(accumulator.getType());
  SmallVector<Value> dynamicSizes;
  if (auto heap = dyn_cast<memref::AllocOp>(allocation))
    llvm::append_range(dynamicSizes, heap.getDynamicSizes());
  else
    llvm::append_range(dynamicSizes, cast<memref::AllocaOp>(allocation).getDynamicSizes());
  SmallVector<Operation *> users(accumulator.getUsers());
  for (Operation *user : users) {
    if (auto release = dyn_cast<memref::DeallocOp>(user)) {
      if (replacementStorage)
        release.getMemrefMutable().assign(replacementStorage);
      else
        release.erase();
      continue;
    }
    auto dimension = dyn_cast<memref::DimOp>(user);
    if (!dimension || !dimension.getConstantIndex())
      return allocation->emitError("compact contraction accumulator retains an unexpected observation");
    int64_t axis = *dimension.getConstantIndex();
    OpBuilder builder(dimension);
    Value extent = type.isDynamicDim(axis)
        ? dynamicSizes[type.getDynamicDimIndex(axis)]
        : builder.create<arith::ConstantIndexOp>(dimension.getLoc(), type.getDimSize(axis)).getResult();
    dimension.replaceAllUsesWith(extent);
    dimension.erase();
  }
  allocation->erase();
  return success();
}

LogicalResult emitContractionEpilogue(OpBuilder &builder,
    ContractionEpilogue &epilogue, ValueRange offsets, ValueRange sizes,
    Value completedTile) {
  auto consumer = epilogue.consumer;
  Block *block = builder.getInsertionBlock();
  auto end = builder.getInsertionPoint();
  Operation *previous = end == block->begin() ? nullptr : &*std::prev(end);
  AffineMap source = consumer.getIndexingMapsArray()[epilogue.accumulatorInput];
  SmallVector<OpFoldResult, 2> loopOffsets(2), loopSizes(2);
  for (auto [axis, expression] : llvm::enumerate(source.getResults())) {
    unsigned loop = cast<AffineDimExpr>(expression).getPosition();
    loopOffsets[loop] = offsets[axis];
    loopSizes[loop] = sizes[axis];
  }
  // Use the native Linalg tiling implementation's operand and index transport.
  // The selected output rectangle is complete, so no extra boundary clamp is
  // needed; the original dtype, scalar operations, maps and attributes survive.
  auto operands = linalg::makeTiledShapes(builder, consumer.getLoc(), consumer,
      consumer->getOperands(), loopOffsets, loopSizes, {}, true);
  if (completedTile) {
    Value original = consumer.getInputs()[epilogue.accumulatorInput];
    llvm::SetVector<Operation *> unusedSlices;
    for (auto [number, input] : llvm::enumerate(consumer.getInputs())) {
      if (input != original) continue;
      Value unused = operands[number];
      operands[number] = completedTile;
      // This temporary source slice is superseded by the actual compact C.
      // Other operand slices and linalg.index retain the logical output origin.
      if (unused != original)
        if (auto view = unused.getDefiningOp<memref::SubViewOp>())
          unusedSlices.insert(view);
    }
    for (Operation *view : unusedSlices)
      if (view->use_empty()) view->erase();
  }
  auto tile = cast<linalg::GenericOp>(builder.clone(*consumer));
  tile->setOperands(operands);
  linalg::offsetIndices(builder, tile, loopOffsets);

  // Native tiling constructs affine index expressions. Lower just this new
  // slice back to the CPU execution family's ordinary arithmetic operations.
  SmallVector<Operation *> affineIndices;
  for (auto current = previous ? std::next(previous->getIterator()) : block->begin();
       current != end; ++current)
    current->walk([&](Operation *operation) {
      if (isa<affine::AffineApplyOp, affine::AffineMinOp, affine::AffineMaxOp>(operation))
        affineIndices.push_back(operation);
    });
  if (affineIndices.empty()) return success();
  RewritePatternSet patterns(builder.getContext());
  populateAffineToStdConversionPatterns(patterns);
  GreedyRewriteConfig config;
  config.scope = block->getParent();
  config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;
  if (failed(applyOpPatternsGreedily(affineIndices, std::move(patterns), config)))
    return consumer.emitError("contraction epilogue index composition did not converge");
  return success();
}

void emitSlicedContractionEpilogue(OpBuilder &builder,
    ContractionEpilogue &epilogue, ArrayRef<unsigned> selectedAxes,
    ValueRange coordinates, Value completedSlice) {
  auto consumer = epilogue.consumer;
  Location loc = consumer.getLoc();
  auto maps = consumer.getIndexingMapsArray();
  AffineMap source = maps[epilogue.accumulatorInput];
  SmallVector<Value> fixed(consumer.getNumLoops());
  for (auto [axis, coordinate] : llvm::zip(selectedAxes, coordinates))
    fixed[cast<AffineDimExpr>(source.getResult(axis)).getPosition()] = coordinate;
  SmallVector<AffineExpr> remaining(fixed.size());
  unsigned rank = 0;
  for (auto [axis, coordinate] : llvm::enumerate(fixed))
    if (!coordinate) remaining[axis] = builder.getAffineDimExpr(rank++);

  SmallVector<Value> operands;
  SmallVector<AffineMap> slicedMaps;
  Value accumulator = consumer.getInputs()[epilogue.accumulatorInput];
  for (auto [memory, map] : llvm::zip(consumer->getOperands(), maps)) {
    auto type = dyn_cast<MemRefType>(memory.getType());
    SmallVector<AffineExpr> expressions;
    SmallVector<OpFoldResult> offsets, sizes, strides;
    SmallVector<int64_t> shape;
    bool sliced = false;
    for (auto [axis, expression] : llvm::enumerate(map.getResults())) {
      auto dimension = dyn_cast<AffineDimExpr>(expression);
      Value coordinate = dimension ? fixed[dimension.getPosition()] : Value{};
      offsets.push_back(coordinate ? OpFoldResult(coordinate) : builder.getIndexAttr(0));
      strides.push_back(builder.getIndexAttr(1));
      if (coordinate) {
        sizes.push_back(builder.getIndexAttr(1));
        sliced = true;
      } else {
        sizes.push_back(type.isDynamicDim(axis)
            ? OpFoldResult(builder.create<memref::DimOp>(loc, memory, axis).getResult())
            : builder.getIndexAttr(type.getDimSize(axis)));
        shape.push_back(type.getDimSize(axis));
        expressions.push_back(dimension ? remaining[dimension.getPosition()] : expression);
      }
    }
    if (memory == accumulator) memory = completedSlice;
    else if (sliced) {
      auto slicedType = cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
          shape, type, offsets, sizes, strides));
      memory = builder.create<memref::SubViewOp>(loc, slicedType, memory, offsets, sizes, strides);
    }
    operands.push_back(memory);
    slicedMaps.push_back(AffineMap::get(rank, 0, expressions, builder.getContext()));
  }
  auto slice = builder.create<linalg::GenericOp>(loc,
      ValueRange(operands).drop_back(), ValueRange(operands).take_back(), slicedMaps,
      SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel),
      [&](OpBuilder &body, Location at, ValueRange arguments) {
        IRMapping mapping;
        Block &original = consumer.getRegion().front();
        mapping.map(original.getArguments(), arguments);
        for (Operation &operation : original.without_terminator()) {
          if (auto index = dyn_cast<linalg::IndexOp>(operation)) {
            Value coordinate = fixed[index.getDim()];
            if (!coordinate)
              coordinate = body.create<linalg::IndexOp>(at,
                  cast<AffineDimExpr>(remaining[index.getDim()]).getPosition());
            mapping.map(index.getResult(), coordinate);
          } else body.clone(operation, mapping);
        }
        body.create<linalg::YieldOp>(at,
            mapping.lookupOrDefault(original.getTerminator()->getOperand(0)));
      });
  slice->setDiscardableAttrs(llvm::to_vector(consumer->getDiscardableAttrs()));
}

} // namespace intent::cpu
