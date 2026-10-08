#include "Intent/Dialect/CPU/Transforms/Contraction/Contraction.h"
#include "Intent/Dialect/CPU/Transforms/Storage/Storage.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/CPU/IR/CPUAttrs.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "Contractions.h"
#include "Intent/Dialect/CPU/Transforms/Structure/ProducerVersions.h"
#include "Intent/Analysis/ContractionAxes.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Intent/Dialect/CPU/Transforms/Implementation/Implementation.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace intent::cpu {
namespace {

linalg::GenericOp bufferProducer(Value buffer, Operation *consumer, bool soleConsumer,
                                bool sameBlock = true) {
  auto allocation = buffer.getDefiningOp<memref::AllocOp>();
  if (!allocation) return {};
  StorageAnalysis storage(consumer->getParentOfType<func::FuncOp>());
  auto current = findCurrentBufferWrite(buffer, consumer, storage);
  auto producer = current ? dyn_cast<linalg::GenericOp>(current->operation) : linalg::GenericOp{};
  if (!producer || producer == consumer ||
      (sameBlock && (producer->getBlock() != consumer->getBlock() || !producer->isBeforeInBlock(consumer))) ||
      producer.getNumResults() ||
      producer.getOutputs().size() != 1 || producer.getNumReductionLoops() ||
      !producer.getIndexingMapsArray().back().isIdentity() ||
      !completelyWritesBuffer(producer, buffer) ||
      !storage.preserves(consumer, buffer)) return {};
  if (soleConsumer) {
    auto observations = queryBufferVersionObservations(buffer, producer, storage);
    if (failed(observations) || observations->reads.size() != 1 ||
        observations->reads.front() != consumer) return {};
  }
  return producer;
}

bool isProjection(linalg::GenericOp operation) {
  if (operation.getInputs().size() != 1) return false;
  Block &body = operation.getRegion().front();
  return body.without_terminator().empty() &&
      body.getTerminator()->getOperand(0) == body.getArgument(0) &&
      operation.getIndexingMapsArray().front().isProjectedPermutation(/*allowZeroInResults=*/true);
}

// The map gives source indices in the reduction's product domain. Only pure
// axis projections are composed; arithmetic and numeric casts stay as inputs.
struct ProjectedSource {
  Value value;
  AffineMap map;
};

bool stripView(ProjectedSource &source, SmallVectorImpl<Operation *> &views) {
  Operation *operation = source.value.getDefiningOp();
  if (auto cast = dyn_cast_or_null<memref::CastOp>(operation)) {
    source.value = cast.getSource();
    views.push_back(operation);
    return true;
  }
  auto axes = unitReshapeAxes(operation);
  if (!axes) return false;
  Value input = operation->getOperand(0);
  auto inputType = cast<MemRefType>(input.getType());
  auto resultType = cast<MemRefType>(source.value.getType());
  SmallVector<AffineExpr> indices(inputType.getRank(), getAffineConstantExpr(0, operation->getContext()));
  for (auto [inputAxis, resultAxis] : *axes)
    indices[inputAxis] = getAffineDimExpr(resultAxis, operation->getContext());
  source.map = AffineMap::get(resultType.getRank(), 0, indices, operation->getContext()).compose(source.map);
  source.value = input;
  views.push_back(operation);
  return true;
}

ProjectedSource projectInput(ProjectedSource source, linalg::GenericOp consumer,
                             SmallVectorImpl<Operation *> &views) {
  StorageAnalysis storage(consumer->getParentOfType<func::FuncOp>());
  while (true) {
    if (stripView(source, views)) continue;
    auto projection = bufferProducer(source.value, consumer, false);
    if (!projection || !isProjection(projection)) return source;
    Value input = projection.getInputs()[0];
    if (!storage.readStable(input, projection, consumer)) return source;
    source.map = projection.getIndexingMapsArray().front().compose(source.map);
    source.value = input;
  }
}

struct ProductSource {
  linalg::GenericOp multiply;
  AffineMap map;
};

std::optional<ProductSource> productSource(Value source, linalg::GenericOp consumer,
                                          SmallVectorImpl<Operation *> &views) {
  auto type = dyn_cast<MemRefType>(source.getType());
  if (!type) return std::nullopt;
  ProjectedSource projected{source, AffineMap::getMultiDimIdentityMap(type.getRank(), consumer.getContext())};
  Operation *user = consumer;
  StorageAnalysis storage(consumer->getParentOfType<func::FuncOp>());
  while (true) {
    if (stripView(projected, views)) continue;
    auto producer = bufferProducer(projected.value, user, true);
    if (!producer) return std::nullopt;
    if (!isProjection(producer)) return ProductSource{producer, projected.map};
    Value input = producer.getInputs()[0];
    if (!storage.readStable(input, producer, consumer)) return std::nullopt;
    projected.map = producer.getIndexingMapsArray().front().compose(projected.map);
    projected.value = input;
    user = producer;
  }
}

struct OperandAxes {
  ProjectedSource source;
  SmallVector<std::optional<unsigned>> projection;
  SmallVector<bool> units;
};

std::optional<OperandAxes> operandAxes(ProjectedSource source) {
  auto type = dyn_cast<MemRefType>(source.value.getType());
  if (!type || !type.getElementType().isF32() || source.map.getNumSymbols() ||
      source.map.getNumResults() != static_cast<unsigned>(type.getRank())) return std::nullopt;
  OperandAxes result{source, SmallVector<std::optional<unsigned>>(source.map.getNumDims()), {}};
  for (int64_t size : type.getShape()) result.units.push_back(size == 1);
  for (auto [axis, expression] : llvm::enumerate(source.map.getResults())) {
    if (auto dimension = dyn_cast<AffineDimExpr>(expression)) {
      if (result.projection[dimension.getPosition()]) return std::nullopt;
      result.projection[dimension.getPosition()] = axis;
    } else if (!isa<AffineConstantExpr>(expression) ||
               cast<AffineConstantExpr>(expression).getValue() != 0 || !result.units[axis]) {
      return std::nullopt;
    }
  }
  return result;
}

Value projectMemory(OpBuilder &builder, Location location, Value source, ArrayRef<int64_t> axes) {
  auto type = cast<MemRefType>(source.getType());
  if (axes.size() == static_cast<size_t>(type.getRank()) &&
      llvm::all_of(llvm::enumerate(axes), [](auto entry) { return entry.index() == static_cast<size_t>(entry.value()); }))
    return source;
  SmallVector<int64_t> sourceStrides;
  int64_t sourceOffset;
  [[maybe_unused]] auto status = type.getStridesAndOffset(sourceStrides, sourceOffset);
  assert(succeeded(status));
  auto metadata = builder.create<memref::ExtractStridedMetadataOp>(location, source);
  SmallVector<int64_t> shape, layout;
  SmallVector<OpFoldResult> sizes, strides;
  for (int64_t axis : axes) {
    shape.push_back(type.getDimSize(axis));
    layout.push_back(sourceStrides[axis]);
    sizes.push_back(type.isDynamicDim(axis) ? OpFoldResult(metadata.getSizes()[axis])
                                           : OpFoldResult(builder.getIndexAttr(type.getDimSize(axis))));
    strides.push_back(ShapedType::isDynamic(sourceStrides[axis]) ? OpFoldResult(metadata.getStrides()[axis])
                                                              : OpFoldResult(builder.getIndexAttr(sourceStrides[axis])));
  }
  auto resultType = MemRefType::get(shape, type.getElementType(),
      StridedLayoutAttr::get(builder.getContext(), sourceOffset, layout), type.getMemorySpace());
  OpFoldResult offset = ShapedType::isDynamic(sourceOffset) ? OpFoldResult(metadata.getOffset())
                                                         : OpFoldResult(builder.getIndexAttr(sourceOffset));
  return builder.create<memref::ReinterpretCastOp>(location, resultType, metadata.getBaseBuffer(), offset, sizes, strides);
}

SmallVector<AffineMap> contractionMaps(const ContractionAxes &axes, MLIRContext *context) {
  unsigned rank = axes.results.size() + axes.reduction.size();
  SmallVector<AffineExpr> lhs(axes.lhsResultAxes.size()), rhs(axes.rhsResultAxes.size()), output;
  for (auto [axis, position] : llvm::enumerate(axes.lhsResultAxes))
    if (position) lhs[axis] = getAffineDimExpr(*position, context);
  for (auto [axis, position] : llvm::enumerate(axes.rhsResultAxes))
    if (position) rhs[axis] = getAffineDimExpr(*position, context);
  for (auto [number, pair] : llvm::enumerate(axes.reduction)) {
    auto dimension = getAffineDimExpr(axes.results.size() + number, context);
    lhs[pair.lhs] = dimension;
    rhs[pair.rhs] = dimension;
  }
  for (unsigned axis = 0; axis < axes.results.size(); ++axis) output.push_back(getAffineDimExpr(axis, context));
  return {AffineMap::get(rank, 0, lhs, context), AffineMap::get(rank, 0, rhs, context),
          AffineMap::get(rank, 0, output, context)};
}

bool fuseProduct(linalg::GenericOp reduction) {
  auto order = reduction->getAttrOfType<ReductionOrderAttr>("intent_cpu.reduction_order");
  if (reduction->hasAttr("intent_cpu.implementation") || !order ||
      !order.getAdjacentReassociation() || !order.getElementPermutation() ||
      reduction.getNumResults() || reduction.getInputs().size() != 1 || reduction.getOutputs().size() != 1 ||
      !cast<MemRefType>(reduction.getOutputs()[0].getType()).getElementType().isF32()) return false;
  Block &body = reduction.getRegion().front();
  auto add = body.getTerminator()->getOperand(0).getDefiningOp<arith::AddFOp>();
  if (!add || std::distance(body.begin(), body.end()) != 2 ||
      !((add.getLhs() == body.getArgument(0) && add.getRhs() == body.getArgument(1)) ||
        (add.getRhs() == body.getArgument(0) && add.getLhs() == body.getArgument(1)))) return false;
  auto initialization = findContractionInitialization(reduction);
  if (!initialization) return false;
  auto inputMap = reduction.getIndexingMapsArray().front();
  if (!inputMap.isPermutation()) return false;
  SmallVector<unsigned> loopToProduct(inputMap.getNumDims());
  for (auto [axis, expression] : llvm::enumerate(inputMap.getResults()))
    loopToProduct[cast<AffineDimExpr>(expression).getPosition()] = axis;
  SmallVector<int64_t> reduced, outputs;
  for (auto [axis, iterator] : llvm::enumerate(reduction.getIteratorTypesArray()))
    if (iterator == utils::IteratorType::reduction) reduced.push_back(loopToProduct[axis]);
  auto outputMap = reduction.getIndexingMapsArray().back();
  if (!outputMap.isProjectedPermutation()) return false;
  for (AffineExpr expression : outputMap.getResults()) outputs.push_back(loopToProduct[cast<AffineDimExpr>(expression).getPosition()]);
  if (outputs.size() + reduced.size() != inputMap.getNumDims()) return false;
  for (int64_t axis : outputs) if (llvm::is_contained(reduced, axis)) return false;
  SmallVector<Operation *> views;
  auto product = productSource(reduction.getInputs()[0], reduction, views);
  if (!product) return false;
  auto multiply = product->multiply;
  Block &productBody = multiply.getRegion().front();
  auto mul = productBody.getTerminator()->getOperand(0).getDefiningOp<arith::MulFOp>();
  if (!mul || !mul.getType().isF32() || std::distance(productBody.begin(), productBody.end()) != 2)
    return false;
  SmallVector<unsigned, 2> inputs;
  for (Value value : mul->getOperands()) {
    auto argument = dyn_cast<BlockArgument>(value);
    if (!argument || argument.getOwner() != &productBody ||
        argument.getArgNumber() >= multiply.getInputs().size()) return false;
    inputs.push_back(argument.getArgNumber());
  }
  // A repeated formal is a square, not a distinct input or a new broadcast.
  // Keep every input that determines the original product iteration domain.
  if (multiply.getInputs().size() != (inputs[0] == inputs[1] ? 1u : 2u)) return false;
  SmallVector<OperandAxes> operands;
  StorageAnalysis storage(reduction->getParentOfType<func::FuncOp>());
  for (unsigned number : inputs) {
    ProjectedSource source{multiply.getInputs()[number], multiply.getIndexingMapsArray()[number].compose(product->map)};
    if (!storage.readStable(source.value, multiply, reduction)) return false;
    source = projectInput(source, reduction, views);
    auto axes = operandAxes(source);
    if (!axes) return false;
    operands.push_back(std::move(*axes));
  }
  // The same ordinary product-sum permission also applies without matrix
  // reuse. Preserve its reduction/producer structure for partial formation and
  // SIMD; the target may contract the cloned, same-dtype scalar/vector pair.
  bool changed = (mul.getFastmath() & arith::FastMathFlags::contract) == arith::FastMathFlags::none ||
      (add.getFastmath() & arith::FastMathFlags::contract) == arith::FastMathFlags::none;
  mul.setFastmath(mul.getFastmath() | arith::FastMathFlags::contract);
  add.setFastmath(add.getFastmath() | arith::FastMathFlags::contract);
  auto relate = [&](bool swap) {
    auto &lhs = operands[swap ? 1 : 0], &rhs = operands[swap ? 0 : 1];
    return ProductContractionAxes::get(lhs.projection, lhs.units, rhs.projection, rhs.units, reduced);
  };
  auto relation = relate(false);
  if (!relation || (relation->axes.lhsFree.empty() && relation->axes.rhsFree.empty())) return changed;
  auto contiguousColumns = [&](const ProductContractionAxes &axes, const OperandAxes &rhs) {
    if (axes.axes.rhsFree.empty()) return false;
    unsigned axis = axes.rhsKept[axes.axes.rhsFree.back()];
    SmallVector<int64_t> strides;
    int64_t offset;
    if (failed(cast<MemRefType>(rhs.source.value.getType()).getStridesAndOffset(strides, offset))) return false;
    return strides[axis] == 1;
  };
  auto swapped = relate(true);
  if (swapped && contiguousColumns(*swapped, operands[0]) && !contiguousColumns(*relation, operands[1])) {
    relation = std::move(swapped);
    std::swap(operands[0], operands[1]);
  }
  // With only one free side there is no reuse across independent matrix rows.
  // Keep strided-column products in the producer-fused reduction path instead
  // of materializing and packing a matrix solely to feed a vector-matrix tile.
  if ((relation->axes.lhsFree.empty() || relation->axes.rhsFree.empty()) &&
      !contiguousColumns(*relation, operands[1])) return changed;
  SmallVector<int64_t> outputAxes;
  for (int64_t productAxis : relation->resultProductAxes) {
    auto position = llvm::find(outputs, productAxis);
    if (position == outputs.end()) return changed;
    outputAxes.push_back(position - outputs.begin());
  }
  for (Value memory : {operands[0].source.value, operands[1].source.value, reduction.getOutputs()[0]}) {
    SmallVector<int64_t> strides;
    int64_t offset;
    if (failed(cast<MemRefType>(memory.getType()).getStridesAndOffset(strides, offset))) return changed;
  }
  OpBuilder builder(reduction);
  auto location = reduction.getLoc();
  Value lhs = projectMemory(builder, location, operands[0].source.value, relation->lhsKept);
  Value rhs = projectMemory(builder, location, operands[1].source.value, relation->rhsKept);
  Value output = projectMemory(builder, location, reduction.getOutputs()[0], outputAxes);
  builder.create<linalg::FillOp>(location, ValueRange{initialization->value}, ValueRange{output});
  SmallVector<utils::IteratorType> iterators(relation->axes.results.size(), utils::IteratorType::parallel);
  iterators.append(relation->axes.reduction.size(), utils::IteratorType::reduction);
  auto contraction = builder.create<linalg::GenericOp>(location, ValueRange{lhs, rhs}, ValueRange{output},
      contractionMaps(relation->axes, builder.getContext()), iterators,
      [&](OpBuilder &nested, Location loc, ValueRange arguments) {
        Value value = nested.create<math::FmaOp>(loc, arguments[0], arguments[1], arguments[2]);
        nested.create<linalg::YieldOp>(loc, value);
      });
  contraction->setAttr("intent_cpu.reduction_order", order);
  reduction.erase();
  if (initialization->erasable) initialization->operation->erase();
  llvm::SmallPtrSet<Operation *, 8> uniqueViews;
  for (Operation *view : views)
    if (uniqueViews.insert(view).second && view->use_empty()) view->erase();
  return true;
}

} // namespace

// Logical shape projections are snapshots. A descriptor view is legal only if
// the source observed by this consumer still has the producer's value.
Value foldContractionInput(Value input, linalg::GenericOp consumer,
                          const ContractionRequirements *requirements, unsigned operand) {
  auto producer = bufferProducer(input, consumer, false, false);
  if (!producer || !isProjection(producer)) return input;
  auto allocation = input.getDefiningOp<memref::AllocOp>();
  auto map = producer.getIndexingMapsArray().front();
  // Descriptor folds preserve the element set. Real broadcasts stay explicit;
  // multiply-reduction normalization instead removes them through its maps.
  for (unsigned axis = 0; axis < map.getNumDims(); ++axis)
    if (!llvm::is_contained(map.getResults(), getAffineDimExpr(axis, map.getContext())) &&
        allocation.getType().getDimSize(axis) != 1) return input;
  Value source = producer.getInputs()[0];
  auto original = dyn_cast<MemRefType>(source.getType());
  if (!original || original.getElementType() != allocation.getType().getElementType() ||
      original.getMemorySpace() != allocation.getType().getMemorySpace()) return input;
  for (auto [axis, expression] : llvm::enumerate(map.getResults()))
    if (isa<AffineConstantExpr>(expression) && original.getDimSize(axis) != 1) return input;
  StorageAnalysis storage(consumer->getParentOfType<func::FuncOp>());
  if (!storage.readStable(source, producer, consumer)) return input;
  source = foldContractionInput(source, consumer);
  auto type = cast<MemRefType>(source.getType());
  SmallVector<int64_t> sourceStrides;
  int64_t sourceOffset;
  if (failed(type.getStridesAndOffset(sourceStrides, sourceOffset))) return input;
  OpBuilder builder(consumer);
  Location loc = producer.getLoc();
  SmallVector<int64_t> staticStrides(allocation.getType().getRank(), 0);
  for (auto [axis, expression] : llvm::enumerate(map.getResults()))
    if (auto dim = dyn_cast<AffineDimExpr>(expression)) staticStrides[dim.getPosition()] = sourceStrides[axis];
  auto viewType = MemRefType::get(allocation.getType().getShape(), type.getElementType(),
      StridedLayoutAttr::get(builder.getContext(), sourceOffset, staticStrides), type.getMemorySpace());
  if (requirements && !requirements->acceptsInputLayout(operand, viewType)) return input;
  auto metadata = builder.create<memref::ExtractStridedMetadataOp>(loc, source);
  SmallVector<OpFoldResult> sizes, strides(allocation.getType().getRank(), builder.getIndexAttr(0));
  unsigned dynamic = 0;
  for (int64_t extent : allocation.getType().getShape())
    sizes.push_back(ShapedType::isDynamic(extent) ? OpFoldResult(allocation.getDynamicSizes()[dynamic++])
                                               : OpFoldResult(builder.getIndexAttr(extent)));
  for (auto [axis, expression] : llvm::enumerate(map.getResults())) {
    auto dim = dyn_cast<AffineDimExpr>(expression);
    if (!dim) continue;
    strides[dim.getPosition()] = ShapedType::isDynamic(sourceStrides[axis])
        ? OpFoldResult(metadata.getStrides()[axis]) : OpFoldResult(builder.getIndexAttr(sourceStrides[axis]));
  }
  OpFoldResult offset = ShapedType::isDynamic(sourceOffset)
      ? OpFoldResult(metadata.getOffset()) : OpFoldResult(builder.getIndexAttr(sourceOffset));
  return builder.create<memref::ReinterpretCastOp>(loc, viewType, metadata.getBaseBuffer(), offset, sizes, strides);
}

std::optional<ContractionInitialization>
findContractionInitialization(linalg::GenericOp operation) {
  StorageAnalysis storage(operation->getParentOfType<func::FuncOp>());
  auto initial = queryUniformBufferValue(operation.getOutputs()[0], operation, storage);
  if (!initial || initial->definition.operation->getBlock() != operation->getBlock() ||
      !(isa<FloatType>(initial->value.getType()) ? matchPattern(initial->value, m_PosZeroFloat())
                                               : matchPattern(initial->value, m_Zero())))
    return std::nullopt;
  return ContractionInitialization{initial->definition.operation, initial->value,
                                   !initial->definition.observed};
}

LogicalResult normalizeContractionSources(func::FuncOp function) {
  SmallVector<linalg::GenericOp> reductions;
  function.walk([&](linalg::GenericOp operation) { if (operation.getNumReductionLoops()) reductions.push_back(operation); });
  for (auto reduction : reductions) fuseProduct(reduction);
  eraseDeadPrivateBuffers(function);
  return success();
}

} // namespace intent::cpu
