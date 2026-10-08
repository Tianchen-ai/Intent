#include "TaskConversion.h"
#include "Quantization.h"
#include "ScalarValues.h"
#include "Views.h"
#include "Intent/Dialect/Intent/IR/Interface.h"
#include "Intent/Dialect/CPU/Transforms/Structure/Computations.h"
#include "Intent/Dialect/CPU/Transforms/Storage/Storage.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallBitVector.h"

using namespace mlir;
namespace wk = ::weft::kernel;

namespace intent::weft_provider {
using namespace task_detail;

namespace {

std::optional<int64_t> localExtent(
    const ValueBoundsConstraintSet::Variable &extent,
    ArrayRef<std::pair<OpFoldResult, int64_t>> bindings,
    ValueRange shapeValues, OpBuilder &builder, ModuleOp module) {
  if (!builder.getInsertionBlock()) return std::nullopt;
  DominanceInfo dominance(module);
  for (auto [source, id] : llvm::reverse(bindings)) {
    Operation *definition = shapeValues[id - 1].getDefiningOp();
    if (definition && dominance.properlyDominates(definition->getBlock(),
            definition->getIterator(), builder.getInsertionBlock(), builder.getInsertionPoint()) &&
        cpu::haveEqualExtents(extent, ValueBoundsConstraintSet::Variable(source)))
      return -id;
  }
  return std::nullopt;
}

} // namespace

LogicalResult TaskConversion::normalizeComputations() {
  for (;;) {
  SmallVector<Operation *> scalarTraversals;
  cpu::StorageAnalysis memory(sourceFunction);
  sourceFunction.walk([&](Operation *operation) {
    if (!operation->getParentOfType<cpu::TasksOp>() ||
        !isa<linalg::GenericOp, cpu::ReduceOp>(operation)) return;
    if (auto generic = dyn_cast<linalg::GenericOp>(operation)) {
      bool requiresTraversal = generic.getOutputs().size() != 1;
      auto extents = generic.getStaticLoopRanges();
      generic.getRegion().walk([&](linalg::IndexOp index) {
        requiresTraversal |= extents[index.getDim()] <= 0;
      });
      generic.getRegion().walk([&](memref::LoadOp load) {
        if (!llvm::any_of(load.getIndices(), [&](Value index) {
              return index.getParentBlock() == &generic.getRegion().front();
            })) return;
        auto origins = memory.origins(load.getMemref());
        auto task = generic->getParentOfType<cpu::TasksOp>();
        bool local = origins.complete && !origins.values.empty() &&
            llvm::all_of(origins.values, [&](Value origin) {
              auto allocation = origin.getDefiningOp();
              return isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(allocation) &&
                  task->isProperAncestor(allocation);
            });
        // Canonical slices carry scalar coordinates. A shaped index into an
        // already supplied local Value uses Extract; an addressable input must
        // keep its original element-read traversal, without admitting the table.
        requiresTraversal |= !local;
      });
      if (requiresTraversal) {
        scalarTraversals.push_back(operation);
        return;
      }
    }
    for (Value operand : operation->getOperands()) {
      auto type = dyn_cast<MemRefType>(operand.getType());
      if (!type) continue;
      for (unsigned axis = 0; axis < type.getRank(); ++axis)
        if (type.isDynamicDim(axis) && !dimensionExtent(operand, axis)) {
          scalarTraversals.push_back(operation);
          return;
        }
    }
  });
  if (scalarTraversals.empty()) return success();
  // Computed domains and coupled tuple combines need their existing complete
  // traversal. Canonical Weft's scalar-kind Reduce cannot encode a tuple region,
  // and Iota cannot invent a public shape symbol for a task-local extent.
  // Expose the actual consumer reads before materializing their producers.
  // The shared version-aware fusion can then transport a still-structured
  // coordinate producer to those reads instead of allocating its full domain.
  if (failed(cpu::materializeStructuredComputation(scalarTraversals.back(), 1, {})))
    return failure();
  RewritePatternSet patterns(b.getContext());
  memref::populateFoldMemRefAliasOpPatterns(patterns);
  populateAffineToStdConversionPatterns(patterns);
  if (failed(applyPatternsGreedily(sourceFunction, std::move(patterns))))
    return sourceFunction.emitError("Weft scalar descriptor normalization did not converge");
  if (failed(cpu::fuseStructuredComputations(sourceFunction)) ||
      failed(cpu::fuseIntermediateBuffers(sourceFunction))) return failure();
  cpu::reuseMemoryValues(sourceFunction);
  initializeAxes();
  }
}

namespace task_detail {

SmallVector<int64_t> shape(Type type) {
  if (auto value = dyn_cast<wk::ValueType>(type)) return llvm::to_vector(value.getShape().asArrayRef());
  if (auto value = dyn_cast<wk::ViewType>(type)) return llvm::to_vector(value.getShape().asArrayRef());
  if (auto value = dyn_cast<wk::SliceType>(type)) return llvm::to_vector(value.getShape().asArrayRef());
  return {};
}

SmallVector<int64_t> axes(Type type) {
  if (auto value = dyn_cast<wk::ValueType>(type)) return llvm::to_vector(value.getAxisIds().asArrayRef());
  if (auto value = dyn_cast<wk::ViewType>(type)) return llvm::to_vector(value.getAxisIds().asArrayRef());
  if (auto value = dyn_cast<wk::SliceType>(type)) return llvm::to_vector(value.getAxisIds().asArrayRef());
  return {};
}

Type element(Type type) {
  if (auto value = dyn_cast<wk::ValueType>(type)) return value.getElementType();
  return type;
}

} // namespace task_detail

int64_t TaskConversion::physicalAxis(int64_t axis) {
  while (axisProjection.count(axis)) axis = axisProjection.lookup(axis);
  return axis;
}

SmallVector<int64_t> TaskConversion::memoryAxes(Value memory) {
  auto result = relations.axes(memory);
  for (int64_t &axis : result) axis = physicalAxis(axis);
  return result;
}

DenseI64ArrayAttr TaskConversion::array(ArrayRef<int64_t> entries) { return b.getDenseI64ArrayAttr(entries); }

wk::EncodingType TaskConversion::dense(Type type) {
  std::string name;
  if (type.isIndex()) name = "i64";
  else if (auto integer = dyn_cast<IntegerType>(type))
    name = (integer.isUnsigned() ? "u" : "i") + std::to_string(integer.getWidth());
  else { llvm::raw_string_ostream stream(name); type.print(stream); }
  return wk::EncodingType::get(b.getContext(), name, "dense", "dense." + name, array({}));
}

wk::EncodingType TaskConversion::encoding(Value memory) {
  auto format = quantizedFormat(memory);
  return format ? quantEncoding(b, *format)
                : dense(cast<MemRefType>(memory.getType()).getElementType());
}

wk::SliceType TaskConversion::sliceType(Value base, ArrayRef<int64_t> dimensions,
                       ArrayRef<int64_t> ids) {
  Type type = base.getType();
  Type encoding = isa<wk::ViewType>(type)
                      ? cast<wk::ViewType>(type).getEncoding()
                      : cast<wk::SliceType>(type).getEncoding();
  return wk::SliceType::get(b.getContext(), encoding, array(dimensions),
                           array(ids));
}

std::optional<intent::QuantFormat> TaskConversion::quantizedFormat(Value memory) {
  for (Value origin : storage->origins(memory).values)
    if (auto found = formats.find(origin); found != formats.end())
      return found->second;
  return std::nullopt;
}

wk::ViewType TaskConversion::scalarView(Type type) {
  return wk::ViewType::get(b.getContext(), dense(type), array({1}), array({nextAxis++}));
}

Type TaskConversion::valueType(Type element, ArrayRef<int64_t> dimensions, ArrayRef<int64_t> ids) {
  return nativeValueType(element, dimensions, ids);
}

Value TaskConversion::index(Location loc, int64_t value) { return b.create<arith::ConstantIndexOp>(loc, value); }

FailureOr<Type> TaskConversion::resultType(Value memory, Type scalar) {
  auto type = cast<MemRefType>(memory.getType());
  auto ids = memoryAxes(memory);
  SmallVector<int64_t> dimensions;
  for (auto [axis, extent] : llvm::enumerate(type.getShape())) {
    if (ShapedType::isDynamic(extent) || extent == 0) {
      auto dimension = dimensionExtent(memory, axis);
      if (!dimension) return emitError(memory.getLoc(), "private value extent has no explicit shape binding"), failure();
      dimensions.push_back(*dimension);
    } else dimensions.push_back(extent);
  }
  Type element = scalar ? scalar : type.getElementType();
  if (element.isIndex()) element = IntegerType::get(b.getContext(), 64, IntegerType::Signed);
  return valueType(element, dimensions, ids);
}

bool TaskConversion::sameBound(OpFoldResult lhs, OpFoldResult rhs) {
  if (lhs == rhs) return true;
  auto constant = [](OpFoldResult value) -> std::optional<int64_t> {
    if (auto attribute = dyn_cast<Attribute>(value)) return cast<IntegerAttr>(attribute).getInt();
    llvm::APInt integer;
    if (matchPattern(cast<Value>(value), m_ConstantInt(&integer))) return integer.getSExtValue();
    return std::nullopt;
  };
  auto a = constant(lhs), c = constant(rhs);
  if (a && c) return *a == *c;
  auto identity = [&](OpFoldResult bound) -> std::optional<int64_t> {
    if (auto value = dyn_cast<Value>(bound)) {
      auto found = llvm::find(shapeValues, value);
      if (found != shapeValues.end()) return -1 - (found - shapeValues.begin());
    }
    return nativeExtent(bound);
  };
  auto left = identity(lhs), right = identity(rhs);
  return left && right && *left == *right;
}

std::optional<int64_t> TaskConversion::nativeExtent(const cpu::ExtentExpression &extent) {
  AffineExpr expression = extent.expression.getResult(0);
  if (auto constant = dyn_cast<AffineConstantExpr>(expression))
    return constant.getValue() >= 0
        ? std::optional<int64_t>(constant.getValue()) : std::nullopt;

  // Weft shape entries are constants or references to the actual public shape
  // arguments. Proving an affine expression is not permission to invent a
  // new runtime symbol for it.
  unsigned position;
  if (auto dimension = dyn_cast<AffineDimExpr>(expression))
    position = dimension.getPosition();
  else if (auto symbol = dyn_cast<AffineSymbolExpr>(expression))
    position = extent.expression.getNumDims() + symbol.getPosition();
  else return std::nullopt;
  auto [value, axis] = extent.operands[position];
  auto argument = dyn_cast<BlockArgument>(value);
  auto function = argument
      ? dyn_cast<func::FuncOp>(argument.getOwner()->getParentOp()) : func::FuncOp{};
  if (function != sourceFunction || !axis) return std::nullopt;
  auto publicView = getPublicView(getPublicInterface(function), argument.getArgNumber());
  if (!publicView) return std::nullopt;
  auto symbol = extentIds.find(publicViewDimensions(publicView)[*axis]);
  return symbol == extentIds.end()
      ? std::nullopt : std::optional<int64_t>(-symbol->second);
}

std::optional<int64_t> TaskConversion::nativeExtent(OpFoldResult extent) {
  auto expression = cpu::queryExtent(extent);
  auto result = succeeded(expression) ? nativeExtent(*expression) : std::nullopt;
  if (result && *result != 0) return result;
  if (auto local = localExtent(ValueBoundsConstraintSet::Variable(extent),
          localShapeBindings, shapeValues, b, output)) return local;
  return result;
}

std::optional<int64_t> TaskConversion::dimensionExtent(Value memory, unsigned axis) {
  if (auto symbol = capturedShapeSymbol(ValueBoundsConstraintSet::Variable(memory, axis)))
    return -*symbol;
  auto expression = cpu::queryExtent(ValueBoundsConstraintSet::Variable(memory, axis));
  auto result = succeeded(expression) ? nativeExtent(*expression) : std::nullopt;
  if (result && *result != 0) return result;
  if (auto local = localExtent(ValueBoundsConstraintSet::Variable(memory, axis),
          localShapeBindings, shapeValues, b, output)) return local;
  return result;
}

std::optional<int64_t>
TaskConversion::shapeSymbol(const ValueBoundsConstraintSet::Variable &extent) {
  if (auto symbol = capturedShapeSymbol(extent)) return symbol;
  if (auto expression = cpu::queryExtent(extent); succeeded(expression))
    if (auto native = nativeExtent(*expression); native && *native < 0)
      return -*native;
  // A dynamic host descriptor still needs an actual shape parameter even
  // when equality analysis proves its extent constant. Select only among the
  // existing public bindings; never synthesize a symbol for an expression.
  auto interface = getPublicInterface(sourceFunction);
  for (BlockArgument argument : sourceFunction.getArguments()) {
    auto memory = dyn_cast<MemRefType>(argument.getType());
    if (!memory) continue;
    auto ids = publicViewDimensions(getPublicView(interface, argument.getArgNumber()));
    for (unsigned axis = 0; axis < static_cast<unsigned>(memory.getRank()); ++axis)
      if (memory.isDynamicDim(axis) && cpu::haveEqualExtents(
              extent, ValueBoundsConstraintSet::Variable(argument, axis)))
        return extentIds.at(ids[axis]);
  }
  return std::nullopt;
}

FailureOr<Value> TaskConversion::view(Value memory) {
  if (values.contains(memory)) {
    if (!viewAxes.count(memory)) {
      auto &mapping = viewAxes[memory];
      for (int64_t axis = 0; axis < cast<MemRefType>(memory.getType()).getRank(); ++axis) mapping.push_back(axis);
    }
    return values.lookup(memory);
  }
  if (auto cast = memory.getDefiningOp<memref::CastOp>()) {
    auto source = view(cast.getSource());
    if (failed(source)) return failure();
    viewAxes[memory] = viewAxes.at(cast.getSource());
    values.map(memory, *source);
    return source;
  }
  if (memory.getDefiningOp() && isAxisView(memory.getDefiningOp())) {
    auto projection = queryAxisView(memory);
    if (failed(projection)) return failure();
    auto source = view(projection->source);
    if (failed(source)) return failure();
    auto &mapping = viewAxes[memory];
    for (auto axis : projection->sourceAxes)
      mapping.push_back(axis ? viewAxes.at(projection->source)[*axis] : -1);
    values.map(memory, *source);
    return source;
  }
  auto operation = memory.getDefiningOp<memref::SubViewOp>();
  if (!operation) return emitError(memory.getLoc(), "CPU memory has no explicit Weft view relation: ") << memory, failure();
  auto base = view(operation.getSource());
  if (failed(base)) return failure();
  auto sourceAxes = viewAxes.at(operation.getSource());
  auto baseShape = shape((*base).getType());
  SmallVector<OpFoldResult> nativeOffsets(baseShape.size(), b.getIndexAttr(0));
  SmallVector<OpFoldResult> nativeSizes(baseShape.size(), b.getIndexAttr(1));
  llvm::SmallBitVector dropped(baseShape.size());
  auto logicalDropped = operation.getDroppedDims();
  for (auto [axis, native] : llvm::enumerate(sourceAxes)) {
    if (native < 0) {
      if (!sameBound(operation.getMixedOffsets()[axis], b.getIndexAttr(0)) ||
          !sameBound(operation.getMixedSizes()[axis], b.getIndexAttr(1)))
        return operation.emitError("Weft inserted unit-axis subview must retain its single element"), failure();
      continue;
    }
    nativeOffsets[native] = operation.getMixedOffsets()[axis];
    nativeSizes[native] = operation.getMixedSizes()[axis];
    if (logicalDropped.test(axis)) dropped.set(native);
  }
  SmallVector<int64_t> mapping;
  for (auto [axis, native] : llvm::enumerate(sourceAxes))
    if (!logicalDropped.test(axis)) {
      int64_t shifted = native;
      for (int64_t before = 0; before < native; ++before) shifted -= dropped.test(before);
      mapping.push_back(shifted);
    }
  SmallVector<int64_t> offsets, extents, dimensions;
  SmallVector<Value> dynamic;
  auto append = [&](ArrayRef<OpFoldResult> bounds, SmallVectorImpl<int64_t> &statics) {
    for (OpFoldResult value : bounds) {
      if (auto attr = dyn_cast<Attribute>(value)) statics.push_back(cast<IntegerAttr>(attr).getInt());
      else {
        llvm::APInt integer;
        if (matchPattern(cast<Value>(value), m_ConstantInt(&integer))) statics.push_back(integer.getSExtValue());
        else { statics.push_back(-1); dynamic.push_back(values.lookup(cast<Value>(value))); }
      }
    }
  };
  for (OpFoldResult stride : operation.getMixedStrides())
    if (!sameBound(stride, b.getIndexAttr(1))) return operation.emitError("Weft subview requires unit coordinate steps"), failure();
  bool projectionOnly = true;
  for (unsigned axis = 0; axis < baseShape.size(); ++axis) {
    if (dropped.test(axis)) continue;
    auto size = nativeSizes[axis];
    bool full = nativeExtent(size) == baseShape[axis];
    projectionOnly &= sameBound(nativeOffsets[axis], b.getIndexAttr(0)) && full;
  }
  if (projectionOnly) {
    SmallVector<Attribute> selectors;
    SmallVector<Value> indices;
    SmallVector<int64_t> keptShape, keptAxes;
    auto baseAxes = axes((*base).getType());
    for (unsigned axis = 0; axis < baseShape.size(); ++axis) {
      selectors.push_back(b.getStringAttr(dropped.test(axis) ? "index" : "all"));
      if (dropped.test(axis)) {
        auto offset = nativeOffsets[axis];
        indices.push_back(isa<Attribute>(offset) ? index(operation.getLoc(), cast<IntegerAttr>(cast<Attribute>(offset)).getInt())
                                                 : values.lookup(cast<Value>(offset)));
      } else { keptShape.push_back(baseShape[axis]); keptAxes.push_back(baseAxes[axis]); }
    }
    Value selected = b.create<wk::SliceOp>(operation.getLoc(),
        sliceType(*base, keptShape, keptAxes),
        *base, indices, b.getArrayAttr(selectors));
    values.map(memory, selected);
    viewAxes[memory] = std::move(mapping);
    return selected;
  }
  append(nativeOffsets, offsets);
  for (OpFoldResult size : nativeSizes) {
    auto extent = nativeExtent(size);
    if (!extent) return operation.emitError("dynamic Weft subview extent has no shape binding"), failure();
    dimensions.push_back(*extent);
    if (*extent >= 0) extents.push_back(*extent);
    else {
      extents.push_back(-1);
      dynamic.push_back(values.lookup(cast<Value>(size)));
    }
  }
  auto format = quantizedFormat(memory);
  if (format) {
    int64_t bytes = *format == intent::QuantFormat::Q4K ? 144 : 292;
    if (offsets.back() != 0 || extents.back() != bytes)
      return operation.emitError("encoded view projection must retain its complete record bytes"), failure();
    extents.back() = dimensions.back() = 256;
  }
  auto ids = axes((*base).getType());
  Value selected = b.create<wk::SubviewOp>(operation.getLoc(),
      sliceType(*base, dimensions, ids),
      *base, dynamic, array(offsets), array(extents));
  if (dropped.any()) {
    SmallVector<Attribute> selectors;
    SmallVector<Value> indices;
    SmallVector<int64_t> keptShape, keptAxes;
    for (unsigned axis = 0; axis < ids.size(); ++axis) {
      if (dropped.test(axis)) {
        selectors.push_back(b.getStringAttr("index"));
        indices.push_back(index(operation.getLoc(), 0));
      } else {
        selectors.push_back(b.getStringAttr("all"));
        keptShape.push_back(dimensions[axis]); keptAxes.push_back(ids[axis]);
      }
    }
    selected = b.create<wk::SliceOp>(operation.getLoc(),
        sliceType(selected, keptShape, keptAxes),
        selected, indices, b.getArrayAttr(selectors));
  }
  values.map(memory, selected);
  viewAxes[memory] = std::move(mapping);
  return selected;
}

FailureOr<Value> TaskConversion::reshapeViewValue(Value memory, Value value, Type target, SmallVector<int64_t> order) {
  auto sourceAxes = axes(value.getType());
  for (int64_t axis : sourceAxes) if (!llvm::is_contained(order, axis)) order.push_back(axis);
  SmallVector<int64_t> sourceDomain, reorderedDomain;
  auto sourceShape = shape(value.getType());
  for (auto [axis, extent] : llvm::zip(sourceAxes, sourceShape))
    if (extent != 1) sourceDomain.push_back(axis);
  for (int64_t axis : order)
    if (sourceShape[llvm::find(sourceAxes, axis) - sourceAxes.begin()] != 1) reorderedDomain.push_back(axis);
  // Moving a unit axis does not change the linear element sequence. Keep
  // the original order so the target need not realize a physical transpose.
  if (sourceDomain == reorderedDomain) order = sourceAxes;
  if (value.getType() == target && order == sourceAxes) return value;
  if (!isa<wk::ValueType>(value.getType()))
    return Value(b.create<wk::NewOp>(memory.getLoc(), target, value, true));
  if (!isa<wk::ValueType>(target)) {
    if (llvm::any_of(shape(value.getType()), [](int64_t size) { return size != 1; }))
      return emitError(memory.getLoc(), "Weft scalar view projection must contain one element"), failure();
    return Value(b.create<wk::ExtractOp>(memory.getLoc(), target, value,
        SmallVector<Value>(sourceAxes.size(), index(memory.getLoc(), 0)),
        b.getArrayAttr(SmallVector<Attribute>(sourceAxes.size(), b.getStringAttr("index")))));
  }
  bool dynamic = llvm::any_of(shape(value.getType()), [](int64_t size) { return size < 0; }) ||
                 llvm::any_of(shape(target), [](int64_t size) { return size < 0; });
  if (dynamic && (shape(value.getType()) != shape(target) || order != sourceAxes)) {
    auto targetShape = shape(target), targetAxes = axes(target);
    SmallVector<unsigned> sourceMembers, targetMembers;
    for (auto [position, extent] : llvm::enumerate(sourceShape))
      if (extent != 1) sourceMembers.push_back(position);
    for (auto [position, extent] : llvm::enumerate(targetShape))
      if (extent != 1) targetMembers.push_back(position);
    if (order != sourceAxes || sourceMembers.size() != targetMembers.size())
      return emitError(memory.getLoc(), "dynamic Weft view must preserve its ordered non-unit dimensions"), failure();
    // Keep axis renaming separate from the existing unit-axis layout
    // projection. The latter preserves every non-unit (axis, shape) pair.
    SmallVector<int64_t> renamedAxes(sourceAxes.size());
    for (unsigned position = 0; position < sourceShape.size(); ++position)
      if (sourceShape[position] == 1) renamedAxes[position] = nextAxis++;
    for (auto [sourcePosition, targetPosition] : llvm::zip(sourceMembers, targetMembers))
      renamedAxes[sourcePosition] = targetAxes[targetPosition];
    if (!wk::isUnitAxisProjection(sourceShape, renamedAxes, targetShape,
                                  targetAxes, renamedAxes))
      return emitError(memory.getLoc(), "dynamic Weft view changes a non-unit dimension"), failure();
    Type renamed = valueType(element(value.getType()), sourceShape, renamedAxes);
    if (value.getType() != renamed)
      value = b.create<wk::ReshapeOp>(memory.getLoc(), renamed, value, array(sourceAxes));
    order = std::move(renamedAxes);
  }
  return Value(b.create<wk::ReshapeOp>(memory.getLoc(), target, value, array(order)));
}

FailureOr<Value> TaskConversion::projectViewValue(Value memory, Value value, Type nativeType, bool inverse) {
  auto logicalType = resultType(memory);
  if (failed(logicalType)) return failure();
  auto nativeAxes = axes(nativeType), logicalAxes = axes(*logicalType);
  auto &mapping = viewAxes.at(memory);
  SmallVector<int64_t> order;
  if (inverse) {
    for (unsigned native = 0; native < nativeAxes.size(); ++native)
      for (auto [logical, position] : llvm::enumerate(mapping))
        if (position == static_cast<int64_t>(native)) order.push_back(logicalAxes[logical]);
  } else {
    for (int64_t native : mapping) if (native >= 0) order.push_back(nativeAxes[native]);
  }
  return reshapeViewValue(memory, value, inverse ? nativeType : *logicalType, std::move(order));
}

SmallVector<Value> TaskConversion::projectIndices(Value memory, ValueRange indices,
    unsigned nativeRank, const IRMapping &mapping) {
  SmallVector<Value> projected(nativeRank, index(memory.getLoc(), 0));
  for (auto [logical, native] : llvm::enumerate(viewAxes.at(memory)))
    if (native >= 0) projected[native] = mapping.lookup(indices[logical]);
  return projected;
}

FailureOr<Value> TaskConversion::dimension(Value memory, unsigned axis) {
  auto type = cast<MemRefType>(memory.getType());
  if (!type.isDynamicDim(axis)) return index(memory.getLoc(), type.getDimSize(axis));
  if (auto found = localReferences.find(memory); found != localReferences.end())
    return dimension(found->second, axis);
  if (controlOwners.contains(memory)) {
    auto found = locals.find(memory);
    if (found == locals.end()) return emitError(memory.getLoc(), "private control value has no dominating supply"), failure();
    OpFoldResult size = found->second.sizes[axis];
    return isa<Attribute>(size) ? index(memory.getLoc(), cast<IntegerAttr>(cast<Attribute>(size)).getInt())
                                : cast<Value>(size);
  }
  if (auto cast = memory.getDefiningOp<memref::CastOp>()) return dimension(cast.getSource(), axis);
  if (isAxisView(memory.getDefiningOp())) {
    auto projection = queryAxisView(memory);
    if (failed(projection)) return failure();
    return projection->sourceAxes[axis] ? dimension(projection->source, *projection->sourceAxes[axis])
                                        : FailureOr<Value>(index(memory.getLoc(), 1));
  }
  if (isLocal(memory)) {
    OpFoldResult size;
    if (auto projection = memory.getDefiningOp<memref::SubViewOp>()) {
      unsigned retained = 0;
      for (unsigned original = 0; original < projection.getSourceType().getRank(); ++original)
        if (!projection.getDroppedDims().test(original) && retained++ == axis)
          size = projection.getMixedSizes()[original];
    } else if (isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(memory.getDefiningOp())) {
      size = memory.getDefiningOp()->getOperand(type.getDynamicDimIndex(axis));
    }
    if (!size) return emitError(memory.getLoc(), "private view dimension has no explicit extent"), failure();
    return isa<Attribute>(size) ? index(memory.getLoc(), cast<IntegerAttr>(cast<Attribute>(size)).getInt())
                                : values.lookup(cast<Value>(size));
  }
  auto region = view(memory);
  if (failed(region)) return failure();
  int64_t native = viewAxes.at(memory)[axis];
  if (native < 0) return index(memory.getLoc(), 1);
  return Value(b.create<wk::ExtentOp>(memory.getLoc(), b.getIndexType(), *region, native));
}

LogicalResult TaskConversion::lower(memref::ExtractStridedMetadataOp metadata) {
  SmallVector<Value> descriptors{metadata.getBaseBuffer(), metadata.getOffset()};
  llvm::append_range(descriptors, metadata.getStrides());
  for (Value descriptor : descriptors)
    for (Operation *user : descriptor.getUsers()) {
      if (descriptor == metadata.getBaseBuffer() &&
          isa<memref::DeallocOp, bufferization::DeallocOp>(user)) continue;
      if (!isa<memref::ReinterpretCastOp>(user))
        return metadata.emitError("Weft strided metadata may only supply a proved axis view; arbitrary address arithmetic is unsupported");
    }
  for (auto [axis, size] : llvm::enumerate(metadata.getSizes())) {
    if (size.use_empty()) continue;
    auto extent = dimension(metadata.getSource(), axis);
    if (failed(extent)) return failure();
    values.map(size, *extent);
  }
  return success();
}

LogicalResult TaskConversion::lower(memref::DimOp dim) {
  auto axis = dim.getConstantIndex();
  if (!axis) return dim.emitError("Weft dimension requires a static axis position");
  auto extent = dimension(dim.getSource(), *axis);
  if (failed(extent)) return failure();
  values.map(dim.getResult(), *extent);
  return success();
}

} // namespace intent::weft_provider
