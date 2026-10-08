#include "Intent/Dialect/GPU/IR/ProgramInterface.h"
#include "Pointwise.h"
#include "Intent/Dialect/GPU/Analysis/Resources.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Transforms/Contraction/Contraction.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <algorithm>

using namespace mlir;

namespace intent::gpu::pointwise {

uint32_t physicalElementBitWidth(Type type) {
  if (auto fragment = dyn_cast<FragmentType>(type))
    type = fragment.getElementType();
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    type = tensor.getElementType();
  if (auto record = dyn_cast<RecordType>(type)) {
    uint32_t width = 0;
    for (Attribute field : record.getFieldTypes())
      width = std::max(
          width,
          physicalElementBitWidth(cast<TypeAttr>(field).getValue()));
    return width;
  }
  return isa<IntegerType, FloatType>(type)
             ? type.getIntOrFloatBitWidth()
             : 0;
}

void collectPhysicalDimensions(Type type,
                               llvm::SmallDenseSet<uint64_t> &dimensions) {
  if (auto fragment = dyn_cast<FragmentType>(type)) {
    for (Attribute attribute : fragment.getAxisMaps()) {
      auto mapping = cast<AxisMapAttr>(attribute);
      if (mapping.getDimensionId() > 0)
        dimensions.insert(mapping.getDimensionId());
    }
    return;
  }
  if (auto record = dyn_cast<RecordType>(type))
    for (Attribute field : record.getFieldTypes())
      collectPhysicalDimensions(cast<TypeAttr>(field).getValue(), dimensions);
}

void collectPartiallyCarriedDimensions(
    Type type, llvm::SmallDenseSet<uint64_t> &dimensions) {
  auto record = dyn_cast<RecordType>(type);
  if (!record)
    return;

  llvm::DenseMap<uint64_t, unsigned> fieldCounts;
  for (Attribute field : record.getFieldTypes()) {
    Type fieldType = cast<TypeAttr>(field).getValue();
    llvm::SmallDenseSet<uint64_t> fieldDimensions;
    collectPhysicalDimensions(fieldType, fieldDimensions);
    for (uint64_t dimension : fieldDimensions)
      ++fieldCounts[dimension];
    collectPartiallyCarriedDimensions(fieldType, dimensions);
  }
  for (auto [dimension, count] : fieldCounts)
    if (count < record.getFieldTypes().size())
      dimensions.insert(dimension);
}

Attribute dimensionAxisKey(MLIRContext *context, uint64_t dimension) {
  return IntegerAttr::get(IntegerType::get(context, 64), dimension);
}

Attribute sourceAxisKey(MLIRContext *context, PhysicalSourceAxis source) {
  return PhysicalSourceAttr::get(context, source.sourceId, source.sourceAxis,
                                 source.derived);
}

bool isSourceAxisKey(Attribute axis) { return isa<PhysicalSourceAttr>(axis); }

FailureOr<uint64_t> axisDimension(Attribute axis) {
  auto dimension = dyn_cast<IntegerAttr>(axis);
  return dimension && dimension.getInt() > 0
             ? FailureOr<uint64_t>(dimension.getInt())
             : FailureOr<uint64_t>(failure());
}

FailureOr<PhysicalSourceAxis> axisSource(Attribute axis) {
  auto source = dyn_cast<PhysicalSourceAttr>(axis);
  return source ? FailureOr<PhysicalSourceAxis>(PhysicalSourceAxis{
                      source.getSourceId(), source.getSourceAxis(),
                      source.getDerived()})
                : FailureOr<PhysicalSourceAxis>(failure());
}

PhysicalExprAttr expression(MLIRContext *context, PhysicalExprKind kind,
                            int64_t value, StringRef symbol) {
  return PhysicalExprAttr::get(context, kind, value,
                               kind == PhysicalExprKind::Parameter
                                   ? Attribute(ParameterRefAttr::get(context, StringAttr::get(context, symbol)))
                                   : Attribute(StringAttr::get(context, symbol)),
                               ArrayAttr::get(context, {}));
}

PhysicalExprAttr binaryExpression(MLIRContext *context, PhysicalExprKind kind,
                                  PhysicalExprAttr lhs,
                                  PhysicalExprAttr rhs) {
  return PhysicalExprAttr::get(
      context, kind, 0, StringAttr::get(context),
      ArrayAttr::get(context, {lhs, rhs}));
}

bool hasCompileTimeExtent(Value value) {
  return value.getDefiningOp<arith::ConstantOp>() ||
         value.getDefiningOp<ParameterOp>() ||
         value.getDefiningOp<PhysicalExprOp>();
}

FailureOr<Value> dimensionArgument(func::FuncOp kernel, uint64_t dimension) {
  if (auto argument = resolveDimension(kernel, dimension)) return Value(argument);
  return failure();
}

FailureOr<uint64_t> rangeDimension(MakeRangeOp range) {
  FailureOr<int64_t> dimension = querySourceDimension(
      range.getResult().getType(),
      PhysicalSourceAxis{range.getSourceId(), range.getSourceAxis(),
                           range.getDerived()});
  return succeeded(dimension) && *dimension > 0
             ? FailureOr<uint64_t>(*dimension)
             : FailureOr<uint64_t>(failure());
}

bool hasAccessDependentSubregionBounds(func::FuncOp kernel, MakeRangeOp range) {
  if (!range->hasAttr(sourceSubregionAttr))
    return false;
  PhysicalProgramAnalysis analysis(kernel);
  for (Value bound : {range.getLogicalStart(), range.getLogicalStop()}) {
    PhysicalRangeFact provenance = analysis.sourceRanges(bound);
    if (!provenance.accesses.empty())
      return true;
  }
  return false;
}

FailureOr<uint64_t> ownershipDimension(func::FuncOp kernel,
                                       MakeRangeOp range) {
  FailureOr<int64_t> parent = querySubregionParentDimension(range);
  if (succeeded(parent) && !hasAccessDependentSubregionBounds(kernel, range))
    return static_cast<uint64_t>(*parent);
  return rangeDimension(range);
}

FailureOr<ParameterAttr> queryOwnershipBlockingParameter(func::FuncOp kernel,
                                                       MakeRangeOp range) {
  FailureOr<ParameterAttr> direct = queryBlockingParameter(kernel, range);
  if (succeeded(direct))
    return *direct;
  FailureOr<int64_t> parent = querySubregionParentDimension(range);
  if (failed(parent))
    return failure();
  ParameterAttr result;
  bool ambiguous = false;
  for (Attribute declaration : getParameterDeclarations(kernel)) {
    auto parameter = cast<ParameterAttr>(declaration);
    PhysicalParameterBinding binding = queryParameterBinding(parameter);
    if (!binding.isExact() || !binding.dimension ||
        *binding.dimension != *parent ||
        parameter.getRole() !=
            ParameterRole::OwnershipN)
      continue;
    if (result && result != parameter)
      ambiguous = true;
    else
      result = parameter;
  }
  return result && !ambiguous ? FailureOr<ParameterAttr>(result)
                              : FailureOr<ParameterAttr>(failure());
}

FailureOr<Attribute> parameterAxis(ParameterAttr parameter) {
  PhysicalParameterBinding binding = queryParameterBinding(parameter);
  if (!binding.isExact())
    return failure();
  if (binding.source)
    return sourceAxisKey(parameter.getContext(), *binding.source);
  if (binding.dimension)
    return dimensionAxisKey(parameter.getContext(), *binding.dimension);
  return failure();
}

PhysicalExprAttr fragmentExtent(ParameterAttr parameter) {
  ParameterAttr schema = parameter;
  PhysicalParameterBinding binding = queryParameterBinding(parameter);
  if (binding.source && schema.getCandidates().size() == 1)
    return expression(parameter.getContext(), PhysicalExprKind::Constant,
                      schema.getCandidates()[0]);
  return expression(parameter.getContext(), PhysicalExprKind::Parameter, 0,
                    schema.getName().getValue());
}

bool hasExactStaticFullCoverage(func::FuncOp kernel, Value source,
                                uint64_t axis) {
  auto fragment = dyn_cast<FragmentType>(source.getType());
  if (!fragment || axis >= fragment.getShape().size())
    return false;
  auto extent = dyn_cast<PhysicalExprAttr>(fragment.getShape()[axis]);
  if (!extent || extent.getKind() !=
                     PhysicalExprKind::Constant ||
      extent.getValue() <= 0)
    return false;
  PhysicalAxisRealizationFact fact =
      PhysicalProgramAnalysis(kernel).axisRealization(source, axis);
  if (!fact.isExact() || !fact.physicalized || fact.constructionScalarSeed ||
      fact.roots.empty())
    return false;
  return llvm::all_of(fact.roots, [&](MakeRangeOp range) {
    if (!samePhysicalScalarExpression(range.getStart(),
                                      range.getLogicalStart()))
      return false;
    auto start = range.getLogicalStart().getDefiningOp<arith::ConstantIndexOp>();
    auto stop = range.getLogicalStop().getDefiningOp<arith::ConstantIndexOp>();
    auto step = range.getStep().getDefiningOp<arith::ConstantIndexOp>();
    if (!step || step.value() <= 0)
      return false;
    if (start && stop)
      return stop.value() >= start.value() &&
             static_cast<__int128>(extent.getValue()) * step.value() >=
                 static_cast<__int128>(stop.value()) - start.value();
    // A nonnegative dynamic start needs at most the bounded stop's lanes.
    // The original tail predicate still selects the subregion's members.
    PhysicalExprAttr startBound =
        queryNonNegativeIndexUpperBound(range.getLogicalStart());
    PhysicalExprAttr stopBound =
        queryNonNegativeIndexUpperBound(range.getLogicalStop());
    return startBound && stopBound &&
           stopBound.getKind() ==
               PhysicalExprKind::Constant &&
           static_cast<__int128>(extent.getValue()) * step.value() >=
               stopBound.getValue();
  });
}

bool isZeroScanIdentity(Value value) {
  while (true) {
    if (auto splat = value.getDefiningOp<SplatOp>()) {
      value = splat.getValue();
      continue;
    }
    if (auto broadcast = value.getDefiningOp<BroadcastOp>()) {
      value = broadcast.getValue();
      continue;
    }
    if (auto reshape = value.getDefiningOp<ReshapeOp>()) {
      value = reshape.getValue();
      continue;
    }
    if (auto cast = value.getDefiningOp<CastOp>()) {
      value = cast.getValue();
      continue;
    }
    break;
  }
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return false;
  if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
    return integer.getValue().isZero();
  if (auto floating = dyn_cast<FloatAttr>(constant.getValue()))
    return floating.getValue().isZero();
  return false;
}

FailureOr<int64_t>
exactSubregionStaticBound(const PhysicalRangeFact &ranges,
                          PhysicalSourceAxis source) {
  if (!ranges.isExact() || ranges.roots.empty())
    return failure();
  std::optional<int64_t> bound;
  for (MakeRangeOp range : ranges.roots) {
    auto current = range->getAttrOfType<IntegerAttr>(sourceSubregionBoundAttr);
    if (!range->hasAttr(sourceSubregionAttr) || !current ||
        current.getInt() <= 0 || !isUnitStepRange(range) ||
        !(sourceAxisIdentity(range) == source) ||
        (bound && *bound != current.getInt()))
      return failure();
    bound = current.getInt();
  }
  return *bound;
}

FailureOr<int64_t>
exactStaticTraversalExtent(const PhysicalRangeFact &ranges) {
  if (!ranges.isExact() || ranges.roots.empty() ||
      failed(queryExactLogicalRange(ranges)))
    return failure();
  std::optional<int64_t> extent;
  for (MakeRangeOp range : ranges.roots) {
    auto current = constantLogicalRangeCardinality(range);
    if (!current || (extent && *extent != *current))
      return failure();
    extent = *current;
  }
  return extent ? FailureOr<int64_t>(*extent)
                : FailureOr<int64_t>(failure());
}

bool hasFullRangeReductionCapture(Value value, MakeRangeOp range,
                                  Operation *anchor) {
  auto kernel = anchor->getParentOfType<func::FuncOp>();
  if (!samePhysicalScalarExpression(range.getStart(), range.getLogicalStart()))
    return false;
  PhysicalSourceAxis source = sourceAxisIdentity(range);
  FailureOr<int64_t> dimension = queryRangeDimension(range);
  if (!kernel || failed(dimension))
    return false;
  DominanceInfo dominance(kernel);
  PhysicalProgramAnalysis analysis(kernel);
  SmallVector<Value> pending{value};
  llvm::DenseSet<Value> visited;
  while (!pending.empty()) {
    Value current = pending.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    Operation *producer = current.getDefiningOp();
    if (!producer)
      continue;
    if (auto loop = dyn_cast<scf::ForOp>(producer)) {
      auto resultType = dyn_cast<FragmentType>(current.getType());
      auto sources = loop->getAttrOfType<ArrayAttr>(reductionSourcesAttr);
      bool consumesSource = sources && llvm::any_of(sources, [&](Attribute attr) {
        auto identity = dyn_cast<PhysicalSourceAttr>(attr);
        return identity &&
               PhysicalSourceAxis{identity.getSourceId(), identity.getSourceAxis(),
                                  identity.getDerived()} == source;
      });
      if (!consumesSource || !resultType ||
          !dominance.dominates(current, anchor) ||
          llvm::any_of(resultType.getAxisMaps(), [&](Attribute attr) {
            return cast<AxisMapAttr>(attr).getDimensionId() == *dimension;
          }) ||
          !samePhysicalScalarExpression(loop.getLowerBound(),
                                         range.getLogicalStart()) ||
          !samePhysicalScalarExpression(loop.getUpperBound(),
                                         range.getLogicalStop()))
        continue;
      bool readsOnly = !loop->walk([](Operation *operation) {
        return isa<LoadOp, GatherOp, scf::ForOp, scf::IfOp, scf::WhileOp>(operation) ||
                       isMemoryEffectFree(operation)
                   ? WalkResult::advance()
                   : WalkResult::interrupt();
      }).wasInterrupted();
      if (!readsOnly)
        continue;
      llvm::DenseSet<Value> dependencies;
      std::function<bool(Value)> usesOutputRange = [&](Value dependency) {
        if (!dependencies.insert(dependency).second)
          return false;
        if (auto argument = dyn_cast<BlockArgument>(dependency)) {
          Operation *owner = argument.getOwner()->getParentOp();
          return owner != kernel && owner != loop && !loop->isAncestor(owner);
        }
        Operation *definition = dependency.getDefiningOp();
        if (!definition)
          return true;
        if (auto candidate = dyn_cast<MakeRangeOp>(definition))
          if (analysis.lockstepRanges({range, candidate}).isExact())
            return true;
        if (llvm::any_of(definition->getOperands(), usesOutputRange))
          return true;
        for (Region &region : definition->getRegions())
          for (Block &block : region)
            for (Operation &nested : block)
              if (llvm::any_of(nested.getOperands(), usesOutputRange))
                return true;
        return false;
      };
      bool dependent = usesOutputRange(current);
      if (!dependent)
        return true;
      continue;
    }
    if (producer->getNumRegions() == 0)
      pending.append(producer->operand_begin(), producer->operand_end());
  }
  return false;
}

void collectProducerRanges(Value value, PhysicalSourceAxis source,
                           llvm::SmallPtrSetImpl<Operation *> &ranges) {
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  if (!kernel)
    return;
  if (queryFragmentAxes(value.getType(), source).empty())
    return;
  PhysicalProgramAnalysis analysis(kernel);
  PhysicalRangeFact fact = analysis.sourceRanges(value, source);
  if (fact.state == PhysicalFactState::Unknown)
    return;
  for (MakeRangeOp range : fact.roots)
    if (sourceAxisIdentity(range) == source)
      ranges.insert(range.getOperation());
}

bool containsSource(Value value, PhysicalSourceAxis source) {
  return !queryFragmentAxes(value.getType(), source).empty();
}

std::optional<int64_t> estimatedFragmentRegisters(func::FuncOp kernel, Value value) {
  auto fragment = dyn_cast<FragmentType>(value.getType());
  if (!fragment || !isa<IntegerType, FloatType>(fragment.getElementType()))
    return std::nullopt;
  int64_t limit = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr).getRegistersPerUnit();
  return minimumFragmentRegisterFootprint(kernel, value, limit,
      FragmentFootprintScope::FullScalarSeedCapacity);
}

bool containsTraversal(Value value, PhysicalSourceAxis source,
                       ArrayRef<int64_t> dimensions) {
  std::function<bool(Type)> contains = [&](Type type) {
    if (auto fragment = dyn_cast<FragmentType>(type))
      return llvm::any_of(queryFragmentAxes(fragment, source),
                          [&](const PhysicalAxisProjection &projection) {
                            return llvm::is_contained(dimensions,
                                                      projection.dimensionId);
                          });
    auto record = dyn_cast<RecordType>(type);
    return record && llvm::any_of(record.getFieldTypes(), [&](Attribute field) {
             return contains(cast<TypeAttr>(field).getValue());
           });
  };
  return contains(value.getType());
}

void collectTraversalDimensions(Type type, PhysicalSourceAxis source,
                                SmallVectorImpl<int64_t> &dimensions) {
  if (auto fragment = dyn_cast<FragmentType>(type)) {
    for (const PhysicalAxisProjection &projection :
         queryFragmentAxes(fragment, source))
      if (!llvm::is_contained(dimensions, projection.dimensionId))
        dimensions.push_back(projection.dimensionId);
    return;
  }
  if (auto record = dyn_cast<RecordType>(type))
    for (Attribute field : record.getFieldTypes())
      collectTraversalDimensions(cast<TypeAttr>(field).getValue(), source,
                                 dimensions);
}

FailureOr<FragmentType> coordinateValueSchema(Type element,
                                              ValueRange coordinates) {
  SmallVector<Attribute> shape;
  SmallVector<Attribute> mappings;
  std::optional<uint64_t> owner;
  for (Value coordinate : coordinates) {
    auto fragment = dyn_cast<FragmentType>(coordinate.getType());
    if (!fragment)
      continue;
    if (owner && *owner != fragment.getOwner())
      return failure();
    owner = fragment.getOwner();
    for (auto [extent, attribute] :
         llvm::zip(fragment.getShape(), fragment.getAxisMaps())) {
      auto mapping = cast<AxisMapAttr>(attribute);
      auto found = llvm::find_if(mappings, [&](Attribute existing) {
        return sourceAxisIdentity(cast<AxisMapAttr>(existing)) ==
               sourceAxisIdentity(mapping);
      });
      if (found != mappings.end()) {
        unsigned axis = std::distance(mappings.begin(), found);
        if (shape[axis] != extent)
          return failure();
        continue;
      }
      shape.push_back(extent);
      mappings.push_back(AxisMapAttr::get(
          element.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
          mapping.getDimensionId(), mappings.size(), mapping.getDerived()));
    }
  }
  if (shape.empty())
    return failure();
  return FragmentType::get(element.getContext(), element,
                           ArrayAttr::get(element.getContext(), shape),
                           ArrayAttr::get(element.getContext(), mappings),
                           /*validity=*/1, owner.value_or(1));
}

bool storeUsesRange(StoreOp store, MakeRangeOp range) {
  return llvm::any_of(store.getCoordinates(), [&](Value coordinate) {
    llvm::SmallPtrSet<Operation *, 8> ranges;
    collectCoordinateRanges(coordinate, ranges);
    return ranges.contains(range.getOperation());
  });
}

std::optional<int64_t> storeAxisForRange(StoreOp store, MakeRangeOp range) {
  std::optional<int64_t> result;
  for (auto [coordinateIndex, coordinate] :
       llvm::enumerate(store.getCoordinates())) {
    llvm::SmallPtrSet<Operation *, 8> ranges;
    collectCoordinateRanges(coordinate, ranges);
    if (!ranges.contains(range.getOperation()))
      continue;
    int64_t sourceAxis = store.getSourceAxes()[coordinateIndex];
    if (!result || sourceAxis > *result)
      result = sourceAxis;
  }
  return result;
}

FailureOr<SmallVector<int64_t>>
traversalDimensionsForStores(ArrayRef<StoreOp> stores, MakeRangeOp range) {
  FailureOr<int64_t> coordinateDimension = queryRangeDimension(range);
  if (failed(coordinateDimension))
    return failure();
  PhysicalSourceAxis source = sourceAxisIdentity(range);
  SmallVector<int64_t> dimensions{*coordinateDimension};
  auto kernel = range->getParentOfType<func::FuncOp>();
  PhysicalProgramAnalysis analysis(kernel);
  auto collect = [&](Value value) -> LogicalResult {
    auto fragment = dyn_cast<FragmentType>(value.getType());
    if (!fragment)
      return success();
    auto axes = queryFragmentAxes(fragment, source);
    if (llvm::all_of(axes, [&](const PhysicalAxisProjection &axis) {
          return axis.dimensionId == *coordinateDimension;
        }))
      return success();
    PhysicalRangeAxisFact selected = analysis.rangeAxes(value, {range});
    if (!selected.isExact())
      return failure();
    for (unsigned axis : selected.fragmentAxes) {
      auto mapping = cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
      if (sourceAxisIdentity(mapping) == source &&
          mapping.getDimensionId() > 0 &&
          !llvm::is_contained(dimensions, mapping.getDimensionId()))
        dimensions.push_back(mapping.getDimensionId());
    }
    return success();
  };
  for (StoreOp store : stores) {
    if (failed(collect(store.getValue())) ||
        (store.getValid() && failed(collect(store.getValid()))))
      return failure();
    for (Value coordinate : store.getCoordinates())
      if (failed(collect(coordinate)))
        return failure();
  }
  return dimensions;
}

FailureOr<int64_t> reuseTraversalDimension(func::FuncOp kernel,
                                           ArrayRef<StoreOp> stores,
                                           MakeRangeOp range) {
  FailureOr<SmallVector<int64_t>> dimensions =
      traversalDimensionsForStores(stores, range);
  if (failed(dimensions))
    return failure();
  PhysicalSourceAxis source = sourceAxisIdentity(range);
  SmallVector<int64_t> selected;
  for (int64_t dimension : *dimensions) {
    bool depends = false;
    for (StoreOp store : stores) {
      PhysicalReductionDependencyFact fact =
          PhysicalProgramAnalysis(kernel).reductionDependency(
              store.getValue(), source, dimension);
      if (!fact.isExact())
        return failure();
      depends |= fact.depends;
    }
    if (depends)
      selected.push_back(dimension);
  }
  return selected.size() == 1 ? FailureOr<int64_t>(selected.front())
                              : FailureOr<int64_t>(failure());
}

std::optional<unsigned> repeatedReductionOutputAxis(
    Value value, MakeRangeOp range) {
  auto reduce = value.getDefiningOp<ReduceOp>();
  if (!reduce || reduce.getSources().size() != 1 || reduce->getNumResults() != 1 ||
      !isUnitStepRange(range))
    return std::nullopt;
  PhysicalSourceAxis source = sourceAxisIdentity(range);
  auto input = dyn_cast<FragmentType>(reduce.getSources().front().getType());
  auto output = queryFragmentAxis(value.getType(), source);
  if (!input || !output.isExact())
    return std::nullopt;
  SmallVector<unsigned> freeAxes;
  bool reducesSource = false;
  for (auto [axis, attribute] : llvm::enumerate(input.getAxisMaps())) {
    bool reduced = llvm::is_contained(reduce.getAxes(), static_cast<int64_t>(axis));
    if (!reduced)
      freeAxes.push_back(axis);
    else
      reducesSource |= sourceAxisIdentity(cast<AxisMapAttr>(attribute)) == source;
  }
  if (!reducesSource || output.fragmentAxis >= freeAxes.size() ||
      !(sourceAxisIdentity(cast<AxisMapAttr>(
            input.getAxisMaps()[freeAxes[output.fragmentAxis]])) == source))
    return std::nullopt;
  auto kernel = reduce->getParentOfType<func::FuncOp>();
  PhysicalProgramAnalysis analysis(kernel);
  auto ranges = analysis.axisRanges(reduce.getSources().front(),
                                    freeAxes[output.fragmentAxis]);
  if (!ranges.isExact() || !ranges.blockers.empty() || ranges.roots.empty())
    return std::nullopt;
  SmallVector<MakeRangeOp> authorities(ranges.roots.begin(), ranges.roots.end());
  authorities.push_back(range);
  if (!analysis.lockstepRanges(authorities).isExact())
    return std::nullopt;
  return output.fragmentAxis;
}

void collectCoordinateRanges(Value coordinate,
                             llvm::SmallPtrSetImpl<Operation *> &ranges) {
  auto kernel = coordinate.getParentRegion()->getParentOfType<func::FuncOp>();
  auto fragment = dyn_cast<FragmentType>(coordinate.getType());
  if (!kernel || !fragment)
    return;
  PhysicalProgramAnalysis analysis(kernel);
  for (unsigned axis = 0; axis < fragment.getShape().size(); ++axis) {
    PhysicalRangeFact fact = analysis.axisRanges(coordinate, axis);
    if (fact.state != PhysicalFactState::Exact)
      continue;
    for (MakeRangeOp range : fact.roots)
      ranges.insert(range.getOperation());
  }
}

void collectStoreRanges(Value value, llvm::SmallPtrSetImpl<Operation *> &ranges,
                        llvm::SmallPtrSetImpl<Operation *> &visited) {
  for (Operation *user : value.getUsers()) {
    if (!visited.insert(user).second)
      continue;
    if (isa<CastOp, BroadcastOp>(user)) {
      collectStoreRanges(user->getResult(0), ranges, visited);
      continue;
    }
    auto store = dyn_cast<StoreOp>(user);
    if (!store || store.getValue() != value)
      continue;
    for (Value coordinate : store.getCoordinates())
      collectCoordinateRanges(coordinate, ranges);
  }
}

HistogramOp histogramSource(Value value) {
  while (Operation *producer = value.getDefiningOp()) {
    if (auto histogram = dyn_cast<HistogramOp>(producer))
      return histogram;
    if (!isa<BroadcastOp, CastOp>(producer) || producer->getNumOperands() != 1 ||
        producer->getNumResults() != 1)
      return HistogramOp();
    value = producer->getOperand(0);
  }
  return HistogramOp();
}

ContractFreeAxisFacts contractFreeAxisFacts(func::FuncOp kernel,
                                            MakeRangeOp range) {
  ContractFreeAxisFacts facts;
  PhysicalProgramAnalysis analysis(kernel);
  kernel.walk([&](ContractOp contract) {
    PhysicalContractFreeAxisFact freeAxes =
        analysis.contractFreeAxes(contract);
    if (!freeAxes.isExact())
      return;
    bool matrixFreeAxes =
        llvm::any_of(freeAxes.axes, [&](const auto &axis) {
          return axis.operand == contract.getLhs();
        }) &&
        llvm::any_of(freeAxes.axes, [&](const auto &axis) {
          return axis.operand == contract.getRhs();
        });
    for (const PhysicalContractFreeAxis &axis : freeAxes.axes) {
      if (!llvm::any_of(axis.ranges.roots, [&](MakeRangeOp root) {
            return sameLogicalRange(root, range);
          }))
        continue;
      unsigned side = ContractFreeAxisNone;
      if (axis.operand == contract.getLhs())
        side |= ContractFreeAxisLhs;
      if (axis.operand == contract.getRhs())
        side |= ContractFreeAxisRhs;
      facts.sides |= side;
      if (matrixFreeAxes)
        facts.matrixSides |= side;
      facts.operandElementBitWidth = std::max(
          facts.operandElementBitWidth,
          std::max(physicalElementBitWidth(contract.getLhs().getType()),
                   physicalElementBitWidth(contract.getRhs().getType())));
      auto fold = contract->getParentOfType<RegionFoldOp>();
      if (!fold && !contract.getLhsBatchAxes().empty() &&
          llvm::count_if(freeAxes.axes, [&](const auto &free) {
            return free.operand == contract.getLhs();
          }) == 1 &&
          llvm::count_if(freeAxes.axes, [&](const auto &free) {
            return free.operand == contract.getRhs();
          }) == 1)
        facts.batchedContraction = true;
      if (fold &&
          lookupParameter(kernel, fold.getSegment()).getCategory() ==
              ParameterCategory::RegionContraction)
        facts.regionContraction = true;
    }
  });
  return facts;
}

unsigned contractFreeAxisSides(func::FuncOp kernel, MakeRangeOp range) {
  return contractFreeAxisFacts(kernel, range).sides;
}

StructuredRangeUses classifyStructuredRanges(
    func::FuncOp kernel, ArrayRef<MakeRangeOp> allRanges,
    const llvm::SmallPtrSetImpl<Operation *> &laneReductions) {
  // Classify ranges consumed by structured operations before considering any
  // pointwise reblocking.  Their logical extent belongs to the structured
  // traversal; replacing it with a pointwise fragment extent would truncate
  // fold/scan/reduction/contract semantics before the owning pass sees them.
  llvm::SmallPtrSet<Operation *, 32> internalTraversalRanges;
  llvm::SmallPtrSet<Operation *, 32> structuredTraversalRanges;
  llvm::SmallPtrSet<Operation *, 32> reductionTraversalRanges;
  llvm::SmallPtrSet<Operation *, 32> reductionFreeRanges;
  llvm::SmallDenseSet<uint64_t> scanSegmentDimensions;
  llvm::SmallDenseSet<PhysicalSourceAxis> scanSegmentSources;
  auto collectAxisInto = [&](Value source, uint64_t axis,
                             llvm::SmallPtrSetImpl<Operation *> &ranges) {
    auto fragment = dyn_cast<FragmentType>(source.getType());
    if (!fragment || axis >= fragment.getAxisMaps().size())
      return;
    auto mapping = cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
    collectProducerRanges(
        source,
        PhysicalSourceAxis{mapping.getSourceId(), mapping.getSourceAxis(),
                           mapping.getDerived()},
        ranges);
  };
  std::function<void(Value, llvm::SmallPtrSetImpl<Operation *> &)>
      collectAllAxesInto = [&](Value value,
                               llvm::SmallPtrSetImpl<Operation *> &ranges) {
    auto fragment = dyn_cast<FragmentType>(value.getType());
    if (!fragment) {
      if (auto record = value.getDefiningOp<MakeRecordOp>())
        for (Value field : record.getFields())
          collectAllAxesInto(field, ranges);
      return;
    }
    PhysicalRangeFact all = PhysicalProgramAnalysis(kernel).sourceRanges(value);
    for (MakeRangeOp range : all.roots)
      ranges.insert(range.getOperation());
    for (Attribute attribute : fragment.getAxisMaps()) {
      auto mapping = cast<AxisMapAttr>(attribute);
      collectProducerRanges(
          value,
          PhysicalSourceAxis{mapping.getSourceId(), mapping.getSourceAxis(),
                           mapping.getDerived()},
          ranges);
    }
  };
  auto collectStructuredRegionRanges = [&](Operation *structured,
                                           ValueRange sources, uint64_t axis) {
    SmallVector<PhysicalSourceAxis> segmentSources;
    for (Value source : sources) {
      auto fragment = dyn_cast<FragmentType>(source.getType());
      if (!fragment || axis >= fragment.getAxisMaps().size())
        continue;
      auto mapping = cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
      PhysicalSourceAxis physical{mapping.getSourceId(),
                                  mapping.getSourceAxis(),
                                  mapping.getDerived()};
      if (!llvm::is_contained(segmentSources, physical))
        segmentSources.push_back(physical);
    }
    for (Region &region : structured->getRegions())
      region.walk([&](MakeRangeOp range) {
        if (llvm::is_contained(
                segmentSources,
                PhysicalSourceAxis{range.getSourceId(), range.getSourceAxis(),
                           range.getDerived()}))
          structuredTraversalRanges.insert(range.getOperation());
      });
  };
  kernel.walk([&](RegionFoldOp fold) {
    ValueRange sources = fold.getSources();
    for (Value source : sources)
      collectAxisInto(source, fold.getAxis(), structuredTraversalRanges);
    collectStructuredRegionRanges(fold, sources, fold.getAxis());
  });
  auto recordScanTraversal = [&](Value source, uint64_t axis) {
    collectAxisInto(source, axis, structuredTraversalRanges);
    auto fragment = dyn_cast<FragmentType>(source.getType());
    if (!fragment || axis >= fragment.getAxisMaps().size())
      return;
    auto mapping = cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
    int64_t dimension = mapping.getDimensionId();
    if (dimension > 0)
      scanSegmentDimensions.insert(static_cast<uint64_t>(dimension));
    scanSegmentSources.insert({mapping.getSourceId(), mapping.getSourceAxis(),
                               mapping.getDerived()});
  };
  kernel.walk([&](RegionScanOp scan) {
    ValueRange sources = scan.getSources();
    for (Value source : sources)
      recordScanTraversal(source, scan.getAxis());
    collectStructuredRegionRanges(scan, sources, scan.getAxis());
  });
  auto collectReductionAxis = [&](Value source, int64_t axis) {
    auto fragment = dyn_cast<FragmentType>(source.getType());
    if (!fragment || axis < 0 ||
        axis >= static_cast<int64_t>(fragment.getShape().size()))
      return;
    PhysicalProgramAnalysis analysis(kernel);
    PhysicalAxisRealizationFact fact = analysis.axisRealization(source, axis);
    for (MakeRangeOp range : allRanges) {
      if (!llvm::any_of(fact.roots, [&](MakeRangeOp root) {
            return sameLogicalRange(root, range) &&
                   analysis.lockstepRanges({root, range}).isExact();
          }))
        continue;
      structuredTraversalRanges.insert(range.getOperation());
      reductionTraversalRanges.insert(range.getOperation());
    }
  };
  kernel.walk([&](ReduceOp reduce) {
    // A newly formed loop guard reduces the physical ownership tile, not an
    // author traversal. Its cardinality follows the tile selected by this pass.
    if (laneReductions.contains(reduce))
      return;
    for (Value source : reduce.getSources()) {
      auto fragment = dyn_cast<FragmentType>(source.getType());
      if (!fragment)
        continue;
      for (int64_t axis : reduce.getAxes())
        collectReductionAxis(source, axis);
      for (unsigned axis = 0; axis < fragment.getShape().size(); ++axis) {
        if (llvm::is_contained(reduce.getAxes(), static_cast<int64_t>(axis)))
          continue;
        PhysicalAxisRealizationFact fact =
            PhysicalProgramAnalysis(kernel).axisRealization(source, axis);
        for (MakeRangeOp range : fact.roots) {
          structuredTraversalRanges.insert(range.getOperation());
          reductionFreeRanges.insert(range.getOperation());
        }
      }
    }
  });
  kernel.walk([&](ScanOp scan) {
    for (Value source : scan.getSources())
      recordScanTraversal(source, scan.getAxis());
  });
  kernel.walk([&](HistogramOp histogram) {
    llvm::SmallPtrSet<Operation *, 16> histogramRanges;
    collectAllAxesInto(histogram.getValues(), histogramRanges);
    structuredTraversalRanges.insert(histogramRanges.begin(),
                                     histogramRanges.end());
    reductionTraversalRanges.insert(histogramRanges.begin(),
                                    histogramRanges.end());
  });
  llvm::SmallPtrSet<Operation *, 32> contractionTraversalRanges;
  llvm::SmallPtrSet<Operation *, 8> contractionOwnedRanges;
  llvm::SmallPtrSet<Operation *, 8> contractionOwnedStores;
  kernel.walk([&](ContractOp contract) {
    auto collectConsumedAxes = [&](Value operand, ArrayRef<int64_t> axes) {
      auto type = cast<FragmentType>(operand.getType());
      for (int64_t axis : axes) {
        auto source = sourceAxisIdentity(cast<AxisMapAttr>(type.getAxisMaps()[axis]));
        // A source can occupy both K and a retained operand position. Its
        // retained occurrence still needs the contract's free-axis mapping.
        if (!queryFragmentAxes(contract.getResult().getType(), source).empty())
          continue;
        collectReductionAxis(operand, axis);
      }
    };
    collectConsumedAxes(contract.getLhs(), contract.getLhsReductionAxes());
    collectConsumedAxes(contract.getRhs(), contract.getRhsReductionAxes());
    collectAllAxesInto(contract.getLhs(), contractionTraversalRanges);
    collectAllAxesInto(contract.getRhs(), contractionTraversalRanges);
    PhysicalProgramAnalysis analysis(kernel);
    PhysicalContractFreeAxisFact freeAxes = analysis.contractFreeAxes(contract);
    bool replayableFreeCoordinates = freeAxes.isExact() &&
        llvm::all_of(freeAxes.axes, [&](const auto &axis) {
          FailureOr<AxisMapAttr> mapping =
              queryAxisMap(axis.operand.getType(), axis.operandAxis);
          if (failed(mapping))
            return false;
          auto source = sourceAxisIdentity(*mapping);
          auto replay = analysis.replayAt(
              axis.operand, source, PhysicalReplayScope::Coordinate,
              /*allowAccesses=*/true, contract.getOperation(), IRMapping{},
              mapping->getDimensionId());
          return replay.isReplayable() &&
                 llvm::any_of(replay.accesses, [&](Operation *access) {
                   auto load = dyn_cast<LoadOp>(access);
                   return load && queryCoordinateIndex(
                       load.getCoordinates(), source,
                       mapping->getDimensionId()).isExact();
                 });
        });
    if (replayableFreeCoordinates) {
      SmallVector<StoreOp> stores;
      if (contraction::hasRangeContractForm(contract, &stores)) {
        // Defer output ownership only when the contraction realizer actually
        // accepts this form. Replayable coordinates alone do not transfer the
        // workset: unsupported forms retain their ordinary pointwise owner.
        llvm::SmallPtrSet<Operation *, 16> visited;
        collectStoreRanges(contract.getResult(), internalTraversalRanges, visited);
        for (const auto &axis : freeAxes.axes)
          for (MakeRangeOp range : axis.ranges.roots)
            contractionOwnedRanges.insert(range.getOperation());
        for (StoreOp store : stores) {
          contractionOwnedStores.insert(store.getOperation());
          auto payload = cast<FragmentType>(store.getValue().getType());
          for (unsigned axis = 0; axis < payload.getShape().size(); ++axis) {
            PhysicalRangeFact ranges = analysis.axisRanges(store.getValue(), axis);
            // An epilogue axis may read several independent sources. Defer
            // each known range to contraction tiling without equating them.
            if (ranges.state == PhysicalFactState::Unknown || !ranges.blockers.empty())
              continue;
            for (MakeRangeOp range : ranges.roots)
              contractionOwnedRanges.insert(range.getOperation());
          }
          for (Value coordinate : store.getCoordinates())
            collectCoordinateRanges(coordinate, contractionOwnedRanges);
        }
      }
    }
  });
  kernel.walk([&](ScaledContractOp contract) {
    collectAllAxesInto(contract.getLhs(), contractionTraversalRanges);
    collectAllAxesInto(contract.getLhsScale(), contractionTraversalRanges);
    collectAllAxesInto(contract.getRhs(), contractionTraversalRanges);
    collectAllAxesInto(contract.getRhsScale(), contractionTraversalRanges);
    llvm::SmallPtrSet<Operation *, 16> visited;
    collectStoreRanges(contract.getResult(), internalTraversalRanges, visited);
  });
  kernel.walk([&](SparseContractOp contract) {
    collectAllAxesInto(contract.getCompressed(), contractionTraversalRanges);
    collectAllAxesInto(contract.getMetadata(), contractionTraversalRanges);
    collectAllAxesInto(contract.getRhs(), contractionTraversalRanges);
    llvm::SmallPtrSet<Operation *, 16> visited;
    collectStoreRanges(contract.getResult(), internalTraversalRanges, visited);
  });
  SmallVector<MakeRangeOp> contractionAuthorities;
  for (Operation *operation : contractionTraversalRanges)
    if (auto range = dyn_cast<MakeRangeOp>(operation))
      contractionAuthorities.push_back(range);
  for (MakeRangeOp range : allRanges)
    if (llvm::any_of(contractionAuthorities, [&](MakeRangeOp authority) {
          return sameLogicalRange(authority, range);
        }))
      contractionTraversalRanges.insert(range.getOperation());
  structuredTraversalRanges.insert(contractionTraversalRanges.begin(),
                                   contractionTraversalRanges.end());
  SmallVector<MakeRangeOp> structuredAuthorities;
  for (Operation *operation : structuredTraversalRanges)
    if (auto range = dyn_cast<MakeRangeOp>(operation))
      structuredAuthorities.push_back(range);
  for (MakeRangeOp range : allRanges)
    if (llvm::any_of(structuredAuthorities, [&](MakeRangeOp authority) {
          return sameLogicalRange(authority, range);
        }))
      structuredTraversalRanges.insert(range.getOperation());
  return {std::move(internalTraversalRanges), std::move(structuredTraversalRanges),
          std::move(reductionTraversalRanges), std::move(reductionFreeRanges),
          std::move(scanSegmentDimensions), std::move(scanSegmentSources),
          std::move(contractionTraversalRanges), std::move(contractionOwnedRanges),
          std::move(contractionOwnedStores)};
}


SmallVector<WriteEffectFacts> readWriteEffects(func::FuncOp kernel) {
  SmallVector<WriteEffectFacts> effects;
  auto append = [&](Operation *operation, ValueRange coordinates, ValueRange payloads) {
    effects.push_back({operation, llvm::to_vector(coordinates), llvm::to_vector(payloads)});
  };
  kernel.walk([&](Operation *operation) {
    if (auto access = dyn_cast<AccessOpInterface>(operation);
        access && access.writesMemory())
      append(operation, access.getAccessCoordinates(), access.getAccessPayloads());
  });
  return effects;
}


FailureOr<Attribute> mappingAxis(func::FuncOp kernel, Attribute attribute) {
  auto expression = dyn_cast<PhysicalExprAttr>(attribute);
  if (expression && expression.getKind() ==
                        PhysicalExprKind::CeilDiv &&
      expression.getOperands().size() == 2) {
    auto divisor = dyn_cast<PhysicalExprAttr>(expression.getOperands()[1]);
    // Capacity projection retains the raw ownership parameter as the first
    // operand; the second operand is its compile-time coverage capacity.
    if (divisor && divisor.getKind() == PhysicalExprKind::Minimum &&
        divisor.getOperands().size() == 2 &&
        isCompileTimePhysicalExpr(divisor))
      divisor = dyn_cast<PhysicalExprAttr>(divisor.getOperands()[0]);
    if (divisor && divisor.getKind() ==
                       PhysicalExprKind::Parameter) {
      FailureOr<ParameterAttr> parameter = queryParameterBySymbol(kernel, divisor.getParameterReference().getName());
      if (succeeded(parameter))
        return parameterAxis(*parameter);
    }
  }
  FailureOr<uint64_t> blocked = blockedDimension(attribute);
  if (succeeded(blocked))
    return dimensionAxisKey(kernel.getContext(), *blocked);
  auto physical = dyn_cast<PhysicalExprAttr>(attribute);
  if (!physical ||
      physical.getKind() !=
          PhysicalExprKind::Dimension)
    return failure();
  return physical.getValue() > 0
             ? FailureOr<Attribute>(dimensionAxisKey(kernel.getContext(),
                                                     physical.getValue()))
             : FailureOr<Attribute>(failure());

}


FailureOr<MappingCoordinates> readMappingCoordinates(func::FuncOp kernel, ExecutionGroupOp mapping) {
  MappingCoordinates result;
  for (auto [axis, extent] : llvm::enumerate(mapping.getLaunchExtents())) {
    FailureOr<Attribute> mapped = mappingAxis(kernel, extent);
    if (succeeded(mapped) && axis < mapping.getCoordinates().size()) {
      auto existing = result.axes.find(*mapped);
      if (existing != result.axes.end() && existing->second != axis)
        return kernel.emitError(
            "one pointwise ownership axis maps to multiple program coordinates");
      result.tiles[*mapped] = mapping.getCoordinates()[axis];
      result.axes[*mapped] = axis;
      result.coordinates[axis] = *mapped;
    }
    auto physical = dyn_cast<PhysicalExprAttr>(extent);
    if (!result.reusableUnit && axis < mapping.getCoordinates().size() &&
        mapping.getCoordinates()[axis].use_empty() && physical &&
        physical.getKind() ==
            PhysicalExprKind::Constant &&
        physical.getValue() == 1)
      result.reusableUnit = axis;
  }
  return result;
}


FailureOr<Attribute> PointwiseRewrite::effectLocalKey(MakeRangeOp range) {
    FailureOr<int64_t> dimension = queryRangeDimension(range);
    if (failed(dimension))
      return failure();
    return ArrayAttr::get(
        kernel.getContext(),
        {sourceAxisKey(kernel.getContext(), sourceAxisIdentity(range)),
         dimensionAxisKey(kernel.getContext(), *dimension)});
  
}


bool PointwiseRewrite::coordinatesUseRange(ValueRange coordinates, MakeRangeOp range) {
    return llvm::any_of(coordinates, [&](Value coordinate) {
      llvm::SmallPtrSet<Operation *, 8> ranges;
      collectCoordinateRanges(coordinate, ranges);
      return ranges.contains(range.getOperation());
    });
  
}


bool PointwiseRewrite::coordinatesDirectlyUseRange(ValueRange coordinates,
                                         MakeRangeOp range) {
    if (!coordinatesUseRange(coordinates, range))
      return false;
    PhysicalSourceAxis source = sourceAxisIdentity(range);
    return llvm::any_of(coordinates, [&](Value coordinate) {
      if (!queryFragmentAxis(coordinate.getType(), source).isExact())
        return false;
      PhysicalReplayFact replay = PhysicalProgramAnalysis(kernel).replayability(
          coordinate, source, PhysicalReplayScope::Coordinate,
          /*allowAccesses=*/true);
      return replay.isReplayable() && !replay.crossesAccess;
    });
  
}


bool PointwiseRewrite::contractionFreeAxisNeedsRange(Operation *operation,
                                           MakeRangeOp range) {
    PhysicalContractFreeAxisFact fact =
        PhysicalProgramAnalysis(kernel).contractFreeAxes(operation);
    return llvm::any_of(fact.axes, [&](const auto &axis) {
      return !axis.realization.physicalized && axis.ranges.isExact() &&
             llvm::any_of(axis.ranges.roots, [&](MakeRangeOp root) {
               return sameLogicalRange(root, range);
             });
    });
  
}


bool PointwiseRewrite::sharesLogicalTraversal(MakeRangeOp lhs, MakeRangeOp rhs) {
    if (sameLogicalRange(lhs, rhs))
      return true;
    return PhysicalProgramAnalysis(kernel).lockstepRanges({lhs, rhs}).isExact();
  
}


bool PointwiseRewrite::hasPointwiseOwnership(MakeRangeOp range) {
    if (isContractionOwned(range.getOperation()) ||
        retainedCartesianRanges.contains(range.getOperation()))
      return false;
    FailureOr<uint64_t> dimension = ownershipDimension(kernel, range);
    // Equal dimensions do not identify Cartesian coordinates. Only a distinct
    // free occurrence can bind source-specific ownership here; contraction
    // blocking owns the remaining operand/result positions.
    if (succeeded(dimension) &&
        nonUniqueContractionDimensions.contains(*dimension) &&
        !independentContractionRanges.contains(range.getOperation()))
      return false;
    PhysicalSourceAxis source{range.getSourceId(), range.getSourceAxis(),
                              range.getDerived()};
    bool effectOwned = ownershipSources.contains(source) ||
                       writeTraversalRanges.contains(range.getOperation());
    return effectOwned &&
           (!range->hasAttr(sourceSubregionAttr) ||
            directOwnershipSources.contains(source) ||
            writeTraversalRanges.contains(range.getOperation())) &&
           !isReductionTraversal(range.getOperation()) &&
           !uses.scanSegmentSources.contains(source) &&
           (failed(dimension) ||
            (!uses.scanSegmentDimensions.contains(*dimension) &&
             !partiallyCarriedStructuredDimensions.contains(*dimension)));
  
}


} // namespace intent::gpu::pointwise
