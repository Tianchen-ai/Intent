#include "Intent/Analysis/CanonicalKernel.h"
#include "Intent/Analysis/ContractionAxes.h"
#include "Intent/Interfaces/StructuredOpInterface.h"

#include "Intent/Dialect/Intent/IR/IntentOps.h"
#include "Intent/Dialect/Intent/IR/IntentTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/DenseSet.h"

#include <limits>

using namespace mlir;

namespace intent {
namespace {

void appendOrigins(CoordinateProvenance &destination,
                   const CoordinateProvenance &source) {
  destination.known = destination.known && source.known;
  for (const CoordinateOrigin &origin : source.origins)
    if (!llvm::is_contained(destination.origins, origin))
      destination.origins.push_back(origin);
}

CoordinateProvenance knownWithoutCoordinates() {
  CoordinateProvenance result;
  result.known = true;
  return result;
}

Value absoluteCoordinateSource(Value source) {
  while (auto subregion = source.getDefiningOp<SubregionOp>()) {
    if (subregion.getInputs().empty())
      break;
    source = subregion.getInputs().front();
  }
  return source;
}

DenseI64ArrayAttr dimensionIDs(RankedTensorType tensor) {
  auto encoding = dyn_cast_or_null<TensorShapeAttr>(tensor.getEncoding());
  return encoding ? encoding.getDimensions() : DenseI64ArrayAttr();
}

LogicalResult collectDomainValues(Value source,
                                  SmallVectorImpl<Value> &domains) {
  if (isa_and_nonnull<DomainOp, SubregionOp>(source.getDefiningOp())) {
    domains.push_back(source);
    return success();
  }
  auto product = source.getDefiningOp<DomainProductOp>();
  if (!product)
    return failure();
  for (Value component : product.getDomains())
    if (failed(collectDomainValues(component, domains)))
      return failure();
  return success();
}

bool hasObservableWrite(Operation *operation) {
  auto interface = dyn_cast<MemoryEffectOpInterface>(operation);
  if (!interface)
    return false;
  SmallVector<MemoryEffects::EffectInstance> effects;
  interface.getEffects(effects);
  return llvm::any_of(effects, [](const MemoryEffects::EffectInstance &effect) {
    return isa<MemoryEffects::Write>(effect.getEffect());
  });
}

LogicalResult collectLogicalWorkset(
    ParallelOp parallel, ArrayRef<Value> parentDomains,
    ArrayRef<BlockArgument> parentCoordinates,
    SmallVectorImpl<LogicalWorksetFact> &worksets) {
  LogicalWorksetFact current;
  current.parallel = parallel;
  current.domains.append(parentDomains.begin(), parentDomains.end());
  current.coordinates.append(parentCoordinates.begin(),
                             parentCoordinates.end());
  SmallVector<Value> localDomains;
  if (failed(collectDomainValues(parallel.getSource(), localDomains)) ||
      localDomains.empty())
    return parallel.emitOpError(
        "parallel workset requires a finite canonical domain product");
  Block &body = parallel.getBody().front();
  if (body.getNumArguments() != localDomains.size())
    return parallel.emitOpError(
        "parallel coordinates do not match the canonical domain product");
  current.domains.append(localDomains.begin(), localDomains.end());
  current.coordinates.append(body.getArguments().begin(),
                             body.getArguments().end());

  SmallVector<ParallelOp> children;
  bool directWrite = false;
  for (Operation &nested : body.without_terminator()) {
    if (auto child = dyn_cast<ParallelOp>(nested)) {
      children.push_back(child);
      continue;
    }
    directWrite |= hasObservableWrite(&nested);
    nested.walk([&](Operation *candidate) {
      if (candidate != &nested)
        directWrite |= hasObservableWrite(candidate);
    });
  }
  // Flatten only a pure parallel nest. Direct effects belong to the outer
  // iteration and must not be repeated for each point of a child workset.
  if (!children.empty() && !directWrite) {
    for (ParallelOp child : children)
      if (failed(collectLogicalWorkset(child, current.domains,
                                       current.coordinates, worksets)))
        return failure();
    return success();
  }

  current.state = CanonicalFactState::Exact;
  current.body = &body;
  worksets.push_back(std::move(current));
  return success();
}

} // namespace

CanonicalKernelAnalysis::CanonicalKernelAnalysis(ModuleOp module)
    : module(module) {}

CoordinateProvenance
CanonicalKernelAnalysis::coordinateProvenance(Value value) {
  if (!value)
    return {};
  if (auto iterator = coordinateCache.find(value);
      iterator != coordinateCache.end())
    return iterator->second;
  if (coordinateActive.lookup(value))
    return {};
  coordinateActive[value] = true;
  CoordinateProvenance result = computeCoordinateProvenance(value);
  coordinateActive.erase(value);
  coordinateCache.try_emplace(value, result);
  return result;
}

CoordinateProvenance
CanonicalKernelAnalysis::computeCoordinateProvenance(Value value) {
  if (auto argument = dyn_cast<BlockArgument>(value))
    return blockArgumentProvenance(argument);
  return resultProvenance(cast<OpResult>(value));
}

CoordinateProvenance CanonicalKernelAnalysis::blockArgumentProvenance(
    BlockArgument argument) {
  Operation *owner = argument.getOwner()->getParentOp();
  if (!owner)
    return knownWithoutCoordinates();
  if (auto loop = dyn_cast<ForOp>(owner)) {
    auto induction = loop.getInductionVars();
    auto coordinate = llvm::find(induction, argument);
    if (coordinate != induction.end()) {
      CoordinateProvenance result = knownWithoutCoordinates();
      result.origins.push_back({absoluteCoordinateSource(loop.getSource()),
                               static_cast<unsigned>(coordinate - induction.begin())});
      return result;
    }
    auto carries = loop.getRegionIterArgs();
    auto carry = llvm::find(carries, argument);
    return carry != carries.end()
               ? coordinateProvenance(loop.getInitArgs()[carry - carries.begin()])
               : CoordinateProvenance();
  }
  if (auto parallel = dyn_cast<ParallelOp>(owner)) {
    CoordinateProvenance result = knownWithoutCoordinates();
    result.origins.push_back({absoluteCoordinateSource(parallel.getSource()), argument.getArgNumber()});
    return result;
  }
  if (auto structured = dyn_cast<StructuredOpInterface>(owner))
    for (const auto &relation : structured.getRegionArgumentRelations(*argument.getOwner()->getParent()))
      if (relation.to == argument &&
          (relation.kind == StructuredRelationKind::Capture ||
           relation.kind == StructuredRelationKind::SourceSlice))
        return coordinateProvenance(relation.from);
  return knownWithoutCoordinates();
}

CoordinateProvenance
CanonicalKernelAnalysis::resultProvenance(OpResult result) {
  Operation *operation = result.getOwner();
  if (isa<IndicesOp>(operation)) {
    CoordinateProvenance provenance = knownWithoutCoordinates();
    auto axis = operation->getAttrOfType<IntegerAttr>("tensor_axis");
    auto rank = cast<RankedTensorType>(operation->getResult(0).getType()).getRank();
    if (axis)
      provenance.origins.push_back(
          {absoluteCoordinateSource(operation->getOperand(0)),
           static_cast<unsigned>(axis.getInt())});
    else if (rank == 1)
      provenance.origins.push_back(
          {absoluteCoordinateSource(operation->getOperand(0)), 0});
    else
      provenance.known = false;
    return provenance;
  }
  if (isa<ConstantOp, DimOp, RegionEndOp, RandomBitsOp, ContractOp,
          ScaledContractOp, SparseContractOp, HistogramOp, ViewLoadOp,
          BufferLoadOp, AtomicLoadOp, AtomicStoreOp, AtomicRMWOp,
          AtomicCompareExchangeOp>(operation))
    return knownWithoutCoordinates();

  if (isa<ExtractOp>(operation)) {
    Value product = operation->getOperand(0);
    auto field = operation->getAttrOfType<IntegerAttr>("field");
    if (field) {
      if (Operation *definition = product.getDefiningOp();
          definition && isa<MakeTupleOp, MakeRecordOp>(definition) &&
          field.getInt() >= 0 &&
          field.getInt() < definition->getNumOperands())
        return coordinateProvenance(definition->getOperand(field.getInt()));
    }
    return coordinateProvenance(product);
  }

  if (isa<GatherOp, ReshapeOp, BroadcastOp, TransposeOp, CastOp, BitcastOp,
          UnaryOp, FullOp>(operation))
    return coordinateProvenance(operation->getOperand(0));

  if (isa<BinaryOp, CompareOp, JoinOp, SelectOp, MaskOp, MakeTupleOp,
          MakeRecordOp>(operation)) {
    CoordinateProvenance combined = knownWithoutCoordinates();
    for (Value operand : operation->getOperands())
      appendOrigins(combined, coordinateProvenance(operand));
    return combined;
  }

  return {};
}

FailureOr<IndexRelationFact>
CanonicalKernelAnalysis::indexRelation(Operation *operation) {
  auto access = dyn_cast<IndexedAccessOpInterface>(operation);
  if (!access) return failure();
  auto relation = access.getIndexRelation();
  auto terms = relation ? relation.getTerms() : ArrayAttr();
  if (!relation || !terms)
    return failure();
  IndexRelationFact result;
  result.source = access.getAccessSource();
  result.sourceRank = relation.getSourceRank();
  result.resultDimensionIdentities.append(
      relation.getResultDimensions().asArrayRef().begin(),
      relation.getResultDimensions().asArrayRef().end());
  unsigned sourceAxis = 0;
  Type resourceType = result.source.getType();
  if (auto view = dyn_cast<ViewType>(resourceType)) resourceType = view.getTensor();
  if (auto buffer = dyn_cast<BufferType>(resourceType)) resourceType = buffer.getTensor();
  auto sourceTensor = dyn_cast<RankedTensorType>(resourceType);
  DominanceInfo dominance(operation->getParentOfType<func::FuncOp>());
  for (Attribute attribute : terms) {
    auto term = dyn_cast<IndexTermAttr>(attribute);
    if (!term)
      return failure();
    IndexTermFact fact;
    fact.kind = term.getKind();
    if (fact.kind != 1)
      fact.sourceAxis = sourceAxis++;
    for (int64_t operand : term.getOperandPositions().asArrayRef()) {
      if (operand == -1) {
        fact.operands.push_back({});
        continue;
      }
      if (operand < 0 || operand >= access.getIndexOperands().size())
        return failure();
      fact.operands.push_back(access.getIndexOperands()[operand]);
    }
    for (int64_t value : term.getStaticValues().asArrayRef())
      fact.staticValues.push_back(
          fact.kind == 5 && value == std::numeric_limits<int64_t>::min()
              ? std::nullopt
              : std::optional<int64_t>(value));
    fact.coordinate = knownWithoutCoordinates();
    if (fact.sourceAxis && (fact.kind == 0 || fact.kind == 5))
      fact.coordinate.origins.push_back(
          {absoluteCoordinateSource(result.source), *fact.sourceAxis});
    else if (fact.kind == 4 && !fact.operands.empty())
      fact.coordinate.origins.push_back(
          {absoluteCoordinateSource(fact.operands.front()), 0});
    else if (fact.kind == 3 && !fact.operands.empty())
      fact.coordinate = coordinateProvenance(fact.operands.front());
    if (fact.kind == 2 && sourceTensor && fact.sourceAxis &&
        !sourceTensor.isDynamicDim(*fact.sourceAxis)) {
      int64_t extent = sourceTensor.getDimSize(*fact.sourceAxis);
      int64_t index = *fact.staticValues.front();
      fact.inBounds = -extent <= index && index < extent;
    } else if (fact.kind == 3 && fact.sourceAxis) {
      for (Operation *user : fact.operands.front().getUsers())
        if (auto assumption = dyn_cast<AssumeInBoundsOp>(user))
          if (assumption.getView() == result.source &&
              assumption.getIndex() == fact.operands.front() &&
              assumption.getAxis() == *fact.sourceAxis &&
              dominance.properlyDominates(assumption.getOperation(), operation))
            fact.inBounds = true;
    }
    result.terms.push_back(std::move(fact));
  }
  unsigned basicRank = llvm::count_if(result.terms, [](const IndexTermFact &term) {
    return term.kind == 0 || term.kind == 1 || term.kind == 4 || term.kind == 5;
  });
  if (basicRank > result.resultDimensionIdentities.size())
    return failure();
  result.advancedRank = result.resultDimensionIdentities.size() - basicRank;
  unsigned resultAxis = 0;
  for (IndexTermFact &term : result.terms) {
    if (term.kind == 0 || term.kind == 1 || term.kind == 4 || term.kind == 5) {
      term.resultAxes.push_back(resultAxis++);
      continue;
    }
    if (term.kind != 3 || term.operands.empty())
      continue;
    auto tensor = dyn_cast<RankedTensorType>(term.operands.front().getType());
    if (!tensor)
      continue;
    if (tensor.getRank() > result.advancedRank)
      return failure();
    if (!result.advancedStart) {
      result.advancedStart = resultAxis;
      resultAxis += result.advancedRank;
    }
    for (unsigned axis = 0; axis < result.advancedRank; ++axis)
      term.resultAxes.push_back(*result.advancedStart + axis);
    for (unsigned axis = 0; axis < tensor.getRank(); ++axis)
      term.indexAxes.push_back(*result.advancedStart + result.advancedRank -
                               tensor.getRank() + axis);
  }
  if (resultAxis != result.resultDimensionIdentities.size())
    return failure();
  return result;
}

FailureOr<SmallVector<TensorOperandProjection, 3>>
CanonicalKernelAnalysis::operandProjections(OpResult result) const {
  auto target = dyn_cast<RankedTensorType>(result.getType());
  if (!target)
    return failure();
  Operation *operation = result.getOwner();
  SmallVector<TensorOperandProjection, 3> projections;
  auto append = [&](unsigned operand) -> TensorOperandProjection & {
    auto source = cast<RankedTensorType>(operation->getOperand(operand).getType());
    projections.push_back({operand, {}});
    projections.back().resultAxes.resize(source.getRank());
    return projections.back();
  };
  if (auto transpose = dyn_cast<TransposeOp>(operation)) {
    auto &projection = append(0);
    for (auto [axis, attribute] : llvm::enumerate(transpose.getPermutation()))
      projection.resultAxes[cast<IntegerAttr>(attribute).getInt()] = axis;
    return projections;
  }
  if (isa<ContractOp, ScaledContractOp, SparseContractOp>(operation)) {
    auto integers = [&](StringRef name, unsigned member) {
      SmallVector<int64_t> values;
      for (Attribute value : operation->getAttrOfType<ArrayAttr>(name))
        values.push_back(cast<IntegerAttr>(cast<ArrayAttr>(value)[member]).getInt());
      return values;
    };
    unsigned rhsOperand = isa<ContractOp>(operation) ? 1 : 2;
    auto lhs = cast<RankedTensorType>(operation->getOperand(0).getType());
    auto rhs = cast<RankedTensorType>(operation->getOperand(rhsOperand).getType());
    auto axes = ContractionAxes::get(lhs.getRank(), rhs.getRank(),
        integers("reduce", 0), integers("reduce", 1),
        integers("batch", 0), integers("batch", 1));
    if (!axes)
      return failure();
    append(0).resultAxes.assign(axes->lhsResultAxes.begin(), axes->lhsResultAxes.end());
    append(rhsOperand).resultAxes.assign(axes->rhsResultAxes.begin(), axes->rhsResultAxes.end());
    return projections;
  }
  if (auto structured = dyn_cast<StructuredOpInterface>(operation)) {
    if (structured.getStructuredKind() != StructuredOpKind::Reduce &&
        structured.getStructuredKind() != StructuredOpKind::Scan)
      return failure();
    if (result.getResultNumber() >= structured.getSources().size())
      return failure();
    unsigned operand = result.getResultNumber();
    if (!isa<RankedTensorType>(operation->getOperand(operand).getType()))
      return failure();
    auto &projection = append(operand);
    auto reduced = structured.getIterationAxes();
    unsigned next = 0;
    for (unsigned axis = 0; axis < projection.resultAxes.size(); ++axis)
      if (structured.getStructuredKind() == StructuredOpKind::Scan ||
          !llvm::is_contained(reduced, axis))
        projection.resultAxes[axis] = next++;
    return projections;
  }
  if (auto reshape = dyn_cast<ReshapeOp>(operation)) {
    auto source = dyn_cast<RankedTensorType>(reshape.getInputs().front().getType());
    if (!source)
      return failure();
    auto &projection = append(0);
    // Only explicit dimension identity and occurrence transport coordinates.
    // A regrouped row-major axis has its own result identity; equal extents
    // alone do not identify a coordinate source.
    auto sourceIDs = dimensionIDs(source), targetIDs = dimensionIDs(target);
    if (!sourceIDs || !targetIDs)
      return failure();
    unsigned sourceBegin = 0, targetBegin = 0;
    unsigned sourceEnd = source.getRank(), targetEnd = target.getRank();
    while (sourceBegin < sourceEnd && targetBegin < targetEnd) {
      if (sourceIDs[sourceBegin] == targetIDs[targetBegin]) {
        projection.resultAxes[sourceBegin++] = targetBegin++;
      } else if (source.getDimSize(sourceBegin) == 1) {
        ++sourceBegin;
      } else if (target.getDimSize(targetBegin) == 1) {
        ++targetBegin;
      } else {
        break;
      }
    }
    while (sourceBegin < sourceEnd && targetBegin < targetEnd) {
      if (sourceIDs[sourceEnd - 1] == targetIDs[targetEnd - 1]) {
        projection.resultAxes[--sourceEnd] = --targetEnd;
      } else if (source.getDimSize(sourceEnd - 1) == 1) {
        --sourceEnd;
      } else if (target.getDimSize(targetEnd - 1) == 1) {
        --targetEnd;
      } else {
        break;
      }
    }
    return projections;
  }
  if (!isa<BroadcastOp, UnaryOp, CastOp, BitcastOp, BinaryOp, CompareOp,
           SelectOp, MaskOp, JoinOp>(operation))
    return failure();
  unsigned count = isa<BroadcastOp>(operation) ? 1 : operation->getNumOperands();
  for (unsigned operand = 0; operand < count; ++operand) {
    auto source = dyn_cast<RankedTensorType>(operation->getOperand(operand).getType());
    if (!source)
      continue;
    if (source.getRank() > target.getRank())
      return failure();
    auto &projection = append(operand);
    for (unsigned axis = 0; axis < source.getRank(); ++axis)
      projection.resultAxes[axis] = isa<JoinOp>(operation)
          ? axis : target.getRank() - source.getRank() + axis;
  }
  return projections;
}

CoordinateProvenance CanonicalKernelAnalysis::axisProvenance(Value value,
                                                            unsigned axis) {
  auto key = std::make_pair(value, axis);
  if (auto found = axisCache.find(key); found != axisCache.end())
    return found->second;
  if (axisActive.lookup(key))
    return {};
  axisActive[key] = true;
  auto result = computeAxisProvenance(value, axis);
  axisActive.erase(key);
  axisCache.try_emplace(key, result);
  return result;
}

CoordinateProvenance
CanonicalKernelAnalysis::computeAxisProvenance(Value value, unsigned axis) {
  auto tensor = dyn_cast<RankedTensorType>(value.getType());
  if (!tensor || axis >= tensor.getRank())
    return {};
  auto ownAxis = [&]() {
    auto result = knownWithoutCoordinates();
    result.origins.push_back({value, axis});
    return result;
  };
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    Operation *owner = argument.getOwner()->getParentOp();
    if (auto loop = dyn_cast<ForOp>(owner)) {
      auto carries = loop.getRegionIterArgs();
      auto found = llvm::find(carries, argument);
      if (found != carries.end())
        return axisProvenance(loop.getInitArgs()[found - carries.begin()], axis);
    }
    if (auto loop = dyn_cast<WhileOp>(owner))
      return axisProvenance(loop.getInitArgs()[argument.getArgNumber()], axis);
    if (auto structured = dyn_cast<StructuredOpInterface>(owner))
      for (const auto &relation : structured.getRegionArgumentRelations(*argument.getOwner()->getParent()))
        if (relation.to == argument &&
            (relation.kind == StructuredRelationKind::Capture ||
             relation.kind == StructuredRelationKind::SourceSlice)) {
          auto source = dyn_cast<RankedTensorType>(relation.from.getType());
          if (source && source.getRank() == tensor.getRank())
            return axisProvenance(relation.from, axis);
        }
    return ownAxis();
  }
  auto result = cast<OpResult>(value);
  Operation *operation = result.getOwner();
  if (auto indices = dyn_cast<IndicesOp>(operation)) {
    auto provenance = knownWithoutCoordinates();
    Value source = indices.getSource();
    if (isa<RankedTensorType>(source.getType()))
      return axisProvenance(source, axis);
    provenance.origins.push_back({absoluteCoordinateSource(source), axis});
    return provenance;
  }
  if (isa<IndexedAccessOpInterface>(operation)) {
    auto relation = indexRelation(operation);
    if (failed(relation))
      return {};
    auto provenance = knownWithoutCoordinates();
    for (const IndexTermFact &term : relation->terms) {
      if (!llvm::is_contained(term.resultAxes, axis))
        continue;
      if (term.kind == 3) {
        for (auto [indexAxis, resultAxis] : llvm::enumerate(term.indexAxes))
          if (resultAxis == axis)
            appendOrigins(provenance,
                          axisProvenance(term.operands.front(), indexAxis));
      } else if (term.kind == 4) {
        provenance.origins.push_back({absoluteCoordinateSource(term.operands.front()), 0});
      } else if ((term.kind == 0 || term.kind == 5) && term.sourceAxis) {
        if (isa<RankedTensorType>(relation->source.getType()))
          appendOrigins(provenance, axisProvenance(relation->source, *term.sourceAxis));
        else
          provenance.origins.push_back({relation->source, *term.sourceAxis});
      }
    }
    return provenance.origins.empty() ? ownAxis() : provenance;
  }
  auto projections = operandProjections(result);
  if (failed(projections))
    return ownAxis();
  auto provenance = knownWithoutCoordinates();
  for (const auto &projection : *projections) {
    Value source = operation->getOperand(projection.operandNumber);
    auto sourceType = cast<RankedTensorType>(source.getType());
    for (auto [sourceAxis, resultAxis] : llvm::enumerate(projection.resultAxes)) {
      if (resultAxis != axis)
        continue;
      // Logical unit expansion introduces a coordinate, not a forwarding edge.
      if (sourceType.getDimSize(sourceAxis) == 1 && tensor.getDimSize(axis) != 1)
        continue;
      appendOrigins(provenance, axisProvenance(source, sourceAxis));
    }
  }
  return provenance.origins.empty() ? ownAxis() : provenance;
}

FailureOr<SmallVector<LogicalWorksetFact, 4>>
CanonicalKernelAnalysis::logicalWorksets(func::FuncOp function) const {
  SmallVector<LogicalWorksetFact, 4> worksets;
  bool requiresWholeBody = false;
  for (Operation &operation : function.getBody().front().without_terminator()) {
    if (auto parallel = dyn_cast<ParallelOp>(operation)) {
      if (failed(collectLogicalWorkset(parallel, {}, {}, worksets)))
        return failure();
    } else {
      auto effects = getEffectsRecursively(&operation);
      requiresWholeBody |= !effects ||
          llvm::any_of(*effects, [](const auto &effect) {
            return !isa<MemoryEffects::Read>(effect.getEffect());
          });
    }
  }
  // Effects outside the parallel regions still belong to this invocation.
  // Keep their order and shared state until independence has been proven.
  if (worksets.empty() || requiresWholeBody) {
    worksets.clear();
    LogicalWorksetFact singleton;
    singleton.state = CanonicalFactState::Exact;
    singleton.body = &function.getBody().front();
    singleton.singleton = true;
    worksets.push_back(std::move(singleton));
  }
  return worksets;
}

LogicalBufferFact
CanonicalKernelAnalysis::logicalBuffer(Operation *operation) const {
  LogicalBufferFact fact;
  auto buffer = dyn_cast_or_null<BufferOp>(operation);
  if (!buffer)
    return fact;
  auto type = dyn_cast<BufferType>(buffer.getResult().getType());
  if (!type || type.getOriginId() == 0)
    return fact;
  fact.state = CanonicalFactState::Exact;
  fact.instanceIdentity = type.getOriginId();
  fact.hasFullInitialValue = static_cast<bool>(buffer.getInitial());
  for (Operation *parent = operation->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isa<ForOp, WhileOp>(parent)) {
      fact.scope = LogicalBufferScope::IterationPrivate;
      break;
    }
    if (isa<func::FuncOp>(parent))
      break;
  }
  return fact;
}

RegionSegmentFact
CanonicalKernelAnalysis::regionSegment(Operation *operation) const {
  RegionSegmentFact fact;
  auto structured = dyn_cast_or_null<StructuredOpInterface>(operation);
  if (!structured || !structured.getSummarizeRegion())
    return fact;
  auto axes = structured.getIterationAxes();
  if (axes.size() != 1) return fact;
  uint64_t axis = axes.front();
  ValueRange sources = structured.getSources();
  int64_t dimension = 0;
  for (Value source : sources) {
    auto tensor = dyn_cast<RankedTensorType>(source.getType());
    DenseI64ArrayAttr ids = tensor ? dimensionIDs(tensor)
                                   : DenseI64ArrayAttr();
    if (!tensor || axis >= static_cast<uint64_t>(tensor.getRank()) || !ids ||
        ids[axis] <= 0 || (dimension != 0 && ids[axis] != dimension))
      return fact;
    dimension = ids[axis];
  }
  auto node = operation->getAttrOfType<IntegerAttr>("intent.node");
  if (!node || node.getInt() < 0 || dimension <= 0)
    return fact;
  fact.state = CanonicalFactState::Exact;
  fact.dimensionIdentity = dimension;
  fact.operationIdentity = node.getInt();
  return fact;
}

LogicalResult CanonicalKernelAnalysis::verify() {
  LogicalResult result = success();
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    if (failed(logicalWorksets(function)))
      return failure();
    WalkResult constructionFacts = function.walk([&](Operation *operation) {
      if (isa<BufferOp>(operation) && !logicalBuffer(operation).isExact()) {
        operation->emitOpError(
            "canonical buffer has no exact lexical allocation fact");
        return WalkResult::interrupt();
      }
      if (isa<RegionFoldOp, RegionScanOp>(operation) &&
          !regionSegment(operation).isExact()) {
        operation->emitOpError(
            "canonical region operation has no exact segment source relation");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (constructionFacts.wasInterrupted())
      return failure();
  }
  module.walk([&](Operation *operation) {
    if (failed(result))
      return WalkResult::interrupt();
    if (isa<IndicesOp>(operation)) {
      CoordinateProvenance provenance =
          coordinateProvenance(operation->getResult(0));
      if (!provenance.known || provenance.origins.size() != 1) {
        operation->emitOpError(
            "canonical coordinate provenance is not exact at its source");
        result = failure();
        return WalkResult::interrupt();
      }
    }
    if (isa<IndexedAccessOpInterface>(operation) && failed(indexRelation(operation))) {
      operation->emitOpError(
          "canonical index relation analysis could not consume the typed relation");
      result = failure();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result;
}

} // namespace intent
