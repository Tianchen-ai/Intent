#include "Construction.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/IR/ProgramInterface.h"
#include "Intent/Dialect/GPU/Transforms/Mapping/ExecutionGroups.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace intent::kir_to_gpu {

static bool isCanonicalEffect(Operation *operation) {
  if (auto buffer = dyn_cast<BufferOp>(operation))
    return static_cast<bool>(buffer.getInitial());
  return isa<ViewStoreOp, BufferStoreOp, ScatterUniqueOp, ScatterReduceOp,
             AtomicStoreOp, AtomicRMWOp, AtomicCompareExchangeOp>(operation);
}

static ArrayAttr observableEffectOrigins(func::FuncOp function) {
  SmallVector<Attribute> origins;
  function.walk([&](Operation *operation) {
    if (!isCanonicalEffect(operation))
      return;
    if (auto node = operation->getAttrOfType<IntegerAttr>("intent.node"))
      origins.push_back(node);
  });
  return ArrayAttr::get(function.getContext(), origins);
}

struct ParallelWorkset {
  intent::ParallelOp operation;
  Block *body = nullptr;
  bool singleton = false;
  SmallVector<IterationAxis> axes;
  SmallVector<BlockArgument> coordinateArguments;
  SmallVector<PhysicalExprAttr> launchExtents;
  PhysicalExprAttr launchLength;
};

static LogicalResult finalizeParallelWorkset(
    ParallelWorkset &workset,
    func::FuncOp function,
    CanonicalKernelAnalysis &canonicalAnalysis) {
  MLIRContext *context = workset.operation.getContext();
  for (const IterationAxis &axis : workset.axes) {
    FailureOr<PhysicalExprAttr> start =
        launchExpression(axis.start, canonicalAnalysis, function);
    FailureOr<PhysicalExprAttr> stop =
        launchExpression(axis.stop, canonicalAnalysis, function);
    FailureOr<PhysicalExprAttr> step =
        axis.step
            ? launchExpression(axis.step, canonicalAnalysis, function)
            : FailureOr<PhysicalExprAttr>(
                  expression(context, PhysicalExprKind::Constant, 1));
    if (failed(start) || failed(stop) || failed(step))
      return axis.source.getDefiningOp()->emitOpError(
          "parallel workset bound is not a launch-visible typed expression");
    PhysicalExprAttr distance = binaryExpression(
        context, PhysicalExprKind::Subtract, *stop, *start);
    workset.launchExtents.push_back(binaryExpression(
        context, PhysicalExprKind::CeilDiv, distance, *step));
  }
  workset.launchLength = workset.launchExtents.front();
  for (PhysicalExprAttr extent : llvm::drop_begin(workset.launchExtents))
    workset.launchLength = binaryExpression(
        context, PhysicalExprKind::Multiply, workset.launchLength, extent);
  if (workset.coordinateArguments.size() != workset.axes.size())
    return workset.operation.emitOpError(
        "parallel workset coordinates do not match its domain product");
  return success();
}

static bool capturesScanPrefixAxis(
    const LogicalWorksetFact &workset,
    CanonicalKernelAnalysis &analysis,
    intent::ParallelOp &scope) {
  if (workset.singleton || !workset.parallel || !workset.body)
    return false;
  SmallVector<CoordinateOrigin> worksetCoordinates;
  for (BlockArgument coordinate : workset.coordinates) {
    CoordinateProvenance provenance = analysis.coordinateProvenance(coordinate);
    if (provenance.known)
      worksetCoordinates.append(provenance.origins.begin(),
                                provenance.origins.end());
  }
  bool captured = false;
  workset.body->walk([&](intent::GatherOp gather) {
    SmallVector<Value> pending{gather.getSource()};
    llvm::DenseSet<Value> visited;
    SmallVector<intent::ScanOp> scans;
    while (!pending.empty()) {
      Value value = pending.pop_back_val();
      if (!visited.insert(value).second)
        continue;
      if (auto scan = value.getDefiningOp<intent::ScanOp>()) {
        if (!workset.parallel->isProperAncestor(scan))
          scans.push_back(scan);
        continue;
      }
      Operation *producer = value.getDefiningOp();
      auto tensor = dyn_cast<RankedTensorType>(value.getType());
      if (!producer || !tensor ||
          !isa<intent::UnaryOp, intent::BinaryOp, intent::CompareOp,
               intent::SelectOp, intent::CastOp, intent::BitcastOp>(producer))
        continue;
      for (Value input : producer->getOperands()) {
        auto operand = dyn_cast<RankedTensorType>(input.getType());
        if (operand && operand.getShape() == tensor.getShape() &&
            dimensionIds(operand) == dimensionIds(tensor))
          pending.push_back(input);
      }
    }
    FailureOr<IndexRelationFact> relation = analysis.indexRelation(gather);
    if (failed(relation))
      return WalkResult::advance();
    for (intent::ScanOp scan : scans) {
      for (const IndexTermFact &term : relation->terms) {
        if (!term.sourceAxis || *term.sourceAxis != scan.getAxis() ||
            !term.coordinate.known)
          continue;
        if (llvm::any_of(term.coordinate.origins, [&](const auto &origin) {
              return llvm::is_contained(worksetCoordinates, origin);
            })) {
          auto enclosing = gather->getParentOfType<intent::ParallelOp>();
          while (enclosing && !enclosing->isProperAncestor(scan))
            enclosing = enclosing->getParentOfType<intent::ParallelOp>();
          if (!captured)
            scope = enclosing;
          else if (!scope || !enclosing)
            scope = {};
          else if (enclosing->isProperAncestor(scope))
            scope = enclosing;
          captured = true;
          break;
        }
      }
    }
    return WalkResult::advance();
  });
  return captured;
}

LogicalResult constructGPUProgram(
    ModuleOp module,
    const GPUCapabilities &capabilities,
    func::FuncOp function) {
  Attribute sourceOrigin = function->getAttr("intent.source");
  if (!sourceOrigin)
    return function.emitError(
        "canonical kernel has no stable source origin for physical construction");
  CanonicalKernelAnalysis canonicalAnalysis(module);
  FailureOr<SmallVector<LogicalWorksetFact, 4>> logicalWorksets =
      canonicalAnalysis.logicalWorksets(function);
  if (failed(logicalWorksets))
    return failure();
  SmallVector<intent::ParallelOp> prefixScopes;
  bool wholeBodyPrefix = false;
  for (const LogicalWorksetFact &workset : *logicalWorksets) {
    intent::ParallelOp scope;
    if (!capturesScanPrefixAxis(workset, canonicalAnalysis, scope))
      continue;
    if (!scope) {
      wholeBodyPrefix = true;
      break;
    }
    SmallVector<IterationAxis> axes;
    for (auto [domain, coordinate] :
         llvm::zip(workset.domains, workset.coordinates)) {
      if (failed(collectIterationAxes(domain, axes)))
        return failure();
      if (coordinate == scope.getBody().front().getArguments().back())
        break;
    }
    if (llvm::any_of(axes, [&](const IterationAxis &axis) {
          return failed(launchExpression(axis.start, canonicalAnalysis, function)) ||
                 failed(launchExpression(axis.stop, canonicalAnalysis, function)) ||
                 (axis.step && failed(launchExpression(axis.step, canonicalAnalysis, function)));
        })) {
      wholeBodyPrefix = true;
      break;
    }
    if (llvm::any_of(prefixScopes, [&](intent::ParallelOp previous) {
          return previous->isAncestor(scope);
        }))
      continue;
    llvm::erase_if(prefixScopes, [&](intent::ParallelOp previous) {
      return scope->isProperAncestor(previous);
    });
    prefixScopes.push_back(scope);
  }
  if (wholeBodyPrefix) {
    LogicalWorksetFact singleton;
    singleton.state = CanonicalFactState::Exact;
    singleton.body = &function.getBody().front();
    singleton.singleton = true;
    logicalWorksets->clear();
    logicalWorksets->push_back(std::move(singleton));
  } else if (!prefixScopes.empty()) {
    // The prefix and its consumers share one instance; enclosing independent
    // coordinates remain launch axes. A scope also owns all sibling worksets.
    SmallVector<LogicalWorksetFact, 4> grouped;
    llvm::SmallPtrSet<Operation *, 4> emitted;
    for (LogicalWorksetFact workset : *logicalWorksets) {
      intent::ParallelOp scope;
      for (intent::ParallelOp candidate : prefixScopes)
        if (workset.parallel && candidate->isAncestor(workset.parallel)) {
          scope = candidate;
          break;
        }
      if (!scope) {
        grouped.push_back(std::move(workset));
        continue;
      }
      if (!emitted.insert(scope.getOperation()).second)
        continue;
      Block &body = scope.getBody().front();
      auto first = llvm::find(workset.coordinates, body.getArgument(0));
      size_t offset = std::distance(workset.coordinates.begin(), first);
      size_t count = offset + body.getNumArguments();
      if (count > workset.coordinates.size() || count > workset.domains.size() ||
          !llvm::equal(ArrayRef<BlockArgument>(workset.coordinates).slice(
                           offset, body.getNumArguments()), body.getArguments()))
        return scope.emitOpError(
            "captured prefix scope has no matching workset coordinate prefix");
      workset.parallel = scope;
      workset.body = &body;
      workset.domains.resize(count);
      workset.coordinates.resize(count);
      grouped.push_back(std::move(workset));
    }
    *logicalWorksets = std::move(grouped);
  }
  SmallVector<ParallelWorkset> worksets;
  for (const LogicalWorksetFact &fact : *logicalWorksets) {
    if (!fact.isExact() || !fact.body)
      return function.emitError(
          "canonical workset analysis produced an incomplete execution group");
    ParallelWorkset workset;
    workset.operation = dyn_cast_or_null<intent::ParallelOp>(fact.parallel);
    workset.body = fact.body;
    workset.singleton = fact.singleton;
    workset.coordinateArguments.append(fact.coordinates.begin(),
                                       fact.coordinates.end());
    for (Value domainValue : fact.domains) {
      if (failed(collectIterationAxes(domainValue, workset.axes)))
        return function.emitError(
            "canonical workset axis has no typed domain/subregion bounds");
    }
    if (workset.singleton) {
      workset.launchExtents.push_back(
          expression(function.getContext(), PhysicalExprKind::Constant, 1));
      workset.launchLength = workset.launchExtents.front();
    } else if (failed(finalizeParallelWorkset(workset, function, canonicalAnalysis))) {
        return failure();
    }
    worksets.push_back(std::move(workset));
  }

  MLIRContext *context = module.getContext();
  OpBuilder builder(context);
  FailureOr<PhysicalABI> abi = buildPhysicalABI(function, builder);
  if (failed(abi))
    return failure();
  PhysicalExprAttr totalLength = worksets.front().launchLength;
  for (const ParallelWorkset &workset : llvm::drop_begin(worksets))
    totalLength = binaryExpression(context, PhysicalExprKind::Add, totalLength,
                                   workset.launchLength);
  auto capabilityAttr = gpu::CapabilitiesAttr::get(
      context, capabilities.computeUnits, capabilities.sharedMemoryPerUnit,
      capabilities.maxDynamicSharedMemoryPerBlock,
      capabilities.registersPerUnit,
      capabilities.maxThreadsPerBlock, capabilities.computeCapabilityMajor,
      capabilities.computeCapabilityMinor,
      capabilities.singleToDoublePrecisionPerfRatio, capabilities.matrixUnits,
      capabilities.dynamicVectorWidth, capabilities.nativeTupleReductions,
      capabilities.nativeTupleReductionRequiresConstantIdentity,
      capabilities.nativeFragmentGather);
  SmallVector<NamedAttribute> functionAttrs{
      builder.getNamedAttr(gpu::kernelAttr, builder.getUnitAttr()),
      builder.getNamedAttr(interfaceAttr, abi->interface),
      builder.getNamedAttr(gpu::parametersAttr, builder.getArrayAttr({})),
      builder.getNamedAttr(gpu::capabilitiesAttr, capabilityAttr),
      builder.getNamedAttr(gpu::programSpaceAttr,
                           builder.getArrayAttr({totalLength})),
      builder.getNamedAttr(gpu::gridRankAttr, builder.getI64IntegerAttr(1)),
      builder.getNamedAttr(gpu::effectOriginsAttr,
                           observableEffectOrigins(function)),
      builder.getNamedAttr(gpu::originAttr, sourceOrigin),
  };
  builder.setInsertionPointAfter(function);
  auto physical = builder.create<func::FuncOp>(
      function.getLoc(), ("__intent_gpu_" + function.getName()).str(),
      FunctionType::get(context, abi->arguments, {}), functionAttrs,
      abi->argumentAttrs);
  Block *entry = physical.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  llvm::DenseMap<StringAttr, Value> parameterValues;
  SmallVector<Value> sourceArguments;
  sourceArguments.reserve(abi->physicalArgumentForSource.size());
  for (std::optional<unsigned> physicalIndex : abi->physicalArgumentForSource)
    sourceArguments.push_back(physicalIndex ? Value(entry->getArgument(*physicalIndex)) : Value{});
  llvm::DenseMap<int64_t, Value> dimensionValues;
  unsigned metadataOffset = abi->interface.getArguments().size();
  for (auto [offset, dimension] : llvm::enumerate(abi->dimensionOrder))
    dimensionValues[dimension] = entry->getArgument(metadataOffset + offset);

  llvm::DenseMap<Value, Value> values;
  for (auto [logical, physicalView] : llvm::zip(
           function.getBody().front().getArguments(), sourceArguments))
    if (physicalView) values[logical] = physicalView;
  for (Operation &operation : function.getBody().front()) {
    if (auto constant = dyn_cast<intent::ConstantOp>(operation)) {
      FailureOr<Value> value = gpu::materializeScalarConstant(
          builder, constant.getLoc(), constant.getValue(),
          constant.getResult().getType());
      if (failed(value))
        return constant.emitOpError(
            "constant cannot be represented by its physical result type");
      values[constant.getResult()] = *value;
      if (Operation *target = value->getDefiningOp())
        if (Attribute node = operation.getAttr("intent.node"))
          target->setAttr(gpu::originAttr, node);
    }
  }
  Value pid = builder.create<gpu::ProgramIdOp>(function.getLoc(),
                                               builder.getIndexType(), 0);
  Value one = builder.create<arith::ConstantIndexOp>(function.getLoc(), 1);
  PhysicalExprAttr launchOffset =
      expression(context, PhysicalExprKind::Constant, 0);
  ScalarRegionLowering rootLowering(builder, values, sourceArguments,
                                    dimensionValues, parameterValues,
                                    canonicalAnalysis, physical);
  auto formWorksetCoordinate = [&](OpBuilder &nested, Location location,
                                   Value source, Value coordinate,
                                   Value step,
                                   unsigned worksetAxis) -> Value {
    auto domainType = dyn_cast<intent::DomainType>(source.getType());
    auto regionType = dyn_cast<intent::RegionType>(source.getType());
    uint64_t sourceId = domainType ? domainType.getOriginId()
                                  : regionType.getSourceId();
    std::optional<int64_t> dimension =
        sourceExtentDimension(source);
    if (!dimension || *dimension <= 0) {
      source.getDefiningOp()->emitOpError(
          "parallel workset coordinate has no logical dimension identity");
      return {};
    }
    auto mapped = nested.create<gpu::WorksetCoordinateOp>(
        location, nested.getIndexType(), coordinate, step,
        sourceId,
        /*sourceAxis=*/0, /*sourceRank=*/1, *dimension);
    mapped->setAttr(gpu::worksetAxisAttr,
                    nested.getI64IntegerAttr(worksetAxis));
    return mapped.getResult();
  };
  bool dispatchLoweringFailed = false;
  for (auto [groupIndex, workset] : llvm::enumerate(worksets)) {
    SmallVector<Value> runtimeExtents;
    SmallVector<Value> starts;
    SmallVector<Value> steps;
    for (const IterationAxis &axis : workset.axes) {
      FailureOr<Value> start = rootLowering.lowerIndexValue(axis.start);
      FailureOr<Value> step =
          axis.step
              ? rootLowering.lowerIndexValue(axis.step)
              : FailureOr<Value>(one);
      if (failed(start) || failed(step))
        return axis.source.getDefiningOp()->emitOpError(
            "parallel workset runtime bounds are unavailable");
      starts.push_back(*start);
      steps.push_back(*step);
    }
    for (auto [axis, extent] : llvm::enumerate(workset.launchExtents)) {
      Location location = workset.singleton ? function.getLoc()
                                            : workset.axes[axis].source.getLoc();
      runtimeExtents.push_back(builder.create<gpu::PhysicalExprOp>(
          location, builder.getIndexType(), extent));
    }
    PhysicalExprAttr launchEnd = binaryExpression(
        context, PhysicalExprKind::Add, launchOffset, workset.launchLength);
    auto lowerGroup = [&](OpBuilder &nested, Value linear) -> LogicalResult {
      SmallVector<Attribute> launchExtents(workset.launchExtents.begin(),
                                           workset.launchExtents.end());
      Location worksetLocation = workset.singleton ? function.getLoc()
                                                   : workset.operation.getLoc();
      auto group = gpu::createExecutionGroup(
          nested, worksetLocation, linear, runtimeExtents,
          nested.getArrayAttr(launchExtents), nested.getDenseI64ArrayAttr(
              SmallVector<int64_t>(runtimeExtents.size(), static_cast<int64_t>(
                  workset.singleton ? gpu::CoordinateRole::Unspecified
                                    : gpu::CoordinateRole::Workset))),
          groupIndex, launchOffset, workset.launchLength);
      OpBuilder body = OpBuilder::atBlockBegin(&group.getBody().front());
      auto childValues = rootLowering.mapping();
      Block &sourceBlock = *workset.body;
      for (auto [axis, localCoordinate] :
           llvm::enumerate(group.getCoordinates())) {
        if (workset.singleton)
          break;
        Value scaled = createBinary(body, worksetLocation,
                                    body.getIndexType(), localCoordinate,
                                    steps[axis], BinaryOperator::Multiply);
        Value coordinate = createBinary(body, worksetLocation,
                                        body.getIndexType(), starts[axis],
                                        scaled, BinaryOperator::Add);
        childValues[workset.coordinateArguments[axis]] = formWorksetCoordinate(
            body, worksetLocation, workset.axes[axis].source, coordinate, steps[axis],
            axis);
      }
      ScalarRegionLowering lowering(body, std::move(childValues),
                                    sourceArguments, dimensionValues,
                                    parameterValues, canonicalAnalysis, physical);
      return lowering.lowerWorksetBlock(sourceBlock);
    };
    if (worksets.size() == 1) {
      if (failed(lowerGroup(builder, pid))) return failure();
      launchOffset = launchEnd;
      continue;
    }
    Value runtimeOffset = builder.create<gpu::PhysicalExprOp>(
        function.getLoc(), builder.getIndexType(), launchOffset);
    Value segmentEnd = builder.create<gpu::PhysicalExprOp>(
        function.getLoc(), builder.getIndexType(), launchEnd);
    Value afterOffset = createCompare(builder, function.getLoc(),
                                      builder.getI1Type(), pid, runtimeOffset,
                                      ComparePredicate::Ge);
    Value beforeEnd = createCompare(builder, function.getLoc(), builder.getI1Type(),
                                    pid, segmentEnd, ComparePredicate::Lt);
    Value active = createBinary(builder, function.getLoc(), builder.getI1Type(),
                                afterOffset, beforeEnd,
                                BinaryOperator::LogicalAnd);
    Location worksetLocation = workset.singleton ? function.getLoc()
                                                 : workset.operation.getLoc();
    auto dispatch = builder.create<scf::IfOp>(
        worksetLocation, active,
        [&](OpBuilder &nested, Location location) {
          Value local = createBinary(nested, location, nested.getIndexType(), pid,
                                     runtimeOffset, BinaryOperator::Subtract);
          if (failed(lowerGroup(nested, local))) {
            dispatchLoweringFailed = true;
            return;
          }
          nested.create<scf::YieldOp>(location);
        });
    launchOffset = launchEnd;
  }
  if (dispatchLoweringFailed)
    return failure();
  builder.create<func::ReturnOp>(function.getLoc());
  if (failed(gpu::closeValueRelations(physical, gpu::ValueRelationScope::Pointwise)))
    return physical.emitError(
        "initial physical value relations are incomplete");
  if (failed(gpu::closeValueRelations(physical, gpu::ValueRelationScope::Contracts)))
    return physical.emitError(
        "initial physical contract relations are incomplete");
  function.erase();
  module->setAttr("intent_gpu.physical", builder.getUnitAttr());
  return success();
}

} // namespace intent::kir_to_gpu
