#include "Pointwise.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Transforms/Control/Predication.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include <algorithm>

using namespace mlir;

namespace intent::gpu::pointwise {

void PointwiseRewrite::refreshProgramFacts() {
  allRanges.clear();
  kernel.walk([&](MakeRangeOp range) { allRanges.push_back(range); });
  uses = classifyStructuredRanges(kernel, allRanges, laneReductions);
  writeEffects = readWriteEffects(kernel);
  writeCoordinates.clear(); writeOperations.clear();
  for (const WriteEffectFacts &effect : writeEffects) {
    writeCoordinates.emplace_back(effect.coordinates.begin(), effect.coordinates.end());
    writeOperations.push_back(effect.operation);
  }
  llvm::SmallPtrSet<Operation *, 32> live;
  for (MakeRangeOp range : allRanges) live.insert(range);
  auto retainCurrent = [&](auto &ranges) {
    SmallVector<Operation *> stale;
    for (Operation *range : ranges)
      if (!live.contains(range)) stale.push_back(range);
    for (Operation *range : stale) ranges.erase(range);
  };
  retainCurrent(internalTraversalRanges);
  retainCurrent(promotedRanges);
  retainCurrent(reuseTraversalRanges);
  retainCurrent(writeTraversalRanges);
  retainCurrent(reductionCaptureWritebackRanges);
  retainCurrent(boundedWritebackRanges);
  retainCurrent(forcedContractionRanges);
  retainCurrent(forcedReductionRanges);
  retainCurrent(retainedCartesianRanges);
  retainCurrent(independentContractionRanges);
  retainCurrent(dependentResourceRanges);
  retainCurrent(retainedGatherRanges);
  internalTraversalRanges.insert(uses.internalTraversalRanges.begin(), uses.internalTraversalRanges.end());
  for (Operation *range : promotedRanges) internalTraversalRanges.erase(range);
  llvm::erase_if(dynamicRanges, [&](MakeRangeOp range) {
    return !live.contains(range) || range.getResult().use_empty();
  });
}

void PointwiseRewrite::selectRanges(
    llvm::function_ref<bool(MakeRangeOp, PhysicalExprAttr)> selectedStatic) {
  dynamicRanges.clear();
  for (MakeRangeOp range : allRanges) {
    if (range->getParentOfType<RegionFoldOp>() || range->getParentOfType<RegionScanOp>()) continue;
    auto fragment = cast<FragmentType>(range.getResult().getType());
    auto extent = cast<PhysicalExprAttr>(fragment.getShape()[0]);
    if (!hasCompileTimeExtent(range.getExtent()) ||
        PhysicalProgramAnalysis(kernel).axisRealization(range.getResult(), 0).constructionScalarSeed ||
        selectedStatic(range, extent)) dynamicRanges.push_back(range);
  }
}

void PointwiseRewrite::filterOwnershipRanges() {
  llvm::erase_if(dynamicRanges, [&](MakeRangeOp range) {
    auto fragment = cast<FragmentType>(range.getResult().getType());
    auto extent = cast<PhysicalExprAttr>(fragment.getShape()[0]);
    auto parent = querySubregionParentDimension(range);
    bool launchParent = succeeded(parent) && !hasAccessDependentSubregionBounds(kernel, range) &&
                        succeeded(dimensionArgument(kernel, static_cast<uint64_t>(*parent)));
    return range->hasAttr(sourceSubregionAttr) &&
           extent.getKind() == PhysicalExprKind::Constant && !launchParent;
  });
}

void PointwiseRewrite::appendWritebackRanges() {
  for (MakeRangeOp range : allRanges)
    if (reuseTraversalRanges.contains(range) && !llvm::is_contained(dynamicRanges, range))
      dynamicRanges.push_back(range);
}

void PointwiseRewrite::appendOwnershipOccurrences() {
  for (auto &entry : occurrenceRoots) {
    auto range = cast<MakeRangeOp>(entry.first);
    if (!llvm::is_contained(dynamicRanges, range)) dynamicRanges.push_back(range);
  }
}

void PointwiseRewrite::selectOwnershipBindings() {
  for (MakeRangeOp range : dynamicRanges) {
    if (internalTraversalRanges.contains(range)) {
      bindings.push_back({range, false, false, true});
      continue;
    }
    if (hasPointwiseOwnership(range)) bindings.push_back({range, true, true});
  }
}

void PointwiseRewrite::selectLocalBindings() {
  for (MakeRangeOp range : dynamicRanges) {
    bool internal = internalTraversalRanges.contains(range), reuse = reuseTraversalRanges.contains(range);
    bool ownership = hasPointwiseOwnership(range);
    if ((internal && !reuse) || (!ownership && !internal && !reuse)) continue;
    bindings.push_back({range, (ownership && !internal) || reuse, false});
  }
}

LogicalResult PointwiseRewrite::reuseMapping() {
  auto coordinates = readMappingCoordinates(kernel, mapping);
  if (failed(coordinates)) return failure();
  for (auto &[axis, ranges] : axes) {
    if (failed(reconcileCoordinate(ranges.front(), axis, *coordinates))) return failure();
    if (!parameters.lookup(axis)) return ranges.front().emitOpError("pointwise mapping lost its blocking parameter");
    auto mapped = coordinates->axes.find(axis);
    if (mapped != coordinates->axes.end()) {
      auto launch = cast<PhysicalExprAttr>(
          mapping.getLaunchExtents()[mapped->second]);
      if (launch.getKind() == PhysicalExprKind::CeilDiv &&
          launch.getOperands().size() == 2) {
        auto extent = cast<PhysicalExprAttr>(launch.getOperands()[1]);
        if (extent.getKind() == PhysicalExprKind::Minimum)
          ownershipExtents[axis] = extent;
      }
    }
  }
  tileCoordinates = std::move(coordinates->tiles);
  return success();
}

LogicalResult PointwiseRewrite::finishOwnership() {
  if (failed(alignHistogramOutputOwnership(kernel))) return failure();
  return finalizeValues();
}

LogicalResult PointwiseRewrite::finishBlocking() {
  if (failed(realizeOwnedHistograms(kernel))) return failure();
  return finalizeValues();
}

LogicalResult PointwiseRewrite::ownership() {
  if (failed(liftWorksets()) || failed(prepareCoverage())) return failure();
  refreshProgramFacts();
  selectRanges([](MakeRangeOp range, PhysicalExprAttr extent) {
    return range.getExtent().getDefiningOp<arith::ConstantIndexOp>() &&
           (range->hasAttr(worksetCoordinateRangeAttr) ||
            (extent.getKind() == PhysicalExprKind::Constant && extent.getValue() > 1));
  });
  if (failed(selectWritebackCandidates(false))) return failure();
  filterOwnershipRanges();
  if (dynamicRanges.empty()) return finishOwnership();
  if (failed(prepareAxisRelations(true))) return failure();
  refreshProgramFacts();
  appendOwnershipOccurrences();
  if (failed(resolveOwnershipDependencies())) return failure();
  promoteOwnershipRanges();
  if (failed(retainGatherSources())) return failure();
  refreshProgramFacts();
  selectOwnershipBindings();
  if (failed(bindAxes()) || failed(chooseOwnership())) return failure();
  if (dynamicRanges.empty()) return finishOwnership();
  if (failed(mapOwnership()) || failed(materializeRanges())) return failure();
  return finishOwnership();
}

LogicalResult PointwiseRewrite::blocking() {
  if (failed(prepareCoverage())) return failure();
  refreshProgramFacts();
  selectRanges([](MakeRangeOp range, PhysicalExprAttr extent) {
    auto fixed = constantPhysicalExpression(extent);
    return fixed && *fixed > 0 && !llvm::isPowerOf2_64(*fixed) && isUnitStepRange(range);
  });
  if (failed(selectWritebackCandidates(true))) return failure();
  appendWritebackRanges();
  if (failed(materializeFixedRanges())) return failure();
  refreshProgramFacts();
  llvm::erase_if(dynamicRanges, [&](MakeRangeOp range) {
    return uses.structuredTraversalRanges.contains(range) &&
           !writeTraversalRanges.contains(range) && !reuseTraversalRanges.contains(range);
  });
  if (failed(realizeWritebacks())) return failure();
  if (dynamicRanges.empty()) return finishBlocking();
  if (failed(prepareAxisRelations(false))) return failure();
  refreshProgramFacts();
  if (failed(resolveOwnershipDependencies()) || failed(retainGatherSources())) return failure();
  refreshProgramFacts();
  if (failed(coverLocalRanges())) return failure();
  selectLocalBindings();
  if (failed(bindAxes()) || failed(reuseMapping()) || failed(materializeRanges())) return failure();
  return finishBlocking();
}
LogicalResult PointwiseRewrite::bindAxes() {
  for (const RangeBinding &request : bindings) {
    MakeRangeOp range = request.range;
    if (request.internalOnly) {
      auto parameter = queryBlockingParameter(kernel, range);
      auto axis = succeeded(parameter) ? parameterAxis(*parameter)
                                      : FailureOr<Attribute>(failure());
      if (succeeded(axis)) internalAxes.insert(*axis);
      else if (auto dimension = rangeDimension(range); succeeded(dimension))
        internalAxes.insert(dimensionAxisKey(module.getContext(), *dimension));
      continue;
    }
    FailureOr<ParameterAttr> parameter =
        queryOwnershipBlockingParameter(kernel, range);
    MakeRangeOp occurrenceRoot = occurrenceRoots.lookup(range.getOperation());
    if (occurrenceRoot) {
      ParameterAttr selected = lookupParameter(kernel, occurrenceParameters.lookup(occurrenceRoot.getOperation()));
      if (!selected) {
        FailureOr<ParameterAttr> existing =
            queryOwnershipBlockingParameter(kernel, occurrenceRoot);
        if (succeeded(existing)) {
          PhysicalParameterBinding binding = queryParameterBinding(*existing);
          // Repeated Cartesian axes have distinct occurrence classes even
          // when their logical extents share the same dimension identity.
          auto dimension = ownershipDimension(kernel, occurrenceRoot);
          bool sharedDimension = binding.isExact() && !binding.source &&
              binding.dimension && succeeded(dimension) &&
              *binding.dimension == static_cast<int64_t>(*dimension) &&
              !independentCartesianDimensions.contains(*dimension) &&
              positionalOccurrences.contains(range.getOperation()) &&
              positionalOccurrences.contains(occurrenceRoot.getOperation());
          if (sharedDimension ||
              (binding.isExact() && binding.source &&
               *binding.source == sourceAxisIdentity(occurrenceRoot)))
            selected = *existing;
        }
      }
      parameter = selected ? FailureOr<ParameterAttr>(selected)
                           : FailureOr<ParameterAttr>(failure());
    }
    bool requiresBlockingParameter = request.requiresTile;
    if (requiresBlockingParameter && succeeded(parameter) &&
        (parameter->getCategory() ==
             ParameterCategory::Coverage ||
         parameter->isDeferred()))
      parameter = failure();
    if (failed(parameter) && requiresBlockingParameter) {
      const bool worksetRange = range->hasAttr(worksetCoordinateRangeAttr);
      Value logicalExtentValue = range.getExtent();
      if (worksetRange) {
        auto coordinate = range.getStart().getDefiningOp<WorksetCoordinateOp>();
        auto position = coordinate
                            ? coordinate->getAttrOfType<IntegerAttr>(worksetAxisAttr)
                            : IntegerAttr();
        if (!position || position.getInt() < 0 ||
            static_cast<size_t>(position.getInt()) >= mapping.getExtents().size())
          return range.emitOpError(
              "workset ownership has no logical cardinality for blocking");
        logicalExtentValue = mapping.getExtents()[position.getInt()];
      }
      auto logicalExtent = dyn_cast_or_null<IntegerAttr>(
          UniformValueAnalysis(describeUniformValue).evaluate(logicalExtentValue));
      auto fragment = cast<FragmentType>(range.getResult().getType());
      auto physicalExtent = cast<PhysicalExprAttr>(fragment.getShape()[0]);
      const bool dynamicSubregion = range->hasAttr(sourceSubregionAttr);
      FailureOr<uint64_t> sourceDimension =
          ownershipDimension(kernel, range);
      const bool launchVisibleDimension =
          succeeded(sourceDimension) &&
          succeeded(dimensionArgument(kernel, *sourceDimension));
      PhysicalExprAttr derivedExtent = queryLaunchRangeExtent(range);
      const bool launchVisibleExtent =
          worksetRange ? !logicalExtent
                       : launchVisibleDimension ||
                             (derivedExtent && !isCompileTimePhysicalExpr(derivedExtent));
      ParameterCategory category = ParameterCategory::Pointwise;
      if (succeeded(sourceDimension)) {
        auto structured = structuredOwnershipCategories.find(*sourceDimension);
        if (structured != structuredOwnershipCategories.end())
          category = structured->second;
      }
      if (category == ParameterCategory::Pointwise &&
          contractFreeAxisFacts(kernel, range).regionContraction)
        category = ParameterCategory::RegionContraction;
      if (dynamicSubregion ||
          launchVisibleExtent ||
          (logicalExtent && physicalExtent.getKind() ==
                                PhysicalExprKind::Constant)) {
        SmallVector<int64_t> candidates;
        if (worksetRange ? logicalExtent && logicalExtent.getInt() == 1
                         : isProvablySingletonLogicalRange(range)) {
          candidates.push_back(1);
        } else if (dynamicSubregion) {
          candidates.assign({1, 2, 4, 8, 16, 32, 64, 128, 256});
        } else if (launchVisibleExtent) {
          candidates.assign({1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
                             2048, 4096});
        } else {
          // Ownership tail validity also permits a padded pointwise tile.
          // Keep the enclosing power-of-two extent available to the profiles;
          // a logical row width must not cap an otherwise legal physical tile.
          uint64_t paddedExtent = llvm::PowerOf2Ceil(
              static_cast<uint64_t>(std::max<int64_t>(logicalExtent.getInt(), 1)));
          for (int64_t candidate : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
                                    2048, 4096, 8192, 16384})
            if (category == ParameterCategory::RegionContraction ||
                static_cast<uint64_t>(candidate) <= paddedExtent)
              candidates.push_back(candidate);
          if (logicalExtent.getInt() > 0 &&
              !llvm::is_contained(candidates, logicalExtent.getInt()))
            candidates.push_back(logicalExtent.getInt());
          llvm::sort(candidates);
        }
        OpBuilder builder(&kernel.getBody().front(),
                          kernel.getBody().front().begin());
        PhysicalSourceAxis source{range.getSourceId(), range.getSourceAxis(),
                              range.getDerived()};
        if (occurrenceRoot)
          source = sourceAxisIdentity(occurrenceRoot);
        auto schema = ParameterAttr::get(
            module.getContext(),
            builder.getStringAttr(launchVisibleDimension && !occurrenceRoot
                                      ? ("FRAGMENT_D" +
                                         Twine(*sourceDimension))
                                            .str()
                                      : ("FRAGMENT_S" + Twine(source.sourceId) +
                                         "_A" + Twine(source.sourceAxis))
                                            .str()), builder.getIndexType(),
            ParameterRole::OwnershipN,
            category,
            pointwiseElementBitWidth,
            DenseI64ArrayAttr::get(module.getContext(), candidates),
            ConfigurationBindingPhase::Shared,
            ParameterBindingAttr::get(
                module.getContext(), succeeded(sourceDimension)
                    ? builder.getI64IntegerAttr(*sourceDimension) : IntegerAttr(),
                !launchVisibleDimension || occurrenceRoot
                    ? PhysicalSourceAttr::get(module.getContext(), source.sourceId,
                                              source.sourceAxis, source.derived)
                    : PhysicalSourceAttr(),
                {}, {}, false, false));
        if (failed(declareParameter(kernel, schema))) return failure();
        parameter = schema;
      }
    }
    FailureOr<Attribute> axis =
        failed(parameter) ? FailureOr<Attribute>(failure())
                          : parameterAxis(*parameter);
    if (failed(parameter) || failed(axis)) {
      InFlightDiagnostic diagnostic = range.emitOpError(
          "dynamic pointwise range has no canonical blocking dimension");
      diagnostic << "; source_id=" << range.getSourceId() << ", fragment="
                 << range.getResult().getType();
      if (succeeded(parameter))
        diagnostic << ", parameter="
                   << parameter->getName().getValue();
      return failure();
    }
    if (occurrenceRoot)
      occurrenceParameters[occurrenceRoot.getOperation()] = parameter->getReference();
    // Repeated Cartesian axes use their proven occurrence classes. A unique
    // logical dimension can still bind connected pointwise values, while a
    // range-local source retains its own traversal relation.
    if (occurrenceRoot) {
      if (failed(retargetSourceExtent(range.getResult(), sourceAxisIdentity(range),
                                      fragmentExtent(*parameter))))
        return failure();
    } else if (FailureOr<uint64_t> dimension = rangeDimension(range);
               succeeded(dimension)) {
      if (failed(retargetDimensionExtent(range.getResult(), *dimension,
                                        fragmentExtent(*parameter))))
        return failure();
    } else if (FailureOr<PhysicalSourceAxis> source = axisSource(*axis);
               succeeded(source)) {
      if (failed(retargetSourceExtent(range.getResult(), *source,
                                      fragmentExtent(*parameter))))
        return failure();
    } else {
      return range.emitOpError("blocking parameter has no typed axis binding");
    }
    auto found = parameters.find(*axis);
    if (found != parameters.end() && found->second != parameter->getReference())
      return range.emitOpError("one physical axis has multiple blocking parameters")
             << "; axis=" << *axis << ", previous="
             << found->second.getName().getValue()
             << ", current=" << parameter->getName().getValue()
             << ", source_id=" << range.getSourceId();
    parameters[*axis] = parameter->getReference();
    axes[*axis].push_back(range);
    if (internalTraversalRanges.contains(range.getOperation()))
      internalAxes.insert(*axis);
    if (request.programOwner)
      ownershipAxes.insert(*axis);
  }
  return success();
}
LogicalResult PointwiseRewrite::materializeRanges() {
  llvm::DenseMap<Value, Value> rangePredicates;
  for (MakeRangeOp range : dynamicRanges) {
    OpBuilder builder(range);
    FailureOr<ParameterAttr> parameter = queryBlockingParameter(kernel, range);
    if (MakeRangeOp root = occurrenceRoots.lookup(range.getOperation()))
      if (ParameterAttr selected = lookupParameter(kernel, occurrenceParameters.lookup(root.getOperation())))
        parameter = selected;
    FailureOr<Attribute> axisKey =
        failed(parameter) ? FailureOr<Attribute>(failure())
                          : parameterAxis(*parameter);
    if (failed(parameter) || failed(axisKey)) {
      return range.emitOpError(
                 "dynamic pointwise range lost its canonical blocking dimension")
             << "; source_id=" << range.getSourceId()
             << ", fragment=" << range.getResult().getType();
    }
    Value tileCoordinate = tileCoordinates.lookup(*axisKey);
    // A coverage-bound parameter is the executable statement that this axis is
    // traversed in full by one program.  It therefore has the canonical tile
    // coordinate zero even if an earlier same-logical-range ownership probe
    // classified another occurrence as a candidate program axis.  Read the
    // current IR fact here instead of caching it while parameters are still
    // being refined; ownership mapping removes coverage_dimension from axes it
    // places on the program grid.
    if (!tileCoordinate &&
        (internalAxes.contains(*axisKey) ||
         parameter->isDeferred()))
      tileCoordinate = builder.create<arith::ConstantIndexOp>(range.getLoc(), 0);
    if (!tileCoordinate)
      return range.emitOpError(
                 "dynamic pointwise range has no physical tile coordinate")
             << "; axis=" << *axisKey
             << ", source_id=" << range.getSourceId()
             << ", ownership=" << ownershipAxes.contains(*axisKey)
             << ", internal=" << internalAxes.contains(*axisKey)
             << ", parameter_name="
             << (*parameter).getName().getValue()
             << ", parameter_role=" << stringifyParameterRole(parameter->getRole())
             << ", has_coverage="
             << parameter->isDeferred()
             << ", launch_extents=" << mapping->getAttr("launch_extents");
    PhysicalExprAttr boundedExtent = ownershipExtents.lookup(*axisKey);
    Value offsetExtent = materializeParameter(
        builder, range.getLoc(), parameter->getReference());
    if (boundedExtent)
      offsetExtent = builder.create<PhysicalExprOp>(
          range.getLoc(), builder.getIndexType(), boundedExtent);
    Value tileOffset = builder.create<BinaryOp>(
        range.getLoc(), builder.getIndexType(), tileCoordinate,
        offsetExtent, BinaryOperator::Multiply);
    Value start;
    Value end;
    if (range->hasAttr(worksetCoordinateRangeAttr)) {
      start = range.getStart();
      Value dimension = logicalDimensions.lookup(*axisKey);
      if (!dimension) {
        FailureOr<uint64_t> logicalDimension = axisDimension(*axisKey);
        if (failed(logicalDimension))
          logicalDimension = rangeDimension(range);
        if (succeeded(logicalDimension)) {
          FailureOr<Value> runtimeDimension =
              dimensionArgument(kernel, *logicalDimension);
          if (succeeded(runtimeDimension))
            dimension = *runtimeDimension;
        }
      }
      if (!dimension)
        return range.emitOpError(
                   "workset coordinate range lost its logical dimension extent")
               << "; axis=" << *axisKey << ", source_id="
               << range.getSourceId() << ", parameter="
               << parameter->getName().getValue();
      Value remaining = builder.create<BinaryOp>(
          range.getLoc(), builder.getIndexType(), dimension, tileOffset,
          BinaryOperator::Subtract);
      Value remainingDistance = builder.create<BinaryOp>(
          range.getLoc(), builder.getIndexType(), remaining, range.getStep(),
          BinaryOperator::Multiply);
      end = builder.create<BinaryOp>(range.getLoc(), builder.getIndexType(),
                                     start, remainingDistance,
                                     BinaryOperator::Add);
    } else {
      Value scaledOffset = builder.create<BinaryOp>(
          range.getLoc(), builder.getIndexType(), tileOffset, range.getStep(),
          BinaryOperator::Multiply);
      Value anchor = range.getLogicalStart();
      auto parentOffset = parentAnchorOffsets.find(range.getOperation());
      if (parentOffset != parentAnchorOffsets.end()) {
        Value offset = builder.create<arith::ConstantIndexOp>(
            range.getLoc(), parentOffset->second);
        anchor = builder.create<BinaryOp>(range.getLoc(), builder.getIndexType(),
                                          anchor, offset, BinaryOperator::Add);
      }
      start = builder.create<BinaryOp>(
          range.getLoc(), builder.getIndexType(), anchor, scaledOffset,
          BinaryOperator::Add);
      end = range.getLogicalStop();
    }
    auto sourceType = cast<FragmentType>(range.getResult().getType());
    PhysicalExprAttr tileExtent = boundedExtent ? boundedExtent
                                               : fragmentExtent(*parameter);
    if (sourceType.getShape()[0] != tileExtent) {
      if (FailureOr<uint64_t> dimension = rangeDimension(range);
          succeeded(dimension) && !queryParameterBinding(*parameter).source) {
        if (failed(retargetDimensionExtent(range.getResult(), *dimension,
                                          tileExtent)))
          return failure();
      } else if (failed(retargetSourceExtent(
                     range.getResult(), sourceAxisIdentity(range), tileExtent))) {
        return failure();
      }
      sourceType = cast<FragmentType>(range.getResult().getType());
    }
    Value physicalExtent = materializeParameter(builder, range.getLoc(), parameter->getReference());
    if (tileExtent.getKind() ==
        PhysicalExprKind::Constant)
      physicalExtent = builder.create<arith::ConstantIndexOp>(
          range.getLoc(), tileExtent.getValue());
    else if (boundedExtent)
      physicalExtent = offsetExtent;
    auto blockedType = FragmentType::get(
        module.getContext(), sourceType.getElementType(),
        builder.getArrayAttr({tileExtent}), sourceType.getAxisMaps(),
        sourceType.getValidity(),
        sourceType.getOwner());
    Value blocked = builder.create<MakeRangeOp>(
        range.getLoc(), blockedType, start, physicalExtent,
        range.getStep(), range.getLogicalStart(),
        range->hasAttr(worksetCoordinateRangeAttr) ? end : range.getLogicalStop(),
        range.getSourceId(), range.getSourceAxis(), range.getDerived());
    inheritRangeAuthority(blocked.getDefiningOp(), range);
    for (BroadcastOp bound : laneBounds.lookup(range.getResult()))
      bound.getValueMutable().assign(end);
    Value endFragment = builder.create<BroadcastOp>(range.getLoc(), blockedType, end);
    auto validComparison = builder.create<CompareOp>(
        range.getLoc(), predicateType(blockedType), blocked, endFragment,
        ComparePredicate::Lt);
    validComparison->setAttr(physicalTailAttr, builder.getUnitAttr());
    // Changing the origin/capacity invalidates sign facts about the old range.
    // Preserve both logical bounds; range simplification can discharge either
    // comparison only after proving the newly materialized coordinates.
    Value lower = builder.create<BroadcastOp>(
        range.getLoc(), blockedType, range.getLogicalStart());
    Value active = builder.create<CompareOp>(
        range.getLoc(), predicateType(blockedType), blocked, lower,
        ComparePredicate::Ge);
    Value valid = builder.create<BinaryOp>(
        range.getLoc(), predicateType(blockedType), active, validComparison,
        BinaryOperator::LogicalAnd);
    range.getResult().replaceAllUsesWith(blocked);
    rangePredicates[blocked] = valid;
    range.erase();
  }

  // Dynamic tail predicates are side-table facts until they are attached to
  // accesses.  Materialize those SSA uses before histogram realization, whose
  // dead-value cleanup would otherwise erase the unused comparisons and leave
  // dangling Values in rangePredicates.
  if (failed(closeValueRelations(kernel, ValueRelationScope::AccessResults)) ||
      failed(addTailValidity(kernel, kernel, rangePredicates,
                             /*includeStores=*/true)))
    return failure();
  return success();
}

} // namespace intent::gpu::pointwise

namespace intent::gpu {
LogicalResult realizePointwiseOwnership(ModuleOp module) {
  auto kernel = getPhysicalKernel(module);
  return failed(kernel) ? failure() : pointwise::PointwiseRewrite(module, *kernel).ownership();
}

LogicalResult realizePointwiseBlocking(ModuleOp module) {
  auto kernel = getPhysicalKernel(module);
  if (failed(kernel) || failed(pointwise::PointwiseRewrite(module, *kernel).blocking()) ||
      failed(realizeVectorContractions(module)) || failed(closeValueRelations(*kernel)))
    return failure();
  return guardInactivePredicatedLoops(module);
}
} // namespace intent::gpu
