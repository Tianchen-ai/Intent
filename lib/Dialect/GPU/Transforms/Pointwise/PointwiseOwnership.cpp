#include "Intent/Dialect/GPU/IR/ProgramInterface.h"
#include "Pointwise.h"
#include "Intent/Dialect/GPU/Transforms/Mapping/ExecutionGroups.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Transforms/Contraction/Contraction.h"
#include "Intent/Dialect/GPU/Transforms/Control/Predication.h"
#include "Intent/Dialect/GPU/Transforms/Control/Traversal.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <algorithm>
#include <limits>

using namespace mlir;

namespace intent::gpu::pointwise {

LogicalResult separateReductionOutputOccurrences(func::FuncOp kernel) {
  // A source reused in two operand positions can be both reduced and retained.
  // Give the retained occurrence its own derived range before ownership sees it.
  // The positional replay leaves the original occurrence available to other uses.
  uint64_t nextSource = nextPhysicalAxisIdentities(kernel).first;
  SmallVector<StoreOp> stores;
  kernel.walk([&](StoreOp store) { stores.push_back(store); });
  for (StoreOp store : stores) {
    SmallVector<Value> coordinates(store.getCoordinates());
    for (auto [position, coordinate] : llvm::enumerate(coordinates)) {
      auto type = dyn_cast<FragmentType>(coordinate.getType());
      if (!type || type.getShape().size() != 1)
        continue;
      PhysicalProgramAnalysis analysis(kernel);
      auto address = analysis.axisRanges(coordinate, 0);
      auto range = queryExactLogicalRange(address);
      if (failed(range))
        continue;
      auto axis = repeatedReductionOutputAxis(store.getValue(), *range);
      if (!axis)
        continue;
      auto source = sourceAxisIdentity(*range);
      IRMapping mapping;
      auto replay = analysis.replayAt(
          store.getValue(), source, PhysicalReplayScope::ValueGraph,
          /*allowAccesses=*/true, store, mapping);
      auto replayable = [&](Value value) {
        return !value || analysis.replayAt(
            value, source, PhysicalReplayScope::ValueGraph,
            /*allowAccesses=*/true, store, mapping).isReplayable();
      };
      if (!replay.isReplayable() || replay.crossesStructuredProgram ||
          !llvm::all_of(replay.contractions, [](Operation *operation) {
            return isa<ContractOp>(operation);
          }) ||
          llvm::any_of(replay.accesses, [&](Operation *access) {
            return llvm::any_of(access->getResults(), [&](Value value) {
              return queryFragmentAxes(value.getType(), source).size() > 1;
            });
          }) ||
          !replayable(coordinate) || !replayable(store.getValid()))
        continue;
      auto roots = analysis.axisRanges(store.getValue(), *axis);
      for (MakeRangeOp root : address.roots)
        if (!llvm::is_contained(roots.roots, root))
          roots.roots.push_back(root);
      auto extent = cast<PhysicalExprAttr>(type.getShape()[0]);
      if (cast<FragmentType>(store.getValue().getType()).getShape()[*axis] !=
              extent ||
          llvm::any_of(roots.roots, [&](MakeRangeOp root) {
            return root.getResult().getType().getShape()[0] != extent;
          }))
        continue;
      OpBuilder builder(store);
      auto original = cast<AxisMapAttr>(type.getAxisMaps()[0]);
      auto selected = AxisMapAttr::get(
          kernel.getContext(), nextSource++, original.getSourceAxis(),
          original.getDimensionId(), 0, true);
      for (MakeRangeOp root : roots.roots) {
        auto fragment = cast<FragmentType>(root.getResult().getType());
        auto replacementType = FragmentType::get(
            kernel.getContext(), fragment.getElementType(), fragment.getShape(),
            builder.getArrayAttr({selected}), fragment.getValidity(),
            fragment.getOwner());
        auto replacement = builder.create<MakeRangeOp>(
            root.getLoc(), replacementType, root.getStart(), root.getExtent(),
            root.getStep(), root.getLogicalStart(), root.getLogicalStop(),
            selected.getSourceId(), selected.getSourceAxis(), true);
        inheritRangeAuthority(replacement, root);
        mapping.map(root.getResult(), replacement.getResult());
      }
      ReplayMaterializationOptions options;
      options.fragmentAxis = *axis;
      options.traversalRanges = roots.roots;
      options.segmentMapping = selected;
      auto payload = materializeReplayedValue(
          builder, store.getLoc(), store.getValue(), source, extent, mapping,
          store, options);
      if (failed(payload))
        return store.emitOpError("reduction output occurrence cannot be separated");
      options.fragmentAxis = 0;
      auto addressValue = materializeReplayedValue(
          builder, store.getLoc(), coordinate, source, extent, mapping, store,
          options);
      if (failed(addressValue))
        return store.emitOpError("reduction output address cannot be separated");
      if (store.getValid()) {
        options.fragmentAxis = *axis;
        auto valid = materializeReplayedValue(
            builder, store.getLoc(), store.getValid(), source, extent, mapping,
            store, options);
        if (failed(valid))
          return store.emitOpError("reduction output validity cannot be separated");
        store.getValidMutable().assign(*valid);
      }
      store.getValueMutable().assign(*payload);
      coordinates[position] = *addressValue;
      store.getCoordinatesMutable().assign(coordinates);
    }
  }
  eraseDeadPhysicalValues(kernel);
  return success();
}

LogicalResult alignHistogramOutputOwnership(func::FuncOp kernel) {
  SmallVector<HistogramOp> histograms;
  kernel.walk([&](HistogramOp histogram) { histograms.push_back(histogram); });
  for (HistogramOp histogram : histograms) {
    SmallVector<StoreOp> stores;
    kernel.walk([&](StoreOp store) {
      if (histogramSource(store.getValue()) == histogram)
        stores.push_back(store);
    });
    if (stores.empty())
      return histogram.emitOpError(
          "histogram result has no physical output ownership effect");
    FailureOr<FragmentType> outputType = coordinateValueSchema(
        histogram.getResult().getType().getElementType(),
        stores.front().getCoordinates());
    if (failed(outputType) || outputType->getShape().size() != 1)
      return histogram.emitOpError(
          "histogram output has no one-axis physical ownership schema");
    for (StoreOp store : llvm::drop_begin(stores)) {
      FailureOr<FragmentType> current = coordinateValueSchema(
          histogram.getResult().getType().getElementType(),
          store.getCoordinates());
      if (failed(current) || *current != *outputType)
        return histogram.emitOpError(
            "histogram output effects do not share one physical ownership schema");
    }
    histogram.getResult().setType(*outputType);
  }
  return success();
}

LogicalResult PointwiseRewrite::prepareAxisRelations(bool preserveReductionPositions) {
  SmallVector<ExecutionGroupOp> mappings;
  kernel.walk([&](ExecutionGroupOp mapping) { mappings.push_back(mapping); });
  if (mappings.size() != 1)
    return kernel.emitError(
        "dynamic pointwise blocking requires one explicit execution workset");
  mapping = mappings.front();

  writeEffects = readWriteEffects(kernel);
  pointwiseElementBitWidth = 0;
  ownershipSources.clear(); directOwnershipSources.clear();
  for (const WriteEffectFacts &effect : writeEffects) {
    for (Value payload : effect.payloads)
      pointwiseElementBitWidth = std::max(pointwiseElementBitWidth, physicalElementBitWidth(payload.getType()));
    for (Value coordinate : effect.coordinates)
      if (auto fragment = dyn_cast<FragmentType>(coordinate.getType()))
        for (Attribute attribute : fragment.getAxisMaps())
          ownershipSources.insert(sourceAxisIdentity(cast<AxisMapAttr>(attribute)));
  }
  for (const WriteEffectFacts &effect : writeEffects)
    for (Value coordinate : effect.coordinates) {
      auto fragment = dyn_cast<FragmentType>(coordinate.getType());
      if (!fragment)
        continue;
      for (Attribute attribute : fragment.getAxisMaps()) {
        auto mapping = cast<AxisMapAttr>(attribute);
        PhysicalSourceAxis source{mapping.getSourceId(), mapping.getSourceAxis(),
                                  mapping.getDerived()};
        PhysicalReplayFact replay = PhysicalProgramAnalysis(kernel).replayability(
            coordinate, source, PhysicalReplayScope::Coordinate,
            /*allowAccesses=*/true);
        if (replay.isReplayable() && !replay.crossesAccess)
          directOwnershipSources.insert(source);
      }
    }
  // Structured traversal ownership is attached to the exact range operations
  // above, not erased source-wide here.  One immutable source axis may have a
  // query ownership projection and an independent fold/scan segment
  // projection; conflating them would discard a real physical decision.

  if (auto roles =
          mapping.getCoordinateRolesAttr()) {
    SmallVector<int64_t> updatedRoles(roles.asArrayRef());
    kernel.walk([&](WorksetCoordinateOp coordinate) {
      auto axis = coordinate->getAttrOfType<IntegerAttr>(worksetAxisAttr);
      if (!axis || axis.getInt() < 0 ||
          static_cast<size_t>(axis.getInt()) >= updatedRoles.size())
        return;
      auto crossesDataLoad = [&](Value root) {
        SmallVector<std::pair<Value, bool>> worklist{{root, false}};
        llvm::SmallDenseSet<Value> beforeLoad;
        llvm::SmallDenseSet<Value> afterLoad;
        while (!worklist.empty()) {
          auto [value, crossed] = worklist.pop_back_val();
          auto &visited = crossed ? afterLoad : beforeLoad;
          if (!visited.insert(value).second)
            continue;
          if (value == coordinate.getResult())
            return crossed;
          Operation *definition = value.getDefiningOp();
          if (!definition)
            continue;
          bool nextCrossed = crossed || isa<LoadOp, GatherOp>(definition);
          for (Value operand : definition->getOperands())
            worklist.emplace_back(operand, nextCrossed);
        }
        return false;
      };
      bool indirect = llvm::any_of(writeEffects, [&](const auto &effect) {
        return llvm::any_of(effect.coordinates, crossesDataLoad);
      });
      updatedRoles[axis.getInt()] = static_cast<int64_t>(
          indirect ? CoordinateRole::IndirectTraversal
                   : CoordinateRole::Workset);
    });
    mapping.setCoordinateRolesAttr(
        DenseI64ArrayAttr::get(module.getContext(), updatedRoles));
  }

  kernel.walk([&](RegionFoldOp fold) {
    auto category = lookupParameter(kernel, fold.getSegment()).getCategory();
    if (category != ParameterCategory::RegionContraction &&
        category != ParameterCategory::RegionReduction)
      return;
    for (Value result : fold.getResults()) {
      llvm::SmallDenseSet<uint64_t> resultDimensions;
      collectPhysicalDimensions(result.getType(), resultDimensions);
      collectPartiallyCarriedDimensions(
          result.getType(), partiallyCarriedStructuredDimensions);
      for (uint64_t dimension : resultDimensions) {
        auto existing = structuredOwnershipCategories.find(dimension);
        if (existing == structuredOwnershipCategories.end() ||
            category == ParameterCategory::RegionContraction)
          structuredOwnershipCategories[dimension] = category;
      }
    }
  });
  kernel.walk([&](HistogramOp histogram) {
    llvm::SmallDenseSet<uint64_t> resultDimensions;
    collectPhysicalDimensions(histogram.getResult().getType(), resultDimensions);
    for (uint64_t dimension : resultDimensions)
      structuredOwnershipCategories.try_emplace(
          dimension, ParameterCategory::Histogram);
  });
  kernel.walk([&](Operation *operation) {
    if (!isa<ContractOp, ScaledContractOp, SparseContractOp>(operation))
      return;
    llvm::SmallDenseSet<uint64_t> seen;
    for (const auto &axis : PhysicalProgramAnalysis(kernel).contractFreeAxes(operation).axes) {
      FailureOr<AxisMapAttr> mapping = queryAxisMap(axis.operand.getType(), axis.operandAxis);
      if (failed(mapping) || mapping->getDimensionId() <= 0)
        continue;
      uint64_t dimension = mapping->getDimensionId();
      if (!seen.insert(dimension).second)
        nonUniqueContractionDimensions.insert(dimension);
    }
    if (auto contract = dyn_cast<ContractOp>(operation))
      for (auto [operand, reductionAxes] :
           {std::pair{contract.getLhs(), contract.getLhsReductionAxes()},
            std::pair{contract.getRhs(), contract.getRhsReductionAxes()}})
        for (int64_t axis : reductionAxes) {
          auto mapping = queryAxisMap(operand.getType(), axis);
          if (succeeded(mapping) && seen.contains(mapping->getDimensionId()))
            nonUniqueContractionDimensions.insert(mapping->getDimensionId());
        }
  });
  for (MakeRangeOp range : allRanges) {
    auto dimension = ownershipDimension(kernel, range);
    if (failed(dimension) || !nonUniqueContractionDimensions.contains(*dimension))
      continue;
    auto facts = contractFreeAxisFacts(kernel, range);
    if (facts.sides != ContractFreeAxisLhs && facts.sides != ContractFreeAxisRhs)
      continue;
    bool reduced = false;
    PhysicalProgramAnalysis analysis(kernel);
    kernel.walk([&](ContractOp contract) {
      auto freeAxes = analysis.contractFreeAxes(contract);
      reduced |= llvm::count_if(freeAxes.axes, [&](const auto &axis) {
        return llvm::any_of(axis.ranges.roots, [&](MakeRangeOp root) {
          return sameLogicalRange(root, range);
        });
      }) > 1;
      // A free producer can feed one operand while its source identity is
      // repeated in the result. Contraction replay owns those Cartesian
      // occurrences; a source-wide pointwise tile would conflate them.
      reduced |= contract.getLhsBatchAxes().empty() &&
                 llvm::any_of(freeAxes.axes, [&](const auto &axis) {
        auto mapping = queryAxisMap(axis.operand.getType(), axis.operandAxis);
        return succeeded(mapping) &&
               queryFragmentAxes(contract.getResult().getType(),
                                 sourceAxisIdentity(*mapping)).size() > 1 &&
               llvm::any_of(axis.ranges.roots, [&](MakeRangeOp root) {
                 return sameLogicalRange(root, range);
               });
      });
      for (auto [operand, axes] :
           {std::pair{contract.getLhs(), contract.getLhsReductionAxes()},
            std::pair{contract.getRhs(), contract.getRhsReductionAxes()}})
        for (int64_t axis : axes) {
          auto roots = analysis.axisRanges(operand, axis);
          reduced |= !roots.isExact() || !roots.blockers.empty() ||
              llvm::any_of(roots.roots, [&](MakeRangeOp root) {
                return sameLogicalRange(root, range);
              });
        }
    });
    kernel.walk([&](Operation *operation) {
      if (!isa<ScaledContractOp, SparseContractOp>(operation))
        return;
      for (Value operand : operation->getOperands())
        reduced |= !queryFragmentAxes(operand.getType(), sourceAxisIdentity(range))
                        .empty();
    });
    if (!reduced)
      independentContractionRanges.insert(range.getOperation());
  }


  uint64_t nextOccurrenceSource = nextPhysicalAxisIdentities(kernel).first;
  for (MakeRangeOp range : allRanges) {
    if (!independentContractionRanges.contains(range.getOperation()) ||
        !hasPointwiseOwnership(range))
      continue;
    MakeRangeOp root = range;
    for (const auto &entry : occurrenceRoots)
      if (sameLogicalRange(range, entry.second)) {
        root = entry.second;
        break;
      }
    occurrenceRoots[range.getOperation()] = root;
    positionalOccurrences.insert(range.getOperation());
  }
  WalkResult occurrences = kernel.walk([&](StoreOp store) {
    auto valueType = dyn_cast<FragmentType>(store.getValue().getType());
    SmallVector<Value> coordinates;
    for (Value coordinate : store.getCoordinates()) {
      auto type = dyn_cast<FragmentType>(coordinate.getType());
      if (!type)
        continue;
      coordinates.push_back(coordinate);
    }
    if (!valueType || coordinates.empty())
      return WalkResult::advance();
    bool cartesian = valueType.getShape().size() == coordinates.size() &&
        llvm::all_of(coordinates, [](Value value) {
          return cast<FragmentType>(value.getType()).getShape().size() == 1;
        });
    if (cartesian) {
      PhysicalProgramAnalysis analysis(kernel);
      llvm::DenseMap<uint64_t, unsigned> positions;
      for (auto [position, coordinate] : llvm::enumerate(coordinates))
        for (MakeRangeOp range : analysis.axisRanges(coordinate, 0).roots) {
          auto dimension = ownershipDimension(kernel, range);
          if (failed(dimension))
            continue;
          auto [previous, inserted] = positions.try_emplace(*dimension, position);
          if (!inserted && previous->second != position)
            independentCartesianDimensions.insert(*dimension);
        }
    }
    auto ordered = store->getParentOfType<scf::ForOp>();
    if (cartesian && ordered && !ordered->hasAttr(independentIterationAttr)) {
      PhysicalProgramAnalysis analysis(kernel);
      DominanceInfo dominance(kernel);
      llvm::DenseMap<Operation *, unsigned> sourceAxes;
      SmallVector<MakeRangeOp> shared;
      for (unsigned axis = 0; axis < valueType.getShape().size(); ++axis) {
        auto ranges = analysis.axisRanges(store.getValue(), axis);
        if (!ranges.isExact() || !ranges.blockers.empty())
          continue;
        for (MakeRangeOp range : ranges.roots) {
          auto dimension = queryRangeDimension(range);
          if (failed(dimension) || nonUniqueContractionDimensions.contains(*dimension) ||
              isProvablySingletonLogicalRange(range) ||
              (!ordered->isProperAncestor(range) &&
               (!dominance.dominates(range.getOperation(), ordered.getOperation()) ||
                !hasExactStaticFullCoverage(kernel, range.getResult(), 0))))
            continue;
          auto [previous, inserted] = sourceAxes.try_emplace(range.getOperation(), axis);
          if (!inserted && previous->second != axis &&
              !llvm::is_contained(shared, range))
            shared.push_back(range);
        }
      }
      // One vector can feed both axes of an ordered Cartesian update. Preserve
      // its full snapshot, including an already complete dominating capture.
      for (MakeRangeOp range : allRanges) {
        bool local = ordered->isProperAncestor(range);
        if (range->hasAttr(worksetCoordinateRangeAttr) ||
            (!local && !llvm::is_contained(shared, range)) ||
            !llvm::any_of(shared, [&](MakeRangeOp root) {
              auto dimension = queryRangeDimension(range);
              auto rootDimension = queryRangeDimension(root);
              return sameLogicalRange(range, root) && succeeded(dimension) &&
                     succeeded(rootDimension) && *dimension == *rootDimension;
            }))
          continue;
        if (local &&
            failed(requireFullDimensionCoverage(kernel, range.getResult(), 0))) {
          store.emitOpError("shared Cartesian producer has no exact local full coverage");
          return WalkResult::interrupt();
        }
        retainedCartesianRanges.insert(range.getOperation());
        deferRange(range.getOperation());
        occurrenceRoots.erase(range.getOperation());
      }
      valueType = cast<FragmentType>(store.getValue().getType());
    }
    PhysicalProgramAnalysis analysis(kernel);
    llvm::DenseMap<Operation *, unsigned> resultAxes;
    for (auto [axis, attribute] : llvm::enumerate(valueType.getAxisMaps())) {
      auto mapping = cast<AxisMapAttr>(attribute);
      int64_t dimension = mapping.getDimensionId();
      if (dimension <= 0 || nonUniqueContractionDimensions.contains(dimension))
        continue;
      bool repeatedDimension =
          llvm::count_if(valueType.getAxisMaps(), [&](Attribute other) {
            return cast<AxisMapAttr>(other).getDimensionId() == dimension;
          }) >= 2;
      // Rank-one Cartesian coordinates concatenate into the result axes.
      // Equal dimension values do not equate those independent occurrences.
      PhysicalRangeFact address;
      if (cartesian) {
        address = analysis.axisRanges(coordinates[axis], 0);
      } else {
        address.state = PhysicalFactState::Exact;
        for (Value coordinate : coordinates) {
          auto source = cast<FragmentType>(coordinate.getType());
          auto projection = queryAxisProjection(source, valueType);
          if (!projection.isExact()) {
            address.state = PhysicalFactState::Unknown;
            break;
          }
          auto sourceAxis = projection.targetToSource[axis];
          if (!sourceAxis)
            continue;
          PhysicalRangeFact ranges = analysis.axisRanges(coordinate, *sourceAxis);
          if (ranges.state == PhysicalFactState::Unknown || !ranges.blockers.empty()) {
            address.state = PhysicalFactState::Unknown;
            break;
          }
          for (MakeRangeOp range : ranges.roots)
            if (!llvm::is_contained(address.roots, range))
              address.roots.push_back(range);
        }
        if (!address.roots.empty() &&
            !analysis.lockstepRanges(address.roots).isExact())
          address.state = PhysicalFactState::Unknown;
        address.unitStep = llvm::all_of(address.roots, isUnitStepRange);
      }
      PhysicalRangeFact payload = analysis.axisRanges(store.getValue(), axis);
      if (llvm::any_of(payload.roots, [&](MakeRangeOp range) {
            return retainedCartesianRanges.contains(range.getOperation());
          }))
        continue;
      if (cartesian && address.isExact() && !address.roots.empty() &&
          address.blockers.empty() && payload.isExact() && payload.roots.empty() &&
          payload.blockers.empty()) {
        // A uniform payload still writes an independent Cartesian address
        // axis. Give that occurrence its own source-bound ownership parameter.
        for (MakeRangeOp range : address.roots)
          occurrenceRoots.try_emplace(range.getOperation(), address.roots.front());
        continue;
      }
      if (address.state == PhysicalFactState::Unknown ||
          payload.state == PhysicalFactState::Unknown ||
          !address.blockers.empty() || !payload.blockers.empty() ||
          address.roots.empty() || payload.roots.empty())
        continue;
      if (llvm::all_of(address.roots, isProvablySingletonLogicalRange))
        continue;
      if (llvm::any_of(payload.roots, [&](MakeRangeOp range) {
            auto dimension = queryRangeDimension(range);
            return succeeded(dimension) &&
                   nonUniqueContractionDimensions.contains(*dimension);
          })) {
        // Positional output rebinding can hide repeated contraction free axes
        // behind different address dimensions. Their operand occurrences must
        // remain separate until contraction blocking binds each side.
        for (MakeRangeOp range : address.roots)
          forcedContractionRanges.insert(range.getOperation());
        // Epilogue reads can materialize the same output coordinates through
        // separate ranges. Their tile must follow the contraction as well.
        for (MakeRangeOp range : payload.roots)
          forcedContractionRanges.insert(range.getOperation());
        continue;
      }
      auto sameBound = [](Value lhs, Value rhs) {
        if (samePhysicalScalarExpression(lhs, rhs))
          return true;
        PhysicalExprAttr left = queryLaunchExpression(lhs);
        PhysicalExprAttr right = queryLaunchExpression(rhs);
        return left && right && left == right;
      };
      bool equivalentSources = cartesian &&
          !repeatedDimension && address.isExact() &&
          llvm::any_of(payload.roots, [&](MakeRangeOp range) {
            return llvm::none_of(address.roots, [&](MakeRangeOp coordinate) {
              return sourceAxisIdentity(range) == sourceAxisIdentity(coordinate);
            });
          }) && llvm::all_of(payload.roots, [&](MakeRangeOp range) {
            auto sourceDimension = queryRangeDimension(range);
            return succeeded(sourceDimension) && *sourceDimension == dimension &&
                   llvm::all_of(address.roots, [&](MakeRangeOp coordinate) {
                     auto addressDimension = queryRangeDimension(coordinate);
                     bool sameExtent = succeeded(addressDimension) &&
                                       *addressDimension == *sourceDimension;
                     // A positional store pairs ordinals of the same logical
                     // extent even when the source and destination start at
                     // different coordinates. Keep each range's own origin.
                     return sameBound(range.getStep(), coordinate.getStep()) &&
                            (sameExtent ||
                             (sameBound(range.getLogicalStart(),
                                        coordinate.getLogicalStart()) &&
                              sameBound(range.getLogicalStop(),
                                        coordinate.getLogicalStop())));
                   });
          });
      bool positionalRemap = equivalentSources || llvm::any_of(payload.roots, [&](MakeRangeOp range) {
        FailureOr<int64_t> sourceDimension = queryRangeDimension(range);
        return succeeded(sourceDimension) && *sourceDimension != dimension;
      });
      if (!repeatedDimension && !positionalRemap)
        continue;
      if (positionalRemap && !address.isExact())
        continue;
      if (preserveReductionPositions && cartesian && positionalRemap)
        for (MakeRangeOp range : payload.roots) {
          if (isReductionTraversal(range.getOperation()) ||
              llvm::any_of(writeEffects, [&](const WriteEffectFacts &effect) {
                return coordinatesUseRange(effect.coordinates, range);
              }))
            continue;
          bool independentAddress = false;
          for (auto [otherAxis, coordinate] : llvm::enumerate(coordinates)) {
            if (otherAxis == axis)
              continue;
            auto otherPayload = analysis.axisRanges(store.getValue(), otherAxis);
            auto otherAddress = analysis.axisRanges(coordinate, 0);
            independentAddress |= otherPayload.isExact() &&
                !llvm::is_contained(otherPayload.roots, range) &&
                otherPayload.blockers.empty() && otherAddress.isExact() &&
                llvm::any_of(otherAddress.roots, [&](MakeRangeOp other) {
                  return other != range && sameLogicalRange(other, range);
                });
          }
          if (!independentAddress)
            continue;
          // A read-only occurrence used in another result position does not
          // own the independent Cartesian address, even with the same domain.
          // Rebind its current value-axis flow, leaving logical coordinates and
          // every write-address occurrence unchanged.
          PhysicalSourceAxis previous = sourceAxisIdentity(range);
          uint64_t sourceId = nextOccurrenceSource++;
          SmallVector<std::pair<Value, FragmentType>> replacements;
          auto collect = [&](Value value) {
            auto type = dyn_cast<FragmentType>(value.getType());
            if (!type)
              return;
            SmallVector<Attribute> maps(type.getAxisMaps().begin(), type.getAxisMaps().end());
            bool changed = false;
            for (auto [position, attribute] : llvm::enumerate(type.getAxisMaps())) {
              auto mapping = cast<AxisMapAttr>(attribute);
              if (!(sourceAxisIdentity(mapping) == previous))
                continue;
              auto roots = analysis.axisRanges(value, position);
              if (roots.state == PhysicalFactState::Unknown || !roots.blockers.empty() ||
                  roots.roots.empty() ||
                  !llvm::all_of(roots.roots, [&](MakeRangeOp root) { return root == range; }))
                continue;
              maps[position] = AxisMapAttr::get(kernel.getContext(), sourceId,
                  mapping.getSourceAxis(), mapping.getDimensionId(), position, true);
              changed = true;
            }
            if (changed)
              replacements.emplace_back(value, FragmentType::get(kernel.getContext(),
                  type.getElementType(), type.getShape(), ArrayAttr::get(kernel.getContext(), maps),
                  type.getValidity(), type.getOwner()));
          };
          kernel.walk([&](Operation *operation) {
            for (Value result : operation->getResults())
              collect(result);
            for (Region &region : operation->getRegions())
              for (Block &block : region)
                for (BlockArgument argument : block.getArguments())
                  collect(argument);
          });
          for (auto [value, type] : replacements)
            value.setType(type);
          range.setSourceId(sourceId);
          range.setDerived(true);
        }
      SmallVector<MakeRangeOp> ranges(address.roots.begin(), address.roots.end());
      for (MakeRangeOp range : payload.roots)
        if (!llvm::is_contained(ranges, range))
          ranges.push_back(range);
      if (positionalRemap && !repeatedDimension) {
        // Separate access sites can materialize the same logical range. Keep
        // their positional ownership connected instead of multiplying grids.
        SmallVector<MakeRangeOp> related;
        for (auto &entry : occurrenceRoots) {
          auto previous = cast<MakeRangeOp>(entry.first);
          auto previousAxis = resultAxes.find(previous.getOperation());
          if (previousAxis != resultAxes.end() && previousAxis->second != axis)
            continue;
          if (positionalOccurrences.contains(entry.first) &&
              llvm::any_of(ranges, [&](MakeRangeOp range) {
                return sameLogicalRange(range, previous);
              }))
            related.push_back(previous);
        }
        for (MakeRangeOp range : related)
          if (!llvm::is_contained(ranges, range))
            ranges.push_back(range);
      }
      MakeRangeOp root = ranges.front();
      FailureOr<int64_t> addressExtent = exactStaticTraversalExtent(address);
      if (!llvm::all_of(ranges, [&](MakeRangeOp range) {
            if (positionalRemap) {
              // The store pairs these tensor axes by ordinal, even when the
              // address domains have different starts. Share their tile width,
              // retaining each domain's own coordinates and source identity.
              PhysicalExprAttr length = queryLaunchRangeExtent(root);
              FailureOr<int64_t> extent = exactStaticTraversalExtent(
                  analysis.axisRanges(range.getResult(), 0));
              bool sameStaticCardinality = succeeded(addressExtent) &&
                                           succeeded(extent) &&
                                           *addressExtent == *extent;
              auto rootDimension = queryRangeDimension(root);
              auto rangeDimension = queryRangeDimension(range);
              bool sameLogicalExtent = succeeded(rootDimension) &&
                                       succeeded(rangeDimension) &&
                                       *rootDimension == *rangeDimension;
              return isUnitStepRange(root) && isUnitStepRange(range) &&
                     (sameLogicalExtent || sameStaticCardinality ||
                      (length && length == queryLaunchRangeExtent(range)));
            }
            return sameBound(root.getLogicalStart(), range.getLogicalStart()) &&
                   sameBound(root.getLogicalStop(), range.getLogicalStop()) &&
                   sameBound(root.getStep(), range.getStep());
          }))
        continue;
      bool retainsReducedAxis = llvm::any_of(payload.roots, [&](MakeRangeOp range) {
        if (!isReductionTraversal(range.getOperation()))
          return false;
        // Query the current result relation: a structured result can rename
        // the producer range without removing its reduction dependency.
        auto dependency = analysis.reductionDependency(
            store.getValue(), sourceAxisIdentity(mapping), dimension);
        return dependency.isExact() && dependency.depends;
      });
      for (MakeRangeOp range : ranges) {
        auto [found, inserted] = resultAxes.try_emplace(range.getOperation(), axis);
        if (!inserted && found->second != axis) {
          store.emitOpError("Cartesian pointwise axes require independent producer ranges")
              << "; first_axis=" << found->second << "; second_axis=" << axis
              << "; shared_range=" << range.getResult();
          return WalkResult::interrupt();
        }
      }
      for (MakeRangeOp range : ranges) {
        auto previous = occurrenceRoots.find(range.getOperation());
        if (previous == occurrenceRoots.end())
          continue;
        MakeRangeOp old = previous->second;
        for (auto &entry : occurrenceRoots)
          if (entry.second == old)
            entry.second = root;
      }
      for (MakeRangeOp range : ranges) {
        occurrenceRoots[range.getOperation()] = root;
        // The occurrence class is axis-specific even when extents repeat.
        if (positionalRemap)
          positionalOccurrences.insert(range.getOperation());
        ownershipSources.insert(sourceAxisIdentity(range));
        if (positionalRemap)
          directOwnershipSources.insert(sourceAxisIdentity(range));
        if (retainsReducedAxis) {
          forcedReductionRanges.insert(range.getOperation());
          deferRange(range.getOperation());
        }
      }
    }
    return WalkResult::advance();
  });
  if (occurrences.wasInterrupted())
    return failure();

  // A write's range can own a resource slice only if every access to that
  // resource uses the same traversal on that axis. In particular, an initial
  // copy cannot be distributed while a later recurrence reads whole prefixes.
  PhysicalProgramAnalysis accessAnalysis(kernel);
  auto projectedRange = [](Value value) {
    while (Operation *operation = value.getDefiningOp()) {
      if (!isa<BroadcastOp, ReshapeOp>(operation))
        break;
      value = operation->getOperand(0);
    }
    return value.getDefiningOp<MakeRangeOp>();
  };
  kernel.walk([&](StoreOp store) {
    for (auto [coordinate, axis] : llvm::zip(store.getCoordinates(), store.getSourceAxes())) {
      llvm::SmallPtrSet<Operation *, 8> roots;
      collectCoordinateRanges(coordinate, roots);
      if (roots.empty())
        continue;
      auto range = projectedRange(coordinate);
      for (Operation *user : store.getResource().getUsers()) {
        if (user == store.getOperation() || isa<DimOp, AssumeInBoundsOp>(user))
          continue;
        auto access = accessAnalysis.footprint(user);
        auto position = llvm::find(access.sourceAxes, axis);
        auto other = position == access.sourceAxes.end() ? MakeRangeOp() :
            projectedRange(access.coordinates[position - access.sourceAxes.begin()]);
        auto occurrence = range ? occurrenceRoots.lookup(range.getOperation()) : MakeRangeOp();
        bool sameOwnership = range && other &&
            (sameLogicalRange(range, other) ||
             (occurrence && occurrence == occurrenceRoots.lookup(other.getOperation())));
        if (access.state != PhysicalFactState::Exact || !sameOwnership ||
            !accessAnalysis.lockstepRanges({range, other}).isExact()) {
          dependentResourceRanges.insert(roots.begin(), roots.end());
          break;
        }
      }
    }
  });
  for (MakeRangeOp range : allRanges)
    if (llvm::any_of(dependentResourceRanges, [&](Operation *operation) {
          auto dependent = cast<MakeRangeOp>(operation);
          auto occurrence = occurrenceRoots.lookup(range.getOperation());
          return sameLogicalRange(range, dependent) ||
              (occurrence && occurrence == occurrenceRoots.lookup(operation));
        }))
      deferRange(range.getOperation());
  // Ordered writes to a parent and its subregion use one absolute tile anchor.
  // Shift the subregion's positional payload by the same ordinal displacement.
  auto constantBound = [](Value value) -> std::optional<int64_t> {
    auto bound = queryLaunchExpression(value);
    if (!bound || bound.getKind() !=
                      PhysicalExprKind::Constant)
      return std::nullopt;
    return bound.getValue();
  };
  SmallVector<StoreOp> precedingStores;
  WalkResult parentAnchors = kernel.walk([&](StoreOp store) {
    PhysicalProgramAnalysis analysis(kernel);
    for (auto [position, coordinate] : llvm::enumerate(store.getCoordinates())) {
      auto type = dyn_cast<FragmentType>(coordinate.getType());
      if (!type || type.getShape().size() != 1)
        continue;
      auto child = queryExactLogicalRange(analysis.axisRanges(coordinate, 0));
      if (failed(child) || !isUnitStepRange(*child))
        continue;
      auto parentDimension = querySubregionParentDimension(*child);
      auto childStart = constantBound(child->getLogicalStart());
      auto childStop = constantBound(child->getLogicalStop());
      if (failed(parentDimension) || !childStart || !childStop || *childStart < 0)
        continue;
      std::optional<int64_t> offset;
      for (StoreOp previous : precedingStores) {
        if (previous.getResource() != store.getResource() ||
            previous->getBlock() != store->getBlock())
          continue;
        auto sourceAxis = store.getSourceAxes()[position];
        for (auto [parentPosition, parentCoordinate] :
             llvm::enumerate(previous.getCoordinates())) {
          if (previous.getSourceAxes()[parentPosition] != sourceAxis)
            continue;
          auto parentType = dyn_cast<FragmentType>(parentCoordinate.getType());
          if (!parentType || parentType.getShape().size() != 1)
            continue;
          auto parent = queryExactLogicalRange(analysis.axisRanges(parentCoordinate, 0));
          if (failed(parent) || !isUnitStepRange(*parent) ||
              !(sourceAxisIdentity(*parent) == sourceAxisIdentity(*child)))
            continue;
          auto dimension = queryRangeDimension(*parent);
          auto start = constantBound(parent->getLogicalStart());
          auto stop = constantBound(parent->getLogicalStop());
          if (failed(dimension) || *dimension != *parentDimension ||
              !start || !stop || *start < 0 || *start > *childStart ||
              *stop < *childStop)
            continue;
          int64_t current = *start - *childStart;
          if (offset && *offset != current) {
            store.emitOpError("subregion writes have incompatible parent tile anchors");
            return WalkResult::interrupt();
          }
          offset = current;
        }
      }
      if (!offset || *offset == 0)
        continue;
      MakeRangeOp root = occurrenceRoots.lookup(child->getOperation());
      if (!root)
        root = *child;
      for (MakeRangeOp range : allRanges) {
        if (range != *child && occurrenceRoots.lookup(range.getOperation()) != root)
          continue;
        auto [entry, inserted] = parentAnchorOffsets.try_emplace(range.getOperation(), *offset);
        if (!inserted && entry->second != *offset) {
          store.emitOpError("shared pointwise producer requires incompatible ordinal anchors");
          return WalkResult::interrupt();
        }
      }
    }
    precedingStores.push_back(store);
    return WalkResult::advance();
  });
  if (parentAnchors.wasInterrupted())
    return failure();
  return success();
}


LogicalResult PointwiseRewrite::resolveOwnershipDependencies() {
  // A read-only operand may reach a write through broadcast/pointwise values
  // while retaining a different source identity from the write coordinates.
  // Share ownership only through that value path and an exact logical range.
  for (MakeRangeOp range : dynamicRanges) {
    if (range->getParentOfType<RegionFoldOp>() ||
        range->getParentOfType<RegionScanOp>() ||
        isReductionTraversal(range.getOperation()))
      continue;
    FailureOr<uint64_t> dimension = rangeDimension(range);
    if (failed(dimension))
      continue;
    SmallVector<Value> worklist{range.getResult()};
    llvm::SmallDenseSet<Value> visited;
    bool reachesWrite = false;
    while (!worklist.empty() && !reachesWrite) {
      Value value = worklist.pop_back_val();
      if (!visited.insert(value).second)
        continue;
      reachesWrite = llvm::any_of(writeEffects, [&](const WriteEffectFacts &effect) {
        return llvm::is_contained(effect.payloads, value) &&
               llvm::any_of(allRanges, [&](MakeRangeOp owned) {
                 FailureOr<uint64_t> ownedDimension = rangeDimension(owned);
                 return succeeded(ownedDimension) && *ownedDimension == *dimension &&
                        sharesLogicalTraversal(range, owned) &&
                        coordinatesUseRange(effect.coordinates, owned);
               });
      });
      for (Operation *user : value.getUsers()) {
        if (auto yield = dyn_cast<scf::YieldOp>(user)) {
          if (auto branch = dyn_cast<scf::IfOp>(yield->getParentOp()))
            for (auto [position, yielded] : llvm::enumerate(yield.getOperands()))
              if (yielded == value && position < branch.getNumResults() &&
                  queryFragmentDimensions(branch.getResult(position).getType(),
                                          *dimension).size() == 1)
                worklist.push_back(branch.getResult(position));
          continue;
        }
        if (!isa<LoadOp, UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp,
                 BitcastOp, BroadcastOp, TransposeOp, ReshapeOp>(user))
          continue;
        for (Value result : user->getResults())
          if (queryFragmentDimensions(result.getType(), *dimension).size() == 1)
            worklist.push_back(result);
      }
    }
    if (reachesWrite)
      ownershipSources.insert(sourceAxisIdentity(range));
  }
  auto dependsOnSource = [&](ValueRange values, PhysicalSourceAxis source) {
    PhysicalProgramAnalysis analysis(kernel);
    for (Value value : values) {
      auto fragment = dyn_cast<FragmentType>(value.getType());
      if (!fragment)
        continue;
      for (unsigned axis = 0; axis < fragment.getShape().size(); ++axis) {
        PhysicalRangeFact ranges = analysis.axisRanges(value, axis);
        if (ranges.state != PhysicalFactState::Unknown && ranges.blockers.empty() &&
            llvm::any_of(ranges.roots, [&](MakeRangeOp range) {
              return sourceAxisIdentity(range) == source;
            }))
          return true;
      }
    }
    return false;
  };
  llvm::DenseMap<PhysicalSourceAxis, uint64_t> sourceDimensions;
  llvm::MapVector<uint64_t, SmallVector<PhysicalSourceAxis>> dimensionSources;
  for (MakeRangeOp range : dynamicRanges) {
    FailureOr<uint64_t> dimension = ownershipDimension(kernel, range);
    if (failed(dimension) || !hasPointwiseOwnership(range))
      continue;
    PhysicalSourceAxis source{range.getSourceId(), range.getSourceAxis(),
                              range.getDerived()};
    uint64_t dimensionId = *dimension;
    sourceDimensions[source] = dimensionId;
    if (!llvm::is_contained(dimensionSources[dimensionId], source))
      dimensionSources[dimensionId].push_back(source);
  }
  auto effectDependsOn = [&](const WriteEffectFacts &effect,
                             PhysicalSourceAxis source) {
    return dependsOnSource(effect.coordinates, source) ||
           dependsOnSource(effect.payloads, source);
  };
  llvm::SmallDenseSet<uint64_t> jointOwnershipDimensions;
  for (auto [dimensionId, sources] : dimensionSources) {
    if (sources.size() < 2)
      continue;
    bool onePreservedSourcePerEffect =
        llvm::all_of(writeEffects, [&](const WriteEffectFacts &effect) {
          return llvm::count_if(sources, [&](PhysicalSourceAxis source) {
                   return effectDependsOn(effect, source);
                 }) == 1;
        });
    if (onePreservedSourcePerEffect)
      jointOwnershipDimensions.insert(dimensionId);
  }
  for (MakeRangeOp range : dynamicRanges) {
    PhysicalSourceAxis source{range.getSourceId(), range.getSourceAxis(),
                              range.getDerived()};
    if (!hasPointwiseOwnership(range)) {
      deferRange(range.getOperation());
      continue;
    }
    bool requiredByEveryEffect =
        llvm::all_of(writeEffects, [&](const WriteEffectFacts &effect) {
          if (effectDependsOn(effect, source))
            return true;
          auto root = occurrenceRoots.lookup(range.getOperation());
          if (!root || !positionalOccurrences.contains(range.getOperation()))
            return false;
          return llvm::any_of(occurrenceRoots, [&](const auto &entry) {
            return entry.second == root &&
                   positionalOccurrences.contains(entry.first) &&
                   effectDependsOn(effect,
                       sourceAxisIdentity(cast<MakeRangeOp>(entry.first)));
          });
        });
    auto dimension = sourceDimensions.find(source);
    bool jointlyOwned =
        dimension != sourceDimensions.end() &&
        jointOwnershipDimensions.contains(dimension->second);
    if (!requiredByEveryEffect && !jointlyOwned)
      deferRange(range.getOperation());
  }
  return success();
}


void PointwiseRewrite::promoteOwnershipRanges() {
  // Equivalent coordinate occurrences must preserve the selected local
  // writeback traversal instead of restoring duplicate program ownership.
  for (MakeRangeOp range : allRanges)
    if (llvm::any_of(reductionCaptureWritebackRanges,
                     [&](Operation *operation) {
                       auto captured = cast<MakeRangeOp>(operation);
                       MakeRangeOp occurrence =
                           occurrenceRoots.lookup(range.getOperation());
                       bool sameOccurrence =
                           occurrence &&
                           occurrence == occurrenceRoots.lookup(
                                             captured.getOperation()) &&
                           positionalOccurrences.contains(range.getOperation()) &&
                           positionalOccurrences.contains(captured.getOperation());
                       return (sameLogicalRange(range, captured) ||
                               sameOccurrence) &&
                              PhysicalProgramAnalysis(kernel)
                                  .lockstepRanges({range, captured})
                                  .isExact();
                     }))
      deferRange(range.getOperation());
  SmallVector<MakeRangeOp> ownershipSeeds;
  for (MakeRangeOp range : dynamicRanges)
    if (hasPointwiseOwnership(range) &&
        !internalTraversalRanges.contains(range.getOperation()))
      ownershipSeeds.push_back(range);
  for (MakeRangeOp seed : ownershipSeeds)
    for (MakeRangeOp range : allRanges) {
      if (range->getParentOfType<RegionFoldOp>() ||
          range->getParentOfType<RegionScanOp>() ||
          isReductionTraversal(range.getOperation()))
        continue;
      MakeRangeOp occurrence = occurrenceRoots.lookup(seed.getOperation());
      bool sameOccurrence = occurrence &&
          occurrence == occurrenceRoots.lookup(range.getOperation()) &&
          positionalOccurrences.contains(seed.getOperation()) &&
          positionalOccurrences.contains(range.getOperation());
      // A positional store pairs producer and address axes by ordinal.
      // Their different source identities still share the selected tile's
      // ownership, including its start, not just its fragment extent.
      if (!sameLogicalRange(seed, range) && !sameOccurrence)
        continue;
      if (dependentResourceRanges.contains(range.getOperation()))
        continue;
      promoteRange(range.getOperation());
      if (!llvm::is_contained(dynamicRanges, range))
        dynamicRanges.push_back(range);
    }
}


LogicalResult PointwiseRewrite::chooseOwnership() {
  llvm::SmallDenseSet<Attribute> internalOwnershipAxes;
  for (Attribute axis : ownershipAxes) {
    if (llvm::any_of(axes.lookup(axis), [](MakeRangeOp range) {
          return !range->hasAttr(worksetCoordinateRangeAttr);
        }))
      internalOwnershipAxes.insert(axis);
  }
  if (!internalOwnershipAxes.empty()) {
    SmallVector<std::pair<int64_t, Attribute>> scalarWorksetAxes;
    for (Attribute axis : ownershipAxes) {
      if (isSourceAxisKey(axis) || internalOwnershipAxes.contains(axis))
        continue;
      std::optional<int64_t> worksetAxis;
      bool exact = llvm::all_of(axes.lookup(axis), [&](MakeRangeOp range) {
        if (!range->hasAttr(worksetCoordinateRangeAttr))
          return false;
        auto coordinate =
            range.getStart().getDefiningOp<WorksetCoordinateOp>();
        auto position = coordinate ? coordinate->getAttrOfType<IntegerAttr>(
                                         worksetAxisAttr)
                                   : IntegerAttr();
        if (!position || position.getInt() < 0 ||
            (worksetAxis && *worksetAxis != position.getInt()))
          return false;
        worksetAxis = position.getInt();
        return true;
      });
      if (exact && worksetAxis)
        scalarWorksetAxes.emplace_back(*worksetAxis, axis);
    }
    llvm::stable_sort(scalarWorksetAxes,
                      [](const auto &lhs, const auto &rhs) {
                        return lhs.first < rhs.first;
                      });
    llvm::SmallDenseSet<Attribute> structuredWorksetAxes;
    for (auto [_, axis] : scalarWorksetAxes)
      if (llvm::any_of(axes.lookup(axis), [&](MakeRangeOp range) {
            return contractFreeAxisSides(kernel, range) !=
                   ContractFreeAxisNone;
          }))
        structuredWorksetAxes.insert(axis);

    llvm::DenseMap<Attribute, int64_t> exactLocalExtents;
    bool exactLocalCoverage = llvm::all_of(
        internalOwnershipAxes, [&](Attribute axis) {
          std::optional<int64_t> extent;
          if (llvm::any_of(axes.lookup(axis), [&](MakeRangeOp range) {
                if (range->hasAttr(worksetCoordinateRangeAttr) ||
                    range->hasAttr(sourceSubregionAttr))
                  return true;
                std::optional<int64_t> current =
                    constantLogicalRangeCardinality(range);
                if (!current || (extent && *extent != *current))
                  return true;
                extent = *current;
                return false;
              }) ||
              !extent)
            return false;
          ParameterAttr parameter = lookupParameter(kernel, parameters.lookup(axis));
          if (!parameter ||
              !llvm::is_contained(
                  parameter.getCandidates().asArrayRef(),
                  *extent))
            return false;
          exactLocalExtents[axis] = *extent;
          return true;
        });

    if ((scalarWorksetAxes.size() >= 2 ||
         !structuredWorksetAxes.empty()) &&
        exactLocalCoverage) {
      // The existing non-workset ranges are exact program-local vectors.  Fix
      // them to full coverage and spend ownership dimensions on either the
      // two innermost Cartesian workset axes or an explicitly proven
      // contraction-free workset axis.  This changes both the fragment schema
      // and the launch mapping; no provider serializer inference is involved.
      for (Attribute axis : internalOwnershipAxes) {
        ParameterAttr parameter = lookupParameter(kernel, parameters.lookup(axis));
        auto updated = parameter.withCandidates(DenseI64ArrayAttr::get(
            module.getContext(), {exactLocalExtents.lookup(axis)})).withBinding(
                parameter.getBinding().withPointwiseLocal(true));
        if (failed(updateParameter(kernel, updated))) return failure();
        ownershipAxes.erase(axis);
        internalAxes.insert(axis);
      }
      for (auto [index, entry] : llvm::enumerate(scalarWorksetAxes))
        if (index + 2 < scalarWorksetAxes.size() &&
            !structuredWorksetAxes.contains(entry.second))
          ownershipAxes.erase(entry.second);
    } else {
      for (auto [_, axis] : scalarWorksetAxes)
        ownershipAxes.erase(axis);
    }
  }

  auto unitExtent = expression(module.getContext(), PhysicalExprKind::Constant, 1);
  for (auto [axis, ranges] : axes) {
    if (ownershipAxes.contains(axis))
      continue;
    for (MakeRangeOp range : ranges) {
      if (!range->hasAttr(worksetCoordinateRangeAttr) ||
          internalTraversalRanges.contains(range.getOperation()))
        continue;
      // An unselected workset coordinate remains a scalar program-grid
      // coordinate.  KIR construction provisionally gives every dynamic ABI
      // dimension a fragment parameter; once ownership selects the actual
      // lane axes, that provisional parameter must not remain in live value
      // types or the provider tuning surface.
      if (failed(retargetSourceExtent(
              range.getResult(),
              PhysicalSourceAxis{range.getSourceId(), range.getSourceAxis(),
                                 range.getDerived()},
              unitExtent)))
        return failure();
    }
  }

  SmallVector<Attribute> pointwiseOwnershipAxes;
  for (auto [axis, ranges] : axes) {
    if (!ownershipAxes.contains(axis))
      continue;
    ParameterAttr parameter = lookupParameter(kernel, parameters.lookup(axis));
    if (!parameter ||
        parameter.getRole() ==
            ParameterRole::ScanChunk)
      continue;
    pointwiseOwnershipAxes.push_back(axis);
  }

  bool orderedByStore = false;
  kernel.walk([&](StoreOp store) {
    if (orderedByStore)
      return;
    llvm::DenseMap<Attribute, int64_t> outputAxes;
    llvm::SmallDenseSet<int64_t> usedOutputAxes;
    for (Attribute axis : pointwiseOwnershipAxes) {
      std::optional<int64_t> outputAxis;
      for (MakeRangeOp range : axes.lookup(axis)) {
        std::optional<int64_t> projection = storeAxisForRange(store, range);
        if (!projection)
          continue;
        if (outputAxis && outputAxis != projection)
          return;
        outputAxis = projection;
      }
      if (!outputAxis || !usedOutputAxes.insert(*outputAxis).second)
        return;
      outputAxes[axis] = *outputAxis;
    }
    llvm::stable_sort(pointwiseOwnershipAxes, [&](Attribute lhs, Attribute rhs) {
      return outputAxes.lookup(lhs) < outputAxes.lookup(rhs);
    });
    orderedByStore = true;
    if (pointwiseOwnershipAxes.size() <= 2)
      return;
    Value copied = store.getValue();
    while (true) {
      if (auto transpose = copied.getDefiningOp<TransposeOp>()) {
        copied = transpose.getValue();
        continue;
      }
      if (auto broadcast = copied.getDefiningOp<BroadcastOp>()) {
        auto source = dyn_cast<FragmentType>(broadcast.getValue().getType());
        auto target = cast<FragmentType>(broadcast.getResult().getType());
        if (source && source.getShape().size() == target.getShape().size() &&
            queryAxisProjection(source, target).isExact()) {
          copied = broadcast.getValue();
          continue;
        }
      }
      break;
    }
    auto load = copied.getDefiningOp<LoadOp>();
    auto input = load ? dyn_cast<ViewType>(load.getResource().getType()) : ViewType();
    if (!input)
      return;
    Attribute inputAxis;
    for (auto [position, coordinate] : llvm::enumerate(load.getCoordinates())) {
      if (load.getSourceAxes()[position] + 1 != input.getRank())
        continue;
      auto range =
          stripAdditiveProjection(coordinate).getDefiningOp<MakeRangeOp>();
      if (!range || !isUnitStepRange(range) ||
          !queryFragmentAxis(coordinate.getType(), sourceAxisIdentity(range))
               .isExact())
        return;
      for (Attribute axis : pointwiseOwnershipAxes)
        if (llvm::is_contained(axes.lookup(axis), range)) {
          if (inputAxis && inputAxis != axis)
            return;
          inputAxis = axis;
        }
    }
    if (!inputAxis || inputAxis == pointwiseOwnershipAxes.back())
      return;
    ParameterAttr inputParameter = lookupParameter(kernel, parameters.lookup(inputAxis));
    if (!inputParameter || llvm::all_of(
            inputParameter.getCandidates().asArrayRef(),
            [](int64_t extent) { return extent == 1; }))
      return;
    bool commonOutputDirection = true;
    kernel.walk([&](StoreOp other) {
      auto output = dyn_cast<ViewType>(other.getResource().getType());
      commonOutputDirection &= output && other.getValue() == store.getValue() &&
          llvm::any_of(axes.lookup(pointwiseOwnershipAxes.back()),
                       [&](MakeRangeOp range) {
                         return storeAxisForRange(other, range) == output.getRank() - 1;
                       });
    });
    if (!commonOutputDirection)
      return;
    // Tile both ends of a permutation for row-major source/destination
    // locality. The access graph still carries the actual view strides.
    pointwiseOwnershipAxes.erase(llvm::find(pointwiseOwnershipAxes, inputAxis));
    pointwiseOwnershipAxes.insert(pointwiseOwnershipAxes.end() - 1, inputAxis);
  });

  bool pointwiseOnlyProgram = true;
  kernel.walk([&](Operation *operation) {
    pointwiseOnlyProgram &=
        !isa<ContractOp, ReduceOp, ScanOp, RegionFoldOp, RegionScanOp,
             ScaledContractOp, SparseContractOp, HistogramOp, ScatterReduceOp>(
            operation);
  });
  if (!pointwiseOnlyProgram && orderedByStore &&
      pointwiseOwnershipAxes.size() > 2 &&
      llvm::all_of(pointwiseOwnershipAxes, [&](Attribute axis) {
        return llvm::all_of(axes.lookup(axis), [&](MakeRangeOp range) {
          return contractFreeAxisSides(kernel, range) == ContractFreeAxisNone;
        });
      })) {
    PhysicalProgramAnalysis analysis(kernel);
    llvm::DenseMap<Attribute, uint64_t> invariantReadVolume;
    bool knownReads = true;
    auto ownerOfRange = [&](MakeRangeOp range) -> Attribute {
      MakeRangeOp occurrence = occurrenceRoots.lookup(range.getOperation());
      for (Attribute axis : pointwiseOwnershipAxes)
        if (llvm::any_of(axes.lookup(axis), [&](MakeRangeOp candidate) {
              return candidate == range ||
                     (occurrence && occurrenceRoots.lookup(
                                        candidate.getOperation()) == occurrence);
            }))
          return axis;
      return {};
    };
    kernel.walk([&](LoadOp load) {
      if (!knownReads)
        return;
      SmallVector<MakeRangeOp> roots;
      SmallVector<Value> dependencies(load.getCoordinates());
      if (load.getValid())
        dependencies.push_back(load.getValid());
      if (load.getFill())
        dependencies.push_back(load.getFill());
      for (Value value : dependencies) {
        auto fact = analysis.sourceRanges(value);
        if (!fact.blockers.empty() || !fact.accesses.empty()) {
          knownReads = false;
          return;
        }
        for (MakeRangeOp root : fact.roots)
          if (!llvm::is_contained(roots, root))
            roots.push_back(root);
      }
      uint64_t volume = physicalElementBitWidth(load.getResult().getType());
      llvm::SmallDenseSet<Attribute> countedOwners;
      llvm::SmallDenseSet<Attribute> varyingAxes;
      for (MakeRangeOp root : roots) {
        Attribute owner = ownerOfRange(root);
        if (owner)
          varyingAxes.insert(owner);
        // Estimate a fragment with the current inner output axis free and the
        // other program-grid axes fixed. Local/reduction ranges remain live.
        if (owner && owner != pointwiseOwnershipAxes.back())
          continue;
        if (owner && !countedOwners.insert(owner).second)
          continue;
        auto extent = constantLogicalRangeCardinality(root);
        if (!extent || *extent <= 0 ||
            volume > std::numeric_limits<uint64_t>::max() / *extent) {
          knownReads = false;
          return;
        }
        volume *= *extent;
      }
      for (Attribute axis : pointwiseOwnershipAxes) {
        if (varyingAxes.contains(axis))
          continue;
        uint64_t &score = invariantReadVolume[axis];
        if (score > std::numeric_limits<uint64_t>::max() - volume) {
          knownReads = false;
          return;
        }
        score += volume;
      }
    });
    Attribute selected = pointwiseOwnershipAxes[pointwiseOwnershipAxes.size() - 2];
    for (Attribute axis : llvm::drop_end(pointwiseOwnershipAxes)) {
      ParameterAttr parameter = lookupParameter(kernel, parameters.lookup(axis));
      if (knownReads &&
          llvm::any_of(parameter.getCandidates().asArrayRef(),
                       [](int64_t extent) { return extent > 1; }) &&
          invariantReadVolume.lookup(axis) > invariantReadVolume.lookup(selected))
        selected = axis;
    }
    // Keep the current output direction and tile the independent axis that
    // reuses the largest input fragment. This estimates repeated reads, not
    // pointer contiguity; the access graph retains the actual ABI strides.
    pointwiseOwnershipAxes.erase(llvm::find(pointwiseOwnershipAxes, selected));
    pointwiseOwnershipAxes.insert(pointwiseOwnershipAxes.end() - 1, selected);
  }

  for (auto [ownershipIndex, axis] :
       llvm::enumerate(pointwiseOwnershipAxes)) {
    ParameterAttr parameter = lookupParameter(kernel, parameters.lookup(axis));
    unsigned contractSides = ContractFreeAxisNone;
    unsigned matrixSides = ContractFreeAxisNone;
    bool batchedContraction = false;
    unsigned contractElementBitWidth = 0;
    for (MakeRangeOp range : axes.lookup(axis)) {
      ContractFreeAxisFacts facts = contractFreeAxisFacts(kernel, range);
      contractSides |= facts.sides;
      matrixSides |= facts.matrixSides;
      batchedContraction |= facts.batchedContraction;
      contractElementBitWidth =
          std::max(contractElementBitWidth, facts.operandElementBitWidth);
    }
    const bool contractionAxis = contractSides != ContractFreeAxisNone;
    const bool scalarGridAxis = !pointwiseOnlyProgram && !contractionAxis &&
        ownershipIndex + 2 < pointwiseOwnershipAxes.size();
    ParameterRole ownershipRole = ParameterRole::OwnershipN;
    auto declaredRole =
        parameter.getRole();
    // Matrix operands constrain M/N orientation. A vector contraction with
    // other output axes uses their existing store order, so independent axes
    // do not accidentally consume the same profile column.
    unsigned orientedSides = pointwiseOwnershipAxes.size() == 1
                                 ? contractSides
                                 : matrixSides;
    if (orientedSides == ContractFreeAxisLhs)
      ownershipRole = ParameterRole::OwnershipM;
    else if (orientedSides == ContractFreeAxisRhs)
      ownershipRole = ParameterRole::OwnershipN;
    else if (parameter.getCategory() ==
            ParameterCategory::Contraction &&
        (declaredRole == ParameterRole::OwnershipM ||
         declaredRole == ParameterRole::OwnershipN))
      ownershipRole = declaredRole;
    else if (!scalarGridAxis &&
             ownershipIndex + 1 < pointwiseOwnershipAxes.size())
      ownershipRole = ParameterRole::OwnershipM;
    DenseI64ArrayAttr candidates;
    if (scalarGridAxis)
      candidates = DenseI64ArrayAttr::get(module.getContext(), {1});
    else if (isSourceAxisKey(axis))
      candidates = parameter.getCandidates();
    else
      candidates = DenseI64ArrayAttr::get(
          module.getContext(),
          {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
           8192, 16384, 32768, 65536});
    ParameterCategory category = parameter.getCategory();
    if (!scalarGridAxis && batchedContraction &&
        (contractSides == ContractFreeAxisLhs ||
         contractSides == ContractFreeAxisRhs) &&
        category == ParameterCategory::Pointwise)
      category = ParameterCategory::Contraction;
    auto schema = ParameterAttr::get(
        module.getContext(), parameter.getName(), parameter.getValueType(),
        ownershipRole, category,
        category == ParameterCategory::Contraction &&
                contractElementBitWidth != 0
            ? contractElementBitWidth
            : pointwiseElementBitWidth,
        candidates, ConfigurationBindingPhase::Shared,
        parameter.getBinding().withCoverageBound({}));
    if (failed(updateParameter(kernel, schema))) return failure();
    if (category == ParameterCategory::Contraction)
      contractionCoordinateRoles[axis] =
          ownershipRole == ParameterRole::OwnershipM
              ? CoordinateRole::ContractionM
              : CoordinateRole::ContractionN;
  }
  if (llvm::count_if(contractionCoordinateRoles, [](const auto &entry) {
        return entry.second == CoordinateRole::ContractionM;
      }) != 1 ||
      llvm::count_if(contractionCoordinateRoles, [](const auto &entry) {
        return entry.second == CoordinateRole::ContractionN;
      }) != 1)
    contractionCoordinateRoles.clear();


  llvm::MapVector<Attribute, SmallVector<MakeRangeOp>> selectedAxes;
  SmallVector<MakeRangeOp> selectedRanges;
  for (auto [axis, ranges] : axes) {
    if (!ownershipAxes.contains(axis))
      continue;
    for (MakeRangeOp range : ranges) {
      if (internalTraversalRanges.contains(range.getOperation()))
        continue;
      selectedAxes[axis].push_back(range);
      selectedRanges.push_back(range);
    }
  }
  axes = std::move(selectedAxes);
  dynamicRanges = std::move(selectedRanges);
  return success();
}


LogicalResult PointwiseRewrite::mapOwnership() {
  OpBuilder mappingBuilder(mapping);
  SmallVector<Value> runtimeExtents(mapping.getExtents());
  SmallVector<Attribute> launchExtents(mapping.getLaunchExtents().begin(),
                                       mapping.getLaunchExtents().end());
  auto existing = readMappingCoordinates(kernel, mapping);
  if (failed(existing)) return failure();
  tileCoordinates = existing->tiles;
  auto &mappedAxes = existing->axes;
  auto &coordinateAxes = existing->coordinates;
  auto &reusableUnitAxis = existing->reusableUnit;
  bool mappingChanged = false;
  SmallVector<std::pair<Attribute, unsigned>> reusedCoordinates;
  SmallVector<Attribute> appendedCoordinates;
  for (auto [axisKey, ranges] : axes) {
    MakeRangeOp range = ranges.front();
    auto position = reconcileCoordinate(range, axisKey, *existing);
    if (failed(position)) return failure();
    std::optional<unsigned> worksetPosition = *position;
    tileCoordinates = existing->tiles;
    ParameterAttr parameter = lookupParameter(kernel, parameters.lookup(axisKey));
    if (!parameter) return range.emitOpError("pointwise mapping lost its blocking parameter");
    if (!ownershipAxes.contains(axisKey)) continue;
    Value dimension;
    PhysicalExprAttr derivedLogicalExtent;
    std::optional<int64_t> staticExtent;
    FailureOr<uint64_t> sourceDimension = ownershipDimension(kernel, range);
    if (worksetPosition) {
      // A workset coordinate is expressed in source-coordinate units, while
      // ownership tiles the finite workset instance domain.  Reuse the
      // Delinearize runtime cardinality here; its launch expression below is
      // the exact same relation and may include non-zero starts or non-unit
      // domain steps.
      dimension = mapping.getExtents()[*worksetPosition];
    } else if (isSourceAxisKey(axisKey)) {
      staticExtent = constantLogicalRangeCardinality(range);
      if (staticExtent)
        dimension = mappingBuilder.create<arith::ConstantIndexOp>(
            mapping.getLoc(), *staticExtent);
      else if (succeeded(sourceDimension)) {
        FailureOr<Value> argument =
            dimensionArgument(kernel, *sourceDimension);
        if (succeeded(argument))
          dimension = *argument;
      }
    } else {
      FailureOr<uint64_t> dimensionId = axisDimension(axisKey);
      if (succeeded(dimensionId)) {
        FailureOr<Value> argument = dimensionArgument(kernel, *dimensionId);
        if (succeeded(argument))
          dimension = *argument;
      }
    }
    if (!parameter) {
      InFlightDiagnostic diagnostic = range.emitOpError(
          "dynamic pointwise range has no launch-visible dimension or blocking parameter");
      diagnostic << "; axis=" << axisKey << ", fragment="
                 << range.getResult().getType();
      return failure();
    }
    if (!dimension) {
      for (MakeRangeOp other : ranges) {
        auto extent = queryLaunchRangeExtent(other);
        if (!extent) {
          derivedLogicalExtent = {};
          break;
        }
        if (!derivedLogicalExtent)
          derivedLogicalExtent = extent;
        else if (derivedLogicalExtent != extent)
          derivedLogicalExtent = binaryExpression(
              module.getContext(), PhysicalExprKind::Maximum,
              derivedLogicalExtent, extent);
      }
      if (derivedLogicalExtent)
        dimension = mappingBuilder.create<PhysicalExprOp>(
            mapping.getLoc(), mappingBuilder.getIndexType(),
            derivedLogicalExtent);
    }
    if (!dimension) {
      InFlightDiagnostic diagnostic = range.emitOpError(
          "dynamic ownership range has no launch-visible logical dimension");
      diagnostic << "; axis=" << axisKey << ", fragment="
                 << range.getResult().getType() << ", parameter="
                 << parameter.getName().getValue();
      return failure();
    }
    logicalDimensions[axisKey] = dimension;
    PhysicalExprAttr logical;
    if (worksetPosition) {
      logical = cast<PhysicalExprAttr>(
          mapping.getLaunchExtents()[*worksetPosition]);
    } else if (derivedLogicalExtent) {
      logical = derivedLogicalExtent;
    } else if (isSourceAxisKey(axisKey) && staticExtent) {
      logical = expression(module.getContext(), PhysicalExprKind::Constant,
                           *staticExtent);
    } else {
      FailureOr<uint64_t> dimensionId = axisDimension(axisKey);
      uint64_t logicalDimension =
          isSourceAxisKey(axisKey) && succeeded(sourceDimension)
              ? *sourceDimension
              : succeeded(dimensionId) ? *dimensionId : 0;
      if (logicalDimension == 0)
        return range.emitOpError(
            "range-local ownership axis lost its logical dimension extent");
      logical = queryArgumentExpression(resolveDimension(kernel, logicalDimension));
      if (!logical)
        return range.emitOpError("range-local ownership dimension has no runtime binding");
    }
    PhysicalExprAttr tile = expression(
        module.getContext(), PhysicalExprKind::Parameter, 0,
        parameter.getName().getValue());
    if (llvm::all_of(ranges, isUnitStepRange) &&
        llvm::any_of(ranges, [&](MakeRangeOp candidate) {
          return contractFreeAxisFacts(kernel, candidate).matrixSides !=
                 ContractFreeAxisNone;
        })) {
      auto bounded = boundedOwnershipExtent(kernel, parameter, logical);
      if (failed(bounded)) return failure();
      if (*bounded != tile) {
        tile = *bounded;
        ownershipExtents[axisKey] = tile;
      }
    }
    PhysicalExprAttr launch = binaryExpression(
        module.getContext(), PhysicalExprKind::CeilDiv, logical, tile);
    Value tiles = mappingBuilder.create<PhysicalExprOp>(
        mapping.getLoc(), mappingBuilder.getIndexType(), launch);
    auto mapped = mappedAxes.find(axisKey);
    if (mapped != mappedAxes.end() || reusableUnitAxis) {
      unsigned axis = mapped != mappedAxes.end() ? mapped->second
                                                 : *reusableUnitAxis;
      runtimeExtents[axis] = tiles;
      launchExtents[axis] = launch;
      reusedCoordinates.emplace_back(axisKey, axis);
      if (mapped == mappedAxes.end())
        reusableUnitAxis.reset();
    } else {
      runtimeExtents.push_back(tiles);
      launchExtents.push_back(launch);
      appendedCoordinates.push_back(axisKey);
    }
    mappingChanged = true;
  }

  if (mappingChanged) {
    SmallVector<int64_t> coordinateRoles(
        runtimeExtents.size(),
        static_cast<int64_t>(CoordinateRole::Unspecified));
    if (auto existing =
            mapping.getCoordinateRolesAttr())
      if (existing.size() == mapping.getCoordinates().size())
        llvm::copy(existing.asArrayRef(), coordinateRoles.begin());
    auto ownershipRole = [&](Attribute axis) {
      auto contraction = contractionCoordinateRoles.find(axis);
      if (contraction != contractionCoordinateRoles.end())
        return contraction->second;
      return llvm::any_of(axes.lookup(axis), [](MakeRangeOp range) {
               return range->hasAttr(worksetCoordinateRangeAttr);
             })
                 ? CoordinateRole::TiledWorkset
                 : CoordinateRole::PointwiseOwnership;
    };
    for (auto [axisKey, axis] : reusedCoordinates) {
      if (coordinateRoles[axis] !=
          static_cast<int64_t>(CoordinateRole::IndirectTraversal))
        coordinateRoles[axis] =
            static_cast<int64_t>(ownershipRole(axisKey));
    }
    unsigned appendedAxis = mapping.getCoordinates().size();
    for (Attribute axisKey : appendedCoordinates) {
      coordinateRoles[appendedAxis++] =
          static_cast<int64_t>(ownershipRole(axisKey));
    }
    PhysicalExprAttr total = cast<PhysicalExprAttr>(launchExtents.front());
    for (Attribute extent : llvm::drop_begin(launchExtents))
      total = binaryExpression(module.getContext(), PhysicalExprKind::Multiply,
                               total, cast<PhysicalExprAttr>(extent));
    auto replacement = rebuildExecutionGroup(
        mappingBuilder, mapping, runtimeExtents,
        mappingBuilder.getArrayAttr(launchExtents),
        mappingBuilder.getDenseI64ArrayAttr(coordinateRoles), total);
    mappingBuilder.setInsertionPointToStart(&replacement.getBody().front());
    for (auto [axis, pair] : llvm::enumerate(llvm::zip(
             mapping.getCoordinates(),
             replacement.getCoordinates().take_front(
                 mapping.getCoordinates().size())))) {
      Value oldCoordinate = std::get<0>(pair);
      Value newCoordinate = std::get<1>(pair);
      auto knownAxis = coordinateAxes.find(axis);
      FailureOr<Attribute> axisKey =
          knownAxis != coordinateAxes.end()
              ? FailureOr<Attribute>(knownAxis->second)
              : axis < mapping.getLaunchExtents().size()
                    ? pointwise::mappingAxis(kernel, mapping.getLaunchExtents()[axis])
                    : FailureOr<Attribute>(failure());
      if (succeeded(axisKey) && ownershipAxes.contains(*axisKey)) {
        ParameterAttr parameter = lookupParameter(kernel, parameters.lookup(*axisKey));
        if (!parameter)
          return kernel.emitError(
              "pointwise ownership mapping lost its blocking parameter");
        Value extent = materializeParameter(
            mappingBuilder, mapping.getLoc(), parameter.getReference());
        if (PhysicalExprAttr bounded = ownershipExtents.lookup(*axisKey))
          extent = mappingBuilder.create<PhysicalExprOp>(
              mapping.getLoc(), mappingBuilder.getIndexType(), bounded);
        newCoordinate = mappingBuilder.create<BinaryOp>(
            mapping.getLoc(), mappingBuilder.getIndexType(), newCoordinate,
            extent, BinaryOperator::Multiply);
        oldCoordinate.replaceAllUsesWith(newCoordinate);
        continue;
      }
      oldCoordinate.replaceAllUsesWith(newCoordinate);
    }
    for (auto [axisKey, axis] : reusedCoordinates)
      tileCoordinates[axisKey] = replacement.getCoordinates()[axis];
    unsigned extra = mapping.getCoordinates().size();
    for (Attribute axisKey : appendedCoordinates)
      tileCoordinates[axisKey] = replacement.getCoordinates()[extra++];
    mapping.erase();
    mapping = replacement;
    kernel->setAttr(programSpaceAttr,
                    ArrayAttr::get(module.getContext(), {total}));
  }
  return success();
}


FailureOr<std::optional<unsigned>> PointwiseRewrite::reconcileCoordinate(MakeRangeOp range, Attribute axisKey, MappingCoordinates &coordinates) {
  const bool worksetRange = range->hasAttr(worksetCoordinateRangeAttr);
  std::optional<unsigned> worksetPosition;
  if (worksetRange) {
    auto coordinate = range.getStart().getDefiningOp<WorksetCoordinateOp>();
    auto worksetAxis =
        coordinate
            ? coordinate->getAttrOfType<IntegerAttr>(worksetAxisAttr)
            : IntegerAttr();
    if (!worksetAxis || worksetAxis.getInt() < 0 ||
        static_cast<size_t>(worksetAxis.getInt()) >=
            mapping.getCoordinates().size())
      return range.emitOpError(
          "workset ownership has no corresponding program coordinate");
    unsigned position = static_cast<unsigned>(worksetAxis.getInt());
    worksetPosition = position;
    auto existingCoordinate = coordinates.coordinates.find(position);
    if (existingCoordinate != coordinates.coordinates.end() &&
        existingCoordinate->second != axisKey) {
      auto dimension = axisDimension(existingCoordinate->second);
      auto worksetDimension = queryRangeDimension(range);
      if (!isSourceAxisKey(axisKey) || failed(dimension) ||
          failed(worksetDimension) ||
          *dimension != coordinate.getDimensionId() ||
          *dimension != *worksetDimension ||
          ownershipAxes.contains(existingCoordinate->second))
        return range.emitOpError(
            "workset ownership conflicts with the existing program coordinate relation");
      // The initial launch names the dimension; the workset range names its
      // exact source occurrence at this same coordinate position.
      coordinates.axes.erase(existingCoordinate->second);
      coordinates.tiles.erase(existingCoordinate->second);
    }
    auto existingAxis = coordinates.axes.find(axisKey);
    if (existingAxis != coordinates.axes.end() && existingAxis->second != position)
      return range.emitOpError(
          "workset ownership axis maps to multiple program coordinates");
    coordinates.axes[axisKey] = position;
    coordinates.coordinates[position] = axisKey;
  }  return worksetPosition;
}


} // namespace intent::gpu::pointwise
