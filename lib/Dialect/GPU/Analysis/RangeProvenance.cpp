#include "PhysicalProgramDetail.h"
#include "ScalarExpressions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLExtras.h"

#include <functional>

using namespace mlir;

namespace intent::gpu {
using namespace detail;

namespace {

// Forwarding slots describe the structural boundary. Range analysis may also
// exclude a counted loop's statically unreachable exit, without changing the
// type constraints that the boundary mutator must preserve on every edge.
ControlFlowEdges executableIncoming(Value value) {
  auto incoming = queryControlFlowIncoming(value);
  if (auto loop = value.getDefiningOp<scf::ForOp>()) {
    auto lower = integerConstant(loop.getLowerBound());
    auto upper = integerConstant(loop.getUpperBound());
    if (lower && upper)
      llvm::erase_if(incoming.edges, [&](const ControlFlowEdge &edge) {
        return (*lower < *upper && edge.kind == ControlFlowEdgeKind::Bypass) ||
               (*lower >= *upper && edge.kind == ControlFlowEdgeKind::Exit);
      });
  }
  return incoming;
}

} // namespace

void PhysicalProgramAnalysis::collectRanges(
    Value value, std::optional<PhysicalSourceAxis> source,
    PhysicalRangeFact &result,
    SmallPtrSetImpl<Operation *> &visited, bool followScalarDependencies) {
  if (!value)
    return;
  if (!followScalarDependencies) {
    auto fragment = dyn_cast<FragmentType>(value.getType());
    if (isa<IntegerType, FloatType, IndexType>(value.getType()) ||
        (fragment && fragment.getShape().empty()))
      return;
  }
  if (auto extract = value.getDefiningOp<ExtractOp>()) {
    if (auto record = extract.getRecord().getDefiningOp<MakeRecordOp>()) {
      uint64_t field = extract.getField();
      if (field >= record.getFields().size()) {
        result.state = PhysicalFactState::Unknown;
        appendUnique(result.blockers, extract);
        return;
      }
      collectRanges(record.getFields()[field], source, result, visited,
                    followScalarDependencies);
      return;
    }
  }
  if (source && !carriesSource(value.getType(), *source))
    return;
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    // Scalar ABI/control coordinates do not introduce a fragment range.
    if (isa<IntegerType, FloatType, IndexType>(argument.getType()))
      return;
    SmallVector<Value> pending{argument};
    llvm::DenseSet<Value> arguments;
    while (!pending.empty()) {
      Value current = pending.pop_back_val();
      auto formal = dyn_cast<BlockArgument>(current);
      if (!formal) {
        collectRanges(current, source, result, visited, followScalarDependencies);
        continue;
      }
      if (isa<IntegerType, FloatType, IndexType>(formal.getType()) ||
          (source && !carriesSource(formal.getType(), *source)))
        continue;
      if (!followScalarDependencies)
        if (auto fragment = dyn_cast<FragmentType>(formal.getType());
            fragment && fragment.getShape().empty())
          continue;
      if (!arguments.insert(formal).second) continue;
      auto outer = structuredSourcesForArgument(formal);
      if (outer.empty()) {
        result.state = PhysicalFactState::Unknown;
        appendUnique(result.blockers, formal.getOwner()->getParentOp());
      } else {
        llvm::append_range(pending, outer);
      }
    }
    return;
  }
  Operation *operation = value.getDefiningOp();
  if (!operation || !visited.insert(operation).second)
    return;
  if (auto range = dyn_cast<MakeRangeOp>(operation)) {
    if (!source || sourceAxisIdentity(range) == *source)
      appendUnique(result.roots, range);
    return;
  }
  if (isa<RegionBranchOpInterface>(operation)) {
    if (auto fragment = dyn_cast<FragmentType>(value.getType())) {
      // A completed loop contributes its result lanes to an indirect access,
      // not the reduction lanes used to compute each index. Follow the typed
      // yield/carry relation already used by axis-specific provenance.
      for (auto [axis, attribute] : llvm::enumerate(fragment.getAxisMaps())) {
        if (source &&
            !(sourceAxisIdentity(cast<AxisMapAttr>(attribute)) == *source))
          continue;
        PhysicalRangeFact fact = axisRanges(value, axis);
        if (result.state != PhysicalFactState::Unknown &&
            fact.state != PhysicalFactState::Exact)
          result.state = fact.state;
        for (MakeRangeOp range : fact.roots)
          appendUnique(result.roots, range);
        for (Operation *access : fact.accesses)
          appendUnique(result.accesses, access);
        for (Operation *blocker : fact.blockers)
          appendUnique(result.blockers, blocker);
      }
      return;
    }
  }
  // Reshape preserves the row-major coordinate relation carried by its
  // reassociation groups.  Source-specific range queries follow that typed
  // relation through the input instead of treating reshape as an opaque value
  // producer.  Any genuinely incompatible roots remain ambiguous in
  // sourceRanges(), where all collected ranges are compared.
  if (auto reshape = dyn_cast<ReshapeOp>(operation)) {
    collectRanges(reshape.getValue(), source, result, visited,
                  followScalarDependencies);
    return;
  }
  // These operations are typed coordinate leaves.  They do not contribute a
  // fragment range root, but reaching one is an exact end of provenance rather
  // than an unknown operation in the producer graph.
  if (isa<arith::ConstantOp, PhysicalExprOp, ParameterOp, ProgramIdOp, DelinearizeOp,
          WorksetCoordinateOp, DimOp, RangeOp, RangeBoundOp>(operation))
    return;
  if (auto scan = dyn_cast<ScanOp>(operation)) {
    bool followed = false;
    for (Value scanSource : scan.getSources()) {
      if (source && !carriesSource(scanSource.getType(), *source))
        continue;
      followed = true;
      collectRanges(scanSource, source, result, visited, followScalarDependencies);
    }
    if (!followed) {
      appendUnique(result.blockers, operation);
      result.state = PhysicalFactState::Unknown;
    }
    return;
  }
  if (isAccessNode(operation))
    appendUnique(result.accesses, operation);
  if (!isValueReplayNode(operation) && !isAccessNode(operation)) {
    appendUnique(result.blockers, operation);
    result.state = PhysicalFactState::Unknown;
    return;
  }
  auto load = dyn_cast<LoadOp>(operation);
  for (Value operand : operation->getOperands()) {
    // Resource identity stays in the access fact, not in fragment range provenance.
    if (load && operand == load.getResource())
      continue;
    collectRanges(operand, source, result, visited, followScalarDependencies);
  }
}

PhysicalRangeFact PhysicalProgramAnalysis::sourceRanges(
    Value value, std::optional<PhysicalSourceAxis> source) {
  if (!source) {
    if (auto cached = unrestrictedRangeCache.find(value);
        cached != unrestrictedRangeCache.end())
      return cached->second;
  }
  PhysicalRangeFact result;
  result.state = PhysicalFactState::Exact;
  SmallPtrSet<Operation *, 32> visited;
  collectRanges(value, source, result, visited);
  if (result.state == PhysicalFactState::Unknown || !result.blockers.empty() ||
      result.roots.empty())
    result.state = PhysicalFactState::Unknown;
  else if (result.roots.size() > 1) {
    MakeRangeOp authority = result.roots.front();
    if (!llvm::all_of(result.roots, [&](MakeRangeOp range) {
          return sameLogicalRange(authority, range);
        }))
      result.state = PhysicalFactState::Ambiguous;
  }
  result.unitStep = !result.roots.empty() &&
                    llvm::all_of(result.roots, isUnitStepRange);
  if (!source)
    unrestrictedRangeCache.try_emplace(value, result);
  return result;
}

PhysicalRangeFact
PhysicalProgramAnalysis::programRanges(PhysicalSourceAxis source) {
  PhysicalRangeFact result;
  result.state = PhysicalFactState::Exact;
  kernel.walk([&](MakeRangeOp range) {
    if (!(sourceAxisIdentity(range) == source))
      return;
    appendUnique(result.roots, range);
  });
  if (result.roots.empty()) {
    result.state = PhysicalFactState::Unknown;
    return result;
  }
  MakeRangeOp authority = result.roots.front();
  if (!llvm::all_of(result.roots, [&](MakeRangeOp range) {
        return sameLogicalRange(authority, range);
      }))
    result.state = PhysicalFactState::Ambiguous;
  result.unitStep = llvm::all_of(result.roots, isUnitStepRange);
  return result;
}

void PhysicalProgramAnalysis::collectAxisRanges(
    Value value, unsigned fragmentAxis, PhysicalRangeFact &result,
    llvm::DenseSet<std::pair<Value, unsigned>> &visited) {
  if (!value || !visited.insert({value, fragmentAxis}).second)
    return;
  auto fragment = dyn_cast<FragmentType>(value.getType());
  if (!fragment || fragmentAxis >= fragment.getShape().size()) {
    result.state = PhysicalFactState::Unknown;
    return;
  }
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    if (auto reduce = dyn_cast<ReduceOp>(argument.getOwner()->getParentOp())) {
      auto structured = cast<StructuredOpInterface>(reduce.getOperation());
      for (auto [capture, formal] : llvm::zip_equal(reduce.getCaptures(), structured.getCombineCaptures()))
        if (formal == argument) {
          collectAxisRanges(capture, fragmentAxis, result, visited);
          return;
        }
      Value source;
      for (auto [input, lhs, rhs] : llvm::zip_equal(reduce.getSources(),
               structured.getCombineLhs(), structured.getCombineRhs()))
        if (argument == lhs || argument == rhs) source = input;
      if (!source) {
        result.state = PhysicalFactState::Unknown;
        appendUnique(result.blockers, reduce);
        return;
      }
      auto sourceType = dyn_cast<FragmentType>(source.getType());
      if (!sourceType) {
        result.state = PhysicalFactState::Unknown;
        appendUnique(result.blockers, reduce);
        return;
      }
      SmallVector<unsigned> freeAxes;
      for (unsigned axis = 0; axis < sourceType.getShape().size(); ++axis)
        if (!llvm::is_contained(reduce.getAxes(), static_cast<int64_t>(axis)))
          freeAxes.push_back(axis);
      if (fragmentAxis >= freeAxes.size()) {
        result.state = PhysicalFactState::Unknown;
        appendUnique(result.blockers, reduce);
        return;
      }
      collectAxisRanges(source, freeAxes[fragmentAxis], result, visited);
      return;
    }
    SmallVector<Value, 2> outer = structuredSourcesForArgument(argument);
    bool followed = false;
    for (Value related : outer) {
      auto outerFragment = dyn_cast<FragmentType>(related.getType());
      if (!outerFragment || fragmentAxis >= outerFragment.getShape().size())
        continue;
      followed = true;
      collectAxisRanges(related, fragmentAxis, result, visited);
    }
    if (!followed) {
      result.state = PhysicalFactState::Unknown;
      appendUnique(result.blockers, argument.getOwner()->getParentOp());
      return;
    }
    return;
  }
  Operation *operation = value.getDefiningOp();
  if (!operation)
    return;
  if (auto range = dyn_cast<MakeRangeOp>(operation)) {
    if (fragmentAxis == 0)
      appendUnique(result.roots, range);
    else
      result.state = PhysicalFactState::Unknown;
    return;
  }
  if (isa<FragmentOpInterface>(operation)) {
    auto relations = queryFragmentOperandRelations(cast<OpResult>(value));
    if (failed(relations)) {
      result.state = PhysicalFactState::Unknown;
      appendUnique(result.blockers, operation);
      return;
    }
    if (!relations->empty() &&
        llvm::all_of(*relations, [&](const FragmentOperandRelation &relation) {
          return llvm::is_contained(relation.invariantResultAxes, fragmentAxis);
        }))
      return;
    auto expected = cast<AxisMapAttr>(fragment.getAxisMaps()[fragmentAxis]);
    for (const FragmentOperandRelation &relation : *relations) {
      auto input = dyn_cast<FragmentType>(relation.sourceType);
      if (!input)
        continue;
      Value operand = operation->getOperand(relation.operandNumber);
      const FragmentAxisGroup *group = relation.groupForResultAxis(fragmentAxis);
      if (!group) {
        result.state = PhysicalFactState::Unknown;
        appendUnique(result.blockers, operation);
        continue;
      }
      std::optional<unsigned> sourceAxis;
      if (group->sourceAxes.size() == 1 && group->resultAxes.size() == 1) {
        sourceAxis = group->sourceAxes.front();
      } else if (group->kind == FragmentAxisRelationKind::Reassociation) {
        // A row-major group is exact, but selecting a single provenance root
        // within a split/merge still needs this analysis's occurrence proof.
        for (unsigned axis : group->sourceAxes) {
          auto mapping = cast<AxisMapAttr>(input.getAxisMaps()[axis]);
          if (mapping.getDimensionId() != expected.getDimensionId())
            continue;
          if (sourceAxis) {
            result.state = PhysicalFactState::Ambiguous;
            appendUnique(result.blockers, operation);
            return;
          }
          sourceAxis = axis;
        }
      }
      if (!sourceAxis) {
        auto extent = cast<PhysicalExprAttr>(fragment.getShape()[fragmentAxis]);
        if (group->kind != FragmentAxisRelationKind::Broadcast &&
            !(extent.getKind() == PhysicalExprKind::Constant &&
              extent.getValue() == 1)) {
          result.state = PhysicalFactState::Unknown;
          appendUnique(result.blockers, operation);
        }
        continue;
      }
      if (group->kind == FragmentAxisRelationKind::Broadcast) {
        PhysicalRangeFact inputRanges;
        inputRanges.state = PhysicalFactState::Exact;
        collectAxisRanges(operand, *sourceAxis, inputRanges, visited);
        // A physical unit may be an unexpanded construction seed. Only a
        // proven singleton logical range makes its broadcast provenance empty.
        if (inputRanges.state != PhysicalFactState::Unknown &&
            inputRanges.blockers.empty() &&
            llvm::all_of(inputRanges.roots, isProvablySingletonLogicalRange))
          continue;
        if (result.state != PhysicalFactState::Unknown &&
            inputRanges.state != PhysicalFactState::Exact)
          result.state = inputRanges.state;
        for (MakeRangeOp root : inputRanges.roots)
          appendUnique(result.roots, root);
        for (Operation *access : inputRanges.accesses)
          appendUnique(result.accesses, access);
        for (Operation *blocker : inputRanges.blockers)
          appendUnique(result.blockers, blocker);
      } else {
        collectAxisRanges(operand, *sourceAxis, result, visited);
      }
    }
    return;
  }
  if (auto scan = dyn_cast<ScanOp>(operation)) {
    auto expected =
        cast<AxisMapAttr>(fragment.getAxisMaps()[fragmentAxis]);
    bool followed = false;
    for (Value scanSource : scan.getSources()) {
      auto sourceType = dyn_cast<FragmentType>(scanSource.getType());
      if (!sourceType || fragmentAxis >= sourceType.getShape().size())
        continue;
      auto mapping =
          cast<AxisMapAttr>(sourceType.getAxisMaps()[fragmentAxis]);
      if (!(sourceAxisIdentity(mapping) == sourceAxisIdentity(expected)) ||
          mapping.getDimensionId() != expected.getDimensionId())
        continue;
      followed = true;
      collectAxisRanges(scanSource, fragmentAxis, result, visited);
    }
    if (!followed) {
      result.state = PhysicalFactState::Unknown;
      appendUnique(result.blockers, operation);
    }
    return;
  }
  if (auto extract = dyn_cast<ExtractOp>(operation)) {
    bool followed = false;
    llvm::SmallPtrSet<Value, 8> visitedRecords;
    auto unknownRecord = [&](Value recordValue) {
      result.state = PhysicalFactState::Unknown;
      Operation *blocker = recordValue ? recordValue.getDefiningOp() : nullptr;
      if (auto argument = dyn_cast_or_null<BlockArgument>(recordValue))
        blocker = argument.getOwner()->getParentOp();
      appendUnique(result.blockers, blocker ? blocker : extract.getOperation());
    };
    std::function<void(Value)> collectRecordField = [&](Value recordValue) {
      if (!recordValue) {
        unknownRecord(recordValue);
        return;
      }
      // Reaching an already visited carry closes a cycle; it is not another
      // unknown branch. Each distinct incoming source is checked separately.
      if (!visitedRecords.insert(recordValue).second)
        return;
      if (auto record = recordValue.getDefiningOp<MakeRecordOp>()) {
        if (extract.getField() >= record.getFields().size()) {
          unknownRecord(recordValue);
          return;
        }
        Value field = record.getFields()[extract.getField()];
        auto type = dyn_cast<FragmentType>(field.getType());
        if (!type || fragmentAxis >= type.getShape().size()) {
          unknownRecord(recordValue);
          return;
        }
        followed = true;
        collectAxisRanges(field, fragmentAxis, result, visited);
        return;
      }
      if (auto argument = dyn_cast<BlockArgument>(recordValue)) {
        auto sources = structuredSourcesForArgument(argument);
        if (sources.empty())
          unknownRecord(recordValue);
        for (Value related : sources)
          collectRecordField(related);
        return;
      }
      auto opResult = dyn_cast<OpResult>(recordValue);
      if (!opResult) {
        unknownRecord(recordValue);
        return;
      }
      if (auto fold = dyn_cast<RegionFoldOp>(opResult.getOwner())) {
        unsigned component = opResult.getResultNumber();
        if (component >= fold.getIdentities().size()) {
          unknownRecord(recordValue);
          return;
        }
        collectRecordField(fold.getIdentities()[component]);
        Region &summarize = fold.getSummarize();
        if (summarize.empty() || summarize.front().empty()) {
          unknownRecord(recordValue);
          return;
        }
        auto yield = dyn_cast<YieldOp>(summarize.front().back());
        if (!yield || component >= yield.getValues().size()) {
          unknownRecord(recordValue);
          return;
        }
        collectRecordField(yield.getValues()[component]);
        return;
      }
      if (isa<RegionBranchOpInterface>(opResult.getOwner())) {
        auto incoming = executableIncoming(opResult);
        if (!incoming.complete || incoming.edges.empty())
          unknownRecord(recordValue);
        for (const ControlFlowEdge &edge : incoming.edges)
          if (edge.operand)
            collectRecordField(edge.operand->get());
          else
            unknownRecord(recordValue);
        return;
      }
      unknownRecord(recordValue);
    };
    collectRecordField(extract.getRecord());
    if (!followed)
      unknownRecord(extract.getRecord());
    return;
  }
  if (auto gather = dyn_cast<GatherOp>(operation)) {
    auto input = dyn_cast<FragmentType>(gather.getSource().getType());
    auto expected = cast<AxisMapAttr>(fragment.getAxisMaps()[fragmentAxis]);
    auto occurrence = [&](FragmentType type) {
      return queryFragmentAxis(type, sourceAxisIdentity(expected),
                               expected.getDimensionId());
    };
    auto retained = input ? occurrence(input) : PhysicalAxisProjection{};
    auto output = occurrence(fragment);
    if (retained.isExact() && retained.dimensionId == expected.getDimensionId() &&
        output.isExact() && output.fragmentAxis == fragmentAxis &&
        !llvm::is_contained(gather.getSourceAxes(), retained.fragmentAxis) &&
        input.getShape()[retained.fragmentAxis] == fragment.getShape()[fragmentAxis]) {
      collectAxisRanges(gather.getSource(), retained.fragmentAxis, result, visited);
      return;
    }
    bool followed = false;
    for (Value coordinate : gather.getCoordinates()) {
      auto type = dyn_cast<FragmentType>(coordinate.getType());
      auto projection = type ? occurrence(type)
                             : PhysicalAxisProjection{};
      if (!projection.isExact() || projection.dimensionId != expected.getDimensionId() ||
          type.getShape()[projection.fragmentAxis] != fragment.getShape()[fragmentAxis])
        continue;
      collectAxisRanges(coordinate, projection.fragmentAxis, result, visited);
      followed = true;
    }
    if (followed)
      return;
    result.state = PhysicalFactState::Unknown;
    appendUnique(result.blockers, operation);
    return;
  }
  if (auto load = dyn_cast<LoadOp>(operation)) {
    appendUnique(result.accesses, operation);
    auto expected = cast<AxisMapAttr>(fragment.getAxisMaps()[fragmentAxis]);
    SmallVector<Value> positionalCoordinates;
    bool rankOneCoordinates = true;
    for (Value coordinate : load.getCoordinates()) {
      auto type = dyn_cast<FragmentType>(coordinate.getType());
      if (!type)
        continue;
      rankOneCoordinates &= type.getShape().size() == 1;
      positionalCoordinates.push_back(coordinate);
    }
    bool cartesian = rankOneCoordinates &&
        positionalCoordinates.size() == fragment.getShape().size();
    if (cartesian) {
      Value coordinate = positionalCoordinates[fragmentAxis];
      auto type = cast<FragmentType>(coordinate.getType());
      auto occurrence = cast<AxisMapAttr>(type.getAxisMaps()[0]);
      if (sourceAxisIdentity(occurrence) == sourceAxisIdentity(expected) &&
          occurrence.getDimensionId() == expected.getDimensionId()) {
        collectAxisRanges(coordinate, 0, result, visited);
        return;
      }
    }
    using CoordinateOccurrence = std::pair<Value, unsigned>;
    enum class OccurrencePriority {
      SourceAndDimension,
      Dimension,
      UniqueSource,
    };
    auto selectOccurrence = [&](OccurrencePriority priority)
        -> SmallVector<CoordinateOccurrence, 2> {
      SmallVector<CoordinateOccurrence, 2> occurrences;
      for (Value coordinate : load.getCoordinates()) {
        auto coordinateType = dyn_cast<FragmentType>(coordinate.getType());
        if (!coordinateType)
          continue;
        SmallVector<unsigned, 2> axes;
        if (priority == OccurrencePriority::Dimension) {
          for (PhysicalDimensionProjection projection : queryFragmentDimensions(
                   coordinateType, expected.getDimensionId()))
            axes.push_back(projection.fragmentAxis);
        } else {
          SmallVector<PhysicalAxisProjection, 2> projections =
              queryFragmentAxes(coordinateType, sourceAxisIdentity(expected));
          if (priority == OccurrencePriority::SourceAndDimension)
            llvm::erase_if(projections,
                           [&](const PhysicalAxisProjection &projection) {
              return projection.dimensionId != expected.getDimensionId();
            });
          else if (projections.size() != 1)
            projections.clear();
          for (PhysicalAxisProjection projection : projections)
            axes.push_back(projection.fragmentAxis);
        }
        SmallVector<unsigned, 2> sameOccurrence;
        llvm::copy_if(axes, std::back_inserter(sameOccurrence),
                      [&](unsigned axis) { return axis == fragmentAxis; });
        if (sameOccurrence.size() == 1)
          axes = std::move(sameOccurrence);
        for (unsigned axis : axes)
          occurrences.emplace_back(coordinate, axis);
      }
      return occurrences;
    };
    SmallVector<CoordinateOccurrence, 2> occurrences =
        selectOccurrence(OccurrencePriority::SourceAndDimension);
    if (occurrences.empty())
      occurrences = selectOccurrence(OccurrencePriority::Dimension);
    if (occurrences.empty())
      occurrences = selectOccurrence(OccurrencePriority::UniqueSource);
    if (occurrences.empty()) {
      auto extent = cast<PhysicalExprAttr>(fragment.getShape()[fragmentAxis]);
      bool broadcastAxis =
          extent.getKind() == PhysicalExprKind::Constant &&
          extent.getValue() == 1 &&
          llvm::all_of(load.getCoordinates(), [&](Value coordinate) {
            auto type = dyn_cast<FragmentType>(coordinate.getType());
            if (!type)
              return true;
            BroadcastProjection projection = queryAxisProjection(type, fragment);
            return projection.isExact() &&
                   !projection.targetToSource[fragmentAxis];
          });
      if (broadcastAxis) {
        // Address coordinates do not vary along this introduced unit axis.
        // Validity or fill may still carry a traversal, so retain their roots
        // instead of mistaking an unexpanded range for a uniform value.
        for (Value dependency : {load.getValid(), load.getFill()}) {
          auto type = dependency ? dyn_cast<FragmentType>(dependency.getType())
                                 : FragmentType();
          if (!type)
            continue;
          auto projection = queryAxisProjection(type, fragment);
          if (!projection.isExact()) {
            result.state = PhysicalFactState::Unknown;
            appendUnique(result.blockers, operation);
            continue;
          }
          if (auto axis = projection.targetToSource[fragmentAxis])
            collectAxisRanges(dependency, *axis, result, visited);
        }
        return;
      }
      result.state = PhysicalFactState::Unknown;
      appendUnique(result.blockers, operation);
      return;
    }
    SmallVector<MakeRangeOp> accessRoots;
    for (const CoordinateOccurrence &occurrence : occurrences) {
      PhysicalRangeFact nested = axisRanges(occurrence.first, occurrence.second);
      if (nested.state == PhysicalFactState::Unknown ||
          !nested.blockers.empty()) {
        result.state = PhysicalFactState::Unknown;
        for (Operation *blocker : nested.blockers)
          appendUnique(result.blockers, blocker);
        continue;
      }
      for (MakeRangeOp range : nested.roots) {
        appendUnique(result.roots, range);
        appendUnique(accessRoots, range);
      }
      for (Operation *access : nested.accesses)
        appendUnique(result.accesses, access);
    }
    // Resolve this load's coordinate occurrence, not ranges accumulated from
    // sibling operands of a pointwise expression over different resources.
    if (result.state != PhysicalFactState::Unknown && !accessRoots.empty()) {
      MakeRangeOp authority = accessRoots.front();
      if (!llvm::all_of(accessRoots, [&](MakeRangeOp range) {
            return sameLogicalRange(authority, range);
          }) && (cartesian || !lockstepRanges(accessRoots).isExact())) {
        result.state = PhysicalFactState::Ambiguous;
        appendUnique(result.blockers, operation);
      }
    }
    return;
  }
  if (auto fold = dyn_cast<RegionFoldOp>(operation)) {
    auto opResult = dyn_cast<OpResult>(value);
    auto yield = dyn_cast<YieldOp>(fold.getSummarize().front().getTerminator());
    if (!opResult || !yield ||
        opResult.getResultNumber() >= fold.getIdentities().size() ||
        opResult.getResultNumber() >= yield.getValues().size()) {
      result.state = PhysicalFactState::Unknown;
      appendUnique(result.blockers, operation);
      return;
    }
    unsigned index = opResult.getResultNumber();
    collectAxisRanges(fold.getIdentities()[index],
                      fragmentAxis, result, visited);
    collectAxisRanges(yield.getValues()[index], fragmentAxis, result, visited);
    return;
  }
  if (auto reduce = dyn_cast<ReduceOp>(operation)) {
    auto opResult = dyn_cast<OpResult>(value);
    auto yield = dyn_cast<YieldOp>(reduce.getCombine().front().getTerminator());
    if (!opResult || !yield ||
        opResult.getResultNumber() >= yield.getValues().size()) {
      result.state = PhysicalFactState::Unknown;
      appendUnique(result.blockers, operation);
      return;
    }
    // Tuple results can depend on another component through the combine body
    // (for example an arg-reduce index selected by the value comparison).
    collectAxisRanges(yield.getValues()[opResult.getResultNumber()],
                      fragmentAxis, result, visited);
    return;
  }
  if (auto contract = dyn_cast<ContractOp>(operation)) {
    auto axes = queryContractionAxes(contract);
    if (!axes || fragmentAxis >= axes->results.size()) {
      result.state = PhysicalFactState::Unknown;
      appendUnique(result.blockers, operation);
      return;
    }
    auto source = axes->results[fragmentAxis];
    Value operand = source.operand == ContractionOperand::Lhs
                        ? Value(contract.getLhs()) : Value(contract.getRhs());
    collectAxisRanges(operand, source.axis, result, visited);
    if (source.operand == ContractionOperand::Lhs) {
      for (const auto &pair : axes->batch) {
        if (pair.lhs != source.axis) continue;
        // A batch result depends on both paired operands. The left operand
        // may broadcast along this axis and have no coordinate range at all.
        collectAxisRanges(contract.getRhs(), pair.rhs, result, visited);
        break;
      }
    }
    return;
  }
  if (isa<RegionBranchOpInterface>(operation)) {
    auto incoming = executableIncoming(value);
    if (!incoming.complete || incoming.edges.empty()) {
      result.state = PhysicalFactState::Unknown;
      appendUnique(result.blockers, operation);
      return;
    }
    for (const ControlFlowEdge &edge : incoming.edges)
      if (edge.operand)
        collectAxisRanges(edge.operand->get(), fragmentAxis, result, visited);
    return;
  }
  if (!isCoordinateReplayNode(operation)) {
    result.state = PhysicalFactState::Unknown;
    appendUnique(result.blockers, operation);
    return;
  }
  bool followed = false;
  for (Value operand : operation->getOperands()) {
    auto operandType = dyn_cast<FragmentType>(operand.getType());
    if (!operandType || operandType.getShape().size() != fragment.getShape().size() ||
        fragmentAxis >= operandType.getShape().size())
      continue;
    followed = true;
    collectAxisRanges(operand, fragmentAxis, result, visited);
  }
  if (!followed && !operation->getOperands().empty()) {
    result.state = PhysicalFactState::Unknown;
    appendUnique(result.blockers, operation);
  }
}

PhysicalRangeFact PhysicalProgramAnalysis::axisRanges(Value value,
                                                      unsigned fragmentAxis) {
  PhysicalRangeFact result;
  result.state = PhysicalFactState::Exact;
  llvm::DenseSet<std::pair<Value, unsigned>> visited;
  collectAxisRanges(value, fragmentAxis, result, visited);
  if (result.state == PhysicalFactState::Unknown || !result.blockers.empty())
    result.state = PhysicalFactState::Unknown;
  else if (result.roots.size() > 1) {
    MakeRangeOp authority = result.roots.front();
    if (!llvm::all_of(result.roots, [&](MakeRangeOp range) {
          return sameLogicalRange(authority, range);
        }))
      result.state = PhysicalFactState::Ambiguous;
  }
  result.unitStep = !result.roots.empty() &&
                    llvm::all_of(result.roots, isUnitStepRange);
  return result;
}

} // namespace intent::gpu
