#include "PhysicalProgramDetail.h"
#include "Intent/Dialect/GPU/Analysis/MemoryEffects.h"
#include "Intent/Dialect/GPU/Analysis/ResourceAlias.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"

#include <functional>

using namespace mlir;

namespace intent::gpu {
using namespace detail;

namespace {

void collectStructuredPrograms(Value value,
                               SmallPtrSetImpl<Operation *> &visited,
                               SmallVectorImpl<Operation *> &programs) {
  Operation *operation = value.getDefiningOp();
  if (!operation || !visited.insert(operation).second)
    return;
  if (isa<RegionFoldOp, RegionScanOp>(operation)) {
    if (!llvm::is_contained(programs, operation))
      programs.push_back(operation);
    return;
  }
  if (isAccessNode(operation) || operation->getNumRegions() != 0)
    return;
  for (Value operand : operation->getOperands())
    collectStructuredPrograms(operand, visited, programs);
}

bool reductionTypeConsumesSource(Type type, ArrayRef<int64_t> axes,
                                 PhysicalSourceAxis source,
                                 std::optional<int64_t> dimension) {
  if (auto record = dyn_cast<RecordType>(type))
    return llvm::any_of(record.getFieldTypes(), [&](Attribute field) {
      return reductionTypeConsumesSource(cast<TypeAttr>(field).getValue(), axes,
                                         source, dimension);
    });
  auto fragment = dyn_cast<FragmentType>(type);
  if (!fragment)
    return false;
  for (int64_t axis : axes) {
    if (axis < 0 || axis >= static_cast<int64_t>(fragment.getAxisMaps().size()))
      continue;
    auto mapping = cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
    if (PhysicalSourceAxis{mapping.getSourceId(), mapping.getSourceAxis(),
                           mapping.getDerived()} == source &&
        (!dimension || mapping.getDimensionId() == *dimension))
      return true;
  }
  return false;
}

bool typeCarriesTraversal(Type type, PhysicalSourceAxis source,
                          int64_t dimension) {
  if (auto fragment = dyn_cast<FragmentType>(type))
    return llvm::any_of(queryFragmentAxes(fragment, source),
                        [&](const PhysicalAxisProjection &projection) {
                          return projection.dimensionId == dimension;
                        });
  if (auto record = dyn_cast<RecordType>(type))
    return llvm::any_of(record.getFieldTypes(), [&](Attribute field) {
      return typeCarriesTraversal(cast<TypeAttr>(field).getValue(), source,
                                  dimension);
    });
  return false;
}

Value singleControlTarget(OpOperand &operand, ControlFlowEdgeKind kind,
                          Region *targetRegion) {
  auto outgoing = queryControlFlowOutgoing(operand);
  if (!outgoing.complete) return {};
  Value selected;
  for (const ControlFlowEdge &edge : outgoing.edges) {
    if (edge.kind != kind || edge.targetRegion != targetRegion) continue;
    if (selected && selected != edge.target) return {};
    selected = edge.target;
  }
  return selected;
}

bool preservesMemoryReadsAt(Operation *reads, Operation *intervening,
                            ResourceAliasAnalysis &aliases, Operation *context) {
  auto effects = getEffectsRecursively(intervening);
  if (!effects) return false;
  auto disjoint = [&](Value lhs, Value rhs) {
    // Nonoverlapping views can still share an allocation. A byte-span guard
    // therefore cannot authorize crossing a release of that allocation.
    if (llvm::any_of(*effects, [&](const auto &effect) {
          return isa<MemoryEffects::Free>(effect.getEffect()) &&
                 effect.getValue() == rhs;
        }))
      return false;
    return aliases.disjointAt(lhs, rhs, context);
  };
  return preservesMemoryReads(reads, intervening, aliases, disjoint);
}

} // namespace

bool canReplayReadAt(LoadOp load, Operation *insertionAnchor) {
  if (!load || !insertionAnchor)
    return false;
  ResourceAliasAnalysis aliases;
  auto preservesRead = [&](Operation *operation) {
    return preservesMemoryReadsAt(load, operation, aliases, insertionAnchor);
  };
  // This motion stays in the same invocation of the enclosing block, including
  // the same iteration when the block belongs to an ordered loop.
  if (insertionAnchor->getBlock() == load->getBlock() &&
      insertionAnchor->isBeforeInBlock(load)) {
    for (Operation *operation = insertionAnchor; operation != load;
         operation = operation->getNextNode())
      if (!preservesRead(operation))
        return false;
    return true;
  }
  // A branch only needs its executed prefix; entering a loop also exposes
  // this read to writes from earlier iterations.
  Operation *ancestor = insertionAnchor;
  while (ancestor && ancestor->getBlock() != load->getBlock()) {
    Operation *parent = ancestor->getParentOp();
    if (!isa_and_nonnull<scf::IfOp, scf::ForOp, scf::WhileOp>(parent))
      return false;
    if (!isa<scf::IfOp>(parent) && !preservesRead(parent))
      return false;
    for (Operation &preceding : *ancestor->getBlock()) {
      if (&preceding == ancestor)
        break;
      if (!preservesRead(&preceding))
        return false;
    }
    ancestor = parent;
  }
  if (!ancestor || ancestor == load ||
      !load->isBeforeInBlock(ancestor))
    return false;
  for (Operation *next = load->getNextNode(); next != ancestor;
       next = next->getNextNode())
    if (!preservesRead(next))
      return false;
  return true;
}

void PhysicalProgramAnalysis::analyzeReplay(
    Value value, std::optional<PhysicalSourceAxis> source,
    PhysicalReplayScope scope, bool allowAccesses,
    Operation *insertionAnchor, std::optional<int64_t> sourceDimension,
    DominanceInfo *dominance,
    const IRMapping *bindings,
    PhysicalReplayFact &result,
    ReplayVisits &visited) {
  if (!value)
    return;
  if (bindings && bindings->lookupOrNull(value))
    return;
  ReplayContext context{source, sourceDimension};
  auto ensureEnclosingReplay = [&](Operation *parent) {
    if (scope != PhysicalReplayScope::ValueGraph || !source || !sourceDimension)
      return false;
    Operation *root = nullptr;
    for (; isa_and_nonnull<scf::IfOp, scf::ForOp>(parent);
         parent = parent->getParentOp()) {
      auto found = visited.find(parent);
      if (found == visited.end() || found->second.empty())
        break;
      root = parent;
    }
    if (!root || !root->getNumResults())
      return false;
    if (!llvm::is_contained(visited.lookup(root), context))
      analyzeReplay(root->getResult(0), source, scope, allowAccesses,
                    insertionAnchor, sourceDimension, dominance, bindings, result, visited);
    return result.state != PhysicalFactState::Unknown &&
           llvm::is_contained(visited.lookup(root), context);
  };
  bool carriesRequestedTraversal = false;
  if (source) {
    if (sourceDimension) {
      carriesRequestedTraversal = llvm::any_of(
          queryFragmentAxes(value.getType(), *source),
          [&](const PhysicalAxisProjection &projection) {
            return projection.dimensionId == *sourceDimension;
          });
    } else {
      carriesRequestedTraversal = carriesSource(value.getType(), *source);
    }
  }
  Operation *definition = value.getDefiningOp();
  bool recheckRegion = isa_and_nonnull<scf::IfOp, scf::ForOp>(definition) &&
      scope == PhysicalReplayScope::ValueGraph && source && sourceDimension &&
      visited.count(definition) && !visited.lookup(definition).empty() &&
      !llvm::is_contained(visited.lookup(definition), context);
  bool mayReuse = source || !isa<FragmentType, RecordType>(value.getType());
  if (insertionAnchor && dominance && mayReuse && !recheckRegion &&
      dominance->dominates(value, insertionAnchor) &&
      !carriesRequestedTraversal) {
    SmallPtrSet<Operation *, 16> dependencyVisited;
    collectStructuredPrograms(value, dependencyVisited,
                              result.structuredPrograms);
    result.crossesStructuredProgram |= !result.structuredPrograms.empty();
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
      analyzeReplay(record.getFields()[field], source, scope, allowAccesses,
                    insertionAnchor, sourceDimension, dominance, bindings, result,
                    visited);
      return;
    }
  }
  bool physicalValue = isa<FragmentType, RecordType>(value.getType());
  if (!physicalValue && !insertionAnchor)
    return;
  if (source && physicalValue && !carriesSource(value.getType(), *source) &&
      !insertionAnchor)
    return;
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    Operation *owner = argument.getOwner()->getParentOp();
    if (isa_and_nonnull<RegionBranchOpInterface>(owner) && !isa<scf::ForOp>(owner)) {
      // Structural forwarding alone does not authorize replaying new control
      // forms. Whole-region replay below deliberately supports only if/for.
      appendUnique(result.blockers, owner);
      result.state = PhysicalFactState::Unknown;
      return;
    }
    if (auto loop = dyn_cast_or_null<scf::ForOp>(owner)) {
      bool replayingLoop = ensureEnclosingReplay(loop);
      // Replaying the complete loop checks its initial operands and all
      // yields. Its block arguments are local bindings, including unchanged
      // carries whose backedge would otherwise recurse into themselves.
      if (replayingLoop)
        return;
      if (!replayingLoop && argument != loop.getInductionVar()) {
        appendUnique(result.blockers, loop);
        result.state = PhysicalFactState::Unknown;
        return;
      }
    }
    SmallVector<Value, 2> outer = structuredSourcesForArgument(argument);
    if (outer.empty()) {
      result.state = PhysicalFactState::Unknown;
      return;
    }
    for (Value related : outer)
      analyzeReplay(related, source, scope, allowAccesses, insertionAnchor,
                    sourceDimension, dominance, bindings, result, visited);
    return;
  }
  Operation *operation = value.getDefiningOp();
  if (!operation)
    return;
  auto &contexts = visited[operation];
  if (llvm::is_contained(contexts, context))
    return;
  contexts.push_back(context);
  if (isa<ContractOp, ScaledContractOp, SparseContractOp>(operation))
    appendUnique(result.contractions, operation);
  if (auto range = dyn_cast<MakeRangeOp>(operation)) {
    if (insertionAnchor && source && sourceDimension &&
        sourceAxisIdentity(range) == *source) {
      FailureOr<int64_t> dimension = queryRangeDimension(range);
      if (failed(dimension)) {
        appendUnique(result.blockers, operation);
        result.state = PhysicalFactState::Unknown;
      }
    }
    return;
  }
  if (isa<scf::IfOp, scf::ForOp>(operation) &&
      scope == PhysicalReplayScope::ValueGraph && source && sourceDimension) {
    if (isa<scf::ForOp>(operation)) {
      auto dependency = reductionDependency(value, *source, *sourceDimension);
      if (!dependency.isExact() || dependency.depends) {
        appendUnique(result.blockers, operation);
        result.state = PhysicalFactState::Unknown;
        return;
      }
    }
    SmallVector<Value> controls;
    if (auto branch = dyn_cast<scf::IfOp>(operation))
      controls.push_back(branch.getCondition());
    else {
      auto loop = cast<scf::ForOp>(operation);
      controls = {loop.getLowerBound(), loop.getUpperBound(), loop.getStep()};
    }
    llvm::DenseSet<Value> controlValues;
    bool dependentControl = false;
    for (unsigned index = 0; index < controls.size(); ++index) {
      Value control = controls[index];
      if (!controlValues.insert(control).second)
        continue;
      dependentControl |= typeCarriesTraversal(control.getType(), *source, *sourceDimension);
      if (Operation *producer = control.getDefiningOp())
        controls.append(producer->getOperands().begin(), producer->getOperands().end());
    }
    bool preservesReads = hasOnlyReadEffects(operation);
    if (insertionAnchor && preservesReads) {
      // Nested control is cloned as part of an already visited replay region.
      // Check motion from that enclosing region, rather than trying to walk
      // an outer insertion point into the nested operation's original block.
      Operation *replayRoot = operation;
      for (Operation *parent = operation->getParentOp();
           isa_and_nonnull<scf::IfOp, scf::ForOp>(parent);
           parent = parent->getParentOp()) {
        auto found = visited.find(parent);
        if (found == visited.end() ||
            !llvm::is_contained(found->second, context))
          break;
        replayRoot = parent;
      }
      ResourceAliasAnalysis aliases;
      Operation *anchor = insertionAnchor;
      while (anchor && anchor->getBlock() != replayRoot->getBlock()) {
        preservesReads &= preservesMemoryReadsAt(replayRoot, anchor, aliases,
                                                insertionAnchor);
        anchor = anchor->getParentOp();
      }
      preservesReads &= anchor &&
          (anchor == replayRoot || replayRoot->isBeforeInBlock(anchor));
      if (preservesReads && anchor != replayRoot)
        for (Operation *next = replayRoot->getNextNode(); next != anchor;
             next = next->getNextNode())
          preservesReads &= preservesMemoryReadsAt(replayRoot, next, aliases,
                                                  insertionAnchor);
    }
    if (!preservesReads || dependentControl) {
      appendUnique(result.blockers, operation);
      result.state = PhysicalFactState::Unknown;
      return;
    }
    for (Value operand : operation->getOperands())
      analyzeReplay(operand, source, scope, allowAccesses, insertionAnchor,
                    sourceDimension, dominance, bindings, result, visited);
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (Value yielded : block.getTerminator()->getOperands())
          analyzeReplay(yielded, source, scope, allowAccesses, insertionAnchor,
                        sourceDimension, dominance, bindings, result, visited);
    return;
  }
  if (!physicalValue) {
    if (isa<arith::ConstantOp>(operation))
      return;
    if (!isPhysicalReplayNode(operation, scope, allowAccesses) ||
        operation->getNumRegions() != 0 || operation->getNumResults() != 1) {
      appendUnique(result.blockers, operation);
      result.state = PhysicalFactState::Unknown;
      return;
    }
    if (isAccessNode(operation)) {
      result.crossesAccess = true;
      appendUnique(result.accesses, operation);
      if (auto load = dyn_cast<LoadOp>(operation);
          load && insertionAnchor && !canReplayReadAt(load, insertionAnchor)) {
        appendUnique(result.blockers, operation);
        result.state = PhysicalFactState::Unknown;
        return;
      }
    }
    for (Value operand : operation->getOperands())
      analyzeReplay(operand, source, scope, allowAccesses, insertionAnchor,
                    sourceDimension, dominance, bindings, result, visited);
    return;
  }
  if (isAccessNode(operation)) {
    result.crossesAccess = true;
    appendUnique(result.accesses, operation);
    if (!allowAccesses) {
      appendUnique(result.blockers, operation);
      result.state = PhysicalFactState::Unknown;
      return;
    }
    if (auto load = dyn_cast<LoadOp>(operation); load && insertionAnchor) {
      bool replayedRegion = ensureEnclosingReplay(load->getParentOp());
      // Enclosing replay regions already prove read-only motion. Standalone
      // reads must also preserve their value across intervening writes.
      if (!replayedRegion && !canReplayReadAt(load, insertionAnchor)) {
        appendUnique(result.blockers, operation);
        result.state = PhysicalFactState::Unknown;
        return;
      }
    }
  } else if (auto fold = dyn_cast<RegionFoldOp>(operation)) {
    result.crossesStructuredProgram = true;
    appendUnique(result.structuredPrograms, operation);
    if (!source || !sourceDimension) {
      appendUnique(result.blockers, operation);
      result.state = PhysicalFactState::Unknown;
      return;
    }
    for (Value segmentSource :
         fold.getSources()) {
      auto fragment = dyn_cast<FragmentType>(segmentSource.getType());
      if (!fragment || fold.getAxis() >= fragment.getAxisMaps().size()) {
        appendUnique(result.blockers, operation);
        result.state = PhysicalFactState::Unknown;
        return;
      }
      auto mapping =
          cast<AxisMapAttr>(fragment.getAxisMaps()[fold.getAxis()]);
      if (sourceAxisIdentity(mapping) == *source &&
          mapping.getDimensionId() == *sourceDimension) {
        appendUnique(result.blockers, operation);
        result.state = PhysicalFactState::Unknown;
        return;
      }
    }
    for (Value operand : fold.getOperands())
      if (typeCarriesTraversal(operand.getType(), *source, *sourceDimension))
        analyzeReplay(operand, source, scope, allowAccesses, insertionAnchor,
                      sourceDimension, dominance, bindings, result, visited);
    return;
  } else if (auto scan = dyn_cast<RegionScanOp>(operation)) {
    result.crossesStructuredProgram = true;
    appendUnique(result.structuredPrograms, operation);
    if (!source || !sourceDimension) {
      appendUnique(result.blockers, operation);
      result.state = PhysicalFactState::Unknown;
      return;
    }
    for (Value segmentSource :
         scan.getSources()) {
      auto fragment = dyn_cast<FragmentType>(segmentSource.getType());
      if (!fragment || scan.getAxis() >= fragment.getAxisMaps().size()) {
        appendUnique(result.blockers, operation);
        result.state = PhysicalFactState::Unknown;
        return;
      }
      auto mapping =
          cast<AxisMapAttr>(fragment.getAxisMaps()[scan.getAxis()]);
      if (sourceAxisIdentity(mapping) == *source &&
          mapping.getDimensionId() == *sourceDimension) {
        appendUnique(result.blockers, operation);
        result.state = PhysicalFactState::Unknown;
        return;
      }
    }
    for (Value operand : scan.getOperands())
      if (typeCarriesTraversal(operand.getType(), *source, *sourceDimension))
        analyzeReplay(operand, source, scope, allowAccesses, insertionAnchor,
                      sourceDimension, dominance, bindings, result, visited);
    return;
  } else if (!isPhysicalReplayNode(operation, scope, allowAccesses)) {
    appendUnique(result.blockers, operation);
    result.state = PhysicalFactState::Unknown;
    return;
  }
  Region *combine = nullptr;
  if (auto reduce = dyn_cast<ReduceOp>(operation))
    combine = &reduce.getCombine();
  else if (auto scan = dyn_cast<ScanOp>(operation)) {
    // A prefix depends on earlier members of this axis. Replaying each tile
    // independently would reset that state; only the other axes are pointwise.
    SmallVector<Type> schemas{value.getType()};
    while (source && !schemas.empty()) {
      Type schema = schemas.pop_back_val();
      if (auto record = dyn_cast<RecordType>(schema)) {
        for (Attribute field : record.getFieldTypes())
          schemas.push_back(cast<TypeAttr>(field).getValue());
        continue;
      }
      auto fragment = cast<FragmentType>(schema);
      auto axis = cast<AxisMapAttr>(fragment.getAxisMaps()[scan.getAxis()]);
      if (sourceAxisIdentity(axis) == *source &&
          (!sourceDimension || axis.getDimensionId() == *sourceDimension)) {
        appendUnique(result.blockers, operation);
        result.state = PhysicalFactState::Unknown;
        return;
      }
    }
    combine = &scan.getCombine();
  }
  if (combine) {
    WalkResult helper = combine->walk([&](Operation *nested) {
      if (isa<YieldOp, arith::ConstantOp>(nested))
        return WalkResult::advance();
      if (!isPhysicalReplayNode(nested, PhysicalReplayScope::Coordinate,
                                /*allowAccesses=*/false)) {
        appendUnique(result.blockers, nested);
        result.state = PhysicalFactState::Unknown;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (helper.wasInterrupted())
      return;
  }
  if (operation->getNumRegions() != 0 && !isa<ReduceOp, ScanOp>(operation)) {
    appendUnique(result.blockers, operation);
    result.state = PhysicalFactState::Unknown;
    return;
  }
  if (auto broadcast = dyn_cast<BroadcastOp>(operation); broadcast && source) {
    auto input = dyn_cast<FragmentType>(broadcast.getValue().getType());
    auto output = dyn_cast<FragmentType>(value.getType());
    PhysicalAxisProjection requested = queryFragmentAxis(output, *source);
    if (input && requested.isExact()) {
      auto relations = queryFragmentOperandRelations(operation);
      if (succeeded(relations)) {
        const FragmentAxisGroup *group =
            relations->front().groupForResultAxis(requested.fragmentAxis);
        if (group && group->sourceAxes.size() == 1) {
          auto mapping = cast<AxisMapAttr>(
              input.getAxisMaps()[group->sourceAxes.front()]);
          if (sourceDimension && mapping.getDimensionId() <= 0) {
            appendUnique(result.blockers, operation);
            result.state = PhysicalFactState::Unknown;
            return;
          }
          std::optional<int64_t> inputDimension =
              sourceDimension ? std::optional<int64_t>(mapping.getDimensionId())
                              : std::nullopt;
          analyzeReplay(broadcast.getValue(), sourceAxisIdentity(mapping), scope,
                        allowAccesses, insertionAnchor, inputDimension,
                        dominance, bindings, result, visited);
          return;
        }
      }
    }
  }
  for (Value operand : operation->getOperands())
    analyzeReplay(operand, source, scope, allowAccesses, insertionAnchor,
                  sourceDimension, dominance, bindings, result, visited);
}

PhysicalReplayFact PhysicalProgramAnalysis::replayability(
    Value value, std::optional<PhysicalSourceAxis> source,
    PhysicalReplayScope scope, bool allowAccesses,
    std::optional<int64_t> sourceDimension) {
  PhysicalReplayFact result;
  result.state = PhysicalFactState::Exact;
  ReplayVisits visited;
  analyzeReplay(value, source, scope, allowAccesses, nullptr,
                sourceDimension, nullptr, nullptr, result,
                visited);
  return result;
}

PhysicalReplayFact PhysicalProgramAnalysis::replayAt(
    Value value, std::optional<PhysicalSourceAxis> source,
    PhysicalReplayScope scope, bool allowAccesses, Operation *insertionAnchor,
    const IRMapping &bindings, std::optional<int64_t> sourceDimension) {
  PhysicalReplayFact result;
  if (!insertionAnchor ||
      insertionAnchor->getParentOfType<func::FuncOp>() != kernel)
    return result;
  result.state = PhysicalFactState::Exact;
  ReplayVisits visited;
  DominanceInfo dominance(kernel);
  analyzeReplay(value, source, scope, allowAccesses, insertionAnchor,
                sourceDimension, &dominance, &bindings, result, visited);
  return result;
}

PhysicalReductionDependencyFact PhysicalProgramAnalysis::reductionDependency(
    Value value, PhysicalSourceAxis source,
    std::optional<int64_t> sourceDimension) {
  using Traversal = std::pair<PhysicalSourceAxis, std::optional<int64_t>>;
  llvm::DenseMap<Value, SmallVector<Traversal, 2>> visited;
  std::function<PhysicalReductionDependencyFact(Value, PhysicalSourceAxis,
                                               std::optional<int64_t>)> analyze =
      [&](Value current, PhysicalSourceAxis source,
          std::optional<int64_t> sourceDimension) -> PhysicalReductionDependencyFact {
    PhysicalReductionDependencyFact exact;
    exact.state = PhysicalFactState::Exact;
    Operation *operation = current.getDefiningOp();
    auto &contexts = visited[current];
    Traversal context{source, sourceDimension};
    if (!operation || llvm::is_contained(contexts, context))
      return exact;
    contexts.push_back(context);
    if (auto range = dyn_cast<MakeRangeOp>(operation)) {
      // Reaching a coordinate proves an ordinary value dependency, not a
      // reduction dependency.  Only an operation that removes or carries this
      // axis may turn the fact into `depends=true` below.  Treating every range
      // leaf as a reduction made pointwise ownership axes both program-mapped
      // and internally traversed, so different programs replayed overlapping
      // writeback domains.
      return exact;
    }
    if (auto reduce = dyn_cast<ReduceOp>(operation)) {
      for (Value input :
           reduce.getSources()) {
        if (reductionTypeConsumesSource(input.getType(), reduce.getAxes(),
                                        source, sourceDimension)) {
          exact.depends = true;
          return exact;
        }
        if (!sourceDimension)
          continue;
        auto fragment = dyn_cast<FragmentType>(input.getType());
        if (!fragment)
          continue;
        for (int64_t axis : reduce.getAxes()) {
          if (axis < 0 ||
              axis >= static_cast<int64_t>(fragment.getAxisMaps().size()))
            continue;
          auto mapping = cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
          PhysicalRangeFact ranges = sourceRanges(
              input, sourceAxisIdentity(mapping));
          if (llvm::any_of(ranges.roots, [&](MakeRangeOp range) {
                FailureOr<int64_t> dimension = queryRangeDimension(range);
                return succeeded(dimension) &&
                       *dimension == *sourceDimension;
              })) {
            exact.depends = true;
            return exact;
          }
        }
      }
      return exact;
    }
    if (auto contract = dyn_cast<ContractOp>(operation)) {
      // Preserve the reduction dependency before vector/matrix realization
      // chooses the native operation for this contraction.
      exact.depends = reductionTypeConsumesSource(
                          contract.getLhs().getType(),
                          contract.getLhsReductionAxes(), source, sourceDimension) ||
                      reductionTypeConsumesSource(
                          contract.getRhs().getType(),
                          contract.getRhsReductionAxes(), source, sourceDimension);
      if (exact.depends)
        return exact;
    }
    if (auto scan = dyn_cast<ScanOp>(operation)) {
      for (Value input : scan.getSources())
        if (reductionTypeConsumesSource(
                input.getType(),
                ArrayRef<int64_t>{static_cast<int64_t>(scan.getAxis())},
                source, sourceDimension)) {
          exact.depends = true;
          return exact;
        }
      return exact;
    }
    if (auto fold = dyn_cast<RegionFoldOp>(operation)) {
      auto result = dyn_cast<OpResult>(current);
      if (sourceDimension && result && result.getOwner() == fold &&
          typeCarriesTraversal(current.getType(), source, *sourceDimension)) {
        exact.depends = true;
        exact.throughStructuredReduction = true;
      }
      return exact;
    }
    if (auto loop = dyn_cast<scf::ForOp>(operation)) {
      auto traversals = loop->getAttrOfType<ArrayAttr>(reductionSourcesAttr);
      if (traversals && llvm::any_of(traversals, [&](Attribute attribute) {
            auto traversal = dyn_cast<PhysicalSourceAttr>(attribute);
            return traversal &&
                   PhysicalSourceAxis{traversal.getSourceId(),
                                      traversal.getSourceAxis(),
                                      traversal.getDerived()} == source;
          })) {
        exact.depends = true;
        return exact;
      }
      OpOperand *initial = singleControlInput(current, ControlFlowEdgeKind::Bypass, nullptr);
      OpOperand *yielded = singleControlInput(current, ControlFlowEdgeKind::Exit, &loop.getRegion());
      Value carry = initial ? singleControlTarget(*initial, ControlFlowEdgeKind::Entry,
                                                 &loop.getRegion()) : Value();
      if (!initial || !yielded || !carry ||
          singleControlTarget(*yielded, ControlFlowEdgeKind::RegionTransfer,
                              &loop.getRegion()) != carry) {
        exact.state = PhysicalFactState::Unknown;
        appendUnique(exact.blockers, operation);
        return exact;
      }
      auto carryFeedsReduction = [&](unsigned carryAxis) {
        SmallVector<std::pair<Value, unsigned>> pending{
            {carry, carryAxis}};
        llvm::DenseMap<Value, SmallVector<unsigned, 2>> reached;
        for (unsigned cursor = 0; cursor < pending.size(); ++cursor) {
          auto [carried, axis] = pending[cursor];
          auto &axes = reached[carried];
          if (llvm::is_contained(axes, axis))
            continue;
          axes.push_back(axis);
          auto input = cast<FragmentType>(carried.getType());
          for (OpOperand &use : carried.getUses()) {
            Operation *user = use.getOwner();
            if (!loop->isProperAncestor(user))
              continue;
            if (auto reduce = dyn_cast<ReduceOp>(user)) {
              if (use.getOperandNumber() < reduce.getSources().size() &&
                  llvm::is_contained(reduce.getAxes(), static_cast<int64_t>(axis)))
                return true;
              continue;
            }
            if (user->getNumResults() != 1)
              continue;
            auto output = dyn_cast<FragmentType>(user->getResult(0).getType());
            if (!output)
              continue;
            if (isa<ReshapeOp>(user)) {
              auto relations = queryFragmentOperandRelations(user);
              bool positional = input.getShape() == output.getShape() &&
                  succeeded(relations) && llvm::all_of(
                      relations->front().groups, [](const FragmentAxisGroup &group) {
                        return group.sourceAxes == group.resultAxes;
                      });
              if (positional)
                pending.emplace_back(user->getResult(0), axis);
            } else if (isa<TransposeOp>(user)) {
              auto relations = queryFragmentOperandRelations(user);
              if (succeeded(relations))
                for (const FragmentAxisGroup &group : relations->front().groups)
                  if (group.sourceAxes.front() == axis)
                    pending.emplace_back(user->getResult(0), group.resultAxes.front());
            } else if (isa<BroadcastOp, UnaryOp, BinaryOp, CompareOp, SelectOp,
                           CastOp, BitcastOp>(user)) {
              auto projection = queryBroadcastProjection(input, output);
              if (projection.isExact())
                for (auto [targetAxis, sourceAxis] :
                     llvm::enumerate(projection.targetToSource))
                  if (sourceAxis && *sourceAxis == axis)
                    pending.emplace_back(user->getResult(0), targetAxis);
            }
          }
        }
        return false;
      };
      SmallVector<Traversal, 4> loopTraversals{context};
      for (const auto &projection : queryFragmentAxes(current.getType(), source)) {
        if (sourceDimension && projection.dimensionId != *sourceDimension)
          continue;
        // A loop carry can be rebound to another operand's positional axes
        // before it is reduced. Its output coordinate roots alone do not
        // expose that recurrence; follow the carry's exact pointwise uses.
        if (carryFeedsReduction(projection.fragmentAxis)) {
          exact.depends = true;
          return exact;
        }
        auto ranges = axisRanges(current, projection.fragmentAxis);
        if (!ranges.isExact() || !ranges.blockers.empty()) {
          exact.state = PhysicalFactState::Unknown;
          appendUnique(exact.blockers, operation);
        } else
          for (MakeRangeOp range : ranges.roots) {
            auto dimension = queryRangeDimension(range);
            if (failed(dimension))
              continue;
            Traversal related{sourceAxisIdentity(range), *dimension};
            if (!llvm::is_contained(loopTraversals, related))
              loopTraversals.push_back(related);
          }
      }
      for (Value related : {initial->get(), yielded->get()}) {
        for (auto [identity, dimension] : loopTraversals) {
          PhysicalReductionDependencyFact nested = analyze(related, identity, dimension);
          if (nested.depends)
            return nested;
          if (!nested.isExact()) {
            exact.state = PhysicalFactState::Unknown;
            llvm::append_range(exact.blockers, nested.blockers);
          }
        }
      }
      return exact;
    }
    if (operation->getNumRegions() != 0) {
      exact.state = PhysicalFactState::Unknown;
      appendUnique(exact.blockers, operation);
      return exact;
    }
    auto relations = queryFragmentOperandRelations(operation);
    for (OpOperand &operandSlot : operation->getOpOperands()) {
      Value operand = operandSlot.get();
      SmallVector<Traversal, 4> traversals{context};
      auto input = dyn_cast<FragmentType>(operand.getType());
      auto output = dyn_cast<FragmentType>(current.getType());
      auto requested = output ? queryFragmentAxis(output, source)
                              : PhysicalAxisProjection{};
      if (input && requested.isExact() &&
          (!sourceDimension || requested.dimensionId == *sourceDimension)) {
        SmallVector<unsigned> inputAxes;
        if (succeeded(relations))
          for (const FragmentOperandRelation &relation : *relations)
            if (relation.operandNumber == operandSlot.getOperandNumber())
              if (const FragmentAxisGroup *group =
                      relation.groupForResultAxis(requested.fragmentAxis))
                llvm::append_range(inputAxes, group->sourceAxes);
        for (unsigned axis : inputAxes) {
          auto mapping = cast<AxisMapAttr>(input.getAxisMaps()[axis]);
          Traversal projected{sourceAxisIdentity(mapping),
                              mapping.getDimensionId()};
          if (!llvm::is_contained(traversals, projected))
            traversals.push_back(projected);
        }
      }
      for (auto [selected, dimension] : traversals) {
        PhysicalReductionDependencyFact nested =
            analyze(operand, selected, dimension);
        if (nested.depends)
          return nested;
        if (!nested.isExact()) {
          exact.state = PhysicalFactState::Unknown;
          llvm::append_range(exact.blockers, nested.blockers);
        }
      }
    }
    return exact;
  };
  return analyze(value, source, sourceDimension);
}

} // namespace intent::gpu
