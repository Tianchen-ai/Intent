#include "PhysicalProgramDetail.h"
#include "ScalarExpressions.h"
#include "Intent/Dialect/GPU/IR/PhysicalExpressions.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLExtras.h"


using namespace mlir;

namespace intent::gpu {
using namespace detail;

namespace {

bool isProgramCoordinateRange(MakeRangeOp range) {
  auto coordinate = range.getStart().getDefiningOp<WorksetCoordinateOp>();
  return coordinate && range->hasAttr(worksetCoordinateRangeAttr) &&
         coordinate.getSourceId() == range.getSourceId() &&
         coordinate.getSourceAxis() == range.getSourceAxis() &&
         !range.getDerived() &&
         sameScalarExpression(coordinate.getStep(), range.getStep());
}

} // namespace

std::optional<ContractionAxes>
queryContractionAxes(Operation *operation, std::string *failureReason) {
  auto read = [&](auto contract, Value lhs, Value rhs)
      -> std::optional<ContractionAxes> {
    auto left = dyn_cast<FragmentType>(lhs.getType());
    auto right = dyn_cast<FragmentType>(rhs.getType());
    auto result = dyn_cast<FragmentType>(contract.getResult().getType());
    if (!left || !right || !result) {
      if (failureReason) *failureReason = "requires fragment operands and result";
      return std::nullopt;
    }
    auto axes = ContractionAxes::get(
        left.getShape().size(), right.getShape().size(),
        contract.getLhsReductionAxes(), contract.getRhsReductionAxes(),
        contract.getLhsBatchAxes(), contract.getRhsBatchAxes(), failureReason);
    if (axes && axes->results.size() != result.getShape().size()) {
      if (failureReason) *failureReason = "result rank disagrees with batch/free axes";
      return std::nullopt;
    }
    return axes;
  };
  if (auto contract = dyn_cast_or_null<ContractOp>(operation))
    return read(contract, contract.getLhs(), contract.getRhs());
  if (auto contract = dyn_cast_or_null<ScaledContractOp>(operation))
    return read(contract, contract.getLhs(), contract.getRhs());
  if (auto contract = dyn_cast_or_null<SparseContractOp>(operation))
    return read(contract, contract.getCompressed(), contract.getRhs());
  if (failureReason) *failureReason = "expected a contraction operation";
  return std::nullopt;
}

PhysicalAxisRealizationFact
PhysicalProgramAnalysis::axisRealization(Value value, unsigned fragmentAxis) {
  PhysicalAxisRealizationFact result;
  result.fragmentAxis = fragmentAxis;
  auto fragment = dyn_cast<FragmentType>(value.getType());
  if (!fragment || fragmentAxis >= fragment.getShape().size() ||
      fragmentAxis >= fragment.getAxisMaps().size())
    return result;

  auto mapping = dyn_cast<AxisMapAttr>(fragment.getAxisMaps()[fragmentAxis]);
  auto extent = dyn_cast<PhysicalExprAttr>(fragment.getShape()[fragmentAxis]);
  if (!mapping || !extent)
    return result;
  result.source = sourceAxisIdentity(mapping);
  result.dimensionId = mapping.getDimensionId();

  if (auto argument = dyn_cast<BlockArgument>(value)) {
    Operation *owner = argument.getOwner()->getParentOp();
    for (const ControlFlowEdge &edge : queryControlFlowIncoming(argument).edges) {
      if (edge.kind != ControlFlowEdgeKind::Entry || !edge.operand)
        continue;
      Value initial = edge.operand->get();
      if (initial.getType() == fragment) {
        PhysicalAxisRealizationFact input =
            axisRealization(initial, fragmentAxis);
        if (input.hasExtentAuthority()) {
          result.state = PhysicalFactState::Exact;
          result.physicalized = input.physicalized;
          result.constructionScalarSeed = input.constructionScalarSeed;
          result.roots = input.roots;
          result.extentAuthority =
              PhysicalAxisRealizationFact::ExtentAuthority::Structural;
          return result;
        }
      }
    }
    auto isSegmentSource = [&](uint64_t sourceCount, uint64_t axis,
                               Region &region) {
      return argument.getOwner()->getParent() == &region &&
             argument.getArgNumber() < sourceCount && fragmentAxis == axis;
    };
    bool segmentSource = false;
    if (auto fold = dyn_cast_or_null<RegionFoldOp>(owner))
      segmentSource = isSegmentSource(fold.getSources().size(), fold.getAxis(),
                                      fold.getSummarize());
    else if (auto scan = dyn_cast_or_null<RegionScanOp>(owner))
      segmentSource =
          isSegmentSource(scan.getSources().size(), scan.getAxis(),
                          scan.getSummarize()) ||
          isSegmentSource(scan.getSources().size(), scan.getAxis(),
                          scan.getEmit());
    if (segmentSource) {
      // RegionFoldOp/RegionScanOp verification binds this exact block argument
      // axis to the operation's segment parameter.  The slice extent is a
      // first-class physical relation, not something to rediscover from the
      // unsliced outer source range.
      result.state = PhysicalFactState::Exact;
      result.extentAuthority =
          PhysicalAxisRealizationFact::ExtentAuthority::Structural;
      return result;
    }
  }

  if (auto broadcast = value.getDefiningOp<BroadcastOp>()) {
    auto source = dyn_cast<FragmentType>(broadcast.getValue().getType());
    if (source) {
      auto relations = queryFragmentOperandRelations(broadcast.getOperation());
      if (succeeded(relations))
        if (std::optional<unsigned> sourceAxis =
                relations->front().correspondingSourceAxis(fragmentAxis);
            sourceAxis && source.getShape()[*sourceAxis] == extent) {
          PhysicalAxisRealizationFact input =
              axisRealization(broadcast.getValue(), *sourceAxis);
          if (input.hasExtentAuthority()) {
            result.state = PhysicalFactState::Exact;
            result.physicalized = input.physicalized;
            result.constructionScalarSeed = input.constructionScalarSeed;
            result.roots = input.roots;
            result.extentAuthority =
                PhysicalAxisRealizationFact::ExtentAuthority::Structural;
            return result;
          }
        }
    }
  }

  if (auto splat = value.getDefiningOp<SplatOp>()) {
    auto size = constantPhysicalExpression(extent);
    if (size) {
      DominanceInfo dominance(kernel);
      SmallVector<MakeRangeOp> visible;
      for (MakeRangeOp range : programRanges(result.source).roots) {
        auto dimension = queryRangeDimension(range);
        if (succeeded(dimension) && *dimension == result.dimensionId &&
            dominance.dominates(range.getOperation(), splat.getOperation()) &&
            valueMatchesExtent(range.getExtent(), extent) &&
            constantLogicalRangeCardinality(range) == size &&
            samePhysicalScalarExpression(range.getStart(),
                                         range.getLogicalStart()))
          visible.push_back(range);
      }
      if (lockstepRanges(visible).isExact()) {
        result.state = PhysicalFactState::Exact;
        result.physicalized = true;
        result.roots = std::move(visible);
        result.extentAuthority =
            PhysicalAxisRealizationFact::ExtentAuthority::Range;
        return result;
      }
      if (*size > 1) {
        // Splat defines every lane of its declared physical fragment. A
        // non-unit static axis therefore has its own extent authority even
        // without a coordinate-producing operand. This is physical capacity,
        // not a proof about an independent logical range's active members.
        // Keep a unit axis distinguishable from a construction scalar seed.
        result.state = PhysicalFactState::Exact;
        result.physicalized = true;
        result.extentAuthority =
            PhysicalAxisRealizationFact::ExtentAuthority::Structural;
        return result;
      }
    }
  }

  if (auto join = value.getDefiningOp<JoinOp>()) {
    if (fragmentAxis == join.getAxis()) {
      result.state = PhysicalFactState::Exact;
      result.physicalized = true;
      result.extentAuthority =
          PhysicalAxisRealizationFact::ExtentAuthority::Structural;
      return result;
    }
    auto lhs = axisRealization(join.getLhs(), fragmentAxis);
    auto rhs = axisRealization(join.getRhs(), fragmentAxis);
    if (lhs.isExact() && lhs.physicalized && !lhs.constructionScalarSeed &&
        rhs.isExact() && rhs.physicalized && !rhs.constructionScalarSeed) {
      result.state = PhysicalFactState::Exact;
      result.physicalized = true;
      auto ranges = axisRanges(value, fragmentAxis);
      if (ranges.isExact())
        result.roots = std::move(ranges.roots);
      result.extentAuthority =
          PhysicalAxisRealizationFact::ExtentAuthority::Structural;
      return result;
    }
  }

  if (auto reshape = value.getDefiningOp<ReshapeOp>()) {
    auto relations = queryFragmentOperandRelations(reshape.getOperation());
    if (failed(relations)) return result;
    if (const FragmentAxisGroup *group = relations->front().groupForResultAxis(fragmentAxis)) {
      if (group->kind == FragmentAxisRelationKind::Corresponding)
        return axisRealization(reshape.getValue(), group->sourceAxes.front());
      auto source = cast<FragmentType>(reshape.getValue().getType());
      if (group->kind == FragmentAxisRelationKind::Reassociation &&
          group->sourceAxes.size() == 1 && group->resultAxes.size() == 1 &&
          source.getShape()[group->sourceAxes.front()] == extent) {
        // Changing an axis occurrence through a one-to-one reassociation does
        // not discard a boundary's extent authority. It also does not realize
        // that boundary's traversal before its incoming slots are closed.
        auto input = axisRealization(reshape.getValue(), group->sourceAxes.front());
        if (input.hasExtentAuthority()) {
          result.state = input.state;
          result.physicalized = input.physicalized;
          result.constructionScalarSeed = input.constructionScalarSeed;
          result.roots = std::move(input.roots);
          result.extentAuthority =
              PhysicalAxisRealizationFact::ExtentAuthority::Structural;
          return result;
        }
      }
      bool physicalized = llvm::all_of(
          group->sourceAxes, [&](unsigned axis) {
            auto input = axisRealization(reshape.getValue(), axis);
            return input.isExact() && input.physicalized &&
                   !input.constructionScalarSeed;
          });
      if (physicalized) {
        result.state = PhysicalFactState::Exact;
        result.physicalized = true;
        auto ranges = axisRanges(value, fragmentAxis);
        if (ranges.isExact())
          result.roots = std::move(ranges.roots);
        result.extentAuthority =
            PhysicalAxisRealizationFact::ExtentAuthority::Structural;
        return result;
      }
    }
  }

  if (Operation *producer = value.getDefiningOp();
      isa_and_nonnull<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp>(producer)) {
    // One input can establish extent authority while other inputs still need
    // relation closure. Query this edge's projection independently.
    for (Value operand : producer->getOperands()) {
      auto source = dyn_cast<FragmentType>(operand.getType());
      if (!source)
        continue;
      BroadcastProjection projection = queryAxisProjection(source, fragment);
      if (!projection.isExact() || !projection.targetToSource[fragmentAxis])
        continue;
      unsigned sourceAxis = *projection.targetToSource[fragmentAxis];
      if (source.getShape()[sourceAxis] != extent)
        continue;
      PhysicalAxisRealizationFact input = axisRealization(operand, sourceAxis);
      bool introducedUnit =
          extent.getKind() == PhysicalExprKind::Constant &&
          extent.getValue() == 1 && input.roots.empty();
      if (input.hasExtentAuthority() && input.physicalized &&
          !input.constructionScalarSeed && !introducedUnit) {
        result.state = PhysicalFactState::Exact;
        result.physicalized = true;
        result.roots = input.roots;
        result.extentAuthority =
            PhysicalAxisRealizationFact::ExtentAuthority::Structural;
        return result;
      }
    }
  }

  if (auto loop = value.getDefiningOp<scf::ForOp>()) {
    auto reductions = loop->getAttrOfType<ArrayAttr>(reductionSourcesAttr);
    OpOperand *initial = singleControlInput(value, ControlFlowEdgeKind::Bypass, nullptr);
    OpOperand *yielded = singleControlInput(value, ControlFlowEdgeKind::Exit, &loop.getRegion());
    bool preservesAxis = reductions && !reductions.empty() &&
                         llvm::none_of(reductions, [&](Attribute attribute) {
                           auto reduction = dyn_cast<PhysicalSourceAttr>(attribute);
                           return reduction &&
                                  PhysicalSourceAxis{reduction.getSourceId(),
                                                     reduction.getSourceAxis(),
                                                     reduction.getDerived()} ==
                                      result.source;
                         });
    if (preservesAxis && initial && yielded &&
        initial->get().getType() == fragment && yielded->get().getType() == fragment) {
      PhysicalRangeFact provenance =
          sourceRanges(yielded->get(), result.source);
      auto kind = extent.getKind();
      bool physicalExtent =
          kind != PhysicalExprKind::Dimension &&
          kind != PhysicalExprKind::ScalarABI &&
          (!(kind == PhysicalExprKind::Constant && extent.getValue() == 1) ||
           (!provenance.roots.empty() &&
            llvm::all_of(provenance.roots, [](MakeRangeOp range) {
              return isProvablySingletonLogicalRange(range) ||
                     isProgramCoordinateRange(range);
            })));
      bool exactYield = provenance.isExact() && !provenance.roots.empty() &&
                        llvm::all_of(provenance.roots, [&](MakeRangeOp range) {
                          FailureOr<int64_t> dimension =
                              queryRangeDimension(range);
                          return sourceAxisIdentity(range) == result.source &&
                                 succeeded(dimension) &&
                                 *dimension == result.dimensionId &&
                                 valueMatchesExtent(range.getExtent(), extent);
                        });
      if (physicalExtent && exactYield) {
        result.state = PhysicalFactState::Exact;
        result.physicalized = true;
        result.roots.append(provenance.roots.begin(), provenance.roots.end());
        result.extentAuthority =
            PhysicalAxisRealizationFact::ExtentAuthority::Range;
        return result;
      }
    }
  }

  if (auto reduce = value.getDefiningOp<ReduceOp>()) {
    auto opResult = dyn_cast<OpResult>(value);
    if (opResult && opResult.getResultNumber() < reduce.getSources().size()) {
      Value sourceValue = reduce.getSources()[opResult.getResultNumber()];
      auto source = dyn_cast<FragmentType>(sourceValue.getType());
      if (source) {
        llvm::SmallDenseSet<int64_t> reduced(reduce.getAxes().begin(),
                                             reduce.getAxes().end());
        SmallVector<unsigned> freeAxes;
        for (unsigned axis = 0; axis < source.getShape().size(); ++axis)
          if (!reduced.contains(axis))
            freeAxes.push_back(axis);
        if (fragmentAxis < freeAxes.size()) {
          unsigned sourceAxis = freeAxes[fragmentAxis];
          PhysicalAxisRealizationFact input =
              axisRealization(sourceValue, sourceAxis);
          if ((!input.hasExtentAuthority() || !input.physicalized) &&
              !input.constructionScalarSeed && input.roots.empty() &&
              input.blockers.empty()) {
            for (Value peer : reduce.getSources()) {
              auto peerType = dyn_cast<FragmentType>(peer.getType());
              if (peer == sourceValue || !peerType ||
                  peerType.getShape() != source.getShape() ||
                  peerType.getAxisMaps() != source.getAxisMaps() ||
                  peerType.getValidity() != source.getValidity() ||
                  peerType.getOwner() != source.getOwner())
                continue;
              auto coverage = axisRealization(peer, sourceAxis);
              if (coverage.hasExtentAuthority() && coverage.physicalized &&
                  !coverage.constructionScalarSeed) {
                coverage.roots.clear();
                input = std::move(coverage);
                break;
              }
            }
          }
          if (input.hasExtentAuthority() &&
              source.getShape()[sourceAxis] == extent) {
            result.state = PhysicalFactState::Exact;
            result.physicalized = input.physicalized;
            result.constructionScalarSeed = input.constructionScalarSeed;
            result.roots = input.roots;
            result.extentAuthority =
                PhysicalAxisRealizationFact::ExtentAuthority::Structural;
            return result;
          }
        }
      }
    }
  }

  if (auto contract = value.getDefiningOp<ContractOp>()) {
    auto axes = queryContractionAxes(contract);
    if (axes && fragmentAxis < axes->results.size()) {
      const auto &axis = axes->results[fragmentAxis];
      Value sourceValue = axis.operand == ContractionOperand::Lhs
                              ? Value(contract.getLhs()) : Value(contract.getRhs());
      auto source = cast<FragmentType>(sourceValue.getType());
      PhysicalAxisRealizationFact input = axisRealization(sourceValue, axis.axis);
      if (input.hasExtentAuthority() && source.getShape()[axis.axis] == extent) {
        result.state = PhysicalFactState::Exact;
        result.physicalized = input.physicalized;
        result.constructionScalarSeed = input.constructionScalarSeed;
        result.roots = input.roots;
        result.extentAuthority =
            PhysicalAxisRealizationFact::ExtentAuthority::Structural;
        return result;
      }
    }
  }

  if (value.getDefiningOp<ExtractOp>()) {
    // ExtractOp::verify requires the result type to equal the selected typed
    // record field.  MakeRecord and structured fold/scan verifiers in turn
    // require their field/result schemas to match the executable values at the
    // region boundary.  The projected extent is therefore already a current-IR
    // structural fact.  Keep any exact producer ranges as independent traversal
    // provenance: dropping them makes a reduction over an extracted record field
    // look unrelated to the range that produced that field.
    PhysicalRangeFact ranges = axisRanges(value, fragmentAxis);
    if (ranges.isExact()) {
      result.roots.append(ranges.roots.begin(), ranges.roots.end());
      result.constructionScalarSeed =
          extent.getKind() ==
              PhysicalExprKind::Constant &&
          extent.getValue() == 1 && !ranges.roots.empty() &&
          llvm::any_of(ranges.roots, [](MakeRangeOp range) {
            return !isProvablySingletonLogicalRange(range) &&
                   !isProgramCoordinateRange(range) &&
                   samePhysicalScalarExpression(range.getStart(),
                                                range.getLogicalStart());
          });
      result.physicalized =
          !result.constructionScalarSeed && !ranges.roots.empty() &&
          llvm::all_of(ranges.roots, [&](MakeRangeOp range) {
            return valueMatchesExtent(range.getExtent(), extent);
          });
    }
    result.state = PhysicalFactState::Exact;
    result.extentAuthority =
        PhysicalAxisRealizationFact::ExtentAuthority::Structural;
    return result;
  }

  PhysicalRangeFact ranges = axisRanges(value, fragmentAxis);
  result.roots.append(ranges.roots.begin(), ranges.roots.end());
  result.blockers.append(ranges.blockers.begin(), ranges.blockers.end());
  result.constructionScalarSeed =
      extent.getKind() ==
          PhysicalExprKind::Constant &&
      extent.getValue() == 1 && !ranges.roots.empty() &&
      llvm::any_of(ranges.roots, [](MakeRangeOp range) {
        return !isProvablySingletonLogicalRange(range) &&
               !isProgramCoordinateRange(range) &&
               samePhysicalScalarExpression(range.getStart(),
                                            range.getLogicalStart());
      });
  result.physicalized =
      !result.constructionScalarSeed && !ranges.roots.empty() &&
      llvm::all_of(ranges.roots, [&](MakeRangeOp range) {
         return valueMatchesExtent(range.getExtent(), extent);
       });
  if (value.getDefiningOp<ReshapeOp>()) {
    if (ranges.isExact() && ranges.roots.empty()) {
      if (extent.getKind() ==
              PhysicalExprKind::Constant &&
          extent.getValue() == 1) {
        result.state = PhysicalFactState::Exact;
        result.physicalized = true;
        result.extentAuthority =
            PhysicalAxisRealizationFact::ExtentAuthority::Structural;
      }
      return result;
    }
    // A verified reshape carries its own row-major physical reassociation.  Its
    // result extent remains exact even when no single pre-reshape range can be
    // projected to one split/merged result axis.  Keep that extent fact
    // separate from range provenance rather than turning a legal reshape into
    // analysis unknown.
    result.state = PhysicalFactState::Exact;
    result.extentAuthority =
        PhysicalAxisRealizationFact::ExtentAuthority::Structural;
    return result;
  }
  if (ranges.state == PhysicalFactState::Unknown || !ranges.blockers.empty())
    return result;
  if (!ranges.roots.empty() && failed(queryExactLogicalRange(ranges)) &&
      !lockstepRanges(ranges.roots).isExact()) {
    result.state = PhysicalFactState::Ambiguous;
    return result;
  }

  result.state = PhysicalFactState::Exact;
  if (result.physicalized)
    result.extentAuthority =
        PhysicalAxisRealizationFact::ExtentAuthority::Range;
  return result;
}

PhysicalContractFreeAxisFact
PhysicalProgramAnalysis::contractFreeAxes(Operation *operation) {
  PhysicalContractFreeAxisFact result;
  auto axes = queryContractionAxes(operation);
  if (!axes) {
    appendUnique(result.blockers, operation);
    return result;
  }
  auto appendOperand = [&](Value value, ArrayRef<unsigned> freeAxes) {
    bool exact = true;
    for (unsigned axis : freeAxes) {
      PhysicalContractFreeAxis fact;
      fact.operand = value;
      fact.operandAxis = axis;
      fact.realization = axisRealization(value, axis);
      fact.ranges = axisRanges(value, axis);
      result.blockers.append(fact.realization.blockers.begin(),
                             fact.realization.blockers.end());
      result.blockers.append(fact.ranges.blockers.begin(),
                             fact.ranges.blockers.end());
      if (!fact.realization.isExact()) {
        appendUnique(result.blockers, operation);
        exact = false;
      }
      result.axes.push_back(std::move(fact));
    }
    return exact;
  };
  Value lhs, rhs;
  if (auto contract = dyn_cast_or_null<ContractOp>(operation)) {
    lhs = contract.getLhs();
    rhs = contract.getRhs();
  } else if (auto contract = dyn_cast_or_null<ScaledContractOp>(operation)) {
    lhs = contract.getLhs();
    rhs = contract.getRhs();
  } else if (auto contract = dyn_cast_or_null<SparseContractOp>(operation)) {
    lhs = contract.getCompressed();
    rhs = contract.getRhs();
  }
  bool lhsExact = appendOperand(lhs, axes->lhsFree);
  bool rhsExact = appendOperand(rhs, axes->rhsFree);
  if (result.axes.empty()) {
    appendUnique(result.blockers, operation);
    return result;
  }
  if (lhsExact && rhsExact)
    result.state = PhysicalFactState::Exact;
  return result;
}

PhysicalRangeAxisFact
PhysicalProgramAnalysis::rangeAxes(Value value,
                                   ArrayRef<MakeRangeOp> selectedRoots) {
  PhysicalRangeAxisFact result;
  auto fragment = dyn_cast<FragmentType>(value.getType());
  if (!fragment || selectedRoots.empty())
    return result;
  if (auto splat = value.getDefiningOp<SplatOp>()) {
    (void)splat;
    result.state = PhysicalFactState::Exact;
    return result;
  }
  if (auto broadcast = value.getDefiningOp<BroadcastOp>();
      broadcast && !isa<FragmentType>(broadcast.getValue().getType())) {
    result.state = PhysicalFactState::Exact;
    return result;
  }
  result.state = PhysicalFactState::Exact;
  for (unsigned axis = 0; axis < fragment.getShape().size(); ++axis) {
    PhysicalRangeFact ranges = axisRanges(value, axis);
    auto axisMap = cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
    auto selectedRange = [&](MakeRangeOp range) {
      return llvm::any_of(selectedRoots, [&](MakeRangeOp selectedRoot) {
        if (range == selectedRoot || sameLogicalRange(range, selectedRoot))
          return true;
        // A renamed coordinate must project through the already selected axis.
        auto dimension = queryRangeDimension(range);
        auto selectedDimension = queryRangeDimension(selectedRoot);
        return succeeded(dimension) && succeeded(selectedDimension) &&
               *dimension == *selectedDimension &&
               axisMap.getDimensionId() == *selectedDimension &&
               sourceAxisIdentity(axisMap) == sourceAxisIdentity(selectedRoot) &&
               lockstepRanges({range, selectedRoot}).isExact();
      });
    };
    bool selected = llvm::any_of(ranges.roots, selectedRange);
    if (selected) {
      if (failed(queryExactLogicalRange(ranges)) &&
          (ranges.state == PhysicalFactState::Unknown ||
           !ranges.blockers.empty() ||
           !llvm::all_of(ranges.roots, selectedRange) ||
           !lockstepRanges(ranges.roots).isExact())) {
        result.state = PhysicalFactState::Ambiguous;
        result.blockers.append(ranges.blockers.begin(), ranges.blockers.end());
        return result;
      }
      result.fragmentAxes.push_back(axis);
      continue;
    }
    if (ranges.state != PhysicalFactState::Unknown)
      continue;
    auto mapping = cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
    bool mayCarrySelectedRoot = llvm::any_of(selectedRoots, [&](MakeRangeOp root) {
      FailureOr<int64_t> dimension = queryRangeDimension(root);
      return sourceAxisIdentity(mapping) == sourceAxisIdentity(root) &&
             succeeded(dimension) && mapping.getDimensionId() == *dimension;
    });
    if (!mayCarrySelectedRoot)
      continue;
    result.state = PhysicalFactState::Unknown;
    result.blockers.append(ranges.blockers.begin(), ranges.blockers.end());
    return result;
  }
  return result;
}

PhysicalLockstepTraversalFact PhysicalProgramAnalysis::lockstepTraversal(
    ValueRange sources, ArrayRef<unsigned> fragmentAxes) {
  PhysicalLockstepTraversalFact result;
  if (sources.empty() || sources.size() != fragmentAxes.size())
    return result;
  SmallVector<MakeRangeOp> authorities;
  for (auto [source, fragmentAxis] : llvm::zip(sources, fragmentAxes)) {
    PhysicalRangeFact ranges = axisRanges(source, fragmentAxis);
    if (!ranges.isExact()) {
      result.state = ranges.state == PhysicalFactState::Ambiguous
                         ? PhysicalLockstepState::Inconsistent
                         : PhysicalLockstepState::Unknown;
      result.blockers.append(ranges.blockers.begin(), ranges.blockers.end());
      return result;
    }
    PhysicalLockstepTraversalFact sourceFact = lockstepRanges(ranges.roots);
    if (!sourceFact.isExact()) {
      result.state = sourceFact.state;
      result.blockers.append(ranges.blockers.begin(), ranges.blockers.end());
      result.blockers.append(sourceFact.blockers.begin(),
                             sourceFact.blockers.end());
      return result;
    }
    authorities.push_back(sourceFact.authority);
  }
  return lockstepRanges(authorities);
}

PhysicalLockstepTraversalFact
PhysicalProgramAnalysis::lockstepRanges(ArrayRef<MakeRangeOp> ranges) {
  PhysicalLockstepTraversalFact result;
  if (ranges.empty())
    return result;
  auto sameValue = [](Value lhs, Value rhs) {
    if (samePhysicalScalarExpression(lhs, rhs))
      return true;
    PhysicalExprAttr left = queryLaunchExpression(lhs);
    PhysicalExprAttr right = queryLaunchExpression(rhs);
    return left && right && left == right;
  };
  auto sameLogicalTraversal = [&](MakeRangeOp lhs, MakeRangeOp rhs) {
    return sameValue(lhs.getLogicalStart(), rhs.getLogicalStart()) &&
           sameValue(lhs.getLogicalStop(), rhs.getLogicalStop()) &&
           sameValue(lhs.getStep(), rhs.getStep());
  };
  auto sameTraversal = [&](MakeRangeOp lhs, MakeRangeOp rhs) {
    auto isZero = [](Value value) {
      PhysicalExprAttr bound = queryNonNegativeIndexUpperBound(value);
      return bound &&
             bound.getKind() == PhysicalExprKind::Constant &&
             bound.getValue() == 0;
    };
    bool sameStart = sameValue(lhs.getStart(), rhs.getStart()) ||
                     (isZero(lhs.getStart()) && isZero(rhs.getStart()));
    return sameStart &&
           sameValue(lhs.getExtent(), rhs.getExtent()) &&
           sameValue(lhs.getStep(), rhs.getStep());
  };
  result.authority = ranges.front();
  for (MakeRangeOp range : llvm::drop_begin(ranges))
    if (!sameTraversal(result.authority, range) ||
        !sameLogicalTraversal(result.authority, range)) {
      result.state = PhysicalLockstepState::Inconsistent;
      return result;
    }
  result.state = PhysicalLockstepState::Exact;
  return result;
}

} // namespace intent::gpu
