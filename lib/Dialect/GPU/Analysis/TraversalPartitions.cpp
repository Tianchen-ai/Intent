#include "Intent/Dialect/GPU/Analysis/TraversalPartitions.h"
#include "Intent/Dialect/GPU/Analysis/IndexRelations.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/IR/ProgramInterface.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;

namespace intent::gpu {
namespace {

bool sameIndex(Value lhs, Value rhs, const IndexRelations &relations) {
  return relations.same(lhs, rhs) ||
      (lhs.getType().isIndex() && rhs.getType().isIndex() &&
       relations.atMost(lhs, rhs) && relations.atMost(rhs, lhs));
}

std::optional<bool> constantPredicate(Value value, const IndexRelations &relations) {
  if (auto constant = relations.constant(value)) return *constant != 0;
  auto compare = value.getDefiningOp<CompareOp>();
  if (!compare || !compare.getLhs().getType().isIndex() ||
      !compare.getRhs().getType().isIndex()) return std::nullopt;
  if (compare.getPredicate() == ComparePredicate::Lt) {
    if (relations.lessThan(compare.getLhs(), compare.getRhs())) return true;
    if (relations.atMost(compare.getRhs(), compare.getLhs())) return false;
  }
  if (compare.getPredicate() == ComparePredicate::Ge) {
    if (relations.atMost(compare.getRhs(), compare.getLhs())) return true;
    if (relations.lessThan(compare.getLhs(), compare.getRhs())) return false;
  }
  return std::nullopt;
}

bool tailCondition(Value condition, Value end, Value upper,
                   const IndexRelations &relations) {
  if (auto compare = condition.getDefiningOp<CompareOp>())
    return compare.getPredicate() == ComparePredicate::Lt &&
        sameIndex(compare.getLhs(), end, relations) &&
        sameIndex(compare.getRhs(), upper, relations);
  auto binary = condition.getDefiningOp<BinaryOp>();
  if (!binary || !condition.getType().isInteger(1)) return false;
  bool conjunction = binary.getOperatorKind() == BinaryOperator::LogicalAnd;
  bool disjunction = binary.getOperatorKind() == BinaryOperator::LogicalOr;
  if (!conjunction && !disjunction) return false;
  for (auto [constant, remaining] : {std::pair{binary.getLhs(), binary.getRhs()},
                                   std::pair{binary.getRhs(), binary.getLhs()}}) {
    auto truth = constantPredicate(constant, relations);
    if (truth && *truth == conjunction && tailCondition(remaining, end, upper, relations))
      return true;
  }
  return false;
}

class CoordinateMatch {
public:
  CoordinateMatch(scf::ForOp prefix, IndexRelations &relations)
      : prefix(prefix), relations(relations) {
    bindings[prefix.getInductionVar()] = prefix.getUpperBound();
    for (auto [argument, result] : llvm::zip(prefix.getRegionIterArgs(), prefix.getResults()))
      bindings[argument] = result;
  }

  bool same(Value lhs, Value rhs) {
    if (auto found = bindings.find(lhs); found != bindings.end())
      return sameIndex(found->second, rhs, relations);
    if (lhs == rhs) return true;
    if (lhs.getType() != rhs.getType()) return false;
    if (sameIndex(lhs, rhs, relations)) return true;
    auto left = dyn_cast<OpResult>(lhs), right = dyn_cast<OpResult>(rhs);
    if (!left || !right || left.getResultNumber() != right.getResultNumber()) return false;
    Operation *a = left.getOwner(), *b = right.getOwner();
    if (!coordinateNode(a) || !coordinateNode(b)) return false;
    if (!OperationEquivalence::isEquivalentTo(a, b,
          [&](Value x, Value y) { return success(same(x, y)); }, nullptr,
          OperationEquivalence::IgnoreLocations)) return false;
    bindings[lhs] = rhs;
    return true;
  }

  void bind(Value lhs, Value rhs) { bindings[lhs] = rhs; }

  bool canMatchClone(Value value) {
    if (prefix.isDefinedOutsideOfLoop(value) || bindings.contains(value) ||
        relations.constant(value)) return true;
    if (auto found = cloneable.find(value); found != cloneable.end()) return found->second;
    Operation *producer = value.getDefiningOp();
    bool result = producer && coordinateNode(producer) &&
        llvm::all_of(producer->getOperands(), [&](Value operand) {
          return canMatchClone(operand);
        });
    cloneable[value] = result;
    return result;
  }

private:
  static bool coordinateNode(Operation *operation) {
    return !operation->getNumRegions() && (isa<MakeRangeOp>(operation) ||
        isPhysicalReplayNode(operation, PhysicalReplayScope::Coordinate, false));
  }

  scf::ForOp prefix;
  IndexRelations &relations;
  llvm::DenseMap<Value, Value> bindings;
  llvm::DenseMap<Value, bool> cloneable;
};

bool collectControlPath(Operation *effect, Region *boundary,
                        SmallVectorImpl<Region *> &path) {
  for (Region *region = effect->getParentRegion(); region != boundary;) {
    Operation *owner = region ? region->getParentOp() : nullptr;
    // A nested traversal must correspond on both sides. Stateful loops and
    // unknown control cannot be inferred from their trip count alone.
    if (auto loop = dyn_cast_or_null<scf::ForOp>(owner)) {
      if (loop.getNumRegionIterArgs()) return false;
    } else if (!isa_and_nonnull<scf::IfOp>(owner)) return false;
    path.push_back(region);
    region = owner->getParentRegion();
  }
  return true;
}

bool matchingControl(Operation *first, Operation *second, scf::ForOp prefix,
                     scf::IfOp suffix, CoordinateMatch &match) {
  SmallVector<Region *> left, right;
  if (!collectControlPath(first, &prefix.getRegion(), left) ||
      !collectControlPath(second, &suffix.getThenRegion(), right) ||
      left.size() != right.size()) return false;
  for (auto [a, b] : llvm::zip(llvm::reverse(left), llvm::reverse(right))) {
    if (auto loop = dyn_cast<scf::ForOp>(a->getParentOp())) {
      auto other = dyn_cast<scf::ForOp>(b->getParentOp());
      if (!other || loop->getAttrs() != other->getAttrs() ||
          !match.same(loop.getLowerBound(), other.getLowerBound()) ||
          !match.same(loop.getUpperBound(), other.getUpperBound()) ||
          !match.same(loop.getStep(), other.getStep())) return false;
      match.bind(loop.getInductionVar(), other.getInductionVar());
    } else {
      auto branch = cast<scf::IfOp>(a->getParentOp());
      auto other = dyn_cast<scf::IfOp>(b->getParentOp());
      if (!other || (a == &branch.getThenRegion()) != (b == &other.getThenRegion()) ||
          !match.same(branch.getCondition(), other.getCondition())) return false;
    }
  }
  return true;
}

bool confinesTail(Value validity, MakeRangeOp range, PhysicalProgramAnalysis &analysis,
                  const IndexRelations &relations) {
  if (relations.coordinateLessThan(range.getResult(), range.getLogicalStop())) return true;
  if (!validity) return false;
  if (auto broadcast = validity.getDefiningOp<BroadcastOp>())
    return confinesTail(broadcast.getValue(), range, analysis, relations);
  if (auto splat = validity.getDefiningOp<SplatOp>())
    return confinesTail(splat.getValue(), range, analysis, relations);
  if (auto reshape = validity.getDefiningOp<ReshapeOp>())
    return confinesTail(reshape.getValue(), range, analysis, relations);
  if (auto transpose = validity.getDefiningOp<TransposeOp>())
    return confinesTail(transpose.getValue(), range, analysis, relations);
  if (auto binary = validity.getDefiningOp<BinaryOp>()) {
    if (binary.getOperatorKind() == BinaryOperator::LogicalAnd)
      return confinesTail(binary.getLhs(), range, analysis, relations) ||
             confinesTail(binary.getRhs(), range, analysis, relations);
    if (binary.getOperatorKind() == BinaryOperator::LogicalOr)
      return confinesTail(binary.getLhs(), range, analysis, relations) &&
             confinesTail(binary.getRhs(), range, analysis, relations);
    return false;
  }
  auto compare = validity.getDefiningOp<CompareOp>();
  if (!compare || compare.getPredicate() != ComparePredicate::Lt) return false;
  Value limit = uniformScalarSource(compare.getRhs());
  return limit && sameIndex(limit, range.getLogicalStop(), relations) &&
      analysis.isTailPredicate(validity, {{range, range.getLogicalStop()}});
}

bool partitionRange(MakeRangeOp range, Value start, Value step, Value upper,
                    const IndexRelations &relations) {
  auto type = range.getResult().getType();
  return type.getShape().size() == 1 && isUnitStepRange(range) &&
      sameIndex(range.getStart(), start, relations) &&
      sameIndex(range.getExtent(), step, relations) &&
      relations.constant(range.getLogicalStart()) == 0 &&
      sameIndex(range.getLogicalStop(), upper, relations) &&
      queryLaunchExpression(step) == type.getShape()[0];
}

bool matchingStores(Operation *first, Operation *second, scf::ForOp prefix,
                    scf::IfOp suffix, Value upper, PhysicalProgramAnalysis &analysis,
                    IndexRelations &relations) {
  auto a = analysis.footprint(first), b = analysis.footprint(second);
  if (a.state != PhysicalFactState::Exact || b.state != PhysicalFactState::Exact ||
      a.rangeState != PhysicalFactState::Exact || b.rangeState != PhysicalFactState::Exact ||
      a.resource != b.resource || a.sourceAxes != b.sourceAxes ||
      a.coordinates.size() != b.coordinates.size()) return false;
  // The anchored coordinate separates the active Store domains. Additional
  // predicates, including other axes' subregion bounds, can only narrow them;
  // they need not describe a native rectangular boundary. This query proves
  // disjointness, not coverage: each Store retains its original validity.
  if (!analysis.accessBounds(first).isExact() || !analysis.accessBounds(second).isExact())
    return false;
  CoordinateMatch match(prefix, relations);
  if (!matchingControl(first, second, prefix, suffix, match)) return false;
  bool anchored = false;
  for (auto [left, right] : llvm::zip(a.coordinates, b.coordinates)) {
    auto x = left.getDefiningOp<MakeRangeOp>(), y = right.getDefiningOp<MakeRangeOp>();
    if (x && y && sameIndex(x.getStart(), prefix.getInductionVar(), relations)) {
      if (x.getResult().getType() != y.getResult().getType() ||
          !(sourceAxisIdentity(x) == sourceAxisIdentity(y)) ||
          !partitionRange(x, prefix.getInductionVar(), prefix.getStep(), upper, relations) ||
          !partitionRange(y, prefix.getUpperBound(), prefix.getStep(), upper, relations) ||
          !confinesTail(b.validity, y, analysis, relations)) return false;
      anchored = true;
      continue;
    }
    if (!match.same(left, right)) return false;
  }
  return anchored;
}

bool prefixAndSuffix(Operation *first, Operation *second,
                     PhysicalProgramAnalysis &analysis) {
  if (!isa<StoreOp>(first) || !isa<StoreOp>(second)) return false;
  IndexRelations relations;
  for (Operation *owner = first->getParentOp(); owner;
       owner = owner->getParentOp()) {
    auto prefix = dyn_cast<scf::ForOp>(owner);
    if (!prefix) continue;
    if (!prefix.getInductionVar().getType().isIndex() ||
        relations.constant(prefix.getLowerBound()) != 0 ||
        !relations.positive(prefix.getStep()))
      continue;
    auto rounded = prefix.getUpperBound().getDefiningOp<BinaryOp>();
    if (!rounded || rounded.getOperatorKind() != BinaryOperator::Multiply)
      continue;
    Value quotient;
    if (relations.same(rounded.getRhs(), prefix.getStep()))
      quotient = rounded.getLhs();
    else if (relations.same(rounded.getLhs(), prefix.getStep()))
      quotient = rounded.getRhs();
    auto division = quotient ? quotient.getDefiningOp<BinaryOp>() : BinaryOp();
    if (!division || division.getOperatorKind() != BinaryOperator::FloorDivide ||
        !relations.same(division.getRhs(), prefix.getStep()) ||
        !relations.nonnegative(division.getLhs()))
      continue;
    for (Operation *other = second->getParentOp(); other;
         other = other->getParentOp()) {
      auto suffix = dyn_cast<scf::IfOp>(other);
      if (!suffix || suffix.getThenRegion().empty() ||
          !suffix.getThenRegion().isAncestor(second->getParentRegion()) ||
          prefix->getBlock() != suffix->getBlock() ||
          !prefix->isBeforeInBlock(suffix))
        continue;
      if (!tailCondition(suffix.getCondition(), prefix.getUpperBound(), division.getLhs(), relations))
        continue;
      if (matchingStores(first, second, prefix, suffix, division.getLhs(), analysis, relations))
        return true;
    }
  }
  return false;
}

} // namespace

bool areDisjointTraversalPartitions(Operation *first, Operation *second,
                                   PhysicalProgramAnalysis &analysis) {
  return prefixAndSuffix(first, second, analysis) || prefixAndSuffix(second, first, analysis);
}

bool canPeelTraversalEffects(scf::ForOp loop, PhysicalProgramAnalysis &analysis) {
  IndexRelations relations;
  llvm::SmallDenseSet<int64_t> origins;
  WalkResult result = loop.walk([&](Operation *operation) {
    auto access = dyn_cast<AccessOpInterface>(operation);
    if (!access || !access.writesMemory()) return WalkResult::advance();
    Value resource = access.getAccessResource();
    if (!operation->hasAttr(originAttr) &&
        (isa<BufferType>(resource.getType()) || isInvocationWorkspace(resource)))
      return WalkResult::advance();
    if (!isa<StoreOp>(operation)) return WalkResult::interrupt();
    // Existing exclusive definitions would also need cross-arm partition
    // proofs after cloning. The current matcher proves corresponding paths.
    auto origin = operation->getAttrOfType<IntegerAttr>(originAttr);
    if (!origin || !origins.insert(origin.getInt()).second) return WalkResult::interrupt();
    auto footprint = analysis.footprint(operation);
    if (footprint.state != PhysicalFactState::Exact ||
        footprint.rangeState != PhysicalFactState::Exact ||
        !loop.isDefinedOutsideOfLoop(footprint.resource) ||
        !analysis.accessBounds(operation).isExact()) return WalkResult::interrupt();
    CoordinateMatch match(loop, relations);
    SmallVector<Region *> control;
    if (!collectControlPath(operation, &loop.getRegion(), control)) return WalkResult::interrupt();
    for (Region *region : llvm::reverse(control)) {
      if (auto nested = dyn_cast<scf::ForOp>(region->getParentOp())) {
        if (!match.canMatchClone(nested.getLowerBound()) ||
            !match.canMatchClone(nested.getUpperBound()) ||
            !match.canMatchClone(nested.getStep())) return WalkResult::interrupt();
        match.bind(nested.getInductionVar(), nested.getInductionVar());
      } else if (!match.canMatchClone(cast<scf::IfOp>(region->getParentOp()).getCondition())) {
        return WalkResult::interrupt();
      }
    }
    bool anchored = false;
    for (Value coordinate : footprint.coordinates) {
      auto range = coordinate.getDefiningOp<MakeRangeOp>();
      if (range && sameIndex(range.getStart(), loop.getInductionVar(), relations)) {
        if (!partitionRange(range, loop.getInductionVar(), loop.getStep(),
                            loop.getUpperBound(), relations) ||
            !confinesTail(footprint.validity, range, analysis, relations))
          return WalkResult::interrupt();
        anchored = true;
      } else if (!match.canMatchClone(coordinate)) return WalkResult::interrupt();
    }
    return anchored ? WalkResult::advance() : WalkResult::interrupt();
  });
  return !result.wasInterrupted();
}

} // namespace intent::gpu
