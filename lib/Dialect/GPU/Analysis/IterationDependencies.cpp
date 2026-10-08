#include "Intent/Dialect/GPU/Analysis/IterationDependencies.h"
#include "Intent/Analysis/IntegerRelations.h"
#include "Intent/Dialect/GPU/Analysis/Helpers.h"
#include "Intent/Dialect/GPU/Analysis/IndexRelations.h"
#include "Intent/Dialect/GPU/Analysis/IntegerRanges.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalExpressionBounds.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "ScalarExpressions.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <functional>
#include <limits>

using namespace mlir;

namespace intent::gpu {
namespace {

bool isOutsideIterationRange(Value coordinate, scf::ForOp loop) {
  if (!coordinate.getType().isIndex() ||
      !loop.isDefinedOutsideOfLoop(coordinate))
    return false;
  IndexRelations relations;
  if (relations.same(coordinate, loop.getUpperBound())) return true;
  auto index = relations.constant(coordinate);
  auto lower = relations.constant(loop.getLowerBound());
  auto upper = relations.constant(loop.getUpperBound());
  if (index && ((lower && *index < *lower) || (upper && *index >= *upper)))
    return true;
  auto successor = loop.getLowerBound().getDefiningOp<BinaryOp>();
  if (!successor || successor.getOperatorKind() != BinaryOperator::Add)
    return false;
  auto induction = dyn_cast<BlockArgument>(coordinate);
  auto owner = induction
                   ? dyn_cast<scf::ForOp>(induction.getOwner()->getParentOp())
                   : scf::ForOp();
  if (!owner || coordinate != owner.getInductionVar() ||
      relations.constant(owner.getStep()) != 1 ||
      !owner->isProperAncestor(loop))
    return false;
  // An active unit-step induction value is below its signed upper bound, so
  // its successor is representable even at the last iteration.
  for (auto [base, offset] :
       {std::pair{successor.getLhs(), successor.getRhs()},
        std::pair{successor.getRhs(), successor.getLhs()}})
    if (base == coordinate && relations.constant(offset) == 1) return true;
  return false;
}

bool isUnitIterationCoordinate(Value coordinate, scf::ForOp loop) {
  llvm::DenseMap<Value, IntegerDifference> differences;
  llvm::DenseSet<Value> active;
  std::function<IntegerDifference(Value)> difference =
      [&](Value value) -> IntegerDifference {
    if (value == loop.getInductionVar()) return 1;
    if (loop.isDefinedOutsideOfLoop(value)) return 0;
    if (auto known = differences.find(value); known != differences.end())
      return known->second;
    if (!value.getType().isIndex() || !active.insert(value).second)
      return std::nullopt;
    Operation *producer = value.getDefiningOp();
    IntegerDifference result;
    if (producer && !producer->getNumRegions() && isMemoryEffectFree(producer)) {
      UniformExpression expression = describeUniformValue(value);
      if (expression.kind == UniformKind::Cast) {
        // A signed i32 IV may be widened to an address without changing the
        // body arithmetic. The shared modular difference query is i64-only.
        if (expression.operands.size() == 1 &&
            isValuePreservingIntegerCast(expression.operands.front(), value.getType()))
          result = difference(expression.operands.front());
      } else {
        bool preservesInteger = true;
        if (expression.kind == UniformKind::Add ||
            expression.kind == UniformKind::Subtract ||
            expression.kind == UniformKind::Multiply)
          preservesInteger = integerOperationDoesNotWrap(value);
        if (preservesInteger)
          result = foldIntegerDifference(expression, difference,
              [](Value operand) { return IndexRelations().constant(operand); }, 64);
      }
    }
    active.erase(value);
    differences[value] = result;
    return result;
  };
  return coordinate.getType().isIndex() &&
         difference(coordinate) == IntegerDifference(1);
}

bool uniformIntegerIs(Value value, unsigned expected) {
  UniformValueAnalysis analysis(describeUniformValue);
  auto integer = dyn_cast_or_null<IntegerAttr>(analysis.evaluate(value));
  return integer && (expected ? integer.getValue().isOne() : integer.getValue().isZero());
}

std::optional<int64_t> nonnegativeIntegerLimit(Type type) {
  if (type.isIndex()) return std::numeric_limits<int64_t>::max();
  auto integer = dyn_cast<IntegerType>(type);
  if (!integer || integer.getWidth() <= 1 || integer.getWidth() > 64)
    return std::nullopt;
  auto maximum = integer.isUnsigned() ? APInt::getMaxValue(integer.getWidth())
                                     : APInt::getSignedMaxValue(integer.getWidth());
  return maximum.getLimitedValue(std::numeric_limits<int64_t>::max());
}

bool zeroOneValue(Value value) {
  if (uniformElementType(value.getType()).isInteger(1) ||
      uniformIntegerIs(value, 0) || uniformIntegerIs(value, 1)) return true;
  if (auto cast = value.getDefiningOp<CastOp>())
    return nonnegativeIntegerLimit(uniformElementType(value.getType())) &&
           zeroOneValue(cast.getValue());
  if (auto select = value.getDefiningOp<SelectOp>())
    return zeroOneValue(select.getTrueValue()) && zeroOneValue(select.getFalseValue());
  if (Value forwarded = laneWiseProjectionSource(value.getDefiningOp()))
    return zeroOneValue(forwarded);
  return false;
}

using PredicateFacts = SmallVector<std::pair<Value, bool>>;

void appendNonzeroFacts(Value value, PredicateFacts &facts);

void appendPredicateFacts(Value value, bool truth, PredicateFacts &facts) {
  if (!value || !value.getType().isInteger(1) ||
      llvm::is_contained(facts, std::pair{value, truth})) return;
  facts.emplace_back(value, truth);
  auto expression = describeUniformValue(value);
  if (expression.kind == UniformKind::Not && expression.operands.size() == 1)
    appendPredicateFacts(expression.operands.front(), !truth, facts);
  else if ((expression.kind == UniformKind::And && truth) ||
           (expression.kind == UniformKind::Or && !truth))
    for (Value operand : expression.operands) appendPredicateFacts(operand, truth, facts);
  else if (auto compare = value.getDefiningOp<CompareOp>();
           compare && (compare.getPredicate() == ComparePredicate::Eq ||
                       compare.getPredicate() == ComparePredicate::Ne)) {
    bool equal = truth == (compare.getPredicate() == ComparePredicate::Eq);
    for (auto [member, constant] :
         {std::pair{compare.getLhs(), compare.getRhs()}, std::pair{compare.getRhs(), compare.getLhs()}})
      if ((equal && uniformIntegerIs(constant, 1)) ||
          (!equal && uniformIntegerIs(constant, 0)))
        appendNonzeroFacts(member, facts);
  }
}

void appendNonzeroFacts(Value value, PredicateFacts &facts) {
  if (value.getType().isInteger(1)) {
    appendPredicateFacts(value, true, facts);
  } else if (auto cast = value.getDefiningOp<CastOp>();
             cast && isa<IntegerType, IndexType>(value.getType()) &&
             isa<IntegerType, IndexType>(cast.getValue().getType())) {
    // Integer casts always map zero to zero, even when narrowing. A nonzero
    // result therefore implies a nonzero source without granting invertibility.
    appendNonzeroFacts(cast.getValue(), facts);
  }
  if (auto select = value.getDefiningOp<SelectOp>()) {
    if (uniformIntegerIs(select.getFalseValue(), 0)) {
      appendPredicateFacts(select.getCondition(), true, facts);
      appendNonzeroFacts(select.getTrueValue(), facts);
    } else if (uniformIntegerIs(select.getTrueValue(), 0)) {
      appendPredicateFacts(select.getCondition(), false, facts);
      appendNonzeroFacts(select.getFalseValue(), facts);
    }
  }
}

PredicateFacts accessPredicateFacts(AccessOpInterface access, scf::ForOp loop) {
  PredicateFacts facts;
  appendPredicateFacts(access.getAccessValidity(), true, facts);
  for (Region *region = access->getParentRegion(); region && region != &loop.getRegion();) {
    Operation *owner = region->getParentOp();
    if (!owner || owner == loop) break;
    if (auto branch = dyn_cast<scf::IfOp>(owner))
      appendPredicateFacts(branch.getCondition(), region == &branch.getThenRegion(), facts);
    region = owner->getParentRegion();
  }
  return facts;
}

struct CountPrefixProof {
  ScanOp scan;
  scf::ForOp loop;
  Value predicate;
  bool predicateTruth = true;
  int64_t maximum;

  static std::optional<CountPrefixProof> query(ScanOp scan, scf::ForOp loop) {
    if (!scan || scan.getSources().size() != 1 || scan.getIdentities().size() != 1 ||
        !scan.getCaptures().empty() || scan.getAxis() != 0 || scan.getReverse() ||
        !scan.getInclusive() || !loop.getInductionVar().getType().isIndex() ||
        IndexRelations().constant(loop.getLowerBound()) != 0)
      return std::nullopt;
    auto kernel = loop->getParentOfType<func::FuncOp>();
    if (!loop.isDefinedOutsideOfLoop(scan.getResult(0)) ||
        !DominanceInfo(kernel).dominates(scan.getResult(0), loop.getOperation()))
      return std::nullopt;
    auto type = dyn_cast<FragmentType>(scan.getSources().front().getType());
    auto limit = type ? nonnegativeIntegerLimit(type.getElementType()) : std::nullopt;
    auto combine = queryBinaryCombine(scan.getCombine());
    if (!type || type.getShape().size() != 1 || !limit ||
        !combine || combine->kind() != BinaryOperator::Add ||
        !uniformIntegerIs(scan.getIdentities().front(), 0) ||
        !zeroOneValue(scan.getSources().front())) return std::nullopt;
    PhysicalProgramAnalysis analysis(scan->getParentOfType<func::FuncOp>());
    auto ranges = analysis.axisRanges(scan.getSources().front(), 0);
    auto root = queryExactLogicalRange(ranges);
    if (failed(root) || !analysis.lockstepRanges(ranges.roots).isExact())
      return std::nullopt;
    for (MakeRangeOp range : ranges.roots)
      if (IndexRelations().constant(range.getStart()) != 0 ||
          IndexRelations().constant(range.getLogicalStart()) != 0 ||
          IndexRelations().constant(range.getStep()) != 1 ||
          !samePhysicalScalarExpression(range.getLogicalStop(), loop.getUpperBound()))
        return std::nullopt;
    CountPrefixProof result{scan, loop, {}, true, *limit};
    Value source = scan.getSources().front();
    if (auto cast = source.getDefiningOp<CastOp>();
        cast && uniformElementType(cast.getValue().getType()).isInteger(1))
      result.predicate = cast.getValue();
    else if (auto select = source.getDefiningOp<SelectOp>()) {
      if (uniformIntegerIs(select.getTrueValue(), 1) && uniformIntegerIs(select.getFalseValue(), 0))
        result.predicate = select.getCondition();
      else if (uniformIntegerIs(select.getTrueValue(), 0) && uniformIntegerIs(select.getFalseValue(), 1)) {
        result.predicate = select.getCondition();
        result.predicateTruth = false;
      }
    }
    return result;
  }

  bool knownPredicate(Value value, bool truth, const PredicateFacts &facts) {
    UniformValueAnalysis uniform(describeUniformValue);
    if (auto constant = uniformBoolean(uniform.evaluate(value))) return *constant == truth;
    if (llvm::any_of(facts, [&](auto fact) {
          return fact.second == truth && samePhysicalScalarExpression(fact.first, value);
        })) return true;
    auto expression = describeUniformValue(value);
    if (expression.kind == UniformKind::Not && expression.operands.size() == 1)
      return knownPredicate(expression.operands.front(), !truth, facts);
    if ((expression.kind == UniformKind::And && truth) ||
        (expression.kind == UniformKind::Or && !truth))
      return llvm::all_of(expression.operands, [&](Value operand) {
        return knownPredicate(operand, truth, facts);
      });
    auto compare = value.getDefiningOp<CompareOp>();
    if (!compare) return false;
    auto predicate = compare.getPredicate();
    Value lhs = compare.getLhs(), rhs = compare.getRhs();
    // These facts come from this actual active loop, not independent intervals.
    if (lhs == loop.getInductionVar() && samePhysicalScalarExpression(rhs, loop.getUpperBound()))
      return (predicate == ComparePredicate::Lt && truth) ||
             (predicate == ComparePredicate::Ge && !truth);
    if (lhs == loop.getInductionVar() && IndexRelations().constant(rhs) == 0)
      return (predicate == ComparePredicate::Ge && truth) ||
             (predicate == ComparePredicate::Lt && !truth);
    return false;
  }

  bool selectedMember(Value value, Value whole, const PredicateFacts &facts,
                      bool zeroFillAllowed = false) {
    if (value.getType() != uniformElementType(whole.getType())) return false;
    if (value == whole && !isa<FragmentType>(whole.getType())) return true;
    UniformValueAnalysis uniform(describeUniformValue);
    auto literal = uniform.evaluate(value), sourceLiteral = uniform.evaluate(whole);
    if (literal && sourceLiteral && equalUniformConstants(literal, sourceLiteral)) return true;
    if (auto select = value.getDefiningOp<SelectOp>()) {
      if (knownPredicate(select.getCondition(), true, facts))
        return selectedMember(select.getTrueValue(), whole, facts, zeroFillAllowed);
      if (knownPredicate(select.getCondition(), false, facts))
        return selectedMember(select.getFalseValue(), whole, facts, zeroFillAllowed);
    }
    if (auto gather = value.getDefiningOp<GatherOp>()) {
      if (gather.getSource() != whole || gather.getCoordinates().size() != 1 ||
          gather.getSourceAxes() != ArrayRef<int64_t>{0} ||
          gather.getCoordinates().front() != loop.getInductionVar() ||
          !gather.getResult().getType().isIntOrIndexOrFloat()) return false;
      return !gather.getValid() || knownPredicate(gather.getValid(), true, facts) ||
             (zeroFillAllowed && gather.getFill() && uniformIntegerIs(gather.getFill(), 0));
    }
    auto scalarCast = value.getDefiningOp<CastOp>();
    auto sourceCast = whole.getDefiningOp<CastOp>();
    if (scalarCast && sourceCast)
      return scalarCast.getValue().getType() == uniformElementType(sourceCast.getValue().getType()) &&
             selectedMember(scalarCast.getValue(), sourceCast.getValue(), facts,
                            zeroFillAllowed && zeroOneValue(sourceCast.getValue()));
    if (auto scalarCompare = value.getDefiningOp<CompareOp>()) {
      auto sourceCompare = whole.getDefiningOp<CompareOp>();
      return sourceCompare && scalarCompare.getPredicate() == sourceCompare.getPredicate() &&
             selectedMember(scalarCompare.getLhs(), sourceCompare.getLhs(), facts) &&
             selectedMember(scalarCompare.getRhs(), sourceCompare.getRhs(), facts);
    }
    if (auto scalarSelect = value.getDefiningOp<SelectOp>()) {
      auto sourceSelect = whole.getDefiningOp<SelectOp>();
      return sourceSelect &&
             selectedMember(scalarSelect.getCondition(), sourceSelect.getCondition(), facts) &&
             selectedMember(scalarSelect.getTrueValue(), sourceSelect.getTrueValue(), facts) &&
             selectedMember(scalarSelect.getFalseValue(), sourceSelect.getFalseValue(), facts);
    }
    if (Value forwarded = laneWiseProjectionSource(whole.getDefiningOp()))
      return selectedMember(value, forwarded, facts, zeroFillAllowed);
    return false;
  }

  bool active(AccessOpInterface access) {
    if (uniformIntegerIs(scan.getSources().front(), 1)) return true;
    PredicateFacts facts = accessPredicateFacts(access, loop);
    for (auto [condition, truth] : facts) {
      if (predicate && truth == predicateTruth &&
          selectedMember(condition, predicate, facts, /*zeroFillAllowed=*/truth)) return true;
      auto compare = condition.getDefiningOp<CompareOp>();
      if (!compare || (compare.getPredicate() != ComparePredicate::Eq &&
                       compare.getPredicate() != ComparePredicate::Ne)) continue;
      bool equal = truth == (compare.getPredicate() == ComparePredicate::Eq);
      for (auto [member, constant] :
           {std::pair{compare.getLhs(), compare.getRhs()}, std::pair{compare.getRhs(), compare.getLhs()}}) {
        if (((equal && uniformIntegerIs(constant, 1)) ||
             (!equal && uniformIntegerIs(constant, 0))) &&
            selectedMember(member, scan.getSources().front(), facts, true)) return true;
        if (predicate && ((equal && uniformIntegerIs(constant, predicateTruth)) ||
                          (!equal && uniformIntegerIs(constant, !predicateTruth))) &&
            selectedMember(member, predicate, facts, predicateTruth)) return true;
      }
    }
    return false;
  }

  bool coordinate(Value value, AccessOpInterface access, int64_t &limit) {
    if (!value.getType().isIndex() || !active(access)) return false;
    PredicateFacts facts = accessPredicateFacts(access, loop);
    bool subtracted = false;
    while (true) {
      auto bound = nonnegativeIntegerLimit(value.getType());
      if (!bound) return false;
      limit = std::min(limit, *bound);
      if (auto cast = value.getDefiningOp<CastOp>()) { value = cast.getValue(); continue; }
      if (auto subtract = value.getDefiningOp<BinaryOp>();
          subtract && !subtracted && subtract.getOperatorKind() == BinaryOperator::Subtract &&
          uniformIntegerIs(subtract.getRhs(), 1)) {
        subtracted = true;
        value = subtract.getLhs();
        continue;
      }
      auto gather = value.getDefiningOp<GatherOp>();
      if (!gather || !subtracted) return false;
      Value root = gather.getSource();
      while (auto cast = root.getDefiningOp<CastOp>()) {
        auto maximum = nonnegativeIntegerLimit(uniformElementType(cast.getType()));
        if (!maximum) return false;
        limit = std::min(limit, *maximum);
        root = cast.getValue();
      }
      return root == scan.getResult(0) && selectedMember(value, gather.getSource(), facts);
    }
  }
};

// These operations only replicate or rearrange the same coordinate values.
// Stripping them here proves set membership, not an axis/layout equivalence.
Value coordinateValues(Value value) {
  while (Operation *producer = value.getDefiningOp()) {
    if (!isa<SplatOp, BroadcastOp, ReshapeOp, TransposeOp>(producer)) break;
    value = producer->getOperand(0);
  }
  return value;
}

bool ownsCoordinate(Value value, ExecutionGroupOp group, unsigned ordinal,
                    func::FuncOp kernel) {
  IndexRelations relations;
  value = coordinateValues(value);
  auto range = value.getDefiningOp<MakeRangeOp>();
  Value start = range ? range.getStart() : value;
  if (auto workset = start.getDefiningOp<WorksetCoordinateOp>()) {
    if (relations.constant(workset.getStep()) != 1) return false;
    start = workset.getCoordinate();
  }
  if (!range) return relations.same(start, group.getCoordinates()[ordinal]);
  if (relations.constant(range.getStep()) != 1 ||
      !relations.alignedUnitWindow(range.getStart(), range.getExtent()))
    return false;
  auto product = detail::stripScalarIdentity(start).getDefiningOp<BinaryOp>();
  if (!product || product.getOperatorKind() != BinaryOperator::Multiply)
    return false;
  Value coordinate = group.getCoordinates()[ordinal];
  if (!((relations.same(product.getLhs(), coordinate) &&
         relations.same(product.getRhs(), range.getExtent())) ||
        (relations.same(product.getRhs(), coordinate) &&
         relations.same(product.getLhs(), range.getExtent()))))
    return false;
  auto launch = cast<PhysicalExprAttr>(group.getLaunchExtents()[ordinal]);
  if (launch.getKind() != PhysicalExprKind::CeilDiv ||
      launch.getOperands().size() != 2 ||
      launch.getOperands()[1] != queryLaunchExpression(range.getExtent()))
    return false;
  auto bound = queryPhysicalExpressionRange(
      cast<PhysicalExprAttr>(launch.getOperands()[0]), kernel);
  // 0 <= program < ceil(stop/width), stop >= 0 proves program*width <=
  // stop-1 in every nonempty launch. Together with the aligned-window lemma,
  // whole represented intervals are disjoint without signed modular wrap.
  return bound && !bound->smin().isNegative();
}

using OwnerCoordinates = SmallVector<std::pair<int64_t, Value>>;

FailureOr<OwnerCoordinates> programCoordinates(StoreOp store, scf::ForOp loop,
                                               func::FuncOp kernel) {
  auto group = loop->getParentOfType<ExecutionGroupOp>();
  if (!group) return failure();
  OwnerCoordinates result;
  llvm::SmallDenseSet<int64_t> axes;
  IndexRelations relations;
  for (auto [ordinal, coordinate] : llvm::enumerate(group.getCoordinates())) {
    if (relations.constant(group.getExtents()[ordinal]) == 1) continue;
    std::optional<std::pair<int64_t, Value>> selected;
    for (auto [axis, value] : llvm::zip(store.getSourceAxes(), store.getCoordinates())) {
      if (loop.isDefinedOutsideOfLoop(value) &&
          ownsCoordinate(value, group, ordinal, kernel)) {
        selected = std::pair<int64_t, Value>{axis, coordinateValues(value)};
        break;
      }
    }
    if (!selected || !axes.insert(selected->first).second) return failure();
    result.push_back(*selected);
  }
  return result;
}

bool preservesProgramCoordinates(AccessOpInterface access,
                                 ArrayRef<std::pair<int64_t, Value>> owner) {
  for (auto [axis, coordinate] : owner) {
    bool found = false;
    for (auto [accessAxis, value] : llvm::zip(access.getAccessSourceAxes(), access.getAccessCoordinates()))
      if (accessAxis == axis && samePhysicalScalarExpression(
              coordinateValues(value), coordinate)) found = true;
    if (!found) return false;
  }
  return true;
}

} // namespace

static FailureOr<IndependentIterationAccesses>
independentIterationAccesses(scf::ForOp loop, ScanOp countPrefix) {
  auto kernel = loop->getParentOfType<func::FuncOp>();
  IndexRelations relations;
  Type inductionType = loop.getInductionVar().getType();
  auto integer = dyn_cast<IntegerType>(inductionType);
  bool signedI32 = integer && !integer.isUnsigned() && integer.getWidth() == 32;
  if (!kernel || loop.getNumResults() || relations.constant(loop.getStep()) != 1 ||
      (!inductionType.isIndex() && !signedI32)) return failure();
  auto space = kernel->getAttrOfType<ArrayAttr>(programSpaceAttr);
  bool singleInstance = space && llvm::all_of(space, [](Attribute attribute) {
    auto extent = cast<PhysicalExprAttr>(attribute);
    return extent.getKind() == PhysicalExprKind::Constant && extent.getValue() == 1;
  });
  IndependentIterationAccesses result;
  auto prefix = CountPrefixProof::query(countPrefix, loop);
  int64_t countLimit = prefix ? prefix->maximum : std::numeric_limits<int64_t>::max();
  bool usesCountPrefix = false;
  struct WrittenSlice { unsigned axis; Value coordinate; bool countPrefix; };
  llvm::DenseMap<Value, WrittenSlice> writtenAxes;
  llvm::DenseMap<Value, OwnerCoordinates> owners;
  llvm::DenseMap<Value, bool> varying;
  bool independent = true;
  loop.walk([&](Operation *operation) {
    if (!isa<LoadOp, StoreOp, scf::ForOp, scf::IfOp, scf::YieldOp>(operation) &&
        !isMemoryEffectFree(operation)) independent = false;
  });
  loop.walk([&](StoreOp store) {
    auto buffer = store.getResource().getDefiningOp<BufferOp>();
    auto type = dyn_cast<BufferType>(store.getResource().getType());
    auto view = dyn_cast<ViewType>(store.getResource().getType());
    auto group = buffer ? dyn_cast<ExecutionGroupOp>(buffer->getParentOp())
                        : ExecutionGroupOp{};
    bool privateBuffer = group && group->getBlock() == &kernel.front() &&
        buffer->getBlock() == &group.getBody().front() && type &&
        type.getScope().getValue() == BufferScope::ProgramPrivate;
    if (!privateBuffer && !view) { independent = false; return; }
    if (view && !llvm::is_contained(result.guardedViews, store.getResource()))
      result.guardedViews.push_back(store.getResource());
    if (view && !singleInstance) {
      auto owner = programCoordinates(store, loop, kernel);
      if (failed(owner)) { independent = false; return; }
      auto [previous, inserted] = owners.try_emplace(store.getResource(), *owner);
      if (!inserted && !preservesProgramCoordinates(
              cast<AccessOpInterface>(store.getOperation()), previous->second)) {
        independent = false;
        return;
      }
    }
    std::optional<unsigned> selected;
    Value selectedCoordinate;
    bool selectedCount = false;
    for (auto [axis, coordinate] : llvm::zip(store.getSourceAxes(), store.getCoordinates())) {
      bool unit = isUnitIterationCoordinate(coordinate, loop);
      int64_t limit = countLimit;
      bool count = !unit && prefix && prefix->coordinate(
          coordinate, cast<AccessOpInterface>(store.getOperation()), limit);
      if (!selected && (unit || count)) {
        selected = axis;
        selectedCoordinate = coordinate;
        selectedCount = count;
        if (count) { countLimit = limit; usesCountPrefix = true; }
      } else if (variesWithIteration(coordinate, loop, varying)) independent = false;
    }
    if (!selected) { independent = false; return; }
    auto [previous, inserted] = writtenAxes.try_emplace(
        store.getResource(), WrittenSlice{*selected, selectedCoordinate, selectedCount});
    independent &= inserted ||
        (previous->second.axis == *selected && previous->second.countPrefix == selectedCount &&
         samePhysicalScalarExpression(
            previous->second.coordinate, selectedCoordinate));
  });
  if (!independent || writtenAxes.empty()) return failure();

  SmallVector<Value> accessedViews(result.guardedViews.begin(), result.guardedViews.end());
  loop.walk([&](LoadOp load) {
    if (isa<ViewType>(load.getResource().getType()) &&
        !llvm::is_contained(accessedViews, load.getResource()))
      accessedViews.push_back(load.getResource());
    if (auto owner = owners.find(load.getResource()); owner != owners.end())
      independent &= preservesProgramCoordinates(
          cast<AccessOpInterface>(load.getOperation()), owner->second);
    auto written = writtenAxes.find(load.getResource());
    if (written == writtenAxes.end()) return;
    bool independentRead = false;
    for (auto [axis, coordinate] : llvm::zip(load.getSourceAxes(), load.getCoordinates()))
      if (axis == written->second.axis) {
        bool same = samePhysicalScalarExpression(coordinate, written->second.coordinate);
        if (written->second.countPrefix) {
          // Inactive members can repeat the previous prefix address. A read of
          // that address is independent only under the same positive-member
          // predicate, or outside the complete possible write domain [0,N).
          independentRead = (same && prefix->active(cast<AccessOpInterface>(load.getOperation()))) ||
                            isOutsideIterationRange(coordinate, loop);
        } else {
          independentRead = same || (written->second.coordinate == loop.getInductionVar() &&
                                    isOutsideIterationRange(coordinate, loop));
        }
      }
    independent &= independentRead;
  });
  if (!independent) return failure();
  for (Value view : accessedViews) {
    auto argument = dyn_cast<BlockArgument>(view);
    if (!argument || argument.getOwner() != &kernel.front()) return failure();
  }
  for (Value written : result.guardedViews)
    for (Value other : accessedViews) {
      if (written == other) continue;
      auto pair = std::pair{written, other};
      if (cast<BlockArgument>(written).getArgNumber() > cast<BlockArgument>(other).getArgNumber())
        std::swap(pair.first, pair.second);
      if (!llvm::is_contained(result.disjointViews, pair)) result.disjointViews.push_back(pair);
    }
  if (usesCountPrefix) result.countUpperBounds.emplace_back(loop.getUpperBound(), countLimit);
  return result;
}

FailureOr<IndependentIterationAccesses>
queryIndependentIterationAccesses(scf::ForOp loop) {
  return independentIterationAccesses(loop, {});
}

FailureOr<IndependentIterationAccesses>
queryIndependentIterationAccesses(scf::ForOp loop, ScanOp countPrefix) {
  return independentIterationAccesses(loop, countPrefix);
}

} // namespace intent::gpu
