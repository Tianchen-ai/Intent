#include "RegionPredicates.h"
#include "RegionSummary.h"
#include "Intent/Analysis/RegionSemantics.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;

namespace intent::gpu::region {

namespace {

Value stripHelperForwarding(Value value) {
  while (Operation *definition = value.getDefiningOp()) {
    if (auto broadcast = dyn_cast<BroadcastOp>(definition)) {
      value = broadcast.getValue();
      continue;
    }
    if (auto cast = dyn_cast<CastOp>(definition)) {
      value = cast.getValue();
      continue;
    }
    if (auto reshape = dyn_cast<ReshapeOp>(definition)) {
      value = reshape.getValue();
      continue;
    }
    if (auto transpose = dyn_cast<TransposeOp>(definition)) {
      value = transpose.getValue();
      continue;
    }
    if (auto splat = dyn_cast<SplatOp>(definition)) {
      value = splat.getValue();
      continue;
    }
    if (auto select = dyn_cast<SelectOp>(definition)) {
      std::optional<bool> condition =
          booleanConstant(select.getCondition());
      if (!condition)
        return {};
      value = *condition ? select.getTrueValue() : select.getFalseValue();
      continue;
    }
    return value;
  }
  return value;
}

BlockArgument rootHelperArgument(Value value) {
  return dyn_cast_or_null<BlockArgument>(stripHelperForwarding(value));
}

struct HelperCoordinateExpression {
  BlockArgument coordinate;
  Value scalar;
  bool subtractScalar = false;
};

bool isHelperScalar(Value value) {
  value = stripHelperForwarding(value);
  if (!value)
    return false;
  if (isa<BlockArgument>(value))
    return value.getType().isIntOrIndex();
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  return constant && isa<IntegerAttr>(constant.getValue());
}

FailureOr<HelperCoordinateExpression>
helperCoordinateExpression(Value value) {
  value = stripHelperForwarding(value);
  if (!value)
    return failure();
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    if (!isa<FragmentType>(argument.getType()))
      return failure();
    return HelperCoordinateExpression{argument, Value(), false};
  }
  auto binary = value.getDefiningOp<BinaryOp>();
  if (!binary)
    return failure();
  BinaryOperator kind = binary.getOperatorKind();
  if (kind != BinaryOperator::Add && kind != BinaryOperator::Subtract)
    return failure();
  BlockArgument lhs = rootHelperArgument(binary.getLhs());
  if (lhs && isa<FragmentType>(lhs.getType()) &&
      isHelperScalar(binary.getRhs()))
    return HelperCoordinateExpression{lhs,
                                      stripHelperForwarding(binary.getRhs()),
                                      kind == BinaryOperator::Subtract};
  if (kind == BinaryOperator::Add) {
    BlockArgument rhs = rootHelperArgument(binary.getRhs());
    if (rhs && isa<FragmentType>(rhs.getType()) &&
        isHelperScalar(binary.getLhs()))
      return HelperCoordinateExpression{rhs,
                                        stripHelperForwarding(binary.getLhs()),
                                        false};
  }
  return failure();
}

bool equalWhenPredicateIsFalse(Value summary, Value identity,
                               Value falsePredicate) {
  UniformBindings bindings;
  bindings[falsePredicate] = uniformZero(uniformElementType(falsePredicate.getType()));
  UniformValueAnalysis facts(describeUniformValue);
  return equalUniformConstants(facts.evaluate(summary, bindings),
                               facts.evaluate(identity, bindings));
}

} // namespace

// Only synthetic padding receives source fill facts. Logical traversal pruning
// separately calls equalWhenPredicateIsFalse without these assumptions.
Value physicalTailMembershipPredicate(RegionFoldOp fold, ValueRange identities,
                                 ParameterRefAttr segment, ArrayRef<SourcePlan> plans) {
  auto yield = dyn_cast<YieldOp>(fold.getSummarize().front().getTerminator());
  if (!yield || yield.getValues().size() != identities.size())
    return {};
  Value candidate;
  UniformValueAnalysis facts(describeUniformValue);
  UniformBindings tailValues;
  for (auto [argument, plan] : llvm::zip(fold.getSummarize().front().getArguments().take_front(fold.getSources().size()), plans))
    if (plan.tailConstant) tailValues[argument] = plan.tailConstant;
  auto consider = [&](Value value) {
    auto fragment = dyn_cast<FragmentType>(value.getType());
    bool carriesSegment = fragment && llvm::any_of(
        fragment.getShape(), [&](Attribute extent) {
          auto expression = dyn_cast<PhysicalExprAttr>(extent);
          return expression &&
                 expression.getKind() == PhysicalExprKind::Parameter &&
                 expression.getParameterReference() == segment;
        });
    if (!fragment || !fragment.getElementType().isInteger(1) ||
        !carriesSegment)
      return;
    bool identityOnly = true;
    UniformBindings assumptions = tailValues;
    assumptions[value] = uniformZero(uniformElementType(value.getType()));
    for (auto [summary, identity] : llvm::zip(yield.getValues(), identities))
      identityOnly &= equalUniformConstants(facts.evaluate(summary, assumptions),
                                           facts.evaluate(identity));
    if (identityOnly)
      candidate = value;
  };
  // A source predicate can feed several independent projections in the helper.
  // Prove the complete summary from their shared input, not one projection.
  for (BlockArgument argument : fold.getSummarize().front().getArguments()
                                    .take_front(fold.getSources().size()))
    consider(argument);
  for (Operation &operation : fold.getSummarize().front().without_terminator())
    for (Value result : operation.getResults())
      consider(result);
  return candidate;
}

bool scanTailIsIdentity(RegionScanOp scan, ArrayRef<SourcePlan> plans,
                        ValueRange identities) {
  auto yield = dyn_cast<YieldOp>(scan.getSummarize().front().getTerminator());
  if (!yield || yield.getValues().size() != identities.size())
    return false;
  UniformBindings bindings;
  for (auto [index, argument] : llvm::enumerate(
           scan.getSummarize().front().getArguments().take_front(
               scan.getSources().size()))) {
    if (plans[index].tailConstant) bindings[argument] = plans[index].tailConstant;
  }
  UniformValueAnalysis facts(describeUniformValue);
  for (auto [summary, identity] : llvm::zip(yield.getValues(), identities))
    if (!equalUniformConstants(facts.evaluate(summary, bindings),
                               facts.evaluate(identity)))
      return false;
  return true;
}

FailureOr<PredicatePartition>
predicatePartition(OpBuilder &builder, RegionFoldOp fold,
                   ArrayRef<SourcePlan> plans, MakeRangeOp master,
                   Value masterExtent, ValueRange identities, Value segment) {
  auto yield = dyn_cast<YieldOp>(fold.getSummarize().front().getTerminator());
  if (!yield || yield.getValues().size() != identities.size())
    return failure();
  auto kernel = fold->getParentOfType<func::FuncOp>();
  if (!kernel)
    return failure();
  PhysicalProgramAnalysis analysis(kernel);
  auto uniqueRange = [&](Value value) -> MakeRangeOp {
    PhysicalRangeFact fact = analysis.sourceRanges(value);
    FailureOr<MakeRangeOp> range = queryExactLogicalRange(fact);
    return succeeded(range) ? *range : MakeRangeOp();
  };
  auto structured = cast<StructuredOpInterface>(fold.getOperation());
  ValueRange sourceArguments = structured.getSummarizeSources();
  ValueRange captureArguments = structured.getSummarizeCaptures();
  ValueRange captures = fold.getCaptures();
  auto boundRanges = [&](BlockArgument source,
                         BlockArgument capture)
      -> std::optional<std::pair<MakeRangeOp, MakeRangeOp>> {
    auto sourcePosition = llvm::find(sourceArguments, source);
    auto capturePosition = llvm::find(captureArguments, capture);
    if (!source || !capture || sourcePosition == sourceArguments.end() ||
        capturePosition == captureArguments.end())
      return std::nullopt;
    unsigned sourceIndex = std::distance(sourceArguments.begin(), sourcePosition);
    unsigned captureIndex = std::distance(captureArguments.begin(), capturePosition);
    if (sourceIndex >= plans.size()) return std::nullopt;
    Value sourceCoordinate = stripHelperForwarding(plans[sourceIndex].source);
    Value captureCoordinate = stripHelperForwarding(captures[captureIndex]);
    if (!sourceCoordinate.getDefiningOp<MakeRangeOp>() ||
        !captureCoordinate.getDefiningOp<MakeRangeOp>())
      return std::nullopt;
    MakeRangeOp captureRange = uniqueRange(captures[captureIndex]);
    MakeRangeOp comparedSource =
        uniqueRange(plans[sourceIndex].source);
    auto comparedType = comparedSource
                            ? dyn_cast<FragmentType>(
                                  comparedSource.getResult().getType())
                            : FragmentType();
    auto masterType = dyn_cast<FragmentType>(master.getResult().getType());
    if (!captureRange || !comparedSource || !isUnitStepRange(captureRange) ||
        !isUnitStepRange(comparedSource) || !comparedType || !masterType ||
        comparedType.getShape().size() != 1 ||
        masterType.getShape().size() != 1 ||
        comparedType.getShape()[0] != masterType.getShape()[0])
      return std::nullopt;
    return std::make_pair(captureRange, comparedSource);
  };
  auto materializeHelperScalar = [&](Value helperScalar) -> FailureOr<Value> {
    helperScalar = stripHelperForwarding(helperScalar);
    if (!helperScalar)
      return failure();
    Value scalar;
    if (auto argument = dyn_cast<BlockArgument>(helperScalar)) {
      auto position = llvm::find(captureArguments, argument);
      if (position == captureArguments.end()) return failure();
      unsigned captureIndex = std::distance(captureArguments.begin(), position);
      scalar = captures[captureIndex];
    } else if (auto constant =
                   helperScalar.getDefiningOp<arith::ConstantOp>()) {
      auto integer = dyn_cast<IntegerAttr>(constant.getValue());
      if (!integer)
        return failure();
      scalar = builder.create<arith::ConstantIndexOp>(fold.getLoc(),
                                                       integer.getInt());
    } else {
      return failure();
    }
    if (!scalar.getType().isIntOrIndex())
      return failure();
    if (!isa<IndexType>(scalar.getType()))
      scalar = builder.create<CastOp>(fold.getLoc(), builder.getIndexType(),
                                      scalar);
    return scalar;
  };

  Location location = fold.getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
  Value one = builder.create<arith::ConstantIndexOp>(location, 1);
  auto expression = [&](UniformKind kind, Value lhs, Value rhs) -> Value {
    BinaryOperator operation;
    switch (kind) {
    case UniformKind::Add: operation = BinaryOperator::Add; break;
    case UniformKind::Subtract: operation = BinaryOperator::Subtract; break;
    case UniformKind::Minimum: operation = BinaryOperator::Minimum; break;
    case UniformKind::Maximum: operation = BinaryOperator::Maximum; break;
    default: llvm_unreachable("unexpected coordinate expression");
    }
    return builder.create<BinaryOp>(location, builder.getIndexType(), lhs, rhs, operation);
  };
  auto intervals = [&](UniformPredicate predicate, MakeRangeOp source, Value captureBegin, Value captureEnd) {
    Value sourceEnd = expression(UniformKind::Add, source.getStart(), masterExtent);
    return *partitionCoordinatePredicate(predicate, {source.getStart(), sourceEnd},
        {captureBegin, captureEnd}, zero, one, expression);
  };
  Value effectiveStart = zero;
  Value effectiveStop = masterExtent;
  Value allTrueStart = zero;
  Value allTrueStop = masterExtent;
  SmallVector<Value> allTruePredicates;
  bool firstMemberIsActive = false;
  bool hasLowerBound = false;
  unsigned upperBoundCount = 0;
  bool foundBound = false;
  for (CompareOp compare : fold.getSummarize().front().getOps<CompareOp>()) {
    BlockArgument lhs = rootHelperArgument(compare.getLhs());
    BlockArgument rhs = rootHelperArgument(compare.getRhs());
    bool identityOnly = true;
    for (auto [summary, identity] : llvm::zip(yield.getValues(), identities))
      identityOnly &=
          equalWhenPredicateIsFalse(summary, identity, compare.getResult());
    BlockArgument upperSource;
    BlockArgument upperCapture;
    bool strictUpper = false;
    switch (compare.getPredicate()) {
    case ComparePredicate::Le:
    case ComparePredicate::Lt:
      upperSource = lhs;
      upperCapture = rhs;
      strictUpper = compare.getPredicate() == ComparePredicate::Lt;
      break;
    case ComparePredicate::Ge:
    case ComparePredicate::Gt:
      upperSource = rhs;
      upperCapture = lhs;
      strictUpper = compare.getPredicate() == ComparePredicate::Gt;
      break;
    default:
      break;
    }
    if (auto ranges = boundRanges(upperSource, upperCapture)) {
      MakeRangeOp captureRange = ranges->first;
      MakeRangeOp comparedSource = ranges->second;
      Value captureEnd = expression(UniformKind::Add, captureRange.getStart(), captureRange.getExtent());
      auto bounds = intervals(strictUpper ? UniformPredicate::Less : UniformPredicate::LessEqual,
          comparedSource, captureRange.getStart(), captureEnd);
      if (identityOnly)
        effectiveStop = expression(UniformKind::Minimum, effectiveStop, bounds.possibleEnd);
      {
        Value wholeSegments = builder.create<BinaryOp>(
            location, builder.getIndexType(), bounds.allTrueEnd, segment,
            BinaryOperator::FloorDivide);
        Value candidateAllTrueStop = builder.create<BinaryOp>(
            location, builder.getIndexType(), wholeSegments, segment,
            BinaryOperator::Multiply);
        allTrueStop = builder.create<BinaryOp>(
            location, builder.getIndexType(), allTrueStop, candidateAllTrueStop,
            BinaryOperator::Minimum);
        allTruePredicates.push_back(compare.getResult());
        if (upperBoundCount++ == 0) {
          firstMemberIsActive =
              !strictUpper &&
              samePhysicalScalarExpression(master.getStart(),
                                           master.getLogicalStart()) &&
              samePhysicalScalarExpression(captureRange.getLogicalStart(),
                                           master.getStart()) &&
              samePhysicalScalarExpression(comparedSource.getStart(),
                                           master.getStart());
        }
      }
      foundBound = true;
    }

    BlockArgument lowerSource;
    FailureOr<HelperCoordinateExpression> lowerCapture = failure();
    bool strictLower = false;
    switch (compare.getPredicate()) {
    case ComparePredicate::Ge:
    case ComparePredicate::Gt:
      lowerSource = lhs;
      lowerCapture = helperCoordinateExpression(compare.getRhs());
      strictLower = compare.getPredicate() == ComparePredicate::Gt;
      break;
    case ComparePredicate::Le:
    case ComparePredicate::Lt:
      lowerSource = rhs;
      lowerCapture = helperCoordinateExpression(compare.getLhs());
      strictLower = compare.getPredicate() == ComparePredicate::Lt;
      break;
    default:
      break;
    }
    if (failed(lowerCapture))
      continue;
    auto ranges = boundRanges(lowerSource, lowerCapture->coordinate);
    if (!ranges)
      continue;
    MakeRangeOp captureRange = ranges->first;
    MakeRangeOp comparedSource = ranges->second;
    Value lower = captureRange.getStart();
    if (lowerCapture->scalar) {
      FailureOr<Value> scalar =
          materializeHelperScalar(lowerCapture->scalar);
      if (failed(scalar))
        continue;
      lower = builder.create<BinaryOp>(
          location, builder.getIndexType(), lower, *scalar,
          lowerCapture->subtractScalar ? BinaryOperator::Subtract
                                       : BinaryOperator::Add);
    }
    Value captureEnd = expression(UniformKind::Add, lower, captureRange.getExtent());
    auto bounds = intervals(strictLower ? UniformPredicate::Greater : UniformPredicate::GreaterEqual,
        comparedSource, lower, captureEnd);
    Value wholeSegments = builder.create<BinaryOp>(
        location, builder.getIndexType(), bounds.possibleBegin, segment,
        BinaryOperator::FloorDivide);
    Value alignedLower = builder.create<BinaryOp>(
        location, builder.getIndexType(), wholeSegments, segment,
        BinaryOperator::Multiply);
    if (identityOnly)
      effectiveStart = expression(UniformKind::Maximum, effectiveStart, alignedLower);
    Value adjustment = builder.create<BinaryOp>(
        location, builder.getIndexType(), segment, one,
        BinaryOperator::Subtract);
    Value roundedLastLower = builder.create<BinaryOp>(
        location, builder.getIndexType(), bounds.allTrueBegin, adjustment,
        BinaryOperator::Add);
    Value firstWholeSegment = builder.create<BinaryOp>(
        location, builder.getIndexType(), roundedLastLower, segment,
        BinaryOperator::FloorDivide);
    Value candidateAllTrueStart = builder.create<BinaryOp>(
        location, builder.getIndexType(), firstWholeSegment, segment,
        BinaryOperator::Multiply);
    allTrueStart = builder.create<BinaryOp>(
        location, builder.getIndexType(), allTrueStart, candidateAllTrueStart,
        BinaryOperator::Maximum);
    allTruePredicates.push_back(compare.getResult());
    hasLowerBound = true;
    foundBound = true;
  }
  if (!foundBound)
    return failure();
  return PredicatePartition{allTrueStart, allTrueStop, effectiveStart,
                            effectiveStop,
                            std::move(allTruePredicates), firstMemberIsActive,
                            upperBoundCount == 1 && !hasLowerBound};
}

} // namespace intent::gpu::region
