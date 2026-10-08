#include "Construction.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <functional>

using namespace mlir;

namespace intent::kir_to_gpu {

LogicalResult ScalarRegionLowering::lower(intent::IfOp ifOperation) {
  Operation *operation = ifOperation.getOperation();
  Location location = ifOperation.getLoc();
  FailureOr<Value> condition = get(ifOperation.getCondition());
  if (failed(condition))
    return failure();
  SmallVector<Type> resultTypes;
  for (auto [index, type] :
       llvm::enumerate(ifOperation.getResultTypes())) {
    FailureOr<Type> converted =
        convertDataType(canonicalAnalysis, type, operation, std::nullopt, /*owner=*/1, index);
    if (failed(converted))
      return ifOperation.emitOpError("if carries an unphysical value");
    resultTypes.push_back(*converted);
  }
  auto target = builder.create<scf::IfOp>(location, resultTypes, *condition,
                                           /*withElseRegion=*/true);
  auto lowerBranch = [&](Region &targetRegion, Region &sourceRegion) {
    Block &targetBlock = targetRegion.front();
    if (!targetBlock.empty() && isa<scf::YieldOp>(targetBlock.back()))
      targetBlock.back().erase();
    OpBuilder nested(&targetBlock, targetBlock.begin());
    ScalarRegionLowering child(nested, values, views, abiDimensions, parameters,
                               canonicalAnalysis, physicalKernel);
    FailureOr<SmallVector<Value>> yielded =
        child.lowerBlock(sourceRegion.front());
    if (failed(yielded) || yielded->size() != resultTypes.size())
      return failure();
    for (auto [index, resultType] : llvm::enumerate(resultTypes)) {
      FailureOr<Value> projected = projectPositionalValue(
          nested, location, (*yielded)[index], resultType);
      if (failed(projected))
        return failure();
      (*yielded)[index] = *projected;
    }
    nested.create<scf::YieldOp>(location, *yielded);
    return success();
  };
  if (failed(lowerBranch(target.getThenRegion(),
                         ifOperation.getThenRegion())) ||
      failed(lowerBranch(target.getElseRegion(),
                         ifOperation.getElseRegion())))
    return failure();
  mapResults(operation, target);
  return success();
}

LogicalResult ScalarRegionLowering::lowerLoop(Operation *operation) {
  Location location = operation->getLoc();
  // Parallel regions remaining inside an execution group may be serialized.
  // Keep their enclosing ordered control and resource environment intact.
  auto forOperation = dyn_cast<intent::ForOp>(operation);
  auto parallel = dyn_cast<intent::ParallelOp>(operation);
  Value sourceDomain = forOperation ? forOperation.getSource() : parallel.getSource();
  Region &sourceRegion = forOperation ? forOperation.getBody() : parallel.getBody();
  ValueRange inductionVariables = forOperation ? ValueRange(forOperation.getInductionVars())
                                              : ValueRange(sourceRegion.front().getArguments());
  ValueRange iterArguments = forOperation ? ValueRange(forOperation.getRegionIterArgs()) : ValueRange{};
  SmallVector<IterationAxis> axes;
  if (failed(collectIterationAxes(sourceDomain, axes)) ||
      axes.empty())
    return operation->emitOpError(
        "iteration source has no exact domain/subregion relation");
  SmallVector<Value> lowers, uppers, steps;
  for (IterationAxis axis : axes) {
    FailureOr<Value> lower = get(axis.start);
    FailureOr<Value> upper = get(axis.stop);
    FailureOr<Value> prototype = get(axis.coordinatePrototype);
    if (failed(lower) || failed(upper) || failed(prototype))
      return operation->emitOpError(
          "physical loop bounds are unavailable");
    Type coordinateType = (*prototype).getType();
    auto alignBound = [&](Value value) -> FailureOr<Value> {
      if (value.getType() == coordinateType)
        return value;
      if (!isa<IntegerType, IndexType>(value.getType()) ||
          !isa<IntegerType, IndexType>(coordinateType))
        return failure();
      return Value(builder.create<gpu::CastOp>(
          location, coordinateType, value));
    };
    lower = alignBound(*lower);
    upper = alignBound(*upper);
    if (failed(lower) || failed(upper))
      return operation->emitOpError(
          "physical subregion bounds cannot adopt their source coordinate type");
    Value step;
    if (axis.step) {
      FailureOr<Value> lowered = get(axis.step);
      if (failed(lowered))
        return operation->emitOpError(
            "physical loop step is unavailable");
      FailureOr<Value> aligned = alignBound(*lowered);
      if (failed(aligned))
        return operation->emitOpError(
            "physical loop step cannot adopt its source coordinate type");
      step = *aligned;
    } else if ((*lower).getType().isIndex()) {
      step = builder.create<arith::ConstantIndexOp>(location, 1);
    } else if (auto integer = dyn_cast<IntegerType>((*lower).getType())) {
      step = builder.create<arith::ConstantOp>(
          location, integer, builder.getIntegerAttr(integer, 1));
    }
    if (!step || (*lower).getType() != (*upper).getType() ||
        (*lower).getType() != step.getType())
      return operation->emitOpError(
          "physical loop bounds must share one scalar type");
    lowers.push_back(*lower);
    uppers.push_back(*upper);
    steps.push_back(step);
  }
  SmallVector<Value> initial;
  ValueRange initArgs = forOperation ? ValueRange(forOperation.getInitArgs()) : ValueRange{};
  for (Value input : initArgs) {
    FailureOr<Value> lowered = get(input);
    if (failed(lowered))
      return failure();
    initial.push_back(*lowered);
  }
  bool nestedFailed = false;
  std::function<SmallVector<Value>(OpBuilder &, unsigned,
                                   SmallVector<Value>, ValueRange)>
      lowerAxis;
  lowerAxis = [&](OpBuilder &nested, unsigned axis,
                  SmallVector<Value> coordinates,
                  ValueRange carries) -> SmallVector<Value> {
    if (axis == axes.size()) {
      auto childValues = values;
      Block &source = sourceRegion.front();
      for (auto [argument, coordinate] :
           llvm::zip(inductionVariables, coordinates))
        childValues[argument] = coordinate;
      for (auto [argument, carry] : llvm::zip(iterArguments, carries))
        childValues[argument] = carry;
      ScalarRegionLowering child(nested, std::move(childValues), views,
                                 abiDimensions, parameters, canonicalAnalysis,
                                 physicalKernel);
      FailureOr<SmallVector<Value>> yielded = child.lowerBlock(source);
      if (failed(yielded)) {
        nestedFailed = true;
        return {};
      }
      return *yielded;
    }
    SmallVector<Value> loopResults;
    auto loop = nested.create<scf::ForOp>(
        location, lowers[axis], uppers[axis], steps[axis], carries,
        [](OpBuilder &bodyBuilder, Location location, Value,
            ValueRange innerCarries) {
          bodyBuilder.create<scf::YieldOp>(location, innerCarries);
        });
    OpBuilder bodyBuilder(loop.getBody()->getTerminator());
    SmallVector<Value> nextCoordinates(coordinates);
    nextCoordinates.push_back(loop.getInductionVar());
    SmallVector<Value> yielded = lowerAxis(
        bodyBuilder, axis + 1, std::move(nextCoordinates),
        loop.getRegionIterArgs());
    if (nestedFailed) {
      loop.erase();
      return {};
    }
    loop.getBody()->getTerminator()->setOperands(yielded);
    if (isa<intent::ParallelOp>(operation))
      loop->setAttr(gpu::independentIterationAttr, nested.getUnitAttr());
    auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
    if (!yield) {
      nestedFailed = true;
      loop.erase();
      return {};
    }
    for (auto [index, item] : llvm::enumerate(llvm::zip(
             loop.getInitArgs(), yield.getOperands()))) {
      Value initial = std::get<0>(item);
      Value yielded = std::get<1>(item);
      if (initial.getType() == yielded.getType())
        continue;
      // The loop-carried value keeps the physical relation established by
      // its initializer, recursively for record-valued algorithm state. A
      // body expression may temporarily acquire result-local axis
      // identities, but those identities must not replace the source
      // provenance carried across iterations.
      OpBuilder before(yield);
      FailureOr<Value> aligned = projectPositionalValue(
          before, location, yielded, initial.getType());
      if (failed(aligned)) {
        operation->emitOpError(
            "loop update cannot preserve its physical carry relation")
            << "; initial=" << initial.getType()
            << "; yielded=" << yielded.getType();
        nestedFailed = true;
        break;
      }
      yield->setOperand(index, *aligned);
    }
    if (nestedFailed) {
      loop.erase();
      return {};
    }
    loopResults.append(loop.getResults().begin(), loop.getResults().end());
    return loopResults;
  };
  SmallVector<Value> resultValues =
      lowerAxis(builder, 0, SmallVector<Value>{}, initial);
  if (nestedFailed)
    return failure();
  for (auto [source, target] :
       llvm::zip(operation->getResults(), resultValues))
    values[source] = target;
  return success();
}

LogicalResult ScalarRegionLowering::lower(intent::WhileOp whileOperation) {
  Operation *operation = whileOperation.getOperation();
  Location location = whileOperation.getLoc();
  SmallVector<Value> initial;
  SmallVector<Type> resultTypes;
  for (Value input : whileOperation.getInitArgs()) {
    FailureOr<Value> lowered = get(input);
    if (failed(lowered))
      return failure();
    initial.push_back(*lowered);
    resultTypes.push_back((*lowered).getType());
  }
  auto target = builder.create<scf::WhileOp>(location, resultTypes, initial);
  {
    OpBuilder::InsertionGuard guard(builder);
    Block *before = builder.createBlock(
        &target.getBefore(), target.getBefore().end(), resultTypes,
        SmallVector<Location>(resultTypes.size(), location));
    auto childValues = values;
    for (auto [source, targetArgument] : llvm::zip(
             whileOperation.getBeforeArguments(),
             before->getArguments()))
      childValues[source] = targetArgument;
    builder.setInsertionPointToStart(before);
    ScalarRegionLowering child(builder, std::move(childValues), views,
                               abiDimensions, parameters, canonicalAnalysis,
                               physicalKernel);
    Block &source = whileOperation.getBefore().front();
    for (Operation &nested : source.without_terminator())
      if (failed(child.lower(&nested)))
        return failure();
    auto condition = cast<intent::ConditionOp>(source.getTerminator());
    FailureOr<Value> predicate = child.get(condition.getCondition());
    SmallVector<Value> forwarded;
    for (Value value : condition.getArgs()) {
      FailureOr<Value> lowered = child.get(value);
      if (failed(lowered))
        return failure();
      forwarded.push_back(*lowered);
    }
    if (failed(predicate))
      return failure();
    builder.create<scf::ConditionOp>(location, *predicate, forwarded);
  }
  {
    OpBuilder::InsertionGuard guard(builder);
    Block *after = builder.createBlock(
        &target.getAfter(), target.getAfter().end(), resultTypes,
        SmallVector<Location>(resultTypes.size(), location));
    auto childValues = values;
    for (auto [source, targetArgument] : llvm::zip(
             whileOperation.getAfterArguments(),
             after->getArguments()))
      childValues[source] = targetArgument;
    builder.setInsertionPointToStart(after);
    ScalarRegionLowering child(builder, std::move(childValues), views,
                               abiDimensions, parameters, canonicalAnalysis,
                               physicalKernel);
    FailureOr<SmallVector<Value>> yielded =
        child.lowerBlock(whileOperation.getAfter().front());
    if (failed(yielded))
      return failure();
    builder.create<scf::YieldOp>(location, *yielded);
  }
  mapResults(operation, target);
  return success();
}

} // namespace intent::kir_to_gpu
