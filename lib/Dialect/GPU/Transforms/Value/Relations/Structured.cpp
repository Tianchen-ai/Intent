#include "Intent/Dialect/GPU/Analysis/Helpers.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/FragmentOpInterface.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Value/Helpers.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Worklist.h"

using namespace mlir;

namespace intent::gpu::value_relations {

WalkResult alignReductionResultRelation(Operation *operation,
                                        RelationWorklist &changes) {
  auto kernel = operation->getParentOfType<func::FuncOp>();
  SmallVector<Value> sources;
  SmallVector<Value> results;
  llvm::SmallDenseSet<int64_t> reducedAxes;
  bool scan = false;
  if (auto reduce = dyn_cast<ReduceOp>(operation)) {
    llvm::append_range(sources, reduce.getSources());
    results.append(reduce.getResults().begin(), reduce.getResults().end());
    reducedAxes.insert(reduce.getAxes().begin(), reduce.getAxes().end());
  } else if (auto currentScan = dyn_cast<ScanOp>(operation)) {
    llvm::append_range(sources, currentScan.getSources());
    results.append(currentScan.getResults().begin(),
                   currentScan.getResults().end());
    scan = true;
  } else {
    return WalkResult::advance();
  }
  if (sources.size() != results.size())
    return WalkResult::interrupt();
  for (auto [index, source] : llvm::enumerate(sources)) {
    auto sourceType = dyn_cast<FragmentType>(source.getType());
    if (!sourceType || llvm::all_of(sources, [&](Value other) {
          auto type = dyn_cast<FragmentType>(other.getType());
          return type && type.getShape() == sourceType.getShape();
        }))
      continue;
    SmallVector<Value> related;
    llvm::copy_if(sources, std::back_inserter(related), [&](Value other) {
      auto type = dyn_cast<FragmentType>(other.getType());
      return type && type.getShape().size() == sourceType.getShape().size() &&
             queryAxisProjection(type, sourceType).isExact();
    });
    FailureOr<FragmentType> refined =
        queryValueSchema(kernel, sourceType, related);
    if (failed(refined)) {
      operation->emitOpError(
          "tuple reduction sources have no common physical extent relation");
      return WalkResult::interrupt();
    }
    auto target =
        FragmentType::get(kernel.getContext(), sourceType.getElementType(),
                          (*refined).getShape(), sourceType.getAxisMaps(),
                          sourceType.getValidity(), sourceType.getOwner());
    OpBuilder builder(operation);
    builder.setListener(&changes);
    FailureOr<Value> projected = projectPhysicalValueToSchema(
        builder, operation->getLoc(), source, target, changes.typeChanged());
    if (failed(projected)) {
      operation->emitOpError(
          "tuple reduction source cannot adopt its physical extent relation");
      return WalkResult::interrupt();
    }
    operation->setOperand(index, *projected);
    sources[index] = *projected;
  }
  for (auto [source, currentResult] : llvm::zip_equal(sources, results)) {
    if (scan) {
      changes.setType(currentResult, source.getType());
      continue;
    }
    Type element = currentResult.getType();
    if (auto fragment = dyn_cast<FragmentType>(element))
      element = fragment.getElementType();
    SmallVector<int64_t> axes(reducedAxes.begin(), reducedAxes.end());
    auto type = inferCollectiveResultType(source.getType(), axes, element);
    if (failed(type)) {
      operation->emitOpError(
          "physical reduction has no valid source/result axis schema");
      return WalkResult::interrupt();
    }
    changes.setType(currentResult, *type);
  }
  return WalkResult::advance();
}

WalkResult alignReductionIdentityRelation(Operation *operation,
                                          RelationWorklist &changes) {
  auto structured = cast<StructuredOpInterface>(operation);
  auto identities = isa<ReduceOp>(operation)
                        ? cast<ReduceOp>(operation).getIdentities()
                        : cast<ScanOp>(operation).getIdentities();
  OpBuilder builder(operation);
  builder.setListener(&changes);
  Region &combine = structured.getCombine();
  if (!llvm::hasSingleElement(combine) || combine.front().empty())
    return WalkResult::interrupt();
  Block &body = combine.front();
  auto yield = dyn_cast<YieldOp>(&body.back());
  if (!yield)
    return WalkResult::interrupt();
  SmallVector<Type> arguments(operation->getResultTypes());
  llvm::append_range(arguments, operation->getResultTypes());
  for (BlockArgument capture : structured.getCombineCaptures())
    arguments.push_back(capture.getType());
  bool signatureChanged =
      !llvm::equal(body.getArgumentTypes(), arguments) ||
      !llvm::equal(yield.getOperandTypes(), operation->getResultTypes());
  Region replacementBody;
  bool rebuild = signatureChanged && succeeded(proveLaneWiseHelper(combine));
  if (rebuild) {
    std::string reason;
    if (failed(cloneLaneWiseHelper(combine, replacementBody, arguments,
                                  operation->getResultTypes(), reason))) {
      operation->emitOpError(
          "lane-wise combine cannot adopt its selected accumulator schema: ")
          << reason;
      return WalkResult::interrupt();
    }
  }
  for (auto [index, identity] : llvm::enumerate(identities)) {
    Type target = operation->getResult(index).getType();
    auto projected = projectPhysicalValueToSchema(
        builder, operation->getLoc(), identity, target, changes.typeChanged());
    if (failed(projected)) {
      operation->emitOpError(
          "physical reduction identity cannot adopt its result relation");
      return WalkResult::interrupt();
    }
    operation->setOperand(identities.getBeginOperandIndex() + index,
                          *projected);
    if (!rebuild) {
      changes.setType(structured.getCombineLhs()[index], target);
      changes.setType(structured.getCombineRhs()[index], target);
    }
  }
  if (rebuild) {
    // The selected accumulator is the execution-domain boundary even when a
    // component is uniform and has no load/range authority of its own. Publish
    // the whole rebuilt helper while retaining the owner and formal identities
    // held by the calling transformation. Notify every removed/inserted operation
    // so the relation worklist cannot retain stale payload references.
    IRRewriter rewriter(operation->getContext());
    rewriter.setListener(&changes);
    rewriter.modifyOpInPlace(operation, [&] {
      while (!body.empty())
        rewriter.eraseOp(&body.back());
      for (auto [argument, replacement] : llvm::zip_equal(
               body.getArguments(), replacementBody.front().getArguments()))
        changes.setType(argument, replacement.getType());
      rewriter.inlineBlockBefore(&replacementBody.front(), &body, body.end(),
                                 body.getArguments());
    });
  }
  return WalkResult::advance();
}

WalkResult alignReductionYield(Operation *operation,
                               RelationWorklist &changes) {
  Region *combine = nullptr;
  ValueRange results;
  if (auto reduce = dyn_cast<ReduceOp>(operation)) {
    combine = &reduce.getCombine();
    results = reduce.getResults();
  } else if (auto scan = dyn_cast<ScanOp>(operation)) {
    combine = &scan.getCombine();
    results = scan.getResults();
  } else {
    return WalkResult::advance();
  }
  if (!combine || !llvm::hasSingleElement(*combine))
    return WalkResult::interrupt();
  auto yield = dyn_cast<YieldOp>(combine->front().getTerminator());
  if (!yield || yield.getValues().size() != results.size())
    return WalkResult::interrupt();
  OpBuilder builder(yield);
  builder.setListener(&changes);
  for (auto [index, target] : llvm::enumerate(results)) {
    FailureOr<Value> projected = projectPhysicalValueToSchema(
        builder, operation->getLoc(), yield.getValues()[index],
        target.getType(), changes.typeChanged());
    if (failed(projected)) {
      operation->emitOpError(
          "physical reduction yield cannot adopt its result relation")
          << "; result_index=" << index
          << "; yield_type=" << yield.getValues()[index].getType()
          << "; result_type=" << target.getType();
      return WalkResult::interrupt();
    }
    yield->setOperand(index, *projected);
  }
  return WalkResult::advance();
}

WalkResult alignStructuredCaptures(Operation *operation,
                                   RelationWorklist &changes) {
  auto sinkUniformCapture = [&](OpOperand &operand,
                                ArrayRef<BlockArgument> arguments) {
    Operation *owner = operand.getOwner();
    Value capture = operand.get();
    auto fragment = dyn_cast<FragmentType>(capture.getType());
    if (!fragment)
      return;
    Value scalar = capture;
    while (isa<FragmentType>(scalar.getType())) {
      UniformExpression expression = describeUniformValue(scalar);
      if (expression.kind != UniformKind::Forward ||
          expression.operands.size() != 1)
        break;
      scalar = expression.operands.front();
    }
    if (scalar.getType() != fragment.getElementType())
      return;
    // Capture the uniform seed and construct its fragment inside the helper.
    // This keeps the region closed while allowing each use to adopt its own
    // physical extent through the ordinary splat projection rule.
    for (BlockArgument argument : arguments) {
      auto type = cast<FragmentType>(argument.getType());
      changes.setType(argument, scalar.getType());
      OpBuilder builder(argument.getOwner(), argument.getOwner()->begin());
      builder.setListener(&changes);
      auto splat = builder.create<SplatOp>(owner->getLoc(), type, argument);
      argument.replaceAllUsesExcept(splat.getResult(), splat.getOperation());
    }
    operand.set(scalar);
  };
  auto align = [&](Operation *owner, Value capture,
                   BlockArgument argument) -> LogicalResult {
    auto authority = dyn_cast<FragmentType>(capture.getType());
    auto target = dyn_cast<FragmentType>(argument.getType());
    if (!authority || !target)
      return capture.getType() == argument.getType()
                 ? success()
                 : owner->emitOpError(
                       "physical structured capture lost its parent schema");
    if (authority.getElementType() != target.getElementType() ||
        authority.getShape().size() != target.getShape().size() ||
        authority.getAxisMaps() != target.getAxisMaps() ||
        authority.getOwner() != target.getOwner())
      return owner->emitOpError(
          "physical structured capture changed its coordinate relation");
    for (auto [axis, attribute] : llvm::enumerate(authority.getAxisMaps())) {
      auto mapping = cast<AxisMapAttr>(attribute);
      if (mapping.getDimensionId() <= 0)
        continue;
      if (failed(retargetDimensionExtent(
              argument, mapping.getDimensionId(),
              cast<PhysicalExprAttr>(authority.getShape()[axis]),
              changes.typeChanged(), &changes)))
        return failure();
    }
    return capture.getType() == argument.getType()
               ? success()
               : owner->emitOpError(
                     "physical structured capture extent is inconsistent");
  };

  auto structured = cast<StructuredOpInterface>(operation);
  auto captures = isa<RegionFoldOp>(operation)
                      ? cast<RegionFoldOp>(operation).getCaptures()
                      : cast<RegionScanOp>(operation).getCaptures();
  for (unsigned index = 0; index < captures.size(); ++index) {
    SmallVector<BlockArgument, 2> arguments{
        cast<BlockArgument>(structured.getSummarizeCaptures()[index])};
    if (structured.getEmitRegion())
      arguments.push_back(
          cast<BlockArgument>(structured.getEmitCaptures()[index]));
    OpOperand &operand =
        operation->getOpOperand(captures.getBeginOperandIndex() + index);
    sinkUniformCapture(operand, arguments);
    for (BlockArgument argument : arguments)
      if (failed(align(operation, operand.get(), argument)))
        return WalkResult::interrupt();
  }
  return WalkResult::advance();
}

} // namespace intent::gpu::value_relations
