#include "Construction.h"
#include "Intent/Analysis/ProductSchema.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "Intent/Dialect/CPU/IR/CollectiveHelpers.h"
#include "Intent/Interfaces/StructuredOpInterface.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

namespace intent::kir_to_cpu {

LogicalResult Construction::helper(Region &original, Region &target, ValueRange captures) {
  OpBuilder::InsertionGuard guard(builder);
  Block *body = new Block;
  target.push_back(body);
  SmallVector<Type> inputTypes;
  appendProductLeafTypes(original.front().getArgumentTypes(), inputTypes);
  if (captures.size() > inputTypes.size())
    return original.getParentOp()->emitError("CPU helper capture schema is incomplete");
  for (auto [index, capture] : llvm::enumerate(captures))
    inputTypes[inputTypes.size() - captures.size() + index] = capture.getType();
  for (Type type : inputTypes) body->addArgument(valueType(type), original.getLoc());
  builder.setInsertionPointToStart(body);
  bindValues(original.front().getArguments(), body->getArguments());
  if (failed(lowerBlock(original.front()))) return failure();
  auto results = flattened(original.front().getTerminator()->getOperands());
  if (isa<cpu::SliceReduceOp>(target.getParentOp()))
    builder.create<cpu::SliceReduceYieldOp>(original.getLoc(), results);
  else if (isa<cpu::ScanOp>(target.getParentOp()))
    builder.create<cpu::ScanYieldOp>(original.getLoc(), results);
  else builder.create<cpu::RegionYieldOp>(original.getLoc(), results);
  return success();
}

LogicalResult Construction::region(Operation *operation) {
  auto schema = cast<StructuredOpInterface>(operation);
  const bool scan = schema.getStructuredKind() == StructuredOpKind::RegionScan;
  int64_t axis = schema.getIterationAxes().front();
  auto sources = flattened(schema.getSources());
  auto identityValues = flattened(schema.getIdentities());
  auto stateValues = flattened(schema.getInitialStates());
  auto captures = flattened(schema.getCaptures());
  Location loc = operation->getLoc();
  SmallVector<Type> results;
  appendProductLeafTypes(operation->getResultTypes(), results);
  for (Type &type : results) type = valueType(type);
  auto outputs = emptyResults(operation->getResults(), loc);
  if (failed(outputs)) return failure();
  auto outputFields = fieldPaths(schema.getEmittedResults().getTypes());
  SmallVector<int64_t> outputAxes;
  if (scan) {
    LogicalResult status = success();
    for (Value output : schema.getEmittedResults())
      walkProductLeaves(output.getType(), [&](Type, ArrayRef<unsigned> path) {
        if (failed(status)) return;
        auto member = analysis.emissionAxis(cast<OpResult>(output), path);
        if (failed(member)) { status = failure(); return; }
        outputAxes.push_back(*member);
      });
    if (failed(status))
      return operation->emitError("CPU region emit has no canonical member axis relation");
  }
  Operation *target;
  if (scan)
    target = builder.create<cpu::RegionScanOp>(loc,
        TypeRange(results).take_front(outputFields.size()),
        TypeRange(results).drop_front(outputFields.size()),
        sources, identityValues, stateValues, captures,
        ValueRange(*outputs).take_front(outputFields.size()),
        ValueRange(*outputs).drop_front(outputFields.size()), axis,
        fieldPaths(schema.getIdentities().getTypes()), fieldPaths(schema.getInitialStates().getTypes()),
        builder.getDenseI64ArrayAttr(outputAxes), IntegerAttr());
  else
    target = builder.create<cpu::RegionFoldOp>(loc, results, sources, identityValues, captures,
        *outputs, axis, fieldPaths(schema.getIdentities().getTypes()), IntegerAttr());
  auto targetSchema = cast<cpu::RegionOpInterface>(target);
  if (failed(helper(*schema.getSummarizeRegion(), targetSchema.getSummarize(), captures))) return failure();
  if (failed(helper(schema.getCombine(), targetSchema.getCombine()))) return failure();
  if (scan) {
    if (failed(helper(*schema.getApplyRegion(), *targetSchema.getApplyRegion()))) return failure();
    if (failed(helper(*schema.getEmitRegion(), *targetSchema.getEmitRegion(), captures))) return failure();
  }
  bindValues(operation->getResults(), target->getResults());
  return success();
}

LogicalResult Construction::scan(ScanOp operation) {
  SmallVector<Type> results;
  appendProductLeafTypes(operation.getResultTypes(), results);
  for (Type &type : results) type = valueType(type);
  auto outputs = emptyResults(operation.getResults(), operation.getLoc());
  if (failed(outputs)) return failure();
  auto target = builder.create<cpu::ScanOp>(operation.getLoc(), results,
      flattened(operation.getSources()), flattened(operation.getIdentities()),
      flattened(operation.getCaptures()), *outputs,
      operation.getAxis(), operation.getInclusive(), operation.getReverse());
  if (failed(helper(operation.getCombine(), target.getCombine(), flattened(operation.getCaptures())))) return failure();
  bindValues(operation.getResults(), target.getResults());
  return success();
}

LogicalResult Construction::reduce(ReduceOp operation) {
  auto sources = flattened(operation.getSources());
  auto identities = flattened(operation.getIdentities());
  auto captures = flattened(operation.getCaptures());
  Location loc = operation.getLoc();
  SmallVector<Type> results;
  appendProductLeafTypes(operation.getResultTypes(), results);
  for (Type &type : results) type = valueType(type);
  auto outputs = emptyResults(operation.getResults(), loc);
  if (failed(outputs)) return failure();
  if (results.size() != identities.size() || sources.size() != identities.size())
    return operation.emitError("CPU reduction source, identity and output leaves differ");
  SmallVector<int64_t> axes;
  for (Attribute axis : operation.getAxes())
    axes.push_back(cast<IntegerAttr>(axis).getInt());
  auto reduction = builder.create<cpu::SliceReduceOp>(loc, results, sources, identities,
      captures, *outputs, axes,
      cpu::ReductionOrderAttr::get(builder.getContext(), true, true));
  if (failed(helper(operation.getCombine(), reduction.getCombine(), captures))) return failure();
  bindValues(operation.getResults(), reduction.getResults());
  return success();
}

LogicalResult Construction::lower(HistogramOp op) {
  Location loc = op.getLoc();
  auto type = cast<RankedTensorType>(op.getResult().getType());
  auto sizes = extents(op.getResult(), loc);
  if (failed(sizes)) return failure();
  Value output = emptyTensor(type, *sizes, loc);
  auto integer = dyn_cast<IntegerType>(getElementTypeOrSelf(op.getValues().getType()));
  auto result = builder.create<cpu::HistogramOp>(loc, tensorType(type), values.lookup(op.getValues()), values.lookup(op.getValid()), output,
      integer && integer.isUnsigned());
  values.map(op.getResult(), result.getResult());
  return success();
}

} // namespace intent::kir_to_cpu
