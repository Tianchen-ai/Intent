#include "Construction.h"
#include "Intent/Analysis/ProductSchema.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;

namespace intent::kir_to_cpu {

LogicalResult Construction::orderedControl(Operation *operation) {
  SmallVector<Type> resultTypes;
  appendProductLeafTypes(operation->getResultTypes(), resultTypes);
  for (Type &type : resultTypes) type = valueType(type);
  Location loc = operation->getLoc();
  auto body = [&](Block &source, Block *target) {
    if (!target->empty() && target->back().hasTrait<OpTrait::IsTerminator>()) target->back().erase();
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(target);
    if (failed(lowerBlock(source))) return failure();
    builder.create<scf::YieldOp>(loc, flattened(source.getTerminator()->getOperands()));
    return success();
  };
  if (auto conditional = dyn_cast<IfOp>(operation)) {
    auto target = builder.create<scf::IfOp>(loc, resultTypes, values.lookup(conditional.getCondition()), true);
    if (failed(body(conditional.getThenRegion().front(), target.thenBlock())) ||
        failed(body(conditional.getElseRegion().front(), target.elseBlock()))) return failure();
    bindValues(operation->getResults(), target.getResults());
    return success();
  }
  if (auto loop = dyn_cast<ForOp>(operation)) {
    if (!domains.count(loop.getSource())) return operation->emitError("CPU ordered for requires a realized domain");
    Domain domain = domains.lookup(loop.getSource());
    auto target = builder.create<scf::ForOp>(loc, domain.begin, domain.end, domain.step,
                                            flattened(loop.getInitArgs()));
    values.map(loop.getInductionVars().front(), target.getInductionVar());
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(target.getBody());
      bindValues(loop.getRegionIterArgs(), target.getRegionIterArgs());
    }
    if (failed(body(loop.getBody().front(), target.getBody()))) return failure();
    bindValues(operation->getResults(), target.getResults());
    return success();
  }
  auto loop = cast<WhileOp>(operation);
  auto initial = flattened(loop.getInitArgs());
  auto target = builder.create<scf::WhileOp>(loc, resultTypes, initial);
  Block &before = target.getBefore().emplaceBlock();
  Block &after = target.getAfter().emplaceBlock();
  for (Type type : TypeRange(initial)) before.addArgument(type, loc);
  for (Type type : resultTypes) after.addArgument(type, loc);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&before);
    bindValues(loop.getBeforeArguments(), before.getArguments());
    if (failed(lowerBlock(loop.getBefore().front()))) return failure();
    auto condition = cast<ConditionOp>(loop.getBefore().front().getTerminator());
    builder.create<scf::ConditionOp>(loc, values.lookup(condition.getCondition()),
                                     flattened(condition.getArgs()));
  }
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&after);
    bindValues(loop.getAfterArguments(), after.getArguments());
  }
  if (failed(body(loop.getAfter().front(), &after))) return failure();
  bindValues(operation->getResults(), target.getResults());
  return success();
}

LogicalResult Construction::lower(ParallelOp op) {
  Location loc = op.getLoc();
  if (!domains.count(op.getSource()))
    return op.emitError("CPU parallel source is not a supported domain");
  Domain domain = domains.lookup(op.getSource());
  auto parallel = builder.create<scf::ParallelOp>(loc, ValueRange{constant(loc, 0)},
      ValueRange{domain.extent}, ValueRange{constant(loc, 1)});
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(parallel.getBody());
  Value coordinate = builder.createOrFold<arith::AddIOp>(loc, domain.begin,
      builder.createOrFold<arith::MulIOp>(loc, parallel.getInductionVars()[0], domain.step));
  values.map(op.getBody().front().getArgument(0), coordinate);
  return lowerBlock(op.getBody().front());
}

} // namespace intent::kir_to_cpu
