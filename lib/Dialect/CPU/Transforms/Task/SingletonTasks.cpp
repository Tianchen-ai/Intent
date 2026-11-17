#include "Intent/Dialect/CPU/Transforms/Task/Tasks.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;

namespace intent::cpu {

void foldSequentialAtomicAdds(func::FuncOp function) {
  SmallVector<AtomicRMWOp> updates;
  function.walk([&](AtomicRMWOp update) {
    if (update.getOrdering() != AtomicOrdering::Relaxed ||
        update.getKind() != AtomicRMWKind::Add ||
        !isa<FloatType, IntegerType>(update.getValue().getType())) return;
    for (Operation *parent = update->getParentOp(); parent != function;
         parent = parent->getParentOp()) {
      if (auto tasks = dyn_cast<TasksOp>(parent)) {
        if (getConstantIntValue(tasks.getCount()) == 1) continue;
      }
      if (!isa<scf::ForOp, scf::IfOp>(parent)) return;
    }
    updates.push_back(update);
  });
  for (auto update : updates) {
    OpBuilder builder(update);
    Location location = update.getLoc();
    Value old = builder.create<memref::LoadOp>(location, update.getTarget(), update.getIndices());
    Value sum = isa<FloatType>(old.getType())
        ? Value(builder.create<arith::AddFOp>(location, old, update.getValue()))
        : Value(builder.create<arith::AddIOp>(location, old, update.getValue()));
    builder.create<memref::StoreOp>(location, sum, update.getTarget(), update.getIndices());
    update.getOldValue().replaceAllUsesWith(old);
    update.erase();
  }
}

LogicalResult specializeSingletonTasks(func::FuncOp function) {
  SmallVector<TasksOp> tasks;
  function.walk([&](TasksOp task) { tasks.push_back(task); });
  for (TasksOp task : tasks) {
    if (getConstantIntValue(task.getCount())) continue;
    bool hasRelaxedAdd = false;
    task.walk([&](AtomicRMWOp update) {
      hasRelaxedAdd |= update.getOrdering() == AtomicOrdering::Relaxed &&
                       update.getKind() == AtomicRMWKind::Add;
    });
    if (!hasRelaxedAdd) continue;
    OpBuilder builder(task);
    Location location = task.getLoc();
    Value one = builder.create<arith::ConstantIndexOp>(location, 1);
    auto choice = builder.create<scf::IfOp>(location,
        builder.create<arith::CmpIOp>(location, arith::CmpIPredicate::eq,
                                     task.getCount(), one), true);
    builder.setInsertionPointToStart(choice.thenBlock());
    IRMapping mapping;
    auto singleton = cast<TasksOp>(builder.clone(*task, mapping));
    singleton.getCountMutable().assign(one);
    task->moveBefore(choice.elseBlock()->getTerminator());
  }
  // Only the explicit count-one branch has no concurrent logical participant.
  // Retain old values and exact source arithmetic; no atomic reduction merging.
  foldSequentialAtomicAdds(function);
  return success();
}

} // namespace intent::cpu
