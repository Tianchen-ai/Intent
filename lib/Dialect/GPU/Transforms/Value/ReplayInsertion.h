#ifndef INTENT_GPU_TRANSFORMS_VALUE_REPLAYINSERTION_H
#define INTENT_GPU_TRANSFORMS_VALUE_REPLAYINSERTION_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include <iterator>

namespace intent::gpu {

class ReplayInsertion {
public:
  explicit ReplayInsertion(mlir::OpBuilder &builder)
      : builder(builder), insertion(builder.saveInsertionPoint()),
        previous(insertion.getPoint() == insertion.getBlock()->begin()
            ? nullptr : &*std::prev(insertion.getPoint())) {}
  ~ReplayInsertion() {
    if (committed) return;
    auto end = insertion.getPoint();
    while (end != insertion.getBlock()->begin()) {
      mlir::Operation *operation = &*std::prev(end);
      if (operation == previous) break;
      operation->erase();
    }
    builder.restoreInsertionPoint(insertion);
  }
  void commit() { committed = true; }

private:
  mlir::OpBuilder &builder;
  mlir::OpBuilder::InsertPoint insertion;
  mlir::Operation *previous;
  bool committed = false;
};

inline bool availableAtInsertionPoint(mlir::Value value, mlir::OpBuilder &builder,
                                      mlir::DominanceInfo &dominance) {
  if (!value || !builder.getInsertionBlock()) return false;
  if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value))
    return dominance.dominates(argument.getOwner(), builder.getInsertionBlock());
  mlir::Operation *definition = value.getDefiningOp();
  return definition && dominance.properlyDominates(
      definition->getBlock(), definition->getIterator(),
      builder.getInsertionBlock(), builder.getInsertionPoint(),
      /*enclosingOk=*/false);
}

} // namespace intent::gpu

#endif
