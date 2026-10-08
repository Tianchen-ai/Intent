#ifndef INTENT_GPU_TRANSFORMS_VALUE_RELATIONS_WORKLIST_H
#define INTENT_GPU_TRANSFORMS_VALUE_RELATIONS_WORKLIST_H

#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <array>
#include <deque>
#include <functional>

namespace intent::gpu::value_relations {

// One queue owns relation invalidation while a transformation closes its IR.
// Rules propagate an already-selected schema through this listener; they do not
// own a separate worklist, analysis snapshot, or ownership policy.
class RelationWorklist : public mlir::RewriterBase::Listener {
public:
  RelationWorklist(mlir::func::FuncOp kernel, ValueRelationScope scope);

  ValueTypeChangeCallback typeChanged();
  void setType(mlir::Value value, mlir::Type type);
  mlir::LogicalResult run();

  void notifyOperationInserted(mlir::Operation *operation,
                               mlir::OpBuilder::InsertPoint point) override;
  void notifyOperationErased(mlir::Operation *operation) override;
  void notifyOperationReplaced(mlir::Operation *operation,
                               mlir::ValueRange replacements) override;
  void notifyOperationModified(mlir::Operation *operation) override;

private:
  enum Rule : unsigned {
    Captures,
    ReductionResult,
    ReductionIdentity,
    Aggregate,
    AccessResult,
    Pointwise,
    ReductionYield,
    AccessValue,
    Reshape,
    ContractOperands,
    ContractAccumulator,
    RuleCount
  };

  bool enabled(Rule rule) const;
  void push(mlir::Operation *operation, Rule rule);
  static unsigned priority(Rule rule);
  void enqueue(mlir::Operation *operation);
  void affected(mlir::Value value);
  void typeChanged(mlir::Value value, mlir::Type previous);

  mlir::func::FuncOp kernel;
  ValueRelationScope scope;
  std::function<void(mlir::Value, mlir::Type)> callback;
  std::array<std::deque<std::pair<mlir::Operation *, Rule>>, RuleCount>
      worklists;
  std::array<llvm::DenseSet<mlir::Operation *>, RuleCount> queued;
  llvm::DenseSet<mlir::Operation *> live;
  llvm::DenseMap<mlir::Value,
                 llvm::SmallVector<std::pair<mlir::Type, mlir::Type>>>
      transitions;
  bool conflict = false;
};

mlir::WalkResult alignPointwiseValue(mlir::Operation *operation,
                                     RelationWorklist &changes);
mlir::LogicalResult refreshReshapeRelation(ReshapeOp reshape,
                                           RelationWorklist &changes);
mlir::WalkResult alignAccessResult(mlir::Operation *operation,
                                   RelationWorklist &changes);
mlir::LogicalResult alignAccessValue(mlir::Operation *operation,
                                     RelationWorklist &changes);
mlir::WalkResult alignContractOperands(mlir::Operation *operation,
                                       RelationWorklist &changes);
mlir::LogicalResult alignContractAccumulator(mlir::Operation *operation,
                                             RelationWorklist &changes);
mlir::WalkResult alignReductionResultRelation(mlir::Operation *operation,
                                              RelationWorklist &changes);
mlir::WalkResult alignReductionIdentityRelation(mlir::Operation *operation,
                                                RelationWorklist &changes);
mlir::WalkResult alignReductionYield(mlir::Operation *operation,
                                     RelationWorklist &changes);
mlir::WalkResult alignStructuredCaptures(mlir::Operation *operation,
                                         RelationWorklist &changes);

} // namespace intent::gpu::value_relations

#endif
