#include "Intent/Dialect/GPU/Analysis/ReductionConsumers.h"
#include "Intent/Dialect/GPU/Analysis/Helpers.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace intent::gpu {
namespace {

template <typename T> void appendUnique(SmallVectorImpl<T> &values, T value) {
  if (!llvm::is_contained(values, value)) values.push_back(value);
}

bool pointwiseNode(Operation *operation) {
  return operation && operation->getNumRegions() == 0 &&
         isMemoryEffectFree(operation) &&
         (isLaneWisePointwiseOperation(operation) ||
          isa<SplatOp, BroadcastOp, ReshapeOp, TransposeOp>(operation));
}

bool inexpensiveNode(Operation *operation) {
  if (!pointwiseNode(operation)) return false;
  if (auto unary = dyn_cast<UnaryOp>(operation)) {
    switch (unary.getOperatorKind()) {
    case UnaryOperator::Negate:
    case UnaryOperator::Not:
    case UnaryOperator::Floor:
    case UnaryOperator::Abs:
      return true;
    default:
      return false;
    }
  }
  if (auto binary = dyn_cast<BinaryOp>(operation)) {
    switch (binary.getOperatorKind()) {
    case BinaryOperator::TrueDivide:
    case BinaryOperator::FloorDivide:
    case BinaryOperator::Remainder:
    case BinaryOperator::Power:
      return false;
    default:
      break;
    }
  }
  return true;
}

class ConsumerQuery {
public:
  ConsumerQuery(ReduceOp reduce, unsigned axis, func::FuncOp kernel)
      : block(reduce->getBlock()), analysis(kernel), dominance(kernel) {
    group.reduce = reduce;
    group.reductionAxis = axis;
  }

  FailureOr<ReductionConsumerGroup> run() {
    SmallVector<MakeRangeOp> ranges;
    for (Value source : group.reduce.getSources()) {
      auto type = dyn_cast<FragmentType>(source.getType());
      if (!type || group.reductionAxis >= type.getShape().size()) return failure();
      if (uniformScalarSource(source)) continue;
      auto fact = analysis.axisRanges(source, group.reductionAxis);
      if (!fact.isExact() || fact.roots.empty()) return failure();
      for (MakeRangeOp range : fact.roots) appendUnique(ranges, range);
    }
    auto lockstep = analysis.lockstepRanges(ranges);
    if (!lockstep.isExact() || !lockstep.authority ||
        !isUnitStepRange(lockstep.authority))
      return failure();
    group.range = lockstep.authority;
    if (failed(findStores())) return failure();

    for (Value value : group.reduce.getSources())
      if (failed(collect(value, /*output=*/false))) return failure();
    for (StoreOp store : group.stores)
      for (Value value : store->getOperands())
        if (failed(collect(value, /*output=*/true))) return failure();

    // A shared coordinate/mask alone is not a retained numerical producer.
    bool sharedPayload = llvm::any_of(group.wholeProducers, [&](Value value) {
      auto element = cast<FragmentType>(value.getType()).getElementType();
      return !element.isIndex() && !element.isInteger(1) &&
             sourceValues.contains(value) && outputValues.contains(value);
    });
    if (!sharedPayload || group.sourceLoads.empty()) return failure();
    llvm::SmallPtrSet<Operation *, 32> replaced;
    replaced.insert(group.reduce);
    for (StoreOp store : group.stores) replaced.insert(store);
    for (Operation *operation : group.producerOperations) replaced.insert(operation);
    for (Operation *operation : group.outputOperations) replaced.insert(operation);
    for (Value value : group.wholeProducers)
      for (Operation *user : value.getUsers())
        if (!replaced.contains(user)) return failure();

    auto lexical = [](Operation *left, Operation *right) {
      return left->isBeforeInBlock(right);
    };
    llvm::sort(group.producerOperations, lexical);
    llvm::sort(group.outputOperations, lexical);
    llvm::sort(group.stores, [&](StoreOp left, StoreOp right) {
      return lexical(left, right);
    });
    return std::move(group);
  }

private:
  FailureOr<PhysicalAxisProjection> projection(Value value) {
    auto matches = queryRangeProjections(value.getType(), group.range);
    if (matches.size() == 1 && matches.front().isExact()) return matches.front();
    if (!matches.empty()) return failure();
    auto sourceAxes = queryFragmentAxes(value.getType(), sourceAxisIdentity(group.range));
    auto dimension = queryRangeDimension(group.range);
    if (!sourceAxes.empty() ||
        (succeeded(dimension) &&
         !queryFragmentDimensions(value.getType(), *dimension).empty()))
      return failure();
    return PhysicalAxisProjection{};
  }

  LogicalResult findStores() {
    SmallVector<Value> pending(group.reduce.getResults());
    llvm::DenseSet<Value> visited;
    for (unsigned i = 0; i < pending.size(); ++i) {
      Value value = pending[i];
      if (!visited.insert(value).second) continue;
      for (Operation *user : value.getUsers()) {
        if (user->getBlock() != block) return failure();
        if (auto store = dyn_cast<StoreOp>(user)) {
          if (store.getValue() != value || store.getValid() == value ||
              store.getResource() == value ||
              llvm::is_contained(store.getCoordinates(), value))
            return failure();
          auto axis = projection(value);
          if (failed(axis)) return failure();
          // A summary-only publication remains outside the member traversal.
          if (axis->isExact()) appendUnique(group.stores, store);
          continue;
        }
        if (!pointwiseNode(user)) return failure();
        llvm::append_range(pending, user->getResults());
      }
    }
    return success(!group.stores.empty());
  }

  LogicalResult collect(Value value, bool output) {
    auto &visited = output ? outputValues : sourceValues;
    if (!visited.insert(value).second) return success();
    if (llvm::is_contained(group.reduce.getResults(), value)) {
      appendUnique(group.captures, value);
      return success();
    }
    auto axis = projection(value);
    if (failed(axis)) return failure();
    if (!axis->isExact()) {
      if (isa<RecordType>(value.getType())) return failure();
      if (dominance.dominates(value, group.reduce.getOperation())) {
        appendUnique(group.captures, value);
        return success();
      }
      Operation *operation = value.getDefiningOp();
      if (!output || !operation || operation->getBlock() != block ||
          !pointwiseNode(operation))
        return failure();
      appendUnique(group.outputOperations, operation);
      for (Value operand : operation->getOperands())
        if (failed(collect(operand, /*output=*/true))) return failure();
      return success();
    }
    Operation *operation = value.getDefiningOp();
    if (!operation || operation->getBlock() != block) return failure();
    if (!isa<LoadOp, MakeRangeOp>(operation) &&
        !(output ? pointwiseNode(operation) : inexpensiveNode(operation)))
      return failure();
    auto &operations = output ? group.outputOperations : group.producerOperations;
    appendUnique(operations, operation);
    if (auto load = dyn_cast<LoadOp>(operation)) appendUnique(group.sourceLoads, load);
    if (!isa<MakeRangeOp>(operation) && !uniformScalarSource(value) &&
        !llvm::is_contained(group.wholeProducers, value)) {
      group.wholeProducers.push_back(value);
      group.projections.push_back({value, *axis});
    }
    for (Value operand : operation->getOperands())
      if (failed(collect(operand, output))) return failure();
    return success();
  }

  Block *block;
  PhysicalProgramAnalysis analysis;
  DominanceInfo dominance;
  ReductionConsumerGroup group;
  llvm::DenseSet<Value> sourceValues, outputValues;
};

} // namespace

FailureOr<ReductionConsumerGroup>
queryReductionConsumerGroup(ReduceOp reduce, unsigned reductionAxis) {
  auto kernel = reduce->getParentOfType<func::FuncOp>();
  if (!kernel || !llvm::is_contained(reduce.getAxes(), int64_t(reductionAxis)) ||
      failed(proveLaneWiseHelper(reduce.getCombine())))
    return failure();
  return ConsumerQuery(reduce, reductionAxis, kernel).run();
}

} // namespace intent::gpu
