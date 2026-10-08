#include "Intent/Dialect/CPU/Transforms/Implementation/Implementation.h"
#include "Intent/Dialect/CPU/Transforms/Implementation/ImplementationInputs.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/CPU/IR/CPUAttrs.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "InputWindows.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Intent/Dialect/CPU/Transforms/Structure/LoopBuilders.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include <functional>

using namespace mlir;
namespace intent::cpu {
namespace {

struct PreparedInput {
  linalg::LinalgOp producer;
  memref::AllocOp allocation;
  memref::DeallocOp end;
  ImplementationAttr consumer;
};

std::optional<PreparedInput> scopedInput(linalg::LinalgOp producer, Operation *loop,
    StorageAnalysis &storage, const llvm::SmallDenseSet<Value> &groupInputs) {
  if (producer->getNumResults() || producer.getNumDpsInits() != 1 || producer.getNumReductionLoops() ||
      !producer.getIndexingMapsArray().back().isIdentity() ||
      !producer->getRegion(0).front().getArguments().back().use_empty()) return std::nullopt;
  auto allocation = producer.getDpsInits()[0].getDefiningOp<memref::AllocOp>();
  if (!allocation || allocation->getBlock() != producer->getBlock() ||
      !allocation->isBeforeInBlock(producer)) return std::nullopt;
  auto stable = [&](Value memory) {
    Value root = storage.uniqueOrigin(memory);
    if (!root) return false;
    if (groupInputs.contains(root)) return true;
    Operation *owner = root.getDefiningOp();
    if (!storage.isReadOnly(memory) &&
        (!isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(owner) ||
         loop->isAncestor(owner))) return false;
    return storage.preserves(loop, memory);
  };
  for (Value input : producer.getDpsInputs())
    if (isa<MemRefType>(input.getType()) && !stable(input)) return std::nullopt;
  for (Operation &operation : producer->getRegion(0).front().without_terminator()) {
    if (operation.getNumRegions()) return std::nullopt;
    if (auto load = dyn_cast<memref::LoadOp>(operation)) {
      if (!stable(load.getMemref())) return std::nullopt;
    } else if (!isMemoryEffectFree(&operation)) return std::nullopt;
  }

  auto lifetime = storage.lifetime(allocation);
  if (!lifetime || !lifetime->aliases.complete) return std::nullopt;
  PreparedInput result{producer, allocation, lifetime->end, {}};
  SmallVector<Operation *> reads;
  for (Operation *user : lifetime->aliases.users) {
    if (user == producer) continue;
    if (!loop->isAncestor(user)) return std::nullopt;
    if (user == result.end || isStorageAliasOperation(user) || isa<memref::DimOp>(user)) continue;
    if (!isa<memref::CopyOp, linalg::LinalgOp, memref::LoadOp>(user) ||
        !storage.preserves(user, allocation)) return std::nullopt;
    reads.push_back(user);
  }
  if (reads.empty() || !producer->isBeforeInBlock(result.end)) return std::nullopt;
  for (Operation *read : reads) {
    Operation *ancestor = producer->getBlock()->findAncestorOpInBlock(*read);
    if (!ancestor || !producer->isBeforeInBlock(ancestor)) return std::nullopt;
  }
  return result;
}

Value inputGroupSize(OpBuilder &builder, scf::ParallelOp parallel, int64_t divisor, int64_t workers) {
  Location loc = parallel.getLoc();
  Value zero = index(builder, loc, 0), one = index(builder, loc, 1);
  Value remaining = index(builder, loc, workers);
  auto account = [&](scf::ParallelOp owner, unsigned axes) {
    for (unsigned axis = 0; axis < axes; ++axis) {
      auto step = getConstantIntValue(owner.getStep()[axis]);
      if (!step || *step <= 0) continue;
      Value extent = builder.create<arith::SubIOp>(loc, owner.getUpperBound()[axis], owner.getLowerBound()[axis]);
      extent = builder.create<arith::MaxSIOp>(loc, extent, zero);
      Value count = builder.create<arith::CeilDivSIOp>(loc, extent, owner.getStep()[axis]);
      count = builder.create<arith::MaxSIOp>(loc, count, one);
      remaining = builder.create<arith::CeilDivSIOp>(loc, remaining, count);
    }
  };
  account(parallel, parallel.getNumLoops() - 1);
  for (Operation *parent = parallel->getParentOp(); parent; parent = parent->getParentOp())
    if (auto outer = dyn_cast<scf::ParallelOp>(parent)) account(outer, outer.getNumLoops());
  Value budget = builder.create<arith::DivSIOp>(loc, parallel.getUpperBound().back(), remaining);
  SmallVector<int64_t> sizes{divisor};
  while (sizes.back() % 2 == 0) sizes.push_back(sizes.back() / 2);
  if (sizes.back() != 1) sizes.push_back(1);
  Value size = one;
  // Choose subgroup sizes that divide the original quotient cohort, retaining
  // enough nominal worksets for the worker budget across parallel axes.
  for (int64_t candidate : llvm::reverse(sizes)) {
    if (candidate == 1) continue;
    Value value = index(builder, loc, candidate);
    Value fits = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, budget, value);
    size = builder.create<arith::SelectOp>(loc, fits, value, size);
  }
  return size;
}

LogicalResult groupScopedInputs(func::FuncOp function, const ImplementationRegistry &implementations) {
  auto interface = function->getAttrOfType<EntryRequirementsAttr>(entryRequirementsAttr);
  if (!interface || !interface.getDisjointOutputs()) return success();
  auto configuration = function->getAttrOfType<ConfigurationAttr>("intent_cpu.configuration");
  llvm::DenseMap<Value, SmallVector<memref::SubViewOp>> requestedWindows;
  auto requests = function.walk([&](linalg::GenericOp operation) -> WalkResult {
    if (!isMatrixContraction(operation) || operation->hasAttr("intent_cpu.microtile")) return WalkResult::advance();
    auto implementation = implementations.lookup(operation);
    if (failed(implementation)) return WalkResult::interrupt();
    auto binding = operation->getAttrOfType<ImplementationAttr>("intent_cpu.implementation");
    auto requirements = (*implementation)->inputRequirements(operation, configuration, binding);
    if (auto reason = checkInputRequirements(operation, requirements)) {
      operation.emitError(*reason);
      return WalkResult::interrupt();
    }
    for (auto requirement : requirements) {
      Value source = operation.getInputs()[requirement.operand];
      auto window = consumerWindow(source, requirement, operation);
      if (!window) continue;
      Value base = window->view.getSource();
      if (!base.getDefiningOp()) continue;
      Operation *owner = base.getDefiningOp()->getParentOp();
      bool serial = true;
      for (Operation *parent = operation->getParentOp(); parent != owner; parent = parent->getParentOp()) {
        if (auto loop = dyn_cast<scf::ForOp>(parent)) serial &= loop.getNumResults() == 0;
        else if (!isa<scf::IfOp>(parent)) { serial = false; break; }
      }
      if (serial) requestedWindows[base].push_back(window->view);
    }
    return WalkResult::advance();
  });
  if (requests.wasInterrupted()) return failure();
  SmallVector<Operation *> loops;
  function.walk([&](Operation *operation) {
    if (isa<scf::ForOp, scf::ParallelOp>(operation)) loops.push_back(operation);
  });
  for (Operation *loop : llvm::reverse(loops)) {
    Value lower, upper, step, induction;
    SmallVector<Value> outerCoordinates;
    Block *body;
    if (auto serial = dyn_cast<scf::ForOp>(loop)) {
      if (serial.getNumResults()) continue;
      lower = serial.getLowerBound(); upper = serial.getUpperBound(); step = serial.getStep();
      induction = serial.getInductionVar(); body = serial.getBody();
    } else {
      auto parallel = cast<scf::ParallelOp>(loop);
      if (!parallel.getNumLoops() || parallel.getNumResults()) continue;
      lower = parallel.getLowerBound().back(); upper = parallel.getUpperBound().back(); step = parallel.getStep().back();
      induction = parallel.getInductionVars().back(); body = parallel.getBody();
      llvm::append_range(outerCoordinates, ValueRange(parallel.getInductionVars()).drop_back());
    }
    if (getConstantIntValue(lower) != 0 || getConstantIntValue(step) != 1) continue;
    StorageAnalysis storage(function);
    auto effects = storage.effects(loop);
    if (!effects.complete || effects.ordered) continue;
    for (auto quotient : body->getOps<arith::FloorDivSIOp>()) {
      auto divisor = getConstantIntValue(quotient.getRhs());
      if (quotient.getLhs() != induction || !divisor || *divisor <= 1) continue;
      SmallVector<PreparedInput> preparations;
      llvm::SetVector<Operation *> dependencies;
      llvm::SmallDenseSet<Value> groupInputs;
      bool groupDependent = false;
      for (Operation &operation : body->without_terminator()) {
        SmallVector<Value> frontier{quotient.getResult()};
        llvm::append_range(frontier, outerCoordinates);
        llvm::append_range(frontier, groupInputs);
        if (auto view = dyn_cast<memref::SubViewOp>(operation);
            view && requestedWindows.count(view.getResult()) && storage.isReadOnly(view)) {
          bool stable = storage.preserves(loop, view);
          if (stable && llvm::any_of(requestedWindows[view], [&](memref::SubViewOp window) {
                return hasIndependentWindowCoordinates(window, loop, quotient.getResult());
              })) {
            auto needed = inputWindowDependencies(ValueRange{view.getResult()}, loop, frontier);
            if (!needed || llvm::any_of(needed->operations, [&](Operation *dependency) {
                  return dependency->getBlock() != body;
                })) continue;
            // Share only the invariant descriptor here. The selected input
            // supply retains its original guards when it later fills storage.
            dependencies.insert(needed->operations.begin(), needed->operations.end());
            groupDependent |= llvm::is_contained(needed->frontier, quotient.getResult());
          }
          continue;
        }
        auto producer = dyn_cast<linalg::LinalgOp>(operation);
        if (!producer) continue;
        auto prepared = scopedInput(producer, loop, storage, groupInputs);
        if (!prepared) continue;
        llvm::SetVector<Value> inputs;
        getUsedValuesDefinedAbove(producer->getRegion(0), inputs);
        auto operands = producer.getDpsInputs();
        inputs.insert(operands.begin(), operands.end());
        inputs.insert(prepared->allocation->operand_begin(), prepared->allocation->operand_end());
        auto needed = inputWindowDependencies(inputs.getArrayRef(), loop, frontier);
        if (!needed || llvm::any_of(needed->operations, [&](Operation *dependency) {
              return dependency->getBlock() != body;
            })) continue;
        dependencies.insert(needed->operations.begin(), needed->operations.end());
        preparations.push_back(*prepared);
        groupInputs.insert(prepared->allocation.getResult());
        groupDependent |= llvm::is_contained(needed->frontier, quotient.getResult());
      }
      if (!groupDependent) continue;
      OpBuilder builder(loop);
      Location loc = loop->getLoc();
      scf::ParallelOp parallel = dyn_cast<scf::ParallelOp>(loop), outerParallel;
      auto capabilities = function->getParentOfType<ModuleOp>()->getAttrOfType<CapabilitiesAttr>("intent_cpu.capabilities");
      Value size = parallel ? inputGroupSize(builder, parallel, *divisor, capabilities.getWorkers())
                            : index(builder, loc, *divisor);
      Value originalDivisor = index(builder, loc, *divisor);
      Value count = builder.create<arith::CeilDivSIOp>(loc, upper, size);
      Value ordinal;
      Block *group;
      if (parallel) {
        SmallVector<Value> ends(parallel.getUpperBound());
        ends.back() = count;
        outerParallel = builder.create<scf::ParallelOp>(loc, parallel.getLowerBound(), ends, parallel.getStep());
        group = outerParallel.getBody(); ordinal = outerParallel.getInductionVars().back();
      } else {
        auto outer = builder.create<scf::ForOp>(loc, lower, count, step);
        group = outer.getBody(); ordinal = outer.getInductionVar();
      }
      builder.setInsertionPointToStart(group);
      Value begin = multiply(builder, loc, ordinal, size);
      Value sourceGroup = builder.create<arith::DivSIOp>(loc, begin, originalDivisor);
      Value remaining = builder.create<arith::SubIOp>(loc, upper, begin);
      Value extent = builder.create<arith::MinSIOp>(loc, size, remaining);
      Value end = add(builder, loc, begin, extent);
      if (parallel) {
        auto inner = builder.create<scf::ForOp>(loc, begin, end, step);
        auto replacements = outerParallel.getInductionVars();
        for (auto [original, replacement] : llvm::zip(outerCoordinates, ArrayRef(replacements).drop_back()))
          original.replaceAllUsesWith(replacement);
        induction.replaceAllUsesWith(inner.getInductionVar());
        for (Operation &operation : llvm::make_early_inc_range(body->without_terminator()))
          operation.moveBefore(inner.getBody()->getTerminator());
        parallel.erase();
        loop = inner;
        body = inner.getBody();
      } else {
        auto serial = cast<scf::ForOp>(loop);
        serial->moveBefore(group->getTerminator());
        serial.setLowerBound(begin); serial.setUpperBound(end);
      }
      quotient.getResult().replaceAllUsesWith(sourceGroup);
      quotient.erase();
      for (auto preparation : preparations) {
        dependencies.insert(preparation.allocation);
        dependencies.insert(preparation.producer);
      }
      SmallVector<Operation *> ordered;
      for (Operation &operation : body->without_terminator())
        if (dependencies.contains(&operation)) ordered.push_back(&operation);
      // Preserve the preparation chain's original order, including allocations
      // used by later views and producers; all consumers finish before release.
      for (Operation *operation : ordered) operation->moveBefore(loop);
      for (auto preparation : preparations) preparation.end->moveAfter(loop);
      break;
    }
  }
  return success();
}

std::optional<PreparedInput> preparedInput(linalg::GenericOp producer) {
  if (producer.getNumResults() || producer.getOutputs().size() != 1 ||
      producer.getNumReductionLoops() || !producer.getIndexingMapsArray().back().isPermutation() ||
      !producer.getRegion().front().getArguments().back().use_empty()) return std::nullopt;
  for (Operation &operation : producer.getRegion().front().without_terminator())
    if (operation.getNumRegions() || !isMemoryEffectFree(&operation)) return std::nullopt;
  Value output = producer.getOutputs()[0];
  auto allocation = output.getDefiningOp<memref::AllocOp>();
  if (!allocation || allocation->getBlock() != producer->getBlock() ||
      !allocation->isBeforeInBlock(producer)) return std::nullopt;
  StorageAnalysis storage(producer->getParentOfType<func::FuncOp>());
  auto lifetime = storage.lifetime(allocation);
  if (!lifetime || !lifetime->aliases.complete) return std::nullopt;
  PreparedInput result{cast<linalg::LinalgOp>(producer.getOperation()), allocation, lifetime->end, {}};
  for (Operation *user : lifetime->aliases.users) {
    if (user == producer || user == result.end) continue;
    if (auto generic = dyn_cast<linalg::GenericOp>(user)) {
      if (llvm::any_of(generic.getOutputs(), [&](Value memory) {
            return llvm::is_contained(lifetime->aliases.values, memory);
          })) return std::nullopt;
      for (Operation &operation : generic.getRegion().front().without_terminator())
        if (operation.getNumRegions() || !isMemoryEffectFree(&operation)) return std::nullopt;
      if (isMatrixContraction(generic)) {
        auto binding = generic->getAttrOfType<ImplementationAttr>("intent_cpu.implementation");
        if (!binding || (result.consumer && result.consumer != binding)) return std::nullopt;
        result.consumer = binding;
      }
    } else if (!isStorageAliasOperation(user) && !isa<memref::DimOp, memref::LoadOp>(user))
      return std::nullopt;
    Operation *ancestor = producer->getBlock()->findAncestorOpInBlock(*user);
    if (!ancestor || !producer->isBeforeInBlock(ancestor)) return std::nullopt;
  }
  if (!result.consumer) return std::nullopt;
  return result;
}

bool equivalent(PreparedInput &lhs, PreparedInput &rhs, StorageAnalysis &storage) {
  if (lhs.producer->getBlock() != rhs.producer->getBlock() ||
      !lhs.producer->isBeforeInBlock(rhs.producer) || lhs.consumer != rhs.consumer ||
      lhs.allocation.getType() != rhs.allocation.getType() ||
      lhs.allocation->getAttrs() != rhs.allocation->getAttrs()) return false;
  auto maps = lhs.producer.getIndexingMapsArray();
  for (int64_t axis = 0; axis < lhs.allocation.getType().getRank(); ++axis) {
    if (!lhs.allocation.getType().isDynamicDim(axis)) continue;
    unsigned dynamic = lhs.allocation.getType().getDynamicDimIndex(axis);
    if (lhs.allocation.getDynamicSizes()[dynamic] == rhs.allocation.getDynamicSizes()[dynamic]) continue;
    AffineExpr loop = maps.back().getResult(axis);
    if (llvm::none_of(ArrayRef(maps).drop_back(), [&](AffineMap map) {
          return llvm::is_contained(map.getResults(), loop);
        })) return false;
  }
  for (Value input : lhs.producer.getDpsInputs())
    if (isa<MemRefType>(input.getType()) &&
        !storage.readStable(input, lhs.producer, rhs.producer)) return false;
  llvm::DenseMap<Value, Value> pairs;
  pairs[lhs.allocation] = rhs.allocation;
  return OperationEquivalence::isEquivalentTo(lhs.producer, rhs.producer,
      [&](Value left, Value right) { return success(left == right || pairs.lookup(left) == right); },
      [&](Value left, Value right) { pairs[left] = right; }, OperationEquivalence::IgnoreLocations);
}

}

LogicalResult reusePreparedInputs(func::FuncOp function, const ImplementationRegistry &implementations) {
  if (failed(groupScopedInputs(function, implementations))) return failure();
  bool changed;
  do {
    changed = false;
    SmallVector<PreparedInput> supplies;
    function.walk([&](linalg::GenericOp operation) {
      if (auto supply = preparedInput(operation)) supplies.push_back(*supply);
    });
    StorageAnalysis storage(function);
    for (unsigned i = 0; i < supplies.size() && !changed; ++i) {
      for (unsigned j = i + 1; j < supplies.size(); ++j) {
        auto &first = supplies[i], &second = supplies[j];
        if (!equivalent(first, second, storage)) continue;
        // The shared representation remains owned by this lexical scope until
        // both consumer groups finish, before either implementation expands.
        if (first.end->isBeforeInBlock(second.end)) first.end->moveAfter(second.end);
        second.producer.erase();
        second.end.erase();
        second.allocation.getResult().replaceAllUsesWith(first.allocation.getResult());
        second.allocation.erase();
        changed = true;
        break;
      }
    }
  } while (changed);
  return success();
}

}
