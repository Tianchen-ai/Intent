#include "ScopePlacement.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"

#include "Intent/Dialect/GPU/Analysis/IndexRelations.h"
#include "Intent/Dialect/GPU/Analysis/ConfigurationExpressions.h"
#include "Intent/Dialect/GPU/Analysis/MemoryEffects.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/ResourceAlias.h"
#include "Intent/Dialect/GPU/Analysis/Resources.h"
#include "Intent/Dialect/GPU/IR/AccessOpInterface.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/IR/ProgramInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include <algorithm>
#include <functional>

using namespace mlir;

namespace intent::gpu::placement {
namespace {

bool hasUnorderedEffects(Operation *operation) {
  return !operation->walk([](Operation *nested) {
    auto access = dyn_cast<AccessOpInterface>(nested);
    if (!access || access.getAccessKind() == AccessKind::Store ||
        hasOnlyReadEffects(nested))
      return WalkResult::advance();
    return WalkResult::interrupt();
  }).wasInterrupted();
}

bool shapeOnly(Operation *operation) {
  return isa<BroadcastOp, SplatOp, ReshapeOp, TransposeOp, JoinOp,
             MakeRecordOp, ExtractOp, arith::ConstantOp>(operation);
}

Value disjointCondition(OpBuilder &builder, Location location,
                        func::FuncOp kernel, Value first, Value second) {
  Value overlap;
  for (ViewOverlapOp fact : kernel.getOps<ViewOverlapOp>())
    if ((fact.getLhs() == first && fact.getRhs() == second) ||
        (fact.getLhs() == second && fact.getRhs() == first)) {
      overlap = fact.getResult();
      break;
    }
  if (!overlap) {
    OpBuilder entry(&kernel.front(), kernel.front().begin());
    overlap = entry.create<ViewOverlapOp>(location, entry.getI1Type(), first, second);
  }
  Value zero = builder.create<arith::ConstantIntOp>(location, 0, 1);
  return builder.create<CompareOp>(location, builder.getI1Type(), overlap, zero,
                                    ComparePredicate::Eq);
}

} // namespace

LogicalResult hoistLoopInvariantValues(func::FuncOp kernel) {
  SmallVector<LoopLikeOpInterface> loops;
  kernel.walk<WalkOrder::PostOrder>([&](LoopLikeOpInterface loop) { loops.push_back(loop); });
  auto capabilities = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  for (LoopLikeOpInterface loopLike : loops) {
    auto loop = dyn_cast<scf::ForOp>(loopLike.getOperation());
    auto live = loop ? nominalLoopWorkingSet(loop) : PhysicalExprAttr();
    int64_t budget = capabilities ? capabilities.getRegistersPerUnit() : 0;
    Builder builder(kernel.getContext());
    auto limit = PhysicalExprAttr::get(kernel.getContext(), PhysicalExprKind::Constant,
        budget - 1, builder.getStringAttr(""), builder.getArrayAttr({}));
    auto fits = [&](PhysicalExprAttr words) {
      return words && budget > 0 && configurationExpressionAtMost(kernel, words, limit);
    };
    auto reserve = [&](TypeRange types) {
      if (!fits(live)) return false;
      if (types.empty()) return true;
      auto extra = nominalPayloadWords(types);
      if (!extra) return false;
      auto total = PhysicalExprAttr::get(kernel.getContext(), PhysicalExprKind::Add,
          0, builder.getStringAttr(""), builder.getArrayAttr({live, extra}));
      if (!fits(total)) return false;
      live = total;
      return true;
    };
    moveLoopInvariantCode(loopLike.getLoopRegions(),
        [&](Value value, Region *) { return loopLike.isDefinedOutsideOfLoop(value); },
        [&](Operation *operation, Region *) {
      if (!isMovableValueOperation(operation)) return false;
      if (shapeOnly(operation)) return true;
      return loop &&
          llvm::all_of(operation->getOperands(), [&](Value value) {
            return loopLike.isDefinedOutsideOfLoop(value);
          }) &&
          (isa<MakeRangeOp>(operation) ||
           isPhysicalReplayNode(operation, PhysicalReplayScope::Coordinate, false) ||
           operation->hasTrait<OpTrait::Elementwise>()) && reserve(operation->getResultTypes());
    }, [&](Operation *operation, Region *) { loopLike.moveOutOfLoop(operation); });
    if (!loop || !fits(live)) continue;

    SmallVector<LoadOp> loads;
    for (LoadOp load : loop.getBody()->getOps<LoadOp>()) loads.push_back(load);
    ResourceAliasAnalysis aliases;
    for (LoadOp load : loads) {
      if (!llvm::all_of(load->getOperands(), [&](Value operand) {
            return loopLike.isDefinedOutsideOfLoop(operand);
          })) continue;
      SmallVector<std::pair<Value, Value>> guarded;
      auto disjoint = [&](Value first, Value second) {
        auto lhs = dyn_cast<BlockArgument>(first), rhs = dyn_cast<BlockArgument>(second);
        if (!lhs || !rhs || lhs == rhs || lhs.getOwner() != &kernel.front() ||
            rhs.getOwner() != &kernel.front() || !getPublicView(first) || !getPublicView(second))
          return false;
        if (lhs.getArgNumber() > rhs.getArgNumber()) std::swap(first, second);
        std::pair<Value, Value> pair{first, second};
        if (!llvm::is_contained(guarded, pair)) guarded.push_back(pair);
        return true;
      };
      if (!preservesMemoryReads(load, loop, aliases, disjoint) ||
          !reserve(load->getResultTypes())) continue;
      bool nonempty = IndexRelations().lessThan(loop.getLowerBound(), loop.getUpperBound());
      if (nonempty && guarded.empty()) {
        load->moveBefore(loop);
        continue;
      }
      OpBuilder builder(loop);
      Location location = load.getLoc();
      Value safe = builder.create<CompareOp>(location, builder.getI1Type(),
          loop.getLowerBound(), loop.getUpperBound(), ComparePredicate::Lt);
      for (auto [first, second] : guarded) {
        Value condition = disjointCondition(builder, location, kernel, first, second);
        safe = builder.create<BinaryOp>(location, builder.getI1Type(), safe, condition,
                                         BinaryOperator::LogicalAnd);
      }
      auto zero = materializeZeroValue(builder, location, load.getType());
      if (failed(zero)) return failure();
      auto saved = builder.create<scf::IfOp>(location, load->getResultTypes(), safe, true);
      builder.setInsertionPointToEnd(saved.thenBlock());
      Operation *read = builder.clone(*load);
      builder.create<scf::YieldOp>(location, read->getResults());
      builder.setInsertionPointToEnd(saved.elseBlock());
      builder.create<scf::YieldOp>(location, ValueRange{*zero});

      // The original read stays at its original point on the aliasing path.
      // Only a direct loop-body read is selected, so subsequent fixed-point
      // invocations cannot wrap this conditional read again.
      builder.setInsertionPoint(load);
      auto selected = builder.create<scf::IfOp>(location, load->getResultTypes(), safe, true);
      load.getResult().replaceAllUsesWith(selected.getResult(0));
      builder.setInsertionPointToEnd(selected.thenBlock());
      builder.create<scf::YieldOp>(location, saved.getResults());
      load->moveBefore(selected.elseBlock(), selected.elseBlock()->end());
      builder.setInsertionPointToEnd(selected.elseBlock());
      builder.create<scf::YieldOp>(location, load->getResults());
    }
  }
  return success();
}

bool independentMemoryEffects(Operation *first, Operation *second,
                              ResourceAliasAnalysis &aliases) {
  if (!hasUnorderedEffects(first) || !hasUnorderedEffects(second))
    return false;
  auto firstEffects = getEffectsRecursively(first);
  auto secondEffects = getEffectsRecursively(second);
  if (!firstEffects || !secondEffects)
    return false;
  auto understood = [](const MemoryEffects::EffectInstance &effect) {
    return isa<MemoryEffects::Read, MemoryEffects::Write,
               MemoryEffects::Allocate, MemoryEffects::Free>(effect.getEffect());
  };
  if (!llvm::all_of(*firstEffects, understood) ||
      !llvm::all_of(*secondEffects, understood))
    return false;
  for (const auto &lhs : *firstEffects) {
    for (const auto &rhs : *secondEffects) {
      if (lhs.getResource() != rhs.getResource() ||
          (isa<MemoryEffects::Read>(lhs.getEffect()) &&
           isa<MemoryEffects::Read>(rhs.getEffect())))
        continue;
      if (!lhs.getValue() || !rhs.getValue() ||
          !aliases.alias(lhs.getValue(), rhs.getValue()).isNo())
        return false;
    }
  }
  return true;
}

bool isMovableValueOperation(Operation *operation) {
  return operation->getNumRegions() == 0 && operation->getNumResults() != 0 &&
         isMemoryEffectFree(operation) && isSpeculatable(operation);
}

bool canMoveBefore(Operation *operation, Operation *before) {
  if (!operation || !before || operation == before ||
      operation->getBlock() != before->getBlock())
    return false;
  if (isMovableValueOperation(operation))
    return true;
  auto load = dyn_cast<LoadOp>(operation);
  return load && canReplayReadAt(load, before);
}

bool moveInputsBefore(Operation *first, Operation *second,
                      func::FuncOp kernel) {
  DominanceInfo dominance(kernel);
  SmallVector<Operation *> hoist;
  llvm::DenseSet<Value> visited;
  std::function<bool(Value)> available = [&](Value value) {
    if (dominance.properlyDominates(value, first))
      return true;
    if (!visited.insert(value).second)
      return true;
    Operation *producer = value.getDefiningOp();
    if (!producer || producer->getBlock() != first->getBlock() ||
        !first->isBeforeInBlock(producer) ||
        !producer->isBeforeInBlock(second) ||
        !canMoveBefore(producer, first) ||
        !llvm::all_of(producer->getOperands(), available))
      return false;
    if (!llvm::is_contained(hoist, producer))
      hoist.push_back(producer);
    return true;
  };
  llvm::SetVector<Value> captures;
  for (Region &region : second->getRegions())
    getUsedValuesDefinedAbove(region, region, captures);
  if (!llvm::all_of(second->getOperands(), available) ||
      !llvm::all_of(captures, available))
    return false;
  for (Operation *operation : hoist)
    operation->moveBefore(first);
  return true;
}

FailureOr<SmallVector<Block *>> conditionalPath(Block *scope,
                                                Operation *operation) {
  SmallVector<Block *> path;
  for (Block *block = operation->getBlock(); block != scope;) {
    auto branch = dyn_cast_or_null<scf::IfOp>(block->getParentOp());
    if (!branch || (block->getParent() != &branch.getThenRegion() &&
                    block->getParent() != &branch.getElseRegion()))
      return failure();
    path.push_back(block);
    block = branch->getBlock();
  }
  std::reverse(path.begin(), path.end());
  return path;
}

bool canMoveEffectBefore(Operation *operation, Operation *before,
                         ResourceAliasAnalysis &aliases) {
  if (failed(conditionalPath(before->getBlock(), operation)))
    return false;
  Operation *boundary = operation;
  while (boundary->getBlock() != before->getBlock()) {
    for (Operation &preceding : *boundary->getBlock()) {
      if (&preceding == boundary)
        break;
      if (!independentMemoryEffects(operation, &preceding, aliases))
        return false;
    }
    boundary = boundary->getParentOp();
  }
  if (!before->isBeforeInBlock(boundary))
    return false;
  for (Operation *preceding = before; preceding != boundary;
       preceding = preceding->getNextNode())
    if (!independentMemoryEffects(operation, preceding, aliases))
      return false;
  return true;
}

LogicalResult ConditionalPlacement::emit(
    OpBuilder &builder, ArrayRef<Block *> path,
    llvm::function_ref<FailureOr<Value>(OpBuilder &, Operation *)> condition,
    llvm::function_ref<LogicalResult(OpBuilder &)> body) {
  OpBuilder::InsertionGuard restore(builder);
  for (Block *block : path) {
    auto original = cast<scf::IfOp>(block->getParentOp());
    scf::IfOp replacement;
    if (Operation *existing = branches.lookup(original)) {
      if (existing->getBlock() != builder.getInsertionBlock() ||
          (builder.getInsertionPoint() != builder.getInsertionBlock()->end() &&
           !existing->isBeforeInBlock(&*builder.getInsertionPoint())))
        return failure();
      replacement = cast<scf::IfOp>(existing);
    } else {
      auto predicate = condition(builder, original);
      if (failed(predicate) || !(*predicate).getType().isInteger(1))
        return failure();
      replacement = builder.create<scf::IfOp>(
          original.getLoc(), *predicate, /*withElseRegion=*/true);
      replacement->setDiscardableAttrs(original->getDiscardableAttrDictionary());
      branches[original] = replacement;
    }
    bool isThen = block->getParent() == &original.getThenRegion();
    Block &target = isThen ? replacement.getThenRegion().front()
                           : replacement.getElseRegion().front();
    builder.setInsertionPoint(target.getTerminator());
  }
  return body(builder);
}

} // namespace intent::gpu::placement
