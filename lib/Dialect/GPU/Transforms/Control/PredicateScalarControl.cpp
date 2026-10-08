#include "PredicationDetail.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "Intent/Dialect/GPU/Transforms/Control/Predication.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <limits>

using namespace mlir;

namespace intent::gpu {
using namespace predication;
namespace {

bool isScalar(Type type) {
  return isa<IntegerType, IndexType, FloatType>(type);
}

bool isPredicatableProduct(Type type) {
  if (auto record = dyn_cast<RecordType>(type))
    return llvm::all_of(record.getFieldTypes(), [](Attribute field) {
      return isPredicatableProduct(cast<TypeAttr>(field).getValue());
    });
  return isScalar(type) || isa<FragmentType>(type);
}

bool dependsOnWorksetCoordinate(Value value) {
  SmallVector<Value> worklist{value};
  llvm::SmallPtrSet<Operation *, 16> visited;
  while (!worklist.empty()) {
    Operation *producer = worklist.pop_back_val().getDefiningOp();
    if (!producer || !visited.insert(producer).second)
      continue;
    if (isa<WorksetCoordinateOp>(producer))
      return true;
    llvm::append_range(worklist, producer->getOperands());
  }
  return false;
}

bool canPredicate(Block &block, bool allowStores = false,
                  bool allowProducts = false, bool allowLoops = false,
                  bool allowAssumptions = false) {
  for (Operation &operation : block.without_terminator()) {
    if (auto assumption = dyn_cast<AssumeInBoundsOp>(operation)) {
      if (!allowAssumptions || !isScalar(assumption.getIndex().getType()))
        return false;
      continue;
    }
    if (auto loop = dyn_cast<scf::ForOp>(operation)) {
      APInt lower, upper, step;
      if (!allowLoops ||
          !matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) ||
          !matchPattern(loop.getUpperBound(), m_ConstantInt(&upper)) ||
          !matchPattern(loop.getStep(), m_ConstantInt(&step)) ||
          !step.isStrictlyPositive() ||
          !llvm::all_of(loop.getResultTypes(), isPredicatableProduct) ||
          !canPredicate(*loop.getBody(), /*allowStores=*/false,
                        /*allowProducts=*/true, /*allowLoops=*/true))
        return false;
      continue;
    }
    if (auto loop = dyn_cast<scf::WhileOp>(operation)) {
      if (!allowLoops || !canPredicateScalarWhile(loop))
        return false;
      continue;
    }
    if (auto store = dyn_cast<StoreOp>(operation)) {
      if (!allowStores || !isScalar(store.getValue().getType()) ||
          !llvm::all_of(store.getCoordinates(), [&](Value coordinate) {
            return isScalar(coordinate.getType()) &&
                   (!allowAssumptions || coordinate.getType().isIntOrIndex());
          }))
        return false;
      continue;
    }
    if (allowProducts &&
        isa<ContractOp, ReduceOp, ReshapeOp, TransposeOp>(operation)) {
      if (!llvm::all_of(operation.getResultTypes(), isPredicatableProduct) ||
          !isMemoryEffectFree(&operation))
        return false;
      if (auto reduce = dyn_cast<ReduceOp>(operation))
        if (!canPredicate(reduce.getCombine().front(), false, true, false))
          return false;
      continue;
    }
    if (operation.getNumRegions() || !operation.getNumResults() ||
        !llvm::all_of(operation.getResultTypes(), [&](Type type) {
          return allowProducts ? isPredicatableProduct(type) : isScalar(type);
        }))
      return false;
    ValueRange coordinates;
    if (auto access = dyn_cast<AccessOpInterface>(operation);
        access && (access.getAccessKind() == AccessKind::Load ||
                   access.getAccessKind() == AccessKind::Gather))
      coordinates = access.getAccessCoordinates();
    if (!llvm::all_of(coordinates, [&](Value coordinate) {
          if (allowAssumptions && !coordinate.getType().isIntOrIndex())
            return false;
          return allowProducts ? isPredicatableProduct(coordinate.getType())
                               : isScalar(coordinate.getType());
        }))
      return false;
    bool product = allowProducts && isa<MakeRecordOp, ExtractOp>(operation) &&
                   isSpeculatable(&operation) && isMemoryEffectFree(&operation);
    auto conversion = dyn_cast<CastOp>(operation);
    bool guardedConversion = conversion && isFloatToIntegerCast(conversion);
    if (!product && !guardedConversion && !canPredicateValueOperation(&operation))
      return false;
  }
  return true;
}

void dischargeScalarAssumptions(Block &block) {
  SmallVector<AssumeInBoundsOp> assumptions;
  for (Operation &operation : block.without_terminator())
    if (auto assumption = dyn_cast<AssumeInBoundsOp>(operation))
      assumptions.push_back(assumption);
  if (assumptions.empty())
    return;
  // A fact inside a branch is not true for inactive lanes after if-conversion.
  // Close the accesses with explicit validity before removing that lexical fact.
  // These guards are true for every active access of a legal source program.
  for (Operation &operation : llvm::make_early_inc_range(block.without_terminator())) {
    auto access = dyn_cast<AccessOpInterface>(operation);
    if (!access || (access.getAccessKind() != AccessKind::Load &&
                    access.getAccessKind() != AccessKind::Store))
      continue;
    Value resource = access.getAccessResource();
    SmallVector<Value> coordinates(access.getAccessCoordinates());
    if (coordinates.empty())
      continue;
    ArrayRef<int64_t> axes = access.getAccessSourceAxes();
    Value valid = access.getAccessValidity();
    OpBuilder builder(&operation);
    Location location = operation.getLoc();
    Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
    for (unsigned position = 0; position < coordinates.size(); ++position) {
      Value &coordinate = coordinates[position];
      int64_t axis = axes[position];
      if (!coordinate.getType().isIndex())
        coordinate = builder.create<CastOp>(location, builder.getIndexType(), coordinate);
      Value extent;
      if (auto buffer = dyn_cast<BufferType>(resource.getType()))
        extent = builder.create<PhysicalExprOp>(location, builder.getIndexType(),
            cast<PhysicalExprAttr>(buffer.getShape()[axis]));
      else
        extent = builder.create<DimOp>(location, builder.getIndexType(), resource, axis);
      Value lower = builder.create<CompareOp>(location, builder.getI1Type(),
          coordinate, zero, ComparePredicate::Ge);
      Value upper = builder.create<CompareOp>(location, builder.getI1Type(),
          coordinate, extent, ComparePredicate::Lt);
      Value bounded = builder.create<BinaryOp>(location, builder.getI1Type(),
          lower, upper, BinaryOperator::LogicalAnd);
      valid = valid ? Value(builder.create<BinaryOp>(location, builder.getI1Type(),
          valid, bounded, BinaryOperator::LogicalAnd)) : bounded;
    }
    access.getAccessCoordinatesMutable().assign(coordinates);
    access.getAccessValidityMutable().assign(valid);
    if (auto fill = access.getAccessFillMutable(); fill && !access.getAccessFill())
      fill->assign(builder.create<arith::ConstantOp>(
          location, builder.getZeroAttr(access.getAccessValueType())).getResult());
  }
  for (AssumeInBoundsOp assumption : assumptions)
    assumption.erase();
}

FailureOr<SmallVector<Value>> predicateBlock(OpBuilder &builder, Block &block,
                                            Value predicate) {
  IRMapping mapping;
  for (Operation &operation : block.without_terminator())
    if (failed(clonePredicatedScalarOperation(builder, &operation, mapping,
                                              predicate)))
      return failure();
  SmallVector<Value> results;
  for (Value value : block.getTerminator()->getOperands())
    results.push_back(mapping.lookupOrDefault(value));
  return results;
}

} // namespace

bool canPredicateScalarBlock(Block &block) {
  return canPredicate(block, /*allowStores=*/true);
}

bool canPredicateScalarWhile(scf::WhileOp loop) {
  if (!loop.getNumResults() || !llvm::hasSingleElement(loop.getBefore()) ||
      !llvm::hasSingleElement(loop.getAfter()) ||
      !llvm::all_of(loop.getResultTypes(), isScalar))
    return false;
  Block &before = loop.getBefore().front();
  Block &after = loop.getAfter().front();
  auto condition = dyn_cast<scf::ConditionOp>(before.getTerminator());
  if (!condition || before.getArgumentTypes() != loop.getResultTypes() ||
      after.getArgumentTypes() != loop.getResultTypes() ||
      !llvm::equal(condition.getArgs(), before.getArguments()))
    return false;
  // Only pure per-element recurrences are coarsened here. Cross-lane effects,
  // nested regions and changes to the state in the condition need other proofs.
  return llvm::all_of(ArrayRef<Block *>{&before, &after}, [&](Block *block) {
    return canPredicate(*block) &&
           llvm::all_of(block->without_terminator(), [](Operation &operation) {
             return isMemoryEffectFree(&operation);
           });
  });
}

LogicalResult predicateScalarControl(ModuleOp module) {
  FailureOr<func::FuncOp> kernel = getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  SmallVector<scf::IfOp> conditionals;
  kernel->walk<WalkOrder::PostOrder>(
      [&](scf::IfOp conditional) { conditionals.push_back(conditional); });
  for (scf::IfOp conditional : conditionals) {
    if (!conditional.getNumResults()) {
      bool emptyElse = conditional.getElseRegion().empty() ||
          conditional.getElseRegion().front().without_terminator().empty();
      if (!emptyElse ||
          !canPredicate(conditional.getThenRegion().front(),
                        /*allowStores=*/true, /*allowProducts=*/false,
                        /*allowLoops=*/false, /*allowAssumptions=*/true))
        continue;
      dischargeScalarAssumptions(conditional.getThenRegion().front());
      OpBuilder builder(conditional);
      if (failed(predicateBlock(builder, conditional.getThenRegion().front(),
                                conditional.getCondition())))
        return failure();
      conditional.erase();
      continue;
    }
    if (conditional.getElseRegion().empty() ||
        !llvm::all_of(conditional.getResultTypes(), isPredicatableProduct))
      continue;
    bool scalarEffects =
        llvm::all_of(conditional.getResultTypes(), isScalar) &&
        canPredicate(conditional.getThenRegion().front(), true, false, false, true) &&
        canPredicate(conditional.getElseRegion().front(), true, false, false, true);
    if (!scalarEffects &&
        (!canPredicate(conditional.getThenRegion().front(), false, true, true) ||
         !canPredicate(conditional.getElseRegion().front(), false, true, true)))
      continue;
    bool hasFragment = false;
    for (Type type : conditional.getResultTypes())
      type.walk([&](FragmentType) { hasFragment = true; });
    // Keep launch-wide choices lazy even before their scalar values are
    // lifted to fragments; inactive transcendental/loop paths can be costly.
    if (isLaunchUniformScalar(conditional.getCondition(), *kernel) ||
        (hasFragment && !dependsOnWorksetCoordinate(conditional.getCondition())))
      continue;
    if (scalarEffects) {
      dischargeScalarAssumptions(conditional.getThenRegion().front());
      dischargeScalarAssumptions(conditional.getElseRegion().front());
    }
    OpBuilder builder(conditional);
    Value otherwise = builder.create<UnaryOp>(
        conditional.getLoc(), builder.getI1Type(), conditional.getCondition(),
        UnaryOperator::Not);
    auto thenValues = predicateBlock(
        builder, conditional.getThenRegion().front(), conditional.getCondition());
    auto elseValues = predicateBlock(
        builder, conditional.getElseRegion().front(), otherwise);
    if (failed(thenValues) || failed(elseValues)) return failure();
    for (auto [result, thenValue, elseValue] :
         llvm::zip(conditional.getResults(), *thenValues, *elseValues)) {
      auto replacement = selectScalarProduct(
          builder, conditional.getLoc(), conditional.getCondition(), thenValue, elseValue);
      if (failed(replacement)) return failure();
      if (Attribute origin = conditional->getAttr(originAttr))
        (*replacement).getDefiningOp()->setAttr(originAttr, origin);
      result.replaceAllUsesWith(*replacement);
    }
    conditional.erase();
  }
  eraseDeadPhysicalValues(*kernel);
  return success();
}

LogicalResult guardInactivePredicatedLoops(ModuleOp module) {
  FailureOr<func::FuncOp> kernel = getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  SmallVector<scf::ForOp> loops;
  kernel->walk<WalkOrder::PostOrder>([&](scf::ForOp loop) { loops.push_back(loop); });
  for (scf::ForOp loop : loops) {
    APInt lower, upper, step;
    if (!loop.getNumResults() || !isMemoryEffectFree(loop) ||
        llvm::any_of(loop.getBody()->without_terminator(), [](Operation &nested) {
          return nested.getNumRegions() != 0;
        }) ||
        !matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) ||
        !matchPattern(loop.getUpperBound(), m_ConstantInt(&upper)) ||
        !matchPattern(loop.getStep(), m_ConstantInt(&step)) ||
        lower.getBitWidth() > 64 || upper.getBitWidth() > 64 ||
        step.getBitWidth() > 64 || !step.isStrictlyPositive() ||
        static_cast<__int128>(upper.getSExtValue()) - lower.getSExtValue() <=
            step.getSExtValue() ||
        static_cast<__int128>(upper.getSExtValue()) + step.getSExtValue() - 1 >
            std::numeric_limits<int64_t>::max())
      continue;
    // If-conversion freezes an inactive recurrence at its incoming value.
    // Recover that loop-invariant predicate from the carry selects; when every
    // lane is inactive, a finite effect-free loop is exactly its initial state.
    SmallVector<SmallVector<std::pair<Value, bool>>> guards;
    Type predicateType;
    bool supported = true;
    for (auto [yielded, carried] :
         llvm::zip(loop.getBody()->getTerminator()->getOperands(),
                   loop.getRegionIterArgs())) {
      if (yielded == carried)
        continue;
      SmallVector<std::pair<Value, bool>> conjunction;
      while (auto select = yielded.getDefiningOp<SelectOp>()) {
        bool inverted = select.getTrueValue() == carried;
        if ((!inverted && select.getFalseValue() != carried) ||
            !loop.isDefinedOutsideOfLoop(select.getCondition()))
          break;
        Type type = select.getCondition().getType();
        auto fragment = dyn_cast<FragmentType>(type);
        if (!fragment || fragment.getShape().empty() ||
            (predicateType && predicateType != type))
          break;
        predicateType = type;
        conjunction.emplace_back(select.getCondition(), inverted);
        yielded = inverted ? select.getFalseValue() : select.getTrueValue();
      }
      if (conjunction.empty()) {
        supported = false;
        break;
      }
      guards.push_back(std::move(conjunction));
    }
    if (!supported || guards.empty())
      continue;
    OpBuilder builder(loop);
    Location location = loop.getLoc();
    Value active;
    for (auto &conjunction : guards) {
      Value selected;
      for (auto [condition, inverted] : conjunction) {
        if (inverted)
          condition = builder.create<UnaryOp>(location, predicateType, condition,
                                               UnaryOperator::Not);
        selected = selected ? Value(builder.create<BinaryOp>(
                                  location, predicateType, selected, condition,
                                  BinaryOperator::LogicalAnd)) : condition;
      }
      active = active ? Value(builder.create<BinaryOp>(
                            location, predicateType, active, selected,
                            BinaryOperator::LogicalOr)) : selected;
    }
    auto conditional = builder.create<scf::IfOp>(
        location, loop.getResultTypes(), anyActiveLane(builder, location, active),
        /*withElseRegion=*/true);
    loop->replaceAllUsesWith(conditional.getResults());
    Block &thenBody = conditional.getThenRegion().front();
    loop->moveBefore(&thenBody, thenBody.begin());
    builder.setInsertionPointToEnd(&thenBody);
    builder.create<scf::YieldOp>(location, loop.getResults());
    builder.setInsertionPointToStart(&conditional.getElseRegion().front());
    builder.create<scf::YieldOp>(location, loop.getInitArgs());
  }
  return success();
}

} // namespace intent::gpu
