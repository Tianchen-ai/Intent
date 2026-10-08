#include "Pointwise.h"
#include "../Reduction/ReductionParameters.h"
#include "Intent/Dialect/GPU/Analysis/IndexRelations.h"
#include "Intent/Dialect/GPU/Analysis/IntegerRanges.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalExpressionBounds.h"
#include "Intent/Dialect/GPU/Analysis/Resources.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Control/Traversal.h"
#include "Intent/Dialect/GPU/Transforms/Value/ExecutionSchema.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"
#include <limits>

using namespace mlir;

namespace intent::gpu::pointwise {
namespace {

bool shapeCarrier(Operation *operation) {
  return isa<BroadcastOp, ReshapeOp, SplatOp>(operation);
}

// This is a nominal budget for simultaneously live data values, not a machine
// register allocation. Shape carriers extend their source lifetime without
// becoming independent payloads; coordinates and predicates are not snapshots.
PhysicalExprAttr livePayloadBudget(Block &block) {
  Builder builder(block.getParentOp()->getContext());
  auto constant = [&](int64_t value) {
    return PhysicalExprAttr::get(builder.getContext(), PhysicalExprKind::Constant,
        value, builder.getStringAttr(""), builder.getArrayAttr({}));
  };
  auto binary = [&](PhysicalExprKind kind, PhysicalExprAttr lhs,
                    PhysicalExprAttr rhs) {
    return PhysicalExprAttr::get(builder.getContext(), kind, 0,
        builder.getStringAttr(""), builder.getArrayAttr({lhs, rhs}));
  };
  llvm::DenseMap<Operation *, unsigned> positions;
  unsigned position = 0;
  for (Operation &operation : block.without_terminator())
    positions[&operation] = position++;
  struct Payload { unsigned first, last; PhysicalExprAttr words; };
  SmallVector<Payload> payloads;
  for (Operation &operation : block.without_terminator()) {
    if (shapeCarrier(&operation)) continue;
    for (Value value : operation.getResults()) {
      auto type = dyn_cast<FragmentType>(value.getType());
      if (!type || type.getElementType().isIndex() ||
          type.getElementType().isInteger(1) || value.use_empty()) continue;
      unsigned last = positions.lookup(&operation);
      SmallVector<Value> pending{value};
      llvm::DenseSet<Value> visited;
      while (!pending.empty()) {
        Value current = pending.pop_back_val();
        if (!visited.insert(current).second) continue;
        for (Operation *user : current.getUsers()) {
          Operation *owner = block.findAncestorOpInBlock(*user);
          if (!owner || !positions.contains(owner)) return {};
          last = std::max(last, positions.lookup(owner));
          if (owner == user && shapeCarrier(user))
            llvm::append_range(pending, user->getResults());
        }
      }
      payloads.push_back({positions.lookup(&operation), last,
                          fragmentRegisterFootprint(type)});
    }
  }
  auto peak = constant(0);
  for (unsigned at = 0; at < position; ++at) {
    auto live = constant(0);
    for (const Payload &payload : payloads)
      if (payload.first <= at && at <= payload.last)
        live = binary(PhysicalExprKind::Add, live, payload.words);
    peak = binary(PhysicalExprKind::Maximum, peak, live);
  }
  return peak;
}

} // namespace

FailureOr<bool> retainCoveredPointwiseGraph(func::FuncOp kernel,
                                           MakeRangeOp range,
                                           ArrayRef<StoreOp> stores) {
  if (stores.empty() || !isUnitStepRange(range) ||
      !samePhysicalScalarExpression(range.getStart(), range.getLogicalStart()))
    return false;
  auto dimension = queryRangeDimension(range);
  auto type = range.getResult().getType();
  if (failed(dimension) || type.getShape().size() != 1 ||
      !PhysicalProgramAnalysis(kernel).axisRealization(range.getResult(), 0).constructionScalarSeed)
    return false;
  // Existing author control is kept outside this local decision. In particular,
  // no ordered loop or structured helper is duplicated with its state/captures.
  for (Operation *parent = range->getParentOp(); parent && parent != kernel;
       parent = parent->getParentOp())
    if (!isa<ExecutionGroupOp>(parent)) return false;
  IndexRelations relations;
  if (!relations.nonnegative(range.getStart()) ||
      !relations.nonnegative(range.getLogicalStop())) return false;
  Block *block = range->getBlock();
  Operation *last = range;
  for (StoreOp store : stores) {
    if (store->getBlock() != block || !range->isBeforeInBlock(store)) return false;
    if (last->isBeforeInBlock(store)) last = store;
  }
  SmallVector<Operation *> operations;
  llvm::SmallPtrSet<Operation *, 32> selected;
  for (Operation *operation = range; operation; operation = operation->getNextNode()) {
    operations.push_back(operation);
    selected.insert(operation);
    if (operation == last) break;
  }
  DominanceInfo dominance(kernel);
  PhysicalProgramAnalysis analysis(kernel);
  PhysicalSourceAxis source = sourceAxisIdentity(range);
  ReduceOp firstReduction;
  unsigned firstAxis = 0;
  bool sharedMember = false, sawWrite = false;
  for (Operation *operation : operations) {
    if (auto store = dyn_cast<StoreOp>(operation)) {
      if (!llvm::is_contained(stores, store)) return false;
      sawWrite = true;
    } else if (isa<LoadOp>(operation)) {
      if (sawWrite) return false;
    } else if (!isMemoryEffectFree(operation)) {
      return false;
    }
    if (operation->getNumRegions() && !isa<ReduceOp>(operation)) return false;
    if (auto other = dyn_cast<MakeRangeOp>(operation); other && other != range) {
      auto fact = analysis.axisRealization(other.getResult(), 0);
      if (!fact.isExact() || !fact.physicalized || fact.constructionScalarSeed)
        return false;
    }
    if (auto reduce = dyn_cast<ReduceOp>(operation)) {
      if (reduce.getAxes().size() != 1) return false;
      unsigned axis = reduce.getAxes().front();
      for (Value input : reduce.getSources()) {
        auto facts = analysis.axisRanges(input, axis);
        if (!facts.isExact() || !facts.blockers.empty() || facts.roots.empty() ||
            !llvm::all_of(facts.roots, [&](MakeRangeOp root) { return root == range; }))
          return false;
      }
      if (!firstReduction) { firstReduction = reduce; firstAxis = axis; }
    }
    SmallVector<Value> inputs(operation->getOperands());
    for (Region &region : operation->getRegions()) {
      llvm::SetVector<Value> captures;
      getUsedValuesDefinedAbove(region, captures);
      llvm::append_range(inputs, captures);
    }
    for (Value input : inputs) {
      Operation *producer = input.getDefiningOp();
      if ((!producer || !selected.contains(producer)) &&
          !dominance.properlyDominates(input, range)) return false;
    }
    for (Value value : operation->getResults()) {
      if (isa<RecordType>(value.getType())) return false;
      for (Operation *user : value.getUsers()) {
        Operation *owner = block->findAncestorOpInBlock(*user);
        if (!owner || !selected.contains(owner)) return false;
      }
      auto fragment = dyn_cast<FragmentType>(value.getType());
      sharedMember |= fragment && !fragment.getElementType().isIndex() &&
          !fragment.getElementType().isInteger(1) &&
          !shapeCarrier(operation) && !value.hasOneUse() &&
          containsTraversal(value, source, ArrayRef<int64_t>{*dimension});
    }
  }
  if (!firstReduction || !sharedMember) return false;
  for (StoreOp store : stores) {
    auto dependency = analysis.reductionDependency(store.getValue(), source, *dimension);
    if (!dependency.isExact() || !dependency.depends || dependency.throughStructuredReduction)
      return false;
  }

  DictionaryAttr originalAttributes = kernel->getAttrDictionary();
  auto chunk = reduction::selectReductionTraversalChunk(
      firstReduction, firstReduction.getSources().front(), firstAxis, range);
  if (failed(chunk)) return failure();
  auto extent = boundedTraversalChunk(*chunk, range);
  if (failed(extent)) return failure();
  auto widthBounds = queryPhysicalExpressionRange(*extent, kernel);
  auto startBounds = queryIntegerRange(range.getStart());
  if (!widthBounds || !widthBounds->smin().isStrictlyPositive() || !startBounds ||
      static_cast<__int128>(startBounds->smax().getSExtValue()) +
              widthBounds->smax().getSExtValue() - 1 > std::numeric_limits<int64_t>::max()) {
    kernel->setAttrs(originalAttributes);
    return false;
  }

  Operation *previous = range->getPrevNode();
  auto rollback = [&]() {
    while (range->getPrevNode() != previous) range->getPrevNode()->erase();
    kernel->setAttrs(originalAttributes);
  };
  OpBuilder builder(range);
  Location location = range.getLoc();
  Value width = builder.create<PhysicalExprOp>(location, builder.getIndexType(), *extent);
  Value span = builder.create<BinaryOp>(location, builder.getIndexType(),
      range.getLogicalStop(), range.getLogicalStart(), BinaryOperator::Subtract);
  Value ordered = builder.create<CompareOp>(location, builder.getI1Type(),
      range.getLogicalStart(), range.getLogicalStop(), ComparePredicate::Le);
  Value covered = builder.create<CompareOp>(location, builder.getI1Type(),
      span, width, ComparePredicate::Le);
  Value condition = builder.create<BinaryOp>(location, builder.getI1Type(),
      ordered, covered, BinaryOperator::LogicalAnd);
  auto choice = builder.create<scf::IfOp>(location, condition, true);
  OpBuilder nested(choice.thenBlock()->getTerminator());
  auto selectedType = cast<FragmentType>(replaceTraversalExtent(
      type, source, ArrayRef<int64_t>{*dimension}, *extent));
  auto current = nested.create<MakeRangeOp>(location, selectedType,
      range.getStart(), width, range.getStep(), range.getLogicalStart(),
      range.getLogicalStop(), range.getSourceId(), range.getSourceAxis(), range.getDerived());
  inheritRangeAuthority(current, range);
  IRMapping mapping;
  mapping.map(range.getResult(), current.getResult());
  for (Operation *operation : llvm::drop_begin(operations)) {
    auto cloned = cloneWithPhysicalSchema(nested, operation, mapping, [&](Value value) {
      return replaceTraversalExtent(value.getType(), source,
                                     ArrayRef<int64_t>{*dimension}, *extent);
    });
    if (failed(cloned)) { rollback(); return false; }
  }
  nested.setInsertionPointAfter(current);
  Value tail = nested.create<CompareOp>(location, predicateType(selectedType), current,
      nested.create<SplatOp>(location, selectedType, range.getLogicalStop()), ComparePredicate::Lt);
  llvm::DenseMap<Value, Value> tails{{current.getResult(), tail}};
  if (failed(addTailValidity(kernel, choice, tails, /*includeStores=*/true))) {
    rollback();
    return failure();
  }
  auto footprint = livePayloadBudget(*choice.thenBlock());
  auto footprintBounds = footprint ? queryPhysicalExpressionRange(footprint, kernel)
                                   : std::nullopt;
  if (!footprintBounds || footprintBounds->smin().isNegative()) {
    rollback();
    return false;
  }
  builder.setInsertionPoint(choice);
  Value words = builder.create<PhysicalExprOp>(location, builder.getIndexType(), footprint);
  Value budget = builder.create<arith::ConstantIndexOp>(location,
      kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr).getRegistersPerUnit());
  Value fits = builder.create<CompareOp>(location, builder.getI1Type(),
      words, budget, ComparePredicate::Lt);
  choice.getConditionMutable().assign(builder.create<BinaryOp>(location,
      builder.getI1Type(), condition, fits, BinaryOperator::LogicalAnd));
  for (Operation *operation : operations)
    operation->moveBefore(choice.elseBlock()->getTerminator());
  return true;
}

} // namespace intent::gpu::pointwise
