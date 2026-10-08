#include "PassDetail.h"
#include "Intent/Dialect/DSA/Transforms/LocalSupplyRelations.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace intent::bangc {
namespace {

// Supply placement runs before asynchronous transfer scheduling. It moves only
// immutable, synchronous local definitions; StorageAnalysis remains the source
// of alias, ordering and completion facts.
Value suppliedValue(Operation *operation) {
  if (auto op = dyn_cast<dsa::LoadTileOp>(operation)) return op.getOutput();
  if (auto op = dyn_cast<dsa::FillOp>(operation)) return op.getOutput();
  if (auto op = dyn_cast<dsa::CastOp>(operation)) return op.getOutput();
  if (auto op = dyn_cast<dsa::UnaryOp>(operation)) return op.getOutput();
  if (auto op = dyn_cast<dsa::BinaryOp>(operation)) return op.getOutput();
  if (auto op = dyn_cast<dsa::TransposeOp>(operation)) return op.getOutput();
  if (auto op = dyn_cast<dsa::PrepareMatrixOp>(operation)) return op.getOutput();
  return {};
}

std::optional<int64_t> tripCount(scf::ForOp loop,
                                dsa::LocalSupplyRelations &relations) {
  if (!loop.getInductionVar().getType().isIndex()) return std::nullopt;
  auto span = dyn_cast<AffineConstantExpr>(
      relations.difference(loop.getUpperBound(), loop.getLowerBound()));
  auto step = relations.interval(loop.getStep());
  if (!span || span.getValue() < 0 || !step || step->first <= 0 ||
      step->first != step->second) return std::nullopt;
  return llvm::divideCeil(static_cast<uint64_t>(span.getValue()),
                          static_cast<uint64_t>(step->first));
}

bool exposeSingleIterationSupply(func::FuncOp function) {
  SmallVector<scf::ForOp> loops;
  function.walk<WalkOrder::PostOrder>([&](scf::ForOp loop) {
    if (!loop.getInitArgs().empty()) return;
    bool prepares = false;
    loop.walk([&](dsa::PrepareMatrixOp) { prepares = true; });
    if (prepares) loops.push_back(loop);
  });
  dsa::LocalSupplyRelations relations(function);
  for (auto loop : loops) {
    if (tripCount(loop, relations) != 1) continue;
    // The current coordinate relation proves the same single iteration as the
    // native SCF promotion, including a nonliteral clipped upper bound.
    IRRewriter rewriter(function.getContext());
    Operation *yield = loop.getBody()->getTerminator();
    rewriter.inlineBlockBefore(loop.getBody(), loop->getBlock(),
                               loop->getIterator(), loop.getLowerBound());
    rewriter.eraseOp(yield);
    rewriter.eraseOp(loop);
    return true;
  }
  return false;
}

struct IterationDomain {
  scf::ForOp loop;
  int64_t lower, upper, step, count;
};

std::optional<IterationDomain> staticDomain(scf::ForOp loop) {
  APInt lower, upper, step;
  if (!matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) ||
      !matchPattern(loop.getUpperBound(), m_ConstantInt(&upper)) ||
      !matchPattern(loop.getStep(), m_ConstantInt(&step)) ||
      lower.getBitWidth() > 64 || upper.getBitWidth() > 64 ||
      step.getBitWidth() > 64 || !step.isStrictlyPositive()) return std::nullopt;
  APInt span = upper.sext(65) - lower.sext(65);
  if (span.isNegative() || span.getActiveBits() > 63) return std::nullopt;
  return IterationDomain{loop, lower.getSExtValue(), upper.getSExtValue(),
                         step.getSExtValue(),
                         static_cast<int64_t>(llvm::divideCeil(
                             span.getZExtValue(), step.getZExtValue()))};
}

bool guardedNonempty(scf::ForOp loop) {
  if (auto domain = staticDomain(loop)) return domain->count > 0;
  auto guard = dyn_cast_or_null<scf::IfOp>(loop->getParentOp());
  if (!guard || loop->getBlock()->getParent() != &guard.getThenRegion()) return false;
  auto condition = guard.getCondition().getDefiningOp<arith::CmpIOp>();
  return condition && condition.getPredicate() == arith::CmpIPredicate::slt &&
         condition.getLhs() == loop.getLowerBound() &&
         condition.getRhs() == loop.getUpperBound();
}

scf::IfOp guardLoop(scf::ForOp loop) {
  OpBuilder builder(loop);
  Value active = builder.create<arith::CmpIOp>(loop.getLoc(), arith::CmpIPredicate::slt,
                                             loop.getLowerBound(), loop.getUpperBound());
  auto guard = builder.create<scf::IfOp>(loop.getLoc(), loop.getResultTypes(), active, true);
  Block &body = guard.getThenRegion().front();
  loop->moveBefore(&body, body.begin());
  if (loop.getNumResults()) {
    builder.setInsertionPointToEnd(&body);
    auto yield = builder.create<scf::YieldOp>(loop.getLoc(), loop.getResults());
    builder.setInsertionPointToEnd(&guard.getElseRegion().front());
    builder.create<scf::YieldOp>(loop.getLoc(), loop.getInitArgs());
    for (auto [oldValue, newValue] : llvm::zip(loop.getResults(), guard.getResults()))
      oldValue.replaceUsesWithIf(newValue, [&](OpOperand &use) {
        return use.getOwner() != yield.getOperation();
      });
  }
  return guard;
}

bool fitsStorage(func::FuncOp function, dsa::ConfigurationAttr config) {
  return storageFitsBudget(function, config, measureStorage(function));
}

SmallVector<std::pair<Operation *, Operation *>> originalPositions(
    func::FuncOp function, ArrayRef<Operation *> operations) {
  DenseSet<Operation *> selected(operations.begin(), operations.end());
  SmallVector<std::pair<Operation *, Operation *>> positions;
  // Dependency order need not equal lexical order (an output allocation often
  // precedes its input in the collected DAG). Rollback uses the original order.
  function.walk<WalkOrder::PreOrder>([&](Operation *operation) {
    if (selected.contains(operation))
      positions.emplace_back(operation, operation->getNextNode());
  });
  return positions;
}

class SupplySlice {
public:
  SupplySlice(func::FuncOp function, scf::ForOp scope,
              ArrayRef<IterationDomain> domains = {})
      : scope(scope), storage(function), dominance(function) {
    for (auto domain : domains) allowed.insert(domain.loop.getInductionVar());
  }

  bool collect(Operation *root) {
    auto effects = storage.effects(scope);
    return effects.complete && !effects.ordered && writer(root);
  }

  // A cached slice replaces the whole producer computation. Intermediate local
  // definitions with other readers must remain at their original execution.
  bool closed(Value output) {
    Value root = storage.uniqueOrigin(output);
    for (auto allocation : allocations) {
      if (allocation.getResult() == root) continue;
      auto accesses = storage.accesses(allocation);
      if (!accesses.complete || accesses.ordered) return false;
      for (auto entry : accesses.entries)
        if (!isa<MemoryEffects::Allocate>(entry.effect.getEffect()) &&
            !nodes.contains(entry.operation)) return false;
    }
    return true;
  }

  SmallVector<Operation *> operations;
  SmallVector<Operation *> writers;
  SmallVector<memref::AllocaOp> allocations;
  DenseSet<Value> varying;
  bool globalRead = false;

private:
  bool value(Value value) {
    if (dominance.properlyDominates(value, scope)) return true;
    if (allowed.contains(value)) { varying.insert(value); return true; }
    Operation *definition = value.getDefiningOp();
    if (!definition || !scope->isProperAncestor(definition)) return false;
    if (auto allocation = dyn_cast<memref::AllocaOp>(definition)) {
      if (local.contains(value)) return true;
      Operation *producer = storage.uniqueWriter(value);
      return producer && writer(producer);
    }
    if (nodes.contains(definition)) return true;
    if (definition->getNumRegions() || !isMemoryEffectFree(definition) ||
        !isSpeculatable(definition) || !onPath(definition)) return false;
    for (Value operand : definition->getOperands()) if (!this->value(operand)) return false;
    nodes.insert(definition);
    operations.push_back(definition);
    return true;
  }

  bool onPath(Operation *operation) {
    for (Operation *parent = operation->getParentOp(); parent != scope;
         parent = parent->getParentOp())
      if (!parent || !isa<scf::ForOp>(parent) ||
          !allowed.contains(cast<scf::ForOp>(parent).getInductionVar())) return false;
    return true;
  }

  bool writer(Operation *operation) {
    if (nodes.contains(operation)) return true;
    Value output = suppliedValue(operation);
    if (!output || !onPath(operation) || !active.insert(operation).second) return false;
    auto completion = storage.completionOfUse(operation);
    auto effects = storage.effects(operation);
    if (failed(completion) || *completion != operation || !effects.complete || effects.ordered)
      return false;
    SmallVector<memref::AllocaOp> written;
    for (auto entry : effects.entries) {
      if (isa<MemoryEffects::Read>(entry.effect.getEffect())) continue;
      if (!isa<MemoryEffects::Write>(entry.effect.getEffect())) return false;
      Value origin = storage.uniqueOrigin(entry.effect.getValue());
      auto allocation = origin ? origin.getDefiningOp<memref::AllocaOp>() : memref::AllocaOp{};
      bool packed = isa<dsa::PrepareMatrixOp>(operation) && origin == output &&
                    allocation && allocation.getType().getMemorySpaceAsInt() == dsa::matrixSpace;
      if (!allocation || !onPath(allocation) ||
          !allocation.getType().hasStaticShape() || !allocation.getType().getLayout().isIdentity() ||
          (allocation.getType().getMemorySpaceAsInt() != dsa::nramSpace && !packed) ||
          storage.uniqueWriter(origin) != operation || !storage.aliases(origin).complete)
        return false;
      auto uses = storage.accesses(origin);
      if (!uses.complete || uses.ordered || llvm::any_of(uses.entries, [&](auto access) {
            return access.operation != operation &&
                   !isa<MemoryEffects::Allocate>(access.effect.getEffect()) &&
                   (!scope->isProperAncestor(access.operation) ||
                    !dominance.properlyDominates(operation, access.operation));
          })) return false;
      if (!llvm::is_contained(written, allocation)) written.push_back(allocation);
    }
    if (written.empty() || !dsa::isCompleteStorageViewOf(output, storage.uniqueOrigin(output)))
      return false;
    for (auto entry : effects.entries) {
      if (!isa<MemoryEffects::Read>(entry.effect.getEffect())) continue;
      Value memory = entry.effect.getValue();
      if (!memory || llvm::any_of(written, [&](auto allocation) {
            return !storage.disjoint(memory, allocation);
          })) return false;
      Value origin = storage.uniqueOrigin(memory);
      Operation *definition = origin ? origin.getDefiningOp() : nullptr;
      if ((!definition || !scope->isProperAncestor(definition)) &&
          !storage.preserves(scope, memory)) return false;
      if (auto type = dyn_cast<MemRefType>(memory.getType());
          type && type.getMemorySpaceAsInt() == 0) globalRead = true;
    }
    for (auto allocation : written) {
      local.insert(allocation.getResult());
      if (nodes.insert(allocation).second) {
        allocations.push_back(allocation);
        operations.push_back(allocation);
      }
    }
    for (Value operand : operation->getOperands()) if (!value(operand)) return false;
    nodes.insert(operation);
    operations.push_back(operation);
    writers.push_back(operation);
    active.erase(operation);
    return true;
  }

  scf::ForOp scope;
  dsa::StorageAnalysis storage;
  DominanceInfo dominance;
  DenseSet<Value> allowed, local;
  DenseSet<Operation *> nodes, active;
};

bool hoistSupply(func::FuncOp function, scf::ForOp loop,
                 dsa::ConfigurationAttr config) {
  if (auto domain = staticDomain(loop); domain && domain->count <= 1) return false;
  dsa::LocalSupplyRelations relations(function);
  SmallVector<Operation *> candidates;
  for (Operation &operation : loop.getBody()->without_terminator())
    if (suppliedValue(&operation)) candidates.push_back(&operation);
  for (Operation *root : llvm::reverse(candidates)) {
    if (isa<dsa::PrepareMatrixOp>(root)) {
      auto count = tripCount(loop, relations);
      if (!count || *count < 2) continue;
    }
    SupplySlice slice(function, loop);
    if (!slice.collect(root)) continue;
    auto positions = originalPositions(function, slice.operations);
    // Budget the real extension before constructing the nonempty guard. These
    // provisional moves are never published if storage cannot hold the slice.
    for (Operation *operation : slice.operations) operation->moveBefore(loop);
    bool fits = fitsStorage(function, config);
    for (auto [operation, next] : llvm::reverse(positions)) operation->moveBefore(next);
    if (!fits) continue;
    if (!guardedNonempty(loop)) guardLoop(loop);
    for (Operation *operation : slice.operations) operation->moveBefore(loop);
    return true;
  }
  return false;
}

Value index(OpBuilder &builder, Location location, int64_t value) {
  return builder.create<arith::ConstantIndexOp>(location, value);
}

Value slotOffset(OpBuilder &builder, Location location,
                 ArrayRef<IterationDomain> domains, const IRMapping &mapping,
                 int64_t elements) {
  Value ordinal = index(builder, location, 0);
  for (auto domain : domains) {
    Value iv = mapping.lookupOrDefault(domain.loop.getInductionVar());
    Value relative = builder.createOrFold<arith::SubIOp>(location, iv, index(builder, location, domain.lower));
    Value member = builder.createOrFold<arith::DivSIOp>(location, relative, index(builder, location, domain.step));
    ordinal = builder.createOrFold<arith::AddIOp>(location,
        builder.createOrFold<arith::MulIOp>(location, ordinal, index(builder, location, domain.count)), member);
  }
  return builder.createOrFold<arith::MulIOp>(location, ordinal, index(builder, location, elements));
}

bool cacheSupply(func::FuncOp function, Operation *root, scf::ForOp scope,
                 ArrayRef<IterationDomain> path, dsa::ConfigurationAttr config) {
  auto repetition = staticDomain(scope);
  if (!repetition || repetition->count < 2) return false;
  Value output = suppliedValue(root);
  auto type = output ? dyn_cast<MemRefType>(output.getType()) : MemRefType{};
  if (!type || type.getRank() != 2 || !type.hasStaticShape() ||
      !type.getLayout().isIdentity() || type.getNumElements() <= 0 ||
      type.getMemorySpaceAsInt() != dsa::nramSpace) return false;
  SupplySlice slice(function, scope, path);
  if (!slice.collect(root) || !slice.globalRead || !slice.closed(output)) return false;
  SmallVector<IterationDomain> domains;
  int64_t slots = 1;
  for (auto domain : path) {
    if (!domain.loop.getInductionVar().getType().isIndex()) return false;
    if (!slice.varying.contains(domain.loop.getInductionVar())) continue;
    if (domain.count <= 0 || slots > std::numeric_limits<int64_t>::max() / domain.count)
      return false;
    slots *= domain.count;
    domains.push_back(domain);
  }
  if (domains.empty() || slots > std::numeric_limits<int64_t>::max() / type.getNumElements())
    return false;
  // The fixed NRAM arena must contain every original padded tile. Nothing is
  // inferred about unread positions between these exact source windows.
  Type element = type.getElementType();
  if (!isa<IntegerType, FloatType>(element)) return false;
  int64_t bytes = llvm::divideCeil(element.getIntOrFloatBitWidth(), 8u);
  if (slots * type.getNumElements() > config.getLocalBytes() / bytes) return false;

  OpBuilder builder(scope);
  Location location = root->getLoc();
  Operation *previous = scope->getPrevNode();
  Value cache = allocate(builder, location, element,
                         {slots * type.getDimSize(0), type.getDimSize(1)}, dsa::nramSpace);
  IRMapping mapping;
  for (auto domain : domains) {
    auto fill = builder.create<scf::ForOp>(location, index(builder, location, domain.lower),
        index(builder, location, domain.upper), index(builder, location, domain.step));
    mapping.map(domain.loop.getInductionVar(), fill.getInductionVar());
    builder.setInsertionPointToStart(fill.getBody());
  }
  for (Operation *operation : slice.operations) builder.clone(*operation, mapping);
  Value offset = slotOffset(builder, location, domains, mapping, type.getNumElements());
  Value columns = index(builder, location, type.getDimSize(1));
  Value one = index(builder, location, 1);
  builder.create<dsa::StoreTileOp>(location, mapping.lookup(output), cache, offset, columns,
      one, index(builder, location, type.getDimSize(0)), columns);

  builder.setInsertionPoint(root);
  Operation *oldPrevious = root->getPrevNode();
  IRMapping current;
  offset = slotOffset(builder, location, domains, current, type.getNumElements());
  columns = index(builder, location, type.getDimSize(1));
  one = index(builder, location, 1);
  auto read = builder.create<dsa::LoadTileOp>(location, cache, output, offset, columns,
      one, index(builder, location, type.getDimSize(0)), columns, builder.getBoolAttr(false));
  // All intermediate storage reads are inside this closed slice. Remove its
  // actual writers, not just the root, so cached supply replaces rather than
  // duplicates the old global reads and numerical work.
  auto positions = originalPositions(function, slice.writers);
  SmallVector<SmallVector<Value>> operands;
  for (Operation *operation : slice.writers) {
    operands.emplace_back(operation->getOperands());
    // Detached operations must not remain in live buffers' use lists while
    // the current-IR lifetime analysis visits their regions and completions.
    operation->setOperands(ValueRange{});
    operation->remove();
  }
  bool fits = fitsStorage(function, config);
  if (!fits) {
    for (auto [operation, original] : llvm::zip(slice.writers, operands))
      operation->setOperands(original);
    for (auto [operation, successor] : llvm::reverse(positions)) {
      OpBuilder restore(successor);
      restore.insert(operation);
    }
    while (root->getPrevNode() != oldPrevious) root->getPrevNode()->erase();
    while (scope->getPrevNode() != previous) scope->getPrevNode()->erase();
    return false;
  }
  // The local copy reads the whole cached padded tile, rather than padding it a
  // second time from logical row/column counts. Original downstream views and
  // all numerical operations therefore observe the same lanes.
  if (auto origin = root->getAttr("intent.origin")) read->setAttr("intent.origin", origin);
  for (Operation *operation : llvm::reverse(slice.writers)) operation->destroy();
  return true;
}

} // namespace

bool placeInvariantSupply(func::FuncOp function, dsa::ConfigurationAttr config) {
  if (exposeSingleIterationSupply(function)) return true;
  SmallVector<scf::ForOp> loops;
  function.walk<WalkOrder::PostOrder>([&](scf::ForOp loop) { loops.push_back(loop); });
  for (scf::ForOp loop : loops)
    if (hoistSupply(function, loop, config)) return true;

  SmallVector<Operation *> roots;
  function.walk([&](Operation *operation) {
    if (suppliedValue(operation)) roots.push_back(operation);
  });
  for (Operation *root : llvm::reverse(roots)) {
    SmallVector<IterationDomain> path;
    SmallVector<std::pair<scf::ForOp, SmallVector<IterationDomain>>> scopes;
    auto inner = dyn_cast<scf::ForOp>(root->getParentOp());
    while (inner) {
      auto domain = staticDomain(inner);
      if (!domain || domain->count <= 0) break;
      path.insert(path.begin(), *domain);
      auto outer = dyn_cast<scf::ForOp>(inner->getParentOp());
      if (!outer) break;
      scopes.emplace_back(outer, path);
      inner = outer;
    }
    // Choose the widest proven consumer scope once. Building a cache per inner
    // scope and then caching that preparation again would add redundant local
    // copies and unnecessarily overlapping resident storage.
    for (auto &entry : llvm::reverse(scopes))
      if (cacheSupply(function, root, entry.first, entry.second, config)) return true;
  }
  return false;
}

} // namespace intent::bangc
