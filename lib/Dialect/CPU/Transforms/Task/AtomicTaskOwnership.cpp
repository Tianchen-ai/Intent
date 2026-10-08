#include "AtomicTaskOwnership.h"
#include "Intent/Analysis/IntegerRanges.h"
#include "Intent/Analysis/IntegerRelations.h"
#include "Intent/Dialect/CPU/Analysis/ExtentRelations.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Intent/Dialect/CPU/Transforms/Structure/LoopBuilders.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace intent::cpu {
namespace {

struct Update {
  Operation *operation;
  Value target;
  SmallVector<Value> indices;
};

struct Ownership {
  scf::ParallelOp root;
  SmallVector<Update, 2> updates;
  Value coordinate;
  int64_t span = 1;
  SmallVector<Operation *> dependencies;
};

// A normalized floor remainder retains a bound even when independent ranges
// lose the correlation between x and floor(x / s) * s. The shared round-down
// proof guarantees that this multiplication cannot wrap for x >= 0, s > 0.
IntegerRangePolicy coordinatePolicy() {
  IntegerRangePolicy policy;
  policy.infer = [](Value value, IntegerRangeAnalysis &ranges)
      -> std::optional<ConstantIntRanges> {
    auto difference = value.getDefiningOp<arith::SubIOp>();
    if (!difference || !value.getType().isIndex()) return std::nullopt;
    IntegerOrderCallbacks<Value> callbacks;
    callbacks.signedWidth = [](Value v) { return v.getType().isIndex() ? 64u : 0u; };
    callbacks.constant = [](Value v) { return getConstantIntValue(v); };
    callbacks.nonnegative = [&](Value v) { return ranges.isNonNegative(v); };
    callbacks.positive = [&](Value v) { return ranges.isPositive(v); };
    callbacks.binary = [](Value v) -> std::optional<IntegerOrderBinary<Value>> {
      if (auto op = v.getDefiningOp<arith::MulIOp>())
        return IntegerOrderBinary<Value>{IntegerOrderKind::Multiply, op.getLhs(), op.getRhs()};
      if (auto op = v.getDefiningOp<arith::FloorDivSIOp>())
        return IntegerOrderBinary<Value>{IntegerOrderKind::FloorDivide, op.getLhs(), op.getRhs()};
      return std::nullopt;
    };
    auto rounded = IntegerOrder<Value>(std::move(callbacks)).roundDown(difference.getRhs());
    if (!rounded || rounded->dividend != difference.getLhs()) return std::nullopt;
    auto divisor = getConstantIntValue(rounded->divisor);
    if (!divisor || *divisor <= 0) return std::nullopt;
    return ConstantIntRanges::fromSigned(APInt(64, 0), APInt(64, *divisor - 1));
  };
  return policy;
}

std::optional<Update> update(Operation *operation) {
  if (auto atomic = dyn_cast<AtomicRMWOp>(operation)) {
    if (atomic.getOrdering() != AtomicOrdering::Relaxed ||
        atomic.getKind() != AtomicRMWKind::Add ||
        !isa<FloatType>(atomic.getValue().getType()) || !atomic.getOldValue().use_empty())
      return std::nullopt;
    return Update{operation, atomic.getTarget(), llvm::to_vector(atomic.getIndices())};
  }
  auto atomic = dyn_cast<memref::GenericAtomicRMWOp>(operation);
  if (!atomic || !atomic.getResult().use_empty() ||
      !llvm::hasSingleElement(atomic.getRegion())) return std::nullopt;
  Block &body = atomic.getRegion().front();
  if (body.getNumArguments() != 1 ||
      !isa<IntegerType, FloatType>(body.getArgument(0).getType())) return std::nullopt;
  for (Operation &instruction : body.without_terminator())
    if (instruction.getNumRegions() || !isMemoryEffectFree(&instruction)) return std::nullopt;
  return Update{operation, atomic.getMemref(), llvm::to_vector(atomic.getIndices())};
}

std::optional<Ownership> query(func::FuncOp function, scf::ParallelOp root,
                               int64_t grain) {
  if (root->getBlock() != &function.front() || root.getNumResults() ||
      !llvm::all_of(root.getLowerBound(), [](Value v) { return matchPattern(v, m_Zero()); }) ||
      !llvm::all_of(root.getStep(), [](Value v) { return matchPattern(v, m_One()); }))
    return std::nullopt;
  auto entry = function->getAttrOfType<EntryRequirementsAttr>(entryRequirementsAttr);
  if (!entry || !entry.getDisjointOutputs()) return std::nullopt;
  Ownership plan{root};
  root.walk([&](Operation *operation) {
    if (auto selected = update(operation)) plan.updates.push_back(std::move(*selected));
  });
  if (plan.updates.empty()) return std::nullopt;
  StorageAnalysis storage(function);
  IntegerRangeAnalysis ranges(coordinatePolicy());
  llvm::SmallPtrSet<Operation *, 4> updates;
  SmallVector<Value> targets;
  llvm::SetVector<Operation *> dependencies;
  std::function<bool(Value)> collect = [&](Value value) {
    if (llvm::is_contained(root.getInductionVars(), value)) return true;
    if (auto argument = dyn_cast<BlockArgument>(value))
      return !root->isAncestor(argument.getOwner()->getParentOp());
    Operation *definition = value.getDefiningOp();
    if (!definition || !root->isAncestor(definition)) return true;
    if (dependencies.contains(definition)) return true;
    if (auto load = dyn_cast<memref::LoadOp>(definition)) {
      if (load->getBlock() != root.getBody() ||
          !storage.preservesContents(root, load.getMemref())) return false;
    } else if (!isMemoryEffectFree(definition) || !isSpeculatable(definition) ||
               definition->getNumRegions() ||
               llvm::any_of(definition->getOperandTypes(), [](Type t) { return isa<MemRefType>(t); }) ||
               llvm::any_of(definition->getResultTypes(), [](Type t) { return isa<MemRefType>(t); })) return false;
    if (definition->getBlock() != root.getBody() &&
        !isa<arith::ConstantOp, arith::IndexCastOp, arith::IndexCastUIOp,
             arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(definition)) return false;
    if (!llvm::all_of(definition->getOperands(), collect)) return false;
    dependencies.insert(definition);
    return true;
  };

  auto capabilities = function->getParentOfType<ModuleOp>()->getAttrOfType<CapabilitiesAttr>("intent_cpu.capabilities");
  for (const Update &atomic : plan.updates) {
    auto type = cast<MemRefType>(atomic.target.getType());
    auto external = storage.externalView(atomic.target);
    if (!type.getRank() || !type.getLayout().isIdentity() || !external || external.getAccess() != 2 ||
        atomic.target != storage.uniqueOrigin(atomic.target) ||
        !atomic.operation->getParentOfType<scf::ForOp>()) return std::nullopt;
    Value coordinate = atomic.indices.front();
    int64_t span = 1;
    auto before = dependencies;
    if (!collect(coordinate)) {
      dependencies = std::move(before);
      // A whole update slice may use base + iv in its first address axis. Keep
      // it in one owner; never split the surrounding computation across tasks.
      auto add = coordinate.getDefiningOp<arith::AddIOp>();
      if (!add) return std::nullopt;
      scf::ForOp inner;
      Value base;
      for (unsigned side = 0; side != 2; ++side) {
        Value iv = add->getOperand(side);
        auto argument = dyn_cast<BlockArgument>(iv);
        auto loop = argument ? dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp()) : scf::ForOp{};
        if (loop && loop.getInductionVar() == iv && root->isAncestor(loop) &&
            loop->isAncestor(atomic.operation) && matchPattern(loop.getLowerBound(), m_Zero()) &&
            matchPattern(loop.getStep(), m_One())) {
          inner = loop; base = add->getOperand(1 - side); break;
        }
      }
      auto count = inner ? getConstantIntValue(inner.getUpperBound()) : std::nullopt;
      if (!count || *count <= 0 || type.isDynamicDim(0) || !collect(base)) return std::nullopt;
      span = *count;
      auto lhs = ranges.range(add.getLhs()), rhs = ranges.range(add.getRhs());
      if (!lhs || !rhs || !provesSignedNoWrap(BinaryOperator::Add, *lhs, *rhs)) return std::nullopt;
      bool aligned = false;
      if (auto constant = getConstantIntValue(base)) aligned = *constant % span == 0;
      if (auto product = base.getDefiningOp<arith::MulIOp>()) {
        auto left = ranges.range(product.getLhs()), right = ranges.range(product.getRhs());
        if (left && right && provesSignedNoWrap(BinaryOperator::Multiply, *left, *right))
          for (Value operand : product->getOperands())
            if (auto factor = getConstantIntValue(operand)) aligned |= *factor % span == 0;
      }
      int64_t rows = type.getDimSize(0);
      int64_t tasks = std::max<int64_t>(1, std::min<int64_t>(
          rows / grain + (rows % grain != 0), capabilities.getWorkers()));
      if (!aligned || rows % tasks || (rows / tasks) % span) return std::nullopt;
      coordinate = base;
    }
    if (!plan.coordinate) { plan.coordinate = coordinate; plan.span = span; }
    else if (plan.coordinate != coordinate || plan.span != span) return std::nullopt;
    if (!targets.empty() && !haveEqualExtents(
            ValueBoundsConstraintSet::Variable(targets.front(), 0),
            ValueBoundsConstraintSet::Variable(atomic.target, 0))) return std::nullopt;
    if (!llvm::is_contained(targets, atomic.target)) {
      for (Value other : targets) if (!storage.disjoint(other, atomic.target)) return std::nullopt;
      targets.push_back(atomic.target);
    }
    updates.insert(atomic.operation);
  }
  if (getConstantIntValue(plan.coordinate)) return std::nullopt;

  DenseMap<Value, bool> locals;
  auto local = [&](Value memory) {
    Value origin = storage.uniqueOrigin(memory);
    if (!origin) return false;
    if (auto known = locals.find(origin); known != locals.end()) return known->second;
    Operation *allocation = origin.getDefiningOp();
    bool result = allocation && isa<memref::AllocOp, memref::AllocaOp>(allocation) &&
        root->isProperAncestor(allocation);
    if (result) {
      auto aliases = storage.aliases(origin);
      result = aliases.complete && llvm::all_of(aliases.users, [&](Operation *user) {
        return root->isProperAncestor(user);
      });
    }
    locals[origin] = result;
    return result;
  };
  bool valid = true, otherWrites = false;
  root.walk([&](Operation *operation) {
    if (!valid || operation == root || updates.contains(operation)) return;
    for (Operation *atomic : updates) if (atomic->isAncestor(operation)) return;
    if (isa<scf::ForOp, scf::IfOp>(operation)) return;
    if (operation->getNumRegions() &&
        !isa<linalg::LinalgOp, ReduceOp, SliceReduceOp, ScanOp, RegionFoldOp, RegionScanOp>(operation)) {
      valid = false; return;
    }
    if (isMemoryEffectFree(operation) || isStorageAliasOperation(operation)) return;
    auto effects = storage.effects(operation);
    if (!effects.complete || effects.ordered) { valid = false; return; }
    for (const auto &effect : effects.entries) {
      if (updates.contains(effect.operation)) continue;
      Value memory = effect.effect.getValue();
      if (!memory) { valid = false; return; }
      if (local(memory)) continue;
      for (Value target : targets)
        if (!storage.disjoint(memory, target)) { valid = false; return; }
      if (isa<MemoryEffects::Read>(effect.effect.getEffect())) {
        if (!storage.preservesContents(root, memory)) { valid = false; return; }
      } else if (isa<MemoryEffects::Write>(effect.effect.getEffect())) {
        auto external = storage.externalView(memory);
        if (!external || external.getAccess() != 1) { valid = false; return; }
        otherWrites = true;
      } else { valid = false; return; }
    }
  });
  if (!valid) return std::nullopt;
  if (otherWrites || plan.span != 1) {
    // Otherwise a point with an inactive update could fail the owner predicate
    // and lose a separate unique output. Prove the key for every original point.
    auto coordinate = ranges.range(plan.coordinate);
    auto rows = ranges.dimension(targets.front(), 0);
    if (!coordinate || !rows || coordinate->smin().isNegative() ||
        (coordinate->smax().sext(128) + APInt(128, plan.span)).sgt(rows->smin().sext(128)))
      return std::nullopt;
  }
  plan.dependencies.assign(dependencies.begin(), dependencies.end());
  return plan;
}

void makePrivate(Update atomic) {
  OpBuilder b(atomic.operation);
  Location loc = atomic.operation->getLoc();
  Value old = b.create<memref::LoadOp>(loc, atomic.target, atomic.indices);
  Value result;
  if (auto add = dyn_cast<AtomicRMWOp>(atomic.operation)) {
    result = b.create<arith::AddFOp>(loc, old, add.getValue());
  } else {
    auto generic = cast<memref::GenericAtomicRMWOp>(atomic.operation);
    Block &body = generic.getRegion().front();
    IRMapping mapping;
    mapping.map(body.getArgument(0), old);
    for (Operation &operation : body.without_terminator()) b.clone(operation, mapping);
    result = mapping.lookupOrDefault(body.getTerminator()->getOperand(0));
  }
  b.create<memref::StoreOp>(loc, result, atomic.target, atomic.indices);
  atomic.operation->erase();
}

scf::ParallelOp rewrite(Ownership plan, int64_t grain) {
  auto root = plan.root;
  auto function = root->getParentOfType<func::FuncOp>();
  auto capabilities = function->getParentOfType<ModuleOp>()->getAttrOfType<CapabilitiesAttr>("intent_cpu.capabilities");
  OpBuilder b(root);
  Location loc = root.getLoc();
  Value zero = index(b, loc, 0), one = index(b, loc, 1);
  Value rows = b.create<memref::DimOp>(loc, plan.updates.front().target, 0);
  Value count = b.create<arith::CeilDivSIOp>(loc, rows, index(b, loc, grain));
  count = b.create<arith::MaxSIOp>(loc, one,
      b.create<arith::MinSIOp>(loc, count, index(b, loc, capabilities.getWorkers())));
  Value width = b.create<arith::DivSIOp>(loc, rows, count);
  Value remainder = b.create<arith::RemSIOp>(loc, rows, count);
  auto owners = b.create<scf::ParallelOp>(loc, ValueRange{zero}, ValueRange{count}, ValueRange{one});
  owners->setDiscardableAttrs(root->getDiscardableAttrDictionary());
  b.setInsertionPointToStart(owners.getBody());
  Value owner = owners.getInductionVars()[0];
  Value begin = add(b, loc, multiply(b, loc, owner, width),
      b.create<arith::MinSIOp>(loc, owner, remainder));
  Value extra = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, owner, remainder);
  Value end = add(b, loc, begin, add(b, loc, width, b.create<arith::SelectOp>(loc, extra, one, zero)));
  IRMapping mapping;
  std::function<void(unsigned)> points = [&](unsigned axis) {
    if (axis != root.getNumLoops()) {
      auto traversal = b.create<scf::ForOp>(loc, root.getLowerBound()[axis],
          root.getUpperBound()[axis], root.getStep()[axis]);
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(traversal.getBody());
      mapping.map(root.getInductionVars()[axis], traversal.getInductionVar());
      points(axis + 1);
      return;
    }
    for (Operation *dependency : plan.dependencies) b.clone(*dependency, mapping);
    Value selected = mapping.lookupOrDefault(plan.coordinate);
    Value lower = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, selected, begin);
    Value upper = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, selected, end);
    auto active = b.create<scf::IfOp>(loc, b.create<arith::AndIOp>(loc, lower, upper), false);
    b.setInsertionPointToStart(active.thenBlock());
    for (Operation &operation : root.getBody()->without_terminator())
      if (!llvm::is_contained(plan.dependencies, &operation)) b.clone(operation, mapping);
    for (const Update &original : plan.updates) {
      Operation *cloned = mapping.lookup(original.operation->getResult(0)).getDefiningOp();
      auto selectedUpdate = update(cloned);
      assert(selectedUpdate && "cloned update preserves the selected helper");
      makePrivate(std::move(*selectedUpdate));
    }
  };
  points(0);
  root.erase();
  return owners;
}

} // namespace

scf::ParallelOp partitionAtomicWorkset(func::FuncOp function,
                                      scf::ParallelOp root, int64_t grain) {
  auto plan = query(function, root, grain);
  return plan ? rewrite(std::move(*plan), grain) : scf::ParallelOp{};
}

} // namespace intent::cpu
