#include "intentdsl/levels/execution/block/ir/FoldStripes.h"
#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/JSON.h"
#include <algorithm>
#include <map>

namespace intentdsl::block {
using namespace mlir;

namespace {
enum class ArgKind { Scalar, Matrix2D, Outer1D, Inner1D };

struct ArgDesc {
  int64_t pubSlot = -1;
  ArgKind kind = ArgKind::Scalar;
  int64_t offset = 0;
  int64_t stepIdx = 0;
  bool striped = false;
};

struct StripePlan {
  int64_t count = 1;
  int64_t stride = 0;
  int64_t domainPubSlot = -1;
  llvm::json::Array pubSlots;
  SmallVector<ArgDesc> args;
};

std::optional<int64_t> publicDimExtent(const llvm::json::Array &pubSlots,
                                       const CallFacts *facts, int64_t slot,
                                       int64_t axis) {
  if (facts && slot >= 0 && size_t(slot) < facts->arguments.size()) {
    if (auto *obj = facts->arguments[slot].getAsObject())
      if (auto *sh = obj->getArray("shape"))
        if (axis >= 0 && size_t(axis) < sh->size())
          if (auto v = (*sh)[axis].getAsInteger())
            return *v;
  }
  if (slot >= 0 && size_t(slot) < pubSlots.size())
    if (auto *obj = pubSlots[slot].getAsObject())
      if (auto *dims = obj->getArray("dims"))
        if (axis >= 0 && size_t(axis) < dims->size())
          if (auto v = (*dims)[axis].getAsInteger())
            return *v;
  return std::nullopt;
}

std::optional<StripePlan> analyzeStripes(func::FuncOp fn,
                                         const CallFacts *facts) {
  if (!fn || !fn->hasAttr("kir.iteration") || fn->hasAttr("kir.owner_axis"))
    return std::nullopt;
  auto bounds = fn->getAttrOfType<DenseI64ArrayAttr>("kir.iteration");
  auto pubAttr = fn->getAttrOfType<StringAttr>("kir.public_abi");
  if (!bounds || bounds.size() != 8 || !pubAttr)
    return std::nullopt;
  auto parsed = llvm::json::parse(pubAttr.getValue());
  if (!parsed || !parsed->getAsArray()) {
    if (!parsed)
      llvm::consumeError(parsed.takeError());
    return std::nullopt;
  }
  StripePlan plan;
  plan.pubSlots = *parsed->getAsArray();
  int64_t outerDim = 0;
  if (bounds[0] < 0) {
    outerDim = bounds[3];
  } else {
    auto ext = publicDimExtent(plan.pubSlots, facts, bounds[0], bounds[1]);
    if (!ext || bounds[2] <= 0 || *ext % bounds[2] != 0)
      return std::nullopt;
    outerDim = *ext / bounds[2];
  }
  if (outerDim <= 0)
    return std::nullopt;
  plan.stride = outerDim;

  plan.args.resize(fn.getNumArguments());
  std::map<int64_t, SmallVector<unsigned>> byPub;
  for (unsigned i = 0; i < fn.getNumArguments(); ++i) {
    auto srcAttr = fn.getArgAttrOfType<IntegerAttr>(i, "kir.source");
    if (!srcAttr || srcAttr.getInt() < 0 ||
        size_t(srcAttr.getInt()) >= plan.pubSlots.size())
      return std::nullopt;
    ArgDesc d;
    d.pubSlot = srcAttr.getInt();
    auto ty = fn.getArgument(i).getType();
    if (!isa<MemRefType>(ty)) {
      if (fn.getArgAttr(i, "kir.indexing"))
        return std::nullopt;
      d.kind = ArgKind::Scalar;
      plan.args[i] = d;
      byPub[d.pubSlot].push_back(i);
      continue;
    }
    if (fn.getArgAttr(i, "kir.owner_projection"))
      return std::nullopt;
    auto mapAttr = fn.getArgAttrOfType<AffineMapAttr>(i, "kir.indexing");
    if (!mapAttr)
      return std::nullopt;
    auto map = mapAttr.getValue();
    if (map.getNumDims() != 2 || map.getNumSymbols() != 0)
      return std::nullopt;
    if (map.getNumResults() == 2) {
      SmallVector<int64_t, 3> c0(3, 0), c1(3, 0);
      if (!intentdsl::coefficients(map.getResult(0), c0) ||
          !intentdsl::coefficients(map.getResult(1), c1))
        return std::nullopt;
      if (c0[0] != 1 || c0[1] != 0 || c0[2] < 0 || c0[2] % plan.stride != 0 ||
          c1[0] != 0 || c1[1] != 1 || c1[2] != 0)
        return std::nullopt;
      d.kind = ArgKind::Matrix2D;
      d.offset = c0[2];
      d.stepIdx = c0[2] / plan.stride;
    } else if (map.getNumResults() == 1) {
      SmallVector<int64_t, 3> c0(3, 0);
      if (!intentdsl::coefficients(map.getResult(0), c0))
        return std::nullopt;
      if (c0[0] == 1 && c0[1] == 0 && c0[2] >= 0 && c0[2] % plan.stride == 0) {
        d.kind = ArgKind::Outer1D;
        d.offset = c0[2];
        d.stepIdx = c0[2] / plan.stride;
      } else if (c0[0] == 0 && c0[1] == 1 && c0[2] == 0) {
        d.kind = ArgKind::Inner1D;
        d.offset = 0;
        d.stepIdx = 0;
      } else {
        return std::nullopt;
      }
    } else {
      return std::nullopt;
    }
    plan.args[i] = d;
    byPub[d.pubSlot].push_back(i);
  }

  int64_t stripeCount = -1;
  for (int64_t s = 0; size_t(s) < plan.pubSlots.size(); ++s) {
    auto it = byPub.find(s);
    if (it == byPub.end() || it->second.empty())
      return std::nullopt;
    auto &group = it->second;
    ArgKind kind = plan.args[group.front()].kind;
    for (unsigned idx : group)
      if (plan.args[idx].kind != kind)
        return std::nullopt;
    bool allZero = llvm::all_of(
        group, [&](unsigned idx) { return plan.args[idx].offset == 0; });
    if (allZero) {
      if (kind == ArgKind::Outer1D)
        return std::nullopt;
      if (kind == ArgKind::Matrix2D) {
        auto ext0 = publicDimExtent(plan.pubSlots, facts, s, 0);
        if (!ext0 || *ext0 != plan.stride)
          return std::nullopt;
        if (plan.domainPubSlot < 0)
          plan.domainPubSlot = s;
      }
      continue;
    }
    if (kind != ArgKind::Matrix2D && kind != ArgKind::Outer1D)
      return std::nullopt;
    llvm::SmallDenseSet<int64_t, 32> steps;
    int64_t maxStep = -1;
    for (unsigned idx : group) {
      steps.insert(plan.args[idx].stepIdx);
      maxStep = std::max(maxStep, plan.args[idx].stepIdx);
    }
    int64_t k = maxStep + 1;
    if (k < 2 || int64_t(steps.size()) != k)
      return std::nullopt;
    for (int64_t step = 0; step < k; ++step)
      if (!steps.contains(step))
        return std::nullopt;
    auto ext0 = publicDimExtent(plan.pubSlots, facts, s, 0);
    if (!ext0 || *ext0 != k * plan.stride)
      return std::nullopt;
    if (stripeCount < 0)
      stripeCount = k;
    else if (stripeCount != k)
      return std::nullopt;
    for (unsigned idx : group)
      plan.args[idx].striped = true;
  }
  if (stripeCount < 2 || plan.domainPubSlot < 0)
    return std::nullopt;
  plan.count = stripeCount;
  return plan;
}

Type scalarTypeFromABI(OpBuilder &b, StringRef dt) {
  if (dt == "f16")
    return b.getF16Type();
  if (dt == "bf16")
    return b.getBF16Type();
  if (dt == "f32")
    return b.getF32Type();
  if (dt == "i32")
    return b.getI32Type();
  if (dt == "i64")
    return b.getI64Type();
  if (dt == "u64")
    return b.getIntegerType(64, false);
  return {};
}
} // namespace

uint64_t foldableStripeCount(Operation *parallelOp) {
  if (!parallelOp || parallelOp->getName().stripDialect() != "parallel")
    return 1;
  auto fn = parallelOp->getParentOfType<func::FuncOp>();
  auto plan = analyzeStripes(fn, nullptr);
  if (!plan)
    return 1;
  uint64_t reductions = 0;
  bool unsupported = false;
  parallelOp->walk([&](Operation *op) {
    auto n = op->getName().stripDialect();
    if (n == "sum" || n == "max_num")
      ++reductions;
    else if (n == "welford" || n == "scan_sum" || n == "count_keys" ||
             n == "ordered_sum" || n == "prefix_scatter")
      unsupported = true;
  });
  if (unsupported || reductions == 0 || reductions % plan->count != 0)
    return 1;
  return uint64_t(plan->count);
}

LogicalResult foldStripes(func::FuncOp fn,
                          const compilerinfra::TargetProfile &profile) {
  if (!fn || !fn->hasAttr("kir.iteration") || fn->hasAttr("kir.owner_axis"))
    return success();
  if (fn->getAttrOfType<StringAttr>("block.topology") != "resident")
    return success();
  auto *facts = profile.services.get<CallFacts>();
  auto plan = analyzeStripes(fn, facts);
  if (!plan)
    return success();

  auto configs = fn->getAttrOfType<ArrayAttr>("block.configs");
  if (!configs || configs.empty())
    return success();
  for (Attribute attr : configs) {
    auto d = dyn_cast<DictionaryAttr>(attr);
    auto r = d ? d.getAs<IntegerAttr>("rows") : IntegerAttr();
    if (!r || r.getInt() <= 0 || plan->stride % r.getInt() != 0)
      return success();
  }

  Block &entry = fn.front();
  const int64_t K = plan->count;
  Value rowsVal, colsVal, activeVal;
  llvm::DenseMap<Value, int64_t> valStripe;
  for (unsigned i = 0; i < entry.getNumArguments(); ++i)
    if (!plan->args[i].striped)
      valStripe[entry.getArgument(i)] = -1;

  SmallVector<Operation *> postStores;
  SmallVector<SmallVector<Operation *>> stripeStores(K);

  for (Operation &op : entry.without_terminator()) {
    if (op.getName().getDialectNamespace() != "block" || op.getNumRegions() > 0)
      return success();
    auto n = op.getName().stripDialect();
    if (n == "pid" || n == "zero" || n == "constant") {
      if (op.getNumResults() != 1)
        return success();
      valStripe[op.getResult(0)] = -1;
      continue;
    }
    if (n == "rows") {
      if (rowsVal)
        return success();
      rowsVal = op.getResult(0);
      valStripe[rowsVal] = -1;
      continue;
    }
    if (n == "columns") {
      if (colsVal)
        return success();
      colsVal = op.getResult(0);
      valStripe[colsVal] = -1;
      continue;
    }
    if (n == "active") {
      if (activeVal)
        return success();
      activeVal = op.getResult(0);
      valStripe[activeVal] = -1;
      continue;
    }
    if (n == "load") {
      auto arg = dyn_cast<BlockArgument>(op.getOperand(0));
      if (!arg || arg.getOwner() != &entry || op.getOperand(1) != rowsVal ||
          op.getOperand(2) != colsVal || op.getOperand(3) != activeVal)
        return success();
      const auto &desc = plan->args[arg.getArgNumber()];
      valStripe[op.getResult(0)] = desc.striped ? desc.stepIdx : -1;
      continue;
    }
    if (n == "store") {
      auto arg = dyn_cast<BlockArgument>(op.getOperand(0));
      if (!arg || arg.getOwner() != &entry || op.getOperand(1) != rowsVal ||
          op.getOperand(2) != colsVal || op.getOperand(3) != activeVal)
        return success();
      const auto &desc = plan->args[arg.getArgNumber()];
      if (desc.kind != ArgKind::Matrix2D)
        return success();
      auto it = valStripe.find(op.getOperand(4));
      if (it == valStripe.end())
        return success();
      if (desc.striped) {
        if (it->second != desc.stepIdx)
          return success();
        stripeStores[desc.stepIdx].push_back(&op);
      } else {
        if (it->second != K - 1 && it->second != -1)
          return success();
        postStores.push_back(&op);
      }
      continue;
    }
    if (op.getNumResults() != 1)
      return success();
    int64_t s = -1;
    for (Value v : op.getOperands()) {
      auto it = valStripe.find(v);
      if (it == valStripe.end())
        return success();
      s = std::max(s, it->second);
    }
    if (s >= 0) {
      for (Value v : op.getOperands()) {
        int64_t vs = valStripe[v];
        if (vs != -1 && vs != s - 1 && vs != s)
          return success();
      }
    }
    valStripe[op.getResult(0)] = s;
  }

  if (!rowsVal || !colsVal || !activeVal)
    return success();

  // Trace recurrence spines backwards from post-loop invariant stores and
  // assign any carry-entry unary adapters (e.g. bf16->f32 cast on carry-in)
  // to the consuming stripe.
  struct Carry {
    unsigned producerIdx = 0;
    Value init;
    SmallVector<Value> next;
  };
  SmallVector<Carry> carries;
  llvm::SmallDenseSet<Value, 4> seenFinal;
  for (Operation *ps : postStores) {
    Value finalVal = ps->getOperand(4);
    if (valStripe.lookup(finalVal) != K - 1 || !seenFinal.insert(finalVal).second)
      continue;
    if (!isa<RankedTensorType>(finalVal.getType()) || !finalVal.getDefiningOp())
      return success();
    Carry c;
    c.next.resize(K);
    c.next[K - 1] = finalVal;
    unsigned unaryDepth = 0;
    Value curOut = finalVal;
    for (int64_t step = K - 1; step >= 0; --step) {
      // Walk ops in stripe `step` feeding `curOut` to find the unique value
      // from `step - 1` that is specific to this recurrence step.
      SmallVector<Operation *, 16> work{curOut.getDefiningOp()};
      llvm::SmallDenseSet<Operation *, 16> visited{curOut.getDefiningOp()};
      Value boundary;
      while (!work.empty()) {
        Operation *inst = work.pop_back_val();
        for (Value opnd : inst->getOperands()) {
          int64_t os = valStripe.lookup(opnd);
          if (os == step) {
            if (Operation *def = opnd.getDefiningOp();
                def && def->getName().stripDialect() != "load" &&
                visited.insert(def).second)
              work.push_back(def);
          } else if (os == step - 1) {
            if (step == 0) {
              // Distinguish carry-in from true loop invariants used across stripes.
              bool usedLater = llvm::any_of(opnd.getUsers(), [&](Operation *u) {
                return u->getNumResults() == 1 && valStripe.lookup(u->getResult(0)) > 0;
              });
              if (usedLater)
                continue;
            }
            if (boundary && boundary != opnd)
              return success();
            boundary = opnd;
          }
        }
      }
      if (!boundary)
        return success();
      Value carryIn = boundary;
      if (step == K - 1) {
        while (carryIn.getType() != finalVal.getType() ||
               (carryIn.getDefiningOp() &&
                carryIn.getDefiningOp()->getName() !=
                    finalVal.getDefiningOp()->getName())) {
          Operation *def = carryIn.getDefiningOp();
          if (!def || def->getNumOperands() != 1 || !carryIn.hasOneUse() ||
              def->getName().stripDialect() == "load" ||
              valStripe.lookup(def->getOperand(0)) != step - 1)
            return success();
          valStripe[carryIn] = step;
          carryIn = def->getOperand(0);
          ++unaryDepth;
        }
      } else {
        for (unsigned d = 0; d < unaryDepth; ++d) {
          Operation *def = carryIn.getDefiningOp();
          if (!def || def->getNumOperands() != 1 || !carryIn.hasOneUse() ||
              def->getName().stripDialect() == "load" ||
              valStripe.lookup(def->getOperand(0)) != step - 1)
            return success();
          valStripe[carryIn] = step;
          carryIn = def->getOperand(0);
        }
        if (carryIn.getType() != finalVal.getType())
          return success();
      }
      if (step > 0)
        c.next[step - 1] = carryIn;
      else
        c.init = carryIn;
      curOut = carryIn;
    }
    carries.push_back(std::move(c));
  }

  SmallVector<Operation *> invariantOps;
  SmallVector<SmallVector<Operation *>> stripeOps(K);
  for (Operation &op : entry.without_terminator()) {
    if (op.getName().stripDialect() == "store")
      continue;
    int64_t s = valStripe.lookup(op.getResult(0));
    if (s < 0)
      invariantOps.push_back(&op);
    else
      stripeOps[s].push_back(&op);
  }

  if (stripeOps[0].empty())
    return success();
  for (int64_t k = 0; k < K; ++k) {
    llvm::sort(stripeStores[k], [&](Operation *a, Operation *b) {
      return plan->args[cast<BlockArgument>(a->getOperand(0)).getArgNumber()]
                 .pubSlot <
             plan->args[cast<BlockArgument>(b->getOperand(0)).getArgNumber()]
                 .pubSlot;
    });
    if (stripeOps[k].size() != stripeOps[0].size() ||
        stripeStores[k].size() != stripeStores[0].size())
      return success();
  }

  // Ensure every striped Out/InOut public slot is written once per stripe.
  for (int64_t s = 0; size_t(s) < plan->pubSlots.size(); ++s) {
    auto *obj = plan->pubSlots[s].getAsObject();
    auto dir = obj->getString("direction");
    bool isStriped = llvm::any_of(plan->args, [&](const ArgDesc &d) {
      return d.pubSlot == s && d.striped;
    });
    if (!isStriped || (dir != "Out" && dir != "InOut"))
      continue;
    int64_t writtenAt0 = llvm::count_if(stripeStores[0], [&](Operation *st) {
      return plan->args[cast<BlockArgument>(st->getOperand(0)).getArgNumber()]
                 .pubSlot == s;
    });
    if (writtenAt0 != 1)
      return success();
  }

  llvm::SmallDenseSet<Operation *, 8> postSet(postStores.begin(),
                                              postStores.end());
  llvm::SmallDenseSet<unsigned, 8> isCarryProducer;
  for (auto &c : carries) {
    auto it = llvm::find_if(stripeOps[0], [&](Operation *op) {
      return op->getResult(0) == c.next[0];
    });
    if (it == stripeOps[0].end())
      return success();
    c.producerIdx = unsigned(std::distance(stripeOps[0].begin(), it));
    for (int64_t k = 0; k < K; ++k)
      if (stripeOps[k][c.producerIdx]->getResult(0) != c.next[k])
        return success();
    isCarryProducer.insert(c.producerIdx);
  }

  // Verify exact isomorphism across all K stripes.
  for (int64_t k = 0; k < K; ++k) {
    llvm::DenseMap<Value, Value> mapK;
    for (const auto &c : carries) {
      Value expectedIn = (k == 0) ? c.init : c.next[k - 1];
      if (mapK.contains(c.init))
        return success();
      mapK[c.init] = expectedIn;
    }
    llvm::SmallDenseSet<Operation *, 64> allowedUsers(stripeOps[k].begin(),
                                                      stripeOps[k].end());
    allowedUsers.insert(stripeStores[k].begin(), stripeStores[k].end());

    for (unsigned j = 0; j < stripeOps[0].size(); ++j) {
      Operation *op0 = stripeOps[0][j], *opK = stripeOps[k][j];
      if (op0->getName() != opK->getName() ||
          op0->getResultTypes() != opK->getResultTypes() ||
          op0->getNumOperands() != opK->getNumOperands())
        return success();
      auto n = op0->getName().stripDialect();
      if (n == "load") {
        const auto &d0 =
            plan->args[cast<BlockArgument>(op0->getOperand(0)).getArgNumber()];
        const auto &dK =
            plan->args[cast<BlockArgument>(opK->getOperand(0)).getArgNumber()];
        if (!d0.striped || !dK.striped || d0.pubSlot != dK.pubSlot ||
            d0.kind != dK.kind || d0.stepIdx != 0 || dK.stepIdx != k)
          return success();
      } else {
        if (op0->getAttrs() != opK->getAttrs())
          return success();
        for (unsigned p = 0; p < op0->getNumOperands(); ++p) {
          Value a0 = op0->getOperand(p), aK = opK->getOperand(p);
          if (auto it = mapK.find(a0); it != mapK.end()) {
            if (aK != it->second)
              return success();
          } else {
            if (valStripe.lookup(a0) != -1 || aK != a0)
              return success();
          }
        }
      }
      Value resK = opK->getResult(0);
      for (Operation *u : resK.getUsers()) {
        if (allowedUsers.contains(u))
          continue;
        if (!isCarryProducer.contains(j))
          return success();
        if (k + 1 < K) {
          if (!llvm::is_contained(stripeOps[k + 1], u))
            return success();
        } else if (!postSet.contains(u)) {
          return success();
        }
      }
      mapK[op0->getResult(0)] = resK;
    }

    for (unsigned s = 0; s < stripeStores[0].size(); ++s) {
      Operation *st0 = stripeStores[0][s], *stK = stripeStores[k][s];
      const auto &d0 =
          plan->args[cast<BlockArgument>(st0->getOperand(0)).getArgNumber()];
      const auto &dK =
          plan->args[cast<BlockArgument>(stK->getOperand(0)).getArgNumber()];
      if (!d0.striped || !dK.striped || d0.pubSlot != dK.pubSlot ||
          d0.stepIdx != 0 || dK.stepIdx != k ||
          stK->getOperand(4) != mapK.lookup(st0->getOperand(4)))
        return success();
    }
  }

  // Ensure invariantOps after postStores don't exist; move invariantOps before
  // stripe_loop and postStores after stripe_loop.
  OpBuilder b(fn.getContext());
  unsigned oldArgCount = entry.getNumArguments();
  SmallVector<Value> newArgs;
  SmallVector<Type> newArgTypes;
  for (size_t s = 0; s < plan->pubSlots.size(); ++s) {
    auto *obj = plan->pubSlots[s].getAsObject();
    auto dt = obj->getString("dtype");
    auto *dims = obj->getArray("dims");
    if (!dt || !dims)
      return failure();
    Type elemTy = scalarTypeFromABI(b, *dt);
    if (!elemTy)
      return failure();
    Type argTy = elemTy;
    if (!dims->empty()) {
      int64_t r = dims->size();
      SmallVector<int64_t> dynShape(r, ShapedType::kDynamic);
      auto layout = StridedLayoutAttr::get(fn.getContext(),
                                           ShapedType::kDynamic, dynShape);
      argTy = MemRefType::get(dynShape, elemTy, layout);
    }
    newArgTypes.push_back(argTy);
    newArgs.push_back(entry.addArgument(argTy, fn.getLoc()));
  }

  // Rebind invariant operations and post-loop stores to the new public args.
  for (unsigned i = 0; i < oldArgCount; ++i)
    if (!plan->args[i].striped && plan->args[i].kind == ArgKind::Scalar)
      entry.getArgument(i).replaceAllUsesWith(newArgs[plan->args[i].pubSlot]);

  Operation *activeOp = activeVal.getDefiningOp();
  activeOp->setOperand(0, newArgs[plan->domainPubSlot]);

  for (Operation *inv : invariantOps) {
    auto n = inv->getName().stripDialect();
    if (n == "load") {
      auto oldArg = cast<BlockArgument>(inv->getOperand(0));
      inv->setOperand(0, newArgs[plan->args[oldArg.getArgNumber()].pubSlot]);
      inv->removeAttr("block.member_axes");
    }
    inv->moveBefore(entry.getTerminator());
  }

  // Emit block.stripe_loop right before terminator, followed by postStores.
  b.setInsertionPoint(entry.getTerminator());
  Location loc = fn.getLoc();
  SmallVector<Value> initVals;
  SmallVector<Type> carryTypes;
  for (const auto &c : carries) {
    initVals.push_back(c.init);
    carryTypes.push_back(c.init.getType());
  }
  OperationState loopSt(loc, "block.stripe_loop");
  loopSt.addOperands(initVals);
  loopSt.addTypes(carryTypes);
  loopSt.addAttribute("count", b.getI64IntegerAttr(K));
  loopSt.addAttribute("stride", b.getI64IntegerAttr(plan->stride));
  loopSt.addRegion();
  Operation *stripeLoop = b.create(loopSt);

  Block *loopBody = b.createBlock(&stripeLoop->getRegion(0));
  Value roundArg = loopBody->addArgument(b.getIndexType(), loc);
  IRMapping bodyMap;
  for (const auto &c : carries)
    bodyMap.map(c.init, loopBody->addArgument(c.init.getType(), loc));

  OperationState localActiveSt(loc, "block.active");
  localActiveSt.addOperands({newArgs[plan->domainPubSlot], rowsVal, colsVal});
  localActiveSt.addTypes(activeVal.getType());
  Value localActive = b.create(localActiveSt)->getResult(0);
  bodyMap.map(activeVal, localActive);

  for (Operation *op0 : stripeOps[0]) {
    auto n = op0->getName().stripDialect();
    if (n == "load") {
      auto oldArg = cast<BlockArgument>(op0->getOperand(0));
      int64_t pubSlot = plan->args[oldArg.getArgNumber()].pubSlot;
      OperationState ldSt(op0->getLoc(), "block.stripe_load");
      ldSt.addOperands(
          {newArgs[pubSlot], roundArg, rowsVal, colsVal, localActive});
      ldSt.addAttribute("stride", b.getI64IntegerAttr(plan->stride));
      ldSt.addTypes(op0->getResultTypes());
      bodyMap.map(op0->getResult(0), b.create(ldSt)->getResult(0));
    } else {
      b.clone(*op0, bodyMap);
    }
  }
  for (Operation *st0 : stripeStores[0]) {
    auto oldArg = cast<BlockArgument>(st0->getOperand(0));
    int64_t pubSlot = plan->args[oldArg.getArgNumber()].pubSlot;
    OperationState stSt(st0->getLoc(), "block.stripe_store");
    stSt.addOperands({newArgs[pubSlot], roundArg, rowsVal, colsVal, localActive,
                      bodyMap.lookupOrDefault(st0->getOperand(4))});
    stSt.addAttribute("stride", b.getI64IntegerAttr(plan->stride));
    b.create(stSt);
  }
  SmallVector<Value> yieldVals;
  for (const auto &c : carries)
    yieldVals.push_back(bodyMap.lookup(c.next[0]));
  OperationState yieldSt(loc, "block.yield");
  yieldSt.addOperands(yieldVals);
  b.create(yieldSt);

  // Move post-loop invariant stores after stripe_loop and rewire carry results.
  for (auto [idx, c] : llvm::enumerate(carries))
    c.next[K - 1].replaceAllUsesWith(stripeLoop->getResult(idx));
  for (Operation *ps : postStores) {
    auto oldArg = cast<BlockArgument>(ps->getOperand(0));
    ps->setOperand(0, newArgs[plan->args[oldArg.getArgNumber()].pubSlot]);
    ps->removeAttr("block.member_axes");
    ps->moveBefore(entry.getTerminator());
  }

  // Erase unrolled stripe stores and ops in reverse order.
  for (int64_t k = K - 1; k >= 0; --k) {
    for (Operation *st : llvm::reverse(stripeStores[k]))
      st->erase();
    for (Operation *op : llvm::reverse(stripeOps[k]))
      op->erase();
  }

  // Erase old projected arguments and update function ABI metadata.
  BitVector deadArgs(entry.getNumArguments(), false);
  deadArgs.set(0, oldArgCount);
  entry.eraseArguments(deadArgs);
  fn.removeArgAttrsAttr();
  fn.setFunctionType(b.getFunctionType(newArgTypes, {}));
  fn->setAttr("kir.abi", fn->getAttr("kir.public_abi"));
  fn->removeAttr("kir.public_abi");
  fn->removeAttr("kir.iteration");
  fn->setAttr("kir.domain_arg", b.getI64IntegerAttr(plan->domainPubSlot));
  return success();
}
} // namespace intentdsl::block
