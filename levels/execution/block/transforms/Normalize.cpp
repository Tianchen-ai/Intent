#include "compilerinfra/Assembly.h"
#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
#include "intentdsl/levels/execution/block/ir/Contraction.h"
#include "intentdsl/levels/execution/block/ir/FoldStripes.h"
#include "intentdsl/levels/execution/block/ir/Owners.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace CI_UNIT_NAMESPACE {
using namespace mlir;
namespace {
void partitionOwnerReductions(func::FuncOp fn,
                              const compilerinfra::TargetProfile &profile) {
  auto &body = fn.front();
  if (!fn->hasAttr("kir.owner_axis") ||
      fn->getAttrOfType<StringAttr>("block.topology") != "stream")
    return;
  auto facts = profile.services.get<intentdsl::CallFacts>();
  if (!facts)
    return;
  auto abiAttr = fn->getAttrOfType<StringAttr>("kir.abi");
  if (!abiAttr)
    return;
  auto abi = llvm::json::parse(abiAttr.getValue());
  if (!abi) {
    llvm::consumeError(abi.takeError());
    return;
  }
  for (auto &slot : *abi->getAsArray())
    if (slot.getAsObject()->getString("direction") == "InOut")
      return;
  auto [owners, columns] = facts->geometry(fn);
  if (!owners || owners >= facts->participants)
    return;

  SmallVector<Operation *> stores;
  Value rowsVal;
  for (auto &op : body) {
    auto n = op.getName().stripDialect();
    if (n == "owner_store")
      stores.push_back(&op);
    else if (n == "rows")
      rowsVal = op.getResult(0);
    else if (n == "tiles") {
      if (op.getNumResults() != 1 ||
          isa<TupleType>(op.getResult(0).getType()))
        return;
    } else if (n != "pid" && n != "identity" && n != "cast" && n != "return")
      return;
  }
  if (stores.size() < 2 || !rowsVal)
    return;

  auto isFullMemberView = [&](BlockArgument arg) {
    auto mapAttr = fn.getArgAttrOfType<AffineMapAttr>(arg.getArgNumber(),
                                                      "kir.indexing");
    if (!mapAttr)
      return true;
    auto map = mapAttr.getValue();
    for (unsigned d = 0; d < map.getNumDims(); ++d) {
      bool used = llvm::any_of(map.getResults(), [&](AffineExpr e) {
        return e.isFunctionOfDim(d);
      });
      if (!used)
        return false;
    }
    return true;
  };

  struct Chain {
    Operation *loop;
    Operation *castOp;
    Operation *store;
    SmallVector<Value, 4> fullViews;
  };
  SmallVector<Chain> chains;
  llvm::SmallPtrSet<Operation *, 32> takenLoops;
  llvm::SmallDenseSet<Value, 32> seenOutSlots;
  for (auto *store : stores) {
    if (!seenOutSlots.insert(store->getOperand(0)).second)
      return;
    Operation *castOp = store->getOperand(2).getDefiningOp();
    Operation *loop = castOp;
    if (castOp && castOp->getName().stripDialect() == "cast") {
      if (!castOp->hasOneUse() || castOp->getBlock() != &body)
        return;
      loop = castOp->getOperand(0).getDefiningOp();
    } else {
      castOp = nullptr;
    }
    if (!loop || loop->getBlock() != &body ||
        loop->getName().stripDialect() != "tiles" || !loop->hasOneUse() ||
        !takenLoops.insert(loop).second)
      return;
    bool dependent = false;
    SmallVector<Value, 4> views;
    llvm::SmallDenseSet<Value, 4> seenViews;
    loop->walk([&](Operation *inner) {
      if (inner->getName().getStringRef() == "block.load") {
        if (auto arg = dyn_cast<BlockArgument>(inner->getOperand(0));
            arg && arg.getOwner() == &body && isFullMemberView(arg) &&
            seenViews.insert(arg).second)
          views.push_back(arg);
      }
      for (auto v : inner->getOperands()) {
        auto *producer = v.getDefiningOp();
        if (producer && producer->getBlock() == &body && producer != loop &&
            producer->getName().stripDialect() != "rows" &&
            producer->getName().stripDialect() != "identity")
          dependent = true;
      }
    });
    if (dependent)
      return;
    chains.push_back({loop, castOp, store, std::move(views)});
  }

  SmallVector<SmallVector<unsigned, 2>> groups;
  SmallVector<int> chainGroup(chains.size(), -1);
  for (unsigned i = 0; i < chains.size(); ++i) {
    if (chainGroup[i] >= 0)
      continue;
    int gIdx = groups.size();
    groups.push_back({i});
    chainGroup[i] = gIdx;
    for (unsigned j = i + 1; j < chains.size() && groups[gIdx].size() < 4; ++j) {
      if (chainGroup[j] >= 0 ||
          chains[j].loop->getOperand(0) != chains[i].loop->getOperand(0))
        continue;
      bool shares = llvm::any_of(groups[gIdx], [&](unsigned m) {
        return llvm::any_of(chains[m].fullViews, [&](Value v) {
          return llvm::is_contained(chains[j].fullViews, v);
        });
      });
      if (shares) {
        groups[gIdx].push_back(j);
        chainGroup[j] = gIdx;
      }
    }
  }

  bool anyMulti = llvm::any_of(
      groups, [](const auto &g) { return g.size() >= 2; });
  if (groups.size() < 2 && !anyMulti)
    return;

  OpBuilder b(body.getTerminator());
  Location loc = fn.getLoc();
  SmallVector<Operation *> groupLoops(groups.size(), nullptr);
  for (auto [gIdx, members] : llvm::enumerate(groups)) {
    if (members.size() == 1) {
      groupLoops[gIdx] = chains[members[0]].loop;
      continue;
    }
    Value domain = chains[members[0]].loop->getOperand(0);
    OperationState fusedSt(loc, "block.tiles");
    fusedSt.addOperands(domain);
    for (unsigned m : members) {
      fusedSt.addOperands(chains[m].loop->getOperand(1));
      fusedSt.addTypes(chains[m].loop->getResult(0).getType());
    }
    fusedSt.addRegion();
    Operation *fused = b.create(fusedSt);
    Block *fb = new Block;
    fused->getRegion(0).push_back(fb);
    Value iv = fb->addArgument(b.getIndexType(), loc);
    for (unsigned m : members)
      fb->addArgument(chains[m].loop->getResult(0).getType(), loc);

    OpBuilder ib(fn.getContext());
    ib.setInsertionPointToStart(fb);
    OperationState colSt(loc, "block.columns");
    colSt.addOperands(iv);
    colSt.addTypes(
        RankedTensorType::get({ShapedType::kDynamic}, ib.getIndexType()));
    Value cols = ib.create(colSt)->getResult(0);
    OperationState actSt(loc, "block.active");
    actSt.addOperands({domain, rowsVal, cols});
    actSt.addTypes(RankedTensorType::get(
        {ShapedType::kDynamic, ShapedType::kDynamic}, ib.getI1Type()));
    Value act = ib.create(actSt)->getResult(0);

    llvm::DenseMap<Value, Value> sharedLoads;
    SmallVector<Value> yieldVals;
    for (auto [k, m] : llvm::enumerate(members)) {
      Block &src = chains[m].loop->getRegion(0).front();
      IRMapping map;
      map.map(src.getArgument(0), iv);
      map.map(src.getArgument(1), fb->getArgument(1 + k));
      for (auto &inst : src.without_terminator()) {
        auto iname = inst.getName().stripDialect();
        if (iname == "columns") {
          map.map(inst.getResult(0), cols);
          continue;
        }
        if (iname == "active") {
          map.map(inst.getResult(0), act);
          continue;
        }
        if (iname == "load") {
          Value view = inst.getOperand(0);
          if (auto existing = sharedLoads.lookup(view)) {
            map.map(inst.getResult(0), existing);
            continue;
          }
          Operation *cloned = ib.clone(inst, map);
          sharedLoads[view] = cloned->getResult(0);
          map.map(inst.getResult(0), cloned->getResult(0));
          continue;
        }
        if (inst.getNumResults() == 1 && inst.getNumRegions() == 0 &&
            llvm::StringSwitch<bool>(iname)
                .Cases("cast", "add", "sub", "mul", "div", "rsqrt", "exp",
                       "sqrt", "maximum", "constant", true)
                .Default(false)) {
          SmallVector<Value, 4> mappedOps;
          for (Value opnd : inst.getOperands())
            mappedOps.push_back(map.lookupOrDefault(opnd));
          Value reused;
          for (Operation &prev : *fb) {
            if (prev.getName() == inst.getName() &&
                prev.getNumResults() == 1 &&
                prev.getResult(0).getType() == inst.getResult(0).getType() &&
                prev.getOperands() == ValueRange(mappedOps) &&
                prev.getAttrDictionary() == inst.getAttrDictionary()) {
              reused = prev.getResult(0);
              break;
            }
          }
          if (reused) {
            map.map(inst.getResult(0), reused);
            continue;
          }
        }
        ib.clone(inst, map);
      }
      yieldVals.push_back(map.lookup(src.back().getOperand(0)));
    }
    OperationState ySt(loc, "block.yield");
    ySt.addOperands(yieldVals);
    ib.create(ySt);

    for (auto [k, m] : llvm::enumerate(members)) {
      chains[m].loop->getResult(0).replaceAllUsesWith(fused->getResult(k));
      chains[m].loop->erase();
      chains[m].loop = fused;
    }
    groupLoops[gIdx] = fused;
  }

  if (groups.size() < 2)
    return;

  b.setInsertionPoint(body.getTerminator());
  for (auto [gIdx, members] : llvm::enumerate(groups)) {
    OperationState region(loc, "block.output_partition");
    region.addAttribute("index", b.getI32IntegerAttr(gIdx));
    region.addRegion();
    auto *branch = b.create(region);
    auto *pBlock = new Block;
    branch->getRegion(0).push_back(pBlock);
    groupLoops[gIdx]->moveBefore(pBlock, pBlock->end());
    for (unsigned m : members)
      if (chains[m].castOp)
        chains[m].castOp->moveBefore(pBlock, pBlock->end());
    for (unsigned m : members)
      chains[m].store->moveBefore(pBlock, pBlock->end());
    OpBuilder pb(fn.getContext());
    pb.setInsertionPointToEnd(pBlock);
    pb.create(OperationState(loc, "block.partition_end"));
  }
}
} // namespace

struct Normalize : PassWrapper<Normalize, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Normalize)
  compilerinfra::TargetProfile profile;
  explicit Normalize(const compilerinfra::TargetProfile &p) : profile(p) {}
  void runOnOperation() final {
    for (auto fn : getOperation().getOps<func::FuncOp>()) {
      if (intentdsl::kir::orderedOwners(fn) &&
          failed(intentdsl::block::forwardOwners(fn))) {
        signalPassFailure();
        return;
      }
      if (!intentdsl::kir::orderedOwners(fn) &&
          !intentdsl::kir::matrixDomain(fn)) {
        if (failed(intentdsl::block::foldStripes(fn, profile))) {
          signalPassFailure();
          return;
        }
        partitionOwnerReductions(fn, profile);
      }
    }
    getOperation().walk([](Operation *op) {
      if (op->getName().getDialectNamespace() != "block" ||
          op->getNumResults() != 1)
        return;
      auto result = dyn_cast<RankedTensorType>(op->getResult(0).getType());
      if (!result || result.getRank() != 2)
        return;
      auto name = op->getName().stripDialect();
      if (name != "add" && name != "sub" && name != "mul" && name != "div")
        return;
      OpBuilder b(op);
      for (auto &operand : op->getOpOperands()) {
        auto t = dyn_cast<RankedTensorType>(operand.get().getType());
        if (!t || t.getRank() != 1)
          continue;
        OperationState expand(op->getLoc(), "block.broadcast_rows");
        expand.addOperands(operand.get());
        expand.addTypes(RankedTensorType::get({ShapedType::kDynamic, 1},
                                              t.getElementType()));
        operand.set(b.create(expand)->getResult(0));
      }
    });
  }
};
void normalize(OpPassManager &pm, const compilerinfra::TargetProfile &profile) {
  pm.addPass(std::make_unique<Normalize>(profile));
}
} // namespace CI_UNIT_NAMESPACE
