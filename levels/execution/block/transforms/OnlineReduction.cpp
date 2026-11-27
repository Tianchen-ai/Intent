#include "compilerinfra/Assembly.h"
#include "compilerinfra/Proofs.h"
#include "intentdsl/levels/algorithm/kir/ir/OnlineNormalization.h"
#include "intentdsl/levels/execution/block/decisions/RowCapacity.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace CI_UNIT_NAMESPACE {
using namespace mlir;
namespace {
Operation *producer(Value v, StringRef name) {
  auto *p = v.getDefiningOp();
  return p && p->getName().getStringRef() == name ? p : nullptr;
}

struct Reduction {
  Operation *loop, *reduce;
  Value init, input, mask;
};

std::optional<Reduction> reduction(Operation &loop, StringRef algebra) {
  if (loop.getName().getStringRef() != "block.tiles" || loop.getNumResults() != 1 ||
      loop.getNumOperands() != 2 || !llvm::hasSingleElement(loop.getRegion(0)))
    return {};
  Block &body = loop.getRegion(0).front();
  if (body.getNumArguments() != 2 || body.empty() || body.back().getNumOperands() != 1)
    return {};
  auto *combine = producer(body.back().getOperand(0), "block.combine");
  auto *init = producer(loop.getOperand(1), "block.identity");
  if (!combine || !init || combine->getOperand(0) != body.getArgument(1) ||
      init->getAttrOfType<StringAttr>("algebra") != algebra)
    return {};
  auto *red = producer(combine->getOperand(1), "block.reduce");
  if (!red || red->getAttrOfType<StringAttr>("algebra") != algebra ||
      !cast<ShapedType>(red->getResult(0).getType()).getElementType().isF32())
    return {};
  return Reduction{&loop, red, loop.getOperand(1), red->getOperand(0), red->getOperand(1)};
}

// Match corresponding reads and pure expressions, including masks, access
// projections and numerical attributes. Recomputing an impure score is illegal.
bool sameScore(Value a, Value b, Block &first, Block &second, IRMapping &map,
               SmallVectorImpl<Value> &reads) {
  if (Value mapped = map.lookupOrNull(a))
    return mapped == b;
  if (a == b)
    return a.getParentBlock() != &first && b.getParentBlock() != &second;
  auto *x = a.getDefiningOp(), *y = b.getDefiningOp();
  if (!x || !y || x->getBlock() != &first || y->getBlock() != &second ||
      x->getName() != y->getName() || x->getNumResults() != 1 || y->getNumResults() != 1 ||
      x->getNumRegions() || y->getNumRegions() || a.getType() != b.getType() ||
      x->getAttrs() != y->getAttrs() || x->getNumOperands() != y->getNumOperands())
    return false;
  if (x->getName().getStringRef() == "block.load")
    reads.push_back(x->getOperand(0));
  else if (!isMemoryEffectFree(x) || !isMemoryEffectFree(y))
    return false;
  for (auto [l, r] : llvm::zip(x->getOperands(), y->getOperands()))
    if (!sameScore(l, r, first, second, map, reads))
      return false;
  map.map(a, b);
  return true;
}

bool fuse(Reduction peak, Reduction mass) {
  auto *exp = producer(mass.input, "block.exp");
  auto *sub = exp ? producer(exp->getOperand(0), "block.sub") : nullptr;
  if (!sub || sub->getOperand(1) != peak.loop->getResult(0) ||
      peak.loop->getOperand(0) != mass.loop->getOperand(0))
    return false;
  Block &first = peak.loop->getRegion(0).front(), &second = mass.loop->getRegion(0).front();
  IRMapping same;
  same.map(first.getArgument(0), second.getArgument(0));
  SmallVector<Value> reads;
  if (!sameScore(peak.input, sub->getOperand(0), first, second, same, reads) ||
      !sameScore(peak.mask, mass.mask, first, second, same, reads) || reads.empty() ||
      !compilerinfra::proveValueReplay(peak.loop, mass.loop, reads).proven)
    return false;
  // Checks above describe the complete reduction chain. Exclude side outputs,
  // competing effects, and a use of the peak between the two loops.
  auto safe = [](Operation &op) {
    return isMemoryEffectFree(&op) || op.getName().getStringRef() == "block.load";
  };
  if (!llvm::all_of(first.without_terminator(), safe) ||
      !llvm::all_of(second.without_terminator(), safe))
    return false;
  for (auto *use : peak.loop->getResult(0).getUsers()) {
    if (use == sub)
      continue;
    auto *ancestor = mass.loop->getBlock()->findAncestorOpInBlock(*use);
    if (!ancestor || ancestor == mass.loop || !mass.loop->isBeforeInBlock(ancestor))
      return false;
  }

  OpBuilder b(mass.loop);
  Location loc = mass.loop->getLoc();
  Type row = peak.init.getType();
  auto predicate = cast<RankedTensorType>(row).clone(b.getI1Type());
  OperationState state(loc, "block.tiles");
  state.addOperands({peak.loop->getOperand(0), peak.init, mass.init});
  state.addTypes({row, row});
  state.addAttribute("block.online_normalization", b.getUnitAttr());
  state.addRegion();
  Operation *loop = b.create(state);
  Block *body = b.createBlock(&loop->getRegion(0), {}, {b.getIndexType(), row, row}, {loc, loc, loc});
  IRMapping values;
  values.map(first.getArgument(0), body->getArgument(0));
  values.map(first.getArgument(1), body->getArgument(1));
  for (auto &op : first.without_terminator())
    b.clone(op, values);
  Value newMax = values.lookup(first.back().getOperand(0));
  auto make = [&](StringRef name, ValueRange args, Type type) {
    OperationState st(loc, ("block." + name).str());
    st.addOperands(args);
    st.addTypes(type);
    return b.create(st)->getResult(0);
  };
  auto update = intentdsl::kir::mergeNormalizationPanel(
      Value(body->getArgument(1)), newMax, Value(body->getArgument(2)),
      [&](Value nextPeak) {
        Value present = make("ne", {nextPeak, peak.init}, predicate);
        return make("select", {present, nextPeak, mass.init}, row);
      },
      [&](Value prior, Value origin) {
        Operation *delta = b.clone(*sub, values);
        delta->setOperands({prior, origin});
        delta->getResult(0).setType(row);
        Operation *scale = b.clone(*exp, values);
        scale->setOperand(0, delta->getResult(0));
        scale->getResult(0).setType(row);
        return scale->getResult(0);
      },
      [&](Value origin) {
        IRMapping weights;
        weights.map(sub->getOperand(0), values.lookup(peak.input));
        weights.map(sub->getOperand(1), origin);
        weights.map(mass.mask, values.lookup(peak.mask));
        b.clone(*sub, weights);
        auto *weight = b.clone(*exp, weights);
        return std::pair{b.clone(*mass.reduce, weights)->getResult(0),
                         weight->getResult(0)};
      },
      [&](Value oldMass, Value scale) {
        return make("mul", {oldMass, scale}, row);
      },
      [&](Value previous, Value current) {
        IRMapping operands;
        operands.map(second.getArgument(1), previous);
        operands.map(mass.reduce->getResult(0), current);
        return b.clone(*second.back().getOperand(0).getDefiningOp(), operands)
            ->getResult(0);
      });
  OperationState yield(loc, "block.yield");
  yield.addOperands({newMax, std::get<1>(update)});
  b.create(yield);

  b.setInsertionPointAfter(loop);
  // All valid scores at -infinity give exp(-inf - -inf). A leading
  // all-negative-infinite panel must still recover if a later score is finite.
  Value ordinary = make("ne", {loop->getResult(0), peak.init}, predicate);
  Value nanRow = make("sub", {peak.init, peak.init}, row);
  Value finalMass = make("select", {ordinary, loop->getResult(1), nanRow}, row);
  mass.loop->getResult(0).replaceAllUsesWith(finalMass);
  mass.loop->erase();
  peak.loop->getResult(0).replaceAllUsesWith(loop->getResult(0));
  peak.loop->erase();
  return true;
}
} // namespace

void formOnlineReductions(func::FuncOp fn, const compilerinfra::TargetProfile &profile) {
  auto caps = profile.services.get<intentdsl::RowCapacity>();
  if (!caps || !caps->onlineRowReductions || caps->rows.empty() ||
      fn->getAttrOfType<StringAttr>("kir.normalization") != "online-eligible" ||
      fn->getAttrOfType<StringAttr>("block.topology") != "stream")
    return;
  auto extent = caps->width(fn);
  if (!extent) {
    llvm::consumeError(extent.takeError());
    return;
  }
  int64_t resident = llvm::max_element(caps->rows, [](const auto &a, const auto &b) {
    return a.width < b.width;
  })->width;
  if (*extent <= resident)
    return;
  std::optional<Reduction> peak;
  for (auto &op : llvm::make_early_inc_range(fn.front())) {
    if (auto r = reduction(op, "max_num"))
      peak = r;
    else if (peak) {
      if (auto mass = reduction(op, "sum"); mass && fuse(*peak, *mass)) {
        // Runtime reuse is limited to the existing large-row width class.
        fn->setAttr("block.guard", Builder(fn.getContext()).getDenseI64ArrayAttr({resident + 1, -1}));
        peak.reset();
      } else if (!isMemoryEffectFree(&op))
        peak.reset();
    }
  }
}
} // namespace CI_UNIT_NAMESPACE
