#pragma once
#include "intentdsl/levels/execution/cpu/ir/ScoreCache.h"
#include "intentdsl/levels/execution/cpu/ir/Task.h"

namespace intentdsl::cpu {
// A batch of owners supplies adjacent BF16 rows of the left view. The fold
// traverses rows of a common right view. Multiplication therefore gives a
// [tokens, owners] score panel without narrowing either operand.
inline int64_t scoreSlotLimit(const RowMachine &machine) {
  // Leave space for the activation panel, value accumulators and index rows.
  return std::max<int64_t>(batchScoreSlots,
                           machine.l2CacheBytes / (4 * sizeof(float)));
}
struct OrderedDotPanel {
  mlir::Operation *dot, *query, *key;
  int64_t depth, tokens;
};

inline std::optional<OrderedDotPanel> orderedDotPanel(mlir::Operation &space,
                                                      const CallFacts &facts,
                                                      int64_t axis,
                                                      int64_t batch) {
  using namespace mlir;
  if (axis < 0 ||
      size_t(axis) >= space.getRegion(0).front().getNumArguments() ||
      batch <= 0)
    return {};
  auto &body = space.getRegion(0).front();
  auto owner = body.getArgument(axis);
  for (auto loop : body.getOps<scf::ForOp>()) {
    if (detail::staticIndexValue(loop.getLowerBound(), facts) != 0 ||
        detail::staticIndexValue(loop.getStep(), facts) != 1)
      continue;
    Value limit = loop.getUpperBound();
    // The bounded owner prefix is represented as min(prefix, original limit).
    if (auto sel = limit.getDefiningOp<arith::SelectOp>()) {
      auto cmp = sel.getCondition().getDefiningOp<arith::CmpIOp>();
      if (cmp && cmp.getPredicate() == arith::CmpIPredicate::slt &&
          cmp.getLhs() == sel.getTrueValue() &&
          cmp.getRhs() == sel.getFalseValue())
        limit = sel.getFalseValue();
    }
    auto tokens = detail::staticIndexValue(limit, facts);
    bool writes =
        loop.walk([](Operation *op) {
              if (op->getNumRegions())
                return WalkResult::advance();
              if (op->getName().getStringRef() == "cpu.buffer_store" &&
                  op->getOperand(1).getDefiningOp() &&
                  op->getOperand(1).getDefiningOp()->getName().getStringRef() ==
                      "cpu.local_buffer")
                return WalkResult::advance();
              auto mem = dyn_cast<MemoryEffectOpInterface>(op);
              return (!mem || mem.hasEffect<MemoryEffects::Write>() ||
                      mem.hasEffect<MemoryEffects::Free>())
                         ? WalkResult::interrupt()
                         : WalkResult::advance();
            })
            .wasInterrupted();
    if (writes)
      continue;
    if (!tokens || *tokens <= 0)
      continue;
    for (auto &dot : loop.getBody()->without_terminator()) {
      if (dot.getName().stripDialect() != "lane_sum" ||
          !dot.hasAttr("denormal-fp-math") ||
          !compilerinfra::allowsSubnormalFlush(&dot) ||
          !arith::bitEnumContainsAll(kir::getFastMathFlags(&dot),
                                     arith::FastMathFlags::reassoc))
        continue;
      auto mul = dot.getOperand(0).getDefiningOp<arith::MulFOp>();
      if (!mul || !compilerinfra::allowsSubnormalFlush(mul))
        continue;
      Operation *q = nullptr, *k = nullptr;
      int64_t depth = 0;
      for (Value v : mul.getOperands()) {
        auto ext = v.getDefiningOp<arith::ExtFOp>();
        auto *read = ext ? ext.getIn().getDefiningOp() : nullptr;
        if (!read || read->getName().stripDialect() != "slice_read")
          continue;
        auto arg = dyn_cast<BlockArgument>(read->getOperand(0));
        auto mt = arg ? dyn_cast<MemRefType>(arg.getType()) : MemRefType{};
        if (!mt || !mt.getElementType().isBF16() || mt.getRank() < 2 ||
            read->getAttrOfType<IntegerAttr>("axis").getInt() !=
                mt.getRank() - 1)
          continue;
        auto *dims = facts.shapeOf(arg);
        auto *rec = arg.getArgNumber() < facts.arguments.size()
                        ? facts.arguments[arg.getArgNumber()].getAsObject()
                        : nullptr;
        auto *strides = rec ? rec->getArray("strides") : nullptr;
        if (!dims || !strides || dims->size() != unsigned(mt.getRank()) ||
            strides->size() != dims->size())
          continue;
        int64_t d = dims->back().getAsInteger().value_or(0);
        if (d < 2 || d % 2 || (depth && depth != d) ||
            strides->back().getAsInteger() != 1 ||
            (*strides)[dims->size() - 2].getAsInteger() != d)
          continue;
        auto coords = read->getOperands().drop_front();
        if (coords.empty())
          continue;
        auto common = [&](Value x) {
          auto independent = [&](auto &&self, Value value) -> bool {
            if (auto ba = dyn_cast<BlockArgument>(value))
              return ba.getOwner() == &body
                         ? ba != owner
                         : isa<func::FuncOp>(ba.getOwner()->getParentOp());
            auto *op = value.getDefiningOp();
            return !op->getNumRegions() &&
                   llvm::all_of(op->getOperands(),
                                [&](Value in) { return self(self, in); });
          };
          return independent(independent, x);
        };
        if (read->getBlock() == &body && coords.back() == owner &&
            llvm::all_of(coords.drop_back(), common))
          q = read;
        if (read->getBlock() == loop.getBody() &&
            llvm::all_of(coords.drop_back(), common) &&
            coords.back() == loop.getInductionVar() &&
            llvm::all_of(
                coords.drop_back(),
                [&](Value x) { return loop.isDefinedOutsideOfLoop(x); }) &&
            (*dims)[dims->size() - 2].getAsInteger().value_or(0) >= *tokens)
          k = read;
        depth = d;
      }
      if (q && k)
        return OrderedDotPanel{&dot, q, k, depth, *tokens};
    }
  }
  return {};
}

inline std::optional<OrderedDotPanel>
selectOrderedDot(mlir::Operation &space, const compilerinfra::TargetProfile &p,
                 int64_t axis, int64_t batch) {
  auto *facts = p.services.get<CallFacts>();
  auto *machine = p.services.get<RowMachine>();
  if (!facts || !machine || batch < machine->packedOwners ||
      batch > std::max(machine->packedOwners, machine->orderedMatrixOwners) ||
      !p.features.count("amx_bf16"))
    return {};
  std::string impl = "exits/" + p.name + "::onednn-brgemm-bf16";
  if (auto err = p.checkImplementation(impl)) {
    llvm::consumeError(std::move(err));
    return {};
  }
  auto panel = orderedDotPanel(space, *facts, axis, batch);
  // Use a whole packed owner vector; amortize each VNNI-pair load across a
  // matrix panel. A four-vector reduction is the minimum wide-slice backend.
  auto lanes = machine->vectorBits / 32;
  if (!panel || lanes <= 0 ||
      panel->tokens > scoreSlotLimit(*machine) / batch ||
      panel->depth > 2 * batchScoreSlots / batch || panel->depth < 4 * lanes ||
      panel->depth % lanes || panel->tokens < batch)
    return {};
  return panel;
}
} // namespace intentdsl::cpu
