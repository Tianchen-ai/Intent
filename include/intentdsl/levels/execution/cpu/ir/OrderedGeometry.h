#pragma once
#include "intentdsl/levels/execution/cpu/cost/PanelLayout.h"
#include "intentdsl/levels/execution/cpu/ir/OrderedValue.h"
#include <numeric>

namespace intentdsl::cpu {
inline mlir::DictionaryAttr
orderedGeometry(mlir::Operation &space, const compilerinfra::TargetProfile &p) {
  using namespace mlir;
  auto machine = p.services.get<RowMachine>();
  auto facts = p.services.get<CallFacts>();
  auto batch = space.getAttrOfType<IntegerAttr>("cpu.owner_batch");
  if (!machine || !facts || !machine->ownerRegisters || !batch ||
      batch.getInt() < 2)
    return {};
  int64_t rows = batch.getInt();
  auto score = space.getAttrOfType<IntegerAttr>("cpu.score_tile");
  int64_t tokens = score ? score.getInt() : machine->orderedDotTokens;
  int64_t columns = 0;
  space.walk([&](Operation *op) {
    if (op->getName().stripDialect() != "slice_write")
      return;
    auto *dims = facts->shapeOf(op->getOperand(1));
    auto axis = op->getAttrOfType<IntegerAttr>("axis");
    if (axis && dims && uint64_t(axis.getInt()) < dims->size())
      columns =
          std::max(columns, (*dims)[axis.getInt()].getAsInteger().value_or(0));
  });
  int64_t lanes = machine->vectorBits / 32;
  int64_t width = columns;
  if (auto pv = space.getAttrOfType<IntegerAttr>("cpu.value_width"))
    width = pv.getInt();
  else if (rows % 4 == 0 && columns > lanes && columns % lanes == 0 &&
           rows * columns >= machine->ownerRegisters * lanes) {
    // Leave registers for the two score operands and reductions.
    int64_t registers = machine->ownerRegisters / 2;
    width = std::gcd(columns, lanes * std::max<int64_t>(1, registers / rows));
  }
  int64_t cached = 0;
  Builder b(space.getContext());
  if (!tokens || !machine->ownerStackBytes) {
    if (!space.hasAttr("cpu.batch_score_window") || score || !columns)
      return {};
    return b.getDictionaryAttr(
        {b.getNamedAttr("value_width", b.getI64IntegerAttr(width))});
  }
  if (auto window =
          space.getAttrOfType<IntegerAttr>("cpu.batch_score_window")) {
    cached = window.getInt();
    auto loops = space.getRegion(0).front().getOps<scf::ForOp>();
    auto span = loops.empty() ? std::nullopt
                              : detail::staticIndexValue(
                                    (*loops.begin()).getUpperBound(), *facts);
    if (!score && (!span || *span > cached)) {
      // Explicit score storage shares the private frame with admitted
      // vector operands, the output accumulator and downstream temporaries.
      int64_t output = width < columns ? rows * columns * sizeof(float) : 0;
      int64_t budget = machine->ownerStackBytes / 4 - output;
      int64_t tileBytes = rows * tokens * sizeof(float);
      cached =
          std::min(cached, std::max<int64_t>(1, budget / tileBytes) * tokens);
    }
  }
  NamedAttrList geometry;
  geometry.set("token_tile", b.getI64IntegerAttr(tokens));
  geometry.set("value_width", b.getI64IntegerAttr(width));
  geometry.set("score_slots", b.getI64IntegerAttr(cached));
  auto inner = space.getAttrOfType<IntegerAttr>("cpu.innermost_owner");
  if (score && inner && machine->orderedDotTokens &&
      space.hasAttr("cpu.value_width")) {
    auto dot = orderedDotPanel(space, *facts, inner.getInt(), rows);
    auto value = dot ? orderedValuePanel(space, *facts, *dot) : std::nullopt;
    if (dot && value) {
      SmallVector<int64_t> slots;
      for (auto [read, transpose] :
           {std::pair{dot->key, true}, std::pair{value->read, false}}) {
        auto arg = cast<BlockArgument>(read->getOperand(0));
        if (invocationPanelElements(space, *facts, *machine, arg, tokens, rows,
                                    transpose))
          slots.push_back(arg.getArgNumber());
      }
      if (!slots.empty())
        geometry.set("packed_slots", b.getDenseI64ArrayAttr(slots));
    }
  }
  return geometry.getDictionary(b.getContext());
}
} // namespace intentdsl::cpu
