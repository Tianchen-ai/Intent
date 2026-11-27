#pragma once

#include "intentdsl/levels/algorithm/kir/ir/ProjectedLayout.h"
#include "intentdsl/levels/storage/dsa/ir/Supply.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/MathExtras.h"

namespace CI_UNIT_NAMESPACE {
// A tile reads a repeated scalar once per innermost run. Promote the
// backing span only when it costs less than those GDRAM transactions and
// fits alongside every candidate's live NRAM allocation.
inline llvm::SmallVector<std::pair<mlir::BlockArgument, int64_t>>
broadcastSpans(mlir::func::FuncOp fn, const intentdsl::CallFacts &facts,
               const intentdsl::dsa::LocalSupply &machine,
               mlir::DictionaryAttr layout = {}) {
  using namespace mlir;
  llvm::SmallVector<std::pair<BlockArgument, int64_t>> spans;
  auto signature = fn->getAttrOfType<StringAttr>("kir.public_abi");
  if (!signature)
    return spans;
  auto parsed = llvm::json::parse(signature.getValue());
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return spans;
  }
  const auto *abi = parsed->getAsArray();
  if (!abi || abi->size() != facts.arguments.size())
    return spans;
  llvm::DenseSet<int64_t> writable;
  for (auto [i, slot] : llvm::enumerate(*abi)) {
    const auto *desc = slot.getAsObject();
    if (!desc)
      return spans;
    auto direction = desc->getString("direction");
    if (direction == "In" || direction == "Scalar")
      continue;
    const auto *record = facts.arguments[i].getAsObject();
    auto group = record ? record->getInteger("alias_group") : std::nullopt;
    if (!group || *group < 0)
      return spans;
    writable.insert(*group);
  }
  if (!layout)
    layout = fn->getAttrOfType<DictionaryAttr>("dsa.layout");
  int64_t liveBytes = 0;
  for (auto cfg : layout.getAs<ArrayAttr>("rows"))
    liveBytes = std::max(liveBytes,
        cast<DictionaryAttr>(cfg).getAs<IntegerAttr>("bytes").getInt());
  int64_t remaining = std::min(machine.budget, machine.capacity) - liveBytes;
  auto workers = layout.getAs<IntegerAttr>("participants").getInt();
  auto owner = fn->getAttrOfType<IntegerAttr>("kir.owner_axis");
  for (auto arg : fn.getArguments()) {
    auto type = dyn_cast<MemRefType>(arg.getType());
    if (!type || type.getRank() < 2)
      continue;
    auto source =
        fn.getArgAttrOfType<IntegerAttr>(arg.getArgNumber(), "kir.source");
    if (!source || source.getInt() < 0 || size_t(source.getInt()) >= abi->size())
      continue;
    const auto *record = facts.arguments[source.getInt()].getAsObject();
    auto group = record ? record->getInteger("alias_group") : std::nullopt;
    if (!group || *group < 0 || writable.contains(*group))
      continue;
    auto shape = intentdsl::projectedLayout(fn, arg.getArgNumber(), facts, true);
    auto strides = intentdsl::projectedLayout(fn, arg.getArgNumber(), facts, false);
    int64_t runAxis = type.getRank() - 1;
    if (owner && owner.getInt() == runAxis)
      --runAxis;
    if (!shape || !strides || strides[runAxis] != 0 || shape[runAxis] < 2)
      continue;
    int64_t reads = llvm::count_if(arg.getUsers(), [&](Operation *op) {
      return op->getName().getStringRef() == "dsa.copy_in" &&
             op->getOperand(0) == arg && !op->hasAttr("dsa.shared");
    });
    if (!reads)
      continue;
    double runs = double(reads) / workers;
    int64_t elements = 1;
    int64_t bytes = llvm::divideCeil(type.getElementTypeBitWidth(), 8u);
    for (auto [n, stride] : llvm::zip(shape.asArrayRef(), strides.asArrayRef())) {
      int64_t last;
      if (n <= 0 || stride < 0 || __builtin_mul_overflow(n - 1, stride, &last) ||
          __builtin_add_overflow(elements, last, &elements)) {
        elements = 0;
        break;
      }
      runs *= n;
    }
    runs /= shape[runAxis];
    if (!elements || elements > remaining / bytes || runs <= 1)
      continue;
    int64_t allocation = llvm::alignTo(elements * bytes, machine.alignment);
    if (allocation > remaining || allocation > runs * machine.alignment)
      continue;
    remaining -= allocation;
    spans.emplace_back(arg, elements);
  }
  return spans;
}
} // namespace CI_UNIT_NAMESPACE
