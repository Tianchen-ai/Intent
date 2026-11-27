#pragma once
#include "../legalize/Slab.h"
#include "intentdsl/levels/algorithm/kir/ir/Domain.h"
#include "intentdsl/levels/storage/dsa/ir/Supply.h"
#include "mlir/IR/Builders.h"

namespace CI_UNIT_NAMESPACE {
struct IndexedProduct {
  mlir::Value weights, indices, table;
};
// Both leaves of the product are scalar loads. One index follows the segment
// member, the other is itself loaded from a consecutive integer stream.
inline std::optional<IndexedProduct> indexedProduct(mlir::Operation *fold) {
  using namespace mlir;
  auto &bb = fold->getRegion(0).front();
  auto *mul = bb.back().getOperand(0).getDefiningOp();
  if (!mul || mul->getName().getStringRef() != "block.mul" ||
      bb.getOperations().size() != 5)
    return {};
  auto load = [](Value v) {
    auto *op = v.getDefiningOp();
    return op && op->getName().getStringRef() == "block.member_load" &&
                   op->getNumOperands() == 2
               ? op
               : nullptr;
  };
  for (unsigned side : {0u, 1u}) {
    auto *w = load(mul->getOperand(side)), *x = load(mul->getOperand(1 - side));
    if (!w || !x || w->getOperand(1) != bb.getArgument(0))
      continue;
    auto *i = load(x->getOperand(1));
    if (i && i->getOperand(1) == bb.getArgument(0) &&
        i->getResult(0).getType().isInteger(32))
      return IndexedProduct{w->getOperand(0), i->getOperand(0),
                            x->getOperand(0)};
  }
  return {};
}
inline mlir::DictionaryAttr segmentGather(mlir::func::FuncOp fn,
                                          const compilerinfra::TargetProfile &p,
                                          mlir::DictionaryAttr layout) {
  using namespace mlir;
  auto caps = p.services.get<intentdsl::dsa::LocalSupply>();
  auto facts = p.services.get<intentdsl::CallFacts>();
  if (!caps || !facts || !layout.contains("independent_columns"))
    return {};
  Operation *fold = nullptr;
  unsigned folds = 0;
  fn.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "block.segment_fold") {
      fold = op;
      ++folds;
    }
  });
  if (folds != 1 || fold->getNumOperands() != 4 ||
      cast<MemRefType>(fn.getArgument(intentdsl::domainSlot(fn)).getType())
              .getRank() != 1)
    return {};
  auto access = indexedProduct(fold);
  if (!access)
    return {};
  for (Value v :
       {access->weights, access->indices, access->table, fold->getOperand(0)})
    if (boundViewField(v, 0, *facts, true) != 1)
      return {};
  llvm::SmallDenseSet<int64_t> groups;
  for (BlockArgument a : fn.getArguments()) {
    if (!isa<MemRefType>(a.getType()))
      continue;
    if (a.getArgNumber() >= facts->arguments.size() ||
        fn.getArgAttr(a.getArgNumber(), "kir.indexing"))
      return {};
    auto rec = facts->arguments[a.getArgNumber()].getAsObject();
    auto g = rec ? rec->getInteger("alias_group") : std::nullopt;
    if (!g || *g < 0 || !groups.insert(*g).second)
      return {};
  }
  auto size = boundViewField(access->table, 0, *facts, false);
  if (!size || *size < 1 || *size > UINT32_MAX / 4)
    return {};
  int64_t width = 0;
  for (auto cfg : fn->getAttrOfType<ArrayAttr>("block.configs"))
    width = std::max(
        width, cast<DictionaryAttr>(cfg).getAs<IntegerAttr>("width").getInt());
  int64_t room = std::min(caps->capacity, caps->budget);
  for (auto row : layout.getAs<ArrayAttr>("rows"))
    room = std::min(
        room,
        std::min(caps->capacity, caps->budget) -
            cast<DictionaryAttr>(row).getAs<IntegerAttr>("bytes").getInt());
  room -= llvm::alignTo(4 * llvm::alignTo(*size, 64), caps->alignment) +
          llvm::alignTo(4 * (width + 1), caps->alignment);
  int64_t span = std::min<int64_t>({width, 65535, room / 12});
  span = span / (caps->alignment / 4) * (caps->alignment / 4);
  if (span < caps->alignment / 4)
    return {};
  NamedAttrList attrs(layout);
  attrs.set(
      "segment_gather",
      Builder(fn.getContext()).getDenseI64ArrayAttr({*size, width, span}));
  return attrs.getDictionary(fn.getContext());
}
} // namespace CI_UNIT_NAMESPACE
