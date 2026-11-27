#pragma once
#include "../legalize/Broadcast.h"
#include "../legalize/Slab.h"
#include "intentdsl/levels/storage/dsa/ir/Matrix.h"
#include "mlir/IR/Builders.h"

namespace CI_UNIT_NAMESPACE {
// The mtp_372 histogram compares one panel against 64 contiguous WRAM rows.
// Each int16 key occupies one 64-byte coefficient row (cncc Vector Functions).
inline mlir::DictionaryAttr keyPanel(mlir::func::FuncOp fn,
                                     const compilerinfra::TargetProfile &p,
                                     mlir::DictionaryAttr layout) {
  using namespace mlir;
  if (fn->getAttrOfType<StringAttr>("block.topology") != "resident" ||
      layout.getAs<ArrayAttr>("rows").empty())
    return {};
  Operation *count = nullptr;
  unsigned matches = 0;
  fn.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "block.key_counts") {
      count = op;
      ++matches;
    }
  });
  if (matches != 1)
    return {};
  auto facts = p.services.get<intentdsl::CallFacts>();
  auto matrix = p.services.get<intentdsl::dsa::MatrixStorage>();
  auto local = p.services.get<intentdsl::dsa::LocalSupply>();
  if (!facts || !matrix || matrix->wramBytes < 4096 || !local)
    return {};
  int64_t bins = facts->geometry(fn).second;
  auto ptr = count->getOperand(0);
  auto elem = cast<MemRefType>(ptr.getType()).getElementType();
  auto span = count->getAttrOfType<IntegerAttr>("span").getInt();
  auto extent = count->getAttrOfType<IntegerAttr>("extent").getInt();
  int64_t samples = std::min(span, extent);
  if (bins <= 0 || bins % 64 || bins > 32767 || samples <= 0 ||
      (!elem.isF32() && !elem.isInteger(64)) ||
      boundViewField(ptr, 0, *facts, true) != 1)
    return {};
  auto capacity = std::min(local->capacity, local->budget);
  int64_t remaining = capacity;
  for (auto cfg : fn->getAttrOfType<ArrayAttr>("block.configs"))
    if (cast<DictionaryAttr>(cfg).getAs<IntegerAttr>("width").getInt() < bins)
      return {};
  for (auto row : layout.getAs<ArrayAttr>("rows"))
    remaining = std::min(
        remaining,
        capacity -
            cast<DictionaryAttr>(row).getAs<IntegerAttr>("bytes").getInt());
  for (auto [v, n] : broadcastSpans(fn, *facts, *local, layout))
    remaining -= llvm::alignTo(
        n * cast<MemRefType>(v.getType()).getElementTypeBitWidth() / 8,
        local->alignment);
  auto bytes = elem.getIntOrFloatBitWidth() / 8;
  if (samples > remaining / bytes)
    return {};
  samples = llvm::alignTo(samples, 32);
  if (samples > remaining / bytes ||
      llvm::alignTo(samples * bytes, 128) +
              llvm::alignTo(samples * (elem.isInteger(64) ? 8 : 2), 128) +
              bins * 64 >
          remaining)
    return {};
  NamedAttrList attrs(layout);
  attrs.set("histogram_panel",
            Builder(fn.getContext()).getDenseI64ArrayAttr({samples, bins}));
  return attrs.getDictionary(fn.getContext());
}
} // namespace CI_UNIT_NAMESPACE
