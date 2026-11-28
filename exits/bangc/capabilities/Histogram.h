#pragma once
#include "../legalize/Broadcast.h"
#include "../legalize/Slab.h"
#include "SegmentGather.h"
#include "intentdsl/levels/storage/dsa/ir/Matrix.h"
#include "mlir/IR/Builders.h"

namespace CI_UNIT_NAMESPACE {
// A relaxed fetch-add over bounded independent counters can reserve a range
// for every key seen in one resident tile and replay tickets in member order.
inline mlir::DictionaryAttr
ticketPanel(mlir::func::FuncOp fn, const compilerinfra::TargetProfile &profile,
            mlir::DictionaryAttr layout) {
  using namespace mlir;
  auto facts = profile.services.get<intentdsl::CallFacts>();
  auto caps = profile.services.get<intentdsl::dsa::LocalSupply>();
  if (!caps || !facts || layout.getAs<IntegerAttr>("batch").getInt() != 1)
    return {};
  Operation *update = nullptr;
  unsigned updates = 0, copies = 0;
  fn.walk([&](Operation *op) {
    auto n = op->getName().getStringRef();
    if (n == "block.atomic_add_at") {
      ++updates;
      update = op;
    }
    if (n == "block.load" || n == "dsa.copy_in")
      ++copies;
  });
  if (updates != 1 || copies != 1 || !update->hasAttr("in_bounds") ||
      update->getAttrOfType<StringAttr>("ordering") != "relaxed")
    return {};
  auto key = update->getOperand(1).getDefiningOp();
  auto delta = update->getOperand(2).getDefiningOp();
  auto mask = update->getOperand(3);
  if (!key || !delta || !delta->getAttrOfType<IntegerAttr>("value") ||
      delta->getAttrOfType<IntegerAttr>("value").getInt() != 1)
    return {};
  if (key->getName().getStringRef() == "block.load") {
    if (key->getNumOperands() != 4 || key->getOperand(3) != mask ||
        !mask.getDefiningOp() ||
        mask.getDefiningOp()->getName().getStringRef() != "block.active")
      return {};
  } else if (key->getName().getStringRef() != "dsa.await" ||
             !isa<BlockArgument>(key->getOperand(0)) || !mask.getDefiningOp() ||
             mask.getDefiningOp()->getName().getStringRef() != "dsa.window")
    return {};
  Value counter = update->getOperand(0);
  if (!counter.hasOneUse() || !isa<BlockArgument>(counter) ||
      !cast<MemRefType>(counter.getType()).getElementType().isInteger(32) ||
      boundViewField(counter, 0, *facts, true) != 1)
    return {};
  auto bins = boundViewField(counter, 0, *facts, false);
  if (!bins || *bins < 1)
    return {};
  llvm::SmallDenseSet<int64_t> groups;
  for (BlockArgument a : fn.getArguments()) {
    if (!isa<MemRefType>(a.getType()))
      continue;
    if (a.getArgNumber() >= facts->arguments.size())
      return {};
    auto rec = facts->arguments[a.getArgNumber()].getAsObject();
    auto g = rec ? rec->getInteger("alias_group") : std::nullopt;
    if (!g || *g < 0 || !groups.insert(*g).second)
      return {};
  }
  int64_t maxMembers = 0;
  for (Attribute config : fn->getAttrOfType<ArrayAttr>("block.configs")) {
    auto c = cast<DictionaryAttr>(config);
    int64_t members = c.getAs<IntegerAttr>("rows").getInt() *
                      c.getAs<IntegerAttr>("width").getInt();
    maxMembers = std::max(maxMembers, members);
  }
  auto cap = std::min(caps->capacity, caps->budget);
  if (*bins > cap / 8 || maxMembers > cap / 4)
    return {};
  int64_t bytes = 2 * llvm::alignTo(4 * (*bins), caps->alignment) +
                  llvm::alignTo(4 * maxMembers, caps->alignment);
  for (auto row : layout.getAs<ArrayAttr>("rows"))
    if (cast<DictionaryAttr>(row).getAs<IntegerAttr>("bytes").getInt() + bytes >
        cap)
      return {};
  NamedAttrList attrs(layout);
  attrs.set("ticket_panel",
            Builder(fn.getContext()).getDenseI64ArrayAttr({*bins, maxMembers}));
  return attrs.getDictionary(fn.getContext());
}
// The mtp_372 histogram compares one panel against 64 contiguous WRAM rows.
// Each int16 key occupies one 64-byte coefficient row (cncc Vector Functions).
inline mlir::DictionaryAttr keyPanel(mlir::func::FuncOp fn,
                                     const compilerinfra::TargetProfile &p,
                                     mlir::DictionaryAttr layout) {
  using namespace mlir;
  if (auto gather = segmentGather(fn, p, layout))
    return gather;
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
  if (!matches)
    return ticketPanel(fn, p, layout);
  if (matches != 1 || !fn->hasAttr("block.private_bin_storage"))
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
