#pragma once
#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
#include "intentdsl/levels/execution/block/decisions/RowCapacity.h"

namespace intentdsl::block {
// A partition owns its complete row of counters. Duplicate input keys can
// collide locally, but distinct partitions publish disjoint output intervals.
inline std::optional<int64_t>
privateBinStorage(mlir::Operation *region,
                  const compilerinfra::TargetProfile &profile) {
  using namespace mlir;
  auto caps = profile.services.get<RowCapacity>();
  auto facts = profile.services.get<CallFacts>();
  if (!caps || !facts || caps->privateBinGranule <= 0 ||
      caps->privateBinBytes <= 0 || !caps->privateBinCapacity ||
      region->getNumRegions() != 1 ||
      !llvm::hasSingleElement(region->getRegion(0)))
    return {};
  auto &body = region->getRegion(0).front();
  if (body.getNumArguments() != 1 || body.getOperations().size() != 3)
    return {};
  auto &count = body.front(), &publish = *count.getNextNode();
  if (count.getName().stripDialect() != "count_keys" ||
      publish.getName().stripDialect() != "write" ||
      count.getOperand(1) != body.getArgument(0) ||
      publish.getOperand(1) != body.getArgument(0) || !count.hasOneUse() ||
      publish.getOperand(2) != count.getResult(0) || publish.hasAttr("axes") ||
      region->getOperand(0) != publish.getOperand(0))
    return {};
  auto fn = region->getParentOfType<func::FuncOp>();
  if (!fn || fn->hasAttr("kir.iteration"))
    return {};
  auto record = [&](Value view) -> const llvm::json::Object * {
    auto arg = dyn_cast<BlockArgument>(view);
    if (!arg || arg.getOwner() != &fn.front() ||
        arg.getArgNumber() >= facts->arguments.size())
      return nullptr;
    return facts->arguments[arg.getArgNumber()].getAsObject();
  };
  auto *in = record(count.getOperand(0)), *out = record(publish.getOperand(0));
  if (!in || !out)
    return {};
  auto src = in->getInteger("alias_group"),
       dst = out->getInteger("alias_group");
  if (!src || !dst || *src < 0 || *dst < 0 || src == dst)
    return {};
  auto *a = in->getArray("shape"), *b = out->getArray("shape");
  auto *x = in->getArray("strides"), *y = out->getArray("strides");
  if (!a || a->size() != 1 || !b || b->size() != 2 || !x || x->size() != 1 ||
      !y || y->size() != 2 || (*x)[0].getAsInteger() != 1 ||
      (*y)[1].getAsInteger() != 1)
    return {};
  int64_t n = (*a)[0].getAsInteger().value_or(-1);
  int64_t rows = (*b)[0].getAsInteger().value_or(-1);
  int64_t bins = (*b)[1].getAsInteger().value_or(-1);
  auto span = count.getAttrOfType<IntegerAttr>("span");
  auto elem = cast<MemRefType>(count.getOperand(0).getType()).getElementType();
  if (n <= 0 || bins <= 0 || (*y)[0].getAsInteger().value_or(-1) < bins ||
      !span || span.getInt() <= 0 || span.getInt() > INT32_MAX ||
      rows != int64_t(llvm::divideCeil(n, span.getInt())) ||
      (!elem.isF32() && !elem.isInteger(64)) ||
      bins % caps->privateBinGranule ||
      bins > *caps->privateBinCapacity / caps->privateBinBytes)
    return {};
  return bins * caps->privateBinBytes;
}
} // namespace intentdsl::block
