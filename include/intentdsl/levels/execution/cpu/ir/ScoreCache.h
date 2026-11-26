#pragma once
#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
#include "intentdsl/levels/execution/cpu/ir/RepeatedScore.h"
#include "llvm/ADT/SmallPtrSet.h"
namespace intentdsl::cpu {
struct ScoreCachePlan {
  int64_t keys = 0;
  int64_t depth = 0;
  double readBytes = 0.0;
  mlir::Operation *record = nullptr;
  mlir::Operation *replay = nullptr;
};
namespace detail {
inline std::optional<int64_t> staticIndexValue(mlir::Value v,
                                               const CallFacts &facts) {
  using namespace mlir;
  if (auto cst = v.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value();
  if (auto dim = v.getDefiningOp<memref::DimOp>()) {
    auto idx = dim.getConstantIndex();
    auto *arr = idx ? facts.shapeOf(dim.getSource()) : nullptr;
    if (arr && *idx < arr->size())
      return (*arr)[*idx].getAsInteger();
    return {};
  }
  if (auto add = v.getDefiningOp<arith::AddIOp>()) {
    auto l = staticIndexValue(add.getLhs(), facts),
         r = staticIndexValue(add.getRhs(), facts);
    if (l && r)
      return *l + *r;
  }
  if (auto sub = v.getDefiningOp<arith::SubIOp>()) {
    auto l = staticIndexValue(sub.getLhs(), facts),
         r = staticIndexValue(sub.getRhs(), facts);
    if (l && r)
      return *l - *r;
  }
  if (auto mul = v.getDefiningOp<arith::MulIOp>()) {
    auto l = staticIndexValue(mul.getLhs(), facts),
         r = staticIndexValue(mul.getRhs(), facts);
    if (l && r)
      return *l * *r;
  }
  return {};
}
inline int64_t tensorSliceExtent(mlir::Value v, const CallFacts &facts) {
  using namespace mlir;
  llvm::SmallPtrSet<Operation *, 8> seen;
  SmallVector<Value, 8> work{v};
  while (!work.empty()) {
    Value cur = work.pop_back_val();
    auto *op = cur.getDefiningOp();
    if (!op || !seen.insert(op).second)
      continue;
    auto kind = op->getName().stripDialect();
    if (kind == "slice_read" || kind == "lane_splat") {
      Value view = op->getOperand(kind == "slice_read" ? 0 : 1);
      auto ax = op->getAttrOfType<IntegerAttr>("axis");
      auto *arr = ax ? facts.shapeOf(view) : nullptr;
      if (arr && size_t(ax.getInt()) < arr->size())
        if (auto ext = (*arr)[ax.getInt()].getAsInteger(); ext && *ext > 0)
          return *ext;
    }
    for (Value in : op->getOperands())
      if (isa<RankedTensorType>(in.getType()))
        work.push_back(in);
  }
  return 0;
}
inline bool dependsOnValue(mlir::Value v, mlir::Value target) {
  if (v == target)
    return true;
  auto *op = v.getDefiningOp();
  if (!op || op->getNumRegions() != 0)
    return false;
  return llvm::any_of(op->getOperands(),
                      [&](mlir::Value x) { return dependsOnValue(x, target); });
}
} // namespace detail
inline std::optional<ScoreCachePlan>
reusableScores(mlir::Operation *o, const compilerinfra::TargetProfile &p,
               int64_t slotLimit = 1024) {
  using namespace mlir;
  auto facts = p.services.get<CallFacts>();
  if (!facts)
    return {};
  auto rep = repeatedScore(*o);
  if (!rep || !matchPattern(rep->first.getLowerBound(), m_Zero()))
    return {};
  auto keys = detail::staticIndexValue(rep->first.getUpperBound(), *facts);
  // Each owner retains one fp32 score per key on the task's stack.
  if (!keys || *keys <= 0 || *keys > slotLimit)
    return {};

  llvm::SmallPtrSet<Operation *, 16> sliceOps;
  SmallVector<Operation *, 16> stack{rep->record};
  Block *foldBody = rep->first.getBody();
  Value iv = rep->first.getInductionVar();
  int64_t depth = 0;
  double readBytes = 0.0;
  while (!stack.empty()) {
    Operation *cur = stack.pop_back_val();
    if (!sliceOps.insert(cur).second)
      continue;
    auto kind = cur->getName().stripDialect();
    if (kind == "lane_sum" && cur->getNumOperands() == 1 &&
        isa<RankedTensorType>(cur->getOperand(0).getType()))
      depth += detail::tensorSliceExtent(cur->getOperand(0), *facts);
    if (kind == "slice_read" &&
        llvm::any_of(cur->getOperands().drop_front(),
                     [&](Value c) { return detail::dependsOnValue(c, iv); })) {
      int64_t ext = detail::tensorSliceExtent(cur->getResult(0), *facts);
      unsigned bits =
          cast<ShapedType>(cur->getResult(0).getType()).getElementTypeBitWidth();
      readBytes += double(ext) * double(bits) / 8.0;
    }
    for (Value in : cur->getOperands())
      if (auto *def = in.getDefiningOp(); def && def->getBlock() == foldBody)
        stack.push_back(def);
  }
  if (depth <= 0)
    return {};
  return ScoreCachePlan{*keys, depth, readBytes, rep->record, rep->replay};
}
} // namespace intentdsl::cpu
