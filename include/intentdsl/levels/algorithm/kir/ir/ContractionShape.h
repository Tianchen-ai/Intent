#pragma once
#include "intentdsl/levels/algorithm/kir/ir/PanelMap.h"
#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
#include "intentdsl/levels/algorithm/kir/ir/StorageLayout.h"
namespace intentdsl::kir {
// Logical right scales are [column, group]. Preserve their physical view
// and express the axis exchange exactly once for all panel consumers.
struct RightScale {
  mlir::Value view;
  unsigned column, group;
  explicit RightScale(mlir::Operation &dot):view(dot.getOperand(3)),column(0),group(1){
    if(auto *transpose=view.getDefiningOp()){
      view=transpose->getOperand(0);column=1;group=0;
    }
  }
  std::vector<std::string> index(std::string n,std::string g) const {
    return column==0?std::vector<std::string>{n,g}:std::vector<std::string>{g,n};
  }
};
struct ContractionShape {
  int64_t m, n, k, batches;
  unsigned lhs, rhs, dst;
  bool grouped = false, routed = false;
};
// Bind a verified logical contraction to the current invocation's extents.
inline llvm::Expected<ContractionShape>
contractionShape(mlir::Operation &op, mlir::Value out, const CallFacts &f) {
  using namespace mlir;
  auto slot = [](Value v) { return cast<BlockArgument>(v).getArgNumber(); };
  auto axis = [&](Value v, size_t i) {
    auto *s = f.shapeOf(v);
    return s && i < s->size() ? (*s)[i].getAsInteger().value_or(-1) : -1;
  };
  {
    auto name = op.getName().stripDialect();
    if (!isContraction(name))
      return llvm::createStringError("expected a typed contraction");
    auto a = op.getOperand(0);
    if (name == "window_contract") {
      auto b = op.getOperand(1), w = op.getOperand(2);
      auto chunks = axis(w, 2), heads = axis(w, 1), k = axis(w, 3),
           groups = axis(b, 2);
      int64_t time = 0, batches = 0, mapped = 0;
      if (chunks < 0 || heads < 0 || k < 0 || groups < 0 ||
          __builtin_mul_overflow(chunks, k, &time) || time != axis(a, 1) ||
          __builtin_mul_overflow(
              groups, op.getAttrOfType<IntegerAttr>("head_group").getInt(),
              &mapped) ||
          mapped != heads || __builtin_mul_overflow(chunks, heads, &batches) ||
          __builtin_mul_overflow(batches, axis(w, 0), &batches))
        return llvm::createStringError(
            "window geometry requires L=C*S and H=G*head_group");
      return ContractionShape{axis(a, 3), axis(b, 3), k,        batches,
                              slot(a),    slot(b),    slot(out)};
    }
    if (name == "q4k_dot") {
      auto x = op.getOperand(1).getDefiningOp()->getOperand(0);
      int64_t k;
      if (__builtin_mul_overflow(axis(x, 0), int64_t(Q8KRecord::elements), &k))
        return llvm::createStringError("Q8_K contraction extent exceeds i64");
      return ContractionShape{axis(a, 0), 1, k, 1, slot(a), slot(x), slot(out)};
    }
    bool scaled = name == "scaled_contract";
    auto b = op.getOperand(scaled ? 2 : 1);
    auto ak = scaled ? 1 : op.getAttrOfType<IntegerAttr>("lhs_k").getInt();
    auto bk = scaled ? 0 : op.getAttrOfType<IntegerAttr>("rhs_k").getInt();
    auto ab = scaled ? -1 : op.getAttrOfType<IntegerAttr>("lhs_batch").getInt();
    auto batch =
        scaled ? -1 : op.getAttrOfType<IntegerAttr>("rhs_batch").getInt();
    int64_t k = axis(a, ak), m = 0, n = 0;
    if (scaled) {
      if (__builtin_mul_overflow(k, int64_t(MX8Block::group), &k))
        return llvm::createStringError("MX contraction extent exceeds i64");
      m = axis(a, 0);
      n = axis(b, 2);
    } else {
      for (int i = 0; i < cast<MemRefType>(a.getType()).getRank(); i++)
        if (i != ak && i != ab)
          m = axis(a, i);
      for (int i = 0; i < cast<MemRefType>(b.getType()).getRank(); i++)
        if (i != bk && i != batch)
          n = axis(b, i);
    }
    int64_t groups = batch != -1 ? axis(b, batch) : ab != -1 ? axis(a, ab) : 1;
    bool grouped = !scaled && op.getNumOperands() > 2;
    if (grouped && op.getNumOperands() == 4)
      m = axis(op.getOperand(3), 0);
    if (m < 0 || n < 0 || k < 0 || groups < 0 ||
        (grouped && axis(op.getOperand(2), 0) != groups + 1))
      return llvm::createStringError("missing matrix geometry or offsets "
                                     "length differs from group count + 1");
    return ContractionShape{
        m,         n,       k,
        groups,    slot(a), slot(b),
        slot(out), grouped, grouped && op.getNumOperands() == 4};
  }
}
} // namespace intentdsl::kir
