#pragma once
#include "intentdsl/levels/algorithm/kir/ir/Performance.h"
namespace intentdsl::kir {
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
    if (name != "contract" && name != "scaled_contract" && name != "q4k_dot")
      return llvm::createStringError("expected a typed contraction");
    auto a = op.getOperand(0);
    if (name == "q4k_dot") {
      auto x = op.getOperand(1).getDefiningOp()->getOperand(0);
      return ContractionShape{axis(a, 0), 1,       axis(x, 0) * 256, 1,
                              slot(a),    slot(x), slot(out)};
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
      k *= 32;
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
