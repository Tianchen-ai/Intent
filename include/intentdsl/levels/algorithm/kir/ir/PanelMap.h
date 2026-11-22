#pragma once
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringExtras.h"

namespace intentdsl::kir {
inline bool isContraction(llvm::StringRef kind) {
  return kind == "contract" || kind == "window_contract" || kind == "q4k_dot" ||
         kind == "scaled_contract";
}
// The caller spells its native min, exp and the two explicit conversions.
// Operands can denote either a scalar element or a whole native panel.
struct PanelMath {
  llvm::StringRef extend, narrow, minimum, exp;
  llvm::StringRef minOptions = "";
};
inline mlir::Value rowScale(mlir::Operation &contract) {
  auto &map = contract.getRegion(0).front();
  auto *round = map.back().getOperand(0).getDefiningOp();
  if (!round || round->getName().getStringRef() != "arith.truncf")
    return {};
  auto *mul = round->getOperand(0).getDefiningOp();
  if (!mul || mul->getName().getStringRef() != "arith.mulf")
    return {};
  llvm::DenseMap<mlir::Value, bool> depends;
  depends[map.getArgument(0)] = true;
  for (auto &op : map.without_terminator())
    depends[op.getResult(0)] = llvm::any_of(
        op.getOperands(), [&](mlir::Value v) { return depends.lookup(v); });
  for (unsigned i = 0; i < 2; ++i) {
    auto *expand = mul->getOperand(i).getDefiningOp();
    if (expand && expand->getName().getStringRef() == "arith.extf" &&
        expand->getOperand(0) == map.getArgument(0) &&
        !depends.lookup(mul->getOperand(1 - i)))
      return mul->getOperand(1 - i);
  }
  return {};
}
inline std::string panelExpression(mlir::Operation &contract,
                                   llvm::ArrayRef<std::string> terms,
                                   PanelMath math, mlir::Value stop = {}) {
  auto &region = contract.getRegion(0).front();
  llvm::DenseMap<mlir::Value, std::string> values;
  for (auto [v, s] : llvm::zip_equal(region.getArguments(), terms))
    values[v] = s;
  if (stop && values.count(stop))
    return values.at(stop);
  for (auto &op : region) {
    auto n = op.getName().getStringRef();
    auto get = [&](unsigned i) { return values.at(op.getOperand(i)); };
    if (op.getName().stripDialect() == "panel_value")
      return get(0);
    std::string expr;
    if (n == "arith.constant") {
      llvm::SmallString<24> num;
      op.getAttrOfType<mlir::FloatAttr>("value").getValue().toString(num);
      expr = num.str().str();
      if (expr.find_first_of(".Ee") == std::string::npos)
        expr += ".0";
    } else if (n == "arith.extf" || n == "arith.truncf" || n == "math.exp") {
      auto form = n == "math.exp"     ? math.exp
                  : n == "arith.extf" ? math.extend
                                      : math.narrow;
      auto [left, right] = form.split('$');
      expr = form.contains('$') ? left.str() + get(0) + right.str()
                                : form.str() + "(" + get(0) + ")";
    } else if (n == "arith.minimumf")
      expr = math.minimum.str() + "(" + get(0) + "," + get(1) +
             math.minOptions.str() + ")";
    else
      expr = "(" + get(0) +
             (n == "arith.mulf"   ? "*"
              : n == "arith.addf" ? "+"
                                  : "-") +
             get(1) + ")";
    values[op.getResult(0)] = std::move(expr);
    if (stop == op.getResult(0))
      return values.at(stop);
  }
  llvm_unreachable("verified yield");
}
} // namespace intentdsl::kir
