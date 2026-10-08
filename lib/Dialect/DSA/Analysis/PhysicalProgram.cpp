#include "Intent/Dialect/DSA/Analysis/PhysicalProgram.h"
#include "Intent/Analysis/IntegerRanges.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Matchers.h"
#include <functional>

using namespace mlir;
namespace intent::dsa {

SignedInterval integerInterval(Value value, func::FuncOp function) {
  IntegerRangePolicy policy;
  policy.infer = [&](Value current, IntegerRangeAnalysis &)
      -> std::optional<ConstantIntRanges> {
    if (!function || function->hasAttr("intent_dsa.group_width"))
      return std::nullopt;
    auto config = function->getAttrOfType<ConfigurationAttr>(
        "intent_dsa.configuration");
    if (!config || config.getTasks() <= 0) return std::nullopt;
    if (current.getDefiningOp<TaskIdOp>())
      return ConstantIntRanges::fromSigned(APInt(64, 0),
                                          APInt(64, config.getTasks() - 1));
    if (current.getDefiningOp<TaskCountOp>())
      return ConstantIntRanges::constant(APInt(64, config.getTasks()));
    return std::nullopt;
  };
  IntegerRangeAnalysis analysis(std::move(policy));
  auto bounds = analysis.range(value);
  if (!bounds || bounds->smin().getBitWidth() > 64) return std::nullopt;
  return std::make_pair(bounds->smin().getSExtValue(),
                        bounds->smax().getSExtValue());
}

bool isSumOfIntegerProducts(Value value, ArrayRef<std::pair<Value, Value>> products) {
  auto addressType = [](Type type) { return type.isIndex() || type.isInteger(64); };
  if (!addressType(value.getType()) || llvm::any_of(products, [&](const auto &product) {
        return !addressType(product.first.getType()) || !addressType(product.second.getType());
      })) return false;
  MLIRContext *context = value.getContext();
  DenseMap<Value, AffineExpr> expressions;
  unsigned symbols = 0;
  std::function<AffineExpr(Value)> expression = [&](Value current) -> AffineExpr {
    if (auto known = expressions.find(current); known != expressions.end()) return known->second;
    auto infer = [&]() -> AffineExpr {
      APInt constant;
      if (matchPattern(current, m_ConstantInt(&constant)))
        return getAffineConstantExpr(constant.getSExtValue(), context);
      if (auto add = current.getDefiningOp<arith::AddIOp>()) return expression(add.getLhs()) + expression(add.getRhs());
      if (auto sub = current.getDefiningOp<arith::SubIOp>()) return expression(sub.getLhs()) - expression(sub.getRhs());
      if (auto mul = current.getDefiningOp<arith::MulIOp>()) return expression(mul.getLhs()) * expression(mul.getRhs());
      if (auto cast = current.getDefiningOp<arith::IndexCastOp>(); cast && addressType(cast.getIn().getType()))
        return expression(cast.getIn());
      return getAffineSymbolExpr(symbols++, context);
    };
    auto result = infer(); expressions[current] = result; return result;
  };
  AffineExpr actual = expression(value), expected = getAffineConstantExpr(0, context);
  for (auto [lhs, rhs] : products) expected = expected + expression(lhs) * expression(rhs);
  auto difference = dyn_cast<AffineConstantExpr>(simplifyAffineExpr(actual - expected, 0, symbols));
  return difference && difference.getValue() == 0;
}

} // namespace intent::dsa
