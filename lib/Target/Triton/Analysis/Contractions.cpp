#include "Intent/Target/Triton/Analysis/Contractions.h"
#include "Intent/Target/Triton/Analysis/Configuration.h"
#include "Intent/Target/Triton/IR/Program.h"
#include "Intent/Dialect/Block/Analysis/PhysicalProgram.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;

namespace intent::triton {
namespace {

block::PhysicalExprAttr expression(MLIRContext *context,
                                 block::PhysicalExprKind kind,
                                 ArrayRef<Attribute> operands = {},
                                 int64_t value = 0) {
  return block::PhysicalExprAttr::get(context, kind, value,
      StringAttr::get(context, ""), ArrayAttr::get(context, operands));
}

block::PhysicalExprAttr constant(MLIRContext *context, int64_t value) {
  return expression(context, block::PhysicalExprKind::Constant, {}, value);
}

SmallVector<block::PhysicalExprAttr> expansionFactors(block::ContractOp contract) {
  SmallVector<block::PhysicalExprAttr> factors{
      cast<block::PhysicalExprAttr>(contract.getLhs().getType().getShape().getValue().back())};
  for (Attribute extent : contract.getAccumulator().getType().getShape())
    factors.push_back(cast<block::PhysicalExprAttr>(extent));
  return factors;
}

// Only constexpr control can exclude a form from native compilation. Runtime
// control still requires both branches to have legal tensor shapes.
block::PhysicalExprAttr constexprCondition(Value value) {
  auto *context = value.getContext();
  auto zero = constant(context, 0), one = constant(context, 1);
  auto make = [&](block::PhysicalExprKind kind, ArrayRef<Attribute> operands) {
    return expression(context, kind, operands);
  };
  if (auto scalar = value.getDefiningOp<arith::ConstantOp>()) {
    auto integer = dyn_cast<IntegerAttr>(scalar.getValue());
    return integer && scalar.getType().isInteger(1)
        ? constant(context, !integer.getValue().isZero()) : block::PhysicalExprAttr();
  }
  if (auto binary = value.getDefiningOp<block::BinaryOp>()) {
    auto kind = binary.getOperatorKind();
    if (kind != BinaryOperator::LogicalAnd && kind != BinaryOperator::LogicalOr)
      return {};
    auto lhs = constexprCondition(binary.getLhs());
    auto rhs = constexprCondition(binary.getRhs());
    if (!lhs || !rhs) return {};
    return make(kind == BinaryOperator::LogicalAnd
                    ? block::PhysicalExprKind::Multiply : block::PhysicalExprKind::Maximum,
                {lhs, rhs});
  }
  auto compare = value.getDefiningOp<block::CompareOp>();
  if (!compare || !compare.getLhs().getType().isIndex() ||
      !compare.getRhs().getType().isIndex())
    return {};
  auto lhs = block::queryLaunchExpression(compare.getLhs());
  auto rhs = block::queryLaunchExpression(compare.getRhs());
  if (!lhs || !rhs || !isTritonFragmentExtent(lhs) || !isTritonFragmentExtent(rhs))
    return {};
  auto difference = make(block::PhysicalExprKind::Subtract, {lhs, rhs});
  auto positive = [&](block::PhysicalExprAttr value) {
    return make(block::PhysicalExprKind::Select,
                {make(block::PhysicalExprKind::Maximum, {value, zero}), one, zero});
  };
  auto negate = [&](block::PhysicalExprAttr value) {
    return make(block::PhysicalExprKind::Select, {value, zero, one});
  };
  switch (compare.getPredicate()) {
  case ComparePredicate::Eq:
    return negate(difference);
  case ComparePredicate::Ne:
    return make(block::PhysicalExprKind::Select, {difference, one, zero});
  case ComparePredicate::Gt:
    return positive(difference);
  case ComparePredicate::Le:
    return negate(positive(difference));
  case ComparePredicate::Lt:
    return positive(make(block::PhysicalExprKind::Subtract, {rhs, lhs}));
  case ComparePredicate::Ge:
    return negate(positive(make(block::PhysicalExprKind::Subtract, {rhs, lhs})));
  }
  llvm_unreachable("unknown comparison predicate");
}

} // namespace

block::PhysicalExprAttr expandedContractionElements(block::ContractOp contract) {
  auto factors = expansionFactors(contract);
  auto result = factors.front();
  for (auto factor : llvm::drop_begin(factors))
    result = expression(contract.getContext(), block::PhysicalExprKind::Multiply,
                        {result, factor});
  return result;
}

block::ConfigurationRequirementAttr
contractionExpansionRequirement(block::ContractOp contract) {
  if (contract.getLhs().getType().getShape().empty()) return {};
  auto *context = contract.getContext();
  auto one = constant(context, 1);
  auto factors = expansionFactors(contract);
  SmallVector<block::PhysicalExprAttr> conditions;
  if (!contract->hasAttr(contractFormAttr)) {
    // Before form selection, only short K requires expansion. The optional
    // large-K branch already includes the expanded element bound in its guard.
    auto quotient = expression(context, block::PhysicalExprKind::FloorDiv,
        {factors.front(), constant(context, expansionReductionThreshold)});
    conditions.push_back(expression(context, block::PhysicalExprKind::Select,
                                    {quotient, constant(context, 0), one}));
  }
  for (Operation *current = contract; Operation *parent = current->getParentOp();
       current = parent) {
    auto conditional = dyn_cast<scf::IfOp>(parent);
    if (!conditional) continue;
    auto condition = constexprCondition(conditional.getCondition());
    if (!condition) continue;
    if (current->getParentRegion() == &conditional.getElseRegion())
      condition = expression(context, block::PhysicalExprKind::Select,
                             {condition, constant(context, 0), one});
    conditions.push_back(condition);
  }
  auto usage = one;
  for (auto factor : factors) {
    // Guard each factor before multiplication: the checked evaluator must not
    // overflow an inactive expansion's product while evaluating select operands.
    for (auto condition : conditions)
      factor = expression(context, block::PhysicalExprKind::Select,
                          {condition, factor, one});
    usage = expression(context, block::PhysicalExprKind::Multiply, {usage, factor});
  }
  return block::ConfigurationRequirementAttr::get(context,
      block::ConfigurationRequirementKind::Legality,
      block::ConfigurationRequirementMetric::FragmentElements,
      block::ConfigurationRequirementPredicate::LessEqual, usage,
      constant(context, maxTritonTensorElements), block::ParameterRefAttr(),
      StringAttr::get(context, "Triton expanded contraction exceeds the maximum element count"));
}

} // namespace intent::triton
