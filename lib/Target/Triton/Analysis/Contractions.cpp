#include "Intent/Target/Triton/Analysis/Contractions.h"
#include "Intent/Target/Triton/Analysis/Configuration.h"
#include "Intent/Target/Triton/IR/Program.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;

namespace intent::triton {
namespace {

gpu::PhysicalExprAttr expression(MLIRContext *context,
                                 gpu::PhysicalExprKind kind,
                                 ArrayRef<Attribute> operands = {},
                                 int64_t value = 0) {
  return gpu::PhysicalExprAttr::get(context, kind, value,
      StringAttr::get(context, ""), ArrayAttr::get(context, operands));
}

gpu::PhysicalExprAttr constant(MLIRContext *context, int64_t value) {
  return expression(context, gpu::PhysicalExprKind::Constant, {}, value);
}

SmallVector<gpu::PhysicalExprAttr> expansionFactors(gpu::ContractOp contract) {
  SmallVector<gpu::PhysicalExprAttr> factors{
      cast<gpu::PhysicalExprAttr>(contract.getLhs().getType().getShape().getValue().back())};
  for (Attribute extent : contract.getAccumulator().getType().getShape())
    factors.push_back(cast<gpu::PhysicalExprAttr>(extent));
  return factors;
}

// Only constexpr control can exclude a form from native compilation. Runtime
// control still requires both branches to have legal tensor shapes.
gpu::PhysicalExprAttr constexprCondition(Value value) {
  auto *context = value.getContext();
  auto zero = constant(context, 0), one = constant(context, 1);
  auto make = [&](gpu::PhysicalExprKind kind, ArrayRef<Attribute> operands) {
    return expression(context, kind, operands);
  };
  if (auto scalar = value.getDefiningOp<arith::ConstantOp>()) {
    auto integer = dyn_cast<IntegerAttr>(scalar.getValue());
    return integer && scalar.getType().isInteger(1)
        ? constant(context, !integer.getValue().isZero()) : gpu::PhysicalExprAttr();
  }
  if (auto binary = value.getDefiningOp<gpu::BinaryOp>()) {
    auto kind = binary.getOperatorKind();
    if (kind != BinaryOperator::LogicalAnd && kind != BinaryOperator::LogicalOr)
      return {};
    auto lhs = constexprCondition(binary.getLhs());
    auto rhs = constexprCondition(binary.getRhs());
    if (!lhs || !rhs) return {};
    return make(kind == BinaryOperator::LogicalAnd
                    ? gpu::PhysicalExprKind::Multiply : gpu::PhysicalExprKind::Maximum,
                {lhs, rhs});
  }
  auto compare = value.getDefiningOp<gpu::CompareOp>();
  if (!compare || !compare.getLhs().getType().isIndex() ||
      !compare.getRhs().getType().isIndex())
    return {};
  auto lhs = gpu::queryLaunchExpression(compare.getLhs());
  auto rhs = gpu::queryLaunchExpression(compare.getRhs());
  if (!lhs || !rhs || !isTritonFragmentExtent(lhs) || !isTritonFragmentExtent(rhs))
    return {};
  auto difference = make(gpu::PhysicalExprKind::Subtract, {lhs, rhs});
  auto positive = [&](gpu::PhysicalExprAttr value) {
    return make(gpu::PhysicalExprKind::Select,
                {make(gpu::PhysicalExprKind::Maximum, {value, zero}), one, zero});
  };
  auto negate = [&](gpu::PhysicalExprAttr value) {
    return make(gpu::PhysicalExprKind::Select, {value, zero, one});
  };
  switch (compare.getPredicate()) {
  case ComparePredicate::Eq:
    return negate(difference);
  case ComparePredicate::Ne:
    return make(gpu::PhysicalExprKind::Select, {difference, one, zero});
  case ComparePredicate::Gt:
    return positive(difference);
  case ComparePredicate::Le:
    return negate(positive(difference));
  case ComparePredicate::Lt:
    return positive(make(gpu::PhysicalExprKind::Subtract, {rhs, lhs}));
  case ComparePredicate::Ge:
    return negate(positive(make(gpu::PhysicalExprKind::Subtract, {rhs, lhs})));
  }
  llvm_unreachable("unknown comparison predicate");
}

} // namespace

gpu::PhysicalExprAttr expandedContractionElements(gpu::ContractOp contract) {
  auto factors = expansionFactors(contract);
  auto result = factors.front();
  for (auto factor : llvm::drop_begin(factors))
    result = expression(contract.getContext(), gpu::PhysicalExprKind::Multiply,
                        {result, factor});
  return result;
}

gpu::ConfigurationRequirementAttr
contractionExpansionRequirement(gpu::ContractOp contract) {
  if (contract.getLhs().getType().getShape().empty()) return {};
  auto *context = contract.getContext();
  auto one = constant(context, 1);
  auto factors = expansionFactors(contract);
  SmallVector<gpu::PhysicalExprAttr> conditions;
  if (!contract->hasAttr(contractFormAttr)) {
    // Before form selection, only short K requires expansion. The optional
    // large-K branch already includes the expanded element bound in its guard.
    auto quotient = expression(context, gpu::PhysicalExprKind::FloorDiv,
        {factors.front(), constant(context, expansionReductionThreshold)});
    conditions.push_back(expression(context, gpu::PhysicalExprKind::Select,
                                    {quotient, constant(context, 0), one}));
  }
  for (Operation *current = contract; Operation *parent = current->getParentOp();
       current = parent) {
    auto conditional = dyn_cast<scf::IfOp>(parent);
    if (!conditional) continue;
    auto condition = constexprCondition(conditional.getCondition());
    if (!condition) continue;
    if (current->getParentRegion() == &conditional.getElseRegion())
      condition = expression(context, gpu::PhysicalExprKind::Select,
                             {condition, constant(context, 0), one});
    conditions.push_back(condition);
  }
  auto usage = one;
  for (auto factor : factors) {
    // Guard each factor before multiplication: the checked evaluator must not
    // overflow an inactive expansion's product while evaluating select operands.
    for (auto condition : conditions)
      factor = expression(context, gpu::PhysicalExprKind::Select,
                          {condition, factor, one});
    usage = expression(context, gpu::PhysicalExprKind::Multiply, {usage, factor});
  }
  return gpu::ConfigurationRequirementAttr::get(context,
      gpu::ConfigurationRequirementKind::Legality,
      gpu::ConfigurationRequirementMetric::FragmentElements,
      gpu::ConfigurationRequirementPredicate::LessEqual, usage,
      constant(context, maxTritonTensorElements), gpu::ParameterRefAttr(),
      StringAttr::get(context, "Triton expanded contraction exceeds the maximum element count"));
}

} // namespace intent::triton
