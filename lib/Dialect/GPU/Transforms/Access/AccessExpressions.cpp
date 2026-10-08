#include "AccessComposition.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"

using namespace mlir;

namespace intent::gpu::access {

bool foldIndexRecompositions(func::FuncOp kernel) {
  bool changed = false;
  SmallVector<BinaryOp> quotients;
  kernel.walk([&](BinaryOp binary) {
    if (binary.getOperatorKind() == BinaryOperator::FloorDivide &&
        uniformElementType(binary.getResult().getType()).isIndex())
      quotients.push_back(binary);
  });
  auto unproject = [](Value value) {
    while (true) {
      if (auto broadcast = value.getDefiningOp<BroadcastOp>())
        value = broadcast.getValue();
      else if (auto reshape = value.getDefiningOp<ReshapeOp>())
        value = reshape.getValue();
      else
        return value;
    }
  };
  for (BinaryOp quotient : quotients) {
    auto range = unproject(quotient.getLhs()).getDefiningOp<MakeRangeOp>();
    Value divisor = unproject(quotient.getRhs());
    if (!range || !isUnitStepRange(range) ||
        !isZero(range.getLogicalStart()) ||
        !samePhysicalScalarExpression(range.getStart(),
                                      range.getLogicalStart()) ||
        !samePhysicalScalarExpression(range.getLogicalStop(), divisor))
      continue;
    // Every active logical member of [0, end) has quotient zero. An empty
    // range has no active members; physical padding retains its existing mask.
    OpBuilder builder(quotient);
    Value zero = builder.create<arith::ConstantIndexOp>(quotient.getLoc(), 0);
    if (auto fragment = dyn_cast<FragmentType>(quotient.getResult().getType()))
      zero = builder.create<SplatOp>(quotient.getLoc(), fragment, zero);
    quotient.getResult().replaceAllUsesWith(zero);
    quotient.erase();
    changed = true;
  }
  SmallVector<BinaryOp> sums;
  kernel.walk([&](BinaryOp op) {
    if (op.getOperatorKind() == BinaryOperator::Add ||
        op.getOperatorKind() == BinaryOperator::Subtract)
      sums.push_back(op);
  });
  for (BinaryOp sum : sums) {
    Type type = sum.getResult().getType();
    Type element = uniformElementType(type);
    if (!isa<IndexType, IntegerType>(element))
      continue;
    unsigned width =
        element.isIndex() ? 64 : cast<IntegerType>(element).getWidth();
    UniformValueAnalysis constants(describeUniformValue);
    auto constant = [&](Value value) -> std::optional<llvm::APInt> {
      auto attr = dyn_cast_or_null<IntegerAttr>(constants.evaluate(value));
      if (!attr || attr.getType() != element)
        return std::nullopt;
      return attr.getValue();
    };
    llvm::MapVector<Value, llvm::APInt> terms;
    auto addTerm = [&](Value value, const llvm::APInt &coefficient) {
      auto [it, inserted] = terms.insert({value, llvm::APInt(width, 0)});
      it->second += coefficient;
    };
    llvm::APInt offset(width, 0);
    SmallVector<std::pair<Value, llvm::APInt>> pending{
        {sum.getResult(), llvm::APInt(width, 1)}};
    bool compatible = true;
    while (!pending.empty()) {
      auto [value, coefficient] = pending.pop_back_val();
      if (value.getType() != type) {
        compatible = false;
        break;
      }
      if (auto literal = constant(value)) {
        offset += coefficient * *literal;
        continue;
      }
      auto binary = value.getDefiningOp<BinaryOp>();
      // Preserve shared arithmetic subexpressions and avoid expanding a DAG.
      if (binary && (value == sum.getResult() || value.hasOneUse())) {
        auto kind = binary.getOperatorKind();
        if (kind == BinaryOperator::Add || kind == BinaryOperator::Subtract) {
          pending.emplace_back(binary.getLhs(), coefficient);
          pending.emplace_back(
              binary.getRhs(),
              kind == BinaryOperator::Add ? coefficient : -coefficient);
          continue;
        }
        if (kind == BinaryOperator::Multiply) {
          if (auto scale = constant(binary.getRhs())) {
            pending.emplace_back(binary.getLhs(), coefficient * *scale);
            continue;
          }
          if (auto scale = constant(binary.getLhs())) {
            pending.emplace_back(binary.getRhs(), coefficient * *scale);
            continue;
          }
        }
      }
      addTerm(value, coefficient);
    }
    if (!compatible)
      continue;

    bool recomposed = false;
    bool merged;
    do {
      merged = false;
      for (auto [remainderValue, scale] : terms) {
        auto remainder = remainderValue.getDefiningOp<BinaryOp>();
        if (scale.isZero() || !remainder ||
            remainder.getLhs().getType() != type ||
            remainder.getOperatorKind() != BinaryOperator::Remainder)
          continue;
        auto divisor = constant(remainder.getRhs());
        if (!divisor || !divisor->isStrictlyPositive())
          continue;
        Value quotientValue;
        for (auto [value, coefficient] : terms) {
          auto quotient = value.getDefiningOp<BinaryOp>();
          if (quotient &&
              quotient.getOperatorKind() == BinaryOperator::FloorDivide &&
              quotient.getLhs() == remainder.getLhs() &&
              constant(quotient.getRhs()) == divisor &&
              coefficient == scale * *divisor) {
            quotientValue = value;
            break;
          }
        }
        if (!quotientValue)
          continue;
        // q*c+r=x also holds modulo 2^width, including negative dividends.
        // Coefficients retain that width; no address or bounds assumptions enter.
        Value dividend = remainder.getLhs();
        terms.erase(quotientValue);
        terms.erase(remainderValue);
        addTerm(dividend, scale);
        merged = recomposed = true;
        break;
      }
    } while (merged);
    if (!recomposed)
      continue;

    OpBuilder builder(sum);
    auto literal = [&](const llvm::APInt &bits) -> Value {
      Value value = builder.create<arith::ConstantOp>(
          sum.getLoc(), IntegerAttr::get(element, bits));
      if (auto fragment = dyn_cast<FragmentType>(type))
        value = builder.create<SplatOp>(sum.getLoc(), fragment, value);
      return value;
    };
    Value result;
    for (auto [value, coefficient] : terms) {
      if (coefficient.isZero())
        continue;
      Value term = value;
      if (!coefficient.isOne())
        term = builder.create<BinaryOp>(sum.getLoc(), type, value,
                                       literal(coefficient),
                                       BinaryOperator::Multiply);
      result = result ? Value(builder.create<BinaryOp>(
                            sum.getLoc(), type, result, term, BinaryOperator::Add))
                      : term;
    }
    if (!offset.isZero() || !result) {
      Value term = literal(offset);
      result = result ? Value(builder.create<BinaryOp>(
                            sum.getLoc(), type, result, term, BinaryOperator::Add))
                      : term;
    }
    sum.getResult().replaceAllUsesWith(result);
    sum.erase();
    changed = true;
  }
  return changed;
}

} // namespace intent::gpu::access

namespace intent::gpu {

LogicalResult simplifyMaskedAccessCoordinates(ModuleOp module) {
  FailureOr<func::FuncOp> kernel = getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  kernel->walk([&](LoadOp load) {
    Value valid = load.getValid();
    if (!valid)
      return;
    auto sameAxes = [&](Type type) {
      auto fragment = dyn_cast<FragmentType>(type);
      auto mask = dyn_cast<FragmentType>(valid.getType());
      if (!fragment || !mask)
        return !fragment && !mask;
      return fragment.getOwner() == mask.getOwner() &&
             fragment.getAxisMaps() == mask.getAxisMaps() &&
             queryBroadcastProjection(fragment, mask).isExact();
    };
    llvm::SmallDenseSet<Value, 16> conjuncts;
    SmallVector<Value> pending{valid};
    while (!pending.empty()) {
      Value value = pending.pop_back_val();
      if (!sameAxes(value.getType()) || !conjuncts.insert(value).second)
        continue;
      if (auto broadcast = value.getDefiningOp<BroadcastOp>()) {
        if (sameAxes(broadcast.getValue().getType()))
          pending.push_back(broadcast.getValue());
        continue;
      }
      auto binary = value.getDefiningOp<BinaryOp>();
      if (!binary ||
          (binary.getOperatorKind() != BinaryOperator::LogicalAnd &&
           binary.getOperatorKind() != BinaryOperator::BitwiseAnd))
        continue;
      pending.push_back(binary.getLhs());
      pending.push_back(binary.getRhs());
    }
    OpBuilder builder(load);
    IRMapping simplified;
    std::function<Value(Value)> simplify = [&](Value value) -> Value {
      if (Value known = simplified.lookupOrNull(value))
        return known;
      if (!sameAxes(value.getType()))
        return value;
      Value result = value;
      if (auto select = value.getDefiningOp<SelectOp>();
          select && conjuncts.contains(select.getCondition())) {
        result = simplify(select.getTrueValue());
      } else if (auto broadcast = value.getDefiningOp<BroadcastOp>();
                 broadcast && sameAxes(broadcast.getValue().getType())) {
        Value operand = simplify(broadcast.getValue());
        if (operand != broadcast.getValue()) {
          IRMapping mapping;
          mapping.map(broadcast.getValue(), operand);
          result = builder.clone(*broadcast, mapping)->getResult(0);
        }
      }
      simplified.map(value, result);
      return result;
    };
    SmallVector<Value> coordinates(load.getCoordinates());
    bool changed = false;
    for (Value &coordinate : coordinates) {
      // Singleton broadcasts with unchanged axes retain the predicate's lane
      // relation. Other uses and the read's original mask stay unchanged.
      Value replacement = simplify(coordinate);
      changed |= replacement != coordinate;
      coordinate = replacement;
    }
    if (changed)
      load.getCoordinatesMutable().assign(coordinates);
  });
  eraseDeadPhysicalValues(*kernel);
  return success();
}

} // namespace intent::gpu
