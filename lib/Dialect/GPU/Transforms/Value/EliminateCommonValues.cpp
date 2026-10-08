#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "ScopePlacement.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"

#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::gpu {
namespace {

std::optional<llvm::APFloat>
widenedConstantReciprocal(const llvm::APFloat &divisor) {
  constexpr auto rounding = llvm::APFloat::rmNearestTiesToEven;
  bool losesInfo = false;
  llvm::APFloat denominator(divisor);
  denominator.convert(llvm::APFloat::IEEEdouble(), rounding, &losesInfo);
  llvm::APFloat inverse(1.0);
  inverse.divide(denominator, rounding);

  // The 53-bit reciprocal times the 24-bit divisor is exact in binary128.
  llvm::APFloat error(inverse);
  error.convert(llvm::APFloat::IEEEquad(), rounding, &losesInfo);
  denominator.convert(llvm::APFloat::IEEEquad(), rounding, &losesInfo);
  error.multiply(denominator, rounding);
  llvm::APFloat one(1.0);
  one.convert(llvm::APFloat::IEEEquad(), rounding, &losesInfo);
  error.subtract(one, rounding);
  error.clearSign();
  llvm::APFloat bound(0x1p-54);
  bound.convert(llvm::APFloat::IEEEquad(), rounding, &losesInfo);
  if (error.compare(bound) == llvm::APFloat::cmpGreaterThan)
    return std::nullopt;
  // This bound keeps exact f32 midpoints (including subnormal ones) in their
  // f64 RN interval. Other f32 quotients are farther from any midpoint than
  // the reciprocal error plus f64 multiplication rounding can reach.
  return inverse;
}

bool foldConstantSelection(SelectOp select) {
  if (!select)
    return false;
  UniformValueAnalysis analysis(describeUniformValue);
  Value selected;
  Attribute trueValue = analysis.evaluate(select.getTrueValue());
  if (select.getTrueValue() == select.getFalseValue() ||
      (trueValue && trueValue == analysis.evaluate(select.getFalseValue())))
    selected = select.getTrueValue();
  else {
    auto condition = uniformBoolean(analysis.evaluate(select.getCondition()));
    if (!condition)
      return false;
    selected = *condition ? select.getTrueValue() : select.getFalseValue();
  }
  if (selected.getType() != select.getResult().getType())
    return false;
  select.getResult().replaceAllUsesWith(selected);
  select.erase();
  return true;
}

} // namespace

void foldScalarIntegerValues(func::FuncOp kernel) {
  UniformValueAnalysis analysis(describeUniformValue);
  kernel.walk([&](Operation *operation) {
    if (operation->getNumResults() != 1 || operation->getNumRegions() != 0 ||
        !operation->getResult(0).getType().isIntOrIndex() ||
        isa<arith::ConstantOp, ParameterOp, PhysicalExprOp>(operation) ||
        !isPure(operation))
      return;
    Value result = operation->getResult(0);
    if (foldConstantSelection(dyn_cast<SelectOp>(operation)))
      return;
    auto constant = dyn_cast_or_null<IntegerAttr>(analysis.evaluate(result));
    if (!constant || constant.getType() != result.getType())
      return;
    OpBuilder builder(operation);
    auto folded = materializeScalarConstant(
        builder, operation->getLoc(), constant, result.getType());
    assert(succeeded(folded) && "integer scalar constant must materialize");
    folded->getDefiningOp()->setDiscardableAttrs(
        llvm::to_vector(operation->getDiscardableAttrs()));
    result.replaceAllUsesWith(*folded);
    operation->erase();
  });
}

void foldExactConstantDivisions(func::FuncOp kernel) {
  auto capabilities =
      kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  bool fastFloat64 =
      capabilities.getSingleToDoublePrecisionPerfRatio() <= 2;
  kernel.walk([&](BinaryOp binary) {
    Type element = uniformElementType(binary.getResult().getType());
    if ((binary.getOperatorKind() == BinaryOperator::FloorDivide ||
         binary.getOperatorKind() == BinaryOperator::Remainder) &&
        isa<IndexType, IntegerType>(element)) {
      auto constant = dyn_cast_or_null<IntegerAttr>(
          UniformValueAnalysis(describeUniformValue).evaluate(binary.getRhs()));
      if (!constant || !constant.getValue().isStrictlyPositive() ||
          !constant.getValue().isPowerOf2())
        return;
      bool remainder = binary.getOperatorKind() == BinaryOperator::Remainder;
      if (!remainder && constant.getValue().isOne()) {
        if (binary.getLhs().getType() == binary.getResult().getType()) {
          binary.getResult().replaceAllUsesWith(binary.getLhs());
          binary.erase();
        }
        return;
      }
      int64_t value = remainder ? constant.getInt() - 1
                                : constant.getValue().logBase2();
      OpBuilder builder(binary);
      Value operand = builder.create<arith::ConstantOp>(
          binary.getLoc(), IntegerAttr::get(element, value));
      if (auto fragment = dyn_cast<FragmentType>(binary.getRhs().getType()))
        operand = builder.create<SplatOp>(binary.getLoc(), fragment, operand);
      // Arithmetic right shift and a low-bit mask implement floor division
      // and remainder by positive powers of two, including negative inputs.
      // Expose these before native layout analysis; a truncation correction
      // otherwise obscures the contiguous groups in reshaped addresses.
      binary->setOperand(1, operand);
      binary.setOperatorKind(remainder ? BinaryOperator::BitwiseAnd
                                       : BinaryOperator::RightShift);
      return;
    }
    if (binary.getOperatorKind() != BinaryOperator::TrueDivide ||
        binary.getApproximate() || binary.getFlushToZero() ||
        (!element.isF32() && !element.isF64()))
      return;
    auto constant = dyn_cast_or_null<FloatAttr>(
        UniformValueAnalysis(describeUniformValue).evaluate(binary.getRhs()));
    if (!constant || !constant.getValue().isNormal())
      return;
    llvm::APFloat inverse(constant.getValue().getSemantics());
    bool widened = false;
    if (!constant.getValue().getExactInverse(&inverse) || !inverse.isNormal()) {
      if (!element.isF32() || !fastFloat64)
        return;
      auto reciprocal = widenedConstantReciprocal(constant.getValue());
      if (!reciprocal)
        return;
      inverse = std::move(*reciprocal);
      widened = true;
    }
    OpBuilder builder(binary);
    Type computationElement = widened ? builder.getF64Type() : element;
    auto computationType = [&](Type type) -> Type {
      if (auto fragment = dyn_cast<FragmentType>(type))
        return FragmentType::get(
            kernel.getContext(), computationElement, fragment.getShape(),
            fragment.getAxisMaps(), fragment.getValidity(), fragment.getOwner());
      return computationElement;
    };
    Value reciprocal = builder.create<arith::ConstantOp>(
        binary.getLoc(), FloatAttr::get(computationElement, inverse));
    if (auto fragment = dyn_cast<FragmentType>(binary.getRhs().getType()))
      reciprocal = builder.create<SplatOp>(
          binary.getLoc(), cast<FragmentType>(computationType(fragment)),
          reciprocal);
    Value lhs = binary.getLhs();
    if (widened)
      lhs = builder.create<CastOp>(binary.getLoc(), computationType(lhs.getType()),
                                   lhs);
    auto product = builder.create<BinaryOp>(
        binary.getLoc(), computationType(binary.getType()), lhs, reciprocal,
        BinaryOperator::Multiply);
    Value result = product.getResult();
    if (widened)
      result = builder.create<CastOp>(binary.getLoc(), binary.getType(), result);
    result.getDefiningOp()->setDiscardableAttrs(
        llvm::to_vector(binary->getDiscardableAttrs()));
    binary.getResult().replaceAllUsesWith(result);
    binary.erase();
  });
}

namespace {

LogicalResult canonicalizeCommonValues(func::FuncOp kernel) {
  foldScalarIntegerValues(kernel);
  // A custom fragment splat is uniform without being an IntegerAttr operand
  // for MLIR's SelectOp fold adaptor. Fold its selected SSA value everywhere,
  // including predicates that prove bounds for a separately reused coordinate.
  kernel->walk<WalkOrder::PostOrder>([](SelectOp select) {
    foldConstantSelection(select);
  });
  RewritePatternSet patterns(kernel.getContext());
  BinaryOp::getCanonicalizationPatterns(patterns, kernel.getContext());
  ReshapeOp::getCanonicalizationPatterns(patterns, kernel.getContext());
  TransposeOp::getCanonicalizationPatterns(patterns, kernel.getContext());
  SelectOp::getCanonicalizationPatterns(patterns, kernel.getContext());
  if (failed(applyPatternsAndFoldGreedily(kernel, std::move(patterns))))
    return kernel->emitError("local value canonicalization did not converge");
  return success();
}

void eliminateCommonSubExpressions(func::FuncOp kernel) {
  IRRewriter rewriter(kernel.getContext());
  DominanceInfo dominance(kernel);
  mlir::eliminateCommonSubExpressions(rewriter, dominance, kernel.getOperation());
}

} // namespace

LogicalResult eliminateCommonValues(ModuleOp module) {
  auto kernel = getPhysicalKernel(module);
  if (failed(kernel) || failed(canonicalizeCommonValues(*kernel)))
    return failure();
  eliminateCommonSubExpressions(*kernel);
  return success();
}

LogicalResult hoistLoopInvariantValues(ModuleOp module) {
  auto kernel = getPhysicalKernel(module);
  if (failed(kernel) || failed(canonicalizeCommonValues(*kernel)) ||
      failed(placement::hoistLoopInvariantValues(*kernel)))
    return failure();
  eliminateCommonSubExpressions(*kernel);
  return success();
}

} // namespace intent::gpu
