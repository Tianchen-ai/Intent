#include "Intent/Dialect/GPU/Analysis/IntegerRanges.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalExpressionBounds.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/ProgramInterface.h"
#include "ScalarExpressions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/Utils/InferIntRangeCommon.h"

using namespace mlir;

namespace intent::gpu {
namespace {

Type elementType(Type type) {
  if (auto fragment = dyn_cast<FragmentType>(type))
    return fragment.getElementType();
  return type;
}

ConstantIntRanges nonNegativeIndex() {
  return ConstantIntRanges::fromSigned(APInt(64, 0),
                                       APInt::getSignedMaxValue(64));
}

std::optional<ConstantIntRanges> inferGPUFacts(Value value,
                                               IntegerRangeAnalysis &analysis) {
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  if (auto physical = value.getDefiningOp<PhysicalExprOp>())
    return queryPhysicalExpressionRange(physical.getExpression(), kernel);
  if (auto parameter = value.getDefiningOp<ParameterOp>()) {
    auto declaration = parameter.getDeclaration();
    if (!declaration || !declaration.isExtent())
      return std::nullopt;
    auto expression = PhysicalExprAttr::get(
        value.getContext(), PhysicalExprKind::Parameter, 0,
        declaration.getReference(), ArrayAttr::get(value.getContext(), {}));
    if (auto range = queryPhysicalExpressionRange(expression, kernel))
      return range;
    // Provider configuration may rebind resident capacity, but the parameter's
    // positive index contract is independent of that finite candidate set.
    return ConstantIntRanges::fromSigned(APInt(64, 1),
                                         APInt::getSignedMaxValue(64));
  }
  if (value.getDefiningOp<ProgramIdOp>())
    return nonNegativeIndex();
  if (auto dimension = value.getDefiningOp<DimOp>()) {
    auto expression = detail::resourceExtentExpression(dimension.getView(),
                                                       dimension.getAxis());
    if (auto range = queryPhysicalExpressionRange(expression, kernel))
      return range->intersection(nonNegativeIndex());
    return nonNegativeIndex();
  }
  if (auto argument = dyn_cast<BlockArgument>(value))
    if (auto binding = getArgumentBinding(argument);
        binding && binding.getKind() == ArgumentKind::Dimension)
      return nonNegativeIndex();
  if (auto coordinate = value.getDefiningOp<WorksetCoordinateOp>())
    return analysis.range(coordinate.getCoordinate());
  if (auto bound = value.getDefiningOp<RangeBoundOp>()) {
    if (auto range = bound.getRange().getDefiningOp<RangeOp>())
      return analysis.range(bound.getBound() == 0   ? range.getStart()
                            : bound.getBound() == 1 ? range.getStop()
                                                    : range.getStep());
  }
  if (auto coordinate = queryDecodedCoordinate(value)) {
    // Mathematical decoding takes a divisor-sign remainder on every axis.
    // Thus defined executions with nonnegative extents have bounded coordinates
    // even when the linear expression cannot be proved free of signed wrap.
    // A zero divisor has no defined decoded value.
    if (!llvm::all_of(coordinate->extents, [&](Value extent) {
          return analysis.isNonNegative(extent);
        }))
      return std::nullopt;
    auto extent = analysis.range(coordinate->extents[coordinate->axis]);
    if (!extent || extent->smax().isZero())
      return std::nullopt;
    APInt maximum = extent->smax() - 1;
    return ConstantIntRanges::fromSigned(APInt(64, 0), maximum);
  }
  if (auto range = value.getDefiningOp<MakeRangeOp>()) {
    auto begin = analysis.range(range.getStart());
    auto extent = analysis.range(range.getExtent());
    auto step = analysis.range(range.getStep());
    if (!begin || !extent || !step || extent->smin().isNegative() ||
        extent->smax().isZero())
      return std::nullopt;
    auto lanes =
        ConstantIntRanges::fromSigned(APInt(64, 0), extent->smax() - 1);
    auto offsets = intrange::inferMul({lanes, *step});
    return intrange::inferAdd({*begin, offsets});
  }
  if (auto extract = value.getDefiningOp<ExtractOp>())
    if (auto record = extract.getRecord().getDefiningOp<MakeRecordOp>())
      return analysis.range(record.getFields()[extract.getField()]);
  return std::nullopt;
}

} // namespace

IntegerRangePolicy integerRangePolicy() {
  IntegerRangePolicy policy;
  policy.elementType = [](Value value) { return elementType(value.getType()); };
  policy.infer = inferGPUFacts;
  return policy;
}

std::optional<ConstantIntRanges> queryIntegerRange(Value value) {
  if (!value)
    return std::nullopt;
  IntegerRangeAnalysis analysis(integerRangePolicy());
  return analysis.range(value);
}

bool isValuePreservingIntegerCast(Value source, Type targetType) {
  auto range = queryIntegerRange(source);
  return range &&
         intent::isValuePreservingIntegerCast(elementType(source.getType()),
                                              elementType(targetType), *range);
}

bool integerOperationDoesNotWrap(Value value) {
  auto binary = value.getDefiningOp<BinaryOp>();
  if (!binary)
    return false;
  auto integer = dyn_cast<IntegerType>(elementType(value.getType()));
  if (integer && integer.isUnsigned())
    return false;
  IntegerRangeAnalysis analysis(integerRangePolicy());
  auto lhs = analysis.range(binary.getLhs()),
       rhs = analysis.range(binary.getRhs());
  return lhs && rhs && provesSignedNoWrap(binary.getOperatorKind(), *lhs, *rhs);
}

} // namespace intent::gpu
