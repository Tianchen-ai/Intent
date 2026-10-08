#include "ScalarExpressions.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Analysis/IntegerRanges.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/IR/ProgramInterface.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/STLExtras.h"

#include <functional>

using namespace mlir;

namespace intent::gpu {
using namespace detail;

namespace detail {

std::optional<int64_t> integerConstant(Value value) {
  if (!value) return std::nullopt;
  auto range = queryIntegerRange(value);
  auto constant = range ? range->getConstantValue() : std::nullopt;
  if (!constant) {
    auto folded = dyn_cast_or_null<IntegerAttr>(
        UniformValueAnalysis(describeUniformValue).evaluate(value));
    if (folded) constant = folded.getValue();
  }
  if (!constant || constant->getBitWidth() > 64) return std::nullopt;
  auto integer = dyn_cast<IntegerType>(uniformElementType(value.getType()));
  if (integer && (integer.isUnsigned() || integer.getWidth() == 1)) {
    if (constant->getActiveBits() > 63) return std::nullopt;
    return constant->getZExtValue();
  }
  return constant->getSExtValue();
}

Value stripBroadcast(Value value) {
  while (auto broadcast = value.getDefiningOp<BroadcastOp>())
    value = broadcast.getValue();
  while (auto splat = value.getDefiningOp<SplatOp>())
    value = splat.getValue();
  return value;
}

Value stripScalarIdentity(Value value) {
  value = stripBroadcast(value);
  while (true) {
    if (Value scalar = uniformScalarSource(value); scalar && scalar != value) {
      value = scalar;
      continue;
    }
    if (auto cast = value.getDefiningOp<CastOp>()) {
      if (cast.getValue().getType() == cast.getResult().getType()) {
        value = stripBroadcast(cast.getValue());
        continue;
      }
      return value;
    }
    auto binary = value.getDefiningOp<BinaryOp>();
    if (!binary)
      return value;
    auto isInteger = [](Value operand, int64_t expected) {
      std::optional<int64_t> value = integerConstant(operand);
      return value && *value == expected;
    };
    if (binary.getOperatorKind() == BinaryOperator::Add) {
      if (isInteger(binary.getLhs(), 0)) {
        value = stripBroadcast(binary.getRhs());
        continue;
      }
      if (isInteger(binary.getRhs(), 0)) {
        value = stripBroadcast(binary.getLhs());
        continue;
      }
    }
    if (binary.getOperatorKind() == BinaryOperator::Multiply) {
      if (isInteger(binary.getLhs(), 0))
        return binary.getLhs();
      if (isInteger(binary.getRhs(), 0))
        return binary.getRhs();
      if (isInteger(binary.getLhs(), 1)) {
        value = stripBroadcast(binary.getRhs());
        continue;
      }
      if (isInteger(binary.getRhs(), 1)) {
        value = stripBroadcast(binary.getLhs());
        continue;
      }
    }
    if (binary.getOperatorKind() == BinaryOperator::Subtract &&
        isInteger(binary.getRhs(), 0)) {
      value = stripBroadcast(binary.getLhs());
      continue;
    }
    if (binary.getOperatorKind() == BinaryOperator::FloorDivide &&
        isInteger(binary.getRhs(), 1)) {
      value = stripBroadcast(binary.getLhs());
      continue;
    }
    return value;
  }
}

bool sameScalarExpression(Value lhs, Value rhs, unsigned depth) {
  if (lhs == rhs)
    return true;
  if (depth >= 32 || lhs.getType() != rhs.getType())
    return false;
  Value left = stripScalarIdentity(lhs);
  Value right = stripScalarIdentity(rhs);
  if (left != lhs || right != rhs)
    return sameScalarExpression(left, right, depth + 1);
  auto leftConstant = lhs.getDefiningOp<arith::ConstantOp>();
  auto rightConstant = rhs.getDefiningOp<arith::ConstantOp>();
  if (leftConstant || rightConstant)
    return leftConstant && rightConstant &&
           leftConstant.getValue() == rightConstant.getValue();
  auto leftParameter = queryParameter(lhs), rightParameter = queryParameter(rhs);
  if (leftParameter || rightParameter)
    return leftParameter && rightParameter &&
           leftParameter.getReference() == rightParameter.getReference() &&
           lhs.getDefiningOp()->getParentOfType<func::FuncOp>() ==
               rhs.getDefiningOp()->getParentOfType<func::FuncOp>();
  auto leftExpression = lhs.getDefiningOp<PhysicalExprOp>();
  auto rightExpression = rhs.getDefiningOp<PhysicalExprOp>();
  if (leftExpression || rightExpression)
    return leftExpression && rightExpression &&
           leftExpression->getParentOfType<func::FuncOp>() ==
               rightExpression->getParentOfType<func::FuncOp>() &&
           leftExpression.getExpression() == rightExpression.getExpression();
  auto leftBinary = lhs.getDefiningOp<BinaryOp>();
  auto rightBinary = rhs.getDefiningOp<BinaryOp>();
  if (leftBinary || rightBinary)
    return leftBinary && rightBinary &&
           leftBinary.getOperatorKind() == rightBinary.getOperatorKind() &&
           leftBinary.getApproximate() == rightBinary.getApproximate() &&
           leftBinary.getFlushToZero() == rightBinary.getFlushToZero() &&
           leftBinary.getStrictRounding() == rightBinary.getStrictRounding() &&
           sameScalarExpression(leftBinary.getLhs(), rightBinary.getLhs(),
                                depth + 1) &&
           sameScalarExpression(leftBinary.getRhs(), rightBinary.getRhs(),
                                depth + 1);
  auto leftCompare = lhs.getDefiningOp<CompareOp>();
  auto rightCompare = rhs.getDefiningOp<CompareOp>();
  if (leftCompare || rightCompare)
    return leftCompare && rightCompare &&
           leftCompare.getPredicate() == rightCompare.getPredicate() &&
           sameScalarExpression(leftCompare.getLhs(), rightCompare.getLhs(),
                                depth + 1) &&
           sameScalarExpression(leftCompare.getRhs(), rightCompare.getRhs(),
                                depth + 1);
  auto leftDim = lhs.getDefiningOp<DimOp>();
  auto rightDim = rhs.getDefiningOp<DimOp>();
  if (leftDim || rightDim) {
    if (!leftDim || !rightDim ||
        leftDim->getParentOfType<func::FuncOp>() !=
            rightDim->getParentOfType<func::FuncOp>())
      return false;
    PhysicalExprAttr leftExtent =
        resourceExtentExpression(leftDim.getView(), leftDim.getAxis());
    PhysicalExprAttr rightExtent =
        resourceExtentExpression(rightDim.getView(), rightDim.getAxis());
    return leftExtent && rightExtent && leftExtent == rightExtent;
  }
  auto leftCast = lhs.getDefiningOp<CastOp>();
  auto rightCast = rhs.getDefiningOp<CastOp>();
  return leftCast && rightCast &&
         sameScalarExpression(leftCast.getValue(), rightCast.getValue(),
                              depth + 1);
}

bool isUnitStepValue(Value value) {
  return integerConstant(value) == 1;
}

PhysicalExprAttr resourceExtentExpression(Value resource, unsigned axis) {
  if (auto view = dyn_cast<ViewType>(resource.getType())) {
    if (axis < view.getRank())
      return cast<PhysicalExprAttr>(view.getLayout().getExtents()[axis]);
    return {};
  }
  if (auto buffer = dyn_cast<BufferType>(resource.getType())) {
    if (axis < buffer.getShape().size())
      return cast<PhysicalExprAttr>(buffer.getShape()[axis]);
    return {};
  }
  if (auto fragment = dyn_cast<FragmentType>(resource.getType())) {
    if (axis < fragment.getShape().size())
      return cast<PhysicalExprAttr>(fragment.getShape()[axis]);
  }
  return {};
}

bool valueMatchesExtent(Value value, PhysicalExprAttr extent) {
  if (queryLaunchExpression(value) == extent)
    return true;
  if (auto physical = value.getDefiningOp<PhysicalExprOp>())
    return physical.getExpression() == extent;
  auto kind = extent.getKind();
  if (kind == PhysicalExprKind::Constant) {
    std::optional<int64_t> constant = integerConstant(value);
    return constant && *constant == extent.getValue();
  }
  if (kind == PhysicalExprKind::Parameter) {
    auto parameter = queryParameter(value);
    return parameter &&
           parameter.getName() == extent.getParameterReference().getName();
  }
  return false;
}

} // namespace detail

PhysicalParameterBinding queryParameterBinding(ParameterAttr parameter) {
  PhysicalParameterBinding result;
  if (!parameter)
    return result;
  auto dimension = parameter.getBinding().getDimension();
  if (dimension && dimension.getInt() <= 0)
    return result;
  if (dimension)
    result.dimension = dimension.getInt();
  if (auto source = parameter.getBinding().getSource())
    result.source =
        PhysicalSourceAxis{source.getSourceId(), source.getSourceAxis(),
                           source.getDerived()};
  result.state = result.dimension || result.source ? PhysicalFactState::Exact
                                                   : PhysicalFactState::Unknown;
  return result;
}

ParameterAttr queryParameter(Value value) {
  if (value.getType().isIndex())
    value = stripScalarIdentity(value);
  if (auto read = value.getDefiningOp<ParameterOp>())
    return read.getDeclaration();
  auto expression = value.getDefiningOp<PhysicalExprOp>();
  if (!expression || !expression.getExpression().getParameterReference())
    return {};
  return lookupParameter(expression->getParentOfType<func::FuncOp>(),
                         expression.getExpression().getParameterReference());
}

FailureOr<ParameterAttr> queryParameterBySymbol(func::FuncOp kernel,
                                                StringAttr symbol) {
  auto parameter = lookupParameter(kernel, symbol);
  return parameter ? FailureOr<ParameterAttr>(parameter)
                   : FailureOr<ParameterAttr>(failure());
}

FailureOr<ParameterAttr> queryBlockingParameter(func::FuncOp kernel,
                                              MakeRangeOp range) {
  auto fragment = dyn_cast<FragmentType>(range.getResult().getType());
  auto extent = fragment && fragment.getShape().size() == 1
                    ? dyn_cast<PhysicalExprAttr>(fragment.getShape()[0])
                    : PhysicalExprAttr();
  if (extent && extent.getKind() ==
                    PhysicalExprKind::Parameter)
    return queryParameterBySymbol(kernel, extent.getParameterReference().getName());

  PhysicalSourceAxis source{range.getSourceId(), range.getSourceAxis(),
                            range.getDerived()};
  FailureOr<int64_t> dimension = queryRangeDimension(range);
  ParameterAttr sourceMatch;
  ParameterAttr dimensionMatch;
  bool sourceAmbiguous = false;
  bool dimensionAmbiguous = false;
  for (Attribute declaration : getParameterDeclarations(kernel)) {
    auto parameter = cast<ParameterAttr>(declaration);
    PhysicalParameterBinding binding = queryParameterBinding(parameter);
    if (!binding.isExact())
      continue;
    if (binding.source && *binding.source == source) {
      if (sourceMatch && sourceMatch != parameter)
        sourceAmbiguous = true;
      else
        sourceMatch = parameter;
    }
    if (succeeded(dimension) && binding.dimension &&
        *binding.dimension == *dimension) {
      if (dimensionMatch && dimensionMatch != parameter)
        dimensionAmbiguous = true;
      else
        dimensionMatch = parameter;
    }
  }
  if (sourceMatch && !sourceAmbiguous)
    return sourceMatch;
  return dimensionMatch && !dimensionAmbiguous
             ? FailureOr<ParameterAttr>(dimensionMatch)
             : FailureOr<ParameterAttr>(failure());
}

bool samePhysicalScalarExpression(Value lhs, Value rhs) {
  if (sameScalarExpression(lhs, rhs))
    return true;
  if (lhs.getType() != rhs.getType() ||
      !isa<IndexType, IntegerType>(lhs.getType()))
    return false;
  UniformValueAnalysis constants(describeUniformValue);
  Attribute left = constants.evaluate(lhs);
  Attribute right = constants.evaluate(rhs);
  return left && right && equalUniformConstants(left, right);
}

bool isLaunchUniformScalar(Value value, func::FuncOp kernel) {
  SmallVector<Value> pending{value};
  llvm::SmallPtrSet<Operation *, 16> visited;
  while (!pending.empty()) {
    Value current = pending.pop_back_val();
    if (!current.getType().isIntOrIndexOrFloat())
      return false;
    if (auto argument = dyn_cast<BlockArgument>(current)) {
      if (argument.getOwner() != &kernel.front())
        return false;
      continue;
    }
    Operation *producer = current.getDefiningOp();
    if (!producer)
      return false;
    if (!visited.insert(producer).second)
      continue;
    if (isa<arith::ConstantOp, PhysicalExprOp, ParameterOp>(producer))
      continue;
    if (!isa<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp>(producer))
      return false;
    llvm::append_range(pending, producer->getOperands());
  }
  return true;
}

bool isSingletonExecutionGroup(ExecutionGroupOp group, func::FuncOp kernel) {
  if (!group || group->getParentOfType<func::FuncOp>() != kernel)
    return false;
  auto space = kernel->getAttrOfType<ArrayAttr>(programSpaceAttr);
  if (!space || space.empty() ||
      !llvm::all_of(space, [](Attribute attribute) {
        auto extent = dyn_cast<PhysicalExprAttr>(attribute);
        return extent && constantPhysicalExpression(extent) == 1;
      }) || constantPhysicalExpression(group.getSegmentOffset()) != 0 ||
      constantPhysicalExpression(group.getSegmentLength()) != 1)
    return false;
  if (!llvm::all_of(group.getLaunchExtents(), [](Attribute attribute) {
        auto extent = dyn_cast<PhysicalExprAttr>(attribute);
        return extent && constantPhysicalExpression(extent) == 1;
      }))
    return false;
  return llvm::all_of(group.getExtents(), [](Value value) {
    auto extent = queryLaunchExpression(value);
    return extent && constantPhysicalExpression(extent) == 1;
  });
}

PhysicalExprAttr queryLaunchExpression(Value value) {
  std::function<PhysicalExprAttr(Value, unsigned)> query =
      [&](Value current, unsigned depth) -> PhysicalExprAttr {
    if (!current || depth >= 32 ||
        !isa<IndexType, IntegerType>(current.getType()))
      return {};
    MLIRContext *context = current.getContext();
    auto expression = [&](PhysicalExprKind kind, int64_t constant = 0,
                          StringRef symbol = {},
                          ArrayRef<Attribute> operands = {}) {
      return PhysicalExprAttr::get(
          context, kind, constant,
          kind == PhysicalExprKind::Parameter
              ? Attribute(ParameterRefAttr::get(context, StringAttr::get(context, symbol)))
              : Attribute(StringAttr::get(context, symbol)),
          ArrayAttr::get(context, operands));
    };
    if (auto constant = current.getDefiningOp<arith::ConstantOp>()) {
      auto integer = dyn_cast<IntegerAttr>(constant.getValue());
      if (!integer || integer.getValue().getBitWidth() > 64)
        return {};
      auto type = dyn_cast<IntegerType>(current.getType());
      if (type && (type.isUnsigned() || type.getWidth() == 1)) {
        if (integer.getValue().getActiveBits() > 63)
          return {};
        return expression(PhysicalExprKind::Constant,
                          integer.getValue().getZExtValue());
      }
      return expression(PhysicalExprKind::Constant, integer.getInt());
    }
    if (auto argument = dyn_cast<BlockArgument>(current))
      return queryArgumentExpression(argument);
    if (auto dim = current.getDefiningOp<DimOp>())
      return resourceExtentExpression(dim.getView(), dim.getAxis());
    if (auto physical = current.getDefiningOp<PhysicalExprOp>())
      return physical.getExpression();
    if (auto parameter = queryParameter(current))
      return expression(PhysicalExprKind::Parameter, 0,
                        parameter.getName().getValue());
    if (auto cast = current.getDefiningOp<arith::IndexCastOp>()) {
      auto integer = dyn_cast<IntegerType>(cast.getIn().getType());
      return integer && !integer.isUnsigned() && integer.getWidth() > 1 &&
                     integer.getWidth() <= 64 && current.getType().isIndex()
                 ? query(cast.getIn(), depth + 1)
                 : PhysicalExprAttr();
    }
    if (auto cast = current.getDefiningOp<CastOp>()) {
      Type source = cast.getValue().getType();
      return source == current.getType() ||
                     gpu::isValuePreservingIntegerCast(cast.getValue(), current.getType())
                 ? query(cast.getValue(), depth + 1)
                 : PhysicalExprAttr();
    }
    if (auto bound = current.getDefiningOp<RangeBoundOp>()) {
      auto range = bound.getRange().getDefiningOp<RangeOp>();
      if (!range)
        return {};
      return query(bound.getBound() == 0 ? range.getStart()
                   : bound.getBound() == 1 ? range.getStop() : range.getStep(),
                   depth + 1);
    }
    if (auto compare = current.getDefiningOp<CompareOp>()) {
      bool equal = compare.getPredicate() == ComparePredicate::Eq;
      if ((!equal && compare.getPredicate() != ComparePredicate::Ne) ||
          compare.getLhs().getType() != compare.getRhs().getType())
        return {};
      PhysicalExprAttr lhs = query(compare.getLhs(), depth + 1);
      PhysicalExprAttr rhs = query(compare.getRhs(), depth + 1);
      if (!lhs || !rhs)
        return {};
      // Earlier fixed-width arithmetic may wrap differently from Python's
      // launch evaluation. Direct ABI values and constants have no such step.
      if (!lhs.getOperands().empty() || !rhs.getOperands().empty())
        return {};
      // Equality only observes a zero difference, which is preserved by
      // same-width integer wraparound and host launch arithmetic alike.
      auto difference = expression(PhysicalExprKind::Subtract, 0, {}, {lhs, rhs});
      return expression(PhysicalExprKind::Select, 0, {},
          {difference, expression(PhysicalExprKind::Constant, equal ? 0 : 1),
           expression(PhysicalExprKind::Constant, equal ? 1 : 0)});
    }
    auto binary = current.getDefiningOp<BinaryOp>();
    if (!binary || !current.getType().isIndex())
      return {};
    if ((binary.getOperatorKind() == BinaryOperator::Add ||
         binary.getOperatorKind() == BinaryOperator::Subtract ||
         binary.getOperatorKind() == BinaryOperator::Multiply) &&
        !integerOperationDoesNotWrap(current))
      return {};
    PhysicalExprAttr lhs = query(binary.getLhs(), depth + 1);
    PhysicalExprAttr rhs = query(binary.getRhs(), depth + 1);
    if (!lhs || !rhs)
      return {};
    PhysicalExprKind kind;
    switch (binary.getOperatorKind()) {
    case BinaryOperator::Add: kind = PhysicalExprKind::Add; break;
    case BinaryOperator::Subtract: kind = PhysicalExprKind::Subtract; break;
    case BinaryOperator::Multiply: kind = PhysicalExprKind::Multiply; break;
    case BinaryOperator::FloorDivide: kind = PhysicalExprKind::FloorDiv; break;
    case BinaryOperator::Minimum: kind = PhysicalExprKind::Minimum; break;
    case BinaryOperator::Maximum: kind = PhysicalExprKind::Maximum; break;
    default: return {};
    }
    return expression(kind, 0, {}, {lhs, rhs});
  };
  return query(value, 0);
}

PhysicalExprAttr queryLaunchRangeExtent(MakeRangeOp range) {
  MLIRContext *context = range.getContext();
  auto expression = [&](PhysicalExprKind kind, int64_t value = 0,
                        ArrayRef<Attribute> operands = {}) {
    return PhysicalExprAttr::get(
        context, kind, value, StringAttr::get(context),
        ArrayAttr::get(context, operands));
  };
  if (auto count = constantLogicalRangeCardinality(range))
    return expression(PhysicalExprKind::Constant, *count);
  PhysicalExprAttr start = queryLaunchExpression(range.getLogicalStart());
  PhysicalExprAttr stop = queryLaunchExpression(range.getLogicalStop());
  PhysicalExprAttr step = queryLaunchExpression(range.getStep());
  if (!start || !stop || !step)
    return {};
  if (step.getKind() == PhysicalExprKind::Constant &&
      step.getValue() <= 0)
    return {};
  PhysicalExprAttr distance =
      expression(PhysicalExprKind::Subtract, 0, {stop, start});
  distance = expression(PhysicalExprKind::Maximum, 0,
                        {distance, expression(PhysicalExprKind::Constant, 0)});
  return expression(PhysicalExprKind::CeilDiv, 0, {distance, step});
}

} // namespace intent::gpu
