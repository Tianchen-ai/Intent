#include "ReductionValues.h"
#include "ReductionAnalysis.h"
#include "Intent/Dialect/GPU/Transforms/Value/Helpers.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Value/ExecutionSchema.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;

namespace intent::gpu::reduction {

void inheritRangeAuthority(Operation *target, MakeRangeOp source) {
  for (StringRef name :
       {originAttr, sourceSubregionAttr, sourceSubregionBoundAttr,
        worksetCoordinateRangeAttr})
    if (Attribute value = source->getAttr(name))
      target->setAttr(name, value);
}

PhysicalExprAttr expression(MLIRContext *context, PhysicalExprKind kind,
                            int64_t value, StringRef symbol,
                            ArrayRef<Attribute> operands) {
  return PhysicalExprAttr::get(
      context, kind, value,
      kind == PhysicalExprKind::Parameter
          ? Attribute(ParameterRefAttr::get(context, StringAttr::get(context, symbol)))
          : Attribute(StringAttr::get(context, symbol)),
      ArrayAttr::get(context, operands));
}

PhysicalExprAttr nextPowerOfTwo(PhysicalExprAttr source) {
  if (source.getKind() == PhysicalExprKind::Constant) {
    uint64_t value = std::max<int64_t>(source.getValue(), 1);
    uint64_t result = 1;
    while (result < value)
      result <<= 1;
    return expression(source.getContext(), PhysicalExprKind::Constant, result);
  }
  return expression(source.getContext(), PhysicalExprKind::NextPowerOfTwo, 0,
                    {}, {source});
}

FragmentType replaceExtent(FragmentType source, unsigned axis,
                           PhysicalExprAttr extent, Type element) {
  SmallVector<Attribute> shape(source.getShape().begin(), source.getShape().end());
  shape[axis] = extent;
  return FragmentType::get(source.getContext(),
                           element ? element : source.getElementType(),
                           ArrayAttr::get(source.getContext(), shape),
                           source.getAxisMaps(), source.getValidity(),
                           source.getOwner());
}

FailureOr<Value> predicateForReductionSource(OpBuilder &builder,
                                             Location location, Value predicate,
                                             FragmentType source,
                                             unsigned reductionAxis) {
  if (reductionAxis >= source.getAxisMaps().size())
    return failure();
  return projectPredicateToFragmentAxis(builder, location, predicate, source,
                                        reductionAxis);
}

FailureOr<SmallVector<Value>> inlinePureRegion(OpBuilder &builder, Region &region,
                                               ValueRange arguments,
                                               std::string &reason) {
  if (region.empty() || region.getBlocks().size() != 1 ||
      region.front().getNumArguments() != arguments.size()) {
    reason = ("combine argument schema mismatch: expected " +
              Twine(region.empty() ? 0 : region.front().getNumArguments()) +
              ", got " + Twine(arguments.size()))
                 .str();
    return failure();
  }
  auto yield = dyn_cast<YieldOp>(region.front().getTerminator());
  if (!yield) {
    reason = ("combine region terminates with " +
              region.front().getTerminator()->getName().getStringRef())
                 .str();
    return failure();
  }
  IRMapping mapping;
  for (auto [argument, value] :
       llvm::zip(region.front().getArguments(), arguments)) {
    if (argument.getType() != value.getType()) {
      reason = "combine invocation does not match its selected formal schema";
      return failure();
    }
    mapping.map(argument, value);
  }
  for (Operation &operation : region.front().without_terminator()) {
    if (failed(cloneWithPhysicalSchema(builder, &operation, mapping,
                                       [](Value value) { return value.getType(); }))) {
      reason = "combine operation cannot preserve its selected schema";
      return failure();
    }
  }
  SmallVector<Value> results;
  for (Value value : yield.getValues()) {
    Value mapped = mapping.lookupOrNull(value);
    if (!mapped) {
      reason = "combine yield value was not mapped by pure-region cloning";
      return failure();
    }
    results.push_back(mapped);
  }
  return results;
}

Type dataElementType(Type type) {
  if (auto fragment = dyn_cast<FragmentType>(type))
    return fragment.getElementType();
  return type;
}

FragmentType withElementType(FragmentType schema, Type elementType) {
  return FragmentType::get(schema.getContext(), elementType, schema.getShape(),
                           schema.getAxisMaps(), schema.getValidity(),
                           schema.getOwner());
}

namespace {

bool sameExecutionSchema(FragmentType lhs, FragmentType rhs) {
  return lhs.getShape() == rhs.getShape() &&
         lhs.getAxisMaps() == rhs.getAxisMaps() &&
         lhs.getValidity() == rhs.getValidity() &&
         lhs.getOwner() == rhs.getOwner();
}

} // namespace

FailureOr<Value> alignToExecutionSchema(OpBuilder &builder, Location location,
                                        Value value,
                                        FragmentType executionSchema) {
  FragmentType target =
      withElementType(executionSchema, dataElementType(value.getType()));
  return projectPhysicalValueToSchema(builder, location, value, target);
}

bool prepareVectorAccumulation(ReduceOp reduce,
                               ArrayRef<FragmentType> accumulatorTypes,
                               Region &combine) {
  ValueRange identities = reduce.getIdentities();
  ValueRange captures = reduce.getCaptures();
  if (!llvm::all_of(captures, [](Value capture) {
        return isa<IntegerType, FloatType, IndexType>(capture.getType());
      }) ||
      !llvm::all_of(llvm::enumerate(accumulatorTypes), [&](auto component) {
        FragmentType type = component.value();
        Type element = type.getElementType();
        auto identity = scalarSource(identities[component.index()]);
        return isa<IntegerType, FloatType, IndexType>(element) &&
               succeeded(identity) && (*identity).getType() == element &&
               dataElementType(reduce.getResult(component.index()).getType()) ==
                   element &&
               sameExecutionSchema(type, accumulatorTypes.front());
      }))
    return false;
  SmallVector<Type> types(accumulatorTypes.begin(), accumulatorTypes.end());
  std::string reason;
  return succeeded(
      liftCombineRegion(reduce.getCombine(), combine, types, reason));
}

} // namespace intent::gpu::reduction
