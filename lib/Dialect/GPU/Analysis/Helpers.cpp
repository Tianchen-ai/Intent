#include "Intent/Dialect/GPU/Analysis/Helpers.h"
#include "Intent/Dialect/GPU/IR/FragmentOpInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::gpu {
namespace {

bool unit(Attribute attribute) {
  auto extent = dyn_cast<PhysicalExprAttr>(attribute);
  return extent && extent.getKind() == PhysicalExprKind::Constant &&
         extent.getValue() == 1;
}

bool laneType(Type type) {
  if (auto fragment = dyn_cast<FragmentType>(type))
    return laneType(fragment.getElementType());
  if (auto record = dyn_cast<RecordType>(type))
    return llvm::all_of(record.getFieldTypes(), [](Attribute field) {
      return laneType(cast<TypeAttr>(field).getValue());
    });
  return type.isIntOrIndexOrFloat();
}

bool orderedProjection(const FragmentOperandRelation &relation) {
  auto input = dyn_cast<FragmentType>(relation.sourceType);
  auto output = dyn_cast<FragmentType>(relation.resultType);
  if (!input || !output || !relation.invariantResultAxes.empty()) return false;
  SmallVector<unsigned> before, after, expectedBefore, expectedAfter;
  for (unsigned axis = 0; axis < input.getShape().size(); ++axis)
    if (!unit(input.getShape()[axis])) expectedBefore.push_back(axis);
  for (unsigned axis = 0; axis < output.getShape().size(); ++axis)
    if (!unit(output.getShape()[axis])) expectedAfter.push_back(axis);
  for (const FragmentAxisGroup &group : relation.groups) {
    SmallVector<Attribute> sourceExtents, resultExtents;
    for (unsigned axis : group.sourceAxes) {
      sourceExtents.push_back(input.getShape()[axis]);
      if (!unit(input.getShape()[axis])) before.push_back(axis);
    }
    for (unsigned axis : group.resultAxes) {
      resultExtents.push_back(output.getShape()[axis]);
      if (!unit(output.getShape()[axis])) after.push_back(axis);
    }
    // A missing source may only introduce units. This excludes a broadcast
    // from a single retained member to a genuinely larger lane domain.
    if (!haveEqualPhysicalElementCounts(sourceExtents, resultExtents))
      return false;
  }
  return before == expectedBefore && after == expectedAfter;
}

} // namespace

bool isLaneWisePointwiseOperation(Operation *operation) {
  return operation && operation->hasTrait<OpTrait::Elementwise>() &&
         operation->hasTrait<OpTrait::Scalarizable>() &&
         isa<FragmentOpInterface>(operation) &&
         operation->getNumRegions() == 0 && isMemoryEffectFree(operation);
}

Value laneWiseProjectionSource(Operation *operation) {
  if (!operation || operation->getNumRegions() ||
      operation->getNumOperands() != 1 || operation->getNumResults() != 1 ||
      !isMemoryEffectFree(operation) ||
      !isa<SplatOp, BroadcastOp, ReshapeOp, TransposeOp>(operation))
    return {};
  Value source = operation->getOperand(0);
  if (source.getType().isIntOrIndexOrFloat())
    return isa<SplatOp, BroadcastOp>(operation) ? source : Value{};
  auto relations = queryFragmentOperandRelations(operation);
  return succeeded(relations) && relations->size() == 1 &&
                 relations->front().operandNumber == 0 &&
                 orderedProjection(relations->front())
             ? source : Value{};
}

bool isLaneWiseValueOperation(Operation *operation) {
  if (!operation || operation->getNumRegions() ||
      !isMemoryEffectFree(operation) ||
      !llvm::all_of(operation->getOperandTypes(), laneType) ||
      !llvm::all_of(operation->getResultTypes(), laneType)) return false;
  return isa<arith::ConstantOp, MakeRecordOp, ExtractOp>(operation) ||
         isLaneWisePointwiseOperation(operation) ||
         bool(laneWiseProjectionSource(operation));
}

LogicalResult proveLaneWiseHelper(Region &region, std::string *reason) {
  auto reject = [&](const Twine &message) {
    if (reason) *reason = message.str();
    return failure();
  };
  if (!llvm::hasSingleElement(region) || region.front().empty())
    return reject("lane-wise helper requires one complete block");
  Block &body = region.front();
  auto yield = dyn_cast<YieldOp>(&body.back());
  if (!yield || !llvm::all_of(body.getArgumentTypes(), laneType) ||
      !llvm::all_of(yield.getOperandTypes(), laneType))
    return reject("lane-wise helper requires typed scalar/fragment/product inputs and yields");
  for (Operation &operation : body) {
    for (Value operand : operation.getOperands())
      if (operand.getParentBlock() != &body)
        return reject("lane-wise helper cannot implicitly capture enclosing values");
    if (&operation != yield.getOperation() && !isLaneWiseValueOperation(&operation))
      return reject("helper operation has no lane-preserving scalarization: " +
                    operation.getName().getStringRef());
  }
  return success();
}

static std::optional<BinaryCombine> queryBinaryComponent(
    Region &region, unsigned component, unsigned left, unsigned right) {
  if (failed(proveLaneWiseHelper(region))) return std::nullopt;
  Block &body = region.front();
  auto yield = cast<YieldOp>(body.getTerminator());
  if (left >= body.getNumArguments() || right >= body.getNumArguments() ||
      component >= yield.getValues().size())
    return std::nullopt;
  auto strip = [](Value value) {
    while (Value source = laneWiseProjectionSource(value.getDefiningOp()))
      value = source;
    return value;
  };
  auto binary = strip(yield.getValues()[component]).getDefiningOp<BinaryOp>();
  if (!binary) return std::nullopt;
  Value lhs = strip(binary.getLhs()), rhs = strip(binary.getRhs());
  if (lhs == body.getArgument(left) && rhs == body.getArgument(right))
    return BinaryCombine{binary, {0, 1}};
  if (lhs == body.getArgument(right) && rhs == body.getArgument(left))
    return BinaryCombine{binary, {1, 0}};
  return std::nullopt;
}

std::optional<BinaryCombine> queryBinaryCombine(Region &region) {
  if (!llvm::hasSingleElement(region) || region.front().getNumArguments() != 2)
    return std::nullopt;
  auto yield = dyn_cast<YieldOp>(region.front().getTerminator());
  if (!yield || yield.getValues().size() != 1) return std::nullopt;
  return queryBinaryComponent(region, 0, 0, 1);
}

std::optional<BinaryReduction> queryBinaryReduction(Value value) {
  auto result = dyn_cast<OpResult>(value);
  auto reduce = result ? dyn_cast<ReduceOp>(result.getOwner()) : ReduceOp{};
  if (!reduce || reduce.getSources().size() != reduce.getNumResults() ||
      reduce.getIdentities().size() != reduce.getNumResults()) return std::nullopt;
  unsigned component = result.getResultNumber();
  auto combine = queryBinaryComponent(reduce.getCombine(), component,
                                      component, reduce.getNumResults() + component);
  if (!combine) return std::nullopt;
  return BinaryReduction{result, *combine};
}

} // namespace intent::gpu
