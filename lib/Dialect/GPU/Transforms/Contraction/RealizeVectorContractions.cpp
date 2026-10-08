#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/STLExtras.h"

#include <numeric>

using namespace mlir;

namespace intent::gpu {
namespace {

AxisMapAttr onAxis(AxisMapAttr source, unsigned axis) {
  return AxisMapAttr::get(source.getContext(), source.getSourceId(),
                          source.getSourceAxis(), source.getDimensionId(), axis,
                          source.getDerived());
}

FragmentType withElement(FragmentType source, Type element) {
  return FragmentType::get(source.getContext(), element, source.getShape(),
                           source.getAxisMaps(), source.getValidity(),
                           source.getOwner());
}

FailureOr<Value> projectOperand(OpBuilder &builder, ContractOp contract,
                                Value operand, ArrayRef<unsigned> targetAxes,
                                FragmentType productType) {
  auto source = cast<FragmentType>(operand.getType());
  if (source.getElementType() != productType.getElementType()) {
    source = withElement(source, productType.getElementType());
    operand = builder.create<CastOp>(contract.getLoc(), source, operand);
  }
  SmallVector<int64_t> permutation(source.getShape().size());
  std::iota(permutation.begin(), permutation.end(), 0);
  llvm::sort(permutation, [&](int64_t left, int64_t right) {
    return targetAxes[left] < targetAxes[right];
  });
  if (!llvm::is_sorted(targetAxes)) {
    SmallVector<Attribute> shape;
    SmallVector<Attribute> mappings;
    for (auto [axis, oldAxis] : llvm::enumerate(permutation)) {
      shape.push_back(source.getShape()[oldAxis]);
      mappings.push_back(onAxis(cast<AxisMapAttr>(source.getAxisMaps()[oldAxis]), axis));
    }
    auto transposed = FragmentType::get(
        contract.getContext(), source.getElementType(), builder.getArrayAttr(shape),
        builder.getArrayAttr(mappings), source.getValidity(), source.getOwner());
    operand = builder.create<TransposeOp>(contract.getLoc(), transposed, operand, permutation);
    source = transposed;
  }
  SmallVector<Attribute> shape(productType.getShape().size(),
      PhysicalExprAttr::get(contract.getContext(),
          PhysicalExprKind::Constant, 1,
          builder.getStringAttr(""), builder.getArrayAttr({})));
  SmallVector<std::optional<unsigned>> sourceAxes(shape.size());
  for (auto [axis, originalAxis] : llvm::enumerate(permutation)) {
    unsigned targetAxis = targetAxes[originalAxis];
    shape[targetAxis] = source.getShape()[axis];
    sourceAxes[targetAxis] = axis;
  }
  SmallVector<Attribute> groups;
  for (auto [axis, sourceAxis] : llvm::enumerate(sourceAxes)) {
    SmallVector<int64_t> inputs;
    if (sourceAxis)
      inputs.push_back(*sourceAxis);
    groups.push_back(ReshapeGroupAttr::get(
        contract.getContext(), builder.getDenseI64ArrayAttr(inputs),
        builder.getDenseI64ArrayAttr({static_cast<int64_t>(axis)})));
  }
  auto expanded = FragmentType::get(
      contract.getContext(), source.getElementType(), builder.getArrayAttr(shape),
      productType.getAxisMaps(), source.getValidity(), source.getOwner());
  if (expanded != source)
    operand = builder.create<ReshapeOp>(contract.getLoc(), expanded, operand,
                                        builder.getArrayAttr(groups));
  return projectPhysicalValueToSchema(builder, contract.getLoc(), operand, productType);
}

LogicalResult realizeVectorContract(ContractOp contract) {
  auto lhs = contract.getLhs().getType();
  auto rhs = contract.getRhs().getType();
  std::string reason;
  auto axes = queryContractionAxes(contract, &reason);
  if (!axes)
    return contract.emitOpError("invalid vector-contraction axis schema: ") << reason;
  if (!axes->lhsFree.empty() && !axes->rhsFree.empty())
    return success();

  OpBuilder builder(contract);
  Location location = contract.getLoc();
  auto resultType = contract.getResult().getType();
  SmallVector<Attribute> shape(resultType.getShape().begin(), resultType.getShape().end());
  SmallVector<Attribute> mappings(resultType.getAxisMaps().begin(), resultType.getAxisMaps().end());
  SmallVector<unsigned> lhsAxes(lhs.getShape().size());
  SmallVector<unsigned> rhsAxes(rhs.getShape().size());
  for (auto [axis, result] : llvm::enumerate(axes->lhsResultAxes))
    if (result) lhsAxes[axis] = *result;
  for (auto [axis, result] : llvm::enumerate(axes->rhsResultAxes))
    if (result) rhsAxes[axis] = *result;
  unsigned resultAxis = axes->results.size();
  SmallVector<int64_t> reductionAxes;
  for (const auto &[left, right] : axes->reduction) {
    lhsAxes[left] = rhsAxes[right] = resultAxis;
    reductionAxes.push_back(resultAxis);
    shape.push_back(lhs.getShape()[left]);
    mappings.push_back(onAxis(cast<AxisMapAttr>(lhs.getAxisMaps()[left]), resultAxis++));
  }
  // A vector contraction need not transpose its matrix operand merely to put
  // the reduction last. Preserve that operand's axes when removing the paired
  // axis still produces the declared result order.
  ArrayRef<unsigned> order = !axes->lhsFree.empty() ? ArrayRef<unsigned>(lhsAxes)
                                                  : ArrayRef<unsigned>(rhsAxes);
  unsigned freeAxis = 0;
  bool preservesResult = reductionAxes.size() == 1 && order.size() == shape.size();
  for (unsigned axis : order)
    if (!llvm::is_contained(reductionAxes, axis))
      preservesResult &= axis == freeAxis++;
  if (preservesResult && freeAxis == resultType.getShape().size()) {
    SmallVector<unsigned> inverse(order.size());
    SmallVector<Attribute> orderedShape, orderedMappings;
    for (auto [axis, original] : llvm::enumerate(order)) {
      inverse[original] = axis;
      orderedShape.push_back(shape[original]);
      orderedMappings.push_back(onAxis(cast<AxisMapAttr>(mappings[original]), axis));
    }
    for (unsigned &axis : lhsAxes)
      axis = inverse[axis];
    for (unsigned &axis : rhsAxes)
      axis = inverse[axis];
    reductionAxes.front() = inverse[reductionAxes.front()];
    shape = std::move(orderedShape);
    mappings = std::move(orderedMappings);
  }
  auto productType = FragmentType::get(
      contract.getContext(), resultType.getElementType(), builder.getArrayAttr(shape),
      builder.getArrayAttr(mappings), resultType.getValidity(), resultType.getOwner());
  FailureOr<Value> left = projectOperand(builder, contract, contract.getLhs(), lhsAxes, productType);
  FailureOr<Value> right = projectOperand(builder, contract, contract.getRhs(), rhsAxes, productType);
  if (failed(left) || failed(right))
    return contract.emitOpError("cannot project paired vector-contraction axes to a reduction fragment");
  auto product = builder.create<BinaryOp>(location, productType, *left, *right, BinaryOperator::Multiply);
  FailureOr<Value> zero = materializeZeroFragment(builder, location, resultType);
  if (failed(zero))
    return contract.emitOpError("cannot form vector-contraction additive identity");
  Value identity = *zero;
  if (resultType.getShape().empty())
    identity = identity.getDefiningOp<BroadcastOp>().getValue();
  auto reduction = builder.create<ReduceOp>(location,
      ValueRange{product.getResult()}, ValueRange{identity}, ValueRange{}, reductionAxes);
  Type reducedType = reduction.getResult(0).getType();
  Block *combine = builder.createBlock(&reduction.getCombine(), {},
                                      {reducedType, reducedType}, {location, location});
  Value combined = builder.create<BinaryOp>(location, reducedType,
      combine->getArgument(0), combine->getArgument(1), BinaryOperator::Add);
  builder.create<YieldOp>(location, combined);
  builder.setInsertionPoint(contract);
  Value result = reduction.getResult(0);
  if (resultType.getShape().empty())
    result = builder.create<SplatOp>(location, resultType, result);
  auto accumulated = builder.create<BinaryOp>(location, resultType, result,
                                               contract.getAccumulator(), BinaryOperator::Add);
  if (Attribute origin = contract->getAttr(originAttr)) {
    product->setAttr(originAttr, origin);
    reduction->setAttr(originAttr, origin);
    accumulated->setAttr(originAttr, origin);
  }
  contract.getResult().replaceAllUsesWith(accumulated.getResult());
  contract.erase();
  return success();
}

} // namespace

LogicalResult realizeVectorContractions(ModuleOp module) {
  SmallVector<ContractOp> contracts;
  module.walk([&](ContractOp contract) { contracts.push_back(contract); });
  for (ContractOp contract : contracts)
    if (failed(realizeVectorContract(contract)))
      return failure();
  return success();
}

} // namespace intent::gpu
