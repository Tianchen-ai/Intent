#include "ComputeForms.h"
#include "Intent/Dialect/Block/Analysis/Helpers.h"
#include "Intent/Dialect/Block/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/Block/Analysis/ValueSchema.h"
#include "Intent/Dialect/Block/IR/Program.h"
#include "Intent/Target/CuTile/IR/CuTileOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;

namespace intent::cutile {
namespace {

Value transposeMatrix(OpBuilder &builder, Location location, Value value) {
  auto type = cast<block::FragmentType>(value.getType());
  unsigned rank = type.getShape().size();
  SmallVector<int64_t> permutation;
  SmallVector<Attribute> shape, maps;
  for (unsigned axis = 0; axis < rank; ++axis) {
    unsigned source = axis < rank - 2 ? axis : 2 * rank - 3 - axis;
    permutation.push_back(source);
    shape.push_back(type.getShape()[source]);
    auto mapping = cast<block::AxisMapAttr>(type.getAxisMaps()[source]);
    maps.push_back(block::AxisMapAttr::get(builder.getContext(),
        mapping.getSourceId(), mapping.getSourceAxis(), mapping.getDimensionId(),
        axis, mapping.getDerived()));
  }
  auto result = block::FragmentType::get(builder.getContext(), type.getElementType(),
      builder.getArrayAttr(shape), builder.getArrayAttr(maps),
      type.getValidity(), type.getOwner());
  return builder.create<block::TransposeOp>(location, result, value, permutation);
}

} // namespace

LogicalResult formMMA(func::FuncOp kernel, block::ContractOp contract,
                      NativeFormRewriter &rewriter) {
  std::string reason;
  auto axes = block::queryContractionAxes(contract, &reason);
  if (!axes)
    return contract.emitOpError("invalid cuTile contraction axis schema: ") << reason;
  if (!axes->hasCanonicalMatrixAxes() || axes->lhsResultAxes.size() > 3)
    return contract.emitOpError(
        "cuTile native MMA requires [M,K] x [K,N] or [B,M,K] x [B,K,N] physical axes");
  OpBuilder builder(contract);
  Location location = contract.getLoc();
  auto product = [&](Value lhs, Value rhs, Value accumulator) -> Value {
    auto mma = builder.create<MMAOp>(location, accumulator.getType(), lhs, rhs,
                                     accumulator, IntegerAttr());
    if (Attribute origin = contract->getAttr(block::originAttr))
      mma->setAttr(block::originAttr, origin);
    return mma;
  };
  auto resultType = cast<block::FragmentType>(contract.getResult().getType());
  auto lhsType = cast<block::FragmentType>(contract.getLhs().getType());
  auto rhsType = cast<block::FragmentType>(contract.getRhs().getType());
  unsigned rank = resultType.getShape().size();
  auto m = cast<block::PhysicalExprAttr>(resultType.getShape()[rank - 2]);
  auto n = cast<block::PhysicalExprAttr>(resultType.getShape()[rank - 1]);
  auto bounds = block::queryPositiveExtentBounds(m, kernel);
  Type element = lhsType.getElementType();
  bool narrow = bounds && bounds->second < 16 &&
      (element.isF16() || element.isBF16()) &&
      rhsType.getElementType() == element && resultType.getElementType().isF32();
  Value result;
  if (narrow) {
    // Native tensor-core M and N tile minima are asymmetric. Put a narrow
    // free axis in N through (A B + C)^T = B^T A^T + C^T. This changes neither
    // ownership nor reduction membership/precision. cuTile still selects the
    // instruction, layout and pipeline; no memory transpose is introduced.
    // Deferred extents are provider compile-time values, so the choice folds
    // for each native specialization rather than branching at execution.
    Value columns = builder.create<block::PhysicalExprOp>(location, builder.getIndexType(), n);
    Value minimum = builder.create<arith::ConstantIndexOp>(location, 32);
    Value selected = builder.create<block::CompareOp>(location, builder.getI1Type(),
        columns, minimum, ComparePredicate::Ge);
    auto choice = builder.create<scf::IfOp>(location, TypeRange{resultType}, selected, true);
    builder.setInsertionPointToStart(choice.thenBlock());
    Value lhs = transposeMatrix(builder, location, contract.getRhs());
    Value rhs = transposeMatrix(builder, location, contract.getLhs());
    Value accumulator = transposeMatrix(builder, location, contract.getAccumulator());
    auto original = cast<block::FragmentType>(accumulator.getType());
    // cuTile otherwise keeps N < 8 on scalar multiply-adds. Duplicate free
    // coordinates to fill the native tile, then retain only the original
    // result. No K padding, memory access or new reduction term is introduced.
    auto eight = block::PhysicalExprAttr::get(builder.getContext(),
        block::PhysicalExprKind::Constant, 8, builder.getStringAttr(""),
        builder.getArrayAttr({}));
    rhs = repeatTileAxis(builder, location, rhs, rank - 1, eight);
    accumulator = repeatTileAxis(builder, location, accumulator, rank - 1, eight);
    Value transposed = product(lhs, rhs, accumulator);
    Value zero = builder.create<arith::ConstantIntOp>(location, 0, 32);
    SmallVector<Value> coordinates(rank, zero);
    SmallVector<int64_t> retained;
    for (unsigned axis = 0; axis < rank; ++axis) retained.push_back(axis);
    transposed = builder.create<ExtractOp>(location, original, transposed,
        coordinates, original.getShape(), retained);
    builder.create<scf::YieldOp>(location, transposeMatrix(builder, location, transposed));
    builder.setInsertionPointToStart(choice.elseBlock());
    builder.create<scf::YieldOp>(location,
        product(contract.getLhs(), contract.getRhs(), contract.getAccumulator()));
    result = choice.getResult(0);
  } else {
    result = product(contract.getLhs(), contract.getRhs(), contract.getAccumulator());
  }
  rewriter.replace(contract, ValueRange{result});
  return success();
}

} // namespace intent::cutile
