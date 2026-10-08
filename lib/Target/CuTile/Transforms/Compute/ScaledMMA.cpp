#include "ComputeForms.h"
#include "Intent/Dialect/GPU/Analysis/Resources.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Target/CuTile/Analysis/Program.h"
#include "Intent/Target/CuTile/IR/CuTileOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/APFloat.h"

using namespace mlir;
namespace intent::cutile {
namespace {

gpu::FragmentType withElement(gpu::FragmentType type, Type element) {
  return gpu::FragmentType::get(type.getContext(), element, type.getShape(),
      type.getAxisMaps(), type.getValidity(), type.getOwner());
}

Value e4m3Carrier(OpBuilder &builder, Location location, Value value) {
  auto type = cast<gpu::FragmentType>(value.getType());
  if (isa<Float8E4M3FNType>(type.getElementType())) return value;
  return builder.create<gpu::BitcastOp>(location,
      withElement(type, Float8E4M3FNType::get(builder.getContext())), value);
}

Value constantLike(OpBuilder &builder, Location location,
                   gpu::FragmentType type, TypedAttr value) {
  Value scalar = builder.create<arith::ConstantOp>(location,
      type.getElementType(), value);
  return builder.create<gpu::SplatOp>(location, type, scalar);
}

Value integerLike(OpBuilder &builder, Location location,
                  gpu::FragmentType type, int64_t value) {
  return constantLike(builder, location, type,
      builder.getIntegerAttr(type.getElementType(), value));
}

Value compare(OpBuilder &builder, Location location, Value lhs, Value rhs,
              ComparePredicate predicate) {
  auto type = withElement(cast<gpu::FragmentType>(lhs.getType()), builder.getI1Type());
  return builder.create<gpu::CompareOp>(location, type, lhs, rhs, predicate);
}

Value binary(OpBuilder &builder, Location location, Value lhs, Value rhs,
             BinaryOperator kind) {
  return builder.create<gpu::BinaryOp>(location, lhs.getType(), lhs, rhs, kind);
}

// E8M0 has no zero code: code 0 is 2^-127, whereas the corresponding IEEE
// exponent field encodes zero. Code 255 is NaN, not infinity.
Value decodeScale(OpBuilder &builder, Location location, Value scale) {
  // The scale interpretation is of unsigned carrier bits, independently of
  // any integer arithmetic signedness on the incoming physical value.
  auto raw = withElement(cast<gpu::FragmentType>(scale.getType()),
      IntegerType::get(builder.getContext(), 8, IntegerType::Unsigned));
  scale = builder.create<gpu::BitcastOp>(location, raw, scale);
  auto integer = withElement(raw, builder.getI32Type());
  Value code = builder.create<gpu::CastOp>(location, integer, scale);
  Value bits = binary(builder, location, code,
      integerLike(builder, location, integer, 23), BinaryOperator::LeftShift);
  auto floating = withElement(integer, builder.getF32Type());
  Value decoded = builder.create<gpu::BitcastOp>(location, floating, bits);
  Value zeroCode = compare(builder, location, code,
      integerLike(builder, location, integer, 0), ComparePredicate::Eq);
  Value smallest = constantLike(builder, location, floating,
      builder.getF32FloatAttr(0x1p-127f));
  decoded = builder.create<gpu::SelectOp>(location, floating,
      zeroCode, smallest, decoded);
  Value nanCode = compare(builder, location, code,
      integerLike(builder, location, integer, 255), ComparePredicate::Eq);
  Value nan = constantLike(builder, location, floating,
      FloatAttr::get(builder.getF32Type(),
          llvm::APFloat::getNaN(llvm::APFloat::IEEEsingle())));
  return builder.create<gpu::SelectOp>(location, floating,
      nanCode, nan, decoded);
}

Value scaleOperand(OpBuilder &builder, Location location, Value operand,
                   Value scale) {
  auto floating = withElement(cast<gpu::FragmentType>(operand.getType()), builder.getF32Type());
  Value decoded = builder.create<gpu::CastOp>(location, floating, operand);
  Value factors = builder.create<gpu::BroadcastOp>(location, floating,
      decodeScale(builder, location, scale));
  return binary(builder, location, decoded, factors, BinaryOperator::Multiply);
}

gpu::AxisMapAttr moveAxis(gpu::AxisMapAttr source, unsigned axis) {
  return gpu::AxisMapAttr::get(source.getContext(), source.getSourceId(),
      source.getSourceAxis(), source.getDimensionId(), axis, source.getDerived());
}

Value matrixOperand(OpBuilder &builder, Location location, Value value,
                    gpu::PhysicalExprAttr reduction,
                    gpu::AxisMapAttr reductionMap, bool lhs) {
  auto type = cast<gpu::FragmentType>(value.getType());
  unsigned retained = lhs ? 0 : 2;
  auto retainedMap = moveAxis(cast<gpu::AxisMapAttr>(type.getAxisMaps()[retained]),
      lhs ? 0 : 1);
  auto reductionAxis = moveAxis(reductionMap, lhs ? 1 : 0);
  SmallVector<Attribute> shape = lhs
      ? SmallVector<Attribute>{type.getShape()[0], reduction}
      : SmallVector<Attribute>{reduction, type.getShape()[2]};
  SmallVector<Attribute> mappings = lhs
      ? SmallVector<Attribute>{retainedMap, reductionAxis}
      : SmallVector<Attribute>{reductionAxis, retainedMap};
  auto result = gpu::FragmentType::get(type.getContext(), type.getElementType(),
      builder.getArrayAttr(shape), builder.getArrayAttr(mappings),
      type.getValidity(), type.getOwner());
  auto group = [&](ArrayRef<int64_t> from, ArrayRef<int64_t> to) {
    return gpu::ReshapeGroupAttr::get(type.getContext(),
        builder.getDenseI64ArrayAttr(from), builder.getDenseI64ArrayAttr(to));
  };
  auto groups = lhs ? builder.getArrayAttr({group({0}, {0}), group({1, 2}, {1})})
                    : builder.getArrayAttr({group({0, 1}, {0}), group({2}, {1})});
  return builder.create<gpu::ReshapeOp>(location, result, value, groups);
}

// E4M3's nonzero magnitudes lie in [2^-9, 448] with at most four significant
// bits. For E8M0 codes 10..246 the scaled values are normal and finite in BF16
// and F32, and therefore exactly representable in either. Check the compact
// scale domain before constructing either computation's live matrix payloads.
// Other codes retain the ordinary F32 MMA; no source-domain restriction is added.
Value normalBF16Scales(OpBuilder &builder, Location location,
                       func::FuncOp kernel, Value scale) {
  auto integer = withElement(cast<gpu::FragmentType>(scale.getType()), builder.getI32Type());
  Value code = builder.create<gpu::CastOp>(location, integer, scale);
  Value lower = compare(builder, location, code,
      integerLike(builder, location, integer, 10), ComparePredicate::Ge);
  Value upper = compare(builder, location, code,
      integerLike(builder, location, integer, 246), ComparePredicate::Le);
  Value valid = binary(builder, location, lower, upper, BinaryOperator::LogicalAnd);
  auto type = cast<gpu::FragmentType>(valid.getType());
  auto capacity = gpu::fragmentElementCount(type);
  auto [source, dimension] = gpu::nextPhysicalAxisIdentities(kernel);
  auto flat = gpu::FragmentType::get(builder.getContext(), builder.getI1Type(),
      builder.getArrayAttr({capacity}), builder.getArrayAttr({gpu::AxisMapAttr::get(
          builder.getContext(), source, 0, dimension, 0, true)}),
      type.getValidity(), type.getOwner());
  auto groups = builder.getArrayAttr({gpu::ReshapeGroupAttr::get(builder.getContext(),
      builder.getDenseI64ArrayAttr({0, 1}), builder.getDenseI64ArrayAttr({0}))});
  valid = builder.create<gpu::ReshapeOp>(location, flat, valid, groups);
  auto all = builder.create<ReduceOp>(location, ValueRange{valid}, ValueRange{},
      0, false, BinaryOperatorAttr::get(builder.getContext(), BinaryOperator::LogicalAnd));
  return all.getOutputs().front();
}

} // namespace

LogicalResult formScaledMMA(func::FuncOp kernel, gpu::ScaledContractOp contract,
                           NativeFormRewriter &rewriter) {
  if (contract.getLhsReductionAxes() != ArrayRef<int64_t>{1, 2} ||
      contract.getRhsReductionAxes() != ArrayRef<int64_t>{0, 1} ||
      !contract.getLhsBatchAxes().empty() || !contract.getRhsBatchAxes().empty() ||
      contract.getLhsFormat() != ScaledFormat::E4M3 ||
      contract.getRhsFormat() != ScaledFormat::E4M3 ||
      contract.getLhsGroupSize() != 32 || contract.getRhsGroupSize() != 32 ||
      !contract.getResult().getType().getElementType().isF32())
    return contract.emitOpError(
        "cuTile scaled MMA requires E4M3/E8M0 group-32 adjacent reduction axes and f32 accumulation");
  auto capabilities = kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
  if (!capabilities)
    return contract.emitOpError("scaled MMA requires selected GPU capabilities");
  OpBuilder builder(contract);
  Location location = contract.getLoc();
  auto rhsScaleType = cast<gpu::FragmentType>(contract.getRhsScale().getType());
  SmallVector<Attribute> shape{rhsScaleType.getShape()[1], rhsScaleType.getShape()[0]};
  SmallVector<Attribute> axes{
      moveAxis(cast<gpu::AxisMapAttr>(rhsScaleType.getAxisMaps()[1]), 0),
      moveAxis(cast<gpu::AxisMapAttr>(rhsScaleType.getAxisMaps()[0]), 1)};
  auto transposed = gpu::FragmentType::get(builder.getContext(), rhsScaleType.getElementType(),
      builder.getArrayAttr(shape), builder.getArrayAttr(axes),
      rhsScaleType.getValidity(), rhsScaleType.getOwner());
  Value rhsScale = builder.create<gpu::TransposeOp>(location, transposed,
      contract.getRhsScale(), ArrayRef<int64_t>{1, 0});
  Value lhsCarrier = e4m3Carrier(builder, location, contract.getLhs());
  Value rhsCarrier = e4m3Carrier(builder, location, contract.getRhs());
  Value result;
  if (supportsE8M0ScaledMMA(capabilities)) {
    result = builder.create<ScaledMMAOp>(location, contract.getResult().getType(),
        lhsCarrier, contract.getLhsScale(), rhsCarrier, rhsScale,
        contract.getAccumulator(), contract.getLhsFormat(), contract.getRhsFormat(),
        contract.getLhsGroupSize(), contract.getRhsGroupSize());
  } else {
    auto lhsType = contract.getLhs().getType();
    auto reduction = gpu::PhysicalExprAttr::get(builder.getContext(),
        gpu::PhysicalExprKind::Multiply, 0, builder.getStringAttr(""),
        builder.getArrayAttr({lhsType.getShape()[1], lhsType.getShape()[2]}));
    auto [source, dimension] = gpu::nextPhysicalAxisIdentities(kernel);
    auto reductionMap = gpu::AxisMapAttr::get(builder.getContext(), source, 0,
        dimension, 0, true);
    Value lhsExact = normalBF16Scales(builder, location, kernel, contract.getLhsScale());
    Value rhsExact = normalBF16Scales(builder, location, kernel, rhsScale);
    Value exact = builder.create<gpu::BinaryOp>(location, builder.getI1Type(),
        lhsExact, rhsExact, BinaryOperator::LogicalAnd);
    auto choose = builder.create<scf::IfOp>(location,
        TypeRange{contract.getResult().getType()}, exact, true);
    for (bool narrow : {true, false}) {
      OpBuilder branch = narrow ? choose.getThenBodyBuilder() : choose.getElseBodyBuilder();
      Value lhs = matrixOperand(branch, location,
          scaleOperand(branch, location, lhsCarrier, contract.getLhsScale()),
          reduction, reductionMap, true);
      Value rhs = matrixOperand(branch, location,
          scaleOperand(branch, location, rhsCarrier, rhsScale),
          reduction, reductionMap, false);
      if (narrow) {
        lhs = branch.create<gpu::CastOp>(location,
            withElement(cast<gpu::FragmentType>(lhs.getType()), builder.getBF16Type()), lhs);
        rhs = branch.create<gpu::CastOp>(location,
            withElement(cast<gpu::FragmentType>(rhs.getType()), builder.getBF16Type()), rhs);
      }
      Value value = branch.create<MMAOp>(location, contract.getResult().getType(),
          lhs, rhs,
          contract.getAccumulator(), IntegerAttr());
      branch.create<scf::YieldOp>(location, value);
    }
    result = choose.getResult(0);
  }
  if (Attribute origin = contract->getAttr(gpu::originAttr))
    result.getDefiningOp()->setAttr(gpu::originAttr, origin);
  rewriter.replace(contract, ValueRange{result});
  return success();
}

} // namespace intent::cutile
