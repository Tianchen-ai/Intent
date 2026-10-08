#include "Construction.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/ProgramInterface.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <limits>

using namespace mlir;

namespace intent::kir_to_gpu {
namespace {

// Quantized records are ordinary byte fragments. This builder expands only
// the closed record primitives; launch ownership and memory realization remain
// the normal GPU passes' responsibility.
class RecordBuilder {
public:
  RecordBuilder(OpBuilder &builder, Operation *source, func::FuncOp kernel)
      : builder(builder), location(source->getLoc()), context(source->getContext()) {
    auto identities = gpu::nextPhysicalAxisIdentities(kernel);
    dimension = identities.second;
    auto identity = resultAxisIdentity(source, 0, 0);
    sourceId = succeeded(identity) ? identity->sourceId : identities.first;
    auto observe = [&](Type type) {
      if (auto view = dyn_cast<intent::ViewType>(type)) type = view.getTensor();
      if (auto buffer = dyn_cast<intent::BufferType>(type)) type = buffer.getTensor();
      if (auto tensor = dyn_cast<RankedTensorType>(type))
        if (auto shape = dyn_cast_or_null<TensorShapeAttr>(tensor.getEncoding()))
          for (int64_t id : shape.getDimensions().asArrayRef()) dimension = std::max(dimension, id + 1);
    };
    // Construction has not visited every canonical value yet. Reserve synthetic
    // record subaxes after *all* canonical dimensions, not just current GPU IR.
    source->getParentOfType<func::FuncOp>().walk([&](Operation *operation) {
      for (Type type : operation->getOperandTypes()) observe(type);
      for (Type type : operation->getResultTypes()) observe(type);
    });
  }

  Type element(Type type) const {
    if (auto fragment = dyn_cast<gpu::FragmentType>(type)) return fragment.getElementType();
    return type;
  }
  Type element(Value value) const { return element(value.getType()); }
  Type withElement(Type type, Type scalar) const {
    if (auto fragment = dyn_cast<gpu::FragmentType>(type))
      return gpu::FragmentType::get(context, scalar, fragment.getShape(),
          fragment.getAxisMaps(), fragment.getValidity(), fragment.getOwner());
    return scalar;
  }
  Value integer(int64_t value, Type type = {}) {
    if (!type) type = builder.getIndexType();
    return *gpu::materializeScalarConstant(
        builder, location, builder.getIntegerAttr(type, value), type);
  }
  Value floating(double value) {
    return builder.create<arith::ConstantOp>(location, builder.getF32FloatAttr(value));
  }
  Value project(Value value, Type type) {
    if (value.getType() == type) return value;
    auto projected = gpu::projectPhysicalValueToSchema(builder, location, value, type);
    if (succeeded(projected)) return *projected;
    emitError(location, "quantized record value has no exact physical projection")
        << ": " << value.getType() << " to " << type;
    valid = false;
    return value;
  }
  Value castTo(Value value, Type scalar) {
    Type type = withElement(value.getType(), scalar);
    if (type == value.getType()) return value;
    return builder.create<gpu::CastOp>(location, type, value);
  }
  Value bits(Value value, Type scalar) {
    return builder.create<gpu::BitcastOp>(location, withElement(value.getType(), scalar), value);
  }
  Value binary(Value lhs, Value rhs, BinaryOperator kind, bool strict = false) {
    Type type = lhs.getType();
    if (auto right = dyn_cast<gpu::FragmentType>(rhs.getType())) {
      auto left = dyn_cast<gpu::FragmentType>(type);
      if (!left || right.getShape().size() > left.getShape().size()) type = rhs.getType();
    }
    lhs = project(lhs, type);
    rhs = project(rhs, type);
    return builder.create<gpu::BinaryOp>(location, type, lhs, rhs, kind,
                                        false, false, strict);
  }
  Value binary(Value lhs, int64_t rhs, BinaryOperator kind) {
    return binary(lhs, integer(rhs, element(lhs)), kind);
  }
  Value compare(Value lhs, Value rhs, ComparePredicate predicate) {
    Type type = lhs.getType();
    if (!isa<gpu::FragmentType>(type) && isa<gpu::FragmentType>(rhs.getType())) type = rhs.getType();
    return builder.create<gpu::CompareOp>(location, withElement(type, builder.getI1Type()),
        project(lhs, type), project(rhs, type), predicate);
  }
  Value select(Value condition, Value yes, Value no) {
    Type type = yes.getType();
    if (auto predicate = dyn_cast<gpu::FragmentType>(condition.getType()))
      type = withElement(predicate, element(yes));
    return builder.create<gpu::SelectOp>(location, type, condition,
                                        project(yes, type), project(no, type));
  }
  Value unary(Value input, UnaryOperator kind) {
    return builder.create<gpu::UnaryOp>(location, input.getType(), input, kind);
  }
  gpu::AxisMapAttr axis(unsigned position) {
    return gpu::AxisMapAttr::get(context, sourceId, syntheticAxis++, dimension++, position, true);
  }
  gpu::FragmentType prefix(gpu::FragmentType source, unsigned count, Type scalar) {
    SmallVector<Attribute> shape(source.getShape().begin(), source.getShape().begin() + count);
    SmallVector<Attribute> maps(source.getAxisMaps().begin(), source.getAxisMaps().begin() + count);
    return gpu::FragmentType::get(context, scalar, builder.getArrayAttr(shape),
        builder.getArrayAttr(maps), source.getValidity(), source.getOwner());
  }
  gpu::FragmentType append(gpu::FragmentType prefix, gpu::FragmentType suffix, Type scalar) {
    SmallVector<Attribute> shape(prefix.getShape().begin(), prefix.getShape().end());
    SmallVector<Attribute> maps(prefix.getAxisMaps().begin(), prefix.getAxisMaps().end());
    for (auto [extent, attribute] : llvm::zip(suffix.getShape(), suffix.getAxisMaps())) {
      auto mapping = mlir::cast<gpu::AxisMapAttr>(attribute);
      maps.push_back(gpu::AxisMapAttr::get(context, mapping.getSourceId(), mapping.getSourceAxis(),
          mapping.getDimensionId(), shape.size(), mapping.getDerived()));
      shape.push_back(extent);
    }
    return gpu::FragmentType::get(context, scalar, builder.getArrayAttr(shape),
        builder.getArrayAttr(maps), prefix.getValidity(), prefix.getOwner());
  }
  Value range(int64_t width, gpu::AxisMapAttr mapping = {}, int64_t stop = -1) {
    if (!mapping) mapping = axis(0);
    mapping = gpu::AxisMapAttr::get(context, mapping.getSourceId(), mapping.getSourceAxis(),
        mapping.getDimensionId(), 0, mapping.getDerived());
    auto type = gpu::FragmentType::get(context, builder.getIndexType(),
        builder.getArrayAttr({expression(context, PhysicalExprKind::Constant, width)}),
        builder.getArrayAttr({mapping}), 1, 1);
    return builder.create<gpu::MakeRangeOp>(location, type, integer(0), integer(width),
        integer(1), integer(0), integer(stop < 0 ? width : stop),
        mapping.getSourceId(), mapping.getSourceAxis(), mapping.getDerived());
  }
  Value gather(Value source, Value index, unsigned axis, Value group = {}) {
    auto type = mlir::cast<gpu::FragmentType>(source.getType());
    unsigned freeCount = group ? type.getShape().size() - 2 : type.getShape().size() - 1;
    auto free = prefix(type, freeCount, type.getElementType());
    Type result = freeCount ? Type(free) : type.getElementType();
    if (auto indices = dyn_cast<gpu::FragmentType>(index.getType())) result = append(free, indices, type.getElementType());
    SmallVector<Value> coordinates;
    SmallVector<int64_t> axes;
    if (group) { coordinates.push_back(group); axes.push_back(axis - 1); }
    coordinates.push_back(index);
    axes.push_back(axis);
    return builder.create<gpu::GatherOp>(location, result, source, coordinates,
                                         Value(), Value(), axes);
  }
  Value reduce(Value input, BinaryOperator kind, double identity = 0) {
    auto source = mlir::cast<gpu::FragmentType>(input.getType());
    unsigned selected = source.getShape().size() - 1;
    Type result = selected ? Type(prefix(source, selected, source.getElementType()))
                           : source.getElementType();
    Value seed = isa<FloatType>(source.getElementType())
        ? floating(identity) : integer(static_cast<int64_t>(identity), source.getElementType());
    seed = project(seed, result);
    auto reduction = builder.create<gpu::ReduceOp>(location, ValueRange{input},
        ValueRange{seed}, ValueRange{}, ArrayRef<int64_t>{static_cast<int64_t>(selected)});
    auto structured = mlir::cast<StructuredOpInterface>(reduction.getOperation());
    OpBuilder::InsertionGuard guard(builder);
    auto types = structured.getCombineArgumentTypes();
    Block *block = builder.createBlock(&reduction.getCombine(), reduction.getCombine().end(), types,
                                      SmallVector<Location>(types.size(), location));
    Value combined = binary(block->getArgument(0), block->getArgument(1), kind);
    builder.create<gpu::YieldOp>(location, ValueRange{combined});
    return reduction.getResult(0);
  }
  Value splitLast(Value input, int64_t outer, int64_t inner) {
    auto source = mlir::cast<gpu::FragmentType>(input.getType());
    unsigned last = source.getShape().size() - 1;
    SmallVector<Attribute> shape(source.getShape().begin(), source.getShape().end() - 1);
    SmallVector<Attribute> maps(source.getAxisMaps().begin(), source.getAxisMaps().end() - 1);
    SmallVector<Attribute> groups;
    for (unsigned i = 0; i < last; ++i)
      groups.push_back(gpu::ReshapeGroupAttr::get(context, builder.getDenseI64ArrayAttr({i}),
                                                 builder.getDenseI64ArrayAttr({i})));
    shape.push_back(expression(context, PhysicalExprKind::Constant, outer));
    shape.push_back(expression(context, PhysicalExprKind::Constant, inner));
    maps.push_back(axis(last));
    maps.push_back(axis(last + 1));
    groups.push_back(gpu::ReshapeGroupAttr::get(context, builder.getDenseI64ArrayAttr({last}),
        builder.getDenseI64ArrayAttr({last, last + 1})));
    auto result = gpu::FragmentType::get(context, source.getElementType(),
        builder.getArrayAttr(shape), builder.getArrayAttr(maps), source.getValidity(), source.getOwner());
    return builder.create<gpu::ReshapeOp>(location, result, input, builder.getArrayAttr(groups));
  }
  // RNE for the declared finite domain. Clamp before conversion: saturation
  // commutes with rounding here, and every floor/parity intermediate is exact.
  Value quantizedIntegers(Value input) {
    Value bounded = binary(binary(input, floating(-128), BinaryOperator::Maximum),
                           floating(127), BinaryOperator::Minimum);
    Value floor = unary(bounded, UnaryOperator::Floor);
    Value fraction = binary(bounded, floor, BinaryOperator::Subtract, true);
    Value integerFloor = castTo(floor, builder.getI32Type());
    Value odd = compare(binary(integerFloor, 1, BinaryOperator::BitwiseAnd),
                        integer(0, builder.getI32Type()), ComparePredicate::Ne);
    Value tie = compare(fraction, floating(0.5), ComparePredicate::Eq);
    Value greater = compare(fraction, floating(0.5), ComparePredicate::Gt);
    Value up = binary(greater, binary(tie, odd, BinaryOperator::LogicalAnd), BinaryOperator::LogicalOr);
    return binary(integerFloor, castTo(up, builder.getI32Type()), BinaryOperator::Add);
  }
  Value littleEndian(Value source, Value group, int64_t offset, unsigned bytes) {
    auto fragment = mlir::cast<gpu::FragmentType>(source.getType());
    Type u32 = IntegerType::get(context, 32, IntegerType::Unsigned);
    Value result;
    for (unsigned byte = 0; byte < bytes; ++byte) {
      Value part = castTo(gather(source, integer(offset + byte), fragment.getShape().size() - 1, group), u32);
      if (byte) part = binary(part, byte * 8, BinaryOperator::LeftShift);
      result = result ? binary(result, part, BinaryOperator::BitwiseOr) : part;
    }
    return result;
  }
  LogicalResult status() const { return success(valid); }

  OpBuilder &builder;
  Location location;
  MLIRContext *context;
  bool valid = true;
  uint64_t sourceId;
  unsigned syntheticAxis = 2;
  int64_t dimension;
};

} // namespace

LogicalResult ScalarRegionLowering::lower(intent::QuantizeOp operation) {
  auto input = get(operation.getInput());
  if (failed(input)) return operation.emitOpError("quantization input is unavailable");
  auto source = dyn_cast<gpu::FragmentType>(input->getType());
  if (!source || source.getShape().size() < 2 || !source.getElementType().isF32())
    return operation.emitOpError("quantization requires an f32 record fragment");
  auto result = convertTensorType(canonicalAnalysis,
      mlir::cast<RankedTensorType>(operation.getResult().getType()), operation);
  if (failed(result)) return operation.emitOpError("quantized bytes have no canonical axis schema");
  RecordBuilder b(builder, operation, physicalKernel);
  Value maximum = b.reduce(*input, BinaryOperator::Maximum, -std::numeric_limits<float>::infinity());
  Value minimum = b.reduce(*input, BinaryOperator::Minimum, std::numeric_limits<float>::infinity());
  Value extreme = b.select(b.compare(b.unary(maximum, UnaryOperator::Abs),
      b.unary(minimum, UnaryOperator::Abs), ComparePredicate::Gt), maximum, minimum);
  Value isZero = b.compare(extreme, b.floating(0), ComparePredicate::Eq);
  // Both operands of select may execute. Use a finite denominator on the zero
  // record, then explicitly select its prescribed positive-zero fields.
  Value safeExtreme = b.select(isZero, b.floating(-127), extreme);
  Value inverse = b.binary(b.floating(-127), safeExtreme, BinaryOperator::TrueDivide);
  Value scale = b.select(isZero, b.floating(0), b.binary(b.floating(1), inverse, BinaryOperator::TrueDivide));
  Value scaled = b.binary(*input, b.project(inverse, source), BinaryOperator::Multiply, true);
  Value q = b.quantizedIntegers(scaled);
  Value sums = b.reduce(b.splitLast(q, 16, 16), BinaryOperator::Add);
  auto byteMapping = mlir::cast<gpu::AxisMapAttr>(result->getAxisMaps().getValue().back());
  Value bytes = b.range(512, byteMapping, 292);
  auto prefix = b.prefix(source, source.getShape().size() - 1, builder.getIndexType());
  auto output = b.append(prefix, mlir::cast<gpu::FragmentType>(bytes.getType()),
                        IntegerType::get(builder.getContext(), 8, IntegerType::Unsigned));
  Value qi = b.binary(b.binary(b.binary(bytes, 4, BinaryOperator::Subtract), 0, BinaryOperator::Maximum),
                      255, BinaryOperator::Minimum);
  Value qByte = b.castTo(b.gather(q, qi, source.getShape().size() - 1), output.getElementType());
  Value sumIndex = b.binary(b.binary(b.binary(bytes, 260, BinaryOperator::Subtract), 0, BinaryOperator::Maximum),
                            31, BinaryOperator::Minimum);
  Value sum = b.gather(sums, b.binary(sumIndex, 2, BinaryOperator::FloorDivide),
                       mlir::cast<gpu::FragmentType>(sums.getType()).getShape().size() - 1);
  Value shiftedSum = b.binary(b.castTo(sum, IntegerType::get(builder.getContext(), 32, IntegerType::Unsigned)),
      b.castTo(b.binary(b.binary(sumIndex, 2, BinaryOperator::Remainder), 8, BinaryOperator::Multiply),
               IntegerType::get(builder.getContext(), 32, IntegerType::Unsigned)), BinaryOperator::RightShift);
  Value sumByte = b.castTo(shiftedSum, output.getElementType());
  Value scaleBits = b.bits(scale, IntegerType::get(builder.getContext(), 32, IntegerType::Unsigned));
  auto wordType = b.withElement(output, b.element(scaleBits));
  Value shift = b.castTo(b.binary(b.binary(bytes, 4, BinaryOperator::Remainder), 8, BinaryOperator::Multiply),
                         b.element(scaleBits));
  Value scaleByte = b.castTo(b.binary(b.project(scaleBits, wordType), b.project(shift, wordType),
                                     BinaryOperator::RightShift), output.getElementType());
  Value qOrSum = b.select(b.project(b.compare(bytes, b.integer(260), ComparePredicate::Lt),
                                   b.withElement(output, builder.getI1Type())), qByte, sumByte);
  Value packed = b.select(b.project(b.compare(bytes, b.integer(4), ComparePredicate::Lt),
                                   b.withElement(output, builder.getI1Type())), scaleByte, qOrSum);
  if (failed(b.status())) return failure();
  values[operation.getResult()] = packed;
  attachOrigin(operation, packed.getDefiningOp());
  return success();
}

LogicalResult ScalarRegionLowering::lower(intent::QuantizedDotOp operation) {
  auto lhs = get(operation.getLhs());
  auto rhs = get(operation.getRhs());
  if (failed(lhs) || failed(rhs)) return operation.emitOpError("quantized dot records are unavailable");
  auto left = dyn_cast<gpu::FragmentType>(lhs->getType());
  auto right = dyn_cast<gpu::FragmentType>(rhs->getType());
  if (!left || !right || left.getShape().size() < 2 || right.getShape().size() < 2)
    return operation.emitOpError("quantized dot requires complete record axes");
  auto groups = logicalExtent(operation.getLoc(), operation.getLhs(), 0);
  if (failed(groups)) return operation.emitOpError("quantized dot record count is unavailable");
  RecordBuilder b(builder, operation, physicalKernel);
  Type resultType = left.getShape().size() == 2 ? Type(builder.getF32Type())
      : Type(b.prefix(left, left.getShape().size() - 2, builder.getF32Type()));
  Value initial = b.project(b.floating(0), resultType);
  auto loop = builder.create<scf::ForOp>(operation.getLoc(), b.integer(0), *groups,
      b.integer(1), ValueRange{initial}, [](OpBuilder &, Location, Value, ValueRange) {});
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    Value group = loop.getInductionVar();
    Value elements = b.range(256);
    Value packedIndex = b.binary(b.binary(b.binary(elements, 64, BinaryOperator::FloorDivide),
        32, BinaryOperator::Multiply), b.binary(elements, 32, BinaryOperator::Remainder), BinaryOperator::Add);
    packedIndex = b.binary(packedIndex, 16, BinaryOperator::Add);
    Value weights = b.castTo(b.gather(*lhs, packedIndex, left.getShape().size() - 1, group), builder.getI32Type());
    Value shift = b.castTo(b.binary(b.binary(elements, 32, BinaryOperator::FloorDivide), 1,
                                   BinaryOperator::BitwiseAnd), builder.getI32Type());
    shift = b.binary(shift, 4, BinaryOperator::Multiply);
    weights = b.binary(b.binary(weights, b.project(shift, weights.getType()), BinaryOperator::RightShift),
                        15, BinaryOperator::BitwiseAnd);
    Value activation = b.gather(*rhs, b.binary(elements, 4, BinaryOperator::Add), right.getShape().size() - 1, group);
    activation = b.castTo(b.bits(activation, builder.getI8Type()), builder.getI32Type());
    activation = b.project(activation, weights.getType());
    Value products = b.binary(weights, activation, BinaryOperator::Multiply);
    Value groupProducts = b.reduce(b.splitLast(products, 8, 32), BinaryOperator::Add);
    auto groupType = mlir::cast<gpu::FragmentType>(groupProducts.getType());
    Value subgroups = b.range(8, mlir::cast<gpu::AxisMapAttr>(groupType.getAxisMaps().getValue().back()));
    auto unpackField = [&](unsigned field) {
      Value lowIndex = b.binary(b.binary(subgroups, 4, BinaryOperator::Remainder), 4 + 4 * field, BinaryOperator::Add);
      Value low = b.castTo(b.gather(*lhs, lowIndex, left.getShape().size() - 1, group), builder.getI32Type());
      Value highIndex = b.binary(b.binary(subgroups, 4, BinaryOperator::Remainder), 12, BinaryOperator::Add);
      Value high = b.castTo(b.gather(*lhs, highIndex, left.getShape().size() - 1, group), builder.getI32Type());
      high = b.binary(b.binary(high, 4 * field, BinaryOperator::RightShift), 15, BinaryOperator::BitwiseAnd);
      high = b.binary(high, b.binary(b.binary(low, 6, BinaryOperator::RightShift), 4,
                                      BinaryOperator::LeftShift), BinaryOperator::BitwiseOr);
      return b.select(b.project(b.compare(subgroups, b.integer(4), ComparePredicate::Lt),
                                  b.withElement(low.getType(), builder.getI1Type())),
                      b.binary(low, 63, BinaryOperator::BitwiseAnd), high);
    };
    Value scales = unpackField(0);
    Value minima = unpackField(1);
    Value sum = b.reduce(b.binary(groupProducts, scales, BinaryOperator::Multiply), BinaryOperator::Add);
    auto readSums = [&](unsigned part) {
      Value base = b.binary(b.binary(subgroups, 4, BinaryOperator::Multiply), 260 + 2 * part, BinaryOperator::Add);
      Value low = b.castTo(b.gather(*rhs, base, right.getShape().size() - 1, group), builder.getI32Type());
      Value high = b.castTo(b.gather(*rhs, b.binary(base, 1, BinaryOperator::Add), right.getShape().size() - 1, group), builder.getI32Type());
      Value word = b.binary(low, b.binary(high, 8, BinaryOperator::LeftShift), BinaryOperator::BitwiseOr);
      return b.castTo(b.castTo(word, builder.getI16Type()), builder.getI32Type());
    };
    Value sums = b.project(b.binary(readSums(0), readSums(1), BinaryOperator::Add), minima.getType());
    Value correction = b.reduce(b.binary(minima, sums, BinaryOperator::Multiply), BinaryOperator::Add);
    Type u16 = IntegerType::get(builder.getContext(), 16, IntegerType::Unsigned);
    Value d = b.castTo(b.bits(b.castTo(b.littleEndian(*lhs, group, 0, 2), u16), builder.getF16Type()), builder.getF32Type());
    Value dmin = b.castTo(b.bits(b.castTo(b.littleEndian(*lhs, group, 2, 2), u16), builder.getF16Type()), builder.getF32Type());
    Value ds = b.project(b.bits(b.littleEndian(*rhs, group, 0, 4), builder.getF32Type()), resultType);
    Value positive = b.binary(d, b.castTo(sum, builder.getF32Type()), BinaryOperator::Multiply, true);
    Value negative = b.binary(dmin, b.castTo(correction, builder.getF32Type()), BinaryOperator::Multiply, true);
    Value value = b.binary(ds, b.binary(positive, negative, BinaryOperator::Subtract, true), BinaryOperator::Multiply, true);
    Value next = b.binary(loop.getRegionIterArgs().front(), value, BinaryOperator::Add, true);
    builder.create<scf::YieldOp>(operation.getLoc(), ValueRange{next});
  }
  if (failed(b.status())) return failure();
  values[operation.getResult()] = loop.getResult(0);
  attachOrigin(operation, loop);
  return success();
}

} // namespace intent::kir_to_gpu
