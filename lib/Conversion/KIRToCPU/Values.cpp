#include "Construction.h"
#include "Intent/Analysis/ProductSchema.h"
#include "Intent/Conversion/LogicalShape.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;

namespace intent::kir_to_cpu {

Value Construction::constant(Location loc, int64_t value) {
  return builder.create<arith::ConstantIndexOp>(loc, value);
}

FailureOr<Value> Construction::indexValue(OpBuilder &builder, Value value,
                                         Type logicalType, Location loc) {
  if (value.getType().isIndex()) return value;
  if (isa<IntegerType>(value.getType()) && !value.getType().isInteger(1)) {
    auto integer = dyn_cast<IntegerType>(logicalType);
    if (!integer && !logicalType.isIndex() && !isa<LogicalIndexType>(logicalType))
      return emitError(loc, "CPU coordinate has no integer interpretation"), failure();
    return integer && integer.isUnsigned()
        ? Value(builder.create<arith::IndexCastUIOp>(loc, builder.getIndexType(), value))
        : Value(builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), value));
  }
  return emitError(loc, "CPU coordinates require index or integer values"), failure();
}

Value Construction::domainExtent(Value begin, Value end, Value step, Location loc) {
  if (matchPattern(begin, m_Zero()) && matchPattern(step, m_One()) && end.getDefiningOp<memref::DimOp>())
    return end;
  Value distance = builder.createOrFold<arith::SubIOp>(loc, end, begin);
  Value nonnegative = builder.createOrFold<arith::MaxSIOp>(loc, distance, constant(loc, 0));
  return builder.createOrFold<arith::CeilDivSIOp>(loc, nonnegative, step);
}

static bool availableAt(Value actual, const OpBuilder &builder) {
  if (!actual) return false;
  Block *scope = builder.getInsertionBlock();
  while (scope && scope != actual.getParentBlock()) {
    Operation *parent = scope->getParentOp();
    if (!parent || parent->hasTrait<OpTrait::IsIsolatedFromAbove>()) return false;
    scope = parent->getBlock();
  }
  if (!scope) return false;
  if (Operation *definition = actual.getDefiningOp()) {
    DominanceInfo dominance;
    if (!dominance.properlyDominates(
            definition->getBlock(), definition->getIterator(),
            builder.getInsertionBlock(), builder.getInsertionPoint(), false))
      return false;
  }
  return true;
}

Value Construction::lookupLeaf(Value value, ArrayRef<unsigned> fieldPath) const {
  Value actual;
  if (!getProductComponents(value.getType())) {
    if (!fieldPath.empty()) return {};
    actual = values.lookupOrNull(value);
  } else {
    auto found = products.find(value);
    if (found == products.end()) return {};
    auto range = getProductLeafRange(value.getType(), fieldPath);
    if (failed(range) || range->size != 1) return {};
    actual = found->second[range->offset];
  }
  return availableAt(actual, builder) ? actual : Value();
}

FailureOr<Value> Construction::extent(Value value, unsigned axis, Location loc,
                                      ArrayRef<unsigned> fieldPath) {
  LogicalShapeMaterialization materialization;
  materialization.lookup = [&](Value source, ArrayRef<unsigned> path,
                               unsigned sourceAxis) -> Value {
    Value actual = lookupLeaf(source, path);
    if (!actual) return {};
    auto type = dyn_cast<ShapedType>(actual.getType());
    if (!type || !type.hasRank() || sourceAxis >= type.getRank()) return {};
    return dimension(builder, loc, actual, sourceAxis);
  };
  materialization.leaf = [&](const TensorExtentFact &fact) -> FailureOr<Value> {
    if (fact.constant) return constant(loc, *fact.constant);
    if (fact.value) {
      Value actual = lookupLeaf(fact.value);
      if (actual) return indexValue(builder, actual, fact.value.getType(), loc);
    }
    if (fact.domain) {
      auto found = domains.find(fact.domain);
      if (found != domains.end() && availableAt(found->second.extent, builder))
        return found->second.extent;
    }
    if (fact.source)
      if (Value actual = materialization.lookup(fact.source, fact.fieldPath,
                                               fact.axis))
        return actual;
    return failure();
  };
  materialization.multiply = [&](Value lhs, Value rhs) -> FailureOr<Value> {
    return Value(builder.createOrFold<arith::MulIOp>(loc, lhs, rhs));
  };
  materialization.subtract = [&](Value lhs, Value rhs) -> FailureOr<Value> {
    return Value(builder.createOrFold<arith::SubIOp>(loc, lhs, rhs));
  };
  materialization.ceilDivide = [&](Value lhs, Value rhs) -> FailureOr<Value> {
    return Value(builder.createOrFold<arith::CeilDivSIOp>(loc, lhs, rhs));
  };
  materialization.maximum = [&](Value lhs, Value rhs) -> FailureOr<Value> {
    return Value(builder.createOrFold<arith::MaxSIOp>(loc, lhs, rhs));
  };
  materialization.exactDivide = [&](Value lhs, Value rhs) -> FailureOr<Value> {
    if (matchPattern(rhs, m_Zero()))
      return emitError(loc, "CPU inferred reshape requires a uniquely determined extent"), failure();
    return Value(builder.createOrFold<arith::DivSIOp>(loc, lhs, rhs));
  };
  auto result = materializeLogicalExtent(analysis, value, axis, materialization,
                                         fieldPath);
  if (failed(result))
    return emitError(loc, "CPU construction cannot resolve the canonical value extent"), failure();
  return result;
}

FailureOr<SmallVector<Value>> Construction::extents(
    Value value, Location loc, ArrayRef<unsigned> fieldPath) {
  auto range = getProductLeafRange(value.getType(), fieldPath);
  if (failed(range) || range->size != 1) return failure();
  SmallVector<Type> leaves;
  appendProductLeafTypes(value.getType(), leaves);
  Type type = leaves[range->offset];
  if (auto view = dyn_cast<ViewType>(type)) type = view.getTensor();
  if (auto buffer = dyn_cast<BufferType>(type)) type = buffer.getTensor();
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor) return failure();
  SmallVector<Value> result;
  for (unsigned axis = 0; axis < tensor.getRank(); ++axis) {
    auto size = extent(value, axis, loc, fieldPath);
    if (failed(size)) return failure();
    result.push_back(*size);
  }
  return result;
}

RankedTensorType Construction::tensorType(Type type) {
  auto tensor = cast<RankedTensorType>(type);
  return RankedTensorType::get(tensor.getShape(), tensor.getElementType());
}

Type Construction::valueType(Type type) {
  return isa<RankedTensorType>(type) ? Type(tensorType(type)) : type;
}

Value Construction::dimension(OpBuilder &b, Location loc, Value value, int64_t axis) {
  return isa<RankedTensorType>(value.getType())
      ? Value(b.createOrFold<tensor::DimOp>(loc, value, axis))
      : Value(b.createOrFold<memref::DimOp>(loc, value, axis));
}

Value Construction::extractElement(OpBuilder &b, Location loc, Value value,
                            ValueRange indices) {
  return isa<RankedTensorType>(value.getType())
      ? Value(b.create<tensor::ExtractOp>(loc, value, indices))
      : Value(b.create<memref::LoadOp>(loc, value, indices));
}

Value Construction::emptyTensor(RankedTensorType tensor, ArrayRef<Value> sizes, Location loc) {
  SmallVector<Value> dynamic;
  for (auto [axis, size] : llvm::enumerate(sizes))
    if (tensor.isDynamicDim(axis))
      dynamic.push_back(size);
  return builder.create<tensor::EmptyOp>(loc, tensor.getShape(),
                                         tensor.getElementType(), dynamic);
}

FailureOr<SmallVector<Value>> Construction::emptyResults(ValueRange values, Location loc) {
  SmallVector<Value> result;
  LogicalResult status = success();
  for (Value value : values) {
    walkProductLeaves(value.getType(), [&](Type leaf, ArrayRef<unsigned> path) {
      if (failed(status)) return;
      if (auto tensor = dyn_cast<RankedTensorType>(leaf)) {
        auto sizes = extents(value, loc, path);
        if (failed(sizes)) { status = failure(); return; }
        result.push_back(emptyTensor(tensor, *sizes, loc));
      } else {
        result.push_back(emptyTensor(RankedTensorType::get({}, leaf), {}, loc));
      }
    });
  }
  if (failed(status)) return failure();
  return result;
}

SmallVector<Value> Construction::flattened(Value value) {
  if (getProductComponents(value.getType())) return products.at(value);
  return {values.lookup(value)};
}

SmallVector<Value> Construction::flattened(ValueRange inputs) {
  SmallVector<Value> result;
  for (Value value : inputs) llvm::append_range(result, flattened(value));
  return result;
}

void Construction::bindProduct(Value original, ValueRange components) {
  if (getProductComponents(original.getType())) products[original] = llvm::to_vector(components);
  else values.map(original, components.front());
}

void Construction::bindValues(ValueRange originals, ValueRange components) {
  auto ranges = getProductLeafRanges(originals.getTypes());
  for (auto [original, range] : llvm::zip(originals, ranges))
    bindProduct(original, components.slice(range.offset, range.size));
}

ArrayAttr Construction::fieldPaths(TypeRange types) {
  SmallVector<Attribute> fields;
  for (auto [index, type] : llvm::enumerate(types))
    walkProductLeaves(type, [&](Type, ArrayRef<unsigned> path) {
      std::string name = std::to_string(index);
      if (!path.empty()) name += "." + getProductPathName(type, path);
      fields.push_back(builder.getStringAttr(name));
    });
  return builder.getArrayAttr(fields);
}

SmallVector<AffineMap> Construction::pointwiseMaps(ValueRange inputs, int64_t rank) {
  SmallVector<AffineMap> maps;
  for (Value input : inputs) {
    SmallVector<AffineExpr> axes;
    if (auto memory = dyn_cast<RankedTensorType>(input.getType()))
      for (int64_t axis = 0; axis < memory.getRank(); ++axis)
        axes.push_back(memory.getDimSize(axis) == 1
            ? builder.getAffineConstantExpr(0)
            : builder.getAffineDimExpr(rank - memory.getRank() + axis));
    maps.push_back(AffineMap::get(rank, 0, axes, builder.getContext()));
  }
  maps.push_back(builder.getMultiDimIdentityMap(rank));
  return maps;
}

Value Construction::elementAt(Value input, ValueRange members, OpBuilder &nested, Location loc) {
  auto type = dyn_cast<ShapedType>(input.getType());
  if (!type) return input;
  SmallVector<Value> indices;
  for (int64_t axis = 0; axis < type.getRank(); ++axis)
    indices.push_back(type.getDimSize(axis) == 1
        ? Value(nested.create<arith::ConstantIndexOp>(loc, 0))
        : members[members.size() - type.getRank() + axis]);
  return extractElement(nested, loc, input, indices);
}

LogicalResult Construction::lower(DimOp op) {
  auto size = extent(op.getSource(), op.getAxis(), op.getLoc());
  if (failed(size)) return failure();
  values.map(op.getResult(), *size);
  return success();
}

LogicalResult Construction::lower(DomainOp op) {
  Location loc = op.getLoc();
  Value begin = values.lookup(op.getBounds()[0]);
  Value end = values.lookup(op.getBounds()[1]);
  Value step = op.getBounds().size() == 3 ? values.lookup(op.getBounds()[2]) : constant(loc, 1);
  auto physicalBegin = indexValue(builder, begin, op.getBounds()[0].getType(), loc);
  auto physicalEnd = indexValue(builder, end, op.getBounds()[1].getType(), loc);
  auto physicalStep = indexValue(builder, step, op.getBounds().size() == 3 ? op.getBounds()[2].getType() : builder.getIndexType(), loc);
  if (failed(physicalBegin) || failed(physicalEnd) || failed(physicalStep)) return failure();
  Value extent = domainExtent(*physicalBegin, *physicalEnd, *physicalStep, loc);
  Domain domain{*physicalBegin, *physicalEnd, *physicalStep, extent};
  domains[op.getResult()] = domain;
  return success();
}

LogicalResult Construction::lower(SubregionOp op) {
  Location loc = op.getLoc();
  auto source = domains.find(op.getInputs()[0]);
  if (source == domains.end()) return op.emitError("CPU subregion requires a realized source domain");
  Domain domain = source->second;
  unsigned position = 1;
  if (op.getHasStart()) {
    Value input = op.getInputs()[position++];
    auto start = indexValue(builder, values.lookup(input), input.getType(), loc);
    if (failed(start)) return failure();
    domain.begin = *start;
  }
  if (op.getHasStop()) {
    Value input = op.getInputs()[position];
    auto stop = indexValue(builder, values.lookup(input), input.getType(), loc);
    if (failed(stop)) return failure();
    domain.end = *stop;
  }
  domain.extent = domainExtent(domain.begin, domain.end, domain.step, loc);
  domains[op.getResult()] = domain;
  return success();
}

LogicalResult Construction::lower(RegionEndOp op) {
  auto source = domains.find(op.getSource());
  if (source == domains.end()) return op.emitError("CPU region end requires a realized source domain");
  values.map(op.getResult(), source->second.end);
  return success();
}

LogicalResult Construction::lower(MakeRecordOp op) {
  bindProduct(op.getResult(), flattened(op.getFields()));
  return success();
}

LogicalResult Construction::lower(MakeTupleOp op) {
  bindProduct(op.getResult(), flattened(op.getComponents()));
  return success();
}

LogicalResult Construction::lower(ExtractOp op) {
  auto range = getProductLeafRange(op.getProduct().getType(),
                                  {static_cast<unsigned>(op.getField())});
  if (failed(range)) return op.emitError("CPU extract has no canonical field schema");
  bindProduct(op.getResult(), ValueRange(products.at(op.getProduct()))
                                  .slice(range->offset, range->size));
  return success();
}

} // namespace intent::kir_to_cpu
