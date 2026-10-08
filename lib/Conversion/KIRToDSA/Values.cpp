#include "Construction.h"
#include "Intent/Conversion/ScalarLowering.h"

namespace intent::kir_to_dsa {

SmallVector<LogicalComponent> logicalComponents(ValueRange values) {
  SmallVector<LogicalComponent> result;
  for (Value value : values)
    walkProductLeaves(value.getType(), [&](Type type, ArrayRef<unsigned> path) {
      result.push_back({value, llvm::to_vector<2>(path), type});
    });
  return result;
}

RankedTensorType logicalTensorType(Value value, ArrayRef<unsigned> fieldPath) {
  Type type = getProductComponentType(value.getType(), fieldPath);
  if (!type) return {};
  if (auto view = dyn_cast<ViewType>(type)) type = view.getTensor();
  if (auto buffer = dyn_cast<BufferType>(type)) type = buffer.getTensor();
  return dyn_cast<RankedTensorType>(type);
}

Value Construction::index(Location loc, int64_t n) { return b.create<arith::ConstantIndexOp>(loc, n); }

Value Construction::add(Location loc, Value a, Value c) { return b.createOrFold<arith::AddIOp>(loc, a, c); }

Value Construction::mul(Location loc, Value a, Value c) { return b.createOrFold<arith::MulIOp>(loc, a, c); }

Value Construction::sub(Location loc, Value a, Value c) { return b.createOrFold<arith::SubIOp>(loc, a, c); }

Value Construction::get(Value source) {
  Value value = values.lookupOrNull(source);
  if (!value && source.getDefiningOp()) {
    if (failed(materialize(source.getDefiningOp()))) return {};
    value = values.lookupOrNull(source);
  }
  if (value) if (auto tensor = dyn_cast<RankedTensorType>(source.getType())) {
    value = projectTensor(source.getLoc(), source, value);
    if (value) values.map(source, value);
  }
  return value;
}

LogicalResult Construction::materialize(Operation *op) {
  if (!materializing.insert(op).second) return op->emitError("cyclic DSA value materialization");
  LogicalResult result = lowerOperation(op);
  materializing.erase(op);
  return result;
}

Value Construction::asIndex(Value value, Location loc) {
  if (!value || !value.getType().isIntOrIndex())
    return {};
  auto result = castScalarValue(b, loc, value, b.getIndexType());
  return succeeded(result) ? *result : Value();
}

Value Construction::boundExtent(Value value, ArrayRef<unsigned> fieldPath,
                                unsigned axis) {
  if (auto found = publicExtents.find(value);
      fieldPath.empty() && found != publicExtents.end())
    return axis < found->second.size() ? found->second[axis] : Value();
  Value physical;
  if (getProductComponents(value.getType())) {
    auto field = getProductLeafRange(value.getType(), fieldPath);
    auto found = products.find(value);
    if (failed(field) || field->size != 1 || found == products.end() ||
        field->offset >= found->second.size()) return {};
    physical = found->second[field->offset];
  } else if (fieldPath.empty()) {
    physical = values.lookupOrNull(value);
  }
  if (!physical) return {};
  if (auto found = localShapes.find(physical); found != localShapes.end())
    return axis < found->second.size() ? found->second[axis].extent : Value();
  // A local allocation's descriptor contains capacity, not logical extent.
  return {};
}

Value Construction::logicalExtent(Value value, unsigned axis, Location loc,
                                  ArrayRef<unsigned> fieldPath) {
  LogicalShapeMaterialization materialization;
  materialization.lookup = [&](Value source, ArrayRef<unsigned> path,
                                unsigned sourceAxis) {
    return boundExtent(source, path, sourceAxis);
  };
  materialization.leaf = [&](const TensorExtentFact &fact) -> FailureOr<Value> {
    if (fact.constant) return index(loc, *fact.constant);
    if (fact.value) {
      Value scalar = asIndex(get(fact.value), loc);
      return scalar ? FailureOr<Value>(scalar) : FailureOr<Value>(failure());
    }
    if (fact.domain) {
      if (!bindDomain(fact.domain)) return failure();
      auto domain = domains.find(fact.domain);
      if (domain == domains.end()) return failure();
      return domain->second.extent;
    }
    if (fact.source) {
      Value size = boundExtent(fact.source, fact.fieldPath, fact.axis);
      if (size) return size;
    }
    return failure();
  };
  materialization.multiply = [&](Value lhs, Value rhs) -> FailureOr<Value> {
    return mul(loc, lhs, rhs);
  };
  materialization.exactDivide = [&](Value lhs, Value rhs) -> FailureOr<Value> {
    if (matchPattern(rhs, m_Zero())) return failure();
    return Value(b.createOrFold<arith::DivSIOp>(loc, lhs, rhs));
  };
  materialization.subtract = [&](Value lhs, Value rhs) -> FailureOr<Value> {
    return sub(loc, lhs, rhs);
  };
  materialization.ceilDivide = [&](Value lhs, Value rhs) -> FailureOr<Value> {
    return Value(b.createOrFold<arith::CeilDivSIOp>(loc, lhs, rhs));
  };
  materialization.maximum = [&](Value lhs, Value rhs) -> FailureOr<Value> {
    return Value(b.createOrFold<arith::MaxSIOp>(loc, lhs, rhs));
  };
  auto result = materializeLogicalExtent(analysis, value, axis, materialization,
                                         fieldPath);
  return succeeded(result) ? *result : Value();
}

TensorExtentFact Construction::knownExtent(Value value, unsigned axis,
                                          ArrayRef<unsigned> fieldPath) {
  auto fact = analysis.tensorExtent(value, axis, fieldPath);
  Value size = boundExtent(value, fieldPath, axis);
  if (!size && fact.source)
    size = boundExtent(fact.source, fact.fieldPath, fact.axis);
  if (!size && fact.value) size = values.lookupOrNull(fact.value);
  if (!size && fact.domain)
    if (auto found = domains.find(fact.domain); found != domains.end())
      size = found->second.extent;
  if (!size) return fact;
  APInt constant;
  if (matchPattern(size, m_ConstantInt(&constant)))
    return {constant.getSExtValue(), {}};
  return {std::nullopt, size};
}

bool Construction::sameIndex(Value a, Value c) {
  if (a == c) return true;
  APInt lhs, rhs;
  return matchPattern(a, m_ConstantInt(&lhs)) && matchPattern(c, m_ConstantInt(&rhs)) && lhs == rhs;
}

LogicalResult Construction::lowerScalarOperation(Operation *operation) {
  SmallVector<Value> operands;
  for (Value input : operation->getOperands()) {
    Value value = get(input);
    if (!value || isa<MemRefType>(value.getType()))
      return operation->emitError("DSA scalar operand is unavailable");
    operands.push_back(value);
  }
  auto result = scalarOperation(operation, operands);
  if (failed(result)) return failure();
  values.map(operation->getResult(0), *result);
  return success();
}

FailureOr<Value> Construction::scalarOperation(Operation *operation,
                                              ValueRange operands) {
  Location loc = operation->getLoc();
  SmallVector<Value> scalars;
  for (auto [input, value] : llvm::zip(operation->getOperands(), operands)) {
    auto scalar = castScalarValue(b, loc, value, getElementTypeOrSelf(input.getType()));
    if (failed(scalar))
      return operation->emitError("DSA element has incompatible scalar dtype"), failure();
    scalars.push_back(*scalar);
  }
  if (auto binary = dyn_cast<BinaryOp>(operation)) {
    Value lhs = scalars[0], rhs = scalars[1];
    if (binary.getOperatorKind() == BinaryOperator::TrueDivide && binary.getApproximate() && lhs.getType().isF32()) {
      Value left = allocate(loc, b.getF32Type(), 1, 1), right = allocateLike(loc, left), output = allocateLike(loc, left);
      b.create<dsa::FillOp>(loc, left, lhs); b.create<dsa::FillOp>(loc, right, rhs);
      b.create<dsa::BinaryOp>(loc, left, right, output, binary.getOperatorKindAttr(), binary.getApproximateAttr(), binary.getFlushToZeroAttr(), Value());
      return loadLocal(loc, output, {});
    }
    if (binary.getApproximate() || binary.getFlushToZero())
      return binary.emitError("DSA scalar approximate arithmetic is not implemented"), failure();
  }
  if (auto unary = dyn_cast<UnaryOp>(operation)) {
    Value input = scalars[0];
    if (unary.getOperatorKind() == UnaryOperator::Exp2 && unary.getApproximate() && input.getType().isF32()) {
      Value source = allocate(loc, b.getF32Type(), 1, 1), output = allocateLike(loc, source);
      b.create<dsa::FillOp>(loc, source, input);
      b.create<dsa::UnaryOp>(loc, source, output, unary.getOperatorKindAttr(), unary.getApproximateAttr(), unary.getFlushToZeroAttr(), Value());
      return loadLocal(loc, output, {});
    }
    if (unary.getApproximate() || unary.getFlushToZero())
      return unary.emitError("DSA scalar numerical mode is not implemented"), failure();
    if (unary.getOperatorKind() == UnaryOperator::Sigmoid) {
      Type declared = input.getType();
      if (declared.isF16() || declared.isBF16()) input = scalarCast(loc, input, b.getF32Type());
      Value one = b.create<arith::ConstantOp>(loc, b.getFloatAttr(input.getType(), 1.0));
      Value zero = b.create<arith::ConstantOp>(loc, b.getZeroAttr(input.getType()));
      Value absolute = b.create<math::AbsFOp>(loc, input);
      Value negative = b.create<arith::NegFOp>(loc, absolute);
      Value exponential = b.create<math::ExpOp>(loc, negative);
      Value denominator = b.create<arith::AddFOp>(loc, one, exponential);
      Value belowZero = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, input, zero);
      Value numerator = b.create<arith::SelectOp>(loc, belowZero, exponential, one);
      return scalarCast(loc, b.create<arith::DivFOp>(loc, numerator, denominator), declared);
    }
  }
  return intent::lowerScalarOperation(operation, scalars, b);
}

bool Construction::constantTrue(Value value) {
  Operation *op = value.getDefiningOp();
  if (!op) return false;
  if (isa<FullOp, BroadcastOp>(op)) return constantTrue(op->getOperand(0));
  auto constant = dyn_cast<ConstantOp>(op);
  auto integer = constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr();
  return integer && integer.getType().isInteger(1) && integer.getValue().isOne();
}

Value Construction::scalarCast(Location loc, Value value, Type type) {
  auto result = castScalarValue(b, loc, value, type);
  return succeeded(result) ? *result : Value();
}

SmallVector<Value> Construction::flatten(ValueRange inputs) {
  SmallVector<Value> result;
  for (Value input : inputs) {
    if (getProductComponents(input.getType())) {
      if (!products.count(input) && input.getDefiningOp() && failed(materialize(input.getDefiningOp()))) return {};
      auto fields = products.lookup(input);
      {
        auto schema = logicalComponents(ValueRange{input});
        if (fields.size() != schema.size()) return {};
        for (unsigned i = 0; i < schema.size(); ++i) if (isa<RankedTensorType>(schema[i].type)) {
          fields[i] = projectTensor(input.getLoc(), input, fields[i], schema[i].path);
          if (!fields[i]) return {};
        }
        products[input] = fields;
      }
      llvm::append_range(result, fields);
    }
    else result.push_back(get(input));
  }
  return result;
}

void Construction::bindProduct(Value original, ValueRange fields) {
  if (getProductComponents(original.getType())) products[original] = llvm::to_vector(fields);
  else values.map(original, fields.front());
}

FailureOr<SmallVector<Value>> Construction::makeSlots(ValueRange sources,
                                                      Location loc) {
  SmallVector<Value> slots;
  for (const auto &component : logicalComponents(sources)) {
    Type field = component.type;
    if (auto tensor = dyn_cast<RankedTensorType>(field)) {
      auto shape = localShape(component.value, loc, component.path);
      if (failed(shape)) return failure();
      slots.push_back(allocateTensor(loc, tensor.getElementType(), *shape));
    } else {
      Value slot = allocate(loc,
          field.isIndex() || isa<LogicalIndexType>(field) ? b.getI64Type() : field,
          1, 1);
      localShapes[slot] = {};
      slots.push_back(slot);
    }
  }
  return slots;
}

LogicalResult Construction::copyTo(Value value, Value slot, Location loc) {
  if (!value) return emitError(loc, "DSA state field has no physical value");
  if (isa<MemRefType>(value.getType())) {
    if (value.getType() != slot.getType()) return emitError(loc, "DSA state update changes its local shape or dtype");
    b.create<memref::CopyOp>(loc, value, slot);
  } else {
    value = scalarCast(loc, value, cast<MemRefType>(slot.getType()).getElementType());
    if (!value) return emitError(loc, "DSA state scalar has no compatible storage type");
    b.create<memref::StoreOp>(loc, value, slot, ValueRange{index(loc, 0), index(loc, 0)});
  }
  return success();
}

void Construction::bindSlots(ValueRange originals, ValueRange slots, Location loc) {
  for (auto [original, range] : llvm::zip_equal(originals, getProductLeafRanges(originals.getTypes()))) {
    SmallVector<Type> flat;
    appendProductLeafTypes(original.getType(), flat);
    SmallVector<Value> fields;
    for (auto [type, slot] : llvm::zip_equal(flat, slots.slice(range.offset, range.size))) {
      Value value = slot;
      if (!isa<RankedTensorType>(type)) {
        value = b.create<memref::LoadOp>(loc, slot, ValueRange{index(loc, 0), index(loc, 0)});
        if (type.isIndex() || isa<LogicalIndexType>(type)) value = asIndex(value, loc);
      }
      fields.push_back(value);
    }
    bindProduct(original, fields);
  }
}

FailureOr<Value> Construction::lowerResults(Block &block, ValueRange slots) {
  if (failed(lowerOperations(block))) return failure();
  auto condition = dyn_cast<ConditionOp>(block.getTerminator());
  Value predicate = condition ? get(condition.getCondition()) : Value();
  auto fields = flatten(condition ? condition.getArgs()
                                 : cast<YieldOp>(block.getTerminator()).getInputs());
  if (fields.size() != slots.size()) return block.getParentOp()->emitError("DSA control state schema mismatch"), failure();
  for (auto [field, slot] : llvm::zip(fields, slots))
    if (failed(copyTo(field, slot, block.getParentOp()->getLoc()))) return failure();
  return predicate;
}

} // namespace intent::kir_to_dsa
