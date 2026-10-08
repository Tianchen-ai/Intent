#include "Construction.h"

namespace intent::kir_to_dsa {

Value Construction::allocate(Location loc, Type element, int64_t rows, int64_t columns, int64_t space) {
  auto type = MemRefType::get({rows, columns}, element, MemRefLayoutAttrInterface{}, b.getI64IntegerAttr(space));
  auto allocation = b.create<memref::AllocaOp>(loc, type);
  allocation.setAlignment(128);
  return allocation;
}

Value Construction::allocateLike(Location loc, Value input, Type element) {
  auto type = cast<MemRefType>(input.getType());
  return allocate(loc, element ? element : type.getElementType(), type.getDimSize(0), type.getDimSize(1));
}

std::optional<LocalAxis> Construction::selectedAxis(
    Value value, unsigned axis, ArrayRef<unsigned> fieldPath) {
  auto type = logicalTensorType(value, fieldPath);
  if (type.getDimSize(axis) == 1) return std::nullopt;
  if (auto selected = valueSlices.find(value);
      fieldPath.empty() && selected != valueSlices.end())
    if (auto found = selected->second.find(axis); found != selected->second.end())
      return found->second;
  auto ids = cast<TensorShapeAttr>(type.getEncoding()).getDimensions();
  if (auto found = axisBindings.find(ids[axis]); found != axisBindings.end())
    return found->second;
  return std::nullopt;
}

FailureOr<LocalShape> Construction::localShape(
    Value source, Location loc, ArrayRef<unsigned> fieldPath) {
  auto type = logicalTensorType(source, fieldPath);
  if (!type) return failure();
  auto ids = cast<TensorShapeAttr>(type.getEncoding()).getDimensions();
  LocalShape shape;
  for (int64_t axis = 0; axis < type.getRank(); ++axis) {
    if (auto selected = selectedAxis(source, axis, fieldPath)) {
      shape.push_back(*selected);
      continue;
    }
    Value size = logicalExtent(source, axis, loc, fieldPath);
    APInt constant;
    if (!size || !matchPattern(size, m_ConstantInt(&constant)) || constant.isNegative())
      return emitError(loc, "DSA local axis needs a selected execution slice or a compile-call shape binding; dimension ") << ids[axis], failure();
    int64_t capacity = constant.getSExtValue();
    shape.push_back({size, index(loc, 0), size, std::max<int64_t>(capacity, 1)});
  }
  int64_t elements = 1;
  for (const auto &axis : shape) {
    if (elements > std::numeric_limits<int64_t>::max() / axis.capacity)
      return emitError(loc, "DSA selected tensor capacity exceeds the address range"), failure();
    elements *= axis.capacity;
  }
  return shape;
}

Type Construction::storageElement(Type type) { return type.isIndex() ? b.getI64Type() : type; }

Value Construction::allocateTensor(Location loc, Type element, const LocalShape &shape) {
  int64_t rows = 1;
  for (unsigned axis = 0; axis + 1 < shape.size(); ++axis) rows *= shape[axis].capacity;
  int64_t cols = shape.empty() ? 1 : shape.back().capacity;
  Value value = allocate(loc, storageElement(element), rows, cols);
  localShapes[value] = shape;
  Type stored = cast<MemRefType>(value.getType()).getElementType();
  Value zero = b.create<arith::ConstantOp>(loc, b.getZeroAttr(stored));
  b.create<dsa::FillOp>(loc, value, zero);
  return value;
}

Value Construction::projectTensor(Location loc, Value source, Value value,
                                  ArrayRef<unsigned> fieldPath) {
  if (!localShapes.count(value)) return value;
  auto type = logicalTensorType(source, fieldPath);
  LocalShape original = localShapes.lookup(value), selected = original;
  auto ids = cast<TensorShapeAttr>(type.getEncoding()).getDimensions();
  bool changed = false;
  for (int64_t axis = 0; axis < type.getRank(); ++axis) {
    auto binding = axisBindings.find(ids[axis]);
    std::optional<LocalAxis> planned = selectedAxis(source, axis, fieldPath);
    if (type.getDimSize(axis) == 1 || (!planned && binding == axisBindings.end())) continue;
    const auto &current = original[axis];
    const auto &required = planned ? *planned : binding->second;
    if (current.capacity == required.capacity && sameIndex(current.begin, required.begin) && sameIndex(current.count, required.count)) continue;
    if (!matchPattern(current.begin, m_Zero()) || !sameIndex(current.count, current.extent) || !sameIndex(current.extent, required.extent)) {
      emitError(loc, "DSA cached tensor projection requires an available complete logical axis"); return {};
    }
    selected[axis] = required; changed = true;
  }
  if (!changed) return value;
  auto physical = cast<MemRefType>(value.getType());
  Value result = allocateTensor(loc, type.getElementType(), selected);
  if (selected.size() > 2) {
    if (failed(eachElement(loc, selected, [&](ValueRange coordinates) {
      SmallVector<Value> source;
      for (unsigned axis = 0; axis < selected.size(); ++axis)
        source.push_back(add(loc, coordinates[axis], sub(loc, selected[axis].begin, original[axis].begin)));
      storeLocal(loc, loadLocal(loc, value, source), result, coordinates);
      return success();
    }))) return {};
    return result;
  }
  Value offset = index(loc, 0);
  if (selected.size() == 2)
    offset = mul(loc, sub(loc, selected[0].begin, original[0].begin), index(loc, physical.getDimSize(1)));
  if (!selected.empty()) offset = add(loc, offset, sub(loc, selected.back().begin, original.back().begin));
  b.create<dsa::LoadTileOp>(loc, value, result, offset,
      index(loc, selected.size() == 2 ? physical.getDimSize(1) : 0), index(loc, 1),
      selected.size() == 2 ? selected[0].count : index(loc, 1), selected.empty() ? index(loc, 1) : selected.back().count);
  return result;
}

SmallVector<Value> Construction::physicalCoordinates(Location loc, Value buffer, ValueRange coordinates) {
  Value row = index(loc, 0);
  if (coordinates.size() >= 2) row = coordinates.front();
  if (coordinates.size() > 2) {
    const auto &shape = localShapes.find(buffer)->second;
    for (unsigned axis = 1; axis + 1 < coordinates.size(); ++axis)
      row = add(loc, mul(loc, row, index(loc, shape[axis].capacity)), coordinates[axis]);
  }
  return {row, coordinates.empty() ? index(loc, 0) : coordinates.back()};
}

Value Construction::loadLocal(Location loc, Value value, ValueRange coordinates) {
  return b.create<memref::LoadOp>(loc, value, physicalCoordinates(loc, value, coordinates));
}

void Construction::storeLocal(Location loc, Value value, Value buffer, ValueRange coordinates) {
  value = scalarCast(loc, value, cast<MemRefType>(buffer.getType()).getElementType());
  b.create<memref::StoreOp>(loc, value, buffer, physicalCoordinates(loc, buffer, coordinates));
}

LogicalResult Construction::eachElement(Location loc, const LocalShape &shape,
    const std::function<LogicalResult(ValueRange)> &body) {
  SmallVector<Value> coordinates;
  std::function<LogicalResult(unsigned)> visit = [&](unsigned axis) -> LogicalResult {
    if (axis == shape.size()) return body(coordinates);
    return loop(loc, index(loc, 0), shape[axis].count, index(loc, 1), [&](Value i) {
      coordinates.push_back(i);
      LogicalResult result = visit(axis + 1);
      coordinates.pop_back();
      return result;
    });
  };
  return visit(0);
}

bool Construction::compactPrefix(const LocalShape &shape) {
  for (unsigned axis = 1; axis < shape.size(); ++axis) {
    APInt count;
    if (!matchPattern(shape[axis].count, m_ConstantInt(&count)) || count.getSExtValue() != shape[axis].capacity) return false;
  }
  return true;
}

bool Construction::completeShape(const LocalShape &shape) {
  return llvm::all_of(shape, [&](const LocalAxis &axis) {
    return matchPattern(axis.begin, m_Zero()) && sameIndex(axis.count, axis.extent);
  });
}

Value Construction::contiguousView(Location loc, Value source, const LocalShape &shape) {
  auto type = cast<MemRefType>(source.getType());
  int64_t rows = 1;
  for (unsigned axis = 0; axis + 1 < shape.size(); ++axis) rows *= shape[axis].capacity;
  int64_t columns = shape.empty() ? 1 : shape.back().capacity;
  auto targetType = MemRefType::get({rows, columns}, type.getElementType(), MemRefLayoutAttrInterface{}, type.getMemorySpace());
  Value result = b.create<memref::ReinterpretCastOp>(loc, targetType, source, int64_t(0),
      ArrayRef<int64_t>{rows, columns}, ArrayRef<int64_t>{columns, 1});
  localShapes[result] = shape;
  return result;
}

LogicalResult Construction::reshapeTensor(Operation *op, const LocalShape &shape) {
  Location loc = op->getLoc();
  Value input = get(op->getOperand(0));
  if (!input || !localShapes.count(input)) return op->emitError("DSA reshape input is unavailable");
  LocalShape original = localShapes.lookup(input);
  if (!completeShape(original) || !completeShape(shape))
    return op->emitError("DSA reshape requires a complete logical tensor before selecting execution slices");
  int64_t elements = 1;
  for (const auto &axis : shape) elements *= axis.capacity;
  Value output;
  if (compactPrefix(original) && compactPrefix(shape) && elements == cast<MemRefType>(input.getType()).getNumElements()) {
    output = contiguousView(loc, input, shape);
  } else {
    output = allocateTensor(loc, cast<RankedTensorType>(op->getResult(0).getType()).getElementType(), shape);
    if (failed(eachElement(loc, shape, [&](ValueRange coordinates) {
      Value linear = index(loc, 0);
      for (unsigned axis = 0; axis < shape.size(); ++axis)
        linear = add(loc, mul(loc, linear, shape[axis].count), coordinates[axis]);
      SmallVector<Value> source(original.size());
      for (int64_t axis = static_cast<int64_t>(original.size()) - 1; axis >= 0; --axis) {
        // A zero-sized tensor has no active iterations; use a nonzero divisor
        // so its unreachable indexing remains well-defined during folding.
        Value count = b.create<arith::MaxSIOp>(loc, original[axis].count, index(loc, 1));
        source[axis] = b.create<arith::RemSIOp>(loc, linear, count);
        linear = b.create<arith::DivSIOp>(loc, linear, count);
      }
      storeLocal(loc, loadLocal(loc, input, source), output, coordinates);
      return success();
    }))) return failure();
  }
  values.map(op->getResult(0), output); return success();
}

LogicalResult Construction::joinTensor(Operation *op, const LocalShape &shape) {
  Location loc = op->getLoc();
  Value lhs = get(op->getOperand(0)), rhs = get(op->getOperand(1));
  if (!lhs || !rhs || !localShapes.count(lhs) || !localShapes.count(rhs))
    return op->emitError("DSA join inputs are unavailable");
  LocalShape inputShape = localShapes.lookup(lhs), rightShape = localShapes.lookup(rhs);
  if (shape.size() != inputShape.size() + 1 || rightShape.size() != inputShape.size() ||
      shape.back().capacity != 2 || !matchPattern(shape.back().begin, m_Zero()) ||
      !sameIndex(shape.back().count, index(loc, 2)))
    return op->emitError("DSA join requires an unsliced trailing pair axis");
  for (unsigned axis = 0; axis < inputShape.size(); ++axis) {
    const LocalAxis *others[] = {&rightShape[axis], &shape[axis]};
    for (const auto *other : others)
      if (other->capacity != inputShape[axis].capacity || !sameIndex(other->begin, inputShape[axis].begin) ||
          !sameIndex(other->count, inputShape[axis].count))
        return op->emitError("DSA join inputs and output must have matching execution slices; axis ") << axis
            << ", input capacity=" << inputShape[axis].capacity << ", other capacity=" << other->capacity
            << ", input begin=" << inputShape[axis].begin << ", other begin=" << other->begin
            << ", input count=" << inputShape[axis].count << ", other count=" << other->count;
  }
  Value output = allocateTensor(loc, cast<RankedTensorType>(op->getResult(0).getType()).getElementType(), shape);
  if (!inputShape.empty() && compactPrefix(inputShape)) {
    auto physical = cast<MemRefType>(lhs.getType());
    LocalShape interleaved = inputShape;
    auto &last = interleaved.back();
    last.extent = mul(loc, last.extent, index(loc, 2));
    last.begin = mul(loc, last.begin, index(loc, 2));
    last.count = mul(loc, last.count, index(loc, 2));
    last.capacity *= 2;
    Value destination = contiguousView(loc, output, interleaved), rows = index(loc, 1);
    for (unsigned axis = 0; axis + 1 < inputShape.size(); ++axis) rows = mul(loc, rows, inputShape[axis].count);
    for (auto [slot, input] : llvm::enumerate(SmallVector<Value>{lhs, rhs}))
      b.create<dsa::StoreTileOp>(loc, input, destination, index(loc, slot), index(loc, 2 * physical.getDimSize(1)),
          index(loc, 2), rows, inputShape.back().count);
  } else if (failed(eachElement(loc, inputShape, [&](ValueRange coordinates) {
    SmallVector<Value> target(coordinates);
    target.push_back(index(loc, 0));
    storeLocal(loc, loadLocal(loc, lhs, coordinates), output, target);
    target.back() = index(loc, 1);
    storeLocal(loc, loadLocal(loc, rhs, coordinates), output, target);
    return success();
  }))) return failure();
  values.map(op->getResult(0), output); return success();
}

bool Construction::canDefer(Operation *op) {
  if (isa<RegionFoldOp, RegionScanOp, IfOp, ForOp, WhileOp, ParallelOp, QuantizeOp>(op)) return false;
  bool aggregate = llvm::any_of(op->getResultTypes(), [&](Type type) {
    return isa<RankedTensorType>(type) || static_cast<bool>(getProductComponents(type));
  });
  if (!aggregate) return false;
  if (auto load = dyn_cast<ViewLoadOp>(op)) {
    auto view = dyn_cast<ViewType>(load.getSource().getType());
    return view && view.getAccess() == 0;
  }
  return isMemoryEffectFree(op);
}

LogicalResult Construction::tensorOperation(Operation *op) {
  Location loc = op->getLoc();
  Value result = op->getResult(0);
  auto type = cast<RankedTensorType>(result.getType());
  auto shape = localShape(result, loc);
  if (failed(shape)) return failure();
  if (isa<ReshapeOp>(op)) return reshapeTensor(op, *shape);
  if (isa<JoinOp>(op)) return joinTensor(op, *shape);
  if (auto matrix = dyn_cast<ContractOp>(op)) return localMatMul(matrix, *shape);
  if (auto matrix = dyn_cast<ScaledContractOp>(op)) return scaledMatMul(matrix, *shape);
  if (auto indices = dyn_cast<IndicesOp>(op)) {
    Value source = indices->getOperand(0);
    if (!bindDomain(source) || shape->size() != 1) return op->emitError("DSA indices require a bound interval");
    Value output = allocateTensor(loc, type.getElementType(), *shape);
    Domain domain = domains.lookup(source);
    if (failed(eachElement(loc, *shape, [&](ValueRange coordinates) {
      Value coordinate = add(loc, domain.begin, mul(loc, add(loc, shape->front().begin, coordinates[0]), domain.step));
      storeLocal(loc, coordinate, output, coordinates);
      return success();
    }))) return failure();
    values.map(result, output); return success();
  }
  if (isa<BroadcastOp, FullOp>(op)) {
    Value input = get(op->getOperand(0));
    if (!input) return op->emitError("DSA broadcast input is unavailable");
    Value output = allocateTensor(loc, type.getElementType(), *shape);
    if (!isa<MemRefType>(input.getType())) {
      Value value = scalarCast(loc, input, storageElement(type.getElementType()));
      if (!value) return op->emitError("DSA broadcast element dtype is unavailable");
      b.create<dsa::FillOp>(loc, output, value);
    } else {
      auto inputType = cast<RankedTensorType>(op->getOperand(0).getType());
      if (inputType.getRank() > type.getRank()) return op->emitError("DSA broadcast cannot remove axes");
      unsigned leading = type.getRank() - inputType.getRank();
      auto sourceShape = localShapes.find(input);
      auto physical = cast<MemRefType>(input.getType());
      bool tiled = type.getRank() == 2 && inputType.getRank() >= 1 &&
          physical.getRank() == 2 && physical.getElementType() == storageElement(type.getElementType()) &&
          physical.getMemorySpace() == b.getI64IntegerAttr(dsa::nramSpace) &&
          sourceShape != localShapes.end() && sourceShape->second.size() == unsigned(inputType.getRank());
      int64_t strides[] = {0, 0};
      if (tiled) {
        for (int64_t axis = 0; axis < inputType.getRank(); ++axis) {
          const auto &sourceAxis = sourceShape->second[axis];
          const auto &targetAxis = (*shape)[leading + axis];
          bool singleton = inputType.getDimSize(axis) == 1 || matchPattern(sourceAxis.extent, m_One());
          if (singleton) {
            tiled &= sourceAxis.capacity >= 1 && matchPattern(sourceAxis.count, m_One()) &&
                matchPattern(sourceAxis.begin, m_Zero());
          } else {
            tiled &= sameIndex(sourceAxis.extent, targetAxis.extent) &&
                sameIndex(sourceAxis.begin, targetAxis.begin) && sameIndex(sourceAxis.count, targetAxis.count);
            strides[leading + axis] = leading + axis == 0 ? physical.getDimSize(1) : 1;
          }
        }
      }
      if (tiled) {
        b.create<dsa::LoadTileOp>(loc, input, output, index(loc, 0),
            index(loc, strides[0]), index(loc, strides[1]), (*shape)[0].count, (*shape)[1].count);
        values.map(result, output); return success();
      }
      if (failed(eachElement(loc, *shape, [&](ValueRange coordinates) {
        SmallVector<Value> source;
        for (int64_t axis = 0; axis < inputType.getRank(); ++axis) {
          bool singleton = inputType.getDimSize(axis) == 1 || matchPattern(localShapes.lookup(input)[axis].extent, m_One());
          source.push_back(singleton ? index(loc, 0) : coordinates[leading + axis]);
        }
        storeLocal(loc, loadLocal(loc, input, source), output, coordinates);
        return success();
      }))) return failure();
    }
    values.map(result, output); return success();
  }
  if (auto transpose = dyn_cast<TransposeOp>(op)) {
    Value input = get(transpose.getInput());
    if (!input) return op->emitError("DSA transpose input is unavailable");
    Value output = allocateTensor(loc, type.getElementType(), *shape);
    auto inputShape = localShapes.find(input);
    if (shape->size() == 2 && inputShape != localShapes.end() && inputShape->second.size() == 2 &&
        cast<IntegerAttr>(transpose.getPermutation()[0]).getInt() == 1 &&
        cast<IntegerAttr>(transpose.getPermutation()[1]).getInt() == 0) {
      bool aligned = true;
      for (unsigned axis = 0; axis < 2; ++axis) {
        const auto &sourceAxis = inputShape->second[1 - axis];
        const auto &targetAxis = (*shape)[axis];
        aligned &= sourceAxis.capacity == targetAxis.capacity && sameIndex(sourceAxis.begin, targetAxis.begin) &&
            sameIndex(sourceAxis.count, targetAxis.count) && sameIndex(sourceAxis.extent, targetAxis.extent);
      }
      if (aligned) {
        b.create<dsa::TransposeOp>(loc, input, output, inputShape->second[0].count, inputShape->second[1].count);
        values.map(result, output); return success();
      }
    }
    if (failed(eachElement(loc, *shape, [&](ValueRange coordinates) {
      SmallVector<Value> source(coordinates.size());
      for (auto [axis, perm] : llvm::enumerate(transpose.getPermutation()))
        source[cast<IntegerAttr>(perm).getInt()] = coordinates[axis];
      storeLocal(loc, loadLocal(loc, input, source), output, coordinates); return success();
    }))) return failure();
    values.map(result, output); return success();
  }
  if (!isa<UnaryOp, BinaryOp, CompareOp, SelectOp, MaskOp, CastOp>(op))
    return op->emitError("tensor operation has no DSA local implementation");
  if (auto binary = dyn_cast<BinaryOp>(op); binary && type.getElementType().isInteger(1) && shape->size() <= 2 &&
      (binary.getOperatorKind() == BinaryOperator::BitwiseAnd || binary.getOperatorKind() == BinaryOperator::LogicalAnd)) {
    Value identitySource = constantTrue(binary.getLhs()) ? binary.getRhs() :
        constantTrue(binary.getRhs()) ? binary.getLhs() : Value();
    if (identitySource && isa<RankedTensorType>(identitySource.getType())) {
      Value input = get(identitySource);
      auto found = localShapes.find(input);
      bool aligned = found != localShapes.end() && found->second.size() == shape->size();
      if (aligned) for (auto [sourceAxis, targetAxis] : llvm::zip(found->second, *shape))
        aligned &= sourceAxis.capacity == targetAxis.capacity && sameIndex(sourceAxis.extent, targetAxis.extent) &&
            sameIndex(sourceAxis.begin, targetAxis.begin) && sameIndex(sourceAxis.count, targetAxis.count);
      if (aligned) {
        Value output = allocateTensor(loc, type.getElementType(), *shape);
        auto physical = cast<MemRefType>(input.getType());
        b.create<dsa::LoadTileOp>(loc, input, output, index(loc, 0), index(loc, physical.getDimSize(1)), index(loc, 1),
            shape->size() == 2 ? (*shape)[0].count : index(loc, 1), shape->empty() ? index(loc, 1) : shape->back().count);
        values.map(result, output);
        return success();
      }
    }
  }
  SmallVector<Value> inputs;
  for (Value operand : op->getOperands()) {
    Value value = get(operand);
    if (!value) return op->emitError("DSA tensor operand is unavailable");
    inputs.push_back(value);
  }
  Value output = allocateTensor(loc, type.getElementType(), *shape);
  bool matching = llvm::all_of(inputs, [&](Value value) {
    auto buffer = dyn_cast<MemRefType>(value.getType());
    return buffer && buffer.getShape() == cast<MemRefType>(output.getType()).getShape();
  });
  bool floating = isa<FloatType>(type.getElementType());
  auto unsignedElement = [](Type type) {
    auto integer = dyn_cast<IntegerType>(getElementTypeOrSelf(type));
    return integer && integer.isUnsigned();
  };
  auto nativeCast = [&](CastOp cast) {
    Type source = getElementTypeOrSelf(cast.getInput().getType());
    Type target = type.getElementType();
    if (isa<Float8E4M3FNType, Float8E5M2Type>(source) ||
        isa<Float8E4M3FNType, Float8E5M2Type>(target)) return false;
    // The DSA tile cast has no signedness operand. Equal-width integer casts
    // preserve bits; other unsigned numeric conversions use scalar primitives.
    return (!unsignedElement(source) && !unsignedElement(target)) ||
        (isa<IntegerType>(source) && isa<IntegerType>(target) &&
         source.getIntOrFloatBitWidth() == target.getIntOrFloatBitWidth());
  };
  if (auto binary = dyn_cast<BinaryOp>(op); binary && matching && floating)
    b.create<dsa::BinaryOp>(loc, inputs[0], inputs[1], output, binary.getOperatorKindAttr(), binary.getApproximateAttr(), binary.getFlushToZeroAttr(), Value());
  else if (auto unary = dyn_cast<UnaryOp>(op); unary && matching && floating)
    b.create<dsa::UnaryOp>(loc, inputs[0], output, unary.getOperatorKindAttr(), unary.getApproximateAttr(), unary.getFlushToZeroAttr(), Value());
  else if (auto cast = dyn_cast<CastOp>(op); cast && matching && nativeCast(cast)) {
    if (cast.getRounding() && *cast.getRounding() != 0)
      return cast.emitError("DSA explicit rounding is not implemented");
    b.create<dsa::CastOp>(loc, inputs[0], output);
  } else if (isa<SelectOp>(op) && matching)
    b.create<dsa::SelectOp>(loc, inputs[0], inputs[1], inputs[2], output, Value{});
  else if (isa<MaskOp>(op) && matching)
    b.create<dsa::SelectOp>(loc, inputs[1], inputs[0], inputs[2], output, Value{});
  else if (auto compare = dyn_cast<CompareOp>(op); compare && matching &&
      !unsignedElement(compare.getLhs().getType()) &&
      mlir::cast<MemRefType>(inputs.front().getType()).getElementType().isInteger(64))
    b.create<dsa::CompareOp>(loc, inputs[0], inputs[1], output, compare.getPredicateAttr(), Value());
  else if (failed(eachElement(loc, *shape, [&](ValueRange coordinates) {
    SmallVector<Value> scalars;
    for (Value input : inputs) scalars.push_back(isa<MemRefType>(input.getType()) ? loadLocal(loc, input, coordinates) : input);
    auto value = scalarOperation(op, scalars);
    if (failed(value)) return failure();
    storeLocal(loc, *value, output, coordinates); return success();
  }))) return failure();
  values.map(result, output); return success();
}

} // namespace intent::kir_to_dsa
