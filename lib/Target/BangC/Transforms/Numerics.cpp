#include "PassDetail.h"
#include "Intent/Target/BangC/NativeWorkspace.h"

using namespace mlir;
namespace intent::bangc {
bool supportedUnary(UnaryOperator kind) {
  return kind == UnaryOperator::Exp || kind == UnaryOperator::Exp2 || kind == UnaryOperator::Log || kind == UnaryOperator::Sqrt ||
      kind == UnaryOperator::Rsqrt || kind == UnaryOperator::Tanh || kind == UnaryOperator::Abs || kind == UnaryOperator::Negate ||
      kind == UnaryOperator::Sin || kind == UnaryOperator::Cos || kind == UnaryOperator::Floor;
}

bool supportedBinary(BinaryOperator kind) {
  return kind == BinaryOperator::Add || kind == BinaryOperator::Subtract || kind == BinaryOperator::Multiply ||
      kind == BinaryOperator::TrueDivide || kind == BinaryOperator::Maximum || kind == BinaryOperator::Minimum ||
      kind == BinaryOperator::MaximumNum || kind == BinaryOperator::MinimumNum;
}

bool supportedScalarBinary(BinaryOperator kind) {
  return kind == BinaryOperator::Add || kind == BinaryOperator::Subtract || kind == BinaryOperator::Multiply;
}

LogicalResult realizeSigmoid(dsa::UnaryOp sigmoid) {
  if (sigmoid.getApproximate() || sigmoid.getFlushToZero())
    return sigmoid.emitError("sigmoid numerical mode has no selected BANG C implementation");
  Location loc = sigmoid.getLoc();
  OpBuilder b(sigmoid);
  auto type = cast<MemRefType>(sigmoid.getInput().getType());
  Value input = sigmoid.getInput(), output = sigmoid.getOutput();
  if (!type.getElementType().isF32()) {
    input = allocate(b, loc, b.getF32Type(), type.getShape(), dsa::nramSpace);
    output = allocate(b, loc, b.getF32Type(), type.getShape(), dsa::nramSpace);
    b.create<dsa::CastOp>(loc, sigmoid.getInput(), input);
  }
  Value absolute = allocate(b, loc, b.getF32Type(), type.getShape(), dsa::nramSpace);
  Value negative = allocate(b, loc, b.getF32Type(), type.getShape(), dsa::nramSpace);
  auto unary = [&](Value from, Value to, UnaryOperator kind) {
    b.create<dsa::UnaryOp>(loc, from, to, UnaryOperatorAttr::get(b.getContext(), kind),
        b.getBoolAttr(false), b.getBoolAttr(false), Value());
  };
  unary(input, absolute, UnaryOperator::Abs);
  unary(absolute, negative, UnaryOperator::Negate);
  unary(negative, absolute, UnaryOperator::Exp);
  auto index = [&](int64_t n) -> Value { return b.create<arith::ConstantIndexOp>(loc, n); };
  Value one = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(1));
  Value zero = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(0));
  auto rows = b.create<scf::ForOp>(loc, index(0), index(type.getDimSize(0)), index(1));
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(rows.getBody());
    auto columns = b.create<scf::ForOp>(loc, index(0), index(type.getDimSize(1)), index(1));
    b.setInsertionPointToStart(columns.getBody());
    SmallVector<Value> coordinates{rows.getInductionVar(), columns.getInductionVar()};
    Value x = b.create<memref::LoadOp>(loc, input, coordinates);
    Value e = b.create<memref::LoadOp>(loc, absolute, coordinates);
    Value denominator = b.create<arith::AddFOp>(loc, one, e);
    Value belowZero = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, x, zero);
    Value numerator = b.create<arith::SelectOp>(loc, belowZero, e, one);
    Value result = b.create<arith::DivFOp>(loc, numerator, denominator);
    b.create<memref::StoreOp>(loc, result, output, coordinates);
  }
  if (output != sigmoid.getOutput()) b.create<dsa::CastOp>(loc, output, sigmoid.getOutput());
  sigmoid.erase();
  return success();
}

void realizeRowSumChannels(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<dsa::ReduceOp> operations;
  function.walk([&](dsa::ReduceOp reduce) {
    auto input = cast<MemRefType>(reduce.getInput().getType());
    if (reduce.getKind() == BinaryOperator::Add && reduce.getAxis() == 1 &&
        input.getDimSize(0) >= 32 && input.getDimSize(0) % 32 == 0 && input.getDimSize(1) >= 32)
      operations.push_back(reduce);
  });
  for (auto reduce : operations) {
    auto input = cast<MemRefType>(reduce.getInput().getType());
    OpBuilder b(reduce);
    Location loc = reduce.getLoc();
    Value storage = allocate(b, loc, input.getElementType(), {input.getDimSize(1), input.getDimSize(0)}, dsa::nramSpace);
    Value rows = b.create<arith::ConstantIndexOp>(loc, input.getDimSize(0));
    auto transpose = b.create<dsa::TransposeOp>(loc, reduce.getInput(), storage, rows, reduce.getCount());
    auto replacement = b.create<dsa::ReduceOp>(loc, storage, reduce.getOutput(), reduce.getScratch(),
        reduce.getCount(), reduce.getIdentity(), reduce.getKindAttr(), b.getI64IntegerAttr(0));
    int64_t nram = 0, wram = 0;
    measureStorage(function, nram, wram);
    if (nram > config.getLocalBytes() || nram > 768 * 1024) {
      replacement.erase(); transpose.erase(); storage.getDefiningOp()->erase();
    } else reduce.erase();
  }
}

static StringRef directExtremaCallee(dsa::BinaryOp binary) {
  auto kind = binary.getKind();
  if ((kind != BinaryOperator::Maximum && kind != BinaryOperator::Minimum) ||
      !isa<MemRefType>(binary.getRhs().getType()) || binary.getOutput() == binary.getRhs()) return {};
  Type element = cast<MemRefType>(binary.getOutput().getType()).getElementType();
  if (!element.isF16() && !element.isF32()) return {};
  dsa::StorageAnalysis storage(binary->getParentOfType<func::FuncOp>());
  if (!storage.disjoint(binary.getOutput(), binary.getRhs())) return {};
  Value lhs = binary.getLhs(), output = binary.getOutput();
  Value origin = storage.uniqueOrigin(lhs);
  if (!storage.disjoint(output, lhs) && output != lhs &&
      !(origin && dsa::isCompleteStorageViewOf(lhs, origin) &&
        dsa::isCompleteStorageViewOf(output, origin))) return {};
  dsa::UniformMemoryAnalysis uniforms(binary->getParentOfType<func::FuncOp>(), storage);
  Value scalar = uniforms.read(binary.getRhs(), binary);
  auto constant = scalar ? scalar.getDefiningOp<arith::ConstantOp>() : arith::ConstantOp{};
  auto value = constant ? dyn_cast<FloatAttr>(constant.getValue()) : FloatAttr{};
  // With a non-NaN RHS, the SDK's src0-preserving unordered result is exactly
  // the propagating operation; no NaN repair workspace is needed.
  if (!value || value.getValue().isNaN()) return {};
  return kind == BinaryOperator::Maximum ? "__bang_maxequal" : "__bang_minequal";
}

LogicalResult realizeNumericExtremaWorkspace(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<dsa::BinaryOp> operations;
  function.walk([&](dsa::BinaryOp binary) {
    if ((binary.getKind() == BinaryOperator::MaximumNum || binary.getKind() == BinaryOperator::MinimumNum ||
         binary.getKind() == BinaryOperator::Maximum || binary.getKind() == BinaryOperator::Minimum) &&
        !binary.getScratch()) operations.push_back(binary);
  });
  for (auto binary : operations) {
    if (!directExtremaCallee(binary).empty()) continue;
    auto type = cast<MemRefType>(binary.getOutput().getType());
    if ((!type.getElementType().isF16() && !type.getElementType().isF32()) ||
        !isa<MemRefType>(binary.getRhs().getType()))
      return binary.emitError("numeric extrema requires a supported floating tile");
    auto shape = extremaScratchShape(type.getElementType(), type.getNumElements());
    int64_t minimumWidth = 1;
    if (binary.getKind() == BinaryOperator::Maximum || binary.getKind() == BinaryOperator::Minimum) {
      dsa::StorageAnalysis storage(function);
      Value output = binary.getOutput(), origin = storage.uniqueOrigin(output);
      for (Value input : ValueRange{binary.getLhs(), binary.getRhs()})
        if (!storage.disjoint(input, output) && input != output &&
            !(origin && dsa::isCompleteStorageViewOf(input, origin) &&
              dsa::isCompleteStorageViewOf(output, origin)))
          minimumWidth = type.getNumElements();
    }
    for (int64_t width = std::max(shape[1], minimumWidth); width >= minimumWidth; width /= 2) {
      OpBuilder b(binary);
      Value scratch = allocate(b, binary.getLoc(), type.getElementType(), {1, width}, dsa::nramSpace);
      binary.getScratchMutable().assign(scratch);
      int64_t nram = 0, wram = 0;
      measureStorage(function, nram, wram);
      if (nram <= config.getLocalBytes() && nram <= 768 * 1024) break;
      binary.getScratchMutable().clear();
      scratch.getDefiningOp()->erase();
    }
    if (!binary.getScratch()) return binary.emitError("numeric extrema has no room for its input-preserving workspace");
  }
  return success();
}

void realizeCompareWorkspace(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<dsa::CompareOp> comparisons;
  function.walk([&](dsa::CompareOp compare) { if (!compare.getScratch()) comparisons.push_back(compare); });
  for (auto compare : comparisons) {
    auto type = cast<MemRefType>(compare.getLhs().getType());
    if (!type.getElementType().isInteger(64) || type.getNumElements() < 256) continue;
    for (int64_t width = int64_t(1) << llvm::Log2_64(std::min<int64_t>(8192, type.getNumElements())); width >= 256; width /= 2) {
      OpBuilder b(compare);
      Value scratch = allocate(b, compare.getLoc(), b.getI32Type(), {6, width}, dsa::nramSpace);
      compare.getScratchMutable().assign(scratch);
      int64_t nram = 0, wram = 0;
      measureStorage(function, nram, wram);
      if (nram <= config.getLocalBytes() && nram <= 768 * 1024) break;
      compare.getScratchMutable().clear();
      scratch.getDefiningOp()->erase();
    }
  }
}

static Value ownedCompleteStorage(Value memory, dsa::StorageAnalysis &storage) {
  Value origin = storage.uniqueOrigin(memory);
  return origin && origin.getDefiningOp<memref::AllocaOp>() &&
                 dsa::isCompleteStorageViewOf(memory, origin) &&
                 storage.aliases(origin).complete
             ? origin : Value{};
}

int floatGrid(Value value, Operation *read, dsa::StorageAnalysis &storage,
              unsigned depth = 0) {
  Type type = value.getType();
  if (auto buffer = dyn_cast<MemRefType>(type)) type = buffer.getElementType();
  if (type.isF16()) return -24;
  if (type.isBF16()) return -133;
  if (!type.isF32() || depth > 32) return -149;
  auto grid = [&](Value input, Operation *before) { return floatGrid(input, before, storage, depth + 1); };
  auto binary = [&](Value lhs, Value rhs, Operation *before, bool multiply) {
    int a = grid(lhs, before), b = grid(rhs, before);
    return multiply ? std::clamp(a + b, -149, 1024) : std::min(a, b);
  };
  FloatAttr constant;
  if (matchPattern(value, m_Constant(&constant))) {
    uint32_t bits = constant.getValue().bitcastToAPInt().getZExtValue();
    uint32_t exponent = (bits >> 23) & 255, fraction = bits & 0x7fffff;
    if (exponent == 255 || (!exponent && !fraction)) return 1024;
    uint32_t significand = fraction | (exponent ? 0x800000 : 0);
    return (exponent ? int(exponent) - 150 : -149) + llvm::countr_zero(significand);
  }
  if (auto load = value.getDefiningOp<memref::LoadOp>()) return grid(load.getMemref(), load);
  if (auto cast = value.getDefiningOp<arith::ExtFOp>()) return grid(cast.getIn(), cast);
  if (auto add = value.getDefiningOp<arith::AddFOp>()) return binary(add.getLhs(), add.getRhs(), add, false);
  if (auto sub = value.getDefiningOp<arith::SubFOp>()) return binary(sub.getLhs(), sub.getRhs(), sub, false);
  if (auto mul = value.getDefiningOp<arith::MulFOp>()) return binary(mul.getLhs(), mul.getRhs(), mul, true);
  Value origin = ownedCompleteStorage(value, storage);
  if (!origin) return -149;
  Operation *writer = storage.lastWriterBefore(value, read);
  auto complete = [&](Value output) {
    return dsa::isCompleteStorageViewOf(output, origin);
  };
  if (auto fill = dyn_cast_or_null<dsa::FillOp>(writer);
      fill && complete(fill.getOutput())) return grid(fill.getValue(), fill);
  if (auto cast = dyn_cast_or_null<dsa::CastOp>(writer);
      cast && complete(cast.getOutput())) return grid(cast.getInput(), cast);
  if (auto copy = dyn_cast_or_null<memref::CopyOp>(writer);
      copy && complete(copy.getTarget())) return grid(copy.getSource(), copy);
  if (auto load = dyn_cast_or_null<dsa::LoadTileOp>(writer);
      load && complete(load.getOutput())) return grid(load.getSource(), load);
  if (auto operation = dyn_cast_or_null<dsa::BinaryOp>(writer);
      operation && complete(operation.getOutput())) {
    auto kind = operation.getKind();
    if (kind == BinaryOperator::Add || kind == BinaryOperator::Subtract || kind == BinaryOperator::Multiply ||
        kind == BinaryOperator::Maximum || kind == BinaryOperator::MaximumNum ||
        kind == BinaryOperator::Minimum || kind == BinaryOperator::MinimumNum)
      return binary(operation.getLhs(), operation.getRhs(), operation, kind == BinaryOperator::Multiply);
  }
  if (auto reduce = dyn_cast_or_null<dsa::ReduceOp>(writer);
      reduce && complete(reduce.getOutput())) {
    auto kind = reduce.getKind();
    if (kind == BinaryOperator::Add || kind == BinaryOperator::Maximum || kind == BinaryOperator::MaximumNum ||
        kind == BinaryOperator::Minimum || kind == BinaryOperator::MinimumNum)
      return std::min(grid(reduce.getInput(), reduce), grid(reduce.getIdentity(), reduce));
  }
  return -149;
}

void realizeExponentialWorkspace(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<dsa::UnaryOp> operations;
  function.walk([&](dsa::UnaryOp unary) {
    if (!unary.getScratch() &&
        ((unary.getKind() == UnaryOperator::Exp && !unary.getApproximate() && !unary.getFlushToZero()) ||
         (unary.getKind() == UnaryOperator::Exp2 && unary.getApproximate() && unary.getFlushToZero())))
      operations.push_back(unary);
  });
  for (auto unary : operations) {
    auto type = cast<MemRefType>(unary.getOutput().getType());
    if (unary.getKind() == UnaryOperator::Exp2) {
      dsa::StorageAnalysis storage(function);
      if (floatGrid(unary.getInput(), unary, storage) >= -126)
        unary->setAttr("bangc.input_non_subnormal", UnitAttr::get(function.getContext()));
      continue;
    }
    auto shape = exponentialScratchShape(type.getElementType(), type.getNumElements());
    if (shape.empty() || unary.getInput() == unary.getOutput() ||
        !unary.getInput().getDefiningOp<memref::AllocaOp>() || !unary.getOutput().getDefiningOp<memref::AllocaOp>()) continue;
    for (int64_t width = shape[1]; width >= 256; width /= 2) {
      OpBuilder b(unary);
      Value scratch = allocate(b, unary.getLoc(), type.getElementType(), {shape[0], width}, dsa::nramSpace);
      unary.getScratchMutable().assign(scratch);
      int64_t nram = 0, wram = 0;
      measureStorage(function, nram, wram);
      if (nram <= config.getLocalBytes() && nram <= 768 * 1024 && wram <= 1024 * 1024) break;
      unary.getScratchMutable().clear();
      scratch.getDefiningOp()->erase();
    }
  }
}

void realizeFlushWorkspace(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<dsa::UnaryOp> operations;
  function.walk([&](dsa::UnaryOp unary) {
    if (unary.getKind() == UnaryOperator::Exp2 && unary.getApproximate() && unary.getFlushToZero())
      operations.push_back(unary);
  });
  for (auto unary : operations) {
    auto output = cast<MemRefType>(unary.getOutput().getType());
    if (unary.getScratch() || !output.getElementType().isF32() || output.getNumElements() < 64) continue;
    for (int64_t width = std::min<int64_t>(8192, output.getNumElements()); width >= 64; width /= 2) {
      OpBuilder builder(unary);
      Value scratch = allocate(builder, unary.getLoc(), builder.getI32Type(), {1, width}, dsa::nramSpace);
      unary.getScratchMutable().assign(scratch);
      int64_t nram = 0, wram = 0;
      measureStorage(function, nram, wram);
      if (nram <= config.getLocalBytes() && nram <= 768 * 1024 && wram <= 1024 * 1024) break;
      unary.getScratchMutable().clear(); scratch.getDefiningOp()->erase();
    }
  }
}

void realizeSelections(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<dsa::SelectOp> selections;
  function.walk([&](dsa::SelectOp select) { selections.push_back(select); });
  for (auto select : selections) {
    auto output = cast<MemRefType>(select.getOutput().getType());
    auto shape = selectionScratchShape(output.getElementType(), output.getNumElements(),
                                       isa<MemRefType>(select.getFalseValue().getType()));
    if (select.getScratch() || !isa<MemRefType>(select.getCondition().getType()) || shape.empty()) continue;
    for (int64_t width = shape[1]; width >= 64; width /= 2) {
      OpBuilder builder(select);
      Value scratch = allocate(builder, select.getLoc(), builder.getI32Type(), {shape[0], width}, dsa::nramSpace);
      select.getScratchMutable().assign(scratch);
      int64_t nram = 0, wram = 0;
      measureStorage(function, nram, wram);
      if (nram <= config.getLocalBytes() && nram <= 768 * 1024 && wram <= 1024 * 1024) break;
      select.getScratchMutable().clear();
      scratch.getDefiningOp()->erase();
    }
  }
}

void realizeApproximateReciprocals(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<dsa::BinaryOp> operations;
  function.walk([&](dsa::BinaryOp binary) {
    if (binary.getKind() == BinaryOperator::TrueDivide && binary.getApproximate() &&
        binary.getFlushToZero() && !binary.getScratch()) operations.push_back(binary);
  });
  for (auto binary : operations) {
    auto type = cast<MemRefType>(binary.getOutput().getType());
    if (!type.getElementType().isF32() ||
        !isa<MemRefType>(binary.getRhs().getType()) || binary.getOutput() == binary.getRhs()) continue;
    dsa::StorageAnalysis storage(function);
    dsa::UniformMemoryAnalysis uniforms(function, storage);
    Value scalar = uniforms.read(binary.getLhs(), binary);
    FloatAttr value;
    if (!scalar || !matchPattern(scalar, m_Constant(&value)) || !value.getValue().isExactlyValue(1.0)) continue;
    for (int64_t width = std::min<int64_t>(8192, type.getNumElements()); width >= 1; width /= 2) {
      OpBuilder b(binary);
      Value scratch = allocate(b, binary.getLoc(), b.getI32Type(), {2, width}, dsa::nramSpace);
      binary.getScratchMutable().assign(scratch);
      int64_t nram = 0, wram = 0;
      measureStorage(function, nram, wram);
      if (nram <= config.getLocalBytes() && nram <= 768 * 1024 && wram <= 1024 * 1024) {
        binary->setAttr("bangc.implementation", b.getStringAttr("reciprocal_f32_ftz"));
        break;
      }
      binary.getScratchMutable().clear(); scratch.getDefiningOp()->erase();
    }
  }
}

bool retainNarrowExtremaInputs(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<dsa::ReduceOp> reductions;
  function.walk([&](dsa::ReduceOp reduce) {
    if (reduce.getAxis() == 1 && (reduce.getKind() == BinaryOperator::MaximumNum ||
        reduce.getKind() == BinaryOperator::MinimumNum)) reductions.push_back(reduce);
  });
  bool changed = false;
  for (auto reduce : reductions) {
    dsa::StorageAnalysis storage(function);
    Value oldInput = reduce.getInput(), oldScratch = reduce.getScratch();
    auto type = cast<MemRefType>(oldInput.getType());
    Value origin = ownedCompleteStorage(oldInput, storage);
    if (!type.getElementType().isF32() || type.getDimSize(0) >= 32 || type.getDimSize(1) < 4 || !origin) continue;
    auto conversion = dyn_cast_or_null<dsa::CastOp>(storage.lastWriterBefore(oldInput, reduce));
    if (!conversion || !dsa::isCompleteStorageViewOf(conversion.getOutput(), origin) ||
        conversion.getOutput().getType() != oldInput.getType() ||
        !ownedCompleteStorage(conversion.getInput(), storage) ||
        !cast<MemRefType>(conversion.getInput().getType()).getElementType().isF16() ||
        !storage.contentsUnchangedBetween(conversion.getInput(), conversion, reduce)) continue;
    OpBuilder b(reduce);
    Value scratch = allocate(b, reduce.getLoc(), b.getF16Type(), {1, type.getDimSize(1)}, dsa::nramSpace);
    reduce.getInputMutable().assign(conversion.getInput()); reduce.getScratchMutable().assign(scratch);
    int64_t nram = 0, wram = 0; measureStorage(function, nram, wram);
    if (nram > config.getLocalBytes() || nram > 768 * 1024 || wram > 1024 * 1024) {
      reduce.getInputMutable().assign(oldInput); reduce.getScratchMutable().assign(oldScratch);
      scratch.getDefiningOp()->erase(); continue;
    }
    if (oldScratch.use_empty()) oldScratch.getDefiningOp()->erase();
    changed = true;
  }
  return changed;
}

void realizeRoundedDivisions(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<dsa::BinaryOp> divisions;
  DenseMap<int64_t, Value> indicesByWidth;
  function.walk([&](dsa::BinaryOp operation) {
    if (operation.getKind() == BinaryOperator::TrueDivide && !operation.getApproximate() && !operation.getFlushToZero())
      divisions.push_back(operation);
  });
  for (auto division : divisions) {
    dsa::StorageAnalysis storage(function);
    auto type = cast<MemRefType>(division.getOutput().getType());
    if (!type.getElementType().isF32() || type.getNumElements() < 256 || type.getNumElements() % 32 ||
        !ownedCompleteStorage(division.getOutput(), storage) ||
        !ownedCompleteStorage(division.getLhs(), storage) ||
        !ownedCompleteStorage(division.getRhs(), storage) ||
        !storage.disjoint(division.getOutput(), division.getLhs()) ||
        !storage.disjoint(division.getOutput(), division.getRhs())) continue;
    dsa::UniformMemoryAnalysis uniforms(function, storage);
    Value uniform = uniforms.read(division.getRhs(), division);
    Value divisor = uniform ? uniform : division.getRhs();
    for (int64_t width = std::min<int64_t>(4096, type.getNumElements()); width >= 64; width /= 2) {
      OpBuilder builder(division);
      Location loc = division.getLoc();
      Value scratch = allocate(builder, loc, builder.getI32Type(), {12, width}, dsa::nramSpace);
      Value indices = indicesByWidth.lookup(width);
      dsa::IotaOp initialize;
      if (!indices) {
        OpBuilder initializer(function.getContext());
        initializer.setInsertionPointToStart(&function.front());
        indices = allocate(initializer, loc, initializer.getF32Type(), {1, width}, dsa::nramSpace);
        initialize = initializer.create<dsa::IotaOp>(loc, indices);
      }
      auto replacement = builder.create<dsa::DivideRNOp>(loc, division.getLhs(), divisor, division.getOutput(), scratch, indices);
      int64_t nram = 0, wram = 0;
      measureStorage(function, nram, wram);
      if (nram > config.getLocalBytes() || nram > 768 * 1024 || wram > 1024 * 1024) {
        replacement.erase(); scratch.getDefiningOp()->erase();
        if (initialize) { initialize.erase(); indices.getDefiningOp()->erase(); }
        continue;
      }
      indicesByWidth[width] = indices;
      division.erase();
      break;
    }
  }
}

void fuseNarrowDivisions(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<dsa::BinaryOp> divisions;
  DenseMap<int64_t, Value> laneIndicesBySize;
  function.walk([&](dsa::BinaryOp op) {
    if (op.getKind() == BinaryOperator::TrueDivide && !op.getApproximate() && !op.getFlushToZero())
      divisions.push_back(op);
  });
  for (auto division : divisions) {
    dsa::StorageAnalysis storage(function);
    Value quotient = division.getOutput();
    auto type = cast<MemRefType>(quotient.getType());
    if (!type.getElementType().isF32() || type.getNumElements() < 128 || type.getNumElements() % 32 ||
        type.getNumElements() > (1 << 24) ||
        !quotient.getDefiningOp<memref::AllocaOp>() ||
        !ownedCompleteStorage(division.getLhs(), storage) ||
        !ownedCompleteStorage(division.getRhs(), storage) ||
        !storage.disjoint(quotient, division.getLhs()) ||
        !storage.disjoint(quotient, division.getRhs()))
      continue;
    dsa::CastOp narrow;
    auto accesses = storage.accesses(quotient);
    bool unique = accesses.complete;
    for (const auto &entry : accesses.entries) {
      if (entry.operation == division ||
          isa<MemoryEffects::Allocate>(entry.effect.getEffect())) continue;
      auto conversion = dyn_cast<dsa::CastOp>(entry.operation);
      if (!isa<MemoryEffects::Read>(entry.effect.getEffect()) || !conversion ||
          !dsa::isCompleteStorageViewOf(conversion.getInput(), quotient) ||
          conversion.getInput().getType() != quotient.getType() ||
          (narrow && narrow != conversion)) { unique = false; break; }
      narrow = conversion;
    }
    if (!unique || !narrow || narrow->getBlock() != division->getBlock() ||
        !cast<MemRefType>(narrow.getOutput().getType()).getElementType().isF16()) continue;
    if (!ownedCompleteStorage(narrow.getOutput(), storage) ||
        !storage.disjoint(narrow.getOutput(), division.getLhs()) ||
        !storage.disjoint(narrow.getOutput(), division.getRhs())) continue;
    // No quotient consumer or intervening effect may observe the internal
    // approximation. The fused implementation certifies the final f16 value.
    Operation *next = division->getNextNode();
    for (; next && next != narrow.getOperation(); next = next->getNextNode()) {
      auto effects = storage.effects(next);
      if (!effects.complete || effects.ordered ||
          llvm::any_of(effects.entries, [](const auto &entry) {
            return !isa<MemoryEffects::Allocate>(entry.effect.getEffect());
          })) break;
    }
    if (next != narrow.getOperation()) continue;
    dsa::UniformMemoryAnalysis uniforms(function, storage);
    Value uniform = uniforms.read(division.getRhs(), division);
    Value divisor = uniform ? uniform : division.getRhs();
    OpBuilder builder(narrow);
    Location loc = narrow.getLoc();
    Value bounds = allocate(builder, loc, builder.getF32Type(), type.getShape(), dsa::nramSpace);
    Value accepted = allocate(builder, loc, builder.getF32Type(), type.getShape(), dsa::nramSpace);
    Value narrowBounds = allocate(builder, loc, builder.getF16Type(), type.getShape(), dsa::nramSpace);
    int64_t lanes = type.getNumElements();
    Value laneIndices = laneIndicesBySize.lookup(lanes);
    dsa::IotaOp initializeIndices;
    if (!laneIndices) {
      OpBuilder initializer(function.getContext());
      initializer.setInsertionPointToStart(&function.front());
      laneIndices = allocate(initializer, loc, initializer.getF32Type(), {1, lanes}, dsa::nramSpace);
      initializeIndices = initializer.create<dsa::IotaOp>(loc, laneIndices);
    }
    auto fused = builder.create<dsa::DivideCastOp>(loc, division.getLhs(), divisor, narrow.getOutput(),
        quotient, bounds, accepted, narrowBounds, laneIndices);
    // Include the extended input lifetimes in the same allocation calculation
    // used for the final program. Retain ordinary divide/cast when the selected
    // local implementation would exceed the task's storage budget.
    int64_t nram = 0, wram = 0;
    measureStorage(function, nram, wram);
    if (nram > config.getLocalBytes() || nram > 768 * 1024 || wram > 1024 * 1024) {
      fused.erase();
      narrowBounds.getDefiningOp()->erase();
      accepted.getDefiningOp()->erase();
      bounds.getDefiningOp()->erase();
      if (initializeIndices) {
        initializeIndices.erase();
        laneIndices.getDefiningOp()->erase();
      }
      continue;
    }
    laneIndicesBySize[lanes] = laneIndices;
    narrow.erase();
    division.erase();
  }
}

LogicalResult realizeNativeComputations(ModuleOp module) {
  auto function = *module.getOps<func::FuncOp>().begin();
  auto config = function->getAttrOfType<dsa::ConfigurationAttr>("intent_dsa.configuration");
  SmallVector<dsa::MatMulOp> matrices;
  function.walk([&](dsa::MatMulOp op) { matrices.push_back(op); });
  for (auto matrix : matrices) if (failed(realizeMatMul(matrix, config))) return failure();
  // Operations without a full-domain native storage-dtype implementation use
  // f32 local arithmetic and an explicit cast back to the declared dtype.
  SmallVector<Operation *> promotedPointwise;
  function.walk([&](Operation *op) {
    Value output;
    if (auto unary = dyn_cast<dsa::UnaryOp>(op)) output = unary.getOutput();
    if (auto binary = dyn_cast<dsa::BinaryOp>(op)) output = binary.getOutput();
    if (!output) return;
    Type element = cast<MemRefType>(output.getType()).getElementType();
    if (arithmeticStorageType(element) != element) promotedPointwise.push_back(op);
  });
  for (Operation *op : promotedPointwise) {
    OpBuilder builder(op);
    auto convert = [&](Value source) -> Value {
      if (source.getType().isBF16())
        return builder.create<arith::ExtFOp>(op->getLoc(), arithmeticStorageType(source.getType()), source);
      auto type = cast<MemRefType>(source.getType());
      Value result = allocate(builder, op->getLoc(), arithmeticStorageType(type.getElementType()), type.getShape(), dsa::nramSpace);
      builder.create<dsa::CastOp>(op->getLoc(), source, result);
      return result;
    };
    Value destination;
    if (auto unary = dyn_cast<dsa::UnaryOp>(op)) {
      destination = unary.getOutput();
      Value input = convert(unary.getInput());
      auto inputType = cast<MemRefType>(input.getType());
      Value output = allocate(builder, op->getLoc(), inputType.getElementType(), inputType.getShape(), dsa::nramSpace);
      unary.getInputMutable().assign(input); unary.getOutputMutable().assign(output);
      builder.setInsertionPointAfter(op);
      builder.create<dsa::CastOp>(op->getLoc(), output, destination);
    } else {
      auto binary = cast<dsa::BinaryOp>(op);
      destination = binary.getOutput();
      Value lhs = convert(binary.getLhs()), rhs = convert(binary.getRhs());
      auto inputType = cast<MemRefType>(lhs.getType());
      Value output = allocate(builder, op->getLoc(), inputType.getElementType(), inputType.getShape(), dsa::nramSpace);
      binary.getLhsMutable().assign(lhs); binary.getRhsMutable().assign(rhs); binary.getOutputMutable().assign(output);
      builder.setInsertionPointAfter(op);
      builder.create<dsa::CastOp>(op->getLoc(), output, destination);
    }
  }
  // The hardware activation approximations truncate part of sigmoid's domain.
  // Keep exp(-abs(x)) and a sign-selected numerator explicit instead.
  SmallVector<dsa::UnaryOp> sigmoids;
  function.walk([&](dsa::UnaryOp op) { if (op.getKind() == UnaryOperator::Sigmoid) sigmoids.push_back(op); });
  for (auto sigmoid : sigmoids) if (failed(realizeSigmoid(sigmoid))) return failure();
  SmallVector<Operation *> bf16Scalar;
  function.walk([&](Operation *op) {
    if (isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp, arith::NegFOp,
            arith::MaximumFOp, arith::MinimumFOp, arith::MaxNumFOp, arith::MinNumFOp, arith::CmpFOp,
            math::ExpOp, math::Exp2Op, math::LogOp, math::SqrtOp, math::RsqrtOp, math::TanhOp, math::AbsFOp,
            math::SinOp, math::CosOp, math::FloorOp>(op) &&
        llvm::any_of(op->getOperandTypes(), [](Type type) { return type.isBF16(); })) bf16Scalar.push_back(op);
  });
  for (Operation *op : bf16Scalar) {
    OpBuilder builder(op);
    SmallVector<Value> arguments;
    for (Value input : op->getOperands())
      arguments.push_back(input.getType().isBF16() ? Value(builder.create<arith::ExtFOp>(op->getLoc(), builder.getF32Type(), input)) : input);
    OperationState state(op->getLoc(), op->getName());
    state.addOperands(arguments); state.addAttributes(op->getAttrs());
    Type original = op->getResult(0).getType();
    state.addTypes(original.isBF16() ? builder.getF32Type() : original);
    Value result = builder.create(state)->getResult(0);
    if (original.isBF16()) result = builder.create<arith::TruncFOp>(op->getLoc(), original, result);
    op->getResult(0).replaceAllUsesWith(result);
    op->erase();
  }
  return success();
}

LogicalResult realizeNativeWorkspace(ModuleOp module,
    llvm::function_ref<LogicalResult()> cleanup) {
  auto function = *module.getOps<func::FuncOp>().begin();
  auto config = function->getAttrOfType<dsa::ConfigurationAttr>("intent_dsa.configuration");
  dsa::bindUniformOperands(function);
  {
    SmallVector<Operation *> arithmetic;
    function.walk([&](Operation *op) {
      if (op->getName().getDialectNamespace() == "arith" && op->getNumResults() == 1 &&
          !isa<arith::ConstantOp>(op) &&
          (op->getResult(0).getType().isIndex() || op->getResult(0).getType().isInteger(64))) arithmetic.push_back(op);
    });
    for (Operation *op : arithmetic) {
      auto interval = integerInterval(op->getResult(0), function);
      if (!interval || interval->first != interval->second) continue;
      OpBuilder b(op);
      Value value = b.create<arith::ConstantOp>(op->getLoc(), b.getIntegerAttr(op->getResult(0).getType(), interval->first));
      op->getResult(0).replaceAllUsesWith(value); op->erase();
    }
    if (failed(cleanup())) return failure();
  }
  reuseConsumedBinaryInputs(function, config);
  reuseConsumedExp2Inputs(function, config);
  if (batchIndependentRowPrograms(function, config) && failed(cleanup()))
    return failure();
  // Native composition reads actual storage versions. Remove fully overwritten
  // initialization before it can obscure a producer's unique value consumer.
  if (failed(composeLocalStorage(function, cleanup))) return failure();
  fuseNarrowDivisions(function, config);
  realizeRoundedDivisions(function, config);
  realizeApproximateReciprocals(function, config);
  dsa::bindUniformOperands(function);
  realizeSelections(function, config);
  realizeRowSumChannels(function, config);
  if (failed(realizeNumericExtremaWorkspace(function, config))) return failure();
  realizeCompareWorkspace(function, config);
  realizeExponentialWorkspace(function, config);
  realizeFlushWorkspace(function, config);
  return composeLocalStorage(function, cleanup);
}

LogicalResult selectNativeImplementations(ModuleOp module) {
  auto function = *module.getOps<func::FuncOp>().begin();
  auto walk = function.walk([&](Operation *op) {
    if (auto unary = dyn_cast<dsa::UnaryOp>(op)) {
      bool approximateExp2 = unary.getKind() == UnaryOperator::Exp2 && unary.getApproximate() &&
          cast<MemRefType>(unary.getInput().getType()).getElementType().isF32();
      if (approximateExp2) op->setAttr("bangc.implementation", StringAttr::get(module.getContext(), "exp2_f32"));
      else if (!supportedUnary(unary.getKind()) || unary.getApproximate() || unary.getFlushToZero()) {
        op->emitError("unary numerical mode has no selected BANG C implementation"); return WalkResult::interrupt();
      }
      Type element = cast<MemRefType>(unary.getInput().getType()).getElementType();
      if (!unary.getApproximate() && (element.isF32() || element.isF16())) {
        StringRef callee;
        switch (unary.getKind()) {
        case UnaryOperator::Exp: callee = "__cn_vector_exp"; break;
        case UnaryOperator::Exp2: callee = "__cn_vector_exp2"; break;
        case UnaryOperator::Log: callee = "__cn_vector_log"; break;
        case UnaryOperator::Tanh: callee = "__cn_vector_tanh"; break;
        case UnaryOperator::Sqrt: callee = "__cn_vector_sqrt"; break;
        case UnaryOperator::Rsqrt: callee = "__cn_vector_rsqrt"; break;
        case UnaryOperator::Sin: callee = "__cn_vector_sin"; break;
        case UnaryOperator::Cos: callee = "__cn_vector_cos"; break;
        case UnaryOperator::Floor: callee = "__cn_vector_floor"; break;
        default: break;
        }
        if (!callee.empty()) op->setAttr("bangc.callee", StringAttr::get(module.getContext(), callee.str() + (element.isF16() ? "_f16" : "_f32")));
      }
    }
    if (auto binary = dyn_cast<dsa::BinaryOp>(op)) {
      Type element = cast<MemRefType>(binary.getLhs().getType()).getElementType();
      if (auto rhs = dyn_cast<MemRefType>(binary.getRhs().getType());
          rhs && rhs != binary.getLhs().getType()) {
        StringRef implementation = rhs.getDimSize(0) == 1 ? "cycle" : "row_scalar";
        op->setAttr("bangc.implementation", StringAttr::get(module.getContext(), implementation));
      }
      if (!isa<MemRefType>(binary.getRhs().getType()) &&
          (!supportedScalarBinary(binary.getKind()) || (!element.isF16() && !element.isF32() && !element.isInteger(32)))) {
        op->emitError("scalar binary kind or dtype has no selected BANG C implementation"); return WalkResult::interrupt();
      }
      bool division = binary.getKind() == BinaryOperator::TrueDivide &&
          cast<MemRefType>(binary.getLhs().getType()).getElementType().isF32();
      if (!supportedBinary(binary.getKind()) || ((binary.getApproximate() || binary.getFlushToZero()) && !division)) {
        op->emitError("binary numerical mode has no selected BANG C implementation"); return WalkResult::interrupt();
      }
      if (binary.getKind() == BinaryOperator::TrueDivide && !op->hasAttr("bangc.implementation"))
        op->setAttr("bangc.implementation", StringAttr::get(module.getContext(), division ? "divide_f32" : "scalar_divide"));
      if (binary.getKind() == BinaryOperator::Maximum || binary.getKind() == BinaryOperator::Minimum) {
        if (binary.getScratch())
          op->setAttr("bangc.implementation", StringAttr::get(module.getContext(), "propagating_extrema"));
        else if (StringRef callee = directExtremaCallee(binary); !callee.empty())
          op->setAttr("bangc.callee", StringAttr::get(module.getContext(), callee));
        else {
          op->emitError("propagating extrema requires an independent NaN-repair workspace");
          return WalkResult::interrupt();
        }
      }
    }
    if (auto reduction = dyn_cast<dsa::ReduceOp>(op)) {
      auto input = cast<MemRefType>(reduction.getInput().getType());
      bool rowBatch = reduction.getAxis() == 1 && input.getDimSize(0) > 1;
      if (!input.getElementType().isF32() ||
          (reduction.getAxis() == 0 && reduction.getKind() != BinaryOperator::Add) ||
          (rowBatch && input.getDimSize(1) >= 65536) ||
          (reduction.getKind() != BinaryOperator::Add && reduction.getKind() != BinaryOperator::Maximum &&
           reduction.getKind() != BinaryOperator::MaximumNum && reduction.getKind() != BinaryOperator::Minimum &&
           reduction.getKind() != BinaryOperator::MinimumNum)) {
        op->emitError("reduction has no selected BANG C f32 local implementation"); return WalkResult::interrupt();
      }
    }
    if (auto compare = dyn_cast<dsa::CompareOp>(op)) {
      if (!cast<MemRefType>(compare.getLhs().getType()).getElementType().isInteger(64)) {
        op->emitError("comparison has no selected BANG C tile implementation"); return WalkResult::interrupt();
      }
      static constexpr const char *predicates[] = {"eq", "ne", "lt", "le", "gt", "ge"};
      op->setAttr("bangc.callee", StringAttr::get(module.getContext(),
          std::string("__cn_vector_") + predicates[static_cast<unsigned>(compare.getPredicate())] + "_s64"));
    }
    if (auto conversion = dyn_cast<dsa::CastOp>(op)) {
      auto source = cast<MemRefType>(conversion.getInput().getType());
      Type destination = cast<MemRefType>(conversion.getOutput().getType()).getElementType();
      if (source.getNumElements() >= 64) {
        StringRef callee;
        if (source.getElementType().isSignlessInteger(32) && destination.isSignlessInteger(64))
          callee = "__cn_vector_cast_s32_to_s64";
        if (source.getElementType().isSignlessInteger(64) && destination.isSignlessInteger(32))
          callee = "__cn_vector_cast_s64_to_s32";
        if (!callee.empty()) op->setAttr("bangc.callee", StringAttr::get(module.getContext(), callee));
      }
    }
    if (auto matrix = dyn_cast<dsa::MatrixTileOp>(op)) {
      op->setAttr("bangc.implementation", StringAttr::get(module.getContext(), "matmul_local_f32_accumulator"));
      dsa::StorageAnalysis storage(function);
      Value memory = ownedCompleteStorage(matrix.getRhs(), storage);
      auto owner = memory ? memory.getDefiningOp<memref::AllocaOp>() : memref::AllocaOp();
      if (!owner) { op->emitError("matrix input has no owned prepared storage"); return WalkResult::interrupt(); }
      owner->setAttr("bangc.layout", StringAttr::get(module.getContext(), "matrix_filter_interleaved64"));
    }
    return WalkResult::advance();
  });
  if (walk.wasInterrupted()) return failure();
  return success();
}
} // namespace intent::bangc
