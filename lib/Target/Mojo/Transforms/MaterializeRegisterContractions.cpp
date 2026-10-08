#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "Intent/Dialect/CPU/IR/CPUAttrs.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "Intent/Target/Mojo/Transforms/Passes.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Intent/Dialect/CPU/Transforms/Implementation/ImplementationInputs.h"
#include "Intent/Dialect/CPU/Transforms/Structure/LoopBuilders.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;

namespace intent::mojo {
using namespace intent::cpu;
namespace {

struct LocalEpilogue {
  memref::AllocaOp allocation;
  linalg::FillOp initialization;
  linalg::GenericOp consumer;
};

std::optional<LocalEpilogue> localEpilogue(linalg::GenericOp contraction) {
  Value partial = contraction.getOutputs()[0];
  auto allocation = partial.getDefiningOp<memref::AllocaOp>();
  auto initialization = dyn_cast_or_null<linalg::FillOp>(contraction->getPrevNode());
  auto consumer = dyn_cast_or_null<linalg::GenericOp>(contraction->getNextNode());
  if (!allocation || !initialization || !consumer ||
      initialization.getOutputs().size() != 1 || initialization.getOutputs()[0] != partial ||
      consumer.getOutputs().size() != 1 || consumer.getNumResults() ||
      consumer.getOutputs()[0] == partial || !llvm::is_contained(consumer.getInputs(), partial) ||
      consumer.getNumReductionLoops() ||
      !llvm::all_of(consumer.getIndexingMapsArray(), [](AffineMap map) { return map.isIdentity(); }) ||
      !consumer.getRegion().front().getArguments().back().use_empty()) return std::nullopt;
  for (Operation *user : partial.getUsers())
    if (user != initialization && user != contraction && user != consumer) return std::nullopt;
  auto shape = cast<MemRefType>(partial.getType()).getShape();
  Type element = cast<MemRefType>(partial.getType()).getElementType();
  StorageAnalysis storage(contraction->getParentOfType<func::FuncOp>());
  Value destination = consumer.getOutputs()[0];
  for (Value memory : consumer->getOperands()) {
    auto type = dyn_cast<MemRefType>(memory.getType());
    if (!type || type.getShape() != shape || type.getElementType() != element) return std::nullopt;
    if (memory != destination && !storage.disjoint(memory, destination))
      return std::nullopt;
  }
  for (Operation &operation : consumer.getRegion().front().without_terminator())
    if (operation.getNumRegions() || operation.getNumResults() != 1 ||
        operation.getResult(0).getType() != element || !isMemoryEffectFree(&operation)) return std::nullopt;
  return LocalEpilogue{allocation, initialization, consumer};
}

constexpr int64_t indexedRowWindow = 32;

struct IndexedContraction {
  Value lhs, output;
  memref::LoadOp load;
  arith::ExtFOp rightWiden;
  math::FmaOp fma;
};

std::optional<IndexedContraction> indexedContraction(linalg::GenericOp operation) {
  if (operation.getNumDpsInputs() != 1 || operation.getNumDpsInits() != 1 || operation.getNumResults())
    return std::nullopt;
  AffineExpr m, k, n;
  bindDims(operation.getContext(), m, k, n);
  if (operation.getIndexingMapsArray() != SmallVector<AffineMap>{
          AffineMap::get(3, 0, {m, k}, operation.getContext()),
          AffineMap::get(3, 0, {m, n}, operation.getContext())} ||
      operation.getIteratorTypesArray() != SmallVector<utils::IteratorType>{
          utils::IteratorType::parallel, utils::IteratorType::reduction, utils::IteratorType::parallel})
    return std::nullopt;
  Value lhs = operation.getInputs()[0], output = operation.getOutputs()[0];
  auto lhsType = dyn_cast<MemRefType>(lhs.getType()), outputType = dyn_cast<MemRefType>(output.getType());
  if (!lhsType || !outputType || !outputType.getElementType().isF32()) return std::nullopt;
  SmallVector<int64_t> outputStrides;
  int64_t outputOffset;
  if (failed(outputType.getStridesAndOffset(outputStrides, outputOffset)) || outputStrides.back() != 1)
    return std::nullopt;
  Block &body = operation.getRegion().front();
  auto fma = body.getTerminator()->getOperand(0).getDefiningOp<math::FmaOp>();
  if (!fma || fma.getC() != body.getArgument(1) || !body.getArgument(1).hasOneUse()) return std::nullopt;
  Value left = fma.getA(), right = fma.getB();
  if (auto widen = left.getDefiningOp<arith::ExtFOp>()) left = widen.getIn();
  if (left != body.getArgument(0)) return std::nullopt;
  auto rightWiden = right.getDefiningOp<arith::ExtFOp>();
  if (rightWiden) right = rightWiden.getIn();
  auto load = right.getDefiningOp<memref::LoadOp>();
  if (!load || load->getBlock() != &body || load.getIndices().size() != 2 || !right.hasOneUse() ||
      (rightWiden && !rightWiden.getResult().hasOneUse())) return std::nullopt;
  Value column = load.getIndices()[1];
  auto columnIndex = column.getDefiningOp<linalg::IndexOp>();
  if (!columnIndex || columnIndex.getDim() != 2 || !column.hasOneUse() || load.getIndices()[0] == column)
    return std::nullopt;
  Value rhs = load.getMemref();
  if (rhs.getParentBlock() == &body) return std::nullopt;
  auto function = operation->getParentOfType<func::FuncOp>();
  StorageAnalysis storage(function);
  auto independent = [&](Value input) {
    return storage.disjoint(input, output);
  };
  if (!independent(lhs) || !storage.isReadOnly(rhs)) return std::nullopt;
  for (Operation &instruction : body.without_terminator()) {
    if (auto coordinate = dyn_cast<linalg::IndexOp>(instruction);
        coordinate && coordinate.getDim() == 2 && coordinate.getResult() != column) return std::nullopt;
    if (auto read = dyn_cast<memref::LoadOp>(instruction)) {
      if (!independent(read.getMemref())) return std::nullopt;
    } else if (!isMemoryEffectFree(&instruction)) return std::nullopt;
    if (instruction.getNumRegions() || instruction.getNumResults() != 1) return std::nullopt;
    if (&instruction == load || &instruction == rightWiden || &instruction == fma) continue;
    for (Value input : instruction.getOperands())
      if (input == column || input == body.getArgument(1) || input == right ||
          input == fma.getResult() || (rightWiden && input == rightWiden.getResult())) return std::nullopt;
  }
  return IndexedContraction{lhs, output, load, rightWiden, fma};
}

FailureOr<bool> materializeIndexedContraction(linalg::GenericOp operation, ImplementationInputs &inputs) {
  auto contraction = indexedContraction(operation);
  if (!contraction) return false;
  auto [lhs, output, load, rightWiden, fma] = *contraction;
  Value rhs = load.getMemref();
  auto outputType = cast<MemRefType>(output.getType());
  Block &body = operation.getRegion().front();
  auto binding = operation->getAttrOfType<ImplementationAttr>("intent_cpu.implementation");
  if (!binding) return false;
  int64_t width = implementationParameter(binding, "vector_width");
  int64_t replicas = implementationParameter(binding, "register_replicas");
  int64_t panelSize = width * replicas;
  InputRequirement requirement{1, outputType.getElementType(), 1, panelSize, width * 4, InputReuse::Consumers, panelSize};
  OpBuilder b(operation);
  Location loc = operation.getLoc();
  Value zero = index(b, loc, 0), one = index(b, loc, 1);
  Value rows = b.create<memref::DimOp>(loc, lhs, 0);
  Value depth = b.create<memref::DimOp>(loc, lhs, 1);
  Value columns = b.create<memref::DimOp>(loc, output, 1);
  Value rhsRows = b.create<memref::DimOp>(loc, rhs, 0);
  Value panelWidth = index(b, loc, panelSize);
  Value nonempty = b.create<arith::AndIOp>(loc,
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, depth, zero),
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, columns, zero));
  nonempty = b.create<arith::AndIOp>(loc, nonempty,
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, rows, zero));
  auto active = b.create<scf::IfOp>(loc, nonempty, false);
  operation->moveBefore(active.thenBlock()->getTerminator());
  auto supplied = inputs.prepareCaptured(operation, load, requirement);
  if (failed(supplied)) return failure();
  b.setInsertionPoint(operation);
  constexpr int64_t rowWindow = indexedRowWindow;
  // Bound gathered RHS payload per row to 16 KiB and preparation to 192 KiB.
  int64_t reductionWindow = std::min<int64_t>(512, 4096 / panelSize);
  loop(b, loc, zero, rows, rowWindow, [&](Value rowBegin) {
    Value rowEnd = b.create<arith::MinSIOp>(loc, add(b, loc, rowBegin, index(b, loc, rowWindow)), rows);
    Value rowCount = b.create<arith::SubIOp>(loc, rowEnd, rowBegin);
    Value leftSupply = b.create<memref::AllocaOp>(loc,
        MemRefType::get({rowWindow, reductionWindow}, outputType.getElementType()));
    Value indexSupply = b.create<memref::AllocaOp>(loc,
        MemRefType::get({rowWindow, reductionWindow}, b.getIndexType()));
    auto panel = [&](Value begin, Value reductionCount, int64_t lanes, int64_t count) {
      SmallVector<OpFoldResult> origins{
          b.create<arith::DivSIOp>(loc, begin, panelWidth).getResult(), b.getIndexAttr(0), b.getIndexAttr(0)};
      SmallVector<OpFoldResult> sizes{b.getIndexAttr(1), rhsRows, b.getIndexAttr(panelSize)};
      SmallVector<OpFoldResult> strides(3, b.getIndexAttr(1));
      auto viewType = cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
          {ShapedType::kDynamic, panelSize}, cast<MemRefType>(supplied->storage.getType()), origins, sizes, strides));
      Value rightMemory = b.create<memref::SubViewOp>(loc, viewType, supplied->storage, origins, sizes, strides);
      Value rightBegin = b.create<arith::RemSIOp>(loc, begin, panelWidth);
      Type scalar = outputType.getElementType();
      Type accumulator = lanes == 1 ? scalar : Type(VectorType::get({lanes}, scalar));
      SmallVector<Value> offsets, rightOffsets;
      for (int64_t replica = 0; replica < count; ++replica) {
        offsets.push_back(add(b, loc, begin, index(b, loc, replica * lanes)));
        rightOffsets.push_back(add(b, loc, rightBegin, index(b, loc, replica * lanes)));
      }
      loop(b, loc, zero, rowCount, 1, [&](Value rowOrdinal) {
        Value row = add(b, loc, rowBegin, rowOrdinal);
        SmallVector<Value> initials;
        for (Value offset : offsets)
          initials.push_back(lanes == 1
              ? Value(b.create<memref::LoadOp>(loc, output, ValueRange{row, offset}))
              : Value(b.create<vector::LoadOp>(loc, cast<VectorType>(accumulator), output, ValueRange{row, offset})));
        auto reduction = b.create<scf::ForOp>(loc, zero, reductionCount, one, initials);
        {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(reduction.getBody());
          Value leftValue = b.create<memref::LoadOp>(loc, leftSupply, ValueRange{rowOrdinal, reduction.getInductionVar()});
          if (lanes != 1) leftValue = b.create<vector::BroadcastOp>(loc, cast<VectorType>(accumulator), leftValue);
          Value logicalK = b.create<memref::LoadOp>(loc, indexSupply, ValueRange{rowOrdinal, reduction.getInductionVar()});
          SmallVector<Value> next;
          for (auto [replica, offset] : llvm::enumerate(rightOffsets)) {
            Value rightValue = lanes == 1
                ? Value(b.create<memref::LoadOp>(loc, rightMemory, ValueRange{logicalK, offset}))
                : Value(b.create<vector::LoadOp>(loc, cast<VectorType>(accumulator),
                                                rightMemory, ValueRange{logicalK, offset}));
            auto update = b.create<math::FmaOp>(loc, leftValue, rightValue, reduction.getRegionIterArgs()[replica]);
            update->setAttrs(fma->getAttrs());
            next.push_back(update);
          }
          b.create<scf::YieldOp>(loc, next);
        }
        for (auto [value, offset] : llvm::zip(reduction.getResults(), offsets)) {
          if (lanes == 1) b.create<memref::StoreOp>(loc, value, output, ValueRange{row, offset});
          else b.create<vector::StoreOp>(loc, value, output, ValueRange{row, offset});
        }
      });
    };
    Value full = b.create<arith::SubIOp>(loc, columns, b.create<arith::RemSIOp>(loc, columns, panelWidth));
    // Continue each output's ascending FMA chain while sharing each RHS panel
    // across the independent rows of the implementation window.
    loop(b, loc, zero, depth, reductionWindow, [&](Value reductionBegin) {
      Value reductionEnd = b.create<arith::MinSIOp>(loc,
          add(b, loc, reductionBegin, index(b, loc, reductionWindow)), depth);
      Value reductionCount = b.create<arith::SubIOp>(loc, reductionEnd, reductionBegin);
      loop(b, loc, zero, rowCount, 1, [&](Value rowOrdinal) {
        Value row = add(b, loc, rowBegin, rowOrdinal);
        loop(b, loc, zero, reductionCount, 1, [&](Value ordinal) {
          Value k = add(b, loc, reductionBegin, ordinal);
          IRMapping mapping;
          mapping.map(body.getArgument(0), b.create<memref::LoadOp>(loc, lhs, ValueRange{row, k}).getResult());
          for (Operation &instruction : body.without_terminator()) {
            if (&instruction == load || &instruction == rightWiden || &instruction == fma) continue;
            if (auto coordinate = dyn_cast<linalg::IndexOp>(instruction)) {
              if (coordinate.getDim() != 2)
                mapping.map(coordinate.getResult(), coordinate.getDim() == 0 ? row : k);
            } else b.clone(instruction, mapping);
          }
          b.create<memref::StoreOp>(loc, mapping.lookupOrDefault(fma.getA()), leftSupply, ValueRange{rowOrdinal, ordinal});
          b.create<memref::StoreOp>(loc, mapping.lookupOrDefault(load.getIndices()[0]), indexSupply,
                                    ValueRange{rowOrdinal, ordinal});
        });
      });
      loop(b, loc, zero, full, panelSize, [&](Value begin) { panel(begin, reductionCount, width, replicas); });
      loop(b, loc, full, columns, 1, [&](Value begin) { panel(begin, reductionCount, 1, 1); });
    });
  });
  operation.erase();
  return true;
}

}

int64_t registerContractionRows(linalg::GenericOp operation) {
  return indexedContraction(operation) ? indexedRowWindow : 1;
}

LogicalResult materializeRegisterContractions(func::FuncOp function) {
  ImplementationInputs inputs(function);
  SmallVector<linalg::GenericOp> operations;
  function.walk([&](linalg::GenericOp operation) {
    if (operation->hasAttr("intent_cpu.microtile") || operation.getNumReductionLoops())
      operations.push_back(operation);
  });
  for (auto operation : operations) {
    auto indexed = materializeIndexedContraction(operation, inputs);
    if (failed(indexed)) return failure();
    if (*indexed) continue;
    auto tile = operation->getAttrOfType<MicrotileAttr>("intent_cpu.microtile");
    if (!tile) continue;
    auto outputType = cast<MemRefType>(operation.getOutputs()[0].getType());
    if (!isMatrixContraction(operation) || outputType.getShape() !=
        ArrayRef<int64_t>({tile.getRows(), tile.getColumns()}))
      return operation.emitError("CPU register contraction does not match its formed microtile");
    Value lhs = operation.getInputs()[0], rhs = operation.getInputs()[1];
    Value output = operation.getOutputs()[0];
    auto epilogue = localEpilogue(operation);
    OpBuilder b(operation);
    Location loc = operation.getLoc();
    int64_t width = tile.getVectorWidth(), columns = tile.getColumns() / width;
    Type accumulator = outputType.getElementType();
    auto vectorType = VectorType::get({width}, accumulator);
    auto binding = operation->getAttrOfType<ImplementationAttr>("intent_cpu.implementation");
    if (!binding) return operation.emitError("CPU register contraction requires a selected implementation");
    auto exactChunk = dyn_cast_or_null<IntegerAttr>(binding.getParameters().get("exact_f32_chunk"));
    if (exactChunk && (!accumulator.isInteger(32) ||
        !cast<MemRefType>(lhs.getType()).getElementType().isInteger(8) ||
        !cast<MemRefType>(rhs.getType()).getElementType().isInteger(8) ||
        exactChunk.getInt() <= 0 || exactChunk.getInt() > 1024))
      return operation.emitError("exact f32 integer contraction requires signed i8 operands, i32 state and at most 1024 products per chunk");
    Type computation = exactChunk ? b.getF32Type() : accumulator;
    auto computationVector = VectorType::get({width}, computation);
    auto widen = [&](Value value, Type type) -> Value {
      if (value.getType() == type) return value;
      if (isa<FloatType>(getElementTypeOrSelf(type)) &&
          isa<IntegerType>(getElementTypeOrSelf(value.getType())))
        return b.create<arith::SIToFPOp>(loc, type, value);
      if (isa<FloatType>(accumulator)) return b.create<arith::ExtFOp>(loc, type, value);
      return b.create<arith::ExtSIOp>(loc, type, value);
    };
    SmallVector<Value> rows, offsets, accumulators;
    for (int64_t row = 0; row < tile.getRows(); ++row) rows.push_back(index(b, loc, row));
    for (int64_t column = 0; column < columns; ++column) offsets.push_back(index(b, loc, column * width));
    auto load = [&](Value source, Value row, Value column, Type target) -> Value {
      Type element = cast<MemRefType>(source.getType()).getElementType();
      Value value = width == 1
          ? Value(b.create<memref::LoadOp>(loc, source, ValueRange{row, column}))
          : Value(b.create<vector::LoadOp>(loc, VectorType::get({width}, element), source, ValueRange{row, column}));
      return widen(value, width == 1 ? target : Type(VectorType::get({width}, target)));
    };
    for (Value row : rows)
      for (Value offset : offsets) {
        if (!epilogue) accumulators.push_back(load(output, row, offset, accumulator));
        else {
          Value initial = epilogue->initialization.getInputs()[0];
          accumulators.push_back(width == 1 ? initial
              : Value(b.create<vector::BroadcastOp>(loc, vectorType, initial)));
        }
      }
    Value depth = b.create<memref::DimOp>(loc, lhs, 1);
    auto reduce = [&](Value base, Value count, ValueRange initials) {
      auto reduction = b.create<scf::ForOp>(loc, index(b, loc, 0), count, index(b, loc, 1), initials);
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(reduction.getBody());
      Value k = add(b, loc, base, reduction.getInductionVar());
      if (width > 1) {
        Value last = b.create<arith::SubIOp>(loc, depth, index(b, loc, 1));
        Value ahead = b.create<arith::MinSIOp>(loc, add(b, loc, k, index(b, loc, 4)), last);
        for (Value offset : offsets)
          b.create<memref::PrefetchOp>(loc, rhs, ValueRange{ahead, offset}, false, 3, true);
      }
      SmallVector<Value> right;
      for (Value offset : offsets) right.push_back(load(rhs, k, offset, computation));
      SmallVector<Value> next;
      for (auto [number, row] : llvm::enumerate(rows)) {
        Value left = b.create<memref::LoadOp>(loc, lhs, ValueRange{row, k});
        left = widen(left, computation);
        if (width != 1) left = b.create<vector::BroadcastOp>(loc, computationVector, left);
        for (int64_t column = 0; column < columns; ++column) {
          Value previous = reduction.getRegionIterArgs()[number * columns + column];
          if (isa<FloatType>(computation))
            next.push_back(b.create<math::FmaOp>(loc, left, right[column], previous));
          else
            next.push_back(b.create<arith::AddIOp>(loc,
                b.create<arith::MulIOp>(loc, left, right[column]), previous));
        }
      }
      b.create<scf::YieldOp>(loc, next);
      return reduction;
    };
    SmallVector<Value> results;
    if (exactChunk) {
      // Each signed i8 product has magnitude at most 2^14. In <= 2^10
      // consecutive products every prefix is an exactly representable f32
      // integer (magnitude <= 2^24). Keep prior i32 state out of this sum.
      Value zero = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(0));
      if (width != 1) zero = b.create<vector::BroadcastOp>(loc, computationVector, zero);
      SmallVector<Value> initials(accumulators.size(), zero);
      Value limit = index(b, loc, exactChunk.getInt());
      auto chunks = b.create<scf::ForOp>(loc, index(b, loc, 0), depth, limit, accumulators);
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(chunks.getBody());
        Value base = chunks.getInductionVar();
        Value count = b.create<arith::MinSIOp>(loc, limit, b.create<arith::SubIOp>(loc, depth, base));
        auto partial = reduce(base, count, initials);
        SmallVector<Value> next;
        for (auto [value, previous] : llvm::zip(partial.getResults(), chunks.getRegionIterArgs())) {
          Value integer = b.create<arith::FPToSIOp>(loc, previous.getType(), value);
          next.push_back(b.create<arith::AddIOp>(loc, previous, integer));
        }
        b.create<scf::YieldOp>(loc, next);
      }
      llvm::append_range(results, chunks.getResults());
    } else {
      llvm::append_range(results, reduce(index(b, loc, 0), depth, accumulators).getResults());
    }
    for (auto [number, row] : llvm::enumerate(rows))
      for (int64_t column = 0; column < columns; ++column) {
        Value value = results[number * columns + column];
        Value destination = output;
        if (epilogue) {
          IRMapping mapping;
          Block &body = epilogue->consumer.getRegion().front();
          for (auto [position, input] : llvm::enumerate(epilogue->consumer.getInputs()))
            mapping.map(body.getArgument(position), input == output ? value : load(input, row, offsets[column], accumulator));
          for (Operation &nested : body.without_terminator()) {
            if (auto constant = dyn_cast<arith::ConstantOp>(&nested); constant && width != 1) {
              Value splat = b.create<arith::ConstantOp>(loc, vectorType,
                  DenseElementsAttr::get(vectorType, ArrayRef<Attribute>{constant.getValue()}));
              mapping.map(constant.getResult(), splat);
            } else {
              Operation *cloned = b.clone(nested, mapping);
              if (width != 1) cloned->getResult(0).setType(vectorType);
            }
          }
          value = mapping.lookup(body.getTerminator()->getOperand(0));
          destination = epilogue->consumer.getOutputs()[0];
        }
        if (width == 1) b.create<memref::StoreOp>(loc, value, destination, ValueRange{row, offsets[column]});
        else b.create<vector::StoreOp>(loc, value, destination, ValueRange{row, offsets[column]});
      }
    operation.erase();
    if (epilogue) {
      epilogue->consumer.erase();
      epilogue->initialization.erase();
      epilogue->allocation.erase();
    }
  }
  return success();
}

}
