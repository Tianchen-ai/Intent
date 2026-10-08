#include "Intent/Dialect/CPU/Transforms/Contraction/Contraction.h"
#include "Intent/Dialect/CPU/Transforms/Storage/Storage.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "Intent/Dialect/CPU/IR/CPUAttrs.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "Intent/Dialect/CPU/Transforms/Implementation/Implementation.h"
#include "Intent/Dialect/CPU/Analysis/Contractions.h"
#include "Contractions.h"
#include "Intent/Dialect/CPU/Transforms/Structure/LoopBuilders.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

namespace intent::cpu {
namespace {

Value shapeValue(OpBuilder &builder, Location loc, Value source, unsigned axis) {
  return builder.createOrFold<memref::DimOp>(loc, source, axis);
}

SmallVector<OpFoldResult> shape(OpBuilder &builder, Location loc, Value source) {
  auto type = cast<MemRefType>(source.getType());
  SmallVector<OpFoldResult> result;
  for (int64_t axis = 0; axis < type.getRank(); ++axis)
    result.push_back(type.isDynamicDim(axis) ? OpFoldResult(shapeValue(builder, loc, source, axis))
                                           : OpFoldResult(builder.getIndexAttr(type.getDimSize(axis))));
  return result;
}

LogicalResult verifyStaticExtents(linalg::GenericOp operation, const ContractionAxes &axes) {
  auto verify = [&](Value value, ArrayRef<unsigned> group) -> LogicalResult {
    auto type = cast<MemRefType>(value.getType());
    if (llvm::any_of(group, [&](unsigned axis) { return type.getDimSize(axis) == 0; }) ||
        llvm::any_of(group, [&](unsigned axis) { return type.isDynamicDim(axis); })) return success();
    int64_t extent = 1;
    for (unsigned axis : group)
      if (llvm::MulOverflow(extent, type.getDimSize(axis), extent))
        return operation.emitError(
            "CPU contraction normalization does not support a flattened static extent exceeding signed 64-bit index");
    return success();
  };
  SmallVector<unsigned> lhsReduction, rhsReduction, outputM, outputN;
  for (auto pair : axes.reduction) {
    lhsReduction.push_back(pair.lhs);
    rhsReduction.push_back(pair.rhs);
  }
  for (unsigned axis : axes.lhsFree) outputM.push_back(*axes.lhsResultAxes[axis]);
  for (unsigned axis : axes.rhsFree) outputN.push_back(*axes.rhsResultAxes[axis]);
  return failure(failed(verify(operation.getInputs()[0], axes.lhsFree)) ||
      failed(verify(operation.getInputs()[0], lhsReduction)) ||
      failed(verify(operation.getInputs()[1], rhsReduction)) ||
      failed(verify(operation.getInputs()[1], axes.rhsFree)) ||
      failed(verify(operation.getOutputs()[0], outputM)) ||
      failed(verify(operation.getOutputs()[0], outputN)));
}

struct MatrixStorage {
  Value value;
  memref::AllocOp allocation;
  SmallVector<unsigned> first, second;
  Value source;
};

class Normalizer {
public:
  Normalizer(linalg::GenericOp operation, ContractionAxes axes)
      : operation(operation), axes(std::move(axes)), b(operation), loc(operation.getLoc()) {}

  LogicalResult run() {
    SmallVector<Value> sources(operation.getInputs());
    Value output = operation.getOutputs()[0];
    SmallVector<Value> batchExtents;
    for (auto pair : axes.batch) batchExtents.push_back(shapeValue(b, loc, sources[0], pair.lhs));
    if (batchExtents.empty()) return compute(sources[0], sources[1], output);
    Value zero = index(b, loc, 0), one = index(b, loc, 1);
    auto batches = b.create<scf::ParallelOp>(loc, SmallVector<Value>(batchExtents.size(), zero),
        batchExtents, SmallVector<Value>(batchExtents.size(), one));
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(batches.getBody());
    SmallVector<unsigned> leftBatch, rightBatch, outputBatch;
    for (auto pair : axes.batch) {
      leftBatch.push_back(pair.lhs);
      rightBatch.push_back(pair.rhs);
      outputBatch.push_back(*axes.lhsResultAxes[pair.lhs]);
    }
    return compute(slice(sources[0], leftBatch, batches.getInductionVars()),
                   slice(sources[1], rightBatch, batches.getInductionVars()),
                   slice(output, outputBatch, batches.getInductionVars()));
  }

private:
  Value slice(Value source, ArrayRef<unsigned> selected, ValueRange coordinates) {
    auto type = cast<MemRefType>(source.getType());
    SmallVector<OpFoldResult> offsets(type.getRank(), b.getIndexAttr(0)), sizes = shape(b, loc, source);
    SmallVector<int64_t> resultShape;
    for (auto [axis, coordinate] : llvm::zip(selected, coordinates)) {
      offsets[axis] = coordinate;
      sizes[axis] = b.getIndexAttr(1);
    }
    for (unsigned axis = 0; axis < static_cast<unsigned>(type.getRank()); ++axis)
      if (!llvm::is_contained(selected, axis)) resultShape.push_back(type.getDimSize(axis));
    auto sliced = cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
        resultShape, type, offsets, sizes, SmallVector<OpFoldResult>(type.getRank(), b.getIndexAttr(1))));
    return b.create<memref::SubViewOp>(loc, sliced, source, offsets, sizes,
        SmallVector<OpFoldResult>(type.getRank(), b.getIndexAttr(1)));
  }

  SmallVector<unsigned> remaining(ArrayRef<unsigned> original, bool lhs) {
    SmallVector<unsigned> removed;
    for (auto pair : axes.batch) removed.push_back(lhs ? pair.lhs : pair.rhs);
    SmallVector<unsigned> result;
    for (unsigned axis : original)
      result.push_back(axis - llvm::count_if(removed, [&](unsigned batch) { return batch < axis; }));
    return result;
  }

  bool contiguous(MemRefType type, ArrayRef<unsigned> group, ArrayRef<int64_t> strides) {
    SmallVector<unsigned> active;
    for (unsigned axis : group) if (type.getDimSize(axis) != 1) active.push_back(axis);
    if (active.size() < 2) return true;
    for (auto [outer, inner] : llvm::zip(ArrayRef(active).drop_back(), ArrayRef(active).drop_front())) {
      if (type.getLayout().isIdentity() && outer < inner) {
        bool units = true;
        for (unsigned axis = outer + 1; axis < inner; ++axis) units &= type.getDimSize(axis) == 1;
        if (units) continue;
      }
      int64_t expected;
      if (ShapedType::isDynamic(strides[outer]) || ShapedType::isDynamic(strides[inner]) ||
          type.isDynamicDim(inner) || llvm::MulOverflow(strides[inner], type.getDimSize(inner), expected) ||
          expected != strides[outer]) return false;
    }
    return true;
  }

  Value extent(Value source, ArrayRef<unsigned> group) {
    Value result = index(b, loc, 1);
    for (unsigned axis : group) result = b.createOrFold<arith::MulIOp>(loc, result, shapeValue(b, loc, source, axis));
    return result;
  }

  int64_t staticExtent(MemRefType type, ArrayRef<unsigned> group) {
    if (llvm::any_of(group, [&](unsigned axis) { return type.getDimSize(axis) == 0; })) return 0;
    if (llvm::any_of(group, [&](unsigned axis) { return type.isDynamicDim(axis); }))
      return ShapedType::kDynamic;
    // Every static group was checked before normalization changed the program.
    int64_t result = 1;
    for (unsigned axis : group) result *= type.getDimSize(axis);
    return result;
  }

  std::optional<Value> view(Value source, ArrayRef<unsigned> first, ArrayRef<unsigned> second) {
    auto type = cast<MemRefType>(source.getType());
    if (type.getRank() == 2 && first == ArrayRef<unsigned>{0} && second == ArrayRef<unsigned>{1})
      return source;
    SmallVector<int64_t> sourceStrides;
    int64_t offset;
    if (failed(type.getStridesAndOffset(sourceStrides, offset)) ||
        !contiguous(type, first, sourceStrides) || !contiguous(type, second, sourceStrides)) return std::nullopt;
    auto metadata = b.create<memref::ExtractStridedMetadataOp>(loc, source);
    SmallVector<int64_t> resultShape, staticStrides;
    SmallVector<OpFoldResult> sizes, strides;
    for (ArrayRef<unsigned> group : {first, second}) {
      int64_t size = staticExtent(type, group);
      resultShape.push_back(size);
      sizes.push_back(ShapedType::isDynamic(size) ? OpFoldResult(extent(source, group)) : OpFoldResult(b.getIndexAttr(size)));
      if (group.empty()) {
        staticStrides.push_back(0); strides.push_back(b.getIndexAttr(0)); continue;
      }
      auto active = llvm::find_if(llvm::reverse(group), [&](unsigned axis) { return type.getDimSize(axis) != 1; });
      unsigned axis = active == llvm::reverse(group).end() ? group.back() : *active;
      staticStrides.push_back(sourceStrides[axis]);
      strides.push_back(ShapedType::isDynamic(sourceStrides[axis])
          ? OpFoldResult(metadata.getStrides()[axis]) : OpFoldResult(b.getIndexAttr(sourceStrides[axis])));
    }
    auto result = MemRefType::get(resultShape, type.getElementType(),
        StridedLayoutAttr::get(b.getContext(), offset, staticStrides), type.getMemorySpace());
    OpFoldResult viewOffset = ShapedType::isDynamic(offset)
        ? OpFoldResult(metadata.getOffset()) : OpFoldResult(b.getIndexAttr(offset));
    return Value(b.create<memref::ReinterpretCastOp>(loc, result, metadata.getBaseBuffer(),
        viewOffset, sizes, strides));
  }

  void unpackCoordinate(Value linear, Value source, ArrayRef<unsigned> group, SmallVectorImpl<Value> &coordinates,
                        OpBuilder &builder) {
    for (unsigned i = group.size(); i > 0; --i) {
      unsigned axis = group[i - 1];
      if (i == 1) { coordinates[axis] = linear; break; }
      Value size = shapeValue(builder, loc, source, axis);
      coordinates[axis] = builder.create<arith::RemSIOp>(loc, linear, size);
      linear = builder.create<arith::DivSIOp>(loc, linear, size);
    }
  }

  MatrixStorage matrixStorage(Value source, ArrayRef<unsigned> first, ArrayRef<unsigned> second, bool input) {
    if (auto direct = view(source, first, second)) return {*direct, {}, {}, {}, {}};
    auto type = cast<MemRefType>(source.getType());
    SmallVector<int64_t> resultShape{staticExtent(type, first), staticExtent(type, second)};
    SmallVector<Value> dynamic;
    if (ShapedType::isDynamic(resultShape[0])) dynamic.push_back(extent(source, first));
    if (ShapedType::isDynamic(resultShape[1])) dynamic.push_back(extent(source, second));
    auto allocation = b.create<memref::AllocOp>(loc,
        MemRefType::get(resultShape, type.getElementType(), MemRefLayoutAttrInterface{},
                        type.getMemorySpace()), dynamic);
    MatrixStorage result{allocation, allocation, llvm::to_vector(first), llvm::to_vector(second), source};
    if (input) {
      b.create<linalg::GenericOp>(loc, ValueRange{}, ValueRange{allocation},
          SmallVector<AffineMap>{b.getMultiDimIdentityMap(2)},
          SmallVector<utils::IteratorType>(2, utils::IteratorType::parallel),
          [&](OpBuilder &builder, Location loc, ValueRange) {
            SmallVector<Value> coordinates(type.getRank());
            unpackCoordinate(builder.create<linalg::IndexOp>(loc, 0), source, first, coordinates, builder);
            unpackCoordinate(builder.create<linalg::IndexOp>(loc, 1), source, second, coordinates, builder);
            Value value = builder.create<memref::LoadOp>(loc, source, coordinates);
            builder.create<linalg::YieldOp>(loc, value);
          });
    }
    return result;
  }

  void copyBack(const MatrixStorage &storage) {
    if (!storage.allocation) return;
    auto type = cast<MemRefType>(storage.source.getType());
    b.create<linalg::GenericOp>(loc, ValueRange{}, ValueRange{storage.source},
        SmallVector<AffineMap>{b.getMultiDimIdentityMap(type.getRank())},
        SmallVector<utils::IteratorType>(type.getRank(), utils::IteratorType::parallel),
        [&](OpBuilder &builder, Location loc, ValueRange) {
          auto linear = [&](ArrayRef<unsigned> group) {
            Value coordinate = index(builder, loc, 0);
            for (unsigned axis : group)
              coordinate = builder.create<arith::AddIOp>(loc,
                  builder.create<arith::MulIOp>(loc, coordinate, shapeValue(builder, loc, storage.source, axis)),
                  builder.create<linalg::IndexOp>(loc, axis));
            return coordinate;
          };
          Value value = builder.create<memref::LoadOp>(loc, storage.value,
              ValueRange{linear(storage.first), linear(storage.second)});
          builder.create<linalg::YieldOp>(loc, value);
        });
  }

  Value dot(Value lhs, Value rhs, Value count, Type accumulator) {
    Value initial = b.create<arith::ConstantOp>(loc, b.getZeroAttr(accumulator));
    auto reduction = b.create<ReduceOp>(loc, accumulator, count, initial, ValueRange{lhs, rhs},
        b.getAffineMapArrayAttr(SmallVector<AffineMap>(2, b.getMultiDimIdentityMap(1))),
        operation->getAttrOfType<ReductionOrderAttr>("intent_cpu.reduction_order"));
    Type element = cast<MemRefType>(lhs.getType()).getElementType();
    Block &body = reduction.getCombine().emplaceBlock();
    body.addArguments(TypeRange{accumulator, element, element}, {loc, loc, loc});
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(&body);
      Value left = body.getArgument(1), right = body.getArgument(2);
      if (element != accumulator) {
        if (isa<FloatType>(element)) {
          left = b.create<arith::ExtFOp>(loc, accumulator, left);
          right = b.create<arith::ExtFOp>(loc, accumulator, right);
        } else {
          left = b.create<arith::ExtSIOp>(loc, accumulator, left);
          right = b.create<arith::ExtSIOp>(loc, accumulator, right);
        }
      }
      Value product = isa<FloatType>(accumulator) ? Value(b.create<arith::MulFOp>(
          loc, left, right, arith::FastMathFlags::contract))
                                                : Value(b.create<arith::MulIOp>(loc, left, right));
      Value sum = isa<FloatType>(accumulator) ? Value(b.create<arith::AddFOp>(
          loc, body.getArgument(0), product, arith::FastMathFlags::contract))
                                            : Value(b.create<arith::AddIOp>(loc, body.getArgument(0), product));
      b.create<YieldOp>(loc, sum);
    }
    return reduction.getResult();
  }

  Value vectorSlice(Value source, Value row, bool lhs) {
    auto type = cast<MemRefType>(source.getType());
    SmallVector<OpFoldResult> offsets{lhs ? OpFoldResult(row) : OpFoldResult(b.getIndexAttr(0)), b.getIndexAttr(0)};
    auto sizes = shape(b, loc, source);
    sizes[lhs ? 0 : 1] = b.getIndexAttr(1);
    auto result = cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
        ArrayRef<int64_t>{type.getDimSize(lhs ? 1 : 0)}, type, offsets, sizes,
        SmallVector<OpFoldResult>(2, b.getIndexAttr(1))));
    return b.create<memref::SubViewOp>(loc, result, source, offsets, sizes,
        SmallVector<OpFoldResult>(2, b.getIndexAttr(1)));
  }

  LogicalResult compute(Value lhs, Value rhs, Value output) {
    SmallVector<unsigned> leftReduction, rightReduction;
    for (auto pair : axes.reduction) { leftReduction.push_back(pair.lhs); rightReduction.push_back(pair.rhs); }
    auto leftFree = remaining(axes.lhsFree, true), rightFree = remaining(axes.rhsFree, false);
    leftReduction = remaining(leftReduction, true); rightReduction = remaining(rightReduction, false);
    auto left = matrixStorage(lhs, leftFree, leftReduction, true);
    auto right = matrixStorage(rhs, rightReduction, rightFree, true);
    Type accumulator = cast<MemRefType>(output.getType()).getElementType();
    bool scalar = leftFree.empty() && rightFree.empty();
    if (scalar) {
      Value zero = index(b, loc, 0);
      Value result = dot(vectorSlice(left.value, zero, true), vectorSlice(right.value, zero, false),
                         shapeValue(b, loc, left.value, 1), accumulator);
      b.create<memref::StoreOp>(loc, result, output, ValueRange{});
    } else {
      SmallVector<unsigned> outputM, outputN;
      for (unsigned axis = 0; axis < leftFree.size(); ++axis) outputM.push_back(axis);
      for (unsigned axis = 0; axis < rightFree.size(); ++axis) outputN.push_back(leftFree.size() + axis);
      auto destination = matrixStorage(output, outputM, outputN, false);
      if (isa<FloatType>(accumulator) && cast<MemRefType>(right.value.getType()).getDimSize(1) == 1) {
        Value zero = index(b, loc, 0), one = index(b, loc, 1);
        Value rows = shapeValue(b, loc, left.value, 0), depth = shapeValue(b, loc, left.value, 1);
        Value vector = vectorSlice(right.value, zero, false);
        auto parallel = b.create<scf::ParallelOp>(loc, ValueRange{zero}, ValueRange{rows}, ValueRange{one});
        {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(parallel.getBody());
          Value row = parallel.getInductionVars()[0];
          Value result = dot(vectorSlice(left.value, row, true), vector, depth, accumulator);
          b.create<memref::StoreOp>(loc, result, destination.value, ValueRange{row, zero});
        }
      } else {
        Value zero = b.create<arith::ConstantOp>(loc, b.getZeroAttr(accumulator));
        b.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{destination.value});
        AffineExpr m, n, k;
        bindDims(b.getContext(), m, n, k);
        auto contraction = b.create<linalg::GenericOp>(loc, ValueRange{left.value, right.value}, ValueRange{destination.value},
            ArrayRef<AffineMap>{AffineMap::get(3, 0, {m, k}, b.getContext()),
                AffineMap::get(3, 0, {k, n}, b.getContext()), AffineMap::get(3, 0, {m, n}, b.getContext())},
            SmallVector<utils::IteratorType>{utils::IteratorType::parallel, utils::IteratorType::parallel,
                                            utils::IteratorType::reduction},
            [&](OpBuilder &builder, Location loc, ValueRange arguments) {
              IRMapping mapping;
              Block &body = operation.getRegion().front();
              mapping.map(body.getArguments(), arguments);
              for (Operation &nested : body.without_terminator()) builder.clone(nested, mapping);
              builder.create<linalg::YieldOp>(loc, mapping.lookup(body.getTerminator()->getOperand(0)));
            });
        contraction->setAttr("intent_cpu.reduction_order", operation->getAttr("intent_cpu.reduction_order"));
      }
      copyBack(destination);
      if (destination.allocation) b.create<memref::DeallocOp>(loc, destination.allocation);
    }
    if (right.allocation) b.create<memref::DeallocOp>(loc, right.allocation);
    if (left.allocation) b.create<memref::DeallocOp>(loc, left.allocation);
    return success();
  }

  linalg::GenericOp operation;
  ContractionAxes axes;
  OpBuilder b;
  Location loc;
};

} // namespace

LogicalResult foldContractionInputs(func::FuncOp function,
                                    const ImplementationRegistry &implementations) {
  SmallVector<linalg::GenericOp> operations;
  function.walk([&](linalg::GenericOp operation) {
    if (operation->hasAttr("intent_cpu.implementation") && isMatrixContraction(operation))
      operations.push_back(operation);
  });
  for (auto operation : operations) {
    auto implementation = implementations.lookup(operation);
    if (failed(implementation)) return failure();
    const auto &requirements = (*implementation)->contraction;
    for (unsigned operand = 0; operand != 2; ++operand)
      operation.getDpsInputOperand(operand)->set(
          foldContractionInput(operation.getInputs()[operand], operation, &requirements, operand));
  }
  eraseDeadPrivateBuffers(function);
  return success();
}

LogicalResult normalizeContractions(func::FuncOp function) {
  SmallVector<linalg::GenericOp> operations;
  function.walk([&](linalg::GenericOp operation) {
    if (!operation->hasAttr("intent_cpu.implementation") && queryContractionAxes(operation))
      operations.push_back(operation);
  });
  for (auto operation : operations) {
    if (!operation->getAttrOfType<ReductionOrderAttr>("intent_cpu.reduction_order"))
      return operation.emitError("CPU contraction normalization requires explicit numerical ordering permissions");
    if (failed(verifyStaticExtents(operation, *queryContractionAxes(operation)))) return failure();
  }
  for (auto operation : operations) {
    auto axes = queryContractionAxes(operation);
    for (unsigned operand = 0; operand != 2; ++operand)
      operation.getDpsInputOperand(operand)->set(foldContractionInput(operation.getInputs()[operand], operation));
    bool column = isa<FloatType>(cast<MemRefType>(operation.getOutputs()[0].getType()).getElementType()) &&
        axes->rhsFree.size() == 1 &&
        cast<MemRefType>(operation.getInputs()[1].getType()).getDimSize(axes->rhsFree.front()) == 1;
    if (isMatrixContraction(operation) && !column) continue;
    auto fill = findContractionInitialization(operation);
    if (!fill) return operation.emitError("CPU contraction normalization requires a closed zero initialization");
    if (failed(Normalizer(operation, *axes).run())) return failure();
    operation.erase();
    if (fill->erasable) fill->operation->erase();
  }
  eraseDeadPrivateBuffers(function);
  return success();
}

} // namespace intent::cpu
