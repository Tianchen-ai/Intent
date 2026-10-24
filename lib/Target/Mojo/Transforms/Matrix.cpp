#include "Intent/Target/Mojo/Transforms/Matrix.h"
#include "Intent/Target/Mojo/Transforms/Passes.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/CPU/Transforms/Structure/LoopBuilders.h"
#include "mlir/Dialect/AMX/AMXDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;

namespace intent::mojo {
using namespace cpu;
namespace {

LogicalResult formMatrixTile(OpBuilder &b, linalg::GenericOp operation,
    const ContractionTile &tile, ConfigurationAttr, ImplementationAttr binding) {
  Location loc = operation.getLoc();
  b.getContext()->getOrLoadDialect<amx::AMXDialect>();
  int64_t microM = implementationParameter(binding, "micro_m");
  int64_t columnTiles = implementationParameter(binding, "micro_n");
  Type element = cast<MemRefType>(tile.lhs.getType()).getElementType();
  bool floating = element.isBF16();
  int64_t pack = floating ? 2 : 4, depth = floating ? 32 : 64;
  Type accumulator = floating ? Type(b.getF32Type()) : Type(b.getI32Type());
  const InputSupply *right = nullptr, *left = nullptr;
  for (const auto &supply : tile.inputs) {
    if (supply.operand == 1 && supply.panelAxis == 0 && supply.panelSize == pack)
      right = &supply;
    else if (supply.operand == 0 && supply.panelAxis == 1 && floating)
      left = &supply;
    else return operation.emitError("AMX contraction received an incompatible input supply");
  }
  if (!right)
    return operation.emitError("AMX contraction requires its declared four-byte B panels");
  Value zero = index(b, loc, 0), one = index(b, loc, 1);
  auto sub = [&](Value a, Value c) -> Value { return b.create<arith::SubIOp>(loc, a, c); };
  auto full = [&](Value size, int64_t quantum) -> Value {
    return sub(size, b.create<arith::RemSIOp>(loc, size, index(b, loc, quantum)));
  };
  // Existing input supply owns the packing and its lifetime. B's representation
  // [K/pack, N, pack] is the four-byte grouping consumed by the tile dot.
  Value packedB = b.create<memref::CollapseShapeOp>(loc, right->storage,
      ArrayRef<ReassociationIndices>{{0}, {1, 2}});
  Value bRow = b.create<arith::DivSIOp>(loc, sub(tile.kBegin, right->begins[0]), index(b, loc, pack));
  Value bColumn = multiply(b, loc, sub(tile.nBegin, right->begins[1]), index(b, loc, pack));
  Value mFull = full(tile.mCount, 16), nFull = full(tile.nCount, 16), kFull = full(tile.depth, depth);
  auto accumulatorType = amx::TileType::get({16, 16}, accumulator);
  auto inputType = amx::TileType::get({16, depth}, element);
  auto lhsMetadata = b.create<memref::ExtractStridedMetadataOp>(loc, tile.lhs);
  Value contiguous = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
      lhsMetadata.getStrides()[1], one);
  auto loadLeft = [&](Value row, Value column) -> Value {
    if (left) {
      Value packed = b.create<memref::CollapseShapeOp>(loc, left->storage,
          ArrayRef<ReassociationIndices>{{0, 1}, {2}});
      Value k = sub(column, left->begins[1]);
      Value width = index(b, loc, left->panelSize);
      Value panel = b.create<arith::DivSIOp>(loc, k, width);
      Value rowBase = multiply(b, loc, panel,
          b.create<memref::DimOp>(loc, left->storage, 1));
      return b.create<amx::TileLoadOp>(loc, inputType, packed,
          ValueRange{add(b, loc, rowBase, sub(row, left->begins[0])),
                     b.create<arith::RemSIOp>(loc, k, width)});
    }
    // AMX accepts the existing row pitch. Only noncontiguous K members require
    // a local tile copy; ordinary row-major inputs need no full-A preparation.
    auto choice = b.create<scf::IfOp>(loc, TypeRange{inputType}, contiguous, true);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(choice.thenBlock());
    auto type = MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, element,
        StridedLayoutAttr::get(b.getContext(), ShapedType::kDynamic, {ShapedType::kDynamic, 1}));
    Value view = b.create<memref::ReinterpretCastOp>(loc, type,
        lhsMetadata.getBaseBuffer(), OpFoldResult(lhsMetadata.getOffset()),
        ArrayRef<OpFoldResult>{lhsMetadata.getSizes()[0], lhsMetadata.getSizes()[1]},
        ArrayRef<OpFoldResult>{lhsMetadata.getStrides()[0], b.getIndexAttr(1)});
    Value direct = b.create<amx::TileLoadOp>(loc, inputType, view, ValueRange{row, column});
    b.create<scf::YieldOp>(loc, direct);
    b.setInsertionPointToStart(choice.elseBlock());
    auto local = b.create<memref::AllocaOp>(loc, MemRefType::get({16, depth}, element));
    local.setAlignment(64);
    loop(b, loc, zero, index(b, loc, 16), 1, [&](Value m) {
      loop(b, loc, zero, index(b, loc, depth), 1, [&](Value k) {
        Value value = b.create<memref::LoadOp>(loc, tile.lhs,
            ValueRange{add(b, loc, row, m), add(b, loc, column, k)});
        b.create<memref::StoreOp>(loc, value, local, ValueRange{m, k});
      });
    });
    Value copied = b.create<amx::TileLoadOp>(loc, inputType, local, ValueRange{zero, zero});
    b.create<scf::YieldOp>(loc, copied);
    return choice.getResult(0);
  };
  bool zeroInitial = tile.first && matchPattern(tile.initial, m_Zero());
  if (tile.first && !zeroInitial)
    b.create<linalg::FillOp>(loc, ValueRange{tile.initial}, ValueRange{tile.output});
  auto rectangle = [&](Value mBegin, Value mEnd, int64_t rowCount,
                       Value begin, Value end, int64_t count) {
    loop(b, loc, mBegin, mEnd, 16 * rowCount, [&](Value m) {
      loop(b, loc, begin, end, 16 * count, [&](Value n) {
        SmallVector<Value> initial, rows, coordinates;
        for (int64_t row = 0; row < rowCount; ++row)
          rows.push_back(add(b, loc, m, index(b, loc, 16 * row)));
        for (int64_t column = 0; column < count; ++column) {
          Value at = add(b, loc, n, index(b, loc, 16 * column));
          coordinates.push_back(at);
          for (Value row : rows)
            initial.push_back(zeroInitial ? Value(b.create<amx::TileZeroOp>(loc, accumulatorType))
                : Value(b.create<amx::TileLoadOp>(loc, accumulatorType, tile.output, ValueRange{row, at})));
        }
        auto reduction = b.create<scf::ForOp>(loc, zero, kFull, index(b, loc, depth), initial);
        {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(reduction.getBody());
          Value k = reduction.getInductionVar();
          SmallVector<Value> left;
          for (Value row : rows)
            left.push_back(loadLeft(add(b, loc, tile.mBegin, row), add(b, loc, tile.kBegin, k)));
          Value row = add(b, loc, bRow, b.create<arith::DivSIOp>(loc, k, index(b, loc, pack)));
          SmallVector<Value> next;
          for (auto [number, at] : llvm::enumerate(coordinates)) {
            Value column = add(b, loc, bColumn, multiply(b, loc, at, index(b, loc, pack)));
            Value rhs = b.create<amx::TileLoadOp>(loc, inputType, packedB, ValueRange{row, column});
            for (auto [row, a] : llvm::enumerate(left)) {
              Value carry = reduction.getRegionIterArgs()[number * rowCount + row];
              next.push_back(floating
                  ? Value(b.create<amx::TileMulFOp>(loc, accumulatorType, a, rhs, carry))
                  : Value(b.create<amx::TileMulIOp>(loc, accumulatorType, a, rhs, carry,
                                                  UnitAttr{}, UnitAttr{})));
            }
          }
          b.create<scf::YieldOp>(loc, next);
        }
        for (auto [number, at] : llvm::enumerate(coordinates)) {
          for (auto [row, position] : llvm::enumerate(rows))
            b.create<amx::TileStoreOp>(loc, tile.output, ValueRange{position, at},
                reduction.getResult(number * rowCount + row));
        }
      });
    });
  };
  Value groupedColumns = full(tile.nCount, 16 * columnTiles);
  auto rows = [&](Value begin, Value end, int64_t count) {
    rectangle(begin, end, count, zero, groupedColumns, columnTiles);
    if (columnTiles > 1) rectangle(begin, end, count, groupedColumns, nFull, 1);
  };
  Value groupedRows = full(tile.mCount, microM);
  rows(zero, groupedRows, microM / 16);
  if (microM > 16) rows(groupedRows, mFull, 1);
  // Boundary members use the original scalar arithmetic. No rounded-up source
  // reads or padded members are introduced, and every output has one owner.
  auto tail = [&](Value mBegin, Value mEnd, Value nBegin, Value nEnd, Value kBegin,
                  bool initialize) {
    loop(b, loc, mBegin, mEnd, 1, [&](Value m) {
      loop(b, loc, nBegin, nEnd, 1, [&](Value n) {
        Value initial = initialize && tile.first ? tile.initial
            : Value(b.create<memref::LoadOp>(loc, tile.output, ValueRange{m, n}));
        auto reduction = b.create<scf::ForOp>(loc, kBegin, tile.depth, one, ValueRange{initial});
        {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(reduction.getBody());
          Value k = add(b, loc, tile.kBegin, reduction.getInductionVar());
          Value a = b.create<memref::LoadOp>(loc, tile.lhs, ValueRange{add(b, loc, tile.mBegin, m), k});
          Value rhs = b.create<memref::LoadOp>(loc, tile.rhs, ValueRange{k, add(b, loc, tile.nBegin, n)});
          Value value;
          if (floating) {
            a = b.create<arith::ExtFOp>(loc, accumulator, a);
            rhs = b.create<arith::ExtFOp>(loc, accumulator, rhs);
            value = b.create<math::FmaOp>(loc, a, rhs, reduction.getRegionIterArgs()[0]);
          } else {
            a = b.create<arith::ExtSIOp>(loc, accumulator, a);
            rhs = b.create<arith::ExtSIOp>(loc, accumulator, rhs);
            value = b.create<arith::AddIOp>(loc, reduction.getRegionIterArgs()[0],
                b.create<arith::MulIOp>(loc, a, rhs));
          }
          b.create<scf::YieldOp>(loc, value);
        }
        b.create<memref::StoreOp>(loc, reduction.getResult(0), tile.output, ValueRange{m, n});
      });
    });
  };
  auto remainder = b.create<scf::IfOp>(loc,
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, kFull, tile.depth), false);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(remainder.thenBlock());
    tail(zero, mFull, zero, nFull, kFull, false);
  }
  tail(zero, mFull, nFull, tile.nCount, zero, true);
  tail(mFull, tile.mCount, zero, tile.nCount, zero, true);
  return success();
}

Value bfloatLattice(OpBuilder &b, Location loc, Value value) {
  Type type = value.getType();
  auto vector = dyn_cast<VectorType>(type);
  Type scalar = vector ? vector.getElementType() : type;
  bool bf16 = scalar.isBF16();
  auto bits = IntegerType::get(b.getContext(), bf16 ? 16 : 32);
  Type integers = vector ? Type(VectorType::get(vector.getShape(), bits)) : Type(bits);
  auto constant = [&](int64_t number) -> Value {
    Attribute literal = b.getIntegerAttr(bits, number);
    if (vector) literal = DenseElementsAttr::get(cast<VectorType>(integers), literal);
    return b.create<arith::ConstantOp>(loc, integers, cast<TypedAttr>(literal));
  };
  Value raw = b.create<arith::BitcastOp>(loc, integers, value);
  Value magnitude = b.create<arith::AndIOp>(loc, raw, constant(bf16 ? 0x7fff : 0x7fffffff));
  Value lower = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge,
      magnitude, constant(bf16 ? (77 << 7) : (36 << 23)));
  Value upper = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ule,
      magnitude, constant(bf16 ? (177 << 7) : (227LL << 23)));
  Value zero = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
      bf16 ? magnitude : raw, constant(0));
  return b.create<arith::OrIOp>(loc, zero, b.create<arith::AndIOp>(loc, lower, upper));
}

Value checkBfloatRectangle(OpBuilder &b, Location loc, Value memory,
    Value rowBegin, Value columnBegin, Value rows, Value columns, Value initial) {
  Value zero = index(b, loc, 0);
  Type element = cast<MemRefType>(memory.getType()).getElementType();
  int64_t width = element.isBF16() ? 32 : 16;
  auto predicates = VectorType::get({width}, b.getI1Type());
  Value lanes = b.create<arith::ConstantOp>(loc, predicates,
      DenseElementsAttr::get(predicates, b.getBoolAttr(true)));
  Value fullColumns = b.create<arith::SubIOp>(loc, columns,
      b.create<arith::RemSIOp>(loc, columns, index(b, loc, width)));
  auto rowLoop = b.create<scf::ForOp>(loc, zero, rows, index(b, loc, 1), ValueRange{lanes});
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(rowLoop.getBody());
  Value row = add(b, loc, rowBegin, rowLoop.getInductionVar());
  auto columnsLoop = b.create<scf::ForOp>(loc, zero, fullColumns, index(b, loc, width), rowLoop.getRegionIterArgs());
  b.setInsertionPointToStart(columnsLoop.getBody());
  Value values = b.create<vector::LoadOp>(loc, VectorType::get({width}, element), memory,
      ValueRange{row, add(b, loc, columnBegin, columnsLoop.getInductionVar())});
  Value valid = b.create<arith::AndIOp>(loc, columnsLoop.getRegionIterArgs()[0], bfloatLattice(b, loc, values));
  b.create<scf::YieldOp>(loc, valid);
  b.setInsertionPointAfter(columnsLoop);
  b.create<scf::YieldOp>(loc, columnsLoop.getResults());
  b.setInsertionPointAfter(rowLoop);
  Value complete = b.create<arith::AndIOp>(loc, initial,
      b.create<vector::ReductionOp>(loc, vector::CombiningKind::AND, rowLoop.getResult(0)));
  auto tails = b.create<scf::ForOp>(loc, zero, rows, index(b, loc, 1), ValueRange{complete});
  b.setInsertionPointToStart(tails.getBody());
  Value tailRow = add(b, loc, rowBegin, tails.getInductionVar());
  auto tail = b.create<scf::ForOp>(loc, fullColumns, columns, index(b, loc, 1), tails.getRegionIterArgs());
  b.setInsertionPointToStart(tail.getBody());
  Value value = b.create<memref::LoadOp>(loc, memory,
      ValueRange{tailRow, add(b, loc, columnBegin, tail.getInductionVar())});
  b.create<scf::YieldOp>(loc, b.create<arith::AndIOp>(loc,
      tail.getRegionIterArgs()[0], bfloatLattice(b, loc, value)).getResult());
  b.setInsertionPointAfter(tail);
  b.create<scf::YieldOp>(loc, tail.getResults());
  return tails.getResult(0);
}

FailureOr<Value> guardBfloatInputs(OpBuilder &b, linalg::GenericOp operation,
    ConfigurationAttr, ImplementationAttr) {
  Location loc = operation.getLoc();
  Value zero = index(b, loc, 0), safe = b.create<arith::ConstantIntOp>(loc, 1, 1);
  auto scalar = [&](Value memory, Value rows, Value columns) -> Value {
    auto outer = b.create<scf::ForOp>(loc, zero, rows, index(b, loc, 1), ValueRange{safe});
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(outer.getBody());
    auto inner = b.create<scf::ForOp>(loc, zero, columns, index(b, loc, 1), outer.getRegionIterArgs());
    b.setInsertionPointToStart(inner.getBody());
    Value value = b.create<memref::LoadOp>(loc, memory,
        ValueRange{outer.getInductionVar(), inner.getInductionVar()});
    b.create<scf::YieldOp>(loc, b.create<arith::AndIOp>(loc,
        inner.getRegionIterArgs()[0], bfloatLattice(b, loc, value)).getResult());
    b.setInsertionPointAfter(inner);
    b.create<scf::YieldOp>(loc, inner.getResults());
    return outer.getResult(0);
  };
  for (Value memory : operation.getInputs()) {
    auto type = cast<MemRefType>(memory.getType());
    auto metadata = b.create<memref::ExtractStridedMetadataOp>(loc, memory);
    auto contiguous = [&](unsigned axis) -> Value {
      return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
          metadata.getStrides()[axis], index(b, loc, 1));
    };
    auto check = [&](unsigned inner) -> Value {
      unsigned outer = 1 - inner;
      auto viewType = MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic},
          type.getElementType(), StridedLayoutAttr::get(b.getContext(),
              ShapedType::kDynamic, {ShapedType::kDynamic, 1}), type.getMemorySpace());
      Value view = b.create<memref::ReinterpretCastOp>(loc, viewType,
          metadata.getBaseBuffer(), OpFoldResult(metadata.getOffset()),
          ArrayRef<OpFoldResult>{metadata.getSizes()[outer], metadata.getSizes()[inner]},
          ArrayRef<OpFoldResult>{metadata.getStrides()[outer], b.getIndexAttr(1)});
      return checkBfloatRectangle(b, loc, view, zero, zero,
          metadata.getSizes()[outer], metadata.getSizes()[inner], safe);
    };
    auto choice = b.create<scf::IfOp>(loc, TypeRange{b.getI1Type()}, contiguous(1), true);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(choice.thenBlock());
      b.create<scf::YieldOp>(loc, check(1));
      b.setInsertionPointToStart(choice.elseBlock());
      auto transpose = b.create<scf::IfOp>(loc, TypeRange{b.getI1Type()}, contiguous(0), true);
      b.setInsertionPointToStart(transpose.thenBlock());
      b.create<scf::YieldOp>(loc, check(0));
      b.setInsertionPointToStart(transpose.elseBlock());
      b.create<scf::YieldOp>(loc, scalar(memory, metadata.getSizes()[0], metadata.getSizes()[1]));
      b.setInsertionPointAfter(transpose);
      b.create<scf::YieldOp>(loc, transpose.getResults());
    }
    safe = choice.getResult(0);
  }
  Value depth = b.create<memref::DimOp>(loc, operation.getInputs()[0], 1);
  return Value(b.create<arith::AndIOp>(loc, safe,
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ule,
          depth, index(b, loc, 1 << 20))));
}

LogicalResult formBfloatTile(OpBuilder &b, linalg::GenericOp operation,
    const ContractionTile &tile, ConfigurationAttr config, ImplementationAttr binding) {
  Location loc = operation.getLoc();
  if (!tile.inputGuard)
    return operation.emitError("BF16 tile requires its contraction input predicate");
  Value zero = index(b, loc, 0), safe = tile.inputGuard;
  // BF16 values in [2^-50,2^50] are multiples of 2^-57. Their products,
  // every rounded partial sum and a checked f32 accumulator are multiples of
  // 2^-114. A nonzero such value cannot be an f32 subnormal. This proves that
  // AMX's mandatory DAZ/FTZ has no effect; other inputs use the original FMA
  // implementation. NaN, infinity and negative-zero accumulators are excluded.
  // The initial accumulator's checked [2^-91,2^100] range has this lattice.
  // The guard bounds the WHOLE reduction by 2^20 members, so every partial
  // (including scalar tails) stays finite and on the lattice across K blocks.
  safe = b.create<arith::AndIOp>(loc, safe, bfloatLattice(b, loc, tile.initial));
  auto choice = b.create<scf::IfOp>(loc, safe, true);
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(choice.thenBlock());
  if (failed(formMatrixTile(b, operation, tile, config, binding))) return failure();
  b.setInsertionPointToStart(choice.elseBlock());
  NamedAttrList parameters;
  for (auto [name, value] : {std::pair<StringRef, int64_t>{"vector_width", implementationParameter(binding, "vector_width")},
       {"micro_m", 1}, {"micro_n", 1}, {"register_replicas", 1}, {"reduction_replicas", 1}})
    parameters.append(name, b.getI64IntegerAttr(value));
  auto scalarBinding = ImplementationAttr::get(b.getContext(), b.getStringAttr("mojo.register_float_direct"),
      parameters.getDictionary(b.getContext()));
  ContractionTile exact = tile;
  exact.lhs = b.create<memref::SubViewOp>(loc, tile.lhs,
      ArrayRef<OpFoldResult>{tile.mBegin, tile.kBegin},
      ArrayRef<OpFoldResult>{tile.mCount, tile.depth},
      ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
  int64_t fallbackColumns = implementationParameter(binding, "vector_width");
  auto exactRight = b.create<memref::AllocaOp>(loc,
      MemRefType::get({config.getTileK(), fallbackColumns}, b.getBF16Type()));
  exactRight.setAlignment(64);
  exact.rhs = exactRight;
  exact.mBegin = exact.nBegin = exact.kBegin = zero;
  exact.inputs = {};
  LogicalResult status = success();
  loop(b, loc, zero, tile.nCount, fallbackColumns, [&](Value column) {
    Value columns = b.create<arith::MinSIOp>(loc, index(b, loc, fallbackColumns),
        b.create<arith::SubIOp>(loc, tile.nCount, column));
    auto view = [&](Value memory, Value row, Value col, Value rows, Value columns) -> Value {
      return b.create<memref::SubViewOp>(loc, memory,
          ArrayRef<OpFoldResult>{row, col}, ArrayRef<OpFoldResult>{rows, columns},
          ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
    };
    Value source = view(tile.rhs, tile.kBegin, add(b, loc, tile.nBegin, column), tile.depth, columns);
    Value destination = view(exactRight, zero, zero, tile.depth, columns);
    auto identity = b.getMultiDimIdentityMap(2);
    b.create<linalg::GenericOp>(loc, ValueRange{source}, ValueRange{destination},
        ArrayRef<AffineMap>{identity, identity},
        SmallVector<utils::IteratorType>(2, utils::IteratorType::parallel),
        [](OpBuilder &nested, Location at, ValueRange arguments) {
          nested.create<linalg::YieldOp>(at, arguments[0]);
        });
    exact.output = view(tile.output, zero, column, tile.mCount, columns);
    exact.nCount = columns;
    if (failed(formRegisterTile(b, operation, exact, config, scalarBinding))) status = failure();
  });
  return status;
}

} // namespace

void registerMatrixImplementations(ImplementationRegistry &registry) {
  Implementation implementation;
  implementation.name = "mojo.amx_integer";
  implementation.applicable = [](Operation *op) {
    auto operation = cast<linalg::GenericOp>(op);
    return isMatrixContraction(operation) &&
        cast<MemRefType>(operation.getOutputs()[0].getType()).getElementType().isInteger(32) &&
        llvm::all_of(operation.getInputs(), [](Value input) {
          return cast<MemRefType>(input.getType()).getElementType().isInteger(8);
        });
  };
  implementation.requiresMatrixI8I32 = true;
  implementation.parameters = {
      ImplementationParameter::local("micro_m", {false, {}, 0, {16, 32}}),
      ImplementationParameter::local("micro_n", {false, {}, 0, {1, 2, 4}})};
  implementation.check = [](Operation *, CapabilitiesAttr capabilities, const Configuration &config)
      -> std::optional<std::string> {
    if (config.tileM % 16 || config.tileN % 16 || config.tileK % 64)
      return "AMX macro tiles require M/N multiples of 16 and K multiples of 64";
    int64_t rows = config.parameter("micro_m") / 16;
    int64_t columns = config.parameter("micro_n");
    if (rows * columns + rows + 2 > 8)
      return "AMX microtile exceeds eight tile registers including input and initialization state";
    if (capabilities.getPrivateBytes() <= rows * 1024 ||
        config.tileK > (capabilities.getPrivateBytes() - rows * 1024) / config.tileN)
      return "AMX packed B block and strided A tile exceed the private storage budget";
    return std::nullopt;
  };
  implementation.inputs = [](linalg::GenericOp operation, ConfigurationAttr, ImplementationAttr) {
    Type element = cast<MemRefType>(operation.getInputs()[0].getType()).getElementType();
    return SmallVector<InputRequirement>{
        {1, element, 0, 4, 64, InputReuse::Group, 4}};
  };
  implementation.formTile = formMatrixTile;
  registry.add<linalg::GenericOp>(std::move(implementation));

  Implementation bfloat;
  bfloat.name = "mojo.amx_bfloat";
  bfloat.applicable = [](Operation *op) {
    auto operation = cast<linalg::GenericOp>(op);
    auto order = operation->getAttrOfType<ReductionOrderAttr>("intent_cpu.reduction_order");
    return isMatrixContraction(operation) && order && order.getElementPermutation() &&
        cast<MemRefType>(operation.getOutputs()[0].getType()).getElementType().isF32() &&
        llvm::all_of(operation.getInputs(), [](Value input) {
          return cast<MemRefType>(input.getType()).getElementType().isBF16();
        });
  };
  bfloat.parameters = {
      ImplementationParameter::local("micro_m", {false, {}, 0, {16, 32}}),
      ImplementationParameter::local("micro_n", {false, {}, 0, {1, 2, 4}}),
      ImplementationParameter::local("vector_width", {true, {}, 32, {}})};
  bfloat.parameterRelations = [](DictionaryAttr values, CapabilitiesAttr capabilities) -> std::optional<std::string> {
    if (!capabilities.getMatrixBF16F32()) return "BF16 tile implementation requires matrix_bf16_f32";
    int64_t rows = cast<IntegerAttr>(values.get("micro_m")).getInt() / 16;
    int64_t columns = cast<IntegerAttr>(values.get("micro_n")).getInt();
    if (rows * columns + rows + 2 > 8) return "BF16 tile microprogram exceeds eight tile registers";
    return std::nullopt;
  };
  bfloat.check = [](Operation *, CapabilitiesAttr capabilities, const Configuration &config) -> std::optional<std::string> {
    if (config.tileM % 16 || config.tileN % 16 || config.tileK % 32)
      return "BF16 matrix tiles require M/N multiples of 16 and K multiples of 32";
    if (config.tileK > (1 << 20))
      return "BF16 guarded accumulation requires a bounded reduction tile";
    int64_t available = capabilities.getPrivateBytes();
    if (config.tileM > available / 4 / config.tileN)
      return "BF16 matrix accumulator exceeds the private storage budget";
    available -= config.tileM * config.tileN * 4;
    if (config.tileK > available / 2 / config.parameter("vector_width"))
      return "BF16 exact-path input block exceeds the private storage budget";
    return std::nullopt;
  };
  bfloat.inputs = [](linalg::GenericOp op, ConfigurationAttr config, ImplementationAttr) {
    Type element = cast<MemRefType>(op.getInputs()[0].getType()).getElementType();
    return SmallVector<InputRequirement>{{0, element, 1, config.getTileK(), 64, InputReuse::Consumers, config.getTileK()},
        {1, element, 0, 2, 64, InputReuse::Consumers, 2}};
  };
  bfloat.formTile = formBfloatTile;
  bfloat.guardContraction = guardBfloatInputs;
  registry.add<linalg::GenericOp>(std::move(bfloat));
}

} // namespace intent::mojo
