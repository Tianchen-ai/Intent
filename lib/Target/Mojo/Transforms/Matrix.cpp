#include "Intent/Target/Mojo/Transforms/Matrix.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/CPU/Transforms/Structure/LoopBuilders.h"
#include "mlir/Dialect/AMX/AMXDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;

namespace intent::mojo {
using namespace cpu;
namespace {

LogicalResult formIntegerTile(OpBuilder &b, linalg::GenericOp operation,
    const ContractionTile &tile, ConfigurationAttr, ImplementationAttr binding) {
  Location loc = operation.getLoc();
  b.getContext()->getOrLoadDialect<amx::AMXDialect>();
  int64_t microM = implementationParameter(binding, "micro_m");
  int64_t columnTiles = implementationParameter(binding, "micro_n");
  const InputSupply *right = nullptr;
  for (const auto &supply : tile.inputs) {
    if (supply.operand == 1 && supply.panelAxis == 0 && supply.panelSize == 4)
      right = &supply;
    else return operation.emitError("AMX integer contraction received an incompatible input supply");
  }
  if (!right)
    return operation.emitError("AMX integer contraction requires its declared four-byte B panels");
  Value zero = index(b, loc, 0), one = index(b, loc, 1);
  auto sub = [&](Value a, Value c) -> Value { return b.create<arith::SubIOp>(loc, a, c); };
  auto full = [&](Value size, int64_t quantum) -> Value {
    return sub(size, b.create<arith::RemSIOp>(loc, size, index(b, loc, quantum)));
  };
  // Existing input supply owns the packing and its lifetime. B's representation
  // [K/4, N, 4] is precisely the four-byte grouping consumed by TDPBSSD.
  Value packedB = b.create<memref::CollapseShapeOp>(loc, right->storage,
      ArrayRef<ReassociationIndices>{{0}, {1, 2}});
  Value bRow = b.create<arith::DivSIOp>(loc, sub(tile.kBegin, right->begins[0]), index(b, loc, 4));
  Value bColumn = multiply(b, loc, sub(tile.nBegin, right->begins[1]), index(b, loc, 4));
  Value mFull = full(tile.mCount, 16), nFull = full(tile.nCount, 16), kFull = full(tile.depth, 64);
  auto accumulatorType = amx::TileType::get({16, 16}, b.getI32Type());
  auto inputType = amx::TileType::get({16, 64}, b.getI8Type());
  auto lhsMetadata = b.create<memref::ExtractStridedMetadataOp>(loc, tile.lhs);
  Value contiguous = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
      lhsMetadata.getStrides()[1], one);
  auto loadLeft = [&](Value row, Value column) -> Value {
    // AMX accepts the existing row pitch. Only noncontiguous K members require
    // a local tile copy; ordinary row-major inputs need no full-A preparation.
    auto choice = b.create<scf::IfOp>(loc, TypeRange{inputType}, contiguous, true);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(choice.thenBlock());
    auto type = MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, b.getI8Type(),
        StridedLayoutAttr::get(b.getContext(), ShapedType::kDynamic, {ShapedType::kDynamic, 1}));
    Value view = b.create<memref::ReinterpretCastOp>(loc, type,
        lhsMetadata.getBaseBuffer(), OpFoldResult(lhsMetadata.getOffset()),
        ArrayRef<OpFoldResult>{lhsMetadata.getSizes()[0], lhsMetadata.getSizes()[1]},
        ArrayRef<OpFoldResult>{lhsMetadata.getStrides()[0], b.getIndexAttr(1)});
    Value direct = b.create<amx::TileLoadOp>(loc, inputType, view, ValueRange{row, column});
    b.create<scf::YieldOp>(loc, direct);
    b.setInsertionPointToStart(choice.elseBlock());
    auto local = b.create<memref::AllocaOp>(loc, MemRefType::get({16, 64}, b.getI8Type()));
    local.setAlignment(64);
    loop(b, loc, zero, index(b, loc, 16), 1, [&](Value m) {
      loop(b, loc, zero, index(b, loc, 64), 1, [&](Value k) {
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
        auto reduction = b.create<scf::ForOp>(loc, zero, kFull, index(b, loc, 64), initial);
        {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(reduction.getBody());
          Value k = reduction.getInductionVar();
          SmallVector<Value> left;
          for (Value row : rows)
            left.push_back(loadLeft(add(b, loc, tile.mBegin, row), add(b, loc, tile.kBegin, k)));
          Value row = add(b, loc, bRow, b.create<arith::DivSIOp>(loc, k, index(b, loc, 4)));
          SmallVector<Value> next;
          for (auto [number, at] : llvm::enumerate(coordinates)) {
            Value column = add(b, loc, bColumn, multiply(b, loc, at, index(b, loc, 4)));
            Value rhs = b.create<amx::TileLoadOp>(loc, inputType, packedB, ValueRange{row, column});
            for (auto [row, a] : llvm::enumerate(left))
              next.push_back(b.create<amx::TileMulIOp>(loc, accumulatorType, a, rhs,
                  reduction.getRegionIterArgs()[number * rowCount + row], UnitAttr{}, UnitAttr{}));
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
  // Boundary members use the same signed i32 arithmetic. No rounded-up source
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
          a = b.create<arith::ExtSIOp>(loc, b.getI32Type(), a);
          rhs = b.create<arith::ExtSIOp>(loc, b.getI32Type(), rhs);
          Value value = b.create<arith::AddIOp>(loc, reduction.getRegionIterArgs()[0],
              b.create<arith::MulIOp>(loc, a, rhs));
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
  implementation.formTile = formIntegerTile;
  registry.add<linalg::GenericOp>(std::move(implementation));
}

} // namespace intent::mojo
