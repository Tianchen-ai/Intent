#include "Intent/Target/Weft/Transforms/Passes.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Quantization.h"
#include "Intent/Dialect/CPU/Transforms/Structure/LoopBuilders.h"
#include "Intent/Dialect/CPU/Transforms/Implementation/ImplementationInputs.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include <optional>
#include <string>

using namespace mlir;
namespace intent::weft_provider {
using namespace intent::cpu;
namespace {

constexpr int64_t directRhsSupply = 1;
constexpr int64_t invocationRhsSupply = 2;
constexpr int64_t directLhsSupply = 1;
constexpr int64_t retainedLhsSupply = 2;

bool hasReductionContiguousRhs(linalg::GenericOp operation) {
  auto type = cast<MemRefType>(operation.getInputs()[1].getType());
  SmallVector<int64_t> strides;
  int64_t offset;
  return succeeded(type.getStridesAndOffset(strides, offset)) && strides[0] == 1;
}

LogicalResult formIntegerTile(OpBuilder &b, linalg::GenericOp operation,
    const ContractionTile &tile, ConfigurationAttr, ImplementationAttr binding) {
  Location loc = operation.getLoc();
  auto rows = getConstantIntValue(tile.mCount), columns = getConstantIntValue(tile.nCount);
  auto depth = getConstantIntValue(tile.depth);
  if (!rows || !columns || !depth)
    return operation.emitError("matrix implementation requires bounded M/N/K tiles");
  auto view = [&](Value source, Value row, Value column, int64_t m, int64_t n) -> Value {
    return b.create<memref::SubViewOp>(loc, source,
        ArrayRef<OpFoldResult>{row, column}, ArrayRef<OpFoldResult>{b.getIndexAttr(m), b.getIndexAttr(n)},
        ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
  };
  const InputSupply *rightSupply = nullptr;
  for (const InputSupply &input : tile.inputs)
    if (input.operand == 1) rightSupply = &input;
  if (rightSupply && (rightSupply->panelAxis != 1 || rightSupply->panelSize != 1 ||
                      rightSupply->begins.size() != 2))
    return operation.emitError("integer contraction requires its declared transposed RHS supply");
  auto right = [&](Value k, Value n, int64_t depth, int64_t columns) -> Value {
    if (!rightSupply) return view(tile.rhs, k, n, depth, columns);
    Value row = b.create<arith::SubIOp>(loc, n, rightSupply->begins[1]);
    Value column = b.create<arith::SubIOp>(loc, k, rightSupply->begins[0]);
    SmallVector<OpFoldResult> offsets{row, column, b.getIndexAttr(0)};
    SmallVector<OpFoldResult> sizes{b.getIndexAttr(columns), b.getIndexAttr(depth), b.getIndexAttr(1)};
    SmallVector<OpFoldResult> strides(3, b.getIndexAttr(1));
    auto type = cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
        {columns, depth}, cast<MemRefType>(rightSupply->storage.getType()), offsets, sizes, strides));
    Value selected = b.create<memref::SubViewOp>(loc, type, rightSupply->storage, offsets, sizes, strides);
    return b.create<memref::TransposeOp>(loc, selected,
        AffineMapAttr::get(AffineMap::getPermutationMap(
            ArrayRef<unsigned>{1, 0}, b.getContext())));
  };
  auto pointwise = [&](Value lhs, Value rhs, Value output) {
    SmallVector<Value> inputs{lhs};
    if (rhs) inputs.push_back(rhs);
    b.create<linalg::GenericOp>(loc, inputs, ValueRange{output},
        SmallVector<AffineMap>(inputs.size() + 1, b.getMultiDimIdentityMap(2)),
        SmallVector<utils::IteratorType>(2, utils::IteratorType::parallel),
        [&](OpBuilder &nested, Location loc, ValueRange args) {
          Value result = rhs ? Value(nested.create<arith::AddIOp>(loc, args[0], args[1])) : args[0];
          nested.create<linalg::YieldOp>(loc, result);
        });
  };
  auto micro = [&](Value m, Value n, int64_t rows, int64_t columns) {
    auto type = MemRefType::get({rows, columns}, b.getI32Type());
    Value accumulator = b.create<memref::AllocaOp>(loc, type);
    b.create<linalg::FillOp>(loc, ValueRange{tile.initial}, ValueRange{accumulator});
    auto issue = [&](Value k, int64_t count) {
      Value lhs = view(tile.lhs, m, k, rows, count);
      Value rhs = right(k, n, count, columns);
      if (count == 1) {
        Value partial = b.create<memref::AllocaOp>(loc, type);
        AffineExpr row, column;
        bindDims(b.getContext(), row, column);
        b.create<linalg::GenericOp>(loc, ValueRange{lhs, rhs}, ValueRange{partial},
            ArrayRef<AffineMap>{AffineMap::get(2, 0, {row, b.getAffineConstantExpr(0)}, b.getContext()),
                AffineMap::get(2, 0, {b.getAffineConstantExpr(0), column}, b.getContext()),
                b.getMultiDimIdentityMap(2)},
            SmallVector<utils::IteratorType>(2, utils::IteratorType::parallel),
            [&](OpBuilder &nested, Location loc, ValueRange args) {
              Value left = nested.create<arith::ExtSIOp>(loc, b.getI32Type(), args[0]);
              Value right = nested.create<arith::ExtSIOp>(loc, b.getI32Type(), args[1]);
              nested.create<linalg::YieldOp>(loc, nested.create<arith::MulIOp>(loc, left, right).getResult());
            });
        pointwise(accumulator, partial, accumulator);
        return;
      }
      auto term = b.create<linalg::GenericOp>(loc, ValueRange{lhs, rhs}, ValueRange{accumulator},
          operation.getIndexingMapsArray(), operation.getIteratorTypesArray(),
          [&](OpBuilder &nested, Location loc, ValueRange args) {
            IRMapping mapping;
            Block &body = operation.getRegion().front();
            mapping.map(body.getArguments(), args);
            for (Operation &value : body.without_terminator()) nested.clone(value, mapping);
            nested.create<linalg::YieldOp>(loc, mapping.lookup(body.getTerminator()->getOperand(0)));
          });
      term->setAttr("intent_cpu.implementation", binding);
    };
    int64_t step = implementationParameter(binding, "micro_k");
    int64_t full = *depth / step * step;
    if (full)
      loop(b, loc, index(b, loc, 0), index(b, loc, full), step,
          [&](Value k) { issue(add(b, loc, tile.kBegin, k), step); });
    for (int64_t k = full; k < *depth; ++k)
      issue(add(b, loc, tile.kBegin, index(b, loc, k)), 1);
    Value out = view(tile.output, b.create<arith::SubIOp>(loc, m, tile.mBegin),
        b.create<arith::SubIOp>(loc, n, tile.nBegin), rows, columns);
    pointwise(accumulator, tile.first ? Value() : out, out);
  };
  auto panels = [&](Value begin, int64_t extent, int64_t panel,
                    const std::function<void(Value, int64_t)> &body) {
    int64_t full = extent / panel * panel;
    if (full)
      loop(b, loc, begin, add(b, loc, begin, index(b, loc, full)), panel,
          [&](Value offset) { body(offset, panel); });
    for (int64_t offset = full; offset < extent; ++offset)
      body(add(b, loc, begin, index(b, loc, offset)), 1);
  };
  int64_t microM = *depth == 1 ? 1 : implementationParameter(binding, "micro_m");
  int64_t microN = *depth == 1 ? 1 : implementationParameter(binding, "micro_n");
  panels(tile.mBegin, *rows, microM, [&](Value m, int64_t rows) {
    panels(tile.nBegin, *columns, microN,
        [&](Value n, int64_t columns) { micro(m, n, rows, columns); });
  });
  return success();
}

LogicalResult formTile(OpBuilder &b, linalg::GenericOp operation,
    const ContractionTile &tile, ConfigurationAttr, ImplementationAttr binding) {
  Location loc = operation.getLoc();
  auto rows = getConstantIntValue(tile.mCount), columns = getConstantIntValue(tile.nCount);
  if (!rows || !columns)
    return operation.emitError("selected Weft contraction requires statically bounded parallel tile extents");
  auto view = [&](Value source, Value m, Value n, OpFoldResult rows, OpFoldResult columns) -> Value {
    return b.create<memref::SubViewOp>(loc, source,
        ArrayRef<OpFoldResult>{m, n}, ArrayRef<OpFoldResult>{rows, columns},
        ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
  };
  auto form = [&](Value m, Value n, int64_t rows, int64_t columns,
                  Value suppliedLhs = {}, Value kBegin = {}, Value depth = {},
                  std::optional<bool> first = std::nullopt) {
    if (!kBegin) kBegin = tile.kBegin;
    if (!depth) depth = tile.depth;
    Value lhs = suppliedLhs ? suppliedLhs
        : view(tile.lhs, m, kBegin, b.getIndexAttr(rows), depth);
    Value rhs = view(tile.rhs, kBegin, n, depth, b.getIndexAttr(columns));
    Value out = view(tile.output, b.create<arith::SubIOp>(loc, m, tile.mBegin),
        b.create<arith::SubIOp>(loc, n, tile.nBegin),
        b.getIndexAttr(rows), b.getIndexAttr(columns));
    auto partial = b.create<memref::AllocaOp>(loc, MemRefType::get({rows, columns}, b.getF32Type()));
    b.create<linalg::FillOp>(loc, ValueRange{tile.initial}, ValueRange{partial});
    auto term = b.create<linalg::GenericOp>(loc, ValueRange{lhs, rhs}, ValueRange{partial},
        operation.getIndexingMapsArray(), operation.getIteratorTypesArray(),
        [&](OpBuilder &nested, Location loc, ValueRange args) {
          IRMapping mapping;
          Block &body = operation.getRegion().front();
          mapping.map(body.getArguments(), args);
          for (Operation &value : body.without_terminator())
            nested.clone(value, mapping);
          nested.create<linalg::YieldOp>(loc,
              mapping.lookup(body.getTerminator()->getOperand(0)));
        });
    term->setAttr("intent_cpu.implementation", binding);
    SmallVector<Value> inputs{partial};
    bool initialize = first.value_or(tile.first);
    if (!initialize) inputs.insert(inputs.begin(), out);
    b.create<linalg::GenericOp>(loc, inputs, ValueRange{out},
        SmallVector<AffineMap>(inputs.size() + 1, b.getMultiDimIdentityMap(2)),
        SmallVector<utils::IteratorType>(2, utils::IteratorType::parallel),
        [&](OpBuilder &nested, Location loc, ValueRange args) {
          Value result = args[0];
          if (!initialize) result = nested.create<arith::AddFOp>(loc, result, args[1]);
          nested.create<linalg::YieldOp>(loc, result);
        });
  };
  // The current Weft RVV stream form requires a non-singleton reduction
  // carrier. Keep the scalar reduction tail local; full K blocks remain 2D.
  if (getConstantIntValue(tile.depth) == 1) {
    loop(b, loc, tile.mBegin, add(b, loc, tile.mBegin, tile.mCount), 1, [&](Value m) {
      loop(b, loc, tile.nBegin, add(b, loc, tile.nBegin, tile.nCount), 1,
          [&](Value n) { form(m, n, 1, 1); });
    });
  } else {
    int64_t panel = implementationParameter(binding, "panel");
    auto panels = [&](Value begin, int64_t extent,
                      const std::function<void(Value, int64_t)> &body) {
      int64_t full = extent / panel * panel;
      if (full)
        loop(b, loc, begin, add(b, loc, begin, index(b, loc, full)), panel,
            [&](Value offset) { body(offset, panel); });
      if (extent != full) body(add(b, loc, begin, index(b, loc, full)), extent - full);
    };
    if (implementationParameter(binding, "lhs_supply") == retainedLhsSupply && *columns > panel) {
      auto depth = getConstantIntValue(tile.depth);
      StorageAnalysis storage(operation->getParentOfType<func::FuncOp>());
      if (!depth || *depth <= 0 || !storage.disjoint(tile.lhs, tile.output) ||
          !storage.disjoint(tile.rhs, tile.output))
        return operation.emitError("retained LHS supply requires a bounded reduction and disjoint output");
      // Keep the existing M panel so each RHS member still serves every row.
      // The independent K window bounds the snapshot retained over N.
      auto issue = [&](Value k, int64_t count, bool first) {
        Value width = index(b, loc, count);
        auto supply = [&](Value m, int64_t rows) {
          Value lhs = view(tile.lhs, m, k, b.getIndexAttr(rows), width);
          Value saved = b.create<memref::AllocaOp>(loc, MemRefType::get({rows, count},
              cast<MemRefType>(lhs.getType()).getElementType()));
          b.create<memref::CopyOp>(loc, lhs, saved);
          if (count == 1) {
            loop(b, loc, tile.nBegin, add(b, loc, tile.nBegin, tile.nCount), 1,
                [&](Value n) { form(m, n, 1, 1, saved, k, width, first); });
          } else {
            panels(tile.nBegin, *columns,
                [&](Value n, int64_t columns) { form(m, n, rows, columns, saved, k, width, first); });
          }
        };
        if (count == 1)
          loop(b, loc, tile.mBegin, add(b, loc, tile.mBegin, tile.mCount), 1,
              [&](Value m) { supply(m, 1); });
        else panels(tile.mBegin, *rows, supply);
      };
      int64_t window = implementationParameter(binding, "lhs_window");
      // This is a physical K supply window, independent of the outer K tile.
      // Each output keeps the same ascending member domain and first-write rule.
      issue(tile.kBegin, std::min(*depth, window), tile.first);
      int64_t full = *depth / window * window;
      if (full > window)
        loop(b, loc, index(b, loc, window), index(b, loc, full), window,
            [&](Value k) { issue(add(b, loc, tile.kBegin, k), window, false); });
      if (*depth > window && full != *depth)
        issue(add(b, loc, tile.kBegin, index(b, loc, full)), *depth - full, false);
    } else {
      panels(tile.mBegin, *rows, [&](Value m, int64_t rows) {
        panels(tile.nBegin, *columns, [&](Value n, int64_t columns) { form(m, n, rows, columns); });
      });
    }
  }
  return success();
}

}

cpu::ImplementationRegistry implementations() {
  ImplementationRegistry result;
  result.addProfile("weft.contract_i8_i32", {"micro_m", "micro_n", "micro_k", "rhs_supply", "panel"});
  result.addProfile("weft.q8_k", {"chunk", "panel"});
  for (StringRef family : {"weft.region_contract_f32", "weft.contract_f32"})
    result.addProfile(family, {"lhs_supply", "lhs_window", "panel"});
  for (StringRef family : {"weft.region_structured", "weft.structured"})
    result.addProfile(family, {"panel"});
  result.profile = [](func::FuncOp function) -> StringRef {
    bool quantize = false, contraction = false, integer = false, region = false;
    function.walk([&](cpu::QuantizeOp) { quantize = true; });
    function.walk([&](linalg::GenericOp op) {
      if (!isMatrixContraction(op)) return;
      contraction = true;
      integer |= cast<MemRefType>(op.getInputs()[0].getType()).getElementType().isSignlessInteger(8);
    });
    function.walk([&](Operation *op) { region |= isa<cpu::RegionFoldOp, cpu::RegionScanOp>(op); });
    if (integer) return "weft.contract_i8_i32";
    if (region) return contraction ? "weft.region_contract_f32" : "weft.region_structured";
    return quantize ? "weft.q8_k" : contraction ? "weft.contract_f32" : "weft.structured";
  };
  auto check = [](Operation *, CapabilitiesAttr, const Configuration &) -> std::optional<std::string> {
    return std::nullopt;
  };
  result.add<func::FuncOp>({"weft.scalar_program", {}, check, {}, {}, {}});
  result.add<cpu::QuantizeOp>({"weft.q8_k", {},
      check, {ImplementationParameter::local("chunk", {false, {}, 0, {32, 64}})},
      {}, [](Operation *operation, ImplementationExpansion &expansion) {
        return expandQuantize(cast<cpu::QuantizeOp>(operation), expansion);
      }});
  auto quantizedDot = [](Operation *operation,
                         ImplementationExpansion &expansion) {
    return expandQuantizedDot(cast<cpu::QuantizedDotOp>(operation), expansion);
  };
  auto checkMatrix = [](DictionaryAttr, CapabilitiesAttr capabilities) -> std::optional<std::string> {
    if (!capabilities.getMatrixI8I32()) return "target does not provide signed i8-to-i32 matrix operations";
    if (capabilities.getVectorBits() != 256)
      return "matrix implementation requires vector_bits=256, got " + std::to_string(capabilities.getVectorBits());
    return std::nullopt;
  };
  Implementation matrixDot{"weft.q4_k_q8_k_matrix", {},
      check, {ImplementationParameter::constant("columns", 4)},
      {}, quantizedDot, {}, [](ImplementationAttr binding) {
        return implementationParameter(binding, "columns");
      }, true};
  matrixDot.parameterRelations = checkMatrix;
  result.add<cpu::QuantizedDotOp>(std::move(matrixDot));
  result.add<cpu::QuantizedDotOp>({"weft.q4_k_q8_k", {},
      check, {}, {}, quantizedDot});
  Implementation integer{"weft.matrix_i8_i32", [](Operation *op) {
      auto generic = dyn_cast<linalg::GenericOp>(op);
      return generic && isMatrixContraction(generic) &&
          cast<MemRefType>(generic.getInputs()[0].getType()).getElementType().isSignlessInteger(8);
    }, [](Operation *operation, CapabilitiesAttr, const Configuration &config) -> std::optional<std::string> {
      int64_t m = config.parameter("micro_m"), n = config.parameter("micro_n");
      if (config.tileM % m != 0)
        return "tile_m " + std::to_string(config.tileM) + " must be divisible by micro_m " + std::to_string(m);
      if (config.tileN % n != 0)
        return "tile_n " + std::to_string(config.tileN) + " must be divisible by micro_n " + std::to_string(n);
      if (config.tileK % 8 != 0)
        return "tile_k " + std::to_string(config.tileK) + " must be divisible by micro_k 8";
      if (config.parameter("rhs_supply") == invocationRhsSupply &&
          !hasReductionContiguousRhs(cast<linalg::GenericOp>(operation)) &&
          !hasInvocationInputScope(operation))
        return "invocation RHS preparation must precede concurrent work items";
      return std::nullopt;
    }, {ImplementationParameter::local("micro_m", {false, {}, 0, {1, 4}}),
        ImplementationParameter::local("micro_n", {false, {}, 0, {4, 16}}),
        ImplementationParameter::local("micro_k", {false, {}, 0, {8}}),
        ImplementationParameter::local("rhs_supply", {false, {}, 0,
            {directRhsSupply, invocationRhsSupply}})},
    formIntegerTile, {}, {true, true, true}, {}, true};
  integer.parameterRelations = checkMatrix;
  integer.inputs = [](linalg::GenericOp operation, ConfigurationAttr,
                      ImplementationAttr binding) -> SmallVector<InputRequirement> {
    if (implementationParameter(binding, "rhs_supply") == directRhsSupply ||
        hasReductionContiguousRhs(operation))
      return {};
    auto type = cast<MemRefType>(operation.getInputs()[1].getType());
    return {{1, type.getElementType(), 1, 1, 64, InputReuse::Consumers, 1,
             InputStorageScope::Invocation}};
  };
  Implementation vectorInteger = integer;
  vectorInteger.name = "weft.vector_i8_i32";
  vectorInteger.requiresMatrixI8I32 = false;
  vectorInteger.parameterRelations = {};
  result.add<linalg::GenericOp>(std::move(integer));
  result.add<linalg::GenericOp>(std::move(vectorInteger));
  result.add<linalg::GenericOp>({"weft.contract_f32", [](Operation *op) {
      auto generic = dyn_cast<linalg::GenericOp>(op);
      auto input = [](Type element) {
        return element.isF16() || element.isBF16() || element.isF32();
      };
      return generic && isMatrixContraction(generic) &&
          input(cast<MemRefType>(generic.getInputs()[0].getType()).getElementType()) &&
          input(cast<MemRefType>(generic.getInputs()[1].getType()).getElementType()) &&
          cast<MemRefType>(generic.getOutputs()[0].getType()).getElementType().isF32();
    }, [](Operation *operation, CapabilitiesAttr, const Configuration &config) -> std::optional<std::string> {
      if (config.parameter("lhs_supply") != retainedLhsSupply) return std::nullopt;
      auto generic = cast<linalg::GenericOp>(operation);
      StorageAnalysis storage(operation->getParentOfType<func::FuncOp>());
      if (!storage.disjoint(generic.getInputs()[0], generic.getOutputs()[0]) ||
          !storage.disjoint(generic.getInputs()[1], generic.getOutputs()[0]))
        return "retained LHS supply requires disjoint input and output storage";
      return std::nullopt;
    }, {ImplementationParameter::minimum("panel", 4,
        {ImplementationParameter::Axis::TileM, ImplementationParameter::Axis::TileN}),
        ImplementationParameter::local("lhs_supply", {false, {}, 0,
            {directLhsSupply, retainedLhsSupply}}),
        ImplementationParameter::local("lhs_window", {false, {}, 0, {16, 32, 64}})},
    formTile, {}, {true, true, true}});
  Implementation structured{"weft.structured", [](Operation *op) {
      if (auto generic = dyn_cast<linalg::GenericOp>(op)) return !isMatrixContraction(generic);
      return true;
    }, check, {ImplementationParameter::local("panel", {true, {}, 0, {}})},
    {}, {}, {}, [](ImplementationAttr binding) {
      return implementationParameter(binding, "panel");
    }};
  result.add<linalg::GenericOp, cpu::ReduceOp, cpu::ScanOp, cpu::HistogramOp>(
      std::move(structured));
  return result;
}

}
