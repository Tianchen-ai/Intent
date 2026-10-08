#include "Quantization.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "Intent/Dialect/CPU/Transforms/Structure/LoopBuilders.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/IRMapping.h"
#include <limits>

using namespace mlir;

namespace intent::mojo {
namespace {

// The closed record formats own these sizes. These are not program tiles or
// assumptions about an external view's alignment/strides.
constexpr int64_t recordElements = 256;

struct RecordBuilder {
  OpBuilder &b;
  Location loc;
  cpu::ImplementationAttr binding;

  Value index(int64_t value) { return cpu::index(b, loc, value); }
  Type lanes(Type element, int64_t width) {
    return width == 1 ? element : Type(VectorType::get({width}, element));
  }
  Value constant(Type type, Attribute scalar) {
    Attribute value = scalar;
    if (auto vector = dyn_cast<VectorType>(type))
      value = DenseElementsAttr::get(vector, cast<TypedAttr>(scalar));
    return b.create<arith::ConstantOp>(loc, type, cast<TypedAttr>(value));
  }
  Value fp(double value, int64_t width = 1) {
    return constant(lanes(b.getF32Type(), width), b.getF32FloatAttr(value));
  }
  Value integer(int64_t value, unsigned bits = 32, int64_t width = 1) {
    return constant(lanes(b.getIntegerType(bits), width), b.getIntegerAttr(b.getIntegerType(bits), value));
  }
  Value broadcast(Value value, int64_t width) {
    return width == 1 ? value : Value(b.create<vector::BroadcastOp>(loc,
        cast<VectorType>(lanes(value.getType(), width)), value));
  }
  Value addIndex(Value base, int64_t amount) {
    return amount ? Value(b.create<arith::AddIOp>(loc, base, index(amount))) : base;
  }
  scf::ForOp loop(Value lower, Value upper, int64_t step, ValueRange initial = {}) {
    auto loop = b.create<scf::ForOp>(loc, lower, upper, index(step), initial);
    loop->setAttr("intent_cpu.implementation", binding);
    return loop;
  }
  Value read(Value memory, ValueRange prefix, Value offset, int64_t width = 1) {
    SmallVector<Value> coordinates(prefix);
    coordinates.push_back(offset);
    if (width == 1) return b.create<memref::LoadOp>(loc, memory, coordinates);
    auto element = cast<MemRefType>(memory.getType()).getElementType();
    return b.create<vector::LoadOp>(loc, cast<VectorType>(lanes(element, width)), memory, coordinates);
  }
  void write(Value value, Value memory, ValueRange prefix, Value offset) {
    SmallVector<Value> coordinates(prefix);
    coordinates.push_back(offset);
    if (isa<VectorType>(value.getType())) b.create<vector::StoreOp>(loc, value, memory, coordinates);
    else b.create<memref::StoreOp>(loc, value, memory, coordinates);
  }
  Value littleEndian(Value memory, ValueRange prefix, int64_t offset, unsigned bytes) {
    Type type = b.getIntegerType(bytes * 8);
    Value result = b.create<arith::ExtUIOp>(loc, type, read(memory, prefix, index(offset)));
    for (unsigned byte = 1; byte < bytes; ++byte) {
      Value next = b.create<arith::ExtUIOp>(loc, type, read(memory, prefix, index(offset + byte)));
      next = b.create<arith::ShLIOp>(loc, next, integer(byte * 8, bytes * 8));
      result = b.create<arith::OrIOp>(loc, result, next);
    }
    return result;
  }
  void writeLittleEndian(Value value, Value memory, ValueRange prefix, Value offset, unsigned bytes) {
    for (unsigned byte = 0; byte < bytes; ++byte) {
      Value part = value;
      if (byte) part = b.create<arith::ShRUIOp>(loc, part,
          integer(byte * 8, cast<IntegerType>(value.getType()).getWidth()));
      write(b.create<arith::TruncIOp>(loc, b.getI8Type(), part), memory, prefix, addIndex(offset, byte));
    }
  }
  Value field6(Value memory, ValueRange prefix, int field, int member) {
    Value value = read(memory, prefix, index(4 + 4 * field + member % 4));
    value = b.create<arith::ExtUIOp>(loc, b.getI32Type(), value);
    if (member < 4) return b.create<arith::AndIOp>(loc, value, integer(63));
    Value high = b.create<arith::ShRUIOp>(loc, value, integer(6));
    high = b.create<arith::ShLIOp>(loc, high, integer(4));
    Value low = b.create<arith::ExtUIOp>(loc, b.getI32Type(),
        read(memory, prefix, index(12 + member - 4)));
    if (field) low = b.create<arith::ShRUIOp>(loc, low, integer(4));
    low = b.create<arith::AndIOp>(loc, low, integer(15));
    return b.create<arith::OrIOp>(loc, high, low);
  }
  Value reduce(Value value, vector::CombiningKind kind) {
    if (!isa<VectorType>(value.getType())) return value;
    return b.create<vector::ReductionOp>(loc, kind, value);
  }
};

// The descriptor's own stride decides whether contiguous SIMD memory operations
// apply. The other branch executes the same record arithmetic on actual scalar
// coordinates; neither branch changes the supplied storage or record layout.
Value contiguous(RecordBuilder &p, ValueRange memories) {
  Value condition;
  for (Value memory : memories) {
    auto type = cast<MemRefType>(memory.getType());
    SmallVector<int64_t> strides;
    int64_t offset;
    if (failed(type.getStridesAndOffset(strides, offset))) return p.integer(0, 1);
    if (strides.back() == 1) continue;
    Value test;
    if (ShapedType::isDynamic(strides.back())) {
      auto metadata = p.b.create<memref::ExtractStridedMetadataOp>(p.loc, memory);
      test = p.b.create<arith::CmpIOp>(p.loc, arith::CmpIPredicate::eq,
          metadata.getStrides().back(), p.index(1));
    } else test = p.integer(0, 1);
    condition = condition ? Value(p.b.create<arith::AndIOp>(p.loc, condition, test)) : test;
  }
  return condition;
}

Value contiguousView(RecordBuilder &p, Value memory) {
  auto type = cast<MemRefType>(memory.getType());
  SmallVector<int64_t> strides;
  int64_t offset;
  (void)type.getStridesAndOffset(strides, offset);
  if (strides.back() == 1) return memory;
  auto metadata = p.b.create<memref::ExtractStridedMetadataOp>(p.loc, memory);
  strides.back() = 1;
  auto layout = StridedLayoutAttr::get(p.b.getContext(), offset, strides);
  auto result = MemRefType::get(type.getShape(), type.getElementType(), layout, type.getMemorySpace());
  SmallVector<OpFoldResult> sizes, steps;
  for (int64_t axis = 0; axis < type.getRank(); ++axis) {
    sizes.push_back(type.isDynamicDim(axis) ? OpFoldResult(metadata.getSizes()[axis])
                                           : p.b.getIndexAttr(type.getDimSize(axis)));
    steps.push_back(ShapedType::isDynamic(strides[axis]) ? OpFoldResult(metadata.getStrides()[axis])
                                                       : p.b.getIndexAttr(strides[axis]));
  }
  OpFoldResult begin = ShapedType::isDynamic(offset) ? OpFoldResult(metadata.getOffset())
                                                   : p.b.getIndexAttr(offset);
  return p.b.create<memref::ReinterpretCastOp>(p.loc, result, metadata.getBaseBuffer(), begin, sizes, steps);
}

void quantize(RecordBuilder &p, cpu::QuantizeOp operation, int64_t width) {
  OpBuilder &b = p.b;
  Location loc = p.loc;
  Value groups = b.create<memref::DimOp>(loc, operation.getInput(), 0);
  auto records = p.loop(p.index(0), groups, 1);
  OpBuilder::InsertionGuard recordScope(b);
  b.setInsertionPointToStart(records.getBody());
  Value record = records.getInductionVar();
  Value high = p.fp(-std::numeric_limits<float>::infinity(), width);
  Value low = p.fp(std::numeric_limits<float>::infinity(), width);
  auto extrema = p.loop(p.index(0), p.index(recordElements), width, {high, low});
  {
    OpBuilder::InsertionGuard scope(b);
    b.setInsertionPointToStart(extrema.getBody());
    Value values = p.read(operation.getInput(), {record}, extrema.getInductionVar(), width);
    high = b.create<arith::MaximumFOp>(loc, extrema.getRegionIterArgs()[0], values);
    low = b.create<arith::MinimumFOp>(loc, extrema.getRegionIterArgs()[1], values);
    b.create<scf::YieldOp>(loc, ValueRange{high, low});
  }
  high = p.reduce(extrema.getResult(0), vector::CombiningKind::MAXIMUMF);
  low = p.reduce(extrema.getResult(1), vector::CombiningKind::MINIMUMF);
  Value greater = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT,
      b.create<math::AbsFOp>(loc, high), b.create<math::AbsFOp>(loc, low));
  Value extreme = b.create<arith::SelectOp>(loc, greater, high, low);
  Value nonzero = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ONE, extreme, p.fp(0));
  auto scales = b.create<scf::IfOp>(loc, TypeRange{b.getF32Type(), b.getF32Type()}, nonzero, true);
  {
    OpBuilder::InsertionGuard scope(b);
    b.setInsertionPointToStart(scales.thenBlock());
    Value inverse = b.create<arith::DivFOp>(loc, p.fp(-127), extreme);
    Value scale = b.create<arith::DivFOp>(loc, p.fp(1), inverse);
    b.create<scf::YieldOp>(loc, ValueRange{inverse, scale});
    b.setInsertionPointToStart(scales.elseBlock());
    Value zero = p.fp(0);
    b.create<scf::YieldOp>(loc, ValueRange{zero, zero});
  }
  Value inverse = p.broadcast(scales.getResult(0), width);
  auto sums = p.loop(p.index(0), p.index(recordElements), 16);
  {
    OpBuilder::InsertionGuard scope(b);
    b.setInsertionPointToStart(sums.getBody());
    Value sum = p.integer(0);
    for (int64_t lane = 0; lane < 16; lane += width) {
      Value offset = p.addIndex(sums.getInductionVar(), lane);
      Value values = p.read(operation.getInput(), {record}, offset, width);
      Value scaled = b.create<arith::MulFOp>(loc, values, inverse);
      Value rounded = b.create<math::RoundEvenOp>(loc, scaled);
      rounded = b.create<arith::MaximumFOp>(loc, rounded, p.fp(-128, width));
      rounded = b.create<arith::MinimumFOp>(loc, rounded, p.fp(127, width));
      Value integer = b.create<arith::FPToSIOp>(loc, p.lanes(b.getI32Type(), width), rounded);
      Value bytes = b.create<arith::TruncIOp>(loc, p.lanes(b.getI8Type(), width), integer);
      p.write(bytes, operation.getOutput(), {record}, p.addIndex(offset, 4));
      sum = b.create<arith::AddIOp>(loc, sum, p.reduce(integer, vector::CombiningKind::ADD));
    }
    Value offset = b.create<arith::DivSIOp>(loc, sums.getInductionVar(), p.index(8));
    Value narrow = b.create<arith::TruncIOp>(loc, b.getI16Type(), sum);
    p.writeLittleEndian(narrow, operation.getOutput(), {record}, p.addIndex(offset, 260), 2);
  }
  Value bits = b.create<arith::BitcastOp>(loc, b.getI32Type(), scales.getResult(1));
  p.writeLittleEndian(bits, operation.getOutput(), {record}, p.index(0), 4);
}

void dot(RecordBuilder &p, cpu::QuantizedDotOp operation, int64_t width) {
  OpBuilder &b = p.b;
  Location loc = p.loc;
  auto outType = cast<MemRefType>(operation.getOutput().getType());
  bool grouped = outType.getRank() == 1;
  int64_t columns = grouped ? outType.getDimSize(0) : 1;
  Value groups = b.create<memref::DimOp>(loc, operation.getRhs(), 0);
  SmallVector<Value> initial(columns, p.fp(0));
  auto records = p.loop(p.index(0), groups, 1, initial);
  {
    OpBuilder::InsertionGuard scope(b);
    b.setInsertionPointToStart(records.getBody());
    Value record = records.getInductionVar();
    auto leftPrefix = [&](int64_t column) {
      SmallVector<Value> prefix;
      if (grouped) prefix.push_back(p.index(column));
      prefix.push_back(record);
      return prefix;
    };
    SmallVector<Value> positive(columns, p.integer(0)), negative(columns, p.integer(0));
    for (int sub = 0; sub < 8; ++sub) {
      SmallVector<Value> partials(columns, p.integer(0, 32, width));
      auto members = p.loop(p.index(0), p.index(32), width, partials);
      {
        OpBuilder::InsertionGuard memberScope(b);
        b.setInsertionPointToStart(members.getBody());
        Value offset = members.getInductionVar();
        Value xq = p.read(operation.getRhs(), {record}, p.addIndex(offset, 4 + sub * 32), width);
        xq = b.create<arith::ExtSIOp>(loc, p.lanes(b.getI32Type(), width), xq);
        SmallVector<Value> next;
        for (int64_t column = 0; column < columns; ++column) {
          Value wq = p.read(operation.getLhs(), leftPrefix(column), p.addIndex(offset, 16 + (sub / 2) * 32), width);
          wq = b.create<arith::ExtUIOp>(loc, p.lanes(b.getI32Type(), width), wq);
          if (sub % 2) wq = b.create<arith::ShRUIOp>(loc, wq, p.integer(4, 32, width));
          wq = b.create<arith::AndIOp>(loc, wq, p.integer(15, 32, width));
          Value product = b.create<arith::MulIOp>(loc, wq, xq);
          next.push_back(b.create<arith::AddIOp>(loc, members.getRegionIterArgs()[column], product));
        }
        b.create<scf::YieldOp>(loc, next);
      }
      Value even = p.littleEndian(operation.getRhs(), {record}, 260 + sub * 4, 2);
      Value odd = p.littleEndian(operation.getRhs(), {record}, 262 + sub * 4, 2);
      even = b.create<arith::ExtSIOp>(loc, b.getI32Type(), even);
      odd = b.create<arith::ExtSIOp>(loc, b.getI32Type(), odd);
      Value total = b.create<arith::AddIOp>(loc, even, odd);
      for (int64_t column = 0; column < columns; ++column) {
        auto prefix = leftPrefix(column);
        Value sum = p.reduce(members.getResult(column), vector::CombiningKind::ADD);
        Value scale = p.field6(operation.getLhs(), prefix, 0, sub);
        Value minimum = p.field6(operation.getLhs(), prefix, 1, sub);
        positive[column] = b.create<arith::AddIOp>(loc, positive[column],
            b.create<arith::MulIOp>(loc, sum, scale));
        negative[column] = b.create<arith::AddIOp>(loc, negative[column],
            b.create<arith::MulIOp>(loc, total, minimum));
      }
    }
    Value ds = b.create<arith::BitcastOp>(loc, b.getF32Type(),
        p.littleEndian(operation.getRhs(), {record}, 0, 4));
    SmallVector<Value> next;
    for (int64_t column = 0; column < columns; ++column) {
      auto prefix = leftPrefix(column);
      Value d = b.create<arith::BitcastOp>(loc, b.getF16Type(),
          p.littleEndian(operation.getLhs(), prefix, 0, 2));
      Value dmin = b.create<arith::BitcastOp>(loc, b.getF16Type(),
          p.littleEndian(operation.getLhs(), prefix, 2, 2));
      d = b.create<arith::ExtFOp>(loc, b.getF32Type(), d);
      dmin = b.create<arith::ExtFOp>(loc, b.getF32Type(), dmin);
      Value s = b.create<arith::SIToFPOp>(loc, b.getF32Type(), positive[column]);
      Value t = b.create<arith::SIToFPOp>(loc, b.getF32Type(), negative[column]);
      Value pTerm = b.create<arith::MulFOp>(loc, d, s);
      Value nTerm = b.create<arith::MulFOp>(loc, dmin, t);
      Value difference = b.create<arith::SubFOp>(loc, pTerm, nTerm);
      Value term = b.create<arith::MulFOp>(loc, ds, difference);
      next.push_back(b.create<arith::AddFOp>(loc, records.getRegionIterArgs()[column], term));
    }
    b.create<scf::YieldOp>(loc, next);
  }
  for (int64_t column = 0; column < columns; ++column) {
    SmallVector<Value> position;
    if (grouped) position.push_back(p.index(column));
    b.create<memref::StoreOp>(loc, records.getResult(column), operation.getOutput(), position);
  }
}

LogicalResult materialize(Operation *operation, cpu::ImplementationAttr binding) {
  if (auto dot = dyn_cast<cpu::QuantizedDotOp>(operation)) {
    auto output = cast<MemRefType>(dot.getOutput().getType());
    if (output.getRank() == 1 && (output.isDynamicDim(0) ||
        output.getDimSize(0) != cpu::implementationParameter(binding, "columns")))
      return operation->emitError("grouped record dot must match its selected output cohort");
  }
  OpBuilder b(operation);
  RecordBuilder p{b, operation->getLoc(), binding};
  int64_t width = cpu::implementationParameter(binding, "vector_width");
  SmallVector<Value> memories;
  if (auto quantize = dyn_cast<cpu::QuantizeOp>(operation))
    memories = {quantize.getInput(), quantize.getOutput()};
  else {
    auto dot = cast<cpu::QuantizedDotOp>(operation);
    memories = {dot.getLhs(), dot.getRhs()};
  }
  auto form = [&](int64_t lanes, bool refine) {
    Operation *selected = operation;
    if (refine) {
      IRMapping mapping;
      for (Value memory : memories) mapping.map(memory, contiguousView(p, memory));
      selected = b.clone(*operation, mapping);
    }
    if (auto op = dyn_cast<cpu::QuantizeOp>(selected)) quantize(p, op, lanes);
    else dot(p, cast<cpu::QuantizedDotOp>(selected), lanes);
    if (refine) selected->erase();
  };
  Value condition = width > 1 ? contiguous(p, memories) : Value();
  if (condition && matchPattern(condition, m_Zero())) form(1, false);
  else if (condition) {
    auto branches = b.create<scf::IfOp>(p.loc, condition, true);
    b.setInsertionPointToStart(branches.thenBlock());
    form(width, true);
    b.setInsertionPointToStart(branches.elseBlock());
    form(1, false);
  } else form(width, false);
  operation->erase();
  return success();
}

std::optional<std::string> check(Operation *operation, cpu::CapabilitiesAttr,
                                 const cpu::Configuration &config) {
  int64_t width = config.parameter("vector_width");
  if (width <= 0 || width > 16 || 16 % width)
    return "quantized records require a vector width dividing the sixteen-element sum group";
  Value output;
  SmallVector<Value> inputs;
  if (auto quantize = dyn_cast<cpu::QuantizeOp>(operation)) {
    output = quantize.getOutput(); inputs = {quantize.getInput()};
  } else {
    auto dot = cast<cpu::QuantizedDotOp>(operation);
    output = dot.getOutput(); inputs = {dot.getLhs(), dot.getRhs()};
  }
  if (!isa<MemRefType>(output.getType()) ||
      llvm::any_of(inputs, [](Value input) { return !isa<MemRefType>(input.getType()); }))
    return "native quantized records require destination-passing buffers";
  if (operation->getNumResults())
    return "native quantized records require bufferized results";
  SmallVector<Value> memories(inputs);
  memories.push_back(output);
  for (Value memory : memories) {
    SmallVector<int64_t> strides;
    int64_t offset;
    if (failed(cast<MemRefType>(memory.getType()).getStridesAndOffset(strides, offset)))
      return "native quantized records require strided memory descriptors";
  }
  cpu::StorageAnalysis storage(operation->getParentOfType<func::FuncOp>());
  for (Value input : inputs)
    if (!storage.disjoint(input, output))
      return "native quantized records require disjoint input and output storage";
  return std::nullopt;
}

} // namespace

void registerQuantizedImplementations(cpu::ImplementationRegistry &registry) {
  auto add = [&](StringRef name, bool dot) {
    cpu::Implementation implementation{name, {}, check,
        {cpu::ImplementationParameter::local("vector_width", {true, 16, 32, {}})}, {}, {}};
    implementation.materialize = materialize;
    // The record microprogram has already selected its SIMD shape. In particular,
    // the ascending floating record carry must not become an unordered reduction.
    implementation.vectorize = [](scf::ForOp, cpu::ImplementationAttr) { return success(); };
    if (dot) {
      implementation.parameters.push_back(cpu::ImplementationParameter::constant("columns", 4));
      implementation.parallelWindow = [](cpu::ImplementationAttr binding) {
        return cpu::implementationParameter(binding, "columns");
      };
      registry.add<cpu::QuantizedDotOp>(std::move(implementation));
    } else registry.add<cpu::QuantizeOp>(std::move(implementation));
  };
  add("mojo.q8_k", false);
  add("mojo.q4_k_q8_k", true);
}

LogicalResult materializeQuantizedComputations(
    func::FuncOp function, const cpu::ImplementationRegistry &registry) {
  SmallVector<Operation *> operations;
  function.walk([&](Operation *operation) {
    if (isa<cpu::QuantizeOp, cpu::QuantizedDotOp>(operation)) operations.push_back(operation);
  });
  for (Operation *operation : operations)
    if (failed(registry.materialize(operation))) return failure();
  return success();
}

} // namespace intent::mojo
