#include "TaskConversion.h"
#include "Reductions.h"
#include "ScalarValues.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Intent/Dialect/CPU/Transforms/Structure/ProducerVersions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;
namespace wk = ::weft::kernel;

namespace intent::weft_provider {
using namespace task_detail;

class TaskConversion::Expansion final : public cpu::ImplementationExpansion {
public:
  Expansion(TaskConversion &lowering, Operation *operation)
      : lowering(lowering), operation(operation) {}

  OpBuilder &builder() override { return lowering.b; }
  int64_t &nextAxis() override { return lowering.nextAxis; }

  FailureOr<Value> view(Value source) override {
    auto supplied = lowering.view(source);
    if (failed(supplied))
      return operation->emitError("implementation requires a supplied operand view"),
             failure();
    auto mapping = lowering.viewAxes.find(source);
    if (mapping == lowering.viewAxes.end() ||
        mapping->second.size() != shape((*supplied).getType()).size())
      return operation->emitError("implementation view requires its declared storage rank"),
             failure();
    for (auto [axis, native] : llvm::enumerate(mapping->second))
      if (native != static_cast<int64_t>(axis))
        return operation->emitError("implementation view requires its declared storage-axis order"),
               failure();
    return *supplied;
  }

  FailureOr<Value> read(Value source) override {
    return lowering.read(source);
  }

  LogicalResult write(Value destination, Value value) override {
    return lowering.write(destination, value);
  }

  LogicalResult bind(Value sourceResult, Value nativeValue) override {
    if (sourceResult.getDefiningOp() != operation)
      return operation->emitError("implementation may only bind its own SSA results");
    lowering.values.map(sourceResult, nativeValue);
    return success();
  }

private:
  TaskConversion &lowering;
  Operation *operation;
};

FailureOr<bool> TaskConversion::expandSelected(Operation *operation) {
  if (!operation->hasAttr("intent_cpu.implementation")) return false;
  auto implementation = implementations.lookup(operation);
  if (failed(implementation)) return failure();
  if (!(*implementation)->expand) return false;
  Expansion expansion(*this, operation);
  if (failed((*implementation)->expand(operation, expansion))) return failure();
  for (Value result : operation->getResults())
    if (!values.lookupOrNull(result))
      return operation->emitError("implementation did not bind every SSA result"), failure();
  return true;
}

FailureOr<Value> TaskConversion::binary(Location loc, Value lhs, Value rhs, StringRef kind) {
  return createBinaryValue(b, loc, lhs, rhs, kind);
}

FailureOr<Value> TaskConversion::expression(Operation *operation, IRMapping &mapping) {
  if (auto load = dyn_cast<memref::LoadOp>(operation)) return indexedRead(load, mapping);
  SmallVector<Value> operands;
  for (Value operand : operation->getOperands()) {
    Value mapped = mapping.lookupOrNull(operand);
    if (!mapped)
      return operation->emitError("Weft scalar operand has no current value binding"),
             failure();
    operands.push_back(mapped);
  }
  return lowerScalarValue(b, operation, operands);
}

FailureOr<Value> TaskConversion::mappedInput(Value input, AffineMap map, ArrayRef<int64_t> loopAxes, bool namedAxes) {
  auto loaded = namedAxes ? readNamedAxes(input) : read(input);
  if (failed(loaded)) return failure();
  if (!isa<MemRefType>(input.getType())) return *loaded;
  auto dimensions = shape((*loaded).getType()), ids = axes((*loaded).getType());
  SmallVector<int64_t> keptShape, keptAxes, mappedAxes;
  SmallVector<Attribute> selectors;
  SmallVector<Value> indices;
  auto memoryIds = memoryAxes(input);
  for (auto [axis, id] : llvm::enumerate(ids)) {
    auto position = llvm::find(memoryIds, id);
    if (position == memoryIds.end()) return emitError(input.getLoc(), "Weft input lost its CPU coordinate relation"), failure();
    AffineExpr expression = map.getResult(position - memoryIds.begin());
    if (auto constant = dyn_cast<AffineConstantExpr>(expression)) {
      if (constant.getValue() != 0 || dimensions[axis] != 1)
        return emitError(input.getLoc(), "constant pointwise projection requires a singleton input axis"), failure();
      selectors.push_back(b.getStringAttr("index")); indices.push_back(index(input.getLoc(), 0));
    } else {
      auto dim = dyn_cast<AffineDimExpr>(expression);
      if (!dim) return emitError(input.getLoc(), "Weft computation requires an explicit dimension or singleton indexing map"), failure();
      selectors.push_back(b.getStringAttr("all")); keptShape.push_back(dimensions[axis]); keptAxes.push_back(ids[axis]);
      mappedAxes.push_back(loopAxes[dim.getPosition()]);
    }
  }
  Value result = *loaded;
  if (!indices.empty()) result = b.create<wk::ExtractOp>(input.getLoc(), valueType(element(result.getType()), keptShape, keptAxes),
      result, indices, b.getArrayAttr(selectors));
  if (mappedAxes != keptAxes) result = b.create<wk::ReshapeOp>(input.getLoc(),
      valueType(element(result.getType()), keptShape, mappedAxes), result, array(keptAxes));
  return result;
}

FailureOr<Value> TaskConversion::reduceValue(Location loc, Value input, unsigned axis, StringRef kind) {
  auto dimensions = shape(input.getType()), ids = axes(input.getType());
  SmallVector<int64_t> outputShape(dimensions), outputAxes(ids);
  outputShape.erase(outputShape.begin() + axis); outputAxes.erase(outputAxes.begin() + axis);
  auto countTrue = [&](Value predicate) -> FailureOr<Value> {
    if (dimensions[axis] <= 0 || dimensions[axis] > int64_t(UINT32_MAX))
      return emitError(loc, "boolean reduction requires a statically bounded u32 contribution count"), failure();
    Type count = IntegerType::get(b.getContext(), 32, IntegerType::Unsigned);
    Value zero = b.create<wk::ConstantOp>(loc, count, b.getIntegerAttr(count, 0));
    Value one = b.create<wk::ConstantOp>(loc, count, b.getIntegerAttr(count, 1));
    Value bits = b.create<wk::SelectOp>(loc, valueType(count, dimensions, ids), predicate, one, zero);
    Value total = b.create<wk::ReduceOp>(loc, valueType(count, outputShape, outputAxes), bits, "add", axis);
    Value bound = kind == "and" ? Value(b.create<wk::ConstantOp>(loc, count, b.getIntegerAttr(count, dimensions[axis]))) : zero;
    return Value(b.create<wk::CompareOp>(loc, valueType(b.getI1Type(), outputShape, outputAxes),
                                       total, bound, kind == "and" ? "eq" : "ne"));
  };
  if (kind == "or" || kind == "and") return countTrue(input);
  bool propagating = kind == "maximum" || kind == "minimum";
  Value result = b.create<wk::ReduceOp>(loc, valueType(element(input.getType()), outputShape, outputAxes),
      input, kind == "maximum" ? "max" : kind == "minimum" ? "min" : kind, axis);
  if (propagating) {
    Value isNaN = b.create<wk::CompareOp>(loc, valueType(b.getI1Type(), dimensions, ids), input, input, "ne");
    auto anyNaN = countTrue(isNaN);
    if (failed(anyNaN)) return failure();
    auto floating = cast<FloatType>(element(input.getType()));
    Value nan = b.create<wk::ConstantOp>(loc, floating,
        FloatAttr::get(floating, llvm::APFloat::getQNaN(floating.getFloatSemantics())));
    result = b.create<wk::SelectOp>(loc, result.getType(), *anyNaN, nan, result);
  }
  return result;
}

FailureOr<Value> TaskConversion::reductionContribution(Block &body,
                                       const NativeReduction &native,
                                       IRMapping &mapping) {
  for (Operation &operation : body.without_terminator()) {
    if (&operation == native.combine) continue;
    if (operation.getNumResults() == 1 && mapping.contains(operation.getResult(0)))
      continue;
    for (Value operand : operation.getOperands())
      if (!mapping.contains(operand))
        return operation.emitError("Weft reduction scalar capture has no current value binding"), failure();
    auto value = expression(&operation, mapping);
    if (failed(value)) return failure();
    mapping.map(operation.getResult(0), *value);
  }
  Value contribution = mapping.lookupOrNull(native.contribution);
  if (!contribution)
    return native.combine->emitError("Weft reduction contribution has no current value binding"), failure();
  return contribution;
}

LogicalResult TaskConversion::generic(linalg::GenericOp operation) {
  if (operation.getOutputs().size() != 1 || operation.getNumResults())
    return operation.emitError("Weft CPU legalization requires one buffer-semantics result");
  Value destination = operation.getOutputs()[0];
  SmallVector<int64_t> loopAxes(operation.getNumLoops(), 0);
  auto maps = operation.getIndexingMapsArray();
  auto outputAxes = memoryAxes(destination);
  for (auto [axis, expression] : llvm::enumerate(maps.back().getResults())) {
    auto dim = dyn_cast<AffineDimExpr>(expression);
    if (!dim) return operation.emitError("Weft output traversal requires dimension projections");
    loopAxes[dim.getPosition()] = outputAxes[axis];
  }
  for (int64_t &axis : loopAxes) if (!axis) axis = nextAxis++;
  if (cpu::isMatrixContraction(operation)) {
    auto lhs = mappedInput(operation.getInputs()[0], maps[0], loopAxes, true);
    auto rhs = mappedInput(operation.getInputs()[1], maps[1], loopAxes, true);
    auto initial = read(destination);
    if (failed(lhs) || failed(rhs) || failed(initial)) return failure();
    auto definition = (*initial).getDefiningOp<wk::NewOp>();
    bool zero = definition && matchPattern(definition->getOperand(0), m_PosZeroFloat());
    if (!zero) {
      cpu::StorageAnalysis storage(operation->getParentOfType<func::FuncOp>());
      auto uniform = cpu::queryUniformBufferValue(destination, operation, storage);
      zero = uniform && (isa<FloatType>(uniform->value.getType())
          ? matchPattern(uniform->value, m_PosZeroFloat())
          : matchPattern(uniform->value, m_Zero()));
    }
    if (definition && isa<IntegerType>(element((*initial).getType())))
      if (auto constant = definition->getOperand(0).getDefiningOp<wk::ConstantOp>())
        if (auto value = dyn_cast<IntegerAttr>(constant.getValue())) zero = value.getValue().isZero();
    bool integer = isa<IntegerType>(element((*initial).getType()));
    if (!zero && !integer)
      return operation.emitError("Weft contraction requires an explicit zero-initialized partial; nonzero fused accumulation has no equivalent canonical operation");
    if (!integer) {
      Type accumulator = element((*initial).getType());
      auto promote = [&](Value value) -> Value {
        if (element(value.getType()) == accumulator) return value;
        return b.create<wk::CastOp>(operation.getLoc(),
            valueType(accumulator, shape(value.getType()), axes(value.getType())), value);
      };
      lhs = promote(*lhs);
      rhs = promote(*rhs);
    }
    int64_t reduction = loopAxes[2];
    Value term = b.create<wk::OuterContractOp>(operation.getLoc(), (*initial).getType(),
        *lhs, *rhs, array({reduction}), TypeAttr::get(element((*initial).getType())));
    if (!zero) {
      auto accumulated = binary(operation.getLoc(), *initial, term, "add");
      if (failed(accumulated)) return failure();
      term = *accumulated;
    }
    return write(destination, term);
  }
  auto iterators = operation.getIteratorTypesArray();
  SmallVector<unsigned> reductions;
  for (auto [axis, type] : llvm::enumerate(iterators))
    if (type == utils::IteratorType::reduction) reductions.push_back(axis);
  if (reductions.size() > 1)
    return operation.emitError("Weft structured reduction requires one explicit reduction axis");
  IRMapping mapping = values;
  Block &body = operation.getRegion().front();
  auto extents = operation.getStaticLoopRanges();
  for (linalg::IndexOp coordinate : body.getOps<linalg::IndexOp>()) {
    unsigned axis = coordinate.getDim();
    if (extents[axis] <= 0)
      return coordinate.emitError("Weft coordinate requires its materialized traversal or a static Iota domain");
    auto type = cast<wk::ValueType>(valueType(b.getIndexType(), {extents[axis]}, {loopAxes[axis]}));
    mapping.map(coordinate.getResult(),
        b.create<wk::IotaOp>(coordinate.getLoc(), type, 0, extents[axis]));
  }
  for (auto [number, input] : llvm::enumerate(operation.getInputs())) {
    auto value = mappedInput(input, maps[number], loopAxes, !reductions.empty());
    if (failed(value)) return failure();
    mapping.map(body.getArgument(number), *value);
  }
  if (!reductions.empty()) {
    Value accumulator = body.getArguments().back();
    auto order = operation->getAttrOfType<cpu::ReductionOrderAttr>("intent_cpu.reduction_order");
    auto native = queryNativeReduction(operation, body, accumulator, order);
    if (failed(native)) return failure();
    auto contribution = reductionContribution(body, *native, mapping);
    if (failed(contribution)) return failure();
    auto ids = axes((*contribution).getType());
    auto position = llvm::find(ids, loopAxes[reductions[0]]);
    if (position == ids.end()) return operation.emitError("Weft reduction lost its current logical axis relation");
    auto reduced = reduceValue(operation.getLoc(), *contribution, position - ids.begin(), native->kind);
    auto initial = read(destination);
    if (failed(reduced) || failed(initial)) return failure();
    auto result = binary(operation.getLoc(), *initial, *reduced, native->kind);
    if (failed(result)) return failure();
    return write(destination, *result);
  }
  if (!operation.getIndexingMapsArray().back().isIdentity())
    return operation.emitError("Weft pointwise legalization requires an identity output traversal");
  if (!body.getArguments().back().use_empty()) return operation.emitError("Weft pointwise output is not a pure definition");
  for (Operation &nested : body.without_terminator()) {
    if (nested.getNumResults() == 1 && mapping.contains(nested.getResult(0)))
      continue;
    auto value = expression(&nested, mapping);
    if (failed(value)) return failure();
    mapping.map(nested.getResult(0), *value);
  }
  return write(destination, mapping.lookup(body.getTerminator()->getOperand(0)));
}

LogicalResult TaskConversion::reduction(cpu::ReduceOp operation) {
  Block &body = operation.getCombine().front();
  auto native = queryNativeReduction(operation, body, body.getArgument(0), operation.getOrder());
  if (failed(native)) return failure();
  IRMapping mapping = values;
  SmallVector<int64_t> loopAxes(cast<AffineMapAttr>(operation.getIndexingMaps()[0]).getValue().getNumDims());
  for (int64_t &axis : loopAxes) axis = nextAxis++;
  for (auto [number, input] : llvm::enumerate(operation.getInputs())) {
    auto value = mappedInput(input, cast<AffineMapAttr>(operation.getIndexingMaps()[number]).getValue(), loopAxes, true);
    if (failed(value)) return failure();
    mapping.map(body.getArgument(number + 1), *value);
  }
  auto contribution = reductionContribution(body, *native, mapping);
  if (failed(contribution)) return failure();
  if (shape((*contribution).getType()).size() != 1)
    return operation.emitError("Weft reduction requires one retained logical input axis");
  auto reduced = reduceValue(operation.getLoc(), *contribution, 0, native->kind);
  auto initial = read(operation.getInitial());
  if (failed(reduced) || failed(initial)) return failure();
  auto result = binary(operation.getLoc(), *initial, *reduced, native->kind);
  if (failed(result)) return failure();
  values.map(operation.getResult(), *result);
  return success();
}

} // namespace intent::weft_provider
