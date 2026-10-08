#include "Intent/Dialect/DSA/Transforms/LocalSupplyRelations.h"
#include "Intent/Dialect/DSA/Transforms/StoragePatterns.h"
#include "Intent/Dialect/Intent/IR/CompileOptions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;
namespace intent::dsa {
namespace {

std::optional<LocalIndexSequence> rejectSequence(Operation *reader, StringRef reason) {
  auto module = reader->getParentOfType<ModuleOp>();
  auto options = module->getAttrOfType<CompileOptionsAttr>(compileOptionsAttr);
  if (options.getOptimizationRemarks()) reader->emitRemark("index-supply: ") << reason;
  return std::nullopt;
}

bool stableAfter(Value memory, Operation *writer, Operation *reader,
                 StorageAnalysis &storage) {
  Operation *scope = writer->getBlock()->findAncestorOpInBlock(*reader);
  if (!scope || scope == writer || !storage.unchangedBetween(memory, writer, scope))
    return false;
  return scope == reader || storage.preserves(scope, memory);
}

std::optional<BinaryOperator> integerOperator(Operation *operation) {
  if (isa<arith::AddIOp>(operation)) return BinaryOperator::Add;
  if (isa<arith::SubIOp>(operation)) return BinaryOperator::Subtract;
  if (isa<arith::MulIOp>(operation)) return BinaryOperator::Multiply;
  if (isa<arith::MinSIOp>(operation)) return BinaryOperator::Minimum;
  if (isa<arith::MaxSIOp>(operation)) return BinaryOperator::Maximum;
  return std::nullopt;
}

IntegerRangePolicy supplyPolicy(func::FuncOp function,
    std::function<std::optional<int64_t>(Value)> constant) {
  IntegerRangePolicy policy;
  policy.infer = [function, constant](Value value, IntegerRangeAnalysis &analysis)
      -> std::optional<ConstantIntRanges> {
    if (value.getDefiningOp<dsa::TaskIdOp>() || value.getDefiningOp<dsa::TaskCountOp>()) {
      if (auto bounds = dsa::integerInterval(value, function))
        return ConstantIntRanges::fromSigned(APInt(64, bounds->first), APInt(64, bounds->second));
      return std::nullopt;
    }
    auto bound = [&](Value value) {
      auto range = analysis.range(value);
      // The existing no-wrap affine relation can retain correlation lost by
      // interval arithmetic, e.g. a clipped tile end minus its own begin.
      if (auto exact = constant(value))
        range = ConstantIntRanges::constant(APInt(64, *exact, true));
      return range;
    };
    if (Operation *operation = value.getDefiningOp(); operation &&
        (value.getType().isIndex() || value.getType().isInteger(64))) {
      if (auto kind = integerOperator(operation)) {
        auto lhs = bound(operation->getOperand(0));
        auto rhs = bound(operation->getOperand(1));
        if (lhs && rhs)
          return inferIntegerBinary(*kind, value.getType(), *lhs, *rhs);
      }
    }
    auto argument = dyn_cast<BlockArgument>(value);
    auto loop = argument ? dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp()) : scf::ForOp{};
    if (!loop || loop.getInductionVar() != value || !value.getType().isIndex())
      return std::nullopt;
    auto lower = bound(loop.getLowerBound());
    auto upper = bound(loop.getUpperBound());
    auto step = bound(loop.getStep());
    if (!lower || !upper || !step || lower->smin() != lower->smax() ||
        step->smin() != step->smax() || step->smin().sle(0) ||
        upper->smax().sle(lower->smin()))
      return std::nullopt;
    // The last actual IV is aligned to the positive step. Widen the arithmetic
    // so deriving this bound does not assume an independently computed add nsw.
    APInt begin = lower->smin().sext(128), end = upper->smax().sext(128);
    APInt stride = step->smin().sext(128);
    APInt last = begin + (end - begin - 1).sdiv(stride) * stride;
    return ConstantIntRanges::fromSigned(lower->smin(), last.trunc(64));
  };
  return policy;
}

} // namespace

LocalSupplyRelations::LocalSupplyRelations(func::FuncOp function)
    : function(function), ranges(supplyPolicy(function,
          [this](Value value) { return constant(value); })) {}

dsa::SignedInterval LocalSupplyRelations::interval(Value value) {
  auto bounds = ranges.range(value);
  if (!bounds || bounds->smin().getBitWidth() > 64) return std::nullopt;
  return std::make_pair(bounds->smin().getSExtValue(), bounds->smax().getSExtValue());
}

std::optional<int64_t> LocalSupplyRelations::constant(Value value) {
  AffineExpr expr = expression(value);
  auto folded = dyn_cast<AffineConstantExpr>(simplifyAffineExpr(expr, 0, symbols));
  return folded ? std::optional<int64_t>(folded.getValue()) : std::nullopt;
}

bool LocalSupplyRelations::equal(Value value, int64_t expected) {
  return constant(value) == expected;
}

AffineExpr LocalSupplyRelations::difference(Value first, Value second) {
  AffineExpr expr = expression(first) - expression(second);
  return simplifyAffineExpr(expr, 0, symbols);
}

bool LocalSupplyRelations::equal(Value first, Value second) {
  auto folded = dyn_cast<AffineConstantExpr>(difference(first, second));
  return folded && folded.getValue() == 0;
}

AffineExpr LocalSupplyRelations::expression(Value value) {
  if (auto found = expressions.find(value); found != expressions.end()) return found->second;
  auto infer = [&]() -> AffineExpr {
    auto bounds = interval(value);
    if (bounds && bounds->first == bounds->second)
      return getAffineConstantExpr(bounds->first, function.getContext());
    Operation *op = value.getDefiningOp();
    if (op && isa<arith::IndexCastOp, arith::ExtSIOp>(op)) {
      auto input = ranges.range(op->getOperand(0));
      if (input && isValuePreservingIntegerCast(op->getOperand(0).getType(), value.getType(), *input))
        return expression(op->getOperand(0));
    }
    if (!op || (!value.getType().isIndex() && !value.getType().isInteger(64)) ||
        op->getNumOperands() != 2)
      return getAffineSymbolExpr(symbols++, function.getContext());
    Value lhs = op->getOperand(0), rhs = op->getOperand(1);
    auto operandBounds = [&](Value operand) {
      auto bounds = interval(operand);
      if (auto exact = constant(operand)) bounds = std::make_pair(*exact, *exact);
      return bounds;
    };
    auto left = operandBounds(lhs), right = operandBounds(rhs);
    if (left && right) {
      if (isa<arith::MinSIOp>(op)) {
        if (left->second <= right->first) return expression(lhs);
        if (right->second <= left->first) return expression(rhs);
      }
      if (isa<arith::MaxSIOp>(op)) {
        if (left->first >= right->second) return expression(lhs);
        if (right->first >= left->second) return expression(rhs);
      }
      auto kind = integerOperator(op);
      if (kind == BinaryOperator::Minimum || kind == BinaryOperator::Maximum)
        kind.reset();
      if (kind && provesSignedNoWrap(*kind,
              ConstantIntRanges::fromSigned(APInt(64, left->first), APInt(64, left->second)),
              ConstantIntRanges::fromSigned(APInt(64, right->first), APInt(64, right->second)))) {
        if (*kind == BinaryOperator::Add) return expression(lhs) + expression(rhs);
        if (*kind == BinaryOperator::Subtract) return expression(lhs) - expression(rhs);
        return expression(lhs) * expression(rhs);
      }
    }
    return getAffineSymbolExpr(symbols++, function.getContext());
  };
  AffineExpr result = infer();
  expressions[value] = result;
  return result;
}

Value LocalIndexSequence::materializeBegin(OpBuilder &builder, Location location) const {
  if (auto constant = dyn_cast<AffineConstantExpr>(begin))
    return builder.create<arith::ConstantIndexOp>(location, constant.getValue());
  IRMapping mapping;
  auto loop = initialization;
  mapping.map(loop.getInductionVar(), builder.create<arith::ConstantIndexOp>(location, 0));
  for (Operation *operation : definitions) builder.clone(*operation, mapping);
  Value result = mapping.lookupOrDefault(value);
  if (!result.getType().isIndex())
    result = builder.create<arith::IndexCastOp>(location, builder.getIndexType(), result);
  return result;
}

std::optional<LocalIndexSequence> queryLocalIndexSequence(
    Value memory, Operation *reader, StorageAnalysis &storage,
    DominanceInfo &dominance, LocalSupplyRelations &relations) {
  while (true) {
    auto type = dyn_cast<MemRefType>(memory.getType());
    Value origin = storage.uniqueOrigin(memory);
    if (!type || type.getRank() != 2 || !type.hasStaticShape() || type.getNumElements() <= 0 ||
        !type.getLayout().isIdentity() || (type.getDimSize(0) != 1 && type.getDimSize(1) != 1) ||
        !type.getElementType().isInteger(64) || type.getMemorySpaceAsInt() != nramSpace ||
        !origin || !origin.getDefiningOp<memref::AllocaOp>() ||
        !isCompleteStorageViewOf(memory, origin) || !storage.aliases(origin).complete)
      return rejectSequence(reader, "coordinate source is not a complete dense local row or column");
    int64_t count = type.getNumElements();
    Operation *writer = storage.lastWriterBefore(memory, reader);
    Value copied;
    if (auto copy = dyn_cast_or_null<memref::CopyOp>(writer)) {
      if (isCompleteStorageViewOf(copy.getTarget(), origin)) copied = copy.getSource();
    } else if (auto load = dyn_cast_or_null<LoadTileOp>(writer)) {
      auto output = cast<MemRefType>(load.getOutput().getType());
      if (!load.getAsynchronous() && isCompleteStorageViewOf(load.getOutput(), origin) &&
          relations.equal(load.getOffset(), 0) &&
          relations.equal(load.getRows(), output.getDimSize(0)) &&
          relations.equal(load.getColumns(), output.getDimSize(1)) &&
          (output.getDimSize(0) == 1 || relations.equal(load.getRowStride(), output.getDimSize(1))) &&
          (output.getDimSize(1) == 1 || relations.equal(load.getColumnStride(), 1)))
        copied = load.getSource();
    }
    if (copied) {
      auto source = cast<MemRefType>(copied.getType());
      if (!source.hasStaticShape() || source.getNumElements() != count ||
          !dominance.properlyDominates(writer, reader) ||
          !storage.disjoint(copied, memory) || !stableAfter(memory, writer, reader, storage))
        return rejectSequence(reader, "coordinate copy does not preserve a complete current snapshot");
      memory = copied;
      reader = writer;
      continue;
    }

    auto writers = storage.writers(memory);
    if (failed(writers)) return rejectSequence(reader, "coordinate writer effects are incomplete");
    memref::StoreOp store;
    SmallVector<FillOp> fills;
    for (Operation *operation : *writers) {
      if (auto item = dyn_cast<memref::StoreOp>(operation);
          item && isCompleteStorageViewOf(item.getMemref(), origin)) {
        if (store) return rejectSequence(reader, "coordinate storage has more than one scalar writer");
        store = item;
      } else if (auto fill = dyn_cast<FillOp>(operation);
                 fill && isCompleteStorageViewOf(fill.getOutput(), origin)) {
        fills.push_back(fill);
      } else return rejectSequence(reader, "coordinate storage has a non-ramp writer");
    }
    auto loop = store ? dyn_cast<scf::ForOp>(store->getParentOp()) : scf::ForOp{};
    if (!loop || !loop.getInitArgs().empty() || loop->isProperAncestor(reader) ||
        !dominance.properlyDominates(loop, reader))
      return rejectSequence(reader, "coordinate ramp has not completed at this read point");
    if (
        !relations.equal(loop.getLowerBound(), 0) ||
        !relations.equal(loop.getUpperBound(), count) || !relations.equal(loop.getStep(), 1))
      return rejectSequence(reader, "coordinate ramp does not cover its complete physical capacity");
    if (
        llvm::any_of(fills, [&](FillOp fill) { return !dominance.properlyDominates(fill, loop); }) ||
        !stableAfter(memory, loop, reader, storage))
      return rejectSequence(reader, "coordinate snapshot may change after initialization");
    auto shape = store.getMemRefType().getShape();
    auto indices = store.getIndices();
    if (!store.getMemRefType().getLayout().isIdentity() || indices.size() != 2 ||
        !((shape == ArrayRef<int64_t>({1, count}) && relations.equal(indices[0], 0) &&
           indices[1] == loop.getInductionVar()) ||
          (shape == ArrayRef<int64_t>({count, 1}) && indices[0] == loop.getInductionVar() &&
           relations.equal(indices[1], 0))))
      return rejectSequence(reader, "coordinate store is not the exact linear lane");

    LocalIndexSequence result{loop, store.getValue(), {}, {}, {}};
    DenseMap<Value, int64_t> coefficients;
    std::function<std::optional<int64_t>(Value)> coefficient = [&](Value value) -> std::optional<int64_t> {
      if (value == loop.getInductionVar()) return 1;
      if (auto found = coefficients.find(value); found != coefficients.end()) return found->second;
      Operation *operation = value.getDefiningOp();
      if (!operation || !loop->isProperAncestor(operation))
        return dominance.properlyDominates(value, reader) ? std::optional<int64_t>(0) : std::nullopt;
      if (operation->getNumRegions() || !isMemoryEffectFree(operation) || !isSpeculatable(operation))
        return std::nullopt;
      std::optional<int64_t> slope;
      if (isa<arith::ConstantOp>(operation)) slope = 0;
      else if (isa<arith::IndexCastOp, arith::ExtSIOp>(operation)) {
        auto bounds = relations.interval(operation->getOperand(0));
        Type from = operation->getOperand(0).getType();
        unsigned width = from.isIndex() ? 64 : cast<IntegerType>(from).getWidth();
        if (bounds && isValuePreservingIntegerCast(from, value.getType(),
                ConstantIntRanges::fromSigned(APInt(width, bounds->first, true), APInt(width, bounds->second, true))))
          slope = coefficient(operation->getOperand(0));
      } else if (isa<arith::AddIOp, arith::SubIOp>(operation) &&
                 (value.getType().isIndex() || value.getType().isInteger(64))) {
        auto lhs = relations.interval(operation->getOperand(0));
        auto rhs = relations.interval(operation->getOperand(1));
        auto kind = isa<arith::AddIOp>(operation) ? BinaryOperator::Add : BinaryOperator::Subtract;
        bool nowrap = lhs && rhs && provesSignedNoWrap(kind,
                ConstantIntRanges::fromSigned(APInt(64, lhs->first, true), APInt(64, lhs->second, true)),
                ConstantIntRanges::fromSigned(APInt(64, rhs->first, true), APInt(64, rhs->second, true)));
        if (nowrap) {
          auto a = coefficient(operation->getOperand(0)), b = coefficient(operation->getOperand(1));
          if (a && b) {
            int64_t total = kind == BinaryOperator::Add ? *a + *b : *a - *b;
            if (total >= -1 && total <= 1) slope = total;
          }
        }
      }
      if (!slope) return std::nullopt;
      coefficients[value] = *slope;
      result.definitions.push_back(operation);
      return slope;
    };
    if (coefficient(store.getValue()) != std::optional<int64_t>(1))
      return rejectSequence(reader, "coordinate expression is not a pure non-wrapping unit sequence");
    result.bounds = relations.interval(store.getValue());
    if (!result.bounds) return rejectSequence(reader, "coordinate bounds are unknown");
    result.begin = relations.difference(store.getValue(), loop.getInductionVar());
    return result;
  }
}

std::optional<LocalCoordinateGrid> queryLocalCoordinateGrid(
    CompareOp comparison, StorageAnalysis &storage, LocalSupplyRelations &relations) {
  auto type = cast<MemRefType>(comparison.getLhs().getType());
  auto output = cast<MemRefType>(comparison.getOutput().getType());
  if (!type.getElementType().isInteger(64) || type.getRank() != 2 || !type.hasStaticShape() ||
      !output.getLayout().isIdentity())
    return std::nullopt;
  auto broadcast = [&](Value memory, bool rows) -> LoadTileOp {
    auto load = dyn_cast_or_null<LoadTileOp>(storage.lastWriterBefore(memory, comparison));
    if (!load || load.getAsynchronous() || !detail::sameCompleteView(storage, load.getOutput(), memory) ||
        !storage.disjoint(load.getSource(), memory) || !stableAfter(memory, load, comparison, storage) ||
        !relations.equal(load.getOffset(), 0) || !relations.equal(load.getRowStride(), rows ? 1 : 0) ||
        !relations.equal(load.getColumnStride(), rows ? 0 : 1) ||
        !relations.equal(load.getColumns(), type.getDimSize(1))) return {};
    auto source = cast<MemRefType>(load.getSource().getType());
    if (!source.getLayout().isIdentity() || source.getMemorySpaceAsInt() != nramSpace ||
        !isCompleteLocalStorageView(load.getSource())) return {};
    SmallVector<int64_t> expected = rows ? SmallVector<int64_t>{type.getDimSize(0), 1}
                                        : SmallVector<int64_t>{1, type.getDimSize(1)};
    return source.getShape() == ArrayRef<int64_t>(expected) ? load : LoadTileOp{};
  };
  auto rows = broadcast(comparison.getLhs(), true), columns = broadcast(comparison.getRhs(), false);
  if (!rows || !columns || !relations.equal(rows.getRows(), columns.getRows())) return std::nullopt;
  return LocalCoordinateGrid{rows, columns};
}

} // namespace intent::dsa
