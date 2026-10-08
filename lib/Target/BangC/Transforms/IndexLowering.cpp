#include "PassDetail.h"
#include "Intent/Dialect/DSA/Transforms/LocalSupplyRelations.h"
#include "Intent/Dialect/Intent/IR/CompileOptions.h"

using namespace mlir;
namespace intent::bangc {
bool realizeAffineRanges(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<Operation *> comparisons;
  function.walk([&](Operation *operation) {
    if (isa<dsa::CompareRangeOp, dsa::CompareOp>(operation)) comparisons.push_back(operation);
  });
  bool changed = false;
  for (Operation *operation : comparisons) {
    auto missed = [&](StringRef reason) {
      auto options = function->getParentOfType<ModuleOp>()->getAttrOfType<CompileOptionsAttr>(compileOptionsAttr);
      if (options.getOptimizationRemarks()) operation->emitRemark("index-supply: ") << reason;
    };
    Value output = isa<dsa::CompareRangeOp>(operation)
        ? cast<dsa::CompareRangeOp>(operation).getOutput() : cast<dsa::CompareOp>(operation).getOutput();
    auto type = cast<MemRefType>(output.getType());
    int64_t rows = type.getDimSize(0), columns = type.getDimSize(1);
    if (!type.getLayout().isIdentity() || columns < 64 ||
        !llvm::isPowerOf2_64(columns) || type.getNumElements() > 65536) continue;
    dsa::StorageAnalysis storage(function);
    DominanceInfo dominance(function);
    dsa::LocalSupplyRelations relations(function);
    std::optional<dsa::LocalIndexSequence> row, column;
    int64_t base = 0;
    bool strict = false;
    if (auto range = dyn_cast<dsa::CompareRangeOp>(operation)) {
      if (!relations.equal(range.getRows(), rows)) { missed("row range is not a full physical tile"); continue; }
      row = dsa::queryLocalIndexSequence(range.getRowCoordinates(), range, storage, dominance, relations);
      base = range.getBase();
    } else {
      auto compare = cast<dsa::CompareOp>(operation);
      if (compare.getPredicate() != ComparePredicate::Ge && compare.getPredicate() != ComparePredicate::Gt) continue;
      strict = compare.getPredicate() == ComparePredicate::Gt;
      auto grid = dsa::queryLocalCoordinateGrid(compare, storage, relations);
      if (!grid) { missed("comparison is not a closed row/column coordinate grid"); continue; }
      if (!relations.equal(grid->rows.getRows(), rows)) { missed("coordinate grid has an unproven tail"); continue; }
      row = dsa::queryLocalIndexSequence(grid->rows.getSource(), grid->rows, storage, dominance, relations);
      column = dsa::queryLocalIndexSequence(grid->columns.getSource(), grid->columns, storage, dominance, relations);
      if (!row || !column) continue;
      auto bounds = [](dsa::SignedInterval value) {
        return ConstantIntRanges::fromSigned(APInt(64, value->first, true), APInt(64, value->second, true));
      };
      auto lhs = bounds(row->bounds), rhs = bounds(column->bounds);
      if (!provesSignedNoWrap(BinaryOperator::Subtract, lhs, rhs)) { missed("relative coordinate may overflow"); continue; }
      if (strict) {
        auto difference = inferIntegerBinary(BinaryOperator::Subtract, IndexType::get(function.getContext()), lhs, rhs);
        if (!difference || !provesSignedNoWrap(BinaryOperator::Subtract, *difference,
                ConstantIntRanges::constant(APInt(64, 1)))) continue;
      }
    }
    if (!row) continue;
    struct Insertions : OpBuilder::Listener {
      SmallVector<Operation *> operations;
      void notifyOperationInserted(Operation *operation, OpBuilder::InsertPoint) override {
        operations.push_back(operation);
      }
    } inserted;
    OpBuilder b(function.getContext(), &inserted);
    b.setInsertionPoint(operation);
    Location loc = operation->getLoc();
    Value begin = row->materializeBegin(b, loc);
    if (column) begin = b.createOrFold<arith::SubIOp>(loc, begin, column->materializeBegin(b, loc));
    if (strict) begin = b.createOrFold<arith::SubIOp>(loc, begin, b.create<arith::ConstantIndexOp>(loc, 1));
    Value scratch = allocate(b, loc, b.getF32Type(), {1, type.getNumElements()}, dsa::nramSpace);
    b.create<dsa::CompareRampOp>(loc, begin, output, scratch, b.getI64IntegerAttr(base));
    if (!storageFitsBudget(function, config, measureStorage(function))) {
      for (Operation *created : llvm::reverse(inserted.operations)) created->erase();
      continue;
    }
    auto options = function->getParentOfType<ModuleOp>()->getAttrOfType<CompileOptionsAttr>(compileOptionsAttr);
    if (options.getOptimizationRemarks()) operation->emitRemark("index-supply: selected implicit coordinate comparison");
    operation->erase(); changed = true;
  }
  return changed;
}

bool realizeFullWidthMasks(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<dsa::CompareRampOp> comparisons;
  function.walk([&](dsa::CompareRampOp compare) {
    if (cast<MemRefType>(compare.getOutput().getType()).getElementType().isInteger(1)) comparisons.push_back(compare);
  });
  bool changed = false;
  for (auto compare : comparisons) {
    Value previous = compare.getOutput();
    if (!previous.getDefiningOp<memref::AllocaOp>()) continue;
    SmallVector<dsa::SelectOp> selections;
    bool eligible = true;
    for (Operation *user : previous.getUsers()) {
      if (user == compare) continue;
      auto select = dyn_cast<dsa::SelectOp>(user);
      if (!select || select.getCondition() != previous || !select.getFalseValue().getType().isF32() ||
          !cast<MemRefType>(select.getOutput().getType()).getElementType().isF32() ||
          user->getBlock() != compare->getBlock() || !compare->isBeforeInBlock(user)) { eligible = false; break; }
      selections.push_back(select);
    }
    if (!eligible || selections.empty()) continue;
    OpBuilder b(compare); Location loc = compare.getLoc();
    Value mask = allocate(b, loc, b.getI32Type(), cast<MemRefType>(previous.getType()).getShape(), dsa::nramSpace);
    auto packed = b.create<dsa::CompareRampOp>(loc, compare.getRowBegin(), mask, Value{}, compare.getBaseAttr());
    SmallVector<dsa::MaskedFillOp> replacements;
    for (auto select : selections) {
      b.setInsertionPoint(select);
      replacements.push_back(b.create<dsa::MaskedFillOp>(select.getLoc(), mask,
          select.getTrueValue(), select.getFalseValue(), select.getOutput()));
    }
    // Preflight with both representations present. Removing the old boolean
    // producers and conversions can only reduce their storage requirements.
    int64_t nram = 0, wram = 0; measureStorage(function, nram, wram);
    if (nram > config.getLocalBytes() || nram > 768 * 1024 || wram > 1024 * 1024) {
      for (auto replacement : replacements) replacement.erase();
      packed.erase(); mask.getDefiningOp()->erase(); continue;
    }
    SmallVector<Value> unused{previous, compare.getScratch()};
    for (auto select : selections) {
      if (select.getScratch() && !llvm::is_contained(unused, select.getScratch())) unused.push_back(select.getScratch());
      select.erase();
    }
    compare.erase();
    for (Value buffer : unused) if (buffer && buffer.use_empty()) buffer.getDefiningOp()->erase();
    changed = true;
  }
  return changed;
}

bool vectorizeIndexLoops(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<scf::ForOp> loops;
  function.walk([&](scf::ForOp loop) { loops.push_back(loop); });
  DenseMap<int64_t, Value> ramps;
  bool changed = false;
  auto integer64 = [](Type type) { return type.isIndex() || type.isInteger(64); };
  auto constant = [](Value value) -> std::optional<int64_t> {
    APInt bits;
    if (matchPattern(value, m_ConstantInt(&bits))) return bits.getSExtValue();
    return std::nullopt;
  };
  for (auto loop : loops) {
    if (!loop.getInitArgs().empty() || !matchPattern(loop.getLowerBound(), m_Zero()) ||
        !matchPattern(loop.getStep(), m_One())) continue;
    memref::StoreOp store;
    SmallVector<memref::LoadOp> loads;
    bool eligible = true;
    for (Operation &op : loop.getBody()->without_terminator()) {
      if (auto write = dyn_cast<memref::StoreOp>(op)) {
        if (store) { eligible = false; break; }
        store = write;
      } else if (auto read = dyn_cast<memref::LoadOp>(op)) loads.push_back(read);
      else if (isa<dsa::StrideOp>(op)) continue;
      else if (auto divide = dyn_cast<arith::FloorDivSIOp>(op)) {
        auto divisor = constant(divide.getRhs());
        if (!integer64(divide.getType()) || !divisor || *divisor <= 0 || !llvm::isPowerOf2_64(*divisor)) {
          eligible = false; break;
        }
      } else if (!isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::IndexCastOp, arith::ConstantOp>(op) ||
                 !llvm::all_of(op.getOperandTypes(), integer64) || !llvm::all_of(op.getResultTypes(), integer64)) {
        eligible = false; break;
      }
    }
    if (!eligible || !store || !store.getValue().getType().isInteger(64)) continue;
    auto type = store.getMemRefType();
    if (type.getRank() != 2 || !type.hasStaticShape() || type.getDimSize(0) != 1 || type.getDimSize(1) < 256 ||
        !type.getLayout().isIdentity() || type.getMemorySpaceAsInt() != dsa::nramSpace ||
        !store.getMemref().getDefiningOp<memref::AllocaOp>() || store.getIndices().size() != 2 ||
        !matchPattern(store.getIndices()[0], m_Zero()) || store.getIndices()[1] != loop.getInductionVar()) continue;
    int64_t width = type.getDimSize(1);
    std::function<bool(Value)> bounded = [&](Value value) {
      if (auto number = constant(value)) return *number <= width;
      if (auto minimum = value.getDefiningOp<arith::MinSIOp>())
        return bounded(minimum.getLhs()) || bounded(minimum.getRhs());
      return false;
    };
    if (!bounded(loop.getUpperBound()) || llvm::any_of(loads, [&](memref::LoadOp load) {
          return load.getMemRefType() != type || !load.getMemref().getDefiningOp<memref::AllocaOp>() ||
              load.getMemref() == store.getMemref() || load.getIndices().size() != 2 ||
              !matchPattern(load.getIndices()[0], m_Zero()) || load.getIndices()[1] != loop.getInductionVar();
        })) continue;
    Operation *previous = loop->getPrevNode();
    OpBuilder b(loop);
    Location loc = loop.getLoc();
    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    Value one = b.create<arith::ConstantIndexOp>(loc, 1);
    Value active = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, loop.getUpperBound(), zero);
    auto guard = b.create<scf::IfOp>(loc, active, false);
    b.setInsertionPointToStart(&guard.getThenRegion().front());
    Value ramp = ramps.lookup(width);
    dsa::IotaOp initialization;
    dsa::IndexLayoutOp rampLayout;
    Value interleavedRamp;
    if (!ramp) {
      OpBuilder init(function.getContext());
      init.setInsertionPointToStart(&function.front());
      interleavedRamp = allocate(init, loc, init.getI64Type(), {1, width}, dsa::nramSpace);
      initialization = init.create<dsa::IotaOp>(loc, interleavedRamp);
      ramp = allocate(init, loc, init.getI32Type(), {2, width}, dsa::nramSpace);
      rampLayout = init.create<dsa::IndexLayoutOp>(loc, interleavedRamp, ramp);
    }
    DenseMap<Value, Value> mapped;
    mapped[loop.getInductionVar()] = ramp;
    DenseMap<Value, Value> copiedInputs;
    auto tile = [&]() { return allocate(b, loc, b.getI32Type(), {2, width}, dsa::nramSpace); };
    auto materialize = [&](Value scalar) {
      Value result = tile();
      b.create<dsa::IndexLayoutOp>(loc, scalar, result);
      return result;
    };
    std::function<Value(Value)> emit = [&](Value value) -> Value {
      if (Value known = mapped.lookup(value)) return known;
      if (auto number = constant(value)) return mapped[value] = b.create<arith::ConstantIntOp>(loc, *number, 64);
      if (loop.isDefinedOutsideOfLoop(value)) {
        Value result = value.getType().isIndex() ? Value(b.create<arith::IndexCastOp>(loc, b.getI64Type(), value)) : value;
        return mapped[value] = result;
      }
      Operation *definition = value.getDefiningOp();
      if (auto load = dyn_cast<memref::LoadOp>(definition)) {
        Value source = load.getMemref();
        Value copied = copiedInputs.lookup(source);
        if (!copied) {
          Value interleaved = allocate(b, loc, b.getI64Type(), {1, width}, dsa::nramSpace);
          b.create<dsa::LoadTileOp>(loc, source, interleaved, zero, zero, one, one, loop.getUpperBound());
          copied = tile();
          b.create<dsa::IndexLayoutOp>(loc, interleaved, copied);
          copiedInputs[source] = copied;
        }
        return mapped[value] = copied;
      }
      if (auto cast = dyn_cast<arith::IndexCastOp>(definition)) return mapped[value] = emit(cast.getIn());
      if (auto stride = dyn_cast<dsa::StrideOp>(definition)) {
        Value scalar = b.create<dsa::StrideOp>(loc, b.getIndexType(), stride.getSource(), stride.getAxis());
        return mapped[value] = b.create<arith::IndexCastOp>(loc, b.getI64Type(), scalar);
      }
      Value lhs = emit(definition->getOperand(0)), rhs = emit(definition->getOperand(1));
      bool leftTile = isa<MemRefType>(lhs.getType()), rightTile = isa<MemRefType>(rhs.getType());
      if (!leftTile && !rightTile) {
        OperationState state(loc, definition->getName());
        state.addOperands({lhs, rhs}); state.addTypes(b.getI64Type()); state.addAttributes(definition->getAttrs());
        return mapped[value] = b.create(state)->getResult(0);
      }
      BinaryOperator kind = isa<arith::AddIOp>(definition) ? BinaryOperator::Add :
          isa<arith::SubIOp>(definition) ? BinaryOperator::Subtract :
          isa<arith::MulIOp>(definition) ? BinaryOperator::Multiply : BinaryOperator::RightShift;
      if (kind == BinaryOperator::RightShift)
        rhs = b.create<arith::ConstantIntOp>(loc, llvm::Log2_64(*constant(definition->getOperand(1))), 64);
      if (!leftTile && (kind == BinaryOperator::Add || kind == BinaryOperator::Multiply)) {
        std::swap(lhs, rhs); leftTile = true;
      }
      if (!leftTile) lhs = materialize(lhs);
      // Scalars are split into full-width low/high words by the selected
      // implementation; no signed-48-bit immediate restriction is needed.
      bool scalar = !isa<MemRefType>(rhs.getType());
      auto number = scalar ? constant(rhs) : std::optional<int64_t>{};
      if (kind == BinaryOperator::Multiply && number && *number > 0 && llvm::isPowerOf2_64(*number)) {
        kind = BinaryOperator::LeftShift;
        number = llvm::Log2_64(*number);
        rhs = b.create<arith::ConstantIntOp>(loc, *number, 64);
      }
      Value result = tile();
      b.create<dsa::IndexBinaryOp>(loc, lhs, rhs, result, BinaryOperatorAttr::get(b.getContext(), kind));
      return mapped[value] = result;
    };
    Value result = emit(store.getValue());
    if (!isa<MemRefType>(result.getType())) result = materialize(result);
    Value interleaved = allocate(b, loc, b.getI64Type(), {1, width}, dsa::nramSpace);
    b.create<dsa::IndexLayoutOp>(loc, result, interleaved);
    b.create<dsa::StoreTileOp>(loc, interleaved, store.getMemref(), zero, zero, one, one, loop.getUpperBound());
    int64_t nram = 0, wram = 0;
    measureStorage(function, nram, wram);
    if (nram + 512 > config.getLocalBytes() || nram + 512 > 768 * 1024 || wram > 1024 * 1024) {
      while (loop->getPrevNode() != previous) loop->getPrevNode()->erase();
      if (initialization) {
        rampLayout.erase(); initialization.erase();
        ramp.getDefiningOp()->erase(); interleavedRamp.getDefiningOp()->erase();
      }
      continue;
    }
    ramps[width] = ramp;
    loop.erase();
    changed = true;
  }
  return changed;
}
} // namespace intent::bangc
