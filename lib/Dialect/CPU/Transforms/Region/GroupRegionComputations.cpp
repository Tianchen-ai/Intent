#include "Intent/Dialect/CPU/Transforms/Configuration/Configuration.h"
#include "Intent/Dialect/CPU/Transforms/Region/Regions.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Intent/Dialect/CPU/Transforms/Structure/LoopBuilders.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallBitVector.h"

using namespace mlir;
namespace intent::cpu {
namespace {

struct RegionGroup {
  scf::ParallelOp parallel;
  RegionFoldOp fold;
  int64_t width = 0;
  llvm::DenseMap<Operation *, unsigned> headViews;
  llvm::DenseSet<Value> lifted;

  bool mark(Value value) {
    return isa<MemRefType>(value.getType()) && lifted.insert(value).second;
  }

  bool batched(linalg::GenericOp operation) const {
    for (Value operand : operation->getOperands())
      if (lifted.contains(operand)) return true;
    return operation.getBody()->walk([&](memref::LoadOp load) {
      return lifted.contains(load.getMemref()) ? WalkResult::interrupt() : WalkResult::advance();
    }).wasInterrupted();
  }

  bool analyze(func::FuncOp function, int64_t rows) {
    if (!parallel.getNumLoops() || parallel.getNumResults() ||
        getConstantIntValue(parallel.getLowerBound().back()) != 0 ||
        getConstantIntValue(parallel.getStep().back()) != 1) return false;
    auto interface = function->getAttrOfType<EntryRequirementsAttr>(entryRequirementsAttr);
    if (!interface || !interface.getDisjointOutputs()) return false;
    for (auto candidate : parallel.getBody()->getOps<RegionFoldOp>()) {
      if (fold) return false;
      fold = candidate;
    }
    if (!fold) return false;
    Value head = parallel.getInductionVars().back();
    int64_t divisor = 0;
    StorageAnalysis storage(function);
    for (Operation *user : head.getUsers()) {
      if (auto quotient = dyn_cast<arith::FloorDivSIOp>(user)) {
        auto period = getConstantIntValue(quotient.getRhs());
        if (quotient.getLhs() != head || !period || *period <= 1 ||
            (divisor && divisor != *period)) return false;
        divisor = *period;
        continue;
      }
      auto view = dyn_cast<memref::SubViewOp>(user);
      if (!view || view->getBlock() != parallel.getBody()) return false;
      auto external = storage.externalView(view.getSource());
      if (!external || (external.getAccess() != 0 && external.getAccess() != 1)) return false;
      auto offsets = view.getMixedOffsets(), sizes = view.getMixedSizes();
      auto strides = view.getMixedStrides();
      if (llvm::is_contained(sizes, OpFoldResult(head)) ||
          llvm::is_contained(strides, OpFoldResult(head))) return false;
      auto dropped = view.getDroppedDims();
      std::optional<unsigned> selected;
      for (auto [axis, offset] : llvm::enumerate(offsets)) {
        if (offset != OpFoldResult(head)) continue;
        if (selected || !dropped.test(axis) || getConstantIntValue(sizes[axis]) != 1 ||
            getConstantIntValue(strides[axis]) != 1) return false;
        selected = axis;
      }
      if (!selected) return false;
      // The new cohort coordinate precedes every retained source coordinate.
      for (unsigned axis = 0; axis < *selected; ++axis)
        if (!dropped.test(axis)) return false;
      headViews[view] = *selected;
      mark(view.getResult());
    }
    width = std::min(rows, divisor);
    if (width <= 1 || divisor % width || headViews.empty()) return false;
    auto program = cast<RegionOpInterface>(fold.getOperation());
    for (Value value : program.getIdentities()) {
      auto type = dyn_cast<MemRefType>(value.getType());
      if (!type || !type.getRank() || type.getDimSize(0) != 1) return false;
      mark(value);
    }
    for (Value value : program.getDestinations()) mark(value);
    bool changed = true;
    while (changed) {
      changed = false;
      for (Region &helper : program->getRegions()) {
        auto schema = program.getRegionSchema(helper);
        if (failed(schema)) return false;
        for (const auto &relation : *schema)
          if (lifted.contains(relation.prototype)) changed |= mark(relation.argument);
      }
      parallel->walk([&](Operation *operation) {
        if (auto generic = dyn_cast<linalg::GenericOp>(operation)) {
          if (batched(generic))
            for (Value output : generic.getOutputs()) changed |= mark(output);
        } else if (auto copy = dyn_cast<memref::CopyOp>(operation)) {
          if (lifted.contains(copy.getSource())) changed |= mark(copy.getTarget());
        } else if (auto view = dyn_cast<memref::SubViewOp>(operation)) {
          if (lifted.contains(view.getSource())) changed |= mark(view.getResult());
        } else if (auto cast = dyn_cast<memref::CastOp>(operation)) {
          if (lifted.contains(cast.getSource())) changed |= mark(cast.getResult());
        } else if (unitReshapeAxes(operation)) {
          if (lifted.contains(operation->getOperand(0))) changed |= mark(operation->getResult(0));
        }
      });
    }
    if (llvm::any_of(program.getSources(), [&](Value value) { return lifted.contains(value); }) ||
        llvm::none_of(program.getCaptures(), [&](Value value) { return lifted.contains(value); })) return false;
    bool valid = true;
    parallel->walk([&](Operation *operation) {
      if (operation == parallel || operation == fold ||
          isa<RegionYieldOp, scf::YieldOp, scf::ReduceOp>(operation)) return;
      if (auto generic = dyn_cast<linalg::GenericOp>(operation)) {
        if (generic.getNumResults() || generic.getNumDpsInits() != 1 ||
            llvm::any_of(generic.getIndexingMapsArray(), [](AffineMap map) { return map.getNumSymbols(); })) valid = false;
        if (batched(generic)) {
          auto ranges = generic.getStaticLoopRanges();
          for (auto [extent, iterator] : llvm::zip(ranges, generic.getIteratorTypesArray()))
            if (extent == 1 && iterator == utils::IteratorType::reduction) valid = false;
        }
        generic.getBody()->walk([&](Operation *nested) {
          if (isa<memref::LoadOp, linalg::IndexOp, linalg::YieldOp>(nested)) return;
          if (nested->getNumRegions() || !isMemoryEffectFree(nested) ||
              llvm::any_of(nested->getResultTypes(), [](Type type) { return isa<MemRefType>(type); })) valid = false;
        });
        return;
      }
      if (isa<linalg::FillOp, linalg::YieldOp, linalg::IndexOp>(operation)) return;
      if (unitReshapeAxes(operation)) {
        valid &= lifted.contains(operation->getOperand(0)) == lifted.contains(operation->getResult(0));
        return;
      }
      if (auto end = dyn_cast<memref::DeallocOp>(operation)) {
        auto allocation = end.getMemref().getDefiningOp<memref::AllocOp>();
        valid &= allocation && parallel->isAncestor(allocation);
        return;
      }
      if (auto dimension = dyn_cast<memref::DimOp>(operation)) {
        if (lifted.contains(dimension.getSource()) && !dimension.getConstantIndex()) valid = false;
        return;
      }
      if (auto copy = dyn_cast<memref::CopyOp>(operation)) {
        if (lifted.contains(copy.getSource()) || lifted.contains(copy.getTarget()))
          valid &= copy.getSource().getType().getShape() == copy.getTarget().getType().getShape();
        return;
      }
      if (auto allocation = dyn_cast<memref::AllocOp>(operation)) {
        valid &= allocation.getType().getLayout().isIdentity();
        return;
      }
      if (auto view = dyn_cast<memref::SubViewOp>(operation)) {
        if (!lifted.contains(view.getResult()) || headViews.count(view)) return;
        if (!lifted.contains(view.getSource())) { valid = false; return; }
        auto source = cast<MemRefType>(view.getSource().getType());
        for (int64_t axis = 0; axis < source.getRank(); ++axis)
          if (source.getDimSize(axis) == 1 &&
              (getConstantIntValue(view.getMixedOffsets()[axis]) != 0 ||
               getConstantIntValue(view.getMixedSizes()[axis]) != 1)) valid = false;
        return;
      }
      if (auto conversion = dyn_cast<memref::CastOp>(operation)) {
        valid &= lifted.contains(conversion.getSource()) == lifted.contains(conversion.getResult());
        if (lifted.contains(conversion.getSource())) {
          auto source = cast<MemRefType>(conversion.getSource().getType());
          auto result = cast<MemRefType>(conversion.getType());
          for (auto [from, to] : llvm::zip(source.getShape(), result.getShape()))
            if ((from == 1) != (to == 1)) valid = false;
          // A unit row's original contiguity says nothing about the new
          // cohort stride. Only a fresh dense allocation proves both.
          if (result.getLayout().isIdentity() &&
              !conversion.getSource().getDefiningOp<memref::AllocOp>()) valid = false;
        }
        return;
      }
      if (auto branch = dyn_cast<scf::IfOp>(operation)) {
        for (Region &region : branch->getRegions())
          for (Block &body : region)
            for (Value value : body.getTerminator()->getOperands())
              if (lifted.contains(value)) valid = false;
        return;
      }
      if (auto load = dyn_cast<memref::LoadOp>(operation)) {
        if (lifted.contains(load.getMemref()) && !load->getParentOfType<linalg::GenericOp>()) valid = false;
        return;
      }
      if (isa<memref::StoreOp>(operation)) {
        valid = false;
        return;
      }
      if (!isMemoryEffectFree(operation) || operation->getNumRegions() ||
          llvm::any_of(operation->getOperandTypes(), [](Type type) { return isa<MemRefType>(type); }) ||
          llvm::any_of(operation->getResultTypes(), [](Type type) { return isa<MemRefType>(type); })) valid = false;
    });
    for (Value value : lifted) {
      if (auto argument = dyn_cast<BlockArgument>(value)) {
        if (argument.getOwner()->getParentOp() != fold) valid = false;
      } else if (!value.getDefiningOp() || !parallel->isAncestor(value.getDefiningOp())) valid = false;
    }
    // ABI disjointness separates input/output buffers. Each writable buffer
    // must additionally use one injective head-row projection; private and
    // helper destinations may not escape the original parallel iteration.
    llvm::DenseMap<Value, Operation *> outputRows;
    auto effects = storage.effects(parallel);
    if (!effects.complete || effects.ordered) return false;
    for (const StorageEffect &entry : effects.entries) {
      const auto &effect = entry.effect;
      if (isa<MemoryEffects::Allocate, MemoryEffects::Free>(effect.getEffect())) continue;
      Value memory = effect.getValue();
      if (!memory || !isa<MemRefType>(memory.getType()) ||
          !isa<MemoryEffects::Read, MemoryEffects::Write>(effect.getEffect())) return false;
      Value origin = storage.uniqueOrigin(memory);
      if (!origin) return false;
      auto external = storage.externalView(origin);
      bool write = isa<MemoryEffects::Write>(effect.getEffect());
      if (external) {
        if (!write && external.getAccess() != 0) valid = false;
        if (!write) continue;
        Value projection = memory;
        while (!headViews.count(projection.getDefiningOp())) {
          if (auto view = projection.getDefiningOp<memref::SubViewOp>()) projection = view.getSource();
          else if (auto cast = projection.getDefiningOp<memref::CastOp>()) projection = cast.getSource();
          else if (unitReshapeAxes(projection.getDefiningOp())) projection = projection.getDefiningOp()->getOperand(0);
          else break;
        }
        auto view = projection.getDefiningOp<memref::SubViewOp>();
        if (external.getAccess() != 1 || !view || !headViews.count(view) ||
            view.getSource() != origin || !view.getSourceType().getLayout().isIdentity() ||
            !lifted.contains(memory)) { valid = false; continue; }
        auto [position, inserted] = outputRows.try_emplace(origin, view);
        if (!inserted && position->second != view) valid = false;
      } else if (write) {
        auto allocation = origin.getDefiningOp<memref::AllocOp>();
        if (!allocation || !parallel->isAncestor(allocation)) valid = false;
      }
    }
    return valid;
  }
};

// The additional axis represents independent original loop iterations. Static
// unit axes retain coordinate zero and may then be removed from their storage.
// In particular, this never interprets an arbitrary affine constant as a head.
struct GroupRewriter {
  RegionGroup &group;
  IRMapping mapping;
  Value extent;

  SmallVector<int64_t> shape(MemRefType type) {
    SmallVector<int64_t> result{ShapedType::kDynamic};
    for (int64_t size : type.getShape())
      if (size != 1) result.push_back(size);
    return result;
  }

  MemRefType type(MemRefType original) {
    auto sizes = shape(original);
    MemRefLayoutAttrInterface layout;
    if (!original.getLayout().isIdentity())
      layout = StridedLayoutAttr::get(original.getContext(), ShapedType::kDynamic,
          SmallVector<int64_t>(sizes.size(), ShapedType::kDynamic));
    return MemRefType::get(sizes, original.getElementType(), layout, original.getMemorySpace());
  }

  SmallVector<OpFoldResult> remap(ArrayRef<OpFoldResult> values) {
    SmallVector<OpFoldResult> result;
    for (OpFoldResult value : values)
      result.push_back(isa<Value>(value) ? OpFoldResult(mapping.lookupOrDefault(cast<Value>(value))) : value);
    return result;
  }

  void cloneBlock(OpBuilder &b, Block &source, Block &destination,
                  ArrayRef<AffineExpr> loops = {}) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&destination);
    for (Operation &operation : source) clone(b, &operation, loops);
  }

  void clone(OpBuilder &b, Operation *operation, ArrayRef<AffineExpr> loops = {}) {
    Location loc = operation->getLoc();
    if (unitReshapeAxes(operation) && group.lifted.contains(operation->getResult(0))) {
      mapping.map(operation->getResult(0), mapping.lookupOrDefault(operation->getOperand(0)));
      return;
    }
    if (auto dimension = dyn_cast<memref::DimOp>(operation)) {
      if (group.lifted.contains(dimension.getSource())) {
        int64_t axis = *dimension.getConstantIndex();
        auto source = cast<MemRefType>(dimension.getSource().getType());
        Value value;
        if (source.getDimSize(axis) == 1) value = index(b, loc, 1);
        else {
          int64_t translated = 1;
          for (int64_t i = 0; i < axis; ++i) translated += source.getDimSize(i) != 1;
          value = b.create<memref::DimOp>(loc, mapping.lookupOrDefault(dimension.getSource()), translated);
        }
        mapping.map(dimension.getResult(), value);
        return;
      }
    }
    if (auto view = dyn_cast<memref::SubViewOp>(operation)) {
      if (group.lifted.contains(view.getResult())) {
        auto offsets = remap(view.getMixedOffsets());
        auto sizes = remap(view.getMixedSizes());
        auto strides = remap(view.getMixedStrides());
        if (auto head = group.headViews.find(view); head != group.headViews.end()) {
          sizes[head->second] = extent;
        } else {
          auto oldOffsets = offsets, oldSizes = sizes, oldStrides = strides;
          offsets = {b.getIndexAttr(0)}; sizes = {extent}; strides = {b.getIndexAttr(1)};
          for (auto [axis, size] : llvm::enumerate(view.getSourceType().getShape())) {
            if (size == 1) continue;
            offsets.push_back(oldOffsets[axis]); sizes.push_back(oldSizes[axis]); strides.push_back(oldStrides[axis]);
          }
        }
        Value source = mapping.lookupOrDefault(view.getSource());
        auto resultType = memref::SubViewOp::inferRankReducedResultType(shape(view.getType()),
            cast<MemRefType>(source.getType()), offsets, sizes, strides);
        Value value = b.create<memref::SubViewOp>(loc, cast<MemRefType>(resultType), source, offsets, sizes, strides);
        mapping.map(view.getResult(), value);
        return;
      }
    }
    if (auto copy = dyn_cast<memref::CopyOp>(operation)) {
      if (!group.lifted.contains(copy.getSource()) && group.lifted.contains(copy.getTarget())) {
        auto sourceType = copy.getSource().getType();
        SmallVector<AffineExpr> sourceIndices;
        unsigned next = 1;
        for (int64_t size : sourceType.getShape())
          sourceIndices.push_back(size == 1 ? b.getAffineConstantExpr(0) : b.getAffineDimExpr(next++));
        auto generic = b.create<linalg::GenericOp>(loc, TypeRange{},
            ValueRange{mapping.lookupOrDefault(copy.getSource())}, ValueRange{mapping.lookupOrDefault(copy.getTarget())},
            ArrayRef<AffineMap>{AffineMap::get(next, 0, sourceIndices, b.getContext()), b.getMultiDimIdentityMap(next)},
            SmallVector<utils::IteratorType>(next, utils::IteratorType::parallel),
            [&](OpBuilder &nested, Location loc, ValueRange arguments) {
              nested.create<linalg::YieldOp>(loc, arguments.front());
            });
        (void)generic;
        return;
      }
    }
    if (auto coordinate = dyn_cast<linalg::IndexOp>(operation)) {
      if (!loops.empty()) {
        AffineExpr expression = loops[coordinate.getDim()];
        Value value = isa<AffineConstantExpr>(expression)
            ? index(b, loc, cast<AffineConstantExpr>(expression).getValue())
            : Value(b.create<linalg::IndexOp>(loc, cast<AffineDimExpr>(expression).getPosition()));
        mapping.map(coordinate.getResult(), value);
        return;
      }
    }
    if (auto load = dyn_cast<memref::LoadOp>(operation)) {
      if (group.lifted.contains(load.getMemref())) {
        SmallVector<Value> indices{b.create<linalg::IndexOp>(loc, 0)};
        for (auto [axis, value] : llvm::enumerate(load.getIndices()))
          if (load.getMemRefType().getDimSize(axis) != 1) indices.push_back(mapping.lookupOrDefault(value));
        Value value = b.create<memref::LoadOp>(loc, mapping.lookupOrDefault(load.getMemref()), indices);
        mapping.map(load.getResult(), value);
        return;
      }
    }
    Operation *replacement = b.cloneWithoutRegions(*operation, mapping);
    for (auto [oldValue, newValue] : llvm::zip(operation->getResults(), replacement->getResults()))
      if (group.lifted.contains(oldValue)) newValue.setType(type(cast<MemRefType>(oldValue.getType())));
    if (auto allocation = dyn_cast<memref::AllocOp>(operation)) {
      if (group.lifted.contains(allocation.getResult())) {
        auto next = cast<memref::AllocOp>(replacement);
        SmallVector<Value> sizes{extent};
        llvm::append_range(sizes, next.getDynamicSizes());
        next.getDynamicSizesMutable().assign(sizes);
      }
    }
    SmallVector<AffineExpr> translatedLoops;
    if (auto generic = dyn_cast<linalg::GenericOp>(operation)) {
      if (group.batched(generic)) {
        SmallVector<utils::IteratorType> iterators{utils::IteratorType::parallel};
        auto ranges = generic.getStaticLoopRanges();
        for (auto [range, iterator] : llvm::zip(ranges, generic.getIteratorTypesArray())) {
          if (range == 1) translatedLoops.push_back(b.getAffineConstantExpr(0));
          else {
            translatedLoops.push_back(b.getAffineDimExpr(iterators.size()));
            iterators.push_back(iterator);
          }
        }
        SmallVector<AffineMap> maps;
        for (auto [operand, map] : llvm::zip(generic->getOperands(), generic.getIndexingMapsArray())) {
          bool lifted = group.lifted.contains(operand);
          SmallVector<AffineExpr> indices;
          if (lifted) indices.push_back(b.getAffineDimExpr(0));
          for (auto [axis, expression] : llvm::enumerate(map.getResults())) {
            if (lifted && cast<MemRefType>(operand.getType()).getDimSize(axis) == 1) continue;
            indices.push_back(expression.replaceDimsAndSymbols(translatedLoops, {}));
          }
          maps.push_back(AffineMap::get(iterators.size(), 0, indices, b.getContext()));
        }
        auto next = cast<linalg::GenericOp>(replacement);
        next.setIndexingMapsAttr(b.getAffineMapArrayAttr(maps));
        SmallVector<Attribute> attributes;
        for (auto iterator : iterators) attributes.push_back(linalg::IteratorTypeAttr::get(b.getContext(), iterator));
        next.setIteratorTypesAttr(b.getArrayAttr(attributes));
        loops = translatedLoops;
      } else loops = {};
    }
    for (auto [source, destination] : llvm::zip(operation->getRegions(), replacement->getRegions())) {
      if (source.empty()) continue;
      SmallVector<Type> arguments;
      if (operation == group.fold) {
        auto program = cast<RegionOpInterface>(operation);
        auto schema = program.getRegionSchema(source);
        assert(succeeded(schema) && "grouping requires a verified region schema");
        for (const auto &relation : *schema)
          arguments.push_back(relation.kind == RegionArgumentKind::Sources
              ? relation.argument.getType()
              : mapping.lookupOrDefault(relation.prototype).getType());
      } else llvm::append_range(arguments, source.front().getArgumentTypes());
      auto *body = new Block;
      destination.push_back(body);
      for (auto [argument, type] : llvm::zip(source.front().getArguments(), arguments))
        mapping.map(argument, body->addArgument(type, argument.getLoc()));
      cloneBlock(b, source.front(), *body, loops);
    }
  }

  LogicalResult rewrite() {
    auto parallel = group.parallel;
    OpBuilder b(parallel);
    Location loc = parallel.getLoc();
    Value width = index(b, loc, group.width);
    SmallVector<Value> upper(parallel.getUpperBound());
    upper.back() = b.create<arith::CeilDivSIOp>(loc, upper.back(), width);
    auto replacement = b.create<scf::ParallelOp>(loc, parallel.getLowerBound(), upper, parallel.getStep());
    replacement->setDiscardableAttrs(parallel->getDiscardableAttrDictionary());
    b.setInsertionPointToStart(replacement.getBody());
    Value begin = multiply(b, loc, replacement.getInductionVars().back(), width);
    Value remaining = b.create<arith::SubIOp>(loc, parallel.getUpperBound().back(), begin);
    extent = b.create<arith::MinSIOp>(loc, width, remaining);
    for (auto [oldValue, newValue] : llvm::zip(parallel.getInductionVars(), replacement.getInductionVars()))
      mapping.map(oldValue, newValue);
    mapping.map(parallel.getInductionVars().back(), begin);
    for (Operation &operation : parallel.getBody()->without_terminator()) clone(b, &operation);
    if (failed(verify(replacement))) return failure();
    parallel.erase();
    return success();
  }
};

} // namespace

LogicalResult groupRegionComputations(func::FuncOp function, const Configuration &configuration) {
  SmallVector<scf::ParallelOp> loops;
  function.walk([&](scf::ParallelOp parallel) { loops.push_back(parallel); });
  for (auto parallel : llvm::reverse(loops)) {
    RegionGroup group;
    group.parallel = parallel;
    if (group.analyze(function, configuration.tileM) && failed(GroupRewriter{group}.rewrite())) return failure();
  }
  return success();
}

} // namespace intent::cpu
