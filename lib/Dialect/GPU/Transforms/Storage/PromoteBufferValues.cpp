#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;

namespace intent::gpu {
namespace {

StoreOp fullInitialization(BufferOp buffer, func::FuncOp kernel) {
  auto storage = buffer.getResult().getType();
  PhysicalProgramAnalysis analysis(kernel);
  DominanceInfo dominance(kernel);
  for (Operation *user : buffer.getResult().getUsers()) {
    auto store = dyn_cast<StoreOp>(user);
    if (!store || store->getBlock() != buffer->getBlock() ||
        store.getCoordinates().size() != storage.getShape().size())
      continue;
    auto value = dyn_cast<FragmentType>(store.getValue().getType());
    if (!value || value.getShape() != storage.getShape() ||
        value.getOwner() != storage.getOwner())
      continue;
    SmallVector<std::pair<MakeRangeOp, Value>> tails;
    bool complete = true;
    for (auto [axis, coordinate] : llvm::enumerate(store.getCoordinates())) {
      auto range = coordinate.getDefiningOp<MakeRangeOp>();
      auto projection = range ? queryAxisProjection(range.getType(), value)
                              : BroadcastProjection{};
      if (store.getSourceAxes()[axis] != static_cast<int64_t>(axis) ||
          !range || !projection.isExact() ||
          projection.targetToSource[axis] != std::optional<unsigned>(0) ||
          !matchPattern(range.getStart(), m_Zero()) ||
          !matchPattern(range.getLogicalStart(), m_Zero()) ||
          !isUnitStepRange(range) ||
          queryLaunchExpression(range.getExtent()) != storage.getShape()[axis] ||
          queryLaunchExpression(range.getLogicalStop()) != storage.getShape()[axis]) {
        complete = false;
        break;
      }
      tails.emplace_back(range, range.getLogicalStop());
    }
    if (!complete || !analysis.isTailPredicate(store.getValid(), tails))
      continue;
    if (llvm::all_of(buffer.getResult().getUsers(), [&](Operation *other) {
          return other == store || dominance.properlyDominates(store, other);
        }))
      return store;
  }
  return {};
}

bool collectStateRegions(BufferOp buffer, StoreOp initial,
                         llvm::DenseSet<Operation *> &writtenRegions) {
  auto storage = buffer.getResult().getType();
  auto orderedAxes = [&](ValueRange coordinates, ArrayRef<int64_t> axes) {
    return coordinates.size() == storage.getShape().size() &&
           llvm::all_of(llvm::enumerate(axes), [](auto item) {
             return item.value() == static_cast<int64_t>(item.index());
           });
  };
  for (Operation *user : buffer.getResult().getUsers()) {
    if (user == initial)
      continue;
    bool writes = false;
    if (auto load = dyn_cast<LoadOp>(user)) {
      if (load.getResource() != buffer.getResult() ||
          !orderedAxes(load.getCoordinates(), load.getSourceAxes()))
        return false;
    } else if (auto store = dyn_cast<StoreOp>(user)) {
      if (store.getResource() != buffer.getResult() ||
          !orderedAxes(store.getCoordinates(), store.getSourceAxes()))
        return false;
      auto value = dyn_cast<FragmentType>(store.getValue().getType());
      unsigned valueAxis = 0;
      for (Value coordinate : store.getCoordinates()) {
        if (coordinate.getType().isIndex())
          continue;
        auto range = coordinate.getDefiningOp<MakeRangeOp>();
        if (!range || !value || valueAxis >= value.getShape().size() ||
            !isUnitStepRange(range) ||
            queryLaunchExpression(range.getExtent()) != value.getShape()[valueAxis])
          return false;
        auto rangeAxis = cast<AxisMapAttr>(range.getType().getAxisMaps()[0]);
        auto payloadAxis = cast<AxisMapAttr>(value.getAxisMaps()[valueAxis++]);
        if (!(sourceAxisIdentity(rangeAxis) == sourceAxisIdentity(payloadAxis)) ||
            rangeAxis.getDimensionId() != payloadAxis.getDimensionId())
          return false;
      }
      if (value ? valueAxis != value.getShape().size()
                : store.getValue().getType() != storage.getElementType())
        return false;
      writes = true;
    } else {
      return false;
    }
    for (Operation *parent = user->getParentOp(); parent != buffer->getParentOp();
         parent = parent->getParentOp()) {
      if (!isa<scf::ForOp, scf::IfOp>(parent) ||
          (writes && parent->hasAttr(reductionSourcesAttr)))
        return false;
      if (writes)
        writtenRegions.insert(parent);
    }
  }
  return true;
}

bool hasNestedScalarIndexedReads(BufferOp buffer) {
  bool nestedScalarRead = false;
  for (Operation *user : buffer.getResult().getUsers()) {
    auto loop = user->getParentOfType<scf::ForOp>();
    if (!loop)
      continue;
    if (auto load = dyn_cast<LoadOp>(user)) {
      if (isa<FragmentType>(load.getType()))
        return false;
      if (loop->getParentOfType<scf::ForOp>() &&
          llvm::any_of(load.getCoordinates(), [](Value coordinate) {
            return !coordinate.getDefiningOp<arith::ConstantOp>();
          }))
        nestedScalarRead = true;
    } else if (auto store = dyn_cast<StoreOp>(user)) {
      if (isa<FragmentType>(store.getValue().getType()))
        return false;
    }
  }
  return nestedScalarRead;
}

bool requiresFragmentGather(BufferOp buffer) {
  auto isAligned = [&](auto &&self, Value start, Value extent) -> bool {
    if (matchPattern(start, m_Zero()) || samePhysicalScalarExpression(start, extent))
      return true;
    auto value = start.getDefiningOp<BinaryOp>();
    if (!value)
      return false;
    if (value.getOperatorKind() == BinaryOperator::Multiply)
      return self(self, value.getLhs(), extent) || self(self, value.getRhs(), extent);
    if (value.getOperatorKind() == BinaryOperator::Add ||
        value.getOperatorKind() == BinaryOperator::Subtract)
      return self(self, value.getLhs(), extent) && self(self, value.getRhs(), extent);
    return false;
  };
  for (Operation *user : buffer.getResult().getUsers()) {
    ValueRange coordinates;
    if (auto load = dyn_cast<LoadOp>(user))
      coordinates = load.getCoordinates();
    else if (auto store = dyn_cast<StoreOp>(user))
      coordinates = store.getCoordinates();
    else
      return true;
    for (Value coordinate : coordinates) {
      if (coordinate.getType().isIndex())
        continue;
      auto range = coordinate.getDefiningOp<MakeRangeOp>();
      if (!range || !isUnitStepRange(range) ||
          !isAligned(isAligned, range.getStart(), range.getExtent()))
        return true;
    }
  }
  return false;
}

class BufferStatePromotion {
public:
  BufferStatePromotion(BufferOp buffer, StoreOp initial,
                       const llvm::DenseSet<Operation *> &writtenRegions)
      : buffer(buffer), initial(initial), writtenRegions(writtenRegions),
        indices(initial.getCoordinates().begin(), initial.getCoordinates().end()),
        stateType(cast<FragmentType>(initial.getValue().getType())) {}

  void run() {
    Value state = initial.getValue();
    auto begin = std::next(initial->getIterator());
    rewrite(*initial->getBlock(), begin, state);
    initial.erase();
    buffer.erase();
  }

private:
  FragmentType withElement(Type element) {
    return FragmentType::get(stateType.getContext(), element, stateType.getShape(),
                            stateType.getAxisMaps(), stateType.getValidity(),
                            stateType.getOwner());
  }

  void rewrite(Block &block, Block::iterator begin, Value &state) {
    for (auto iterator = begin; iterator != block.end();) {
      Operation *operation = &*iterator++;
      if (auto load = dyn_cast<LoadOp>(operation);
          load && load.getResource() == buffer.getResult()) {
        OpBuilder builder(load);
        Value value = builder.create<GatherOp>(
            load.getLoc(), load.getResult().getType(), state,
            load.getCoordinates(), load.getValid(), load.getFill(),
            load.getSourceAxes());
        if (Attribute origin = load->getAttr(originAttr))
          value.getDefiningOp()->setAttr(originAttr, origin);
        load.getResult().replaceAllUsesWith(value);
        load.erase();
      } else if (auto store = dyn_cast<StoreOp>(operation);
                 store && store.getResource() == buffer.getResult()) {
        OpBuilder builder(store);
        Location location = store.getLoc();
        auto predicate = withElement(builder.getI1Type());
        Value valid;
        SmallVector<Value> offsets;
        SmallVector<int64_t> valueAxes;
        auto indexType = withElement(builder.getIndexType());
        for (auto [axis, coordinate] : llvm::enumerate(store.getCoordinates())) {
          auto coordinateType = cast<FragmentType>(indices[axis].getType());
          auto axisPredicate = FragmentType::get(
              coordinateType.getContext(), builder.getI1Type(),
              coordinateType.getShape(), coordinateType.getAxisMaps(),
              coordinateType.getValidity(), coordinateType.getOwner());
          Value selected;
          if (auto range = coordinate.getDefiningOp<MakeRangeOp>()) {
            Value start = builder.create<SplatOp>(location, coordinateType,
                                                  range.getStart());
            Value offset = builder.create<BinaryOp>(location, coordinateType,
                indices[axis], start, BinaryOperator::Subtract);
            Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
            Value zeroIndices = builder.create<SplatOp>(location, coordinateType, zero);
            Value extent = builder.create<SplatOp>(location, coordinateType,
                                                   range.getExtent());
            Value nonNegative = builder.create<CompareOp>(
                location, axisPredicate, offset, zeroIndices, ComparePredicate::Ge);
            Value inRange = builder.create<CompareOp>(
                location, axisPredicate, offset, extent, ComparePredicate::Lt);
            selected = builder.create<BinaryOp>(location, axisPredicate,
                nonNegative, inRange, BinaryOperator::LogicalAnd);
            offsets.push_back(builder.create<BroadcastOp>(location, indexType, offset));
            valueAxes.push_back(valueAxes.size());
          } else {
            Value index = builder.create<SplatOp>(location, coordinateType, coordinate);
            selected = builder.create<CompareOp>(location, axisPredicate,
                indices[axis], index, ComparePredicate::Eq);
          }
          selected = builder.create<BroadcastOp>(location, predicate, selected);
          valid = valid ? Value(builder.create<BinaryOp>(location, predicate,
                                  valid, selected, BinaryOperator::LogicalAnd))
                        : selected;
        }
        if (store.getValid()) {
          Value active;
          if (isa<FragmentType>(store.getValid().getType())) {
            Value scalarFalse = builder.create<arith::ConstantIntOp>(location, 0, 1);
            Value falseValues = builder.create<SplatOp>(location, predicate, scalarFalse);
            active = builder.create<GatherOp>(location, predicate, store.getValid(),
                                               offsets, valid, falseValues, valueAxes);
          } else {
            active = builder.create<SplatOp>(location, predicate, store.getValid());
          }
          valid = builder.create<BinaryOp>(location, predicate, valid, active,
                                           BinaryOperator::LogicalAnd);
        }
        Value value;
        if (isa<FragmentType>(store.getValue().getType())) {
          Value scalarZero = builder.create<arith::ConstantOp>(
              location, stateType.getElementType(),
              builder.getZeroAttr(stateType.getElementType()));
          Value zeroValues = builder.create<SplatOp>(location, stateType, scalarZero);
          value = builder.create<GatherOp>(location, stateType, store.getValue(),
                                            offsets, valid, zeroValues, valueAxes);
        } else {
          value = builder.create<SplatOp>(location, stateType, store.getValue());
        }
        state = builder.create<SelectOp>(location, stateType, valid, value, state);
        if (Attribute origin = store->getAttr(originAttr))
          state.getDefiningOp()->setAttr(originAttr, origin);
        store.erase();
      } else if (auto loop = dyn_cast<scf::ForOp>(operation)) {
        if (!writtenRegions.contains(loop)) {
          Value unchanged = state;
          rewrite(*loop.getBody(), loop.getBody()->begin(), unchanged);
          continue;
        }
        OpBuilder builder(loop);
        SmallVector<Value> initials(loop.getInitArgs());
        initials.push_back(state);
        auto replacement = builder.create<scf::ForOp>(
            loop.getLoc(), loop.getLowerBound(), loop.getUpperBound(),
            loop.getStep(), initials);
        replacement->setAttrs(loop->getAttrs());
        replacement->removeAttr(independentIterationAttr);
        for (auto [oldArgument, argument] :
             llvm::zip(loop.getBody()->getArguments(),
                       replacement.getBody()->getArguments()))
          oldArgument.replaceAllUsesWith(argument);
        auto *body = replacement.getBody();
        if (!body->empty())
          body->getTerminator()->erase();
        body->getOperations().splice(body->end(), loop.getBody()->getOperations());
        Value carried = replacement.getRegionIterArgs().back();
        rewrite(*body, body->begin(), carried);
        auto yield = body->getTerminator();
        yield->insertOperands(yield->getNumOperands(), carried);
        for (auto [oldResult, result] :
             llvm::zip(loop.getResults(), replacement.getResults()))
          oldResult.replaceAllUsesWith(result);
        state = replacement.getResults().back();
        loop.erase();
      } else if (auto branch = dyn_cast<scf::IfOp>(operation)) {
        if (!writtenRegions.contains(branch)) {
          for (Region &region : branch->getRegions())
            if (!region.empty()) {
              Value unchanged = state;
              rewrite(region.front(), region.front().begin(), unchanged);
            }
          continue;
        }
        OpBuilder builder(branch);
        SmallVector<Type> types(branch.getResultTypes());
        types.push_back(stateType);
        auto replacement = builder.create<scf::IfOp>(
            branch.getLoc(), types, branch.getCondition(), true);
        replacement->setAttrs(branch->getAttrs());
        for (auto [oldRegion, region] :
             llvm::zip(branch->getRegions(), replacement->getRegions())) {
          Block &body = region.front();
          if (!body.empty())
            body.getTerminator()->erase();
          if (!oldRegion.empty())
            body.getOperations().splice(body.end(),
                                        oldRegion.front().getOperations());
          else {
            builder.setInsertionPointToEnd(&body);
            builder.create<scf::YieldOp>(branch.getLoc());
          }
          Value carried = state;
          rewrite(body, body.begin(), carried);
          auto yield = body.getTerminator();
          yield->insertOperands(yield->getNumOperands(), carried);
        }
        for (auto [oldResult, result] :
             llvm::zip(branch.getResults(), replacement.getResults()))
          oldResult.replaceAllUsesWith(result);
        state = replacement.getResults().back();
        branch.erase();
      }
    }
  }

  BufferOp buffer;
  StoreOp initial;
  const llvm::DenseSet<Operation *> &writtenRegions;
  SmallVector<Value> indices;
  FragmentType stateType;
};

} // namespace

static LogicalResult promoteBufferValuesImpl(ModuleOp module) {
  auto kernel = getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  auto capabilities = (*kernel)->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  SmallVector<BufferOp> buffers;
  // Allocations belong to their execution group or lexical control scope.
  // Promotion starts after the dominating initialization in that same block.
  kernel->walk([&](BufferOp buffer) { buffers.push_back(buffer); });
  llvm::DenseSet<int64_t> removedEffects;
  bool changed = false;
  // Limit promoted words by the maximum CTA thread count. Native lowering still
  // owns the register layout and allocation of these fragments.
  int64_t remainingWords = capabilities.getMaxThreadsPerBlock();
  for (BufferOp buffer : buffers) {
    auto type = buffer.getResult().getType();
    if (type.getScope().getValue() != BufferScope::ProgramPrivate ||
        type.isInvocationWorkspace() || type.getShape().empty())
      continue;
    unsigned words = type.getElementType().isIndex()
                         ? 2
                         : std::max(1u,
                                    (type.getElementType().getIntOrFloatBitWidth() + 31) / 32);
    int64_t elements = 1;
    bool fits = true;
    for (Attribute axis : type.getShape()) {
      auto extent = constantPhysicalExpression(cast<PhysicalExprAttr>(axis));
      if (!extent || *extent <= 0 || *extent > remainingWords / words / elements) {
        fits = false;
        break;
      }
      elements *= *extent;
    }
    if (!fits)
      continue;
    StoreOp initial = fullInitialization(buffer, *kernel);
    llvm::DenseSet<Operation *> writtenRegions;
    if (initial && collectStateRegions(buffer, initial, writtenRegions)) {
      // Scalar selections and aligned Cartesian tiles need only native extract.
      // Other coordinates require indexed access to a resident fragment.
      if (!capabilities.getNativeFragmentGather() && requiresFragmentGather(buffer))
        continue;
      // A scalar recurrence repeatedly indexing its state has no vector reuse.
      // Promoting it would carry and dynamically extract a complete tile through
      // every inner iteration. Keep the existing buffer accesses in that case.
      if (elements > 1 && hasNestedScalarIndexedReads(buffer))
        continue;
      for (Operation *user : buffer.getResult().getUsers())
        if (isa<StoreOp>(user))
          if (auto origin = user->getAttrOfType<IntegerAttr>(originAttr))
            removedEffects.insert(origin.getInt());
      BufferStatePromotion(buffer, initial, writtenRegions).run();
      remainingWords -= elements * words;
      changed = true;
    }
  }
  if (!changed)
    return success();
  // These private writes now exist as SSA state transitions. Other observable
  // effects, including another physical slice of the same origin, stay required.
  (*kernel).walk([&](Operation *operation) {
    if (auto access = dyn_cast<AccessOpInterface>(operation);
        access && access.writesMemory())
      if (auto origin = operation->getAttrOfType<IntegerAttr>(originAttr))
        removedEffects.erase(origin.getInt());
  });
  SmallVector<Attribute> effects;
  for (Attribute origin : (*kernel)->getAttrOfType<ArrayAttr>(effectOriginsAttr))
    if (!removedEffects.contains(cast<IntegerAttr>(origin).getInt()))
      effects.push_back(origin);
  (*kernel)->setAttr(effectOriginsAttr,
                     ArrayAttr::get(module.getContext(), effects));
  return realizeAccessComposition(module);
}

LogicalResult promoteBufferValues(ModuleOp module) {
  if (failed(promoteBufferValuesImpl(module))) return failure();
  auto kernel = getPhysicalKernel(module);
  return failed(kernel) ? failure() : closeValueRelations(*kernel);
}

} // namespace intent::gpu
