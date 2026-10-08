#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "ContiguousAccesses.h"
#include "ProducerReuse.h"
#include "Intent/Analysis/IntegerRanges.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Intent/Dialect/CPU/Transforms/Structure/ProducerVersions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

namespace intent::cpu {
namespace {

struct BoundedCoordinate {
  AffineExpr expression;
  int64_t minimum, maximum;
};

std::optional<BoundedCoordinate> add(BoundedCoordinate lhs, BoundedCoordinate rhs,
                                     bool subtract = false) {
  int64_t minimum, maximum;
  if (subtract) {
    if (llvm::SubOverflow(lhs.minimum, rhs.maximum, minimum) ||
        llvm::SubOverflow(lhs.maximum, rhs.minimum, maximum)) return std::nullopt;
  } else if (llvm::AddOverflow(lhs.minimum, rhs.minimum, minimum) ||
             llvm::AddOverflow(lhs.maximum, rhs.maximum, maximum)) return std::nullopt;
  return BoundedCoordinate{subtract ? lhs.expression - rhs.expression : lhs.expression + rhs.expression,
                           minimum, maximum};
}

std::optional<BoundedCoordinate> multiply(BoundedCoordinate input, int64_t factor) {
  int64_t first, last;
  if (llvm::MulOverflow(input.minimum, factor, first) ||
      llvm::MulOverflow(input.maximum, factor, last)) return std::nullopt;
  return BoundedCoordinate{input.expression * factor, std::min(first, last), std::max(first, last)};
}

int64_t floorDivide(int64_t value, int64_t divisor) {
  // divisor is positive; signed_min / -1 is excluded by construction.
  return value / divisor - (value % divisor < 0);
}

std::optional<SmallVector<int64_t>> rowMajorStrides(MemRefType type) {
  if (!type.hasStaticShape() || !type.getLayout().isIdentity()) return std::nullopt;
  SmallVector<int64_t> strides(type.getRank());
  int64_t stride = 1;
  for (int64_t axis = type.getRank() - 1; axis >= 0; --axis) {
    if (type.getDimSize(axis) <= 0) return std::nullopt;
    strides[axis] = stride;
    if (llvm::MulOverflow(stride, type.getDimSize(axis), stride)) return std::nullopt;
  }
  return strides;
}

// AffineExpr denotes mathematical integers. Only expressions whose actual
// current-IR domain fits the DSL's signed 64-bit index arithmetic enter it.
// No candidate extent or two unknown strides establish a numerical bound.
class Coordinates {
public:
  Coordinates(MLIRContext *context, ArrayRef<int64_t> sizes,
              StorageAnalysis &storage)
      : context(context), sizes(sizes.begin(), sizes.end()),
        storage(storage),
        ranges(IntegerRangePolicy{
            {}, [this](Value value, IntegerRangeAnalysis &)
                    -> std::optional<ConstantIntRanges> {
              auto bound = memberAxes.find(value);
              if (bound == memberAxes.end()) return std::nullopt;
              return ConstantIntRanges::fromSigned(
                  APInt(64, 0), APInt(64, this->sizes[bound->second] - 1, true));
            }}) {}

  void bind(Value value, unsigned axis) {
    memberAxes[value] = axis;
  }

  void bind(linalg::GenericOp operation) {
    SmallVector<BoundedCoordinate> coordinates;
    for (auto [axis, size] : llvm::enumerate(sizes))
      coordinates.push_back({getAffineDimExpr(axis, context), 0, size - 1});
    iterationCoordinates[operation] = std::move(coordinates);
  }

  std::optional<BoundedCoordinate> get(Value value) {
    if (!value.getType().isIndex() && !value.getType().isSignlessInteger(64)) return std::nullopt;
    if (!active.insert(value).second) return std::nullopt;
    auto leave = llvm::make_scope_exit([&] { active.erase(value); });
    // The same producer SSA may be queried at different consumer coordinates.
    // Only immutable external symbols are cached across those substitutions.
    return compute(value);
  }

  ArrayRef<Value> getSymbols() const { return symbols; }
  bool resolvedRead(Operation *operation) const {
    return resolvedReads.contains(operation);
  }

  std::optional<AffineExpr> displacement(Value memory, ValueRange indices,
                                        ArrayRef<int64_t> destinationStrides) {
    auto type = cast<MemRefType>(memory.getType());
    SmallVector<int64_t> strides;
    int64_t offset;
    if (failed(type.getStridesAndOffset(strides, offset)) || ShapedType::isDynamic(offset) ||
        llvm::any_of(strides, ShapedType::isDynamic)) return std::nullopt;
    BoundedCoordinate address{getAffineConstantExpr(0, context), 0, 0};
    for (auto [index, stride] : llvm::zip_equal(indices, strides)) {
      auto coordinate = get(index);
      auto term = coordinate ? multiply(*coordinate, stride) : std::nullopt;
      auto sum = term ? add(address, *term) : std::nullopt;
      if (!sum) return std::nullopt;
      address = *sum;
    }
    if (!add(address, {getAffineConstantExpr(offset, context), offset, offset})) return std::nullopt;
    AffineExpr linear = getAffineConstantExpr(0, context);
    for (auto [axis, stride] : llvm::enumerate(destinationStrides))
      linear = linear + getAffineDimExpr(axis, context) * stride;
    AffineExpr difference = simplifyAffineExpr(address.expression - linear, sizes.size(), symbols.size());
    bool memberDependent = false;
    difference.walk([&](AffineExpr expression) { memberDependent |= isa<AffineDimExpr>(expression); });
    if (memberDependent) return std::nullopt;
    // Simplification can reassociate arithmetic. Validate the operations that
    // will actually be materialized, including the source descriptor offset.
    auto bound = materializedBounds(difference);
    if (!bound || !add(*bound, {getAffineConstantExpr(offset, context), offset, offset})) return std::nullopt;
    return difference;
  }

private:
  std::optional<BoundedCoordinate> withBounds(AffineExpr expression, Value value) {
    auto bounds = ranges.range(value);
    if (!bounds || !bounds->smin().isSignedIntN(64) ||
        !bounds->smax().isSignedIntN(64)) return std::nullopt;
    return BoundedCoordinate{expression, bounds->smin().getSExtValue(),
                             bounds->smax().getSExtValue()};
  }

  std::optional<BoundedCoordinate> materializedBounds(AffineExpr expression) {
    if (auto constant = dyn_cast<AffineConstantExpr>(expression))
      return BoundedCoordinate{expression, constant.getValue(), constant.getValue()};
    if (auto symbol = dyn_cast<AffineSymbolExpr>(expression))
      return symbolBounds[symbol.getPosition()];
    auto binary = dyn_cast<AffineBinaryOpExpr>(expression);
    if (!binary) return std::nullopt;
    auto lhs = materializedBounds(binary.getLHS()), rhs = materializedBounds(binary.getRHS());
    if (!lhs || !rhs) return std::nullopt;
    if (expression.getKind() == AffineExprKind::Add) return add(*lhs, *rhs);
    if (expression.getKind() == AffineExprKind::Mul) {
      if (auto factor = dyn_cast<AffineConstantExpr>(binary.getRHS())) return multiply(*lhs, factor.getValue());
      if (auto factor = dyn_cast<AffineConstantExpr>(binary.getLHS())) return multiply(*rhs, factor.getValue());
      return std::nullopt;
    }
    auto divisor = dyn_cast<AffineConstantExpr>(binary.getRHS());
    if (!divisor || divisor.getValue() <= 0) return std::nullopt;
    BoundedCoordinate quotient{expression, floorDivide(lhs->minimum, divisor.getValue()),
                                floorDivide(lhs->maximum, divisor.getValue())};
    if (expression.getKind() == AffineExprKind::FloorDiv) return quotient;
    if (expression.getKind() != AffineExprKind::Mod) return std::nullopt;
    auto product = multiply(quotient, divisor.getValue());
    if (!product || !add(*lhs, *product, true)) return std::nullopt;
    return BoundedCoordinate{expression, 0, divisor.getValue() - 1};
  }

  std::optional<BoundedCoordinate> project(
      AffineExpr expression, ArrayRef<BoundedCoordinate> coordinates) {
    if (auto dimension = dyn_cast<AffineDimExpr>(expression))
      return coordinates[dimension.getPosition()];
    if (auto constant = dyn_cast<AffineConstantExpr>(expression))
      return BoundedCoordinate{expression, constant.getValue(), constant.getValue()};
    auto binary = dyn_cast<AffineBinaryOpExpr>(expression);
    if (!binary) return std::nullopt;
    auto lhs = project(binary.getLHS(), coordinates);
    auto rhs = project(binary.getRHS(), coordinates);
    if (!lhs || !rhs) return std::nullopt;
    if (expression.getKind() == AffineExprKind::Add) return add(*lhs, *rhs);
    if (expression.getKind() == AffineExprKind::Mul) {
      if (lhs->minimum == lhs->maximum) return multiply(*rhs, lhs->minimum);
      if (rhs->minimum == rhs->maximum) return multiply(*lhs, rhs->minimum);
    }
    return std::nullopt;
  }

  bool pointwiseInput(Value memory, Operation *point) {
    auto generic = dyn_cast<linalg::GenericOp>(point);
    if (!generic || generic.getNumResults() || generic.getNumReductionLoops())
      return false;
    auto maps = generic.getIndexingMapsArray();
    AffineMap outputMap;
    for (auto [number, output] : llvm::enumerate(generic.getOutputs())) {
      if (output != memory) continue;
      if (outputMap) return false;
      outputMap = maps[generic.getInputs().size() + number];
    }
    if (!outputMap || failed(fullOutputProjection(memory, outputMap))) return false;
    bool reads = false;
    for (auto [number, input] : llvm::enumerate(generic.getInputs())) {
      if (input != memory) continue;
      if (maps[number] != outputMap) return false;
      reads = true;
    }
    // An in-place element formal observes its old element before that same
    // iteration writes it. Explicit reads or a different map need a separate
    // cross-iteration dependence proof and cannot use the preceding definition.
    return reads && llvm::all_of(generic.getRegion().front().without_terminator(),
        [](Operation &operation) {
          return !operation.getNumRegions() && isMemoryEffectFree(&operation);
        });
  }

  std::optional<BoundedCoordinate> read(
      Value memory, ArrayRef<BoundedCoordinate> coordinates, Operation *point) {
    auto allocation = memory.getDefiningOp<memref::AllocOp>();
    if (!allocation) return std::nullopt;
    auto lifetime = storage.lifetime(allocation);
    if (!lifetime || !lifetime->aliases.complete ||
        lifetime->aliases.values.size() != 1 || !lifetime->contains(point))
      return std::nullopt;
    if (!storage.preserves(point, memory) && !pointwiseInput(memory, point))
      return std::nullopt;
    auto current = findCurrentBufferWrite(memory, point, storage);
    if (!current) return std::nullopt;
    Operation *writer = current->operation;
    if (auto fill = dyn_cast<linalg::FillOp>(writer)) {
      if (fill.getNumResults() || fill.getOutputs().size() != 1 ||
          fill.getOutputs().front() != memory) return std::nullopt;
      return get(fill.getInputs().front());
    }
    auto producer = dyn_cast<linalg::GenericOp>(writer);
    if (!producer || producer.getNumResults() || producer.getNumReductionLoops() ||
        llvm::count(producer.getOutputs(), memory) != 1)
      return std::nullopt;
    auto output = llvm::find(producer.getOutputs(), memory);
    if (output == producer.getOutputs().end()) return std::nullopt;
    unsigned number = std::distance(producer.getOutputs().begin(), output);
    auto maps = producer.getIndexingMapsArray();
    auto inverse = fullOutputProjection(memory, maps[producer.getInputs().size() + number]);
    if (failed(inverse)) return std::nullopt;
    Block &body = producer.getRegion().front();
    if (!body.getArgument(producer.getInputs().size() + number).use_empty())
      return std::nullopt;
    auto payload = analyzeProducerResult(producer, storage, number);
    if (failed(payload)) return std::nullopt;
    // This is an observation of the stored result, not motion of the producer's
    // reads. Each input is interpreted at its original producer, before later
    // overwrites; only the resulting pure coordinate expression is retained.
    SmallVector<BoundedCoordinate> members;
    for (AffineExpr expression : inverse->getResults()) {
      auto member = project(expression, coordinates);
      if (!member) return std::nullopt;
      members.push_back(*member);
    }
    if (iterationCoordinates.contains(producer)) return std::nullopt;
    iterationCoordinates[producer] = std::move(members);
    auto leave = llvm::make_scope_exit([&] { iterationCoordinates.erase(producer); });
    return get(body.getTerminator()->getOperand(number));
  }

  std::optional<BoundedCoordinate> compute(Value value) {
    if (auto member = memberAxes.find(value); member != memberAxes.end())
      return withBounds(getAffineDimExpr(member->second, context), value);
    if (auto constant = getConstantIntValue(value))
      return BoundedCoordinate{getAffineConstantExpr(*constant, context), *constant, *constant};
    if (auto cast = value.getDefiningOp<arith::IndexCastOp>()) {
      auto input = get(cast.getIn());
      return input;
    }
    if (auto index = value.getDefiningOp<linalg::IndexOp>()) {
      auto found = iterationCoordinates.find(index->getParentOp());
      if (found != iterationCoordinates.end()) return found->second[index.getDim()];
      return std::nullopt;
    }
    if (auto load = value.getDefiningOp<memref::LoadOp>()) {
      SmallVector<BoundedCoordinate> coordinates;
      for (Value index : load.getIndices()) {
        auto coordinate = get(index);
        if (!coordinate) return std::nullopt;
        coordinates.push_back(*coordinate);
      }
      auto result = read(load.getMemref(), coordinates, load);
      if (result) resolvedReads.insert(load);
      return result;
    }
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      if (auto generic = dyn_cast<linalg::GenericOp>(argument.getOwner()->getParentOp())) {
        auto found = iterationCoordinates.find(generic);
        unsigned number = argument.getArgNumber();
        if (found == iterationCoordinates.end() || number >= generic.getInputs().size())
          return std::nullopt;
        Value input = generic.getInputs()[number];
        if (!isa<MemRefType>(input.getType())) return get(input);
        AffineMap map = generic.getIndexingMapsArray()[number];
        if (map.getNumSymbols()) return std::nullopt;
        SmallVector<BoundedCoordinate> coordinates;
        for (AffineExpr expression : map.getResults()) {
          auto coordinate = project(expression, found->second);
          if (!coordinate) return std::nullopt;
          coordinates.push_back(*coordinate);
        }
        return read(input, coordinates, generic);
      }
      bool induction = false;
      if (auto loop = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
          loop && argument == loop.getInductionVar()) {
        induction = true;
      } else if (auto parallel = dyn_cast<scf::ParallelOp>(argument.getOwner()->getParentOp())) {
        induction = llvm::is_contained(parallel.getInductionVars(), value);
      }
      if (!induction) return std::nullopt;
      auto found = llvm::find(symbols, value);
      if (found != symbols.end()) return symbolBounds[std::distance(symbols.begin(), found)];
      auto result = withBounds(getAffineSymbolExpr(symbols.size(), context), value);
      if (!result) return std::nullopt;
      symbols.push_back(value);
      symbolBounds.push_back(*result);
      return result;
    }
    Operation *operation = value.getDefiningOp();
    if (!operation || operation->getNumOperands() != 2) return std::nullopt;
    Value lhs = operation->getOperand(0), rhs = operation->getOperand(1);
    if (isa<arith::AddIOp, arith::SubIOp>(operation)) {
      auto left = get(lhs), right = get(rhs);
      auto checked = left && right
          ? add(*left, *right, isa<arith::SubIOp>(operation)) : std::nullopt;
      return checked;
    }
    if (isa<arith::MulIOp>(operation)) {
      auto left = get(lhs), right = get(rhs);
      if (!left || !right) return std::nullopt;
      if (left->minimum == left->maximum) return multiply(*right, left->minimum);
      if (right->minimum == right->maximum) return multiply(*left, right->minimum);
      return std::nullopt;
    }
    if (!isa<arith::FloorDivSIOp, arith::DivSIOp, arith::RemSIOp>(operation)) return std::nullopt;
    auto divisor = get(rhs), input = get(lhs);
    if (!divisor || divisor->minimum != divisor->maximum ||
        divisor->minimum <= 0 || !input) return std::nullopt;
    int64_t factor = divisor->minimum;
    if (!isa<arith::FloorDivSIOp>(operation) && input->minimum < 0) return std::nullopt;
    if (isa<arith::RemSIOp>(operation))
      return BoundedCoordinate{input->expression % factor, 0, factor - 1};
    return BoundedCoordinate{input->expression.floorDiv(factor),
                             floorDivide(input->minimum, factor),
                             floorDivide(input->maximum, factor)};
  }

  MLIRContext *context;
  SmallVector<int64_t> sizes;
  SmallVector<Value> symbols;
  SmallVector<BoundedCoordinate> symbolBounds;
  llvm::DenseMap<Value, unsigned> memberAxes;
  llvm::DenseSet<Value> active;
  llvm::DenseSet<Operation *> resolvedReads;
  llvm::DenseMap<Operation *, SmallVector<BoundedCoordinate>> iterationCoordinates;
  StorageAnalysis &storage;
  IntegerRangeAnalysis ranges;
};

Value materialize(AffineExpr expression, ArrayRef<Value> symbols, OpBuilder &builder, Location loc) {
  if (auto constant = dyn_cast<AffineConstantExpr>(expression))
    return builder.create<arith::ConstantIndexOp>(loc, constant.getValue());
  if (auto symbol = dyn_cast<AffineSymbolExpr>(expression)) return symbols[symbol.getPosition()];
  auto binary = cast<AffineBinaryOpExpr>(expression);
  Value lhs = materialize(binary.getLHS(), symbols, builder, loc);
  Value rhs = materialize(binary.getRHS(), symbols, builder, loc);
  switch (expression.getKind()) {
  case AffineExprKind::Add: return builder.createOrFold<arith::AddIOp>(loc, lhs, rhs);
  case AffineExprKind::Mul: return builder.createOrFold<arith::MulIOp>(loc, lhs, rhs);
  case AffineExprKind::FloorDiv: return builder.createOrFold<arith::FloorDivSIOp>(loc, lhs, rhs);
  case AffineExprKind::Mod: {
    Value quotient = builder.createOrFold<arith::FloorDivSIOp>(loc, lhs, rhs);
    Value product = builder.createOrFold<arith::MulIOp>(loc, quotient, rhs);
    return builder.createOrFold<arith::SubIOp>(loc, lhs, product);
  }
  default: llvm_unreachable("coordinate composition emitted an unsupported affine expression");
  }
}

Value contiguousView(Value source, MemRefType shape, ArrayRef<int64_t> strides,
                     AffineExpr displacement, ArrayRef<Value> symbols, Operation *before) {
  OpBuilder builder(before);
  Location loc = before->getLoc();
  auto metadata = builder.create<memref::ExtractStridedMetadataOp>(loc, source);
  Value offset = builder.createOrFold<arith::AddIOp>(loc, metadata.getOffset(),
      materialize(displacement, symbols, builder, loc));
  SmallVector<OpFoldResult> sizes, steps;
  for (int64_t size : shape.getShape()) sizes.push_back(builder.getIndexAttr(size));
  for (int64_t stride : strides) steps.push_back(builder.getIndexAttr(stride));
  int64_t knownOffset = getConstantIntValue(offset).value_or(ShapedType::kDynamic);
  auto type = MemRefType::get(shape.getShape(), shape.getElementType(),
      StridedLayoutAttr::get(builder.getContext(), knownOffset, strides), shape.getMemorySpace());
  return builder.create<memref::ReinterpretCastOp>(loc, type, metadata.getBaseBuffer(),
      OpFoldResult(offset), sizes, steps);
}

bool foldRead(linalg::GenericOp producer, func::FuncOp function) {
  if (producer.getNumResults() || producer.getOutputs().size() != 1 ||
      producer.getNumReductionLoops() || !producer.getIndexingMapsArray().back().isIdentity()) return false;
  auto allocation = producer.getOutputs()[0].getDefiningOp<memref::AllocOp>();
  if (!allocation || allocation->getBlock() != producer->getBlock()) return false;
  auto strides = rowMajorStrides(allocation.getType());
  StorageAnalysis storage(function);
  auto lifetime = storage.lifetime(allocation);
  if (!strides || !lifetime || !lifetime->aliases.complete || lifetime->aliases.values.size() != 1) return false;
  Block &body = producer.getRegion().front();
  if (!body.getArguments().back().use_empty()) return false;
  auto read = body.getTerminator()->getOperand(0).getDefiningOp<memref::LoadOp>();
  if (!read || read->getBlock() != &body) return false;
  if (failed(analyzeLinalgProducer(producer, storage))) return false;
  Coordinates coordinates(function.getContext(), allocation.getType().getShape(), storage);
  coordinates.bind(producer);
  auto sourceType = read.getMemRefType();
  if (sourceType.getElementType() != allocation.getType().getElementType() ||
      sourceType.getMemorySpace() != allocation.getType().getMemorySpace()) return false;
  auto displacement = coordinates.displacement(read.getMemref(), read.getIndices(), *strides);
  if (!displacement) return false;
  for (Operation &operation : body.without_terminator())
    if (&operation != read && !coordinates.resolvedRead(&operation) &&
        (!isMemoryEffectFree(&operation) || operation.getNumRegions())) return false;
  DominanceInfo dominance(function);
  if (!dominance.dominates(read.getMemref(), allocation) ||
      llvm::any_of(coordinates.getSymbols(), [&](Value symbol) { return !dominance.dominates(symbol, allocation); })) return false;
  if (!storage.disjoint(read.getMemref(), allocation)) return false;
  for (Operation *user : lifetime->aliases.users) {
    if (user == producer || user == lifetime->end || isa<memref::DimOp>(user)) continue;
    // An effect-free operation can still observe the backing pointer or its
    // alignment. Only element-value consumers may replace a snapshot by a view.
    bool elements = isa<memref::LoadOp, memref::CopyOp, linalg::LinalgOp, ReduceOp>(user);
    if (auto scan = dyn_cast<ScanOp>(user))
      elements = !scan.isDestinationPassing() &&
          !llvm::is_contained(scan.getCaptures(), allocation.getResult()) &&
          !llvm::is_contained(scan.getInitials(), allocation.getResult());
    if (!elements) return false;
    if (!storage.preserves(user, allocation) ||
        !storage.readStable(read.getMemref(), producer, user)) return false;
  }
  Value view = contiguousView(read.getMemref(), allocation.getType(), *strides,
                              *displacement, coordinates.getSymbols(), allocation);
  producer.erase();
  lifetime->end.erase();
  allocation.replaceAllUsesWith(view);
  allocation.erase();
  return true;
}

bool composeWrite(memref::StoreOp store, func::FuncOp function) {
  auto read = store.getValue().getDefiningOp<memref::LoadOp>();
  auto allocation = read ? read.getMemref().getDefiningOp<memref::AllocOp>() : memref::AllocOp();
  if (!allocation || !read.getResult().hasOneUse() || read->getBlock() != store->getBlock()) return false;
  auto strides = rowMajorStrides(allocation.getType());
  StorageAnalysis storage(function);
  auto lifetime = storage.lifetime(allocation);
  if (!strides || !lifetime || !lifetime->aliases.complete || !allocation.getType().getRank()) return false;
  SmallVector<scf::ForOp> loops;
  Operation *root = store;
  while (root->getBlock() != allocation->getBlock()) {
    auto loop = dyn_cast<scf::ForOp>(root->getParentOp());
    if (!loop || loop.getNumResults() || !matchPattern(loop.getLowerBound(), m_Zero()) ||
        !matchPattern(loop.getStep(), m_One())) return false;
    loops.push_back(loop);
    root = loop;
  }
  std::reverse(loops.begin(), loops.end());
  if (loops.size() != strides->size()) return false;
  Coordinates coordinates(function.getContext(), allocation.getType().getShape(), storage);
  for (auto [axis, loop] : llvm::enumerate(loops)) {
    if (getConstantIntValue(loop.getUpperBound()) != allocation.getType().getDimSize(axis) ||
        read.getIndices()[axis] != loop.getInductionVar()) return false;
    coordinates.bind(loop.getInductionVar(), axis);
  }
  auto displacement = coordinates.displacement(store.getMemref(), store.getIndices(), *strides);
  if (!displacement) return false;
  bool valid = true;
  root->walk([&](Operation *operation) {
    if (operation == read || operation == store || coordinates.resolvedRead(operation) ||
        isa<scf::ForOp, scf::YieldOp>(operation)) return;
    if (operation->getNumRegions() || !isMemoryEffectFree(operation)) valid = false;
  });
  auto destination = store.getMemRefType();
  if (!valid || destination.getElementType() != allocation.getType().getElementType() ||
      destination.getMemorySpace() != allocation.getType().getMemorySpace()) return false;
  // memref.copy itself must preserve the scalar traversal before the separate
  // output-forwarding transform considers it. Overlapping ranges stay as loops.
  if (!storage.disjoint(allocation, store.getMemref())) return false;
  DominanceInfo dominance(function);
  if (!dominance.dominates(store.getMemref(), allocation) ||
      llvm::any_of(coordinates.getSymbols(), [&](Value symbol) { return !dominance.dominates(symbol, allocation); })) return false;
  Value view = contiguousView(store.getMemref(), allocation.getType(), *strides,
                              *displacement, coordinates.getSymbols(), allocation);
  OpBuilder(root).create<memref::CopyOp>(root->getLoc(), allocation.getResult(), view);
  root->erase();
  return true;
}

} // namespace

void foldContiguousAccesses(func::FuncOp function) {
  bool changed;
  do {
    changed = false;
    SmallVector<linalg::GenericOp> readers;
    function.walk([&](linalg::GenericOp operation) { readers.push_back(operation); });
    for (auto reader : readers)
      if (foldRead(reader, function)) { changed = true; break; }
  } while (changed);
  do {
    changed = false;
    SmallVector<memref::StoreOp> writers;
    function.walk([&](memref::StoreOp operation) { writers.push_back(operation); });
    for (auto writer : writers)
      if (composeWrite(writer, function)) { changed = true; break; }
  } while (changed);
}

} // namespace intent::cpu
