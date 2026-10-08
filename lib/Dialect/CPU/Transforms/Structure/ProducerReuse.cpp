#include "ProducerReuse.h"
#include "Intent/Analysis/IntegerRanges.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Intent/Dialect/CPU/Analysis/ExtentRelations.h"
#include "Intent/Dialect/CPU/Analysis/ViewRelations.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallBitVector.h"

using namespace mlir;

namespace intent::cpu {
namespace {

bool boundValue(const ProducerReplay &payload, Value value) {
  return llvm::is_contained(payload.frontier, value) ||
         llvm::is_contained(payload.externalValues, value) ||
         matchPattern(value, m_Constant());
}

bool sameCoordinates(ArrayRef<Value> lhs, ArrayRef<Value> rhs) {
  return lhs.size() == rhs.size() && llvm::all_of(llvm::zip(lhs, rhs), [](auto pair) {
    auto [a, b] = pair;
    if (a == b) return true;
    auto first = getConstantIntValue(a), second = getConstantIntValue(b);
    return first && second && *first == *second;
  });
}

bool singleIteration(scf::ForOp loop, IntegerRangeAnalysis &ranges) {
  auto lower = ranges.range(loop.getLowerBound());
  auto upper = ranges.range(loop.getUpperBound());
  auto step = ranges.range(loop.getStep());
  if (!lower || !upper || !step || !step->smin().isStrictlyPositive()) return false;
  return (upper->smax().sext(65) - lower->smin().sext(65)).sle(step->smin().sext(65));
}

bool nonempty(scf::ForOp loop, IntegerRangeAnalysis &ranges) {
  auto lower = ranges.range(loop.getLowerBound());
  auto upper = ranges.range(loop.getUpperBound());
  return lower && upper && lower->smax().slt(upper->smin()) &&
         ranges.isPositive(loop.getStep());
}

bool nonRepeating(const ProducerReplayGroup &group, Block *scope,
                  IntegerRangeAnalysis &ranges) {
  Operation *current = group.anchor;
  while (current->getBlock() != scope) {
    Operation *parent = current->getParentOp();
    if (!parent) return false;
    if (auto loop = dyn_cast<scf::ForOp>(parent)) {
      auto found = llvm::find(group.coordinates, loop.getInductionVar());
      if (found == group.coordinates.end()) {
        if (!singleIteration(loop, ranges)) return false;
      } else if (group.vectorWidth > 1 && found + 1 == group.coordinates.end()) {
        auto step = ranges.range(loop.getStep());
        if (!step || step->smin().slt(APInt(64, group.vectorWidth))) return false;
      }
    } else if (auto generic = dyn_cast<linalg::GenericOp>(parent)) {
      auto extents = generic.getStaticLoopRanges();
      for (unsigned axis = 0; axis < extents.size(); ++axis) {
        if (extents[axis] == 0 || extents[axis] == 1) continue;
        bool present = llvm::any_of(group.coordinates, [&](Value coordinate) {
          auto index = coordinate.getDefiningOp<linalg::IndexOp>();
          return index && index->getParentOp() == parent && index.getDim() == axis;
        });
        if (!present) return false;
      }
    } else if (!isa<scf::IfOp>(parent)) {
      return false;
    }
    current = parent;
  }
  return true;
}

bool exclusiveBranches(const ProducerReplayGroup &lhs,
                       const ProducerReplayGroup &rhs, Block *scope) {
  if (lhs.vectorWidth != rhs.vectorWidth ||
      !sameCoordinates(lhs.coordinates, rhs.coordinates)) return false;
  for (Operation *parent = lhs.anchor; parent && parent->getBlock() != scope;
       parent = parent->getParentOp()) {
    auto branch = dyn_cast<scf::IfOp>(parent->getParentOp());
    if (!branch) continue;
    bool leftThen = branch.getThenRegion().isAncestor(lhs.anchor->getParentRegion());
    bool leftElse = branch.getElseRegion().isAncestor(lhs.anchor->getParentRegion());
    bool rightThen = branch.getThenRegion().isAncestor(rhs.anchor->getParentRegion());
    bool rightElse = branch.getElseRegion().isAncestor(rhs.anchor->getParentRegion());
    if ((leftThen && rightElse) || (leftElse && rightThen)) return true;
  }
  return false;
}

bool disjointCoordinates(const ProducerReplayGroup &lhs,
                         const ProducerReplayGroup &rhs,
                         IntegerRangeAnalysis &ranges) {
  if (lhs.coordinates.size() != rhs.coordinates.size()) return false;
  for (auto [axis, values] : llvm::enumerate(llvm::zip(lhs.coordinates, rhs.coordinates))) {
    bool last = axis + 1 == lhs.coordinates.size();
    auto interval = [&](Value coordinate, const ProducerReplayGroup &group)
        -> std::optional<std::pair<Value, Value>> {
      auto argument = dyn_cast<BlockArgument>(coordinate);
      auto loop = argument ? dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp()) : scf::ForOp{};
      if (!loop || coordinate != loop.getInductionVar() ||
          !ranges.isPositive(loop.getStep())) return std::nullopt;
      if (last && group.vectorWidth > 1 && group.vectorLimit != loop.getUpperBound())
        return std::nullopt;
      return std::pair{loop.getLowerBound(), loop.getUpperBound()};
    };
    auto firstInterval = interval(std::get<0>(values), lhs);
    auto secondInterval = interval(std::get<1>(values), rhs);
    if (firstInterval && secondInterval &&
        (firstInterval->second == secondInterval->first ||
         secondInterval->second == firstInterval->first)) return true;
    auto first = ranges.range(std::get<0>(values));
    auto second = ranges.range(std::get<1>(values));
    if (!first || !second) continue;
    APInt firstLast = first->smax().sext(65) + APInt(65, last ? lhs.vectorWidth - 1 : 0);
    APInt secondLast = second->smax().sext(65) + APInt(65, last ? rhs.vectorWidth - 1 : 0);
    if (firstLast.slt(second->smin().sext(65)) ||
        secondLast.slt(first->smin().sext(65))) return true;
  }
  return false;
}

} // namespace

bool isLoadReplacement(const ProducerReplay &payload, Value result) {
  if (boundValue(payload, result)) return true;
  auto load = result.getDefiningOp<memref::LoadOp>();
  return load && llvm::all_of(load->getOperands(), [&](Value operand) {
    return boundValue(payload, operand);
  });
}

bool isRebuildableCoordinate(const ProducerReplay &payload, Value result) {
  llvm::DenseMap<Value, bool> known;
  llvm::DenseSet<Value> active;
  std::function<bool(Value)> inspect = [&](Value value) {
    if (!value.getType().isIntOrIndex()) return false;
    if (auto found = known.find(value); found != known.end()) return found->second;
    if (!active.insert(value).second) return false;
    auto infer = [&]() {
      if (matchPattern(value, m_Constant()) ||
          llvm::is_contained(payload.externalValues, value)) return true;
      if (llvm::is_contained(payload.frontier, value)) {
        if (value.getDefiningOp<linalg::IndexOp>()) return true;
        auto argument = dyn_cast<BlockArgument>(value);
        if (!argument) return false;
        Operation *owner = argument.getOwner()->getParentOp();
        if (auto loop = dyn_cast<scf::ForOp>(owner))
          return value == loop.getInductionVar();
        if (auto generic = dyn_cast<linalg::GenericOp>(owner)) {
          unsigned number = argument.getArgNumber();
          // A scalar input is already computed. A memref element formal would
          // introduce a data read when replayed and is not a coordinate leaf.
          return number < generic.getInputs().size() &&
                 generic.getInputs()[number].getType().isIntOrIndex();
        }
        return false;
      }
      Operation *operation = value.getDefiningOp();
      if (!operation || !llvm::is_contained(payload.nodes, operation) ||
          operation->getNumRegions() || !isMemoryEffectFree(operation) ||
          !isSpeculatable(operation)) return false;
      bool supported = isa<arith::AddIOp, arith::SubIOp, arith::CmpIOp,
          arith::AndIOp, arith::OrIOp, arith::XOrIOp, arith::SelectOp,
          arith::MinSIOp, arith::MaxSIOp, arith::MinUIOp, arith::MaxUIOp,
          arith::IndexCastOp, arith::IndexCastUIOp, arith::ExtSIOp,
          arith::ExtUIOp, arith::TruncIOp>(operation);
      if (auto multiply = dyn_cast<arith::MulIOp>(operation))
        supported = getConstantIntValue(multiply.getLhs()).has_value() ||
                    getConstantIntValue(multiply.getRhs()).has_value();
      if (isa<arith::ShLIOp, arith::ShRSIOp, arith::ShRUIOp>(operation)) {
        auto amount = getConstantIntValue(operation->getOperand(1));
        Type type = operation->getOperand(0).getType();
        unsigned width = type.isIndex() ? 64 : cast<IntegerType>(type).getWidth();
        supported = amount && *amount >= 0 && static_cast<uint64_t>(*amount) < width;
      }
      return supported && llvm::all_of(operation->getOperands(), inspect);
    };
    bool answer = infer();
    active.erase(value);
    known.try_emplace(value, answer);
    return answer;
  };
  return inspect(result);
}

FailureOr<ProducerReplay> analyzeProducerResult(linalg::GenericOp producer,
                                                StorageAnalysis &storage,
                                                unsigned resultNumber) {
  // The whole producer must be safe to eliminate, including unused operations.
  auto whole = analyzeLinalgProducer(producer, storage);
  if (failed(whole)) return failure();
  Block &body = producer.getRegion().front();
  if (resultNumber >= body.getTerminator()->getNumOperands()) return failure();
  Value result = body.getTerminator()->getOperand(resultNumber);
  auto selected = analyzeProducerValue(result, producer, whole->frontier, storage);
  if (failed(selected)) return failure();
  llvm::DenseSet<Value> required;
  required.insert(result);
  for (Operation *operation : selected->nodes)
    for (Value operand : operation->getOperands()) required.insert(operand);
  llvm::erase_if(selected->frontier, [&](Value value) { return !required.contains(value); });
  for (auto [number, input] : llvm::enumerate(producer->getOperands())) {
    if (!llvm::is_contained(selected->frontier, body.getArgument(number)) ||
        !isa<MemRefType>(input.getType())) continue;
    if (!llvm::is_contained(selected->reads, input)) selected->reads.push_back(input);
  }
  return selected;
}

FailureOr<AffineMap> fullOutputProjection(Value output, AffineMap outputMap) {
  auto type = dyn_cast<ShapedType>(output.getType());
  if (!type || !type.hasRank() || outputMap.getNumSymbols() ||
      outputMap.getNumResults() != type.getRank()) return failure();
  for (auto [axis, expression] : llvm::enumerate(outputMap.getResults())) {
    if (isa<AffineDimExpr>(expression)) continue;
    auto constant = dyn_cast<AffineConstantExpr>(expression);
    if (!constant || constant.getValue() != 0) return failure();
    int64_t extent = type.getDimSize(axis);
    if (ShapedType::isDynamic(extent)) {
      auto allocation = output.getDefiningOp<memref::AllocOp>();
      if (!allocation) return failure();
      auto size = getConstantIntValue(allocation.getDynamicSizes()[
          allocation.getType().getDynamicDimIndex(axis)]);
      if (!size || *size != 1) return failure();
    } else if (extent != 1) return failure();
  }
  // MLIR's projected-permutation query counts zero results against the number
  // of input dimensions. Drop only the unit axes proved above before checking
  // bijectivity, then invert the original map to retain their physical slots.
  if (!outputMap.dropZeroResults().isPermutation()) return failure();
  AffineMap coordinates = inversePermutation(outputMap);
  if (!coordinates) return failure();
  return coordinates;
}

bool isNonRepeatingProjection(AffineMap coordinates,
                              ArrayRef<int64_t> iterationExtents) {
  if (coordinates.getNumSymbols() || coordinates.getNumDims() != iterationExtents.size())
    return false;
  llvm::SmallBitVector represented(iterationExtents.size());
  for (AffineExpr expression : coordinates.getResults()) {
    if (auto axis = dyn_cast<AffineDimExpr>(expression)) represented.set(axis.getPosition());
    else if (!isa<AffineConstantExpr>(expression)) return false;
  }
  for (auto [axis, extent] : llvm::enumerate(iterationExtents))
    if (extent != 0 && extent != 1 && !represented.test(axis)) return false;
  return true;
}

// Reconstruct a stored representation from one no-wider element read. This is
// a traffic heuristic, not zero work: conversion executes in each consumer,
// while the removed version no longer writes or reads its wider materialization.
// Numeric computation, packed-format decoding and multi-source payloads do not
// qualify. In particular, float8 conversion is not assumed to be a cheap cast.
Value convertedLoadSource(const ProducerReplay &payload, Value result) {
  auto bytes = [](Type type) -> unsigned {
    if (type.isIndex()) return 8;
    if (auto integer = dyn_cast<IntegerType>(type)) {
      unsigned width = integer.getWidth();
      return width == 1 || width == 8 || width == 16 || width == 32 || width == 64
          ? (width + 7) / 8 : 0;
    }
    if (type.isF16() || type.isBF16()) return 2;
    if (type.isF32()) return 4;
    if (type.isF64()) return 8;
    return 0;
  };
  unsigned resultBytes = bytes(result.getType());
  if (!resultBytes) return {};
  Value source = result;
  while (!isLoadReplacement(payload, source)) {
    Operation *conversion = source.getDefiningOp();
    if (!conversion || !llvm::is_contained(payload.nodes, conversion) ||
        conversion->getNumOperands() != 1 || conversion->getNumResults() != 1 ||
        !bytes(conversion->getOperand(0).getType()) ||
        !bytes(conversion->getResult(0).getType()) ||
        !isa<arith::ExtFOp, arith::ExtSIOp, arith::ExtUIOp,
             arith::IndexCastOp, arith::IndexCastUIOp, arith::BitcastOp>(conversion) ||
        !isMemoryEffectFree(conversion) || !isSpeculatable(conversion))
      return {};
    source = conversion->getOperand(0);
  }
  unsigned sourceBytes = bytes(source.getType());
  if (source == result || !sourceBytes || sourceBytes > resultBytes) return {};
  return source;
}

namespace {

bool replacesConvertedLoad(const ProducerReplay &payload, Value result,
                           AffineMap coordinates) {
  Value source = convertedLoadSource(payload, result);
  if (!source) return false;
  if (payload.reads.empty()) return true;
  // A stored representation also buys known dense addressing. Removing it must
  // not add dynamic-stride versioning or turn a contiguous consumer into a
  // permuted/strided traversal just because its source element occupies less.
  auto producer = dyn_cast<linalg::GenericOp>(payload.scope);
  auto formal = dyn_cast<BlockArgument>(source);
  if (!producer || !formal || formal.getOwner() != &producer.getRegion().front() ||
      formal.getArgNumber() >= producer.getInputs().size() ||
      producer.getOutputs().size() != 1) return false;
  Value input = producer.getInputs()[formal.getArgNumber()];
  if (payload.reads.size() != 1 || payload.reads.front() != input ||
      !isContiguousDescriptor(input)) return false;
  auto addressAxes = [&](Value memory, AffineMap map) {
    SmallVector<AffineExpr> axes;
    for (auto [axis, expression] : llvm::enumerate(map.getResults())) {
      auto extent = queryExtentValue(memory, axis);
      if (extent && getConstantIntValue(*extent) == 1) continue;
      axes.push_back(expression);
    }
    return AffineMap::get(map.getNumDims(), map.getNumSymbols(), axes,
                          map.getContext()).compose(coordinates);
  };
  auto maps = producer.getIndexingMapsArray();
  return addressAxes(input, maps[formal.getArgNumber()]) ==
         addressAxes(producer.getOutputs()[0], maps.back());
}

} // namespace

bool shouldFuseProducerResult(const ProducerReplay &payload, Value result,
                              AffineMap coordinates,
                              ArrayRef<int64_t> iterationExtents,
                              bool removesProducer,
                              unsigned consumerTraversals) {
  return isLoadReplacement(payload, result) ||
         isRebuildableCoordinate(payload, result) ||
         (removesProducer && isNonRepeatingProjection(coordinates, iterationExtents) &&
          (consumerTraversals == 1 || replacesConvertedLoad(payload, result, coordinates)));
}

FailureOr<SmallVector<ProducerReplayGroup>> groupProducerReplays(
    const ProducerReplay &payload, Value result, Operation *producer,
    Block *allocationScope, ArrayRef<ProducerReplayUse> uses,
    bool removesProducer, StorageAnalysis &storage) {
  bool coordinateReplay = isRebuildableCoordinate(payload, result);
  bool perUseReplay = isLoadReplacement(payload, result) || coordinateReplay;
  bool speculatableReplay = coordinateReplay && payload.reads.empty();
  if (uses.empty() || (!perUseReplay && !removesProducer)) return failure();
  // A read scheduled for replacement must not itself be a stored coordinate
  // binding of another group (e.g. table[table[i]]). No query result keeps that
  // soon-to-be-erased SSA value alive across the rewrite.
  for (const auto &use : uses)
    for (Value coordinate : use.coordinates)
      if (llvm::any_of(uses, [&](const ProducerReplayUse &other) {
            return coordinate.getDefiningOp() == other.operation;
          })) return failure();
  for (const auto &use : uses)
    if (use.vectorLimit && llvm::any_of(uses, [&](const ProducerReplayUse &other) {
          return use.vectorLimit.getDefiningOp() == other.operation;
        })) return failure();
  DominanceInfo dominance(producer->getParentOfType<func::FuncOp>());
  IntegerRangeAnalysis ranges;
  SmallVector<ProducerReplayGroup> groups;
  for (const auto &use : uses) {
    if (use.vectorWidth <= 0 ||
        !canReplayProducerAt(payload, producer, use.operation, storage)) return failure();
    Operation *anchor = use.operation;
    while (anchor->getBlock() != allocationScope) {
      auto loop = dyn_cast<scf::ForOp>(anchor->getParentOp());
      if (!loop || (!speculatableReplay && !nonempty(loop, ranges)) ||
          llvm::any_of(use.coordinates, [&](Value coordinate) {
            return !dominance.dominates(coordinate, loop);
          }) || !canReplayProducerAt(payload, producer, loop, storage)) break;
      anchor = loop;
    }
    auto found = llvm::find_if(groups, [&](const ProducerReplayGroup &group) {
      return group.anchor->getBlock() == anchor->getBlock() &&
             group.vectorWidth == use.vectorWidth &&
             group.vectorLimit == use.vectorLimit &&
             group.uses.front()->getResult(0).getType() == use.operation->getResult(0).getType() &&
             sameCoordinates(group.coordinates, use.coordinates);
    });
    if (found == groups.end()) {
      groups.push_back({anchor, use.coordinates, {use.operation}, use.vectorWidth, use.vectorLimit});
    } else {
      if (anchor->isBeforeInBlock(found->anchor)) found->anchor = anchor;
      found->uses.push_back(use.operation);
    }
  }
  if (!perUseReplay) {
    for (const auto &group : groups)
      if (!nonRepeating(group, allocationScope, ranges)) return failure();
    for (auto [index, group] : llvm::enumerate(groups))
      for (const auto &other : ArrayRef(groups).drop_front(index + 1))
        if (!exclusiveBranches(group, other, allocationScope) &&
            !disjointCoordinates(group, other, ranges)) return failure();
  }
  return groups;
}

bool isRemovableProducerMetadata(Operation *operation) {
  if (auto dimension = dyn_cast<memref::DimOp>(operation))
    return dimension.getConstantIndex().has_value();
  return isa<memref::DeallocOp>(operation);
}

void eraseUnusedProducer(linalg::GenericOp producer) {
  if (producer.getOutputs().size() != 1) return;
  Value buffer = producer.getOutputs()[0];
  auto allocation = buffer.getDefiningOp<memref::AllocOp>();
  if (!allocation) return;
  for (Operation *user : buffer.getUsers())
    if (user != producer && !isRemovableProducerMetadata(user)) return;
  SmallVector<Operation *> users(buffer.getUsers());
  for (Operation *user : users) {
    auto dimension = dyn_cast<memref::DimOp>(user);
    if (!dimension) continue;
    int64_t axis = *dimension.getConstantIndex();
    OpBuilder builder(dimension);
    Value extent = allocation.getType().isDynamicDim(axis)
        ? allocation.getDynamicSizes()[allocation.getType().getDynamicDimIndex(axis)]
        : Value(builder.create<arith::ConstantIndexOp>(dimension.getLoc(), allocation.getType().getDimSize(axis)));
    dimension.replaceAllUsesWith(extent);
    dimension.erase();
  }
  producer.erase();
  SmallVector<Operation *> remaining(buffer.getUsers());
  for (Operation *user : remaining) cast<memref::DeallocOp>(user).erase();
  allocation.erase();
}

} // namespace intent::cpu
