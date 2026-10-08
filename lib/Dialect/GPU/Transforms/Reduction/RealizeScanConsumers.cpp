#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/Helpers.h"
#include "Intent/Dialect/GPU/Analysis/IndexRelations.h"
#include "Intent/Dialect/GPU/Analysis/IterationDependencies.h"
#include "Intent/Dialect/GPU/Analysis/MemoryEffects.h"
#include "Intent/Dialect/GPU/Analysis/ResourceAlias.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/PhysicalExpressions.h"
#include "Intent/Dialect/GPU/IR/ProgramInterface.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Transforms/Value/ExecutionSchema.h"
#include "Intent/Dialect/GPU/Transforms/Control/Traversal.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Transforms/Control/Predication.h"
#include "Intent/Dialect/GPU/Transforms/Storage/Storage.h"
#include "Intent/Dialect/GPU/Transforms/Storage/Workspace.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/StringSet.h"

#include <functional>

using namespace mlir;

namespace intent::gpu {
namespace {

struct ScanConsumerMatch {
  ScanOp scan;
  scf::ForOp loop;
  Value identity;
  SmallVector<MakeRangeOp> ranges;
  SmallVector<GatherOp> terminals;
  SmallVector<CastOp> projections;
  SmallVector<LoadOp> sourceLoads;
  SmallVector<GatherOp> members;
  IndependentIterationAccesses independence;
};

bool isScanProjection(CastOp cast) {
  if (!cast || !canPredicateValueOperation(cast))
    return false;
  auto source = dyn_cast<FragmentType>(cast.getValue().getType());
  auto result = dyn_cast<FragmentType>(cast.getType());
  return source && result && source.getShape() == result.getShape() &&
         source.getAxisMaps() == result.getAxisMaps() &&
         source.getOwner() == result.getOwner() &&
         source.getValidity() == result.getValidity();
}

bool isLastScanRead(GatherOp gather, Value stop) {
  if (gather.getCoordinates().size() != 1 ||
      gather.getSourceAxes() != ArrayRef<int64_t>{0} ||
      !gather.getResult().getType().isIntOrIndexOrFloat())
    return false;
  Value coordinate = gather.getCoordinates().front();
  auto value = IndexRelations().constant(coordinate);
  auto end = IndexRelations().constant(stop);
  if (value && end && *end > 0 && *value == *end - 1)
    return true;
  auto subtract = coordinate.getDefiningOp<BinaryOp>();
  return subtract && subtract.getOperatorKind() == BinaryOperator::Subtract &&
         samePhysicalScalarExpression(subtract.getLhs(), stop) &&
         IndexRelations().constant(subtract.getRhs()) == 1;
}

std::optional<ScanConsumerMatch>
matchScanConsumer(ScanOp scan, PhysicalProgramAnalysis &analysis,
                  bool closeProducers = true) {
  if (scan.getSources().size() != 1 || scan.getIdentities().size() != 1 ||
      scan.getCaptures().size() || scan.getAxis() != 0 ||
      scan.getReverse())
    return std::nullopt;
  auto type = dyn_cast<FragmentType>(scan.getResult(0).getType());
  if (!type || type.getShape().size() != 1)
    return std::nullopt;
  ScanConsumerMatch match;
  match.scan = scan;
  match.identity = scan.getIdentities().front();
  while (true) {
    if (auto splat = match.identity.getDefiningOp<SplatOp>())
      match.identity = splat.getValue();
    else if (auto broadcast = match.identity.getDefiningOp<BroadcastOp>())
      match.identity = broadcast.getValue();
    else
      break;
  }
  if (match.identity.getType() != type.getElementType())
    return std::nullopt;
  PhysicalRangeFact ranges = analysis.axisRanges(scan.getSources().front(), 0);
  auto root = queryExactLogicalRange(ranges);
  if (failed(root) || ranges.roots.empty() ||
      !analysis.lockstepRanges(ranges.roots).isExact())
    return std::nullopt;
  SmallVector<Value> projections{scan.getResult(0)};
  llvm::DenseSet<Value> visitedProjections;
  for (unsigned position = 0; position < projections.size(); ++position) {
    if (!visitedProjections.insert(projections[position]).second)
      continue;
    for (Operation *user : projections[position].getUsers()) {
      if (auto cast = dyn_cast<CastOp>(user);
          isScanProjection(cast) && cast->getBlock() == scan->getBlock()) {
        projections.push_back(cast.getResult());
        match.projections.push_back(cast);
        continue;
      }
    auto gather = dyn_cast<GatherOp>(user);
    if (gather && gather.getSource() == projections[position] &&
        isLastScanRead(gather, (*root).getLogicalStop())) {
      match.terminals.push_back(gather);
      continue;
    }
    auto loop = user->getParentOfType<scf::ForOp>();
    if (!gather || gather.getSource() != projections[position] ||
        !loop ||
        loop.getNumRegionIterArgs() ||
        loop->getBlock() != scan->getBlock() || !scan->isBeforeInBlock(loop) ||
        (match.loop && match.loop != loop) || gather.getCoordinates().size() != 1 ||
        gather.getCoordinates().front() != loop.getInductionVar() ||
        gather.getSourceAxes() != ArrayRef<int64_t>{0})
      return std::nullopt;
    match.loop = loop;
    }
  }
  if (!match.loop || !canPredicateScalarBlock(*match.loop.getBody()) ||
      !match.loop.getInductionVar().getType().isIndex())
    return std::nullopt;
  auto isConstant = [&](Value value, int64_t expected) {
    PhysicalExprAttr expression = queryLaunchExpression(value);
    return expression &&
           expression.getKind() ==
               PhysicalExprKind::Constant &&
           expression.getValue() == expected;
  };
  if (!isConstant(match.loop.getLowerBound(), 0) ||
      !isConstant(match.loop.getStep(), 1))
    return std::nullopt;
  DominanceInfo dominance(scan->getParentOfType<func::FuncOp>());
  for (GatherOp terminal : match.terminals) {
    if (!scan.getInclusive() || match.loop->isAncestor(terminal) ||
        !dominance.properlyDominates(match.loop, terminal))
      return std::nullopt;
  }
  match.ranges.assign(ranges.roots.begin(), ranges.roots.end());
  for (MakeRangeOp range : ranges.roots)
    if (!isConstant(range.getStart(), 0) ||
        !isConstant(range.getLogicalStart(), 0) ||
        !isConstant(range.getStep(), 1) ||
        !samePhysicalScalarExpression(range.getLogicalStop(),
                                     match.loop.getUpperBound()))
      return std::nullopt;

  if (!match.loop->hasAttr(independentIterationAttr)) {
    auto independent = queryIndependentIterationAccesses(match.loop, scan);
    if (failed(independent)) return std::nullopt;
    match.independence = std::move(*independent);
  }

  // Replay eligibility is a coordinate/value fact. The memory epoch is checked
  // separately, through the whole consumer loop and in the actual guard arm.
  llvm::DenseSet<Operation *> sourceLoads;
  for (Operation *access : ranges.accesses) {
    auto load = dyn_cast<LoadOp>(access);
    auto view =
        load ? dyn_cast<ViewType>(load.getResource().getType()) : ViewType();
    if (!load || !view ||
        !analysis.boundaryValidity(load, /*allowRangeGuards=*/true).isExact())
      return std::nullopt;
    std::optional<unsigned> memberCoordinate;
    for (auto [index, coordinate] : llvm::enumerate(load.getCoordinates())) {
      auto fragment = dyn_cast<FragmentType>(coordinate.getType());
      if (!fragment) {
        if (!coordinate.getType().isIndex() ||
            !match.loop.isDefinedOutsideOfLoop(coordinate))
          return std::nullopt;
        continue;
      }
      if (memberCoordinate || fragment.getShape().size() != 1 ||
          !fragment.getElementType().isIndex())
        return std::nullopt;
      auto coordinates = analysis.axisRanges(coordinate, 0);
      if (!coordinates.isExact() || coordinates.roots.empty() ||
          !llvm::all_of(coordinates.roots, [&](MakeRangeOp root) {
            return llvm::is_contained(ranges.roots, root);
          }) || queryLaunchExpression(match.loop.getUpperBound()) !=
                     view.getLayout().getExtents()[load.getSourceAxes()[index]])
        return std::nullopt;
      memberCoordinate = index;
    }
    if (!memberCoordinate)
      return std::nullopt;
    sourceLoads.insert(load);
    match.sourceLoads.push_back(load);
  }
  llvm::DenseSet<Value> visited;
  std::function<bool(Value)> safeSource = [&](Value value) {
    if (!visited.insert(value).second)
      return true;
    Operation *producer = value.getDefiningOp();
    if (!producer)
      return isa<BlockArgument>(value);
    if (isa<LoadOp>(producer))
      return sourceLoads.contains(producer);
    if (isa<MakeRangeOp>(producer))
      return llvm::is_contained(ranges.roots, cast<MakeRangeOp>(producer));
    if (isa<GatherOp>(producer) || !canPredicateValueOperation(producer))
      return false;
    return llvm::all_of(producer->getOperands(), safeSource);
  };
  if (!safeSource(scan.getSources().front()))
    return std::nullopt;
  // Close the full-domain producers' live uses before duplicating their DAG in
  // a bounded traversal. A second full consumer would retain the original
  // value and defeat the storage/lifetime benefit of streaming it.
  llvm::DenseSet<Value> prefixValues(projections.begin(), projections.end());
  match.loop.walk([&](GatherOp gather) {
    if ((visited.contains(gather.getSource()) ||
         prefixValues.contains(gather.getSource())) &&
        gather.getSourceAxes() == ArrayRef<int64_t>{0} &&
        gather.getCoordinates().size() == 1 &&
        gather.getCoordinates().front() == match.loop.getInductionVar() &&
        gather.getType().isIntOrIndexOrFloat())
      match.members.push_back(gather);
  });
  for (Value value : visited) {
    if (!closeProducers) break;
    if (!isa<FragmentType>(value.getType()) ||
        value.getDefiningOp<MakeRangeOp>()) continue;
    for (Operation *user : value.getUsers()) {
      if (user == scan || llvm::any_of(user->getResults(),
              [&](Value result) { return visited.contains(result); })) continue;
      auto gather = dyn_cast<GatherOp>(user);
      if (!gather || !llvm::is_contained(match.members, gather))
        return std::nullopt;
    }
  }
  return match;
}

bool matchesMandatoryMemberRead(ScanConsumerMatch &match, LoadOp source,
                               PhysicalProgramAnalysis &analysis) {
  if (!match.loop->hasAttr(independentIterationAttr)) return false;
  for (Operation &operation : match.loop.getBody()->without_terminator()) {
    if (isa<StoreOp>(operation)) break;
    auto reader = dyn_cast<LoadOp>(operation);
    if (!reader || reader.getResource() != source.getResource() ||
        reader.getSourceAxes() != source.getSourceAxes() ||
        reader.getCoordinates().size() != source.getCoordinates().size() ||
        !analysis.boundaryValidity(reader).isExact()) continue;
    if (llvm::all_of(llvm::zip(source.getCoordinates(), reader.getCoordinates()),
        [&](auto pair) {
          auto [full, member] = pair;
          if (isa<FragmentType>(full.getType()))
            return member == match.loop.getInductionVar();
          return samePhysicalScalarExpression(full, member);
        })) return true;
  }
  return false;
}

// Include every write between the original SSA read and the end of the
// consumer, not just writes before its loop. A scan source is a snapshot; a
// later chunk may only reload it when the entire traversal preserves its epoch.
bool preservesScanSourceEpoch(ScanConsumerMatch &match,
                              PhysicalProgramAnalysis &analysis,
                              bool collectGuards) {
  ResourceAliasAnalysis aliases;
  auto kernel = match.scan->getParentOfType<func::FuncOp>();
  for (LoadOp load : match.sourceLoads) {
    if (!collectGuards) {
      bool memberRead = matchesMandatoryMemberRead(match, load, analysis);
      auto effects = getEffectsRecursively(match.loop);
      if (!effects || llvm::any_of(*effects, [](const auto &effect) {
            return isa<MemoryEffects::Free>(effect.getEffect());
          })) return false;
      if (!canReplayReadAt(load, match.loop) ||
          !preservesMemoryReads(load, match.loop, aliases,
              [&](Value lhs, Value rhs) {
                return aliases.disjointAt(lhs, rhs, match.loop) ||
                       (lhs == rhs && memberRead);
              })) return false;
      continue;
    }
    if (load->getBlock() != match.loop->getBlock() ||
        !load->isBeforeInBlock(match.loop)) return false;
    bool memberRead = matchesMandatoryMemberRead(match, load, analysis);
    for (Operation *operation = load->getNextNode(); operation;
         operation = operation->getNextNode()) {
      auto effects = getEffectsRecursively(operation);
      if (!effects || llvm::any_of(*effects, [](const auto &effect) {
            return isa<MemoryEffects::Free>(effect.getEffect());
          })) return false;
      auto disjoint = [&](Value lhs, Value rhs) {
        if (aliases.disjointAt(lhs, rhs, match.loop)) return true;
        if (operation == match.loop && lhs == rhs && memberRead) return true;
        auto left = dyn_cast<BlockArgument>(lhs), right = dyn_cast<BlockArgument>(rhs);
        if (lhs == rhs || !left || !right ||
            left.getOwner() != &kernel.front() || right.getOwner() != &kernel.front() ||
            !getPublicView(lhs) || !getPublicView(rhs)) return false;
        auto pair = std::pair<Value, Value>{lhs, rhs};
        auto reverse = std::pair<Value, Value>{rhs, lhs};
        if (!llvm::is_contained(match.independence.disjointViews, pair) &&
            !llvm::is_contained(match.independence.disjointViews, reverse))
          match.independence.disjointViews.push_back(pair);
        return true;
      };
      if (!preservesMemoryReads(load, operation, aliases, disjoint)) return false;
      if (operation == match.loop) break;
    }
    // In the selected arm, re-query the same public replay authority used by
    // all other consumers. Prospective byte-span guards grant no permission.
  }
  return true;
}

FailureOr<Value> materializeScanGuard(func::FuncOp kernel,
                                     ScanConsumerMatch &match) {
  Value condition;
  OpBuilder builder(match.loop);
  Location location = match.scan.getLoc();
  auto append = [&](Value predicate) {
    condition = condition ? builder.create<BinaryOp>(
        location, builder.getI1Type(), condition, predicate,
        BinaryOperator::LogicalAnd).getResult() : predicate;
  };
  for (Value view : match.independence.guardedViews) {
    auto injective = materializeNonOverlappingView(kernel, view);
    if (failed(injective)) return failure();
    append(*injective);
  }
  for (auto [lhs, rhs] : match.independence.disjointViews) {
    OpBuilder entry(&kernel.front(), kernel.front().begin());
    Value overlap = entry.create<ViewOverlapOp>(location, entry.getI1Type(), lhs, rhs);
    Value zero = builder.create<arith::ConstantOp>(location, builder.getBoolAttr(false));
    append(builder.create<CompareOp>(location, builder.getI1Type(), overlap,
                                     zero, ComparePredicate::Eq));
  }
  for (auto [count, maximum] : match.independence.countUpperBounds) {
    Value limit = builder.create<arith::ConstantOp>(
        location, builder.getIntegerAttr(count.getType(), maximum));
    append(builder.create<CompareOp>(location, builder.getI1Type(), count,
                                     limit, ComparePredicate::Le));
  }
  return condition;
}

LogicalResult restoreReplayedReadBounds(OpBuilder &builder) {
  SmallVector<LoadOp> reads;
  for (Operation &operation : *builder.getInsertionBlock())
    if (auto load = dyn_cast<LoadOp>(operation))
      reads.push_back(load);
  for (LoadOp load : reads) {
    auto type = dyn_cast<FragmentType>(load.getType());
    if (!type || !isa<ViewType>(load.getResource().getType()))
      return failure();
    OpBuilder at(load);
    Value valid = load.getValid();
    for (auto [axis, coordinate] :
         llvm::zip(load.getSourceAxes(), load.getCoordinates())) {
      Value zero = at.create<arith::ConstantIndexOp>(load.getLoc(), 0);
      Value extent = at.create<DimOp>(load.getLoc(), at.getIndexType(),
                                     load.getResource(), axis);
      Type predicateType = at.getI1Type();
      if (auto fragment = dyn_cast<FragmentType>(coordinate.getType())) {
        auto projectedZero = projectPhysicalValueToSchema(at, load.getLoc(), zero, fragment);
        auto projectedExtent = projectPhysicalValueToSchema(at, load.getLoc(), extent, fragment);
        if (failed(projectedZero) || failed(projectedExtent))
          return failure();
        zero = *projectedZero;
        extent = *projectedExtent;
        predicateType = FragmentType::get(
            load.getContext(), at.getI1Type(), fragment.getShape(),
            fragment.getAxisMaps(), fragment.getValidity(), fragment.getOwner());
      }
      Value lower = at.create<CompareOp>(load.getLoc(), predicateType,
                                        coordinate, zero, ComparePredicate::Ge);
      Value upper = at.create<CompareOp>(load.getLoc(), predicateType,
                                        coordinate, extent, ComparePredicate::Lt);
      Value bounds = at.create<BinaryOp>(load.getLoc(), predicateType,
                                        lower, upper, BinaryOperator::LogicalAnd);
      auto bounded = materializeValidityConjunction(at, load.getLoc(), valid, bounds, type);
      if (failed(bounded))
        return failure();
      valid = *bounded;
    }
    load.getValidMutable().assign(valid);
  }
  return success();
}

struct ScanTraversal {
  ParameterOp chunk;
  Value carry;
};

FailureOr<ScanTraversal> realizeScanConsumerMatch(func::FuncOp kernel,
                                               ScanConsumerMatch &match) {
  ScanOp scan = match.scan;
  auto original = cast<FragmentType>(scan.getResult(0).getType());
  auto mapping = cast<AxisMapAttr>(original.getAxisMaps()[0]);
  PhysicalSourceAxis source = sourceAxisIdentity(mapping);
  llvm::StringSet<> names;
  for (Attribute declaration : getParameterDeclarations(kernel))
    names.insert(cast<ParameterAttr>(declaration).getName().getValue());
  std::string name = ("SCAN_CHUNK_S" + Twine(source.sourceId) + "_A" +
                      Twine(source.sourceAxis))
                         .str();
  while (names.contains(name))
    name += "_";
  Type element = original.getElementType();
  auto identity = match.identity.getDefiningOp<arith::ConstantOp>();
  auto zero = identity ? dyn_cast<IntegerAttr>(identity.getValue()) : IntegerAttr();
  auto binaryCombine = queryBinaryCombine(scan.getCombine());
  bool invertIntegerSum =
      isa<IntegerType, IndexType>(element) && !element.isInteger(1) && zero &&
      zero.getValue().isZero() &&
      binaryCombine && binaryCombine->kind() == BinaryOperator::Add;
  auto reference = getOrCreatePhysicalParameter(
      kernel, name, ParameterRole::ScanChunk, ParameterCategory::Scan,
      element.isIndex() ? 64 : element.getIntOrFloatBitWidth(),
      {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192},
      ParameterBindingAttr::get(kernel.getContext(), {},
          PhysicalSourceAttr::get(kernel.getContext(), source.sourceId, source.sourceAxis, source.derived),
          {}, {}, false, false));
  if (failed(reference)) return failure();
  OpBuilder entry(&kernel.front(), kernel.front().begin());
  auto chunk = materializeParameter(entry, scan.getLoc(), *reference);
  PhysicalExprAttr extent = queryLaunchExpression(chunk);
  auto fragment = [&](Type element) {
    return FragmentType::get(kernel.getContext(), element,
                             ArrayAttr::get(kernel.getContext(), {extent}),
                             original.getAxisMaps(), original.getValidity(),
                             original.getOwner());
  };
  OpBuilder builder(match.loop);
  bool failedBody = false;
  auto traversal = createTraversalLoop(
      builder, scan.getLoc(), match.loop.getLowerBound(), match.loop.getUpperBound(), chunk,
      ValueRange{match.identity},
      [&](OpBuilder &nested, Location location, Value offset, ValueRange carry) {
        auto yield = nested.create<scf::YieldOp>(location, carry);
        nested.setInsertionPoint(yield);
        Value one = nested.create<arith::ConstantIndexOp>(location, 1);
        auto range = nested.create<MakeRangeOp>(
            location, fragment(nested.getIndexType()), offset, chunk, one,
            match.loop.getLowerBound(), match.loop.getUpperBound(),
            source.sourceId, source.sourceAxis, source.derived);
        auto lift = [&](Value value) -> Value {
          auto existing = dyn_cast<FragmentType>(value.getType());
          Type element = existing ? existing.getElementType() : value.getType();
          Type target = fragment(element);
          return value.getType() == target
                     ? value
                     : Value(nested.create<BroadcastOp>(location, target, value));
        };
        Value upperTail = nested.create<CompareOp>(
            location, fragment(nested.getI1Type()), range,
            lift(match.loop.getUpperBound()), ComparePredicate::Lt);
        Value lowerTail = nested.create<CompareOp>(
            location, fragment(nested.getI1Type()), range,
            lift(match.loop.getLowerBound()), ComparePredicate::Ge);
        Value tail = nested.create<BinaryOp>(
            location, fragment(nested.getI1Type()), lowerTail, upperTail,
            BinaryOperator::LogicalAnd);
        Value identity = lift(match.identity);
        IRMapping replay;
        replay.map(scan.getIdentities().front(), identity);
        for (MakeRangeOp root : match.ranges)
          replay.map(root.getResult(), range.getResult());
        ReplayMaterializationOptions options;
        options.scope = PhysicalReplayScope::Coordinate;
        options.traversalRanges = match.ranges;
        options.fragmentAxis = 0;
        options.segmentTail = tail;
        options.segmentMapping = mapping;
        options.materializeZeroFill = true;
        FailureOr<Value> replayed = materializeReplayedValue(
            nested, location, scan.getSources().front(), source, extent, replay,
            match.loop, options);
        if (failed(replayed)) {
          failedBody = true;
          return;
        }
        // Original full-domain bounds may have been simplified before replay.
        // Re-establish them on each actual selected resource axis, without
        // replacing clamped coordinates or changing the original load fill.
        // matchScanConsumer proved these reads valid over every original member.
        if (failed(restoreReplayedReadBounds(nested))) {
          failedBody = true;
          return;
        }
        Value sourceSlice = nested.create<SelectOp>(
            location, (*replayed).getType(), tail, *replayed, identity);
        replay.map(scan.getSources().front(), sourceSlice);
        auto local = cast<ScanOp>(nested.clone(*scan, replay));
        local.setInclusive(true);
        if (failed(retargetSourceExtent(local.getResult(0), source, extent))) {
          failedBody = true;
          return;
        }
        IRMapping combine;
        Block &body = local.getCombine().front();
        for (BlockArgument argument : body.getArguments())
          if (failed(retargetSourceExtent(argument, source, extent))) {
            failedBody = true;
            return;
          }
        combine.map(body.getArgument(0), lift(carry.front()));
        combine.map(body.getArgument(1), local.getResult(0));
        for (Operation &operation : body.without_terminator())
          nested.clone(operation, combine);
        Value prefix = combine.lookup(
            cast<YieldOp>(body.getTerminator()).getValues().front());
        Value consumerPrefix = prefix;
        if (!scan.getInclusive() && invertIntegerSum) {
          // Integer addition/subtraction are modular. Remove this lane's input
          // exactly, including the previous chunk's carry, without a lane shift.
          consumerPrefix = nested.create<BinaryOp>(
              location, prefix.getType(), prefix, sourceSlice,
              BinaryOperator::Subtract);
        } else if (!scan.getInclusive()) {
          Value ordinal = nested.create<BinaryOp>(
              location, fragment(nested.getIndexType()), range, lift(offset),
              BinaryOperator::Subtract);
          Value shifted = nested.create<BinaryOp>(
              location, fragment(nested.getIndexType()), ordinal, lift(one),
              BinaryOperator::Subtract);
          Value zero = nested.create<arith::ConstantIndexOp>(location, 0);
          Value lower = nested.create<CompareOp>(
              location, fragment(nested.getI1Type()), shifted, lift(zero), ComparePredicate::Ge);
          Value upper = nested.create<CompareOp>(
              location, fragment(nested.getI1Type()), shifted, lift(chunk), ComparePredicate::Lt);
          Value valid = nested.create<BinaryOp>(
              location, fragment(nested.getI1Type()), lower, upper, BinaryOperator::LogicalAnd);
          consumerPrefix = nested.create<GatherOp>(
              location, prefix.getType(), prefix, ValueRange{shifted}, valid,
              lift(carry.front()), nested.getDenseI64ArrayAttr({0}));
        }
        IRMapping consumers;
        consumers.map(match.loop.getInductionVar(), range);
        IRMapping prefixValues;
        prefixValues.map(scan.getResult(0), consumerPrefix);
        for (GatherOp gather : match.members) {
          Value root = gather.getSource();
          SmallVector<CastOp> projections;
          while (root) {
            auto cast = root.getDefiningOp<CastOp>();
            if (!isScanProjection(cast))
              break;
            projections.push_back(cast);
            root = cast.getValue();
          }
          Value selected;
          if (root == scan.getResult(0)) {
            for (CastOp cast : llvm::reverse(projections))
              if (!prefixValues.lookupOrNull(cast.getResult())) {
                auto result = llvm::cast<FragmentType>(cast.getResult().getType());
                Value projected = nested.create<CastOp>(
                    cast.getLoc(), fragment(result.getElementType()),
                    prefixValues.lookup(cast.getValue()));
                prefixValues.map(cast.getResult(), projected);
              }
            selected = prefixValues.lookup(gather.getSource());
          } else {
            selected = replay.lookupOrNull(gather.getSource());
            if (!selected) {
              auto member = materializeReplayedValue(
                  nested, location, gather.getSource(), source, extent, replay,
                  match.loop, options);
              if (failed(member)) { failedBody = true; return; }
              selected = *member;
            }
          }
          consumers.map(gather.getResult(), selected);
        }
        for (Operation &operation : match.loop.getBody()->without_terminator()) {
          if (failed(clonePredicatedScalarOperation(
                  nested, &operation, consumers, tail,
                  fragment(nested.getIndexType())))) {
            failedBody = true;
            return;
          }
        }
        Value last = nested.create<BinaryOp>(location, nested.getIndexType(), chunk,
                                            one, BinaryOperator::Subtract);
        Value zero = nested.create<arith::ConstantIndexOp>(location, 0);
        Value lower = nested.create<CompareOp>(location, nested.getI1Type(), last,
                                              zero, ComparePredicate::Ge);
        Value upper = nested.create<CompareOp>(location, nested.getI1Type(), last,
                                              chunk, ComparePredicate::Lt);
        Value valid = nested.create<BinaryOp>(location, nested.getI1Type(), lower,
                                             upper, BinaryOperator::LogicalAnd);
        Value next = nested.create<GatherOp>(
            location, original.getElementType(), prefix, ValueRange{last}, valid,
            match.identity, nested.getDenseI64ArrayAttr({0}));
        yield->setOperand(0, next);
      });
  if (failedBody)
    return failure();
  Value carry = traversal.getResult(0);
  match.loop.erase();
  return ScanTraversal{chunk, carry};
}

LogicalResult replaceScanTerminals(ScanConsumerMatch &match, Value carry) {
  ScanOp scan = match.scan;
  for (GatherOp terminal : match.terminals) {
    OpBuilder at(terminal);
    Value result = carry;
    SmallVector<CastOp> projections;
    for (Value source = terminal.getSource(); source != scan.getResult(0);) {
      auto cast = source.getDefiningOp<CastOp>();
      projections.push_back(cast);
      source = cast.getValue();
    }
    for (CastOp projection : llvm::reverse(projections)) {
      IRMapping mapping;
      mapping.map(projection.getValue(), result);
      auto cast = cloneWithSchema(at, projection, mapping,
          TypeRange{mlir::cast<FragmentType>(projection.getType()).getElementType()});
      if (failed(cast))
        return failure();
      result = cast->front();
    }
    if (terminal.getValid()) {
      Value fill = terminal.getFill();
      if (!fill)
        fill = at.create<arith::ConstantOp>(terminal.getLoc(), at.getZeroAttr(result.getType()));
      result = at.create<SelectOp>(terminal.getLoc(), result.getType(),
                                  terminal.getValid(), result, fill);
    }
    terminal.getResult().replaceAllUsesWith(result);
    terminal.erase();
  }
  return success();
}

FailureOr<bool> realizeGuardedScanConsumer(func::FuncOp kernel,
                                         ScanConsumerMatch &match,
                                         PhysicalProgramAnalysis &analysis) {
  if (!preservesScanSourceEpoch(match, analysis, /*collectGuards=*/true))
    return false;
  auto condition = materializeScanGuard(kernel, match);
  if (failed(condition)) return failure();
  if (!*condition) {
    if (!preservesScanSourceEpoch(match, analysis, /*collectGuards=*/false))
      return false;
    auto result = realizeScanConsumerMatch(kernel, match);
    if (failed(result) || failed(replaceScanTerminals(match, result->carry)))
      return failure();
    return true;
  }
  // Only the pure scan/projections and zero-result consumer move into the
  // branches. Intervening values and effects remain in their original block.
  // Only a real selected control arm may consume contextual alias knowledge.
  OpBuilder builder(match.loop);
  bool terminal = !match.terminals.empty();
  Type element = cast<FragmentType>(match.scan.getResult(0).getType()).getElementType();
  auto branch = builder.create<scf::IfOp>(match.scan.getLoc(),
      terminal ? TypeRange{element} : TypeRange{}, *condition, true);
  for (Region &region : branch->getRegions())
    if (!region.front().empty()) region.front().back().erase();
  IRMapping selected;
  builder.setInsertionPointToStart(branch.thenBlock());
  builder.clone(*match.scan, selected);
  for (CastOp projection : match.projections)
    if (projection->isBeforeInBlock(match.loop)) builder.clone(*projection, selected);
  builder.clone(*match.loop, selected);
  ScanConsumerMatch fast = match;
  fast.scan = cast<ScanOp>(selected.lookup(match.scan.getResult(0)).getDefiningOp());
  fast.loop = cast<scf::ForOp>(cast<BlockArgument>(
      selected.lookup(match.loop.getInductionVar())).getOwner()->getParentOp());
  for (GatherOp &member : fast.members)
    member = cast<GatherOp>(selected.lookup(member.getResult()).getDefiningOp());
  fast.terminals.clear();
  // Producer reads retain their original epoch. The false path still uses the
  // immutable full-domain SSA values, without replay through consumer writes.
  SmallVector<CastOp> movedProjections;
  for (CastOp projection : match.projections)
    if (projection->isBeforeInBlock(match.loop)) movedProjections.push_back(projection);
  match.scan->moveBefore(branch.elseBlock(), branch.elseBlock()->end());
  for (CastOp projection : movedProjections)
    projection->moveBefore(branch.elseBlock(), branch.elseBlock()->end());
  match.loop->moveBefore(branch.elseBlock(), branch.elseBlock()->end());
  PhysicalProgramAnalysis selectedAnalysis(kernel);
  if (!preservesScanSourceEpoch(fast, selectedAnalysis, /*collectGuards=*/false))
    return fast.scan.emitOpError("selected scan traversal does not preserve its source epoch");
  auto traversal = realizeScanConsumerMatch(kernel, fast);
  if (failed(traversal)) return failure();
  builder.setInsertionPointToEnd(branch.thenBlock());
  builder.create<scf::YieldOp>(branch.getLoc(),
      terminal ? ValueRange{traversal->carry} : ValueRange{});
  builder.setInsertionPointToEnd(branch.elseBlock());
  Value raw;
  if (terminal) {
    Value one = builder.create<arith::ConstantIndexOp>(branch.getLoc(), 1);
    Value zero = builder.create<arith::ConstantIndexOp>(branch.getLoc(), 0);
    Value last = builder.create<BinaryOp>(branch.getLoc(), builder.getIndexType(),
        match.loop.getUpperBound(), one, BinaryOperator::Subtract);
    Value valid = builder.create<CompareOp>(branch.getLoc(), builder.getI1Type(),
        match.loop.getUpperBound(), zero, ComparePredicate::Gt);
    raw = builder.create<GatherOp>(branch.getLoc(), element, match.scan.getResult(0),
        ValueRange{last}, valid, match.identity, builder.getDenseI64ArrayAttr({0}));
  }
  builder.create<scf::YieldOp>(branch.getLoc(),
      terminal ? ValueRange{raw} : ValueRange{});
  if (terminal && failed(replaceScanTerminals(match, branch.getResult(0))))
    return failure();
  return true;
}

FailureOr<bool> materializeScanSnapshot(ScanOp scan, func::FuncOp kernel,
                                       PhysicalProgramAnalysis &analysis) {
  auto type = dyn_cast<FragmentType>(scan.getResult(0).getType());
  if (scan.getSources().size() != 1 || scan.getIdentities().size() != 1 ||
      scan.getCaptures().size() || scan.getAxis() != 0 || scan.getReverse() ||
      !isProgramAllocationContext(scan, kernel) ||
      !type || type.getShape().size() != 1)
    return false;
  PhysicalRangeFact ranges = analysis.axisRanges(scan.getSources().front(), 0);
  FailureOr<MakeRangeOp> root = queryExactLogicalRange(ranges);
  if (failed(root))
    return false;
  SmallVector<GatherOp> readers;
  SmallVector<StoreOp> stores;
  llvm::SmallPtrSet<Operation *, 8> pointwiseUsers;
  for (Operation *user : scan.getResult(0).getUsers()) {
    if (auto store = dyn_cast<StoreOp>(user);
        store && store.getValue() == scan.getResult(0) &&
        store->getBlock() == scan->getBlock() &&
        scan->isBeforeInBlock(store) && store.getCoordinates().size() == 1 &&
        store.getSourceAxes() == ArrayRef<int64_t>{0}) {
      stores.push_back(store);
      continue;
    }
    if (auto gather = dyn_cast<GatherOp>(user)) {
      if (gather.getSource() != scan.getResult(0) ||
          gather.getCoordinates().size() != 1 ||
          gather.getSourceAxes() != ArrayRef<int64_t>{0})
        return false;
      // Keep genuine terminal reads on the scan until its complete traversal
      // supplies the final carry. Other indexed reads need the stored prefix.
      if (!scan.getInclusive() || !isLastScanRead(gather, (*root).getLogicalStop()))
        readers.push_back(gather);
      continue;
    }
    if (user->getBlock() == scan->getBlock() && scan->isBeforeInBlock(user) &&
        canPredicateValueOperation(user)) {
      pointwiseUsers.insert(user);
      continue;
    }
    return false;
  }
  if (readers.empty() && stores.empty() && pointwiseUsers.empty())
    return false;
  for (MakeRangeOp range : ranges.roots) {
    auto begin = queryLaunchExpression(range.getStart());
    auto logicalBegin = queryLaunchExpression(range.getLogicalStart());
    auto zero = [](PhysicalExprAttr value) {
      return value && value.getKind() == PhysicalExprKind::Constant &&
             value.getValue() == 0;
    };
    if (!zero(begin) || !zero(logicalBegin) || !isUnitStepRange(range))
      return false;
  }
  PhysicalExprAttr stop = queryLaunchExpression((*root).getLogicalStop());
  if (!stop)
    return false;
  auto capabilities = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  Type element = type.getElementType();
  unsigned bits = element.isIndex() ? 64 : element.getIntOrFloatBitWidth();
  if (stop.getKind() == PhysicalExprKind::Constant &&
      static_cast<__int128>(stop.getValue()) * ((bits + 31) / 32) <=
      capabilities.getRegistersPerUnit())
    return false;
  for (StoreOp store : stores) {
    auto range = store.getCoordinates().front().getDefiningOp<MakeRangeOp>();
    if (!range || !sameLogicalRange(range, *root) ||
        !analysis.lockstepRanges({range, *root}).isExact() ||
        (store.getValid() &&
         !analysis.isTailPredicate(store.getValid(), {{range, range.getLogicalStop()}})))
      return false;
  }
  for (Operation *access : ranges.accesses) {
    auto load = dyn_cast<LoadOp>(access);
    if (!load || !canReplayReadAt(load, scan))
      return false;
  }

  OpBuilder builder(scan);
  Value workspace = createProgramBuffer(
      builder, scan.getLoc(), type.getElementType(),
      builder.getArrayAttr({stop}), type.getOwner()).getResult();
  builder.setInsertionPointAfter(scan);
  Value zero = builder.create<arith::ConstantIndexOp>(scan.getLoc(), 0);
  Value one = builder.create<arith::ConstantIndexOp>(scan.getLoc(), 1);
  auto copy = builder.create<scf::ForOp>(scan.getLoc(), zero,
                                        (*root).getLogicalStop(), one);
  copy->setAttr(independentIterationAttr, builder.getUnitAttr());
  builder.setInsertionPointToStart(copy.getBody());
  Value coordinate = copy.getInductionVar();
  Value value = builder.create<GatherOp>(scan.getLoc(), element, scan.getResult(0),
                                        ValueRange{coordinate}, Value(), Value(),
                                        builder.getDenseI64ArrayAttr({0}));
  builder.create<StoreOp>(scan.getLoc(), workspace, ValueRange{coordinate}, value,
                           Value(), builder.getDenseI64ArrayAttr({0}));
  if (!pointwiseUsers.empty()) {
    builder.setInsertionPointAfter(copy);
    Value coordinate = (*root).getResult();
    auto coordinateType = cast<FragmentType>(coordinate.getType());
    auto boolean = FragmentType::get(kernel.getContext(), builder.getI1Type(),
        type.getShape(), type.getAxisMaps(), type.getValidity(), type.getOwner());
    Value stop = builder.create<BroadcastOp>(scan.getLoc(), coordinateType,
                                            (*root).getLogicalStop());
    Value valid = builder.create<CompareOp>(scan.getLoc(), boolean, coordinate,
                                            stop, ComparePredicate::Lt);
    auto fill = materializeZeroFragment(builder, scan.getLoc(), type);
    if (failed(fill))
      return failure();
    Value snapshot = builder.create<LoadOp>(scan.getLoc(), type, workspace,
        ValueRange{coordinate}, valid, *fill, builder.getDenseI64ArrayAttr({0}));
    scan.getResult(0).replaceUsesWithIf(snapshot, [&](OpOperand &use) {
      return pointwiseUsers.contains(use.getOwner());
    });
  }
  for (GatherOp reader : readers) {
    builder.setInsertionPoint(reader);
    auto load = builder.create<LoadOp>(
        reader.getLoc(), reader.getResult().getType(), workspace,
        reader.getCoordinates(), reader.getValid(), reader.getFill(), reader.getSourceAxes());
    if (Attribute origin = reader->getAttr(originAttr))
      load->setAttr(originAttr, origin);
    reader.getResult().replaceAllUsesWith(load.getResult());
    reader.erase();
  }
  // Preserve the scan's immutable snapshot before any external output write.
  // Direct vector stores use the same private prefix as indexed consumers;
  // their copy is tiled after the bounded scan has been formed.
  for (StoreOp store : stores) {
    builder.setInsertionPoint(store);
    auto fill = materializeZeroFragment(builder, store.getLoc(), type);
    if (failed(fill))
      return failure();
    Value value = builder.create<LoadOp>(
        store.getLoc(), type, workspace, store.getCoordinates(),
        store.getValid(), store.getValid() ? *fill : Value(), store.getSourceAxes());
    store.getValueMutable().assign(value);
  }
  // The copy is an independent consumer of an immutable scan. Reuse the same
  // bounded prefix traversal; the original ordered consumers stay after it.
  PhysicalProgramAnalysis currentAnalysis(kernel);
  auto match = matchScanConsumer(scan, currentAnalysis, /*closeProducers=*/false);
  if (!match)
    return scan.emitOpError("scan snapshot has no legal bounded prefix traversal");
  if (!preservesScanSourceEpoch(*match, currentAnalysis, /*collectGuards=*/false))
    return scan.emitOpError("scan snapshot traversal changes its source epoch");
  auto traversal = realizeScanConsumerMatch(kernel, *match);
  if (failed(traversal) || failed(replaceScanTerminals(*match, traversal->carry)))
    return failure();
  ParameterOp chunk = traversal->chunk;
  for (StoreOp store : stores) {
    builder.setInsertionPoint(store);
    Location location = store.getLoc();
    auto outputRange = store.getCoordinates().front().getDefiningOp<MakeRangeOp>();
    auto coordinateType = outputRange.getResult().getType();
    PhysicalExprAttr extent = queryLaunchExpression(chunk);
    auto fragment = [&](Type element) {
      return FragmentType::get(
          kernel.getContext(), element, builder.getArrayAttr({extent}),
          coordinateType.getAxisMaps(), coordinateType.getValidity(),
          coordinateType.getOwner());
    };
    Value begin = builder.create<arith::ConstantIndexOp>(location, 0);
    Value step = builder.create<arith::ConstantIndexOp>(location, 1);
    auto loop = builder.create<scf::ForOp>(
        location, begin, outputRange.getLogicalStop(), chunk);
    builder.setInsertionPointToStart(loop.getBody());
    Value coordinate = builder.create<MakeRangeOp>(
        location, fragment(builder.getIndexType()), loop.getInductionVar(),
        chunk, step, begin, outputRange.getLogicalStop(),
        outputRange.getSourceId(), outputRange.getSourceAxis(), outputRange.getDerived());
    Value end = builder.create<BroadcastOp>(
        location, fragment(builder.getIndexType()), outputRange.getLogicalStop());
    Value upper = builder.create<CompareOp>(
        location, fragment(builder.getI1Type()), coordinate, end, ComparePredicate::Lt);
    Value first = builder.create<BroadcastOp>(
        location, fragment(builder.getIndexType()), begin);
    Value lower = builder.create<CompareOp>(
        location, fragment(builder.getI1Type()), coordinate, first, ComparePredicate::Ge);
    Value valid = builder.create<BinaryOp>(
        location, fragment(builder.getI1Type()), lower, upper, BinaryOperator::LogicalAnd);
    auto outputType = fragment(element);
    auto fill = materializeZeroFragment(builder, location, outputType);
    if (failed(fill))
      return failure();
    Value value = builder.create<LoadOp>(
        location, outputType, workspace, ValueRange{coordinate}, valid, *fill,
        store.getSourceAxes());
    auto replacement = builder.create<StoreOp>(
        location, store.getResource(), ValueRange{coordinate}, value, valid,
        store.getSourceAxes());
    replacement->setAttrs(store->getAttrs());
    store.erase();
  }
  eraseDeadPhysicalValues(kernel);
  return true;
}
} // namespace

static LogicalResult realizeScanConsumerTraversalsImpl(ModuleOp module) {
  FailureOr<func::FuncOp> kernel = getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  llvm::DenseSet<Operation *> considered;
  while (true) {
    PhysicalProgramAnalysis analysis(*kernel);
    std::optional<ScanConsumerMatch> match;
    kernel->walk([&](ScanOp scan) {
      if (considered.contains(scan)) return WalkResult::advance();
      match = matchScanConsumer(scan, analysis);
      return match ? WalkResult::interrupt() : WalkResult::advance();
    });
    if (match) {
      considered.insert(match->scan);
      auto result = realizeGuardedScanConsumer(*kernel, *match, analysis);
      if (failed(result))
        return failure();
      if (*result) continue;
    }
    SmallVector<ScanOp> scans;
    kernel->walk([&](ScanOp scan) { scans.push_back(scan); });
    bool materialized = false;
    for (ScanOp scan : scans) {
      FailureOr<bool> result = materializeScanSnapshot(scan, *kernel, analysis);
      if (failed(result))
        return failure();
      if (*result) {
        // Snapshot cleanup can erase an earlier considered scan. Drop those
        // identities before the next rewrite allocates any new operations.
        llvm::DenseSet<Operation *> live;
        kernel->walk([&](ScanOp current) { live.insert(current); });
        SmallVector<Operation *> erased;
        for (Operation *previous : considered)
          if (!live.contains(previous)) erased.push_back(previous);
        for (Operation *previous : erased) considered.erase(previous);
        materialized = true;
        break;
      }
    }
    if (!materialized)
      break;
  }
  eraseDeadPhysicalValues(*kernel);
  return success();
}
LogicalResult realizeScanConsumerTraversals(ModuleOp module) {
  if (failed(realizeScanConsumerTraversalsImpl(module))) return failure();
  auto kernel = getPhysicalKernel(module);
  return failed(kernel) ? failure() : closeValueRelations(*kernel);
}

} // namespace intent::gpu
