#include "ReplayPolicy.h"
#include "ReplayInsertion.h"
#include "Intent/Dialect/GPU/Analysis/Resources.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/Analysis/IndexRelations.h"
#include "Intent/Dialect/GPU/Analysis/IntegerRanges.h"
#include "Intent/Dialect/GPU/IR/PhysicalExpressions.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Storage/Storage.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"
#include <limits>

using namespace mlir;

namespace intent::gpu {
namespace {

bool hasReplayPayload(Operation *operation) {
  // Whole-control replay is proved by replayAt, rather than by the leaf-node
  // query. Include those owners in the cost slice without granting legality.
  return operation &&
         (isa<scf::IfOp, scf::ForOp>(operation) ||
          isPhysicalReplayNode(operation, PhysicalReplayScope::ValueGraph,
                               /*allowAccesses=*/true));
}

bool expensiveOperation(Operation *operation) {
  // These operations either traverse members or perform a matrix primitive.
  // Their provider-local implementation and machine scheduling remain opaque.
  if (isa<ContractOp, ScaledContractOp, SparseContractOp,
          ReduceOp, ScanOp, scf::ForOp>(operation)) return true;
  if (auto binary = dyn_cast<BinaryOp>(operation)) {
    switch (binary.getOperatorKind()) {
    case BinaryOperator::TrueDivide:
    case BinaryOperator::FloorDivide:
    case BinaryOperator::Remainder:
    case BinaryOperator::Power:
      return true;
    default:
      return false;
    }
  }
  auto unary = dyn_cast<UnaryOp>(operation);
  if (!unary) return false;
  switch (unary.getOperatorKind()) {
  case UnaryOperator::Exp:
  case UnaryOperator::Exp2:
  case UnaryOperator::Log:
  case UnaryOperator::Log1p:
  case UnaryOperator::Lgamma:
  case UnaryOperator::Sin:
  case UnaryOperator::Asin:
  case UnaryOperator::Cos:
  case UnaryOperator::Erf:
  case UnaryOperator::Erfc:
  case UnaryOperator::I0:
  case UnaryOperator::Rsqrt:
  case UnaryOperator::Sigmoid:
  case UnaryOperator::Tanh:
  case UnaryOperator::Sqrt:
    return true;
  case UnaryOperator::Negate:
  case UnaryOperator::Not:
  case UnaryOperator::Floor:
  case UnaryOperator::Abs:
    return false;
  }
  llvm_unreachable("unknown unary replay cost");
}

bool expensivePayload(Operation *operation) {
  // A cloned region carries its computation with it. Looking only at the outer
  // result opcode loses the cost of math inside an if or a structured payload.
  return operation->walk([](Operation *nested) {
    return expensiveOperation(nested) ? WalkResult::interrupt()
                                      : WalkResult::advance();
  }).wasInterrupted();
}

// Preserve the existing multi-output/reduction sharing case even when its
// shared arithmetic is individually cheap. Do not count stores of the reduced
// result as independent writes of the original fragment.
bool sharedReductionOutputs(Value value, unsigned minimumStores = 2,
                            std::optional<PhysicalSourceAxis> source = {},
                            int64_t dimension = 0) {
  SmallVector<Value> pending{value};
  llvm::DenseSet<Operation *> visited, stores;
  bool reduction = false;
  while (!pending.empty()) {
    Value current = pending.pop_back_val();
    for (Operation *user : current.getUsers()) {
      if (auto store = dyn_cast<StoreOp>(user)) {
        bool selected = !source || llvm::any_of(store.getCoordinates(), [&](Value coordinate) {
          return queryFragmentAxis(coordinate.getType(), *source, dimension).isExact();
        });
        if (store.getValue() == current && selected) stores.insert(user);
        continue;
      }
      if (auto reduce = dyn_cast<ReduceOp>(user)) {
        auto axis = source ? queryFragmentAxis(current.getType(), *source, dimension)
                           : PhysicalAxisProjection{};
        reduction |= llvm::is_contained(reduce.getSources(), current) &&
            (!source || (axis.isExact() && llvm::is_contained(
                reduce.getAxes(), static_cast<int64_t>(axis.fragmentAxis))));
        continue;
      }
      if (auto gather = dyn_cast<GatherOp>(user); gather && gather.getSource() == current) {
        if (visited.insert(user).second) pending.push_back(gather.getResult());
        continue;
      }
      if (user->getNumRegions() || !visited.insert(user).second ||
          !isPhysicalReplayNode(user, PhysicalReplayScope::ValueGraph,
                                /*allowAccesses=*/false)) continue;
      llvm::append_range(pending, user->getResults());
    }
  }
  return reduction && stores.size() >= minimumStores;
}

bool inactiveBeyondRange(Value predicate,
                         ArrayRef<std::pair<MakeRangeOp, Value>> ranges,
                         PhysicalProgramAnalysis &analysis) {
  if (!predicate) return false;
  auto constant = dyn_cast_or_null<IntegerAttr>(
      UniformValueAnalysis(describeUniformValue).evaluate(predicate));
  if (constant && constant.getType().isInteger(1)) return constant.getValue().isZero();
  Operation *producer = predicate.getDefiningOp();
  if (isa_and_nonnull<BroadcastOp, SplatOp, ReshapeOp, TransposeOp>(producer))
    return inactiveBeyondRange(producer->getOperand(0), ranges, analysis);
  if (auto binary = dyn_cast_or_null<BinaryOp>(producer)) {
    bool lhs = inactiveBeyondRange(binary.getLhs(), ranges, analysis);
    bool rhs = inactiveBeyondRange(binary.getRhs(), ranges, analysis);
    switch (binary.getOperatorKind()) {
    case BinaryOperator::LogicalAnd:
    case BinaryOperator::BitwiseAnd:
      return lhs || rhs;
    case BinaryOperator::LogicalOr:
    case BinaryOperator::BitwiseOr:
      return lhs && rhs;
    default:
      return false;
    }
  }
  auto compare = dyn_cast_or_null<CompareOp>(producer);
  return compare && compare.getPredicate() == ComparePredicate::Lt &&
         analysis.isTailPredicate(predicate, ranges);
}

Attribute tailValue(Value value, unsigned axis, PhysicalProgramAnalysis &analysis) {
  auto type = cast<FragmentType>(value.getType());
  auto relation = cast<AxisMapAttr>(type.getAxisMaps()[axis]);
  auto ranges = analysis.axisRanges(value, axis);
  if (!ranges.isExact() || !ranges.blockers.empty() ||
      failed(queryExactLogicalRange(ranges)) || !analysis.lockstepRanges(ranges.roots).isExact())
    return {};
  SmallVector<std::pair<MakeRangeOp, Value>> tails;
  for (MakeRangeOp range : ranges.roots) tails.emplace_back(range, range.getLogicalStop());
  UniformValueAnalysis values(describeUniformValue);
  UniformBindings bindings;
  SmallVector<Value> pending{value};
  llvm::DenseSet<Value> visited;
  while (!pending.empty()) {
    Value current = pending.pop_back_val();
    if (!visited.insert(current).second) continue;
    auto fragment = dyn_cast<FragmentType>(current.getType());
    Type element = fragment ? fragment.getElementType() : current.getType();
    if (element.isInteger(1) && inactiveBeyondRange(current, tails, analysis)) {
      bindings[current] = IntegerAttr::get(IntegerType::get(value.getContext(), 1), 0);
      continue;
    }
    Operation *producer = current.getDefiningOp();
    if (producer && !producer->getNumRegions() && hasReplayPayload(producer))
      llvm::append_range(pending, producer->getOperands());
  }
  // A consumer may already have neutralized the pure value after its reads.
  // Use that actual predicate before requiring every upstream access to be
  // independently inactive (e.g. a completed contraction is an SSA snapshot).
  if (Attribute constant = values.evaluate(value, bindings)) return constant;
  for (Operation *operation : ranges.accesses) {
    Value result, fill, valid;
    if (auto load = dyn_cast<LoadOp>(operation)) {
      result = load.getResult();
      fill = load.getFill();
      valid = load.getValid();
    } else if (auto gather = dyn_cast<GatherOp>(operation)) {
      result = gather.getResult();
      fill = gather.getFill();
      valid = gather.getValid();
    } else {
      return {};
    }
    auto projection = queryFragmentAxis(result.getType(), sourceAxisIdentity(relation),
                                         relation.getDimensionId());
    if (!projection.isExact()) continue;
    // Axis provenance alone does not imply inactivity: clamped or wrapped
    // coordinates can still perform real reads beyond another range's tail.
    if (!fill || !inactiveBeyondRange(valid, tails, analysis)) return {};
    Attribute constant = values.evaluate(fill);
    if (!constant) return {};
    bindings[valid] = IntegerAttr::get(IntegerType::get(value.getContext(), 1), 0);
    bindings[result] = constant;
  }
  return values.evaluate(value, bindings);
}

} // namespace

FailureOr<MakeRangeOp> completeSnapshotRange(
    Value value, unsigned axis, PhysicalProgramAnalysis &analysis) {
  auto type = dyn_cast<FragmentType>(value.getType());
  if (!type || axis >= type.getShape().size()) return failure();
  auto realization = analysis.axisRealization(value, axis);
  auto ranges = analysis.axisRanges(value, axis);
  auto authority = queryExactLogicalRange(ranges);
  auto capacity = constantPhysicalExpression(cast<PhysicalExprAttr>(type.getShape()[axis]));
  if (!realization.isExact() || !realization.physicalized ||
      realization.constructionScalarSeed || failed(authority) ||
      !isUnitStepRange(*authority) || !analysis.lockstepRanges(ranges.roots).isExact() ||
      !capacity || *capacity <= 0 ||
      constantLogicalRangeCardinality(*authority) != capacity ||
      !samePhysicalScalarExpression((*authority).getStart(), (*authority).getLogicalStart()) ||
      !IndexRelations().atMost((*authority).getStart(), (*authority).getLogicalStop()))
    return failure();
  return *authority;
}

ReplayPolicy::ReplayPolicy(func::FuncOp kernel, ValueRange roots,
                           ArrayRef<Operation *> replacedConsumers,
                           llvm::function_ref<bool(Value)> rebuilt)
    : kernel(kernel), consumers(replacedConsumers.begin(), replacedConsumers.end()) {
  SmallVector<Value> pending(roots.begin(), roots.end());
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    // Dominating scalar results are captured by the existing replay helpers.
    // In particular a scalar reduction is still live after its users' suffixes
    // move; its entire producer DAG must not be counted as deleted work.
    if (!isa<FragmentType, RecordType>(value.getType()) ||
        (rebuilt && !rebuilt(value))) continue;
    Operation *producer = value.getDefiningOp();
    if (!hasReplayPayload(producer) ||
        !slice.insert(producer).second) continue;
    llvm::append_range(pending, producer->getOperands());
    llvm::SetVector<Value> captures;
    for (Region &region : producer->getRegions())
      getUsedValuesDefinedAbove(region, captures);
    llvm::append_range(pending, captures);
  }
}

bool ReplayPolicy::survives(Operation *operation,
                            llvm::DenseSet<Operation *> &active) const {
  if (!active.insert(operation).second) return true;
  for (Operation *user : operation->getUsers()) {
    bool removed = llvm::any_of(consumers, [&](Operation *consumer) {
      return consumer == user || consumer->isAncestor(user);
    });
    if (removed) continue;
    Operation *owner = user;
    while (owner && !slice.contains(owner)) owner = owner->getParentOp();
    if (!owner || survives(owner, active)) {
      active.erase(operation);
      return true;
    }
  }
  active.erase(operation);
  return false;
}

bool ReplayPolicy::duplicatesExpensiveWork(Value root,
                                          const IRMapping *bindings) const {
  SmallVector<Value> pending{root};
  llvm::DenseSet<Value> visited;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    if (!visited.insert(value).second ||
        (bindings && bindings->lookupOrNull(value)) ||
        !isa<FragmentType, RecordType>(value.getType())) continue;
    Operation *producer = value.getDefiningOp();
    if (!producer || !slice.contains(producer)) continue;
    // Replay projects a known record field directly. Other fields do not become
    // new work merely because they share the same aggregate construction.
    if (auto extract = dyn_cast<ExtractOp>(producer)) {
      if (auto record = extract.getRecord().getDefiningOp<MakeRecordOp>()) {
        pending.push_back(record.getFields()[extract.getField()]);
        continue;
      }
    }
    if (expensivePayload(producer)) {
      llvm::DenseSet<Operation *> active;
      if (survives(producer, active)) return true;
    }
    llvm::append_range(pending, producer->getOperands());
    llvm::SetVector<Value> captures;
    for (Region &region : producer->getRegions())
      getUsedValuesDefinedAbove(region, captures);
    llvm::append_range(pending, captures);
  }
  return false;
}

bool ReplayPolicy::removesProducer(Value value) const {
  Operation *producer = value.getDefiningOp();
  if (!producer || !slice.contains(producer)) return false;
  llvm::DenseSet<Operation *> active;
  return !survives(producer, active);
}

bool ReplayPolicy::retains(Value value, Operation *anchor,
                           const IRMapping *bindings) const {
  auto fragment = dyn_cast<FragmentType>(value.getType());
  Operation *producer = value.getDefiningOp();
  if (!kernel || !fragment || !producer || !anchor ||
      !DominanceInfo(kernel).dominates(value, anchor)) return false;
  if (!duplicatesExpensiveWork(value, bindings)) {
    llvm::DenseSet<Operation *> active;
    if (!sharedReductionOutputs(value) || !survives(producer, active)) return false;
  }
  PhysicalProgramAnalysis analysis(kernel);
  for (unsigned axis = 0; axis < fragment.getShape().size(); ++axis) {
    auto realization = analysis.axisRealization(value, axis);
    if (!realization.isExact() || !realization.physicalized ||
        realization.constructionScalarSeed) return false;
  }
  auto capabilities = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  if (!capabilities) return false;
  auto footprint = minimumFragmentRegisterFootprint(
      kernel, value, capabilities.getRegistersPerUnit(),
      FragmentFootprintScope::PhysicalShape);
  // This is a profitability opportunity for a fitting candidate, not a resource
  // legality guarantee for every configuration. Existing resource checks remain.
  return footprint && *footprint < capabilities.getRegistersPerUnit();
}

bool ReplayPolicy::preservesSharedTraversal(Value root, PhysicalSourceAxis source,
                                            int64_t dimension) const {
  SmallVector<Value> pending{root};
  llvm::DenseSet<Value> visited;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    if (!visited.insert(value).second) continue;
    auto projection = queryFragmentAxis(value.getType(), source, dimension);
    if (projection.isExact() && sharedReductionOutputs(
            value, duplicatesExpensiveWork(value) ? 1 : 2, source, dimension)) return true;
    Operation *producer = value.getDefiningOp();
    if (!hasReplayPayload(producer)) continue;
    llvm::append_range(pending, producer->getOperands());
  }
  return false;
}

FailureOr<Value> ReplayPolicy::retainSlice(
    OpBuilder &builder, Value value, unsigned axis, PhysicalExprAttr extent,
    Value coordinates, Operation *anchor, AxisMapAttr resultMapping,
    const IRMapping *bindings) const {
  if (!retains(value, anchor, bindings)) return Value();
  return materializeSnapshotSlice(builder, value, axis, extent, coordinates,
                                  anchor, resultMapping);
}

FailureOr<Value> materializeSnapshotSlice(
    OpBuilder &builder, Value value, unsigned axis, PhysicalExprAttr extent,
    Value coordinates, Operation *anchor, AxisMapAttr resultMapping) {
  auto source = dyn_cast<FragmentType>(value.getType());
  auto kernel = anchor ? anchor->getParentOfType<func::FuncOp>() : func::FuncOp{};
  if (!source || !kernel) return Value();
  auto coordinate = dyn_cast<FragmentType>(coordinates.getType());
  if (!coordinate || !coordinate.getElementType().isIndex() ||
      axis >= source.getShape().size()) return Value();
  DominanceInfo dominance(kernel);
  if (!availableAtInsertionPoint(value, builder, dominance) ||
      !availableAtInsertionPoint(coordinates, builder, dominance))
    return Value();
  PhysicalProgramAnalysis analysis(kernel);
  for (unsigned axis = 0; axis < source.getShape().size(); ++axis) {
    auto realization = analysis.axisRealization(value, axis);
    if (!realization.isExact() || !realization.physicalized ||
        realization.constructionScalarSeed) return Value();
  }
  auto ranges = analysis.axisRanges(value, axis);
  Attribute tail = tailValue(value, axis, analysis);
  if (!tail) return Value();
  auto authority = queryExactLogicalRange(ranges);
  if (failed(authority) || !isUnitStepRange(*authority) ||
      !analysis.lockstepRanges(ranges.roots).isExact() ||
      !availableAtInsertionPoint((*authority).getStart(), builder, dominance) ||
      !availableAtInsertionPoint((*authority).getExtent(), builder, dominance) ||
      !availableAtInsertionPoint((*authority).getLogicalStop(), builder, dominance)) return Value();
  IndexRelations relations;
  auto capacity = cast<PhysicalExprAttr>(source.getShape()[axis]);
  bool complete = succeeded(completeSnapshotRange(value, axis, analysis));
  if (!complete &&
      (!relations.same((*authority).getStart(), (*authority).getLogicalStart()) ||
       !relations.nonnegative((*authority).getStart()) ||
       !relations.atMost((*authority).getStart(), capacity) ||
       !relations.atMost((*authority).getLogicalStop(), capacity))) return Value();
  auto requested = coordinates.getDefiningOp<MakeRangeOp>();
  if (!requested || !isUnitStepRange(requested)) return Value();
  auto atMostStop = [&](Value bound) {
    return complete ? relations.atMost(bound, (*authority).getLogicalStop())
                    : relations.atMost(bound, capacity);
  };
  bool boundedStart = relations.atMost((*authority).getStart(), requested.getStart()) &&
                      atMostStop(requested.getStart());
  for (Operation *parent = requested->getParentOp(); !boundedStart && parent;
       parent = parent->getParentOp()) {
    auto loop = dyn_cast<scf::ForOp>(parent);
    boundedStart = loop && relations.same(requested.getStart(), loop.getInductionVar()) &&
        relations.atMost((*authority).getStart(), loop.getLowerBound()) &&
        atMostStop(loop.getUpperBound());
  }
  auto capacityBounds = queryPositiveExtentBounds(capacity, kernel);
  auto widthBounds = queryPositiveExtentBounds(
      cast<PhysicalExprAttr>(requested.getResult().getType().getShape()[0]), kernel);
  // Every requested ordinal is nonnegative relative to the saved start. A
  // complete saved domain plus a representable positive lane progression means
  // the only missing lanes are the already-proved logical tail, never a gap in
  // an earlier physical tile or a wrapped/negative prefix.
  if (!boundedStart || !capacityBounds || !widthBounds ||
      capacityBounds->second > std::numeric_limits<int64_t>::max() - widthBounds->second)
    return Value();
  if (complete) {
    auto stopBounds = queryIntegerRange((*authority).getLogicalStop());
    if (!stopBounds ||
        static_cast<__int128>(stopBounds->smax().getSExtValue()) + widthBounds->second - 1 >
            std::numeric_limits<int64_t>::max()) return Value();
  }
  SmallVector<Attribute> shape(source.getShape().begin(), source.getShape().end());
  SmallVector<Attribute> axes(source.getAxisMaps().begin(), source.getAxisMaps().end());
  shape[axis] = extent;
  if (resultMapping)
    axes[axis] = AxisMapAttr::get(source.getContext(), resultMapping.getSourceId(),
        resultMapping.getSourceAxis(), resultMapping.getDimensionId(), axis,
        resultMapping.getDerived());
  auto target = FragmentType::get(source.getContext(), builder.getIndexType(),
      builder.getArrayAttr(shape), builder.getArrayAttr(axes), source.getValidity(), source.getOwner());
  if (!queryBroadcastProjection(coordinate, target).isExact()) return Value();
  auto result = materializeRetainedSlice(builder, anchor->getLoc(), value, axis, extent,
                                         coordinates, anchor, resultMapping);
  if (failed(result)) return failure();
  if (tail != uniformZero(source.getElementType())) {
    auto gather = (*result).getDefiningOp<GatherOp>();
    OpBuilder fillBuilder(gather);
    auto scalar = materializeScalarConstant(fillBuilder, gather.getLoc(), tail,
                                            source.getElementType());
    if (failed(scalar)) return failure();
    Value fill = fillBuilder.create<SplatOp>(gather.getLoc(), gather.getResult().getType(), *scalar);
    gather.getFillMutable().assign(fill);
  }
  return result;
}

LogicalResult ReplayPolicy::bindSlices(
    OpBuilder &builder, Value root, PhysicalSourceAxis source, int64_t dimension,
    PhysicalExprAttr extent, Value coordinates, Operation *anchor,
    IRMapping &mapping, AxisMapAttr resultMapping,
    ArrayRef<int64_t> selectedDimensions) const {
  SmallVector<Value> pending{root};
  llvm::DenseSet<Value> visited;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    if (!visited.insert(value).second || mapping.lookupOrNull(value)) continue;
    auto fragment = dyn_cast<FragmentType>(value.getType());
    auto projection = fragment ? queryFragmentAxis(fragment, source, dimension)
                               : PhysicalAxisProjection{};
    unsigned selectedAxes = 0;
    if (fragment)
      for (const auto &candidate : queryFragmentAxes(fragment, source))
        if (selectedDimensions.empty() ? candidate.dimensionId == dimension
                                       : llvm::is_contained(selectedDimensions, candidate.dimensionId))
          ++selectedAxes;
    if (projection.isExact() && selectedAxes == 1) {
      auto retained = retainSlice(builder, value, projection.fragmentAxis, extent,
                                  coordinates, anchor, resultMapping, &mapping);
      if (failed(retained)) return failure();
      if (*retained) {
        mapping.map(value, *retained);
        continue;
      }
    }
    Operation *producer = value.getDefiningOp();
    if (!producer || producer->getNumRegions() || !slice.contains(producer)) continue;
    llvm::append_range(pending, producer->getOperands());
  }
  return success();
}

} // namespace intent::gpu
