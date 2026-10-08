#include "Intent/Dialect/CPU/Analysis/RegionPredicates.h"
#include "Intent/Analysis/IntegerRanges.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Intent/Dialect/CPU/Analysis/UniformValues.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/APInt.h"

using namespace mlir;
namespace intent::cpu {
namespace {

std::optional<SmallVector<Value>> fullAliases(Value value) {
  value = canonicalUniformMemory(value);
  StorageAnalysis storage(value.getParentRegion()->getParentOfType<func::FuncOp>());
  auto aliases = storage.aliases(value);
  if (!aliases.complete || llvm::any_of(aliases.values, [&](Value alias) {
        return canonicalUniformMemory(alias) != value;
      })) return std::nullopt;
  return aliases.values;
}

Value stripIndexWidthCasts(Value value) {
  auto indexWidth = [](Type type) { return type.isIndex() || type.isInteger(64); };
  while (auto cast = value.getDefiningOp<arith::IndexCastOp>()) {
    if (!indexWidth(cast.getIn().getType()) || !indexWidth(cast.getType())) break;
    value = cast.getIn();
  }
  return value;
}

using SignedBounds = std::pair<llvm::APInt, llvm::APInt>;

llvm::APInt wideInteger(int64_t value) { return llvm::APInt(65, value, true); }

std::optional<SignedBounds>
coordinateBounds(std::optional<ConstantIntRanges> range) {
  if (!range || range->smin().getBitWidth() > 64) return std::nullopt;
  return SignedBounds{range->smin().sext(65), range->smax().sext(65)};
}

bool shiftedSequenceDoesNotWrap(memref::AllocOp allocation, Value origin,
                                 IntegerRangeAnalysis &ranges) {
  auto begin = coordinateBounds(ranges.range(origin));
  if (!begin) return false;
  auto minimum = llvm::APInt::getSignedMinValue(64).sext(65);
  auto maximum = llvm::APInt::getSignedMaxValue(64).sext(65);
  auto type = allocation.getType();
  auto constant = type.isDynamicDim(0) ? getConstantIntValue(allocation.getDynamicSizes()[0])
                                     : std::optional<int64_t>(type.getDimSize(0));
  if (constant)
    return *constant >= 0 && (begin->second + wideInteger(*constant)).sle(maximum);
  auto extent = allocation.getDynamicSizes()[0].getDefiningOp<arith::MaxSIOp>();
  if (!extent) return false;
  Value difference;
  if (getConstantIntValue(extent.getRhs()) == 0) difference = extent.getLhs();
  else if (getConstantIntValue(extent.getLhs()) == 0) difference = extent.getRhs();
  auto subtraction = difference ? difference.getDefiningOp<arith::SubIOp>() : arith::SubIOp();
  if (!subtraction || stripIndexWidthCasts(subtraction.getRhs()) != stripIndexWidthCasts(origin))
    return false;
  auto end = coordinateBounds(ranges.range(subtraction.getLhs()));
  if (!end) return false;
  // A non-wrapping max(end - begin, 0) keeps every populated coordinate
  // between its signed endpoints, even when the common origin is negative.
  return (end->first - begin->second).sge(minimum) &&
         (end->second - begin->first).sle(maximum);
}

Operation *lastWriter(Value value, Operation *before = nullptr) {
  auto aliases = fullAliases(value);
  if (!aliases) return nullptr;
  StorageAnalysis storage(value.getParentRegion()->getParentOfType<func::FuncOp>());
  Value observed = canonicalUniformMemory(value);
  Operation *last = nullptr;
  for (Value alias : *aliases) for (Operation *user : alias.getUsers()) {
    if (user == before) continue;
    auto effects = storage.effects(user);
    if (!effects.complete || effects.ordered) return nullptr;
    for (const StorageEffect &entry : effects.entries) {
      if (!isa<MemoryEffects::Write>(entry.effect.getEffect())) continue;
      Value target = entry.effect.getValue();
      if (target && storage.disjoint(value, target)) continue;
      if (!target || canonicalUniformMemory(target) != observed) return nullptr;
      Operation *writer = entry.operation;
      if (before && writer->getBlock() != before->getBlock()) return nullptr;
      if (before && !writer->isBeforeInBlock(before)) continue;
      if (last && writer->getBlock() != last->getBlock()) return nullptr;
      if (!last || last->isBeforeInBlock(writer)) last = writer;
    }
  }
  return last;
}
struct ComputationResult {
  linalg::GenericOp operation;
  unsigned output;
};
std::optional<ComputationResult> producer(Value value) {
  llvm::SmallDenseSet<Value> seen;
  while (seen.insert(value).second) {
    value = canonicalUniformMemory(value);
    Operation *writer = lastWriter(value);
    if (auto copy = dyn_cast_or_null<memref::CopyOp>(writer)) value = copy.getSource();
    else {
      auto generic = dyn_cast_or_null<linalg::GenericOp>(writer);
      if (!generic) return std::nullopt;
      for (auto [index, output] : llvm::enumerate(generic.getOutputs()))
        if (canonicalUniformMemory(output) == value)
          return ComputationResult{generic, static_cast<unsigned>(index)};
      return std::nullopt;
    }
  }
  return std::nullopt;
}
bool initializedFalse(Value value, Operation *before) {
  auto fill = dyn_cast_or_null<linalg::FillOp>(lastWriter(value, before));
  return fill && uniformBoolean(UniformValueAnalysis(describeScalarValue).evaluate(fill.getInputs()[0])) == false;
}

struct CoordinatePredicate {
  Value value;
  unsigned source, capture, sourceAxis;
  UniformPredicate comparison;
};

std::optional<CoordinatePredicate> coordinatePredicate(
    RegionOpInterface program, linalg::GenericOp generic, arith::CmpIOp compare) {
  Block &helper = program.getSummarize().front();
  auto operand = [&](Value value) -> std::optional<std::pair<BlockArgument, unsigned>> {
    auto argument = dyn_cast<BlockArgument>(stripIndexWidthCasts(value));
    if (!argument || argument.getOwner() != &generic.getRegion().front() ||
        argument.getArgNumber() >= generic.getNumDpsInputs()) return std::nullopt;
    Value input = canonicalUniformMemory(generic.getInputs()[argument.getArgNumber()]);
    AffineMap coordinates = generic.getIndexingMapsArray()[argument.getArgNumber()];
    Operation *consumer = generic;
    llvm::SmallDenseSet<Value> seen;
    while (!isa<BlockArgument>(input)) {
      if (!seen.insert(input).second) return std::nullopt;
      auto forward = producer(input);
      if (!forward || forward->operation.getNumReductionLoops() ||
          forward->operation->getBlock() != consumer->getBlock() ||
          !forward->operation->isBeforeInBlock(consumer)) return std::nullopt;
      auto maps = forward->operation.getIndexingMapsArray();
      AffineMap outputMap = maps[forward->operation.getNumDpsInputs() + forward->output];
      if (!outputMap.isPermutation()) return std::nullopt;
      Block &body = forward->operation.getRegion().front();
      auto yielded = dyn_cast<BlockArgument>(
          stripIndexWidthCasts(body.getTerminator()->getOperand(forward->output)));
      if (!yielded || yielded.getOwner() != &body ||
          yielded.getArgNumber() >= forward->operation.getNumDpsInputs() ||
          llvm::any_of(body.without_terminator(), [](Operation &op) {
            return op.getNumRegions() || !isMemoryEffectFree(&op);
          })) return std::nullopt;
      coordinates = maps[yielded.getArgNumber()].compose(inversePermutation(outputMap)).compose(coordinates);
      input = canonicalUniformMemory(forward->operation.getInputs()[yielded.getArgNumber()]);
      consumer = forward->operation;
    }
    auto formal = dyn_cast<BlockArgument>(input);
    auto type = dyn_cast<MemRefType>(input.getType());
    if (!formal || formal.getOwner() != &helper || !type || type.getRank() != 1 ||
        (!type.getElementType().isIndex() && !type.getElementType().isInteger(64))) return std::nullopt;
    auto axis = dyn_cast<AffineDimExpr>(coordinates.getResult(0));
    if (!axis) return std::nullopt;
    return std::make_pair(formal, axis.getPosition());
  };
  auto left = operand(compare.getLhs()), right = operand(compare.getRhs());
  if (!left || !right || left->second == right->second) return std::nullopt;
  auto sources = program.getRegionArguments(program.getSummarize(), RegionArgumentKind::Sources);
  auto captures = program.getRegionArguments(program.getSummarize(), RegionArgumentKind::Captures);
  bool reverse = !llvm::is_contained(sources, left->first);
  auto source = reverse ? *right : *left, capture = reverse ? *left : *right;
  if (!llvm::is_contained(sources, source.first) ||
      !llvm::is_contained(captures, capture.first)) return std::nullopt;
  UniformPredicate predicate;
  switch (compare.getPredicate()) {
  case arith::CmpIPredicate::slt: predicate = reverse ? UniformPredicate::Greater : UniformPredicate::Less; break;
  case arith::CmpIPredicate::sle: predicate = reverse ? UniformPredicate::GreaterEqual : UniformPredicate::LessEqual; break;
  case arith::CmpIPredicate::sgt: predicate = reverse ? UniformPredicate::Less : UniformPredicate::Greater; break;
  case arith::CmpIPredicate::sge: predicate = reverse ? UniformPredicate::LessEqual : UniformPredicate::GreaterEqual; break;
  default: return std::nullopt;
  }
  return CoordinatePredicate{compare,
      static_cast<unsigned>(llvm::find(sources, source.first) - sources.begin()),
      static_cast<unsigned>(llvm::find(captures, capture.first) - captures.begin()),
      source.second, predicate};
}

std::optional<std::pair<unsigned, Operation *>> validityField(
    RegionOpInterface program, ArrayRef<CoordinatePredicate> predicates) {
  UniformValueAnalysis values(describeScalarValue);
  auto summaryOutputs = program.getRegionArguments(program.getSummarize(), RegionArgumentKind::Destinations);
  auto combinedOutputs = program.getRegionArguments(program.getCombine(), RegionArgumentKind::Destinations);
  auto left = program.getRegionArguments(program.getCombine(), RegionArgumentKind::LeftSummary);
  auto right = program.getRegionArguments(program.getCombine(), RegionArgumentKind::RightSummary);
  for (auto [field, slot] : llvm::enumerate(summaryOutputs)) {
    auto type = cast<MemRefType>(slot.getType());
    if (!type.getElementType().isInteger(1)) continue;
    auto produced = producer(slot);
    if (!produced) continue;
    auto reduction = produced->operation;
    if (reduction.getNumDpsInits() != 1 || reduction.getNumReductionLoops() != 1 ||
        !initializedFalse(reduction.getOutputs()[0], reduction)) continue;
    Block &body = reduction.getRegion().front();
    Value membership;
    for (const CoordinatePredicate &predicate : predicates) {
      auto owner = predicate.value.getDefiningOp()->getParentOfType<linalg::GenericOp>();
      if (owner == reduction) {
        if (reduction.getIteratorTypesArray()[predicate.sourceAxis] == utils::IteratorType::reduction)
          membership = predicate.value;
      } else if (!owner.getNumReductionLoops()) {
        auto maps = owner.getIndexingMapsArray();
        for (auto [number, yielded] : llvm::enumerate(owner.getRegion().front().getTerminator()->getOperands())) {
          if (yielded != predicate.value) continue;
          AffineMap outputMap = maps[owner.getNumDpsInputs() + number];
          for (auto [axis, expression] : llvm::enumerate(outputMap.getResults())) {
            auto dimension = dyn_cast<AffineDimExpr>(expression);
            if (!dimension || dimension.getPosition() != predicate.sourceAxis) continue;
            for (auto [input, memory] : llvm::enumerate(reduction.getInputs())) {
              if (canonicalUniformMemory(memory) != canonicalUniformMemory(owner.getOutputs()[number])) continue;
              auto coordinate = dyn_cast<AffineDimExpr>(reduction.getIndexingMapsArray()[input].getResult(axis));
              if (coordinate && reduction.getIteratorTypesArray()[coordinate.getPosition()] == utils::IteratorType::reduction)
                membership = body.getArgument(input);
            }
          }
        }
      }
      if (membership && isBooleanUnion(values, body.getTerminator()->getOperand(0),
              membership, body.getArguments().back())) break;
      membership = {};
    }
    if (!membership) continue;
    Value identity = canonicalUniformMemory(program.getIdentities()[field]);
    if (!initializedFalse(identity, program.getOperation())) continue;
    auto combinedResult = producer(combinedOutputs[field]);
    if (!combinedResult) continue;
    auto combined = combinedResult->operation;
    if (combined.getNumReductionLoops() ||
        !combined.getIndexingMapsArray()[combined.getNumDpsInputs() + combinedResult->output].isIdentity()) continue;
    Value lhs, rhs;
    for (auto [index, input] : llvm::enumerate(combined.getInputs())) {
      if (!combined.getIndexingMapsArray()[index].isIdentity()) continue;
      if (canonicalUniformMemory(input) == left[field]) lhs = combined.getRegion().front().getArgument(index);
      if (canonicalUniformMemory(input) == right[field]) rhs = combined.getRegion().front().getArgument(index);
    }
    Value result = combined.getRegion().front().getTerminator()->getOperand(combinedResult->output);
    if (lhs && rhs && isBooleanUnion(values, result, lhs, rhs) && hasTrueStateInvariant(values, result, lhs))
      return std::make_pair(static_cast<unsigned>(field), reduction.getOperation());
  }
  return std::nullopt;
}

bool summaryIsIdentityWhenFalse(RegionOpInterface program, ValueRange predicates) {
  StorageAnalysis storage(program.getOperation()->getParentOfType<func::FuncOp>());
  for (Region &region : program.getOperation()->getRegions()) {
    auto destinations = program.getRegionArguments(region, RegionArgumentKind::Destinations);
    for (Operation &operation : region.front().without_terminator()) {
      auto effects = storage.effects(&operation);
      if (!effects.complete || effects.ordered) return false;
      for (const StorageEffect &entry : effects.entries) {
        const auto &effect = entry.effect;
        if (isa<MemoryEffects::Read, MemoryEffects::Allocate>(effect.getEffect())) continue;
        if (!effect.getValue()) return false;
        Value root = storage.uniqueOrigin(effect.getValue());
        if (!root) return false;
        if (auto argument = dyn_cast<BlockArgument>(root)) {
          if (!llvm::is_contained(destinations, argument) ||
              !isa<MemoryEffects::Write>(effect.getEffect())) return false;
        } else {
          auto allocation = root.getDefiningOp();
          if (!isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(allocation) ||
              !region.isAncestor(allocation->getParentRegion())) return false;
        }
      }
    }
  }
  UniformBindings scalars;
  for (Value predicate : predicates)
    scalars[predicate] = IntegerAttr::get(IntegerType::get(predicate.getContext(), 1), 0);
  UniformMemoryAnalysis outer(storage), summary(storage, std::move(scalars));
  for (Operation &operation : *program.getOperation()->getBlock()) {
    if (&operation == program.getOperation()) break;
    outer.visit(&operation);
  }
  Block &helper = program.getSummarize().front();
  auto schema = program.getRegionSchema(program.getSummarize());
  if (failed(schema)) return false;
  for (const auto &relation : *schema)
    if (relation.kind != RegionArgumentKind::Destinations)
      summary.write(relation.argument, outer.read(relation.prototype));
  for (Operation &operation : helper.without_terminator()) summary.visit(&operation);
  for (auto [result, identity] : llvm::zip(
           program.getRegionArguments(program.getSummarize(), RegionArgumentKind::Destinations),
           program.getIdentities()))
    if (!equalUniformConstants(summary.read(result), outer.read(identity))) return false;
  return true;
}

}

std::optional<CoordinateSequence> coordinateSequence(Value memory, Operation *at) {
  IntegerRangeAnalysis ranges;
  CoordinateSequence result{{}, true, true, {}};
  Value selectedMemory = memory;
  while (true) {
    if (auto cast = memory.getDefiningOp<memref::CastOp>()) { memory = cast.getSource(); continue; }
    if (auto view = memory.getDefiningOp<memref::SubViewOp>()) {
      if (view.getSourceType().getRank() != 1 || view.getType().getRank() != 1 ||
          getConstantIntValue(view.getMixedStrides()[0]) != 1) return std::nullopt;
      auto offset = view.getMixedOffsets()[0];
      result.offsets.push_back(offset);
      result.startsAtOrigin &= getConstantIntValue(offset) == 0;
      result.nonnegativeOffsets &= isa<Attribute>(offset) ? cast<IntegerAttr>(cast<Attribute>(offset)).getInt() >= 0
                                                          : ranges.isNonNegative(cast<Value>(offset));
      memory = view.getSource();
      continue;
    }
    if (auto argument = dyn_cast<BlockArgument>(memory))
      if (auto tasks = dyn_cast<TasksOp>(argument.getOwner()->getParentOp()); tasks && argument.getArgNumber()) {
        memory = tasks.getCaptures()[argument.getArgNumber() - 1];
        continue;
      }
    break;
  }
  auto allocation = memory.getDefiningOp<memref::AllocOp>();
  auto type = dyn_cast<MemRefType>(memory.getType());
  if (!allocation || !type || type.getRank() != 1 ||
      (!type.getElementType().isIndex() && !type.getElementType().isInteger(64))) return std::nullopt;
  linalg::GenericOp writer;
  SmallVector<Value> aliases{memory};
  llvm::SmallDenseSet<Value> visited;
  for (unsigned i = 0; i < aliases.size(); ++i) {
    Value alias = aliases[i];
    if (!visited.insert(alias).second) continue;
    for (OpOperand &use : alias.getUses()) {
      Operation *user = use.getOwner();
      if (auto generic = dyn_cast<linalg::GenericOp>(user)) {
        if (!llvm::is_contained(generic.getOutputs(), alias)) continue;
        if (writer || alias != memory || generic.getOutputs().size() != 1 || generic.getNumReductionLoops()) return std::nullopt;
        writer = generic;
      } else if (auto cast = dyn_cast<memref::CastOp>(user)) aliases.push_back(cast.getResult());
      else if (auto view = dyn_cast<memref::SubViewOp>(user)) aliases.push_back(view.getResult());
      else if (auto tasks = dyn_cast<TasksOp>(user)) {
        if (use.getOperandNumber() >= 1) aliases.push_back(tasks.getBody().front().getArgument(use.getOperandNumber()));
      } else if (isa<RegionFoldOp, RegionScanOp>(user)) {
        if (llvm::is_contained(mlir::cast<RegionOpInterface>(user).getDestinations(), alias)) return std::nullopt;
      } else if (auto copy = dyn_cast<memref::CopyOp>(user)) {
        if (copy.getTarget() == alias) return std::nullopt;
      } else if (!isa<memref::LoadOp, memref::DimOp, memref::DeallocOp>(user)) return std::nullopt;
    }
  }
  if (!writer || !writer.getIndexingMapsArray().back().isIdentity()) return std::nullopt;
  DominanceInfo dominance(at->getParentOfType<func::FuncOp>());
  if (!dominance.properlyDominates(writer, at)) return std::nullopt;
  Value yielded = stripIndexWidthCasts(writer.getRegion().front().getTerminator()->getOperand(0));
  if (auto addition = yielded.getDefiningOp<arith::AddIOp>()) {
    Value left = stripIndexWidthCasts(addition.getLhs());
    Value right = stripIndexWidthCasts(addition.getRhs());
    if (left.getDefiningOp<linalg::IndexOp>()) {
      yielded = left;
      result.origin = right;
    } else if (right.getDefiningOp<linalg::IndexOp>()) {
      yielded = right;
      result.origin = left;
    } else return std::nullopt;
    if (!dominance.properlyDominates(result.origin, writer) ||
        !shiftedSequenceDoesNotWrap(allocation, result.origin, ranges)) return std::nullopt;
    // Range construction also adds view offsets and an exclusive endpoint.
    // Prove those relative index operations fit before removing the origin.
    auto extent = coordinateBounds(ranges.dimension(selectedMemory, 0));
    if (!extent || extent->first.isNegative()) return std::nullopt;
    llvm::APInt upper = extent->second;
    for (OpFoldResult offset : result.offsets) {
      auto constant = getConstantIntValue(offset);
      auto bounds = constant ? std::optional<SignedBounds>({wideInteger(*constant), wideInteger(*constant)})
                             : coordinateBounds(ranges.range(cast<Value>(offset)));
      if (!bounds || bounds->first.isNegative()) return std::nullopt;
      upper += bounds->second;
      if (upper.sgt(llvm::APInt::getSignedMaxValue(64).sext(65))) return std::nullopt;
    }
  }
  auto index = yielded.getDefiningOp<linalg::IndexOp>();
  if (!index || index.getDim() != 0 || index->getBlock() != &writer.getRegion().front())
    return std::nullopt;
  return result;
}

std::optional<RegionPredicatePlan> analyzeRegionPredicate(RegionOpInterface program) {
  if (program.isScan() || !getEffectsRecursively(program.getOperation())) return std::nullopt;
  Block &helper = program.getSummarize().front();
  SmallVector<CoordinatePredicate> predicates;
  for (auto generic : helper.getOps<linalg::GenericOp>()) {
    for (auto compare : generic.getRegion().front().getOps<arith::CmpIOp>())
      if (auto predicate = coordinatePredicate(program, generic, compare))
        predicates.push_back(*predicate);
  }
  if (predicates.empty()) return std::nullopt;
  const CoordinatePredicate &selected = predicates.front();
  SmallVector<CoordinatePredicate> equivalent;
  SmallVector<Value> conditions;
  for (const CoordinatePredicate &predicate : predicates)
    if (predicate.source == selected.source && predicate.capture == selected.capture &&
        predicate.comparison == selected.comparison) {
      equivalent.push_back(predicate);
      conditions.push_back(predicate.value);
    }
  auto validity = validityField(program, equivalent);
  bool identity = summaryIsIdentityWhenFalse(program, conditions);
  return RegionPredicatePlan{std::move(conditions), selected.source, selected.capture,
      selected.comparison, validity ? std::optional<unsigned>(validity->first) : std::nullopt,
      validity ? validity->second : nullptr, identity};
}

}
