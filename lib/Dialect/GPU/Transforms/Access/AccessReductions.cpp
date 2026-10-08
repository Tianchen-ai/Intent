#include "AccessComposition.h"
#include "../Value/ReplayPolicy.h"
#include "Intent/Dialect/GPU/Analysis/Helpers.h"
#include "Intent/Dialect/GPU/Transforms/Value/Helpers.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include <limits>

using namespace mlir;

namespace intent::gpu::access {

FailureOr<bool> composeReducedGather(GatherOp gather) {
  auto reduce = gather.getSource().getDefiningOp<ReduceOp>();
  auto output = dyn_cast<FragmentType>(gather.getSource().getType());
  if (!reduce || !output || isa<FragmentType>(gather.getResult().getType()) ||
      reduce.getSources().size() != 1 || reduce.getIdentities().size() != 1 ||
      reduce.getCaptures().size() != 0 || reduce.getNumResults() != 1 ||
      gather.getSourceAxes().size() != output.getShape().size() ||
      !queryBinaryCombine(reduce.getCombine()) ||
      std::distance(reduce.getCombine().front().begin(),
                    reduce.getCombine().front().end()) != 2)
    return false;

  auto input = cast<FragmentType>(reduce.getSources().front().getType());
  auto kernel = gather->getParentOfType<func::FuncOp>();
  ReplayPolicy reuse(kernel, ValueRange{gather.getSource()}, {gather.getOperation()},
                     [&](Value value) { return value == gather.getSource(); });
  IRMapping bindings;
  for (Value value : reduce.getSources()) bindings.map(value, value);
  for (Value value : reduce.getIdentities()) bindings.map(value, value);
  if (reuse.duplicatesExpensiveWork(gather.getSource(), &bindings)) return false;
  PhysicalProgramAnalysis analysis(kernel);
  SmallVector<Value> freeCoordinates(output.getShape().size());
  for (auto [coordinate, axis] :
       llvm::zip(gather.getCoordinates(), gather.getSourceAxes())) {
    if (axis < 0 || axis >= static_cast<int64_t>(freeCoordinates.size()) ||
        freeCoordinates[axis] || isa<FragmentType>(coordinate.getType()))
      return false;
    freeCoordinates[axis] = coordinate;
  }
  SmallVector<Attribute> shape;
  for (unsigned axis = 0; axis < input.getShape().size(); ++axis) {
    if (!llvm::is_contained(reduce.getAxes(), static_cast<int64_t>(axis)))
      continue;
    auto range = queryExactLogicalRange(analysis.axisRanges(reduce.getSources().front(), axis));
    auto extent = cast<PhysicalExprAttr>(input.getShape()[axis]);
    if (failed(range) ||
        extent.getKind() != PhysicalExprKind::Constant ||
        extent.getValue() <= 0 ||
        constantLogicalRangeCardinality(*range) != extent.getValue())
      return false;
    shape.push_back(extent);
  }

  // Select complete reduction fibers from immutable SSA. Loads remain behind
  // gathers until composeLoadGather proves that moving each read is legal.
  OpBuilder builder(gather);
  Location location = gather.getLoc();
  auto [sourceId, dimensionId] = nextPhysicalAxisIdentities(kernel);
  SmallVector<Attribute> mappings;
  SmallVector<Value> coordinates;
  SmallVector<int64_t> sourceAxes;
  SmallVector<int64_t> reductionAxes;
  unsigned freeAxis = 0, reductionAxis = 0;
  Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
  Value one = builder.create<arith::ConstantIndexOp>(location, 1);
  for (unsigned axis = 0; axis < input.getShape().size(); ++axis) {
    sourceAxes.push_back(axis);
    if (!llvm::is_contained(reduce.getAxes(), static_cast<int64_t>(axis))) {
      coordinates.push_back(freeCoordinates[freeAxis++]);
      continue;
    }
    auto extent = cast<PhysicalExprAttr>(shape[reductionAxis]);
    auto mapping = AxisMapAttr::get(builder.getContext(), sourceId,
                                    reductionAxis, dimensionId++, reductionAxis, true);
    mappings.push_back(mapping);
    auto rangeType = FragmentType::get(builder.getContext(), builder.getIndexType(),
        builder.getArrayAttr({extent}), builder.getArrayAttr({AxisMapAttr::get(
            builder.getContext(), sourceId, reductionAxis, mapping.getDimensionId(), 0, true)}),
        input.getValidity(), input.getOwner());
    Value stop = builder.create<arith::ConstantIndexOp>(location, extent.getValue());
    coordinates.push_back(builder.create<MakeRangeOp>(
        location, rangeType, zero, stop, one, zero, stop, sourceId, reductionAxis, true));
    reductionAxes.push_back(reductionAxis++);
  }
  auto selectedType = FragmentType::get(builder.getContext(), input.getElementType(),
      builder.getArrayAttr(shape), builder.getArrayAttr(mappings),
      input.getValidity(), input.getOwner());
  auto bounded = gatherBounds(builder, location, output, gather.getCoordinates(),
                              gather.getSourceAxes(), builder.getIndexType());
  if (failed(bounded))
    return failure();
  auto predicate = FragmentType::get(builder.getContext(), builder.getI1Type(),
      selectedType.getShape(), selectedType.getAxisMaps(),
      selectedType.getValidity(), selectedType.getOwner());
  Value valid = builder.create<BroadcastOp>(location, predicate, *bounded);
  auto zeroFill = materializeZeroFragment(builder, location, selectedType);
  if (failed(zeroFill))
    return failure();
  Value selected = builder.create<GatherOp>(location, selectedType,
      reduce.getSources().front(), coordinates, valid, *zeroFill, sourceAxes);
  Type resultType = gather.getResult().getType();
  Value identityFill = builder.create<arith::ConstantOp>(location, builder.getZeroAttr(resultType));
  Value identity = builder.create<GatherOp>(location, resultType,
      reduce.getIdentities().front(), gather.getCoordinates(), *bounded,
      identityFill, gather.getSourceAxes());
  auto projected = builder.create<ReduceOp>(location, ValueRange{selected},
      ValueRange{identity}, ValueRange{}, reductionAxes);
  if (failed(scalarizeElementwiseCallback(reduce.getCombine(), projected.getCombine())))
    return failure();
  Value replacement = projected.getResult(0);
  if (gather.getValid())
    replacement = builder.create<SelectOp>(location, resultType, gather.getValid(),
                                           replacement, gather.getFill());
  gather.getResult().replaceAllUsesWith(replacement);
  gather.erase();
  return true;
}

FailureOr<bool> composeReductionGathers(ReduceOp reduce) {
  if (reduce.getSources().size() != 1 || reduce.getIdentities().size() != 1 ||
      reduce.getCaptures().size() != 0 || reduce.getNumResults() != 1 ||
      reduce.getAxes() != ArrayRef<int64_t>{0})
    return false;
  Value input = reduce.getSources().front();
  auto schema = dyn_cast<FragmentType>(input.getType());
  auto result = dyn_cast<FragmentType>(reduce.getResult(0).getType());
  if (!schema || schema.getShape().size() != 1 ||
      (result && !result.getShape().empty()))
    return false;
  auto extent = cast<PhysicalExprAttr>(schema.getShape()[0]);
  int64_t size = extent.getValue();
  if (extent.getKind() != PhysicalExprKind::Constant ||
      size <= 0 || (size & (size - 1)) != 0 ||
      size > std::numeric_limits<int64_t>::max() / 2)
    return false;

  SmallVector<GatherOp> gathers;
  SmallVector<MakeRangeOp> ranges;
  llvm::DenseSet<Value> visited;
  std::function<bool(Value)> collect = [&](Value value) {
    auto fragment = dyn_cast<FragmentType>(value.getType());
    if (!fragment || !visited.insert(value).second)
      return true;
    if (fragment.getShape() != schema.getShape() ||
        fragment.getAxisMaps() != schema.getAxisMaps() ||
        fragment.getValidity() != schema.getValidity() ||
        fragment.getOwner() != schema.getOwner())
      return false;
    if (auto range = value.getDefiningOp<MakeRangeOp>()) {
      ranges.push_back(range);
      return true;
    }
    if (auto gather = value.getDefiningOp<GatherOp>()) {
      auto source = cast<FragmentType>(gather.getSource().getType());
      if (source.getShape() != schema.getShape() ||
          source.getOwner() != schema.getOwner() ||
          gather.getSourceAxes() != ArrayRef<int64_t>{0} ||
          gather.getCoordinates().size() != 1 || !gather.getValid() ||
          !gather.getFill() ||
          !gather.getCoordinates().front().getDefiningOp<MakeRangeOp>())
        return false;
      if (!collect(gather.getCoordinates().front()) ||
          !collect(gather.getValid()) || !collect(gather.getFill()))
        return false;
      gathers.push_back(gather);
      return true;
    }
    Operation *producer = value.getDefiningOp();
    if (!isa_and_nonnull<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp,
                        BitcastOp, SplatOp, BroadcastOp, ReshapeOp>(producer) ||
        producer->getNumResults() != 1 || !isMemoryEffectFree(producer))
      return false;
    return llvm::all_of(producer->getOperands(), collect);
  };
  if (!collect(input) || gathers.empty() || ranges.empty())
    return false;
  MakeRangeOp range = ranges.front();
  if (isZero(range.getStart()) || !isUnitStepRange(range) ||
      !samePhysicalScalarExpression(range.getStart(), range.getLogicalStart()))
    return false;
  for (MakeRangeOp other : ranges)
    if (!sameLogicalRange(range, other) ||
        !samePhysicalScalarExpression(range.getStart(), other.getStart()) ||
        !samePhysicalScalarExpression(range.getExtent(), other.getExtent()))
      return false;
  PhysicalExprAttr bound = queryNonNegativeIndexUpperBound(range.getStart());
  if (!bound ||
      bound.getKind() != PhysicalExprKind::Constant ||
      bound.getValue() > size)
    return false;

  UniformValueAnalysis uniform(describeUniformValue);
  for (GatherOp gather : gathers) {
    Value coordinate = gather.getCoordinates().front();
    SmallVector<Value> predicates{gather.getValid()};
    bool bounded = false;
    while (!predicates.empty()) {
      Value predicate = predicates.pop_back_val();
      if (auto binary = predicate.getDefiningOp<BinaryOp>();
          binary &&
          (binary.getOperatorKind() == BinaryOperator::LogicalAnd ||
           binary.getOperatorKind() == BinaryOperator::BitwiseAnd)) {
        predicates.push_back(binary.getLhs());
        predicates.push_back(binary.getRhs());
      } else if (auto compare = predicate.getDefiningOp<CompareOp>();
                 compare && compare.getPredicate() == ComparePredicate::Lt &&
                 compare.getLhs() == coordinate) {
        auto upper = dyn_cast_or_null<IntegerAttr>(
            uniform.evaluate(compare.getRhs()));
        bounded |= upper && upper.getInt() <= size;
      }
    }
    if (!bounded)
      return false;
  }
  auto kernel = reduce->getParentOfType<func::FuncOp>();
  PhysicalProgramAnalysis analysis(kernel);
  PhysicalSourceAxis source = sourceAxisIdentity(range);
  if (!analysis.replayability(input, source, PhysicalReplayScope::ValueGraph,
                             /*allowAccesses=*/true).isReplayable())
    return false;

  SmallVector<Value> replayRoots{input};
  IRMapping bindings;
  for (MakeRangeOp original : ranges)
    bindings.map(original.getResult(), original.getResult());
  for (GatherOp gather : gathers) {
    // The rewrite projects the existing parent SSA and rebuilds only the
    // validity/fill and suffix; it never replays the gathered parent graph.
    bindings.map(gather.getResult(), gather.getResult());
    replayRoots.push_back(gather.getValid());
    replayRoots.push_back(gather.getFill());
  }
  ReplayPolicy reuse(kernel, replayRoots, {reduce.getOperation()},
                     [&](Value value) { return visited.contains(value); });
  if (llvm::any_of(replayRoots, [&](Value value) {
        return reuse.duplicatesExpensiveWork(value, &bindings);
      }))
    return false;

  // Ordinary reduce permits a permutation of its inputs. For a shifted
  // [k, k + N) range, visit [N, N + k) then [k, N) instead. This retains
  // every original value, including padding/fill, while valid gathers become
  // identity reads from the immutable parent fragment. Other users keep their
  // original coordinates; no scan or ordered state is permuted.
  OpBuilder builder(reduce);
  Location location = reduce.getLoc();
  auto [sourceId, dimensionId] = nextPhysicalAxisIdentities(kernel);
  auto axis = AxisMapAttr::get(builder.getContext(), sourceId, 0, dimensionId,
                             0, true);
  auto typeFor = [&](Type element) {
    return FragmentType::get(builder.getContext(), element, schema.getShape(),
                            builder.getArrayAttr({axis}), schema.getValidity(),
                            schema.getOwner());
  };
  auto indexType = typeFor(builder.getIndexType());
  Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
  Value one = builder.create<arith::ConstantIndexOp>(location, 1);
  Value count = builder.create<arith::ConstantIndexOp>(location, size);
  Value ordinal = builder.create<MakeRangeOp>(
      location, indexType, zero, count, one, zero, count, sourceId, 0, true);
  Value start = builder.create<SplatOp>(location, indexType, range.getStart());
  Value prefix = builder.create<CompareOp>(location, typeFor(builder.getI1Type()),
                                          ordinal, start, ComparePredicate::Lt);
  Value shifted = builder.create<BinaryOp>(
      location, indexType, ordinal,
      builder.create<SplatOp>(location, indexType, count), BinaryOperator::Add);
  Value coordinate = builder.create<SelectOp>(location, indexType, prefix,
                                              shifted, ordinal);
  IRMapping mapping;
  for (MakeRangeOp original : ranges)
    mapping.map(original.getResult(), coordinate);
  ReplayMaterializationOptions options;
  options.traversalRanges = ranges;
  options.fragmentAxis = 0;
  options.segmentMapping = axis;
  auto replay = [&](Value value) {
    return materializeReplayedValue(builder, location, value, source, extent,
                                    mapping, reduce, options);
  };
  auto group = ReshapeGroupAttr::get(builder.getContext(),
                                    builder.getDenseI64ArrayAttr({0}),
                                    builder.getDenseI64ArrayAttr({0}));
  for (GatherOp gather : gathers) {
    auto valid = replay(gather.getValid());
    auto fill = replay(gather.getFill());
    if (failed(valid) || failed(fill))
      return reduce.emitOpError("cannot permute gather validity/fill"), failure();
    auto type = typeFor(uniformElementType(gather.getResult().getType()));
    Value parent = builder.create<ReshapeOp>(
        location, type, gather.getSource(), builder.getArrayAttr({group}));
    Value replacement = builder.create<SelectOp>(location, type, *valid,
                                                 parent, *fill);
    mapping.map(gather.getResult(), replacement);
  }
  auto replacement = replay(input);
  if (failed(replacement))
    return reduce.emitOpError("cannot permute gather reduction inputs"), failure();
  reduce->setOperand(0, *replacement);
  return true;
}

} // namespace intent::gpu::access
