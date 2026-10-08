#include "Pointwise.h"
#include "Intent/Dialect/GPU/Analysis/Helpers.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/Transforms/Control/Traversal.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Dominance.h"
#include <algorithm>
#include <limits>

using namespace mlir;

namespace intent::gpu::pointwise {

void inheritRangeAuthority(Operation *target, MakeRangeOp source) {
  for (StringRef name :
       {originAttr, sourceSubregionAttr, sourceSubregionBoundAttr,
        worksetCoordinateRangeAttr})
    if (Attribute value = source->getAttr(name))
      target->setAttr(name, value);
}

struct ProductConstraint {
  int64_t constant = 1;
  StringAttr parameter;
  unsigned parameterCount = 0;
};



bool collectProductConstraint(PhysicalExprAttr extent,
                              ProductConstraint &constraint) {
  auto kind = extent.getKind();
  if (kind == PhysicalExprKind::Constant) {
    if (extent.getValue() <= 0)
      return false;
    if (constraint.constant >
        std::numeric_limits<int64_t>::max() / extent.getValue())
      return false;
    constraint.constant *= extent.getValue();
    return true;
  }
  if (kind == PhysicalExprKind::Parameter) {
    ++constraint.parameterCount;
    if (constraint.parameter && constraint.parameter != extent.getParameterReference().getName())
      return false;
    constraint.parameter = extent.getParameterReference().getName();
    return true;
  }
  if (kind != PhysicalExprKind::Multiply || extent.getOperands().size() != 2)
    return false;
  return collectProductConstraint(
             cast<PhysicalExprAttr>(extent.getOperands()[0]), constraint) &&
         collectProductConstraint(
             cast<PhysicalExprAttr>(extent.getOperands()[1]), constraint);
}

bool collectProductConstraint(FragmentType fragment, ArrayRef<unsigned> axes,
                              ProductConstraint &constraint) {
  for (unsigned axis : axes)
    if (!collectProductConstraint(
            cast<PhysicalExprAttr>(fragment.getShape()[axis]), constraint))
      return false;
  return true;
}

LogicalResult bindStructurallyRequiredStaticFragments(func::FuncOp kernel) {
  llvm::MapVector<ParameterAttr, int64_t> required;
  auto requireExtent = [&](Operation *operation, StringAttr symbol,
                           int64_t candidate) -> LogicalResult {
    FailureOr<ParameterAttr> declaration = queryParameterBySymbol(kernel, symbol);
    if (failed(declaration))
      return success();
    ParameterAttr parameter = *declaration;
    PhysicalParameterBinding binding = queryParameterBinding(parameter);
    if (!binding.isExact() || !binding.source)
      return success();
    if (!llvm::is_contained(
            parameter.getCandidates().asArrayRef(), candidate))
      return operation->emitOpError(
          "reshape requires a static fragment extent outside its legal domain");
    auto found = required.find(parameter);
    if (found != required.end() && found->second != candidate)
      return operation->emitOpError(
          "one static fragment has incompatible structural extent requirements");
    required[parameter] = candidate;
    return success();
  };
  SmallVector<ReshapeOp> reshapes;
  SmallVector<std::pair<Value, unsigned>> fixedAxes;
  kernel.walk([&](ReshapeOp reshape) { reshapes.push_back(reshape); });
  for (ReshapeOp reshape : reshapes) {
    auto source = dyn_cast<FragmentType>(reshape.getValue().getType());
    auto result = dyn_cast<FragmentType>(reshape.getResult().getType());
    if (!source || !result)
      continue;
    auto relations = queryFragmentOperandRelations(reshape.getOperation());
    if (failed(relations))
      return reshape.emitOpError(
          "static fragment constraint has no physical axis relation");
    for (const auto &group : relations->front().groups) {
      ArrayRef<unsigned> sourceAxes = group.sourceAxes;
      ArrayRef<unsigned> resultAxes = group.resultAxes;
      // Unmerged axes carry their tile extent through the reshape. They do
      // not become full-coverage dimensions because other axes are reshaped.
      if (sourceAxes.size() <= 1 && resultAxes.size() <= 1)
        continue;
      ProductConstraint sourceProduct;
      ProductConstraint resultProduct;
      if (!collectProductConstraint(source, sourceAxes, sourceProduct) ||
          !collectProductConstraint(result, resultAxes, resultProduct))
        continue;
      if (sourceAxes.size() > 1 && resultAxes.size() == 1 &&
          sourceProduct.parameterCount == 0 &&
          resultProduct.parameterCount == 0 && resultProduct.constant > 1 &&
          sourceProduct.constant == resultProduct.constant)
        fixedAxes.emplace_back(reshape.getResult(), resultAxes.front());
      if (sourceAxes.size() == 1 && resultAxes.size() > 1 &&
          sourceProduct.parameterCount == 0 && resultProduct.parameterCount == 0 &&
          sourceProduct.constant == resultProduct.constant)
        for (unsigned axis : resultAxes)
          fixedAxes.emplace_back(reshape.getResult(), axis);
      if (sourceProduct.parameterCount != 1 ||
          resultProduct.parameterCount != 0 ||
          resultProduct.constant % sourceProduct.constant != 0)
        continue;
      int64_t candidate = resultProduct.constant / sourceProduct.constant;
      if (failed(requireExtent(reshape, sourceProduct.parameter, candidate)))
        return failure();
    }
  }

  // A retained static merge fixes its lane count through explicit pointwise
  // and one-to-one reshape relations, including positional source rebinding.
  // Propagate that structural constraint, not equality of unrelated shapes.
  for (auto [seed, seedAxis] : fixedAxes) {
    int64_t extent = cast<PhysicalExprAttr>(
        cast<FragmentType>(seed.getType()).getShape()[seedAxis]).getValue();
    SmallVector<std::pair<Value, unsigned>> pending{{seed, seedAxis}};
    llvm::DenseSet<std::pair<Value, unsigned>> visited;
    while (!pending.empty()) {
      auto [value, axis] = pending.pop_back_val();
      if (!visited.insert({value, axis}).second)
        continue;
      for (Operation *user : value.getUsers()) {
        if (user->getNumResults() != 1)
          continue;
        auto result = dyn_cast<FragmentType>(user->getResult(0).getType());
        if (!result)
          continue;
        std::optional<unsigned> resultAxis;
        if (!isa<ReshapeOp, BroadcastOp, UnaryOp, BinaryOp, CompareOp, SelectOp,
                 CastOp, BitcastOp>(user))
          continue;
        auto relations = queryFragmentOperandRelations(user);
        if (failed(relations))
          continue;
        for (const auto &relation : *relations) {
          if (user->getOperand(relation.operandNumber) != value)
            continue;
          for (const auto &group : relation.groups)
            if (group.sourceAxes.size() == 1 && group.resultAxes.size() == 1 &&
                group.sourceAxes[0] == axis)
              resultAxis = group.resultAxes[0];
        }
        if (!resultAxis)
          continue;
        auto constrain = [&](PhysicalExprAttr expression,
                             bool mayBroadcast = false) -> LogicalResult {
          if (expression.getKind() ==
              PhysicalExprKind::Parameter)
            return requireExtent(user, expression.getParameterReference().getName(), extent);
          if (expression.getKind() ==
                  PhysicalExprKind::Constant &&
              expression.getValue() != extent &&
              !(mayBroadcast && expression.getValue() == 1))
            return user->emitOpError(
                "pointwise relation conflicts with a static reshape extent");
          return success();
        };
        if (failed(constrain(
                cast<PhysicalExprAttr>(result.getShape()[*resultAxis]))))
          return failure();
        if (!isa<ReshapeOp>(user))
          for (const auto &relation : *relations) {
            Value operand = user->getOperand(relation.operandNumber);
            auto input = dyn_cast<FragmentType>(operand.getType());
            if (!input)
              continue;
            const auto *group = relation.groupForResultAxis(*resultAxis);
            if (group && group->sourceAxes.size() == 1 &&
                failed(constrain(cast<PhysicalExprAttr>(
                    input.getShape()[group->sourceAxes[0]]), true)))
              return failure();
          }
        pending.emplace_back(user->getResult(0), *resultAxis);
      }
    }
  }

  for (auto [parameter, candidate] : required) {
    StringAttr parameterName = parameter.getName();
    PhysicalExprAttr fixedExtent =
        expression(kernel.getContext(), PhysicalExprKind::Constant, candidate);
    // A parameter binding applies to every typed occurrence of its symbol,
    // including values reached through positional axis rebinding.
    AttrTypeReplacer replacer;
    replacer.addReplacement(
        [&](PhysicalExprAttr current) -> std::optional<Attribute> {
          if (current.getKind() ==
                  PhysicalExprKind::Parameter &&
              current.getParameterReference().getName() == parameterName)
            return fixedExtent;
          return std::nullopt;
        });
    replacer.recursivelyReplaceElementsIn(kernel.getOperation(),
                                          /*replaceAttrs=*/true,
                                          /*replaceLocs=*/false,
                                          /*replaceTypes=*/true);
    SmallVector<ParameterOp> reads;
    kernel.walk([&](ParameterOp read) {
      if (read.getReference() == parameter.getReference()) reads.push_back(read);
    });
    for (ParameterOp read : reads) {
      OpBuilder builder(read);
      Value fixed = builder.create<arith::ConstantIndexOp>(read.getLoc(), candidate);
      read.getResult().replaceAllUsesWith(fixed);
      read.erase();
    }
  }
  eraseUnusedParameters(kernel);
  return success();
}

LogicalResult requireFullDimensionCoverage(func::FuncOp kernel, Value source,
                                           uint64_t axis) {
  if (hasExactStaticFullCoverage(kernel, source, axis))
    return success();
  return realizeFullCoverageDimension(kernel, source, axis);
}

LogicalResult requireScanFullCoverage(func::FuncOp kernel, ScanOp scan,
                                      Value source, uint64_t axis) {
  auto fragment = dyn_cast<FragmentType>(source.getType());
  if (!fragment || axis >= fragment.getShape().size())
    return failure();
  if (hasExactStaticFullCoverage(kernel, source, axis))
    return success();
  auto sourceMapping = cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
  int64_t dimension = sourceMapping.getDimensionId();
  if (dimension <= 0)
    return requireFullDimensionCoverage(kernel, source, axis);
  PhysicalProgramAnalysis analysis(kernel);
  PhysicalRangeFact sourceRanges = analysis.axisRanges(source, axis);
  if (sourceRanges.state == PhysicalFactState::Unknown ||
      sourceRanges.roots.empty()) {
    PhysicalReplayFact replay = analysis.replayability(
        source, sourceAxisIdentity(sourceMapping),
        PhysicalReplayScope::ValueGraph, /*allowAccesses=*/false);
    if (!sourceRanges.roots.empty() || !replay.isReplayable()) {
      InFlightDiagnostic diagnostic = kernel.emitError(
          "scan source has no exact full-coverage range authority");
      diagnostic << "; source_type=" << source.getType();
      for (Operation *blocker : sourceRanges.blockers)
        diagnostic << ", blocker=" << blocker->getName();
      return failure();
    }
  }
  bool subregion = llvm::any_of(sourceRanges.roots, [](MakeRangeOp range) {
    return range->hasAttr(sourceSubregionAttr);
  });
  if (subregion) {
    auto combine = queryBinaryCombine(scan.getCombine());
    if (scan.getSources().size() != 1 || scan.getIdentities().size() != 1 ||
        scan.getCaptures().size() != 0 ||
        !combine || combine->kind() != BinaryOperator::Add ||
        !isZeroScanIdentity(
            scan.getIdentities().front()))
      return scan.emitOpError(
          "dynamic subregion scan has no proven tail-neutral combine");
    auto chunkExtent = cast<PhysicalExprAttr>(fragment.getShape()[axis]);
    FailureOr<int64_t> bound =
        exactSubregionStaticBound(sourceRanges,
                                  sourceAxisIdentity(sourceMapping));
    if (chunkExtent.getKind() !=
            PhysicalExprKind::Constant ||
        failed(bound) || chunkExtent.getValue() < *bound)
      return scan.emitOpError(
          "dynamic subregion scan has no static source bound");
    PhysicalSourceAxis sourceAxis = sourceAxisIdentity(sourceMapping);
    for (MakeRangeOp range : sourceRanges.roots) {
      if (!range->hasAttr(sourceSubregionAttr) || !isUnitStepRange(range) ||
          !(sourceAxisIdentity(range) == sourceAxis))
        return range.emitOpError(
            "subregion scan source has incompatible physical ranges");
      if (failed(retargetSourceExtent(range.getResult(), sourceAxis,
                                      chunkExtent)))
        return failure();
    }
    return retargetFragmentAxisExtent(source, axis, chunkExtent);
  }
  if (failed(dimensionArgument(kernel, dimension))) {
    FailureOr<int64_t> staticExtent =
        exactStaticTraversalExtent(sourceRanges);
    if (failed(staticExtent))
      return scan.emitOpError(
                 "scan full-coverage dimension has neither runtime ABI nor exact static range authority")
             << "; dimension=" << dimension
             << "; source=" << source.getType();
    PhysicalExprAttr covered = expression(
        kernel.getContext(), PhysicalExprKind::Constant, *staticExtent);
    for (MakeRangeOp range : sourceRanges.roots) {
      FailureOr<uint64_t> rangeIdentity = rangeDimension(range);
      if (failed(rangeIdentity) ||
          *rangeIdentity != static_cast<uint64_t>(dimension) ||
          range->hasAttr(sourceSubregionAttr))
        return range.emitOpError(
            "static scan source range does not match its logical dimension");
      if (failed(retargetDimensionExtent(range.getResult(), dimension, covered)))
        return failure();
    }
    return retargetFragmentAxisExtent(source, axis, covered);
  }
  std::string parameterName = ("FULL_D" + Twine(dimension)).str();
  ParameterAttr parameter = lookupParameter(kernel, StringAttr::get(kernel.getContext(), parameterName));
  OpBuilder builder(&kernel.getBody().front(), kernel.getBody().front().begin());
  if (!parameter) {
    static constexpr int64_t candidates[] = {
        64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
    auto schema = ParameterAttr::get(
        kernel.getContext(), builder.getStringAttr(parameterName), builder.getIndexType(),
        ParameterRole::ScanChunk,
        ParameterCategory::Coverage,
        /*elementBitWidth=*/0,
        DenseI64ArrayAttr::get(kernel.getContext(), candidates),
        ConfigurationBindingPhase::Deferred,
        ParameterBindingAttr::get(kernel.getContext(), builder.getI64IntegerAttr(dimension),
                                 {}, {}, {}, false, false));
    if (failed(declareParameter(kernel, schema))) return failure();
    parameter = schema;
  }
  parameter = parameter.withPhase(ConfigurationBindingPhase::Deferred).withBinding(
      parameter.getBinding().withDimension(builder.getI64IntegerAttr(dimension)));
  if (failed(updateParameter(kernel, parameter))) return failure();
  PhysicalExprAttr covered = fragmentExtent(parameter);
  for (MakeRangeOp range : sourceRanges.roots) {
    FailureOr<uint64_t> rangeIdentity = rangeDimension(range);
    if (failed(rangeIdentity) ||
        *rangeIdentity != static_cast<uint64_t>(dimension) ||
        range->hasAttr(sourceSubregionAttr))
      return range.emitOpError(
          "scan source range does not match its full-coverage dimension");
    if (failed(retargetDimensionExtent(range.getResult(), dimension, covered)))
      return failure();
  }
  if (failed(retargetFragmentAxisExtent(source, axis, covered)))
    return failure();
  if (failed(
          bindFullCoverageDimension(kernel, dimension,
              materializeParameter(builder, source.getLoc(), parameter.getReference()))))
    return kernel.emitError(
        "scan source could not bind its exact full-coverage parameter");
  return success();
}

LogicalResult requireStructuredReductionFullCoverage(func::FuncOp kernel,
                                                      Value source,
                                                      uint64_t axis) {
  auto fragment = dyn_cast<FragmentType>(source.getType());
  if (!fragment || axis >= fragment.getShape().size())
    return failure();
  if (hasExactStaticFullCoverage(kernel, source, axis))
    return success();
  auto sourceMapping = cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
  PhysicalProgramAnalysis analysis(kernel);
  PhysicalRangeFact ranges = analysis.axisRanges(source, axis);
  bool subregion = llvm::any_of(ranges.roots, [](MakeRangeOp range) {
    return range->hasAttr(sourceSubregionAttr);
  });
  if (subregion) {
    PhysicalSourceAxis sourceAxis = sourceAxisIdentity(sourceMapping);
    FailureOr<int64_t> bound = exactSubregionStaticBound(ranges, sourceAxis);
    auto extent = cast<PhysicalExprAttr>(fragment.getShape()[axis]);
    if (failed(bound) || extent.getKind() !=
            PhysicalExprKind::Constant ||
        extent.getValue() < *bound)
      return realizeFullCoverageDimension(kernel, source, axis);
    for (MakeRangeOp range : ranges.roots)
      if (failed(retargetSourceExtent(range.getResult(), sourceAxis, extent)))
        return failure();
    return retargetFragmentAxisExtent(source, axis, extent);
  }
  return realizeFullCoverageDimension(kernel, source, axis);
}

FragmentType predicateType(FragmentType source) {
  return FragmentType::get(source.getContext(), IntegerType::get(source.getContext(), 1),
                           source.getShape(), source.getAxisMaps(),
                           source.getValidity(),
                           source.getOwner());
}

FailureOr<unsigned> tailPredicateAxis(FragmentType target, MakeRangeOp range) {
  FailureOr<int64_t> dimension = queryRangeDimension(range);
  auto source = queryFragmentAxis(target, sourceAxisIdentity(range));
  if (source.isExact() && succeeded(dimension) && source.dimensionId == *dimension)
    return source.fragmentAxis;
  if (range->hasAttr(sourceSubregionAttr))
    return failure();
  auto axis = succeeded(dimension) ? queryFragmentDimension(target, *dimension)
                                   : PhysicalDimensionProjection{};
  return axis.isExact() ? FailureOr<unsigned>(axis.fragmentAxis)
                        : FailureOr<unsigned>(failure());
}

bool hasTailPredicate(
    Value coordinate,
    const llvm::DenseMap<Value, Value> &rangePredicates) {
  llvm::SmallPtrSet<Operation *, 8> ranges;
  collectCoordinateRanges(coordinate, ranges);
  return llvm::any_of(ranges, [&](Operation *operation) {
    auto range = dyn_cast<MakeRangeOp>(operation);
    return range && rangePredicates.contains(range.getResult());
  });
}

FailureOr<Value> accessValidity(OpBuilder &builder, Location location,
                                ValueRange coordinates,
                                llvm::DenseMap<Value, Value> &rangePredicates,
                                FragmentType valueType, Value existing) {
  FragmentType target = predicateType(valueType);
  Value result;
  if (existing) {
    FailureOr<Value> broadcast =
        projectPhysicalValueToSchema(builder, location, existing, target);
    if (failed(broadcast))
      return emitError(location, "pointwise access existing validity has incompatible schema")
             << "; predicate=" << existing.getType() << "; target=" << target;
    result = *broadcast;
  }
  SmallVector<Value> fragmentCoordinates;
  for (Value coordinate : coordinates)
    if (isa<FragmentType>(coordinate.getType()))
      fragmentCoordinates.push_back(coordinate);
  bool cartesian = fragmentCoordinates.size() == target.getShape().size() &&
      llvm::all_of(fragmentCoordinates, [](Value coordinate) {
        return cast<FragmentType>(coordinate.getType()).getShape().size() == 1;
      });
  unsigned fragmentAxis = 0;
  for (Value coordinate : coordinates) {
    std::optional<unsigned> positionalAxis;
    if (isa<FragmentType>(coordinate.getType())) {
      if (cartesian)
        positionalAxis = fragmentAxis;
      ++fragmentAxis;
    }
    llvm::SmallPtrSet<Operation *, 8> ranges;
    collectCoordinateRanges(coordinate, ranges);
    for (Operation *operation : ranges) {
      auto range = dyn_cast<MakeRangeOp>(operation);
      if (!range)
        continue;
      auto found = rangePredicates.find(range.getResult());
      if (found == rangePredicates.end())
        continue;
      FailureOr<unsigned> axis = tailPredicateAxis(target, range);
      if (failed(axis) && positionalAxis)
        axis = *positionalAxis;
      if (failed(axis))
        continue;
      FailureOr<Value> broadcast = projectPredicateToFragmentAxis(
          builder, location, found->second, target, *axis);
      if (failed(broadcast))
        return emitError(location, "pointwise access tail has incompatible schema")
               << "; tail=" << found->second.getType() << "; target=" << target;
      result = result ? Value(builder.create<BinaryOp>(
                            location, target, result, *broadcast,
                            BinaryOperator::LogicalAnd))
                      : *broadcast;
    }
  }
  return result ? FailureOr<Value>(result) : FailureOr<Value>(failure());
}

LogicalResult addTailValidity(func::FuncOp kernel, Operation *scope,
                              llvm::DenseMap<Value, Value> &rangePredicates,
                              bool includeStores) {
  SmallVector<Operation *> reads;
  SmallVector<StoreOp> stores;
  SmallVector<HistogramOp> histograms;
  scope->walk([&](Operation *operation) {
    if (isa<LoadOp, GatherOp>(operation)) reads.push_back(operation);
  });
  scope->walk([&](StoreOp store) { stores.push_back(store); });
  scope->walk([&](HistogramOp histogram) { histograms.push_back(histogram); });

  auto updateRead = [&](auto load) -> LogicalResult {
    bool affected = llvm::any_of(load.getCoordinates(), [&](Value coordinate) {
      return hasTailPredicate(coordinate, rangePredicates);
    });
    if (!affected)
      return success();
    auto valueType = dyn_cast<FragmentType>(load.getResult().getType());
    if (!valueType)
      return load.emitOpError(
          "pointwise blocked read must produce a physical fragment");
    OpBuilder builder(load);
    FailureOr<Value> valid = accessValidity(
        builder, load.getLoc(), load.getCoordinates(), rangePredicates, valueType,
        load.getValid());
    if (failed(valid)) {
      InFlightDiagnostic diagnostic =
          load.emitOpError("could not form pointwise tail validity");
      diagnostic << "; value=" << valueType;
      if (load.getValid())
        diagnostic << "; existing=" << load.getValid().getType();
      else
        diagnostic << "; existing=<none>";
      return failure();
    }
    Value fill;
    if (load.getFill()) {
      FailureOr<Value> broadcast =
          projectPhysicalValueToSchema(builder, load.getLoc(), load.getFill(), valueType);
      if (failed(broadcast))
        return load.emitOpError("could not broadcast the existing load fill");
      fill = *broadcast;
    } else {
      FailureOr<Value> zero = materializeZeroFragment(builder, load.getLoc(), valueType);
      if (failed(zero))
        return load.emitOpError("pointwise load element type has no zero fill");
      fill = *zero;
    }
    load.getValidMutable().assign(*valid);
    load.getFillMutable().assign(fill);
    return success();
  };
  for (Operation *read : reads) {
    LogicalResult updated = isa<LoadOp>(read) ? updateRead(cast<LoadOp>(read))
                                             : updateRead(cast<GatherOp>(read));
    if (failed(updated)) return failure();
  }

  for (HistogramOp histogram : histograms) {
    auto valueType = dyn_cast<FragmentType>(histogram.getValues().getType());
    if (!valueType)
      return histogram.emitOpError(
          "pointwise blocked histogram must consume a physical fragment");
    FragmentType validType = predicateType(valueType);
    OpBuilder builder(histogram);
    FailureOr<Value> existing = projectPhysicalValueToSchema(
        builder, histogram.getLoc(), histogram.getValid(), validType);
    if (failed(existing))
      return histogram.emitOpError(
          "could not broadcast the existing histogram validity");
    Value valid = *existing;
    bool affected = false;
    for (auto [rangeValue, predicate] : rangePredicates) {
      auto range = rangeValue.getDefiningOp<MakeRangeOp>();
      if (!range)
        continue;
      llvm::SmallPtrSet<Operation *, 8> ranges;
      collectProducerRanges(
          histogram.getValues(),
          PhysicalSourceAxis{range.getSourceId(), range.getSourceAxis(),
                           range.getDerived()},
          ranges);
      if (!ranges.contains(range.getOperation()))
        continue;
      FailureOr<Value> broadcast =
          projectPhysicalValueToSchema(builder, histogram.getLoc(), predicate, validType);
      if (failed(broadcast))
        return histogram.emitOpError(
            "could not project pointwise tail validity onto histogram values");
      valid = builder.create<BinaryOp>(histogram.getLoc(), validType, valid,
                                       *broadcast, BinaryOperator::LogicalAnd);
      affected = true;
    }
    if (!affected)
      continue;
    histogram.getValidMutable().assign(valid);
  }

  if (!includeStores)
    return success();
  SmallVector<std::pair<MakeRangeOp, IRMapping>> storeReplays;
  for (StoreOp store : stores) {
    bool affected = llvm::any_of(store.getCoordinates(), [&](Value coordinate) {
      return hasTailPredicate(coordinate, rangePredicates);
    });
    if (!affected)
      continue;
    OpBuilder builder(store);
    Value payload = store.getValue();
    Value existingValidity = store.getValid();
    auto valueType = dyn_cast<FragmentType>(payload.getType());
    if (valueType) {
      SmallVector<MakeRangeOp> blockedRanges;
      for (Value coordinate : store.getCoordinates()) {
        llvm::SmallPtrSet<Operation *, 8> roots;
        collectCoordinateRanges(coordinate, roots);
        for (Operation *root : roots) {
          auto range = dyn_cast<MakeRangeOp>(root);
          if (range && rangePredicates.contains(range.getResult()) &&
              !llvm::is_contained(blockedRanges, range))
            blockedRanges.push_back(range);
        }
      }
      for (MakeRangeOp range : blockedRanges) {
        FailureOr<int64_t> dimension = queryRangeDimension(range);
        auto sourceProjection = queryFragmentAxis(valueType, sourceAxisIdentity(range));
        PhysicalDimensionProjection dimensionProjection =
            succeeded(dimension)
                ? queryFragmentDimension(valueType, *dimension)
                : PhysicalDimensionProjection{};
        if (failed(dimension) ||
            (!sourceProjection.isExact() && !dimensionProjection.isExact()))
          continue;
        unsigned payloadAxis = sourceProjection.isExact()
                                   ? sourceProjection.fragmentAxis
                                   : dimensionProjection.fragmentAxis;
        auto rangeType = cast<FragmentType>(range.getResult().getType());
        auto blockedExtent =
            cast<PhysicalExprAttr>(rangeType.getShape()[0]);
        if (HistogramOp histogram = histogramSource(payload)) {
          if (failed(retargetDimensionExtent(histogram.getResult(), *dimension,
                                            blockedExtent)))
            return failure();
          payload = store.getValue();
          valueType = dyn_cast<FragmentType>(payload.getType());
          if (!valueType)
            return store.emitOpError(
                "histogram writeback lost its physical result schema");
        }
        if (valueType.getShape()[payloadAxis] == blockedExtent)
          continue;
        IRMapping mapping;
        DominanceInfo dominance(kernel);
        for (auto &entry : storeReplays) {
          MakeRangeOp previous = entry.first;
          auto previousDimension = queryRangeDimension(previous);
          if (failed(previousDimension) || *previousDimension != *dimension ||
              !samePhysicalScalarExpression(previous.getStart(), range.getStart()) ||
              !samePhysicalScalarExpression(previous.getExtent(), range.getExtent()) ||
              !samePhysicalScalarExpression(previous.getLogicalStart(), range.getLogicalStart()) ||
              !samePhysicalScalarExpression(previous.getLogicalStop(), range.getLogicalStop()) ||
              !samePhysicalScalarExpression(previous.getStep(), range.getStep()))
            continue;
          for (auto [original, replayed] : entry.second.getValueMap())
            if (dominance.dominates(replayed, store.getOperation()))
              mapping.map(original, replayed);
        }
        auto payloadMapping =
            cast<AxisMapAttr>(valueType.getAxisMaps()[payloadAxis]);
        SmallVector<int64_t> traversalDimensions{payloadMapping.getDimensionId()};
        FailureOr<Value> replayed = replayPointwiseValue(
            builder, payload, sourceAxisIdentity(range), traversalDimensions,
            blockedExtent,
            range.getResult(), rangePredicates.lookup(range.getResult()),
            store.getOperation(), mapping);
        if (failed(replayed))
          return store.emitOpError(
                     "pointwise store payload cannot be replayed to its blocked coordinate")
                 << "; source_id=" << range.getSourceId()
                 << ", source_axis=" << range.getSourceAxis()
                 << ", dimension="
                 << (succeeded(dimension) ? *dimension : -1)
                 << ", payload=" << payload.getType()
                 << ", coordinate=" << range.getResult().getType();
        if (existingValidity) {
          FailureOr<Value> replayedValidity = replayPointwiseValue(
              builder, existingValidity, sourceAxisIdentity(range),
              traversalDimensions, blockedExtent, range.getResult(),
              rangePredicates.lookup(range.getResult()), store.getOperation(),
              mapping);
          if (failed(replayedValidity))
            return store.emitOpError(
                "pointwise store validity cannot be replayed with its payload");
          existingValidity = *replayedValidity;
        }
        storeReplays.emplace_back(range, mapping);
        payload = *replayed;
        valueType = dyn_cast<FragmentType>(payload.getType());
        if (!valueType)
          return store.emitOpError(
              "pointwise replay lost the store payload fragment schema");
      }
    }
    if (!valueType) {
      if (!isa<IntegerType, FloatType, IndexType>(payload.getType()))
        return store.emitOpError(
            "pointwise blocked store has no projectable value schema");
      SmallVector<Attribute> shape;
      SmallVector<Attribute> mappings;
      std::optional<uint64_t> owner;
      for (Value coordinate : store.getCoordinates()) {
        auto fragment = dyn_cast<FragmentType>(coordinate.getType());
        if (!fragment)
          continue;
        if (owner && *owner != fragment.getOwner())
          return store.emitOpError(
              "pointwise store coordinates have conflicting ownership");
        owner = fragment.getOwner();
        for (auto [extent, attribute] :
             llvm::zip(fragment.getShape(), fragment.getAxisMaps())) {
          auto mapping = cast<AxisMapAttr>(attribute);
          auto found = llvm::find_if(mappings, [&](Attribute existing) {
            auto axis = cast<AxisMapAttr>(existing);
            return sourceAxisIdentity(axis) == sourceAxisIdentity(mapping);
          });
          if (found != mappings.end())
            continue;
          shape.push_back(extent);
          mappings.push_back(AxisMapAttr::get(
              store.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
              mapping.getDimensionId(), mappings.size(), mapping.getDerived()));
        }
      }
      if (shape.empty())
        return store.emitOpError(
            "pointwise blocked store has no coordinate fragment authority");
      auto target = FragmentType::get(
          store.getContext(), payload.getType(),
          ArrayAttr::get(store.getContext(), shape),
          ArrayAttr::get(store.getContext(), mappings), /*validity=*/1,
          owner.value_or(1));
      FailureOr<Value> projected = projectPhysicalValueToSchema(
          builder, store.getLoc(), payload, target);
      if (failed(projected))
        return store.emitOpError(
            "pointwise store value cannot be projected to its coordinate schema");
      payload = *projected;
      valueType = target;
    }
    FailureOr<Value> valid = accessValidity(
        builder, store.getLoc(), store.getCoordinates(), rangePredicates,
        valueType, existingValidity);
    if (failed(valid))
      return store.emitOpError("could not form pointwise store validity");
    store.getValueMutable().assign(payload);
    store.getValidMutable().assign(*valid);
  }
  return success();
}

LogicalResult PointwiseRewrite::prepareCoverage() {
  bool scanCoverageFailed = false;
  kernel.walk([&](ScanOp scan) {
    for (Value source : scan.getSources())
      scanCoverageFailed |=
          failed(requireScanFullCoverage(kernel, scan, source, scan.getAxis()));
  });
  if (scanCoverageFailed)
    return kernel.emitError(
        "dynamic scan axis has no launch-visible full-coverage realization");
  SmallVector<std::pair<Value, uint64_t>> structuredReductionSources;
  kernel.walk([&](ReduceOp reduce) {
    if (laneReductions.contains(reduce))
      return;
    for (Value source :
         reduce.getSources()) {
      auto fragment = dyn_cast<FragmentType>(source.getType());
      if (!fragment)
        continue;
      for (int64_t axis : reduce.getAxes()) {
        if (axis < 0 || axis >= static_cast<int64_t>(fragment.getShape().size()))
          continue;
        auto mapping =
            cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
        PhysicalReplayFact replay = PhysicalProgramAnalysis(kernel).replayability(
            source,
            PhysicalSourceAxis{mapping.getSourceId(), mapping.getSourceAxis(),
                           mapping.getDerived()},
            PhysicalReplayScope::ValueGraph, /*allowAccesses=*/true);
        bool structuredBlocker = llvm::any_of(
            replay.blockers, [](Operation *blocker) {
              return isa<ScanOp, scf::ForOp>(blocker);
            });
        if (structuredBlocker)
          structuredReductionSources.emplace_back(source,
                                                   static_cast<uint64_t>(axis));
      }
    }
  });
  for (auto [source, axis] : structuredReductionSources)
    if (failed(requireStructuredReductionFullCoverage(kernel, source, axis)))
      return kernel.emitError(
          "structured reduction source has no exact full-coverage realization");
  return success();
}


LogicalResult PointwiseRewrite::finalizeValues() {
  eraseDeadPhysicalValues(kernel);
  if (failed(bindStructurallyRequiredStaticFragments(kernel)))
    return failure();
  if (failed(closeValueRelations(kernel)))
    return failure();
  PhysicalProgramAnalysis analysis(kernel);
  WalkResult localized = kernel.walk([&](GatherOp gather) {
    auto source = dyn_cast<FragmentType>(gather.getSource().getType());
    if (!source)
      return WalkResult::advance();
    for (auto [position, axis] : llvm::enumerate(gather.getSourceAxes())) {
      Value coordinate = gather.getCoordinates()[position];
      Value origin = coordinate;
      while (auto broadcast = origin.getDefiningOp<BroadcastOp>())
        origin = broadcast.getValue();
      auto range = origin.getDefiningOp<MakeRangeOp>();
      auto begin = range ? queryLaunchExpression(range.getLogicalStart())
                         : PhysicalExprAttr();
      if (!range || !isUnitStepRange(range) ||
          !begin || begin.getKind() != PhysicalExprKind::Constant ||
          begin.getValue() != 0 ||
          samePhysicalScalarExpression(range.getStart(), range.getLogicalStart()) ||
          range.getResult().getType().getShape()[0] != source.getShape()[axis])
        continue;
      auto roots = analysis.axisRanges(gather.getSource(), axis);
      if (!roots.isExact() || roots.roots.empty() || !roots.blockers.empty() ||
          !llvm::all_of(roots.roots, [&](MakeRangeOp root) {
            return sameLogicalRange(root, range) &&
                   samePhysicalScalarExpression(root.getStart(), range.getStart()) &&
                   samePhysicalScalarExpression(root.getExtent(), range.getExtent());
          }))
        continue;
      OpBuilder builder(gather);
      auto type = cast<FragmentType>(coordinate.getType());
      auto start = projectPhysicalValueToSchema(builder, gather.getLoc(),
                                                  range.getStart(), type);
      if (failed(start))
        return WalkResult::interrupt();
      Value ordinal = builder.create<BinaryOp>(
          gather.getLoc(), type, coordinate, *start, BinaryOperator::Subtract);
      gather.getCoordinatesMutable().slice(position, 1).assign(ordinal);
    }
    return WalkResult::advance();
  });
  if (localized.wasInterrupted())
    return failure();
  eraseDeadPhysicalValues(kernel);
  return success();
}


LogicalResult PointwiseRewrite::materializeFixedRanges() {
  llvm::DenseMap<Value, Value> fixedRangePredicates;
  SmallVector<MakeRangeOp> unresolved;
  for (MakeRangeOp range : dynamicRanges) {
    if (internalTraversalRanges.contains(range.getOperation())) {
      unresolved.push_back(range);
      continue;
    }
    FailureOr<ParameterAttr> parameter = queryBlockingParameter(kernel, range);
    if (failed(parameter) ||
        parameter->getRole() !=
            ParameterRole::ScanChunk) {
      unresolved.push_back(range);
      continue;
    }
    // Region fold/scan owns this logical traversal.  Its physical segment
    // parameter is the loop step, not a replacement for the logical source
    // extent; replacing the extent would silently truncate the program to
    // one segment before structured lowering sees it.
  }
  dynamicRanges = std::move(unresolved);
  unresolved.clear();
  for (MakeRangeOp range : dynamicRanges) {
    auto fragment = cast<FragmentType>(range.getResult().getType());
    auto physicalExtent = cast<PhysicalExprAttr>(fragment.getShape()[0]);
    const bool fixedSubregion =
        range->hasAttr(sourceSubregionAttr) &&
        physicalExtent.getKind() ==
            PhysicalExprKind::Constant;
    if (!fixedSubregion && boundedWritebackRanges.contains(range.getOperation()) &&
        reuseTraversalRanges.contains(range.getOperation())) {
      unresolved.push_back(range);
      continue;
    }
    if (PhysicalProgramAnalysis(kernel)
            .axisRealization(range.getResult(), 0)
            .constructionScalarSeed) {
      FailureOr<int64_t> staticExtent = exactStaticTraversalExtent(
          PhysicalProgramAnalysis(kernel).axisRanges(range.getResult(), 0));
      if (failed(staticExtent) ||
          physicalExtent.getKind() !=
              PhysicalExprKind::Constant ||
          !samePhysicalScalarExpression(range.getStart(),
                                        range.getLogicalStart())) {
        unresolved.push_back(range);
        continue;
      }
      uint64_t covered = llvm::PowerOf2Ceil(
          static_cast<uint64_t>(std::max<int64_t>(*staticExtent, 1)));
      if (covered > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        return range.emitOpError("static traversal extent exceeds index range");
      physicalExtent = expression(kernel.getContext(),
                                  PhysicalExprKind::Constant, covered);
      if (failed(retargetSourceExtent(range.getResult(),
                                      sourceAxisIdentity(range), physicalExtent)))
        return failure();
      fragment = cast<FragmentType>(range.getResult().getType());
    }
    auto fixedExtent = constantPhysicalExpression(physicalExtent);
    if (fixedExtent && *fixedExtent > 0 &&
        !llvm::isPowerOf2_64(*fixedExtent) &&
        isUnitStepRange(range)) {
      uint64_t padded = llvm::PowerOf2Ceil(
          static_cast<uint64_t>(*fixedExtent));
      if (padded > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        return range.emitOpError("static fragment padding exceeds index range");
      physicalExtent = expression(kernel.getContext(),
                                  PhysicalExprKind::Constant, padded);
      auto dimension = queryRangeDimension(range);
      LogicalResult retargeted =
          succeeded(dimension)
              ? retargetDimensionExtent(range.getResult(), *dimension,
                                        physicalExtent)
              : retargetSourceExtent(range.getResult(), sourceAxisIdentity(range),
                                     physicalExtent);
      if (failed(retargeted))
        return failure();
      fragment = cast<FragmentType>(range.getResult().getType());
    }
    if (!fixedSubregion && reuseTraversalRanges.contains(range.getOperation())) {
      unresolved.push_back(range);
      continue;
    }
    if (physicalExtent.getKind() !=
        PhysicalExprKind::Constant) {
      unresolved.push_back(range);
      continue;
    }
    OpBuilder builder(range);
    Value extent = builder.create<arith::ConstantIndexOp>(
        range.getLoc(), physicalExtent.getValue());
    auto blocked = builder.create<MakeRangeOp>(
        range.getLoc(), fragment, range.getStart(), extent, range.getStep(),
        range.getLogicalStart(), range.getLogicalStop(), range.getSourceId(),
        range.getSourceAxis(), range.getDerived());
    inheritRangeAuthority(blocked, range);
    Value endFragment =
        builder.create<BroadcastOp>(range.getLoc(), fragment,
                                    range.getLogicalStop());
    auto validComparison = builder.create<CompareOp>(
        range.getLoc(), predicateType(fragment), blocked.getResult(),
        endFragment, ComparePredicate::Lt);
    validComparison->setAttr(physicalTailAttr, builder.getUnitAttr());
    Value lower = builder.create<BroadcastOp>(
        range.getLoc(), fragment, range.getLogicalStart());
    Value active = builder.create<CompareOp>(
        range.getLoc(), predicateType(fragment), blocked.getResult(), lower,
        ComparePredicate::Ge);
    Value valid = builder.create<BinaryOp>(
        range.getLoc(), predicateType(fragment), active, validComparison,
        BinaryOperator::LogicalAnd);
    range.getResult().replaceAllUsesWith(blocked.getResult());
    fixedRangePredicates[blocked.getResult()] = valid;
    range.erase();
  }
  // A fixed physical extent immediately requires executable tail validity.
  // Keeping the predicate only in a side map across later rewrites leaves it
  // without an IR use and lets dead-value cleanup invalidate the fact.
  if (!fixedRangePredicates.empty()) {
    if (failed(closeValueRelations(kernel, ValueRelationScope::AccessResults)) ||
        failed(addTailValidity(kernel, kernel, fixedRangePredicates,
                               /*includeStores=*/true)))
      return failure();
    fixedRangePredicates.clear();
  }
  dynamicRanges = std::move(unresolved);
  return success();
}


LogicalResult PointwiseRewrite::coverLocalRanges() {
  SmallVector<MakeRangeOp> fullCoverageRanges;
  // Contraction blocking owns the runtime row loop and tail validity for
  // device-derived subregions. They are not pointwise full-coverage values.
    llvm::erase_if(dynamicRanges, [&](MakeRangeOp range) {
      return internalTraversalRanges.contains(range.getOperation()) &&
             uses.contractionTraversalRanges.contains(range.getOperation()) &&
             hasAccessDependentSubregionBounds(kernel, range);
    });
    for (MakeRangeOp range : dynamicRanges) {
      if (retainedGatherRanges.contains(range.getOperation())) {
        fullCoverageRanges.push_back(range);
        continue;
      }
      if (!internalTraversalRanges.contains(range.getOperation()) ||
          (uses.structuredTraversalRanges.contains(range.getOperation()) &&
           !writeTraversalRanges.contains(range.getOperation())) ||
          uses.reductionFreeRanges.contains(range.getOperation()) ||
          reuseTraversalRanges.contains(range.getOperation()))
        continue;
      if (failed(requireFullDimensionCoverage(kernel, range.getResult(), 0)))
        return range.emitOpError(
                   "internal pointwise traversal has no exact full-coverage realization")
               << "; source_id=" << range.getSourceId()
               << ", fragment=" << range.getResult().getType();
      fullCoverageRanges.push_back(range);
    }
  llvm::erase_if(dynamicRanges, [&](MakeRangeOp range) {
    return llvm::is_contained(fullCoverageRanges, range) ||
           hasExactStaticFullCoverage(kernel, range.getResult(), 0);
  });
  return success();
}


LogicalResult PointwiseRewrite::retainGatherSources() {
  SmallVector<std::pair<Value, unsigned>> retainedGatherAxes;
  kernel.walk([&](GatherOp gather) {
    auto fragment = dyn_cast<FragmentType>(gather.getSource().getType());
    if (!fragment)
      return;
    for (int64_t axis : gather.getSourceAxes()) {
      auto mapping = cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
      auto replay = PhysicalProgramAnalysis(kernel).replayability(
          gather.getSource(), sourceAxisIdentity(mapping),
          PhysicalReplayScope::ValueGraph, /*allowAccesses=*/true);
      if (llvm::any_of(replay.blockers, [](Operation *operation) {
            return isa<scf::ForOp>(operation);
          }) && !llvm::is_contained(retainedGatherAxes,
                                    std::pair<Value, unsigned>{gather.getSource(), axis}))
        retainedGatherAxes.emplace_back(gather.getSource(), axis);
    }
  });
  for (auto [source, axis] : retainedGatherAxes) {
    if (failed(requireStructuredReductionFullCoverage(kernel, source, axis)))
      return emitError(source.getLoc(), "ordered tensor gather has no full source-axis realization");
    auto roots = PhysicalProgramAnalysis(kernel).axisRanges(source, axis);
    for (MakeRangeOp range : allRanges)
      if (llvm::any_of(roots.roots, [&](MakeRangeOp root) {
            return sameLogicalRange(root, range) &&
                   samePhysicalScalarExpression(root.getStart(), range.getStart());
          })) {
        // A surviving gather consumes an immutable ordered carry, including
        // members outside an output tile. Retain that source axis; unrelated
        // free axes still participate in ordinary ownership selection.
        deferRange(range.getOperation());
        retainedGatherRanges.insert(range.getOperation());
      }
  }
  return success();
}


} // namespace intent::gpu::pointwise
