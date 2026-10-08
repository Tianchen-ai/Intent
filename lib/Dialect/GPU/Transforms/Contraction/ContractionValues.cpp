#include "Intent/Dialect/GPU/IR/ProgramInterface.h"
#include "ContractionDetail.h"
#include "../Value/ReplayPolicy.h"
#include "../Value/SourceReplay.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Transforms/Value/ExecutionSchema.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "Intent/Dialect/GPU/Transforms/Control/Traversal.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Transforms/Storage/Storage.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <functional>
#include <limits>
#include <tuple>


using namespace mlir;

namespace intent::gpu::contraction {

static Value stripCoverageProjection(Value value);

static bool impliesLogicalUpperBound(Value predicate, MakeRangeOp range,
                               llvm::SmallDenseSet<Value> &visited);

static Value broadcast(OpBuilder &builder, Location location, FragmentType result,
                Value value);

LogicalResult prepareRetainedContractionReads(func::FuncOp kernel) {
  // Close retained read snapshots before creating any contraction slices.
  // Materializing one operand later can otherwise retarget an existing slice
  // that shares the same logical dimension with that snapshot.
  WalkResult snapshots = kernel.walk([&](ContractOp contract) {
    for (auto [value, axes] :
         {std::pair{contract.getLhs(), contract.getLhsReductionAxes()},
          std::pair{contract.getRhs(), contract.getRhsReductionAxes()}}) {
      for (int64_t axis : axes) {
        PhysicalProgramAnalysis analysis(kernel);
        auto type = cast<FragmentType>(value.getType());
        auto relation = cast<AxisMapAttr>(type.getAxisMaps()[axis]);
        // Operands already supplied as a native structured value need no
        // producer replay. Preserve that existing contraction realization.
        if (!analysis.replayability(value, sourceAxisIdentity(relation),
                PhysicalReplayScope::ValueGraph, true,
                relation.getDimensionId()).isReplayable()) continue;
        if (failed(prepareSourceReplay(value, axis, contract,
                                       PhysicalReplayScope::ValueGraph)))
          return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return failure(snapshots.wasInterrupted());
}

PhysicalExprAttr expression(MLIRContext *context, PhysicalExprKind kind,
                            int64_t value, StringRef symbol,
                            ArrayRef<Attribute> operands) {
  return PhysicalExprAttr::get(
      context, kind, value,
      kind == PhysicalExprKind::Parameter
          ? Attribute(ParameterRefAttr::get(context, StringAttr::get(context, symbol)))
          : Attribute(StringAttr::get(context, symbol)),
      ArrayAttr::get(context, operands));
}

PhysicalExprAttr parameterExpression(MLIRContext *context, StringRef name) {
  return expression(context, PhysicalExprKind::Parameter, 0, name);
}

PhysicalExprAttr binaryExpression(MLIRContext *context, PhysicalExprKind kind,
                                  PhysicalExprAttr lhs,
                                  PhysicalExprAttr rhs) {
  return expression(context, kind, 0, {}, {lhs, rhs});
}

Value binary(OpBuilder &builder, Location location, Type result, Value lhs,
             Value rhs, BinaryOperator kind) {
  return builder.create<BinaryOp>(location, result, lhs, rhs, kind);
}

Value compare(OpBuilder &builder, Location location, Type result, Value lhs,
              Value rhs, ComparePredicate predicate) {
  auto comparison =
      builder.create<CompareOp>(location, result, lhs, rhs, predicate);
  comparison->setAttr(physicalTailAttr, builder.getUnitAttr());
  return comparison.getResult();
}

void inheritRangeAuthority(Value derived, MakeRangeOp source) {
  Operation *operation = derived.getDefiningOp();
  for (StringRef name : {sourceSubregionAttr, sourceSubregionBoundAttr})
    if (Attribute value = source->getAttr(name))
      operation->setAttr(name, value);
}

LogicalResult refineOwnershipParameter(func::FuncOp kernel, MakeRangeOp range,
                                       FailureOr<unsigned> mappingAxis,
                                       ParameterAttr replacement) {
  if (failed(mappingAxis))
    return success();
  FailureOr<ParameterAttr> previous = queryBlockingParameter(kernel, range);
  if (failed(previous))
    return range.emitOpError(
        "pointwise ownership axis has no typed blocking parameter to refine");
  if (previous->getReference() == replacement.getReference())
    return success();
  ParameterAttr previousSchema = *previous;
  auto previousRole = previousSchema.getRole();
  if (previousSchema.getCategory() !=
          ParameterCategory::Pointwise ||
      (previousRole != ParameterRole::OwnershipM &&
       previousRole != ParameterRole::OwnershipN))
    return range.emitOpError(
        "contraction can only refine a provisional pointwise ownership parameter");
  replacement = lookupParameter(kernel, replacement.getReference());
  auto inherited = previous->getBinding(), current = replacement.getBinding();
  auto conflicts = [](Attribute lhs, Attribute rhs) { return lhs && rhs && lhs != rhs; };
  if (conflicts(inherited.getDimension(), current.getDimension()) ||
      conflicts(inherited.getSource(), current.getSource()) ||
      conflicts(inherited.getCoverageBound(), current.getCoverageBound()))
    return range.emitOpError("contraction ownership refinement has conflicting parameter bindings");
  auto merged = current
      .withDimension(current.getDimension() ? current.getDimension() : inherited.getDimension())
      .withSource(current.getSource() ? current.getSource() : inherited.getSource())
      .withCoverageBound(current.getCoverageBound() ? current.getCoverageBound() : inherited.getCoverageBound());
  if (previous->isDeferred()) replacement = replacement.withPhase(ConfigurationBindingPhase::Deferred);
  replacement = replacement.withBinding(merged);
  if (failed(updateParameter(kernel, replacement))) return failure();
  return replaceParameter(kernel, previous->getReference(), replacement.getReference());
}

FailureOr<Value> buildRangeTailPredicate(OpBuilder &builder, Location location,
                                         Value range,
                                         MakeRangeOp authority) {
  auto rangeType = dyn_cast<FragmentType>(range.getType());
  if (!rangeType || rangeType.getShape().size() != 1)
    return failure();
  Value logicalStop = builder.create<SplatOp>(
      location, rangeType, authority.getLogicalStop());
  auto predicateType = FragmentType::get(
      rangeType.getContext(), builder.getI1Type(), rangeType.getShape(),
      rangeType.getAxisMaps(), rangeType.getValidity(), rangeType.getOwner());
  return builder
      .create<CompareOp>(location, predicateType, range, logicalStop,
                         ComparePredicate::Lt)
      .getResult();
}

LogicalResult appendTailValidity(Location location, Value source,
                                 ArrayRef<MakeRangeOp> ranges,
                                 Value tailPredicate,
                                 IRMapping &mapping) {
  SmallVector<Value> pending{source};
  llvm::DenseSet<Value> visited;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    Operation *producer = value.getDefiningOp();
    if (!producer)
      continue;
    if (auto originalLoad = dyn_cast<LoadOp>(producer)) {
      Value mapped = mapping.lookupOrNull(originalLoad.getResult());
      auto load = mapped ? mapped.getDefiningOp<LoadOp>() : LoadOp();
      if (!load)
        continue;
      auto resultType = dyn_cast<FragmentType>(load.getResult().getType());
      if (!resultType)
        return load.emitOpError(
            "blocked contraction tail requires a fragment load result");
      auto predicateType = FragmentType::get(
          resultType.getContext(), IntegerType::get(resultType.getContext(), 1),
          resultType.getShape(),
          resultType.getAxisMaps(), resultType.getValidity(),
          resultType.getOwner());
      OpBuilder validityBuilder(load);
      auto kernel = originalLoad->getParentOfType<func::FuncOp>();
      auto axes = PhysicalProgramAnalysis(kernel).rangeAxes(
          originalLoad.getResult(), ranges);
      if (!axes.isExact() || axes.fragmentAxes.size() != 1)
        return load.emitOpError(
            "blocked contraction tail has no unique load-axis relation");
      FailureOr<Value> projected = projectPredicateToFragmentAxis(
          validityBuilder, location, tailPredicate, predicateType,
          axes.fragmentAxes.front());
      if (failed(projected))
        return load.emitOpError(
            "blocked contraction tail cannot project to its load coordinates");
      Value valid = *projected;
      if (load.getValid()) {
        Value existing = load.getValid();
        if (existing.getType() != predicateType) {
          FailureOr<Value> projectedExisting = projectPhysicalValueToSchema(
              validityBuilder, location, existing, predicateType);
          if (failed(projectedExisting))
            return load.emitOpError(
                "blocked contraction tail cannot preserve existing load validity");
          existing = *projectedExisting;
        }
        valid = validityBuilder.create<BinaryOp>(
            location, predicateType, existing, valid,
            BinaryOperator::LogicalAnd);
      }
      load.getValidMutable().assign(ValueRange{valid});
      if (!load.getFill()) {
        Type elementType = resultType.getElementType();
        Value zero = validityBuilder.create<arith::ConstantOp>(
            location, elementType, validityBuilder.getZeroAttr(elementType));
        Value fill =
            validityBuilder.create<SplatOp>(location, resultType, zero);
        load.getFillMutable().assign(ValueRange{fill});
      }
      continue;
    }
    pending.append(producer->operand_begin(), producer->operand_end());
  }
  return success();
}

LogicalResult markNativeCoverage(func::FuncOp kernel, Value source,
                                 ArrayRef<int64_t> axes) {
  auto fragment = dyn_cast<FragmentType>(source.getType());
  if (!fragment)
    return failure();
  for (unsigned axis = 0; axis < fragment.getShape().size(); ++axis) {
    auto extent = cast<PhysicalExprAttr>(fragment.getShape()[axis]);
    if (extent.getKind() != PhysicalExprKind::Constant ||
        extent.getValue() <= 0 || llvm::isPowerOf2_64(extent.getValue()))
      continue;
    if (failed(realizeFullCoverageDimension(kernel, source, axis)))
      return failure();
    fragment = cast<FragmentType>(source.getType());
  }
  for (int64_t axis : axes) {
    if (axis < 0 || axis >= static_cast<int64_t>(fragment.getShape().size()))
      return failure();
    auto extent = cast<PhysicalExprAttr>(fragment.getShape()[axis]);
    FailureOr<ParameterAttr> parameter = parameterForExtent(kernel, extent);
    if (failed(parameter)) {
      PhysicalAxisRealizationFact coverage =
          PhysicalProgramAnalysis(kernel).axisRealization(source, axis);
      if (extent.getKind() ==
              PhysicalExprKind::Constant &&
          extent.getValue() == 1 && coverage.isExact() &&
          !coverage.constructionScalarSeed && coverage.roots.empty())
        continue;
      if (!coverage.isExact() || !coverage.physicalized ||
          coverage.constructionScalarSeed) {
        if (!coverage.roots.empty() &&
            llvm::all_of(coverage.roots, [](MakeRangeOp range) {
              return isUnitStepRange(range) && samePhysicalScalarExpression(
                  range.getStart(), range.getLogicalStart());
            }) && succeeded(realizeFullCoverageDimension(kernel, source, axis))) {
          fragment = cast<FragmentType>(source.getType());
          continue;
        }
        return kernel.emitError(
            "native contraction axis has no exact physical coverage")
               << "; axis=" << axis << "; source=" << source.getType()
               << "; construction_seed=" << coverage.constructionScalarSeed;
      }
      continue;
    }
    ParameterRole role = parameter->getRole();
    if (role == ParameterRole::ScanChunk ||
        role == ParameterRole::Reduction ||
        role == ParameterRole::FullCoverage)
      continue;
    PhysicalParameterBinding binding = queryParameterBinding(*parameter);
    if (!binding.isExact() || !binding.dimension)
      return kernel.emitOpError(
          "native contraction coverage parameter has no logical dimension");
    uint64_t dimension = *binding.dimension;
    bool launchVisible = bool(resolveDimension(kernel, dimension));
    if (!launchVisible)
      return kernel.emitOpError(
          "native contraction cannot fully cover a data-dependent dimension");
    static constexpr int64_t fullCoverageCandidates[] = {
        16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
        16384, 32768, 65536};
    ParameterAttr schema = *parameter;
    if (auto current = schema.isDeferred() ? schema.getBinding().getDimension() : IntegerAttr()) {
      if (current.getInt() != static_cast<int64_t>(dimension))
        return kernel.emitOpError(
            "one physical parameter covers multiple logical dimensions");
    }
    schema = schema.withCandidates(DenseI64ArrayAttr::get(kernel.getContext(), fullCoverageCandidates))
        .withPhase(ConfigurationBindingPhase::Deferred)
        .withBinding(schema.getBinding().withDimension(
            IntegerAttr::get(IntegerType::get(kernel.getContext(), 64), dimension)));
    if (failed(updateParameter(kernel, schema))) return failure();
    OpBuilder entry(&kernel.front(), kernel.front().begin());
    if (failed(bindFullCoverageDimension(kernel, dimension,
          materializeParameter(entry, kernel.getLoc(), schema.getReference()))))
      return kernel.emitOpError(
          "native contraction coverage could not bind its exact logical dimension");

    // A logical subregion keeps its dynamic end, but a provider-native
    // contraction still needs a compile-time fragment extent.  The coverage
    // parameter covers the parent logical dimension; the existing subregion
    // predicate remains the authority for active members in the final tile.
    // Retarget only the exact provenance carried by this operand axis.
    PhysicalExprAttr parameterExtent = parameterExpression(
        kernel.getContext(), parameter->getName().getValue());
    auto sourceMapping = cast<AxisMapAttr>(
        fragment.getAxisMaps()[static_cast<unsigned>(axis)]);
    PhysicalSourceAxis source{sourceMapping.getSourceId(),
                              sourceMapping.getSourceAxis()};
    SmallVector<MakeRangeOp> subregions;
    kernel.walk([&](MakeRangeOp range) {
      FailureOr<int64_t> sourceDimension = queryRangeDimension(range);
      if (sourceAxisIdentity(range) == source &&
          range->hasAttr(sourceSubregionAttr) && succeeded(sourceDimension) &&
          *sourceDimension == static_cast<int64_t>(dimension))
        subregions.push_back(range);
    });
    for (MakeRangeOp range : subregions) {
      if (failed(retargetSourceExtent(range.getResult(), source, parameterExtent)))
        return failure();
      range->setOperand(1, materializeParameter(entry, range.getLoc(), parameter->getReference()));
    }
  }
  return success();
}

LogicalResult markNativeCoverage(func::FuncOp kernel, ContractOp contract) {
  if (failed(markNativeCoverage(kernel, contract.getLhs(),
                                contract.getLhsReductionAxes())) ||
      failed(markNativeCoverage(kernel, contract.getRhs(),
                                contract.getRhsReductionAxes())))
    return contract.emitOpError(
        "native contraction reduction coverage is not representable");
  return success();
}

LogicalResult markNativeCoverage(func::FuncOp kernel,
                                 ScaledContractOp contract) {
  if (failed(markNativeCoverage(kernel, contract.getLhs(),
                                contract.getLhsReductionAxes())) ||
      failed(markNativeCoverage(kernel, contract.getRhs(),
                                contract.getRhsReductionAxes())))
    return contract.emitOpError(
        "native scaled contraction reduction coverage is not representable");
  return success();
}

static Value stripCoverageProjection(Value value) {
  while (Operation *producer = value.getDefiningOp()) {
    if (auto broadcast = dyn_cast<BroadcastOp>(producer))
      value = broadcast.getValue();
    else if (auto reshape = dyn_cast<ReshapeOp>(producer))
      value = reshape.getValue();
    else if (auto transpose = dyn_cast<TransposeOp>(producer))
      value = transpose.getValue();
    else if (auto splat = dyn_cast<SplatOp>(producer))
      value = splat.getValue();
    else
      break;
  }
  return value;
}

static bool impliesLogicalUpperBound(Value predicate, MakeRangeOp range,
                               llvm::SmallDenseSet<Value> &visited) {
  if (!predicate || !visited.insert(predicate).second)
    return false;
  predicate = stripCoverageProjection(predicate);
  if (isZeroScalar(predicate))
    return true;
  if (auto binary = predicate.getDefiningOp<BinaryOp>())
    if (binary.getOperatorKind() == BinaryOperator::LogicalAnd ||
        binary.getOperatorKind() == BinaryOperator::BitwiseAnd)
      return impliesLogicalUpperBound(binary.getLhs(), range, visited) ||
             impliesLogicalUpperBound(binary.getRhs(), range, visited);
  auto compare = predicate.getDefiningOp<CompareOp>();
  if (!compare || compare.getPredicate() != ComparePredicate::Lt)
    return false;
  Value coordinate = stripCoverageProjection(compare.getLhs());
  Value bound = stripCoverageProjection(compare.getRhs());
  auto coordinateRange = coordinate.getDefiningOp<MakeRangeOp>();
  return coordinateRange && sameLogicalRange(coordinateRange, range) &&
         samePhysicalScalarExpression(coordinateRange.getStart(), range.getStart()) &&
         samePhysicalScalarExpression(coordinateRange.getExtent(), range.getExtent()) &&
         samePhysicalScalarExpression(bound, range.getLogicalStop());
}

bool isZeroPastLogicalEnd(Value value, MakeRangeOp range) {
  while (true) {
    value = stripCoverageProjection(value);
    if (auto cast = value.getDefiningOp<CastOp>()) {
      value = cast.getValue();
      continue;
    }
    break;
  }
  if (isZeroScalar(value))
    return true;
  Value valid;
  Value fill;
  if (auto load = value.getDefiningOp<LoadOp>()) {
    valid = load.getValid();
    fill = load.getFill();
  } else if (auto gather = value.getDefiningOp<GatherOp>()) {
    valid = gather.getValid();
    fill = gather.getFill();
  } else if (auto select = value.getDefiningOp<SelectOp>()) {
    valid = select.getCondition();
    fill = select.getFalseValue();
  }
  if (!fill || !isZeroScalar(fill))
    return false;
  llvm::SmallDenseSet<Value> visited;
  return impliesLogicalUpperBound(valid, range, visited);
}

LogicalResult neutralizeFullCoverageOperand(ContractOp contract,
                                            OpOperand &operand,
                                            ArrayRef<int64_t> axes,
                                            func::FuncOp kernel) {
  for (int64_t axis : axes) {
    Value source = operand.get();
    auto type = cast<FragmentType>(source.getType());
    bool fullCoverage = isFullCoverageExtent(contract, type.getShape()[axis]);
    auto extent = cast<PhysicalExprAttr>(type.getShape()[axis]);
    if (!fullCoverage && extent.getKind() !=
                             PhysicalExprKind::Constant)
      continue;
    PhysicalProgramAnalysis analysis(kernel);
    PhysicalRangeFact fact = analysis.axisRanges(source, static_cast<unsigned>(axis));
    auto traversal = analysis.lockstepRanges(fact.roots);
    FailureOr<MakeRangeOp> range =
        fact.state != PhysicalFactState::Unknown && fact.blockers.empty() &&
                traversal.isExact()
            ? FailureOr<MakeRangeOp>(traversal.authority)
            : FailureOr<MakeRangeOp>(failure());
    if (!fullCoverage && (failed(range) || !isUnitStepRange(*range)))
      continue;
    if (failed(range) || !isUnitStepRange(*range))
      return contract.emitOpError(
          "full-coverage contraction tail requires an exact unit-step range");
    Value stop = (*range).getLogicalStop();

    // Preserve a proven zero extension instead of materializing a second mask.
    // Derived operations such as exp do not preserve it and still need the
    // explicit contraction identity below.
    if (isZeroPastLogicalEnd(source, *range))
      continue;
    OpBuilder builder(contract);
    Location location = contract.getLoc();
    auto coordinateType = range->getResult().getType();
    auto predicateType = FragmentType::get(
        kernel.getContext(), builder.getI1Type(), coordinateType.getShape(),
        coordinateType.getAxisMaps(), coordinateType.getValidity(),
        coordinateType.getOwner());
    Value stopFragment =
        builder.create<BroadcastOp>(location, coordinateType, stop);
    Value tail = builder.create<CompareOp>(
        location, predicateType, range->getResult(), stopFragment,
        ComparePredicate::Lt);
    FailureOr<Value> valid = projectPredicateToFragmentAxis(
        builder, location, tail, type, static_cast<unsigned>(axis));
    FailureOr<Value> zero = materializeZeroFragment(builder, location, type);
    if (failed(valid) || failed(zero))
      return contract.emitOpError(
          "could not materialize full-coverage contraction identity");
    // Load padding is insufficient for derived operands: exp(padding) can be
    // infinite, and multiplying it by the other operand's zero produces NaN.
    operand.set(builder.create<SelectOp>(location, type, *valid, source, *zero));
  }
  return success();
}

FragmentType fragmentType(MLIRContext *context, Type element,
                          ArrayRef<PhysicalExprAttr> shape,
                          ArrayRef<AxisMapAttr> sourceMappings,
                          uint64_t owner) {
  SmallVector<Attribute> extents(shape.begin(), shape.end());
  SmallVector<Attribute> mappings;
  for (auto [axis, source] : llvm::enumerate(sourceMappings))
    mappings.push_back(AxisMapAttr::get(context, source.getSourceId(),
                                        source.getSourceAxis(),
                                        source.getDimensionId(), axis,
                                        source.getDerived()));
  return FragmentType::get(context, element, ArrayAttr::get(context, extents),
                           ArrayAttr::get(context, mappings), 1, owner);
}

FailureOr<Value> scalarSource(Value value) {
  while (isa<FragmentType>(value.getType())) {
    UniformExpression expression = describeUniformValue(value);
    if (expression.kind != UniformKind::Forward ||
        expression.operands.size() != 1)
      return failure();
    value = expression.operands.front();
  }
  return value;
}

bool isZeroScalar(Value value) {
  FailureOr<Value> scalar = scalarSource(value);
  if (failed(scalar))
    return false;
  auto constant = (*scalar).getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return false;
  if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
    return integer.getValue().isZero();
  if (auto floating = dyn_cast<FloatAttr>(constant.getValue()))
    return floating.getValue().isZero();
  return false;
}

static Value broadcast(OpBuilder &builder, Location location, FragmentType result,
                Value value) {
  return builder.create<BroadcastOp>(location, result, value);
}

Value broadcastAxis(OpBuilder &builder, Location location, FragmentType result,
                    Value value, unsigned targetAxis) {
  auto source = cast<FragmentType>(value.getType());
  assert(source.getShape().size() == 1 && targetAxis < result.getShape().size());
  SmallVector<Attribute> shape, groups;
  for (unsigned axis = 0; axis < result.getShape().size(); ++axis) {
    shape.push_back(axis == targetAxis ? source.getShape()[0] :
                    expression(builder.getContext(), PhysicalExprKind::Constant, 1));
    SmallVector<int64_t> sourceAxes;
    if (axis == targetAxis) sourceAxes.push_back(0);
    groups.push_back(ReshapeGroupAttr::get(builder.getContext(),
        builder.getDenseI64ArrayAttr(sourceAxes),
        builder.getDenseI64ArrayAttr({static_cast<int64_t>(axis)})));
  }
  auto expanded = FragmentType::get(builder.getContext(), source.getElementType(),
      builder.getArrayAttr(shape), result.getAxisMaps(), source.getValidity(), source.getOwner());
  Value reshaped = builder.create<ReshapeOp>(location, expanded, value, builder.getArrayAttr(groups));
  return builder.create<BroadcastOp>(location, result, reshaped);
}

Value rangeBoundsValidity(OpBuilder &builder, Location location,
                          FragmentType indexType,
                          FragmentType predicateType, Value coordinate,
                          Value logicalStop) {
  Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
  Value lower = builder.create<CompareOp>(
      location, predicateType, coordinate,
      broadcast(builder, location, indexType, zero), ComparePredicate::Ge);
  Value upper = compare(builder, location, predicateType, coordinate,
                        broadcast(builder, location, indexType, logicalStop),
                        ComparePredicate::Lt);
  return binary(builder, location, predicateType, lower, upper,
                BinaryOperator::LogicalAnd);
}

FailureOr<Value> retargetFill(OpBuilder &builder, Location location,
                              Value original, FragmentType result) {
  if (original) {
    FailureOr<Value> scalar = scalarSource(original);
    if (failed(scalar) || (*scalar).getType() != result.getElementType())
      return failure();
    return Value(builder.create<SplatOp>(location, result, *scalar));
  }
  Value zero;
  if (isa<FloatType>(result.getElementType()))
    zero = builder.create<arith::ConstantOp>(
        location, result.getElementType(),
        builder.getFloatAttr(result.getElementType(), 0.0));
  else if (auto integer = dyn_cast<IntegerType>(result.getElementType()))
    if (integer.isSignless()) {
      zero = builder.create<arith::ConstantOp>(
          location, integer, builder.getIntegerAttr(integer, 0));
    } else {
      auto signless = IntegerType::get(builder.getContext(), integer.getWidth());
      Value raw = builder.create<arith::ConstantOp>(
          location, signless, builder.getIntegerAttr(signless, 0));
      zero = builder.create<CastOp>(location, integer, raw);
    }
  else
    return failure();
  return Value(builder.create<SplatOp>(location, result, zero));
}

} // namespace intent::gpu::contraction
