#include "ReductionParameters.h"
#include "ReductionAnalysis.h"
#include "ReductionValues.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/IR/PhysicalExpressions.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/IR/ProgramInterface.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Transforms/Control/Traversal.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "llvm/Support/MathExtras.h"

#include <tuple>

using namespace mlir;

namespace intent::gpu::reduction {

FailureOr<ParameterAttr> selectReductionChunk(
    func::FuncOp kernel, PhysicalSourceAxis source, MakeRangeOp range,
    ParameterRole role, unsigned elementBitWidth, StringRef name,
    ArrayRef<int64_t> candidates) {
  auto sourceBinding = PhysicalSourceAttr::get(
      kernel.getContext(), source.sourceId, source.sourceAxis, source.derived);
  IntegerAttr dimension;
  if (auto identity = queryRangeDimension(range); succeeded(identity))
    dimension = IntegerAttr::get(IntegerType::get(kernel.getContext(), 64), *identity);
  ParameterAttr existing;
  for (Attribute attribute : getParameterDeclarations(kernel)) {
    auto declaration = cast<ParameterAttr>(attribute);
    auto binding = declaration.getBinding();
    if (!declaration.isExtent() || declaration.isDeferred() ||
        declaration.getRole() != role ||
        declaration.getCategory() != ParameterCategory::Reduction ||
        binding.getSource() != sourceBinding ||
        (dimension && binding.getDimension() && dimension != binding.getDimension()))
      continue;
    if (existing)
      return range.emitOpError("reduction traversal has multiple physical chunk declarations"), failure();
    existing = declaration;
  }
  // Different projections of one traversal can have different physical extents.
  // The existing source binding owns its domain even when this projection has no
  // parameter in its shape. Keep the widest consumer in the resource metadata.
  auto binding = existing ? existing.getBinding()
      : ParameterBindingAttr::get(kernel.getContext(), dimension, sourceBinding,
                                  {}, {}, false, false);
  auto reference = getOrCreatePhysicalParameter(
      kernel, existing ? existing.getName().getValue() : name, role,
      ParameterCategory::Reduction, elementBitWidth,
      existing ? existing.getCandidates().asArrayRef() : candidates, binding);
  if (failed(reference)) return failure();
  auto declaration = lookupParameter(kernel, *reference);
  if (dimension && !declaration.getBinding().getDimension()) {
    declaration = declaration.withBinding(declaration.getBinding().withDimension(dimension));
    if (failed(updateParameter(kernel, declaration))) return failure();
  }
  return declaration;
}

FailureOr<ParameterAttr> selectReductionTraversalChunk(
    ReduceOp reduce, Value source, unsigned axis, MakeRangeOp range,
    bool tileProducerFreeAxis) {
  auto kernel = reduce->getParentOfType<func::FuncOp>();
  auto type = cast<FragmentType>(source.getType());
  auto identity = sourceAxisIdentity(cast<AxisMapAttr>(type.getAxisMaps()[axis]));
  auto extent = cast<PhysicalExprAttr>(type.getShape()[axis]);
  FailureOr<ParameterAttr> fullCoverage = failure();
  if (!tileProducerFreeAxis && !hasNonUnitFreeAxis(reduce))
    fullCoverage = fullCoverageParameter(kernel, extent);
  auto selected = parameterForExtent(kernel, extent);
  ParameterRole role = tileProducerFreeAxis ? ParameterRole::ReductionOuter
                                            : ParameterRole::Reduction;
  if (auto parent = reduce->getParentOfType<scf::ForOp>()) {
    // Capacity clipping retains the same outer traversal role. Its step is an
    // actual typed expression, not necessarily a bare parameter SSA value.
    ParameterAttr outer;
    bool ambiguous = false;
    if (auto step = queryLaunchExpression(parent.getStep())) {
      AttrTypeWalker references;
      references.addWalk([&](ParameterRefAttr reference) {
        auto declaration = lookupParameter(kernel, reference);
        if (!declaration) { ambiguous = true; return; }
        if (declaration.getRole() == ParameterRole::ReductionOuter) {
          if (outer && outer != declaration) ambiguous = true;
          outer = declaration;
        } else if (declaration.getRole() != ParameterRole::FullCoverage)
          ambiguous = true;
      });
      references.walk(step);
    }
    if (outer && !ambiguous) role = ParameterRole::ReductionInner;
  }
  if (reduce.getSources().size() == 1 &&
      !range->hasAttr(sourceSubregionAttr) && succeeded(fullCoverage))
    return *fullCoverage;
  if (succeeded(selected) && !selected->isDeferred() && selected->getRole() == role)
    return *selected;

  std::string name = ("REDUCE_CHUNK_" + Twine(identity.sourceId) + "_A" +
                     Twine(identity.sourceAxis) + (identity.derived ? "_DERIVED" : "")).str();
  if (queryFragmentAxes(type, identity).size() > 1)
    name += ("_F" + Twine(axis)).str();
  if (role == ParameterRole::ReductionInner) name += "_INNER";
  else if (role == ParameterRole::ReductionOuter) name += "_OUTER";
  SmallVector<int64_t> candidates{4, 8, 16, 32, 64, 128, 256,
                                  512, 1024, 2048, 4096, 8192};
  if (auto capacity = queryLogicalRangeCapacity(range))
    if (auto constant = constantPhysicalExpression(capacity)) {
      uint64_t padded = llvm::PowerOf2Ceil(
          static_cast<uint64_t>(std::max<int64_t>(*constant, candidates.front())));
      llvm::erase_if(candidates, [&](int64_t candidate) {
        return static_cast<uint64_t>(candidate) > padded;
      });
      name += ("_E" + Twine(padded)).str();
    }
  return selectReductionChunk(kernel, identity, range, role,
      type.getElementType().getIntOrFloatBitWidth(), name, candidates);
}

Value parameterValue(func::FuncOp kernel, ParameterAttr declaration) {
  OpBuilder entry(&kernel.front(), kernel.front().begin());
  return materializeParameter(entry, kernel.getLoc(), declaration.getReference());
}

FailureOr<ParameterAttr> fullCoverageParameter(func::FuncOp kernel,
                                             PhysicalExprAttr extent) {
  if (extent.getKind() !=
      PhysicalExprKind::Parameter)
    return failure();
  ParameterAttr parameter = lookupParameter(kernel, extent.getParameterReference());
  auto coverage = parameter ? parameter.getBinding().getDimension() : IntegerAttr();
  if (!parameter || !parameter.isDeferred() || !coverage || coverage.getInt() <= 0)
    return failure();
  uint64_t dimension = static_cast<uint64_t>(coverage.getInt());
  bool launchVisible = bool(resolveDimension(kernel, dimension));
  if (!launchVisible)
    return failure();
  static constexpr int64_t candidates[] = {
      1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048,
      4096, 8192, 16384, 32768, 65536};
  parameter = parameter.withCandidates(DenseI64ArrayAttr::get(kernel.getContext(), candidates));
  if (failed(updateParameter(kernel, parameter))) return failure();
  return parameter;
}

namespace {

FailureOr<ParameterAttr> parameterForDimension(func::FuncOp kernel,
                                             int64_t dimension) {
  ParameterAttr result;
  bool ambiguous = false;
  for (Attribute attribute : getParameterDeclarations(kernel)) {
    auto parameter = cast<ParameterAttr>(attribute);
    auto bound = parameter.getBinding().getDimension();
    if (!parameter.isExtent() || !bound || bound.getInt() != dimension) continue;
    ParameterRole role = parameter.getRole();
    bool ownership =
        role == ParameterRole::OwnershipM ||
        role == ParameterRole::OwnershipN;
    if (!ownership && !parameter.isDeferred()) continue;
    if (result && result != parameter) {
      ambiguous = true;
      continue;
    }
    result = parameter;
  }
  return result && !ambiguous ? FailureOr<ParameterAttr>(result)
                              : FailureOr<ParameterAttr>(failure());
}

PhysicalExprAttr selectedParameterExtent(ParameterAttr parameter) {
  ParameterAttr schema = parameter;
  PhysicalParameterBinding binding = queryParameterBinding(parameter);
  if (binding.source && schema.getCandidates().size() == 1)
    return expression(parameter.getContext(), PhysicalExprKind::Constant,
                      schema.getCandidates()[0]);
  return expression(parameter.getContext(), PhysicalExprKind::Parameter, 0,
                    schema.getName().getValue());
}

FailureOr<ParameterAttr> parameterForOwnedRange(func::FuncOp kernel,
                                             MakeRangeOp range) {
  FailureOr<ParameterAttr> parameter = queryBlockingParameter(kernel, range);
  if (failed(parameter) || !isUnitStepRange(range) ||
      parameter->isDeferred())
    return failure();
  PhysicalParameterBinding binding = queryParameterBinding(*parameter);
  FailureOr<int64_t> dimension = queryRangeDimension(range);
  auto schema = *parameter;
  if (!binding.isExact() ||
      (binding.source && !(*binding.source == sourceAxisIdentity(range))) ||
      (binding.dimension &&
       (failed(dimension) || *binding.dimension != *dimension)) ||
      (schema.getRole() != ParameterRole::OwnershipM &&
       schema.getRole() != ParameterRole::OwnershipN))
    return failure();
  auto fragment = cast<FragmentType>(range.getResult().getType());
  if (fragment.getShape()[0] != selectedParameterExtent(*parameter))
    return failure();

  // Ownership materializes start = logical_start + coordinate * tile * step.
  // A matching parameter declaration alone does not distinguish a scalar seed.
  auto otherOperand = [](Value value, Value term,
                         BinaryOperator kind) -> Value {
    auto binary = value.getDefiningOp<BinaryOp>();
    if (!binary || binary.getOperatorKind() != kind)
      return {};
    if (samePhysicalScalarExpression(binary.getLhs(), term))
      return binary.getRhs();
    if (samePhysicalScalarExpression(binary.getRhs(), term))
      return binary.getLhs();
    return {};
  };
  Value offset = otherOperand(range.getStart(), range.getLogicalStart(),
                              BinaryOperator::Add);
  Value tileOffset = offset ? otherOperand(offset, range.getStep(),
                                          BinaryOperator::Multiply)
                            : Value();
  Value coordinate;
  if (auto multiply = tileOffset ? tileOffset.getDefiningOp<BinaryOp>() : BinaryOp();
      multiply && multiply.getOperatorKind() == BinaryOperator::Multiply) {
    if (queryParameter(multiply.getLhs()) == *parameter) coordinate = multiply.getRhs();
    else if (queryParameter(multiply.getRhs()) == *parameter) coordinate = multiply.getLhs();
  }
  auto result = dyn_cast_or_null<BlockArgument>(coordinate);
  auto mapping = result ? dyn_cast<ExecutionGroupOp>(result.getOwner()->getParentOp())
                        : ExecutionGroupOp();
  auto roles = mapping
                   ? mapping.getCoordinateRolesAttr()
                   : DenseI64ArrayAttr();
  if (!roles || result.getArgNumber() >= roles.size() ||
      roles[result.getArgNumber()] !=
          static_cast<int64_t>(CoordinateRole::PointwiseOwnership))
    return failure();
  return *parameter;
}

FailureOr<ParameterAttr> fullCoverageParameterForDimension(func::FuncOp kernel,
                                                         int64_t dimension) {
  if (dimension <= 0 || failed(dimensionArgument(kernel, dimension)))
    return failure();
  static constexpr int64_t candidates[] = {
      1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048,
      4096, 8192, 16384, 32768, 65536};
  auto reference = getOrCreatePhysicalParameter(
      kernel, ("REDUCE_FULL_D" + Twine(dimension)).str(),
      ParameterRole::OwnershipN, ParameterCategory::Coverage,
      /*elementBitWidth=*/0, candidates);
  if (failed(reference))
    return failure();
  auto parameter = lookupParameter(kernel, *reference);
  parameter = parameter.withBinding(parameter.getBinding().withDimension(
      IntegerAttr::get(IntegerType::get(kernel.getContext(), 64), dimension)));
  if (failed(updateParameter(kernel, parameter))) return failure();
  return parameter;
}

} // namespace

LogicalResult bindReductionFreeAxes(ReduceOp reduce, func::FuncOp kernel) {
  SmallVector<std::tuple<Value, PhysicalSourceAxis, int64_t>> pending;
  for (Value source : reduce.getSources()) {
    auto fragment = dyn_cast<FragmentType>(source.getType());
    if (!fragment)
      continue;
    for (auto [axis, attribute] : llvm::enumerate(fragment.getShape())) {
      if (llvm::is_contained(reduce.getAxes(), static_cast<int64_t>(axis)))
        continue;
      auto extent = cast<PhysicalExprAttr>(attribute);
      PhysicalAxisRealizationFact realization =
          PhysicalProgramAnalysis(kernel).axisRealization(source, axis);
      if (isCompileTimeExtent(extent) &&
          !realization.constructionScalarSeed)
        continue;
      auto mapping = cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
      if (realization.constructionScalarSeed ||
          extent.getKind() ==
              PhysicalExprKind::Dimension) {
        if (mapping.getDimensionId() <= 0)
          return reduce.emitOpError(
              "reduction free axis has no logical dimension authority");
        pending.emplace_back(
            source,
            PhysicalSourceAxis{mapping.getSourceId(), mapping.getSourceAxis(),
                           mapping.getDerived()},
            mapping.getDimensionId());
        continue;
      }
      if (extent.getKind() !=
          PhysicalExprKind::Parameter)
        return reduce.emitOpError(
            "reduction free axis has no physical parameter authority");
      FailureOr<ParameterAttr> parameter = parameterForExtent(kernel, extent);
      if (failed(parameter))
        return reduce.emitOpError(
            "reduction free axis has no physical parameter authority");
      ParameterRole role = parameter->getRole();
      bool ownership =
          role == ParameterRole::OwnershipM ||
          role == ParameterRole::OwnershipN;
      if (!ownership && !parameter->isDeferred())
        return reduce.emitOpError(
            "reduction free axis is neither ownership-blocked nor exact full coverage");
    }
  }

  for (auto [source, sourceAxis, dimension] : pending) {
    PhysicalProgramAnalysis analysis(kernel);
    PhysicalRangeFact ranges = analysis.sourceRanges(source, sourceAxis);
    FailureOr<ParameterAttr> parameter = parameterForDimension(kernel, dimension);
    bool projectedOwnership = false;
    PhysicalAxisProjection projection =
        queryFragmentAxis(source.getType(), sourceAxis);
    if (failed(parameter) && projection.isExact()) {
      PhysicalRangeFact projected =
          analysis.axisRanges(source, projection.fragmentAxis);
      if (projected.isExact() && projected.unitStep &&
          analysis.lockstepRanges(projected.roots).isExact()) {
        FailureOr<ParameterAttr> owner =
            parameterForOwnedRange(kernel, projected.roots.front());
        if (succeeded(owner) &&
            llvm::all_of(projected.roots, [&](MakeRangeOp range) {
              FailureOr<ParameterAttr> current =
                  parameterForOwnedRange(kernel, range);
              return succeeded(current) && *current == *owner;
            })) {
          parameter = *owner;
          ranges = std::move(projected);
          projectedOwnership = true;
        }
      }
    }
    bool fullCoverage = false;
    if (failed(parameter)) {
      parameter = fullCoverageParameterForDimension(kernel, dimension);
      fullCoverage = succeeded(parameter);
    }
    if (failed(parameter))
      return reduce.emitOpError(
                 "reduction free axis has neither prior ownership nor exact full-coverage authority")
             << "; dimension=" << dimension << "; source=" << source.getType()
             << "; source_id=" << sourceAxis.sourceId
             << "; source_axis=" << sourceAxis.sourceAxis;
    if (ranges.roots.empty()) {
      // A free axis introduced by a typed broadcast has no coordinate range of
      // its own.  Its dimension identity is nevertheless exact, so project
      // only this value flow onto the already selected ownership extent.
      if (failed(retargetDimensionExtent(source, dimension,
                                        selectedParameterExtent(*parameter))))
        return failure();
      if (fullCoverage)
        return reduce.emitOpError(
                   "reduction full-coverage free axis has no coordinate range for tail validity")
               << "; source_id=" << sourceAxis.sourceId
               << ", source_axis=" << sourceAxis.sourceAxis
               << ", dimension=" << dimension;
      continue;
    }
    if (ranges.state == PhysicalFactState::Unknown)
      return reduce.emitOpError(
                 "reduction free axis has no physical range projection")
             << "; source_id=" << sourceAxis.sourceId
             << ", source_axis=" << sourceAxis.sourceAxis
             << ", dimension=" << dimension;
    MakeRangeOp authority = ranges.roots.front();
    if (!llvm::all_of(ranges.roots, [&](MakeRangeOp range) {
          return sameLogicalRange(authority, range);
        }))
      return reduce.emitOpError(
                 "reduction free axis has conflicting physical range projections")
             << "; source_id=" << sourceAxis.sourceId
             << ", source_axis=" << sourceAxis.sourceAxis
             << ", dimension=" << dimension;
    for (MakeRangeOp range : ranges.roots) {
      LogicalResult retargeted =
          projectedOwnership
              ? retargetSourceExtent(range.getResult(), sourceAxisIdentity(range),
                                     selectedParameterExtent(*parameter))
              : retargetDimensionExtent(range.getResult(), dimension,
                                        selectedParameterExtent(*parameter));
      if (failed(retargeted))
        return failure();
    }
    if (projectedOwnership &&
        failed(retargetSourceExtent(source, sourceAxis,
                                    selectedParameterExtent(*parameter))))
      return failure();
    if (fullCoverage &&
        failed(bindFullCoverageDimension(kernel, dimension,
                                         parameterValue(kernel, *parameter))))
      return reduce.emitOpError(
                 "reduction free axis full-coverage binding failed")
             << "; dimension=" << dimension;
  }

  auto capabilities = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  if (capabilities && capabilities.getRegistersPerUnit() > 0) {
    for (Value source : reduce.getSources()) {
      auto fragment = dyn_cast<FragmentType>(source.getType());
      if (!fragment)
        continue;
      for (auto [axis, attribute] : llvm::enumerate(fragment.getShape())) {
        if (llvm::is_contained(reduce.getAxes(), static_cast<int64_t>(axis)))
          continue;
        auto parameter = parameterForExtent(kernel, cast<PhysicalExprAttr>(attribute));
        if (failed(parameter))
          continue;
        auto schema = *parameter;
        auto role = schema.getRole();
        if (role != ParameterRole::OwnershipM && role != ParameterRole::OwnershipN)
          continue;
        Type element = fragment.getElementType();
        unsigned bits = element.isIndex() ? 64 : element.getIntOrFloatBitWidth();
        __int128 registers = std::max(1u, (bits + 31) / 32);
        bool known = true;
        for (auto [otherAxis, otherAttribute] : llvm::enumerate(fragment.getShape())) {
          if (axis == otherAxis)
            continue;
          auto extent = cast<PhysicalExprAttr>(otherAttribute);
          if (extent.getKind() != PhysicalExprKind::Constant ||
              extent.getValue() <= 0) {
            known = false;
            break;
          }
          registers *= extent.getValue();
        }
        if (!known)
          continue;
        SmallVector<int64_t> candidates;
        for (int64_t candidate : schema.getCandidates().asArrayRef())
          if (registers * candidate <= capabilities.getRegistersPerUnit())
            candidates.push_back(candidate);
        if (!candidates.empty() &&
            candidates.size() != static_cast<size_t>(schema.getCandidates().size()))
          if (failed(updateParameter(kernel, schema.withCandidates(
                  DenseI64ArrayAttr::get(kernel.getContext(), candidates)))))
            return failure();
      }
    }
  }
  return closeValueRelations(kernel, ValueRelationScope::ReductionInputs);
}

} // namespace intent::gpu::reduction
