#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Interfaces/StructuredOpInterface.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include <algorithm>
#include <functional>
#include <optional>

using namespace mlir;

namespace intent::gpu {

SmallVector<StructuredSchemaGroup>
queryStructuredSchemaGroups(Operation *operation) {
  if (!isa<RegionFoldOp, RegionScanOp>(operation)) return {};
  auto structured = cast<StructuredOpInterface>(operation);
  auto fold = dyn_cast<RegionFoldOp>(operation);
  auto scan = dyn_cast<RegionScanOp>(operation);
  auto identities = fold ? fold.getIdentities() : scan.getIdentities();
  SmallVector<StructuredSchemaGroup> groups;
  for (unsigned index = 0; index < identities.size(); ++index) {
    StructuredSchemaGroup group{
        &structured.getSummarizeRegion()->front().getTerminator()->getOpOperand(index),
        identities.getBeginOperandIndex() + index,
        {structured.getCombineLhs()[index], structured.getCombineRhs()[index]}, {},
        {&structured.getCombine().front().getTerminator()->getOpOperand(index)}};
    if (fold) group.results.push_back(fold.getResult(index));
    else group.arguments.push_back(structured.getApplySummaries()[index]);
    groups.push_back(std::move(group));
  }
  if (scan) {
    auto initial = scan.getInitialStates();
    for (unsigned index = 0; index < initial.size(); ++index)
      groups.push_back({&structured.getApplyRegion()->front().getTerminator()->getOpOperand(index),
          initial.getBeginOperandIndex() + index,
          {structured.getApplyStates()[index], structured.getEmitStates()[index]},
          {scan.getFinalStates()[index]}, {}});
  }
  return groups;
}

FailureOr<unsigned> queryRegionScanEmissionAxis(
    RegionScanOp scan, unsigned output, ArrayRef<unsigned> fieldPath) {
  auto structured = cast<StructuredOpInterface>(scan.getOperation());
  if (!llvm::hasSingleElement(scan.getEmit()) ||
      scan.getEmit().front().empty() ||
      output >= scan.getEmittedResults().size())
    return failure();
  auto yields = structured.getEmitYields();
  if (output >= yields.size())
    return failure();
  auto fieldType = [&](Type type) -> Type {
    for (unsigned field : fieldPath) {
      auto record = dyn_cast<RecordType>(type);
      if (!record || field >= record.getFieldTypes().size())
        return {};
      type = cast<TypeAttr>(record.getFieldTypes()[field]).getValue();
    }
    return type;
  };
  auto slice = dyn_cast_or_null<FragmentType>(
      fieldType(yields[output].getType()));
  auto result = dyn_cast_or_null<FragmentType>(
      fieldType(scan.getEmittedResults()[output].getType()));
  if (!slice || !result || slice.getElementType() != result.getElementType() ||
      slice.getShape().size() != result.getShape().size() ||
      slice.getAxisMaps() != result.getAxisMaps())
    return failure();

  std::optional<unsigned> memberAxis;
  if (scan.getEmit().front().getNumArguments() < scan.getSources().size())
    return failure();
  for (BlockArgument source : structured.getEmitSources()) {
    auto type = dyn_cast<FragmentType>(source.getType());
    if (!type || scan.getAxis() >= type.getShape().size())
      return failure();
    auto member = cast<AxisMapAttr>(type.getAxisMaps()[scan.getAxis()]);
    for (auto [axis, attribute] : llvm::enumerate(slice.getAxisMaps())) {
      auto mapping = cast<AxisMapAttr>(attribute);
      if (!(sourceAxisIdentity(mapping) == sourceAxisIdentity(member)) ||
          mapping.getDimensionId() != member.getDimensionId())
        continue;
      if (memberAxis && *memberAxis != axis)
        return failure();
      memberAxis = axis;
    }
  }
  return memberAxis ? FailureOr<unsigned>(*memberAxis)
                    : FailureOr<unsigned>(failure());
}

FailureOr<func::FuncOp> getPhysicalKernel(ModuleOp module) {
  SmallVector<func::FuncOp> kernels;
  for (func::FuncOp function : module.getOps<func::FuncOp>())
    if (function->hasAttr(kernelAttr))
      kernels.push_back(function);
  if (kernels.size() != 1) {
    module.emitError("shared GPU module requires exactly one physical kernel");
    return failure();
  }
  return kernels.front();
}

bool isShapeBound(PhysicalExprAttr bound) {
  return bound.getKind() != PhysicalExprKind::ScalarABI &&
         llvm::all_of(bound.getOperands(), [](Attribute operand) {
           return isShapeBound(cast<PhysicalExprAttr>(operand));
         });
}

Type scalarCallbackType(Type type) {
  if (auto fragment = dyn_cast<FragmentType>(type))
    return fragment.getElementType();
  if (auto record = dyn_cast<RecordType>(type)) {
    SmallVector<Attribute> fields;
    for (Attribute field : record.getFieldTypes())
      fields.push_back(TypeAttr::get(
          scalarCallbackType(cast<TypeAttr>(field).getValue())));
    return RecordType::get(type.getContext(), record.getFieldNames(),
                           ArrayAttr::get(type.getContext(), fields),
                           record.getOwner());
  }
  return type;
}

bool canPredicateValueOperation(Operation *operation) {
  if (isa<LoadOp, GatherOp>(operation))
    return true;
  if (auto binary = dyn_cast<BinaryOp>(operation)) {
    auto kind = binary.getOperatorKind();
    if (kind == BinaryOperator::FloorDivide ||
        kind == BinaryOperator::Remainder) {
      // Physical chunk counts divide by a positive compile-time width. Such
      // scalar arithmetic remains defined in an inactive predicated branch.
      APInt divisor;
      bool positive = matchPattern(binary.getRhs(), m_ConstantInt(&divisor)) &&
                      divisor.isStrictlyPositive();
      if (auto parameter = binary.getRhs().getDefiningOp<ParameterOp>()) {
        auto candidates = parameter.getDeclaration().getCandidates().asArrayRef();
        positive = !candidates.empty() && llvm::all_of(
            candidates, [](int64_t value) { return value > 0; });
      }
      if (!positive)
        return false;
    }
    if (kind == BinaryOperator::LeftShift ||
        kind == BinaryOperator::RightShift)
      return false;
  }
  auto elementType = [](Type type) {
    auto fragment = dyn_cast<FragmentType>(type);
    return fragment ? fragment.getElementType() : type;
  };
  if (auto cast = dyn_cast<CastOp>(operation)) {
    Type source = elementType(cast.getValue().getType());
    Type result = elementType(cast.getType());
    if ((isa<FloatType>(source) && !isa<FloatType>(result)) ||
        isa<Float8E4M3FNType>(result))
      return false;
  }
  return isa<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp,
             SplatOp, BroadcastOp, MakeRangeOp, DimOp, PhysicalExprOp,
             arith::ConstantOp>(operation) &&
         isSpeculatable(operation) && isMemoryEffectFree(operation);
}

bool variesWithIteration(Value value, scf::ForOp loop,
                        llvm::DenseMap<Value, bool> &known) {
  if (loop.isDefinedOutsideOfLoop(value))
    return false;
  if (auto found = known.find(value); found != known.end())
    return found->second;
  bool varies = true;
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    auto nested = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
    if (nested && nested != loop && argument == nested.getInductionVar())
      varies = variesWithIteration(nested.getLowerBound(), loop, known) ||
               variesWithIteration(nested.getUpperBound(), loop, known) ||
               variesWithIteration(nested.getStep(), loop, known);
  } else if (Operation *producer = value.getDefiningOp();
             producer && !producer->getNumRegions() &&
             canPredicateValueOperation(producer) &&
             !isa<LoadOp, GatherOp>(producer)) {
    varies = llvm::any_of(producer->getOperands(), [&](Value operand) {
      return variesWithIteration(operand, loop, known);
    });
  }
  known[value] = varies;
  return varies;
}

std::pair<uint64_t, int64_t> nextPhysicalAxisIdentities(func::FuncOp kernel) {
  uint64_t nextSource = 1;
  int64_t nextDimension = 1;
  AttrTypeWalker identities;
  identities.addWalk([&](AxisMapAttr axis) {
    nextSource = std::max(nextSource, axis.getSourceId() + 1);
    nextDimension = std::max(nextDimension, axis.getDimensionId() + 1);
  });
  identities.addWalk([&](PhysicalSourceAttr source) {
    nextSource = std::max(nextSource, source.getSourceId() + 1);
  });
  identities.addWalk([&](ParameterBindingAttr binding) {
    if (auto dimension = binding.getDimension())
      nextDimension = std::max(nextDimension, dimension.getInt() + 1);
  });
  identities.addWalk([&](ViewType view) {
    nextSource = std::max(nextSource, view.getSourceId() + 1);
  });
  identities.addWalk([&](ViewLayoutAttr layout) {
    for (int64_t dimension : layout.getDimensionIds().asArrayRef())
      nextDimension = std::max(nextDimension, dimension + 1);
  });
  identities.addWalk([&](PhysicalExprAttr expression) {
    if (expression.getKind() == PhysicalExprKind::Dimension)
      nextDimension = std::max(nextDimension, expression.getValue() + 1);
  });
  kernel.walk([&](Operation *operation) {
    identities.walk(operation->getAttrDictionary());
    for (Type type : operation->getOperandTypes()) identities.walk(type);
    for (Type type : operation->getResultTypes()) identities.walk(type);
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          identities.walk(argument.getType());
    if (auto dimension = operation->getAttrOfType<IntegerAttr>(dimensionAttr))
      nextDimension = std::max(nextDimension, dimension.getInt() + 1);
  });
  return {nextSource, nextDimension};
}

PhysicalExprAttr multiplyExtent(PhysicalExprAttr lhs, PhysicalExprAttr rhs) {
  auto leftKind = lhs.getKind();
  auto rightKind = rhs.getKind();
  if (leftKind == PhysicalExprKind::Constant && lhs.getValue() == 1)
    return rhs;
  if (rightKind == PhysicalExprKind::Constant && rhs.getValue() == 1)
    return lhs;
  if (leftKind == PhysicalExprKind::Constant &&
      rightKind == PhysicalExprKind::Constant)
    return PhysicalExprAttr::get(
        lhs.getContext(), PhysicalExprKind::Constant,
        lhs.getValue() * rhs.getValue(), StringAttr::get(lhs.getContext()),
        ArrayAttr::get(lhs.getContext(), {}));
  return PhysicalExprAttr::get(
      lhs.getContext(), PhysicalExprKind::Multiply, 0,
      StringAttr::get(lhs.getContext()),
      ArrayAttr::get(lhs.getContext(), {lhs, rhs}));
}

PhysicalExprAttr productExtent(MLIRContext *context,
                               ArrayRef<Attribute> shape,
                               ArrayRef<int64_t> axes, unsigned prefix) {
  PhysicalExprAttr product = PhysicalExprAttr::get(
      context, PhysicalExprKind::Constant, 1,
      StringAttr::get(context), ArrayAttr::get(context, {}));
  for (int64_t axis : axes)
    product = multiplyExtent(
        product, cast<PhysicalExprAttr>(shape[prefix + axis]));
  return product;
}

bool isIntroducedReshapeUnitAxis(Value value, unsigned fragmentAxis) {
  auto reshape = value.getDefiningOp<ReshapeOp>();
  if (!reshape) return false;
  auto relations = queryFragmentOperandRelations(reshape.getOperation());
  return succeeded(relations) && relations->front().isIntroducedUnitAxis(fragmentAxis);
}

std::optional<unsigned> reshapeInputAxis(ReshapeOp reshape,
                                         unsigned resultAxis) {
  auto relations = queryFragmentOperandRelations(reshape.getOperation());
  return succeeded(relations) ? relations->front().correspondingSourceAxis(resultAxis)
                              : std::nullopt;
}

Value stripAdditiveProjection(Value value, bool singleUse) {
  auto elementType = [](Type type) {
    auto fragment = dyn_cast<FragmentType>(type);
    return fragment ? fragment.getElementType() : type;
  };
  while (value) {
    if (singleUse && !value.hasOneUse())
      return {};
    Operation *operation = value.getDefiningOp();
    if (!operation)
      break;
    if (auto cast = dyn_cast<CastOp>(operation)) {
      if (elementType(cast.getValue().getType()) !=
          elementType(cast.getResult().getType()))
        break;
    } else if (!isa<BroadcastOp, ReshapeOp, TransposeOp>(operation)) {
      break;
    }
    value = operation->getOperand(0);
  }
  return value;
}

bool isLiteralZeroProjection(Value value) {
  while (Operation *operation = value.getDefiningOp()) {
    if (auto extract = dyn_cast<ExtractOp>(operation)) {
      auto record = extract.getRecord().getDefiningOp<MakeRecordOp>();
      if (!record)
        return false;
      value = record.getFields()[extract.getField()];
      continue;
    }
    if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
      if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
        return integer.getValue().isZero();
      if (auto floating = dyn_cast<FloatAttr>(constant.getValue()))
        return floating.getValue().isZero();
      return false;
    }
    if (!isa<SplatOp, BroadcastOp, ReshapeOp, TransposeOp, CastOp>(operation))
      return false;
    value = operation->getOperand(0);
  }
  return false;
}

FailureOr<FragmentType> queryValueSchema(func::FuncOp kernel,
                                             FragmentType target,
                                             ValueRange contributors) {
  SmallVector<Attribute> shape(target.getShape().begin(), target.getShape().end());
  SmallVector<Attribute> mappings(target.getAxisMaps().begin(),
                                  target.getAxisMaps().end());
  SmallVector<bool> refined(shape.size(), false);
  SmallVector<Value> authorities(shape.size());
  SmallVector<unsigned> authorityAxes(shape.size(), 0);
  bool changed = false;
  PhysicalProgramAnalysis analysis(kernel);
  for (Value contributor : contributors) {
    auto source = dyn_cast<FragmentType>(contributor.getType());
    if (!source)
      continue;
    // An axis replicated through a pure value expression follows the
    // consumer's coordinates; other axes may still own an independent
    // traversal. Projection materializes the selected schema without
    // retargeting the original producer or its other users.
    auto uniformAxes = uniformFragmentAxes(contributor);
    SmallVector<bool> sourceAuthority(source.getShape().size(), false);
    bool hasAuthority = false;
    for (unsigned sourceAxis = 0; sourceAxis < source.getShape().size();
         ++sourceAxis) {
      if (uniformAxes[sourceAxis])
        continue;
      PhysicalAxisRealizationFact realization =
          analysis.axisRealization(contributor, sourceAxis);
      auto sourceExtent =
          cast<PhysicalExprAttr>(source.getShape()[sourceAxis]);
      bool singleton =
          sourceExtent.getKind() ==
              PhysicalExprKind::Constant &&
          sourceExtent.getValue() == 1;
      // A logical singleton may broadcast. A selected one-lane slice of a
      // larger logical range instead owns that consumer's physical extent.
      sourceAuthority[sourceAxis] = realization.hasExtentAuthority() &&
                                    !realization.constructionScalarSeed &&
                                    (!singleton ||
                                     (realization.physicalized &&
                                      llvm::any_of(realization.roots, [](MakeRangeOp range) {
                                        return !isProvablySingletonLogicalRange(range);
                                      })));
      hasAuthority |= sourceAuthority[sourceAxis];
    }
    if (!hasAuthority)
      continue;
    BroadcastProjection projection = queryAxisProjection(source, target);
    if (!projection.isExact())
      return failure();
    for (auto [targetAxis, sourceAxis] :
         llvm::enumerate(projection.targetToSource)) {
      if (!sourceAxis)
        continue;
      if (!sourceAuthority[*sourceAxis])
        continue;
      Attribute extent = source.getShape()[*sourceAxis];
      auto sourceMapping = cast<AxisMapAttr>(source.getAxisMaps()[*sourceAxis]);
      auto candidate = AxisMapAttr::get(
          target.getContext(), sourceMapping.getSourceId(),
          sourceMapping.getSourceAxis(), sourceMapping.getDimensionId(),
          targetAxis, sourceMapping.getDerived());
      if (refined[targetAxis]) {
        if (shape[targetAxis] != extent)
          return failure();
        auto selected = cast<AxisMapAttr>(mappings[targetAxis]);
        if (selected == candidate)
          continue;
        // A derived axis only records a broadcast occurrence.  Once an exact
        // non-derived source relation reaches the same physical axis, it is the
        // unique coordinate authority.  Two unrelated direct sources (or two
        // unrelated derived occurrences) remain ambiguous unless the canonical
        // result already selected one of them.
        if (selected.getDerived() != candidate.getDerived()) {
          if (!candidate.getDerived()) {
            mappings[targetAxis] = candidate;
            authorities[targetAxis] = contributor;
            authorityAxes[targetAxis] = *sourceAxis;
            changed = true;
          }
          continue;
        }
        auto canonical = cast<AxisMapAttr>(target.getAxisMaps()[targetAxis]);
        if (selected == canonical)
          continue;
        if (candidate == canonical) {
          mappings[targetAxis] = candidate;
          changed = true;
          continue;
        }
        // Distinct coordinate SSA graphs may be lockstep occurrences of one
        // result axis (for example row and column advanced indices driven by
        // the same filter traversal).  Producer identity and the intermediate
        // value's dimension are not enough to decide that relation.  Consume
        // the shared range analysis and retain the result's canonical axis only
        // when both current traversals are proven identical.
        if (authorities[targetAxis]) {
          SmallVector<Value> sources = {authorities[targetAxis], contributor};
          SmallVector<unsigned> axes = {authorityAxes[targetAxis], *sourceAxis};
          PhysicalLockstepTraversalFact lockstep =
              analysis.lockstepTraversal(sources, axes);
          if (lockstep.isExact()) {
            mappings[targetAxis] = canonical;
            changed = true;
            continue;
          }
        }
        return failure();
      }
      shape[targetAxis] = extent;
      mappings[targetAxis] = candidate;
      authorities[targetAxis] = contributor;
      authorityAxes[targetAxis] = *sourceAxis;
      refined[targetAxis] = true;
      changed = true;
    }
  }
  if (!changed)
    return target;
  return FragmentType::get(target.getContext(), target.getElementType(),
                           ArrayAttr::get(target.getContext(), shape),
                           ArrayAttr::get(target.getContext(), mappings),
                           target.getValidity(), target.getOwner());
}

FailureOr<FragmentType> queryAccessResultSchema(
    func::FuncOp kernel, AccessOpInterface access) {
  auto target = cast<FragmentType>(access.getAccessValueType());
  SmallVector<Attribute> shape(target.getShape().begin(),
                               target.getShape().end());
  SmallVector<bool> refined(shape.size(), false);
  bool changed = false;
  PhysicalProgramAnalysis analysis(kernel);
  for (auto [coordinateIndex, coordinate] : llvm::enumerate(access.getAccessCoordinates())) {
    auto source = dyn_cast<FragmentType>(coordinate.getType());
    if (!source)
      continue;
    auto projection = queryAccessCoordinateAxes(access, coordinateIndex);
    if (projection.state == BroadcastProjectionState::Ambiguous)
      return failure();
    for (unsigned sourceAxis = 0; sourceAxis < source.getShape().size();
         ++sourceAxis) {
      PhysicalAxisRealizationFact realization =
          analysis.axisRealization(coordinate, sourceAxis);
      auto extent = cast<PhysicalExprAttr>(source.getShape()[sourceAxis]);
      bool singleton =
          extent.getKind() ==
              PhysicalExprKind::Constant &&
          extent.getValue() == 1;
      PhysicalRangeFact ranges = analysis.axisRanges(coordinate, sourceAxis);
      FailureOr<MakeRangeOp> range = queryExactLogicalRange(ranges);
      bool scalarCoordinate = singleton && succeeded(range) &&
                              (*range).getResult() == coordinate &&
                              isUnitStepRange(*range) &&
                              queryLaunchExpression((*range).getExtent()) == extent;
      if (!scalarCoordinate &&
          (!realization.hasExtentAuthority() ||
           realization.constructionScalarSeed || singleton))
        continue;

      std::optional<unsigned> targetAxis;
      for (auto [axis, from] : llvm::enumerate(projection.targetToSource))
        if (from && *from == sourceAxis)
          targetAxis = axis;
      // A coordinate may carry an ownership axis that the indexed result does
      // not expose.  Such an axis is not an access-result extent authority.
      if (!targetAxis)
        continue;
      if (refined[*targetAxis] && shape[*targetAxis] != extent) {
        emitError(coordinate.getLoc(), "access coordinate relations disagree")
            << "; payload_axis=" << *targetAxis
            << "; selected_extent=" << shape[*targetAxis]
            << "; coordinate_extent=" << extent
            << "; coordinate=" << coordinate;
        return failure();
      }
      shape[*targetAxis] = extent;
      refined[*targetAxis] = true;
      changed |= target.getShape()[*targetAxis] != extent;
    }
  }
  if (!changed)
    return target;
  // The index relation selected the access-result coordinate identity during
  // KIR-to-GPU construction.  Coordinates refine only its physical extents;
  // replacing those axis maps from a rank-aligned broadcast would create a
  // second, and potentially different, index relation.
  return FragmentType::get(target.getContext(), target.getElementType(),
                           ArrayAttr::get(target.getContext(), shape),
                           target.getAxisMaps(), target.getValidity(),
                           target.getOwner());
}

FailureOr<uint64_t> blockedDimension(Attribute attribute) {
  auto extent = dyn_cast<PhysicalExprAttr>(attribute);
  if (!extent ||
      extent.getKind() !=
          PhysicalExprKind::CeilDiv ||
      extent.getOperands().size() != 2)
    return failure();
  auto logical = dyn_cast<PhysicalExprAttr>(extent.getOperands()[0]);
  auto block = dyn_cast<PhysicalExprAttr>(extent.getOperands()[1]);
  if (!logical || !block ||
      logical.getKind() !=
          PhysicalExprKind::Dimension ||
      block.getKind() !=
          PhysicalExprKind::Parameter)
    return failure();
  return logical.getValue() > 0
             ? FailureOr<uint64_t>(logical.getValue())
             : FailureOr<uint64_t>(failure());
}

bool hasBlockedDimension(func::FuncOp kernel, uint64_t dimension) {
  bool found = false;
  kernel.walk([&](ExecutionGroupOp mapping) {
    for (Attribute extent : mapping.getLaunchExtents()) {
      FailureOr<uint64_t> blocked = blockedDimension(extent);
      found |= succeeded(blocked) && *blocked == dimension;
    }
  });
  return found;
}

} // namespace intent::gpu
