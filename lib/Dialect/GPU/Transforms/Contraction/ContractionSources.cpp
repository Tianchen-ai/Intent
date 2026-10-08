#include "ContractionDetail.h"
#include "../Value/SourceReplay.h"
#include "Intent/Dialect/GPU/Analysis/Helpers.h"
#include "Intent/Analysis/ContractionAxes.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
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

static ReshapeOp exposeTransposedContractSplit(ContractOp contract);

static void orientContractionOutputs(func::FuncOp kernel);

FailureOr<bool> rewriteContractionSnapshot(
    func::FuncOp kernel,
    llvm::function_ref<FailureOr<bool>(ContractOp)> rewrite) {
  SmallVector<ContractOp> contracts;
  kernel.walk([&](ContractOp contract) { contracts.push_back(contract); });
  bool changed = false;
  for (ContractOp contract : contracts) {
    auto result = rewrite(contract);
    if (failed(result))
      return failure();
    changed |= *result;
  }
  return changed;
}

void fuseContractionAdds(func::FuncOp kernel) {
  SmallVector<BinaryOp> additions;
  kernel.walk([&](BinaryOp binary) {
    if (binary.getOperatorKind() == BinaryOperator::Add &&
        !binary.getStrictRounding())
      additions.push_back(binary);
  });
  for (BinaryOp add : additions) {
    for (unsigned operand : {1u, 0u}) {
      Value projected = add->getOperand(operand);
      if (projected.getType() != add.getResult().getType())
        continue;
      Value value =
          stripAdditiveProjection(projected, /*singleUse=*/true);
      auto contract = value ? value.getDefiningOp<ContractOp>() : ContractOp();
      if (!contract || contract->getBlock() != add->getBlock() ||
          !isLiteralZeroProjection(contract.getAccumulator()))
        continue;
      SmallVector<Operation *> projections;
      for (Value current = projected; current != value;) {
        Operation *projection = current.getDefiningOp();
        projections.push_back(projection);
        current = projection->getOperand(0);
      }
      OpBuilder builder(add);
      FailureOr<Value> carry = projectPhysicalValueToSchema(
          builder, add.getLoc(), add->getOperand(1 - operand),
          contract.getResult().getType());
      if (failed(carry))
        continue;
      // Clone only the pure contraction at its consumer. Inputs already
      // dominate this point; memory reads and other effects are not moved.
      auto fused = cast<ContractOp>(builder.clone(*contract));
      fused.getAccumulatorMutable().assign(*carry);
      IRMapping mapping;
      mapping.map(value, fused.getResult());
      Value result = fused.getResult();
      for (Operation *projection : llvm::reverse(projections)) {
        Operation *clone = builder.clone(*projection, mapping);
        result = clone->getResult(0);
        mapping.map(projection->getResult(0), result);
      }
      add.getResult().replaceAllUsesWith(result);
      add.erase();
      break;
    }
  }
  eraseDeadPhysicalValues(kernel);
}

FragmentType eraseFragmentAxis(FragmentType source, unsigned erasedAxis) {
  SmallVector<Attribute> shape;
  SmallVector<Attribute> mappings;
  for (auto [axis, extent] : llvm::enumerate(source.getShape())) {
    if (axis == erasedAxis)
      continue;
    shape.push_back(extent);
    auto mapping = cast<AxisMapAttr>(source.getAxisMaps()[axis]);
    mappings.push_back(AxisMapAttr::get(
        source.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
        mapping.getDimensionId(), mappings.size(), mapping.getDerived()));
  }
  return FragmentType::get(source.getContext(), source.getElementType(),
                           ArrayAttr::get(source.getContext(), shape),
                           ArrayAttr::get(source.getContext(), mappings),
                           source.getValidity(), source.getOwner());
}

SmallVector<int64_t> eraseAxis(ArrayRef<int64_t> axes, unsigned erasedAxis) {
  SmallVector<int64_t> result;
  result.reserve(axes.size());
  for (int64_t axis : axes) {
    if (axis == static_cast<int64_t>(erasedAxis))
      continue;
    result.push_back(axis > static_cast<int64_t>(erasedAxis) ? axis - 1 : axis);
  }
  return result;
}

FailureOr<bool> collapseMultiReductionContract(ContractOp contract) {
  SmallVector<int64_t> lhsAxes(contract.getLhsReductionAxes());
  SmallVector<int64_t> rhsAxes(contract.getRhsReductionAxes());
  if (lhsAxes.size() <= 1 || lhsAxes.size() != rhsAxes.size() ||
      !contract.getLhsBatchAxes().empty() || !contract.getRhsBatchAxes().empty())
    return false;
  if (!contract.getLhs().getDefiningOp<LoadOp>() ||
      !contract.getRhs().getDefiningOp<LoadOp>())
    return false;
  func::FuncOp kernel = contract->getParentOfType<func::FuncOp>();
  PhysicalProgramAnalysis analysis(kernel);
  auto unit = expression(kernel.getContext(), PhysicalExprKind::Constant, 1);
  for (Value operand : {contract.getLhs(), contract.getRhs()}) {
    auto load = operand.getDefiningOp<LoadOp>();
    if (!isa<ViewType>(load.getResource().getType()))
      return false;
    auto type = cast<FragmentType>(operand.getType());
    llvm::DenseMap<Operation *, unsigned> rootAxes;
    for (unsigned axis = 0; axis < type.getShape().size(); ++axis) {
      PhysicalRangeFact ranges = analysis.axisRanges(operand, axis);
      auto realization = analysis.axisRealization(operand, axis);
      bool introducedUnit = ranges.roots.empty() && realization.isExact() &&
                            !realization.constructionScalarSeed &&
                            type.getShape()[axis] == unit;
      if (ranges.state != PhysicalFactState::Exact && !introducedUnit)
        return false;
      for (MakeRangeOp root : ranges.roots) {
        auto [found, inserted] = rootAxes.try_emplace(root.getOperation(), axis);
        if (!inserted && found->second != axis)
          return false;
      }
    }
  }
  SmallVector<PhysicalExprAttr> reductionExtents;
  for (auto [lhsAxis, rhsAxis] : llvm::zip(lhsAxes, rhsAxes)) {
    SmallVector<MakeRangeOp> paired;
    for (auto [operand, axis] :
         {std::pair<Value, int64_t>{contract.getLhs(), lhsAxis},
          std::pair<Value, int64_t>{contract.getRhs(), rhsAxis}}) {
      auto load = operand.getDefiningOp<LoadOp>();
      if (!canReplayReadAt(load, contract))
        return false;
      PhysicalRangeFact ranges = analysis.axisRanges(operand, axis);
      if (failed(queryExactLogicalRange(ranges)))
        return false;
      auto projection = analysis.rangeAxes(operand, ranges.roots);
      if (!projection.isExact() ||
          projection.fragmentAxes != ArrayRef<unsigned>{static_cast<unsigned>(axis)})
        return false;
      auto dimension = queryRangeDimension(ranges.roots.front());
      if (failed(dimension))
        return false;
      auto source = sourceAxisIdentity(ranges.roots.front());
      auto replay = analysis.replayAt(
          operand, source, PhysicalReplayScope::ValueGraph,
          /*allowAccesses=*/true, contract, IRMapping{}, *dimension);
      if (!replay.isReplayable())
        return false;
      for (MakeRangeOp range : ranges.roots) {
        auto realization = analysis.axisRealization(range.getResult(), 0);
        auto currentDimension = queryRangeDimension(range);
        auto cardinality = constantLogicalRangeCardinality(range);
        auto physical = constantPhysicalExpression(
            cast<PhysicalExprAttr>(range.getResult().getType().getShape()[0]));
        bool complete = cardinality && physical && *physical >= *cardinality;
        if (!isZeroScalar(range.getStart()) ||
            !isZeroScalar(range.getLogicalStart()) || !isUnitStepRange(range) ||
            range->hasAttr(sourceSubregionAttr) ||
            !queryLaunchExpression(range.getLogicalStop()) ||
            !(sourceAxisIdentity(range) == source) || failed(currentDimension) ||
            *currentDimension != *dimension ||
            (!complete && !realization.constructionScalarSeed &&
             !samePhysicalScalarExpression(range.getExtent(),
                                           range.getLogicalStop())))
          return false;
        paired.push_back(range);
      }
    }
    if (!analysis.lockstepRanges(paired).isExact())
      return false;
    reductionExtents.push_back(
        queryLaunchExpression(paired.front().getLogicalStop()));
  }

  // Put a complete power-of-two tile inside a non-power-of-two outer group.
  // The outer coordinates then stay constant across aligned K lanes, allowing
  // the provider's axis analysis to share their delinearization arithmetic.
  auto tail = reductionExtents.back();
  if (tail.getKind() == PhysicalExprKind::Constant &&
      tail.getValue() > 0 && !llvm::isPowerOf2_64(tail.getValue())) {
    unsigned inner = lhsAxes.size() - 1;
    int64_t largest = 0;
    for (auto [position, extent] : llvm::enumerate(reductionExtents)) {
      if (extent.getKind() !=
              PhysicalExprKind::Constant ||
          extent.getValue() < contractionReductionCandidates[0] ||
          !llvm::isPowerOf2_64(extent.getValue()) ||
          extent.getValue() <= largest)
        continue;
      inner = position;
      largest = extent.getValue();
    }
    std::rotate(lhsAxes.begin() + inner, lhsAxes.begin() + inner + 1,
                lhsAxes.end());
    std::rotate(rhsAxes.begin() + inner, rhsAxes.begin() + inner + 1,
                rhsAxes.end());
  }

  auto [sourceId, dimensionId] = nextPhysicalAxisIdentities(kernel);
  OpBuilder builder(contract);
  // Expand only the contraction's replayable operands, using logical extents
  // before flattening. Padding each source axis would change the flattened
  // coordinate relation; the eventual K tile owns the single tail instead.
  auto complete = [&](Value value, ArrayRef<int64_t> axes) -> FailureOr<Value> {
    for (int64_t axis : axes) {
      auto ranges = PhysicalProgramAnalysis(kernel).axisRanges(value, axis);
      MakeRangeOp root = ranges.roots.front();
      if (llvm::all_of(ranges.roots, [](MakeRangeOp range) {
            return samePhysicalScalarExpression(range.getExtent(),
                                                range.getLogicalStop());
          }))
        continue;
      auto extent = queryLaunchExpression(root.getLogicalStop());
      auto original = root.getResult().getType();
      auto type = FragmentType::get(
          kernel.getContext(), original.getElementType(),
          builder.getArrayAttr({extent}), original.getAxisMaps(),
          original.getValidity(), original.getOwner());
      Value full = builder.create<MakeRangeOp>(
          contract.getLoc(), type, root.getStart(), root.getLogicalStop(),
          root.getStep(), root.getLogicalStart(), root.getLogicalStop(),
          root.getSourceId(), root.getSourceAxis(), root.getDerived());
      inheritRangeAuthority(full, root);
      IRMapping mapping;
      auto replayed = materializeSourceRanges(builder, contract.getLoc(), value,
                                       extent, ranges.roots, full, mapping,
                                       contract.getOperation());
      if (failed(replayed))
        return failure();
      value = *replayed;
    }
    return value;
  };
  auto completeLhs = complete(contract.getLhs(), lhsAxes);
  auto completeRhs = complete(contract.getRhs(), rhsAxes);
  if (failed(completeLhs) || failed(completeRhs))
    return contract.emitOpError("cannot replay complete paired reduction ranges");
  auto collapse = [&](Value value, ArrayRef<int64_t> reductionAxes)
      -> std::pair<Value, int64_t> {
    SmallVector<int64_t> reductions(reductionAxes);
    auto source = cast<FragmentType>(value.getType());
    bool contiguous = llvm::all_of(
        llvm::enumerate(reductions), [&](auto entry) {
          return entry.value() == reductions.front() +
                                      static_cast<int64_t>(entry.index());
        });
    if (!contiguous) {
      // Preserve free-axis order and use the same paired-reduction order on
      // both operands to make their exact ranges one collapsible group.
      SmallVector<int64_t> permutation;
      for (unsigned axis = 0; axis < source.getShape().size(); ++axis)
        if (!llvm::is_contained(reductions, static_cast<int64_t>(axis)))
          permutation.push_back(axis);
      int64_t firstReduction = permutation.size();
      llvm::append_range(permutation, reductions);
      SmallVector<Attribute> transposedShape, transposedMappings;
      for (auto [resultAxis, sourceAxis] : llvm::enumerate(permutation)) {
        transposedShape.push_back(source.getShape()[sourceAxis]);
        auto mapping = cast<AxisMapAttr>(source.getAxisMaps()[sourceAxis]);
        transposedMappings.push_back(AxisMapAttr::get(
            kernel.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
            mapping.getDimensionId(), resultAxis, mapping.getDerived()));
      }
      source = FragmentType::get(
          kernel.getContext(), source.getElementType(),
          builder.getArrayAttr(transposedShape),
          builder.getArrayAttr(transposedMappings), source.getValidity(),
          source.getOwner());
      value = builder.create<TransposeOp>(contract.getLoc(), source, value,
                                           permutation);
      for (auto [position, axis] : llvm::enumerate(reductions))
        axis = firstReduction + static_cast<int64_t>(position);
    }
    SmallVector<Attribute> shape, mappings, groups;
    for (unsigned axis = 0; axis < source.getShape().size();) {
      unsigned resultAxis = shape.size();
      SmallVector<int64_t> sourceAxes;
      auto extent = cast<PhysicalExprAttr>(source.getShape()[axis]);
      AxisMapAttr mapping;
      if (axis == static_cast<unsigned>(reductions.front())) {
        sourceAxes.append(reductions.begin(), reductions.end());
        for (int64_t reduced : ArrayRef<int64_t>(reductions).drop_front()) {
          auto factor = cast<PhysicalExprAttr>(source.getShape()[reduced]);
          if (extent == unit)
            extent = factor;
          else if (factor != unit)
            extent = binaryExpression(kernel.getContext(), PhysicalExprKind::Multiply,
                                      extent, factor);
        }
        mapping = AxisMapAttr::get(kernel.getContext(), sourceId, 0,
                                   dimensionId, resultAxis, true);
        axis += reductions.size();
      } else {
        sourceAxes.push_back(axis);
        auto original = cast<AxisMapAttr>(source.getAxisMaps()[axis++]);
        mapping = AxisMapAttr::get(
            kernel.getContext(), original.getSourceId(), original.getSourceAxis(),
            original.getDimensionId(), resultAxis, original.getDerived());
      }
      shape.push_back(extent);
      mappings.push_back(mapping);
      groups.push_back(ReshapeGroupAttr::get(
          kernel.getContext(), builder.getDenseI64ArrayAttr(sourceAxes),
          builder.getDenseI64ArrayAttr({static_cast<int64_t>(resultAxis)})));
    }
    auto target = FragmentType::get(
        kernel.getContext(), source.getElementType(), builder.getArrayAttr(shape),
        builder.getArrayAttr(mappings), source.getValidity(), source.getOwner());
    return {builder.create<ReshapeOp>(contract.getLoc(), target, value,
                                      builder.getArrayAttr(groups)),
            reductions.front()};
  };
  auto [lhs, lhsAxis] = collapse(*completeLhs, lhsAxes);
  auto [rhs, rhsAxis] = collapse(*completeRhs, rhsAxes);
  auto replacement = builder.create<ContractOp>(
      contract.getLoc(), contract.getResult().getType(), lhs, rhs,
      contract.getAccumulator(), ArrayRef<int64_t>{lhsAxis},
      ArrayRef<int64_t>{rhsAxis}, ArrayRef<int64_t>{}, ArrayRef<int64_t>{});
  if (Attribute origin = contract->getAttr(originAttr))
    replacement->setAttr(originAttr, origin);
  contract.getResult().replaceAllUsesWith(replacement.getResult());
  contract.erase();
  return true;
}

static ReshapeOp exposeTransposedContractSplit(ContractOp contract) {
  auto transpose = dyn_cast<TransposeOp>(*contract.getResult().getUsers().begin());
  if (!transpose || transpose.getPermutation() != ArrayRef<int64_t>{1, 0} ||
      !transpose.getResult().hasOneUse())
    return {};
  auto output = dyn_cast<ReshapeOp>(*transpose.getResult().getUsers().begin());
  if (!output)
    return {};
  auto target = cast<FragmentType>(output.getResult().getType());
  SmallVector<int64_t> freeAxes;
  std::optional<int64_t> columnAxis;
  llvm::SmallDenseSet<int64_t> units;
  for (Attribute attribute : output.getReassociation()) {
    auto group = cast<ReshapeGroupAttr>(attribute);
    if (group.getSourceAxes().empty()) {
      for (int64_t axis : group.getResultAxes().asArrayRef()) {
        auto extent = cast<PhysicalExprAttr>(target.getShape()[axis]);
        if (extent.getKind() != PhysicalExprKind::Constant ||
            extent.getValue() != 1)
          return {};
        units.insert(axis);
      }
    } else if (group.getSourceAxes().asArrayRef() == ArrayRef<int64_t>{0} &&
               group.getResultAxes().size() == 1 && !columnAxis) {
      columnAxis = group.getResultAxes()[0];
    } else if (group.getSourceAxes().asArrayRef() == ArrayRef<int64_t>{1} && freeAxes.empty()) {
      llvm::append_range(freeAxes, group.getResultAxes().asArrayRef());
    } else {
      return {};
    }
  }
  if (!columnAxis || freeAxes.size() < 2 ||
      freeAxes.size() + units.size() + 1 != target.getShape().size())
    return {};
  auto rowExtent = cast<PhysicalExprAttr>(contract.getResult().getType().getShape()[0]);
  if (rowExtent.getKind() != PhysicalExprKind::Constant ||
      rowExtent.getValue() <= 0)
    return {};
  int64_t product = 1;
  for (int64_t axis : freeAxes) {
    auto extent = cast<PhysicalExprAttr>(target.getShape()[axis]);
    if (extent.getKind() != PhysicalExprKind::Constant ||
        extent.getValue() <= 0 || product > rowExtent.getValue() / extent.getValue())
      return {};
    product *= extent.getValue();
  }
  if (product != rowExtent.getValue() ||
      target.getShape()[*columnAxis] != contract.getResult().getType().getShape()[1])
    return {};

  // Commute a pure transpose with the declared row-major split, exposing
  // [split rows..., column] to the existing contraction composition below.
  OpBuilder builder(transpose);
  SmallVector<Attribute> shape, mappings;
  SmallVector<int64_t> splitAxes;
  auto appendAxis = [&](int64_t axis) {
    auto mapping = cast<AxisMapAttr>(target.getAxisMaps()[axis]);
    mappings.push_back(AxisMapAttr::get(
        contract.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
        mapping.getDimensionId(), shape.size(), mapping.getDerived()));
    shape.push_back(target.getShape()[axis]);
  };
  for (int64_t axis : freeAxes) {
    splitAxes.push_back(shape.size());
    appendAxis(axis);
  }
  appendAxis(*columnAxis);
  auto expanded = FragmentType::get(
      contract.getContext(), target.getElementType(), builder.getArrayAttr(shape),
      builder.getArrayAttr(mappings), target.getValidity(), target.getOwner());
  auto split = builder.create<ReshapeOp>(
      output.getLoc(), expanded, contract.getResult(), builder.getArrayAttr({
          ReshapeGroupAttr::get(contract.getContext(), builder.getDenseI64ArrayAttr({0}),
                               builder.getDenseI64ArrayAttr(splitAxes)),
          ReshapeGroupAttr::get(contract.getContext(), builder.getDenseI64ArrayAttr({1}),
                               builder.getDenseI64ArrayAttr({static_cast<int64_t>(freeAxes.size())}))}));
  if (Attribute origin = output->getAttr(originAttr))
    split->setAttr(originAttr, origin);
  shape.clear();
  mappings.clear();
  SmallVector<int64_t> permutation;
  SmallVector<Attribute> restore;
  for (int64_t axis = 0; axis < static_cast<int64_t>(target.getShape().size()); ++axis) {
    if (units.contains(axis)) {
      restore.push_back(ReshapeGroupAttr::get(
          contract.getContext(), builder.getDenseI64ArrayAttr({}),
          builder.getDenseI64ArrayAttr({axis})));
      continue;
    }
    int64_t inputAxis = axis == *columnAxis
                            ? static_cast<int64_t>(freeAxes.size())
                            : std::distance(freeAxes.begin(), llvm::find(freeAxes, axis));
    permutation.push_back(inputAxis);
    restore.push_back(ReshapeGroupAttr::get(
        contract.getContext(), builder.getDenseI64ArrayAttr({static_cast<int64_t>(shape.size())}),
        builder.getDenseI64ArrayAttr({axis})));
    appendAxis(axis);
  }
  auto reordered = FragmentType::get(
      contract.getContext(), target.getElementType(), builder.getArrayAttr(shape),
      builder.getArrayAttr(mappings), target.getValidity(), target.getOwner());
  Value value = builder.create<TransposeOp>(
      output.getLoc(), reordered, split, builder.getDenseI64ArrayAttr(permutation));
  if (Attribute origin = output->getAttr(originAttr))
    value.getDefiningOp()->setAttr(originAttr, origin);
  if (!units.empty()) {
    value = builder.create<ReshapeOp>(output.getLoc(), target, value, builder.getArrayAttr(restore));
    if (Attribute origin = output->getAttr(originAttr))
      value.getDefiningOp()->setAttr(originAttr, origin);
  }
  output.getResult().replaceAllUsesWith(value);
  output.erase();
  transpose.erase();
  return split;
}

static void orientContractionOutputs(func::FuncOp kernel) {
  SmallVector<ContractOp> contracts;
  kernel.walk([&](ContractOp contract) { contracts.push_back(contract); });
  for (ContractOp contract : contracts) {
    auto result = contract.getResult().getType();
    auto lhs = contract.getLhs().getType();
    auto rhs = contract.getRhs().getType();
    if (!lhs.getElementType().isF32() || !rhs.getElementType().isF32() ||
        !result.getElementType().isF32() ||
        !contract.getLhsBatchAxes().empty() ||
        !contract.getRhsBatchAxes().empty())
      continue;
    unsigned lhsFree =
        lhs.getShape().size() - contract.getLhsReductionAxes().size();
    unsigned rhsFree =
        rhs.getShape().size() - contract.getRhsReductionAxes().size();
    if (!lhsFree || !rhsFree)
      continue;
    PhysicalProgramAnalysis analysis(kernel);

    SmallVector<Value> pending{contract.getResult()};
    llvm::DenseSet<Value> visited;
    bool compatible = true;
    bool sawStore = false;
    while (compatible && !pending.empty()) {
      Value value = pending.pop_back_val();
      if (!visited.insert(value).second)
        continue;
      for (OpOperand &use : value.getUses()) {
        Operation *user = use.getOwner();
        if (auto store = dyn_cast<StoreOp>(user)) {
          auto axes = store.getSourceAxes();
          if (store.getValue() != value || axes.empty()) {
            compatible = false;
            break;
          }
          auto position = llvm::find(axes, static_cast<int64_t>(axes.size() - 1));
          if (position == axes.end()) {
            compatible = false;
            break;
          }
          Value coordinate = store.getCoordinates()[position - axes.begin()];
          while (auto projection = coordinate.getDefiningOp()) {
            if (!isa<BroadcastOp, ReshapeOp>(projection))
              break;
            coordinate = projection->getOperand(0);
          }
          auto type = dyn_cast<FragmentType>(coordinate.getType());
          if (!type || type.getShape().size() != 1) {
            compatible = false;
            break;
          }
          auto destination =
              queryExactLogicalRange(analysis.sourceRanges(coordinate));
          if (failed(destination)) {
            compatible = false;
            break;
          }
          auto matchesFreeAxis = [&](Value operand, ArrayRef<int64_t> reduction) {
            auto fragment = cast<FragmentType>(operand.getType());
            for (unsigned axis = 0; axis < fragment.getShape().size(); ++axis) {
              if (llvm::is_contained(reduction, static_cast<int64_t>(axis)))
                continue;
              auto ranges = analysis.axisRanges(operand, axis);
              if (ranges.isExact() && !ranges.roots.empty() &&
                  ranges.blockers.empty() &&
                  llvm::all_of(ranges.roots, [&](MakeRangeOp range) {
                    return sameLogicalRange(range, *destination);
                  }))
                return true;
            }
            return false;
          };
          if (!matchesFreeAxis(contract.getLhs(),
                               contract.getLhsReductionAxes()) ||
              matchesFreeAxis(contract.getRhs(), contract.getRhsReductionAxes())) {
            compatible = false;
            break;
          }
          sawStore = true;
          continue;
        }
        if (isa<scf::YieldOp>(user) &&
            isa<scf::ForOp, scf::IfOp>(user->getParentOp())) {
          pending.push_back(
              user->getParentOp()->getResult(use.getOperandNumber()));
          continue;
        }
        if (!isa<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp,
                 SplatOp, BroadcastOp, ReshapeOp, TransposeOp>(user)) {
          compatible = false;
          break;
        }
        llvm::append_range(pending, user->getResults());
      }
    }
    if (!compatible || !sawStore)
      continue;

    // Matrix columns follow the innermost output coordinate. Swap the free-axis
    // groups and restore the original result schema for all existing users.
    SmallVector<int64_t> permutation;
    for (unsigned axis = lhsFree; axis < lhsFree + rhsFree; ++axis)
      permutation.push_back(axis);
    for (unsigned axis = 0; axis < lhsFree; ++axis)
      permutation.push_back(axis);
    SmallVector<Attribute> shape, mappings;
    SmallVector<int64_t> inverse(permutation.size());
    for (auto [position, axis] : llvm::enumerate(permutation)) {
      shape.push_back(result.getShape()[axis]);
      auto mapping = cast<AxisMapAttr>(result.getAxisMaps()[axis]);
      mappings.push_back(AxisMapAttr::get(
          kernel.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
          mapping.getDimensionId(), position, mapping.getDerived()));
      inverse[axis] = position;
    }
    OpBuilder builder(contract);
    auto transposed = FragmentType::get(
        kernel.getContext(), result.getElementType(), builder.getArrayAttr(shape),
        builder.getArrayAttr(mappings), result.getValidity(), result.getOwner());
    Value accumulator = builder.create<TransposeOp>(
        contract.getLoc(), transposed, contract.getAccumulator(), permutation);
    auto replacement = cast<ContractOp>(builder.clone(*contract));
    replacement->setOperand(0, contract.getRhs());
    replacement->setOperand(1, contract.getLhs());
    replacement.getAccumulatorMutable().assign(accumulator);
    replacement.setLhsReductionAxesAttr(contract.getRhsReductionAxesAttr());
    replacement.setRhsReductionAxesAttr(contract.getLhsReductionAxesAttr());
    replacement.getResult().setType(transposed);
    Value restored = builder.create<TransposeOp>(
        contract.getLoc(), result, replacement.getResult(), inverse);
    contract.getResult().replaceAllUsesWith(restored);
    contract.erase();
  }
}

LogicalResult fuseMultiplyReductions(ModuleOp module) {
  auto kernel = getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  SmallVector<ReduceOp> reductions;
  kernel->walk([&](ReduceOp reduce) { reductions.push_back(reduce); });
  for (ReduceOp reduce : reductions) {
    if (reduce.getSources().size() != 1 || reduce.getIdentities().size() != 1 ||
        reduce.getCaptures().size() != 0 || reduce.getNumResults() != 1 ||
        !isLiteralZeroProjection(reduce.getIdentities().front()))
      continue;
    auto resultType = dyn_cast<FragmentType>(reduce.getResult(0).getType());
    // The existing matrix path provides full-precision f32 accumulation.
    // Other accumulator formats keep their native reduction until that path
    // can preserve their precision without imposing new provider restrictions.
    if (!resultType || !resultType.getElementType().isF32())
      continue;
    auto combine = queryBinaryCombine(reduce.getCombine());
    if (!combine || combine->kind() != BinaryOperator::Add ||
        combine->operation.getStrictRounding())
      continue;

    Value product = reduce.getSources().front();
    // Builtin sum may retain an identity cast. A numeric conversion, including
    // default accumulator widening, is a rounding boundary and cannot fuse.
    while (product.hasOneUse()) {
      auto cast = product.getDefiningOp<CastOp>();
      if (!cast || cast.getValue().getType() != product.getType())
        break;
      product = cast.getValue();
    }
    auto multiply = product.getDefiningOp<BinaryOp>();
    auto productType = dyn_cast<FragmentType>(product.getType());
    if (!multiply || !product.hasOneUse() || !productType ||
        multiply.getOperatorKind() != BinaryOperator::Multiply ||
        multiply.getStrictRounding() ||
        productType.getElementType() != resultType.getElementType())
      continue;
    auto unbroadcast = [&](Value value) {
      auto source = dyn_cast<FragmentType>(value.getType());
      BroadcastProjection projection;
      if (source)
        projection = queryBroadcastProjection(source, productType);
      if (!projection.isExact())
        return std::pair{value, projection};
      while (Operation *producer = value.getDefiningOp()) {
        if (auto broadcast = dyn_cast<BroadcastOp>(producer)) {
          auto input = dyn_cast<FragmentType>(broadcast.getValue().getType());
          auto output = cast<FragmentType>(value.getType());
          if (!input || input.getValidity() != output.getValidity())
            break;
          auto relations = queryFragmentOperandRelations(broadcast.getOperation());
          if (failed(relations) || !relations->front().hasCompatibleExtents())
            break;
          for (auto &axis : projection.targetToSource) {
            if (!axis)
              continue;
            const auto *group = relations->front().groupForResultAxis(*axis);
            if (!group || group->sourceAxes.size() > 1)
              return std::pair{value, BroadcastProjection{}};
            axis = group->sourceAxes.empty()
                       ? std::nullopt
                       : std::optional<unsigned>(group->sourceAxes.front());
          }
          value = broadcast.getValue();
        } else if (auto conversion = dyn_cast<CastOp>(producer)) {
          if (conversion.getValue().getType() != value.getType())
            break;
          value = conversion.getValue();
        } else if (auto reshape = dyn_cast<ReshapeOp>(producer)) {
          auto source = dyn_cast<FragmentType>(reshape.getValue().getType());
          auto target = dyn_cast<FragmentType>(value.getType());
          auto relations = queryFragmentOperandRelations(reshape.getOperation());
          if (!source || !target || source.getShape() != target.getShape() ||
              source.getValidity() != target.getValidity() ||
              failed(relations) ||
              !llvm::all_of(relations->front().groups, [](const auto &group) {
                return group.sourceAxes.size() == 1 &&
                       group.sourceAxes == group.resultAxes;
              }))
            break;
          value = reshape.getValue();
        } else {
          break;
        }
      }
      return std::pair{value, projection};
    };
    auto [lhs, lhsProjection] = unbroadcast(multiply.getLhs());
    auto [rhs, rhsProjection] = unbroadcast(multiply.getRhs());
    auto lhsType = dyn_cast<FragmentType>(lhs.getType());
    auto rhsType = dyn_cast<FragmentType>(rhs.getType());
    if (!lhsType || !rhsType ||
        lhsType.getElementType() != productType.getElementType() ||
        rhsType.getElementType() != productType.getElementType())
      continue;
    if (!lhsProjection.isExact() || !rhsProjection.isExact())
      continue;
    PhysicalProgramAnalysis analysis(*kernel);
    auto isLogicalUnitAxis = [&](Value value, unsigned axis) {
      auto type = cast<FragmentType>(value.getType());
      if (constantPhysicalExpression(
              cast<PhysicalExprAttr>(type.getShape()[axis])) != 1)
        return false;
      // A one-element physical tile can still carry a non-unit logical axis.
      // Drop only introduced units or ranges proven logically singleton.
      auto ranges = analysis.axisRanges(value, axis);
      return ranges.isExact() &&
             llvm::all_of(ranges.roots, [](MakeRangeOp range) {
               // A lifted workset coordinate starts as one logical iteration;
               // ownership may subsequently pack several iterations together.
               return !range.getStart().getDefiningOp<WorksetCoordinateOp>() &&
                      isProvablySingletonLogicalRange(range);
             });
    };
    SmallVector<bool> lhsUnits, rhsUnits;
    for (unsigned axis = 0; axis < lhsType.getShape().size(); ++axis)
      lhsUnits.push_back(isLogicalUnitAxis(lhs, axis));
    for (unsigned axis = 0; axis < rhsType.getShape().size(); ++axis)
      rhsUnits.push_back(isLogicalUnitAxis(rhs, axis));
    auto lhsAxes = lhsProjection.targetToSource;
    auto rhsAxes = rhsProjection.targetToSource;
    bool compatible = true;
    for (unsigned axis = 0; axis < productType.getShape().size(); ++axis) {
      if (llvm::is_contained(reduce.getAxes(), axis))
        continue;
      auto left = lhsAxes[axis];
      auto right = rhsAxes[axis];
      // Keep the existing GPU eligibility boundary for a common unit result:
      // its representative must already be present on the original lhs.
      if ((!left || lhsUnits[*left]) && (!right || rhsUnits[*right])) {
        if (!left) {
          compatible = false;
          break;
        }
        // The old GPU rewrite discarded this rhs unit before orientation.
        // Keep that representative choice when the shared query is rerun with
        // swapped operands; an omitted source axis is proven singleton here.
        rhsAxes[axis] = std::nullopt;
      }
    }
    if (!compatible)
      continue;
    SmallVector<int64_t> reductionAxes(reduce.getAxes());
    llvm::sort(reductionAxes);
    auto axes = ProductContractionAxes::get(
        lhsAxes, lhsUnits, rhsAxes, rhsUnits, reductionAxes);
    // Preserve matrix-vector reuse before ownership can lift an independent
    // workset axis onto the other operand. Remaining vector contractions are
    // realized as native reductions after blocking.
    if (!axes || (axes->axes.lhsFree.empty() && axes->axes.rhsFree.empty()))
      continue;
    // Preserve the innermost result coordinates as the matrix column axes.
    // Operand orientation is GPU policy; recompute the shared relation for
    // the selected orientation instead of exchanging partial axis lists.
    std::optional<int64_t> lhsOutput, rhsOutput;
    for (auto [position, result] : llvm::enumerate(axes->axes.results))
      (result.operand == ContractionOperand::Lhs ? lhsOutput : rhsOutput) =
          axes->resultProductAxes[position];
    if (axes->axes.batch.empty() && lhsOutput && rhsOutput &&
        *lhsOutput > *rhsOutput) {
      auto oriented = ProductContractionAxes::get(
          rhsAxes, rhsUnits, lhsAxes, lhsUnits, reductionAxes);
      if (!oriented)
        return reduce.emitOpError("cannot preserve the multiply-reduction axis relation when orienting operands");
      std::swap(lhs, rhs);
      axes = std::move(oriented);
    }
    OpBuilder builder(reduce);
    auto squeeze = [&](Value value, ArrayRef<int64_t> kept) -> Value {
      auto source = cast<FragmentType>(value.getType());
      if (kept.size() == source.getShape().size())
        return value;
      SmallVector<Attribute> shape, mappings, groups;
      for (unsigned axis = 0; axis < source.getShape().size(); ++axis) {
        SmallVector<int64_t> resultAxes;
        if (llvm::is_contained(kept, axis)) {
          resultAxes.push_back(shape.size());
          shape.push_back(source.getShape()[axis]);
          auto map = cast<AxisMapAttr>(source.getAxisMaps()[axis]);
          mappings.push_back(AxisMapAttr::get(
              module.getContext(), map.getSourceId(), map.getSourceAxis(),
              map.getDimensionId(), mappings.size(), map.getDerived()));
        }
        groups.push_back(ReshapeGroupAttr::get(
            module.getContext(), builder.getDenseI64ArrayAttr({axis}),
            builder.getDenseI64ArrayAttr(resultAxes)));
      }
      auto target = FragmentType::get(
          module.getContext(), source.getElementType(),
          builder.getArrayAttr(shape), builder.getArrayAttr(mappings),
          source.getValidity(), source.getOwner());
      return builder.create<ReshapeOp>(reduce.getLoc(), target, value,
                                       builder.getArrayAttr(groups));
    };
    lhs = squeeze(lhs, axes->lhsKept);
    rhs = squeeze(rhs, axes->rhsKept);
    SmallVector<Attribute> shape, mappings;
    for (ContractionResultAxis result : axes->axes.results) {
      Value value = result.operand == ContractionOperand::Lhs ? lhs : rhs;
      auto type = cast<FragmentType>(value.getType());
      shape.push_back(type.getShape()[result.axis]);
      auto map = cast<AxisMapAttr>(type.getAxisMaps()[result.axis]);
      mappings.push_back(AxisMapAttr::get(
          module.getContext(), map.getSourceId(), map.getSourceAxis(),
          map.getDimensionId(), mappings.size(), map.getDerived()));
    }
    auto contractedType = FragmentType::get(
        module.getContext(), resultType.getElementType(),
        builder.getArrayAttr(shape), builder.getArrayAttr(mappings),
        resultType.getValidity(), resultType.getOwner());
    auto zero = projectPhysicalValueToSchema(
        builder, reduce.getLoc(), reduce.getIdentities().front(), contractedType);
    if (failed(zero))
      return reduce.emitOpError("cannot form the multiply-reduction identity");
    SmallVector<int64_t> lhsReduced, rhsReduced, lhsBatch, rhsBatch;
    for (ContractionAxisPair pair : axes->axes.reduction) {
      lhsReduced.push_back(pair.lhs);
      rhsReduced.push_back(pair.rhs);
    }
    for (ContractionAxisPair pair : axes->axes.batch) {
      lhsBatch.push_back(pair.lhs);
      rhsBatch.push_back(pair.rhs);
    }
    auto contract = builder.create<ContractOp>(
        reduce.getLoc(), contractedType, lhs, rhs, *zero, lhsReduced, rhsReduced,
        lhsBatch, rhsBatch);
    if (Attribute origin = reduce->getAttr(originAttr))
      contract->setAttr(originAttr, origin);
    ArrayRef<int64_t> permutation = axes->resultPermutation;
    Value replacement = contract.getResult();
    if (!llvm::all_of(llvm::enumerate(permutation), [](auto item) {
          return item.index() == static_cast<unsigned>(item.value());
        })) {
      SmallVector<Attribute> reorderedShape, reorderedMappings;
      for (int64_t axis : permutation) {
        reorderedShape.push_back(shape[axis]);
        auto map = cast<AxisMapAttr>(mappings[axis]);
        reorderedMappings.push_back(AxisMapAttr::get(
            module.getContext(), map.getSourceId(), map.getSourceAxis(),
            map.getDimensionId(), reorderedMappings.size(), map.getDerived()));
      }
      auto reorderedType = FragmentType::get(
          module.getContext(), resultType.getElementType(),
          builder.getArrayAttr(reorderedShape),
          builder.getArrayAttr(reorderedMappings),
          resultType.getValidity(), resultType.getOwner());
      replacement = builder.create<TransposeOp>(
          reduce.getLoc(), reorderedType, replacement, permutation);
    }
    if (replacement.getType() != resultType)
      replacement = builder.create<BroadcastOp>(reduce.getLoc(), resultType,
                                                replacement);
    reduce.getResult(0).replaceAllUsesWith(replacement);
    reduce.erase();
  }
  eraseDeadPhysicalValues(*kernel);
  return success();
}

} // namespace intent::gpu::contraction

namespace intent::gpu {
using namespace contraction;

static LogicalResult normalizeContractionSourcesImpl(ModuleOp module) {
  FailureOr<func::FuncOp> kernel = getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  if (failed(fuseMultiplyReductions(module)) ||
      failed(realizeAccessComposition(module)))
    return failure();
  orientContractionOutputs(*kernel);
  // Both stages operate on a fresh snapshot: a source rewrite may erase
  // contractions or expose a different current value graph.
  auto collapsed = rewriteContractionSnapshot(*kernel, collapseMultiReductionContract);
  if (failed(collapsed) || (*collapsed && failed(realizeAccessComposition(module))))
    return failure();
  auto projected = rewriteContractionSnapshot(*kernel, projectContractResult);
  if (failed(projected))
    return failure();
  if (*projected) {
    eraseDeadPhysicalValues(*kernel);
    if (failed(realizeAccessComposition(module)))
      return failure();
  }
  SmallVector<ContractOp> contracts;
  kernel->walk([&](ContractOp contract) { contracts.push_back(contract); });
  bool changed = false;
  for (ContractOp contract : contracts) {
    if (!contract.getResult().hasOneUse() ||
        contract.getLhs().getType().getShape().size() != 2 ||
        contract.getRhs().getType().getShape().size() != 2 ||
        contract.getLhsReductionAxes() != ArrayRef<int64_t>{1} ||
        (contract.getRhsReductionAxes() != ArrayRef<int64_t>{0} &&
         contract.getRhsReductionAxes() != ArrayRef<int64_t>{1}) ||
        !contract.getLhsBatchAxes().empty() ||
        !contract.getRhsBatchAxes().empty())
      continue;
    auto output = dyn_cast<ReshapeOp>(*contract.getResult().getUsers().begin());
    auto identity = contract.getAccumulator().getDefiningOp<SplatOp>();
    if (!identity)
      continue;
    if (!output) {
      output = exposeTransposedContractSplit(contract);
      changed |= static_cast<bool>(output);
    }
    if (!output || !identity || output.getReassociation().size() != 2)
      continue;
    auto target = cast<FragmentType>(output.getResult().getType());
    auto outputLeft = cast<ReshapeGroupAttr>(output.getReassociation()[0]);
    auto outputRight = cast<ReshapeGroupAttr>(output.getReassociation()[1]);
    bool splitLeft = outputLeft.getResultAxes().size() > 1;
    auto outputFree = splitLeft ? outputLeft : outputRight;
    auto outputOther = splitLeft ? outputRight : outputLeft;
    unsigned freeRank = outputFree.getResultAxes().size();
    unsigned outputFreeAxis = splitLeft ? 0 : 1;
    unsigned outputOffset = splitLeft ? 0 : 1;
    if (freeRank < 2 || target.getShape().size() != freeRank + 1 ||
        outputFree.getSourceAxes().asArrayRef() !=
            ArrayRef<int64_t>{static_cast<int64_t>(outputFreeAxis)} ||
        outputOther.getSourceAxes().asArrayRef() !=
            ArrayRef<int64_t>{static_cast<int64_t>(1 - outputFreeAxis)} ||
        outputOther.getResultAxes().asArrayRef() !=
            ArrayRef<int64_t>{splitLeft ? static_cast<int64_t>(freeRank) : 0} ||
        llvm::any_of(llvm::enumerate(outputFree.getResultAxes().asArrayRef()),
                     [&](auto item) {
                       return item.value() !=
                              static_cast<int64_t>(outputOffset + item.index());
                     }))
      continue;
    Value sourceValue = splitLeft ? contract.getLhs() : contract.getRhs();
    auto matrix = cast<FragmentType>(sourceValue.getType());
    auto source = matrix;
    unsigned reductionAxis = splitLeft ? 1 : contract.getRhsReductionAxes()[0];
    unsigned operandFreeAxis = 1 - reductionAxis;
    auto input = sourceValue.getDefiningOp<ReshapeOp>();
    ReshapeGroupAttr inputReduction;
    ReshapeGroupAttr inputFree;
    bool cancelsInput = false;
    if (input && input.getReassociation().size() == 2) {
      auto inputType = cast<FragmentType>(input.getValue().getType());
      inputFree = cast<ReshapeGroupAttr>(input.getReassociation()[operandFreeAxis]);
      inputReduction = cast<ReshapeGroupAttr>(input.getReassociation()[reductionAxis]);
      cancelsInput =
          inputFree.getResultAxes().asArrayRef() ==
              ArrayRef<int64_t>{static_cast<int64_t>(operandFreeAxis)} &&
          inputReduction.getResultAxes().asArrayRef() ==
              ArrayRef<int64_t>{static_cast<int64_t>(reductionAxis)} &&
          inputFree.getSourceAxes().size() == freeRank;
      for (unsigned axis = 0; cancelsInput && axis < freeRank; ++axis) {
        unsigned inputAxis = inputFree.getSourceAxes()[axis];
        unsigned outputAxis = outputOffset + axis;
        auto original = cast<AxisMapAttr>(inputType.getAxisMaps()[inputAxis]);
        auto restored = cast<AxisMapAttr>(target.getAxisMaps()[outputAxis]);
        cancelsInput = original.getDimensionId() > 0 &&
                       original.getDimensionId() == restored.getDimensionId() &&
                       inputType.getShape()[inputAxis] == target.getShape()[outputAxis];
      }
      if (cancelsInput) {
        sourceValue = input.getValue();
        source = inputType;
      }
    }

    // Move the declared row-major free-axis split through the contraction.
    // Either operand can carry the split free axis. Cancel an inverse input
    // reshape while retaining its coordinate roots, or split the flat operand.
    OpBuilder builder(contract);
    if (splitLeft && contract.getRhsReductionAxes() == ArrayRef<int64_t>{1}) {
      Value rhs = contract.getRhs();
      contract.getRhsMutable().assign(builder.create<TransposeOp>(
          contract.getLoc(), transposeLastTwo(cast<FragmentType>(rhs.getType())),
          rhs, builder.getDenseI64ArrayAttr({1, 0})));
      contract->setAttr("rhs_reduction_axes", builder.getDenseI64ArrayAttr({0}));
    }
    SmallVector<Attribute> shape, mappings, groups;
    SmallVector<Attribute> freeMappings;
    unsigned expandedReductionAxis = reductionAxis == 0 ? 0 : freeRank;
    auto appendAxis = [&](Attribute extent, Attribute attribute) {
      auto axis = cast<AxisMapAttr>(attribute);
      mappings.push_back(AxisMapAttr::get(
          module.getContext(), axis.getSourceId(), axis.getSourceAxis(),
          axis.getDimensionId(), shape.size(), axis.getDerived()));
      shape.push_back(extent);
    };
    for (unsigned matrixAxis = 0; matrixAxis < 2; ++matrixAxis) {
      if (matrixAxis == reductionAxis) {
        appendAxis(matrix.getShape()[matrixAxis], matrix.getAxisMaps()[matrixAxis]);
        groups.push_back(ReshapeGroupAttr::get(
            module.getContext(), cancelsInput ? inputReduction.getSourceAxes()
                : builder.getDenseI64ArrayAttr({static_cast<int64_t>(matrixAxis)}),
            builder.getDenseI64ArrayAttr({static_cast<int64_t>(expandedReductionAxis)})));
        continue;
      }
      SmallVector<int64_t> expandedFreeAxes;
      for (unsigned axis = 0; axis < freeRank; ++axis) {
        unsigned outputAxis = outputOffset + axis;
        unsigned inputAxis = cancelsInput ? inputFree.getSourceAxes()[axis] : matrixAxis;
        expandedFreeAxes.push_back(shape.size());
        appendAxis(target.getShape()[outputAxis],
                   cancelsInput ? source.getAxisMaps()[inputAxis]
                                : target.getAxisMaps()[outputAxis]);
        freeMappings.push_back(mappings.back());
        if (cancelsInput)
          groups.push_back(ReshapeGroupAttr::get(
              module.getContext(), builder.getDenseI64ArrayAttr({inputAxis}),
              builder.getDenseI64ArrayAttr({expandedFreeAxes.back()})));
      }
      if (!cancelsInput)
        groups.push_back(ReshapeGroupAttr::get(
            module.getContext(), builder.getDenseI64ArrayAttr({static_cast<int64_t>(matrixAxis)}),
            builder.getDenseI64ArrayAttr(expandedFreeAxes)));
    }
    auto operandType = FragmentType::get(
        module.getContext(), matrix.getElementType(), builder.getArrayAttr(shape),
        builder.getArrayAttr(mappings), matrix.getValidity(), matrix.getOwner());
    while (auto cast = sourceValue.getDefiningOp<CastOp>()) {
      if (cast.getValue().getType() != sourceValue.getType())
        break;
      sourceValue = cast.getValue();
    }
    Value operand = builder.create<ReshapeOp>(
        contract.getLoc(), operandType, sourceValue, builder.getArrayAttr(groups));
    Attribute other = splitLeft ? contract.getRhs().getType().getAxisMaps()[1]
                               : contract.getLhs().getType().getAxisMaps()[0];
    SmallVector<Attribute> resultMappings;
    for (unsigned axis = 0; axis <= freeRank; ++axis) {
      bool isOther = splitLeft ? axis == freeRank : axis == 0;
      auto mapping = cast<AxisMapAttr>(isOther ? other : freeMappings[axis - outputOffset]);
      resultMappings.push_back(AxisMapAttr::get(
          module.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
          mapping.getDimensionId(), axis, mapping.getDerived()));
    }
    auto resultType = FragmentType::get(
        module.getContext(), target.getElementType(), target.getShape(),
        builder.getArrayAttr(resultMappings), target.getValidity(),
        target.getOwner());
    Value accumulator = builder.create<SplatOp>(output.getLoc(), resultType,
                                               identity.getValue());
    (splitLeft ? contract.getLhsMutable() : contract.getRhsMutable()).assign(operand);
    contract.getAccumulatorMutable().assign(accumulator);
    contract->setAttr(splitLeft ? "lhs_reduction_axes" : "rhs_reduction_axes",
                      builder.getDenseI64ArrayAttr({static_cast<int64_t>(expandedReductionAxis)}));
    contract.getResult().setType(resultType);
    builder.setInsertionPointAfter(contract);
    Value projected = builder.create<BroadcastOp>(output.getLoc(), target,
                                                 contract.getResult());
    output.getResult().replaceAllUsesWith(projected);
    output.erase();
    changed = true;
  }
  if (changed) {
    SmallVector<CastOp> identityCasts;
    kernel->walk([&](CastOp cast) {
      if (cast.getValue().getType() == cast.getResult().getType())
        identityCasts.push_back(cast);
    });
    for (CastOp cast : identityCasts) {
      cast.getResult().replaceAllUsesWith(cast.getValue());
      cast.erase();
    }
    eraseDeadPhysicalValues(*kernel);
    if (failed(realizeAccessComposition(module)))
      return failure();
  }
  return success();
}

LogicalResult normalizeContractionSources(ModuleOp module) {
  if (failed(normalizeContractionSourcesImpl(module))) return failure();
  auto kernel = getPhysicalKernel(module);
  return failed(kernel) ? failure() : closeValueRelations(*kernel);
}

} // namespace intent::gpu
