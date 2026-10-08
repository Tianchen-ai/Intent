#include "ContractionDetail.h"
#include "../Value/SourceReplay.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Transforms/Control/Traversal.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/PhysicalParameters.h"
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





LogicalResult decomposeMultiReductionContract(ContractOp contract) {
  if (!contract->getBlock() || contract.getLhsReductionAxes().size() <= 1)
    return success();
  if (contract.getLhsReductionAxes().size() !=
          contract.getRhsReductionAxes().size() ||
      contract.getLhsBatchAxes().size() != contract.getRhsBatchAxes().size())
    return contract.emitOpError(
        "multi-pair contraction decomposition requires paired reduction and batch axes");
  func::FuncOp kernel = contract->getParentOfType<func::FuncOp>();
  Location location = contract.getLoc();
  std::string failureReason;
  auto unit = expression(kernel.getContext(), PhysicalExprKind::Constant, 1);
  std::function<FailureOr<Value>(OpBuilder &, Value, Value,
                                SmallVector<int64_t>, SmallVector<int64_t>,
                                SmallVector<int64_t>, SmallVector<int64_t>, Value)>
      build;
  build = [&](OpBuilder &builder, Value lhs, Value rhs,
              SmallVector<int64_t> lhsReductions,
              SmallVector<int64_t> rhsReductions,
              SmallVector<int64_t> lhsBatch, SmallVector<int64_t> rhsBatch,
              Value accumulator) -> FailureOr<Value> {
    if (lhsReductions.size() == 1) {
      auto product = builder.create<ContractOp>(
          location, contract.getResult().getType(), lhs, rhs, accumulator,
          lhsReductions, rhsReductions, lhsBatch, rhsBatch);
      if (Attribute origin = contract->getAttr(originAttr))
        product->setAttr(originAttr, origin);
      return product.getResult();
    }

    unsigned lhsAxis = lhsReductions.front();
    unsigned rhsAxis = rhsReductions.front();
    PhysicalProgramAnalysis analysis(kernel);
    PhysicalRangeFact lhsFact = analysis.axisRanges(lhs, lhsAxis);
    PhysicalRangeFact rhsFact = analysis.axisRanges(rhs, rhsAxis);
    FailureOr<MakeRangeOp> lhsRange = queryExactLogicalRange(lhsFact);
    FailureOr<MakeRangeOp> rhsRange = queryExactLogicalRange(rhsFact);
    if (failed(lhsRange) || failed(rhsRange)) {
      failureReason = "a reduction pair has no exact source-axis ranges";
      return failure();
    }
    SmallVector<MakeRangeOp> pairedRanges(lhsFact.roots.begin(), lhsFact.roots.end());
    llvm::append_range(pairedRanges, rhsFact.roots);
    PhysicalLockstepTraversalFact lockstep = analysis.lockstepRanges(pairedRanges);
    if (!lockstep.isExact()) {
      failureReason = "a reduction pair does not have one lockstep traversal";
      return failure();
    }
    Value logicalEnd = (lockstep.authority).getLogicalStop();
    FailureOr<Value> step = scalarSource(lockstep.authority.getStep());
    if (failed(step)) {
      failureReason = "a reduction pair has no exact logical bounds and step";
      return failure();
    }

    bool failedBody = false;
    auto loop = builder.create<scf::ForOp>(
        location, lockstep.authority.getLogicalStart(), logicalEnd, *step,
        ValueRange{accumulator},
        [](OpBuilder &, Location, Value, ValueRange) {});
    auto buildBody = [&](OpBuilder &nested, Location nestedLocation, Value coordinate,
            ValueRange carries) {
          auto slice = [&](Value value, unsigned axis,
                           const PhysicalRangeFact &fact) -> FailureOr<Value> {
            MakeRangeOp root = fact.roots.front();
            auto rangeType = cast<FragmentType>(root.getResult().getType());
            auto indexType = FragmentType::get(
                kernel.getContext(), nested.getIndexType(),
                nested.getArrayAttr({unit}), rangeType.getAxisMaps(),
                rangeType.getValidity(), rangeType.getOwner());
            Value one = nested.create<arith::ConstantIndexOp>(nestedLocation, 1);
            Value selected = nested.create<MakeRangeOp>(
                nestedLocation, indexType, coordinate, one, root.getStep(),
                root.getLogicalStart(), root.getLogicalStop(), root.getSourceId(),
                root.getSourceAxis(), root.getDerived());
            inheritRangeAuthority(selected, root);
            IRMapping mapping;
            for (MakeRangeOp range : fact.roots)
              mapping.map(range.getResult(), selected);
            FailureOr<Value> replayed = materializeSourceRanges(
                nested, nestedLocation, value, unit, fact.roots, selected,
                mapping, loop.getOperation());
            if (failed(replayed))
              return failure();
            auto source = cast<FragmentType>((*replayed).getType());
            auto target = eraseFragmentAxis(source, axis);
            SmallVector<Attribute> groups;
            unsigned resultAxis = 0;
            for (unsigned sourceAxis = 0;
                 sourceAxis < source.getShape().size(); ++sourceAxis) {
              SmallVector<int64_t> resultAxes;
              if (sourceAxis != axis)
                resultAxes.push_back(resultAxis++);
              groups.push_back(ReshapeGroupAttr::get(
                  kernel.getContext(),
                  nested.getDenseI64ArrayAttr({static_cast<int64_t>(sourceAxis)}),
                  nested.getDenseI64ArrayAttr(resultAxes)));
            }
            return Value(nested.create<ReshapeOp>(
                nestedLocation, target, *replayed, nested.getArrayAttr(groups)));
          };
          FailureOr<Value> lhsSlice = slice(lhs, lhsAxis, lhsFact);
          FailureOr<Value> rhsSlice = slice(rhs, rhsAxis, rhsFact);
          if (failed(lhsSlice) || failed(rhsSlice)) {
            failureReason = "a reduction-pair slice could not preserve its value graph";
            failedBody = true;
            return;
          }
          FailureOr<Value> product = build(
              nested, *lhsSlice, *rhsSlice, eraseAxis(lhsReductions, lhsAxis),
              eraseAxis(rhsReductions, rhsAxis), eraseAxis(lhsBatch, lhsAxis),
              eraseAxis(rhsBatch, rhsAxis), carries.front());
          if (failed(product)) {
            failedBody = true;
            if (failureReason.empty())
              failureReason = "nested reduction-pair decomposition failed";
            return;
          }
          nested.create<scf::YieldOp>(nestedLocation, *product);
        };
    OpBuilder bodyBuilder = OpBuilder::atBlockEnd(loop.getBody());
    buildBody(bodyBuilder, location, loop.getInductionVar(), loop.getRegionIterArgs());
    if (failedBody) {
      loop.erase();
      return failure();
    }
    return loop.getResult(0);
  };

  OpBuilder builder(contract);
  FailureOr<Value> replacement = build(
      builder, contract.getLhs(), contract.getRhs(),
      SmallVector<int64_t>(contract.getLhsReductionAxes()),
      SmallVector<int64_t>(contract.getRhsReductionAxes()),
      SmallVector<int64_t>(contract.getLhsBatchAxes()),
      SmallVector<int64_t>(contract.getRhsBatchAxes()), contract.getAccumulator());
  if (failed(replacement))
    return contract.emitOpError(
               "multi-pair contraction could not be decomposed into provider-native contractions: ")
           << failureReason;
  contract.getResult().replaceAllUsesWith(*replacement);
  contract.erase();
  return success();
}

FailureOr<bool> realizeSegmentNativeReduction(ContractOp contract,
                                              func::FuncOp kernel) {
  if (contract.getLhsReductionAxes().size() != 1 ||
      contract.getRhsReductionAxes().size() != 1)
    return false;
  auto segmentParameter = [&](Value operand,
                              int64_t axis) -> FailureOr<ParameterAttr> {
    auto fragment = dyn_cast<FragmentType>(operand.getType());
    if (!fragment || axis < 0 ||
        axis >= static_cast<int64_t>(fragment.getShape().size()))
      return failure();
    auto extent = cast<PhysicalExprAttr>(fragment.getShape()[axis]);
    return regionContractionParameter(kernel, extent);
  };
  FailureOr<ParameterAttr> lhsSegment = segmentParameter(
      contract.getLhs(), contract.getLhsReductionAxes().front());
  FailureOr<ParameterAttr> rhsSegment = segmentParameter(
      contract.getRhs(), contract.getRhsReductionAxes().front());
  if (failed(lhsSegment) || failed(rhsSegment) ||
      *lhsSegment != *rhsSegment)
    return false;
  scf::ForOp segmentLoop =
      enclosingRegionContractionSegment(contract.getOperation());
  if (segmentLoop &&
      queryParameter(segmentLoop.getStep()) != *lhsSegment)
    return contract.emitOpError(
               "reduction extent disagrees with its enclosing region-contraction segment"),
           failure();
  if (failed(markNativeCoverage(kernel, contract)))
    return failure();
  return true;
}

FailureOr<bool> realizeStructuredNativeReduction(
    ContractOp contract, func::FuncOp kernel,
    SmallVectorImpl<ContractOp> &replayed) {
  if (contract.getLhsReductionAxes().size() != 1 ||
      contract.getRhsReductionAxes().size() != 1)
    return false;
  PhysicalProgramAnalysis analysis(kernel);
  bool fullReduction = true;
  bool retainedSource = false;
  for (auto [operand, axis] :
       {std::pair<Value, int64_t>{contract.getLhs(),
                                  contract.getLhsReductionAxes().front()},
        std::pair<Value, int64_t>{contract.getRhs(),
                                  contract.getRhsReductionAxes().front()}}) {
    auto type = cast<FragmentType>(operand.getType());
    auto parameter = parameterForExtent(
        kernel, cast<PhysicalExprAttr>(type.getShape()[axis]));
    fullReduction &= succeeded(parameter) &&
        parameter->getRole() ==
            ParameterRole::FullCoverage &&
        parameter->getCategory() ==
            ParameterCategory::Coverage;
    auto mapping = cast<AxisMapAttr>(type.getAxisMaps()[axis]);
    retainedSource |= !analysis.replayAt(
        operand, sourceAxisIdentity(mapping), PhysicalReplayScope::ValueGraph,
        /*allowAccesses=*/true, contract, IRMapping{},
        mapping.getDimensionId()).isReplayable();
  }
  if (fullReduction && retainedSource &&
      freeAxesReadyForReductionTraversal(contract, kernel)) {
    if (failed(markNativeCoverage(kernel, contract)))
      return failure();
    return true;
  }
  LoadOp lhsLoad = matrixOperandLoad(contract.getLhs());
  LoadOp rhsLoad = matrixOperandLoad(contract.getRhs());
  if (!lhsLoad || !rhsLoad)
    return false;
  struct SegmentFact {
    bool present = false;
    ParameterAttr parameter;
    std::optional<PhysicalSourceAxis> source;
  };
  auto operandSegment = [&](Value operand,
                            ArrayRef<int64_t> reductionAxes)
      -> FailureOr<SegmentFact> {
    SegmentFact result;
    auto fragment = dyn_cast<FragmentType>(operand.getType());
    if (!fragment)
      return failure();
    for (auto [axis, attribute] : llvm::enumerate(fragment.getShape())) {
      if (llvm::is_contained(reductionAxes, static_cast<int64_t>(axis)))
        continue;
      auto extent = cast<PhysicalExprAttr>(attribute);
      FailureOr<ParameterAttr> parameter =
          regionContractionParameter(kernel, extent);
      PhysicalRangeFact ranges = analysis.axisRanges(operand, axis);
      bool subregion = llvm::any_of(ranges.roots, [](MakeRangeOp range) {
        return range->hasAttr(sourceSubregionAttr);
      });
      if (failed(parameter) && !subregion)
        continue;
      auto mapping = cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
      PhysicalSourceAxis source = sourceAxisIdentity(mapping);
      if (result.source && !(*result.source == source))
        return failure();
      if (succeeded(parameter) && result.parameter &&
          result.parameter != *parameter)
        return failure();
      if (succeeded(parameter))
        result.parameter = *parameter;
      result.present = true;
      result.source = source;
      if (subregion &&
          llvm::any_of(ranges.roots, [](MakeRangeOp range) {
            return !range->hasAttr(sourceSubregionAttr);
          })) {
        return failure();
      }
    }
    return result;
  };
  FailureOr<SegmentFact> lhsSegment = operandSegment(
      contract.getLhs(), contract.getLhsReductionAxes());
  FailureOr<SegmentFact> rhsSegment = operandSegment(
      contract.getRhs(), contract.getRhsReductionAxes());
  if (failed(lhsSegment) || failed(rhsSegment))
    return contract.emitOpError(
               "operand carries an ambiguous region-contraction segment relation"),
           failure();
  if (lhsSegment->present == rhsSegment->present)
    return false;
  SegmentFact &segment = lhsSegment->present ? *lhsSegment : *rhsSegment;
  // A logical subregion only identifies the operand-local member relation; it
  // is not itself an executable region-contraction segment.  Native segment
  // coverage requires the typed parameter created by the structured traversal
  // realization.  Otherwise leave the contract to ordinary M/N/K blocking,
  // which materializes the subregion traversal explicitly.
  if (!segment.parameter)
    return false;
  scf::ForOp segmentLoop =
      enclosingRegionContractionSegment(contract.getOperation());
  if (segmentLoop && queryParameter(segmentLoop.getStep()) !=
                         segment.parameter)
    return contract.emitOpError(
               "operand extent disagrees with its enclosing region-contraction segment"),
           failure();

  bool lhsInvariant = !lhsSegment->present;
  bool rhsInvariant = !rhsSegment->present;
  if (segmentLoop) {
    DominanceInfo dominance(kernel);
    bool lhsDominates = dominance.dominates(lhsLoad.getOperation(),
                                            segmentLoop.getOperation());
    bool rhsDominates = dominance.dominates(rhsLoad.getOperation(),
                                            segmentLoop.getOperation());
    if (lhsDominates != lhsInvariant || rhsDominates != rhsInvariant)
      return contract.emitOpError(
                 "typed segment ownership disagrees with lexical load invariance"),
             failure();
  }

  Value segmented = lhsInvariant ? contract.getRhs() : contract.getLhs();
  if (reductionUsesOwnershipExtent(
          contract, segmented,
          lhsInvariant ? contract.getRhsReductionAxes()
                       : contract.getLhsReductionAxes())) {
    unsigned lhsAxis = contract.getLhsReductionAxes().front();
    unsigned rhsAxis = contract.getRhsReductionAxes().front();
    if (contract.getLhs().getType().getShape()[lhsAxis] !=
        contract.getRhs().getType().getShape()[rhsAxis])
      return false;
    SmallVector<MakeRangeOp> ranges;
    for (auto [operand, axis] :
         {std::pair<Value, unsigned>{contract.getLhs(), lhsAxis},
          std::pair<Value, unsigned>{contract.getRhs(), rhsAxis}}) {
      FailureOr<MakeRangeOp> range =
          queryExactLogicalRange(analysis.axisRanges(operand, axis));
      if (failed(range) || !isUnitStepRange(*range) ||
          !isZeroPastLogicalEnd(operand, *range))
        return false;
      ranges.push_back(*range);
    }
    OpBuilder builder(contract);
    Location location = contract.getLoc();
    Value complete;
    for (MakeRangeOp range : ranges) {
      Value span = binary(builder, location, builder.getIndexType(),
                          range.getLogicalStop(), range.getLogicalStart(),
                          BinaryOperator::Subtract);
      Value beginsAtStart = compare(
          builder, location, builder.getI1Type(), range.getStart(),
          range.getLogicalStart(), ComparePredicate::Eq);
      Value coversExtent = compare(builder, location, builder.getI1Type(),
                                    range.getExtent(), span,
                                    ComparePredicate::Ge);
      Value covered = binary(builder, location, builder.getI1Type(),
                              beginsAtStart, coversExtent,
                              BinaryOperator::LogicalAnd);
      complete = complete
                     ? binary(builder, location, builder.getI1Type(), complete,
                              covered, BinaryOperator::LogicalAnd)
                     : covered;
    }
    auto choice = builder.create<scf::IfOp>(
        location, TypeRange{contract.getResult().getType()}, complete,
        /*withElseRegion=*/true);
    OpBuilder native = choice.getThenBodyBuilder();
    auto product = cast<ContractOp>(native.clone(*contract));
    native.create<scf::YieldOp>(location, product.getResult());
    OpBuilder blocked = choice.getElseBodyBuilder();
    auto reduction = cast<ContractOp>(blocked.clone(*contract));
    blocked.create<scf::YieldOp>(location, reduction.getResult());
    if (failed(realizeReductionTraversal(reduction, kernel, replayed)))
      return failure();
    contract.getResult().replaceAllUsesWith(choice.getResult(0));
    contract.erase();
    return true;
  }

  Value invariant = lhsInvariant ? contract.getLhs() : contract.getRhs();
  unsigned reductionAxis = static_cast<unsigned>(
      lhsInvariant ? contract.getLhsReductionAxes().front()
                   : contract.getRhsReductionAxes().front());
  auto invariantType = cast<FragmentType>(invariant.getType());
  auto reductionMapping =
      cast<AxisMapAttr>(invariantType.getAxisMaps()[reductionAxis]);
  LoadOp invariantLoad = lhsInvariant ? lhsLoad : rhsLoad;
  PhysicalAxisProjection loadProjection = queryFragmentAxis(
      invariantLoad.getResult().getType(),
      sourceAxisIdentity(reductionMapping));
  if (!loadProjection.isExact() ||
      failed(realizeFullCoverageDimension(
          kernel, invariantLoad.getResult(), loadProjection.fragmentAxis)))
    return contract.emitOpError(
        "structured contraction invariant has no exact full-coverage reduction realization");
  if (failed(markNativeCoverage(kernel, contract)))
    return failure();
  return true;
}

LogicalResult realizeReductionTraversal(ContractOp contract,
                                        func::FuncOp kernel,
                                        SmallVectorImpl<ContractOp> &replayed) {
  if (contract.getLhsReductionAxes().size() != 1 ||
      contract.getRhsReductionAxes().size() != 1)
    return contract.emitOpError(
        "reduction-only contraction blocking requires one reduction pair");

  FragmentType lhsType = contract.getLhs().getType();
  FragmentType rhsType = contract.getRhs().getType();
  FragmentType resultType = contract.getResult().getType();
  // Selected tiles can be clamped by a launch dimension. Their typed
  // realization, rather than a constant-only expression, owns the free axes.
  if (!freeAxesReadyForReductionTraversal(contract, kernel))
    return contract.emitOpError(
        "reduction-only contraction blocking requires already-selected free-axis fragments");

  unsigned lhsReduction = contract.getLhsReductionAxes().front();
  unsigned rhsReduction = contract.getRhsReductionAxes().front();
  FailureOr<AxisMapAttr> lhsMap = queryAxisMap(lhsType, lhsReduction);
  FailureOr<AxisMapAttr> rhsMap = queryAxisMap(rhsType, rhsReduction);
  if (failed(lhsMap) || failed(rhsMap))
    return contract.emitOpError(
        "reduction-only contraction blocking lost paired reduction provenance");
  PhysicalProgramAnalysis physicalAnalysis(kernel);
  SmallVector<MakeRangeOp> lhsRanges;
  SmallVector<MakeRangeOp> rhsRanges;
  if (failed(collectPairedReductionRanges(contract, lhsRanges, rhsRanges))) {
    PhysicalRangeFact lhsFact =
        physicalAnalysis.axisRanges(contract.getLhs(), lhsReduction);
    PhysicalRangeFact rhsFact =
        physicalAnalysis.axisRanges(contract.getRhs(), rhsReduction);
    InFlightDiagnostic diagnostic = contract.emitOpError(
        "reduction-only contraction blocking requires explicit paired ranges");
    diagnostic << "; lhs_state=" << static_cast<unsigned>(lhsFact.state)
               << ", lhs_roots=" << lhsFact.roots.size()
               << ", lhs_blockers=" << lhsFact.blockers.size()
               << ", rhs_state=" << static_cast<unsigned>(rhsFact.state)
               << ", rhs_roots=" << rhsFact.roots.size()
               << ", rhs_blockers=" << rhsFact.blockers.size()
               << ", lhs_type=" << contract.getLhs().getType()
               << ", rhs_type=" << contract.getRhs().getType();
    for (Operation *blocker : lhsFact.blockers)
      diagnostic << ", lhs_blocker=" << blocker->getName();
    for (Operation *blocker : rhsFact.blockers)
      diagnostic << ", rhs_blocker=" << blocker->getName();
    return failure();
  }
  MakeRangeOp lhsRange = lhsRanges.front();
  MakeRangeOp rhsRange = rhsRanges.front();
  Value logicalEnd = (lhsRange).getLogicalStop();
  FailureOr<Value> lhsStep = scalarSource(lhsRange.getStep());
  FailureOr<Value> rhsStep = scalarSource(rhsRange.getStep());
  if (failed(lhsStep) || failed(rhsStep) ||
      !isIntegerConstant(*lhsStep, 1) || !isIntegerConstant(*rhsStep, 1))
    return contract.emitOpError(
        "reduction-only contraction blocking requires a unit-step range with an exact logical end");

  // These are logical full ranges, even when their current fragment has a
  // constant padded extent. Establish the identity before replaying the
  // producer graph into K tiles; masking loads alone does not neutralize exp.
  if (failed(neutralizeFullCoverageOperand(
          contract, contract.getLhsMutable(), contract.getLhsReductionAxes(),
          kernel)) ||
      failed(neutralizeFullCoverageOperand(
          contract, contract.getRhsMutable(), contract.getRhsReductionAxes(),
          kernel)))
    return failure();

  std::string suffix =
      ("_" + Twine(lhsMap->getSourceId()) + "_" +
       Twine(rhsMap->getSourceId()))
          .str();
  auto blockKRef = getOrCreatePhysicalParameter(
      kernel, "BLOCK_K" + suffix, ParameterRole::Reduction,
      ParameterCategory::Contraction,
      lhsType.getElementType().getIntOrFloatBitWidth(),
      contractionReductionCandidates);
  if (failed(blockKRef))
    return failure();
  ParameterAttr blockK = lookupParameter(kernel, *blockKRef);
  FailureOr<int64_t> lhsDimension = queryRangeDimension(lhsRange);
  FailureOr<int64_t> rhsDimension = queryRangeDimension(rhsRange);
  if (succeeded(lhsDimension) && succeeded(rhsDimension) &&
      *lhsDimension == *rhsDimension)
    if (failed(updateParameter(kernel, blockK.withBinding(blockK.getBinding().withDimension(
            IntegerAttr::get(IntegerType::get(kernel.getContext(), 64), *lhsDimension))))))
      return failure();
  auto boundedK = boundedTraversalChunk(
      lookupParameter(kernel, *blockKRef), lhsRange);
  if (failed(boundedK)) return failure();
  PhysicalExprAttr unitK = *boundedK;
  OpBuilder parameterBuilder(&kernel.front(), kernel.front().begin());
  Value blockKValue = unitK.getKind() == PhysicalExprKind::Parameter
      ? materializeParameter(parameterBuilder, contract.getLoc(), *blockKRef)
            .getResult()
      : parameterBuilder.create<PhysicalExprOp>(
            contract.getLoc(), parameterBuilder.getIndexType(), unitK).getResult();
  MLIRContext *context = kernel.getContext();
  auto lhsIndexType = fragmentType(
      context, IndexType::get(context), {unitK},
      {cast<AxisMapAttr>(lhsRange.getResult().getType().getAxisMaps()[0])},
      lhsType.getOwner());
  auto rhsIndexType = fragmentType(
      context, IndexType::get(context), {unitK},
      {cast<AxisMapAttr>(rhsRange.getResult().getType().getAxisMaps()[0])},
      rhsType.getOwner());

  Location location = contract.getLoc();
  OpBuilder builder(contract);
  Value one = builder.create<arith::ConstantIndexOp>(location, 1);
  bool bodyFailed = false;
  SmallVector<ContractOp> nativeProducts;
  auto loop = createTraversalLoop(builder,
      location, lhsRange.getLogicalStart(), logicalEnd, blockKValue,
      ValueRange{contract.getAccumulator()},
      [&](OpBuilder &nested, Location nestedLocation, Value kStart,
          ValueRange carries) {
        Value lhsK = nested.create<MakeRangeOp>(
            nestedLocation, lhsIndexType, kStart, blockKValue, one,
            lhsRange.getLogicalStart(), lhsRange.getLogicalStop(),
            lhsRange.getSourceId(), lhsRange.getSourceAxis(),
            lhsRange.getDerived());
        inheritRangeAuthority(lhsK, lhsRange);
        Value rhsOffset = binary(
            nested, nestedLocation, nested.getIndexType(), kStart,
            lhsRange.getLogicalStart(), BinaryOperator::Subtract);
        Value rhsStart = binary(
            nested, nestedLocation, nested.getIndexType(),
            rhsRange.getLogicalStart(), rhsOffset, BinaryOperator::Add);
        Value rhsK = nested.create<MakeRangeOp>(
            nestedLocation, rhsIndexType, rhsStart, blockKValue, one,
            rhsRange.getLogicalStart(), rhsRange.getLogicalStop(),
            rhsRange.getSourceId(), rhsRange.getSourceAxis(),
            rhsRange.getDerived());
        inheritRangeAuthority(rhsK, rhsRange);
        FailureOr<Value> lhsTail = buildRangeTailPredicate(
            nested, nestedLocation, lhsK, lhsRange);
        FailureOr<Value> rhsTail = buildRangeTailPredicate(
            nested, nestedLocation, rhsK, rhsRange);
        if (failed(lhsTail) || failed(rhsTail)) {
          bodyFailed = true;
          return;
        }
        IRMapping lhsReplay;
        IRMapping rhsReplay;
        for (MakeRangeOp range : lhsRanges)
          lhsReplay.map(range.getResult(), lhsK);
        for (MakeRangeOp range : rhsRanges)
          rhsReplay.map(range.getResult(), rhsK);
        FailureOr<Value> lhs = materializeSourceRanges(
            nested, nestedLocation, contract.getLhs(), unitK, lhsRanges, lhsK,
            lhsReplay, contract.getOperation());
        FailureOr<Value> rhs = materializeSourceRanges(
            nested, nestedLocation, contract.getRhs(), unitK, rhsRanges, rhsK,
            rhsReplay, contract.getOperation());
        if (failed(lhs) || failed(rhs)) {
          bodyFailed = true;
          return;
        }
        if (failed(appendTailValidity(nestedLocation, contract.getLhs(),
                                      lhsRanges, *lhsTail, lhsReplay)) ||
            failed(appendTailValidity(nestedLocation, contract.getRhs(),
                                      rhsRanges, *rhsTail, rhsReplay))) {
          bodyFailed = true;
          return;
        }
        auto product = nested.create<ContractOp>(
            nestedLocation, resultType, *lhs, *rhs, carries.front(),
            contract.getLhsReductionAxes(), contract.getRhsReductionAxes(),
            contract.getLhsBatchAxes(), contract.getRhsBatchAxes());
        nativeProducts.push_back(product);
        if (Attribute origin = contract->getAttr(originAttr))
          product->setAttr(originAttr, origin);
        nested.create<scf::YieldOp>(nestedLocation, product.getResult());
      });
  if (bodyFailed) {
    loop.erase();
    return contract.emitOpError(
        "reduction-only contraction blocking could not replay its load graph");
  }
  loop.walk([&](ContractOp nested) {
    if (!llvm::is_contained(nativeProducts, nested))
      replayed.push_back(nested);
  });

  contract.getResult().replaceAllUsesWith(loop.getResult(0));
  contract.erase();
  return success();
}

FailureOr<bool> realizeFullResultTraversal(
    ContractOp contract, func::FuncOp kernel,
    SmallVectorImpl<ContractOp> &pending) {
  std::string reason;
  auto axes = queryContractionAxes(contract, &reason);
  if (!axes)
    return contract.emitOpError("invalid full-result contraction axis schema: ")
               << reason, failure();
  if (!axes->batch.empty())
    return false;
  FragmentType resultType = contract.getResult().getType();
  auto capabilities = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  uint64_t retainedBytes =
      (resultType.getElementType().getIntOrFloatBitWidth() + 7) / 8;
  bool boundedResult = static_cast<bool>(capabilities);
  for (Attribute attribute : resultType.getShape()) {
    auto extent = cast<PhysicalExprAttr>(attribute);
    auto width = constantPhysicalExpression(extent);
    if (!width && extent.getKind() ==
                      PhysicalExprKind::Parameter) {
      auto parameter = parameterForExtent(kernel, extent);
      if (succeeded(parameter)) {
        auto candidates = (*parameter).getCandidates().asArrayRef();
        if (!candidates.empty())
          width = *std::min_element(candidates.begin(), candidates.end());
      }
    }
    if (!width || *width <= 0 ||
        retainedBytes > std::numeric_limits<uint64_t>::max() / *width) {
      boundedResult = false;
      break;
    }
    retainedBytes *= *width;
  }
  // Introduce retained-result traversal only when even the smallest existing
  // fragment exceeds the local budget; larger candidates do not force a loop
  // on otherwise small ownership tiles.
  const bool largeRetainedResult =
      boundedResult && retainedBytes >
                           static_cast<uint64_t>(
                               capabilities.getMaxDynamicSharedMemoryPerBlock());
  auto needsTraversal = [&](PhysicalExprAttr extent) {
    return isFullCoverageExtent(contract, extent) ||
           (largeRetainedResult && extent.getKind() ==
                                      PhysicalExprKind::Constant &&
            extent.getValue() > 1);
  };
  if (llvm::none_of(resultType.getShape(), [&](Attribute extent) {
        return needsTraversal(cast<PhysicalExprAttr>(extent));
      }))
    return false;
  SmallVector<std::pair<OpOperand *, unsigned>> freeAxes(resultType.getShape().size());
  for (auto [resultAxis, source] : llvm::enumerate(axes->results)) {
    OpOperand *operand = source.operand == ContractionOperand::Lhs
                            ? &contract.getLhsMutable()
                            : &contract.getRhsMutable();
    auto type = cast<FragmentType>(operand->get().getType());
    if (resultType.getShape()[resultAxis] != type.getShape()[source.axis])
      return false;
    freeAxes[resultAxis] = {operand, source.axis};
  }
  // Slicing may replay loads at the contraction.  External views can alias,
  // so retain their original snapshots across every intervening write.
  if (!canReplayContractionReads(contract))
    return false;

  for (auto [resultAxis, selected] : llvm::enumerate(freeAxes)) {
    auto fullExtent = cast<PhysicalExprAttr>(resultType.getShape()[resultAxis]);
    if (!needsTraversal(fullExtent))
      continue;
    PhysicalProgramAnalysis analysis(kernel);
    PhysicalRangeFact fact = analysis.axisRanges(selected.first->get(), selected.second);
    auto traversal = analysis.lockstepRanges(fact.roots);
    FailureOr<MakeRangeOp> authority =
        fact.state != PhysicalFactState::Unknown && fact.blockers.empty() &&
                traversal.isExact()
            ? FailureOr<MakeRangeOp>(traversal.authority)
            : FailureOr<MakeRangeOp>(failure());
    FailureOr<ParameterAttr> fullParameter = parameterForExtent(kernel, fullExtent);
    FailureOr<AxisMapAttr> axisMap =
        queryAxisMap(selected.first->get().getType(), selected.second);
    if (failed(authority) || failed(axisMap) ||
        (failed(fullParameter) && fullExtent.getKind() !=
                                     PhysicalExprKind::Constant) ||
        !fact.unitStep ||
        !samePhysicalScalarExpression((*authority).getStart(),
                                      (*authority).getLogicalStart()))
      continue;
    if (failed(fullParameter)) {
      auto cardinality = constantLogicalRangeCardinality(*authority);
      UniformValueAnalysis constants(describeUniformValue);
      auto actualExtent = dyn_cast_or_null<IntegerAttr>(
          constants.evaluate((*authority).getExtent()));
      if (!cardinality || *cardinality <= 0 ||
          *cardinality > fullExtent.getValue() || !actualExtent ||
          actualExtent.getInt() != fullExtent.getValue())
        continue;
    }
    bool lhsAxis = selected.first == &contract.getLhsMutable();
    std::string name =
        ((lhsAxis ? "BLOCK_M_CACHE_" : "BLOCK_N_CACHE_") +
         Twine(axisMap->getSourceId()) + "_" + Twine(axisMap->getSourceAxis()) +
         "_" + Twine(axisMap->getDerived() ? 1 : 0) + "_D" +
         Twine(axisMap->getDimensionId()))
            .str();
    Type inputElement = cast<FragmentType>(selected.first->get().getType())
                            .getElementType();
    auto blockRef = getOrCreatePhysicalParameter(
        kernel, name,
        lhsAxis ? ParameterRole::OwnershipM : ParameterRole::OwnershipN,
        ParameterCategory::Contraction,
        inputElement.isIndex() ? 64 : inputElement.getIntOrFloatBitWidth(),
        {32, 64, 128, 256, 512});
    if (failed(blockRef))
      return failure();
    ParameterAttr block = lookupParameter(kernel, *blockRef);
    if (FailureOr<int64_t> dimension = queryRangeDimension(*authority);
        succeeded(dimension))
      if (failed(updateParameter(kernel, block.withBinding(block.getBinding().withDimension(
              IntegerAttr::get(IntegerType::get(kernel.getContext(), 64), *dimension))))))
        return failure();
    OpBuilder parameterBuilder(&kernel.front(), kernel.front().begin());
    Value blockValue = materializeParameter(parameterBuilder, contract.getLoc(), *blockRef);
    auto tileExtent = parameterExpression(kernel.getContext(), name);
    SmallVector<Attribute> tileShape(resultType.getShape().begin(),
                                     resultType.getShape().end());
    tileShape[resultAxis] = tileExtent;
    auto tileResultType = FragmentType::get(
        kernel.getContext(), resultType.getElementType(),
        ArrayAttr::get(kernel.getContext(), tileShape), resultType.getAxisMaps(),
        resultType.getValidity(), resultType.getOwner());
    auto fullRangeType = FragmentType::get(
        kernel.getContext(), IndexType::get(kernel.getContext()),
        ArrayAttr::get(kernel.getContext(), {fullExtent}),
        ArrayAttr::get(kernel.getContext(), {AxisMapAttr::get(
            kernel.getContext(), axisMap->getSourceId(), axisMap->getSourceAxis(),
            axisMap->getDimensionId(), 0, axisMap->getDerived())}),
        resultType.getValidity(), resultType.getOwner());
    auto tileRangeType = FragmentType::get(
        kernel.getContext(), fullRangeType.getElementType(),
        ArrayAttr::get(kernel.getContext(), {tileExtent}), fullRangeType.getAxisMaps(),
        fullRangeType.getValidity(), fullRangeType.getOwner());
    auto indexType = FragmentType::get(
        kernel.getContext(), IndexType::get(kernel.getContext()),
        resultType.getShape(), resultType.getAxisMaps(), resultType.getValidity(),
        resultType.getOwner());
    auto predicateType = FragmentType::get(
        kernel.getContext(), IntegerType::get(kernel.getContext(), 1),
        resultType.getShape(), resultType.getAxisMaps(), resultType.getValidity(),
        resultType.getOwner());
    OpBuilder builder(contract);
    Location location = contract.getLoc();
    Value fullSize = succeeded(fullParameter)
                         ? materializeParameter(builder, location, fullParameter->getReference()).getResult()
                         : Value(builder.create<arith::ConstantIndexOp>(
                               location, fullExtent.getValue()));
    Value fullRange = builder.create<MakeRangeOp>(
        location, fullRangeType, (*authority).getLogicalStart(),
        fullSize, (*authority).getStep(),
        (*authority).getLogicalStart(), (*authority).getLogicalStop(),
        axisMap->getSourceId(), axisMap->getSourceAxis(), axisMap->getDerived());
    inheritRangeAuthority(fullRange, *authority);
    Value coordinates = broadcastAxis(builder, location, indexType, fullRange,
                                      static_cast<unsigned>(resultAxis));
    // A tile that covers the retained output needs no slice/update traversal.
    // The native branch keeps the original full shape, so an oversized tile
    // does not pad the product or assemble a redundant partial result. This
    // launch-uniform binding becomes a compile-time branch in the target DSL;
    // its original snapshots and accumulator still define the entire product.
    scf::IfOp fullTile;
    PhysicalProgramAnalysis nativeAnalysis(kernel);
    bool hasReductionSeed = false;
    for (auto [operand, axes] :
         {std::pair{contract.getLhs(), contract.getLhsReductionAxes()},
          std::pair{contract.getRhs(), contract.getRhsReductionAxes()}})
      for (int64_t axis : axes)
        hasReductionSeed |=
            nativeAnalysis.axisRealization(operand, axis).constructionScalarSeed;
    // A scalar construction seed is not a selected reduction tile. Retain
    // the traversal so its reduction can be blocked before using native MMA.
    if (!hasReductionSeed && !fullReductionNeedsTraversal(contract)) {
      Value covers = builder.create<CompareOp>(location, builder.getI1Type(),
          fullSize, blockValue, ComparePredicate::Le);
      // Keep the full shapes reachable by the old equality branch. Domain
      // membership also preserves padding for undersized or non-native full
      // extents, without assuming a contiguous power-of-two candidate domain.
      auto candidates = block.getCandidates().asArrayRef();
      Value nativeExtent;
      if (auto fixed = constantPhysicalExpression(fullExtent)) {
        nativeExtent = builder.create<arith::ConstantIntOp>(
            location, llvm::is_contained(candidates, *fixed), 1);
      } else {
        for (int64_t candidate : candidates) {
          Value width = builder.create<arith::ConstantIndexOp>(location, candidate);
          Value equal = builder.create<CompareOp>(location, builder.getI1Type(),
              fullSize, width, ComparePredicate::Eq);
          nativeExtent = nativeExtent
              ? builder.create<BinaryOp>(location, builder.getI1Type(),
                    nativeExtent, equal, BinaryOperator::LogicalOr).getResult()
              : equal;
        }
      }
      Value fits = builder.create<BinaryOp>(location, builder.getI1Type(),
          covers, nativeExtent, BinaryOperator::LogicalAnd);
      fullTile = builder.create<scf::IfOp>(location, TypeRange{resultType}, fits, true);
      builder.setInsertionPointToStart(fullTile.thenBlock());
      auto native = cast<ContractOp>(builder.clone(*contract));
      if (failed(markNativeCoverage(kernel, native))) return failure();
      builder.create<scf::YieldOp>(location, native.getResult());
      builder.setInsertionPointToStart(fullTile.elseBlock());
    }
    // Each iteration fills one disjoint slice of an immutable full result.
    // The operand and dot fragments retain their independent, bounded tiles.
    auto loop = builder.create<scf::ForOp>(
        location, (*authority).getLogicalStart(), (*authority).getLogicalStop(),
        blockValue, ValueRange{contract.getAccumulator()},
        [](OpBuilder &body, Location location, Value, ValueRange carries) {
          body.create<scf::YieldOp>(location, carries);
        });
    Operation *yield = loop.getBody()->getTerminator();
    OpBuilder nested(yield);
    Value tileRange = nested.create<MakeRangeOp>(
        location, tileRangeType, loop.getInductionVar(), blockValue,
        (*authority).getStep(), (*authority).getLogicalStart(),
        (*authority).getLogicalStop(), axisMap->getSourceId(),
        axisMap->getSourceAxis(), axisMap->getDerived());
    inheritRangeAuthority(tileRange, *authority);
    FailureOr<Value> tail =
        buildRangeTailPredicate(nested, location, tileRange, *authority);
    if (failed(tail))
      return failure();
    Value origin = nested.create<BroadcastOp>(
        location, tileRangeType, (*authority).getLogicalStart());
    Value ordinal = binary(nested, location, tileRangeType, tileRange, origin,
                           BinaryOperator::Subtract);
    SmallVector<Value> operands;
    for (OpOperand *operand :
         {&contract.getLhsMutable(), &contract.getRhsMutable(),
          &contract.getAccumulatorMutable()}) {
      Value value = operand->get();
      bool accumulator = operand == &contract.getAccumulatorMutable();
      if (operand != selected.first && !accumulator) {
        operands.push_back(value);
        continue;
      }
      unsigned axis = accumulator ? resultAxis : selected.second;
      auto source = cast<FragmentType>(value.getType());
      SmallVector<Attribute> shape(source.getShape().begin(), source.getShape().end());
      shape[axis] = tileExtent;
      auto slicedType = FragmentType::get(
          kernel.getContext(), source.getElementType(), nested.getArrayAttr(shape),
          source.getAxisMaps(), source.getValidity(), source.getOwner());
      FailureOr<Value> fill = materializeZeroFragment(nested, location, slicedType);
      if (failed(fill))
        return failure();
      if (accumulator && isLiteralZeroProjection(value)) {
        operands.push_back(*fill);
        continue;
      }
      auto sourceMap = cast<AxisMapAttr>(source.getAxisMaps()[axis]);
      auto coordinateType = FragmentType::get(
          kernel.getContext(), nested.getIndexType(), nested.getArrayAttr({tileExtent}),
          nested.getArrayAttr({AxisMapAttr::get(
              kernel.getContext(), sourceMap.getSourceId(), sourceMap.getSourceAxis(),
              sourceMap.getDimensionId(), 0, sourceMap.getDerived())}),
          source.getValidity(), source.getOwner());
      Value coordinate = nested.create<ReshapeOp>(
          location, coordinateType, ordinal, nested.getArrayAttr({ReshapeGroupAttr::get(
              kernel.getContext(), nested.getDenseI64ArrayAttr({0}),
              nested.getDenseI64ArrayAttr({0}))}));
      auto valid = projectPredicateToFragmentAxis(nested, location, *tail, slicedType, axis);
      if (failed(valid))
        return failure();
      auto coordinatePredicate = FragmentType::get(
          kernel.getContext(), nested.getI1Type(), coordinateType.getShape(),
          coordinateType.getAxisMaps(), coordinateType.getValidity(), coordinateType.getOwner());
      Value zero = nested.create<arith::ConstantIndexOp>(location, 0);
      Value lower = nested.create<BroadcastOp>(location, coordinateType, zero);
      Value upper = nested.create<BroadcastOp>(location, coordinateType, fullSize);
      Value bounded = binary(nested, location, coordinatePredicate,
          compare(nested, location, coordinatePredicate, coordinate, lower, ComparePredicate::Ge),
          compare(nested, location, coordinatePredicate, coordinate, upper, ComparePredicate::Lt),
          BinaryOperator::LogicalAnd);
      auto projectedBounds = projectPredicateToFragmentAxis(nested, location, bounded, slicedType, axis);
      if (failed(projectedBounds))
        return failure();
      Value activeSlice = binary(nested, location, (*valid).getType(), *valid,
                                 *projectedBounds, BinaryOperator::LogicalAnd);
      operands.push_back(nested.create<GatherOp>(
          location, slicedType, value, ValueRange{coordinate}, activeSlice, *fill,
          ArrayRef<int64_t>{static_cast<int64_t>(axis)}));
    }
    FailureOr<Value> accumulator = projectPhysicalValueToSchema(
        nested, location, operands[2], tileResultType);
    if (failed(accumulator))
      return contract.emitOpError(
          "full-result traversal could not project its accumulator slice");
    auto tile = nested.create<ContractOp>(
        location, tileResultType, operands[0], operands[1],
        *accumulator, contract.getLhsReductionAxes(), contract.getRhsReductionAxes(),
        contract.getLhsBatchAxes(), contract.getRhsBatchAxes());
    if (Attribute origin = contract->getAttr(originAttr))
      tile->setAttr(originAttr, origin);
    Value start = nested.create<BroadcastOp>(location, indexType,
                                            loop.getInductionVar());
    Value local = binary(nested, location, indexType, coordinates, start,
                         BinaryOperator::Subtract);
    Value zero = nested.create<arith::ConstantIndexOp>(location, 0);
    Value zeroes = nested.create<BroadcastOp>(location, indexType, zero);
    Value width = nested.create<BroadcastOp>(location, indexType, blockValue);
    Value stop = nested.create<BroadcastOp>(location, indexType,
                                           (*authority).getLogicalStop());
    Value lower = compare(nested, location, predicateType, local, zeroes,
                          ComparePredicate::Ge);
    Value upper = compare(nested, location, predicateType, local, width,
                          ComparePredicate::Lt);
    Value active = compare(nested, location, predicateType, coordinates, stop,
                           ComparePredicate::Lt);
    Value valid = binary(nested, location, predicateType, lower, upper,
                         BinaryOperator::LogicalAnd);
    valid = binary(nested, location, predicateType, valid, active,
                   BinaryOperator::LogicalAnd);
    Value assembled = nested.create<GatherOp>(
        location, resultType, tile.getResult(), ValueRange{local}, valid,
        loop.getRegionIterArgs().front(),
        ArrayRef<int64_t>{static_cast<int64_t>(resultAxis)});
    yield->setOperands(ValueRange{assembled});
    loop.walk([&](ContractOp product) { pending.push_back(product); });
    if (fullTile) {
      builder.setInsertionPointAfter(loop);
      builder.create<scf::YieldOp>(location, loop.getResult(0));
    }
    contract.getResult().replaceAllUsesWith(fullTile ? fullTile.getResult(0) : loop.getResult(0));
    contract.erase();
    return true;
  }
  return false;
}

LogicalResult realizeSparseReductionTraversal(SparseContractOp contract,
                                              func::FuncOp kernel) {
  if (contract.getLhsReductionAxes() != ArrayRef<int64_t>{1} ||
      contract.getRhsReductionAxes() != ArrayRef<int64_t>{0} ||
      !contract.getLhsBatchAxes().empty() ||
      !contract.getRhsBatchAxes().empty() ||
      contract.getFormat().getCompressionAxis() != 1)
    return contract.emitOpError(
        "shared sparse blocking requires rank-two [M,KC] x [K,N] physical axes");
  auto compressedType = dyn_cast<FragmentType>(contract.getCompressed().getType());
  auto rhsType = dyn_cast<FragmentType>(contract.getRhs().getType());
  auto resultType = dyn_cast<FragmentType>(contract.getResult().getType());
  if (!compressedType || !rhsType || !resultType ||
      compressedType.getShape().size() != 2 || rhsType.getShape().size() != 2 ||
      resultType.getShape().size() != 2 ||
      llvm::any_of(resultType.getShape(), [](Attribute extent) {
        return !isCompileTimeExtent(cast<PhysicalExprAttr>(extent));
      }))
    return contract.emitOpError(
        "shared sparse blocking requires selected rank-two free-axis fragments");

  FailureOr<AxisMapAttr> compressedMap = queryAxisMap(compressedType, 1);
  FailureOr<AxisMapAttr> denseMap = queryAxisMap(rhsType, 0);
  if (failed(compressedMap) || failed(denseMap))
    return contract.emitOpError(
        "shared sparse blocking lost compressed/dense reduction provenance");

  SmallVector<Value> metadataComponents;
  RecordType metadataRecordType;
  MakeRecordOp metadataRecord;
  if (auto component = dyn_cast<FragmentType>(contract.getMetadata().getType())) {
    (void)component;
    metadataComponents.push_back(contract.getMetadata());
  } else {
    metadataRecordType = dyn_cast<RecordType>(contract.getMetadata().getType());
    metadataRecord = contract.getMetadata().getDefiningOp<MakeRecordOp>();
    if (!metadataRecordType || !metadataRecord ||
        metadataRecord.getFields().size() !=
            metadataRecordType.getFieldTypes().size())
      return contract.emitOpError(
          "shared sparse blocking requires materialized metadata components");
    metadataComponents.append(metadataRecord.getFields().begin(),
                              metadataRecord.getFields().end());
  }
  if (metadataComponents.empty())
    return contract.emitOpError(
        "shared sparse blocking requires at least one metadata component");
  auto metadataComponentType =
      dyn_cast<FragmentType>(metadataComponents.front().getType());
  if (!metadataComponentType || metadataComponentType.getShape().size() != 2)
    return contract.emitOpError(
        "shared sparse blocking requires rank-two metadata fragments");
  FailureOr<AxisMapAttr> metadataMap = queryAxisMap(metadataComponentType, 1);
  if (failed(metadataMap))
    return contract.emitOpError(
        "shared sparse blocking lost metadata-group provenance");
  for (Value component : llvm::drop_begin(metadataComponents)) {
    auto type = dyn_cast<FragmentType>(component.getType());
    FailureOr<AxisMapAttr> mapping =
        type && type.getShape().size() == 2
            ? queryAxisMap(type, 1)
            : FailureOr<AxisMapAttr>(failure());
    if (!type || failed(mapping) ||
        !(sourceAxisIdentity(*mapping) == sourceAxisIdentity(*metadataMap)))
      return contract.emitOpError(
          "shared sparse metadata components do not traverse one group relation");
  }

  PhysicalProgramAnalysis physicalAnalysis(kernel);
  auto rangesFor = [&](Value value, unsigned fragmentAxis, AxisMapAttr mapping,
                       SmallVectorImpl<MakeRangeOp> &ranges) {
    PhysicalRangeFact fact = physicalAnalysis.axisRanges(value, fragmentAxis);
    if (failed(queryExactLogicalRange(fact)) &&
        queryFragmentAxes(value.getType(), sourceAxisIdentity(mapping)).size() ==
            1)
      fact = physicalAnalysis.sourceRanges(value, sourceAxisIdentity(mapping));
    if (failed(queryExactLogicalRange(fact)) || !fact.unitStep)
      return failure();
    ranges.assign(fact.roots.begin(), fact.roots.end());
    return llvm::all_of(ranges, [&](MakeRangeOp range) {
             return sourceAxisIdentity(range) == sourceAxisIdentity(mapping);
           })
               ? success()
               : failure();
  };
  SmallVector<MakeRangeOp> compressedRanges;
  SmallVector<MakeRangeOp> denseRanges;
  SmallVector<MakeRangeOp> metadataRanges;
  if (failed(rangesFor(contract.getCompressed(), 1, *compressedMap,
                       compressedRanges)) ||
      failed(rangesFor(contract.getRhs(), 0, *denseMap, denseRanges)))
    return contract.emitOpError(
        "shared sparse blocking requires explicit compressed and dense ranges");
  for (Value component : metadataComponents) {
    SmallVector<MakeRangeOp> componentRanges;
    if (failed(rangesFor(component, 1, *metadataMap, componentRanges)))
      return contract.emitOpError(
          "shared sparse blocking requires an explicit metadata-group range");
    for (MakeRangeOp range : componentRanges)
      if (!llvm::is_contained(metadataRanges, range))
        metadataRanges.push_back(range);
  }

  MakeRangeOp compressedRange = compressedRanges.front();
  MakeRangeOp denseRange = denseRanges.front();
  MakeRangeOp metadataRange = metadataRanges.front();
  if (llvm::any_of(metadataRanges, [&](MakeRangeOp range) {
        return !sameLogicalRange(range, metadataRange);
      }))
    return contract.emitOpError(
        "shared sparse metadata components have different logical group ranges");
  Value denseLogicalEnd = (denseRange).getLogicalStop();
  FailureOr<Value> compressedStep = scalarSource(compressedRange.getStep());
  FailureOr<Value> denseStep = scalarSource(denseRange.getStep());
  FailureOr<Value> metadataStep = scalarSource(metadataRange.getStep());
  if (failed(compressedStep) || failed(denseStep) ||
      failed(metadataStep) || !isIntegerConstant(*compressedStep, 1) ||
      !isIntegerConstant(*denseStep, 1) ||
      !isIntegerConstant(*metadataStep, 1))
    return contract.emitOpError(
        "shared sparse blocking requires unit-step ranges with an exact dense K end");

  MLIRContext *context = kernel.getContext();
  const int64_t groupSize = contract.getFormat().getKind() == 0 ? 2 : 4;
  std::string suffix =
      ("_sparse_" + Twine(compressedMap->getSourceId()) + "_" +
       Twine(denseMap->getSourceId()))
          .str();
  auto blockKRef = getOrCreatePhysicalParameter(
      kernel, "BLOCK_K" + suffix, ParameterRole::Reduction,
      ParameterCategory::Contraction,
      std::max(compressedType.getElementType().getIntOrFloatBitWidth(),
               rhsType.getElementType().getIntOrFloatBitWidth()),
      {32, 64, 128});
  if (failed(blockKRef))
    return failure();
  ParameterAttr blockK = lookupParameter(kernel, *blockKRef);
  if (FailureOr<int64_t> dimension = queryRangeDimension(denseRange);
      succeeded(dimension))
    if (failed(updateParameter(kernel, blockK.withBinding(blockK.getBinding().withDimension(
            IntegerAttr::get(IntegerType::get(context, 64), *dimension))))))
      return failure();
  OpBuilder parameterBuilder(&kernel.front(), kernel.front().begin());
  Value blockKValue = materializeParameter(parameterBuilder, contract.getLoc(), *blockKRef);

  PhysicalExprAttr unitDenseK = parameterExpression(
      context, blockK.getName().getValue());
  PhysicalExprAttr two = expression(context, PhysicalExprKind::Constant, 2);
  PhysicalExprAttr group =
      expression(context, PhysicalExprKind::Constant, groupSize);
  PhysicalExprAttr unitCompressedK = binaryExpression(
      context, PhysicalExprKind::FloorDiv, unitDenseK, two);
  PhysicalExprAttr unitMetadataK = binaryExpression(
      context, PhysicalExprKind::FloorDiv, unitDenseK, group);
  FragmentType compressedIndexType = fragmentType(
      context, IndexType::get(context), {unitCompressedK}, {*compressedMap},
      compressedType.getOwner());
  FragmentType denseIndexType = fragmentType(
      context, IndexType::get(context), {unitDenseK}, {*denseMap},
      rhsType.getOwner());
  FragmentType metadataIndexType = fragmentType(
      context, IndexType::get(context), {unitMetadataK}, {*metadataMap},
      metadataComponentType.getOwner());

  Location location = contract.getLoc();
  OpBuilder builder(contract);
  Value one = builder.create<arith::ConstantIndexOp>(location, 1);
  Value twoValue = builder.create<arith::ConstantIndexOp>(location, 2);
  Value groupValue =
      builder.create<arith::ConstantIndexOp>(location, groupSize);
  Value compressedExtent = builder.create<PhysicalExprOp>(
      location, builder.getIndexType(), unitCompressedK);
  Value metadataExtent = builder.create<PhysicalExprOp>(
      location, builder.getIndexType(), unitMetadataK);
  bool bodyFailed = false;
  auto loop = createTraversalLoop(builder,
      location, denseRange.getLogicalStart(), denseLogicalEnd,
      blockKValue, ValueRange{contract.getAccumulator()},
      [&](OpBuilder &nested, Location nestedLocation, Value denseStart,
          ValueRange carries) {
        Value denseOffset = binary(
            nested, nestedLocation, nested.getIndexType(), denseStart,
            denseRange.getLogicalStart(), BinaryOperator::Subtract);
        Value compressedStart = binary(
            nested, nestedLocation, nested.getIndexType(),
            compressedRange.getLogicalStart(),
            binary(nested, nestedLocation, nested.getIndexType(), denseOffset,
                   twoValue, BinaryOperator::FloorDivide),
            BinaryOperator::Add);
        Value metadataStart = binary(
            nested, nestedLocation, nested.getIndexType(),
            metadataRange.getLogicalStart(),
            binary(nested, nestedLocation, nested.getIndexType(), denseOffset,
                   groupValue, BinaryOperator::FloorDivide),
            BinaryOperator::Add);
        Value compressedK = nested.create<MakeRangeOp>(
            nestedLocation, compressedIndexType, compressedStart,
            compressedExtent, one, compressedRange.getLogicalStart(),
            compressedRange.getLogicalStop(), compressedMap->getSourceId(),
            compressedMap->getSourceAxis(), compressedMap->getDerived());
        inheritRangeAuthority(compressedK, compressedRange);
        Value denseK = nested.create<MakeRangeOp>(
            nestedLocation, denseIndexType, denseStart, blockKValue, one,
            denseRange.getLogicalStart(), denseRange.getLogicalStop(),
            denseMap->getSourceId(), denseMap->getSourceAxis(),
            denseMap->getDerived());
        inheritRangeAuthority(denseK, denseRange);
        Value metadataK = nested.create<MakeRangeOp>(
            nestedLocation, metadataIndexType, metadataStart, metadataExtent,
            one, metadataRange.getLogicalStart(), metadataRange.getLogicalStop(),
            metadataMap->getSourceId(), metadataMap->getSourceAxis(),
            metadataMap->getDerived());
        inheritRangeAuthority(metadataK, metadataRange);

        FailureOr<Value> compressedTail = buildRangeTailPredicate(
            nested, nestedLocation, compressedK, compressedRange);
        FailureOr<Value> denseTail = buildRangeTailPredicate(
            nested, nestedLocation, denseK, denseRange);
        FailureOr<Value> metadataTail = buildRangeTailPredicate(
            nested, nestedLocation, metadataK, metadataRange);
        if (failed(compressedTail) || failed(denseTail) ||
            failed(metadataTail)) {
          bodyFailed = true;
          return;
        }

        IRMapping compressedReplay;
        for (MakeRangeOp range : compressedRanges)
          compressedReplay.map(range.getResult(), compressedK);
        IRMapping denseReplay;
        for (MakeRangeOp range : denseRanges)
          denseReplay.map(range.getResult(), denseK);
        IRMapping metadataReplay;
        for (MakeRangeOp range : metadataRanges)
          metadataReplay.map(range.getResult(), metadataK);
        FailureOr<Value> compressed = materializeSourceRanges(
            nested, nestedLocation, contract.getCompressed(), unitCompressedK,
            compressedRanges, compressedK, compressedReplay,
            contract.getOperation());
        FailureOr<Value> rhs = materializeSourceRanges(
            nested, nestedLocation, contract.getRhs(), unitDenseK, denseRanges,
            denseK, denseReplay, contract.getOperation());
        SmallVector<Value> replayedMetadata;
        for (Value component : metadataComponents) {
          FailureOr<Value> replayed = materializeSourceRanges(
              nested, nestedLocation, component, unitMetadataK, metadataRanges,
              metadataK, metadataReplay, contract.getOperation());
          if (failed(replayed)) {
            bodyFailed = true;
            return;
          }
          replayedMetadata.push_back(*replayed);
        }
        if (failed(compressed) || failed(rhs)) {
          bodyFailed = true;
          return;
        }
        if (failed(appendTailValidity(nestedLocation, contract.getCompressed(),
                                      compressedRanges, *compressedTail,
                                      compressedReplay)) ||
            failed(appendTailValidity(nestedLocation, contract.getRhs(),
                                      denseRanges, *denseTail, denseReplay)) ||
            failed(appendTailValidity(nestedLocation, contract.getMetadata(),
                                      metadataRanges, *metadataTail,
                                      metadataReplay))) {
          bodyFailed = true;
          return;
        }
        Value metadata = replayedMetadata.front();
        if (metadataRecordType) {
          SmallVector<Attribute> fieldTypes;
          for (Value component : replayedMetadata)
            fieldTypes.push_back(TypeAttr::get(component.getType()));
          auto blockedRecordType = RecordType::get(
              context, metadataRecordType.getFieldNames(),
              ArrayAttr::get(context, fieldTypes), metadataRecordType.getOwner());
          metadata = nested.create<MakeRecordOp>(nestedLocation,
                                                 blockedRecordType,
                                                 replayedMetadata);
        }
        auto product = nested.create<SparseContractOp>(
            nestedLocation, resultType, *compressed, metadata, *rhs,
            contract.getLogicalExtent(), carries.front(), contract.getFormat(),
            contract.getLhsReductionAxes(), contract.getRhsReductionAxes(),
            contract.getLhsBatchAxes(), contract.getRhsBatchAxes());
        if (Attribute origin = contract->getAttr(originAttr))
          product->setAttr(originAttr, origin);
        nested.create<scf::YieldOp>(nestedLocation, product.getResult());
      });
  if (bodyFailed) {
    loop.erase();
    return contract.emitOpError(
        "shared sparse blocking could not replay its compressed/metadata/dense graphs");
  }
  contract.getResult().replaceAllUsesWith(loop.getResult(0));
  contract.erase();
  return success();
}

} // namespace intent::gpu::contraction
