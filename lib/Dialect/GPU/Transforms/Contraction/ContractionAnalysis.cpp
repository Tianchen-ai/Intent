#include "ContractionDetail.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"

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

static bool isOwnershipExtent(Operation *origin, Attribute attribute);

static bool hasUnrealizedPhysicalAxis(Value value);

static LogicalResult collectReductionAxisRanges(
    Value value, unsigned axis, SmallVectorImpl<MakeRangeOp> &ranges);

static bool isTransparentMatrixReshape(ReshapeOp reshape);

static Value stripTransparentMatrixReshapes(Value value);

bool operandsNeedPhysicalRealization(Operation *operation, ValueRange fragments,
                                     Value lhs, ArrayRef<int64_t> lhsReduction,
                                     Value rhs, ArrayRef<int64_t> rhsReduction,
                                     ValueRange replayedOperands) {
  for (Value value : fragments) {
    auto type = cast<FragmentType>(value.getType());
    if (llvm::any_of(type.getShape(), [&](Attribute extent) {
          return !isCompileTimeExtent(cast<PhysicalExprAttr>(extent)) ||
                 isFullCoverageExtent(operation, extent);
        }))
      return true;
  }
  return reductionUsesOwnershipExtent(operation, lhs, lhsReduction) ||
         reductionUsesOwnershipExtent(operation, rhs, rhsReduction) ||
         llvm::any_of(replayedOperands, hasUnrealizedPhysicalAxis);
}

ProgramSegment queryContractionProgramSegment(func::FuncOp kernel,
                                              Operation *operation,
                                              ProgramMappingScope scope) {
  auto mapping = operation->getParentOfType<ExecutionGroupOp>();
  if (mapping &&
      (!mapping.getLinear().getDefiningOp<ProgramIdOp>() ||
       (scope == ProgramMappingScope::SameBlock &&
        operation->getBlock() != &mapping.getBody().front())))
    mapping = {};
  return {mapping, kernel->getAttrOfType<ArrayAttr>(programSpaceAttr),
          mapping ? mapping.getSegmentOffset()
                  : PhysicalExprAttr(),
          mapping ? mapping.getSegmentLength()
                  : PhysicalExprAttr()};
}

bool isCompileTimeExtent(PhysicalExprAttr expression) {
  auto kind = expression.getKind();
  if (kind == PhysicalExprKind::Constant || kind == PhysicalExprKind::Parameter)
    return true;
  if (kind == PhysicalExprKind::Dimension ||
      kind == PhysicalExprKind::ScalarABI)
    return false;
  return llvm::all_of(expression.getOperands(), [](Attribute operand) {
    return isCompileTimeExtent(cast<PhysicalExprAttr>(operand));
  });
}

bool isFullCoverageExtent(Operation *origin, Attribute attribute) {
  if (!origin)
    return false;
  auto kernel = origin->getParentOfType<func::FuncOp>();
  auto extent = dyn_cast<PhysicalExprAttr>(attribute);
  if (!kernel || !extent)
    return false;
  auto parameter = parameterForExtent(kernel, extent);
  return succeeded(parameter) && parameter->isDeferred();
}

static bool isOwnershipExtent(Operation *origin, Attribute attribute) {
  if (!origin)
    return false;
  auto kernel = origin->getParentOfType<func::FuncOp>();
  auto extent = dyn_cast<PhysicalExprAttr>(attribute);
  if (!kernel || !extent)
    return false;
  auto parameter = parameterForExtent(kernel, extent);
  if (failed(parameter))
    return false;
  auto role = parameter->getRole();
  return role == ParameterRole::OwnershipM || role == ParameterRole::OwnershipN;
}

bool reductionUsesOwnershipExtent(Operation *origin, Value operand,
                                  ArrayRef<int64_t> reductionAxes) {
  auto fragment = dyn_cast<FragmentType>(operand.getType());
  if (!fragment)
    return false;
  return llvm::any_of(reductionAxes, [&](int64_t axis) {
    return axis >= 0 && axis < static_cast<int64_t>(fragment.getShape().size()) &&
           isOwnershipExtent(origin, fragment.getShape()[axis]);
  });
}

bool hasFragmentSchema(ContractOp contract) {
  return isa<FragmentType>(contract.getLhs().getType()) &&
         isa<FragmentType>(contract.getRhs().getType()) &&
         isa<FragmentType>(contract.getResult().getType());
}

bool hasFragmentSchema(ScaledContractOp contract) {
  return isa<FragmentType>(contract.getLhs().getType()) &&
         isa<FragmentType>(contract.getLhsScale().getType()) &&
         isa<FragmentType>(contract.getRhs().getType()) &&
         isa<FragmentType>(contract.getRhsScale().getType()) &&
         isa<FragmentType>(contract.getAccumulator().getType()) &&
         isa<FragmentType>(contract.getResult().getType());
}

bool hasFragmentSchema(SparseContractOp contract) {
  if (!isa<FragmentType>(contract.getCompressed().getType()) ||
      !isa<FragmentType>(contract.getRhs().getType()) ||
      !isa<FragmentType>(contract.getAccumulator().getType()) ||
      !isa<FragmentType>(contract.getResult().getType()))
    return false;
  Type metadata = contract.getMetadata().getType();
  if (isa<FragmentType>(metadata))
    return true;
  auto record = dyn_cast<RecordType>(metadata);
  return record && llvm::all_of(record.getFieldTypes(), [](Attribute field) {
           return isa<FragmentType>(cast<TypeAttr>(field).getValue());
         });
}

static bool hasUnrealizedPhysicalAxis(Value value) {
  auto fragment = dyn_cast<FragmentType>(value.getType());
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  if (!fragment || !kernel)
    return false;
  PhysicalProgramAnalysis analysis(kernel);
  for (unsigned axis = 0; axis < fragment.getShape().size(); ++axis) {
    PhysicalAxisRealizationFact fact = analysis.axisRealization(value, axis);
    if (fact.constructionScalarSeed ||
        (fact.isExact() && !fact.physicalized))
      return true;
  }
  return false;
}

bool requiresPhysicalRealization(ContractOp contract) {
  return operandsNeedPhysicalRealization(
             contract, ValueRange{contract.getLhs(), contract.getRhs(),
                                  contract.getResult()},
             contract.getLhs(), contract.getLhsReductionAxes(),
             contract.getRhs(), contract.getRhsReductionAxes(),
             ValueRange{contract.getLhs(), contract.getRhs()}) ||
         fullReductionNeedsTraversal(contract);
}

bool requiresPhysicalRealization(ScaledContractOp contract) {
  return operandsNeedPhysicalRealization(
      contract, ValueRange{contract.getLhs(), contract.getLhsScale(),
                           contract.getRhs(), contract.getRhsScale(),
                           contract.getAccumulator(), contract.getResult()},
      contract.getLhs(), contract.getLhsReductionAxes(),
      contract.getRhs(), contract.getRhsReductionAxes(),
      ValueRange{contract.getLhs(), contract.getLhsScale(),
                 contract.getRhs(), contract.getRhsScale()});
}

MakeRangeOp sourceRange(Value value) {
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  if (!kernel)
    return {};
  PhysicalProgramAnalysis analysis(kernel);
  PhysicalRangeFact fact = analysis.sourceRanges(value);
  // Read-dependent coordinates use indirect-row replay, not direct range blocking.
  if (!fact.accesses.empty())
    return {};
  FailureOr<MakeRangeOp> range = queryExactLogicalRange(fact);
  return succeeded(range) ? *range : MakeRangeOp();
}

bool containsSource(Value value, PhysicalSourceAxis source) {
  return queryFragmentAxis(value.getType(), source).isExact();
}

FailureOr<unsigned> mappingAxisForScalar(Value value, ExecutionGroupOp mapping) {
  auto roles =
      mapping.getCoordinateRolesAttr();
  if (!roles || roles.size() != mapping.getCoordinates().size())
    return failure();
  const int64_t pointwiseOwnership =
      static_cast<int64_t>(CoordinateRole::PointwiseOwnership);
  llvm::SmallDenseSet<unsigned, 2> axes;
  bool otherCoordinate = false;
  llvm::SmallPtrSet<Operation *, 16> visited;
  std::function<void(Value)> collect = [&](Value current) {
    for (auto [axis, coordinate] : llvm::enumerate(mapping.getCoordinates()))
      if (current == coordinate) {
        if (roles[axis] == pointwiseOwnership)
          axes.insert(axis);
        else
          otherCoordinate = true;
        return;
      }
    Operation *producer = current.getDefiningOp();
    if (!producer || producer->getNumRegions() != 0 ||
        !visited.insert(producer).second)
      return;
    for (Value operand : producer->getOperands())
      collect(operand);
  };
  collect(value);
  return !otherCoordinate && axes.size() == 1
             ? FailureOr<unsigned>(*axes.begin())
             : FailureOr<unsigned>(failure());
}

static LogicalResult collectReductionAxisRanges(
    Value value, unsigned axis, SmallVectorImpl<MakeRangeOp> &ranges) {
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  auto type = dyn_cast<FragmentType>(value.getType());
  if (!kernel || !type || axis >= type.getShape().size())
    return failure();
  PhysicalProgramAnalysis analysis(kernel);
  PhysicalRangeFact fact = analysis.axisRanges(value, axis);
  FailureOr<MakeRangeOp> authority = queryExactLogicalRange(fact);
  if (failed(authority) || !fact.unitStep ||
      !analysis.lockstepRanges(fact.roots).isExact())
    return failure();
  PhysicalRangeAxisFact selected = analysis.rangeAxes(value, fact.roots);
  if (!selected.isExact() ||
      selected.fragmentAxes != ArrayRef<unsigned>{axis} ||
      llvm::any_of(fact.roots, [&](MakeRangeOp range) {
        return !sameLogicalRange(range, *authority) ||
               range.getResult().getType().getShape()[0] !=
                   type.getShape()[axis];
      }))
    return failure();
  ranges.assign(fact.roots.begin(), fact.roots.end());
  return success();
}

LogicalResult collectPairedReductionRanges(
    ContractOp contract, SmallVectorImpl<MakeRangeOp> &lhsRanges,
    SmallVectorImpl<MakeRangeOp> &rhsRanges) {
  if (contract.getLhsReductionAxes().size() != 1 ||
      contract.getRhsReductionAxes().size() != 1)
    return failure();
  auto lhsType = dyn_cast<FragmentType>(contract.getLhs().getType());
  auto rhsType = dyn_cast<FragmentType>(contract.getRhs().getType());
  if (!lhsType || !rhsType)
    return failure();
  if (failed(collectReductionAxisRanges(
          contract.getLhs(), contract.getLhsReductionAxes().front(), lhsRanges)) ||
      failed(collectReductionAxisRanges(
          contract.getRhs(), contract.getRhsReductionAxes().front(), rhsRanges)))
    return failure();
  MakeRangeOp lhs = lhsRanges.front(), rhs = rhsRanges.front();
  if (samePhysicalScalarExpression(lhs.getLogicalStart(), rhs.getLogicalStart()) &&
      samePhysicalScalarExpression(lhs.getLogicalStop(), rhs.getLogicalStop()))
    return success();
  auto lhsSize = constantLogicalRangeCardinality(lhs);
  auto rhsSize = constantLogicalRangeCardinality(rhs);
  return success(lhsSize && rhsSize && *lhsSize == *rhsSize);
}

bool hasExplicitPairedReductionRanges(ContractOp contract) {
  SmallVector<MakeRangeOp> lhsRanges, rhsRanges;
  return succeeded(collectPairedReductionRanges(contract, lhsRanges, rhsRanges));
}

FailureOr<MakeRangeOp> producerRange(Value value, PhysicalSourceAxis source) {
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  if (!kernel)
    return failure();
  PhysicalRangeFact fact =
      PhysicalProgramAnalysis(kernel).sourceRanges(value, source);
  return queryExactLogicalRange(fact);
}

bool isIntegerConstant(Value value, int64_t expected) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  auto integer = constant ? dyn_cast<IntegerAttr>(constant.getValue())
                          : IntegerAttr();
  return integer && integer.getInt() == expected;
}

bool isTailPredicate(Value value,
                     ArrayRef<std::pair<MakeRangeOp, Value>> ranges) {
  if (!value)
    return true;
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  return kernel && PhysicalProgramAnalysis(kernel).isTailPredicate(value, ranges);
}

FailureOr<ParameterAttr> parameterForExtent(func::FuncOp kernel,
                                          PhysicalExprAttr extent) {
  if (!extent || extent.getKind() !=
                     PhysicalExprKind::Parameter)
    return failure();
  return queryParameterBySymbol(kernel, extent.getParameterReference().getName());
}

static bool isTransparentMatrixReshape(ReshapeOp reshape) {
  auto source = dyn_cast<FragmentType>(reshape.getValue().getType());
  auto target = dyn_cast<FragmentType>(reshape.getResult().getType());
  if (!source || !target || source.getElementType() != target.getElementType())
    return false;
  auto projectedAxes = [](FragmentType type) {
    SmallVector<std::pair<PhysicalSourceAxis, Attribute>> axes;
    for (auto [extent, mapping] :
         llvm::zip(type.getShape(), type.getAxisMaps())) {
      auto axis = cast<AxisMapAttr>(mapping);
      auto expression = cast<PhysicalExprAttr>(extent);
      bool introducedUnit =
          axis.getDerived() &&
          expression.getKind() ==
              PhysicalExprKind::Constant &&
          expression.getValue() == 1;
      if (!introducedUnit)
        axes.emplace_back(sourceAxisIdentity(axis), extent);
    }
    return axes;
  };
  return projectedAxes(source) == projectedAxes(target);
}

static Value stripTransparentMatrixReshapes(Value value) {
  while (auto reshape = value.getDefiningOp<ReshapeOp>()) {
    if (!isTransparentMatrixReshape(reshape))
      break;
    value = reshape.getValue();
  }
  return value;
}

LoadOp matrixOperandLoad(Value value) {
  value = stripTransparentMatrixReshapes(value);
  if (auto load = value.getDefiningOp<LoadOp>())
    return load;
  auto transpose = value.getDefiningOp<TransposeOp>();
  if (!transpose)
    return {};
  ArrayRef<int64_t> permutation = transpose.getPermutation();
  if (permutation.size() < 2)
    return {};
  for (unsigned axis = 0; axis + 2 < permutation.size(); ++axis)
    if (permutation[axis] != static_cast<int64_t>(axis))
      return {};
  unsigned penultimate = permutation.size() - 2;
  unsigned last = permutation.size() - 1;
  if (permutation[penultimate] != static_cast<int64_t>(last) ||
      permutation[last] != static_cast<int64_t>(penultimate))
    return {};
  return stripTransparentMatrixReshapes(transpose.getValue())
      .getDefiningOp<LoadOp>();
}

FailureOr<unsigned> accessCoordinatePosition(LoadOp load,
                                             AxisMapAttr mapping,
                                             Value operand) {
  return PhysicalProgramAnalysis(load->getParentOfType<func::FuncOp>())
      .accessCoordinatePosition(load, mapping, operand);
}

FailureOr<unsigned> directRankOneAccessPosition(ValueRange coordinates,
                                                Type valueType,
                                                unsigned valueAxis) {
  auto fragment = dyn_cast<FragmentType>(valueType);
  if (!fragment || fragment.getShape().size() != coordinates.size() ||
      valueAxis >= coordinates.size() ||
      !llvm::all_of(coordinates, [](Value coordinate) {
        auto type = dyn_cast<FragmentType>(coordinate.getType());
        return type && type.getShape().size() == 1;
      }))
    return failure();
  return valueAxis;
}

bool freeAxesNeedRealization(ContractOp contract, func::FuncOp kernel) {
  return PhysicalProgramAnalysis(kernel)
      .contractFreeAxes(contract.getOperation())
      .needsRealization();
}

bool freeAxesReadyForReductionTraversal(ContractOp contract,
                                        func::FuncOp kernel) {
  PhysicalContractFreeAxisFact freeAxes =
      PhysicalProgramAnalysis(kernel).contractFreeAxes(contract.getOperation());
  if (!freeAxes.isExact() || freeAxes.axes.empty())
    return false;
  return llvm::all_of(freeAxes.axes, [&](const PhysicalContractFreeAxis &axis) {
    if (axis.realization.physicalized)
      return true;
    auto fragment = cast<FragmentType>(axis.operand.getType());
    auto extent = cast<PhysicalExprAttr>(
        fragment.getShape()[axis.operandAxis]);
    if (extent.getKind() ==
        PhysicalExprKind::Parameter) {
      FailureOr<ParameterAttr> parameter =
          queryParameterBySymbol(kernel, extent.getParameterReference().getName());
      if (failed(parameter))
        return false;
      auto role =
          parameter->getRole();
      return role == ParameterRole::OwnershipM ||
             role == ParameterRole::OwnershipN ||
             role == ParameterRole::FullCoverage;
    }
    return extent.getKind() ==
               PhysicalExprKind::Constant &&
           extent.getValue() == 1 &&
           axis.realization.isExact() &&
           !axis.realization.constructionScalarSeed &&
           axis.realization.roots.empty();
  });
}

bool reductionAxesNeedTraversal(ContractOp contract, func::FuncOp kernel) {
  PhysicalProgramAnalysis analysis(kernel);
  auto operandNeedsTraversal = [&](Value operand, ArrayRef<int64_t> axes) {
    auto fragment = cast<FragmentType>(operand.getType());
    return llvm::any_of(axes, [&](int64_t axis) {
      if (axis < 0 || axis >= static_cast<int64_t>(fragment.getShape().size()))
        return false;
      PhysicalAxisRealizationFact fact =
          analysis.axisRealization(operand, static_cast<unsigned>(axis));
      return isOwnershipExtent(contract, fragment.getShape()[axis]) ||
             isFullCoverageExtent(contract, fragment.getShape()[axis]) ||
             fact.constructionScalarSeed ||
             (fact.isExact() && !fact.physicalized);
    });
  };
  return operandNeedsTraversal(contract.getLhs(),
                               contract.getLhsReductionAxes()) ||
         operandNeedsTraversal(contract.getRhs(),
                               contract.getRhsReductionAxes()) ||
         fullReductionNeedsTraversal(contract);
}

bool hasCompleteEpilogue(ContractOp contract) {
  return collectContractionEpilogue(contract.getResult()).has_value();
}

bool outputCoordinatesNeedRealization(ContractOp contract) {
  if (contract.getResult().getType().getShape().size() != 2 ||
      !contract.getLhsBatchAxes().empty() ||
      !contract.getRhsBatchAxes().empty())
    return false;
  auto epilogue = collectContractionEpilogue(contract.getResult());
  if (!epilogue)
    return false;
  auto kernel = contract->getParentOfType<func::FuncOp>();
  PhysicalProgramAnalysis analysis(kernel);
  return llvm::any_of(epilogue->stores, [&](StoreOp store) {
    if (store.getCoordinates().size() != 2 ||
        store.getSourceAxes() != ArrayRef<int64_t>{0, 1})
      return false;
    MakeRangeOp outputRanges[2], sourceRanges[2];
    for (unsigned axis = 0; axis < 2; ++axis) {
      auto output = sourceRange(store.getCoordinates()[axis]);
      auto source = queryExactLogicalRange(
          analysis.axisRanges(contract.getResult(), axis));
      if (!output || failed(source))
        continue;
      if (samePhysicalScalarExpression(output.getLogicalStart(), source->getLogicalStart()) &&
          samePhysicalScalarExpression(output.getLogicalStop(), source->getLogicalStop()) &&
          samePhysicalScalarExpression(output.getStep(), source->getStep())) {
        outputRanges[axis] = output;
        sourceRanges[axis] = *source;
        if (output.getResult().getType().getShape() ==
                source->getResult().getType().getShape() &&
            !samePhysicalScalarExpression(output.getStart(), source->getStart()))
          return true;
      }
    }
    // Reusing an operand in two matrix roles does not equate the independent
    // output coordinates. Realize their M/N ownership before a K-only loop.
    return sourceRanges[0] && sourceRanges[1] &&
           sameLogicalRange(sourceRanges[0], sourceRanges[1]) &&
           outputRanges[0] && outputRanges[1] &&
           outputRanges[0] != outputRanges[1];
  });
}

scf::ForOp enclosingRegionContractionSegment(Operation *operation) {
  for (Operation *parent = operation->getParentOp(); parent;
       parent = parent->getParentOp()) {
    auto loop = dyn_cast<scf::ForOp>(parent);
    if (!loop)
      continue;
    auto segment = queryParameter(loop.getStep());
    if (!segment ||
        segment.getRole() !=
            ParameterRole::ScanChunk ||
        segment.getCategory() !=
            ParameterCategory::RegionContraction)
      continue;
    return loop;
  }
  return {};
}

FailureOr<ParameterAttr>
regionContractionParameter(func::FuncOp kernel, PhysicalExprAttr extent) {
  if (!extent ||
      extent.getKind() !=
          PhysicalExprKind::Parameter)
    return failure();
  FailureOr<ParameterAttr> parameter =
      queryParameterBySymbol(kernel, extent.getParameterReference().getName());
  if (failed(parameter) ||
      (*parameter).getRole() !=
          ParameterRole::ScanChunk ||
      (*parameter).getCategory() !=
          ParameterCategory::RegionContraction)
    return failure();
  return *parameter;
}

bool canReplayContractionReads(ContractOp contract) {
  SmallVector<Value> pending(contract->getOperands());
  llvm::DenseSet<Value> seen;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    if (!seen.insert(value).second)
      continue;
    Operation *producer = value.getDefiningOp();
    if (!producer)
      continue;
    if (auto load = dyn_cast<LoadOp>(producer);
        load && !canReplayReadAt(load, contract))
      return false;
    llvm::append_range(pending, producer->getOperands());
  }
  return true;
}

bool fullReductionNeedsTraversal(ContractOp contract) {
  if (contract.getLhsReductionAxes().size() != 1 ||
      contract.getRhsReductionAxes().size() != 1)
    return false;
  auto kernel = contract->getParentOfType<func::FuncOp>();
  PhysicalProgramAnalysis analysis(kernel);
  int64_t smallestTile = *std::min_element(
      std::begin(contractionReductionCandidates),
      std::end(contractionReductionCandidates));
  SmallVector<MakeRangeOp> paired;
  for (auto [operand, axis] :
       {std::pair<Value, int64_t>{contract.getLhs(),
                                  contract.getLhsReductionAxes().front()},
        std::pair<Value, int64_t>{contract.getRhs(),
                                  contract.getRhsReductionAxes().front()}}) {
    auto type = cast<FragmentType>(operand.getType());
    auto physical = cast<PhysicalExprAttr>(type.getShape()[axis]);
    auto extent = constantPhysicalExpression(physical);
    if (extent && *extent <= smallestTile && *extent > 0 &&
        llvm::isPowerOf2_64(*extent))
      return false;
    PhysicalRangeFact ranges = analysis.axisRanges(operand, axis);
    if (!ranges.unitStep || failed(queryExactLogicalRange(ranges)))
      return false;
    for (MakeRangeOp range : ranges.roots) {
      auto size = constantLogicalRangeCardinality(range);
      bool complete = size && extent && *size > 0 && *size <= *extent;
      complete |= isZeroScalar(range.getLogicalStart()) &&
                  samePhysicalScalarExpression(range.getExtent(),
                                               range.getLogicalStop());
      if (!complete || queryLaunchExpression(range.getExtent()) != physical ||
          !samePhysicalScalarExpression(range.getStart(),
                                        range.getLogicalStart()))
        return false;
      paired.push_back(range);
    }
  }
  // A complete logical extent does not select a hardware reduction tile.
  // Only retile a complete range here; an existing segment retains its bounds.
  return analysis.lockstepRanges(paired).isExact() &&
         canReplayContractionReads(contract);
}

bool hasRangeContractForm(ContractOp contract,
                         SmallVectorImpl<StoreOp> *stores) {
  auto schema = queryContractionAxes(contract);
  auto lhsLoad = matrixOperandLoad(contract.getLhs());
  auto rhsLoad = matrixOperandLoad(contract.getRhs());
  if (!schema || !lhsLoad || !rhsLoad || schema->reduction.size() != 1 ||
      !schema->batch.empty() || schema->lhsFree.size() != 1 ||
      schema->rhsFree.size() != 1)
    return false;
  SmallVector<std::tuple<LoadOp, AxisMapAttr, Value>> axes;
  for (auto [load, operand, axis] :
       {std::tuple<LoadOp, Value, unsigned>{lhsLoad, contract.getLhs(),
                                          schema->lhsFree.front()},
        std::tuple<LoadOp, Value, unsigned>{
            lhsLoad, contract.getLhs(),
            static_cast<unsigned>(contract.getLhsReductionAxes().front())},
        std::tuple<LoadOp, Value, unsigned>{
            rhsLoad, contract.getRhs(),
            static_cast<unsigned>(contract.getRhsReductionAxes().front())},
        std::tuple<LoadOp, Value, unsigned>{rhsLoad, contract.getRhs(),
                                          schema->rhsFree.front()}}) {
    FailureOr<AxisMapAttr> mapping = queryAxisMap(operand.getType(), axis);
    if (failed(mapping))
      return false;
    axes.emplace_back(load, *mapping, operand);
  }
  for (auto [load, mapping, operand] : axes) {
    FailureOr<unsigned> coordinate = accessCoordinatePosition(load, mapping, operand);
    if (failed(coordinate))
      return false;
    Value value = load.getCoordinates()[*coordinate];
    if (!sourceRange(value) &&
        failed(producerRange(
            value, sourceAxisIdentity(mapping))))
      return false;
  }
  auto epilogue = collectContractionEpilogue(contract.getResult());
  if (!epilogue)
    return false;
  if (stores)
    llvm::append_range(*stores, epilogue->stores);
  return true;
}

} // namespace intent::gpu::contraction
