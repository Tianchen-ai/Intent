#include "ContractionDetail.h"
#include "../Value/SourceReplay.h"
#include "../Value/ScopePlacement.h"
#include "Intent/Dialect/GPU/Transforms/Mapping/ExecutionGroups.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Transforms/Control/Traversal.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/Analysis/MemoryEffects.h"
#include "Intent/Dialect/GPU/Analysis/ResourceAlias.h"

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/AccessOpInterface.h"
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



// Track products at the creation site. Retiling an epilogue must not rescan
// every existing contraction in the kernel to discover its new value graph.
class ContractionCreationListener : public OpBuilder::Listener {
public:
  void notifyOperationInserted(Operation *operation,
                               OpBuilder::InsertPoint) override {
    operation->walk([&](ContractOp product) {
      if (observed.insert(product.getOperation()).second)
        products.push_back(product);
    });
  }

  SmallVector<ContractOp> products;

private:
  llvm::SmallPtrSet<Operation *, 16> observed;
};

LogicalResult realizeContract(ContractOp contract, func::FuncOp kernel,
                              SmallVectorImpl<ContractOp> &pending) {
  if (!contract->getBlock())
    return success();
  ContractionCreationListener created;
  const bool required = requiresPhysicalRealization(contract) ||
                        outputCoordinatesNeedRealization(contract);
  auto unhandled = [&](const Twine &reason) -> LogicalResult {
    if (!required)
      return success();
    return contract.emitOpError()
           << "cannot form a complete physical contraction: " << reason
           << "; lhs=" << contract.getLhs().getType()
           << "; rhs=" << contract.getRhs().getType()
           << "; result=" << contract.getResult().getType()
           << "; lhs_reduction=" << contract.getLhsReductionAxes()
           << "; rhs_reduction=" << contract.getRhsReductionAxes()
           << "; lhs_batch=" << contract.getLhsBatchAxes()
           << "; rhs_batch=" << contract.getRhsBatchAxes()
           << "; free_axes_need_realization="
           << freeAxesNeedRealization(contract, kernel)
           << "; selected_free_axes="
           << freeAxesReadyForReductionTraversal(contract, kernel);
  };
  std::string reason;
  auto axes = queryContractionAxes(contract, &reason);
  if (!axes)
    return contract.emitOpError("invalid contraction blocking axis schema: ") << reason;
  if (axes->reduction.size() != 1 || !axes->batch.empty())
    return unhandled("requires one reduction pair and no batch axes");

  auto lhsType = contract.getLhs().getType();
  auto rhsType = contract.getRhs().getType();
  if (axes->lhsFree.size() != 1 || axes->rhsFree.size() != 1)
    return unhandled("each operand must have one physical free axis");
  unsigned lhsFree = axes->lhsFree.front();
  unsigned rhsFree = axes->rhsFree.front();
  unsigned lhsReduction = contract.getLhsReductionAxes().front();
  unsigned rhsReduction = contract.getRhsReductionAxes().front();
  FailureOr<AxisMapAttr> rowMap = queryAxisMap(lhsType, lhsFree);
  FailureOr<AxisMapAttr> lhsReductionMap = queryAxisMap(lhsType, lhsReduction);
  FailureOr<AxisMapAttr> rhsReductionMap = queryAxisMap(rhsType, rhsReduction);
  FailureOr<AxisMapAttr> columnMap = queryAxisMap(rhsType, rhsFree);
  if (failed(rowMap) || failed(lhsReductionMap) || failed(rhsReductionMap) ||
      failed(columnMap) ||
      lhsReductionMap->getDimensionId() <= 0 ||
      rhsReductionMap->getDimensionId() <= 0)
    return unhandled("paired reduction axes have no logical dimensions");

  auto matrixAccess = [&](Value operand, AxisMapAttr free,
                          AxisMapAttr reduction) -> LoadOp {
    auto replay = PhysicalProgramAnalysis(kernel).replayability(
        operand, std::nullopt, PhysicalReplayScope::ValueGraph,
        /*allowAccesses=*/true);
    if (!replay.isReplayable())
      return {};
    for (Operation *access : replay.accesses) {
      auto load = dyn_cast<LoadOp>(access);
      if (!load)
        continue;
      auto freeCoordinate = accessCoordinatePosition(load, free, operand);
      auto reductionCoordinate = accessCoordinatePosition(load, reduction, operand);
      if (succeeded(freeCoordinate) && succeeded(reductionCoordinate) &&
          *freeCoordinate != *reductionCoordinate)
        return load;
    }
    return {};
  };
  auto lhsLoad = matrixAccess(contract.getLhs(), *rowMap, *lhsReductionMap);
  auto rhsLoad = matrixAccess(contract.getRhs(), *columnMap, *rhsReductionMap);
  if (!lhsLoad || !rhsLoad || !canReplayContractionReads(contract))
    return unhandled("operands have no replayable matrix coordinate accesses");

  FailureOr<unsigned> lhsRowCoordinate =
      accessCoordinatePosition(lhsLoad, *rowMap, contract.getLhs());
  FailureOr<unsigned> lhsReductionCoordinate =
      accessCoordinatePosition(lhsLoad, *lhsReductionMap, contract.getLhs());
  FailureOr<unsigned> rhsReductionCoordinate =
      accessCoordinatePosition(rhsLoad, *rhsReductionMap, contract.getRhs());
  FailureOr<unsigned> rhsColumnCoordinate =
      accessCoordinatePosition(rhsLoad, *columnMap, contract.getRhs());
  if (failed(lhsRowCoordinate) || failed(lhsReductionCoordinate) ||
      failed(rhsReductionCoordinate) || failed(rhsColumnCoordinate)) {
    std::string details;
    llvm::raw_string_ostream stream(details);
    stream << "load coordinates do not cover all free/reduction axes"
           << "; lhs_row=" << succeeded(lhsRowCoordinate)
           << ", lhs_reduction=" << succeeded(lhsReductionCoordinate)
           << ", rhs_reduction=" << succeeded(rhsReductionCoordinate)
           << ", rhs_column=" << succeeded(rhsColumnCoordinate)
           << ", row_map=" << *rowMap << ", lhs_coordinate_types=[";
    llvm::interleaveComma(lhsLoad.getCoordinates(), stream,
                          [&](Value coordinate) { stream << coordinate.getType(); });
    stream << "]";
    return unhandled(stream.str());
  }
  Value originalLhsRowCoordinate = lhsLoad.getCoordinates()[*lhsRowCoordinate];
  auto rowRange = sourceRange(originalLhsRowCoordinate);
  const bool indirectRow = !rowRange;
  if (!rowRange) {
    FailureOr<MakeRangeOp> discovered =
        producerRange(originalLhsRowCoordinate,
                      sourceAxisIdentity(*rowMap));
    if (succeeded(discovered))
      rowRange = *discovered;
  }
  auto lhsReductionRange =
      sourceRange(lhsLoad.getCoordinates()[*lhsReductionCoordinate]);
  auto rhsReductionRange =
      sourceRange(rhsLoad.getCoordinates()[*rhsReductionCoordinate]);
  auto columnRange = sourceRange(rhsLoad.getCoordinates()[*rhsColumnCoordinate]);
  FailureOr<int64_t> lhsReductionDimension =
      lhsReductionRange ? queryRangeDimension(lhsReductionRange)
                        : FailureOr<int64_t>(failure());
  FailureOr<int64_t> rhsReductionDimension =
      rhsReductionRange ? queryRangeDimension(rhsReductionRange)
                        : FailureOr<int64_t>(failure());
  if (!rowRange || !lhsReductionRange || !rhsReductionRange || !columnRange ||
      failed(lhsReductionDimension) || failed(rhsReductionDimension) ||
      !PhysicalProgramAnalysis(kernel)
           .lockstepRanges({lhsReductionRange, rhsReductionRange}).isExact())
    return unhandled("physical coordinates are not explicit compatible ranges");
  for (auto [operand, axis, range] : {
           std::tuple<Value, unsigned, MakeRangeOp>{contract.getLhs(), lhsFree,
                                                   rowRange},
           {contract.getLhs(), lhsReduction, lhsReductionRange},
           {contract.getRhs(), rhsReduction, rhsReductionRange},
           {contract.getRhs(), rhsFree, columnRange}}) {
    auto selected = PhysicalProgramAnalysis(kernel).rangeAxes(operand, {range});
    if (!selected.isExact() || selected.fragmentAxes != ArrayRef<unsigned>{axis})
      return unhandled("matrix coordinates do not select independent operand axes");
  }
  Value rowLogicalEnd = (rowRange).getLogicalStop();
  Value reductionLogicalEnd = (lhsReductionRange).getLogicalStop();
  Value columnLogicalEnd = (columnRange).getLogicalStop();

  auto rowBound = rowRange.getStart().getDefiningOp<RangeBoundOp>();
  auto rowSourceRange =
      rowBound ? rowBound.getRange().getDefiningOp<RangeOp>() : RangeOp();
  PhysicalExprAttr rowRangeExtent = queryLaunchRangeExtent(rowRange);
  PhysicalExprAttr columnRangeExtent = queryLaunchRangeExtent(columnRange);
  const bool runtimeRowTraversal =
      indirectRow ||
      (!rowRangeExtent &&
       (rowRange->hasAttr(sourceSubregionAttr) ||
        (rowSourceRange && rowSourceRange->hasAttr(sourceSubregionAttr))));
  const bool persistentRowTraversal = runtimeRowTraversal && !indirectRow;
  const ParameterCategory contractionCategory =
      persistentRowTraversal ? ParameterCategory::PersistentContraction
                             : ParameterCategory::Contraction;

  if (!isUnitStepRange(rowRange) ||
      !isUnitStepRange(lhsReductionRange) ||
      !isUnitStepRange(columnRange))
    return unhandled("blocking currently requires unit-step source ranges");

  auto epilogue = collectContractionEpilogue(contract.getResult());
  if (!epilogue)
    return unhandled("result does not have a closed store epilogue");
  SmallVector<SmallVector<Block *>> storeScopes;
  llvm::SmallPtrSet<Operation *, 8> checkedConditions;
  if (runtimeRowTraversal) {
    ResourceAliasAnalysis aliases;
    PhysicalProgramAnalysis analysis(kernel);
    for (auto [index, store] : llvm::enumerate(epilogue->stores)) {
      auto path = placement::conditionalPath(contract->getBlock(), store);
      if (failed(path))
        return unhandled("runtime row output crosses a non-conditional execution scope");
      if (!placement::canMoveEffectBefore(store, contract, aliases))
        return unhandled("runtime row output cannot cross intervening memory effects");
      auto resourceBindings = scalarCaptureBindings(store.getResource(), contract);
      if (failed(resourceBindings) ||
          !analysis.replayAt(store.getResource(), sourceAxisIdentity(*rowMap),
                             PhysicalReplayScope::ValueGraph,
                             /*allowAccesses=*/true, contract, *resourceBindings)
               .isReplayable())
        return unhandled("runtime row output resource is unavailable in its relocated scope");
      for (unsigned earlier = 0; earlier < index; ++earlier) {
        bool exclusive = llvm::any_of(storeScopes[earlier], [&](Block *first) {
          return llvm::any_of(*path, [&](Block *second) {
            return first != second && first->getParentOp() == second->getParentOp();
          });
        });
        if (!exclusive && !placement::independentMemoryEffects(
                              epilogue->stores[earlier], store, aliases))
          return unhandled("runtime row outputs lack independent memory effects");
      }
      for (Block *block : *path) {
        auto branch = cast<scf::IfOp>(block->getParentOp());
        if (checkedConditions.contains(branch))
          continue;
        auto bindings = scalarCaptureBindings(branch.getCondition(), contract);
        if (failed(bindings) ||
            !analysis.replayAt(branch.getCondition(), sourceAxisIdentity(*rowMap),
                               PhysicalReplayScope::ValueGraph,
                               /*allowAccesses=*/true, contract, *bindings)
                 .isReplayable())
          return unhandled("runtime row output condition has no stable scalar replay");
        checkedConditions.insert(branch);
      }
      storeScopes.push_back(std::move(*path));
    }
  }
  SmallVector<std::pair<MakeRangeOp, Value>> outputTailRanges = {
      {rowRange, rowLogicalEnd},
      {columnRange, columnLogicalEnd},
  };

  SmallVector<AssumeInBoundsOp> rowAssumptions;
  if (indirectRow)
    kernel.walk([&](AssumeInBoundsOp assumption) {
      if (!containsSource(
              assumption.getIndex(),
              sourceAxisIdentity(*rowMap)))
        return;
      FailureOr<MakeRangeOp> root =
          producerRange(assumption.getIndex(),
                        sourceAxisIdentity(*rowMap));
      if (succeeded(root) && sameLogicalRange(*root, rowRange))
        rowAssumptions.push_back(assumption);
    });

  ProgramSegment segment = queryContractionProgramSegment(
      kernel, contract, ProgramMappingScope::Dominating);
  ExecutionGroupOp mapping = segment.mapping;
  if (!mapping)
    return unhandled("contract is not dominated by the current program mapping");
  ArrayAttr programSpace = segment.space;
  PhysicalExprAttr segmentOffset = segment.offset;
  PhysicalExprAttr segmentLength = segment.length;
  if (!segment.isComplete()) {
    std::string details;
    llvm::raw_string_ostream stream(details);
    stream << "current rule requires one full-program execution segment"
           << "; program_space=" << programSpace
           << ", segment_offset=" << segmentOffset
           << ", segment_length=" << segmentLength;
    return unhandled(stream.str());
  }

  unsigned rowResourceAxis = lhsLoad.getSourceAxes()[*lhsRowCoordinate];
  unsigned columnResourceAxis = rhsLoad.getSourceAxes()[*rhsColumnCoordinate];
  MLIRContext *context = kernel.getContext();
  Location location = contract.getLoc();
  auto sourceSuffix = [](AxisMapAttr mapping) {
    return ("_" + Twine(mapping.getSourceId()) + "_" +
            Twine(mapping.getSourceAxis()) + "_" +
            Twine(static_cast<unsigned>(mapping.getDerived())))
        .str();
  };
  std::string suffix = sourceSuffix(*rowMap);
  suffix += sourceSuffix(*lhsReductionMap);
  suffix += sourceSuffix(*rhsReductionMap);
  suffix += sourceSuffix(*columnMap);
  suffix +=
      ("_" + Twine(static_cast<unsigned>(contractionCategory)) + "_" +
       Twine(static_cast<unsigned>(indirectRow)))
          .str();
  auto blockMRef = getOrCreatePhysicalParameter(
      kernel, "BLOCK_M" + suffix, ParameterRole::OwnershipM,
      contractionCategory,
      lhsType.getElementType().getIntOrFloatBitWidth(),
      {32, 64, 128, 256});
  auto blockNRef = getOrCreatePhysicalParameter(
      kernel, "BLOCK_N" + suffix, ParameterRole::OwnershipN,
      contractionCategory,
      rhsType.getElementType().getIntOrFloatBitWidth(),
      {16, 32, 64, 128, 256});
  auto blockKRef = getOrCreatePhysicalParameter(
      kernel, "BLOCK_K" + suffix, ParameterRole::Reduction,
      contractionCategory,
      std::max(lhsType.getElementType().getIntOrFloatBitWidth(),
               rhsType.getElementType().getIntOrFloatBitWidth()),
      contractionReductionCandidates);
  if (failed(blockMRef) || failed(blockNRef) || failed(blockKRef))
    return failure();
  ParameterAttr blockM = lookupParameter(kernel, *blockMRef);
  ParameterAttr blockN = lookupParameter(kernel, *blockNRef);
  ParameterAttr blockK = lookupParameter(kernel, *blockKRef);
  FailureOr<unsigned> existingRowAxis =
      mappingAxisForScalar(rowRange.getStart(), mapping);
  FailureOr<unsigned> existingColumnAxis =
      mappingAxisForScalar(columnRange.getStart(), mapping);
  Value reductionStart = lhsReductionRange.getStart();
  if (succeeded(mappingAxisForScalar(reductionStart, mapping)))
    reductionStart = lhsReductionRange.getLogicalStart();
  if (succeeded(existingRowAxis) && succeeded(existingColumnAxis) &&
      *existingRowAxis == *existingColumnAxis)
    return unhandled(
        "row and column ownership share one mapping coordinate that cannot be refined independently");
  if (failed(refineOwnershipParameter(kernel, rowRange, existingRowAxis,
                                      blockM)) ||
      failed(refineOwnershipParameter(kernel, columnRange, existingColumnAxis,
                                      blockN)))
    return failure();
  for (auto [parameter, range] :
       {std::pair<ParameterAttr, MakeRangeOp>{blockM, rowRange},
        std::pair<ParameterAttr, MakeRangeOp>{blockN, columnRange},
        std::pair<ParameterAttr, MakeRangeOp>{blockK, lhsReductionRange}})
    if (FailureOr<int64_t> dimension = queryRangeDimension(range);
        succeeded(dimension))
      if (failed(updateParameter(kernel, lookupParameter(kernel, parameter.getReference()).withBinding(
              lookupParameter(kernel, parameter.getReference()).getBinding().withDimension(
                  IntegerAttr::get(IntegerType::get(kernel.getContext(), 64), *dimension))))))
        return failure();
  ParameterAttr rowWorkers;
  if (runtimeRowTraversal) {
    SmallVector<int64_t, 5> rowWorkerCandidates = {1, 2, 4, 8};
    if (persistentRowTraversal)
      rowWorkerCandidates.push_back(16);
    auto reference = getOrCreatePhysicalParameter(
        kernel, "ROW_WORKERS" + suffix, ParameterRole::TraversalWorkers,
        contractionCategory,
        lhsType.getElementType().getIntOrFloatBitWidth(),
        rowWorkerCandidates);
    if (failed(reference)) return failure();
    rowWorkers = lookupParameter(kernel, *reference);
  }
  if (runtimeRowTraversal && !rowWorkers)
    return failure();
  ArrayAttr parameterGroup = ArrayAttr::get(
      context,
      {PhysicalSourceAttr::get(context, rowMap->getSourceId(),
                               rowMap->getSourceAxis(), rowMap->getDerived()),
       PhysicalSourceAttr::get(
           context, lhsReductionMap->getSourceId(),
           lhsReductionMap->getSourceAxis(), lhsReductionMap->getDerived()),
       PhysicalSourceAttr::get(
           context, rhsReductionMap->getSourceId(),
           rhsReductionMap->getSourceAxis(), rhsReductionMap->getDerived()),
       PhysicalSourceAttr::get(context, columnMap->getSourceId(),
                               columnMap->getSourceAxis(),
                               columnMap->getDerived()),
       IntegerAttr::get(IntegerType::get(context, 32),
                        static_cast<uint32_t>(contractionCategory)),
       BoolAttr::get(context, indirectRow)});
  for (ParameterAttr parameter : {blockM, blockN, blockK, rowWorkers}) {
    if (!parameter) continue;
    auto current = lookupParameter(kernel, parameter.getReference());
    if (failed(updateParameter(kernel, current.withBinding(
            current.getBinding().withGroup(parameterGroup))))) return failure();
  }
  auto boundedK = boundedTraversalChunk(
      lookupParameter(kernel, *blockKRef), lhsReductionRange);
  if (failed(boundedK)) return failure();
  PhysicalExprAttr unitK = *boundedK;
  auto freeExtent = [&](ParameterRefAttr reference, MakeRangeOp range,
                        FailureOr<unsigned> existing) -> FailureOr<PhysicalExprAttr> {
    // An existing ownership coordinate already uses the range's physical
    // extent in its start. Preserve that same current-IR expression when
    // refining the parameter; only a newly placed axis chooses its capacity.
    if (succeeded(existing)) {
      auto extent = cast<PhysicalExprAttr>(
          range.getResult().getType().getShape()[0]);
      auto launch = cast<PhysicalExprAttr>(
          mapping.getLaunchExtents()[*existing]);
      if (!PhysicalProgramAnalysis(kernel)
               .axisRealization(range.getResult(), 0).constructionScalarSeed &&
          queryLaunchExpression(range.getExtent()) == extent &&
          launch.getKind() == PhysicalExprKind::CeilDiv &&
          launch.getOperands().size() == 2 &&
          launch.getOperands()[1] == extent)
        return extent;
      return parameterExpression(context, reference.getName().getValue());
    }
    return boundedOwnershipExtent(kernel, lookupParameter(kernel, reference),
                                  queryLogicalRangeCapacity(range));
  };
  auto boundedM = freeExtent(*blockMRef, rowRange, existingRowAxis);
  auto boundedN = freeExtent(*blockNRef, columnRange, existingColumnAxis);
  if (failed(boundedM) || failed(boundedN)) return failure();
  PhysicalExprAttr unitM = *boundedM;
  PhysicalExprAttr unitN = *boundedN;
  OpBuilder parameterBuilder(&kernel.front(), kernel.front().begin());
  auto materializeExtent = [&](PhysicalExprAttr extent) -> Value {
    if (extent.getKind() == PhysicalExprKind::Parameter)
      return materializeParameter(parameterBuilder, location,
                                  extent.getParameterReference());
    return parameterBuilder.create<PhysicalExprOp>(
        location, parameterBuilder.getIndexType(), extent);
  };
  Value blockMValue = materializeExtent(unitM);
  Value blockNValue = materializeExtent(unitN);
  Value blockKValue = unitK.getKind() == PhysicalExprKind::Parameter
      ? materializeParameter(parameterBuilder, location, *blockKRef).getResult()
      : parameterBuilder.create<PhysicalExprOp>(
            location, parameterBuilder.getIndexType(), unitK).getResult();
  Value rowWorkersValue = rowWorkers
      ? materializeParameter(parameterBuilder, location, rowWorkers.getReference()).getResult() : Value();
  PhysicalExprAttr unitRowWorkers;
  if (rowWorkers)
    unitRowWorkers = parameterExpression(
        context, rowWorkers.getName().getValue());

  OpBuilder mapBuilder(mapping, &created);
  SmallVector<Value> mappingExtents(mapping.getExtents());
  SmallVector<Attribute> launchExtents(mapping.getLaunchExtents().begin(),
                                       mapping.getLaunchExtents().end());
  auto rowExpression = cast<PhysicalExprAttr>(
      cast<ViewType>(lhsLoad.getResource().getType())
          .getLayout()
          .getExtents()[rowResourceAxis]);
  auto columnExpression = cast<PhysicalExprAttr>(
      cast<ViewType>(rhsLoad.getResource().getType())
          .getLayout()
          .getExtents()[columnResourceAxis]);
  if (rowRangeExtent)
    rowExpression = rowRangeExtent;
  if (columnRangeExtent)
    columnExpression = columnRangeExtent;
  // Runtime coordinate decoding and launch coverage consume the same typed
  // expressions. Do not reconstruct ceil-div with potentially wrapping adds.
  Value rowTiles = mapBuilder.create<PhysicalExprOp>(
      location, mapBuilder.getIndexType(),
      binaryExpression(context, PhysicalExprKind::CeilDiv, rowExpression, unitM));
  Value columnTiles = mapBuilder.create<PhysicalExprOp>(
      location, mapBuilder.getIndexType(),
      binaryExpression(context, PhysicalExprKind::CeilDiv, columnExpression, unitN));
  SmallVector<int64_t> coordinateRoles(mapping.getCoordinates().size(), -1);
  if (auto existing =
          mapping.getCoordinateRolesAttr())
    if (existing.size() == mapping.getCoordinates().size())
      llvm::copy(existing.asArrayRef(), coordinateRoles.begin());
  auto bindMappingAxis = [&](FailureOr<unsigned> existing, Value extent,
                             PhysicalExprAttr launchExtent,
                             CoordinateRole role) {
    unsigned axis;
    if (succeeded(existing)) {
      axis = *existing;
      mappingExtents[axis] = extent;
      launchExtents[axis] = launchExtent;
    } else {
      axis = mappingExtents.size();
      mappingExtents.push_back(extent);
      launchExtents.push_back(launchExtent);
      coordinateRoles.push_back(-1);
    }
    coordinateRoles[axis] = static_cast<int64_t>(role);
    return axis;
  };
  unsigned rowMappingAxis = bindMappingAxis(
      existingRowAxis, runtimeRowTraversal ? rowWorkersValue : rowTiles,
      runtimeRowTraversal
          ? unitRowWorkers
          : binaryExpression(context, PhysicalExprKind::CeilDiv, rowExpression,
                             unitM),
      indirectRow ? CoordinateRole::IndirectTraversal
                  : runtimeRowTraversal ? CoordinateRole::TraversalWorker
                                        : CoordinateRole::ContractionM);
  unsigned columnMappingAxis = bindMappingAxis(
      existingColumnAxis, columnTiles,
      binaryExpression(context, PhysicalExprKind::CeilDiv, columnExpression,
                       unitN),
      CoordinateRole::ContractionN);
  segmentLength = cast<PhysicalExprAttr>(launchExtents.front());
  for (Attribute extent : llvm::drop_begin(launchExtents))
    segmentLength = binaryExpression(context, PhysicalExprKind::Multiply,
                                     segmentLength,
                                     cast<PhysicalExprAttr>(extent));
  auto expandedMapping = rebuildExecutionGroup(
      mapBuilder, mapping, mappingExtents, mapBuilder.getArrayAttr(launchExtents),
      mapBuilder.getDenseI64ArrayAttr(coordinateRoles), segmentLength);
  for (auto [oldCoordinate, newCoordinate] : llvm::zip(
           mapping.getCoordinates(),
           expandedMapping.getCoordinates().take_front(mapping.getCoordinates().size())))
    oldCoordinate.replaceAllUsesWith(newCoordinate);
  Value rowTile = runtimeRowTraversal
                      ? Value()
                      : expandedMapping.getCoordinates()[rowMappingAxis];
  Value rowWorker = runtimeRowTraversal
                        ? expandedMapping.getCoordinates()[rowMappingAxis]
                        : Value();
  Value columnTile = expandedMapping.getCoordinates()[columnMappingAxis];
  mapping.erase();
  kernel->setAttr(programSpaceAttr,
                  ArrayAttr::get(context, {segmentLength}));
  kernel->setAttr(gridRankAttr,
                  IntegerAttr::get(IntegerType::get(context, 64), 1));

  OpBuilder builder(contract, &created);
  Value one = builder.create<arith::ConstantIndexOp>(location, 1);
  Value rowStop = rowLogicalEnd;
  Value columnStart = columnRange.getStart();
  if (failed(existingColumnAxis))
    columnStart = binary(
        builder, location, builder.getIndexType(), columnStart,
        binary(builder, location, builder.getIndexType(), columnTile,
               blockNValue, BinaryOperator::Multiply),
        BinaryOperator::Add);
  Value columnStop = columnLogicalEnd;
  Value reductionStop = reductionLogicalEnd;

  FragmentType rowIndexType = fragmentType(
      context, builder.getIndexType(), {unitM}, {*rowMap}, lhsType.getOwner());
  FragmentType columnIndexType = fragmentType(
      context, builder.getIndexType(), {unitN}, {*columnMap}, rhsType.getOwner());
  FragmentType reductionIndexType = fragmentType(
      context, builder.getIndexType(), {unitK}, {*lhsReductionMap},
      lhsType.getOwner());
  FragmentType rhsReductionIndexType = fragmentType(
      context, builder.getIndexType(), {unitK}, {*rhsReductionMap},
      rhsType.getOwner());
  FragmentType rowPredicateType = fragmentType(
      context, builder.getI1Type(), {unitM}, {*rowMap}, lhsType.getOwner());
  FragmentType columnPredicateType = fragmentType(
      context, builder.getI1Type(), {unitN}, {*columnMap}, rhsType.getOwner());
  FragmentType reductionPredicateType = fragmentType(
      context, builder.getI1Type(), {unitK}, {*lhsReductionMap},
      lhsType.getOwner());
  FragmentType rhsReductionPredicateType = fragmentType(
      context, builder.getI1Type(), {unitK}, {*rhsReductionMap},
      rhsType.getOwner());
  FragmentType blockedLhsType = fragmentType(
      context, lhsType.getElementType(), {unitM, unitK},
      {*rowMap, *lhsReductionMap}, lhsType.getOwner());
  FragmentType blockedRhsType = fragmentType(
      context, rhsType.getElementType(), {unitK, unitN},
      {*rhsReductionMap, *columnMap}, rhsType.getOwner());
  FragmentType blockedResultType = fragmentType(
      context, contract.getResult().getType().getElementType(), {unitM, unitN},
      {*rowMap, *columnMap}, contract.getResult().getType().getOwner());
  FragmentType lhsPredicateType = fragmentType(
      context, builder.getI1Type(), {unitM, unitK},
      {*rowMap, *lhsReductionMap}, lhsType.getOwner());
  FragmentType rhsPredicateType = fragmentType(
      context, builder.getI1Type(), {unitK, unitN},
      {*rhsReductionMap, *columnMap}, rhsType.getOwner());
  FragmentType outputPredicateType = fragmentType(
      context, builder.getI1Type(), {unitM, unitN},
      {*rowMap, *columnMap}, contract.getResult().getType().getOwner());

  Value columns = builder.create<MakeRangeOp>(
      location, columnIndexType, columnStart, blockNValue, one,
      columnRange.getLogicalStart(), columnRange.getLogicalStop(),
      columnMap->getSourceId(), columnMap->getSourceAxis(),
      columnMap->getDerived());
  inheritRangeAuthority(columns, columnRange);
  if (!columnRange->hasAttr(sourceSubregionAttr))
    columns.getDefiningOp()->setAttr(programBoundedOriginAttr,
                                    builder.getUnitAttr());
  Value columnValid = rangeBoundsValidity(
      builder, location, columnIndexType, columnPredicateType, columns,
      columnStop);
  auto emitRowBlock = [&](OpBuilder &rowBuilder,
                          Value rowStart) -> LogicalResult {
    Value rows = rowBuilder.create<MakeRangeOp>(
        location, rowIndexType, rowStart, blockMValue, one,
        rowRange.getLogicalStart(), rowRange.getLogicalStop(),
        rowMap->getSourceId(), rowMap->getSourceAxis(), rowMap->getDerived());
    inheritRangeAuthority(rows, rowRange);
    if (!runtimeRowTraversal && !rowRange->hasAttr(sourceSubregionAttr))
      rows.getDefiningOp()->setAttr(programBoundedOriginAttr,
                                   rowBuilder.getUnitAttr());
    Value rowValid = rangeBoundsValidity(rowBuilder, location, rowIndexType,
                                         rowPredicateType, rows, rowStop);
    IRMapping rowReplay;
    if (runtimeRowTraversal)
      rowReplay.map(rowRange.getResult(), rows);
    if (indirectRow) {
      for (AssumeInBoundsOp assumption : rowAssumptions) {
        FailureOr<Value> index = materializeSourceRanges(
            rowBuilder, location, kernel, assumption.getIndex(),
            sourceAxisIdentity(*rowMap),
            unitM, rowRange, rows, rowReplay, contract.getOperation());
        if (failed(index))
          return assumption.emitOpError(
              "blocked contraction could not replay an in-bounds assertion");
        auto replacement = rowBuilder.create<AssumeInBoundsOp>(
            location, *index, assumption.getResource(), assumption.getAxis());
        if (Attribute origin = assumption->getAttr(originAttr))
          replacement->setAttr(originAttr, origin);
      }
    }
    ContractionEpilogueMaterialization epilogueValues(
        kernel, *epilogue, blockedResultType, rows, columns,
        contract.getAccumulator());
    FailureOr<Value> accumulator = epilogueValues.capture(
        rowBuilder, contract.getAccumulator(), contract.getOperation());
    if (failed(accumulator))
      return contract.emitOpError(
          "blocked contraction accumulator has no exact result projection");

    std::string loopBodyFailure;
    auto loop = rowBuilder.create<scf::ForOp>(
        location, reductionStart, reductionStop,
        blockKValue, ValueRange{*accumulator},
        [](OpBuilder &body, Location location, Value, ValueRange carries) {
          body.create<scf::YieldOp>(location, carries);
        });
    OpBuilder reductionBuilder(loop.getBody()->getTerminator(), &created);
    auto emitReductionBody =
        [&](OpBuilder &nested, Location nestedLocation, Value kStart,
            ValueRange carries) {
          Value reductions = nested.create<MakeRangeOp>(
              nestedLocation, reductionIndexType, kStart, blockKValue, one,
              lhsReductionRange.getLogicalStart(),
              lhsReductionRange.getLogicalStop(),
              lhsReductionMap->getSourceId(), lhsReductionMap->getSourceAxis(),
              lhsReductionMap->getDerived());
          inheritRangeAuthority(reductions, lhsReductionRange);
          Value rhsReductionCoordinates = nested.create<MakeRangeOp>(
              nestedLocation, rhsReductionIndexType, kStart, blockKValue, one,
              rhsReductionRange.getLogicalStart(), rhsReductionRange.getLogicalStop(),
              rhsReductionMap->getSourceId(), rhsReductionMap->getSourceAxis(),
              rhsReductionMap->getDerived());
          inheritRangeAuthority(rhsReductionCoordinates, rhsReductionRange);
          Value reductionValid = rangeBoundsValidity(
              nested, nestedLocation, reductionIndexType,
              reductionPredicateType, reductions, reductionStop);
          Value rhsReductionValid = rangeBoundsValidity(
              nested, nestedLocation, rhsReductionIndexType,
              rhsReductionPredicateType, rhsReductionCoordinates, reductionStop);
          Value lhsRows =
              broadcastAxis(nested, nestedLocation, lhsPredicateType, rowValid, 0);
          Value lhsReductions = broadcastAxis(nested, nestedLocation,
                                          lhsPredicateType, reductionValid, 1);
          Value lhsValid = binary(nested, nestedLocation, lhsPredicateType,
                                  lhsRows, lhsReductions,
                                  BinaryOperator::LogicalAnd);
          Value rhsReductions = broadcastAxis(nested, nestedLocation,
                                          rhsPredicateType, rhsReductionValid, 0);
          Value rhsColumns =
              broadcastAxis(nested, nestedLocation, rhsPredicateType, columnValid, 1);
          Value rhsValid = binary(nested, nestedLocation, rhsPredicateType,
                                  rhsReductions, rhsColumns,
                                  BinaryOperator::LogicalAnd);
          // Replay each operand independently: a shared source can occupy
          // different M/N roles, and pointwise producers remain part of the dot.
          auto replayOperand = [&](Value source, const IRMapping &freeSeeds,
                                   MakeRangeOp freeRange,
                                   Value freeCoordinates,
                                   PhysicalExprAttr freeExtent, Value freeValid,
                                   MakeRangeOp reductionRange,
                                   Value reductionCoordinates, Value kValid,
                                   FragmentType target, Value valid)
              -> FailureOr<Value> {
            IRMapping freeReplay(freeSeeds);
            freeReplay.map(freeRange.getResult(), freeCoordinates);
            auto freeValue = materializeSourceRanges(
                nested, nestedLocation, source, freeExtent,
                ArrayRef<MakeRangeOp>(freeRange), freeCoordinates, freeReplay,
                contract.getOperation());
            if (failed(freeValue) ||
                failed(appendTailValidity(nestedLocation, source,
                                         ArrayRef<MakeRangeOp>(freeRange),
                                         freeValid, freeReplay)))
              return failure();
            IRMapping reductionReplay;
            reductionReplay.map(reductionRange.getResult(),
                                reductionCoordinates);
            auto value = materializeSourceRanges(
                nested, nestedLocation, *freeValue, unitK,
                ArrayRef<MakeRangeOp>(reductionRange), reductionCoordinates,
                reductionReplay, loop.getBody()->getTerminator());
            if (failed(value) ||
                failed(appendTailValidity(nestedLocation, *freeValue,
                                         ArrayRef<MakeRangeOp>(reductionRange),
                                         kValid, reductionReplay)))
              return failure();
            auto projected = projectPhysicalValueToSchema(
                nested, nestedLocation, *value, target);
            auto zero = materializeZeroFragment(nested, nestedLocation, target);
            if (failed(projected) || failed(zero))
              return failure();
            return nested
                .create<SelectOp>(nestedLocation, target, valid, *projected, *zero)
                .getResult();
          };
          auto lhs = replayOperand(
              contract.getLhs(), rowReplay, rowRange, rows, unitM, rowValid,
              lhsReductionRange, reductions, reductionValid, blockedLhsType,
              lhsValid);
          auto rhs = replayOperand(
              contract.getRhs(), IRMapping{}, columnRange, columns, unitN,
              columnValid,
              rhsReductionRange, rhsReductionCoordinates, rhsReductionValid,
              blockedRhsType, rhsValid);
          if (failed(lhs) || failed(rhs)) {
            loopBodyFailure = "operand value graph could not be replayed";
            return;
          }
          Value product = nested.create<ContractOp>(
              nestedLocation, blockedResultType, *lhs, *rhs, carries.front(),
              ArrayRef<int64_t>{1}, ArrayRef<int64_t>{0}, ArrayRef<int64_t>{},
              ArrayRef<int64_t>{});
          loop.getBody()->getTerminator()->setOperands(product);
        };
    emitReductionBody(reductionBuilder, location, loop.getInductionVar(),
                      loop.getRegionIterArgs());
    if (!loopBodyFailure.empty()) {
      loop.erase();
      return contract.emitOpError(
                 "blocked contraction could not materialize its loop body: ")
             << loopBodyFailure;
    }

    Value outputRows =
        broadcastAxis(rowBuilder, location, outputPredicateType, rowValid, 0);
    Value outputColumns =
        broadcastAxis(rowBuilder, location, outputPredicateType, columnValid, 1);
    Value outputValid = binary(rowBuilder, location, outputPredicateType,
                               outputRows, outputColumns,
                               BinaryOperator::LogicalAnd);
    placement::ConditionalPlacement storePlacement;
    for (auto [pathIndex, store] : llvm::enumerate(epilogue->stores)) {
      auto emitStore = [&](OpBuilder &rowBuilder) -> LogicalResult {
      OpBuilder::InsertionGuard storeInsertion(rowBuilder);
      if (!runtimeRowTraversal)
        rowBuilder.setInsertionPoint(store);
      auto output = epilogueValues.storeValue(
          rowBuilder, store, loop.getResult(0),
          runtimeRowTraversal ? contract.getOperation() : store.getOperation());
      if (failed(output))
        return store.emitOpError(
            "blocked contraction could not replay its pointwise epilogue");
      SmallVector<Value> coordinates(store.getCoordinates());
      Value rowValidity;
      if (indirectRow) {
        // Address arithmetic and residual validity remain under the original
        // conditional path. Do not share a branch-local mapping with a sibling.
        IRMapping storeRowReplay;
        storeRowReplay.map(rowRange.getResult(), rows);
        for (Value &coordinate : coordinates) {
          auto replayed = materializeSourceRanges(
              rowBuilder, location, kernel, coordinate,
              sourceAxisIdentity(*rowMap), unitM, rowRange, rows,
              storeRowReplay, contract.getOperation());
          if (failed(replayed))
            return store.emitOpError(
                "blocked contraction could not replay an output coordinate graph");
          coordinate = *replayed;
        }
        if (store.getValid()) {
          auto replayed = materializeSourceRanges(
              rowBuilder, location, kernel, store.getValid(),
              sourceAxisIdentity(*rowMap), unitM, rowRange, rows,
              storeRowReplay, contract.getOperation());
          if (failed(replayed))
            return store.emitOpError(
                "blocked contraction could not replay output validity");
          rowValidity = *replayed;
        }
      }
      FailureOr<unsigned> storeColumn = directRankOneAccessPosition(
          store.getCoordinates(), store.getValue().getType(), 1);
      if (failed(storeColumn))
        storeColumn = queryCoordinatePosition(
            store.getCoordinates(), sourceAxisIdentity(*columnMap));
      FailureOr<unsigned> storeRow = failure();
      if (!indirectRow) {
        storeRow = directRankOneAccessPosition(
            store.getCoordinates(), store.getValue().getType(), 0);
        if (failed(storeRow))
          storeRow = queryCoordinatePosition(
              store.getCoordinates(), sourceAxisIdentity(*rowMap));
      }
      if (failed(storeColumn))
        return store.emitOpError(
            "blocked contract output lost its logical source coordinates");
      SmallVector<std::pair<MakeRangeOp, Value>> storeTailRanges(outputTailRanges);
      if (!indirectRow) {
        if (failed(storeRow))
          return store.emitOpError(
              "blocked contract output lost its logical source coordinates");
        for (auto [position, authority, end] : {
                 std::tuple<unsigned, MakeRangeOp, Value>{*storeRow, rowRange, rowStop},
                 std::tuple<unsigned, MakeRangeOp, Value>{*storeColumn, columnRange, columnStop}}) {
          MakeRangeOp stored = sourceRange(store.getCoordinates()[position]);
          if (!stored)
            continue;
          Value storedEnd = (stored).getLogicalStop();
          if (samePhysicalScalarExpression(storedEnd, end) &&
              samePhysicalScalarExpression(stored.getLogicalStart(), authority.getLogicalStart()) &&
              samePhysicalScalarExpression(stored.getStep(), authority.getStep()))
            storeTailRanges.emplace_back(stored, storedEnd);
        }
        coordinates[*storeRow] = rows;
      }
      coordinates[*storeColumn] = columns;
      FailureOr<Value> valid = failure();
      if (indirectRow) {
        Value replayed = rowValidity;
        if (!replayed) {
          valid = outputValid;
        } else {
          IRMapping columnReplay;
          columnReplay.map(columnRange.getResult(), columns);
          FailureOr<Value> columnValidity = materializeSourceRanges(
              rowBuilder, location, kernel, replayed,
              sourceAxisIdentity(*columnMap), unitN, columnRange, columns,
              columnReplay, &*rowBuilder.getInsertionPoint());
          if (failed(columnValidity))
            return store.emitOpError(
                "blocked contraction could not replay output column validity");
          valid = materializeValidityConjunction(
              rowBuilder, location, outputValid, *columnValidity,
              outputPredicateType);
        }
      } else {
        Value originalValidity = store.getValid();
        // The new row/column mask covers proven tails. Other predicates are
        // replayed through each exact range occurrence below.
        if (originalValidity &&
            PhysicalProgramAnalysis(kernel).isTailPredicate(originalValidity, storeTailRanges))
          originalValidity = Value();
        if (originalValidity) {
          IRMapping replay;
          replay.map(rowRange.getResult(), rows);
          FailureOr<Value> replayed = materializeSourceRanges(
              rowBuilder, location, kernel, originalValidity,
              sourceAxisIdentity(*rowMap), unitM, rowRange, rows, replay,
              &*rowBuilder.getInsertionPoint());
          if (failed(replayed))
            return store.emitOpError(
                "blocked contraction could not relocate output validity");
          originalValidity = *replayed;
        }
        if (originalValidity) {
          IRMapping replay;
          replay.map(columnRange.getResult(), columns);
          FailureOr<Value> replayed = materializeSourceRanges(
              rowBuilder, location, kernel, originalValidity,
              sourceAxisIdentity(*columnMap), unitN, columnRange, columns,
              replay, &*rowBuilder.getInsertionPoint());
          if (failed(replayed))
            return store.emitOpError(
                "blocked contraction could not relocate output column validity");
          originalValidity = *replayed;
        }
        valid = materializeValidityConjunction(
            rowBuilder, location, outputValid, originalValidity,
            outputPredicateType);
      }
      if (failed(valid))
        return store.emitOpError(
            "blocked contract output validity could not be retargeted");
      Value resource = store.getResource();
      if (runtimeRowTraversal) {
        auto captured = epilogueValues.capture(rowBuilder, resource, contract);
        if (failed(captured))
          return store.emitOpError("blocked contraction could not replay its output resource");
        resource = *captured;
        for (auto [position, coordinate] : llvm::enumerate(coordinates)) {
          if (position == *storeColumn ||
              (succeeded(storeRow) && position == *storeRow))
            continue;
          if (!isa<FragmentType>(coordinate.getType())) {
            auto captured = epilogueValues.capture(rowBuilder, coordinate, contract);
            if (failed(captured))
              return store.emitOpError("blocked contraction could not replay a scalar output coordinate");
            coordinates[position] = *captured;
          }
        }
      }
      auto replacement = rowBuilder.create<StoreOp>(
          location, resource, coordinates, *output, *valid,
          store.getSourceAxes());
      if (Attribute origin = store->getAttr(originAttr))
        replacement->setAttr(originAttr, origin);
      return success();
      };
      if (runtimeRowTraversal) {
        auto condition = [&](OpBuilder &nested, Operation *original)
            -> FailureOr<Value> {
          auto branch = cast<scf::IfOp>(original);
          // Execution-group refinement may have replaced original coordinates
          // since preflight. Recompute bindings from the current SSA graph.
          auto bindings = scalarCaptureBindings(branch.getCondition(), contract);
          if (failed(bindings))
            return failure();
          return materializeReplayedValue(
              nested, branch.getLoc(), branch.getCondition(),
              sourceAxisIdentity(*rowMap), unitM, *bindings, contract);
        };
        if (failed(storePlacement.emit(rowBuilder, storeScopes[pathIndex],
                                        condition, emitStore)))
          return store.emitOpError("blocked contraction could not preserve its output control scope");
      } else if (failed(emitStore(rowBuilder))) {
        return failure();
      }
    }
    return success();
  };

  if (runtimeRowTraversal) {
    Value rowStart = rowRange.getStart();
    if (failed(existingRowAxis))
      rowStart = binary(
          builder, location, builder.getIndexType(), rowStart,
          binary(builder, location, builder.getIndexType(), rowWorker,
                 blockMValue, BinaryOperator::Multiply),
          BinaryOperator::Add);
    Value rowStep = binary(builder, location, builder.getIndexType(),
                           blockMValue, rowWorkersValue,
                           BinaryOperator::Multiply);
    auto rowLoop = builder.create<scf::ForOp>(
        location, rowStart, rowStop, rowStep);
    OpBuilder nested(rowLoop.getBody()->getTerminator(), &created);
    if (failed(emitRowBlock(nested, rowLoop.getInductionVar())) ||
        failed(preserveRuntimeContractionSnapshot(
            rowLoop, rowRange, columnRange, columns, columnValid,
            blockMValue, blockNValue, rowWorker, columnTile, &created))) {
      rowLoop.erase();
      return failure();
    }
  } else {
    Value rowStart = rowRange.getStart();
    if (failed(existingRowAxis))
      rowStart = binary(
          builder, location, builder.getIndexType(), rowStart,
          binary(builder, location, builder.getIndexType(), rowTile,
                 blockMValue, BinaryOperator::Multiply),
          BinaryOperator::Add);
    if (failed(emitRowBlock(builder, rowStart)))
      return failure();
  }

  // Retiling a pointwise epilogue can replay a sibling matrix expression.
  // Retire the obsolete epilogue before its original products are visited,
  // and let the new products receive their own bounded reduction traversal.
  // Keep products themselves alive until their worklist entry is consumed.
  eraseContractionEpilogue(*epilogue);
  llvm::append_range(pending, created.products);
  for (AssumeInBoundsOp assumption : rowAssumptions)
    if (assumption->getBlock())
      assumption.erase();
  return success();
}

LogicalResult realizeScaledContract(ScaledContractOp contract,
                                    func::FuncOp kernel) {
  if (!contract->getBlock())
    return success();
  if (!requiresPhysicalRealization(contract))
    return success();
  PhysicalProgramAnalysis physicalAnalysis(kernel);
  auto sourceLoad = [&](Value value) -> LoadOp {
    PhysicalReplayFact fact = physicalAnalysis.replayability(
        value, std::nullopt, PhysicalReplayScope::ValueGraph,
        /*allowAccesses=*/true);
    return fact.isReplayable() && fact.accesses.size() == 1
               ? dyn_cast<LoadOp>(fact.accesses.front())
               : LoadOp();
  };
  auto lhsLoad = sourceLoad(contract.getLhs());
  auto lhsScaleLoad = sourceLoad(contract.getLhsScale());
  auto rhsLoad = sourceLoad(contract.getRhs());
  auto rhsScaleLoad = sourceLoad(contract.getRhsScale());
  auto lhsType = contract.getLhs().getType();
  auto lhsScaleType = contract.getLhsScale().getType();
  auto rhsType = contract.getRhs().getType();
  auto rhsScaleType = contract.getRhsScale().getType();
  auto resultType = contract.getResult().getType();
  auto reject = [&](const Twine &reason) -> LogicalResult {
    return contract.emitOpError()
           << "cannot form a complete block-scaled contraction: " << reason;
  };
  ArrayRef<int64_t> lhsReduction = contract.getLhsReductionAxes();
  ArrayRef<int64_t> rhsReduction = contract.getRhsReductionAxes();
  bool fixedAxes = lhsReduction.size() == 2 && rhsReduction.size() == 2 &&
                   lhsReduction[0] == 1 && lhsReduction[1] == 2 &&
                   rhsReduction[0] == 0 && rhsReduction[1] == 1 &&
                   contract.getLhsBatchAxes().empty() &&
                   contract.getRhsBatchAxes().empty();
  if (!lhsLoad || !lhsScaleLoad || !rhsLoad || !rhsScaleLoad ||
      lhsType.getShape().size() != 3 || lhsScaleType.getShape().size() != 2 ||
      rhsType.getShape().size() != 3 || rhsScaleType.getShape().size() != 2 ||
      resultType.getShape().size() != 2 || !fixedAxes)
    return reject("requires one replayable source load for every operand and the closed [M,G,C]/[M,G] x [G,C,N]/[N,G] schema");
  if (contract.getLhsGroupSize() <= 0 ||
      contract.getLhsGroupSize() != contract.getRhsGroupSize())
    return reject("requires one equal positive scale-group size");
  constexpr unsigned lhsFree = 0;
  constexpr unsigned lhsBlockAxis = 1;
  constexpr unsigned lhsInnerAxis = 2;
  constexpr unsigned rhsBlockAxis = 0;
  constexpr unsigned rhsInnerAxis = 1;
  constexpr unsigned rhsFree = 2;
  auto lhsInnerExtent =
      cast<PhysicalExprAttr>(lhsType.getShape()[lhsInnerAxis]);
  auto rhsInnerExtent =
      cast<PhysicalExprAttr>(rhsType.getShape()[rhsInnerAxis]);

  FailureOr<AxisMapAttr> rowMap = queryAxisMap(lhsType, lhsFree);
  FailureOr<AxisMapAttr> lhsBlockMap = queryAxisMap(lhsType, lhsBlockAxis);
  FailureOr<AxisMapAttr> lhsInnerMap = queryAxisMap(lhsType, lhsInnerAxis);
  FailureOr<AxisMapAttr> rhsBlockMap = queryAxisMap(rhsType, rhsBlockAxis);
  FailureOr<AxisMapAttr> rhsInnerMap = queryAxisMap(rhsType, rhsInnerAxis);
  FailureOr<AxisMapAttr> columnMap = queryAxisMap(rhsType, rhsFree);
  auto scaleMapFor = [&](FragmentType scale,
                         AxisMapAttr data) -> FailureOr<AxisMapAttr> {
    PhysicalAxisProjection projection = queryFragmentAxis(
        scale, sourceAxisIdentity(data));
    return projection.isExact() ? queryAxisMap(scale, projection.fragmentAxis)
                                : FailureOr<AxisMapAttr>(failure());
  };
  FailureOr<AxisMapAttr> lhsScaleRowMap =
      succeeded(rowMap) ? scaleMapFor(lhsScaleType, *rowMap)
                        : FailureOr<AxisMapAttr>(failure());
  FailureOr<AxisMapAttr> lhsScaleBlockMap =
      succeeded(lhsBlockMap) ? scaleMapFor(lhsScaleType, *lhsBlockMap)
                             : FailureOr<AxisMapAttr>(failure());
  FailureOr<AxisMapAttr> rhsScaleBlockMap =
      succeeded(rhsBlockMap) ? scaleMapFor(rhsScaleType, *rhsBlockMap)
                             : FailureOr<AxisMapAttr>(failure());
  FailureOr<AxisMapAttr> rhsScaleColumnMap =
      succeeded(columnMap) ? scaleMapFor(rhsScaleType, *columnMap)
                           : FailureOr<AxisMapAttr>(failure());
  if (failed(rowMap) || failed(lhsBlockMap) || failed(lhsInnerMap) ||
      failed(rhsBlockMap) || failed(rhsInnerMap) || failed(columnMap) ||
      failed(lhsScaleRowMap) || failed(lhsScaleBlockMap) ||
      failed(rhsScaleBlockMap) || failed(rhsScaleColumnMap))
    return reject("data and scale operands do not preserve the paired coordinate provenance");
  auto sameSource = [](AxisMapAttr lhs, AxisMapAttr rhs) {
    return sourceAxisIdentity(lhs) == sourceAxisIdentity(rhs);
  };
  auto sameDimension = [](AxisMapAttr lhs, AxisMapAttr rhs) {
    return lhs.getDimensionId() > 0 &&
           lhs.getDimensionId() == rhs.getDimensionId();
  };
  if (!sameSource(*rowMap, *lhsScaleRowMap) ||
      !sameDimension(*lhsBlockMap, *rhsBlockMap) ||
      !sameSource(*lhsBlockMap, *lhsScaleBlockMap) ||
      !sameDimension(*lhsBlockMap, *rhsScaleBlockMap) ||
      !sameDimension(*lhsInnerMap, *rhsInnerMap) ||
      !sameSource(*columnMap, *rhsScaleColumnMap))
    return reject("data and scale operands do not preserve the paired coordinate provenance");

  auto rangeFor = [&](LoadOp load,
                      AxisMapAttr source) -> FailureOr<MakeRangeOp> {
    FailureOr<unsigned> coordinate = queryCoordinatePosition(
        load.getCoordinates(),
        sourceAxisIdentity(source));
    if (failed(coordinate))
      return failure();
    PhysicalProgramAnalysis analysis(kernel);
    PhysicalRangeFact fact = analysis.sourceRanges(
        load.getCoordinates()[*coordinate],
        sourceAxisIdentity(source));
    return queryExactLogicalRange(fact);
  };
  FailureOr<MakeRangeOp> rowRange = rangeFor(lhsLoad, *rowMap);
  FailureOr<MakeRangeOp> blockRange =
      rangeFor(lhsLoad, *lhsBlockMap);
  FailureOr<MakeRangeOp> innerRange =
      rangeFor(lhsLoad, *lhsInnerMap);
  FailureOr<MakeRangeOp> columnRange =
      rangeFor(rhsLoad, *columnMap);
  FailureOr<MakeRangeOp> lhsScaleRowRange =
      rangeFor(lhsScaleLoad, *lhsScaleRowMap);
  FailureOr<MakeRangeOp> lhsScaleBlockRange =
      rangeFor(lhsScaleLoad, *lhsScaleBlockMap);
  FailureOr<MakeRangeOp> rhsBlockRange =
      rangeFor(rhsLoad, *rhsBlockMap);
  FailureOr<MakeRangeOp> rhsInnerRange =
      rangeFor(rhsLoad, *rhsInnerMap);
  FailureOr<MakeRangeOp> rhsScaleBlockRange =
      rangeFor(rhsScaleLoad, *rhsScaleBlockMap);
  FailureOr<MakeRangeOp> rhsScaleColumnRange =
      rangeFor(rhsScaleLoad, *rhsScaleColumnMap);
  if (failed(rowRange) || failed(blockRange) || failed(innerRange) ||
      failed(columnRange) || failed(lhsScaleRowRange) ||
      failed(lhsScaleBlockRange) || failed(rhsBlockRange) ||
      failed(rhsInnerRange) || failed(rhsScaleBlockRange) ||
      failed(rhsScaleColumnRange))
    return reject("physical data coordinates are not explicit source ranges");
  if (!isUnitStepRange(*rowRange) || !isUnitStepRange(*blockRange) ||
      !isUnitStepRange(*innerRange) || !isUnitStepRange(*columnRange))
    return reject("blocking currently requires unit-step source ranges");
  auto epilogue = collectContractionEpilogue(contract.getResult());
  if (!epilogue)
    return reject("result does not have a closed store epilogue");
  Value rowLogicalEnd = (*rowRange).getLogicalStop();
  Value blockLogicalEnd = (*blockRange).getLogicalStop();
  Value innerLogicalEnd = (*innerRange).getLogicalStop();
  Value columnLogicalEnd = (*columnRange).getLogicalStop();
  Value lhsScaleRowLogicalEnd = (*lhsScaleRowRange).getLogicalStop();
  Value lhsScaleBlockLogicalEnd = (*lhsScaleBlockRange).getLogicalStop();
  Value rhsBlockLogicalEnd = (*rhsBlockRange).getLogicalStop();
  Value rhsInnerLogicalEnd = (*rhsInnerRange).getLogicalStop();
  Value rhsScaleBlockLogicalEnd = (*rhsScaleBlockRange).getLogicalStop();
  Value rhsScaleColumnLogicalEnd = (*rhsScaleColumnRange).getLogicalStop();

  SmallVector<std::pair<MakeRangeOp, Value>> lhsTailRanges = {
      {*rowRange, rowLogicalEnd},
      {*blockRange, blockLogicalEnd},
      {*innerRange, innerLogicalEnd},
  };
  SmallVector<std::pair<MakeRangeOp, Value>> lhsScaleTailRanges = {
      {*lhsScaleRowRange, lhsScaleRowLogicalEnd},
      {*lhsScaleBlockRange, lhsScaleBlockLogicalEnd},
  };
  SmallVector<std::pair<MakeRangeOp, Value>> rhsTailRanges = {
      {*rhsBlockRange, rhsBlockLogicalEnd},
      {*rhsInnerRange, rhsInnerLogicalEnd},
      {*columnRange, columnLogicalEnd},
  };
  SmallVector<std::pair<MakeRangeOp, Value>> rhsScaleTailRanges = {
      {*rhsScaleBlockRange, rhsScaleBlockLogicalEnd},
      {*rhsScaleColumnRange, rhsScaleColumnLogicalEnd},
  };
  SmallVector<std::pair<MakeRangeOp, Value>> outputTailRanges = {
      {*rowRange, rowLogicalEnd},
      {*columnRange, columnLogicalEnd},
  };
  auto validIsTail = [&](Value valid,
                         ArrayRef<std::pair<MakeRangeOp, Value>> ranges) {
    return !valid || succeeded(scalarSource(valid)) ||
           isTailPredicate(valid, ranges);
  };
  if (!validIsTail(lhsLoad.getValid(), lhsTailRanges) ||
      !validIsTail(lhsScaleLoad.getValid(), lhsScaleTailRanges) ||
      !validIsTail(rhsLoad.getValid(), rhsTailRanges) ||
      !validIsTail(rhsScaleLoad.getValid(), rhsScaleTailRanges))
    return reject("data/scale loads contain non-tail residual validity");
  for (StoreOp store : epilogue->stores)
    if (!validIsTail(store.getValid(), outputTailRanges))
      return reject("result store contains non-tail residual validity");
  if ((lhsLoad.getFill() && !isZeroScalar(lhsLoad.getFill())) ||
      (rhsLoad.getFill() && !isZeroScalar(rhsLoad.getFill())))
    return reject("invalid data fill is not the contraction zero");
  FailureOr<Value> initialAccumulator = scalarSource(contract.getAccumulator());
  if (failed(initialAccumulator) ||
      (*initialAccumulator).getType() != resultType.getElementType())
    return reject("accumulator is not an explicit scalarizable value");

  ProgramSegment segment = queryContractionProgramSegment(
      kernel, contract, ProgramMappingScope::SameBlock);
  ExecutionGroupOp mapping = segment.mapping;
  PhysicalExprAttr segmentLength = segment.length;
  if (!segment.isComplete())
    return reject("requires one complete current program segment");

  auto coordinateFor = [](ValueRange coordinates, AxisMapAttr mapping) {
    return queryCoordinatePosition(
        coordinates,
        sourceAxisIdentity(mapping));
  };
  FailureOr<unsigned> lhsRowCoordinate =
      coordinateFor(lhsLoad.getCoordinates(), *rowMap);
  FailureOr<unsigned> lhsBlockCoordinate =
      coordinateFor(lhsLoad.getCoordinates(), *lhsBlockMap);
  FailureOr<unsigned> lhsInnerCoordinate =
      coordinateFor(lhsLoad.getCoordinates(), *lhsInnerMap);
  FailureOr<unsigned> lhsScaleRowCoordinate =
      coordinateFor(lhsScaleLoad.getCoordinates(), *lhsScaleRowMap);
  FailureOr<unsigned> lhsScaleBlockCoordinate =
      coordinateFor(lhsScaleLoad.getCoordinates(), *lhsScaleBlockMap);
  FailureOr<unsigned> rhsBlockCoordinate =
      coordinateFor(rhsLoad.getCoordinates(), *rhsBlockMap);
  FailureOr<unsigned> rhsInnerCoordinate =
      coordinateFor(rhsLoad.getCoordinates(), *rhsInnerMap);
  FailureOr<unsigned> rhsColumnCoordinate =
      coordinateFor(rhsLoad.getCoordinates(), *columnMap);
  FailureOr<unsigned> rhsScaleBlockCoordinate =
      coordinateFor(rhsScaleLoad.getCoordinates(), *rhsScaleBlockMap);
  FailureOr<unsigned> rhsScaleColumnCoordinate =
      coordinateFor(rhsScaleLoad.getCoordinates(), *rhsScaleColumnMap);
  if (failed(lhsRowCoordinate) || failed(lhsBlockCoordinate) ||
      failed(lhsInnerCoordinate) || failed(lhsScaleRowCoordinate) ||
      failed(lhsScaleBlockCoordinate) || failed(rhsBlockCoordinate) ||
      failed(rhsInnerCoordinate) || failed(rhsColumnCoordinate) ||
      failed(rhsScaleBlockCoordinate) || failed(rhsScaleColumnCoordinate))
    return reject("load coordinates do not cover all data and scale axes");

  unsigned rowResourceAxis = lhsLoad.getSourceAxes()[*lhsRowCoordinate];
  unsigned columnResourceAxis = rhsLoad.getSourceAxes()[*rhsColumnCoordinate];
  MLIRContext *context = kernel.getContext();
  Location location = contract.getLoc();
  std::string suffix =
      ("_" + Twine(rowMap->getSourceId()) + "_" +
       Twine(columnMap->getSourceId()))
          .str();
  auto blockMRef = getOrCreatePhysicalParameter(
      kernel, "BLOCK_M" + suffix, ParameterRole::OwnershipM,
      ParameterCategory::Contraction,
      lhsType.getElementType().getIntOrFloatBitWidth(), {64, 128});
  auto blockNRef = getOrCreatePhysicalParameter(
      kernel, "BLOCK_N" + suffix, ParameterRole::OwnershipN,
      ParameterCategory::Contraction,
      rhsType.getElementType().getIntOrFloatBitWidth(), {64, 128});
  auto blockKRef = getOrCreatePhysicalParameter(
      kernel, "BLOCK_K_GROUPS" + suffix, ParameterRole::Reduction,
      ParameterCategory::Contraction,
      std::max(lhsType.getElementType().getIntOrFloatBitWidth(),
               rhsType.getElementType().getIntOrFloatBitWidth()),
      {2, 4, 8});
  if (failed(blockMRef) || failed(blockNRef) || failed(blockKRef))
    return failure();
  ParameterAttr blockM = lookupParameter(kernel, *blockMRef);
  ParameterAttr blockN = lookupParameter(kernel, *blockNRef);
  ParameterAttr blockK = lookupParameter(kernel, *blockKRef);
  FailureOr<unsigned> existingRowAxis =
      mappingAxisForScalar(rowRange->getStart(), mapping);
  FailureOr<unsigned> existingColumnAxis =
      mappingAxisForScalar(columnRange->getStart(), mapping);
  if (succeeded(existingRowAxis) && succeeded(existingColumnAxis) &&
      *existingRowAxis == *existingColumnAxis)
    return reject(
        "row and column ownership share one mapping coordinate that cannot be refined independently");
  if (failed(refineOwnershipParameter(kernel, *rowRange, existingRowAxis,
                                      blockM)) ||
      failed(refineOwnershipParameter(kernel, *columnRange, existingColumnAxis,
                                      blockN)))
    return failure();
  for (auto [parameter, range] :
       {std::pair<ParameterAttr, MakeRangeOp>{blockM, *rowRange},
        std::pair<ParameterAttr, MakeRangeOp>{blockN, *columnRange},
        std::pair<ParameterAttr, MakeRangeOp>{blockK, *blockRange}})
    if (FailureOr<int64_t> dimension = queryRangeDimension(range);
        succeeded(dimension))
      if (failed(updateParameter(kernel, lookupParameter(kernel, parameter.getReference()).withBinding(
              lookupParameter(kernel, parameter.getReference()).getBinding().withDimension(
                  IntegerAttr::get(IntegerType::get(kernel.getContext(), 64), *dimension))))))
        return failure();
  OpBuilder parameterBuilder(&kernel.front(), kernel.front().begin());
  Value blockMValue = materializeParameter(parameterBuilder, location, *blockMRef);
  Value blockNValue = materializeParameter(parameterBuilder, location, *blockNRef);
  Value blockKValue = materializeParameter(parameterBuilder, location, *blockKRef);
  PhysicalExprAttr unitM = parameterExpression(
      context, blockM.getName().getValue());
  PhysicalExprAttr unitN = parameterExpression(
      context, blockN.getName().getValue());
  PhysicalExprAttr unitK = parameterExpression(
      context, blockK.getName().getValue());

  OpBuilder mapBuilder(mapping);
  Value rowExtent = mapBuilder.create<DimOp>(
      location, mapBuilder.getIndexType(), lhsLoad.getResource(), rowResourceAxis);
  Value columnExtent = mapBuilder.create<DimOp>(
      location, mapBuilder.getIndexType(), rhsLoad.getResource(),
      columnResourceAxis);
  auto ceilDiv = [&](Value extent, Value divisor) {
    Value one = mapBuilder.create<arith::ConstantIndexOp>(location, 1);
    Value adjusted = binary(
        mapBuilder, location, mapBuilder.getIndexType(), extent,
        binary(mapBuilder, location, mapBuilder.getIndexType(), divisor, one,
               BinaryOperator::Subtract),
        BinaryOperator::Add);
    return binary(mapBuilder, location, mapBuilder.getIndexType(), adjusted,
                  divisor, BinaryOperator::FloorDivide);
  };
  Value rowTiles = ceilDiv(rowExtent, blockMValue);
  Value columnTiles = ceilDiv(columnExtent, blockNValue);
  SmallVector<Value> mappingExtents(mapping.getExtents());
  SmallVector<Attribute> launchExtents(mapping.getLaunchExtents().begin(),
                                       mapping.getLaunchExtents().end());
  auto rowExpression = cast<PhysicalExprAttr>(
      cast<ViewType>(lhsLoad.getResource().getType())
          .getLayout()
          .getExtents()[rowResourceAxis]);
  auto columnExpression = cast<PhysicalExprAttr>(
      cast<ViewType>(rhsLoad.getResource().getType())
          .getLayout()
          .getExtents()[columnResourceAxis]);
  SmallVector<int64_t> coordinateRoles(mapping.getCoordinates().size(), -1);
  if (auto existing =
          mapping.getCoordinateRolesAttr())
    if (existing.size() == mapping.getCoordinates().size())
      llvm::copy(existing.asArrayRef(), coordinateRoles.begin());
  auto bindMappingAxis = [&](FailureOr<unsigned> existing, Value extent,
                             PhysicalExprAttr launchExtent,
                             CoordinateRole role) {
    unsigned axis;
    if (succeeded(existing)) {
      axis = *existing;
      mappingExtents[axis] = extent;
      launchExtents[axis] = launchExtent;
    } else {
      axis = mappingExtents.size();
      mappingExtents.push_back(extent);
      launchExtents.push_back(launchExtent);
      coordinateRoles.push_back(-1);
    }
    coordinateRoles[axis] = static_cast<int64_t>(role);
    return axis;
  };
  unsigned rowMappingAxis = bindMappingAxis(
      existingRowAxis, rowTiles,
      binaryExpression(context, PhysicalExprKind::CeilDiv, rowExpression,
                       unitM),
      CoordinateRole::ContractionM);
  unsigned columnMappingAxis = bindMappingAxis(
      existingColumnAxis, columnTiles,
      binaryExpression(context, PhysicalExprKind::CeilDiv, columnExpression,
                       unitN),
      CoordinateRole::ContractionN);
  segmentLength = cast<PhysicalExprAttr>(launchExtents.front());
  for (Attribute extent : llvm::drop_begin(launchExtents))
    segmentLength = binaryExpression(context, PhysicalExprKind::Multiply,
                                     segmentLength,
                                     cast<PhysicalExprAttr>(extent));
  auto expandedMapping = rebuildExecutionGroup(
      mapBuilder, mapping, mappingExtents, mapBuilder.getArrayAttr(launchExtents),
      mapBuilder.getDenseI64ArrayAttr(coordinateRoles), segmentLength);
  for (auto [oldCoordinate, newCoordinate] : llvm::zip(
           mapping.getCoordinates(),
           expandedMapping.getCoordinates().take_front(mapping.getCoordinates().size())))
    oldCoordinate.replaceAllUsesWith(newCoordinate);
  Value rowTile = expandedMapping.getCoordinates()[rowMappingAxis];
  Value columnTile = expandedMapping.getCoordinates()[columnMappingAxis];
  mapping.erase();
  kernel->setAttr(programSpaceAttr,
                  ArrayAttr::get(context, {segmentLength}));

  OpBuilder builder(contract);
  Value one = builder.create<arith::ConstantIndexOp>(location, 1);
  Value rowStart = rowRange->getStart();
  if (failed(existingRowAxis))
    rowStart = binary(
        builder, location, builder.getIndexType(), rowStart,
        binary(builder, location, builder.getIndexType(), rowTile,
               blockMValue, BinaryOperator::Multiply),
        BinaryOperator::Add);
  Value rowStop = rowLogicalEnd;
  Value columnStart = columnRange->getStart();
  if (failed(existingColumnAxis))
    columnStart = binary(
        builder, location, builder.getIndexType(), columnStart,
        binary(builder, location, builder.getIndexType(), columnTile,
               blockNValue, BinaryOperator::Multiply),
        BinaryOperator::Add);
  Value columnStop = columnLogicalEnd;
  Value blockStop = blockLogicalEnd;

  FragmentType rowIndexType = fragmentType(
      context, builder.getIndexType(), {unitM}, {*rowMap}, lhsType.getOwner());
  FragmentType columnIndexType = fragmentType(
      context, builder.getIndexType(), {unitN}, {*columnMap}, rhsType.getOwner());
  FragmentType blockIndexType = fragmentType(
      context, builder.getIndexType(), {unitK}, {*lhsBlockMap}, lhsType.getOwner());
  FragmentType rowPredicateType = fragmentType(
      context, builder.getI1Type(), {unitM}, {*rowMap}, lhsType.getOwner());
  FragmentType columnPredicateType = fragmentType(
      context, builder.getI1Type(), {unitN}, {*columnMap}, rhsType.getOwner());
  FragmentType blockPredicateType = fragmentType(
      context, builder.getI1Type(), {unitK}, {*lhsBlockMap}, lhsType.getOwner());
  FragmentType blockedLhsType = fragmentType(
      context, lhsType.getElementType(), {unitM, unitK, lhsInnerExtent},
      {*rowMap, *lhsBlockMap, *lhsInnerMap}, lhsType.getOwner());
  FragmentType blockedLhsScaleType = fragmentType(
      context, lhsScaleType.getElementType(), {unitM, unitK},
      {*lhsScaleRowMap, *lhsScaleBlockMap}, lhsScaleType.getOwner());
  FragmentType blockedRhsType = fragmentType(
      context, rhsType.getElementType(), {unitK, rhsInnerExtent, unitN},
      {*rhsBlockMap, *rhsInnerMap, *columnMap}, rhsType.getOwner());
  FragmentType blockedRhsScaleType = fragmentType(
      context, rhsScaleType.getElementType(), {unitN, unitK},
      {*rhsScaleColumnMap, *rhsScaleBlockMap}, rhsScaleType.getOwner());
  FragmentType blockedResultType = fragmentType(
      context, resultType.getElementType(), {unitM, unitN},
      {*rowMap, *columnMap}, resultType.getOwner());
  FragmentType lhsPredicateType = fragmentType(
      context, builder.getI1Type(), {unitM, unitK, lhsInnerExtent},
      {*rowMap, *lhsBlockMap, *lhsInnerMap}, lhsType.getOwner());
  FragmentType lhsScalePredicateType = fragmentType(
      context, builder.getI1Type(), {unitM, unitK},
      {*lhsScaleRowMap, *lhsScaleBlockMap}, lhsScaleType.getOwner());
  FragmentType rhsPredicateType = fragmentType(
      context, builder.getI1Type(), {unitK, rhsInnerExtent, unitN},
      {*rhsBlockMap, *rhsInnerMap, *columnMap}, rhsType.getOwner());
  FragmentType rhsScalePredicateType = fragmentType(
      context, builder.getI1Type(), {unitN, unitK},
      {*rhsScaleColumnMap, *rhsScaleBlockMap}, rhsScaleType.getOwner());
  FragmentType outputPredicateType = fragmentType(
      context, builder.getI1Type(), {unitM, unitN},
      {*rowMap, *columnMap}, resultType.getOwner());

  Value rows = builder.create<MakeRangeOp>(
      location, rowIndexType, rowStart, blockMValue, one,
      rowRange->getLogicalStart(), rowRange->getLogicalStop(),
      rowMap->getSourceId(), rowMap->getSourceAxis(), rowMap->getDerived());
  inheritRangeAuthority(rows, *rowRange);
  if (!(*rowRange)->hasAttr(sourceSubregionAttr))
    rows.getDefiningOp()->setAttr(programBoundedOriginAttr,
                                 builder.getUnitAttr());
  Value columns = builder.create<MakeRangeOp>(
      location, columnIndexType, columnStart, blockNValue, one,
      columnRange->getLogicalStart(), columnRange->getLogicalStop(),
      columnMap->getSourceId(), columnMap->getSourceAxis(),
      columnMap->getDerived());
  inheritRangeAuthority(columns, *columnRange);
  if (!(*columnRange)->hasAttr(sourceSubregionAttr))
    columns.getDefiningOp()->setAttr(programBoundedOriginAttr,
                                    builder.getUnitAttr());
  Value rowValid = rangeBoundsValidity(builder, location, rowIndexType,
                                       rowPredicateType, rows, rowStop);
  Value columnValid = rangeBoundsValidity(
      builder, location, columnIndexType, columnPredicateType, columns,
      columnStop);
  Value accumulator = builder.create<SplatOp>(
      location, blockedResultType, *initialAccumulator);
  bool loopBodyFailed = false;
  auto loop = createTraversalLoop(builder,
      location, blockRange->getStart(), blockStop, blockKValue,
      ValueRange{accumulator},
      [&](OpBuilder &nested, Location nestedLocation, Value blockStart,
          ValueRange carries) {
        Value blocks = nested.create<MakeRangeOp>(
            nestedLocation, blockIndexType, blockStart, blockKValue, one,
            blockRange->getLogicalStart(), blockRange->getLogicalStop(),
            lhsBlockMap->getSourceId(), lhsBlockMap->getSourceAxis(),
            lhsBlockMap->getDerived());
        inheritRangeAuthority(blocks, *blockRange);
        Value blockValid = rangeBoundsValidity(
            nested, nestedLocation, blockIndexType, blockPredicateType, blocks,
            blockStop);
        Value lhsRows = broadcastAxis(nested, nestedLocation, lhsPredicateType,
                                  rowValid, 0);
        Value lhsBlocks = broadcastAxis(nested, nestedLocation, lhsPredicateType,
                                    blockValid, 1);
        Value lhsValid = binary(nested, nestedLocation, lhsPredicateType,
                                lhsRows, lhsBlocks,
                                BinaryOperator::LogicalAnd);
        Value lhsScaleRows = broadcastAxis(
            nested, nestedLocation, lhsScalePredicateType, rowValid, 0);
        Value lhsScaleBlocks = broadcastAxis(
            nested, nestedLocation, lhsScalePredicateType, blockValid, 1);
        Value lhsScaleValid = binary(
            nested, nestedLocation, lhsScalePredicateType, lhsScaleRows,
            lhsScaleBlocks, BinaryOperator::LogicalAnd);
        Value rhsBlocks = broadcastAxis(nested, nestedLocation, rhsPredicateType,
                                    blockValid, 0);
        Value rhsColumns = broadcastAxis(nested, nestedLocation, rhsPredicateType,
                                     columnValid, 2);
        Value rhsValid = binary(nested, nestedLocation, rhsPredicateType,
                                rhsBlocks, rhsColumns,
                                BinaryOperator::LogicalAnd);
        Value rhsScaleBlocks = broadcastAxis(
            nested, nestedLocation, rhsScalePredicateType, blockValid, 1);
        Value rhsScaleColumns = broadcastAxis(
            nested, nestedLocation, rhsScalePredicateType, columnValid, 0);
        Value rhsScaleValid = binary(
            nested, nestedLocation, rhsScalePredicateType, rhsScaleBlocks,
            rhsScaleColumns, BinaryOperator::LogicalAnd);

        FailureOr<Value> retargetedLhs = materializeRetargetedValidity(
            nested, nestedLocation, lhsLoad.getValid(), lhsTailRanges,
            lhsValid, lhsPredicateType);
        FailureOr<Value> retargetedLhsScale = materializeRetargetedValidity(
            nested, nestedLocation, lhsScaleLoad.getValid(),
            lhsScaleTailRanges, lhsScaleValid, lhsScalePredicateType);
        FailureOr<Value> retargetedRhs = materializeRetargetedValidity(
            nested, nestedLocation, rhsLoad.getValid(), rhsTailRanges,
            rhsValid, rhsPredicateType);
        FailureOr<Value> retargetedRhsScale = materializeRetargetedValidity(
            nested, nestedLocation, rhsScaleLoad.getValid(),
            rhsScaleTailRanges, rhsScaleValid, rhsScalePredicateType);
        if (failed(retargetedLhs) || failed(retargetedLhsScale) ||
            failed(retargetedRhs) || failed(retargetedRhsScale)) {
          loopBodyFailed = true;
          return;
        }
        lhsValid = *retargetedLhs;
        lhsScaleValid = *retargetedLhsScale;
        rhsValid = *retargetedRhs;
        rhsScaleValid = *retargetedRhsScale;

        SmallVector<Value> lhsCoordinates(lhsLoad.getCoordinates());
        lhsCoordinates[*lhsRowCoordinate] = rows;
        lhsCoordinates[*lhsBlockCoordinate] = blocks;
        SmallVector<Value> lhsScaleCoordinates(lhsScaleLoad.getCoordinates());
        lhsScaleCoordinates[*lhsScaleRowCoordinate] = rows;
        lhsScaleCoordinates[*lhsScaleBlockCoordinate] = blocks;
        SmallVector<Value> rhsCoordinates(rhsLoad.getCoordinates());
        rhsCoordinates[*rhsBlockCoordinate] = blocks;
        rhsCoordinates[*rhsColumnCoordinate] = columns;
        SmallVector<Value> rhsScaleCoordinates(rhsScaleLoad.getCoordinates());
        rhsScaleCoordinates[*rhsScaleBlockCoordinate] = blocks;
        IRMapping rhsScaleReplay;
        rhsScaleReplay.map(rhsScaleColumnRange->getResult(), columns);
        FailureOr<Value> rhsScaleColumn = materializeSourceRanges(
            nested, nestedLocation, kernel,
            rhsScaleLoad.getCoordinates()[*rhsScaleColumnCoordinate],
            sourceAxisIdentity(*columnMap),
            unitN, *rhsScaleColumnRange, columns,
            rhsScaleReplay, contract.getOperation());
        if (failed(rhsScaleColumn)) {
          loopBodyFailed = true;
          return;
        }
        rhsScaleCoordinates[*rhsScaleColumnCoordinate] = *rhsScaleColumn;
        FailureOr<Value> lhsFill = retargetFill(
            nested, nestedLocation, lhsLoad.getFill(), blockedLhsType);
        FailureOr<Value> lhsScaleFill = retargetFill(
            nested, nestedLocation, lhsScaleLoad.getFill(),
            blockedLhsScaleType);
        FailureOr<Value> rhsFill = retargetFill(
            nested, nestedLocation, rhsLoad.getFill(), blockedRhsType);
        FailureOr<Value> rhsScaleFill = retargetFill(
            nested, nestedLocation, rhsScaleLoad.getFill(),
            blockedRhsScaleType);
        if (failed(lhsFill) || failed(lhsScaleFill) || failed(rhsFill) ||
            failed(rhsScaleFill)) {
          loopBodyFailed = true;
          return;
        }
        Value lhs = nested.create<LoadOp>(
            nestedLocation, blockedLhsType, lhsLoad.getResource(),
            lhsCoordinates, lhsValid, *lhsFill, lhsLoad.getSourceAxes());
        Value lhsScale = nested.create<LoadOp>(
            nestedLocation, blockedLhsScaleType, lhsScaleLoad.getResource(),
            lhsScaleCoordinates, lhsScaleValid, *lhsScaleFill,
            lhsScaleLoad.getSourceAxes());
        Value rhs = nested.create<LoadOp>(
            nestedLocation, blockedRhsType, rhsLoad.getResource(),
            rhsCoordinates, rhsValid, *rhsFill, rhsLoad.getSourceAxes());
        Value rhsScale = nested.create<LoadOp>(
            nestedLocation, blockedRhsScaleType, rhsScaleLoad.getResource(),
            rhsScaleCoordinates, rhsScaleValid, *rhsScaleFill,
            rhsScaleLoad.getSourceAxes());
        auto product = nested.create<ScaledContractOp>(
            nestedLocation, blockedResultType, lhs, lhsScale, rhs, rhsScale,
            carries.front(), ArrayRef<int64_t>{1, 2},
            ArrayRef<int64_t>{0, 1}, ArrayRef<int64_t>{},
            ArrayRef<int64_t>{}, contract.getLhsFormat(),
            contract.getRhsFormat(), contract.getLhsGroupSize(),
            contract.getRhsGroupSize());
        if (Attribute origin = contract->getAttr(originAttr))
          product->setAttr(originAttr, origin);
        nested.create<scf::YieldOp>(nestedLocation, product.getResult());
      });
  if (loopBodyFailed) {
    loop.erase();
    return reject("blocked loop body could not be materialized");
  }

  Value outputRows = broadcastAxis(builder, location, outputPredicateType, rowValid, 0);
  Value outputColumns =
      broadcastAxis(builder, location, outputPredicateType, columnValid, 1);
  Value outputValid = binary(builder, location, outputPredicateType, outputRows,
                             outputColumns, BinaryOperator::LogicalAnd);
  ContractionEpilogueMaterialization epilogueValues(
      kernel, *epilogue, blockedResultType, rows, columns,
      contract.getAccumulator());
  for (StoreOp store : epilogue->stores) {
    OpBuilder::InsertionGuard storeInsertion(builder);
    builder.setInsertionPoint(store);
    auto output = epilogueValues.storeValue(
        builder, store, loop.getResult(0), store.getOperation());
    if (failed(output))
      return reject("pointwise epilogue could not be replayed");
    SmallVector<Value> coordinates(store.getCoordinates());
    FailureOr<unsigned> storeRow = queryCoordinatePosition(
        store.getCoordinates(),
        sourceAxisIdentity(*rowMap));
    FailureOr<unsigned> storeColumn = queryCoordinatePosition(
        store.getCoordinates(),
        sourceAxisIdentity(*columnMap));
    if (failed(storeRow) || failed(storeColumn))
      return store.emitOpError(
          "blocked scaled-contract output lost source coordinates");
    coordinates[*storeRow] = rows;
    coordinates[*storeColumn] = columns;
    // Strip the original tail against its original ranges, as for the input
    // accesses above. Replaying only one coordinate first would leave the mask
    // and its tail authority describing different traversals. Any scalar
    // residual already dominates this original store insertion point.
    FailureOr<Value> valid = materializeRetargetedValidity(
        builder, location, store.getValid(), outputTailRanges,
        outputValid, outputPredicateType);
    if (failed(valid))
      return reject("result store residual validity could not be retargeted");
    auto replacement = builder.create<StoreOp>(
        location, store.getResource(), coordinates, *output, *valid,
        store.getSourceAxes());
    if (Attribute origin = store->getAttr(originAttr))
      replacement->setAttr(originAttr, origin);
  }

  eraseContractionEpilogue(*epilogue);
  return success();
}

} // namespace intent::gpu::contraction
