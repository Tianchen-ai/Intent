#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/PhysicalParameters.h"

#include "Intent/Dialect/GPU/IR/Program.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace intent::gpu {
namespace {

PhysicalExprAttr parameterExpression(ParameterRefAttr reference) {
  return PhysicalExprAttr::get(
      reference.getContext(), PhysicalExprKind::Parameter, 0,
      reference, ArrayAttr::get(reference.getContext(), {}));
}

void preserveBoundedTileOrigins(func::FuncOp kernel, ExecutionGroupOp mapping) {
  kernel.walk([&](MakeRangeOp range) {
    if (range->hasAttr(sourceSubregionAttr) || !isUnitStepRange(range))
      return;
    auto start = range.getLogicalStart().getDefiningOp<arith::ConstantOp>();
    auto startValue =
        start ? dyn_cast<IntegerAttr>(start.getValue()) : IntegerAttr();
    auto block = range.getExtent().getDefiningOp<ParameterOp>();
    FailureOr<int64_t> dimension = queryRangeDimension(range);
    if (!startValue || startValue.getInt() != 0 || !block || failed(dimension) ||
        llvm::any_of(block.getDeclaration().getCandidates().asArrayRef(),
                     [](int64_t value) { return value <= 0; }))
      return;
    for (auto [axis, coordinate] : llvm::enumerate(mapping.getCoordinates())) {
      Attribute launchExtent = mapping.getLaunchExtents()[axis];
      FailureOr<uint64_t> mappedDimension = blockedDimension(launchExtent);
      auto extent = dyn_cast<PhysicalExprAttr>(launchExtent);
      if (failed(mappedDimension) ||
          *mappedDimension != static_cast<uint64_t>(*dimension) ||
          !extent || extent.getOperands().size() != 2)
        continue;
      auto divisor = dyn_cast<PhysicalExprAttr>(extent.getOperands()[1]);
      if (!divisor || divisor.getKind() !=
                          PhysicalExprKind::Parameter ||
          divisor.getParameterReference().getName() != block.getDeclaration().getName())
        continue;
      for (Operation *user : coordinate.getUsers()) {
        auto product = dyn_cast<BinaryOp>(user);
        if (!product || product.getOperatorKind() != BinaryOperator::Multiply)
          continue;
        bool scaledCoordinate =
            (product.getLhs() == coordinate &&
             product.getRhs() == range.getExtent()) ||
            (product.getRhs() == coordinate &&
             product.getLhs() == range.getExtent());
        if (scaledCoordinate &&
            samePhysicalScalarExpression(range.getStart(), product.getResult())) {
          range->setAttr(programBoundedOriginAttr,
                         UnitAttr::get(kernel.getContext()));
          break;
        }
      }
    }
  });
}

LogicalResult realizeGroupedContractionMapping(
    ModuleOp module, func::FuncOp kernel, ExecutionGroupOp mapping,
    DenseI64ArrayAttr roles) {
  const int64_t contractionM =
      static_cast<int64_t>(CoordinateRole::ContractionM);
  const int64_t contractionN =
      static_cast<int64_t>(CoordinateRole::ContractionN);
  std::optional<unsigned> rowAxis;
  std::optional<unsigned> columnAxis;
  for (auto [axis, role] : llvm::enumerate(roles.asArrayRef())) {
    if (role == contractionM) {
      if (rowAxis)
        return mapping.emitOpError(
            "grouped contraction mapping requires one M coordinate");
      rowAxis = axis;
    }
    if (role == contractionN) {
      if (columnAxis)
        return mapping.emitOpError(
            "grouped contraction mapping requires one N coordinate");
      columnAxis = axis;
    }
  }
  if (!rowAxis || !columnAxis)
    return success();

  // The permutation preserves the rectangular tile domain, including its last
  // partial tile. Keep exact origin bounds before replacing coordinate uses.
  preserveBoundedTileOrigins(kernel, mapping);

  OpBuilder parameterBuilder(&kernel.getBody().front(),
                             kernel.getBody().front().begin());
  auto group = getOrCreatePhysicalParameter(kernel, "GROUP_SIZE_M",
      ParameterRole::TraversalGroup, ParameterCategory::Execution, 0, {1, 2, 4, 8});
  if (failed(group)) return failure();
  Value groupSize = materializeParameter(parameterBuilder, mapping.getLoc(), *group);

  OpBuilder builder(mapping);
  builder.setInsertionPointToStart(&mapping.getBody().front());
  llvm::SmallPtrSet<Operation *, 16> swizzleOperations;
  auto binary = [&](Value lhs, Value rhs, BinaryOperator kind) {
    auto operation = builder.create<BinaryOp>(mapping.getLoc(),
                                              builder.getIndexType(), lhs, rhs,
                                              kind);
    swizzleOperations.insert(operation.getOperation());
    return operation.getResult();
  };
  Value rowCount = mapping.getExtents()[*rowAxis];
  Value columnCount = mapping.getExtents()[*columnAxis];
  Value row = mapping.getCoordinates()[*rowAxis];
  Value column = mapping.getCoordinates()[*columnAxis];
  Value groupIndex = binary(row, groupSize, BinaryOperator::FloorDivide);
  Value firstRow = binary(groupIndex, groupSize, BinaryOperator::Multiply);
  Value liveRows = binary(rowCount, firstRow, BinaryOperator::Subtract);
  Value activeGroupSize =
      binary(liveRows, groupSize, BinaryOperator::MinimumNum);
  Value rowInGroup = binary(row, groupSize, BinaryOperator::Remainder);
  Value groupOffset = binary(
      binary(rowInGroup, columnCount, BinaryOperator::Multiply), column,
      BinaryOperator::Add);
  Value groupedRow =
      binary(firstRow,
             binary(groupOffset, activeGroupSize, BinaryOperator::Remainder),
             BinaryOperator::Add);
  Value groupedColumn =
      binary(groupOffset, activeGroupSize, BinaryOperator::FloorDivide);

  auto replaceExternalUses = [&](Value source, Value replacement) {
    for (OpOperand &use : llvm::make_early_inc_range(source.getUses()))
      if (!swizzleOperations.contains(use.getOwner()))
        use.set(replacement);
  };
  replaceExternalUses(row, groupedRow);
  replaceExternalUses(column, groupedColumn);
  return success();
}

} // namespace

LogicalResult refineProgramMapping(ModuleOp module) {
  FailureOr<func::FuncOp> physicalKernel = getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  func::FuncOp kernel = *physicalKernel;

  SmallVector<ExecutionGroupOp> mappings;
  kernel.walk([&](ExecutionGroupOp mapping) { mappings.push_back(mapping); });
  if (mappings.size() != 1)
    return success();
  ExecutionGroupOp mapping = mappings.front();
  if (mapping->getBlock() != &kernel.getBody().front())
    return success();

  auto roles =
      mapping.getCoordinateRolesAttr();
  if (!roles || static_cast<size_t>(roles.size()) !=
                    mapping.getCoordinates().size())
    return success();
  if (failed(realizeGroupedContractionMapping(module, kernel, mapping, roles)))
    return failure();
  const int64_t traversalWorker =
      static_cast<int64_t>(CoordinateRole::TraversalWorker);
  SmallVector<unsigned> traversalAxes;
  for (auto [axis, role] : llvm::enumerate(roles.asArrayRef()))
    if (role == traversalWorker)
      traversalAxes.push_back(axis);
  // Independent tiles already have a complete launch grid. Only traversal
  // workers need to revisit logical work through a grid-stride loop.
  if (traversalAxes.empty())
    return success();

  auto program = mapping.getLinear().getDefiningOp<ProgramIdOp>();
  if (!program || program.getAxis() != 0)
    return mapping.emitOpError(
        "persistent traversal requires one linear program coordinate");
  if (!program.getResult().hasOneUse())
    return success();
  for (Operation &operation : kernel.getBody().front()) {
    if (&operation == mapping.getOperation())
      break;
    if (!isMemoryEffectFree(&operation))
      return success();
  }
  auto programSpace = kernel->getAttrOfType<ArrayAttr>(programSpaceAttr);
  auto segmentOffset =
      mapping.getSegmentOffset();
  auto segmentLength =
      mapping.getSegmentLength();
  if (!programSpace || programSpace.size() != 1 || !segmentOffset ||
      !segmentLength ||
      segmentOffset.getKind() !=
          PhysicalExprKind::Constant ||
      segmentOffset.getValue() != 0 || programSpace[0] != segmentLength)
    return mapping.emitOpError(
        "persistent traversal requires one full-program execution segment");

  for (unsigned axis : traversalAxes) {
    auto parameter =
        mapping.getExtents()[axis].getDefiningOp<ParameterOp>();
    if (!parameter ||
        parameter.getDeclaration().getRole() !=
            ParameterRole::TraversalWorkers)
      return mapping.emitOpError(
          "traversal-worker coordinate lacks its physical parameter");
  }

  auto capabilities =
      kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  if (!capabilities || capabilities.getComputeUnits() <= 0)
    return kernel.emitError(
        "persistent traversal requires a positive compute-unit capability");
  int64_t residentCount = capabilities.getComputeUnits();
  OpBuilder parameterBuilder(&kernel.getBody().front(),
                             kernel.getBody().front().begin());
  auto resident = getOrCreatePhysicalParameter(kernel, "RESIDENT_WORKERS",
      ParameterRole::ResidentWorkers, ParameterCategory::Execution, 0, {residentCount});
  if (failed(resident)) return failure();
  auto residentWorkers = materializeParameter(parameterBuilder, mapping.getLoc(), *resident);

  OpBuilder builder(mapping);
  Value totalTasks = builder.create<PhysicalExprOp>(
      mapping.getLoc(), builder.getIndexType(), segmentLength);
  auto loop = builder.create<scf::ForOp>(
      mapping.getLoc(), program.getResult(), totalTasks,
      residentWorkers.getResult());
  mapping->setOperand(0, loop.getInductionVar());
  mapping->moveBefore(loop.getBody()->getTerminator());

  PhysicalExprAttr residentExtent = parameterExpression(*resident);
  auto boundedResidentExtent = PhysicalExprAttr::get(
      module.getContext(), PhysicalExprKind::Minimum, 0,
      StringAttr::get(module.getContext()),
      ArrayAttr::get(module.getContext(), {segmentLength, residentExtent}));
  kernel->setAttr(programSpaceAttr,
                  ArrayAttr::get(module.getContext(), {boundedResidentExtent}));
  kernel->setAttr(gridRankAttr,
                  IntegerAttr::get(IntegerType::get(module.getContext(), 64), 1));
  return success();
}

} // namespace intent::gpu
