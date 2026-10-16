#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Analysis/IndexRelations.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/PhysicalParameters.h"

#include "Intent/Dialect/GPU/IR/Program.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace intent::gpu {
namespace {

PhysicalExprAttr parameterExpression(ParameterRefAttr reference) {
  return PhysicalExprAttr::get(
      reference.getContext(), PhysicalExprKind::Parameter, 0,
      reference, ArrayAttr::get(reference.getContext(), {}));
}

void preserveBoundedTileOrigins(func::FuncOp kernel) {
  kernel.walk([&](MakeRangeOp range) {
    if (range->hasAttr(sourceSubregionAttr) || !isUnitStepRange(range))
      return;
    FailureOr<int64_t> dimension = queryRangeDimension(range);
    IndexRelations relations;
    if (relations.constant(range.getLogicalStart()) != 0 || failed(dimension))
      return;
    if (relations.mappedTileOrigin(range.getStart(), range.getExtent(), *dimension))
      range->setAttr(programBoundedOriginAttr, UnitAttr::get(kernel.getContext()));
  });
}

// Query a loop bound's direction in one independently mapped coordinate. This
// is a scheduling estimate, not a replacement for integer legality or bounds.
class CoordinateOrder {
public:
  explicit CoordinateOrder(Value coordinate) : coordinate(coordinate) {}

  std::optional<int> get(Value value) {
    if (value == coordinate) return 1;
    if (auto found = memo.find(value); found != memo.end()) return found->second;
    auto result = infer(value);
    memo[value] = result;
    return result;
  }

private:
  std::optional<int> infer(Value value) {
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      if (argument.getOwner() == coordinate.getParentBlock() ||
          isa<func::FuncOp>(argument.getOwner()->getParentOp())) return 0;
      if (auto loop = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
          loop && argument == loop.getInductionVar() &&
          get(loop.getLowerBound()) == 0 &&
          get(loop.getUpperBound()) == 0 &&
          get(loop.getStep()) == 0) return 0;
      return std::nullopt;
    }
    Operation *producer = value.getDefiningOp();
    if (!producer) return std::nullopt;
    auto binary = dyn_cast<BinaryOp>(producer);
    if (!binary) {
      if (producer->getNumRegions()) return std::nullopt;
      for (Value operand : producer->getOperands()) {
        auto order = get(operand);
        if (!order || *order) return std::nullopt;
      }
      return 0;
    }
    Value lhs = binary.getLhs(), rhs = binary.getRhs();
    auto kind = binary.getOperatorKind();
    bool minimum = kind == BinaryOperator::Minimum || kind == BinaryOperator::MinimumNum;
    bool maximum = kind == BinaryOperator::Maximum || kind == BinaryOperator::MaximumNum;
    if (minimum || maximum) {
      if (relations.atMost(lhs, rhs)) return get(minimum ? lhs : rhs);
      if (relations.atMost(rhs, lhs)) return get(minimum ? rhs : lhs);
    }
    auto a = get(lhs), b = get(rhs);
    if (!a || !b) return std::nullopt;
    if (!*a && !*b) return 0;
    auto merge = [](int a, int b) -> std::optional<int> {
      if (!a) return b;
      if (!b || a == b) return a;
      return std::nullopt;
    };
    if (minimum || maximum || kind == BinaryOperator::Add) return merge(*a, *b);
    if (kind == BinaryOperator::Subtract) return merge(*a, -*b);
    if (kind == BinaryOperator::Multiply) {
      if (!*b && relations.positive(rhs)) return a;
      if (!*a && relations.positive(lhs)) return b;
    }
    if (kind == BinaryOperator::FloorDivide && !*b && relations.positive(rhs)) return a;
    return std::nullopt;
  }

  Value coordinate;
  IndexRelations relations;
  DenseMap<Value, std::optional<int>> memo;
};

void balanceVariableMatrixTraversals(func::FuncOp kernel, ExecutionGroupOp mapping) {
  ArrayRef<int64_t> roles = mapping.getCoordinateRoles();
  std::optional<unsigned> rowAxis;
  for (auto [axis, role] : llvm::enumerate(roles)) {
    if (role == static_cast<int64_t>(CoordinateRole::ContractionM) ||
        role == static_cast<int64_t>(CoordinateRole::PointwiseOwnership)) {
      if (rowAxis) return;
      rowAxis = axis;
    } else if (role != static_cast<int64_t>(CoordinateRole::Workset)) {
      return;
    }
  }
  if (!rowAxis || roles.size() < 2) return;
  auto offset = mapping.getSegmentOffset();
  auto space = kernel->getAttrOfType<ArrayAttr>(programSpaceAttr);
  if (!offset || offset.getKind() != PhysicalExprKind::Constant || offset.getValue() ||
      !space || space.size() != 1 || mapping.getSegmentLength() != space[0]) return;
  Value row = mapping.getCoordinates()[*rowAxis];
  CoordinateOrder order(row);
  int direction = 0;
  bool conflicting = false;
  mapping.walk([&](scf::ForOp loop) {
    // A row-independent enclosing traversal repeats the same work curve.
    // Row-dependent nests require a separate cost model. Its bounds need not
    // be launch expressions: they do not become part of the launch grid.
    for (auto parent = loop->getParentOfType<scf::ForOp>(); parent;
         parent = parent->getParentOfType<scf::ForOp>())
      if (order.get(parent.getLowerBound()) != 0 ||
          order.get(parent.getUpperBound()) != 0 ||
          order.get(parent.getStep()) != 0) return;
    bool matrix = false;
    loop.walk([&](ContractOp contract) {
      matrix |= contract->getParentOfType<scf::ForOp>() == loop;
    });
    if (!matrix) return;
    auto lower = order.get(loop.getLowerBound()), upper = order.get(loop.getUpperBound());
    auto step = order.get(loop.getStep());
    if (!lower || !upper || !step || *step || (*lower && *upper)) return;
    int current = *upper - *lower;
    if (!current) return;
    if (direction && direction != current) conflicting = true;
    direction = current;
  });
  if (!direction || conflicting) return;
  auto capabilities = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  if (!capabilities || capabilities.getComputeUnits() <= 0) return;

  // Keep the same rectangular set of independent programs. Visit equal-cost
  // rows across batches together, with the longest traversals first, instead
  // of restarting the short-to-long sequence for every independent matrix.
  preserveBoundedTileOrigins(kernel);
  OpBuilder builder = OpBuilder::atBlockBegin(&mapping.getBody().front());
  Location loc = mapping.getLoc();
  SmallVector<unsigned> permutation{*rowAxis};
  for (unsigned axis = 0; axis < roles.size(); ++axis)
    if (axis != *rowAxis) permutation.push_back(axis);
  SmallVector<Value> extents, replacements(roles.size());
  SmallVector<Attribute> launchExtents;
  SmallVector<int64_t> reorderedRoles;
  for (auto [position, original] : llvm::enumerate(permutation)) {
    extents.push_back(mapping.getExtents()[original]);
    launchExtents.push_back(mapping.getLaunchExtents()[original]);
    reorderedRoles.push_back(roles[original]);
    replacements[original] = mapping.getCoordinates()[position];
  }
  SmallVector<Operation *> coordinateOps;
  if (direction > 0) {
    Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
    Value last = builder.create<BinaryOp>(loc, builder.getIndexType(), extents.front(),
                                         one, BinaryOperator::Subtract);
    auto reversed = builder.create<BinaryOp>(loc, builder.getIndexType(), last,
                                              replacements[*rowAxis], BinaryOperator::Subtract);
    coordinateOps.push_back(reversed);
    replacements[*rowAxis] = reversed;
  }
  // With no more programs than compute units there is no queued wave to
  // balance. Preserve the original locality in that case. Launch extents and
  // configuration values make this choice uniform; providers can specialize
  // it without duplicating the compute body.
  Value programs = builder.create<PhysicalExprOp>(
      loc, builder.getIndexType(), mapping.getSegmentLength());
  Value units = builder.create<arith::ConstantIndexOp>(
      loc, capabilities.getComputeUnits());
  Value queued = builder.create<CompareOp>(loc, builder.getI1Type(), programs,
                                         units, ComparePredicate::Gt);
  auto original = builder.create<DelinearizeOp>(
      loc, SmallVector<Type>(roles.size(), builder.getIndexType()),
      mapping.getLinear(), mapping.getExtents());
  for (auto [replacement, coordinate] :
       llvm::zip(replacements, original.getCoordinates())) {
    auto selected = builder.create<SelectOp>(loc, builder.getIndexType(), queued,
                                             replacement, coordinate);
    coordinateOps.push_back(selected);
    replacement = selected;
  }
  SmallVector<std::pair<OpOperand *, Value>> uses;
  for (auto [original, replacement] : llvm::zip(mapping.getCoordinates(), replacements))
    for (OpOperand &use : original.getUses())
      if (!llvm::is_contained(coordinateOps, use.getOwner()))
        uses.emplace_back(&use, replacement);
  for (auto [use, replacement] : uses) use->set(replacement);
  mapping.getExtentsMutable().assign(extents);
  mapping.setLaunchExtentsAttr(builder.getArrayAttr(launchExtents));
  mapping.setCoordinateRolesAttr(builder.getDenseI64ArrayAttr(reorderedRoles));
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
  preserveBoundedTileOrigins(kernel);

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
  balanceVariableMatrixTraversals(kernel, mapping);
  roles = mapping.getCoordinateRolesAttr();
  const int64_t traversalWorker =
      static_cast<int64_t>(CoordinateRole::TraversalWorker);
  SmallVector<unsigned> traversalAxes;
  for (auto [axis, role] : llvm::enumerate(roles.asArrayRef()))
    if (role == traversalWorker)
      traversalAxes.push_back(axis);
  // Independent matrices can reuse resident workers across their batch domain.
  // A single matrix retains its full tile grid: bounding that grid can reduce
  // occupancy without amortizing traversal across independent matrices.
  // Ownership and grouped order remain unchanged; provider configurations
  // determine tile and resident granularity.
  bool batchedMatrix = false;
  for (auto [axis, role] : llvm::enumerate(roles.asArrayRef()))
    if (!mapping.getCoordinates()[axis].use_empty() &&
        (role == static_cast<int64_t>(CoordinateRole::Workset) ||
         role == static_cast<int64_t>(CoordinateRole::TiledWorkset) ||
         role == static_cast<int64_t>(CoordinateRole::PointwiseOwnership)))
      batchedMatrix = true;
  bool matrixM = false, matrixN = false;
  for (Attribute attribute : getParameterDeclarations(kernel)) {
    auto parameter = cast<ParameterAttr>(attribute);
    if (parameter.getCategory() != ParameterCategory::Contraction ||
        parameter.getElementBitWidth() == 0 ||
        parameter.getElementBitWidth() > 16)
      continue;
    matrixM |= parameter.getRole() == ParameterRole::OwnershipM;
    matrixN |= parameter.getRole() == ParameterRole::OwnershipN;
  }
  auto capabilities = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  bool residentMatrix = batchedMatrix && matrixM && matrixN && capabilities &&
                        capabilities.getMatrixUnits();
  if (traversalAxes.empty() && !residentMatrix)
    return success();

  auto program = mapping.getLinear().getDefiningOp<ProgramIdOp>();
  if (!program || program.getAxis() != 0) {
    if (traversalAxes.empty())
      return success();
    return mapping.emitOpError(
        "persistent traversal requires one linear program coordinate");
  }
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
      segmentOffset.getValue() != 0 || programSpace[0] != segmentLength) {
    if (traversalAxes.empty())
      return success();
    return mapping.emitOpError(
        "persistent traversal requires one full-program execution segment");
  }

  for (unsigned axis : traversalAxes) {
    auto parameter =
        mapping.getExtents()[axis].getDefiningOp<ParameterOp>();
    if (!parameter ||
        parameter.getDeclaration().getRole() !=
            ParameterRole::TraversalWorkers)
      return mapping.emitOpError(
          "traversal-worker coordinate lacks its physical parameter");
  }

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
