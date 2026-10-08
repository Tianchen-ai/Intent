#include "PhysicalProgramDetail.h"
#include "ScalarExpressions.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/IR/ProgramInterface.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/STLExtras.h"
#include <functional>


using namespace mlir;

namespace intent::gpu {
using namespace detail;

namespace {

bool isExclusiveProgramRange(MakeRangeOp range, func::FuncOp kernel) {
  if (!range || integerConstant(range.getLogicalStart()) != 0 ||
      integerConstant(range.getStep()) != 1)
    return false;
  Value extent = stripScalarIdentity(range.getExtent());
  if (auto expression = extent.getDefiningOp<PhysicalExprOp>();
      expression && expression.getExpression().getKind() ==
          PhysicalExprKind::Parameter) {
    auto parameter = queryParameterBySymbol(kernel, expression.getExpression().getParameterReference().getName());
    if (failed(parameter))
      return false;
  }
  bool positive = integerConstant(extent).value_or(0) > 0;
  if (auto parameter = queryParameter(extent))
    positive = llvm::all_of(parameter.getCandidates().asArrayRef(),
                           [](int64_t value) { return value > 0; });
  auto multiply = stripScalarIdentity(range.getStart()).getDefiningOp<BinaryOp>();
  if (!positive || !multiply ||
      multiply.getOperatorKind() != BinaryOperator::Multiply)
    return false;
  Value coordinate;
  if (sameScalarExpression(multiply.getLhs(), extent))
    coordinate = stripScalarIdentity(multiply.getRhs());
  else if (sameScalarExpression(multiply.getRhs(), extent))
    coordinate = stripScalarIdentity(multiply.getLhs());
  auto argument = dyn_cast_or_null<BlockArgument>(coordinate);
  auto group = argument ? dyn_cast<ExecutionGroupOp>(argument.getOwner()->getParentOp())
                        : ExecutionGroupOp();
  auto decoded = coordinate ? queryDecodedCoordinate(coordinate) : std::nullopt;
  if (!decoded || decoded->extents.size() != 1)
    return false;
  auto program = decoded->linear.getDefiningOp<ProgramIdOp>();
  auto space = kernel->getAttrOfType<ArrayAttr>(programSpaceAttr);
  auto launchExtent = group ? cast<PhysicalExprAttr>(group.getLaunchExtents()[0])
                           : queryLaunchExpression(decoded->extents[0]);
  if (!program || program.getAxis() != 0 || !space || space.empty() ||
      space[0] != launchExtent)
    return false;
  for (Attribute attribute : space.getValue().drop_front()) {
    auto expression = cast<PhysicalExprAttr>(attribute);
    if (expression.getKind() != PhysicalExprKind::Constant ||
        expression.getValue() != 1)
      return false;
  }
  auto launch = cast<PhysicalExprAttr>(space[0]);
  return launch.getKind() == PhysicalExprKind::CeilDiv &&
         launch.getOperands().size() == 2 &&
         launch.getOperands()[0] == queryLaunchExpression(range.getLogicalStop()) &&
         launch.getOperands()[1] == queryLaunchExpression(extent);
}

Value stripRangeProjection(Value value) {
  Value original = value;
  SmallVector<Operation *> projections;
  while (value) {
    if (auto broadcast = value.getDefiningOp<BroadcastOp>()) {
      projections.push_back(broadcast);
      value = broadcast.getValue();
      continue;
    }
    auto reshape = value.getDefiningOp<ReshapeOp>();
    if (!reshape)
      break;
    auto relations = queryFragmentOperandRelations(reshape.getOperation());
    if (failed(relations) || llvm::any_of(
            relations->front().groups, [](const FragmentAxisGroup &group) {
              return group.sourceAxes.size() > 1 || group.resultAxes.size() > 1;
            }))
      break;
    projections.push_back(reshape);
    value = reshape.getValue();
  }
  auto range = value.getDefiningOp<MakeRangeOp>();
  if (!range)
    return original;
  unsigned axis = 0;
  for (Operation *operation : llvm::reverse(projections)) {
    auto input = cast<FragmentType>(operation->getOperand(0).getType());
    auto output = cast<FragmentType>(operation->getResult(0).getType());
    std::optional<unsigned> projected;
    auto relations = queryFragmentOperandRelations(operation);
    if (failed(relations))
      return original;
    for (const FragmentAxisGroup &group : relations->front().groups)
      if (group.sourceAxes.size() == 1 && group.resultAxes.size() == 1 &&
          group.sourceAxes.front() == axis) {
        if (projected)
          return original;
        projected = group.resultAxes.front();
      }
    if (!projected || input.getShape()[axis] != output.getShape()[*projected])
      return original;
    axis = *projected;
  }
  return value;
}

bool isPrivateWorkspaceProgramIndex(Value coordinate, Value buffer,
                                    Operation *access, unsigned axis) {
  auto program = coordinate.getDefiningOp<ProgramIdOp>();
  auto kernel = access->getParentOfType<func::FuncOp>();
  auto space = kernel->getAttrOfType<ArrayAttr>(programSpaceAttr);
  if (!program || !space || axis >= space.size() || program.getAxis() != axis)
    return false;
  DominanceInfo dominance(kernel);
  return llvm::any_of(buffer.getUsers(), [&](Operation *user) {
    auto assumption = dyn_cast<AssumeInBoundsOp>(user);
    return assumption && assumption.getResource() == buffer &&
           assumption.getIndex() == coordinate &&
           assumption.getAxis() == axis &&
           dominance.properlyDominates(assumption, access);
  });
}

// An execution group's decoded coordinates identify its physical program.
// A compact workspace may omit coordinates fixed by the actual access guard:
// the remaining coordinates still distinguish every program that can use it.
bool guardedProgramSlices(ArrayRef<AccessOpInterface> accesses,
                          func::FuncOp kernel) {
  auto space = kernel->getAttrOfType<ArrayAttr>(programSpaceAttr);
  if (!space || space.size() != 1) return false;
  std::optional<DecodedCoordinate> reference;
  SmallVector<std::optional<int64_t>> referenceAxes, referenceFixed;
  auto sameDecode = [](const DecodedCoordinate &lhs,
                       const DecodedCoordinate &rhs) {
    return sameScalarExpression(lhs.linear, rhs.linear) &&
        lhs.extents.size() == rhs.extents.size() &&
        llvm::all_of(llvm::zip(lhs.extents, rhs.extents), [](auto pair) {
          return sameScalarExpression(std::get<0>(pair), std::get<1>(pair));
        });
  };
  for (AccessOpInterface access : accesses) {
    SmallVector<std::pair<DecodedCoordinate, int64_t>> coordinates, fixed;
    for (auto [value, axis] : llvm::zip(access.getAccessCoordinates(),
                                       access.getAccessSourceAxes()))
      if (auto decoded = queryDecodedCoordinate(stripScalarIdentity(value)))
        coordinates.emplace_back(*decoded, axis);
    std::function<void(Value, bool)> condition = [&](Value value, bool truth) {
      if (auto binary = value.getDefiningOp<BinaryOp>(); binary &&
          ((truth && binary.getOperatorKind() == BinaryOperator::LogicalAnd) ||
           (!truth && binary.getOperatorKind() == BinaryOperator::LogicalOr))) {
        condition(binary.getLhs(), truth);
        condition(binary.getRhs(), truth);
        return;
      }
      auto compare = value.getDefiningOp<CompareOp>();
      if (!compare || (truth ? compare.getPredicate() != ComparePredicate::Eq
                             : compare.getPredicate() != ComparePredicate::Ne))
        return;
      for (auto [coordinate, scalar] : {
               std::pair<Value, Value>{compare.getLhs(), compare.getRhs()},
               {compare.getRhs(), compare.getLhs()}})
        if (auto decoded = queryDecodedCoordinate(stripScalarIdentity(coordinate)))
          if (auto constant = integerConstant(scalar))
            fixed.emplace_back(*decoded, *constant);
    };
    for (Operation *current = access.getOperation(); current && current != kernel;
         current = current->getParentOp())
      if (auto branch = dyn_cast_or_null<scf::IfOp>(current->getParentOp()))
        condition(branch.getCondition(),
                  current->getParentRegion() == &branch.getThenRegion());
    if (coordinates.empty() && fixed.empty()) return false;
    const DecodedCoordinate &decoded =
        coordinates.empty() ? fixed.front().first : coordinates.front().first;
    auto program = stripScalarIdentity(decoded.linear).getDefiningOp<ProgramIdOp>();
    if (!program || program.getAxis() != 0 ||
        llvm::any_of(decoded.extents, [](Value extent) {
          return !queryLaunchExpression(extent);
        }) || (reference && !sameDecode(*reference, decoded)))
      return false;
    PhysicalExprAttr capacity;
    for (Value extent : decoded.extents) {
      auto expression = queryLaunchExpression(extent);
      capacity = !capacity ? expression : PhysicalExprAttr::get(
          kernel.getContext(), PhysicalExprKind::Multiply, 0,
          StringAttr::get(kernel.getContext(), ""),
          ArrayAttr::get(kernel.getContext(), {capacity, expression}));
    }
    // Decoding is modulo the Cartesian capacity. Its complete tuple is an
    // injective program identity only within this exact launch domain.
    if (capacity != space[0]) return false;
    SmallVector<std::optional<int64_t>> axes(decoded.extents.size());
    SmallVector<std::optional<int64_t>> values(decoded.extents.size());
    for (auto [selected, axis] : coordinates) {
      if (!sameDecode(decoded, selected)) return false;
      if (!axes[selected.axis]) axes[selected.axis] = axis;
    }
    for (auto [selected, value] : fixed) {
      if (!sameDecode(decoded, selected)) continue;
      if (values[selected.axis] && *values[selected.axis] != value) return false;
      values[selected.axis] = value;
    }
    for (unsigned axis = 0; axis < axes.size(); ++axis) {
      if (axes[axis]) values[axis].reset();
      else if (!values[axis]) return false;
    }
    if (reference && (axes != referenceAxes || values != referenceFixed))
      return false;
    reference = decoded;
    referenceAxes = std::move(axes);
    referenceFixed = std::move(values);
  }
  return reference.has_value();
}

} // namespace

namespace detail {

bool isCoordinateReplayNode(Operation *operation) {
  return isa<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp,
             BroadcastOp, SplatOp, ReshapeOp, TransposeOp, JoinOp, DimOp,
             MakeRecordOp, ExtractOp, RandomBitsOp>(operation);
}

bool isValueReplayNode(Operation *operation) {
  return isCoordinateReplayNode(operation) ||
         isa<ContractOp, ScaledContractOp, SparseContractOp, ReduceOp, ScanOp>(
             operation);
}

bool isAccessNode(Operation *operation) {
  return isa<LoadOp, GatherOp>(operation);
}

void appendUnique(SmallVectorImpl<MakeRangeOp> &destination, MakeRangeOp range) {
  if (!llvm::is_contained(destination, range))
    destination.push_back(range);
}

void appendUnique(SmallVectorImpl<Operation *> &destination,
                  Operation *operation) {
  if (operation && !llvm::is_contained(destination, operation))
    destination.push_back(operation);
}

OpOperand *singleControlInput(Value target, ControlFlowEdgeKind kind,
                              Region *sourceRegion) {
  auto incoming = queryControlFlowIncoming(target);
  if (!incoming.complete) return nullptr;
  OpOperand *selected = nullptr;
  for (const ControlFlowEdge &edge : incoming.edges) {
    if (edge.kind != kind || edge.sourceRegion != sourceRegion) continue;
    if (!edge.operand || (selected && selected != edge.operand)) return nullptr;
    selected = edge.operand;
  }
  return selected;
}

} // namespace detail

bool isPhysicalReplayNode(Operation *operation, PhysicalReplayScope scope,
                          bool allowAccesses) {
  if (!operation || operation->getNumResults() == 0)
    return false;
  if (isAccessNode(operation))
    return allowAccesses && operation->getNumRegions() == 0;
  if (operation->getNumRegions() != 0 && !isa<ReduceOp, ScanOp>(operation))
    return false;
  return scope == PhysicalReplayScope::Coordinate
             ? isCoordinateReplayNode(operation)
             : isValueReplayNode(operation);
}

PhysicalProgramAnalysis::PhysicalProgramAnalysis(func::FuncOp kernel)
    : kernel(kernel) {}

bool PhysicalProgramAnalysis::carriesSource(Type type,
                                            PhysicalSourceAxis source) const {
  if (auto fragment = dyn_cast<FragmentType>(type))
    return llvm::any_of(fragment.getAxisMaps(), [&](Attribute attribute) {
      auto mapping = cast<AxisMapAttr>(attribute);
      return mapping.getSourceId() == source.sourceId &&
             mapping.getSourceAxis() == source.sourceAxis &&
             mapping.getDerived() == source.derived;
    });
  if (auto record = dyn_cast<RecordType>(type))
    return llvm::any_of(record.getFieldTypes(), [&](Attribute field) {
      return carriesSource(cast<TypeAttr>(field).getValue(), source);
    });
  return false;
}

SmallVector<Value, 2> PhysicalProgramAnalysis::structuredSourcesForArgument(
    BlockArgument argument) const {
  SmallVector<Value, 2> sources;
  Block *block = argument.getOwner();
  Operation *owner = block ? block->getParentOp() : nullptr;
  if (auto structured = dyn_cast_or_null<StructuredOpInterface>(owner)) {
    bool fold = structured.getStructuredKind() == StructuredOpKind::RegionFold;
    bool scan = structured.getStructuredKind() == StructuredOpKind::RegionScan;
    if (!fold && !scan) return sources;
    for (const auto &relation :
         structured.getRegionArgumentRelations(*block->getParent())) {
      if (relation.to != argument) continue;
      if (block->getParent() == structured.getSummarizeRegion() &&
          (relation.kind == StructuredRelationKind::SourceSlice ||
           (fold && relation.kind == StructuredRelationKind::Capture)))
        sources.push_back(relation.from);
    }
    if (fold && block->getParent() == &structured.getCombine()) {
      for (auto [identity, lhs, rhs, summary] : llvm::zip_equal(
               structured.getIdentities(), structured.getCombineLhs(),
               structured.getCombineRhs(), structured.getSummarizeYields())) {
        if (argument != lhs && argument != rhs) continue;
        sources.push_back(identity);
        sources.push_back(summary);
      }
    }
    return sources;
  }
  auto incoming = queryControlFlowIncoming(argument);
  if (!incoming.complete)
    return sources;
  for (const ControlFlowEdge &edge : incoming.edges)
    if (edge.operand && !llvm::is_contained(sources, edge.operand->get()))
      sources.push_back(edge.operand->get());
  return sources;
}

bool PhysicalProgramAnalysis::isProgramOwnedRange(MakeRangeOp range) const {
  return isExclusiveProgramRange(range, kernel);
}

bool PhysicalProgramAnalysis::hasDisjointWorkspaceSlices(Value buffer) const {
  ArrayAttr shape;
  if (auto type = dyn_cast<BufferType>(buffer.getType()))
    shape = type.getShape();
  else if (auto view = dyn_cast<ViewType>(buffer.getType());
           view && isInvocationWorkspace(buffer))
    shape = view.getLayout().getExtents();
  else
    return false;
  SmallVector<AccessOpInterface> accesses;
  for (OpOperand &use : buffer.getUses()) {
    Operation *user = use.getOwner();
    if (isa<DimOp, AssumeInBoundsOp>(user))
      continue;
    auto access = dyn_cast<AccessOpInterface>(user);
    if (!access || !access.isMemoryAccess() ||
        &use != &access.getAccessResourceOperand())
      return false;
    accesses.push_back(access);
  }
  if (accesses.empty())
    return true;
  auto space = kernel->getAttrOfType<ArrayAttr>(programSpaceAttr);
  // Every access must identify the whole program, including all grid axes.
  bool programPrefix = space && !space.empty() &&
                       shape.size() >= space.size();
  for (AccessOpInterface access : accesses) {
    if (!programPrefix)
      break;
    ValueRange coordinates = access.getAccessCoordinates();
    ArrayRef<int64_t> sourceAxes = access.getAccessSourceAxes();
    for (unsigned axis = 0; axis < space.size(); ++axis) {
      auto position = llvm::find(sourceAxes, axis);
      if (position == sourceAxes.end() ||
          !isPrivateWorkspaceProgramIndex(
              coordinates[position - sourceAxes.begin()], buffer, access, axis)) {
        programPrefix = false;
        break;
      }
    }
  }
  if (programPrefix)
    return true;
  if (guardedProgramSlices(accesses, kernel))
    return true;
  for (unsigned axis = 0; axis < shape.size(); ++axis) {
    MakeRangeOp owner;
    bool consistent = true;
    for (AccessOpInterface access : accesses) {
      ValueRange coordinates = access.getAccessCoordinates();
      ArrayRef<int64_t> sourceAxes = access.getAccessSourceAxes();
      auto position = llvm::find(sourceAxes, axis);
      auto range = position == sourceAxes.end() ? MakeRangeOp() :
          stripRangeProjection(coordinates[position - sourceAxes.begin()])
              .getDefiningOp<MakeRangeOp>();
      // Program-owned intervals establish disjointness independently of the
      // backing allocation's candidate envelope. Access bounds are checked
      // separately against that allocation's actual shape.
      if (!range || !isExclusiveProgramRange(range, kernel) ||
          (owner && (!sameLogicalRange(owner, range) ||
                     !sameScalarExpression(owner.getStart(), range.getStart()) ||
                     !sameScalarExpression(owner.getExtent(), range.getExtent())))) {
        consistent = false;
        break;
      }
      owner = range;
    }
    if (consistent && owner)
      return true;
  }
  return false;
}

} // namespace intent::gpu
