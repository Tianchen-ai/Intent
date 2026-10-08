#include "ContractionDetail.h"
#include "Intent/Dialect/GPU/Transforms/Mapping/ExecutionGroups.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/PhysicalParameters.h"
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

static SmallVector<int64_t> remapAxes(ArrayRef<int64_t> axes,
                               ArrayRef<int64_t> permutation);

static Type transposeLoopSchema(Type type);

static void matrixFields(Type type, SmallVectorImpl<FragmentType> &fields);

static Value transposeLoopValue(OpBuilder &builder, Location location, Value value);

static bool supportsMatrixLoopTranspose(scf::ForOp loop);

static void transposeClonedLoop(scf::ForOp loop);

void pruneContractionProgramCoordinates(func::FuncOp kernel) {
  // A reduction can consume a provisional pointwise axis. Once its old value
  // graph is gone, that unused axis must not multiply the output workset.
  SmallVector<ExecutionGroupOp> mappings;
  kernel.walk([&](ExecutionGroupOp mapping) { mappings.push_back(mapping); });
  for (ExecutionGroupOp mapping : mappings) {
    auto roles = mapping.getCoordinateRolesAttr();
    auto space = kernel->getAttrOfType<ArrayAttr>(programSpaceAttr);
    auto offset = mapping.getSegmentOffset();
    auto length = mapping.getSegmentLength();
    auto program = mapping.getLinear().getDefiningOp<ProgramIdOp>();
    if (!roles || roles.size() != mapping.getCoordinates().size() ||
        !llvm::is_contained(roles.asArrayRef(),
                            static_cast<int64_t>(CoordinateRole::ContractionM)) ||
        !llvm::is_contained(roles.asArrayRef(),
                            static_cast<int64_t>(CoordinateRole::ContractionN)) ||
        !program || program.getAxis() != 0 ||
        !program.getResult().hasOneUse() || !space ||
        space.size() != 1 || !offset || !length || space[0] != length ||
        offset.getKind() != PhysicalExprKind::Constant ||
        offset.getValue() != 0)
      continue;
    SmallVector<Type> types;
    SmallVector<Value> extents;
    SmallVector<Attribute> launch;
    SmallVector<int64_t> retainedRoles;
    SmallVector<Value> retainedCoordinates;
    for (auto [axis, coordinate] : llvm::enumerate(mapping.getCoordinates())) {
      if (coordinate.use_empty() &&
          roles[axis] == static_cast<int64_t>(CoordinateRole::PointwiseOwnership))
        continue;
      types.push_back(coordinate.getType());
      extents.push_back(mapping.getExtents()[axis]);
      launch.push_back(mapping.getLaunchExtents()[axis]);
      retainedRoles.push_back(roles[axis]);
      retainedCoordinates.push_back(coordinate);
    }
    if (types.size() == mapping.getCoordinates().size())
      continue;
    OpBuilder builder(mapping);
    length = cast<PhysicalExprAttr>(launch.front());
    for (Attribute extent : llvm::drop_begin(launch))
      length = binaryExpression(kernel.getContext(), PhysicalExprKind::Multiply,
                                length, cast<PhysicalExprAttr>(extent));
    auto compact = rebuildExecutionGroup(
        builder, mapping, extents, builder.getArrayAttr(launch),
        builder.getDenseI64ArrayAttr(retainedRoles), length);
    kernel->setAttr(programSpaceAttr, builder.getArrayAttr({length}));
    for (auto [old, current] :
         llvm::zip(retainedCoordinates, compact.getCoordinates()))
      old.replaceAllUsesWith(current);
    mapping.erase();
  }
}

FragmentType transposeLastTwo(FragmentType source) {
  const unsigned rank = source.getShape().size();
  SmallVector<Attribute> shape(source.getShape().begin(), source.getShape().end());
  std::swap(shape[rank - 2], shape[rank - 1]);
  SmallVector<Attribute> mappings;
  mappings.reserve(rank);
  for (unsigned resultAxis = 0; resultAxis < rank; ++resultAxis) {
    unsigned sourceAxis = resultAxis;
    if (resultAxis == rank - 2)
      sourceAxis = rank - 1;
    else if (resultAxis == rank - 1)
      sourceAxis = rank - 2;
    auto mapping = cast<AxisMapAttr>(source.getAxisMaps()[sourceAxis]);
    mappings.push_back(AxisMapAttr::get(
        source.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
        mapping.getDimensionId(), resultAxis, mapping.getDerived()));
  }
  return FragmentType::get(
      source.getContext(), source.getElementType(),
      ArrayAttr::get(source.getContext(), shape),
      ArrayAttr::get(source.getContext(), mappings), source.getValidity(),
      source.getOwner());
}

static SmallVector<int64_t> remapAxes(ArrayRef<int64_t> axes,
                               ArrayRef<int64_t> permutation) {
  SmallVector<int64_t> inverse(permutation.size());
  for (auto [resultAxis, sourceAxis] : llvm::enumerate(permutation))
    inverse[static_cast<unsigned>(sourceAxis)] = resultAxis;
  SmallVector<int64_t> result;
  result.reserve(axes.size());
  for (int64_t axis : axes)
    result.push_back(inverse[static_cast<unsigned>(axis)]);
  return result;
}

LogicalResult normalizeMatrixContractForms(func::FuncOp kernel) {
  SmallVector<ContractOp> contracts;
  kernel.walk([&](ContractOp contract) { contracts.push_back(contract); });
  for (ContractOp contract : contracts) {
    std::string reason;
    auto axes = queryContractionAxes(contract, &reason);
    if (!axes)
      return contract.emitOpError("invalid contraction axis schema: ") << reason;
    if (axes->reduction.size() != 1)
      continue;
    // A rank-lifted outer ownership axis can sit beside a logical unit free
    // axis introduced by reshape (for example [H, 1, K]).  The unit carries no
    // independent matrix work.  Squeeze it, perform the canonical matrix
    // contraction, and restore the logical result shape afterwards.  This
    // keeps the outer axis as M/N rather than degrading it into a batch of
    // one-row contractions.
    {
      auto retainedAxes = [](ArrayRef<std::optional<unsigned>> resultPositions) {
        SmallVector<unsigned> result;
        for (auto [axis, position] : llvm::enumerate(resultPositions))
          if (position)
            result.push_back(axis);
        return result;
      };
      PhysicalProgramAnalysis analysis(kernel);
      auto unitAxis = [&](Value value, unsigned axis) {
        auto type = cast<FragmentType>(value.getType());
        auto extent = dyn_cast<PhysicalExprAttr>(type.getShape()[axis]);
        if (!extent || extent.getKind() !=
                           PhysicalExprKind::Constant ||
            extent.getValue() != 1)
          return false;
        auto realization = analysis.axisRealization(value, axis);
        return realization.hasExtentAuthority() &&
               !realization.constructionScalarSeed;
      };
      auto lhs = contract.getLhs().getType();
      auto rhs = contract.getRhs().getType();
      SmallVector<unsigned> lhsFree = retainedAxes(axes->lhsResultAxes);
      SmallVector<unsigned> rhsFree = retainedAxes(axes->rhsResultAxes);
      SmallVector<unsigned> lhsErased;
      SmallVector<unsigned> rhsErased;
      if (lhsFree.size() > 1)
        for (unsigned axis : lhsFree)
          if (unitAxis(contract.getLhs(), axis))
            lhsErased.push_back(axis);
      if (rhsFree.size() > 1)
        for (unsigned axis : rhsFree)
          if (unitAxis(contract.getRhs(), axis))
            rhsErased.push_back(axis);
      const size_t lhsRemaining = lhsFree.size() - lhsErased.size();
      const size_t rhsRemaining = rhsFree.size() - rhsErased.size();
      bool removesBatches = llvm::all_of(contract.getLhsBatchAxes(), [&](int64_t axis) {
        return llvm::is_contained(lhsErased, axis);
      }) && llvm::all_of(contract.getRhsBatchAxes(), [&](int64_t axis) {
        return llvm::is_contained(rhsErased, axis);
      });
      if ((!lhsErased.empty() || !rhsErased.empty()) && lhsRemaining == 1 &&
          rhsRemaining == 1 && removesBatches) {
        auto eraseAxes = [&](FragmentType source,
                             ArrayRef<unsigned> erased) {
          SmallVector<Attribute> shape;
          SmallVector<Attribute> mappings;
          for (auto [axis, extent] : llvm::enumerate(source.getShape())) {
            if (llvm::is_contained(erased, axis))
              continue;
            shape.push_back(extent);
            auto mapping = cast<AxisMapAttr>(source.getAxisMaps()[axis]);
            mappings.push_back(AxisMapAttr::get(
                source.getContext(), mapping.getSourceId(),
                mapping.getSourceAxis(), mapping.getDimensionId(),
                mappings.size(), mapping.getDerived()));
          }
          return FragmentType::get(
              source.getContext(), source.getElementType(),
              ArrayAttr::get(source.getContext(), shape),
              ArrayAttr::get(source.getContext(), mappings),
              source.getValidity(), source.getOwner());
        };
        auto remap = [](ArrayRef<int64_t> axes,
                        ArrayRef<unsigned> erased) {
          SmallVector<int64_t> result;
          for (int64_t axis : axes) {
            int64_t shift = llvm::count_if(erased, [&](unsigned removed) {
              return removed < static_cast<unsigned>(axis);
            });
            result.push_back(axis - shift);
          }
          return result;
        };
        SmallVector<unsigned> resultErased;
        FragmentType originalResult = contract.getResult().getType();
        auto appendResultAxes = [&](ArrayRef<std::optional<unsigned>> positions,
                                    ArrayRef<unsigned> erased,
                                    ArrayRef<int64_t> pairedBatch) {
          for (unsigned axis : erased) {
            if (llvm::is_contained(pairedBatch, axis))
              continue;
            auto position = positions[axis];
            if (!position || llvm::is_contained(resultErased, *position))
              return failure();
            resultErased.push_back(*position);
          }
          return success();
        };
        if (failed(appendResultAxes(axes->lhsResultAxes, lhsErased, {})) ||
            failed(appendResultAxes(axes->rhsResultAxes, rhsErased,
                                    contract.getRhsBatchAxes())))
          return contract.emitOpError(
              "singleton matrix axes have no exact result projection");
        FragmentType squeezedLhs = eraseAxes(lhs, lhsErased);
        FragmentType squeezedRhs = eraseAxes(rhs, rhsErased);
        FragmentType squeezedResult = eraseAxes(originalResult, resultErased);
        OpBuilder builder(contract);
        auto eraseRelation = [&](unsigned sourceRank,
                                 ArrayRef<unsigned> erased) {
          SmallVector<Attribute> groups;
          unsigned resultAxis = 0;
          for (unsigned sourceAxis = 0; sourceAxis < sourceRank; ++sourceAxis) {
            SmallVector<int64_t> resultAxes;
            if (!llvm::is_contained(erased, sourceAxis))
              resultAxes.push_back(resultAxis++);
            groups.push_back(ReshapeGroupAttr::get(
                contract.getContext(),
                DenseI64ArrayAttr::get(
                    contract.getContext(),
                    {static_cast<int64_t>(sourceAxis)}),
                DenseI64ArrayAttr::get(contract.getContext(), resultAxes)));
          }
          return ArrayAttr::get(contract.getContext(), groups);
        };
        auto reshapeTo = [&](Value value, FragmentType target,
                             ArrayRef<unsigned> erased) -> FailureOr<Value> {
          auto source = dyn_cast<FragmentType>(value.getType());
          if (!source)
            return failure();
          if (source == target)
            return value;
          auto reshape = builder.create<ReshapeOp>(
              contract.getLoc(), target, value,
              eraseRelation(source.getShape().size(), erased));
          if (Attribute origin = contract->getAttr(originAttr))
            reshape->setAttr(originAttr, origin);
          return reshape.getResult();
        };
        FailureOr<Value> normalizedLhs =
            reshapeTo(contract.getLhs(), squeezedLhs, lhsErased);
        FailureOr<Value> normalizedRhs =
            reshapeTo(contract.getRhs(), squeezedRhs, rhsErased);
        FailureOr<Value> normalizedAccumulator =
            reshapeTo(contract.getAccumulator(), squeezedResult, resultErased);
        if (failed(normalizedLhs) || failed(normalizedRhs) ||
            failed(normalizedAccumulator))
          return contract.emitOpError(
              "cannot squeeze logical unit free axes for matrix ownership");
        contract->setOperand(0, *normalizedLhs);
        contract->setOperand(1, *normalizedRhs);
        contract->setOperand(2, *normalizedAccumulator);
        contract->setAttr(
            "lhs_reduction_axes",
            builder.getDenseI64ArrayAttr(
                remap(contract.getLhsReductionAxes(), lhsErased)));
        contract->setAttr(
            "rhs_reduction_axes",
            builder.getDenseI64ArrayAttr(
                remap(contract.getRhsReductionAxes(), rhsErased)));
        contract->setAttr("lhs_batch_axes", builder.getDenseI64ArrayAttr({}));
        contract->setAttr("rhs_batch_axes", builder.getDenseI64ArrayAttr({}));
        contract.getResult().setType(squeezedResult);

        SmallVector<Attribute> restoredGroups;
        unsigned sourceAxis = 0;
        for (unsigned resultAxis = 0;
             resultAxis < originalResult.getShape().size(); ++resultAxis) {
          SmallVector<int64_t> sourceAxes;
          if (!llvm::is_contained(resultErased, resultAxis))
            sourceAxes.push_back(sourceAxis++);
          restoredGroups.push_back(ReshapeGroupAttr::get(
              contract.getContext(),
              DenseI64ArrayAttr::get(contract.getContext(), sourceAxes),
              DenseI64ArrayAttr::get(
                  contract.getContext(),
                  {static_cast<int64_t>(resultAxis)})));
        }
        builder.setInsertionPointAfter(contract);
        auto restored = builder.create<ReshapeOp>(
            contract.getLoc(), originalResult, contract.getResult(),
            ArrayAttr::get(contract.getContext(), restoredGroups));
        if (Attribute origin = contract->getAttr(originAttr))
          restored->setAttr(originAttr, origin);
        contract.getResult().replaceUsesWithIf(
            restored.getResult(), [&](OpOperand &use) {
              return use.getOwner() != restored.getOperation();
            });
      }
    }
    auto lhs = contract.getLhs().getType();
    auto rhs = contract.getRhs().getType();
    unsigned lhsRank = lhs.getShape().size();
    unsigned rhsRank = rhs.getShape().size();
    if (lhsRank < 2 || rhsRank < 2)
      continue;
    OpBuilder builder(contract);
    if (contract.getLhsReductionAxes().front() ==
        static_cast<int64_t>(lhsRank - 2)) {
      SmallVector<int64_t> permutation;
      for (unsigned axis = 0; axis < lhsRank; ++axis)
        permutation.push_back(axis);
      std::swap(permutation[lhsRank - 2], permutation[lhsRank - 1]);
      Value transposed = builder.create<TransposeOp>(
          contract.getLoc(), transposeLastTwo(lhs), contract.getLhs(),
          permutation);
      contract->setOperand(0, transposed);
      contract->setAttr("lhs_reduction_axes",
                        builder.getDenseI64ArrayAttr(remapAxes(
                            contract.getLhsReductionAxes(), permutation)));
      contract->setAttr("lhs_batch_axes",
                        builder.getDenseI64ArrayAttr(
                            remapAxes(contract.getLhsBatchAxes(), permutation)));
    }
    if (contract.getRhsReductionAxes().front() ==
        static_cast<int64_t>(rhsRank - 1)) {
      SmallVector<int64_t> permutation;
      for (unsigned axis = 0; axis < rhsRank; ++axis)
        permutation.push_back(axis);
      std::swap(permutation[rhsRank - 2], permutation[rhsRank - 1]);
      Value transposed = builder.create<TransposeOp>(
          contract.getLoc(), transposeLastTwo(rhs), contract.getRhs(),
          permutation);
      contract->setOperand(1, transposed);
      contract->setAttr("rhs_reduction_axes",
                        builder.getDenseI64ArrayAttr(remapAxes(
                            contract.getRhsReductionAxes(), permutation)));
      contract->setAttr("rhs_batch_axes",
                        builder.getDenseI64ArrayAttr(
                            remapAxes(contract.getRhsBatchAxes(), permutation)));
    }
  }
  return success();
}

static Type transposeLoopSchema(Type type) {
  if (auto fragment = dyn_cast<FragmentType>(type))
    return fragment.getShape().size() == 2 ? transposeLastTwo(fragment) : type;
  if (auto record = dyn_cast<RecordType>(type)) {
    SmallVector<Attribute> fields;
    for (Attribute field : record.getFieldTypes())
      fields.push_back(TypeAttr::get(
          transposeLoopSchema(cast<TypeAttr>(field).getValue())));
    return RecordType::get(type.getContext(), record.getFieldNames(),
                           ArrayAttr::get(type.getContext(), fields),
                           record.getOwner());
  }
  return type;
}

static void matrixFields(Type type, SmallVectorImpl<FragmentType> &fields) {
  if (auto fragment = dyn_cast<FragmentType>(type)) {
    if (fragment.getShape().size() == 2)
      fields.push_back(fragment);
  } else if (auto record = dyn_cast<RecordType>(type)) {
    for (Attribute field : record.getFieldTypes())
      matrixFields(cast<TypeAttr>(field).getValue(), fields);
  }
}

static Value transposeLoopValue(OpBuilder &builder, Location location, Value value) {
  if (auto fragment = dyn_cast<FragmentType>(value.getType())) {
    if (fragment.getShape().size() != 2)
      return value;
    auto target = transposeLastTwo(fragment);
    if (auto splat = value.getDefiningOp<SplatOp>())
      return builder.create<SplatOp>(location, target, splat.getValue());
    return builder.create<TransposeOp>(location, target, value,
                                       ArrayRef<int64_t>{1, 0});
  }
  if (auto record = dyn_cast<RecordType>(value.getType())) {
    SmallVector<Value> fields;
    for (auto [index, type] : llvm::enumerate(record.getFieldTypes())) {
      Value field = builder.create<ExtractOp>(
          location, cast<TypeAttr>(type).getValue(), value, index);
      fields.push_back(transposeLoopValue(builder, location, field));
    }
    return builder.create<MakeRecordOp>(
        location, cast<RecordType>(transposeLoopSchema(record)), fields);
  }
  return value;
}

static bool supportsMatrixLoopTranspose(scf::ForOp loop) {
  auto supportedType = [&](Type type) {
    std::function<bool(Type)> supported = [&](Type current) {
      if (auto fragment = dyn_cast<FragmentType>(current))
        return fragment.getShape().size() <= 2;
      if (auto record = dyn_cast<RecordType>(current))
        return llvm::all_of(record.getFieldTypes(), [&](Attribute field) {
          return supported(cast<TypeAttr>(field).getValue());
        });
      return true;
    };
    return supported(type);
  };
  return !loop.walk([&](Operation *operation) {
    if (!llvm::all_of(operation->getOperandTypes(), supportedType) ||
        !llvm::all_of(operation->getResultTypes(), supportedType))
      return WalkResult::interrupt();
    if (auto load = dyn_cast<LoadOp>(operation))
      return isa<ViewType>(load.getResource().getType())
                 ? WalkResult::advance() : WalkResult::interrupt();
    if (auto contract = dyn_cast<ContractOp>(operation)) {
      if (contract.getLhs().getType().getShape().size() != 2 ||
          contract.getRhs().getType().getShape().size() != 2 ||
          !contract.getLhsBatchAxes().empty() ||
          !contract.getRhsBatchAxes().empty() ||
          contract.getLhsReductionAxes().size() != 1 ||
          contract.getRhsReductionAxes().size() != 1)
        return WalkResult::interrupt();
      return WalkResult::advance();
    }
    if (auto reshape = dyn_cast<ReshapeOp>(operation)) {
      auto source = dyn_cast<FragmentType>(reshape.getValue().getType());
      auto target = dyn_cast<FragmentType>(reshape.getResult().getType());
      if (!source || !target)
        return WalkResult::interrupt();
      auto nonUnit = [](FragmentType type) {
        return llvm::count_if(type.getShape(), [](Attribute extent) {
          auto expression = cast<PhysicalExprAttr>(extent);
          return expression.getKind() !=
                     PhysicalExprKind::Constant ||
                 expression.getValue() != 1;
        });
      };
      if (source.getShape() != target.getShape() &&
          (nonUnit(source) > 1 || nonUnit(target) > 1))
        return WalkResult::interrupt();
      return succeeded(inferReshapeReassociation(
                 cast<FragmentType>(transposeLoopSchema(source)),
                 cast<FragmentType>(transposeLoopSchema(target))))
                 ? WalkResult::advance() : WalkResult::interrupt();
    }
    if (auto reduce = dyn_cast<ReduceOp>(operation))
      return reduce.getAxes().size() == 1 ? WalkResult::advance()
                                          : WalkResult::interrupt();
    return isa<arith::ConstantOp, scf::ForOp, scf::IfOp, scf::YieldOp,
               UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp,
               SplatOp, BroadcastOp, TransposeOp, MakeRecordOp, ExtractOp,
               MakeRangeOp, RangeBoundOp, DimOp, AssumeInBoundsOp, YieldOp>(operation)
               ? WalkResult::advance() : WalkResult::interrupt();
  }).wasInterrupted();
}

static void transposeClonedLoop(scf::ForOp loop) {
  loop.walk([&](Operation *operation) {
    for (Value result : operation->getResults())
      result.setType(transposeLoopSchema(result.getType()));
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          argument.setType(transposeLoopSchema(argument.getType()));
  });
  loop.walk([&](Operation *operation) {
    if (auto contract = dyn_cast<ContractOp>(operation)) {
      Value lhs = contract.getLhs(), rhs = contract.getRhs();
      auto lhsAxes = remapAxes(contract.getRhsReductionAxes(), {1, 0});
      auto rhsAxes = remapAxes(contract.getLhsReductionAxes(), {1, 0});
      contract->setOperand(0, rhs);
      contract->setOperand(1, lhs);
      contract.setLhsReductionAxesAttr(
          DenseI64ArrayAttr::get(loop.getContext(), lhsAxes));
      contract.setRhsReductionAxesAttr(
          DenseI64ArrayAttr::get(loop.getContext(), rhsAxes));
    } else if (auto reduce = dyn_cast<ReduceOp>(operation)) {
      auto source = dyn_cast<FragmentType>(reduce.getSources().front().getType());
      if (source && source.getShape().size() == 2)
        reduce.setAxesAttr(DenseI64ArrayAttr::get(
            loop.getContext(), remapAxes(reduce.getAxes(), {1, 0})));
    } else if (auto reshape = dyn_cast<ReshapeOp>(operation)) {
      auto relation = inferReshapeReassociation(
          cast<FragmentType>(reshape.getValue().getType()),
          cast<FragmentType>(reshape.getResult().getType()));
      assert(succeeded(relation) && "loop transpose preflight checked reshapes");
      reshape.setReassociationAttr(*relation);
    }
  });
}

FailureOr<bool> projectContractResult(ContractOp contract) {
  if (!contract.getResult().hasOneUse())
    return false;
  auto gather = dyn_cast<GatherOp>(*contract.getResult().getUsers().begin());
  auto resultType = contract.getResult().getType();
  if (!gather || gather.getSource() != contract.getResult() ||
      gather.getCoordinates().size() > resultType.getShape().size() ||
      failed(scalarSource(contract.getAccumulator())) ||
      !canReplayContractionReads(contract))
    return false;
  auto gatheredType = dyn_cast<FragmentType>(gather.getResult().getType());
  if (gatheredType && gatheredType.getShape().size() > resultType.getShape().size())
    return false;
  auto kernel = contract->getParentOfType<func::FuncOp>();
  DominanceInfo dominance(kernel);
  llvm::SmallDenseSet<Value> checked;
  std::function<bool(Value)> canMoveCoordinate = [&](Value value) {
    if (dominance.dominates(value, contract.getOperation()))
      return true;
    if (!checked.insert(value).second)
      return true;
    Operation *producer = value.getDefiningOp();
    return producer && producer->getNumRegions() == 0 &&
           !isa<RecordType>(value.getType()) &&
           (gatheredType || !isa<FragmentType>(value.getType())) &&
           (isa<arith::ConstantOp, PhysicalExprOp>(producer) ||
            isPhysicalReplayNode(producer, PhysicalReplayScope::Coordinate,
                                 /*allowAccesses=*/false)) &&
           llvm::all_of(producer->getOperands(), canMoveCoordinate);
  };
  if (!gatheredType && gather.getValid() && !canMoveCoordinate(gather.getValid()))
    return false;
  SmallVector<Value> selected(resultType.getShape().size());
  SmallVector<int64_t> resultOrder(resultType.getShape().size(), -1);
  llvm::SmallDenseSet<unsigned> selectedResultAxes;
  for (auto [coordinate, axis] : llvm::zip(gather.getCoordinates(), gather.getSourceAxes())) {
    if (auto scalar = scalarSource(coordinate); succeeded(scalar))
      coordinate = *scalar;
    if (axis < 0 || axis >= static_cast<int64_t>(selected.size()) ||
        selected[axis] ||
        !canMoveCoordinate(coordinate))
      return false;
    if (gatheredType && isa<FragmentType>(coordinate.getType())) {
      auto range = coordinate.getDefiningOp<MakeRangeOp>();
      if (!range || !isUnitStepRange(range))
        return false;
      auto projections = queryRangeProjections(gatheredType, range);
      if (projections.size() != 1 ||
          !selectedResultAxes.insert(projections.front().fragmentAxis).second ||
          gatheredType.getShape()[projections.front().fragmentAxis] !=
              range.getResult().getType().getShape()[0])
        return false;
      resultOrder[axis] = projections.front().fragmentAxis;
    } else if (isa<FragmentType>(coordinate.getType())) {
      return false;
    }
    selected[axis] = coordinate;
  }
  for (unsigned axis = 0; axis < selected.size(); ++axis) {
    if (selected[axis])
      continue;
    if (!gatheredType)
      return false;
    auto mapping = cast<AxisMapAttr>(resultType.getAxisMaps()[axis]);
    auto retained = queryFragmentAxis(gatheredType, sourceAxisIdentity(mapping));
    if (!retained.isExact() || retained.dimensionId != mapping.getDimensionId() ||
        gatheredType.getShape()[retained.fragmentAxis] != resultType.getShape()[axis] ||
        !selectedResultAxes.insert(retained.fragmentAxis).second)
      return false;
    resultOrder[axis] = retained.fragmentAxis;
  }
  if (gatheredType && selectedResultAxes.size() != gatheredType.getShape().size())
    return false;
  auto isUnit = [](Attribute attribute) {
    auto extent = cast<PhysicalExprAttr>(attribute);
    return extent.getKind() == PhysicalExprKind::Constant &&
           extent.getValue() == 1;
  };
  SmallVector<SmallVector<unsigned>, 2> resultAxes(2);
  unsigned nextResultAxis = 0;
  unsigned side = 0;
  for (auto [operand, reductions] :
       {std::pair{contract.getLhs(), contract.getLhsReductionAxes()},
        std::pair{contract.getRhs(), contract.getRhsReductionAxes()}}) {
    auto type = cast<FragmentType>(operand.getType());
    resultAxes[side].resize(type.getShape().size());
    for (unsigned axis = 0; axis < type.getShape().size(); ++axis) {
      if (llvm::is_contained(reductions, static_cast<int64_t>(axis)))
        continue;
      auto batch = llvm::find(contract.getRhsBatchAxes(), static_cast<int64_t>(axis));
      if (side == 1 && batch != contract.getRhsBatchAxes().end()) {
        unsigned pair = std::distance(contract.getRhsBatchAxes().begin(), batch);
        resultAxes[side][axis] = resultAxes[0][contract.getLhsBatchAxes()[pair]];
      } else {
        resultAxes[side][axis] = nextResultAxis++;
      }
    }
    ++side;
  }
  PhysicalProgramAnalysis analysis(kernel);
  side = 0;
  for (auto [operand, reductions] :
       {std::pair{contract.getLhs(), contract.getLhsReductionAxes()},
        std::pair{contract.getRhs(), contract.getRhsReductionAxes()}}) {
    auto type = cast<FragmentType>(operand.getType());
    for (unsigned axis = 0; axis < type.getShape().size(); ++axis) {
      if (llvm::is_contained(reductions, static_cast<int64_t>(axis)))
        continue;
      auto mapping = cast<AxisMapAttr>(type.getAxisMaps()[axis]);
      auto resultMapping = cast<AxisMapAttr>(
          resultType.getAxisMaps()[resultAxes[side][axis]]);
      bool pairedRhsBatch = side == 1 && llvm::is_contained(
          contract.getRhsBatchAxes(), static_cast<int64_t>(axis));
      if (!pairedRhsBatch &&
          (mapping.getDimensionId() <= 0 ||
           mapping.getDimensionId() != resultMapping.getDimensionId()))
        return false;
      if (!selected[resultAxes[side][axis]])
        continue;
      if (isUnit(type.getShape()[axis]) &&
          isIntegerConstant(selected[resultAxes[side][axis]], 0))
        continue;
      auto roots = analysis.axisRanges(operand, axis);
      auto range = queryExactLogicalRange(roots);
      auto occurrences = analysis.rangeAxes(operand, roots.roots);
      if (failed(range) || !isUnitStepRange(*range) ||
          !analysis.lockstepRanges(roots.roots).isExact() ||
          !occurrences.isExact() ||
          occurrences.fragmentAxes != SmallVector<unsigned>{axis} ||
          !isIntegerConstant((*range).getStart(), 0) ||
          !isIntegerConstant((*range).getLogicalStart(), 0) ||
          (!samePhysicalScalarExpression((*range).getExtent(),
                                         (*range).getLogicalStop()) &&
           !isFullCoverageExtent(contract,
                                 cast<PhysicalExprAttr>(type.getShape()[axis]))))
        return false;
    }
    ++side;
  }
  OpBuilder builder(contract);
  IRMapping coordinates;
  std::function<Value(Value)> moveCoordinate = [&](Value value) -> Value {
    if (Value mapped = coordinates.lookupOrNull(value))
      return mapped;
    if (dominance.dominates(value, contract.getOperation()))
      return value;
    Operation *producer = value.getDefiningOp();
    for (Value operand : producer->getOperands())
      coordinates.map(operand, moveCoordinate(operand));
    Operation *clone = builder.clone(*producer, coordinates);
    return clone->getResult(cast<OpResult>(value).getResultNumber());
  };
  for (Value &coordinate : selected)
    if (coordinate)
      coordinate = moveCoordinate(coordinate);
  PhysicalExprAttr unit = expression(kernel.getContext(), PhysicalExprKind::Constant, 1);
  Value one = builder.create<arith::ConstantIndexOp>(contract.getLoc(), 1);
  side = 0;
  SmallVector<Value> operands;
  for (auto [original, reductions] :
       {std::pair{contract.getLhs(), contract.getLhsReductionAxes()},
        std::pair{contract.getRhs(), contract.getRhsReductionAxes()}}) {
    Value operand = original;
    auto originalType = cast<FragmentType>(original.getType());
    for (unsigned axis = 0; axis < originalType.getShape().size(); ++axis) {
      if (llvm::is_contained(reductions, static_cast<int64_t>(axis)))
        continue;
      Value coordinate = selected[resultAxes[side][axis]];
      if (!coordinate)
        continue;
      if (isUnit(originalType.getShape()[axis]) && isIntegerConstant(coordinate, 0))
        continue;
      auto roots = PhysicalProgramAnalysis(kernel).axisRanges(operand, axis);
      auto authority = queryExactLogicalRange(roots);
      if (failed(authority))
        return contract.emitOpError("contraction projection lost its free-axis range");
      auto mapping = cast<AxisMapAttr>(originalType.getAxisMaps()[axis]);
      unsigned resultAxis = resultAxes[side][axis];
      auto selectedRange = coordinate.getDefiningOp<MakeRangeOp>();
      auto projectedExtent = selectedRange
                                 ? cast<PhysicalExprAttr>(
                                       gatheredType.getShape()[resultOrder[resultAxis]])
                                 : unit;
      auto projectedMapping = selectedRange
                                  ? cast<AxisMapAttr>(
                                        gatheredType.getAxisMaps()[resultOrder[resultAxis]])
                                  : mapping;
      auto coordinateType = fragmentType(kernel.getContext(), builder.getIndexType(),
                                          {projectedExtent}, {projectedMapping},
                                          originalType.getOwner());
      Value range;
      if (selectedRange) {
        range = builder.create<MakeRangeOp>(
            contract.getLoc(), coordinateType, selectedRange.getStart(),
            selectedRange.getExtent(), one, (*authority).getLogicalStart(),
            (*authority).getLogicalStop(), projectedMapping.getSourceId(),
            projectedMapping.getSourceAxis(), projectedMapping.getDerived());
        inheritRangeAuthority(range, *authority);
      } else {
        range = builder.create<SplatOp>(contract.getLoc(), coordinateType,
                                       coordinate);
      }
      IRMapping replay;
      for (MakeRangeOp root : roots.roots)
        replay.map(root.getResult(), range);
      ReplayMaterializationOptions options;
      options.fragmentAxis = axis;
      options.traversalRanges = roots.roots;
      if (selectedRange)
        options.segmentMapping = projectedMapping;
      auto predicateType = fragmentType(
          kernel.getContext(), builder.getI1Type(), {projectedExtent},
          {projectedMapping}, originalType.getOwner());
      // Keep each input bounded by its own free axis. The original gather
      // predicate and fill still guard the projected result below.
      options.segmentTail = rangeBoundsValidity(
          builder, contract.getLoc(), coordinateType, predicateType, range,
          (*authority).getLogicalStop());
      options.materializeZeroFill = true;
      auto projected = materializeReplayedValue(
          builder, contract.getLoc(), operand, sourceAxisIdentity(mapping),
          projectedExtent, replay, contract, options);
      if (failed(projected))
        return contract.emitOpError("cannot project a contraction input");
      operand = *projected;
    }
    operands.push_back(operand);
    ++side;
  }
  SmallVector<Attribute> shape(resultType.getShape().size(), unit);
  auto projectedType = FragmentType::get(
      kernel.getContext(), resultType.getElementType(), builder.getArrayAttr(shape),
      resultType.getAxisMaps(), resultType.getValidity(), resultType.getOwner());
  if (gatheredType) {
    SmallVector<Attribute> projectedShape, projectedMaps;
    for (auto [axis, gatheredAxis] : llvm::enumerate(resultOrder)) {
      projectedShape.push_back(gatheredAxis >= 0
                                   ? gatheredType.getShape()[gatheredAxis] : unit);
      auto mapping = cast<AxisMapAttr>(gatheredAxis >= 0
          ? gatheredType.getAxisMaps()[gatheredAxis] : resultType.getAxisMaps()[axis]);
      projectedMaps.push_back(AxisMapAttr::get(
          kernel.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
          mapping.getDimensionId(), axis, mapping.getDerived()));
    }
    projectedType = FragmentType::get(
        kernel.getContext(), gatheredType.getElementType(),
        builder.getArrayAttr(projectedShape), builder.getArrayAttr(projectedMaps),
        gatheredType.getValidity(), gatheredType.getOwner());
  }
  auto accumulator = projectPhysicalValueToSchema(
      builder, contract.getLoc(), contract.getAccumulator(), projectedType);
  if (failed(accumulator))
    return contract.emitOpError("contraction has no uniform accumulator projection");
  auto projected = builder.create<ContractOp>(
      contract.getLoc(), projectedType, operands[0], operands[1], *accumulator,
      contract.getLhsReductionAxes(), contract.getRhsReductionAxes(),
      contract.getLhsBatchAxes(), contract.getRhsBatchAxes());
  builder.setInsertionPoint(gather);
  if (gatheredType) {
    Value result = projected.getResult();
    SmallVector<Attribute> retainedShape, retainedMaps;
    SmallVector<int64_t> permutation(gatheredType.getShape().size());
    for (auto [axis, gatheredAxis] : llvm::enumerate(resultOrder)) {
      if (gatheredAxis < 0)
        continue;
      unsigned retainedAxis = retainedShape.size();
      permutation[gatheredAxis] = retainedAxis;
      retainedShape.push_back(projectedType.getShape()[axis]);
      auto mapping = cast<AxisMapAttr>(projectedType.getAxisMaps()[axis]);
      retainedMaps.push_back(AxisMapAttr::get(
          kernel.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
          mapping.getDimensionId(), retainedAxis, mapping.getDerived()));
    }
    if (retainedShape.size() != resultOrder.size()) {
      auto retainedType = FragmentType::get(
          kernel.getContext(), gatheredType.getElementType(),
          builder.getArrayAttr(retainedShape), builder.getArrayAttr(retainedMaps),
          gatheredType.getValidity(), gatheredType.getOwner());
      auto reassociation = inferReshapeReassociation(projectedType, retainedType);
      if (failed(reassociation))
        return contract.emitOpError("contraction projection cannot drop its selected singleton axes");
      result = builder.create<ReshapeOp>(gather.getLoc(), retainedType, result,
                                        *reassociation);
    }
    if (llvm::any_of(llvm::enumerate(permutation), [](auto entry) {
          return entry.index() != static_cast<unsigned>(entry.value());
        }))
      result = builder.create<TransposeOp>(gather.getLoc(), gatheredType, result,
                                           permutation);
    if (gather.getValid())
      result = builder.create<SelectOp>(gather.getLoc(), gatheredType,
                                       gather.getValid(), result, gather.getFill());
    gather.getResult().replaceAllUsesWith(result);
    gather.erase();
    contract.erase();
    return true;
  }
  Value zero = builder.create<arith::ConstantIndexOp>(gather.getLoc(), 0);
  SmallVector<Value> zeros(selected.size(), zero);
  auto replacement = builder.create<GatherOp>(
      gather.getLoc(), gather.getResult().getType(), projected, zeros,
      gather.getValid(), gather.getFill(), gather.getSourceAxes());
  gather.getResult().replaceAllUsesWith(replacement.getResult());
  gather.erase();
  contract.erase();
  return true;
}

LogicalResult normalizeMatrixContractShapes(func::FuncOp kernel) {
  llvm::DenseSet<StringAttr> units;
  for (Attribute declaration : getParameterDeclarations(kernel)) {
    auto schema = cast<ParameterAttr>(declaration);
    auto role = schema.getRole();
    if ((role != ParameterRole::OwnershipM &&
         role != ParameterRole::OwnershipN) ||
        schema.getCandidates().size() != 1 || schema.getCandidates()[0] != 1)
      continue;
    units.insert(schema.getName());
  }
  kernel.walk([&](ParameterOp read) {
    if (!units.contains(read.getReference().getName())) return;
    OpBuilder builder(read);
    Value constant = builder.create<arith::ConstantOp>(
        read.getLoc(), read.getResult().getType(),
        builder.getIntegerAttr(read.getResult().getType(), 1));
    read.getResult().replaceAllUsesWith(constant);
  });
  // Ownership and coverage are already closed at this boundary. A singleton
  // physical candidate is an exact extent, independent of the target API.
  AttrTypeReplacer replacer;
  replacer.addReplacement([&](PhysicalExprAttr extent) -> std::optional<Attribute> {
    if (extent.getKind() != PhysicalExprKind::Parameter ||
        !units.contains(extent.getParameterReference().getName()))
      return std::nullopt;
    return expression(kernel.getContext(), PhysicalExprKind::Constant, 1);
  });
  replacer.recursivelyReplaceElementsIn(kernel.getOperation(),
                                        /*replaceAttrs=*/true,
                                        /*replaceLocs=*/false,
                                        /*replaceTypes=*/true);
  auto [nextSource, nextDimension] = nextPhysicalAxisIdentities(kernel);
  SmallVector<ContractOp> contracts;
  kernel.walk([&](ContractOp contract) { contracts.push_back(contract); });
  for (ContractOp contract : contracts) {
    auto lhs = contract.getLhs().getType();
    auto rhs = contract.getRhs().getType();
    std::string reason;
    auto axes = queryContractionAxes(contract, &reason);
    if (!axes)
      return contract.emitOpError("invalid matrix-contraction axis schema: ") << reason;
    if (axes->reduction.size() != 1 ||
        (lhs.getShape().size() <= 2 && rhs.getShape().size() <= 2))
      continue;
    ArrayRef<int64_t> lhsBatch = contract.getLhsBatchAxes();
    ArrayRef<int64_t> rhsBatch = contract.getRhsBatchAxes();
    unsigned batchRank = lhsBatch.size();
    if (batchRank > 1)
      continue;
    if (batchRank && axes->hasCanonicalMatrixAxes())
      continue;
    SmallVector<int64_t> lhsFree(axes->lhsFree.begin(), axes->lhsFree.end());
    SmallVector<int64_t> rhsFree(axes->rhsFree.begin(), axes->rhsFree.end());
    if (lhsFree.empty() || rhsFree.empty())
      return contract.emitOpError("matrix normalization requires free axes on both operands");
    OpBuilder builder(contract);
    Location location = contract.getLoc();
    auto remap = [&](AxisMapAttr axis, unsigned position) {
      return AxisMapAttr::get(kernel.getContext(), axis.getSourceId(),
          axis.getSourceAxis(), axis.getDimensionId(), position, axis.getDerived());
    };
    auto collapsedAxis = [&](FragmentType type, ArrayRef<int64_t> axes,
                             unsigned position) {
      if (axes.size() == 1)
        return remap(cast<AxisMapAttr>(type.getAxisMaps()[axes.front()]), position);
      return AxisMapAttr::get(kernel.getContext(), nextSource++, 0,
                                   nextDimension++, position, true);
    };
    auto product = [&](FragmentType type, ArrayRef<int64_t> axes) {
      auto extent = cast<PhysicalExprAttr>(type.getShape()[axes.front()]);
      for (int64_t axis : axes.drop_front())
        extent = PhysicalExprAttr::get(kernel.getContext(),
            PhysicalExprKind::Multiply, 0,
            builder.getStringAttr(""),
            builder.getArrayAttr({extent, type.getShape()[axis]}));
      return extent;
    };
    auto makeType = [&](FragmentType source, ArrayRef<Attribute> shape,
                        ArrayRef<Attribute> mappings) {
      return FragmentType::get(kernel.getContext(), source.getElementType(),
          builder.getArrayAttr(shape), builder.getArrayAttr(mappings),
          source.getValidity(), source.getOwner());
    };
    auto transpose = [&](Value value, ArrayRef<int64_t> permutation) {
      if (llvm::all_of(llvm::enumerate(permutation), [](auto entry) {
            return static_cast<int64_t>(entry.index()) == entry.value();
          }))
        return value;
      auto type = cast<FragmentType>(value.getType());
      SmallVector<Attribute> shape, mappings;
      for (auto [position, axis] : llvm::enumerate(permutation)) {
        shape.push_back(type.getShape()[axis]);
        mappings.push_back(remap(cast<AxisMapAttr>(type.getAxisMaps()[axis]), position));
      }
      return Value(builder.create<TransposeOp>(location,
          makeType(type, shape, mappings), value, permutation));
    };
    auto reshape = [&](Value value, FragmentType target) -> FailureOr<Value> {
      if (value.getType() == target) return value;
      auto relation = inferReshapeReassociation(
          cast<FragmentType>(value.getType()), target);
      if (failed(relation)) return failure();
      return Value(builder.create<ReshapeOp>(location, target, value, *relation));
    };
    auto originalResult = contract.getResult().getType();
    SmallVector<int64_t> resultPermutation;
    auto appendResultAxes = [&](ArrayRef<std::optional<unsigned>> positions,
                                ArrayRef<int64_t> selected) {
      for (int64_t axis : selected)
        resultPermutation.push_back(*positions[axis]);
    };
    appendResultAxes(axes->lhsResultAxes, lhsBatch);
    appendResultAxes(axes->lhsResultAxes, lhsFree);
    appendResultAxes(axes->rhsResultAxes, rhsFree);
    auto m = product(lhs, lhsFree);
    auto n = product(rhs, rhsFree);
    auto mAxis = collapsedAxis(lhs, lhsFree, batchRank);
    auto nAxis = collapsedAxis(rhs, rhsFree, batchRank + 1);
    int64_t lhsK = contract.getLhsReductionAxes().front();
    int64_t rhsK = contract.getRhsReductionAxes().front();
    SmallVector<int64_t> lhsPermutation(lhsBatch);
    llvm::append_range(lhsPermutation, lhsFree);
    lhsPermutation.push_back(lhsK);
    SmallVector<int64_t> rhsPermutation(rhsBatch);
    rhsPermutation.push_back(rhsK);
    llvm::append_range(rhsPermutation, rhsFree);
    SmallVector<Attribute> lhsShape, lhsMappings, rhsShape, rhsMappings;
    SmallVector<Attribute> resultShape, resultMappings;
    SmallVector<int64_t> matrixBatch;
    for (auto [position, pair] : llvm::enumerate(llvm::zip(lhsBatch, rhsBatch))) {
      auto [left, right] = pair;
      lhsShape.push_back(lhs.getShape()[left]);
      lhsMappings.push_back(remap(
          cast<AxisMapAttr>(lhs.getAxisMaps()[left]), position));
      rhsShape.push_back(rhs.getShape()[right]);
      rhsMappings.push_back(remap(
          cast<AxisMapAttr>(rhs.getAxisMaps()[right]), position));
      int64_t resultAxis = resultPermutation[position];
      resultShape.push_back(originalResult.getShape()[resultAxis]);
      resultMappings.push_back(remap(
          cast<AxisMapAttr>(originalResult.getAxisMaps()[resultAxis]), position));
      matrixBatch.push_back(position);
    }
    llvm::append_range(lhsShape, ArrayRef<Attribute>{m, lhs.getShape()[lhsK]});
    llvm::append_range(lhsMappings, ArrayRef<Attribute>{mAxis,
        remap(cast<AxisMapAttr>(lhs.getAxisMaps()[lhsK]), batchRank + 1)});
    llvm::append_range(rhsShape, ArrayRef<Attribute>{rhs.getShape()[rhsK], n});
    llvm::append_range(rhsMappings, ArrayRef<Attribute>{
        remap(cast<AxisMapAttr>(rhs.getAxisMaps()[rhsK]), batchRank), nAxis});
    llvm::append_range(resultShape, ArrayRef<Attribute>{m, n});
    llvm::append_range(resultMappings, ArrayRef<Attribute>{mAxis, nAxis});
    auto matrixResult = makeType(originalResult, resultShape, resultMappings);
    FailureOr<Value> matrixLhs = reshape(transpose(contract.getLhs(), lhsPermutation),
        makeType(lhs, lhsShape, lhsMappings));
    FailureOr<Value> matrixRhs = reshape(transpose(contract.getRhs(), rhsPermutation),
        makeType(rhs, rhsShape, rhsMappings));
    Value orderedAccumulator = transpose(contract.getAccumulator(), resultPermutation);
    auto orderedResult = cast<FragmentType>(orderedAccumulator.getType());
    FailureOr<Value> matrixAccumulator = reshape(orderedAccumulator, matrixResult);
    if (failed(matrixLhs) || failed(matrixRhs) || failed(matrixAccumulator))
      return contract.emitOpError("matrix form has no exact row-major reshape");
    contract->setOperands(ValueRange{*matrixLhs, *matrixRhs, *matrixAccumulator});
    contract->setAttr("lhs_reduction_axes", builder.getDenseI64ArrayAttr({batchRank + 1}));
    contract->setAttr("rhs_reduction_axes", builder.getDenseI64ArrayAttr({batchRank}));
    contract->setAttr("lhs_batch_axes", builder.getDenseI64ArrayAttr(matrixBatch));
    contract->setAttr("rhs_batch_axes", builder.getDenseI64ArrayAttr(matrixBatch));
    contract.getResult().setType(matrixResult);
    builder.setInsertionPointAfter(contract);
    FailureOr<Value> restored = reshape(contract.getResult(), orderedResult);
    if (failed(restored))
      return contract.emitOpError("matrix result has no inverse row-major reshape");
    Operation *firstRestore = (*restored).getDefiningOp();
    SmallVector<int64_t> inverse(resultPermutation.size());
    for (auto [position, axis] : llvm::enumerate(resultPermutation))
      inverse[axis] = position;
    Value result = transpose(*restored, inverse);
    contract.getResult().replaceUsesWithIf(result, [&](OpOperand &use) {
      return use.getOwner() != firstRestore &&
             use.getOwner() != result.getDefiningOp();
    });
  }
  for (ContractOp contract : contracts) {
    auto unitBatch = [](FragmentType type) {
      return type.getShape().size() == 3 &&
             constantPhysicalExpression(
                 cast<PhysicalExprAttr>(type.getShape()[0])) == 1;
    };
    if (contract.getLhsBatchAxes() != ArrayRef<int64_t>{0} ||
        contract.getRhsBatchAxes() != ArrayRef<int64_t>{0} ||
        contract.getLhsReductionAxes() != ArrayRef<int64_t>{2} ||
        contract.getRhsReductionAxes() != ArrayRef<int64_t>{1} ||
        !unitBatch(contract.getLhs().getType()) ||
        !unitBatch(contract.getRhs().getType()) ||
        !unitBatch(contract.getAccumulator().getType()) ||
        !unitBatch(contract.getResult().getType()))
      continue;
    OpBuilder builder(contract);
    auto matrixType = [&](FragmentType type) {
      SmallVector<Attribute> axes;
      for (Attribute attribute : type.getAxisMaps().getValue().drop_front()) {
        auto axis = cast<AxisMapAttr>(attribute);
        axes.push_back(AxisMapAttr::get(
            kernel.getContext(), axis.getSourceId(), axis.getSourceAxis(),
            axis.getDimensionId(), axes.size(), axis.getDerived()));
      }
      return FragmentType::get(
          kernel.getContext(), type.getElementType(),
          builder.getArrayAttr(type.getShape().getValue().drop_front()),
          builder.getArrayAttr(axes), type.getValidity(), type.getOwner());
    };
    SmallVector<Value> operands;
    for (Value operand : {contract.getLhs(), contract.getRhs(),
                          contract.getAccumulator()}) {
      auto source = cast<FragmentType>(operand.getType());
      auto target = matrixType(source);
      auto reassociation = inferReshapeReassociation(source, target);
      if (failed(reassociation))
        return contract.emitOpError("unit-batch operand has no exact row-major reshape");
      operands.push_back(builder.create<ReshapeOp>(
          contract.getLoc(), target, operand, *reassociation));
    }
    auto original = contract.getResult().getType();
    auto projected = matrixType(original);
    auto reassociation = inferReshapeReassociation(projected, original);
    if (failed(reassociation))
      return contract.emitOpError("unit-batch result has no inverse row-major reshape");
    auto matrix = builder.create<ContractOp>(
        contract.getLoc(), projected, operands[0], operands[1], operands[2],
        ArrayRef<int64_t>{1}, ArrayRef<int64_t>{0},
        ArrayRef<int64_t>{}, ArrayRef<int64_t>{});
    auto restored = builder.create<ReshapeOp>(
        contract.getLoc(), original, matrix, *reassociation);
    if (Attribute origin = contract->getAttr(originAttr)) {
      matrix->setAttr(originAttr, origin);
      restored->setAttr(originAttr, origin);
    }
    contract.getResult().replaceAllUsesWith(restored);
    contract.erase();
  }
  kernel.walk([&](scf::ForOp loop) {
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    for (auto [index, carried] : llvm::enumerate(loop.getRegionIterArgs())) {
      if (!carried.hasOneUse())
        continue;
      auto projected = dyn_cast<ReshapeOp>(*carried.getUsers().begin());
      auto restored = yield.getOperand(index).getDefiningOp<ReshapeOp>();
      if (!projected || !restored ||
          projected->getBlock() != loop.getBody() ||
          restored->getBlock() != loop.getBody() ||
          !restored.getResult().hasOneUse() ||
          restored.getResult().getType() != carried.getType() ||
          restored.getValue().getType() != projected.getResult().getType() ||
          projected.getResult().getType() == carried.getType())
        continue;

      // Keep the carry in the computation's shape. The inverse pure views at
      // the loop boundaries also preserve the value of a zero-trip loop.
      OpBuilder builder(loop);
      IRMapping initialMapping;
      initialMapping.map(carried, loop.getInitArgs()[index]);
      Operation *initial = builder.clone(*projected, initialMapping);
      Type type = projected.getResult().getType();
      loop.getInitArgsMutable()[index].assign(initial->getResult(0));
      carried.setType(type);
      loop.getResult(index).setType(type);
      yield->setOperand(index, restored.getValue());

      builder.setInsertionPointAfter(loop);
      IRMapping resultMapping;
      resultMapping.map(restored.getValue(), loop.getResult(index));
      Operation *result = builder.clone(*restored, resultMapping);
      loop.getResult(index).replaceUsesWithIf(result->getResult(0),
          [&](OpOperand &use) { return use.getOwner() != result; });
      projected.getResult().replaceAllUsesWith(carried);
      restored.erase();
      projected.erase();
    }
  });
  return success();
}

} // namespace intent::gpu::contraction

namespace intent::gpu {
using namespace contraction;

static LogicalResult orientLoopContractionsImpl(ModuleOp module) {
  auto kernel = getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  SmallVector<scf::ForOp> loops;
  kernel->walk([&](scf::ForOp loop) {
    if (!loop->getParentOfType<scf::ForOp>())
      loops.push_back(loop);
  });
  for (scf::ForOp loop : loops) {
    SmallVector<FragmentType> fields;
    for (Value value : loop.getInitArgs())
      matrixFields(value.getType(), fields);
    if (fields.empty())
      continue;
    FragmentType matrix = fields.front();
    if (!isa<FloatType>(matrix.getElementType()) ||
        !llvm::all_of(fields, [&](FragmentType field) {
          return field == matrix;
        }))
      continue;
    auto row = parameterForExtent(
        *kernel, cast<PhysicalExprAttr>(matrix.getShape()[0]));
    auto column = parameterForExtent(
        *kernel, cast<PhysicalExprAttr>(matrix.getShape()[1]));
    if (failed(row) || failed(column) ||
        row->getCategory() !=
            ParameterCategory::RegionContraction ||
        column->getRole() !=
            ParameterRole::FullCoverage)
      continue;
    llvm::DenseSet<Value> visited;
    ContractOp carriedContract;
    std::function<bool(Value)> dependsOnMatrixContract = [&](Value value) {
      if (!visited.insert(value).second)
        return false;
      Operation *producer = value.getDefiningOp();
      if (!producer || !loop->isProperAncestor(producer))
        return false;
      if (auto contract = dyn_cast<ContractOp>(producer))
        if (contract.getResult().getType().getShape() == matrix.getShape() &&
            contract.getResult().getType().getElementType() ==
                matrix.getElementType()) {
          carriedContract = contract;
          return true;
        }
      return llvm::any_of(producer->getOperands(), dependsOnMatrixContract);
    };
    bool carriesContract = llvm::any_of(
        loop.getBody()->getTerminator()->getOperands(), dependsOnMatrixContract);
    if (!carriesContract || !supportsMatrixLoopTranspose(loop))
      continue;
    // Orient the complete connected product graph, including joins. Every
    // matrix value is transposed together, so shared producers remain shared
    // and pointwise joins retain their original arithmetic order.
    llvm::DenseSet<Operation *> chain;
    SmallVector<ContractOp> pending{carriedContract};
    bool hasProducer = false;
    while (!pending.empty()) {
      ContractOp consumer = pending.pop_back_val();
      if (!chain.insert(consumer.getOperation()).second)
        continue;
      visited.clear();
      std::function<void(Value)> traceProducer = [&](Value value) {
        if (!visited.insert(value).second)
          return;
        Operation *producer = value.getDefiningOp();
        if (!producer || !loop->isProperAncestor(producer))
          return;
        if (auto contract = dyn_cast<ContractOp>(producer)) {
          hasProducer |= contract != carriedContract;
          pending.push_back(contract);
          return;
        }
        for (Value operand : producer->getOperands())
          traceProducer(operand);
      };
      traceProducer(consumer.getLhs());
      traceProducer(consumer.getRhs());
      traceProducer(consumer.getAccumulator());
    }
    bool disconnectedProducts = false;
    loop.walk([&](ContractOp contract) {
      disconnectedProducts |= !chain.contains(contract.getOperation());
    });
    if (!hasProducer || disconnectedProducts)
      continue;

    // Keep a strongly rectangular matrix's short axis in the column position.
    // The predicate uses existing tile parameters, not a new structural tuner.
    OpBuilder builder(loop);
    Location location = loop.getLoc();
    Value four = builder.create<arith::ConstantIndexOp>(location, 4);
    Value quarter = binary(builder, location, builder.getIndexType(),
                           materializeParameter(builder, location, column->getReference()), four,
                           BinaryOperator::FloorDivide);
    Value rectangular = builder.create<CompareOp>(
        location, builder.getI1Type(), materializeParameter(builder, location, row->getReference()), quarter,
        ComparePredicate::Le);
    auto choice = builder.create<scf::IfOp>(
        location, loop.getResultTypes(), rectangular, true);
    auto branchBuilder = [](Region &region) {
      Block &block = region.front();
      if (!block.empty() && isa<scf::YieldOp>(block.back()))
        block.back().erase();
      return OpBuilder(&block, block.end());
    };
    OpBuilder transposed = branchBuilder(choice.getThenRegion());
    IRMapping mapping;
    llvm::DenseSet<Value> captured;
    loop.walk([&](Operation *operation) {
      for (Value operand : operation->getOperands()) {
        Operation *owner = operand.getParentRegion()->getParentOp();
        if (owner == loop || loop->isProperAncestor(owner) ||
            !captured.insert(operand).second)
          continue;
        SmallVector<FragmentType> matrices;
        matrixFields(operand.getType(), matrices);
        if (!matrices.empty())
          mapping.map(operand,
                      transposeLoopValue(transposed, location, operand));
      }
    });
    auto replacement = cast<scf::ForOp>(transposed.clone(*loop, mapping));
    transposeClonedLoop(replacement);
    SmallVector<Value> results;
    for (Value result : replacement.getResults())
      results.push_back(transposeLoopValue(transposed, location, result));
    transposed.create<scf::YieldOp>(location, results);
    OpBuilder original = branchBuilder(choice.getElseRegion());
    IRMapping originalMapping;
    auto unchanged = cast<scf::ForOp>(original.clone(*loop, originalMapping));
    original.create<scf::YieldOp>(location, unchanged.getResults());
    loop.replaceAllUsesWith(choice.getResults());
    loop.erase();
  }
  eraseDeadPhysicalValues(*kernel);
  return success();
}

LogicalResult orientLoopContractions(ModuleOp module) {
  if (failed(orientLoopContractionsImpl(module))) return failure();
  auto kernel = getPhysicalKernel(module);
  return failed(kernel) ? failure() : closeValueRelations(*kernel);
}

} // namespace intent::gpu
