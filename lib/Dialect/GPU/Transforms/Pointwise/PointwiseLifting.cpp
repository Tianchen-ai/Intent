#include "Pointwise.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Transforms/Value/ExecutionSchema.h"
#include "Intent/Dialect/GPU/Transforms/Control/Predication.h"
#include "Intent/Dialect/GPU/Transforms/Value/SchemaMutation.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;

namespace intent::gpu::pointwise {
namespace {

FailureOr<SmallVector<int64_t>> contractionResultPermutation(
    ContractOp operation, const ExecutionSchema &schema,
    TypeRange originalOperands, FragmentType originalResult) {
  auto lhs = cast<FragmentType>(originalOperands[0]);
  auto rhs = cast<FragmentType>(originalOperands[1]);
  auto original = ContractionAxes::get(
      lhs.getShape().size(), rhs.getShape().size(),
      operation.getLhsReductionAxes(), operation.getRhsReductionAxes(),
      operation.getLhsBatchAxes(), operation.getRhsBatchAxes());
  auto output = schema.project(originalResult);
  if (!original || failed(output)) return failure();
  SmallVector<LiftedFragmentSchema> operands;
  SmallVector<FragmentType> beforeOperands{lhs, rhs};
  SmallVector<FragmentType> afterOperands{
      operation.getLhs().getType(), operation.getRhs().getType()};
  for (auto [before, after] : llvm::zip(beforeOperands, afterOperands)) {
    if (before == after) {
      LiftedFragmentSchema unchanged{before, {}, {}};
      for (unsigned axis = 0; axis < before.getShape().size(); ++axis)
        unchanged.oldToNew.push_back(axis);
      operands.push_back(std::move(unchanged));
    } else {
      auto lifted = schema.project(before);
      if (failed(lifted) || lifted->type != after) return failure();
      operands.push_back(std::move(*lifted));
    }
  }
  if (failed(remapSchemaAxes(operation, originalOperands, TypeRange{originalResult})))
    return failure();
  // The operands and axis attributes have moved, while the result and its
  // accumulator still have their original types. Derive the new result axes
  // from the transported operands before closing those two result boundaries.
  auto current = ContractionAxes::get(
      operation.getLhs().getType().getShape().size(),
      operation.getRhs().getType().getShape().size(),
      operation.getLhsReductionAxes(), operation.getRhsReductionAxes(),
      operation.getLhsBatchAxes(), operation.getRhsBatchAxes());
  if (!current || current->results.size() != output->type.getShape().size())
    return failure();
  SmallVector<int64_t> permutation(current->results.size(), -1);
  auto resultAxis = [&](unsigned operand, unsigned axis) {
    return operand ? current->rhsResultAxes[axis] : current->lhsResultAxes[axis];
  };
  auto assign = [&](unsigned result, unsigned selected) {
    if (permutation[result] >= 0 && permutation[result] != selected)
      return failure();
    permutation[result] = selected;
    return success();
  };
  for (auto [index, axis] : llvm::enumerate(original->results)) {
    unsigned operand = axis.operand == ContractionOperand::Rhs;
    auto result = resultAxis(operand, operands[operand].oldToNew[axis.axis]);
    if (!result || failed(assign(*result, output->oldToNew[index]))) return failure();
  }
  for (auto [operand, projected] : llvm::enumerate(operands))
    for (auto [execution, axis] : llvm::enumerate(projected.executionToNew)) {
      auto result = resultAxis(operand, axis);
      if (result && failed(assign(*result, output->executionToNew[execution])))
        return failure();
    }
  llvm::SmallDenseSet<int64_t> assigned;
  for (int64_t axis : permutation)
    if (axis < 0 || !assigned.insert(axis).second) return failure();
  return permutation;
}

} // namespace

LogicalResult rankLiftPointwiseValueGraph(
    func::FuncOp kernel, ArrayRef<MakeRangeOp> liftedRanges,
    llvm::SmallPtrSetImpl<Operation *> &laneReductions,
    llvm::DenseMap<Value, SmallVector<BroadcastOp>> &laneBounds) {
  if (liftedRanges.empty())
    return success();

  SmallVector<Attribute> executionExtents, executionAxes;
  for (MakeRangeOp range : llvm::reverse(liftedRanges)) {
    auto fragment = cast<FragmentType>(range.getResult().getType());
    auto axis = cast<AxisMapAttr>(fragment.getAxisMaps()[0]);
    executionExtents.push_back(fragment.getShape()[0]);
    executionAxes.push_back(AxisMapAttr::get(
        kernel.getContext(), axis.getSourceId(), axis.getSourceAxis(),
        axis.getDimensionId(), executionAxes.size(), axis.getDerived()));
  }
  auto execution = FragmentType::get(
      kernel.getContext(), IndexType::get(kernel.getContext()),
      ArrayAttr::get(kernel.getContext(), executionExtents),
      ArrayAttr::get(kernel.getContext(), executionAxes), 1, 1);
  ExecutionSchema schema(execution);
  auto liftedValueType = [&](Type original) -> Type {
    auto lifted = schema.lift(original);
    return succeeded(lifted) ? *lifted : Type{};
  };
  // Producer types change before their users are visited. Keep only the old
  // typed relations needed to transport those users during this mutation.
  llvm::DenseMap<Operation *, SmallVector<Type>> originalOperands, originalResults;
  kernel.walk([&](Operation *operation) {
    originalOperands.try_emplace(operation, operation->getOperandTypes());
    originalResults.try_emplace(operation, operation->getResultTypes());
  });
  auto remapAxes = [&](Operation *operation) {
    return remapSchemaAxes(operation, originalOperands.lookup(operation),
                           originalResults.lookup(operation));
  };
  std::function<bool(Type)> carriesLiftedAxis = [&](Type type) {
    if (auto fragment = dyn_cast<FragmentType>(type))
      return llvm::any_of(fragment.getAxisMaps(), [&](Attribute attribute) {
        auto axis = cast<AxisMapAttr>(attribute);
        return llvm::any_of(executionAxes, [&](Attribute attribute) {
          auto selected = cast<AxisMapAttr>(attribute);
          return sourceAxisIdentity(axis) == sourceAxisIdentity(selected) &&
                 axis.getDimensionId() == selected.getDimensionId();
        });
      });
    if (auto record = dyn_cast<RecordType>(type))
      return llvm::any_of(record.getFieldTypes(), [&](Attribute field) {
        return carriesLiftedAxis(cast<TypeAttr>(field).getValue());
      });
    return false;
  };

  llvm::SmallDenseSet<Value> liftedValues;
  SmallVector<GatherOp> rankLiftedUnitGathers;
  for (MakeRangeOp range : liftedRanges)
    liftedValues.insert(range.getResult());
  auto dependsOnLiftedAxis = [&](Value value) {
    // A full collective and a scalar consumer may name the same domain.
    // Only SSA dependence on the selected occurrence makes a value lane-varying.
    return liftedValues.contains(value);
  };
  llvm::SmallPtrSet<Operation *, 32> liftedOperations;
  auto liftBoundary = [&](Value value, Type target) -> LogicalResult {
    setPhysicalValueType(value, target);
    liftedValues.insert(value);
    return projectSchemaBoundary(value, [&](Value changed, Type) {
      liftedValues.insert(changed);
    });
  };
  auto rewriteValue = [&](Operation *operation,
                           TypeRange selectedTypes) -> LogicalResult {
    OpBuilder builder(operation);
    IRMapping mapping;
    auto replacements = cloneWithSchema(builder, operation, mapping, selectedTypes);
    if (failed(replacements)) return failure();
    for (auto [original, replacement] : llvm::zip(operation->getResults(), *replacements)) {
      original.replaceAllUsesWith(replacement);
      liftedValues.erase(original);
      liftedValues.insert(replacement);
      if (Operation *producer = replacement.getDefiningOp();
          producer && !originalOperands.count(producer))
        liftedOperations.insert(producer);
    }
    liftedOperations.erase(operation);
    originalOperands.erase(operation);
    originalResults.erase(operation);
    operation->erase();
    return success();
  };
  std::function<WalkResult(Operation *)> liftOperation;
  liftOperation = [&](Operation *operation) {
    if (isa<MakeRangeOp>(operation) || liftedOperations.contains(operation) ||
        operation->getParentOfType<scf::WhileOp>())
      return WalkResult::advance();
    if (!originalOperands.count(operation)) {
      // Schema construction has already closed inserted projections. They
      // forward lane dependence, but have no old attributes to transport.
      liftedOperations.insert(operation);
      for (Value result : operation->getResults())
        if (carriesLiftedAxis(result.getType())) liftedValues.insert(result);
      return WalkResult::advance();
    }
    if (auto loop = dyn_cast<scf::WhileOp>(operation)) {
      bool dependent = false;
      loop->walk([&](Operation *nested) {
        dependent |= llvm::any_of(nested->getOperands(), dependsOnLiftedAxis);
      });
      if (!dependent)
        return WalkResult::advance();
      if (!canPredicateScalarWhile(loop)) {
        loop.emitOpError("cannot predicate the lifted ordered recurrence");
        return WalkResult::interrupt();
      }
      OpBuilder builder(loop);
      Location location = loop.getLoc();
      auto shape = predicateType(execution);
      Value active;
      for (MakeRangeOp range : liftedRanges) {
        auto indexType = execution;
        auto project = [&](Value value, Type type) -> Value {
          return builder.create<BroadcastOp>(location, type, value);
        };
        Value coordinate = project(range.getResult(), indexType);
        Value lower = builder.create<CompareOp>(
            location, shape, coordinate, project(range.getLogicalStart(), indexType),
            ComparePredicate::Ge);
        auto stop = builder.create<BroadcastOp>(location, indexType, range.getLogicalStop());
        laneBounds[range.getResult()].push_back(stop);
        Value upper = builder.create<CompareOp>(
            location, shape, coordinate, stop, ComparePredicate::Lt);
        Value bounded = builder.create<BinaryOp>(
            location, shape, lower, upper, BinaryOperator::LogicalAnd);
        active = active ? Value(builder.create<BinaryOp>(
                              location, shape, active, bounded, BinaryOperator::LogicalAnd))
                        : bounded;
      }
      IRMapping mapping;
      if (failed(clonePredicatedScalarOperation(builder, loop, mapping, active, shape)))
        return WalkResult::interrupt();
      Operation *replacement = mapping.lookup(loop.getResult(0)).getDefiningOp();
      // The cloned any reduction consumes the new lanes; they are not free
      // reduction axes to be lifted again in the next fixed-point iteration.
      replacement->walk([&](Operation *nested) {
        liftedOperations.insert(nested);
        if (isa<ReduceOp>(nested))
          laneReductions.insert(nested);
        for (Value result : nested->getResults())
          if (carriesLiftedAxis(result.getType()))
            liftedValues.insert(result);
      });
      for (Value result : loop.getResults()) {
        result.replaceAllUsesWith(mapping.lookup(result));
      }
      loop->walk([&](Operation *nested) {
        for (Value result : nested->getResults()) liftedValues.erase(result);
        for (Region &region : nested->getRegions())
          for (Block &block : region)
            for (Value argument : block.getArguments()) liftedValues.erase(argument);
        liftedOperations.erase(nested);
        originalOperands.erase(nested);
        originalResults.erase(nested);
      });
      loop.erase();
      return WalkResult::advance();
    }
    bool dependsOnLiftedRange =
        llvm::any_of(operation->getOperands(), [&](Value operand) {
          return dependsOnLiftedAxis(operand);
        });
    if (!dependsOnLiftedRange)
      return WalkResult::advance();
    if (auto yield = dyn_cast<scf::YieldOp>(operation)) {
      if (auto branch = dyn_cast<scf::IfOp>(yield->getParentOp())) {
        for (auto [index, value] : llvm::enumerate(yield.getResults())) {
          if (!dependsOnLiftedAxis(value))
            continue;
          if (failed(liftBoundary(branch.getResult(index), value.getType())))
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      auto loop = dyn_cast<scf::ForOp>(yield->getParentOp());
      if (!loop)
        return WalkResult::interrupt();
      for (auto [index, value] : llvm::enumerate(yield.getResults()))
        if (dependsOnLiftedAxis(value) &&
            failed(liftBoundary(loop.getResult(index), value.getType())))
          return WalkResult::interrupt();
      return WalkResult::advance();
    }
    if (auto loop = dyn_cast<scf::ForOp>(operation)) {
      if (dependsOnLiftedAxis(loop.getLowerBound()) ||
          dependsOnLiftedAxis(loop.getUpperBound()) ||
          dependsOnLiftedAxis(loop.getStep()))
        return WalkResult::interrupt();
      for (auto [index, value] : llvm::enumerate(loop.getInitArgs()))
        if (dependsOnLiftedAxis(value) &&
            failed(liftBoundary(loop.getResult(index), value.getType())))
          return WalkResult::interrupt();
      return WalkResult::advance();
    }
    if (!liftedOperations.insert(operation).second)
      return WalkResult::advance();
    if (operation->getNumResults() == 0)
      return WalkResult::advance();

    if (auto fold = dyn_cast<RegionFoldOp>(operation)) {
      if (!llvm::hasSingleElement(fold.getSummarize()) ||
          !llvm::hasSingleElement(fold.getCombine()))
        return WalkResult::interrupt();
      auto structured = cast<StructuredOpInterface>(fold.getOperation());
      unsigned identityCount = fold.getIdentities().size();
      if (llvm::any_of(fold.getSources(), dependsOnLiftedAxis) ||
          llvm::any_of(fold.getIdentities(), dependsOnLiftedAxis)) {
        fold.emitOpError("rank lifting requires independent sources and identities");
        return WalkResult::interrupt();
      }

      Block &summarize = fold.getSummarize().front();
      bool dependentCapture = false;
      for (auto [capture, formal] : llvm::zip_equal(fold.getCaptures(), structured.getSummarizeCaptures())) {
        if (!dependsOnLiftedAxis(capture)) continue;
        dependentCapture = true;
        auto argument = cast<BlockArgument>(formal);
        argument.setType(capture.getType());
        liftedValues.insert(argument);
      }
      if (!dependentCapture)
        return WalkResult::interrupt();
      for (Operation &nested : llvm::make_early_inc_range(summarize.without_terminator()))
        if (liftOperation(&nested).wasInterrupted())
          return WalkResult::interrupt();
      auto summarizeYield = dyn_cast<YieldOp>(summarize.getTerminator());
      if (!summarizeYield ||
          summarizeYield.getValues().size() != identityCount)
        return WalkResult::interrupt();

      Block &combine = fold.getCombine().front();
      if (failed(verifyStructuredArity(structured))) return WalkResult::interrupt();
      SmallVector<Type> resultTypes;
      SmallVector<bool> dependentResults;
      for (unsigned index = 0; index < identityCount; ++index) {
        Value summary = summarizeYield.getValues()[index];
        resultTypes.push_back(summary.getType());
        dependentResults.push_back(dependsOnLiftedAxis(summary));
      }
      if (!llvm::any_of(dependentResults, [](bool value) { return value; }))
        return WalkResult::interrupt();
      if (failed(closeSchemaBoundary(fold, [&](Value value, Type) {
            liftedValues.insert(value);
          }))) return WalkResult::interrupt();
      for (Operation &nested : llvm::make_early_inc_range(combine.without_terminator()))
        if (liftOperation(&nested).wasInterrupted())
          return WalkResult::interrupt();
      auto combineYield = dyn_cast<YieldOp>(combine.getTerminator());
      if (!combineYield || combineYield.getValues().size() != identityCount)
        return WalkResult::interrupt();
      for (auto [index, value] : llvm::enumerate(combineYield.getValues()))
        if (value.getType() != resultTypes[index])
          return WalkResult::interrupt();
      return WalkResult::advance();
    }

    if (auto contract = dyn_cast<ContractOp>(operation)) {
      bool lhs = dependsOnLiftedAxis(contract.getLhs());
      bool rhs = dependsOnLiftedAxis(contract.getRhs());
      if (lhs == rhs || dependsOnLiftedAxis(contract.getAccumulator())) {
        contract.emitOpError("cannot rank-lift contraction operand relation")
            << "; lhs=" << lhs << "; rhs=" << rhs
            << "; accumulator=" << dependsOnLiftedAxis(contract.getAccumulator());
        return WalkResult::interrupt();
      }
      OpBuilder builder(contract);
      auto previous = cast<FragmentType>(originalResults.lookup(operation)[0]);
      auto projected = schema.project(previous);
      auto order = contractionResultPermutation(
          contract, schema, originalOperands.lookup(operation), previous);
      if (failed(projected) || failed(order)) {
        contract.emitOpError("contraction cannot transport the selected execution axes");
        return WalkResult::interrupt();
      }
      auto prefixed = projected->type;
      auto target = prefixed;
      SmallVector<int64_t> permutation = *order;
      if (llvm::all_of(llvm::enumerate(permutation), [](auto axis) {
            return axis.index() == static_cast<size_t>(axis.value());
          })) permutation.clear();
      if (!permutation.empty()) {
        // Contraction preserves its native free/batch order. Restore the
        // caller's selected execution order with an explicit transpose.
        SmallVector<Attribute> shape, mappings;
        for (auto [axis, original] : llvm::enumerate(permutation)) {
          shape.push_back(prefixed.getShape()[original]);
          auto mapping = cast<AxisMapAttr>(prefixed.getAxisMaps()[original]);
          mappings.push_back(AxisMapAttr::get(
              kernel.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
              mapping.getDimensionId(), axis, mapping.getDerived()));
        }
        target = FragmentType::get(
            kernel.getContext(), prefixed.getElementType(),
            builder.getArrayAttr(shape), builder.getArrayAttr(mappings),
            prefixed.getValidity(), prefixed.getOwner());
      }
      FailureOr<Value> accumulator = projectPhysicalValueToSchema(
          builder, contract.getLoc(), contract.getAccumulator(), target);
      if (failed(accumulator))
        return WalkResult::interrupt();
      contract->setOperand(2, *accumulator);
      contract.getResult().setType(target);
      liftedValues.insert(contract.getResult());
      if (!permutation.empty()) {
        SmallVector<int64_t> inverse(permutation.size());
        for (auto [axis, original] : llvm::enumerate(permutation))
          inverse[original] = axis;
        builder.setInsertionPointAfter(contract);
        auto restored = builder.create<TransposeOp>(
            contract.getLoc(), prefixed, contract.getResult(), inverse);
        contract.getResult().replaceUsesWithIf(restored, [&](OpOperand &use) {
          return use.getOwner() != restored.getOperation();
        });
        liftedOperations.insert(restored);
        liftedValues.insert(restored.getResult());
      }
      return WalkResult::advance();
    }
    if (auto reduce = dyn_cast<ReduceOp>(operation)) {
      bool sourceDepends = llvm::any_of(
          reduce.getSources(),
          dependsOnLiftedAxis);
      if (!sourceDepends)
        return WalkResult::interrupt();
      OpBuilder builder(reduce);
      for (unsigned index = 0; index < reduce.getSources().size(); ++index) {
        Value source = reduce.getSources()[index];
        Type target = liftedValueType(source.getType());
        if (!target) return WalkResult::interrupt();
        FailureOr<Value> projected = projectPhysicalValueToSchema(
            builder, reduce.getLoc(), source, target);
        if (failed(projected)) {
          reduce.emitOpError("source cannot adopt the lifted free axes");
          return WalkResult::interrupt();
        }
        reduce->setOperand(index, *projected);
      }
      if (failed(remapAxes(reduce))) return WalkResult::interrupt();
      auto structured = cast<StructuredOpInterface>(reduce.getOperation());
      if (failed(verifyStructuredArity(structured))) return WalkResult::interrupt();
      Block &combine = reduce.getCombine().front();
      for (auto [index, value] : llvm::enumerate(reduce.getResults())) {
        Type target = liftedValueType(value.getType());
        if (!target) return WalkResult::interrupt();
        value.setType(target);
        structured.getCombineLhs()[index].setType(value.getType());
        structured.getCombineRhs()[index].setType(value.getType());
        liftedValues.insert(structured.getCombineLhs()[index]);
        liftedValues.insert(structured.getCombineRhs()[index]);
        liftedValues.insert(value);
      }
      for (Operation &nested : llvm::make_early_inc_range(combine.without_terminator()))
        if (liftOperation(&nested).wasInterrupted())
          return WalkResult::interrupt();
      return WalkResult::advance();
    }
    if (auto gather = dyn_cast<GatherOp>(operation)) {
      if (dependsOnLiftedAxis(gather.getSource())) {
        if (failed(remapAxes(gather))) return WalkResult::interrupt();
        rankLiftedUnitGathers.push_back(gather);
      }
    }
    if (!isStructuredFreeAxisValueOp(operation)) {
      operation->emitOpError("has no rank-lifting rule for a structured free axis");
      return WalkResult::interrupt();
    }
    SmallVector<Type> selectedTypes;
    for (Value value : operation->getResults()) {
      Type lifted = liftedValueType(value.getType());
      if (!lifted || (lifted == value.getType() &&
          !isa<FragmentType, RecordType>(lifted)))
        return WalkResult::interrupt();
      selectedTypes.push_back(lifted);
    }
    if (isa<SplatOp, BroadcastOp, UnaryOp, BinaryOp, CompareOp, SelectOp,
            CastOp, BitcastOp, MakeRecordOp, ExtractOp>(operation)) {
      if (failed(rewriteValue(operation, selectedTypes))) return WalkResult::interrupt();
    } else {
      for (auto [value, type] : llvm::zip(operation->getResults(), selectedTypes)) {
        value.setType(type);
        liftedValues.insert(value);
      }
      if (isa<TransposeOp, ReshapeOp>(operation) && failed(remapAxes(operation)))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  };
  // A carry can make earlier body operations lane-dependent. Reach a fixed
  // point through SSA uses without changing ordered loop control or traversal.
  size_t previousLiftedCount;
  do {
    previousLiftedCount = liftedValues.size() + liftedOperations.size();
    WalkResult result = kernel.walk(
        [&](Operation *operation) { return liftOperation(operation); });
    if (result.wasInterrupted())
      return failure();
  } while (previousLiftedCount != liftedValues.size() + liftedOperations.size());

  // A carry boundary may project a scalar seed before its producer acquires
  // lanes. Close that changed boundary through the same projection constructor.
  SmallVector<SplatOp> splats;
  kernel.walk([&](SplatOp splat) {
    if (isa<FragmentType>(splat.getValue().getType())) splats.push_back(splat);
  });
  for (SplatOp splat : splats)
    if (failed(rewriteValue(splat, TypeRange{splat.getType()})))
      return splat.emitOpError("lifted scalar seed cannot adopt its selected schema");

  auto scalarIntegerConstant = [](Value value,
                                  int64_t expected) -> bool {
    while (true) {
      if (auto splat = value.getDefiningOp<SplatOp>()) {
        value = splat.getValue();
        continue;
      }
      if (auto broadcast = value.getDefiningOp<BroadcastOp>()) {
        value = broadcast.getValue();
        continue;
      }
      if (auto reshape = value.getDefiningOp<ReshapeOp>()) {
        value = reshape.getValue();
        continue;
      }
      if (auto cast = value.getDefiningOp<CastOp>()) {
        value = cast.getValue();
        continue;
      }
      break;
    }
    auto constant = value.getDefiningOp<arith::ConstantOp>();
    if (auto boolean =
            constant ? dyn_cast<BoolAttr>(constant.getValue()) : BoolAttr())
      return static_cast<int64_t>(boolean.getValue()) == expected;
    auto integer = constant ? dyn_cast<IntegerAttr>(constant.getValue())
                            : IntegerAttr();
    return integer && integer.getInt() == expected;
  };
  // Gathering every logical unit suffix from a rank-lifted value is a typed
  // squeeze of those suffix axes.  Keep that fact in shared IR as reshape so
  // providers do not need a special partial-tile extraction convention.
  for (GatherOp gather : rankLiftedUnitGathers) {
    auto source = dyn_cast<FragmentType>(gather.getSource().getType());
    auto target = dyn_cast<FragmentType>(gather.getResult().getType());
    if (!source || !target) {
      InFlightDiagnostic diagnostic = gather.emitOpError(
          "rank-lifted unit gather has no fragment squeeze relation");
      diagnostic << "; source=" << gather.getSource().getType()
                 << "; result=" << gather.getResult().getType();
      if (gather.getValid()) {
        diagnostic << "; valid=" << gather.getValid().getType();
        if (Operation *producer = gather.getValid().getDefiningOp())
          diagnostic << "; valid_producer=" << producer->getName();
      }
      return failure();
    }
    for (auto [coordinate, sourceAxis] :
         llvm::zip(gather.getCoordinates(), gather.getSourceAxes()))
      if (sourceAxis < 0 ||
          sourceAxis >= static_cast<int64_t>(source.getShape().size()) ||
          !isStaticUnitExtent(source.getShape()[sourceAxis]) ||
          !scalarIntegerConstant(coordinate, 0))
        return gather.emitOpError(
            "rank-lifted unit gather selects a non-unit source axis");
    OpBuilder builder(gather);
    SmallVector<Attribute> groups;
    unsigned resultAxis = 0;
    for (unsigned axis = 0; axis < source.getShape().size(); ++axis) {
      SmallVector<int64_t> retained;
      if (!llvm::is_contained(gather.getSourceAxes(),
                             static_cast<int64_t>(axis))) {
        auto sourceMap = cast<AxisMapAttr>(source.getAxisMaps()[axis]);
        if (resultAxis >= target.getShape().size())
          return gather.emitOpError("unit gather has too many retained axes");
        auto targetMap = cast<AxisMapAttr>(target.getAxisMaps()[resultAxis]);
        if (source.getShape()[axis] != target.getShape()[resultAxis] ||
            !(sourceAxisIdentity(sourceMap) == sourceAxisIdentity(targetMap)) ||
            sourceMap.getDimensionId() != targetMap.getDimensionId())
          return gather.emitOpError(
              "unit gather lost its retained-axis relation");
        retained.push_back(resultAxis++);
      }
      groups.push_back(ReshapeGroupAttr::get(
          kernel.getContext(), builder.getDenseI64ArrayAttr({axis}),
          builder.getDenseI64ArrayAttr(retained)));
    }
    if (resultAxis != target.getShape().size())
      return gather.emitOpError("unit gather has too few retained axes");
    Value replacement = builder.create<ReshapeOp>(
        gather.getLoc(), target, gather.getSource(), builder.getArrayAttr(groups));
    if (gather.getValid() &&
        !scalarIntegerConstant(gather.getValid(), 1))
      replacement = builder.create<SelectOp>(
          gather.getLoc(), target, gather.getValid(), replacement,
          gather.getFill());
    if (Attribute origin = gather->getAttr(originAttr))
      replacement.getDefiningOp()->setAttr(originAttr, origin);
    gather.getResult().replaceAllUsesWith(replacement);
    gather.erase();
  }
  // Ownership queries consume operand relations, not just the lifted result
  // types. Make scalar and coordinate broadcasts explicit before those queries.
  return closeValueRelations(kernel, ValueRelationScope::Pointwise);
}

} // namespace intent::gpu::pointwise
