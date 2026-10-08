#include "Pointwise.h"
#include "Intent/Analysis/ControlFlow.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Transforms/Contraction/Contraction.h"
#include "Intent/Dialect/GPU/Transforms/Control/Predication.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueRelations.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;

namespace intent::gpu::pointwise {

static bool isCartesianPointwiseValueOp(Operation *operation) {
  return isa<SplatOp, BroadcastOp, UnaryOp, BinaryOp, CompareOp, SelectOp,
             CastOp, BitcastOp, LoadOp, GatherOp, RandomBitsOp>(operation);
}

bool isStructuredFreeAxisValueOp(Operation *operation) {
  return isCartesianPointwiseValueOp(operation) ||
         isa<ReshapeOp, TransposeOp, ContractOp, ReduceOp, MakeRecordOp,
             ExtractOp>(operation);
}

bool isStaticUnitExtent(Attribute attribute) {
  auto extent = dyn_cast<PhysicalExprAttr>(attribute);
  return extent &&
         extent.getKind() ==
             PhysicalExprKind::Constant &&
         extent.getValue() == 1;
}

static bool analyzeStructuredFreeBlock(Block &block,
                                ArrayRef<unsigned> dependentArguments,
                                SmallVectorImpl<bool> &dependentResults,
                                bool &sawContract) {
  llvm::SmallDenseSet<Value> dependent;
  for (unsigned index : dependentArguments) {
    if (index >= block.getNumArguments())
      return false;
    dependent.insert(block.getArgument(index));
  }
  for (Operation &operation : block) {
    if (auto yield = dyn_cast<YieldOp>(operation)) {
      dependentResults.clear();
      for (Value value : yield.getValues())
        dependentResults.push_back(dependent.contains(value));
      return true;
    }
    bool operationDepends =
        llvm::any_of(operation.getOperands(), [&](Value operand) {
          return dependent.contains(operand);
        });
    if (!operationDepends)
      continue;
    if (auto contract = dyn_cast<ContractOp>(operation)) {
      bool lhs = dependent.contains(contract.getLhs());
      bool rhs = dependent.contains(contract.getRhs());
      if (lhs == rhs || dependent.contains(contract.getAccumulator()))
        return false;
      sawContract = true;
    } else if (auto reduce = dyn_cast<ReduceOp>(operation)) {
      ValueRange sources =
          reduce.getSources();
      auto boundaries = llvm::concat<const Value>(reduce.getIdentities(), reduce.getCaptures());
      if (!llvm::any_of(sources, [&](Value value) {
            return dependent.contains(value);
          }) ||
          llvm::any_of(boundaries, [&](Value value) {
            return dependent.contains(value);
          }))
        return false;
    } else if (auto gather = dyn_cast<GatherOp>(operation)) {
      auto source = dyn_cast<FragmentType>(gather.getSource().getType());
      if (!source || !dependent.contains(gather.getSource()) ||
          llvm::any_of(gather.getCoordinates(), [&](Value value) {
            return dependent.contains(value);
          }))
        return false;
      for (int64_t sourceAxis : gather.getSourceAxes())
        if (sourceAxis < 0 ||
            sourceAxis >= static_cast<int64_t>(source.getShape().size()) ||
            !isStaticUnitExtent(source.getShape()[sourceAxis]))
          return false;
    } else if (!isStructuredFreeAxisValueOp(&operation) ||
               operation.getNumRegions() != 0) {
      return false;
    }
    if (operation.getNumResults() == 0)
      return false;
    for (Value result : operation.getResults())
      dependent.insert(result);
  }
  return false;
}

static bool analyzeStructuredRegionFold(
    RegionFoldOp fold, const std::function<bool(Value)> &depends,
    SmallVectorImpl<bool> &dependentResults, bool &sawContract) {
  if (!llvm::hasSingleElement(fold.getSummarize()) ||
      !llvm::hasSingleElement(fold.getCombine()))
    return false;
  auto structured = cast<StructuredOpInterface>(fold.getOperation());
  unsigned identityCount = fold.getIdentities().size();
  if (llvm::any_of(fold.getSources(), depends) || llvm::any_of(fold.getIdentities(), depends))
    return false;

  SmallVector<unsigned> summarizeArguments;
  for (auto [capture, argument] : llvm::zip_equal(fold.getCaptures(), structured.getSummarizeCaptures()))
    if (depends(capture))
      summarizeArguments.push_back(cast<BlockArgument>(argument).getArgNumber());
  if (summarizeArguments.empty())
    return false;
  if (!analyzeStructuredFreeBlock(fold.getSummarize().front(),
                                  summarizeArguments, dependentResults,
                                  sawContract) ||
      dependentResults.size() != identityCount ||
      !llvm::any_of(dependentResults, [](bool value) { return value; }))
    return false;

  SmallVector<unsigned> combineArguments;
  for (auto [index, dependent] : llvm::enumerate(dependentResults))
    if (dependent) {
      combineArguments.push_back(index);
      combineArguments.push_back(identityCount + index);
    }
  SmallVector<bool> combinedResults;
  if (!analyzeStructuredFreeBlock(fold.getCombine().front(), combineArguments,
                                  combinedResults, sawContract) ||
      combinedResults != dependentResults)
    return false;
  return true;
}

static FailureOr<bool> propagateOrderedCarryDependency(
    Value value, Operation *user, llvm::function_ref<void(Value)> enqueue,
    llvm::SmallPtrSetImpl<Operation *> &loops) {
  auto propagateSlots = [&]() -> FailureOr<bool> {
    for (OpOperand &operand : user->getOpOperands()) {
      if (operand.get() != value) continue;
      auto outgoing = queryControlFlowOutgoing(operand);
      if (!outgoing.complete || outgoing.edges.empty()) return failure();
      for (const ControlFlowEdge &edge : outgoing.edges) {
        if (!edge.target) return failure();
        enqueue(edge.target);
      }
    }
    return true;
  };
  auto propagateWhile = [&](scf::WhileOp loop) -> FailureOr<bool> {
    if (!canPredicateScalarWhile(loop))
      return failure();
    loops.insert(loop);
    // A varying trip count makes even an initially uniform recurrence vary.
    for (Value argument : loop.getBeforeArguments())
      enqueue(argument);
    for (Value argument : loop.getAfterArguments())
      enqueue(argument);
    for (Value result : loop.getResults())
      enqueue(result);
    return true;
  };
  if (auto loop = dyn_cast<scf::WhileOp>(user))
    return propagateWhile(loop);
  if (auto condition = dyn_cast<scf::ConditionOp>(user))
    return propagateWhile(cast<scf::WhileOp>(condition->getParentOp()));
  if (auto yield = dyn_cast<scf::YieldOp>(user)) {
    if (auto loop = dyn_cast<scf::WhileOp>(yield->getParentOp()))
      return propagateWhile(loop);
    if (auto branch = dyn_cast<scf::IfOp>(yield->getParentOp())) {
      if (!isLaunchUniformScalar(branch.getCondition(),
                                 branch->getParentOfType<func::FuncOp>()))
        return failure();
      loops.insert(branch);
      return propagateSlots();
    }
    auto loop = dyn_cast<scf::ForOp>(yield->getParentOp());
    if (!loop)
      return failure();
    loops.insert(loop);
    return propagateSlots();
  }
  if (auto loop = dyn_cast<scf::ForOp>(user)) {
    if (value == loop.getLowerBound() || value == loop.getUpperBound() ||
        value == loop.getStep())
      return failure();
    loops.insert(loop);
    return propagateSlots();
  }
  return false;
}

static bool hasSupportedOrderedBodies(
    const llvm::SmallPtrSetImpl<Operation *> &loops,
    llvm::function_ref<bool(Value)> ownsCoordinate) {
  for (Operation *loop : loops) {
    WalkResult effects = loop->walk([&](Operation *operation) {
      if (isa<scf::ForOp, ReduceOp>(operation))
        return WalkResult::advance();
      if (auto branch = dyn_cast<scf::IfOp>(operation))
        return isLaunchUniformScalar(branch.getCondition(),
                                     branch->getParentOfType<func::FuncOp>())
                   ? WalkResult::advance()
                   : WalkResult::interrupt();
      if (auto whileLoop = dyn_cast<scf::WhileOp>(operation))
        return canPredicateScalarWhile(whileLoop) ? WalkResult::advance()
                                                 : WalkResult::interrupt();
      if (operation->getNumRegions() != 0)
        return WalkResult::interrupt();
      if (auto store = dyn_cast<StoreOp>(operation)) {
        if (llvm::none_of(store.getCoordinates(), ownsCoordinate))
          return WalkResult::interrupt();
        if (auto buffer = dyn_cast<BufferType>(store.getResource().getType());
            buffer && buffer.getScope().getValue() !=
                          BufferScope::InvocationWorkspace)
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      return isa<scf::YieldOp, scf::ConditionOp, LoadOp>(operation) ||
                     isMemoryEffectFree(operation)
                 ? WalkResult::advance()
                 : WalkResult::interrupt();
    });
    if (effects.wasInterrupted())
      return false;
  }
  return true;
}

static bool supportsCartesianPointwiseValueGraph(
    ArrayRef<WorksetCoordinateOp> coordinates, bool allowOrderedLoops = false) {
  llvm::SmallDenseSet<Value> dependent;
  SmallVector<Value> worklist;
  auto enqueue = [&](Value value) {
    if (dependent.insert(value).second)
      worklist.push_back(value);
  };
  for (WorksetCoordinateOp coordinate : coordinates) {
    enqueue(coordinate.getResult());
  }
  llvm::SmallPtrSet<Operation *, 32> visited;
  llvm::SmallPtrSet<Operation *, 8> loops;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    for (Operation *user : value.getUsers()) {
      if (allowOrderedLoops) {
        FailureOr<bool> propagated =
            propagateOrderedCarryDependency(value, user, enqueue, loops);
        if (failed(propagated))
          return false;
        if (*propagated)
          continue;
      }
      if (!visited.insert(user).second)
        continue;
      if (user->getNumResults() == 0) {
        // A terminal pointwise store can consume the promoted fragment.  A
        // control-flow terminator or effectful region cannot: promoting the
        // predicate/carry would change scalar program structure into a lane
        // program without a physical control-flow realization.
        auto store = dyn_cast<StoreOp>(user);
        if (!store || user->getNumRegions() != 0)
          return false;
        // Coarsening program ownership does not replicate local allocation
        // instances. Keep those stores scalar until storage is made explicit.
        if (auto buffer = dyn_cast<BufferType>(store.getResource().getType());
            buffer && buffer.getScope().getValue() !=
                          BufferScope::InvocationWorkspace)
          return false;
        continue;
      }
      if (!isCartesianPointwiseValueOp(user) || user->getNumRegions() != 0)
        return false;
      for (Value result : user->getResults()) {
        Type type = result.getType();
        if (!isa<IntegerType, FloatType, IndexType, FragmentType>(type))
          return false;
        enqueue(result);
      }
    }
  }
  return hasSupportedOrderedBodies(loops, [&](Value value) {
    return dependent.contains(value);
  });
}

static bool supportsStructuredFreeAxisValueGraph(
    ArrayRef<WorksetCoordinateOp> coordinates,
    llvm::SmallPtrSetImpl<Operation *> *ownedStores = nullptr,
    llvm::DenseMap<Operation *, bool> *contractSides = nullptr,
    bool *containsContraction = nullptr) {
  if (coordinates.size() > 1) {
    llvm::SmallPtrSet<Operation *, 4> stores;
    for (auto [index, coordinate] : llvm::enumerate(coordinates)) {
      llvm::SmallPtrSet<Operation *, 4> currentStores;
      if (!supportsStructuredFreeAxisValueGraph({coordinate}, &currentStores))
        return false;
      if (index == 0)
        stores.insert(currentStores.begin(), currentStores.end());
      else if (stores.size() != currentStores.size() ||
               !llvm::all_of(stores, [&](Operation *store) {
                 return currentStores.contains(store);
               }))
        return false;
    }
  }
  llvm::SmallDenseSet<Value> dependent;
  SmallVector<Value> worklist;
  auto enqueue = [&](Value value) {
    if (dependent.insert(value).second)
      worklist.push_back(value);
  };
  for (WorksetCoordinateOp coordinate : coordinates)
    enqueue(coordinate.getResult());
  llvm::SmallPtrSet<Operation *, 32> visited;
  llvm::SmallPtrSet<Operation *, 8> loops;
  SmallVector<Operation *> operations;
  bool sawNestedContract = false;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    for (Operation *user : value.getUsers()) {
      FailureOr<bool> propagated =
          propagateOrderedCarryDependency(value, user, enqueue, loops);
      if (failed(propagated))
        return false;
      if (*propagated)
        continue;
      if (!visited.insert(user).second)
        continue;
      operations.push_back(user);
      if (user->getNumResults() == 0) {
        auto store = dyn_cast<StoreOp>(user);
        if (!store || user->getNumRegions() != 0)
          return false;
        if (auto buffer = dyn_cast<BufferType>(store.getResource().getType());
            buffer && buffer.getScope().getValue() !=
                          BufferScope::InvocationWorkspace)
          return false;
        continue;
      }
      if (auto fold = dyn_cast<RegionFoldOp>(user)) {
        SmallVector<bool> resultDependencies;
        if (!analyzeStructuredRegionFold(
                fold,
                [&](Value operand) { return dependent.contains(operand); },
                resultDependencies, sawNestedContract))
          return false;
        for (auto [result, resultDepends] :
             llvm::zip(fold.getResults(), resultDependencies))
          if (resultDepends && dependent.insert(result).second)
            worklist.push_back(result);
        continue;
      }
      if (!isStructuredFreeAxisValueOp(user))
        return false;
      for (Value result : user->getResults())
        if (dependent.insert(result).second)
          worklist.push_back(result);
    }
  }

  auto depends = [&](Value value) { return dependent.contains(value); };
  bool sawContract = sawNestedContract;
  bool sawReduction = false;
  bool sawOwnedStore = false;
  for (Operation *operation : operations) {
    if (auto fold = dyn_cast<RegionFoldOp>(operation)) {
      // Another producer path may reach this fold after its first visit. Check
      // the completed dependence set before committing any ownership rewrite.
      SmallVector<bool> resultDependencies;
      if (!analyzeStructuredRegionFold(fold, depends, resultDependencies,
                                       sawContract))
        return false;
      continue;
    }
    if (auto contract = dyn_cast<ContractOp>(operation)) {
      bool lhs = depends(contract.getLhs());
      bool rhs = depends(contract.getRhs());
      if (lhs == rhs || depends(contract.getAccumulator()))
        return false;
      if (contractSides)
        (*contractSides)[contract] = lhs;
      sawContract = true;
      continue;
    }
    if (auto reduce = dyn_cast<ReduceOp>(operation)) {
      ValueRange sources =
          reduce.getSources();
      auto boundaries = llvm::concat<const Value>(reduce.getIdentities(), reduce.getCaptures());
      if (!llvm::any_of(sources, depends) || llvm::any_of(boundaries, depends))
        return false;
      sawReduction = true;
      continue;
    }
    if (auto gather = dyn_cast<GatherOp>(operation)) {
      auto source = dyn_cast<FragmentType>(gather.getSource().getType());
      if (!source || !depends(gather.getSource()) ||
          llvm::any_of(gather.getCoordinates(), depends))
        return false;
      for (int64_t sourceAxis : gather.getSourceAxes())
        if (sourceAxis < 0 ||
            sourceAxis >= static_cast<int64_t>(source.getShape().size()) ||
            !isStaticUnitExtent(source.getShape()[sourceAxis]))
          return false;
      continue;
    }
    if (auto store = dyn_cast<StoreOp>(operation)) {
      bool ownedCoordinate = llvm::any_of(store.getCoordinates(), depends);
      if (!ownedCoordinate || !depends(store.getValue()))
        return false;
      sawOwnedStore = true;
      if (ownedStores)
        ownedStores->insert(store.getOperation());
    }
  }
  if (containsContraction)
    *containsContraction = sawContract;
  return (sawContract || sawReduction) && sawOwnedStore &&
         hasSupportedOrderedBodies(loops, depends);
}

static SmallVector<WorksetCoordinateOp> orthogonalContractCoordinates(
    ArrayRef<WorksetCoordinateOp> coordinates) {
  for (auto [index, coordinate] : llvm::enumerate(llvm::reverse(coordinates))) {
    WorksetCoordinateOp first = coordinate;
    llvm::SmallPtrSet<Operation *, 4> firstStores;
    llvm::DenseMap<Operation *, bool> firstSides;
    if (!supportsStructuredFreeAxisValueGraph({first}, &firstStores, &firstSides) ||
        firstSides.empty() ||
        !llvm::all_of(firstSides, [](const auto &entry) {
          auto contract = cast<ContractOp>(entry.first);
          return llvm::all_of(contract.getResult().getType().getShape(),
                              isStaticUnitExtent) &&
                 contract.getLhsReductionAxes().size() == 1 &&
                 contract.getRhsReductionAxes().size() == 1 &&
                 contract.getLhsBatchAxes().empty() &&
                 contract.getRhsBatchAxes().empty();
        }))
      continue;
    bool firstIsLhs = firstSides.begin()->second;
    if (!llvm::all_of(firstSides, [&](const auto &entry) {
          return entry.second == firstIsLhs;
        }))
      continue;
    for (WorksetCoordinateOp second :
         llvm::reverse(coordinates.drop_back(index + 1))) {
      if (first.getSourceId() == second.getSourceId() &&
          first.getSourceAxis() == second.getSourceAxis())
        continue;
      llvm::SmallPtrSet<Operation *, 4> secondStores;
      llvm::DenseMap<Operation *, bool> secondSides;
      if (!supportsStructuredFreeAxisValueGraph(
              {second}, &secondStores, &secondSides) ||
          firstStores.size() != secondStores.size() ||
          !llvm::all_of(firstStores, [&](Operation *store) {
            return secondStores.contains(store);
          }) ||
          firstSides.size() != secondSides.size() ||
          !llvm::all_of(firstSides, [&](const auto &entry) {
            auto side = secondSides.find(entry.first);
            return side != secondSides.end() && side->second != entry.second;
          }))
        continue;
      // Each lift prepends its free axis. Lift RHS first to retain the
      // contraction result order [lhs free axes, rhs free axes].
      return firstIsLhs ? SmallVector<WorksetCoordinateOp>{second, first}
                        : SmallVector<WorksetCoordinateOp>{first, second};
    }
  }
  return {};
}

LogicalResult PointwiseRewrite::liftWorksets() {
  if (failed(separateReductionOutputOccurrences(kernel))) return failure();
  SmallVector<WorksetCoordinateOp> pointwiseCoordinates;
  kernel.walk([&](WorksetCoordinateOp coordinate) {
    pointwiseCoordinates.push_back(coordinate);
  });
  SmallVector<MakeRangeOp> existingRanges;
  kernel.walk([&](MakeRangeOp range) { existingRanges.push_back(range); });
  SmallVector<WorksetCoordinateOp> lifted;
  bool liftSeparately = false;
  bool preserveContractionOrder = false;
  if (existingRanges.empty()) {
    if (supportsCartesianPointwiseValueGraph(pointwiseCoordinates, true))
      lifted.append(pointwiseCoordinates.begin(), pointwiseCoordinates.end());
    else {
      // A coordinate used by ordered control can stay scalar without
      // excluding other independent coordinates from the ownership tile.
      for (WorksetCoordinateOp coordinate : llvm::reverse(pointwiseCoordinates)) {
        SmallVector<WorksetCoordinateOp> candidates(lifted);
        candidates.insert(candidates.begin(), coordinate);
        if (supportsCartesianPointwiseValueGraph(candidates, true))
          lifted = std::move(candidates);
      }
    }
  } else {
    SmallVector<std::pair<int64_t, WorksetCoordinateOp>> uncovered;
    for (WorksetCoordinateOp coordinate : pointwiseCoordinates) {
      PhysicalSourceAxis source{coordinate.getSourceId(),
                                coordinate.getSourceAxis(), false};
      bool covered = llvm::any_of(existingRanges, [&](MakeRangeOp range) {
        // A collective range over the same domain does not represent this
        // scalar workset occurrence or its independent output coordinates.
        return sourceAxisIdentity(range) == source &&
               range->hasAttr(worksetCoordinateRangeAttr) &&
               range.getStart() == coordinate.getResult();
      });
      auto worksetAxis =
          coordinate->getAttrOfType<IntegerAttr>(worksetAxisAttr);
      if (!covered && worksetAxis && worksetAxis.getInt() >= 0)
        uncovered.emplace_back(worksetAxis.getInt(), coordinate);
    }
    llvm::stable_sort(uncovered, [](const auto &lhs, const auto &rhs) {
      return lhs.first < rhs.first;
    });
    SmallVector<WorksetCoordinateOp> uncoveredCoordinates;
    for (auto [_, coordinate] : uncovered)
      uncoveredCoordinates.push_back(coordinate);
    lifted = orthogonalContractCoordinates(uncoveredCoordinates);
    liftSeparately = !lifted.empty();
    // Independent reductions may expose reuse on an outer output axis. Keep
    // all proven free axes available for the later ownership choice instead
    // of fixing the choice to the two innermost source loops.
    if (lifted.empty() && uncoveredCoordinates.size() >= 2) {
      llvm::SmallDenseSet<PhysicalSourceAxis> sources;
      bool distinctSources =
          llvm::all_of(uncoveredCoordinates, [&](auto coordinate) {
            return sources.insert({coordinate.getSourceId(),
                                   coordinate.getSourceAxis(), false}).second;
          });
      bool containsContraction = false;
      if (distinctSources &&
          supportsStructuredFreeAxisValueGraph(uncoveredCoordinates, nullptr,
                                              nullptr, &containsContraction) &&
          !containsContraction) {
        lifted = uncoveredCoordinates;
        preserveContractionOrder = true;
      }
    }
    // A blocked pointwise program can keep one existing local vector range
    // while tiling the two innermost independent workset axes (for example
    // sequence/head around a feature vector).  Lift only when both axes
    // exist: a lone workset coordinate should remain the scalar program owner
    // of the existing local range.
    if (lifted.empty() && uncovered.size() >= 2) {
      SmallVector<WorksetCoordinateOp> candidates{
          uncovered[uncovered.size() - 2].second,
          uncovered.back().second};
      llvm::DenseMap<Operation *, bool> contractSides;
      if (supportsCartesianPointwiseValueGraph(candidates)) {
        lifted = std::move(candidates);
      } else if (supportsStructuredFreeAxisValueGraph(
                     candidates, nullptr, &contractSides)) {
        lifted = std::move(candidates);
        preserveContractionOrder = !contractSides.empty();
      }
    }
    if (lifted.empty())
      for (auto [_, coordinate] : llvm::reverse(uncovered))
        if (supportsStructuredFreeAxisValueGraph({coordinate}) ||
            supportsCartesianPointwiseValueGraph({coordinate}, true)) {
          lifted.push_back(coordinate);
          break;
        }
  }

  SmallVector<MakeRangeOp> liftedRanges;
  for (WorksetCoordinateOp coordinate : lifted) {
    OpBuilder builder(coordinate);
    builder.setInsertionPointAfter(coordinate);
    Value extent = builder.create<arith::ConstantIndexOp>(coordinate.getLoc(), 1);
    Value step = coordinate.getStep();
    PhysicalExprAttr unit = expression(module.getContext(),
                                       PhysicalExprKind::Constant, 1);
    int64_t dimension = coordinate.getDimensionId();
    if (dimension <= 0)
      return coordinate.emitOpError(
          "workset coordinate has no logical dimension identity");
    auto type = FragmentType::get(
        module.getContext(), builder.getIndexType(), builder.getArrayAttr({unit}),
        builder.getArrayAttr({AxisMapAttr::get(
            module.getContext(), coordinate.getSourceId(),
            coordinate.getSourceAxis(), dimension,
            /*fragmentAxis=*/0, /*derived=*/false)}),
        /*validity=*/1, /*owner=*/1);
    Value logicalStop = builder.create<BinaryOp>(
        coordinate.getLoc(), builder.getIndexType(), coordinate.getResult(),
        step, BinaryOperator::Add);
    auto range = builder.create<MakeRangeOp>(
        coordinate.getLoc(), type, coordinate.getResult(), extent, step,
        coordinate.getResult(), logicalStop,
        coordinate.getSourceId(), coordinate.getSourceAxis(),
        /*derived=*/false);
    range->setAttr(worksetCoordinateRangeAttr, builder.getUnitAttr());
    Operation *logicalStopProducer = logicalStop.getDefiningOp();
    coordinate.getResult().replaceUsesWithIf(
        range.getResult(), [&](OpOperand &use) {
          return use.getOwner() != range.getOperation() &&
                 use.getOwner() != logicalStopProducer;
        });
    liftedRanges.push_back(range);
    if (liftSeparately) {
      if (failed(rankLiftPointwiseValueGraph(kernel, {range}, laneReductions,
                                             laneBounds)))
        return kernel.emitError(
            "failed to rank-lift an independent contraction axis");
      liftedRanges.clear();
    }
  }
  // Native matrix forms use the last free operand axis for M/N. Preserve
  // logical nesting on a proven free side so the innermost workset axis,
  // rather than an outer independent axis, occupies that matrix dimension.
  if (preserveContractionOrder)
    std::reverse(liftedRanges.begin(), liftedRanges.end());
  if (failed(rankLiftPointwiseValueGraph(kernel, liftedRanges, laneReductions,
                                         laneBounds)))
    return kernel.emitError(
        "failed to rank-lift a legal pointwise ownership graph");
  // Newly explicit free axes must reach the existing contraction recognizer
  // before ownership freezes any axis to scalar grid execution.
  if (!lifted.empty() && failed(contraction::fuseMultiplyReductions(module)))
    return failure();
  return success();
}


} // namespace intent::gpu::pointwise
