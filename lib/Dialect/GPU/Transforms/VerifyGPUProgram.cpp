#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Analysis/TraversalPartitions.h"
#include "Intent/Dialect/GPU/IR/ProgramInterface.h"
#include "Intent/Dialect/GPU/IR/TypeVerification.h"

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/IR/PhysicalExpressions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;

namespace intent::gpu {
namespace {

LogicalResult verifyExpressionSymbols(Operation *owner,
                                      PhysicalExprAttr expression,
                                      const llvm::StringSet<> &parameters) {
  auto kind = expression.getKind();
  if (kind == PhysicalExprKind::Parameter &&
      !parameters.contains(expression.getParameterReference().getName().getValue()))
    return owner->emitOpError("launch expression references an undeclared physical parameter");
  for (Attribute operand : expression.getOperands())
    if (failed(verifyExpressionSymbols(owner, cast<PhysicalExprAttr>(operand),
                                       parameters)))
      return failure();
  return success();
}

void collectTypeExpressions(Type type,
                            SmallVectorImpl<PhysicalExprAttr> &expressions) {
  ArrayAttr shape;
  if (auto view = dyn_cast<ViewType>(type))
    shape = view.getLayout().getExtents();
  else if (auto fragment = dyn_cast<FragmentType>(type))
    shape = fragment.getShape();
  else if (auto buffer = dyn_cast<BufferType>(type))
    shape = buffer.getShape();
  if (!shape)
    return;
  for (Attribute extent : shape)
    expressions.push_back(cast<PhysicalExprAttr>(extent));
}

bool hasObservableEffect(Operation *operation) {
  auto access = dyn_cast<AccessOpInterface>(operation);
  if (!access || !access.writesMemory()) return false;
  Value resource = access.getAccessResource();
  // Compiler-created scratch writes are not additional author effects. An
  // author's buffer write still carries its origin and participates in coverage.
  if (!operation->hasAttr(originAttr) &&
      (isa<BufferType>(resource.getType()) || isInvocationWorkspace(resource)))
    return false;
  return true;
}

bool mutuallyExclusiveEffects(Operation *lhs, Operation *rhs) {
  // A versioned operation may occur once in each exclusive branch, but never
  // twice on the same dynamic control path.
  for (Region *region = lhs->getParentRegion(); region;
       region = region->getParentRegion()) {
    auto branch = dyn_cast_or_null<scf::IfOp>(region->getParentOp());
    if (!branch || branch.getElseRegion().empty())
      continue;
    Region &other = region == &branch.getThenRegion()
                        ? branch.getElseRegion() : branch.getThenRegion();
    if (other.isAncestor(rhs->getParentRegion()))
      return true;
  }
  return false;
}

bool requiresRangeProvenance(Value coordinate) {
  auto fragment = dyn_cast<FragmentType>(coordinate.getType());
  if (!fragment)
    return false;
  return llvm::any_of(fragment.getShape(), [](Attribute extent) {
    auto expression = cast<PhysicalExprAttr>(extent);
    return expression.getKind() !=
               PhysicalExprKind::Constant ||
           expression.getValue() != 1;
  });
}

LogicalResult verifyStructuredSegment(Operation *operation,
                                      func::FuncOp kernel,
                                      ParameterRefAttr segment) {
  FailureOr<ParameterAttr> declaration =
      queryParameterBySymbol(kernel, segment.getName());
  if (failed(declaration))
    return operation->emitOpError(
        "structured segment does not reference one exact physical parameter declaration");
  return declaration->getRole() ==
                 ParameterRole::ScanChunk
             ? success()
             : operation->emitOpError(
                   "structured segment parameter has the wrong physical role");
}

LogicalResult verifyBufferResources(func::FuncOp kernel,
                                    PhysicalProgramAnalysis &analysis) {
  auto verifyUses = [](Value buffer) -> LogicalResult {
    for (OpOperand &use : buffer.getUses()) {
      Operation *user = use.getOwner();
      if (isa<AssumeInBoundsOp, DimOp>(user))
        continue;
      auto access = dyn_cast<AccessOpInterface>(user);
      if (!access || !access.isMemoryAccess() ||
          &use != &access.getAccessResourceOperand())
        return user->emitOpError("physical buffer use is not an explicit resource access");
    }
    return success();
  };
  auto space = kernel->getAttrOfType<ArrayAttr>(programSpaceAttr);
  bool singleton = llvm::all_of(space, [](Attribute extent) {
    return constantPhysicalExpression(cast<PhysicalExprAttr>(extent)) == 1;
  });
  auto verifyResource = [&](Value resource, Operation *owner) -> LogicalResult {
    if (failed(verifyUses(resource))) return failure();
    if (isInvocationWorkspace(resource) && !singleton &&
        !analysis.hasDisjointWorkspaceSlices(resource))
      return owner->emitOpError(
          "invocation workspace has no proven disjoint program slices");
    return success();
  };
  llvm::DenseSet<uint64_t> instances;
  LogicalResult result = success();
  kernel.walk([&](BufferOp buffer) {
    if (failed(result))
      return WalkResult::interrupt();
    BufferType type = buffer.getResult().getType();
    if (!instances.insert(type.getInstance()).second) {
      buffer.emitOpError("physical buffer instance identity is duplicated");
      result = failure();
      return WalkResult::interrupt();
    }
    if (failed(verifyResource(buffer.getResult(), buffer))) {
      result = failure();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (failed(result))
    return failure();
  for (BlockArgument argument : kernel.getArguments())
    if (isInvocationWorkspace(argument) &&
        failed(verifyResource(argument, kernel)))
      return failure();
  return success();
}

} // namespace

LogicalResult verifyGPUProgram(ModuleOp module) {
  if (failed(verifyGPUTypeInvariants(module.getOperation())))
    return failure();
  if (failed(mlir::verify(module.getOperation())))
    return failure();
  FailureOr<func::FuncOp> physicalKernel = getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  func::FuncOp kernel = *physicalKernel;
  if (!kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr) ||
      !kernel->getAttrOfType<ArrayAttr>(programSpaceAttr) ||
      !kernel->getAttrOfType<IntegerAttr>(gridRankAttr))
    return kernel.emitError("physical kernel is missing capabilities or launch schema");
  auto programSpace = kernel->getAttrOfType<ArrayAttr>(programSpaceAttr);
  if (Attribute attribute = kernel->getAttr(configurationsAttr)) {
    auto configurations = dyn_cast<ConfigurationSetAttr>(attribute);
    if (!configurations)
      return kernel.emitError("candidate bindings require a typed configuration set");
    auto space = ParameterSpace::read(kernel);
    if (failed(space) || failed(space->requirements()))
      return failure();
    // Parameter mutation leaves the requirements intact while discarding stale
    // rows. A consumer requesting a materialized candidate set rejects emptiness.
    if (!configurations.getRows().empty() &&
        failed(space->configurations(configurations.getStage())))
      return failure();
  }
  auto expectedEffects = kernel->getAttrOfType<ArrayAttr>(effectOriginsAttr);
  int64_t gridRank = kernel->getAttrOfType<IntegerAttr>(gridRankAttr).getInt();
  if (gridRank <= 0 || programSpace.size() != static_cast<size_t>(gridRank) ||
      !expectedEffects)
    return kernel.emitError("physical program-space rank is invalid");
  for (Attribute extent : programSpace)
    if (!isa<PhysicalExprAttr>(extent))
      return kernel.emitError("program-space extents must be typed physical expressions");

  llvm::StringSet<> parameterNames;
  if (failed(verifyProgramInterface(kernel))) return failure();
  SmallVector<PhysicalExprAttr> abiExpressions;
  for (Type type : kernel.getArgumentTypes())
    collectTypeExpressions(type, abiExpressions);
  auto declarations = ParameterSpace::read(kernel);
  if (failed(declarations))
    return failure();
  for (ParameterAttr parameter : declarations->declarations()) {
    parameterNames.insert(parameter.getName().getValue());
  }
  for (Attribute extent : programSpace)
    if (failed(verifyExpressionSymbols(kernel, cast<PhysicalExprAttr>(extent),
                                       parameterNames)))
      return failure();
  for (PhysicalExprAttr expression : abiExpressions)
    if (failed(verifyExpressionSymbols(kernel, expression, parameterNames)))
      return failure();

  llvm::DenseSet<int64_t> expectedEffectOrigins;
  for (Attribute origin : expectedEffects) {
    auto node = dyn_cast<IntegerAttr>(origin);
    if (!node || node.getInt() < 0 ||
        !expectedEffectOrigins.insert(node.getInt()).second)
      return kernel.emitError(
          "expected effect origins must be unique non-negative IDs");
  }
  llvm::DenseSet<int64_t> actualEffectOrigins;
  llvm::DenseMap<int64_t, SmallVector<Operation *>> effectDefinitions;
  llvm::DenseMap<int64_t, std::pair<PhysicalExprAttr, PhysicalExprAttr>>
      executionGroups;
  PhysicalProgramAnalysis physicalAnalysis(kernel);
  WalkResult result = kernel.walk([&](Operation *operation) {
    StringRef dialect = operation->getName().getDialectNamespace();
    if (dialect == "intent") {
      operation->emitOpError("canonical KIR is illegal inside a physical GPU kernel");
      return WalkResult::interrupt();
    }
    if (dialect != "intent_gpu" && dialect != "arith" && dialect != "scf" &&
        dialect != "func" && dialect != "builtin") {
      operation->emitOpError("operation dialect is not legal in shared GPU IR");
      return WalkResult::interrupt();
    }
    if (auto physical = dyn_cast<PhysicalExprOp>(operation))
      if (failed(verifyExpressionSymbols(operation, physical.getExpression(),
                                         parameterNames)))
        return WalkResult::interrupt();
    if (operation->hasAttr(sourceSubregionAttr)) {
      auto parent =
          operation->getAttrOfType<IntegerAttr>(sourceSubregionAttr);
      if (!parent || parent.getInt() <= 0 ||
          !isa<RangeOp, MakeRangeOp>(operation)) {
        operation->emitOpError(
            "physical subregion requires a typed coordinate range and parent identity");
        return WalkResult::interrupt();
      }
    }
    if (operation->hasAttr(sourceSubregionBoundAttr)) {
      auto bound =
          operation->getAttrOfType<IntegerAttr>(sourceSubregionBoundAttr);
      if (!operation->hasAttr(sourceSubregionAttr) || !bound ||
          bound.getInt() <= 0) {
        operation->emitOpError(
            "physical subregion bound requires a positive typed subregion relation");
        return WalkResult::interrupt();
      }
    }
    if (auto program = dyn_cast<ProgramIdOp>(operation)) {
      if (program.getAxis() >= static_cast<uint64_t>(gridRank)) {
        operation->emitOpError("program coordinate is outside the current grid");
        return WalkResult::interrupt();
      }
    }
    if (auto group = dyn_cast<ExecutionGroupOp>(operation)) {
      auto offset = group.getSegmentOffset();
      auto length = group.getSegmentLength();
      if (failed(verifyExpressionSymbols(operation, offset, parameterNames)) ||
          failed(verifyExpressionSymbols(operation, length, parameterNames)))
        return WalkResult::interrupt();
      auto [entry, inserted] = executionGroups.try_emplace(
          group.getGroupId(), std::make_pair(offset, length));
      if (!inserted) {
        operation->emitOpError(
            "physical execution group identity is duplicated");
        return WalkResult::interrupt();
      }
    }
    if (auto fold = dyn_cast<RegionFoldOp>(operation)) {
      if (failed(verifyStructuredSegment(operation, kernel, fold.getSegment())))
        return WalkResult::interrupt();
    } else if (auto scan = dyn_cast<RegionScanOp>(operation)) {
      if (failed(verifyStructuredSegment(operation, kernel, scan.getSegment())))
        return WalkResult::interrupt();
    }
    if (operation->hasAttr(independentIterationAttr)) {
      auto loop = dyn_cast<scf::ForOp>(operation);
      if (!loop || loop.getNumRegionIterArgs() ||
          !isa<UnitAttr>(operation->getAttr(independentIterationAttr))) {
        operation->emitOpError(
            "independent iteration requires an explicit loop without carried state");
        return WalkResult::interrupt();
      }
    }
    if (isa<AccessOpInterface>(operation)) {
      PhysicalAccessFootprint footprint = physicalAnalysis.footprint(operation);
      if (footprint.state != PhysicalFactState::Exact) {
        operation->emitOpError(
            "physical access has no exact current-IR footprint");
        return WalkResult::interrupt();
      }
      bool needsRanges = llvm::any_of(footprint.coordinates,
                                      requiresRangeProvenance);
      if (needsRanges && footprint.rangeState != PhysicalFactState::Exact) {
        InFlightDiagnostic diagnostic = operation->emitOpError(
            "non-scalar physical access has unknown range provenance");
        for (Operation *blocker : footprint.blockers)
          diagnostic << "; blocker=" << blocker->getName();
        return WalkResult::interrupt();
      }
      PhysicalAccessBoundsFact bounds =
          physicalAnalysis.accessBounds(operation);
      if (!bounds.isExact()) {
        InFlightDiagnostic diagnostic = operation->emitOpError(
            "physical access is not proven within its resource bounds");
        for (int64_t axis : bounds.unprovenAxes)
          diagnostic << "; unproven_axis=" << axis;
        for (int64_t axis : bounds.missingLowerAxes)
          diagnostic << "; missing_lower_axis=" << axis;
        for (int64_t axis : bounds.missingUpperAxes)
          diagnostic << "; missing_upper_axis=" << axis;
        diagnostic << "; resource=" << footprint.resource.getType();
        for (auto [coordinate, sourceAxis] :
             llvm::zip(footprint.coordinates, footprint.sourceAxes))
          diagnostic << "; coordinate_axis=" << sourceAxis << ":"
                     << coordinate;
        if (footprint.validity)
          diagnostic << "; validity=" << footprint.validity;
        for (Operation *blocker : bounds.blockers)
          diagnostic << "; blocker=" << blocker->getName();
        return WalkResult::interrupt();
      }
    }
    if (hasObservableEffect(operation)) {
      auto origin = operation->getAttrOfType<IntegerAttr>(originAttr);
      if (!origin || !llvm::all_of(effectDefinitions[origin.getInt()],
                                   [&](Operation *previous) {
                                     return mutuallyExclusiveEffects(previous, operation) ||
                                            areDisjointTraversalPartitions(previous, operation, physicalAnalysis);
                                   })) {
        operation->emitOpError(
            "observable effect requires one canonical origin per control path");
        return WalkResult::interrupt();
      }
      actualEffectOrigins.insert(origin.getInt());
      effectDefinitions[origin.getInt()].push_back(operation);
    }
    SmallVector<PhysicalExprAttr> expressions;
    for (Type type : operation->getOperandTypes())
      collectTypeExpressions(type, expressions);
    for (Type type : operation->getResultTypes())
      collectTypeExpressions(type, expressions);
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          collectTypeExpressions(argument.getType(), expressions);
    for (PhysicalExprAttr expression : expressions)
      if (failed(verifyExpressionSymbols(operation, expression, parameterNames)))
        return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();
  if (actualEffectOrigins != expectedEffectOrigins) {
    InFlightDiagnostic diagnostic = kernel.emitError(
        "physical kernel effect coverage is incomplete");
    diagnostic << "; expected_effect_origins=";
    for (int64_t origin : expectedEffectOrigins)
      diagnostic << origin << ",";
    diagnostic << "; actual_effect_origins=";
    for (int64_t origin : actualEffectOrigins)
      diagnostic << origin << ",";
    return failure();
  }
  return verifyBufferResources(kernel, physicalAnalysis);
}

} // namespace intent::gpu
