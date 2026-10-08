add_mlir_library(MLIRIntentGPUTransforms
  Access/AccessCoordinates.cpp
  Access/AccessExpressions.cpp
  Access/AccessGatherProjection.cpp
  Access/AccessGathers.cpp
  Access/AccessLoads.cpp
  Access/AccessReductions.cpp
  Access/AccessReshapes.cpp
  Access/RealizeAccessComposition.cpp

  Contraction/ContractionAnalysis.cpp
  Contraction/ContractionBlocking.cpp
  Contraction/ContractionEpilogue.cpp
  Contraction/ContractionProjection.cpp
  Contraction/ContractionSources.cpp
  Contraction/ContractionSnapshots.cpp
  Contraction/ContractionTraversal.cpp
  Contraction/ContractionValues.cpp
  Contraction/RealizeContractionBlocking.cpp
  Contraction/RealizeVectorContractions.cpp

  Pointwise/PointwiseAnalysis.cpp
  Pointwise/PointwiseCoverage.cpp
  Pointwise/PointwiseLifting.cpp
  Pointwise/PointwiseOwnership.cpp
  Pointwise/PointwiseReuse.cpp
  Pointwise/PointwiseTraversal.cpp
  Pointwise/PointwiseWorksets.cpp
  Pointwise/RealizePointwiseBlocking.cpp

  Reduction/CompletedReductions.cpp
  Reduction/OnlineSummary.cpp
  Reduction/RealizeOnlineReduction.cpp
  Reduction/RealizeReductionBlocking.cpp
  Reduction/RealizeReductionConsumers.cpp
  Reduction/RealizeScanConsumers.cpp
  Reduction/ReductionAnalysis.cpp
  Reduction/ReductionChains.cpp
  Reduction/ReductionCoverage.cpp
  Reduction/ReductionDecomposition.cpp
  Reduction/ReductionNormalization.cpp
  Reduction/ReductionParameters.cpp
  Reduction/ReductionTraversal.cpp
  Reduction/ReductionValues.cpp

  Region/RealizeRegionFold.cpp
  Region/RealizeRegionScan.cpp
  Region/RegionCoRealization.cpp
  Region/RegionPredicates.cpp
  Region/RegionSummary.cpp
  Region/RegionCloning.cpp
  Region/RegionScanOutputs.cpp
  Region/RegionSources.cpp

  Control/FuseIndependentTraversals.cpp
  Control/PredicateScalarControl.cpp
  Control/PredicatedCloning.cpp
  Control/Traversal.cpp
  Control/TraversalTails.cpp
  Control/VectorizeBufferLoops.cpp

  Value/EliminateCommonValues.cpp
  Value/ExecutionSchema.cpp
  Value/Helpers.cpp
  Value/PhysicalCloning.cpp
  Value/ReplayMaterialization.cpp
  Value/ReplayPolicy.cpp
  Value/SourceReplay.cpp
  Value/SchemaMutation.cpp
  Value/ScopePlacement.cpp
  Value/SimplifyRangePredicates.cpp
  Value/ValueMaterialization.cpp
  Value/Relations/Access.cpp
  Value/Relations/Contraction.cpp
  Value/Relations/ExtentPropagation.cpp
  Value/Relations/Pointwise.cpp
  Value/Relations/Structured.cpp
  Value/Relations/Worklist.cpp

  Storage/LowerWorkspace.cpp
  Storage/FragmentSnapshot.cpp
  Storage/Workspace.cpp
  Storage/MaterializeRetainedValues.cpp
  Storage/PromoteBufferValues.cpp
  Storage/SchedulePrivateStores.cpp

  Mapping/ExecutionGroups.cpp
  Mapping/ProgramInterface.cpp
  Mapping/RefineProgramMapping.cpp
  Mapping/UniformBranches.cpp

  Configuration/ConfigurationAnalysis.cpp
  Configuration/ConfigurationConstraints.cpp
  Configuration/ConfigurationProfiles.cpp
  Configuration/MaterializeConfigTuples.cpp
  Configuration/PhysicalParameters.cpp
  Configuration/Resources.cpp
  Configuration/TuningProfiles.cpp

  Passes.cpp
  VerifyGPUProgram.cpp

  DEPENDS
  IntentGPUTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRSCFUtils
  MLIRSideEffectInterfaces
  MLIRTransformUtils
  MLIRIntentGPUAnalysis
  MLIRIntentGPUDialect
)
