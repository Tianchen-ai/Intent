add_mlir_library(MLIRIntentCPUTransforms
  Configuration/Configurations.cpp
  Configuration/FinalizedCandidates.cpp
  Configuration/Passes.cpp
  Configuration/TuningProfiles.cpp

  Implementation/GroupQuantizedDots.cpp
  Implementation/Implementation.cpp
  Implementation/ImplementationInputs.cpp
  Implementation/InputCopies.cpp
  Implementation/InputProducers.cpp
  Implementation/InputWindows.cpp
  Implementation/Passes.cpp
  Implementation/ReusePreparedInputs.cpp

  Contraction/BlockContractions.cpp
  Contraction/ContractionEpilogues.cpp
  Contraction/ContractionSources.cpp
  Contraction/NormalizeContractions.cpp
  Contraction/Passes.cpp

  Collective/Histograms.cpp
  Collective/NormalizeReductions.cpp
  Collective/Passes.cpp
  Collective/SliceCollectives.cpp

  Control/Passes.cpp
  Control/TraversalAccesses.cpp
  Control/TraversalFusion.cpp

  Region/GroupRegionComputations.cpp
  Region/Passes.cpp
  Region/RealizeRegions.cpp

  Storage/AccessAliases.cpp
  Storage/Bufferization.cpp
  Storage/CollectiveBufferization.cpp
  Storage/FuseIntermediateBuffers.cpp
  Storage/InvariantMemory.cpp
  Storage/MemoryAccess.cpp
  Storage/MemoryOptimization.cpp
  Storage/MemoryValues.cpp
  Storage/Passes.cpp
  Storage/ReusePrivateStorage.cpp
  Storage/ScratchCapacity.cpp
  Storage/ScratchPlacement.cpp
  Storage/ScratchReuse.cpp

  Structure/ContiguousAccesses.cpp
  Structure/FoldUniformComputations.cpp
  Structure/FuseStructuredComputations.cpp
  Structure/IntegerSources.cpp
  Structure/ProducerReplay.cpp
  Structure/ProducerReuse.cpp
  Structure/ProducerVersions.cpp
  Structure/ParallelReductions.cpp
  Structure/ReductionSources.cpp
  Structure/ReductionSupply.cpp
  Structure/MaterializeStructuredComputations.cpp
  Structure/Passes.cpp

  Task/BlockStructuredComputations.cpp
  Task/AtomicTaskOwnership.cpp
  Task/GroupWorksetComputations.cpp
  Task/PartitionTasks.cpp
  Task/Passes.cpp

  Vector/ContiguousMemory.cpp
  Vector/ProducerVectorization.cpp
  Vector/VectorReductions.cpp
  Vector/VectorizeLoops.cpp

  Passes.cpp

  DEPENDS IntentCPUTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRIntentCPUAnalysis
  MLIRIntentAnalysis
  MLIRIntentCPUDialect
  MLIRAnalysis
  MLIRAffineDialect
  MLIRAffineToStandard
  MLIRAffineUtils
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRBufferizationToMemRef
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRMemRefTransforms
  MLIRSCFDialect
  MLIRSCFUtils
  MLIRTensorDialect
  MLIRTensorTransforms
  MLIRVectorDialect
  MLIRVectorTransforms
  MLIRValueBoundsOpInterface
  MLIRIR
  MLIRPass
  MLIRTransforms
)
