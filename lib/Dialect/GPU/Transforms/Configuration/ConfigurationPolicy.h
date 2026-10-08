#ifndef INTENT_GPU_TRANSFORMS_CONFIGURATIONPOLICY_H
#define INTENT_GPU_TRANSFORMS_CONFIGURATIONPOLICY_H

#include "Intent/Dialect/GPU/Analysis/Resources.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/TuningProfiles.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace intent::gpu::configuration {

enum class TuningClass {
  Pointwise, PointwiseReduction, StatefulReduction, OnlineMoment, Reduction,
  MultiAxisReduction, RegionReduction, RegionContraction, Scan, Contraction,
  PersistentContraction, Histogram, Execution
};

struct TuningProfile {
  int64_t ownershipM, ownershipN, reduction, reductionOuter, scan;
  int64_t traversalWorkers, traversalGroup;
};

struct ParameterClassification {
  TuningClass kind;
  unsigned width;
  bool reductionRow;
};

struct CorrelatedProfileParameters {
  ParameterAttr pointwise, reduction, contraction;
};

struct ContractionFreeExtent {
  llvm::SmallVector<PhysicalExprAttr> extents;
  llvm::SmallVector<ParameterAttr> parameters;
  ParameterRole role;
};

// Recomputed for one unchanged current program. These facts select profile
// projections; the resulting configuration set is published only in the IR.
struct ConfigurationFacts {
  llvm::SmallVector<ParameterAttr> parameters;
  llvm::DenseMap<ParameterAttr, ParameterClassification> classifications;
  bool hasTwoAxisPointwiseOwnership = false;
  bool hasFixedPointwiseLocal = false;
  bool pointwiseOnlyProgram = true;
  bool smallRegionRows = false;
  bool multipleRegionMatrixAccumulators = false;
  llvm::DenseMap<ParameterAttr, int64_t> pointwiseLocalMultiplicity;
  llvm::SmallVector<ParameterAttr> pointwiseRowAxes;
  llvm::MapVector<unsigned, llvm::SmallVector<ParameterAttr>> rowGroups;
  llvm::SmallVector<CorrelatedProfileParameters> correlatedProfiles;
  llvm::SmallVector<ContractionFreeExtent> freeExtents;
};

ConfigurationFacts analyzeConfigurationPolicy(
    mlir::func::FuncOp kernel, llvm::ArrayRef<ParameterAttr> parameters,
    const FragmentResourceAnalysis &resources);

using ProfileBindingConsumer =
    llvm::function_ref<void(mlir::NamedAttrList &, const TuningProfile &)>;
mlir::LogicalResult projectConfigurationProfiles(
    mlir::func::FuncOp kernel, const ConfigurationFacts &facts,
    const TuningProfiles &tables, ProfileBindingConsumer consume);
int64_t requestedValue(const TuningProfile &profile, ParameterRole role);
int64_t selectCandidate(llvm::ArrayRef<int64_t> candidates, int64_t requested);

void bindContractionFreeExtents(
    llvm::ArrayRef<ContractionFreeExtent> groups, mlir::NamedAttrList &bindings,
    const TuningProfile &profile, mlir::Builder &builder);
mlir::LogicalResult bindTraversalFragmentFootprints(
    mlir::func::FuncOp kernel, llvm::ArrayRef<ParameterAttr> parameters,
    const FragmentResourceAnalysis &resources,
    mlir::NamedAttrList &bindings, mlir::Builder &builder);

} // namespace intent::gpu::configuration

#endif
