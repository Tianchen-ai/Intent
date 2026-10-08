#ifndef INTENT_GPU_TRANSFORMS_POINTWISE_H
#define INTENT_GPU_TRANSFORMS_POINTWISE_H

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/PhysicalParameters.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <functional>
#include <optional>

namespace intent::gpu { class ReplayPolicy; }

namespace intent::gpu::pointwise {
using namespace mlir;

enum ContractFreeAxisSide : unsigned {
  ContractFreeAxisNone = 0,
  ContractFreeAxisLhs = 1,
  ContractFreeAxisRhs = 2,
};

struct ContractFreeAxisFacts {
  unsigned sides = ContractFreeAxisNone;
  unsigned matrixSides = ContractFreeAxisNone;
  bool regionContraction = false;
  bool batchedContraction = false;
  unsigned operandElementBitWidth = 0;
};

struct StructuredRangeUses {
  llvm::SmallPtrSet<Operation *, 32> internalTraversalRanges;
  llvm::SmallPtrSet<Operation *, 32> structuredTraversalRanges;
  llvm::SmallPtrSet<Operation *, 32> reductionTraversalRanges;
  llvm::SmallPtrSet<Operation *, 32> reductionFreeRanges;
  llvm::SmallDenseSet<uint64_t> scanSegmentDimensions;
  llvm::SmallDenseSet<PhysicalSourceAxis> scanSegmentSources;
  llvm::SmallPtrSet<Operation *, 32> contractionTraversalRanges;
  llvm::SmallPtrSet<Operation *, 8> contractionOwnedRanges;
  llvm::SmallPtrSet<Operation *, 8> contractionOwnedStores;
};

struct WriteEffectFacts {
  Operation *operation;
  SmallVector<Value> coordinates;
  SmallVector<Value> payloads;
};

struct MappingCoordinates {
  llvm::DenseMap<Attribute, Value> tiles;
  llvm::DenseMap<Attribute, unsigned> axes;
  llvm::DenseMap<unsigned, Attribute> coordinates;
  std::optional<unsigned> reusableUnit;
};

SmallVector<WriteEffectFacts> readWriteEffects(func::FuncOp kernel);
StructuredRangeUses classifyStructuredRanges(
    func::FuncOp kernel, ArrayRef<MakeRangeOp> allRanges,
    const llvm::SmallPtrSetImpl<Operation *> &laneReductions);
FailureOr<Attribute> mappingAxis(func::FuncOp kernel, Attribute attribute);
FailureOr<MappingCoordinates> readMappingCoordinates(func::FuncOp kernel, ExecutionGroupOp mapping);

void inheritRangeAuthority(Operation *target, MakeRangeOp source);
uint32_t physicalElementBitWidth(Type type);
void collectPhysicalDimensions(Type type,
                               llvm::SmallDenseSet<uint64_t> &dimensions);
void collectPartiallyCarriedDimensions(
    Type type, llvm::SmallDenseSet<uint64_t> &dimensions);
Attribute dimensionAxisKey(MLIRContext *context, uint64_t dimension);
Attribute sourceAxisKey(MLIRContext *context, PhysicalSourceAxis source);
bool isSourceAxisKey(Attribute axis);
FailureOr<uint64_t> axisDimension(Attribute axis);
FailureOr<PhysicalSourceAxis> axisSource(Attribute axis);
PhysicalExprAttr expression(MLIRContext *context, PhysicalExprKind kind,
                            int64_t value = 0, StringRef symbol = {});
PhysicalExprAttr binaryExpression(MLIRContext *context, PhysicalExprKind kind,
                                  PhysicalExprAttr lhs,
                                  PhysicalExprAttr rhs);
bool hasCompileTimeExtent(Value value);
FailureOr<Value> dimensionArgument(func::FuncOp kernel, uint64_t dimension);
FailureOr<uint64_t> rangeDimension(MakeRangeOp range);
bool hasAccessDependentSubregionBounds(func::FuncOp kernel, MakeRangeOp range);
FailureOr<uint64_t> ownershipDimension(func::FuncOp kernel,
                                       MakeRangeOp range);
FailureOr<ParameterAttr> queryOwnershipBlockingParameter(func::FuncOp kernel,
                                                       MakeRangeOp range);
FailureOr<Attribute> parameterAxis(ParameterAttr parameter);
PhysicalExprAttr fragmentExtent(ParameterAttr parameter);
LogicalResult bindStructurallyRequiredStaticFragments(func::FuncOp kernel);
bool hasExactStaticFullCoverage(func::FuncOp kernel, Value source,
                                uint64_t axis);
LogicalResult requireFullDimensionCoverage(func::FuncOp kernel, Value source,
                                           uint64_t axis);
bool isZeroScanIdentity(Value value);
FailureOr<int64_t>
exactSubregionStaticBound(const PhysicalRangeFact &ranges,
                          PhysicalSourceAxis source);
FailureOr<int64_t>
exactStaticTraversalExtent(const PhysicalRangeFact &ranges);
LogicalResult requireScanFullCoverage(func::FuncOp kernel, ScanOp scan,
                                      Value source, uint64_t axis);
LogicalResult requireStructuredReductionFullCoverage(func::FuncOp kernel,
                                                      Value source,
                                                      uint64_t axis);
FragmentType predicateType(FragmentType source);
FailureOr<unsigned> tailPredicateAxis(FragmentType target, MakeRangeOp range);
bool hasTailPredicate(
    Value coordinate,
    const llvm::DenseMap<Value, Value> &rangePredicates);
FailureOr<Value> accessValidity(OpBuilder &builder, Location location,
                                ValueRange coordinates,
                                llvm::DenseMap<Value, Value> &rangePredicates,
                                FragmentType valueType, Value existing);
bool hasFullRangeReductionCapture(Value value, MakeRangeOp range,
                                  Operation *anchor);
LogicalResult addTailValidity(func::FuncOp kernel, Operation *scope,
                              llvm::DenseMap<Value, Value> &rangePredicates,
                              bool includeStores);
void collectProducerRanges(Value value, PhysicalSourceAxis source,
                           llvm::SmallPtrSetImpl<Operation *> &ranges);
Type replaceTraversalExtent(Type type, PhysicalSourceAxis source,
                            ArrayRef<int64_t> dimensions,
                            PhysicalExprAttr extent);
bool containsSource(Value value, PhysicalSourceAxis source);
std::optional<int64_t> estimatedFragmentRegisters(func::FuncOp kernel, Value value);
bool containsTraversal(Value value, PhysicalSourceAxis source,
                       ArrayRef<int64_t> dimensions);
void collectTraversalDimensions(Type type, PhysicalSourceAxis source,
                                SmallVectorImpl<int64_t> &dimensions);
FailureOr<FragmentType> coordinateValueSchema(Type element,
                                              ValueRange coordinates);
FailureOr<Value> replayPointwiseValue(OpBuilder &builder, Value value,
                                      PhysicalSourceAxis source,
                                      ArrayRef<int64_t> traversalDimensions,
                                      PhysicalExprAttr blockedExtent,
                                      Value blockedRange, Value blockedValidity,
                                      Operation *insertionAnchor,
                                      IRMapping &mapping,
                                      const ReplayPolicy *policy = nullptr);
bool storeUsesRange(StoreOp store, MakeRangeOp range);
std::optional<int64_t> storeAxisForRange(StoreOp store, MakeRangeOp range);
FailureOr<SmallVector<int64_t>>
traversalDimensionsForStores(ArrayRef<StoreOp> stores, MakeRangeOp range);
FailureOr<int64_t> reuseTraversalDimension(func::FuncOp kernel,
                                           ArrayRef<StoreOp> stores,
                                           MakeRangeOp range);
std::optional<unsigned> repeatedReductionOutputAxis(
    Value value, MakeRangeOp range);
LogicalResult separateReductionOutputOccurrences(func::FuncOp kernel);
LogicalResult realizeReusePointwiseTraversal(func::FuncOp kernel,
                                             MakeRangeOp range,
                                             ArrayRef<StoreOp> stores,
                                             bool effectLocal,
                                             ArrayRef<LoadOp> retainedReads,
                                             llvm::function_ref<void(StoreOp, StoreOp)> replaceStore);
FailureOr<bool> retainCoveredPointwiseGraph(func::FuncOp kernel,
                                           MakeRangeOp range,
                                           ArrayRef<StoreOp> stores);
void collectCoordinateRanges(Value coordinate,
                             llvm::SmallPtrSetImpl<Operation *> &ranges);
void collectStoreRanges(Value value, llvm::SmallPtrSetImpl<Operation *> &ranges,
                        llvm::SmallPtrSetImpl<Operation *> &visited);
HistogramOp histogramSource(Value value);
LogicalResult alignHistogramOutputOwnership(func::FuncOp kernel);
LogicalResult realizeOwnedHistograms(func::FuncOp kernel);
bool isStructuredFreeAxisValueOp(Operation *operation);
bool isStaticUnitExtent(Attribute attribute);
LogicalResult rankLiftPointwiseValueGraph(
    func::FuncOp kernel, ArrayRef<MakeRangeOp> liftedRanges,
    llvm::SmallPtrSetImpl<Operation *> &laneReductions,
    llvm::DenseMap<Value, SmallVector<BroadcastOp>> &laneBounds);
ContractFreeAxisFacts contractFreeAxisFacts(func::FuncOp kernel,
                                            MakeRangeOp range);
unsigned contractFreeAxisSides(func::FuncOp kernel, MakeRangeOp range);

// One transformation's mutation state. No instance or classification survives
// either complete driver; all executable decisions are written into current IR.
class PointwiseRewrite {
public:
  explicit PointwiseRewrite(ModuleOp module, func::FuncOp kernel) : module(module), kernel(kernel) {}
  LogicalResult ownership();
  LogicalResult blocking();
private:
  void deferRange(Operation *range) {
    internalTraversalRanges.insert(range);
    promotedRanges.erase(range);
  }
  void promoteRange(Operation *range) {
    internalTraversalRanges.erase(range);
    promotedRanges.insert(range);
  }
  LogicalResult prepareCoverage();
  LogicalResult liftWorksets();
  LogicalResult finalizeValues();
  LogicalResult finishOwnership();
  LogicalResult finishBlocking();
  void refreshProgramFacts();
  void selectRanges(llvm::function_ref<bool(MakeRangeOp, PhysicalExprAttr)> selectedStatic);
  LogicalResult selectWritebackCandidates(bool accountResourcePressure);
  void filterOwnershipRanges();
  void appendWritebackRanges();
  LogicalResult materializeFixedRanges();
  LogicalResult realizeWritebacks();
  LogicalResult prepareAxisRelations(bool preserveReductionPositions);
  void appendOwnershipOccurrences();
  LogicalResult resolveOwnershipDependencies();
  void promoteOwnershipRanges();
  LogicalResult retainGatherSources();
  LogicalResult coverLocalRanges();
  void selectOwnershipBindings();
  void selectLocalBindings();
  LogicalResult bindAxes();
  LogicalResult chooseOwnership();
  LogicalResult mapOwnership();
  LogicalResult reuseMapping();
  LogicalResult materializeRanges();
  FailureOr<std::optional<unsigned>> reconcileCoordinate(MakeRangeOp range, Attribute axisKey, MappingCoordinates &coordinates);
  bool isContractionOwned(Operation *range) const { return uses.contractionOwnedRanges.contains(range) || forcedContractionRanges.contains(range); }
  bool isReductionTraversal(Operation *range) const { return uses.reductionTraversalRanges.contains(range) || forcedReductionRanges.contains(range); }
  FailureOr<Attribute> effectLocalKey(MakeRangeOp range);
  bool coordinatesUseRange(ValueRange coordinates, MakeRangeOp range);
  bool coordinatesDirectlyUseRange(ValueRange coordinates,
                                         MakeRangeOp range);
  bool contractionFreeAxisNeedsRange(Operation *operation,
                                           MakeRangeOp range);
  bool sharesLogicalTraversal(MakeRangeOp lhs, MakeRangeOp rhs);
  bool hasPointwiseOwnership(MakeRangeOp range);
  ModuleOp module;
  func::FuncOp kernel;
  llvm::SmallPtrSet<Operation *, 4> laneReductions;
  llvm::DenseMap<Value, SmallVector<BroadcastOp>> laneBounds;
  SmallVector<MakeRangeOp> allRanges, dynamicRanges;
  StructuredRangeUses uses;
  SmallVector<WriteEffectFacts> writeEffects;
  SmallVector<SmallVector<Value, 4>> writeCoordinates;
  SmallVector<Operation *> writeOperations;
  // Selected local traversals/occurrences are rewrite state, not cached proofs.
  llvm::SmallPtrSet<Operation *, 32> internalTraversalRanges;
  llvm::SmallPtrSet<Operation *, 32> promotedRanges;
  llvm::SmallPtrSet<Operation *, 16> reuseTraversalRanges, writeTraversalRanges;
  llvm::SmallPtrSet<Operation *, 16> reductionCaptureWritebackRanges, boundedWritebackRanges;
  llvm::SmallPtrSet<Operation *, 16> forcedContractionRanges, forcedReductionRanges;
  llvm::MapVector<Attribute, SmallVector<StoreOp>> effectLocalStores;
  ExecutionGroupOp mapping;
  uint32_t pointwiseElementBitWidth = 0;
  llvm::SmallDenseSet<PhysicalSourceAxis> ownershipSources, directOwnershipSources;
  struct RangeBinding {
    MakeRangeOp range;
    bool requiresTile;
    bool programOwner;
    bool internalOnly = false;
  };
  SmallVector<RangeBinding> bindings;
  llvm::MapVector<Attribute, SmallVector<MakeRangeOp>> axes;
  llvm::DenseMap<Attribute, ParameterRefAttr> parameters;
  llvm::DenseMap<Attribute, PhysicalExprAttr> ownershipExtents;
  llvm::SmallDenseSet<Attribute> ownershipAxes;
  llvm::SmallDenseSet<Attribute> internalAxes;
  llvm::SmallDenseSet<uint64_t> partiallyCarriedStructuredDimensions;
  llvm::DenseMap<uint64_t, ParameterCategory> structuredOwnershipCategories;
  llvm::SmallDenseSet<uint64_t> nonUniqueContractionDimensions;
  llvm::SmallPtrSet<Operation *, 8> retainedCartesianRanges;
  llvm::SmallPtrSet<Operation *, 8> independentContractionRanges;
  llvm::DenseMap<Operation *, MakeRangeOp> occurrenceRoots;
  llvm::DenseMap<Operation *, ParameterRefAttr> occurrenceParameters;
  llvm::SmallPtrSet<Operation *, 8> positionalOccurrences;
  llvm::SmallDenseSet<uint64_t> independentCartesianDimensions;
  llvm::SmallPtrSet<Operation *, 8> dependentResourceRanges;
  llvm::DenseMap<Operation *, int64_t> parentAnchorOffsets;
  llvm::SmallPtrSet<Operation *, 16> retainedGatherRanges;
  llvm::DenseMap<Attribute, CoordinateRole> contractionCoordinateRoles;
  llvm::DenseMap<Attribute, Value> tileCoordinates;
  llvm::DenseMap<Attribute, Value> logicalDimensions;
};

} // namespace intent::gpu::pointwise
#endif
