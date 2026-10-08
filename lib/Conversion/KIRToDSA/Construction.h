#ifndef INTENT_CONVERSION_KIRTODSA_CONSTRUCTION_H
#define INTENT_CONVERSION_KIRTODSA_CONSTRUCTION_H

#include "Intent/Conversion/KIRToDSA/KIRToDSA.h"
#include "Intent/Conversion/IndexedAccess.h"
#include "Intent/Conversion/LogicalShape.h"
#include "Intent/Dialect/Intent/IR/CompileOptions.h"
#include "Intent/Analysis/CanonicalKernel.h"
#include "Intent/Analysis/ContractionAxes.h"
#include "Intent/Analysis/ProductSchema.h"
#include "Intent/Interfaces/StructuredOpInterface.h"
#include "Intent/Analysis/RegionSemantics.h"
#include "Intent/Analysis/OnlineSummary.h"
#include "Intent/Dialect/Intent/IR/IntentOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/ScopeExit.h"
#include <functional>
#include <limits>

namespace intent::kir_to_dsa {
using namespace mlir;

struct Domain {
  Value begin, end, step, extent;
  int64_t dimension;
  std::optional<int64_t> capacity;
};
// An axis keeps its source extent even when only one execution slice is local.
struct LocalAxis { Value extent, begin, count; int64_t capacity; };
using LocalShape = SmallVector<LocalAxis, 2>;
using AxisRequirements = SmallVector<bool, 4>;
struct WorksetTiling {
  DenseMap<Value, AxisRequirements> requirements;
  SmallVector<Operation *> writes;
  unsigned axis = 0;
  Value extentSource;
  unsigned extentAxis = 0;
};
struct TileDomain { Value extent; int64_t capacity; };
struct AffineIndices {
  Value base;
  SmallVector<Value> steps;
};

struct LogicalComponent {
  Value value;
  SmallVector<unsigned, 2> path;
  Type type;
};
SmallVector<LogicalComponent> logicalComponents(ValueRange values);
RankedTensorType logicalTensorType(Value value,
                                  ArrayRef<unsigned> fieldPath = {});

class Construction {
public:
  Construction(ModuleOp original, ModuleOp target, dsa::ConfigurationAttr configuration,
                 DictionaryAttr shapes, DictionaryAttr strides);
  LogicalResult lower(func::FuncOp source);

private:
  // Construction.cpp: operation dispatch and ordered control.
  LogicalResult loop(Location loc, Value begin, Value end, Value step,
                       const std::function<LogicalResult(Value)> &body);
  LogicalResult orderedControl(Operation *operation);
  LogicalResult lowerBlock(Block &block);
  LogicalResult requireFullExtent(Value extent, int64_t dimension);
  LogicalResult lowerOperations(Block &block);
  LogicalResult lowerOperation(Operation *operation);

  // Values.cpp: scalar semantics, logical extents, and product state.
  Value index(Location loc, int64_t n);
  Value add(Location loc, Value a, Value c);
  Value mul(Location loc, Value a, Value c);
  Value sub(Location loc, Value a, Value c);
  Value get(Value source);
  LogicalResult materialize(Operation *op);
  Value asIndex(Value value, Location loc);
  Value boundExtent(Value value, ArrayRef<unsigned> fieldPath, unsigned axis);
  TensorExtentFact knownExtent(Value value, unsigned axis,
                               ArrayRef<unsigned> fieldPath = {});
  Value logicalExtent(Value value, unsigned axis, Location loc,
                      ArrayRef<unsigned> fieldPath = {});
  bool sameIndex(Value a, Value c);
  LogicalResult lowerScalarOperation(Operation *operation);
  FailureOr<Value> scalarOperation(Operation *source, ValueRange operands);
  bool constantTrue(Value value);
  Value scalarCast(Location loc, Value value, Type type);
  SmallVector<Value> flatten(ValueRange inputs);
  void bindProduct(Value original, ValueRange fields);
  FailureOr<SmallVector<Value>> makeSlots(ValueRange sources, Location loc);
  LogicalResult copyTo(Value value, Value slot, Location loc);
  void bindSlots(ValueRange originals, ValueRange slots, Location loc);
  FailureOr<Value> lowerResults(Block &block, ValueRange slots);

  // Tensors.cpp: local storage, projection, and elementwise construction.
  Value allocate(Location loc, Type element, int64_t rows, int64_t columns, int64_t space = dsa::nramSpace);
  Value allocateLike(Location loc, Value input, Type element = {});
  std::optional<LocalAxis> selectedAxis(Value value, unsigned axis,
                                      ArrayRef<unsigned> fieldPath = {});
  Type storageElement(Type type);
  FailureOr<LocalShape> localShape(Value value, Location loc,
                                 ArrayRef<unsigned> fieldPath = {});
  Value allocateTensor(Location loc, Type element, const LocalShape &shape);
  Value projectTensor(Location loc, Value source, Value value,
                      ArrayRef<unsigned> fieldPath = {});
  SmallVector<Value> physicalCoordinates(Location loc, Value buffer, ValueRange coordinates);
  Value loadLocal(Location loc, Value value, ValueRange coordinates);
  void storeLocal(Location loc, Value value, Value buffer, ValueRange coordinates);
  LogicalResult eachElement(Location loc, const LocalShape &shape,
        const std::function<LogicalResult(ValueRange)> &body);
  bool compactPrefix(const LocalShape &shape);
  bool completeShape(const LocalShape &shape);
  Value contiguousView(Location loc, Value source, const LocalShape &shape);
  LogicalResult reshapeTensor(Operation *op, const LocalShape &shape);
  LogicalResult joinTensor(Operation *op, const LocalShape &shape);
  bool canDefer(Operation *op);
  LogicalResult tensorOperation(Operation *op);
  LogicalResult histogram(HistogramOp operation);
  LogicalResult atomic(AtomicRMWOp operation);
  LogicalResult quantize(QuantizeOp operation);
  LogicalResult quantizedDot(QuantizedDotOp operation);
  LogicalResult scaledMatMul(ScaledContractOp operation, const LocalShape &shape);

  // AccessCoordinates.cpp: logical coordinate reification and local geometry.
  IndexTermMaterialization indexMaterialization(Location loc);
  FailureOr<SmallVector<Value>> accessCoordinates(
      const IndexRelationFact &relation, const LocalShape &shape,
      ValueRange coordinates, Location loc);
  FailureOr<AffineIndices> affineAccessTerm(Value source,
      const IndexTermFact &term, unsigned resultRank, Location loc);
  FailureOr<AffineIndices> affineIndex(Value original);
  // Access.cpp: physical tile, gathered-row, and element access selection.
  LogicalResult tensorAccess(Operation *op);
  Value stride(Location loc, Value view, int64_t axis);

  // Contractions.cpp: contraction axes and local matrix products.
  std::optional<ContractionAxes> contractionAxes(ContractOp matrix);
  SmallVector<ContractOp> sharedInputProducts(ContractOp matrix);
  LogicalResult localMatMul(ContractOp matrix, const LocalShape &shape, Value output = {}, bool stream = true);
  LogicalResult localMatrixProduct(ContractOp matrix, const ContractionAxes &axes,
                                  const LocalShape &shape, Value output);

  // Collectives.cpp: typed helper binding, reductions, and scans.
  void bindHelperValue(Value formal, ArrayRef<Value> fields, bool rebase = false, int64_t sourceAxis = -1);
  FailureOr<SmallVector<Value>> helper(Block &block, ArrayRef<SmallVector<Value>> arguments,
                                        unsigned sourceCount = 0, int64_t sourceAxis = -1);
  SmallVector<SmallVector<Value>> splitFields(TypeRange types, ValueRange fields);
  Value collectiveView(Location loc, Value buffer, const LocalShape &shape);
  LogicalResult buildCollectiveHelper(Operation *operation, Block &source,
      TypeRange stateTypes, ValueRange states, ValueRange captures);
  LogicalResult emitSliceReduction(ReduceOp source, ValueRange inputs,
      ValueRange initials, ValueRange captures, ValueRange outputs,
      ArrayRef<int64_t> axes);
  LogicalResult emitScan(ScanOp source, ValueRange inputs, ValueRange initials,
      ValueRange captures, ValueRange outputs, ValueRange finals);
  LogicalResult streamReduction(ReduceOp reduce, unsigned axis,
      int64_t dimension, const LocalShape &shape);
  std::optional<LogicalResult> stageProductReduction(ReduceOp reduce);
  LogicalResult reduceTensor(ReduceOp reduce);
  LogicalResult scanTensor(ScanOp scan);
  bool canStreamScan(ScanOp scan, Block &block);
  LogicalResult streamScan(ScanOp scan, Block &block);

  // Regions.cpp: structured summary traversal and consumer replay.
  FailureOr<Value> partitionQueryAxis(Block &block, RegionFoldOp fold);
  bool replayableSlice(Value value, int64_t dimension, DenseSet<Value> &visited);
  void forgetReplayedTensors(Value value, DenseSet<Value> &visited);
  FailureOr<SmallVector<Operation *>> regionConsumers(Operation *region, unsigned outputCount, int64_t dimension);
  Value regionTraversalEnd(RegionFoldOp fold, Value size, int64_t sourceDimension);
  FailureOr<SmallVector<Value>> jointSummary(RegionFoldOp fold, OnlineSummary plan,
        ArrayRef<SmallVector<Value>> arguments, ArrayRef<Value> state);
  LogicalResult lowerRegion(Operation *op);

  // Worksets.cpp: execution slices, interval binding, and task distribution.
  bool singletonAxis(Value value, unsigned axis);
  bool equalAxisExtent(Value lhs, unsigned a, Value rhs, unsigned c,
                       ArrayRef<unsigned> lhsPath = {},
                       ArrayRef<unsigned> rhsPath = {});
  std::optional<WorksetTiling> planExecutionSlices(Block &block, unsigned selectedAxis = 0,
        ArrayRef<std::pair<Value, unsigned>> sources = {});
  std::optional<TileDomain> sliceDomain(const WorksetTiling &plan,
                                         int64_t capacity, bool clampCapacity);
  void bindExecutionSlice(const WorksetTiling &plan, TileDomain domain,
                            Value begin, Value count);
  void coarsenMatrixWorkset(Block &block, ArrayRef<WorksetTiling> slices,
                            SmallVectorImpl<TileDomain> &domains,
                            bool distribute);
  LogicalResult lowerTiledWorkset(Block &block, const WorksetTiling &plan, bool prepared = false,
        Value selectedOutput = {}, bool distribute = false);
  Value independentRowControl(Block &block);
  LogicalResult lowerStructuredBlock(Block &block);
  bool bindDomain(Value value);
  std::optional<int64_t> upperDistance(Value end, Value begin);
  bool independentDomains(const LogicalWorksetFact &workset);
  LogicalResult distributeWorkset(Location loc);

  CanonicalKernelAnalysis analysis;
  ModuleOp target;
  OpBuilder b;
  dsa::ConfigurationAttr config;
  DictionaryAttr shapeBindings;
  DictionaryAttr strideBindings;
  func::FuncOp function;
  IRMapping values;
  DenseMap<Value, SmallVector<Value>> products;
  DenseMap<Value, SmallVector<Value>> publicExtents;
  DenseMap<Value, Domain> domains;
  Value taskId, taskCount;
  unsigned parallelDepth = 0;
  SmallVector<int64_t> fullExtents;
  DenseSet<Operation *> materializing;
  DenseMap<int64_t, LocalAxis> axisBindings;
  DenseMap<Value, LocalShape> localShapes;
  DenseMap<Value, DenseMap<unsigned, LocalAxis>> valueSlices;
  bool distributedTiles = false;
  DenseSet<Operation *> streamedOperations;
  std::optional<LogicalWorksetFact> distributedWorkset;
  ParallelOp distributedRoot;
};

} // namespace intent::kir_to_dsa

#endif
