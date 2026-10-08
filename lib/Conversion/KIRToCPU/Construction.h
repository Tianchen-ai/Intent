#ifndef INTENT_CONVERSION_KIRTOCPU_CONSTRUCTION_H
#define INTENT_CONVERSION_KIRTOCPU_CONSTRUCTION_H

#include "Intent/Analysis/CanonicalKernel.h"
#include "Intent/Conversion/KIRToCPU/KIRToCPU.h"
#include "Intent/Conversion/IndexedAccess.h"
#include "Intent/Dialect/Intent/IR/IntentOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"

namespace intent::kir_to_cpu {
using namespace mlir;

struct Domain {
  Value begin, end, step, extent;
};

// One conversion context owns actual values, product leaves and domains.
// Source operations remain immutable; all lowering writes the physical module.
class Construction {
public:
  Construction(CanonicalKernelAnalysis &analysis, ModuleOp physical,
               CPUEntryLayout entryLayout);
  LogicalResult lower(func::FuncOp source);

private:
  // Construction.cpp
  LogicalResult lowerBlock(Block &block);

  // Values.cpp
  Value constant(Location loc, int64_t value);
  static FailureOr<Value> indexValue(OpBuilder &builder, Value value,
                                     Type logicalType, Location loc);
  Value domainExtent(Value begin, Value end, Value step, Location loc);
  Value lookupLeaf(Value value, ArrayRef<unsigned> fieldPath = {}) const;
  FailureOr<Value> extent(Value value, unsigned axis, Location loc,
                          ArrayRef<unsigned> fieldPath = {});
  FailureOr<SmallVector<Value>> extents(Value value, Location loc,
                                       ArrayRef<unsigned> fieldPath = {});
  static RankedTensorType tensorType(Type type);
  static Type valueType(Type type);
  static Value dimension(OpBuilder &b, Location loc, Value value, int64_t axis);
  static Value extractElement(OpBuilder &b, Location loc, Value value,
                              ValueRange indices);
  Value emptyTensor(RankedTensorType tensor, ArrayRef<Value> sizes, Location loc);
  FailureOr<SmallVector<Value>> emptyResults(ValueRange values, Location loc);
  SmallVector<Value> flattened(Value value);
  SmallVector<Value> flattened(ValueRange inputs);
  void bindProduct(Value original, ValueRange components);
  void bindValues(ValueRange originals, ValueRange components);
  ArrayAttr fieldPaths(TypeRange types);
  SmallVector<AffineMap> pointwiseMaps(ValueRange inputs, int64_t rank);
  Value elementAt(Value input, ValueRange members, OpBuilder &nested, Location loc);
  LogicalResult lower(DimOp op);
  LogicalResult lower(DomainOp op);
  LogicalResult lower(SubregionOp op);
  LogicalResult lower(RegionEndOp op);
  LogicalResult lower(MakeRecordOp op);
  LogicalResult lower(MakeTupleOp op);
  LogicalResult lower(ExtractOp op);

  // Access.cpp
  bool alwaysValid(Operation *operation);
  LogicalResult lower(BufferOp op);
  LogicalResult load(Operation *operation);
  LogicalResult store(Operation *operation);

  // Access/Coordinates.cpp: family spelling for the canonical index reifier.
  IndexTermMaterialization indexMaterialization(OpBuilder &nested, Location loc);
  FailureOr<SmallVector<Value>> indexedCoordinates(
      const IndexRelationFact &fact, ValueRange members,
      OpBuilder &nested, Location loc);
  static bool hasTensorIndices(const IndexRelationFact &fact);

  // Access/Views.cpp: rectangular value slices and memory descriptors.
  FailureOr<Value> indexed(Operation *operation);

  // Access/Elements.cpp: value snapshots and unique writes over result members.
  LogicalResult indexedWrite(Operation *operation);
  FailureOr<Value> indexedRead(Operation *operation,
                               const IndexRelationFact &fact, Value source);

  // Access/Atomics.cpp: ordered atomic observations and collision reduction.
  LogicalResult atomicAccess(Operation *operation);
  LogicalResult scatterReduce(ScatterReduceOp operation);

  // Arithmetic.cpp
  FailureOr<Value> arithmetic(Operation *operation, ValueRange arguments, OpBuilder &builder);
  LogicalResult pointwise(Operation *operation);
  LogicalResult lower(ConstantOp op);

  // Collectives.cpp
  LogicalResult helper(Region &original, Region &target, ValueRange captures = {});
  LogicalResult region(Operation *operation);
  LogicalResult scan(ScanOp operation);
  LogicalResult reduce(ReduceOp operation);
  LogicalResult lower(HistogramOp op);

  // Contractions.cpp
  Value emitContraction(Value lhs, Value rhs, Value destination,
                        ArrayRef<AffineMap> maps, unsigned parallelRank,
                        unsigned reductionRank, Location loc);
  Value matrix(Value lhs, Value rhs, Value destination, Location loc);
  LogicalResult contract(ContractOp operation);
  LogicalResult sparseContract(SparseContractOp operation);
  LogicalResult scaledContract(ScaledContractOp operation);
  LogicalResult lower(QuantizeOp op);
  LogicalResult lower(QuantizedDotOp op);

  // ControlFlow.cpp
  LogicalResult orderedControl(Operation *operation);
  LogicalResult lower(ParallelOp op);

  // Tensor.cpp
  LogicalResult lower(FullOp op);
  LogicalResult lower(IndicesOp op);
  LogicalResult lower(JoinOp op);
  LogicalResult lower(BroadcastOp op);
  LogicalResult tensorShape(Operation *operation);

  LogicalResult lowerOperation(Operation *operation);

  CanonicalKernelAnalysis &analysis;
  ModuleOp module;
  OpBuilder builder;
  CPUEntryLayout entryLayout;
  func::FuncOp function;
  IRMapping values;
  llvm::DenseMap<Value, SmallVector<Value>> products;
  llvm::DenseMap<Value, Domain> domains;
};

} // namespace intent::kir_to_cpu
#endif
