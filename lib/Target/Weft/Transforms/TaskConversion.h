#ifndef INTENT_TARGET_WEFT_TRANSFORMS_TASKCONVERSION_H
#define INTENT_TARGET_WEFT_TRANSFORMS_TASKCONVERSION_H

#include "TaskLowering.h"
#include "Intent/Dialect/CPU/Analysis/AxisRelations.h"
#include "Intent/Dialect/CPU/Analysis/ExtentRelations.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Intent/Dialect/CPU/Transforms/Implementation/Implementation.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseSet.h"

namespace intent::weft_provider {

namespace task_detail {
llvm::SmallVector<int64_t> shape(mlir::Type type);
llvm::SmallVector<int64_t> axes(mlir::Type type);
mlir::Type element(mlir::Type type);
} // namespace task_detail

struct NativeReduction;

// One current task conversion owns its value, view, storage and control bindings.
class TaskConversion {
public:
  TaskConversion(mlir::func::FuncOp function, mlir::ModuleOp output,
                 const llvm::DenseMap<mlir::Value, intent::QuantFormat> &formats,
                 const cpu::ImplementationRegistry &implementations);
  mlir::LogicalResult normalizeComputations();
  llvm::SmallVector<mlir::Value>
  shapeArguments(cpu::TasksOp task, mlir::OpBuilder &builder);
  mlir::LogicalResult lower(cpu::TasksOp tasks, llvm::StringRef name,
                            llvm::SmallVectorImpl<unsigned> &argumentPositions);

private:
  struct LocalValue {
    mlir::Value value;
    llvm::SmallVector<mlir::OpFoldResult> sizes;
  };
  struct LocalProjection {
    mlir::Type type;
    llvm::SmallVector<mlir::Value> offsets;
    llvm::SmallVector<mlir::Attribute> selectors;
  };
  class Expansion;

  mlir::LogicalResult block(mlir::Block &source);
  mlir::LogicalResult lower(mlir::Operation *operation);

  int64_t physicalAxis(int64_t axis);
  void initializeAxes();
  llvm::SmallVector<int64_t> memoryAxes(mlir::Value memory);
  mlir::DenseI64ArrayAttr array(llvm::ArrayRef<int64_t> entries);
  ::weft::kernel::EncodingType dense(mlir::Type type);
  ::weft::kernel::EncodingType encoding(mlir::Value memory);
  ::weft::kernel::SliceType sliceType(mlir::Value base,
                                     llvm::ArrayRef<int64_t> dimensions,
                                     llvm::ArrayRef<int64_t> ids);
  std::optional<intent::QuantFormat> quantizedFormat(mlir::Value memory);
  ::weft::kernel::ViewType scalarView(mlir::Type type);
  mlir::Type valueType(mlir::Type element, llvm::ArrayRef<int64_t> dimensions,
                       llvm::ArrayRef<int64_t> ids);
  mlir::Value index(mlir::Location loc, int64_t value);
  mlir::FailureOr<mlir::Type> resultType(mlir::Value memory, mlir::Type scalar = {});
  bool sameBound(mlir::OpFoldResult lhs, mlir::OpFoldResult rhs);
  std::optional<int64_t> nativeExtent(const cpu::ExtentExpression &extent);
  std::optional<int64_t> nativeExtent(mlir::OpFoldResult extent);
  std::optional<int64_t> dimensionExtent(mlir::Value memory, unsigned axis);
  std::optional<int64_t>
  shapeSymbol(const mlir::ValueBoundsConstraintSet::Variable &extent);
  std::optional<int64_t>
  capturedShapeSymbol(const mlir::ValueBoundsConstraintSet::Variable &extent);
  mlir::FailureOr<mlir::Value> view(mlir::Value memory);
  mlir::FailureOr<mlir::Value>
  reshapeViewValue(mlir::Value memory, mlir::Value value, mlir::Type target,
                   llvm::SmallVector<int64_t> order);
  mlir::FailureOr<mlir::Value>
  projectViewValue(mlir::Value memory, mlir::Value value, mlir::Type nativeType,
                   bool inverse = false);
  llvm::SmallVector<mlir::Value>
  projectIndices(mlir::Value memory, mlir::ValueRange indices, unsigned nativeRank,
                 const mlir::IRMapping &mapping);
  mlir::FailureOr<mlir::Value> dimension(mlir::Value memory, unsigned axis);
  mlir::LogicalResult lower(mlir::memref::ExtractStridedMetadataOp metadata);
  mlir::LogicalResult lower(mlir::memref::DimOp dim);

  mlir::Value localRoot(mlir::Value memory);
  bool sameStorageShape(mlir::Value first, mlir::Value second);
  mlir::Value fullLocalOwner(mlir::Value memory);
  bool isLocal(mlir::Value memory);
  mlir::LogicalResult allocateLocal(mlir::Value memory);
  mlir::FailureOr<mlir::Value> readNative(mlir::Value memory);
  mlir::FailureOr<mlir::Value> read(mlir::Value memory);
  mlir::FailureOr<mlir::Value> readNamedAxes(mlir::Value memory);
  llvm::SmallVector<mlir::Value>
  projectionIndices(const LocalProjection &projection, mlir::Location loc);
  mlir::FailureOr<LocalProjection>
  localProjection(mlir::Value memory, const LocalValue &state);
  mlir::FailureOr<llvm::SmallVector<mlir::Value>>
  localIndices(mlir::Value memory, mlir::ValueRange indices,
               const LocalValue &state, const mlir::IRMapping &mapping);
  mlir::FailureOr<mlir::Value> alignValue(mlir::Value value, mlir::Type target,
                                        mlir::Location loc);
  mlir::LogicalResult materializeLocal(mlir::Value root);
  mlir::Value fillLocal(mlir::Value owner, mlir::Value scalar, mlir::Location loc,
                        const LocalProjection *projection = nullptr);
  mlir::LogicalResult
  write(mlir::Value memory, mlir::Value value,
        llvm::SmallVector<mlir::OpFoldResult> sizes = {});
  mlir::LogicalResult lower(mlir::memref::CopyOp copy);
  mlir::LogicalResult lower(mlir::linalg::FillOp fill);
  mlir::LogicalResult lower(mlir::memref::StoreOp store);
  mlir::LogicalResult lower(mlir::memref::LoadOp load);
  mlir::LogicalResult lower(cpu::AtomicRMWOp atomic);
  mlir::LogicalResult lower(mlir::bufferization::DeallocOp release);
  mlir::FailureOr<mlir::Value> indexedRead(mlir::memref::LoadOp load,
                                         const mlir::IRMapping &mapping);

  mlir::FailureOr<bool> expandSelected(mlir::Operation *operation);
  mlir::FailureOr<mlir::Value> binary(mlir::Location loc, mlir::Value lhs,
                                    mlir::Value rhs, llvm::StringRef kind);
  mlir::FailureOr<mlir::Value> expression(mlir::Operation *operation,
                                        mlir::IRMapping &mapping);
  mlir::FailureOr<mlir::Value>
  mappedInput(mlir::Value input, mlir::AffineMap map,
               llvm::ArrayRef<int64_t> loopAxes, bool namedAxes = false);
  mlir::FailureOr<mlir::Value> reduceValue(mlir::Location loc, mlir::Value input,
                                         unsigned axis, llvm::StringRef kind);
  mlir::FailureOr<mlir::Value>
  reductionContribution(mlir::Block &body, const NativeReduction &native,
                         mlir::IRMapping &mapping);
  mlir::LogicalResult generic(mlir::linalg::GenericOp operation);
  mlir::LogicalResult reduction(cpu::ReduceOp operation);

  mlir::FailureOr<llvm::SmallVector<mlir::Value>>
  writtenEnclosingLocals(mlir::Operation *scope);
  mlir::LogicalResult checkCarry(mlir::Value root, const LocalValue &before,
                                 mlir::Operation *scope);
  mlir::LogicalResult bindLocalReference(mlir::Value memory, mlir::Operation *scope);
  bool immutableBorrow(mlir::Value memory);
  bool completePrivateValue(mlir::Value memory);
  mlir::LogicalResult isolateControlValue(mlir::Value memory, mlir::Operation *scope,
                                          mlir::ValueRange boundaries);
  mlir::LogicalResult prepareControlValues(mlir::Operation *scope,
                                          mlir::ValueRange boundaries);
  mlir::FailureOr<llvm::SmallVector<mlir::Type>>
  controlTypes(mlir::ValueRange boundaries);
  mlir::Value controlOwner(mlir::Type type, mlir::Location loc,
                           mlir::Value initial = {});
  mlir::Value nativeOwner(mlir::Value value);
  bool canReuseControlOwner(mlir::Value input, mlir::Value boundary,
                            mlir::Operation *scope);
  void ownControlInputs(llvm::SmallVectorImpl<mlir::Value> &initial,
                        mlir::ValueRange inputs,
                        mlir::ValueRange boundaries, mlir::Operation *scope);
  mlir::FailureOr<mlir::Value> alignControlValue(mlir::Value value, mlir::Type target,
                                               mlir::Location loc);
  mlir::FailureOr<llvm::SmallVector<mlir::Value>>
  controlValues(mlir::ValueRange inputs, mlir::ValueRange boundaries,
                 mlir::ValueRange owners = {});
  mlir::LogicalResult mapControlValues(mlir::ValueRange source,
                                       mlir::ValueRange target);
  mlir::LogicalResult lower(mlir::scf::IfOp conditional);
  mlir::LogicalResult lower(mlir::scf::ForOp loop);
  mlir::LogicalResult lower(mlir::scf::WhileOp loop);

  mlir::func::FuncOp sourceFunction;
  std::unique_ptr<cpu::StorageAnalysis> storage;
  cpu::TasksOp currentTask;
  cpu::AxisRelations relations;
  mlir::ModuleOp output;
  mlir::OpBuilder b;
  mlir::IRMapping values;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<int64_t>> viewAxes;
  llvm::DenseMap<mlir::Value, LocalValue> locals;
  llvm::DenseMap<mlir::Value, mlir::Value> localReferences;
  llvm::DenseSet<mlir::Value> controlOwners;
  llvm::SmallVector<mlir::Value> shapeValues;
  llvm::SmallVector<std::pair<mlir::OpFoldResult, int64_t>> localShapeBindings;
  llvm::DenseMap<mlir::Value, mlir::Value> operandReads;
  llvm::DenseMap<mlir::Value, mlir::Value> readOnlySupplies;
  const llvm::DenseMap<mlir::Value, intent::QuantFormat> &formats;
  const cpu::ImplementationRegistry &implementations;
  llvm::DenseMap<int64_t, int64_t> extentIds;
  struct CapturedExtent { unsigned argument; unsigned axis; };
  llvm::SmallVector<CapturedExtent> capturedExtents;
  llvm::DenseMap<int64_t, int64_t> axisProjection;
  int64_t nextAxis;
};

} // namespace intent::weft_provider
#endif
