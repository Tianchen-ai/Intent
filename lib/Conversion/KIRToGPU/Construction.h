#ifndef INTENT_CONVERSION_KIRTOGPU_CONSTRUCTION_H
#define INTENT_CONVERSION_KIRTOGPU_CONSTRUCTION_H

#include "Intent/Analysis/CanonicalKernel.h"
#include "Intent/Conversion/KIRToGPU/KIRToGPU.h"
#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/Intent/IR/Interface.h"
#include "Intent/Dialect/Intent/IR/IntentOps.h"
#include "Intent/Interfaces/StructuredOpInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include <initializer_list>
#include <optional>

namespace intent::kir_to_gpu {
using namespace mlir;
using gpu::AxisMapAttr;
using gpu::FragmentType;
using gpu::PhysicalExprAttr;
using gpu::PhysicalExprKind;

struct PhysicalAxisIdentity {
  uint64_t sourceId;
  uint32_t sourceAxis;
  int64_t dimensionId;
  bool derived = false;
};

struct MetadataBinding {
  int64_t dimension;
  unsigned sourceABI;
  unsigned sourceAxis;
};

struct PhysicalABI {
  SmallVector<Type> arguments;
  SmallVector<DictionaryAttr> argumentAttrs;
  SmallVector<std::optional<unsigned>> physicalArgumentForSource;
  InterfaceAttr interface;
  llvm::DenseMap<int64_t, MetadataBinding> dimensions;
  SmallVector<int64_t> dimensionOrder;
};

struct IterationAxis {
  Value start;
  Value stop;
  Value step;
  Value coordinatePrototype;
  Value source;
};

DenseI64ArrayAttr dimensionIds(RankedTensorType tensor);

RankedTensorType viewTensor(Value value);

PhysicalExprAttr expression(
    MLIRContext *context,
    PhysicalExprKind kind,
    int64_t value = 0,
    StringRef symbol = {},
    ArrayRef<Attribute> operands = {});

PhysicalExprAttr parameterExpression(MLIRContext *context, StringRef name);

PhysicalExprAttr dimensionExpression(func::FuncOp function, int64_t dimension);

PhysicalExprAttr binaryExpression(
    MLIRContext *context,
    PhysicalExprKind kind,
    PhysicalExprAttr lhs,
    PhysicalExprAttr rhs);

FragmentType fragmentType(
    MLIRContext *context,
    Type element,
    ArrayRef<PhysicalExprAttr> shape,
    ArrayRef<PhysicalAxisIdentity> axes,
    uint64_t owner = 1);

Value createBinary(
    OpBuilder &builder,
    Location location,
    Type result,
    Value lhs,
    Value rhs,
    BinaryOperator kind);

bool samePhysicalShape(gpu::FragmentType lhs, gpu::FragmentType rhs);

FailureOr<Value> retargetBroadcast(
    OpBuilder &builder,
    Location location,
    Value value,
    gpu::FragmentType target);

FailureOr<Value> projectPositionalValue(
    OpBuilder &builder,
    Location location,
    Value identity,
    Type targetType);

Value createCompare(
    OpBuilder &builder,
    Location location,
    Type result,
    Value lhs,
    Value rhs,
    ComparePredicate predicate);

FailureOr<PhysicalABI> buildPhysicalABI(
    func::FuncOp function,
    OpBuilder &builder);

FailureOr<PhysicalExprAttr> launchExpression(
    Value value,
    CanonicalKernelAnalysis &canonicalAnalysis,
    func::FuncOp function = {});

FailureOr<PhysicalExprAttr> launchExtentExpression(
    CanonicalKernelAnalysis &canonicalAnalysis, Value value, unsigned axis,
    func::FuncOp function, ArrayRef<unsigned> fieldPath = {});

std::optional<int64_t> sourceExtentDimension(Value source);

std::optional<int64_t> integerConstant(Value value);

std::optional<int64_t> subregionStaticExtentBound(Value source);

FailureOr<PhysicalExprAttr> fragmentExtentExpression(
    CanonicalKernelAnalysis &canonicalAnalysis,
    RankedTensorType tensor,
    Operation *origin,
    unsigned axis);

Type convertScalarType(Type type, uint64_t owner = 1);

FailureOr<PhysicalAxisIdentity> resultAxisIdentity(
    Operation *operation,
    unsigned resultIndex = 0,
    unsigned axis = 0);

FailureOr<FragmentType> convertTensorType(
    CanonicalKernelAnalysis &canonicalAnalysis,
    RankedTensorType tensor,
    Operation *origin,
    std::optional<FragmentType> prototype = std::nullopt,
    uint64_t owner = 1,
    unsigned resultIndex = 0);

FailureOr<Type> convertDataType(
    CanonicalKernelAnalysis &canonicalAnalysis,
    Type type,
    Operation *origin,
    std::optional<Type> prototype = std::nullopt,
    uint64_t owner = 1,
    unsigned resultIndex = 0);

LogicalResult collectIterationAxes(
    Value source,
    SmallVectorImpl<IterationAxis> &axes);

LogicalResult constructGPUProgram(
    ModuleOp module,
    const GPUCapabilities &capabilities,
    func::FuncOp function);

// One lexical conversion context owns the source-to-physical bindings. Child
// regions copy those bindings exactly where the canonical region boundary does.
class ScalarRegionLowering {
public:
  ScalarRegionLowering(
      OpBuilder &builder,
      llvm::DenseMap<Value, Value> values,
      ArrayRef<Value> views,
      llvm::DenseMap<int64_t, Value> abiDimensions,
      llvm::DenseMap<StringAttr, Value> parameters,
      CanonicalKernelAnalysis &canonicalAnalysis,
      func::FuncOp physicalKernel);

  FailureOr<SmallVector<Value>> lowerBlock(Block &source);

  FailureOr<SmallVector<Value>> lowerWorksetBlock(Block &source);

  llvm::DenseMap<Value, Value> &mapping();

  FailureOr<Value> lowerValue(Value source);

  FailureOr<Value> lowerIndexValue(Value source);

private:
  LogicalResult lower(Operation *operation);

  FailureOr<Value> get(Value source);

  LogicalResult alignOperands(
      Operation *operation,
      std::initializer_list<Value *> operands,
      ArrayRef<unsigned> positions);

  FailureOr<Type> elementwiseResultType(
      Type logical,
      Operation *origin,
      Value prototype);

  void mapResults(Operation *source, Operation *target);

  FailureOr<gpu::ParameterAttr> getOrCreateRegionSegment(Operation *operation);

  LogicalResult lowerPureRegion(
      Region &source,
      Region &target,
      ArrayRef<Type> argumentTypes,
      ArrayRef<Type> resultTypes = {});

  void attachOrigin(Operation *source, Operation *target);

  Value rangeExtent(Location location, Value start, Value stop, Value step);

  Value ceilDivide(Location location, Value numerator, Value denominator);

  Value rangeBound(Location location, Value range, unsigned bound);

  FailureOr<Value> physicalExtentValue(
      Location location,
      PhysicalExprAttr expression);

  FailureOr<Value> logicalExtent(Location location, Value value, unsigned axis,
                                 ArrayRef<unsigned> fieldPath = {});

  FailureOr<Value> asIndex(Location location, Value value);

  FailureOr<Value> asLogicalIndex(Location location, Value value);

  FailureOr<unsigned> physicalResourceAxis(
      Type logicalResource,
      Type physicalResource,
      unsigned logicalAxis);

  FailureOr<Value> resourceExtent(
      Location location,
      Value resource,
      unsigned axis);

  FailureOr<SmallVector<Value>> accessCoordinates(
      Operation *operation, SmallVectorImpl<int64_t> &sourceAxes);

  FailureOr<Type> accessResultType(
      Operation *operation,
      Type logical,
      ArrayRef<Value> coordinates,
      std::optional<Type> prototype = std::nullopt);

  FailureOr<Value> accessValue(
      Operation *operation,
      Value logical,
      ArrayRef<Value> coordinates);

  FailureOr<Value> projectAccessOperand(
      Location location,
      Value value,
      gpu::FragmentType accessType);

  SmallVector<bool> canonicalAccessAxisProofs(
      Operation *operation,
      Value resource);

  FailureOr<Value> materializeAccessValidity(
      Operation *operation,
      Value resource,
      ArrayRef<Value> coordinates,
      ArrayRef<int64_t> sourceAxes,
      Type payloadType,
      Value existing);

  FailureOr<Value> zeroAccessFill(Location location, Type type);

  LogicalResult lowerIndexedRead(Operation *operation);

  LogicalResult lowerIndexedWrite(Operation *operation);

  LogicalResult lower(intent::ConstantOp constant);

  LogicalResult lower(intent::DimOp dim);

  LogicalResult lower(intent::DomainOp domain);

  LogicalResult lower(intent::SubregionOp subregion);

  LogicalResult lower(intent::RegionEndOp end);

  LogicalResult lower(intent::IndicesOp indices);

  LogicalResult lower(intent::FullOp full);

  LogicalResult lower(intent::BroadcastOp broadcastOp);

  LogicalResult lower(intent::ReshapeOp reshape);

  LogicalResult lower(intent::TransposeOp transpose);

  LogicalResult lower(intent::JoinOp join);

  LogicalResult lower(intent::UnaryOp unary);

  LogicalResult lower(intent::BinaryOp binary);

  LogicalResult lower(intent::CompareOp compare);

  LogicalResult lower(intent::SelectOp select);

  LogicalResult lower(intent::CastOp cast);

  LogicalResult lower(intent::BitcastOp bitcast);

  LogicalResult lower(intent::MaskOp mask);

  LogicalResult lowerStructured(StructuredOpInterface schema);

  LogicalResult lower(intent::ContractOp contract);

  LogicalResult lower(intent::ScaledContractOp contract);

  LogicalResult lower(intent::SparseContractOp contract);

  LogicalResult lower(intent::HistogramOp histogram);

  LogicalResult lower(intent::QuantizeOp quantize);

  LogicalResult lower(intent::QuantizedDotOp dot);

  LogicalResult lower(intent::RandomBitsOp random);

  LogicalResult lower(intent::BufferOp buffer);

  LogicalResult lower(intent::GatherOp gather);

  LogicalResult lower(intent::ViewLoadOp load);

  LogicalResult lower(intent::BufferLoadOp load);

  LogicalResult lower(intent::ViewStoreOp store);

  LogicalResult lower(intent::BufferStoreOp store);

  LogicalResult lower(intent::ScatterUniqueOp store);

  LogicalResult lower(intent::ScatterReduceOp scatter);

  LogicalResult lower(intent::AtomicLoadOp atomic);

  LogicalResult lower(intent::AtomicStoreOp atomic);

  LogicalResult lower(intent::AtomicRMWOp atomic);

  LogicalResult lower(intent::AtomicCompareExchangeOp atomic);

  LogicalResult lower(intent::MakeRecordOp record);

  LogicalResult lower(intent::MakeTupleOp tuple);

  LogicalResult lower(intent::ExtractOp extract);

  LogicalResult lower(intent::IfOp ifOperation);

  LogicalResult lowerLoop(Operation *operation);

  LogicalResult lower(intent::WhileOp whileOperation);

  LogicalResult lower(intent::AssumeInBoundsOp assumption);

  OpBuilder &builder;
  llvm::DenseMap<Value, Value> values;
  ArrayRef<Value> views;
  // Only entry ABI dimension bindings; local shapes are value/axis relations.
  llvm::DenseMap<int64_t, Value> abiDimensions;
  llvm::DenseMap<StringAttr, Value> parameters;
  CanonicalKernelAnalysis &canonicalAnalysis;
  func::FuncOp physicalKernel;
};

} // namespace intent::kir_to_gpu
#endif
