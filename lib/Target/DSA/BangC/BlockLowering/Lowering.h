#ifndef INTENT_TARGET_DSA_BANGC_BLOCKLOWERING_LOWERING_H
#define INTENT_TARGET_DSA_BANGC_BLOCKLOWERING_LOWERING_H
#include "Intent/Target/DSA/BangC/BlockLowering.h"
#include "Intent/Dialect/Block/IR/BlockOps.h"
#include "Intent/Dialect/Block/IR/Program.h"
#include "Intent/Dialect/Block/IR/PhysicalExpressions.h"
#include "Intent/Dialect/Block/Analysis/ProgramInterface.h"
#include "Intent/Dialect/Block/Analysis/PhysicalParameters.h"
#include "Intent/Conversion/ScalarLowering.h"
#include "Intent/Analysis/IntegerRanges.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include <functional>

namespace intent::bangc::block_lowering {
using namespace mlir;
class Lowering {
public:
  Lowering(func::FuncOp source, ModuleOp output, physical::ConfigurationAttr config,
      DictionaryAttr parameters, DictionaryAttr shapes, DictionaryAttr strides,
      block::ProgramInterface interface);
  LogicalResult run();

private:
  FailureOr<int64_t> constant(block::PhysicalExprAttr expression);
  FailureOr<SmallVector<int64_t>> shape(Type type);
  FailureOr<Value> expression(block::PhysicalExprAttr expression);
  Value index(int64_t value);
  Value get(Value value);
  Value allocate(Type logical);
  Value allocate(Type element, ArrayRef<int64_t> shape);
  Value read(Value memory, ValueRange coordinates);
  FailureOr<Value> element(Value original, Type domain, ValueRange coordinates);
  LogicalResult write(Value value, Value memory, ValueRange coordinates);
  SmallVector<Value> coordinates(Value memory, ValueRange logical);
  LogicalResult each(ArrayRef<int64_t> shape, function_ref<LogicalResult(ValueRange)> body);
  LogicalResult copy(Value source, Value destination);
  LogicalResult operations(Block &block);
  LogicalResult operation(Operation *operation);
  LogicalResult elementwise(Operation *operation);
  LogicalResult projection(Operation *operation);
  LogicalResult access(Operation *operation);
  FailureOr<bool> rectangularAccess(Operation *operation, Value resource, Value result);
  struct AffineCoordinate {
    Value base;
    SmallVector<int64_t> steps;
  };
  std::optional<AffineCoordinate> affineCoordinate(Value value);
  LogicalResult collective(Operation *operation);
  LogicalResult contraction(block::ContractOp operation);
  LogicalResult scaledContraction(block::ScaledContractOp operation);
  LogicalResult control(Operation *operation);
  FailureOr<Value> scalar(Operation *operation, ValueRange inputs);
  std::optional<ConstantIntRanges> integerRange(Value value);
  FailureOr<SmallVector<Value>> slots(ValueRange values);
  SmallVector<Value> flatten(ValueRange values);
  SmallVector<Type> leaves(Type type);
  LogicalResult publish(ValueRange source, ValueRange slots);
  LogicalResult bindState(ValueRange source, ValueRange slots);
  Value view(Value memory, ArrayRef<int64_t> shape);
  Location loc;
  func::FuncOp source, entry;
  ModuleOp output;
  OpBuilder b;
  physical::ConfigurationAttr config;
  DictionaryAttr parameters, shapes, strides;
  block::ProgramInterface interface;
  IRMapping values;
  DenseMap<Value, SmallVector<int64_t>> extents;
  DenseMap<block::ArgumentRefAttr, int64_t> staticArguments;
  DenseMap<Value, SmallVector<Value>> records;
  DenseMap<Operation *, Block *> deferred;
  DenseSet<Operation *> materializing;
  SmallVector<Value> programCoordinates;
};
}
#endif
