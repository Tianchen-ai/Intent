#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "intentdsl/levels/algorithm/kir/ir/Dialect.h"
#include "Dialect.cpp.inc"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#define GET_OP_CLASSES
#include "Ops.h.inc"
#define GET_OP_CLASSES
#include "Ops.cpp.inc"
void intentdsl::kir::KirDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Ops.cpp.inc"
      >();
}

mlir::LogicalResult intentdsl::kir::welford::verify() {
  auto samples = getSamples().getType();
  auto state = mlir::dyn_cast<mlir::TupleType>(getState().getType());
  mlir::Builder builder(getContext());
  llvm::SmallVector<mlir::Type> fields{
      builder.getI32Type(), builder.getF32Type(), builder.getF32Type()};
  if (samples.getRank() != 1 || !samples.getElementType().isF32() ||
      !state || state.getTypes() != llvm::ArrayRef(fields))
    return emitOpError("requires an f32 member vector and (i32 count, f32 mean, f32 M2)");
  return mlir::success();
}

mlir::LogicalResult intentdsl::kir::field::verify() {
  auto state = mlir::dyn_cast<mlir::TupleType>(getRecord().getType());
  if (!state || getPosition() >= state.size() ||
      getElement().getType() != state.getType(getPosition()))
    return emitOpError("field does not match the declared tuple");
  return mlir::success();
}
