#ifndef INTENT_DIALECT_INTENT_IR_INTERFACE_H
#define INTENT_DIALECT_INTENT_IR_INTERFACE_H

#include "Intent/Dialect/Intent/IR/IntentAttrs.h"
#include "Intent/Dialect/Intent/IR/IntentTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/JSON.h"

namespace intent {

inline constexpr llvm::StringLiteral interfaceAttr = "intent.interface";
inline constexpr llvm::StringLiteral sourceParameterAttr = "intent.parameter";

ParameterAttr getSourceParameter(mlir::BlockArgument argument);
mlir::LogicalResult verifySourceInterface(mlir::func::FuncOp function);

// Runtime author order excludes specialized constexpr parameters. Family
// construction binds this contract to its own physical argument representation.
mlir::FailureOr<InterfaceAttr> buildPublicInterface(mlir::func::FuncOp kernel);
InterfaceAttr getPublicInterface(mlir::func::FuncOp function);
ViewType getPublicView(InterfaceAttr interface, unsigned publicOrdinal);
mlir::LogicalResult verifyPublicInterface(mlir::Operation *owner,
                                         InterfaceAttr interface);
mlir::FailureOr<llvm::json::Object>
serializePublicInterface(mlir::Operation *owner, InterfaceAttr interface);

// A public view's shape and identities are carried by its canonical tensor type.
mlir::RankedTensorType publicViewTensor(ViewType view);
mlir::DenseI64ArrayAttr publicViewDimensions(ViewType view);
std::string scalarABIName(mlir::Type type);

} // namespace intent
#endif
