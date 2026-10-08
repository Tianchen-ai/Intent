#ifndef INTENT_TARGET_WEFT_IR_PROGRAM_H
#define INTENT_TARGET_WEFT_IR_PROGRAM_H

#include "Intent/Target/Weft/IR/WeftAttrs.h"
#include "mlir/IR/BuiltinOps.h"

namespace weft::kernel { class KernelOp; }

namespace intent::weft_provider {

inline constexpr llvm::StringLiteral hostModuleName = "host";
inline constexpr llvm::StringLiteral deviceModuleName = "device";
inline constexpr llvm::StringLiteral taskBindingsAttr = "intent_weft.tasks";
inline constexpr llvm::StringLiteral argumentAlignmentAttr = "intent_weft.alignment";

struct ProgramModules {
  mlir::ModuleOp host;
  mlir::ModuleOp device;
};

mlir::FailureOr<ProgramModules> getProgramModules(mlir::ModuleOp program);
mlir::FailureOr<mlir::DenseI32ArrayAttr>
taskCallAccesses(::weft::kernel::KernelOp kernel);
mlir::LogicalResult verifyProgram(mlir::ModuleOp program);

} // namespace intent::weft_provider
#endif
