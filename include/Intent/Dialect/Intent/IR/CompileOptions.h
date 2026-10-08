#ifndef INTENT_DIALECT_INTENT_IR_COMPILEOPTIONS_H
#define INTENT_DIALECT_INTENT_IR_COMPILEOPTIONS_H

#include "Intent/Dialect/Intent/IR/IntentAttrs.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/JSON.h"

namespace intent {
inline constexpr llvm::StringLiteral compileOptionsAttr = "intent.compile_options";

mlir::FailureOr<CompileOptionsAttr> readCompileOptions(mlir::Operation *operation);
llvm::json::Object serializeCompileOptions(CompileOptionsAttr options);
}
#endif
