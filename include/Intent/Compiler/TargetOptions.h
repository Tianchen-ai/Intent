#ifndef INTENT_COMPILER_TARGETOPTIONS_H
#define INTENT_COMPILER_TARGETOPTIONS_H

#include "Intent/Compiler/Compiler.h"
#include "llvm/Support/FormatVariadic.h"

namespace intent::compiler {
inline mlir::FailureOr<llvm::json::Object> readTargetOptions(
    mlir::ModuleOp module, const Request &request, llvm::ArrayRef<llvm::StringRef> fields,
    llvm::ArrayRef<llvm::StringRef> optional = {}) {
  auto parsed = llvm::json::parse(request.targetOptions);
  if (!parsed) {
    module.emitError("invalid target options: ") << llvm::toString(parsed.takeError());
    return mlir::failure();
  }
  auto object = parsed->getAsObject();
  if (!object || llvm::any_of(*object, [&](const auto &entry) {
        return !llvm::is_contained(fields, entry.first.str()) &&
               !llvm::is_contained(optional, entry.first.str());
      }))
    return module.emitError("target options do not match the selected provider's fields"), mlir::failure();
  for (auto field : fields)
    if (!object->get(field))
      return module.emitError("missing target option: ") << field, mlir::failure();
  return std::move(*object);
}

inline mlir::LogicalResult readIntegerOption(mlir::ModuleOp module,
    const llvm::json::Object &options, llvm::StringRef name, int64_t &destination) {
  auto value = options.getInteger(name);
  if (!value) return module.emitError("target option must be an integer: ") << name;
  destination = *value;
  return mlir::success();
}

inline mlir::LogicalResult readBooleanOption(mlir::ModuleOp module,
    const llvm::json::Object &options, llvm::StringRef name, bool &destination) {
  auto value = options.getBoolean(name);
  if (!value) return module.emitError("target option must be boolean: ") << name;
  destination = *value;
  return mlir::success();
}
}
#endif
