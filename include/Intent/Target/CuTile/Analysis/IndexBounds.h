#ifndef INTENT_TARGET_CUTILE_ANALYSIS_INDEXBOUNDS_H
#define INTENT_TARGET_CUTILE_ANALYSIS_INDEXBOUNDS_H
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <optional>
namespace intent::cutile {
using ArrayIndexBounds = llvm::SmallVector<std::pair<mlir::Value, mlir::ArrayAttr>>;
// Current native accesses must prove every array bound before index narrowing.
std::optional<ArrayIndexBounds> arrayIndexTileBounds(mlir::func::FuncOp kernel);
bool isProvably(mlir::Value value, int64_t expected);
mlir::Value stripIndexIdentities(mlir::Value value);
}
#endif
