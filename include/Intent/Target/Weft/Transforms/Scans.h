#ifndef INTENT_TARGET_WEFT_TRANSFORMS_SCANS_H
#define INTENT_TARGET_WEFT_TRANSFORMS_SCANS_H

#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::weft_provider {
bool hasNativeScanCombine(cpu::ScanOp operation);
bool isNativeScanTile(cpu::ScanOp operation);
mlir::LogicalResult tileScans(mlir::func::FuncOp function);
} // namespace intent::weft_provider
#endif
