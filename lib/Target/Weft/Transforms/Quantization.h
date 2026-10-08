#ifndef INTENT_TARGET_WEFT_TRANSFORMS_QUANTIZATION_H
#define INTENT_TARGET_WEFT_TRANSFORMS_QUANTIZATION_H

#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"

namespace intent::cpu { class ImplementationExpansion; }

namespace intent::weft_provider {
::weft::kernel::EncodingType quantEncoding(mlir::OpBuilder &builder, intent::QuantFormat format);
void declareQuantEncodings(mlir::ModuleOp module);
mlir::LogicalResult expandQuantize(cpu::QuantizeOp operation,
                                 cpu::ImplementationExpansion &expansion);
mlir::LogicalResult expandQuantizedDot(cpu::QuantizedDotOp operation,
                                     cpu::ImplementationExpansion &expansion);
}
#endif
