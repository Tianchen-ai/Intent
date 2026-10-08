#ifndef INTENT_DIALECT_GPU_IR_INTEGERRANGES_H
#define INTENT_DIALECT_GPU_IR_INTEGERRANGES_H

namespace mlir {
class DialectRegistry;
}

namespace intent::gpu {

void registerIntegerRangeInterfaces(mlir::DialectRegistry &registry);

} // namespace intent::gpu
#endif
