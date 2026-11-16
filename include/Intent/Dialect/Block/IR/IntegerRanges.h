#ifndef INTENT_DIALECT_Block_IR_INTEGERRANGES_H
#define INTENT_DIALECT_Block_IR_INTEGERRANGES_H

namespace mlir {
class DialectRegistry;
}

namespace intent::block {

void registerIntegerRangeInterfaces(mlir::DialectRegistry &registry);

} // namespace intent::block
#endif
