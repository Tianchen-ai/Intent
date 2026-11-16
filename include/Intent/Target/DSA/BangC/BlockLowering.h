#ifndef INTENT_TARGET_DSA_BANGC_BLOCKLOWERING_H
#define INTENT_TARGET_DSA_BANGC_BLOCKLOWERING_H
#include "Intent/Target/DSA/BangC/Physical/IR/DSAOps.h"
#include "mlir/IR/BuiltinOps.h"
namespace intent::bangc {
mlir::LogicalResult constructBlockStorage(mlir::ModuleOp module,
    physical::ConfigurationAttr configuration, physical::CapabilitiesAttr hardware,
    mlir::DictionaryAttr shapes, mlir::DictionaryAttr strides);
mlir::LogicalResult lowerBlockToBangC(mlir::ModuleOp module,
    physical::ConfigurationAttr configuration, mlir::DictionaryAttr parameters,
    mlir::DictionaryAttr shapes, mlir::DictionaryAttr strides);
}
#endif
