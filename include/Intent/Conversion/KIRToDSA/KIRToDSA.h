#ifndef INTENT_CONVERSION_KIRTODSA_H
#define INTENT_CONVERSION_KIRTODSA_H
#include "Intent/Dialect/DSA/IR/DSAOps.h"
namespace intent {
mlir::LogicalResult lowerCanonicalKIRToDSA(mlir::ModuleOp module,
    dsa::ConfigurationAttr configuration, mlir::DictionaryAttr shapes = {},
    mlir::DictionaryAttr strides = {});
}
#endif
