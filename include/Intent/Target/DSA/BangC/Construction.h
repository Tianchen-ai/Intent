#ifndef INTENT_CONVERSION_KIRTODSA_H
#define INTENT_CONVERSION_KIRTODSA_H
#include "Intent/Target/DSA/BangC/Physical/IR/DSAOps.h"
namespace intent {
mlir::LogicalResult lowerCanonicalKIRToDSA(mlir::ModuleOp module,
    bangc::physical::ConfigurationAttr configuration, mlir::DictionaryAttr shapes = {},
    mlir::DictionaryAttr strides = {});
}
#endif
