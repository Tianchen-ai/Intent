#ifndef INTENT_CONVERSION_KIRTOBlock_KIRTOBlock_H
#define INTENT_CONVERSION_KIRTOBlock_KIRTOBlock_H

#include "mlir/IR/BuiltinOps.h"

namespace intent {

struct BlockCapabilities {
  mlir::Attribute target;
  bool matrixUnits;
  bool dynamicVectorWidth;
  bool nativeTupleReductions;
  bool nativeTupleReductionRequiresConstantIdentity;
  bool nativeFragmentGather;
};

/// Consumes canonical KIR and replaces it with one complete conservative
/// provider-neutral executable Block program.  The result never references KIR.
mlir::LogicalResult lowerCanonicalKIRToBlock(mlir::ModuleOp module,
                                           const BlockCapabilities &capabilities);

} // namespace intent

#endif
