#ifndef INTENT_TARGET_TRITON_IR_TRITONDIALECT_TD
#define INTENT_TARGET_TRITON_IR_TRITONDIALECT_TD

include "mlir/IR/DialectBase.td"

def IntentTriton_Dialect : Dialect {
  let name = "intent_triton";
  let cppNamespace = "::intent::triton";
  let summary = "Triton-local access and provider forms";
}

#endif
