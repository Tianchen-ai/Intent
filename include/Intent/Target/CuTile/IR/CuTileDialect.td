#ifndef INTENT_TARGET_CUTILE_IR_CUTILEDIALECT_TD
#define INTENT_TARGET_CUTILE_IR_CUTILEDIALECT_TD

include "mlir/IR/DialectBase.td"

def IntentCuTile_Dialect : Dialect {
  let name = "intent_cutile";
  let cppNamespace = "::intent::cutile";
  let summary = "cuTile-local executable forms";
}

#endif
