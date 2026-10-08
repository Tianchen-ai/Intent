#ifndef INTENT_DIALECT_DSA_IR_DSADIALECT_TD
#define INTENT_DIALECT_DSA_IR_DSADIALECT_TD
include "mlir/IR/OpBase.td"
def IntentDSA_Dialect : Dialect {
  let name = "intent_dsa";
  let cppNamespace = "::intent::dsa";
  let summary = "Local-memory tile programs with explicit work and data supply";
  let useDefaultAttributePrinterParser = 1;
}
#endif
