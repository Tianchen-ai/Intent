#ifndef INTENT_DIALECT_DSA_IR_DSADIALECT_TD
#define INTENT_DIALECT_DSA_IR_DSADIALECT_TD
include "mlir/IR/OpBase.td"
def IntentBangCPhysical_Dialect : Dialect {
  let name = "intent_bangc";
  let cppNamespace = "::intent::bangc::physical";
  let summary = "Local-memory tile programs with explicit work and data supply";
  let useDefaultAttributePrinterParser = 1;
}
#endif
