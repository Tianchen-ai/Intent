#ifndef INTENT_TARGET_DSA_ASCEND_IR_TARGET_TD
#define INTENT_TARGET_DSA_ASCEND_IR_TARGET_TD
include "mlir/IR/AttrTypeBase.td"
def IntentAscend_Dialect : Dialect {
  let name = "intent_ascend";
  let cppNamespace = "::intent::ascend";
  let useDefaultAttributePrinterParser = 1;
}
def IntentAscend_Device : AttrDef<IntentAscend_Dialect, "Device"> {
  let mnemonic = "device";
  let parameters = (ins "::mlir::StringAttr":$architecture,
                         "int64_t":$cubeCores, "int64_t":$vectorCores);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}
#endif
