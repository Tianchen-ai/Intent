include "mlir/IR/DialectBase.td"

def IntentWeft_Dialect : Dialect {
  let name = "intent_weft";
  let cppNamespace = "::intent::weft_provider";
  let useDefaultAttributePrinterParser = 1;
}
