#ifndef INTENT_DIALECT_CPU_IR_CPUDIALECT_TD
#define INTENT_DIALECT_CPU_IR_CPUDIALECT_TD

include "mlir/IR/DialectBase.td"

def IntentCPU_Dialect : Dialect {
  let name = "intent_cpu";
  let cppNamespace = "::intent::cpu";
  let summary = "Provider-neutral executable CPU programs";
  let useDefaultAttributePrinterParser = 1;
  let hasCanonicalizer = 1;
}

#endif
