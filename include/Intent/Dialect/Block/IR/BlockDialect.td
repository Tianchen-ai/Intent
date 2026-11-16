#ifndef INTENT_DIALECT_Block_IR_BlockDIALECT_TD
#define INTENT_DIALECT_Block_IR_BlockDIALECT_TD

include "mlir/IR/DialectBase.td"

def IntentBlock_Dialect : Dialect {
  let name = "intent_block";
  let cppNamespace = "::intent::block";
  let summary = "Provider-neutral executable block programs";
  let description = [{
    Intent Block IR owns launch-visible program mapping, parameterized fragments,
    explicit accesses, structured control and structured compute.  It excludes
    provider API spelling and distributed machine layout.
  }];
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  let hasOperationAttrVerify = 1;
  let hasConstantMaterializer = 1;
}

#endif
