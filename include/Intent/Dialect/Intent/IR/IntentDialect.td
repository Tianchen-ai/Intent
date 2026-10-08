#ifndef INTENT_DIALECT_INTENT_IR_INTENTDIALECT_TD
#define INTENT_DIALECT_INTENT_IR_INTENTDIALECT_TD

include "mlir/IR/DialectBase.td"

def Intent_Dialect : Dialect {
  let name = "intent";
  let cppNamespace = "::intent";
  let summary = "Structured, logical single-kernel algorithms before realization";
  let description = [{
    Intent preserves source-visible domains, regions, tensor-flow, structured
    computation, state and effects while excluding physical workers, layouts,
    storage placement, pipelines and launch configuration.
  }];
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  let hasOperationAttrVerify = 1;
  let hasRegionArgAttrVerify = 1;
}

#endif
