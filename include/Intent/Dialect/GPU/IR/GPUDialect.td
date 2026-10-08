#ifndef INTENT_DIALECT_GPU_IR_GPUDIALECT_TD
#define INTENT_DIALECT_GPU_IR_GPUDIALECT_TD

include "mlir/IR/DialectBase.td"

def IntentGPU_Dialect : Dialect {
  let name = "intent_gpu";
  let cppNamespace = "::intent::gpu";
  let summary = "Provider-neutral executable GPU block programs";
  let description = [{
    Intent GPU IR owns launch-visible program mapping, parameterized fragments,
    explicit accesses, structured control and structured compute.  It excludes
    provider API spelling and distributed machine layout.
  }];
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  let hasOperationAttrVerify = 1;
  let hasConstantMaterializer = 1;
}

#endif
