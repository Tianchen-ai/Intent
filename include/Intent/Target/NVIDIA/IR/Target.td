#ifndef INTENT_TARGET_NVIDIA_IR_TARGET_TD
#define INTENT_TARGET_NVIDIA_IR_TARGET_TD
include "mlir/IR/AttrTypeBase.td"
def IntentNVIDIA_Dialect : Dialect {
  let name = "intent_nvidia";
  let cppNamespace = "::intent::nvidia";
  let useDefaultAttributePrinterParser = 1;
}
def IntentNVIDIA_Device : AttrDef<IntentNVIDIA_Dialect, "Device"> {
  let mnemonic = "device";
  let parameters = (ins "int64_t":$computeUnits, "int64_t":$sharedMemoryPerUnit,
    "int64_t":$maxDynamicSharedMemoryPerBlock, "int64_t":$registersPerUnit,
    "int64_t":$maxThreadsPerBlock, "int64_t":$computeCapabilityMajor,
    "int64_t":$computeCapabilityMinor, "int64_t":$singleToDoublePrecisionPerfRatio,
    "bool":$matrixUnits, "bool":$dynamicVectorWidth);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}
#endif
