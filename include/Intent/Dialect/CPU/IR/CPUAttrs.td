#ifndef INTENT_DIALECT_CPU_IR_CPUATTRS_TD
#define INTENT_DIALECT_CPU_IR_CPUATTRS_TD

include "Intent/Dialect/CPU/IR/CPUDialect.td"
include "mlir/IR/AttrTypeBase.td"

class IntentCPU_Attr<string name, string mnemonicName>
    : AttrDef<IntentCPU_Dialect, name> {
  let mnemonic = mnemonicName;
  let genVerifyDecl = 1;
}

def IntentCPU_EntryRequirementsAttr : IntentCPU_Attr<"EntryRequirements", "entry_requirements"> {
  let parameters = (ins "bool":$contiguousViews, "bool":$disjointOutputs);
  let assemblyFormat = "`<` $contiguousViews `,` $disjointOutputs `>`";
}

def IntentCPU_CapabilitiesAttr : IntentCPU_Attr<"Capabilities", "capabilities"> {
  let parameters = (ins "int64_t":$vectorBits, "int64_t":$workers,
    "int64_t":$privateBytes, "bool":$matrixI8I32);
  let assemblyFormat = "`<` $vectorBits `,` $workers `,` $privateBytes `,` $matrixI8I32 `>`";
}

def IntentCPU_ConfigurationAttr : IntentCPU_Attr<"Configuration", "configuration"> {
  let parameters = (ins "int64_t":$taskGrain,
    "int64_t":$tileM, "int64_t":$tileN, "int64_t":$tileK, "int64_t":$regionSize);
  let assemblyFormat =
    "`<` $taskGrain `,` $tileM `,` $tileN `,` $tileK `,` $regionSize `>`";
}

def IntentCPU_ImplementationAttr : IntentCPU_Attr<"Implementation", "implementation"> {
  let parameters = (ins "::mlir::StringAttr":$name,
                        "::mlir::DictionaryAttr":$parameters);
  let assemblyFormat = "`<` $name `,` $parameters `>`";
}

def IntentCPU_ReductionOrderAttr : IntentCPU_Attr<"ReductionOrder", "reduction_order"> {
  let parameters = (ins "bool":$adjacentReassociation, "bool":$elementPermutation);
  let assemblyFormat = "`<` $adjacentReassociation `,` $elementPermutation `>`";
}

def IntentCPU_MicrotileAttr : IntentCPU_Attr<"Microtile", "microtile"> {
  let parameters = (ins "int64_t":$rows, "int64_t":$columns,
    "int64_t":$vectorWidth);
  let assemblyFormat = "`<` $rows `,` $columns `,` $vectorWidth `>`";
}

#endif
