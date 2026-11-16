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

def IntentCPU_VectorCapabilitiesAttr : IntentCPU_Attr<"VectorCapabilities", "vector_capabilities"> {
  let parameters = (ins "int64_t":$minimumBits, "bool":$scalable);
  let assemblyFormat = "`<` $minimumBits `,` $scalable `>`";
}
def IntentCPU_ContractionCapabilityAttr : IntentCPU_Attr<"ContractionCapability", "contraction_capability"> {
  let parameters = (ins "::mlir::Type":$lhs, "::mlir::Type":$rhs,
      "::mlir::Type":$accumulator, "bool":$flushSubnormals);
  let assemblyFormat = "`<` $lhs `,` $rhs `,` $accumulator `,` $flushSubnormals `>`";
}
def IntentCPU_ExecutionConfigAttr : IntentCPU_Attr<"ExecutionConfig", "execution_config"> {
  let parameters = (ins "int64_t":$workers, "int64_t":$privateBytes);
  let assemblyFormat = "`<` $workers `,` $privateBytes `>`";
}
def IntentCPU_CapabilitiesAttr : IntentCPU_Attr<"Capabilities", "capabilities"> {
  let parameters = (ins "::intent::cpu::VectorCapabilitiesAttr":$vector,
      "::mlir::ArrayAttr":$contractions, "::intent::cpu::ExecutionConfigAttr":$execution);
  let assemblyFormat = "`<` $vector `,` $contractions `,` $execution `>`";
  let extraClassDeclaration = [{
    int64_t getVectorBits() const;
    int64_t getWorkers() const;
    int64_t getPrivateBytes() const;
    bool supportsContraction(::mlir::Type lhs, ::mlir::Type rhs, ::mlir::Type accumulator) const;
    bool getMatrixI8I32() const;
    bool getMatrixBF16F32() const;
  }];
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
