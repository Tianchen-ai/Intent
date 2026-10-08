include "Intent/Target/Weft/IR/WeftDialect.td"
include "mlir/IR/AttrTypeBase.td"

def IntentWeft_TaskBinding : AttrDef<IntentWeft_Dialect, "TaskBinding"> {
  let mnemonic = "task_binding";
  let parameters = (ins "::mlir::SymbolRefAttr":$hostEntry,
      "::mlir::SymbolRefAttr":$hostCallee, "::mlir::SymbolRefAttr":$deviceKernel,
      "int64_t":$ordinal, "int64_t":$coordinateArgument);
  let assemblyFormat = "`<` $hostEntry `,` $hostCallee `,` $deviceKernel `,` $ordinal `,` $coordinateArgument `>`";
  let genVerifyDecl = 1;
}

def IntentWeft_ArgumentAlignment : AttrDef<IntentWeft_Dialect, "ArgumentAlignment"> {
  let mnemonic = "argument_alignment";
  let parameters = (ins "int64_t":$bytes);
  let assemblyFormat = "`<` $bytes `>`";
  let genVerifyDecl = 1;
}
