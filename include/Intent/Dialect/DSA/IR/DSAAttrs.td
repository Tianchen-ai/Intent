#ifndef INTENT_DIALECT_DSA_IR_DSAATTRS_TD
#define INTENT_DIALECT_DSA_IR_DSAATTRS_TD
include "Intent/Dialect/DSA/IR/DSADialect.td"
include "Intent/Dialect/Intent/IR/IntentAttrs.td"
include "mlir/IR/AttrTypeBase.td"
def IntentDSA_EntryRequirements : AttrDef<IntentDSA_Dialect, "EntryRequirements"> {
  let mnemonic = "entry_requirements";
  let parameters = (ins "bool":$disjointOutputs);
  let assemblyFormat = "`<` $disjointOutputs `>`";
  let genVerifyDecl = 1;
}
def IntentDSA_Configuration : AttrDef<IntentDSA_Dialect, "Configuration"> {
  let mnemonic = "configuration";
  let parameters = (ins "int64_t":$tile, "int64_t":$tileM,
    "int64_t":$tileN, "int64_t":$tileK, "int64_t":$regionTile, "int64_t":$tasks,
    "int64_t":$localBytes);
  let assemblyFormat = "`<` $tile `,` $tileM `,` $tileN `,` $tileK `,` $regionTile `,` $tasks `,` $localBytes `>`";
  let genVerifyDecl = 1;
}
#endif
