include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
// welford is registered below the dialect declaration.
include "mlir/Interfaces/SideEffectInterfaces.td"
def Iblock_Dialect : Dialect {
  let name = "block";
  let cppNamespace = "::intentdsl::iblock";
}
class Inst<string mnemonic, list<Trait> traits = []>
    : Op<Iblock_Dialect, mnemonic, traits>;
def Iblock_parallel : Inst<"parallel"> {
  let arguments = (ins AnyMemRef:$domain);
  let results = (outs);
  let regions = (region AnyRegion:$body);
}
def Iblock_end : Inst<"end", [Terminator]> {
  let arguments = (ins);
  let results = (outs);
}
def Iblock_read : Inst<"read"> {
  let arguments = (ins AnyMemRef:$view, Variadic<Index>:$indices);
  let results = (outs AnyType:$result);
}
def Iblock_OwnerReadOp : Inst<"read_owner"> {
  let arguments = (ins AnyMemRef:$source,Index:$owner); let results = (outs AnyType:$scalar);
}
def Iblock_OwnerWriteOp : Inst<"write_owner"> {
  let arguments = (ins AnyMemRef:$destination,Index:$owner,AnyType:$scalar);
}
def Iblock_write : Inst<"write"> {
  let arguments = (ins AnyMemRef:$view, Index:$row, AnyTensor:$value);
  let results = (outs);
}
def Iblock_pid : Inst<"pid"> {
  let arguments = (ins);
  let results = (outs Index:$result);
}
def Iblock_rows : Inst<"rows"> {
  let arguments = (ins Index:$instance);
  let results = (outs AnyType:$result);
}
def Iblock_columns : Inst<"columns"> {
  let arguments = (ins Index:$tile);
  let results = (outs AnyType:$result);
}
def Iblock_active : Inst<"active"> {
  let arguments = (ins AnyMemRef:$domain, AnyTensor:$rows, AnyTensor:$columns);
  let results = (outs AnyType:$result);
}
def Iblock_load : Inst<"load"> {
  let arguments = (ins AnyMemRef:$view, AnyTensor:$rows, AnyTensor:$columns,
      AnyTensor:$mask);
  let results = (outs AnyType:$result);
}
def Iblock_store : Inst<"store"> {
  let arguments = (ins AnyMemRef:$view, AnyTensor:$rows, AnyTensor:$columns,
      AnyTensor:$mask, AnyTensor:$value);
  let results = (outs);
}
def Iblock_Dialect_OwnerLoadOp : Op<Iblock_Dialect,"owner_load"> {
  let arguments = (ins AnyMemRef:$view,AnyTensor:$rows); let results = (outs AnyTensor:$values);
}
def Iblock_Dialect_OwnerStoreOp : Op<Iblock_Dialect,"owner_store"> {
  let arguments = (ins AnyMemRef:$view,AnyTensor:$rows,AnyTensor:$values);
}
def Iblock_reduce : Inst<"reduce"> {
  let arguments = (ins AnyTensor:$value, AnyTensor:$mask, StrAttr:$algebra);
  let results = (outs AnyType:$result);
}
def Iblock_tiles : Inst<"tiles"> {
  let arguments = (ins AnyMemRef:$domain, Variadic<AnyType>:$initial);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$body);
}
def Iblock_yield : Inst<"yield", [Terminator]> {
  let arguments = (ins Variadic<AnyType>:$values);
  let results = (outs);
}
def Iblock_zero : Inst<"zero"> {
  let arguments = (ins);
  let results = (outs Index:$result);
}
def Iblock_combine : Inst<"combine"> {
  let arguments = (ins AnyTensor:$acc, AnyTensor:$partial, StrAttr:$algebra);
  let results = (outs AnyType:$result);
}
def Iblock_identity : Inst<"identity"> {
  let arguments = (ins StrAttr:$algebra);
  let results = (outs AnyType:$result);
}
def Iblock_broadcast_rows : Inst<"broadcast_rows"> {
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
}
def Iblock_cast : Inst<"cast"> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
}
def Iblock_add : Inst<"add"> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
}
def Iblock_sub : Inst<"sub"> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
}
def Iblock_mul : Inst<"mul"> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
}
def Iblock_div : Inst<"div"> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
}
def Iblock_exp : Inst<"exp"> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
}
def Iblock_rsqrt : Inst<"rsqrt"> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
}
def Iblock_sum : Inst<"sum"> {
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyType:$result);
}
def Iblock_max_num : Inst<"max_num"> {
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyType:$result);
}

def Iblock_constant : Inst<"constant"> {let arguments=(ins AnyAttr:$value);let results=(outs AnyType:$result);}
def Iblock_maximum : Inst<"maximum"> {let arguments=(ins AnyType:$lhs,AnyType:$rhs);let results=(outs AnyType:$result);}

foreach mnemonic=["xor","and","shl","shr","gt"] in {
  def Iblock_#mnemonic#Op : Op<Iblock_Dialect,mnemonic> {
    let arguments=(ins AnyType:$lhs,AnyType:$rhs); let results=(outs AnyType:$result);
  }
}
def Iblock_select : Op<Iblock_Dialect,"select"> {
  let arguments=(ins AnyType:$predicate,AnyType:$onTrue,AnyType:$onFalse); let results=(outs AnyType:$result);
}
def Iblock_ordinal : Op<Iblock_Dialect,"ordinal"> {
  let arguments=(ins AnyMemRef:$domain,Index:$row); let results=(outs AnyType:$result);
}
def Iblock_linear_index : Op<Iblock_Dialect,"linear_index"> {
  let arguments=(ins AnyMemRef:$domain,AnyTensor:$rows,AnyTensor:$columns); let results=(outs AnyType:$result);
}

def Iblock_sqrt : Inst<"sqrt"> { let arguments=(ins AnyType:$input); let results=(outs AnyType:$result); }

def Iblock_uniform : Inst<"uniform"> {
  let arguments=(ins UI64:$seed,AnyTensor:$counter);let results=(outs AnyTensor:$result);
}
def Iblock_Dialect_welfordOp : Op<Iblock_Dialect,"welford"> {
  let arguments = (ins AnyRankedTensor:$samples); let results = (outs AnyTuple:$summary);
}
def Iblock_Dialect_fieldOp : Op<Iblock_Dialect,"field"> {
  let arguments = (ins AnyTuple:$state, I64Attr:$position); let results = (outs AnyType:$element);
}
def Iblock_Dialect_summary_fieldOp : Op<Iblock_Dialect,"summary_field"> {
  let arguments = (ins AnyTuple:$state, I64Attr:$position); let results = (outs AnyRankedTensor:$element);
}
def Iblock_Dialect_summary_reduceOp : Op<Iblock_Dialect,"summary_reduce"> {
  let arguments = (ins AnyRankedTensor:$samples, I1Tensor:$mask); let results = (outs AnyTuple:$summary);
}
def Iblock_Dialect_summary_identityOp : Op<Iblock_Dialect,"summary_identity"> {
  let arguments = (ins StrAttr:$algebra); let results = (outs AnyTuple:$summary);
}
def Iblock_Dialect_summary_mergeOp : Op<Iblock_Dialect,"summary_merge"> {
  let arguments = (ins AnyTuple:$left, AnyTuple:$right, StrAttr:$algebra); let results = (outs AnyTuple:$summary);
}
def Iblock_Dialect_ScanSum : Op<Iblock_Dialect, "scan_sum", [SameOperandsAndResultType]> {
  let arguments=(ins AnyRankedTensor:$input); let results=(outs AnyRankedTensor:$prefix);
}
def Iblock_Dialect_GatherRow : Op<Iblock_Dialect,"gather_row"> {
  let arguments=(ins AnyMemRef:$table,AnyMemRef:$indices,Index:$owner,UnitAttr:$in_bounds);
  let results=(outs AnyRankedTensor:$fragment);
}
def Iblock_Dialect_IndexedTileRead : Op<Iblock_Dialect,"indexed_tile_read"> {
  let arguments=(ins AnyMemRef:$table,AnyMemRef:$indices,AnyTensor:$rows,AnyTensor:$columns,I1Tensor:$active);
  let results=(outs AnyRankedTensor:$fragment);
}
