include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
def Itsrc_Dialect : Dialect {
  let name = "triton_src";
  let cppNamespace = "::intentdsl::itsrc";
}
class Inst<string mnemonic, list<Trait> traits = []>
    : Op<Itsrc_Dialect, mnemonic, traits>;
def Itsrc_pid : Inst<"pid"> {
  let arguments = (ins);
  let results = (outs Index:$result);
}
def Itsrc_rows : Inst<"rows"> {
  let arguments = (ins Index:$instance);
  let results = (outs AnyType:$result);
}
def Itsrc_columns : Inst<"columns"> {
  let arguments = (ins Index:$tile);
  let results = (outs AnyType:$result);
}
def Itsrc_active : Inst<"active"> {
  let arguments = (ins AnyMemRef:$domain, AnyTensor:$rows, AnyTensor:$columns);
  let results = (outs AnyType:$result);
}
def Itsrc_offsets : Inst<"offsets"> {
  let arguments = (ins AnyMemRef:$view, AnyTensor:$rows, AnyTensor:$columns, BoolAttr:$matrix);
  let results = (outs AnyTensor:$result);
}
def Itsrc_load : Inst<"load"> {
  let arguments = (ins AnyMemRef:$view, AnyTensor:$offsets, AnyTensor:$mask);
  let results = (outs AnyTensor:$result);
}
def Itsrc_store : Inst<"store"> {
  let arguments = (ins AnyMemRef:$view, AnyTensor:$offsets, AnyTensor:$mask,
      AnyTensor:$value);
  let results = (outs);
}
def Itsrc_Dialect_OwnerLoadOp : Op<Itsrc_Dialect,"owner_load"> {
  let arguments = (ins AnyMemRef:$view,AnyTensor:$rows); let results = (outs AnyTensor:$values);
}
def Itsrc_Dialect_OwnerStoreOp : Op<Itsrc_Dialect,"owner_store"> {
  let arguments = (ins AnyMemRef:$view,AnyTensor:$rows,AnyTensor:$values);
}
def Itsrc_neutralize : Inst<"neutralize"> {
  let arguments = (ins AnyTensor:$value, AnyTensor:$mask, StrAttr:$algebra);
  let results = (outs AnyTensor:$result);
}
def Itsrc_uniform : Inst<"uniform"> {
  let arguments = (ins UI64:$key, RankedTensorOf<[UI64]>:$position);
  let results = (outs RankedTensorOf<[F32]>:$value);
}
def Itsrc_reduce : Inst<"reduce"> {
  let arguments = (ins AnyTensor:$value, StrAttr:$algebra);
  let results = (outs AnyType:$result);
}
def Itsrc_tiles : Inst<"tiles"> {
  let arguments = (ins AnyMemRef:$domain, Variadic<AnyType>:$initial);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$body);
}
def Itsrc_yield : Inst<"yield", [Terminator]> {
  let arguments = (ins Variadic<AnyType>:$values);
  let results = (outs);
}
def Itsrc_zero : Inst<"zero"> {
  let arguments = (ins);
  let results = (outs Index:$result);
}
def Itsrc_combine : Inst<"combine"> {
  let arguments = (ins AnyTensor:$acc, AnyTensor:$partial, StrAttr:$algebra);
  let results = (outs AnyType:$result);
}
def Itsrc_identity : Inst<"identity"> {
  let arguments = (ins StrAttr:$algebra);
  let results = (outs AnyType:$result);
}
def Itsrc_broadcast_rows : Inst<"broadcast_rows"> {
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
}
def Itsrc_cast : Inst<"cast"> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
}
def Itsrc_add : Inst<"add"> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
}
def Itsrc_sub : Inst<"sub"> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
}
def Itsrc_mul : Inst<"mul"> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
}
def Itsrc_div : Inst<"div"> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
}
def Itsrc_exp : Inst<"exp"> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
}
def Itsrc_rsqrt : Inst<"rsqrt"> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
}

def Itsrc_constant : Inst<"constant"> {let arguments=(ins AnyAttr:$value);let results=(outs AnyType:$result);}
def Itsrc_maximum : Inst<"maximum"> {let arguments=(ins AnyType:$lhs,AnyType:$rhs);let results=(outs AnyType:$result);}

foreach mnemonic=["xor","and","shl","shr","gt"] in {
  def Itsrc_#mnemonic#Op : Op<Itsrc_Dialect,mnemonic> {
    let arguments=(ins AnyType:$lhs,AnyType:$rhs); let results=(outs AnyType:$result);
  }
}
def Itsrc_select : Op<Itsrc_Dialect,"select"> {
  let arguments=(ins AnyType:$predicate,AnyType:$onTrue,AnyType:$onFalse); let results=(outs AnyType:$result);
}

def Itsrc_linear_index : Op<Itsrc_Dialect,"linear_index"> {
  let arguments=(ins AnyMemRef:$domain,AnyTensor:$rows,AnyTensor:$columns); let results=(outs AnyType:$result);
}

def Itsrc_sqrt : Inst<"sqrt"> { let arguments=(ins AnyType:$input); let results=(outs AnyType:$result); }
def Itsrc_Dialect_summary_fieldOp : Op<Itsrc_Dialect,"summary_field"> {
  let arguments = (ins AnyTuple:$state, I64Attr:$position); let results = (outs AnyRankedTensor:$element);
}
def Itsrc_Dialect_summary_reduceOp : Op<Itsrc_Dialect,"summary_reduce"> {
  let arguments = (ins AnyRankedTensor:$samples, I1Tensor:$mask); let results = (outs AnyTuple:$summary);
}
def Itsrc_Dialect_summary_identityOp : Op<Itsrc_Dialect,"summary_identity"> {
  let arguments = (ins StrAttr:$algebra); let results = (outs AnyTuple:$summary);
}
def Itsrc_Dialect_summary_mergeOp : Op<Itsrc_Dialect,"summary_merge"> {
  let arguments = (ins AnyTuple:$left, AnyTuple:$right, StrAttr:$algebra); let results = (outs AnyTuple:$summary);
}
def Itsrc_Dialect_ScanSum : Op<Itsrc_Dialect, "scan_sum", [SameOperandsAndResultType]> {
  let arguments=(ins AnyRankedTensor:$input); let results=(outs AnyRankedTensor:$prefix);
}
def Itsrc_Dialect_IndexedTileRead : Op<Itsrc_Dialect,"indexed_tile_read"> {
  let arguments=(ins AnyMemRef:$table,AnyMemRef:$indices,AnyTensor:$rows,AnyTensor:$columns,I1Tensor:$active);
  let results=(outs AnyRankedTensor:$fragment);
}
def Itsrc_Dialect_MemberLoad : Op<Itsrc_Dialect,"member_load"> {
  let arguments=(ins AnyMemRef:$source,Variadic<AnyType>:$coordinates);let results=(outs AnyType:$value);
}
def Itsrc_Dialect_Term : Op<Itsrc_Dialect,"term",[Terminator]> {let arguments=(ins F32:$value);}
def Itsrc_Dialect_SegmentFold : Op<Itsrc_Dialect,"segment_fold",[SingleBlock]> {
  let arguments=(ins AnyMemRef:$offsets,AnyTensor:$rows,AnyTensor:$cols,I1Tensor:$active,UnitAttr:$strict);
  let results=(outs F32Tensor:$value);let regions=(region AnyRegion:$terms);
}
def Itsrc_Dialect_SegmentSize : Op<Itsrc_Dialect,"segment_size"> {
  let arguments=(ins AnyMemRef:$offsets,AnyTensor:$rows,AnyTensor:$cols,I1Tensor:$active);let results=(outs F32Tensor:$value);
}
