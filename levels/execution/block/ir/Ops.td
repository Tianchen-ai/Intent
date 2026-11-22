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
def Iblock_Dialect_Ordered : Op<Iblock_Dialect,"ordered_sum",[SingleBlock]> {
  let arguments=(ins AnyMemRef:$offsets,Index:$owner); let results=(outs F32Tensor:$value); let regions=(region AnyRegion:$terms);
}
def Iblock_Dialect_Length : Op<Iblock_Dialect,"segment_length"> {
  let arguments=(ins AnyMemRef:$offsets,Index:$owner);let results=(outs F32Tensor:$value);
}
def Iblock_Dialect_MemberLoad : Op<Iblock_Dialect,"member_load"> {
  let arguments=(ins AnyMemRef:$source,Variadic<AnyType>:$coordinates);let results=(outs AnyType:$value);
}
def Iblock_Dialect_Term : Op<Iblock_Dialect,"term",[Terminator]> {let arguments=(ins F32:$value);}
def Iblock_Dialect_SegmentFold : Op<Iblock_Dialect,"segment_fold",[SingleBlock]> {
  let arguments=(ins AnyMemRef:$offsets,AnyTensor:$rows,AnyTensor:$cols,I1Tensor:$active,UnitAttr:$strict);
  let results=(outs F32Tensor:$value);let regions=(region AnyRegion:$terms);
}
def Iblock_Dialect_SegmentSize : Op<Iblock_Dialect,"segment_size"> {
  let arguments=(ins AnyMemRef:$offsets,AnyTensor:$rows,AnyTensor:$cols,I1Tensor:$active);let results=(outs F32Tensor:$value);
}
def Iblock_Dialect_CountKeys : Op<Iblock_Dialect,"count_keys"> {
  let arguments=(ins AnyMemRef:$source,Index:$owner,I64Attr:$span);let results=(outs I32Tensor:$counts);
}
def Iblock_Dialect_ColumnSum : Op<Iblock_Dialect,"column_sum"> {
  let arguments=(ins AnyMemRef:$source,Index:$owner);let results=(outs I32Tensor:$counts);
}
def Iblock_Dialect_KeyCounts : Op<Iblock_Dialect,"key_counts"> {
  let arguments=(ins AnyMemRef:$source,AnyTensor:$rows,AnyTensor:$columns,I1Tensor:$active,I64Attr:$extent,I64Attr:$span);let results=(outs I32Tensor:$counts);
}
def Iblock_Dialect_SumColumns : Op<Iblock_Dialect,"sum_columns"> {
  let arguments=(ins AnyMemRef:$source,AnyTensor:$rows,AnyTensor:$columns,I1Tensor:$active,I64Attr:$extent);let results=(outs I32Tensor:$counts);
}
def Iblock_Dialect_Different : Op<Iblock_Dialect,"ne"> {let arguments=(ins AnyType:$lhs,AnyType:$rhs);let results=(outs AnyTensor:$mask);}
def Iblock_Dialect_ColumnsI32 : Op<Iblock_Dialect,"member_index"> {let arguments=(ins AnyMemRef:$domain,Index:$row);let results=(outs I32Tensor:$members);}
def Iblock_Dialect_PrefixScatter : Op<Iblock_Dialect,"prefix_scatter"> {let arguments=(ins AnyMemRef:$output,AnyMemRef:$counts,Index:$owner,I1Tensor:$predicate,I32Tensor:$prefix,I32Tensor:$values,I32Attr:$fill);}
def Iblock_Dialect_ColumnsTile : Op<Iblock_Dialect,"member_tile"> {let arguments=(ins AnyTensor:$columns,I1Tensor:$active);let results=(outs I32Tensor:$members);}
def Iblock_Dialect_ScatterFragment : Op<Iblock_Dialect,"scatter_fragment"> {let arguments=(ins AnyMemRef:$output,AnyMemRef:$counts,AnyTensor:$rows,AnyTensor:$columns,I1Tensor:$active,I1Tensor:$predicate,I32Tensor:$prefix,I32Tensor:$values,I32Attr:$fill);}
def Iblock_Dialect_Input : Op<Iblock_Dialect,"index_read"> {let arguments=(ins AnyMemRef:$base,I32Tensor:$key,UnitAttr:$in_bounds);let results=(outs I32Tensor:$values);}
def Iblock_Dialect_InputTile : Op<Iblock_Dialect,"index_read_at"> {let arguments=(ins AnyMemRef:$base,I32Tensor:$key,I1Tensor:$valid,UnitAttr:$in_bounds);let results=(outs I32Tensor:$values);}
def Iblock_Dialect_Ticket : Op<Iblock_Dialect,"atomic_add"> {let arguments=(ins AnyMemRef:$base,I32Tensor:$key,I32:$delta,StrAttr:$ordering,UnitAttr:$in_bounds);let results=(outs I32Tensor:$values);}
def Iblock_Dialect_TicketTile : Op<Iblock_Dialect,"atomic_add_at"> {let arguments=(ins AnyMemRef:$base,I32Tensor:$key,I32:$delta,I1Tensor:$valid,StrAttr:$ordering,UnitAttr:$in_bounds);let results=(outs I32Tensor:$values);}
def Iblock_Dialect_Put : Op<Iblock_Dialect,"unique_put"> {let arguments=(ins AnyMemRef:$base,I32Tensor:$key,I32Tensor:$value,UnitAttr:$in_bounds);}
def Iblock_Dialect_PutTile : Op<Iblock_Dialect,"unique_put_at"> {let arguments=(ins AnyMemRef:$base,I32Tensor:$key,I32Tensor:$value,I1Tensor:$valid,UnitAttr:$in_bounds);}
def Iblock_Dialect_WriteOffsets : Op<Iblock_Dialect,"write_offsets"> {let arguments=(ins AnyMemRef:$offsets,AnyMemRef:$total,Index:$owner,I32Tensor:$prefix);}
def Iblock_Dialect_PublishOffsets : Op<Iblock_Dialect,"publish_offsets"> {let arguments=(ins AnyMemRef:$offsets,AnyMemRef:$total,AnyTensor:$columns,I1Tensor:$active,I32Tensor:$prefix);}
def Iblock_Dialect_Ranges : Op<Iblock_Dialect,"fill_ranges"> {let arguments=(ins AnyMemRef:$out,AnyMemRef:$offsets,Index:$owner,I64Attr:$unit,UnitAttr:$disjoint_in_bounds);}
def Iblock_Dialect_RangesAt : Op<Iblock_Dialect,"fill_ranges_at"> {let arguments=(ins AnyMemRef:$out,AnyMemRef:$offsets,AnyTensor:$columns,I1Tensor:$active,I64Attr:$unit,UnitAttr:$disjoint_in_bounds);}
include "intentdsl/levels/algorithm/kir/ir/Matrix.td"
defm Matrix : MatrixExpressions<Iblock_Dialect>;
