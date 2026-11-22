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
def Itsrc_Dialect_KeyCounts : Op<Itsrc_Dialect,"key_counts"> {
  let arguments=(ins AnyMemRef:$source,AnyTensor:$rows,AnyTensor:$columns,I1Tensor:$active,I64Attr:$extent,I64Attr:$span);let results=(outs I32Tensor:$counts);
}
def Itsrc_AtomicBins : Op<Itsrc_Dialect,"atomic_bins",[SingleBlock,NoTerminator]> {
  let arguments=(ins AnyMemRef:$source, AnyMemRef:$dest, AnyTensor:$columns, I64Attr:$extent, I64Attr:$span, I64Attr:$native_panel, I32Attr:$address_bits,BoolAttr:$transposed);
  let regions=(region AnyRegion:$body);
}
def Itsrc_ClearBins : Op<Itsrc_Dialect,"clear_bins">;
def Itsrc_CtaBarrier : Op<Itsrc_Dialect,"cta_barrier">;
def Itsrc_BinPanels : Op<Itsrc_Dialect,"bin_panels",[SingleBlock,NoTerminator]> {
  let arguments=(ins I64Attr:$count);
  let regions=(region AnyRegion:$body);
}
def Itsrc_BinAdd : Op<Itsrc_Dialect,"bin_add"> { let arguments=(ins Index:$piece); }
def Itsrc_Dialect_SumColumns : Op<Itsrc_Dialect,"sum_columns"> {
  let arguments=(ins AnyMemRef:$source,AnyTensor:$rows,AnyTensor:$columns,I1Tensor:$active,I64Attr:$extent);let results=(outs I32Tensor:$counts);
}
def Itsrc_Dialect_Different : Op<Itsrc_Dialect,"ne"> {let arguments=(ins AnyType:$lhs,AnyType:$rhs);let results=(outs AnyTensor:$mask);}
def Itsrc_Dialect_ColumnsTile : Op<Itsrc_Dialect,"member_tile"> {let arguments=(ins AnyTensor:$columns,I1Tensor:$active);let results=(outs I32Tensor:$members);}
def Itsrc_Dialect_ScatterFragment : Op<Itsrc_Dialect,"scatter_fragment"> {let arguments=(ins AnyMemRef:$output,AnyMemRef:$counts,AnyTensor:$rows,AnyTensor:$columns,I1Tensor:$active,I1Tensor:$predicate,I32Tensor:$prefix,I32Tensor:$values,I32Attr:$fill);}
def Itsrc_Dialect_InputTile : Op<Itsrc_Dialect,"index_read_at"> {let arguments=(ins AnyMemRef:$base,I32Tensor:$key,I1Tensor:$valid,UnitAttr:$in_bounds);let results=(outs I32Tensor:$values);}
def Itsrc_Dialect_TicketTile : Op<Itsrc_Dialect,"atomic_add_at"> {let arguments=(ins AnyMemRef:$base,I32Tensor:$key,I32:$delta,I1Tensor:$valid,StrAttr:$ordering,UnitAttr:$in_bounds);let results=(outs I32Tensor:$values);}
def Itsrc_Dialect_PutTile : Op<Itsrc_Dialect,"unique_put_at"> {let arguments=(ins AnyMemRef:$base,I32Tensor:$key,I32Tensor:$value,I1Tensor:$valid,UnitAttr:$in_bounds);}
def Itsrc_Dialect_PublishOffsets : Op<Itsrc_Dialect,"publish_offsets"> {let arguments=(ins AnyMemRef:$offsets,AnyMemRef:$total,AnyTensor:$columns,I1Tensor:$active,I32Tensor:$prefix);}
def Itsrc_Dialect_RangesAt : Op<Itsrc_Dialect,"fill_ranges_at"> {let arguments=(ins AnyMemRef:$out,AnyMemRef:$offsets,AnyTensor:$columns,I1Tensor:$active,I64Attr:$unit,UnitAttr:$disjoint_in_bounds);}
include "intentdsl/levels/algorithm/kir/ir/Matrix.td"
defm Matrix : MatrixExpressions<Itsrc_Dialect>;
include "intentdsl/levels/algorithm/kir/ir/Ordered.td"
defm Ordered : OrderedOwners<Itsrc_Dialect>;
include "intentdsl/levels/execution/block/ir/Columns.td"
defm Columns : ColumnTiles<Itsrc_Dialect>;
defm NativePanels : MatrixPanelLoop<Itsrc_Dialect>;
include "intentdsl/levels/execution/block/ir/ScaleDispatch.td"
defm Scale : ScaleDispatch<Itsrc_Dialect>;
def MatrixEnd : Op<Itsrc_Dialect,"end",[Terminator]>;
