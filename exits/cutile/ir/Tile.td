include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
def TileSource : Dialect { let name="ct_src"; let cppNamespace="::intentdsl::cutile"; }
class Inst<string name,dag args,dag outputs,list<Trait> traits=[]> : Op<TileSource,name,traits> {
  let arguments=args; let results=outputs;
}
def Bid : Inst<"pid",(ins),(outs Index:$id)>;
def OutputPartition : Op<TileSource,"output_partition"> {
  let arguments=(ins I32Attr:$index);
  let regions=(region AnyRegion:$body);
}
def PartitionEnd : Op<TileSource,"partition_end",[Terminator]>;
def Zero : Inst<"zero",(ins),(outs Index:$index)>;
foreach i=["rows","columns"] in {
 def Tile_#i : Inst<i,(ins Index:$origin),(outs AnyTensor:$members)>;
}
def Active : Inst<"active",(ins MemRefRankOf<[AnyType],[1,2,3,4]>:$domain,AnyTensor:$rows,AnyTensor:$columns),(outs I1Tensor:$mask)>;
def TileSource_OwnerLoadOp : Op<TileSource,"owner_load"> {
  let arguments = (ins AnyMemRef:$view,AnyTensor:$rows); let results = (outs AnyTensor:$values);
}
def TileSource_OwnerStoreOp : Op<TileSource,"owner_store"> {
  let arguments = (ins AnyMemRef:$view,AnyTensor:$rows,AnyTensor:$values);
}
def Load : Inst<"tile_load",(ins AnyMemRef:$array,Index:$rowBlock,Index:$columnBlock,BoolAttr:$matrix,BoolAttr:$allow_tma),(outs AnyTensor:$values)>;
def Store : Inst<"tile_store",(ins AnyMemRef:$array,Index:$rowBlock,Index:$columnBlock,AnyTensor:$values,BoolAttr:$matrix,BoolAttr:$allow_tma),(outs)>;
def SlabRead : Inst<"gather_slab",(ins AnyMemRef:$array,AnyTensor:$rows,AnyTensor:$columns,I1Tensor:$active),(outs AnyTensor:$values)>;
def SlabWrite : Inst<"scatter_slab",(ins AnyMemRef:$array,AnyTensor:$rows,AnyTensor:$columns,I1Tensor:$active,AnyTensor:$values),(outs)>;
def Reduce : Inst<"reduce",(ins AnyTensor:$values,I1Tensor:$mask,StrAttr:$algebra),(outs AnyTensor:$summary)>;
def Uniform : Inst<"uniform",(ins UI64:$key,RankedTensorOf<[UI64]>:$position),(outs RankedTensorOf<[F32]>:$value)>;
def Identity : Inst<"identity",(ins StrAttr:$algebra),(outs AnyType:$result)>;
def Combine : Inst<"combine",(ins AnyType:$lhs,AnyType:$rhs,StrAttr:$algebra),(outs AnyType:$result)>;
def Sweep : Inst<"tiles",(ins MemRefRankOf<[AnyType],[1,2,3,4]>:$domain,Variadic<AnyType>:$initial),(outs Variadic<AnyType>:$summary)> {let regions=(region AnyRegion:$body);}
def StripeLoop : Inst<"stripe_loop",(ins Variadic<AnyTensor>:$initial,I64Attr:$count,I64Attr:$stride),(outs Variadic<AnyTensor>:$summary)> {let regions=(region AnyRegion:$body);}
def StripeTileLoad : Inst<"stripe_tile_load",(ins AnyMemRef:$array,Index:$round,Index:$rowBlock,Index:$columnBlock,I64Attr:$stride,BoolAttr:$allow_tma),(outs AnyTensor:$values)>;
def StripeTileStore : Inst<"stripe_tile_store",(ins AnyMemRef:$array,Index:$round,Index:$rowBlock,Index:$columnBlock,AnyTensor:$values,I64Attr:$stride,BoolAttr:$allow_tma),(outs)>;
def StripeOwnerLoad : Inst<"stripe_owner_load",(ins AnyMemRef:$array,Index:$round,AnyTensor:$rows,I64Attr:$stride),(outs AnyTensor:$values)>;
def Yield : Inst<"yield",(ins Variadic<AnyType>:$carry),(outs),[Terminator]>;
foreach i=["broadcast_rows","exp","rsqrt","sqrt","cast"] in {def Tile_#i : Inst<i,(ins AnyType:$x),(outs AnyType:$result)>;}
foreach i=["add","sub","mul","div"] in {def Tile_#i : Inst<i,(ins AnyType:$x,AnyType:$y),(outs AnyType:$result)>;}

def Constant : Inst<"constant",(ins AnyAttr:$value),(outs AnyType:$result)>;
def Maximum : Inst<"maximum",(ins AnyType:$a,AnyType:$b),(outs AnyType:$result)>;

foreach mnemonic=["xor","and","shl","shr","gt"] in {
  def Tile_#mnemonic#Op : Op<TileSource,mnemonic> {
    let arguments=(ins AnyType:$lhs,AnyType:$rhs); let results=(outs AnyType:$result);
  }
}
def Tile_select : Op<TileSource,"select"> {
  let arguments=(ins AnyType:$predicate,AnyType:$onTrue,AnyType:$onFalse); let results=(outs AnyType:$result);
}

def Tile_linear_index : Op<TileSource,"linear_index"> {
  let arguments=(ins AnyMemRef:$domain,AnyTensor:$rows,AnyTensor:$columns); let results=(outs AnyType:$result);
}
def TileSource_summary_fieldOp : Op<TileSource,"summary_field"> {
  let arguments = (ins AnyTuple:$state, I64Attr:$position); let results = (outs AnyRankedTensor:$element);
}
def TileSource_summary_reduceOp : Op<TileSource,"summary_reduce"> {
  let arguments = (ins AnyRankedTensor:$samples, I1Tensor:$mask); let results = (outs AnyTuple:$summary);
}
def TileSource_summary_identityOp : Op<TileSource,"summary_identity"> {
  let arguments = (ins StrAttr:$algebra); let results = (outs AnyTuple:$summary);
}
def TileSource_summary_mergeOp : Op<TileSource,"summary_merge"> {
  let arguments = (ins AnyTuple:$left, AnyTuple:$right, StrAttr:$algebra); let results = (outs AnyTuple:$summary);
}
def TileSource_ScanSum : Op<TileSource, "scan_sum", [SameOperandsAndResultType]> {
  let arguments=(ins AnyRankedTensor:$input); let results=(outs AnyRankedTensor:$prefix);
}
def TileSource_IndexedTileRead : Op<TileSource,"indexed_tile_read"> {
  let arguments=(ins AnyMemRef:$table,AnyMemRef:$indices,AnyTensor:$rows,AnyTensor:$columns,I1Tensor:$active);
  let results=(outs AnyRankedTensor:$fragment);
}
def TileSource_MemberLoad : Op<TileSource,"member_load"> {
  let arguments=(ins AnyMemRef:$source,Variadic<AnyType>:$coordinates);let results=(outs AnyType:$value);
}
def TileSource_Term : Op<TileSource,"term",[Terminator]> {let arguments=(ins F32:$value);}
def TileSource_SegmentFold : Op<TileSource,"segment_fold",[SingleBlock]> {
  let arguments=(ins AnyMemRef:$offsets,AnyTensor:$rows,AnyTensor:$cols,I1Tensor:$active,UnitAttr:$strict);
  let results=(outs F32Tensor:$value);let regions=(region AnyRegion:$terms);
}
def TileSource_SegmentSize : Op<TileSource,"segment_size"> {
  let arguments=(ins AnyMemRef:$offsets,AnyTensor:$rows,AnyTensor:$cols,I1Tensor:$active);let results=(outs F32Tensor:$value);
}
def TileSource_KeyCounts : Op<TileSource,"key_counts"> {
  let arguments=(ins AnyMemRef:$source,AnyTensor:$rows,AnyTensor:$columns,I1Tensor:$active,I64Attr:$extent,I64Attr:$span);let results=(outs I32Tensor:$counts);
}
def TileSource_SumColumns : Op<TileSource,"sum_columns"> {
  let arguments=(ins AnyMemRef:$source,AnyTensor:$rows,AnyTensor:$columns,I1Tensor:$active,I64Attr:$extent);let results=(outs I32Tensor:$counts);
}
def TileSource_PrivateBinStore : Op<TileSource,"private_bin_store",[SingleBlock,NoTerminator]> {
  let arguments=(ins AnyMemRef:$keys,AnyMemRef:$dest,AnyTensor:$rows,AnyTensor:$columns,I1Tensor:$active,I1:$non_empty,I64Attr:$extent,I64Attr:$span,I64Attr:$source_step,BoolAttr:$transposed);
  let regions=(region AnyRegion:$body);
}
def ArrayDim : Inst<"array_dim",(ins AnyMemRef:$array,I64Attr:$axis),(outs I64:$size),[Pure]>;
def ZeroBins : Op<TileSource,"zero_bins",[MemoryEffects<[MemWrite]>]> {let arguments=(ins AnyMemRef:$dest,BoolAttr:$transposed);}
def BinPanels : Op<TileSource,"bin_panels",[SingleBlock,NoTerminator]> {let arguments=(ins I32:$limit);let regions=(region AnyRegion:$body);}
def ReadBinKeys : Op<TileSource,"read_bin_keys",[MemoryEffects<[MemRead]>]> {let arguments=(ins AnyMemRef:$source,I64:$panel);let results=(outs AnyRankedTensor:$keys);}
def BinIndex : Op<TileSource,"bin_index",[Pure]> {let arguments=(ins AnyRankedTensor:$raw);let results=(outs AnyRankedTensor:$key);}
def AtomicBins : Op<TileSource,"atomic_bins",[MemoryEffects<[MemRead,MemWrite]>]> {let arguments=(ins AnyMemRef:$dest,AnyRankedTensor:$keys,I32Tensor:$increment,BoolAttr:$transposed,StrAttr:$scope);}
def TileSource_Different : Op<TileSource,"ne"> {let arguments=(ins AnyType:$lhs,AnyType:$rhs);let results=(outs AnyTensor:$mask);}
def TileSource_ColumnsTile : Op<TileSource,"member_tile"> {let arguments=(ins AnyTensor:$columns,I1Tensor:$active);let results=(outs I32Tensor:$members);}
def TileSource_ScatterFragment : Op<TileSource,"scatter_fragment"> {let arguments=(ins AnyMemRef:$output,AnyMemRef:$counts,AnyTensor:$rows,AnyTensor:$columns,I1Tensor:$active,I1Tensor:$predicate,I32Tensor:$prefix,I32Tensor:$values,I32Attr:$fill);}
def TileSource_InputTile : Op<TileSource,"index_read_at"> {let arguments=(ins AnyMemRef:$base,I32Tensor:$key,I1Tensor:$valid,UnitAttr:$in_bounds);let results=(outs I32Tensor:$values);}
def TileSource_TicketTile : Op<TileSource,"atomic_add_at"> {let arguments=(ins AnyMemRef:$base,I32Tensor:$key,I32:$delta,I1Tensor:$valid,StrAttr:$ordering,UnitAttr:$in_bounds);let results=(outs I32Tensor:$values);}
def TileSource_PutTile : Op<TileSource,"unique_put_at"> {let arguments=(ins AnyMemRef:$base,I32Tensor:$key,I32Tensor:$value,I1Tensor:$valid,UnitAttr:$in_bounds);}
def TileSource_PublishOffsets : Op<TileSource,"publish_offsets"> {let arguments=(ins AnyMemRef:$offsets,AnyMemRef:$total,AnyTensor:$columns,I1Tensor:$active,I32Tensor:$prefix);}
def TileSource_RangesAt : Op<TileSource,"fill_ranges_at"> {let arguments=(ins AnyMemRef:$out,AnyMemRef:$offsets,AnyTensor:$columns,I1Tensor:$active,I64Attr:$unit,UnitAttr:$disjoint_in_bounds);}
include "intentdsl/levels/algorithm/kir/ir/Matrix.td"
defm Matrix : MatrixExpressions<TileSource>;
include "intentdsl/levels/algorithm/kir/ir/Ordered.td"
defm Ordered : OrderedOwners<TileSource>;
include "intentdsl/levels/execution/block/ir/ScorePanel.td"
defm Score : ScoreTiles<TileSource>;
include "intentdsl/levels/execution/block/ir/Columns.td"
defm Columns : ColumnTiles<TileSource>;
defm NativePanels : MatrixPanelLoop<TileSource>;
include "intentdsl/levels/execution/block/ir/ScaleDispatch.td"
defm Scale : ScaleDispatch<TileSource>;
def MatrixEnd : Op<TileSource,"end",[Terminator]>;
class TileInstruction<string mnemonic, list<Trait> traits = []> : Op<TileSource, mnemonic, traits>;
include "intentdsl/levels/execution/block/ir/TileInstructions.td"
def NativeRectRead : Op<TileSource,"rect_read",[MemoryEffects<[MemRead]>]> {
  let arguments=(ins AnyMemRef:$base, Variadic<I64>:$indices,BoolAttr:$allow_tma);
  let results=(outs AnyRankedTensor:$tile);
}
def NativeScaleMMA : Op<TileSource,"scaled_mma",[Pure]> {
  let arguments=(ins AnyRankedTensor:$lhs,AnyRankedTensor:$ls,AnyRankedTensor:$rhs,AnyRankedTensor:$rs,AnyRankedTensor:$acc);
  let results=(outs AnyRankedTensor:$value);
}
def ArrayReshape : Inst<"reshape",(ins AnyRankedTensor:$value),(outs AnyRankedTensor:$result),[Pure]>;
def ArrayTranspose : Inst<"transpose",(ins AnyRankedTensor:$value),(outs AnyRankedTensor:$result),[Pure]>;
def ExpandScale : Inst<"expand_scale",(ins AnyRankedTensor:$value,I64Attr:$axis),(outs AnyRankedTensor:$result),[Pure]>;
include "intentdsl/levels/execution/block/ir/RowInstructions.td"
def RuntimeSource : Op<TileSource,"runtime_source"> {let arguments=(ins StrAttr:$implementation);}
def RectWrite : Op<TileSource,"rect_write",[MemoryEffects<[MemWrite]>]> {let arguments=(ins AnyMemRef:$base, AnyRankedTensor:$value, Variadic<I64>:$indices);}
