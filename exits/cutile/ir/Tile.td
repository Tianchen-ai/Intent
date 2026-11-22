include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
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
