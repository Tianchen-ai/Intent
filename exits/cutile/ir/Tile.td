include "mlir/IR/OpBase.td"
def TileSource : Dialect { let name="ct_src"; let cppNamespace="::intentdsl::cutile"; }
class Inst<string name,dag args,dag outputs,list<Trait> traits=[]> : Op<TileSource,name,traits> {
  let arguments=args; let results=outputs;
}
def Bid : Inst<"pid",(ins),(outs Index:$id)>;
def Zero : Inst<"zero",(ins),(outs Index:$index)>;
foreach i=["rows","columns"] in {
 def Tile_#i : Inst<i,(ins Index:$origin),(outs AnyTensor:$members)>;
}
def Active : Inst<"active",(ins MemRefRankOf<[AnyType],[1,2,3,4]>:$domain,AnyTensor:$rows,AnyTensor:$columns),(outs I1Tensor:$mask)>;
def Load : Inst<"tile_load",(ins AnyMemRef:$array,Index:$rowBlock,Index:$columnBlock,BoolAttr:$matrix,BoolAttr:$allow_tma),(outs AnyTensor:$values)>;
def Store : Inst<"tile_store",(ins AnyMemRef:$array,Index:$rowBlock,Index:$columnBlock,AnyTensor:$values,BoolAttr:$matrix,BoolAttr:$allow_tma),(outs)>;
def Reduce : Inst<"reduce",(ins AnyTensor:$values,I1Tensor:$mask,StrAttr:$algebra),(outs AnyTensor:$summary)>;
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
