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
def Active : Inst<"active",(ins AnyMemRef:$domain,AnyTensor:$rows,AnyTensor:$columns),(outs AnyTensor:$mask)>;
def Load : Inst<"tile_load",(ins AnyMemRef:$array,Index:$rowBlock,Index:$columnBlock,BoolAttr:$matrix,BoolAttr:$allow_tma),(outs AnyTensor:$values)>;
def Store : Inst<"tile_store",(ins AnyMemRef:$array,Index:$rowBlock,Index:$columnBlock,AnyTensor:$values,BoolAttr:$matrix,BoolAttr:$allow_tma),(outs)>;
def Reduce : Inst<"reduce",(ins AnyTensor:$values,AnyTensor:$mask,StrAttr:$algebra),(outs AnyTensor:$summary)>;
def Identity : Inst<"identity",(ins StrAttr:$algebra),(outs AnyType:$result)>;
def Combine : Inst<"combine",(ins AnyType:$lhs,AnyType:$rhs,StrAttr:$algebra),(outs AnyType:$result)>;
def Sweep : Inst<"tiles",(ins AnyMemRef:$domain,Variadic<AnyType>:$initial),(outs Variadic<AnyType>:$summary)> {let regions=(region AnyRegion:$body);}
def Yield : Inst<"yield",(ins Variadic<AnyType>:$carry),(outs),[Terminator]>;
foreach i=["broadcast_rows","exp","rsqrt","cast"] in {def Tile_#i : Inst<i,(ins AnyType:$x),(outs AnyType:$result)>;}
foreach i=["add","sub","mul","div"] in {def Tile_#i : Inst<i,(ins AnyType:$x,AnyType:$y),(outs AnyType:$result)>;}
