include "mlir/IR/OpBase.td"
def RowMojo_Dialect : Dialect {let name="mojo_row"; let cppNamespace="::intentdsl::rowmojo";}
class RowInst<string mnemonic,list<Trait> traits=[]> : Op<RowMojo_Dialect,mnemonic,traits>;
def ScanColumns : RowInst<"sweep",[SingleBlock]> {
 let arguments=(ins Variadic<AnyType>:$initial, StrAttr:$range, I64Attr:$width, StrAttr:$algebra);
 let results=(outs Variadic<AnyType>:$result);let regions=(region AnyRegion:$body);
}
def LaneRead : RowInst<"read"> {let arguments=(ins AnyMemRef:$source,Index:$row,Index:$column);let results=(outs AnyFixedVectorOfNonZeroRank:$result);}
def LaneWrite : RowInst<"write"> {let arguments=(ins AnyMemRef:$dest,Index:$row,Index:$column,AnyFixedVectorOfNonZeroRank:$value);}
def ValueExpr : RowInst<"apply"> {let arguments=(ins Variadic<AnyType>:$inputs,StrAttr:$kind);let results=(outs AnyType:$result);}
def StepYield : RowInst<"yield",[Terminator]> {let arguments=(ins Variadic<AnyType>:$values);}
def OwnerRead : RowInst<"owner_read"> {let arguments=(ins AnyMemRef:$source,Index:$row);let results=(outs AnyType:$value);}
def OwnerWrite : RowInst<"owner_write"> {let arguments=(ins AnyMemRef:$destination,Index:$row,AnyType:$value);}
def MojoIndirectRead : Op<RowMojo_Dialect,"indexed_read"> {
  let arguments=(ins AnyMemRef:$table,AnyMemRef:$ids,Index:$row,Index:$col);
  let results=(outs AnyVectorOfAnyRank:$lanes);
}
