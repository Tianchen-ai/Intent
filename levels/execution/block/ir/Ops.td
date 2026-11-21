include "mlir/IR/OpBase.td"
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
def Iblock_width : Inst<"width"> {
  let arguments = (ins);
  let results = (outs AnyType:$result);
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
