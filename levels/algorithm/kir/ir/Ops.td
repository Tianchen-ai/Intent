include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
def Kir_Dialect : Dialect {
  let name = "kir";
  let cppNamespace = "::intentdsl::kir";
}
class Inst<string mnemonic, list<Trait> traits = []>
    : Op<Kir_Dialect, mnemonic, traits>;
def Kir_parallel : Inst<"parallel"> {
  let arguments = (ins AnyMemRef:$domain);
  let results = (outs);
  let regions = (region AnyRegion:$body);
}
def Kir_end : Inst<"end", [Terminator]> {
  let arguments = (ins);
  let results = (outs);
}
def Kir_read : Inst<"read"> {
  let arguments = (ins AnyMemRef:$view, Variadic<Index>:$indices);
  let results = (outs AnyType:$result);
}
def Kir_write : Inst<"write"> {
  let arguments = (ins AnyMemRef:$view, Index:$row, AnyTensor:$value);
  let results = (outs);
}
def Kir_cast : Inst<"cast"> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
}
def Kir_add : Inst<"add"> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
}
def Kir_sub : Inst<"sub"> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
}
def Kir_mul : Inst<"mul"> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
}
def Kir_div : Inst<"div"> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
}
def Kir_exp : Inst<"exp"> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
}
def Kir_rsqrt : Inst<"rsqrt"> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
}
def Kir_sum : Inst<"sum"> {
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyType:$result);
}
def Kir_max_num : Inst<"max_num"> {
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyType:$result);
}
