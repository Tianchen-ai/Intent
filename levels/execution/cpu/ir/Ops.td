include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
def Cpu_Dialect : Dialect {
  let name = "cpu";
  let cppNamespace = "::intentdsl::cpu";
}
class Inst<string mnemonic, list<Trait> traits = []>
    : Op<Cpu_Dialect, mnemonic, traits>;
def Cpu_parallel : Inst<"parallel"> {
  let arguments = (ins AnyMemRef:$domain);
  let results = (outs);
  let regions = (region AnyRegion:$body);
}
def Cpu_end : Inst<"end", [Terminator]> {
  let arguments = (ins);
  let results = (outs);
}
def Cpu_read : Inst<"read"> {
  let arguments = (ins AnyMemRef:$view, Variadic<Index>:$indices);
  let results = (outs AnyType:$result);
}
def Cpu_write : Inst<"write"> {
  let arguments = (ins AnyMemRef:$view, Index:$row, AnyTensor:$value);
  let results = (outs);
}
def Cpu_cast : Inst<"cast"> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
}
def Cpu_add : Inst<"add"> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
}
def Cpu_sub : Inst<"sub"> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
}
def Cpu_mul : Inst<"mul"> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
}
def Cpu_div : Inst<"div"> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
}
def Cpu_exp : Inst<"exp"> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
}
def Cpu_rsqrt : Inst<"rsqrt"> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
}
def Cpu_sum : Inst<"sum"> {
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyType:$result);
}
def Cpu_max_num : Inst<"max_num"> {
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyType:$result);
}

def Cpu_task : Inst<"task", [IsolatedFromAbove, SingleBlock]> {
  let arguments = (ins Variadic<AnyType>:$captures, I64Attr:$domain_arg, ArrayAttr:$rows);
  let results = (outs Index:$completion);
  let regions = (region AnyRegion:$body);
}
def Cpu_join : Inst<"join"> { let arguments=(ins Index:$completion); }
