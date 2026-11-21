include "mlir/IR/OpBase.td"
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
def Itsrc_neutralize : Inst<"neutralize"> {
  let arguments = (ins AnyTensor:$value, AnyTensor:$mask, StrAttr:$algebra);
  let results = (outs AnyTensor:$result);
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
