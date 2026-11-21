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

// The empty summary is (0, 0, 0). The mean and centered second moment
// describe the same population as the integer count.
def Kir_welford : Inst<"welford", [Pure]> {
  let arguments = (ins AnyRankedTensor:$samples);
  let results = (outs AnyType:$state);
  let hasVerifier = 1;
}
def Kir_field : Inst<"field", [Pure]> {
  let arguments = (ins AnyType:$record, I64Attr:$position);
  let results = (outs AnyType:$field);
  let hasVerifier = 1;
}

def Kir_constant : Inst<"constant"> {let arguments=(ins AnyAttr:$value);let results=(outs AnyType:$result);}
def Kir_maximum : Inst<"maximum"> {let arguments=(ins AnyType:$lhs,AnyType:$rhs);let results=(outs AnyType:$result);}

foreach mnemonic=["xor","and","shl","shr","gt"] in {
  def Kir_#mnemonic#Op : Op<Kir_Dialect,mnemonic> {
    let arguments=(ins AnyType:$lhs,AnyType:$rhs); let results=(outs AnyType:$result);
  }
}
def Kir_select : Op<Kir_Dialect,"select"> {
  let arguments=(ins AnyType:$predicate,AnyType:$onTrue,AnyType:$onFalse); let results=(outs AnyType:$result);
}
def Kir_ordinal : Op<Kir_Dialect,"ordinal"> {
  let arguments=(ins AnyMemRef:$domain,Index:$row); let results=(outs AnyType:$result);
}

def Kir_sqrt : Inst<"sqrt"> { let arguments=(ins AnyType:$input); let results=(outs AnyType:$result); }

def Kir_uniform : Inst<"uniform"> {
  let arguments=(ins UI64:$seed,AnyTensor:$counter);let results=(outs AnyTensor:$result);
}
