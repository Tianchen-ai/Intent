include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
def Cpu_Dialect : Dialect {
  let name = "cpu";
  let cppNamespace = "::intentdsl::cpu";
}
class Inst<string mnemonic, list<Trait> traits = []>
    : Op<Cpu_Dialect, mnemonic, traits>;
def CPU_WelfordOp : Inst<"welford"> { let arguments=(ins AnyRankedTensor:$samples); let results=(outs AnyTuple:$summary); }
def CPU_FieldOp : Inst<"field"> { let arguments=(ins AnyTuple:$summary,I64Attr:$position); let results=(outs AnyType:$element); }
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
def Cpu_OwnerReadOp : Inst<"read_owner"> {
  let arguments = (ins AnyMemRef:$source,Index:$owner); let results = (outs AnyType:$scalar);
}
def Cpu_OwnerWriteOp : Inst<"write_owner"> {
  let arguments = (ins AnyMemRef:$destination,Index:$owner,AnyType:$scalar);
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

def Cpu_constant : Inst<"constant"> {let arguments=(ins AnyAttr:$value);let results=(outs AnyType:$result);}
def Cpu_maximum : Inst<"maximum"> {let arguments=(ins AnyType:$lhs,AnyType:$rhs);let results=(outs AnyType:$result);}

foreach mnemonic=["xor","and","shl","shr","gt"] in {
  def Cpu_#mnemonic#Op : Op<Cpu_Dialect,mnemonic> {
    let arguments=(ins AnyType:$lhs,AnyType:$rhs); let results=(outs AnyType:$result);
  }
}
def Cpu_select : Op<Cpu_Dialect,"select"> {
  let arguments=(ins AnyType:$predicate,AnyType:$onTrue,AnyType:$onFalse); let results=(outs AnyType:$result);
}
def Cpu_ordinal : Op<Cpu_Dialect,"ordinal"> {
  let arguments=(ins AnyMemRef:$domain,Index:$row); let results=(outs AnyType:$result);
}

def Cpu_sqrt : Inst<"sqrt"> { let arguments=(ins AnyType:$input); let results=(outs AnyType:$result); }

def Cpu_uniform : Inst<"uniform"> {
  let arguments=(ins UI64:$seed,AnyTensor:$counter);let results=(outs AnyTensor:$result);
}
def Cpu_Dialect_ScanSum : Op<Cpu_Dialect, "scan_sum", [SameOperandsAndResultType]> {
  let arguments=(ins AnyRankedTensor:$input); let results=(outs AnyRankedTensor:$prefix);
}
def Cpu_Dialect_GatherRow : Op<Cpu_Dialect,"gather_row"> {
  let arguments=(ins AnyMemRef:$table,AnyMemRef:$indices,Index:$owner,UnitAttr:$in_bounds);
  let results=(outs AnyRankedTensor:$fragment);
}
def Cpu_Dialect_Ordered : Op<Cpu_Dialect,"ordered_sum",[SingleBlock]> {
  let arguments=(ins AnyMemRef:$offsets,Index:$owner); let results=(outs F32Tensor:$value); let regions=(region AnyRegion:$terms);
}
def Cpu_Dialect_Length : Op<Cpu_Dialect,"segment_length"> {
  let arguments=(ins AnyMemRef:$offsets,Index:$owner);let results=(outs F32Tensor:$value);
}
def Cpu_Dialect_MemberLoad : Op<Cpu_Dialect,"member_load"> {
  let arguments=(ins AnyMemRef:$source,Variadic<AnyType>:$coordinates);let results=(outs AnyType:$value);
}
def Cpu_Dialect_Term : Op<Cpu_Dialect,"term",[Terminator]> {let arguments=(ins F32:$value);}
