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
def Cpu_Dialect_CountKeys : Op<Cpu_Dialect,"count_keys"> {
  let arguments=(ins AnyMemRef:$source,Index:$owner,I64Attr:$span);let results=(outs I32Tensor:$counts);
}
def Cpu_Dialect_ColumnSum : Op<Cpu_Dialect,"column_sum"> {
  let arguments=(ins AnyMemRef:$source,Index:$owner);let results=(outs I32Tensor:$counts);
}
def Cpu_Dialect_Different : Op<Cpu_Dialect,"ne"> {let arguments=(ins AnyType:$lhs,AnyType:$rhs);let results=(outs AnyTensor:$mask);}
def Cpu_Dialect_ColumnsI32 : Op<Cpu_Dialect,"member_index"> {let arguments=(ins AnyMemRef:$domain,Index:$row);let results=(outs I32Tensor:$members);}
def Cpu_Dialect_PrefixScatter : Op<Cpu_Dialect,"prefix_scatter"> {let arguments=(ins AnyMemRef:$output,AnyMemRef:$counts,Index:$owner,I1Tensor:$predicate,I32Tensor:$prefix,I32Tensor:$values,I32Attr:$fill);}
def CpuPackTraversal : Op<Cpu_Dialect,"pack_members"> {let arguments=(ins AnyMemRef:$input,AnyMemRef:$output,AnyMemRef:$counts,F32Attr:$threshold,I32Attr:$fill);}
def Cpu_Dialect_Input : Op<Cpu_Dialect,"index_read"> {let arguments=(ins AnyMemRef:$base,I32Tensor:$key,UnitAttr:$in_bounds);let results=(outs I32Tensor:$values);}
def Cpu_Dialect_Ticket : Op<Cpu_Dialect,"atomic_add"> {let arguments=(ins AnyMemRef:$base,I32Tensor:$key,I32:$delta,StrAttr:$ordering,UnitAttr:$in_bounds);let results=(outs I32Tensor:$values);}
def Cpu_Dialect_Put : Op<Cpu_Dialect,"unique_put"> {let arguments=(ins AnyMemRef:$base,I32Tensor:$key,I32Tensor:$value,UnitAttr:$in_bounds);}
def CpuAtomicTraversal : Op<Cpu_Dialect,"atomic_members",[SingleBlock]> {let regions=(region AnyRegion:$body);}
def Cpu_Dialect_WriteOffsets : Op<Cpu_Dialect,"write_offsets"> {let arguments=(ins AnyMemRef:$offsets,AnyMemRef:$total,Index:$owner,I32Tensor:$prefix);}
def Cpu_Dialect_Ranges : Op<Cpu_Dialect,"fill_ranges"> {let arguments=(ins AnyMemRef:$out,AnyMemRef:$offsets,Index:$owner,I64Attr:$unit,UnitAttr:$disjoint_in_bounds);}
def CpuTally : Op<Cpu_Dialect,"tally_keys"> {let arguments=(ins AnyMemRef:$keys);let results=(outs AnyMemRef:$private_counts);}
def CpuReserve : Op<Cpu_Dialect,"reserve_tickets"> {let arguments=(ins AnyMemRef:$counters,AnyMemRef:$counts,StrAttr:$ordering);let results=(outs AnyMemRef:$private_bases);}
def CpuDistribute : Op<Cpu_Dialect,"distribute_tickets"> {let arguments=(ins AnyMemRef:$keys,AnyMemRef:$offsets,AnyMemRef:$bases,AnyMemRef:$out);}
include "intentdsl/levels/algorithm/kir/ir/Matrix.td"
defm Matrix : MatrixExpressions<Cpu_Dialect>;
include "intentdsl/levels/algorithm/kir/ir/Ordered.td"
defm Ordered : OrderedOwners<Cpu_Dialect>;
def OwnerTasks : Inst<"owner_task",[SingleBlock]> {
  let arguments=(ins Variadic<AnyType>:$captures);let results=(outs Index:$completion);let regions=(region AnyRegion:$body);
}
def MatrixTasks : Inst<"matrix_task",[SingleBlock]> {
  let arguments=(ins Variadic<AnyMemRef>:$captures);let results=(outs Index:$completion);let regions=(region AnyRegion:$body);
}
