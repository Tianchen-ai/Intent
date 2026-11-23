include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
def Kir_Dialect : Dialect {
  let name = "kir";
  let cppNamespace = "::intentdsl::kir";
  let useDefaultTypePrinterParser = 1;
}
class StorageFormat<string name, string spelling> : TypeDef<Kir_Dialect, name> {
  let mnemonic = spelling;
  let assemblyFormat = "";
}
def Q4K : StorageFormat<"Q4K", "q4_k"> {
  let extraClassDeclaration = [{ ::intentdsl::kir::Q4KRecord getLayout() const { return {}; } }];
}
def Q8K : StorageFormat<"Q8K", "q8_k"> {
  let extraClassDeclaration = [{ ::intentdsl::kir::Q8KRecord getLayout() const { return {}; } }];
}
def MX8 : StorageFormat<"MX8", "mx_e4m3_e8m0"> {
  let extraClassDeclaration = [{ ::intentdsl::kir::MX8Block getLayout() const { return {}; } }];
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
def Kir_OwnerReadOp : Inst<"read_owner"> {
  let arguments = (ins AnyMemRef:$source,Index:$owner); let results = (outs AnyType:$scalar);
}
def Kir_OwnerWriteOp : Inst<"write_owner"> {
  let arguments = (ins AnyMemRef:$destination,Index:$owner,AnyType:$scalar);
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
  let results = (outs AnyType:$element);
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
def Kir_Dialect_ScanSum : Op<Kir_Dialect, "scan_sum", [SameOperandsAndResultType]> {
  let arguments=(ins AnyRankedTensor:$input); let results=(outs AnyRankedTensor:$prefix);
}
def Kir_Dialect_GatherRow : Op<Kir_Dialect,"gather_row"> {
  let arguments=(ins AnyMemRef:$table,AnyMemRef:$indices,Index:$owner,UnitAttr:$in_bounds);
  let results=(outs AnyRankedTensor:$fragment);
}
def Kir_Dialect_Ordered : Op<Kir_Dialect,"ordered_sum",[SingleBlock]> {
  let arguments=(ins AnyMemRef:$offsets,Index:$owner); let results=(outs F32Tensor:$value); let regions=(region AnyRegion:$terms);
}
def Kir_Dialect_Length : Op<Kir_Dialect,"segment_length"> {
  let arguments=(ins AnyMemRef:$offsets,Index:$owner);let results=(outs F32Tensor:$value);
}
def Kir_Dialect_MemberLoad : Op<Kir_Dialect,"member_load"> {
  let arguments=(ins AnyMemRef:$source,Variadic<AnyType>:$coordinates);let results=(outs AnyType:$value);
}
def Kir_Dialect_Term : Op<Kir_Dialect,"term",[Terminator]> {let arguments=(ins F32:$value);}
def Kir_Dialect_CountKeys : Op<Kir_Dialect,"count_keys"> {
  let arguments=(ins AnyMemRef:$source,Index:$owner,I64Attr:$span);let results=(outs I32Tensor:$counts);
}
def Kir_Dialect_ColumnSum : Op<Kir_Dialect,"column_sum"> {
  let arguments=(ins AnyMemRef:$source,Index:$owner);let results=(outs I32Tensor:$counts);
}
def Kir_Dialect_Different : Op<Kir_Dialect,"ne"> {let arguments=(ins AnyType:$lhs,AnyType:$rhs);let results=(outs AnyTensor:$mask);}
def Kir_Dialect_ColumnsI32 : Op<Kir_Dialect,"member_index"> {let arguments=(ins AnyMemRef:$domain,Index:$row);let results=(outs I32Tensor:$members);}
def Kir_Dialect_PrefixScatter : Op<Kir_Dialect,"prefix_scatter"> {let arguments=(ins AnyMemRef:$output,AnyMemRef:$counts,Index:$owner,I1Tensor:$predicate,I32Tensor:$prefix,I32Tensor:$values,I32Attr:$fill);}
def Kir_Dialect_Input : Op<Kir_Dialect,"index_read"> {let arguments=(ins AnyMemRef:$base,I32Tensor:$key,UnitAttr:$in_bounds);let results=(outs I32Tensor:$values);}
def Kir_Dialect_Ticket : Op<Kir_Dialect,"atomic_add"> {let arguments=(ins AnyMemRef:$base,I32Tensor:$key,I32:$delta,StrAttr:$ordering,UnitAttr:$in_bounds);let results=(outs I32Tensor:$values);}
def Kir_Dialect_Put : Op<Kir_Dialect,"unique_put"> {let arguments=(ins AnyMemRef:$base,I32Tensor:$key,I32Tensor:$value,UnitAttr:$in_bounds);}
def Kir_Dialect_WriteOffsets : Op<Kir_Dialect,"write_offsets"> {let arguments=(ins AnyMemRef:$offsets,AnyMemRef:$total,Index:$owner,I32Tensor:$prefix);}
def Kir_Dialect_Ranges : Op<Kir_Dialect,"fill_ranges"> {let arguments=(ins AnyMemRef:$out,AnyMemRef:$offsets,Index:$owner,I64Attr:$unit,UnitAttr:$disjoint_in_bounds);}
include "intentdsl/levels/algorithm/kir/ir/Matrix.td"
defm Matrix : MatrixExpressions<Kir_Dialect>;
include "intentdsl/levels/algorithm/kir/ir/Ordered.td"
defm Ordered : OrderedOwners<Kir_Dialect>;
