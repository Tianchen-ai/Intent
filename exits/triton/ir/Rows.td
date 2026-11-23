// ROWS/WIDTH are finite launch-row parameters. M/N and ABI strides are
// bound by the caller. Symbolic tile dimensions survive to Triton JIT.
def Meta : Inst<"dimension", [Pure]> {let arguments=(ins StrAttr:$name);let results=(outs I64:$size);}
def RowSplat : Inst<"splat", [Pure]> {let arguments=(ins AnyType:$scalar, StrArrayAttr:$dimensions);let results=(outs AnyRankedTensor:$tile);}
def RowExpand : Inst<"expand", [Pure]> {let arguments=(ins AnyType:$input,I64Attr:$axis);let results=(outs AnyRankedTensor:$tile);}
def Stride : Inst<"stride", [Pure]> {let arguments=(ins AnyMemRef:$view,I64Attr:$axis);let results=(outs I64:$pitch);}
def RowAddr : Inst<"view_offset",[Pure]> {
  let arguments=(ins AnyMemRef:$view,Variadic<AnyType>:$coordinates, I32Attr:$index_bits);
  let results=(outs AnyType:$offset);
}
def MaskedLoad : Inst<"read_at",[MemoryEffects<[MemRead]>]> {let arguments=(ins AnyMemRef:$view,AnyType:$offset,AnyType:$active,I64Attr:$fill);let results=(outs AnyType:$value);}
def MaskedStore : Inst<"write_at",[MemoryEffects<[MemWrite]>]> {let arguments=(ins AnyMemRef:$view,AnyType:$offset,AnyType:$value,AnyType:$mask);}
def RowMath : Inst<"map",[Pure]> {let arguments=(ins Variadic<AnyType>:$args,StrAttr:$kind);let results=(outs AnyType:$value);}
def RowLanes : Inst<"range",[Pure]> {let arguments=(ins I64Attr:$size);let results=(outs AnyRankedTensor:$lanes);}
def BinCount : Inst<"bin_count",[Pure]> {let arguments=(ins AnyRankedTensor:$keys);let results=(outs AnyRankedTensor:$counts);}
def RowReduce : Inst<"axis_reduce",[Pure]> {let arguments=(ins AnyRankedTensor:$input,I64Attr:$axis,StrAttr:$algebra);let results=(outs AnyType:$reduced);}
def ExtractLane : Inst<"extract_lane",[Pure]> {let arguments=(ins AnyRankedTensor:$panel,Index:$lane);let results=(outs AnyRankedTensor:$value);}
