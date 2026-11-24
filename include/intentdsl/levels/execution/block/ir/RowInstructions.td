// ROWS/WIDTH are finite launch-row parameters. M/N and ABI strides are
// bound by the caller. Symbolic tile dimensions survive to Triton JIT.
def Meta : TileInstruction<"dimension", [Pure]> {let arguments=(ins StrAttr:$name);let results=(outs I64:$size);}
def RowSplat : TileInstruction<"splat", [Pure]> {let arguments=(ins AnyType:$scalar, StrArrayAttr:$dimensions);let results=(outs AnyRankedTensor:$tile);}
def RowExpand : TileInstruction<"expand", [Pure]> {let arguments=(ins AnyType:$input,I64Attr:$axis);let results=(outs AnyRankedTensor:$tile);}
def Stride : TileInstruction<"stride", [Pure]> {let arguments=(ins AnyMemRef:$view,I64Attr:$axis);let results=(outs I64:$pitch);}
def RowAddr : TileInstruction<"view_offset",[Pure]> {
  let arguments=(ins AnyMemRef:$view,Variadic<AnyType>:$coordinates, I32Attr:$index_bits);
  let results=(outs AnyType:$offset);
}
def MaskedLoad : TileInstruction<"read_at",[MemoryEffects<[MemRead]>]> {let arguments=(ins AnyMemRef:$view,AnyType:$offset,AnyType:$active,I64Attr:$fill);let results=(outs AnyType:$value);}
def MaskedStore : TileInstruction<"write_at",[MemoryEffects<[MemWrite]>]> {let arguments=(ins AnyMemRef:$view,AnyType:$offset,AnyType:$value,AnyType:$mask);}
def RowMath : TileInstruction<"map",[Pure]> {let arguments=(ins Variadic<AnyType>:$args,StrAttr:$kind);let results=(outs AnyType:$value);}
def RowLanes : TileInstruction<"range",[Pure]> {let arguments=(ins I64Attr:$size);let results=(outs AnyRankedTensor:$lanes);}
def BinCount : TileInstruction<"bin_count",[Pure]> {let arguments=(ins AnyRankedTensor:$keys);let results=(outs AnyRankedTensor:$counts);}
def FetchAdd : TileInstruction<"fetch_add",[MemoryEffects<[MemRead,MemWrite]>]> {
  let arguments=(ins AnyMemRef:$view, AnyType:$offset, AnyType:$delta, AnyType:$active, StrAttr:$scope);
  let results=(outs AnyRankedTensor:$old);
}
def RowReduce : TileInstruction<"axis_reduce",[Pure]> {let arguments=(ins AnyRankedTensor:$input,I64Attr:$axis,StrAttr:$algebra);let results=(outs AnyType:$reduced);}
def ExtractLane : TileInstruction<"extract_lane",[Pure]> {let arguments=(ins AnyRankedTensor:$panel,Index:$lane,I64Attr:$axis);let results=(outs AnyRankedTensor:$value);}
