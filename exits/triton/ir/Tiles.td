// Address coordinates are i64 scalars or i64 tile axes. Each axis of a
// memref is projected into the result tile (-1 denotes a scalar).
def TileAddr : Inst<"tile_address", [Pure]> {
  let arguments=(ins AnyMemRef:$view, Variadic<AnyType>:$coordinates,
                     DenseI64ArrayAttr:$projection, DenseI64ArrayAttr:$strides);
  let results=(outs AnyType:$offset);
}
def TileRead : Inst<"tile_read", [MemoryEffects<[MemRead]>]> {
  let arguments=(ins AnyMemRef:$view, AnyType:$offset, Optional<AnyType>:$mask);
  let results=(outs AnyType:$fragment);
}
def TileWrite : Inst<"tile_write", [MemoryEffects<[MemWrite]>]> {
  let arguments=(ins AnyMemRef:$view, AnyType:$offset, AnyType:$fragment, Optional<AnyType>:$mask);
}
def TileAxis : Inst<"tile_axis", [Pure]> {let results=(outs AnyRankedTensor:$lane);}
def TilePid : Inst<"tile_pid", [Pure]> {let arguments=(ins I64Attr:$axis);let results=(outs I64:$id);}
def TilePrefix : Inst<"tile_prefix", [Pure]> {let arguments=(ins AnyRankedTensor:$tile); let results=(outs AnyRankedTensor:$logical);}
def FromLEHalf : Inst<"from_le_half",[Pure]> {let arguments=(ins AnyRankedTensor:$lo,AnyRankedTensor:$hi);let results=(outs AnyRankedTensor:$number);}
def E8M0 : Inst<"e8m0",[Pure]> {let arguments=(ins AnyRankedTensor:$code);let results=(outs AnyRankedTensor:$factor);}
def TileBroadcast : Inst<"tile_broadcast", [Pure]> {
  let arguments=(ins AnyType:$value, I64Attr:$axis); let results=(outs AnyRankedTensor:$expanded);
}
def TileDot : Inst<"tile_dot", [Pure]> {
  let arguments=(ins AnyRankedTensor:$a, AnyRankedTensor:$b, Optional<AnyRankedTensor>:$acc,
                     BoolAttr:$transpose_rhs);
  let results=(outs AnyRankedTensor:$sum);
}
def TileReduce : Inst<"tile_reduce", [Pure]> {
  let arguments=(ins AnyRankedTensor:$fragment, StrAttr:$algebra, I64Attr:$axis); let results=(outs AnyType:$result);
}
def TileProgram : Inst<"tile_program", [SingleBlock]> {
  let arguments=(ins Variadic<AnyMemRef>:$domain, StrAttr:$calling_convention);
  let regions=(region AnyRegion:$program);
}
