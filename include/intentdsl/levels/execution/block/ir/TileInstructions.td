// Address coordinates are i64 scalars or i64 tile axes. Each axis of a
// memref is projected into the result tile (-1 denotes a scalar).
def TileAddr : TileInstruction<"tile_address", [Pure]> {
  let arguments=(ins AnyMemRef:$view, Variadic<AnyType>:$coordinates,
                     DenseI64ArrayAttr:$projection, DenseI64ArrayAttr:$strides, I32Attr:$index_bits);
  let results=(outs AnyType:$offset);
}
def TileRead : TileInstruction<"tile_read", [MemoryEffects<[MemRead]>]> {
  let arguments=(ins AnyMemRef:$view, AnyType:$offset, Optional<AnyType>:$mask);
  let results=(outs AnyType:$fragment);
}
def TileWrite : TileInstruction<"tile_write", [MemoryEffects<[MemWrite]>]> {
  let arguments=(ins AnyMemRef:$view, AnyType:$offset, AnyType:$fragment, Optional<AnyType>:$mask);
}
def TileAtomicAdd : TileInstruction<"tile_atomic_add"> {
  let arguments=(ins Arg<AnyMemRef, "", [MemRead, MemWrite]>:$view, AnyType:$offset, AnyType:$fragment, Optional<AnyType>:$mask);
}
def TileAtomicFetchAdd : TileInstruction<"tile_atomic_fetch_add"> {
  let arguments=(ins Arg<AnyMemRef, "", [MemRead, MemWrite]>:$view, AnyType:$offset, AnyType:$fragment, Optional<AnyType>:$mask, StrAttr:$order, StrAttr:$scope);
  let results=(outs AnyType:$old);
}
def TileAtomicXchg : TileInstruction<"tile_atomic_xchg"> {
  let arguments=(ins Arg<AnyMemRef, "", [MemRead, MemWrite]>:$view, AnyType:$offset, AnyType:$fragment, Optional<AnyType>:$mask, StrAttr:$order, StrAttr:$scope);
}
def TileAxis : TileInstruction<"tile_axis", [Pure]> {let results=(outs AnyRankedTensor:$lane);}
def DropOwner : TileInstruction<"drop_unit_owner", [Pure]> {let arguments=(ins AnyRankedTensor:$tile);let results=(outs AnyRankedTensor:$members);}
def TilePid : TileInstruction<"tile_pid", [Pure]> {let arguments=(ins I64Attr:$axis);let results=(outs I64:$id);}
def TilePrefix : TileInstruction<"tile_prefix", [Pure]> {let arguments=(ins AnyRankedTensor:$tile); let results=(outs AnyRankedTensor:$logical);}
def FromLEHalf : TileInstruction<"from_le_half",[Pure]> {let arguments=(ins AnyRankedTensor:$lo,AnyRankedTensor:$hi);let results=(outs AnyRankedTensor:$number);}
def E8M0 : TileInstruction<"e8m0",[Pure]> {let arguments=(ins AnyRankedTensor:$code);let results=(outs AnyRankedTensor:$factor);}
def TileBroadcast : TileInstruction<"tile_broadcast", [Pure]> {
  let arguments=(ins AnyType:$value, I64Attr:$axis); let results=(outs AnyRankedTensor:$expanded);
}
def TileDot : TileInstruction<"tile_dot", [Pure]> {
  let arguments=(ins AnyRankedTensor:$a, AnyRankedTensor:$b, Optional<AnyRankedTensor>:$acc,
                     BoolAttr:$transpose_rhs);
  let results=(outs AnyRankedTensor:$sum);
}
def TileReduce : TileInstruction<"tile_reduce", [Pure]> {
  let arguments=(ins AnyRankedTensor:$fragment, StrAttr:$algebra, I64Attr:$axis); let results=(outs AnyType:$result);
}
def TileProgram : TileInstruction<"tile_program", [SingleBlock]> {
  let arguments=(ins Variadic<AnyMemRef>:$domain, StrAttr:$calling_convention);
  let regions=(region AnyRegion:$program);
}
