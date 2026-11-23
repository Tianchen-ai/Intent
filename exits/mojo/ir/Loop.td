include "mlir/IR/OpBase.td"
def RowMojo_Dialect : Dialect {let name="mojo_row"; let cppNamespace="::intentdsl::rowmojo";}
class RowInst<string mnemonic,list<Trait> traits=[]> : Op<RowMojo_Dialect,mnemonic,traits>;
def ScanColumns : RowInst<"sweep",[SingleBlock]> {
 let arguments=(ins Variadic<AnyType>:$initial, StrAttr:$range, I64Attr:$width, StrAttr:$algebra);
 let results=(outs Variadic<AnyType>:$result);let regions=(region AnyRegion:$body);
}
def LaneRead : RowInst<"read"> {let arguments=(ins AnyMemRef:$source,Index:$row,Index:$column);let results=(outs AnyFixedVectorOfNonZeroRank:$result);}
def LaneWrite : RowInst<"write"> {let arguments=(ins AnyMemRef:$dest,Index:$row,Index:$column,AnyFixedVectorOfNonZeroRank:$value);}
def ValueExpr : RowInst<"apply"> {let arguments=(ins Variadic<AnyType>:$inputs,StrAttr:$kind);let results=(outs AnyType:$result);}
def StepYield : RowInst<"yield",[Terminator]> {let arguments=(ins Variadic<AnyType>:$values);}
def OwnerRead : RowInst<"owner_read"> {let arguments=(ins AnyMemRef:$source,Index:$row);let results=(outs AnyType:$value);}
def MojoSegmentFold : RowInst<"segment_fold",[SingleBlock]> {
  let arguments=(ins AnyMemRef:$offsets,Index:$row,Index:$column);
  let results=(outs AnyVectorOfAnyRank:$values);let regions=(region AnyRegion:$body);
}
def MojoSegmentSize : RowInst<"segment_size"> {
  let arguments=(ins AnyMemRef:$offsets,Index:$row,Index:$column);let results=(outs AnyVectorOfAnyRank:$values);
}
def OwnerWrite : RowInst<"owner_write"> {let arguments=(ins AnyMemRef:$destination,Index:$row,AnyType:$value);}
def MojoIndirectRead : Op<RowMojo_Dialect,"indexed_read"> {
  let arguments=(ins AnyMemRef:$table,AnyMemRef:$ids,Index:$row,Index:$col);
  let results=(outs AnyVectorOfAnyRank:$lanes);
}
def RowMojo_Dialect_KeyCounts : Op<RowMojo_Dialect,"key_counts"> {
  let arguments=(ins AnyMemRef:$source,Index:$row,Index:$column,I64Attr:$extent,I64Attr:$span);let results=(outs AnyVectorOfAnyRank:$counts);
}
def RowMojo_Dialect_SumColumns : Op<RowMojo_Dialect,"sum_columns"> {
  let arguments=(ins AnyMemRef:$source,Index:$row,Index:$column,I64Attr:$extent);let results=(outs AnyVectorOfAnyRank:$counts);
}

// Terminal matrix operations (v0.4)
def MojoMatrixProgram : RowInst<"matrix_program",[SingleBlock]> {
  let regions=(region AnyRegion:$panel, AnyRegion:$host);
}
def MojoPanelFn : RowInst<"panel_fn",[SingleBlock]> {
  let arguments=(ins ArrayAttr:$param_dtypes, BoolAttr:$uses_brgemm, BoolAttr:$uses_round_bf16, BoolAttr:$uses_vnni, BoolAttr:$uses_stack);
  let regions=(region AnyRegion:$body);
}
def MojoIfFullTile : RowInst<"if_full_tile",[SingleBlock]> {
  let arguments=(ins AnyType:$rows_val, AnyType:$cols_val, I64Attr:$rows, I64Attr:$cols, BoolAttr:$check_cols);
  let regions=(region AnyRegion:$body);
}
def MojoForRange : RowInst<"for_range",[SingleBlock]> {
  let arguments=(ins AnyType:$lb, AnyType:$ub, AnyType:$step);
  let regions=(region AnyRegion:$body);
}
def MojoWhileLt : RowInst<"while_lt",[SingleBlock]> {
  let arguments=(ins AnyType:$init, AnyType:$bound, AnyType:$step, BoolAttr:$check_step);
  let regions=(region AnyRegion:$body);
}
def MojoAsInt : RowInst<"as_int"> {
  let arguments=(ins AnyType:$input);
  let results=(outs Index:$result);
}
def MojoAddress : RowInst<"address"> {
  let arguments=(ins AnyType:$base, Variadic<AnyType>:$indices, DenseI64ArrayAttr:$strides, I32Attr:$index_bits);
  let results=(outs Index:$offset);
}
def MojoStackAlloc : RowInst<"stack_alloc"> {
  let arguments=(ins StrAttr:$dtype, I64Attr:$count);
  let results=(outs Index:$buffer);
}
def MojoAccDecl : RowInst<"acc_decl"> {
  let arguments=(ins Variadic<AnyType>:$init, StrAttr:$dtype, I64Attr:$width);
  let results=(outs AnyType:$slot);
}
def MojoAccRead : RowInst<"acc_read"> {
  let arguments=(ins AnyType:$slot);
  let results=(outs AnyType:$value);
}
def MojoAccWrite : RowInst<"acc_write"> {
  let arguments=(ins AnyType:$slot, AnyType:$value);
}
def MojoLoadVec : RowInst<"load_vec"> {
  let arguments=(ins AnyType:$ptr, Index:$offset, StrAttr:$dtype, I64Attr:$width, I64Attr:$lane_stride);
  let results=(outs AnyType:$value);
}
def MojoLoadI32Xor : RowInst<"load_i32_xor"> {
  let arguments=(ins AnyType:$ptr, Index:$offset);
  let results=(outs AnyType:$value);
}
def MojoLoadBitcastVec : RowInst<"load_bitcast_vec"> {
  let arguments=(ins AnyType:$ptr, Index:$offset, StrAttr:$scalar_type, I64Attr:$width);
  let results=(outs AnyType:$value);
}
def MojoSplatCast : RowInst<"splat_cast"> {
  let arguments=(ins AnyType:$input, StrAttr:$dtype, I64Attr:$width, BoolAttr:$from_scalar);
  let results=(outs AnyType:$value);
}
def MojoStoreVec : RowInst<"store_vec"> {
  let arguments=(ins AnyType:$ptr, Index:$offset, AnyType:$value, I64Attr:$width, I64Attr:$lane_stride);
}
def MojoSimdFma : RowInst<"simd_fma"> {
  let arguments=(ins AnyType:$acc, AnyType:$lhs, AnyType:$rhs, BoolAttr:$is_int);
  let results=(outs AnyType:$result);
}
def MojoVnniDot4 : RowInst<"vnni_dot4"> {
  let arguments=(ins AnyType:$acc, AnyType:$lhs, AnyType:$rhs);
  let results=(outs AnyType:$result);
}
def MojoQ4KSuperblock : RowInst<"q4k_superblock"> {
  let arguments=(ins AnyType:$w_ptr, AnyType:$q_ptr, Index:$w_base, Index:$group, I64Attr:$byte_stride);
  let results=(outs AnyType:$result);
}
def MojoPtrOffset : RowInst<"ptr_offset"> {
  let arguments=(ins AnyType:$ptr, Index:$offset);
  let results=(outs Index:$shifted);
}
def MojoBrgemmCall : RowInst<"brgemm_call"> {
  let arguments=(ins AnyType:$handle, AnyType:$a_ptr, AnyType:$b_ptr, AnyType:$c_ptr, AnyType:$scratch);
  let results=(outs I32:$status);
}
def MojoQ4KUnpackRow : RowInst<"q4k_unpack_row"> {
  let arguments=(ins AnyType:$w_ptr, Index:$w_base, AnyType:$u8_buf, Index:$row_idx, Index:$sub, I64Attr:$byte_stride);
}
def MojoQ4KAccumRow : RowInst<"q4k_accum_row"> {
  let arguments=(ins AnyType:$w_ptr, Index:$w_base, AnyType:$q_ptr, AnyType:$s32_buf, AnyType:$acc_buf, Index:$row_idx, Index:$group, Index:$sub, I64Attr:$byte_stride);
}
def MojoEpilogueAdd : RowInst<"epilogue_add"> {
  let arguments=(ins AnyType:$lhs, AnyType:$rhs);
  let results=(outs AnyType:$result);
}
def MojoEpilogueRoundBf16 : RowInst<"epilogue_round_bf16"> {
  let arguments=(ins AnyType:$input, I64Attr:$width);
  let results=(outs AnyType:$result);
}
def MojoEpilogueCast : RowInst<"epilogue_cast"> {
  let arguments=(ins AnyType:$input, StrAttr:$dtype);
  let results=(outs AnyType:$result);
}
def MojoEnd : RowInst<"end",[Terminator]> {}

// Host C region ops
def MojoHostFn : RowInst<"host_fn",[SingleBlock]> {
  let arguments=(ins StrAttr:$params, DenseI64ArrayAttr:$extents, DenseI64ArrayAttr:$tile,
                     BoolAttr:$uses_brgemm, BoolAttr:$uses_q8k, BoolAttr:$uses_bf16_math,
                     BoolAttr:$uses_f16_math, BoolAttr:$uses_mx8_math, I64Attr:$panel_arity);
  let regions=(region AnyRegion:$body);
}
def MojoHostAlloc : RowInst<"host_alloc"> {
  let arguments=(ins StrAttr:$buf_name, StrAttr:$placement, StrAttr:$c_type,
                     StrAttr:$count_expr, I64Attr:$elem_bytes, ArrayAttr:$cleanup);
}
def MojoHostPrep : RowInst<"host_prep"> {
  let arguments=(ins StrAttr:$kind, ArrayAttr:$cleanup);
  let regions=(region VariadicRegion<AnyRegion>:$map);
}
def MojoHostColLoop : RowInst<"host_col_loop",[SingleBlock]> {
  let regions=(region AnyRegion:$body);
}
def MojoHostPanelYield : RowInst<"host_panel_yield",[Terminator]> {
  let arguments=(ins AnyType:$value);
}
def MojoHostBrgemmPlan : RowInst<"host_brgemm_plan"> {
  let arguments=(ins I64Attr:$a_dt, I64Attr:$b_dt, I64Attr:$c_dt,
                     StrAttr:$lda_expr, StrAttr:$ldb_expr, BoolAttr:$is_q4k,
                     BoolAttr:$grouped, I64Attr:$offsets_slot, I64Attr:$offsets_stride,
                     ArrayAttr:$cleanup);
}
def MojoHostParallelTiles : RowInst<"host_parallel_tiles",[SingleBlock]> {
  let arguments=(ins BoolAttr:$grouped, BoolAttr:$routed, BoolAttr:$windowed,
                     I64Attr:$head_extent, I64Attr:$chunk_extent, I64Attr:$multiplicity,
                     I64Attr:$offsets_slot, I64Attr:$offsets_stride,
                     I64Attr:$routes_slot, I64Attr:$routes_stride,
                     BoolAttr:$uses_brgemm, BoolAttr:$is_q4k);
  let regions=(region AnyRegion:$body);
}
def MojoHostViewPtr : RowInst<"host_view_ptr"> {
  let arguments=(ins I64Attr:$index, StrAttr:$source_kind, StrAttr:$expr);
}
def MojoHostCallPanel : RowInst<"host_call_panel"> {
  let arguments=(ins I64Attr:$arity, BoolAttr:$uses_brgemm);
}
def MojoHostScatterRouted : RowInst<"host_scatter_routed"> {
  let arguments=(ins I64Attr:$dst_slot, I64Attr:$multiplicity, DenseI64ArrayAttr:$dst_strides,
                     I64Attr:$elem_bytes, I64Attr:$routes_slot, I64Attr:$routes_stride);
}
def MojoHostFree : RowInst<"host_free"> {
  let arguments=(ins StrAttr:$buf_name);
}
