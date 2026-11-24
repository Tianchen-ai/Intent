include "mlir/IR/OpBase.td"
def RowMojo_Dialect : Dialect {let name="mojo_row"; let cppNamespace="::intentdsl::rowmojo";}
class RowInst<string mnemonic,list<Trait> traits=[]> : Op<RowMojo_Dialect,mnemonic,traits>;
def ScanColumns : RowInst<"sweep",[SingleBlock]> {
 let arguments=(ins Variadic<AnyType>:$initial, StrAttr:$range, I64Attr:$width, StrAttr:$algebra);
 let results=(outs Variadic<AnyType>:$result);let regions=(region AnyRegion:$body);
}
def LaneRead : RowInst<"read"> {let arguments=(ins AnyMemRef:$source,Index:$row,Index:$column);let results=(outs AnyFixedVectorOfNonZeroRank:$result);}
def LaneWrite : RowInst<"write"> {let arguments=(ins AnyMemRef:$dest,Index:$row,Index:$column,AnyFixedVectorOfNonZeroRank:$value);}
def RowCacheStore : RowInst<"row_cache_store"> {let arguments=(ins Index:$column,AnyType:$value,I64Attr:$slot);}
def RowCacheLoad : RowInst<"row_cache_load"> {let arguments=(ins Index:$column,I64Attr:$slot);let results=(outs AnyType:$value);}
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
  let arguments=(ins StrAttr:$dtype, I64Attr:$count, OptionalAttr<StrAttr>:$role);
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
def MojoSplatLane : RowInst<"splat_lane"> {
  let arguments=(ins AnyType:$input, StrAttr:$dtype, I64Attr:$width, I64Attr:$lane);
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
  let arguments=(ins AnyType:$w_ptr, AnyType:$q_ptr, Index:$w_base, Index:$group, I64Attr:$byte_stride, BoolAttr:$use_vnni);
  let results=(outs AnyType:$result);
}
def MojoPtrOffset : RowInst<"ptr_offset"> {
  let arguments=(ins AnyType:$ptr, Index:$offset);
  let results=(outs Index:$shifted);
}
def MojoBrgemmCall : RowInst<"brgemm_call"> {
  let arguments=(ins AnyType:$handle, AnyType:$a_ptr, AnyType:$b_ptr, AnyType:$c_ptr, AnyType:$scratch, BoolAttr:$allow_subnormal_ftz);
  let results=(outs I32:$status);
}
def MojoCheckStatus : RowInst<"check_status"> {
  let arguments=(ins I32:$status);
}
def MojoPanelReturn : RowInst<"panel_return"> {
  let arguments=(ins I32Attr:$code);
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

// Host C region ops (explicit loops, OpenMP, scratch, and atomic statements)
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
def MojoHostIf : RowInst<"host_if",[SingleBlock]> {
  let arguments=(ins StrAttr:$cond);
  let regions=(region AnyRegion:$body);
}
def MojoHostOmpParallel : RowInst<"host_omp_parallel",[SingleBlock]> {
  let arguments=(ins StrAttr:$num_threads, StrAttr:$if_cond, BoolAttr:$reduce_rc);
  let regions=(region AnyRegion:$body);
}
def MojoHostOmpFor : RowInst<"host_omp_for",[SingleBlock]> {
  let arguments=(ins StrAttr:$iv, StrAttr:$lb, StrAttr:$ub, StrAttr:$step,
                     BoolAttr:$parallel, BoolAttr:$nowait, BoolAttr:$simd, StrAttr:$if_cond);
  let regions=(region AnyRegion:$body);
}
def MojoHostFor : RowInst<"host_for",[SingleBlock]> {
  let arguments=(ins StrAttr:$iv, StrAttr:$lb, StrAttr:$ub, StrAttr:$step);
  let regions=(region AnyRegion:$body);
}
def MojoHostGuardContinue : RowInst<"host_guard_continue"> {
  let arguments=(ins StrAttr:$cond);
}
def MojoHostLetI64 : RowInst<"host_let_i64"> {
  let arguments=(ins StrAttr:$name, StrAttr:$expr);
}
def MojoHostLoadScalar : RowInst<"host_load_scalar"> {
  let arguments=(ins I64Attr:$slot, StrAttr:$c_type, StrAttr:$index_expr);
  let results=(outs AnyType:$val);
}
def MojoHostPtrOffset : RowInst<"host_ptr_offset"> {
  let arguments=(ins StrAttr:$c_type, StrAttr:$name, StrAttr:$base_expr, StrAttr:$offset_expr);
}
def MojoHostSpanOp : RowInst<"host_span_op"> {
  let arguments=(ins StrAttr:$op, StrAttr:$dst_expr, StrAttr:$src_expr, StrAttr:$count_expr, I64Attr:$elem_bytes);
}
def MojoHostElemOp : RowInst<"host_elem_op"> {
  let arguments=(ins StrAttr:$op, StrAttr:$dst_expr, StrAttr:$src_expr, StrAttr:$aux_expr, I64Attr:$elem_bytes);
}
def MojoHostPanelYield : RowInst<"host_panel_yield"> {
  let arguments=(ins AnyType:$value, BoolAttr:$widen_bf16, StrAttr:$dst_expr);
}
def MojoHostBrgemmPlanDecl : RowInst<"host_brgemm_plan_decl"> {}
def MojoHostBrgemmMarkRows : RowInst<"host_brgemm_mark_rows"> {
  let arguments=(ins StrAttr:$len_expr);
}
def MojoHostBrgemmLookup : RowInst<"host_brgemm_lookup"> {
  let arguments=(ins I64Attr:$a_dt, I64Attr:$b_dt, I64Attr:$c_dt,
                     StrAttr:$lda_expr, StrAttr:$ldb_expr, StrAttr:$n_expr, StrAttr:$k_expr,
                     ArrayAttr:$cleanup);
}
def MojoHostThreadScratchAlloc : RowInst<"host_thread_scratch_alloc"> {
  let arguments=(ins I64Attr:$stack_bytes, I64Attr:$alignment, StrAttr:$size_var);
}
def MojoHostThreadScratchFree : RowInst<"host_thread_scratch_free"> {
  let arguments=(ins I64Attr:$stack_bytes);
}
def MojoHostBrgemmCtxDecl : RowInst<"host_brgemm_ctx_decl"> {}
def MojoHostBrgemmCtxSwitch : RowInst<"host_brgemm_ctx_switch"> {}
def MojoHostBrgemmCtxRelease : RowInst<"host_brgemm_ctx_release"> {}
def MojoHostTileCoords : RowInst<"host_tile_coords"> {
  let arguments=(ins BoolAttr:$windowed, I64Attr:$head_extent, I64Attr:$chunk_extent);
}
def MojoHostGroupedBounds : RowInst<"host_grouped_bounds"> {
  let arguments=(ins I64Attr:$offsets_slot, I64Attr:$offsets_stride);
}
def MojoHostRouteCoords : RowInst<"host_route_coords"> {
  let arguments=(ins BoolAttr:$routed, I64Attr:$routes_slot, I64Attr:$routes_stride, I64Attr:$multiplicity);
}
def MojoHostViewPtr : RowInst<"host_view_ptr"> {
  let arguments=(ins I64Attr:$index, StrAttr:$source_kind, StrAttr:$expr);
}
def MojoHostCallPanel : RowInst<"host_call_panel"> {
  let arguments=(ins I64Attr:$arity, BoolAttr:$uses_brgemm);
}
def MojoHostFree : RowInst<"host_free"> {
  let arguments=(ins StrAttr:$buf_name);
}

// Terminal ordered operations (v0.4)
def MojoOrderedProgram : RowInst<"ordered_program",[SingleBlock]> {
  let regions=(region AnyRegion:$kernel, AnyRegion:$host);
}
def MojoOwnerFn : RowInst<"owner_fn",[SingleBlock]> {
  let arguments=(ins ArrayAttr:$param_types, BoolAttr:$uses_stack);
  let regions=(region AnyRegion:$body);
}
def MojoOwnerCoord : RowInst<"owner_coord"> {
  let arguments=(ins AnyType:$owner, I64Attr:$pitch, I64Attr:$extent);
  let results=(outs Index:$coord);
}
def MojoScalarDecl : RowInst<"scalar_decl"> {
  let arguments=(ins Variadic<AnyType>:$init, StrAttr:$scalar_type);
  let results=(outs AnyType:$slot);
}
def MojoLoadScalar : RowInst<"load_scalar"> {
  let arguments=(ins AnyType:$ptr, Index:$offset);
  let results=(outs AnyType:$value);
}
def MojoStoreScalar : RowInst<"store_scalar"> {
  let arguments=(ins AnyType:$ptr, Index:$offset, AnyType:$value);
}
def MojoLoadSlice : RowInst<"load_slice"> {
  let arguments=(ins AnyType:$ptr, Index:$offset, I64Attr:$width);
  let results=(outs AnyType:$value);
}
def MojoStoreSlice : RowInst<"store_slice"> {
  let arguments=(ins AnyType:$ptr, Index:$offset, AnyType:$value);
}
def MojoAccInsertLane : RowInst<"acc_insert_lane"> {
  let arguments=(ins AnyType:$slot, AnyType:$lane, AnyType:$scalar);
}
def MojoExtractLaneDyn : RowInst<"extract_lane_dyn"> {
  let arguments=(ins AnyType:$vec, AnyType:$lane);
  let results=(outs AnyType:$scalar);
}
def MojoSplatScalar : RowInst<"splat_scalar"> {
  let arguments=(ins AnyType:$scalar, StrAttr:$dtype, I64Attr:$width);
  let results=(outs AnyType:$value);
}
def MojoLaneSum : RowInst<"lane_sum"> {
  let arguments=(ins AnyType:$input);
  let results=(outs F32:$result);
}
def MojoLaneMax : RowInst<"lane_max"> {
  let arguments=(ins AnyType:$input, I64Attr:$width);
  let results=(outs F32:$result);
}
def MojoReduceSum16x16 : RowInst<"reduce_sum_16x16"> {
  let arguments=(ins Variadic<AnyType>:$inputs);
  let results=(outs AnyType:$result);
}
def MojoSimdCmpI : RowInst<"simd_cmpi"> {
  let arguments=(ins AnyType:$lhs, AnyType:$rhs, StrAttr:$pred);
  let results=(outs AnyType:$result);
}
def MojoPackApply : RowInst<"pack_apply"> {
  let arguments=(ins Variadic<AnyType>:$scalars, StrAttr:$in_dtype, StrAttr:$out_dtype, StrAttr:$kind);
  let results=(outs AnyType:$packed);
}
def MojoExtractLane : RowInst<"extract_lane"> {
  let arguments=(ins AnyType:$packed, I64Attr:$lane);
  let results=(outs AnyType:$scalar);
}
def MojoCHostFn : RowInst<"c_host_fn",[SingleBlock]> {
  let arguments=(ins StrAttr:$params, StrAttr:$extern_decl);
  let regions=(region AnyRegion:$body);
}
def MojoCOmpFor : RowInst<"c_omp_for",[SingleBlock]> {
  let arguments=(ins I64Attr:$count, I64Attr:$grain, BoolAttr:$cyclic);
  let regions=(region AnyRegion:$body);
}
def MojoCOwnerCall : RowInst<"c_owner_call"> {
  let arguments=(ins StrAttr:$call_args, BoolAttr:$cyclic);
}

// Terminal row operations (v0.4)
def MojoRowProgram : RowInst<"row_program",[SingleBlock]> {
  let regions=(region AnyRegion:$kernel, AnyRegion:$host);
}
def MojoRuntimeSource : RowInst<"runtime_source"> {
  let arguments=(ins StrAttr:$implementation);
}
def MojoRowFn : RowInst<"row_fn",[SingleBlock]> {
  let arguments=(ins ArrayAttr:$param_specs, BoolAttr:$uses_atomic, BoolAttr:$uses_stack,
                     BoolAttr:$uses_memcpy, BoolAttr:$has_key_bounds);
  let regions=(region AnyRegion:$body);
}
def MojoDynAddress : RowInst<"dyn_address"> {
  let arguments=(ins AnyType:$base, Variadic<AnyType>:$coords_and_strides, I64Attr:$terms, I32Attr:$index_bits);
  let results=(outs Index:$offset);
}
def MojoAtomicFetchAddI32 : RowInst<"atomic_fetch_add_i32"> {
  let arguments=(ins AnyType:$ptr, Index:$offset, I32:$delta);
  let results=(outs I32:$old_value);
}
def MojoMemcpyElems : RowInst<"memcpy_elems"> {
  let arguments=(ins AnyType:$dst, Index:$dst_off, AnyType:$src, Index:$src_off, Index:$count);
}
def MojoSimdIotaAdd : RowInst<"simd_iota_add"> {
  let arguments=(ins AnyType:$base_idx, StrAttr:$dtype, I64Attr:$width);
  let results=(outs AnyType:$result);
}
def MojoSimdSelect : RowInst<"simd_select"> {
  let arguments=(ins AnyType:$cond, AnyType:$on_true, AnyType:$on_false);
  let results=(outs AnyType:$result);
}
def MojoSimdCmpGt : RowInst<"simd_cmp_gt"> {
  let arguments=(ins AnyType:$lhs, AnyType:$rhs);
  let results=(outs AnyType:$result);
}
def MojoScalarCast : RowInst<"scalar_cast"> {
  let arguments=(ins AnyType:$input, StrAttr:$dtype);
  let results=(outs AnyType:$result);
}
def MojoScalarConst : RowInst<"scalar_const"> {
  let arguments=(ins StrAttr:$dtype, StrAttr:$literal);
  let results=(outs AnyType:$result);
}
def MojoTupleDecl : RowInst<"tuple_decl"> {
  let arguments=(ins Variadic<AnyType>:$init);
  let results=(outs AnyType:$slot);
}
def MojoTupleRead : RowInst<"tuple_read"> {
  let arguments=(ins AnyType:$slot);
  let results=(outs AnyType:$value);
}
def MojoTupleWrite : RowInst<"tuple_write"> {
  let arguments=(ins AnyType:$slot, AnyType:$value);
}
def MojoTupleField : RowInst<"tuple_field"> {
  let arguments=(ins AnyType:$tuple_val, I64Attr:$position);
  let results=(outs AnyType:$field);
}
def MojoUniform : RowInst<"uniform"> {
  let arguments=(ins UI64:$seed, AnyType:$counter, I64Attr:$width);
  let results=(outs AnyType:$samples);
}
def MojoWelfordVecStep : RowInst<"welford_vec_step"> {
  let arguments=(ins AnyType:$x, AnyType:$mean, AnyType:$m2, I32:$step_n, I64Attr:$width);
  let results=(outs AnyType:$next_mean, AnyType:$next_m2);
}
def MojoWelfordVecReduce : RowInst<"welford_vec_reduce"> {
  let arguments=(ins I32:$step_n, AnyType:$mean, AnyType:$m2, I64Attr:$width);
  let results=(outs AnyType:$summary);
}
def MojoWelfordScalarStep : RowInst<"welford_scalar_step"> {
  let arguments=(ins AnyType:$state, F32:$sample);
  let results=(outs AnyType:$summary);
}
def MojoTranspose8x8 : RowInst<"transpose_8x8"> {
  let arguments=(ins AnyType:$src, Index:$src_off, Index:$src_s0,
                     AnyType:$dst, Index:$dst_off, Index:$dst_s0, StrAttr:$dtype);
}
def MojoSimdScanSum : RowInst<"simd_scan_sum"> {
  let arguments=(ins AnyType:$input, AnyType:$carry, StrAttr:$dtype, I64Attr:$width);
  let results=(outs AnyType:$prefix, AnyType:$next_carry);
}
def MojoCRowHostFn : RowInst<"c_row_host_fn",[SingleBlock]> {
  let arguments=(ins StrAttr:$params, StrAttr:$extern_decl);
  let regions=(region AnyRegion:$body);
}
def MojoCRowPrep : RowInst<"c_row_prep"> {
  let arguments=(ins BoolAttr:$single_dispatcher, BoolAttr:$member_tasks, BoolAttr:$swap_dest);
}
def MojoCKeyOmpFor : RowInst<"c_key_omp_for",[SingleBlock]> {
  let arguments=(ins I64Attr:$key_extent);
  let regions=(region AnyRegion:$body);
}
def MojoCTile2DOmp : RowInst<"c_tile_2d_omp",[SingleBlock]> {
  let arguments=(ins I64Attr:$tile);
  let regions=(region AnyRegion:$body);
}
def MojoCRowSingleThread : RowInst<"c_row_single_thread",[SingleBlock]> {
  let regions=(region AnyRegion:$body);
}
def MojoCRowOmpParallel : RowInst<"c_row_omp_parallel",[SingleBlock]> {
  let arguments=(ins OptionalAttr<BoolAttr>:$dynamic_chunks);
  let regions=(region AnyRegion:$body);
}
def MojoCFlatColOmp : RowInst<"c_flat_col_omp",[SingleBlock]> {
  let arguments=(ins I64Attr:$col_grain);
  let regions=(region AnyRegion:$body);
}
def MojoCProjectedWhile : RowInst<"c_projected_while",[SingleBlock]> {
  let arguments=(ins I64Attr:$proj_rank);
  let regions=(region AnyRegion:$body);
}
def MojoCRowCall : RowInst<"c_row_call"> {
  let arguments=(ins StrAttr:$mode, StrAttr:$call_args, I64Attr:$elem_bytes);
}
