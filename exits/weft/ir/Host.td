include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def WeftHost_Dialect : Dialect {
  let name = "weft_host";
  let cppNamespace = "::intentdsl::wefthost";
}

// ---------------- v0.4 Terminal Program Containers ----------------

def MatrixProgramOp : Op<WeftHost_Dialect, "matrix_program"> {
  let arguments = (ins DenseI64ArrayAttr:$tile, DenseI64ArrayAttr:$extents,
                       StrAttr:$implementation, BoolAttr:$uses_brgemm);
  let regions = (region AnyRegion:$task, AnyRegion:$host);
}

def RowProgramOp : Op<WeftHost_Dialect, "row_program"> {
  let arguments = (ins ArrayAttr:$rows, ArrayAttr:$shape_parameters,
                       DenseI64ArrayAttr:$contiguous_slots,
                       DenseI64ArrayAttr:$task_ranks);
  let regions = (region AnyRegion:$task, AnyRegion:$host);
}

def OwnerProgramOp : Op<WeftHost_Dialect, "owner_program"> {
  let arguments = (ins I64Attr:$owners, I64Attr:$grain,
                       ArrayAttr:$shape_parameters,
                       DenseI64ArrayAttr:$contiguous_slots,
                       DenseI64ArrayAttr:$task_ranks);
  let regions = (region AnyRegion:$task, AnyRegion:$host);
}

def EndOp : Op<WeftHost_Dialect, "end", [Pure, ReturnLike, Terminator]> {}

// ---------------- Region 0 ("task.weftir") Terminal Ops ----------------

def EncodingDeclOp : Op<WeftHost_Dialect, "encoding_decl"> {
  let arguments = (ins StrAttr:$sym_name, I64Attr:$alignment,
                       I64Attr:$elements, I64Attr:$storage_bits,
                       ArrayAttr:$field_names, ArrayAttr:$field_types,
                       DenseI64ArrayAttr:$field_extents,
                       ArrayAttr:$field_layouts,
                       DenseI64ArrayAttr:$field_bit_offsets,
                       DenseI64ArrayAttr:$field_storage_bits);
}

def TaskKernelOp : Op<WeftHost_Dialect, "task_kernel", [IsolatedFromAbove]> {
  let arguments = (ins StrAttr:$sym_name, StrAttr:$source,
                       ArrayAttr:$view_types, OptionalAttr<ArrayAttr>:$arg_writes,
                       OptionalAttr<ArrayAttr>:$arg_access,
                       OptionalAttr<DenseI64ArrayAttr>:$arg_aliases,
                       OptionalAttr<ArrayAttr>:$shape_symbols,
                       DenseI64ArrayAttr:$strided_slots);
  let regions = (region AnyRegion:$body);
}

def TaskConstOp : Op<WeftHost_Dialect, "task_const"> {
  let arguments = (ins StrAttr:$lit, StrAttr:$val_type);
  let results = (outs Index:$res);
}

def TaskNewOp : Op<WeftHost_Dialect, "task_new"> {
  let arguments = (ins Index:$scalar, StrAttr:$in_type, StrAttr:$val_type);
  let results = (outs Index:$res);
}

def TaskSliceOp : Op<WeftHost_Dialect, "task_slice"> {
  let arguments = (ins Variadic<Index>:$operands, StrAttr:$view_type,
                       ArrayAttr:$selectors, StrAttr:$slice_type,
                       StrAttr:$val_type, BoolAttr:$admit);
  let results = (outs Index:$res);
}

def TaskSubviewOp : Op<WeftHost_Dialect, "task_subview"> {
  let arguments = (ins Variadic<Index>:$operands, I64Attr:$static_extent,
                       OptionalAttr<DenseI64ArrayAttr>:$static_extents,
                       OptionalAttr<I64Attr>:$container_extent,
                       StrAttr:$slice_type, StrAttr:$val_type, BoolAttr:$admit);
  let results = (outs Index:$res);
}

def TaskUnaryOp : Op<WeftHost_Dialect, "task_unary"> {
  let arguments = (ins Index:$input, StrAttr:$kind, StrAttr:$val_type);
  let results = (outs Index:$res);
}

def TaskBinaryOp : Op<WeftHost_Dialect, "task_binary"> {
  let arguments = (ins Index:$lhs, Index:$rhs, StrAttr:$kind,
                       StrAttr:$lhs_type, StrAttr:$rhs_type, StrAttr:$val_type);
  let results = (outs Index:$res);
}

def TaskCompareOp : Op<WeftHost_Dialect, "task_compare"> {
  let arguments = (ins Index:$lhs, Index:$rhs, StrAttr:$predicate,
                       StrAttr:$lhs_type, StrAttr:$rhs_type, StrAttr:$val_type);
  let results = (outs Index:$res);
}

def TaskSelectOp : Op<WeftHost_Dialect, "task_select"> {
  let arguments = (ins Index:$cond, Index:$true_val, Index:$false_val,
                       StrAttr:$cond_type, OptionalAttr<StrAttr>:$true_type,
                       OptionalAttr<StrAttr>:$false_type, StrAttr:$val_type);
  let results = (outs Index:$res);
}

def TaskCastOp : Op<WeftHost_Dialect, "task_cast"> {
  let arguments = (ins Index:$input, StrAttr:$op_kind,
                       StrAttr:$in_type, StrAttr:$val_type);
  let results = (outs Index:$res);
}

def TaskReduceOp : Op<WeftHost_Dialect, "task_reduce"> {
  let arguments = (ins Index:$input, StrAttr:$kind, I64Attr:$axis,
                       StrAttr:$in_type, StrAttr:$val_type);
  let results = (outs Index:$res);
}

def TaskScanOp : Op<WeftHost_Dialect, "task_scan"> {
  let arguments = (ins Index:$input, I64Attr:$axis, BoolAttr:$reverse,
                       StrAttr:$elem_type, StrAttr:$val_type);
  let results = (outs Index:$res);
  let regions = (region AnyRegion:$combine);
}

def TaskScanYieldOp : Op<WeftHost_Dialect, "task_scan_yield", [Pure, ReturnLike, Terminator]> {
  let arguments = (ins Index:$value, StrAttr:$val_type);
}

def TaskIotaOp : Op<WeftHost_Dialect, "task_iota"> {
  let arguments = (ins I64Attr:$start, I64Attr:$end, StrAttr:$val_type);
  let results = (outs Index:$res);
}

def TaskLocalAllocOp : Op<WeftHost_Dialect, "task_local_alloc"> {
  let arguments = (ins I64Attr:$elements, I64Attr:$alignment,
                       StrAttr:$ptr_type);
  let results = (outs Index:$res);
}

def TaskPtrLoadOp : Op<WeftHost_Dialect, "task_ptr_load"> {
  let arguments = (ins Index:$ptr, Index:$offset, Index:$mask, Index:$other,
                       StrAttr:$ptr_type, StrAttr:$val_type);
  let results = (outs Index:$res);
}

def TaskPtrStoreOp : Op<WeftHost_Dialect, "task_ptr_store"> {
  let arguments = (ins Index:$ptr, Index:$offset, Index:$mask, Index:$value,
                       StrAttr:$ptr_type, StrAttr:$val_type);
}

def TaskAtomicRmwOp : Op<WeftHost_Dialect, "task_atomic_rmw"> {
  let arguments = (ins Index:$value, Index:$slice, StrAttr:$kind,
                       StrAttr:$ordering, StrAttr:$val_type,
                       StrAttr:$slice_type);
  let results = (outs Index:$res);
}

def TaskFieldOp : Op<WeftHost_Dialect, "task_field"> {
  let arguments = (ins Index:$record, StrAttr:$field_name,
                       StrAttr:$in_type, StrAttr:$val_type);
  let results = (outs Index:$res);
}

def TaskFold2Op : Op<WeftHost_Dialect, "task_fold2"> {
  let arguments = (ins Index:$input, StrAttr:$in_type, StrAttr:$val_type);
  let results = (outs Index:$res);
}

def TaskContractOp : Op<WeftHost_Dialect, "task_contract", [Pure]> {
  let arguments = (ins Variadic<Index>:$inputs, StrAttr:$kind,
                       DenseI64ArrayAttr:$over, StrAttr:$acc_type,
                       StrAttr:$engine, BoolAttr:$allow_subnormal_ftz,
                       ArrayAttr:$in_types, StrAttr:$val_type);
  let results = (outs Index:$res);
}

def TaskForOp : Op<WeftHost_Dialect, "task_for"> {
  let arguments = (ins Index:$lb, Index:$ub, Index:$step,
                       Variadic<Index>:$inits, ArrayAttr:$iter_types);
  let results = (outs Variadic<Index>:$results);
  let regions = (region AnyRegion:$body);
}

def TaskIfOp : Op<WeftHost_Dialect, "task_if"> {
  let arguments = (ins Index:$cond, ArrayAttr:$res_types);
  let results = (outs Variadic<Index>:$results);
  let regions = (region AnyRegion:$thenRegion, AnyRegion:$elseRegion);
}

def TaskYieldOp : Op<WeftHost_Dialect, "task_yield", [Pure, ReturnLike, Terminator]> {
  let arguments = (ins Variadic<Index>:$values, ArrayAttr:$val_types);
}

def TaskSubquantLevelOp : Op<WeftHost_Dialect, "task_subquant_level"> {
  let arguments = (ins Index:$root_dom, Variadic<Index>:$inits);
  let results = (outs Variadic<Index>:$results);
  let regions = (region AnyRegion:$body);
}

def TaskExtractOp : Op<WeftHost_Dialect, "task_extract"> {
  let arguments = (ins Index:$input, Index:$point, StrAttr:$selector,
                       StrAttr:$in_type, StrAttr:$point_type, StrAttr:$val_type);
  let results = (outs Index:$res);
}

def TaskLookupOp : Op<WeftHost_Dialect, "task_lookup"> {
  let arguments = (ins Index:$table, Index:$indices, StrAttr:$bounds,
                       StrAttr:$table_type, StrAttr:$idx_type, StrAttr:$val_type);
  let results = (outs Index:$res);
}

def TaskHandoffOp : Op<WeftHost_Dialect, "task_handoff", [Pure, ReturnLike, Terminator]> {
  let arguments = (ins Variadic<Index>:$vals);
}

def TaskCommitOp : Op<WeftHost_Dialect, "task_commit"> {
  let arguments = (ins Index:$value, Index:$slice,
                       StrAttr:$val_type, StrAttr:$slice_type);
}

// ---------------- Region 1 ("host.c") Terminal Ops ----------------

def AddressOp : Op<WeftHost_Dialect, "address"> {
  let arguments = (ins Variadic<Index>:$indices, DenseI64ArrayAttr:$strides,
                       I32Attr:$index_bits);
  let results = (outs Index:$offset);
}

def HostFnOp : Op<WeftHost_Dialect, "host_fn", [IsolatedFromAbove]> {
  let arguments = (ins StrAttr:$params, DenseI64ArrayAttr:$extents,
                       DenseI64ArrayAttr:$tile, BoolAttr:$uses_q8k,
                       BoolAttr:$uses_bf16_math, BoolAttr:$uses_f16_math,
                       BoolAttr:$uses_mx8_math);
  let regions = (region AnyRegion:$body);
}

def HostRowFnOp : Op<WeftHost_Dialect, "host_row_fn", [IsolatedFromAbove]> {
  let arguments = (ins StrAttr:$params);
  let regions = (region AnyRegion:$body);
}

def HostRowOffsetOp : Op<WeftHost_Dialect, "host_row_offset"> {
  let arguments = (ins Variadic<Index>:$coords, StrAttr:$mode,
                       ArrayAttr:$dim_vars, ArrayAttr:$stride_vars);
  let results = (outs Index:$offset);
}

def HostCallRowTaskOp : Op<WeftHost_Dialect, "host_call_row_task"> {
  let arguments = (ins Variadic<Index>:$shape_vals, StrAttr:$sym_name,
                       ArrayAttr:$slot_args, ArrayAttr:$shape_exprs,
                       ArrayAttr:$stride_exprs);
}

def HostOwnerFnOp : Op<WeftHost_Dialect, "host_owner_fn", [IsolatedFromAbove]> {
  let arguments = (ins StrAttr:$params, I64Attr:$owners, I64Attr:$grain);
  let regions = (region AnyRegion:$body);
}

def HostCallOwnerTaskOp : Op<WeftHost_Dialect, "host_call_owner_task"> {
  let arguments = (ins Index:$lo, Index:$hi, StrAttr:$sym_name,
                       ArrayAttr:$slot_args, BoolAttr:$empty_extent,
                       ArrayAttr:$stride_exprs);
}

def HostAllocOp : Op<WeftHost_Dialect, "host_alloc"> {
  let arguments = (ins StrAttr:$buf_name, StrAttr:$placement, StrAttr:$c_type,
                       StrAttr:$count_expr, I64Attr:$elem_bytes,
                       ArrayAttr:$cleanup);
}

def HostIfOp : Op<WeftHost_Dialect, "host_if"> {
  let arguments = (ins Variadic<Index>:$vals, StrAttr:$cond);
  let regions = (region AnyRegion:$body);
}

def HostOmpParallelOp : Op<WeftHost_Dialect, "host_omp_parallel"> {
  let arguments = (ins Variadic<Index>:$vals, StrAttr:$num_threads,
                       StrAttr:$if_cond, BoolAttr:$release_hw_ctx);
  let regions = (region AnyRegion:$body);
}

def HostOmpForOp : Op<WeftHost_Dialect, "host_omp_for"> {
  let arguments = (ins Variadic<Index>:$vals, StrAttr:$iv, StrAttr:$lb,
                       StrAttr:$ub, StrAttr:$step, BoolAttr:$parallel,
                       BoolAttr:$simd, BoolAttr:$x86_simd, StrAttr:$schedule,
                       StrAttr:$num_threads, StrAttr:$if_cond,
                       OptionalAttr<BoolAttr>:$nowait);
  let regions = (region AnyRegion:$body);
}

def HostForOp : Op<WeftHost_Dialect, "host_for"> {
  let arguments = (ins Variadic<Index>:$vals, StrAttr:$iv, StrAttr:$lb,
                       StrAttr:$ub, StrAttr:$step);
  let regions = (region AnyRegion:$body);
}

def HostWhileOp : Op<WeftHost_Dialect, "host_while"> {
  let arguments = (ins Variadic<Index>:$vals, StrAttr:$cond);
  let regions = (region AnyRegion:$body);
}

def HostGuardReturnOp : Op<WeftHost_Dialect, "host_guard_return"> {
  let arguments = (ins StrAttr:$cond, I32Attr:$code);
}

def HostGuardContinueOp : Op<WeftHost_Dialect, "host_guard_continue"> {
  let arguments = (ins Variadic<Index>:$vals, StrAttr:$cond);
}

def HostLetI64Op : Op<WeftHost_Dialect, "host_let_i64"> {
  let arguments = (ins Variadic<Index>:$vals, StrAttr:$name, StrAttr:$expr,
                       OptionalAttr<BoolAttr>:$is_mut);
}

def HostDefI64Op : Op<WeftHost_Dialect, "host_def_i64"> {
  let arguments = (ins Variadic<Index>:$vals, StrAttr:$expr,
                       OptionalAttr<BoolAttr>:$is_mut,
                       OptionalAttr<I64Attr>:$max_bound);
  let results = (outs Index:$val);
}

def HostLoadScalarOp : Op<WeftHost_Dialect, "host_load_scalar"> {
  let arguments = (ins I64Attr:$slot, StrAttr:$c_type, StrAttr:$index_expr);
  let results = (outs AnyType:$val);
}

def HostPtrOffsetOp : Op<WeftHost_Dialect, "host_ptr_offset"> {
  let arguments = (ins Variadic<Index>:$vals, StrAttr:$c_type, StrAttr:$name,
                       StrAttr:$base_expr, StrAttr:$offset_expr);
}

def HostSpanOp : Op<WeftHost_Dialect, "host_span_op"> {
  let arguments = (ins Variadic<Index>:$vals, StrAttr:$op, StrAttr:$dst_expr,
                       StrAttr:$src_expr, StrAttr:$count_expr,
                       I64Attr:$elem_bytes);
}

def HostElemOp : Op<WeftHost_Dialect, "host_elem_op"> {
  let arguments = (ins Variadic<Index>:$vals, StrAttr:$op, StrAttr:$dst_expr,
                       StrAttr:$src_expr, StrAttr:$aux_expr,
                       I64Attr:$elem_bytes);
}

def HostPanelYieldOp : Op<WeftHost_Dialect, "host_panel_yield"> {
  let arguments = (ins AnyType:$value, BoolAttr:$widen_bf16, StrAttr:$dst_expr);
}

def HostViewPtrOp : Op<WeftHost_Dialect, "host_view_ptr"> {
  let arguments = (ins Variadic<Index>:$offset, I64Attr:$index,
                       StrAttr:$c_ptr_type, StrAttr:$base_expr,
                       I64Attr:$byte_offset);
}

def HostCallTaskOp : Op<WeftHost_Dialect, "host_call_task"> {
  let arguments = (ins Index:$rows, Index:$cols, I64Attr:$arity,
                       DenseI64ArrayAttr:$strides);
}

def HostFreeOp : Op<WeftHost_Dialect, "host_free"> {
  let arguments = (ins StrAttr:$buf_name);
}
