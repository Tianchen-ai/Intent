include "mlir/IR/OpBase.td"

def WeftHost_Dialect : Dialect {
  let name = "weft_host";
  let cppNamespace = "::intentdsl::wefthost";
}

// Dispatch disjoint intervals of the domain to one serial task instance per
// interval. Matrix captures advance with the interval; vectors and scalars are shared.
def DispatchRows : Op<WeftHost_Dialect, "dispatch_rows", [IsolatedFromAbove, SingleBlock]> {
  let arguments = (ins Variadic<AnyType>:$captures, I64Attr:$domain_arg,
                       ArrayAttr:$rows, StrAttr:$task_symbol);
  let results = (outs Index:$completion);
  let regions = (region AnyRegion:$task);
}

def PanelSweep : Op<WeftHost_Dialect, "sweep", [SingleBlock]> {
  let arguments = (ins StrAttr:$combine, StrAttr:$tail);
  let results = (outs Variadic<AnyType>:$summary);
  let regions = (region AnyRegion:$panels);
}
def OwnerCapture : Op<WeftHost_Dialect, "capture_owner"> {
  let arguments = (ins AnyMemRef:$view, Index:$owner);
  let results = (outs AnyType:$snapshot);
}
def PanelRead : Op<WeftHost_Dialect, "read_panel"> {
  let arguments = (ins AnyMemRef:$source, Index:$row, Index:$offset, Index:$extent);
  let results = (outs AnyTensor:$value);
}
def PanelWrite : Op<WeftHost_Dialect, "write_panel"> {
  let arguments = (ins AnyMemRef:$dest, Index:$row, Index:$offset, Index:$extent, AnyTensor:$value);
}
def PanelYield : Op<WeftHost_Dialect, "yield", [Terminator]> {
  let arguments = (ins Variadic<AnyType>:$partial);
}

// ---------------- v0.4 Terminal Matrix Program Ops ----------------

def MatrixProgramOp : Op<WeftHost_Dialect, "matrix_program"> {
  let arguments = (ins DenseI64ArrayAttr:$tile, DenseI64ArrayAttr:$extents,
                       StrAttr:$implementation, BoolAttr:$uses_brgemm);
  let regions = (region AnyRegion:$task, AnyRegion:$host);
}

def EndOp : Op<WeftHost_Dialect, "end", [Terminator]> {}

// Region 0 ("task.weftir") terminal ops
def EncodingDeclOp : Op<WeftHost_Dialect, "encoding_decl"> {
  let arguments = (ins StrAttr:$sym_name, BoolAttr:$is_weight);
}

def TaskKernelOp : Op<WeftHost_Dialect, "task_kernel"> {
  let arguments = (ins StrAttr:$sym_name, StrAttr:$source,
                       ArrayAttr:$view_types, ArrayAttr:$arg_writes,
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
                       StrAttr:$cond_type, StrAttr:$val_type);
  let results = (outs Index:$res);
}

def TaskCastOp : Op<WeftHost_Dialect, "task_cast"> {
  let arguments = (ins Index:$input, StrAttr:$op_kind,
                       StrAttr:$in_type, StrAttr:$val_type);
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

def TaskContractOp : Op<WeftHost_Dialect, "task_contract"> {
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

def TaskYieldOp : Op<WeftHost_Dialect, "task_yield", [Terminator]> {
  let arguments = (ins Variadic<Index>:$values, ArrayAttr:$val_types);
}

def TaskSubquantLevelOp : Op<WeftHost_Dialect, "task_subquant_level"> {
  let arguments = (ins Index:$root_dom, Index:$init_s, Index:$init_t);
  let results = (outs Index:$res_s, Index:$res_t);
  let regions = (region AnyRegion:$body);
}

def TaskExtractOp : Op<WeftHost_Dialect, "task_extract"> {
  let arguments = (ins Index:$input, Index:$point, StrAttr:$selector,
                       StrAttr:$in_type, StrAttr:$point_type, StrAttr:$val_type);
  let results = (outs Index:$res);
}

def TaskHandoffOp : Op<WeftHost_Dialect, "task_handoff", [Terminator]> {
  let arguments = (ins Index:$val_s, Index:$val_t);
}

def TaskCommitOp : Op<WeftHost_Dialect, "task_commit"> {
  let arguments = (ins Index:$value, Index:$slice,
                       StrAttr:$val_type, StrAttr:$slice_type);
}

// Region 1 ("host.c") terminal ops
def AddressOp : Op<WeftHost_Dialect, "address"> {
  let arguments = (ins Variadic<Index>:$indices, DenseI64ArrayAttr:$strides,
                       I32Attr:$index_bits);
  let results = (outs Index:$offset);
}

def HostFnOp : Op<WeftHost_Dialect, "host_fn"> {
  let arguments = (ins StrAttr:$params, DenseI64ArrayAttr:$extents,
                       DenseI64ArrayAttr:$tile, BoolAttr:$uses_q8k,
                       BoolAttr:$uses_bf16_math, BoolAttr:$uses_f16_math,
                       BoolAttr:$uses_mx8_math);
  let regions = (region AnyRegion:$body);
}

def HostAllocOp : Op<WeftHost_Dialect, "host_alloc"> {
  let arguments = (ins StrAttr:$buf_name, StrAttr:$placement, StrAttr:$c_type,
                       StrAttr:$count_expr, I64Attr:$elem_bytes,
                       ArrayAttr:$cleanup);
}

def HostPrepOp : Op<WeftHost_Dialect, "host_prep"> {
  let arguments = (ins StrAttr:$kind, StrAttr:$dst_buf, ArrayAttr:$cleanup);
  let regions = (region VariadicRegion<AnyRegion>:$mapRegion);
}

def HostColLoopOp : Op<WeftHost_Dialect, "host_col_loop"> {
  let regions = (region AnyRegion:$body);
}

def HostPanelYieldOp : Op<WeftHost_Dialect, "host_panel_yield", [Terminator]> {
  let arguments = (ins AnyType:$value);
}

def HostParallelTilesOp : Op<WeftHost_Dialect, "host_parallel_tiles"> {
  let arguments = (ins DenseI64ArrayAttr:$extents, DenseI64ArrayAttr:$tile,
                       BoolAttr:$grouped, BoolAttr:$routed, BoolAttr:$windowed,
                       I64Attr:$head_extent, I64Attr:$chunk_extent,
                       I64Attr:$multiplicity, I64Attr:$offsets_slot,
                       I64Attr:$offsets_stride, I64Attr:$routes_slot,
                       I64Attr:$routes_stride);
  let regions = (region AnyRegion:$body);
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

def HostScatterRoutedOp : Op<WeftHost_Dialect, "host_scatter_routed"> {
  let arguments = (ins I64Attr:$dst_slot, I64Attr:$multiplicity,
                       DenseI64ArrayAttr:$dst_strides, I64Attr:$elem_bytes,
                       I64Attr:$routes_slot, I64Attr:$routes_stride);
}

def HostFreeOp : Op<WeftHost_Dialect, "host_free"> {
  let arguments = (ins StrAttr:$buf_name);
}
