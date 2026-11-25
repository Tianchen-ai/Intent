include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def BangcSrc_Dialect : Dialect {
  let name = "bangc_src";
  let cppNamespace = "::intentdsl::bangc_src";
}

class BangcOp<string mnemonic, list<Trait> traits = []>
    : Op<BangcSrc_Dialect, mnemonic, traits>;

def RowProgramOp : BangcOp<"row_program", [NoTerminator]> {
  let arguments = (ins I64Attr:$batch, I64Attr:$depth);
  let regions = (region AnyRegion:$sources, AnyRegion:$device, AnyRegion:$host);
}

def MatrixProgramOp : BangcOp<"matrix_program", [NoTerminator]> {
  let arguments = (ins I64Attr:$tile_m, I64Attr:$tile_n, I64Attr:$tile_k);
  let regions = (region AnyRegion:$sources, AnyRegion:$device, AnyRegion:$host);
}

def OrderedProgramOp : BangcOp<"ordered_program", [NoTerminator]> {
  let arguments = (ins I64Attr:$owner_tile, I64Attr:$owners,
                       I64Attr:$participants, I64Attr:$owner_bytes);
  let regions = (region AnyRegion:$sources, AnyRegion:$device, AnyRegion:$host);
}

def NramBufferOp : BangcOp<"nram_buffer",
                           [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins I64Attr:$elements);
  let results = (outs AnyMemRef:$buffer);
}

def WramBufferOp : BangcOp<"wram_buffer",
                           [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins I64Attr:$elements);
  let results = (outs AnyMemRef:$buffer);
}

def WramConvOp : BangcOp<"wram_conv",
                         [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins Variadic<AnyType>:$inputs, I64Attr:$bm, I64Attr:$bn,
                       I64Attr:$bk, StrAttr:$mode);
}

def CycleMulOp : BangcOp<"cycle_mul",
                         [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$src, AnyMemRef:$cycle,
                       I64:$lanes, I64:$period);
}

def VecMinScalarOp : BangcOp<"vec_min_scalar",
                             [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$scratch, AnyMemRef:$src,
                       F32:$bound, I64:$live, I64:$lanes);
}

def Bf16PanelGuardOp : BangcOp<"bf16_panel_ok",
                               [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyMemRef:$panel, AnyMemRef:$mag_scratch,
                       AnyMemRef:$flag_scratch, I64:$elements);
  let results = (outs I1:$ok);
}

def Bf16IeeeDotOp : BangcOp<"bf16_ieee_dot",
                            [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyMemRef:$acc, AnyMemRef:$lhs, AnyMemRef:$rhs_t,
                       I64:$nr, I64:$nc, I64:$nk, I64:$bk, I64:$bn);
}

def Bf16AccNarrowOp : BangcOp<"bf16_acc_narrow",
                              [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$acc, I64:$elements,
                       I1:$strict);
}

def I32ToBf16Op : BangcOp<"i32_to_bf16", [Pure]> {
  let arguments = (ins I32:$value);
  let results = (outs BF16:$result);
}

def FromLeHalfOp : BangcOp<"from_le_half", [Pure]> {
  let arguments = (ins I32:$lo, I32:$hi);
  let results = (outs F32:$result);
}

def RuntimeSourceOp : BangcOp<"runtime_source"> {
  let arguments = (ins StrAttr:$implementation);
}

def DeviceKernelOp : BangcOp<"device_kernel", [SingleBlock, NoTerminator]> {
  let arguments = (ins Variadic<AnyType>:$args);
  let regions = (region AnyRegion:$body);
}

def HostEnqueueOp : BangcOp<"host_enqueue"> {
  let arguments = (ins Variadic<AnyType>:$args);
}

def NramAllocOp : BangcOp<"nram_alloc",
                          [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let results = (outs AnyMemRef:$storage);
}

def NramPtrOp : BangcOp<"nram_ptr", [Pure]> {
  let arguments = (ins AnyMemRef:$storage, I64:$byte_offset);
  let results = (outs AnyMemRef:$ptr);
}

def AddressOp : BangcOp<"address", [Pure, AttrSizedOperandSegments]> {
  let arguments = (ins AnyMemRef:$base, Variadic<I64>:$coords,
                       Variadic<I64>:$strides, I32Attr:$index_bits);
  let results = (outs AnyMemRef:$ptr);
}

def DimOp : BangcOp<"dim", [Pure]> {
  let arguments = (ins AnyMemRef:$view, I64Attr:$axis);
  let results = (outs I64:$extent);
}

def StrideOp : BangcOp<"stride", [Pure]> {
  let arguments = (ins AnyMemRef:$view, I64Attr:$axis);
  let results = (outs I64:$step);
}

def SymbolOp : BangcOp<"symbol",
                       [Pure, DeclareOpInterfaceMethods<InferIntRangeInterface,
                                                        ["inferResultRanges"]>]> {
  let arguments = (ins StrAttr:$name);
  let results = (outs I64:$value);
}

def LocalVarOp : BangcOp<"local_var",
                         [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins StrAttr:$extent);
  let results = (outs AnyMemRef:$slot);
}

def MomentArrayOp : BangcOp<"moment_array",
                            [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins StrAttr:$extent);
  let results = (outs AnyTuple:$array);
}

def MomentGetOp : BangcOp<"moment_get",
                          [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyTuple:$array, I64:$index);
  let results = (outs AnyTuple:$moment);
}

def MomentSetOp : BangcOp<"moment_set",
                          [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyTuple:$array, I64:$index, AnyTuple:$moment);
}

def SummaryZeroOp : BangcOp<"summary_zero", [Pure]> {
  let results = (outs AnyTuple:$moment);
}

def SummaryReduceOp : BangcOp<"summary_reduce",
                              [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyMemRef:$input, AnyMemRef:$scratch, I64:$live);
  let results = (outs AnyTuple:$moment);
}

def SummaryMergeOp : BangcOp<"summary_merge", [Pure]> {
  let arguments = (ins AnyTuple:$lhs, AnyTuple:$rhs);
  let results = (outs AnyTuple:$moment);
}

def SummaryFieldOp : BangcOp<"summary_field", [Pure]> {
  let arguments = (ins AnyTuple:$moment, I64Attr:$position);
  let results = (outs AnyType:$value);
}

def Memcpy1DOp : BangcOp<"memcpy_1d",
                         [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$src, I64:$elements,
                       StrAttr:$direction, BoolAttr:$async);
}

def Memcpy2DOp : BangcOp<"memcpy_2d",
                         [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$src, I64:$width,
                       I64:$dst_stride, I64:$src_stride, I64:$seg_num,
                       StrAttr:$direction, BoolAttr:$async);
}

def LoadScalarOp : BangcOp<"load_scalar",
                           [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyMemRef:$ptr, I64:$index);
  let results = (outs AnyType:$value);
}

def StoreScalarOp : BangcOp<"store_scalar",
                            [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyMemRef:$ptr, I64:$index, AnyType:$value);
}

def SyncOp : BangcOp<"sync",
                     [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins StrAttr:$domain);
}

def FillOp : BangcOp<"fill",
                     [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyMemRef:$dst, I64:$lanes, AnyType:$value);
}

def AtomicAddOp : BangcOp<"atomic_add",
                          [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyMemRef:$old_nram, AnyMemRef:$global_addr,
                       AnyMemRef:$delta_nram, I64:$lanes);
}

def VecBinaryOp : BangcOp<"vec_binary",
                          [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$lhs, AnyMemRef:$rhs,
                       I64:$lanes, StrAttr:$op);
}

def VecScalarOp : BangcOp<"vec_scalar",
                          [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$src, AnyType:$scalar,
                       I64:$lanes, StrAttr:$op);
}

def VecCastOp : BangcOp<"vec_cast",
                        [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$src, I64:$lanes,
                       StrAttr:$kind);
}

def PositiveHalfOrNanOp : BangcOp<"positive_half_or_nan",
                                  [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$src, I64:$lanes);
}

def ScalarMathOp : BangcOp<"scalar_math", [Pure]> {
  let arguments = (ins Variadic<AnyType>:$inputs, StrAttr:$op);
  let results = (outs AnyType:$result);
}

def RowReduceOp : BangcOp<"row_reduce",
                          [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyMemRef:$out, AnyMemRef:$scratch, AnyMemRef:$input,
                       I64:$members, I64:$pitch, I64:$live, StrAttr:$algebra);
}

def RowExpOp : BangcOp<"row_exp",
                       [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyMemRef:$out, AnyMemRef:$scratch, AnyMemRef:$input,
                       I64:$active, I64:$lanes);
}

def RowExp2Op : BangcOp<"row_exp2",
                        [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyMemRef:$out, AnyMemRef:$scratch, AnyMemRef:$input,
                       I64:$active, I64:$lanes);
}

def RowLogOp : BangcOp<"row_log",
                       [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyMemRef:$out, AnyMemRef:$scratch, AnyMemRef:$input,
                       I64:$active, I64:$lanes);
}

def VecCmpMaskOp : BangcOp<"vec_cmpf_mask",
                           [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyMemRef:$mask_out, AnyMemRef:$scratch, AnyMemRef:$lhs,
                       AnyType:$rhs, I64:$lanes, I32Attr:$pred_code,
                       BoolAttr:$rhs_is_vec);
}

def VecBitSelectOp : BangcOp<"vec_bitselect",
                             [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$scratch, AnyMemRef:$mask,
                       AnyType:$true_val, AnyType:$false_val, I64:$lanes,
                       StrAttr:$mode);
}

def QuantInitLutOp : BangcOp<"quant_init_lut",
                             [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyMemRef:$lut, I32:$value_code, I32:$pack_bits,
                       I32:$zero_bias, I32:$s0, I32:$s1);
}

def QuantInitRecordLutOp : BangcOp<"quant_init_record_lut",
                                   [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyMemRef:$lut, I32:$sunit0, I32:$bits0,
                       I32:$zero_bias, I32:$sunit1, I32:$bits1, I32:$doff1);
}

def QuantE8m0GuardOp : BangcOp<"quant_e8m0_ok",
                               [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyMemRef:$sa, I64:$na, AnyMemRef:$sb, I64:$nb,
                       AnyMemRef:$scratch, I32:$min_exp, I32:$max_exp);
  let results = (outs I1:$ok);
}

def QuantDecodeBlockScaledOp
    : BangcOp<"quant_decode_block_scaled",
              [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$raw, AnyMemRef:$scales,
                       AnyMemRef:$lut, AnyMemRef:$idx_scratch,
                       AnyMemRef:$scale_scratch, AnyMemRef:$tr_scratch,
                       I64:$active_rows, I64:$padded_rows,
                       I64:$groups_per_panel, I64:$block_elems,
                       I32:$scale_bias, I32:$col_major);
}

def QuantDecodeGroupedRhsOp
    : BangcOp<"quant_decode_grouped_rhs",
              [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyMemRef:$trans_f16, AnyMemRef:$qw_bytes,
                       AnyMemRef:$sc_f16, AnyMemRef:$qz_words, AnyMemRef:$lut,
                       AnyMemRef:$idx_scratch, AnyMemRef:$work_a,
                       AnyMemRef:$work_b, I64:$nk, I64:$nc, I64:$bk, I64:$bn,
                       I64:$block_elems, I64:$kk_mod_block, I32:$has_zero,
                       DenseI32ArrayAttr:$lane_shifts);
}

def QuantRecordGemvOp
    : BangcOp<"quant_record_gemv",
              [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyMemRef:$acc, AnyMemRef:$rec_panel, AnyMemRef:$rec_t,
                       AnyMemRef:$q_act_f32, AnyMemRef:$lut, AnyMemRef:$idx16,
                       AnyMemRef:$w16, AnyMemRef:$h16, AnyMemRef:$wf32,
                       F32:$ds, I64:$nr, DenseI32ArrayAttr:$plane_params);
}

def Reduce32SumOp : BangcOp<"reduce32_sum",
                            [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyMemRef:$buffer);
  let results = (outs F32:$sum);
}

def SumpoolOp : BangcOp<"sumpool",
                        [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$src, I64:$channel,
                       I64:$height);
}

def TileTransposeOp : BangcOp<"tile_transpose",
                              [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$src, I64:$rows, I64:$cols);
}

def NarrowQuotientOp : BangcOp<"narrow_quotient",
                               [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyMemRef:$out, AnyMemRef:$quotient, AnyMemRef:$scratch,
                       AnyMemRef:$numerator, AnyMemRef:$divisor, I64:$members,
                       I64:$pitch, I64:$live, I64:$row_lanes);
}

def Bf16TileOp : BangcOp<"bf16_tile",
                         [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins Variadic<AnyMemRef>:$buffers, I64:$live, I64:$lanes,
                       StrAttr:$mode);
}

def Bf16ProductOp : BangcOp<"bf16_product",
                            [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$lhs, AnyMemRef:$rhs,
                       AnyMemRef:$scratch, I64:$live, I64:$lanes);
}

def VecSqrtRnOp : BangcOp<"vec_sqrt_rn",
                          [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$src, AnyMemRef:$scratch,
                       I64:$live, I64:$lanes);
}

def VecDivRnOp : BangcOp<"vec_div_rn",
                         [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyMemRef:$dst, AnyType:$lhs, AnyType:$rhs,
                       AnyMemRef:$scratch, I64:$live, I64:$lanes,
                       StrAttr:$mode);
}

def Bf16ScalarOp : BangcOp<"bf16_scalar", [Pure]> {
  let arguments = (ins Variadic<AnyType>:$inputs, StrAttr:$mode);
  let results = (outs AnyType:$result);
}

def ScanSumOp : BangcOp<"scan_sum",
                        [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$src, AnyMemRef:$scratch,
                       I64:$lanes, I64:$live);
}

def PhiloxUniformOp : BangcOp<"philox_uniform",
                              [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$scratch, AnyInteger:$seed,
                       I64:$base_index, I64:$live);
}

def PhiloxAtOp : BangcOp<"philox_at", [Pure]> {
  let arguments = (ins AnyInteger:$seed, AnyInteger:$index);
  let results = (outs F32:$value);
}
