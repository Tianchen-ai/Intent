include "mlir/IR/OpBase.td"
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

def NramBufferOp : BangcOp<"nram_buffer"> {
  let arguments = (ins I64Attr:$elements);
  let results = (outs AnyMemRef:$buffer);
}

def WramBufferOp : BangcOp<"wram_buffer"> {
  let arguments = (ins I64Attr:$elements);
  let results = (outs AnyMemRef:$buffer);
}

def WramConvOp : BangcOp<"wram_conv"> {
  let arguments = (ins Variadic<AnyType>:$inputs, I64Attr:$bm, I64Attr:$bn,
                       I64Attr:$bk, StrAttr:$mode);
}

def CycleMulOp : BangcOp<"cycle_mul"> {
  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$src, AnyMemRef:$cycle,
                       I64:$lanes, I64:$period);
}

def VecMinScalarOp : BangcOp<"vec_min_scalar"> {
  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$scratch, AnyMemRef:$src,
                       F32:$bound, I64:$live, I64:$lanes);
}

def Bf16PanelGuardOp : BangcOp<"bf16_panel_ok"> {
  let arguments = (ins AnyMemRef:$panel, AnyMemRef:$mag_scratch,
                       AnyMemRef:$flag_scratch, I64:$elements);
  let results = (outs I1:$ok);
}

def Bf16IeeeDotOp : BangcOp<"bf16_ieee_dot"> {
  let arguments = (ins AnyMemRef:$acc, AnyMemRef:$lhs, AnyMemRef:$rhs_t,
                       I64:$nr, I64:$nc, I64:$nk, I64:$bk, I64:$bn);
}

def Bf16AccNarrowOp : BangcOp<"bf16_acc_narrow"> {
  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$acc, I64:$elements,
                       I1:$strict);
}

def I32ToBf16Op : BangcOp<"i32_to_bf16"> {
  let arguments = (ins I32:$value);
  let results = (outs BF16:$result);
}

def FromLeHalfOp : BangcOp<"from_le_half"> {
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

def NramAllocOp : BangcOp<"nram_alloc"> {
  let results = (outs AnyMemRef:$storage);
}

def NramPtrOp : BangcOp<"nram_ptr"> {
  let arguments = (ins AnyMemRef:$storage, I64:$byte_offset);
  let results = (outs AnyMemRef:$ptr);
}

def AddressOp : BangcOp<"address", [AttrSizedOperandSegments]> {
  let arguments = (ins AnyMemRef:$base, Variadic<I64>:$coords,
                       Variadic<I64>:$strides, I32Attr:$index_bits);
  let results = (outs AnyMemRef:$ptr);
}

def DimOp : BangcOp<"dim"> {
  let arguments = (ins AnyMemRef:$view, I64Attr:$axis);
  let results = (outs I64:$extent);
}

def StrideOp : BangcOp<"stride"> {
  let arguments = (ins AnyMemRef:$view, I64Attr:$axis);
  let results = (outs I64:$step);
}

def SymbolOp : BangcOp<"symbol"> {
  let arguments = (ins StrAttr:$name);
  let results = (outs I64:$value);
}

def LocalVarOp : BangcOp<"local_var"> {
  let arguments = (ins StrAttr:$extent);
  let results = (outs AnyMemRef:$slot);
}

def MomentArrayOp : BangcOp<"moment_array"> {
  let arguments = (ins StrAttr:$extent);
  let results = (outs AnyTuple:$array);
}

def MomentGetOp : BangcOp<"moment_get"> {
  let arguments = (ins AnyTuple:$array, I64:$index);
  let results = (outs AnyTuple:$moment);
}

def MomentSetOp : BangcOp<"moment_set"> {
  let arguments = (ins AnyTuple:$array, I64:$index, AnyTuple:$moment);
}

def SummaryZeroOp : BangcOp<"summary_zero"> {
  let results = (outs AnyTuple:$moment);
}

def SummaryReduceOp : BangcOp<"summary_reduce"> {
  let arguments = (ins AnyMemRef:$input, AnyMemRef:$scratch, I64:$live);
  let results = (outs AnyTuple:$moment);
}

def SummaryMergeOp : BangcOp<"summary_merge"> {
  let arguments = (ins AnyTuple:$lhs, AnyTuple:$rhs);
  let results = (outs AnyTuple:$moment);
}

def SummaryFieldOp : BangcOp<"summary_field"> {
  let arguments = (ins AnyTuple:$moment, I64Attr:$position);
  let results = (outs AnyType:$value);
}

def Memcpy1DOp : BangcOp<"memcpy_1d"> {
  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$src, I64:$elements,
                       StrAttr:$direction, BoolAttr:$async);
}

def Memcpy2DOp : BangcOp<"memcpy_2d"> {
  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$src, I64:$width,
                       I64:$dst_stride, I64:$src_stride, I64:$seg_num,
                       StrAttr:$direction, BoolAttr:$async);
}

def LoadScalarOp : BangcOp<"load_scalar"> {
  let arguments = (ins AnyMemRef:$ptr, I64:$index);
  let results = (outs AnyType:$value);
}

def StoreScalarOp : BangcOp<"store_scalar"> {
  let arguments = (ins AnyMemRef:$ptr, I64:$index, AnyType:$value);
}

def SyncOp : BangcOp<"sync"> {
  let arguments = (ins StrAttr:$domain);
}

def FillOp : BangcOp<"fill"> {
  let arguments = (ins AnyMemRef:$dst, I64:$lanes, AnyType:$value);
}

def AtomicAddOp : BangcOp<"atomic_add"> {
  let arguments = (ins AnyMemRef:$old_nram, AnyMemRef:$global_addr,
                       AnyMemRef:$delta_nram, I64:$lanes);
}

def VecBinaryOp : BangcOp<"vec_binary"> {
  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$lhs, AnyMemRef:$rhs,
                       I64:$lanes, StrAttr:$op);
}

def VecScalarOp : BangcOp<"vec_scalar"> {
  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$src, AnyType:$scalar,
                       I64:$lanes, StrAttr:$op);
}

def VecCastOp : BangcOp<"vec_cast"> {
  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$src, I64:$lanes,
                       StrAttr:$kind);
}

def PositiveHalfOrNanOp : BangcOp<"positive_half_or_nan"> {
  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$src, I64:$lanes);
}

def ScalarMathOp : BangcOp<"scalar_math"> {
  let arguments = (ins Variadic<AnyType>:$inputs, StrAttr:$op);
  let results = (outs AnyType:$result);
}

def RowReduceOp : BangcOp<"row_reduce"> {
  let arguments = (ins AnyMemRef:$out, AnyMemRef:$scratch, AnyMemRef:$input,
                       I64:$members, I64:$pitch, I64:$live, StrAttr:$algebra);
}

def RowExpOp : BangcOp<"row_exp"> {
  let arguments = (ins AnyMemRef:$out, AnyMemRef:$scratch, AnyMemRef:$input,
                       I64:$active, I64:$lanes);
}

def RowExp2Op : BangcOp<"row_exp2"> {
  let arguments = (ins AnyMemRef:$out, AnyMemRef:$scratch, AnyMemRef:$input,
                       I64:$active, I64:$lanes);
}

def Reduce32SumOp : BangcOp<"reduce32_sum"> {
  let arguments = (ins AnyMemRef:$buffer);
  let results = (outs F32:$sum);
}

def SumpoolOp : BangcOp<"sumpool"> {
  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$src, I64:$channel,
                       I64:$height);
}

def TileTransposeOp : BangcOp<"tile_transpose"> {
  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$src, I64:$rows, I64:$cols);
}

def NarrowQuotientOp : BangcOp<"narrow_quotient"> {
  let arguments = (ins AnyMemRef:$out, AnyMemRef:$quotient, AnyMemRef:$scratch,
                       AnyMemRef:$numerator, AnyMemRef:$divisor, I64:$members,
                       I64:$pitch, I64:$live, I64:$row_lanes);
}

def Bf16TileOp : BangcOp<"bf16_tile"> {
  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$src, I64:$live, I64:$lanes,
                       StrAttr:$mode);
}

def Bf16ProductOp : BangcOp<"bf16_product"> {
  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$lhs, AnyMemRef:$rhs,
                       AnyMemRef:$scratch, I64:$live, I64:$lanes);
}

def VecSqrtRnOp : BangcOp<"vec_sqrt_rn"> {
  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$src, AnyMemRef:$scratch,
                       I64:$live, I64:$lanes);
}

def VecDivRnOp : BangcOp<"vec_div_rn"> {
  let arguments = (ins AnyMemRef:$dst, AnyType:$lhs, AnyType:$rhs,
                       AnyMemRef:$scratch, I64:$live, I64:$lanes,
                       StrAttr:$mode);
}

def Bf16ScalarOp : BangcOp<"bf16_scalar"> {
  let arguments = (ins Variadic<AnyType>:$inputs, StrAttr:$mode);
  let results = (outs AnyType:$result);
}

def ScanSumOp : BangcOp<"scan_sum"> {
  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$src, AnyMemRef:$scratch,
                       I64:$lanes, I64:$live);
}

def PhiloxUniformOp : BangcOp<"philox_uniform"> {
  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$scratch, AnyInteger:$seed,
                       I64:$base_index, I64:$live);
}

def PhiloxAtOp : BangcOp<"philox_at"> {
  let arguments = (ins AnyInteger:$seed, AnyInteger:$index);
  let results = (outs F32:$value);
}
