#ifndef INTENT_DIALECT_CPU_IR_CPUOPS_TD
#define INTENT_DIALECT_CPU_IR_CPUOPS_TD

include "Intent/Dialect/CPU/IR/CPUAttrs.td"
include "Intent/Dialect/CPU/IR/CPUOpInterfaces.td"
include "Intent/Dialect/Intent/IR/IntentAttrs.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Dialect/Bufferization/IR/BufferViewFlowOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class IntentCPU_Op<string name, list<Trait> traits = []>
    : Op<IntentCPU_Dialect, name, traits>;

def IntentCPU_ShapedValue : AnyTypeOf<[AnyRankedTensor, AnyMemRef]>;

def IntentCPU_TasksOp : IntentCPU_Op<"tasks", [SingleBlock,
    IsolatedFromAbove, AutomaticAllocationScope, RecursiveMemoryEffects,
    DeclareOpInterfaceMethods<BufferViewFlowOpInterface>]> {
  let summary = "Synchronous tasks over [0, count), with explicit captured values";
  let arguments = (ins Index:$count, Variadic<AnyType>:$captures);
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
}

def IntentCPU_TaskDispatchOp : IntentCPU_Op<"task_dispatch", [SingleBlock,
    AutomaticAllocationScope, RecursiveMemoryEffects]> {
  let summary = "Synchronous task invocation with an explicit active worker budget";
  let arguments = (ins Index:$count, Index:$worker_count);
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
}

def IntentCPU_TaskYieldOp : IntentCPU_Op<"task_yield", [Pure, Terminator]> {
  let hasVerifier = 1;
}

def IntentCPU_InvokeOp : IntentCPU_Op<"invoke", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
    DeclareOpInterfaceMethods<BufferViewFlowOpInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Synchronous invocation borrowing its explicitly accessed storage";
  let description = [{
    Invokes a lowered callable and completes all its work before returning.
    Arguments remain owned by the caller: the callee neither retains their
    addresses nor releases their storage. Results are written to explicit
    destination operands. Each operand has a data-access mask: 0 for none,
    1 for read, 2 for write, or 3 for both; scalar arguments require 0.
    The provider binding verifier checks these effects against the callable.
  }];
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$arguments,
      DenseI32ArrayAttr:$accesses);
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    bool mayBeTerminalBuffer(::mlir::Value value);
  }];
}

def IntentCPU_ReduceOp : IntentCPU_Op<"reduce", [SingleBlock,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "One-axis reduction with explicit scalar combine and numerical ordering permissions";
  let arguments = (ins Index:$extent,
    AnyTypeOf<[AnyFloat, AnyInteger, Index]>:$initial,
    Variadic<AnyType>:$inputs, ArrayAttr:$indexingMaps,
    IntentCPU_ReductionOrderAttr:$order);
  let results = (outs AnyTypeOf<[AnyFloat, AnyInteger, Index]>:$result);
  let regions = (region AnyRegion:$combine);
  let hasVerifier = 1;
}

def IntentCPU_YieldOp : IntentCPU_Op<"yield", [Pure, Terminator,
    HasParent<"ReduceOp">]> {
  let arguments = (ins AnyTypeOf<[AnyFloat, AnyInteger, Index]>:$value);
}

def IntentCPU_SliceReduceOp : IntentCPU_Op<"slice_reduce", [SingleBlock,
    IsolatedFromAbove, AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<BufferViewFlowOpInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Product reduction over complete source slices with explicit combine";
  let arguments = (ins Variadic<IntentCPU_ShapedValue>:$sources,
      Variadic<AnyType>:$identities, Variadic<AnyType>:$captures,
      Variadic<IntentCPU_ShapedValue>:$outputs, DenseI64ArrayAttr:$axes,
      IntentCPU_ReductionOrderAttr:$order);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$combine);
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    bool mayBeTerminalBuffer(::mlir::Value value);
    bool isDestinationPassing() {
      return !getCombine().empty() && getCombine().front().getNumArguments() ==
          3 * getSources().size() + getCaptures().size();
    }
  }];
}

def IntentCPU_SliceReduceYieldOp : IntentCPU_Op<"slice_reduce_yield", [Pure,
    Terminator, HasParent<"SliceReduceOp">]> {
  let arguments = (ins Variadic<AnyType>:$values);
}

def IntentCPU_ScanOp : IntentCPU_Op<"scan", [SingleBlock, IsolatedFromAbove, AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<BufferViewFlowOpInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Ordered prefixes with typed scalar or destination-passing slice combines";
  let arguments = (ins Variadic<IntentCPU_ShapedValue>:$sources,
      Variadic<AnyType>:$initials, Variadic<AnyType>:$captures,
      Variadic<IntentCPU_ShapedValue>:$outputs,
      I64Attr:$axis, BoolAttr:$inclusive, BoolAttr:$reverse);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$combine);
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    bool mayBeTerminalBuffer(::mlir::Value value);
    bool isDestinationPassing() {
      return !getCombine().empty() && getCombine().front().getNumArguments() ==
          3 * getSources().size() + getCaptures().size();
    }
  }];
}

def IntentCPU_ScanYieldOp : IntentCPU_Op<"scan_yield", [Pure, Terminator,
    HasParent<"ScanOp">]> {
  let arguments = (ins Variadic<AnyType>:$values);
}

def IntentCPU_HistogramOp : IntentCPU_Op<"histogram", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Typed active values contributing to logical integer bins";
  let arguments = (ins IntentCPU_ShapedValue:$values,
      IntentCPU_ShapedValue:$valid,
      IntentCPU_ShapedValue:$output, BoolAttr:$unsigned_values);
  let results = (outs Optional<AnyRankedTensor>:$result);
  let hasVerifier = 1;
}

// Synchronization orders surrounding memory effects, not only the target cell.
class IntentCPU_AtomicOp<string name> : IntentCPU_Op<name,
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let hasVerifier = 1;
}

def IntentCPU_AtomicLoadOp : IntentCPU_AtomicOp<"atomic_load"> {
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$target, Variadic<Index>:$indices,
      Intent_AtomicOrderingAttr:$ordering);
  let results = (outs AnyType:$value);
}

def IntentCPU_AtomicStoreOp : IntentCPU_AtomicOp<"atomic_store"> {
  let arguments = (ins Arg<AnyMemRef, "", [MemWrite]>:$target, AnyType:$value,
      Variadic<Index>:$indices, Intent_AtomicOrderingAttr:$ordering);
}

def IntentCPU_AtomicRMWOp : IntentCPU_AtomicOp<"atomic_rmw"> {
  let arguments = (ins Arg<AnyMemRef, "", [MemRead, MemWrite]>:$target, AnyType:$value,
      Variadic<Index>:$indices, Intent_AtomicOrderingAttr:$ordering,
      Intent_AtomicRMWKindAttr:$kind, BoolAttr:$unsigned_integer);
  let results = (outs AnyType:$old_value);
}

def IntentCPU_AtomicCompareExchangeOp : IntentCPU_AtomicOp<"atomic_compare_exchange"> {
  let arguments = (ins Arg<AnyMemRef, "", [MemRead, MemWrite]>:$target, AnyType:$expected, AnyType:$desired,
      Variadic<Index>:$indices, Intent_AtomicOrderingAttr:$ordering);
  let results = (outs AnyType:$old_value, I1:$success);
}

def IntentCPU_QuantizeOp : IntentCPU_Op<"quantize", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Quantize complete records into invocation-owned storage";
  let arguments = (ins IntentCPU_ShapedValue:$input,
      IntentCPU_ShapedValue:$output, Intent_QuantFormatAttr:$format);
  let results = (outs Optional<AnyRankedTensor>:$result);
  let hasVerifier = 1;
}

def IntentCPU_QuantizedDotOp : IntentCPU_Op<"quantized_dot", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Independent closed record dots with explicit output ownership";
  let arguments = (ins IntentCPU_ShapedValue:$lhs,
      IntentCPU_ShapedValue:$rhs,
      IntentCPU_ShapedValue:$output,
      Intent_QuantFormatAttr:$lhs_format, Intent_QuantFormatAttr:$rhs_format);
  let results = (outs Optional<AnyRankedTensor>:$result);
  let hasVerifier = 1;
}

class IntentCPU_RegionOp<string name> : IntentCPU_Op<name,
    [IsolatedFromAbove, AttrSizedOperandSegments, IntentCPU_RegionOpInterface,
     DeclareOpInterfaceMethods<BufferViewFlowOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    bool mayBeTerminalBuffer(::mlir::Value value);
  }];
}

def IntentCPU_RegionFoldOp : IntentCPU_RegionOp<"region_fold"> {
  let summary = "Ordered source slices and destination-passing summary helpers";
  let arguments = (ins Variadic<IntentCPU_ShapedValue>:$sources, Variadic<AnyType>:$identities,
      Variadic<AnyType>:$captures, Variadic<IntentCPU_ShapedValue>:$summary_outputs,
      I64Attr:$axis, ArrayAttr:$summary_fields, OptionalAttr<I64Attr>:$segment_size);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$summarize, SizedRegion<1>:$combine);
}

def IntentCPU_RegionScanOp : IntentCPU_Op<"region_scan", [
    IsolatedFromAbove, AttrSizedOperandSegments, AttrSizedResultSegments,
    IntentCPU_RegionOpInterface,
    DeclareOpInterfaceMethods<BufferViewFlowOpInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Region transitions, incoming state, slice outputs and final state";
  let arguments = (ins Variadic<IntentCPU_ShapedValue>:$sources, Variadic<AnyType>:$identities,
      Variadic<AnyType>:$initial_states, Variadic<AnyType>:$captures,
      Variadic<IntentCPU_ShapedValue>:$emitted_outputs, Variadic<IntentCPU_ShapedValue>:$final_state_outputs,
      I64Attr:$axis, ArrayAttr:$summary_fields, ArrayAttr:$state_fields,
      DenseI64ArrayAttr:$output_axes, OptionalAttr<I64Attr>:$segment_size);
  let results = (outs Variadic<AnyType>:$emitted_results,
      Variadic<AnyType>:$final_states);
  let hasVerifier = 1;
  let regions = (region SizedRegion<1>:$summarize, SizedRegion<1>:$combine,
      SizedRegion<1>:$apply, SizedRegion<1>:$emit);
  let extraClassDeclaration = [{
    bool mayBeTerminalBuffer(::mlir::Value value);
    ::mlir::Region *getApplyRegion() { return &getApply(); }
    ::mlir::Region *getEmitRegion() { return &getEmit(); }
  }];
}

def IntentCPU_RegionYieldOp : IntentCPU_Op<"region_yield", [Pure, Terminator]> {
  let arguments = (ins Variadic<AnyType>:$values);
}

include "Intent/Dialect/CPU/IR/CPUValueOps.td"

#endif
