#ifndef INTENT_TARGET_TRITON_IR_TRITONOPS_TD
#define INTENT_TARGET_TRITON_IR_TRITONOPS_TD

include "Intent/Dialect/Block/IR/BlockTypes.td"
include "Intent/Dialect/Block/IR/BlockAttrs.td"
include "Intent/Target/Triton/IR/TritonDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"

class IntentTriton_Op<string mnemonic, list<Trait> traits = []> :
    Op<IntentTriton_Dialect, mnemonic, traits>;

def IntentTriton_CtaBarrierOp : IntentTriton_Op<"cta_barrier", [
    MemoryEffects<[MemRead, MemWrite]>
  ]>;

def IntentTriton_TensorDescriptorChoiceOp :
    IntentTriton_Op<"tensor_descriptor_choice", [Pure]> {
  let arguments = (ins Variadic<IntentBlock_ViewValue>:$descriptors,
                       StrAttr:$construction,
                       StrAttr:$selection,
                       IntentBlock_ParameterRefAttr:$config_parameter,
                       StrAttr:$eligibility_argument);
  let results = (outs I1:$result);
  let hasVerifier = 1;
}

def IntentTriton_TensorDescriptorAllocatorOp :
    IntentTriton_Op<"tensor_descriptor_allocator"> {
  let arguments = (ins I64Attr:$size_argument,
                       I64Attr:$alignment_argument,
                       I64Attr:$stream_argument,
                       StrAttr:$lifetime,
                       StrAttr:$implementation);
  let hasVerifier = 1;
}

def IntentTriton_TensorDescriptorOp : IntentTriton_Op<"tensor_descriptor", [
    Pure,
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<ViewLikeOpInterface>
  ]> {
  let arguments = (ins IntentBlock_ViewValue:$base,
                       Variadic<Index>:$shape,
                       Variadic<Index>:$strides,
                       Variadic<Index>:$block_shape,
                       DenseI64ArrayAttr:$source_block_axes,
                       DenseI64ArrayAttr:$initial_block_shape,
                       StrAttr:$base_layout,
                       StrAttr:$padding,
                       DenseI64ArrayAttr:$aligned_stride_axes,
                       DenseI64ArrayAttr:$unit_stride_axes,
                       BoolAttr:$require_positive_shape,
                       BoolAttr:$require_positive_strides,
                       BoolAttr:$require_power_of_two_block_shape,
                       I64Attr:$alignment,
                       I64Attr:$minimum_contiguous_bytes,
                       I64Attr:$pipeline_block_alignment,
                       I64Attr:$maximum_shape_extent,
                       I64Attr:$maximum_block_elements);
  let results = (outs IntentBlock_ViewValue:$result);
  let hasVerifier = 1;
}

def IntentTriton_DescriptorLoadOp : IntentTriton_Op<"descriptor_load", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins IntentBlock_ViewValue:$descriptor,
                       Variadic<I32>:$offsets);
  let results = (outs IntentBlock_FragmentValue:$result);
  let hasVerifier = 1;
}

def IntentTriton_DescriptorStoreOp : IntentTriton_Op<"descriptor_store", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins IntentBlock_ViewValue:$descriptor,
                       Variadic<I32>:$offsets,
                       IntentBlock_FragmentValue:$value);
  let hasVerifier = 1;
}

def IntentTriton_SplitOp : IntentTriton_Op<"split", [Pure]> {
  let arguments = (ins IntentBlock_FragmentValue:$source);
  let results = (outs IntentBlock_FragmentValue:$low,
                      IntentBlock_FragmentValue:$high);
  let hasVerifier = 1;
}

class IntentTriton_ElementwiseCollective<string mnemonic> :
    IntentTriton_Op<mnemonic, [Pure, IsolatedFromAbove, AttrSizedOperandSegments,
        DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Variadic<IntentBlock_DataValue>:$sources,
                       Variadic<IntentBlock_DataValue>:$identities,
                       I64Attr:$axis,
                       BoolAttr:$reverse);
  let results = (outs Variadic<AnyType>:$outputs);
  let regions = (region AnyRegion:$combine);
  let assemblyFormat = [{
    `sources` `(` $sources `:` type($sources) `)`
    `identities` `(` ($identities^ `:` type($identities))? `)`
    `axis` $axis `reverse` $reverse
    `combine` $combine `->` `(` type($outputs) `)` attr-dict
  }];
  let hasVerifier = 1;
}

// Triton's native callbacks are scalar, unlike a shared fragment combiner.
def IntentTriton_ReduceOp : IntentTriton_ElementwiseCollective<"reduce"> {
  let description = [{
    Native scalar-callback reduction. A nonnegative axis reduces that source
    axis. Axis -1 reduces every axis of the complete source tuple to scalars,
    using Triton's whole-tuple axis=None primitive; its native lowering owns
    the coordinated element ordering. No other negative axis is valid.
  }];
}
def IntentTriton_ScanOp : IntentTriton_ElementwiseCollective<"scan">;

def IntentTriton_MapElementwiseOp : IntentTriton_Op<"map_elementwise", [
    IsolatedFromAbove, RecursiveMemoryEffects
  ]> {
  let arguments = (ins Variadic<IntentBlock_DataValue>:$inputs);
  let results = (outs IntentBlock_FragmentValue:$result);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
}

#endif
