#ifndef INTENT_TARGET_CUTILE_IR_CUTILEOPS_TD
#define INTENT_TARGET_CUTILE_IR_CUTILEOPS_TD

include "Intent/Dialect/GPU/IR/GPUTypes.td"
include "Intent/Dialect/Intent/IR/IntentAttrs.td"
include "Intent/Target/CuTile/IR/CuTileDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"

class IntentCuTile_Op<string mnemonic, list<Trait> traits = []> :
    Op<IntentCuTile_Dialect, mnemonic, traits>;

def IntentCuTile_ArrayViewOp : IntentCuTile_Op<"array_view", [Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    DeclareOpInterfaceMethods<ViewLikeOpInterface>]> {
  let summary = "Conditional host alias collapsing contiguous full inner axes";
  let arguments = (ins IntentGPU_ViewValue:$base,
                       DenseI64ArrayAttr:$group_ends);
  // The result owns its collapsed layout and aliases the original resource.
  // Its accesses require the host-computed eligibility condition to hold.
  let results = (outs IntentGPU_ViewValue:$result, I1:$eligible);
  let hasVerifier = 1;
}

def IntentCuTile_TileLoadOp : IntentCuTile_Op<"tile_load", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins IntentGPU_ViewValue:$resource,
                       IntentGPU_ScalarValue:$allow_tma,
                       Variadic<AnyTypeOf<[Index, I32]>>:$tile_indices,
                       // 1..10 select an explicit latency; 11 retains inference.
                       Optional<Index>:$latency_policy,
                       // True only when every view axis divides into full tiles.
                       Optional<I1>:$full_tiles);
  let results = (outs IntentGPU_FragmentValue:$result);
  let hasVerifier = 1;
}

def IntentCuTile_TileStoreOp : IntentCuTile_Op<"tile_store", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins IntentGPU_ViewValue:$resource,
                       IntentGPU_ScalarValue:$allow_tma,
                       Variadic<AnyTypeOf<[Index, I32]>>:$tile_indices,
                       IntentGPU_FragmentValue:$value);
  let hasVerifier = 1;
}

def IntentCuTile_TileAtomicAddOp : IntentCuTile_Op<"tile_atomic_add", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Resource-ordered relaxed device atomic add without an old value";
  let arguments = (ins IntentGPU_ViewValue:$resource,
                       Variadic<AnyTypeOf<[Index, I32]>>:$tile_indices,
                       IntentGPU_FragmentValue:$value);
  let hasVerifier = 1;
}

def IntentCuTile_ScalarLoadOp : IntentCuTile_Op<"scalar_load", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins IntentGPU_ViewValue:$resource,
                       Variadic<Index>:$indices,
                       Optional<IntentGPU_ScalarValue>:$valid,
                       Optional<IntentGPU_ScalarValue>:$fill,
                       OptionalAttr<UnitAttr>:$in_bounds);
  let results = (outs IntentGPU_ScalarValue:$result);
  let hasVerifier = 1;
}

def IntentCuTile_ScalarStoreOp : IntentCuTile_Op<"scalar_store", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins IntentGPU_ViewValue:$resource,
                       Variadic<Index>:$indices,
                       IntentGPU_ScalarValue:$value,
                       Optional<IntentGPU_ScalarValue>:$valid,
                       OptionalAttr<UnitAttr>:$in_bounds);
  let hasVerifier = 1;
}

def IntentCuTile_GatherLoadOp : IntentCuTile_Op<"gather_load", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins IntentGPU_ViewValue:$resource,
                       Variadic<IntentGPU_DataValue>:$coordinates,
                       Optional<IntentGPU_DataValue>:$valid,
                       Optional<IntentGPU_ScalarValue>:$fill,
                       Optional<Index>:$latency_policy,
                       DenseI64ArrayAttr:$source_axes,
                       // Proven for active coordinates; does not replace valid.
                       OptionalAttr<UnitAttr>:$in_bounds);
  let results = (outs IntentGPU_FragmentValue:$result);
  let hasVerifier = 1;
}

def IntentCuTile_ScatterStoreOp : IntentCuTile_Op<"scatter_store", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins IntentGPU_ViewValue:$resource,
                       Variadic<IntentGPU_DataValue>:$coordinates,
                       IntentGPU_FragmentValue:$value,
                       Optional<IntentGPU_DataValue>:$valid,
                       DenseI64ArrayAttr:$source_axes,
                       // Proven for active coordinates; does not replace valid.
                       OptionalAttr<UnitAttr>:$in_bounds);
  let hasVerifier = 1;
}

def IntentCuTile_AtomicRMWOp : IntentCuTile_Op<"atomic_rmw", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins IntentGPU_ViewValue:$resource,
                       Variadic<IntentGPU_DataValue>:$coordinates,
                       IntentGPU_DataValue:$value,
                       Intent_AtomicRMWKindAttr:$kind,
                       Intent_AtomicOrderingAttr:$ordering,
                       IntentGPU_AtomicSharingDomainAttr:$sharing,
                       // Active coordinates are in bounds; inactive coordinates
                       // use the view-end sentinel and still need native bounds checks.
                       OptionalAttr<UnitAttr>:$in_bounds);
  let results = (outs IntentGPU_DataValue:$result);
  let hasVerifier = 1;
}

def IntentCuTile_ExtractOp : IntentCuTile_Op<"extract", [Pure]> {
  let arguments = (ins IntentGPU_FragmentValue:$source,
                       Variadic<IntentGPU_ScalarValue>:$coordinates,
                       ArrayAttr:$extraction_shape,
                       DenseI64ArrayAttr:$retained_axes);
  let results = (outs IntentGPU_DataValue:$result);
  let hasVerifier = 1;
}

def IntentCuTile_MMAOp : IntentCuTile_Op<"mma", [Pure]> {
  let arguments = (ins IntentGPU_FragmentValue:$lhs,
                       IntentGPU_FragmentValue:$rhs,
                       IntentGPU_FragmentValue:$accumulator,
                       OptionalAttr<I64Attr>:$reduction_chunk);
  let results = (outs IntentGPU_FragmentValue:$result);
  let hasVerifier = 1;
}

def IntentCuTile_ScaledMMAOp : IntentCuTile_Op<"scaled_mma", [Pure]> {
  let arguments = (ins IntentGPU_FragmentValue:$lhs,
                       IntentGPU_FragmentValue:$lhs_scale,
                       IntentGPU_FragmentValue:$rhs,
                       IntentGPU_FragmentValue:$rhs_scale,
                       IntentGPU_FragmentValue:$accumulator,
                       Intent_ScaledFormatAttr:$lhs_format,
                       Intent_ScaledFormatAttr:$rhs_format,
                       I64Attr:$lhs_group_size,
                       I64Attr:$rhs_group_size);
  let results = (outs IntentGPU_FragmentValue:$result);
  let hasVerifier = 1;
}

class IntentCuTile_Collective<string mnemonic> :
    IntentCuTile_Op<mnemonic, [Pure, IsolatedFromAbove, AttrSizedOperandSegments,
        DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Variadic<IntentGPU_DataValue>:$sources,
                       Variadic<IntentGPU_DataValue>:$identities,
                       I64Attr:$axis,
                       BoolAttr:$reverse,
                       OptionalAttr<Intent_BinaryOperatorAttr>:$kind);
  let results = (outs Variadic<IntentGPU_DataValue>:$outputs);
  let regions = (region AnyRegion:$combine);
  let assemblyFormat = [{
    `sources` `(` $sources `:` type($sources) `)`
    `identities` `(` ($identities^ `:` type($identities))? `)`
    `axis` $axis `reverse` $reverse (`kind` $kind^)?
    `combine` $combine `->` `(` type($outputs) `)` attr-dict
  }];
  let hasVerifier = 1;
}

def IntentCuTile_ReduceOp : IntentCuTile_Collective<"reduce">;
def IntentCuTile_ScanOp : IntentCuTile_Collective<"scan">;

#endif
