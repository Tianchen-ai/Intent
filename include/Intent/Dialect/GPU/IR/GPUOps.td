#ifndef INTENT_DIALECT_GPU_IR_GPUOPS_TD
#define INTENT_DIALECT_GPU_IR_GPUOPS_TD

include "Intent/Dialect/GPU/IR/GPUDialect.td"
include "Intent/Dialect/GPU/IR/GPUAttrs.td"
include "Intent/Dialect/GPU/IR/GPUTypes.td"
include "Intent/Dialect/GPU/IR/AccessOpInterface.td"
include "Intent/Dialect/GPU/IR/FragmentOpInterface.td"
include "Intent/Dialect/Intent/IR/IntentAttrs.td"
include "Intent/Interfaces/StructuredOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class IntentGPU_Op<string mnemonic, list<Trait> traits = []> :
    Op<IntentGPU_Dialect, mnemonic, traits>;

class IntentGPU_AccessOp<string mnemonic, string kind, string resource,
                         string payloads, string valueType, int payloadCount,
                         list<Trait> traits = [], string extra = ""> :
    IntentGPU_Op<mnemonic, !listconcat(traits, [IntentGPU_AccessOpInterface])> {
  let extraClassDeclaration =
    "::intent::gpu::AccessKind getAccessKind() { return ::intent::gpu::AccessKind::" # kind # "; }\n" #
    "::mlir::OpOperand &getAccessResourceOperand() { return " # resource # "; }\n" #
    "::mlir::MutableOperandRange getAccessPayloadsMutable() { return " # payloads # "; }\n" #
    "::mlir::Type getAccessValueType() { return " # valueType # "; }\n" #
    "::mlir::LogicalResult verifyAccessOperandGroups() {\n" #
    "  if (getODSOperands(0).size() != 1) return emitOpError(\"requires one access resource/source operand\");\n" #
    "  for (int slot = 0; slot < " # !cast<string>(payloadCount) # "; ++slot)\n" #
    "    if (getODSOperands(2 + slot).size() != 1) return emitOpError(\"requires one operand per access payload group\");\n" #
    "  return ::mlir::success();\n" #
    "}\n" # extra;
}

class IntentGPU_ReadOp<string mnemonic, string kind, string resource,
                       list<Trait> traits = []> :
    IntentGPU_AccessOp<mnemonic, kind, resource,
      "::mlir::MutableOperandRange(getOperation(), 0, 0)",
      "getResult().getType()", 0, traits,
    "::mlir::Value getAccessFill() { return getFill(); }\n" #
    "std::optional<::mlir::MutableOperandRange> getAccessFillMutable() { return getFillMutable(); }\n">;

def IntentGPU_ParameterOp : IntentGPU_Op<"parameter", [Pure]> {
  let summary = "Read a kernel-owned positive index parameter";
  let arguments = (ins IntentGPU_ParameterRefAttr:$reference);
  let results = (outs Index:$result);
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    ::intent::gpu::ParameterAttr getDeclaration();
  }];
}

def IntentGPU_PhysicalExprOp : IntentGPU_Op<"physical_expr", [Pure]> {
  let arguments = (ins IntentGPU_PhysicalExpr:$expression);
  let results = (outs Index:$result);
  let hasVerifier = 1;
}

def IntentGPU_ProgramIdOp : IntentGPU_Op<"program_id", [Pure]> {
  let arguments = (ins I64Attr:$axis);
  let results = (outs Index:$result);
  let hasVerifier = 1;
}

def IntentGPU_WorksetCoordinateOp : IntentGPU_Op<"workset_coordinate", [Pure]> {
  let summary = "Scalar logical coordinate owned by one physical program";
  let arguments = (ins Index:$coordinate,
                       Index:$step,
                       I64Attr:$source_id,
                       I64Attr:$source_axis,
                       I64Attr:$source_rank,
                       I64Attr:$dimension_id);
  let results = (outs Index:$result);
  let hasVerifier = 1;
}

def IntentGPU_DelinearizeOp : IntentGPU_Op<"delinearize", [Pure]> {
  let summary = "Pure row-major coordinate decoding";
  let arguments = (ins Index:$linear,
                       Variadic<Index>:$extents);
  let results = (outs Variadic<Index>:$coordinates);
  let hasVerifier = 1;
}

def IntentGPU_ExecutionGroupOp : IntentGPU_Op<"execution_group", [
    RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"ExecutionGroupYieldOp">,
    DeclareOpInterfaceMethods<RegionBranchOpInterface>]> {
  let summary = "One physical workset and its program-coordinate domain";
  let arguments = (ins Index:$linear, Variadic<Index>:$extents,
                       ArrayAttr:$launch_extents,
                       DenseI64ArrayAttr:$coordinate_roles,
                       I64Attr:$group_id,
                       IntentGPU_PhysicalExpr:$segment_offset,
                       IntentGPU_PhysicalExpr:$segment_length);
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    ::mlir::Block::BlockArgListType getCoordinates() {
      return getBody().front().getArguments();
    }
  }];
}

def IntentGPU_ExecutionGroupYieldOp : IntentGPU_Op<"execution_group_yield", [
    Pure, Terminator, HasParent<"ExecutionGroupOp">, ReturnLike]> {
  let arguments = (ins);
}

def IntentGPU_DimOp : IntentGPU_Op<"dim", [Pure]> {
  let arguments = (ins IntentGPU_ViewValue:$view, I64Attr:$axis);
  let results = (outs Index:$result);
  let hasVerifier = 1;
}

def IntentGPU_ViewOverlapOp : IntentGPU_Op<"view_overlap", [Pure]> {
  let summary = "Launch-time overlap of external view byte-span bounds";
  let arguments = (ins IntentGPU_ViewValue:$lhs, IntentGPU_ViewValue:$rhs);
  let results = (outs I1:$result);
  let hasVerifier = 1;
}

def IntentGPU_RangeOp : IntentGPU_Op<"range", [Pure]> {
  let arguments = (ins Index:$start, Index:$stop, Index:$step);
  let results = (outs IntentGPU_RangeValue:$result);
  let hasVerifier = 1;
}

def IntentGPU_RangeBoundOp : IntentGPU_Op<"range_bound", [Pure]> {
  let arguments = (ins IntentGPU_RangeValue:$range, I64Attr:$bound);
  let results = (outs Index:$result);
  let hasVerifier = 1;
}

def IntentGPU_MakeRangeOp : IntentGPU_Op<"make_range", [Pure]> {
  let arguments = (ins Index:$start, Index:$extent, Index:$step,
                       Index:$logical_start, Index:$logical_stop,
                       I64Attr:$source_id,
                       I64Attr:$source_axis,
                       DefaultValuedAttr<BoolAttr, "false">:$derived);
  let results = (outs IntentGPU_FragmentValue:$result);
  let hasVerifier = 1;
}

def IntentGPU_SplatOp : IntentGPU_Op<"splat", [Pure, DeclareOpInterfaceMethods<IntentGPU_FragmentOpInterface>]> {
  let arguments = (ins IntentGPU_ScalarValue:$value);
  let results = (outs IntentGPU_FragmentValue:$result);
  let hasVerifier = 1;
}

def IntentGPU_BroadcastOp : IntentGPU_Op<"broadcast", [Pure, DeclareOpInterfaceMethods<IntentGPU_FragmentOpInterface>]> {
  let arguments = (ins IntentGPU_DataValue:$value);
  let results = (outs IntentGPU_FragmentValue:$result);
  let hasVerifier = 1;
  let hasFolder = 1;
}

def IntentGPU_UnaryOp : IntentGPU_Op<"unary", [Pure, Elementwise, Scalarizable, DeclareOpInterfaceMethods<IntentGPU_FragmentOpInterface>]> {
  let arguments = (ins IntentGPU_DataValue:$input,
                       Intent_UnaryOperatorAttr:$operator_kind,
                       DefaultValuedAttr<BoolAttr, "false">:$approximate,
                       DefaultValuedAttr<BoolAttr, "false">:$flush_to_zero);
  let results = (outs IntentGPU_DataValue:$result);
  let hasVerifier = 1;
}

def IntentGPU_BinaryOp : IntentGPU_Op<"binary", [Pure, Elementwise, Scalarizable, DeclareOpInterfaceMethods<IntentGPU_FragmentOpInterface>]> {
  let description = [{
    Elementwise arithmetic with explicit numerical permissions. For f32
    add/subtract/multiply, strict_rounding requires a separate nearest-even
    result with gradual underflow and forbids contraction across this result.
    This is an execution requirement, independent of approximate and FTZ modes.
  }];
  let arguments = (ins IntentGPU_DataValue:$lhs, IntentGPU_DataValue:$rhs,
                       Intent_BinaryOperatorAttr:$operator_kind,
                       DefaultValuedAttr<BoolAttr, "false">:$approximate,
                       DefaultValuedAttr<BoolAttr, "false">:$flush_to_zero,
                       DefaultValuedAttr<BoolAttr, "false">:$strict_rounding);
  let results = (outs IntentGPU_DataValue:$result);
  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def IntentGPU_CompareOp : IntentGPU_Op<"compare", [Pure, Elementwise, Scalarizable, DeclareOpInterfaceMethods<IntentGPU_FragmentOpInterface>]> {
  let arguments = (ins IntentGPU_DataValue:$lhs, IntentGPU_DataValue:$rhs,
                       Intent_ComparePredicateAttr:$predicate);
  let results = (outs IntentGPU_DataValue:$result);
  let hasVerifier = 1;
}

def IntentGPU_SelectOp : IntentGPU_Op<"select", [Pure, Elementwise, Scalarizable, DeclareOpInterfaceMethods<IntentGPU_FragmentOpInterface>]> {
  let arguments = (ins IntentGPU_DataValue:$condition,
                       IntentGPU_DataValue:$true_value,
                       IntentGPU_DataValue:$false_value);
  let results = (outs IntentGPU_DataValue:$result);
  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def IntentGPU_CastOp : IntentGPU_Op<"cast", [Pure, Elementwise, Scalarizable, DeclareOpInterfaceMethods<IntentGPU_FragmentOpInterface>]> {
  let arguments = (ins IntentGPU_DataValue:$value);
  let results = (outs IntentGPU_DataValue:$result);
  let hasVerifier = 1;
  let hasFolder = 1;
}

def IntentGPU_BitcastOp : IntentGPU_Op<"bitcast", [Pure, Elementwise, Scalarizable, DeclareOpInterfaceMethods<IntentGPU_FragmentOpInterface>]> {
  let arguments = (ins IntentGPU_DataValue:$value);
  let results = (outs IntentGPU_DataValue:$result);
  let hasVerifier = 1;
  let hasFolder = 1;
}

def IntentGPU_ReshapeOp : IntentGPU_Op<"reshape", [Pure, DeclareOpInterfaceMethods<IntentGPU_FragmentOpInterface>]> {
  let arguments = (ins IntentGPU_DataValue:$value,
                       ArrayAttr:$reassociation);
  let results = (outs IntentGPU_DataValue:$result);
  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def IntentGPU_TransposeOp : IntentGPU_Op<"transpose", [Pure, DeclareOpInterfaceMethods<IntentGPU_FragmentOpInterface>]> {
  let arguments = (ins IntentGPU_FragmentValue:$value,
                       DenseI64ArrayAttr:$permutation);
  let results = (outs IntentGPU_FragmentValue:$result);
  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def IntentGPU_JoinOp : IntentGPU_Op<"join", [
    Pure, DeclareOpInterfaceMethods<IntentGPU_FragmentOpInterface>]> {
  let arguments = (ins IntentGPU_FragmentValue:$lhs,
                       IntentGPU_FragmentValue:$rhs,
                       I64Attr:$axis);
  let results = (outs IntentGPU_FragmentValue:$result);
  let hasVerifier = 1;
}

def IntentGPU_MakeRecordOp : IntentGPU_Op<"make_record", [Pure]> {
  let arguments = (ins Variadic<IntentGPU_DataValue>:$fields);
  let results = (outs IntentGPU_RecordValue:$result);
  let hasVerifier = 1;
  let hasFolder = 1;
}

def IntentGPU_ExtractOp : IntentGPU_Op<"extract", [Pure]> {
  let arguments = (ins IntentGPU_RecordValue:$record, I64Attr:$field);
  let results = (outs IntentGPU_DataValue:$result);
  let hasVerifier = 1;
  let hasFolder = 1;
}

def IntentGPU_LoadOp : IntentGPU_ReadOp<"load", "Load", "getResourceMutable()", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins IntentGPU_ResourceValue:$resource,
                       Variadic<IntentGPU_DataValue>:$coordinates,
                       Optional<IntentGPU_DataValue>:$valid,
                       Optional<IntentGPU_DataValue>:$fill,
                       DenseI64ArrayAttr:$source_axes);
  let results = (outs IntentGPU_DataValue:$result);
}

def IntentGPU_GatherOp : IntentGPU_ReadOp<"gather", "Gather", "getSourceMutable()", [Pure, AttrSizedOperandSegments]> {
  let arguments = (ins IntentGPU_DataValue:$source,
                       Variadic<IntentGPU_DataValue>:$coordinates,
                       Optional<IntentGPU_DataValue>:$valid,
                       Optional<IntentGPU_DataValue>:$fill,
                       DenseI64ArrayAttr:$source_axes);
  let results = (outs IntentGPU_DataValue:$result);
}

def IntentGPU_AssumeInBoundsOp : IntentGPU_Op<"assume_in_bounds", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins IntentGPU_DataValue:$index,
                       IntentGPU_ResourceValue:$resource,
                       I64Attr:$axis);
  let hasVerifier = 1;
}

def IntentGPU_StoreOp : IntentGPU_AccessOp<"store", "Store", "getResourceMutable()",
    "::mlir::MutableOperandRange(getValueMutable())", "getValue().getType()", 1, [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins IntentGPU_ResourceValue:$resource,
                       Variadic<IntentGPU_DataValue>:$coordinates,
                       IntentGPU_DataValue:$value,
                       Optional<IntentGPU_DataValue>:$valid,
                       DenseI64ArrayAttr:$source_axes);
  let hasVerifier = 1;
}

def IntentGPU_ContractOp : IntentGPU_Op<"contract", [Pure]> {
  let arguments = (ins IntentGPU_FragmentValue:$lhs,
                       IntentGPU_FragmentValue:$rhs,
                       IntentGPU_FragmentValue:$accumulator,
                       DenseI64ArrayAttr:$lhs_reduction_axes,
                       DenseI64ArrayAttr:$rhs_reduction_axes,
                       DenseI64ArrayAttr:$lhs_batch_axes,
                       DenseI64ArrayAttr:$rhs_batch_axes);
  let results = (outs IntentGPU_FragmentValue:$result);
  let hasVerifier = 1;
}

def IntentGPU_ReduceOp : IntentGPU_Op<"reduce", [Pure, IsolatedFromAbove, AttrSizedOperandSegments, Intent_StructuredOpInterface, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Variadic<IntentGPU_DataValue>:$sources,
                       Variadic<IntentGPU_DataValue>:$identities,
                       Variadic<IntentGPU_DataValue>:$captures,
                       DenseI64ArrayAttr:$axes);
  let results = (outs Variadic<IntentGPU_DataValue>:$results);
  let regions = (region SizedRegion<1>:$combine);
  let hasVerifier = 1;
  let assemblyFormat = [{
    `sources` `(` $sources `:` type($sources) `)`
    `identities` `(` $identities `:` type($identities) `)`
    `captures` `(` ($captures^ `:` type($captures))? `)`
    `axes` $axes
    `combine` $combine
    `->` `(` type($results) `)`
    attr-dict
  }];
  let extraClassDeclaration = [{
    ::intent::StructuredOpKind getStructuredKind() {
      return ::intent::StructuredOpKind::Reduce;
    }
    ::llvm::SmallVector<int64_t> getIterationAxes() {
      return ::llvm::to_vector(getAxes());
    }
  }];
}

def IntentGPU_ScanOp : IntentGPU_Op<"scan", [Pure, IsolatedFromAbove, AttrSizedOperandSegments, Intent_StructuredOpInterface, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Variadic<IntentGPU_DataValue>:$sources,
                       Variadic<IntentGPU_DataValue>:$identities,
                       Variadic<IntentGPU_DataValue>:$captures,
                       I64Attr:$axis,
                       BoolAttr:$inclusive,
                       BoolAttr:$reverse);
  let results = (outs Variadic<IntentGPU_DataValue>:$results);
  let regions = (region SizedRegion<1>:$combine);
  let hasVerifier = 1;
  let assemblyFormat = [{
    `sources` `(` $sources `:` type($sources) `)`
    `identities` `(` $identities `:` type($identities) `)`
    `captures` `(` ($captures^ `:` type($captures))? `)`
    `axis` $axis
    `inclusive` $inclusive `reverse` $reverse
    `combine` $combine
    `->` `(` type($results) `)`
    attr-dict
  }];
  let extraClassDeclaration = [{
    ::intent::StructuredOpKind getStructuredKind() {
      return ::intent::StructuredOpKind::Scan;
    }
    ::llvm::SmallVector<int64_t> getIterationAxes() {
      return {static_cast<int64_t>(getAxis())};
    }
  }];
}

def IntentGPU_RegionFoldOp : IntentGPU_Op<"region_fold", [Pure, IsolatedFromAbove, AttrSizedOperandSegments, Intent_StructuredOpInterface]> {
  let arguments = (ins Variadic<IntentGPU_DataValue>:$sources,
                       Variadic<IntentGPU_DataValue>:$identities,
                       Variadic<IntentGPU_DataValue>:$captures,
                       I64Attr:$axis,
                       IntentGPU_ParameterRefAttr:$segment);
  let results = (outs Variadic<IntentGPU_DataValue>:$results);
  let regions = (region SizedRegion<1>:$summarize,
                         SizedRegion<1>:$combine);
  let hasVerifier = 1;
  let assemblyFormat = [{
    `sources` `(` $sources `:` type($sources) `)`
    `identities` `(` $identities `:` type($identities) `)`
    `captures` `(` ($captures^ `:` type($captures))? `)`
    `axis` $axis
    `segment` $segment
    `summarize` $summarize
    `combine` $combine
    `->` `(` type($results) `)`
    attr-dict
  }];
  let extraClassDeclaration = [{
    ::intent::StructuredOpKind getStructuredKind() {
      return ::intent::StructuredOpKind::RegionFold;
    }
    ::llvm::SmallVector<int64_t> getIterationAxes() {
      return {static_cast<int64_t>(getAxis())};
    }
    ::mlir::Region *getSummarizeRegion() { return &getSummarize(); }
  }];
}

def IntentGPU_RegionScanOp : IntentGPU_Op<"region_scan", [Pure, IsolatedFromAbove, AttrSizedOperandSegments, Intent_StructuredOpInterface, AttrSizedResultSegments]> {
  let arguments = (ins Variadic<IntentGPU_DataValue>:$sources,
                       Variadic<IntentGPU_DataValue>:$identities,
                       Variadic<IntentGPU_DataValue>:$initial_states,
                       Variadic<IntentGPU_DataValue>:$captures,
                       I64Attr:$axis,
                       IntentGPU_ParameterRefAttr:$segment);
  let results = (outs Variadic<IntentGPU_DataValue>:$emitted_results,
                      Variadic<IntentGPU_DataValue>:$final_states);
  let regions = (region SizedRegion<1>:$summarize,
                         SizedRegion<1>:$combine,
                         SizedRegion<1>:$apply,
                         SizedRegion<1>:$emit);
  let hasVerifier = 1;
  let assemblyFormat = [{
    `sources` `(` $sources `:` type($sources) `)`
    `identities` `(` $identities `:` type($identities) `)`
    `initial_states` `(` $initial_states `:` type($initial_states) `)`
    `captures` `(` ($captures^ `:` type($captures))? `)`
    `axis` $axis
    `segment` $segment
    `summarize` $summarize
    `combine` $combine
    `apply` $apply
    `emit` $emit
    `->` `(` type($emitted_results) `)` `final_states` `(` type($final_states) `)`
    attr-dict
  }];
  let extraClassDeclaration = [{
    ::intent::StructuredOpKind getStructuredKind() {
      return ::intent::StructuredOpKind::RegionScan;
    }
    ::llvm::SmallVector<int64_t> getIterationAxes() {
      return {static_cast<int64_t>(getAxis())};
    }
    ::mlir::Region *getSummarizeRegion() { return &getSummarize(); }
    ::mlir::Region *getApplyRegion() { return &getApply(); }
    ::mlir::Region *getEmitRegion() { return &getEmit(); }
  }];
}

def IntentGPU_ScaledContractOp : IntentGPU_Op<"scaled_contract", [Pure]> {
  let arguments = (ins IntentGPU_FragmentValue:$lhs,
                       IntentGPU_FragmentValue:$lhs_scale,
                       IntentGPU_FragmentValue:$rhs,
                       IntentGPU_FragmentValue:$rhs_scale,
                       IntentGPU_FragmentValue:$accumulator,
                       DenseI64ArrayAttr:$lhs_reduction_axes,
                       DenseI64ArrayAttr:$rhs_reduction_axes,
                       DenseI64ArrayAttr:$lhs_batch_axes,
                       DenseI64ArrayAttr:$rhs_batch_axes,
                       Intent_ScaledFormatAttr:$lhs_format,
                       Intent_ScaledFormatAttr:$rhs_format,
                       I64Attr:$lhs_group_size, I64Attr:$rhs_group_size);
  let results = (outs IntentGPU_FragmentValue:$result);
  let hasVerifier = 1;
}

def IntentGPU_SparseContractOp : IntentGPU_Op<"sparse_contract", [Pure]> {
  let arguments = (ins IntentGPU_FragmentValue:$compressed,
                       IntentGPU_DataValue:$metadata,
                       IntentGPU_FragmentValue:$rhs,
                       Index:$logical_extent,
                       IntentGPU_FragmentValue:$accumulator,
                       Intent_SparseFormat:$format,
                       DenseI64ArrayAttr:$lhs_reduction_axes,
                       DenseI64ArrayAttr:$rhs_reduction_axes,
                       DenseI64ArrayAttr:$lhs_batch_axes,
                       DenseI64ArrayAttr:$rhs_batch_axes);
  let results = (outs IntentGPU_FragmentValue:$result);
  let hasVerifier = 1;
}

def IntentGPU_HistogramOp : IntentGPU_Op<"histogram", [Pure]> {
  let arguments = (ins IntentGPU_FragmentValue:$values, Index:$bins,
                       IntentGPU_FragmentValue:$valid);
  let results = (outs IntentGPU_FragmentValue:$result);
  let hasVerifier = 1;
}

def IntentGPU_ScatterReduceOp : IntentGPU_AccessOp<"scatter_reduce", "ScatterReduce", "getResourceMutable()",
    "::mlir::MutableOperandRange(getValueMutable())", "getValue().getType()", 1, [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins IntentGPU_ResourceValue:$resource,
                       Variadic<IntentGPU_DataValue>:$coordinates,
                       IntentGPU_DataValue:$value,
                       Optional<IntentGPU_DataValue>:$valid,
                       DenseI64ArrayAttr:$source_axes,
                       IntentGPU_AtomicSharingDomainAttr:$sharing);
  let regions = (region SizedRegion<1>:$combine);
  let hasVerifier = 1;
}

def IntentGPU_AtomicLoadOp : IntentGPU_AccessOp<"atomic_load", "AtomicLoad", "getResourceMutable()",
    "::mlir::MutableOperandRange(getOperation(), 0, 0)", "getResult().getType()", 0, [
    // Write conservatively represents atomic synchronization/observation, not
    // a payload write; ordinary load CSE must not merge atomic observations.
    AttrSizedOperandSegments, MemoryEffects<[MemRead, MemWrite]>
  ]> {
  let arguments = (ins IntentGPU_ResourceValue:$resource,
                       Variadic<IntentGPU_DataValue>:$coordinates,
                       Optional<IntentGPU_DataValue>:$valid,
                       Intent_AtomicOrderingAttr:$ordering,
                       IntentGPU_AtomicSharingDomainAttr:$sharing,
                       DenseI64ArrayAttr:$source_axes);
  let results = (outs IntentGPU_DataValue:$result);
  let hasVerifier = 1;
}

def IntentGPU_AtomicStoreOp : IntentGPU_AccessOp<"atomic_store", "AtomicStore", "getResourceMutable()",
    "::mlir::MutableOperandRange(getValueMutable())", "getValue().getType()", 1, [
    AttrSizedOperandSegments, MemoryEffects<[MemWrite]>
  ]> {
  let arguments = (ins IntentGPU_ResourceValue:$resource,
                       Variadic<IntentGPU_DataValue>:$coordinates,
                       IntentGPU_DataValue:$value,
                       Optional<IntentGPU_DataValue>:$valid,
                       Intent_AtomicOrderingAttr:$ordering,
                       IntentGPU_AtomicSharingDomainAttr:$sharing,
                       DenseI64ArrayAttr:$source_axes);
  let hasVerifier = 1;
}

def IntentGPU_AtomicRMWOp : IntentGPU_AccessOp<"atomic_rmw", "AtomicRMW", "getResourceMutable()",
    "::mlir::MutableOperandRange(getValueMutable())", "getValue().getType()", 1, [
    AttrSizedOperandSegments, MemoryEffects<[MemRead, MemWrite]>
  ]> {
  let arguments = (ins IntentGPU_ResourceValue:$resource,
                       Variadic<IntentGPU_DataValue>:$coordinates,
                       IntentGPU_DataValue:$value,
                       Optional<IntentGPU_DataValue>:$valid,
                       Intent_AtomicRMWKindAttr:$kind,
                       Intent_AtomicOrderingAttr:$ordering,
                       IntentGPU_AtomicSharingDomainAttr:$sharing,
                       DenseI64ArrayAttr:$source_axes);
  let results = (outs IntentGPU_DataValue:$result);
  let hasVerifier = 1;
}

def IntentGPU_AtomicCompareExchangeOp : IntentGPU_AccessOp<"atomic_compare_exchange", "AtomicCompareExchange", "getResourceMutable()",
    "::mlir::MutableOperandRange(getOperation(), getExpectedMutable().getOperandNumber(), 2)", "getExpected().getType()", 2, [
    AttrSizedOperandSegments, MemoryEffects<[MemRead, MemWrite]>
  ]> {
  let arguments = (ins IntentGPU_ResourceValue:$resource,
                       Variadic<IntentGPU_DataValue>:$coordinates,
                       IntentGPU_DataValue:$expected,
                       IntentGPU_DataValue:$desired,
                       Optional<IntentGPU_DataValue>:$valid,
                       Intent_AtomicOrderingAttr:$ordering,
                       IntentGPU_AtomicSharingDomainAttr:$sharing,
                       DenseI64ArrayAttr:$source_axes);
  let results = (outs IntentGPU_RecordValue:$result);
  let hasVerifier = 1;
}

def IntentGPU_RandomBitsOp : IntentGPU_Op<"random_bits", [
    Pure, DeclareOpInterfaceMethods<IntentGPU_FragmentOpInterface>]> {
  let arguments = (ins I64:$seed, IntentGPU_DataValue:$counter);
  let results = (outs IntentGPU_DataValue:$result);
  let hasVerifier = 1;
}

def IntentGPU_YieldOp : IntentGPU_Op<"yield", [Pure, Terminator]> {
  let arguments = (ins Variadic<IntentGPU_DataValue>:$values);
}

def IntentGPU_BufferOp : IntentGPU_Op<"buffer", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins Optional<IntentGPU_DataValue>:$initial_value);
  let results = (outs IntentGPU_BufferValue:$result);
  let hasVerifier = 1;
}

#endif
