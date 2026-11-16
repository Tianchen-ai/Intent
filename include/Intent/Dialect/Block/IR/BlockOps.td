#ifndef INTENT_DIALECT_Block_IR_BlockOPS_TD
#define INTENT_DIALECT_Block_IR_BlockOPS_TD

include "Intent/Dialect/Block/IR/BlockDialect.td"
include "Intent/Dialect/Block/IR/BlockAttrs.td"
include "Intent/Dialect/Block/IR/BlockTypes.td"
include "Intent/Dialect/Block/IR/AccessOpInterface.td"
include "Intent/Dialect/Block/IR/FragmentOpInterface.td"
include "Intent/Dialect/Intent/IR/IntentAttrs.td"
include "Intent/Interfaces/StructuredOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class IntentBlock_Op<string mnemonic, list<Trait> traits = []> :
    Op<IntentBlock_Dialect, mnemonic, traits>;

class IntentBlock_AccessOp<string mnemonic, string kind, string resource,
                         string payloads, string valueType, int payloadCount,
                         list<Trait> traits = [], string extra = ""> :
    IntentBlock_Op<mnemonic, !listconcat(traits, [IntentBlock_AccessOpInterface])> {
  let extraClassDeclaration =
    "::intent::block::AccessKind getAccessKind() { return ::intent::block::AccessKind::" # kind # "; }\n" #
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

class IntentBlock_ReadOp<string mnemonic, string kind, string resource,
                       list<Trait> traits = []> :
    IntentBlock_AccessOp<mnemonic, kind, resource,
      "::mlir::MutableOperandRange(getOperation(), 0, 0)",
      "getResult().getType()", 0, traits,
    "::mlir::Value getAccessFill() { return getFill(); }\n" #
    "std::optional<::mlir::MutableOperandRange> getAccessFillMutable() { return getFillMutable(); }\n">;

def IntentBlock_ParameterOp : IntentBlock_Op<"parameter", [Pure]> {
  let summary = "Read a kernel-owned positive index parameter";
  let arguments = (ins IntentBlock_ParameterRefAttr:$reference);
  let results = (outs Index:$result);
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    ::intent::block::ParameterAttr getDeclaration();
  }];
}

def IntentBlock_PhysicalExprOp : IntentBlock_Op<"physical_expr", [Pure]> {
  let arguments = (ins IntentBlock_PhysicalExpr:$expression);
  let results = (outs Index:$result);
  let hasVerifier = 1;
}

def IntentBlock_ProgramIdOp : IntentBlock_Op<"program_id", [Pure]> {
  let arguments = (ins I64Attr:$axis);
  let results = (outs Index:$result);
  let hasVerifier = 1;
}

def IntentBlock_WorksetCoordinateOp : IntentBlock_Op<"workset_coordinate", [Pure]> {
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

def IntentBlock_DelinearizeOp : IntentBlock_Op<"delinearize", [Pure]> {
  let summary = "Pure row-major coordinate decoding";
  let arguments = (ins Index:$linear,
                       Variadic<Index>:$extents);
  let results = (outs Variadic<Index>:$coordinates);
  let hasVerifier = 1;
}

def IntentBlock_ExecutionGroupOp : IntentBlock_Op<"execution_group", [
    RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"ExecutionGroupYieldOp">,
    DeclareOpInterfaceMethods<RegionBranchOpInterface>]> {
  let summary = "One physical workset and its program-coordinate domain";
  let arguments = (ins Index:$linear, Variadic<Index>:$extents,
                       ArrayAttr:$launch_extents,
                       DenseI64ArrayAttr:$coordinate_roles,
                       I64Attr:$group_id,
                       IntentBlock_PhysicalExpr:$segment_offset,
                       IntentBlock_PhysicalExpr:$segment_length);
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    ::mlir::Block::BlockArgListType getCoordinates() {
      return getBody().front().getArguments();
    }
  }];
}

def IntentBlock_ExecutionGroupYieldOp : IntentBlock_Op<"execution_group_yield", [
    Pure, Terminator, HasParent<"ExecutionGroupOp">, ReturnLike]> {
  let arguments = (ins);
}

def IntentBlock_DimOp : IntentBlock_Op<"dim", [Pure]> {
  let arguments = (ins IntentBlock_ViewValue:$view, I64Attr:$axis);
  let results = (outs Index:$result);
  let hasVerifier = 1;
}

def IntentBlock_ViewOverlapOp : IntentBlock_Op<"view_overlap", [Pure]> {
  let summary = "Launch-time overlap of external view byte-span bounds";
  let arguments = (ins IntentBlock_ViewValue:$lhs, IntentBlock_ViewValue:$rhs);
  let results = (outs I1:$result);
  let hasVerifier = 1;
}

def IntentBlock_RangeOp : IntentBlock_Op<"range", [Pure]> {
  let arguments = (ins Index:$start, Index:$stop, Index:$step);
  let results = (outs IntentBlock_RangeValue:$result);
  let hasVerifier = 1;
}

def IntentBlock_RangeBoundOp : IntentBlock_Op<"range_bound", [Pure]> {
  let arguments = (ins IntentBlock_RangeValue:$range, I64Attr:$bound);
  let results = (outs Index:$result);
  let hasVerifier = 1;
}

def IntentBlock_MakeRangeOp : IntentBlock_Op<"make_range", [Pure]> {
  let arguments = (ins Index:$start, Index:$extent, Index:$step,
                       Index:$logical_start, Index:$logical_stop,
                       I64Attr:$source_id,
                       I64Attr:$source_axis,
                       DefaultValuedAttr<BoolAttr, "false">:$derived);
  let results = (outs IntentBlock_FragmentValue:$result);
  let hasVerifier = 1;
}

def IntentBlock_SplatOp : IntentBlock_Op<"splat", [Pure, DeclareOpInterfaceMethods<IntentBlock_FragmentOpInterface>]> {
  let arguments = (ins IntentBlock_ScalarValue:$value);
  let results = (outs IntentBlock_FragmentValue:$result);
  let hasVerifier = 1;
}

def IntentBlock_BroadcastOp : IntentBlock_Op<"broadcast", [Pure, DeclareOpInterfaceMethods<IntentBlock_FragmentOpInterface>]> {
  let arguments = (ins IntentBlock_DataValue:$value);
  let results = (outs IntentBlock_FragmentValue:$result);
  let hasVerifier = 1;
  let hasFolder = 1;
}

def IntentBlock_UnaryOp : IntentBlock_Op<"unary", [Pure, Elementwise, Scalarizable, DeclareOpInterfaceMethods<IntentBlock_FragmentOpInterface>]> {
  let arguments = (ins IntentBlock_DataValue:$input,
                       Intent_UnaryOperatorAttr:$operator_kind,
                       DefaultValuedAttr<BoolAttr, "false">:$approximate,
                       DefaultValuedAttr<BoolAttr, "false">:$flush_to_zero);
  let results = (outs IntentBlock_DataValue:$result);
  let hasVerifier = 1;
}

def IntentBlock_BinaryOp : IntentBlock_Op<"binary", [Pure, Elementwise, Scalarizable, DeclareOpInterfaceMethods<IntentBlock_FragmentOpInterface>]> {
  let description = [{
    Elementwise arithmetic with explicit numerical permissions. For f32
    add/subtract/multiply, strict_rounding requires a separate nearest-even
    result with gradual underflow and forbids contraction across this result.
    This is an execution requirement, independent of approximate and FTZ modes.
  }];
  let arguments = (ins IntentBlock_DataValue:$lhs, IntentBlock_DataValue:$rhs,
                       Intent_BinaryOperatorAttr:$operator_kind,
                       DefaultValuedAttr<BoolAttr, "false">:$approximate,
                       DefaultValuedAttr<BoolAttr, "false">:$flush_to_zero,
                       DefaultValuedAttr<BoolAttr, "false">:$strict_rounding);
  let results = (outs IntentBlock_DataValue:$result);
  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def IntentBlock_CompareOp : IntentBlock_Op<"compare", [Pure, Elementwise, Scalarizable, DeclareOpInterfaceMethods<IntentBlock_FragmentOpInterface>]> {
  let arguments = (ins IntentBlock_DataValue:$lhs, IntentBlock_DataValue:$rhs,
                       Intent_ComparePredicateAttr:$predicate);
  let results = (outs IntentBlock_DataValue:$result);
  let hasVerifier = 1;
}

def IntentBlock_SelectOp : IntentBlock_Op<"select", [Pure, Elementwise, Scalarizable, DeclareOpInterfaceMethods<IntentBlock_FragmentOpInterface>]> {
  let arguments = (ins IntentBlock_DataValue:$condition,
                       IntentBlock_DataValue:$true_value,
                       IntentBlock_DataValue:$false_value);
  let results = (outs IntentBlock_DataValue:$result);
  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def IntentBlock_CastOp : IntentBlock_Op<"cast", [Pure, Elementwise, Scalarizable, DeclareOpInterfaceMethods<IntentBlock_FragmentOpInterface>]> {
  let arguments = (ins IntentBlock_DataValue:$value);
  let results = (outs IntentBlock_DataValue:$result);
  let hasVerifier = 1;
  let hasFolder = 1;
}

def IntentBlock_BitcastOp : IntentBlock_Op<"bitcast", [Pure, Elementwise, Scalarizable, DeclareOpInterfaceMethods<IntentBlock_FragmentOpInterface>]> {
  let arguments = (ins IntentBlock_DataValue:$value);
  let results = (outs IntentBlock_DataValue:$result);
  let hasVerifier = 1;
  let hasFolder = 1;
}

def IntentBlock_ReshapeOp : IntentBlock_Op<"reshape", [Pure, DeclareOpInterfaceMethods<IntentBlock_FragmentOpInterface>]> {
  let arguments = (ins IntentBlock_DataValue:$value,
                       ArrayAttr:$reassociation);
  let results = (outs IntentBlock_DataValue:$result);
  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def IntentBlock_TransposeOp : IntentBlock_Op<"transpose", [Pure, DeclareOpInterfaceMethods<IntentBlock_FragmentOpInterface>]> {
  let arguments = (ins IntentBlock_FragmentValue:$value,
                       DenseI64ArrayAttr:$permutation);
  let results = (outs IntentBlock_FragmentValue:$result);
  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def IntentBlock_JoinOp : IntentBlock_Op<"join", [
    Pure, DeclareOpInterfaceMethods<IntentBlock_FragmentOpInterface>]> {
  let arguments = (ins IntentBlock_FragmentValue:$lhs,
                       IntentBlock_FragmentValue:$rhs,
                       I64Attr:$axis);
  let results = (outs IntentBlock_FragmentValue:$result);
  let hasVerifier = 1;
}

def IntentBlock_MakeRecordOp : IntentBlock_Op<"make_record", [Pure]> {
  let arguments = (ins Variadic<IntentBlock_DataValue>:$fields);
  let results = (outs IntentBlock_RecordValue:$result);
  let hasVerifier = 1;
  let hasFolder = 1;
}

def IntentBlock_ExtractOp : IntentBlock_Op<"extract", [Pure]> {
  let arguments = (ins IntentBlock_RecordValue:$record, I64Attr:$field);
  let results = (outs IntentBlock_DataValue:$result);
  let hasVerifier = 1;
  let hasFolder = 1;
}

def IntentBlock_LoadOp : IntentBlock_ReadOp<"load", "Load", "getResourceMutable()", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins IntentBlock_ResourceValue:$resource,
                       Variadic<IntentBlock_DataValue>:$coordinates,
                       Optional<IntentBlock_DataValue>:$valid,
                       Optional<IntentBlock_DataValue>:$fill,
                       DenseI64ArrayAttr:$source_axes);
  let results = (outs IntentBlock_DataValue:$result);
}

def IntentBlock_GatherOp : IntentBlock_ReadOp<"gather", "Gather", "getSourceMutable()", [Pure, AttrSizedOperandSegments]> {
  let arguments = (ins IntentBlock_DataValue:$source,
                       Variadic<IntentBlock_DataValue>:$coordinates,
                       Optional<IntentBlock_DataValue>:$valid,
                       Optional<IntentBlock_DataValue>:$fill,
                       DenseI64ArrayAttr:$source_axes);
  let results = (outs IntentBlock_DataValue:$result);
}

def IntentBlock_AssumeInBoundsOp : IntentBlock_Op<"assume_in_bounds", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins IntentBlock_DataValue:$index,
                       IntentBlock_ResourceValue:$resource,
                       I64Attr:$axis,
                       Optional<IntentBlock_DataValue>:$valid);
  let builders = [OpBuilder<(ins "::mlir::Value":$index,
      "::mlir::Value":$resource, "uint64_t":$axis), [{
    build($_builder, $_state, index, resource, axis, ::mlir::Value{});
  }]>];
  let hasVerifier = 1;
}

def IntentBlock_StoreOp : IntentBlock_AccessOp<"store", "Store", "getResourceMutable()",
    "::mlir::MutableOperandRange(getValueMutable())", "getValue().getType()", 1, [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins IntentBlock_ResourceValue:$resource,
                       Variadic<IntentBlock_DataValue>:$coordinates,
                       IntentBlock_DataValue:$value,
                       Optional<IntentBlock_DataValue>:$valid,
                       DenseI64ArrayAttr:$source_axes);
  let hasVerifier = 1;
}

def IntentBlock_ContractOp : IntentBlock_Op<"contract", [Pure]> {
  let arguments = (ins IntentBlock_FragmentValue:$lhs,
                       IntentBlock_FragmentValue:$rhs,
                       IntentBlock_FragmentValue:$accumulator,
                       DenseI64ArrayAttr:$lhs_reduction_axes,
                       DenseI64ArrayAttr:$rhs_reduction_axes,
                       DenseI64ArrayAttr:$lhs_batch_axes,
                       DenseI64ArrayAttr:$rhs_batch_axes);
  let results = (outs IntentBlock_FragmentValue:$result);
  let hasVerifier = 1;
}

def IntentBlock_ReduceOp : IntentBlock_Op<"reduce", [Pure, IsolatedFromAbove, AttrSizedOperandSegments, Intent_StructuredOpInterface, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Variadic<IntentBlock_DataValue>:$sources,
                       Variadic<IntentBlock_DataValue>:$identities,
                       Variadic<IntentBlock_DataValue>:$captures,
                       DenseI64ArrayAttr:$axes);
  let results = (outs Variadic<IntentBlock_DataValue>:$results);
  let regions = (region SizedRegion<1>:$combine);
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
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

def IntentBlock_ScanOp : IntentBlock_Op<"scan", [Pure, IsolatedFromAbove, AttrSizedOperandSegments, Intent_StructuredOpInterface, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Variadic<IntentBlock_DataValue>:$sources,
                       Variadic<IntentBlock_DataValue>:$identities,
                       Variadic<IntentBlock_DataValue>:$captures,
                       I64Attr:$axis,
                       BoolAttr:$inclusive,
                       BoolAttr:$reverse);
  let results = (outs Variadic<IntentBlock_DataValue>:$results);
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

def IntentBlock_RegionFoldOp : IntentBlock_Op<"region_fold", [Pure, IsolatedFromAbove, AttrSizedOperandSegments, Intent_StructuredOpInterface]> {
  let arguments = (ins Variadic<IntentBlock_DataValue>:$sources,
                       Variadic<IntentBlock_DataValue>:$identities,
                       Variadic<IntentBlock_DataValue>:$captures,
                       I64Attr:$axis,
                       IntentBlock_ParameterRefAttr:$segment);
  let results = (outs Variadic<IntentBlock_DataValue>:$results);
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

def IntentBlock_RegionScanOp : IntentBlock_Op<"region_scan", [Pure, IsolatedFromAbove, AttrSizedOperandSegments, Intent_StructuredOpInterface, AttrSizedResultSegments]> {
  let arguments = (ins Variadic<IntentBlock_DataValue>:$sources,
                       Variadic<IntentBlock_DataValue>:$identities,
                       Variadic<IntentBlock_DataValue>:$initial_states,
                       Variadic<IntentBlock_DataValue>:$captures,
                       I64Attr:$axis,
                       IntentBlock_ParameterRefAttr:$segment);
  let results = (outs Variadic<IntentBlock_DataValue>:$emitted_results,
                      Variadic<IntentBlock_DataValue>:$final_states);
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

def IntentBlock_ScaledContractOp : IntentBlock_Op<"scaled_contract", [Pure]> {
  let arguments = (ins IntentBlock_FragmentValue:$lhs,
                       IntentBlock_FragmentValue:$lhs_scale,
                       IntentBlock_FragmentValue:$rhs,
                       IntentBlock_FragmentValue:$rhs_scale,
                       IntentBlock_FragmentValue:$accumulator,
                       DenseI64ArrayAttr:$lhs_reduction_axes,
                       DenseI64ArrayAttr:$rhs_reduction_axes,
                       DenseI64ArrayAttr:$lhs_batch_axes,
                       DenseI64ArrayAttr:$rhs_batch_axes,
                       Intent_ScaledFormatAttr:$lhs_format,
                       Intent_ScaledFormatAttr:$rhs_format,
                       I64Attr:$lhs_group_size, I64Attr:$rhs_group_size);
  let results = (outs IntentBlock_FragmentValue:$result);
  let hasVerifier = 1;
}

def IntentBlock_SparseContractOp : IntentBlock_Op<"sparse_contract", [Pure]> {
  let arguments = (ins IntentBlock_FragmentValue:$compressed,
                       IntentBlock_DataValue:$metadata,
                       IntentBlock_FragmentValue:$rhs,
                       Index:$logical_extent,
                       IntentBlock_FragmentValue:$accumulator,
                       Intent_SparseFormat:$format,
                       DenseI64ArrayAttr:$lhs_reduction_axes,
                       DenseI64ArrayAttr:$rhs_reduction_axes,
                       DenseI64ArrayAttr:$lhs_batch_axes,
                       DenseI64ArrayAttr:$rhs_batch_axes);
  let results = (outs IntentBlock_FragmentValue:$result);
  let hasVerifier = 1;
}

def IntentBlock_HistogramOp : IntentBlock_Op<"histogram", [Pure]> {
  let arguments = (ins IntentBlock_FragmentValue:$values, Index:$bins,
                       IntentBlock_FragmentValue:$valid);
  let results = (outs IntentBlock_FragmentValue:$result);
  let hasVerifier = 1;
}

def IntentBlock_ScatterReduceOp : IntentBlock_AccessOp<"scatter_reduce", "ScatterReduce", "getResourceMutable()",
    "::mlir::MutableOperandRange(getValueMutable())", "getValue().getType()", 1, [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let arguments = (ins IntentBlock_ResourceValue:$resource,
                       Variadic<IntentBlock_DataValue>:$coordinates,
                       IntentBlock_DataValue:$value,
                       Optional<IntentBlock_DataValue>:$valid,
                       DenseI64ArrayAttr:$source_axes,
                       IntentBlock_AtomicSharingDomainAttr:$sharing);
  let regions = (region SizedRegion<1>:$combine);
  let hasVerifier = 1;
}

def IntentBlock_AtomicLoadOp : IntentBlock_AccessOp<"atomic_load", "AtomicLoad", "getResourceMutable()",
    "::mlir::MutableOperandRange(getOperation(), 0, 0)", "getResult().getType()", 0, [
    // Write conservatively represents atomic synchronization/observation, not
    // a payload write; ordinary load CSE must not merge atomic observations.
    AttrSizedOperandSegments, MemoryEffects<[MemRead, MemWrite]>
  ]> {
  let arguments = (ins IntentBlock_ResourceValue:$resource,
                       Variadic<IntentBlock_DataValue>:$coordinates,
                       Optional<IntentBlock_DataValue>:$valid,
                       Intent_AtomicOrderingAttr:$ordering,
                       IntentBlock_AtomicSharingDomainAttr:$sharing,
                       DenseI64ArrayAttr:$source_axes);
  let results = (outs IntentBlock_DataValue:$result);
  let hasVerifier = 1;
}

def IntentBlock_AtomicStoreOp : IntentBlock_AccessOp<"atomic_store", "AtomicStore", "getResourceMutable()",
    "::mlir::MutableOperandRange(getValueMutable())", "getValue().getType()", 1, [
    AttrSizedOperandSegments, MemoryEffects<[MemWrite]>
  ]> {
  let arguments = (ins IntentBlock_ResourceValue:$resource,
                       Variadic<IntentBlock_DataValue>:$coordinates,
                       IntentBlock_DataValue:$value,
                       Optional<IntentBlock_DataValue>:$valid,
                       Intent_AtomicOrderingAttr:$ordering,
                       IntentBlock_AtomicSharingDomainAttr:$sharing,
                       DenseI64ArrayAttr:$source_axes);
  let hasVerifier = 1;
}

def IntentBlock_AtomicRMWOp : IntentBlock_AccessOp<"atomic_rmw", "AtomicRMW", "getResourceMutable()",
    "::mlir::MutableOperandRange(getValueMutable())", "getValue().getType()", 1, [
    AttrSizedOperandSegments, MemoryEffects<[MemRead, MemWrite]>
  ]> {
  let arguments = (ins IntentBlock_ResourceValue:$resource,
                       Variadic<IntentBlock_DataValue>:$coordinates,
                       IntentBlock_DataValue:$value,
                       Optional<IntentBlock_DataValue>:$valid,
                       Intent_AtomicRMWKindAttr:$kind,
                       Intent_AtomicOrderingAttr:$ordering,
                       IntentBlock_AtomicSharingDomainAttr:$sharing,
                       DenseI64ArrayAttr:$source_axes);
  let results = (outs IntentBlock_DataValue:$result);
  let hasVerifier = 1;
}

def IntentBlock_AtomicCompareExchangeOp : IntentBlock_AccessOp<"atomic_compare_exchange", "AtomicCompareExchange", "getResourceMutable()",
    "::mlir::MutableOperandRange(getOperation(), getExpectedMutable().getOperandNumber(), 2)", "getExpected().getType()", 2, [
    AttrSizedOperandSegments, MemoryEffects<[MemRead, MemWrite]>
  ]> {
  let arguments = (ins IntentBlock_ResourceValue:$resource,
                       Variadic<IntentBlock_DataValue>:$coordinates,
                       IntentBlock_DataValue:$expected,
                       IntentBlock_DataValue:$desired,
                       Optional<IntentBlock_DataValue>:$valid,
                       Intent_AtomicOrderingAttr:$ordering,
                       IntentBlock_AtomicSharingDomainAttr:$sharing,
                       DenseI64ArrayAttr:$source_axes);
  let results = (outs IntentBlock_RecordValue:$result);
  let hasVerifier = 1;
}

def IntentBlock_RandomBitsOp : IntentBlock_Op<"random_bits", [
    Pure, DeclareOpInterfaceMethods<IntentBlock_FragmentOpInterface>]> {
  let arguments = (ins I64:$seed, IntentBlock_DataValue:$counter);
  let results = (outs IntentBlock_DataValue:$result);
  let hasVerifier = 1;
}

def IntentBlock_YieldOp : IntentBlock_Op<"yield", [Pure, Terminator]> {
  let arguments = (ins Variadic<IntentBlock_DataValue>:$values);
}

def IntentBlock_BufferOp : IntentBlock_Op<"buffer", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins Optional<IntentBlock_DataValue>:$initial_value);
  let results = (outs IntentBlock_BufferValue:$result);
  let hasVerifier = 1;
}

#endif
