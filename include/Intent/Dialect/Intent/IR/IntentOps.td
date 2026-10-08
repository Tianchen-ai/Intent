#ifndef INTENT_DIALECT_INTENT_IR_INTENTOPS_TD
#define INTENT_DIALECT_INTENT_IR_INTENTOPS_TD

include "Intent/Dialect/Intent/IR/IntentDialect.td"
include "Intent/Dialect/Intent/IR/IntentAttrs.td"
include "Intent/Dialect/Intent/IR/IntentTypes.td"
include "Intent/Dialect/Intent/IR/IndexedAccessOpInterface.td"
include "Intent/Interfaces/StructuredOpInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Intent_Op<string mnemonic, list<Trait> traits = []>
    : Op<Intent_Dialect, mnemonic, traits> {
  let hasVerifier = 1;
}

class Intent_UnaryDataOp<string mnemonic, list<Trait> traits = [Pure]>
    : Intent_Op<mnemonic, traits> {
  let arguments = (ins Intent_DataValue:$input);
  let results = (outs Intent_DataValue:$result);
}

class Intent_BinaryDataOp<string mnemonic, list<Trait> traits = [Pure]>
    : Intent_Op<mnemonic, traits> {
  let arguments = (ins Intent_DataValue:$lhs, Intent_DataValue:$rhs);
  let results = (outs Intent_DataValue:$result);
}

def Intent_ConstantOp : Intent_Op<"constant", [Pure]> {
  let arguments = (ins AnyAttr:$value);
  let results = (outs Intent_ScalarValue:$result);
}

def Intent_DimOp : Intent_Op<"dim", [Pure]> {
  let arguments = (ins Intent_Value:$source, I64Attr:$axis,
                       I64Attr:$dimension);
  let results = (outs Index:$result);
}

def Intent_DomainOp : Intent_Op<"domain", [Pure]> {
  let arguments = (ins Variadic<Intent_Value>:$bounds,
                       ArrayAttr:$extent_dimensions);
  let results = (outs Intent_DomainValue:$result);
}
def Intent_DomainProductOp : Intent_Op<"domain_product", [Pure]> {
  let arguments = (ins Variadic<Intent_DomainValue>:$domains,
                       ArrayAttr:$extent_dimensions);
  let results = (outs Intent_DomainValue:$result);
}
def Intent_SubregionOp : Intent_Op<"subregion", [Pure]> {
  let arguments = (ins Variadic<Intent_Value>:$inputs,
                       BoolAttr:$has_start, BoolAttr:$has_stop,
                       ArrayAttr:$extent_dimensions);
  let results = (outs Intent_RegionValue:$result);
}
def Intent_IndicesOp : Intent_Op<"indices", [Pure]> {
  let arguments = (ins Intent_Value:$source, OptionalAttr<I64Attr>:$tensor_axis);
  let results = (outs AnyRankedTensor:$result);
}
def Intent_RegionEndOp : Intent_Op<"region_end", [Pure]> {
  let arguments = (ins Intent_IterationValue:$source);
  let results = (outs Index:$result);
}
def Intent_AssumeInBoundsOp : Intent_Op<"assume_in_bounds"> {
  let arguments = (ins Intent_Value:$index, Intent_AddressableValue:$view,
                       I64Attr:$axis);
  let results = (outs);
}

def Intent_ParallelOp : Intent_Op<"parallel", [RecursiveMemoryEffects]> {
  let arguments = (ins Intent_IterationValue:$source);
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $source `:` type($source) $body attr-dict
  }];
}
def Intent_IfOp : Intent_Op<"if", [RecursiveMemoryEffects,
    DeclareOpInterfaceMethods<RegionBranchOpInterface>]> {
  let arguments = (ins I1:$condition);
  let results = (outs Variadic<Intent_Value>:$results);
  let regions = (region SizedRegion<1>:$then_region,
                         SizedRegion<1>:$else_region);
  let assemblyFormat = [{
    $condition `then` $then_region `else` $else_region
    `->` `(` type($results) `)` attr-dict
  }];
}
def Intent_ForOp : Intent_Op<"for", [RecursiveMemoryEffects,
    DeclareOpInterfaceMethods<RegionBranchOpInterface,
                              ["getEntrySuccessorOperands"]>]> {
  let arguments = (ins Intent_IterationValue:$source,
                       Variadic<Intent_Value>:$init_args);
  let results = (outs Variadic<Intent_Value>:$results);
  let regions = (region SizedRegion<1>:$body);
  let extraClassDeclaration = [{
    ::mlir::Block::BlockArgListType getInductionVars() {
      return getBody().front().getArguments().drop_back(getInitArgs().size());
    }
    ::mlir::Block::BlockArgListType getRegionIterArgs() {
      return getBody().front().getArguments().take_back(getInitArgs().size());
    }
  }];
  let assemblyFormat = [{
    $source `:` type($source)
    `init` `(` ($init_args^ `:` type($init_args))? `)`
    $body `->` `(` type($results) `)` attr-dict
  }];
}
def Intent_WhileOp : Intent_Op<"while", [RecursiveMemoryEffects,
    DeclareOpInterfaceMethods<RegionBranchOpInterface,
                              ["getEntrySuccessorOperands"]>]> {
  let arguments = (ins Variadic<Intent_Value>:$init_args);
  let results = (outs Variadic<Intent_Value>:$results);
  let regions = (region SizedRegion<1>:$before, SizedRegion<1>:$after);
  let extraClassDeclaration = [{
    ::mlir::Block::BlockArgListType getBeforeArguments() {
      return getBefore().front().getArguments();
    }
    ::mlir::Block::BlockArgListType getAfterArguments() {
      return getAfter().front().getArguments();
    }
  }];
  let assemblyFormat = [{
    `init` `(` ($init_args^ `:` type($init_args))? `)`
    `before` $before `after` $after
    `->` `(` type($results) `)` attr-dict
  }];
}
def Intent_YieldOp : Intent_Op<"yield", [Terminator, ReturnLike]> {
  let arguments = (ins Variadic<Intent_Value>:$inputs);
  let results = (outs);
  let assemblyFormat = [{
    ($inputs^ `:` type($inputs))? attr-dict
  }];
}
def Intent_ConditionOp : Intent_Op<"condition", [Terminator,
    HasParent<"WhileOp">,
    DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface>]> {
  let arguments = (ins I1:$condition, Variadic<Intent_Value>:$args);
  let results = (outs);
  let assemblyFormat = [{
    $condition `forward` `(` ($args^ `:` type($args))? `)` attr-dict
  }];
}

def Intent_ViewLoadOp : Intent_Op<"view_load", [MemoryEffects<[MemRead]>,
    AttrSizedOperandSegments, DeclareOpInterfaceMethods<Intent_IndexedAccessOpInterface,
      ["getAccessValidity", "getAccessFill"]>]> {
  let arguments = (ins Intent_Value:$source, Variadic<Intent_Value>:$indices,
                       Optional<Intent_DataValue>:$valid, Optional<Intent_DataValue>:$fill,
                       Intent_IndexRelation:$index);
  let results = (outs Intent_DataValue:$result);
}
def Intent_ViewStoreOp : Intent_Op<"view_store", [MemoryEffects<[MemWrite]>,
    DeclareOpInterfaceMethods<Intent_IndexedAccessOpInterface, ["getStoredValue"]>]> {
  let arguments = (ins Intent_Value:$source, Variadic<Intent_Value>:$indices,
                       Intent_DataValue:$value, Intent_IndexRelation:$index);
  let results = (outs);
}
def Intent_ReshapeOp : Intent_Op<"reshape", [Pure]> {
  let arguments = (ins Variadic<Intent_Value>:$inputs, Intent_ShapeRelation:$shape);
  let results = (outs AnyRankedTensor:$result);
}
def Intent_JoinOp : Intent_BinaryDataOp<"join">;
def Intent_TransposeOp : Intent_UnaryDataOp<"transpose"> {
  let arguments = (ins Intent_DataValue:$input, ArrayAttr:$permutation);
}
def Intent_BroadcastOp : Intent_Op<"broadcast", [Pure]> {
  let arguments = (ins Variadic<Intent_Value>:$inputs, Intent_ShapeRelation:$shape);
  let results = (outs AnyRankedTensor:$result);
}
def Intent_FullOp : Intent_Op<"full", [Pure]> {
  let arguments = (ins Variadic<Intent_Value>:$inputs, Intent_ShapeRelation:$shape);
  let results = (outs AnyRankedTensor:$result);
}
def Intent_MakeTupleOp : Intent_Op<"make_tuple", [Pure]> {
  let arguments = (ins Variadic<Intent_Value>:$components);
  let results = (outs Intent_TupleValue:$result);
}
def Intent_MakeRecordOp : Intent_Op<"make_record", [Pure]> {
  let arguments = (ins Variadic<Intent_Value>:$fields);
  let results = (outs Intent_RecordValue:$result);
}
def Intent_ExtractOp : Intent_Op<"extract", [Pure]> {
  let arguments = (ins AnyTypeOf<[Intent_TupleValue, Intent_RecordValue]>:$product,
                       I64Attr:$field);
  let results = (outs Intent_Value:$result);
}

def Intent_UnaryOp : Intent_UnaryDataOp<"unary"> {
  let arguments = (ins Intent_DataValue:$input,
                       Intent_UnaryOperatorAttr:$operator_kind,
                       DefaultValuedAttr<BoolAttr, "false">:$approximate,
                       DefaultValuedAttr<BoolAttr, "false">:$flush_to_zero);
}
def Intent_BinaryOp : Intent_BinaryDataOp<"binary"> {
  let arguments = (ins Intent_DataValue:$lhs, Intent_DataValue:$rhs,
                       Intent_BinaryOperatorAttr:$operator_kind,
                       DefaultValuedAttr<BoolAttr, "false">:$approximate,
                       DefaultValuedAttr<BoolAttr, "false">:$flush_to_zero);
}
def Intent_CompareOp : Intent_Op<"compare", [Pure]> {
  let arguments = (ins Intent_DataValue:$lhs, Intent_DataValue:$rhs,
                       Intent_ComparePredicateAttr:$predicate);
  let results = (outs Intent_DataValue:$result);
}
def Intent_SelectOp : Intent_Op<"select", [Pure]> {
  let arguments = (ins Intent_DataValue:$condition,
                       Intent_DataValue:$true_value,
                       Intent_DataValue:$false_value);
  let results = (outs Intent_DataValue:$result);
}
def Intent_CastOp : Intent_UnaryDataOp<"cast"> {
  let arguments = (ins Intent_DataValue:$input,
                       OptionalAttr<I64Attr>:$rounding);
}
def Intent_BitcastOp : Intent_UnaryDataOp<"bitcast">;
def Intent_MaskOp : Intent_Op<"mask", [Pure]> {
  let arguments = (ins Intent_DataValue:$value,
                       Intent_DataValue:$predicate,
                       Intent_DataValue:$fill);
  let results = (outs Intent_DataValue:$result);
}

def Intent_ReduceOp : Intent_Op<"reduce", [Pure, IsolatedFromAbove, AttrSizedOperandSegments, Intent_StructuredOpInterface]> {
  let arguments = (ins Variadic<Intent_Value>:$sources,
                       Variadic<Intent_Value>:$identities,
                       Variadic<Intent_Value>:$captures,
                       ArrayAttr:$axes);
  let results = (outs Variadic<Intent_Value>:$results);
  let regions = (region SizedRegion<1>:$combine);
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
      ::llvm::SmallVector<int64_t> axes;
      for (auto axis : getAxes())
        axes.push_back(::mlir::cast<::mlir::IntegerAttr>(axis).getInt());
      return axes;
    }
  }];
}

def Intent_ScanOp : Intent_Op<"scan", [Pure, IsolatedFromAbove, AttrSizedOperandSegments, Intent_StructuredOpInterface]> {
  let arguments = (ins Variadic<Intent_Value>:$sources,
                       Variadic<Intent_Value>:$identities,
                       Variadic<Intent_Value>:$captures,
                       I64Attr:$axis,
                       BoolAttr:$inclusive,
                       BoolAttr:$reverse);
  let results = (outs Variadic<Intent_Value>:$results);
  let regions = (region SizedRegion<1>:$combine);
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

def Intent_RegionFoldOp : Intent_Op<"region_fold", [Pure, IsolatedFromAbove, AttrSizedOperandSegments, Intent_StructuredOpInterface]> {
  let arguments = (ins Variadic<Intent_Value>:$sources,
                       Variadic<Intent_Value>:$identities,
                       Variadic<Intent_Value>:$captures,
                       I64Attr:$axis);
  let results = (outs Variadic<Intent_Value>:$results);
  let regions = (region SizedRegion<1>:$summarize,
                         SizedRegion<1>:$combine);
  let assemblyFormat = [{
    `sources` `(` $sources `:` type($sources) `)`
    `identities` `(` $identities `:` type($identities) `)`
    `captures` `(` ($captures^ `:` type($captures))? `)`
    `axis` $axis
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

def Intent_RegionScanOp : Intent_Op<"region_scan", [Pure, IsolatedFromAbove, AttrSizedOperandSegments, Intent_StructuredOpInterface, AttrSizedResultSegments]> {
  let arguments = (ins Variadic<Intent_Value>:$sources,
                       Variadic<Intent_Value>:$identities,
                       Variadic<Intent_Value>:$initial_states,
                       Variadic<Intent_Value>:$captures,
                       I64Attr:$axis);
  let results = (outs Variadic<Intent_Value>:$emitted_results,
                      Variadic<Intent_Value>:$final_states);
  let regions = (region SizedRegion<1>:$summarize,
                         SizedRegion<1>:$combine,
                         SizedRegion<1>:$apply,
                         SizedRegion<1>:$emit);
  let assemblyFormat = [{
    `sources` `(` $sources `:` type($sources) `)`
    `identities` `(` $identities `:` type($identities) `)`
    `initial_states` `(` $initial_states `:` type($initial_states) `)`
    `captures` `(` ($captures^ `:` type($captures))? `)`
    `axis` $axis
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

def Intent_ContractOp : Intent_Op<"contract", [Pure]> {
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs,
                       ArrayAttr:$reduce, ArrayAttr:$batch);
  let results = (outs AnyRankedTensor:$result);
}
def Intent_QuantizeOp : Intent_Op<"quantize", [Pure]> {
  let arguments = (ins AnyRankedTensor:$input, Intent_QuantFormatAttr:$format);
  let results = (outs AnyRankedTensor:$result);
}
def Intent_QuantizedDotOp : Intent_Op<"quantized_dot", [Pure]> {
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs,
                       Intent_QuantFormatAttr:$lhs_format,
                       Intent_QuantFormatAttr:$rhs_format);
  let results = (outs AnyRankedTensor:$result);
}
def Intent_ScaledContractOp : Intent_Op<"scaled_contract", [Pure]> {
  let arguments = (ins AnyRankedTensor:$lhs,
                       AnyRankedTensor:$lhs_scale,
                       AnyRankedTensor:$rhs,
                       AnyRankedTensor:$rhs_scale,
                       ArrayAttr:$reduce,
                       ArrayAttr:$batch,
                       Intent_ScaledFormatAttr:$lhs_format,
                       Intent_ScaledFormatAttr:$rhs_format,
                       I64Attr:$lhs_group_size,
                       I64Attr:$rhs_group_size);
  let results = (outs AnyRankedTensor:$result);
}
def Intent_SparseContractOp : Intent_Op<"sparse_contract", [Pure]> {
  let arguments = (ins AnyRankedTensor:$compressed,
                       Intent_Value:$metadata,
                       AnyRankedTensor:$rhs,
                       Index:$logical_extent,
                       Intent_SparseFormat:$format,
                       ArrayAttr:$reduce,
                       ArrayAttr:$batch);
  let results = (outs AnyRankedTensor:$result);
}
def Intent_HistogramOp : Intent_Op<"histogram", [Pure]> {
  let arguments = (ins AnyRankedTensor:$values,
                       Index:$bins,
                       AnyRankedTensor:$valid,
                       I64Attr:$result_dimension);
  let results = (outs AnyRankedTensor:$result);
}

def Intent_GatherOp : Intent_Op<"gather", [Pure, AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<Intent_IndexedAccessOpInterface,
      ["getAccessValidity", "getAccessFill"]>]> {
  let arguments = (ins Intent_Value:$source, Variadic<Intent_Value>:$indices,
                       Optional<Intent_DataValue>:$valid, Optional<Intent_DataValue>:$fill,
                       Intent_IndexRelation:$index);
  let results = (outs Intent_DataValue:$result);
}
def Intent_ScatterUniqueOp : Intent_Op<"scatter_unique", [MemoryEffects<[MemWrite]>,
    DeclareOpInterfaceMethods<Intent_IndexedAccessOpInterface, ["getStoredValue"]>]> {
  let arguments = (ins Intent_Value:$source, Variadic<Intent_Value>:$indices,
                       Intent_DataValue:$value, Intent_IndexRelation:$index);
  let results = (outs);
}
def Intent_ScatterReduceOp : Intent_Op<"scatter_reduce", [MemoryEffects<[MemWrite]>,
    DeclareOpInterfaceMethods<Intent_IndexedAccessOpInterface, ["getStoredValue"]>]> {
  let arguments = (ins Intent_Value:$source, Variadic<Intent_Value>:$indices,
                       Intent_DataValue:$value, Intent_IndexRelation:$index);
  let results = (outs);
  let regions = (region SizedRegion<1>:$combine);
}

def Intent_BufferOp : Intent_Op<"buffer", [AttrSizedOperandSegments]> {
  let arguments = (ins Variadic<Intent_Value>:$extents, Optional<Intent_DataValue>:$initial,
                       Intent_ShapeRelation:$shape);
  let results = (outs Intent_BufferValue:$result);
}
def Intent_BufferLoadOp : Intent_Op<"buffer_load", [MemoryEffects<[MemRead]>,
    Intent_IndexedAccessOpInterface]> {
  let arguments = (ins Intent_Value:$source, Variadic<Intent_Value>:$indices,
                       Intent_IndexRelation:$index);
  let results = (outs Intent_DataValue:$result);
}
def Intent_BufferStoreOp : Intent_Op<"buffer_store", [MemoryEffects<[MemWrite]>,
    DeclareOpInterfaceMethods<Intent_IndexedAccessOpInterface, ["getStoredValue"]>]> {
  let arguments = (ins Intent_Value:$source, Variadic<Intent_Value>:$indices,
                       Intent_DataValue:$value, Intent_IndexRelation:$index);
  let results = (outs);
}
// Atomic observations and synchronization cannot be folded as ordinary reads.
def Intent_AtomicLoadOp : Intent_Op<"atomic_load", [MemoryEffects<[MemRead, MemWrite]>,
    Intent_IndexedAccessOpInterface]> {
  let arguments = (ins Intent_Value:$source, Variadic<Intent_Value>:$indices,
                       Intent_IndexRelation:$index,
                       Intent_AtomicOrderingAttr:$ordering);
  let results = (outs Intent_DataValue:$result);
}
def Intent_AtomicStoreOp : Intent_Op<"atomic_store", [MemoryEffects<[MemRead, MemWrite]>,
    DeclareOpInterfaceMethods<Intent_IndexedAccessOpInterface, ["getStoredValue"]>]> {
  let arguments = (ins Intent_Value:$source, Variadic<Intent_Value>:$indices,
                       Intent_DataValue:$value, Intent_IndexRelation:$index,
                       Intent_AtomicOrderingAttr:$ordering);
  let results = (outs);
}
def Intent_AtomicRMWOp : Intent_Op<"atomic_rmw", [MemoryEffects<[MemRead, MemWrite]>,
    DeclareOpInterfaceMethods<Intent_IndexedAccessOpInterface, ["getStoredValue"]>]> {
  let arguments = (ins Intent_Value:$source, Variadic<Intent_Value>:$indices,
                       Intent_DataValue:$value, Intent_IndexRelation:$index,
                       Intent_AtomicOrderingAttr:$ordering,
                       Intent_AtomicRMWKindAttr:$kind);
  let results = (outs Intent_DataValue:$result);
}
def Intent_AtomicCompareExchangeOp : Intent_Op<"atomic_compare_exchange",
    [MemoryEffects<[MemRead, MemWrite]>,
     DeclareOpInterfaceMethods<Intent_IndexedAccessOpInterface,
       ["getCompareValue", "getReplacementValue"]>]> {
  let arguments = (ins Intent_Value:$source, Variadic<Intent_Value>:$indices,
                       Intent_DataValue:$expected, Intent_DataValue:$desired,
                       Intent_IndexRelation:$index,
                       Intent_AtomicOrderingAttr:$ordering);
  let results = (outs Intent_RecordValue:$result);
}
def Intent_RandomBitsOp : Intent_Op<"random_bits", [Pure]> {
  let arguments = (ins UI64:$seed, Intent_Value:$logical_counter);
  let results = (outs Intent_DataValue:$result);
}

def Intent_ReturnOp : Intent_Op<"return", [Terminator, ReturnLike]> {
  let arguments = (ins Variadic<Intent_Value>:$inputs);
  let results = (outs);
}

#endif
