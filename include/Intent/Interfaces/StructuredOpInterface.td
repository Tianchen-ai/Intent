#ifndef INTENT_INTERFACES_STRUCTUREDOPINTERFACE_TD
#define INTENT_INTERFACES_STRUCTUREDOPINTERFACE_TD

include "mlir/IR/OpBase.td"

def Intent_StructuredOpInterface : OpInterface<"StructuredOpInterface"> {
  let cppNamespace = "::intent";
  let description = [{
    Intrinsic operand, region and result relations of SSA structured operations.
    These queries read the current operation, without selecting physical layout,
    traversal or an implementation. Schema ties do not imply equal values.
  }];
  let methods = [
    InterfaceMethod<"Structured semantics, including its ordering contract.",
      "::intent::StructuredOpKind", "getStructuredKind">,
    InterfaceMethod<"Source operands.", "::mlir::ValueRange", "getSources">,
    InterfaceMethod<"Identity operands defining the accumulator schema.",
      "::mlir::ValueRange", "getIdentities">,
    InterfaceMethod<"Explicit captured operands.", "::mlir::ValueRange", "getCaptures">,
    InterfaceMethod<"Initial state operands of a region scan.",
      "::mlir::ValueRange", "getInitialStates", (ins), [{}], [{ return {}; }]>,
    InterfaceMethod<"Source-aligned outputs assembled by a region scan.",
      "::mlir::ValueRange", "getEmittedResults", (ins), [{}], [{ return {}; }]>,
    InterfaceMethod<"Final state results of a region scan.",
      "::mlir::ValueRange", "getFinalStates", (ins), [{}], [{ return {}; }]>,
    InterfaceMethod<"The combine region.", "::mlir::Region &", "getCombine">,
    InterfaceMethod<"The summarizer region, when the operation segments its source.",
      "::mlir::Region *", "getSummarizeRegion", (ins), [{}], [{ return nullptr; }]>,
    InterfaceMethod<"The incoming-state application region of a region scan.",
      "::mlir::Region *", "getApplyRegion", (ins), [{}], [{ return nullptr; }]>,
    InterfaceMethod<"The source-slice emitter region of a region scan.",
      "::mlir::Region *", "getEmitRegion", (ins), [{}], [{ return nullptr; }]>,
    InterfaceMethod<"Logical or current physical iteration axes.",
      "::llvm::SmallVector<int64_t>", "getIterationAxes">
  ];
  let extraClassDeclaration = [{
    ::llvm::SmallVector<::mlir::Type> getCombineArgumentTypes();
    ::llvm::SmallVector<::mlir::Type> getSummarizeArgumentTypes(::mlir::TypeRange sliceTypes);
    ::llvm::SmallVector<::mlir::Type> getApplyArgumentTypes();
    ::llvm::SmallVector<::mlir::Type> getEmitArgumentTypes(::mlir::TypeRange sliceTypes);
    ::mlir::Block::BlockArgListType getCombineLhs();
    ::mlir::Block::BlockArgListType getCombineRhs();
    ::mlir::Block::BlockArgListType getCombineCaptures();
    ::mlir::Block::BlockArgListType getSummarizeSources();
    ::mlir::Block::BlockArgListType getSummarizeCaptures();
    ::mlir::Block::BlockArgListType getApplySummaries();
    ::mlir::Block::BlockArgListType getApplyStates();
    ::mlir::Block::BlockArgListType getEmitSources();
    ::mlir::Block::BlockArgListType getEmitStates();
    ::mlir::Block::BlockArgListType getEmitCaptures();
    ::mlir::ValueRange getCombineYields();
    ::mlir::ValueRange getSummarizeYields();
    ::mlir::ValueRange getApplyYields();
    ::mlir::ValueRange getEmitYields();
    // Input-to-formal relations are available as soon as this region's block
    // signature is built. They never inspect another region or any terminator.
    ::llvm::SmallVector<::intent::StructuredValueRelation>
    getRegionArgumentRelations(::mlir::Region &region);
    // Whole-operation relations additionally require all regions and yields.
    ::llvm::SmallVector<::intent::StructuredValueRelation> getValueRelations();
  }];
}

#endif
