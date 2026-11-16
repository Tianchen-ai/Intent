#ifndef INTENT_DIALECT_Block_IR_BlockTYPES_TD
#define INTENT_DIALECT_Block_IR_BlockTYPES_TD

include "Intent/Dialect/Block/IR/BlockDialect.td"
include "Intent/Dialect/Block/IR/BlockAttrs.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonTypeConstraints.td"

def IntentBlock_ViewType : TypeDef<IntentBlock_Dialect, "View"> {
  let mnemonic = "view";
  let summary = "One physical external-view ABI resource";
  let parameters = (ins
    "::mlir::Type":$elementType,
    "uint32_t":$access,
    "uint64_t":$sourceId,
    "::intent::block::ViewLayoutAttr":$layout
  );
  let assemblyFormat =
      "`<` $elementType `,` $access `,` $sourceId `,` $layout `>`";
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    uint32_t getRank() const { return getLayout().getExtents().size(); }
  }];
}

def IntentBlock_FragmentType : TypeDef<IntentBlock_Dialect, "Fragment"> {
  let mnemonic = "fragment";
  let summary = "A parameterized value fragment owned by one physical program";
  let parameters = (ins
    "::mlir::Type":$elementType,
    "::mlir::ArrayAttr":$shape,
    "::mlir::ArrayAttr":$axisMaps,
    "uint32_t":$validity,
    "uint64_t":$owner
  );
  let assemblyFormat =
      "`<` $elementType `,` $shape `,` $axisMaps `,` $validity `,` $owner `>`";
  let genVerifyDecl = 1;
}

def IntentBlock_RangeType : TypeDef<IntentBlock_Dialect, "Range"> {
  let mnemonic = "range";
  let summary = "A runtime physical traversal range with logical provenance";
  let parameters = (ins
    "uint64_t":$sourceId,
    "uint32_t":$sourceAxis,
    "int64_t":$dimensionId,
    "bool":$derived
  );
  let assemblyFormat =
      "`<` $sourceId `,` $sourceAxis `,` $dimensionId `,` $derived `>`";
  let genVerifyDecl = 1;
}

def IntentBlock_BufferType : TypeDef<IntentBlock_Dialect, "Buffer"> {
  let mnemonic = "buffer";
  let summary = "A physical mutable resource with explicit scope and ownership";
  let parameters = (ins
    "::mlir::Type":$elementType,
    "::mlir::ArrayAttr":$shape,
    "::intent::block::BufferScopeAttr":$scope,
    "uint64_t":$instance,
    "uint64_t":$owner,
    "::intent::block::BufferInitializationAttr":$initialization,
    "uint32_t":$visibility
  );
  let assemblyFormat =
      "`<` $elementType `,` $shape `,` $scope `,` $instance `,` $owner `,` $initialization `,` $visibility `>`";
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    bool isInvocationWorkspace() const {
      return getScope().getValue() == BufferScope::InvocationWorkspace;
    }
  }];
}

def IntentBlock_RecordType : TypeDef<IntentBlock_Dialect, "Record"> {
  let mnemonic = "record";
  let summary = "A physical product value with stable named components";
  let parameters = (ins
    "::mlir::ArrayAttr":$fieldNames,
    "::mlir::ArrayAttr":$fieldTypes,
    "uint64_t":$owner
  );
  let assemblyFormat = "`<` $fieldNames `,` $fieldTypes `,` $owner `>`";
  let genVerifyDecl = 1;
}

def IntentBlock_ViewValue : DialectType<IntentBlock_Dialect,
    CPred<"::llvm::isa<::intent::block::ViewType>($_self)">,
    "physical external view", "::intent::block::ViewType">;
def IntentBlock_FragmentValue : DialectType<IntentBlock_Dialect,
    CPred<"::llvm::isa<::intent::block::FragmentType>($_self)">,
    "physical fragment", "::intent::block::FragmentType">;
def IntentBlock_RangeValue : DialectType<IntentBlock_Dialect,
    CPred<"::llvm::isa<::intent::block::RangeType>($_self)">,
    "physical range", "::intent::block::RangeType">;
def IntentBlock_BufferValue : DialectType<IntentBlock_Dialect,
    CPred<"::llvm::isa<::intent::block::BufferType>($_self)">,
    "physical buffer", "::intent::block::BufferType">;
def IntentBlock_RecordValue : DialectType<IntentBlock_Dialect,
    CPred<"::llvm::isa<::intent::block::RecordType>($_self)">,
    "physical record", "::intent::block::RecordType">;

def IntentBlock_ScalarValue : AnyTypeOf<[AnyInteger, AnyFloat, Index]>;
def IntentBlock_DataValue : AnyTypeOf<[
  IntentBlock_ScalarValue, IntentBlock_FragmentValue, IntentBlock_RecordValue
]>;
def IntentBlock_ResourceValue : AnyTypeOf<[IntentBlock_ViewValue, IntentBlock_BufferValue]>;
def IntentBlock_Value : AnyTypeOf<[
  IntentBlock_DataValue, IntentBlock_ResourceValue, IntentBlock_RangeValue
]>;

#endif
